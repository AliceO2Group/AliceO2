// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file makeIonTail.C
/// \brief add or correct for ion tail on TPC digits file
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include <algorithm>
#include <memory>
#include <vector>
#include <string_view>
#include <fmt/format.h>

#include "Framework/Logger.h"
#include "TFile.h"
#include "TChain.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/DebugStreamer.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCBase/Utils.h"
#include "TPCReconstruction/IonTailCorrection.h"
#include "TPCBase/CRUCalibHelpers.h"

using namespace o2::tpc;

void makeIonTail(std::string_view inputFile = "tpcdigits.root", float itMultFactor = 1.f, float kTime = 0.0515, float threshold = -100.f, bool suppressBelowThreshold = false, bool enableDebugStream = false, bool roundTo12b = true, std::string_view itParamFile = "")
{
  if (enableDebugStream) {
    o2::conf::ConfigurableParam::setValue<int>("DebugStreamerParam", "StreamLevel", o2::utils::StreamFlags::streamITCorr);
  }

  o2::conf::ConfigurableParam::setValue<float>("TPCITCorr", "ITMultFactor", itMultFactor);
  o2::conf::ConfigurableParam::setValue<float>("TPCITCorr", "kTime", kTime);
  o2::conf::ConfigurableParam::setValue("TPCITCorr.padITCorrFile", itParamFile.data());

  std::string suppressed = "";
  if (suppressBelowThreshold) {
    suppressed = fmt::format(".{}ADC", threshold);
  }
  if (roundTo12b) {
    suppressed += ".12b";
  }

  auto outputFile = fmt::format("tpcdigits.itCorr.{:.2f}.{:.4f}{}.root", itMultFactor, kTime, suppressed);
  if (itParamFile.size()) {
    outputFile = fmt::format("tpcdigits.itCorr.paramFile{}.root", suppressed);
  }

  TChain* tree = o2::tpc::utils::buildChain(fmt::format("ls {}", inputFile), "o2sim", "o2sim");
  const Long64_t nEntries = tree->GetEntries();

  size_t originalDigits = 0;
  size_t digitsAboveThreshold = 0;

  // Initialize File for later writing
  std::unique_ptr<TFile> fOut{TFile::Open(outputFile.data(), "RECREATE")};
  TTree tOut("o2sim", "o2sim");

  std::array<std::vector<o2::tpc::Digit>*, 36> digitizedSignal;
  for (size_t iSec = 0; iSec < digitizedSignal.size(); ++iSec) {
    digitizedSignal[iSec] = nullptr;
    tree->SetBranchAddress(Form("TPCDigit_%zu", iSec), &digitizedSignal[iSec]);
    tOut.Branch(Form("TPCDigit_%zu", iSec), &digitizedSignal[iSec]);
  }

  IonTailCorrection itCorr;

  for (Long64_t iEvent = 0; iEvent < nEntries; ++iEvent) {
    tree->GetEntry(iEvent);

    for (size_t iSector = 0; iSector < 36; ++iSector) {
      auto digits = digitizedSignal[iSector];
      if (roundTo12b) {
        for (auto& digit : *digits) {
          digit.setCharge(cru_calib_helpers::fixedSizeToFloat(cru_calib_helpers::floatToFixedSize(digit.getChargeFloat())));
        }
      }
      originalDigits += digits->size();
      itCorr.filterDigitsDirect(*digits);
      if (suppressBelowThreshold) {
        digits->erase(std::remove_if(digits->begin(), digits->end(),
                                     [threshold](const auto& digit) {
                                       return digit.getChargeFloat() < threshold;
                                     }),
                      digits->end());
        digitsAboveThreshold += digits->size();
      } else {
        digitsAboveThreshold += std::count_if(digits->begin(), digits->end(), [threshold](const auto& digit) { return digit.getChargeFloat() >= threshold; });
      }
    }

    tOut.Fill();
  }

  const float digitFraction = float(digitsAboveThreshold) / float(originalDigits);
  std::string addInfo = fmt::format("itMultFactor {} and kTime {}", itMultFactor, kTime);
  if (itParamFile.size()) {
    addInfo = fmt::format("IT parameters loaded from {}", itParamFile);
  }
  LOGP(info, "Found {} / {}  = {:.2f} digits with charge >= {} in {} entries with {}",
       digitsAboveThreshold, originalDigits, digitFraction, threshold, nEntries, addInfo);

  fOut->Write();
  fOut->Close();
}
