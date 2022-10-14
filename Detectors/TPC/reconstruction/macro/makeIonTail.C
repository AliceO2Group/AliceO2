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
#include <numeric>
#include <array>
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
#include "TPCBase/CRU.h"
#include "CommonUtils/TreeStreamRedirector.h"

using namespace o2::tpc;
size_t digitsInSaturateion(std::vector<Digit>& digits, bool correctCharge = false, CalPad* pedestals = nullptr, o2::utils::TreeStreamRedirector* stream = nullptr);

void makeIonTail(std::string_view inputFile = "tpcdigits.root", float itMultFactor = 1.f, float kTime = 0.0515, float threshold = -100.f, bool suppressBelowThreshold = false, bool removeSaturation = false, bool enableDebugStream = false, bool roundTo12b = true, std::string_view itParamFile = "", std::string_view pedetalFile = "", const std::array<float, 4> noiseSigmas = {-3.f, -3.f, -3.f, -3.f})
{
  if (enableDebugStream) {
    o2::conf::ConfigurableParam::setValue<int>("DebugStreamerParam", "StreamLevel", o2::utils::StreamFlags::streamITCorr);
  }
  o2::conf::ConfigurableParam::setValue<float>("TPCITCorr", "ITMultFactor", itMultFactor);
  o2::conf::ConfigurableParam::setValue<float>("TPCITCorr", "kTime", kTime);
  o2::conf::ConfigurableParam::setValue("TPCITCorr.padITCorrFile", itParamFile.data());

  std::string suppressed = "";
  std::string thresholdString = fmt::format(">= {}ADC counts", threshold);
  if (suppressBelowThreshold) {
    suppressed = fmt::format(".{}ADC", threshold);
    if (!pedetalFile.empty() && noiseSigmas[0] > 0) {
      suppressed = fmt::format(".{:.1f}_{:.1f}_{:.1f}_{:.1f}sigma", noiseSigmas[0], noiseSigmas[1], noiseSigmas[2], noiseSigmas[3]);
      thresholdString = fmt::format(">= {:.1f}, {:.1f}, {:.1f}, {:.1f} sigma noise", noiseSigmas[0], noiseSigmas[1], noiseSigmas[2], noiseSigmas[3]);
    }
  }
  if (roundTo12b) {
    suppressed += ".12b";
  }

  auto outputFile = fmt::format("tpcdigits.itCorr.{:.2f}.{:.4f}{}{}.root", itMultFactor, kTime, suppressed, removeSaturation ? ".rs" : "");
  if (itParamFile.size()) {
    outputFile = fmt::format("tpcdigits.itCorr.{:.2f}.paramFile{}{}.root", itMultFactor, suppressed, removeSaturation ? ".rs" : "");
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

  // pedestals
  CalPad* pedestals = nullptr;
  CalPad* noise = nullptr;
  if (!pedetalFile.empty()) {
    auto calPads = utils::readCalPads(pedetalFile, "Pedestals,Noise");
    pedestals = calPads[0];
    noise = calPads[1];
  }
  // debug stream
  o2::utils::TreeStreamRedirector pcstream("DigitFilterDebug.root", "recreate");

  IonTailCorrection itCorr;

  size_t saturatedSignals = 0;

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

      // count digits in threshold
      const auto satSignalsSec = digitsInSaturateion(*digits, removeSaturation, pedestals, &pcstream);
      saturatedSignals += satSignalsSec;

      itCorr.filterDigitsDirect(*digits);

      // Threshold detection function
      auto isBelowThreshold = [threshold, noise, &noiseSigmas](const auto& digit) {
        auto thresholdMod = threshold;
        if (noise && (noiseSigmas[0] > 0)) {
          const CRU cru(digit.getCRU());
          const int part = cru.partition();
          const int stackID = (part < 2) ? 0 : part - 1;
          const auto sector = CRU(digit.getCRU()).sector();
          thresholdMod = noiseSigmas[stackID] * noise->getValue(cru.sector(), digit.getRow(), digit.getPad());
        }
        return digit.getChargeFloat() < thresholdMod;
      };

      if (suppressBelowThreshold) {
        digits->erase(std::remove_if(digits->begin(), digits->end(), isBelowThreshold),
                      digits->end());
        digitsAboveThreshold += digits->size();
      } else {
        digitsAboveThreshold += digits->size() - std::count_if(digits->begin(), digits->end(), isBelowThreshold);
      }

      fmt::print("Found {} saturated signals in sector {}\n", satSignalsSec, iSector);
    }

    tOut.Fill();
  }

  const float digitFraction = float(digitsAboveThreshold) / float(originalDigits) * 100;
  const float saturationFraction = float(saturatedSignals) / float(originalDigits) * 100;

  std::string addInfo = fmt::format("itMultFactor {}, kTime {} and saturation signal removal {}", itMultFactor, kTime, removeSaturation);
  if (itParamFile.size()) {
    addInfo = fmt::format("IT parameters loaded from {}", itParamFile);
  }
  LOGP(info, "Found {} / {}  = {:.2f}% digits with charge {} and {} / {} = {:.2f}% saturated signals in {} entries with {}",
       digitsAboveThreshold, originalDigits, digitFraction, thresholdString, saturatedSignals, originalDigits, saturationFraction, nEntries, addInfo);

  fOut->Write();
  fOut->Close();
}

size_t digitsInSaturateion(std::vector<Digit>& digits, bool correctCharge /*= false*/, CalPad* pedestals /*= nullptr*/, o2::utils::TreeStreamRedirector* stream /*= nullptr*/)
{
  int lastRow = -1;
  int lastPad = -1;
  int lastTime = -1;
  float lastCharge = 0;
  float refSat = 90;
  float upperSat = 30;
  float lowerSat = 40;
  float initSat = 3;
  size_t nAboveThreshold = 0;
  bool checkSaturation = false;
  size_t nSaturatedSignals = 0;
  const int nMean = 3;
  int posMean = 0;
  int nPosMean = 0;
  float satVals[nMean] = {0};
  bool isSaturated = false;

  IonTailCorrection::sortDigitsOneSectorPerPad(digits);

  for (auto& digit : digits) {
    const auto sector = CRU(digit.getCRU()).sector();
    auto row = digit.getRow();
    auto pad = digit.getPad();
    auto time = digit.getTimeStamp();
    float pedestal = 0;
    float threshold = 920;
    if (pedestals) {
      pedestal = pedestals->getValue(sector, row, pad);
      threshold = 1020.f - pedestal;
      // fmt::print("threshold: {}\n", threshold);
    }

    // reset charge cumulation if pad has changed
    if (row != lastRow || pad != lastPad) {
      lastTime = time;
      nAboveThreshold = 0;
      refSat = 90;
      upperSat = 30;
      lowerSat = 40;
      initSat = 5;
      nPosMean = 0;
      posMean = 0;
      isSaturated = false;
    }

    bool isInLimit = false;
    auto origCharge = digit.getChargeFloat();
    if (time == lastTime + 1) {
      // fmt::print("checkSaturation: origCharge = {}, refSat = {}, lastCharge = {}\n", origCharge, refSat, lastCharge);
      if (origCharge > threshold) {
        checkSaturation = true;
        isInLimit = true;
      }
      if (checkSaturation && (origCharge < refSat + upperSat) && (origCharge > refSat - upperSat) && std::abs(origCharge - lastCharge) < initSat) {
        ++nAboveThreshold;
        if (nAboveThreshold > 2) {
          ++nSaturatedSignals;
          isSaturated = true;
          satVals[posMean++] = origCharge;
          posMean %= nMean;
          if (nPosMean < nMean) {
            ++nPosMean;
          }
          refSat = std::accumulate(satVals, satVals + nPosMean, 0.f) / float(nPosMean);
          upperSat = 5;
          lowerSat = 5;
          initSat = lowerSat;
        }
      }
    } else {
      isSaturated = false;
      checkSaturation = false;
      nAboveThreshold = 0;
      refSat = 90;
      upperSat = 30;
      lowerSat = 40;
      initSat = 5;
      nPosMean = 0;
      posMean = 0;
    }
    if (isSaturated || isInLimit) {
      if (stream) {
        int sec = sector;
        (*stream) << "satSig"
                  << "sector=" << sec
                  << "row=" << row
                  << "pad=" << pad
                  << "time=" << time
                  << "charge=" << origCharge
                  << "lastCharge=" << lastCharge
                  << "refSat=" << refSat
                  << "pedestal=" << pedestal
                  << "isInLimit=" << isInLimit
                  << "\n";
      }
    }
    if (correctCharge && isSaturated) {
      digit.setCharge(origCharge - refSat);
      // digit.setCharge(-origCharge - 10);
    }
    lastRow = row;
    lastPad = pad;
    lastCharge = origCharge;
    lastTime = time;
  }
  return nSaturatedSignals;
}
