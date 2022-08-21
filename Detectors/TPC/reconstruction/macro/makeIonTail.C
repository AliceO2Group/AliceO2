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

#include <memory>
#include <vector>
#include <string_view>

#include "Framework/Logger.h"
#include "TFile.h"
#include "TChain.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCBase/Utils.h"
#include "TPCReconstruction/IonTailCorrection.h"

using namespace o2::tpc;

void makeIonTail(std::string_view inputFile = "tpcdigits.root", std::string_view outputFile = "tpcdigits.itCorr.root")
{
  TChain* tree = o2::tpc::utils::buildChain(fmt::format("ls {}", inputFile), "o2sim", "o2sim");
  const Long64_t nEntries = tree->GetEntries();

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
      itCorr.filterDigitsDirect(*digits);
    }

    tOut.Fill();
  }

  fOut->Write();
  fOut->Close();
}
