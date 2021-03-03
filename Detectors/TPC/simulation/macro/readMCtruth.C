// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file readMCtruth.cxx
/// \brief This macro demonstrates how to extract the MC truth information from
/// the digits
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <vector>
#include "TFile.h"
#include "TTree.h"
#include "DataFormatsTPC/Digit.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <iostream>
#include "Framework/Logger.h"
#endif

void readMCtruth(std::string filename)
{
  TFile* digitFile = TFile::Open(filename.data());
  TTree* digitTree = (TTree*)digitFile->Get("o2sim");

  o2::dataformats::IOMCTruthContainerView* plabels[36] = {0};
  std::vector<o2::tpc::Digit>* digits[36] = {0};
  int nBranches = 0;
  bool mcPresent = false, perSector = false;
  if (digitTree->GetBranch("TPPCDigit")) {
    LOG(INFO) << "Joint digit branch is found";
    nBranches = 1;
    digitTree->SetBranchAddress("TPCDigit", &digits[0]);
    if (digitTree->GetBranch("TPCDigitMCTruth")) {
      mcPresent = true;
      digitTree->SetBranchAddress("TPCDigitMCTruth", &plabels[0]);
    }
  } else {
    nBranches = 36;
    perSector = true;
    for (int i = 0; i < 36; i++) {
      std::string digBName = fmt::format("TPCDigit_{:d}", i).c_str();
      if (digitTree->GetBranch(digBName.c_str())) {
        digitTree->SetBranchAddress(digBName.c_str(), &digits[i]);
        std::string digMCBName = fmt::format("TPCDigitMCTruth_{:d}", i).c_str();
        if (digitTree->GetBranch(digMCBName.c_str())) {
          mcPresent = true;
          digitTree->SetBranchAddress(digMCBName.c_str(), &plabels[i]);
        }
      }
    }
  }
  o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel> labels[36];

  for (int iEvent = 0; iEvent < digitTree->GetEntriesFast(); ++iEvent) {
    digitTree->GetEntry(iEvent);
    for (int ib = 0; ib < nBranches; ib++) {
      if (plabels[ib]) {
        plabels[ib]->copyandflatten(labels[ib]);
        delete plabels[ib];
        plabels[ib] = nullptr;
      }
    }

    for (int ib = 0; ib < nBranches; ib++) {
      if (!digits[ib]) {
        continue;
      }
      int nd = digits[ib]->size();
      for (int idig = 0; idig < nd; idig++) {
        const auto& digit = (*digits[ib])[idig];
        o2::MCCompLabel lab;
        if (mcPresent) {
          lab = labels[ib].getLabels(idig)[0];
        }
        std::cout << "Digit " << digit << " from " << lab << "\n";
      }
    }
  }
}
