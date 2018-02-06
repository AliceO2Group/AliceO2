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
#include <vector>

void readMCtruth(std::string filename)
{
  TFile* digitFile = TFile::Open(filename.data());
  TTree* digitTree = (TTree*)digitFile->Get("o2sim");

  std::vector<o2::TPC::Digit>* digits = nullptr;
  digitTree->SetBranchAddress("TPCDigit", &digits);

  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mMCTruthArray;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mcTruthArray(&mMCTruthArray);
  digitTree->SetBranchAddress("TPCDigitMCTruth", &mcTruthArray);

  for (int iEvent = 0; iEvent < digitTree->GetEntriesFast(); ++iEvent) {
    int digit = 0;
    digitTree->GetEntry(iEvent);
    for (auto& inputdigit : *digits) {
      gsl::span<const o2::MCCompLabel> mcArray = mMCTruthArray.getLabels(digit);
      for (int j = 0; j < static_cast<int>(mcArray.size()); ++j) {
        std::cout << "Digit " << digit << " from Event "
                  << mMCTruthArray.getElement(mMCTruthArray.getMCTruthHeader(digit).index + j).getEventID()
                  << " with Track ID "
                  << mMCTruthArray.getElement(mMCTruthArray.getMCTruthHeader(digit).index + j).getTrackID() << "\n";
      }
      ++digit;
    }
  }
}
