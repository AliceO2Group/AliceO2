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

#include <TSystem.h>
#include <TTree.h>
#include "CommonUtils/NameConf.h"
#include "Framework/Logger.h"
#include "DataFormatsFDD/Digit.h"
#include <TFile.h>
#include <cstring>

using namespace o2::fdd;
int main()
{
  struct EventFDD_t {
    o2::fdd::Digit mDigit;
    std::vector<o2::fdd::ChannelData> mVecChannelData;
    bool operator==(const EventFDD_t& other) const
    {
      return mDigit == other.mDigit && mVecChannelData == other.mVecChannelData;
    }
    void print() const
    {
      mDigit.printLog();
      for (const auto& entry : mVecChannelData) {
        entry.printLog();
      }
    }
  };
  std::vector<EventFDD_t> vecTotalEvents, vecTotalEvents2;
  gSystem->Exec("$O2_ROOT/bin/o2-sim -n 10 -m FDD -g pythia8pp");
  gSystem->Exec("$O2_ROOT/bin/o2-sim-digitizer-workflow -b");
  TFile flIn("fdddigits.root");
  std::unique_ptr<TTree> treeInput((TTree*)flIn.Get("o2sim"));
  std::vector<Digit> vecDigits;
  std::vector<Digit>* ptrVecDigits = &vecDigits;
  std::vector<ChannelData> vecChannelData;
  std::vector<ChannelData>* ptrVecChannelData = &vecChannelData;
  treeInput->SetBranchAddress(Digit::sDigitBranchName, &ptrVecDigits);
  treeInput->SetBranchAddress(ChannelData::sDigitBranchName, &ptrVecChannelData);
  std::cout << "Tree nEntries:" << treeInput->GetEntries() << std::endl;
  for (int iEvent = 0; iEvent < treeInput->GetEntries(); iEvent++) { //Iterating TFs in tree
    treeInput->GetEntry(iEvent);
    for (const auto& digit : (*ptrVecDigits)) { //Iterating over all digits in given TF
      auto itBegin = ptrVecChannelData->begin();
      std::advance(itBegin, digit.ref.getFirstEntry());
      auto itEnd = ptrVecChannelData->begin();
      std::advance(itEnd, digit.ref.getFirstEntry() + digit.ref.getEntries());
      //Event within given TF
      auto eventFDD = EventFDD_t{digit, std::vector<ChannelData>{itBegin, itEnd}};
      vecTotalEvents.push_back(eventFDD);
    }
  }
  std::cout << "\n===================================\n";
  for (auto const& entry : vecTotalEvents) {
    entry.print();
  }
  std::cout << "\n===================================\n";

  std::cout << "\nTOTAL EVENTS: " << vecTotalEvents.size() << std::endl;
  std::cout << "Simulation completed!" << std::endl;
  gSystem->Exec("$O2_ROOT/bin/o2-fdd-digit2raw --file-for link --configKeyValues \"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0\"");
  gSystem->Exec("$O2_ROOT/bin/o2-raw-file-reader-workflow -b --input-conf FDDraw.cfg --configKeyValues \"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0\"|$O2_ROOT/bin/o2-fdd-flp-dpl-workflow -b");
  TFile flIn2("o2_fdddigits.root");
  std::unique_ptr<TTree> treeInput2((TTree*)flIn2.Get("o2sim"));
  std::cout << "Reconstruction completed!" << std::endl;

  treeInput2->SetBranchAddress(Digit::sDigitBranchName, &ptrVecDigits);
  treeInput2->SetBranchAddress(ChannelData::sDigitBranchName, &ptrVecChannelData);
  std::cout << "Tree nEntries: " << treeInput2->GetEntries() << std::endl;
  for (int iEvent = 0; iEvent < treeInput2->GetEntries(); iEvent++) { //Iterating TFs in tree
    treeInput2->GetEntry(iEvent);
    for (const auto& digit : (*ptrVecDigits)) { //Iterating over all digits in given TF
      auto itBegin = ptrVecChannelData->begin();
      std::advance(itBegin, digit.ref.getFirstEntry());
      auto itEnd = ptrVecChannelData->begin();
      std::advance(itEnd, digit.ref.getFirstEntry() + digit.ref.getEntries());
      //Event within given TF
      auto eventFDD = EventFDD_t{digit, std::vector<ChannelData>{itBegin, itEnd}};
      vecTotalEvents2.push_back(eventFDD);
    }
  }
  std::cout << "\n===================================\n";
  for (auto const& entry : vecTotalEvents2) {
    entry.print();
  }
  std::cout << "\n===================================\n";
  std::cout << "\nTOTAL EVENTS: " << vecTotalEvents2.size() << std::endl;
  if (vecTotalEvents == vecTotalEvents2) {
    std::cout << "\n TEST IS OK!\n";
  } else {
    std::cout << "\nDIFFERENCE BETWEEN SRC AND DEST\n";
    std::cout << "\n===============================\n";
    for (int iEntry = 0; iEntry < std::max(vecTotalEvents.size(), vecTotalEvents2.size()); iEntry++) {
      if (iEntry < vecTotalEvents.size() && iEntry < vecTotalEvents2.size()) {
        if (vecTotalEvents[iEntry] == vecTotalEvents2[iEntry]) {
          continue;
        }
      }
      std::cout << "\nEntryID: " << iEntry;
      std::cout << "\n------------------------------SOURCE------------------------------\n";
      if (iEntry < vecTotalEvents.size()) {
        vecTotalEvents[iEntry].print();
      } else {
        std::cout << "\nEMPTY!\n";
      }
      std::cout << "\n------------------------------DESTINATION------------------------------\n";
      if (iEntry < vecTotalEvents2.size()) {
        vecTotalEvents2[iEntry].print();
      } else {
        std::cout << "\nEMPTY!\n";
      }
    }
    std::cout << "\nERROR!\n";
  }
  return 0;
}
