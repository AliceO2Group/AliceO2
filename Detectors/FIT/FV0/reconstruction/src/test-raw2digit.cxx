// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <TSystem.h>
#include <TTree.h>
#include "DetectorsCommonDataFormats/NameConf.h"
#include "Framework/Logger.h"
#include "DataFormatsFV0/BCData.h"
#include "DataFormatsFV0/ChannelData.h"

#include <TFile.h>
#include <cstring>

using namespace o2::fv0;
int main()
{
  using Digit = o2::fv0::BCData;
  struct EventFV0_t {
    Digit mDigit;
    std::vector<o2::fv0::ChannelData> mVecChannelData;
    bool operator==(const EventFV0_t& other) const
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
  std::vector<EventFV0_t> vecTotalEvents, vecTotalEvents2;
  gSystem->Exec("$O2_ROOT/bin/o2-sim -n 10 -m FV0 -g pythia8pp");
  gSystem->Exec("$O2_ROOT/bin/o2-sim-digitizer-workflow -b");
  TFile flIn("fv0digits.root");
  std::unique_ptr<TTree> treeInput((TTree*)flIn.Get("o2sim"));
  std::vector<Digit> vecDigits;
  std::vector<Digit>* ptrVecDigits = &vecDigits;
  std::vector<ChannelData> vecChannelData;
  std::vector<ChannelData>* ptrVecChannelData = &vecChannelData;
  treeInput->SetBranchAddress("FV0DigitBC", &ptrVecDigits);
  treeInput->SetBranchAddress("FV0DigitCh", &ptrVecChannelData);
  std::cout << "Tree nEntries:" << treeInput->GetEntries() << std::endl;
  for (int iEvent = 0; iEvent < treeInput->GetEntries(); iEvent++) { //Iterating TFs in tree
    treeInput->GetEntry(iEvent);
    for (const auto& digit : (*ptrVecDigits)) { //Iterating over all digits in given TF
      auto itBegin = ptrVecChannelData->begin();
      std::advance(itBegin, digit.ref.getFirstEntry());
      auto itEnd = ptrVecChannelData->begin();
      std::advance(itEnd, digit.ref.getFirstEntry() + digit.ref.getEntries());
      //Event within given TF
      auto eventFV0 = EventFV0_t{digit, std::vector<ChannelData>{itBegin, itEnd}};
      vecTotalEvents.push_back(eventFV0);
    }
  }
  std::cout << "\n===================================\n";
  for (const auto& entry : vecTotalEvents) {
    entry.print();
  }
  std::cout << "\n===================================\n";

  std::cout << "\nTOTAL EVENTS: " << vecTotalEvents.size() << std::endl;
  std::cout << "Simulation completed!" << std::endl;
  gSystem->Exec("$O2_ROOT/bin/o2-fv0-digi2raw -v 1 --file-per-link --configKeyValues \"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0\"");
  gSystem->Exec("$O2_ROOT/bin/o2-raw-file-reader-workflow -b --input-conf FV0raw.cfg --configKeyValues \"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0\"|$O2_ROOT/bin/o2-fv0-flp-dpl-workflow -b");
  TFile flIn2("o2_fv0digits.root");
  std::unique_ptr<TTree> treeInput2((TTree*)flIn2.Get("o2sim"));
  std::cout << "Reconstruction completed!" << std::endl;

  treeInput2->SetBranchAddress("FV0DIGITSBC", &ptrVecDigits);
  treeInput2->SetBranchAddress("FV0DIGITSCH", &ptrVecChannelData);
  std::cout << "Tree nEntries: " << treeInput2->GetEntries() << std::endl;
  for (int iEvent = 0; iEvent < treeInput2->GetEntries(); iEvent++) { //Iterating TFs in tree
    treeInput2->GetEntry(iEvent);
    for (const auto& digit : (*ptrVecDigits)) { //Iterating over all digits in given TF
      auto itBegin = ptrVecChannelData->begin();
      std::advance(itBegin, digit.ref.getFirstEntry());
      auto itEnd = ptrVecChannelData->begin();
      std::advance(itEnd, digit.ref.getFirstEntry() + digit.ref.getEntries());
      //Event within given TF
      auto eventFV0 = EventFV0_t{digit, std::vector<ChannelData>{itBegin, itEnd}};
      vecTotalEvents2.push_back(eventFV0);
    }
  }
  std::cout << "\n===================================\n";
  for (const auto& entry : vecTotalEvents2) {
    entry.print();
  }
  std::cout << "\n===================================\n";
  std::cout << "\nTOTAL EVENTS: " << vecTotalEvents2.size() << std::endl;
  if (vecTotalEvents == vecTotalEvents2) {
    std::cout << "TEST IS OK!\n";
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
