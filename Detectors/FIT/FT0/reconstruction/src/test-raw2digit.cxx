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
#include "DataFormatsFT0/Digit.h"
#include <TFile.h>
#include <cstring>

using namespace o2::ft0;
int main()
{
  struct EventFT0_t {
    o2::ft0::Digit mDigit;
    std::vector<o2::ft0::ChannelData> mVecChannelData;
    bool operator==(const EventFT0_t& other) const
    {
      return mDigit == other.mDigit && mVecChannelData == other.mVecChannelData;
    }
    void print() const
    {
      std::cout << std::endl;
      std::cout << "------DIGITS--------\n";
      std::cout << "ref: " << mDigit.ref.getFirstEntry() << "|" << mDigit.ref.getEntries();
      std::cout << "\nbc-orbit: " << mDigit.mIntRecord.bc << "|" << mDigit.mIntRecord.orbit;
      //std::cout<<"\nmEventStatus: "<< static_cast<uint16_t>(mDigit.mEventStatus);//Excluded, init problem
      std::cout << "\nTriggers\n";
      std::cout << "triggersignals: " << static_cast<uint16_t>(mDigit.mTriggers.triggersignals);
      std::cout << "\nnChanA: " << static_cast<uint16_t>(mDigit.mTriggers.nChanA);
      std::cout << "\nnChanC: " << static_cast<uint16_t>(mDigit.mTriggers.nChanC);
      std::cout << "\namplA: " << mDigit.mTriggers.amplA << std::endl;
      std::cout << "amplC: " << mDigit.mTriggers.amplC << std::endl;
      std::cout << "timeA: " << mDigit.mTriggers.timeA << std::endl;
      std::cout << "timeC: " << mDigit.mTriggers.timeC << std::endl;
      std::cout << "------CHANNEL DATA--------";
      for (const auto& entry : mVecChannelData) {
        std::cout << "\nChId: " << static_cast<uint16_t>(entry.ChId);
        //std::cout<<"\nChainQTC: "<< static_cast<uint16_t>(entry.ChainQTC);//Excluded, init problem
        std::cout << "\nCFDTime: " << entry.CFDTime << std::endl;
        std::cout << "QTCAmpl: " << entry.QTCAmpl << std::endl;
        std::cout << "-------------------------";
      }
    }
  };
  std::vector<EventFT0_t> vecTotalEvents, vecTotalEvents2;
  gSystem->Exec("$O2_ROOT/bin/o2-sim -n 10 -m FT0 -g pythia8");
  gSystem->Exec("$O2_ROOT/bin/o2-sim-digitizer-workflow -b");
  TFile flIn("ft0digits.root");
  std::unique_ptr<TTree> treeInput((TTree*)flIn.Get("o2sim"));
  std::vector<Digit> vecDigits;
  std::vector<Digit>* ptrVecDigits = &vecDigits;
  std::vector<ChannelData> vecChannelData;
  std::vector<ChannelData>* ptrVecChannelData = &vecChannelData;
  treeInput->SetBranchAddress("FT0DIGITSBC", &ptrVecDigits);
  treeInput->SetBranchAddress("FT0DIGITSCH", &ptrVecChannelData);
  std::cout << "Tree nEntries:" << treeInput->GetEntries() << std::endl;
  for (int iEvent = 0; iEvent < treeInput->GetEntries(); iEvent++) { //Iterating TFs in tree
    treeInput->GetEntry(iEvent);
    for (const auto& digit : (*ptrVecDigits)) { //Iterating over all digits in given TF
      auto itBegin = ptrVecChannelData->begin();
      std::advance(itBegin, digit.ref.getFirstEntry());
      auto itEnd = ptrVecChannelData->begin();
      std::advance(itEnd, digit.ref.getFirstEntry() + digit.ref.getEntries());
      //Event within given TF
      auto eventFT0 = EventFT0_t{digit, std::vector<ChannelData>{itBegin, itEnd}};
      vecTotalEvents.push_back(eventFT0);
    }
  }
  std::cout << "\n===================================\n";
  for (auto const& entry : vecTotalEvents) {
    entry.print();
  }
  std::cout << "\n===================================\n";

  std::cout << "Simulation completed!" << std::endl;
  gSystem->Exec("$O2_ROOT/bin/o2-ft0-digi2raw --file-per-link");
  gSystem->Exec("$O2_ROOT/bin/o2-raw-file-reader-workflow -b --input-conf FT0raw.cfg|$O2_ROOT/bin/o2-ft0-flp-dpl-workflow -b");
  TFile flIn2("o2_ft0digits.root");
  std::unique_ptr<TTree> treeInput2((TTree*)flIn2.Get("o2sim"));
  std::cout << "Reconstruction completed!" << std::endl;

  treeInput2->SetBranchAddress("FT0DIGITSBC", &ptrVecDigits);
  treeInput2->SetBranchAddress("FT0DIGITSCH", &ptrVecChannelData);
  std::cout << "Tree nEntries: " << treeInput2->GetEntries() << std::endl;
  for (int iEvent = 0; iEvent < treeInput2->GetEntries(); iEvent++) { //Iterating TFs in tree
    treeInput2->GetEntry(iEvent);
    for (const auto& digit : (*ptrVecDigits)) { //Iterating over all digits in given TF
      auto itBegin = ptrVecChannelData->begin();
      std::advance(itBegin, digit.ref.getFirstEntry());
      auto itEnd = ptrVecChannelData->begin();
      std::advance(itEnd, digit.ref.getFirstEntry() + digit.ref.getEntries());
      //Event within given TF
      auto eventFT0 = EventFT0_t{digit, std::vector<ChannelData>{itBegin, itEnd}};
      vecTotalEvents2.push_back(eventFT0);
    }
  }
  std::cout << "\n===================================\n";
  for (auto const& entry : vecTotalEvents2) {
    entry.print();
  }
  std::cout << "\n===================================\n";
  if (vecTotalEvents == vecTotalEvents2) {
    std::cout << "TEST IS OK!\n";
  } else {
    std::cout << "ERROR!\n";
  }
  return 0;
}
