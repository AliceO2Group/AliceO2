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

/// \file checkAOD2CTPDigits.C
/// \brief create CTP config, test it and add to database
/// \author Roman Lietava

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <fairlogger/Logger.h>
#include "TFile.h"
#include "TTree.h"
#include <string>
#include <iostream>
#include <vector>
#include "TKey.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#endif
// To produce CTP digits:
// o2-raw-tf-reader-workflow -b --onlyDet CTP,FT0  --input-data list.txt  | o2-ctp-reco-workflow -b  --use-verbose-mode
// To produce AOD:
// WORKFLOW_PARAMETERS="AOD" WORKFLOW_DETECTORS="CTP,FT0,FV0" IGNORE_EXISTING_SHMFILES=1 $O2_ROOT/prodtests/full-system-test/run-workflow-on-inputlist.sh TF list.txt
using namespace o2::ctp;
int CheckAOD2CTPDigits(bool files = 0)
{
  if (files == 0) {
    return 0;
  }
  int fRunNumber;
  ULong64_t fGlobalBC;
  ULong64_t fTriggerMask;
  std::unique_ptr<TFile> file(TFile::Open("AO2D.root"));
  file->ls();
  TIter keyList(file->GetListOfKeys());
  TKey* key;
  TTree* tree;
  ;
  std::map<ULong64_t, ULong64_t> bc2classmask;
  // Find aod trigger info
  while ((key = (TKey*)keyList())) {
    std::cout << key->GetName() << std::endl;
    tree = (TTree*)file->Get(Form("%s/O2bc", key->GetName()));
    if (tree != 0) {
      std::cout << "found O2bc" << std::endl;
      tree->SetBranchAddress("fRunNumber", &fRunNumber);
      tree->SetBranchAddress("fGlobalBC", &fGlobalBC);
      tree->SetBranchAddress("fTriggerMask", &fTriggerMask);
      std::cout << "# of entries:" << tree->GetEntries() << std::endl;
      int NN = tree->GetEntries();
      int Nloop = NN;
      for (int n{0}; n < Nloop; ++n) {
        tree->GetEvent(n);
        if (fTriggerMask) {
          bc2classmask[fGlobalBC] = fTriggerMask;
          // std::cout << std::dec <<  n << " Run:" << fRunNumber << " GBC:" << std::hex << fGlobalBC << " TM:0x" << std::hex << fTriggerMask << " count:" << bc2classmask.count(fGlobalBC) << std::endl;
        }
      }
    } else {
      return 1;
    }
  }
  // Read CTP digits and check if every class mask in digits is in AOD
  // CTP digits
  TFile* fileDigits = TFile::Open("ctpdigits.root");
  //
  fileDigits->ls();
  o2::ctp::CTPDigit* dig = new o2::ctp::CTPDigit;
  //
  // tree->Print();
  TTreeReader reader("o2sim", fileDigits);
  TTreeReaderValue<std::vector<o2::ctp::CTPDigit>> ctpdigs(reader, "CTPDigits");
  // TTreeReaderArray<o2::ctp::CTPDigit> ctpdigs(reader,"CTPDigits");
  bool firstE = true;
  //
  std::bitset<48> tvxmask;
  tvxmask.set(2);
  int ncls = 0;
  int nnotf = 0;
  while (reader.Next()) {
    if (ctpdigs.GetSetupStatus() < 0) {
      std::cout << "Error:" << std::dec << ctpdigs.GetSetupStatus() << " for:" << ctpdigs.GetBranchName() << std::endl;
      return 1;
    }
    // std::cout << "size:" << std::dec << ctpdigs.GetSize() << std::endl;
    std::cout << "size:" << std::dec << ctpdigs->size() << std::endl;
    for (auto const& dig : *ctpdigs) {
      if (dig.CTPClassMask.count()) {
        ULong64_t gbc = dig.intRecord.toLong();
        // int del = 280+17;
        if (bc2classmask.count(gbc)) {
          // std::cout << std::hex << gbc << "aod clsmask:" << bc2classmask[gbc] << " " << dig.CTPClassMask.to_ullong() << " inps:" << dig.CTPInputMask.to_ullong() << std::endl;
          // dig.printStream(std::cout);
        } else {
          std::cout << std::dec << dig.intRecord.orbit << " " << dig.intRecord.bc << " " << std::hex << dig.intRecord.toLong() << " not found " << bc2classmask.count(gbc) << std::endl;
        }
      }
    }
  }
  return 0;
}
