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
#include "CommonDataFormat/InteractionRecord.h"
#endif
// To produce CTP digits:
// o2-raw-tf-reader-workflow -b --onlyDet CTP,FT0  --input-data list.txt  | o2-ctp-reco-workflow -b  --use-verbose-mode
// To produce AOD:
// WORKFLOW_PARAMETERS="AOD" WORKFLOW_DETECTORS="CTP,FT0,FV0" IGNORE_EXISTING_SHMFILES=1 $O2_ROOT/prodtests/full-system-test/run-workflow-on-inputlist.sh TF list.txt
using namespace o2::ctp;
int CheckAOD2CTPDigits(bool files = 1)
{
  if (files == 0) {
    return 0;
  }
  int fRunNumber;
  ULong64_t fGlobalBC;
  ULong64_t fTriggerMask;
  ULong64_t fInputMask;
  std::unique_ptr<TFile> file(TFile::Open("AO2D.root"));
  if (file == nullptr) {
    std::cout << "Can not open file AO2D.root" << std::endl;
    return 1;
  }
  file->ls();
  TIter keyList(file->GetListOfKeys());
  TKey* key;
  TTree* tree;
  std::map<ULong64_t, ULong64_t> bc2classmask;
  std::map<ULong64_t, ULong64_t> bc2inputmask;
  // Find aod trigger info
  int i = 0;
  while ((key = (TKey*)keyList())) {
    std::cout << "loop:" << i << " " << key->GetName() << std::endl;
    i++;
    std::string name = key->GetName();
    if (name.find("metaData") != std::string::npos) {
      std::cout << "Skipping:" << name << std::endl;
      continue;
    }
    tree = (TTree*)file->Get(Form("%s/O2bc_001", key->GetName()));
    if (tree != 0) {
      std::cout << "found O2bc" << std::endl;
      tree->SetBranchAddress("fRunNumber", &fRunNumber);
      tree->SetBranchAddress("fGlobalBC", &fGlobalBC);
      tree->SetBranchAddress("fTriggerMask", &fTriggerMask);
      tree->SetBranchAddress("fInputMask", &fInputMask);
      std::cout << "# of entries:" << tree->GetEntries() << std::endl;
      int NN = tree->GetEntries();
      int Nloop = NN;
      for (int n{0}; n < Nloop; ++n) {
        tree->GetEvent(n);
        if (fTriggerMask) {
          bc2classmask[fGlobalBC] = fTriggerMask;
          // std::cout << std::dec <<  n << " Run:" << fRunNumber << " GBC:" << std::hex << fGlobalBC << " TM:0x" << std::hex << fTriggerMask << " count:" << bc2classmask.count(fGlobalBC) << std::endl;
        }
        if (fInputMask) {
          bc2inputmask[fGlobalBC] = fInputMask;
        }
        if (1) {
          if (fInputMask || fTriggerMask) {
            auto ir = o2::InteractionRecord::long2IR(fGlobalBC);
            // auto bcc = ir.orbit;
            // if(fTriggerMask) std::cout << "===>";
            // std::cout << std::hex << ir.orbit << " " << ir.bc << " " << fInputMask << " " << fTriggerMask << std::dec << " (" << ir.orbit << " " << ir.bc << ")"  << std::endl;
          }
        }
      }
    } else {
      std::cout << "return 1" << std::endl;
      return 1;
    }
  }
  // return 0;
  //  Read CTP digits and check if every class mask in digits is in AOD
  //  CTP digits
  TFile* fileDigits = TFile::Open("ctpdigitsWOD.root");
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
  std::cout << "Doing ctpdigits" << std::endl;
  std::map<ULong64_t, ULong64_t> bc2classmaskD;
  std::map<ULong64_t, ULong64_t> bc2inputmaskD;
  std::bitset<48> tvxmask;
  tvxmask.set(2);
  int nnotfinp = 0;
  int nokinp = 0;
  int nnotf = 0;
  int nok = 0;
  while (reader.Next()) {
    if (ctpdigs.GetSetupStatus() < 0) {
      std::cout << "Error:" << std::dec << ctpdigs.GetSetupStatus() << " for:" << ctpdigs.GetBranchName() << std::endl;
      return 1;
    }
    // std::cout << "size:" << std::dec << ctpdigs.GetSize() << std::endl;
    // std::cout << "size:" << std::dec << ctpdigs->size() << std::endl;
    for (auto const& dig : *ctpdigs) {
      ULong64_t gbc = dig.intRecord.toLong();
      if (dig.CTPClassMask.count()) {
        bc2classmaskD[gbc] = dig.CTPClassMask.to_ullong();
        // int del = 280+17;
        // auto it = bc2classmask.find (gbc);
        // if(it != bc2classmask.end()) {
        if (bc2classmask.count(gbc)) {
          // bc2classmask.erase(gbc);
          nok++;
          // std::cout << std::hex << gbc << " tc aod clsmask:" << bc2classmask[gbc] << " " << dig.CTPClassMask.to_ullong()  << std::endl;
          //  dig.printStream(std::cout);
        } else {
          std::cout << std::dec << dig.intRecord.orbit << " " << dig.intRecord.bc << " " << std::hex << dig.intRecord.toLong() << " not found " << bc2classmask.count(gbc) << std::endl;
          nnotf++;
        }
      }
      if (dig.CTPInputMask.count()) {
        bc2inputmaskD[gbc] = dig.CTPInputMask.to_ullong();
        // auto it = bc2inputmask.find (gbc);
        // if(it != bc2inputmask.end()) {
        if (bc2inputmask.count(gbc)) {
          // bc2inputmask.erase(gbc);
          nokinp++;
          // std::cout << std::hex << gbc << " ir aod inpmask:" << bc2inputmask[gbc] << " " << dig.CTPInputMask.to_ullong() << std::endl;
        } else {
          std::cout << std::dec << dig.intRecord.orbit << " " << dig.intRecord.bc << " " << std::hex << dig.intRecord.toLong() << " not found " << bc2inputmask.count(gbc) << std::endl;
          nnotfinp++;
        }
      }
    }
  }
  std::cout << "TClasses ===> nok:" << nok << " NOT found in digits:" << nnotf << " left:" << bc2classmask.size() << std::endl;
  std::cout << "Inputs   ===> nok:" << nokinp << " NOT found in digits:" << nnotfinp << " left:" << bc2inputmask.size() << std::endl;
  nok = 0;
  nokinp = 0;
  nnotf = 0;
  nnotfinp = 0;
  for (auto const& tm : bc2classmask) {
    if (bc2classmaskD.count(tm.first)) {
      nok++;
    } else {
      nnotf++;
    }
  }
  for (auto const& tm : bc2inputmask) {
    if (bc2inputmaskD.count(tm.first)) {
      nokinp++;
    } else {
      nnotfinp++;
    }
  }

  std::cout << "TClasses ===> nok:" << nok << " NOT found in aod:" << nnotf << " left:" << bc2classmaskD.size() << std::endl;
  std::cout << "Inputs   ===> nok:" << nokinp << " NOT found in aod:" << nnotfinp << " left:" << bc2inputmaskD.size() << std::endl;
  return 0;
}
