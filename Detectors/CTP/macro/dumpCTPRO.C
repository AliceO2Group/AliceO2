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

/// \file checkCTPDigits.C
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
// Check if trigger class mask has corresponding input class mask
// Tp be generalised to use CTP config
using namespace o2::ctp;
int dumpCTPRO(bool files = 0)
{
  if (files == 0) {
    return 0;
  }
  // CTP digits
  // TFile* fileDigits = TFile::Open("/home/rl/tests/digits/ctpdigits.root");
  TFile* fileDigits = TFile::Open("/home/rl/pp/529690/ctpdigits.root");
  // TFile* fileDigits = TFile::Open("/home/rl/PbPb/529403/1820/ctpdigits.root");
  // TFile* fileDigits = TFile::Open("/home/rl/PbPb/529403/1820/ctpdigitsNOParams.root");
  //
  fileDigits->ls();
  o2::ctp::CTPDigit* dig = new o2::ctp::CTPDigit;
  //
  // tree->Print();
  TTreeReader reader("o2sim", fileDigits);
  TTreeReaderArray<o2::ctp::CTPDigit> ctpdigs(reader, "CTPDigits");
  bool firstE = true;
  //
  int ORBIT = 3564;
  int ninps = 0;
  int nClass = 0;
  uint32_t orbitmax = 0;
  uint32_t orbitmin = 0xffffffff;
  while (reader.Next()) {
    if (ctpdigs.GetSetupStatus() < 0) {
      std::cout << "Error:" << std::dec << ctpdigs.GetSetupStatus() << " for:" << ctpdigs.GetBranchName() << std::endl;
      return 1;
    }
    // std::cout << "size:" << std::dec << ctpdigs.GetSize() << std::endl;
    int i;
    for (i = 0; i < ctpdigs.GetSize(); i++) {
      o2::ctp::CTPDigit* dig = &ctpdigs[i];
      uint64_t inpMask = dig->CTPInputMask.to_ullong();
      uint64_t trgMask = dig->CTPClassMask.to_ullong();
      std::cout << dig->intRecord.orbit << " " << dig->intRecord.bc << " " << std::hex << inpMask << " " << trgMask << std::dec << std::endl;
    }
  }
  return 0;
}
