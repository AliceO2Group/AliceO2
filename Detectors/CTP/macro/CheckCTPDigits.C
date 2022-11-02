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
int CheckCTPDigits(bool files = 0)
{
  if (files == 0) {
    return 0;
  }
  // CTP digits
  TFile* fileDigits = TFile::Open("ctpdigits.root");
  //
  fileDigits->ls();
  o2::ctp::CTPDigit* dig = new o2::ctp::CTPDigit;
  //
  // tree->Print();
  TTreeReader reader("o2sim", fileDigits);
  // TTreeReaderValue<std::vector<o2::ctp::CTPDigit>> ctpdigs(reader,"CTPDigits");
  TTreeReaderArray<o2::ctp::CTPDigit> ctpdigs(reader, "CTPDigits");
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
    std::cout << "size:" << std::dec << ctpdigs.GetSize() << std::endl;
    int del = 280 + 14;
    int i, j, bc;
    for (i = del; i < ctpdigs.GetSize(); i++) {
      o2::ctp::CTPDigit* dig = &ctpdigs[i];
      if (dig->CTPClassMask.count() > 0) {
        std::cout << std::dec << "=======> BC tm:" << dig->intRecord.bc << " O:" << dig->intRecord.orbit << std::hex << " 0b" << dig->CTPClassMask << " gbc:" << dig->intRecord.toLong() << std::endl;
        int found = 0;
        ncls++;
        std::stringstream ss;
        for (j = 0; j < del; j++) {
          ss << std::dec << ctpdigs[i - j].intRecord.bc << " O:" << ctpdigs[i - j].intRecord.orbit << " " << std::hex << ctpdigs[i - j].CTPInputMask.to_ullong() << std::endl;
          bc = (dig->intRecord.bc - del) % 3564;
          if (bc < 0)
            bc += 3564;
          if (bc == ctpdigs[i - j].intRecord.bc) {
            std::cout << " found:" << std::dec << j << " 0b" << ctpdigs[i - j].CTPInputMask << std::endl;
            found = 1;
            auto istvx = ctpdigs[i - j].CTPInputMask & tvxmask;
            if (istvx.count() == 0)
              std::cout << "error: tcx missing " << std::endl;
            break;
          }
        }
        // std::cout << std::endl;
        if (0) {
          if (found == 0) {
            std::cout << " NOT FOUND:" << std::dec << bc;
            std::cout << std::dec << " =======> BC tm:" << dig->intRecord.bc << ""
                      << " O:" << dig->intRecord.orbit << std::hex << " 0b" << dig->CTPClassMask << std::endl;
            std::cout << ss.str() << std::endl;
            nnotf++;
          } else {
            std::cout << " FOUND:" << std::dec << bc << " " << ctpdigs[i - j].intRecord.bc;
            std::cout << std::dec << " =======> BC tm:" << dig->intRecord.bc << ""
                      << " O:" << dig->intRecord.orbit << std::hex << " 0b" << dig->CTPClassMask << std::endl;
          }
        }
      }
    }
  }
  std::cout << "# of cls masks:" << std::dec << ncls++ << " NOT found:" << nnotf << std::endl;
  return 0;
}
