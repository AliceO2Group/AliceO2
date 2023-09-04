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

/// \file TestCTPScalers.C
/// \brief create CTP scalers, test it and add to database
/// \author Roman Lietava
#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <fairlogger/Logger.h>
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsCTP/Scalers.h"
#include "DataFormatsCTP/Configuration.h"
#include "TFile.h"
#include "TString.h"
#include <string>
#include <map>
#include <iostream>
#endif
using namespace o2::ctp;
void ReadCTPRunScalersFromFile(std::string name = "s.root")
{
  std::cout << "Reading file:" << name << std::endl;
  TFile* myFile = TFile::Open(name.c_str());
  bool doscalers = 1;
  bool doconfig = 0;
  if (doscalers) {
    // CTPRunScalers* ctpscalers = myFile->Get<CTPRunScalers>("ccdb_object");
    CTPRunScalers* ctpscalers = myFile->Get<CTPRunScalers>("CTPRunScalers");
    if (ctpscalers != nullptr) {
      ctpscalers->printStream(std::cout);
      ctpscalers->convertRawToO2();
      ctpscalers->printIntegrals();
    } else {
      std::cout << "Scalers not there ?" << std::endl;
    }
  }
  if (doconfig) {
    CTPConfiguration* ctpconfig = myFile->Get<CTPConfiguration>("CTPConfig");
    if (ctpconfig != nullptr) {
      ctpconfig->printStream(std::cout);
    } else {
      std::cout << "Config not there ?" << std::endl;
    }
  }
}
