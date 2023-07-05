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
void GetAndSave(std::string ccdbHost = "http://ccdb-test.cern.ch:8080")
{
  std::string CCDBPathCTPScalers = "CTP/Calib/Scalers";
  // std::vector<string> runs = {"518541","518543","518546","518547"};
  // std::vector<long> timestamps = {1655116302316,1655118513690,1655121997478,1655123792911};
  std::vector<string> runs = {"519903", "519904", "519905", "519906"};
  std::vector<long> timestamps = {1656658674161, 1656660737184, 1656667772462, 1656669421115};
  // std::vector<string> runs = {"518543"};
  // std::vector<long> timestamps = {1655118513690};
  int i = 0;
  CTPRunManager mng;
  // mng.setCCDBHost(ccdbHost);
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  mgr.setURL(ccdbHost);
  for (auto const& run : runs) {
    CTPConfiguration ctpcfg;
    CTPRunScalers scl;
    map<string, string> metadata; // can be empty
    metadata["runNumber"] = run;
    CTPRunScalers* ctpscalers = mgr.getSpecific<CTPRunScalers>(CCDBPathCTPScalers, timestamps[i], metadata);
    if (ctpscalers == nullptr) {
      std::cout << run << " CTPRunScalers not in database, timestamp:" << timestamps[i] << std::endl;
    } else {
      // ctpscalers->printStream(std::cout);
      std::string name = run + ".root";
      TFile* myFile = TFile::Open(name.c_str(), "RECREATE");
      myFile->WriteObject(ctpscalers, "CTPRunScalers");
      // myFile->Write();
      std::cout << run << " ok" << std::endl;
    }
    i++;
  }
}
