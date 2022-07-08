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

#include "FairLogger.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsCTP/Scalers.h"
#include "DataFormatsCTP/Configuration.h"
#include <string>
#include <map>
#include <iostream>
#endif
using namespace o2::ctp;
void SaveInputsConfig(std::string filename = "inputs.cfg", std::string ccdbHost = "http://ccdb-test.cern.ch:8080")
{
  bool ok;
  CTPInputsConfiguration ctpcfginps;
  ctpcfginps.createInputsConfigFromFile(filename);
  ctpcfginps.printStream(std::cout);
  //
  // data base
  using namespace std::chrono_literals;
  std::chrono::seconds days365 = 31536000s;
  long time365days = std::chrono::duration_cast<std::chrono::milliseconds>(days365).count();
  const auto now = std::chrono::system_clock::now();
  const long timeStamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  long tmin = timeStamp;
  long tmax = timeStamp + time365days;
  o2::ccdb::CcdbApi api;
  api.init(ccdbHost);           // or http://localhost:8080 for a local installation
  map<string, string> metadata; // can be empty
  // store abitrary user object in strongly typed manner
  api.storeAsTFileAny(&ctpcfginps, "CTP/Calib/Inputs", metadata, tmin, tmax);
}
