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

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "Framework/Logger.h"
#include "CCDB/CcdbApi.h"
#include "ZDCBase/Constants.h"
#include "ZDCReconstruction/BaselineParam.h"
#include <string>
#include <TFile.h>
#include <map>

#endif

using namespace o2::zdc;
using namespace std;

void CreateBaselineCalib(long tmin = 0, long tmax = -1, std::string ccdbHost = "")
{

  BaselineParam conf;

  // This object allows to provide average pedestals
  // Default object has not valid data = -std::numeric_limits<float>::infinity()
  // that makes sure the pedestal are not used

//   conf.setCalib(IdZNAC, -std::numeric_limits<float>::infinity());
//   conf.setCalib(IdZNA1, -std::numeric_limits<float>::infinity());
//   conf.setCalib(IdZNA2, -std::numeric_limits<float>::infinity());
//   conf.setCalib(IdZNA3, -std::numeric_limits<float>::infinity());
//   conf.setCalib(IdZNA4, -std::numeric_limits<float>::infinity());
//   conf.setCalib(IdZNASum, -std::numeric_limits<float>::infinity());
// 
//   conf.setCalib(IdZPAC, -std::numeric_limits<float>::infinity());
//   conf.setCalib(IdZPA1, -std::numeric_limits<float>::infinity());
//   conf.setCalib(IdZPA2, -std::numeric_limits<float>::infinity());
//   conf.setCalib(IdZPA3, -std::numeric_limits<float>::infinity());
//   conf.setCalib(IdZPA4, -std::numeric_limits<float>::infinity());
//   conf.setCalib(IdZPASum, -std::numeric_limits<float>::infinity());
// 
//   conf.setCalib(IdZEM1, -std::numeric_limits<float>::infinity());
//   conf.setCalib(IdZEM2, -std::numeric_limits<float>::infinity());
// 
//   conf.setCalib(IdZNCC, -std::numeric_limits<float>::infinity());
//   conf.setCalib(IdZNC1, -std::numeric_limits<float>::infinity());
//   conf.setCalib(IdZNC2, -std::numeric_limits<float>::infinity());
//   conf.setCalib(IdZNC3, -std::numeric_limits<float>::infinity());
//   conf.setCalib(IdZNC4, -std::numeric_limits<float>::infinity());
//   conf.setCalib(IdZNCSum, -std::numeric_limits<float>::infinity());
// 
//   conf.setCalib(IdZPCC, -std::numeric_limits<float>::infinity());
//   conf.setCalib(IdZPC1, -std::numeric_limits<float>::infinity());
//   conf.setCalib(IdZPC2, -std::numeric_limits<float>::infinity());
//   conf.setCalib(IdZPC3, -std::numeric_limits<float>::infinity());
//   conf.setCalib(IdZPC4, -std::numeric_limits<float>::infinity());
//   conf.setCalib(IdZPCSum, -std::numeric_limits<float>::infinity());

  conf.print();

  o2::ccdb::CcdbApi api;
  map<string, string> metadata; // can be empty
  if (ccdbHost.size() == 0 || ccdbHost == "external") {
    ccdbHost = "http://alice-ccdb.cern.ch:8080";
  } else if (ccdbHost == "internal") {
    ccdbHost = "http://o2-ccdb.internal/";
  } else if (ccdbHost == "test") {
    ccdbHost = "http://ccdb-test.cern.ch:8080";
  } else if (ccdbHost == "local") {
    ccdbHost = "http://localhost:8080";
  }
  api.init(ccdbHost.c_str());
  LOG(info) << "CCDB server: " << api.getURL();
  // store abitrary user object in strongly typed manner
  api.storeAsTFileAny(&conf, CCDBPathBaselineCalib, metadata, tmin, tmax);

  // return conf;
}
