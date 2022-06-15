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

#include <string>
#include <map>
#include "Framework/Logger.h"
#include "CCDB/CcdbApi.h"
#include "ZDCBase/Constants.h"
#include "ZDCCalib/BaselineCalibConfig.h"

#endif

using namespace o2::zdc;
using namespace std;

void CreateBaselineCalibConfig(long tmin = 0, long tmax = -1, std::string ccdbHost = "")
{

  // This object configures the measurement of the average baseline for the ZDC channels
  BaselineCalibConfig conf;

  // Threshold to include the baseline into the average
//  conf.setCuts(1700, 2000);
  conf.setMinEntries(200);
  conf.setDescription("Simulated data");

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
  api.storeAsTFileAny(&conf, CCDBPathBaselineCalibConfig, metadata, tmin, tmax);
}
