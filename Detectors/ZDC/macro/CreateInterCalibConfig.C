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
#include "ZDCCalib/InterCalib.h"
#include "ZDCCalib/InterCalibConfig.h"
#include "ZDCBase/Constants.h"
#include <string>
#include <TFile.h>
#include <map>

#endif

using namespace o2::zdc;
using namespace std;

void CreateInterCalibConfig(long tmin = 0, long tmax = -1, std::string ccdbHost = "")
{

  // This object allows for the configuration of the intercalibration of the 4 towers of each calorimeter
  // and the calibration of ZEM2 relative to ZEM1

  InterCalibConfig conf;

  // Enable intercalibration for all calorimeters
  // If intercalibration is disabled the intercalibration coefficients
  // are copied from previous valid object and flagged as not modified
  //          ZNA   ZPA   ZNC   ZPC   ZEM2
  conf.enable(true, true, true, true, true);

  // The version for this macro considers NO energy calibration, i.e. all coefficients = 1
  // It is necessary to set the binning
  conf.setBinning1D(1200, 0, 12000);
  conf.setBinning2D(300, 0, 12000);

  conf.setDescription("Simulated, no energy scaling");

  conf.setMinEntries(100);

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
  api.storeAsTFileAny(&conf, CCDBPathInterCalibConfig, metadata, tmin, tmax);

  // return conf;
}
