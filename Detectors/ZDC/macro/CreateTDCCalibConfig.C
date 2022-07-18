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
#include <string>
#include <TFile.h>
#include <map>

#endif

#include "ZDCBase/Helpers.h"
#include "ZDCBase/Constants.h"
#include "ZDCCalib/TDCCalibConfig.h"

using namespace o2::zdc;
using namespace std;

void CreateTDCCalibConfig(long tmin = 0, long tmax = -1, std::string ccdbHost = "")
{

  // This object allows for the configuration of the TDC calibration of the common PM
  // and the sum of each calorimeter and the two ZEMs

  TDCCalibConfig conf;

  // Enable TDC calibration for all calorimeters
  // If TDC calibration is disabled the calibration coefficients
  // are copied from previous valid object and flagged as not modified
  //          ZNAC  ZNAS  ZPAC  ZPAS  ZEM1  ZECM2 ZNCC  ZNCS  ZPCC  ZPCS
  conf.enable(true, true, true, true, true, true, true, true, true, true);

  // Set the binning
  conf.setBinning1D(100, -5, 5);
  conf.setBinning2D(50, -5, 5);

  conf.setDescription("Simulated data");

  //conf.setMinEntries(100); //To be decided the number of minimum entries

  conf.print();

  std::string ccdb_host = ccdbShortcuts(ccdbHost, conf.Class_Name(), CCDBPathTDCCalibConfig);

  if (endsWith(ccdb_host, ".root")) {
    TFile f(ccdb_host.data(), "recreate");
    f.WriteObjectAny(&conf, conf.Class_Name(), "ccdb_object");
    f.Close();
    return;
  }

  o2::ccdb::CcdbApi api;
  map<string, string> metadata; // can be empty
  api.init(ccdb_host.c_str());
  LOG(info) << "CCDB server: " << api.getURL();
  // store abitrary user object in strongly typed manner
  api.storeAsTFileAny(&conf, CCDBPathTDCCalibConfig, metadata, tmin, tmax);
}