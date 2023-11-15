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
#include "ZDCReconstruction/ZDCTowerParam.h"
#include <string>
#include <TFile.h>
#include <TString.h>
#include <map>

#endif

#include "ZDCBase/Helpers.h"
using namespace o2::zdc;
using namespace std;

void CreateTowerCalib(long tmin = 0, long tmax = -1, std::string ccdbHost = "")
{
  // Shortcuts: internal, external, test, local, root

  ZDCTowerParam conf;

  // This object allows for the calibration of the 4 towers of each calorimeter
  // The relative calibration coefficients of towers w.r.t. the common PM
  // need to be provided
  // I.e. energy calibration is the product of Common PM calibration (or ZEM1)
  // and tower intercalibration coefficient (or ZEM2)

  conf.setTowerCalib(IdZNA1, 1.);
  conf.setTowerCalib(IdZNA2, 1.);
  conf.setTowerCalib(IdZNA3, 1.);
  conf.setTowerCalib(IdZNA4, 1.);

  conf.setTowerCalib(IdZPA1, 1.);
  conf.setTowerCalib(IdZPA2, 1.);
  conf.setTowerCalib(IdZPA3, 1.);
  conf.setTowerCalib(IdZPA4, 1.);

  conf.setTowerCalib(IdZNC1, 1.);
  conf.setTowerCalib(IdZNC2, 1.);
  conf.setTowerCalib(IdZNC3, 1.);
  conf.setTowerCalib(IdZNC4, 1.);

  conf.setTowerCalib(IdZPC1, 1.);
  conf.setTowerCalib(IdZPC2, 1.);
  conf.setTowerCalib(IdZPC3, 1.);
  conf.setTowerCalib(IdZPC4, 1.);

  // ZEM2 has special calibration: can be calibrated
  // as a common PM and as a tower (equalized to ZEM1)
  // The coefficient applied is the product of the two
  conf.setTowerCalib(IdZEM2, 1.);

  conf.print();

  std::string ccdb_host = ccdbShortcuts(ccdbHost, conf.Class_Name(), CCDBPathTowerCalib);

  if (endsWith(ccdb_host, ".root")) {
    TFile f(TString::Format(ccdb_host.data(), tmin, tmax), "recreate");
    f.WriteObjectAny(&conf, conf.Class_Name(), "ccdb_object");
    f.Close();
    return;
  }

  o2::ccdb::CcdbApi api;
  map<string, string> metadata; // can be empty
  api.init(ccdb_host.c_str());
  LOG(info) << "CCDB server: " << api.getURL();
  // store abitrary user object in strongly typed manner
  api.storeAsTFileAny(&conf, CCDBPathTowerCalib, metadata, tmin, tmax);

  // return conf;
}
