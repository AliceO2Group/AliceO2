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
#include <map>

#endif

using namespace o2::zdc;
using namespace std;

void CreateTowerCalib(long tmin = 0, long tmax = -1,
                      std::string ccdbHost = "http://ccdb-test.cern.ch:8080")
{

  ZDCTowerParam conf;

  // This object allows for the calibration of the 4 towers of each calorimeter
  // The relative calibration coefficients of towers w.r.t. the common PM
  // need to be provided

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

  conf.print();

  o2::ccdb::CcdbApi api;
  map<string, string> metadata; // can be empty
  api.init(ccdbHost.c_str());   // or http://localhost:8080 for a local installation
  // store abitrary user object in strongly typed manner
  api.storeAsTFileAny(&conf, CCDBPathTowerCalib, metadata, tmin, tmax);

  // return conf;
}
