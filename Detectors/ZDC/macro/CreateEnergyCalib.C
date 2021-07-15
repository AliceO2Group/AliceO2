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
#include "ZDCReconstruction/ZDCEnergyParam.h"
#include <string>
#include <TFile.h>
#include <map>

#endif

using namespace o2::zdc;
using namespace std;

void CreateEnergyCalib(long tmin = 0, long tmax = -1,
                       std::string ccdbHost = "http://ccdb-test.cern.ch:8080")
{

  ZDCEnergyParam conf;

  // This object allows for the calibration of 4 common photomultipliers and 2 ZEM
  // Optionally also the analog sum can have a calibration coefficient otherwise
  // the coefficient of the common PM will be used
  conf.setEnergyCalib(IdZNAC, 1.);
  conf.setEnergyCalib(IdZPAC, 1.);
  conf.setEnergyCalib(IdZEM1, 1.);
  conf.setEnergyCalib(IdZEM2, 1.);
  conf.setEnergyCalib(IdZNCC, 1.);
  conf.setEnergyCalib(IdZPCC, 1.);

  //   conf.setEnergyCalib(IdZNASum, 1.);
  //   conf.setEnergyCalib(IdZPASum, 1.);
  //   conf.setEnergyCalib(IdZNCSum, 1.);
  //   conf.setEnergyCalib(IdZPCSum, 1.);

  conf.print();

  o2::ccdb::CcdbApi api;
  map<string, string> metadata; // can be empty
  api.init(ccdbHost.c_str());   // or http://localhost:8080 for a local installation
  // store abitrary user object in strongly typed manner
  api.storeAsTFileAny(&conf, CCDBPathEnergyCalib, metadata, tmin, tmax);

  // return conf;
}
