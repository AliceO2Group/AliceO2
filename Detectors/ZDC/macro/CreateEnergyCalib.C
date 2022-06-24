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

#include "ZDCBase/Helpers.h"
using namespace o2::zdc;
using namespace std;

void CreateEnergyCalib(long tmin = 0, long tmax = -1, std::string ccdbHost = "")
{
  // Shortcuts: internal, external, test, local, root

  // This object allows for the calibration of 4 common photomultipliers and 2 ZEM
  // Optionally also the analog sum can have a calibration coefficient otherwise
  // the coefficient of the common PM will be used
  ZDCEnergyParam conf;

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

  std::string ccdb_host = ccdbShortcuts(ccdbHost, conf.Class_Name(), CCDBPathEnergyCalib);

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
  api.storeAsTFileAny(&conf, CCDBPathEnergyCalib, metadata, tmin, tmax);
}
