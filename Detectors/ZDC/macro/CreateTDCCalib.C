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
#include <TH1.h>
#include <TFile.h>
#include <TRandom.h>
#endif

#include "ZDCBase/Constants.h"
#include "ZDCReconstruction/ZDCTDCParam.h"
#include "ZDCBase/Helpers.h"
using namespace std;

void CreateTDCCalib(long tmin = 0, long tmax = -1, std::string ccdbHost = "", float def_shift = 12.5)
{
  // Shortcuts: internal, external, test, local, root

  o2::zdc::ZDCTDCParam conf;

  conf.setShift(o2::zdc::TDCZNAC, def_shift);
  conf.setShift(o2::zdc::TDCZNAS, def_shift);
  conf.setShift(o2::zdc::TDCZPAC, def_shift);
  conf.setShift(o2::zdc::TDCZPAS, def_shift);
  conf.setShift(o2::zdc::TDCZEM1, def_shift);
  conf.setShift(o2::zdc::TDCZEM2, def_shift);
  conf.setShift(o2::zdc::TDCZNCC, def_shift);
  conf.setShift(o2::zdc::TDCZNCS, def_shift);
  conf.setShift(o2::zdc::TDCZPCC, def_shift);
  conf.setShift(o2::zdc::TDCZPCS, def_shift);

  conf.setFactor(o2::zdc::TDCZNAC, 1.);
  conf.setFactor(o2::zdc::TDCZNAS, 1.);
  conf.setFactor(o2::zdc::TDCZPAC, 1.);
  conf.setFactor(o2::zdc::TDCZPAS, 1.);
  conf.setFactor(o2::zdc::TDCZEM1, 1.);
  conf.setFactor(o2::zdc::TDCZEM2, 1.);
  conf.setFactor(o2::zdc::TDCZNCC, 1.);
  conf.setFactor(o2::zdc::TDCZNCS, 1.);
  conf.setFactor(o2::zdc::TDCZPCC, 1.);
  conf.setFactor(o2::zdc::TDCZPCS, 1.);

  conf.print();

  std::string ccdb_host = o2::zdc::ccdbShortcuts(ccdbHost, conf.Class_Name(), o2::zdc::CCDBPathTDCCalib);

  if (o2::zdc::endsWith(ccdb_host, ".root")) {
    TFile f(TString::Format(ccdb_host.data(), tmin, tmax), "recreate");
    f.WriteObjectAny(&conf, conf.Class_Name(), "ccdb_object");
    f.Close();
    return;
  }

  o2::ccdb::CcdbApi api;
  map<string, string> metadata; // can be empty
  api.init(ccdb_host.c_str());
  LOG(info) << "Storing " << o2::zdc::CCDBPathTDCCalib << " on CCDB server: " << api.getURL();
  // store abitrary user object in strongly typed manner
  api.storeAsTFileAny(&conf, o2::zdc::CCDBPathTDCCalib, metadata, tmin, tmax);

  //  return conf;
}
