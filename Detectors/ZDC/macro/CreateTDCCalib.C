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

using namespace std;

void CreateTDCCalib(long tmin = 0, long tmax = -1, std::string ccdbHost = "", float def_shift = 12.5)
{
  o2::zdc::ZDCTDCParam conf;
  // TODO: extract shift from TDC spectra
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
  LOG(info) << "Storing " << o2::zdc::CCDBPathTDCCalib << " on CCDB server: " << api.getURL();
  // store abitrary user object in strongly typed manner
  api.storeAsTFileAny(&conf, o2::zdc::CCDBPathTDCCalib, metadata, tmin, tmax);

  //  return conf;
}
