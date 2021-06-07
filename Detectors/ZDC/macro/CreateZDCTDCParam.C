// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "FairLogger.h"
#include "CCDB/CcdbApi.h"
#include <string>
#include <TH1.h>
#include <TFile.h>
#include <TRandom.h>
#endif

#include "ZDCBase/Constants.h"
#include "ZDCReconstruction/ZDCTDCParam.h"

using namespace std;

void CreateZDCTDCParam(long tmin = 0, long tmax = -1, std::string ccdbHost = "http://ccdb-test.cern.ch:8080")
{
  o2::zdc::ZDCTDCParam conf;
  float def_shift = 14.5;
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

  conf.print();

  o2::ccdb::CcdbApi api;
  map<string, string> metadata; // can be empty
  api.init(ccdbHost.c_str());   // or http://localhost:8080 for a local installation
  // store abitrary user object in strongly typed manner
  api.storeAsTFileAny(&conf, o2::zdc::CCDBPathTDCCalib, metadata, tmin, tmax);

  //  return conf;
}
