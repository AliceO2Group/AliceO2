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
#include "ZDCReconstruction/RecoConfigZDC.h"
#include "ZDCBase/Constants.h"
#include <string>
#include <TFile.h>
#include <map>

#endif

#include "ZDCBase/Helpers.h"
using namespace o2::zdc;
using namespace std;

void CreateRecoConfigZDC(long tmin = 0, long tmax = -1, std::string ccdbHost = "")
{
  // Shortcuts: internal, external, test, local, root

  RecoConfigZDC conf;

  // Offline trigger
  // Set trigger bitsincoincidence to ignore dead channels
  // conf.setBit(TDCZNAC);
  // conf.setBit(TDCZNAS);
  // conf.setBit(TDCZPAC);
  // conf.setBit(TDCZPAS);
  // conf.setBit(TDCZEM1);
  // conf.setBit(TDCZEM2);
  // conf.setBit(TDCZNCC);
  // conf.setBit(TDCZNCS);
  // conf.setBit(TDCZPCC);
  // conf.setBit(TDCZPCS);

  // conf.setTripleTrigger();
  conf.setDoubleTrigger();
  // conf.setSingleTrigger();

  // TDC
  int def_search = 250; // Unit of ~10 ps
  conf.setSearch(TDCZNAC, def_search);
  conf.setSearch(TDCZNAS, def_search);
  conf.setSearch(TDCZPAC, def_search);
  conf.setSearch(TDCZPAS, def_search);
  conf.setSearch(TDCZEM1, def_search);
  conf.setSearch(TDCZEM2, def_search);
  conf.setSearch(TDCZNCC, def_search);
  conf.setSearch(TDCZNCS, def_search);
  conf.setSearch(TDCZPCC, def_search);
  conf.setSearch(TDCZPCS, def_search);

  // Charge integration
  int beg_sig = 6;
  int end_sig = 8;
  int beg_ped = -12;
  int end_ped = -8;

  // Integration limits for all signals
  // Values should be in range -12..11
  // Channel ID, signal begin, end, pedestal begin, end

  conf.setIntegration(IdZNAC, beg_sig, end_sig, beg_ped, end_ped);
  conf.setIntegration(IdZNA1, beg_sig, end_sig, beg_ped, end_ped);
  conf.setIntegration(IdZNA2, beg_sig, end_sig, beg_ped, end_ped);
  conf.setIntegration(IdZNA3, beg_sig, end_sig, beg_ped, end_ped);
  conf.setIntegration(IdZNA4, beg_sig, end_sig, beg_ped, end_ped);
  conf.setIntegration(IdZNASum, beg_sig, end_sig, beg_ped, end_ped);
  //
  conf.setIntegration(IdZPAC, beg_sig, end_sig, beg_ped, end_ped);
  conf.setIntegration(IdZPA1, beg_sig, end_sig, beg_ped, end_ped);
  conf.setIntegration(IdZPA2, beg_sig, end_sig, beg_ped, end_ped);
  conf.setIntegration(IdZPA3, beg_sig, end_sig, beg_ped, end_ped);
  conf.setIntegration(IdZPA4, beg_sig, end_sig, beg_ped, end_ped);
  conf.setIntegration(IdZPASum, beg_sig, end_sig, beg_ped, end_ped);
  //
  conf.setIntegration(IdZEM1, beg_sig, end_sig, beg_ped, end_ped);
  conf.setIntegration(IdZEM2, beg_sig, end_sig, beg_ped, end_ped);
  //
  conf.setIntegration(IdZNCC, beg_sig, end_sig, beg_ped, end_ped);
  conf.setIntegration(IdZNC1, beg_sig, end_sig, beg_ped, end_ped);
  conf.setIntegration(IdZNC2, beg_sig, end_sig, beg_ped, end_ped);
  conf.setIntegration(IdZNC3, beg_sig, end_sig, beg_ped, end_ped);
  conf.setIntegration(IdZNC4, beg_sig, end_sig, beg_ped, end_ped);
  conf.setIntegration(IdZNCSum, beg_sig, end_sig, beg_ped, end_ped);
  //
  conf.setIntegration(IdZPCC, beg_sig, end_sig, beg_ped, end_ped);
  conf.setIntegration(IdZPC1, beg_sig, end_sig, beg_ped, end_ped);
  conf.setIntegration(IdZPC2, beg_sig, end_sig, beg_ped, end_ped);
  conf.setIntegration(IdZPC3, beg_sig, end_sig, beg_ped, end_ped);
  conf.setIntegration(IdZPC4, beg_sig, end_sig, beg_ped, end_ped);
  conf.setIntegration(IdZPCSum, beg_sig, end_sig, beg_ped, end_ped);

  // Pile-up detection from pedestal estimation
  // Default value ADCRange will never allow to revert to orbit pedestal and will never identify pile-up
  // Values <=0 will identify all events as pile-up and use always orbit pedestal
  float ped_thr_high = ADCRange;
  float ped_thr_low = ADCRange;
  conf.setPedThreshold(IdZNAC, ped_thr_high, ped_thr_low);
  conf.setPedThreshold(IdZNA1, ped_thr_high, ped_thr_low);
  conf.setPedThreshold(IdZNA2, ped_thr_high, ped_thr_low);
  conf.setPedThreshold(IdZNA3, ped_thr_high, ped_thr_low);
  conf.setPedThreshold(IdZNA4, ped_thr_high, ped_thr_low);
  conf.setPedThreshold(IdZNASum, ped_thr_high, ped_thr_low);

  conf.setPedThreshold(IdZPAC, ped_thr_high, ped_thr_low);
  conf.setPedThreshold(IdZPA1, ped_thr_high, ped_thr_low);
  conf.setPedThreshold(IdZPA2, ped_thr_high, ped_thr_low);
  conf.setPedThreshold(IdZPA3, ped_thr_high, ped_thr_low);
  conf.setPedThreshold(IdZPA4, ped_thr_high, ped_thr_low);
  conf.setPedThreshold(IdZPASum, ped_thr_high, ped_thr_low);

  conf.setPedThreshold(IdZEM1, ped_thr_high, ped_thr_low);
  conf.setPedThreshold(IdZEM2, ped_thr_high, ped_thr_low);

  conf.setPedThreshold(IdZNCC, ped_thr_high, ped_thr_low);
  conf.setPedThreshold(IdZNC1, ped_thr_high, ped_thr_low);
  conf.setPedThreshold(IdZNC2, ped_thr_high, ped_thr_low);
  conf.setPedThreshold(IdZNC3, ped_thr_high, ped_thr_low);
  conf.setPedThreshold(IdZNC4, ped_thr_high, ped_thr_low);
  conf.setPedThreshold(IdZNCSum, ped_thr_high, ped_thr_low);

  conf.setPedThreshold(IdZPCC, ped_thr_high, ped_thr_low);
  conf.setPedThreshold(IdZPC1, ped_thr_high, ped_thr_low);
  conf.setPedThreshold(IdZPC2, ped_thr_high, ped_thr_low);
  conf.setPedThreshold(IdZPC3, ped_thr_high, ped_thr_low);
  conf.setPedThreshold(IdZPC4, ped_thr_high, ped_thr_low);
  conf.setPedThreshold(IdZPCSum, ped_thr_high, ped_thr_low);

  conf.print();

  std::string ccdb_host = ccdbShortcuts(ccdbHost, conf.Class_Name(), CCDBPathRecoConfigZDC);

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
  api.storeAsTFileAny(&conf, CCDBPathRecoConfigZDC, metadata, tmin, tmax);
}
