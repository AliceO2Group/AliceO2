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

#include <vector>
#include <string>
#include "TFile.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsDCS/AliasExpander.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/DataPointIdentifier.h"

#include <unordered_map>
#include <chrono>

using DPID = o2::dcs::DataPointIdentifier;

int makeHMPIDCCDBEntryForDCS(const std::string url = "http://localhost:8080")
{

  //  std::string url(argv[0]);
  // macro to populate CCDB for HMPID with the configuration for DCS
  std::unordered_map<DPID, std::string> dpid2DataDesc;
  std::vector<std::string> aliases; // vector of strings that will hold DataPoints identifiers
  
   // ==| Environment Pressure  (mBar) |=================================
  aliases.push_back("HMP_DET/HMP_ENV/HMP_ENV_PENV.actual.value");

  int maxChambers = 7;
  // ==|(CH4) Chamber Pressures  (mBar?) |=================================
  aliases.push_back(fmt::format("HMP_DET/HMP_MP[0..{}]/HMP_MP[0..{}]_GAS/HMP_MP[0..{}]_GAS_PMWPC.actual.value",maxChambers,maxChambers,maxChambers));	

  //==| Temperature C6F14 IN/OUT / RADIATORS  (C) |=================================
  int iRad = 3; 

  aliases.push_back(fmt::format("HMP_DET/HMP_MP[0..{}]/HMP_MP[0..{}]_LIQ_LOOP.actual.sensors.Rad[0..{}]In_Temp",maxChambers,maxChambers,iRad));
  aliases.push_back(fmt::format("HMP_DET/HMP_MP[0..{}]/HMP_MP[0..{}]_LIQ_LOOP.actual.sensors.Rad[0..{}]Out_Temp",maxChambers,maxChambers,iRad));	

  // ===| HV / SECTORS (V) |=========================================================	      
  int iSec = 6; 
  aliases.push_back(o2::dcs::test::DataPointHint<double>{fmt::format("HMP_DET/HMP_MP[0..{}]/HMP_MP[0..{}]_PW/HMP_MP[0..{}]_SEC[0..{}]/HMP_MP[0..{}]_SEC[0..{}]_HV.actual.vMon",maxChambers,maxChambers,maxChambers,iSec,maxChambers,iSec), 2400., 2500.});


  // string for DPs of Refractive Index Parameters =============================================================
  aliases.push_back(fmt::format("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure[00..29].waveLenght"));
  aliases.push_back(fmt::format("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure[00..29].argonReference"));
  aliases.push_back(fmt::format("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure[00..29].argonCell"));
  aliases.push_back(fmt::format("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure[00..29].c6f14Cell"));
  aliases.push_back(fmt::format("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure[00..29].c6f14Reference"));     

  std::vector<std::string> expaliases = o2::dcs::expandAliases(aliases);

  DPID dpidtmp;
  for (size_t i = 0; i < expaliases.size(); ++i) {
    DPID::FILL(dpidtmp, expaliases[i], o2::dcs::DeliveryType::DPVAL_DOUBLE);
    dpid2DataDesc[dpidtmp] = "HMPIDDATAPOINTS";
  }

  o2::ccdb::CcdbApi api;
  api.init(url); // or http://localhost:8080 for a local installation
  std::map<std::string, std::string> md;
  long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  api.storeAsTFileAny(&dpid2DataDesc, "HMPID/Config/DCSDPconfig", md, ts, 99999999999999);

  return 0;
}


