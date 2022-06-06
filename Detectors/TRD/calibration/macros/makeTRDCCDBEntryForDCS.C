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

int makeTRDCCDBEntryForDCS(const std::string url = "http://localhost:8080")
{

  // macro to populate CCDB for TRD with the configuration for DCS
  std::unordered_map<DPID, std::string> dpid2DataDesc;
  std::vector<std::string> aliasesFloat;
  std::vector<std::string> aliasesInt;
  std::vector<std::string> aliasesString;
  aliasesFloat.insert(aliasesFloat.end(), {"trd_gasCO2", "trd_gasH2O", "trd_gasO2"});
  aliasesFloat.insert(aliasesFloat.end(), {"trd_gaschromatographCO2", "trd_gaschromatographN2", "trd_gaschromatographXe"});
  aliasesFloat.insert(aliasesFloat.end(), {"trd_hvAnodeImon[00..539]", "trd_hvAnodeUmon[00..539]", "trd_hvDriftImon[00..539]", "trd_hvDriftUmon[00..539]"});
  aliasesFloat.insert(aliasesFloat.end(), {"trd_aliEnvTempCavern", "trd_aliEnvTempP2"});
  aliasesFloat.insert(aliasesFloat.end(), {"trd_aliEnvPressure00", "trd_aliEnvPressure01", "trd_aliEnvPressure02"});
  aliasesInt.insert(aliasesInt.end(), {"trd_runNo", "trd_runType"});
  // aliasesFloat.insert(aliasesFloat.end(), {"trd_cavernHumidity", "trd_fedEnvTemp[00..539]"});
  // aliasesInt.insert(aliasesInt.end(), {"trd_fedChamberStatus[00..539]"});
  // aliasesString.insert(aliasesString.end(), {"trd_fedCFGtag[00..539]"});

  DPID dpidTmp;
  for (const auto& ali : o2::dcs::expandAliases(aliasesFloat)) {
    DPID::FILL(dpidTmp, ali, o2::dcs::DeliveryType::DPVAL_DOUBLE);
    dpid2DataDesc[dpidTmp] = "TRDDATAPOINTS";
  }
  for (const auto& ali : o2::dcs::expandAliases(aliasesInt)) {
    DPID::FILL(dpidTmp, ali, o2::dcs::DeliveryType::DPVAL_INT);
    dpid2DataDesc[dpidTmp] = "TRDDATAPOINTS";
  }
  for (const auto& ali : o2::dcs::expandAliases(aliasesString)) {
    DPID::FILL(dpidTmp, ali, o2::dcs::DeliveryType::DPVAL_STRING);
    dpid2DataDesc[dpidTmp] = "TRDDATAPOINTS";
  }

  o2::ccdb::CcdbApi api;
  api.init(url);
  std::map<std::string, std::string> md;
  long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  api.storeAsTFileAny(&dpid2DataDesc, "TRD/Config/DCSDPconfig", md, ts, ts + 10 * o2::ccdb::CcdbObjectInfo::YEAR);

  return 0;
}
