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
  aliasesFloat.insert(aliasesFloat.end(), {"trd_gasCO2", "trd_gasH2O", "trd_gasO2"});
  aliasesFloat.insert(aliasesFloat.end(), {"trd_gaschromatographCO2", "trd_gaschromatographN2", "trd_gaschromatographXe"});
  // aliasesFloat.insert(aliasesFloat.end(), {"trd_hvAnodeImon[00..539]", "trd_hvAnodeUmon[00..539]", "trd_hvDriftImon[00..539]", "trd_hvDriftImon[00..539]"});
  // std::vector<std::string> aliasesInt = {"trd_fedChamberStatus[00..539]", "trd_runNo", "trd_runType"};
  std::vector<std::string> aliasesInt = {"trd_runNo", "trd_runType"};
  std::vector<std::string> expAliasesFloat = o2::dcs::expandAliases(aliasesFloat);
  std::vector<std::string> expAliasesInt = o2::dcs::expandAliases(aliasesInt);

  DPID dpidTmp;
  for (size_t i = 0; i < expAliasesFloat.size(); ++i) {
    DPID::FILL(dpidTmp, expAliasesFloat[i], o2::dcs::DeliveryType::DPVAL_DOUBLE);
    dpid2DataDesc[dpidTmp] = "TRDDATAPOINTS";
  }
  for (size_t i = 0; i < expAliasesInt.size(); ++i) {
    DPID::FILL(dpidTmp, expAliasesInt[i], o2::dcs::DeliveryType::DPVAL_INT);
    dpid2DataDesc[dpidTmp] = "TRDDATAPOINTS";
  }

  o2::ccdb::CcdbApi api;
  api.init(url);
  std::map<std::string, std::string> md;
  long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  api.storeAsTFileAny(&dpid2DataDesc, "TRD/Config/DCSDPconfig", md, ts);

  return 0;
}
