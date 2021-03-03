// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

int makeCCDBEntryForDCS(const std::string url = "http://localhost:8080")
{

  //  std::string url(argv[0]);
  // macro to populate CCDB for TOF with the configuration for DCS
  std::unordered_map<DPID, std::string> dpid2DataDesc_Common;
  std::unordered_map<DPID, std::string> dpid2DataDesc_Common_1;
  std::vector<std::string> aliasesCommonStr = {"ADAPOS_LG/TEST_000100", "ADAPOS_LG/TEST_000110"};
  std::vector<std::string> aliasesCommon_1_Int = {"ADAPOS_LG/TEST_000240"};
  std::vector<std::string> aliasesCommon_1_Str = {"ADAPOS_LG/TEST_000200"};

  DPID dpidtmp;
  for (size_t i = 0; i < aliasesCommonStr.size(); ++i) {
    DPID::FILL(dpidtmp, aliasesCommonStr[i], o2::dcs::DeliveryType::RAW_STRING);
    dpid2DataDesc_Common[dpidtmp] = "COMMON";
  }
  for (size_t i = 0; i < aliasesCommon_1_Int.size(); ++i) {
    DPID::FILL(dpidtmp, aliasesCommon_1_Int[i], o2::dcs::DeliveryType::RAW_INT);
    dpid2DataDesc_Common_1[dpidtmp] = "COMMON1";
  }
  for (size_t i = 0; i < aliasesCommon_1_Str.size(); ++i) {
    DPID::FILL(dpidtmp, aliasesCommon_1_Str[i], o2::dcs::DeliveryType::RAW_STRING);
    dpid2DataDesc_Common_1[dpidtmp] = "COMMON1";
  }

  o2::ccdb::CcdbApi api;
  api.init(url); // or http://localhost:8080 for a local installation
  std::map<std::string, std::string> md;
  long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  api.storeAsTFileAny(&dpid2DataDesc_Common, "COMMON/DCSconfig", md, ts);
  api.storeAsTFileAny(&dpid2DataDesc_Common_1, "COMMON1/DCSconfig", md, ts);

  return 0;
}
