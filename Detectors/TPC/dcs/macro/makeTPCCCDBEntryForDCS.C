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

#include <fmt/format.h>
#include <vector>
#include <string>
#include "TFile.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsDCS/AliasExpander.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "TPCdcs/DCSDPHints.h"

#include <variant>
#include <unordered_map>
#include <chrono>

using DPID = o2::dcs::DataPointIdentifier;

/// macro to populate CCDB for TPC with the configuration for DCS
int makeTPCCCDBEntryForDCS(const std::string url = "http://localhost:8080")
{

  std::unordered_map<DPID, std::string> dpid2DataDesc;
  std::vector<std::string> aliases;
  auto dphints = o2::tpc::dcs::getTPCDCSDPHints();
  auto getAlias = [](const auto& hint) {
    return hint.aliasPattern;
  };

  for (const auto& dphint : dphints) {
    aliases.emplace_back(std::visit(getAlias, dphint));
  }

  std::vector<std::string> expaliases = o2::dcs::expandAliases(aliases);

  if (url == "printOnly") {
    for (const auto& alias : expaliases) {
      fmt::print("{}\n", alias.data());
    }
    return 0;
  }

  DPID dpidtmp;
  for (const auto& alias : expaliases) {
    std::string type = "DPVAL_DOUBLE";
    if (alias.find("STATUS") == (alias.size() - std::string("STATUS").size())) {
      DPID::FILL(dpidtmp, alias, o2::dcs::DeliveryType::DPVAL_INT);
      type = "DPVAL_INT";
    } else {
      DPID::FILL(dpidtmp, alias, o2::dcs::DeliveryType::DPVAL_DOUBLE);
    }
    dpid2DataDesc[dpidtmp] = "TPCDATAPOINTS";
    LOGP(info, "{} ({})", alias, type);
  }

  o2::ccdb::CcdbApi api;
  api.init(url); // or http://localhost:8080 for a local installation
  std::map<std::string, std::string> md;
  long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  api.storeAsTFileAny(&dpid2DataDesc, "TPC/Config/DCSDPconfig", md, ts, 99999999999999);

  return 0;
}
