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

int makeTPCCCDBEntryForDCS(const std::string url = "http://localhost:8080")
{

  //  std::string url(argv[0]);
  // macro to populate CCDB for TOF with the configuration for DCS
  std::unordered_map<DPID, std::string> dpid2DataDesc;
  std::vector<std::string> aliases;
  int maxSectors = 17;
  aliases.emplace_back("TPC_GC_ARGON");
  aliases.emplace_back("TPC_GC_CO2");
  aliases.emplace_back("TPC_GC_N2");
  aliases.emplace_back("TPC_GC_NEON");
  aliases.emplace_back("TPC_GC_O2");
  aliases.emplace_back("TPC_GC_WATER");
  aliases.emplace_back("TPC_An_L1Sr141_H2O");
  aliases.emplace_back("TPC_An_L1Sr141_O2");
  aliases.emplace_back("TPC_PT_351_TEMPERATURE");
  aliases.emplace_back("TPC_PT_376_TEMPERATURE");
  aliases.emplace_back("TPC_PT_415_TEMPERATURE");
  aliases.emplace_back("TPC_PT_447_TEMPERATURE");
  aliases.emplace_back("TPC_PT_477_TEMPERATURE");
  aliases.emplace_back("TPC_PT_488_TEMPERATURE");
  aliases.emplace_back("TPC_PT_537_TEMPERATURE");
  aliases.emplace_back("TPC_PT_575_TEMPERATURE");
  aliases.emplace_back("TPC_PT_589_TEMPERATURE");
  aliases.emplace_back("TPC_PT_629_TEMPERATURE");
  aliases.emplace_back("TPC_PT_664_TEMPERATURE");
  aliases.emplace_back("TPC_PT_695_TEMPERATURE");
  aliases.emplace_back("TPC_PT_735_TEMPERATURE");
  aliases.emplace_back("TPC_PT_757_TEMPERATURE");
  aliases.emplace_back("TPC_PT_797_TEMPERATURE");
  aliases.emplace_back("TPC_PT_831_TEMPERATURE");
  aliases.emplace_back("TPC_PT_851_TEMPERATURE");
  aliases.emplace_back("TPC_PT_895_TEMPERATURE");
  aliases.emplace_back(fmt::format("TPC_HV_A[00..{:02}]_I_G[1..4]B_U", maxSectors));
  aliases.emplace_back(fmt::format("TPC_HV_A[00..{:02}]_O[1..3]_G[1..4]B_U", maxSectors));
  aliases.emplace_back(fmt::format("TPC_HV_A[00..{:02}]_I_G[1..4]T_U", maxSectors));
  aliases.emplace_back(fmt::format("TPC_HV_A[00..{:02}]_O[1..3]_G[1..4]T_U", maxSectors));
  aliases.emplace_back(fmt::format("TPC_HV_C[00..{:02}]_I_G[1..4]B_U", maxSectors));
  aliases.emplace_back(fmt::format("TPC_HV_C[00..{:02}]_O[1..3]_G[1..4]B_U", maxSectors));
  aliases.emplace_back(fmt::format("TPC_HV_C[00..{:02}]_I_G[1..4]T_U", maxSectors));
  aliases.emplace_back(fmt::format("TPC_HV_C[00..{:02}]_O[1..3]_G[1..4]T_U", maxSectors));
  aliases.emplace_back(fmt::format("TPC_HV_A[00..{:02}]_I_G[1..4]B_I", maxSectors));
  aliases.emplace_back(fmt::format("TPC_HV_A[00..{:02}]_O[1..3]_G[1..4]B_I", maxSectors));
  aliases.emplace_back(fmt::format("TPC_HV_A[00..{:02}]_I_G[1..4]T_I", maxSectors));
  aliases.emplace_back(fmt::format("TPC_HV_A[00..{:02}]_O[1..3]_G[1..4]T_I", maxSectors));
  aliases.emplace_back(fmt::format("TPC_HV_C[00..{:02}]_I_G[1..4]B_I", maxSectors));
  aliases.emplace_back(fmt::format("TPC_HV_C[00..{:02}]_O[1..3]_G[1..4]B_I", maxSectors));
  aliases.emplace_back(fmt::format("TPC_HV_C[00..{:02}]_I_G[1..4]T_I", maxSectors));
  aliases.emplace_back(fmt::format("TPC_HV_C[00..{:02}]_O[1..3]_G[1..4]T_I", maxSectors));

  std::vector<std::string> expaliases = o2::dcs::expandAliases(aliases);

  DPID dpidtmp;
  for (size_t i = 0; i < expaliases.size(); ++i) {
    DPID::FILL(dpidtmp, expaliases[i], o2::dcs::DeliveryType::RAW_DOUBLE);
    dpid2DataDesc[dpidtmp] = "TPCDATAPOINTS";
    LOG(info) << expaliases[i];
  }

  o2::ccdb::CcdbApi api;
  api.init(url); // or http://localhost:8080 for a local installation
  std::map<std::string, std::string> md;
  long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  api.storeAsTFileAny(&dpid2DataDesc, "TPC/Config/DCSDPconfig", md, ts);

  return 0;
}
