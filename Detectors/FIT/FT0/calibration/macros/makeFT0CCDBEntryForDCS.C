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

int makeFT0CCDBEntryForDCS(const std::string url = "http://localhost:8080")
{
  // Macro to populate CCDB for FT0 with the configuration for DCS

  std::unordered_map<DPID, std::string> dpid2DataDesc;
  std::vector<std::string> aliasesHV = {"FT0/HV/FT0_A/MCP_A[1..5]/actual/iMon",
                                        "FT0/HV/FT0_A/MCP_B[1..5]/actual/iMon",
                                        "FT0/HV/FT0_A/MCP_C[1..2]/actual/iMon",
                                        "FT0/HV/FT0_A/MCP_C[4..5]/actual/iMon",
                                        "FT0/HV/FT0_A/MCP_D[1..5]/actual/iMon",
                                        "FT0/HV/FT0_A/MCP_E[1..5]/actual/iMon",
                                        "FT0/HV/FT0_C/MCP_A[2..5]/actual/iMon",
                                        "FT0/HV/FT0_C/MCP_B[1..6]/actual/iMon",
                                        "FT0/HV/FT0_C/MCP_C[1..2]/actual/iMon",
                                        "FT0/HV/FT0_C/MCP_C[5..6]/actual/iMon",
                                        "FT0/HV/FT0_C/MCP_D[1..2]/actual/iMon",
                                        "FT0/HV/FT0_C/MCP_D[5..6]/actual/iMon",
                                        "FT0/HV/FT0_C/MCP_E[1..6]/actual/iMon",
                                        "FT0/HV/FT0_C/MCP_F[2..5]/actual/iMon",
                                        "FT0/HV/MCP_LC/actual/iMon"};
  std::string aliasesADCZERO = "FT0/PM/channel[000..211]/actual/ADC[0..1]_BASELINE";
  std::vector<std::string> expAliasesHV = o2::dcs::expandAliases(aliasesHV);
  std::vector<std::string> expAliasesADCZERO = o2::dcs::expandAlias(aliasesADCZERO);

  std::cout << "DP aliases:" << std::endl;
  DPID dpidtmp;
  for (size_t i = 0; i < expAliasesHV.size(); ++i) {
    std::cout << expAliasesHV[i] << std::endl;
    DPID::FILL(dpidtmp, expAliasesHV[i], o2::dcs::DeliveryType::DPVAL_DOUBLE);
    dpid2DataDesc[dpidtmp] = "FT0DATAPOINTS";
  }
  for (size_t i = 0; i < expAliasesADCZERO.size(); ++i) {
    std::cout << expAliasesADCZERO[i] << std::endl;
    DPID::FILL(dpidtmp, expAliasesADCZERO[i], o2::dcs::DeliveryType::DPVAL_UINT);
    dpid2DataDesc[dpidtmp] = "FT0DATAPOINTS";
  }

  o2::ccdb::CcdbApi api;
  api.init(url); // or http://localhost:8080 for a local installation
  std::map<std::string, std::string> md;
  long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  api.storeAsTFileAny(&dpid2DataDesc, "FT0/Config/DCSDPconfig", md, ts, 99999999999999);

  return 0;
}
