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

/// \file makeFT0CCDBEntryForDCS.C
/// \brief Macro for uploading a DCS data point definition object to CCDB
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#include "CCDB/CcdbApi.h"
#include "DetectorsDCS/AliasExpander.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "TFile.h"

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

using DPID = o2::dcs::DataPointIdentifier;

int makeFT0CCDBEntryForDCS(const std::string ccdbUrl = "http://localhost:8080",
                           const std::string fileName = "")
{
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
  std::string aliasesADC = "FT0/PM/channel[000..211]/actual/ADC[0..1]_BASELINE";
  std::vector<std::string> expAliasesHV = o2::dcs::expandAliases(aliasesHV);
  std::vector<std::string> expAliasesADC = o2::dcs::expandAlias(aliasesADC);

  LOG(info) << "DCS DP IDs:";

  DPID dpIdTmp;
  for (size_t i = 0; i < expAliasesHV.size(); i++) {
    DPID::FILL(dpIdTmp, expAliasesHV[i], o2::dcs::DeliveryType::DPVAL_DOUBLE);
    dpid2DataDesc[dpIdTmp] = "FT0DATAPOINTS";
    LOG(info) << dpIdTmp;
  }
  for (size_t i = 0; i < expAliasesADC.size(); i++) {
    DPID::FILL(dpIdTmp, expAliasesADC[i], o2::dcs::DeliveryType::DPVAL_UINT);
    dpid2DataDesc[dpIdTmp] = "FT0DATAPOINTS";
    LOG(info) << dpIdTmp;
  }

  LOG(info) << "Total number of DPs: " << dpid2DataDesc.size();

  if (!ccdbUrl.empty()) {
    const std::string ccdbPath = "FT0/Config/DCSDPconfig";
    LOGP(info, "Storing DCS DP definition object on {}/{}", ccdbUrl, ccdbPath);
    o2::ccdb::CcdbApi api;
    api.init(ccdbUrl);
    std::map<std::string, std::string> metadata;
    long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    api.storeAsTFileAny(&dpid2DataDesc, ccdbPath, metadata, ts, 99999999999999);
  }

  if (!fileName.empty()) {
    LOG(info) << "Storing DCS DP definitions locally in " << fileName;
    TFile file(fileName.c_str(), "recreate");
    file.WriteObjectAny(&dpid2DataDesc, "std::unordered_map<o2::dcs::DataPointIdentifier, std::string>", "DCSDPconfig");
    file.Close();
  }

  return 0;
}
