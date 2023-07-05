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

/// \file makeFV0CCDBEntryForDCS.C
/// \brief Macro for uploading a DCS data point definition object to CCDB
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#include "CCDB/CcdbApi.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "DetectorsDCS/AliasExpander.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "TFile.h"

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

using DPID = o2::dcs::DataPointIdentifier;

int makeFV0CCDBEntryForDCS(const std::string ccdbUrl = "http://localhost:8080",
                           const std::string fileName = "")
{
  // FV0/PM/??/actual/ADC0_BASELINE
  // FV0/PM/??/actual/ADC1_BASELINE
  // FV0/HV/??/actual/iMon

  // where ?? can be SX1,SX2,SX3,SX4,SX51,SX52 - where X={A,B,C,D,E,F,G,H}
  //                 or SREF
  std::unordered_map<DPID, std::string> dpid2DataDesc;
  std::vector<std::string> aliasesHV = {"FV0/HV/S[A,B,C,D,E,F,G,H][1..4]/actual/iMon",
                                        "FV0/HV/S[A,B,C,D,E,F,G,H][51,52]/actual/iMon",
                                        "FV0/HV/SREF/actual/iMon"};
  std::vector<std::string> aliasesADC = {"FV0/PM/S[A,B,C,D,E,F,G,H][1..4]/actual/ADC[0,1]_BASELINE",
                                         "FV0/PM/S[A,B,C,D,E,F,G,H][51,52]/actual/ADC[0,1]_BASELINE",
                                         "FV0/PM/SREF/actual/ADC[0,1]_BASELINE"};

  std::vector<std::string> aliasesRates = {"FV0/Trigger1_Charge/CNT_RATE",
                                           "FV0/Trigger2_Nchannels/CNT_RATE",
                                           "FV0/Trigger3_InnerRings/CNT_RATE",
                                           "FV0/Trigger4_OuterRings/CNT_RATE",
                                           "FV0/Trigger5_OrA/CNT_RATE",
                                           "FV0/Background/[0..9]/CNT_RATE",
                                           "FV0/Background/[A,B,C,D,E,F,G,H]/CNT_RATE"};
  std::vector<std::string> expAliasesHV = o2::dcs::expandAliases(aliasesHV);
  std::vector<std::string> expAliasesADC = o2::dcs::expandAliases(aliasesADC);
  std::vector<std::string> expAliasesRates = o2::dcs::expandAliases(aliasesRates);

  LOG(info) << "DCS DP IDs:";

  DPID dpIdTmp;
  for (size_t i = 0; i < expAliasesHV.size(); i++) {
    DPID::FILL(dpIdTmp, expAliasesHV[i], o2::dcs::DeliveryType::DPVAL_DOUBLE);
    dpid2DataDesc[dpIdTmp] = "FV0DATAPOINTS";
    LOG(info) << dpIdTmp;
  }
  for (size_t i = 0; i < expAliasesADC.size(); i++) {
    DPID::FILL(dpIdTmp, expAliasesADC[i], o2::dcs::DeliveryType::DPVAL_UINT);
    dpid2DataDesc[dpIdTmp] = "FV0DATAPOINTS";
    LOG(info) << dpIdTmp;
  }
  for (size_t i = 0; i < expAliasesRates.size(); i++) {
    DPID::FILL(dpIdTmp, expAliasesRates[i], o2::dcs::DeliveryType::DPVAL_DOUBLE);
    dpid2DataDesc[dpIdTmp] = "FV0DATAPOINTS";
    LOG(info) << dpIdTmp;
  }

  LOG(info) << "Total number of DPs: " << dpid2DataDesc.size();

  if (!ccdbUrl.empty()) {
    const std::string ccdbPath = "FV0/Config/DCSDPconfig";
    LOGP(info, "Storing DCS DP definition object on {}/{}", ccdbUrl, ccdbPath);
    o2::ccdb::CcdbApi api;
    api.init(ccdbUrl);
    std::map<std::string, std::string> metadata;
    long ts = o2::ccdb::getCurrentTimestamp();
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
