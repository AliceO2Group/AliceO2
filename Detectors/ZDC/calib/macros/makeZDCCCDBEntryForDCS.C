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

int makeZDCCCDBEntryForDCS(const std::string url = "http://ccdb-test.cern.ch:8080") // localhost
{

  //  std::string url(argv[0]);
  // macro to populate CCDB for ZDC with the configuration for DCS
  std::unordered_map<DPID, std::string> dpid2DataDesc;
  std::vector<std::string> aliases = {"ZDC_ZNA_POS.actual.position",
                                      "ZDC_ZPA_POS.actual.position",
                                      "ZDC_ZNC_POS.actual.position",
                                      "ZDC_ZPC_POS.actual.position",
                                      "ZDC_ZNA_HV0.actual.vMon",
                                      "ZDC_ZNA_HV1.actual.vMon",
                                      "ZDC_ZNA_HV2.actual.vMon",
                                      "ZDC_ZNA_HV3.actual.vMon",
                                      "ZDC_ZNA_HV4.actual.vMon",
                                      "ZDC_ZPA_HV0.actual.vMon",
                                      "ZDC_ZPA_HV1.actual.vMon",
                                      "ZDC_ZPA_HV2.actual.vMon",
                                      "ZDC_ZPA_HV3.actual.vMon",
                                      "ZDC_ZPA_HV4.actual.vMon",
                                      "ZDC_ZNC_HV0.actual.vMon",
                                      "ZDC_ZNC_HV1.actual.vMon",
                                      "ZDC_ZNC_HV2.actual.vMon",
                                      "ZDC_ZNC_HV3.actual.vMon",
                                      "ZDC_ZNC_HV4.actual.vMon",
                                      "ZDC_ZPC_HV0.actual.vMon",
                                      "ZDC_ZPC_HV1.actual.vMon",
                                      "ZDC_ZPC_HV2.actual.vMon",
                                      "ZDC_ZPC_HV3.actual.vMon",
                                      "ZDC_ZPC_HV4.actual.vMon",
                                      "ZDC_ZEM_HV0.actual.vMon",
                                      "ZDC_ZEM_HV1.actual.vMon",
                                      "ZDC_ZNA_HV0_D[1..2]",
                                      "ZDC_ZNC_HV0_D[1..2]",
                                      "ZDC_ZPA_HV0_D[1..2]",
                                      "ZDC_ZPC_HV0_D[1..2]"};
  std::vector<std::string> aliasesInt = {"ZDC_CONFIG_[00..32]"};
  std::vector<std::string> expaliases = o2::dcs::expandAliases(aliases);
  std::vector<std::string> expaliasesInt = o2::dcs::expandAliases(aliasesInt);

  DPID dpidtmp;
  for (size_t i = 0; i < expaliases.size(); ++i) {
    DPID::FILL(dpidtmp, expaliases[i], o2::dcs::DeliveryType::DPVAL_DOUBLE);
    dpid2DataDesc[dpidtmp] = "ZDCDATAPOINTS";
  }
  for (size_t i = 0; i < expaliasesInt.size(); ++i) {
    DPID::FILL(dpidtmp, expaliasesInt[i], o2::dcs::DeliveryType::DPVAL_INT);
    dpid2DataDesc[dpidtmp] = "ZDCDATAPOINTS";
  }

  o2::ccdb::CcdbApi api;
  api.init(url); // or http://localhost:8080 for a local installation
  std::map<std::string, std::string> md;
  long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  api.storeAsTFileAny(&dpid2DataDesc, "ZDC/Calib/DCSconfig", md, ts, 99999999999999);

  return 0;
}
