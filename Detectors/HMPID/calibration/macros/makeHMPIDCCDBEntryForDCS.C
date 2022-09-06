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

int makeHMPIDCCDBEntryForDCS(const std::string url = "localhost:8080")
{

  // std::string url(argv[0]);
  // macro to populate CCDB for HMPID with the configuration for DCS
  std::vector<std::string> aliases; // vector of strings that will hold DataPoints identifiers

  aliases = {"HMP_ENV_PENV",
             "HMP_MP[0..6]_GAS_PMWPC",
             "HMP_MP[0..6]_LIQ_LOOP_RAD_[0..2]_IN_TEMP",
             "HMP_MP[0..6]_LIQ_LOOP_RAD_[0..2]_OUT_TEMP",
             "HMP_MP_[0..6]_SEC_[0..5]_HV_VMON",
             "HMP_TRANPLANT_MEASURE_[0..29]_WAVELENGHT",
             "HMP_TRANPLANT_MEASURE_[0..29]_ARGONREFERENCE",
             "HMP_TRANPLANT_MEASURE_[0..29]_ARGONCELL",
             "HMP_TRANPLANT_MEASURE_[0..29]_C6F14REFERENCE",
             "HMP_TRANPLANT_MEASURE_[0..29]_C6F14CELL"};

  std::vector<std::string> expaliases = o2::dcs::expandAliases(aliases);

  std::unordered_map<DPID, std::string> dpid2DataDesc;
  DPID dpidtmp;
  for (size_t i = 0; i < expaliases.size(); ++i) {
    DPID::FILL(dpidtmp, expaliases[i], o2::dcs::DeliveryType::DPVAL_DOUBLE);
    dpid2DataDesc[dpidtmp] = "HMPDATAPOINTS";
  }

  o2::ccdb::CcdbApi api;
  api.init(url); // or http://localhost:8080 for a local installation
  std::map<std::string, std::string> md;
  long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  api.storeAsTFileAny(&dpid2DataDesc, "HMP/Config/DCSDPconfig", md, ts, 99999999999999);

  return 0;
}
