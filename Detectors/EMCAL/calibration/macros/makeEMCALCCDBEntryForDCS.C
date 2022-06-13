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

int makeEMCALCCDBEntryForDCS(const std::string url = "http://ccdb-test.cern.ch:8080")
{

  //  std::string url(argv[0]);
  // macro to populate CCDB for EMC with the configuration for DCS
  std::unordered_map<DPID, std::string> dpid2DataDesc;

  std::vector<std::string> aliasesTEMP = {"EMC_PT_[00..83].Temperature", "EMC_PT_[88..91].Temperature", "EMC_PT_[96..159].Temperature"};
  std::vector<std::string> aliasesUINT = {"EMC_DDL_LIST[0..1]", "EMC_SRU[00..19]_CFG", "EMC_SRU[00..19]_FMVER",
                                          "EMC_TRU[00..45]_PEAKFINDER", "EMC_TRU[00..45]_L0ALGSEL", "EMC_TRU[00..45]_COSMTHRESH",
                                          "EMC_TRU[00..45]_GLOBALTHRESH", "EMC_TRU[00..45]_MASK[0..5]",
                                          "EMC_STU_ERROR_COUNT_TRU[0..67]", "DMC_STU_ERROR_COUNT_TRU[0..55]"};
  std::vector<std::string> aliasesINT = {"EMC_STU_FWVERS", "EMC_STU_GA[0..1]", "EMC_STU_GB[0..1]", "EMC_STU_GC[0..1]",
                                         "EMC_STU_JA[0..1]", "EMC_STU_JB[0..1]", "EMC_STU_JC[0..1]", "EMC_STU_PATCHSIZE", "EMC_STU_GETRAW",
                                         "EMC_STU_MEDIAN", "EMC_STU_REGION", "DMC_STU_FWVERS", "DMC_STU_PHOS_scale[0..3]", "DMC_STU_GA[0..1]",
                                         "DMC_STU_GB[0..1]", "DMC_STU_GC[0..1]", "DMC_STU_JA[0..1]", "DMC_STU_JB[0..1]", "DMC_STU_JC[0..1]",
                                         "DMC_STU_PATCHSIZE", "DMC_STU_GETRAW", "DMC_STU_MEDIAN", "DMC_STU_REGION", "EMC_RUNNUMBER"};

  std::vector<std::string> expaliasesTEMP = o2::dcs::expandAliases(aliasesTEMP);
  std::vector<std::string> expaliasesUINT = o2::dcs::expandAliases(aliasesUINT);
  std::vector<std::string> expaliasesINT = o2::dcs::expandAliases(aliasesINT);

  DPID dpidtmp;
  for (size_t i = 0; i < expaliasesTEMP.size(); ++i) {
    DPID::FILL(dpidtmp, expaliasesTEMP[i], o2::dcs::DeliveryType::DPVAL_DOUBLE);
    dpid2DataDesc[dpidtmp] = "EMCDATAPOINTS";
  }
  for (size_t i = 0; i < expaliasesINT.size(); ++i) {
    DPID::FILL(dpidtmp, expaliasesINT[i], o2::dcs::DeliveryType::DPVAL_INT);
    dpid2DataDesc[dpidtmp] = "EMCDATAPOINTS";
  }
  for (size_t i = 0; i < expaliasesUINT.size(); ++i) {
    DPID::FILL(dpidtmp, expaliasesUINT[i], o2::dcs::DeliveryType::DPVAL_UINT);
    dpid2DataDesc[dpidtmp] = "EMCDATAPOINTS";
  }

  o2::ccdb::CcdbApi api;
  api.init(url); // or http://localhost:8080 for a local installation
  std::map<std::string, std::string> md;
  long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  api.storeAsTFileAny(&dpid2DataDesc, "EMC/Config/DCSDPconfig", md, ts);

  return 0;
}
