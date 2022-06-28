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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CommonUtils/NameConf.h"
#endif
#include <vector>
#include <string>
#include "TFile.h"
#include "DetectorsDCS/AliasExpander.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include <unordered_map>
#include <chrono>
#include "CCDB/CcdbApi.h"

using DPID = o2::dcs::DataPointIdentifier;

int makeMFTCCDBEntryForDCS(std::string ccdb_path = o2::base::NameConf::getCCDBServer())
{

  std::unordered_map<DPID, std::string> dpid2DataDesc;
  std::vector<std::string> aliases = {"MFT_PSU_ZONE/H[0..1]/D[0..4]/F[0..1]/Z[0..3]/Current/Analog",
                                      "MFT_PSU_ZONE/H[0..1]/D[0..4]/F[0..1]/Z[0..3]/Current/BackBias",
                                      "MFT_PSU_ZONE/H[0..1]/D[0..4]/F[0..1]/Z[0..3]/Current/Digital",
                                      "MFT_PSU_ZONE/H[0..1]/D[0..4]/F[0..1]/Z[0..3]/Voltage/BackBias",
                                      "MFT_RU_LV/H0/D0/F0/Z[0..3]/iMon",
                                      "MFT_RU_LV/H0/D1/F0/Z[0..3]/iMon",
                                      "MFT_RU_LV/H0/D2/F0/Z[0..3]/iMon",
                                      "MFT_RU_LV/H0/D3/F0/Z[0..3]/iMon",
                                      "MFT_RU_LV/H0/D4/F0/Z[0..3]/iMon",
                                      "MFT_RU_LV/H0/D0/F1/Z[0..3]/iMon",
                                      "MFT_RU_LV/H0/D1/F1/Z[0..3]/iMon",
                                      "MFT_RU_LV/H0/D2/F1/Z[0..3]/iMon",
                                      "MFT_RU_LV/H0/D3/F1/Z[0..3]/iMon",
                                      "MFT_RU_LV/H0/D4/F1/Z[0..3]/iMon",
                                      "MFT_RU_LV/H1/D0/F0/Z[0..3]/iMon",
                                      "MFT_RU_LV/H1/D1/F0/Z[0..3]/iMon",
                                      "MFT_RU_LV/H1/D2/F0/Z[0..3]/iMon",
                                      "MFT_RU_LV/H1/D3/F0/Z[0..3]/iMon",
                                      "MFT_RU_LV/H1/D4/F0/Z[0..3]/iMon",
                                      "MFT_RU_LV/H1/D0/F1/Z[0..3]/iMon",
                                      "MFT_RU_LV/H1/D1/F1/Z[0..3]/iMon",
                                      "MFT_RU_LV/H1/D2/F1/Z[0..3]/iMon",
                                      "MFT_RU_LV/H1/D3/F1/Z[0..3]/iMon",
                                      "MFT_RU_LV/H1/D4/F1/Z[0..3]/iMon"};

  std::vector<std::string> expaliases = o2::dcs::expandAliases(aliases);

  DPID dpidtmp;
  for (size_t i = 0; i < expaliases.size(); ++i) {
    DPID::FILL(dpidtmp, expaliases[i], o2::dcs::DeliveryType::DPVAL_DOUBLE);
    dpid2DataDesc[dpidtmp] = "MFTDATAPOINTS";
  }

  o2::ccdb::CcdbApi api;
  api.init(ccdb_path);
  std::map<std::string, std::string> md;
  long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  api.storeAsTFileAny(&dpid2DataDesc, "MFT/Config/DCSDPconfig", md, ts);

  return 0;
}
