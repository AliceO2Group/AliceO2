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

int makeITSCCDBEntryForDCS(std::string ccdb_path = o2::base::NameConf::getCCDBServer())
{

  std::unordered_map<DPID, std::string> dpid2DataDesc;
  std::vector<std::string> aliases;

  // fill aliases
  int nStaves[] = {12, 16, 20, 24, 30, 42, 48};
  for (int iL = 0; iL < 7; iL++) {
    for (int iS = 0; iS < nStaves[iL]; iS++) {
      std::string stv = iS > 9 ? std::to_string(iS) : std::string(1, '0').append(std::to_string(iS));
      aliases.push_back("ITS_L" + std::to_string(iL) + "_" + stv + "_STROBE");
    }
  }

  std::vector<std::string> expaliases = o2::dcs::expandAliases(aliases);

  DPID dpidtmp;
  for (size_t i = 0; i < expaliases.size(); ++i) {
    DPID::FILL(dpidtmp, expaliases[i], o2::dcs::DeliveryType::DPVAL_INT);
    dpid2DataDesc[dpidtmp] = "ITSDATAPOINTS";
  }

  o2::ccdb::CcdbApi api;
  api.init(ccdb_path);
  std::map<std::string, std::string> md;
  md["comment"] = "uploaded with O2 makeITSCCDBEntryForDCS.C";
  long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  api.storeAsTFileAny(&dpid2DataDesc, "ITS/Config/DCSDPconfig", md, ts, ts + 365L * 10 * 24 * 3600 * 1000); // validity is 10 years

  return 0;
}
