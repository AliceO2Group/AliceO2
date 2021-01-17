// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

int main(int argc, char* argv[]) //const std::string url = "http://localhost:8080")
{

  std::string url(argv[0]);
  // macro to populate CCDB for TOF with the configuration for DCS
  std::unordered_map<DPID, std::string> dpid2DataDesc;
  std::vector<std::string> aliases = {"tof_hv_vp_[00..89]", "tof_hv_vn_[00..89]", "tof_hv_ip_[00..89]", "tof_hv_in_[00..89]"};
  std::vector<std::string> aliasesInt = {"TOF_FEACSTATUS_[00..71]", "TOF_HVSTATUS_SM[00..01]MOD[0..1]"};
  std::vector<std::string> expaliases = o2::dcs::expandAliases(aliases);
  std::vector<std::string> expaliasesInt = o2::dcs::expandAliases(aliasesInt);

  DPID dpidtmp;
  for (size_t i = 0; i < expaliases.size(); ++i) {
    DPID::FILL(dpidtmp, expaliases[i], o2::dcs::DeliveryType::RAW_DOUBLE);
    dpid2DataDesc[dpidtmp] = "TOFDATAPOINTS";
  }
  for (size_t i = 0; i < expaliasesInt.size(); ++i) {
    DPID::FILL(dpidtmp, expaliasesInt[i], o2::dcs::DeliveryType::RAW_INT);
    dpid2DataDesc[dpidtmp] = "TOFDATAPOINTS";
  }

  o2::ccdb::CcdbApi api;
  api.init(url); // or http://localhost:8080 for a local installation
  std::map<std::string, std::string> md;
  long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  api.storeAsTFileAny(&dpid2DataDesc, "TOF/DCSconfig", md, ts);

  return 0;
}
