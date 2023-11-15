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
#include "CCDB/BasicCCDBManager.h"
#include "Framework/Logger.h"
#include <vector>
#endif
#include "CommonConstants/LHCConstants.h"

// Snippet to get ms-timestamp for given run / orbit

long getTimeStamp(int runNumber, uint32_t orbit, const std::string& ccdbHost = "http://alice-ccdb.cern.ch")
{
  static int prevRunNumber = -1;
  static int64_t orbitResetMUS = 0;
  if (runNumber != prevRunNumber) {
    auto& cm = o2::ccdb::BasicCCDBManager::instance();
    if (!ccdbHost.empty()) {
      cm.setURL(ccdbHost);
    }
    auto lims = cm.getRunDuration(runNumber);
    if (lims.first == 0 && lims.second == 0) {
      LOGP(info, "Failed to fetch run {} info from RCT", runNumber);
      return -1;
    }
    auto* orbitReset = cm.getForTimeStamp<std::vector<Long64_t>>("CTP/Calib/OrbitReset", lims.first);
    orbitResetMUS = (*orbitReset)[0];
  }
  return std::ceil((orbit * o2::constants::lhc::LHCOrbitNS / 1000 + orbitResetMUS) / 1000);
}
