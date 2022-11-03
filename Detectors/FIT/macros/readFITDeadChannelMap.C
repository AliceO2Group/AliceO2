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

/// \file readFITDeadChannelMap.C
/// \brief Macro to read the FIT dead channel maps from CCDB
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "CCDB/CcdbApi.h"
#include "DataFormatsFIT/DeadChannelMap.h"

#include <algortihm>
#include <chrono>
#include <string>
#include <time.h>
#include <vector>

#endif

#include "Framework/Logger.h"

#include <boost/algorithm/string.hpp>

void readFITDeadChannelMap(std::string detectorName = "FT0",
                           long timestamp = -1,
                           const std::string& ccdbUrl = "https://alice-ccdb.cern.ch",
                           const bool verbose = false)
{
  // Parse and check detector name
  boost::to_upper(detectorName);
  if (detectorName != "FT0" && detectorName != "FV0" && detectorName != "FDD") {
    LOGP(fatal, "Invalid detector name provided: '{}'. Please use [FT0/FV0/FDD].", detectorName);
    return;
  }
  LOGP(info, "Fetching dead channel map for {}.", detectorName);

  // Init CCDB stuff
  o2::ccdb::CcdbApi ccdbApi;
  ccdbApi.init(ccdbUrl);
  const std::string ccdbPath = detectorName + "/Calib/DeadChannelMap";
  std::map<std::string, std::string> metadata;

  if (timestamp < 0) {
    timestamp = o2::ccdb::getCurrentTimestamp();
  }

  o2::fit::DeadChannelMap* map = ccdbApi.retrieveFromTFileAny<o2::fit::DeadChannelMap>(ccdbPath, metadata, timestamp);
  if (!map) {
    LOGP(fatal, "Dead channel map not found in {}/{} for timestamp {}.", ccdbUrl, ccdbPath, timestamp);
    return;
  }

  std::vector<uint8_t> aliveChannels;
  std::vector<uint8_t> deadChannels;

  uint8_t chId;
  for (auto& i : map->map) {
    chId = i.first;
    if (verbose) {
      LOGP(info, "Channel {}: {}", chId, i.second);
    }
    if (map->isChannelAlive(chId)) {
      aliveChannels.push_back(chId);
    } else {
      deadChannels.push_back(chId);
    }
  }

  std::sort(aliveChannels.begin(), aliveChannels.end());
  std::sort(deadChannels.begin(), deadChannels.end());

  LOG(info) << "Alive channels: ";
  for (auto& ch : aliveChannels) {
    LOGP(info, "{}", ch);
  }
  LOG(info) << "Dead channels: ";
  for (auto& ch : deadChannels) {
    LOGP(info, "{}", ch);
  }

  LOGP(info, "Number of channels      = {}", map->map.size());
  LOGP(info, "Number of alive channels = {}", aliveChannels.size());
  LOGP(info, "Number of dead channels  = {}", deadChannels.size());

  return;
}
