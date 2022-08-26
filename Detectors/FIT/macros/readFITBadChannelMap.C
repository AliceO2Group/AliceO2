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

/// \file readFITBadChannelMap.C
/// \brief Macro to read the FIT bad channel maps from CCDB
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DataFormatsFIT/BadChannelMap.h"

#include <chrono>
#include <string>
#include <time.h>
#include <vector>
#include <iterator>

void readFITBadChannelMap(const std::string& detectorName = "FT0",
                          long timestamp = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP,
                          const std::string& ccdbUrl = "http://localhost:8080",
                          const bool verbose = false)
{
  LOG(info) << "Reading bad channel map for " << detectorName;
  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);

  const std::string ccdbPath = detectorName + "/Calib/BadChannelMap";

  std::map<std::string, std::string> metadata;
  if (timestamp == o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP) {
    timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  }

  o2::fit::BadChannelMap* map = api.retrieveFromTFileAny<o2::fit::BadChannelMap>(ccdbPath, metadata, timestamp);
  if (map == nullptr) {
    LOGP(fatal, "Bad channel map not found in {}/{} for timestamp {}", ccdbUrl, ccdbPath, timestamp);
    return;
  }

  std::vector<uint8_t> goodChannels;
  std::vector<uint8_t> badChannels;

  uint8_t chId;
  for (auto& i : map->map) {
    chId = i.first;
    if (verbose) {
      LOGP(info, "Channel {}: {}", chId, i.second);
    }
    if (map->isChannelGood(chId)) {
      goodChannels.push_back(chId);
    } else {
      badChannels.push_back(chId);
    }
  }

  LOG(info) << "Good channels: ";
  for (auto& ch : goodChannels) {
    LOGP(info, "{}", ch);
  }
  LOG(info) << "Bad channels: ";
  for (auto& ch : badChannels) {
    LOGP(info, "{}", ch);
  }

  LOG(info) << "Number of channels = " << map->map.size();
  return;
}

#endif