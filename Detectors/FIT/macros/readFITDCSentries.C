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

/// \file readFITDCSentries.C
/// \brief Macro to read the FIT DCS information from CCDB
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/CcdbApi.h"
#include "DataFormatsFIT/DCSDPValues.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DeliveryType.h"

#include <chrono>
#include <string>
#include <time.h>
#include <unordered_map>

using DPID = o2::dcs::DataPointIdentifier;

void readFITDCSentries(const std::string detectorName = "FT0",
                       long timestamp = 99999999999999,
                       const std::string ccdbUrl = "http://localhost:8080",
                       const bool printEmpty = false)
{
  LOG(info) << "Reading entries for " << detectorName;
  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);

  const std::string ccdbPath = detectorName + "/Calib/DCSDPs";

  std::map<std::string, std::string> metadata;
  if (timestamp == 99999999999999) {
    timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  }

  std::unordered_map<DPID, o2::fit::DCSDPValues>* map = api.retrieveFromTFileAny<std::unordered_map<DPID, o2::fit::DCSDPValues>>(ccdbPath, metadata, timestamp);
  if (map == nullptr) {
    LOGP(fatal, "DCS DPs not found in {}/{} for timestamp {}", ccdbUrl, ccdbPath, timestamp);
    return;
  }

  int nEmptyDPs = 0;

  if (!printEmpty) {
    LOG(info) << "Not printing DPs with no values";
  }

  for (auto& i : *map) {
    if (i.second.empty() && !printEmpty) {
      nEmptyDPs++;
      continue;
    }
    LOG(info) << "DPID = " << i.first;
    i.second.print();
  }

  LOG(info) << "Size of map = " << map->size();
  LOG(info) << "Empty DPs = " << nEmptyDPs;

  return;
}

// AM: WIP

// void readFITDCSDP(const std::string detectorName = "FT0",
//                   const std::string dpAlias = "FT0/HV/FT0_A/MCP_A1/actual/iMon",
//                   long timeStart = -1,
//                   long timeEnd = -1,
//                   const std::string ccdbUrl = "http://localhost:8080")
// {
//   LOG(info) << "Reading values for " << dpAlias;

//   o2::ccdb::CcdbApi api;
//   api.init(ccdbUrl);

//   const std::string ccdbPath = detectorName + "/Calib/DCSDPs";

//   std::map<std::string, std::string> metadata;

//   if (timeStart == -1 && timeEnd == -1) {
//     // Reading values from the last 60 s
//     timeEnd = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
//     timeStart = timeEnd - o2::ccdb::CcdbObjectInfo::MINUTE;
//   } else if (timeStart == -1) {
//     timeStart = timeEnd - o2::ccdb::CcdbObjectInfo::MINUTE;
//   } else if (timeEnd == -1) {
//     timeEnd = timeStart + o2::ccdb::CcdbObjectInfo::MINUTE;
//   }
//   time_t startTimeSeconds = timeStart / 1000;
//   time_t endTimeSeconds = timeEnd / 1000;
//   LOGP(info, "Start time = {}", asctime(localtime(&startTimeSeconds)));
//   LOGP(info, "End time   = {}", asctime(localtime(&endTimeSeconds)));
//   LOGP(info, "Epoch      = {} to {}", timeStart, timeEnd);

//   std::unordered_map<DPID, o2::fit::DCSDPValues>* mapStart = api.retrieveFromTFileAny<std::unordered_map<DPID, o2::fit::DCSDPValues>>(ccdbPath, metadata, timeStart);
//   std::unordered_map<DPID, o2::fit::DCSDPValues>* mapEnd = api.retrieveFromTFileAny<std::unordered_map<DPID, o2::fit::DCSDPValues>>(ccdbPath, metadata, timeEnd);

//   if (mapStart != nullptr) {
//     uint16_t firstTime = -1;
//     uint16_t lastTime = -1;

//     DPID dpid;
//     DPID::FILL(dpid, dpAlias, o2::dcs::DeliveryType::DPVAL_DOUBLE);

//     o2::fit::DCSDPValues dpValuesStart = mapStart->at(dpid);

//     LOG(info) << dpid;

//     time_t timestamp;
//     for (auto& value : dpValuesStart.values) {
//       timestamp = value.first / 1000;
//       LOGP(info, "{}       {}", asctime(localtime(&timestamp)), value.second);
//     }
//   }
// }

#endif
