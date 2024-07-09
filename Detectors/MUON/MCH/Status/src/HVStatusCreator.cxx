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

#include <algorithm>
#include <map>

#include "MCHStatus/HVStatusCreator.h"

#include "MCHConditions/DetectionElement.h"
#include "MCHGlobalMapping/Mapper.h"
#include "MCHStatus/StatusMap.h"
#include "MCHStatus/StatusMapCreatorParam.h"

using DPMAP2 = std::unordered_map<std::string, std::map<uint64_t, double>>;

/// Converts DCS data point value to double HV value
double dpConverter(o2::dcs::DataPointValue v)
{
  union Converter {
    uint64_t raw_data;
    double value;
  } converter;
  converter.raw_data = v.payload_pt1;
  return converter.value;
};

/// Decode the DCS DPMAP to be processed for HV issues
DPMAP2 decodeDPMAP(const o2::mch::HVStatusCreator::DPMAP& dpMap)
{
  DPMAP2 dpsMapPerAlias{};

  auto timeMargin = o2::mch::StatusMapCreatorParam::Instance().timeMargin;

  for (const auto& [dpId, dpsHV] : dpMap) {
    std::string alias = dpId.get_alias();

    if (alias.find("vMon") != std::string::npos) {
      auto& dps2 = dpsMapPerAlias[alias];

      // copy first point to the beginning of time + margin (will be subtracted later on)
      dps2.emplace(timeMargin, dpConverter(dpsHV.front()));

      for (const auto& value : dpsHV) {
        dps2.emplace(value.get_epoch_time(), dpConverter(value));
      }

      // copy last point to the end of time - margin (will be added later on)
      dps2.emplace(std::numeric_limits<uint64_t>::max() - timeMargin, dpConverter(dpsHV.back()));
    }
  }

  return dpsMapPerAlias;
}

namespace o2::mch
{

void HVStatusCreator::findBadHVs(const DPMAP& dpMap)
{
  // clear current list of issues
  mBadHVTimeRanges.clear();

  // decode the DCS DPMAP
  DPMAP2 dpsMapPerAlias = decodeDPMAP(dpMap);

  auto minDuration = StatusMapCreatorParam::Instance().hvMinDuration;
  auto timeMargin = StatusMapCreatorParam::Instance().timeMargin;

  // find list of HV issues per alias
  for (const auto& [alias, dpsHV] : dpsMapPerAlias) {
    int chamber = o2::mch::dcs::toInt(o2::mch::dcs::aliasToChamber(alias));
    auto chamberThreshold = StatusMapCreatorParam::Instance().hvLimits[chamber];

    std::vector<TimeRange> hvIssuesList{};
    uint64_t tStart = 0;
    uint64_t tStop = 0;
    bool ongoingIssue = false;

    for (auto [timestamp, valueHV] : dpsHV) {
      if (valueHV < chamberThreshold) {
        if (!ongoingIssue) {
          tStart = timestamp;
          tStop = tStart;
          ongoingIssue = true;
        } else {
          tStop = timestamp;
        }
      } else {
        if (ongoingIssue) {
          tStop = timestamp;
          if (tStop - tStart > minDuration) {
            hvIssuesList.emplace_back(tStart - timeMargin, tStop + timeMargin);
          }
          ongoingIssue = false;
        }
      }
    }

    // ongoing issue at the end of the object
    if (ongoingIssue && tStop - tStart > minDuration) {
      hvIssuesList.emplace_back(tStart - timeMargin, tStop + timeMargin);
    }

    // add issues for the alias if non-empty
    if (!hvIssuesList.empty()) {
      mBadHVTimeRanges.emplace(alias, hvIssuesList);
    }
  }
}

bool HVStatusCreator::findCurrentBadHVs(uint64_t timestamp)
{
  // list issues at the given time stamp
  std::set<std::string> currentBadHVs{};
  for (const auto& [alias, timeRanges] : mBadHVTimeRanges) {
    auto it = std::find_if(timeRanges.begin(), timeRanges.end(),
                           [timestamp](const TimeRange& r) { return r.contains(timestamp); });
    if (it != timeRanges.end()) {
      currentBadHVs.emplace(alias);
    }
  }

  // check if the list of issues has changed and update it in this case
  if (currentBadHVs != mCurrentBadHVs) {
    mCurrentBadHVs.swap(currentBadHVs);
    return true;
  }

  return false;
}

void HVStatusCreator::updateStatusMap(StatusMap& statusMap) const
{
  for (const auto& alias : mCurrentBadHVs) {
    int deId = dcs::aliasToDetElemId(alias).value();
    if (deId < 500) {
      for (auto dsIndex : dcs::aliasToDsIndices(alias)) {
        statusMap.addDS(dsIndex, StatusMap::kBadHV);
      }
    } else {
      statusMap.addDE(deId, StatusMap::kBadHV);
    }
  }
}

} // namespace o2::mch
