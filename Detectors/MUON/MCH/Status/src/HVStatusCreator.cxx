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

#include "MCHStatus/HVStatusCreator.h"

#include "MCHConditions/DetectionElement.h"
#include "MCHGlobalMapping/Mapper.h"
#include "MCHStatus/StatusMap.h"

namespace o2::mch
{

void HVStatusCreator::findBadHVs(const DPMAP& dpMap)
{

}

bool HVStatusCreator::findCurrentBadHVs(uint64_t timestamp)
{
  // list issues at the given time stamp
  std::set<std::string> currentBadHVs{};
  for (const auto& [alias, timeRanges] : mBadHVTimeRanges) {
    auto it = std::find_if(timeRanges.begin(), timeRanges.end(),
                           [timestamp](const TimeRange& timeRange) { return timeRange.contains(timestamp); });
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

void HVStatusCreator::updateStatusMap(StatusMap& statusMap)
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
