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

#include "MCHConditions/StatusMap.h"

#include "Framework/Logger.h"
#include "MCHMappingInterface/Segmentation.h"
#include "MCHRawElecMap/DsDetId.h"
#include "MCHRawElecMap/Mapper.h"

ClassImp(o2::mch::StatusMap);

namespace o2::mch
{

void StatusMap::add(gsl::span<const DsChannelId> badchannels, uint32_t mask)
{
  for (auto id : badchannels) {
    mStatus[raw::convert(id)] |= mask;
  }
}

void StatusMap::add(gsl::span<const DsChannelDetId> badchannels, uint32_t mask)
{
  for (auto id : badchannels) {
    mStatus[id] |= mask;
  }
}

uint32_t StatusMap::status(const DsChannelDetId& id) const
{
  auto s = mStatus.find(id);
  if (s != mStatus.end()) {
    return s->second;
  }
  return 0;
}

std::map<int, std::vector<int>> applyMask(const o2::mch::StatusMap& statusMap,
                                          uint32_t mask)
{
  std::map<int, std::vector<int>> rejectList;

  std::unique_ptr<o2::mch::mapping::Segmentation> seg{nullptr};
  int previousDeId{-1};

  for (const auto& status : statusMap) {
    auto channelId = status.first;
    auto deId = channelId.getDeId();
    if (deId != previousDeId) {
      previousDeId = deId;
      seg = std::make_unique<o2::mch::mapping::Segmentation>(deId);
    }
    auto dsId = channelId.getDsId();
    auto channel = channelId.getChannel();
    auto padId = seg->findPadByFEE(dsId, channel);
    if (seg->isValid(padId)) {
      if (mask && (mask & status.second)) {
        rejectList[deId].emplace_back(padId);
      }
    } else {
      LOGP(warning, "Got an invalid pad in bad channel map DE {} DS {} CH {} !",
           deId, dsId, channel);
    }
  }
  return rejectList;
}

} // namespace o2::mch
