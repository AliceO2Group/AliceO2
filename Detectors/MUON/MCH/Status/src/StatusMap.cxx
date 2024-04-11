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

#include "MCHStatus/StatusMap.h"

#include <stdexcept>

#include "Framework/Logger.h"
#include "MCHConstants/DetectionElements.h"
#include "MCHMappingInterface/Segmentation.h"

#include <fmt/format.h>

ClassImp(o2::mch::StatusMap);

namespace o2::mch
{

void assertValidMask(uint32_t mask)
{
  static constexpr uint32_t maxMask = StatusMap::kBadPedestal | StatusMap::kRejectList | StatusMap::kBadHV;
  if (mask > maxMask) {
    throw std::runtime_error(fmt::format("invalid mask {} (max allowed is {}",
                                         mask, maxMask));
  }
}

void StatusMap::add(gsl::span<const DsChannelId> badchannels, uint32_t mask)
{
  assertValidMask(mask);
  for (auto id : badchannels) {
    try {
      ChannelCode cc(id.getSolarId(), id.getElinkId(), id.getChannel());
      mStatus[cc] |= mask;
    } catch (const std::exception& e) {
      LOGP(warning, "Error processing channel - SolarId: {} ElinkId: {} Channel: {}. Error: {}. This channel is skipped.",
           id.getSolarId(), id.getElinkId(), id.getChannel(), e.what());
    }
  }
}

void StatusMap::add(gsl::span<const ChannelCode> badchannels, uint32_t mask)
{
  assertValidMask(mask);
  for (auto id : badchannels) {
    mStatus[id] |= mask;
  }
}

void StatusMap::addDS(DsIndex badDS, uint32_t mask)
{
  if (badDS >= NumberOfDualSampas) {
    LOGP(warning, "Error processing Dual Sampa - index: {}. This DS is skipped.", badDS);
    return;
  }
  addDS(getDsDetId(badDS), mask);
}

void StatusMap::addDS(raw::DsDetId badDS, uint32_t mask)
{
  assertValidMask(mask);
  auto deId = badDS.deId();
  if (!constants::isValidDetElemId(deId)) {
    LOGP(warning, "Error processing Dual Sampa - {}. This DS is skipped.", raw::asString(badDS));
    return;
  }
  const auto& seg = mapping::segmentation(deId);
  seg.forEachPadInDualSampa(badDS.dsId(), [&](int dePadIndex) {
    try {
      ChannelCode cc(deId, dePadIndex);
      mStatus[cc] |= mask;
    } catch (const std::exception& e) {
      LOGP(warning, "Error processing channel - {} padIndex: {}. Error: {}. This channel is skipped.",
           raw::asString(badDS), dePadIndex, e.what());
    }
  });
}

void StatusMap::addDE(uint16_t badDE, uint32_t mask)
{
  assertValidMask(mask);
  if (!constants::isValidDetElemId(badDE)) {
    LOGP(warning, "Error processing DE - Id: {}. This DE is skipped.", badDE);
    return;
  }
  const auto& seg = mapping::segmentation(badDE);
  seg.forEachPad([&](int dePadIndex) {
    try {
      ChannelCode cc(badDE, dePadIndex);
      mStatus[cc] |= mask;
    } catch (const std::exception& e) {
      LOGP(warning, "Error processing channel - deId: {} padIndex: {}. Error: {}. This channel is skipped.",
           badDE, dePadIndex, e.what());
    }
  });
}

uint32_t StatusMap::status(const ChannelCode& id) const
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

  if (mask == 0) {
    return rejectList;
  }

  for (const auto& status : statusMap) {
    auto channel = status.first;
    if (!channel.isValid()) {
      continue;
    }
    if ((mask & status.second) != 0) {
      rejectList[channel.getDeId()].emplace_back(channel.getDePadIndex());
    }
  }
  return rejectList;
}

} // namespace o2::mch
