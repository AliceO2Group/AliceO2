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
#include "Framework/TimesliceIndex.h"

namespace o2::framework
{

TimesliceIndex::TimesliceIndex(size_t maxLanes, size_t maxChannels)
  : mMaxLanes{maxLanes}
{
  mOldestPossibleTimeslices.resize(maxChannels);
}

void TimesliceIndex::resize(size_t s)
{
  mVariables.resize(s);
  mPublishedVariables.resize(s);
  mDirty.resize(s, false);
}

void TimesliceIndex::associate(TimesliceId timestamp, TimesliceSlot slot)
{
  assert(mVariables.size() > slot.index);
  mVariables[slot.index].put({0, static_cast<uint64_t>(timestamp.value)});
  mVariables[slot.index].commit();
  mDirty[slot.index] = true;
}

TimesliceSlot TimesliceIndex::findOldestSlot(TimesliceId timestamp) const
{
  size_t lane = timestamp.value % mMaxLanes;
  TimesliceSlot oldest{lane};
  auto oldPVal = std::get_if<uint64_t>(&mVariables[oldest.index].get(0));
  if (oldPVal == nullptr) {
    return oldest;
  }
  uint64_t oldTimestamp = *oldPVal;

  for (size_t i = lane + mMaxLanes; i < mVariables.size(); i += mMaxLanes) {
    auto newPVal = std::get_if<uint64_t>(&mVariables[i].get(0));
    if (newPVal == nullptr) {
      return TimesliceSlot{i};
    }
    uint64_t newTimestamp = *newPVal;

    if (oldTimestamp > newTimestamp) {
      oldest = TimesliceSlot{i};
      oldTimestamp = newTimestamp;
    }
  }
  return oldest;
}

std::tuple<TimesliceIndex::ActionTaken, TimesliceSlot> TimesliceIndex::replaceLRUWith(data_matcher::VariableContext& newContext, TimesliceId timestamp)
{
  auto oldestSlot = findOldestSlot(timestamp);
  if (TimesliceIndex::isValid(oldestSlot) == false) {
    mVariables[oldestSlot.index] = newContext;
    return std::make_tuple(ActionTaken::ReplaceUnused, oldestSlot);
  }
  auto oldTimestamp = std::get_if<uint64_t>(&mVariables[oldestSlot.index].get(0));
  if (oldTimestamp == nullptr) {
    mVariables[oldestSlot.index] = newContext;
    return std::make_tuple(ActionTaken::ReplaceUnused, oldestSlot);
  }

  auto newTimestamp = std::get_if<uint64_t>(&newContext.get(0));
  if (newTimestamp == nullptr) {
    return std::make_tuple(ActionTaken::DropInvalid, TimesliceSlot{TimesliceSlot::INVALID});
  }

  if (*newTimestamp > *oldTimestamp) {
    switch (mBackpressurePolicy) {
      case BackpressureOp::DropAncient:
        mVariables[oldestSlot.index] = newContext;
        return std::make_tuple(ActionTaken::ReplaceObsolete, oldestSlot);
      case BackpressureOp::DropRecent:
        return std::make_tuple(ActionTaken::DropObsolete, TimesliceSlot{TimesliceSlot::INVALID});
      case BackpressureOp::Wait:
        return std::make_tuple(ActionTaken::Wait, TimesliceSlot{TimesliceSlot::INVALID});
    }
  } else {
    switch (mBackpressurePolicy) {
      case BackpressureOp::DropRecent:
        mVariables[oldestSlot.index] = newContext;
        return std::make_tuple(ActionTaken::ReplaceObsolete, oldestSlot);
      case BackpressureOp::DropAncient:
        return std::make_tuple(ActionTaken::DropObsolete, TimesliceSlot{TimesliceSlot::INVALID});
      case BackpressureOp::Wait:
        return std::make_tuple(ActionTaken::Wait, TimesliceSlot{TimesliceSlot::INVALID});
    }
  }
  O2_BUILTIN_UNREACHABLE();
}

TimesliceIndex::OldestInputInfo TimesliceIndex::setOldestPossibleInput(TimesliceId timestamp, ChannelIndex channel)
{
  // Each channel oldest possible input must be monotoically increasing.
  assert(mOldestPossibleTimeslices[channel.value].value <= timestamp.value);
  mOldestPossibleTimeslices[channel.value] = timestamp;
  OldestInputInfo result{timestamp, channel};
  for (int ci = 0; ci < mOldestPossibleTimeslices.size(); ci++) {
    auto& a = mOldestPossibleTimeslices[ci];
    if (a.value < result.timeslice.value) {
      result = {a, ChannelIndex{ci}};
    }
  }
  mOldestPossibleInput = result;
  return mOldestPossibleInput;
}

bool TimesliceIndex::validateSlot(TimesliceSlot slot, TimesliceId currentOldest)
{
  if (mDirty[slot.index] == true) {
    return true;
  }

  auto timestamp = std::get_if<uint64_t>(&mVariables[slot.index].get(0));
  if (timestamp != nullptr && *timestamp < mOldestPossibleInput.timeslice.value) {
    markAsInvalid(slot);
    return false;
  }
  return true;
}

TimesliceIndex::OldestOutputInfo TimesliceIndex::updateOldestPossibleOutput()
{
  auto oldestInput = getOldestPossibleInput();
  OldestOutputInfo result{oldestInput.timeslice, oldestInput.channel};

  for (size_t i = 0; i < mVariables.size(); i++) {
    // We do not check invalid slots.
    if (isValid(TimesliceSlot{i}) == false) {
      continue;
    }
    auto timestamp = std::get_if<uint64_t>(&mVariables[i].get(0));
    if (timestamp != nullptr && *timestamp < result.timeslice.value) {
      result.timeslice = TimesliceId{*timestamp};
      result.slot = {i};
      result.channel = {(int)-1};
    }
  }
  mOldestPossibleOutput = result;
  return result;
}

} // namespace o2::framework
