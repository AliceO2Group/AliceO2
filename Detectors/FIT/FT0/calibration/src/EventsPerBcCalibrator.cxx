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

#include "FT0Calibration/EventsPerBcCalibrator.h"
#include "CommonUtils/MemFileHelper.h"

namespace o2::ft0
{
void EventsPerBcContainer::print() const
{
  LOG(info) << entries << " entries";
}

void EventsPerBcContainer::fill(const o2::dataformats::TFIDInfo& ti, const gsl::span<const o2::ft0::Digit> data)
{
  size_t oldEntries = entries;
  for (const auto& digit : data) {
    if (digit.mTriggers.getVertex() && digit.mTriggers.getAmplA() >= mMinAmplitudeSideA && digit.mTriggers.getAmplC() >= mMinAmplitudeSideC && (digit.mTriggers.getAmplA() + digit.mTriggers.getAmplC()) >= mMinSumOfAmplitude) {
      mTvx[digit.mIntRecord.bc]++;
      entries++;
    }
  }
  LOG(debug) << "Container is filled with " << entries - oldEntries << " new events";
}

void EventsPerBcContainer::merge(const EventsPerBcContainer* prev)
{
  for (int bc = 0; bc < o2::constants::lhc::LHCMaxBunches; bc++) {
    mTvx[bc] += prev->mTvx[bc];
  }
  entries += prev->entries;
}

void EventsPerBcCalibrator::initOutput()
{
  mTvxPerBcs.clear();
  mTvxPerBcInfos.clear();
}

EventsPerBcCalibrator::EventsPerBcCalibrator(uint32_t minNumberOfEntries, int32_t minAmplitudeSideA, int32_t minAmplitudeSideC, int32_t minSumOfAmplitude) : mMinNumberOfEntries(minNumberOfEntries), mMinAmplitudeSideA(minAmplitudeSideA), mMinAmplitudeSideC(minAmplitudeSideC), mMinSumOfAmplitude(minSumOfAmplitude)
{
  LOG(info) << "Defined threshold for number of entires per slot: " << mMinNumberOfEntries;
  LOG(info) << "Defined threshold for side A amplitude for event: " << mMinAmplitudeSideA;
  LOG(info) << "Defined threshold for side C amplitude for event: " << mMinAmplitudeSideC;
}

bool EventsPerBcCalibrator::hasEnoughData(const EventsPerBcCalibrator::Slot& slot) const
{
  return slot.getContainer()->entries > mMinNumberOfEntries;
}

void EventsPerBcCalibrator::finalizeSlot(EventsPerBcCalibrator::Slot& slot)
{
  LOG(info) << "Finalizing slot from " << slot.getStartTimeMS() << " to " << slot.getEndTimeMS();
  o2::ft0::EventsPerBcContainer* data = slot.getContainer();
  mTvxPerBcs.emplace_back(data->mTvx);

  auto clName = o2::utils::MemFileHelper::getClassName(mTvxPerBcs.back());
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);

  std::map<std::string, std::string> metaData;
  mTvxPerBcInfos.emplace_back(std::make_unique<o2::ccdb::CcdbObjectInfo>("FT0/Calib/EventsPerBc", clName, flName, metaData, slot.getStartTimeMS(), slot.getEndTimeMS()));
  LOG(info) << "Created object valid from " << mTvxPerBcInfos.back()->getStartValidityTimestamp() << " to " << mTvxPerBcInfos.back()->getEndValidityTimestamp();
}

EventsPerBcCalibrator::Slot& EventsPerBcCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<EventsPerBcContainer>(mMinAmplitudeSideA, mMinAmplitudeSideC, mMinSumOfAmplitude));
  return slot;
}
} // namespace o2::ft0