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

#include "FV0Calibration/FV0CalibCollector.h"
#include "Framework/Logger.h"
#include <cassert>
#include <iostream>
#include <sstream>
#include <TStopwatch.h>

namespace o2
{
namespace fv0
{

using Slot = o2::calibration::TimeSlot<o2::fv0::FV0CalibInfoSlot>;

//_____________________________________________
void FV0CalibInfoSlot::fill(const gsl::span<const o2::fv0::FV0CalibrationInfoObject> data)
{
  // fill container
  // we first order the data that arrived, to improve speed when filling
  int nd = data.size();
  LOG(info) << "FV0CalibInfoSlot::fill entries in incoming data = " << nd;
  std::vector<int> ord(nd);
  std::iota(ord.begin(), ord.end(), 0);
  std::sort(ord.begin(), ord.end(), [&data](int i, int j) { return data[i].getChannelIndex() < data[j].getChannelIndex(); });
  int chPrev = 0, offsPrev = 0;
  for (int i = 0; i < nd; i++) {
    if (std::abs(data[ord[i]].getTime()) > HISTO_RANGE) {
      continue;
    }
    const auto& dti = data[ord[i]];
    auto ch = dti.getChannelIndex();
    auto offset = offsPrev;
    if (ch > chPrev) {
      offset += std::accumulate(mEntriesSlot.begin() + chPrev, mEntriesSlot.begin() + ch, 0);
    }
    offsPrev = offset;
    chPrev = ch;
    auto it = mFV0CollectedCalibInfoSlot.emplace(mFV0CollectedCalibInfoSlot.begin() + offset, data[ord[i]].getChannelIndex(), data[ord[i]].getTime(), data[ord[i]].getCharge());
    mEntriesSlot[ch]++;
  }
}
//_____________________________________________
void FV0CalibInfoSlot::merge(const FV0CalibInfoSlot* prev)
{
  // merge data of 2 slots

  LOG(info) << "Merging two slots with entries: current slot -> " << mFV0CollectedCalibInfoSlot.size() << " , previous slot -> " << prev->mFV0CollectedCalibInfoSlot.size();

  int offset = 0, offsetPrev = 0;
  std::vector<o2::fv0::FV0CalibrationInfoObject> tmpVector;
  for (int ch = 0; ch < NCHANNELS; ch++) {
    if (mEntriesSlot[ch] != 0) {
      for (int i = offset; i < offset + mEntriesSlot[ch]; i++) {
        tmpVector.emplace_back(mFV0CollectedCalibInfoSlot[i]);
      }
      offset += mEntriesSlot[ch];
    }
    if (prev->mEntriesSlot[ch] != 0) {
      for (int i = offsetPrev; i < offsetPrev + prev->mEntriesSlot[ch]; i++) {
        tmpVector.emplace_back(prev->mFV0CollectedCalibInfoSlot[i]);
      }
      offsetPrev += prev->mEntriesSlot[ch];
      mEntriesSlot[ch] += prev->mEntriesSlot[ch];
    }
  }
  mFV0CollectedCalibInfoSlot.swap(tmpVector);
  LOG(debug) << "After merging the size is " << mFV0CollectedCalibInfoSlot.size();
  return;
}
//_____________________________________________
void FV0CalibInfoSlot::print() const
{
  // to print number of entries in the tree and the channel with the max number of entries

  LOG(info) << "Total number of entries " << mFV0CollectedCalibInfoSlot.size();
  auto maxElementIndex = std::max_element(mEntriesSlot.begin(), mEntriesSlot.end());
  auto channelIndex = std::distance(mEntriesSlot.begin(), maxElementIndex);
  LOG(info) << "The maximum number of entries per channel in the current mFV0CollectedCalibInfo is " << *maxElementIndex << " for channel " << channelIndex;
  return;
}

//_____________________________________________
void FV0CalibInfoSlot::printEntries() const
{
  // to print number of entries in the tree and per channel

  LOG(debug) << "Total number of entries " << mFV0CollectedCalibInfoSlot.size();
  for (int i = 0; i < mEntriesSlot.size(); ++i) {
    if (mEntriesSlot[i] != 0) {
      LOG(info) << "channel " << i << " has " << mEntriesSlot[i] << " entries";
    }
  }
  return;
}

//===================================================================

//_____________________________________________
void FV0CalibCollector::initOutput()
{
  // emptying the vectors

  mFV0CollectedCalibInfo.clear();
  for (int ch = 0; ch < NCHANNELS; ch++) {
    mEntries[ch] = 0;
  }

  return;
}

//_____________________________________________
bool FV0CalibCollector::hasEnoughData(const Slot& slot) const
{
  // We define that we have enough data if the tree is big enough.
  // each FV0CalibrationInfoObject is composed of two int8 and one int16 --> 32 bytes
  // E.g. supposing that we have 500000 entries per channel  --> 500 eneries per one amplitude bin
  // we can check if we have  500000*Constants::nFv0Channels entries in the vector

  if (mTest) {
    return true;
  }
  const o2::fv0::FV0CalibInfoSlot* c = slot.getContainer();
  LOG(info) << "we have " << c->getCollectedCalibInfoSlot().size() << " entries";
  int maxNumberOfHits = mAbsMaxNumOfHits ? mMaxNumOfHits : mMaxNumOfHits * NCHANNELS;
  if (mTFsendingPolicy || c->getCollectedCalibInfoSlot().size() > maxNumberOfHits) {
    return true;
  }
  return false;
}

//_____________________________________________
void FV0CalibCollector::finalizeSlot(Slot& slot)
{

  o2::fv0::FV0CalibInfoSlot* c = slot.getContainer();
  mFV0CollectedCalibInfo = c->getCollectedCalibInfoSlot();
  LOG(info) << "vector of  received with size = " << mFV0CollectedCalibInfo.size();
  mEntries = c->getEntriesPerChannel();
  return;
}

//_____________________________________________
Slot& FV0CalibCollector::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{

  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<FV0CalibInfoSlot>());
  return slot;
}

} // end namespace fv0
} // end namespace o2
