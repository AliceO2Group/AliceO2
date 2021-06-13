// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FT0Calibration/FT0CalibCollector.h"
#include "Framework/Logger.h"
#include <cassert>
#include <iostream>
#include <sstream>
#include <TStopwatch.h>

namespace o2
{
namespace ft0
{

using Slot = o2::calibration::TimeSlot<o2::ft0::FT0CalibInfoSlot>;

//_____________________________________________
void FT0CalibInfoSlot::fill(const gsl::span<const o2::ft0::FT0CalibrationInfoObject> data)
{
  // fill container
  // we first order the data that arrived, to improve speed when filling
  int nd = data.size();
  LOG(INFO) << "FT0CalibInfoSlot::fill entries in incoming data = " << nd;
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
    auto it = mFT0CollectedCalibInfoSlot.emplace(mFT0CollectedCalibInfoSlot.begin() + offset, data[ord[i]].getChannelIndex(), data[ord[i]].getTime(), data[ord[i]].getAmp());
    mEntriesSlot[ch]++;
  }
}
//_____________________________________________
void FT0CalibInfoSlot::merge(const FT0CalibInfoSlot* prev)
{
  // merge data of 2 slots

  LOG(INFO) << "Merging two slots with entries: current slot -> " << mFT0CollectedCalibInfoSlot.size() << " , previous slot -> " << prev->mFT0CollectedCalibInfoSlot.size();

  int offset = 0, offsetPrev = 0;
  std::vector<o2::ft0::FT0CalibrationInfoObject> tmpVector;
  for (int ch = 0; ch < NCHANNELS; ch++) {
    if (mEntriesSlot[ch] != 0) {
      for (int i = offset; i < offset + mEntriesSlot[ch]; i++) {
        tmpVector.emplace_back(mFT0CollectedCalibInfoSlot[i]);
      }
      offset += mEntriesSlot[ch];
    }
    if (prev->mEntriesSlot[ch] != 0) {
      for (int i = offsetPrev; i < offsetPrev + prev->mEntriesSlot[ch]; i++) {
        tmpVector.emplace_back(prev->mFT0CollectedCalibInfoSlot[i]);
      }
      offsetPrev += prev->mEntriesSlot[ch];
      mEntriesSlot[ch] += prev->mEntriesSlot[ch];
    }
  }
  mFT0CollectedCalibInfoSlot.swap(tmpVector);
  LOG(DEBUG) << "After merging the size is " << mFT0CollectedCalibInfoSlot.size();
  return;
}
//_____________________________________________
void FT0CalibInfoSlot::print() const
{
  // to print number of entries in the tree and the channel with the max number of entries

  LOG(INFO) << "Total number of entries " << mFT0CollectedCalibInfoSlot.size();
  auto maxElementIndex = std::max_element(mEntriesSlot.begin(), mEntriesSlot.end());
  auto channelIndex = std::distance(mEntriesSlot.begin(), maxElementIndex);
  LOG(INFO) << "The maximum number of entries per channel in the current mFT0CollectedCalibInfo is " << *maxElementIndex << " for channel " << channelIndex;
  return;
}

//_____________________________________________
void FT0CalibInfoSlot::printEntries() const
{
  // to print number of entries in the tree and per channel

  LOG(DEBUG) << "Total number of entries " << mFT0CollectedCalibInfoSlot.size();
  for (int i = 0; i < mEntriesSlot.size(); ++i) {
    if (mEntriesSlot[i] != 0) {
      LOG(INFO) << "channel " << i << " has " << mEntriesSlot[i] << " entries";
    }
  }
  return;
}

//===================================================================

//_____________________________________________
void FT0CalibCollector::initOutput()
{
  // emptying the vectors

  mFT0CollectedCalibInfo.clear();
  for (int ch = 0; ch < NCHANNELS; ch++) {
    mEntries[ch] = 0;
  }

  return;
}

//_____________________________________________
bool FT0CalibCollector::hasEnoughData(const Slot& slot) const
{
  // We define that we have enough data if the tree is big enough.
  // each FT0CalibrationInfoObject is composed of two int8 and one int16 --> 32 bytes
  // E.g. supposing that we have 500000 entries per channel  --> 500 eneries per one amplitude bin
  // we can check if we have  500000*o2::ft0::Geometry::NCHANNELS entries in the vector

  if (mTest) {
    return true;
  }
  const o2::ft0::FT0CalibInfoSlot* c = slot.getContainer();
  LOG(INFO) << "we have " << c->getCollectedCalibInfoSlot().size() << " entries";
  int maxNumberOfHits = mAbsMaxNumOfHits ? mMaxNumOfHits : mMaxNumOfHits * NCHANNELS;
  if (mTFsendingPolicy || c->getCollectedCalibInfoSlot().size() > maxNumberOfHits) {
    return true;
  }
  return false;
}

//_____________________________________________
void FT0CalibCollector::finalizeSlot(Slot& slot)
{

  o2::ft0::FT0CalibInfoSlot* c = slot.getContainer();
  mFT0CollectedCalibInfo = c->getCollectedCalibInfoSlot();
  LOG(INFO) << "vector of  received with size = " << mFT0CollectedCalibInfo.size();
  mEntries = c->getEntriesPerChannel();
  return;
}

//_____________________________________________
Slot& FT0CalibCollector::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{

  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<FT0CalibInfoSlot>());
  return slot;
}

} // end namespace ft0
} // end namespace o2
