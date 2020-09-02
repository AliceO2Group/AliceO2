// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TOFCalibration/TOFCalibCollector.h"
#include "Framework/Logger.h"
#include <cassert>
#include <iostream>
#include <sstream>
#include <TStopwatch.h>

namespace o2
{
namespace tof
{

using Slot = o2::calibration::TimeSlot<o2::tof::TOFCalibInfoSlot>;

//_____________________________________________
void TOFCalibInfoSlot::fill(const gsl::span<const o2::dataformats::CalibInfoTOF> data)
{
  // fill container
  // we do not apply any calibration at this stage, it will be applied when we
  // process the data before filling the CCDB in the separate process

  // we first order the data that arrived, to improve speed when filling
  int nd = data.size();
  LOG(DEBUG) << "entries in incoming data = " << nd;
  std::vector<int> ord(nd);
  std::iota(ord.begin(), ord.end(), 0);
  std::sort(ord.begin(), ord.end(), [&data](int i, int j) { return data[i].getTOFChIndex() < data[j].getTOFChIndex(); });
  int chPrev = 0, offsPrev = 0;
  for (int i = 0; i < nd; i++) {
    const auto& dti = data[ord[i]];
    auto ch = dti.getTOFChIndex();
    auto offset = offsPrev;
    if (ch > chPrev) {
      offset += std::accumulate(mEntriesSlot.begin() + chPrev, mEntriesSlot.begin() + ch, 0);
    }
    offsPrev = offset;
    chPrev = ch;
    mTOFCollectedCalibInfoSlot.emplace(mTOFCollectedCalibInfoSlot.begin() + offset, data[ord[i]].getTimestamp(), data[ord[i]].getDeltaTimePi(), data[ord[i]].getTot(), data[ord[i]].getFlags());
    mEntriesSlot[ch]++;
  }
}
//_____________________________________________
void TOFCalibInfoSlot::merge(const TOFCalibInfoSlot* prev)
{
  // merge data of 2 slots

  LOG(DEBUG) << "Merging two slots with entries: current slot -> " << mTOFCollectedCalibInfoSlot.size() << " , previous slot -> " << prev->mTOFCollectedCalibInfoSlot.size();

  int offset = 0, offsetPrev = 0;
  std::vector<o2::dataformats::CalibInfoTOFshort> tmpVector;
  for (int ch = 0; ch < Geo::NCHANNELS; ch++) {
    if (mEntriesSlot[ch] != 0) {
      for (int i = offset; i < offset + mEntriesSlot[ch]; i++) {
        tmpVector.emplace_back(mTOFCollectedCalibInfoSlot[i]);
      }
      offset += mEntriesSlot[ch];
    }
    if (prev->mEntriesSlot[ch] != 0) {
      for (int i = offsetPrev; i < offsetPrev + prev->mEntriesSlot[ch]; i++) {
        tmpVector.emplace_back(prev->mTOFCollectedCalibInfoSlot[i]);
      }
      offsetPrev += prev->mEntriesSlot[ch];
      mEntriesSlot[ch] += prev->mEntriesSlot[ch];
    }
  }
  mTOFCollectedCalibInfoSlot.swap(tmpVector);
  LOG(DEBUG) << "After merging the size is " << mTOFCollectedCalibInfoSlot.size();
  return;
}
//_____________________________________________
void TOFCalibInfoSlot::print() const
{
  // to print number of entries in the tree and the channel with the max number of entries

  LOG(INFO) << "Total number of entries " << mTOFCollectedCalibInfoSlot.size();
  auto maxElementIndex = std::max_element(mEntriesSlot.begin(), mEntriesSlot.end());
  auto channelIndex = std::distance(mEntriesSlot.begin(), maxElementIndex);
  LOG(INFO) << "The maximum number of entries per channel in the current mTOFCollectedCalibInfo is " << *maxElementIndex << " for channel " << channelIndex;
  return;
}

//_____________________________________________
void TOFCalibInfoSlot::printEntries() const
{
  // to print number of entries in the tree and per channel

  LOG(INFO) << "Total number of entries " << mTOFCollectedCalibInfoSlot.size();
  for (int i = 0; i < mEntriesSlot.size(); ++i) {
    if (mEntriesSlot[i] != 0) {
      LOG(INFO) << "channel " << i << " has " << mEntriesSlot[i] << " entries";
    }
  }
  return;
}

//===================================================================

//_____________________________________________
void TOFCalibCollector::initOutput()
{
  // emptying the vectors

  mTOFCollectedCalibInfo.clear();
  for (int ch = 0; ch < Geo::NCHANNELS; ch++) {
    mEntries[ch] = 0;
  }

  return;
}

//_____________________________________________
bool TOFCalibCollector::hasEnoughData(const Slot& slot) const
{

  // We define that we have enough data if the tree is big enough.
  // each CalibInfoTOFShort is composed of one int, two floats, one unsigned char --> 13 bytes
  // E.g. supposing that we have 256 entries per channel (which is an upper limit ) --> ~523 MB
  // we can check if we have at least 1 GB of data --> 500*o2::tof::Geo::NCHANNELS entries in the vector
  // (see header file for the fact that mMaxNumOfHits = 500)
  // The case in which mScaleMaxNumOfHits = false allows for a fast check

  if (mTest)
    return true;
  const o2::tof::TOFCalibInfoSlot* c = slot.getContainer();
  LOG(INFO) << "we have " << c->getCollectedCalibInfoSlot().size() << " entries";
  int maxNumberOfHits = mAbsMaxNumOfHits ? mMaxNumOfHits : mMaxNumOfHits * o2::tof::Geo::NCHANNELS;
  if (mTFsendingPolicy || c->getCollectedCalibInfoSlot().size() > maxNumberOfHits)
    return true;
  return false;
}

//_____________________________________________
void TOFCalibCollector::finalizeSlot(Slot& slot)
{
  // here we fill the tree with the remaining stuff that was not filled before

  o2::tof::TOFCalibInfoSlot* c = slot.getContainer();
  mTOFCollectedCalibInfo = c->getCollectedCalibInfoSlot();
  LOG(DEBUG) << "vector of CalibTOFInfoShort received with size = " << mTOFCollectedCalibInfo.size();
  mEntries = c->getEntriesPerChannel();
  return;
}

//_____________________________________________
Slot& TOFCalibCollector::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{

  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<TOFCalibInfoSlot>());
  return slot;
}

} // end namespace tof
} // end namespace o2
