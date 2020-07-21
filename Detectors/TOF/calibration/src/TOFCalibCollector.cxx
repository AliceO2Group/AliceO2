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

  for (int i = data.size(); i--;) {
    auto ch = data[i].getTOFChIndex();
    auto offset = std::accumulate(mEntriesSlot.begin(), mEntriesSlot.begin() + ch, 0);
    mTOFCollectedCalibInfoSlot.emplace(mTOFCollectedCalibInfoSlot.begin() + offset, data[i].getTimestamp(), data[i].getDeltaTimePi(), data[i].getTot(), data[i].getFlags());
    mEntriesSlot[ch]++;
  }
}

//_____________________________________________
void TOFCalibInfoSlot::merge(const TOFCalibInfoSlot* prev)
{
  // merge data of 2 slots

  int addedPerChannel = 0;
  int offset = 0;

  LOG(DEBUG) << "Merging two slots with entries: current slot -> " << mTOFCollectedCalibInfoSlot.size() << " , previous slot -> " << prev->mTOFCollectedCalibInfoSlot.size();
  for (int ch = 0; ch < Geo::NCHANNELS; ch++) {
    offset += mEntriesSlot[ch];
    if (prev->mEntriesSlot[ch] == 0)
      continue;
    auto offsetprevStart = addedPerChannel;
    auto offsetprevEnd = addedPerChannel + prev->mEntriesSlot[ch];
    addedPerChannel += prev->mEntriesSlot[ch];
    mTOFCollectedCalibInfoSlot.insert(mTOFCollectedCalibInfoSlot.begin() + offset, prev->mTOFCollectedCalibInfoSlot.begin() + offsetprevStart, prev->mTOFCollectedCalibInfoSlot.begin() + offsetprevEnd);
    mEntriesSlot[ch] += prev->mEntriesSlot[ch];
  }
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
  LOG(DEBUG) << "we have " << c->getCollectedCalibInfoSlot().size() << " entries";
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
