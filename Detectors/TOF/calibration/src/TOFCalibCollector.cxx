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

#include "TOFCalibration/TOFCalibCollector.h"
#include "Framework/Logger.h"
#include <cassert>
#include <iostream>
#include <sstream>
#include <numeric>
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
  LOG(info) << "entries in incoming data = " << nd;
  for (int i = 0; i < nd; i++) {
    auto flags = data[i].getFlags();
    if (flags & o2::dataformats::CalibInfoTOF::kMultiHit) { // skip multi-hit clusters
      continue;
    }
    if (flags & o2::dataformats::CalibInfoTOF::kNoBC) { // skip events far from Int BC
      continue;
    }

    mTOFCollectedCalibInfoSlot.emplace_back(data[i].getTOFChIndex(), data[i].getTimestamp(), data[i].getDeltaTimePi() - mLHCphase, data[i].getTot(), data[i].getFlags());
    mEntriesSlot[data[i].getTOFChIndex()]++;
  }
}
//_____________________________________________
void TOFCalibInfoSlot::fill(const gsl::span<const o2::tof::CalibInfoCluster> data)
{
  // fill container
  // we do not apply any calibration at this stage, it will be applied when we
  // process the data before filling the CCDB in the separate process

  // we first order the data that arrived, to improve speed when filling
  int nd = data.size();
  LOG(debug) << "entries in incoming data = " << nd;
  std::vector<int> ord(nd);
  std::iota(ord.begin(), ord.end(), 0);
  std::sort(ord.begin(), ord.end(), [&data](int i, int j) { return data[i].getCH() < data[j].getCH(); });
  int chPrev = 0, offsPrev = 0;
  for (int i = 0; i < nd; i++) {
    const auto& dti = data[ord[i]];
    auto ch = dti.getCH();
    auto dch = dti.getDCH();
    auto dt = dti.getDT();
    auto tot1 = dti.getTOT1();
    auto tot2 = dti.getTOT2();

    // we order them so that the channel number of the first cluster is smaller than
    // the one of the second cluster
    if (dch < 0) {
      ch += dch;
      dt = -dt;
      dch = -dch;
      float inv = tot1;
      tot1 = tot2;
      tot2 = inv;
    }

    auto offset = offsPrev;
    if (ch > chPrev) {
      offset += std::accumulate(mEntriesSlot.begin() + chPrev, mEntriesSlot.begin() + ch, 0);
    }
    offsPrev = offset;
    chPrev = ch;

    // TO be adjusted
    //mTOFCollectedCalibInfoSlot.emplace(mTOFCollectedCalibInfoSlot.begin() + offset, data[ord[i]].getTimestamp(), data[ord[i]].getDeltaTimePi(), data[ord[i]].getTot(), data[ord[i]].getFlags());
    //mEntriesSlot[ch]++;
  }
}
//_____________________________________________
void TOFCalibInfoSlot::merge(const TOFCalibInfoSlot* prev)
{
  // merge data of 2 slots

  LOG(debug) << "Merging two slots with entries: current slot -> " << mTOFCollectedCalibInfoSlot.size() << " , previous slot -> " << prev->mTOFCollectedCalibInfoSlot.size();

  int offset = 0, offsetPrev = 0;
  std::vector<o2::dataformats::CalibInfoTOF> tmpVector;
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
  LOG(debug) << "After merging the size is " << mTOFCollectedCalibInfoSlot.size();
  return;
}
//_____________________________________________
void TOFCalibInfoSlot::print() const
{
  // to print number of entries in the tree and the channel with the max number of entries

  LOG(info) << "Total number of entries " << mTOFCollectedCalibInfoSlot.size();
  auto maxElementIndex = std::max_element(mEntriesSlot.begin(), mEntriesSlot.end());
  auto channelIndex = std::distance(mEntriesSlot.begin(), maxElementIndex);
  LOG(info) << "The maximum number of entries per channel in the current mTOFCollectedCalibInfo is " << *maxElementIndex << " for channel " << channelIndex;
  return;
}

//_____________________________________________
void TOFCalibInfoSlot::printEntries() const
{
  // to print number of entries in the tree and per channel

  LOG(info) << "Total number of entries " << mTOFCollectedCalibInfoSlot.size();
  for (int i = 0; i < mEntriesSlot.size(); ++i) {
    if (mEntriesSlot[i] != 0) {
      LOG(info) << "channel " << i << " has " << mEntriesSlot[i] << " entries";
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

  if (mTest) {
    return true;
  }
  const o2::tof::TOFCalibInfoSlot* c = slot.getContainer();
  LOG(info) << "we have " << c->getCollectedCalibInfoSlot().size() << " entries";
  int maxNumberOfHits = mAbsMaxNumOfHits ? mMaxNumOfHits : mMaxNumOfHits * o2::tof::Geo::NCHANNELS;
  if (mTFsendingPolicy || c->getCollectedCalibInfoSlot().size() > maxNumberOfHits) {
    return true;
  }
  return false;
}

//_____________________________________________
void TOFCalibCollector::finalizeSlot(Slot& slot)
{
  // here we fill the tree with the remaining stuff that was not filled before

  o2::tof::TOFCalibInfoSlot* c = slot.getContainer();
  mTOFCollectedCalibInfo = c->getCollectedCalibInfoSlot();
  // let's sort before to write
  std::sort(mTOFCollectedCalibInfo.begin(), mTOFCollectedCalibInfo.end(), [](const o2::dataformats::CalibInfoTOF& a, const o2::dataformats::CalibInfoTOF& b) { return a.getTOFChIndex() < b.getTOFChIndex(); });
  LOG(debug) << "vector of CalibTOFInfoShort received with size = " << mTOFCollectedCalibInfo.size();
  mEntries = c->getEntriesPerChannel();
  return;
}

//_____________________________________________
Slot& TOFCalibCollector::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{

  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<TOFCalibInfoSlot>(mLHCphase));
  return slot;
}

} // end namespace tof
} // end namespace o2
