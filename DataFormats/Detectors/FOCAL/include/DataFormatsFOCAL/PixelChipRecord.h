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
#ifndef ALICEO2_FOCAL_PIXELCHIPRECORD_H
#define ALICEO2_FOCAL_PIXELCHIPRECORD_H

#include <cstdint>
#include <iosfwd>
#include "Rtypes.h"
#include "CommonDataFormat/RangeReference.h"

namespace o2::focal
{
class PixelChipRecord
{
  using DataRange = o2::dataformats::RangeReference<int>;

 public:
  PixelChipRecord() = default;
  PixelChipRecord(int layerID, int laneID, int chipID, int firsthit, int nhits) : mLayerID(layerID), mLaneID(laneID), mChipID(chipID), mHitIndexRange(firsthit, nhits) {}
  ~PixelChipRecord() = default;

  void setLayerID(int layerID) { mLayerID = layerID; }
  void setLaneID(int laneID) { mLaneID = laneID; }
  void setChipID(int chipID) { mChipID = chipID; }
  void setIndexFirstHit(int firsthit) { mHitIndexRange.setFirstEntry(firsthit); }
  void setNumberOfHits(int nhits) { mHitIndexRange.setEntries(nhits); }

  int getLayerID() const { return mLayerID; }
  int getLaneID() const { return mLaneID; }
  int getChipID() const { return mChipID; }
  int getFirstHit() const { return mHitIndexRange.getFirstEntry(); }
  int getNumberOfHits() const { return mHitIndexRange.getFirstEntry(); }

  void printStream(std::ostream& stream) const;

 private:
  int mLayerID = -1;        /// Layer index
  int mLaneID = -1;         /// Lane index
  int mChipID = -1;         /// Chip index
  DataRange mHitIndexRange; /// Index range of hits belonging to theh chip

  ClassDefNV(PixelChipRecord, 1);
};

std::ostream& operator<<(std::ostream& stream, const PixelChipRecord& trg);

}; // namespace o2::focal

#endif