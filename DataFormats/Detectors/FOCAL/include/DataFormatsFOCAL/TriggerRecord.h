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
#ifndef ALICEO2_FOCAL_TRIGGERRECORD_H
#define ALICEO2_FOCAL_TRIGGERRECORD_H

#include <cstdint>
#include <iosfwd>
#include "Rtypes.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"

namespace o2::focal
{
class TriggerRecord
{
  using BCData = o2::InteractionRecord;
  using DataRange = o2::dataformats::RangeReference<int>;

 public:
  TriggerRecord() = default;
  TriggerRecord(const BCData& bunchcrossing, int firstpadentry, int npadentries, int firstchipentry, int nchipentries, int firsthitentry, int nhitentries) : mBCData(bunchcrossing), mPadDataRange(firstpadentry, npadentries), mPixelChipRange(firstchipentry, nchipentries), mPixelHitRange(firsthitentry, nhitentries) {}
  ~TriggerRecord() = default;

  void setBCData(const BCData& data) { mBCData = data; }
  void setPadDataRange(int firstentry, int nentries) { mPadDataRange.set(firstentry, nentries); }
  void setPixelChipDataRange(int firstentry, int nentries) { mPixelChipRange.set(firstentry, nentries); }
  void setPixelHitDataRange(int firstentry, int nentries) { mPixelHitRange.set(firstentry, nentries); }
  void setIndexFirstPadObject(int firstentry) { mPadDataRange.setFirstEntry(firstentry); }
  void setIndexFirstPixelChipObject(int firstentry) { mPixelChipRange.setFirstEntry(firstentry); }
  void setIndexFirstPixelHitObject(int firstentry) { mPixelHitRange.setFirstEntry(firstentry); }
  void setNumberOfPadObjects(int nentries) { mPadDataRange.setEntries(nentries); }
  void setNumberOfPixelChipObjects(int nentries) { mPixelChipRange.setEntries(nentries); }
  void setNumberOfPixelHitObjects(int nentries) { mPixelHitRange.setEntries(nentries); }

  const BCData& getBCData() const { return mBCData; }
  BCData& getBCData() { return mBCData; }
  int getNumberOfPadObjects() const { return mPadDataRange.getEntries(); }
  int getNumberOfPixelChipObjects() const { return mPixelChipRange.getEntries(); }
  int getNumberOfPixelHitObjects() const { return mPixelHitRange.getEntries(); }
  int getFirstPadEntry() const { return mPadDataRange.getFirstEntry(); }
  int getFirstPixelChipEntry() const { return mPixelChipRange.getFirstEntry(); }
  int getFirstPixelHitEntry() const { return mPixelHitRange.getFirstEntry(); }

  void printStream(std::ostream& stream) const;

 private:
  BCData mBCData;            /// Bunch crossing data of the trigger
  DataRange mPadDataRange;   /// Index range of the pad data for the same trigger (number of pads and first entry in the container)
  DataRange mPixelChipRange; /// Index range of the pixel chips in the same trigger  (number of chips and first entry in the container)
  DataRange mPixelHitRange;  /// Index range of the pixel hits in the same trigger  (number of hits and first entry in the container)

  ClassDefNV(TriggerRecord, 1);
};

std::ostream& operator<<(std::ostream& stream, const TriggerRecord& trg);

}; // namespace o2::focal

#endif