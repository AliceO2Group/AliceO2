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

#ifndef ALICEO2_CPV_TRIGGERRECORD_H
#define ALICEO2_CPV_TRIGGERRECORD_H

#include <iosfwd>
#include "Rtypes.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"

namespace o2
{

namespace cpv
{

/// \class TriggerRecord
/// \brief Header for data corresponding to the same hardware trigger
/// adapted from DataFormatsEMCAL/TriggerRecord
class TriggerRecord
{
  using BCData = o2::InteractionRecord;
  using DataRange = o2::dataformats::RangeReference<int>;

 public:
  TriggerRecord() = default;
  TriggerRecord(const BCData& bunchcrossing, int firstentry, int nentries) : mBCData(bunchcrossing), mDataRange(firstentry, nentries) {}
  ~TriggerRecord() = default;

  void setBCData(const BCData& data) { mBCData = data; }
  void setDataRange(int firstentry, int nentries) { mDataRange.set(firstentry, nentries); }
  void setIndexFirstObject(int firstentry) { mDataRange.setFirstEntry(firstentry); }
  void setNumberOfObjects(int nentries) { mDataRange.setEntries(nentries); }

  const BCData& getBCData() const { return mBCData; }
  BCData& getBCData() { return mBCData; }
  int getNumberOfObjects() const { return mDataRange.getEntries(); }
  int getFirstEntry() const { return mDataRange.getFirstEntry(); }

  void PrintStream(std::ostream& stream) const;

 private:
  BCData mBCData;       /// Bunch crossing data of the trigger
  DataRange mDataRange; /// Index of the triggering event (event index and first entry in the container)

  ClassDefNV(TriggerRecord, 1);
};

std::ostream& operator<<(std::ostream& stream, const TriggerRecord& trg);

} // namespace cpv

} // namespace o2

#endif
