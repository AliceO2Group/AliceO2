// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_EMCAL_TRIGGERRECORD_H
#define ALICEO2_EMCAL_TRIGGERRECORD_H

#include <cstdint>
#include <iosfwd>
#include "Rtypes.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"

namespace o2
{

namespace emcal
{

/// \class TriggerRecord
/// \brief Header for data corresponding to the same hardware trigger
/// \ingroup EMCALDataFormat
/// \author Markus Fasel <markus.fasel@cern.ch>
/// \since Nov 17, 2019
/// adapted from DataFormatsITSMFT/ROFRecord
class TriggerRecord
{
  using BCData = o2::InteractionRecord;
  using DataRange = o2::dataformats::RangeReference<int>;

 public:
  TriggerRecord() = default;
  TriggerRecord(const BCData& bunchcrossing, int firstentry, int nentries) : mBCData(bunchcrossing), mDataRange(firstentry, nentries), mTriggerBits(0) {}
  TriggerRecord(const BCData& bunchcrossing, uint32_t triggerbits, int firstentry, int nentries) : TriggerRecord(bunchcrossing, firstentry, nentries) { mTriggerBits = triggerbits; }
  ~TriggerRecord() = default;

  void setBCData(const BCData& data) { mBCData = data; }
  void setTriggerBits(uint32_t triggerbits) { mTriggerBits = triggerbits; }
  void setDataRange(int firstentry, int nentries) { mDataRange.set(firstentry, nentries); }
  void setIndexFirstObject(int firstentry) { mDataRange.setFirstEntry(firstentry); }
  void setNumberOfObjects(int nentries) { mDataRange.setEntries(nentries); }

  const BCData& getBCData() const { return mBCData; }
  BCData& getBCData() { return mBCData; }
  uint32_t getTriggerBits() const { return mTriggerBits; }
  int getNumberOfObjects() const { return mDataRange.getEntries(); }
  int getFirstEntry() const { return mDataRange.getFirstEntry(); }

  void printStream(std::ostream& stream) const;

 private:
  BCData mBCData;        /// Bunch crossing data of the trigger
  DataRange mDataRange;  /// Index of the triggering event (event index and first entry in the container)
  uint32_t mTriggerBits; /// Trigger bits as from the Raw Data Header

  ClassDefNV(TriggerRecord, 2);
};

std::ostream& operator<<(std::ostream& stream, const TriggerRecord& trg);

} // namespace emcal

} // namespace o2

#endif
