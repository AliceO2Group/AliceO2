// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_TRIGGERRECORD_H
#define ALICEO2_TRD_TRIGGERRECORD_H

#include <iosfwd>
#include "Rtypes.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"

namespace o2
{

namespace trd
{

/// \class TriggerRecord
/// \brief Header for data corresponding to the same hardware trigger
/// adapted from DataFormatsITSMFT/ROFRecord
class TriggerRecord
{
  using BCData = o2::InteractionRecord;
  using DataRange = o2::dataformats::RangeReference<int>;

 public:
  TriggerRecord() = default;
  TriggerRecord(const BCData& bunchcrossing, int digitentry, int ndigitentries, int trackletentry = 0, int ntrackletentries = 0) : mBCData(bunchcrossing), mTrackletDataRange(trackletentry, ntrackletentries), mDigitDataRange(digitentry, ndigitentries) {}
  // The above default of 0 for tracklet info, makes the digitizer look more decent as one can simply not put in the tracklet info as you dont have it.
  ~TriggerRecord() = default;

  void setBCData(const BCData& data) { mBCData = data; }

  const BCData& getBCData() const { return mBCData; }
  BCData& getBCData() { return mBCData; }

  //Digit information
  void setFirstDigit(int firstentry) { mDigitDataRange.setFirstEntry(firstentry); }
  int getFirstDigit() const { return mDigitDataRange.getFirstEntry(); }
  void setNumberOfDigit(int nentries) { mDigitDataRange.setEntries(nentries); }
  int getNumberOfDigits() const { return mDigitDataRange.getEntries(); }
  void setDigitRange(int firstentry, int nentries) { mDigitDataRange.set(firstentry, nentries); }

  //tracklet information
  void setFirstTracklet(int firstentry) { mTrackletDataRange.setFirstEntry(firstentry); }
  int getFirstTracklet() const { return mTrackletDataRange.getFirstEntry(); }
  void setNumberOfTracklet(int nentries) { mTrackletDataRange.setEntries(nentries); }
  int getNumberOfTracklets() const { return mTrackletDataRange.getEntries(); }
  void setTrackletRange(int firstentry, int nentries) { mTrackletDataRange.set(firstentry, nentries); }

  void printStream(std::ostream& stream) const;

  bool operator==(const TriggerRecord& o) const
  {
    return mBCData == o.mBCData && mDigitDataRange == o.mDigitDataRange && mTrackletDataRange == o.mTrackletDataRange;
  }

 private:
  BCData mBCData;       /// Bunch crossing data of the trigger
  DataRange mDigitDataRange;    /// Index of the underlying digit data, indexes into the vector/array/span
  DataRange mTrackletDataRange; /// Index of the underlying tracklet data, indexes into the vector/array/span

  ClassDefNV(TriggerRecord, 2);
};

std::ostream& operator<<(std::ostream& stream, const TriggerRecord& trg);

} // namespace trd

} // namespace o2

#endif
