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

#ifndef ALICEO2_TRD_EVENTRECORD_H
#define ALICEO2_TRD_EVENTRECORD_H

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include <fairlogger/Logger.h>
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/RawDataStats.h"
#include "DataFormatsTRD/Digit.h"

namespace o2::framework
{
class ProcessingContext;
}

namespace o2::trd
{
class TriggerRecord;

/// \class EventRecord
/// \brief Stores a TRD event
class EventRecord
{
  using BCData = o2::InteractionRecord;

 public:
  EventRecord() = default;
  EventRecord(BCData bunchcrossing) : mBCData(bunchcrossing) {}
  ~EventRecord() = default;

  const BCData& getBCData() const { return mBCData; }

  void addDigit(Digit digit) { mDigits.push_back(digit); }
  void addTracklet(Tracklet64 tracklet) { mTracklets.push_back(tracklet); }

  const std::vector<Digit>& getDigits() const { return mDigits; }
  const std::vector<Tracklet64>& getTracklets() const { return mTracklets; }
  bool getIsCalibTrigger() const { return mIsCalibTrigger; }
  float getTotalTime() const { return mTimeTaken; }
  float getDigitTime() const { return mTimeTakenForDigits; }
  float getTrackletTime() const { return mTimeTakenForTracklets; }

  // needed, in order to check if a trigger already exist for this bunch crossing
  bool operator==(const EventRecord& o) const { return mBCData == o.mBCData; }

  // sort the tracklets (and optionally digits) by detector, pad row, pad column
  void sortData(bool sortDigits);

  void incTrackletTime(float timeadd) { mTimeTakenForTracklets += timeadd; }
  void incDigitTime(float timeadd) { mTimeTakenForDigits += timeadd; }
  void incTime(float duration) { mTimeTaken += duration; }
  void setIsCalibTrigger() { mIsCalibTrigger = true; }

 private:
  BCData mBCData;                       /// orbit and Bunch crossing data of the physics trigger
  std::vector<Digit> mDigits{};         /// digit data, for this event
  std::vector<Tracklet64> mTracklets{}; /// tracklet data, for this event
  float mTimeTaken = 0.;                // total parsing time [us] (including digit and tracklet parsing time)
  float mTimeTakenForDigits = 0.;       // time take to process tracklet data blocks [us].
  float mTimeTakenForTracklets = 0.;    // time take to process digit data blocks [us].
  bool mIsCalibTrigger = false;         // flag calibration trigger
};

/// \class EventRecordContainer
/// \brief Stores the TRD data for one TF i.e. a vector of EventRecords and some statistics
class EventRecordContainer
{

 public:
  EventRecordContainer() = default;
  ~EventRecordContainer() = default;

  void sendData(o2::framework::ProcessingContext& pc, bool generatestats, bool sortDigits);

  void setCurrentEventRecord(const InteractionRecord& ir);
  EventRecord& getCurrentEventRecord() { return mEventRecords.at(mCurrEventRecord); }

  // statistics to keep
  void incLinkErrorFlags(int hcid, unsigned int flag) { mTFStats.mLinkErrorFlag[hcid] |= flag; }
  void incLinkNoData(int hcid) { mTFStats.mLinkNoData[hcid]++; }
  void incLinkWords(int hcid, int count) { mTFStats.mLinkWords[hcid] += count; }
  void incLinkWordsRead(int hcid, int count) { mTFStats.mLinkWordsRead[hcid] += count; }
  void incLinkWordsRejected(int hcid, int count) { mTFStats.mLinkWordsRejected[hcid] += count; }
  void incMajorVersion(int version) { mTFStats.mDataFormatRead[version]++; }

  void incParsingError(int error, int hcid)
  {
    mTFStats.mParsingErrors[error]++;
    if (hcid >= 0) { // hcid==-1 is reserved for those errors where we don't have the corresponding link ID
      if (error == NoError) {
        mTFStats.mParsingOK[hcid]++;
      } else {
        mTFStats.mParsingErrorsByLink.push_back(hcid * TRDLastParsingError + error);
      }
    }
  }
  void reset();
  void accumulateStats();

 private:
  int mCurrEventRecord = 0;
  std::vector<EventRecord> mEventRecords;
  TRDDataCountersPerTimeFrame mTFStats;
};

} // namespace o2::trd

#endif
