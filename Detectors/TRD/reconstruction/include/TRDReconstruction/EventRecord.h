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

  // needed, in order to check if a trigger already exist for this bunch crossing
  bool operator==(const EventRecord& o) const { return mBCData == o.mBCData; }

  // only the tracklets are sorted by detector ID
  // TODO: maybe at some point a finer sorting might be helpful (padrow, padcolumn?)
  void sortTrackletsByDetector();

  //statistics stuff these get passed to the per tf data at the end of the timeframe,
  //but as we read in per link, events are seperated hence these counters
  const TRDDataCountersPerEvent& getEventStats() const { return mEventStats; }
  void incTrackletTime(double timeadd) { mEventStats.mTimeTakenForTracklets += timeadd; }
  void incDigitTime(double timeadd) { mEventStats.mTimeTakenForDigits += timeadd; }
  void incTime(double duration) { mEventStats.mTimeTaken += duration; }
  void incWordsRead(int count) { mEventStats.mWordsRead += count; }         // words read in
  void incWordsRejected(int count) { mEventStats.mWordsRejected += count; } // words read in
  void incTrackletsFound(int count) { mEventStats.mTrackletsFound += count; }
  void incDigitsFound(int count) { mEventStats.mDigitsFound += count; }
  // OS: Do we need to keep event statistics at all? Are they not anyhow accumulated for a whole TF?
  //     or are they also used on a per event basis somewhere?

 private:
  BCData mBCData;                       /// orbit and Bunch crossing data of the physics trigger
  std::vector<Digit> mDigits{};         /// digit data, for this event
  std::vector<Tracklet64> mTracklets{}; /// tracklet data, for this event
  TRDDataCountersPerEvent mEventStats{}; /// statistics, for this trigger
};

/// \class EventRecordContainer
/// \brief Stores the TRD data for one TF i.e. a vector of EventRecords and some statistics
class EventRecordContainer
{

 public:
  EventRecordContainer() = default;
  ~EventRecordContainer() = default;

  void sendData(o2::framework::ProcessingContext& pc, bool generatestats);

  void setCurrentEventRecord(const InteractionRecord& ir);
  EventRecord& getCurrentEventRecord() { return mEventRecords.at(mCurrEventRecord); }

  //statistics to keep
  void incTrackletTime(double timeadd) { mTFStats.mTimeTakenForTracklets += timeadd; }
  void incDigitTime(double timeadd) { mTFStats.mTimeTakenForDigits += timeadd; }
  void incTrackletsFound(int count) { mTFStats.mTrackletsFound += count; }
  void incDigitsFound(int count) { mTFStats.mDigitsFound += count; }
  void incLinkErrorFlags(int sm, int side, int stacklayer, unsigned int flag) { mTFStats.mLinkErrorFlag[(sm * 2 + side) * 30 + stacklayer] |= flag; }
  void incLinkNoData(int sm, int side, int stacklayer) { mTFStats.mLinkNoData[(sm * 2 + side) * 30 + stacklayer]++; }
  void incLinkWords(int sm, int side, int stacklayer, int count) { mTFStats.mLinkWords[(sm * 2 + side) * 30 + stacklayer] += count; }
  void incLinkWordsRead(int sm, int side, int stacklayer, int count) { mTFStats.mLinkWordsRead[(sm * 2 + side) * 30 + stacklayer] += count; }
  void incLinkWordsRejected(int sm, int side, int stacklayer, int count) { mTFStats.mLinkWordsRejected[(sm * 2 + side) * 30 + stacklayer] += count; }
  void incMajorVersion(int version) { mTFStats.mDataFormatRead[version]++; }

  void incParsingError(int error, int hcid)
  {
    if (error >= TRDLastParsingError) {
      LOG(info) << "wrong error number to inc ParsingError in TrackletParsing : error" << error << " for hcid:" << hcid;
    } else {
      mTFStats.mParsingErrors[error]++;
      if (hcid >= 0) { // hcid==-1 is reserved for those errors where we don't have the corresponding link ID
        if (hcid * TRDLastParsingError + error < o2::trd::constants::NCHAMBER * 2 * TRDLastParsingError) {
          //prevent bounding errors
          mTFStats.mParsingErrorsByLink[hcid * TRDLastParsingError + error]++;
        }
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
