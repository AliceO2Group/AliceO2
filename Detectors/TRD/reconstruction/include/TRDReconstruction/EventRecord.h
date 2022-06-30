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

#include <iosfwd>
#include "Rtypes.h"
#include "TH2F.h"
#include <array>
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include "FairLogger.h"
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
/// adapted from TriggerRecord

class EventRecord
{
  using BCData = o2::InteractionRecord;

 public:
  EventRecord() = default;
  EventRecord(BCData& bunchcrossing) : mBCData(bunchcrossing)
  {
    mTracklets.reserve(30);
    mDigits.reserve(20);
  }
  ~EventRecord() = default;

  void setBCData(const BCData& data) { mBCData = data; }

  const BCData& getBCData() const { return mBCData; }
  BCData& getBCData() { return mBCData; }

  //Digit information
  std::vector<Digit>& getDigits();
  void addDigits(Digit& digit);
  void addDigits(std::vector<Digit>::iterator& start, std::vector<Digit>::iterator& end);

  //tracklet information
  std::vector<Tracklet64>& getTracklets();
  void addTracklet(Tracklet64& tracklet);
  void addTracklets(std::vector<Tracklet64>::iterator& start, std::vector<Tracklet64>::iterator& end);
  void addTracklets(std::vector<Tracklet64>& tracklets);
  void popTracklets(int popcount);
  //void printStream(std::ostream& stream) const;
  void sortByHCID();

  //statistics stuff these get passed to the per tf data at the end of the timeframe,
  //but as we read in per link, events are seperated hence these counters
  void clearStats();
  void incTrackletTime(double timeadd) { mEventStats.mTimeTakenForTracklets += timeadd; }

  void incDigitTime(double timeadd) { mEventStats.mTimeTakenForDigits += timeadd; }
  void incTime(double duration) { mEventStats.mTimeTaken += duration; }
  void incWordsRead(int count) { mEventStats.mWordsRead += count; }         // words read in
  void incWordsRejected(int count) { mEventStats.mWordsRejected += count; } // words read in
  void incTrackletsFound(int count) { mEventStats.mTrackletsFound += count; }
  void incDigitsFound(int count) { mEventStats.mDigitsFound += count; }
  void setDataPerLink(int link, int length)
  { /* mEventStats.mLinkLength[link] = length;*/
  }
  //std::array<uint8_t, 1080> mLinkErrorFlag{}; //status of the error flags for this event, 8bit values from cru halfchamber header.
  bool operator==(const EventRecord& o) const
  {
    return mBCData == o.mBCData; //&& mDigits == o.mDigits && mTracklets == o.mTracklets ;
  }
  void clear()
  {
    mDigits.clear();
    mTracklets.clear();
  }

  void incStats(int tracklets, int digits, int wordsread, int wordsrejected);
  o2::trd::TRDDataCountersPerEvent mEventStats;

 private:
  BCData mBCData;                       /// orbit and Bunch crossing data of the physics trigger
  std::vector<Digit> mDigits{};         /// digit data, for this event
  std::vector<Tracklet64> mTracklets{}; /// tracklet data, for this event
  //statistics stuff these get passed to the per tf data at the end of the timeframe,
  //but as we read in per link, events are seperated hence these counters
};

class EventStorage
{
  //store a timeframes events for later collating sending on as a message
 public:
  EventStorage() = default;
  ~EventStorage() = default;
  //storage of eventrecords
  //a vector of eventrecords and the associated funationality to go with it.
  void clear() { mEventRecords.clear(); }
  void addDigits(InteractionRecord& ir, Digit& digit);
  void addDigits(InteractionRecord& ir, std::vector<Digit>::iterator start, std::vector<Digit>::iterator end);
  void addTracklet(InteractionRecord& ir, Tracklet64& tracklet);
  void addTracklets(InteractionRecord& ir, std::vector<Tracklet64>& tracklets);
  void addTracklets(InteractionRecord& ir, std::vector<Tracklet64>::iterator& start, std::vector<Tracklet64>::iterator& end);
  void unpackData(std::vector<TriggerRecord>& triggers, std::vector<Tracklet64>& tracklets, std::vector<Digit>& digits);
  void sendData(o2::framework::ProcessingContext& pc, bool generatestats);
  EventRecord& getEventRecord(InteractionRecord& ir);
  //this could replace by keeing a running total on addition TODO
  void sumTrackletsDigitsTriggers(uint64_t& tracklets, uint64_t& digits, uint64_t& triggers);
  int sumTracklets();
  int sumDigits();
  std::vector<Tracklet64>& getTracklets(InteractionRecord& ir);
  std::vector<Digit>& getDigits(InteractionRecord& ir);
  void printIR();
  //  void setHisto(TH1F* packagetime) { mPackagingTime = packagetime; }

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
  // left here for now, but this can be calculated inside qc, so possibly no point having it here.
  /*   void incMCMTrackletCount(int hcid, int rob, int mcm, int count)
  {
    LOG(info) << " mcm tracklet count increment for mcm : " << hcid / 2 * constants::NROBC1 * constants::NMCMROB + rob * constants::NMCMROB + mcm << " made up of hcid:" << hcid << " rob : " << rob << " mcm : " << mcm;
    mTFStats.mMCMTrackletsFound[hcid / 2 * constants::NROBC1 * constants::NMCMROB + rob * constants::NMCMROB + mcm] += count;
  }
  void incMCMDigitCount(int hcid, int rob, int mcm, int count)
  {
    LOG(info) << " mcm tracklet count increment for mcm : " << hcid / 2 * constants::NROBC1 * constants::NMCMROB + rob * constants::NMCMROB + mcm << " made up of hcid:" << hcid << " rob : " << rob << " mcm : " << mcm;
    mTFStats.mMCMDigitsFound[hcid / 2 * constants::NROBC1 * constants::NMCMROB + rob * constants::NMCMROB + mcm] += count;
  }*/

  void incParsingError(int error, int sm, int side, int stacklayer)
  {
    if (error > TRDLastParsingError) {
      LOG(info) << "wrong error number to inc ParsingError in TrackletParsing : " << error << " for " << sm << "_" << side << "_" << stacklayer;
    } else {
      mTFStats.mParsingErrors[error]++;
      if (sm >= 0) { // sm=-1 is reserved for those errors where we dont know or cant know the underlying source.
        if ((sm * 2 + side) * 30 * TRDLastParsingError + TRDLastParsingError * stacklayer + error < o2::trd::constants::NSECTOR * 60 * TRDLastParsingError) {
          //prevent bounding errors
          mTFStats.mParsingErrorsByLink[(sm * 2 + side) * 30 * TRDLastParsingError + TRDLastParsingError * stacklayer + error]++;
        }
      }
    }
  } // halfsm and stacklayer are the x and y of the 2d histograms
  void resetCounters();
  void accumulateStats();

 private:
  std::vector<EventRecord> mEventRecords;
  //these 2 are hacks to be able to send back a blank vector if interaction record is not found.
  std::vector<Tracklet64> mDummyTracklets;
  std::vector<Digit> mDummyDigits;

  TRDDataCountersPerTimeFrame mTFStats;

  //  TH1F* mPackagingTime{nullptr};
};

std::ostream& operator<<(std::ostream& stream, const EventRecord& trg);

} // namespace o2::trd

#endif
