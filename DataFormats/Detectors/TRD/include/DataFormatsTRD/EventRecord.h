// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_EVENTRECORD_H
#define ALICEO2_TRD_EVENTRECORD_H

#include <iosfwd>
#include "Rtypes.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include "FairLogger.h"
#include "DataFormatsTRD/Tracklet64.h"

namespace o2::trd
{
class Digit;
class Tracklet64;
class CompressedDigit;
class TriggerRecord;

/// \class EventRecord
/// \brief Stores a TRD event
/// adapted from TriggerRecord

class EventRecord
{
  using BCData = o2::InteractionRecord;

 public:
  EventRecord() = default;
  EventRecord(const BCData& bunchcrossing) : mBCData(bunchcrossing)
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
  //void printStream(std::ostream& stream) const;

  bool operator==(const EventRecord& o) const
  {
    return mBCData == o.mBCData; //&& mDigits == o.mDigits && mTracklets == o.mTracklets ;
  }
  void clear()
  {
    mDigits.clear();
    mTracklets.clear();
  }

 private:
  BCData mBCData;                       /// orbit and Bunch crossing data of the physics trigger
  std::vector<Digit> mDigits{};         /// digit data, for this event
  std::vector<Tracklet64> mTracklets{}; /// tracklet data, for this event

  ClassDefNV(EventRecord, 1);
};

class EventStorage
{
  //
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
  void unpackDataForSending(std::vector<TriggerRecord>& triggers, std::vector<Tracklet64>& tracklets, std::vector<Digit>& digits);
  int sumTracklets();
  int sumDigits();
  std::vector<Tracklet64>& getTracklets(InteractionRecord& ir);
  std::vector<Digit>& getDigits(InteractionRecord& ir);
  void printIR();

 private:
  std::vector<EventRecord> mEventRecords;
  //these 2 are hacks to be able to send bak a blank vector if interaction record is not found.
  std::vector<Tracklet64> mDummyTracklets;
  std::vector<Digit> mDummyDigits;
  ClassDefNV(EventStorage, 1);
};
std::ostream& operator<<(std::ostream& stream, const EventRecord& trg);

} // namespace o2::trd

#endif
