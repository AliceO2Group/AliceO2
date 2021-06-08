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
namespace o2::trd
{
class Digit;
class Tracklet64;
class CompressedDigit;

/// \class EventRecord
/// \brief Stores a TRD event
/// adapted from TriggerRecord

class EventRecord
{
  using BCData = o2::InteractionRecord;

 public:
  EventRecord() = default;
  EventRecord(const BCData& bunchcrossing) : mBCData(bunchcrossing) {}
  ~EventRecord() = default;

  void setBCData(const BCData& data) { mBCData = data; }

  const BCData& getBCData() const { return mBCData; }
  BCData& getBCData() { return mBCData; }

  //Digit information
  std::vector<Digit>& getDigits() { return mDigits; }
  std::vector<CompressedDigit>& getCompressedDigits() { return mCompressedDigits; }
  void addDigits(Digit& digit) { mDigits.push_back(digit); }
  void addCompressedDigits(CompressedDigit& digit) { mCompressedDigits.push_back(digit); }
  void addDigits(std::vector<Digit>::iterator& start, std::vector<Digit>::iterator& end) { mDigits.insert(mDigits.end(), start, end); }
  void addCompressedDigits(std::vector<CompressedDigit>::iterator& start, std::vector<CompressedDigit>::iterator& end) { mCompressedDigits.insert(mCompressedDigits.end(), start, end); }

  //tracklet information
  std::vector<Tracklet64>& getTracklets() { return mTracklets; }
  void addTracklet(Tracklet64& tracklet) { mTracklets.push_back(tracklet); }
  void addTracklets(std::vector<Tracklet64>::iterator& start, std::vector<Tracklet64>::iterator& end) { mTracklets.insert(mTracklets.end(), start, end); }

  void printStream(std::ostream& stream) const;

  bool operator==(const EventRecord& o) const
  {
    return mBCData == o.mBCData; //&& mDigits == o.mDigits && mTracklets == o.mTracklets && mCompressedDigits == o.mCompressedDigits;
  }
  void clear()
  {
    mDigits.clear();
    mTracklets.clear();
    mCompressedDigits.clear();
  }

 private:
  BCData mBCData;                                 /// orbit and Bunch crossing data of the trigger
  std::vector<Digit> mDigits;                     /// Index of the underlying digit data, indexes into the vector/array/span
  std::vector<Tracklet64> mTracklets;             /// Index of the underlying tracklet data, indexes into the vector/array/span
  std::vector<CompressedDigit> mCompressedDigits; /// Index of the underlying digit data, indexes into the vector/array/span

  ClassDefNV(EventRecord, 1);
};

class EventStorage
{
  //storage of eventrecords
  //a vector of eventrecords and the associated funationality to go with it.
 public:
  void clear() { mEventRecords.clear(); }
  void addDigits(InteractionRecord& ir, Digit& digit)
  {
    bool added = false;
    for (auto eventrecord : mEventRecords) {
      if (ir == eventrecord.getBCData()) {
        //TODO replace this with a hash/map not a vector
        eventrecord.addDigits(digit);
        added = true;
      }
    }
    if (!added) {
      // unseen ir so add it
      mEventRecords.push_back(ir);
      mEventRecords.end()->addDigits(digit);
    }
  }
  void addCompressedDigits(InteractionRecord& ir, CompressedDigit& digit)
  {
    bool added = false;
    for (auto eventrecord : mEventRecords) {
      if (ir == eventrecord.getBCData()) {
        //TODO replace this with a hash/map not a vector
        eventrecord.addCompressedDigits(digit);
        added = true;
      }
    }
    if (!added) {
      // unseen ir so add it
      mEventRecords.push_back(ir);
      mEventRecords.end()->addCompressedDigits(digit);
    }
  }
  void addDigits(InteractionRecord& ir, std::vector<Digit>::iterator start, std::vector<Digit>::iterator end)
  {
    bool added = false;
    for (auto eventrecord : mEventRecords) {
      if (ir == eventrecord.getBCData()) {
        //TODO replace this with a hash/map not a vector
        eventrecord.addDigits(start, end);
        added = true;
      }
    }
    if (!added) {
      // unseen ir so add it
      mEventRecords.push_back(ir);
      mEventRecords.end()->addDigits(start, end);
    }
  }
  void addCompressedDigits(InteractionRecord& ir, std::vector<CompressedDigit>::iterator start, std::vector<CompressedDigit>::iterator end)
  {
    bool added = false;
    for (auto eventrecord : mEventRecords) {
      if (ir == eventrecord.getBCData()) {
        //TODO replace this with a hash/map not a vector
        eventrecord.addCompressedDigits(start, end);
        added = true;
      }
    }
    if (!added) {
      // unseen ir so add it
      mEventRecords.push_back(ir);
      mEventRecords.end()->addCompressedDigits(start, end);
    }
  }
  void addTracklet(InteractionRecord& ir, Tracklet64& tracklet)
  {
    bool added = false;
    for (auto eventrecord : mEventRecords) {
      if (ir == eventrecord.getBCData()) {
        //TODO replace this with a hash/map not a vector
        eventrecord.addTracklet(tracklet);
        added = true;
      }
    }
    if (!added) {
      // unseen ir so add it
      mEventRecords.push_back(ir);
      mEventRecords.end()->addTracklet(tracklet);
    }
  }
  void addTracklets(InteractionRecord& ir, std::vector<Tracklet64>::iterator start, std::vector<Tracklet64>::iterator end)
  {
    bool added = false;
    for (auto eventrecord : mEventRecords) {
      if (ir == eventrecord.getBCData()) {
        //TODO replace this with a hash/map not a vector
        eventrecord.addTracklets(start, end); //mTracklets.insert(mTracklets.end(),start,end);
        added = true;
      }
    }
    if (!added) {
      // unseen ir so add it
      mEventRecords.push_back(ir);
      mEventRecords.end()->addTracklets(start, end);
    }
  }
  void unpackDataForSending(std::vector<TriggerRecord>& triggers, std::vector<Tracklet64>& tracklets, std::vector<Digit>& digits)
  {
    int digitcount = 0;
    int trackletcount = 0;
    for (auto event : mEventRecords) {
      tracklets.insert(std::end(tracklets), std::begin(event.getTracklets()), std::end(event.getTracklets()));
      digits.insert(std::end(digits), std::begin(event.getDigits()), std::end(event.getDigits()));
      triggers.emplace_back(event.getBCData(), digitcount, event.getDigits().size(), trackletcount, event.getTracklets().size());
      digitcount += event.getDigits().size();
      trackletcount += event.getTracklets().size();
    }
  }
  void unpackDataForSending(std::vector<TriggerRecord>& triggers, std::vector<Tracklet64>& tracklets, std::vector<CompressedDigit>& digits)
  {
    int digitcount = 0;
    int trackletcount = 0;
    for (auto event : mEventRecords) {
      tracklets.insert(std::end(tracklets), std::begin(event.getTracklets()), std::end(event.getTracklets()));
      digits.insert(std::end(digits), std::begin(event.getCompressedDigits()), std::end(event.getCompressedDigits()));
      triggers.emplace_back(event.getBCData(), digitcount, event.getDigits().size(), trackletcount, event.getTracklets().size());
      digitcount += event.getDigits().size();
      trackletcount += event.getTracklets().size();
    }
  }
  int sumTracklets()
  {
    int sum = 0;
    for (auto event : mEventRecords) {
      sum += event.getTracklets().size();
    }
    return sum;
  }
  int sumDigits()
  {
    int sum = 0;
    for (auto event : mEventRecords) {
      sum += event.getDigits().size();
    }
    return sum;
  }
  std::vector<Tracklet64>& getTracklets(InteractionRecord& ir)
  {
    bool found = false;
    for (auto event : mEventRecords) {
      if (ir == event.getBCData()) {
        found = true;
        return event.getTracklets();
      }
    }
    LOG(warn) << "attempted to get tracklets from IR: " << ir << " total tracklets of:" << sumTracklets();
    printIR();
    return mDummyTracklets;
  }
  std::vector<Digit>& getDigits(InteractionRecord& ir)
  {
    bool found = false;
    for (auto event : mEventRecords) {
      if (ir == event.getBCData()) {
        found = true;
        return event.getDigits();
      }
    }
    LOG(fatal) << "attempted to get digits from IR: " << ir << " total digits of:" << sumDigits();
    printIR();
    return mDummyDigits;
  }

  std::vector<CompressedDigit>& getCompressedDigits(InteractionRecord& ir)
  {
    bool found = false;
    for (auto event : mEventRecords) {
      if (ir == event.getBCData()) {
        found = true;
        return event.getCompressedDigits();
      }
    }
    LOG(fatal) << "attempted to get digits from IR: " << ir << " total digits of:" << sumDigits();
    printIR();
    return mDummyCompressedDigits;
  }

  void printIR()
  {
    std::string records;
    int count = 0;
    for (auto event : mEventRecords) {
      LOG(info) << "[" << count << "]" << event.getBCData() << " ";
      count++;
    }
  }

 private:
  std::vector<EventRecord> mEventRecords;
  //these 2 are hacks to be able to send bak a blank vector if interaction record is not found.
  std::vector<Tracklet64> mDummyTracklets;
  std::vector<Digit> mDummyDigits;
  std::vector<CompressedDigit> mDummyCompressedDigits;
  ClassDefNV(EventStorage, 1);
};
std::ostream& operator<<(std::ostream& stream, const EventRecord& trg);

} // namespace o2::trd

#endif
