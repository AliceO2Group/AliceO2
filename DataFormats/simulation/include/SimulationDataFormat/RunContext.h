// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_SIMULATIONDATAFORMAT_RUNCONTEXT_H
#define ALICEO2_SIMULATIONDATAFORMAT_RUNCONTEXT_H

#include <vector>
#include <TChain.h>
#include <TBranch.h>
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/BunchFilling.h"

namespace o2
{
namespace steer
{
// a structure describing EventPart
// (an elementary constituent of a collision)
struct EventPart {
  EventPart() = default;
  EventPart(int s, int e) : sourceID(s), entryID(e) {}
  int sourceID = 0; // the ID of the source (0->backGround; > 1 signal source)
  // the sourceID should correspond to the chain ID
  int entryID = 0; // the event/entry ID inside the chain corresponding to sourceID

  static bool isSignal(EventPart e) { return e.sourceID > 1; }
  static bool isBackGround(EventPart e) { return !isSignal(e); }
  ClassDefNV(EventPart, 1);
};

// class fully describing the Collision contexts
class RunContext
{
 public:
  RunContext() : mNofEntries{0}, mMaxPartNumber{0}, mEventRecords(), mEventParts() {}

  int getNCollisions() const { return mNofEntries; }
  void setNCollisions(int n) { mNofEntries = n; }

  void setMaxNumberParts(int maxp) { mMaxPartNumber = maxp; }
  int getMaxNumberParts() const { return mMaxPartNumber; }

  std::vector<o2::InteractionTimeRecord>& getEventRecords() { return mEventRecords; }
  std::vector<std::vector<o2::steer::EventPart>>& getEventParts() { return mEventParts; }

  const std::vector<o2::InteractionTimeRecord>& getEventRecords() const { return mEventRecords; }
  const std::vector<std::vector<o2::steer::EventPart>>& getEventParts() const { return mEventParts; }

  o2::BunchFilling& getBunchFilling() { return mBCFilling; }
  const o2::BunchFilling& getBunchFilling() const { return (const o2::BunchFilling&)mBCFilling; }

  void setMuPerBC(float m) { mMuBC = m; }
  float getMuPerBC() const { return mMuBC; }

  void printCollisionSummary() const;

 private:
  int mNofEntries = 0;
  int mMaxPartNumber = 0; // max number of parts in any given collision
  float mMuBC;            // probability of hadronic interaction per bunch

  std::vector<o2::InteractionTimeRecord> mEventRecords;
  // for each collision we record the constituents (which shall not exceed mMaxPartNumber)
  std::vector<std::vector<o2::steer::EventPart>> mEventParts;

  o2::BunchFilling mBCFilling; // patter of active BCs

  // it would also be appropriate to record the filenames
  // that went into the chain

  ClassDefNV(RunContext, 2);
};
} // namespace steer
} // namespace o2

#endif
