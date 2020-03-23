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
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsParameters/GRPObject.h"
#include <FairLogger.h>

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
class DigitizationContext
{
 public:
  DigitizationContext() : mNofEntries{0}, mMaxPartNumber{0}, mEventRecords(), mEventParts() {}

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

  // we need a method to fill the file names
  void setSimPrefixes(std::vector<std::string> const& p);
  std::vector<std::string> const& getSimPrefixes() const { return mSimPrefixes; }

  /// Common functions the setup input TChains for reading, given the state (prefixes) encapsulated
  /// by this context. The input vector needs to be empty otherwise nothing will be done.
  /// return boolean saying if input simchains was modified or not
  bool initSimChains(o2::detectors::DetID detid, std::vector<TChain*>& simchains) const;

  /// function reading the hits from a chain (previously initialized with initSimChains
  /// The hits pointer will be initialized (what to we do about ownership??)
  template <typename T>
  void retrieveHits(std::vector<TChain*> const& chains,
                    const char* brname,
                    int sourceID,
                    int entryID,
                    std::vector<T>* hits) const;

  /// returns the GRP object associated to this context
  o2::parameters::GRPObject const& getGRP() const;

 private:
  int mNofEntries = 0;
  int mMaxPartNumber = 0; // max number of parts in any given collision
  float mMuBC;            // probability of hadronic interaction per bunch

  std::vector<o2::InteractionTimeRecord> mEventRecords;
  // for each collision we record the constituents (which shall not exceed mMaxPartNumber)
  std::vector<std::vector<o2::steer::EventPart>> mEventParts;

  o2::BunchFilling mBCFilling; // patter of active BCs

  std::vector<std::string> mSimPrefixes;             // identifiers to the hit sim products; the index corresponds to the source ID of event record
  mutable o2::parameters::GRPObject* mGRP = nullptr; //!

  ClassDefNV(DigitizationContext, 2);
};

/// function reading the hits from a chain (previously initialized with initSimChains
template <typename T>
inline void DigitizationContext::retrieveHits(std::vector<TChain*> const& chains,
                                              const char* brname,
                                              int sourceID,
                                              int entryID,
                                              std::vector<T>* hits) const
{
  auto br = chains[sourceID]->GetBranch(brname);
  if (!br) {
    LOG(ERROR) << "No branch found";
    return;
  }
  br->SetAddress(&hits);
  br->GetEntry(entryID);
}

} // namespace steer
} // namespace o2

#endif
