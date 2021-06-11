// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   Trigger.h
/// \author Antonio Franco - INFN Bari
/// \version 1.0
/// \date 2/03/2021

#ifndef DETECTORS_HMPID_BASE_INCLUDE_HMPIDBASE_TRIGGER_H_
#define DETECTORS_HMPID_BASE_INCLUDE_HMPIDBASE_TRIGGER_H_

#include <iosfwd>
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"

namespace o2
{
namespace hmpid
{
/// \class Trigger
/// \brief HMPID Trigger declaration
class Trigger
{
  using DataRange = o2::dataformats::RangeReference<int>;

 public:
  static inline uint64_t getTriggerID(uint32_t Orbit, uint16_t BC) { return ((Orbit << 12) | (0x0FFF & BC)); };

 public:
  Trigger() = default;
  Trigger(InteractionRecord ir, int32_t first, int32_t n) : mIr(ir), mDataRange(first, n) {}

  const InteractionRecord& getIr() const { return mIr; };
  uint32_t getOrbit() const { return mIr.orbit; };
  uint16_t getBc() const { return mIr.bc; };
  uint64_t getTriggerID() const { return ((mIr.orbit << 12) | (0x0FFF & mIr.bc)); };
  void setDataRange(int firstentry, int nentries) { mDataRange.set(firstentry, nentries); }
  int getNumberOfObjects() const { return mDataRange.getEntries(); }
  int getFirstEntry() const { return mDataRange.getFirstEntry(); }
  int getLastEntry() const { return mDataRange.getFirstEntry() + mDataRange.getEntries() - 1; }
  void setOrbit(uint32_t orbit)
  {
    mIr.orbit = orbit;
    return;
  }
  void setBC(uint16_t bc)
  {
    mIr.bc = bc;
    return;
  }
  void setTriggerID(uint64_t trigger)
  {
    mIr.orbit = (trigger >> 12);
    mIr.bc = (trigger & 0x0FFF);
    return;
  }

  // Operators definition !
  friend inline bool operator<(const Trigger& l, const Trigger& r) { return l.getTriggerID() < r.getTriggerID(); };
  friend inline bool operator==(const Trigger& l, const Trigger& r) { return l.getTriggerID() == r.getTriggerID(); };
  friend inline bool operator>(const Trigger& l, const Trigger& r) { return r < l; };
  friend inline bool operator<=(const Trigger& l, const Trigger& r) { return !(l > r); };
  friend inline bool operator>=(const Trigger& l, const Trigger& r) { return !(l < r); };
  friend inline bool operator!=(const Trigger& l, const Trigger& r) { return !(l == r); };

  // Digit ASCII format (Orbit,BunchCrossing)[LHC Time nSec]
  friend std::ostream& operator<<(std::ostream& os, const Trigger& d);

 private:
  // Members
  InteractionRecord mIr;
  DataRange mDataRange; /// Index of the triggering event (event index and first entry in the container)

  ClassDefNV(Trigger, 2);
};

} // namespace hmpid
} // namespace o2

#endif /* DETECTORS_HMPID_BASE_INCLUDE_HMPIDBASE_TRIGGER_H_ */
