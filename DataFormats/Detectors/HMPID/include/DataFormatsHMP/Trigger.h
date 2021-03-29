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

#include <vector>
#include <iosfwd>
#include <iostream>
//#include "CommonDataFormat/TimeStamp.h"
//#include "CommonConstants/LHCConstants.h"
#include "CommonDataFormat/InteractionRecord.h"

namespace o2
{
namespace hmpid
{
/// \class Trigger
/// \brief HMPID Trigger declaration
class Event
{
 public:
  static inline uint64_t getTriggerID(uint32_t Orbit, uint16_t BC) { return ((Orbit << 12) | (0x0FFF & BC)); };

 public:
  Event() = default;
  Event(InteractionRecord ir, int32_t first, int32_t last)
  {
    mIr.bc = ir.bc;
    mIr.orbit = ir.orbit;
    mFirstDigit = first;
    mLastDigit = last;
  };
  InteractionRecord& getIr() { return mIr; };
  uint32_t getOrbit() const { return mIr.orbit; };
  uint16_t getBc() const { return mIr.bc; };
  uint64_t getTriggerID() const { return ((mIr.orbit << 12) | (0x0FFF & mIr.bc)); };
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
  void setDigitsPointer(int32_t first, int32_t last)
  {
    mFirstDigit = first;
    mLastDigit = last;
    return;
  }

  // Operators definition !
  friend inline bool operator<(const Event& l, const Event& r) { return l.getTriggerID() < r.getTriggerID(); };
  friend inline bool operator==(const Event& l, const Event& r) { return l.getTriggerID() == r.getTriggerID(); };
  friend inline bool operator>(const Event& l, const Event& r) { return r < l; };
  friend inline bool operator<=(const Event& l, const Event& r) { return !(l > r); };
  friend inline bool operator>=(const Event& l, const Event& r) { return !(l < r); };
  friend inline bool operator!=(const Event& l, const Event& r) { return !(l == r); };

  // Digit ASCII format (Orbit,BunchCrossing)[LHC Time nSec]
  friend std::ostream& operator<<(std::ostream& os, const Event& d)
  {
    os << "(" << d.mIr.orbit << "," << d.mIr.bc << " @ " << d.mIr.bc2ns() << " ns) [" << d.mFirstDigit << " .. " << d.mLastDigit << "]";
    return os;
  };

 public:
  int32_t mFirstDigit = 0;
  int32_t mLastDigit = -1;

 private:
  // Members
  InteractionRecord mIr;

  ClassDefNV(Event, 2);
};

} // namespace hmpid
} // namespace o2

#endif /* DETECTORS_HMPID_BASE_INCLUDE_HMPIDBASE_TRIGGER_H_ */
