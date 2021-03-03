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
#include "TMath.h"
#include "CommonDataFormat/TimeStamp.h"
#include "CommonConstants/LHCConstants.h"


namespace o2
{
namespace hmpid
{

/// \class Trigger
/// \brief HMPID Trigger declaration
class Trigger
{
 public:
  // Trigger time Conversion Functions
  static inline uint64_t OrbitBcToEventId(uint32_t Orbit, uint16_t BC) { return ((Orbit << 12) | (0x0FFF & BC)); };
  static inline uint32_t EventIdToOrbit(uint64_t EventId) { return (EventId >> 12); };
  static inline uint16_t EventIdToBc(uint64_t EventId) { return (EventId & 0x0FFF); };
  static double OrbitBcToTimeNs(uint32_t Orbit, uint16_t BC) { return (BC * o2::constants::lhc::LHCBunchSpacingNS + Orbit * o2::constants::lhc::LHCOrbitNS);};
  static uint32_t TimeNsToOrbit(double TimeNs) { return (uint32_t)(TimeNs/o2::constants::lhc::LHCOrbitNS);};
  static uint16_t TimeNsToBc(double TimeNs) { return (uint16_t)(std::fmod(TimeNs, o2::constants::lhc::LHCOrbitNS) / o2::constants::lhc::LHCBunchSpacingNS);};
  static void TimeNsToOrbitBc(double TimeNs, uint32_t& Orbit, uint16_t& Bc)
  {
    Orbit = TimeNsToOrbit(TimeNs);
    Bc = TimeNsToBc(TimeNs);
    return;
  };

  // Operators definition !
  friend inline bool operator<(const Trigger& l, const Trigger& r) { return OrbitBcToEventId(l.mOrbit,l.mBc) < OrbitBcToEventId(r.mOrbit, r.mBc); };
  friend inline bool operator==(const Trigger& l, const Trigger& r) { return OrbitBcToEventId(l.mOrbit,l.mBc) == OrbitBcToEventId(r.mOrbit, r.mBc); };
  friend inline bool operator>(const Trigger& l, const Trigger& r) { return r < l; };
  friend inline bool operator<=(const Trigger& l, const Trigger& r) { return !(l > r); };
  friend inline bool operator>=(const Trigger& l, const Trigger& r) { return !(l < r); };
  friend inline bool operator!=(const Trigger& l, const Trigger& r) { return !(l == r); };

  // Digit ASCII format (Orbit,BunchCrossing)[LHC Time nSec]
  friend std::ostream& operator<<(std::ostream& os, const Trigger& d)
  {
    os << "(" << d.mOrbit << "," << d.mBc << ")[" << OrbitBcToTimeNs(d.mOrbit,d.mBc) << " ns]" ;
    return os;
  };

 public:
  Trigger() = default;
  Trigger(uint16_t bc, uint32_t orbit) {
    mBc = bc;
    mOrbit = orbit;
  };
  uint32_t getOrbit() const { return mOrbit; }
  uint16_t getBc() const { return mBc; }
  uint64_t getTriggerID() const { return OrbitBcToEventId(mOrbit, mBc); }
  void setOrbit(uint32_t orbit)
  {
    mOrbit = orbit;
    return;
  }
  void setBC(uint16_t bc)
  {
    mBc = bc;
    return;
  }
  void setTriggerID(uint64_t trigger)
  {
    mOrbit = (trigger >> 12);
    mBc = (trigger & 0x0FFF);
    return;
  }
 private:
  // Members
  uint16_t mBc = 0.;
  uint32_t mOrbit = 0;

  ClassDefNV(Trigger, 2);
};

} // namespace hmpid
} // namespace o2

#endif /* DETECTORS_HMPID_BASE_INCLUDE_HMPIDBASE_TRIGGER_H_ */
