// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @brief  Interaction record encoding BC, orbit, time

#ifndef ALICEO2_INTERACTIONRECORD_H
#define ALICEO2_INTERACTIONRECORD_H

#include <Rtypes.h>
#include <iosfwd>
#include <cmath>
#include <cstdint>
#include "CommonConstants/LHCConstants.h"

namespace o2
{
struct InteractionRecord {
  // information about bunch crossing and orbit
  static constexpr uint16_t DummyBC = 0xffff;
  static constexpr uint32_t DummyOrbit = 0xffffffff;
  static constexpr double DummyTime = DummyBC * o2::constants::lhc::LHCBunchSpacingNS + DummyOrbit * o2::constants::lhc::LHCOrbitNS;

  uint16_t bc = DummyBC;       ///< bunch crossing ID of interaction
  uint32_t orbit = DummyOrbit; ///< LHC orbit

  InteractionRecord() = default;

  InteractionRecord(double tNS)
  {
    setFromNS(tNS);
  }

  InteractionRecord(uint16_t b, uint32_t orb) : bc(b), orbit(orb)
  {
  }

  void clear()
  {
    bc = 0xffff;
    orbit = 0xffffffff;
  }

  bool isDummy() const
  {
    return bc > o2::constants::lhc::LHCMaxBunches;
  }

  void setFromNS(double ns)
  {
    bc = ns2bc(ns, orbit);
  }

  static double bc2ns(int bc, unsigned int orbit)
  {
    return bc * o2::constants::lhc::LHCBunchSpacingNS + orbit * o2::constants::lhc::LHCOrbitNS;
  }

  static int ns2bc(double ns, unsigned int& orb)
  {
    orb = ns > 0 ? ns / o2::constants::lhc::LHCOrbitNS : 0;
    ns -= orb * o2::constants::lhc::LHCOrbitNS;
    return std::round(ns / o2::constants::lhc::LHCBunchSpacingNS);
  }

  bool operator==(const InteractionRecord& other) const
  {
    return (bc == other.bc) && (orbit == other.orbit);
  }

  bool operator!=(const InteractionRecord& other) const
  {
    return (bc != other.bc) || (orbit != other.orbit);
  }

  int differenceInBC(const InteractionRecord& other) const
  {
    // return differenc in bunch-crossings
    int diffBC = int(bc) - other.bc;
    if (orbit != other.orbit) {
      diffBC += (int(orbit) - other.orbit) * o2::constants::lhc::LHCMaxBunches;
    }
    return diffBC;
  }

  int64_t toLong() const
  {
    // return as single long number
    return (int64_t(orbit) * o2::constants::lhc::LHCMaxBunches) + bc;
  }

  bool operator>(const InteractionRecord& other) const
  {
    return (orbit == other.orbit) ? (bc > other.bc) : (orbit > other.orbit);
  }

  bool operator<(const InteractionRecord& other) const
  {
    return (orbit == other.orbit) ? (bc < other.bc) : (orbit < other.orbit);
  }

  void print() const;

  friend std::ostream& operator<<(std::ostream& stream, InteractionRecord const& ir);

  ClassDefNV(InteractionRecord, 3);
};

struct InteractionTimeRecord : public InteractionRecord {
  double timeNS = 0.; ///< time in NANOSECONDS from start of run (orbit=0)

  InteractionTimeRecord() = default;

  InteractionTimeRecord(const InteractionRecord& ir, double tNS) : InteractionRecord(ir), timeNS(tNS)
  {
  }

  InteractionTimeRecord(double tNS)
  {
    setFromNS(tNS);
  }

  void setFromNS(double ns)
  {
    timeNS = ns;
    InteractionRecord::setFromNS(ns);
  }

  void clear()
  {
    InteractionRecord::clear();
    timeNS = 0.;
  }

  void print() const;

  friend std::ostream& operator<<(std::ostream& stream, InteractionTimeRecord const& ir);

  ClassDefNV(InteractionTimeRecord, 1);
};
}

#endif
