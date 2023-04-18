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

/// @brief  Interaction record encoding BC, orbit, time

#ifndef ALICEO2_INTERACTIONRECORD_H
#define ALICEO2_INTERACTIONRECORD_H

#include "GPUCommonRtypes.h"
#ifndef GPUCA_ALIGPUCODE
#include <iosfwd>
#include <cstdint>
#endif
#include <cmath>
#include "CommonConstants/LHCConstants.h"

namespace o2
{
struct InteractionRecord {
  // information about bunch crossing and orbit
  static constexpr uint16_t DummyBC = 0xffff;
  static constexpr uint32_t DummyOrbit = 0xffffffff;
  static constexpr double DummyTime = DummyBC * o2::constants::lhc::LHCBunchSpacingNS + DummyOrbit * o2::constants::lhc::LHCOrbitNS;
  static constexpr int64_t MaxGlobalBCs = (int64_t(DummyOrbit) * o2::constants::lhc::LHCMaxBunches) + (o2::constants::lhc::LHCMaxBunches - 1);
  static constexpr InteractionRecord getIRMaxBC() { return {o2::constants::lhc::LHCMaxBunches - 1, DummyOrbit}; }
  uint16_t bc = DummyBC;       ///< bunch crossing ID of interaction
  uint32_t orbit = DummyOrbit; ///< LHC orbit

  constexpr InteractionRecord() = default;

  InteractionRecord(double tNS)
  {
    setFromNS(tNS);
  }

  constexpr InteractionRecord(uint16_t b, uint32_t orb) : bc(b), orbit(orb)
  {
  }

  InteractionRecord(const InteractionRecord& src) = default;
  InteractionRecord& operator=(const InteractionRecord& src) = default;

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

  double bc2ns() const
  {
    return bc2ns(bc, orbit);
  }

  bool operator==(const InteractionRecord& other) const
  {
    return (bc == other.bc) && (orbit == other.orbit);
  }

  bool operator!=(const InteractionRecord& other) const
  {
    return (bc != other.bc) || (orbit != other.orbit);
  }

  int64_t differenceInBC(const InteractionRecord& other) const
  {
    // return difference in bunch-crossings
    int64_t diffBC = int(bc) - other.bc;
    if (orbit != other.orbit) {
      diffBC += (int64_t(orbit) - other.orbit) * o2::constants::lhc::LHCMaxBunches;
    }
    return diffBC;
  }

  float differenceInBCNS(const InteractionRecord& other) const
  {
    // return difference in bunch-crossings in ns
    return differenceInBC(other) * o2::constants::lhc::LHCBunchSpacingNS;
  }

  float differenceInBCMUS(const InteractionRecord& other) const
  {
    // return difference in bunch-crossings in ms
    return differenceInBC(other) * o2::constants::lhc::LHCBunchSpacingMUS;
  }

  int64_t toLong() const
  {
    // return as single long number
    return (int64_t(orbit) * o2::constants::lhc::LHCMaxBunches) + bc;
  }

  void setFromLong(int64_t l)
  {
    // set from long BC counter
    bc = l % o2::constants::lhc::LHCMaxBunches;
    orbit = l / o2::constants::lhc::LHCMaxBunches;
  }

  static InteractionRecord long2IR(int64_t l)
  {
    // set from long BC counter
    return {uint16_t(l % o2::constants::lhc::LHCMaxBunches), uint32_t(l / o2::constants::lhc::LHCMaxBunches)};
  }

  bool operator>(const InteractionRecord& other) const
  {
    return (orbit == other.orbit) ? (bc > other.bc) : (orbit > other.orbit);
  }

  bool operator>=(const InteractionRecord& other) const
  {
    return !((*this) < other);
  }

  bool operator<(const InteractionRecord& other) const
  {
    return (orbit == other.orbit) ? (bc < other.bc) : (orbit < other.orbit);
  }

  bool operator<=(const InteractionRecord& other) const
  {
    return !((*this) > other);
  }

  InteractionRecord operator--()
  {
    // prefix decrement operator
    if (!bc--) {
      orbit--;
      bc = o2::constants::lhc::LHCMaxBunches - 1;
      if (orbit == DummyOrbit) { // wrapped?
        orbit = 0;
        bc = 0;
      }
    }
    return InteractionRecord(*this);
  }

  InteractionRecord operator--(int)
  {
    // postfix decrement operator, no check for orbit wrap
    InteractionRecord tmp(*this);
    if (!bc--) {
      orbit--;
      bc = o2::constants::lhc::LHCMaxBunches - 1;
      if (orbit == DummyOrbit) { // wrapped?
        orbit = 0;
        bc = 0;
      }
    }
    return tmp;
  }

  InteractionRecord operator++()
  {
    // prefix increment operator
    if ((++bc) == o2::constants::lhc::LHCMaxBunches) {
      orbit++;
      bc = 0;
      if (orbit == 0) { // wrapped?
        orbit = DummyOrbit;
        bc = o2::constants::lhc::LHCMaxBunches - 1;
      }
    }
    return InteractionRecord(*this);
  }

  InteractionRecord operator++(int)
  {
    // postfix increment operator
    InteractionRecord tmp(*this);
    if ((++bc) == o2::constants::lhc::LHCMaxBunches) {
      orbit++;
      bc = 0;
      if (orbit == 0) { // wrapped?
        orbit = DummyOrbit;
        bc = o2::constants::lhc::LHCMaxBunches - 1;
      }
    }
    return tmp;
  }

  InteractionRecord& operator+=(int64_t dbc)
  {
    // bc self-addition operator, avoid wrapping
    auto l = toLong();
    if (dbc >= 0) {
      if (MaxGlobalBCs - dbc < l) {
        l = MaxGlobalBCs;
        dbc = 0;
      }
    } else {
      if (l < -dbc) {
        l = 0;
        dbc = 0;
      }
    }
    l += dbc;
    bc = l % o2::constants::lhc::LHCMaxBunches;
    orbit = l / o2::constants::lhc::LHCMaxBunches;
    return *this;
  }

  InteractionRecord& operator-=(int64_t dbc)
  {
    // bc self-subtraction operator
    return operator+=(-dbc);
  }

  InteractionRecord& operator+=(const InteractionRecord& add)
  {
    // InteractionRecord self-addition operator
    return operator+=(add.toLong());
  }

  InteractionRecord& operator-=(const InteractionRecord& add)
  {
    // InteractionRecord self-subtraction operator
    return operator-=(add.toLong());
  }

  InteractionRecord operator+(int64_t dbc) const
  {
    // bc addition operator
    InteractionRecord tmp(*this);
    tmp += dbc;
    return tmp;
  }

  InteractionRecord operator-(int64_t dbc) const
  {
    // bc subtraction operator
    InteractionRecord tmp(*this);
    tmp -= dbc;
    return tmp;
  }

  InteractionRecord operator+(const InteractionRecord& add) const
  {
    // InteractionRecord addition operator, no check for orbit wrap
    InteractionRecord tmp(*this);
    tmp += add;
    return tmp;
  }

  InteractionRecord operator-(const InteractionRecord& add) const
  {
    // InteractionRecord subtraction operator, no check for orbit wrap
    InteractionRecord tmp(*this);
    tmp -= add;
    return tmp;
  }

#ifndef GPUCA_ALIGPUCODE
  void print() const;
  std::string asString() const;
  friend std::ostream& operator<<(std::ostream& stream, InteractionRecord const& ir);
#endif
  ClassDefNV(InteractionRecord, 3);
};

struct InteractionTimeRecord : public InteractionRecord {
  double timeInBCNS = 0.; ///< time in NANOSECONDS relative to orbit/bc

  InteractionTimeRecord() = default;

  /// create from the interaction record and time in the bunch (in ns)
  InteractionTimeRecord(const InteractionRecord& ir, double t_in_bc) : InteractionRecord(ir), timeInBCNS(t_in_bc)
  {
  }

  /// create from the abs. (since orbit=0/bc=0) time in NS
  InteractionTimeRecord(double tNS) : InteractionRecord(tNS)
  {
    timeInBCNS = tNS - bc2ns();
  }

  /// set the from the abs. (since orbit=0/bc=0) time in NS
  void setFromNS(double tNS)
  {
    InteractionRecord::setFromNS(tNS);
    timeInBCNS = tNS - bc2ns();
  }

  void clear()
  {
    InteractionRecord::clear();
    timeInBCNS = 0.;
  }

  double getTimeOffsetWrtBC() const
  {
    return timeInBCNS;
  }

  /// get time in ns from orbit=0/bc=0
  double getTimeNS() const
  {
    return timeInBCNS + bc2ns();
  }

  bool operator==(const InteractionTimeRecord& other) const
  {
    return this->InteractionRecord::operator==(other) && (timeInBCNS == other.timeInBCNS);
  }

  bool operator!=(const InteractionTimeRecord& other) const
  {
    return this->InteractionRecord::operator!=(other) || (timeInBCNS != other.timeInBCNS);
  }

  bool operator>(const InteractionTimeRecord& other) const
  {
    return (this->InteractionRecord::operator>(other)) || (this->InteractionRecord::operator==(other) && (timeInBCNS > other.timeInBCNS));
  }

  bool operator>=(const InteractionTimeRecord& other) const
  {
    return !((*this) < other);
  }

  bool operator<(const InteractionTimeRecord& other) const
  {
    return (this->InteractionRecord::operator<(other)) || (this->InteractionRecord::operator==(other) && (timeInBCNS < other.timeInBCNS));
  }

  bool operator<=(const InteractionTimeRecord& other) const
  {
    return !((*this) > other);
  }

#ifndef GPUCA_ALIGPUCODE
  void print() const;
  std::string asString() const;
  friend std::ostream& operator<<(std::ostream& stream, InteractionTimeRecord const& ir);
#endif

  ClassDefNV(InteractionTimeRecord, 1);
};
} // namespace o2

namespace std
{
// defining std::hash for InteractionRecord to be used with std containers
template <>
struct hash<o2::InteractionRecord> {
 public:
  size_t operator()(const o2::InteractionRecord& ir) const
  {
    return ir.toLong();
  }
};
} // namespace std

#endif
