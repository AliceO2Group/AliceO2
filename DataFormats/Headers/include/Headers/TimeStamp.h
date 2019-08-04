// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//-*- Mode: C++ -*-
#ifndef TIMESTAMP_H
#define TIMESTAMP_H

/// @file   TimeStamp.h
/// @author Matthias Richter
/// @since  2017-01-25
/// @brief  A std chrono implementation of LHC clock and timestamp

#include "Headers/DataHeader.h"
#include <chrono>
#include <cassert>
#include <type_traits> // for std::integral_constant

namespace o2
{
namespace header
{

// https://lhc-machine-outreach.web.cern.ch/lhc-machine-outreach/collisions.htm
// https://www.lhc-closer.es/taking_a_closer_look_at_lhc/0.buckets_and_bunches

namespace lhc_clock_parameter
{
// number of bunches and the 40 MHz clock with 25 ns bunch spacing
// gives revolution time of 89.1 us and 11.223345 kHz
// this depends on the assumption that the particles are moving effectively
// at speed of light. There are also documents specifying the orbit time
// to 89.4 us
// Note: avoid to define the revolution frequency and use the integral numbers
// for bunch places and bunch spacing in nano seconds
// TODO: this eventually needs to be configurable
static constexpr int gNumberOfBunches = 3564;
static constexpr int gBunchSpacingNanoSec = 25;
static constexpr int gOrbitTimeNanoSec = std::ratio<gNumberOfBunches * gBunchSpacingNanoSec>::num;

using OrbitPrecision = std::integral_constant<int, 0>;
using BunchPrecision = std::integral_constant<int, 1>;

// the type of the clock tick depends on whether to use also the bunches
// as substructure of the orbit.
// a trait class to extrat the properties of the clock, namely the type
// of the tick and the period
template <typename T>
struct Property {
  // the default does not specify anything and is never going to be used
};

template <>
struct Property<OrbitPrecision> {
  using rep = uint32_t;
  // avoid rounding errors by using the integral numbers in the std::ratio
  // template to define the period
  using period = std::ratio_multiply<std::ratio<gOrbitTimeNanoSec>, std::nano>;
};

template <>
struct Property<BunchPrecision> {
  using rep = uint64_t;
  // this is effectively the LHC clock and the ratio is the
  // bunch spacing
  using period = std::ratio_multiply<std::ratio<gBunchSpacingNanoSec>, std::nano>;
};
} // namespace lhc_clock_parameter

// a chrono clock implementation
// - always relative to run start
// - need run start to calculate the epoch
// - based on revolution frequency and number of bunches
// TODO: the reference time is probably the start of the fill
template <typename RefTimePoint, typename Precision = lhc_clock_parameter::OrbitPrecision>
class LHCClock
{
 public:
  LHCClock(const RefTimePoint& start) : mReference(start) {}
  /// forbidden, always need a reference
  LHCClock() = delete;
  ~LHCClock() = default;
  LHCClock(const LHCClock&) = default;
  LHCClock& operator=(const LHCClock&) = default;

  using rep = typename lhc_clock_parameter::Property<Precision>::rep;
  using period = typename lhc_clock_parameter::Property<Precision>::period;
  using duration = std::chrono::duration<rep, period>;
  using time_point = std::chrono::time_point<LHCClock>;
  // this follows the naming convention of std chrono
  static const bool is_steady = true;

  /// the now() function is the main characteristics of the clock
  /// calculate now from the system clock and the reference start time
  time_point now() noexcept
  {
    // tp1 - tp2 results in a duration, we use to create a time_point with characteristics
    // of the clock.
    return time_point(std::chrono::duration_cast<duration>(std::chrono::system_clock::now()) - mReference);
  }

 private:
  /// external reference: start time of the run
  RefTimePoint mReference;
};

// TODO: is it correct to define this types always relative to the system clock?
using LHCOrbitClock = LHCClock<std::chrono::system_clock::time_point, lhc_clock_parameter::OrbitPrecision>;
using LHCBunchClock = LHCClock<std::chrono::system_clock::time_point, lhc_clock_parameter::BunchPrecision>;

class TimeStamp
{
 public:
  using TimeUnitID = o2::header::Descriptor<2>;
  // TODO: type aliases for the types of ticks and subticks

  TimeStamp() = default;
  TimeStamp(uint64_t ts) : mTimeStamp64(ts) {}
  TimeStamp(const TimeUnitID& unit, uint32_t tick, uint16_t subtick = 0)
    : mUnit(unit), mTicks(tick), mSubTicks(subtick) {}
  ~TimeStamp() = default;

  static TimeUnitID const sClockLHC;
  static TimeUnitID const sMicroSeconds;

  operator uint64_t() const { return mTimeStamp64; }

  /// get the duration in the units of the specified clock or duration type
  /// the template parameter can either be a clock or duration type following std::chrono concept
  template <class T, typename Rep = typename T::rep, typename Period = typename T::period>
  auto get() const
  {
    static_assert(std::is_same<typename T::rep, Rep>::value && std::is_same<typename T::period, Period>::value,
                  "only clock and duration types defining the rep and period member types are allowed");
    using duration = std::chrono::duration<Rep, Period>;
    if (mUnit == sClockLHC) {
      // cast each part individually, if the precision of the return type
      // is smaller the values are simply truncated
      return std::chrono::duration_cast<duration>(LHCOrbitClock::duration(mPeriod) + LHCBunchClock::duration(mBCNumber));
    }
    if (mUnit == sMicroSeconds) {
      // TODO: is there a better way to mark the subticks invalid for the
      // micro seconds representation? First step is probably to remove/rename the
      // variable
      assert(mSubTicks == 0);
      return std::chrono::duration_cast<duration>(std::chrono::microseconds(mTicks));
    }
    // invalid time unit identifier
    // TODO: define error policy
    assert(0);
    return std::chrono::duration_cast<duration>(std::chrono::seconds(0));
  }

  // TODO: implement transformation from one unit to the other
  //void transform(const TimeUnitID& unit) {
  //  if (mUnit == unit) return;
  //  ...
  //}

 private:
  union {
    uint64_t mTimeStamp64;
    struct {
      uint16_t mUnit;
      // the unions are probably not a good idea as the members have too different
      // meaning depending on the unit, but take it as a fist working assumption
      union {
        uint16_t mBCNumber;
        uint16_t mSubTicks;
      };
      union {
        uint32_t mPeriod;
        uint32_t mTicks;
      };
    };
  };
};
} //namespace header
} //namespace o2

#endif
