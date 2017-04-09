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

namespace o2 {
namespace Header {

// https://lhc-machine-outreach.web.cern.ch/lhc-machine-outreach/collisions.htm
// https://www.lhc-closer.es/taking_a_closer_look_at_lhc/0.buckets_and_bunches

namespace LHCClockParameter {
  // number of bunches and the 40 MHz clock with 25 ns bunch spacing
  // gives revolution time of 89.1 us and 11.223345 kHz
  // this depends on the assumption that the particles are moving effectively
  // at speed of light. There are also documents specifying the orbit time
  // to 89.4 us
  // Note: avoid to define the revolution frequency and use the integral numbers
  // for bunch places and bunch spacing in nano seconds
  // TODO: this eventually needs to be configurable
  const int gNumberOfBunches = 3564;
  const int gBunchSpacingNanoSec = 25;
  const int gOrbitTimeNanoSec = std::ratio<gNumberOfBunches*gBunchSpacingNanoSec>::num;

  typedef o2::Header::Internal::intWrapper<0> OrbitPrecision;
  typedef o2::Header::Internal::intWrapper<1> BunchPrecision;

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
    typedef uint32_t rep;
    // avoid rounding errors by using the integral numbers in the std::ratio
    // template to define the period
    typedef std::ratio_multiply<std::ratio<gOrbitTimeNanoSec>, std::nano> period;
  };
  template <>
  struct Property<BunchPrecision> {
    typedef uint64_t rep;
    // this is effectively the LHC clock and the ratio is the
    // bunch spacing
    typedef std::ratio_multiply<std::ratio<gBunchSpacingNanoSec>, std::nano> period;
  };
};

// a chrono clock implementation
// - always relative to run start
// - need run start to calculate the epoch
// - based on revolution frequency and number of bunches
// TODO: the reference time is probably the start of the fill
template <typename RefTimePoint, typename Precision = LHCClockParameter::OrbitPrecision>
class LHCClock {
public:
  LHCClock(const RefTimePoint& start) : mReference(start) {}
  /// forbidden, always need a reference
  LHCClock() = delete;
  ~LHCClock() = default;
  LHCClock(const LHCClock&) = default;
  LHCClock& operator=(const LHCClock&) = default;

  typedef typename LHCClockParameter::Property<Precision>::rep    rep;
  typedef typename LHCClockParameter::Property<Precision>::period period;
  typedef std::chrono::duration<rep, period> duration;
  typedef std::chrono::time_point<LHCClock>  time_point;
  // this follows the naming convention of std chrono
  static const bool is_steady =              true;

  /// the now() function is the main characteristics of the clock
  /// calculate now from the system clock and the reference start time
  time_point now() noexcept {
    // tp1 - tp2 results in a duration, we use to create a time_point with characteristics
    // of the clock.
    return time_point(std::chrono::duration_cast<duration>(std::chrono::system_clock::now()) - mReference);
  }

private:
  /// external reference: start time of the run
  RefTimePoint mReference;
};

// TODO: is it correct to define this types always relative to the system clock?
typedef LHCClock<std::chrono::system_clock::time_point, LHCClockParameter::OrbitPrecision> LHCOrbitClock;
typedef LHCClock<std::chrono::system_clock::time_point, LHCClockParameter::BunchPrecision> LHCBunchClock;

class TimeStamp
{
 public:
  typedef o2::Header::Descriptor<2> TimeUnitID;
  // TODO: typedefs for the types of ticks and subticks

  TimeStamp() = default;
  TimeStamp(uint64_t ts) : mTimeStamp64(ts) {}
  TimeStamp(const TimeUnitID& unit, uint32_t tick, uint16_t subtick = 0)
    : mUnit(unit), mTicks(tick), mSubTicks(subtick) {}
  ~TimeStamp() = default;

  static TimeUnitID const sClockLHC;
  static TimeUnitID const sMicroSeconds;

  operator uint64_t() const {return mTimeStamp64;}

  template<class Clock>
  typename Clock::duration get() const {
    typedef typename Clock::duration duration;
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
} //namespace Header
} //namespace AliceO2

#endif
