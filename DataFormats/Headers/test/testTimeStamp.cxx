// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test TimeStamp class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <chrono>
#include "Headers/TimeStamp.h"

namespace o2
{
namespace header
{

BOOST_AUTO_TEST_CASE(LHCClock_test)
{
  // 32 bit number for the tick if the clock precision
  // is on the orbit level
  BOOST_CHECK(sizeof(LHCOrbitClock::rep) == 4);

  float orbitPeriod = LHCOrbitClock::period::num;
  orbitPeriod /= LHCOrbitClock::period::den;
  BOOST_CHECK(orbitPeriod > 0.000085 && orbitPeriod < 0.000091);

  // 64 bit number for the tick if the clock precision
  // is on the bunch level
  BOOST_CHECK(sizeof(LHCBunchClock::rep) == 8);

  float bunchPeriod = LHCBunchClock::period::num;
  bunchPeriod /= LHCBunchClock::period::den;
  BOOST_CHECK(bunchPeriod > 0.0000000249 && bunchPeriod < 0.0000000251);

  // duration comparison
  LHCOrbitClock::duration fourOrbits(4);
  BOOST_CHECK(fourOrbits == std::chrono::nanoseconds(4 * lhc_clock_parameter::gOrbitTimeNanoSec));
}

BOOST_AUTO_TEST_CASE(TimeStamp_test)
{
  // check the TimeStamp member layout with respect to manually
  // assembled 64 bit field
  // 'AC' (accelerator) specifies the LHCClock
  // 40404040 orbits are almost 1h
  // together with 1512 bunches we pass 1h
  // TODO: extend from fixed values to random values
  uint64_t bunches = 1512;
  uint64_t orbits = 40404040;
  uint64_t ts64 = String2<uint16_t>("AC") | bunches << 16 | orbits << 32;
  TimeStamp timestamp("AC", orbits, bunches);

  // using the type cast operator
  BOOST_CHECK(ts64 == timestamp);

  // checking cast to LHCClock
  auto timeInLHCOrbitClock = timestamp.get<LHCOrbitClock>();
  auto timeInLHCBunchClock = timestamp.get<LHCBunchClock>();
  BOOST_CHECK(timeInLHCOrbitClock.count() == orbits);
  BOOST_CHECK(timeInLHCOrbitClock == LHCOrbitClock::duration(orbits));

  // time in LHCOrbitClock cuts off the bunches
  BOOST_CHECK(timeInLHCOrbitClock <= timeInLHCBunchClock);

  // get explicitely the time ignoring bunch counter and cast to seconds
  // that must be less then 1h
  auto timeInSeconds = std::chrono::duration_cast<std::chrono::seconds>(timestamp.get<LHCOrbitClock>());
  BOOST_CHECK(timeInSeconds < std::chrono::hours(1));

  // directly retrieving time in hours takes both orbits and bunches for
  // internal calculation before casting to hours
  auto timeInHours = timestamp.get<std::chrono::hours>();
  BOOST_CHECK(timeInHours.count() == 1);

  // setting timestamp with a value in unit mico seconds
  uint64_t tenSeconds = 10000000;
  timestamp = String2<uint16_t>("US") | tenSeconds << 32;

  // check conversion of the us value to LHCClock
  auto timeInOrbitPrecision = timestamp.get<LHCOrbitClock>();
  auto timeInBunchPrecision = timestamp.get<LHCBunchClock>();
  uint64_t expectedOrbits = tenSeconds * 1000 / (lhc_clock_parameter::gNumberOfBunches * lhc_clock_parameter::gBunchSpacingNanoSec);
  BOOST_CHECK(timeInOrbitPrecision.count() == expectedOrbits);

  // conversion in orbit precision ignores bunches
  BOOST_CHECK(timeInOrbitPrecision <= timeInBunchPrecision);
}
} // namespace header
} // namespace o2
