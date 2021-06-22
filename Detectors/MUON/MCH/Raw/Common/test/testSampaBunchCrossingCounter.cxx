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
/// @author  Laurent Aphecetche

#define BOOST_TEST_MODULE Test MCHRaw SampaBunchCrossingCounter
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "MCHRawCommon/SampaBunchCrossingCounter.h"
#include "MCHRawCommon/CoDecParam.h"

using namespace o2::mch::raw;

constexpr uint32_t BXMAX = (1 << 20) - 1;

BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(sampatime)

BOOST_AUTO_TEST_CASE(SampaBunchCrossingCounterToIRConversionMustThrowIfBxDoesNotFitIn20Bits)
{
  uint32_t bx = BXMAX + 1;
  BOOST_CHECK_THROW(orbitBC(bx, 0),
                    std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(SampaBunchCrossingCounterToIRConversionMustNotThrowIfBxFitIn20Bits)
{
  uint32_t bx = BXMAX;
  BOOST_CHECK_NO_THROW(orbitBC(bx, 0));
}

BOOST_AUTO_TEST_CASE(SampaBunchCrossingCounterSpansABitLessThan294Orbits)
{
  o2::conf::ConfigurableParam::setValue("MCHCoDecParam", "sampaBcOffset", 0);
  uint32_t bx = BXMAX;
  auto [orbit, bc] = orbitBC(bx, 0);
  BOOST_CHECK_EQUAL(orbit, 294);
  BOOST_CHECK_EQUAL(bc, 759);
}

BOOST_AUTO_TEST_CASE(Orbit294BC759MakesSampaBXCounterRolloverFromFirstOrbitZero)
{
  auto bx = sampaBunchCrossingCounter(294, 759, 0);
  BOOST_CHECK_EQUAL(bx, 0);
}

BOOST_AUTO_TEST_CASE(Orbit294BC759MakesSampaBXCounterRolloverFromAnyFirstOrbit)
{
  uint32_t firstOrbit = 12345;
  auto bx = sampaBunchCrossingCounter(294 + firstOrbit, 759, firstOrbit);
  BOOST_CHECK_EQUAL(bx, 0);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
