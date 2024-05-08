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

#define BOOST_TEST_MODULE Test quality_control FlagTypes class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <type_traits>
#include <iostream>

// o2 includes
#include "DataFormatsQualityControl/FlagType.h"
#include "DataFormatsQualityControl/FlagTypeFactory.h"

using namespace o2::quality_control;

BOOST_AUTO_TEST_CASE(FlagTypes)
{
  static_assert(std::is_constructible<FlagType, uint16_t, const char*, bool>::value == false,
                "FlagType should not be constructible outside of its static methods and the factory.");

  FlagType fDefault;
  BOOST_CHECK_EQUAL(fDefault, FlagTypeFactory::Invalid());

  auto f1 = FlagTypeFactory::Unknown();
  BOOST_CHECK_EQUAL(f1.getID(), 14);
  BOOST_CHECK_EQUAL(f1.getName(), "Unknown");
  BOOST_CHECK_EQUAL(f1.getBad(), true);

  BOOST_CHECK_NO_THROW(std::cout << f1 << std::endl);

  auto f2 = f1;
  BOOST_CHECK_EQUAL(f2.getID(), 14);
  BOOST_CHECK_EQUAL(f1.getName(), f2.getName());
  BOOST_CHECK_EQUAL(f2.getName(), "Unknown");
  BOOST_CHECK_EQUAL(f2.getBad(), true);

  BOOST_CHECK_EQUAL(f1, f2);
  BOOST_CHECK((f1 != f2) == false);
  BOOST_CHECK(!(f1 < f2));
  BOOST_CHECK(!(f1 > f2));

  auto f3 = FlagTypeFactory::LimitedAcceptanceMCNotReproducible();
  BOOST_CHECK(f3 < f1);
  BOOST_CHECK(!(f3 > f1));
}
