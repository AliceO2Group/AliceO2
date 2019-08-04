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

#define BOOST_TEST_MODULE Test MCHContour Interval
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <iostream>
#include "../include/MCHContour/Interval.h"

using namespace o2::mch::contour::impl;

BOOST_AUTO_TEST_SUITE(o2_mch_contour)

BOOST_AUTO_TEST_SUITE(interval)

BOOST_AUTO_TEST_CASE(IntervalCtorThrowsIfBeginIsAfterEnd)
{
  BOOST_CHECK_THROW(Interval<int> a(24, 3), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(IntervalCtorThrowsIfBeginEqualsEnd)
{
  BOOST_CHECK_THROW(Interval<double> b(24.24, 24.24), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(IntIntervalIsFullyContainedInInterval)
{
  Interval<int> i{1, 5};
  BOOST_CHECK_EQUAL(Interval<int>(0, 4).isFullyContainedIn(i), false);
  BOOST_CHECK_EQUAL(Interval<int>(1, 2).isFullyContainedIn(i), true);
}

BOOST_AUTO_TEST_CASE(DoubleIntervalIsFullyContainedInInterval)
{
  Interval<double> f{0.01, 0.05};
  BOOST_CHECK_EQUAL(Interval<double>(0, 0.04).isFullyContainedIn(f), false);
  BOOST_CHECK_EQUAL(Interval<double>(0.01, 0.02).isFullyContainedIn(f), true);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
