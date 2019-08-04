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

#define BOOST_TEST_MODULE Test MCHContour BBox
#define BOOST_TEST_MAIN

#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <iostream>
#include "../include/MCHContour/BBox.h"

using namespace o2::mch::contour;

BOOST_AUTO_TEST_SUITE(o2_mch_contour)

BOOST_AUTO_TEST_SUITE(bbox)

BOOST_AUTO_TEST_CASE(BBoxMustBeCreatedValid) { BOOST_CHECK_THROW(BBox<int>(2, 2, 0, 0), std::invalid_argument); }

BOOST_AUTO_TEST_CASE(CheckBBoxBoundaries)
{
  BBox<double> test{-15.0, -10.0, 5.0, 20.0};
  BOOST_TEST(test.xmin() == -15.0);
  BOOST_TEST(test.xmax() == 5.0);
  BOOST_TEST(test.ymin() == -10.0);
  BOOST_TEST(test.ymax() == 20.0);
}

BOOST_AUTO_TEST_CASE(CheckBBoxCenter)
{
  BBox<double> test{-15.0, -10.0, 5.0, 20.0};
  BOOST_TEST(test.xcenter() == -5.0);
  BOOST_TEST(test.ycenter() == 5.0);
}

BOOST_AUTO_TEST_CASE(Intersect)
{
  BBox<double> one{0.0, 0.0, 4.0, 2.0};
  BBox<double> two{2.0, -1.0, 5.0, 1.0};
  BBox<double> expected{2.0, 0.0, 4.0, 1.0};
  BOOST_TEST(intersect(one, two) == expected);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
