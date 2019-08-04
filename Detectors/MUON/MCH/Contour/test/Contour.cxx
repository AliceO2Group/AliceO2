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

#define BOOST_TEST_MODULE Test MCHContour Contour
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>
#include "../include/MCHContour/Contour.h"

using namespace o2::mch::contour;

BOOST_AUTO_TEST_SUITE(o2_mch_contour)

BOOST_AUTO_TEST_SUITE(contour)

BOOST_AUTO_TEST_CASE(ContourAreEqualAsLongAsTheyContainTheSameSetOfVertices)
{
  Contour<double> aCollectionWithOnePolygon{{{0, 2}, {0, 0}, {2, 0}, {2, 4}, {1, 4}, {1, 2}, {0, 2}}};

  Contour<double> anotherCollectionWithTwoPolygonsButSameVertices{
    {{2, 4}, {2, 0}}, {{1, 4}, {1, 2}, {0, 2}, {0, 0}}

  };

  BOOST_CHECK(aCollectionWithOnePolygon == anotherCollectionWithTwoPolygonsButSameVertices);
}
BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
