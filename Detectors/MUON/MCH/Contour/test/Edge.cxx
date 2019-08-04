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

#define BOOST_TEST_MODULE Test MCHContour Edge
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>
#include "../include/MCHContour/Edge.h"

using namespace o2::mch::contour::impl;
using namespace o2::mch::contour;

BOOST_AUTO_TEST_SUITE(o2_mch_contour)

BOOST_AUTO_TEST_SUITE(edge)

BOOST_AUTO_TEST_CASE(AVerticalLeftEdgeIsTopToBottom)
{
  double dummy{0};
  VerticalEdge<double> edge{dummy, 12, 1};
  BOOST_CHECK(isLeftEdge(edge));
  BOOST_CHECK(isTopToBottom(edge));
}

BOOST_AUTO_TEST_CASE(AVerticalRightEdgeIsBottomToTop)
{
  double dummy{0};
  VerticalEdge<double> edge{dummy, 1, 12};
  BOOST_CHECK(isRightEdge(edge));
  BOOST_CHECK(isBottomToTop(edge));
}

BOOST_AUTO_TEST_CASE(ALeftToRightHorizontalEdgeHasEndPointGreaterThanStartPoint)
{
  double dummy{0};
  HorizontalEdge<double> edge{dummy, 1, 12};
  BOOST_CHECK(isLeftToRight(edge));
}

BOOST_AUTO_TEST_CASE(ARightToLeftHorizontalEdgeHasEndPointSmallerThanStartPoint)
{
  double dummy{0};
  HorizontalEdge<double> edge{dummy, 12, 1};
  BOOST_CHECK(isRightToLeft(edge));
}

BOOST_AUTO_TEST_CASE(AVerticalEdgeWithBeginAboveEndIsALefty)
{
  VerticalEdge<double> vi{0, 12, 10};
  BOOST_CHECK_EQUAL(isLeftEdge(vi), true);
  BOOST_CHECK_EQUAL(isRightEdge(vi), false);
}

BOOST_AUTO_TEST_CASE(AVerticalEdgeWithBeginAboveEndIsARighty)
{
  VerticalEdge<double> vi{0, 10, 12};
  BOOST_CHECK_EQUAL(isRightEdge(vi), true);
  BOOST_CHECK_EQUAL(isLeftEdge(vi), false);
}

BOOST_AUTO_TEST_CASE(AVerticalEdgeHasATopAndBottom)
{
  VerticalEdge<double> edge{2, 10, 12};
  BOOST_CHECK_EQUAL(bottom(edge), 10);
  BOOST_CHECK_EQUAL(top(edge), 12);
}

BOOST_AUTO_TEST_CASE(BeginAndEndForALeftEdgeVertical)
{
  VerticalEdge<double> e{0, 7, 1};

  BOOST_CHECK_EQUAL(e.begin(), (Vertex<double>{0, 7}));
  BOOST_CHECK_EQUAL(e.end(), (Vertex<double>{0, 1}));
  BOOST_CHECK_EQUAL(top(e), 7);
  BOOST_CHECK_EQUAL(bottom(e), 1);
}

BOOST_AUTO_TEST_CASE(BeginAndEndForARightEdgeVertical)
{
  VerticalEdge<double> e{0, 1, 7};

  BOOST_CHECK_EQUAL(e.begin(), (Vertex<double>{0, 1}));
  BOOST_CHECK_EQUAL(e.end(), (Vertex<double>{0, 7}));
  BOOST_CHECK_EQUAL(top(e), 7);
  BOOST_CHECK_EQUAL(bottom(e), 1);
}

BOOST_AUTO_TEST_CASE(BeginAndEndForALeftToRightHorizontal)
{
  HorizontalEdge<double> e{0, 1, 7};
  BOOST_CHECK_EQUAL(e.begin(), (Vertex<double>{1, 0}));
  BOOST_CHECK_EQUAL(e.end(), (Vertex<double>{7, 0}));
}

BOOST_AUTO_TEST_CASE(BeginAndEndForARightToLeftHorizontal)
{
  HorizontalEdge<double> e{0, 7, 1};
  BOOST_CHECK_EQUAL(e.begin(), (Vertex<double>{7, 0}));
  BOOST_CHECK_EQUAL(e.end(), (Vertex<double>{1, 0}));
}

BOOST_AUTO_TEST_CASE(VectorOfVerticals)
{
  // clang-format off
  std::vector<VerticalEdge<double>> testVerticals{{0.0, 7.0, 1.0}, {1.0, 1.0, 0.0}, {3.0, 0.0, 1.0},
                                                  {5.0, 1.0, 0.0}, {6.0, 0.0, 7.0}, {2.0, 5.0, 3.0},
                                                  {4.0, 3.0, 5.0}};
  // clang-format on
  BOOST_TEST(testVerticals.size() == 7);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
