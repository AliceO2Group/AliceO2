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

#define BOOST_TEST_MODULE Test MCHContour Polygon
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>
#include "../include/MCHContour/Polygon.h"
#include "../include/MCHContour/BBox.h"

using namespace o2::mch::contour;

struct POLYGONS {
  POLYGONS()
    : testPads{{{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.0, 0.0}}},
               {{{1.0, 3.0}, {2.0, 3.0}, {2.0, 4.0}, {1.0, 4.0}, {1.0, 3.0}}},
               {{{1.0, 0.0}, {2.0, 0.0}, {2.0, 1.0}, {1.0, 1.0}, {1.0, 0.0}}},
               {{{0.0, 1.0}, {1.0, 1.0}, {1.0, 2.0}, {0.0, 2.0}, {0.0, 1.0}}},
               {{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}, {1.0, 2.0}, {1.0, 1.0}}},
               {{{1.0, 2.0}, {2.0, 2.0}, {2.0, 3.0}, {1.0, 3.0}, {1.0, 2.0}}}}
  {
  }

  std::vector<Polygon<double>> testPads;
  Polygon<double> polygon;
  // clang-format off
  Polygon<double> testPolygon{{{0.1, 0.1},
                               {1.1, 0.1},
                               {1.1, 1.1},
                               {2.1, 1.1},
                               {2.1, 3.1},
                               {1.1, 3.1},
                               {1.1, 2.1},
                               {0.1, 2.1},
                               {0.1, 0.1}}};
  // clang-format on
  Polygon<int> counterClockwisePolygon{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}};
  Polygon<int> clockwisePolygon{{0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}};
  Polygon<double> clockwisePolygonDouble{{0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}};
  Polygon<double> testPolygon2{
    {{-5.0, 10.0}, {-5.0, -2.0}, {0.0, -2.0}, {0.0, -10.0}, {5.0, -10.0}, {5.0, 10.0}, {-5.0, 10.0}}};
};

BOOST_AUTO_TEST_SUITE(o2_mch_contour)

BOOST_FIXTURE_TEST_SUITE(polygon, POLYGONS)

BOOST_AUTO_TEST_CASE(CreateCounterClockwiseOrientedPolygon)
{
  BOOST_CHECK(counterClockwisePolygon.isCounterClockwiseOriented());
}

BOOST_AUTO_TEST_CASE(CreateClockwiseOrientedPolygon) { BOOST_CHECK(!clockwisePolygon.isCounterClockwiseOriented()); }

BOOST_AUTO_TEST_CASE(SignedArea) { BOOST_CHECK_CLOSE(testPolygon.signedArea(), 4.0, 0.1); }

BOOST_AUTO_TEST_CASE(AClosePolygonIsAPolygonWhereLastVertexIsTheSameAsFirstOne)
{
  Polygon<int> p{{0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}};
  BOOST_CHECK(p.isClosed());
}

BOOST_AUTO_TEST_CASE(ClosingAClosedPolygonIsANop) { BOOST_CHECK(testPolygon == close(testPolygon)); }

BOOST_AUTO_TEST_CASE(ClosePolygon)
{
  Polygon<int> opened{{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  Polygon<int> expected{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}};
  auto closed = close(opened);
  BOOST_TEST(expected == closed);
}

BOOST_AUTO_TEST_CASE(ThrowIfClosingAPolygonResultInANonManhattanPolygon)
{
  Polygon<int> triangle{{0, 0}, {1, 0}, {1, 1}};

  BOOST_CHECK_THROW(close(triangle), std::logic_error);
}

BOOST_AUTO_TEST_CASE(AnOpenedPolygonCannotBeEqualToAClosedOneEvenWithSameSetOfVertices)
{
  Polygon<double> opened{
    {0, 2},
    {0, 0},
    {2, 0},
    {2, 4},
    {1, 4},
    {1, 2},
  };

  auto closed{close(opened)};

  BOOST_CHECK(closed != opened);
}

BOOST_AUTO_TEST_CASE(PolygonAreEqualAsLongAsTheyContainTheSameVerticesIrrespectiveOfOrder)
{
  Polygon<double> a{{0, 2}, {0, 0}, {2, 0}, {2, 4}, {1, 4}, {1, 2}, {0, 2}};

  Polygon<double> b{{2, 4}, {2, 0}, {1, 4}, {1, 2}, {0, 2}, {0, 0}, {2, 4}};

  Polygon<double> c{{2, 4}, {2, 0}, {1, 4}, {1, 2}, {0, 2}, {1, 1}};

  BOOST_CHECK(a == b);
  BOOST_CHECK(a != c);
}

BOOST_AUTO_TEST_CASE(ContainsThrowsIfCalledOnNonClosedPolygon)
{
  Polygon<double> opened{{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  BOOST_CHECK_THROW(opened.contains(0, 0), std::invalid_argument);
};

BOOST_AUTO_TEST_CASE(ContainsReturnsTrueIfPointIsInsidePolygon)
{
  BOOST_CHECK_EQUAL(testPolygon2.contains(0, 0), true);
  BOOST_CHECK_EQUAL(testPolygon2.contains(-4.999, -1.999), true);
}

BOOST_AUTO_TEST_CASE(ContainsReturnsFalseIfPointIsExactlyOnAPolygonEdge)
{
  BOOST_CHECK_EQUAL(testPolygon2.contains(-2.5, -2), false);
}

BOOST_AUTO_TEST_CASE(BBoxCreation)
{
  BBox<double> expected{-5.0, -10.0, 5.0, 10.0};
  BOOST_TEST(getBBox(testPolygon2) == expected);
}

BOOST_AUTO_TEST_CASE(PolygonCenter)
{
  Polygon<double> p{{-80, -20}, {-70, -20}, {-70, -19.5}, {-80, -19.5}, {-80, -20}};

  auto box = getBBox(p);
  std::cout << box << "\n";
  BOOST_CHECK_EQUAL(box.xcenter(), -75.0);
  BOOST_CHECK_EQUAL(box.ycenter(), -19.75);
}

BOOST_AUTO_TEST_CASE(ConstructionByVectorIterators)
{
  std::vector<Vertex<int>> vertices{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}};

  Polygon<int> p(vertices.begin(), vertices.end());

  BOOST_CHECK_EQUAL(p, counterClockwisePolygon);
}

BOOST_AUTO_TEST_CASE(PointOutsidePolygonDistanceToPolygonClosestToOneSegment)
{
  BOOST_CHECK_EQUAL(squaredDistancePointToPolygon(Vertex<double>{-1.0, -6.0}, testPolygon2), 1.0);
  BOOST_CHECK_EQUAL(squaredDistancePointToPolygon(Vertex<double>{3.0, -14.0}, testPolygon2), 16.0);
}

BOOST_AUTO_TEST_CASE(PointOutsidePolygonDistanceToPolygonClosestToOneSegmentEndPoint)
{
  BOOST_CHECK_EQUAL(squaredDistancePointToPolygon(Vertex<double>{-1.0, -14.0}, testPolygon2), 17.0);
  BOOST_CHECK_EQUAL(squaredDistancePointToPolygon(Vertex<double>{7.0, -14.0}, testPolygon2), 20.0);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
