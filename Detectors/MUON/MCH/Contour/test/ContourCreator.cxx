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

#define BOOST_TEST_MODULE Test MCHContour ContourCreator
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <chrono>
#include <iostream>
#include "../include/MCHContour/ContourCreator.h"

using namespace o2::mch::contour;
using namespace o2::mch::contour::impl;

struct ContourCreatorPolygons {
  ContourCreatorPolygons()
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
  Polygon<double> testPolygon{{{0.1, 0.1},
                               {1.1, 0.1},
                               {1.1, 1.1},
                               {2.1, 1.1},
                               {2.1, 3.1},
                               {1.1, 3.1},
                               {1.1, 2.1},
                               {0.1, 2.1},
                               {0.1, 0.1}}};
  Polygon<int> counterClockwisePolygon{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}};
  Polygon<int> clockwisePolygon{{0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}};
  Polygon<double> clockwisePolygonDouble{{0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}};
};

BOOST_AUTO_TEST_SUITE(o2_mch_contour)

BOOST_FIXTURE_TEST_SUITE(contourCreator, ContourCreatorPolygons)

BOOST_AUTO_TEST_CASE(ContourCreationGeneratesEmptyContourForEmptyInput)
{
  std::vector<Polygon<double>> list;
  auto contour = createContour(list);
  BOOST_CHECK(contour.empty());
}

BOOST_AUTO_TEST_CASE(ContourCreationThrowsIfInputPolygonsAreNotCounterClockwiseOriented)
{
  std::vector<Polygon<double>> list;
  list.push_back(clockwisePolygonDouble);
  BOOST_CHECK_THROW(createContour(list), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(ContourCreationReturnsInputIfInputIsASinglePolygon)
{
  std::vector<Polygon<double>> list;
  Polygon<double> onePolygon{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}};
  list.push_back(onePolygon);
  auto contour = createContour(list);
  BOOST_REQUIRE(contour.size() == 1);
  BOOST_CHECK_EQUAL(contour[0], onePolygon);
}

BOOST_AUTO_TEST_CASE(VerticalEdgeSortingMustSortSameAbcissaPointsLeftEdgeFirst)
{
  std::vector<VerticalEdge<double>> edges;
  constexpr double sameX{42};
  VerticalEdge<double> lastEdge{sameX + 1, 2, 0};
  VerticalEdge<double> leftEdgeBottom{sameX, 2, 0};
  VerticalEdge<double> leftEdgeTop{sameX, 10, 5};
  VerticalEdge<double> rightEdge{sameX, 0, 2};

  edges.push_back(lastEdge);
  edges.push_back(rightEdge);
  edges.push_back(leftEdgeTop);
  edges.push_back(leftEdgeBottom);

  sortVerticalEdges(edges);

  BOOST_CHECK_EQUAL(edges[0], leftEdgeBottom);
  BOOST_CHECK_EQUAL(edges[1], leftEdgeTop);
  BOOST_CHECK_EQUAL(edges[2], rightEdge);
  BOOST_CHECK_EQUAL(edges[3], lastEdge);
}

BOOST_AUTO_TEST_CASE(VerticalsToHorizontals)
{
  // clang-format off
  std::vector<VerticalEdge<double>> testVerticals{{0.0, 7.0, 1.0}, {1.0, 1.0, 0.0}, {3.0, 0.0, 1.0},
                                                  {5.0, 1.0, 0.0}, {6.0, 0.0, 7.0}, {2.0, 5.0, 3.0},
                                                  {4.0, 3.0, 5.0}};
  // clang-format on
  std::vector<HorizontalEdge<double>> he{verticalsToHorizontals(testVerticals)};

  std::vector<HorizontalEdge<double>> expected{{1, 0, 1}, {0, 1, 3}, {1, 3, 5}, {0, 5, 6}, {7, 6, 0}, {3, 2, 4}, {5, 4, 2}};

  BOOST_CHECK(he == expected);
}

BOOST_AUTO_TEST_CASE(FinalizeContourThrowsIfNumberOfVerticalsDifferFromNumberOfHorizontals)
{
  std::vector<VerticalEdge<double>> v{{0, 1, 0}, {1, 0, 1}};
  std::vector<HorizontalEdge<double>> h{{0, 0, 1}};
  BOOST_CHECK_THROW(finalizeContour(v, h), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(FinalizeContourThrowsIfEndOfVerticalsDoNotMatchBeginOfHorizontals)
{
  std::vector<VerticalEdge<double>> v{{0, 7, 1}};
  std::vector<HorizontalEdge<double>> wrong{{1, 2, 3}};
  BOOST_CHECK_THROW(finalizeContour(v, wrong), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(FinalizeContourIEEEExample)
{
  // clang-format off
  std::vector<VerticalEdge<double>> testVerticals{{0.0, 7.0, 1.0}, {1.0, 1.0, 0.0}, {3.0, 0.0, 1.0},
                                                  {5.0, 1.0, 0.0}, {6.0, 0.0, 7.0}, {2.0, 5.0, 3.0},
                                                  {4.0, 3.0, 5.0}};
  // clang-format on
  auto he{verticalsToHorizontals(testVerticals)};

  auto contour = finalizeContour(testVerticals, he);

  Contour<double> expected{
    {{0, 7}, {0, 1}, {1, 1}, {1, 0}, {3, 0}, {3, 1}, {5, 1}, {5, 0}, {6, 0}, {6, 7}, {0, 7}},
    {{2, 5}, {2, 3}, {4, 3}, {4, 5}, {2, 5}}};

  BOOST_TEST(contour == expected);
}

BOOST_AUTO_TEST_CASE(FinalizeContourWithOneCommonVertex)
{
  std::vector<VerticalEdge<double>> ve{{0, 2, 0}, {1, 0, 2}, {1, 4, 2}, {2, 2, 4}};

  auto he{verticalsToHorizontals(ve)};

  auto contour = finalizeContour(ve, he);

  Contour<double> expected{{{0, 2}, {0, 0}, {1, 0}, {1, 2}, {0, 2}},
                           {{1, 4}, {1, 2}, {2, 2}, {2, 4}, {1, 4}}};

  BOOST_TEST(contour == expected);
}

BOOST_AUTO_TEST_CASE(CreateContourWithOneCommonVertex)
{
  std::vector<Polygon<double>> input{{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}},
                                     {{0, 1}, {1, 1}, {1, 2}, {0, 2}, {0, 1}},
                                     {{1, 2}, {2, 2}, {2, 3}, {1, 3}, {1, 2}},
                                     {{1, 3}, {2, 3}, {2, 4}, {1, 4}, {1, 3}}};

  auto contour = createContour(input);

  Contour<double> expected{{{0, 2}, {0, 0}, {1, 0}, {1, 2}, {0, 2}},
                           {{1, 4}, {1, 2}, {2, 2}, {2, 4}, {1, 4}}};

  BOOST_CHECK(contour == expected);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
