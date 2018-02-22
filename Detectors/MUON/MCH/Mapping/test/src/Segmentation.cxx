//
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

#define BOOST_TEST_MODULE Test MCHMappingTest Segmentation
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>

#include "boost/format.hpp"
#include "MCHMappingInterface/Segmentation.h"
#include "MCHMappingSegContour/SegmentationContours.h"
#include "MCHMappingSegContour/SegmentationSVGWriter.h"
#include "MCHContour/SVGWriter.h"
#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/monomorphic/generators/xrange.hpp>
#include <boost/test/data/test_case.hpp>
#include <fstream>
#include <iostream>

using namespace o2::mch::mapping;
namespace bdata = boost::unit_test::data;
using Point = std::pair<double, double>;

BOOST_AUTO_TEST_SUITE(o2_mch_mapping)
BOOST_AUTO_TEST_SUITE(segmentation)

BOOST_AUTO_TEST_CASE(NumberOfDetectionElementsIs156)
{
  std::vector<int> des;
  forEachDetectionElement([&des](int detElemId) { des.push_back(detElemId); });
  BOOST_CHECK_EQUAL(des.size(), 156);
}

BOOST_AUTO_TEST_CASE(GetSegmentationMustNotThrowIfDetElemIdIsValid)
{
  forOneDetectionElementOfEachSegmentationType([](int detElemId) {
    BOOST_CHECK_NO_THROW(Segmentation(detElemId, true));
    BOOST_CHECK_NO_THROW(Segmentation(detElemId, false));
  });
}

BOOST_AUTO_TEST_CASE(GetSegmentationThrowsIfDetElemIdIsNotValid)
{
  BOOST_CHECK_THROW(Segmentation(-1, true), std::runtime_error);
  BOOST_CHECK_THROW(Segmentation(121, true), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(NofBendingPads)
{
  // we explicitly don't make a loop
  // we prefer this solution to more clearly document the number of pads per DE-type
  // sorted by number of pads.

  BOOST_CHECK_EQUAL(Segmentation(100, true).nofPads(), 14392);
  BOOST_CHECK_EQUAL(Segmentation(300, true).nofPads(), 13947);
  BOOST_CHECK_EQUAL(Segmentation(902, true).nofPads(), 4480);
  BOOST_CHECK_EQUAL(Segmentation(702, true).nofPads(), 4160);
  BOOST_CHECK_EQUAL(Segmentation(701, true).nofPads(), 4096);
  BOOST_CHECK_EQUAL(Segmentation(601, true).nofPads(), 3648);
  BOOST_CHECK_EQUAL(Segmentation(501, true).nofPads(), 3568);
  BOOST_CHECK_EQUAL(Segmentation(602, true).nofPads(), 3200);
  BOOST_CHECK_EQUAL(Segmentation(700, true).nofPads(), 3200);
  BOOST_CHECK_EQUAL(Segmentation(502, true).nofPads(), 3120);
  BOOST_CHECK_EQUAL(Segmentation(600, true).nofPads(), 3008);
  BOOST_CHECK_EQUAL(Segmentation(500, true).nofPads(), 2928);
  BOOST_CHECK_EQUAL(Segmentation(903, true).nofPads(), 2880);
  BOOST_CHECK_EQUAL(Segmentation(703, true).nofPads(), 2560);
  BOOST_CHECK_EQUAL(Segmentation(904, true).nofPads(), 2240);
  BOOST_CHECK_EQUAL(Segmentation(503, true).nofPads(), 1920);
  BOOST_CHECK_EQUAL(Segmentation(704, true).nofPads(), 1920);
  BOOST_CHECK_EQUAL(Segmentation(504, true).nofPads(), 1280);
  BOOST_CHECK_EQUAL(Segmentation(905, true).nofPads(), 1280);
  BOOST_CHECK_EQUAL(Segmentation(705, true).nofPads(), 960);
  BOOST_CHECK_EQUAL(Segmentation(706, true).nofPads(), 640);
}

BOOST_AUTO_TEST_CASE(NofNonBendingPads)
{
  BOOST_CHECK_EQUAL(Segmentation(100, false).nofPads(), 14280);
  BOOST_CHECK_EQUAL(Segmentation(300, false).nofPads(), 13986);
  BOOST_CHECK_EQUAL(Segmentation(902, false).nofPads(), 3136);
  BOOST_CHECK_EQUAL(Segmentation(702, false).nofPads(), 2912);
  BOOST_CHECK_EQUAL(Segmentation(701, false).nofPads(), 2880);
  BOOST_CHECK_EQUAL(Segmentation(601, false).nofPads(), 2560);
  BOOST_CHECK_EQUAL(Segmentation(501, false).nofPads(), 2496);
  BOOST_CHECK_EQUAL(Segmentation(602, false).nofPads(), 2240);
  BOOST_CHECK_EQUAL(Segmentation(700, false).nofPads(), 2240);
  BOOST_CHECK_EQUAL(Segmentation(502, false).nofPads(), 2176);
  BOOST_CHECK_EQUAL(Segmentation(600, false).nofPads(), 2112);
  BOOST_CHECK_EQUAL(Segmentation(500, false).nofPads(), 2048);
  BOOST_CHECK_EQUAL(Segmentation(903, false).nofPads(), 2016);
  BOOST_CHECK_EQUAL(Segmentation(703, false).nofPads(), 1792);
  BOOST_CHECK_EQUAL(Segmentation(904, false).nofPads(), 1568);
  BOOST_CHECK_EQUAL(Segmentation(503, false).nofPads(), 1344);
  BOOST_CHECK_EQUAL(Segmentation(704, false).nofPads(), 1344);
  BOOST_CHECK_EQUAL(Segmentation(504, false).nofPads(), 896);
  BOOST_CHECK_EQUAL(Segmentation(905, false).nofPads(), 896);
  BOOST_CHECK_EQUAL(Segmentation(705, false).nofPads(), 672);
  BOOST_CHECK_EQUAL(Segmentation(706, false).nofPads(), 448);
}

BOOST_AUTO_TEST_CASE(TotalNofBendingFECInSegTypes)
{
  int nb{0};
  int nnb{0};
  forOneDetectionElementOfEachSegmentationType([&](int detElemId) {
    nb += Segmentation(detElemId, true).nofDualSampas();
    nnb += Segmentation(detElemId, false).nofDualSampas();
  });
  BOOST_CHECK_EQUAL(nb, 1246);
  BOOST_CHECK_EQUAL(nnb, 1019);
}

BOOST_AUTO_TEST_CASE(NofBendingFEC)
{
  BOOST_CHECK_EQUAL(Segmentation(100, true).nofDualSampas(), 226);
  BOOST_CHECK_EQUAL(Segmentation(300, true).nofDualSampas(), 221);
  BOOST_CHECK_EQUAL(Segmentation(902, true).nofDualSampas(), 70);
  BOOST_CHECK_EQUAL(Segmentation(702, true).nofDualSampas(), 65);
  BOOST_CHECK_EQUAL(Segmentation(701, true).nofDualSampas(), 64);
  BOOST_CHECK_EQUAL(Segmentation(601, true).nofDualSampas(), 57);
  BOOST_CHECK_EQUAL(Segmentation(501, true).nofDualSampas(), 56);
  BOOST_CHECK_EQUAL(Segmentation(602, true).nofDualSampas(), 50);
  BOOST_CHECK_EQUAL(Segmentation(700, true).nofDualSampas(), 50);
  BOOST_CHECK_EQUAL(Segmentation(502, true).nofDualSampas(), 49);
  BOOST_CHECK_EQUAL(Segmentation(600, true).nofDualSampas(), 47);
  BOOST_CHECK_EQUAL(Segmentation(500, true).nofDualSampas(), 46);
  BOOST_CHECK_EQUAL(Segmentation(903, true).nofDualSampas(), 45);
  BOOST_CHECK_EQUAL(Segmentation(703, true).nofDualSampas(), 40);
  BOOST_CHECK_EQUAL(Segmentation(904, true).nofDualSampas(), 35);
  BOOST_CHECK_EQUAL(Segmentation(503, true).nofDualSampas(), 30);
  BOOST_CHECK_EQUAL(Segmentation(704, true).nofDualSampas(), 30);
  BOOST_CHECK_EQUAL(Segmentation(504, true).nofDualSampas(), 20);
  BOOST_CHECK_EQUAL(Segmentation(905, true).nofDualSampas(), 20);
  BOOST_CHECK_EQUAL(Segmentation(705, true).nofDualSampas(), 15);
  BOOST_CHECK_EQUAL(Segmentation(706, true).nofDualSampas(), 10);
}

BOOST_AUTO_TEST_CASE(NofNonBendingFEC)
{
  BOOST_CHECK_EQUAL(Segmentation(100, false).nofDualSampas(), 225);
  BOOST_CHECK_EQUAL(Segmentation(300, false).nofDualSampas(), 222);
  BOOST_CHECK_EQUAL(Segmentation(902, false).nofDualSampas(), 50);
  BOOST_CHECK_EQUAL(Segmentation(701, false).nofDualSampas(), 46);
  BOOST_CHECK_EQUAL(Segmentation(702, false).nofDualSampas(), 46);
  BOOST_CHECK_EQUAL(Segmentation(601, false).nofDualSampas(), 40);
  BOOST_CHECK_EQUAL(Segmentation(501, false).nofDualSampas(), 39);
  BOOST_CHECK_EQUAL(Segmentation(700, false).nofDualSampas(), 36);
  BOOST_CHECK_EQUAL(Segmentation(602, false).nofDualSampas(), 35);
  BOOST_CHECK_EQUAL(Segmentation(502, false).nofDualSampas(), 34);
  BOOST_CHECK_EQUAL(Segmentation(600, false).nofDualSampas(), 33);
  BOOST_CHECK_EQUAL(Segmentation(903, false).nofDualSampas(), 33);
  BOOST_CHECK_EQUAL(Segmentation(500, false).nofDualSampas(), 32);
  BOOST_CHECK_EQUAL(Segmentation(703, false).nofDualSampas(), 29);
  BOOST_CHECK_EQUAL(Segmentation(904, false).nofDualSampas(), 26);
  BOOST_CHECK_EQUAL(Segmentation(704, false).nofDualSampas(), 22);
  BOOST_CHECK_EQUAL(Segmentation(503, false).nofDualSampas(), 21);
  BOOST_CHECK_EQUAL(Segmentation(905, false).nofDualSampas(), 16);
  BOOST_CHECK_EQUAL(Segmentation(504, false).nofDualSampas(), 14);
  BOOST_CHECK_EQUAL(Segmentation(705, false).nofDualSampas(), 12);
  BOOST_CHECK_EQUAL(Segmentation(706, false).nofDualSampas(), 8);
}

BOOST_AUTO_TEST_CASE(CountPadsInSegmentations)
{
  int n{0};
  forOneDetectionElementOfEachSegmentationType([&n](int detElemId) {
    for (auto plane : {true, false}) {
      Segmentation seg{detElemId, plane};
      n += seg.nofPads();
    }
  });
  BOOST_CHECK_EQUAL(n, 143469);
}

BOOST_AUTO_TEST_CASE(LoopOnSegmentations)
{
  int n{0};
  forOneDetectionElementOfEachSegmentationType([&n](int detElemId) {
    n += 2; // two planes (bending, non-bending)
  });
  BOOST_CHECK_EQUAL(n, 42);
}

BOOST_AUTO_TEST_CASE(DualSampasWithLessThan64Pads)
{
  std::map<int, int> non64;
  forOneDetectionElementOfEachSegmentationType([&non64](int detElemId) {
    for (auto plane : {true, false}) {
      Segmentation seg{detElemId, plane};
      for (int i = 0; i < seg.nofDualSampas(); ++i) {
        int n{0};
        seg.forEachPadInDualSampa(seg.dualSampaId(i), [&n](int /*paduid*/) {
          ++n;
        });
        if (n != 64) {
          non64[n]++;
        }
      }
    }
  });

  BOOST_CHECK_EQUAL(non64[31], 1);
  BOOST_CHECK_EQUAL(non64[32], 2);
  BOOST_CHECK_EQUAL(non64[39], 1);
  BOOST_CHECK_EQUAL(non64[40], 3);
  BOOST_CHECK_EQUAL(non64[46], 2);
  BOOST_CHECK_EQUAL(non64[48], 10);
  BOOST_CHECK_EQUAL(non64[49], 1);
  BOOST_CHECK_EQUAL(non64[50], 1);
  BOOST_CHECK_EQUAL(non64[52], 3);
  BOOST_CHECK_EQUAL(non64[54], 2);
  BOOST_CHECK_EQUAL(non64[55], 3);
  BOOST_CHECK_EQUAL(non64[56], 114);
  BOOST_CHECK_EQUAL(non64[57], 3);
  BOOST_CHECK_EQUAL(non64[58], 2);
  BOOST_CHECK_EQUAL(non64[59], 1);
  BOOST_CHECK_EQUAL(non64[60], 6);
  BOOST_CHECK_EQUAL(non64[62], 4);
  BOOST_CHECK_EQUAL(non64[63], 7);

  int n{0};
  for (auto p: non64) {
    n += p.second;
  }

  BOOST_CHECK_EQUAL(n, 166);
}

void dumpToFile(std::string fileName, const Segmentation &seg, const std::vector<Point> &points)
{
  std::ofstream out(fileName);
  o2::mch::contour::SVGWriter w(getBBox(seg));

  w.addStyle(svgSegmentationDefaultStyle());

  svgSegmentation(seg, w, true, true, true, false);

  w.svgGroupStart("testpoints");
  w.points(points, 0.1);
  w.svgGroupEnd();
  w.writeHTML(out);
}

/// Check that for all points within the segmentation contour
/// the findPadByPosition actually returns a valid pad
std::vector<Point> checkGaps(const Segmentation &seg, double xstep = 1.0, double ystep = 1.0)
{
  std::vector<Point> gaps;
  auto bbox = o2::mch::mapping::getBBox(seg);
  auto env = o2::mch::mapping::getEnvelop(seg);

  if (env.size()!=1) {
    throw std::runtime_error("assumption env contour = one polygon is not verified");
  }

  for (double x = bbox.xmin() - xstep; x <= bbox.xmax() + xstep; x += xstep) {
    for (double y = bbox.ymin() - ystep; y <= bbox.ymax() + ystep; y += ystep) {
      double distanceToEnveloppe = std::sqrt(o2::mch::contour::squaredDistancePointToPolygon({x, y}, env[0]));
      bool withinEnveloppe = env.contains(x,y) && (distanceToEnveloppe > 1E-5);
      if (withinEnveloppe && !seg.isValid(seg.findPadByPosition(x, y))) {
        gaps.push_back(std::make_pair(x, y));
      }
    }
  }
  return gaps;
}


BOOST_DATA_TEST_CASE(NoGapWithinPads,
                     boost::unit_test::data::make({100, 300, 500, 501, 502, 503, 504, 600, 601, 602, 700, 701, 702, 703, 704, 705, 706, 902, 903, 904, 905}) * boost::unit_test::data::make({true,false}),
                     detElemId, isBendingPlane)
{
  Segmentation seg{detElemId, isBendingPlane};
  auto g = checkGaps(seg);

  if (!g.empty()) {
    dumpToFile("bug-gap-" + std::to_string(detElemId) + "-" + (isBendingPlane ? "B" : "NB") + ".html", seg, g);
  }
  BOOST_TEST(g.empty());
}

struct SEG
{
    Segmentation seg{100, true};
};

BOOST_FIXTURE_TEST_SUITE(HasPadBy, SEG)

BOOST_AUTO_TEST_CASE(ThrowsIfDualSampaChannelIsNotBetween0And63)
{
  BOOST_CHECK_THROW(seg.findPadByFEE(102, -1), std::out_of_range);
  BOOST_CHECK_THROW(seg.findPadByFEE(102, 64), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(ReturnsTrueIfPadIsConnected)
{
  BOOST_CHECK_EQUAL(seg.isValid(seg.findPadByFEE(102, 3)), true);
}

BOOST_AUTO_TEST_CASE(ReturnsFalseIfPadIsNotConnected)
{
  BOOST_CHECK_EQUAL(seg.isValid(seg.findPadByFEE(214, 14)), false);
}

BOOST_AUTO_TEST_CASE(HasPadByPosition)
{
  BOOST_CHECK_EQUAL(seg.isValid(seg.findPadByPosition(40.0, 30.0)), true);
}

BOOST_AUTO_TEST_CASE(CheckPositionOfOnePadInDE100Bending)
{
  BOOST_CHECK_EQUAL(seg.findPadByFEE(76, 9), seg.findPadByPosition(1.575, 18.69));
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
