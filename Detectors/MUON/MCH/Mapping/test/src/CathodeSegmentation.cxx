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

#define BOOST_TEST_MODULE Test MCHMappingTest CathodeSegmentation
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>

#include "boost/format.hpp"
#include "MCHMappingInterface/CathodeSegmentation.h"
#include "MCHMappingSegContour/CathodeSegmentationContours.h"
#include "MCHMappingSegContour/CathodeSegmentationSVGWriter.h"
#include "MCHContour/SVGWriter.h"
#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/monomorphic/generators/xrange.hpp>
#include <boost/test/data/test_case.hpp>
#include <limits>
#include <fstream>
#include <iostream>

using namespace o2::mch::mapping;
namespace bdata = boost::unit_test::data;

BOOST_AUTO_TEST_SUITE(o2_mch_mapping)
BOOST_AUTO_TEST_SUITE(cathode_segmentation)

BOOST_AUTO_TEST_CASE(NumberOfDetectionElementsIs156)
{
  std::vector<int> des;
  forEachDetectionElement([&des](int detElemId) { des.push_back(detElemId); });
  BOOST_CHECK_EQUAL(des.size(), 156);
}

BOOST_AUTO_TEST_CASE(GetCathodeSegmentationMustNotThrowIfDetElemIdIsValid)
{
  forOneDetectionElementOfEachSegmentationType([](int detElemId) {
    BOOST_CHECK_NO_THROW(CathodeSegmentation(detElemId, true));
    BOOST_CHECK_NO_THROW(CathodeSegmentation(detElemId, false));
  });
}

BOOST_AUTO_TEST_CASE(GetCathodeSegmentationThrowsIfDetElemIdIsNotValid)
{
  BOOST_CHECK_THROW(CathodeSegmentation(-1, true), std::runtime_error);
  BOOST_CHECK_THROW(CathodeSegmentation(121, true), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(NofBendingPads)
{
  // we explicitly don't make a loop
  // we prefer this solution to more clearly document the number of pads per DE-type
  // sorted by number of pads.

  BOOST_CHECK_EQUAL(CathodeSegmentation(100, true).nofPads(), 14392);
  BOOST_CHECK_EQUAL(CathodeSegmentation(300, true).nofPads(), 13947);
  BOOST_CHECK_EQUAL(CathodeSegmentation(902, true).nofPads(), 4480);
  BOOST_CHECK_EQUAL(CathodeSegmentation(702, true).nofPads(), 4160);
  BOOST_CHECK_EQUAL(CathodeSegmentation(701, true).nofPads(), 4096);
  BOOST_CHECK_EQUAL(CathodeSegmentation(601, true).nofPads(), 3648);
  BOOST_CHECK_EQUAL(CathodeSegmentation(501, true).nofPads(), 3568);
  BOOST_CHECK_EQUAL(CathodeSegmentation(602, true).nofPads(), 3200);
  BOOST_CHECK_EQUAL(CathodeSegmentation(700, true).nofPads(), 3200);
  BOOST_CHECK_EQUAL(CathodeSegmentation(502, true).nofPads(), 3120);
  BOOST_CHECK_EQUAL(CathodeSegmentation(600, true).nofPads(), 3008);
  BOOST_CHECK_EQUAL(CathodeSegmentation(500, true).nofPads(), 2928);
  BOOST_CHECK_EQUAL(CathodeSegmentation(903, true).nofPads(), 2880);
  BOOST_CHECK_EQUAL(CathodeSegmentation(703, true).nofPads(), 2560);
  BOOST_CHECK_EQUAL(CathodeSegmentation(904, true).nofPads(), 2240);
  BOOST_CHECK_EQUAL(CathodeSegmentation(503, true).nofPads(), 1920);
  BOOST_CHECK_EQUAL(CathodeSegmentation(704, true).nofPads(), 1920);
  BOOST_CHECK_EQUAL(CathodeSegmentation(504, true).nofPads(), 1280);
  BOOST_CHECK_EQUAL(CathodeSegmentation(905, true).nofPads(), 1280);
  BOOST_CHECK_EQUAL(CathodeSegmentation(705, true).nofPads(), 960);
  BOOST_CHECK_EQUAL(CathodeSegmentation(706, true).nofPads(), 640);
}

BOOST_AUTO_TEST_CASE(NofNonBendingPads)
{
  BOOST_CHECK_EQUAL(CathodeSegmentation(100, false).nofPads(), 14280);
  BOOST_CHECK_EQUAL(CathodeSegmentation(300, false).nofPads(), 13986);
  BOOST_CHECK_EQUAL(CathodeSegmentation(902, false).nofPads(), 3136);
  BOOST_CHECK_EQUAL(CathodeSegmentation(702, false).nofPads(), 2912);
  BOOST_CHECK_EQUAL(CathodeSegmentation(701, false).nofPads(), 2880);
  BOOST_CHECK_EQUAL(CathodeSegmentation(601, false).nofPads(), 2560);
  BOOST_CHECK_EQUAL(CathodeSegmentation(501, false).nofPads(), 2496);
  BOOST_CHECK_EQUAL(CathodeSegmentation(602, false).nofPads(), 2240);
  BOOST_CHECK_EQUAL(CathodeSegmentation(700, false).nofPads(), 2240);
  BOOST_CHECK_EQUAL(CathodeSegmentation(502, false).nofPads(), 2176);
  BOOST_CHECK_EQUAL(CathodeSegmentation(600, false).nofPads(), 2112);
  BOOST_CHECK_EQUAL(CathodeSegmentation(500, false).nofPads(), 2048);
  BOOST_CHECK_EQUAL(CathodeSegmentation(903, false).nofPads(), 2016);
  BOOST_CHECK_EQUAL(CathodeSegmentation(703, false).nofPads(), 1792);
  BOOST_CHECK_EQUAL(CathodeSegmentation(904, false).nofPads(), 1568);
  BOOST_CHECK_EQUAL(CathodeSegmentation(503, false).nofPads(), 1344);
  BOOST_CHECK_EQUAL(CathodeSegmentation(704, false).nofPads(), 1344);
  BOOST_CHECK_EQUAL(CathodeSegmentation(504, false).nofPads(), 896);
  BOOST_CHECK_EQUAL(CathodeSegmentation(905, false).nofPads(), 896);
  BOOST_CHECK_EQUAL(CathodeSegmentation(705, false).nofPads(), 672);
  BOOST_CHECK_EQUAL(CathodeSegmentation(706, false).nofPads(), 448);
}

BOOST_AUTO_TEST_CASE(TotalNofBendingFECInSegTypes)
{
  int nb{0};
  int nnb{0};
  forOneDetectionElementOfEachSegmentationType([&](int detElemId) {
    nb += CathodeSegmentation(detElemId, true).nofDualSampas();
    nnb += CathodeSegmentation(detElemId, false).nofDualSampas();
  });
  BOOST_CHECK_EQUAL(nb, 1246);
  BOOST_CHECK_EQUAL(nnb, 1019);
}

BOOST_AUTO_TEST_CASE(BendingBoundingBox)
{
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(100, true)), o2::mch::contour::BBox<double>(0, 0, 89.04, 89.46));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(300, true)), o2::mch::contour::BBox<double>(-1, -0.75, 116, 117.25));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(500, true)), o2::mch::contour::BBox<double>(-75, -20, 57.5, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(501, true)), o2::mch::contour::BBox<double>(-75, -20, 80, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(502, true)), o2::mch::contour::BBox<double>(-80, -20, 75, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(503, true)), o2::mch::contour::BBox<double>(-60, -20, 60, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(504, true)), o2::mch::contour::BBox<double>(-40, -20, 40, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(600, true)), o2::mch::contour::BBox<double>(-80, -20, 57.5, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(601, true)), o2::mch::contour::BBox<double>(-80, -20, 80, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(602, true)), o2::mch::contour::BBox<double>(-80, -20, 80, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(700, true)), o2::mch::contour::BBox<double>(-100, -20, 100, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(701, true)), o2::mch::contour::BBox<double>(-120, -20, 120, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(702, true)), o2::mch::contour::BBox<double>(-100, -20, 100, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(703, true)), o2::mch::contour::BBox<double>(-100, -20, 100, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(704, true)), o2::mch::contour::BBox<double>(-80, -20, 80, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(705, true)), o2::mch::contour::BBox<double>(-60, -20, 60, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(706, true)), o2::mch::contour::BBox<double>(-40, -20, 40, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(902, true)), o2::mch::contour::BBox<double>(-120, -20, 120, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(903, true)), o2::mch::contour::BBox<double>(-120, -20, 120, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(904, true)), o2::mch::contour::BBox<double>(-100, -20, 100, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(905, true)), o2::mch::contour::BBox<double>(-80, -20, 80, 20));
}

BOOST_AUTO_TEST_CASE(NonBendingBoundingBox)
{
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(100, false)), o2::mch::contour::BBox<double>(-0.315, 0.21, 89.145, 89.25));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(300, false)), o2::mch::contour::BBox<double>(-0.625, -0.5, 115.625, 117.5));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(500, false)), o2::mch::contour::BBox<double>(-74.2857, -20, 58.5714, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(501, false)), o2::mch::contour::BBox<double>(-74.2857, -20, 80, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(502, false)), o2::mch::contour::BBox<double>(-80, -20, 74.2857, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(503, false)), o2::mch::contour::BBox<double>(-60, -20, 60, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(504, false)), o2::mch::contour::BBox<double>(-40, -20, 40, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(600, false)), o2::mch::contour::BBox<double>(-80, -20, 58.5714, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(601, false)), o2::mch::contour::BBox<double>(-80, -20, 80, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(602, false)), o2::mch::contour::BBox<double>(-80, -20, 80, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(700, false)), o2::mch::contour::BBox<double>(-100, -20, 100, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(701, false)), o2::mch::contour::BBox<double>(-120, -20, 120, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(702, false)), o2::mch::contour::BBox<double>(-100, -20, 100, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(703, false)), o2::mch::contour::BBox<double>(-100, -20, 100, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(704, false)), o2::mch::contour::BBox<double>(-80, -20, 80, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(705, false)), o2::mch::contour::BBox<double>(-60, -20, 60, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(706, false)), o2::mch::contour::BBox<double>(-40, -20, 40, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(902, false)), o2::mch::contour::BBox<double>(-120, -20, 120, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(903, false)), o2::mch::contour::BBox<double>(-120, -20, 120, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(904, false)), o2::mch::contour::BBox<double>(-100, -20, 100, 20));
  BOOST_CHECK_EQUAL(getBBox(CathodeSegmentation(905, false)), o2::mch::contour::BBox<double>(-80, -20, 80, 20));
}

BOOST_AUTO_TEST_CASE(NofBendingFEC)
{
  BOOST_CHECK_EQUAL(CathodeSegmentation(100, true).nofDualSampas(), 226);
  BOOST_CHECK_EQUAL(CathodeSegmentation(300, true).nofDualSampas(), 221);
  BOOST_CHECK_EQUAL(CathodeSegmentation(902, true).nofDualSampas(), 70);
  BOOST_CHECK_EQUAL(CathodeSegmentation(702, true).nofDualSampas(), 65);
  BOOST_CHECK_EQUAL(CathodeSegmentation(701, true).nofDualSampas(), 64);
  BOOST_CHECK_EQUAL(CathodeSegmentation(601, true).nofDualSampas(), 57);
  BOOST_CHECK_EQUAL(CathodeSegmentation(501, true).nofDualSampas(), 56);
  BOOST_CHECK_EQUAL(CathodeSegmentation(602, true).nofDualSampas(), 50);
  BOOST_CHECK_EQUAL(CathodeSegmentation(700, true).nofDualSampas(), 50);
  BOOST_CHECK_EQUAL(CathodeSegmentation(502, true).nofDualSampas(), 49);
  BOOST_CHECK_EQUAL(CathodeSegmentation(600, true).nofDualSampas(), 47);
  BOOST_CHECK_EQUAL(CathodeSegmentation(500, true).nofDualSampas(), 46);
  BOOST_CHECK_EQUAL(CathodeSegmentation(903, true).nofDualSampas(), 45);
  BOOST_CHECK_EQUAL(CathodeSegmentation(703, true).nofDualSampas(), 40);
  BOOST_CHECK_EQUAL(CathodeSegmentation(904, true).nofDualSampas(), 35);
  BOOST_CHECK_EQUAL(CathodeSegmentation(503, true).nofDualSampas(), 30);
  BOOST_CHECK_EQUAL(CathodeSegmentation(704, true).nofDualSampas(), 30);
  BOOST_CHECK_EQUAL(CathodeSegmentation(504, true).nofDualSampas(), 20);
  BOOST_CHECK_EQUAL(CathodeSegmentation(905, true).nofDualSampas(), 20);
  BOOST_CHECK_EQUAL(CathodeSegmentation(705, true).nofDualSampas(), 15);
  BOOST_CHECK_EQUAL(CathodeSegmentation(706, true).nofDualSampas(), 10);
}

BOOST_AUTO_TEST_CASE(NofNonBendingFEC)
{
  BOOST_CHECK_EQUAL(CathodeSegmentation(100, false).nofDualSampas(), 225);
  BOOST_CHECK_EQUAL(CathodeSegmentation(300, false).nofDualSampas(), 222);
  BOOST_CHECK_EQUAL(CathodeSegmentation(902, false).nofDualSampas(), 50);
  BOOST_CHECK_EQUAL(CathodeSegmentation(701, false).nofDualSampas(), 46);
  BOOST_CHECK_EQUAL(CathodeSegmentation(702, false).nofDualSampas(), 46);
  BOOST_CHECK_EQUAL(CathodeSegmentation(601, false).nofDualSampas(), 40);
  BOOST_CHECK_EQUAL(CathodeSegmentation(501, false).nofDualSampas(), 39);
  BOOST_CHECK_EQUAL(CathodeSegmentation(700, false).nofDualSampas(), 36);
  BOOST_CHECK_EQUAL(CathodeSegmentation(602, false).nofDualSampas(), 35);
  BOOST_CHECK_EQUAL(CathodeSegmentation(502, false).nofDualSampas(), 34);
  BOOST_CHECK_EQUAL(CathodeSegmentation(600, false).nofDualSampas(), 33);
  BOOST_CHECK_EQUAL(CathodeSegmentation(903, false).nofDualSampas(), 33);
  BOOST_CHECK_EQUAL(CathodeSegmentation(500, false).nofDualSampas(), 32);
  BOOST_CHECK_EQUAL(CathodeSegmentation(703, false).nofDualSampas(), 29);
  BOOST_CHECK_EQUAL(CathodeSegmentation(904, false).nofDualSampas(), 26);
  BOOST_CHECK_EQUAL(CathodeSegmentation(704, false).nofDualSampas(), 22);
  BOOST_CHECK_EQUAL(CathodeSegmentation(503, false).nofDualSampas(), 21);
  BOOST_CHECK_EQUAL(CathodeSegmentation(905, false).nofDualSampas(), 16);
  BOOST_CHECK_EQUAL(CathodeSegmentation(504, false).nofDualSampas(), 14);
  BOOST_CHECK_EQUAL(CathodeSegmentation(705, false).nofDualSampas(), 12);
  BOOST_CHECK_EQUAL(CathodeSegmentation(706, false).nofDualSampas(), 8);
}

BOOST_AUTO_TEST_CASE(CountPadsInCathodeSegmentations)
{
  int n{0};
  forOneDetectionElementOfEachSegmentationType([&n](int detElemId) {
    for (auto plane : {true, false}) {
      CathodeSegmentation seg{detElemId, plane};
      n += seg.nofPads();
    }
  });
  BOOST_CHECK_EQUAL(n, 143469);
}

BOOST_AUTO_TEST_CASE(LoopOnCathodeSegmentations)
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
      CathodeSegmentation seg{detElemId, plane};
      for (int i = 0; i < seg.nofDualSampas(); ++i) {
        int n{0};
        seg.forEachPadInDualSampa(seg.dualSampaId(i), [&n](int /*catPadIndex*/) { ++n; });
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
  for (auto p : non64) {
    n += p.second;
  }

  BOOST_CHECK_EQUAL(n, 166);
}

struct SEG {
  CathodeSegmentation seg{100, true};
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

BOOST_AUTO_TEST_CASE(CheckCopy)
{
  CathodeSegmentation copy{seg};
  BOOST_TEST((copy == seg));
  BOOST_TEST(copy.nofPads() == seg.nofPads());
}

BOOST_AUTO_TEST_CASE(CheckAssignment)
{
  CathodeSegmentation copy{200, true};
  copy = seg;
  BOOST_TEST((copy == seg));
  BOOST_TEST(copy.nofPads() == seg.nofPads());
}
BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
