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

///
/// @author  Laurent Aphecetche

#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "boost/format.hpp"
#include "MCHMappingInterface/Segmentation.h"
#include "MCHMappingSegContour/SegmentationContours.h"
#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/monomorphic/generators/xrange.hpp>
#include <boost/test/data/test_case.hpp>
#include <fstream>
#include <iostream>
#include "TestParameters.h"
#include <fmt/format.h>

using namespace o2::mch::mapping;
namespace bdata = boost::unit_test::data;

BOOST_AUTO_TEST_SUITE(o2_mch_mapping)
BOOST_AUTO_TEST_SUITE(segmentation_both_cathodes)

BOOST_AUTO_TEST_CASE(GetSegmentationMustNotThrowIfDetElemIdIsValid)
{
  // do not use the o2::mch::mapping::segmentation() cache here as we want to test the object
  // construction (which is used also by the o2::mch::mapping::segmentation() function...)
  forOneDetectionElementOfEachSegmentationType([](int detElemId) {
    BOOST_CHECK_NO_THROW(Segmentation{detElemId});
  });
}

BOOST_AUTO_TEST_CASE(GetSegmentationThrowsIfDetElemIdIsNotValid)
{
  // do not use the o2::mch::mapping::segmentation() cache here as we want to test the object
  // construction (which is used also by the o2::mch::mapping::segmentation() function...)
  BOOST_CHECK_THROW(Segmentation(-1), std::runtime_error);
  BOOST_CHECK_THROW(Segmentation(121), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(CheckNofPads)
{
  // Explicitly don't make a loop to more clearly document the number of pads
  // per detection element.
  //
  // Sorted by decreasing number of pads.
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(100).nofPads(), 28672);
#ifdef MCH_MAPPING_RUN3_AND_ABOVE
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(300).nofPads(), 27873);
#else
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(300).nofPads(), 27933);
#endif
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(902).nofPads(), 7616);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(702).nofPads(), 7072);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(701).nofPads(), 6976);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(601).nofPads(), 6208);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(501).nofPads(), 6064);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(602).nofPads(), 5440);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(700).nofPads(), 5440);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(502).nofPads(), 5296);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(600).nofPads(), 5120);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(500).nofPads(), 4976);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(903).nofPads(), 4896);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(703).nofPads(), 4352);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(904).nofPads(), 3808);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(503).nofPads(), 3264);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(704).nofPads(), 3264);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(504).nofPads(), 2176);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(905).nofPads(), 2176);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(705).nofPads(), 1632);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(706).nofPads(), 1088);
}

BOOST_AUTO_TEST_CASE(TotalNofFECInSegTypesIs2265)
{
  int n{0};
  forOneDetectionElementOfEachSegmentationType([&](int detElemId) {
    n += o2::mch::mapping::segmentation(detElemId).nofDualSampas();
  });
#ifdef MCH_MAPPING_RUN3_AND_ABOVE
  BOOST_CHECK_EQUAL(n, 2264);
#else
  BOOST_CHECK_EQUAL(n, 2265);
#endif
}

BOOST_AUTO_TEST_CASE(CheckBoundingBoxesAreAsExpected)
{
  // BOOST_CHECK_EQUAL(getBBox(Segmentation(300)), o2::mch::contour::BBox<double>(-1, -0.75, 116, 117.25));
  //   BOOST_CHECK_EQUAL(getBBox(Segmentation(500)), o2::mch::contour::BBox<double>(-75, -20, 57.5, 20));
  BOOST_CHECK_EQUAL(getBBox(o2::mch::mapping::segmentation(501)), o2::mch::contour::BBox<double>(-75, -20, 80, 20));
  BOOST_CHECK_EQUAL(getBBox(o2::mch::mapping::segmentation(502)), o2::mch::contour::BBox<double>(-80, -20, 75, 20));
  BOOST_CHECK_EQUAL(getBBox(o2::mch::mapping::segmentation(503)), o2::mch::contour::BBox<double>(-60, -20, 60, 20));
  BOOST_CHECK_EQUAL(getBBox(o2::mch::mapping::segmentation(504)), o2::mch::contour::BBox<double>(-40, -20, 40, 20));
  //   BOOST_CHECK_EQUAL(getBBox(Segmentation(600)), o2::mch::contour::BBox<double>(-80, -20, 57.5, 20));
  BOOST_CHECK_EQUAL(getBBox(o2::mch::mapping::segmentation(601)), o2::mch::contour::BBox<double>(-80, -20, 80, 20));
  BOOST_CHECK_EQUAL(getBBox(o2::mch::mapping::segmentation(602)), o2::mch::contour::BBox<double>(-80, -20, 80, 20));
  BOOST_CHECK_EQUAL(getBBox(o2::mch::mapping::segmentation(700)), o2::mch::contour::BBox<double>(-100, -20, 100, 20));
  BOOST_CHECK_EQUAL(getBBox(o2::mch::mapping::segmentation(701)), o2::mch::contour::BBox<double>(-120, -20, 120, 20));
  BOOST_CHECK_EQUAL(getBBox(o2::mch::mapping::segmentation(702)), o2::mch::contour::BBox<double>(-100, -20, 100, 20));
  BOOST_CHECK_EQUAL(getBBox(o2::mch::mapping::segmentation(703)), o2::mch::contour::BBox<double>(-100, -20, 100, 20));
  BOOST_CHECK_EQUAL(getBBox(o2::mch::mapping::segmentation(704)), o2::mch::contour::BBox<double>(-80, -20, 80, 20));
  BOOST_CHECK_EQUAL(getBBox(o2::mch::mapping::segmentation(705)), o2::mch::contour::BBox<double>(-60, -20, 60, 20));
  BOOST_CHECK_EQUAL(getBBox(o2::mch::mapping::segmentation(706)), o2::mch::contour::BBox<double>(-40, -20, 40, 20));
  BOOST_CHECK_EQUAL(getBBox(o2::mch::mapping::segmentation(902)), o2::mch::contour::BBox<double>(-120, -20, 120, 20));
  BOOST_CHECK_EQUAL(getBBox(o2::mch::mapping::segmentation(903)), o2::mch::contour::BBox<double>(-120, -20, 120, 20));
  BOOST_CHECK_EQUAL(getBBox(o2::mch::mapping::segmentation(904)), o2::mch::contour::BBox<double>(-100, -20, 100, 20));
  BOOST_CHECK_EQUAL(getBBox(o2::mch::mapping::segmentation(905)), o2::mch::contour::BBox<double>(-80, -20, 80, 20));
}

BOOST_AUTO_TEST_CASE(CheckNofBendingFEC)
{
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(100).nofDualSampas(), 451);
#ifdef MCH_MAPPING_RUN3_AND_ABOVE
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(300).nofDualSampas(), 442);
#else
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(300).nofDualSampas(), 443);
#endif
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(902).nofDualSampas(), 120);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(702).nofDualSampas(), 111);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(701).nofDualSampas(), 110);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(601).nofDualSampas(), 97);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(501).nofDualSampas(), 95);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(602).nofDualSampas(), 85);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(700).nofDualSampas(), 86);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(502).nofDualSampas(), 83);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(600).nofDualSampas(), 80);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(500).nofDualSampas(), 78);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(903).nofDualSampas(), 78);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(703).nofDualSampas(), 69);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(904).nofDualSampas(), 61);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(503).nofDualSampas(), 51);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(704).nofDualSampas(), 52);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(504).nofDualSampas(), 34);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(905).nofDualSampas(), 36);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(705).nofDualSampas(), 27);
  BOOST_CHECK_EQUAL(o2::mch::mapping::segmentation(706).nofDualSampas(), 18);
}

BOOST_AUTO_TEST_CASE(PadCountInSegmentationTypesMustBe143469)
{
  int n{0};
  forOneDetectionElementOfEachSegmentationType([&n](int detElemId) {
    n += o2::mch::mapping::segmentation(detElemId).nofPads();
  });
#ifdef MCH_MAPPING_RUN3_AND_ABOVE
  BOOST_CHECK_EQUAL(n, 143409);
#else
  BOOST_CHECK_EQUAL(n, 143469);
#endif
}

BOOST_AUTO_TEST_CASE(PadCountInAllSegmentationsMustBe1064008)
{
  int n{0};
  forEachDetectionElement([&n](int detElemId) {
    n += o2::mch::mapping::segmentation(detElemId).nofPads();
  });
#ifdef MCH_MAPPING_RUN3_AND_ABOVE
  BOOST_CHECK_EQUAL(n, 1063528);
#else
  BOOST_CHECK_EQUAL(n, 1064008);
#endif
}

BOOST_AUTO_TEST_CASE(NumberOfSegmentationsMustBe21)
{
  int n{0};
  forOneDetectionElementOfEachSegmentationType([&n](int detElemId) {
    n++;
  });
  BOOST_CHECK_EQUAL(n, 21);
}

BOOST_AUTO_TEST_CASE(CheckPadOffsetsAfterCopy)
{
  forEachDetectionElement([](int detElemId) {
    bool ok{true};
    auto s = o2::mch::mapping::segmentation(detElemId);
    auto seg = s;
    for (auto padid = 0; padid < seg.bending().nofPads(); padid++) {
      if (seg.isBendingPad(padid) != true) {
        ok = false;
        break;
      }
    }
    BOOST_CHECK_MESSAGE(ok == true, "inconsistent isBendingPad for bending plane");
    ok = true;
    for (auto padid = seg.bending().nofPads(); padid < seg.nofPads(); padid++) {
      if (seg.isBendingPad(padid) != false) {
        ok = false;
        break;
      }
    }
    BOOST_CHECK_MESSAGE(ok == true, "inconsistent isBendingPad for non-bending plane");
  });
}

struct SEG100 {
  Segmentation seg{100};
};

BOOST_AUTO_TEST_CASE(TestForEachPadAndPadIndexRange)
{
  int npads = 0;
  forOneDetectionElementOfEachSegmentationType([&npads](int detElemId) {
    int n = 0;
    int pmin = std::numeric_limits<int>::max();
    int pmax = 0;
    Segmentation seg{detElemId};
    seg.forEachPad([&n, &pmin, &pmax, &npads](int dePadIndex) {
      npads++;
      n++;
      pmin = std::min(pmin, dePadIndex);
      pmax = std::max(pmax, dePadIndex);
    });
    BOOST_CHECK_EQUAL(n, seg.nofPads());
    BOOST_CHECK_EQUAL(pmin, 0);
    BOOST_CHECK_EQUAL(pmax, seg.nofPads() - 1);
  });
}

BOOST_AUTO_TEST_CASE(CheckOnePadPositionPresentOnOnlyBendingPlaneDE600)
{
  Segmentation de600(600);
  double x = 54.34;
  double y = -12.40;
  int b, nb;
  bool ok = de600.findPadPairByPosition(x, y, b, nb);
  TestParameters params;
  int testChannel = 12;
  if (params.isSegmentationRun3) {
    testChannel = 44;
  }
  BOOST_CHECK_EQUAL(ok, false);
  int p = de600.findPadByFEE(307, testChannel);
  BOOST_CHECK_EQUAL(p, b);
  BOOST_CHECK_EQUAL(de600.isValid(b), true);
  BOOST_CHECK_EQUAL(de600.isValid(nb), false);
}

BOOST_AUTO_TEST_CASE(CheckOnePadPositionPresentOnOnlyBendingPlaneDE825)
{
  Segmentation de825(825);
  double x = 110.09;
  double y = -1.25;
  int b, nb;
  bool ok = de825.findPadPairByPosition(x, y, b, nb);
  TestParameters params;
  int testChannel{55};
  if (params.isSegmentationRun3) {
    testChannel = 23;
  }
  BOOST_CHECK_EQUAL(ok, false);
  int p = de825.findPadByFEE(1333, testChannel);
  BOOST_CHECK_EQUAL(p, nb);
  BOOST_CHECK_EQUAL(de825.isValid(b), false);
  BOOST_CHECK_EQUAL(de825.isValid(nb), true);
}

BOOST_AUTO_TEST_CASE(CheckOnePadPositionTotallyAbsentDE604)
{
  Segmentation de604(604);
  double x = -19.25;
  double y = 20.04;
  int b, nb;
  bool ok = de604.findPadPairByPosition(x, y, b, nb);
  BOOST_CHECK_EQUAL(ok, false);
  BOOST_CHECK_EQUAL(de604.isValid(nb), false);
  BOOST_CHECK_EQUAL(de604.isValid(b), false);
}

// All the remaining tests of this file are using specific
// detection elements DE100

BOOST_FIXTURE_TEST_SUITE(DE100, SEG100)

BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(1E-3))
BOOST_AUTO_TEST_CASE(CheckOnePosition)
{
  int b, nb;
  bool ok = seg.findPadPairByPosition(24.2, 23.70, b, nb);
  BOOST_CHECK_EQUAL(ok, true);
  BOOST_TEST(seg.padPositionX(b) == 24.255);
  BOOST_TEST(seg.padPositionY(b) == 23.73);
  BOOST_TEST(seg.padSizeX(b) == 0.63);
  BOOST_TEST(seg.padSizeY(b) == 0.42);
}

bool checkSameCathode(const Segmentation& seg, int depadindex, const std::vector<int>& padindices)
{
  bool isBending = seg.isBendingPad(depadindex);
  for (auto n : padindices) {
    if (seg.isBendingPad(n) != isBending) {
      return false;
    }
  }
  return true;
}

struct PadInfo {
  int fec, ch;
  double x, y, sx, sy;
};

// areEqual returns true if the two values are within 1 micron.
bool areEqual(double a, double b)
{
  return std::fabs(a - b) < 1E-4; // 1 micron expressed in centimeters
}

// testNeighbours returns true if the neighbours of dePadIndex, as
// returned by the Segmentation::forEachNeighbouringPad, are the same
// as the elements of expected vector.
bool testNeighbours(const Segmentation& seg, int dePadIndex, std::vector<PadInfo>& expected)
{
  std::vector<int> nei;
  seg.forEachNeighbouringPad(dePadIndex, [&nei](int depadindex) {
    nei.push_back(depadindex);
  });

  if (nei.size() != expected.size()) {
    return false;
  }
  auto notFound = nei.size();
  for (auto n : nei) {
    for (auto e : expected) {
      if (seg.padDualSampaId(n) == e.fec &&
          seg.padDualSampaChannel(n) == e.ch &&
          areEqual(seg.padPositionX(n), e.x) &&
          areEqual(seg.padPositionY(n), e.y) &&
          areEqual(seg.padSizeX(n), e.sx) &&
          areEqual(seg.padSizeY(n), e.sy)) {
        notFound--;
      }
    }
  }
  return notFound == 0;
}

BOOST_AUTO_TEST_CASE(CheckOnePadNeighbours)
{
  // Below are the neighbouring pads of the pad(s) @ (24.0, 24.0)cm
  // for DE 100.
  // What is tested below is not the PAD (index might depend on
  // the underlying implementation) but the rest of the information :
  // (FEC,CH), (X,Y), (SX,SY)
  //
  // PAD       5208 FEC   95 CH  0 X  23.625 Y  23.730 SX   0.630 SY   0.420
  // PAD       5209 FEC   95 CH  3 X  23.625 Y  24.150 SX   0.630 SY   0.420
  // PAD       5210 FEC   95 CH  4 X  23.625 Y  24.570 SX   0.630 SY   0.420
  // PAD       5226 FEC   95 CH 42 X  24.255 Y  24.570 SX   0.630 SY   0.420
  // PAD       5242 FEC   95 CH 43 X  24.885 Y  24.570 SX   0.630 SY   0.420
  // PAD       5241 FEC   95 CH  2 X  24.885 Y  24.150 SX   0.630 SY   0.420
  // PAD       5240 FEC   95 CH 46 X  24.885 Y  23.730 SX   0.630 SY   0.420
  // PAD       5224 FEC   95 CH 31 X  24.255 Y  23.730 SX   0.630 SY   0.420
  // PAD      19567 FEC 1119 CH 48 X  23.310 Y  23.520 SX   0.630 SY   0.420
  // PAD      19568 FEC 1119 CH 46 X  23.310 Y  23.940 SX   0.630 SY   0.420
  // PAD      19569 FEC 1119 CH  0 X  23.310 Y  24.360 SX   0.630 SY   0.420
  // PAD      19585 FEC 1119 CH 42 X  23.940 Y  24.360 SX   0.630 SY   0.420
  // PAD      19601 FEC 1119 CH  1 X  24.570 Y  24.360 SX   0.630 SY   0.420
  // PAD      19600 FEC 1119 CH 44 X  24.570 Y  23.940 SX   0.630 SY   0.420
  // PAD      19599 FEC 1119 CH 30 X  24.570 Y  23.520 SX   0.630 SY   0.420
  // PAD      19583 FEC 1119 CH 29 X  23.940 Y  23.520 SX   0.630 SY   0.420

  std::vector<PadInfo> bendingNeighbours{
    {95, 0, 23.625, 23.730, 0.630, 0.420},
    {95, 3, 23.625, 24.150, 0.630, 0.420},
    {95, 4, 23.625, 24.570, 0.630, 0.420},
    {95, 42, 24.255, 24.570, 0.630, 0.420},
    {95, 43, 24.885, 24.570, 0.630, 0.420},
    {95, 2, 24.885, 24.150, 0.630, 0.420},
    {95, 46, 24.885, 23.730, 0.630, 0.420},
    {95, 31, 24.255, 23.730, 0.630, 0.420}};

  std::vector<PadInfo> nonBendingNeighbours{
    {1119, 48, 23.310, 23.520, 0.630, 0.420},
    {1119, 46, 23.310, 23.940, 0.630, 0.420},
    {1119, 0, 23.310, 24.360, 0.630, 0.420},
    {1119, 42, 23.940, 24.360, 0.630, 0.420},
    {1119, 1, 24.570, 24.360, 0.630, 0.420},
    {1119, 44, 24.570, 23.940, 0.630, 0.420},
    {1119, 30, 24.570, 23.520, 0.630, 0.420},
    {1119, 29, 23.940, 23.520, 0.630, 0.420}};

  TestParameters params;
  if (params.isSegmentationRun3) {
    for (auto& p : bendingNeighbours) {
      p.ch = manu2ds(100, p.ch);
    }
    for (auto& p : nonBendingNeighbours) {
      p.ch = manu2ds(100, p.ch);
    }
  }

  int pb, pnb;
  bool ok = seg.findPadPairByPosition(24.0, 24.0, pb, pnb);
  BOOST_CHECK_EQUAL(ok, true);
  BOOST_CHECK_EQUAL(testNeighbours(seg, pb, bendingNeighbours), true);
  BOOST_CHECK_EQUAL(testNeighbours(seg, pnb, nonBendingNeighbours), true);
}

BOOST_AUTO_TEST_CASE(CircularTest)
{
  std::vector<std::pair<int, int>> tp{
    {95, 45},
    {1119, 45} // both pads @pos 24.0, 24.0cm
  };

  for (auto p : tp) {
    auto dsid = p.first;
    auto dsch = p.second;
    auto dePadIndex = seg.findPadByFEE(dsid, dsch);
    BOOST_CHECK_EQUAL(seg.padDualSampaId(dePadIndex), dsid);
    BOOST_CHECK_EQUAL(seg.padDualSampaChannel(dePadIndex), dsch);
  }
}

BOOST_AUTO_TEST_CASE(ThrowsIfDualSampaChannelIsNotBetween0And63)
{
  BOOST_CHECK_THROW(seg.findPadByFEE(102, -1), std::out_of_range);
  BOOST_CHECK_THROW(seg.findPadByFEE(102, 64), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(ReturnsTrueIfPadIsConnected) { BOOST_CHECK_EQUAL(seg.isValid(seg.findPadByFEE(102, 3)), true); }

BOOST_AUTO_TEST_CASE(ReturnsFalseIfPadIsNotConnected)
{
  TestParameters params;
  int testChannel{14};
  if (params.isSegmentationRun3) {
    testChannel = 39;
  }
  BOOST_CHECK_EQUAL(seg.isValid(seg.findPadByFEE(214, testChannel)), false);
}

BOOST_AUTO_TEST_CASE(ReturnsFalseIfCatPadIdIsOutOfRange)
{
  forOneDetectionElementOfEachSegmentationType([](int detElemId) {
    Segmentation seg{detElemId};
    BOOST_TEST_INFO_SCOPE(fmt::format("DeId {}", detElemId));
    BOOST_CHECK_EQUAL(seg.isValid(0), true);
    BOOST_CHECK_EQUAL(seg.isValid(seg.nofPads() - 1), true);
    BOOST_CHECK_EQUAL(seg.isValid(-1), false);
    BOOST_CHECK_EQUAL(seg.isValid(seg.nofPads()), false);
  });
}
BOOST_AUTO_TEST_CASE(HasPadByPosition)
{
  int b, nb;
  bool ok = seg.findPadPairByPosition(40.0, 30.0, b, nb);
  BOOST_CHECK_EQUAL(ok, true);
}

BOOST_AUTO_TEST_CASE(CheckOnePadPositionPresentOnOnlyBendingPlane)
{
  double x = 1.575;
  double y = 18.69;
  int b, nb;
  bool ok = seg.findPadPairByPosition(x, y, b, nb);
  TestParameters params;
  int testChannel{9};
  if (params.isSegmentationRun3) {
    testChannel = 47;
  }
  BOOST_CHECK_EQUAL(ok, false);
  BOOST_CHECK_EQUAL(seg.findPadByFEE(76, testChannel), b);
  BOOST_CHECK_EQUAL(seg.isValid(nb), false);
}

BOOST_AUTO_TEST_CASE(WhenOnlyOneCathodeHasAPadTheValidIndexMustRelativeToDeNotToCathode)
{
  double x = 8.8;
  double y = 18;
  int b, nb;
  bool ok = seg.findPadPairByPosition(x, y, b, nb);
  BOOST_CHECK_EQUAL(ok, false);
  BOOST_CHECK_EQUAL(seg.isValid(nb), true);
  BOOST_CHECK_EQUAL(seg.isValid(b), false);
  BOOST_CHECK_EQUAL(seg.isBendingPad(nb), false);
}

BOOST_AUTO_TEST_CASE(CheckCopy)
{
  Segmentation copy{seg};
  BOOST_TEST((copy == seg));
  BOOST_TEST(copy.nofPads() == seg.nofPads());
}

BOOST_AUTO_TEST_CASE(CheckAssignment)
{
  Segmentation copy{200};
  copy = seg;
  BOOST_TEST((copy == seg));
  BOOST_TEST(copy.nofPads() == seg.nofPads());
}

BOOST_AUTO_TEST_CASE(RangeOfDualSampaIdIs1To1361)
{
  int minDsId{std::numeric_limits<int>::max()};
  int maxDsId{std::numeric_limits<int>::min()};

  forOneDetectionElementOfEachSegmentationType([&](int detElemId) {
    const auto& seg = o2::mch::mapping::segmentation(detElemId);
    seg.forEachDualSampa([&](int dualSampaId) {
      minDsId = std::min(minDsId, dualSampaId);
      maxDsId = std::max(maxDsId, dualSampaId);
    });
  });
  BOOST_CHECK_EQUAL(minDsId, 1);
  BOOST_CHECK_EQUAL(maxDsId, 1361);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
