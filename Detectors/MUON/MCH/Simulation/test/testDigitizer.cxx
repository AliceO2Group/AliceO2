// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// \file testDigitisation.cxx
/// \brief This task tests the Digitizer and the Response of the MCH digitization
/// \author Michael Winn, DPhN/IRFU/CEA, michael.winn@cern.ch

#define BOOST_TEST_MODULE Test MCHSimulation Digitization
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "CommonUtils/ConfigurableParam.h"
#include "DataFormatsMCH/Digit.h"
#include "MCHGeometryCreator/Geometry.h"
#include "MCHGeometryTest/Helpers.h"
#include "MCHGeometryTransformer/Transformations.h"
#include "MCHMappingInterface/Segmentation.h"
#include "MCHSimulation/Digitizer.h"
#include "MCHSimulation/Hit.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TGeoManager.h"
#include "boost/format.hpp"
#include <boost/test/data/test_case.hpp>

using IR = o2::InteractionRecord;
using o2::mch::Digit;
using o2::mch::groupIR;
using o2::mch::Hit;
using o2::mch::mapping::Segmentation;

struct GEOMETRY {
  GEOMETRY()
  {
    if (!gGeoManager) {
      o2::mch::test::createStandaloneGeometry();
    }
  }
};

namespace
{
o2::math_utils::Point3D<float> entrancePoint1(-17.7993, 8.929883, -522.201); //x,y,z coordinates in cm
o2::math_utils::Point3D<float> exitPoint1(-17.8136, 8.93606, -522.62);
o2::math_utils::Point3D<float> entrancePoint2(-49.2793, 28.8673, -1441.25);
o2::math_utils::Point3D<float> exitPoint2(-49.2965, 28.8806, -1441.75);

int check(const Hit& hit, const Digit& digit, Segmentation& seg, o2::mch::geo::TransformationCreator transformation)
{
  int padid = digit.getPadID();
  bool check = seg.isValid(padid);
  if (!check) {
    BOOST_FAIL(" digit-pad not belonging to hit det-element-ID ");
    return 0;
  }
  double padposX = seg.padPositionX(padid);
  double padsizeX = seg.padSizeX(padid);
  double padposY = seg.padPositionY(padid);
  double padsizeY = seg.padSizeY(padid);
  auto t = transformation(digit.getDetID());

  o2::math_utils::Point3D<float> pos(hit.GetX(), hit.GetY(), hit.GetZ());
  o2::math_utils::Point3D<float> lpos;
  t.MasterToLocal(pos, lpos);

  // very loose check : check that digit position is within 10 pads of the
  // hit center in both directions
  BOOST_CHECK(std::abs(lpos.x() - padposX) < padsizeX * 10);
  BOOST_CHECK(std::abs(lpos.y() - padposY) < padsizeY * 10);
  return 1;
}
} // namespace

/// \brief Test of the Digitization
/// A couple of values are filled into Hits and we check whether we get reproducible output in terms of digits
/// and MClabels

BOOST_FIXTURE_TEST_SUITE(digitization, GEOMETRY)

BOOST_AUTO_TEST_CASE(DigitizerTest)
{
  // FIXME: must set a (global) seed here to get reproducible results

  auto transformation = o2::mch::geo::transformationFromTGeoManager(*gGeoManager);

  o2::mch::Digitizer digitizer(transformation);
  int trackId1 = 0;
  int trackId2 = 1;
  short detElemId1 = 101;
  short detElemId2 = 1012;

  float eloss1 = 1e-6;
  float eloss2 = 1e-6;
  float length = 0.f;
  float tof = 0.0;

  std::vector<o2::mch::Hit> hits(2);
  hits.at(0) = o2::mch::Hit(trackId1, detElemId1, entrancePoint1, exitPoint1, eloss1, length, tof);
  hits.at(1) = o2::mch::Hit(trackId2, detElemId2, entrancePoint2, exitPoint2, eloss2, length, tof);

  Segmentation seg1{detElemId1};
  Segmentation seg2{detElemId2};

  digitizer.startCollision({0, 0});

  o2::conf::ConfigurableParam::setValue("MCHDigitizer", "noiseProba", 0.0);
  digitizer.processHits(hits, 0, 0);

  std::vector<Digit> digits;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> labels;

  digitizer.extractDigitsAndLabels(digits, labels);

  int digitcounter1 = 0;
  int digitcounter2 = 0;
  int count = 0;

  for (const auto& digit : digits) {
    auto label = (labels.getLabels(count))[0];
    int trackID = label.getTrackID();
    ++count;

    if (trackID == trackId1) {
      digitcounter1 += check(hits.at(0), digit, seg1, transformation);
    } else if (trackID == trackId2) {
      digitcounter2 += check(hits.at(1), digit, seg2, transformation);
    } else {
      BOOST_FAIL(" MC-labels not matching between hit and digit ");
    }
  }
  BOOST_TEST(digitcounter1 > 0);
  BOOST_TEST(digitcounter1 < 20);
  BOOST_TEST(digitcounter2 > 0);
  BOOST_TEST(digitcounter2 < 10);
}

const std::vector<IR> testIRs = {
  /* bc, orbit */
  {123, 0},
  {125, 0},
  {125, 0},
  {125, 1},
  {130, 1},
  {134, 1},
  {135, 1},
  {137, 1},
};

bool isSame(const std::map<IR, std::vector<int>>& result,
            const std::map<IR, std::vector<int>>& expected)
{
  if (result == expected) {
    return true;
  }
  std::cout << result.size() << " " << expected.size() << "\n";

  std::cout << "Expected:\n";
  for (auto p : expected) {
    std::cout << p.first << "-> ";
    for (auto v : p.second) {
      std::cout << v << ",";
    }
    std::cout << "\n";
  }
  std::cout << "Got:\n";
  for (auto p : result) {
    std::cout << p.first << "-> ";
    for (auto v : p.second) {
      std::cout << v << ",";
    }
    std::cout << "\n";
  }
  return false;
}

BOOST_AUTO_TEST_CASE(GroupIRMustThrowOnNonSortedRecords)
{
  const std::vector<IR> notSorted = {
    {125, 0},
    {123, 0},
  };
  BOOST_CHECK_THROW(groupIR(notSorted), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(IdenticalIRsShouldBeMerged)
{
  std::map<IR, std::vector<int>> expected;
  expected[IR{123, 0}] = {0};
  expected[IR{125, 0}] = {1, 2};
  expected[IR{125, 1}] = {3};
  expected[IR{130, 1}] = {4};
  expected[IR{134, 1}] = {5};
  expected[IR{135, 1}] = {6};
  expected[IR{137, 1}] = {7};

  auto g = groupIR(testIRs, 0);
  BOOST_CHECK_EQUAL(isSame(g, expected), true);
}

BOOST_AUTO_TEST_CASE(IRSeparatedByLessThan4BCShouldBeMerged)
{
  std::map<IR, std::vector<int>> expected;
  expected[IR{123, 0}] = {0, 1, 2};
  expected[IR{125, 1}] = {3};
  expected[IR{130, 1}] = {4};
  expected[IR{134, 1}] = {5, 6, 7};

  auto g = groupIR(testIRs, 4);
  BOOST_CHECK_EQUAL(isSame(g, expected), true);
}
BOOST_AUTO_TEST_SUITE_END()
