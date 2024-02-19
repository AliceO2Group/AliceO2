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
#include "MCHSimulation/DigitizerParam.h"
#include "MCHSimulation/Hit.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TGeoManager.h"
#include "boost/format.hpp"
#include <boost/test/data/test_case.hpp>
#include <boost/property_tree/ptree.hpp>
#include <algorithm>
#include <unordered_map>

using IR = o2::InteractionRecord;
using o2::mch::Digit;
using o2::mch::Hit;
using o2::mch::ROFRecord;
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
short detElemId1 = 101;
o2::math_utils::Point3D<float> entrancePoint1(-17.7993, 8.929883, -522.201); // x,y,z coordinates in cm
o2::math_utils::Point3D<float> exitPoint1(-17.8136, 8.93606, -522.62);
short detElemId2 = 1012;
o2::math_utils::Point3D<float> entrancePoint2(-49.2793, 28.8673, -1441.25);
o2::math_utils::Point3D<float> exitPoint2(-49.2965, 28.8806, -1441.75);
short detElemId3 = 1012;
o2::math_utils::Point3D<float> entrancePoint3(-50.1793, 28.2673, -1441.25);
o2::math_utils::Point3D<float> exitPoint3(-50.1965, 28.2806, -1441.75);

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
  o2::math_utils::Point3D<float> lpos{};
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
  auto transformation = o2::mch::geo::transformationFromTGeoManager(*gGeoManager);
  o2::conf::ConfigurableParam::setValue("MCHDigitizer", "seed", 123);
  o2::mch::Digitizer digitizer(transformation);

  int trackId1 = 0;
  int trackId2 = 1;
  int trackId3 = 2;
  IR collisionTime1(3, 1);
  IR collisionTime2(5, 1);
  float eloss = 1.e-6;
  float length = 0.f;
  float tof = 0.f;

  std::vector<o2::mch::Hit> hits1(2);
  hits1.at(0) = o2::mch::Hit(trackId1, detElemId1, entrancePoint1, exitPoint1, eloss, length, tof);
  hits1.at(1) = o2::mch::Hit(trackId2, detElemId2, entrancePoint2, exitPoint2, eloss, length, tof);

  std::vector<o2::mch::Hit> hits2(1);
  hits2.at(0) = o2::mch::Hit(trackId3, detElemId3, entrancePoint3, exitPoint3, eloss, length, tof);

  digitizer.processHits(hits1, collisionTime1, 0, 0);
  digitizer.processHits(hits2, collisionTime2, 0, 0);

  auto firstIR = IR::long2IR(std::max(int64_t(0), collisionTime1.toLong() - 100));
  auto lastIR = collisionTime2 + 100;
  digitizer.addNoise(firstIR, lastIR);

  std::vector<ROFRecord> rofs{};
  std::vector<Digit> digits{};
  o2::dataformats::MCLabelContainer labels{};
  digitizer.digitize(rofs, digits, labels);

  Segmentation seg1{detElemId1};
  Segmentation seg2{detElemId2};
  int digitcounter1 = 0;
  int digitcounter2 = 0;
  int digitcounter3 = 0;
  int64_t previousROFtime = -1;
  std::unordered_map<int, Digit> digitsMap{};

  for (const auto& rof : rofs) {

    // check ROF alignment on 4 BC
    if (rof.getBCData().bc % 4 != 0) {
      BOOST_FAIL(" ROF IR not aligned on 4 BC ");
    }

    // check ROFs ordering in ascending IR
    auto rofTime = rof.getBCData().toLong();
    if (rofTime < previousROFtime) {
      BOOST_FAIL(" ROF not ordered in ascending IR ");
    } else if (rofTime == previousROFtime) {
      BOOST_FAIL(" 2 ROFs with the same IR ");
    }
    previousROFtime = rofTime;

    for (int iDigit = rof.getFirstIdx(); iDigit <= rof.getLastIdx(); ++iDigit) {
      const auto& digit = digits[iDigit];

      // check hit-digit association
      for (const auto& label : labels.getLabels(iDigit)) {
        int trackID = label.getTrackID();
        if (trackID == trackId1) {
          digitcounter1 += check(hits1.at(0), digit, seg1, transformation);
        } else if (trackID == trackId2) {
          digitcounter2 += check(hits1.at(1), digit, seg2, transformation);
        } else if (trackID == trackId3) {
          digitcounter3 += check(hits2.at(0), digit, seg2, transformation);
        } else if (!label.isNoise()) {
          BOOST_FAIL(" MC-labels not matching between hit and digit ");
        }
      }

      // check pileup handling within the readout window
      auto itDigit = digitsMap.emplace((digit.getDetID() << 16) + digit.getPadID(), digit);
      if (!itDigit.second &&
          digit.getTime() - itDigit.first->second.getTime() < 4 * (itDigit.first->second.getNofSamples() + 2)) {
        BOOST_FAIL(" same pad has multiple digits in overlapping readout windows ");
      }
    }
  }

  BOOST_TEST(digitcounter1 > 0);
  BOOST_TEST(digitcounter1 < 20);
  BOOST_TEST(digitcounter2 > 0);
  BOOST_TEST(digitcounter2 < 10);
  BOOST_TEST(digitcounter3 > 0);
  BOOST_TEST(digitcounter3 < 10);
}
BOOST_AUTO_TEST_SUITE_END()
