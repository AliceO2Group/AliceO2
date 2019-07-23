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

#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "TGeoManager.h"
#include "MCHSimulation/Digit.h"
#include "MCHSimulation/Digitizer.h"
#include "MCHSimulation/Hit.h"
#include "MCHSimulation/Geometry.h"
#include "MCHSimulation/GeometryTest.h"
#include "MCHMappingInterface/Segmentation.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TGeoManager.h"
#include "boost/format.hpp"
#include <boost/test/data/test_case.hpp>

struct GEOMETRY {
  GEOMETRY()
  {
    if (!gGeoManager) {
      o2::mch::test::createStandaloneGeometry();
    }
  }
};

/// \brief Test of the Digitization
/// A couple of values are filled into Hits and we check whether we get reproducible output in terms of digits
/// and MClabels

BOOST_AUTO_TEST_SUITE(o2_mch_simulation)

BOOST_AUTO_TEST_CASE(DigitizerTest)
{

  o2::mch::Digitizer digitizer;
  int trackId1 = 0;
  int trackId2 = 1;
  short detElemId1 = 101;
  short detElemId2 = 1012;
  Point3D<float> entrancePoint1(-17.7993, 8.929883, -522.201); //x,y,z coordinates in cm
  Point3D<float> exitPoint1(-17.8136, 8.93606, -522.62);
  Point3D<float> entrancePoint2(-49.2793, 28.8673, -1441.25);
  Point3D<float> exitPoint2(-49.2965, 28.8806, -1441.75);
  float eloss1 = 1e-6;
  float eloss2 = 1e-6;
  float length = 0.f;
  float tof = 0.0;

  std::vector<o2::mch::Hit> hits(2);
  hits.at(0) = o2::mch::Hit(trackId1, detElemId1, entrancePoint1, exitPoint1, eloss1, length, tof);
  hits.at(1) = o2::mch::Hit(trackId2, detElemId2, entrancePoint2, exitPoint2, eloss2, length, tof);

  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mctruthcontainer;
  std::vector<o2::mch::Digit> digits;
  o2::mch::mapping::Segmentation seg1{ detElemId1 };
  o2::mch::mapping::Segmentation seg2{ detElemId2 };
  digitizer.process(hits, digits);
  digitizer.provideMC(mctruthcontainer);

  int digitcounter1 = 0;
  int digitcounter2 = 0;
  int count = 0;

  for (auto& digit : digits) {

    int padid = digit.getPadID();
    auto label = (mctruthcontainer.getLabels(count))[0];
    int trackID = label.getTrackID();
    ++count;

    if (trackID == trackId1) {
      bool check = seg1.isValid(digit.getPadID());
      if (!check)
        BOOST_FAIL(" digit-pad not belonging to hit det-element-ID ");
      double padposX = seg1.padPositionX(padid);
      double padsizeX = seg1.padSizeX(padid);
      double padposY = seg1.padPositionY(padid);
      double padsizeY = seg1.padSizeY(padid);
      auto t = o2::mch::getTransformation(detElemId1, *gGeoManager);

      Point3D<float> pos(hits.at(0).GetX(), hits.at(0).GetY(), hits.at(0).GetZ());
      Point3D<float> lpos;
      t.MasterToLocal(pos, lpos);

      BOOST_CHECK_CLOSE(lpos.x(), padposX, padsizeX * 4.0);
      BOOST_CHECK_CLOSE(lpos.y(), padposY, padsizeY * 10.0);
      //non uniform pad sizes?
      digitcounter1++;
    } else if (trackID == trackId2) {
      bool check = seg2.isValid(digit.getPadID());
      if (!check)
        BOOST_FAIL(" digit-pad not belonging to hit det-element-ID ");
      double padposX = seg2.padPositionX(padid);
      double padsizeX = seg2.padSizeX(padid);
      double padposY = seg2.padPositionY(padid);
      double padsizeY = seg2.padSizeY(padid);
      auto t = o2::mch::getTransformation(detElemId2, *gGeoManager);

      Point3D<float> pos(hits.at(1).GetX(), hits.at(1).GetY(), hits.at(1).GetZ());
      Point3D<float> lpos;
      t.MasterToLocal(pos, lpos);

      BOOST_CHECK_CLOSE(lpos.x(), padposX, padsizeX * 4.0);
      BOOST_CHECK_CLOSE(lpos.y(), padposY, padsizeY * 10.0);
      digitcounter2++;

    } else {
      BOOST_FAIL(" MC-labels not matching between hit and digit ");
    };
  }

  if (digitcounter1 == 0)
    BOOST_FAIL(" no digit at all from hit in station 1 ");
  if (digitcounter1 > 9)
    BOOST_FAIL("more than 10 digits for one hit in station 1 ");
  if (digitcounter2 == 0)
    BOOST_FAIL(" no digit at all from hit in station 2 ");
  if (digitcounter2 > 9)
    BOOST_FAIL(" more than 10 digits for one hit in station 2 ");
}

BOOST_AUTO_TEST_CASE(mergingDigitizer)
{
  //merging
  o2::mch::Digitizer digitizer;
  int trackId1 = 0;
  int trackId2 = 1;
  short detElemId1 = 101;
  short detElemId2 = 1012;
  Point3D<float> entrancePoint1(-17.7993, 8.929883, -522.201); //x,y,z coord. (cm)
  Point3D<float> exitPoint1(-17.8136, 8.93606, -522.62);
  Point3D<float> entrancePoint2(-49.2793, 28.8673, -1441.25);
  Point3D<float> exitPoint2(-49.2965, 28.8806, -1441.75);
  float eloss1 = 1e-6;
  float eloss2 = 1e-6;
  float length = 0.f;
  float tof = 0.0;

  std::vector<o2::mch::Hit> hits(2);
  hits.at(0) = o2::mch::Hit(trackId1, detElemId1, entrancePoint1, exitPoint1, eloss1, length, tof);
  hits.at(1) = o2::mch::Hit(trackId2, detElemId2, entrancePoint2, exitPoint2, eloss2, length, tof);

  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mctruthcontainer;
  std::vector<o2::mch::Digit> digits;

  digitizer.process(hits, digits);
  digitizer.provideMC(mctruthcontainer);

  std::vector<o2::MCCompLabel> labels = digitizer.getTrackLabels();

  int rep1 = 9;
  int rep2 = 9;

  for (int i = 0; i < rep1; i++) {
    digits.emplace_back(digits.at(0).getTimeStamp(), digits.at(0).getDetID(), digits.at(0).getPadID(), digits.at(0).getADC());
    labels.emplace_back(labels.at(0).getTrackID(), labels.at(0).getEventID(), labels.at(0).getSourceID(), false);
  }
  for (int i = 0; i < rep2; i++) {
    digits.emplace_back(digits.at(1).getTimeStamp(), digits.at(0).getDetID(), digits.at(1).getPadID(), digits.at(1).getADC());
    labels.emplace_back(labels.at(1).getTrackID(), labels.at(1).getEventID(), labels.at(1).getSourceID(), false);
  }

  digitizer.mergeDigits(digits, labels);

  std::vector<o2::mch::Digit> mergeddigits = digitizer.getDigits();
  std::vector<o2::MCCompLabel> mergedlabels = digitizer.getTrackLabels();

  BOOST_CHECK_CLOSE(mergeddigits.at(0).getADC(), digits.at(0).getADC() * (float)(rep1 + 1), digits.at(0).getADC() / 10000.);
  BOOST_CHECK_CLOSE(mergeddigits.at(1).getADC(), digits.at(1).getADC() * (float)(rep2 + 1), digits.at(1).getADC() / 10000.);
  BOOST_CHECK_CLOSE((float)mergedlabels.at(0).getTrackID(), (float)labels.at(0).getTrackID(), 0.1);
  BOOST_CHECK_CLOSE((float)mergedlabels.at(1).getTrackID(), (float)labels.at(1).getTrackID(), 0.1);

  BOOST_CHECK_CLOSE((float)(digits.size() - rep1 - rep2), (float)mergeddigits.size(), 0.1);

} //testing

BOOST_AUTO_TEST_SUITE_END()
