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

/// \file testCoordinateTransformscxx
/// \brief Test local to row-column (float) coordinate transformations in PadPlane class
/// \author Jason Barrella - jbarrell@cern.ch, Sean Murray - murrays@cern.ch

#define BOOST_TEST_MODULE Test CoordinateTransforms
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <numeric>

#include "DataFormatsTRD/Constants.h"
#include "TRDBase/TrackletTransformer.h"

#include "TRDBase/PadPlane.h"
#include "DetectorsBase/GeometryManager.h"
#include "TRDBase/Geometry.h"

namespace o2
{
namespace trd
{

using namespace o2::trd::constants;
using namespace std;

void testRCPoint(double calculatedPoint, double predictedPoint)
{
  float e = 0.000001;
  BOOST_CHECK(fabs(predictedPoint - calculatedPoint) <= e);
}

BOOST_AUTO_TEST_CASE(LocaltoRCTest)
{
  auto mGeo = o2::trd::Geometry::instance();
  mGeo->createPadPlaneArray();

  int hcid = 776;
  // This C1 chamber has 16 pad rows with I pad length = 90mm and O pad length = 75mm
  int detector = hcid / 2;
  int stack = mGeo->getStack(detector);
  int layer = mGeo->getLayer(detector);

  auto padPlane = mGeo->getPadPlane(layer, stack);
  double lengthIPad = padPlane->getLengthIPad();
  double lengthOPad = padPlane->getLengthOPad();

  double padIWidth = padPlane->getWidthIPad();
  double padOWidth = padPlane->getWidthOPad();

  double tiltingAngle = padPlane->getTiltingAngle();

  // Test padrows
  auto p1 = padPlane->getPadRow(0);
  // Center of the chamber. This should return the lower edge of padrow 8.
  // Since we are using float values, the lower edge of padrow 8 correspond with float value 8.0.
  testRCPoint(p1, 8.);

  auto p2 = padPlane->getPadRow(lengthIPad / 2.);
  // With an I pad length of 9 cm, 9. / 2. should put us half way into the preceeding pad (pad 7) since padrow number
  // decreses in positive z direction.
  testRCPoint(p2, 7.5);

  auto p3 = padPlane->getPadRow(-lengthIPad / 2.);
  // Same as above but in the other direction.
  testRCPoint(p3, 8.5);

  auto p4 = padPlane->getPadRow(lengthIPad * 4.2);
  // Arbitrary distance in z.
  testRCPoint(p4, 8 - 4.2);

  auto p5 = padPlane->getPadRow(lengthIPad * 7 + lengthOPad);
  // Lower border case. Take center and add 7 pads * 9 cm + 1 pad * 7.5 cm.
  testRCPoint(p5, 0);

  auto p6 = padPlane->getPadRow(-lengthIPad * 7 - lengthOPad);
  // Upper border case. Take center and subtract 7 pads * 9 cm - 1 pad * 7.5 cm.
  testRCPoint(p6, 16);

  // Test pads with pad tilting
  double p13 = padPlane->getPad(0, 0.01);
  // Center of chamber plus epsilon (in z). Note the discontinuity in pad vs. z at z=0. This puts us at a point on the lower
  // (in z direction) end of padrow 7. Since we have a pad tilt of -2 deg (pads tilted clockwise for particle coming
  // from interaction vertex). After a lot thought and doodles on Miro (https://miro.com/app/board/o9J_lKgybMc=/) we
  // find that we expect a small negative offset which would place us in the upper half of pad 71.
  // To calculate that offset, we multiply the tangent of the tilting angle by the distance that our point is away
  // from the center of its local padrow. Some unit conversions are also neccessary...
  testRCPoint(p13, 72 + TMath::Tan(TMath::DegToRad() * tiltingAngle) * (0.5 * lengthIPad - 0.01) / padIWidth);

  double p14 = padPlane->getPad(0, -lengthIPad / 2.);
  // Move from center of chamber to to center of padrow 8 = 8.5.
  // This should now place us as edge of pad 72 = 72.0 since no offset is applied from pad tilt.
  testRCPoint(p14, 72);

  double p15 = padPlane->getPad(0, lengthIPad / 2.);
  // Should be the same in the other direction at padrow = 7.5
  testRCPoint(p15, 72);

  double p16 = padPlane->getPad(padIWidth * 42, lengthIPad / 2.);
  // Adding an arbitrary number of pads should just increase position by same number
  testRCPoint(p16, 72 + 42);

  double p17 = padPlane->getPad(padIWidth * 42, -lengthIPad * 4 - 2.3);
  // Moving to arbitrary point in both y and z. In y, we are 42 pads from the center which would be 72 + 42 = 114
  // if we were in the center of the padrow and no pad tilting is considered. However, we are not in the center of
  // a padrow, but rather 2.3 cm into padrow 8 + 4 = 12. This puts us below the center which is at 4.5 cm
  // into the padrow and therefore, we expect a small postiive offset since the pad tilting angle is -2 deg.
  testRCPoint(p17, 72 + 42 + TMath::Tan(TMath::DegToRad() * tiltingAngle) * (2.3 - 0.5 * lengthIPad) / padIWidth);

  double p18 = padPlane->getPad(padIWidth * 71 + padOWidth, lengthIPad / 2.);
  // Border case right on the upper edge of the padrow in y direction and at the center of the padrow in z
  testRCPoint(p18, 144);

  double p19 = padPlane->getPad(-padIWidth * 71 - padOWidth, lengthIPad / 2.);
  // Border case right on the lower edge of the padrow in y direction and at the center of the padrow in z
  testRCPoint(p19, 0);

  double p20 = padPlane->getPad(0, lengthIPad * 7 + 1.5);
  // Ensure that shorter length of outer padrows is considered correctly
  testRCPoint(p20, 72 + TMath::Tan(TMath::DegToRad() * tiltingAngle) * (0.5 * lengthOPad - 1.5) / padIWidth);
}

} // namespace trd
} // namespace o2
