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

/// \file ITS3Services.h
/// \brief Definition of the ITS3Services class
/// \author Fabrizio Grosa <fgrosa@cern.ch>

#include <TGeoManager.h>        // for TGeoManager
#include <TGeoTube.h>           // for TGeoTube, TGeoTubeSeg
#include <TGeoCompositeShape.h> // for TGeoCompositeShape
#include <TGeoVolume.h>         // for TGeoVolume, TGeoVolumeAssembly
#include <TGeoBBox.h>           // for TGeoBBox
#include <TGeoCone.h>           // for TGeoConeSeg, TGeoCone
#include <TGeoPcon.h>           // for TGeoPcon
#include <TGeoMatrix.h>         // for TGeoCombiTrans, TGeoRotation, etc
#include <TGeoCompositeShape.h> // for TGeoCompositeShape
#include <TMath.h>              // for TMath

#include <fairlogger/Logger.h> // for LOG

#include "ITS3Simulation/ITS3Services.h"

using namespace o2::its3;

/// \cond CLASSIMP
ClassImp(ITS3Services);
/// \endcond

ITS3Services::~ITS3Services() = default;

TGeoVolume* ITS3Services::createCYSSAssembly()
{
  //
  // Creates the CYSS Assembly (i.e. the supporting cylinder and cone)
  // Based on ITSSimulation/V3Services.cxx
  //

  const double sCyssFlangeAZpos = 0.9;
  const double sCyssFlangeCZpos = 0.1;
  const double sIBCYSSFlangeCZPos = 17.15;

  double zlen, zpos;

  TGeoVolume* cyssVol = new TGeoVolumeAssembly("IBCYSSAssembly");
  cyssVol->SetVisibility(true);

  TGeoVolume* cyssCylinder = createCYSSCylinder();
  zlen = (static_cast<TGeoTubeSeg*>(cyssCylinder->GetShape()))->GetDz();
  zpos = sIBCYSSFlangeCZPos - sCyssFlangeCZpos - zlen;
  cyssVol->AddNode(cyssCylinder, 1, new TGeoTranslation(0, 0, -zpos));
  cyssVol->AddNode(cyssCylinder, 2, new TGeoCombiTrans(0, 0, -zpos, new TGeoRotation("", 180, 0, 0)));

  TGeoVolume* cyssCone = createCYSSCone();
  zpos = -zpos + zlen - (static_cast<TGeoPcon*>(cyssCone->GetShape()))->GetZ(2);
  cyssVol->AddNode(cyssCone, 1, new TGeoTranslation(0, 0, zpos));
  cyssVol->AddNode(cyssCone, 2, new TGeoCombiTrans(0, 0, zpos, new TGeoRotation("", 180, 0, 0)));

  TGeoVolume* cyssFlangeA = createCYSSFlangeA();
  int nZPlanes = (static_cast<TGeoPcon*>(cyssCone->GetShape()))->GetNz();
  zpos = zpos + (static_cast<TGeoPcon*>(cyssCone->GetShape()))->GetZ(nZPlanes - 1) + sCyssFlangeAZpos;
  cyssVol->AddNode(cyssFlangeA, 1, new TGeoCombiTrans(0, 0, zpos, new TGeoRotation("", 180, 180, 0)));
  cyssVol->AddNode(cyssFlangeA, 2, new TGeoCombiTrans(0, 0, zpos, new TGeoRotation("", 0, 180, 0)));

  TGeoVolume* cyssFlangeC = createCYSSFlangeC();
  zpos = sIBCYSSFlangeCZPos;
  cyssVol->AddNode(cyssFlangeC, 1, new TGeoTranslation(0, 0, -zpos));
  cyssVol->AddNode(cyssFlangeC, 2, new TGeoCombiTrans(0, 0, -zpos, new TGeoRotation("", 180, 0, 0)));

  // Return the whole assembly
  return cyssVol;
}

TGeoVolume* ITS3Services::createCYSSCylinder()
{
  //
  // Creates the cylinder of the Inner Barrel CYSS
  // (ALIITSUP0191)
  // Based on ITSSimulation/V3Services.cxx
  //

  const double sCyssCylInnerD = 9.56;
  const double sCyssCylOuterD = 10.;
  const double sCyssCylZLength = 35.3;
  const double sCyssCylFabricThick = 0.01;

  // Local variables
  double rmin, rmax, zlen, phimin, phimax, dphi;

  // First create the shapes
  rmin = sCyssCylInnerD / 2;
  rmax = sCyssCylOuterD / 2;
  zlen = sCyssCylZLength / 2;
  TGeoTubeSeg* cyssOuterCylSh = new TGeoTubeSeg(rmin, rmax, zlen, 180, 360);

  rmin += sCyssCylFabricThick;
  rmax -= sCyssCylFabricThick;
  zlen -= sCyssCylFabricThick;

  dphi = std::asin(sCyssCylFabricThick / rmax);
  phimin = 180 + dphi * TMath::RadToDeg();
  phimax = 360 - dphi * TMath::RadToDeg();

  TGeoTubeSeg* cyssInnerCylSh = new TGeoTubeSeg(rmin, rmax, zlen, phimin, phimax);

  // We have all shapes: now create the real volumes
  TGeoMedium* medPrepreg = gGeoManager->GetMedium("IT3_F6151B05M$");
  TGeoMedium* medRohacell = gGeoManager->GetMedium("IT3_ROHACELL$");

  TGeoVolume* cyssOuterCylVol = new TGeoVolume("IBCYSSCylinder", cyssOuterCylSh, medPrepreg);
  cyssOuterCylVol->SetLineColor(35);

  TGeoVolume* cyssInnerCylVol = new TGeoVolume("IBCYSSCylinderFoam", cyssInnerCylSh, medRohacell);
  cyssInnerCylVol->SetLineColor(kGreen);

  cyssOuterCylVol->AddNode(cyssInnerCylVol, 1, nullptr);

  // Finally return the cylinder volume
  return cyssOuterCylVol;
}

TGeoVolume* ITS3Services::createCYSSCone()
{
  //
  // Creates the cone of the Inner Barrel CYSS
  // (ALIITSUP0190)
  // Based on ITSSimulation/V3Services.cxx
  //

  const double sCyssConeTotalLength = 15.;

  const double sCyssConeIntSectDmin = 10.;
  const double sCyssConeIntSectDmax = 10.12;
  const double sCyssConeIntSectZlen = 2.3;
  const double sCyssConeIntCylZlen = 1.5;

  const double sCyssConeExtSectDmin = 24.6;
  const double sCyssConeExtSectDmax = 25.72;
  const double sCyssConeExtSectZlen = 4.2;
  const double sCyssConeExtCylZlen = 4.;

  const double sCyssConeOpeningAngle = 40.; // Deg
  const double sCyssConeFabricThick = 0.03;

  // Local variables
  double rmin, rmax, zlen1, zlen2, phimin, phirot, dphi;
  double x1, y1, x2, y2, x3, y3, m, xin, yin;

  // The CYSS Cone is physically a single piece made by a cylindrical
  // section, a conical section, and a second cylindrical section
  // The cone and the second cylinder have a foam core
  // Both are implemented as two Pcon's

  TGeoPcon* cyssConeSh = new TGeoPcon(180, 180, 6);
  rmin = sCyssConeIntSectDmin / 2;
  rmax = sCyssConeIntSectDmax / 2;
  cyssConeSh->DefineSection(0, 0, rmin, rmax);
  cyssConeSh->DefineSection(1, sCyssConeIntCylZlen, rmin, rmax);
  zlen1 = sCyssConeTotalLength - sCyssConeExtSectZlen;
  rmax = yFrom2Points(sCyssConeIntCylZlen, sCyssConeIntSectDmax / 2, zlen1, sCyssConeExtSectDmax / 2, sCyssConeIntSectZlen);
  cyssConeSh->DefineSection(2, sCyssConeIntSectZlen, rmin, rmax);
  zlen2 = sCyssConeTotalLength - sCyssConeExtCylZlen;
  rmin = yFrom2Points(sCyssConeIntSectZlen, sCyssConeIntSectDmin / 2, zlen2, sCyssConeExtSectDmin / 2, zlen1);
  rmax = sCyssConeExtSectDmax / 2;
  cyssConeSh->DefineSection(3, zlen1, rmin, rmax);
  rmin = sCyssConeExtSectDmin / 2;
  cyssConeSh->DefineSection(4, zlen2, rmin, rmax);
  cyssConeSh->DefineSection(5, sCyssConeTotalLength, rmin, rmax);

  dphi = std::asin(sCyssConeFabricThick / (0.5 * sCyssConeIntSectDmax));
  phimin = 180 + dphi * TMath::RadToDeg();
  phirot = 180 - 2 * dphi * TMath::RadToDeg();

  // The foam cone is built from the points of the outer cone
  TGeoPcon* cyssConeFoamSh = new TGeoPcon(phimin, phirot, 5);

  m = std::tan(sCyssConeOpeningAngle * TMath::DegToRad());
  x1 = cyssConeSh->GetZ(2);
  y1 = cyssConeSh->GetRmin(2);
  x2 = cyssConeSh->GetZ(1);
  y2 = cyssConeSh->GetRmin(1);
  x3 = x1;
  y3 = y2 + m * (x3 - x2);

  insidePoint(x1, y1, x2, y2, x3, y3, -sCyssConeFabricThick, xin, yin);
  cyssConeFoamSh->DefineSection(0, xin, yin, yin);

  x3 = cyssConeSh->GetZ(3);
  y3 = cyssConeSh->GetRmin(3);

  insidePoint(x3, y3, x1, y1, x2, y2, -sCyssConeFabricThick, xin, yin);
  zlen1 = xin;
  rmin = yin;
  rmax = y2 + m * (zlen1 - x2);
  cyssConeFoamSh->DefineSection(1, zlen1, rmin, rmax);

  x1 = cyssConeSh->GetZ(5);
  y1 = cyssConeSh->GetRmax(5);
  x2 = cyssConeSh->GetZ(3);
  y2 = cyssConeSh->GetRmax(3);
  x3 = cyssConeSh->GetZ(2);
  y3 = cyssConeSh->GetRmax(2);

  insidePoint(x1, y1, x2, y2, x3, y3, -sCyssConeFabricThick, xin, yin);
  zlen1 = xin;
  rmin = cyssConeFoamSh->GetRmin(1) + m * (zlen1 - cyssConeFoamSh->GetZ(1));
  rmax = sCyssConeExtSectDmax / 2 - sCyssConeFabricThick;
  cyssConeFoamSh->DefineSection(2, zlen1, rmin, rmax);

  rmin = sCyssConeExtSectDmin / 2 + sCyssConeFabricThick;
  zlen1 = cyssConeSh->GetZ(4);
  cyssConeFoamSh->DefineSection(3, zlen1, rmin, rmax);

  zlen1 = sCyssConeTotalLength - sCyssConeFabricThick;
  cyssConeFoamSh->DefineSection(4, zlen1, rmin, rmax);

  // We have all shapes: now create the real volumes
  TGeoMedium* medPrepreg = gGeoManager->GetMedium("IT3_F6151B05M$");
  TGeoMedium* medRohacell = gGeoManager->GetMedium("IT3_ROHACELL$");

  TGeoVolume* cyssConeVol = new TGeoVolume("IBCYSSCone", cyssConeSh, medPrepreg);
  cyssConeVol->SetLineColor(35);

  TGeoVolume* cyssConeFoamVol = new TGeoVolume("IBCYSSConeFoam", cyssConeFoamSh, medRohacell);
  cyssConeFoamVol->SetLineColor(kGreen);

  cyssConeVol->AddNode(cyssConeFoamVol, 1, nullptr);

  // Finally return the cone volume
  return cyssConeVol;
}

TGeoVolume* ITS3Services::createCYSSFlangeA()
{
  //
  // Creates the Flange on Side A for the Inner Barrel CYSS
  // (ALIITSUP0189)
  // Based on ITSSimulation/V3Services.cxx
  //

  // Radii of the steps
  const double sCyssFlangeAStep1Dmin = 25.41;
  const double sCyssFlangeAStep1Dmax = 28.70;
  const double sCyssFlangeAStep2Dmax = 25.90;
  const double sCyssFlangeAStep3Dmin = 24.30;
  const double sCyssFlangeAStep3Dmax = 24.55;
  const double sCyssFlangeAStep4Dmax = 23.90;
  const double sCyssFlangeAInnerD = 23.60;
  const double sCyssFlangeAInRingD = 23.80;

  // Heights of the steps
  const double sCyssFlangeATotHei = 3.9;
  const double sCyssFlangeAStep1H = 0.55;
  const double sCyssFlangeAInRingH = 0.7;
  const double sCyssFlangeAInRingUp = 0.1;
  const double sCyssFlangeAStep2H = 0.9;
  const double sCyssFlangeAStep3H = 1.;
  const double sCyssFlangeAStep4H = 0.85;

  // The wings
  const double sCyssFlangeAWingD = 30.7;
  const double sCyssFlangeAWingW = 1.6;

  // Holes
  const double sCyssFlangeANotchW = 0.3;
  const double sCyssFlangeAHolesDpos = 27.4;

  const double sCyssFlangeAHole1Num = 8;
  const double sCyssFlangeAHole1D = 0.55;
  const double sCyssFlangeAHole1Phi0 = 10;    // Deg
  const double sCyssFlangeAHole1PhiStep = 20; // Deg

  const double sCyssFlangeAHole2D = 0.4;
  const double sCyssFlangeAHole2Phi = 20; // Deg

  const double sCyssFlangeAHole3D = 0.7;
  const double sCyssFlangeAHole3Phi = 6; // Deg

  const double sCyssFlangeAWingHoleD = 0.81;
  const double sCyssFlangeAWingHoleYpos = 0.9;
  const double sCyssFlangeAWingHoleRpos = 14.6;

  // Local variables
  double rmin, rmax, zlen, phi, dphi;
  double xpos, ypos;

  // The CYSS Flange on Side A is physically a single piece.
  // It is implemented as a CompositeShape of two Pcon's and one TubeSeg
  // minus a huge number of holes

  // The flange body
  TGeoPcon* cyssFlangeABody = new TGeoPcon("cyssflangeabody", 180, 180, 12);

  rmin = sCyssFlangeAStep1Dmin / 2;
  rmax = sCyssFlangeAStep1Dmax / 2;
  cyssFlangeABody->DefineSection(0, 0, rmin, rmax);
  cyssFlangeABody->DefineSection(1, sCyssFlangeAStep1H, rmin, rmax);
  rmax = sCyssFlangeAStep2Dmax / 2;
  cyssFlangeABody->DefineSection(2, sCyssFlangeAStep1H, rmin, rmax);
  cyssFlangeABody->DefineSection(3, sCyssFlangeAInRingH, rmin, rmax);
  rmin = sCyssFlangeAStep3Dmin / 2;
  cyssFlangeABody->DefineSection(4, sCyssFlangeAInRingH, rmin, rmax);
  cyssFlangeABody->DefineSection(5, sCyssFlangeAStep2H, rmin, rmax);
  rmax = sCyssFlangeAStep3Dmax / 2;
  cyssFlangeABody->DefineSection(6, sCyssFlangeAStep2H, rmin, rmax);
  zlen = sCyssFlangeATotHei - sCyssFlangeAStep3H;
  cyssFlangeABody->DefineSection(7, zlen, rmin, rmax);
  rmin = sCyssFlangeAInnerD / 2;
  cyssFlangeABody->DefineSection(8, zlen, rmin, rmax);
  zlen = sCyssFlangeATotHei - sCyssFlangeAStep4H;
  cyssFlangeABody->DefineSection(9, zlen, rmin, rmax);
  rmax = sCyssFlangeAStep4Dmax / 2;
  cyssFlangeABody->DefineSection(10, zlen, rmin, rmax);
  cyssFlangeABody->DefineSection(11, sCyssFlangeATotHei, rmin, rmax);

  // The inner ring
  // We define half of it and put two copies to leave the notch space
  rmin = sCyssFlangeAStep3Dmin / 2;
  phi = 0.5 * (sCyssFlangeANotchW / rmin) * TMath::RadToDeg();

  TGeoPcon* cyssFlangeAInRing = new TGeoPcon("cflangearing", 180, 90 - phi, 4);

  rmin = sCyssFlangeAInnerD / 2;
  rmax = sCyssFlangeAInRingD / 2;
  cyssFlangeAInRing->DefineSection(0, sCyssFlangeAInRingUp, rmin, rmax);
  cyssFlangeAInRing->DefineSection(1, sCyssFlangeAInRingH, rmin, rmax);
  rmax = sCyssFlangeAStep3Dmin / 2;
  cyssFlangeAInRing->DefineSection(2, sCyssFlangeAInRingH, rmin, rmax);
  cyssFlangeAInRing->DefineSection(3, sCyssFlangeAStep2H, rmin, rmax);

  TGeoRotation* flangeARingRot = new TGeoRotation("cringrot", 90 + phi, 0, 0);
  flangeARingRot->RegisterYourself();

  TString cyssFlangeAComposite = Form("cyssflangeabody+cflangearing+cflangearing:cringrot");

  // The wings
  rmin = sCyssFlangeAStep1Dmax / 2;
  rmax = sCyssFlangeAWingD / 2;
  zlen = sCyssFlangeAStep1H / 2;
  phi = 0.5 * (sCyssFlangeAWingW / rmin) * TMath::RadToDeg();

  TGeoTubeSeg* cyssFlangeAWing = new TGeoTubeSeg("cflangeawing", rmin, rmax, zlen, 270 - phi, 270 + phi);

  TGeoTranslation* cwingTR1 = new TGeoTranslation("cwingtr1", 0, 0, zlen);
  cwingTR1->RegisterYourself();

  TGeoCombiTrans* cwingCT2 = new TGeoCombiTrans("cwingct2", 0, 0, zlen, new TGeoRotation("", 90 - phi, 0, 0));
  cwingCT2->RegisterYourself();

  TGeoCombiTrans* cwingCT3 = new TGeoCombiTrans("cwingct3", 0, 0, zlen, new TGeoRotation("", -90 + phi, 0, 0));
  cwingCT3->RegisterYourself();

  cyssFlangeAComposite += "+cflangeawing:cwingtr1+cflangeawing:cwingct2+cflangeawing:cwingct3";

  // The (many) holes
  zlen = cyssFlangeAWing->GetDz();

  // The 8 round holes (4 on each side)
  rmax = sCyssFlangeAHole1D / 2;
  TGeoTube* hole1 = new TGeoTube("hole1", 0, rmax, 2 * zlen);

  for (int i = 0; i < sCyssFlangeAHole1Num / 2; i++) {
    double phi = sCyssFlangeAHole1Phi0 + i * sCyssFlangeAHole1PhiStep;
    xpos = 0.5 * sCyssFlangeAHolesDpos * std::sin(phi * TMath::DegToRad());
    ypos = 0.5 * sCyssFlangeAHolesDpos * std::cos(phi * TMath::DegToRad());
    TGeoTranslation* hole1Tr1 = new TGeoTranslation(Form("hole1Tr1%d", i), xpos, -ypos, zlen);
    hole1Tr1->RegisterYourself();
    TGeoTranslation* hole1Tr2 = new TGeoTranslation(Form("hole1Tr2%d", i), -xpos, -ypos, zlen);
    hole1Tr2->RegisterYourself();
    cyssFlangeAComposite += Form("-hole1:hole1Tr1%d-hole1:hole1Tr2%d", i, i);
  }

  // The 2 smaller round holes (1 on each side)
  rmax = sCyssFlangeAHole2D / 2;
  TGeoTube* hole2 = new TGeoTube("hole2", 0, rmax, 2 * zlen);

  xpos = 0.5 * sCyssFlangeAHolesDpos * std::sin(sCyssFlangeAHole2Phi * TMath::DegToRad());
  ypos = 0.5 * sCyssFlangeAHolesDpos * std::cos(sCyssFlangeAHole2Phi * TMath::DegToRad());
  TGeoTranslation* hole2Tr1 = new TGeoTranslation("hole2Tr1", xpos, -ypos, zlen);
  hole2Tr1->RegisterYourself();
  TGeoTranslation* hole2Tr2 = new TGeoTranslation("hole2Tr2", -xpos, -ypos, zlen);
  hole2Tr2->RegisterYourself();

  cyssFlangeAComposite += "-hole2:hole2Tr1-hole2:hole2Tr2";

  // The 2 bigger round holes (1 on each side)
  rmax = sCyssFlangeAHole3D / 2;
  TGeoTube* hole3 = new TGeoTube("hole3", 0, rmax, 2 * zlen);

  xpos = 0.5 * sCyssFlangeAHolesDpos * std::sin(sCyssFlangeAHole3Phi * TMath::DegToRad());
  ypos = 0.5 * sCyssFlangeAHolesDpos * std::cos(sCyssFlangeAHole3Phi * TMath::DegToRad());
  TGeoTranslation* hole3Tr1 = new TGeoTranslation("hole3Tr1", xpos, -ypos, zlen);
  hole3Tr1->RegisterYourself();
  TGeoTranslation* hole3Tr2 = new TGeoTranslation("hole3Tr2", -xpos, -ypos, zlen);
  hole3Tr2->RegisterYourself();

  cyssFlangeAComposite += "-hole3:hole3Tr1-hole3:hole3Tr2";

  // The holes in the wings
  rmax = sCyssFlangeAWingHoleD / 2;
  TGeoTube* wingHole = new TGeoTube("wingHole", 0, rmax, 2 * zlen);

  TGeoTranslation* wingHoleTr1 = new TGeoTranslation("wingHoleTr1", 0, -sCyssFlangeAWingHoleRpos, zlen);
  wingHoleTr1->RegisterYourself();

  TGeoTranslation* wingHoleTr2 = new TGeoTranslation("wingHoleTr2", sCyssFlangeAWingHoleRpos, -sCyssFlangeAWingHoleYpos, zlen);
  wingHoleTr2->RegisterYourself();

  TGeoTranslation* wingHoleTr3 = new TGeoTranslation("wingHoleTr3", -sCyssFlangeAWingHoleRpos, -sCyssFlangeAWingHoleYpos, zlen);
  wingHoleTr3->RegisterYourself();

  cyssFlangeAComposite += "-wingHole:wingHoleTr1-wingHole:wingHoleTr2-wingHole:wingHoleTr3";

  // Lastly the hollows (évidements): a nightmare deserving its own method
  TString cyssFlangeAHollows = createHollowsCYSSFlangeA(zlen);

  cyssFlangeAComposite += cyssFlangeAHollows.Data();

  // The final flange shape
  TGeoCompositeShape* cyssFlangeASh = new TGeoCompositeShape(cyssFlangeAComposite.Data());

  // We have all shapes: now create the real volumes
  TGeoMedium* medAlu = gGeoManager->GetMedium("IT3_ALUMINUM$");

  TGeoVolume* cyssFlangeAVol = new TGeoVolume("IBCYSSFlangeA", cyssFlangeASh, medAlu);
  cyssFlangeAVol->SetLineColor(kCyan);
  cyssFlangeAVol->SetFillColor(kCyan);

  // Finally return the flange volume
  return cyssFlangeAVol;
}

TGeoVolume* ITS3Services::createCYSSFlangeC()
{
  //
  // Creates the Flange on Side C for the Inner Barrel CYSS
  // (ALIITSUP0098)
  // Based on ITSSimulation/V3Services.cxx
  //

  // Radii of the rings
  const double sCyssFlangeCDmin1 = 4.4;
  const double sCyssFlangeCDmin2 = 5.7;
  const double sCyssFlangeCDmin3 = 7.3;

  const double sCyssFlangeCDmax1 = 5.88;
  const double sCyssFlangeCDmax2 = 7.48;
  const double sCyssFlangeCDmax3 = 9.40;

  const double sCyssFlangeCDWallIn = 8.9;
  const double sCyssFlangeCDWallOut = 9.56;

  const double sCyssFlangeCDExt = 10.;

  // Thicknesses and heights
  const double sCyssFlangeCTotH = 1.;
  const double sCyssFlangeCExtThick = 0.1;

  const double sCyssFlangeCHmax1 = 0.15;
  const double sCyssFlangeCHmax2 = 0.40;
  const double sCyssFlangeCHmax3 = 0.65;

  const double sCyssFlangeCHmin2 = 0.25;
  const double sCyssFlangeCHmin3 = 0.50;

  // Holes
  const double sHoles22Dia = 0.22;
  const double sHoles22Phi = 60; // Deg

  const double sHoles30Dia = 0.3;
  const double sHoles30Phi = 15; // Deg

  const double sHoles12Dia = 0.12;
  const double sHoles12Phi = 75; // Deg

  const double sHolesDdist[3] = {5., 6.4, 8.};

  const double sCyssFlangeCNotchH = 0.32;
  const double sCyssFlangeCNotchW = 0.30;

  // Local variables
  double rmin, rmax, zlen;
  double xpos, ypos;

  // The CYSS Flange on Side C is physically a single piece.
  // It is implemented as a CompositeShape of two Pcon's minus the holes

  // The flange body
  TGeoPcon* cyssFlangeCDisks = new TGeoPcon("cyssflangecdisks", 180, 180, 12);

  rmin = sCyssFlangeCDmin1 / 2;
  rmax = sCyssFlangeCDmax1 / 2;
  cyssFlangeCDisks->DefineSection(0, 0, rmin, rmax);
  cyssFlangeCDisks->DefineSection(1, sCyssFlangeCHmax1, rmin, rmax);
  rmin = sCyssFlangeCDmin2 / 2;
  cyssFlangeCDisks->DefineSection(2, sCyssFlangeCHmax1, rmin, rmax);
  cyssFlangeCDisks->DefineSection(3, sCyssFlangeCHmin2, rmin, rmax);
  rmax = sCyssFlangeCDmax2 / 2;
  cyssFlangeCDisks->DefineSection(4, sCyssFlangeCHmin2, rmin, rmax);
  cyssFlangeCDisks->DefineSection(5, sCyssFlangeCHmax2, rmin, rmax);
  rmin = sCyssFlangeCDmin3 / 2;
  cyssFlangeCDisks->DefineSection(6, sCyssFlangeCHmax2, rmin, rmax);
  cyssFlangeCDisks->DefineSection(7, sCyssFlangeCHmin3, rmin, rmax);
  rmax = sCyssFlangeCDWallOut / 2;
  cyssFlangeCDisks->DefineSection(8, sCyssFlangeCHmin3, rmin, rmax);
  cyssFlangeCDisks->DefineSection(9, sCyssFlangeCHmax3, rmin, rmax);
  rmin = sCyssFlangeCDWallIn / 2;
  cyssFlangeCDisks->DefineSection(10, sCyssFlangeCHmax3, rmin, rmax);
  cyssFlangeCDisks->DefineSection(11, sCyssFlangeCTotH, rmin, rmax);

  TGeoPcon* cyssFlangeCExt = new TGeoPcon("cflangecext", 180, 180, 4);

  rmin = sCyssFlangeCDmax3 / 2;
  rmax = sCyssFlangeCDExt / 2;
  cyssFlangeCExt->DefineSection(0, 0, rmin, rmax);
  cyssFlangeCExt->DefineSection(1, sCyssFlangeCExtThick, rmin, rmax);
  rmax = sCyssFlangeCDWallOut / 2;
  cyssFlangeCExt->DefineSection(2, sCyssFlangeCExtThick, rmin, rmax);
  cyssFlangeCExt->DefineSection(3, sCyssFlangeCHmin3, rmin, rmax);

  TString cyssFlangeCComposite = Form("cyssflangecdisks+cflangecext");

  // The flange holes
  rmax = sHoles22Dia / 2;
  zlen = sCyssFlangeCTotH / 2;
  TGeoTube* hole22 = new TGeoTube("hole22", 0, rmax, 1.1 * zlen);

  for (int j = 0; j < 3; j++) {
    ypos = sHolesDdist[j] / 2;
    TGeoTranslation* holeCTr = new TGeoTranslation(Form("holeCTr%d", j), 0, -ypos, zlen);
    holeCTr->RegisterYourself();
    cyssFlangeCComposite += Form("-hole22:holeCTr%d", j);

    xpos = std::sin(sHoles22Phi * TMath::DegToRad()) * sHolesDdist[j] / 2;
    ypos = std::cos(sHoles22Phi * TMath::DegToRad()) * sHolesDdist[j] / 2;
    TGeoTranslation* holeLTr = new TGeoTranslation(Form("holeLTr%d", j), xpos, -ypos, zlen);
    holeLTr->RegisterYourself();
    cyssFlangeCComposite += Form("-hole22:holeLTr%d", j);

    TGeoTranslation* holeRTr = new TGeoTranslation(Form("holeRTr%d", j), -xpos, -ypos, zlen);
    holeRTr->RegisterYourself();
    cyssFlangeCComposite += Form("-hole22:holeRTr%d", j);
  }

  rmax = sHoles30Dia / 2;
  TGeoTube* hole30 = new TGeoTube("hole30", 0, rmax, zlen);

  for (int k = 0; k < 3; k++) {
    double phi = (k + 1) * sHoles30Phi;
    for (int j = 0; j < 3; j++) {
      xpos = std::sin(phi * TMath::DegToRad()) * sHolesDdist[j] / 2;
      ypos = std::cos(phi * TMath::DegToRad()) * sHolesDdist[j] / 2;

      TGeoTranslation* holeLTr = new TGeoTranslation(Form("holeLTr%d%d", k, j), xpos, -ypos, zlen);
      holeLTr->RegisterYourself();
      cyssFlangeCComposite += Form("-hole30:holeLTr%d%d", k, j);

      TGeoTranslation* holeRTr = new TGeoTranslation(Form("holeRTr%d%d", k, j), -xpos, -ypos, zlen);
      holeRTr->RegisterYourself();
      cyssFlangeCComposite += Form("-hole30:holeRTr%d%d", k, j);
    }
  }

  rmax = sHoles12Dia / 2;
  TGeoTube* hole12 = new TGeoTube("hole12", 0, rmax, 1.1 * zlen);

  for (int j = 0; j < 3; j++) {
    xpos = std::sin(sHoles12Phi * TMath::DegToRad()) * sHolesDdist[j] / 2;
    ypos = std::cos(sHoles12Phi * TMath::DegToRad()) * sHolesDdist[j] / 2;
    TGeoTranslation* holeLTr = new TGeoTranslation(Form("holeLTrM%d", j), xpos, -ypos, zlen);
    holeLTr->RegisterYourself();
    cyssFlangeCComposite += Form("-hole12:holeLTrM%d", j);

    TGeoTranslation* holeRTr = new TGeoTranslation(Form("holeRTrM%d", j), -xpos, -ypos, zlen);
    holeRTr->RegisterYourself();
    cyssFlangeCComposite += Form("-hole12:holeRTrM%d", j);
  }

  TGeoBBox* notch = new TGeoBBox("notch", sCyssFlangeCNotchW / 2, (sCyssFlangeCDWallOut - sCyssFlangeCDWallIn), sCyssFlangeCNotchH);

  ypos = (sCyssFlangeCDWallIn + sCyssFlangeCDWallOut) / 4;
  TGeoTranslation* notchTr = new TGeoTranslation("notchTr", 0, -ypos, sCyssFlangeCTotH);
  notchTr->RegisterYourself();

  cyssFlangeCComposite += "-notch:notchTr";

  // The final flange shape
  TGeoCompositeShape* cyssFlangeCSh = new TGeoCompositeShape(cyssFlangeCComposite.Data());

  // We have all shapes: now create the real volumes
  TGeoMedium* medAlu = gGeoManager->GetMedium("IT3_ALUMINUM$");

  TGeoVolume* cyssFlangeCVol = new TGeoVolume("IBCYSSFlangeC", cyssFlangeCSh, medAlu);
  cyssFlangeCVol->SetLineColor(kCyan);
  cyssFlangeCVol->SetFillColor(kCyan);

  // Finally return the flange volume
  return cyssFlangeCVol;
}

TString ITS3Services::createHollowsCYSSFlangeA(double zlen)
{
  //
  // Creates the very complicate hollow holes in the Flange
  // on Side A for the Inner Barrel CYSS
  // (ALIITSUP0189)
  // Based on ITSSimulation/V3Services.cxx
  //

  const double sCyssFlangeAHolesDpos = 27.4;

  const double sCyssFlangeAHole1Phi0 = 10;    // Deg
  const double sCyssFlangeAHole1PhiStep = 20; // Deg

  const double sCyssFlangeAHole2Phi = 20; // Deg

  const double sCyssFlangeAHollowD = 0.7;
  const double sCyssFlangeAHollowPhi0 = 13; // Deg
  const double sCyssFlangeAHollowPhi1 = 8;  // Deg

  // Local variables
  double rmin, rmax, phi, dphi;
  double xpos, ypos;

  TString cyssFlangeAHollows;

  //
  rmax = sCyssFlangeAHollowD / 2;
  TGeoTubeSeg* roundHalf = new TGeoTubeSeg("roundhalf", 0, rmax, 2 * zlen, 0, 180);

  double rHoles = sCyssFlangeAHolesDpos / 2;

  xpos = rHoles * std::cos(sCyssFlangeAHollowPhi0 * TMath::DegToRad());
  ypos = rHoles * std::sin(sCyssFlangeAHollowPhi0 * TMath::DegToRad());
  TGeoCombiTrans* roundTr1 = new TGeoCombiTrans("roundtr1", xpos, -ypos, zlen, new TGeoRotation("", -sCyssFlangeAHollowPhi0, 0, 0));
  roundTr1->RegisterYourself();
  TGeoCombiTrans* roundTr2 = new TGeoCombiTrans("roundtr2", -xpos, -ypos, zlen, new TGeoRotation("", sCyssFlangeAHollowPhi0, 0, 0));
  roundTr2->RegisterYourself();

  cyssFlangeAHollows += "-roundhalf:roundtr1-roundhalf:roundtr2";

  TGeoTranslation* noRot = new TGeoTranslation("norot", 0, 0, zlen);
  noRot->RegisterYourself();
  TGeoCombiTrans* yRot180 = new TGeoCombiTrans("yrot180", 0, 0, zlen, new TGeoRotation("", 0, 180, 180));
  yRot180->RegisterYourself();

  rmin = sCyssFlangeAHolesDpos / 2 - sCyssFlangeAHollowD / 2;
  rmax = sCyssFlangeAHolesDpos / 2 + sCyssFlangeAHollowD / 2;

  for (int j = 1; j < 4; j++) {
    phi = 90 - (sCyssFlangeAHole1Phi0 + j * sCyssFlangeAHole1PhiStep + 0.5 * sCyssFlangeAHollowPhi1);
    xpos = rHoles * std::cos(phi * TMath::DegToRad());
    ypos = rHoles * std::sin(phi * TMath::DegToRad());
    TGeoCombiTrans* roundTr3 = new TGeoCombiTrans(Form("roundtr%d", j + 2), xpos, -ypos, zlen, new TGeoRotation("", 180 - phi, 0, 0));
    roundTr3->RegisterYourself();
    TGeoCombiTrans* roundTr4 = new TGeoCombiTrans(Form("roundtr%d", j + 5), -xpos, -ypos, zlen, new TGeoRotation("", phi - 180, 0, 0));
    roundTr4->RegisterYourself();

    cyssFlangeAHollows += Form("-roundhalf:roundtr%d-roundhalf:roundtr%d", j + 2, j + 5);

    phi = 360 - phi - 0.05;
    if (j == 3) {
      dphi = 360 - sCyssFlangeAHollowPhi0 + 0.05;
    } else {
      dphi = phi + (sCyssFlangeAHole1PhiStep - sCyssFlangeAHollowPhi1) + 0.1;
    }

    TGeoTubeSeg* hollow1 = new TGeoTubeSeg(Form("hollow%d", j), rmin, rmax, 2 * zlen, phi, dphi);

    cyssFlangeAHollows += Form("-hollow%d:norot-hollow%d:yrot180", j, j);

    phi = 90 - (sCyssFlangeAHole1Phi0 + j * sCyssFlangeAHole1PhiStep - 0.5 * sCyssFlangeAHollowPhi1);
    xpos = rHoles * std::cos(phi * TMath::DegToRad());
    ypos = rHoles * std::sin(phi * TMath::DegToRad());
    TGeoCombiTrans* roundTr5 = new TGeoCombiTrans(Form("roundtr%d", j + 8), xpos, -ypos, zlen, new TGeoRotation("", -phi, 0, 0));
    roundTr5->RegisterYourself();
    TGeoCombiTrans* roundTr6 = new TGeoCombiTrans(Form("roundtr%d", j + 11), -xpos, -ypos, zlen, new TGeoRotation("", phi, 0, 0));
    roundTr6->RegisterYourself();

    cyssFlangeAHollows += Form("-roundhalf:roundtr%d-roundhalf:roundtr%d", j + 8, j + 11);
  }

  phi = 90 - (sCyssFlangeAHole2Phi + 0.5 * sCyssFlangeAHollowPhi1);
  xpos = rHoles * std::cos(phi * TMath::DegToRad());
  ypos = rHoles * std::sin(phi * TMath::DegToRad());
  TGeoCombiTrans* roundTr15 = new TGeoCombiTrans("roundtr15", xpos, -ypos, zlen, new TGeoRotation("", 180 - phi, 0, 0));
  roundTr15->RegisterYourself();
  TGeoCombiTrans* roundTr16 = new TGeoCombiTrans("roundtr16", -xpos, -ypos, zlen, new TGeoRotation("", phi - 180, 0, 0));
  roundTr16->RegisterYourself();

  cyssFlangeAHollows += "-roundhalf:roundtr15-roundhalf:roundtr16";

  phi = 360 - phi - 0.5;
  dphi = phi + (sCyssFlangeAHole1Phi0 + sCyssFlangeAHole1PhiStep - sCyssFlangeAHole2Phi - sCyssFlangeAHollowPhi1) + 0.5;
  TGeoTubeSeg* hollow4 = new TGeoTubeSeg("hollow4", rmin, rmax, 2 * zlen, phi, dphi);

  cyssFlangeAHollows += "-hollow4:norot-hollow4:yrot180";

  //
  phi = 90 - (sCyssFlangeAHole2Phi - 0.5 * sCyssFlangeAHollowPhi1);
  xpos = rHoles * std::cos(phi * TMath::DegToRad());
  ypos = rHoles * std::sin(phi * TMath::DegToRad());
  TGeoCombiTrans* roundTr17 = new TGeoCombiTrans("roundtr17", xpos, -ypos, zlen, new TGeoRotation("", -phi, 0, 0));
  roundTr17->RegisterYourself();
  TGeoCombiTrans* roundTr18 = new TGeoCombiTrans("roundtr18", -xpos, -ypos, zlen, new TGeoRotation("", phi, 0, 0));
  roundTr18->RegisterYourself();

  cyssFlangeAHollows += "-roundhalf:roundtr17-roundhalf:roundtr18";

  phi = 90 - (sCyssFlangeAHole1Phi0 + 0.5 * sCyssFlangeAHollowPhi1);
  xpos = rHoles * std::cos(phi * TMath::DegToRad());
  ypos = rHoles * std::sin(phi * TMath::DegToRad());
  TGeoCombiTrans* roundTr19 = new TGeoCombiTrans("roundtr19", xpos, -ypos, zlen, new TGeoRotation("", 180 - phi, 0, 0));
  roundTr19->RegisterYourself();
  TGeoCombiTrans* roundTr20 = new TGeoCombiTrans("roundtr20", -xpos, -ypos, zlen, new TGeoRotation("", phi - 180, 0, 0));
  roundTr20->RegisterYourself();

  cyssFlangeAHollows += "-roundhalf:roundtr19-roundhalf:roundtr20";

  TGeoCombiTrans* zRotPhi = new TGeoCombiTrans("zrotphi", 0, 0, zlen, new TGeoRotation("", -sCyssFlangeAHole1Phi0, 0, 0));
  zRotPhi->RegisterYourself();
  TGeoCombiTrans* yzRot180Phi = new TGeoCombiTrans("yzrot180phi", 0, 0, zlen, new TGeoRotation("", 0, 180, 180 - sCyssFlangeAHole1Phi0));
  yzRot180Phi->RegisterYourself();

  cyssFlangeAHollows += "-hollow4:zrotphi-hollow4:yzrot180phi";

  // Finally we return the string
  return cyssFlangeAHollows;
}

void ITS3Services::insidePoint(double x0, double y0, double x1, double y1, double x2,
                               double y2, double c, double& x, double& y) const
{
  /// Given two intersecting lines defined by the points (x0,y0), (x1,y1) and
  /// (x1,y1), (x2,y2) {intersecting at (x1,y1)} the point (x,y) a distance
  /// c away is returned such that two lines a distance c away from the
  /// lines defined above intersect at (x,y).
  /// Based on ITSSimulation/V11Geometry.cxx

  double dx01, dx12, dy01, dy12, r01, r12, m;

  dx01 = x0 - x1;
  dx12 = x1 - x2;
  dy01 = y0 - y1;
  dy12 = y1 - y2;
  r01 = std::sqrt(dy01 * dy01 + dx01 * dx01);
  r12 = std::sqrt(dy12 * dy12 + dx12 * dx12);
  m = dx12 * dy01 - dy12 * dx01;
  if (m * m < std::numeric_limits<double>::epsilon()) {
    if (dy01 == 0.0) {
      x = x1 + c;
      y = y1;
      return;
    } else if (dx01 == 0.0) {
      x = x1;
      y = y1 + c;
      return;
    } else {
      x = x1 - 0.5 * c * r01 / dy01;
      y = y1 + 0.5 * c * r01 / dx01;
    }
    return;
  }
  x = x1 + c * (dx12 * r01 - dx01 * r12) / m;
  y = y1 + c * (dy12 * r01 - dy01 * r12) / m;
}

double ITS3Services::yFrom2Points(double x0, double y0, double x1, double y1, double x) const
{
  /// Given the two points (x0,y0) and (x1,y1) and the location x, returns
  /// the value y corresponding to that point x on the line defined by the
  /// two points. Returns the value y corresponding to the point x on the line defined by
  /// the two points (x0,y0) and (x1,y1).

  if (x0 == x1 && y0 == y1) {
    return 0.0;
  } else if (x0 == x1) {
    return 0.5 * (y0 + y1);
  }

  double m = (y0 - y1) / (x0 - x1);

  return m * (x - x0) + y0;
}