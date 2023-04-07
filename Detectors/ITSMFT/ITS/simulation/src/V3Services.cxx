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

/// \file V3Services.cxx
/// \brief Implementation of the V3Services class
/// \author Mario Sitta <sitta@to.infn.it>
/// \author Parinya Namwongsa <parinya.namwongsa@cern.ch>

#include "ITSSimulation/V3Services.h"
#include "ITSSimulation/V11Geometry.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTSimulation/AlpideChip.h"

#include <fairlogger/Logger.h> // for LOG

#include <TGeoArb8.h>    // for TGeoArb8
#include <TGeoBBox.h>    // for TGeoBBox
#include <TGeoCone.h>    // for TGeoConeSeg, TGeoCone
#include <TGeoPcon.h>    // for TGeoPcon
#include <TGeoManager.h> // for TGeoManager, gGeoManager
#include <TGeoMatrix.h>  // for TGeoCombiTrans, TGeoRotation, etc
//#include <TGeoTrd1.h>           // for TGeoTrd1
#include <TGeoTube.h>           // for TGeoTube, TGeoTubeSeg
#include <TGeoVolume.h>         // for TGeoVolume, TGeoVolumeAssembly
#include <TGeoXtru.h>           // for TGeoXtru
#include <TGeoCompositeShape.h> // for TGeoCompositeShape
#include "TMathBase.h"          // for Abs
#include <TMath.h>              // for Sin, RadToDeg, DegToRad, Cos, Tan, etc

#include <cstdio> // for snprintf

class TGeoMedium;

using namespace TMath;
using namespace o2::its;

// Parameters
const Double_t V3Services::sIBWheelACZdist = 306.0 * sMm;
const Double_t V3Services::sIBCYSSFlangeCZPos = 171.5 * sMm; // Computed from different drawings
const Double_t V3Services::sOBWheelThickness = 2.0 * sMm;
const Double_t V3Services::sMBWheelsZpos = 457.0 * sMm;
const Double_t V3Services::sOBWheelsZpos = 770.0 * sMm;
const Double_t V3Services::sOBConesZpos = 798.0 * sMm;

ClassImp(V3Services);

#define SQ(A) (A) * (A)

V3Services::V3Services()
  : V11Geometry()
{
}

V3Services::V3Services(const char* name)
  : V11Geometry(0, name)
{
}

V3Services::~V3Services() = default;

TGeoVolume* V3Services::createIBEndWheelsSideA(const TGeoManager* mgr)
{
  //
  // Creates the Inner Barrel End Wheels on Side A
  //
  // Input:
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //         a TGeoVolume(Assembly) with all the wheels
  //
  // Created:      19 Jun 2019  Mario Sitta
  //               (partially based on P.Namwongsa implementation in AliRoot)
  //

  TGeoVolume* endWheelsVol = new TGeoVolumeAssembly("EndWheelsSideA");
  endWheelsVol->SetVisibility(kTRUE);

  for (Int_t jLay = 0; jLay < sNumberInnerLayers; jLay++) {
    ibEndWheelSideA(jLay, endWheelsVol, mgr);
  }

  // Return the wheels
  return endWheelsVol;
}

TGeoVolume* V3Services::createIBEndWheelsSideC(const TGeoManager* mgr)
{
  //
  // Creates the Inner Barrel End Wheels on Side C
  //
  // Input:
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //         a TGeoVolume(Assembly) with all the wheels
  //
  // Created:      15 May 2019  Mario Sitta
  //               (partially based on P.Namwongsa implementation in AliRoot)
  //

  TGeoVolume* endWheelsVol = new TGeoVolumeAssembly("EndWheelsSideC");
  endWheelsVol->SetVisibility(kTRUE);

  for (Int_t jLay = 0; jLay < sNumberInnerLayers; jLay++) {
    ibEndWheelSideC(jLay, endWheelsVol, mgr);
  }

  // Return the wheels
  return endWheelsVol;
}

TGeoVolume* V3Services::createCYSSAssembly(const TGeoManager* mgr)
{
  //
  // Creates the CYSS Assembly (i.e. the supporting cylinder and cone)
  //
  // Input:
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //         a TGeoVolume(Assembly) with all the elements
  //
  // Created:      21 Oct 2019  Mario Sitta
  // Updated:      02 Dec 2019  Mario Sitta   Full cylinder implemented
  //

  static const Double_t sCyssFlangeAZpos = 9.0 * sMm;
  static const Double_t sCyssFlangeCZpos = 1.0 * sMm;

  Double_t zlen, zpos;

  TGeoVolume* cyssVol = new TGeoVolumeAssembly("IBCYSSAssembly");
  cyssVol->SetVisibility(kTRUE);

  TGeoVolume* cyssCylinder = ibCyssCylinder(mgr);
  zlen = (static_cast<TGeoTubeSeg*>(cyssCylinder->GetShape()))->GetDz();
  zpos = sIBCYSSFlangeCZPos - sCyssFlangeCZpos - zlen;
  cyssVol->AddNode(cyssCylinder, 1, new TGeoTranslation(0, 0, -zpos));
  cyssVol->AddNode(cyssCylinder, 2, new TGeoCombiTrans(0, 0, -zpos, new TGeoRotation("", 180, 0, 0)));

  TGeoVolume* cyssCone = ibCyssCone(mgr);
  zpos = -zpos + zlen - (static_cast<TGeoPcon*>(cyssCone->GetShape()))->GetZ(2);
  cyssVol->AddNode(cyssCone, 1, new TGeoTranslation(0, 0, zpos));
  cyssVol->AddNode(cyssCone, 2, new TGeoCombiTrans(0, 0, zpos, new TGeoRotation("", 180, 0, 0)));

  TGeoVolume* cyssFlangeA = ibCyssFlangeSideA(mgr);
  Int_t nZPlanes = (static_cast<TGeoPcon*>(cyssCone->GetShape()))->GetNz();
  zpos = zpos + (static_cast<TGeoPcon*>(cyssCone->GetShape()))->GetZ(nZPlanes - 1) + sCyssFlangeAZpos;
  cyssVol->AddNode(cyssFlangeA, 1, new TGeoCombiTrans(0, 0, zpos, new TGeoRotation("", 180, 180, 0)));
  cyssVol->AddNode(cyssFlangeA, 2, new TGeoCombiTrans(0, 0, zpos, new TGeoRotation("", 0, 180, 0)));

  TGeoVolume* cyssFlangeC = ibCyssFlangeSideC(mgr);
  zpos = sIBCYSSFlangeCZPos;
  cyssVol->AddNode(cyssFlangeC, 1, new TGeoTranslation(0, 0, -zpos));
  cyssVol->AddNode(cyssFlangeC, 2, new TGeoCombiTrans(0, 0, -zpos, new TGeoRotation("", 180, 0, 0)));

  // Return the whole assembly
  return cyssVol;
}

void V3Services::createMBEndWheelsSideA(TGeoVolume* mother, const TGeoManager* mgr)
{
  //
  // Creates the Middle Barrel End Wheels on Side A
  //
  // Input:
  //         mother : the volume hosting the wheels
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      24 Sep 2019  Mario Sitta
  //

  for (Int_t jLay = 0; jLay < sNumberMiddlLayers; jLay++) {
    obEndWheelSideA(jLay, mother, mgr);
  }
}

void V3Services::createMBEndWheelsSideC(TGeoVolume* mother, const TGeoManager* mgr)
{
  //
  // Creates the Middle Barrel End Wheels on Side C
  //
  // Input:
  //         mother : the volume hosting the wheels
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      26 Sep 2019  Mario Sitta
  //

  for (Int_t jLay = 0; jLay < sNumberMiddlLayers; jLay++) {
    mbEndWheelSideC(jLay, mother, mgr);
  }
}

void V3Services::createOBEndWheelsSideA(TGeoVolume* mother, const TGeoManager* mgr)
{
  //
  // Creates the Outer Barrel End Wheels on Side A
  //
  // Input:
  //         mother : the volume hosting the wheels
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      27 Sep 2019  Mario Sitta
  //

  for (Int_t jLay = 0; jLay < sNumberOuterLayers; jLay++) {
    obEndWheelSideA(jLay + sNumberMiddlLayers, mother, mgr);
  }
}

void V3Services::createOBEndWheelsSideC(TGeoVolume* mother, const TGeoManager* mgr)
{
  //
  // Creates the Outer Barrel End Wheels on Side C
  //
  // Input:
  //         mother : the volume hosting the wheels
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      27 Sep 2019  Mario Sitta
  //

  for (Int_t jLay = 0; jLay < sNumberOuterLayers; jLay++) {
    obEndWheelSideC(jLay, mother, mgr);
  }
}

void V3Services::createOBConeSideA(TGeoVolume* mother, const TGeoManager* mgr)
{
  //
  // Creates the Outer Barrel Cone on Side A
  //
  // Input:
  //         mother : the volume hosting the cones
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      03 Feb 2020  Mario Sitta
  //

  obConeSideA(mother, mgr);
  obConeTraysSideA(mother, mgr);
}

void V3Services::createOBConeSideC(TGeoVolume* mother, const TGeoManager* mgr)
{
  //
  // Creates the Outer Barrel Cone on Side C
  //
  // Input:
  //         mother : the volume hosting the cones
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      26 Jan 2020  Mario Sitta
  //

  obConeSideC(mother, mgr);
}

void V3Services::createOBCYSSCylinder(TGeoVolume* mother, const TGeoManager* mgr)
{
  //
  // Creates the Outer Barrel CYSS Cylinder
  // Volume and method names correspond to element names in blueprints
  //
  // Input:
  //         mother : the volume hosting the cones
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      02 Mar 2020  Mario Sitta
  // Last change:  14 Apr 2022  Matteo Concas

  obCYSS11(mother, mgr);
}

void V3Services::createIBGammaConvWire(TGeoVolume* mother, const TGeoManager* mgr)
{
  //
  // Creates the Inner Barrel Gamma Conversion Wire
  // Volume and method names correspond to element names in blueprints
  //
  // Input:
  //         mother : the volume hosting the cones
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      28 Sep 2022  Mario Sitta
  //

  ibConvWire(mother, mgr);
}

void V3Services::createOBGammaConvWire(TGeoVolume* mother, const TGeoManager* mgr)
{
  //
  // Creates the Outer Barrel Gamma Conversion Wire
  // Volume and method names correspond to element names in blueprints
  //
  // Input:
  //         mother : the volume hosting the cones
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      08 Sep 2022  Mario Sitta
  //

  obConvWire(mother, mgr);
}

void V3Services::ibEndWheelSideA(const Int_t iLay, TGeoVolume* endWheel, const TGeoManager* mgr)
{
  //
  // Creates the single End Wheel on Side A
  // for a given layer of the Inner Barrel
  // (Layer 0: ALIITSSUP0183+ALIITSUP0127)
  // (Layer 1: ALIITSSUP0173+ALIITSUP0124)
  // (Layer 2: ALIITSSUP0139+ALIITSUP0125)
  //
  // Input:
  //         iLay : the layer number
  //         endWheel : the whole end wheel volume
  //                    where to place the current created wheel
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      19 Jun 2019  Mario Sitta
  //               (partially based on P.Namwongsa implementation in AliRoot)
  //

  // The Basis Cone A Side and the Reinforcement C Side are physically two
  // different pieces put together. For sake of simplicity here they are
  // made out of the same TGeoPcon volume. Moreover they are two halves,
  // so here they are made as a single cone.
  static const Double_t sConeATotalLength[3] = {191.0 * sMm, 184.0 * sMm, 177 * sMm};
  static const Double_t sConeAIntSectZlen1[3] = {40.35 * sMm, 39.0 * sMm, 36.0 * sMm};
  static const Double_t sConeAIntSectZlen2[3] = {47.0 * sMm, 44.0 * sMm, 41.0 * sMm};
  static const Double_t sConeAIntSectDmin[3] = {55.8 * sMm, 71.8 * sMm, 87.8 * sMm};
  static const Double_t sConeAIntSectDmax[3] = {57.0 * sMm, 73.0 * sMm, 89.0 * sMm};
  static const Double_t sConeAExtSectZlen1[3] = {60.0 * sMm, 47.0 * sMm, 44.0 * sMm};
  static const Double_t sConeAExtSectZlen2[3] = {66.0 * sMm, 52.0 * sMm, 50.0 * sMm};
  static const Double_t sConeAExtSectDmin[3] = {114.0 * sMm, 174.0 * sMm, 234.0 * sMm};
  static const Double_t sConeAExtSectDmax[3] = {116.0 * sMm, 176.0 * sMm, 236.0 * sMm};
  static const Double_t sConeASectThicker = 0.8 * sMm;
  static const Double_t sConeAOpeningAngle[3] = {20.0, 30.0, 40.2}; // Deg

  static const Int_t sConeAWallNHoles[3] = {6, 8, 10};
  static const Double_t sConeAWallHoleD = 4.5 * sMm;
  static const Double_t sConeAWallHoleZpos = 4.0 * sMm;

  static const Double_t sConeACentralHole1D = 3.0 * sMm;
  static const Double_t sConeACentralHole2D = 3.4 * sMm;
  static const Double_t sConeACentralHole3D = 3.0 * sMm;
  static const Double_t sConeACentralHole1Z = 20.0 * sMm;
  static const Double_t sConeACentralHole2Z = 30.0 * sMm;
  static const Double_t sConeACentralHole3Z[3] = {177.0 * sMm, 170.0 * sMm, 163.0 * sMm};

  // The Cone Reinforcement
  static const Double_t sConeARenfDmin[3] = {54.3 * sMm, 69.85 * sMm, 85.0 * sMm};
  static const Double_t sConeARenfZlen = 2.5 * sMm;
  static const Double_t sConeARenfZpos = 14.5 * sMm;

  // The Middle Ring
  static const Double_t sConeAMidRingDmin[3] = {56.0 * sMm, 116.0 * sMm, 176.0 * sMm};
  static const Double_t sConeAMidRingDmax[3] = {58.0 * sMm, 118.0 * sMm, 178.0 * sMm};
  static const Double_t sConeAMidRingZlen = 42.0 * sMm;

  static const Double_t sConeAMidRingZpos[3] = {5.0 * sMm, 0.0 * sMm, 0.0 * sMm};

  // The Ribs
  static const Int_t sConeANRibs[3] = {6, 8, 10};
  static const Double_t sConeARibsZpos = 17.0 * sMm;

  // The End Wheel Steps
  static const Double_t sConeAStepXdispl[3] = {4.0 * sMm, 6.5 * sMm, 8.5 * sMm};
  static const Double_t sConeAStepYdispl[3] = {24.4 * sMm, 32.1 * sMm, 39.6 * sMm};
  static const Double_t sConeAStepR[3] = {27.8 * sMm, 35.8 * sMm, 43.8 * sMm};

  static const Double_t sConeAStepZlen = 14.0 * sMm;

  static const Double_t sConeAStepHoleXpos = 3.0 * sMm;
  static const Double_t sConeAStepHoleZpos = 4.0 * sMm;
  static const Double_t sConeAStepHoleZdist = 4.0 * sMm;

  static const Double_t sConeAStepHolePhi[3] = {30.0, 22.5, 18.0};   // Deg
  static const Double_t sConeAStepHolePhi0[3] = {0.7, -16.2, -10.5}; // Deg

  // Local variables
  Double_t xlen, ylen, zlen;
  Double_t rmin, rmax, thick, phimin, dphi;
  Double_t xpos, ypos, zpos, zref;

  // Create the whole cone (Basic + Reinforcement) as a CompositeShape
  // (a single Pcon minus the holes)
  TGeoPcon* coneabasis = new TGeoPcon(Form("coneabasis%d", iLay), 0, 360, 15);

  rmin = sConeAIntSectDmin[iLay] / 2;
  rmax = sConeAIntSectDmax[iLay] / 2;
  coneabasis->DefineSection(0, 0., rmin, rmax);
  zpos = sConeARenfZpos;
  coneabasis->DefineSection(1, zpos, rmin, rmax);
  rmin = sConeARenfDmin[iLay] / 2;
  coneabasis->DefineSection(2, zpos, rmin, rmax);
  zpos += sConeARenfZlen;
  coneabasis->DefineSection(3, zpos, rmin, rmax);
  rmin = coneabasis->GetRmin(0);
  coneabasis->DefineSection(4, zpos, rmin, rmax);
  coneabasis->DefineSection(5, sConeAIntSectZlen1[iLay], rmin, rmax);
  rmax += sConeASectThicker;
  coneabasis->DefineSection(6, sConeAIntSectZlen1[iLay], rmin, rmax);
  coneabasis->DefineSection(7, sConeAIntSectZlen2[iLay], rmin, rmax);
  rmin = coneabasis->GetRmax(1);
  coneabasis->DefineSection(8, sConeAIntSectZlen2[iLay], rmin, rmax);
  rmin = sConeAExtSectDmin[iLay] / 2 - sConeASectThicker;
  rmax = sConeAExtSectDmin[iLay] / 2;
  zlen = sConeAIntSectZlen2[iLay] + (rmin - coneabasis->GetRmin(4)) / TMath::Tan(sConeAOpeningAngle[iLay] * TMath::DegToRad());
  coneabasis->DefineSection(9, zlen, rmin, rmax);
  zlen = sConeATotalLength[iLay] - sConeAExtSectZlen2[iLay];
  coneabasis->DefineSection(10, zlen, rmin, rmax);
  rmax = sConeAExtSectDmax[iLay] / 2;
  coneabasis->DefineSection(11, zlen, rmin, rmax);
  zlen = sConeATotalLength[iLay] - sConeAExtSectZlen1[iLay];
  coneabasis->DefineSection(12, zlen, rmin, rmax);
  rmin = sConeAExtSectDmin[iLay] / 2;
  coneabasis->DefineSection(13, zlen, rmin, rmax);
  coneabasis->DefineSection(14, sConeATotalLength[iLay], rmin, rmax);

  TString coneAComposite = Form("coneabasis%d", iLay);

  // The holes in the vertical wall
  thick = coneabasis->GetRmax(0) - coneabasis->GetRmin(0);
  TGeoTube* coneawallhole = new TGeoTube(Form("coneawallhole%d", iLay), 0, sConeAWallHoleD / 2, 4 * thick);

  rmin = sConeAIntSectDmax[iLay] / 2 - thick / 2;
  zpos = sConeAWallHoleZpos;
  dphi = 180. / sConeAWallNHoles[iLay];
  phimin = dphi / 2.;
  for (Int_t ihole = 0; ihole < 2 * sConeAWallNHoles[iLay]; ihole++) {
    Double_t phi = phimin + ihole * dphi;
    xpos = rmin * TMath::Sin(phi * TMath::DegToRad());
    ypos = rmin * TMath::Cos(phi * TMath::DegToRad());
    TGeoCombiTrans* coneawhmat = new TGeoCombiTrans(Form("coneawhmat%dl%d", ihole, iLay), xpos, ypos, zpos, new TGeoRotation("", -phi, 90, 0));
    coneawhmat->RegisterYourself();
    coneAComposite += Form("-coneawallhole%d:coneawhmat%dl%d", iLay, ihole, iLay);
  }

  // The central holes
  TGeoTube* coneacenthole1 = new TGeoTube(Form("coneacenthole1l%d", iLay), 0, sConeACentralHole1D / 2, 4 * thick);

  TGeoCombiTrans* coneach1mat1 = new TGeoCombiTrans(Form("coneach1mat1l%d", iLay), 0, rmin, sConeACentralHole1Z, new TGeoRotation("", 0, 90, 0));
  coneach1mat1->RegisterYourself();
  TGeoCombiTrans* coneach1mat2 = new TGeoCombiTrans(Form("coneach1mat2l%d", iLay), 0, -rmin, sConeACentralHole1Z, new TGeoRotation("", 0, 90, 0));
  coneach1mat2->RegisterYourself();

  coneAComposite += Form("-coneacenthole1l%d:coneach1mat1l%d-coneacenthole1l%d:coneach1mat2l%d", iLay, iLay, iLay, iLay);

  TGeoTube* coneacenthole2 = new TGeoTube(Form("coneacenthole2l%d", iLay), 0, sConeACentralHole2D / 2, 4 * thick);

  TGeoCombiTrans* coneach2mat1 = new TGeoCombiTrans(Form("coneach2mat1l%d", iLay), 0, rmin, sConeACentralHole2Z, new TGeoRotation("", 0, 90, 0));
  coneach2mat1->RegisterYourself();
  TGeoCombiTrans* coneach2mat2 = new TGeoCombiTrans(Form("coneach2mat2l%d", iLay), 0, -rmin, sConeACentralHole2Z, new TGeoRotation("", 0, 90, 0));
  coneach2mat2->RegisterYourself();

  coneAComposite += Form("-coneacenthole2l%d:coneach2mat1l%d-coneacenthole2l%d:coneach2mat2l%d", iLay, iLay, iLay, iLay);

  TGeoTube* coneacenthole3 = new TGeoTube(Form("coneacenthole3l%d", iLay), 0, sConeACentralHole3D / 2, 4 * thick);

  rmin = sConeAExtSectDmax[iLay] / 2 - thick / 2;
  TGeoCombiTrans* coneach3mat1 = new TGeoCombiTrans(Form("coneach3mat1l%d", iLay), 0, rmin, sConeACentralHole3Z[iLay], new TGeoRotation("", 0, 90, 0));
  coneach3mat1->RegisterYourself();
  TGeoCombiTrans* coneach3mat2 = new TGeoCombiTrans(Form("coneach3mat2l%d", iLay), 0, -rmin, sConeACentralHole3Z[iLay], new TGeoRotation("", 0, 90, 0));
  coneach3mat2->RegisterYourself();

  coneAComposite += Form("-coneacenthole3l%d:coneach3mat1l%d-coneacenthole3l%d:coneach3mat2l%d", iLay, iLay, iLay, iLay);

  TGeoCompositeShape* coneABasisSh = new TGeoCompositeShape(coneAComposite.Data());

  // The Middle Ring (a Tube)
  rmin = sConeAMidRingDmin[iLay] / 2;
  rmax = sConeAMidRingDmax[iLay] / 2;
  zlen = sConeAMidRingZlen / 2;
  TGeoTube* midRingSh = new TGeoTube(Form("midRingSh%d", iLay), rmin, rmax, zlen);

  // A Rib (a TGeoXtru)
  TGeoXtru* coneARibSh = ibEndWheelARibShape(iLay);

  // Now the Step as a Composite Shape (subtraction of a Pcon from a BBox)
  // (cutting volume should be slightly larger than desired region)
  rmin = sConeAStepR[iLay];

  xlen = TMath::Sqrt(rmin * rmin - sConeAStepYdispl[iLay] * sConeAStepYdispl[iLay]) - sConeAStepXdispl[iLay];
  ylen = TMath::Sqrt(rmin * rmin - sConeAStepXdispl[iLay] * sConeAStepXdispl[iLay]) - sConeAStepYdispl[iLay];
  TGeoBBox* stepBoxSh = new TGeoBBox(Form("stepBoxASh%d", iLay), xlen / 2, ylen / 2, sConeAStepZlen / 2);

  xpos = sConeAStepXdispl[iLay] + stepBoxSh->GetDX();
  ypos = sConeAStepYdispl[iLay] + stepBoxSh->GetDY();
  TGeoTranslation* stepBoxTr = new TGeoTranslation(Form("stepBoxATr%d", iLay), xpos, ypos, 0);
  stepBoxTr->RegisterYourself();

  phimin = 90. - TMath::ACos(sConeAStepYdispl[iLay] / rmin) * TMath::RadToDeg() - 5;
  dphi = 90. - TMath::ASin(sConeAStepXdispl[iLay] / rmin) * TMath::RadToDeg() - phimin + 10;
  rmax = rmin + 2 * stepBoxSh->GetDY();

  TGeoPcon* stepPconSh = new TGeoPcon(Form("stepPconASh%d", iLay), phimin, dphi, 2);
  stepPconSh->DefineSection(0, -1.05 * sConeAStepZlen / 2, rmin, rmax);
  stepPconSh->DefineSection(1, 1.05 * sConeAStepZlen / 2, rmin, rmax);

  TGeoCompositeShape* stepASh = new TGeoCompositeShape(Form("stepBoxASh%d:stepBoxATr%d-stepPconASh%d", iLay, iLay, iLay));

  // We have all shapes: now create the real volumes
  TGeoMedium* medCarbon = mgr->GetMedium(Form("%s_M55J6K$", GetDetName())); // TO BE CHECKED
  TGeoMedium* medPEEK = mgr->GetMedium(Form("%s_PEEKCF30$", GetDetName()));

  TGeoVolume* coneABasisVol = new TGeoVolume(Form("ConeABasis%d", iLay), coneABasisSh, medCarbon);
  coneABasisVol->SetFillColor(kBlue);
  coneABasisVol->SetLineColor(kBlue);

  TGeoVolume* midRingVol = new TGeoVolume(Form("ConeAMidRing%d", iLay), midRingSh, medCarbon);
  coneABasisVol->SetFillColor(kBlue);
  coneABasisVol->SetLineColor(kBlue);

  TGeoVolume* coneARibVol = new TGeoVolume(Form("ConeARibVol%d", iLay), coneARibSh, medCarbon);
  coneARibVol->SetFillColor(kBlue);
  coneARibVol->SetLineColor(kBlue);

  TGeoVolume* stepAVol = new TGeoVolume(Form("ConeAStep%d", iLay), stepASh, medPEEK);
  stepAVol->SetFillColor(kBlue);
  stepAVol->SetLineColor(kBlue);

  // Finally put everything in the mother volume
  // (origin of local coordinates is at smaller end of Cone Basis)
  zref = sIBWheelACZdist / 2 - (sConeAStepHoleZpos + sConeAStepHoleZdist);

  zpos = zref;
  endWheel->AddNode(coneABasisVol, 1, new TGeoTranslation(0, 0, zpos));

  zpos = zref + sConeATotalLength[iLay] - sConeAMidRingZpos[iLay] - midRingSh->GetDz();
  endWheel->AddNode(midRingVol, 1, new TGeoTranslation(0, 0, zpos));

  rmin = sConeAExtSectDmin[iLay] / 2 - 0.035;
  zpos = zref + sConeATotalLength[iLay] - sConeARibsZpos;
  dphi = 180. / sConeANRibs[iLay];
  for (Int_t irib = 0; irib < 2 * sConeANRibs[iLay]; irib++) {
    Double_t phi = irib * dphi;
    xpos = rmin * TMath::Sin(phi * TMath::DegToRad());
    ypos = rmin * TMath::Cos(phi * TMath::DegToRad());
    endWheel->AddNode(coneARibVol, 1, new TGeoCombiTrans(xpos, -ypos, zpos, new TGeoRotation("", 90 + phi, 90, -90)));
  }

  // The position of the Steps is given wrt the holes (see eg. ALIITSUP0187)
  dphi = 180. - sConeAStepHolePhi0[iLay];

  Int_t numberOfStaves = GeometryTGeo::Instance()->getNumberOfStaves(iLay);
  zpos = zref + (static_cast<TGeoBBox*>(stepAVol->GetShape()))->GetDZ();
  for (Int_t j = 0; j < numberOfStaves; j++) {
    Double_t phi = dphi + j * sConeAStepHolePhi[iLay];
    endWheel->AddNode(stepAVol, j + 1, new TGeoCombiTrans(0, 0, zpos, new TGeoRotation("", 180, 180, -90 - phi)));
  }
}

void V3Services::ibEndWheelSideC(const Int_t iLay, TGeoVolume* endWheel, const TGeoManager* mgr)
{
  //
  // Creates the single End Wheel on Side C
  // for a given layer of the Inner Barrel
  // (Layer 0: ALIITSSUP0186+ALIITSUP0126)
  // (Layer 1: ALIITSSUP0176+ALIITSUP0123)
  // (Layer 2: ALIITSSUP0143+ALIITSUP0121)
  //
  // Input:
  //         iLay : the layer number
  //         endWheel : the whole end wheel volume
  //                    where to place the current created wheel
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      15 May 2019  Mario Sitta
  //               (partially based on P.Namwongsa implementation in AliRoot)
  //

  // The Basis C Side and the Reinforcement C Side are physically two
  // different pieces put together. For sake of simplicity here they are
  // made out of the same TGeoPcon volume. Moreover they are two halves,
  // so here they are made as a single cylinder.
  // The End Wheel Basis
  static const Double_t sEndWheelCDmax[3] = {57.0 * sMm, 73.0 * sMm, 89.0 * sMm};
  static const Double_t sEndWheelCDmin[3] = {44.5 * sMm, 58.0 * sMm, 74.0 * sMm};
  static const Double_t sEndWheelCHeigh[3] = {25.0 * sMm, 22.5 * sMm, 20.0 * sMm};
  static const Double_t sEndWheelCThick = 0.6 * sMm;

  static const Int_t sEndWCWallNHoles[3] = {6, 8, 10};
  static const Double_t sEndWCWallHoleD = 4.5 * sMm;
  static const Double_t sEndWCWallHoleZpos = 4.0 * sMm;

  static const Int_t sEndWCBaseNBigHoles = 5;
  static const Int_t sEndWCBaseNSmalHoles = 6;
  static const Double_t sEndWCBaseBigHoleD = 3.6 * sMm;
  static const Double_t sEndWCBaseSmalHoleD = 2.5 * sMm;
  static const Double_t sEndWCBaseHolesDpos[3] = {50.0 * sMm, 64.0 * sMm, 80.0 * sMm};
  static const Double_t sEndWCBaseHolesPhi = 15.0; // Deg

  // The End Wheel Reinforcement
  static const Double_t sEndWCRenfDmin[3] = {44.0 * sMm, 58.0 * sMm, 74.0 * sMm};
  static const Double_t sEndWCRenfDint[3] = {55.0 * sMm, 71.0 * sMm, 87.0 * sMm};
  static const Double_t sEndWCRenfHeigh[3] = {4.0 * sMm, 3.0 * sMm, 3.0 * sMm};
  static const Double_t sEndWCRenfThick = 0.6 * sMm;

  static const Double_t sEndWCRenfZpos = 14.2 * sMm;

  static const Int_t sEndWCRenfNSmalHoles[3] = {5, 7, 9};

  // The End Wheel Steps
  static const Double_t sEndWCStepXdispl[3] = {4.0 * sMm, 6.5 * sMm, 8.5 * sMm};
  static const Double_t sEndWCStepYdispl[3] = {24.4 * sMm, 32.1 * sMm, 39.6 * sMm};
  static const Double_t sEndWCStepR[3] = {27.8 * sMm, 35.8 * sMm, 43.8 * sMm};

  static const Double_t sEndWCStepZlen = 14.0 * sMm;

  static const Double_t sEndWCStepHoleXpos = 3.0 * sMm;
  static const Double_t sEndWCStepHoleZpos = 4.0 * sMm;
  static const Double_t sEndWCStepHoleZdist = 4.0 * sMm;

  static const Double_t sEndWCStepHolePhi[3] = {30.0, 22.5, 18.0}; // Deg
  static const Double_t sEndWCStepHolePhi0[2] = {9.5, 10.5};       // Deg - Lay 1-2
  static const Double_t sEndWCStepYlow = 7.0 * sMm;                // Lay 0 only

  // Local variables
  Double_t xlen, ylen, zlen;
  Double_t rmin, rmax, phimin, dphi;
  Double_t xpos, ypos, zpos;

  // Create the whole wheel (Basic + Reinforcement) as a CompositeShape
  // (a single Pcon minus the (copious!) holes)
  TGeoPcon* endwcbasis = new TGeoPcon(Form("endwcbasis%d", iLay), 0, 360, 10);

  rmin = sEndWheelCDmax[iLay] / 2 - sEndWheelCThick;
  endwcbasis->DefineSection(0, 0., rmin, sEndWheelCDmax[iLay] / 2);
  endwcbasis->DefineSection(1, sEndWCRenfZpos, rmin, sEndWheelCDmax[iLay] / 2);
  endwcbasis->DefineSection(2, sEndWCRenfZpos, sEndWCRenfDmin[iLay] / 2, sEndWheelCDmax[iLay] / 2);
  zlen = sEndWCRenfZpos + sEndWCRenfThick;
  endwcbasis->DefineSection(3, zlen, sEndWCRenfDmin[iLay] / 2, sEndWheelCDmax[iLay] / 2);
  endwcbasis->DefineSection(4, zlen, sEndWCRenfDint[iLay] / 2, sEndWheelCDmax[iLay] / 2);
  zlen = sEndWCRenfZpos + sEndWCRenfHeigh[iLay];
  endwcbasis->DefineSection(5, zlen, sEndWCRenfDint[iLay] / 2, sEndWheelCDmax[iLay] / 2);
  endwcbasis->DefineSection(6, zlen, rmin, sEndWheelCDmax[iLay] / 2);
  zlen = sEndWheelCHeigh[iLay] - sEndWheelCThick;
  endwcbasis->DefineSection(7, zlen, rmin, sEndWheelCDmax[iLay] / 2);
  endwcbasis->DefineSection(8, zlen, sEndWheelCDmin[iLay] / 2, sEndWheelCDmax[iLay] / 2);
  endwcbasis->DefineSection(9, sEndWheelCHeigh[iLay], sEndWheelCDmin[iLay] / 2, sEndWheelCDmax[iLay] / 2);

  TString endWheelComposite = Form("endwcbasis%d", iLay);

  // The holes in the vertical wall
  TGeoTube* endwcwalhol = new TGeoTube(Form("endwcwalhol%d", iLay), 0, sEndWCWallHoleD / 2, 4 * sEndWheelCThick);

  rmin = sEndWheelCDmax[iLay] / 2 - sEndWheelCThick / 2;
  zpos = sEndWCWallHoleZpos;
  dphi = 180. / sEndWCWallNHoles[iLay];
  phimin = dphi / 2.;
  for (Int_t ihole = 0; ihole < 2 * sEndWCWallNHoles[iLay]; ihole++) {
    Double_t phi = phimin + ihole * dphi;
    xpos = rmin * TMath::Sin(phi * TMath::DegToRad());
    ypos = rmin * TMath::Cos(phi * TMath::DegToRad());
    TGeoCombiTrans* endwcwhmat = new TGeoCombiTrans(Form("endwcwhmat%dl%d", ihole, iLay), xpos, ypos, zpos, new TGeoRotation("", -phi, 90, 0));
    endwcwhmat->RegisterYourself();
    endWheelComposite += Form("-endwcwalhol%d:endwcwhmat%dl%d", iLay, ihole, iLay);
  }

  // The holes in the base
  TGeoTube* endwcbasBhol = new TGeoTube(Form("endwcbasBhol%d", iLay), 0, sEndWCBaseBigHoleD / 2, 1.5 * sEndWheelCThick);

  TGeoTube* endwcbasShol = new TGeoTube(Form("endwcbasShol%d", iLay), 0, sEndWCBaseSmalHoleD / 2, 1.5 * sEndWheelCThick);

  rmin = sEndWCBaseHolesDpos[iLay] / 2;
  zpos = (endwcbasis->GetZ(8) + endwcbasis->GetZ(9)) / 2;

  char holename[strlen(endwcbasBhol->GetName()) + 1];

  phimin = 0.;
  for (Int_t ihole = 0; ihole < (sEndWCBaseNBigHoles + sEndWCBaseNSmalHoles); ihole++) {
    phimin += sEndWCBaseHolesPhi;
    xpos = rmin * TMath::Cos(phimin * TMath::DegToRad());
    ypos = rmin * TMath::Sin(phimin * TMath::DegToRad());
    TGeoTranslation* endwcbshmata = new TGeoTranslation(Form("endwcbshmat%dl%da", ihole, iLay), xpos, ypos, zpos);
    endwcbshmata->RegisterYourself();
    TGeoTranslation* endwcbshmatb = new TGeoTranslation(Form("endwcbshmat%dl%db", ihole, iLay), -xpos, -ypos, zpos);
    endwcbshmatb->RegisterYourself();
    if ((ihole > 1 && ihole < 5) || (ihole > 5 && ihole < 9)) { // Small holes
      strcpy(holename, endwcbasShol->GetName());
    } else {
      strcpy(holename, endwcbasBhol->GetName());
    }
    endWheelComposite += Form("-%s:endwcbshmat%dl%da-%s:endwcbshmat%dl%db", holename, ihole, iLay, holename, ihole, iLay);
  }

  // The holes in the reinforcement
  zpos = (endwcbasis->GetZ(2) + endwcbasis->GetZ(3)) / 2;

  phimin = 0.;
  dphi = 180. / (sEndWCRenfNSmalHoles[iLay] + 1);
  for (Int_t ihole = 0; ihole < sEndWCRenfNSmalHoles[iLay]; ihole++) {
    phimin += dphi;
    xpos = rmin * TMath::Cos(phimin * TMath::DegToRad());
    ypos = rmin * TMath::Sin(phimin * TMath::DegToRad());
    TGeoTranslation* endwcrshmata = new TGeoTranslation(Form("endwcrshmat%dl%da", ihole, iLay), xpos, ypos, zpos);
    endwcrshmata->RegisterYourself();
    TGeoTranslation* endwcrshmatb = new TGeoTranslation(Form("endwcrshmat%dl%db", ihole, iLay), -xpos, -ypos, zpos);
    endwcrshmatb->RegisterYourself();
    endWheelComposite += Form("-endwcbasShol%d:endwcrshmat%dl%da-endwcbasShol%d:endwcrshmat%dl%db", iLay, ihole, iLay, iLay, ihole, iLay);
  }

  TGeoCompositeShape* endWheelCSh = new TGeoCompositeShape(endWheelComposite.Data());

  // Now the Step as a Composite Shape (subtraction of a Pcon from a BBox)
  // (cutting volume should be slightly larger than desired region)
  rmin = sEndWCStepR[iLay];

  xlen = TMath::Sqrt(rmin * rmin - sEndWCStepYdispl[iLay] * sEndWCStepYdispl[iLay]) - sEndWCStepXdispl[iLay];
  ylen = TMath::Sqrt(rmin * rmin - sEndWCStepXdispl[iLay] * sEndWCStepXdispl[iLay]) - sEndWCStepYdispl[iLay];
  TGeoBBox* stepBoxSh = new TGeoBBox(Form("stepBoxCSh%d", iLay), xlen / 2, ylen / 2, sEndWCStepZlen / 2);

  xpos = sEndWCStepXdispl[iLay] + stepBoxSh->GetDX();
  ypos = sEndWCStepYdispl[iLay] + stepBoxSh->GetDY();
  TGeoTranslation* stepBoxTr = new TGeoTranslation(Form("stepBoxCTr%d", iLay), xpos, ypos, 0);
  stepBoxTr->RegisterYourself();

  phimin = 90. - TMath::ACos(sEndWCStepYdispl[iLay] / rmin) * TMath::RadToDeg() - 5;
  dphi = 90. - TMath::ASin(sEndWCStepXdispl[iLay] / rmin) * TMath::RadToDeg() - phimin + 10;
  rmax = rmin + 2 * stepBoxSh->GetDY();

  TGeoPcon* stepPconSh = new TGeoPcon(Form("stepPconCSh%d", iLay), phimin, dphi, 2);
  stepPconSh->DefineSection(0, -1.05 * sEndWCStepZlen / 2, rmin, rmax);
  stepPconSh->DefineSection(1, 1.05 * sEndWCStepZlen / 2, rmin, rmax);

  TGeoCompositeShape* stepCSh = new TGeoCompositeShape(Form("stepBoxCSh%d:stepBoxCTr%d-stepPconCSh%d", iLay, iLay, iLay));

  // We have all shapes: now create the real volumes
  TGeoMedium* medCarbon = mgr->GetMedium(Form("%s_M55J6K$", GetDetName())); // TO BE CHECKED
  TGeoMedium* medPEEK = mgr->GetMedium(Form("%s_PEEKCF30$", GetDetName()));

  TGeoVolume* endWheelCVol = new TGeoVolume(Form("EndWheelCBasis%d", iLay), endWheelCSh, medCarbon);
  endWheelCVol->SetFillColor(kBlue);
  endWheelCVol->SetLineColor(kBlue);

  TGeoVolume* stepCVol = new TGeoVolume(Form("EndWheelCStep%d", iLay), stepCSh, medPEEK);
  stepCVol->SetFillColor(kBlue);
  stepCVol->SetLineColor(kBlue);

  // Finally put everything in the mother volume
  zpos = sIBWheelACZdist / 2 - (sEndWCStepHoleZpos + sEndWCStepHoleZdist);
  endWheel->AddNode(endWheelCVol, 1, new TGeoCombiTrans(0, 0, -zpos, new TGeoRotation("", 0, 180, 0)));

  // The position of the Steps is given wrt the holes (see eg. ALIITSUP0187)
  if (iLay == 0) {
    dphi = (sEndWCStepYlow / sEndWCStepYdispl[iLay]) * TMath::RadToDeg();
  } else {
    dphi = sEndWCStepHolePhi0[iLay - 1];
  }

  Int_t numberOfStaves = GeometryTGeo::Instance()->getNumberOfStaves(iLay);
  zpos += (static_cast<TGeoBBox*>(stepCVol->GetShape()))->GetDZ();
  for (Int_t j = 0; j < numberOfStaves; j++) {
    Double_t phi = dphi + j * sEndWCStepHolePhi[iLay];
    endWheel->AddNode(stepCVol, j + 1, new TGeoCombiTrans(0, 0, -zpos, new TGeoRotation("", 180, 180, -90 - phi)));
  }
}

TGeoXtru* V3Services::ibEndWheelARibShape(const Int_t iLay)
{
  //
  // Creates the shape of a Rib on Side A cone
  // (Layer 0: ALIITSSUP0182)
  // (Layer 1: ALIITSSUP0172)
  // (Layer 2: ALIITSSUP0136)
  //
  // Input:
  //         iLay : the layer number
  //
  // Output:
  //
  // Return:
  //        the Rib shape as a TGeoXtru
  //
  // Created:      23 Aug 2019  Mario Sitta
  //

  static const Int_t sConeARibNVert = 8;

  static const Double_t sConeARibWidth = 27.3 * sMm;
  static const Double_t sConeARibTotalLen[3] = {98.03 * sMm, 104.65 * sMm, 101.43 * sMm};
  static const Double_t sConeARibIntLen[3] = {50.0 * sMm, 40.0 * sMm, 28.5 * sMm};
  static const Double_t sConeARibStep[3] = {1.0 * sMm, 1.1 * sMm, 1.0 * sMm};
  static const Double_t sConeARibLenToStep[3] = {42.0 * sMm, 29.0 * sMm, 26.0 * sMm};
  static const Double_t sConeARibLenAfterStep[3] = {50.5 * sMm, 37.0 * sMm, 33.5 * sMm};
  static const Double_t sConeARibVertexPos[3] = {9.0 * sMm, 40.0 * sMm, 58.0 * sMm};
  static const Double_t sConeARibIntAngle = 45.0;               // Deg
  static const Double_t sConeARibVertexAngle[2] = {20.0, 30.0}; // Deg

  static const Double_t sConeARibThick = 0.9 * sMm;

  // Local variables
  Double_t xtru[sConeARibNVert], ytru[sConeARibNVert];

  // Rib shapes for Layer 0 and Layers 1,2 are different
  xtru[0] = 0.;
  ytru[0] = 0.;
  xtru[1] = sConeARibLenToStep[iLay];
  ytru[1] = 0.;
  xtru[2] = xtru[1];
  ytru[2] = sConeARibStep[iLay];
  xtru[3] = sConeARibLenAfterStep[iLay];
  ytru[3] = ytru[2];
  xtru[4] = sConeARibTotalLen[iLay];
  if (iLay == 0) {
    ytru[4] = sConeARibWidth - sConeARibVertexPos[iLay];
    xtru[5] = sConeARibIntLen[iLay] + sConeARibVertexPos[iLay] / TMath::Tan(sConeARibIntAngle * TMath::DegToRad());
  } else {
    ytru[4] = sConeARibVertexPos[iLay];
    xtru[5] = sConeARibIntLen[iLay] + (sConeARibVertexPos[iLay] - sConeARibWidth) / TMath::Tan(sConeARibVertexAngle[iLay - 1] * TMath::DegToRad());
  }
  ytru[5] = ytru[4];
  xtru[6] = sConeARibIntLen[iLay];
  ytru[6] = sConeARibWidth;
  xtru[7] = 0.;
  ytru[7] = ytru[6];

  // The actual Xtru
  TGeoXtru* ribShape = new TGeoXtru(2);
  ribShape->DefinePolygon(sConeARibNVert, xtru, ytru);
  ribShape->DefineSection(0, -sConeARibThick / 2);
  ribShape->DefineSection(1, sConeARibThick / 2);

  return ribShape;
}

TGeoVolume* V3Services::ibCyssCylinder(const TGeoManager* mgr)
{
  //
  // Creates the cylinder of the Inner Barrel CYSS
  // (ALIITSUP0191)
  //
  // Input:
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //         the cylinder as a TGeoVolume
  //
  // Created:      21 Oct 2019  Mario Sitta
  //

  static const Double_t sCyssCylInnerD = 95.6 * sMm;
  static const Double_t sCyssCylOuterD = 100.0 * sMm;
  static const Double_t sCyssCylZLength = 353.0 * sMm;
  static const Double_t sCyssCylFabricThick = 0.1 * sMm;

  // Local variables
  Double_t rmin, rmax, zlen, phimin, phimax, dphi;

  // First create the shapes
  rmin = sCyssCylInnerD / 2;
  rmax = sCyssCylOuterD / 2;
  zlen = sCyssCylZLength / 2;
  TGeoTubeSeg* cyssOuterCylSh = new TGeoTubeSeg(rmin, rmax, zlen, 180, 360);

  rmin += sCyssCylFabricThick;
  rmax -= sCyssCylFabricThick;
  zlen -= sCyssCylFabricThick;

  dphi = TMath::ASin(sCyssCylFabricThick / rmax);
  phimin = 180 + dphi * TMath::RadToDeg();
  phimax = 360 - dphi * TMath::RadToDeg();

  TGeoTubeSeg* cyssInnerCylSh = new TGeoTubeSeg(rmin, rmax, zlen, phimin, phimax);

  // We have all shapes: now create the real volumes
  TGeoMedium* medPrepreg = mgr->GetMedium(Form("%s_AS4C200$", GetDetName()));
  TGeoMedium* medRohacell = mgr->GetMedium(Form("%s_RIST110$", GetDetName()));

  TGeoVolume* cyssOuterCylVol = new TGeoVolume("IBCYSSCylinder", cyssOuterCylSh, medPrepreg);
  cyssOuterCylVol->SetLineColor(35);

  TGeoVolume* cyssInnerCylVol = new TGeoVolume("IBCYSSCylinderFoam", cyssInnerCylSh, medRohacell);
  cyssInnerCylVol->SetLineColor(kGreen);

  cyssOuterCylVol->AddNode(cyssInnerCylVol, 1, nullptr);

  // Finally return the cylinder volume
  return cyssOuterCylVol;
}

TGeoVolume* V3Services::ibCyssCone(const TGeoManager* mgr)
{
  //
  // Creates the cone of the Inner Barrel CYSS
  // (ALIITSUP0190)
  //
  // Input:
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //         the cone as a TGeoVolume
  //
  // Created:      24 Oct 2019  Mario Sitta
  //

  static const Double_t sCyssConeTotalLength = 150.0 * sMm;

  static const Double_t sCyssConeIntSectDmin = 100.0 * sMm;
  static const Double_t sCyssConeIntSectDmax = 101.2 * sMm;
  static const Double_t sCyssConeIntSectZlen = 23.0 * sMm;
  static const Double_t sCyssConeIntCylZlen = 15.0 * sMm;

  static const Double_t sCyssConeExtSectDmin = 246.0 * sMm;
  static const Double_t sCyssConeExtSectDmax = 257.2 * sMm;
  static const Double_t sCyssConeExtSectZlen = 42.0 * sMm;
  static const Double_t sCyssConeExtCylZlen = 40.0 * sMm;

  static const Double_t sCyssConeOpeningAngle = 40.0; // Deg

  static const Double_t sCyssConeFabricThick = 0.3 * sMm;

  // Local variables
  Double_t rmin, rmax, zlen1, zlen2, phimin, phirot, dphi;
  Double_t x1, y1, x2, y2, x3, y3, m, xin, yin;

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

  dphi = TMath::ASin(sCyssConeFabricThick / (0.5 * sCyssConeIntSectDmax));
  phimin = 180 + dphi * TMath::RadToDeg();
  phirot = 180 - 2 * dphi * TMath::RadToDeg();

  // The foam cone is built from the points of the outer cone
  TGeoPcon* cyssConeFoamSh = new TGeoPcon(phimin, phirot, 5);

  m = TMath::Tan(sCyssConeOpeningAngle * TMath::DegToRad());
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
  TGeoMedium* medPrepreg = mgr->GetMedium(Form("%s_AS4C200$", GetDetName()));
  TGeoMedium* medRohacell = mgr->GetMedium(Form("%s_RIST110$", GetDetName()));

  TGeoVolume* cyssConeVol = new TGeoVolume("IBCYSSCone", cyssConeSh, medPrepreg);
  cyssConeVol->SetLineColor(35);

  TGeoVolume* cyssConeFoamVol = new TGeoVolume("IBCYSSConeFoam", cyssConeFoamSh, medRohacell);
  cyssConeFoamVol->SetLineColor(kGreen);

  cyssConeVol->AddNode(cyssConeFoamVol, 1, nullptr);

  // Finally return the cone volume
  return cyssConeVol;
}

TGeoVolume* V3Services::ibCyssFlangeSideA(const TGeoManager* mgr)
{
  //
  // Creates the Flange on Side A for the Inner Barrel CYSS
  // (ALIITSUP0189)
  //
  // Input:
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //         the flange as a TGeoVolume
  //
  // Created:      28 Oct 2019  Mario Sitta
  //

  // Radii of the steps
  static const Double_t sCyssFlangeAStep1Dmin = 254.1 * sMm;
  static const Double_t sCyssFlangeAStep1Dmax = 287.0 * sMm;
  static const Double_t sCyssFlangeAStep2Dmax = 259.0 * sMm;
  static const Double_t sCyssFlangeAStep3Dmin = 243.0 * sMm;
  static const Double_t sCyssFlangeAStep3Dmax = 245.5 * sMm;
  static const Double_t sCyssFlangeAStep4Dmax = 239.0 * sMm;
  static const Double_t sCyssFlangeAInnerD = 236.0 * sMm;
  static const Double_t sCyssFlangeAInRingD = 238.0 * sMm;

  // Heights of the steps
  static const Double_t sCyssFlangeATotHei = 39.0 * sMm;
  static const Double_t sCyssFlangeAStep1H = 5.5 * sMm;
  static const Double_t sCyssFlangeAInRingH = 7.0 * sMm;
  static const Double_t sCyssFlangeAInRingUp = 1.0 * sMm;
  static const Double_t sCyssFlangeAStep2H = 9.0 * sMm;
  static const Double_t sCyssFlangeAStep3H = 10.0 * sMm;
  static const Double_t sCyssFlangeAStep4H = 8.5 * sMm;

  // The wings
  static const Double_t sCyssFlangeAWingD = 307.0 * sMm;
  static const Double_t sCyssFlangeAWingW = 16.0 * sMm;

  // Holes
  static const Double_t sCyssFlangeANotchW = 3.0 * sMm;

  static const Double_t sCyssFlangeAHolesDpos = 274.0 * sMm;

  static const Double_t sCyssFlangeAHole1Num = 8;
  static const Double_t sCyssFlangeAHole1D = 5.5 * sMm;
  static const Double_t sCyssFlangeAHole1Phi0 = 10;    // Deg
  static const Double_t sCyssFlangeAHole1PhiStep = 20; // Deg

  static const Double_t sCyssFlangeAHole2D = 4.0 * sMm;
  static const Double_t sCyssFlangeAHole2Phi = 20; // Deg

  static const Double_t sCyssFlangeAHole3D = 7.0 * sMm;
  static const Double_t sCyssFlangeAHole3Phi = 6; // Deg

  static const Double_t sCyssFlangeAWingHoleD = 8.1 * sMm;
  static const Double_t sCyssFlangeAWingHoleYpos = 9.0 * sMm;
  static const Double_t sCyssFlangeAWingHoleRpos = 146.0 * sMm;

  // Local variables
  Double_t rmin, rmax, zlen, phi, dphi;
  Double_t xpos, ypos;

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

  for (Int_t i = 0; i < sCyssFlangeAHole1Num / 2; i++) {
    Double_t phi = sCyssFlangeAHole1Phi0 + i * sCyssFlangeAHole1PhiStep;
    xpos = 0.5 * sCyssFlangeAHolesDpos * TMath::Sin(phi * TMath::DegToRad());
    ypos = 0.5 * sCyssFlangeAHolesDpos * TMath::Cos(phi * TMath::DegToRad());
    TGeoTranslation* hole1Tr1 = new TGeoTranslation(Form("hole1Tr1%d", i), xpos, -ypos, zlen);
    hole1Tr1->RegisterYourself();
    TGeoTranslation* hole1Tr2 = new TGeoTranslation(Form("hole1Tr2%d", i), -xpos, -ypos, zlen);
    hole1Tr2->RegisterYourself();
    cyssFlangeAComposite += Form("-hole1:hole1Tr1%d-hole1:hole1Tr2%d", i, i);
  }

  // The 2 smaller round holes (1 on each side)
  rmax = sCyssFlangeAHole2D / 2;
  TGeoTube* hole2 = new TGeoTube("hole2", 0, rmax, 2 * zlen);

  xpos = 0.5 * sCyssFlangeAHolesDpos * TMath::Sin(sCyssFlangeAHole2Phi * TMath::DegToRad());
  ypos = 0.5 * sCyssFlangeAHolesDpos * TMath::Cos(sCyssFlangeAHole2Phi * TMath::DegToRad());
  TGeoTranslation* hole2Tr1 = new TGeoTranslation("hole2Tr1", xpos, -ypos, zlen);
  hole2Tr1->RegisterYourself();
  TGeoTranslation* hole2Tr2 = new TGeoTranslation("hole2Tr2", -xpos, -ypos, zlen);
  hole2Tr2->RegisterYourself();

  cyssFlangeAComposite += "-hole2:hole2Tr1-hole2:hole2Tr2";

  // The 2 bigger round holes (1 on each side)
  rmax = sCyssFlangeAHole3D / 2;
  TGeoTube* hole3 = new TGeoTube("hole3", 0, rmax, 2 * zlen);

  xpos = 0.5 * sCyssFlangeAHolesDpos * TMath::Sin(sCyssFlangeAHole3Phi * TMath::DegToRad());
  ypos = 0.5 * sCyssFlangeAHolesDpos * TMath::Cos(sCyssFlangeAHole3Phi * TMath::DegToRad());
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
  TString cyssFlangeAHollows = ibCreateHollowsCyssFlangeSideA(zlen);

  cyssFlangeAComposite += cyssFlangeAHollows.Data();

  // The final flange shape
  TGeoCompositeShape* cyssFlangeASh = new TGeoCompositeShape(cyssFlangeAComposite.Data());

  // We have all shapes: now create the real volumes
  TGeoMedium* medAlu = mgr->GetMedium(Form("%s_ALUMINUM$", GetDetName()));

  TGeoVolume* cyssFlangeAVol = new TGeoVolume("IBCYSSFlangeA", cyssFlangeASh, medAlu);
  cyssFlangeAVol->SetLineColor(kCyan);
  cyssFlangeAVol->SetFillColor(kCyan);

  // Finally return the flange volume
  return cyssFlangeAVol;
}

TString V3Services::ibCreateHollowsCyssFlangeSideA(const Double_t zlen)
{
  //
  // Creates the very complicate hollow holes in the Flange
  // on Side A for the Inner Barrel CYSS
  // (ALIITSUP0189)
  //
  // Input:
  //         zlen : the thickness of the ring where the hollows are located
  //
  // Output:
  //
  // Return:
  //         the string describing the holes and their positions
  //
  // Created:      04 Nov 2019  Mario Sitta
  //

  static const Double_t sCyssFlangeAHolesDpos = 274.0 * sMm;

  static const Double_t sCyssFlangeAHole1Phi0 = 10;    // Deg
  static const Double_t sCyssFlangeAHole1PhiStep = 20; // Deg

  static const Double_t sCyssFlangeAHole2Phi = 20; // Deg

  static const Double_t sCyssFlangeAHollowD = 7.0 * sMm;
  static const Double_t sCyssFlangeAHollowPhi0 = 13; // Deg
  static const Double_t sCyssFlangeAHollowPhi1 = 8;  // Deg

  // Local variables
  Double_t rmin, rmax, phi, dphi;
  Double_t xpos, ypos;

  TString cyssFlangeAHollows;

  //
  rmax = sCyssFlangeAHollowD / 2;
  TGeoTubeSeg* roundHalf = new TGeoTubeSeg("roundhalf", 0, rmax, 2 * zlen, 0, 180);

  Double_t rHoles = sCyssFlangeAHolesDpos / 2;

  xpos = rHoles * TMath::Cos(sCyssFlangeAHollowPhi0 * TMath::DegToRad());
  ypos = rHoles * TMath::Sin(sCyssFlangeAHollowPhi0 * TMath::DegToRad());
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

  for (Int_t j = 1; j < 4; j++) {
    phi = 90 - (sCyssFlangeAHole1Phi0 + j * sCyssFlangeAHole1PhiStep + 0.5 * sCyssFlangeAHollowPhi1);
    xpos = rHoles * TMath::Cos(phi * TMath::DegToRad());
    ypos = rHoles * TMath::Sin(phi * TMath::DegToRad());
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
    xpos = rHoles * TMath::Cos(phi * TMath::DegToRad());
    ypos = rHoles * TMath::Sin(phi * TMath::DegToRad());
    TGeoCombiTrans* roundTr5 = new TGeoCombiTrans(Form("roundtr%d", j + 8), xpos, -ypos, zlen, new TGeoRotation("", -phi, 0, 0));
    roundTr5->RegisterYourself();
    TGeoCombiTrans* roundTr6 = new TGeoCombiTrans(Form("roundtr%d", j + 11), -xpos, -ypos, zlen, new TGeoRotation("", phi, 0, 0));
    roundTr6->RegisterYourself();

    cyssFlangeAHollows += Form("-roundhalf:roundtr%d-roundhalf:roundtr%d", j + 8, j + 11);
  }

  //
  phi = 90 - (sCyssFlangeAHole2Phi + 0.5 * sCyssFlangeAHollowPhi1);
  xpos = rHoles * TMath::Cos(phi * TMath::DegToRad());
  ypos = rHoles * TMath::Sin(phi * TMath::DegToRad());
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
  xpos = rHoles * TMath::Cos(phi * TMath::DegToRad());
  ypos = rHoles * TMath::Sin(phi * TMath::DegToRad());
  TGeoCombiTrans* roundTr17 = new TGeoCombiTrans("roundtr17", xpos, -ypos, zlen, new TGeoRotation("", -phi, 0, 0));
  roundTr17->RegisterYourself();
  TGeoCombiTrans* roundTr18 = new TGeoCombiTrans("roundtr18", -xpos, -ypos, zlen, new TGeoRotation("", phi, 0, 0));
  roundTr18->RegisterYourself();

  cyssFlangeAHollows += "-roundhalf:roundtr17-roundhalf:roundtr18";

  phi = 90 - (sCyssFlangeAHole1Phi0 + 0.5 * sCyssFlangeAHollowPhi1);
  xpos = rHoles * TMath::Cos(phi * TMath::DegToRad());
  ypos = rHoles * TMath::Sin(phi * TMath::DegToRad());
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

TGeoVolume* V3Services::ibCyssFlangeSideC(const TGeoManager* mgr)
{
  //
  // Creates the Flange on Side C for the Inner Barrel CYSS
  // (ALIITSUP0098)
  //
  // Input:
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //         the flange as a TGeoVolume
  //
  // Created:      23 Oct 2019  Mario Sitta
  //

  // Radii of the rings
  static const Double_t sCyssFlangeCDmin1 = 44.0 * sMm;
  static const Double_t sCyssFlangeCDmin2 = 57.0 * sMm;
  static const Double_t sCyssFlangeCDmin3 = 73.0 * sMm;

  static const Double_t sCyssFlangeCDmax1 = 58.8 * sMm;
  static const Double_t sCyssFlangeCDmax2 = 74.8 * sMm;
  static const Double_t sCyssFlangeCDmax3 = 94.0 * sMm;

  static const Double_t sCyssFlangeCDWallIn = 89.0 * sMm;
  static const Double_t sCyssFlangeCDWallOut = 95.6 * sMm;

  static const Double_t sCyssFlangeCDExt = 100.0 * sMm;

  // Thicknesses and heights
  static const Double_t sCyssFlangeCTotH = 10.0 * sMm;
  static const Double_t sCyssFlangeCExtThick = 1.0 * sMm;

  static const Double_t sCyssFlangeCHmax1 = 1.5 * sMm;
  static const Double_t sCyssFlangeCHmax2 = 4.0 * sMm;
  static const Double_t sCyssFlangeCHmax3 = 6.5 * sMm;

  static const Double_t sCyssFlangeCHmin2 = 2.5 * sMm;
  static const Double_t sCyssFlangeCHmin3 = 5.0 * sMm;

  // Holes
  static const Double_t sHoles22Dia = 2.2 * sMm;
  static const Double_t sHoles22Phi = 60; // Deg

  static const Double_t sHoles30Dia = 3.0 * sMm;
  static const Double_t sHoles30Phi = 15; // Deg

  static const Double_t sHoles12Dia = 1.2 * sMm;
  static const Double_t sHoles12Phi = 75; // Deg

  static const Double_t sHolesDdist[3] = {50.0 * sMm, 64.0 * sMm, 80.0 * sMm};

  static const Double_t sCyssFlangeCNotchH = 3.2 * sMm;
  static const Double_t sCyssFlangeCNotchW = 3.0 * sMm;

  // Local variables
  Double_t rmin, rmax, zlen;
  Double_t xpos, ypos;

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

  for (Int_t j = 0; j < 3; j++) {
    ypos = sHolesDdist[j] / 2;
    TGeoTranslation* holeCTr = new TGeoTranslation(Form("holeCTr%d", j), 0, -ypos, zlen);
    holeCTr->RegisterYourself();
    cyssFlangeCComposite += Form("-hole22:holeCTr%d", j);

    xpos = TMath::Sin(sHoles22Phi * TMath::DegToRad()) * sHolesDdist[j] / 2;
    ypos = TMath::Cos(sHoles22Phi * TMath::DegToRad()) * sHolesDdist[j] / 2;
    TGeoTranslation* holeLTr = new TGeoTranslation(Form("holeLTr%d", j), xpos, -ypos, zlen);
    holeLTr->RegisterYourself();
    cyssFlangeCComposite += Form("-hole22:holeLTr%d", j);

    TGeoTranslation* holeRTr = new TGeoTranslation(Form("holeRTr%d", j), -xpos, -ypos, zlen);
    holeRTr->RegisterYourself();
    cyssFlangeCComposite += Form("-hole22:holeRTr%d", j);
  }

  rmax = sHoles30Dia / 2;
  TGeoTube* hole30 = new TGeoTube("hole30", 0, rmax, zlen);

  for (Int_t k = 0; k < 3; k++) {
    Double_t phi = (k + 1) * sHoles30Phi;
    for (Int_t j = 0; j < 3; j++) {
      xpos = TMath::Sin(phi * TMath::DegToRad()) * sHolesDdist[j] / 2;
      ypos = TMath::Cos(phi * TMath::DegToRad()) * sHolesDdist[j] / 2;

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

  for (Int_t j = 0; j < 3; j++) {
    xpos = TMath::Sin(sHoles12Phi * TMath::DegToRad()) * sHolesDdist[j] / 2;
    ypos = TMath::Cos(sHoles12Phi * TMath::DegToRad()) * sHolesDdist[j] / 2;
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
  TGeoMedium* medAlu = mgr->GetMedium(Form("%s_ALUMINUM$", GetDetName()));

  TGeoVolume* cyssFlangeCVol = new TGeoVolume("IBCYSSFlangeC", cyssFlangeCSh, medAlu);
  cyssFlangeCVol->SetLineColor(kCyan);
  cyssFlangeCVol->SetFillColor(kCyan);

  // Finally return the flange volume
  return cyssFlangeCVol;
}

void V3Services::obEndWheelSideA(const Int_t iLay, TGeoVolume* mother, const TGeoManager* mgr)
{
  //
  // Creates the single End Wheel on Side A
  // for a given layer of the Middle and Outer Barrels
  // (Layer 3: ALICE-W3-01-Side_A, Layer 4: ALICE-W4-01-Side_A,
  //  Layer 5: ALICE-W5-01-Side_A  Layer 6: ALICE-W6-01-Side_A)
  //
  // Input:
  //         iLay : the layer number (0,1: Middle, 2,3: Outer)
  //         endWheel : the volume where to place the current created wheel
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      24 Sep 2019  Mario Sitta
  // Updated:      27 Sep 2019  Mario Sitta
  //

  // The support ring
  static const Double_t sOBWheelTotZlen = 55.0 * sMm;
  static const Double_t sOBWheelSuppZlen = 35.0 * sMm;

  static const Double_t sOBWheelSuppRmin[4] = {204.0 * sMm, 254.0 * sMm, 352.5 * sMm, 402.0 * sMm};
  static const Double_t sOBWheelSuppRmax[4] = {241.0 * sMm, 291.0 * sMm, 389.0 * sMm, 448.5 * sMm};

  // The support blocks
  static const Double_t sOBWheelShelfWide[4] = {56.05 * sMm, 55.15 * sMm, 54.10 * sMm, 53.81 * sMm}; // TO BE CHECKED
  static const Double_t sOBWheelShelfHoleZpos = 48.0 * sMm;

  static const Double_t sOBWheelShelfRpos[4] = {213.0 * sMm, 262.5 * sMm, 361.0 * sMm, 410.5 * sMm};
  static const Double_t sOBWheelShelfPhi0[4] = {0.0, 0.0, 0.0, 0.0}; // Deg

  // Local variables
  Double_t xlen, ylen, zlen;
  Double_t rmin, rmax, phimin, dphi;
  Double_t xpos, ypos, zpos;

  // The Support Wheel is physically a single piece, a hollow ring
  // plus the stave support shelves
  // For the sake of simplicity we build it up with four TGeoTube's
  // one per each wall of the ring (inner, outer, lower, upper) plus
  // as many TGeoBBox's as needed for the shelves

  // The inner ring
  TGeoTube* innerRingSh = new TGeoTube(sOBWheelSuppRmin[iLay], sOBWheelSuppRmax[iLay], sOBWheelThickness / 2);

  // The outer ring
  TGeoTube* outerRingSh = new TGeoTube(sOBWheelSuppRmin[iLay], sOBWheelSuppRmax[iLay], sOBWheelThickness / 2);

  // The lower ring
  rmax = sOBWheelSuppRmin[iLay] + sOBWheelThickness;
  zlen = sOBWheelSuppZlen - 2 * sOBWheelThickness;
  TGeoTube* lowerRingSh = new TGeoTube(sOBWheelSuppRmin[iLay], rmax, zlen / 2);

  // The upper ring
  rmin = sOBWheelSuppRmax[iLay] - sOBWheelThickness;
  TGeoTube* upperRingSh = new TGeoTube(rmin, sOBWheelSuppRmax[iLay], zlen / 2);

  // The shelf support
  xlen = sOBWheelShelfWide[iLay];
  ylen = 2 * sOBWheelThickness;
  zlen = sOBWheelTotZlen - sOBWheelSuppZlen;
  TGeoBBox* shelfSh = new TGeoBBox(xlen / 2, ylen / 2, zlen / 2);

  // We have all shapes: now create the real volumes
  TGeoMedium* medCarbon = mgr->GetMedium(Form("%s_M55J6K$", GetDetName())); // TO BE CHECKED

  Int_t nLay = iLay + sNumberInnerLayers;

  TGeoVolume* ringInnerVol = new TGeoVolume(Form("OBEndWheelAInnerRing%d", nLay), innerRingSh, medCarbon);
  ringInnerVol->SetFillColor(kBlue);
  ringInnerVol->SetLineColor(kBlue);

  TGeoVolume* ringOuterVol = new TGeoVolume(Form("OBEndWheelAOuterRing%d", nLay), outerRingSh, medCarbon);
  ringOuterVol->SetFillColor(kBlue);
  ringOuterVol->SetLineColor(kBlue);

  TGeoVolume* ringLowerVol = new TGeoVolume(Form("OBEndWheelALowerRing%d", nLay), lowerRingSh, medCarbon);
  ringLowerVol->SetFillColor(kBlue);
  ringLowerVol->SetLineColor(kBlue);

  TGeoVolume* ringUpperVol = new TGeoVolume(Form("OBEndWheelAUpperRing%d", nLay), upperRingSh, medCarbon);
  ringUpperVol->SetFillColor(kBlue);
  ringUpperVol->SetLineColor(kBlue);

  TGeoVolume* shelfVol = new TGeoVolume(Form("OBEndWheelAShelf%d", nLay), shelfSh, medCarbon);
  shelfVol->SetFillColor(kBlue);
  shelfVol->SetLineColor(kBlue);

  // Finally put everything in the mother volume
  // In blueprints the Z position is given wrt the shelf holes
  // First the ring
  if (iLay < sNumberMiddlLayers) {
    zpos = sMBWheelsZpos + sOBWheelShelfHoleZpos;
  } else {
    zpos = sOBWheelsZpos + sOBWheelShelfHoleZpos;
  }

  zpos -= outerRingSh->GetDz();
  mother->AddNode(ringOuterVol, 1, new TGeoTranslation(0, 0, zpos));

  zpos -= (outerRingSh->GetDz() + lowerRingSh->GetDz());
  mother->AddNode(ringLowerVol, 1, new TGeoTranslation(0, 0, zpos));
  mother->AddNode(ringUpperVol, 1, new TGeoTranslation(0, 0, zpos));

  zpos -= (lowerRingSh->GetDz() + innerRingSh->GetDz());
  mother->AddNode(ringInnerVol, 1, new TGeoTranslation(0, 0, zpos));

  // Then the support blocks
  Int_t numberOfStaves = GeometryTGeo::Instance()->getNumberOfStaves(nLay);
  Double_t alpha = 360. / numberOfStaves;

  rmin = sOBWheelShelfRpos[iLay] + shelfSh->GetDY();
  zpos -= (innerRingSh->GetDz() + shelfSh->GetDZ());

  for (Int_t j = 0; j < numberOfStaves; j++) { // As in V3Layer::createLayer
    Double_t phi = j * alpha + sOBWheelShelfPhi0[iLay];
    xpos = rmin * cosD(phi);
    ypos = rmin * sinD(phi);
    phi += 90;
    mother->AddNode(shelfVol, j, new TGeoCombiTrans(xpos, ypos, zpos, new TGeoRotation("", phi, 0, 0)));
  }
}

void V3Services::mbEndWheelSideC(const Int_t iLay, TGeoVolume* mother, const TGeoManager* mgr)
{
  //
  // Creates the single End Wheel on Side C
  // for a given layer of the Middle Barrel
  // (wheels on Side C are very different for Middle and Outer Barrels,
  // so we cannot use a single method for both as for Side A)
  // (Layer 3: ALICE-W3-04-Side_C, Layer 4: ALICE-W4-05-Side_C)
  //
  // Input:
  //         iLay : the layer number (0,1: Middle, 2,3: Outer)
  //         endWheel : the volume where to place the current created wheel
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      26 Sep 2019  Mario Sitta
  //

  // The support ring
  static const Double_t sOBWheelTotZlen[2] = {63.0 * sMm, 55.0 * sMm};
  static const Double_t sOBWheelSuppZlen[2] = {43.0 * sMm, 35.0 * sMm};

  static const Double_t sOBWheelSuppRmin[2] = {200.5 * sMm, 254.0 * sMm};
  static const Double_t sOBWheelSuppRmax[2] = {237.5 * sMm, 291.0 * sMm};
  static const Double_t sOBWheelFlangeR[2] = {255.0 * sMm, 239.5 * sMm};
  static const Double_t sOBWheelFlangeZlen = 8.0 * sMm;

  // The support blocks
  static const Double_t sOBWheelShelfWide[2] = {56.05 * sMm, 55.15 * sMm}; // TO BE CHECKED
  static const Double_t sOBWheelShelfHoleZpos[2] = {56.0 * sMm, 48.0 * sMm};

  static const Double_t sOBWheelShelfRpos[2] = {213.0 * sMm, 262.5 * sMm};
  static const Double_t sOBWheelShelfPhi0[2] = {0.0, 0.0}; // Deg

  // Local variables
  Double_t xlen, ylen, zlen;
  Double_t rmin, rmax, phimin, dphi;
  Double_t xpos, ypos, zpos;
  Int_t nsect;

  // The Support Wheel is physically a single piece, a hollow ring
  // with a flange plus the stave support shelves
  // Unfortunately the flange is on opposite sides on the two layers
  // (externally to the ring for layer 3, internally for layer 4)
  // For the sake of simplicity we build it up with three TGeoTube's and
  // one TGeoPcon for each wall of the ring (inner, outer, lower, upper)
  // plus as many TGeoBBox's as needed for the shelves

  // The inner ring
  TGeoTube* innerRingSh = new TGeoTube(sOBWheelSuppRmin[iLay], sOBWheelSuppRmax[iLay], sOBWheelThickness / 2);

  // The outer ring with the flange
  if (iLay == 0) {
    nsect = 6;
  } else {
    nsect = 4;
  }

  TGeoPcon* outerRingSh = new TGeoPcon(0, 360, nsect);

  if (iLay == 0) {
    rmin = sOBWheelSuppRmax[0] - 2 * sOBWheelThickness;
    outerRingSh->DefineSection(0, 0., rmin, sOBWheelFlangeR[0]);
    outerRingSh->DefineSection(1, 2 * sOBWheelThickness, rmin, sOBWheelFlangeR[0]);
    outerRingSh->DefineSection(2, 2 * sOBWheelThickness, rmin, sOBWheelSuppRmax[0]);
    outerRingSh->DefineSection(3, sOBWheelFlangeZlen, rmin, sOBWheelSuppRmax[0]);
    outerRingSh->DefineSection(4, sOBWheelFlangeZlen, sOBWheelSuppRmin[0], sOBWheelSuppRmax[0]);
    zlen = sOBWheelFlangeZlen + sOBWheelThickness;
    outerRingSh->DefineSection(5, zlen, sOBWheelSuppRmin[0], sOBWheelSuppRmax[0]);
  } else {
    outerRingSh->DefineSection(0, 0., sOBWheelFlangeR[1], sOBWheelSuppRmax[1]);
    outerRingSh->DefineSection(1, sOBWheelThickness, sOBWheelFlangeR[1], sOBWheelSuppRmax[1]);
    rmax = sOBWheelSuppRmin[1] + sOBWheelThickness;
    outerRingSh->DefineSection(2, sOBWheelThickness, sOBWheelFlangeR[1], rmax);
    outerRingSh->DefineSection(3, 2 * sOBWheelThickness, sOBWheelFlangeR[1], rmax);
  }

  // The lower ring
  if (iLay == 0) {
    zlen = sOBWheelSuppZlen[iLay] - sOBWheelFlangeZlen - 2 * sOBWheelThickness;
  } else {
    zlen = sOBWheelSuppZlen[iLay] - sOBWheelThickness - outerRingSh->GetZ(nsect - 1);
  }

  rmax = sOBWheelSuppRmin[iLay] + sOBWheelThickness;
  TGeoTube* lowerRingSh = new TGeoTube(sOBWheelSuppRmin[iLay], rmax, zlen / 2);

  // The upper ring
  if (iLay == 1) { // For odd layers the upper and lower rings length is the same
    zlen = sOBWheelSuppZlen[iLay] - 2 * sOBWheelThickness;
  }

  rmin = sOBWheelSuppRmax[iLay] - sOBWheelThickness;
  TGeoTube* upperRingSh = new TGeoTube(rmin, sOBWheelSuppRmax[iLay], zlen / 2);

  // The shelf support
  xlen = sOBWheelShelfWide[iLay];
  ylen = 2 * sOBWheelThickness;
  zlen = sOBWheelTotZlen[iLay] - sOBWheelSuppZlen[iLay];
  TGeoBBox* shelfSh = new TGeoBBox(xlen / 2, ylen / 2, zlen / 2);

  // We have all shapes: now create the real volumes
  TGeoMedium* medCarbon = mgr->GetMedium(Form("%s_M55J6K$", GetDetName())); // TO BE CHECKED

  Int_t nLay = iLay + sNumberInnerLayers;

  TGeoVolume* ringInnerVol = new TGeoVolume(Form("OBEndWheelCInnerRing%d", nLay), innerRingSh, medCarbon);
  ringInnerVol->SetFillColor(kBlue);
  ringInnerVol->SetLineColor(kBlue);

  TGeoVolume* ringOuterVol = new TGeoVolume(Form("OBEndWheelCOuterRing%d", nLay), outerRingSh, medCarbon);
  ringOuterVol->SetFillColor(kBlue);
  ringOuterVol->SetLineColor(kBlue);

  TGeoVolume* ringLowerVol = new TGeoVolume(Form("OBEndWheelCLowerRing%d", nLay), lowerRingSh, medCarbon);
  ringLowerVol->SetFillColor(kBlue);
  ringLowerVol->SetLineColor(kBlue);

  TGeoVolume* ringUpperVol = new TGeoVolume(Form("OBEndWheelCUpperRing%d", nLay), upperRingSh, medCarbon);
  ringUpperVol->SetFillColor(kBlue);
  ringUpperVol->SetLineColor(kBlue);

  TGeoVolume* shelfVol = new TGeoVolume(Form("OBEndWheelCShelf%d", nLay), shelfSh, medCarbon);
  shelfVol->SetFillColor(kBlue);
  shelfVol->SetLineColor(kBlue);

  // Finally put everything in the mother volume
  // In blueprints the Z position is given wrt the shelf holes
  // First the ring
  zpos = sMBWheelsZpos + sOBWheelShelfHoleZpos[iLay] - sOBWheelSuppZlen[iLay];

  zpos += innerRingSh->GetDz();
  mother->AddNode(ringInnerVol, 1, new TGeoTranslation(0, 0, -zpos));

  zpos += (innerRingSh->GetDz() + upperRingSh->GetDz());
  mother->AddNode(ringUpperVol, 1, new TGeoTranslation(0, 0, -zpos));

  zpos += (-upperRingSh->GetDz() + lowerRingSh->GetDz());
  mother->AddNode(ringLowerVol, 1, new TGeoTranslation(0, 0, -zpos));

  zpos += (lowerRingSh->GetDz() + outerRingSh->GetZ(nsect - 1));
  mother->AddNode(ringOuterVol, 1, new TGeoTranslation(0, 0, -zpos));

  // Then the support blocks
  Int_t numberOfStaves = GeometryTGeo::Instance()->getNumberOfStaves(nLay);
  Double_t alpha = 360. / numberOfStaves;

  rmin = sOBWheelShelfRpos[iLay] + shelfSh->GetDY();
  zpos = sMBWheelsZpos + sOBWheelShelfHoleZpos[iLay] - sOBWheelSuppZlen[iLay];
  zpos -= shelfSh->GetDZ();

  for (Int_t j = 0; j < numberOfStaves; j++) { // As in V3Layer::createLayer
    Double_t phi = j * alpha + sOBWheelShelfPhi0[iLay];
    xpos = rmin * cosD(phi);
    ypos = rmin * sinD(phi);
    phi += 90;
    mother->AddNode(shelfVol, j, new TGeoCombiTrans(xpos, ypos, -zpos, new TGeoRotation("", phi, 0, 0)));
  }
}

void V3Services::obEndWheelSideC(const Int_t iLay, TGeoVolume* mother, const TGeoManager* mgr)
{
  //
  // Creates the single End Wheel on Side C
  // for a given layer of the Outer Barrel
  // (wheels on Side C are very different for Middle and Outer Barrels,
  // so we cannot use a single method for both as for Side A)
  // (Layer 5: ALICE-W5-04-Side_C, Layer 6: ALICE-W6-04-Side_C)
  //
  // Input:
  //         iLay : the layer number (0,1: Middle, 2,3: Outer)
  //         endWheel : the volume where to place the current created wheel
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      07 Oct 2019  Mario Sitta
  //

  // The support ring
  static const Double_t sOBWheelTotZlen[2] = {37.0 * sMm, 35.0 * sMm};

  static const Double_t sOBWheelSuppRmin = 354.0 * sMm;
  static const Double_t sOBWheelSuppRmax[2] = {389.5 * sMm, 448.5 * sMm};
  static const Double_t sOBWheelIntFlangeR[2] = {335.0 * sMm, 393.0 * sMm};
  static const Double_t sOBWheelExtFlangeR = 409.0 * sMm;
  static const Double_t sOBWheelIntFlangeZ = 4.0 * sMm; // TO BE CHECKED!

  static const Double_t sOBWheelShelfRpos[2] = {361.0 * sMm, 410.5 * sMm};
  static const Double_t sOBWheelShelfHoleZpos[2] = {28.0 * sMm, 26.0 * sMm};

  // Local variables
  Double_t xlen, ylen, zlen;
  Double_t rmin, rmax, phimin, dphi;
  Double_t xpos, ypos, zpos;
  Int_t nsect;

  // The Support Wheels are physically a single piece, a hollow ring
  // with one or two flanges
  // For the sake of simplicity we build it up with a TGeoTube for the
  // external wall and a TGeoPcon for the remaining of the ring for layer 6
  // and with two TGeoPcon's for layer 5

  // The upper ring: a Pcon for Layer 5, a Tube for Layer 6
  TGeoShape* upperRingSh;

  rmin = sOBWheelSuppRmax[iLay] - sOBWheelThickness;
  if (iLay == 0) {
    nsect = 4;
    TGeoPcon* ring = new TGeoPcon(0, 360, nsect);
    ring->DefineSection(0, sOBWheelThickness, rmin, sOBWheelExtFlangeR);
    ring->DefineSection(1, 2 * sOBWheelThickness, rmin, sOBWheelExtFlangeR);
    ring->DefineSection(2, 2 * sOBWheelThickness, rmin, sOBWheelSuppRmax[iLay]);
    zlen = sOBWheelTotZlen[iLay] - sOBWheelThickness;
    ring->DefineSection(3, zlen, rmin, sOBWheelSuppRmax[iLay]);
    upperRingSh = (TGeoShape*)ring;
  } else {
    zlen = sOBWheelTotZlen[iLay] - 2 * sOBWheelThickness;
    TGeoTube* ring = new TGeoTube(rmin, sOBWheelSuppRmax[iLay], zlen / 2);
    upperRingSh = (TGeoShape*)ring;
  }

  // The lower ring: a Pcon
  TGeoPcon* lowerRingSh;

  if (iLay == 0) {
    nsect = 14;
    lowerRingSh = new TGeoPcon(0, 360, nsect);
    lowerRingSh->DefineSection(0, 0., sOBWheelSuppRmin, sOBWheelExtFlangeR);
    lowerRingSh->DefineSection(1, sOBWheelThickness, sOBWheelSuppRmin, sOBWheelExtFlangeR);
    rmax = sOBWheelSuppRmin + sOBWheelThickness;
    lowerRingSh->DefineSection(2, sOBWheelThickness, sOBWheelSuppRmin, rmax);
    lowerRingSh->DefineSection(3, sOBWheelIntFlangeZ, sOBWheelSuppRmin, rmax);
    lowerRingSh->DefineSection(4, sOBWheelIntFlangeZ, sOBWheelIntFlangeR[iLay], rmax);
    zpos = sOBWheelIntFlangeZ + 2 * sOBWheelThickness;
    lowerRingSh->DefineSection(5, zpos, sOBWheelIntFlangeR[iLay], rmax);
    lowerRingSh->DefineSection(6, zpos, sOBWheelSuppRmin, rmax);
    zpos += sOBWheelIntFlangeZ;
    lowerRingSh->DefineSection(7, zpos, sOBWheelSuppRmin, rmax);
    rmax = sOBWheelShelfRpos[iLay] + sOBWheelThickness;
    lowerRingSh->DefineSection(8, zpos, sOBWheelSuppRmin, rmax);
    zpos += sOBWheelThickness;
    lowerRingSh->DefineSection(9, zpos, sOBWheelSuppRmin, rmax);
    lowerRingSh->DefineSection(10, zpos, sOBWheelShelfRpos[iLay], rmax);
    zpos = sOBWheelTotZlen[iLay] - sOBWheelThickness;
    lowerRingSh->DefineSection(11, zpos, sOBWheelShelfRpos[iLay], rmax);
    lowerRingSh->DefineSection(12, zpos, sOBWheelShelfRpos[iLay], sOBWheelSuppRmax[iLay]);
    lowerRingSh->DefineSection(13, sOBWheelTotZlen[iLay], sOBWheelShelfRpos[iLay], sOBWheelSuppRmax[iLay]);
  } else {
    nsect = 10;
    lowerRingSh = new TGeoPcon(0, 360, nsect);
    lowerRingSh->DefineSection(0, 0., sOBWheelShelfRpos[iLay], sOBWheelSuppRmax[iLay]);
    lowerRingSh->DefineSection(1, sOBWheelThickness, sOBWheelShelfRpos[iLay], sOBWheelSuppRmax[iLay]);
    rmax = sOBWheelShelfRpos[iLay] + sOBWheelThickness;
    lowerRingSh->DefineSection(2, sOBWheelThickness, sOBWheelShelfRpos[iLay], rmax);
    lowerRingSh->DefineSection(3, sOBWheelIntFlangeZ, sOBWheelShelfRpos[iLay], rmax);
    lowerRingSh->DefineSection(4, sOBWheelIntFlangeZ, sOBWheelIntFlangeR[iLay], rmax);
    zpos = sOBWheelIntFlangeZ + 2 * sOBWheelThickness;
    lowerRingSh->DefineSection(5, zpos, sOBWheelIntFlangeR[iLay], rmax);
    lowerRingSh->DefineSection(6, zpos, sOBWheelShelfRpos[iLay], rmax);
    zpos = sOBWheelTotZlen[iLay] - sOBWheelThickness;
    lowerRingSh->DefineSection(7, zpos, sOBWheelShelfRpos[iLay], rmax);
    lowerRingSh->DefineSection(8, zpos, sOBWheelShelfRpos[iLay], sOBWheelSuppRmax[iLay]);
    lowerRingSh->DefineSection(9, sOBWheelTotZlen[iLay], sOBWheelShelfRpos[iLay], sOBWheelSuppRmax[iLay]);
  }

  // We have all shapes: now create the real volumes
  TGeoMedium* medCarbon = mgr->GetMedium(Form("%s_M55J6K$", GetDetName())); // TO BE CHECKED

  Int_t nLay = iLay + sNumberInnerLayers + sNumberMiddlLayers;

  TGeoVolume* ringUpperVol = new TGeoVolume(Form("OBEndWheelCUpperRing%d", nLay), upperRingSh, medCarbon);
  ringUpperVol->SetFillColor(kBlue);
  ringUpperVol->SetLineColor(kBlue);

  TGeoVolume* ringLowerVol = new TGeoVolume(Form("OBEndWheelCLowerRing%d", nLay), lowerRingSh, medCarbon);
  ringLowerVol->SetFillColor(kBlue);
  ringLowerVol->SetLineColor(kBlue);

  // Finally put everything in the mother volume
  // In blueprints the Z position is given wrt the shelf holes
  zpos = sOBWheelsZpos + sOBWheelShelfHoleZpos[iLay];

  mother->AddNode(ringLowerVol, 1, new TGeoTranslation(0, 0, -zpos));

  if (iLay == 1) {
    zpos -= (sOBWheelThickness + (static_cast<TGeoTube*>(upperRingSh))->GetDz());
  }
  mother->AddNode(ringUpperVol, 1, new TGeoTranslation(0, 0, -zpos));
}

void V3Services::obConeSideA(TGeoVolume* mother, const TGeoManager* mgr)
{
  //
  // Creates the Cone structure on Side A of the Outer Barrel
  // (ALICE-W4-04-Cone_4A)
  //
  // Input:
  //         mother : the volume where to place the current created cone
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      03 Feb 2020  Mario Sitta
  //

  static const Double_t sOBConeATotZlen = 350.0 * sMm;
  static const Double_t sOBConeAStartCyl2 = 170.0 * sMm;
  static const Double_t sOBConeAEndCyl1 = 160.8 * sMm;
  static const Double_t sOBConeAThinCylZ = 36.0 * sMm;

  static const Double_t sOBConeAIntR = 291.5 * sMm;
  static const Double_t sOBConeAExtR = 302.5 * sMm;

  static const Double_t sOBConeARingExtR = 339.5 * sMm;
  static const Double_t sOBConeARingZlen = 55.0 * sMm;
  static const Double_t sOBConeARingZout = 35.0 * sMm;

  static const Double_t sOBConeAThickAll = 2.0 * sMm;
  static const Double_t sOBConeAThickThin = 1.0 * sMm;

  static const Double_t sOBConeAReinfZIn = 1.0 * sMm;
  static const Double_t sOBConeAReinfRIn = 301.6 * sMm;

  static const Double_t sOBConeAReinfThick = 6.5 * sMm;

  static const Int_t sOBConeARibNVert = 8;

  // Local variables
  Double_t rmin, rmax, zlen;
  Double_t xpos, ypos, zpos;

  // The OB Cone on Side A is physically a single piece.
  // It is implemented using two Pcon plus a Xtru for the reinforcements

  Double_t phi = sOBConeAReinfThick / sOBConeAIntR;
  phi *= TMath::RadToDeg();

  // The main cone: a Pcon
  TGeoPcon* obConeSh = new TGeoPcon(phi, 180 - 2 * phi, 10);

  rmin = sOBConeAIntR;
  rmax = sOBConeAReinfRIn;
  obConeSh->DefineSection(0, 0., rmin, rmax);
  obConeSh->DefineSection(1, sOBConeAReinfZIn, rmin, rmax);
  rmax = rmin + sOBConeAThickThin;
  obConeSh->DefineSection(2, sOBConeAReinfZIn, rmin, rmax);
  obConeSh->DefineSection(3, sOBConeAThinCylZ, rmin, rmax);
  rmax = rmin + sOBConeAThickAll;
  obConeSh->DefineSection(4, sOBConeAThinCylZ, rmin, rmax);
  zlen = sOBConeATotZlen - sOBConeAStartCyl2;
  obConeSh->DefineSection(5, zlen, rmin, rmax);
  rmin = sOBConeAExtR;
  rmax = rmin + sOBConeAThickAll;
  zlen = sOBConeATotZlen - sOBConeAEndCyl1;
  obConeSh->DefineSection(6, zlen, rmin, rmax);
  zlen = sOBConeATotZlen - sOBConeAThickAll;
  obConeSh->DefineSection(7, zlen, rmin, rmax);
  rmax = sOBConeARingExtR;
  obConeSh->DefineSection(8, zlen, rmin, rmax);
  obConeSh->DefineSection(9, sOBConeATotZlen, rmin, rmax);

  // The external ring: a Pcon
  TGeoPcon* obConeRingSh = new TGeoPcon(phi, 180 - 2 * phi, 6);

  rmin = obConeSh->GetRmax(7);
  rmax = rmin + sOBConeAThickAll;
  obConeRingSh->DefineSection(0, 0., rmin, rmax);
  zlen = sOBConeARingZlen - sOBConeARingZout;
  obConeRingSh->DefineSection(1, zlen, rmin, rmax);
  rmax = sOBConeARingExtR;
  obConeRingSh->DefineSection(2, zlen, rmin, rmax);
  zlen += sOBConeAThickAll;
  obConeRingSh->DefineSection(3, zlen, rmin, rmax);
  rmin = rmax - sOBConeAThickAll;
  obConeRingSh->DefineSection(4, zlen, rmin, rmax);
  zlen = sOBConeARingZlen - sOBConeAThickAll;
  obConeRingSh->DefineSection(5, zlen, rmin, rmax);

  // The reinforcement rib: a Xtru
  Double_t xr[sOBConeARibNVert], yr[sOBConeARibNVert];

  xr[0] = 0;
  yr[0] = 0;
  xr[1] = obConeSh->GetRmax(0) - obConeSh->GetRmin(0);
  yr[1] = yr[0];
  xr[2] = xr[1];
  yr[2] = obConeSh->GetZ(5);
  xr[7] = xr[0];
  yr[7] = yr[2];
  xr[6] = obConeSh->GetRmin(6) - obConeSh->GetRmin(5);
  yr[6] = obConeSh->GetZ(6);
  xr[3] = xr[6] + (xr[1] - xr[0]);
  yr[3] = yr[6];
  xr[5] = xr[6];
  yr[5] = sOBConeATotZlen - sOBConeARingZout + sOBConeAThickAll;
  xr[4] = xr[3];
  yr[4] = yr[5];

  TGeoXtru* obConeRibSh = new TGeoXtru(2);
  obConeRibSh->DefinePolygon(sOBConeARibNVert, xr, yr);
  obConeRibSh->DefineSection(0, 0);
  obConeRibSh->DefineSection(1, sOBConeAThickAll);

  // We have all shapes: now create the real volumes
  TGeoMedium* medCarbon = mgr->GetMedium(Form("%s_M55J6K$", GetDetName())); // TO BE CHECKED

  TGeoVolume* obConeVol = new TGeoVolume("OBConeSideA", obConeSh, medCarbon);
  obConeVol->SetFillColor(kBlue);
  obConeVol->SetLineColor(kBlue);

  TGeoVolume* obConeRingVol = new TGeoVolume("OBConeRingSideA", obConeRingSh, medCarbon);
  obConeRingVol->SetFillColor(kBlue);
  obConeRingVol->SetLineColor(kBlue);

  TGeoVolume* obConeRibVol = new TGeoVolume("OBConeRibSideA", obConeRibSh, medCarbon);
  obConeRibVol->SetFillColor(kBlue);
  obConeRibVol->SetLineColor(kBlue);

  // Finally put everything in the mother volume
  zpos = sOBConesZpos - sOBConeATotZlen;

  mother->AddNode(obConeVol, 1, new TGeoTranslation(0, 0, zpos));
  mother->AddNode(obConeVol, 2, new TGeoCombiTrans(0, 0, zpos, new TGeoRotation("", 180, 0, 0)));

  zpos = sOBConesZpos - sOBConeARingZlen;

  mother->AddNode(obConeRingVol, 1, new TGeoTranslation(0, 0, zpos));
  mother->AddNode(obConeRingVol, 2, new TGeoCombiTrans(0, 0, zpos, new TGeoRotation("", 180, 0, 0)));

  xpos = obConeSh->GetRmin(0);
  ypos = sOBConeAReinfThick;
  zpos = sOBConesZpos - sOBConeATotZlen;

  mother->AddNode(obConeRibVol, 1, new TGeoCombiTrans(xpos, ypos, zpos, new TGeoRotation("", 0, 90, 0)));
  mother->AddNode(obConeRibVol, 4, new TGeoCombiTrans(-xpos, -ypos, zpos, new TGeoRotation("", 0, -90, 180)));

  ypos = sOBConeAReinfThick - sOBConeAThickAll;

  mother->AddNode(obConeRibVol, 3, new TGeoCombiTrans(-xpos, ypos, zpos, new TGeoRotation("", 0, -90, -180)));
  mother->AddNode(obConeRibVol, 4, new TGeoCombiTrans(xpos, -ypos, zpos, new TGeoRotation("", 0, 90, 0)));
}

void V3Services::obConeTraysSideA(TGeoVolume* mother, const TGeoManager* mgr)
{
  //
  // Creates the Cone Trays on Side A of the Outer Barrel
  // (ALICE-W3-08-vassoio+ALICE-W5-08_vassoio)
  //
  // Input:
  //         mother : the volume where to place the current created cone
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      05 Feb 2020  Mario Sitta
  //

  static const Double_t sOBTrayZlen[2] = {112.0 * sMm, 115.0 * sMm};
  static const Double_t sOBTrayRmin[2] = {222.0 * sMm, 370.0 * sMm};
  static const Double_t sOBTrayRmax[2] = {240.0 * sMm, 386.0 * sMm};

  static const Double_t sOBTrayZpos[2] = {181.0 * sMm, 20.0 * sMm};

  static const Double_t sOBTrayThick = 2.0 * sMm;

  static const Double_t sOBTrayReinfWide[2] = {27.0 * sMm, 24.0 * sMm};
  static const Double_t sOBTrayReinfYpos = 6.0 * sMm;

  // Local variables
  Double_t rmin, rmax, zlen;
  Double_t xpos, ypos, zpos;

  // Each OB Tray on Side A is physically a single piece.
  // It is implemented using a Pcon plus a BBox for the reinforcements

  TGeoPcon* obTraySh[2];
  TGeoBBox* obTrayRibSh[2];

  for (Int_t j = 0; j < 2; j++) {
    Double_t phi = (sOBTrayReinfYpos + sOBTrayThick) / sOBTrayRmin[j];
    phi *= TMath::RadToDeg();

    // The main body: a Pcon
    obTraySh[j] = new TGeoPcon(180 + phi, 180 - 2 * phi, 4);

    rmin = sOBTrayRmin[j];
    rmax = sOBTrayRmax[j];
    obTraySh[j]->DefineSection(0, 0., rmin, rmax);
    obTraySh[j]->DefineSection(1, sOBTrayThick, rmin, rmax);
    rmin = rmax - sOBTrayThick;
    obTraySh[j]->DefineSection(2, sOBTrayThick, rmin, rmax);
    obTraySh[j]->DefineSection(3, sOBTrayZlen[j], rmin, rmax);

    // The reinforcement rib: a BBox
    obTrayRibSh[j] = new TGeoBBox(sOBTrayReinfWide[j] / 2,
                                  sOBTrayThick / 2,
                                  sOBTrayZlen[j] / 2);
  } // for (j = 0,1)

  // We have all shapes: now create the real volumes
  TGeoMedium* medCarbon = mgr->GetMedium(Form("%s_M55J6K$", GetDetName())); // TO BE CHECKED

  TGeoVolume *obTrayVol[2], *obTrayRibVol[2];

  for (Int_t j = 0; j < 2; j++) {
    obTrayVol[j] = new TGeoVolume(Form("OBConeTray%d", j), obTraySh[j], medCarbon);
    obTrayVol[j]->SetFillColor(kBlue);
    obTrayVol[j]->SetLineColor(kBlue);

    obTrayRibVol[j] = new TGeoVolume(Form("OBConeTrayRib%d", j), obTrayRibSh[j], medCarbon);
    obTrayRibVol[j]->SetFillColor(kBlue);
    obTrayRibVol[j]->SetLineColor(kBlue);
  }

  // Finally put everything in the mother volume

  for (Int_t j = 0; j < 2; j++) {
    if (j == 0) {
      zpos = sOBConesZpos - sOBTrayZpos[j] - sOBTrayZlen[j];
    } else {
      zpos = sOBConesZpos + sOBTrayZpos[j];
    }

    mother->AddNode(obTrayVol[j], 1, new TGeoTranslation(0, 0, zpos));
    mother->AddNode(obTrayVol[j], 2, new TGeoCombiTrans(0, 0, zpos, new TGeoRotation("", 180, 0, 0)));

    xpos = obTraySh[j]->GetRmin(0) + obTrayRibSh[j]->GetDX();
    ypos = sOBTrayReinfYpos + obTrayRibSh[j]->GetDY();
    zpos += obTrayRibSh[j]->GetDZ();

    mother->AddNode(obTrayRibVol[j], 1, new TGeoTranslation(xpos, -ypos, zpos));
    mother->AddNode(obTrayRibVol[j], 2, new TGeoTranslation(-xpos, -ypos, zpos));
    mother->AddNode(obTrayRibVol[j], 3, new TGeoTranslation(xpos, ypos, zpos));
    mother->AddNode(obTrayRibVol[j], 4, new TGeoTranslation(-xpos, ypos, zpos));
  }
}

void V3Services::obConeSideC(TGeoVolume* mother, const TGeoManager* mgr)
{
  //
  // Creates the Cone structure on Side C of the Outer Barrel
  // (ALICE-W4-06-Cone_4C)
  //
  // Input:
  //         mother : the volume where to place the current created cone
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      26 Jan 2020  Mario Sitta
  //

  static const Double_t sOBConeCTotZlen = 332.5 * sMm;
  static const Double_t sOBConeCStartCyl2 = 132.8 * sMm;
  static const Double_t sOBConeCEndCyl1 = 82.4 * sMm;
  static const Double_t sOBConeCThinCylZ = 36.0 * sMm;

  static const Double_t sOBConeCIntR = 291.5 * sMm;
  static const Double_t sOBConeCExtR = 315.0 * sMm;

  static const Double_t sOBConeCRingExtR = 333.0 * sMm;
  static const Double_t sOBConeCRingZlen = 61.0 * sMm;
  static const Double_t sOBConeCRingZout = 42.0 * sMm;

  static const Double_t sOBConeCThickAll = 2.0 * sMm;
  static const Double_t sOBConeCThickThin = 1.0 * sMm;

  static const Double_t sOBConeCReinfZIn = 2.0 * sMm;
  static const Double_t sOBConeCReinfRIn = 301.6 * sMm;
  static const Double_t sOBConeCReinfROut = 351.5 * sMm;

  static const Double_t sOBConeCReinfThick = 6.5 * sMm;

  static const Int_t sOBConeCRibNVert = 8;

  // Local variables
  Double_t rmin, rmax, zlen;
  Double_t xpos, ypos, zpos;

  // The OB Cone on Side C is physically a single piece.
  // It is implemented using two Pcon plus a Xtru for the reinforcements

  Double_t phi = sOBConeCReinfThick / sOBConeCIntR;
  phi *= TMath::RadToDeg();

  // The main cone: a Pcon
  TGeoPcon* obConeSh = new TGeoPcon(phi, 180 - 2 * phi, 10);

  rmin = sOBConeCExtR;
  rmax = sOBConeCReinfROut;
  obConeSh->DefineSection(0, 0., rmin, rmax);
  obConeSh->DefineSection(1, sOBConeCThickAll, rmin, rmax);
  rmax = rmin + sOBConeCThickAll;
  obConeSh->DefineSection(2, sOBConeCThickAll, rmin, rmax);
  obConeSh->DefineSection(3, sOBConeCEndCyl1, rmin, rmax);
  rmin = sOBConeCIntR;
  rmax = rmin + sOBConeCThickAll;
  obConeSh->DefineSection(4, sOBConeCStartCyl2, rmin, rmax);
  zlen = sOBConeCTotZlen - sOBConeCThinCylZ;
  obConeSh->DefineSection(5, zlen, rmin, rmax);
  rmax = rmin + sOBConeCThickThin;
  obConeSh->DefineSection(6, zlen, rmin, rmax);
  zlen = sOBConeCTotZlen - sOBConeCReinfZIn;
  obConeSh->DefineSection(7, zlen, rmin, rmax);
  rmax = sOBConeCReinfRIn;
  obConeSh->DefineSection(8, zlen, rmin, rmax);
  obConeSh->DefineSection(9, sOBConeCTotZlen, rmin, rmax);

  // The external ring: a Pcon
  TGeoPcon* obConeRingSh = new TGeoPcon(phi, 180 - 2 * phi, 8);

  rmin = sOBConeCRingExtR - sOBConeCThickAll;
  rmax = sOBConeCReinfROut;
  obConeRingSh->DefineSection(0, 0., rmin, rmax);
  obConeRingSh->DefineSection(1, sOBConeCThickAll, rmin, rmax);
  rmax = sOBConeCRingExtR;
  obConeRingSh->DefineSection(2, sOBConeCThickAll, rmin, rmax);
  zlen = sOBConeCRingZout - sOBConeCThickAll;
  obConeRingSh->DefineSection(3, zlen, rmin, rmax);
  rmin = sOBConeCExtR + sOBConeCThickAll;
  obConeRingSh->DefineSection(4, zlen, rmin, rmax);
  obConeRingSh->DefineSection(5, sOBConeCRingZout, rmin, rmax);
  rmax = rmin + sOBConeCThickAll;
  obConeRingSh->DefineSection(6, sOBConeCRingZout, rmin, rmax);
  obConeRingSh->DefineSection(7, sOBConeCRingZlen, rmin, rmax);

  // The reinforcement rib: a Xtru
  Double_t xr[sOBConeCRibNVert], yr[sOBConeCRibNVert];

  xr[0] = 0;
  yr[0] = 0;
  xr[1] = obConeSh->GetRmax(9) - obConeSh->GetRmin(9);
  yr[1] = yr[0];
  xr[2] = xr[1];
  yr[2] = obConeSh->GetZ(9) - obConeSh->GetZ(4);
  xr[7] = xr[0];
  yr[7] = yr[2];
  xr[6] = obConeSh->GetRmin(3) - obConeSh->GetRmin(4);
  yr[6] = obConeSh->GetZ(9) - obConeSh->GetZ(3);
  xr[3] = xr[6] + (xr[1] - xr[0]);
  yr[3] = yr[6];
  xr[5] = xr[6];
  yr[5] = sOBConeCTotZlen - sOBConeCRingZout;
  xr[4] = xr[3];
  yr[4] = yr[5];

  TGeoXtru* obConeRibSh = new TGeoXtru(2);
  obConeRibSh->DefinePolygon(sOBConeCRibNVert, xr, yr);
  obConeRibSh->DefineSection(0, 0);
  obConeRibSh->DefineSection(1, sOBConeCThickAll);

  // We have all shapes: now create the real volumes
  TGeoMedium* medCarbon = mgr->GetMedium(Form("%s_M55J6K$", GetDetName())); // TO BE CHECKED

  TGeoVolume* obConeVol = new TGeoVolume("OBConeSideC", obConeSh, medCarbon);
  obConeVol->SetFillColor(kBlue);
  obConeVol->SetLineColor(kBlue);

  TGeoVolume* obConeRingVol = new TGeoVolume("OBConeRingSideC", obConeRingSh, medCarbon);
  obConeRingVol->SetFillColor(kBlue);
  obConeRingVol->SetLineColor(kBlue);

  TGeoVolume* obConeRibVol = new TGeoVolume("OBConeRibSideC", obConeRibSh, medCarbon);
  obConeRibVol->SetFillColor(kBlue);
  obConeRibVol->SetLineColor(kBlue);

  // Finally put everything in the mother volume
  zpos = sOBConesZpos;

  mother->AddNode(obConeVol, 1, new TGeoTranslation(0, 0, -zpos));
  mother->AddNode(obConeVol, 2, new TGeoCombiTrans(0, 0, -zpos, new TGeoRotation("", 180, 0, 0)));

  zpos -= sOBConeCThickAll;

  mother->AddNode(obConeRingVol, 1, new TGeoTranslation(0, 0, -zpos));
  mother->AddNode(obConeRingVol, 2, new TGeoCombiTrans(0, 0, -zpos, new TGeoRotation("", 180, 0, 0)));

  xpos = obConeSh->GetRmin(9);
  ypos = sOBConeCReinfThick;
  zpos = sOBConesZpos - obConeSh->GetZ(9);

  mother->AddNode(obConeRibVol, 1, new TGeoCombiTrans(xpos, -ypos, -zpos, new TGeoRotation("", 0, -90, 0)));
  mother->AddNode(obConeRibVol, 2, new TGeoCombiTrans(-xpos, ypos, -zpos, new TGeoRotation("", 0, 90, 180)));

  ypos = sOBConeCReinfThick - sOBConeCThickAll;

  mother->AddNode(obConeRibVol, 3, new TGeoCombiTrans(xpos, ypos, -zpos, new TGeoRotation("", 0, -90, 0)));
  mother->AddNode(obConeRibVol, 4, new TGeoCombiTrans(-xpos, -ypos, -zpos, new TGeoRotation("", 0, 90, 180)));
}

void V3Services::obCYSS11(TGeoVolume* mother, const TGeoManager* mgr)
{
  //
  // Creates the CYSS element 11 with all its sub-elements
  // OB CYSS 11 : ALIITSUP0564 (the whole assembly)
  // OB CYSS 12 : ALIITSUP0565 (the outermost cylinder)
  // OB CYSS 13 : ALIITSUP0566 (the intermediate cylinder)
  // OB CYSS 14 : ALIITSUP0567 (the innermost cylinder)
  //
  // Input:
  //         mother : the volume where to place the current created cylinder
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      02 Mar 2020  Mario Sitta
  // Updated:      30 Nov 2022  Mario Sitta  Fix materials and thicknesses
  //

  static const Double_t sOBCYSS14Zlen = 1556.8 * sMm;
  static const Double_t sOBCYSS14DInt = 898.0 * sMm;
  static const Double_t sOBCYSS14DExt = 902.0 * sMm;
  static const Double_t sOBCYSS14PhiCut = 4.0 * sMm;

  static const Double_t sOBCYSS13Zlen = 1481.0 * sMm;
  static const Double_t sOBCYSS13DInt = sOBCYSS14DExt;
  static const Double_t sOBCYSS13DExt = 918.0 * sMm;
  static const Double_t sOBCYSS13PhiCut = 10.55 * sMm;

  static const Double_t sOBCYSS12Zlen = 1520.6 * sMm;
  static const Double_t sOBCYSS12DInt = sOBCYSS13DExt;
  static const Double_t sOBCYSS12DExt = 922.0 * sMm;
  static const Double_t sOBCYSS12PhiCut = 4.0 * sMm;

  static const Double_t sOBCYSS20Zlen = 1500.6 * sMm;
  static const Double_t sOBCYSS20Width = 7.1 * sMm;
  static const Double_t sOBCYSS20Height = 6.35 * sMm;

  // Local variables
  Double_t rmin, rmax, phi;
  Double_t xpos, ypos, zpos;

  // The OB CYSS is made by three cylinders plus other elements
  // (rings, bars, etc) fixed together

  // The three cylinders
  rmin = sOBCYSS14DInt / 2;
  rmax = sOBCYSS14DExt / 2;

  phi = sOBCYSS14PhiCut / rmin;
  phi *= TMath::RadToDeg();

  TGeoTubeSeg* obCyss14Sh = new TGeoTubeSeg(rmin, rmax, sOBCYSS14Zlen / 2, phi, 180. - phi);

  //
  rmin = sOBCYSS13DInt / 2;
  rmax = sOBCYSS13DExt / 2;

  phi = sOBCYSS13PhiCut / rmin;
  phi *= TMath::RadToDeg();

  TGeoTubeSeg* obCyss13Sh = new TGeoTubeSeg(rmin, rmax, sOBCYSS13Zlen / 2, phi, 180. - phi);

  //
  rmin = sOBCYSS12DInt / 2;
  rmax = sOBCYSS12DExt / 2;

  phi = sOBCYSS12PhiCut / rmin;
  phi *= TMath::RadToDeg();

  TGeoTubeSeg* obCyss12Sh = new TGeoTubeSeg(rmin, rmax, sOBCYSS12Zlen / 2, phi, 180. - phi);

  // The middle bar
  TGeoBBox* obCyss20Sh = new TGeoBBox(sOBCYSS20Width / 2, sOBCYSS20Height / 2, sOBCYSS20Zlen / 2);

  // We have all shapes: now create the real volumes
  TGeoMedium* medRist = mgr->GetMedium(Form("%s_RIST110$", GetDetName()));
  TGeoMedium* medCarbon = mgr->GetMedium(Form("%s_AS4C200$", GetDetName()));

  TGeoVolume* obCyss14Vol = new TGeoVolume("OBCYSS14", obCyss14Sh, medCarbon);
  obCyss14Vol->SetFillColor(kBlue);
  obCyss14Vol->SetLineColor(kBlue);

  TGeoVolume* obCyss13Vol = new TGeoVolume("OBCYSS13", obCyss13Sh, medRist);
  obCyss13Vol->SetFillColor(kBlue);
  obCyss13Vol->SetLineColor(kBlue);

  TGeoVolume* obCyss12Vol = new TGeoVolume("OBCYSS12", obCyss12Sh, medCarbon);
  obCyss12Vol->SetFillColor(kBlue);
  obCyss12Vol->SetLineColor(kBlue);

  TGeoVolume* obCyss20Vol = new TGeoVolume("OBCYSS20", obCyss20Sh, medCarbon);
  obCyss20Vol->SetFillColor(kYellow);
  obCyss20Vol->SetLineColor(kYellow);

  // Finally put everything in the mother volume
  mother->AddNode(obCyss14Vol, 1, nullptr);
  mother->AddNode(obCyss14Vol, 2, new TGeoRotation("", 180, 0, 0));

  mother->AddNode(obCyss13Vol, 1, nullptr);
  mother->AddNode(obCyss13Vol, 2, new TGeoRotation("", 180, 0, 0));

  mother->AddNode(obCyss12Vol, 1, nullptr);
  mother->AddNode(obCyss12Vol, 2, new TGeoRotation("", 180, 0, 0));

  xpos = (obCyss13Sh->GetRmin() + obCyss13Sh->GetRmax()) / 2;
  ypos = sOBCYSS13PhiCut - obCyss20Sh->GetDY();
  mother->AddNode(obCyss20Vol, 1, new TGeoTranslation(xpos, ypos, 0));
  mother->AddNode(obCyss20Vol, 2, new TGeoTranslation(-xpos, ypos, 0));
  mother->AddNode(obCyss20Vol, 3, new TGeoTranslation(xpos, -ypos, 0));
  mother->AddNode(obCyss20Vol, 4, new TGeoTranslation(-xpos, -ypos, 0));
}

void V3Services::ibConvWire(TGeoVolume* mother, const TGeoManager* mgr)
{
  //
  // Creates the 1mm wire for Gamma Conversion studies and its supports
  // Blueprints (from EDMS) : ALIITSUP0949 (the whole assembly)
  //                          ALIITSUP0469 (the actual wire)
  //                          ALIITSUP0466 (each tension screw)
  //                          ALIITSUP0918 (the inner support plate)
  //                          ALIITSUP0914 (the outer support plate)
  //
  // Input:
  //         mother : the volume where to place the current created cylinder
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      28 Sep 2022  Mario Sitta
  //

  // The wire itself
  static const Double_t sIBGWireLength = 492.0 * sMm;
  static const Double_t sIBGWireDiam = 1.0 * sMm;

  static const Double_t sIBGWireXPosIn = 55.1 * sMm;
  static const Double_t sIBGWireXPosOut = 148.6 * sMm;
  static const Double_t sIBGWireYPos = 14.0 * sMm;
  static const Double_t sIBGWireZPos = 68.25 * sMm;
  static const Double_t sIBGWirePhiPos = 10.9;  // Deg
  static const Double_t sIBGWireThetaPos = 8.7; // Deg

  // The inner wire support
  static const Double_t sIBGWireIntSuppBaseCentWid = 11.54 * sMm;
  static const Double_t sIBGWireIntSuppBaseCentHi = 11.3 * sMm;
  static const Double_t sIBGWireIntSuppBaseCentThik = 3.9 * sMm;

  static const Double_t sIBGWireIntSuppZpos = 171.5 * sMm;
  static const Double_t sIBGWireIntSuppXDist = 116.2 * sMm;

  // The outer wire support
  static const Double_t sIBGWireOutSuppLength = 26.5 * sMm;
  static const Double_t sIBGWireOutSuppThick = 3.0 * sMm;

  static const Double_t sIBGWireOutSuppHoleXpos = 16.0 * sMm;
  static const Double_t sIBGWireOutSuppXDist = 292.0 * sMm;
  static const Double_t sIBGWireOutSuppYpos = 1.0 * sMm;
  static const Double_t sIBGWireOutSuppZpos = 312.9 * sMm;

  // Local variables
  Double_t xpos, ypos, zpos;

  // The wire: a Tube
  TGeoTube* ibWireSh = new TGeoTube(0, sIBGWireDiam / 2, sIBGWireLength / 2);

  // Create the support shapes
  TGeoCompositeShape* ibWireIntSuppLeftSh = ibConvWireIntSupport(kTRUE);
  TGeoCompositeShape* ibWireIntSuppRightSh = ibConvWireIntSupport(kFALSE);
  TGeoCompositeShape* ibWireOutSuppSh = ibConvWireOutSupport();

  // We have all shapes: now create the real volumes
  TGeoMedium* medAl = mgr->GetMedium(Form("%s_ALUMINUM$", GetDetName()));
  TGeoMedium* medTungsten = mgr->GetMedium(Form("%s_TUNGSTEN$", GetDetName()));

  TGeoVolume* ibWireVol = new TGeoVolume("IBGammaConvWire", ibWireSh, medTungsten);
  ibWireVol->SetFillColor(kGray);
  ibWireVol->SetLineColor(kGray);

  TGeoVolume* ibWireIntSuppLeftVol = new TGeoVolume("IBGammaConvWireInnerSupportLeft", ibWireIntSuppLeftSh, medAl);
  ibWireIntSuppLeftVol->SetFillColor(kRed);
  ibWireIntSuppLeftVol->SetLineColor(kRed);

  TGeoVolume* ibWireIntSuppRightVol = new TGeoVolume("IBGammaConvWireInnerSupportRight", ibWireIntSuppRightSh, medAl);
  ibWireIntSuppRightVol->SetFillColor(kRed);
  ibWireIntSuppRightVol->SetLineColor(kRed);

  TGeoVolume* ibWireOutSuppVol = new TGeoVolume("IBGammaConvWireOuterSupport", ibWireOutSuppSh, medAl);
  ibWireOutSuppVol->SetFillColor(kRed);
  ibWireOutSuppVol->SetLineColor(kRed);

  // Finally put everything in the mother volume
  xpos = (sIBGWireXPosIn + sIBGWireXPosOut) / 2;
  ypos = sIBGWireYPos;
  zpos = sIBGWireZPos;
  mother->AddNode(ibWireVol, 1, new TGeoCombiTrans(xpos, -ypos, zpos, new TGeoRotation("", 90 - sIBGWireThetaPos, sIBGWirePhiPos, 0)));
  mother->AddNode(ibWireVol, 2, new TGeoCombiTrans(-xpos, -ypos, zpos, new TGeoRotation("", 90 + sIBGWireThetaPos, -sIBGWirePhiPos, 0)));

  xpos = sIBGWireIntSuppXDist / 2 - sIBGWireIntSuppBaseCentWid / 2;
  ypos = sIBGWireIntSuppBaseCentHi / 2;
  zpos = sIBGWireIntSuppZpos + sIBGWireIntSuppBaseCentThik;
  mother->AddNode(ibWireIntSuppLeftVol, 1, new TGeoTranslation(xpos, -ypos, -zpos));
  mother->AddNode(ibWireIntSuppRightVol, 1, new TGeoTranslation(-xpos, -ypos, -zpos));

  xpos = sIBGWireOutSuppXDist / 2 - sIBGWireOutSuppHoleXpos;
  ypos = -sIBGWireOutSuppLength - sIBGWireOutSuppYpos;
  zpos = sIBGWireOutSuppZpos - sIBGWireOutSuppThick;
  mother->AddNode(ibWireOutSuppVol, 1, new TGeoTranslation(xpos, ypos, zpos));

  zpos = sIBGWireOutSuppZpos;
  mother->AddNode(ibWireOutSuppVol, 2, new TGeoCombiTrans(-xpos, ypos, zpos, new TGeoRotation("", 180, 180, 0)));
}

TGeoCompositeShape* V3Services::ibConvWireIntSupport(const Bool_t left)
{
  //
  // Creates the shape of the internal support of the Inner Barrel Gamma
  // Conversion wire: being pretty complicate, we devote a dedicate method
  // for it; the shape is a bit simplified but the total material budget
  // is preserved
  // The left and right supports are specular (or better, chiral), so we
  // cannot create one volume and place two copies one of them rotated,
  // but we have to create two different (specular) copies
  // Blueprints (from EDMS) : ALIITSUP0918, ALIITSUP0919
  //
  // Input:
  //         left : if true create the left copy, otherwise the right copy
  //
  // Output:
  //
  // Return:
  //         The support shape as a TGeoCompositeShape
  //
  // Created:      01 Oct 2022  Mario Sitta
  //

  // The outer wire support
  static const Double_t sIBGWireIntSuppBaseFullWid = 19.5 * sMm;
  static const Double_t sIBGWireIntSuppBaseFullSpan = 18.5 * sMm;
  static const Double_t sIBGWireIntSuppBaseOutSpan = 2.5 * sMm;
  static const Double_t sIBGWireIntSuppBaseCentWid = 11.94 * sMm;
  static const Double_t sIBGWireIntSuppBaseCentHi = 11.3 * sMm;
  static const Double_t sIBGWireIntSuppBaseCentThik = 3.9 * sMm;
  static const Double_t sIBGWireIntSuppFingerLen = 9.6 * sMm;
  static const Double_t sIBGWireIntSuppFingerWid = 4.8 * sMm;
  static const Double_t sIBGWireIntSuppFingerShift = 1.4 * sMm;
  static const Double_t sIBGWireIntSuppFingerThik = 8.1 * sMm;
  static const Double_t sIBGWireIntSuppFingerPhi = 15.0; // Deg
  static const Double_t sIBGWireIntSuppSpikyWid = 10.0 * sMm;
  static const Double_t sIBGWireIntSuppSpikyHi = 12.6 * sMm;
  static const Double_t sIBGWireIntSuppSpikyXin = 2.5 * sMm;
  static const Double_t sIBGWireIntSuppSpikyYin = 9.2 * sMm;
  static const Double_t sIBGWireIntSuppSpikyThik = 3.0 * sMm;

  // Local variables
  Double_t xtru[11], ytru[11];
  Double_t xlen, ylen, zlen;
  Double_t xpos, ypos, zpos, phirot;

  Int_t shapeId = 0;
  if (left) {
    shapeId = 1;
  }
  // The outer wire support parts:
  // - the central part: a BBox
  xlen = (sIBGWireIntSuppBaseCentWid + sIBGWireIntSuppFingerShift) / 2;
  ylen = sIBGWireIntSuppBaseCentHi / 2;
  zlen = sIBGWireIntSuppBaseCentThik / 2;
  TGeoBBox* intSuppCent = new TGeoBBox(xlen, ylen, zlen);
  intSuppCent->SetName(Form("suppcent%d", shapeId));

  // - the lateral part: a Xtru
  xtru[0] = 0;
  ytru[0] = 0;
  xtru[1] = sIBGWireIntSuppBaseFullWid - sIBGWireIntSuppBaseCentWid;
  ytru[1] = ytru[0];
  xtru[2] = xtru[1] + sIBGWireIntSuppFingerLen * TMath::Cos(sIBGWireIntSuppFingerPhi * TMath::DegToRad());
  ytru[2] = sIBGWireIntSuppFingerLen * TMath::Sin(sIBGWireIntSuppFingerPhi * TMath::DegToRad());
  xtru[3] = xtru[2] - sIBGWireIntSuppFingerWid * TMath::Sin(sIBGWireIntSuppFingerPhi * TMath::DegToRad());
  ytru[3] = ytru[2] + sIBGWireIntSuppFingerWid * TMath::Cos(sIBGWireIntSuppFingerPhi * TMath::DegToRad());
  xtru[4] = xtru[1];
  ytru[4] = ytru[3] - sIBGWireIntSuppFingerLen * TMath::Sin(sIBGWireIntSuppFingerPhi * TMath::DegToRad());
  xtru[5] = sIBGWireIntSuppBaseOutSpan;
  ytru[5] = sIBGWireIntSuppBaseCentHi;
  xtru[6] = -sIBGWireIntSuppFingerShift;
  ytru[6] = ytru[5];

  TGeoXtru* intSuppFing = new TGeoXtru(2);
  intSuppFing->DefinePolygon(7, xtru, ytru);
  intSuppFing->DefineSection(0, 0);
  intSuppFing->DefineSection(1, sIBGWireIntSuppFingerThik);
  intSuppFing->SetName(Form("suppfinger%d", shapeId));

  ypos = -intSuppCent->GetDY();
  if (left) {
    xpos = -intSuppCent->GetDX();
    zpos = -intSuppCent->GetDZ() + sIBGWireIntSuppFingerThik;
    phirot = 180;
  } else {
    xpos = intSuppCent->GetDX();
    zpos = -intSuppCent->GetDZ();
    phirot = 0;
  }
  TGeoCombiTrans* intSuppFingMat = new TGeoCombiTrans(xpos, ypos, zpos, new TGeoRotation("", phirot, phirot, 0));
  intSuppFingMat->SetName(Form("suppfingermat%d", shapeId));
  intSuppFingMat->RegisterYourself();

  // - the spiky part: a Xtru
  xtru[0] = 0;
  ytru[0] = 0;
  xtru[1] = sIBGWireIntSuppBaseCentWid;
  ytru[1] = ytru[0];
  xtru[2] = xtru[1];
  ytru[2] = sIBGWireIntSuppBaseCentHi;
  xtru[3] = xtru[2] - (sIBGWireIntSuppBaseFullSpan - sIBGWireIntSuppSpikyWid - sIBGWireIntSuppBaseOutSpan);
  ytru[3] = ytru[2];
  xtru[4] = xtru[3];
  ytru[4] = ytru[3] + sIBGWireIntSuppSpikyHi;
  xtru[5] = xtru[4] - (sIBGWireIntSuppSpikyWid - sIBGWireIntSuppSpikyXin) / 2;
  ytru[5] = ytru[4] - (sIBGWireIntSuppSpikyHi - sIBGWireIntSuppSpikyYin);
  xtru[6] = xtru[5];
  ytru[6] = ytru[3];
  xtru[7] = xtru[6] - sIBGWireIntSuppSpikyXin;
  ytru[7] = ytru[6];
  xtru[8] = xtru[7];
  ytru[8] = ytru[5];
  xtru[9] = xtru[8] - (sIBGWireIntSuppSpikyWid - sIBGWireIntSuppSpikyXin) / 2;
  ytru[9] = ytru[4];
  xtru[10] = xtru[9];
  ytru[10] = ytru[3];

  TGeoXtru* intSuppSpiky = new TGeoXtru(2);
  intSuppSpiky->DefinePolygon(11, xtru, ytru);
  intSuppSpiky->DefineSection(0, 0);
  intSuppSpiky->DefineSection(1, sIBGWireIntSuppSpikyThik);
  intSuppSpiky->SetName(Form("suppspiky%d", shapeId));

  ypos = -intSuppCent->GetDY();
  if (left) {
    xpos = intSuppCent->GetDX();
    zpos = intSuppCent->GetDZ() - sIBGWireIntSuppSpikyThik;
    phirot = 180;
  } else {
    xpos = -intSuppCent->GetDX();
    zpos = -intSuppCent->GetDZ() - sIBGWireIntSuppSpikyThik;
    phirot = 0;
  }
  TGeoCombiTrans* intSuppSpikyMat = new TGeoCombiTrans(xpos, ypos, zpos, new TGeoRotation("", phirot, phirot, 0));
  intSuppSpikyMat->SetName(Form("suppspikymat%d", shapeId));
  intSuppSpikyMat->RegisterYourself();

  // The actual wire outer support: a CompositeShape
  TString compoShape = Form("suppcent%d", shapeId);
  compoShape += Form("+suppfinger%d:suppfingermat%d", shapeId, shapeId);
  compoShape += Form("+suppspiky%d:suppspikymat%d", shapeId, shapeId);
  TGeoCompositeShape* supportShape = new TGeoCompositeShape(compoShape);

  // Finally return the support shape
  return supportShape;
}

TGeoCompositeShape* V3Services::ibConvWireOutSupport()
{
  //
  // Creates the shape of the external support of the Inner Barrel Gamma
  // Conversion wire: being pretty complicate, we devote a dedicate method
  // for it; the shape is a bit simplified but the total material budget
  // is preserved
  // Blueprints (from EDMS) : ALIITSUP0914
  //
  // Input:
  //
  // Output:
  //
  // Return:
  //         The support shape as a TGeoCompositeShape
  //
  // Created:      30 Sep 2022  Mario Sitta
  //

  // The outer wire support
  static const Double_t sIBGWireOutSuppWideIn = 22.5 * sMm;
  static const Double_t sIBGWireOutSuppWideOut = 24.0 * sMm;
  static const Double_t sIBGWireOutSuppWideTot = 31.0 * sMm;
  static const Double_t sIBGWireOutSuppLenIn = 8.0 * sMm;
  static const Double_t sIBGWireOutSuppLenOut = 9.3 * sMm;
  static const Double_t sIBGWireOutSuppLength = 26.5 * sMm;
  static const Double_t sIBGWireOutSuppLenToSide = 10.5 * sMm;
  static const Double_t sIBGWireOutSuppThick = 3.0 * sMm;
  static const Double_t sIBGWireOutSuppPhi = 30.0; // Deg
  static const Double_t sIBGWireOutSuppLenToPlate = 16.3 * sMm;
  static const Double_t sIBGWireOutSuppWidToPlate = 27.75 * sMm;
  static const Double_t sIBGWireOutSuppPlateWid = 17.0 * sMm;

  // Local variables
  Double_t xtru[8], ytru[8], xyarb[16];
  Double_t xlen, ylen, zlen;
  Double_t xpos, ypos, zpos;

  // The outer wire support parts:
  // - the base: a Xtru
  xtru[0] = 0;
  ytru[0] = 0;
  xtru[1] = sIBGWireOutSuppWideIn;
  ytru[1] = ytru[0];
  xtru[2] = xtru[1];
  xtru[3] = sIBGWireOutSuppWideOut;
  ytru[3] = sIBGWireOutSuppLenToSide;
  ytru[2] = ytru[3] - (xtru[3] - xtru[2]) * TMath::Tan((90 - sIBGWireOutSuppPhi) * TMath::DegToRad());
  xtru[4] = xtru[3];
  ytru[4] = sIBGWireOutSuppLength - sIBGWireOutSuppLenIn;
  xtru[5] = xtru[2];
  ytru[5] = ytru[4];
  xtru[6] = xtru[5];
  ytru[6] = sIBGWireOutSuppLength;
  xtru[7] = xtru[0];
  ytru[7] = ytru[6];

  TGeoXtru* ibWireOutSuppBase = new TGeoXtru(2);
  ibWireOutSuppBase->DefinePolygon(8, xtru, ytru);
  ibWireOutSuppBase->DefineSection(0, 0);
  ibWireOutSuppBase->DefineSection(1, sIBGWireOutSuppThick);
  ibWireOutSuppBase->SetName("ibwireoutsuppbase");

  // - the inclined side: an Arb8
  zlen = ibWireOutSuppBase->GetY(4) - ibWireOutSuppBase->GetY(3);
  ylen = zlen * TMath::Tan(sIBGWireOutSuppPhi * TMath::DegToRad());

  xyarb[0] = sIBGWireOutSuppThick / 2;
  xyarb[1] = 0;

  xyarb[2] = -sIBGWireOutSuppThick / 2;
  xyarb[3] = 0;

  xyarb[4] = xyarb[2];
  xyarb[5] = 0.0;

  xyarb[6] = xyarb[0];
  xyarb[7] = 0.0;

  xyarb[8] = sIBGWireOutSuppPlateWid / 2;
  xyarb[9] = 0;

  xyarb[10] = -sIBGWireOutSuppPlateWid / 2;
  xyarb[11] = 0;

  xyarb[12] = xyarb[10];
  xyarb[13] = ylen;

  xyarb[14] = xyarb[8];
  xyarb[15] = ylen;

  TGeoArb8* ibWireOutSuppArb = new TGeoArb8(zlen / 2, xyarb);
  ibWireOutSuppArb->SetName("ibwireoutsupparb");

  xpos = ibWireOutSuppBase->GetX(3);
  ypos = (ibWireOutSuppBase->GetY(3) + ibWireOutSuppBase->GetY(4)) / 2;
  zpos = sIBGWireOutSuppThick / 2;
  TGeoCombiTrans* ibOutSuppArbMat = new TGeoCombiTrans(xpos, ypos, zpos, new TGeoRotation("", 0, -90, -90));
  ibOutSuppArbMat->SetName("iboutsupparbmat");
  ibOutSuppArbMat->RegisterYourself();

  // - the vertical plate: a BBox
  xlen = sIBGWireOutSuppWideTot - sIBGWireOutSuppWidToPlate;
  ylen = sIBGWireOutSuppLength + sIBGWireOutSuppLenOut - ibWireOutSuppBase->GetY(4);
  zlen = sIBGWireOutSuppPlateWid;
  TGeoBBox* ibWireOutSuppPlate = new TGeoBBox(xlen / 2, ylen / 2, zlen / 2);
  ibWireOutSuppPlate->SetName("ibwireoutsuppplate");

  xpos += (xyarb[15] - ibWireOutSuppPlate->GetDX());
  ypos += (ibWireOutSuppArb->GetDz() + ibWireOutSuppPlate->GetDY());
  TGeoTranslation* ibOutSuppPlateMat = new TGeoTranslation(xpos, ypos, zpos);
  ibOutSuppPlateMat->SetName("iboutsuppplatemat");
  ibOutSuppPlateMat->RegisterYourself();

  // The actual wire outer support: a CompositeShape
  TGeoCompositeShape* supportShape = new TGeoCompositeShape("ibwireoutsuppbase+ibwireoutsupparb:iboutsupparbmat+ibwireoutsuppplate:iboutsuppplatemat");

  // Finally return the support shape
  return supportShape;
}

void V3Services::obConvWire(TGeoVolume* mother, const TGeoManager* mgr)
{
  //
  // Creates the 1mm wire for Gamma Conversion studies and its supports
  // Blueprints (from EDMS) : ALIITSUP0870 (the whole assembly)
  //                          ALIITSUP0869 (the actual wire)
  //                          ALIITSUP0868 (each tension screw)
  //                          ALIITSUP0866 (the support plate)
  //                          ALIITSUP0867 (each wire holder)
  //
  // Input:
  //         mother : the volume where to place the current created cylinder
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      08 Sep 2022  Mario Sitta
  //

  // The wire
  static const Double_t sOBGWireOverallLen = 1014.0 * sMm;
  static const Double_t sOBGWireDout = 1.0 * sMm;

  // The wire support
  static const Double_t sOBGWireSuppLenTot = 55.0 * sMm;
  static const Double_t sOBGWireSuppLenWide = (16.5 + 18.0) * sMm;
  static const Double_t sOBGWireSuppLenNarrow = 3.5 * sMm;
  static const Double_t sOBGWireSuppWideMin = 15.0 * sMm;
  static const Double_t sOBGWireSuppWideMax = 20.0 * sMm;
  static const Double_t sOBGWireSuppThick = 1.8 * sMm;
  static const Double_t sOBGWireSuppBulgeH = 5.0 * sMm;
  static const Double_t sOBGWireSuppBulgeD = 5.9 * sMm;
  static const Double_t sOBGWireSuppBulgeYdist = 21.0 * sMm;
  static const Double_t sOBGWireSuppHoleYpos = 16.5 * sMm;
  static const Double_t sOBGWireSuppCylDmin = 3.1 * sMm;
  static const Double_t sOBGWireSuppCylDmax = 6.0 * sMm;

  static const Double_t sOBGWireSuppZpos = 487.5 * sMm;
  static const Double_t sOBGWireSuppXtrans = 4.8 * sMm;
  static const Double_t sOBGWireSuppYtrans = 4.7 * sMm;

  // The wire screw
  static const Double_t sOBGWireScrewDout = 3.0 * sMm;
  static const Double_t sOBGWireScrewLen = 37.0 * sMm;

  // Local variables
  Double_t xtru[8], ytru[8];
  Double_t zlen, radius;
  Double_t xpos, ypos, zpos;

  // The wire: a Tube
  zlen = sOBGWireOverallLen - 2 * sOBGWireScrewLen;
  TGeoTube* obWireSh = new TGeoTube(0, sOBGWireDout / 2, zlen / 2);

  // The wire support parts:
  // - the base: a Xtru
  zpos = sOBGWireSuppLenTot - 0.5 * sOBGWireSuppCylDmax - sOBGWireSuppLenNarrow;

  xtru[0] = sOBGWireSuppWideMax / 2;
  ytru[0] = 0.;
  xtru[1] = xtru[0];
  ytru[1] = sOBGWireSuppLenWide;
  xtru[2] = sOBGWireSuppWideMin / 2;
  ytru[2] = zpos;
  xtru[3] = xtru[2];
  ytru[3] = ytru[2] + sOBGWireSuppLenNarrow;
  for (Int_t j = 0; j < 4; j++) {
    xtru[4 + j] = -xtru[3 - j];
    ytru[4 + j] = ytru[3 - j];
  }

  TGeoXtru* obWireSuppBase = new TGeoXtru(2);
  obWireSuppBase->DefinePolygon(8, xtru, ytru);
  obWireSuppBase->DefineSection(0, 0);
  obWireSuppBase->DefineSection(1, sOBGWireSuppThick);
  obWireSuppBase->SetName("obwiresuppbase");

  // - the screw bulge: a Tube
  TGeoTube* obWireSuppBulge = new TGeoTube(0, sOBGWireSuppBulgeD / 2, sOBGWireSuppBulgeH / 2);
  obWireSuppBulge->SetName("obwiresuppbulge");

  ypos = sOBGWireSuppHoleYpos - 0.5 * sOBGWireSuppBulgeYdist;
  zpos = sOBGWireSuppThick + obWireSuppBulge->GetDz();
  TGeoTranslation* obWireSuppBulgeMat1 = new TGeoTranslation(0, ypos, zpos);
  obWireSuppBulgeMat1->SetName("obwiresuppbulgemat1");
  obWireSuppBulgeMat1->RegisterYourself();

  ypos = sOBGWireSuppHoleYpos + 0.5 * sOBGWireSuppBulgeYdist;
  TGeoTranslation* obWireSuppBulgeMat2 = new TGeoTranslation(0, ypos, zpos);
  obWireSuppBulgeMat2->SetName("obwiresuppbulgemat2");
  obWireSuppBulgeMat2->RegisterYourself();

  // - the terminal cylinder: a Tube
  TGeoTube* obWireSuppCyl = new TGeoTube(0, sOBGWireSuppCylDmax / 2, xtru[3]);
  obWireSuppCyl->SetName("obwiresuppcyl");

  ypos = ytru[3];
  zpos = obWireSuppCyl->GetRmax();
  TGeoCombiTrans* obWireSuppCylMat = new TGeoCombiTrans(0, ypos, zpos, new TGeoRotation("", 90, 90, 0));
  obWireSuppCylMat->SetName("obwiresuppcylmat");
  obWireSuppCylMat->RegisterYourself();

  // - the terminal cylinder hole: a Tube
  TGeoTube* obWireSuppCylHol = new TGeoTube(0, sOBGWireSuppCylDmin / 2, 1.05 * xtru[3]);
  obWireSuppCylHol->SetName("obwiresuppcylhol");

  TGeoCombiTrans* obWireSuppCylHolMat = new TGeoCombiTrans(0, ypos, zpos, new TGeoRotation("", 90, 90, 0));
  obWireSuppCylHolMat->SetName("obwiresuppcylholmat");
  obWireSuppCylHolMat->RegisterYourself();

  // The actual wire support: a CompositeShape
  TGeoCompositeShape* obWireSuppSh = new TGeoCompositeShape("obwiresuppbase+obwiresuppbulge:obwiresuppbulgemat1+obwiresuppbulge:obwiresuppbulgemat2+obwiresuppcyl:obwiresuppcylmat-obwiresuppcylhol:obwiresuppcylholmat");

  // The wire screw: a Tube
  TGeoTube* obWireScrewSh = new TGeoTube(0, sOBGWireScrewDout / 2, sOBGWireScrewLen / 2);

  // We have all shapes: now create the real volumes
  TGeoMedium* medTungsten = mgr->GetMedium(Form("%s_TUNGSTEN$", GetDetName()));
  TGeoMedium* medTitanium = mgr->GetMedium(Form("%s_TITANIUM$", GetDetName()));
  TGeoMedium* medBrass = mgr->GetMedium(Form("%s_BRASS$", GetDetName()));

  TGeoVolume* obWireVol = new TGeoVolume("OBGammaConvWire", obWireSh, medTungsten);
  obWireVol->SetFillColor(kGray);
  obWireVol->SetLineColor(kGray);

  TGeoVolume* obWireSuppVol = new TGeoVolume("OBGammaConvWireSupport", obWireSuppSh, medTitanium);
  obWireSuppVol->SetFillColor(kRed);
  obWireSuppVol->SetLineColor(kRed);

  TGeoVolume* obWireScrewVol = new TGeoVolume("OBGammaConvWireScrew", obWireScrewSh, medBrass);
  obWireScrewVol->SetFillColor(kYellow);
  obWireScrewVol->SetLineColor(kYellow);

  // To simplify a lot the overall placing we put everything in a Assembly
  TGeoVolumeAssembly* obGammaConvWire = new TGeoVolumeAssembly("OBGammaConversionWire");

  zpos = sOBGWireSuppZpos;
  obGammaConvWire->AddNode(obWireSuppVol, 1, new TGeoCombiTrans(0, 0, zpos, new TGeoRotation("", 90, 90, 90)));
  obGammaConvWire->AddNode(obWireSuppVol, 2, new TGeoCombiTrans(0, 0, -zpos, new TGeoRotation("", 90, 90, 90)));

  xpos = obWireSuppCyl->GetRmax();
  ypos = ytru[3]; // The same as the hole in the support
  zpos = 0.5 * sOBGWireOverallLen - obWireScrewSh->GetDz();
  obGammaConvWire->AddNode(obWireScrewVol, 1, new TGeoTranslation(xpos, -ypos, zpos));
  obGammaConvWire->AddNode(obWireScrewVol, 2, new TGeoTranslation(xpos, -ypos, -zpos));

  obGammaConvWire->AddNode(obWireVol, 1, new TGeoTranslation(xpos, -ypos, 0));

  // Finally put everything in the mother volume
  TGeoVolume* obConeVol = gGeoManager->GetVolume("OBConeSideA");
  if (!obConeVol) { // Should never happen, OB cones are created before us
    LOG(error) << "OBConeSideA not defined in geometry, using default radius";
    radius = 292.5 * sMm;
  } else {
    radius = (static_cast<TGeoPcon*>(obConeVol->GetShape()))->GetRmax(2);
  }

  xpos = radius - sOBGWireSuppXtrans;
  ypos = sOBGWireSuppLenTot + sOBGWireSuppYtrans;
  mother->AddNode(obGammaConvWire, 1, new TGeoCombiTrans(xpos, ypos, 0, new TGeoRotation("", 15, 0, 0)));
}
