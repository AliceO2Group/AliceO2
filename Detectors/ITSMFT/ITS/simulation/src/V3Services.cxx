// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file V3Services.cxx
/// \brief Implementation of the V3Services class
/// \author Mario Sitta <sitta@to.infn.it>
/// \author Parinya Namwongsa <parinya.namwongsa@cern.ch>

#include "ITSSimulation/V3Services.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSSimulation/Detector.h"
#include "ITSMFTSimulation/AlpideChip.h"

#include "FairLogger.h" // for LOG

//#include <TGeoArb8.h>           // for TGeoArb8
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
//const Double_t V3Services::sIBWheelACZdist = 308.0 * sMm;
const Double_t V3Services::sIBWheelACZdist = 306.0 * sMm;

ClassImp(V3Services);

#define SQ(A) (A) * (A)

V3Services::V3Services()
  : V11Geometry()
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

  for (Int_t jLay = 0; jLay < sNumberInnerLayers; jLay++)
    ibEndWheelSideA(jLay, endWheelsVol, mgr);

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

  for (Int_t jLay = 0; jLay < sNumberInnerLayers; jLay++)
    ibEndWheelSideC(jLay, endWheelsVol, mgr);

  // Return the wheels
  return endWheelsVol;
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
  TGeoMedium* medCarbon = mgr->GetMedium("ITS_M55J6K$"); // TO BE CHECKED
  TGeoMedium* medPEEK = mgr->GetMedium("ITS_PEEKCF30$");

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
  static const Double_t sEndWCStepHolePhi0[2] = {9.5, 10.5};       // Deg
  static const Double_t sEndWCStepYlow = 7.0 * sMm;

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
    if ((ihole > 1 && ihole < 5) || (ihole > 5 && ihole < 9)) // Small holes
      strcpy(holename, endwcbasShol->GetName());
    else
      strcpy(holename, endwcbasBhol->GetName());
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
  TGeoMedium* medCarbon = mgr->GetMedium("ITS_M55J6K$"); // TO BE CHECKED
  TGeoMedium* medPEEK = mgr->GetMedium("ITS_PEEKCF30$");

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
  dphi = sEndWCStepHolePhi0[iLay];

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
