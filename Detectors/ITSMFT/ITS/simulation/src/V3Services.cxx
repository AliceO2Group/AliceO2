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
const Double_t V3Services::sIBWheelACZdist = 308.0 * sMm;

ClassImp(V3Services)

#define SQ(A) (A) * (A)

  V3Services::V3Services()
  : V11Geometry()
{
}

V3Services::~V3Services() = default;

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
  //         lay : the layer number
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
  static const Double_t sEndWheelCDmax[3] = { 57.0 * sMm, 73.0 * sMm, 89.0 * sMm };
  static const Double_t sEndWheelCDmin[3] = { 44.5 * sMm, 58.0 * sMm, 74.0 * sMm };
  static const Double_t sEndWheelCHeigh[3] = { 25.0 * sMm, 22.5 * sMm, 20.0 * sMm };
  static const Double_t sEndWheelCThick = 0.6 * sMm;

  static const Int_t sEndWCWallNHoles[3] = { 6, 8, 10 };
  static const Double_t sEndWCWallHoleD = 4.5 * sMm;
  static const Double_t sEndWCWallHoleZpos = 4.0 * sMm;

  static const Int_t sEndWCBaseNBigHoles = 5;
  static const Int_t sEndWCBaseNSmalHoles = 6;
  static const Double_t sEndWCBaseBigHoleD = 3.6 * sMm;
  static const Double_t sEndWCBaseSmalHoleD = 2.5 * sMm;
  static const Double_t sEndWCBaseHolesDpos[3] = { 50.0 * sMm, 64.0 * sMm, 80.0 * sMm };
  static const Double_t sEndWCBaseHolesPhi = 15.0; // Deg

  // The End Wheel Reinforcement
  static const Double_t sEndWCRenfDmin[3] = { 44.0 * sMm, 58.0 * sMm, 74.0 * sMm };
  static const Double_t sEndWCRenfDint[3] = { 55.0 * sMm, 71.0 * sMm, 87.0 * sMm };
  static const Double_t sEndWCRenfHeigh[3] = { 4.0 * sMm, 3.0 * sMm, 3.0 * sMm };
  static const Double_t sEndWCRenfThick = 0.6 * sMm;

  static const Double_t sEndWCRenfZpos = 14.2 * sMm;

  static const Int_t sEndWCRenfNSmalHoles[3] = { 5, 7, 9 };

  // The End Wheel Steps
  static const Double_t sEndWCStepXdispl[3] = { 4.0 * sMm, 6.5 * sMm, 8.5 * sMm };
  static const Double_t sEndWCStepYdispl[3] = { 24.4 * sMm, 32.1 * sMm, 39.6 * sMm };
  static const Double_t sEndWCStepR[3] = { 27.8 * sMm, 35.8 * sMm, 43.8 * sMm };

  static const Double_t sEndWCStepZlen = 14.0 * sMm;

  static const Double_t sEndWCStepHoleXpos = 3.0 * sMm;
  static const Double_t sEndWCStepHoleZpos = 4.0 * sMm;
  static const Double_t sEndWCStepHoleZdist = 4.0 * sMm;

  static const Double_t sEndWCStepHolePhi[3] = { 30.0, 22.5, 18.0 }; // Deg
  static const Double_t sEndWCStepHolePhi0[2] = { 9.5, 10.5 };       // Deg
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
  TGeoBBox* stepBoxSh = new TGeoBBox(Form("stepBoxSh%d", iLay), xlen / 2, ylen / 2, sEndWCStepZlen / 2);

  xpos = sEndWCStepXdispl[iLay] + stepBoxSh->GetDX();
  ypos = sEndWCStepYdispl[iLay] + stepBoxSh->GetDY();
  TGeoTranslation* stepBoxTr = new TGeoTranslation(Form("stepBoxTr%d", iLay), xpos, ypos, 0);
  stepBoxTr->RegisterYourself();

  phimin = 90. - TMath::ACos(sEndWCStepYdispl[iLay] / rmin) * TMath::RadToDeg() - 5;
  dphi = 90. - TMath::ASin(sEndWCStepXdispl[iLay] / rmin) * TMath::RadToDeg() - phimin + 10;
  rmax = rmin + 2 * stepBoxSh->GetDY();

  TGeoPcon* stepPconSh = new TGeoPcon(Form("stepPconSh%d", iLay), phimin, dphi, 2);
  stepPconSh->DefineSection(0, -1.05 * sEndWCStepZlen / 2, rmin, rmax);
  stepPconSh->DefineSection(1, 1.05 * sEndWCStepZlen / 2, rmin, rmax);

  TGeoCompositeShape* stepCSh = new TGeoCompositeShape(Form("stepBoxSh%d:stepBoxTr%d-stepPconSh%d", iLay, iLay, iLay));

  // We have all shapes: now create the real volumes
  TGeoMedium* medCarbon = mgr->GetMedium("ITS_M55J6K$"); // TO BE CHECKED
  TGeoMedium* medPEEK = mgr->GetMedium("ITS_PEEKCF30$");

  TGeoVolume* endWheelCVol = new TGeoVolume(Form("EndWheelCBasis%d", iLay), endWheelCSh, medCarbon);
  endWheelCVol->SetFillColor(kBlue);
  endWheelCVol->SetLineColor(kBlue);

  TGeoVolume* endWFakeVol = new TGeoVolume(Form("EndWheelCFake%d", iLay), endwcwalhol, medCarbon);
  endWFakeVol->SetFillColor(kBlue);
  endWFakeVol->SetLineColor(kBlue);

  TGeoVolume* stepCVol = new TGeoVolume(Form("EndWheelCStep%d", iLay), stepCSh, medPEEK);
  stepCVol->SetFillColor(kBlue);
  stepCVol->SetLineColor(kBlue);

  // Finally put everything in the mother volume
  zpos = sIBWheelACZdist / 2 - (sEndWCStepHoleZpos + sEndWCStepHoleZdist);
  endWheel->AddNode(endWheelCVol, 1, new TGeoCombiTrans(0, 0, -zpos, new TGeoRotation("", 0, 180, 0)));

  // The position of the Steps is given wrt the holes (see eg. ALIITSUP0187)
  if (iLay == 0) { // For Layer 0 we have the linear displacement
    Double_t cathetus = sEndWCStepYlow - sEndWCStepHoleXpos + (static_cast<TGeoBBox*>(stepCVol->GetShape()))->GetDX();
    Double_t radius = TMath::Sqrt(sEndWCStepYdispl[iLay] * sEndWCStepYdispl[iLay] + cathetus * cathetus);
    dphi = TMath::ASin(cathetus / radius) * TMath::RadToDeg();
  } else { // For Layers 1 & 2 we have a displacement angle
    dphi = sEndWCStepHolePhi0[iLay - 1];
  }

  Int_t numberOfStaves = GeometryTGeo::Instance()->getNumberOfStaves(iLay);
  zpos += (static_cast<TGeoBBox*>(stepCVol->GetShape()))->GetDZ();
  for (Int_t j = 0; j < numberOfStaves; j++) {
    Double_t phi = dphi + j * sEndWCStepHolePhi[iLay];
    endWheel->AddNode(stepCVol, j + 1, new TGeoCombiTrans(0, 0, -zpos, new TGeoRotation("", 180, 180, -90 - phi)));
  }
  /*
  // TEST TEST TEST
  if(iLay == 1) {
  rmin = sEndWheelCDmax[iLay] / 2 - sEndWheelCThick / 2;
  zpos = sEndWCWallHoleZpos;
  dphi = 180. / sEndWCWallNHoles[iLay];
  phimin = dphi / 2.;
  for(Int_t ihole = 0; ihole < 2 * sEndWCWallNHoles[iLay]; ihole++) {
    Double_t phi = phimin + ihole * dphi;
    xpos = rmin*TMath::Sin(phi * TMath::DegToRad());
    ypos = rmin*TMath::Cos(phi * TMath::DegToRad());
    endWheel->AddNode(endWFakeVol, 1+ihole, new TGeoCombiTrans(xpos, ypos, zpos, new TGeoRotation("", -phi, 90, 0)));
  }
  }
*/
}
