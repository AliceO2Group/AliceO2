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

/// \file V3Cage.cxx
/// \brief Implementation of the V3Cage class
/// \author Mario Sitta <sitta@to.infn.it>

#include "ITSSimulation/V3Cage.h"
#include "ITSSimulation/V11Geometry.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSSimulation/Detector.h"

#include <fairlogger/Logger.h> // for LOG

//#include <TGeoArb8.h>           // for TGeoArb8
//#include <TGeoBBox.h>    // for TGeoBBox
//#include <TGeoCone.h>    // for TGeoConeSeg, TGeoCone
#include <TGeoPcon.h>    // for TGeoPcon
#include <TGeoManager.h> // for TGeoManager, gGeoManager
#include <TGeoMatrix.h>  // for TGeoCombiTrans, TGeoRotation, etc
//#include <TGeoTrd1.h>           // for TGeoTrd1
#include <TGeoTube.h>           // for TGeoTube, TGeoTubeSeg
#include <TGeoVolume.h>         // for TGeoVolume, TGeoVolumeAssembly
#include <TGeoXtru.h>           // for TGeoXtru
#include <TGeoCompositeShape.h> // for TGeoCompositeShape
#include <TMathBase.h>          // for Abs
#include <TMath.h>              // for Sin, RadToDeg, DegToRad, Cos, Tan, etc

#include <cstdio> // for snprintf

class TGeoMedium;

using namespace TMath;
using namespace o2::its;

// Parameters
const Double_t V3Cage::sCageYInBarrel = 30. * sCm; // This is hardcoded in Detector.cxx

// Cover element (each)
const Double_t V3Cage::sCageCoverZLength = 586.0 * sMm;
const Double_t V3Cage::sCageCoverRint = 540.0 * sMm;
const Double_t V3Cage::sCageCoverRext = 550.0 * sMm;
const Double_t V3Cage::sCageCoverXWidth = 982.0 * sMm;
const Double_t V3Cage::sCageCoverXBaseInt = 944.0 * sMm;
const Double_t V3Cage::sCageCoverXBaseExt = 948.0 * sMm;
const Double_t V3Cage::sCageCoverYBaseHeight = 245.0 * sMm;
const Double_t V3Cage::sCageCoverCoreRint = 541.0 * sMm;
const Double_t V3Cage::sCageCoverCoreRext = 549.0 * sMm;
const Double_t V3Cage::sCageCoverSheetThick = 2 * sMm;
const Double_t V3Cage::sCageCoverRibZLength = 25.0 * sMm;
const Double_t V3Cage::sCageCoverRibRint = 540.0 * sMm;
const Double_t V3Cage::sCageCoverRibRext = 548.0 * sMm;
const Double_t V3Cage::sCageCoverRibXWidth = 984.8 * sMm;
const Double_t V3Cage::sCageCoverRibXBaseInt = 944.0 * sMm;
const Double_t V3Cage::sCageCoverRibYBaseHi = 245.0 * sMm;
const Double_t V3Cage::sCageCoverRibFoldHi = 31.5 * sMm;

const Double_t V3Cage::sCageEndCapDext = 1096.0 * sMm;
const Double_t V3Cage::sCageEndCapDint = 304.0 * sMm;
const Double_t V3Cage::sCageEndCapFoamThick = 8.0 * sMm;
const Double_t V3Cage::sCageEndCapFabThick = 1.0 * sMm;
const Double_t V3Cage::sCageEndCapXWidth = 988.0 * sMm;
const Double_t V3Cage::sCageEndCapSideHoleR = 32.0 * sMm;
const Double_t V3Cage::sCageEndCapSideHoleX = 532.0 * sMm;
const Double_t V3Cage::sCageEndCapCableCutWid = 268.0 * sMm;
const Double_t V3Cage::sCageEndCapCableCutR = 408.5 * sMm;
const Double_t V3Cage::sCageEndCapCableCutPhi = 25.0; // Deg
const Double_t V3Cage::sCageECRoundCrossDmin = 300 * sMm;
const Double_t V3Cage::sCageECRoundCrossDmid = 303 * sMm;
const Double_t V3Cage::sCageECRoundCrossDmax = 312 * sMm;
const Double_t V3Cage::sCageECRoundCrossZext = 6 * sMm;
const Double_t V3Cage::sCageECRoundCrossZint = 5 * sMm;
const Double_t V3Cage::sCageECCableCrosTotHi = 139.0 * sMm;
const Double_t V3Cage::sCageECCableCrosTotZ = 12 * sMm;
const Double_t V3Cage::sCageECCableCrosInXWid = 266.8 * sMm;
const Double_t V3Cage::sCageECCableCrosInThik = 4 * sMm;
const Double_t V3Cage::sCageECCableCrosInZLen = 10.2 * sMm;
const Double_t V3Cage::sCageECCableCrosSidWid = 8 * sMm;

ClassImp(V3Cage);

V3Cage::V3Cage()
  : V11Geometry()
{
}

V3Cage::~V3Cage() = default;

void V3Cage::createAndPlaceCage(TGeoVolume* mother, const TGeoManager* mgr)
{
  //
  // Creates the Cage elements and place them into the mother volume
  //
  // Input:
  //         mother : the mother volume hosting the whole Cage
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      22 Apr 2022  Mario Sitta
  //

  // Local variables
  Double_t zunit, zpos;

  // Create the cover elements
  TGeoVolume* cageCover = createCageCover(mgr);
  TGeoVolume* cageCoverRib = createCageCoverRib(mgr);
  TGeoVolume* cageEndCap = createCageEndCap(mgr);

  // Now place all elements
  mother->AddNode(cageCover, 1, new TGeoTranslation(0, sCageYInBarrel, 0));
  mother->AddNode(cageCover, 2, new TGeoCombiTrans(0, sCageYInBarrel, 0, new TGeoRotation("", 180, 0, 0)));

  zunit = (sCageCoverZLength + sCageCoverRibZLength) / 2.;

  zpos = zunit;
  mother->AddNode(cageCoverRib, 1, new TGeoTranslation(0, sCageYInBarrel, zpos));
  mother->AddNode(cageCoverRib, 2, new TGeoCombiTrans(0, sCageYInBarrel, zpos, new TGeoRotation("", 180, 0, 0)));
  mother->AddNode(cageCoverRib, 3, new TGeoTranslation(0, sCageYInBarrel, -zpos));
  mother->AddNode(cageCoverRib, 4, new TGeoCombiTrans(0, sCageYInBarrel, -zpos, new TGeoRotation("", 180, 0, 0)));

  zpos = 2 * zunit;
  mother->AddNode(cageCover, 3, new TGeoTranslation(0, sCageYInBarrel, zpos));
  mother->AddNode(cageCover, 4, new TGeoCombiTrans(0, sCageYInBarrel, zpos, new TGeoRotation("", 180, 0, 0)));
  mother->AddNode(cageCover, 5, new TGeoTranslation(0, sCageYInBarrel, -zpos));
  mother->AddNode(cageCover, 6, new TGeoCombiTrans(0, sCageYInBarrel, -zpos, new TGeoRotation("", 180, 0, 0)));

  // The end cap is only on C side
  zpos += sCageCoverZLength / 2 + sCageECCableCrosTotZ / 2;
  mother->AddNode(cageEndCap, 1, new TGeoTranslation(0, sCageYInBarrel, -zpos));

  // Third ribs are only on A side
  zpos = 3 * zunit;
  mother->AddNode(cageCoverRib, 5, new TGeoTranslation(0, sCageYInBarrel, zpos));
  mother->AddNode(cageCoverRib, 6, new TGeoCombiTrans(0, sCageYInBarrel, zpos, new TGeoRotation("", 180, 0, 0)));

  return;
}

TGeoVolume* V3Cage::createCageCover(const TGeoManager* mgr)
{
  //
  // Creates a Cage cover element (from drawings ALIITSUP0226, ALIITSUP0225)
  //
  // Input:
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //         The cover as a TGeoVolume
  //
  // Created:      22 Apr 2022  Mario Sitta
  //

  // Local variables
  static const Int_t coverFoldNVert = 6;

  Double_t xvert[coverFoldNVert], yvert[coverFoldNVert];
  Double_t halfBase, zlen, alpha;

  zlen = 0.5 * sCageCoverZLength;

  // The cover core: a TGeoTubeSeg
  halfBase = 0.5 * sCageCoverXBaseInt - sCageCoverSheetThick;
  alpha = TMath::ACos(halfBase / sCageCoverCoreRext) * TMath::RadToDeg();
  TGeoTubeSeg* coreSh = new TGeoTubeSeg("coverCore", sCageCoverCoreRint, sCageCoverCoreRext, zlen, alpha, 180. - alpha);

  // The upper and lower sheets: a TGeoTubeSeg
  // (as a whole volume - will become a sheet when the core is inserted inside)
  halfBase = 0.5 * sCageCoverXBaseInt;
  alpha = TMath::ACos(halfBase / sCageCoverRint) * TMath::RadToDeg();
  TGeoTubeSeg* sheet = new TGeoTubeSeg("coverSheet", sCageCoverRint, sCageCoverCoreRext, zlen, alpha, 180. - alpha);

  // The lateral fold: a Xtru
  xvert[0] = sCageCoverXBaseExt / 2.;
  yvert[0] = sCageCoverYBaseHeight + sCageCoverSheetThick;
  xvert[1] = sCageCoverXWidth / 2.;
  yvert[1] = yvert[0];
  xvert[2] = xvert[1];
  yvert[2] = sCageCoverYBaseHeight;
  xvert[3] = sCageCoverXBaseInt / 2.;
  yvert[3] = yvert[2];
  xvert[4] = xvert[3];
  yvert[4] = TMath::Sqrt(sCageCoverRint * sCageCoverRint - sCageCoverXBaseInt * sCageCoverXBaseInt / 4) + sCageCoverSheetThick;
  xvert[5] = xvert[0];
  yvert[5] = yvert[4];

  TGeoXtru* fold = new TGeoXtru(2);
  fold->SetName("coverFold");
  fold->DefinePolygon(coverFoldNVert, xvert, yvert);
  fold->DefineSection(0, -zlen);
  fold->DefineSection(1, zlen);

  // A BBox to cut away the curved portion above the fold
  TGeoBBox* cutfold = new TGeoBBox("cutFold", xvert[1] - xvert[0], yvert[5] - yvert[1], 1.05 * zlen);

  // Some matrices to create the composite shape
  TGeoRotation* rotfold = new TGeoRotation("rotFold", 180, 180, 0);
  rotfold->RegisterYourself();

  TGeoTranslation* cutbox1 = new TGeoTranslation("cutBox1", xvert[1], yvert[5], 0);
  cutbox1->RegisterYourself();

  TGeoTranslation* cutbox2 = new TGeoTranslation("cutBox2", -xvert[1], yvert[5], 0);
  cutbox2->RegisterYourself();

  // The cover shape: a CompositeShape
  TGeoCompositeShape* coverSh = new TGeoCompositeShape("coverSheet-cutFold:cutBox1-cutFold:cutBox2+coverFold+coverFold:rotFold");

  // We have all shapes: now create the real volumes
  TGeoMedium* medRohacell = mgr->GetMedium("ITS_ROHACELL$");
  TGeoMedium* medPrepreg = mgr->GetMedium("ITS_M46J6K$");

  TGeoVolume* coverVol = new TGeoVolume("CageCover", coverSh, medPrepreg);
  coverVol->SetFillColor(kBlue);
  coverVol->SetLineColor(kBlue);

  TGeoVolume* coverCoreVol = new TGeoVolume("CageCoverCore", coreSh, medRohacell);
  coverCoreVol->SetFillColor(kYellow);
  coverCoreVol->SetLineColor(kYellow);

  coverVol->AddNode(coverCoreVol, 1, nullptr);

  // Finally return the cover volume
  return coverVol;
}

TGeoVolume* V3Cage::createCageCoverRib(const TGeoManager* mgr)
{
  //
  // Creates a Cage cover rib element (from drawing ALIITSUP0228)
  //
  // Input:
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //         The cover rib as a TGeoVolume
  //
  // Created:      22 Apr 2022  Mario Sitta
  //

  // Local variables
  static const Int_t ribFoldNVert = 3;

  Double_t xvert[ribFoldNVert], yvert[ribFoldNVert];
  Double_t halfBase, zlen, alpha;

  zlen = 0.5 * sCageCoverRibZLength;

  // The rib main segment: a TGeoTubeSeg
  halfBase = 0.5 * sCageCoverRibXBaseInt;
  alpha = TMath::ACos(halfBase / sCageCoverRibRint) * TMath::RadToDeg();
  TGeoTubeSeg* mainSh = new TGeoTubeSeg("coverRibMain", sCageCoverRibRint, sCageCoverRibRext, zlen, alpha, 180. - alpha);

  // The lateral fold: a Xtru
  xvert[0] = sCageCoverRibXWidth / 2.;
  yvert[0] = sCageCoverRibYBaseHi;
  xvert[1] = sCageCoverRibXBaseInt / 2.;
  yvert[1] = yvert[0];
  xvert[2] = xvert[1];
  yvert[2] = yvert[1] + sCageCoverRibFoldHi;

  TGeoXtru* fold = new TGeoXtru(2);
  fold->SetName("coverRibFold");
  fold->DefinePolygon(ribFoldNVert, xvert, yvert);
  fold->DefineSection(0, -zlen);
  fold->DefineSection(1, zlen);

  // Some matrices to create the composite shape
  TGeoRotation* rotfold = new TGeoRotation("rotRibFold", 180, 180, 0);
  rotfold->RegisterYourself();

  // The cover rib shape: a CompositeShape
  TGeoCompositeShape* ribSh = new TGeoCompositeShape("coverRibMain+coverRibFold+coverRibFold:rotRibFold");

  // We have all shapes: now create the real volume
  TGeoMedium* medAl = mgr->GetMedium("ITS_ALUMINUM$");

  TGeoVolume* ribVol = new TGeoVolume("CageCoverRib", ribSh, medAl);
  ribVol->SetFillColor(kGray);
  ribVol->SetLineColor(kGray);

  // Finally return the cover rib volume
  return ribVol;
}

TGeoVolume* V3Cage::createCageEndCap(const TGeoManager* mgr)
{
  //
  // Creates the Cage End Cap (from drawings ALIITSUP0235, ALIITSUP0229)
  //
  // Input:
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //         The end cap as a TGeoVolumeAssembly
  //
  // Created:      30 Jun 2022  Mario Sitta
  //

  // Local variables
  Double_t rmin, rmid, rmax;
  Double_t xlen, ylen, zlen;
  Double_t xpos, ypos, zpos;

  // The TGeoVolumeAssembly holding all elements
  TGeoVolumeAssembly* endCapVol = new TGeoVolumeAssembly("CageEndCap");

  // The basic fabric disk: a Tube
  rmin = sCageEndCapDint / 2;
  rmax = sCageEndCapDext / 2;
  zlen = sCageEndCapFabThick / 2;

  TGeoTube* baseFabric = new TGeoTube(rmin, rmax, zlen);
  baseFabric->SetName("endCapBaseFabric");

  // The basic foam disk: a Tube
  zlen = sCageEndCapFoamThick / 2;

  TGeoTube* baseFoam = new TGeoTube(rmin, rmax, zlen);
  baseFoam->SetName("endCapBaseFoam");

  // Common cut-out shapes
  xlen = (sCageEndCapDext - sCageEndCapXWidth) / 2;
  ylen = 0.6 * rmax;

  TGeoBBox* sideCut = new TGeoBBox(xlen, ylen, 2 * zlen);
  sideCut->SetName("endCapBoxCut");

  TGeoTube* sideHole = new TGeoTube(0, sCageEndCapSideHoleR, 2 * zlen);
  sideHole->SetName("endCapSideHole");

  xlen = sCageEndCapCableCutWid / 2;
  ylen = rmax - sCageEndCapCableCutR;

  TGeoBBox* cableCut = new TGeoBBox(xlen, ylen, 2 * zlen);
  cableCut->SetName("endCapCableCut");

  // Some matrices to create the composite shapes
  xpos = rmax;

  TGeoTranslation* boxCutTr1 = new TGeoTranslation("boxCutTr1", xpos, 0, 0);
  boxCutTr1->RegisterYourself();

  TGeoTranslation* boxCutTr2 = new TGeoTranslation("boxCutTr2", -xpos, 0, 0);
  boxCutTr2->RegisterYourself();

  xpos = sCageEndCapSideHoleX / 2;

  TGeoTranslation* sidHolTr1 = new TGeoTranslation("sideHoleTr1", xpos, 0, 0);
  sidHolTr1->RegisterYourself();

  TGeoTranslation* sidHolTr2 = new TGeoTranslation("sideHoleTr2", -xpos, 0, 0);
  sidHolTr2->RegisterYourself();

  xpos = rmax * TMath::Sin(sCageEndCapCableCutPhi * TMath::DegToRad());
  ypos = rmax * TMath::Cos(sCageEndCapCableCutPhi * TMath::DegToRad());

  TGeoCombiTrans* cableMat1 = new TGeoCombiTrans(xpos, ypos, 0, new TGeoRotation("", -sCageEndCapCableCutPhi, 0, 0));
  cableMat1->SetName("cableMat1");
  cableMat1->RegisterYourself();

  TGeoCombiTrans* cableMat2 = new TGeoCombiTrans(-xpos, ypos, 0, new TGeoRotation("", sCageEndCapCableCutPhi, 0, 0));
  cableMat2->SetName("cableMat2");
  cableMat2->RegisterYourself();

  TGeoCombiTrans* cableMat3 = new TGeoCombiTrans(xpos, -ypos, 0, new TGeoRotation("", -180 + sCageEndCapCableCutPhi, 0, 0));
  cableMat3->SetName("cableMat3");
  cableMat3->RegisterYourself();

  TGeoCombiTrans* cableMat4 = new TGeoCombiTrans(-xpos, -ypos, 0, new TGeoRotation("", 180 - sCageEndCapCableCutPhi, 0, 0));
  cableMat4->SetName("cableMat4");
  cableMat4->RegisterYourself();

  // The external fabric panel (each): a CompositeShape
  TGeoCompositeShape* fabricSh = new TGeoCompositeShape("endCapBaseFabric-endCapBoxCut:boxCutTr1-endCapBoxCut:boxCutTr2-endCapSideHole:sideHoleTr1-endCapSideHole:sideHoleTr2-endCapCableCut:cableMat1-endCapCableCut:cableMat2-endCapCableCut:cableMat3-endCapCableCut:cableMat4");

  // The internal foam panel: a CompositeShape
  TGeoCompositeShape* foamSh = new TGeoCompositeShape("endCapBaseFoam-endCapBoxCut:boxCutTr1-endCapBoxCut:boxCutTr2-endCapSideHole:sideHoleTr1-endCapSideHole:sideHoleTr2-endCapCableCut:cableMat1-endCapCableCut:cableMat2-endCapCableCut:cableMat3-endCapCableCut:cableMat4");

  // The round crossing ring: a Pcon (ALIITSUP0281)
  // (in real world it is made of two rimmed rings placed face-to-face;
  // for simplicity and to spare volumes it is implemented as a single
  // Pcon encompassing both rings as mounted in the final setup)
  TGeoPcon* rndCrosSh = new TGeoPcon(0, 360, 6);

  rmin = sCageECRoundCrossDmin / 2;
  rmid = sCageECRoundCrossDmid / 2;
  rmax = sCageECRoundCrossDmax / 2;

  rndCrosSh->DefineSection(0, -sCageECRoundCrossZext, rmin, rmax);
  rndCrosSh->DefineSection(1, -sCageECRoundCrossZint, rmin, rmax);
  rndCrosSh->DefineSection(2, -sCageECRoundCrossZint, rmin, rmid);
  rndCrosSh->DefineSection(3, sCageECRoundCrossZint, rmin, rmid);
  rndCrosSh->DefineSection(4, sCageECRoundCrossZint, rmin, rmax);
  rndCrosSh->DefineSection(5, sCageECRoundCrossZext, rmin, rmax);

  // The (weirdly shaped) cable crossing: a CompositeShape
  TGeoCompositeShape* cblCrosSh = createCageEndCapCableCross(mgr);

  // We have all shapes: now create the real volume
  TGeoMedium* medFabric = mgr->GetMedium("ITS_M46J6K$");
  TGeoMedium* medFoam = mgr->GetMedium("ITS_ROHACELL$");
  TGeoMedium* medAl = mgr->GetMedium("ITS_ALUMINUM$");

  TGeoVolume* fabVol = new TGeoVolume("CageEndCapFabric", fabricSh, medFabric);
  fabVol->SetFillColor(kBlue);
  fabVol->SetLineColor(kBlue);

  TGeoVolume* foamVol = new TGeoVolume("CageEndCapFoam", foamSh, medFoam);
  foamVol->SetFillColor(kYellow);
  foamVol->SetLineColor(kYellow);

  TGeoVolume* rndCrosVol = new TGeoVolume("CageEndCapRoundCross", rndCrosSh, medAl);
  rndCrosVol->SetFillColor(kGray);
  rndCrosVol->SetLineColor(kGray);

  TGeoVolume* cblCrosVol = new TGeoVolume("CageEndCapCableCross", cblCrosSh, medAl);
  cblCrosVol->SetFillColor(kGray);
  cblCrosVol->SetLineColor(kGray);

  // Then build up the end cap
  endCapVol->AddNode(foamVol, 1, nullptr);

  zpos = (sCageEndCapFoamThick + sCageEndCapFabThick) / 2;
  endCapVol->AddNode(fabVol, 1, new TGeoTranslation(0, 0, zpos));
  endCapVol->AddNode(fabVol, 2, new TGeoTranslation(0, 0, -zpos));

  endCapVol->AddNode(rndCrosVol, 1, nullptr);

  rmax = sCageEndCapDext / 2 - sCageECCableCrosTotHi;
  xpos = rmax * TMath::Sin(sCageEndCapCableCutPhi * TMath::DegToRad());
  ypos = rmax * TMath::Cos(sCageEndCapCableCutPhi * TMath::DegToRad());
  endCapVol->AddNode(cblCrosVol, 1, new TGeoCombiTrans(xpos, ypos, 0, new TGeoRotation("", -sCageEndCapCableCutPhi, 0, 0)));
  endCapVol->AddNode(cblCrosVol, 2, new TGeoCombiTrans(-xpos, ypos, 0, new TGeoRotation("", sCageEndCapCableCutPhi, 0, 0)));
  endCapVol->AddNode(cblCrosVol, 3, new TGeoCombiTrans(xpos, -ypos, 0, new TGeoRotation("", -180 + sCageEndCapCableCutPhi, 0, 0)));
  endCapVol->AddNode(cblCrosVol, 4, new TGeoCombiTrans(-xpos, -ypos, 0, new TGeoRotation("", 180 - sCageEndCapCableCutPhi, 0, 0)));

  // Finally return the end cap volume
  return endCapVol;
}

TGeoCompositeShape* V3Cage::createCageEndCapCableCross(const TGeoManager* mgr)
{
  //
  // Creates the Cable Crossing frame for the Cage End Cap (ALIITSUP0282)
  // Since it is pretty cumbersome, we create it in a separate method
  //
  // Input:
  //         mgr : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //         The cable crossing frame as a TGeoCompositeShape
  //
  // Created:      01 Jul 2022  Mario Sitta
  //

  // Local variables
  Double_t rmin, rmid, rmax;
  Double_t xlen, ylen, zlen;
  Double_t xpos, ypos, zpos;

  // The string holding the complex crossing cable structure
  TString crossShape = "";

  // The inner lower part: a BBox and its matrix
  xlen = sCageECCableCrosInXWid / 2;
  ylen = sCageECCableCrosInThik / 2;
  zlen = sCageECCableCrosInZLen / 2;
  TGeoBBox* cabCrosLow = new TGeoBBox(xlen, ylen, zlen);
  cabCrosLow->SetName("CabCrosLow");

  TGeoTranslation* matCrosLow = new TGeoTranslation("MatCrosLow", 0, ylen, 0);
  matCrosLow->RegisterYourself();

  crossShape = "CabCrosLow:MatCrosLow";

  // The side wall of lower part: a BBox and its two matrices
  xlen = (sCageECCableCrosInXWid + 2 * (sCageECCableCrosSidWid - sCageECCableCrosInThik)) / 2;
  ylen = sCageECCableCrosSidWid / 2;
  zlen = (sCageECCableCrosTotZ - sCageECCableCrosInZLen) / 4; // We have 2 sides
  TGeoBBox* cabCrosSide = new TGeoBBox(xlen, ylen, zlen);
  cabCrosSide->SetName("CabCrosSide");

  ypos = 2 * cabCrosLow->GetDY() - cabCrosSide->GetDY();
  zpos = cabCrosLow->GetDZ() + cabCrosSide->GetDZ();
  TGeoTranslation* matCrosSid1 = new TGeoTranslation("MatCrosSid1", 0, ypos, zpos);
  matCrosSid1->RegisterYourself();
  TGeoTranslation* matCrosSid2 = new TGeoTranslation("MatCrosSid2", 0, ypos, -zpos);
  matCrosSid2->RegisterYourself();

  crossShape += "+CabCrosSide:MatCrosSid1+CabCrosSide:MatCrosSid2";

  // The inner lateral part: a BBox and its two matrices
  // (in blueprint the lateral height is not specified, we have to compute it)
  rmax = sCageEndCapDext / 2;
  xlen = sCageECCableCrosInXWid / 2;

  Double_t apothem = TMath::Sqrt(rmax * rmax - xlen * xlen);
  Double_t sagitta = rmax - apothem;

  xlen = sCageECCableCrosInThik / 2;
  ylen = (sCageECCableCrosTotHi - sagitta - sCageECCableCrosInThik) / 2;
  zlen = sCageECCableCrosInZLen / 2;
  TGeoBBox* cabCrosLat = new TGeoBBox(xlen, ylen, zlen);
  cabCrosLat->SetName("CabCrosLateral");

  xpos = 0.5 * sCageECCableCrosInXWid - cabCrosLat->GetDX();
  ypos = cabCrosLat->GetDY() + sCageECCableCrosInThik;
  TGeoTranslation* matCrosLat1 = new TGeoTranslation("MatCrosLat1", xpos, ypos, 0);
  matCrosLat1->RegisterYourself();
  TGeoTranslation* matCrosLat2 = new TGeoTranslation("MatCrosLat2", -xpos, ypos, 0);
  matCrosLat2->RegisterYourself();

  crossShape += "+CabCrosLateral:MatCrosLat1+CabCrosLateral:MatCrosLat2";

  // The side wall of lateral part: a BBox and its four matrices
  xlen = sCageECCableCrosSidWid / 2;
  zlen = cabCrosSide->GetDZ();
  TGeoBBox* cabCrosLatSide = new TGeoBBox(xlen, ylen, zlen);
  cabCrosLatSide->SetName("CabCrosLatSide");

  xpos = cabCrosSide->GetDX() - cabCrosLatSide->GetDX();
  zpos = cabCrosLat->GetDZ() + cabCrosLatSide->GetDZ();
  TGeoTranslation* matCrosLatSid1 = new TGeoTranslation("MatCrosLatSid1", xpos, ypos, zpos);
  matCrosLatSid1->RegisterYourself();
  TGeoTranslation* matCrosLatSid2 = new TGeoTranslation("MatCrosLatSid2", xpos, ypos, -zpos);
  matCrosLatSid2->RegisterYourself();
  TGeoTranslation* matCrosLatSid3 = new TGeoTranslation("MatCrosLatSid3", -xpos, ypos, zpos);
  matCrosLatSid3->RegisterYourself();
  TGeoTranslation* matCrosLatSid4 = new TGeoTranslation("MatCrosLatSid4", -xpos, ypos, -zpos);
  matCrosLatSid4->RegisterYourself();

  crossShape += "+CabCrosLatSide:MatCrosLatSid1+CabCrosLatSide:MatCrosLatSid2+CabCrosLatSide:MatCrosLatSid3+CabCrosLatSide:MatCrosLatSid4";

  // The top rounded part: a TubeSeg and its matrix
  xlen = sCageECCableCrosInXWid / 2;
  Double_t phi = TMath::ASin(xlen / rmax) * TMath::RadToDeg();
  rmin = rmax - sCageECCableCrosSidWid;
  zlen = sCageECCableCrosTotZ / 2;
  TGeoTubeSeg* cabCrosRnd = new TGeoTubeSeg(rmin, rmax, zlen, 90 - phi, 90 + phi);
  cabCrosRnd->SetName("CabCrosRoundTop");

  ypos = -rmax + sCageECCableCrosTotHi;
  TGeoTranslation* matCrosRnd = new TGeoTranslation("MatCrosRound", 0, ypos, 0);
  matCrosRnd->RegisterYourself();

  crossShape += "+CabCrosRoundTop:MatCrosRound";

  // Finally create and return the cable crossing shape
  // (the origin of its reference system is below the lower face of cabCrosLow)
  TGeoCompositeShape* cableCross = new TGeoCompositeShape(crossShape.Data());

  return cableCross;
}
