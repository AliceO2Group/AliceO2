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

#include "FairLogger.h" // for LOG

//#include <TGeoArb8.h>           // for TGeoArb8
//#include <TGeoBBox.h>    // for TGeoBBox
//#include <TGeoCone.h>    // for TGeoConeSeg, TGeoCone
//#include <TGeoPcon.h>    // for TGeoPcon
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
