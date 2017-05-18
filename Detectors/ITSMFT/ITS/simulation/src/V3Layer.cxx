/// \file SegmentationPixel.cxx
/// \brief Implementation of the SegmentationPixel class
/// \author Mario Sitta <sitta@to.infn.it>
/// \author Chinorat Kobdaj (kobdaj@g.sut.ac.th)

#include "ITSSimulation/V3Layer.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSSimulation/Detector.h"

#include "FairLogger.h"           // for LOG

#include <TGeoArb8.h>             // for TGeoArb8
#include <TGeoBBox.h>             // for TGeoBBox
#include <TGeoCone.h>             // for TGeoConeSeg, TGeoCone
#include <TGeoPcon.h>             // for TGeoPcon
#include <TGeoManager.h>          // for TGeoManager, gGeoManager
#include <TGeoMatrix.h>           // for TGeoCombiTrans, TGeoRotation, etc
#include <TGeoTrd1.h>             // for TGeoTrd1
#include <TGeoTube.h>             // for TGeoTube, TGeoTubeSeg
#include <TGeoVolume.h>           // for TGeoVolume, TGeoVolumeAssembly
#include <TGeoXtru.h>             // for TGeoXtru
#include <TGeoCompositeShape.h>   // for TGeoCompositeShape
#include "TMathBase.h"            // for Abs
#include <TMath.h>                // for Sin, RadToDeg, DegToRad, Cos, Tan, etc

#include <cstdio>                // for snprintf

class TGeoMedium;

using namespace TMath;
using namespace o2::ITS;

// General Parameters
const Int_t V3Layer::sNumberOfInnerLayers = 3;

const Double_t V3Layer::sDefaultSensorThick = 18 * sMicron;
const Double_t V3Layer::sDefaultChipThick   = 50 * sMicron;

// Inner Barrel Parameters
const Int_t V3Layer::sIBChipsPerRow = 9;
const Int_t V3Layer::sIBNChipRows = 1;

const Double_t V3Layer::sIBFlexCableAlThick   =  50.0  *sMicron;
const Double_t V3Layer::sIBFlexCableKapThick  = 125.0  *sMicron;
const Double_t V3Layer::sIBGlueThick          = 100.0  *sMicron;
const Double_t V3Layer::sIBCarbonFleeceThick  =  20.0  *sMicron;
const Double_t V3Layer::sIBCarbonPaperThick   =  30.0  *sMicron;
const Double_t V3Layer::sIBK13D2UThick        =  70.0  *sMicron;
const Double_t V3Layer::sIBCoolPipeInnerD     =   1.024*sMm;
const Double_t V3Layer::sIBCoolPipeThick      =  25.4  *sMicron;
const Double_t V3Layer::sIBCoolPipeXDist      =   5.0  *sMm;
const Double_t V3Layer::sIBTopVertexWidth1    =   0.258*sMm;
const Double_t V3Layer::sIBTopVertexWidth2    =   0.072*sCm;
const Double_t V3Layer::sIBTopVertexHeight    =   0.04 *sCm;
const Double_t V3Layer::sIBTopVertexAngle     =  60.0; // Deg
const Double_t V3Layer::sIBSideVertexWidth    =   0.05 *sCm;
const Double_t V3Layer::sIBSideVertexHeight   =   0.074*sCm;
const Double_t V3Layer::sIBTopFilamentLength  =   0.9  *sCm;
const Double_t V3Layer::sIBTopFilamentSide    =   0.02 *sCm;
const Double_t V3Layer::sIBTopFilamentAlpha   =  57.0; // Deg
const Double_t V3Layer::sIBTopFilamentGamma   =  65.0; // Deg

const Double_t V3Layer::sIBConnectorXWidth    =  10.0  *sMm;
const Double_t V3Layer::sIBConnectorYTot      =   4.7  *sMm;
const Double_t V3Layer::sIBConnectBlockZLen   =  16.5  *sMm;
const Double_t V3Layer::sIBConnBodyYHeight    =   2.5  *sMm;
const Double_t V3Layer::sIBConnTailYShift     =   0.9  *sMm;
const Double_t V3Layer::sIBConnTailYMid       =   2.5  *sMm;
const Double_t V3Layer::sIBConnTailZLen       =   2.5  *sMm;
const Double_t V3Layer::sIBConnTailOpenPhi    = 120.0; // Deg
const Double_t V3Layer::sIBConnRoundHoleD     =   2.0  *sMm;
const Double_t V3Layer::sIBConnRoundHoleZ     =(7.0-2.0)*sMm;
const Double_t V3Layer::sIBConnSquareHoleX    =   3.0  *sMm;
const Double_t V3Layer::sIBConnSquareHoleZ    =   3.3  *sMm;
const Double_t V3Layer::sIBConnSquareHoleZPos =   9.0  *sMm;
const Double_t V3Layer::sIBConnInsertHoleD    =   3.0  *sMm;
const Double_t V3Layer::sIBConnInsertHoleZPos =   9.0  *sMm;
const Double_t V3Layer::sIBConnTubeHole1D     =   1.6  *sMm;
const Double_t V3Layer::sIBConnTubeHole1ZLen  =   3.0  *sMm;
const Double_t V3Layer::sIBConnTubeHole2D     =   1.2  *sMm;
const Double_t V3Layer::sIBConnTubeHole3XPos  =   1.0  *sMm;
const Double_t V3Layer::sIBConnTubeHole3ZPos  =   2.0  *sMm;
const Double_t V3Layer::sIBConnTubesXDist     =   5.0  *sMm;
const Double_t V3Layer::sIBConnTubesYPos      =   1.25 *sMm;
const Double_t V3Layer::sIBConnInsertInnerX   =   2.0  *sMm;
const Double_t V3Layer::sIBConnInsertZThick   =   0.7  *sMm;
const Double_t V3Layer::sIBConnInsertD        =   2.0  *sMm;
const Double_t V3Layer::sIBConnInsertHeight   =   2.3  *sMm;
const Double_t V3Layer::sIBConnectAFitExtD    =   1.65 *sMm;
const Double_t V3Layer::sIBConnectAFitIntD    =   1.19 *sMm;
const Double_t V3Layer::sIBConnectAFitZLen    =  12.5  *sMm;
const Double_t V3Layer::sIBConnectAFitZOut    =  10.0  *sMm;
const Double_t V3Layer::sIBConnPlugInnerD     =   0.8  *sMm;
const Double_t V3Layer::sIBConnPlugTotLen     =   1.7  *sMm;
const Double_t V3Layer::sIBConnPlugThick      =   0.5  *sMm;

const Double_t V3Layer::sIBStaveHeight        =   0.5  *sCm;

// Outer Barrel Parameters
const Int_t V3Layer::sOBChipsPerRow = 7;
const Int_t V3Layer::sOBNChipRows = 2;

const Double_t V3Layer::sOBHalfStaveWidth = 3.01 * sCm;
const Double_t V3Layer::sOBModuleWidth = sOBHalfStaveWidth;
const Double_t V3Layer::sOBModuleGap = 0.01 * sCm;
const Double_t V3Layer::sOBChipXGap = 0.01 * sCm;
const Double_t V3Layer::sOBChipZGap = 0.01 * sCm;
const Double_t V3Layer::sOBFlexCableAlThick = 0.005 * sCm;
const Double_t V3Layer::sOBFlexCableKapThick = 0.01 * sCm;
const Double_t V3Layer::sOBBusCableAlThick = 0.02 * sCm;
const Double_t V3Layer::sOBBusCableKapThick = 0.02 * sCm;
const Double_t V3Layer::sOBColdPlateThick = 0.012 * sCm;
const Double_t V3Layer::sOBCarbonPlateThick = 0.012 * sCm;
const Double_t V3Layer::sOBGlueThick = 0.03 * sCm;
const Double_t V3Layer::sOBModuleZLength = 21.06 * sCm;
const Double_t V3Layer::sOBHalfStaveYTrans = 1.76 * sMm;
const Double_t V3Layer::sOBHalfStaveXOverlap = 4.3 * sMm;
const Double_t V3Layer::sOBGraphiteFoilThick = 30.0 * sMicron;
const Double_t V3Layer::sOBCoolTubeInnerD = 2.052 * sMm;
const Double_t V3Layer::sOBCoolTubeThick = 32.0 * sMicron;
const Double_t V3Layer::sOBCoolTubeXDist = 11.1 * sMm;

const Double_t V3Layer::sOBSpaceFrameWidth = 42.0 * sMm;
const Double_t V3Layer::sOBSpaceFrameTotHigh = 43.1 * sMm;
const Double_t V3Layer::sOBSFrameBeamRadius = 0.6 * sMm;
const Double_t V3Layer::sOBSpaceFrameLa = 3.0 * sMm;
const Double_t V3Layer::sOBSpaceFrameHa = 0.721979 * sMm;
const Double_t V3Layer::sOBSpaceFrameLb = 3.7 * sMm;
const Double_t V3Layer::sOBSpaceFrameHb = 0.890428 * sMm;
const Double_t V3Layer::sOBSpaceFrameL = 0.25 * sMm;
const Double_t V3Layer::sOBSFBotBeamAngle = 56.5;
const Double_t V3Layer::sOBSFrameBeamSidePhi = 65.0;

ClassImp(V3Layer)

#define SQ(A) (A) * (A)

V3Layer::V3Layer()
  : V11Geometry(),
    mLayerNumber(0),
    mPhi0(0),
    mLayerRadius(0),
    mZLength(0),
    mSensorThickness(0),
    mChipThickness(0),
    mStaveWidth(0),
    mStaveTilt(0),
    mNumberOfStaves(0),
    mNumberOfModules(0),
    mNumberOfChips(0),
    mChipTypeID(0),
    mIsTurbo(0),
    mBuildLevel(0),
    mStaveModel(Detector::kIBModelDummy)
{
  for (int i = kNHLevels; i--;) {
    mHierarchy[i] = 0;
  }
}

V3Layer::V3Layer(Int_t lay, Bool_t turbo, Int_t debug)
  : V11Geometry(debug),
    mLayerNumber(lay),
    mPhi0(0),
    mLayerRadius(0),
    mZLength(0),
    mSensorThickness(0),
    mChipThickness(0),
    mStaveWidth(0),
    mStaveTilt(0),
    mNumberOfStaves(0),
    mNumberOfModules(0),
    mNumberOfChips(0),
    mChipTypeID(0),
    mIsTurbo(turbo),
    mBuildLevel(0),
    mStaveModel(Detector::kIBModelDummy)
{
  for (int i = kNHLevels; i--;) {
    mHierarchy[i] = 0;
  }
}

V3Layer::~V3Layer()
{
}

void V3Layer::createLayer(TGeoVolume *motherVolume)
{
  char volumeName[30];
  Double_t xpos, ypos, zpos;
  Double_t alpha;

  // Check if the user set the proper parameters
  if (mLayerRadius <= 0) {
    LOG(FATAL) << "Wrong layer radius " << mLayerRadius << FairLogger::endl;
  }

  if (mZLength <= 0) {
    LOG(FATAL) << "Wrong layer length " << mZLength << FairLogger::endl;
  }

  if (mNumberOfStaves <= 0) {
    LOG(FATAL) << "Wrong number of staves " << mNumberOfStaves << FairLogger::endl;
  }

  if (mNumberOfChips <= 0) {
    LOG(FATAL) << "Wrong number of chips " << mNumberOfChips << FairLogger::endl;
  }

  if (mLayerNumber >= sNumberOfInnerLayers && mNumberOfModules <= 0) {
    LOG(FATAL) << "Wrong number of modules " << mNumberOfModules << FairLogger::endl;
  }

  if (mChipThickness <= 0) {
    LOG(INFO) << "Chip thickness wrong or not set " << mChipThickness << " using default "
              << sDefaultChipThick << FairLogger::endl;
    mChipThickness = sDefaultChipThick;
  }

  if (mSensorThickness <= 0) {
    LOG(INFO) << "Sensor thickness wrong or not set " << mSensorThickness << " using default "
              << sDefaultSensorThick << FairLogger::endl;
    mSensorThickness = sDefaultSensorThick;
  }

  if (mSensorThickness > mChipThickness) {
    LOG(WARNING) << "Sensor thickness " << mSensorThickness << " is greater than chip thickness "
                 << mChipThickness << " fixing" << FairLogger::endl;
    mSensorThickness = mChipThickness;
  }

  // If a Turbo layer is requested, do it and exit
  if (mIsTurbo) {
    createLayerTurbo(motherVolume);
    return;
  }

  // First create the stave container
  alpha = (360. / (2 * mNumberOfStaves)) * DegToRad();

  //  mStaveWidth = mLayerRadius*Tan(alpha);

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSLayerPattern(), mLayerNumber);
  TGeoVolume *layerVolume = new TGeoVolumeAssembly(volumeName);
  layerVolume->SetUniqueID(mChipTypeID);

  // layerVolume->SetVisibility(kFALSE);
  layerVolume->SetVisibility(kTRUE);
  layerVolume->SetLineColor(1);

  TGeoVolume *stavVol = createStave();

  // Now build up the layer
  alpha = 360. / mNumberOfStaves;
  Double_t r = mLayerRadius + ((TGeoBBox *) stavVol->GetShape())->GetDY();
  for (Int_t j = 0; j < mNumberOfStaves; j++) {
    Double_t phi = j * alpha + mPhi0;
    xpos = r * cosD(phi); // r*sinD(-phi);
    ypos = r * sinD(phi); // r*cosD(-phi);
    zpos = 0.;
    phi += 90;
    layerVolume->AddNode(stavVol, j,
                         new TGeoCombiTrans(xpos, ypos, zpos, new TGeoRotation("", phi, 0, 0)));
  }

  // Finally put everything in the mother volume
  motherVolume->AddNode(layerVolume, 1, nullptr);

  //  geometry is served
  return;
}

void V3Layer::createLayerTurbo(TGeoVolume *motherVolume)
{
  char volumeName[30];
  Double_t xpos, ypos, zpos;
  Double_t alpha;

  // Check if the user set the proper (remaining) parameters
  if (mStaveWidth <= 0) {
    LOG(FATAL) << "Wrong stave width " << mStaveWidth << FairLogger::endl;
  }

  if (Abs(mStaveTilt) > 45) {
    LOG(WARNING) << "Stave tilt angle (" << mStaveTilt << ") greater than 45deg"
                 << FairLogger::endl;
  }

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSLayerPattern(), mLayerNumber);
  TGeoVolume *layerVolume = new TGeoVolumeAssembly(volumeName);
  layerVolume->SetUniqueID(mChipTypeID);
  layerVolume->SetVisibility(kTRUE);
  layerVolume->SetLineColor(1);
  TGeoVolume *stavVol = createStave();

  // Now build up the layer
  alpha = 360. / mNumberOfStaves;
  Double_t r = mLayerRadius /* +chip thick ?! */;
  for (Int_t j = 0; j < mNumberOfStaves; j++) {
    Double_t phi = j * alpha + mPhi0;
    xpos = r * cosD(phi); // r*sinD(-phi);
    ypos = r * sinD(phi); // r*cosD(-phi);
    zpos = 0.;
    phi += 90;
    layerVolume->AddNode(
      stavVol, j,
      new TGeoCombiTrans(xpos, ypos, zpos, new TGeoRotation("", phi - mStaveTilt, 0, 0)));
  }

  // Finally put everything in the mother volume
  motherVolume->AddNode(layerVolume, 1, nullptr);

  return;
}

TGeoVolume *V3Layer::createStave(const TGeoManager * /*mgr*/)
{
//
// Creates the actual Stave
//
// Input:
//         mgr  : the GeoManager (used only to get the proper material)
//
// Output:
//
// Return:
//
// Created:      22 Jun 2011  Mario Sitta
// Updated:      18 Dec 2013  Mario Sitta  Handle IB and OB
// Updated:      12 Jan 2015  Mario Sitta  Fix overlap with new OB space frame
//                            (by moving the latter, not the sensors to avoid
//                             spoiling their position in space)
// Updated:      03 Mar 2015  Mario Sitta  Fix chip position
// Updated:      16 Mar 2017  Mario Sitta  AliceO2 version
//
  char volumeName[30];

  Double_t xlen, ylen, zlen;
  Double_t xpos, ypos;
  Double_t alpha;

  // First create all needed shapes
  alpha = (360. / (2 * mNumberOfStaves)) * DegToRad();

  // The stave
  xlen = mLayerRadius * Tan(alpha);
  if (mIsTurbo) {
    xlen = 0.5 * mStaveWidth;
  }
  ylen = 0.5 * mChipThickness;
  zlen = 0.5 * mZLength;

  Double_t yplus = 0.46;
  TGeoXtru *stave = new TGeoXtru(2); // z sections
  Double_t xv[5] = {xlen, xlen, 0, -xlen, -xlen};
  Double_t yv[5] = {ylen + 0.09, -0.15, -yplus - mSensorThickness, -0.15, ylen + 0.09};
  stave->DefinePolygon(5, xv, yv);
  stave->DefineSection(0, -zlen, 0, 0, 1.);
  stave->DefineSection(1, +zlen, 0, 0, 1.);

  // We have all shapes: now create the real volumes

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSStavePattern(), mLayerNumber);
  //  TGeoVolume *staveVol = new TGeoVolume(volumeName, stave, medAir);
  TGeoVolume *staveVol = new TGeoVolumeAssembly(volumeName);

  //  staveVol->SetVisibility(kFALSE);
  staveVol->SetVisibility(kTRUE);
  staveVol->SetLineColor(2);
  TGeoVolume *mechStaveVol = nullptr;

  // Now build up the stave
  if (mLayerNumber < sNumberOfInnerLayers) {
    TGeoVolume *modVol = createStaveInnerB(xlen, ylen, zlen);
    ypos = ((TGeoBBox*)(modVol->GetShape()))->GetDY() - mChipThickness; // = 0 if not kIBModel4
    staveVol->AddNode(modVol, 0, new TGeoTranslation(0, ypos, 0));
    mHierarchy[kHalfStave] = 1;

    // Mechanical stave structure
    mechStaveVol = createStaveStructInnerB(xlen, zlen);
    if (mechStaveVol) {
      ypos = ((TGeoBBox*)(modVol->GetShape()))->GetDY() - ypos;
      if (mStaveModel != Detector::kIBModel4)
	ypos += ((TGeoBBox*)(mechStaveVol->GetShape()))->GetDY();
      staveVol->AddNode(mechStaveVol, 1,
                        new TGeoCombiTrans(0, -ypos, 0, new TGeoRotation("", 0, 0, 180)));
    }
  } else {
    TGeoVolume *hstaveVol = createStaveOuterB();
    if (mStaveModel == Detector::kOBModel0) { // Create simplified stave struct as in v0
      staveVol->AddNode(hstaveVol, 0);
      mHierarchy[kHalfStave] = 1;
    } else { // (if mStaveModel) Create new stave struct as in TDR
      xpos = ((TGeoBBox *) (hstaveVol->GetShape()))->GetDX() - sOBHalfStaveXOverlap / 2;
      // ypos is CF height as computed in createSpaceFrameOuterB1
      ypos = (sOBSpaceFrameTotHigh - sOBHalfStaveYTrans) / 2;
      staveVol->AddNode(hstaveVol, 0, new TGeoTranslation(-xpos, ypos, 0));
      staveVol->AddNode(hstaveVol, 1, new TGeoTranslation(xpos, ypos + sOBHalfStaveYTrans, 0));
      mHierarchy[kHalfStave] = 2; // RS
      mechStaveVol = createSpaceFrameOuterB();

      if (mechStaveVol) {
        staveVol->AddNode(mechStaveVol, 1,
                          new TGeoCombiTrans(0, 0, 0, new TGeoRotation("", 180, 0, 0)));
      }
    }
  }
  // Done, return the stave
  return staveVol;
}

TGeoVolume *V3Layer::createStaveInnerB(const Double_t xsta, const Double_t ysta,
                                              const Double_t zsta, const TGeoManager *mgr)
{
  Double_t xmod, ymod, zmod;
  char volumeName[30];

  // First we create the module (i.e. the HIC with 9 chips)
  TGeoVolume *moduleVol = createModuleInnerB(xsta, ysta, zsta);

  // Then we create the fake halfstave and the actual stave
  xmod = ((TGeoBBox *) (moduleVol->GetShape()))->GetDX();
  ymod = ((TGeoBBox *) (moduleVol->GetShape()))->GetDY();
  zmod = ((TGeoBBox *) (moduleVol->GetShape()))->GetDZ();

  TGeoBBox *hstave = new TGeoBBox(xmod, ymod, zmod);

  TGeoMedium *medAir = mgr->GetMedium("ITS_AIR$");

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSHalfStavePattern(), mLayerNumber);
  TGeoVolume *hstaveVol = new TGeoVolume(volumeName, hstave, medAir);

  // Finally build it up
  hstaveVol->AddNode(moduleVol, 0);
  mHierarchy[kModule] = 1;

  // Done, return the stave structure
  return hstaveVol;
}

TGeoVolume *V3Layer::createModuleInnerB(Double_t xmod, Double_t ymod, Double_t zmod,
                                               const TGeoManager *mgr)
{
  Double_t ytot, zchip;
  Double_t ypos, zpos;
  char volumeName[30];

  // First create the single chip
  zchip = zmod / sIBChipsPerRow;
  TGeoVolume *chipVol = createChipInnerB(xmod, ymod, zchip);

  // Then create the module and populate it with the chips
  // (and the FPC Kapton and Aluminum in the most recent IB model)
  ytot = ymod;
  if (mStaveModel == Detector::kIBModel4)
    ytot += 0.5*(sIBFlexCableKapThick + sIBFlexCableAlThick);

  TGeoBBox *module = new TGeoBBox(xmod, ytot, zmod);

  TGeoBBox *kapCable = new TGeoBBox(xmod, sIBFlexCableKapThick/2, zmod);
  TGeoBBox *aluCable = new TGeoBBox(xmod, sIBFlexCableAlThick /2, zmod);

  TGeoMedium *medAir      = mgr->GetMedium("ITS_AIR$");
  TGeoMedium *medKapton   = mgr->GetMedium("ITS_KAPTON(POLYCH2)$");
  TGeoMedium *medAluminum = mgr->GetMedium("ITS_ALUMINUM$");

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSModulePattern(), mLayerNumber);
  TGeoVolume *modVol = new TGeoVolume(volumeName, module, medAir);

  TGeoVolume *kapCableVol = new TGeoVolume("FPCKapton", kapCable, medKapton);
  kapCableVol->SetLineColor(kBlue);
  kapCableVol->SetFillColor(kBlue);

  TGeoVolume *aluCableVol = new TGeoVolume("FPCAluminum",
					   aluCable, medAluminum);
  aluCableVol->SetLineColor(kCyan);
  aluCableVol->SetFillColor(kCyan);

  // mm (not used)  zlen = ((TGeoBBox*)chipVol->GetShape())->GetDZ();
  ypos = -ytot + ymod; // = 0 if not kIBModel4
  for (Int_t j = 0; j < sIBChipsPerRow; j++) {
    zpos = -zmod + j * 2 * zchip + zchip;
    modVol->AddNode(chipVol, j, new TGeoTranslation(0, ypos, zpos));
    mHierarchy[kChip]++;
  }

  if (mStaveModel == Detector::kIBModel4) {
    ypos += (ymod + aluCable->GetDY());
    if (mBuildLevel < 1)   // Aluminum
      modVol->AddNode(aluCableVol, 1, new TGeoTranslation(0, ypos, 0));

    ypos += (aluCable->GetDY() + kapCable->GetDY());
    if (mBuildLevel < 4)   // Kapton
      modVol->AddNode(kapCableVol, 1, new TGeoTranslation(0, ypos, 0));
  }

  // Done, return the module
  return modVol;
}

TGeoVolume *V3Layer::createStaveStructInnerB(const Double_t xsta,
					     const Double_t zsta,
					     const TGeoManager *mgr)
{
//
// Create the mechanical stave structure
//
// Created:      22 Mar 2013  Chinorat Kobdaj
// Updated:      26 Apr 2013  Mario Sitta
// Updated:      04 Apr 2017  Mario Sitta  O2 version - All models obsolete except last one
//

  TGeoVolume *mechStavVol = nullptr;

  switch (mStaveModel) {
    case Detector::kIBModelDummy:
      mechStavVol = createStaveModelInnerBDummy(xsta, zsta, mgr);
      break;
    case Detector::kIBModel0:
    case Detector::kIBModel1:
    case Detector::kIBModel21:
    case Detector::kIBModel22:
    case Detector::kIBModel3:
      LOG(FATAL) << "Stave model " << mStaveModel << " obsolete and no longer supported" << FairLogger::endl;
      break;
    case Detector::kIBModel4:
      mechStavVol = createStaveModelInnerB4(xsta, zsta, mgr);
      break;
    default:
      LOG(FATAL) << "Unknown stave model " << mStaveModel << FairLogger::endl;
      break;
  }
  return mechStavVol;
}

TGeoVolume *V3Layer::createStaveModelInnerBDummy(const Double_t,
						 const Double_t,
						 const TGeoManager *) const
{
//
// Create dummy stave
//
// Created:      22 Mar 2013  Chinorat Kobdaj
// Updated:      26 Apr 2013  Mario Sitta
// Updated:      04 Apr 2017  Mario Sitta  O2 version
//

  // Done, return the stave structur
  return nullptr;
}

// model4
//________________________________________________________________________
TGeoVolume* V3Layer::createStaveModelInnerB4(const Double_t xstave,
					     const Double_t zstave,
					     const TGeoManager *mgr)
{
//
// Create the mechanical stave structure for Model 4 of TDR
//
// Input:
//         xstave : stave X half length
//         zstave : stave Z half length
//         mgr    : the GeoManager (used only to get the proper material)
//
// Output:
//
// Return:
//
// Created:      04 Dec 2014  Mario Sitta
// Updated:      03 Mar 2015  Mario Sitta  FPC in right position (beyond chip)
// Updated:      06 Mar 2015  Mario Sitta  Space Frame corrected (C.G. data)
// Updated:      30 Apr 2015  Mario Sitta  End-stave connectors added
// Updated:      04 Apr 2017  Mario Sitta  O2 version
//

  
  // Local parameters
  Double_t layerHeight = 0.;

  Double_t rPipeMin = sIBCoolPipeInnerD/2;
  Double_t rPipeMax = rPipeMin + sIBCoolPipeThick;

  Double_t topFilTheta = sIBTopFilamentAlpha*TMath::DegToRad();
  Double_t topFilLProj = xstave/TMath::Sin(topFilTheta); // Top filament length projected on stave XZ plane
  Double_t topFilYLen = xstave/TMath::Tan(topFilTheta); // Filament length on Y
  Int_t  nFilaments = (Int_t)(zstave/topFilYLen);
  // Question: would it be better to fix the number of filaments and
  // compute the angle alpha from it, or leave as it is now, i.e. fix the
  // filament inclination angle alpha and compute their number ?

  const Int_t nv = 6;
  Double_t xv[nv], yv[nv]; // The stave container Xtru
  Double_t xlen, ylen, zlen;
  Double_t xpos, ypos, zpos, ylay;
  Double_t beta, gamma, theta;


  // First create all needed shapes
  TGeoBBox *glue     = new TGeoBBox(xstave, sIBGlueThick/2, zstave);

  TGeoBBox *fleecbot = new TGeoBBox(xstave, sIBCarbonFleeceThick/2, zstave);

  TGeoBBox *cfplate  = new TGeoBBox(xstave, sIBK13D2UThick/2, zstave);

  TGeoTube *pipe     = new TGeoTube(rPipeMin, rPipeMax, zstave);

  TGeoTube *water    = new TGeoTube(0., rPipeMin, zstave);

  TGeoTubeSeg *cpaptub  = new TGeoTubeSeg(rPipeMax,
					  rPipeMax + sIBCarbonPaperThick,
					  zstave, 0, 180);

  TGeoBBox *cpapvert = new TGeoBBox(sIBCarbonPaperThick/2,
				    pipe->GetRmax()/2, zstave);

  xlen = sIBCoolPipeXDist/2 - pipe->GetRmax() - sIBCarbonPaperThick;
  TGeoBBox *cpapmid  = new TGeoBBox(xlen, sIBCarbonPaperThick/2, zstave);

  xlen = xstave -sIBCoolPipeXDist/2 -pipe->GetRmax() - sIBCarbonPaperThick;
  TGeoBBox *cpaplr   = new TGeoBBox(xlen/2, sIBCarbonPaperThick/2, zstave);

  TGeoTubeSeg *fleecpipe = new TGeoTubeSeg(cpaptub->GetRmax(),
			       cpaptub->GetRmax() + sIBCarbonFleeceThick,
					   zstave, 0, 180); 

  TGeoBBox *fleecvert = new TGeoBBox(sIBCarbonFleeceThick/2,
			 	     (pipe->GetRmax() - sIBCarbonPaperThick)/2,
				     zstave);

  xlen = sIBCoolPipeXDist/2 - pipe->GetRmax() - sIBCarbonPaperThick
       - sIBCarbonFleeceThick;
  TGeoBBox *fleecmid  = new TGeoBBox(xlen, sIBCarbonFleeceThick/2, zstave);

  xlen = xstave - sIBCoolPipeXDist/2 - pipe->GetRmax()
       - sIBCarbonPaperThick - sIBCarbonFleeceThick;
  TGeoBBox *fleeclr   = new TGeoBBox(xlen/2, sIBCarbonFleeceThick/2, zstave);

  // The spaceframe structure
  TGeoTrd1 *topv  = new TGeoTrd1(sIBTopVertexWidth1/2,
				 sIBTopVertexWidth2/2, zstave,
				 sIBTopVertexHeight/2);

  xv[0] = 0;
  yv[0] = 0;
  xv[1] = sIBSideVertexWidth;
  yv[1] = yv[0];
  xv[2] = xv[0];
  yv[2] = sIBSideVertexHeight;

  TGeoXtru *sidev = new TGeoXtru(2);
  sidev->DefinePolygon(3, xv, yv);
  sidev->DefineSection(0,-zstave);
  sidev->DefineSection(1, zstave);

  TGeoBBox *topfil = new TGeoBBox(sIBTopFilamentLength/2,
				  sIBTopFilamentSide/2,
				  sIBTopFilamentSide/2);

  // The half stave container (an XTru to avoid overlaps between neighbours)
  layerHeight = 2*(    glue->GetDY() + fleecbot->GetDY() + cfplate->GetDY()
                   + cpaplr->GetDY() +  fleeclr->GetDY() );

  xv[0] = xstave;
  yv[0] = 0;
  xv[1] = xv[0];
  yv[1] = layerHeight + sIBSideVertexHeight + topfil->GetDZ();;
  xv[2] = sIBTopVertexWidth2/2;
  yv[2] = sIBStaveHeight;
  for (Int_t i = 0; i<nv/2; i++) {
    xv[3+i] = -xv[2-i];
    yv[3+i] =  yv[2-i];
  }

  TGeoXtru *mechStruct = new TGeoXtru(2);
  mechStruct->DefinePolygon(nv, xv, yv);
  mechStruct->SetName("mechStruct");
  mechStruct->DefineSection(0,-zstave);
  mechStruct->DefineSection(1, zstave);

  // The connectors' containers
  zlen = sIBConnectBlockZLen - sIBConnTailZLen + sIBConnectAFitZOut;
  TGeoBBox *connAside = new TGeoBBox("connAsideIB", sIBConnectorXWidth/2,
				     sIBConnectorYTot/2, zlen/2);

  zlen = sIBConnectBlockZLen - sIBConnTailZLen;
  TGeoBBox *connCside = new TGeoBBox("connCsideIB", sIBConnectorXWidth/2,
				     sIBConnectorYTot/2, zlen/2);

  // The StaveStruct container, a Composite Shape
  ypos = connAside->GetDY() - sIBConnTailYShift + layerHeight;
  zpos = zstave + connAside->GetDZ();
  TGeoTranslation *transAside = new TGeoTranslation("transAsideIB",
						    0, ypos, zpos);
  transAside->RegisterYourself();

  ypos = connCside->GetDY() - sIBConnTailYShift + layerHeight;
  zpos = zstave + connCside->GetDZ();
  TGeoTranslation *transCside = new TGeoTranslation("transCsideIB",
						    0, ypos,-zpos);
  transCside->RegisterYourself();

  TGeoCompositeShape *mechStavSh = new TGeoCompositeShape(
	  "mechStruct+connAsideIB:transAsideIB+connCsideIB:transCsideIB");


  // We have all shapes: now create the real volumes

  TGeoMedium *medAir          = mgr->GetMedium("ITS_AIR$");
  TGeoMedium *medWater        = mgr->GetMedium("ITS_WATER$");
  TGeoMedium *medM55J6K       = mgr->GetMedium("ITS_M55J6K$"); 
  TGeoMedium *medM60J3K       = mgr->GetMedium("ITS_M60J3K$"); 
  TGeoMedium *medKapton       = mgr->GetMedium("ITS_KAPTON(POLYCH2)$");
  TGeoMedium *medGlue         = mgr->GetMedium("ITS_GLUE$");
  TGeoMedium *medK13D2U2k     = mgr->GetMedium("ITS_K13D2U2k$");
  TGeoMedium *medFGS003       = mgr->GetMedium("ITS_FGS003$"); 
  TGeoMedium *medCarbonFleece = mgr->GetMedium("ITS_CarbonFleece$"); 


  char volname[30];
  snprintf(volname, 30, "%s%d_StaveStruct",
	   GeometryTGeo::getITSStavePattern(), mLayerNumber);
  TGeoVolume *mechStavVol = new TGeoVolume(volname, mechStavSh, medAir);
  mechStavVol->SetLineColor(12);
  mechStavVol->SetFillColor(12); 
  mechStavVol->SetVisibility(kFALSE);

  TGeoVolume *glueVol = new TGeoVolume("Glue", glue, medGlue);
  glueVol->SetLineColor(kBlack);
  glueVol->SetFillColor(kBlack);

  TGeoVolume *fleecbotVol = new TGeoVolume("CarbonFleeceBottom",
					   fleecbot, medCarbonFleece);
  fleecbotVol->SetFillColor(kViolet);
  fleecbotVol->SetLineColor(kViolet);

  TGeoVolume *cfplateVol = new TGeoVolume("CFPlate", cfplate, medK13D2U2k);
  cfplateVol->SetFillColor(5);  // Yellow
  cfplateVol->SetLineColor(5);

  TGeoVolume *pipeVol = new TGeoVolume("PolyimidePipe", pipe, medKapton);
  pipeVol->SetFillColor(35);  // Blue shade
  pipeVol->SetLineColor(35);

  TGeoVolume *waterVol= new TGeoVolume("Water", water, medWater);
  waterVol->SetFillColor(4);  // Bright blue
  waterVol->SetLineColor(4);

  TGeoVolume *cpaptubVol = new TGeoVolume("ThermasolPipeCover",
					  cpaptub, medFGS003);
  cpaptubVol->SetFillColor(2);  // Red
  cpaptubVol->SetLineColor(2);

  TGeoVolume *cpapvertVol = new TGeoVolume("ThermasolVertical",
					   cpapvert, medFGS003);
  cpapvertVol->SetFillColor(2);  // Red
  cpapvertVol->SetLineColor(2);

  TGeoVolume *cpapmidVol = new TGeoVolume("ThermasolMiddle",
					  cpapmid, medFGS003);
  cpapmidVol->SetFillColor(2);  // Red
  cpapmidVol->SetLineColor(2);

  TGeoVolume *cpaplrVol = new TGeoVolume("ThermasolLeftRight",
					 cpaplr, medFGS003);
  cpaplrVol->SetFillColor(2);  // Red
  cpaplrVol->SetLineColor(2);

  TGeoVolume *fleecpipeVol = new TGeoVolume("CarbonFleecePipeCover",
					    fleecpipe, medCarbonFleece);
  fleecpipeVol->SetFillColor(28);  // Brown shade
  fleecpipeVol->SetLineColor(28);

  TGeoVolume *fleecvertVol = new TGeoVolume("CarbonFleeceVertical",
					    fleecvert, medCarbonFleece);
  fleecvertVol->SetFillColor(28);  // Brown shade
  fleecvertVol->SetLineColor(28);

  TGeoVolume *fleecmidVol = new TGeoVolume("CarbonFleeceMiddle",
					   fleecmid, medCarbonFleece);
  fleecmidVol->SetFillColor(28);  // Brown shade
  fleecmidVol->SetLineColor(28);

  TGeoVolume *fleeclrVol = new TGeoVolume("CarbonFleeceLeftRight",
					  fleeclr, medCarbonFleece);
  fleeclrVol->SetFillColor(28);  // Brown shade
  fleeclrVol->SetLineColor(28);

  TGeoVolume *topvVol = new TGeoVolume("TopVertex", topv, medM55J6K);
  topvVol->SetFillColor(12);  // Gray shade
  topvVol->SetLineColor(12);
  
  TGeoVolume *sidevVol = new TGeoVolume("SideVertex", sidev, medM55J6K);
  sidevVol->SetFillColor(12);  // Gray shade
  sidevVol->SetLineColor(12);
  
  TGeoVolume *topfilVol = new TGeoVolume("TopFilament", topfil, medM60J3K);
  topfilVol->SetFillColor(12);  // Gray shade
  topfilVol->SetLineColor(12);
  

  // Now build up the half stave
  ypos = glue->GetDY();
  if (mBuildLevel < 2)   // Glue
    mechStavVol->AddNode(glueVol, 1, new TGeoTranslation(0, ypos, 0));

  ypos += (glue->GetDY() + fleecbot->GetDY());
  if (mBuildLevel < 5)   // Carbon
    mechStavVol->AddNode(fleecbotVol, 1, new TGeoTranslation(0, ypos, 0));

  ypos += (fleecbot->GetDY() + cfplate->GetDY());
  if (mBuildLevel < 5)   // Carbon
    mechStavVol->AddNode(cfplateVol, 1, new TGeoTranslation(0, ypos, 0));

  ylay = ypos + cfplate->GetDY(); // The level where tubes etc. lay

  xpos = sIBCoolPipeXDist/2;
  ypos = ylay + pipe->GetRmax();
  if (mBuildLevel < 4) { // Kapton
    mechStavVol->AddNode(pipeVol, 1, new TGeoTranslation(-xpos, ypos, 0));
    mechStavVol->AddNode(pipeVol, 2, new TGeoTranslation( xpos, ypos, 0));
  }

  if (mBuildLevel < 3) { // Water
    mechStavVol->AddNode(waterVol, 1, new TGeoTranslation(-xpos, ypos, 0));
    mechStavVol->AddNode(waterVol, 2, new TGeoTranslation( xpos, ypos, 0));
  }

  if (mBuildLevel < 5) { // Carbon (stave components)
    mechStavVol->AddNode(cpaptubVol, 1, new TGeoTranslation(-xpos, ypos, 0));
    mechStavVol->AddNode(cpaptubVol, 2, new TGeoTranslation( xpos, ypos, 0));

    mechStavVol->AddNode(fleecpipeVol,1, new TGeoTranslation(-xpos, ypos, 0));
    mechStavVol->AddNode(fleecpipeVol,2, new TGeoTranslation( xpos, ypos, 0));

    xpos = sIBCoolPipeXDist/2 - pipe->GetRmax() - cpapvert->GetDX();
    ypos = ylay + cpapvert->GetDY();
    mechStavVol->AddNode(cpapvertVol, 1, new TGeoTranslation(-xpos, ypos, 0));
    mechStavVol->AddNode(cpapvertVol, 2, new TGeoTranslation( xpos, ypos, 0));

    xpos = sIBCoolPipeXDist/2 + pipe->GetRmax() + cpapvert->GetDX();
    mechStavVol->AddNode(cpapvertVol, 3, new TGeoTranslation(-xpos, ypos, 0));
    mechStavVol->AddNode(cpapvertVol, 4, new TGeoTranslation( xpos, ypos, 0));

    ypos = ylay + sIBCarbonPaperThick/2;
    mechStavVol->AddNode(cpapmidVol, 1, new TGeoTranslation(0, ypos, 0));

    xpos = xstave - cpaplr->GetDX();
    mechStavVol->AddNode(cpaplrVol, 1, new TGeoTranslation(-xpos, ypos, 0));
    mechStavVol->AddNode(cpaplrVol, 2, new TGeoTranslation( xpos, ypos, 0));

    xpos = sIBCoolPipeXDist/2 - pipe->GetRmax() - 2*cpapvert->GetDX()
         - fleecvert->GetDX();
    ypos = ylay + sIBCarbonPaperThick + fleecvert->GetDY();
    mechStavVol->AddNode(fleecvertVol, 1, new TGeoTranslation(-xpos, ypos, 0));
    mechStavVol->AddNode(fleecvertVol, 2, new TGeoTranslation( xpos, ypos, 0));

    xpos = sIBCoolPipeXDist/2 + pipe->GetRmax() + 2*cpapvert->GetDX()
         + fleecvert->GetDX();
    mechStavVol->AddNode(fleecvertVol, 3, new TGeoTranslation(-xpos, ypos, 0));
    mechStavVol->AddNode(fleecvertVol, 4, new TGeoTranslation( xpos, ypos, 0));

    ypos = ylay + sIBCarbonPaperThick + sIBCarbonFleeceThick/2;
    mechStavVol->AddNode(fleecmidVol, 1, new TGeoTranslation(0, ypos, 0));

    xpos = xstave - fleeclr->GetDX();
    mechStavVol->AddNode(fleeclrVol, 1, new TGeoTranslation(-xpos, ypos, 0));
    mechStavVol->AddNode(fleeclrVol, 2, new TGeoTranslation( xpos, ypos, 0));
  }

  ylay += (sIBCarbonPaperThick + sIBCarbonFleeceThick);

  if (mBuildLevel < 5) { // Carbon (spaceframe)
    ypos = sIBStaveHeight - sIBTopFilamentSide - topv->GetDz(); // Due to rotation, z is on Y
    mechStavVol->AddNode(topvVol, 1,
			 new TGeoCombiTrans(0, ypos, 0,
					    new TGeoRotation("",0,-90,0)));

    xpos = xstave - sidev->GetX(1);
    ypos = ylay;
    mechStavVol->AddNode(sidevVol, 1, new TGeoTranslation( xpos, ypos, 0));
    mechStavVol->AddNode(sidevVol, 2, new TGeoCombiTrans(-xpos, ypos, 0,
					    new TGeoRotation("",90,180,-90)));

    gamma = sIBTopFilamentGamma;
    theta = 90. - sIBTopFilamentAlpha;
    xpos = xstave/2 + topfil->GetDZ();
    ypos = ( layerHeight + sIBStaveHeight )/2 +
	   sIBSideVertexWidth/TMath::Sin(gamma*TMath::DegToRad())/2 ;
    for(int i=0; i<nFilaments; i++){ // i<28 (?)
      // 1) Front Left Top Filament
//      zpos = -zstave + (i*2*topFilYLen) + topFilLProj/4; // ?????
//      zpos = -zstave + (i*2*topFilYLen) + topFilLProj/2;
      zpos = -zstave + (i*2*topFilYLen) + topFilLProj/4 + topfil->GetDY();
      mechStavVol->AddNode(topfilVol, i*4+1,
			 new TGeoCombiTrans( xpos, ypos, zpos,
			      new TGeoRotation("", 90, theta, gamma)));
      // 2) Front Right Top Filament
      mechStavVol->AddNode(topfilVol, i*4+2,
			 new TGeoCombiTrans(-xpos, ypos, zpos,
			      new TGeoRotation("", 90,-theta,-gamma)));
      // 3) Back Left  Top Filament
      zpos += topFilYLen;
      mechStavVol->AddNode(topfilVol, i*4+3,
			 new TGeoCombiTrans( xpos, ypos, zpos,
			      new TGeoRotation("", 90,-theta, gamma)));
      // 4) Back Right Top Filament
      mechStavVol->AddNode(topfilVol, i*4+4,
			 new TGeoCombiTrans(-xpos, ypos, zpos,
			      new TGeoRotation("", 90, theta,-gamma)));
    }
  }


  // Add the end-stave connectors
  TGeoVolume *connectorASide, *connectorCSide;

  // Check whether we have already all pieces
  // Otherwise create them
  connectorASide = mgr->GetVolume("IBConnectorASide");

  if (!connectorASide) {
    CreateIBConnectors(mgr);
    connectorASide = mgr->GetVolume("IBConnectorASide");
  }
  connectorCSide = mgr->GetVolume("IBConnectorCSide");

  ypos = ((TGeoBBox*)connectorASide->GetShape())->GetDY()
       - sIBConnTailYShift + ylay;
  zpos = zstave +
        (sIBConnectBlockZLen - sIBConnTailZLen + sIBConnectAFitZOut)/2;
  mechStavVol->AddNode(connectorASide, 1, new TGeoTranslation(0, ypos, zpos));

  zpos = zstave + (sIBConnectBlockZLen - sIBConnTailZLen)/2;
  mechStavVol->AddNode(connectorCSide, 1, new TGeoCombiTrans(0, ypos,-zpos,
					     new TGeoRotation("",90,180,-90)));


  // Done, return the stave structure
  return mechStavVol;
}

TGeoVolume *V3Layer::createStaveOuterB(const TGeoManager *mgr)
{
  TGeoVolume *mechStavVol = nullptr;

  switch (mStaveModel) {
    case Detector::kOBModelDummy:
      mechStavVol = createStaveModelOuterBDummy(mgr);
      break;
    case Detector::kOBModel0:
      mechStavVol = createStaveModelOuterB0(mgr);
      break;
    case Detector::kOBModel1:
      mechStavVol = createStaveModelOuterB1(mgr);
      break;
    default:
      LOG(FATAL) << "Unknown stave model " << mStaveModel << FairLogger::endl;
      break;
  }
  return mechStavVol;
}

void V3Layer::CreateIBConnectors(const TGeoManager *mgr)
{
//
// Create the end-stave connectors for IB staves
// (simply call the actual creator methods)
//
// Created:      20 Apr 2015  Mario Sitta
//

  CreateIBConnectorsASide(mgr);
  CreateIBConnectorsCSide(mgr);
}

void V3Layer::CreateIBConnectorsASide(const TGeoManager *mgr)
{
//
// Create the A-Side end-stave connectors for IB staves
//
// Created:      22 Apr 2015  Mario Sitta
// Updated:      04 Apr 2017  Mario Sitta  O2 version
//

  // Local variables
  const Int_t nv = 8;
  Double_t xv[nv], yv[nv];
  Double_t xlen, ylen, zlen;
  Double_t xpos, ypos, zpos;
  char volname[30];


  // Gather all material pointers
  TGeoMedium *medAir      = mgr->GetMedium("ITS_AIR$");
  TGeoMedium *medPEEK     = mgr->GetMedium("ITS_PEEKCF30$");
  TGeoMedium *medWC       = mgr->GetMedium("ITS_TUNGCARB$");
  TGeoMedium *medInox304  = mgr->GetMedium("ITS_INOX304$");


  // First create all elements

  // The connector block, two Composite Shapes:
  // the body...
  xlen = sIBConnectorXWidth;
  ylen = sIBConnBodyYHeight;
  zlen = sIBConnectBlockZLen - sIBConnTailZLen;
  TGeoBBox *connBody = new TGeoBBox("connBodyA", xlen/2, ylen/2, zlen/2);


  TGeoTube *connRoundHole = new TGeoTube("connRoundHoleA", 0.,
					 sIBConnRoundHoleD /2,
					 sIBConnBodyYHeight/1.5);

  zpos = -connBody->GetDZ() + sIBConnRoundHoleZ;
  TGeoCombiTrans *connRoundHoleTrans = new TGeoCombiTrans ("roundHoleTransA",
					 0, 0, zpos,
					   new TGeoRotation("",0,90,0));
  connRoundHoleTrans->RegisterYourself();


  xlen = sIBConnSquareHoleX/2;
  ylen = sIBConnBodyYHeight/1.5;
  zlen = sIBConnSquareHoleZ/2;
  TGeoBBox *connSquareHole = new TGeoBBox("connSquareHoleA", xlen, ylen, zlen);

  zpos = -connBody->GetDZ() + sIBConnSquareHoleZPos;
  TGeoTranslation *connSquareHoleTrans = new TGeoTranslation(
					 "squareHoleTransA", 0, 0, zpos);
  connSquareHoleTrans->RegisterYourself();


  TGeoTube *connTubeHole2 = new TGeoTube("tube2HoleA", 0, sIBConnTubeHole2D/2, connBody->GetDZ());

  xpos = sIBConnTubesXDist/2;
  ypos = -connBody->GetDY() + sIBConnTubesYPos;
  
  TGeoTranslation *connTubes2Trans1 = new TGeoTranslation("tubes2Trans1A",-xpos, ypos, 0);
  connTubes2Trans1->RegisterYourself();
  
  TGeoTranslation *connTubes2Trans2 = new TGeoTranslation("tubes2Trans2A",
							  xpos, ypos, 0);
  connTubes2Trans2->RegisterYourself();


  zlen = sIBConnTubeHole1ZLen - sIBConnTailZLen;
  TGeoTube *connTubeHole3 = new TGeoTube("tube3HoleA", 0, sIBConnTubeHole1D/2, zlen);

  zpos = connBody->GetDZ();
  TGeoTranslation *connTubes3Trans1 = new TGeoTranslation("tubes3Trans1A",
							 -xpos, ypos,-zpos);
  connTubes3Trans1->RegisterYourself();
  TGeoTranslation *connTubes3Trans2 = new TGeoTranslation("tubes3Trans2A",
							  xpos, ypos,-zpos);
  connTubes3Trans2->RegisterYourself();

  zlen = sIBConnectAFitZLen - sIBConnectAFitZOut;
  TGeoTube *connFitHole = new TGeoTube("fitHoleA", 0, sIBConnectAFitExtD/2, zlen);

  TGeoTranslation *connFitHoleTrans1 = new TGeoTranslation("fitTrans1A",
							  -xpos, ypos, zpos);
  connFitHoleTrans1->RegisterYourself();
  TGeoTranslation *connFitHoleTrans2 = new TGeoTranslation("fitTrans2A",
							   xpos, ypos, zpos);
  connFitHoleTrans2->RegisterYourself();


  TGeoCompositeShape *connBodySh = new TGeoCompositeShape(
   "connBodyA-connRoundHoleA:roundHoleTransA-connSquareHoleA:squareHoleTransA-tube2HoleA:tubes2Trans1A-tube2HoleA:tubes2Trans2A-fitHoleA:fitTrans1A-fitHoleA:fitTrans2A-tube3HoleA:tubes3Trans1A-tube3HoleA:tubes3Trans2A");


  TGeoVolume *connBlockBody = new TGeoVolume("IBConnectorBlockBodyASide",
					     connBodySh,medPEEK);
  connBlockBody->SetFillColor(42);  // Brownish shade
  connBlockBody->SetLineColor(42);

  // ...and the tail
  xv[0] = sIBConnectorXWidth/2;
  yv[0] = sIBConnTailYShift;
  xv[1] = xv[0];
  yv[1] = sIBConnTailYMid;
  xv[2] = xv[1] -
      (sIBConnectorYTot - sIBConnTailYMid)/tanD(90 - sIBConnTailOpenPhi/2);
  yv[2] = sIBConnectorYTot;

  for (Int_t i = 0; i<3; i++) {
    xv[3+i] = -xv[2-i];
    yv[3+i] =  yv[2-i];
  }

  TGeoXtru *connTail = new TGeoXtru(2);
  connTail->SetName("connTailA");
  connTail->DefinePolygon(6, xv, yv);
  connTail->DefineSection(0, 0);
  connTail->DefineSection(1, sIBConnTailZLen);


  TGeoTube *connTubeHole1 = new TGeoTube("tube1HoleA", 0,
					 sIBConnTubeHole1D/2,
					 sIBConnTubeHole1ZLen/1.5);

  xpos = sIBConnTubesXDist/2;
  ypos = sIBConnTubesYPos;
  zpos = connTail->GetZ(1)/2;
  TGeoTranslation *connTubes1Trans1 = new TGeoTranslation("tubes1Trans1A",
							  -xpos, ypos, zpos);
  connTubes1Trans1->RegisterYourself();
  TGeoTranslation *connTubes1Trans2 = new TGeoTranslation("tubes1Trans2A",
							   xpos, ypos, zpos);
  connTubes1Trans2->RegisterYourself();


  TGeoCompositeShape *connTailSh = new TGeoCompositeShape(
		"connTailA-tube1HoleA:tubes1Trans1A-tube1HoleA:tubes1Trans2A");


  TGeoVolume *connBlockTail = new TGeoVolume("IBConnectorBlockTailASide",
					     connTailSh,medPEEK);
  connBlockTail->SetFillColor(42);  // Brownish shade
  connBlockTail->SetLineColor(42);


  // The steel insert, an Xtru
  xv[0] = (sIBConnSquareHoleX - sIBConnInsertInnerX)/2;
  yv[0] =  sIBConnSquareHoleZ/2 - sIBConnInsertZThick;
  xv[1] = xv[0];
  yv[1] = -sIBConnSquareHoleZ/2;
  xv[2] =  sIBConnSquareHoleX/2;
  yv[2] = yv[1];
  xv[3] = xv[2];
  yv[3] =  sIBConnSquareHoleZ/2;

  for (Int_t i = 0; i<nv/2; i++) {
    xv[4+i] = -xv[3-i];
    yv[4+i] =  yv[3-i];
  }

  TGeoXtru *connInsertSh = new TGeoXtru(2);
  connInsertSh->DefinePolygon(nv, xv, yv);
  connInsertSh->DefineSection(0,-sIBConnInsertHeight/2);
  connInsertSh->DefineSection(1, sIBConnInsertHeight/2);

  TGeoVolume *connInsert = new TGeoVolume("IBConnectorInsertASide",
					  connInsertSh, medWC);
  connInsert->SetFillColor(kGray);
  connInsert->SetLineColor(kGray);


  // The fitting tubes, a Tube
  TGeoTube *connFitSh = new TGeoTube(sIBConnectAFitIntD/2,
				     sIBConnectAFitExtD/2,
				     sIBConnectAFitZLen/2);

  TGeoVolume *connFit = new TGeoVolume("IBConnectorFitting",
				       connFitSh, medInox304);
  connFit->SetFillColor(kGray);
  connFit->SetLineColor(kGray);


  // Now create the container: cannot be a simple box
  // to avoid fake overlaps with stave elements
  xlen = sIBConnectorXWidth;
  ylen = sIBConnectorYTot;
  zlen = sIBConnectBlockZLen - sIBConnTailZLen + sIBConnectAFitZOut;

  TGeoBBox *connBox = new TGeoBBox("connBoxA", xlen/2, ylen/2, zlen/2);

  ypos = -connBox->GetDY();
  zpos = -connBox->GetDZ() - connTail->GetZ(1);
  TGeoTranslation *transTailA = new TGeoTranslation("transTailA",
						     0, ypos, zpos);
  transTailA->RegisterYourself();

  TGeoTube *connTubeHollow = new TGeoTube("tubeHollowA", 0,
					 sIBConnTubeHole1D/2,
					 sIBConnTubeHole1ZLen/2);

  xpos = sIBConnTubesXDist/2;
  ypos = -connBox->GetDY() + sIBConnTubesYPos;
  zpos = -connBox->GetDZ() - connTail->GetZ(1) + sIBConnTubeHole1ZLen/2;
  TGeoTranslation *connTubeHollTrans1 = new TGeoTranslation("tubeHollTrans1A",
							    -xpos, ypos, zpos);
  connTubeHollTrans1->RegisterYourself();
  TGeoTranslation *connTubeHollTrans2 = new TGeoTranslation("tubeHollTrans2A",
							     xpos, ypos, zpos);
  connTubeHollTrans2->RegisterYourself();

  TGeoCompositeShape *connBoxSh = new TGeoCompositeShape(
      "connBoxA+connTailA:transTailA-tubeHollowA:tubeHollTrans1A-tubeHollowA:tubeHollTrans2A");

  TGeoVolume *connBoxASide = new TGeoVolume("IBConnectorASide",
					    connBoxSh, medAir);


  // Finally build up the connector
  // (NB: the origin is in the connBox, i.e. w/o the tail in Z)
  ypos = -connBox->GetDY();
  zpos = -connBox->GetDZ() - connTail->GetZ(1);
  connBoxASide->AddNode(connBlockTail, 1, new TGeoTranslation(0, ypos, zpos));

  ypos = -connBox->GetDY() + connBody->GetDY();
  zpos = -connBox->GetDZ() + connBody->GetDZ();
  connBoxASide->AddNode(connBlockBody, 1, new TGeoTranslation(0, ypos, zpos));

  zpos = -connBox->GetDZ() + sIBConnSquareHoleZPos;
  connBoxASide->AddNode(connInsert, 1, new TGeoCombiTrans(0, ypos, zpos,
					   new TGeoRotation("",0,-90,0)));

  xpos = sIBConnTubesXDist/2;
  ypos = -connBox->GetDY() + sIBConnTubesYPos;
  zpos =  connBox->GetDZ() - connFitSh->GetDz();
  connBoxASide->AddNode(connFit, 1, new TGeoTranslation( xpos, ypos, zpos));
  connBoxASide->AddNode(connFit, 2, new TGeoTranslation(-xpos, ypos, zpos));

}

void V3Layer::CreateIBConnectorsCSide(const TGeoManager *mgr)
{
//
// Create the C-Side end-stave connectors for IB staves
//
// Created:      05 May 2015  Mario Sitta
// Updated:      04 Apr 2017  Mario Sitta  O2 version
//

  // Local variables
  const Int_t nv = 8;
  Double_t xv[nv], yv[nv];
  Double_t xlen, ylen, zlen;
  Double_t xpos, ypos, zpos;
  char volname[30];


  // Gather all material pointers
  TGeoMedium *medAir      = mgr->GetMedium("ITS_AIR$");
  TGeoMedium *medPEEK     = mgr->GetMedium("ITS_PEEKCF30$");
  TGeoMedium *medWC       = mgr->GetMedium("ITS_TUNGCARB$");


  // First create all elements

  // The connector block, two Composite Shapes:
  // the body...
  xlen = sIBConnectorXWidth;
  ylen = sIBConnBodyYHeight;
  zlen = sIBConnectBlockZLen - sIBConnTailZLen;
  TGeoBBox *connBody = new TGeoBBox("connBodyC", xlen/2, ylen/2, zlen/2);


  TGeoTube *connRoundHole = new TGeoTube("connRoundHoleC", 0.,
					 sIBConnRoundHoleD /2,
					 sIBConnBodyYHeight/1.5);

  zpos = -connBody->GetDZ() + sIBConnRoundHoleZ;
  TGeoCombiTrans *connRoundHoleTrans = new TGeoCombiTrans ("roundHoleTransC",
					 0, 0, zpos,
					   new TGeoRotation("",0,90,0));
  connRoundHoleTrans->RegisterYourself();


  TGeoTube *connInsertHole = new TGeoTube("connInsertHoleC", 0,
					  sIBConnInsertHoleD/2,
					  sIBConnBodyYHeight/1.5);

  zpos = -connBody->GetDZ() + sIBConnInsertHoleZPos;
  TGeoCombiTrans *connInsertHoleTrans = new TGeoCombiTrans(
					   "insertHoleTransC", 0, 0, zpos,
					      new TGeoRotation("",0,90,0));
  connInsertHoleTrans->RegisterYourself();


  TGeoTube *connTubeHole2 = new TGeoTube("tube2HoleC", 0,
					 sIBConnTubeHole2D/2,
					 connBody->GetDZ());

  xpos = sIBConnTubesXDist/2;
  ypos = -connBody->GetDY() + sIBConnTubesYPos;
  zpos = sIBConnTubeHole3ZPos;
  TGeoTranslation *connTubes2Trans1 = new TGeoTranslation("tubes2Trans1C",
							  -xpos, ypos,-zpos);
  connTubes2Trans1->RegisterYourself();
  TGeoTranslation *connTubes2Trans2 = new TGeoTranslation("tubes2Trans2C",
							   xpos, ypos,-zpos);
  connTubes2Trans2->RegisterYourself();


  zlen = sIBConnectorXWidth;
  TGeoTube *connTubeHole3 = new TGeoTube("tube3HoleC", 0,
					 sIBConnTubeHole2D/2,
					 zlen/2);

  xpos = sIBConnTubeHole3XPos;
  zpos = connBody->GetDZ() - sIBConnTubeHole3ZPos;
  TGeoCombiTrans *connTubes3Trans = new TGeoCombiTrans("tubes3TransC",
						        xpos, ypos, zpos,
					       new TGeoRotation("",90,-90,90));
  connTubes3Trans->RegisterYourself();


  zlen = sIBConnTubeHole1ZLen - sIBConnTailZLen;
  TGeoTube *connTubeHole4 = new TGeoTube("tube4HoleC", 0,
					 sIBConnTubeHole1D/2,
					 zlen);

  xpos = sIBConnTubesXDist/2;
  zpos = connBody->GetDZ();
  TGeoTranslation *connTubes4Trans1 = new TGeoTranslation("tubes4Trans1C",
							  -xpos, ypos,-zpos);
  connTubes4Trans1->RegisterYourself();
  TGeoTranslation *connTubes4Trans2 = new TGeoTranslation("tubes4Trans2C",
							   xpos, ypos,-zpos);
  connTubes4Trans2->RegisterYourself();


  TGeoCompositeShape *connBodySh = new TGeoCompositeShape(
   "connBodyC-connRoundHoleC:roundHoleTransC-connInsertHoleC:insertHoleTransC-tube2HoleC:tubes2Trans1C-tube2HoleC:tubes2Trans2C-tube3HoleC:tubes3TransC-tube4HoleC:tubes4Trans1C-tube4HoleC:tubes4Trans2C");


  TGeoVolume *connBlockBody = new TGeoVolume("IBConnectorBlockBodyCSide",
					     connBodySh,medPEEK);
  connBlockBody->SetFillColor(42);  // Brownish shade
  connBlockBody->SetLineColor(42);

  // ...and the tail
  xv[0] = sIBConnectorXWidth/2;
  yv[0] = sIBConnTailYShift;
  xv[1] = xv[0];
  yv[1] = sIBConnTailYMid;
  xv[2] = xv[1] -
      (sIBConnectorYTot - sIBConnTailYMid)/tanD(90-sIBConnTailOpenPhi/2);
  yv[2] = sIBConnectorYTot;

  for (Int_t i = 0; i<3; i++) {
    xv[3+i] = -xv[2-i];
    yv[3+i] =  yv[2-i];
  }

  TGeoXtru *connTail = new TGeoXtru(2);
  connTail->SetName("connTailC");
  connTail->DefinePolygon(6, xv, yv);
  connTail->DefineSection(0, 0);
  connTail->DefineSection(1, sIBConnTailZLen);


  TGeoTube *connTubeHole1 = new TGeoTube("tube1HoleC", 0,
					 sIBConnTubeHole1D/2,
					 sIBConnTubeHole1ZLen/1.5);

  xpos = sIBConnTubesXDist/2;
  ypos = sIBConnTubesYPos;
  zpos = connTail->GetZ(1)/2;
  TGeoTranslation *connTubes1Trans1 = new TGeoTranslation("tubes1Trans1C",
							  -xpos, ypos, zpos);
  connTubes1Trans1->RegisterYourself();
  TGeoTranslation *connTubes1Trans2 = new TGeoTranslation("tubes1Trans2C",
							   xpos, ypos, zpos);
  connTubes1Trans2->RegisterYourself();


  TGeoCompositeShape *connTailSh = new TGeoCompositeShape(
		"connTailC-tube1HoleC:tubes1Trans1C-tube1HoleC:tubes1Trans2C");


  TGeoVolume *connBlockTail = new TGeoVolume("IBConnectorBlockTailCSide",
					     connTailSh,medPEEK);
  connBlockTail->SetFillColor(42);  // Brownish shade
  connBlockTail->SetLineColor(42);


  // The steel insert, an Tube
  TGeoTube *connInsertSh = new TGeoTube(sIBConnInsertD/2,
					sIBConnInsertHoleD/2,
					sIBConnInsertHeight/2);

  TGeoVolume *connInsert = new TGeoVolume("IBConnectorInsertCSide",
					  connInsertSh, medWC);
  connInsert->SetFillColor(kGray);
  connInsert->SetLineColor(kGray);


  // The plug, a Pcon
  TGeoPcon *connPlugSh = new TGeoPcon(0,360,4);
  connPlugSh->DefineSection(0,                 0., 0., sIBConnTubeHole2D/2);
  connPlugSh->DefineSection(1, sIBConnPlugThick, 0., sIBConnTubeHole2D/2);
  connPlugSh->DefineSection(2, sIBConnPlugThick,
			        sIBConnPlugInnerD/2, sIBConnTubeHole2D/2);
  connPlugSh->DefineSection(3, sIBConnPlugTotLen,
			        sIBConnPlugInnerD/2, sIBConnTubeHole2D/2);

  TGeoVolume *connPlug = new TGeoVolume("IBConnectorPlugC",
					connPlugSh,medPEEK);
  connPlug->SetFillColor(44);  // Brownish shade (a bit darker to spot it)
  connPlug->SetLineColor(44);


  // Now create the container: cannot be a simple box
  // to avoid fake overlaps with stave elements
  xlen = sIBConnectorXWidth;
  ylen = sIBConnectorYTot;
  zlen = sIBConnectBlockZLen - sIBConnTailZLen;

  TGeoBBox *connBox = new TGeoBBox("connBoxC", xlen/2, ylen/2, zlen/2);

  ypos = -connBox->GetDY();
  zpos = -connBox->GetDZ() - connTail->GetZ(1);
  TGeoTranslation *transTailC = new TGeoTranslation("transTailC",
						     0, ypos, zpos);
  transTailC->RegisterYourself();

  TGeoTube *connTubeHollow = new TGeoTube("tubeHollowC", 0,
					 sIBConnTubeHole1D/2,
					 sIBConnTubeHole1ZLen/2);

  xpos = sIBConnTubesXDist/2;
  ypos = -connBox->GetDY() + sIBConnTubesYPos;
  zpos = -connBox->GetDZ() - connTail->GetZ(1) + sIBConnTubeHole1ZLen/2;
  TGeoTranslation *connTubeHollTrans1 = new TGeoTranslation("tubeHollTrans1C",
							    -xpos, ypos, zpos);
  connTubeHollTrans1->RegisterYourself();
  TGeoTranslation *connTubeHollTrans2 = new TGeoTranslation("tubeHollTrans2C",
							     xpos, ypos, zpos);
  connTubeHollTrans2->RegisterYourself();

  TGeoCompositeShape *connBoxSh = new TGeoCompositeShape(
      "connBoxC+connTailC:transTailC-tubeHollowC:tubeHollTrans1C-tubeHollowC:tubeHollTrans2C");

  TGeoVolume *connBoxCSide = new TGeoVolume("IBConnectorCSide",
					    connBoxSh, medAir);


  // Finally build up the connector
  // (NB: the origin is in the connBox, i.e. w/o the tail in Z)
  ypos = -connBoxSh->GetDY();
  zpos = -connBodySh->GetDZ() - connTail->GetZ(1);
  connBoxCSide->AddNode(connBlockTail, 1, new TGeoTranslation(0, ypos, zpos));

  ypos = -connBoxSh->GetDY() + connBodySh->GetDY();
  connBoxCSide->AddNode(connBlockBody, 1, new TGeoTranslation(0, ypos, 0));

  zpos = -connBox->GetDZ() + sIBConnInsertHoleZPos;
  connBoxCSide->AddNode(connInsert, 1, new TGeoCombiTrans(0, ypos, zpos,
					   new TGeoRotation("",0,90,0)));

  xpos =  connBox->GetDX();
  ypos = -connBox->GetDY() + sIBConnTubesYPos;
  zpos =  connBox->GetDZ() - sIBConnTubeHole3ZPos;;
  connBoxCSide->AddNode(connPlug, 1, new TGeoCombiTrans(xpos, ypos, zpos,
					   new TGeoRotation("",90,-90,90)));

}

TGeoVolume *V3Layer::createStaveModelOuterBDummy(const TGeoManager *) const
{
  // Done, return the stave structure
  return nullptr;
}

TGeoVolume *V3Layer::createStaveModelOuterB0(const TGeoManager *mgr)
{
//
// Creation of the mechanical stave structure for the Outer Barrel as in v0
// (we fake the module and halfstave volumes to have always
// the same formal geometry hierarchy)
//
// Input:
//         mgr  : the GeoManager (used only to get the proper material)
//
// Output:
//
// Return:
//
// Created:      20 Dec 2013  Mario Sitta
// Updated:      12 Mar 2014  Mario Sitta
// Updated:      16 Mar 2017  Mario Sitta  AliceO2 version
//

  // Local variables
  Double_t xmod, ymod, zmod;
  Double_t xlen, ylen, zlen;
  Double_t ypos, zpos;
  char volumeName[30];

  // First create all needed shapes
  // The chip
  xlen = sOBHalfStaveWidth;
  ylen = 0.5 * mChipThickness;
  zlen = sOBModuleZLength / 2;

  TGeoVolume *chipVol = createChipInnerB(xlen, ylen, zlen);

  xmod = ((TGeoBBox *) chipVol->GetShape())->GetDX();
  ymod = ((TGeoBBox *) chipVol->GetShape())->GetDY();
  zmod = ((TGeoBBox *) chipVol->GetShape())->GetDZ();

  TGeoBBox *module = new TGeoBBox(xmod, ymod, zmod);

  zlen = sOBModuleZLength * mNumberOfModules;
  TGeoBBox *hstave = new TGeoBBox(xlen, ylen, zlen / 2);

  // We have all shapes: now create the real volumes
  TGeoMedium *medAir = mgr->GetMedium("ITS_AIR$");

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSModulePattern(), mLayerNumber);
  TGeoVolume *modVol = new TGeoVolume(volumeName, module, medAir);
  modVol->SetVisibility(kTRUE);

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSHalfStavePattern(), mLayerNumber);
  TGeoVolume *hstaveVol = new TGeoVolume(volumeName, hstave, medAir);

  // Finally build it up
  modVol->AddNode(chipVol, 0);
  mHierarchy[kChip] = 1;

  for (Int_t j = 0; j < mNumberOfModules; j++) {
    ypos = 0.021; // Remove small overlap - M.S: 21may13
    zpos = -hstave->GetDZ() + j * 2 * zmod + zmod;
    hstaveVol->AddNode(modVol, j, new TGeoTranslation(0, ypos, zpos));
    mHierarchy[kModule]++;
  }
  // Done, return the stave structure
  return hstaveVol;
}

TGeoVolume *V3Layer::createStaveModelOuterB1(const TGeoManager *mgr)
{
  Double_t yFlex1 = sOBFlexCableAlThick;
  Double_t yFlex2 = sOBFlexCableKapThick;
  Double_t flexOverlap = 5; // to be checked
  Double_t xHalmSt = sOBHalfStaveWidth / 2;
  Double_t rCoolMin = sOBCoolTubeInnerD / 2;
  Double_t rCoolMax = rCoolMin + sOBCoolTubeThick;
  Double_t kLay1 = 0.004; // to be checked
  Double_t kLay2 = sOBGraphiteFoilThick;

  Double_t xlen, ylen;
  Double_t ymod, zmod;
  Double_t xtru[12], ytru[12];
  Double_t xpos, ypos, ypos1, zpos /*, zpos5cm*/;
  Double_t zlen;
  char volumeName[30];

  zlen = (mNumberOfModules * sOBModuleZLength + (mNumberOfModules - 1) * sOBModuleGap) / 2;

  // First create all needed shapes
  TGeoVolume *moduleVol = createModuleOuterB();
  moduleVol->SetVisibility(kTRUE);
  ymod = ((TGeoBBox *) (moduleVol->GetShape()))->GetDY();
  zmod = ((TGeoBBox *) (moduleVol->GetShape()))->GetDZ();

  TGeoBBox *busAl = new TGeoBBox("BusAl", xHalmSt, sOBBusCableAlThick / 2, zlen);
  TGeoBBox *busKap = new TGeoBBox("BusKap", xHalmSt, sOBBusCableKapThick / 2, zlen);

  TGeoBBox *coldPlate =
    new TGeoBBox("ColdPlate", sOBHalfStaveWidth / 2, sOBColdPlateThick / 2, zlen);

  TGeoTube *coolTube = new TGeoTube("CoolingTube", rCoolMin, rCoolMax, zlen);
  TGeoTube *coolWater = new TGeoTube("CoolingWater", 0., rCoolMin, zlen);

  xlen = xHalmSt - sOBCoolTubeXDist / 2 - coolTube->GetRmax();
  TGeoBBox *graphlat = new TGeoBBox("GraphLateral", xlen / 2, kLay2 / 2, zlen);

  xlen = sOBCoolTubeXDist / 2 - coolTube->GetRmax();
  TGeoBBox *graphmid = new TGeoBBox("GraphMiddle", xlen, kLay2 / 2, zlen);

  ylen = coolTube->GetRmax() - kLay2;
  TGeoBBox *graphvert = new TGeoBBox("GraphVertical", kLay2 / 2, ylen / 2, zlen);

  TGeoTubeSeg *graphtub =
    new TGeoTubeSeg("GraphTube", rCoolMax, rCoolMax + kLay2, zlen, 180., 360.);

  xlen = xHalmSt - sOBCoolTubeXDist / 2 - coolTube->GetRmax() - kLay2;
  TGeoBBox *fleeclat = new TGeoBBox("FleecLateral", xlen / 2, kLay1 / 2, zlen);

  xlen = sOBCoolTubeXDist / 2 - coolTube->GetRmax() - kLay2;
  TGeoBBox *fleecmid = new TGeoBBox("FleecMiddle", xlen, kLay1 / 2, zlen);

  ylen = coolTube->GetRmax() - kLay2 - kLay1;
  TGeoBBox *fleecvert = new TGeoBBox("FleecVertical", kLay1 / 2, ylen / 2, zlen);

  TGeoTubeSeg *fleectub =
    new TGeoTubeSeg("FleecTube", rCoolMax + kLay2, rCoolMax + kLay1 + kLay2, zlen, 180., 360.);

  TGeoBBox *flex1_5cm = new TGeoBBox("Flex1MV_5cm", xHalmSt, yFlex1 / 2, flexOverlap / 2);
  TGeoBBox *flex2_5cm = new TGeoBBox("Flex2MV_5cm", xHalmSt, yFlex2 / 2, flexOverlap / 2);

  // The half stave container (an XTru to avoid overlaps between neightbours)
  xtru[0] = xHalmSt;
  ytru[0] = 0;
  xtru[1] = xtru[0];
  ytru[1] = -2 * (ymod + busAl->GetDY() + busKap->GetDY() + coldPlate->GetDY() + graphlat->GetDY() +
                  fleeclat->GetDY());
  xtru[2] = sOBCoolTubeXDist / 2 + fleectub->GetRmax();
  ytru[2] = ytru[1];
  xtru[3] = xtru[2];
  ytru[3] = ytru[2] - (coolTube->GetRmax() + fleectub->GetRmax());
  xtru[4] = sOBCoolTubeXDist / 2 - fleectub->GetRmax();
  ytru[4] = ytru[3];
  xtru[5] = xtru[4];
  ytru[5] = ytru[2];
  for (Int_t i = 0; i < 6; i++) {
    xtru[6 + i] = -xtru[5 - i];
    ytru[6 + i] = ytru[5 - i];
  }
  TGeoXtru *halmStave = new TGeoXtru(2);
  halmStave->DefinePolygon(12, xtru, ytru);
  halmStave->DefineSection(0, -mZLength / 2);
  halmStave->DefineSection(1, mZLength / 2);

  // We have all shapes: now create the real volumes

  TGeoMedium *medAluminum = mgr->GetMedium("ITS_ALUMINUM$");
  TGeoMedium *medCarbon = mgr->GetMedium("ITS_CARBON$");
  TGeoMedium *medKapton = mgr->GetMedium("ITS_KAPTON(POLYCH2)$");
  TGeoMedium *medWater = mgr->GetMedium("ITS_WATER$");
  TGeoMedium *medCarbonFleece = mgr->GetMedium("ITS_CarbonFleece$");
  TGeoMedium *medFGS003 = mgr->GetMedium("ITS_FGS003$"); // amec thermasol
  TGeoMedium *medAir = mgr->GetMedium("ITS_AIR$");

  TGeoVolume *busAlVol = new TGeoVolume("BusAlVol", busAl, medAluminum);
  busAlVol->SetLineColor(kCyan);
  busAlVol->SetFillColor(busAlVol->GetLineColor());
  busAlVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume *busKapVol = new TGeoVolume("BusKapVol", busKap, medKapton);
  busKapVol->SetLineColor(kBlue);
  busKapVol->SetFillColor(busKapVol->GetLineColor());
  busKapVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume *coldPlateVol = new TGeoVolume("ColdPlateVol", coldPlate, medCarbon);
  coldPlateVol->SetLineColor(kYellow - 3);
  coldPlateVol->SetFillColor(coldPlateVol->GetLineColor());
  coldPlateVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume *coolTubeVol = new TGeoVolume("CoolingTubeVol", coolTube, medKapton);
  coolTubeVol->SetLineColor(kGray);
  coolTubeVol->SetFillColor(coolTubeVol->GetLineColor());
  coolTubeVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume *coolWaterVol = new TGeoVolume("CoolingWaterVol", coolWater, medWater);
  coolWaterVol->SetLineColor(kBlue);
  coolWaterVol->SetFillColor(coolWaterVol->GetLineColor());
  coolWaterVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume *graphlatVol = new TGeoVolume("GraphiteFoilLateral", graphlat, medFGS003);
  graphlatVol->SetLineColor(kGreen);
  graphlatVol->SetFillColor(graphlatVol->GetLineColor());
  graphlatVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume *graphmidVol = new TGeoVolume("GraphiteFoilMiddle", graphmid, medFGS003);
  graphmidVol->SetLineColor(kGreen);
  graphmidVol->SetFillColor(graphmidVol->GetLineColor());
  graphmidVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume *graphvertVol = new TGeoVolume("GraphiteFoilVertical", graphvert, medFGS003);
  graphvertVol->SetLineColor(kGreen);
  graphvertVol->SetFillColor(graphvertVol->GetLineColor());
  graphvertVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume *graphtubVol = new TGeoVolume("GraphiteFoilPipeCover", graphtub, medFGS003);
  graphtubVol->SetLineColor(kGreen);
  graphtubVol->SetFillColor(graphtubVol->GetLineColor());
  graphtubVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume *fleeclatVol = new TGeoVolume("CarbonFleeceLateral", fleeclat, medCarbonFleece);
  fleeclatVol->SetLineColor(kViolet);
  fleeclatVol->SetFillColor(fleeclatVol->GetLineColor());
  fleeclatVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume *fleecmidVol = new TGeoVolume("CarbonFleeceMiddle", fleecmid, medCarbonFleece);
  fleecmidVol->SetLineColor(kViolet);
  fleecmidVol->SetFillColor(fleecmidVol->GetLineColor());
  fleecmidVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume *fleecvertVol = new TGeoVolume("CarbonFleeceVertical", fleecvert, medCarbonFleece);
  fleecvertVol->SetLineColor(kViolet);
  fleecvertVol->SetFillColor(fleecvertVol->GetLineColor());
  fleecvertVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume *fleectubVol = new TGeoVolume("CarbonFleecePipeCover", fleectub, medCarbonFleece);
  fleectubVol->SetLineColor(kViolet);
  fleectubVol->SetFillColor(fleectubVol->GetLineColor());
  fleectubVol->SetFillStyle(4000); // 0% transparent

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSHalfStavePattern(), mLayerNumber);
  TGeoVolume *halmStaveVol = new TGeoVolume(volumeName, halmStave, medAir);
  //   halmStaveVol->SetLineColor(12);
  //   halmStaveVol->SetFillColor(12);
  //   halmStaveVol->SetVisibility(kTRUE);

  TGeoVolume *flex1_5cmVol = new TGeoVolume("Flex1Vol5cm", flex1_5cm, medAluminum);
  TGeoVolume *flex2_5cmVol = new TGeoVolume("Flex2Vol5cm", flex2_5cm, medKapton);

  flex1_5cmVol->SetLineColor(kRed);
  flex2_5cmVol->SetLineColor(kGreen);

  // Now build up the half stave
  ypos = -busKap->GetDY();
  halmStaveVol->AddNode(busKapVol, 1, new TGeoTranslation(0, ypos, 0));

  ypos -= (busKap->GetDY() + busAl->GetDY());
  halmStaveVol->AddNode(busAlVol, 1, new TGeoTranslation(0, ypos, 0));

  ypos -= (busAl->GetDY() + ymod);
  for (Int_t j = 0; j < mNumberOfModules; j++) {
    zpos = -zlen + j * (2 * zmod + sOBModuleGap) + zmod;
    halmStaveVol->AddNode(moduleVol, j, new TGeoTranslation(0, ypos, zpos));
    mHierarchy[kModule]++;
  }

  ypos -= (ymod + coldPlate->GetDY());
  halmStaveVol->AddNode(coldPlateVol, 1, new TGeoTranslation(0, ypos, 0));

  coolTubeVol->AddNode(coolWaterVol, 1, nullptr);

  xpos = sOBCoolTubeXDist / 2;
  ypos1 = ypos - (coldPlate->GetDY() + coolTube->GetRmax());
  halmStaveVol->AddNode(coolTubeVol, 1, new TGeoTranslation(-xpos, ypos1, 0));
  halmStaveVol->AddNode(coolTubeVol, 2, new TGeoTranslation(xpos, ypos1, 0));

  halmStaveVol->AddNode(graphtubVol, 1, new TGeoTranslation(-xpos, ypos1, 0));
  halmStaveVol->AddNode(graphtubVol, 2, new TGeoTranslation(xpos, ypos1, 0));

  halmStaveVol->AddNode(fleectubVol, 1, new TGeoTranslation(-xpos, ypos1, 0));
  halmStaveVol->AddNode(fleectubVol, 2, new TGeoTranslation(xpos, ypos1, 0));

  xpos = xHalmSt - graphlat->GetDX();
  ypos1 = ypos - (coldPlate->GetDY() + graphlat->GetDY());
  halmStaveVol->AddNode(graphlatVol, 1, new TGeoTranslation(-xpos, ypos1, 0));
  halmStaveVol->AddNode(graphlatVol, 2, new TGeoTranslation(xpos, ypos1, 0));

  halmStaveVol->AddNode(graphmidVol, 1, new TGeoTranslation(0, ypos1, 0));

  xpos = xHalmSt - 2 * graphlat->GetDX() + graphvert->GetDX();
  ypos1 = ypos - (coldPlate->GetDY() + 2 * graphlat->GetDY() + graphvert->GetDY());
  halmStaveVol->AddNode(graphvertVol, 1, new TGeoTranslation(-xpos, ypos1, 0));
  halmStaveVol->AddNode(graphvertVol, 2, new TGeoTranslation(xpos, ypos1, 0));
  xpos = graphmid->GetDX() - graphvert->GetDX();
  halmStaveVol->AddNode(graphvertVol, 3, new TGeoTranslation(-xpos, ypos1, 0));
  halmStaveVol->AddNode(graphvertVol, 4, new TGeoTranslation(xpos, ypos1, 0));

  xpos = xHalmSt - fleeclat->GetDX();
  ypos1 = ypos - (coldPlate->GetDY() + 2 * graphlat->GetDY() + fleeclat->GetDY());
  halmStaveVol->AddNode(fleeclatVol, 1, new TGeoTranslation(-xpos, ypos1, 0));
  halmStaveVol->AddNode(fleeclatVol, 2, new TGeoTranslation(xpos, ypos1, 0));

  halmStaveVol->AddNode(fleecmidVol, 1, new TGeoTranslation(0, ypos1, 0));

  xpos = xHalmSt - 2 * fleeclat->GetDX() + fleecvert->GetDX();
  ypos1 = ypos -
          (coldPlate->GetDY() + 2 * graphlat->GetDY() + 2 * fleeclat->GetDY() + fleecvert->GetDY());
  halmStaveVol->AddNode(fleecvertVol, 1, new TGeoTranslation(-xpos, ypos1, 0));
  halmStaveVol->AddNode(fleecvertVol, 2, new TGeoTranslation(xpos, ypos1, 0));
  xpos = fleecmid->GetDX() - fleecvert->GetDX();
  halmStaveVol->AddNode(fleecvertVol, 3, new TGeoTranslation(-xpos, ypos1, 0));
  halmStaveVol->AddNode(fleecvertVol, 4, new TGeoTranslation(xpos, ypos1, 0));

  // THE FOLLOWING IS ONLY A REMINDER FOR WHAT IS STILL MISSING

  //   for (Int_t j=0; j<mNumberOfChips; j++) {

  //     zpos = -(zact + (mNumberOfChips-1)*modGap)/2 + j*(zMod + modGap) + zMod/2;
  //     zpos5cm = -(zact + (mNumberOfChips-1)*modGap)/2 + (j+1)*(zMod + modGap) + flexOverlap/2 ;

  //     halmStaveVol->AddNode(moduleVol, j, new TGeoTranslation(xPos, -ylen + yPos + 2*rCoolMax +
  // yCPlate + yGlue + yModPlate + ymod, zpos));
  //     halmStaveVol->AddNode(moduleVol, mNumberOfChips+j, new TGeoTranslation(-xPos, -ylen + yPos
  // +
  // 2*rCoolMax + yCPlate + yGlue + yModPlate + ymod +deltaY, zpos));

  //     if((j+1)!=mNumberOfChips){
  //       halmStaveVol->AddNode(flex1_5cmVol,j,new TGeoTranslation(xPos,-ylen + yPos + 2*rCoolMax +
  // yCPlate + yGlue + yModPlate + 2*ymod + yFlex1 + yFlex2 + yFlex1/2,zpos5cm));
  //       halmStaveVol->AddNode(flex1_5cmVol,mNumberOfChips+j,new TGeoTranslation(-xPos,-ylen +
  // yPos +
  // 2*rCoolMax + yCPlate + yGlue + yModPlate + 2*ymod + yFlex1 + yFlex2 + yFlex1/2
  // +deltaY,zpos5cm));
  //       halmStaveVol->AddNode(flex2_5cmVol,j,new TGeoTranslation(xPos,-ylen + yPos + 2*rCoolMax +
  // yCPlate + yGlue + yModPlate + 2*ymod + 2*yFlex1 + 3*yFlex2/2,zpos5cm));
  //       halmStaveVol->AddNode(flex2_5cmVol,mNumberOfChips+j,new TGeoTranslation(-xPos,-ylen +
  // yPos +
  // 2*rCoolMax + yCPlate + yGlue + yModPlate + 2*ymod + 2*yFlex1 + 3*yFlex2/2 +deltaY,zpos5cm));
  //     }
  //     else {
  //       halmStaveVol->AddNode(flex1_5cmVol,j,new TGeoTranslation(xPos,-ylen + yPos + 2*rCoolMax +
  // yCPlate + yGlue + yModPlate + 2*ymod + yFlex1/2,zpos5cm-modGap));
  //       halmStaveVol->AddNode(flex1_5cmVol,mNumberOfChips+j,new TGeoTranslation(-xPos,-ylen +
  // yPos +
  // 2*rCoolMax + yCPlate + yGlue + yModPlate + 2*ymod + yFlex1/2 +deltaY,zpos5cm-modGap));
  //       halmStaveVol->AddNode(flex2_5cmVol,j,new TGeoTranslation(xPos,-ylen + yPos + 2*rCoolMax +
  // yCPlate + yGlue + yModPlate +2*ymod + yFlex1 + yFlex2/2,zpos5cm-modGap));
  //       halmStaveVol->AddNode(flex2_5cmVol,mNumberOfChips+j,new TGeoTranslation(-xPos,-ylen +
  // yPos +
  // 2*rCoolMax + yCPlate + yGlue + yModPlate + 2*ymod + yFlex1 + yFlex2/2 +deltaY,zpos5cm-modGap));

  //       }
  //   }
  // Done, return the half stave structure
  return halmStaveVol;
}

TGeoVolume *V3Layer::createSpaceFrameOuterB(const TGeoManager *mgr)
{
  TGeoVolume *mechStavVol = nullptr;

  switch (mStaveModel) {
    case Detector::kOBModelDummy:
    case Detector::kOBModel0:
      mechStavVol = createSpaceFrameOuterBDummy(mgr);
      break;
    case Detector::kOBModel1:
      mechStavVol = createSpaceFrameOuterB1(mgr);
      break;
    default:
      LOG(FATAL) << "Unknown stave model " << mStaveModel << FairLogger::endl;
      break;
  }

  return mechStavVol;
}

TGeoVolume *V3Layer::createSpaceFrameOuterBDummy(const TGeoManager *) const
{
  // Done, return the stave structur
  return nullptr;
}

TGeoVolume *V3Layer::createSpaceFrameOuterB1(const TGeoManager *mgr)
{
  // Materials defined in Detector
  TGeoMedium *medCarbon = mgr->GetMedium("ITS_CARBON$");
  TGeoMedium *medAir = mgr->GetMedium("ITS_AIR$");

  // Local parameters
  Double_t sframeWidth = sOBSpaceFrameWidth;
  Double_t sframeHeight = sOBSpaceFrameTotHigh - sOBHalfStaveYTrans;
  Double_t staveBeamRadius = sOBSFrameBeamRadius;
  Double_t staveLa = sOBSpaceFrameLa;
  Double_t staveHa = sOBSpaceFrameHa;
  Double_t staveLb = sOBSpaceFrameLb;
  Double_t staveHb = sOBSpaceFrameHb;
  Double_t stavel = sOBSpaceFrameL;
  Double_t bottomBeamAngle = sOBSFBotBeamAngle;
  Double_t triangleHeight = sframeHeight - staveBeamRadius;
  Double_t halmTheta = TMath::ATan(0.5 * sframeWidth / triangleHeight);
  //  Double_t alpha              = TMath::Pi()*3./4. - halmTheta/2.;
  Double_t beta = (TMath::Pi() - 2. * halmTheta) / 4.;
  //  Double_t distCenterSideDown = 0.5*sframeWidth/TMath::Cos(beta);

  Double_t zlen;
  Double_t xpos, ypos, zpos;
  Double_t seglen;
  char volumeName[30];

  zlen = mNumberOfModules * sOBModuleZLength + (mNumberOfModules - 1) * sOBModuleGap;

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSHalfStavePattern(), mLayerNumber);
  if (gGeoManager->GetVolume(volumeName)) { // Should always be so
    sframeHeight -= ((TGeoBBox *) gGeoManager->GetVolume(volumeName)->GetShape())->GetDY() * 2;
    zlen = ((TGeoBBox *) gGeoManager->GetVolume(volumeName)->GetShape())->GetDZ() * 2;
  }
  seglen = zlen / mNumberOfModules;

  // First create all needed shapes and volumes
  TGeoBBox *spaceFrame = new TGeoBBox(sframeWidth / 2, sframeHeight / 2, zlen / 2);
  TGeoBBox *segment = new TGeoBBox(sframeWidth / 2, sframeHeight / 2, seglen / 2);

  TGeoVolume *spaceFrameVol = new TGeoVolume("CarbonFrameVolume", spaceFrame, medAir);
  spaceFrameVol->SetVisibility(kFALSE);

  TGeoVolume *segmentVol = new TGeoVolume("segmentVol", segment, medAir);

  // SpaceFrame

  //--- the top V of the Carbon Fiber Stave (segment)
  TGeoArb8 *cmStavTop1 = createStaveSide("CFstavTopCornerVol1shape", seglen / 2., halmTheta, -1,
                                         staveLa, staveHa, stavel);
  TGeoVolume *cmStavTopVol1 = new TGeoVolume("CFstavTopCornerVol1", cmStavTop1, medCarbon);
  cmStavTopVol1->SetLineColor(35);

  TGeoArb8 *cmStavTop2 = createStaveSide("CFstavTopCornerVol2shape", seglen / 2., halmTheta, 1,
                                         staveLa, staveHa, stavel);
  TGeoVolume *cmStavTopVol2 = new TGeoVolume("CFstavTopCornerVol2", cmStavTop2, medCarbon);
  cmStavTopVol2->SetLineColor(35);

  TGeoTranslation *trTop1 = new TGeoTranslation(0, sframeHeight / 2, 0);

  //--- the 2 side V
  TGeoArb8 *cmStavSide1 =
    createStaveSide("CFstavSideCornerVol1shape", seglen / 2., beta, -1, staveLb, staveHb, stavel);
  TGeoVolume *cmStavSideVol1 = new TGeoVolume("CFstavSideCornerVol1", cmStavSide1, medCarbon);
  cmStavSideVol1->SetLineColor(35);

  TGeoArb8 *cmStavSide2 =
    createStaveSide("CFstavSideCornerVol2shape", seglen / 2., beta, 1, staveLb, staveHb, stavel);
  TGeoVolume *cmStavSideVol2 = new TGeoVolume("CFstavSideCornerVol2", cmStavSide2, medCarbon);
  cmStavSideVol2->SetLineColor(35);

  xpos = -sframeWidth / 2;
  ypos = -sframeHeight / 2 + staveBeamRadius + staveHb * TMath::Sin(beta);
  TGeoCombiTrans *ctSideR = new TGeoCombiTrans(
    xpos, ypos, 0, new TGeoRotation("", 180 - 2 * beta * TMath::RadToDeg(), 0, 0));
  TGeoCombiTrans *ctSideL = new TGeoCombiTrans(
    -xpos, ypos, 0, new TGeoRotation("", -180 + 2 * beta * TMath::RadToDeg(), 0, 0));

  segmentVol->AddNode(cmStavTopVol1, 1, trTop1);
  segmentVol->AddNode(cmStavTopVol2, 1, trTop1);
  segmentVol->AddNode(cmStavSideVol1, 1, ctSideR);
  segmentVol->AddNode(cmStavSideVol1, 2, ctSideL);
  segmentVol->AddNode(cmStavSideVol2, 1, ctSideR);
  segmentVol->AddNode(cmStavSideVol2, 2, ctSideL);

  //--- The beams
  // Beams on the sides
  Double_t beamPhiPrime = TMath::ASin(
    1. / TMath::Sqrt((1 + TMath::Sin(2 * beta) * TMath::Sin(2 * beta) /
                          (tanD(sOBSFrameBeamSidePhi) * tanD(sOBSFrameBeamSidePhi)))));
  Double_t beamLength = TMath::Sqrt(sframeHeight * sframeHeight /
                                    (TMath::Sin(beamPhiPrime) * TMath::Sin(beamPhiPrime)) +
                                    sframeWidth * sframeWidth / 4.) -
                        staveLa / 2 - staveLb / 2;
  TGeoTubeSeg *sideBeam = new TGeoTubeSeg(0, staveBeamRadius, beamLength / 2, 0, 180);
  TGeoVolume *sideBeamVol = new TGeoVolume("CFstavSideBeamVol", sideBeam, medCarbon);
  sideBeamVol->SetLineColor(35);

  TGeoRotation *beamRot1 = new TGeoRotation("", /*90-2*beta*/ halmTheta * TMath::RadToDeg(),
                                            -beamPhiPrime * TMath::RadToDeg(), -90);
  TGeoRotation *beamRot2 =
    new TGeoRotation("", 90 - 2. * beta * TMath::RadToDeg(), beamPhiPrime * TMath::RadToDeg(), -90);
  TGeoRotation *beamRot3 =
    new TGeoRotation("", 90 + 2. * beta * TMath::RadToDeg(), beamPhiPrime * TMath::RadToDeg(), -90);
  TGeoRotation *beamRot4 = new TGeoRotation("", 90 + 2. * beta * TMath::RadToDeg(),
                                            -beamPhiPrime * TMath::RadToDeg(), -90);

  TGeoCombiTrans *beamTransf[8];
  xpos = 0.49 * triangleHeight * TMath::Tan(halmTheta); // was 0.5, fix small overlap
  ypos = staveBeamRadius / 2;
  zpos = seglen / 8;
  beamTransf[0] = new TGeoCombiTrans(xpos, ypos, -3 * zpos, beamRot1);

  beamTransf[1] = new TGeoCombiTrans(xpos, ypos, -3 * zpos, beamRot1);
  addTranslationToCombiTrans(beamTransf[1], 0, 0, seglen / 2);

  beamTransf[2] = new TGeoCombiTrans(xpos, ypos, -zpos, beamRot2);

  beamTransf[3] = new TGeoCombiTrans(xpos, ypos, -zpos, beamRot2);
  addTranslationToCombiTrans(beamTransf[3], 0, 0, seglen / 2);

  beamTransf[4] = new TGeoCombiTrans(-xpos, ypos, -3 * zpos, beamRot3);

  beamTransf[5] = new TGeoCombiTrans(-xpos, ypos, -3 * zpos, beamRot3);
  addTranslationToCombiTrans(beamTransf[5], 0, 0, seglen / 2);

  beamTransf[6] = new TGeoCombiTrans(-xpos, ypos, -zpos, beamRot4);
  beamTransf[7] = new TGeoCombiTrans(-xpos, ypos, 3 * zpos, beamRot4);

  //--- Beams of the bottom
  TGeoTubeSeg *bottomBeam1 =
    new TGeoTubeSeg(0, staveBeamRadius, sframeWidth / 2. - staveLb / 3, 0, 180);
  TGeoVolume *bottomBeam1Vol = new TGeoVolume("CFstavBottomBeam1Vol", bottomBeam1, medCarbon);
  bottomBeam1Vol->SetLineColor(35);

  TGeoTubeSeg *bottomBeam2 =
    new TGeoTubeSeg(0, staveBeamRadius, sframeWidth / 2. - staveLb / 3, 0, 90);
  TGeoVolume *bottomBeam2Vol = new TGeoVolume("CFstavBottomBeam2Vol", bottomBeam2, medCarbon);
  bottomBeam2Vol->SetLineColor(35);

  TGeoTubeSeg *bottomBeam3 = new TGeoTubeSeg(
    0, staveBeamRadius, 0.5 * sframeWidth / sinD(bottomBeamAngle) - staveLb / 3, 0, 180);
  TGeoVolume *bottomBeam3Vol = new TGeoVolume("CFstavBottomBeam3Vol", bottomBeam3, medCarbon);
  bottomBeam3Vol->SetLineColor(35);

  TGeoRotation *bottomBeamRot1 = new TGeoRotation("", 90, 90, 90);
  TGeoRotation *bottomBeamRot2 = new TGeoRotation("", -90, 90, -90);

  TGeoCombiTrans *bottomBeamTransf1 =
    new TGeoCombiTrans("", 0, -(sframeHeight / 2 - staveBeamRadius), 0, bottomBeamRot1);
  TGeoCombiTrans *bottomBeamTransf2 =
    new TGeoCombiTrans(0, -(sframeHeight / 2 - staveBeamRadius), -seglen / 2, bottomBeamRot1);
  TGeoCombiTrans *bottomBeamTransf3 =
    new TGeoCombiTrans(0, -(sframeHeight / 2 - staveBeamRadius), seglen / 2, bottomBeamRot2);
  // be careful for beams #3: when "reading" from -z to +z and
  // from the bottom of the stave, it should draw a Lambda, and not a V
  TGeoRotation *bottomBeamRot4 = new TGeoRotation("", -90, bottomBeamAngle, -90);
  TGeoRotation *bottomBeamRot5 = new TGeoRotation("", -90, -bottomBeamAngle, -90);

  TGeoCombiTrans *bottomBeamTransf4 =
    new TGeoCombiTrans(0, -(sframeHeight / 2 - staveBeamRadius), -seglen / 4, bottomBeamRot4);
  TGeoCombiTrans *bottomBeamTransf5 =
    new TGeoCombiTrans(0, -(sframeHeight / 2 - staveBeamRadius), seglen / 4, bottomBeamRot5);

  segmentVol->AddNode(sideBeamVol, 1, beamTransf[0]);
  segmentVol->AddNode(sideBeamVol, 2, beamTransf[1]);
  segmentVol->AddNode(sideBeamVol, 3, beamTransf[2]);
  segmentVol->AddNode(sideBeamVol, 4, beamTransf[3]);
  segmentVol->AddNode(sideBeamVol, 5, beamTransf[4]);
  segmentVol->AddNode(sideBeamVol, 6, beamTransf[5]);
  segmentVol->AddNode(sideBeamVol, 7, beamTransf[6]);
  segmentVol->AddNode(sideBeamVol, 8, beamTransf[7]);
  segmentVol->AddNode(bottomBeam1Vol, 1, bottomBeamTransf1);
  segmentVol->AddNode(bottomBeam2Vol, 1, bottomBeamTransf2);
  segmentVol->AddNode(bottomBeam2Vol, 2, bottomBeamTransf3);
  segmentVol->AddNode(bottomBeam3Vol, 1, bottomBeamTransf4);
  segmentVol->AddNode(bottomBeam3Vol, 2, bottomBeamTransf5);

  // Then build up the space frame
  for (Int_t i = 0; i < mNumberOfModules; i++) {
    zpos = -spaceFrame->GetDZ() + (1 + 2 * i) * segment->GetDZ();
    spaceFrameVol->AddNode(segmentVol, i, new TGeoTranslation(0, 0, zpos));
  }

  // Done, return the space frame structure
  return spaceFrameVol;
}

TGeoVolume *V3Layer::createChipInnerB(const Double_t xchip, const Double_t ychip,
                                             const Double_t zchip, const TGeoManager *mgr)
{
  char volumeName[30];
  Double_t xlen, ylen, zlen;
  Double_t xpos, ypos, zpos;

  // First create all needed shapes

  // The chip
  TGeoBBox *chip = new TGeoBBox(xchip, ychip, zchip);

  // The sensor
  xlen = chip->GetDX();
  ylen = 0.5 * mSensorThickness;
  zlen = chip->GetDZ();
  TGeoBBox *sensor = new TGeoBBox(xlen, ylen, zlen);

  // We have all shapes: now create the real volumes
  TGeoMedium *medSi = mgr->GetMedium("ITS_SI$");

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSChipPattern(), mLayerNumber);
  TGeoVolume *chipVol = new TGeoVolume(volumeName, chip, medSi);
  chipVol->SetVisibility(kTRUE);
  chipVol->SetLineColor(1);

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSSensorPattern(), mLayerNumber);
  TGeoVolume *sensVol = new TGeoVolume(volumeName, sensor, medSi);
  sensVol->SetVisibility(kTRUE);
  sensVol->SetLineColor(8);
  sensVol->SetLineWidth(1);
  sensVol->SetFillColor(sensVol->GetLineColor());
  sensVol->SetFillStyle(4000); // 0% transparent

  // Now build up the chip
  xpos = 0.;
  ypos = -chip->GetDY() + sensor->GetDY();
  zpos = 0.;

  chipVol->AddNode(sensVol, 1, new TGeoTranslation(xpos, ypos, zpos));

  // Done, return the chip
  return chipVol;
}

TGeoVolume *V3Layer::createModuleOuterB(const TGeoManager *mgr)
{
//
// Creates the OB Module: HIC + FPC + Carbon plate
//
// Input:
//         mgr  : the GeoManager (used only to get the proper material)
//
// Output:
//
// Return:
//         the module as a TGeoVolume
//
// Created:      18 Dec 2013  M. Sitta, A. Barbano
// Updated:      26 Feb 2014  M. Sitta
// Updated:      12 Nov 2014  M. Sitta  Model2 is w/o Carbon Plate and Glue
//                                      and Cu instead of Al
// Updated:      16 Mar 2017  M. Sitta  AliceO2 version
//
  char volumeName[30];

  Double_t xGap = sOBChipXGap;
  Double_t zGap = sOBChipZGap;

  Double_t xchip, ychip, zchip;
  Double_t xlen, ylen, zlen;
  Double_t xpos, ypos, zpos;

  // First create all needed shapes

  // The chip (the same as for IB)
  xlen = (sOBHalfStaveWidth / 2 - xGap / 2) / sOBNChipRows;
  ylen = 0.5 * mChipThickness;
  zlen = (sOBModuleZLength - (sOBChipsPerRow - 1) * zGap) / (2 * sOBChipsPerRow);

  TGeoVolume *chipVol = createChipInnerB(xlen, ylen, zlen);

  xchip = ((TGeoBBox *) chipVol->GetShape())->GetDX();
  ychip = ((TGeoBBox *) chipVol->GetShape())->GetDY();
  zchip = ((TGeoBBox *) chipVol->GetShape())->GetDZ();

  // The module carbon plate
  xlen = sOBHalfStaveWidth / 2;
  ylen = sOBCarbonPlateThick / 2;
  zlen = sOBModuleZLength / 2;
  TGeoBBox *modPlate = new TGeoBBox("CarbonPlate", xlen, ylen, zlen);

  // The glue
  ylen = sOBGlueThick / 2;
  TGeoBBox *glue = new TGeoBBox("Glue", xlen, ylen, zlen);

  // The flex cables
  ylen = sOBFlexCableAlThick / 2;
  TGeoBBox *flexAl = new TGeoBBox("FlexAl", xlen, ylen, zlen);

  ylen = sOBFlexCableKapThick / 2;
  TGeoBBox *flexKap = new TGeoBBox("FlexKap", xlen, ylen, zlen);

  // The module
  xlen = sOBHalfStaveWidth / 2;
  ylen = ychip + modPlate->GetDY() + glue->GetDY() + flexAl->GetDY() + flexKap->GetDY();
  zlen = sOBModuleZLength / 2;
  TGeoBBox *module = new TGeoBBox("OBModule", xlen, ylen, zlen);

  // We have all shapes: now create the real volumes

  TGeoMedium *medAir = mgr->GetMedium("ITS_AIR$");
  TGeoMedium *medCarbon = mgr->GetMedium("ITS_CARBON$");
  TGeoMedium *medGlue = mgr->GetMedium("ITS_GLUE$");
  TGeoMedium *medAluminum = mgr->GetMedium("ITS_ALUMINUM$");
  TGeoMedium *medKapton = mgr->GetMedium("ITS_KAPTON(POLYCH2)$");

  TGeoVolume *modPlateVol = new TGeoVolume("CarbonPlateVol", modPlate, medCarbon);
  modPlateVol->SetLineColor(kMagenta - 8);
  modPlateVol->SetFillColor(modPlateVol->GetLineColor());
  modPlateVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume *glueVol = new TGeoVolume("GlueVol", glue, medGlue);
  glueVol->SetLineColor(kBlack);
  glueVol->SetFillColor(glueVol->GetLineColor());
  glueVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume *flexAlVol = new TGeoVolume("FlexAlVol", flexAl, medAluminum);
  flexAlVol->SetLineColor(kRed);
  flexAlVol->SetFillColor(flexAlVol->GetLineColor());
  flexAlVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume *flexKapVol = new TGeoVolume("FlexKapVol", flexKap, medKapton);
  flexKapVol->SetLineColor(kGreen);
  flexKapVol->SetFillColor(flexKapVol->GetLineColor());
  flexKapVol->SetFillStyle(4000); // 0% transparent

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSModulePattern(), mLayerNumber);
  TGeoVolume *modVol = new TGeoVolume(volumeName, module, medAir);
  modVol->SetVisibility(kTRUE);

  // Now build up the module
  ypos = -module->GetDY() + modPlate->GetDY();
  modVol->AddNode(modPlateVol, 1, new TGeoTranslation(0, ypos, 0));

  ypos += (modPlate->GetDY() + glue->GetDY());
  modVol->AddNode(glueVol, 1, new TGeoTranslation(0, ypos, 0));

  xpos = -module->GetDX() + xchip;
  ypos += (glue->GetDY() + ychip);
  for (Int_t k = 0; k < sOBChipsPerRow; k++) { // put 7x2 chip into one module
    zpos = -module->GetDZ() + zchip + k * (2 * zchip + zGap);
    modVol->AddNode(chipVol, 2 * k, new TGeoTranslation(xpos, ypos, zpos));
    modVol->AddNode(chipVol, 2 * k + 1,
                    new TGeoCombiTrans(-xpos, ypos, zpos, new TGeoRotation("", 0, 180, 180)));
    mHierarchy[kChip] += 2;
  }

  ypos += (ychip + flexAl->GetDY());
  modVol->AddNode(flexAlVol, 1, new TGeoTranslation(0, ypos, 0));

  ypos += (flexAl->GetDY() + flexKap->GetDY());
  modVol->AddNode(flexKapVol, 1, new TGeoTranslation(0, ypos, 0));

  // Done, return the module
  return modVol;
}

Double_t V3Layer::radiusOmTurboContainer()
{
  Double_t rr, delta, z, lstav, rstav;

  if (mChipThickness > 89.) { // Very big angle: avoid overflows since surely
    return -1;                 // the radius from lower vertex is the right value
  }

  rstav = mLayerRadius + 0.5 * mChipThickness;
  delta = (0.5 * mChipThickness) / cosD(mStaveTilt);
  z = (0.5 * mChipThickness) * tanD(mStaveTilt);

  rr = rstav - delta;
  lstav = (0.5 * mStaveWidth) - z;

  if ((rr * sinD(mStaveTilt) < lstav)) {
    return (rr * cosD(mStaveTilt));
  } else {
    return -1;
  }
}

void V3Layer::setNumberOfUnits(Int_t u)
{
  if (mLayerNumber < sNumberOfInnerLayers) {
    mNumberOfChips = u;
  } else {
    mNumberOfModules = u;
    mNumberOfChips = sOBChipsPerRow;
  }
}

void V3Layer::setStaveTilt(const Double_t t)
{
  if (mIsTurbo) {
    mStaveTilt = t;
  } else {
    LOG(ERROR) << "Not a Turbo layer" << FairLogger::endl;
  }
}

void V3Layer::setStaveWidth(const Double_t w)
{
  if (mIsTurbo) {
    mStaveWidth = w;
  } else {
    LOG(ERROR) << "Not a Turbo layer" << FairLogger::endl;
  }
}

TGeoArb8 *V3Layer::createStaveSide(const char *name, Double_t dz, Double_t angle,
                                          Double_t xSign, Double_t L, Double_t H, Double_t l)
{
  // Create one half of the V shape corner of CF stave

  TGeoArb8 *cmStavSide = new TGeoArb8(dz);
  cmStavSide->SetName(name);

  // Points must be in clockwise order
  cmStavSide->SetVertex(0, 0, 0);
  cmStavSide->SetVertex(2, xSign * (L * TMath::Sin(angle) - l * TMath::Cos(angle)),
                        -L * TMath::Cos(angle) - l * TMath::Sin(angle));
  cmStavSide->SetVertex(4, 0, 0);
  cmStavSide->SetVertex(6, xSign * (L * TMath::Sin(angle) - l * TMath::Cos(angle)),
                        -L * TMath::Cos(angle) - l * TMath::Sin(angle));
  if (xSign < 0) {
    cmStavSide->SetVertex(1, 0, -H);
    cmStavSide->SetVertex(3, xSign * L * TMath::Sin(angle), -L * TMath::Cos(angle));
    cmStavSide->SetVertex(5, 0, -H);
    cmStavSide->SetVertex(7, xSign * L * TMath::Sin(angle), -L * TMath::Cos(angle));
  } else {
    cmStavSide->SetVertex(1, xSign * L * TMath::Sin(angle), -L * TMath::Cos(angle));
    cmStavSide->SetVertex(3, 0, -H);
    cmStavSide->SetVertex(5, xSign * L * TMath::Sin(angle), -L * TMath::Cos(angle));
    cmStavSide->SetVertex(7, 0, -H);
  }
  return cmStavSide;
}

TGeoCombiTrans *V3Layer::createCombiTrans(const char *name, Double_t dy, Double_t dz,
                                                 Double_t dphi, Bool_t planeSym)
{
  TGeoTranslation t1(dy * cosD(90. + dphi), dy * sinD(90. + dphi), dz);
  TGeoRotation r1("", 0., 0., dphi);
  TGeoRotation r2("", 90, 180, -90 - dphi);

  TGeoCombiTrans *combiTrans1 = new TGeoCombiTrans(name);
  combiTrans1->SetTranslation(t1);
  if (planeSym) {
    combiTrans1->SetRotation(r1);
  } else {
    combiTrans1->SetRotation(r2);
  }
  return combiTrans1;
}

void V3Layer::addTranslationToCombiTrans(TGeoCombiTrans *ct, Double_t dx, Double_t dy,
                                                Double_t dz) const
{
  // Add a dx,dy,dz translation to the initial TGeoCombiTrans
  const Double_t *vect = ct->GetTranslation();
  Double_t newVect[3] = {vect[0] + dx, vect[1] + dy, vect[2] + dz};
  ct->SetTranslation(newVect);
}
