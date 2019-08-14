// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file V1Layer.cxx
/// \brief Implementation of the V1Layer class
/// \author Mario Sitta <sitta@to.infn.it>
/// \author Chinorat Kobdaj (kobdaj@g.sut.ac.th)

#include "ITSSimulation/V1Layer.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSSimulation/Detector.h"

#include "FairLogger.h" // for LOG

#include <TGeoArb8.h>    // for TGeoArb8
#include <TGeoBBox.h>    // for TGeoBBox
#include <TGeoCone.h>    // for TGeoConeSeg, TGeoCone
#include <TGeoManager.h> // for TGeoManager, gGeoManager
#include <TGeoMatrix.h>  // for TGeoCombiTrans, TGeoRotation, etc
#include <TGeoTrd1.h>    // for TGeoTrd1
#include <TGeoTube.h>    // for TGeoTube, TGeoTubeSeg
#include <TGeoVolume.h>  // for TGeoVolume, TGeoVolumeAssembly
#include <TGeoXtru.h>    // for TGeoXtru
#include "TMathBase.h"   // for Abs
#include <TMath.h>       // for Sin, RadToDeg, DegToRad, Cos, Tan, etc

#include <cstdio> // for snprintf

class TGeoMedium;

using namespace TMath;
using namespace o2::its;

// General Parameters
const Int_t V1Layer::sNumberOmInnerLayers = 3;

const Double_t V1Layer::sDefaultSensorThick = 300 * sMicron;
const Double_t V1Layer::sDefaultStaveThick = 1 * sCm;

// Inner Barrel Parameters
const Int_t V1Layer::sIBChipsPerRow = 9;
const Int_t V1Layer::sIBNChipRows = 1;

// Outer Barrel Parameters
const Int_t V1Layer::sOBChipsPerRow = 7;
const Int_t V1Layer::sOBNChipRows = 2;

const Double_t V1Layer::sOBHalfStaveWidth = 3.01 * sCm;
const Double_t V1Layer::sOBModuleWidth = sOBHalfStaveWidth;
const Double_t V1Layer::sOBModuleGap = 0.01 * sCm;
const Double_t V1Layer::sOBChipXGap = 0.01 * sCm;
const Double_t V1Layer::sOBChipZGap = 0.01 * sCm;
const Double_t V1Layer::sOBFlexCableAlThick = 0.005 * sCm;
const Double_t V1Layer::sOBFlexCableKapThick = 0.01 * sCm;
const Double_t V1Layer::sOBBusCableAlThick = 0.02 * sCm;
const Double_t V1Layer::sOBBusCableKapThick = 0.02 * sCm;
const Double_t V1Layer::sOBColdPlateThick = 0.012 * sCm;
const Double_t V1Layer::sOBCarbonPlateThick = 0.012 * sCm;
const Double_t V1Layer::sOBGlueThick = 0.03 * sCm;
const Double_t V1Layer::sOBModuleZLength = 21.06 * sCm;
const Double_t V1Layer::sOBHalfStaveYTrans = 1.76 * sMm;
const Double_t V1Layer::sOBHalfStaveXOverlap = 4.3 * sMm;
const Double_t V1Layer::sOBGraphiteFoilThick = 30.0 * sMicron;
const Double_t V1Layer::sOBCoolTubeInnerD = 2.052 * sMm;
const Double_t V1Layer::sOBCoolTubeThick = 32.0 * sMicron;
const Double_t V1Layer::sOBCoolTubeXDist = 11.1 * sMm;

const Double_t V1Layer::sOBSpaceFrameWidth = 42.0 * sMm;
const Double_t V1Layer::sOBSpaceFrameTotHigh = 43.1 * sMm;
const Double_t V1Layer::sOBSFrameBeamRadius = 0.6 * sMm;
const Double_t V1Layer::sOBSpaceFrameLa = 3.0 * sMm;
const Double_t V1Layer::sOBSpaceFrameHa = 0.721979 * sMm;
const Double_t V1Layer::sOBSpaceFrameLb = 3.7 * sMm;
const Double_t V1Layer::sOBSpaceFrameHb = 0.890428 * sMm;
const Double_t V1Layer::sOBSpaceFrameL = 0.25 * sMm;
const Double_t V1Layer::sOBSFBotBeamAngle = 56.5;
const Double_t V1Layer::sOBSFrameBeamSidePhi = 65.0;

ClassImp(V1Layer);

#define SQ(A) (A) * (A)

V1Layer::V1Layer()
  : V11Geometry(),
    mLayerNumber(0),
    mPhi0(0),
    mLayerRadius(0),
    mZLength(0),
    mSensorThickness(0),
    mStaveThickness(0),
    mStaveWidth(0),
    mStaveTilt(0),
    mNumberOfStaves(0),
    mNumberOfModules(0),
    mNumberOfChips(0),
    mChipTypeID(0),
    mIsTurbo(false),
    mBuildLevel(0),
    mStaveModel(Detector::kIBModelDummy)
{
  for (int i = kNHLevels; i--;) {
    mHierarchy[i] = 0;
  }
}

V1Layer::V1Layer(Int_t debug)
  : V11Geometry(debug),
    mLayerNumber(0),
    mPhi0(0),
    mLayerRadius(0),
    mZLength(0),
    mSensorThickness(0),
    mStaveThickness(0),
    mStaveWidth(0),
    mStaveTilt(0),
    mNumberOfStaves(0),
    mNumberOfModules(0),
    mNumberOfChips(0),
    mChipTypeID(0),
    mIsTurbo(false),
    mBuildLevel(0),
    mStaveModel(Detector::kIBModelDummy)
{
  for (int i = kNHLevels; i--;) {
    mHierarchy[i] = 0;
  }
}

V1Layer::V1Layer(Int_t lay, Int_t debug)
  : V11Geometry(debug),
    mLayerNumber(lay),
    mPhi0(0),
    mLayerRadius(0),
    mZLength(0),
    mSensorThickness(0),
    mStaveThickness(0),
    mStaveWidth(0),
    mStaveTilt(0),
    mNumberOfStaves(0),
    mNumberOfModules(0),
    mNumberOfChips(0),
    mChipTypeID(0),
    mIsTurbo(false),
    mBuildLevel(0),
    mStaveModel(Detector::kIBModelDummy)
{
  for (int i = kNHLevels; i--;) {
    mHierarchy[i] = 0;
  }
}

V1Layer::V1Layer(Int_t lay, Bool_t turbo, Int_t debug)
  : V11Geometry(debug),
    mLayerNumber(lay),
    mPhi0(0),
    mLayerRadius(0),
    mZLength(0),
    mSensorThickness(0),
    mStaveThickness(0),
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

V1Layer::V1Layer(const V1Layer& s)
  : V11Geometry(s.getDebug()),
    mLayerNumber(s.mLayerNumber),
    mPhi0(s.mPhi0),
    mLayerRadius(s.mLayerRadius),
    mZLength(s.mZLength),
    mSensorThickness(s.mSensorThickness),
    mStaveThickness(s.mStaveThickness),
    mStaveWidth(s.mStaveWidth),
    mStaveTilt(s.mStaveTilt),
    mNumberOfStaves(s.mNumberOfStaves),
    mNumberOfModules(s.mNumberOfModules),
    mNumberOfChips(s.mNumberOfChips),
    mChipTypeID(s.mChipTypeID),
    mIsTurbo(s.mIsTurbo),
    mBuildLevel(s.mBuildLevel),
    mStaveModel(s.mStaveModel)
{
  for (int i = kNHLevels; i--;) {
    mHierarchy[i] = s.mHierarchy[i];
  }
}

V1Layer& V1Layer::operator=(const V1Layer& s)
{
  if (&s == this) {
    return *this;
  }

  mLayerNumber = s.mLayerNumber;
  mPhi0 = s.mPhi0;
  mLayerRadius = s.mLayerRadius;
  mZLength = s.mZLength;
  mSensorThickness = s.mSensorThickness;
  mStaveThickness = s.mStaveThickness;
  mStaveWidth = s.mStaveWidth;
  mStaveTilt = s.mStaveTilt;
  mNumberOfStaves = s.mNumberOfStaves;
  mNumberOfModules = s.mNumberOfModules;
  mNumberOfChips = s.mNumberOfChips;
  mIsTurbo = s.mIsTurbo;
  mChipTypeID = s.mChipTypeID;
  mBuildLevel = s.mBuildLevel;
  mStaveModel = s.mStaveModel;
  for (int i = kNHLevels; i--;) {
    mHierarchy[i] = s.mHierarchy[i];
  }

  return *this;
}

V1Layer::~V1Layer() = default;

void V1Layer::createLayer(TGeoVolume* motherVolume)
{
  char volumeName[30];
  Double_t xpos, ypos, zpos;
  Double_t alpha;

  // Check if the user set the proper parameters
  if (mLayerRadius <= 0) {
    LOG(FATAL) << "Wrong layer radius " << mLayerRadius;
  }

  if (mZLength <= 0) {
    LOG(FATAL) << "Wrong layer length " << mZLength;
  }

  if (mNumberOfStaves <= 0) {
    LOG(FATAL) << "Wrong number of staves " << mNumberOfStaves;
  }

  if (mNumberOfChips <= 0) {
    LOG(FATAL) << "Wrong number of chips " << mNumberOfChips;
  }

  if (mLayerNumber >= sNumberOmInnerLayers && mNumberOfModules <= 0) {
    LOG(FATAL) << "Wrong number of modules " << mNumberOfModules;
  }

  if (mStaveThickness <= 0) {
    LOG(INFO) << "Stave thickness wrong or not set " << mStaveThickness << " using default "
              << sDefaultStaveThick;
    mStaveThickness = sDefaultStaveThick;
  }

  if (mSensorThickness <= 0) {
    LOG(INFO) << "Sensor thickness wrong or not set " << mSensorThickness << " using default "
              << sDefaultSensorThick;
    mSensorThickness = sDefaultSensorThick;
  }

  if (mSensorThickness > mStaveThickness) {
    LOG(WARNING) << "Sensor thickness " << mSensorThickness << " is greater than stave thickness "
                 << mStaveThickness << " fixing";
    mSensorThickness = mStaveThickness;
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
  TGeoVolume* layerVolume = new TGeoVolumeAssembly(volumeName);
  layerVolume->SetUniqueID(mChipTypeID);

  // layerVolume->SetVisibility(kFALSE);
  layerVolume->SetVisibility(kTRUE);
  layerVolume->SetLineColor(1);

  TGeoVolume* stavVol = createStave();

  // Now build up the layer
  alpha = 360. / mNumberOfStaves;
  Double_t r = mLayerRadius + ((TGeoBBox*)stavVol->GetShape())->GetDY();
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

void V1Layer::createLayerTurbo(TGeoVolume* motherVolume)
{
  char volumeName[30];
  Double_t xpos, ypos, zpos;
  Double_t alpha;

  // Check if the user set the proper (remaining) parameters
  if (mStaveWidth <= 0) {
    LOG(FATAL) << "Wrong stave width " << mStaveWidth;
  }

  if (Abs(mStaveTilt) > 45) {
    LOG(WARNING) << "Stave tilt angle (" << mStaveTilt << ") greater than 45deg";
  }

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSLayerPattern(), mLayerNumber);
  TGeoVolume* layerVolume = new TGeoVolumeAssembly(volumeName);
  layerVolume->SetUniqueID(mChipTypeID);
  layerVolume->SetVisibility(kTRUE);
  layerVolume->SetLineColor(1);
  TGeoVolume* stavVol = createStave();

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

TGeoVolume* V1Layer::createStave(const TGeoManager* /*mgr*/)
{
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
  ylen = 0.5 * mStaveThickness;
  zlen = 0.5 * mZLength;

  Double_t yplus = 0.46;
  auto* stave = new TGeoXtru(2); // z sections
  Double_t xv[5] = {xlen, xlen, 0, -xlen, -xlen};
  Double_t yv[5] = {ylen + 0.09, -0.15, -yplus - mSensorThickness, -0.15, ylen + 0.09};
  stave->DefinePolygon(5, xv, yv);
  stave->DefineSection(0, -zlen, 0, 0, 1.);
  stave->DefineSection(1, +zlen, 0, 0, 1.);

  // We have all shapes: now create the real volumes

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSStavePattern(), mLayerNumber);
  //  TGeoVolume *staveVol = new TGeoVolume(volumeName, stave, medAir);
  TGeoVolume* staveVol = new TGeoVolumeAssembly(volumeName);

  //  staveVol->SetVisibility(kFALSE);
  staveVol->SetVisibility(kTRUE);
  staveVol->SetLineColor(2);
  TGeoVolume* mechStaveVol = nullptr;

  // Now build up the stave
  if (mLayerNumber < sNumberOmInnerLayers) {
    TGeoVolume* modVol = createStaveInnerB(xlen, ylen, zlen);
    staveVol->AddNode(modVol, 0);
    mHierarchy[kHalfStave] = 1;

    // Mechanical stave structure
    mechStaveVol = createStaveStructInnerB(xlen, zlen);
    if (mechStaveVol) {
      ypos = ((TGeoBBox*)(modVol->GetShape()))->GetDY() +
             ((TGeoBBox*)(mechStaveVol->GetShape()))->GetDY();
      staveVol->AddNode(mechStaveVol, 1,
                        new TGeoCombiTrans(0, -ypos, 0, new TGeoRotation("", 0, 0, 180)));
    }
  } else {
    TGeoVolume* hstaveVol = createStaveOuterB();
    if (mStaveModel == Detector::kOBModel0) { // Create simplified stave struct as in v0
      staveVol->AddNode(hstaveVol, 0);
      mHierarchy[kHalfStave] = 1;
    } else { // (if mStaveModel) Create new stave struct as in TDR
      xpos = ((TGeoBBox*)(hstaveVol->GetShape()))->GetDX() - sOBHalfStaveXOverlap / 2;
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

TGeoVolume* V1Layer::createStaveInnerB(const Double_t xsta, const Double_t ysta,
                                       const Double_t zsta, const TGeoManager* mgr)
{
  Double_t xmod, ymod, zmod;
  char volumeName[30];

  // First we create the module (i.e. the HIC with 9 chips)
  TGeoVolume* moduleVol = createModuleInnerB(xsta, ysta, zsta);

  // Then we create the fake halfstave and the actual stave
  xmod = ((TGeoBBox*)(moduleVol->GetShape()))->GetDX();
  ymod = ((TGeoBBox*)(moduleVol->GetShape()))->GetDY();
  zmod = ((TGeoBBox*)(moduleVol->GetShape()))->GetDZ();

  auto* hstave = new TGeoBBox(xmod, ymod, zmod);

  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSHalfStavePattern(), mLayerNumber);
  auto* hstaveVol = new TGeoVolume(volumeName, hstave, medAir);

  // Finally build it up
  hstaveVol->AddNode(moduleVol, 0);
  mHierarchy[kModule] = 1;

  // Done, return the stave structure
  return hstaveVol;
}

TGeoVolume* V1Layer::createModuleInnerB(Double_t xmod, Double_t ymod, Double_t zmod,
                                        const TGeoManager* mgr)
{
  Double_t zchip;
  Double_t zpos;
  char volumeName[30];

  // First create the single chip
  zchip = zmod / sIBChipsPerRow;
  TGeoVolume* chipVol = createChipInnerB(xmod, ymod, zchip);

  // Then create the module and populate it with the chips
  auto* module = new TGeoBBox(xmod, ymod, zmod);

  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSModulePattern(), mLayerNumber);
  auto* modVol = new TGeoVolume(volumeName, module, medAir);

  // mm (not used)  zlen = ((TGeoBBox*)chipVol->GetShape())->GetDZ();
  for (Int_t j = 0; j < sIBChipsPerRow; j++) {
    zpos = -zmod + j * 2 * zchip + zchip;
    modVol->AddNode(chipVol, j, new TGeoTranslation(0, 0, zpos));
    mHierarchy[kChip]++;
  }
  // Done, return the module
  return modVol;
}

TGeoVolume* V1Layer::createStaveStructInnerB(const Double_t xsta, const Double_t zsta,
                                             const TGeoManager* mgr)
{
  TGeoVolume* mechStavVol = nullptr;

  switch (mStaveModel) {
    case Detector::kIBModelDummy:
      mechStavVol = createStaveModelInnerBDummy(xsta, zsta, mgr);
      break;
    case Detector::kIBModel0:
      mechStavVol = createStaveModelInnerB0(xsta, zsta, mgr);
      break;
    case Detector::kIBModel1:
      mechStavVol = createStaveModelInnerB1(xsta, zsta, mgr);
      break;
    case Detector::kIBModel21:
      mechStavVol = createStaveModelInnerB21(xsta, zsta, mgr);
      break;
    case Detector::kIBModel22:
      mechStavVol = createStaveModelInnerB22(xsta, zsta, mgr);
      break;
    case Detector::kIBModel3:
      mechStavVol = createStaveModelInnerB3(xsta, zsta, mgr);
      break;
    default:
      LOG(FATAL) << "Unknown stave model " << mStaveModel;
      break;
  }
  return mechStavVol;
}

TGeoVolume* V1Layer::createStaveModelInnerBDummy(const Double_t, const Double_t,
                                                 const TGeoManager*) const
{
  // Done, return the stave structur
  return nullptr;
}

TGeoVolume* V1Layer::createStaveModelInnerB0(const Double_t xsta, const Double_t zsta,
                                             const TGeoManager* mgr)
{
  // Materials defined in Detector
  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");
  TGeoMedium* medWater = mgr->GetMedium("ITS_WATER$");

  TGeoMedium* medM60J3K = mgr->GetMedium("ITS_M60J3K$");
  TGeoMedium* medKapton = mgr->GetMedium("ITS_KAPTON(POLYCH2)$");
  TGeoMedium* medGlue = mgr->GetMedium("ITS_GLUE$");
  TGeoMedium* medFlexCable = mgr->GetMedium("ITS_FLEXCABLE$");

  // Local parameters
  Double_t kConeOutRadius = 0.15 / 2;
  Double_t kConeInRadius = 0.1430 / 2;
  Double_t kStaveLength = zsta * 2;
  Double_t kStaveWidth = xsta * 2 - kConeOutRadius * 2;
  Double_t kWidth = kStaveWidth / 4; // 1/2 of kWidth
  Double_t kStaveHeight = 0.3;
  Double_t kHeight = kStaveHeight / 2;
  Double_t kAlpha = 90 - 67; // 90-33.69;
  Double_t kTheta = kAlpha * TMath::DegToRad();
  Double_t kS1 = kWidth / TMath::Sin(kTheta);
  Double_t kL1 = kWidth / TMath::Tan(kTheta);
  Double_t kS2 = TMath::Sqrt(kHeight * kHeight + kS1 * kS1); // TMath::Sin(the2);
  Double_t kThe2 = TMath::ATan(kHeight / kS1);
  Double_t kBeta = kThe2 * TMath::RadToDeg();
  // Int_t  loop = kStaveLength/(kL1);
  // Double_t s3 = kWidth/(2*TMath::Sin(kTheta));
  // Double_t s4 = 3*kWidth/(2*TMath::Sin(kTheta));

  LOG(DEBUG1) << "BuildLevel " << mBuildLevel;

  char volumeName[30];
  snprintf(volumeName, 30, "%s%d_StaveStruct", GeometryTGeo::getITSStavePattern(),
           mLayerNumber);

  Double_t z = 0, y = -0.011 + 0.0150, x = 0;

  TGeoVolume* mechStavVol = nullptr;

  if (mBuildLevel < 5) {

    // world (trapezoid)
    auto* mechStruct = new TGeoXtru(2); // z sections
    Double_t xv[5] = {
      kStaveWidth / 2 + 0.1, kStaveWidth / 2 + 0.1, 0, -kStaveWidth / 2 - 0.1,
      -kStaveWidth / 2 - 0.1};
    Double_t yv[5] = {-kConeOutRadius * 2 - 0.07, 0, kStaveHeight, 0, -kConeOutRadius * 2 - 0.07};
    mechStruct->DefinePolygon(5, xv, yv);
    mechStruct->DefineSection(0, -kStaveLength - 0.1, 0, 0, 1.);
    mechStruct->DefineSection(1, kStaveLength + 0.1, 0, 0, 1.);

    mechStavVol = new TGeoVolume(volumeName, mechStruct, medAir);
    mechStavVol->SetLineColor(12);
    mechStavVol->SetFillColor(12);
    mechStavVol->SetVisibility(kTRUE);

    // detailed structure ++++++++++++++
    // Pipe Kapton grey-35
    auto* coolTube = new TGeoTube(kConeInRadius, kConeOutRadius, kStaveLength / 2);
    auto* volCoolTube = new TGeoVolume("pipe", coolTube, medKapton);
    volCoolTube->SetFillColor(35);
    volCoolTube->SetLineColor(35);
    mechStavVol->AddNode(volCoolTube, 0, new TGeoTranslation(x + (kStaveWidth / 2), y - (kHeight - kConeOutRadius), 0));
    mechStavVol->AddNode(volCoolTube, 1, new TGeoTranslation(x - (kStaveWidth / 2), y - (kHeight - kConeOutRadius), 0));
  }

  if (mBuildLevel < 4) {
    auto* coolTubeW = new TGeoTube(0., kConeInRadius, kStaveLength / 2);
    auto* volCoolTubeW = new TGeoVolume("pipeWater", coolTubeW, medWater);
    volCoolTubeW->SetFillColor(4);
    volCoolTubeW->SetLineColor(4);
    mechStavVol->AddNode(volCoolTubeW, 0, new TGeoTranslation(x + (kStaveWidth / 2), y - (kHeight - kConeOutRadius), 0));
    mechStavVol->AddNode(volCoolTubeW, 1, new TGeoTranslation(x - (kStaveWidth / 2), y - (kHeight - kConeOutRadius), 0));
  }

  // frequency of filament
  // n = 4 means very dense(4 filaments per interval)
  // n = 2 means dense(2 filaments per interval)
  Int_t n = 4;
  Int_t loop = (Int_t)(kStaveLength / (4 * kL1 / n) + 2 / n) - 1;
  if (mBuildLevel < 3) {
    // Top CFRP Filament black-12 Carbon structure TGeoBBox (length,thickness,width)
    auto* t2 = new TGeoBBox(kS2, 0.007 / 2, 0.15 / 2); //(kS2,0.002,0.02);
    auto* volT2 = new TGeoVolume("TopFilament", t2, medM60J3K);
    volT2->SetLineColor(12);
    volT2->SetFillColor(12);

    for (int i = 1; i < loop; i++) { // i<60;i++){
      mechStavVol->AddNode(
        volT2, 4 * i + 0,
        new TGeoCombiTrans(
          x + kWidth, y + (2 * kConeOutRadius),
          z - kStaveLength / 2 + (i * (4 / n) * kL1) + kS1 / 2, // z-14.25+(i*2*kL1),
          new TGeoRotation("volT2", 90, 90 - kAlpha, 90 - kBeta)));
      mechStavVol->AddNode(
        volT2, 4 * i + 1,
        new TGeoCombiTrans(
          x - kWidth, y + (2 * kConeOutRadius),
          z - kStaveLength / 2 + (i * (4 / n) * kL1) + kS1 / 2, // z-14.25+(i*2*kL1),
          new TGeoRotation("volT2", 90, -90 + kAlpha, -90 + kBeta)));
      mechStavVol->AddNode(
        volT2, 4 * i + 2,
        new TGeoCombiTrans(
          x + kWidth, y + (2 * kConeOutRadius),
          z - kStaveLength / 2 + (i * (4 / n) * kL1) + kS1 / 2, // z-14.25+(i*2*kL1),
          new TGeoRotation("volT2", 90, -90 + kAlpha, 90 - kBeta)));
      mechStavVol->AddNode(
        volT2, 4 * i + 3,
        new TGeoCombiTrans(
          x - kWidth, y + (2 * kConeOutRadius),
          z - kStaveLength / 2 + (i * (4 / n) * kL1) + kS1 / 2, // z-14.25+(i*2*kL1),
          new TGeoRotation("volT2", 90, 90 - kAlpha, -90 + kBeta)));
    }

    // Bottom CFRP Filament black-12 Carbon structure  TGeoBBox (thickness,width,length)
    auto* t1 = new TGeoBBox(0.007 / 2, 0.15 / 2, kS1); //(0.002,0.02,kS1);
    auto* volT1 = new TGeoVolume("CFRPBottom", t1, medM60J3K);
    volT1->SetLineColor(12);
    volT1->SetFillColor(12);

    for (int i = 1; i < loop; i++) {
      mechStavVol->AddNode(
        volT1, 4 * i + 0,
        new TGeoCombiTrans(x + kWidth, y - kHeight, z - kStaveLength / 2 + ((4 / n) * kL1 * i) + kS1 / 2, // z-14.25+(i*2*kL1),
                           new TGeoRotation("volT1", -90, kAlpha, 0)));
      mechStavVol->AddNode(
        volT1, 4 * i + 1,
        new TGeoCombiTrans(x - kWidth, y - kHeight, z - kStaveLength / 2 + ((4 / n) * kL1 * i) + kS1 / 2, // z-14.25+(i*2*kL1),
                           new TGeoRotation("volT1", 90, kAlpha, 0)));
      mechStavVol->AddNode(
        volT1, 4 * i + 2,
        new TGeoCombiTrans(x + kWidth, y - kHeight, z - kStaveLength / 2 + (i * (4 / n) * kL1) + kS1 / 2, // z-14.25+(i*2*kL1),
                           new TGeoRotation("volT1", -90, -kAlpha, 0)));
      mechStavVol->AddNode(
        volT1, 4 * i + 3,
        new TGeoCombiTrans(x - kWidth, y - kHeight, z - kStaveLength / 2 + (i * (4 / n) * kL1) + kS1 / 2, // z-14.25+(i*2*kL1),
                           new TGeoRotation("volT1", -90, +kAlpha, 0)));
    }
  }

  if (mBuildLevel < 2) {
    // Glue CFRP-Silicon layers TGeoBBox(thickness,width,kS1);
    auto* tG = new TGeoBBox(0.0075 / 2, 0.18 / 2, kS1);
    auto* volTG = new TGeoVolume("Glue1", tG, medGlue);
    volTG->SetLineColor(5);
    volTG->SetFillColor(5);

    for (int i = 1; i < loop; i++) { // i<60;i++){
      mechStavVol->AddNode(
        volTG, 4 * i + 0,
        new TGeoCombiTrans(x + kWidth, y - 0.16, z - kStaveLength / 2 + ((4 / n) * kL1 * i) + kS1 / 2, // z-14.25+(2*kL1*i),
                           new TGeoRotation("volTG", -90, kAlpha, 0)));
      mechStavVol->AddNode(
        volTG, 4 * i + 1,
        new TGeoCombiTrans(x - kWidth, y - 0.16, z - kStaveLength / 2 + ((4 / n) * kL1 * i) + kS1 / 2, // z-14.25+(2*kL1*i),
                           new TGeoRotation("volTG", 90, kAlpha, 0)));
      mechStavVol->AddNode(
        volTG, 4 * i + 2,
        new TGeoCombiTrans(x + kWidth, y - 0.16, z - kStaveLength / 2 + ((4 / n) * i * kL1) + kS1 / 2, // z-14.25+(i*2*kL1),
                           new TGeoRotation("volTG", -90, -kAlpha, 0)));
      mechStavVol->AddNode(
        volTG, 4 * i + 3,
        new TGeoCombiTrans(x - kWidth, y - 0.16, z - kStaveLength / 2 + (i * (4 / n) * kL1) + kS1 / 2, // z-14.25+(i*2*kL1),
                           new TGeoRotation("volTG", -90, +kAlpha, 0)));
    }

    auto* glue = new TGeoBBox(xsta, 0.005 / 2, zsta);
    auto* volGlue = new TGeoVolume("Glue2", glue, medGlue);
    volGlue->SetLineColor(5);
    volGlue->SetFillColor(5);
    // mechStavVol->AddNode(volGlue, 0, new TGeoCombiTrans(x, y-0.16, z, new TGeoRotation("",0, 0,
    // 0)));
    mechStavVol->AddNode(volGlue, 1, new TGeoCombiTrans(x, y - 0.165 - mSensorThickness - 0.005, z, new TGeoRotation("", 0, 0, 0)));
  }

  if (mBuildLevel < 1) {
    // Flex cable brown-28 TGeoBBox(width,thickness,length);
    auto* kapCable = new TGeoBBox(xsta, 0.01 / 2, zsta);
    auto* volCable = new TGeoVolume("FlexCable", kapCable, medFlexCable);
    volCable->SetLineColor(28);
    volCable->SetFillColor(28);
    mechStavVol->AddNode(volCable, 0,
                         new TGeoCombiTrans(x, y - 0.165 - mSensorThickness - 0.005 - 0.01, z,
                                            new TGeoRotation("", 0, 0, 0)));
  }
  // Done, return the stave structur
  return mechStavVol;
}

TGeoVolume* V1Layer::createStaveModelInnerB1(const Double_t xsta, const Double_t zsta,
                                             const TGeoManager* mgr)
{
  // Materials defined in Detector
  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");
  TGeoMedium* medWater = mgr->GetMedium("ITS_WATER$");

  TGeoMedium* medM60J3K = mgr->GetMedium("ITS_M60J3K$");
  TGeoMedium* medKapton = mgr->GetMedium("ITS_KAPTON(POLYCH2)$");
  TGeoMedium* medGlue = mgr->GetMedium("ITS_GLUE$");
  TGeoMedium* medFlexCable = mgr->GetMedium("ITS_FLEXCABLE$");

  // Local parameters
  Double_t kConeOutRadius = 0.15 / 2;
  //    Double_t kConeInRadius = 0.1430/2;
  Double_t kStaveLength = zsta * 2;
  //    Double_t kStaveWidth = xsta*2-kConeOutRadius*2;
  Double_t kStaveWidth = xsta * 2;
  Double_t kWidth = kStaveWidth / 4; // 1/2 of kWidth
  Double_t kStaveHeight = 0.3;
  Double_t kHeight = kStaveHeight / 2;
  Double_t kAlpha = 90 - 33.; // 90-30;
  Double_t kTheta = kAlpha * TMath::DegToRad();
  Double_t kS1 = kWidth / TMath::Sin(kTheta);
  Double_t kL1 = kWidth / TMath::Tan(kTheta);
  Double_t kS2 = TMath::Sqrt(kHeight * kHeight + kS1 * kS1); // TMath::Sin(the2);
  Double_t kThe2 = TMath::ATan(kHeight / kS1);
  Double_t kBeta = kThe2 * TMath::RadToDeg();
  Int_t loop = (Int_t)((kStaveLength / (2 * kL1)) / 2);

  TGeoVolume* mechStavVol = nullptr;

  char volumeName[30];
  snprintf(volumeName, 30, "%s%d_StaveStruct", GeometryTGeo::getITSStavePattern(),
           mLayerNumber);

  // detailed structure ++++++++++++++
  Double_t z = 0, y = -0.011 + 0.0150, x = 0;

  // Polimide micro channels numbers
  Double_t yMC = y - kHeight + 0.01;
  Int_t nb = (Int_t)(kStaveWidth / 0.1) + 1;
  Double_t xstaMC = (nb * 0.1 - 0.08) / 2;

  if (mBuildLevel < 5) {
    // world (trapezoid)
    auto* mechStruct = new TGeoXtru(2); // z sections
    Double_t xv[5] = {
      kStaveWidth / 2 + 0.1, kStaveWidth / 2 + 0.1, 0, -kStaveWidth / 2 - 0.1,
      -kStaveWidth / 2 - 0.1};
    Double_t yv[5] = {-kConeOutRadius * 2 - 0.07, 0, kStaveHeight, 0, -kConeOutRadius * 2 - 0.07};
    mechStruct->DefinePolygon(5, xv, yv);
    mechStruct->DefineSection(0, -kStaveLength - 0.1, 0, 0, 1.);
    mechStruct->DefineSection(1, kStaveLength + 0.1, 0, 0, 1.);

    mechStavVol = new TGeoVolume(volumeName, mechStruct, medAir);
    mechStavVol->SetLineColor(12);
    mechStavVol->SetFillColor(12);
    mechStavVol->SetVisibility(kTRUE);

    // Polimide micro channels numbers
    auto* tM0 = new TGeoBBox(xstaMC, 0.005 / 2, zsta);
    auto* volTM0 = new TGeoVolume("MicroChanCover", tM0, medKapton);
    volTM0->SetLineColor(35);
    volTM0->SetFillColor(35);
    mechStavVol->AddNode(volTM0, 0,
                         new TGeoCombiTrans(x, -0.0125 + yMC, z, new TGeoRotation("", 0, 0, 0)));
    mechStavVol->AddNode(volTM0, 1,
                         new TGeoCombiTrans(x, +0.0125 + yMC, z, new TGeoRotation("", 0, 0, 0)));

    auto* tM0b = new TGeoBBox(0.02 / 2, 0.02 / 2, zsta);
    auto* volTM0b = new TGeoVolume("MicroChanWalls", tM0b, medKapton);
    volTM0b->SetLineColor(35);
    volTM0b->SetFillColor(35);
    for (Int_t ib = 0; ib < nb; ib++) {
      mechStavVol->AddNode(volTM0b, ib, new TGeoCombiTrans(x + ib * 0.1 - xstaMC + 0.01, yMC, z, new TGeoRotation("", 0, 0, 0)));
    }
  }

  if (mBuildLevel < 4) {
    // Water in Polimide micro channels
    auto* water = new TGeoBBox(0.08 / 2, 0.02 / 2, zsta + 0.1);
    auto* volWater = new TGeoVolume("Water", water, medWater);
    volWater->SetLineColor(4);
    volWater->SetFillColor(4);
    for (Int_t ib = 0; ib < (nb - 1); ib++) {
      mechStavVol->AddNode(volWater, ib, new TGeoCombiTrans(x + ib * 0.1 - xstaMC + 0.06, yMC, z, new TGeoRotation("", 0, 0, 0)));
    }
  }

  if (mBuildLevel < 3) {
    // Bottom filament CFRP black-12 Carbon structure TGeoBBox (thickness,width,length)
    Double_t filWidth = 0.04;
    Double_t filHeight = 0.02;
    auto* t1 = new TGeoBBox(filHeight / 2, filWidth / 2, kS1);
    auto* volT1 = new TGeoVolume("CFRPBottom", t1, medM60J3K);
    volT1->SetLineColor(12);
    volT1->SetFillColor(12);
    for (int i = 0; i < loop; i++) { // i<30;i++){
      mechStavVol->AddNode(volT1, 4 * i + 0,
                           new TGeoCombiTrans(x + kWidth, y - kHeight + 0.04 + filHeight / 2,
                                              z - kStaveLength / 2 + (4 * kL1) + kS1 / 2,
                                              new TGeoRotation("volT1", -90, kAlpha, 0)));
      mechStavVol->AddNode(volT1, 4 * i + 1,
                           new TGeoCombiTrans(x - kWidth, y - kHeight + 0.04 + filHeight / 2,
                                              z - kStaveLength / 2 + (4 * kL1 * i) + kS1 / 2,
                                              new TGeoRotation("volT1", 90, kAlpha, 0)));
      mechStavVol->AddNode(
        volT1, 4 * i + 2,
        new TGeoCombiTrans(x + kWidth, y - kHeight + 0.04 + filHeight / 2,
                           z - kStaveLength / 2 + 2 * kL1 + (i * 4 * kL1) + kS1 / 2,
                           new TGeoRotation("volT1", -90, -kAlpha, 0)));
      mechStavVol->AddNode(
        volT1, 4 * i + 3,
        new TGeoCombiTrans(x - kWidth, y - kHeight + 0.04 + filHeight / 2,
                           z - kStaveLength / 2 + 2 * kL1 + (i * 4 * kL1) + kS1 / 2,
                           new TGeoRotation("volT1", -90, +kAlpha, 0)));
    }

    // Top filament CFRP black-12 Carbon structure TGeoBBox (length,thickness,width)
    auto* t2 = new TGeoBBox(kS2, filHeight / 2, filWidth / 2);
    auto* volT2 = new TGeoVolume("CFRPTop", t2, medM60J3K);
    volT2->SetLineColor(12);
    volT2->SetFillColor(12);
    for (int i = 0; i < loop; i++) { // i<30;i++){
      mechStavVol->AddNode(
        volT2, 4 * i + 0,
        new TGeoCombiTrans(x + kWidth, y + 0.04 + filHeight / 2,
                           z - kStaveLength / 2 + (i * 4 * kL1) + kS1 / 2,
                           new TGeoRotation("volT2", 90, 90 - kAlpha, 90 - kBeta)));
      mechStavVol->AddNode(
        volT2, 4 * i + 1,
        new TGeoCombiTrans(x - kWidth, y + 0.04 + filHeight / 2,
                           z - kStaveLength / 2 + (i * 4 * kL1) + kS1 / 2,
                           new TGeoRotation("volT2", 90, -90 + kAlpha, -90 + kBeta)));
      mechStavVol->AddNode(
        volT2, 4 * i + 2,
        new TGeoCombiTrans(x + kWidth, y + 0.04 + filHeight / 2,
                           z - kStaveLength / 2 + 2 * kL1 + (i * 4 * kL1) + kS1 / 2,
                           new TGeoRotation("volT2", 90, -90 + kAlpha, 90 - kBeta)));
      mechStavVol->AddNode(
        volT2, 4 * i + 3,
        new TGeoCombiTrans(x - kWidth, y + 0.04 + filHeight / 2,
                           z - kStaveLength / 2 + 2 * kL1 + (i * 4 * kL1) + kS1 / 2,
                           new TGeoRotation("volT2", 90, 90 - kAlpha, -90 + kBeta)));
    }
  }

  if (mBuildLevel < 2) {
    // Glue between filament and polimide micro channel
    auto* t3 = new TGeoBBox(0.01 / 2, 0.04, kS1);
    auto* volT3 = new TGeoVolume("FilamentGlue", t3, medGlue);
    volT3->SetLineColor(5);
    volT3->SetFillColor(5);
    for (int i = 0; i < loop; i++) { // i<30;i++){
      mechStavVol->AddNode(volT3, 4 * i + 0,
                           new TGeoCombiTrans(x + kWidth, y - kHeight + 0.0325,
                                              z - kStaveLength / 2 + (4 * kL1 * i) + kS1 / 2,
                                              new TGeoRotation("volT1", -90, kAlpha, 0)));
      mechStavVol->AddNode(volT3, 4 * i + 1,
                           new TGeoCombiTrans(x - kWidth, y - kHeight + 0.0325,
                                              z - kStaveLength / 2 + (4 * kL1 * i) + kS1 / 2,
                                              new TGeoRotation("volT1", 90, kAlpha, 0)));
      mechStavVol->AddNode(
        volT3, 4 * i + 2,
        new TGeoCombiTrans(x + kWidth, y - kHeight + 0.0325,
                           z - kStaveLength / 2 + 2 * kL1 + (i * 4 * kL1) + kS1 / 2,
                           new TGeoRotation("volT1", -90, -kAlpha, 0)));
      mechStavVol->AddNode(
        volT3, 4 * i + 3,
        new TGeoCombiTrans(x - kWidth, y - kHeight + 0.0325,
                           z - kStaveLength / 2 + 2 * kL1 + (i * 4 * kL1) + kS1 / 2,
                           new TGeoRotation("volT1", -90, +kAlpha, 0)));
    }

    // Glue microchannel and sensor
    auto* glueM = new TGeoBBox(xsta, 0.01 / 2, zsta);
    auto* volGlueM = new TGeoVolume("MicroChanGlue", glueM, medGlue);
    volGlueM->SetLineColor(5);
    volGlueM->SetFillColor(5);
    mechStavVol->AddNode(volGlueM, 0,
                         new TGeoCombiTrans(x, y - 0.16, z, new TGeoRotation("", 0, 0, 0)));

    // Glue sensor and kapton
    auto* glue = new TGeoBBox(xsta, 0.005 / 2, zsta);
    auto* volGlue = new TGeoVolume("SensorGlue", glue, medGlue);
    volGlue->SetLineColor(5);
    volGlue->SetFillColor(5);
    mechStavVol->AddNode(volGlue, 1, new TGeoCombiTrans(x, y - 0.165 - mSensorThickness - 0.005, z, new TGeoRotation("", 0, 0, 0)));
  }

  if (mBuildLevel < 1) {
    auto* kapCable = new TGeoBBox(xsta, 0.01 / 2, zsta);
    auto* volCable = new TGeoVolume("FlexCable", kapCable, medFlexCable);
    volCable->SetLineColor(28);
    volCable->SetFillColor(28);
    mechStavVol->AddNode(volCable, 0,
                         new TGeoCombiTrans(x, y - 0.165 - mSensorThickness - 0.005 - 0.01, z,
                                            new TGeoRotation("", 0, 0, 0)));
  }
  // Done, return the stave structur
  return mechStavVol;
}

TGeoVolume* V1Layer::createStaveModelInnerB21(const Double_t xsta, const Double_t zsta,
                                              const TGeoManager* mgr)
{
  // Materials defined in Detector
  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");
  TGeoMedium* medWater = mgr->GetMedium("ITS_WATER$");

  TGeoMedium* medM60J3K = mgr->GetMedium("ITS_M60J3K$");
  TGeoMedium* medKapton = mgr->GetMedium("ITS_KAPTON(POLYCH2)$");
  TGeoMedium* medGlue = mgr->GetMedium("ITS_GLUE$");
  TGeoMedium* medFlexCable = mgr->GetMedium("ITS_FLEXCABLE$");
  TGeoMedium* medK13D2U2k = mgr->GetMedium("ITS_K13D2U2k$");
  TGeoMedium* medFGS003 = mgr->GetMedium("ITS_FGS003$");
  TGeoMedium* medCarbonFleece = mgr->GetMedium("ITS_CarbonFleece$");

  // Local parameters
  Double_t kConeOutRadius = 0.151384 / 2;
  Double_t kConeInRadius = 0.145034 / 2;
  Double_t kStaveLength = zsta;
  Double_t kStaveWidth = xsta * 2;
  Double_t kWidth = (kStaveWidth + 0.005) / 4;
  Double_t kStaveHeigth = 0.33; // 0.33;
  Double_t kHeight = (kStaveHeigth + 0.025) / 2;
  Double_t kAlpha = 57; // 56.31;
  Double_t kTheta = kAlpha * TMath::DegToRad();
  Double_t kS1 = (kStaveWidth / 4) / TMath::Sin(kTheta);
  Double_t kL1 = (kStaveWidth / 4) / TMath::Tan(kTheta);
  Double_t kS2 = sqrt(kHeight * kHeight + kS1 * kS1); // TMath::Sin(the2);
  Double_t kThe2 = TMath::ATan(kHeight / kS1);
  Double_t kBeta = kThe2 * TMath::RadToDeg();
  // Double_t lay1 = 0.003157;
  Double_t kLay1 = 0.003; // Amec carbon
  // Double_t lay2 = 0.0043215;//C Fleece carbon
  Double_t kLay2 = 0.002; // C Fleece carbon
  Double_t kLay3 = 0.007; // K13D2U carbon
  Int_t loop = (Int_t)(kStaveLength / (2 * kL1));

  char volumeName[30];
  snprintf(volumeName, 30, "%s%d_StaveStruct", GeometryTGeo::getITSStavePattern(),
           mLayerNumber);

  Double_t z = 0, y = -(kConeOutRadius + 0.03) + 0.0385, x = 0;

  TGeoVolume* mechStavVol = nullptr;

  if (mBuildLevel < 5) {
    // world (trapezoid)
    auto* mechStruct = new TGeoXtru(2); // z sections
    Double_t xv[5] = {
      kStaveWidth / 2 + 0.1, kStaveWidth / 2 + 0.1, 0, -kStaveWidth / 2 - 0.1,
      -kStaveWidth / 2 - 0.1};
    Double_t yv[5] = {-kConeOutRadius * 2 - 0.07, 0, kStaveHeigth, 0, -kConeOutRadius * 2 - 0.07};
    mechStruct->DefinePolygon(5, xv, yv);
    mechStruct->DefineSection(0, -kStaveLength - 0.1, 0, 0, 1.);
    mechStruct->DefineSection(1, kStaveLength + 0.1, 0, 0, 1.);

    mechStavVol = new TGeoVolume(volumeName, mechStruct, medAir);
    mechStavVol->SetLineColor(12);
    mechStavVol->SetFillColor(12);
    mechStavVol->SetVisibility(kTRUE);

    // Pipe Kapton grey-35
    auto* cone1 =
      new TGeoCone(kStaveLength, kConeInRadius, kConeOutRadius, kConeInRadius, kConeOutRadius);
    auto* volCone1 = new TGeoVolume("PolyimidePipe", cone1, medKapton);
    volCone1->SetFillColor(35);
    volCone1->SetLineColor(35);
    mechStavVol->AddNode(volCone1, 1, new TGeoTranslation(x + 0.25, y, z));
    mechStavVol->AddNode(volCone1, 2, new TGeoTranslation(x - 0.25, y, z));
  }

  if (mBuildLevel < 4) {
    auto* coolTubeW = new TGeoTube(0., kConeInRadius, kStaveLength);
    auto* volCoolTubeW = new TGeoVolume("Water", coolTubeW, medWater);
    volCoolTubeW->SetFillColor(4);
    volCoolTubeW->SetLineColor(4);
    mechStavVol->AddNode(volCoolTubeW, 0, new TGeoTranslation(x - 0.25, y, z));
    mechStavVol->AddNode(volCoolTubeW, 1, new TGeoTranslation(x + 0.25, y, z));
  }

  if (mBuildLevel < 3) {
    // top fillament
    // Top filament M60J black-12 Carbon structure TGeoBBox (length,thickness,width)
    auto* t2 =
      new TGeoBBox(kS2, 0.02 / 2, 0.04 / 2); // TGeoBBox *t2=new TGeoBBox(kS2,0.01,0.02);
    auto* volT2 = new TGeoVolume("TopFilament", t2, medM60J3K);
    volT2->SetLineColor(12);
    volT2->SetFillColor(12);

    for (int i = 0; i < loop; i++) { // i<28;i++){
      mechStavVol->AddNode(
        volT2, i * 4 + 1,
        new TGeoCombiTrans(x + kWidth, y + kHeight + (0.12 / 2) - 0.014 + 0.007,
                           z - kStaveLength + (i * 4 * kL1) + kS1 / 2,
                           new TGeoRotation("volT2", 90, 90 - kAlpha, 90 - kBeta)));
      mechStavVol->AddNode(
        volT2, i * 4 + 2,
        new TGeoCombiTrans(x - kWidth, y + kHeight + (0.12 / 2) - 0.014 + 0.007,
                           z - kStaveLength + (i * 4 * kL1) + kS1 / 2,
                           new TGeoRotation("volT2", 90, -90 + kAlpha, -90 + kBeta)));
      mechStavVol->AddNode(
        volT2, i * 4 + 3,
        new TGeoCombiTrans(x + kWidth, y + kHeight + (0.12 / 2) - 0.014 + 0.007,
                           z - kStaveLength + 2 * kL1 + (i * 4 * kL1) + kS1 / 2,
                           new TGeoRotation("volT2", 90, -90 + kAlpha, 90 - kBeta)));
      mechStavVol->AddNode(
        volT2, i * 4 + 4,
        new TGeoCombiTrans(x - kWidth, y + kHeight + (0.12 / 2) - 0.014 + 0.007,
                           z - kStaveLength + 2 * kL1 + (i * 4 * kL1) + kS1 / 2,
                           new TGeoRotation("volT2", 90, 90 - kAlpha, -90 + kBeta)));
      //    mechStavVol->AddNode(volT2,i*4+1,new
      // TGeoCombiTrans(x+kWidth+0.0036,y+kHeight-(0.12/2)+0.072,z+kStaveLength+(i*4*kL1)+kS1/2, new
      // TGeoRotation("volT2",90,90-kAlpha,90-kBeta)));
    }

    // wall side structure out
    auto* box4 = new TGeoBBox(0.03 / 2, 0.12 / 2, kStaveLength - 0.50);
    auto* plate4 = new TGeoVolume("WallOut", box4, medM60J3K);
    plate4->SetFillColor(35);
    plate4->SetLineColor(35);
    mechStavVol->AddNode(plate4, 1,
                         new TGeoCombiTrans(x + (2 * kStaveWidth / 4) - (0.03 / 2),
                                            y - 0.0022 - kConeOutRadius + 0.12 / 2 + 0.007, z,
                                            new TGeoRotation("plate4", 0, 0, 0)));
    mechStavVol->AddNode(plate4, 2,
                         new TGeoCombiTrans(x - (2 * kStaveWidth / 4) + (0.03 / 2),
                                            y - 0.0022 - kConeOutRadius + 0.12 / 2 + 0.007, z,
                                            new TGeoRotation("plate4", 0, 0, 0)));
    // wall side in
    auto* box5 = new TGeoBBox(0.015 / 2, 0.12 / 2, kStaveLength - 0.50);
    auto* plate5 = new TGeoVolume("WallIn", box5, medM60J3K);
    plate5->SetFillColor(12);
    plate5->SetLineColor(12);
    mechStavVol->AddNode(plate5, 1,
                         new TGeoCombiTrans(x + (2 * kStaveWidth / 4) - 0.03 - 0.015 / 2,
                                            y - 0.0022 - kConeOutRadius + 0.12 / 2 + 0.007, z,
                                            new TGeoRotation("plate5", 0, 0, 0)));
    mechStavVol->AddNode(plate5, 2,
                         new TGeoCombiTrans(x - (2 * kStaveWidth / 4) + 0.03 + 0.015 / 2,
                                            y - 0.0022 - kConeOutRadius + 0.12 / 2 + 0.007, z,
                                            new TGeoRotation("plate5", 0, 0, 0)));

    // Amec Thermasol red-2 cover tube FGS300
    auto* cons1 =
      new TGeoConeSeg(kStaveLength - 0.50, kConeOutRadius, kConeOutRadius + kLay1, kConeOutRadius,
                      kConeOutRadius + kLay1, 0, 180);
    auto* cone11 = new TGeoVolume("ThermasolPipeCover", cons1, medFGS003);
    cone11->SetFillColor(2);
    cone11->SetLineColor(2);
    mechStavVol->AddNode(cone11, 1,
                         new TGeoCombiTrans(x + 0.25, y, z, new TGeoRotation("Cone11", 0, 0, 0)));
    mechStavVol->AddNode(cone11, 2,
                         new TGeoCombiTrans(x - 0.25, y, z, new TGeoRotation("Cone11", 0, 0, 0)));

    auto* box2 =
      new TGeoBBox((0.50 - (2 * kConeOutRadius)) / 2, kLay1 / 2, kStaveLength - 0.50);
    auto* plate2 = new TGeoVolume("ThermasolMiddle", box2, medFGS003);
    plate2->SetFillColor(2);
    plate2->SetLineColor(2);
    mechStavVol->AddNode(plate2, 1, new TGeoCombiTrans(x, y - kConeOutRadius + (kLay1 / 2), z, new TGeoRotation("plate2", 0, 0, 0)));

    auto* box21 =
      new TGeoBBox((0.75 - 0.25 - kConeOutRadius - kLay1) / 2, kLay1 / 2, kStaveLength - 0.50);
    auto* plate21 = new TGeoVolume("ThermasolLeftRight", box21, medFGS003);
    plate21->SetFillColor(2);
    plate21->SetLineColor(2);
    mechStavVol->AddNode(
      plate21, 1, new TGeoCombiTrans(x + 0.25 + kConeOutRadius + (0.75 - 0.25 - kConeOutRadius) / 2 - (kLay1 / 2), y - kConeOutRadius + (kLay1 / 2), z, new TGeoRotation("plate21", 0, 0, 0)));
    mechStavVol->AddNode(
      plate21, 2, new TGeoCombiTrans(x - 0.25 - kConeOutRadius - (0.75 - 0.25 - kConeOutRadius) / 2 + (kLay1 / 2), y - kConeOutRadius + (kLay1 / 2), z, new TGeoRotation("plate21", 0, 0, 0)));

    auto* box22 = new TGeoBBox((kLay1 / 2), kConeOutRadius / 2, kStaveLength - 0.50);
    auto* plate22 = new TGeoVolume("ThermasolVertical", box22, medFGS003);
    plate22->SetFillColor(2);
    plate22->SetLineColor(2);
    mechStavVol->AddNode(plate22, 1, new TGeoCombiTrans(x + 0.25 + kConeOutRadius + (kLay1 / 2), y - kConeOutRadius / 2, z, new TGeoRotation("plate22", 0, 0, 0)));
    mechStavVol->AddNode(plate22, 2, new TGeoCombiTrans(x + 0.25 - kConeOutRadius - (kLay1 / 2), y - kConeOutRadius / 2, z, new TGeoRotation("plate22", 0, 0, 0)));
    mechStavVol->AddNode(plate22, 3, new TGeoCombiTrans(x - 0.25 + kConeOutRadius + (kLay1 / 2), y - kConeOutRadius / 2, z, new TGeoRotation("plate22", 0, 0, 0)));
    mechStavVol->AddNode(plate22, 4, new TGeoCombiTrans(x - 0.25 - kConeOutRadius - (kLay1 / 2), y - kConeOutRadius / 2, z, new TGeoRotation("plate22", 0, 0, 0)));

    // C Fleece
    auto* cons2 =
      new TGeoConeSeg(kStaveLength - 0.50, kConeOutRadius + kLay1, kConeOutRadius + kLay1 + kLay2,
                      kConeOutRadius + kLay1, kConeOutRadius + kLay1 + kLay2, 0, 180);
    auto* cone12 = new TGeoVolume("CFleecePipeCover", cons2, medCarbonFleece);
    cone12->SetFillColor(28);
    cone12->SetLineColor(28);
    mechStavVol->AddNode(cone12, 1,
                         new TGeoCombiTrans(x + 0.25, y, z, new TGeoRotation("Cone12", 0, 0, 0)));
    mechStavVol->AddNode(cone12, 2,
                         new TGeoCombiTrans(x - 0.25, y, z, new TGeoRotation("Cone12", 0, 0, 0)));

    auto* box3 =
      new TGeoBBox((0.50 - (2 * (kConeOutRadius + kLay1))) / 2, kLay2 / 2, kStaveLength - 0.50);
    auto* plate3 = new TGeoVolume("CFleeceMiddle", box3, medCarbonFleece);
    plate3->SetFillColor(28);
    plate3->SetLineColor(28);
    mechStavVol->AddNode(plate3, 1, new TGeoCombiTrans(x, y - kConeOutRadius + kLay1 + (kLay2 / 2), z, new TGeoRotation("plate3", 0, 0, 0)));

    auto* box31 =
      new TGeoBBox((0.75 - 0.25 - kConeOutRadius - kLay1) / 2, kLay2 / 2, kStaveLength - 0.50);
    auto* plate31 = new TGeoVolume("CFleeceLeftRight", box31, medCarbonFleece);
    plate31->SetFillColor(28);
    plate31->SetLineColor(28);
    mechStavVol->AddNode(
      plate31, 1,
      new TGeoCombiTrans(
        x + 0.25 + kConeOutRadius + kLay1 + (0.75 - 0.25 - kConeOutRadius - kLay1) / 2,
        y - kConeOutRadius + kLay1 + (kLay2 / 2), z, new TGeoRotation("plate31", 0, 0, 0)));
    mechStavVol->AddNode(
      plate31, 2,
      new TGeoCombiTrans(
        x - 0.25 - kConeOutRadius - kLay1 - (0.75 - 0.25 - kConeOutRadius - kLay1) / 2,
        y - kConeOutRadius + kLay1 + (kLay2 / 2), z, new TGeoRotation("plate31", 0, 0, 0)));

    auto* box32 = new TGeoBBox((kLay2 / 2), (kConeOutRadius - kLay1) / 2, kStaveLength - 0.50);
    auto* plate32 = new TGeoVolume("CFleeceVertical", box32, medCarbonFleece);
    plate32->SetFillColor(28);
    plate32->SetLineColor(28);
    mechStavVol->AddNode(plate32, 1,
                         new TGeoCombiTrans(x + 0.25 + kConeOutRadius + kLay1 + (kLay2 / 2),
                                            y + (kLay1 - kConeOutRadius) / 2, z,
                                            new TGeoRotation("plate32", 0, 0, 0)));
    mechStavVol->AddNode(plate32, 2,
                         new TGeoCombiTrans(x + 0.25 - kConeOutRadius - kLay1 - (kLay2 / 2),
                                            y + (kLay1 - kConeOutRadius) / 2, z,
                                            new TGeoRotation("plate32", 0, 0, 0)));
    mechStavVol->AddNode(plate32, 3,
                         new TGeoCombiTrans(x - 0.25 + kConeOutRadius + kLay1 + (kLay2 / 2),
                                            y + (kLay1 - kConeOutRadius) / 2, z,
                                            new TGeoRotation("plate32", 0, 0, 0)));
    mechStavVol->AddNode(plate32, 4,
                         new TGeoCombiTrans(x - 0.25 - kConeOutRadius - kLay1 - (kLay2 / 2),
                                            y + (kLay1 - kConeOutRadius) / 2, z,
                                            new TGeoRotation("plate32", 0, 0, 0)));

    // K13D2U carbon plate
    auto* box1 = new TGeoBBox(2 * kWidth, kLay3 / 2, kStaveLength - 0.50);
    auto* plate1 = new TGeoVolume("CarbonPlate", box1, medK13D2U2k);
    plate1->SetFillColor(5);
    plate1->SetLineColor(5);
    mechStavVol->AddNode(plate1, 1, new TGeoCombiTrans(x, y - (kConeOutRadius + (kLay3 / 2)), z, new TGeoRotation("plate1", 0, 0, 0)));

    // C Fleece bottom plate
    auto* box6 = new TGeoBBox(2 * kWidth, kLay2 / 2, kStaveLength - 0.50);
    auto* plate6 = new TGeoVolume("CFleeceBottom", box6, medCarbonFleece);
    plate6->SetFillColor(2);
    plate6->SetLineColor(2);
    mechStavVol->AddNode(plate6, 1,
                         new TGeoCombiTrans(x, y - (kConeOutRadius + kLay3 + (kLay2 / 2)), z,
                                            new TGeoRotation("plate1", 0, 0, 0)));
  }

  if (mBuildLevel < 2) {
    // Glue layers and kapton
    auto* glue = new TGeoBBox(kStaveWidth / 2, 0.005 / 2, zsta);
    auto* volGlue = new TGeoVolume("Glue", glue, medGlue);
    volGlue->SetLineColor(5);
    volGlue->SetFillColor(5);
    mechStavVol->AddNode(
      volGlue, 0, new TGeoCombiTrans(x, y - (kConeOutRadius + kLay3 + (kLay2 / 2) + (0.01 / 2)), z, new TGeoRotation("", 0, 0, 0)));
    mechStavVol->AddNode(volGlue, 1,
                         new TGeoCombiTrans(x, y - (kConeOutRadius + kLay3 + (kLay2 / 2) + 0.01 + mSensorThickness + (0.01 / 2)),
                                            z, new TGeoRotation("", 0, 0, 0)));
  }

  if (mBuildLevel < 1) {
    auto* kapCable = new TGeoBBox(kStaveWidth / 2, 0.01 / 2, zsta);
    auto* volCable = new TGeoVolume("FlexCable", kapCable, medFlexCable);
    volCable->SetLineColor(28);
    volCable->SetFillColor(28);
    mechStavVol->AddNode(volCable, 0,
                         new TGeoCombiTrans(x, y - (kConeOutRadius + kLay3 + (kLay2 / 2) + 0.01 + mSensorThickness + 0.01 + (0.01 / 2)),
                                            z, new TGeoRotation("", 0, 0, 0)));
  }
  // Done, return the stave structure
  return mechStavVol;
}

// new model22
TGeoVolume* V1Layer::createStaveModelInnerB22(const Double_t xsta, const Double_t zsta,
                                              const TGeoManager* mgr)
{
  // Materials defined in Detector
  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");
  TGeoMedium* medWater = mgr->GetMedium("ITS_WATER$");

  TGeoMedium* medM60J3K = mgr->GetMedium("ITS_M60J3K$");
  TGeoMedium* medKapton = mgr->GetMedium("ITS_KAPTON(POLYCH2)$");
  TGeoMedium* medGlue = mgr->GetMedium("ITS_GLUE$");
  TGeoMedium* medFlexCable = mgr->GetMedium("ITS_FLEXCABLE$");
  TGeoMedium* medK13D2U2k = mgr->GetMedium("ITS_K13D2U2k$");
  TGeoMedium* medFGS003 = mgr->GetMedium("ITS_FGS003$");
  TGeoMedium* medCarbonFleece = mgr->GetMedium("ITS_CarbonFleece$");

  // Local parameters
  Double_t kConeOutRadius = (0.1024 + 0.0025) / 2; // 0.107/2;
  Double_t kConeInRadius = 0.1024 / 2;             // 0.10105/2
  Double_t kStaveLength = zsta;
  Double_t kStaveWidth = xsta * 2;
  Double_t kWidth = (kStaveWidth) / 4;
  Double_t kStaveHeight = 0.283; // 0.33;
  Double_t kHeight = (kStaveHeight) / 2;
  Double_t kAlpha = 57; // 56.31;
  Double_t kTheta = kAlpha * TMath::DegToRad();
  Double_t kS1 = ((kStaveWidth) / 4) / TMath::Sin(kTheta);
  Double_t kL1 = (kStaveWidth / 4) / TMath::Tan(kTheta);
  Double_t kS2 = sqrt(kHeight * kHeight + kS1 * kS1); // TMath::Sin(kThe2);
  Double_t kThe2 = TMath::ATan(kHeight / (0.375 - 0.036));
  Double_t kBeta = kThe2 * TMath::RadToDeg();
  Double_t klay1 = 0.003; // Amec carbon
  Double_t klay2 = 0.002; // C Fleece carbon
  Double_t klay3 = 0.007; // CFplate K13D2U carbon
  Double_t klay4 = 0.007; // GluekStaveLength/2
  Double_t klay5 = 0.01;  // Flex cable
  Double_t kTopVertexMaxWidth = 0.072;
  Double_t kTopVertexHeight = 0.04;
  Double_t kSideVertexMWidth = 0.052;
  Double_t kSideVertexHeight = 0.11;

  Int_t loop = (Int_t)(kStaveLength / (2 * kL1));

  char volumeName[30];
  snprintf(volumeName, 30, "%s%d_StaveStruct", GeometryTGeo::getITSStavePattern(),
           mLayerNumber);

  Double_t z = 0, y = -(2 * kConeOutRadius) + klay1 + klay2 + mSensorThickness / 2 - 0.0004, x = 0;

  TGeoVolume* mechStavVol = nullptr;

  if (mBuildLevel < 5) {
    // world (trapezoid)
    auto* mechStruct = new TGeoXtru(2); // z sections
    Double_t xv[6] = {
      kStaveWidth / 2, kStaveWidth / 2, 0.012,
      -0.012, -kStaveWidth / 2, -kStaveWidth / 2};
    // Double_t yv[6] = {-2*(kConeOutRadius+klay1+1.5*klay2+klay3+klay4+mSensorThickness+klay5),
    //                   0-0.02,kStaveHeight+0.01,kStaveHeight+0.01,0-0.02,
    // -2*(kConeOutRadius+klay1+1.5*klay2+klay3+klay4+mSensorThickness+klay5)};
    // (kConeOutRadius*2)-0.0635
    Double_t yv[6] = {
      -(kConeOutRadius * 2) - 0.06395, 0 - 0.02, kStaveHeight + 0.01,
      kStaveHeight + 0.01, 0 - 0.02, -(kConeOutRadius * 2) - 0.06395}; // (kConeOutRadius*2)-0.064
    mechStruct->DefinePolygon(6, xv, yv);
    mechStruct->DefineSection(0, -kStaveLength, 0, 0, 1.);
    mechStruct->DefineSection(1, kStaveLength, 0, 0, 1.);

    mechStavVol = new TGeoVolume(volumeName, mechStruct, medAir);
    mechStavVol->SetLineColor(12);
    mechStavVol->SetFillColor(12);
    mechStavVol->SetVisibility(kTRUE);

    // Polyimide Pipe Kapton grey-35
    auto* cone1 = new TGeoCone(kStaveLength, kConeInRadius, kConeOutRadius - 0.0001,
                               kConeInRadius, kConeOutRadius - 0.0001);
    auto* volCone1 = new TGeoVolume("PolyimidePipe", cone1, medKapton);
    volCone1->SetFillColor(35);
    volCone1->SetLineColor(35);
    mechStavVol->AddNode(volCone1, 1, new TGeoTranslation(x + 0.25, y, z));
    mechStavVol->AddNode(volCone1, 2, new TGeoTranslation(x - 0.25, y, z));
  }

  if (mBuildLevel < 4) {
    auto* coolTubeW = new TGeoTube(0., kConeInRadius - 0.0001, kStaveLength);
    auto* volCoolTubeW = new TGeoVolume("Water", coolTubeW, medWater);
    volCoolTubeW->SetFillColor(4);
    volCoolTubeW->SetLineColor(4);
    mechStavVol->AddNode(volCoolTubeW, 0, new TGeoTranslation(x - 0.25, y, z));
    mechStavVol->AddNode(volCoolTubeW, 1, new TGeoTranslation(x + 0.25, y, z));
  }

  if (mBuildLevel < 3) {
    // top fillament
    // Top filament M60J black-12 Carbon structure TGeoBBox (length,thickness,width)
    auto* t2 = new TGeoBBox(
      kS2 - 0.028, 0.02 / 2,
      0.02 / 2); // 0.04/2//TGeoBBox *t2=new TGeoBBox(kS2,0.01,0.02);//kS2-0.03 old Config.C
    auto* volT2 = new TGeoVolume("TopFilament", t2, medM60J3K);
    volT2->SetLineColor(12);
    volT2->SetFillColor(12);
    for (int i = 0; i < loop; i++) { // i<28;i++){
      // 1) Front Left Top Filament
      mechStavVol->AddNode(
        volT2, i * 4 + 1,
        new TGeoCombiTrans(x + kWidth + 0.0036, y + kHeight + 0.01,
                           z - kStaveLength + 0.1 + (i * 4 * kL1) + kS1 / 2,
                           new TGeoRotation("volT2", 90, 90 - kAlpha, 90 - kBeta)));
      // 2) Front Right Top Filament
      mechStavVol->AddNode(
        volT2, i * 4 + 2,
        new TGeoCombiTrans(x - kWidth - 0.0036, y + kHeight + 0.01,
                           z - kStaveLength + 0.1 + (i * 4 * kL1) + kS1 / 2,
                           new TGeoRotation("volT2", 90, -90 + kAlpha, -90 + kBeta)));
      // 3) Back Left  Top Filament
      mechStavVol->AddNode(
        volT2, i * 4 + 3,
        new TGeoCombiTrans(x + kWidth + 0.0036, y + kHeight + 0.01,
                           z - kStaveLength + 0.1 + 2 * kL1 + (i * 4 * kL1) + kS1 / 2,
                           new TGeoRotation("volT2", 90, -90 + kAlpha, 90 - kBeta)));
      // 4) Back Right Top Filament
      mechStavVol->AddNode(
        volT2, i * 4 + 4,
        new TGeoCombiTrans(x - kWidth - 0.0036, y + kHeight + 0.01,
                           z - kStaveLength + 0.1 + 2 * kL1 + (i * 4 * kL1) + kS1 / 2,
                           new TGeoRotation("volT2", 90, 90 - kAlpha, -90 + kBeta)));
    }

    // Vertex  structure
    // top ver trd1
    auto* trd1 = new TGeoTrd1(0, kTopVertexMaxWidth / 2, kStaveLength, kTopVertexHeight / 2);
    auto* ibdv = new TGeoVolume("TopVertex", trd1, medM60J3K);
    ibdv->SetFillColor(12);
    ibdv->SetLineColor(12);
    mechStavVol->AddNode(
      ibdv, 1, new TGeoCombiTrans(x, y + kStaveHeight + 0.03, z, new TGeoRotation("ibdv", 0., -90, 0))); // y+kStaveHeight+0.056

    // left trd2
    auto* trd2 = new TGeoTrd1(0, kSideVertexMWidth / 2, kStaveLength, kSideVertexHeight / 2);
    auto* ibdv2 = new TGeoVolume("LeftVertex", trd2, medM60J3K);
    ibdv2->SetFillColor(12);
    ibdv2->SetLineColor(12);
    mechStavVol->AddNode(
      ibdv2, 1,
      new TGeoCombiTrans(
        x + kStaveWidth / 2 - 0.06, y - 0.0355, z,
        new TGeoRotation("ibdv2", -103.3, 90, 0))); // x-kStaveWidth/2-0.09 old Config.C y-0.0355,

    // right trd3
    auto* trd3 = new TGeoTrd1(0, kSideVertexMWidth / 2, kStaveLength, kSideVertexHeight / 2);
    auto* ibdv3 = new TGeoVolume("RightVertex", trd3, medM60J3K);
    ibdv3->SetFillColor(12);
    ibdv3->SetLineColor(12);
    mechStavVol->AddNode(
      ibdv3, 1, new TGeoCombiTrans(x - kStaveWidth / 2 + 0.06, y - 0.0355, z, new TGeoRotation("ibdv3", 103.3, 90, 0))); // x-kStaveWidth/2+0.09 old Config.C

    // Carbon Fleece
    auto* cons2 =
      new TGeoConeSeg(zsta, kConeOutRadius + klay1, kConeOutRadius + klay1 + klay2,
                      kConeOutRadius + klay1, kConeOutRadius + klay1 + klay2, 0, 180);
    auto* cone12 = new TGeoVolume("CarbonFleecePipeCover", cons2, medCarbonFleece);
    cone12->SetFillColor(28);
    cone12->SetLineColor(28);
    mechStavVol->AddNode(cone12, 1,
                         new TGeoCombiTrans(x + 0.25, y, z, new TGeoRotation("cone12", 0, 0, 0)));
    mechStavVol->AddNode(cone12, 2,
                         new TGeoCombiTrans(x - 0.25, y, z, new TGeoRotation("cone12", 0, 0, 0)));

    auto* box3 = new TGeoBBox((0.50 - (2 * (kConeOutRadius + klay1))) / 2, klay2 / 2,
                              zsta); // kStaveLength-0.50);
    auto* plate3 = new TGeoVolume("CarbonFleeceMiddle", box3, medCarbonFleece);
    plate3->SetFillColor(28);
    plate3->SetLineColor(28);
    mechStavVol->AddNode(plate3, 1, new TGeoCombiTrans(x, y - kConeOutRadius + klay1 + (klay2 / 2), z, new TGeoRotation("plate3", 0, 0, 0)));

    auto* box31 =
      new TGeoBBox((0.75 - 0.25 - kConeOutRadius - klay1) / 2 + 0.0025, klay2 / 2, zsta);
    auto* plate31 = new TGeoVolume("CarbonFleeceLeftRight", box31, medCarbonFleece);
    plate31->SetFillColor(28);
    plate31->SetLineColor(28);
    mechStavVol->AddNode(
      plate31, 1,
      new TGeoCombiTrans(
        x + 0.25 + kConeOutRadius + klay1 + (0.75 - 0.25 - kConeOutRadius - klay1) / 2,
        y - kConeOutRadius + klay1 + (klay2 / 2), z, new TGeoRotation("plate31", 0, 0, 0)));
    mechStavVol->AddNode(
      plate31, 2,
      new TGeoCombiTrans(
        x - 0.25 - kConeOutRadius - klay1 - (0.75 - 0.25 - kConeOutRadius - klay1) / 2,
        y - kConeOutRadius + klay1 + (klay2 / 2), z, new TGeoRotation("plate31", 0, 0, 0)));

    auto* box32 = new TGeoBBox((klay2 / 2), (kConeOutRadius - klay1) / 2, zsta);
    auto* plate32 = new TGeoVolume("CarbonFleeceVertical", box32, medCarbonFleece);
    plate32->SetFillColor(28);
    plate32->SetLineColor(28);
    mechStavVol->AddNode(plate32, 1,
                         new TGeoCombiTrans(x + 0.25 + kConeOutRadius + klay1 + (klay2 / 2),
                                            y + (klay1 - kConeOutRadius) / 2, z,
                                            new TGeoRotation("plate32", 0, 0, 0)));
    mechStavVol->AddNode(plate32, 2,
                         new TGeoCombiTrans(x + 0.25 - kConeOutRadius - klay1 - (klay2 / 2),
                                            y + (klay1 - kConeOutRadius) / 2, z,
                                            new TGeoRotation("plate32", 0, 0, 0)));
    mechStavVol->AddNode(plate32, 3,
                         new TGeoCombiTrans(x - 0.25 + kConeOutRadius + klay1 + (klay2 / 2),
                                            y + (klay1 - kConeOutRadius) / 2, z,
                                            new TGeoRotation("plate32", 0, 0, 0)));
    mechStavVol->AddNode(plate32, 4,
                         new TGeoCombiTrans(x - 0.25 - kConeOutRadius - klay1 - (klay2 / 2),
                                            y + (klay1 - kConeOutRadius) / 2, z,
                                            new TGeoRotation("plate32", 0, 0, 0)));

    // Amec Thermasol red-2 cover tube FGS300 or Carbon Paper
    auto* cons1 =
      new TGeoConeSeg(zsta, kConeOutRadius, kConeOutRadius + klay1 - 0.0001, kConeOutRadius,
                      kConeOutRadius + klay1 - 0.0001, 0, 180); // kConeOutRadius+klay1-0.0001
    auto* cone11 = new TGeoVolume("ThermasolPipeCover", cons1, medFGS003);
    cone11->SetFillColor(2);
    cone11->SetLineColor(2);
    mechStavVol->AddNode(cone11, 1,
                         new TGeoCombiTrans(x + 0.25, y, z, new TGeoRotation("cone11", 0, 0, 0)));
    mechStavVol->AddNode(cone11, 2,
                         new TGeoCombiTrans(x - 0.25, y, z, new TGeoRotation("cone11", 0, 0, 0)));

    auto* box2 =
      new TGeoBBox((0.50 - (2 * kConeOutRadius)) / 2, (klay1 / 2), zsta); // kStaveLength-0.50);
    auto* plate2 = new TGeoVolume("ThermasolMiddle", box2, medFGS003);
    plate2->SetFillColor(2);
    plate2->SetLineColor(2);
    mechStavVol->AddNode(plate2, 1, new TGeoCombiTrans(x, y - kConeOutRadius + (klay1 / 2), z, new TGeoRotation("plate2", 0, 0, 0)));

    auto* box21 =
      new TGeoBBox((0.75 - 0.25 - kConeOutRadius - klay1) / 2 + 0.0025, (klay1 / 2), zsta);
    auto* plate21 = new TGeoVolume("ThermasolLeftRight", box21, medFGS003);
    plate21->SetFillColor(2);
    plate21->SetLineColor(2);
    mechStavVol->AddNode(
      plate21, 1,
      new TGeoCombiTrans(
        x + 0.25 + kConeOutRadius + (0.75 - 0.25 - kConeOutRadius) / 2 - (klay1 / 2) + 0.0025,
        y - kConeOutRadius + (klay1 / 2), z, new TGeoRotation("plate21", 0, 0, 0)));
    mechStavVol->AddNode(
      plate21, 2,
      new TGeoCombiTrans(
        x - 0.25 - kConeOutRadius - (0.75 - 0.25 - kConeOutRadius) / 2 + (klay1 / 2) - 0.0025,
        y - kConeOutRadius + (klay1 / 2), z, new TGeoRotation("plate21", 0, 0, 0)));

    auto* box22 = new TGeoBBox((klay1 / 2), kConeOutRadius / 2, zsta);
    auto* plate22 = new TGeoVolume("ThermasolVertical", box22, medFGS003);
    plate22->SetFillColor(2);
    plate22->SetLineColor(2);
    mechStavVol->AddNode(plate22, 1, new TGeoCombiTrans(x + 0.25 + kConeOutRadius + (klay1 / 2), y - kConeOutRadius / 2, z, new TGeoRotation("plate22", 0, 0, 0)));
    mechStavVol->AddNode(plate22, 2, new TGeoCombiTrans(x + 0.25 - kConeOutRadius - (klay1 / 2), y - kConeOutRadius / 2, z, new TGeoRotation("plate22", 0, 0, 0)));
    mechStavVol->AddNode(plate22, 3, new TGeoCombiTrans(x - 0.25 + kConeOutRadius + (klay1 / 2), y - kConeOutRadius / 2, z, new TGeoRotation("plate22", 0, 0, 0)));
    mechStavVol->AddNode(plate22, 4, new TGeoCombiTrans(x - 0.25 - kConeOutRadius - (klay1 / 2), y - kConeOutRadius / 2, z, new TGeoRotation("plate22", 0, 0, 0)));

    // K13D2U CF plate
    auto* box1 = new TGeoBBox(2 * kWidth, (klay3) / 2, zsta);
    auto* plate1 = new TGeoVolume("CFPlate", box1, medK13D2U2k);
    plate1->SetFillColor(5);
    plate1->SetLineColor(5);
    mechStavVol->AddNode(plate1, 1, new TGeoCombiTrans(x, y - (kConeOutRadius + (klay3 / 2)), z, new TGeoRotation("plate1", 0, 0, 0)));

    // C Fleece bottom plate
    auto* box6 = new TGeoBBox(2 * kWidth, (klay2) / 2, zsta);
    auto* plate6 = new TGeoVolume("CarbonFleeceBottom", box6, medCarbonFleece);
    plate6->SetFillColor(2);
    plate6->SetLineColor(2);
    mechStavVol->AddNode(plate6, 1,
                         new TGeoCombiTrans(x, y - (kConeOutRadius + klay3 + (klay2 / 2)), z,
                                            new TGeoRotation("plate6", 0, 0, 0)));
  }
  if (mBuildLevel < 2) {
    // Glue klayers and kapton
    auto* glue = new TGeoBBox(kStaveWidth / 2, (klay4) / 2, zsta);
    auto* volGlue = new TGeoVolume("Glue", glue, medGlue);
    volGlue->SetLineColor(5);
    volGlue->SetFillColor(5);
    // mechStavVol->AddNode(volGlue, 0, new
    // TGeoCombiTrans(x,y-(kConeOutRadius+klay3+klay2+(klay4/2)), z, new TGeoRotation("",0, 0, 0)));
    mechStavVol->AddNode(
      volGlue, 0,
      new TGeoCombiTrans(x, y - (kConeOutRadius + klay3 + klay2 + (klay4) / 2) + 0.00005, z,
                         new TGeoRotation("", 0, 0, 0)));
  }

  if (mBuildLevel < 1) {
    // Flex Cable or Bus
    auto* kapCable = new TGeoBBox(kStaveWidth / 2, klay5 / 2, zsta); // klay5/2
    auto* volCable = new TGeoVolume("FlexCable", kapCable, medFlexCable);
    volCable->SetLineColor(28);
    volCable->SetFillColor(28);
    //      mechStavVol->AddNode(volCable, 0, new TGeoCombiTrans(x,
    // y-(kConeOutRadius+klay3+klay2+klay4+mSensorThickness+(klay5)/2)+0.0002, z, new
    // TGeoRotation("",0,
    // 0, 0)));
    mechStavVol->AddNode(
      volCable, 0,
      new TGeoCombiTrans(
        x, y - (kConeOutRadius + klay3 + klay2 + klay4 + mSensorThickness + (klay5) / 2) + 0.01185,
        z, new TGeoRotation("", 0, 0, 0)));
  }
  // Done, return the stave structe
  return mechStavVol;
}

// model3
TGeoVolume* V1Layer::createStaveModelInnerB3(const Double_t xsta, const Double_t zsta,
                                             const TGeoManager* mgr)
{
  // Materials defined in Detector
  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");
  TGeoMedium* medWater = mgr->GetMedium("ITS_WATER$");

  TGeoMedium* medM60J3K = mgr->GetMedium("ITS_M60J3K$");
  TGeoMedium* medKapton = mgr->GetMedium("ITS_KAPTON(POLYCH2)$");
  TGeoMedium* medGlue = mgr->GetMedium("ITS_GLUE$");
  TGeoMedium* medFlexCable = mgr->GetMedium("ITS_FLEXCABLE$");
  // TGeoMedium *medK13D2U2k  = mgr->GetMedium("ITS_K13D2U2k$");
  // TGeoMedium *medFGS003    = mgr->GetMedium("ITS_FGS003$");
  // TGeoMedium *medCarbonFleece = mgr->GetMedium("ITS_CarbonFleece$");

  // Local parameters
  Double_t kConeOutRadius = 0.15 / 2;
  Double_t kStaveLength = zsta * 2;
  Double_t kStaveWidth = xsta * 2;
  Double_t w = kStaveWidth / 4; // 1/2 of W
  Double_t staveHeight = 0.3;
  Double_t h = staveHeight / 2;
  Double_t alpha = 90 - 33.; // 90-30;
  Double_t the1 = alpha * TMath::DegToRad();
  Double_t s1 = w / TMath::Sin(the1);
  Double_t l = w / TMath::Tan(the1);
  Double_t s2 = TMath::Sqrt(h * h + s1 * s1); // TMath::Sin(the2);
  Double_t the2 = TMath::ATan(h / s1);
  Double_t beta = the2 * TMath::RadToDeg();
  Double_t klay4 = 0.007; // Glue
  Double_t klay5 = 0.01;  // Flexcable
  Int_t loop = (Int_t)((kStaveLength / (2 * l)) / 2);
  Double_t hh = 0.01;
  Double_t ang1 = 0 * TMath::DegToRad();
  Double_t ang2 = 0 * TMath::DegToRad();
  Double_t ang3 = 0 * TMath::DegToRad();
  Int_t chips = 4;
  Double_t headWidth = 0.25;
  Double_t smcLength = kStaveLength / chips - 2 * headWidth; // 6.25;
  Double_t smcWidth = kStaveWidth;
  Double_t smcSide1Thick = 0.03;
  Double_t vaporThick = 0.032;
  Double_t liquidThick = 0.028;
  Double_t smcSide2Thick = 0.01;
  Double_t smcSide3Thick = 0.0055;
  Double_t smcSide4Thick = 0.0095;
  Double_t smcSide5Thick = 0.0075;
  Double_t smcSpace = 0.01;

  char volumeName[30];
  snprintf(volumeName, 30, "%s%d_StaveStruct", GeometryTGeo::getITSStavePattern(),
           mLayerNumber);

  // detailed structure ++++++++++++++
  Double_t z = 0, y = 0 - 0.007, x = 0;

  // Polimide micro channels numbers
  Double_t yMC = y - h + 0.01;
  Int_t nb = (Int_t)(kStaveWidth / 0.1) + 1;
  Double_t xstaMC = (nb * 0.1 - 0.08) / 2;

  TGeoVolume* mechStavVol = nullptr;
  if (mBuildLevel < 5) {
    // world (trapezoid)
    auto* mechStruct = new TGeoXtru(2); // z sections
    Double_t xv[5] = {
      kStaveWidth / 2 + 0.1, kStaveWidth / 2 + 0.1, 0, -kStaveWidth / 2 - 0.1,
      -kStaveWidth / 2 - 0.1};
    Double_t yv[5] = {-kConeOutRadius * 2 - 0.07, 0, staveHeight, 0, -kConeOutRadius * 2 - 0.07};
    mechStruct->DefinePolygon(5, xv, yv);
    mechStruct->DefineSection(0, -kStaveLength - 0.1, 0, 0, 1.);
    mechStruct->DefineSection(1, kStaveLength + 0.1, 0, 0, 1.);
    mechStavVol = new TGeoVolume(volumeName, mechStruct, medAir);
    mechStavVol->SetLineColor(12);
    mechStavVol->SetFillColor(12);
    mechStavVol->SetVisibility(kTRUE);

    // Silicon micro channels numbers

    auto* tM0a = new TGeoBBox(smcWidth / 2, 0.003 / 2, headWidth / 2);
    auto* volTM0a = new TGeoVolume("microChanTop1", tM0a, medKapton);
    volTM0a->SetLineColor(35);
    volTM0a->SetFillColor(35);

    for (Int_t mo = 1; mo <= chips; mo++) {
      mechStavVol->AddNode(
        volTM0a, 0,
        new TGeoCombiTrans(x, yMC + 0.03, z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth + smcLength / 2 + (headWidth / 2),
                           new TGeoRotation("", ang1, ang2, ang3)));
      mechStavVol->AddNode(
        volTM0a, 1,
        new TGeoCombiTrans(x, yMC + 0.03, z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth - smcLength / 2 - (headWidth / 2),
                           new TGeoRotation("", ang1, ang2, ang3)));
    }
    auto* tM0c = new TGeoBBox(0.3 / 2, 0.003 / 2, smcLength / 2);
    auto* volTM0c = new TGeoVolume("microChanTop2", tM0c, medKapton);
    volTM0c->SetLineColor(35);
    volTM0c->SetFillColor(35);
    for (Int_t mo = 1; mo <= chips; mo++) {
      mechStavVol->AddNode(
        volTM0c, 0, new TGeoCombiTrans(x + (smcWidth / 2) - (0.3 / 2), yMC + 0.03, z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth, new TGeoRotation("", ang1, ang2, ang3)));
      mechStavVol->AddNode(
        volTM0c, 1, new TGeoCombiTrans(x - (smcWidth / 2) + (0.3 / 2), yMC + 0.03, z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth, new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
    }
    auto* tM0c1 = new TGeoBBox(0.2225 / 2, 0.003 / 2, smcLength / 2);
    auto* volTM0c1 = new TGeoVolume("microChanBot1", tM0c1, medKapton);
    volTM0c1->SetLineColor(6);
    volTM0c1->SetFillColor(6);
    for (Int_t mo = 1; mo <= chips; mo++) {
      mechStavVol->AddNode(
        volTM0c1, 0, new TGeoCombiTrans(x + smcWidth / 2 - (smcSide1Thick) - (vaporThick) - (smcSide2Thick) - (smcSide3Thick) - (0.2225 / 2), yMC + 0.03 - hh - (0.003), z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth, new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
      mechStavVol->AddNode(
        volTM0c1, 1, new TGeoCombiTrans(x - smcWidth / 2 + (smcSide1Thick) + (liquidThick) + (smcSide2Thick) + (smcSide4Thick) + (0.2225 / 2), yMC + 0.03 - hh - (0.003), z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth, new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
    }
    auto* tM0c2 = new TGeoBBox(0.072 / 2, 0.003 / 2, smcLength / 2);
    auto* volTM0c2 = new TGeoVolume("microChanBot2", tM0c2, medKapton);
    volTM0c2->SetLineColor(35);
    volTM0c2->SetFillColor(35);
    for (Int_t mo = 1; mo <= chips; mo++) {
      mechStavVol->AddNode(
        volTM0c2, 0, new TGeoCombiTrans(x + smcWidth / 2 - (0.072 / 2), yMC + 0.03 - (0.035 + 0.0015) - (0.003) / 2, z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth, new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
    }
    auto* tM0c2r = new TGeoBBox(0.068 / 2, 0.003 / 2, smcLength / 2);
    auto* volTM0c2r = new TGeoVolume("microChanBot3", tM0c2r, medKapton);
    volTM0c2r->SetLineColor(35);
    volTM0c2r->SetFillColor(35);
    for (Int_t mo = 1; mo <= chips; mo++) {
      mechStavVol->AddNode(
        volTM0c2r, 0, new TGeoCombiTrans(x - smcWidth / 2 + (0.068 / 2), yMC + 0.03 - (0.035 + 0.0015) - (0.003) / 2, z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth, new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
    }
    auto* tM0d = new TGeoBBox(smcSide1Thick / 2, 0.035 / 2, smcLength / 2);
    auto* volTM0d = new TGeoVolume("microChanSide1", tM0d, medKapton);
    volTM0d->SetLineColor(12);
    volTM0d->SetFillColor(12);
    for (Int_t mo = 1; mo <= chips; mo++) {
      mechStavVol->AddNode(
        volTM0d, 0, new TGeoCombiTrans(x + smcWidth / 2 - (smcSide1Thick / 2), yMC + 0.03 - 0.0015 - (0.035) / 2, z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth, new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
      mechStavVol->AddNode(
        volTM0d, 1, new TGeoCombiTrans(x - smcWidth / 2 + (smcSide1Thick / 2), yMC + 0.03 - 0.0015 - (0.035) / 2, z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth, new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
    }

    auto* tM0d1 = new TGeoBBox(smcSide2Thick / 2, 0.035 / 2, smcLength / 2);
    auto* volTM0d1 = new TGeoVolume("microChanSide2", tM0d1, medKapton);
    volTM0d1->SetLineColor(12);
    volTM0d1->SetFillColor(12);
    for (Int_t mo = 1; mo <= chips; mo++) {
      mechStavVol->AddNode(
        volTM0d1, 0,
        new TGeoCombiTrans(x + smcWidth / 2 - (smcSide1Thick) - (vaporThick) - (smcSide2Thick / 2),
                           yMC + 0.03 - (0.003 + 0.035) / 2,
                           z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth,
                           new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
      mechStavVol->AddNode(
        volTM0d1, 1,
        new TGeoCombiTrans(x - smcWidth / 2 + (smcSide1Thick) + (liquidThick) + (smcSide2Thick / 2),
                           yMC + 0.03 - (0.003 + 0.035) / 2,
                           z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth,
                           new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
    }
    auto* tM0d2 = new TGeoBBox(smcSide3Thick / 2, (hh + 0.003) / 2, smcLength / 2);
    auto* volTM0d2 = new TGeoVolume("microChanSide3", tM0d2, medKapton);
    volTM0d2->SetLineColor(12);
    volTM0d2->SetFillColor(12);
    for (Int_t mo = 1; mo <= chips; mo++) {
      mechStavVol->AddNode(
        volTM0d2, 0, new TGeoCombiTrans(x + smcWidth / 2 - (smcSide1Thick) - (vaporThick) - (smcSide2Thick) - (smcSide3Thick / 2), yMC + 0.03 - (0.003 + hh + 0.003) / 2, z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth, new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
    }
    auto* tM0d2r = new TGeoBBox(smcSide4Thick / 2, (hh + 0.003) / 2, smcLength / 2);
    auto* volTM0d2r = new TGeoVolume("microChanSide4", tM0d2r, medKapton);
    volTM0d2r->SetLineColor(12);
    volTM0d2r->SetFillColor(12);
    for (Int_t mo = 1; mo <= chips; mo++) {
      mechStavVol->AddNode(
        volTM0d2r, 0,
        new TGeoCombiTrans(x - smcWidth / 2 + (smcSide1Thick) + (liquidThick) + (smcSide2Thick) +
                             (smcSide4Thick / 2),
                           yMC + 0.03 - (0.003 + hh + 0.003) / 2,
                           z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth,
                           new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
    }
    auto* tM0e = new TGeoBBox(smcSide5Thick / 2, hh / 2, smcLength / 2);
    auto* volTM0e = new TGeoVolume("microChanSide5", tM0e, medKapton);
    volTM0e->SetLineColor(12);
    volTM0e->SetFillColor(12);
    for (Int_t mo = 1; mo <= chips; mo++) {
      for (Int_t ie = 0; ie < 11; ie++) {
        mechStavVol->AddNode(
          volTM0e, 0,
          new TGeoCombiTrans(x - (ie * (smcSpace + smcSide5Thick)) + smcWidth / 2 -
                               (smcSide1Thick) - (vaporThick) - (smcSide2Thick) - (smcSide3Thick)-smcSpace - (smcSide5Thick / 2),
                             yMC + 0.03 - (0.003 + hh) / 2,
                             z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth,
                             new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
        mechStavVol->AddNode(
          volTM0e, 1,
          new TGeoCombiTrans(x + (ie * (smcSpace + smcSide5Thick)) - smcWidth / 2 +
                               (smcSide1Thick) + (liquidThick) + (smcSide2Thick) + (smcSide4Thick) +
                               smcSpace + (smcSide5Thick / 2),
                             yMC + 0.03 - (0.003 + hh) / 2,
                             z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth,
                             new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
      }
    }

    auto* tM0f = new TGeoBBox(0.02 / 2, hh / 2, smcLength / 2);
    auto* volTM0f = new TGeoVolume("microChanTop3", tM0f, medKapton);
    // Double_t smcChannels=12;
    Double_t smcCloseWallvapor = smcWidth / 2 - smcSide1Thick - vaporThick - smcSide2Thick -
                                 smcSide3Thick - 12 * smcSpace - 11 * smcSide5Thick;
    Double_t smcCloseWallliquid = smcWidth / 2 - smcSide1Thick - liquidThick - smcSide2Thick -
                                  smcSide4Thick - 12 * smcSpace - 11 * smcSide5Thick;
    volTM0f->SetLineColor(12);
    volTM0f->SetFillColor(12);
    for (Int_t mo = 1; mo <= chips; mo++) {
      mechStavVol->AddNode(
        volTM0f, 0,
        new TGeoCombiTrans(x + smcCloseWallvapor - (0.02) / 2, yMC + 0.03 - (0.003 + hh) / 2,
                           z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth,
                           new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
      mechStavVol->AddNode(
        volTM0f, 1,
        new TGeoCombiTrans(x - smcCloseWallliquid + (0.02) / 2, yMC + 0.03 - (0.003 + hh) / 2,
                           z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth,
                           new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
    }
    // Head(back) microchannel

    auto* tM0hb = new TGeoBBox(smcWidth / 2, 0.025 / 2, headWidth / 2);
    auto* volTM0hb = new TGeoVolume("microChanHeadBackBottom1", tM0hb, medKapton);
    volTM0hb->SetLineColor(4);
    volTM0hb->SetFillColor(4);
    for (Int_t mo = 1; mo <= chips; mo++) {
      mechStavVol->AddNode(
        volTM0hb, 0, new TGeoCombiTrans(x, yMC + 0.03 - 0.0145 - (0.025 / 2), z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth + smcLength / 2 + (headWidth / 2), new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
      mechStavVol->AddNode(
        volTM0hb, 1, new TGeoCombiTrans(x, yMC + 0.03 - 0.0145 - (0.025) / 2, z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth - smcLength / 2 - (headWidth / 2), new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
    }
    auto* tM0h1 = new TGeoBBox(smcWidth / 2, 0.013 / 2, 0.05 / 2);
    auto* volTM0h1 = new TGeoVolume("microChanHeadBackBottom2", tM0h1, medKapton);
    volTM0h1->SetLineColor(5);
    volTM0h1->SetFillColor(5);
    for (Int_t mo = 1; mo <= chips; mo++) {
      mechStavVol->AddNode(
        volTM0h1, 0, new TGeoCombiTrans(x, yMC + 0.03 - 0.0015 - (0.013 / 2), z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth - smcLength / 2 - headWidth + (0.05 / 2), new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
    }
    auto* tM0h2 = new TGeoBBox(smcWidth / 2, 0.003 / 2, 0.18 / 2);
    auto* volTM0h2 = new TGeoVolume("microChanHeadBackBottom7", tM0h2, medKapton);
    volTM0h2->SetLineColor(6);
    volTM0h2->SetFillColor(6);
    for (Int_t mo = 1; mo <= chips; mo++) {
      mechStavVol->AddNode(
        volTM0h2, 0, new TGeoCombiTrans(x, yMC + 0.03 - 0.0015 - 0.01 - (0.003 / 2), z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth - smcLength / 2 - 0.02 - (0.18 / 2), new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
    }
    auto* tM0h3 = new TGeoBBox(smcWidth / 2, 0.013 / 2, 0.02 / 2);
    auto* volTM0h3 = new TGeoVolume("microChanHeadBackBottom3", tM0h3, medKapton);
    volTM0h3->SetLineColor(5);
    volTM0h3->SetFillColor(5);
    for (Int_t mo = 1; mo <= chips; mo++) {
      mechStavVol->AddNode(
        volTM0h3, 0, new TGeoCombiTrans(x, yMC + 0.03 - 0.0015 - (0.013 / 2), z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth - smcLength / 2 - (0.02 / 2), new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
    }
    auto* tM0b1 = new TGeoBBox(smcWidth / 2, 0.013 / 2, 0.03 / 2);
    auto* volTM0b1 = new TGeoVolume("microChanHeadBackBottom4", tM0b1, medKapton);
    volTM0b1->SetLineColor(5);
    volTM0b1->SetFillColor(5);
    for (Int_t mo = 1; mo <= chips; mo++) {
      mechStavVol->AddNode(
        volTM0b1, 0, new TGeoCombiTrans(x, yMC + 0.03 - 0.0015 - (0.013 / 2), z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth + smcLength / 2 + headWidth - (0.03 / 2), new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
    }
    auto* tM0b2 = new TGeoBBox(smcWidth / 2, 0.003 / 2, 0.2 / 2);
    auto* volTM0b2 = new TGeoVolume("microChanHeadBackBottom5", tM0b2, medKapton);
    volTM0b2->SetLineColor(6);
    volTM0b2->SetFillColor(6);
    for (Int_t mo = 1; mo <= chips; mo++) {
      mechStavVol->AddNode(
        volTM0b2, 0, new TGeoCombiTrans(x, yMC + 0.03 - 0.0015 - 0.01 - (0.003 / 2), z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth + smcLength / 2 + 0.02 + (0.2 / 2), new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
    }
    auto* tM0b3 = new TGeoBBox(smcWidth / 2, 0.013 / 2, 0.02 / 2);
    auto* volTM0b3 = new TGeoVolume("microChanHeadBackBottom6", tM0b3, medKapton);
    volTM0b3->SetLineColor(5);
    volTM0b3->SetFillColor(5);
    for (Int_t mo = 1; mo <= chips; mo++) {
      mechStavVol->AddNode(
        volTM0b3, 0, new TGeoCombiTrans(x, yMC + 0.03 - 0.0015 - (0.013 / 2), z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth + smcLength / 2 + (0.02 / 2), new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
    }

    auto* tM0b = new TGeoBBox(0.02 / 2, 0.02 / 2, zsta);
    auto* volTM0b = new TGeoVolume("microChanWalls", tM0b, medKapton);
    volTM0b->SetLineColor(35);
    volTM0b->SetFillColor(35);
    for (Int_t ib = 0; ib < nb; ib++) {
      // mechStavVol->AddNode(volTM0b, ib, new TGeoCombiTrans(x+ib*0.1-xstaMC+0.01,yMC, z, new
      // TGeoRotation("",0, 0, 0)));
    }
  }

  if (mBuildLevel < 4) {
    // cooling  inlet outlet
    auto* tM0dv = new TGeoBBox(vaporThick / 2, 0.035 / 2, smcLength / 2);
    auto* volTM0dv = new TGeoVolume("microChanVapor", tM0dv, medWater);
    volTM0dv->SetLineColor(2);
    volTM0dv->SetFillColor(2);
    for (Int_t mo = 1; mo <= chips; mo++) {
      mechStavVol->AddNode(
        volTM0dv, 0, new TGeoCombiTrans(x + smcWidth / 2 - (smcSide1Thick) - (vaporThick / 2), yMC + 0.03 - 0.0015 - (0.035) / 2, z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth, new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
    }
    auto* tM0dl = new TGeoBBox(liquidThick / 2, 0.035 / 2, smcLength / 2);
    auto* volTM0dl = new TGeoVolume("microChanLiquid", tM0dl, medWater);
    volTM0dl->SetLineColor(3);
    volTM0dl->SetFillColor(3);
    for (Int_t mo = 1; mo <= chips; mo++) {
      mechStavVol->AddNode(
        volTM0dl, 0, new TGeoCombiTrans(x - smcWidth / 2 + (smcSide1Thick) + (liquidThick / 2), yMC + 0.03 - 0.0015 - (0.035) / 2, z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth, new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
    }
    // small cooling fluid now using water wait for freeon value
    auto* tM0dlq = new TGeoBBox(smcSpace / 2, hh / 2, smcLength / 2);
    auto* volTM0dlq = new TGeoVolume("smallLiquid", tM0dlq, medWater);
    volTM0dlq->SetLineColor(3);
    volTM0dlq->SetFillColor(3);
    auto* tM0dvp = new TGeoBBox(smcSpace / 2, hh / 2, smcLength / 2);
    auto* volTM0dvp = new TGeoVolume("microChanVapor", tM0dvp, medWater);
    volTM0dvp->SetLineColor(2);
    volTM0dvp->SetFillColor(2);
    for (Int_t mo = 1; mo <= chips; mo++) {
      for (Int_t is = 0; is < 12; is++) {
        mechStavVol->AddNode(
          volTM0dlq, 0, new TGeoCombiTrans(x + (is * (smcSpace + smcSide5Thick)) - smcWidth / 2 + (smcSide1Thick) + (vaporThick) + (smcSide2Thick) + (smcSide3Thick) + smcSpace / 2, yMC + 0.03 - (0.003 + hh) / 2, z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth, new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
        mechStavVol->AddNode(
          volTM0dvp, 1, new TGeoCombiTrans(x - (is * (smcSpace + smcSide5Thick)) + smcWidth / 2 - (smcSide1Thick) - (vaporThick) - (smcSide2Thick) - (smcSide3Thick)-smcSpace / 2, yMC + 0.03 - (0.003 + hh) / 2, z + (mo - 3) * kStaveLength / 4 + smcLength / 2 + headWidth, new TGeoRotation("", ang1, ang2, ang3))); //("",0, 0, 0)));
      }
    }
  }

  if (mBuildLevel < 3) {
    // Bottom filament CFRP black-12 Carbon structure TGeoBBox (thickness,width,length)
    Double_t filWidth = 0.04;
    Double_t filHeight = 0.02;
    auto* t1 = new TGeoBBox(filHeight / 2, filWidth / 2, s1);
    auto* volT1 = new TGeoVolume("bottomFilament", t1, medM60J3K);
    volT1->SetLineColor(12);
    volT1->SetFillColor(12);
    for (int i = 0; i < loop; i++) { // i<30;i++){
      mechStavVol->AddNode(volT1, 4 * i + 0,
                           new TGeoCombiTrans(x + w, y - h + 0.04 + filHeight / 2,
                                              z - kStaveLength / 2 + (4 * l * i) + s1 / 2,
                                              new TGeoRotation("volT1", -90, alpha, 0)));
      mechStavVol->AddNode(volT1, 4 * i + 1,
                           new TGeoCombiTrans(x - w, y - h + 0.04 + filHeight / 2,
                                              z - kStaveLength / 2 + (4 * l * i) + s1 / 2,
                                              new TGeoRotation("volT1", 90, alpha, 0)));
      mechStavVol->AddNode(volT1, 4 * i + 2,
                           new TGeoCombiTrans(x + w, y - h + 0.04 + filHeight / 2,
                                              z - kStaveLength / 2 + 2 * l + (i * 4 * l) + s1 / 2,
                                              new TGeoRotation("volT1", -90, -alpha, 0)));
      mechStavVol->AddNode(volT1, 4 * i + 3,
                           new TGeoCombiTrans(x - w, y - h + 0.04 + filHeight / 2,
                                              z - kStaveLength / 2 + 2 * l + (i * 4 * l) + s1 / 2,
                                              new TGeoRotation("volT1", -90, +alpha, 0)));
    }

    // Top filament CERP black-12 Carbon structure TGeoBBox (length,thickness,width)
    auto* t2 = new TGeoBBox(s2, filHeight / 2, filWidth / 2);
    auto* volT2 = new TGeoVolume("topFilament", t2, medM60J3K);
    volT2->SetLineColor(12);
    volT2->SetFillColor(12);
    for (int i = 0; i < loop; i++) { // i<30;i++){
      mechStavVol->AddNode(
        volT2, 4 * i + 0, new TGeoCombiTrans(x + w, y + 0.04 + filHeight / 2, z - kStaveLength / 2 + (i * 4 * l) + s1 / 2, new TGeoRotation("volT2", 90, 90 - alpha, 90 - beta)));
      mechStavVol->AddNode(
        volT2, 4 * i + 1,
        new TGeoCombiTrans(x - w, y + 0.04 + filHeight / 2,
                           z - kStaveLength / 2 + (i * 4 * l) + s1 / 2,
                           new TGeoRotation("volT2", 90, -90 + alpha, -90 + beta)));
      mechStavVol->AddNode(
        volT2, 4 * i + 2,
        new TGeoCombiTrans(x + w, y + 0.04 + filHeight / 2,
                           z - kStaveLength / 2 + 2 * l + (i * 4 * l) + s1 / 2,
                           new TGeoRotation("volT2", 90, -90 + alpha, 90 - beta)));
      mechStavVol->AddNode(
        volT2, 4 * i + 3,
        new TGeoCombiTrans(x - w, y + 0.04 + filHeight / 2,
                           z - kStaveLength / 2 + 2 * l + (i * 4 * l) + s1 / 2,
                           new TGeoRotation("volT2", 90, 90 - alpha, -90 + beta)));
    }
  }

  if (mBuildLevel < 2) {
    // Glue Filament and Silicon MicroChannel
    auto* tM0 = new TGeoBBox(xstaMC / 5, klay4 / 2, zsta);
    auto* volTM0 = new TGeoVolume("glueFM", tM0, medGlue);
    volTM0->SetLineColor(5);
    volTM0->SetFillColor(5);
    mechStavVol->AddNode(volTM0, 0, new TGeoCombiTrans(x - xsta / 2 - 0.25, 0.03 + yMC, z, new TGeoRotation("", 0, 0, 0)));
    mechStavVol->AddNode(volTM0, 1, new TGeoCombiTrans(x + xsta / 2 + 0.25, 0.03 + yMC, z, new TGeoRotation("", 0, 0, 0)));

    // Glue microchannel and sensor
    auto* glueM = new TGeoBBox(xstaMC / 5, klay4 / 2, zsta);
    auto* volGlueM = new TGeoVolume("glueMS", glueM, medGlue);
    volGlueM->SetLineColor(5);
    volGlueM->SetFillColor(5);
    mechStavVol->AddNode(volGlueM, 0, new TGeoCombiTrans(x - xsta / 2 - 0.25, yMC - 0.01, z, new TGeoRotation("", 0, 0, 0)));
    mechStavVol->AddNode(volGlueM, 1, new TGeoCombiTrans(x + xsta / 2 + 0.25, yMC - 0.01, z, new TGeoRotation("", 0, 0, 0)));

    // Glue sensor and kapton
    auto* glue = new TGeoBBox(xsta, klay4 / 2, zsta);
    auto* volGlue = new TGeoVolume("glueSensorBus", glue, medGlue);
    volGlue->SetLineColor(5);
    volGlue->SetFillColor(5);
    mechStavVol->AddNode(volGlue, 1, new TGeoCombiTrans(x, y - 0.154 - mSensorThickness - klay4 / 2, z, new TGeoRotation("", 0, 0, 0)));
  }

  if (mBuildLevel < 1) {
    auto* kapCable = new TGeoBBox(xsta, klay5 / 2, zsta);
    auto* volCable = new TGeoVolume("Flexcable", kapCable, medFlexCable);
    volCable->SetLineColor(28);
    volCable->SetFillColor(28);
    mechStavVol->AddNode(volCable, 0,
                         new TGeoCombiTrans(x, y - 0.154 - mSensorThickness - klay4 - klay5 / 2, z,
                                            new TGeoRotation("", 0, 0, 0)));
  }
  // Done, return the stave structure
  return mechStavVol;
}

TGeoVolume* V1Layer::createStaveOuterB(const TGeoManager* mgr)
{
  TGeoVolume* mechStavVol = nullptr;

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
      LOG(FATAL) << "Unknown stave model " << mStaveModel;
      break;
  }
  return mechStavVol;
}

TGeoVolume* V1Layer::createStaveModelOuterBDummy(const TGeoManager*) const
{
  // Done, return the stave structure
  return nullptr;
}

TGeoVolume* V1Layer::createStaveModelOuterB0(const TGeoManager* mgr)
{
  Double_t xmod, ymod, zmod;
  Double_t xlen, ylen, zlen;
  Double_t ypos, zpos;
  char volumeName[30];

  // First create all needed shapes
  // The chip
  xlen = sOBHalfStaveWidth;
  ylen = 0.5 * mStaveThickness; // TO BE CHECKED
  zlen = sOBModuleZLength / 2;

  TGeoVolume* chipVol = createChipInnerB(xlen, ylen, zlen);

  xmod = ((TGeoBBox*)chipVol->GetShape())->GetDX();
  ymod = ((TGeoBBox*)chipVol->GetShape())->GetDY();
  zmod = ((TGeoBBox*)chipVol->GetShape())->GetDZ();

  auto* module = new TGeoBBox(xmod, ymod, zmod);

  zlen = sOBModuleZLength * mNumberOfModules;
  auto* hstave = new TGeoBBox(xlen, ylen, zlen / 2);

  // We have all shapes: now create the real volumes
  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSModulePattern(), mLayerNumber);
  auto* modVol = new TGeoVolume(volumeName, module, medAir);
  modVol->SetVisibility(kTRUE);

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSHalfStavePattern(), mLayerNumber);
  auto* hstaveVol = new TGeoVolume(volumeName, hstave, medAir);

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

TGeoVolume* V1Layer::createStaveModelOuterB1(const TGeoManager* mgr)
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
  TGeoVolume* moduleVol = createModuleOuterB();
  moduleVol->SetVisibility(kTRUE);
  ymod = ((TGeoBBox*)(moduleVol->GetShape()))->GetDY();
  zmod = ((TGeoBBox*)(moduleVol->GetShape()))->GetDZ();

  auto* busAl = new TGeoBBox("BusAl", xHalmSt, sOBBusCableAlThick / 2, zlen);
  auto* busKap = new TGeoBBox("BusKap", xHalmSt, sOBBusCableKapThick / 2, zlen);

  auto* coldPlate =
    new TGeoBBox("ColdPlate", sOBHalfStaveWidth / 2, sOBColdPlateThick / 2, zlen);

  auto* coolTube = new TGeoTube("CoolingTube", rCoolMin, rCoolMax, zlen);
  auto* coolWater = new TGeoTube("CoolingWater", 0., rCoolMin, zlen);

  xlen = xHalmSt - sOBCoolTubeXDist / 2 - coolTube->GetRmax();
  auto* graphlat = new TGeoBBox("GraphLateral", xlen / 2, kLay2 / 2, zlen);

  xlen = sOBCoolTubeXDist / 2 - coolTube->GetRmax();
  auto* graphmid = new TGeoBBox("GraphMiddle", xlen, kLay2 / 2, zlen);

  ylen = coolTube->GetRmax() - kLay2;
  auto* graphvert = new TGeoBBox("GraphVertical", kLay2 / 2, ylen / 2, zlen);

  auto* graphtub =
    new TGeoTubeSeg("GraphTube", rCoolMax, rCoolMax + kLay2, zlen, 180., 360.);

  xlen = xHalmSt - sOBCoolTubeXDist / 2 - coolTube->GetRmax() - kLay2;
  auto* fleeclat = new TGeoBBox("FleecLateral", xlen / 2, kLay1 / 2, zlen);

  xlen = sOBCoolTubeXDist / 2 - coolTube->GetRmax() - kLay2;
  auto* fleecmid = new TGeoBBox("FleecMiddle", xlen, kLay1 / 2, zlen);

  ylen = coolTube->GetRmax() - kLay2 - kLay1;
  auto* fleecvert = new TGeoBBox("FleecVertical", kLay1 / 2, ylen / 2, zlen);

  auto* fleectub =
    new TGeoTubeSeg("FleecTube", rCoolMax + kLay2, rCoolMax + kLay1 + kLay2, zlen, 180., 360.);

  auto* flex1_5cm = new TGeoBBox("Flex1MV_5cm", xHalmSt, yFlex1 / 2, flexOverlap / 2);
  auto* flex2_5cm = new TGeoBBox("Flex2MV_5cm", xHalmSt, yFlex2 / 2, flexOverlap / 2);

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
  auto* halmStave = new TGeoXtru(2);
  halmStave->DefinePolygon(12, xtru, ytru);
  halmStave->DefineSection(0, -mZLength / 2);
  halmStave->DefineSection(1, mZLength / 2);

  // We have all shapes: now create the real volumes

  TGeoMedium* medAluminum = mgr->GetMedium("ITS_ALUMINUM$");
  TGeoMedium* medCarbon = mgr->GetMedium("ITS_CARBON$");
  TGeoMedium* medKapton = mgr->GetMedium("ITS_KAPTON(POLYCH2)$");
  TGeoMedium* medWater = mgr->GetMedium("ITS_WATER$");
  TGeoMedium* medCarbonFleece = mgr->GetMedium("ITS_CarbonFleece$");
  TGeoMedium* medFGS003 = mgr->GetMedium("ITS_FGS003$"); // amec thermasol
  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");

  auto* busAlVol = new TGeoVolume("BusAlVol", busAl, medAluminum);
  busAlVol->SetLineColor(kCyan);
  busAlVol->SetFillColor(busAlVol->GetLineColor());
  busAlVol->SetFillStyle(4000); // 0% transparent

  auto* busKapVol = new TGeoVolume("BusKapVol", busKap, medKapton);
  busKapVol->SetLineColor(kBlue);
  busKapVol->SetFillColor(busKapVol->GetLineColor());
  busKapVol->SetFillStyle(4000); // 0% transparent

  auto* coldPlateVol = new TGeoVolume("ColdPlateVol", coldPlate, medCarbon);
  coldPlateVol->SetLineColor(kYellow - 3);
  coldPlateVol->SetFillColor(coldPlateVol->GetLineColor());
  coldPlateVol->SetFillStyle(4000); // 0% transparent

  auto* coolTubeVol = new TGeoVolume("CoolingTubeVol", coolTube, medKapton);
  coolTubeVol->SetLineColor(kGray);
  coolTubeVol->SetFillColor(coolTubeVol->GetLineColor());
  coolTubeVol->SetFillStyle(4000); // 0% transparent

  auto* coolWaterVol = new TGeoVolume("CoolingWaterVol", coolWater, medWater);
  coolWaterVol->SetLineColor(kBlue);
  coolWaterVol->SetFillColor(coolWaterVol->GetLineColor());
  coolWaterVol->SetFillStyle(4000); // 0% transparent

  auto* graphlatVol = new TGeoVolume("GraphiteFoilLateral", graphlat, medFGS003);
  graphlatVol->SetLineColor(kGreen);
  graphlatVol->SetFillColor(graphlatVol->GetLineColor());
  graphlatVol->SetFillStyle(4000); // 0% transparent

  auto* graphmidVol = new TGeoVolume("GraphiteFoilMiddle", graphmid, medFGS003);
  graphmidVol->SetLineColor(kGreen);
  graphmidVol->SetFillColor(graphmidVol->GetLineColor());
  graphmidVol->SetFillStyle(4000); // 0% transparent

  auto* graphvertVol = new TGeoVolume("GraphiteFoilVertical", graphvert, medFGS003);
  graphvertVol->SetLineColor(kGreen);
  graphvertVol->SetFillColor(graphvertVol->GetLineColor());
  graphvertVol->SetFillStyle(4000); // 0% transparent

  auto* graphtubVol = new TGeoVolume("GraphiteFoilPipeCover", graphtub, medFGS003);
  graphtubVol->SetLineColor(kGreen);
  graphtubVol->SetFillColor(graphtubVol->GetLineColor());
  graphtubVol->SetFillStyle(4000); // 0% transparent

  auto* fleeclatVol = new TGeoVolume("CarbonFleeceLateral", fleeclat, medCarbonFleece);
  fleeclatVol->SetLineColor(kViolet);
  fleeclatVol->SetFillColor(fleeclatVol->GetLineColor());
  fleeclatVol->SetFillStyle(4000); // 0% transparent

  auto* fleecmidVol = new TGeoVolume("CarbonFleeceMiddle", fleecmid, medCarbonFleece);
  fleecmidVol->SetLineColor(kViolet);
  fleecmidVol->SetFillColor(fleecmidVol->GetLineColor());
  fleecmidVol->SetFillStyle(4000); // 0% transparent

  auto* fleecvertVol = new TGeoVolume("CarbonFleeceVertical", fleecvert, medCarbonFleece);
  fleecvertVol->SetLineColor(kViolet);
  fleecvertVol->SetFillColor(fleecvertVol->GetLineColor());
  fleecvertVol->SetFillStyle(4000); // 0% transparent

  auto* fleectubVol = new TGeoVolume("CarbonFleecePipeCover", fleectub, medCarbonFleece);
  fleectubVol->SetLineColor(kViolet);
  fleectubVol->SetFillColor(fleectubVol->GetLineColor());
  fleectubVol->SetFillStyle(4000); // 0% transparent

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSHalfStavePattern(), mLayerNumber);
  auto* halmStaveVol = new TGeoVolume(volumeName, halmStave, medAir);
  //   halmStaveVol->SetLineColor(12);
  //   halmStaveVol->SetFillColor(12);
  //   halmStaveVol->SetVisibility(kTRUE);

  auto* flex1_5cmVol = new TGeoVolume("Flex1Vol5cm", flex1_5cm, medAluminum);
  auto* flex2_5cmVol = new TGeoVolume("Flex2Vol5cm", flex2_5cm, medKapton);

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

TGeoVolume* V1Layer::createSpaceFrameOuterB(const TGeoManager* mgr)
{
  TGeoVolume* mechStavVol = nullptr;

  switch (mStaveModel) {
    case Detector::kOBModelDummy:
    case Detector::kOBModel0:
      mechStavVol = createSpaceFrameOuterBDummy(mgr);
      break;
    case Detector::kOBModel1:
      mechStavVol = createSpaceFrameOuterB1(mgr);
      break;
    default:
      LOG(FATAL) << "Unknown stave model " << mStaveModel;
      break;
  }

  return mechStavVol;
}

TGeoVolume* V1Layer::createSpaceFrameOuterBDummy(const TGeoManager*) const
{
  // Done, return the stave structur
  return nullptr;
}

TGeoVolume* V1Layer::createSpaceFrameOuterB1(const TGeoManager* mgr)
{
  // Materials defined in Detector
  TGeoMedium* medCarbon = mgr->GetMedium("ITS_CARBON$");
  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");

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
    sframeHeight -= ((TGeoBBox*)gGeoManager->GetVolume(volumeName)->GetShape())->GetDY() * 2;
    zlen = ((TGeoBBox*)gGeoManager->GetVolume(volumeName)->GetShape())->GetDZ() * 2;
  }
  seglen = zlen / mNumberOfModules;

  // First create all needed shapes and volumes
  auto* spaceFrame = new TGeoBBox(sframeWidth / 2, sframeHeight / 2, zlen / 2);
  auto* segment = new TGeoBBox(sframeWidth / 2, sframeHeight / 2, seglen / 2);

  auto* spaceFrameVol = new TGeoVolume("CarbonFrameVolume", spaceFrame, medAir);
  spaceFrameVol->SetVisibility(kFALSE);

  auto* segmentVol = new TGeoVolume("segmentVol", segment, medAir);

  // SpaceFrame

  //--- the top V of the Carbon Fiber Stave (segment)
  TGeoArb8* cmStavTop1 = createStaveSide("CFstavTopCornerVol1shape", seglen / 2., halmTheta, -1,
                                         staveLa, staveHa, stavel);
  auto* cmStavTopVol1 = new TGeoVolume("CFstavTopCornerVol1", cmStavTop1, medCarbon);
  cmStavTopVol1->SetLineColor(35);

  TGeoArb8* cmStavTop2 = createStaveSide("CFstavTopCornerVol2shape", seglen / 2., halmTheta, 1,
                                         staveLa, staveHa, stavel);
  auto* cmStavTopVol2 = new TGeoVolume("CFstavTopCornerVol2", cmStavTop2, medCarbon);
  cmStavTopVol2->SetLineColor(35);

  auto* trTop1 = new TGeoTranslation(0, sframeHeight / 2, 0);

  //--- the 2 side V
  TGeoArb8* cmStavSide1 =
    createStaveSide("CFstavSideCornerVol1shape", seglen / 2., beta, -1, staveLb, staveHb, stavel);
  auto* cmStavSideVol1 = new TGeoVolume("CFstavSideCornerVol1", cmStavSide1, medCarbon);
  cmStavSideVol1->SetLineColor(35);

  TGeoArb8* cmStavSide2 =
    createStaveSide("CFstavSideCornerVol2shape", seglen / 2., beta, 1, staveLb, staveHb, stavel);
  auto* cmStavSideVol2 = new TGeoVolume("CFstavSideCornerVol2", cmStavSide2, medCarbon);
  cmStavSideVol2->SetLineColor(35);

  xpos = -sframeWidth / 2;
  ypos = -sframeHeight / 2 + staveBeamRadius + staveHb * TMath::Sin(beta);
  auto* ctSideR = new TGeoCombiTrans(
    xpos, ypos, 0, new TGeoRotation("", 180 - 2 * beta * TMath::RadToDeg(), 0, 0));
  auto* ctSideL = new TGeoCombiTrans(
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
  auto* sideBeam = new TGeoTubeSeg(0, staveBeamRadius, beamLength / 2, 0, 180);
  auto* sideBeamVol = new TGeoVolume("CFstavSideBeamVol", sideBeam, medCarbon);
  sideBeamVol->SetLineColor(35);

  auto* beamRot1 = new TGeoRotation("", /*90-2*beta*/ halmTheta * TMath::RadToDeg(),
                                    -beamPhiPrime * TMath::RadToDeg(), -90);
  auto* beamRot2 =
    new TGeoRotation("", 90 - 2. * beta * TMath::RadToDeg(), beamPhiPrime * TMath::RadToDeg(), -90);
  auto* beamRot3 =
    new TGeoRotation("", 90 + 2. * beta * TMath::RadToDeg(), beamPhiPrime * TMath::RadToDeg(), -90);
  auto* beamRot4 = new TGeoRotation("", 90 + 2. * beta * TMath::RadToDeg(),
                                    -beamPhiPrime * TMath::RadToDeg(), -90);

  TGeoCombiTrans* beamTransf[8];
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
  auto* bottomBeam1 =
    new TGeoTubeSeg(0, staveBeamRadius, sframeWidth / 2. - staveLb / 3, 0, 180);
  auto* bottomBeam1Vol = new TGeoVolume("CFstavBottomBeam1Vol", bottomBeam1, medCarbon);
  bottomBeam1Vol->SetLineColor(35);

  auto* bottomBeam2 =
    new TGeoTubeSeg(0, staveBeamRadius, sframeWidth / 2. - staveLb / 3, 0, 90);
  auto* bottomBeam2Vol = new TGeoVolume("CFstavBottomBeam2Vol", bottomBeam2, medCarbon);
  bottomBeam2Vol->SetLineColor(35);

  auto* bottomBeam3 = new TGeoTubeSeg(
    0, staveBeamRadius, 0.5 * sframeWidth / sinD(bottomBeamAngle) - staveLb / 3, 0, 180);
  auto* bottomBeam3Vol = new TGeoVolume("CFstavBottomBeam3Vol", bottomBeam3, medCarbon);
  bottomBeam3Vol->SetLineColor(35);

  auto* bottomBeamRot1 = new TGeoRotation("", 90, 90, 90);
  auto* bottomBeamRot2 = new TGeoRotation("", -90, 90, -90);

  auto* bottomBeamTransf1 =
    new TGeoCombiTrans("", 0, -(sframeHeight / 2 - staveBeamRadius), 0, bottomBeamRot1);
  auto* bottomBeamTransf2 =
    new TGeoCombiTrans(0, -(sframeHeight / 2 - staveBeamRadius), -seglen / 2, bottomBeamRot1);
  auto* bottomBeamTransf3 =
    new TGeoCombiTrans(0, -(sframeHeight / 2 - staveBeamRadius), seglen / 2, bottomBeamRot2);
  // be careful for beams #3: when "reading" from -z to +z and
  // from the bottom of the stave, it should draw a Lambda, and not a V
  auto* bottomBeamRot4 = new TGeoRotation("", -90, bottomBeamAngle, -90);
  auto* bottomBeamRot5 = new TGeoRotation("", -90, -bottomBeamAngle, -90);

  auto* bottomBeamTransf4 =
    new TGeoCombiTrans(0, -(sframeHeight / 2 - staveBeamRadius), -seglen / 4, bottomBeamRot4);
  auto* bottomBeamTransf5 =
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

TGeoVolume* V1Layer::createChipInnerB(const Double_t xchip, const Double_t ychip,
                                      const Double_t zchip, const TGeoManager* mgr)
{
  char volumeName[30];
  Double_t xlen, ylen, zlen;
  Double_t xpos, ypos, zpos;

  // First create all needed shapes

  // The chip
  auto* chip = new TGeoBBox(xchip, ychip, zchip);

  // The sensor
  xlen = chip->GetDX();
  ylen = 0.5 * mSensorThickness;
  zlen = chip->GetDZ();
  auto* sensor = new TGeoBBox(xlen, ylen, zlen);

  // We have all shapes: now create the real volumes
  TGeoMedium* medSi = mgr->GetMedium("ITS_SI$");

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSChipPattern(), mLayerNumber);
  auto* chipVol = new TGeoVolume(volumeName, chip, medSi);
  chipVol->SetVisibility(kTRUE);
  chipVol->SetLineColor(1);

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSSensorPattern(), mLayerNumber);
  auto* sensVol = new TGeoVolume(volumeName, sensor, medSi);
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

TGeoVolume* V1Layer::createModuleOuterB(const TGeoManager* mgr)
{
  char volumeName[30];

  Double_t xGap = sOBChipXGap;
  Double_t zGap = sOBChipZGap;

  Double_t xchip, ychip, zchip;
  Double_t xlen, ylen, zlen;
  Double_t xpos, ypos, zpos;

  // First create all needed shapes

  // The chip (the same as for IB)
  xlen = (sOBHalfStaveWidth / 2 - xGap / 2) / sOBNChipRows;
  ylen = 0.5 * mStaveThickness; // TO BE CHECKED
  zlen = (sOBModuleZLength - (sOBChipsPerRow - 1) * zGap) / (2 * sOBChipsPerRow);

  TGeoVolume* chipVol = createChipInnerB(xlen, ylen, zlen);

  xchip = ((TGeoBBox*)chipVol->GetShape())->GetDX();
  ychip = ((TGeoBBox*)chipVol->GetShape())->GetDY();
  zchip = ((TGeoBBox*)chipVol->GetShape())->GetDZ();

  // The module carbon plate
  xlen = sOBHalfStaveWidth / 2;
  ylen = sOBCarbonPlateThick / 2;
  zlen = sOBModuleZLength / 2;
  auto* modPlate = new TGeoBBox("CarbonPlate", xlen, ylen, zlen);

  // The glue
  ylen = sOBGlueThick / 2;
  auto* glue = new TGeoBBox("Glue", xlen, ylen, zlen);

  // The flex cables
  ylen = sOBFlexCableAlThick / 2;
  auto* flexAl = new TGeoBBox("FlexAl", xlen, ylen, zlen);

  ylen = sOBFlexCableKapThick / 2;
  auto* flexKap = new TGeoBBox("FlexKap", xlen, ylen, zlen);

  // The module
  xlen = sOBHalfStaveWidth / 2;
  ylen = ychip + modPlate->GetDY() + glue->GetDY() + flexAl->GetDY() + flexKap->GetDY();
  zlen = sOBModuleZLength / 2;
  auto* module = new TGeoBBox("OBModule", xlen, ylen, zlen);

  // We have all shapes: now create the real volumes

  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");
  TGeoMedium* medCarbon = mgr->GetMedium("ITS_CARBON$");
  TGeoMedium* medGlue = mgr->GetMedium("ITS_GLUE$");
  TGeoMedium* medAluminum = mgr->GetMedium("ITS_ALUMINUM$");
  TGeoMedium* medKapton = mgr->GetMedium("ITS_KAPTON(POLYCH2)$");

  auto* modPlateVol = new TGeoVolume("CarbonPlateVol", modPlate, medCarbon);
  modPlateVol->SetLineColor(kMagenta - 8);
  modPlateVol->SetFillColor(modPlateVol->GetLineColor());
  modPlateVol->SetFillStyle(4000); // 0% transparent

  auto* glueVol = new TGeoVolume("GlueVol", glue, medGlue);
  glueVol->SetLineColor(kBlack);
  glueVol->SetFillColor(glueVol->GetLineColor());
  glueVol->SetFillStyle(4000); // 0% transparent

  auto* flexAlVol = new TGeoVolume("FlexAlVol", flexAl, medAluminum);
  flexAlVol->SetLineColor(kRed);
  flexAlVol->SetFillColor(flexAlVol->GetLineColor());
  flexAlVol->SetFillStyle(4000); // 0% transparent

  auto* flexKapVol = new TGeoVolume("FlexKapVol", flexKap, medKapton);
  flexKapVol->SetLineColor(kGreen);
  flexKapVol->SetFillColor(flexKapVol->GetLineColor());
  flexKapVol->SetFillStyle(4000); // 0% transparent

  snprintf(volumeName, 30, "%s%d", GeometryTGeo::getITSModulePattern(), mLayerNumber);
  auto* modVol = new TGeoVolume(volumeName, module, medAir);
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

Double_t V1Layer::radiusOmTurboContainer()
{
  Double_t rr, delta, z, lstav, rstav;

  if (mStaveThickness > 89.) { // Very big angle: avoid overflows since surely
    return -1;                 // the radius from lower vertex is the right value
  }

  rstav = mLayerRadius + 0.5 * mStaveThickness;
  delta = (0.5 * mStaveThickness) / cosD(mStaveTilt);
  z = (0.5 * mStaveThickness) * tanD(mStaveTilt);

  rr = rstav - delta;
  lstav = (0.5 * mStaveWidth) - z;

  if ((rr * sinD(mStaveTilt) < lstav)) {
    return (rr * cosD(mStaveTilt));
  } else {
    return -1;
  }
}

void V1Layer::setNumberOfUnits(Int_t u)
{
  if (mLayerNumber < sNumberOmInnerLayers) {
    mNumberOfChips = u;
  } else {
    mNumberOfModules = u;
    mNumberOfChips = sOBChipsPerRow;
  }
}

void V1Layer::setStaveTilt(const Double_t t)
{
  if (mIsTurbo) {
    mStaveTilt = t;
  } else {
    LOG(ERROR) << "Not a Turbo layer";
  }
}

void V1Layer::setStaveWidth(const Double_t w)
{
  if (mIsTurbo) {
    mStaveWidth = w;
  } else {
    LOG(ERROR) << "Not a Turbo layer";
  }
}

TGeoArb8* V1Layer::createStaveSide(const char* name, Double_t dz, Double_t angle,
                                   Double_t xSign, Double_t L, Double_t H, Double_t l)
{
  // Create one half of the V shape corner of CF stave

  auto* cmStavSide = new TGeoArb8(dz);
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

TGeoCombiTrans* V1Layer::createCombiTrans(const char* name, Double_t dy, Double_t dz,
                                          Double_t dphi, Bool_t planeSym)
{
  TGeoTranslation t1(dy * cosD(90. + dphi), dy * sinD(90. + dphi), dz);
  TGeoRotation r1("", 0., 0., dphi);
  TGeoRotation r2("", 90, 180, -90 - dphi);

  auto* combiTrans1 = new TGeoCombiTrans(name);
  combiTrans1->SetTranslation(t1);
  if (planeSym) {
    combiTrans1->SetRotation(r1);
  } else {
    combiTrans1->SetRotation(r2);
  }
  return combiTrans1;
}

void V1Layer::addTranslationToCombiTrans(TGeoCombiTrans* ct, Double_t dx, Double_t dy,
                                         Double_t dz) const
{
  // Add a dx,dy,dz translation to the initial TGeoCombiTrans
  const Double_t* vect = ct->GetTranslation();
  Double_t newVect[3] = {vect[0] + dx, vect[1] + dy, vect[2] + dz};
  ct->SetTranslation(newVect);
}
