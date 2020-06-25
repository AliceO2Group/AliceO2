// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file V3Layer.cxx
/// \brief Implementation of the V3Layer class
/// \author Mario Sitta <sitta@to.infn.it>
/// \author Chinorat Kobdaj (kobdaj@g.sut.ac.th)

#include "ITSSimulation/V3Layer.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSSimulation/Detector.h"
#include "ITSMFTSimulation/AlpideChip.h"
#include "ITSMFTBase/SegmentationAlpide.h"

#include "FairLogger.h" // for LOG

#include <TGeoArb8.h>           // for TGeoArb8
#include <TGeoBBox.h>           // for TGeoBBox
#include <TGeoCone.h>           // for TGeoConeSeg, TGeoCone
#include <TGeoPcon.h>           // for TGeoPcon
#include <TGeoManager.h>        // for TGeoManager, gGeoManager
#include <TGeoMatrix.h>         // for TGeoCombiTrans, TGeoRotation, etc
#include <TGeoTrd1.h>           // for TGeoTrd1
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
using namespace o2::itsmft;
using AlpideChip = o2::itsmft::AlpideChip;

// General Parameters
const Int_t V3Layer::sNumberOfInnerLayers = 3;

// Inner Barrel Parameters
const Int_t V3Layer::sIBChipsPerRow = 9;
const Int_t V3Layer::sIBNChipRows = 1;
const Double_t V3Layer::sIBChipZGap = 150.0 * sMicron;

const Double_t V3Layer::sIBModuleZLength = 27.12 * sCm;
const Double_t V3Layer::sIBFPCWiderXPlus = 850.0 * sMicron;
const Double_t V3Layer::sIBFPCWiderXNeg = 300.0 * sMicron;
const Double_t V3Layer::sIBFlexCableAlThick = 25.0 * sMicron;
const Double_t V3Layer::sIBFPCAlGNDWidth = (4.1 + 11.15) * sMm;
const Double_t V3Layer::sIBFPCAlAnodeWidth1 = 13.0 * sMm;
const Double_t V3Layer::sIBFPCAlAnodeWidth2 = 14.7 * sMm;
const Double_t V3Layer::sIBFlexCableKapThick = 75.0 * sMicron;
const Double_t V3Layer::sIBFlexCablePolyThick = 20.0 * sMicron;
const Double_t V3Layer::sIBFlexCapacitorXWid = 0.2 * sMm;
const Double_t V3Layer::sIBFlexCapacitorYHi = 0.2 * sMm;
const Double_t V3Layer::sIBFlexCapacitorZLen = 0.4 * sMm;
const Double_t V3Layer::sIBColdPlateWidth = 15.4 * sMm;
const Double_t V3Layer::sIBColdPlateZLen = 290.0 * sMm;
const Double_t V3Layer::sIBGlueThick = 50.0 * sMicron;
const Double_t V3Layer::sIBCarbonFleeceThick = 20.0 * sMicron;
const Double_t V3Layer::sIBCarbonPaperThick = 30.0 * sMicron;
const Double_t V3Layer::sIBCarbonPaperWidth = 12.5 * sMm;
const Double_t V3Layer::sIBCarbonPaperZLen = 280.0 * sMm;
const Double_t V3Layer::sIBK13D2UThick = 70.0 * sMicron;
const Double_t V3Layer::sIBCoolPipeInnerD = 1.024 * sMm;
const Double_t V3Layer::sIBCoolPipeThick = 25.4 * sMicron;
const Double_t V3Layer::sIBCoolPipeXDist = 5.0 * sMm;
const Double_t V3Layer::sIBCoolPipeZLen = 302.0 * sMm;
const Double_t V3Layer::sIBTopVertexWidth1 = 0.258 * sMm;
const Double_t V3Layer::sIBTopVertexWidth2 = 0.072 * sCm;
const Double_t V3Layer::sIBTopVertexHeight = 0.04 * sCm;
const Double_t V3Layer::sIBTopVertexAngle = 60.0; // Deg
const Double_t V3Layer::sIBSideVertexWidth = 0.05 * sCm;
const Double_t V3Layer::sIBSideVertexHeight = 0.074 * sCm;
const Double_t V3Layer::sIBTopFilamentSide = 0.04 * sCm;
const Double_t V3Layer::sIBTopFilamentAlpha = 109.8; // Deg
const Double_t V3Layer::sIBTopFilamentInterZ = 15.0 * sMm;
const Double_t V3Layer::sIBEndSupportThick = 0.149 * sMm;
const Double_t V3Layer::sIBEndSupportZLen = 2.5 * sMm;
const Double_t V3Layer::sIBEndSupportXUp = 1.0 * sMm;
const Double_t V3Layer::sIBEndSupportOpenPhi = 120.0; // Deg

const Double_t V3Layer::sIBConnectorXWidth = 10.0 * sMm;
const Double_t V3Layer::sIBConnectorYTot = 4.7 * sMm;
const Double_t V3Layer::sIBConnectBlockZLen = 16.5 * sMm;
const Double_t V3Layer::sIBConnBodyYHeight = 2.5 * sMm;
const Double_t V3Layer::sIBConnTailYShift = 0.9 * sMm;
const Double_t V3Layer::sIBConnTailYMid = 2.5 * sMm;
const Double_t V3Layer::sIBConnTailZLen = 2.5 * sMm;
const Double_t V3Layer::sIBConnTailOpenPhi = 120.0; // Deg
const Double_t V3Layer::sIBConnRoundHoleD = 2.0 * sMm;
const Double_t V3Layer::sIBConnRoundHoleZ = (9.0 - 4.0) * sMm;
const Double_t V3Layer::sIBConnSquareHoleX = 2.0 * sMm;
const Double_t V3Layer::sIBConnSquareHoleZ = 2.8 * sMm;
const Double_t V3Layer::sIBConnSquareHoleZPos = 9.0 * sMm;
const Double_t V3Layer::sIBConnInsertHoleD = 2.0 * sMm;
const Double_t V3Layer::sIBConnInsertHoleZPos = 9.0 * sMm;
const Double_t V3Layer::sIBConnTubeHole1D = 1.6 * sMm;
const Double_t V3Layer::sIBConnTubeHole1ZLen = 3.0 * sMm;
const Double_t V3Layer::sIBConnTubeHole1ZLen2 = 2.7 * sMm;
const Double_t V3Layer::sIBConnTubeHole2D = 1.2 * sMm;
const Double_t V3Layer::sIBConnTubeHole3XPos = 1.0 * sMm;
const Double_t V3Layer::sIBConnTubeHole3ZPos = 14.5 * sMm;
const Double_t V3Layer::sIBConnTubesXDist = 5.0 * sMm;
const Double_t V3Layer::sIBConnTubesYPos = 1.25 * sMm;
const Double_t V3Layer::sIBConnInsertD = 2.0 * sMm;
const Double_t V3Layer::sIBConnInsertHeight = 2.3 * sMm;
const Double_t V3Layer::sIBConnSideHole1D = 1.0 * sMm;
const Double_t V3Layer::sIBConnSideHole1YPos = 1.25 * sMm;
const Double_t V3Layer::sIBConnSideHole1ZPos = 11.5 * sMm;
const Double_t V3Layer::sIBConnSideHole1XWid = 1.0 * sMm;
const Double_t V3Layer::sIBConnSideHole2YPos = 1.25 * sMm;
const Double_t V3Layer::sIBConnSideHole2ZPos = 11.0 * sMm;
const Double_t V3Layer::sIBConnSideHole2XWid = 1.0 * sMm;
const Double_t V3Layer::sIBConnSideHole2YWid = 1.0 * sMm;
const Double_t V3Layer::sIBConnSideHole2ZWid = 1.0 * sMm;
const Double_t V3Layer::sIBConnectAFitExtD = 1.65 * sMm;
const Double_t V3Layer::sIBConnectAFitIntD = 1.19 * sMm;
const Double_t V3Layer::sIBConnectAFitZLen = 12.5 * sMm;
const Double_t V3Layer::sIBConnectAFitZOut = 10.0 * sMm;
const Double_t V3Layer::sIBConnPlugInnerD = 0.8 * sMm;
const Double_t V3Layer::sIBConnPlugTotLen = 1.7 * sMm;
const Double_t V3Layer::sIBConnPlugInnerLen = 1.0 * sMm;

const Double_t V3Layer::sIBStaveHeight = 0.5 * sCm;

// Outer Barrel Parameters
const Int_t V3Layer::sOBChipsPerRow = 7;
const Int_t V3Layer::sOBNChipRows = 2;

const Double_t V3Layer::sOBChipThickness = 100.0 * sMicron;

const Double_t V3Layer::sOBHalfStaveWidth = 3.01 * sCm;
const Double_t V3Layer::sOBModuleGap = 200.0 * sMicron;
const Double_t V3Layer::sOBChipXGap = 150.0 * sMicron;
const Double_t V3Layer::sOBChipZGap = 150.0 * sMicron;
const Double_t V3Layer::sOBFlexCableXWidth = 3.3 * sCm;
const Double_t V3Layer::sOBFlexCableAlThick = 0.005 * sCm;
const Double_t V3Layer::sOBFlexCableKapThick = 75.0 * sMicron;
const Double_t V3Layer::sOBFPCSoldMaskThick = 30.0 * sMicron;
const Double_t V3Layer::sOBFPCCopperThick = 18.0 * sMicron;
const Double_t V3Layer::sOBFPCCuAreaFracGnd = 0.954; // F.Benotto
const Double_t V3Layer::sOBFPCCuAreaFracSig = 0.617; // F.Benotto
const Double_t V3Layer::sOBGlueFPCThick = 50 * sMicron;
const Double_t V3Layer::sOBGlueColdPlThick = 80 * sMicron;
const Double_t V3Layer::sOBPowerBusXWidth = 3.04 * sCm;
const Double_t V3Layer::sOBPowerBusAlThick = 100.0 * sMicron;
const Double_t V3Layer::sOBPowerBusAlFrac = 0.90; // L.Greiner
const Double_t V3Layer::sOBPowerBusDielThick = 50.0 * sMicron;
const Double_t V3Layer::sOBPowerBusKapThick = 27.5 * sMicron;
const Double_t V3Layer::sOBBiasBusXWidth = 7.7 * sMm;
const Double_t V3Layer::sOBBiasBusAlThick = 25.0 * sMicron;
const Double_t V3Layer::sOBBiasBusAlFrac = 0.90; // L.Greiner
const Double_t V3Layer::sOBBiasBusDielThick = 50.0 * sMicron;
const Double_t V3Layer::sOBBiasBusKapThick = 25.0 * sMicron;
const Double_t V3Layer::sOBColdPlateXWidth = 3.04 * sCm;
const Double_t V3Layer::sOBColdPlateZLenML = 87.55 * sCm;
const Double_t V3Layer::sOBColdPlateZLenOL = 150.15 * sCm;
const Double_t V3Layer::sOBColdPlateThick = 0.012 * sCm;
const Double_t V3Layer::sOBHalfStaveYPos = 2.067 * sCm;
const Double_t V3Layer::sOBHalfStaveYTrans = 1.76 * sMm;
const Double_t V3Layer::sOBHalfStaveXOverlap = 7.2 * sMm;
const Double_t V3Layer::sOBGraphiteFoilThick = 30.0 * sMicron;
const Double_t V3Layer::sOBCarbonFleeceThick = 20.0 * sMicron;
const Double_t V3Layer::sOBCoolTubeInnerD = 2.05 * sMm;
const Double_t V3Layer::sOBCoolTubeThick = 32.0 * sMicron;
const Double_t V3Layer::sOBCoolTubeXDist = 10.0 * sMm;

const Double_t V3Layer::sOBCPConnectorXWidth = 16.0 * sMm;
const Double_t V3Layer::sOBCPConnBlockZLen = 15.0 * sMm;
const Double_t V3Layer::sOBCPConnBlockYHei = 3.6 * sMm;
const Double_t V3Layer::sOBCPConnHollowZLen = 3.0 * sMm;
const Double_t V3Layer::sOBCPConnHollowYHei = 0.9 * sMm;
const Double_t V3Layer::sOBCPConnSquareHoleX = 4.0 * sMm;
const Double_t V3Layer::sOBCPConnSquareHoleZ = 5.0 * sMm;
const Double_t V3Layer::sOBCPConnSqrHoleZPos = 4.0 * sMm;
const Double_t V3Layer::sOBCPConnSqrInsertRZ = 3.5 * sMm;
const Double_t V3Layer::sOBCPConnRoundHoleD = 4.0 * sMm;
const Double_t V3Layer::sOBCPConnRndHoleZPos = 7.0 * sMm;
const Double_t V3Layer::sOBCPConnTubesXDist = 10.0 * sMm;
const Double_t V3Layer::sOBCPConnTubesYPos = 1.8 * sMm;
const Double_t V3Layer::sOBCPConnTubeHole1D = 2.6 * sMm;
const Double_t V3Layer::sOBCPConnTubeHole1Z = 3.5 * sMm;
const Double_t V3Layer::sOBCPConnTubeHole2D = 2.2 * sMm;
const Double_t V3Layer::sOBCPConnFitHoleD = 2.8 * sMm;
const Double_t V3Layer::sOBCPConnTubeHole3XP = 1.0 * sMm;
const Double_t V3Layer::sOBCPConnTubeHole3ZP = 2.0 * sMm;
const Double_t V3Layer::sOBCPConnInstZThick = 1.0 * sMm;
const Double_t V3Layer::sOBCPConnInsertYHei = 3.4 * sMm;
const Double_t V3Layer::sOBCPConnAFitExtD = 2.8 * sMm;
const Double_t V3Layer::sOBCPConnAFitThick = 0.3 * sMm;
const Double_t V3Layer::sOBCPConnAFitZLen = 17.0 * sMm;
const Double_t V3Layer::sOBCPConnAFitZIn = 3.0 * sMm;
const Double_t V3Layer::sOBCPConnPlugInnerD = 0.8 * sMm;
const Double_t V3Layer::sOBCPConnPlugTotLen = 1.7 * sMm;
const Double_t V3Layer::sOBCPConnPlugThick = 0.5 * sMm;

const Double_t V3Layer::sOBSpaceFrameZLen[2] = {900.0 * sMm, 1526.0 * sMm};
const Int_t V3Layer::sOBSpaceFrameNUnits[2] = {23, 39};
const Double_t V3Layer::sOBSpaceFrameUnitLen = 39.1 * sMm;
const Double_t V3Layer::sOBSpaceFrameWidth = 42.44 * sMm;
const Double_t V3Layer::sOBSpaceFrameHeight = 36.45 * sMm;
const Double_t V3Layer::sOBSpaceFrameTopVL = 4.0 * sMm;
const Double_t V3Layer::sOBSpaceFrameTopVH = 0.35 * sMm;
const Double_t V3Layer::sOBSpaceFrameSideVL = 4.5 * sMm;
const Double_t V3Layer::sOBSpaceFrameSideVH = 0.35 * sMm;
const Double_t V3Layer::sOBSpaceFrameVAlpha = 60.0; // deg
const Double_t V3Layer::sOBSpaceFrameVBeta = 68.0;  // deg
const Double_t V3Layer::sOBSFrameBaseRibDiam = 1.33 * sMm;
const Double_t V3Layer::sOBSFrameBaseRibPhi = 54.0; // deg
const Double_t V3Layer::sOBSFrameSideRibDiam = 1.25 * sMm;
const Double_t V3Layer::sOBSFrameSideRibPhi = 70.0; // deg
const Double_t V3Layer::sOBSFrameULegLen = 14.2 * sMm;
const Double_t V3Layer::sOBSFrameULegWidth = 1.5 * sMm;
const Double_t V3Layer::sOBSFrameULegHeight1 = 2.7 * sMm;
const Double_t V3Layer::sOBSFrameULegHeight2 = 5.0 * sMm;
const Double_t V3Layer::sOBSFrameULegThick = 0.3 * sMm;
const Double_t V3Layer::sOBSFrameULegXPos = 12.9 * sMm;
const Double_t V3Layer::sOBSFrameConnWidth = 42.0 * sMm;
const Double_t V3Layer::sOBSFrameConnTotLen = 29.0 * sMm;
const Double_t V3Layer::sOBSFrameConnTotHei = 4.8 * sMm;
const Double_t V3Layer::sOBSFrameConnTopLen = 14.0 * sMm;
const Double_t V3Layer::sOBSFrameConnInsWide = 36.869 * sMm;
const Double_t V3Layer::sOBSFrameConnInsBase = 39.6 * sMm;
const Double_t V3Layer::sOBSFrameConnInsHei = 2.8 * sMm;
const Double_t V3Layer::sOBSFrameConnHoleZPos = 7.0 * sMm;
const Double_t V3Layer::sOBSFrameConnHoleZDist = 15.0 * sMm;
const Double_t V3Layer::sOBSFrameConnTopHoleD = 3.0 * sMm;
const Double_t V3Layer::sOBSFrConnTopHoleXDist = 24.0 * sMm;
const Double_t V3Layer::sOBSFrameConnAHoleWid = 4.0 * sMm;
const Double_t V3Layer::sOBSFrameConnAHoleLen = 5.0 * sMm;
const Double_t V3Layer::sOBSFrConnASideHoleD = 3.0 * sMm;
const Double_t V3Layer::sOBSFrConnASideHoleL = 2.5 * sMm;
const Double_t V3Layer::sOBSFrConnASideHoleY = 2.3 * sMm;
const Double_t V3Layer::sOBSFrameConnCHoleZPos = 3.0 * sMm;
const Double_t V3Layer::sOBSFrConnCHoleXDist = 32.0 * sMm;
const Double_t V3Layer::sOBSFrConnCTopHoleD = 4.0 * sMm;
const Double_t V3Layer::sOBSFrameConnInsHoleD = 5.0 * sMm;
const Double_t V3Layer::sOBSFrameConnInsHoleX = 25.8 * sMm;

ClassImp(V3Layer);

#define SQ(A) (A) * (A)

V3Layer::V3Layer()
  : V11Geometry(),
    mLayerNumber(0),
    mPhi0(0),
    mLayerRadius(0),
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
    mStaveModel(Detector::kIBModelDummy),
    mAddGammaConv(kFALSE),
    mGammaConvDiam(0),
    mGammaConvXPos(0),
    mIBModuleZLength(0),
    mOBModuleZLength(0)
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
    mStaveModel(Detector::kIBModelDummy),
    mAddGammaConv(kFALSE),
    mGammaConvDiam(0),
    mGammaConvXPos(0),
    mIBModuleZLength(0),
    mOBModuleZLength(0)
{
  for (int i = kNHLevels; i--;) {
    mHierarchy[i] = 0;
  }
}

V3Layer::~V3Layer() = default;

void V3Layer::createLayer(TGeoVolume* motherVolume)
{
  const Int_t nameLen = 30;
  char volumeName[nameLen];
  Double_t xpos, ypos, zpos;
  Double_t alpha;

  // Check if the user set the proper parameters
  if (mLayerRadius <= 0) {
    LOG(FATAL) << "Wrong layer radius " << mLayerRadius;
  }

  if (mNumberOfStaves <= 0) {
    LOG(FATAL) << "Wrong number of staves " << mNumberOfStaves;
  }

  if (mNumberOfChips <= 0) {
    LOG(FATAL) << "Wrong number of chips " << mNumberOfChips;
  }

  if (mLayerNumber >= sNumberOfInnerLayers && mNumberOfModules <= 0) {
    LOG(FATAL) << "Wrong number of modules " << mNumberOfModules;
  }

  if (mChipThickness <= 0) {
    LOG(FATAL) << "Chip thickness wrong or not set " << mChipThickness;
  }

  if (mSensorThickness <= 0) {
    LOG(FATAL) << "Sensor thickness wrong or not set " << mSensorThickness;
  }

  if (mSensorThickness > mChipThickness) {
    LOG(FATAL) << "Sensor thickness " << mSensorThickness << " is greater than chip thickness " << mChipThickness;
  }

  // If a Turbo layer is requested, do it and exit
  if (mIsTurbo) {
    createLayerTurbo(motherVolume);
    return;
  }

  // First create the stave container
  alpha = (360. / (2 * mNumberOfStaves)) * DegToRad();

  //  mStaveWidth = mLayerRadius*Tan(alpha);

  snprintf(volumeName, nameLen, "%s%d", GeometryTGeo::getITSLayerPattern(), mLayerNumber);
  TGeoVolume* layerVolume = new TGeoVolumeAssembly(volumeName);
  layerVolume->SetUniqueID(mChipTypeID);

  // layerVolume->SetVisibility(kFALSE);
  layerVolume->SetVisibility(kTRUE);
  layerVolume->SetLineColor(1);

  TGeoVolume* stavVol = createStave();

  // Now build up the layer
  alpha = 360. / mNumberOfStaves;
  Double_t r = mLayerRadius + (static_cast<TGeoBBox*>(stavVol->GetShape()))->GetDY();
  for (Int_t j = 0; j < mNumberOfStaves; j++) {
    Double_t phi = j * alpha + mPhi0;
    xpos = r * cosD(phi); // r*sinD(-phi);
    ypos = r * sinD(phi); // r*cosD(-phi);
    zpos = 0.;
    phi += 90;
    layerVolume->AddNode(stavVol, j, new TGeoCombiTrans(xpos, ypos, zpos, new TGeoRotation("", phi, 0, 0)));
  }

  // Finally put everything in the mother volume
  motherVolume->AddNode(layerVolume, 1, nullptr);

  //  geometry is served
  return;
}

void V3Layer::createLayerTurbo(TGeoVolume* motherVolume)
{
  const Int_t nameLen = 30;
  char volumeName[nameLen];
  Double_t xpos, ypos, zpos;
  Double_t alpha;

  // Check if the user set the proper (remaining) parameters
  if (mStaveWidth <= 0) {
    LOG(FATAL) << "Wrong stave width " << mStaveWidth;
  }

  if (Abs(mStaveTilt) > 45) {
    LOG(WARNING) << "Stave tilt angle (" << mStaveTilt << ") greater than 45deg";
  }

  snprintf(volumeName, nameLen, "%s%d", GeometryTGeo::getITSLayerPattern(), mLayerNumber);
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
    layerVolume->AddNode(stavVol, j,
                         new TGeoCombiTrans(xpos, ypos, zpos, new TGeoRotation("", phi - mStaveTilt, 0, 0)));
  }

  // Finally put everything in the mother volume
  motherVolume->AddNode(layerVolume, 1, nullptr);

  return;
}

TGeoVolume* V3Layer::createStave(const TGeoManager* /*mgr*/)
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
  // Updated:      10 Jan 2018  Mario Sitta  Compute all dimensions using
  //                                         AlpideChip as basis
  //

  const Int_t nameLen = 30;
  char volumeName[nameLen];

  Double_t xpos, ypos;
  Double_t alpha;

  // First create all needed shapes
  alpha = (360. / (2 * mNumberOfStaves)) * DegToRad();

  // The stave
  snprintf(volumeName, nameLen, "%s%d", GeometryTGeo::getITSStavePattern(), mLayerNumber);
  TGeoVolume* staveVol = new TGeoVolumeAssembly(volumeName);
  staveVol->SetVisibility(kTRUE);
  staveVol->SetLineColor(2);

  TGeoVolume* mechStaveVol = nullptr;

  // Now build up the stave
  if (mLayerNumber < sNumberOfInnerLayers) {
    TGeoVolume* modVol = createStaveInnerB();
    ypos = (static_cast<TGeoBBox*>(modVol->GetShape()))->GetDY() - mChipThickness; // = 0 if not kIBModel4
    staveVol->AddNode(modVol, 0, new TGeoTranslation(0, ypos, 0));
    mHierarchy[kHalfStave] = 1;

    // Mechanical stave structure
    mechStaveVol = createStaveStructInnerB();
    if (mechStaveVol) {
      ypos = (static_cast<TGeoBBox*>(modVol->GetShape()))->GetDY() - ypos;
      if (mStaveModel != Detector::kIBModel4)
        ypos += (static_cast<TGeoBBox*>(mechStaveVol->GetShape()))->GetDY();
      staveVol->AddNode(mechStaveVol, 1, new TGeoCombiTrans(0, -ypos, 0, new TGeoRotation("", 0, 0, 180)));
    }
  } else {
    TGeoVolume* hstaveVol = createStaveOuterB();
    if (mStaveModel == Detector::kOBModel0) { // Create simplified stave struct as in v0
      staveVol->AddNode(hstaveVol, 0);
      mHierarchy[kHalfStave] = 1;
    } else { // (if mStaveModel) Create new stave struct as in TDR
      xpos = (static_cast<TGeoBBox*>(hstaveVol->GetShape()))->GetDX() - sOBHalfStaveXOverlap / 2;
      // ypos is now a parameter to avoid HS displacement wrt nominal radii
      ypos = sOBHalfStaveYPos;
      staveVol->AddNode(hstaveVol, 0, new TGeoTranslation(-xpos, ypos, 0));
      staveVol->AddNode(hstaveVol, 1, new TGeoTranslation(xpos, ypos + sOBHalfStaveYTrans, 0));
      mHierarchy[kHalfStave] = 2; // RS
      mechStaveVol = createSpaceFrameOuterB();

      if (mechStaveVol) {
        if (mBuildLevel < 6) // Carbon
          staveVol->AddNode(mechStaveVol, 1,
                            new TGeoCombiTrans(0, -sOBSFrameULegHeight1, 0, new TGeoRotation("", 180, 0, 0)));
      }
    }
  }

  staveVol->GetShape()->ComputeBBox(); // RS: enfore recompting of BBox

  // Done, return the stave
  return staveVol;
}

TGeoVolume* V3Layer::createStaveInnerB(const TGeoManager* mgr)
{
  Double_t xmod, ymod, zmod;
  const Int_t nameLen = 30;
  char volumeName[nameLen];

  // First we create the module (i.e. the HIC with 9 chips)
  TGeoVolume* moduleVol = createModuleInnerB();

  // Then we create the fake halfstave and the actual stave
  xmod = (static_cast<TGeoBBox*>(moduleVol->GetShape()))->GetDX();
  ymod = (static_cast<TGeoBBox*>(moduleVol->GetShape()))->GetDY();
  zmod = (static_cast<TGeoBBox*>(moduleVol->GetShape()))->GetDZ();

  TGeoBBox* hstave = new TGeoBBox(xmod, ymod, zmod);

  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");

  snprintf(volumeName, nameLen, "%s%d", GeometryTGeo::getITSHalfStavePattern(), mLayerNumber);
  TGeoVolume* hstaveVol = new TGeoVolume(volumeName, hstave, medAir);

  // Finally build it up
  hstaveVol->AddNode(moduleVol, 0);
  mHierarchy[kModule] = 1;

  // Done, return the stave structure
  return hstaveVol;
}

TGeoVolume* V3Layer::createModuleInnerB(const TGeoManager* mgr)
{
  Double_t xtot, ytot, ztot, xchip, zchip, ymod;
  Double_t xpos, ypos, zpos;
  Bool_t dummyChip;
  const Int_t nameLen = 30;
  char chipName[nameLen], sensName[nameLen], volumeName[nameLen];

  // For material budget studies
  if (mBuildLevel < 6)
    dummyChip = kFALSE; // will be made of Si
  else
    dummyChip = kTRUE; // will be made of Air

  // First create the single chip
  snprintf(chipName, nameLen, "%s%d", GeometryTGeo::getITSChipPattern(), mLayerNumber);
  snprintf(sensName, nameLen, "%s%d", GeometryTGeo::getITSSensorPattern(), mLayerNumber);

  ymod = 0.5 * mChipThickness;

  TGeoVolume* chipVol = AlpideChip::createChip(ymod, mSensorThickness / 2, chipName, sensName, dummyChip);

  xchip = (static_cast<TGeoBBox*>(chipVol->GetShape()))->GetDX();
  zchip = (static_cast<TGeoBBox*>(chipVol->GetShape()))->GetDZ();

  mIBModuleZLength = 2 * zchip * sIBChipsPerRow + (sIBChipsPerRow - 1) * sIBChipZGap;

  // Then create the Glue, the Kapton and the two Aluminum cables
  xtot = xchip + (sIBFPCWiderXPlus + sIBFPCWiderXNeg) / 2;
  ztot = mIBModuleZLength / 2;

  TGeoBBox* glue = new TGeoBBox(xchip, sIBGlueThick / 2, ztot);
  TGeoBBox* kapCable = new TGeoBBox(xtot, sIBFlexCableKapThick / 2, ztot);

  TGeoVolume* aluGndCableVol = createIBFPCAlGnd(xtot, ztot);
  TGeoVolume* aluAnodeCableVol = createIBFPCAlAnode(xtot, ztot);

  // Finally create the module and populate it with the chips
  // (and the FPC Kapton and Aluminum in the most recent IB model)
  Double_t ygnd = (static_cast<TGeoBBox*>(aluGndCableVol->GetShape()))->GetDY();
  Double_t yano = (static_cast<TGeoBBox*>(aluAnodeCableVol->GetShape()))->GetDY();

  ytot = ymod;
  if (mStaveModel == Detector::kIBModel4)
    ytot += (sIBGlueThick / 2 + ygnd + sIBFlexCableKapThick / 2 + yano + sIBFlexCapacitorYHi / 2);

  TGeoBBox* module = new TGeoBBox(xtot, ytot, ztot);

  // Now the volumes
  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");
  TGeoMedium* medKapton = mgr->GetMedium("ITS_KAPTON(POLYCH2)$");
  TGeoMedium* medGlue = mgr->GetMedium("ITS_GLUE_IBFPC$");

  snprintf(volumeName, nameLen, "%s%d", GeometryTGeo::getITSModulePattern(), mLayerNumber);
  TGeoVolume* modVol = new TGeoVolume(volumeName, module, medAir);

  TGeoVolume* glueVol = new TGeoVolume("FPCGlue", glue, medGlue);
  glueVol->SetLineColor(kBlack);
  glueVol->SetFillColor(kBlack);

  TGeoVolume* kapCableVol = new TGeoVolume("FPCKapton", kapCable, medKapton);
  kapCableVol->SetLineColor(kBlue);
  kapCableVol->SetFillColor(kBlue);

  // Build up the module
  // Chips are rotated by 180deg around Y axis
  // in order to have the correct X and Z axis orientation
  xpos = -xtot + (static_cast<TGeoBBox*>(chipVol->GetShape()))->GetDX() + sIBFPCWiderXNeg;
  ypos = -ytot + ymod; // = 0 if not kIBModel4
  for (Int_t j = 0; j < sIBChipsPerRow; j++) {
    zpos = ztot - j * (2 * zchip + sIBChipZGap) - zchip;
    modVol->AddNode(chipVol, j, new TGeoCombiTrans(xpos, ypos, zpos, new TGeoRotation("", 0, 180, 180)));
    mHierarchy[kChip]++;
  }

  if (mStaveModel == Detector::kIBModel4) {
    ypos += (ymod + glue->GetDY());
    if (mBuildLevel < 2) // Glue
      modVol->AddNode(glueVol, 1, new TGeoTranslation(xpos, ypos, 0));
    ypos += glue->GetDY();

    if (mBuildLevel < 4) { // Kapton
      ypos += ygnd;
      modVol->AddNode(aluGndCableVol, 1, new TGeoTranslation(0, ypos, 0));

      ypos += (ygnd + kapCable->GetDY());
      modVol->AddNode(kapCableVol, 1, new TGeoTranslation(0, ypos, 0));

      ypos += (kapCable->GetDY() + yano);
      modVol->AddNode(aluAnodeCableVol, 1, new TGeoTranslation(0, ypos, 0));

      ypos += yano;
    }
  }

  // Add the capacitors
  createIBCapacitors(modVol, zchip, ypos);

  // Done, return the module
  return modVol;
}

void V3Layer::createIBCapacitors(TGeoVolume* modvol, Double_t zchip, Double_t yzero, const TGeoManager* mgr)
{
  //
  // Adds the capacitors to the IB FPC
  //
  // Created:      13 Feb 2018  Mario Sitta
  // Updated:      03 Apr 2019  Mario Sitta  Fix positions (180' rotation)
  //

  // Position of the various capacitors (A.Junique private communication
  // where: X_capacitor = Z_module , Y_capacitor = X_module)
  // Capacitors (different groups)
  const Double_t xGroup1A = 4265.9 * sMicron;
  const Double_t zGroup1A[2] = {-7142.9 * sMicron, 7594.1 * sMicron};
  const Double_t xGroup1B = 690.9 * sMicron;
  const Double_t zGroup1B = -7142.9 * sMicron;
  const Double_t xGroup2 = 6300.0 * sMicron;
  const Double_t zGroup2 = 15075.0 * sMicron;
  const Double_t xGroup3 = 5575.0 * sMicron;
  const Double_t zGroup3 = 131900.0 * sMicron;
  const Double_t xGroup4[2] = {5600.0 * sMicron, 5575.0 * sMicron};
  const Double_t zGroup4[sIBChipsPerRow] = {275.0 * sMicron, 250.0 * sMicron, 275.0 * sMicron,
                                            250.0 * sMicron, 250.0 * sMicron, 300.0 * sMicron,
                                            250.0 * sMicron, 300.0 * sMicron, 250.0 * sMicron};
  const Int_t nGroup5A = 5, nGroup5B = 4;
  const Double_t xGroup5A[2] = {1400.0 * sMicron, 1350.0 * sMicron};
  const Double_t zGroup5A[nGroup5A] = {-112957.5 * sMicron, -82854.5 * sMicron, 7595.5 * sMicron, 37745.5 * sMicron,
                                       128194.1 * sMicron};
  const Double_t xGroup5B = 1100.0 * sMicron;
  const Double_t zGroup5B[nGroup5B] = {-51525.0 * sMicron, -21375.0 * sMicron, 69075.0 * sMicron, 99225.0 * sMicron};
  // Resistors
  const Int_t nResist = 2;
  const Double_t xResist = -7975.0 * sMicron;
  const Double_t zResist[nResist] = {114403.0 * sMicron, 119222.0 * sMicron};

  Double_t xpos, ypos, zpos;
  Int_t nCapacitors;

  TGeoVolume *capacitor, *resistor;

  // Check whether we already have the volume, otherwise create it
  // (so as to avoid creating multiple copies of the very same volume
  // for each layer)
  capacitor = mgr->GetVolume("IBFPCCapacitor");

  if (!capacitor) {
    TGeoBBox* capsh = new TGeoBBox(sIBFlexCapacitorXWid / 2, sIBFlexCapacitorYHi / 2, sIBFlexCapacitorZLen / 2);

    TGeoMedium* medCeramic = mgr->GetMedium("ITS_CERAMIC$");

    capacitor = new TGeoVolume("IBFPCCapacitor", capsh, medCeramic);
    capacitor->SetLineColor(kBlack);
    capacitor->SetFillColor(kBlack);

    TGeoBBox* ressh = new TGeoBBox(sIBFlexCapacitorXWid / 2,  // Resistors have
                                   sIBFlexCapacitorYHi / 2,   // the same dim's
                                   sIBFlexCapacitorZLen / 2); // as capacitors

    resistor = new TGeoVolume("IBFPCResistor", ressh, medCeramic);
    resistor->SetLineColor(kBlack);
    resistor->SetFillColor(kBlack);
  } else { // Volumes already defined, get them
    resistor = mgr->GetVolume("IBFPCResistor");
  }

  // Place all the capacitors (they are really a lot...)
  ypos = yzero + sIBFlexCapacitorYHi / 2;

  xpos = xGroup1A;
  for (Int_t j = 0; j < sIBChipsPerRow; j++) {
    zpos = -mIBModuleZLength / 2 + j * (2 * zchip + sIBChipZGap) + zchip + zGroup1A[0];
    modvol->AddNode(capacitor, 2 * j + 1, new TGeoTranslation(-xpos, ypos, -zpos));
    zpos = -mIBModuleZLength / 2 + j * (2 * zchip + sIBChipZGap) + zchip + zGroup1A[1];
    modvol->AddNode(capacitor, 2 * j + 2, new TGeoTranslation(-xpos, ypos, -zpos));
  }

  nCapacitors = 2 * sIBChipsPerRow;
  xpos = xGroup1B;
  for (Int_t j = 0; j < sIBChipsPerRow; j++) {
    zpos = -mIBModuleZLength / 2 + j * (2 * zchip + sIBChipZGap) + zchip + zGroup1B;
    modvol->AddNode(capacitor, j + 1 + nCapacitors, new TGeoTranslation(-xpos, ypos, -zpos));
  }

  nCapacitors += sIBChipsPerRow;
  xpos = xGroup2;
  // We have only 8 in these group, missing the central one
  for (Int_t j = 0; j < sIBChipsPerRow - 1; j++) {
    zpos = -mIBModuleZLength / 2 + j * (2 * zchip + sIBChipZGap) + zchip + zGroup2;
    modvol->AddNode(capacitor, j + 1 + nCapacitors, new TGeoTranslation(-xpos, ypos, -zpos));
  }

  nCapacitors += (sIBChipsPerRow - 1);
  xpos = xGroup3;
  zpos = zGroup3;
  modvol->AddNode(capacitor, 1 + nCapacitors, new TGeoTranslation(-xpos, ypos, -zpos));

  nCapacitors++;
  for (Int_t j = 0; j < sIBChipsPerRow; j++) {
    if (j == (sIBChipsPerRow - 1))
      xpos = xGroup4[1];
    else
      xpos = xGroup4[0];
    zpos = -mIBModuleZLength / 2 + j * (2 * zchip + sIBChipZGap) + zchip + zGroup4[j];
    modvol->AddNode(capacitor, j + 1 + nCapacitors, new TGeoTranslation(-xpos, ypos, -zpos));
  }

  nCapacitors += sIBChipsPerRow;
  for (Int_t j = 0; j < nGroup5A; j++) {
    if (j == 0)
      xpos = xGroup5A[0];
    else
      xpos = xGroup5A[1];
    zpos = zGroup5A[j];
    modvol->AddNode(capacitor, j + 1 + nCapacitors, new TGeoTranslation(-xpos, ypos, -zpos));
  }

  nCapacitors += nGroup5A;
  xpos = xGroup5B;
  for (Int_t j = 0; j < nGroup5B; j++) {
    zpos = zGroup5B[j];
    modvol->AddNode(capacitor, j + 1 + nCapacitors, new TGeoTranslation(-xpos, ypos, -zpos));
  }

  // Place the resistors
  xpos = xResist;
  for (Int_t j = 0; j < nResist; j++) {
    zpos = zResist[j];
    modvol->AddNode(resistor, j + 1, new TGeoTranslation(-xpos, ypos, -zpos));
  }
}

TGeoVolume* V3Layer::createIBFPCAlGnd(const Double_t xcable, const Double_t zcable, const TGeoManager* mgr)
{
  //
  // Create the IB FPC Aluminum Ground cable
  //
  // Created:      20 Oct 2017  Mario Sitta
  //

  Double_t ytot, ypos;

  // First create all needed shapes
  ytot = sIBFlexCablePolyThick + sIBFlexCableAlThick;
  TGeoBBox* coverlay = new TGeoBBox(xcable, ytot / 2, zcable);
  TGeoBBox* aluminum = new TGeoBBox(xcable, sIBFlexCableAlThick / 2, zcable);

  // Then the volumes
  TGeoMedium* medKapton = mgr->GetMedium("ITS_KAPTON(POLYCH2)$");
  TGeoMedium* medAluminum = mgr->GetMedium("ITS_ALUMINUM$");

  TGeoVolume* coverlayVol = new TGeoVolume("FPCCoverlayGround", coverlay, medKapton);
  coverlayVol->SetLineColor(kBlue);
  coverlayVol->SetFillColor(kBlue);

  TGeoVolume* aluminumVol = new TGeoVolume("FPCAluminumGround", aluminum, medAluminum);
  aluminumVol->SetLineColor(kCyan);
  aluminumVol->SetFillColor(kCyan);

  ypos = coverlay->GetDY() - aluminum->GetDY();
  if (mBuildLevel < 1) // Aluminum
    coverlayVol->AddNode(aluminumVol, 1, new TGeoTranslation(0, ypos, 0));

  return coverlayVol;
}

TGeoVolume* V3Layer::createIBFPCAlAnode(const Double_t xcable, const Double_t zcable, const TGeoManager* mgr)
{
  //
  // Create the IB FPC Aluminum Anode cable
  //
  //
  // Created:      20 Oct 2017  Mario Sitta
  // Updated:      03 Apr 2019  Mario Sitta  Fix Al position (180' rotation)
  //

  Double_t ytot, ypos;
  Double_t xtru[4], ytru[4];

  // First create all needed shapes
  ytot = sIBFlexCablePolyThick + sIBFlexCableAlThick;
  TGeoBBox* coverlay = new TGeoBBox(xcable, ytot / 2, zcable);

  // A trapezoid
  xtru[0] = -sIBFPCAlAnodeWidth2 / 2;
  ytru[0] = -zcable;
  xtru[1] = sIBFPCAlAnodeWidth2 / 2;
  ytru[1] = ytru[0];
  xtru[2] = xtru[0] + sIBFPCAlAnodeWidth1;
  ytru[2] = zcable;
  xtru[3] = xtru[0];
  ytru[3] = ytru[2];

  TGeoXtru* aluminum = new TGeoXtru(2);
  aluminum->DefinePolygon(4, xtru, ytru);
  aluminum->DefineSection(0, -sIBFlexCableAlThick / 2);
  aluminum->DefineSection(1, sIBFlexCableAlThick / 2);

  // Then the volumes
  TGeoMedium* medKapton = mgr->GetMedium("ITS_KAPTON(POLYCH2)$");
  TGeoMedium* medAluminum = mgr->GetMedium("ITS_ALUMINUM$");

  TGeoVolume* coverlayVol = new TGeoVolume("FPCCoverlayAnode", coverlay, medKapton);
  coverlayVol->SetLineColor(kBlue);
  coverlayVol->SetFillColor(kBlue);

  TGeoVolume* aluminumVol = new TGeoVolume("FPCAluminumAnode", aluminum, medAluminum);
  aluminumVol->SetLineColor(kCyan);
  aluminumVol->SetFillColor(kCyan);

  ypos = -coverlay->GetDY() + aluminum->GetZ(1);
  if (mBuildLevel < 1) // Aluminum
    coverlayVol->AddNode(aluminumVol, 1, new TGeoCombiTrans(0, ypos, 0, new TGeoRotation("", 0, -90, 0)));

  return coverlayVol;
}

TGeoVolume* V3Layer::createStaveStructInnerB(const TGeoManager* mgr)
{
  //
  // Create the mechanical stave structure
  //
  // Created:      22 Mar 2013  Chinorat Kobdaj
  // Updated:      26 Apr 2013  Mario Sitta
  // Updated:      04 Apr 2017  Mario Sitta  O2 version - All models obsolete except last one
  // Updated:      25 Jan 2018  Mario Sitta  Stave width is now a constant
  //

  TGeoVolume* mechStavVol = nullptr;

  switch (mStaveModel) {
    case Detector::kIBModelDummy:
      mechStavVol = createStaveModelInnerBDummy(mgr);
      break;
    case Detector::kIBModel0:
    case Detector::kIBModel1:
    case Detector::kIBModel21:
    case Detector::kIBModel22:
    case Detector::kIBModel3:
      LOG(FATAL) << "Stave model " << mStaveModel << " obsolete and no longer supported";
      break;
    case Detector::kIBModel4:
      mechStavVol = createStaveModelInnerB4(mgr);
      break;
    default:
      LOG(FATAL) << "Unknown stave model " << mStaveModel;
      break;
  }
  return mechStavVol;
}

TGeoVolume* V3Layer::createStaveModelInnerBDummy(const TGeoManager*) const
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
TGeoVolume* V3Layer::createStaveModelInnerB4(const TGeoManager* mgr)
{
  //
  // Create the mechanical stave structure for Model 4 of TDR
  //
  // Input:
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
  // Updated:      25 Jan 2018  Mario Sitta  Stave width is now a constant
  // Updated:      03 Feb 2018  Mario Sitta  To last drawings (ALIITSUP0051)
  //

  // Local parameters
  const Double_t xstave = sIBColdPlateWidth / 2;

  Double_t layerHeight = 0.;

  Double_t rPipeMin = sIBCoolPipeInnerD / 2;
  Double_t rPipeMax = rPipeMin + sIBCoolPipeThick;

  const Int_t nv = 16;
  Double_t xv[nv], yv[nv]; // The stave container Xtru
  Double_t xlen, ylen, zlen, ztot;
  Double_t xpos, ypos, zpos, ylay, yposPipe;
  Double_t beta, gamma, theta;

  // First create all needed shapes
  ztot = sIBColdPlateZLen / 2;

  TGeoBBox* glue = new TGeoBBox(xstave, sIBGlueThick / 2, ztot);

  TGeoBBox* fleecbot = new TGeoBBox(xstave, sIBCarbonFleeceThick / 2, ztot);

  TGeoBBox* cfplate = new TGeoBBox(xstave, sIBK13D2UThick / 2, ztot);

  TGeoTube* pipe = new TGeoTube(rPipeMin, rPipeMax, sIBCoolPipeZLen / 2);

  TGeoTube* water = new TGeoTube(0., rPipeMin, sIBCoolPipeZLen / 2);

  TGeoTubeSeg* cpaptub = new TGeoTubeSeg(rPipeMax, rPipeMax + sIBCarbonPaperThick, sIBCarbonPaperZLen / 2, 0, 180);

  TGeoBBox* cpapvert = new TGeoBBox(sIBCarbonPaperThick / 2, pipe->GetRmax() / 2, sIBCarbonPaperZLen / 2);

  xlen = sIBCoolPipeXDist / 2 - pipe->GetRmax() - sIBCarbonPaperThick;
  TGeoBBox* cpapmid = new TGeoBBox(xlen, sIBCarbonPaperThick / 2, sIBCarbonPaperZLen / 2);

  xlen = sIBCarbonPaperWidth / 2 - sIBCoolPipeXDist / 2 - pipe->GetRmax() - sIBCarbonPaperThick;
  TGeoBBox* cpaplr = new TGeoBBox(xlen / 2, sIBCarbonPaperThick / 2, sIBCarbonPaperZLen / 2);

  TGeoTubeSeg* fleecpipe = new TGeoTubeSeg(cpaptub->GetRmax(), cpaptub->GetRmax() + sIBCarbonFleeceThick, ztot, 0, 180);

  TGeoBBox* fleecvert = new TGeoBBox(sIBCarbonFleeceThick / 2, (pipe->GetRmax() - sIBCarbonPaperThick) / 2, ztot);

  xlen = sIBCoolPipeXDist / 2 - pipe->GetRmax() - sIBCarbonPaperThick - sIBCarbonFleeceThick;
  TGeoBBox* fleecmid = new TGeoBBox(xlen, sIBCarbonFleeceThick / 2, ztot);

  xlen = xstave - sIBCoolPipeXDist / 2 - pipe->GetRmax() - sIBCarbonPaperThick - sIBCarbonFleeceThick;
  TGeoBBox* fleeclr = new TGeoBBox(xlen / 2, sIBCarbonFleeceThick / 2, ztot);

  // The total height of the layer can now be computed
  layerHeight = 2 * (glue->GetDY() + fleecbot->GetDY() + cfplate->GetDY() + cpaplr->GetDY() + fleeclr->GetDY());

  // The spaceframe structure
  TGeoTrd1* topv = new TGeoTrd1(sIBTopVertexWidth1 / 2, sIBTopVertexWidth2 / 2, ztot, sIBTopVertexHeight / 2);

  xv[0] = 0;
  yv[0] = 0;
  xv[1] = sIBSideVertexWidth;
  yv[1] = yv[0];
  xv[2] = xv[0];
  yv[2] = sIBSideVertexHeight;

  TGeoXtru* sidev = new TGeoXtru(2);
  sidev->DefinePolygon(3, xv, yv);
  sidev->DefineSection(0, -ztot);
  sidev->DefineSection(1, ztot);

  xv[0] = sIBEndSupportXUp / 2;
  yv[0] = sIBStaveHeight - sIBEndSupportThick;
  xv[1] = xstave - sIBSideVertexWidth;
  yv[1] = layerHeight + sIBSideVertexHeight;
  xv[2] = xstave;
  yv[2] = layerHeight;
  xv[3] = xv[2];
  yv[3] = 0;
  xv[4] = xstave + sIBEndSupportThick;
  yv[4] = yv[3];
  xv[5] = xv[4];
  yv[5] = yv[2];
  xv[6] = xv[1] + sIBEndSupportThick * sinD(sIBEndSupportOpenPhi / 2);
  yv[6] = yv[1] + sIBEndSupportThick * cosD(sIBEndSupportOpenPhi / 2);
  xv[7] = xv[0];
  yv[7] = sIBStaveHeight;
  for (Int_t i = 0; i < nv / 2; i++) {
    xv[8 + i] = -xv[7 - i];
    yv[8 + i] = yv[7 - i];
  }

  TGeoXtru* endsupp = new TGeoXtru(2);
  endsupp->DefinePolygon(16, xv, yv);
  endsupp->DefineSection(0, -sIBEndSupportZLen / 2);
  endsupp->DefineSection(1, sIBEndSupportZLen / 2);

  xlen = TMath::Sqrt((yv[7] - yv[6]) * (yv[7] - yv[6]) + (xv[7] - xv[6]) * (xv[7] - xv[6]) +
                     sIBTopFilamentInterZ * sIBTopFilamentInterZ / 4);
  theta = TMath::ATan((yv[7] - yv[6]) / (xv[7] - xv[6])) * TMath::RadToDeg();
  TGeoBBox* topfil = new TGeoBBox(xlen / 2, sIBTopFilamentSide / 2, sIBTopFilamentSide / 2);

  // The half stave container (an XTru to avoid overlaps between neighbours)
  xv[0] = xstave + sIBTopFilamentSide;
  yv[0] = 0;
  xv[1] = xv[0];
  yv[1] = layerHeight + sIBSideVertexHeight + topfil->GetDZ();
  ;
  xv[2] = sIBEndSupportXUp / 2;
  yv[2] = sIBStaveHeight + sIBTopFilamentSide / sinD(-theta); // theta is neg
  for (Int_t i = 0; i < 3; i++) {
    xv[3 + i] = -xv[2 - i];
    yv[3 + i] = yv[2 - i];
  }

  TGeoXtru* mechStruct = new TGeoXtru(2);
  mechStruct->DefinePolygon(6, xv, yv);
  mechStruct->SetName("mechStruct");
  mechStruct->DefineSection(0, -ztot);
  mechStruct->DefineSection(1, ztot);

  // The connectors' containers
  zlen = sIBConnectBlockZLen - sIBConnTailZLen + sIBConnectAFitZOut;
  TGeoBBox* connAside = new TGeoBBox("connAsideIB", sIBConnectorXWidth / 2, sIBConnBodyYHeight / 2, zlen / 2);

  zlen = sIBConnectBlockZLen - sIBConnTailZLen;
  TGeoBBox* connCside = new TGeoBBox("connCsideIB", sIBConnectorXWidth / 2, sIBConnBodyYHeight / 2, zlen / 2);

  // The StaveStruct container, a Composite Shape
  yposPipe = 2 * glue->GetDY() + 2 * fleecbot->GetDY() + 2 * cfplate->GetDY() + pipe->GetRmax();
  ypos = connAside->GetDY() - sIBConnTubesYPos + yposPipe;
  zpos = ztot + connAside->GetDZ();
  TGeoTranslation* transAside = new TGeoTranslation("transAsideIB", 0, ypos, zpos);
  transAside->RegisterYourself();

  ypos = connCside->GetDY() - sIBConnTubesYPos + yposPipe;
  zpos = ztot + connCside->GetDZ();
  TGeoTranslation* transCside = new TGeoTranslation("transCsideIB", 0, ypos, -zpos);
  transCside->RegisterYourself();

  TGeoCompositeShape* mechStavSh =
    new TGeoCompositeShape("mechStruct+connAsideIB:transAsideIB+connCsideIB:transCsideIB");

  // We have all shapes: now create the real volumes

  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");
  TGeoMedium* medWater = mgr->GetMedium("ITS_WATER$");
  TGeoMedium* medM55J6K = mgr->GetMedium("ITS_M55J6K$");
  TGeoMedium* medM60J3K = mgr->GetMedium("ITS_M60J3K$");
  TGeoMedium* medKapton = mgr->GetMedium("ITS_KAPTON(POLYCH2)$");
  TGeoMedium* medGlue = mgr->GetMedium("ITS_GLUE$");
  TGeoMedium* medK13D2U2k = mgr->GetMedium("ITS_K13D2U2k$");
  TGeoMedium* medFGS003 = mgr->GetMedium("ITS_FGS003$");
  TGeoMedium* medCarbonFleece = mgr->GetMedium("ITS_CarbonFleece$");

  const Int_t nameLen = 30;
  char volname[nameLen];
  snprintf(volname, nameLen, "%s%d_StaveStruct", GeometryTGeo::getITSStavePattern(), mLayerNumber);
  TGeoVolume* mechStavVol = new TGeoVolume(volname, mechStavSh, medAir);
  mechStavVol->SetLineColor(12);
  mechStavVol->SetFillColor(12);
  mechStavVol->SetVisibility(kFALSE);

  TGeoVolume* glueVol = new TGeoVolume("Glue", glue, medGlue);
  glueVol->SetLineColor(kBlack);
  glueVol->SetFillColor(kBlack);

  TGeoVolume* fleecbotVol = new TGeoVolume("CarbonFleeceBottom", fleecbot, medCarbonFleece);
  fleecbotVol->SetFillColor(kViolet);
  fleecbotVol->SetLineColor(kViolet);

  TGeoVolume* cfplateVol = new TGeoVolume("CFPlate", cfplate, medK13D2U2k);
  cfplateVol->SetFillColor(5); // Yellow
  cfplateVol->SetLineColor(5);

  TGeoVolume* pipeVol = new TGeoVolume("PolyimidePipe", pipe, medKapton);
  pipeVol->SetFillColor(35); // Blue shade
  pipeVol->SetLineColor(35);

  TGeoVolume* waterVol = new TGeoVolume("Water", water, medWater);
  waterVol->SetFillColor(4); // Bright blue
  waterVol->SetLineColor(4);

  TGeoVolume* cpaptubVol = new TGeoVolume("ThermasolPipeCover", cpaptub, medFGS003);
  cpaptubVol->SetFillColor(2); // Red
  cpaptubVol->SetLineColor(2);

  TGeoVolume* cpapvertVol = new TGeoVolume("ThermasolVertical", cpapvert, medFGS003);
  cpapvertVol->SetFillColor(2); // Red
  cpapvertVol->SetLineColor(2);

  TGeoVolume* cpapmidVol = new TGeoVolume("ThermasolMiddle", cpapmid, medFGS003);
  cpapmidVol->SetFillColor(2); // Red
  cpapmidVol->SetLineColor(2);

  TGeoVolume* cpaplrVol = new TGeoVolume("ThermasolLeftRight", cpaplr, medFGS003);
  cpaplrVol->SetFillColor(2); // Red
  cpaplrVol->SetLineColor(2);

  TGeoVolume* fleecpipeVol = new TGeoVolume("CarbonFleecePipeCover", fleecpipe, medCarbonFleece);
  fleecpipeVol->SetFillColor(28); // Brown shade
  fleecpipeVol->SetLineColor(28);

  TGeoVolume* fleecvertVol = new TGeoVolume("CarbonFleeceVertical", fleecvert, medCarbonFleece);
  fleecvertVol->SetFillColor(28); // Brown shade
  fleecvertVol->SetLineColor(28);

  TGeoVolume* fleecmidVol = new TGeoVolume("CarbonFleeceMiddle", fleecmid, medCarbonFleece);
  fleecmidVol->SetFillColor(28); // Brown shade
  fleecmidVol->SetLineColor(28);

  TGeoVolume* fleeclrVol = new TGeoVolume("CarbonFleeceLeftRight", fleeclr, medCarbonFleece);
  fleeclrVol->SetFillColor(28); // Brown shade
  fleeclrVol->SetLineColor(28);

  TGeoVolume* topvVol = new TGeoVolume("TopVertex", topv, medM55J6K);
  topvVol->SetFillColor(12); // Gray shade
  topvVol->SetLineColor(12);

  TGeoVolume* sidevVol = new TGeoVolume("SideVertex", sidev, medM55J6K);
  sidevVol->SetFillColor(12); // Gray shade
  sidevVol->SetLineColor(12);

  TGeoVolume* topfilVol = new TGeoVolume("TopFilament", topfil, medM60J3K);
  topfilVol->SetFillColor(12); // Gray shade
  topfilVol->SetLineColor(12);

  TGeoVolume* endsuppVol = new TGeoVolume("EndSupport", endsupp, medM55J6K);
  endsuppVol->SetFillColor(12); // Gray shade
  endsuppVol->SetLineColor(12);

  // Now build up the half stave
  ypos = glue->GetDY();
  if (mBuildLevel < 2) // Glue
    mechStavVol->AddNode(glueVol, 1, new TGeoTranslation(0, ypos, 0));

  ypos += (glue->GetDY() + fleecbot->GetDY());
  if (mBuildLevel < 5) // Carbon
    mechStavVol->AddNode(fleecbotVol, 1, new TGeoTranslation(0, ypos, 0));

  ypos += (fleecbot->GetDY() + cfplate->GetDY());
  if (mBuildLevel < 5) // Carbon
    mechStavVol->AddNode(cfplateVol, 1, new TGeoTranslation(0, ypos, 0));

  ylay = ypos + cfplate->GetDY(); // The level where tubes etc. lay

  xpos = sIBCoolPipeXDist / 2;
  ypos = ylay + pipe->GetRmax();
  yposPipe = ypos;       // Save for later use
  if (mBuildLevel < 4) { // Kapton
    mechStavVol->AddNode(pipeVol, 1, new TGeoTranslation(-xpos, ypos, 0));
    mechStavVol->AddNode(pipeVol, 2, new TGeoTranslation(xpos, ypos, 0));
  }

  if (mBuildLevel < 3) { // Water
    mechStavVol->AddNode(waterVol, 1, new TGeoTranslation(-xpos, ypos, 0));
    mechStavVol->AddNode(waterVol, 2, new TGeoTranslation(xpos, ypos, 0));
  }

  if (mBuildLevel < 5) { // Carbon (stave components)
    mechStavVol->AddNode(cpaptubVol, 1, new TGeoTranslation(-xpos, ypos, 0));
    mechStavVol->AddNode(cpaptubVol, 2, new TGeoTranslation(xpos, ypos, 0));

    mechStavVol->AddNode(fleecpipeVol, 1, new TGeoTranslation(-xpos, ypos, 0));
    mechStavVol->AddNode(fleecpipeVol, 2, new TGeoTranslation(xpos, ypos, 0));

    xpos = sIBCoolPipeXDist / 2 - pipe->GetRmax() - cpapvert->GetDX();
    ypos = ylay + cpapvert->GetDY();
    mechStavVol->AddNode(cpapvertVol, 1, new TGeoTranslation(-xpos, ypos, 0));
    mechStavVol->AddNode(cpapvertVol, 2, new TGeoTranslation(xpos, ypos, 0));

    xpos = sIBCoolPipeXDist / 2 + pipe->GetRmax() + cpapvert->GetDX();
    mechStavVol->AddNode(cpapvertVol, 3, new TGeoTranslation(-xpos, ypos, 0));
    mechStavVol->AddNode(cpapvertVol, 4, new TGeoTranslation(xpos, ypos, 0));

    ypos = ylay + sIBCarbonPaperThick / 2;
    mechStavVol->AddNode(cpapmidVol, 1, new TGeoTranslation(0, ypos, 0));

    xpos = xstave - cpaplr->GetDX();
    mechStavVol->AddNode(cpaplrVol, 1, new TGeoTranslation(-xpos, ypos, 0));
    mechStavVol->AddNode(cpaplrVol, 2, new TGeoTranslation(xpos, ypos, 0));

    xpos = sIBCoolPipeXDist / 2 - pipe->GetRmax() - 2 * cpapvert->GetDX() - fleecvert->GetDX();
    ypos = ylay + sIBCarbonPaperThick + fleecvert->GetDY();
    mechStavVol->AddNode(fleecvertVol, 1, new TGeoTranslation(-xpos, ypos, 0));
    mechStavVol->AddNode(fleecvertVol, 2, new TGeoTranslation(xpos, ypos, 0));

    xpos = sIBCoolPipeXDist / 2 + pipe->GetRmax() + 2 * cpapvert->GetDX() + fleecvert->GetDX();
    mechStavVol->AddNode(fleecvertVol, 3, new TGeoTranslation(-xpos, ypos, 0));
    mechStavVol->AddNode(fleecvertVol, 4, new TGeoTranslation(xpos, ypos, 0));

    ypos = ylay + sIBCarbonPaperThick + sIBCarbonFleeceThick / 2;
    mechStavVol->AddNode(fleecmidVol, 1, new TGeoTranslation(0, ypos, 0));

    xpos = xstave - fleeclr->GetDX();
    mechStavVol->AddNode(fleeclrVol, 1, new TGeoTranslation(-xpos, ypos, 0));
    mechStavVol->AddNode(fleeclrVol, 2, new TGeoTranslation(xpos, ypos, 0));
  }

  ylay += (sIBCarbonPaperThick + sIBCarbonFleeceThick);

  if (mBuildLevel < 5) {                                        // Carbon (spaceframe)
    ypos = sIBStaveHeight - sIBEndSupportThick - topv->GetDz(); // Due to rotation, z is on Y
    mechStavVol->AddNode(topvVol, 1, new TGeoCombiTrans(0, ypos, 0, new TGeoRotation("", 0, -90, 0)));

    xpos = xstave - sidev->GetX(1);
    ypos = ylay;
    mechStavVol->AddNode(sidevVol, 1, new TGeoTranslation(xpos, ypos, 0));
    mechStavVol->AddNode(sidevVol, 2, new TGeoCombiTrans(-xpos, ypos, 0, new TGeoRotation("", 90, 180, -90)));

    zpos = ztot - endsupp->GetZ(1);
    mechStavVol->AddNode(endsuppVol, 1, new TGeoTranslation(0, 0, zpos));
    mechStavVol->AddNode(endsuppVol, 2, new TGeoTranslation(0, 0, -zpos));

    gamma = 180. - sIBTopFilamentAlpha;
    xpos = xstave / 2 + topfil->GetDZ();
    ypos = (endsupp->GetY(7) + endsupp->GetY(6)) / 2;
    Int_t nFilamentGroups = (Int_t)(2 * (ztot - sIBEndSupportZLen) / sIBTopFilamentInterZ);
    // theta was computed when filament volume was created
    for (int i = 0; i < nFilamentGroups; i++) { // i<19
      // 1) Front Left Top Filament
      zpos = -(ztot - sIBEndSupportZLen) + i * sIBTopFilamentInterZ + sIBTopFilamentInterZ / 4;
      mechStavVol->AddNode(topfilVol, i * 4 + 1,
                           new TGeoCombiTrans(xpos, ypos, zpos, new TGeoRotation("", 90 + theta, gamma / 2, -90)));
      // 2) Front Right Top Filament
      mechStavVol->AddNode(topfilVol, i * 4 + 2,
                           new TGeoCombiTrans(-xpos, ypos, zpos, new TGeoRotation("", 90 - theta, -gamma / 2, -90)));
      // 3) Back Left  Top Filament
      zpos += sIBTopFilamentInterZ / 2;
      mechStavVol->AddNode(topfilVol, i * 4 + 3,
                           new TGeoCombiTrans(xpos, ypos, zpos, new TGeoRotation("", 90 + theta, -gamma / 2, -90)));
      // 4) Back Right Top Filament
      mechStavVol->AddNode(topfilVol, i * 4 + 4,
                           new TGeoCombiTrans(-xpos, ypos, zpos, new TGeoRotation("", 90 - theta, gamma / 2, -90)));
    }
  }

  // Add the end-stave connectors
  TGeoVolume *connectorASide, *connectorCSide;

  // Check whether we have already all pieces
  // Otherwise create them
  connectorASide = mgr->GetVolume("IBConnectorASide");

  if (!connectorASide) {
    createIBConnectors(mgr);
    connectorASide = mgr->GetVolume("IBConnectorASide");
  }
  connectorCSide = mgr->GetVolume("IBConnectorCSide");

  ypos = (static_cast<TGeoBBox*>(connectorASide->GetShape()))->GetDY() - sIBConnTubesYPos +
         yposPipe; // We center the pipe and hole axes
  zpos = ztot + (sIBConnectBlockZLen - sIBConnTailZLen + sIBConnectAFitZOut) / 2;
  mechStavVol->AddNode(connectorASide, 1, new TGeoTranslation(0, ypos, zpos));

  zpos = ztot + (sIBConnectBlockZLen - sIBConnTailZLen) / 2;
  mechStavVol->AddNode(connectorCSide, 1, new TGeoCombiTrans(0, ypos, -zpos, new TGeoRotation("", 90, 180, -90)));

  // Done, return the stave structure
  return mechStavVol;
}

void V3Layer::createIBConnectors(const TGeoManager* mgr)
{
  //
  // Create the end-stave connectors for IB staves
  // (simply call the actual creator methods)
  //
  // Created:      20 Apr 2015  Mario Sitta
  //

  createIBConnectorsASide(mgr);
  createIBConnectorsCSide(mgr);
}

void V3Layer::createIBConnectorsASide(const TGeoManager* mgr)
{
  //
  // Create the A-Side end-stave connectors for IB staves
  //
  // Created:      22 Apr 2015  Mario Sitta
  // Updated:      04 Apr 2017  Mario Sitta  O2 version
  // Updated:      28 Jan 2018  Mario Sitta  To last drawings (ALIITSUP0051)
  // Updated:      19 Jun 2019  Mario Sitta  Avoid fake overlaps with EndWheels
  //

  // Local variables
  const Int_t nv = 8;
  Double_t xv[nv], yv[nv];
  Double_t xlen, ylen, zlen;
  Double_t xpos, ypos, zpos;

  // Gather all material pointers
  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");
  TGeoMedium* medPEEK = mgr->GetMedium("ITS_PEEKCF30$");
  TGeoMedium* medInox304 = mgr->GetMedium("ITS_INOX304$");

  // First create all elements
  // (All measures refer to the blueprint ALIITSUP0051)

  // The connector block, two Composite Shapes:
  // the body...
  xlen = sIBConnectorXWidth;
  ylen = sIBConnBodyYHeight;
  zlen = sIBConnectBlockZLen - sIBConnTailZLen;
  TGeoBBox* connBody = new TGeoBBox("connBodyA", xlen / 2, ylen / 2, zlen / 2);

  TGeoTube* connRoundHole = new TGeoTube("connRoundHoleA", 0., sIBConnRoundHoleD / 2, sIBConnBodyYHeight / 1.5);

  zpos = -connBody->GetDZ() + sIBConnRoundHoleZ;
  TGeoCombiTrans* connRoundHoleTrans =
    new TGeoCombiTrans("roundHoleTransA", 0, 0, zpos, new TGeoRotation("", 0, 90, 0));
  connRoundHoleTrans->RegisterYourself();

  xlen = sIBConnSquareHoleX / 2;
  ylen = sIBConnBodyYHeight / 1.5;
  zlen = sIBConnSquareHoleZ / 2;
  TGeoBBox* connSquareHole = new TGeoBBox("connSquareHoleA", xlen, ylen, zlen);

  zpos = -connBody->GetDZ() + sIBConnSquareHoleZPos;
  TGeoTranslation* connSquareHoleTrans = new TGeoTranslation("squareHoleTransA", 0, 0, zpos);
  connSquareHoleTrans->RegisterYourself();

  TGeoTube* connTubeHole2 = new TGeoTube("tube2HoleA", 0, sIBConnTubeHole2D / 2, connBody->GetDZ());

  xpos = sIBConnTubesXDist / 2;
  ypos = -connBody->GetDY() + sIBConnTubesYPos;

  TGeoTranslation* connTubes2Trans1 = new TGeoTranslation("tubes2Trans1A", -xpos, ypos, 0);
  connTubes2Trans1->RegisterYourself();

  TGeoTranslation* connTubes2Trans2 = new TGeoTranslation("tubes2Trans2A", xpos, ypos, 0);
  connTubes2Trans2->RegisterYourself();

  zlen = sIBConnTubeHole1ZLen - sIBConnTailZLen;
  TGeoTube* connTubeHole3 = new TGeoTube("tube3HoleA", 0, sIBConnTubeHole1D / 2, zlen);

  zpos = connBody->GetDZ();
  TGeoTranslation* connTubes3Trans1 = new TGeoTranslation("tubes3Trans1A", -xpos, ypos, -zpos);
  connTubes3Trans1->RegisterYourself();
  TGeoTranslation* connTubes3Trans2 = new TGeoTranslation("tubes3Trans2A", xpos, ypos, -zpos);
  connTubes3Trans2->RegisterYourself();

  zlen = sIBConnTubeHole1ZLen2;
  TGeoTube* connFitHole = new TGeoTube("fitHoleA", 0, sIBConnectAFitExtD / 2, zlen);

  TGeoTranslation* connFitHoleTrans1 = new TGeoTranslation("fitTrans1A", -xpos, ypos, zpos);
  connFitHoleTrans1->RegisterYourself();
  TGeoTranslation* connFitHoleTrans2 = new TGeoTranslation("fitTrans2A", xpos, ypos, zpos);
  connFitHoleTrans2->RegisterYourself();

  zlen = sIBConnSideHole1XWid / 1.5;
  TGeoTube* sideHole1 = new TGeoTube("sideHole1A", 0, sIBConnSideHole1D / 2, zlen);

  xpos = connBody->GetDX() - sIBConnSideHole1XWid + sideHole1->GetDz();
  ypos = connBody->GetDY() - sIBConnSideHole1YPos;
  zpos = -connBody->GetDZ() + (sIBConnSideHole1ZPos - sIBConnTailZLen);
  TGeoCombiTrans* connSideHole1Trans =
    new TGeoCombiTrans("sideHole1TransA", xpos, ypos, zpos, new TGeoRotation("", 90, 90, 0));
  connSideHole1Trans->RegisterYourself();

  TGeoBBox* sideHole2Box =
    new TGeoBBox("sideHole2AB", sIBConnSideHole2XWid, sIBConnSideHole2YWid / 2, sIBConnSideHole2ZWid / 2);

  xpos = -connBody->GetDX();
  ypos = connBody->GetDY() - sIBConnSideHole2YPos;
  zpos = -connBody->GetDZ() + (sIBConnSideHole2ZPos - sIBConnTailZLen) + sideHole2Box->GetDZ();
  TGeoTranslation* sideHole2BTrans = new TGeoTranslation("sideHole2TransBA", xpos, ypos, zpos);
  sideHole2BTrans->RegisterYourself();

  TGeoTubeSeg* sideHole2TubeSeg =
    new TGeoTubeSeg("sideHole2ATS", 0, sIBConnSideHole2YWid / 2, sIBConnSideHole2XWid, 0, 180);

  zpos = -connBody->GetDZ() + (sIBConnSideHole2ZPos - sIBConnTailZLen);
  TGeoCombiTrans* sideHole2TSTrans1 =
    new TGeoCombiTrans("sideHole2TSTrans1A", xpos, ypos, zpos, new TGeoRotation("", 90, -90, 0));
  sideHole2TSTrans1->RegisterYourself();

  zpos = -connBody->GetDZ() + (sIBConnSideHole2ZPos - sIBConnTailZLen) + 2 * sideHole2Box->GetDZ();
  TGeoCombiTrans* sideHole2TSTrans2 =
    new TGeoCombiTrans("sideHole2TSTrans2A", xpos, ypos, zpos, new TGeoRotation("", 90, 90, 0));
  sideHole2TSTrans2->RegisterYourself();

  TGeoCompositeShape* connBodySh = new TGeoCompositeShape(
    "connBodyA-connRoundHoleA:roundHoleTransA-connSquareHoleA:squareHoleTransA-tube2HoleA:tubes2Trans1A-tube2HoleA:"
    "tubes2Trans2A-fitHoleA:fitTrans1A-fitHoleA:fitTrans2A-tube3HoleA:tubes3Trans1A-tube3HoleA:tubes3Trans2A-"
    "sideHole1A:sideHole1TransA-sideHole2AB:sideHole2TransBA-sideHole2ATS:sideHole2TSTrans1A-sideHole2ATS:"
    "sideHole2TSTrans2A");

  TGeoVolume* connBlockBody = new TGeoVolume("IBConnectorBlockBodyASide", connBodySh, medPEEK);
  connBlockBody->SetFillColor(42); // Brownish shade
  connBlockBody->SetLineColor(42);

  // ...and the tail
  xv[0] = sIBConnectorXWidth / 2;
  yv[0] = sIBConnTailYShift;
  xv[1] = xv[0];
  yv[1] = sIBConnTailYMid;
  xv[2] = xv[1] - (sIBConnectorYTot - sIBConnTailYMid) / tanD(90 - sIBConnTailOpenPhi / 2);
  yv[2] = sIBConnectorYTot;

  for (Int_t i = 0; i < 3; i++) {
    xv[3 + i] = -xv[2 - i];
    yv[3 + i] = yv[2 - i];
  }

  TGeoXtru* connTail = new TGeoXtru(2);
  connTail->SetName("connTailA");
  connTail->DefinePolygon(6, xv, yv);
  connTail->DefineSection(0, 0);
  connTail->DefineSection(1, sIBConnTailZLen);

  TGeoTube* connTubeHole1 = new TGeoTube("tube1HoleA", 0, sIBConnTubeHole1D / 2, sIBConnTubeHole1ZLen / 1.5);

  xpos = sIBConnTubesXDist / 2;
  ypos = sIBConnTubesYPos;
  zpos = connTail->GetZ(1) / 2;
  TGeoTranslation* connTubes1Trans1 = new TGeoTranslation("tubes1Trans1A", -xpos, ypos, zpos);
  connTubes1Trans1->RegisterYourself();
  TGeoTranslation* connTubes1Trans2 = new TGeoTranslation("tubes1Trans2A", xpos, ypos, zpos);
  connTubes1Trans2->RegisterYourself();

  TGeoCompositeShape* connTailSh =
    new TGeoCompositeShape("connTailA-tube1HoleA:tubes1Trans1A-tube1HoleA:tubes1Trans2A");

  TGeoVolume* connBlockTail = new TGeoVolume("IBConnectorBlockTailASide", connTailSh, medPEEK);
  connBlockTail->SetFillColor(42); // Brownish shade
  connBlockTail->SetLineColor(42);

  // The fitting tubes, a Tube
  TGeoTube* connFitSh = new TGeoTube(sIBConnectAFitIntD / 2, sIBConnectAFitExtD / 2, sIBConnectAFitZLen / 2);

  TGeoVolume* connFit = new TGeoVolume("IBConnectorFitting", connFitSh, medInox304);
  connFit->SetFillColor(kGray);
  connFit->SetLineColor(kGray);

  // Now create the container: cannot be a simple box
  // to avoid fake overlaps with stave elements
  xlen = sIBConnectorXWidth;
  ylen = sIBConnBodyYHeight;
  zlen = sIBConnectBlockZLen - sIBConnTailZLen + sIBConnectAFitZOut;

  TGeoBBox* connBox = new TGeoBBox("connBoxA", xlen / 2, ylen / 2, zlen / 2);

  ypos = -sIBConnectorYTot / 2 + connBox->GetDY();
  TGeoTranslation* transBodyA = new TGeoTranslation("transBodyA", 0, ypos, 0);
  transBodyA->RegisterYourself();

  ypos = -sIBConnectorYTot / 2;
  zpos = -connBox->GetDZ() - connTail->GetZ(1);
  TGeoTranslation* transTailA = new TGeoTranslation("transTailA", 0, ypos, zpos);
  transTailA->RegisterYourself();

  TGeoTube* connTubeHollow = new TGeoTube("tubeHollowA", 0, sIBConnTubeHole1D / 2, sIBConnTubeHole1ZLen / 2);

  xpos = sIBConnTubesXDist / 2;
  ypos = -sIBConnectorYTot / 2 + sIBConnTubesYPos;
  zpos = -connBox->GetDZ() - connTail->GetZ(1) + sIBConnTubeHole1ZLen / 2;
  TGeoTranslation* connTubeHollTrans1 = new TGeoTranslation("tubeHollTrans1A", -xpos, ypos, zpos);
  connTubeHollTrans1->RegisterYourself();
  TGeoTranslation* connTubeHollTrans2 = new TGeoTranslation("tubeHollTrans2A", xpos, ypos, zpos);
  connTubeHollTrans2->RegisterYourself();

  zpos = -connBox->GetDZ() + connTubeHole2->GetDz() - 2 * connFitHole->GetDz();
  TGeoTranslation* connTubes2Trans1Body = new TGeoTranslation("tubes2Trans1BA", -xpos, ypos, zpos);
  connTubes2Trans1Body->RegisterYourself();
  TGeoTranslation* connTubes2Trans2Body = new TGeoTranslation("tubes2Trans2BA", xpos, ypos, zpos);
  connTubes2Trans2Body->RegisterYourself();

  TGeoCompositeShape* connBoxSh = new TGeoCompositeShape(
    "connBoxA:transBodyA-tube2HoleA:tubes2Trans1BA-tube2HoleA:tubes2Trans2BA+connTailA:transTailA-tubeHollowA:tubeHollTrans1A-"
    "tubeHollowA:tubeHollTrans2A");

  TGeoVolume* connBoxASide = new TGeoVolume("IBConnectorASide", connBoxSh, medAir);

  // Finally build up the connector
  // (NB: the origin is in the connBox, i.e. w/o the tail in Z)
  ypos = -sIBConnectorYTot / 2;
  zpos = -connBox->GetDZ() - connTail->GetZ(1);
  connBoxASide->AddNode(connBlockTail, 1, new TGeoTranslation(0, ypos, zpos));

  ypos = -sIBConnectorYTot / 2 + connBody->GetDY();
  zpos = -connBox->GetDZ() + connBody->GetDZ();
  connBoxASide->AddNode(connBlockBody, 1, new TGeoTranslation(0, ypos, zpos));

  xpos = sIBConnTubesXDist / 2;
  ypos = -sIBConnectorYTot / 2 + sIBConnTubesYPos;
  zpos = connBox->GetDZ() - connFitSh->GetDz();
  connBoxASide->AddNode(connFit, 1, new TGeoTranslation(xpos, ypos, zpos));
  connBoxASide->AddNode(connFit, 2, new TGeoTranslation(-xpos, ypos, zpos));
}

void V3Layer::createIBConnectorsCSide(const TGeoManager* mgr)
{
  //
  // Create the C-Side end-stave connectors for IB staves
  //
  // Created:      05 May 2015  Mario Sitta
  // Updated:      04 Apr 2017  Mario Sitta  O2 version
  // Updated:      28 Jan 2018  Mario Sitta  To last drawings (ALIITSUP0051)
  // Updated:      15 May 2019  Mario Sitta  Avoid fake overlaps with EndWheels
  //

  // Local variables
  const Int_t nv = 8;
  Double_t xv[nv], yv[nv];
  Double_t xlen, ylen, zlen;
  Double_t xpos, ypos, zpos;

  // Gather all material pointers
  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");
  TGeoMedium* medPEEK = mgr->GetMedium("ITS_PEEKCF30$");

  // First create all elements
  // (All measures refer to the blueprint ALIITSUP0051)

  // The connector block, two Composite Shapes:
  // the body...
  xlen = sIBConnectorXWidth;
  ylen = sIBConnBodyYHeight;
  zlen = sIBConnectBlockZLen - sIBConnTailZLen;
  TGeoBBox* connBody = new TGeoBBox("connBodyC", xlen / 2, ylen / 2, zlen / 2);

  TGeoTube* connRoundHole = new TGeoTube("connRoundHoleC", 0., sIBConnRoundHoleD / 2, sIBConnBodyYHeight / 1.5);

  zpos = -connBody->GetDZ() + sIBConnRoundHoleZ;
  TGeoCombiTrans* connRoundHoleTrans =
    new TGeoCombiTrans("roundHoleTransC", 0, 0, zpos, new TGeoRotation("", 0, 90, 0));
  connRoundHoleTrans->RegisterYourself();

  TGeoTube* connInsertHole = new TGeoTube("connInsertHoleC", 0, sIBConnInsertHoleD / 2, sIBConnBodyYHeight / 1.5);

  zpos = -connBody->GetDZ() + sIBConnInsertHoleZPos;
  TGeoCombiTrans* connInsertHoleTrans =
    new TGeoCombiTrans("insertHoleTransC", 0, 0, zpos, new TGeoRotation("", 0, 90, 0));
  connInsertHoleTrans->RegisterYourself();

  TGeoTube* connTubeHole2 = new TGeoTube("tube2HoleC", 0, sIBConnTubeHole2D / 2, connBody->GetDZ());

  xpos = sIBConnTubesXDist / 2;
  ypos = -connBody->GetDY() + sIBConnTubesYPos;
  zpos = sIBConnectBlockZLen - sIBConnTubeHole3ZPos;
  TGeoTranslation* connTubes2Trans1 = new TGeoTranslation("tubes2Trans1C", -xpos, ypos, -zpos);
  connTubes2Trans1->RegisterYourself();
  TGeoTranslation* connTubes2Trans2 = new TGeoTranslation("tubes2Trans2C", xpos, ypos, -zpos);
  connTubes2Trans2->RegisterYourself();

  zlen = sIBConnectorXWidth;
  TGeoTube* connTubeHole3 = new TGeoTube("tube3HoleC", 0, sIBConnTubeHole2D / 2, zlen / 2);

  xpos = sIBConnTubeHole3XPos;
  zpos = connBody->GetDZ() - (sIBConnectBlockZLen - sIBConnTubeHole3ZPos);
  TGeoCombiTrans* connTubes3Trans =
    new TGeoCombiTrans("tubes3TransC", xpos, ypos, zpos, new TGeoRotation("", 90, -90, 90));
  connTubes3Trans->RegisterYourself();

  zlen = sIBConnTubeHole1ZLen - sIBConnTailZLen;
  TGeoTube* connTubeHole4 = new TGeoTube("tube4HoleC", 0, sIBConnTubeHole1D / 2, zlen);

  xpos = sIBConnTubesXDist / 2;
  zpos = connBody->GetDZ();
  TGeoTranslation* connTubes4Trans1 = new TGeoTranslation("tubes4Trans1C", -xpos, ypos, -zpos);
  connTubes4Trans1->RegisterYourself();
  TGeoTranslation* connTubes4Trans2 = new TGeoTranslation("tubes4Trans2C", xpos, ypos, -zpos);
  connTubes4Trans2->RegisterYourself();

  zlen = sIBConnSideHole1XWid / 1.5;
  TGeoTube* sideHole1 = new TGeoTube("sideHole1C", 0, sIBConnSideHole1D / 2, zlen);

  xpos = -connBody->GetDX() + sIBConnSideHole1XWid - sideHole1->GetDz();
  ypos = connBody->GetDY() - sIBConnSideHole1YPos;
  zpos = -connBody->GetDZ() + (sIBConnSideHole1ZPos - sIBConnTailZLen);
  TGeoCombiTrans* connSideHole1Trans =
    new TGeoCombiTrans("sideHole1TransC", xpos, ypos, zpos, new TGeoRotation("", 90, 90, 0));
  connSideHole1Trans->RegisterYourself();

  TGeoBBox* sideHole2Box =
    new TGeoBBox("sideHole2CB", sIBConnSideHole2XWid, sIBConnSideHole2YWid / 2, sIBConnSideHole2ZWid / 2);

  xpos = connBody->GetDX();
  ypos = connBody->GetDY() - sIBConnSideHole2YPos;
  zpos = -connBody->GetDZ() + (sIBConnSideHole2ZPos - sIBConnTailZLen) + sideHole2Box->GetDZ();
  TGeoTranslation* sideHole2BTrans = new TGeoTranslation("sideHole2TransBC", xpos, ypos, zpos);
  sideHole2BTrans->RegisterYourself();

  TGeoTubeSeg* sideHole2TubeSeg =
    new TGeoTubeSeg("sideHole2CTS", 0, sIBConnSideHole2YWid / 2, sIBConnSideHole2XWid, 180, 360);

  zpos = -connBody->GetDZ() + (sIBConnSideHole2ZPos - sIBConnTailZLen);
  TGeoCombiTrans* sideHole2TSTrans1 =
    new TGeoCombiTrans("sideHole2TSTrans1C", xpos, ypos, zpos, new TGeoRotation("", -90, 90, 0));
  sideHole2TSTrans1->RegisterYourself();

  zpos = -connBody->GetDZ() + (sIBConnSideHole2ZPos - sIBConnTailZLen) + 2 * sideHole2Box->GetDZ();
  TGeoCombiTrans* sideHole2TSTrans2 =
    new TGeoCombiTrans("sideHole2TSTrans2C", xpos, ypos, zpos, new TGeoRotation("", -90, -90, 0));
  sideHole2TSTrans2->RegisterYourself();

  TGeoCompositeShape* connBodySh = new TGeoCompositeShape(
    "connBodyC-tube2HoleC:tubes2Trans1C-tube2HoleC:tubes2Trans2C-tube3HoleC:tubes3TransC-tube4HoleC:tubes4Trans1C-"
    "tube4HoleC:tubes4Trans2C-sideHole1C:sideHole1TransC-sideHole2CTS:sideHole2TSTrans1C-sideHole2CTS:"
    "sideHole2TSTrans2C-sideHole2CB:sideHole2TransBC-connRoundHoleC:roundHoleTransC-connInsertHoleC:insertHoleTransC");

  TGeoVolume* connBlockBody = new TGeoVolume("IBConnectorBlockBodyCSide", connBodySh, medPEEK);
  connBlockBody->SetFillColor(42); // Brownish shade
  connBlockBody->SetLineColor(42);

  // ...and the tail
  xv[0] = sIBConnectorXWidth / 2;
  yv[0] = sIBConnTailYShift;
  xv[1] = xv[0];
  yv[1] = sIBConnTailYMid;
  xv[2] = xv[1] - (sIBConnectorYTot - sIBConnTailYMid) / tanD(90 - sIBConnTailOpenPhi / 2);
  yv[2] = sIBConnectorYTot;

  for (Int_t i = 0; i < 3; i++) {
    xv[3 + i] = -xv[2 - i];
    yv[3 + i] = yv[2 - i];
  }

  TGeoXtru* connTail = new TGeoXtru(2);
  connTail->SetName("connTailC");
  connTail->DefinePolygon(6, xv, yv);
  connTail->DefineSection(0, 0);
  connTail->DefineSection(1, sIBConnTailZLen);

  TGeoTube* connTubeHole1 = new TGeoTube("tube1HoleC", 0, sIBConnTubeHole1D / 2, sIBConnTubeHole1ZLen / 1.5);

  xpos = sIBConnTubesXDist / 2;
  ypos = sIBConnTubesYPos;
  zpos = connTail->GetZ(1) / 2;
  TGeoTranslation* connTubes1Trans1 = new TGeoTranslation("tubes1Trans1C", -xpos, ypos, zpos);
  connTubes1Trans1->RegisterYourself();
  TGeoTranslation* connTubes1Trans2 = new TGeoTranslation("tubes1Trans2C", xpos, ypos, zpos);
  connTubes1Trans2->RegisterYourself();

  TGeoCompositeShape* connTailSh =
    new TGeoCompositeShape("connTailC-tube1HoleC:tubes1Trans1C-tube1HoleC:tubes1Trans2C");

  TGeoVolume* connBlockTail = new TGeoVolume("IBConnectorBlockTailCSide", connTailSh, medPEEK);
  connBlockTail->SetFillColor(42); // Brownish shade
  connBlockTail->SetLineColor(42);

  // The plug, a Pcon
  zlen = sIBConnPlugTotLen - sIBConnPlugInnerLen;
  TGeoPcon* connPlugSh = new TGeoPcon(0, 360, 4);
  connPlugSh->DefineSection(0, 0., 0., sIBConnTubeHole2D / 2);
  connPlugSh->DefineSection(1, zlen, 0., sIBConnTubeHole2D / 2);
  connPlugSh->DefineSection(2, zlen, sIBConnPlugInnerD / 2, sIBConnTubeHole2D / 2);
  connPlugSh->DefineSection(3, sIBConnPlugTotLen, sIBConnPlugInnerD / 2, sIBConnTubeHole2D / 2);

  TGeoVolume* connPlug = new TGeoVolume("IBConnectorPlugC", connPlugSh, medPEEK);
  connPlug->SetFillColor(44); // Brownish shade (a bit darker to spot it)
  connPlug->SetLineColor(44);

  // Now create the container: cannot be a simple box
  // to avoid fake overlaps with stave elements
  xlen = sIBConnectorXWidth;
  ylen = sIBConnBodyYHeight;
  zlen = sIBConnectBlockZLen - sIBConnTailZLen;

  TGeoBBox* connBox = new TGeoBBox("connBoxC", xlen / 2, ylen / 2, zlen / 2);

  ypos = -sIBConnectorYTot / 2 + connBox->GetDY();
  TGeoTranslation* transBodyC = new TGeoTranslation("transBodyC", 0, ypos, 0);
  transBodyC->RegisterYourself();

  ypos = -sIBConnectorYTot / 2;
  zpos = -connBox->GetDZ() - connTail->GetZ(1);
  TGeoTranslation* transTailC = new TGeoTranslation("transTailC", 0, ypos, zpos);
  transTailC->RegisterYourself();

  TGeoTube* connTubeHollow = new TGeoTube("tubeHollowC", 0, sIBConnTubeHole1D / 2, sIBConnTubeHole1ZLen / 2);

  xpos = sIBConnTubesXDist / 2;
  ypos = -sIBConnectorYTot / 2 + sIBConnTubesYPos;
  zpos = -connBox->GetDZ() - connTail->GetZ(1) + sIBConnTubeHole1ZLen / 2;
  TGeoTranslation* connTubeHollTrans1 = new TGeoTranslation("tubeHollTrans1C", -xpos, ypos, zpos);
  connTubeHollTrans1->RegisterYourself();
  TGeoTranslation* connTubeHollTrans2 = new TGeoTranslation("tubeHollTrans2C", xpos, ypos, zpos);
  connTubeHollTrans2->RegisterYourself();

  zpos = connBody->GetDZ() - (sIBConnectBlockZLen - sIBConnTubeHole3ZPos);
  TGeoTranslation* connTubes2Trans1Body = new TGeoTranslation("tubes2Trans1BC", -xpos, ypos, -zpos);
  connTubes2Trans1Body->RegisterYourself();
  TGeoTranslation* connTubes2Trans2Body = new TGeoTranslation("tubes2Trans2BC", xpos, ypos, -zpos);
  connTubes2Trans2Body->RegisterYourself();

  TGeoCompositeShape* connBoxSh = new TGeoCompositeShape(
    "connBoxC:transBodyC-tube2HoleC:tubes2Trans1BC-tube2HoleC:tubes2Trans2BC+connTailC:transTailC-tubeHollowC:tubeHollTrans1C-"
    "tubeHollowC:tubeHollTrans2C");

  TGeoVolume* connBoxCSide = new TGeoVolume("IBConnectorCSide", connBoxSh, medAir);

  // Finally build up the connector
  // (NB: the origin is in the connBox, i.e. w/o the tail in Z)
  ypos = -connBoxSh->GetDY();
  zpos = -connBodySh->GetDZ() - connTail->GetZ(1);
  connBoxCSide->AddNode(connBlockTail, 1, new TGeoTranslation(0, ypos, zpos));

  ypos = -connBoxSh->GetDY() + connBody->GetDY();
  connBoxCSide->AddNode(connBlockBody, 1, new TGeoTranslation(0, ypos, 0));

  xpos = connBox->GetDX();
  ypos = -sIBConnectorYTot / 2 + sIBConnTubesYPos;
  zpos = connBody->GetDZ() - (sIBConnectBlockZLen - sIBConnTubeHole3ZPos);
  connBoxCSide->AddNode(connPlug, 1, new TGeoCombiTrans(xpos, ypos, zpos, new TGeoRotation("", 90, -90, 90)));
}

TGeoVolume* V3Layer::createStaveOuterB(const TGeoManager* mgr)
{
  // Create the chip stave for the Outer Barrel
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
  // Updated:      19 Jul 2017  Mario Sitta  O2 version
  //
  TGeoVolume* mechStavVol = nullptr;

  switch (mStaveModel) {
    case Detector::kOBModelDummy:
      mechStavVol = createStaveModelOuterBDummy(mgr);
      break;
    case Detector::kOBModel0:
    case Detector::kOBModel1:
      LOG(FATAL) << "Stave model " << mStaveModel << " obsolete and no longer supported";
      break;
    case Detector::kOBModel2:
      mechStavVol = createStaveModelOuterB2(mgr);
      break;
    default:
      LOG(FATAL) << "Unknown stave model " << mStaveModel;
      break;
  }
  return mechStavVol;
}

TGeoVolume* V3Layer::createStaveModelOuterBDummy(const TGeoManager*) const
{
  //
  // Create dummy stave
  //
  // Input:
  //         mgr  : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      20 Dec 2013  Mario Sitta
  //

  // Done, return the stave structure
  return nullptr;
}

TGeoVolume* V3Layer::createStaveModelOuterB2(const TGeoManager* mgr)
{
  //
  // Create the mechanical half stave structure
  // for the Outer Barrel as in TDR
  //
  // Input:
  //         mgr  : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      20 Nov 2013  Anastasia Barbano
  // Updated:      16 Jan 2014  Mario Sitta
  // Updated:      24 Feb 2014  Mario Sitta
  // Updated:      11 Nov 2014  Mario Sitta  Model2
  // Updated:      03 Dec 2014  Mario Sitta  Revised with C.Gargiulo latest infos
  // Updated:      19 Jul 2017  Mario Sitta  O2 version
  // Updated:      04 Aug 2018  Mario Sitta  Updated geometry
  // Updated:      25 Aug 2018  Mario Sitta  To latest blueprints
  //

  // Local parameters
  Double_t yFlex1 = sOBFlexCableAlThick;
  Double_t yFlex2 = sOBFlexCableKapThick;
  Double_t flexOverlap = 5; // to be checked - unused for the time being
  Double_t yCFleece = sOBCarbonFleeceThick;
  Double_t yGraph = sOBGraphiteFoilThick;
  Double_t xHalfSt, yHalfSt;

  Double_t xmod, ymod, zmod, ypowbus;
  Double_t xtru[12], ytru[12];
  Double_t xpos, ypos, ypos1, zpos /*, zpos5cm*/;
  Double_t xlen, ylen, zlen;
  const Int_t nameLen = 30;
  char volname[nameLen];

  Double_t rCoolMin, rCoolMax;
  rCoolMin = sOBCoolTubeInnerD / 2;

  rCoolMax = rCoolMin + sOBCoolTubeThick;

  // First create all needed shapes

  TGeoVolume* moduleVol = createModuleOuterB();
  moduleVol->SetVisibility(kTRUE);
  xmod = (static_cast<TGeoBBox*>(moduleVol->GetShape()))->GetDX();
  ymod = (static_cast<TGeoBBox*>(moduleVol->GetShape()))->GetDY();
  zmod = (static_cast<TGeoBBox*>(moduleVol->GetShape()))->GetDZ();

  if (mLayerNumber <= 4)
    zlen = sOBColdPlateZLenML / 2; // Middle Layer
  else
    zlen = sOBColdPlateZLenOL / 2; // Outer Layer

  xlen = sOBColdPlateXWidth / 2;

  TGeoBBox* coldPlate = new TGeoBBox("ColdPlate", xlen, sOBColdPlateThick / 2, zlen);

  TGeoBBox* fleeccent = new TGeoBBox("FleeceCent", xlen, yCFleece / 2, zlen);

  TGeoTube* coolTube = new TGeoTube("CoolingTube", rCoolMin, rCoolMax, zlen);
  TGeoTube* coolWater = new TGeoTube("CoolingWater", 0., rCoolMin, zlen);

  xlen = sOBColdPlateXWidth / 2 - sOBCoolTubeXDist / 2 - coolTube->GetRmax();
  TGeoBBox* graphlat = new TGeoBBox("GraphLateral", xlen / 2, yGraph / 2, zlen);

  xlen = sOBCoolTubeXDist / 2 - coolTube->GetRmax();
  TGeoBBox* graphmid = new TGeoBBox("GraphMiddle", xlen, yGraph / 2, zlen);

  ylen = coolTube->GetRmax() - yGraph;
  TGeoBBox* graphvert = new TGeoBBox("GraphVertical", yGraph / 2, ylen / 2, zlen);

  TGeoTubeSeg* graphtub = new TGeoTubeSeg("GraphTube", rCoolMax, rCoolMax + yGraph, zlen, 180., 360.);

  xlen = sOBColdPlateXWidth / 2 - sOBCoolTubeXDist / 2 - coolTube->GetRmax() - yGraph;
  TGeoBBox* fleeclat = new TGeoBBox("FleecLateral", xlen / 2, yCFleece / 2, zlen);

  xlen = sOBCoolTubeXDist / 2 - coolTube->GetRmax() - yGraph;
  TGeoBBox* fleecmid = new TGeoBBox("FleecMiddle", xlen, yCFleece / 2, zlen);

  ylen = coolTube->GetRmax() - yGraph - yCFleece;
  TGeoBBox* fleecvert = new TGeoBBox("FleecVertical", yCFleece / 2, ylen / 2, zlen);

  TGeoTubeSeg* fleectub =
    new TGeoTubeSeg("FleecTube", rCoolMax + yGraph, rCoolMax + yCFleece + yGraph, zlen, 180., 360.);

  TGeoTube* gammaConvRod;
  if (mAddGammaConv)
    gammaConvRod = new TGeoTube("GammaConver", 0, 0.5 * mGammaConvDiam, zlen - sOBCPConnHollowZLen);

  //  TGeoBBox* flex1_5cm = new TGeoBBox("Flex1MV_5cm", xHalfSt, yFlex1 / 2, flexOverlap / 2);
  //  TGeoBBox* flex2_5cm = new TGeoBBox("Flex2MV_5cm", xHalfSt, yFlex2 / 2, flexOverlap / 2);

  // The power bus
  TGeoVolume* powerBusVol = createOBPowerBiasBuses(zlen);
  powerBusVol->SetVisibility(kTRUE);
  ypowbus = (static_cast<TGeoBBox*>(powerBusVol->GetShape()))->GetDY();

  // The half stave container (an XTru to avoid overlaps between neightbours)
  xHalfSt = xmod; // add the cross cables when done!
  yHalfSt = ypowbus + ymod + coldPlate->GetDY() + 2 * fleeccent->GetDY() + graphlat->GetDY() + fleeclat->GetDY();
  if (mAddGammaConv)
    yHalfSt += mGammaConvDiam;

  xtru[0] = xHalfSt;
  ytru[0] = 0;
  xtru[1] = xtru[0];
  ytru[1] = -2 * yHalfSt;
  xtru[2] = sOBCoolTubeXDist / 2 + fleectub->GetRmax();
  ytru[2] = ytru[1];
  xtru[3] = xtru[2];
  ytru[3] = ytru[2] - (coolTube->GetRmax() + fleectub->GetRmax());
  if (mAddGammaConv)
    ytru[3] -= mGammaConvDiam;
  xtru[4] = sOBCoolTubeXDist / 2 - fleectub->GetRmax();
  ytru[4] = ytru[3];
  xtru[5] = xtru[4];
  ytru[5] = ytru[2];
  for (Int_t i = 0; i < 6; i++) {
    xtru[6 + i] = -xtru[5 - i];
    ytru[6 + i] = ytru[5 - i];
  }
  TGeoXtru* halfStaveCent = new TGeoXtru(2);
  halfStaveCent->DefinePolygon(12, xtru, ytru);
  halfStaveCent->DefineSection(0, -zlen);
  halfStaveCent->DefineSection(1, zlen);
  snprintf(volname, nameLen, "staveCentral%d", mLayerNumber);
  halfStaveCent->SetName(volname);

  // The connectors' containers
  TGeoBBox* connAside = new TGeoBBox("connAsideOB", sOBCPConnectorXWidth / 2, sOBCPConnBlockYHei / 2,
                                     (sOBCPConnBlockZLen + sOBCPConnAFitZLen - sOBCPConnAFitZIn) / 2);

  TGeoBBox* connCside =
    new TGeoBBox("connCsideOB", sOBCPConnectorXWidth / 2, sOBCPConnBlockYHei / 2, sOBCPConnBlockZLen / 2);

  // The StaveStruct container, a Composite Shape
  if (mAddGammaConv)
    yHalfSt -= mGammaConvDiam;
  ypos = 2 * yHalfSt + connAside->GetDY() - sOBCPConnHollowYHei;
  zpos = zlen + connAside->GetDZ() - sOBCPConnHollowZLen;
  snprintf(volname, nameLen, "transAsideOB%d", mLayerNumber);
  TGeoTranslation* transAside = new TGeoTranslation(volname, 0, -ypos, zpos);
  transAside->RegisterYourself();

  zpos = zlen + connCside->GetDZ() - sOBCPConnHollowZLen;
  snprintf(volname, nameLen, "transCsideOB%d", mLayerNumber);
  TGeoTranslation* transCside = new TGeoTranslation(volname, 0, -ypos, -zpos);
  transCside->RegisterYourself();

  char componame[70];
  snprintf(componame, 70, "staveCentral%d+connAsideOB:transAsideOB%d+connCsideOB:transCsideOB%d", mLayerNumber,
           mLayerNumber, mLayerNumber);

  TGeoCompositeShape* halfStave = new TGeoCompositeShape(componame);

  // We have all shapes: now create the real volumes

  TGeoMedium* medAluminum = mgr->GetMedium("ITS_ALUMINUM$");
  TGeoMedium* medK13D2U120 = mgr->GetMedium("ITS_K13D2U120$");
  TGeoMedium* medKapton = mgr->GetMedium("ITS_KAPTON(POLYCH2)$");
  TGeoMedium* medWater = mgr->GetMedium("ITS_WATER$");
  TGeoMedium* medCarbonFleece = mgr->GetMedium("ITS_CarbonFleece$");
  TGeoMedium* medFGS003 = mgr->GetMedium("ITS_FGS003$"); // amec thermasol
  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");
  TGeoMedium* medTungsten = mgr->GetMedium("ITS_TUNGSTEN$");

  TGeoVolume* coldPlateVol = new TGeoVolume("ColdPlateVol", coldPlate, medK13D2U120);
  coldPlateVol->SetLineColor(kYellow - 3);
  coldPlateVol->SetFillColor(coldPlateVol->GetLineColor());
  coldPlateVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* fleeccentVol = new TGeoVolume("CarbonFleeceCentral", fleeccent, medCarbonFleece);
  fleeccentVol->SetLineColor(kViolet);
  fleeccentVol->SetFillColor(fleeccentVol->GetLineColor());
  fleeccentVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* coolTubeVol = new TGeoVolume("CoolingTubeVol", coolTube, medKapton);
  coolTubeVol->SetLineColor(kGray);
  coolTubeVol->SetFillColor(coolTubeVol->GetLineColor());
  coolTubeVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* coolWaterVol;
  coolWaterVol = new TGeoVolume("CoolingWaterVol", coolWater, medWater);
  coolWaterVol->SetLineColor(kBlue);
  coolWaterVol->SetFillColor(coolWaterVol->GetLineColor());
  coolWaterVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* graphlatVol = new TGeoVolume("GraphiteFoilLateral", graphlat, medFGS003);
  graphlatVol->SetLineColor(kGreen);
  graphlatVol->SetFillColor(graphlatVol->GetLineColor());
  graphlatVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* graphmidVol = new TGeoVolume("GraphiteFoilMiddle", graphmid, medFGS003);
  graphmidVol->SetLineColor(kGreen);
  graphmidVol->SetFillColor(graphmidVol->GetLineColor());
  graphmidVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* graphvertVol = new TGeoVolume("GraphiteFoilVertical", graphvert, medFGS003);
  graphvertVol->SetLineColor(kGreen);
  graphvertVol->SetFillColor(graphvertVol->GetLineColor());
  graphvertVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* graphtubVol = new TGeoVolume("GraphiteFoilPipeCover", graphtub, medFGS003);
  graphtubVol->SetLineColor(kGreen);
  graphtubVol->SetFillColor(graphtubVol->GetLineColor());
  graphtubVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* fleeclatVol = new TGeoVolume("CarbonFleeceLateral", fleeclat, medCarbonFleece);
  fleeclatVol->SetLineColor(kViolet);
  fleeclatVol->SetFillColor(fleeclatVol->GetLineColor());
  fleeclatVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* fleecmidVol = new TGeoVolume("CarbonFleeceMiddle", fleecmid, medCarbonFleece);
  fleecmidVol->SetLineColor(kViolet);
  fleecmidVol->SetFillColor(fleecmidVol->GetLineColor());
  fleecmidVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* fleecvertVol = new TGeoVolume("CarbonFleeceVertical", fleecvert, medCarbonFleece);
  fleecvertVol->SetLineColor(kViolet);
  fleecvertVol->SetFillColor(fleecvertVol->GetLineColor());
  fleecvertVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* fleectubVol = new TGeoVolume("CarbonFleecePipeCover", fleectub, medCarbonFleece);
  fleectubVol->SetLineColor(kViolet);
  fleectubVol->SetFillColor(fleectubVol->GetLineColor());
  fleectubVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* gammaConvRodVol;
  if (mAddGammaConv) {
    gammaConvRodVol = new TGeoVolume("GammaConversionRod", gammaConvRod, medTungsten);
    gammaConvRodVol->SetLineColor(kBlack);
    gammaConvRodVol->SetFillColor(gammaConvRodVol->GetLineColor());
    gammaConvRodVol->SetFillStyle(4000); // 0% transparent
  }

  snprintf(volname, nameLen, "%s%d", GeometryTGeo::getITSHalfStavePattern(), mLayerNumber);
  TGeoVolume* halfStaveVol = new TGeoVolume(volname, halfStave, medAir);
  //   halfStaveVol->SetLineColor(12);
  //   halfStaveVol->SetFillColor(12);
  //   halfStaveVol->SetVisibility(kTRUE);

  //  TGeoVolume* flex1_5cmVol = new TGeoVolume("Flex1Vol5cm", flex1_5cm, medAluminum);
  //  TGeoVolume* flex2_5cmVol = new TGeoVolume("Flex2Vol5cm", flex2_5cm, medKapton);

  //  flex1_5cmVol->SetLineColor(kRed);
  //  flex2_5cmVol->SetLineColor(kGreen);

  // Now build up the half stave
  ypos = -ypowbus;
  halfStaveVol->AddNode(powerBusVol, 1, new TGeoTranslation(0, ypos, 0));

  ypos -= (ypowbus + ymod);
  for (Int_t j = 0; j < mNumberOfModules; j++) {
    zpos = zlen - j * (2 * zmod + sOBModuleGap) - zmod;
    halfStaveVol->AddNode(moduleVol, j, new TGeoTranslation(0, ypos, zpos));
    mHierarchy[kModule]++;
  }

  ypos -= ymod;

  ypos -= fleeccent->GetDY();
  if (mBuildLevel < 6) // Carbon
    halfStaveVol->AddNode(fleeccentVol, 1, new TGeoTranslation(0, ypos, 0));
  ypos -= fleeccent->GetDY();

  ypos -= coldPlate->GetDY();
  if (mBuildLevel < 6) // Carbon
    halfStaveVol->AddNode(coldPlateVol, 1, new TGeoTranslation(0, ypos, 0));
  ypos -= coldPlate->GetDY();

  ypos -= fleeccent->GetDY();
  if (mBuildLevel < 6) // Carbon
    halfStaveVol->AddNode(fleeccentVol, 2, new TGeoTranslation(0, ypos, 0));

  xpos = sOBCoolTubeXDist / 2;
  ypos1 = ypos - (fleeccent->GetDY() + coolTube->GetRmax());
  if (mBuildLevel < 4) { // Water
    halfStaveVol->AddNode(coolWaterVol, 1, new TGeoTranslation(-xpos, ypos1, 0));
    halfStaveVol->AddNode(coolWaterVol, 2, new TGeoTranslation(xpos, ypos1, 0));
  }

  if (mBuildLevel < 5) { // Kapton
    halfStaveVol->AddNode(coolTubeVol, 1, new TGeoTranslation(-xpos, ypos1, 0));
    halfStaveVol->AddNode(coolTubeVol, 2, new TGeoTranslation(xpos, ypos1, 0));
  }

  if (mBuildLevel < 6) { // Carbon
    halfStaveVol->AddNode(graphtubVol, 1, new TGeoTranslation(-xpos, ypos1, 0));
    halfStaveVol->AddNode(graphtubVol, 2, new TGeoTranslation(xpos, ypos1, 0));

    halfStaveVol->AddNode(fleectubVol, 1, new TGeoTranslation(-xpos, ypos1, 0));
    halfStaveVol->AddNode(fleectubVol, 2, new TGeoTranslation(xpos, ypos1, 0));
  }

  xpos = sOBColdPlateXWidth / 2 - graphlat->GetDX();
  ypos1 = ypos - (fleeccent->GetDY() + graphlat->GetDY());
  if (mBuildLevel < 6) { // Carbon
    halfStaveVol->AddNode(graphlatVol, 1, new TGeoTranslation(-xpos, ypos1, 0));
    halfStaveVol->AddNode(graphlatVol, 2, new TGeoTranslation(xpos, ypos1, 0));

    halfStaveVol->AddNode(graphmidVol, 1, new TGeoTranslation(0, ypos1, 0));

    xpos = sOBColdPlateXWidth / 2 - 2 * graphlat->GetDX() + graphvert->GetDX();
    ypos1 = ypos - (fleeccent->GetDY() + 2 * graphlat->GetDY() + graphvert->GetDY());
    halfStaveVol->AddNode(graphvertVol, 1, new TGeoTranslation(-xpos, ypos1, 0));
    halfStaveVol->AddNode(graphvertVol, 2, new TGeoTranslation(xpos, ypos1, 0));
    xpos = graphmid->GetDX() - graphvert->GetDX();
    halfStaveVol->AddNode(graphvertVol, 3, new TGeoTranslation(-xpos, ypos1, 0));
    halfStaveVol->AddNode(graphvertVol, 4, new TGeoTranslation(xpos, ypos1, 0));
  }

  xpos = sOBColdPlateXWidth / 2 - fleeclat->GetDX();
  ypos1 = ypos - (fleeccent->GetDY() + 2 * graphlat->GetDY() + fleeclat->GetDY());
  if (mBuildLevel < 6) { // Carbon
    halfStaveVol->AddNode(fleeclatVol, 1, new TGeoTranslation(-xpos, ypos1, 0));
    halfStaveVol->AddNode(fleeclatVol, 2, new TGeoTranslation(xpos, ypos1, 0));

    halfStaveVol->AddNode(fleecmidVol, 1, new TGeoTranslation(0, ypos1, 0));

    xpos = sOBColdPlateXWidth / 2 - 2 * fleeclat->GetDX() + fleecvert->GetDX();
    ypos1 = ypos - (fleeccent->GetDY() + 2 * graphlat->GetDY() + 2 * fleeclat->GetDY() + fleecvert->GetDY());
    halfStaveVol->AddNode(fleecvertVol, 1, new TGeoTranslation(-xpos, ypos1, 0));
    halfStaveVol->AddNode(fleecvertVol, 2, new TGeoTranslation(xpos, ypos1, 0));
    xpos = fleecmid->GetDX() - fleecvert->GetDX();
    halfStaveVol->AddNode(fleecvertVol, 3, new TGeoTranslation(-xpos, ypos1, 0));
    halfStaveVol->AddNode(fleecvertVol, 4, new TGeoTranslation(xpos, ypos1, 0));
  }

  // Add the Gamma Converter Rod (only on Layer 3) - M.S. 17 Oct 2016
  if (mAddGammaConv) {
    xpos = mGammaConvXPos;
    ypos1 = ypos - (fleeccent->GetDY() + 2 * graphlat->GetDY() + 2 * fleeclat->GetDY() + gammaConvRod->GetRmax());
    halfStaveVol->AddNode(gammaConvRodVol, 1, new TGeoTranslation(xpos, ypos1, 0));
  }

  // Add the end-stave connectors
  TGeoVolume *connectorASide, *connectorCSide;

  // Check whether we have already all pieces
  // Otherwise create them
  connectorASide = mgr->GetVolume("OBColdPlateConnectorASide");

  if (!connectorASide) {
    createOBColdPlateConnectors();
    connectorASide = mgr->GetVolume("OBColdPlateConnectorASide");
  }
  connectorCSide = mgr->GetVolume("OBColdPlateConnectorCSide");

  ypos = 2 * yHalfSt + (static_cast<TGeoBBox*>(connectorASide->GetShape()))->GetDY() - sOBCPConnHollowYHei;
  zpos = zlen + (static_cast<TGeoBBox*>(connectorASide->GetShape()))->GetDZ() - sOBCPConnHollowZLen;
  halfStaveVol->AddNode(connectorASide, 1, new TGeoCombiTrans(0, -ypos, zpos, new TGeoRotation("", 180, 0, 0)));

  zpos = zlen + (static_cast<TGeoBBox*>(connectorCSide->GetShape()))->GetDZ() - sOBCPConnHollowZLen;
  halfStaveVol->AddNode(connectorCSide, 1, new TGeoCombiTrans(0, -ypos, -zpos, new TGeoRotation("", 180, 0, 0)));

  // Done, return the half stave structure
  return halfStaveVol;
}

TGeoVolume* V3Layer::createOBPowerBiasBuses(const Double_t zcable, const TGeoManager* mgr)
{
  //
  // Create the OB Power Bus and Bias Bus cables
  //
  // Input:
  //         zcable : the cable half Z length
  //         mgr    : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //         a TGeoVolume with both the Power and the Bias Buses
  //
  // Created:      05 Aug 2018  Mario Sitta
  // Updated:      06 Sep 2018  Mario Sitta
  //

  Double_t xcable, ytot, ypos;

  // First create all needed shapes
  xcable = sOBPowerBusXWidth / 2;
  TGeoBBox* gndPB = new TGeoBBox(xcable, sOBPowerBusAlThick / 2, zcable);
  TGeoBBox* dielPB = new TGeoBBox(xcable, sOBPowerBusDielThick / 2, zcable);
  TGeoBBox* kapPB = new TGeoBBox(xcable, sOBPowerBusKapThick / 2, zcable);
  xcable *= sOBPowerBusAlFrac;
  TGeoBBox* topPB = new TGeoBBox(xcable, sOBPowerBusAlThick / 2, zcable);

  xcable = sOBBiasBusXWidth / 2;
  TGeoBBox* botBB = new TGeoBBox(xcable, sOBBiasBusAlThick / 2, zcable);
  TGeoBBox* dielBB = new TGeoBBox(xcable, sOBBiasBusDielThick / 2, zcable);
  TGeoBBox* kapBB = new TGeoBBox(xcable, sOBBiasBusKapThick / 2, zcable);
  xcable *= sOBBiasBusAlFrac;
  TGeoBBox* topBB = new TGeoBBox(xcable, sOBBiasBusAlThick / 2, zcable);

  // Then the volumes
  TGeoMedium* medKapton = mgr->GetMedium("ITS_KAPTON(POLYCH2)$");
  TGeoMedium* medAluminum = mgr->GetMedium("ITS_ALUMINUM$");
  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");

  TGeoVolume* gndPBVol = new TGeoVolume("PowerBusGround", gndPB, medAluminum);
  gndPBVol->SetLineColor(kCyan);
  gndPBVol->SetFillColor(gndPBVol->GetLineColor());
  gndPBVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* dielPBVol = new TGeoVolume("PowerBusDielectric", dielPB, medKapton);
  dielPBVol->SetLineColor(kBlue);
  dielPBVol->SetFillColor(dielPBVol->GetLineColor());
  dielPBVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* kapPBVol = new TGeoVolume("PowerBusKapton", kapPB, medKapton);
  kapPBVol->SetLineColor(kBlue);
  kapPBVol->SetFillColor(kapPBVol->GetLineColor());
  kapPBVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* topPBVol = new TGeoVolume("PowerBusTop", topPB, medAluminum);
  topPBVol->SetLineColor(kCyan);
  topPBVol->SetFillColor(topPBVol->GetLineColor());
  topPBVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* botBBVol = new TGeoVolume("BiasBusBottom", botBB, medAluminum);
  botBBVol->SetLineColor(kCyan);
  botBBVol->SetFillColor(botBBVol->GetLineColor());
  botBBVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* dielBBVol = new TGeoVolume("BiasBusDielectric", dielBB, medKapton);
  dielBBVol->SetLineColor(kBlue);
  dielBBVol->SetFillColor(dielBBVol->GetLineColor());
  dielBBVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* kapBBVol = new TGeoVolume("BiasBusKapton", kapBB, medKapton);
  kapBBVol->SetLineColor(kBlue);
  kapBBVol->SetFillColor(kapBBVol->GetLineColor());
  kapBBVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* topBBVol = new TGeoVolume("BiasBusTop", topBB, medAluminum);
  topBBVol->SetLineColor(kCyan);
  topBBVol->SetFillColor(topBBVol->GetLineColor());
  topBBVol->SetFillStyle(4000); // 0% transparent

  // Finally the volume containing both the Power Bus and the Bias Bus
  xcable = sOBPowerBusXWidth / 2;
  ytot = 2 * kapPB->GetDY() + topPB->GetDY() + dielPB->GetDY() + gndPB->GetDY() + 2 * kapBB->GetDY() + topBB->GetDY() + dielBB->GetDY() + botBB->GetDY();

  TGeoBBox* pnbBus = new TGeoBBox(xcable, ytot, zcable);

  TGeoVolume* pnbBusVol = new TGeoVolume("OBPowerBiasBus", pnbBus, medAir);

  // Volumes are piled up from bottom to top
  ypos = -pnbBus->GetDY() + kapPB->GetDY();
  if (mBuildLevel < 5) // Kapton
    pnbBusVol->AddNode(kapPBVol, 1, new TGeoTranslation(0, ypos, 0));

  ypos += (kapPB->GetDY() + gndPB->GetDY());
  if (mBuildLevel < 2) // Aluminum
    pnbBusVol->AddNode(gndPBVol, 1, new TGeoTranslation(0, ypos, 0));

  ypos += (gndPB->GetDY() + dielPB->GetDY());
  if (mBuildLevel < 5) // Kapton
    pnbBusVol->AddNode(dielPBVol, 1, new TGeoTranslation(0, ypos, 0));

  ypos += (dielPB->GetDY() + topPB->GetDY());
  if (mBuildLevel < 2) // Aluminum
    pnbBusVol->AddNode(topPBVol, 1, new TGeoTranslation(0, ypos, 0));

  ypos += (topPB->GetDY() + kapPB->GetDY());
  if (mBuildLevel < 5) // Kapton
    pnbBusVol->AddNode(kapPBVol, 2, new TGeoTranslation(0, ypos, 0));

  //
  ypos += (kapPB->GetDY() + kapBB->GetDY());
  if (mBuildLevel < 5) // Kapton
    pnbBusVol->AddNode(kapBBVol, 1, new TGeoTranslation(0, ypos, 0));

  ypos += (kapBB->GetDY() + botBB->GetDY());
  if (mBuildLevel < 2) // Aluminum
    pnbBusVol->AddNode(botBBVol, 1, new TGeoTranslation(0, ypos, 0));

  ypos += (botBB->GetDY() + dielBB->GetDY());
  if (mBuildLevel < 5) // Kapton
    pnbBusVol->AddNode(dielBBVol, 1, new TGeoTranslation(0, ypos, 0));

  ypos += (dielBB->GetDY() + topBB->GetDY());
  if (mBuildLevel < 2) // Aluminum
    pnbBusVol->AddNode(topBBVol, 1, new TGeoTranslation(0, ypos, 0));

  ypos += (topBB->GetDY() + kapBB->GetDY());
  if (mBuildLevel < 5) // Kapton
    pnbBusVol->AddNode(kapBBVol, 2, new TGeoTranslation(0, ypos, 0));

  //
  return pnbBusVol;
}

void V3Layer::createOBColdPlateConnectors()
{
  //
  // Create the Cold Plate connectors for OB half staves
  // (simply call the actual creator methods)
  //
  // Input:
  //
  // Output:
  //
  // Return:
  //
  // Created:      26 May 2015  Mario Sitta
  //

  createOBColdPlateConnectorsASide();
  createOBColdPlateConnectorsCSide();
}

void V3Layer::createOBColdPlateConnectorsASide()
{
  //
  // Create the A-Side end-stave connectors for IB staves
  //
  // Input:
  //
  // Output:
  //
  // Return:
  //
  // Created:      26 May 2015  Mario Sitta
  // Updated:      20 Jul 2017  Mario Sitta  O2 version
  // Updated:      15 Oct 2018  Mario Sitta  To latest blueprints
  //

  // The geoManager
  const TGeoManager* mgr = gGeoManager;

  // Local variables
  const Int_t nv = 16;
  Double_t xv[nv], yv[nv];
  Double_t xlen, ylen, zlen;
  Double_t xpos, ypos, zpos;

  // Gather all material pointers
  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");
  TGeoMedium* medPEEK = mgr->GetMedium("ITS_PEEKCF30$");
  TGeoMedium* medInox304 = mgr->GetMedium("ITS_INOX304$");

  // First create all elements

  // The connector block, a Composite Shape
  xlen = sOBCPConnectorXWidth;
  ylen = sOBCPConnBlockYHei;
  zlen = sOBCPConnBlockZLen;
  TGeoBBox* connBlock = new TGeoBBox("connBlockA", xlen / 2, ylen / 2, zlen / 2);

  xv[0] = sOBCPConnectorXWidth * 0.6;
  yv[0] = -sOBCPConnHollowYHei;
  xv[1] = xv[0];
  yv[1] = sOBCPConnHollowYHei;
  xv[2] = sOBCPConnTubesXDist / 2 + sOBCPConnTubeHole1D / 2;
  yv[2] = yv[1];
  xv[3] = xv[2];
  yv[3] = sOBCPConnTubesYPos;
  xv[4] = sOBCPConnTubesXDist / 2 - sOBCPConnTubeHole1D / 2;
  yv[4] = yv[3];
  xv[5] = xv[4];
  yv[5] = yv[2];

  for (Int_t i = 0; i < 6; i++) {
    xv[6 + i] = -xv[5 - i];
    yv[6 + i] = yv[5 - i];
  }

  TGeoXtru* connBlockHoll = new TGeoXtru(2);
  connBlockHoll->SetName("connBlockHollA");
  connBlockHoll->DefinePolygon(12, xv, yv);
  connBlockHoll->DefineSection(0, -sOBCPConnHollowZLen);
  connBlockHoll->DefineSection(1, sOBCPConnHollowZLen);

  ypos = -connBlock->GetDY();
  zpos = -connBlock->GetDZ();
  TGeoTranslation* transBlockHoll = new TGeoTranslation("transBlockHollA", 0, ypos, zpos);
  transBlockHoll->RegisterYourself();

  xlen = sOBCPConnSquareHoleX / 2;
  ylen = sOBCPConnBlockYHei / 1.5;
  zlen = sOBCPConnSquareHoleZ / 2;
  TGeoBBox* connSquareHole = new TGeoBBox("connASquareHole", xlen, ylen, zlen);

  zpos =
    -connBlock->GetDZ() + (sOBCPConnSqrHoleZPos + connSquareHole->GetDZ());
  TGeoTranslation* transSquareHole = new TGeoTranslation("transASquareHole", 0, 0, zpos);
  transSquareHole->RegisterYourself();

  zlen = sOBCPConnTubeHole1Z;
  TGeoTube* connTubeHole1 = new TGeoTube("tube1AHole", 0, sOBCPConnTubeHole1D / 2, zlen);

  xpos = sOBCPConnTubesXDist / 2;
  ypos = -connBlock->GetDY() + sOBCPConnTubesYPos;
  zpos = connBlock->GetDZ();
  TGeoTranslation* trans1Tube1AHole = new TGeoTranslation("trans1Tube1AHole", -xpos, ypos, -zpos);
  trans1Tube1AHole->RegisterYourself();
  TGeoTranslation* trans2Tube1AHole = new TGeoTranslation("trans2Tube1AHole", xpos, ypos, -zpos);
  trans2Tube1AHole->RegisterYourself();

  zlen = sOBCPConnBlockZLen;
  TGeoTube* connTubeHole2 = new TGeoTube("tube2AHole", 0, sOBCPConnTubeHole2D / 2, zlen);

  TGeoTranslation* trans1Tube2AHole = new TGeoTranslation("trans1Tube2AHole", -xpos, ypos, 0);
  trans1Tube2AHole->RegisterYourself();
  TGeoTranslation* trans2Tube2AHole = new TGeoTranslation("trans2Tube2AHole", xpos, ypos, 0);
  trans2Tube2AHole->RegisterYourself();

  zlen = sOBCPConnAFitZIn;
  TGeoTube* connFitHole = new TGeoTube("fitAHole", 0, sOBCPConnFitHoleD / 2, zlen);

  TGeoTranslation* trans1FitAHole = new TGeoTranslation("trans1FitAHole", -xpos, ypos, zpos);
  trans1FitAHole->RegisterYourself();
  TGeoTranslation* trans2FitAHole = new TGeoTranslation("trans2FitAHole", xpos, ypos, zpos);
  trans2FitAHole->RegisterYourself();

  TGeoCompositeShape* connBlockSh = new TGeoCompositeShape(
    "connBlockA-connBlockHollA:transBlockHollA-connASquareHole:transASquareHole-tube1AHole:trans1Tube1AHole-tube1AHole:"
    "trans2Tube1AHole-tube2AHole:trans1Tube2AHole-tube2AHole:trans2Tube2AHole-fitAHole:trans1FitAHole-fitAHole:"
    "trans2FitAHole");

  TGeoVolume* connBlockA = new TGeoVolume("OBColdPlateConnectorBlockASide", connBlockSh, medPEEK);
  connBlockA->SetFillColor(42); // Brownish shade
  connBlockA->SetLineColor(42);

  // The fitting tubes, a Tube
  Double_t rmin = sOBCPConnAFitExtD / 2 - sOBCPConnAFitThick;
  TGeoTube* connFitSh = new TGeoTube(rmin, sOBCPConnAFitExtD / 2, sOBCPConnAFitZLen / 2);

  TGeoVolume* connFit = new TGeoVolume("OBColdPlateConnectorFitting", connFitSh, medInox304);
  connFit->SetFillColor(kGray);
  connFit->SetLineColor(kGray);

  // Now create the container: cannot be a simple box
  // to avoid fake overlaps with stave elements
  xlen = sOBCPConnectorXWidth;
  ylen = sOBCPConnBlockYHei;
  zlen = sOBCPConnBlockZLen + (sOBCPConnAFitZLen - sOBCPConnAFitZIn);
  TGeoBBox* connBox = new TGeoBBox("connectorOBCPA", xlen / 2, ylen / 2, zlen / 2);

  ypos = -connBox->GetDY();
  zpos = -connBox->GetDZ();
  TGeoTranslation* transBoxHoll = new TGeoTranslation("transBoxHollA", 0, ypos, zpos);
  transBoxHoll->RegisterYourself();

  xpos = sOBCPConnTubesXDist / 2;
  ypos = -connBox->GetDY() + sOBCPConnTubesYPos;
  zpos = connBox->GetDZ();
  TGeoTranslation* trans1BoxHole = new TGeoTranslation("trans1BoxAHole", -xpos, ypos, -zpos);
  trans1BoxHole->RegisterYourself();
  TGeoTranslation* trans2BoxHole = new TGeoTranslation("trans2BoxAHole", xpos, ypos, -zpos);
  trans2BoxHole->RegisterYourself();

  TGeoCompositeShape* connectSh = new TGeoCompositeShape(
    "connectorOBCPA-connBlockHollA:transBoxHollA-tube1AHole:trans1BoxAHole-tube1AHole:trans2BoxAHole");

  TGeoVolume* connectorASide = new TGeoVolume("OBColdPlateConnectorASide", connectSh, medAir);

  // Finally build up the connector
  zpos = -connectSh->GetDZ() + connBlock->GetDZ();
  connectorASide->AddNode(connBlockA, 1, new TGeoTranslation(0, 0, zpos));

  xpos = sOBCPConnTubesXDist / 2;
  ypos = -connBlock->GetDY() + sOBCPConnTubesYPos;
  zpos = connectSh->GetDZ() - connFitSh->GetDz();
  connectorASide->AddNode(connFit, 1, new TGeoTranslation(-xpos, ypos, zpos));
  connectorASide->AddNode(connFit, 2, new TGeoTranslation(xpos, ypos, zpos));
}

void V3Layer::createOBColdPlateConnectorsCSide()
{
  //
  // Create the C-Side end-stave connectors for IB staves
  //
  // Input:
  //
  // Output:
  //
  // Return:
  //
  // Created:      29 May 2015  Mario Sitta
  // Updated:      20 Jul 2017  Mario Sitta  O2 version
  // Updated:      15 Oct 2018  Mario Sitta  To latest blueprints
  //

  // The geoManager
  const TGeoManager* mgr = gGeoManager;

  // Local variables
  const Int_t nv = 16;
  Double_t xv[nv], yv[nv];
  Double_t xlen, ylen, zlen;
  Double_t xpos, ypos, zpos;

  // Gather all material pointers
  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");
  TGeoMedium* medPEEK = mgr->GetMedium("ITS_PEEKCF30$");

  // First create all elements

  // The connector block, a Composite Shape
  xlen = sOBCPConnectorXWidth;
  ylen = sOBCPConnBlockYHei;
  zlen = sOBCPConnBlockZLen;
  TGeoBBox* connBlock = new TGeoBBox("connBlockC", xlen / 2, ylen / 2, zlen / 2);

  xv[0] = sOBCPConnectorXWidth * 0.6;
  yv[0] = -sOBCPConnHollowYHei;
  xv[1] = xv[0];
  yv[1] = sOBCPConnHollowYHei;
  xv[2] = sOBCPConnTubesXDist / 2 + sOBCPConnTubeHole1D / 2;
  yv[2] = yv[1];
  xv[3] = xv[2];
  yv[3] = sOBCPConnTubesYPos;
  xv[4] = sOBCPConnTubesXDist / 2 - sOBCPConnTubeHole1D / 2;
  yv[4] = yv[3];
  xv[5] = xv[4];
  yv[5] = yv[2];

  for (Int_t i = 0; i < 6; i++) {
    xv[6 + i] = -xv[5 - i];
    yv[6 + i] = yv[5 - i];
  }

  TGeoXtru* connBlockHoll = new TGeoXtru(2);
  connBlockHoll->SetName("connBlockHollC");
  connBlockHoll->DefinePolygon(12, xv, yv);
  connBlockHoll->DefineSection(0, -sOBCPConnHollowZLen);
  connBlockHoll->DefineSection(1, sOBCPConnHollowZLen);

  ypos = -connBlock->GetDY();
  zpos = connBlock->GetDZ();
  TGeoTranslation* transBlockHoll = new TGeoTranslation("transBlockHollC", 0, ypos, zpos);
  transBlockHoll->RegisterYourself();

  TGeoTube* connRoundHole = new TGeoTube("connCRoundHole", 0, sOBCPConnRoundHoleD / 2, sOBCPConnBlockYHei / 1.5);

  zpos = connBlock->GetDZ() - sOBCPConnRndHoleZPos;
  TGeoCombiTrans* transRoundHole = new TGeoCombiTrans("transCRoundHole", 0, 0, zpos, new TGeoRotation("", 0, 90, 0));
  transRoundHole->RegisterYourself();

  zlen = sOBCPConnTubeHole1Z;
  TGeoTube* connTubeHole1 = new TGeoTube("tube1CHole", 0, sOBCPConnTubeHole1D / 2, zlen);

  xpos = sOBCPConnTubesXDist / 2;
  ypos = -connBlock->GetDY() + sOBCPConnTubesYPos;
  zpos = connBlock->GetDZ();
  TGeoTranslation* trans1Tube1AHole = new TGeoTranslation("trans1Tube1CHole", -xpos, ypos, zpos);
  trans1Tube1AHole->RegisterYourself();
  TGeoTranslation* trans2Tube1AHole = new TGeoTranslation("trans2Tube1CHole", xpos, ypos, zpos);
  trans2Tube1AHole->RegisterYourself();

  TGeoTube* connTubeHole2 = new TGeoTube("tube2CHole", 0, sOBCPConnTubeHole2D / 2, connBlock->GetDZ());

  zpos = sOBCPConnTubeHole3ZP;
  TGeoTranslation* connTubes2Trans1 = new TGeoTranslation("trans1Tube2CHole", -xpos, ypos, zpos);
  connTubes2Trans1->RegisterYourself();
  TGeoTranslation* connTubes2Trans2 = new TGeoTranslation("trans2Tube2CHole", xpos, ypos, zpos);
  connTubes2Trans2->RegisterYourself();

  TGeoTube* connTubeHole3 = new TGeoTube("tube3CHole", 0, sOBCPConnTubeHole2D / 2, connBlock->GetDX());

  xpos = -sOBCPConnTubeHole3XP;
  zpos = -connBlock->GetDZ() + sOBCPConnTubeHole3ZP;
  TGeoCombiTrans* connTubes3Trans =
    new TGeoCombiTrans("transTube3CHole", xpos, ypos, zpos, new TGeoRotation("", 90, -90, 90));
  connTubes3Trans->RegisterYourself();

  TGeoCompositeShape* connBlockSh = new TGeoCompositeShape(
    "connBlockC-connBlockHollC:transBlockHollC-connCRoundHole:transCRoundHole-tube1CHole:trans1Tube1CHole-tube1CHole:"
    "trans2Tube1CHole-tube2CHole:trans1Tube2CHole-tube2CHole:trans2Tube2CHole-tube3CHole:transTube3CHole");

  TGeoVolume* connBlockC = new TGeoVolume("OBColdPlateConnectorBlockCSide", connBlockSh, medPEEK);
  connBlockC->SetFillColor(42); // Brownish shade
  connBlockC->SetLineColor(42);

  // The plug, a Pcon
  TGeoPcon* connPlugSh = new TGeoPcon(0, 360, 4);
  connPlugSh->DefineSection(0, 0., 0., sOBCPConnTubeHole2D / 2);
  connPlugSh->DefineSection(1, sOBCPConnPlugThick, 0., sOBCPConnTubeHole2D / 2);
  connPlugSh->DefineSection(2, sOBCPConnPlugThick, sOBCPConnPlugInnerD / 2, sOBCPConnTubeHole2D / 2);
  connPlugSh->DefineSection(3, sOBCPConnPlugTotLen, sOBCPConnPlugInnerD / 2, sOBCPConnTubeHole2D / 2);

  TGeoVolume* connPlug = new TGeoVolume("OBCPConnectorPlugC", connPlugSh, medPEEK);
  connPlug->SetFillColor(44); // Brownish shade (a bit darker to spot it)
  connPlug->SetLineColor(44);

  // Now create the container: cannot be a simple box
  // to avoid fake overlaps with stave elements
  xlen = sOBCPConnectorXWidth;
  ylen = sOBCPConnBlockYHei;
  zlen = sOBCPConnBlockZLen;
  TGeoBBox* connBox = new TGeoBBox("connectorOBCPC", xlen / 2, ylen / 2, zlen / 2);

  ypos = -connBox->GetDY();
  zpos = connBox->GetDZ();
  TGeoTranslation* transBoxHoll = new TGeoTranslation("transBoxHollC", 0, ypos, zpos);
  transBoxHoll->RegisterYourself();

  xpos = sOBCPConnTubesXDist / 2;
  ypos = -connBox->GetDY() + sOBCPConnTubesYPos;
  zpos = connBox->GetDZ();
  TGeoTranslation* trans1BoxHole = new TGeoTranslation("trans1BoxCHole", -xpos, ypos, zpos);
  trans1BoxHole->RegisterYourself();
  TGeoTranslation* trans2BoxHole = new TGeoTranslation("trans2BoxCHole", xpos, ypos, zpos);
  trans2BoxHole->RegisterYourself();

  TGeoCompositeShape* connectSh = new TGeoCompositeShape(
    "connectorOBCPC-connBlockHollC:transBoxHollC-tube1CHole:trans1BoxCHole-tube1CHole:trans2BoxCHole");

  TGeoVolume* connectorCSide = new TGeoVolume("OBColdPlateConnectorCSide", connectSh, medAir);

  // Finally build up the connector
  connectorCSide->AddNode(connBlockC, 1);

  xpos = -connBlock->GetDX();
  ypos = -connBlock->GetDY() + sOBCPConnTubesYPos;
  zpos = -connBlock->GetDZ() + sOBCPConnTubeHole3ZP;
  connectorCSide->AddNode(connPlug, 1, new TGeoCombiTrans(xpos, ypos, zpos, new TGeoRotation("", 90, 90, 90)));
}

TGeoVolume* V3Layer::createSpaceFrameOuterB(const TGeoManager* mgr)
{
  TGeoVolume* mechStavVol = nullptr;

  switch (mStaveModel) {
    case Detector::kOBModelDummy:
    case Detector::kOBModel0:
      mechStavVol = createSpaceFrameOuterBDummy(mgr);
      break;
    case Detector::kOBModel1:
    case Detector::kOBModel2:
      mechStavVol = createSpaceFrameOuterB2(mgr);
      break;
    default:
      LOG(FATAL) << "Unknown stave model " << mStaveModel;
      break;
  }

  return mechStavVol;
}

TGeoVolume* V3Layer::createSpaceFrameOuterBDummy(const TGeoManager*) const
{
  //
  // Create dummy stave
  //
  // Input:
  //         mgr  : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //

  // Done, return the stave structur
  return nullptr;
}

TGeoVolume* V3Layer::createSpaceFrameOuterB2(const TGeoManager* mgr)
{
  //
  // Create the space frame for the Outer Barrel (Model 2)
  // The building blocks are created in another method to avoid
  // replicating the same volumes for all OB staves
  //
  // Input:
  //         mgr  : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //         a TGeoVolume with the Space Frame of a stave
  //
  // Created:      03 Feb 2015  Mario Sitta
  // Updated:      04 Jun 2015  Mario Sitta  Change container to avoid overlaps
  // Updated:      20 Jul 2017  Mario Sitta  O2 version
  //

  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");

  TGeoVolume *unitVol[2], *next2EndVol[2], *endVol[2];
  Double_t *xtru, *ytru;
  Double_t zlen, zpos;
  Int_t nPoints;
  const Int_t nameLen = 30;
  char volname[nameLen];

  // Check whether we have already all pieces
  // Otherwise create them
  unitVol[0] = mgr->GetVolume("SpaceFrameUnit0");

  if (!unitVol[0]) {
    createOBSpaceFrameObjects(mgr);
    unitVol[0] = mgr->GetVolume("SpaceFrameUnit0");
  }

  unitVol[1] = mgr->GetVolume("SpaceFrameUnit1");

  next2EndVol[0] = mgr->GetVolume("SpaceFrameNext2EndUnit0");
  next2EndVol[1] = mgr->GetVolume("SpaceFrameNext2EndUnit1");

  endVol[0] = mgr->GetVolume("SpaceFrameEndUnit0");
  endVol[1] = mgr->GetVolume("SpaceFrameEndUnit1");

  // Get the shape of the units
  // and create a similar shape for the Space Frame container
  TGeoXtru* volShape = static_cast<TGeoXtru*>(unitVol[0]->GetShape());

  nPoints = volShape->GetNvert();
  xtru = new Double_t[nPoints];
  ytru = new Double_t[nPoints];

  for (Int_t i = 0; i < nPoints; i++) {
    xtru[i] = volShape->GetX(i);
    ytru[i] = volShape->GetY(i);
  }

  Int_t nUnits = sOBSpaceFrameNUnits[mLayerNumber / 5]; // 3,4 -> 0 - 5,6 -> 1
  zlen = (nUnits - 2) * sOBSpaceFrameUnitLen;           // Take end units out

  TGeoXtru* spaceFrameCentral = new TGeoXtru(2);
  spaceFrameCentral->DefinePolygon(nPoints, xtru, ytru);
  spaceFrameCentral->DefineSection(0, -zlen / 2);
  spaceFrameCentral->DefineSection(1, zlen / 2);
  snprintf(volname, nameLen, "sframecentral%d", mLayerNumber);
  spaceFrameCentral->SetName(volname);

  zpos = zlen / 2 + sOBSpaceFrameUnitLen / 2;
  snprintf(volname, nameLen, "endUnit0Trans%d", mLayerNumber);
  TGeoCombiTrans* endUnit0Trans = new TGeoCombiTrans(volname, 0, 0, -zpos, new TGeoRotation("", 90, 180, -90));
  endUnit0Trans->RegisterYourself();
  snprintf(volname, nameLen, "endUnit1Trans%d", mLayerNumber);
  TGeoTranslation* endUnit1Trans = new TGeoTranslation(volname, 0, 0, zpos);
  endUnit1Trans->RegisterYourself();

  // The Space Frame container: a Composite Shape to avoid overlaps
  // between the U-legs space and the end-stave connectors
  // ("endunitcontainer" is defined in CreateOBSpaceFrameObjects)
  char componame[100];
  snprintf(componame, 100, "sframecentral%d+endunitcontainer:endUnit0Trans%d+endunitcontainer:endUnit1Trans%d",
           mLayerNumber, mLayerNumber, mLayerNumber);

  TGeoCompositeShape* spaceFrame = new TGeoCompositeShape(componame);

  snprintf(volname, nameLen, "SpaceFrameVolumeLay%d", mLayerNumber);
  TGeoVolume* spaceFrameVol = new TGeoVolume(volname, spaceFrame, medAir);
  spaceFrameVol->SetVisibility(kFALSE);

  // Finally build up the space frame
  TGeoXtru* frameUnit = static_cast<TGeoXtru*>(unitVol[0]->GetShape());

  zpos = -spaceFrame->GetDZ() + frameUnit->GetDZ() + sOBSFrameConnTopLen;
  spaceFrameVol->AddNode(endVol[0], 1, new TGeoCombiTrans(0, 0, zpos, new TGeoRotation("", 90, 180, -90)));

  zpos += (2 * frameUnit->GetDZ());
  spaceFrameVol->AddNode(next2EndVol[0], 1, new TGeoTranslation(0, 0, zpos));

  for (Int_t i = 2; i < nUnits - 2; i++) {
    zpos += (2 * frameUnit->GetDZ());
    Int_t j = i / 2;
    Int_t k = i - j * 2; // alternatively 0 or 1
    spaceFrameVol->AddNode(unitVol[k], j, new TGeoTranslation(0, 0, zpos));
  }

  zpos += (2 * frameUnit->GetDZ());
  spaceFrameVol->AddNode(next2EndVol[1], 1, new TGeoTranslation(0, 0, zpos));

  zpos += (2 * frameUnit->GetDZ());
  spaceFrameVol->AddNode(endVol[1], 1, new TGeoTranslation(0, 0, zpos));

  // Done, clean up and return the space frame structure
  delete[] xtru;
  delete[] ytru;

  return spaceFrameVol;
}

void V3Layer::createOBSpaceFrameObjects(const TGeoManager* mgr)
{
  //
  // Create the space frame building blocks for the Outer Barrel
  // This method is practically identical to previous versions of
  // CreateSpaceFrameOuterB1
  // NB: it is pretty cumbersome, because we don't want to use assemblies
  // so we are forced to have well-crafted containers to avoid fake overlaps
  //
  // Input:
  //         mgr  : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //         a TGeoVolume with the Space Frame of a stave
  //
  // Created:      03 Feb 2015  Mario Sitta
  // Updated:      03 Jun 2015  Mario Sitta  End units w/o U-legs
  // Updated:      20 Jul 2017  Mario Sitta  O2 version
  // Updated:      09 Sep 2019  Mario Sitta  Connectors added
  // Updated:      27 Sep 2019  Mario Sitta  New TopV for End Units
  //

  // Materials defined in AliITSUv2
  TGeoMedium* medCarbon = mgr->GetMedium("ITS_M55J6K$");
  TGeoMedium* medF6151B05M = mgr->GetMedium("ITS_F6151B05M$");
  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");

  // Local parameters
  Double_t halfFrameWidth = sOBSpaceFrameWidth / 2;
  Double_t triangleHeight = sOBSpaceFrameHeight;
  Double_t sframeHeight = triangleHeight + sOBSFrameBaseRibDiam + sOBSFrameULegHeight2 * 2;
  Double_t staveLa = sOBSpaceFrameTopVL;
  Double_t staveHa = sOBSpaceFrameTopVH;
  Double_t staveLb = sOBSpaceFrameSideVL;
  Double_t staveHb = sOBSpaceFrameSideVH;
  Double_t alphaDeg = sOBSpaceFrameVAlpha;
  Double_t alphaRad = alphaDeg * TMath::DegToRad() / 2;
  Double_t beta = sOBSpaceFrameVBeta * TMath::DegToRad() / 2;
  Double_t sideRibRadius = sOBSFrameSideRibDiam / 2;
  Double_t sidePhiDeg = sOBSFrameSideRibPhi;
  Double_t sidePhiRad = sidePhiDeg * TMath::DegToRad();
  Double_t baseRibRadius = sOBSFrameBaseRibDiam / 2;
  Double_t basePhiDeg = sOBSFrameBaseRibPhi;
  Double_t basePhiRad = basePhiDeg * TMath::DegToRad();
  Double_t ulegHalfLen = sOBSFrameULegLen / 2;
  Double_t ulegHalfWidth = sOBSFrameULegWidth / 2;
  Double_t ulegHigh1 = sOBSFrameULegHeight1;
  Double_t ulegHigh2 = sOBSFrameULegHeight2;
  Double_t ulegThick = sOBSFrameULegThick;
  Double_t topVFactorEU = 0.60; // Fraction of TopV total length for End Units

  Double_t xlen, zlen;
  Double_t xpos, ypos, zpos;
  Double_t unitlen;
  Double_t xtru[22], ytru[22];

  unitlen = sOBSpaceFrameUnitLen;

  xlen = halfFrameWidth + sideRibRadius;

  // We need a properly shaped Xtru to accomodate the ribs avoiding
  // overlaps with the HalfStave cooling tubes
  xtru[0] = sOBSFrameULegXPos - ulegHalfLen;
  ytru[0] = -(triangleHeight / 2 + baseRibRadius);
  xtru[1] = xtru[0];
  ytru[1] = ytru[0] - ulegHigh1;
  xtru[2] = xtru[1] + ulegThick;
  ytru[2] = ytru[1];
  xtru[3] = xtru[2];
  ytru[3] = ytru[0] - ulegThick;
  xtru[7] = sOBSFrameULegXPos + ulegHalfLen;
  ytru[7] = ytru[0];
  xtru[6] = xtru[7];
  ytru[6] = ytru[1];
  xtru[5] = xtru[6] - ulegThick;
  ytru[5] = ytru[6];
  xtru[4] = xtru[5];
  ytru[4] = ytru[3];
  xtru[8] = xlen;
  ytru[8] = ytru[7];
  xtru[9] = xtru[8];
  ytru[9] = 0.9 * ytru[8];
  xtru[10] = 0.3 * xtru[8];
  ytru[10] = triangleHeight / 2;
  for (Int_t i = 0; i < 11; i++) { // Reflect on the X negative side
    xtru[i + 11] = -xtru[10 - i];
    ytru[i + 11] = ytru[10 - i];
  }
  ytru[15] = ytru[0] - ulegHigh2; // U-legs on negative X are longer
  ytru[16] = ytru[15];
  ytru[19] = ytru[15];
  ytru[20] = ytru[15];

  // The space frame single units
  // We need two units because the base ribs are alternately oriented
  // The next-to-end units are slightly different
  TGeoXtru* frameUnit = new TGeoXtru(2);
  frameUnit->DefinePolygon(22, xtru, ytru);
  frameUnit->DefineSection(0, -unitlen / 2);
  frameUnit->DefineSection(1, unitlen / 2);

  TGeoXtru* next2EndUnit = new TGeoXtru(2);
  next2EndUnit->DefinePolygon(22, xtru, ytru);
  next2EndUnit->DefineSection(0, -unitlen / 2);
  next2EndUnit->DefineSection(1, unitlen / 2);

  // The end units have no U-legs, but they contain the end-stave connectors
  // so we build a CompositeShape using two Xtru's
  xtru[0] = xlen;
  ytru[0] = -(triangleHeight / 2 + baseRibRadius);
  xtru[1] = xtru[0];
  ytru[1] = 0.9 * ytru[0];
  xtru[2] = 0.3 * xtru[0];
  ytru[2] = triangleHeight / 2;
  for (Int_t i = 0; i < 3; i++) { // Reflect on the X negative side
    xtru[i + 3] = -xtru[2 - i];
    ytru[i + 3] = ytru[2 - i];
  }

  TGeoXtru* endUnitBody = new TGeoXtru(2);
  endUnitBody->SetName("endunitbody");
  endUnitBody->DefinePolygon(6, xtru, ytru);
  endUnitBody->DefineSection(0, -unitlen / 2);
  endUnitBody->DefineSection(1, 0.8 * unitlen / 2);

  xtru[2] = 0.25 * (3 * xtru[1] + xtru[2]);
  ytru[2] = 0.25 * (3 * ytru[1] + ytru[2]);
  for (Int_t i = 0; i < 3; i++) { // Reflect on the X negative side
    xtru[i + 3] = -xtru[2 - i];
    ytru[i + 3] = ytru[2 - i];
  }

  TGeoXtru* endUnitBodyLow = new TGeoXtru(2);
  endUnitBodyLow->SetName("endunitbodylow");
  endUnitBodyLow->DefinePolygon(6, xtru, ytru);
  endUnitBodyLow->DefineSection(0, 0.8 * unitlen / 2);
  endUnitBodyLow->DefineSection(1, unitlen / 2);

  // (See createOBSpaceFrameConnector lower down for details)
  xtru[0] = sOBSFrameConnWidth / 2.;
  ytru[0] = 0.;
  xtru[1] = xtru[0];
  ytru[1] = sOBSFrameConnInsHei;
  xtru[2] = xtru[1] - sOBSFrameConnTotHei + sOBSFrameConnInsHei;
  ytru[2] = sOBSFrameConnTotHei;
  for (Int_t i = 0; i < 3; i++) { // Reflect on the X negative side
    xtru[i + 3] = -xtru[2 - i];
    ytru[i + 3] = ytru[2 - i];
  }

  TGeoXtru* endUnitConn = new TGeoXtru(2);
  endUnitConn->SetName("endunitconn");
  endUnitConn->DefinePolygon(6, xtru, ytru);
  endUnitConn->DefineSection(0, 0.);
  endUnitConn->DefineSection(1, sOBSFrameConnTopLen);

  // We create a fake side V to have its dimensions, needed for
  // the creation of the end unit container
  TGeoXtru* vside =
    createStaveSide("fakeCornerSide", unitlen / 2., alphaRad, beta, staveLb, staveHb, kFALSE);

  ypos = -triangleHeight / 2 + vside->GetY(3);
  TGeoTranslation* endUnitConnTrans = new TGeoTranslation("endunitconntrans", 0, ypos, unitlen / 2);
  endUnitConnTrans->RegisterYourself();

  TGeoCompositeShape* endUnit = new TGeoCompositeShape("endunitbody+endunitbodylow+endunitconn:endunitconntrans");
  endUnit->SetName("endunitcontainer"); // Will be used when create spaceframe

  // The air containers
  TGeoVolume* unitVol[2];
  unitVol[0] = new TGeoVolume("SpaceFrameUnit0", frameUnit, medAir);
  unitVol[1] = new TGeoVolume("SpaceFrameUnit1", frameUnit, medAir);
  unitVol[0]->SetVisibility(kFALSE);
  unitVol[1]->SetVisibility(kFALSE);

  TGeoVolume* next2EndVol[2];
  next2EndVol[0] = new TGeoVolume("SpaceFrameNext2EndUnit0", next2EndUnit, medAir);
  next2EndVol[1] = new TGeoVolume("SpaceFrameNext2EndUnit1", next2EndUnit, medAir);
  next2EndVol[0]->SetVisibility(kFALSE);
  next2EndVol[1]->SetVisibility(kFALSE);

  TGeoVolume* endVol[2];
  endVol[0] = new TGeoVolume("SpaceFrameEndUnit0", endUnit, medAir);
  endVol[1] = new TGeoVolume("SpaceFrameEndUnit1", endUnit, medAir);
  endVol[0]->SetVisibility(kFALSE);
  endVol[1]->SetVisibility(kFALSE);

  // The actual volumes

  //--- The top V of the Carbon Fiber Stave (segment)
  TGeoXtru* cfStavTop =
    createStaveSide("CFstavTopCornerVolshape", unitlen / 2., alphaRad, beta, staveLa, staveHa, kTRUE);

  TGeoVolume* cfStavTopVol = new TGeoVolume("CFstavTopCornerVol", cfStavTop, medCarbon);
  cfStavTopVol->SetLineColor(35);

  unitVol[0]->AddNode(cfStavTopVol, 1, new TGeoTranslation(0, triangleHeight / 2, 0));

  unitVol[1]->AddNode(cfStavTopVol, 1, new TGeoTranslation(0, triangleHeight / 2, 0));

  next2EndVol[0]->AddNode(cfStavTopVol, 1, new TGeoTranslation(0, triangleHeight / 2, 0));

  next2EndVol[1]->AddNode(cfStavTopVol, 1, new TGeoTranslation(0, triangleHeight / 2, 0));

  zlen = topVFactorEU * unitlen;
  TGeoXtru* cfStavTopEU =
    createStaveSide("CFstavTopCornerEUVolshape", zlen / 2., alphaRad, beta, staveLa, staveHa, kTRUE);

  TGeoVolume* cfStavTopVolEU = new TGeoVolume("CFstavTopCornerEUVol", cfStavTopEU, medCarbon);
  cfStavTopVol->SetLineColor(35);

  zpos = endUnitBody->GetDZ() - zlen / 2.;

  endVol[0]->AddNode(cfStavTopVolEU, 1, new TGeoTranslation(0, triangleHeight / 2, -zpos));

  endVol[1]->AddNode(cfStavTopVolEU, 1, new TGeoTranslation(0, triangleHeight / 2, -zpos));

  //--- The two side V's
  TGeoXtru* cfStavSide =
    createStaveSide("CFstavSideCornerVolshape", unitlen / 2., alphaRad, beta, staveLb, staveHb, kFALSE);

  TGeoVolume* cfStavSideVol = new TGeoVolume("CFstavSideCornerVol", cfStavSide, medCarbon);
  cfStavSideVol->SetLineColor(35);

  unitVol[0]->AddNode(cfStavSideVol, 1, new TGeoTranslation(halfFrameWidth, -triangleHeight / 2, 0));
  unitVol[0]->AddNode(cfStavSideVol, 2,
                      new TGeoCombiTrans(-halfFrameWidth, -triangleHeight / 2, 0, new TGeoRotation("", 90, 180, -90)));

  unitVol[1]->AddNode(cfStavSideVol, 1, new TGeoTranslation(halfFrameWidth, -triangleHeight / 2, 0));
  unitVol[1]->AddNode(cfStavSideVol, 2,
                      new TGeoCombiTrans(-halfFrameWidth, -triangleHeight / 2, 0, new TGeoRotation("", 90, 180, -90)));

  next2EndVol[0]->AddNode(cfStavSideVol, 1, new TGeoTranslation(halfFrameWidth, -triangleHeight / 2, 0));
  next2EndVol[0]->AddNode(
    cfStavSideVol, 2, new TGeoCombiTrans(-halfFrameWidth, -triangleHeight / 2, 0, new TGeoRotation("", 90, 180, -90)));

  next2EndVol[1]->AddNode(cfStavSideVol, 1, new TGeoTranslation(halfFrameWidth, -triangleHeight / 2, 0));
  next2EndVol[1]->AddNode(
    cfStavSideVol, 2, new TGeoCombiTrans(-halfFrameWidth, -triangleHeight / 2, 0, new TGeoRotation("", 90, 180, -90)));

  endVol[0]->AddNode(cfStavSideVol, 1, new TGeoTranslation(halfFrameWidth, -triangleHeight / 2, 0));
  endVol[0]->AddNode(cfStavSideVol, 2,
                     new TGeoCombiTrans(-halfFrameWidth, -triangleHeight / 2, 0, new TGeoRotation("", 90, 180, -90)));

  endVol[1]->AddNode(cfStavSideVol, 1, new TGeoTranslation(halfFrameWidth, -triangleHeight / 2, 0));
  endVol[1]->AddNode(cfStavSideVol, 2,
                     new TGeoCombiTrans(-halfFrameWidth, -triangleHeight / 2, 0, new TGeoRotation("", 90, 180, -90)));

  //--- The beams
  // Ribs on the sides
  Double_t ribZProj = triangleHeight / TMath::Tan(sidePhiRad);
  Double_t sideRibLen =
    TMath::Sqrt(ribZProj * ribZProj + triangleHeight * triangleHeight + halfFrameWidth * halfFrameWidth);

  TGeoTubeSeg* sideRib = new TGeoTubeSeg(0, sideRibRadius, sideRibLen / 2, 0, 180);
  TGeoVolume* sideRibVol = new TGeoVolume("CFstavSideBeamVol", sideRib, medCarbon);
  sideRibVol->SetLineColor(35);

  TGeoCombiTrans* sideTransf[4];
  xpos = halfFrameWidth / 2 + 0.8 * staveHa * TMath::Cos(alphaRad / 2);
  ypos = -sideRibRadius / 2;
  zpos = unitlen / 4;

  sideTransf[0] = new TGeoCombiTrans(xpos, ypos, -zpos, new TGeoRotation("", 90 - alphaDeg, -sidePhiDeg, -90));
  sideTransf[1] = new TGeoCombiTrans(xpos, ypos, zpos, new TGeoRotation("", 90 - alphaDeg, sidePhiDeg, -90));
  sideTransf[2] = new TGeoCombiTrans(-xpos, ypos, -zpos, new TGeoRotation("", 90 + alphaDeg, sidePhiDeg, -90));
  sideTransf[3] = new TGeoCombiTrans(-xpos, ypos, zpos, new TGeoRotation("", 90 + alphaDeg, -sidePhiDeg, -90));

  unitVol[0]->AddNode(sideRibVol, 1, sideTransf[0]);
  unitVol[0]->AddNode(sideRibVol, 2, sideTransf[1]);
  unitVol[0]->AddNode(sideRibVol, 3, sideTransf[2]);
  unitVol[0]->AddNode(sideRibVol, 4, sideTransf[3]);

  unitVol[1]->AddNode(sideRibVol, 1, sideTransf[0]);
  unitVol[1]->AddNode(sideRibVol, 2, sideTransf[1]);
  unitVol[1]->AddNode(sideRibVol, 3, sideTransf[2]);
  unitVol[1]->AddNode(sideRibVol, 4, sideTransf[3]);

  next2EndVol[0]->AddNode(sideRibVol, 1, sideTransf[0]);
  next2EndVol[0]->AddNode(sideRibVol, 2, sideTransf[1]);
  next2EndVol[0]->AddNode(sideRibVol, 3, sideTransf[2]);
  next2EndVol[0]->AddNode(sideRibVol, 4, sideTransf[3]);

  next2EndVol[1]->AddNode(sideRibVol, 1, sideTransf[0]);
  next2EndVol[1]->AddNode(sideRibVol, 2, sideTransf[1]);
  next2EndVol[1]->AddNode(sideRibVol, 3, sideTransf[2]);
  next2EndVol[1]->AddNode(sideRibVol, 4, sideTransf[3]);

  endVol[0]->AddNode(sideRibVol, 1, sideTransf[0]);
  endVol[0]->AddNode(sideRibVol, 2, sideTransf[1]);
  endVol[0]->AddNode(sideRibVol, 3, sideTransf[2]);
  endVol[0]->AddNode(sideRibVol, 4, sideTransf[3]);

  endVol[1]->AddNode(sideRibVol, 1, sideTransf[0]);
  endVol[1]->AddNode(sideRibVol, 2, sideTransf[1]);
  endVol[1]->AddNode(sideRibVol, 3, sideTransf[2]);
  endVol[1]->AddNode(sideRibVol, 4, sideTransf[3]);

  // Ribs on the bottom
  // Rib1 are the inclined ones, Rib2 the straight ones
  Double_t baseRibLen = 0.98 * 2 * halfFrameWidth / TMath::Sin(basePhiRad);

  TGeoTubeSeg* baseRib1 = new TGeoTubeSeg(0, baseRibRadius, baseRibLen / 2, 0, 180);
  TGeoVolume* baseRib1Vol = new TGeoVolume("CFstavBaseBeam1Vol", baseRib1, medCarbon);
  baseRib1Vol->SetLineColor(35);

  TGeoTubeSeg* baseRib2 = new TGeoTubeSeg(0, baseRibRadius, halfFrameWidth, 0, 90);
  TGeoVolume* baseRib2Vol = new TGeoVolume("CFstavBaseBeam2Vol", baseRib2, medCarbon);
  baseRib2Vol->SetLineColor(35);

  TGeoTubeSeg* baseEndRib = new TGeoTubeSeg(0, baseRibRadius, halfFrameWidth, 0, 180);
  TGeoVolume* baseEndRibVol = new TGeoVolume("CFstavBaseEndBeamVol", baseEndRib, medCarbon);
  baseEndRibVol->SetLineColor(35);

  TGeoCombiTrans* baseTransf[6];
  ypos = triangleHeight / 2;
  zpos = unitlen / 2;

  baseTransf[0] = new TGeoCombiTrans("", 0, -ypos, -zpos, new TGeoRotation("", 90, 90, 90));
  baseTransf[1] = new TGeoCombiTrans("", 0, -ypos, zpos, new TGeoRotation("", -90, 90, -90));
  baseTransf[2] = new TGeoCombiTrans(0, -ypos, 0, new TGeoRotation("", -90, basePhiDeg, -90));
  baseTransf[3] = new TGeoCombiTrans(0, -ypos, 0, new TGeoRotation("", -90, -basePhiDeg, -90));
  zpos -= baseEndRib->GetRmax();
  baseTransf[4] = new TGeoCombiTrans("", 0, -ypos, -zpos, new TGeoRotation("", 90, 90, 90));
  baseTransf[5] = new TGeoCombiTrans("", 0, -ypos, zpos, new TGeoRotation("", 90, 90, 90));

  unitVol[0]->AddNode(baseRib2Vol, 1, baseTransf[0]);
  unitVol[0]->AddNode(baseRib2Vol, 2, baseTransf[1]);
  unitVol[0]->AddNode(baseRib1Vol, 1, baseTransf[2]);

  unitVol[1]->AddNode(baseRib2Vol, 1, baseTransf[0]);
  unitVol[1]->AddNode(baseRib2Vol, 2, baseTransf[1]);
  unitVol[1]->AddNode(baseRib1Vol, 1, baseTransf[3]);

  next2EndVol[0]->AddNode(baseRib2Vol, 1, baseTransf[0]);
  next2EndVol[0]->AddNode(baseRib2Vol, 2, baseTransf[1]);
  next2EndVol[0]->AddNode(baseRib1Vol, 1, baseTransf[3]);

  next2EndVol[1]->AddNode(baseRib2Vol, 1, baseTransf[0]);
  next2EndVol[1]->AddNode(baseRib2Vol, 2, baseTransf[1]);
  next2EndVol[1]->AddNode(baseRib1Vol, 1, baseTransf[3]);

  endVol[0]->AddNode(baseEndRibVol, 1, baseTransf[4]);
  endVol[0]->AddNode(baseRib2Vol, 1, baseTransf[1]);
  endVol[0]->AddNode(baseRib1Vol, 1, baseTransf[2]);

  endVol[1]->AddNode(baseEndRibVol, 1, baseTransf[5]);
  endVol[1]->AddNode(baseRib2Vol, 1, baseTransf[0]);
  endVol[1]->AddNode(baseRib1Vol, 1, baseTransf[2]);

  // The Space Frame connectors
  ypos = -triangleHeight / 2 + cfStavSide->GetY(3);
  zpos = unitlen / 2;
  createOBSpaceFrameConnector(endVol[0], ypos, zpos, kFALSE); // Side C
  createOBSpaceFrameConnector(endVol[1], ypos, zpos, kTRUE);  // Side A

  // U-Legs
  // The shorter
  xtru[0] = ulegHalfLen;
  ytru[0] = 0;
  xtru[1] = xtru[0];
  ytru[1] = -ulegHigh1;
  xtru[2] = xtru[1] - ulegThick;
  ytru[2] = ytru[1];
  xtru[3] = xtru[2];
  ytru[3] = ytru[0] - ulegThick;
  for (Int_t i = 0; i < 4; i++) { // Reflect on the X negative side
    xtru[i + 4] = -xtru[3 - i];
    ytru[i + 4] = ytru[3 - i];
  }

  TGeoXtru* uleg1full = new TGeoXtru(2); // This will go in the next end units
  uleg1full->DefinePolygon(8, xtru, ytru);
  uleg1full->DefineSection(0, -ulegHalfWidth);
  uleg1full->DefineSection(1, ulegHalfWidth);

  TGeoXtru* uleg1half = new TGeoXtru(2); // This will go in the middle unitys
  uleg1half->DefinePolygon(8, xtru, ytru);
  uleg1half->DefineSection(0, -ulegHalfWidth / 2);
  uleg1half->DefineSection(1, ulegHalfWidth / 2);

  TGeoVolume* uleg1fullVol = new TGeoVolume("CFstavULeg1FullVol", uleg1full, medF6151B05M);
  uleg1fullVol->SetLineColor(35);

  TGeoVolume* uleg1halfVol = new TGeoVolume("CFstavULeg1HalfVol", uleg1half, medF6151B05M);
  uleg1halfVol->SetLineColor(35);

  // The longer
  ytru[1] = -ulegHigh2;
  ytru[2] = -ulegHigh2;
  ytru[5] = -ulegHigh2;
  ytru[6] = -ulegHigh2;

  TGeoXtru* uleg2full = new TGeoXtru(2); // This will go in the next end units
  uleg2full->DefinePolygon(8, xtru, ytru);
  uleg2full->DefineSection(0, -ulegHalfWidth);
  uleg2full->DefineSection(1, ulegHalfWidth);

  TGeoXtru* uleg2half = new TGeoXtru(2); // This will go in the middle unitys
  uleg2half->DefinePolygon(8, xtru, ytru);
  uleg2half->DefineSection(0, -ulegHalfWidth / 2);
  uleg2half->DefineSection(1, ulegHalfWidth / 2);

  TGeoVolume* uleg2fullVol = new TGeoVolume("CFstavULeg2FullVol", uleg2full, medF6151B05M);
  uleg2fullVol->SetLineColor(35);

  TGeoVolume* uleg2halfVol = new TGeoVolume("CFstavULeg2HalfVol", uleg2half, medF6151B05M);
  uleg2halfVol->SetLineColor(35);

  xpos = sOBSFrameULegXPos;
  ypos = triangleHeight / 2 + baseRibRadius;
  zpos = unitlen / 2 - uleg1half->GetZ(1);

  unitVol[0]->AddNode(uleg1halfVol, 1, // Shorter on +X
                      new TGeoTranslation(xpos, -ypos, -zpos));
  unitVol[0]->AddNode(uleg1halfVol, 2, new TGeoTranslation(xpos, -ypos, zpos));

  unitVol[1]->AddNode(uleg1halfVol, 1, new TGeoTranslation(xpos, -ypos, -zpos));
  unitVol[1]->AddNode(uleg1halfVol, 2, new TGeoTranslation(xpos, -ypos, zpos));

  unitVol[0]->AddNode(uleg2halfVol, 1, // Longer on -X
                      new TGeoTranslation(-xpos, -ypos, -zpos));
  unitVol[0]->AddNode(uleg2halfVol, 2, new TGeoTranslation(-xpos, -ypos, zpos));

  unitVol[1]->AddNode(uleg2halfVol, 1, new TGeoTranslation(-xpos, -ypos, -zpos));
  unitVol[1]->AddNode(uleg2halfVol, 2, new TGeoTranslation(-xpos, -ypos, zpos));

  next2EndVol[0]->AddNode(uleg1halfVol, 1, new TGeoTranslation(xpos, -ypos, zpos));
  next2EndVol[0]->AddNode(uleg2halfVol, 1, new TGeoTranslation(-xpos, -ypos, zpos));

  next2EndVol[1]->AddNode(uleg1halfVol, 1, new TGeoTranslation(xpos, -ypos, -zpos));
  next2EndVol[1]->AddNode(uleg2halfVol, 1, new TGeoTranslation(-xpos, -ypos, -zpos));

  zpos = unitlen / 2 - uleg1full->GetZ(1);
  next2EndVol[0]->AddNode(uleg1fullVol, 1, new TGeoTranslation(xpos, -ypos, -zpos));
  next2EndVol[0]->AddNode(uleg2fullVol, 1, new TGeoTranslation(-xpos, -ypos, -zpos));

  next2EndVol[1]->AddNode(uleg1fullVol, 1, new TGeoTranslation(xpos, -ypos, zpos));
  next2EndVol[1]->AddNode(uleg2fullVol, 1, new TGeoTranslation(-xpos, -ypos, zpos));

  // Done
  return;
}

void V3Layer::createOBSpaceFrameConnector(TGeoVolume* mother, const Double_t ymot, const Double_t zmot, const Bool_t sideA, const TGeoManager* mgr)
{
  //
  // Creates the OB Space Frame Connectors
  // (ALIITSUP0070+ALIITSUP0069)
  //
  // Input:
  //         mother : the SF unit volume to contain the connector
  //         ymot   : the Y position of the connector in the mother volume
  //         zmot   : the Z position of the connector in the mother volume
  //         sideA  : true for Side A, false for Side C
  //         mgr    : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      09 Sep 2019  M. Sitta

  // Materials defined in AliITSUv2
  TGeoMedium* medPEEK = mgr->GetMedium("ITS_PEEKCF30$");

  // Local parameters
  TString connName, compoShape;

  Double_t xlen, ylen, zlen;
  Double_t xpos, ypos, zpos;
  Double_t xtru[6], ytru[6];

  // The external (higher) part: a Xtru
  ylen = sOBSFrameConnTotHei - sOBSFrameConnInsHei;

  xtru[0] = sOBSFrameConnWidth / 2.;
  ytru[0] = 0.;
  xtru[1] = xtru[0];
  ytru[1] = sOBSFrameConnInsHei;
  xtru[2] = xtru[1] - ylen; // Because side is at 45' so dx = dy
  ytru[2] = sOBSFrameConnTotHei;
  for (Int_t i = 0; i < 3; i++) { // Reflect on the X negative side
    xtru[i + 3] = -xtru[2 - i];
    ytru[i + 3] = ytru[2 - i];
  }

  TGeoXtru* topConn = new TGeoXtru(2);
  topConn->SetName("connectorTop");
  topConn->DefinePolygon(6, xtru, ytru);
  topConn->DefineSection(0, 0.);
  topConn->DefineSection(1, sOBSFrameConnTopLen);

  // The insert: a Xtru
  zlen = sOBSFrameConnTotLen - sOBSFrameConnTopLen;

  xtru[0] = sOBSFrameConnInsBase / 2.;
  ytru[0] = 0.;
  xtru[1] = sOBSFrameConnInsWide / 2.;
  ytru[1] = sOBSFrameConnInsHei;
  xtru[2] = -xtru[1];
  ytru[2] = ytru[1];
  xtru[3] = -xtru[0];
  ytru[3] = ytru[0];

  TGeoXtru* insConn = new TGeoXtru(2);
  insConn->SetName("connectorIns");
  insConn->DefinePolygon(4, xtru, ytru);
  insConn->DefineSection(0, -zlen);
  insConn->DefineSection(1, 0.);

  // The holes in the external (higher) part: Tube's and a BBox
  TGeoTube* topHoleR = new TGeoTube("topholer", 0., sOBSFrameConnTopHoleD / 2., 1.1 * sOBSFrameConnTotHei);

  xpos = sOBSFrConnTopHoleXDist / 2.;
  ypos = sOBSFrameConnTotHei / 2.;
  zpos = sOBSFrameConnTopLen - sOBSFrameConnHoleZPos;
  TGeoCombiTrans* topHoleR1Trans = new TGeoCombiTrans("topholer1tr", xpos, ypos, zpos, new TGeoRotation("", 0, 90, 0));
  topHoleR1Trans->RegisterYourself();

  TGeoCombiTrans* topHoleR2Trans = new TGeoCombiTrans("topholer2tr", -xpos, ypos, zpos, new TGeoRotation("", 0, 90, 0));
  topHoleR2Trans->RegisterYourself();

  xpos = sOBSFrConnCHoleXDist / 2.;
  zpos = sOBSFrameConnTopLen - sOBSFrameConnCHoleZPos;
  TGeoCombiTrans* topCHoleR1Trans = new TGeoCombiTrans("topcholer1tr", xpos, ypos, zpos, new TGeoRotation("", 0, 90, 0));
  topCHoleR1Trans->RegisterYourself();

  TGeoCombiTrans* topCHoleR2Trans = new TGeoCombiTrans("topcholer2tr", -xpos, ypos, zpos, new TGeoRotation("", 0, 90, 0));
  topCHoleR2Trans->RegisterYourself();

  TGeoBBox* topAHole = new TGeoBBox("topahole", sOBSFrameConnAHoleWid / 2., sOBSFrameConnTotHei, sOBSFrameConnAHoleLen / 2.);

  zpos = sOBSFrameConnTopLen - sOBSFrameConnHoleZPos;
  TGeoTranslation* topAHoleTrans = new TGeoTranslation("topaholetr", 0, ypos, zpos);
  topAHoleTrans->RegisterYourself();

  TGeoTube* topCHole = new TGeoTube("topchole", 0., sOBSFrConnCTopHoleD / 2., sOBSFrameConnTotHei);

  TGeoCombiTrans* topCHoleTrans = new TGeoCombiTrans("topcholetr", 0, ypos, zpos, new TGeoRotation("", 0, 90, 0));
  topCHoleTrans->RegisterYourself();

  TGeoTube* topASide = new TGeoTube("topaside", 0., sOBSFrConnASideHoleD / 2., 1.1 * sOBSFrConnASideHoleL);

  zpos = sOBSFrameConnTopLen + topASide->GetDz() - sOBSFrConnASideHoleL;
  TGeoTranslation* topASideTrans = new TGeoTranslation("topasidetr", 0, sOBSFrConnASideHoleY, zpos);
  topASideTrans->RegisterYourself();

  // The holes in the insert: a Tube
  TGeoTube* insHole = new TGeoTube("inshole", 0., sOBSFrameConnInsHoleD / 2., sOBSFrameConnInsHei);

  xpos = sOBSFrameConnInsHoleX / 2.;
  ypos = sOBSFrameConnInsHei / 2.;
  zpos = sOBSFrameConnTopLen - sOBSFrameConnHoleZPos - sOBSFrameConnHoleZDist;
  TGeoCombiTrans* insHole1Trans = new TGeoCombiTrans("inshole1tr", xpos, ypos, zpos, new TGeoRotation("", 0, 90, 0));
  insHole1Trans->RegisterYourself();

  TGeoCombiTrans* insHole2Trans = new TGeoCombiTrans("inshole2tr", -xpos, ypos, zpos, new TGeoRotation("", 0, 90, 0));
  insHole2Trans->RegisterYourself();

  // The connector: a CompositeShape
  if (sideA) {
    connName = "OBSFConnectorA";
    compoShape = "(connectorTop-topholer:topholer2tr-topholer:topholer1tr-topahole:topaholetr-topaside:topasidetr)+(connectorIns-inshole:inshole1tr-inshole:inshole2tr)";
  } else {
    connName = "OBSFConnectorC";
    compoShape = "(connectorTop-topholer:topholer2tr-topholer:topholer1tr-topholer:topcholer1tr-topholer:topcholer2tr-topchole:topcholetr)+(connectorIns-inshole:inshole1tr-inshole:inshole2tr)";
  }

  TGeoCompositeShape* obsfConnSh = new TGeoCompositeShape(compoShape.Data());

  TGeoVolume* obsfConnVol = new TGeoVolume(connName, obsfConnSh, medPEEK);

  // Finally put the connector into its mother volume
  mother->AddNode(obsfConnVol, 1, new TGeoTranslation(0, ymot, zmot));
}

TGeoVolume* V3Layer::createModuleOuterB(const TGeoManager* mgr)
{
  //
  // Creates the OB Module: HIC + FPC
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
  // Updated:      20 Jul 2017  M. Sitta  O2 version
  // Updated:      30 Jul 2018  M. Sitta  Updated geometry
  //

  const Int_t nameLen = 30;
  char chipName[nameLen], sensName[nameLen], volName[nameLen];

  Double_t xGap = sOBChipXGap;
  Double_t zGap = sOBChipZGap;

  Double_t xchip, ychip, zchip;
  Double_t xlen, ylen, zlen;
  Double_t xpos, ypos, zpos;

  Bool_t dummyChip;

  // First create all needed shapes

  // For material budget studies
  if (mBuildLevel < 7)
    dummyChip = kFALSE; // will be made of Si
  else
    dummyChip = kTRUE; // will be made of Air

  // The chip (the same as for IB)
  snprintf(chipName, nameLen, "%s%d", GeometryTGeo::getITSChipPattern(), mLayerNumber);
  snprintf(sensName, nameLen, "%s%d", GeometryTGeo::getITSSensorPattern(), mLayerNumber);

  ylen = 0.5 * sOBChipThickness;

  TGeoVolume* chipVol = AlpideChip::createChip(ylen, mSensorThickness / 2, chipName, sensName, dummyChip);

  xchip = (static_cast<TGeoBBox*>(chipVol->GetShape()))->GetDX();
  ychip = (static_cast<TGeoBBox*>(chipVol->GetShape()))->GetDY();
  zchip = (static_cast<TGeoBBox*>(chipVol->GetShape()))->GetDZ();

  mOBModuleZLength = 2 * zchip * sOBChipsPerRow + (sOBChipsPerRow - 1) * sOBChipZGap;

  zlen = mOBModuleZLength / 2;

  // The glue
  xlen = (4 * xchip + xGap) / 2;
  ylen = sOBGlueFPCThick / 2;
  TGeoBBox* glueFPC = new TGeoBBox("GlueFPC", xlen, ylen, zlen);

  ylen = sOBGlueColdPlThick / 2;
  TGeoBBox* glueCP = new TGeoBBox("GlueCP", xlen, ylen, zlen);

  // The FPC cables
  xlen = sOBFlexCableXWidth / 2;
  ylen = sOBFlexCableKapThick / 2;
  TGeoBBox* flexKap = new TGeoBBox("MidFlexKap", xlen, ylen, zlen);

  TGeoVolume* cuGndCableVol = createOBFPCCuGnd(zlen);
  TGeoVolume* cuSignalCableVol = createOBFPCCuSig(zlen);

  // The module
  Double_t ygnd = (static_cast<TGeoBBox*>(cuGndCableVol->GetShape()))->GetDY();
  Double_t ysig = (static_cast<TGeoBBox*>(cuSignalCableVol->GetShape()))->GetDY();

  xlen = (static_cast<TGeoBBox*>(cuGndCableVol->GetShape()))->GetDX();
  ylen = glueCP->GetDY() + ychip + glueFPC->GetDY() + ysig + flexKap->GetDY() + ygnd;
  TGeoBBox* module = new TGeoBBox("OBModule", xlen, ylen, zlen);

  // We have all shapes: now create the real volumes

  TGeoMedium* medAir = mgr->GetMedium("ITS_AIR$");
  TGeoMedium* medGlue = mgr->GetMedium("ITS_GLUE$");
  TGeoMedium* medKapton = mgr->GetMedium("ITS_KAPTON(POLYCH2)$");

  TGeoVolume* glueFPCVol = new TGeoVolume("GlueFPCVol", glueFPC, medGlue);
  glueFPCVol->SetLineColor(kBlack);
  glueFPCVol->SetFillColor(glueFPCVol->GetLineColor());
  glueFPCVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* glueCPVol = new TGeoVolume("GlueColdPlVol", glueCP, medGlue);
  glueCPVol->SetLineColor(kBlack);
  glueCPVol->SetFillColor(glueCPVol->GetLineColor());
  glueCPVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* flexKapVol = new TGeoVolume("FPCMidKapVol", flexKap, medKapton);
  flexKapVol->SetLineColor(kGreen);
  flexKapVol->SetFillColor(flexKapVol->GetLineColor());
  flexKapVol->SetFillStyle(4000); // 0% transparent

  snprintf(volName, nameLen, "%s%d", GeometryTGeo::getITSModulePattern(), mLayerNumber);
  TGeoVolume* modVol = new TGeoVolume(volName, module, medAir);
  modVol->SetVisibility(kTRUE);

  // Now build up the module
  ypos = -module->GetDY() + glueCP->GetDY();

  if (mBuildLevel < 3) // Glue
    modVol->AddNode(glueCPVol, 1, new TGeoTranslation(0, ypos, 0));

  xpos = xchip + xGap / 2;
  ypos += (ychip + glueCP->GetDY());
  // We use two loops here to have the same chip numbering as in HW
  //   X ^  | 6| 5| 4| 3| 2| 1| 0|
  // ----|--------------------------> Z
  //     |  | 7| 8| 9|10|11|12|13|
  //
  for (Int_t k = 0; k < sOBChipsPerRow; k++) // put first 7 chip row
  {
    zpos = module->GetDZ() - zchip - k * (2 * zchip + zGap);
    modVol->AddNode(chipVol, k, new TGeoCombiTrans(xpos, ypos, zpos, new TGeoRotation("", 0, 180, 180)));
    mHierarchy[kChip] += 1;
  }

  for (Int_t k = 0; k < sOBChipsPerRow; k++) // put second 7 chip row
  {
    zpos = -module->GetDZ() + zchip + k * (2 * zchip + zGap);
    modVol->AddNode(chipVol, k + sOBChipsPerRow, new TGeoTranslation(-xpos, ypos, zpos));
    mHierarchy[kChip] += 1;
  }

  ypos += (ychip + glueFPC->GetDY());
  if (mBuildLevel < 3) // Glue
    modVol->AddNode(glueFPCVol, 1, new TGeoTranslation(0, ypos, 0));
  ypos += glueFPC->GetDY();

  if (mBuildLevel < 5) { // Kapton
    ypos += ysig;
    modVol->AddNode(cuSignalCableVol, 1, new TGeoTranslation(0, ypos, 0));

    ypos += (ysig + flexKap->GetDY());
    modVol->AddNode(flexKapVol, 1, new TGeoTranslation(0, ypos, 0));

    ypos += (flexKap->GetDY() + ygnd);
    modVol->AddNode(cuGndCableVol, 1, new TGeoTranslation(0, ypos, 0));
  }

  // Done, return the module
  return modVol;
}

TGeoVolume* V3Layer::createOBFPCCuGnd(const Double_t zcable, const TGeoManager* mgr)
{
  //
  // Create the OB FPC Copper Ground cable
  //
  // Input:
  //         zcable : the cable half Z length
  //         mgr    : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //         the FPC cable as a TGeoVolume
  //
  // Created:      30 Jul 2018  Mario Sitta
  //

  Double_t xcable, ytot, ypos;

  // First create all needed shapes
  xcable = sOBFlexCableXWidth / 2;
  ytot = sOBFPCSoldMaskThick + sOBFPCCopperThick;
  TGeoBBox* soldmask = new TGeoBBox(xcable, ytot / 2, zcable);
  xcable *= sOBFPCCuAreaFracGnd;
  TGeoBBox* copper = new TGeoBBox(xcable, sOBFPCCopperThick / 2, zcable);

  // Then the volumes
  TGeoMedium* medKapton = mgr->GetMedium("ITS_KAPTON(POLYCH2)$");
  TGeoMedium* medCopper = mgr->GetMedium("ITS_COPPER$");

  TGeoVolume* soldmaskVol = new TGeoVolume("FPCGndSolderMask", soldmask, medKapton);
  soldmaskVol->SetLineColor(kBlue);
  soldmaskVol->SetFillColor(kBlue);

  TGeoVolume* copperVol = new TGeoVolume("FPCCopperGround", copper, medCopper);
  copperVol->SetLineColor(kCyan);
  copperVol->SetFillColor(kCyan);

  ypos = -soldmask->GetDY() + copper->GetDY();
  if (mBuildLevel < 1) // Copper
    soldmaskVol->AddNode(copperVol, 1, new TGeoTranslation(0, ypos, 0));

  return soldmaskVol;
}

TGeoVolume* V3Layer::createOBFPCCuSig(const Double_t zcable, const TGeoManager* mgr)
{
  //
  // Create the OB FPC Copper Signal cable
  //
  // Input:
  //         zcable : the cable half Z length
  //         mgr    : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //         the FPC cable as a TGeoVolume
  //
  // Created:      30 Jul 2018  Mario Sitta
  //

  Double_t xcable, ytot, ypos;

  // First create all needed shapes
  xcable = sOBFlexCableXWidth / 2;
  ytot = sOBFPCSoldMaskThick + sOBFPCCopperThick;
  TGeoBBox* soldmask = new TGeoBBox(xcable, ytot / 2, zcable);
  xcable *= sOBFPCCuAreaFracSig;
  TGeoBBox* copper = new TGeoBBox(xcable, sOBFPCCopperThick / 2, zcable);

  // Then the volumes
  TGeoMedium* medKapton = mgr->GetMedium("ITS_KAPTON(POLYCH2)$");
  TGeoMedium* medCopper = mgr->GetMedium("ITS_COPPER$");

  TGeoVolume* soldmaskVol = new TGeoVolume("FPCSigSolderMask", soldmask, medKapton);
  soldmaskVol->SetLineColor(kBlue);
  soldmaskVol->SetFillColor(kBlue);

  TGeoVolume* copperVol = new TGeoVolume("FPCCopperSignal", copper, medCopper);
  copperVol->SetLineColor(kCyan);
  copperVol->SetFillColor(kCyan);

  ypos = soldmask->GetDY() - copper->GetDY();
  if (mBuildLevel < 1) // Copper
    soldmaskVol->AddNode(copperVol, 1, new TGeoTranslation(0, ypos, 0));

  return soldmaskVol;
}

Double_t V3Layer::getGammaConversionRodDiam()
{
  //
  // Gets the diameter of the gamma conversion rods, if defined
  //
  //
  // Input:
  //
  // Output:
  //
  // Return:
  //         the diameter of the gamma conversion rods for this layer
  //
  // Created:      26 Oct 2016  Mario Sitta
  //

  if (!mAddGammaConv) {
    LOG(WARNING) << "Gamma Conversion rods not defined for this layer";
  }
  return mGammaConvDiam;
}

Double_t V3Layer::getGammaConversionRodXPos()
{
  //
  // Gets the X position of the gamma conversion rods, if defined
  //
  //
  // Input:
  //
  // Output:
  //
  // Return:
  //         the X position of the gamma conversion rods for this layer
  //         in the Half Stave reference system
  //
  // Created:      26 Oct 2016  Mario Sitta
  //

  if (!mAddGammaConv) {
    LOG(WARNING) << "Gamma Conversion rods not defined for this layer";
  }
  return mGammaConvXPos;
}

Double_t V3Layer::radiusOmTurboContainer()
{
  Double_t rr, delta, z, lstav, rstav;

  if (mChipThickness > 89.) { // Very big angle: avoid overflows since surely
    return -1;                // the radius from lower vertex is the right value
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
    LOG(ERROR) << "Not a Turbo layer";
  }
}

void V3Layer::setStaveWidth(const Double_t w)
{
  if (mIsTurbo) {
    mStaveWidth = w;
  } else {
    LOG(ERROR) << "Not a Turbo layer";
  }
}

TGeoXtru* V3Layer::createStaveSide(const char* name, Double_t dz, Double_t alpha, Double_t beta, Double_t L, Double_t H,
                                   Bool_t top)
{
  //
  // Creates the V-shaped sides of the OB space frame
  // (from a similar method with same name and function
  // in AliITSv11GeometrySDD class by L.Gaudichet)
  //
  // Updated:      15 Dec 2014  Mario Sitta  Rewritten using Xtru
  // Updated:      09 Jan 2015  Mario Sitta  Rewritten again using different
  //                                         aperture angles (info by C.Gargiulo)
  // Updated:      21 Jul 2017  Mario Sitta  O2 version
  //

  // Create the V shape corner of CF stave

  const Int_t nv = 6;
  Double_t xv[nv], yv[nv];

  TGeoXtru* cfStavSide = new TGeoXtru(2);
  cfStavSide->SetName(name);

  Double_t theta = TMath::PiOver2() - beta;
  Double_t gamma = beta - alpha;
  // Points must be in clockwise order
  if (top) { // TOP - vertices not in order
    xv[3] = 0;
    yv[3] = 0;
    xv[2] = L * TMath::Sin(alpha);
    yv[2] = -L * TMath::Cos(alpha);
    xv[1] = xv[2] - H * TMath::Cos(alpha);
    yv[1] = yv[2] - H * TMath::Sin(alpha);
    xv[0] = 0;
    yv[0] = yv[1] + TMath::Tan(theta) * xv[1];
    xv[4] = -xv[2]; // Reflect
    yv[4] = yv[2];
    xv[5] = -xv[1];
    yv[5] = yv[1];
  } else { // SIDE
    Double_t m = -TMath::Tan(alpha), n = TMath::Tan(gamma);
    xv[0] = 0;
    yv[0] = 0;
    xv[1] = -L * TMath::Cos(2 * alpha);
    yv[1] = L * TMath::Sin(2 * alpha);
    xv[2] = xv[1] - H * TMath::Sin(2 * alpha);
    yv[2] = yv[1] - H * TMath::Cos(2 * alpha);
    xv[4] = -L;
    yv[4] = H;
    xv[5] = xv[4];
    yv[5] = 0;
    xv[3] = (yv[4] - n * xv[4]) / (m - n);
    yv[3] = m * xv[3];
  }

  cfStavSide->DefinePolygon(nv, xv, yv);
  cfStavSide->DefineSection(0, -dz);
  cfStavSide->DefineSection(1, dz);

  return cfStavSide;
}

TGeoCombiTrans* V3Layer::createCombiTrans(const char* name, Double_t dy, Double_t dz, Double_t dphi, Bool_t planeSym)
{
  TGeoTranslation t1(dy * cosD(90. + dphi), dy * sinD(90. + dphi), dz);
  TGeoRotation r1("", 0., 0., dphi);
  TGeoRotation r2("", 90, 180, -90 - dphi);

  TGeoCombiTrans* combiTrans1 = new TGeoCombiTrans(name);
  combiTrans1->SetTranslation(t1);
  if (planeSym) {
    combiTrans1->SetRotation(r1);
  } else {
    combiTrans1->SetRotation(r2);
  }
  return combiTrans1;
}

void V3Layer::addTranslationToCombiTrans(TGeoCombiTrans* ct, Double_t dx, Double_t dy, Double_t dz) const
{
  // Add a dx,dy,dz translation to the initial TGeoCombiTrans
  const Double_t* vect = ct->GetTranslation();
  Double_t newVect[3] = {vect[0] + dx, vect[1] + dy, vect[2] + dz};
  ct->SetTranslation(newVect);
}
