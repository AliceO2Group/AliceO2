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

/// \file FCTLayer.cxx
/// \brief Implementation of the FCTLayer class
/// \author Mario Sitta <sitta@to.infn.it>
/// \author Chinorat Kobdaj (kobdaj@g.sut.ac.th)

#include "FCTSimulation/FCTLayer.h"
#include "FCTBase/GeometryTGeo.h"
#include "FCTSimulation/Detector.h"

#include <fairlogger/Logger.h> // for LOG

#include <TGeoManager.h> // for TGeoManager, gGeoManager
#include <TGeoMatrix.h>  // for TGeoCombiTrans, TGeoRotation, etc
#include <TGeoMedium.h>
#include <TGeoTube.h>           // for TGeoTube, TGeoTubeSeg
#include <TGeoVolume.h>         // for TGeoVolume, TGeoVolumeAssembly
#include <TGeoCompositeShape.h> // for TGeoCompositeShape
#include "TMathBase.h"          // for Abs
#include <TMath.h>              // for Sin, RadToDeg, DegToRad, Cos, Tan, etc

#include <cstdio> // for snprintf

using namespace TMath;
using namespace o2::fct;
using namespace o2::itsmft;

ClassImp(FCTLayer);

FCTLayer::~FCTLayer() = default;

FCTLayer::FCTLayer(Int_t layerNumber, std::string layerName, Float_t z, Float_t rIn, Float_t rOut_SideL, Float_t Layerx2X0, Int_t type) : mLayerNumber(layerNumber), mLayerName(layerName), mx2X0(Layerx2X0), mType(type), mInnerRadius(rIn)
{
  // Creates a simple parametrized FCT layer covering the given
  // (rIn, rOut_SideL) range at the z layer position
  mZ = -std::abs(z);
  if (type == 0) { // Disk
    mOuterRadius = rOut_SideL;
  } else if (type == 1) { // Square
    mSideLength = rOut_SideL;
  }
  Float_t Si_X0 = 9.37;   // In cm
  Float_t Pb_X0 = 0.5612; // In cm

  if (mType == 0) {
    mChipThickness = Layerx2X0 * Si_X0;
    LOG(info) << "Creating FCT Disk Layer " << mLayerNumber;
    LOG(info) << "   Using silicon X0 = " << Si_X0 << " to emulate layer radiation length.";
    LOG(info) << "   Layer z = " << mZ << " ; R_in = " << mInnerRadius << " ; R_out = " << mOuterRadius << " ; x2X0 = " << mx2X0 << " ; ChipThickness = " << mChipThickness;
  } else if (mType == 1) {
    mChipThickness = Layerx2X0 * Si_X0;
    LOG(info) << "Creating FCT Square Layer " << mLayerNumber;
    LOG(info) << "   Using silicon X0 = " << Si_X0 << " to emulate layer radiation length.";
    LOG(info) << "   Layer z = " << mZ << " ; R_in = " << mInnerRadius << " ; L_side = " << mSideLength << " ; x2X0 = " << mx2X0 << " ; ChipThickness = " << mChipThickness;
  } else if (mType == 2) {
    mChipThickness = Layerx2X0 * Pb_X0;
    LOG(info) << "Creating FCT Converter Layer " << mLayerNumber;
    LOG(info) << "   Using lead X0 = " << Pb_X0 << " to emulate layer radiation length.";
    LOG(info) << "   Layer z = " << mZ << " ; R_in = " << mInnerRadius << " ; R_out = " << mOuterRadius << " ; x2X0 = " << mx2X0 << " ; ChipThickness = " << mChipThickness;
  }
}

void FCTLayer::createLayer(TGeoVolume* motherVolume)
{
  if (mType == 0) {
    createDiskLayer(motherVolume);
  } else if (mType == 1) {
    createSquareLayer(motherVolume);
  } else if (mType == 2) {
    createConverterLayer(motherVolume);
  }
  return;
}

void FCTLayer::createDiskLayer(TGeoVolume* motherVolume)
{
  if (mLayerNumber < 0) {
    return;
  }
  // Create tube, set sensitive volume, add to mother volume

  std::string chipName = o2::fct::GeometryTGeo::getFCTChipPattern() + std::to_string(mLayerNumber),
              sensName = Form("%s_%d", GeometryTGeo::getFCTSensorPattern(), mLayerNumber);
  TGeoTube* sensor = new TGeoTube(mInnerRadius, mOuterRadius, mChipThickness / 2);
  TGeoTube* chip = new TGeoTube(mInnerRadius, mOuterRadius, mChipThickness / 2);
  TGeoTube* layer = new TGeoTube(mInnerRadius, mOuterRadius, mChipThickness / 2);

  TGeoMedium* medSi = gGeoManager->GetMedium("FCT_SI$");
  TGeoMedium* medAir = gGeoManager->GetMedium("FCT_AIR$");

  TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
  sensVol->SetLineColor(kGreen + 3);
  TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
  chipVol->SetLineColor(kGreen + 3);
  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
  layerVol->SetLineColor(kGreen + 3);

  LOG(info) << "Inserting " << sensVol->GetName() << " inside " << chipVol->GetName();
  chipVol->AddNode(sensVol, 1, nullptr);

  LOG(info) << "Inserting " << chipVol->GetName() << " inside " << layerVol->GetName();
  layerVol->AddNode(chipVol, 1, nullptr);

  // Finally put everything in the mother volume
  auto FwdDiskRotation = new TGeoRotation("FwdDiskRotation", 0, 0, 180);
  auto FwdDiskCombiTrans = new TGeoCombiTrans(0, 0, mZ, FwdDiskRotation);

  LOG(info) << "Inserting " << layerVol->GetName() << " inside " << motherVolume->GetName();
  motherVolume->AddNode(layerVol, 1, FwdDiskCombiTrans);
}

void FCTLayer::createSquareLayer(TGeoVolume* motherVolume)
{
  if (mLayerNumber < 0) {
    return;
  }

  LOG(info) << "Constructing a layer and adding it to the motherVolume";

  std::string chipName = o2::fct::GeometryTGeo::getFCTChipPattern() + std::to_string(mLayerNumber),
              sensName = Form("%s_%d", GeometryTGeo::getFCTSensorPattern(), mLayerNumber);
  TGeoBBox* sensorBox = new TGeoBBox("SensorBox", mSideLength, mSideLength, mChipThickness / 2);
  TGeoBBox* chipBox = new TGeoBBox("ChipBox", mSideLength, mSideLength, mChipThickness / 2);
  TGeoBBox* layerBox = new TGeoBBox("LayerBox", mSideLength, mSideLength, mChipThickness / 2);

  TGeoTube* sensorCutout = new TGeoTube("SensorTube", 0., mInnerRadius, mChipThickness / 2);
  TGeoTube* chipCutout = new TGeoTube("ChipTube", 0., mInnerRadius, mChipThickness / 2);
  TGeoTube* layerCutout = new TGeoTube("LayerTube", 0., mInnerRadius, mChipThickness / 2);

  TGeoCompositeShape* SensorComp = new TGeoCompositeShape("SensorComp", "SensorBox - SensorTube");
  TGeoCompositeShape* ChipComp = new TGeoCompositeShape("SensorComp", "ChipBox - ChipTube");
  TGeoCompositeShape* LayerComp = new TGeoCompositeShape("SensorComp", "LayerBox - LayerTube");

  TGeoMedium* medSi = gGeoManager->GetMedium("FCT_SI$");
  TGeoMedium* medAir = gGeoManager->GetMedium("FCT_AIR$");

  TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), SensorComp, medSi);
  TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), ChipComp, medSi);
  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), LayerComp, medAir);

  LOG(info) << "Inserting " << sensVol->GetName() << " inside " << chipVol->GetName();
  chipVol->AddNode(sensVol, 1, nullptr);

  LOG(info) << "Inserting " << chipVol->GetName() << " inside " << layerVol->GetName();
  layerVol->AddNode(chipVol, 1, nullptr);

  // Finally put everything in the mother volume
  auto FwdLayerRotation = new TGeoRotation("FwdDiskRotation", 0, 0, 180);
  auto FwdLayerCombiTrans = new TGeoCombiTrans(0, 0, mZ, FwdLayerRotation);

  LOG(info) << "Inserting " << layerVol->GetName() << " inside " << motherVolume->GetName();
  motherVolume->AddNode(layerVol, 1, FwdLayerCombiTrans);
}

void FCTLayer::createConverterLayer(TGeoVolume* motherVolume)
{
  if (mLayerNumber < 0) {
    return;
  }

  LOG(info) << "Constructing a passive converter layer and adding it to the motherVolume";
  std::string chipName = o2::fct::GeometryTGeo::getFCTChipPattern() + std::to_string(mLayerNumber),
              sensName = Form("%s_%d", GeometryTGeo::getFCTSensorPattern(), mLayerNumber);
  TGeoTube* sensor = new TGeoTube(mInnerRadius, mOuterRadius, mChipThickness / 2);
  TGeoTube* chip = new TGeoTube(mInnerRadius, mOuterRadius, mChipThickness / 2);
  TGeoTube* layer = new TGeoTube(mInnerRadius, mOuterRadius, mChipThickness / 2);

  TGeoMedium* medPb = gGeoManager->GetMedium("FCT_Pb$");
  TGeoMedium* medAir = gGeoManager->GetMedium("FCT_AIR$");

  TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medPb);
  TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medPb);
  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medPb);

  LOG(info) << "Inserting " << sensVol->GetName() << " inside " << chipVol->GetName();
  chipVol->AddNode(sensVol, 1, nullptr);

  LOG(info) << "Inserting " << chipVol->GetName() << " inside " << layerVol->GetName();
  layerVol->AddNode(chipVol, 1, nullptr);

  // Finally put everything in the mother volume
  auto FwdDiskRotation = new TGeoRotation("FwdDiskRotation", 0, 0, 180);
  auto FwdDiskCombiTrans = new TGeoCombiTrans(0, 0, mZ, FwdDiskRotation);

  LOG(info) << "Inserting " << layerVol->GetName() << " inside " << motherVolume->GetName();
  motherVolume->AddNode(layerVol, 1, FwdDiskCombiTrans);
}
