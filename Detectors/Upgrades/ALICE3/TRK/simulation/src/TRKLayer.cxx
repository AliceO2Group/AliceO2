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

#include "TRKSimulation/TRKLayer.h"
#include "TRKBase/GeometryTGeo.h"

#include "Framework/Logger.h"

#include <TGeoTube.h>
#include <TGeoBBox.h>
#include <TGeoVolume.h>

#include <TMath.h>

namespace o2
{
namespace trk
{
TRKLayer::TRKLayer(int layerNumber, std::string layerName, float rInn, float rOut, float zLength, float layerX2X0)
  : mLayerNumber(layerNumber), mLayerName(layerName), mInnerRadius(rInn), mOuterRadius(rOut), mZ(zLength), mX2X0(layerX2X0), mModuleWidth(4.54), mLayout(kCylinder)
{
  float Si_X0 = 9.5f;
  mChipThickness = mX2X0 * Si_X0;
  LOGP(info, "Creating layer: id: {} rInner: {} rOuter: {} zLength: {} x2X0: {}", mLayerNumber, mInnerRadius, mOuterRadius, mZ, mX2X0);
}

TRKLayer::TRKLayer(int layerNumber, std::string layerName, float rInn, float zLength, float thick)
  : mLayerNumber(layerNumber), mLayerName(layerName), mInnerRadius(rInn), mZ(zLength), mChipThickness(thick), mModuleWidth(4.54), mLayout(kCylinder)
{
  float Si_X0 = 9.5f;
  mOuterRadius = rInn + thick;
  mX2X0 = mChipThickness / Si_X0;
  LOGP(info, "Creating layer: id: {} rInner: {} rOuter: {} zLength: {} x2X0: {}", mLayerNumber, mInnerRadius, mOuterRadius, mZ, mX2X0);
}

TGeoVolume* TRKLayer::createSensor(std::string type, double width)
{
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  std::string sensName = GeometryTGeo::getTRKSensorPattern() + std::to_string(mLayerNumber);

  TGeoShape* sensor;

  if (type == "cylinder") {
    sensor = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2); // TO BE CHECKED !!!
  } else if (type == "flat") {
    if (width < 0) {
      LOGP(fatal, "Attempting to create sensor with invalid width");
    }
    sensor = new TGeoBBox(width / 2, mChipThickness / 2, mZ / 2); // TO BE CHECKED !!!
  } else {
    LOGP(fatal, "Sensor of type '{}' is not implemented", type);
  }

  TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
  sensVol->SetLineColor(kYellow);

  return sensVol;
};

TGeoVolume* TRKLayer::createDeadzone(std::string type, double width)
{
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  std::string deadName = GeometryTGeo::getTRKDeadzonePattern() + std::to_string(mLayerNumber);

  TGeoShape* deadzone;

  if (type == "cylinder") {
    deadzone = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2); // TO BE CHECKED !!!
  } else if (type == "flat") {
    if (width < 0) {
      LOGP(fatal, "Attempting to create deadzone with invalid width");
    }
    deadzone = new TGeoBBox(width / 2, mChipThickness / 2, mZ / 2); // TO BE CHECKED !!!
  } else {
    LOGP(fatal, "Deadzone of type '{}' is not implemented", type);
  }

  TGeoVolume* deadVol = new TGeoVolume(deadName.c_str(), deadzone, medSi);
  deadVol->SetLineColor(kGray);

  return deadVol;
};

TGeoVolume* TRKLayer::createChip(std::string type, double width)
{
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  std::string chipName = GeometryTGeo::getTRKChipPattern() + std::to_string(mLayerNumber);

  TGeoShape* chip;

  TGeoVolume* sensVol;
  TGeoVolume* deadVol;

  if (type == "cylinder") {
    chip = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2);
    sensVol = createSensor("cylinder");
    deadVol = createDeadzone("cylinder");
  } else if (type == "flat") {
    if (width < 0) {
      LOGP(fatal, "Attempting to create chip with invalid width");
    }
    chip = new TGeoBBox(width / 2, mChipThickness / 2, mZ / 2); // TO BE CHECKED !!!
    sensVol = createSensor("flat", width);
    deadVol = createDeadzone("flat", width);
  } else {
    LOGP(fatal, "Sensor of type '{}' is not implemented", type);
  }

  TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);

  LOGP(info, "Inserting {} in {} ", sensVol->GetName(), chipVol->GetName());
  chipVol->AddNode(sensVol, 1, nullptr);

  LOGP(info, "Inserting {} in {} ", deadVol->GetName(), chipVol->GetName());
  chipVol->AddNode(deadVol, 1, nullptr);

  chipVol->SetLineColor(kYellow);

  return chipVol;
}

TGeoVolume* TRKLayer::createModule(std::string type, double width)
{
  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");
  std::string moduleName = GeometryTGeo::getTRKModulePattern() + std::to_string(mLayerNumber);

  TGeoShape* module;
  TGeoVolume* moduleVol;

  if (type == "cylinder") {
    module = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2);
    moduleVol = new TGeoVolume(moduleName.c_str(), module, medAir);

    TGeoVolume* chipVol = createChip("cylinder");
    LOGP(info, "Inserting {} in {} ", chipVol->GetName(), moduleVol->GetName());
    moduleVol->AddNode(chipVol, 1, nullptr);
  } else if (type == "flat") {
    if (width < 0) {
      LOGP(fatal, "Attempting to create module with invalid width");
    }

    module = new TGeoBBox(width / 2, mChipThickness / 2, mZ / 2); // TO BE CHECKED !!!
    moduleVol = new TGeoVolume(moduleName.c_str(), module, medAir);

    int nChips = 4;

    for (int iChip = 0; iChip < nChips; iChip++) {
      TGeoVolume* chipVol = createChip("flat", mModuleWidth);

      // Put the chips in the correct position
      TGeoCombiTrans* trans = new TGeoCombiTrans();
      trans->SetTranslation(0, 0, iChip * (mModuleWidth + 0.1)); // TO BE CHECKED !!!

      LOGP(info, "Inserting {} in {} ", chipVol->GetName(), moduleVol->GetName());
      moduleVol->AddNode(chipVol, iChip, trans);
    }
  } else {
    LOGP(fatal, "Chip of type '{}' is not implemented", type);
  }

  moduleVol->SetLineColor(kYellow);

  return moduleVol;
}

TGeoVolume* TRKLayer::createStave(std::string type, double width)
{
  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");
  std::string staveName = GeometryTGeo::getTRKStavePattern() + std::to_string(mLayerNumber);

  TGeoShape* stave;
  TGeoVolume* staveVol;

  if (type == "cylinder") {
    stave = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2);
    staveVol = new TGeoVolume(staveName.c_str(), stave, medAir);

    TGeoVolume* moduleVol = createModule("cylinder");
    LOGP(info, "Inserting {} in {} ", moduleVol->GetName(), staveVol->GetName());
    staveVol->AddNode(moduleVol, 1, nullptr);
  } else if (type == "flat") {
    if (width < 0) {
      LOGP(fatal, "Attempting to create stave with invalid width");
    }

    stave = new TGeoBBox(width / 2, mChipThickness / 2, mZ / 2);
    staveVol = new TGeoVolume(staveName.c_str(), stave, medAir);

    int nModules = 10;

    for (int iModule = 0; iModule < nModules; iModule++) {
      TGeoVolume* moduleVol = createModule("flat", mModuleWidth);

      // Put the modules in the correct position
      TGeoCombiTrans* trans = new TGeoCombiTrans();
      trans->SetTranslation(0, 0, iModule * (mModuleWidth + 0.1)); // TO BE CHECKED !!!

      LOGP(info, "Inserting {} in {} ", moduleVol->GetName(), staveVol->GetName());
      staveVol->AddNode(moduleVol, iModule, trans);
    }
  } else if (type == "staggered") {
    double width = mModuleWidth * 2; // Each stave has two modules (based on the LOI design)
    stave = new TGeoBBox(width / 2, mLogicalVolumeThickness / 2, mZ / 2);
    staveVol = new TGeoVolume(staveName.c_str(), stave, medAir);

    int nModules = 10;

    for (int iModule = 0; iModule < nModules; iModule++) {
      TGeoVolume* moduleVolLeft = createModule("flat", mModuleWidth);
      TGeoVolume* moduleVolRight = createModule("flat", mModuleWidth);

      // Put the modules in the correct position
      TGeoCombiTrans* transLeft = new TGeoCombiTrans();
      transLeft->SetTranslation(-mModuleWidth / 2 + 0.05, 0, iModule * (mModuleWidth + 0.1)); // TO BE CHECKED !!! 1mm overlap between the modules
      LOGP(info, "Inserting {} in {} ", moduleVolLeft->GetName(), staveVol->GetName());
      staveVol->AddNode(moduleVolLeft, iModule * 2, transLeft);

      TGeoCombiTrans* transRight = new TGeoCombiTrans();
      transRight->SetTranslation(mModuleWidth / 2 - 0.05, 0, iModule * (mModuleWidth + 0.1)); // TO BE CHECKED !!! 1mm overlap between the modules
      LOGP(info, "Inserting {} in {} ", moduleVolRight->GetName(), staveVol->GetName());
      staveVol->AddNode(moduleVolRight, iModule * 2 + 1, transRight);
    }
  } else {
    LOGP(fatal, "Chip of type '{}' is not implemented", type);
  }

  staveVol->SetLineColor(kYellow);

  return staveVol;
}

void TRKLayer::createLayer(TGeoVolume* motherVolume)
{
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");

  std::string staveName = GeometryTGeo::getTRKStavePattern() + std::to_string(mLayerNumber),
              chipName = GeometryTGeo::getTRKChipPattern() + std::to_string(mLayerNumber),
              sensName = GeometryTGeo::getTRKSensorPattern() + std::to_string(mLayerNumber);

  double layerThickness = mChipThickness;
  if (mLayout != eLayout::kCylinder) {
    layerThickness = mLogicalVolumeThickness;
  }
  TGeoTube* layer = new TGeoTube(mInnerRadius - 0.333 * layerThickness, mInnerRadius + 0.667 * layerThickness, mZ / 2);

  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
  layerVol->SetLineColor(kYellow);

  if (mLayout == eLayout::kCylinder) {
    TGeoVolume* staveVol = createStave("cylinder");
    LOGP(info, "Inserting {} in {} ", staveVol->GetName(), layerVol->GetName());
    layerVol->AddNode(staveVol, 1, nullptr);
  } else if (mLayout == eLayout::kTurboStaves) {
    // Compute the number of staves
    double width = mModuleWidth; // Each stave has two modules (based on the LOI design)
    if (mInnerRadius > 25) {
      width *= 2; // Outer layers have two modules per stave
    }
    int nStaves = (int)std::ceil(mInnerRadius * 2 * TMath::Pi() / width);
    nStaves += nStaves % 2; // Require an even number of staves

    // Compute the size of the overlap region
    double theta = 2 * TMath::Pi() / nStaves;
    double theta1 = std::atan(width / 2 / mInnerRadius);
    double st = std::sin(theta);
    double ct = std::cos(theta);
    double theta2 = std::atan((mInnerRadius * st - width / 2 * ct) / (mInnerRadius * ct + width / 2 * st));
    double overlap = (theta1 - theta2) * mInnerRadius;
    LOGP(info, "Creating a layer with {} staves and {} mm overlap", nStaves, overlap * 10);

    for (int iStave = 0; iStave < nStaves; iStave++) {
      TGeoVolume* staveVol = createStave("flat", width);

      // Put the staves in the correct position and orientation
      TGeoCombiTrans* trans = new TGeoCombiTrans();
      double theta = 360. * iStave / nStaves;
      TGeoRotation* rot = new TGeoRotation("rot", theta + 90 + 3, 0, 0);
      trans->SetRotation(rot);
      trans->SetTranslation(mInnerRadius * std::cos(2. * TMath::Pi() * iStave / nStaves), mInnerRadius * std::sin(2 * TMath::Pi() * iStave / nStaves), 0);

      LOGP(info, "Inserting {} in {} ", staveVol->GetName(), layerVol->GetName());
      layerVol->AddNode(staveVol, iStave, trans);
    }
  } else if (mLayout == kStaggered) {
    // Compute the number of staves
    double width = mModuleWidth * 2; // Each stave has two modules (based on the LOI design)
    int nStaves = (int)std::ceil(mInnerRadius * 2 * TMath::Pi() / width);
    nStaves += nStaves % 2; // Require an even number of staves

    // Compute the size of the overlap region
    double theta = 2 * TMath::Pi() / nStaves;
    double theta1 = std::atan(width / 2 / mInnerRadius);
    double st = std::sin(theta);
    double ct = std::cos(theta);
    double theta2 = std::atan((mInnerRadius * st - width / 2 * ct) / (mInnerRadius * ct + width / 2 * st));
    double overlap = (theta1 - theta2) * mInnerRadius;
    LOGP(info, "Creating a layer with {} staves and {} mm overlap", nStaves, overlap * 10);

    for (int iStave = 0; iStave < nStaves; iStave++) {
      TGeoVolume* staveVol = createStave("staggered");

      // Put the staves in the correct position and orientation
      TGeoCombiTrans* trans = new TGeoCombiTrans();
      double theta = 360. * iStave / nStaves;
      TGeoRotation* rot = new TGeoRotation("rot", theta + 90, 0, 0);
      trans->SetRotation(rot);
      trans->SetTranslation(mInnerRadius * std::cos(2. * TMath::Pi() * iStave / nStaves), mInnerRadius * std::sin(2 * TMath::Pi() * iStave / nStaves), 0);

      LOGP(info, "Inserting {} in {} ", staveVol->GetName(), layerVol->GetName());
      layerVol->AddNode(staveVol, iStave, trans);
    }
  } else {
    LOGP(fatal, "Layout not implemented");
  }
  LOGP(info, "Inserting {} in {} ", layerVol->GetName(), motherVolume->GetName());
  motherVolume->AddNode(layerVol, 1, nullptr);
}
// ClassImp(TRKLayer);

} // namespace trk
} // namespace o2