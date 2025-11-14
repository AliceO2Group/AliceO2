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
#include "TRKBase/Specs.h"

#include "Framework/Logger.h"

#include <TGeoTube.h>
#include <TGeoBBox.h>
#include <TGeoVolume.h>

#include <TMath.h>

namespace o2
{
namespace trk
{
TRKLayer::TRKLayer(int layerNumber, std::string layerName, float rInn, float rOut, int numberOfModules, float layerX2X0)
  : mLayerNumber(layerNumber), mLayout(kCylinder), mLayerName(layerName), mInnerRadius(rInn), mOuterRadius(rOut), mNumberOfModules(numberOfModules), mX2X0(layerX2X0), mChipWidth(constants::moduleMLOT::chip::width), mChipLength(constants::moduleMLOT::chip::length), mDeadzoneWidth(constants::moduleMLOT::chip::passiveEdgeReadOut), mSensorThickness(constants::moduleMLOT::silicon::thickness), mHalfNumberOfChips(4)
{
  float Si_X0 = 9.5f;
  mChipThickness = mX2X0 * Si_X0;
  LOGP(info, "Creating layer: id: {} rInner: {} rOuter: {} zLength: {} x2X0: {}", mLayerNumber, mInnerRadius, mOuterRadius, getZ(), mX2X0);
}

TRKLayer::TRKLayer(int layerNumber, std::string layerName, float rInn, int numberOfModules, float thick)
  : mLayerNumber(layerNumber), mLayout(kCylinder), mLayerName(layerName), mInnerRadius(rInn), mNumberOfModules(numberOfModules), mChipThickness(thick), mChipWidth(constants::moduleMLOT::chip::width), mChipLength(constants::moduleMLOT::chip::length), mDeadzoneWidth(constants::moduleMLOT::chip::passiveEdgeReadOut), mSensorThickness(constants::moduleMLOT::silicon::thickness), mHalfNumberOfChips(4)
{
  float Si_X0 = 9.5f;
  mOuterRadius = rInn + thick;
  mX2X0 = mChipThickness / Si_X0;
  LOGP(info, "Creating layer: id: {} rInner: {} rOuter: {} zLength: {} x2X0: {}", mLayerNumber, mInnerRadius, mOuterRadius, getZ(), mX2X0);
}

TGeoVolume* TRKLayer::createSensor(std::string type)
{
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  std::string sensName = GeometryTGeo::getTRKSensorPattern() + std::to_string(mLayerNumber);

  TGeoShape* sensor;

  if (type == "cylinder") {
    sensor = new TGeoTube(mInnerRadius, mInnerRadius + mSensorThickness, mChipLength / 2); // TO BE CHECKED !!!
  } else if (type == "flat") {
    sensor = new TGeoBBox((mChipWidth - mDeadzoneWidth) / 2, mSensorThickness / 2, mChipLength / 2); // TO BE CHECKED !!!
  } else {
    LOGP(fatal, "Sensor of type '{}' is not implemented", type);
  }

  TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
  sensVol->SetLineColor(kYellow);

  return sensVol;
};

TGeoVolume* TRKLayer::createDeadzone(std::string type)
{
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  std::string deadName = GeometryTGeo::getTRKDeadzonePattern() + std::to_string(mLayerNumber);

  TGeoShape* deadzone;

  if (type == "cylinder") {
    deadzone = new TGeoTube(mInnerRadius, mInnerRadius + mSensorThickness, mChipLength / 2); // TO BE CHECKED !!!
  } else if (type == "flat") {
    deadzone = new TGeoBBox(mDeadzoneWidth / 2, mSensorThickness / 2, mChipLength / 2); // TO BE CHECKED !!!
  } else {
    LOGP(fatal, "Deadzone of type '{}' is not implemented", type);
  }

  TGeoVolume* deadVol = new TGeoVolume(deadName.c_str(), deadzone, medSi);
  deadVol->SetLineColor(kGray);

  return deadVol;
};

TGeoVolume* TRKLayer::createMetalStack(std::string type)
{
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  std::string metalName = GeometryTGeo::getTRKMetalStackPattern() + std::to_string(mLayerNumber);

  TGeoShape* metalStack;

  if (type == "cylinder") {
    metalStack = new TGeoTube(mInnerRadius + mSensorThickness, mInnerRadius + mChipThickness, mChipLength / 2); // TO BE CHECKED !!!
  } else if (type == "flat") {
    metalStack = new TGeoBBox(mChipWidth / 2, (mChipThickness - mSensorThickness) / 2, mChipLength / 2); // TO BE CHECKED !!!
  } else {
    LOGP(fatal, "Metal stack of type '{}' is not implemented", type);
  }

  TGeoVolume* metalVol = new TGeoVolume(metalName.c_str(), metalStack, medSi);
  metalVol->SetLineColor(kGray);

  return metalVol;
};

TGeoVolume* TRKLayer::createChip(std::string type)
{
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  std::string chipName = GeometryTGeo::getTRKChipPattern() + std::to_string(mLayerNumber);

  TGeoShape* chip;
  TGeoVolume* chipVol;

  TGeoVolume* sensVol;
  TGeoVolume* deadVol;
  TGeoVolume* metalVol;

  if (type == "cylinder") {
    chip = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mChipLength / 2);
    chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);

    sensVol = createSensor("cylinder");
    metalVol = createMetalStack("cylinder");

    TGeoCombiTrans* transSens = new TGeoCombiTrans();
    transSens->SetTranslation(0, -(mChipThickness - mSensorThickness) / 2, 0); // TO BE CHECKED !!!
    LOGP(debug, "Inserting {} in {} ", sensVol->GetName(), chipVol->GetName());
    chipVol->AddNode(sensVol, 1, transSens);

    TGeoCombiTrans* transMetal = new TGeoCombiTrans();
    transMetal->SetTranslation(0, mSensorThickness / 2, 0); // TO BE CHECKED !!!
    LOGP(debug, "Inserting {} in {} ", metalVol->GetName(), chipVol->GetName());
    chipVol->AddNode(metalVol, 1, transMetal);

    // deadVol = createDeadzone("cylinder");
  } else if (type == "flat") {
    chip = new TGeoBBox(mChipWidth / 2, mChipThickness / 2, mChipLength / 2); // TO BE CHECKED !!!
    chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);

    sensVol = createSensor("flat");
    deadVol = createDeadzone("flat");
    metalVol = createMetalStack("flat");

    TGeoCombiTrans* transSens = new TGeoCombiTrans();
    transSens->SetTranslation(-mDeadzoneWidth / 2, -(mChipThickness - mSensorThickness) / 2, 0); // TO BE CHECKED !!!
    LOGP(debug, "Inserting {} in {} ", sensVol->GetName(), chipVol->GetName());
    chipVol->AddNode(sensVol, 1, transSens);

    TGeoCombiTrans* transDead = new TGeoCombiTrans();
    transDead->SetTranslation((mChipWidth - mDeadzoneWidth) / 2, -(mChipThickness - mSensorThickness) / 2, 0); // TO BE CHECKED !!!
    LOGP(debug, "Inserting {} in {} ", deadVol->GetName(), chipVol->GetName());
    chipVol->AddNode(deadVol, 1, transDead);

    TGeoCombiTrans* transMetal = new TGeoCombiTrans();
    transMetal->SetTranslation(0, mSensorThickness / 2, 0); // TO BE CHECKED !!!
    LOGP(debug, "Inserting {} in {} ", metalVol->GetName(), chipVol->GetName());
    chipVol->AddNode(metalVol, 1, transMetal);
  } else {
    LOGP(fatal, "Sensor of type '{}' is not implemented", type);
  }

  chipVol->SetLineColor(kYellow);

  return chipVol;
}

TGeoVolume* TRKLayer::createModule(std::string type)
{
  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");
  std::string moduleName = GeometryTGeo::getTRKModulePattern() + std::to_string(mLayerNumber);

  TGeoShape* module;
  TGeoVolume* moduleVol;

  if (type == "cylinder") {
    module = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mChipLength / 2);
    moduleVol = new TGeoVolume(moduleName.c_str(), module, medAir);

    TGeoVolume* chipVol = createChip("cylinder");
    LOGP(debug, "Inserting {} in {} ", chipVol->GetName(), moduleVol->GetName());
    moduleVol->AddNode(chipVol, 1, nullptr);
  } else if (type == "flat") {
    double moduleWidth = constants::moduleMLOT::width;
    double moduleLength = constants::moduleMLOT::length;

    module = new TGeoBBox(moduleWidth / 2, mChipThickness / 2, moduleLength / 2); // TO BE CHECKED !!!
    moduleVol = new TGeoVolume(moduleName.c_str(), module, medAir);

    for (int iChip = 0; iChip < mHalfNumberOfChips; iChip++) {
      TGeoVolume* chipVolLeft = createChip("flat");
      TGeoVolume* chipVolRight = createChip("flat");

      // Put the chips in the correct position
      double xLeft = -moduleWidth / 2 + constants::moduleMLOT::gaps::outerEdgeLongSide + constants::moduleMLOT::chip::width / 2;
      double zLeft = -moduleLength / 2 + constants::moduleMLOT::gaps::outerEdgeShortSide + iChip * (constants::moduleMLOT::chip::length + constants::moduleMLOT::gaps::interChips) + constants::moduleMLOT::chip::length / 2;

      TGeoCombiTrans* transLeft = new TGeoCombiTrans();
      transLeft->SetTranslation(xLeft, 0, zLeft); // TO BE CHECKED !!!
      TGeoRotation* rot = new TGeoRotation();
      rot->RotateY(180);
      transLeft->SetRotation(rot);
      LOGP(debug, "Inserting {} in {} ", chipVolLeft->GetName(), moduleVol->GetName());
      moduleVol->AddNode(chipVolLeft, iChip * 2, transLeft);

      double xRight = +moduleWidth / 2 - constants::moduleMLOT::gaps::outerEdgeLongSide - constants::moduleMLOT::chip::width / 2;
      double zRight = -moduleLength / 2 + constants::moduleMLOT::gaps::outerEdgeShortSide + iChip * (constants::moduleMLOT::chip::length + constants::moduleMLOT::gaps::interChips) + constants::moduleMLOT::chip::length / 2;

      TGeoCombiTrans* transRight = new TGeoCombiTrans();
      transRight->SetTranslation(xRight, 0, zRight); // TO BE CHECKED !!!
      LOGP(debug, "Inserting {} in {} ", chipVolRight->GetName(), moduleVol->GetName());
      moduleVol->AddNode(chipVolRight, iChip * 2 + 1, transRight);
    }
  } else {
    LOGP(fatal, "Chip of type '{}' is not implemented", type);
  }

  moduleVol->SetLineColor(kYellow);

  return moduleVol;
}

TGeoVolume* TRKLayer::createHalfStave(std::string type)
{
  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");
  std::string halfStaveName = GeometryTGeo::getTRKHalfStavePattern() + std::to_string(mLayerNumber);

  TGeoShape* halfStave;
  TGeoVolume* halfStaveVol;

  if (type == "cylinder") {
    halfStave = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mChipLength / 2);
    halfStaveVol = new TGeoVolume(halfStaveName.c_str(), halfStave, medAir);

    TGeoVolume* moduleVol = createModule("cylinder");
    LOGP(debug, "Inserting {} in {} ", moduleVol->GetName(), halfStaveVol->GetName());
    halfStaveVol->AddNode(moduleVol, 1, nullptr);
  } else if (type == "flat") {
    double moduleLength = constants::moduleMLOT::length;
    double halfStaveWidth = constants::OT::halfstave::width;
    double halfStaveLength = constants::moduleMLOT::length * mNumberOfModules;

    halfStave = new TGeoBBox(halfStaveWidth / 2, mChipThickness / 2, halfStaveLength / 2);
    halfStaveVol = new TGeoVolume(halfStaveName.c_str(), halfStave, medAir);

    for (int iModule = 0; iModule < mNumberOfModules; iModule++) {
      TGeoVolume* moduleVol = createModule("flat");

      // Put the modules in the correct position
      double zPos = -0.5 * mNumberOfModules * moduleLength + (iModule + 0.5) * moduleLength;

      TGeoCombiTrans* trans = new TGeoCombiTrans();
      trans->SetTranslation(0, 0, zPos); // TO BE CHECKED !!!

      LOGP(debug, "Inserting {} in {} ", moduleVol->GetName(), halfStaveVol->GetName());
      halfStaveVol->AddNode(moduleVol, iModule, trans);
    }
  }
  return halfStaveVol;
}

TGeoVolume* TRKLayer::createStave(std::string type)
{
  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");
  std::string staveName = GeometryTGeo::getTRKStavePattern() + std::to_string(mLayerNumber);

  TGeoShape* stave;
  TGeoVolume* staveVol;

  if (type == "cylinder") {
    stave = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mChipLength / 2);
    staveVol = new TGeoVolume(staveName.c_str(), stave, medAir);

    TGeoVolume* moduleVol = createModule("cylinder");
    LOGP(debug, "Inserting {} in {} ", moduleVol->GetName(), staveVol->GetName());
    staveVol->AddNode(moduleVol, 1, nullptr);
  } else if (type == "flat") {
    double moduleLength = constants::moduleMLOT::length;
    double staveWidth = constants::ML::width;
    double staveLength = constants::moduleMLOT::length * mNumberOfModules;

    stave = new TGeoBBox(staveWidth / 2, mChipThickness / 2, staveLength / 2);
    staveVol = new TGeoVolume(staveName.c_str(), stave, medAir);

    for (int iModule = 0; iModule < mNumberOfModules; iModule++) {
      TGeoVolume* moduleVol = createModule("flat");

      // Put the modules in the correct position
      double zPos = -0.5 * mNumberOfModules * moduleLength + (iModule + 0.5) * moduleLength;

      TGeoCombiTrans* trans = new TGeoCombiTrans();
      trans->SetTranslation(0, 0, zPos); // TO BE CHECKED !!!

      LOGP(debug, "Inserting {} in {} ", moduleVol->GetName(), staveVol->GetName());
      staveVol->AddNode(moduleVol, iModule, trans);
    }
  } else if (type == "staggered") {
    /*double moduleWidth = constants::moduleMLOT::width;
    double moduleLength = constants::moduleMLOT::length;*/

    double halfstaveWidth = constants::ML::width;
    double staveWidth = constants::OT::width; // Each stave has two modules (based on the LOI design)
    double staveLength = constants::moduleMLOT::length * mNumberOfModules;

    stave = new TGeoBBox(staveWidth / 2, mLogicalVolumeThickness / 2, staveLength / 2);
    staveVol = new TGeoVolume(staveName.c_str(), stave, medAir);

    // Put the half staves in the correct position
    TGeoVolume* halfStaveVolLeft = createHalfStave("flat");
    TGeoVolume* halfStaveVolRight = createHalfStave("flat");

    TGeoCombiTrans* transLeft = new TGeoCombiTrans();
    transLeft->SetTranslation(-halfstaveWidth / 2 + 0.05, 0, 0); // TO BE CHECKED !!! 1mm overlap between the modules
    LOGP(debug, "Inserting {} in {} ", halfStaveVolLeft->GetName(), staveVol->GetName());
    staveVol->AddNode(halfStaveVolLeft, 0, transLeft);

    TGeoCombiTrans* transRight = new TGeoCombiTrans();
    transRight->SetTranslation(halfstaveWidth / 2 - 0.05, 0.2, 0); // TO BE CHECKED !!! 1mm overlap between the modules
    LOGP(debug, "Inserting {} in {} ", halfStaveVolRight->GetName(), staveVol->GetName());
    staveVol->AddNode(halfStaveVolRight, 1, transRight);
  } else {
    LOGP(fatal, "Chip of type '{}' is not implemented", type);
  }

  staveVol->SetLineColor(kYellow);

  return staveVol;
}

void TRKLayer::createLayer(TGeoVolume* motherVolume)
{
  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");

  double layerThickness = mChipThickness;
  if (mLayout != eLayout::kCylinder) {
    layerThickness = mLogicalVolumeThickness;
  }

  TGeoTube* layer;
  TGeoVolume* layerVol;

  if (mLayout == eLayout::kCylinder) {
    layer = new TGeoTube(mInnerRadius - 0.333 * layerThickness, mInnerRadius + 0.667 * layerThickness, mChipLength / 2);
    layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);

    TGeoVolume* staveVol = createStave("cylinder");
    LOGP(debug, "Inserting {} in {} ", staveVol->GetName(), layerVol->GetName());
    layerVol->AddNode(staveVol, 1, nullptr);
  } else if (mLayout == eLayout::kTurboStaves) {
    double layerLength = constants::moduleMLOT::length * mNumberOfModules;
    double staveWidth = constants::ML::width; // Each stave has two modules (based on the LOI design)

    if (mInnerRadius > 25) {
      staveWidth = constants::OT::width; // Outer layers have two modules per stave
    }

    layer = new TGeoTube(mInnerRadius - 0.333 * layerThickness, mInnerRadius + 0.667 * layerThickness, layerLength / 2);
    layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);

    // Compute the number of staves
    int nStaves = (int)std::ceil(mInnerRadius * 2 * TMath::Pi() / staveWidth);
    nStaves += nStaves % 2; // Require an even number of staves

    // Compute the size of the overlap region
    double theta = 2 * TMath::Pi() / nStaves;
    double theta1 = std::atan(staveWidth / 2 / mInnerRadius);
    double st = std::sin(theta);
    double ct = std::cos(theta);
    double theta2 = std::atan((mInnerRadius * st - staveWidth / 2 * ct) / (mInnerRadius * ct + staveWidth / 2 * st));
    double overlap = (theta1 - theta2) * mInnerRadius;
    LOGP(info, "Creating a layer with {} staves and {} mm overlap", nStaves, overlap * 10);

    for (int iStave = 0; iStave < nStaves; iStave++) {
      TGeoVolume* staveVol = createStave("flat");

      // Put the staves in the correct position and orientation
      TGeoCombiTrans* trans = new TGeoCombiTrans();
      double theta = 360. * iStave / nStaves;
      TGeoRotation* rot = new TGeoRotation("rot", theta + 90 + 3, 0, 0);
      trans->SetRotation(rot);
      trans->SetTranslation(mInnerRadius * std::cos(2. * TMath::Pi() * iStave / nStaves), mInnerRadius * std::sin(2 * TMath::Pi() * iStave / nStaves), 0);

      LOGP(debug, "Inserting {} in {} ", staveVol->GetName(), layerVol->GetName());
      layerVol->AddNode(staveVol, iStave, trans);
    }
  } else if (mLayout == kStaggered) {
    double layerLength = constants::moduleMLOT::length * mNumberOfModules;

    layer = new TGeoTube(mInnerRadius - 0.333 * layerThickness, mInnerRadius + 0.667 * layerThickness, layerLength / 2);
    layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);

    // Compute the number of staves
    double staveWidth = constants::OT::width; // Each stave has two modules (based on the LOI design)
    int nStaves = (int)std::ceil(mInnerRadius * 2 * TMath::Pi() / staveWidth);
    nStaves += nStaves % 2; // Require an even number of staves

    // Compute the size of the overlap region
    double theta = 2 * TMath::Pi() / nStaves;
    double theta1 = std::atan(staveWidth / 2 / mInnerRadius);
    double st = std::sin(theta);
    double ct = std::cos(theta);
    double theta2 = std::atan((mInnerRadius * st - staveWidth / 2 * ct) / (mInnerRadius * ct + staveWidth / 2 * st));
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

      LOGP(debug, "Inserting {} in {} ", staveVol->GetName(), layerVol->GetName());
      layerVol->AddNode(staveVol, iStave, trans);
    }
  } else {
    LOGP(fatal, "Layout not implemented");
  }
  layerVol->SetLineColor(kYellow);

  LOGP(debug, "Inserting {} in {} ", layerVol->GetName(), motherVolume->GetName());
  motherVolume->AddNode(layerVol, 1, nullptr);
}
// ClassImp(TRKLayer);

} // namespace trk
} // namespace o2