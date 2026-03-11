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
TRKCylindricalLayer::TRKCylindricalLayer(int layerNumber, std::string layerName, float rInn, float length, float thickOrX2X0, MatBudgetParamMode mode)
  : mLayerNumber(layerNumber), mLayerName(layerName), mInnerRadius(rInn), mLength(length)
{
  if (mode == MatBudgetParamMode::Thickness) {
    mChipThickness = thickOrX2X0;
    mX2X0 = thickOrX2X0 / Si_X0;
    mOuterRadius = rInn + thickOrX2X0;
  } else if (mode == MatBudgetParamMode::X2X0) {
    mX2X0 = thickOrX2X0;
    mChipThickness = thickOrX2X0 * Si_X0;
    mOuterRadius = rInn + thickOrX2X0 * Si_X0;
  }

  LOGP(info, "Creating layer: id: {} rInner: {} rOuter: {} zLength: {} x2X0: {}", mLayerNumber, mInnerRadius, mOuterRadius, mLength, mX2X0);
}

TGeoVolume* TRKCylindricalLayer::createSensor()
{
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  std::string sensName = GeometryTGeo::getTRKSensorPattern() + std::to_string(mLayerNumber);
  TGeoShape* sensor = new TGeoTube(mInnerRadius, mInnerRadius + sSensorThickness, mLength / 2);
  TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
  sensVol->SetLineColor(kYellow);

  return sensVol;
};

TGeoVolume* TRKCylindricalLayer::createMetalStack()
{
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  std::string metalName = GeometryTGeo::getTRKMetalStackPattern() + std::to_string(mLayerNumber);
  TGeoShape* metalStack = new TGeoTube(mInnerRadius + sSensorThickness, mInnerRadius + mChipThickness, mLength / 2);
  TGeoVolume* metalVol = new TGeoVolume(metalName.c_str(), metalStack, medSi);
  metalVol->SetLineColor(kGray);

  return metalVol;
};

void TRKCylindricalLayer::createLayer(TGeoVolume* motherVolume)
{
  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");
  TGeoTube* layer = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mLength / 2);
  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
  layerVol->SetLineColor(kYellow);

  TGeoVolume* sensVol = createSensor();
  LOGP(debug, "Inserting {} in {} ", sensVol->GetName(), layerVol->GetName());
  layerVol->AddNode(sensVol, 1, nullptr);

  TGeoVolume* metalVol = createMetalStack();
  LOGP(debug, "Inserting {} in {} ", metalVol->GetName(), layerVol->GetName());
  layerVol->AddNode(metalVol, 1, nullptr);

  LOGP(debug, "Inserting {} in {} ", layerVol->GetName(), motherVolume->GetName());
  motherVolume->AddNode(layerVol, 1, nullptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TRKSegmentedLayer::TRKSegmentedLayer(int layerNumber, std::string layerName, float rInn, int numberOfModules, float thickOrX2X0, MatBudgetParamMode mode)
  : TRKCylindricalLayer(layerNumber, layerName, rInn, numberOfModules * sModuleLength, thickOrX2X0, mode), mNumberOfModules(numberOfModules)
{
}

TGeoVolume* TRKSegmentedLayer::createSensor()
{
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  std::string sensName = GeometryTGeo::getTRKSensorPattern() + std::to_string(mLayerNumber);
  TGeoShape* sensor = new TGeoBBox((sChipWidth - sDeadzoneWidth) / 2, sSensorThickness / 2, sChipLength / 2);
  TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
  sensVol->SetLineColor(kYellow);

  return sensVol;
}

TGeoVolume* TRKSegmentedLayer::createDeadzone()
{
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  std::string deadName = GeometryTGeo::getTRKDeadzonePattern() + std::to_string(mLayerNumber);
  TGeoShape* deadzone = new TGeoBBox(sDeadzoneWidth / 2, sSensorThickness / 2, sChipLength / 2);
  TGeoVolume* deadVol = new TGeoVolume(deadName.c_str(), deadzone, medSi);
  deadVol->SetLineColor(kGray);

  return deadVol;
}

TGeoVolume* TRKSegmentedLayer::createMetalStack()
{
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  std::string metalName = GeometryTGeo::getTRKMetalStackPattern() + std::to_string(mLayerNumber);
  TGeoShape* metalStack = new TGeoBBox(sChipWidth / 2, (mChipThickness - sSensorThickness) / 2, sChipLength / 2);
  TGeoVolume* metalVol = new TGeoVolume(metalName.c_str(), metalStack, medSi);
  metalVol->SetLineColor(kGray);

  return metalVol;
}

TGeoVolume* TRKSegmentedLayer::createChip()
{
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  std::string chipName = GeometryTGeo::getTRKChipPattern() + std::to_string(mLayerNumber);
  TGeoShape* chip = new TGeoBBox(sChipWidth / 2, mChipThickness / 2, sChipLength / 2);
  TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
  chipVol->SetLineColor(kYellow);

  TGeoVolume* sensVol = createSensor();
  TGeoCombiTrans* transSens = new TGeoCombiTrans();
  // transSens->SetTranslation(-sDeadzoneWidth / 2, -(mChipThickness - sSensorThickness) / 2, 0);
  transSens->SetTranslation(-sDeadzoneWidth / 2, (mChipThickness - sSensorThickness) / 2, 0);
  LOGP(debug, "Inserting {} in {} ", sensVol->GetName(), chipVol->GetName());
  chipVol->AddNode(sensVol, 1, transSens);

  TGeoVolume* deadVol = createDeadzone();
  TGeoCombiTrans* transDead = new TGeoCombiTrans();
  // transDead->SetTranslation((sChipWidth - sDeadzoneWidth) / 2, -(mChipThickness - sSensorThickness) / 2, 0);
  transDead->SetTranslation((sChipWidth - sDeadzoneWidth) / 2, (mChipThickness - sSensorThickness) / 2, 0);
  LOGP(debug, "Inserting {} in {} ", deadVol->GetName(), chipVol->GetName());
  chipVol->AddNode(deadVol, 1, transDead);

  TGeoVolume* metalVol = createMetalStack();
  TGeoCombiTrans* transMetal = new TGeoCombiTrans();
  // transMetal->SetTranslation(0, sSensorThickness / 2, 0);
  transMetal->SetTranslation(0, -sSensorThickness / 2, 0);
  LOGP(debug, "Inserting {} in {} ", metalVol->GetName(), chipVol->GetName());
  chipVol->AddNode(metalVol, 1, transMetal);

  return chipVol;
}

TGeoVolume* TRKSegmentedLayer::createModule()
{
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  std::string moduleName = GeometryTGeo::getTRKModulePattern() + std::to_string(mLayerNumber);
  TGeoShape* module = new TGeoBBox(sModuleWidth / 2, mChipThickness / 2, sModuleLength / 2);
  TGeoVolume* moduleVol = new TGeoVolume(moduleName.c_str(), module, medSi);
  moduleVol->SetLineColor(kYellow);

  for (int iChip = 0; iChip < sHalfNumberOfChips; iChip++) {
    TGeoVolume* chipVolLeft = createChip();
    double xLeft = -sModuleWidth / 2 + constants::moduleMLOT::gaps::outerEdgeLongSide + constants::moduleMLOT::chip::width / 2;
    double zLeft = -sModuleLength / 2 + constants::moduleMLOT::gaps::outerEdgeShortSide + iChip * (constants::moduleMLOT::chip::length + constants::moduleMLOT::gaps::interChips) + constants::moduleMLOT::chip::length / 2;
    TGeoCombiTrans* transLeft = new TGeoCombiTrans();
    transLeft->SetTranslation(xLeft, 0, zLeft);
    TGeoRotation* rot = new TGeoRotation();
    rot->RotateY(180);
    transLeft->SetRotation(rot);
    LOGP(debug, "Inserting {} in {} ", chipVolLeft->GetName(), moduleVol->GetName());
    moduleVol->AddNode(chipVolLeft, iChip * 2, transLeft);

    TGeoVolume* chipVolRight = createChip();
    double xRight = +sModuleWidth / 2 - constants::moduleMLOT::gaps::outerEdgeLongSide - constants::moduleMLOT::chip::width / 2;
    double zRight = -sModuleLength / 2 + constants::moduleMLOT::gaps::outerEdgeShortSide + iChip * (constants::moduleMLOT::chip::length + constants::moduleMLOT::gaps::interChips) + constants::moduleMLOT::chip::length / 2;
    TGeoCombiTrans* transRight = new TGeoCombiTrans();
    transRight->SetTranslation(xRight, 0, zRight);
    LOGP(debug, "Inserting {} in {} ", chipVolRight->GetName(), moduleVol->GetName());
    moduleVol->AddNode(chipVolRight, iChip * 2 + 1, transRight);
  }

  return moduleVol;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TRKMLLayer::TRKMLLayer(int layerNumber, std::string layerName, float rInn, int numberOfModules, float thickOrX2X0, MatBudgetParamMode mode)
  : TRKSegmentedLayer(layerNumber, layerName, rInn, numberOfModules, thickOrX2X0, mode)
{
}

TGeoVolume* TRKMLLayer::createStave()
{
  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");
  std::string staveName = GeometryTGeo::getTRKStavePattern() + std::to_string(mLayerNumber);
  TGeoShape* stave = new TGeoBBox(sStaveWidth / 2, mChipThickness / 2, mLength / 2);
  TGeoVolume* staveVol = new TGeoVolume(staveName.c_str(), stave, medAir);
  staveVol->SetLineColor(kYellow);

  for (int iModule = 0; iModule < mNumberOfModules; iModule++) {
    TGeoVolume* moduleVol = createModule();
    double zPos = -0.5 * mNumberOfModules * sModuleLength + (iModule + 0.5) * sModuleLength;
    TGeoCombiTrans* trans = new TGeoCombiTrans();
    trans->SetTranslation(0, 0, zPos);
    LOGP(debug, "Inserting {} in {} ", moduleVol->GetName(), staveVol->GetName());
    staveVol->AddNode(moduleVol, iModule, trans);
  }

  return staveVol;
}

void TRKMLLayer::createLayer(TGeoVolume* motherVolume)
{
  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");
  TGeoTube* layer = new TGeoTube(mInnerRadius - 0.333 * sLogicalVolumeThickness, mInnerRadius + 0.667 * sLogicalVolumeThickness, mLength / 2);
  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
  layerVol->SetLineColor(kYellow);

  // Compute the number of staves
  int nStaves = (int)std::ceil(mInnerRadius * 2 * TMath::Pi() / sStaveWidth);
  nStaves += nStaves % 2; // Require an even number of staves

  // Compute the size of the overlap region
  double theta = 2 * TMath::Pi() / nStaves;
  double theta1 = std::atan(sStaveWidth / 2 / mInnerRadius);
  double st = std::sin(theta);
  double ct = std::cos(theta);
  double theta2 = std::atan((mInnerRadius * st - sStaveWidth / 2 * ct) / (mInnerRadius * ct + sStaveWidth / 2 * st));
  double overlap = (theta1 - theta2) * mInnerRadius;
  LOGP(info, "Creating a layer with {} staves and {} mm overlap", nStaves, overlap * 10);

  for (int iStave = 0; iStave < nStaves; iStave++) {
    TGeoVolume* staveVol = createStave();
    TGeoCombiTrans* trans = new TGeoCombiTrans();
    double theta = 360. * iStave / nStaves;
    // TGeoRotation* rot = new TGeoRotation("rot", theta - 90 + 4, 0, 0);
    TGeoRotation* rot = new TGeoRotation("rot", theta + 90 + 4, 0, 0);
    trans->SetRotation(rot);
    trans->SetTranslation(mInnerRadius * std::cos(2. * TMath::Pi() * iStave / nStaves), mInnerRadius * std::sin(2 * TMath::Pi() * iStave / nStaves), 0);
    LOGP(debug, "Inserting {} in {} ", staveVol->GetName(), layerVol->GetName());
    layerVol->AddNode(staveVol, iStave, trans);
  }

  LOGP(debug, "Inserting {} in {} ", layerVol->GetName(), motherVolume->GetName());
  motherVolume->AddNode(layerVol, 1, nullptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TRKOTLayer::TRKOTLayer(int layerNumber, std::string layerName, float rInn, int numberOfModules, float thickOrX2X0, MatBudgetParamMode mode)
  : TRKSegmentedLayer(layerNumber, layerName, rInn, numberOfModules, thickOrX2X0, mode)
{
}

TGeoVolume* TRKOTLayer::createHalfStave()
{
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  std::string halfStaveName = GeometryTGeo::getTRKHalfStavePattern() + std::to_string(mLayerNumber);
  TGeoShape* halfStave = new TGeoBBox(sHalfStaveWidth / 2, mChipThickness / 2, mLength / 2);
  TGeoVolume* halfStaveVol = new TGeoVolume(halfStaveName.c_str(), halfStave, medSi);
  halfStaveVol->SetLineColor(kYellow);

  for (int iModule = 0; iModule < mNumberOfModules; iModule++) {
    TGeoVolume* moduleVol = createModule();
    double zPos = -0.5 * mNumberOfModules * sModuleLength + (iModule + 0.5) * sModuleLength;
    TGeoCombiTrans* trans = new TGeoCombiTrans();
    trans->SetTranslation(0, 0, zPos);
    LOGP(debug, "Inserting {} in {} ", moduleVol->GetName(), halfStaveVol->GetName());
    halfStaveVol->AddNode(moduleVol, iModule, trans);
  }

  return halfStaveVol;
}

TGeoVolume* TRKOTLayer::createStave()
{
  std::string staveName = GeometryTGeo::getTRKStavePattern() + std::to_string(mLayerNumber);
  TGeoVolume* staveVol = new TGeoVolumeAssembly(staveName.c_str());

  TGeoVolume* halfStaveVolLeft = createHalfStave();
  TGeoCombiTrans* transLeft = new TGeoCombiTrans();
  transLeft->SetTranslation(-(sHalfStaveWidth - sInStaveOverlap) / 2, 0, 0);
  LOGP(debug, "Inserting {} in {} ", halfStaveVolLeft->GetName(), staveVol->GetName());
  staveVol->AddNode(halfStaveVolLeft, 0, transLeft);

  TGeoVolume* halfStaveVolRight = createHalfStave();
  TGeoCombiTrans* transRight = new TGeoCombiTrans();
  transRight->SetTranslation((sHalfStaveWidth - sInStaveOverlap) / 2, 0.2, 0);
  LOGP(debug, "Inserting {} in {} ", halfStaveVolRight->GetName(), staveVol->GetName());
  staveVol->AddNode(halfStaveVolRight, 1, transRight);

  return staveVol;
}

void TRKOTLayer::createLayer(TGeoVolume* motherVolume)
{
  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");
  TGeoTube* layer = new TGeoTube(mInnerRadius - 0.333 * sLogicalVolumeThickness, mInnerRadius + 0.667 * sLogicalVolumeThickness, mLength / 2);
  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
  layerVol->SetLineColor(kYellow);

  // Compute the number of staves
  int nStaves = (int)std::ceil(mInnerRadius * 2 * TMath::Pi() / sStaveWidth);
  nStaves += nStaves % 2; // Require an even number of staves

  // Compute the size of the overlap region
  double theta = 2 * TMath::Pi() / nStaves;
  double theta1 = std::atan(sStaveWidth / 2 / mInnerRadius);
  double st = std::sin(theta);
  double ct = std::cos(theta);
  double theta2 = std::atan((mInnerRadius * st - sStaveWidth / 2 * ct) / (mInnerRadius * ct + sStaveWidth / 2 * st));
  double overlap = (theta1 - theta2) * mInnerRadius;
  LOGP(info, "Creating a layer with {} staves and {} mm overlap", nStaves, overlap * 10);

  for (int iStave = 0; iStave < nStaves; iStave++) {
    TGeoVolume* staveVol = createStave();
    TGeoCombiTrans* trans = new TGeoCombiTrans();
    double theta = 360. * iStave / nStaves;
    // TGeoRotation* rot = new TGeoRotation("rot", theta - 90, 0, 0);
    TGeoRotation* rot = new TGeoRotation("rot", theta + 90, 0, 0);
    trans->SetRotation(rot);
    trans->SetTranslation(mInnerRadius * std::cos(2. * TMath::Pi() * iStave / nStaves), mInnerRadius * std::sin(2 * TMath::Pi() * iStave / nStaves), 0);
    LOGP(debug, "Inserting {} in {} ", staveVol->GetName(), layerVol->GetName());
    layerVol->AddNode(staveVol, iStave, trans);
  }

  LOGP(debug, "Inserting {} in {} ", layerVol->GetName(), motherVolume->GetName());
  motherVolume->AddNode(layerVol, 1, nullptr);
}
// ClassImp(TRKLayer);

} // namespace trk
} // namespace o2