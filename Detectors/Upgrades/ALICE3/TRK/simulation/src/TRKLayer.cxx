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

#include "Framework/Logger.h"

#include "TRKBase/GeometryTGeo.h"
#include "TRKBase/Specs.h"
#include <TGeoBBox.h>
#include <TGeoTube.h>
#include <TGeoVolume.h>
#include <TMath.h>

#include <cassert>
#include <cmath>
#include <string>
#include <utility>

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

TRKSegmentedLayer::TRKSegmentedLayer(int layerNumber, std::string layerName, float rInn, float tiltAngle, int numberOfStaves, int numberOfModules, float thickOrX2X0, MatBudgetParamMode mode)
  : TRKCylindricalLayer(layerNumber, layerName, rInn, numberOfModules * sModuleLength, thickOrX2X0, mode), mTiltAngle(tiltAngle), mNumberOfStaves(numberOfStaves), mNumberOfModules(numberOfModules)
{
  assert(numberOfStaves % 2 == 0 && "Error: numberOfStaves must be even!");
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

  TGeoVolume* deadVol = createDeadzone();
  TGeoCombiTrans* transDead = new TGeoCombiTrans();

  TGeoVolume* metalVol = createMetalStack();
  TGeoCombiTrans* transMetal = new TGeoCombiTrans();

  if (!mIsFlipped) {
    transSens->SetTranslation(-sDeadzoneWidth / 2, (mChipThickness - sSensorThickness) / 2, 0);
    transDead->SetTranslation((sChipWidth - sDeadzoneWidth) / 2, (mChipThickness - sSensorThickness) / 2, 0);
    transMetal->SetTranslation(0, -sSensorThickness / 2, 0);
  } else {
    transSens->SetTranslation(-sDeadzoneWidth / 2, -(mChipThickness - sSensorThickness) / 2, 0);
    transDead->SetTranslation((sChipWidth - sDeadzoneWidth) / 2, -(mChipThickness - sSensorThickness) / 2, 0);
    transMetal->SetTranslation(0, sSensorThickness / 2, 0);
  }

  LOGP(debug, "Inserting {} in {} ", sensVol->GetName(), chipVol->GetName());
  chipVol->AddNode(sensVol, 1, transSens);

  LOGP(debug, "Inserting {} in {} ", deadVol->GetName(), chipVol->GetName());
  chipVol->AddNode(deadVol, 1, transDead);

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

std::pair<float, float> TRKSegmentedLayer::getBoundingRadii(double staveWidth) const
{
  const float avgRadius = 0.5 * (mInnerRadius + mOuterRadius);
  const float staveSizeX = staveWidth;
  const float staveSizeY = mOuterRadius - mInnerRadius;

  /*const float deltaForTilt = 0.5 * (std::sin(TMath::DegToRad() * mTiltAngle) * staveSizeX + std::cos(TMath::DegToRad() * mTiltAngle) * staveSizeY);

  float radiusMin = std::sqrt(avgRadius * avgRadius + 0.25 * staveSizeX * staveSizeX + 0.25 * staveSizeY * staveSizeY - avgRadius * 2. * deltaForTilt);
  float radiusMax = std::sqrt(avgRadius * avgRadius + 0.25 * staveSizeX * staveSizeX + 0.25 * staveSizeY * staveSizeY + avgRadius * 2. * deltaForTilt);*/

  const double alpha = TMath::DegToRad() * std::abs(mTiltAngle);

  // The maximum distance from the center is always the outer top corner
  double u_max = avgRadius * std::sin(alpha) + staveSizeX / 2.0;
  double v_max = avgRadius * std::cos(alpha) + staveSizeY / 2.0;
  double radiusMax = std::sqrt(u_max * u_max + v_max * v_max);

  // The perpendicular distance from the center to the line where the inner face lies
  double perpDistance = avgRadius * std::cos(alpha) - staveSizeY / 2.0;

  // The projection of the center along the width of the stave
  double projDistance = avgRadius * std::sin(alpha);

  double radiusMin;
  if (projDistance <= staveSizeX / 2.0) {
    // The center projects directly inside the flat face.
    // The closest point is on the face itself, not on the corner
    radiusMin = perpDistance;
  } else {
    // The center projects outside the face. The closest point is the inner corner
    double u_min = projDistance - staveSizeX / 2.0;
    radiusMin = std::sqrt(u_min * u_min + perpDistance * perpDistance);
  }

  // Add a 0.5 mm safety margin to prevent false-positive overlaps in ROOT's geometry checker caused by floating-point inaccuracies
  const float precisionMargin = 0.05f;

  return {radiusMin - precisionMargin, radiusMax + precisionMargin};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TRKMLLayer::TRKMLLayer(int layerNumber, std::string layerName, float rInn, float staggerOffset, float tiltAngle, int numberOfStaves, int numberOfModules, float thickOrX2X0, MatBudgetParamMode mode)
  : TRKSegmentedLayer(layerNumber, layerName, rInn, tiltAngle, numberOfStaves, numberOfModules, thickOrX2X0, mode), mStaggerOffset(staggerOffset)
{
  if (mLayerNumber == sFlippedLayerNumber) {
    mOuterRadius = rInn;
    mInnerRadius = rInn - mChipThickness;
    mIsFlipped = true;
    mStaggerOffset = -staggerOffset;
    LOGP(info, "Layer {} is flipped: sensor and metal stack positions are switched", mLayerNumber);
  }
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
  // Retrieve exact bounding boundaries and create the logical container volume
  auto [rMin, rMax] = getBoundingRadii(sStaveWidth);

  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");
  // TGeoTube* layer = new TGeoTube(mInnerRadius - 0.333 * sLogicalVolumeThickness, mInnerRadius + 0.667 * sLogicalVolumeThickness, mLength / 2);
  TGeoTube* layer = new TGeoTube(rMin, rMax, mLength / 2);
  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
  layerVol->SetLineColor(kYellow);

  // Compute the number of staves
  // int nStaves = (int)std::ceil(mInnerRadius * 2 * TMath::Pi() / sStaveWidth);
  // nStaves += nStaves % 2; // Require an even number of staves

  // Nominal average radii used as placement barycenters for the staves
  const double avgRadiusInner = 0.5 * (mInnerRadius + mOuterRadius);
  const double avgRadiusOuter = avgRadiusInner + mStaggerOffset;

  // Compute the size of the overlap region
  double theta = 2. * TMath::Pi() / mNumberOfStaves;
  double theta1 = std::atan(sStaveWidth / 2 / mInnerRadius);
  double st = std::sin(theta);
  double ct = std::cos(theta);
  double theta2 = std::atan((mInnerRadius * st - sStaveWidth / 2 * ct) / (mInnerRadius * ct + sStaveWidth / 2 * st));
  double overlap = (theta1 - theta2) * mInnerRadius;
  LOGP(info, "Creating a layer with {} staves and {} mm overlap", mNumberOfStaves, overlap * 10);

  for (int iStave = 0; iStave < mNumberOfStaves; iStave++) {
    TGeoVolume* staveVol = createStave();
    TGeoCombiTrans* trans = new TGeoCombiTrans();
    // If the number of staves is a multiple of 4, rotate by half a stave to avoid having the first one exactly on the x
    double phi = (mNumberOfStaves % 4 == 0) ? theta * (iStave + 0.5) : theta * iStave;
    double phiDeg = phi * TMath::RadToDeg();
    TGeoRotation* rot = new TGeoRotation("rot", phiDeg + 90 + mTiltAngle, 0, 0);
    trans->SetRotation(rot);
    // float trueRadius = (mLayerNumber == 3 || mLayerNumber == 4) ? (iStave % 2 == 0 ? mInnerRadius : mInnerRadius + mStaggerOffset) : mInnerRadius;
    float trueRadius = (mLayerNumber == 3 || mLayerNumber == 4) ? (iStave % 2 == 0 ? avgRadiusInner : avgRadiusOuter) : avgRadiusInner;
    trans->SetTranslation(trueRadius * std::cos(phi), trueRadius * std::sin(phi), 0);
    LOGP(debug, "Inserting {} in {} ", staveVol->GetName(), layerVol->GetName());
    layerVol->AddNode(staveVol, iStave, trans);
  }

  LOGP(debug, "Inserting {} in {} ", layerVol->GetName(), motherVolume->GetName());
  motherVolume->AddNode(layerVol, 1, nullptr);
}

std::pair<float, float> TRKMLLayer::getBoundingRadii(double staveWidth) const
{
  // Get the baseline RMin from the base class
  auto [defaultRadiusMin, defaultRadiusMax] = TRKSegmentedLayer::getBoundingRadii(staveWidth);

  // If we are not in the staggered layers, return the baseline values
  if (mLayerNumber != 3 && mLayerNumber != 4) {
    return {defaultRadiusMin, defaultRadiusMax};
  }

  /*// For staggered layers, we must recalculate RMax based on the outer shifted row
  const float avgRadiusInner = 0.5 * (mInnerRadius + mOuterRadius);
  const float avgRadiusOuter = avgRadiusInner + mStaggerOffset;

  const float staveSizeX = staveWidth;
  const float staveSizeY = mOuterRadius - mInnerRadius;

  const float deltaForTiltOuter = 0.5 * (std::sin(TMath::DegToRad() * mTiltAngle) * staveSizeX + std::cos(TMath::DegToRad() * mTiltAngle) * staveSizeY);

  const float radiusMax = std::sqrt(avgRadiusOuter * avgRadiusOuter + 0.25 * staveSizeX * staveSizeX + 0.25 * staveSizeY * staveSizeY + avgRadiusOuter * 2. * deltaForTiltOuter);*/

  const float avgRadiusInner = 0.5 * (mInnerRadius + mOuterRadius);
  const float avgRadiusStaggered = avgRadiusInner + mStaggerOffset;

  const float staveSizeX = staveWidth;
  const float staveSizeY = mOuterRadius - mInnerRadius;
  const float alpha = TMath::DegToRad() * std::abs(mTiltAngle);

  const float precisionMargin = 0.05f;

  // If the layer is NOT flipped (e.g., Layer 4), the stagger goes outwards
  // Therefore, we must recalculate only the maximum radius based on the outer shifted row
  if (!mIsFlipped) {
    float u_max = avgRadiusStaggered * std::sin(alpha) + staveSizeX / 2.0;
    float v_max = avgRadiusStaggered * std::cos(alpha) + staveSizeY / 2.0;
    float radiusMax = std::sqrt(u_max * u_max + v_max * v_max);

    return {defaultRadiusMin, radiusMax + precisionMargin};
  }
  // If the layer IS flipped (e.g., Layer 3), the stagger goes inwards
  // Therefore, we must recalculate only the minimum radius based on the inner shifted row
  else {
    double perpDistance = avgRadiusStaggered * std::cos(alpha) - staveSizeY / 2.0;
    double projDistance = avgRadiusStaggered * std::sin(alpha);
    double newRadiusMin;

    if (projDistance <= staveSizeX / 2.0) {
      newRadiusMin = perpDistance;
    } else {
      double u_min = projDistance - staveSizeX / 2.0;
      newRadiusMin = std::sqrt(u_min * u_min + perpDistance * perpDistance);
    }

    return {newRadiusMin - precisionMargin, defaultRadiusMax};
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TRKOTLayer::TRKOTLayer(int layerNumber, std::string layerName, float rInn, float tiltAngle, int numberOfStaves, int numberOfModules, float thickOrX2X0, MatBudgetParamMode mode)
  : TRKSegmentedLayer(layerNumber, layerName, rInn, tiltAngle, numberOfStaves, numberOfModules, thickOrX2X0, mode)
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
  // Retrieve exact bounding boundaries automatically inherited from TRKSegmentedLayer
  auto [rMin, rMax] = getBoundingRadii(sStaveWidth);

  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");
  // TGeoTube* layer = new TGeoTube(mInnerRadius - 0.333 * sLogicalVolumeThickness, mInnerRadius + 0.667 * sLogicalVolumeThickness, mLength / 2);
  TGeoTube* layer = new TGeoTube(rMin, rMax, mLength / 2);
  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
  layerVol->SetLineColor(kYellow);

  // Compute the number of staves
  int nStaves = (int)std::ceil(mInnerRadius * 2 * TMath::Pi() / sStaveWidth);
  nStaves += nStaves % 2; // Require an even number of staves

  // Nominal average radius used as the placement barycenter for all staves
  const double avgRadius = 0.5 * (mInnerRadius + mOuterRadius);

  // Compute the size of the overlap region
  double theta = 2. * TMath::Pi() / nStaves;
  double theta1 = std::atan(sStaveWidth / 2 / mInnerRadius);
  double st = std::sin(theta);
  double ct = std::cos(theta);
  double theta2 = std::atan((mInnerRadius * st - sStaveWidth / 2 * ct) / (mInnerRadius * ct + sStaveWidth / 2 * st));
  double overlap = (theta1 - theta2) * mInnerRadius;
  LOGP(info, "Creating a layer with {} staves and {} mm overlap", nStaves, overlap * 10);

  for (int iStave = 0; iStave < nStaves; iStave++) {
    TGeoVolume* staveVol = createStave();
    TGeoCombiTrans* trans = new TGeoCombiTrans();
    double phi = theta * iStave;
    double phiDeg = phi * TMath::RadToDeg();
    TGeoRotation* rot = new TGeoRotation("rot", phiDeg + 90 + mTiltAngle, 0, 0);
    trans->SetRotation(rot);
    // trans->SetTranslation(mInnerRadius * std::cos(phi), mInnerRadius * std::sin(phi), 0);
    trans->SetTranslation(avgRadius * std::cos(phi), avgRadius * std::sin(phi), 0);
    LOGP(debug, "Inserting {} in {} ", staveVol->GetName(), layerVol->GetName());
    layerVol->AddNode(staveVol, iStave, trans);
  }

  LOGP(debug, "Inserting {} in {} ", layerVol->GetName(), motherVolume->GetName());
  motherVolume->AddNode(layerVol, 1, nullptr);
}

std::pair<float, float> TRKOTLayer::getBoundingRadii(double staveWidth) const
{
  auto [radiusMin, radiusMax] = TRKSegmentedLayer::getBoundingRadii(staveWidth);

  return {radiusMin - 0.201f, radiusMax};
}
// ClassImp(TRKLayer);

} // namespace trk
} // namespace o2
