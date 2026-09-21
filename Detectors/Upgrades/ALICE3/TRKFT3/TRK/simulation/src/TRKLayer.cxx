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
#include "TRKBase/TRKBaseParam.h"
#include <TGeoBBox.h>
#include <TGeoCompositeShape.h>
#include <TGeoTube.h>
#include <TGeoVolume.h>
#include <TMath.h>

#include <algorithm>
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
  TGeoVolume* deadVol = createDeadzone();
  TGeoVolume* metalVol = createMetalStack();
  TGeoCombiTrans* transSens = new TGeoCombiTrans();
  TGeoCombiTrans* transDead = new TGeoCombiTrans();
  TGeoCombiTrans* transMetal = new TGeoCombiTrans();

  const double sensY = mIsFlipped ? -(mChipThickness - sSensorThickness) / 2 : (mChipThickness - sSensorThickness) / 2;
  const double metalY = mIsFlipped ? sSensorThickness / 2 : -sSensorThickness / 2;
  transSens->SetTranslation(-sDeadzoneWidth / 2, sensY, 0);
  transDead->SetTranslation((sChipWidth - sDeadzoneWidth) / 2, sensY, 0);
  transMetal->SetTranslation(0, metalY, 0);

  chipVol->AddNode(sensVol, 1, transSens);
  chipVol->AddNode(deadVol, 1, transDead);
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
  float lengthHalfBarrel = mLength / 2;
  TGeoShape* halfStave = new TGeoBBox(sHalfStaveWidth / 2, mChipThickness / 2, lengthHalfBarrel / 2);
  TGeoVolume* halfStaveVol = new TGeoVolume(halfStaveName.c_str(), halfStave, medSi);
  halfStaveVol->SetLineColor(kYellow);

  int nModulesPerHalfBarrel = mNumberOfModules / 2;
  for (int iModule = 0; iModule < nModulesPerHalfBarrel; iModule++) {
    double zPos = -0.5 * nModulesPerHalfBarrel * sModuleLength + (iModule + 0.5) * sModuleLength;
    TGeoCombiTrans* trans = new TGeoCombiTrans();
    trans->SetTranslation(0, 0, zPos);
    halfStaveVol->AddNode(createModule(), iModule, trans);
  }

  return halfStaveVol;
}

TGeoVolume* TRKOTLayer::createStave()
{
  std::string staveName = GeometryTGeo::getTRKStavePattern() + std::to_string(mLayerNumber);
  TGeoVolume* staveVol = new TGeoVolumeAssembly(staveName.c_str());

  TGeoCombiTrans* transLeft = new TGeoCombiTrans();
  transLeft->SetTranslation(-(sHalfStaveWidth - sInStaveOverlap) / 2, 0, 0);
  staveVol->AddNode(createHalfStave(), 0, transLeft);

  TGeoCombiTrans* transRight = new TGeoCombiTrans();
  transRight->SetTranslation((sHalfStaveWidth - sInStaveOverlap) / 2, 0.2, 0);
  staveVol->AddNode(createHalfStave(), 1, transRight);

  return staveVol;
}

void TRKOTLayer::createLayer(TGeoVolume* motherVolume)
{
  auto [rMin, rMax] = getBoundingRadii(sStaveWidth);

  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");
  TGeoTube* layer = new TGeoTube(rMin, rMax, (mLength + sGapBetweenOuterTrackerBarrelHalves) / 2);
  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
  layerVol->SetLineColor(kYellow);

  int nStavesHalfBarrel = (int)std::ceil(mInnerRadius * 2 * TMath::Pi() / sStaveWidth);
  nStavesHalfBarrel += nStavesHalfBarrel % 2;

  const double avgRadius = 0.5 * (mInnerRadius + mOuterRadius);
  const double theta = 2. * TMath::Pi() / nStavesHalfBarrel;
  const float lengthHalfBarrel = mLength / 2;
  const int nStaves = nStavesHalfBarrel * 2;
  LOGP(info, "Creating OT layer {} with two half-barrels of {} staves each", mLayerNumber, nStavesHalfBarrel);

  for (int iStave = 0; iStave < nStaves; iStave++) {
    int whichHalfBarrel = iStave / nStavesHalfBarrel;
    double phi = theta * iStave;
    TGeoRotation* rot = new TGeoRotation("rot");
    if (whichHalfBarrel == 1) {
      rot->RotateY(180.);
    }
    rot->RotateZ(phi * TMath::RadToDeg() + 90 + (whichHalfBarrel == 0 ? +1 : -1) * mTiltAngle);
    double zPos = (whichHalfBarrel == 0 ? -1 : 1) * (0.5 * lengthHalfBarrel + sGapBetweenOuterTrackerBarrelHalves / 2);
    TGeoCombiTrans* trans = new TGeoCombiTrans();
    trans->SetRotation(rot);
    trans->SetTranslation(avgRadius * std::cos(phi), avgRadius * std::sin(phi), zPos);
    layerVol->AddNode(createStave(), iStave, trans);
  }

  motherVolume->AddNode(layerVol, 1, nullptr);
}

std::pair<float, float> TRKOTLayer::getBoundingRadii(double staveWidth) const
{
  auto [radiusMin, radiusMax] = TRKSegmentedLayer::getBoundingRadii(staveWidth);
  return {radiusMin - 0.201f, radiusMax};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TRKOTLayerRealistic::TRKOTLayerRealistic(int layerNumber, std::string layerName, float rInn, float tiltAngle, int numberOfStaves, int numberOfModules, float thickOrX2X0, MatBudgetParamMode mode)
  : TRKSegmentedLayer(layerNumber, layerName, rInn, tiltAngle, numberOfStaves, numberOfModules, thickOrX2X0, mode)
{
  // Outermost layer is flipped: cooling pipe and support rings on the inner side.
  if (mLayerNumber == constants::ML::nLayers + 2) {
    mIsFlipped = true;
  }
}

TGeoVolume* TRKOTLayerRealistic::createChip()
{
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  std::string chipName = GeometryTGeo::getTRKChipPattern() + std::to_string(mLayerNumber);
  TGeoShape* chip = new TGeoBBox(sChipWidth / 2, constants::OT::sensorThickness / 2, sChipLength / 2);
  TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
  chipVol->SetLineColor(kYellow);

  // Active sensor and passive read-out edge tile the chip width.
  chipVol->AddNode(createSensor(), 1, new TGeoTranslation(-sDeadzoneWidth / 2, 0, 0));
  chipVol->AddNode(createDeadzone(), 1, new TGeoTranslation((sChipWidth - sDeadzoneWidth) / 2, 0, 0));
  return chipVol;
}

TGeoVolume* TRKOTLayerRealistic::createFPC()
{
  TGeoMedium* med = gGeoManager->GetMedium("TRK_FPC$");
  TGeoShape* shape = new TGeoBBox(constants::OT::fpc::width / 2, constants::OT::fpc::thickness / 2, constants::OT::fpc::length / 2);
  TGeoVolume* vol = new TGeoVolume((GeometryTGeo::getTRKModulePattern() + std::to_string(mLayerNumber) + "_FPC").c_str(), shape, med);
  vol->SetLineColor(kOrange);
  return vol;
}

TGeoVolume* TRKOTLayerRealistic::createColdPlate()
{
  TGeoMedium* med = gGeoManager->GetMedium("TRK_CARBONFIBER$");
  TGeoShape* shape = new TGeoBBox(constants::OT::coldPlate::width / 2, constants::OT::coldPlate::thickness / 2, constants::OT::coldPlate::length / 2);
  TGeoVolume* vol = new TGeoVolume((GeometryTGeo::getTRKModulePattern() + std::to_string(mLayerNumber) + "_ColdPlate").c_str(), shape, med);
  vol->SetLineColor(kGray + 2);
  return vol;
}

double TRKOTLayerRealistic::getRowHalfLength() const
{
  const int nModulesPerRow = mNumberOfModules / 2;
  return (nModulesPerRow * constants::OT::fpc::length + (nModulesPerRow - 1) * constants::OT::interModuleGap) / 2;
}

double TRKOTLayerRealistic::getPipeTrim() const
{
  // The mid-rapidity ring sits at the pipe radius, between the z = 0 wall and the pipe.
  const double wallThickness = TRKBaseParam::Instance().otBarrelWallThickness;
  const double pipeStart = wallThickness + 2 * constants::OT::supportRing::zClearance + constants::OT::supportRing::zWidth;
  return std::max(0., pipeStart - constants::OT::barrelHalvesZGap / 2);
}

TGeoVolume* TRKOTLayerRealistic::createSupportRing(double rMin, double rMax, double phi1, double phi2, int id)
{
  // Hollow rectangular-section half-ring, open at the two azimuthal ends.
  TGeoMedium* med = gGeoManager->GetMedium("TRK_CARBONFIBER$");
  const double t = constants::OT::supportRing::wallThickness;
  const double dz = constants::OT::supportRing::zWidth / 2;
  const std::string base = GeometryTGeo::getTRKLayerPattern() + std::to_string(mLayerNumber) + "_SupportRing" + std::to_string(id);
  new TGeoTubeSeg((base + "_outsh").c_str(), rMin, rMax, dz, phi1, phi2);
  new TGeoTubeSeg((base + "_insh").c_str(), rMin + t, rMax - t, dz - t, phi1, phi2);
  TGeoShape* shape = new TGeoCompositeShape((base + "sh").c_str(), (base + "_outsh-" + base + "_insh").c_str());
  TGeoVolume* vol = new TGeoVolume(base.c_str(), shape, med);
  vol->SetLineColor(kGray + 2);
  return vol;
}

TGeoVolume* TRKOTLayerRealistic::createCoolingPipe()
{
  TGeoMedium* med = gGeoManager->GetMedium("TRK_CARBONFIBER$");
  TGeoShape* tube = new TGeoTube(constants::OT::coolingPipe::rInner, constants::OT::coolingPipe::rOuter,
                                 getRowHalfLength() - getPipeTrim() / 2);
  TGeoVolume* vol = new TGeoVolume((GeometryTGeo::getTRKStavePattern() + std::to_string(mLayerNumber) + "_CoolingPipe").c_str(), tube, med);
  vol->SetLineColor(kBlue + 2);
  return vol;
}

TGeoVolume* TRKOTLayerRealistic::createEndOfStaveCard()
{
  TGeoMedium* medFR4 = gGeoManager->GetMedium("TRK_FR4$");
  TGeoMedium* medCu = gGeoManager->GetMedium("TRK_COPPER$");
  const std::string name = GeometryTGeo::getTRKStavePattern() + std::to_string(mLayerNumber) + "_EOSCard";

  TGeoShape* board = new TGeoBBox(constants::OT::eosCard::width / 2, constants::OT::eosCard::thickness / 2, constants::OT::eosCard::length / 2);
  TGeoVolume* cardVol = new TGeoVolume(name.c_str(), board, medFR4);
  cardVol->SetLineColor(kGreen + 3);

  // Copper thickness is configurable: it displaces FR4 inside the fixed board envelope and
  // is what sets the card material budget, so it is the knob for x/X0 scans.
  const double cuThickness = TRKBaseParam::Instance().otEosCardCuThickness;
  const int nPlanes = constants::OT::eosCard::nCopperLayers;
  if (cuThickness * nPlanes >= constants::OT::eosCard::thickness) {
    LOGP(fatal, "TRKBase.otEosCardCuThickness = {} cm x {} planes does not fit in the {} cm end-of-stave card",
         cuThickness, nPlanes, constants::OT::eosCard::thickness);
  }
  TGeoShape* plane = new TGeoBBox(constants::OT::eosCard::width / 2, cuThickness / 2, constants::OT::eosCard::length / 2);
  TGeoVolume* planeVol = new TGeoVolume((name + "_Cu").c_str(), plane, medCu);
  planeVol->SetLineColor(kOrange + 7);

  // Evenly spaced, the outermost two flush with the board surfaces.
  const double span = constants::OT::eosCard::thickness - cuThickness;
  for (int iPlane = 0; iPlane < nPlanes; iPlane++) {
    const double y = (nPlanes > 1) ? -span / 2 + iPlane * span / (nPlanes - 1) : 0.;
    cardVol->AddNode(planeVol, iPlane, new TGeoTranslation(0, y, 0));
  }

  return cardVol;
}

void TRKOTLayerRealistic::addConnector(TGeoVolume* moduleVol, double rMid)
{
  TGeoMedium* med = gGeoManager->GetMedium("TRK_LCPCU$");
  std::string name = GeometryTGeo::getTRKModulePattern() + std::to_string(mLayerNumber) + "_Connector";
  TGeoShape* shape = new TGeoBBox(constants::OT::connector::width / 2, constants::OT::connector::thickness / 2, constants::OT::connector::length / 2);
  TGeoVolume* vol = new TGeoVolume(name.c_str(), shape, med);
  vol->SetLineColor(kBlue);

  // Centred in phi, inset from the module short edge in z.
  const double z = constants::OT::fpc::length / 2 - constants::OT::connectorZDepth;
  moduleVol->AddNode(vol, 0, new TGeoTranslation(0, rMid, z));
}

void TRKOTLayerRealistic::addCapacitors(TGeoVolume* moduleVol, double rMid)
{
  TGeoMedium* med = gGeoManager->GetMedium("TRK_BATIO3$");
  std::string name = GeometryTGeo::getTRKModulePattern() + std::to_string(mLayerNumber) + "_Cap";
  TGeoShape* shape = new TGeoBBox(constants::OT::capacitor::width / 2, constants::OT::capacitor::thickness / 2, constants::OT::capacitor::length / 2);
  TGeoVolume* vol = new TGeoVolume(name.c_str(), shape, med);
  vol->SetLineColor(kCyan);

  const double pitchX = sChipWidth + constants::OT::interChipGap;
  const double pitchZ = sChipLength + constants::OT::interChipGap;
  const double chipX[2] = {-0.5 * pitchX, +0.5 * pitchX};
  const double chipZ[4] = {-1.5 * pitchZ, -0.5 * pitchZ, +0.5 * pitchZ, +1.5 * pitchZ};
  const double dX[5] = {-0.80, +0.80, -0.80, +0.80, 0.0}; // per chip: 4 corners + centre [cm]
  const double dZ[5] = {-0.95, -0.95, +0.95, +0.95, 0.0};

  // Skip capacitors that fall under the connector footprint (+1 mm clearance).
  const double connZ = constants::OT::fpc::length / 2 - constants::OT::connectorZDepth;
  const double skipX = constants::OT::connector::width / 2 + constants::OT::capacitor::width / 2 + 0.1;
  const double skipZ = constants::OT::connector::length / 2 + constants::OT::capacitor::length / 2 + 0.1;

  int capCopy = 0;
  for (int iZ = 0; iZ < 4; iZ++) {
    for (int iX = 0; iX < 2; iX++) {
      for (int iCap = 0; iCap < constants::OT::capacitor::perChip; iCap++) {
        const double x = chipX[iX] + dX[iCap];
        const double z = chipZ[iZ] + dZ[iCap];
        if (std::abs(x) < skipX && std::abs(z - connZ) < skipZ) {
          continue;
        }
        moduleVol->AddNode(vol, capCopy++, new TGeoTranslation(x, rMid, z));
      }
    }
  }
}

void TRKOTLayerRealistic::addBrackets(TGeoVolume* moduleVol, double rMid)
{
  TGeoMedium* med = gGeoManager->GetMedium("TRK_PEEK$");
  std::string name = GeometryTGeo::getTRKModulePattern() + std::to_string(mLayerNumber) + "_Bracket";
  TGeoShape* shape = new TGeoBBox(constants::OT::bracket::width / 2, constants::OT::bracket::thickness / 2, constants::OT::bracket::length / 2);
  TGeoVolume* vol = new TGeoVolume(name.c_str(), shape, med);
  vol->SetLineColor(kGreen + 2);

  const double z = constants::OT::coldPlate::length / 2 - constants::OT::bracketZDepth;
  moduleVol->AddNode(vol, 0, new TGeoTranslation(0, rMid, -z));
  moduleVol->AddNode(vol, 1, new TGeoTranslation(0, rMid, +z));
}

TGeoVolume* TRKOTLayerRealistic::createModule()
{
  std::string modName = GeometryTGeo::getTRKModulePattern() + std::to_string(mLayerNumber);
  TGeoVolume* moduleVol = new TGeoVolumeAssembly(modName.c_str());

  // Flush component stack about the chip mid-plane (local r = 0).
  const double chipHalf = constants::OT::sensorThickness / 2;
  const double fpcMidY = -(chipHalf + constants::OT::fpc::thickness / 2);
  const double coldPlateMidY = +(chipHalf + constants::OT::coldPlate::thickness / 2);
  const double connMidY = -(chipHalf + constants::OT::fpc::thickness + constants::OT::connector::thickness / 2);
  const double capMidY = -(chipHalf + constants::OT::fpc::thickness + constants::OT::capacitor::thickness / 2);
  const double bracketMidY = +(chipHalf + constants::OT::coldPlate::thickness + constants::OT::bracket::thickness / 2);

  // 8 chips: 2 phi columns x 4 z rows, on a uniform chip+gap pitch.
  const double pitchX = sChipWidth + constants::OT::interChipGap;
  const double pitchZ = sChipLength + constants::OT::interChipGap;
  const double chipX[2] = {-0.5 * pitchX, +0.5 * pitchX};
  const double chipZ[4] = {-1.5 * pitchZ, -0.5 * pitchZ, +0.5 * pitchZ, +1.5 * pitchZ};

  moduleVol->AddNode(createColdPlate(), 0, new TGeoTranslation(0, coldPlateMidY, 0));

  int chipCopy = 0;
  for (int iZ = 0; iZ < 4; iZ++) {
    for (int iX = 0; iX < 2; iX++) {
      TGeoCombiTrans* trans = new TGeoCombiTrans();
      trans->SetTranslation(chipX[iX], 0., chipZ[iZ]);
      if (iX == 0) { // inner column rotated so its dead zone faces the outer module edge
        TGeoRotation* rot = new TGeoRotation();
        rot->RotateY(180.);
        trans->SetRotation(rot);
      }
      moduleVol->AddNode(createChip(), chipCopy++, trans);
    }
  }

  moduleVol->AddNode(createFPC(), 0, new TGeoTranslation(0, fpcMidY, 0));
  addConnector(moduleVol, connMidY);
  addCapacitors(moduleVol, capMidY);
  addBrackets(moduleVol, bracketMidY);
  return moduleVol;
}

TGeoVolume* TRKOTLayerRealistic::createHalfStave()
{
  std::string rowName = GeometryTGeo::getTRKHalfStavePattern() + std::to_string(mLayerNumber);
  TGeoVolume* rowVol = new TGeoVolumeAssembly(rowName.c_str());

  const int nModulesPerRow = mNumberOfModules / 2;
  const double moduleLength = constants::OT::fpc::length;
  const double step = moduleLength + constants::OT::interModuleGap;
  const double rowHalfLen = getRowHalfLength();

  for (int iModule = 0; iModule < nModulesPerRow; iModule++) {
    double zPos = -rowHalfLen + moduleLength / 2 + iModule * step;
    TGeoCombiTrans* trans = new TGeoCombiTrans();
    trans->SetTranslation(0, 0, zPos);
    rowVol->AddNode(createModule(), iModule, trans);
  }

  return rowVol;
}

TGeoVolume* TRKOTLayerRealistic::createStave()
{
  std::string staveName = GeometryTGeo::getTRKStavePattern() + std::to_string(mLayerNumber);
  TGeoVolume* staveVol = new TGeoVolumeAssembly(staveName.c_str());

  // Two rows overlapping in phi and staggered in r. They straddle the stave origin, so the
  // stave is tangent to the barrel circle at its centre and every row-to-row radial step,
  // within a stave and between neighbours, is rowRadialStagger.
  const double edgeDead = constants::moduleMLOT::gaps::outerEdgeLongSide + constants::moduleMLOT::chip::passiveEdgeReadOut;
  const double inStaveOverlap = 2 * edgeDead + constants::OT::rowActiveOverlap;
  const double rowOffset = constants::OT::fpc::width - inStaveOverlap;

  TGeoCombiTrans* tRow0 = new TGeoCombiTrans();
  tRow0->SetTranslation(-rowOffset / 2, 0, 0);
  staveVol->AddNode(createHalfStave(), 0, tRow0);
  TGeoCombiTrans* tRow1 = new TGeoCombiTrans();
  tRow1->SetTranslation(rowOffset / 2, constants::OT::rowRadialStagger, 0);
  staveVol->AddNode(createHalfStave(), 1, tRow1);

  // Shortened at the mid-rapidity end for the support ring, hence off-centre.
  TGeoCombiTrans* tPipe = new TGeoCombiTrans();
  tPipe->SetTranslation(0, constants::OT::coolingPipe::rLocalOffset, getPipeTrim() / 2);
  staveVol->AddNode(createCoolingPipe(), 0, tPipe);

  // Past the last module at the outer z end (local +z in both eta half-barrels).
  TGeoCombiTrans* tCard = new TGeoCombiTrans();
  tCard->SetTranslation(0, constants::OT::rowRadialStagger / 2,
                        getRowHalfLength() + constants::OT::eosCard::zGap + constants::OT::eosCard::length / 2);
  staveVol->AddNode(createEndOfStaveCard(), 0, tCard);
  return staveVol;
}

void TRKOTLayerRealistic::createLayer(TGeoVolume* motherVolume)
{
  const double edgeDead = constants::moduleMLOT::gaps::outerEdgeLongSide + constants::moduleMLOT::chip::passiveEdgeReadOut;
  const double inStaveOverlap = 2 * edgeDead + constants::OT::rowActiveOverlap;
  const double staveWidth = 2 * constants::OT::fpc::width - inStaveOverlap;

  // One eta half-barrel = one row of modules, length set by the FPC.
  const double lengthHalfBarrel = 2 * getRowHalfLength();

  // The envelope reaches past the last module to hold the end-of-stave cards.
  const double halfLength = lengthHalfBarrel + constants::OT::barrelHalvesZGap / 2 + constants::OT::eosCard::zGap + constants::OT::eosCard::length;

  // Cut on the vertical plane (x = 0) and at mid-rapidity into four quarter barrels. The
  // envelope is slotted along both cuts so the separation walls run continuously in r; the
  // slots clear the walls only, the staves stand back by barrelWallClearance.
  const double wallThickness = TRKBaseParam::Instance().otBarrelWallThickness;
  const bool hasWalls = wallThickness > 0.;
  const double slotHalfWidth = wallThickness / 2 + constants::OT::barrelWallSlotMargin;
  // The two mid-rapidity walls sit back to back, so the z slot must clear both.
  const double zSlotHalfWidth = wallThickness + constants::OT::barrelWallSlotMargin;
  if (hasWalls && zSlotHalfWidth >= constants::OT::barrelHalvesZGap / 2) {
    LOGP(fatal, "TRKBase.otBarrelWallThickness = {} cm leaves no room for the staves in the {} cm gap between the eta half-barrels",
         wallThickness, constants::OT::barrelHalvesZGap);
  }

  auto [rMin, rMax] = getBoundingRadii(staveWidth);
  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");
  TGeoShape* layer = nullptr;
  if (hasWalls) {
    const std::string tubeName = mLayerName + "_envelopesh";
    const std::string slotName = mLayerName + "_wallslotsh";
    const std::string zSlotName = mLayerName + "_midslotsh";
    new TGeoTube(tubeName.c_str(), rMin, rMax, halfLength);
    new TGeoBBox(slotName.c_str(), slotHalfWidth, rMax + 1., halfLength + 1.);
    new TGeoBBox(zSlotName.c_str(), rMax + 1., rMax + 1., zSlotHalfWidth);
    layer = new TGeoCompositeShape((mLayerName + "sh").c_str(), (tubeName + "-" + slotName + "-" + zSlotName).c_str());
  } else {
    layer = new TGeoTube(rMin, rMax, halfLength);
  }
  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
  layerVol->SetLineColor(kYellow);

  const double avgRadius = 0.5 * (mInnerRadius + mOuterRadius);

  // Arc lost at each of the two azimuthal cuts: the wall plus the passive stave edge,
  // never less than the bare chip-to-chip gap.
  const double accGap = std::max(constants::OT::halfBarrelChipGap + 2 * constants::moduleMLOT::chip::passiveEdgeReadOut,
                                 2 * (wallThickness / 2 + constants::OT::barrelWallClearance + edgeDead));

  // Smallest even count still leaving rowActiveOverlap between neighbours: the two boundary
  // staves take activeStaveWidth + accGap each, so only nStaves - 2 junctions share the rest.
  const double activeStaveWidth = staveWidth - 2 * edgeDead;
  int nStavesHalfBarrel = (int)std::ceil(2. + (avgRadius * 2 * TMath::Pi() - 2 * (activeStaveWidth + accGap)) /
                                                (activeStaveWidth - constants::OT::rowActiveOverlap));
  nStavesHalfBarrel += nStavesHalfBarrel % 2;

  const int nHalf = nStavesHalfBarrel / 2;
  const double thetaGap = (activeStaveWidth + accGap) / avgRadius;
  const double thetaInt = (2. * TMath::Pi() - 2. * thetaGap) / (nStavesHalfBarrel - 2);
  const double overlap = activeStaveWidth - avgRadius * thetaInt;
  LOGP(info, "Creating realistic OT layer {}: {} staves/half-barrel, internal overlap {} mm, boundary gap {} mm, flipped={}",
       mLayerNumber, nStavesHalfBarrel, overlap * 10, accGap * 10, mIsFlipped);

  const int nStaves = nStavesHalfBarrel * 2;

  for (int iStave = 0; iStave < nStaves; iStave++) {
    int whichHalfBarrel = iStave / nStavesHalfBarrel;
    int sInHB = iStave % nStavesHalfBarrel;
    int azHalf = sInHB / nHalf;
    int sInAz = sInHB % nHalf;

    // Stave centres placed so the boundary gaps land on the cut plane, keeping the
    // region where the beam-pipe supports run clear of staves in both half-barrels.
    const double phiCut = TMath::Pi() / 2;
    double phi = phiCut + azHalf * TMath::Pi() + thetaGap / 2 + sInAz * thetaInt;

    TGeoRotation* rot = new TGeoRotation("rot");
    rot->RotateX(180.); // cooling pipe faces the larger-R side (inner for the flipped layer); keeps local phi
    if (whichHalfBarrel == 1) {
      rot->RotateY(180.);
    }
    if (mIsFlipped) {
      rot->RotateZ(180.);
    }
    rot->RotateZ(phi * TMath::RadToDeg() + 90 + (whichHalfBarrel == 0 ? +1 : -1) * mTiltAngle);

    double zPos = (whichHalfBarrel == 0 ? -1 : 1) * (0.5 * lengthHalfBarrel + constants::OT::barrelHalvesZGap / 2);
    TGeoCombiTrans* trans = new TGeoCombiTrans();
    trans->SetRotation(rot);
    trans->SetTranslation(avgRadius * std::cos(phi), avgRadius * std::sin(phi), zPos);
    layerVol->AddNode(createStave(), iStave, trans);
  }

  // Support half-rings carrying the stave space frames, centred on the cooling pipe radius.
  // One per quarter barrel per z end (mid-rapidity and under the end-of-stave cards): 8 per layer.
  const double ringRMid = avgRadius + (mIsFlipped ? -1. : 1.) * constants::OT::coolingPipe::rLocalOffset;
  const double ringRMin = ringRMid - constants::OT::supportRing::radialHeight / 2;
  const double ringRMax = ringRMid + constants::OT::supportRing::radialHeight / 2;
  // Stand off the cut plane by the same clearance the boundary staves keep.
  const double ringDPhi = TMath::RadToDeg() *
                          std::asin((wallThickness / 2 + constants::OT::barrelWallClearance) / ringRMin);
  const double zRingMid = wallThickness + constants::OT::supportRing::zClearance +
                          constants::OT::supportRing::zWidth / 2;
  const double zRingEos = lengthHalfBarrel + constants::OT::barrelHalvesZGap / 2 +
                          constants::OT::eosCard::zGap + constants::OT::eosCard::length / 2;

  for (int azHalf = 0; azHalf < 2; ++azHalf) {
    TGeoVolume* ringVol = createSupportRing(ringRMin, ringRMax,
                                            90. + 180. * azHalf + ringDPhi,
                                            270. + 180. * azHalf - ringDPhi, azHalf);
    int iRing = 0;
    for (int whichHalfBarrel = 0; whichHalfBarrel < 2; ++whichHalfBarrel) {
      const double zSign = (whichHalfBarrel == 0) ? -1. : 1.;
      for (double zAbs : {zRingMid, zRingEos}) {
        layerVol->AddNode(ringVol, iRing++, new TGeoTranslation(0., 0., zSign * zAbs));
      }
    }
  }

  motherVolume->AddNode(layerVol, 1, nullptr);
}

std::pair<float, float> TRKOTLayerRealistic::getBoundingRadii(double staveWidth) const
{
  auto [radiusMin, radiusMax] = TRKSegmentedLayer::getBoundingRadii(staveWidth);
  const float connectorReach = constants::OT::sensorThickness / 2 + constants::OT::fpc::thickness + constants::OT::connector::thickness;
  const float pipeOuterReach = constants::OT::coolingPipe::rLocalOffset + constants::OT::coolingPipe::rOuter;
  const float ringReach = constants::OT::coolingPipe::rLocalOffset + constants::OT::supportRing::radialHeight / 2;
  const float outerReach = std::max(pipeOuterReach, ringReach);
  const float margin = 0.1f;
  if (!mIsFlipped) {
    return {radiusMin - connectorReach - margin, radiusMax + outerReach + margin};
  }
  return {radiusMin - outerReach - margin, radiusMax + connectorReach + margin};
}
// ClassImp(TRKLayer);

} // namespace trk
} // namespace o2
