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

#include "TRKSimulation/VDLayer.h"
#include "TRKBase/GeometryTGeo.h"
#include "TRKBase/Specs.h"

#include "Framework/Logger.h"

#include "TGeoTube.h"
#include "TGeoBBox.h"
#include "TGeoVolume.h"
#include "TGeoMatrix.h"
#include "TGeoManager.h"

#include "TMath.h"

namespace o2
{
namespace trk
{
// Base layer constructor
VDLayer::VDLayer(int layerNumber, const std::string& layerName, double layerX2X0)
  : mLayerNumber(layerNumber), mLayerName(layerName), mX2X0(layerX2X0), mModuleWidth(4.54)
{
  constexpr double kSiX0_cm = 9.5; // Radiation length of Silicon in cm
  mChipThickness = mX2X0 * kSiX0_cm;

  mSensorThickness = o2::trk::constants::VD::silicon::thickness; // cm
}

// VDCylindricalLayer constructor
VDCylindricalLayer::VDCylindricalLayer(int layerNumber, const std::string& layerName, double layerX2X0, double radius,
                                       double phiSpanDeg, double lengthZ, double lengthSensZ)
  : VDLayer(layerNumber, layerName, layerX2X0), mRadius(radius), mPhiSpanDeg(phiSpanDeg), mLengthZ(lengthZ), mLengthSensZ(lengthSensZ)
{
  LOGP(info, "Creating VD cylindrical layer: id: {} name: {} x2X0: {} radius: {} phiSpanDeg: {} lengthZ: {} lengthSensZ: {} chipThickness = {} cm",
       mLayerNumber, layerName, mX2X0, radius, phiSpanDeg, lengthZ, lengthSensZ, mChipThickness);
}

// VDRectangularLayer constructor
VDRectangularLayer::VDRectangularLayer(int layerNumber, const std::string& layerName, double layerX2X0,
                                       double width, double lengthZ, double lengthSensZ)
  : VDLayer(layerNumber, layerName, layerX2X0), mWidth(width), mLengthZ(lengthZ), mLengthSensZ(lengthSensZ)
{

  if (mLengthSensZ <= 0 || mLengthSensZ > mLengthZ) {
    LOGP(fatal, "Invalid sensor length: sensZ={} layerZ={}", mLengthSensZ, mLengthZ);
  }
  LOGP(info, "Creating VD rectangular layer: id: {} name: {} x2X0: {} width: {} lengthZ: {} lengthSensZ: {} chipThickness = {} cm",
       mLayerNumber, layerName, mX2X0, width, lengthZ, lengthSensZ, mChipThickness);
}

// VDDiskLayer constructor
VDDiskLayer::VDDiskLayer(int layerNumber, const std::string& layerName, double layerX2X0, double rMin, double rMax,
                         double phiSpanDeg, double zPos)
  : VDLayer(layerNumber, layerName, layerX2X0), mRMin(rMin), mRMax(rMax), mPhiSpanDeg(phiSpanDeg), mZPos(zPos)
{

  LOGP(info, "Creating VD disk layer: id: {} name: {} x2X0: {} rMin: {} rMax: {} phiSpanDeg: {} zPos: {} chipThickness = {} cm",
       mLayerNumber, layerName, mX2X0, rMin, rMax, phiSpanDeg, zPos, mChipThickness);
}

/*
** Create sensor
*/

TGeoVolume* VDCylindricalLayer::createSensor() const
{
  if (!gGeoManager) {
    LOGP(error, "gGeoManager is null");
    return nullptr;
  }
  auto* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  if (!medSi) {
    LOGP(error, "Missing medium TRK_SILICON$");
    return nullptr;
  }
  std::string sensName = Form("%s_%s%d", this->mLayerName.c_str(), GeometryTGeo::getTRKSensorPattern(), this->mLayerNumber);
  const double rIn = mRadius;
  const double rOut = mRadius + mSensorThickness;
  const double halfZ = 0.5 * mLengthSensZ;
  const double halfPhi = 0.5 * mPhiSpanDeg; // degrees
  auto* shape = new TGeoTubeSeg(rIn, rOut, halfZ, -halfPhi, +halfPhi);
  auto* vol = new TGeoVolume(sensName.c_str(), shape, medSi);
  vol->SetLineColor(kYellow);
  vol->SetTransparency(30);
  return vol;
}

TGeoVolume* VDRectangularLayer::createSensor() const
{
  if (!gGeoManager) {
    LOGP(error, "gGeoManager is null");
    return nullptr;
  }
  auto* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  if (!medSi) {
    LOGP(error, "Missing medium TRK_SILICON$");
    return nullptr;
  }
  std::string sensName = Form("%s_%s%d", this->mLayerName.c_str(), GeometryTGeo::getTRKSensorPattern(), this->mLayerNumber);
  const double hx = 0.5 * mWidth;
  const double hy = 0.5 * mSensorThickness;
  const double hz = 0.5 * mLengthSensZ; // <-- use sensor Z length, not full layer

  auto* shape = new TGeoBBox(hx, hy, hz);
  auto* vol = new TGeoVolume(sensName.c_str(), shape, medSi);
  vol->SetLineColor(kYellow);
  vol->SetTransparency(30);

  return vol;
}

TGeoVolume* VDDiskLayer::createSensor() const
{
  if (!gGeoManager) {
    LOGP(error, "gGeoManager is null");
    return nullptr;
  }
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  if (!medSi) {
    LOGP(error, "Missing medium TRK_SILICON$");
    return nullptr;
  }
  if (mRMin < 0 || mRMax <= mRMin || mChipThickness <= 0 || mPhiSpanDeg <= 0 || mPhiSpanDeg > 360.0) {
    LOGP(error, "Invalid disk sensor dims: rMin={}, rMax={}, t={}, phiSpanDeg={}",
         mRMin, mRMax, mChipThickness, mPhiSpanDeg);
    return nullptr;
  }
  std::string sensName = Form("%s_%s%d", this->mLayerName.c_str(), GeometryTGeo::getTRKSensorPattern(), this->mLayerNumber);
  const double halfThickness = 0.5 * mSensorThickness; // active sensor thickness along Z
  const double halfPhi = 0.5 * mPhiSpanDeg;            // degrees

  // Same geometry as the layer (identical radii + phi span + thickness)
  auto* shape = new TGeoTubeSeg(mRMin, mRMax, halfThickness, -halfPhi, +halfPhi);

  auto* sensVol = new TGeoVolume(sensName.c_str(), shape, medSi);
  sensVol->SetLineColor(kYellow);
  sensVol->SetTransparency(30);

  return sensVol;
}

/*
** Create metal stack
*/

TGeoVolume* VDCylindricalLayer::createMetalStack() const
{
  if (!gGeoManager) {
    LOGP(error, "gGeoManager is null");
    return nullptr;
  }
  auto* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  if (!medSi) {
    LOGP(error, "Missing medium TRK_SILICON$");
    return nullptr;
  }

  const double metalT = mChipThickness - mSensorThickness;
  if (metalT <= 0) {
    return nullptr; // nothing to add
  }

  std::string name = Form("%s_%s%d", mLayerName.c_str(),
                          GeometryTGeo::getTRKMetalStackPattern(), mLayerNumber);

  const double rIn = mRadius + mSensorThickness;
  const double rOut = mRadius + mChipThickness;
  const double halfZ = 0.5 * mLengthSensZ;
  const double halfPhi = 0.5 * mPhiSpanDeg;

  auto* shape = new TGeoTubeSeg(rIn, rOut, halfZ, -halfPhi, +halfPhi);
  auto* vol = new TGeoVolume(name.c_str(), shape, medSi);
  vol->SetLineColor(kGray);
  vol->SetTransparency(30);
  return vol;
}

TGeoVolume* VDRectangularLayer::createMetalStack() const
{
  if (!gGeoManager) {
    LOGP(error, "gGeoManager is null");
    return nullptr;
  }
  auto* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  if (!medSi) {
    LOGP(error, "Missing medium TRK_SILICON$");
    return nullptr;
  }

  const double metalT = mChipThickness - mSensorThickness;
  if (metalT <= 0) {
    return nullptr;
  }

  std::string name = Form("%s_%s%d", mLayerName.c_str(),
                          GeometryTGeo::getTRKMetalStackPattern(), mLayerNumber);

  const double hx = 0.5 * mWidth;
  const double hy = 0.5 * metalT;
  const double hz = 0.5 * mLengthSensZ;

  auto* shape = new TGeoBBox(hx, hy, hz);
  auto* vol = new TGeoVolume(name.c_str(), shape, medSi);
  vol->SetLineColor(kGray);
  vol->SetTransparency(30);
  return vol;
}

TGeoVolume* VDDiskLayer::createMetalStack() const
{
  if (!gGeoManager) {
    LOGP(error, "gGeoManager is null");
    return nullptr;
  }
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  if (!medSi) {
    LOGP(error, "Missing medium TRK_SILICON$");
    return nullptr;
  }

  const double metalT = mChipThickness - mSensorThickness;
  if (metalT <= 0) {
    return nullptr;
  }

  if (mRMin < 0 || mRMax <= mRMin || mPhiSpanDeg <= 0 || mPhiSpanDeg > 360.0) {
    LOGP(error, "Invalid disk metal dims: rMin={}, rMax={}, metalT={}, phiSpanDeg={}",
         mRMin, mRMax, metalT, mPhiSpanDeg);
    return nullptr;
  }

  std::string name = Form("%s_%s%d", mLayerName.c_str(),
                          GeometryTGeo::getTRKMetalStackPattern(), mLayerNumber);

  const double halfThickness = 0.5 * metalT;
  const double halfPhi = 0.5 * mPhiSpanDeg;

  auto* shape = new TGeoTubeSeg(mRMin, mRMax, halfThickness, -halfPhi, +halfPhi);
  auto* vol = new TGeoVolume(name.c_str(), shape, medSi);
  vol->SetLineColor(kGray);
  vol->SetTransparency(30);
  return vol;
}

/*
** Create chip
*/

TGeoVolume* VDCylindricalLayer::createChip() const
{
  if (!gGeoManager) {
    LOGP(error, "gGeoManager is null");
    return nullptr;
  }
  auto* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  if (!medSi) {
    LOGP(error, "Missing medium TRK_SILICON$");
    return nullptr;
  }

  std::string chipName = Form("%s_%s%d", mLayerName.c_str(),
                              GeometryTGeo::getTRKChipPattern(), mLayerNumber);

  const double rIn = mRadius;
  const double rOut = mRadius + mChipThickness;
  const double halfZ = 0.5 * mLengthSensZ;
  const double halfPhi = 0.5 * mPhiSpanDeg;

  auto* chipShape = new TGeoTubeSeg(rIn, rOut, halfZ, -halfPhi, +halfPhi);
  auto* chipVol = new TGeoVolume(chipName.c_str(), chipShape, medSi);

  // sensor
  if (auto* sensVol = createSensor()) {
    LOGP(debug, "Inserting {} in {} ", sensVol->GetName(), chipVol->GetName());
    chipVol->AddNode(sensVol, 1, nullptr);
  }

  // metal stack
  if (auto* metalVol = createMetalStack()) {
    LOGP(debug, "Inserting {} in {} ", metalVol->GetName(), chipVol->GetName());
    chipVol->AddNode(metalVol, 1, nullptr); // concentric, no translation needed
  }

  chipVol->SetLineColor(kYellow);
  chipVol->SetTransparency(30);
  return chipVol;
}

TGeoVolume* VDRectangularLayer::createChip() const
{
  if (!gGeoManager) {
    LOGP(error, "gGeoManager is null");
    return nullptr;
  }
  auto* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  if (!medSi) {
    LOGP(error, "Missing medium TRK_SILICON$");
    return nullptr;
  }

  std::string chipName = Form("%s_%s%d", mLayerName.c_str(),
                              GeometryTGeo::getTRKChipPattern(), mLayerNumber);

  const double hx = 0.5 * mWidth;
  const double hy = 0.5 * mChipThickness;
  const double hz = 0.5 * mLengthSensZ;

  auto* chipShape = new TGeoBBox(hx, hy, hz);
  auto* chipVol = new TGeoVolume(chipName.c_str(), chipShape, medSi);

  // sensor (place it on the "bottom" side, like TRK)
  if (auto* sensVol = createSensor()) {
    auto* transSens = new TGeoTranslation(0.0, -(mChipThickness - mSensorThickness) / 2, 0.0);
    LOGP(debug, "Inserting {} in {} ", sensVol->GetName(), chipVol->GetName());
    chipVol->AddNode(sensVol, 1, transSens);
  }

  // metal stack (remaining thickness on top)
  if (auto* metalVol = createMetalStack()) {
    auto* transMetal = new TGeoTranslation(0.0, +mSensorThickness / 2, 0.0);
    LOGP(debug, "Inserting {} in {} ", metalVol->GetName(), chipVol->GetName());
    chipVol->AddNode(metalVol, 1, transMetal);
  }

  chipVol->SetLineColor(kYellow);
  chipVol->SetTransparency(30);
  return chipVol;
}

TGeoVolume* VDDiskLayer::createChip() const
{
  if (!gGeoManager) {
    LOGP(error, "gGeoManager is null");
    return nullptr;
  }
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  if (!medSi) {
    LOGP(error, "Missing medium TRK_SILICON$");
    return nullptr;
  }

  if (mRMin < 0 || mRMax <= mRMin || mChipThickness <= 0 ||
      mPhiSpanDeg <= 0 || mPhiSpanDeg > 360.0) {
    LOGP(error, "Invalid disk chip dims: rMin={}, rMax={}, t={}, phi={}",
         mRMin, mRMax, mChipThickness, mPhiSpanDeg);
    return nullptr;
  }

  std::string chipName = Form("%s_%s%d", mLayerName.c_str(),
                              GeometryTGeo::getTRKChipPattern(), mLayerNumber);

  const double halfThickness = 0.5 * mChipThickness;
  const double halfPhi = 0.5 * mPhiSpanDeg;

  auto* chipShape = new TGeoTubeSeg(mRMin, mRMax, halfThickness, -halfPhi, +halfPhi);
  auto* chipVol = new TGeoVolume(chipName.c_str(), chipShape, medSi);
  chipVol->SetLineColor(kYellow);
  chipVol->SetTransparency(30);

  // Sensor slab (sensitive) placed on one side in Z (TRK-like stacking convention)
  if (auto* sensVol = createSensor()) {
    const double zSens = -(mChipThickness - mSensorThickness) / 2.0;
    auto* tSens = new TGeoTranslation(0.0, 0.0, zSens);
    LOGP(debug, "Inserting {} in {} ", sensVol->GetName(), chipVol->GetName());
    chipVol->AddNode(sensVol, 1, tSens);
  }

  // Metal stack slab (non-sensitive), remaining thickness, also silicon
  if (auto* metalVol = createMetalStack()) {
    const double zMetal = +mSensorThickness / 2.0;
    auto* tMetal = new TGeoTranslation(0.0, 0.0, zMetal);
    LOGP(debug, "Inserting {} in {} ", metalVol->GetName(), chipVol->GetName());
    chipVol->AddNode(metalVol, 1, tMetal);
  }

  return chipVol;
}

/*
** Create layer
*/

// Cylindrical layer
void VDCylindricalLayer::createLayer(TGeoVolume* motherVolume, TGeoMatrix* combiTrans) const
{
  if (!motherVolume || !gGeoManager) {
    LOGP(error, "Null motherVolume or gGeoManager");
    return;
  }
  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");
  if (!medAir) {
    LOGP(error, "Missing TRK_AIR$");
    return;
  }

  // Sanity
  if (mRadius <= 0 || mChipThickness <= 0 || mLengthZ <= 0 ||
      mPhiSpanDeg <= 0 || mPhiSpanDeg > 360.0 ||
      mLengthSensZ <= 0 || mLengthSensZ > mLengthZ) {
    LOGP(error, "Invalid cylindrical dimensions: r={}, t={}, Z={}, phi={}, sensZ={}",
         mRadius, mChipThickness, mLengthZ, mPhiSpanDeg, mLengthSensZ);
    return;
  }

  // AIR container (layer)
  const double rIn = mRadius;
  const double rOut = mRadius + mChipThickness;
  const double halfZ = 0.5 * mLengthZ;
  const double halfPhi = 0.5 * mPhiSpanDeg; // degrees

  auto* layerShape = new TGeoTubeSeg(rIn, rOut, halfZ, -halfPhi, +halfPhi);
  auto* layerVol = new TGeoVolume(mLayerName.c_str(), layerShape, medAir);
  layerVol->SetLineColor(kYellow);
  layerVol->SetTransparency(30);

  // Chip volume (must use mLengthSensZ internally)
  TGeoVolume* chipVol = VDCylindricalLayer::createChip();
  if (!chipVol) {
    LOGP(error, "VDCylindricalLayer::createChip() returned null");
    return;
  }
  LOGP(debug, "Inserting {} in {} ", chipVol->GetName(), layerVol->GetName());
  layerVol->AddNode(chipVol, 1, nullptr);

  // Tiling: edge-to-edge if sensor shorter than layer; else single centered
  // const auto zCenters = (mLengthSensZ < mLengthZ)
  // ? centersNoGapZ(mLengthZ, mLengthSensZ)
  // : std::vector<double>{0.0};
  //
  // int copyNo = 1;
  // for (double zc : zCenters) {
  // TGeoTranslation tz(0.0, 0.0, zc);
  // layerVol->AddNode(sensorVol, copyNo++, (zc == 0.0 && zCenters.size() == 1) ? nullptr : &tz);
  // }

  motherVolume->AddNode(layerVol, 1, combiTrans);
}

// Rectangular layer
void VDRectangularLayer::createLayer(TGeoVolume* motherVolume, TGeoMatrix* combiTrans) const
{
  if (!motherVolume || !gGeoManager) {
    LOGP(error, "Null motherVolume or gGeoManager");
    return;
  }
  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");
  if (!medAir) {
    LOGP(error, "Missing TRK_AIR$");
    return;
  }

  if (mWidth <= 0 || mChipThickness <= 0 || mLengthZ <= 0 ||
      mLengthSensZ <= 0 || mLengthSensZ > mLengthZ) {
    LOGP(error, "Invalid rectangular dims: W={}, t={}, Z={}, sensZ={}",
         mWidth, mChipThickness, mLengthZ, mLengthSensZ);
    return;
  }

  // AIR container (layer)
  const double hx = 0.5 * mWidth;
  const double hy = 0.5 * mChipThickness;
  const double hz = 0.5 * mLengthZ;

  auto* layerShape = new TGeoBBox(hx, hy, hz);
  auto* layerVol = new TGeoVolume(mLayerName.c_str(), layerShape, medAir);
  layerVol->SetLineColor(kYellow);
  layerVol->SetTransparency(30);

  // Sensor volume (uses mLengthSensZ internally)
  TGeoVolume* chipVol = VDRectangularLayer::createChip();
  if (!chipVol) {
    LOGP(error, "VDRectangularLayer::chipVol() returned null");
    return;
  }

  LOGP(debug, "Inserting {} in {} ", chipVol->GetName(), layerVol->GetName());
  layerVol->AddNode(chipVol, 1, nullptr);

  // Tiling along Z, edge - to - edge if needed
  // const auto zCenters = (mLengthSensZ < mLengthZ)
  // ? centersNoGapZ(mLengthZ, mLengthSensZ)
  // : std::vector<double>{0.0};
  //
  // int copyNo = 1;
  // for (double zc : zCenters) {
  // TGeoTranslation tz(0.0, 0.0, zc);
  // layerVol->AddNode(sensorVol, copyNo++, (zc == 0.0 && zCenters.size() == 1) ? nullptr : &tz);
  // }

  motherVolume->AddNode(layerVol, 1, combiTrans);
}

// Disk layer
void VDDiskLayer::createLayer(TGeoVolume* motherVolume, TGeoMatrix* combiTrans) const
{
  if (!motherVolume || !gGeoManager) {
    LOGP(error, "Null motherVolume or gGeoManager");
    return;
  }
  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");
  if (!medAir) {
    LOGP(error, "Missing TRK_AIR$");
    return;
  }

  if (mRMin < 0 || mRMax <= mRMin || mChipThickness <= 0 ||
      mPhiSpanDeg <= 0 || mPhiSpanDeg > 360.0) {
    LOGP(error, "Invalid disk dims: rMin={}, rMax={}, t={}, phi={}",
         mRMin, mRMax, mChipThickness, mPhiSpanDeg);
    return;
  }

  // For disks the thickness is along Z and equals mChipThickness
  const double halfThickness = 0.5 * mChipThickness;
  const double halfPhi = 0.5 * mPhiSpanDeg;

  // AIR container (layer)
  auto* layerShape = new TGeoTubeSeg(mRMin, mRMax, halfThickness, -halfPhi, +halfPhi);
  auto* layerVol = new TGeoVolume(mLayerName.c_str(), layerShape, medAir);
  layerVol->SetLineColor(kYellow);
  layerVol->SetTransparency(30);

  // Sensor (same size & shape as the layer for disks)
  TGeoVolume* chipVol = VDDiskLayer::createChip();
  if (!chipVol) {
    LOGP(error, "VDDiskLayer::createChip() returned null");
    return;
  }

  // Insert single sensor (no Z-segmentation for disks)
  layerVol->AddNode(chipVol, 1, nullptr);

  TGeoTranslation tz(0.0, 0.0, mZPos);
  motherVolume->AddNode(layerVol, 1, combiTrans ? combiTrans : &tz);
}

// ClassImp(VDLayer);
// ClassImp(VDCylindricalLayer);
// ClassImp(VDRectangularLayer);
// ClassImp(VDDiskLayer);

} // namespace trk
} // namespace o2