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

#include "IOTOFSimulation/Layer.h"
#include "IOTOFBase/GeometryTGeo.h"

#include "Framework/Logger.h"

#include <TGeoBBox.h>
#include <TGeoMatrix.h>
#include <TGeoTube.h>
#include <TGeoVolume.h>
#include <TMath.h>

#include <algorithm>
#include <cmath>

namespace o2
{
namespace iotof
{
Layer::Layer(std::string layerName, float rInn, float rOut, float zLength, float zOffset, float layerX2X0,
             int layout, int nStaves, float staveSize, double staveTiltAngle, int modulesPerStave, float sensorThickness)
  : mLayerName(layerName),
    mInnerRadius(rInn),
    mOuterRadius(rOut),
    mZLength(zLength),
    mZOffset(zOffset),
    mSensorThickness(sensorThickness),
    mX2X0(layerX2X0),
    mLayout(layout),
    mStaves(nStaves, staveSize),
    mModulesPerStave(modulesPerStave),
    mTiltAngle(staveTiltAngle)
{
  const float Si_X0 = 9.5f; // cm, radiation length of silicon
  mChipThickness = mX2X0 * Si_X0;
  std::string name = "";
  switch (layout) {
    case kBarrel:
    case kBarrelSegmented:
      name = "barrel";
      mOuterRadius = mInnerRadius + mChipThickness;
      break;
    case kDisk:
    case kDiskSegmented:
      name = "forward";
      mZLength = mChipThickness;
      break;
    default:
      LOG(fatal) << "Invalid layout " << layout;
  }
  // Sanity checks
  if (mInnerRadius > mOuterRadius) {
    LOG(fatal) << "Invalid layer dimensions: rInner " << mInnerRadius << " cm is larger than rOuter " << mOuterRadius << " cm";
  }
  if ((mStaves.first != 0 || mStaves.second != 0.0f) && (layout != kBarrelSegmented && layout != kDiskSegmented)) {
    LOG(fatal) << "Invalid configuration: number of segments " << mStaves.first << " is set for non-segmented layout " << layout;
  }
  if ((mStaves.first <= 1 || mStaves.second <= 0.0f) && (layout == kBarrelSegmented || layout == kDiskSegmented)) {
    LOG(fatal) << "Invalid configuration: number of segments " << mStaves.first << " must be positive for segmented layout " << layout;
  }
  if (mModulesPerStave <= 0 && (layout == kBarrelSegmented || layout == kDiskSegmented)) {
    LOG(fatal) << "Invalid configuration: number of sensors per segment " << mModulesPerStave << " must be positive for segmented layout " << layout;
  }
  if (std::abs(mTiltAngle) > 0.1 && (layout != kBarrelSegmented && layout != kDiskSegmented)) {
    LOG(fatal) << "Invalid configuration: tilt angle " << mTiltAngle << " is set for non-segmented layout " << layout;
  }
  if ((mTiltAngle < 0.0 || mTiltAngle > 90.0) && (layout == kBarrelSegmented || layout == kDiskSegmented)) {
    LOG(fatal) << "Invalid configuration: tilt angle " << mTiltAngle << " is too large, it must be between 0 and 90 degrees";
  }
  if (mSensorThickness < 0.0f || mSensorThickness > mChipThickness) {
    LOG(fatal) << "Invalid configuration: sensor thickness " << mSensorThickness << " cm is out of range (0, " << mChipThickness << ") cm";
  }
  if (sensorThickness > 0.0f && (layout == kBarrel || layout == kDisk)) {
    LOG(fatal) << "Invalid configuration: sensor thickness " << mSensorThickness << " cm is set for non-segmented layout, it should be 0";
  }

  LOGP(info, "TOF: Creating {} layer: rInner: {} (cm) rOuter: {} (cm) zLength: {} (cm) zOffset: {} x2X0: {}", name.c_str(), mInnerRadius, mOuterRadius, mZLength, mZOffset, mX2X0);
}

void setLayerStyle(TGeoVolume* obj)
{
  obj->SetLineColor(kRed - 7);
  obj->SetFillColor(kRed - 7);
  obj->SetLineWidth(1);
  obj->SetTransparency(70);
}
void setStaveStyle(TGeoVolume* obj)
{
  obj->SetLineColor(kRed - 5);
  obj->SetFillColor(kRed - 9);
  obj->SetLineWidth(2);
  obj->SetTransparency(45);
}
void setModuleStyle(TGeoVolume* obj)
{
  obj->SetLineColor(kRed - 3);
  obj->SetFillColor(kRed - 8);
  obj->SetLineWidth(2);
  obj->SetTransparency(35);
}
void setChipStyle(TGeoVolume* obj)
{
  obj->SetLineColor(kOrange);
  obj->SetFillColor(kOrange - 9);
  obj->SetLineWidth(3);
  obj->SetTransparency(15);
}
void setSensorStyle(TGeoVolume* obj)
{
  obj->SetLineColor(kRed);
  obj->SetFillColor(kRed - 9);
  obj->SetLineWidth(3);
  obj->SetTransparency(5);
}

std::vector<std::string> ITOFLayer::mRegister;
void ITOFLayer::createLayer(TGeoVolume* motherVolume)
{
  const char* chipName = o2::iotof::GeometryTGeo::getITOFChipPattern();
  const char* sensName = o2::iotof::GeometryTGeo::getITOFSensorPattern();
  const char* moduleName = o2::iotof::GeometryTGeo::getITOFModulePattern();
  const char* staveName = o2::iotof::GeometryTGeo::getITOFStavePattern();

  TGeoMedium* medSi = gGeoManager->GetMedium("TF3_SILICON$");
  TGeoMedium* medAir = gGeoManager->GetMedium("TF3_AIR$");
  LOGP(info, "Media: {} {}", (void*)medSi, (void*)medAir);

  switch (mLayout) {
    case kBarrel: {
      TGeoTube* sensor = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);
      TGeoTube* chip = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);
      TGeoTube* layer = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);

      TGeoVolume* sensVol = new TGeoVolume(sensName, sensor, medSi);
      TGeoVolume* chipVol = new TGeoVolume(chipName, chip, medSi);
      TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
      setSensorStyle(sensVol);
      setChipStyle(chipVol);
      setLayerStyle(layerVol);

      LOGP(info, "Inserting Barrel {} in {} ", sensVol->GetName(), chipVol->GetName());
      ITOFLayer::mRegister.push_back(sensVol->GetName());
      chipVol->AddNode(sensVol, 1, nullptr);

      LOGP(info, "Inserting Barrel {} in {} ", chipVol->GetName(), layerVol->GetName());
      layerVol->AddNode(chipVol, 1, nullptr);

      LOGP(info, "Inserting Barrel {} in {} ", layerVol->GetName(), motherVolume->GetName());
      motherVolume->AddNode(layerVol, 1, nullptr);
      return;
    }
    case kBarrelSegmented: {
      // First we create the volume for the whole layer, which will be used as mother volume for the segments
      const double avgRadius = 0.5 * (mInnerRadius + mOuterRadius);
      const double staveSizeX = mStaves.second;                                                                                                                    // cm
      const double staveSizeY = mOuterRadius - mInnerRadius;                                                                                                       // cm
      const double staveSizeZ = mZLength;                                                                                                                          // cm
      const double deltaForTilt = 0.5 * (std::sin(TMath::DegToRad() * mTiltAngle) * staveSizeX + std::cos(TMath::DegToRad() * mTiltAngle) * staveSizeY);           // we increase the size of the layer to account for the tilt of the staves
      const double radiusMax = std::sqrt(avgRadius * avgRadius + 0.25 * staveSizeX * staveSizeX + 0.25 * staveSizeY * staveSizeY + avgRadius * 2. * deltaForTilt); // we increase the outer radius to account for the tilt of the staves
      const double radiusMin = std::sqrt(avgRadius * avgRadius + 0.25 * staveSizeX * staveSizeX + 0.25 * staveSizeY * staveSizeY - avgRadius * 2. * deltaForTilt); // we decrease the inner radius to account for the tilt of the staves
      TGeoTube* layer = new TGeoTube(radiusMin - 0.05, radiusMax + 0.05, mZLength / 2);                                                                            // cm, small margins to ensure staves are fully encapsulated in the layer volume
      TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
      setLayerStyle(layerVol);

      // Now we create the volume for a single stave
      TGeoBBox* stave = new TGeoBBox(staveSizeX * 0.5, staveSizeY * 0.5, staveSizeZ * 0.5);
      TGeoVolume* staveVol = new TGeoVolume(staveName, stave, medAir);
      setStaveStyle(staveVol);

      // Now we create the volume for a single module (sensor + chip)
      const int modulesPerStaveX = 1;                           // we assume that each stave is divided in 2 modules along the x direction
      const double moduleSizeX = staveSizeX / modulesPerStaveX; // cm
      const double moduleSizeY = staveSizeY;                    // cm
      const double moduleSizeZ = staveSizeZ / mModulesPerStave; // cm
      TGeoBBox* module = new TGeoBBox(moduleSizeX * 0.5, moduleSizeY * 0.5, moduleSizeZ * 0.5);
      TGeoVolume* moduleVol = new TGeoVolume(moduleName, module, medAir);
      setModuleStyle(moduleVol);

      // Now we create the volume of the chip, which is the same for all modules
      const int chipsPerModuleX = 2;                          // we assume that each module is divided in 2 chips along the x direction
      const int chipsPerModuleZ = 2;                          // we assume that each module is divided in 2 chips along the z direction
      const double chipSizeX = moduleSizeX / chipsPerModuleX; // cm
      const double chipSizeY = moduleSizeY;                   // cm
      const double chipSizeZ = moduleSizeZ / chipsPerModuleZ; // cm
      TGeoBBox* chip = new TGeoBBox(chipSizeX * 0.5, chipSizeY * 0.5, chipSizeZ * 0.5);
      TGeoVolume* chipVol = new TGeoVolume(chipName, chip, medSi);
      setChipStyle(chipVol);

      // Finally we create the volume of the sensor, which is the same for all chips
      const int sensorsPerChipX = 1;                          // we assume that each chip is divided in 2 sensors along the x direction
      const int sensorsPerChipZ = 1;                          // we assume that each chip is divided in 2 sensors along the z direction
      const double sensorSizeX = chipSizeX / sensorsPerChipX; // cm
      const double sensorSizeY = mSensorThickness;            // cm
      const double sensorSizeZ = chipSizeZ / sensorsPerChipZ; // cm
      TGeoBBox* sensor = new TGeoBBox(sensorSizeX * 0.5, sensorSizeY * 0.5, sensorSizeZ * 0.5);
      TGeoVolume* sensVol = new TGeoVolume(sensName, sensor, medSi);
      setSensorStyle(sensVol);
      ITOFLayer::mRegister.push_back(sensVol->GetName());

      // Now we build a chip from sensors
      for (int i = 0; i < sensorsPerChipX; ++i) {
        for (int j = 0; j < sensorsPerChipZ; ++j) {
          LOGP(info, "iTOF: Creating sensor {}/{} for chip {}/{}", i + 1, sensorsPerChipX, j + 1, sensorsPerChipZ);
          auto* translation = new TGeoTranslation((i + 0.5) * sensorSizeX - 0.5 * chipSizeX,
                                                  0,
                                                  (j + 0.5) * sensorSizeZ - 0.5 * chipSizeZ);
          chipVol->AddNode(sensVol, 1 + i * sensorsPerChipZ + j, translation);
        }
      }

      // Now we build a module from chips
      for (int i = 0; i < chipsPerModuleX; ++i) {
        for (int j = 0; j < chipsPerModuleZ; ++j) {
          LOGP(info, "iTOF: Creating chip {}/{} for module {}/{}", i + 1, chipsPerModuleX, j + 1, chipsPerModuleZ);
          auto* translation = new TGeoTranslation((i + 0.5) * chipSizeX - 0.5 * moduleSizeX, 0, (j + 0.5) * chipSizeZ - 0.5 * moduleSizeZ);
          moduleVol->AddNode(chipVol, 1 + i * chipsPerModuleZ + j, translation);
        }
      }

      // Now we build a stave from modules
      for (int i = 0; i < modulesPerStaveX; ++i) {
        for (int j = 0; j < mModulesPerStave; ++j) {
          LOGP(info, "iTOF: Creating module {}/{} for stave {}/{}", i + 1, modulesPerStaveX, j + 1, mModulesPerStave);
          auto* translation = new TGeoTranslation((i + 0.5) * moduleSizeX - 0.5 * staveSizeX, 0, (j + 0.5) * moduleSizeZ - 0.5 * staveSizeZ);
          staveVol->AddNode(moduleVol, 1 + i * mModulesPerStave + j, translation);
        }
      }

      // We finally put all the staves in the layer
      for (int i = 0; i < mStaves.first; ++i) {
        LOGP(info, "iTOF: Creating stave {}/{} for layer {}", i + 1, mStaves.first, layerVol->GetName());
        const double phi = TMath::TwoPi() * i / mStaves.first;
        const double x = avgRadius * TMath::Cos(phi);
        const double y = avgRadius * TMath::Sin(phi);
        auto* rotation = new TGeoRotation(Form("segmentRot%d", i + 1), phi * TMath::RadToDeg() + 90 + mTiltAngle, 0, 0);
        auto* transformation = new TGeoCombiTrans(x, y, 0, rotation);

        LOGP(info, "Inserting Barrel {} in {} ", chipVol->GetName(), layerVol->GetName());
        layerVol->AddNode(staveVol, 1 + i, transformation);
      }
      LOGP(info, "Inserting Barrel {} in {} at r={} cm", layerVol->GetName(), motherVolume->GetName(), avgRadius);
      motherVolume->AddNode(layerVol, 1, nullptr);
      return;
    }
    default:
      LOG(fatal) << "Invalid layout " << mLayout;
  }
}

std::vector<std::string> OTOFLayer::mRegister;
void OTOFLayer::createLayer(TGeoVolume* motherVolume)
{
  const char* chipName = o2::iotof::GeometryTGeo::getOTOFChipPattern();
  const char* sensName = o2::iotof::GeometryTGeo::getOTOFSensorPattern();
  const char* moduleName = o2::iotof::GeometryTGeo::getOTOFModulePattern();
  const char* staveName = o2::iotof::GeometryTGeo::getOTOFStavePattern();

  TGeoMedium* medSi = gGeoManager->GetMedium("TF3_SILICON$");
  TGeoMedium* medAir = gGeoManager->GetMedium("TF3_AIR$");
  LOGP(info, "Media: {} {}", (void*)medSi, (void*)medAir);

  switch (mLayout) {
    case kBarrel: {
      TGeoTube* sensor = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);
      TGeoTube* chip = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);
      TGeoTube* layer = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);

      TGeoVolume* sensVol = new TGeoVolume(sensName, sensor, medSi);
      TGeoVolume* chipVol = new TGeoVolume(chipName, chip, medSi);
      TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
      setSensorStyle(sensVol);
      setChipStyle(chipVol);
      setLayerStyle(layerVol);

      LOGP(info, "Inserting {} in {} ", sensVol->GetName(), chipVol->GetName());
      OTOFLayer::mRegister.push_back(sensVol->GetName());
      chipVol->AddNode(sensVol, 1, nullptr);

      LOGP(info, "Inserting {} in {} ", chipVol->GetName(), layerVol->GetName());
      layerVol->AddNode(chipVol, 1, nullptr);

      LOGP(info, "Inserting {} in {} ", layerVol->GetName(), motherVolume->GetName());
      motherVolume->AddNode(layerVol, 1, nullptr);
      return;
    }
    case kBarrelSegmented: {
      // First we create the volume for the whole layer, which will be used as mother volume for the segments
      const double avgRadius = 0.5 * (mInnerRadius + mOuterRadius);
      const double staveSizeX = mStaves.second;                                                                                                                    // cm
      const double staveSizeY = mOuterRadius - mInnerRadius;                                                                                                       // cm
      const double staveSizeZ = mZLength;                                                                                                                          // cm
      const double deltaForTilt = 0.5 * (std::sin(TMath::DegToRad() * mTiltAngle) * staveSizeX + std::cos(TMath::DegToRad() * mTiltAngle) * staveSizeY);           // we increase the size of the layer to account for the tilt of the staves
      const double radiusMax = std::sqrt(avgRadius * avgRadius + 0.25 * staveSizeX * staveSizeX + 0.25 * staveSizeY * staveSizeY + avgRadius * 2. * deltaForTilt); // we increase the outer radius to account for the tilt of the staves
      const double radiusMin = std::sqrt(avgRadius * avgRadius + 0.25 * staveSizeX * staveSizeX + 0.25 * staveSizeY * staveSizeY - avgRadius * 2. * deltaForTilt); // we decrease the inner radius to account for the tilt of the staves
      TGeoTube* layer = new TGeoTube(radiusMin, radiusMax, mZLength / 2);
      TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
      setLayerStyle(layerVol);

      // Now we create the volume for a single stave
      TGeoBBox* stave = new TGeoBBox(staveSizeX * 0.5, staveSizeY * 0.5, staveSizeZ * 0.5);
      TGeoVolume* staveVol = new TGeoVolume(staveName, stave, medAir);
      setStaveStyle(staveVol);

      // Now we create the volume for a single module (sensor + chip)
      const int modulesPerStaveX = 1;                           // we assume that each stave is divided in 2 modules along the x direction
      const double moduleSizeX = staveSizeX / modulesPerStaveX; // cm
      const double moduleSizeY = staveSizeY;                    // cm
      const double moduleSizeZ = staveSizeZ / mModulesPerStave; // cm
      TGeoBBox* module = new TGeoBBox(moduleSizeX * 0.5, moduleSizeY * 0.5, moduleSizeZ * 0.5);
      TGeoVolume* moduleVol = new TGeoVolume(moduleName, module, medAir);
      setModuleStyle(moduleVol);

      // Now we create the volume of the chip, which is the same for all modules
      const int chipsPerModuleX = 2;                          // we assume that each module is divided in 2 chips along the x direction
      const int chipsPerModuleZ = 2;                          // we assume that each module is divided in 2 chips along the z direction
      const double chipSizeX = moduleSizeX / chipsPerModuleX; // cm
      const double chipSizeY = moduleSizeY;                   // cm
      const double chipSizeZ = moduleSizeZ / chipsPerModuleZ; // cm
      TGeoBBox* chip = new TGeoBBox(chipSizeX * 0.5, chipSizeY * 0.5, chipSizeZ * 0.5);
      TGeoVolume* chipVol = new TGeoVolume(chipName, chip, medSi);
      setChipStyle(chipVol);

      // Finally we create the volume of the sensor, which is the same for all chips
      const int sensorsPerChipX = 1;                          // we assume that each chip is divided in 2 sensors along the x direction
      const int sensorsPerChipZ = 1;                          // we assume that each chip is divided in 2 sensors along the z direction
      const double sensorSizeX = chipSizeX / sensorsPerChipX; // cm
      const double sensorSizeY = mSensorThickness;            // cm
      const double sensorSizeZ = chipSizeZ / sensorsPerChipZ; // cm
      TGeoBBox* sensor = new TGeoBBox(sensorSizeX * 0.5, sensorSizeY * 0.5, sensorSizeZ * 0.5);
      TGeoVolume* sensVol = new TGeoVolume(sensName, sensor, medSi);
      setSensorStyle(sensVol);
      OTOFLayer::mRegister.push_back(sensVol->GetName());

      // Now we build a chip from sensors
      for (int i = 0; i < sensorsPerChipX; ++i) {
        for (int j = 0; j < sensorsPerChipZ; ++j) {
          LOGP(info, "oTOF: Creating sensor {}/{} for chip {}/{}", i + 1, sensorsPerChipX, j + 1, sensorsPerChipZ);
          auto* translation = new TGeoTranslation((i + 0.5) * sensorSizeX - 0.5 * chipSizeX,
                                                  0,
                                                  (j + 0.5) * sensorSizeZ - 0.5 * chipSizeZ);
          chipVol->AddNode(sensVol, 1 + i * sensorsPerChipZ + j, translation);
        }
      }

      // Now we build a module from chips
      for (int i = 0; i < chipsPerModuleX; ++i) {
        for (int j = 0; j < chipsPerModuleZ; ++j) {
          LOGP(info, "oTOF: Creating chip {}/{} for module {}/{}", i + 1, chipsPerModuleX, j + 1, chipsPerModuleZ);
          auto* translation = new TGeoTranslation((i + 0.5) * chipSizeX - 0.5 * moduleSizeX, 0, (j + 0.5) * chipSizeZ - 0.5 * moduleSizeZ);
          moduleVol->AddNode(chipVol, 1 + i * chipsPerModuleZ + j, translation);
        }
      }

      // Now we build a stave from modules
      for (int i = 0; i < modulesPerStaveX; ++i) {
        for (int j = 0; j < mModulesPerStave; ++j) {
          LOGP(info, "oTOF: Creating module {}/{} for stave {}/{}", i + 1, modulesPerStaveX, j + 1, mModulesPerStave);
          auto* translation = new TGeoTranslation((i + 0.5) * moduleSizeX - 0.5 * staveSizeX, 0, (j + 0.5) * moduleSizeZ - 0.5 * staveSizeZ);
          staveVol->AddNode(moduleVol, 1 + i * mModulesPerStave + j, translation);
        }
      }

      // We finally put all the staves in the layer
      for (int i = 0; i < mStaves.first; ++i) {
        LOGP(info, "oTOF: Creating stave {}/{} for layer {}", i + 1, mStaves.first, layerVol->GetName());
        const double phi = TMath::TwoPi() * i / mStaves.first;
        const double x = avgRadius * TMath::Cos(phi);
        const double y = avgRadius * TMath::Sin(phi);
        auto* rotation = new TGeoRotation(Form("segmentRot%d", i + 1), phi * TMath::RadToDeg() + 90 + mTiltAngle, 0, 0);
        auto* transformation = new TGeoCombiTrans(x, y, 0, rotation);

        LOGP(info, "Inserting Barrel {} in {} ", chipVol->GetName(), layerVol->GetName());
        layerVol->AddNode(staveVol, 1 + i, transformation);
      }
      LOGP(info, "Inserting Barrel {} in {} at r={} cm", layerVol->GetName(), motherVolume->GetName(), avgRadius);
      motherVolume->AddNode(layerVol, 1, nullptr);
      return;
    }
    default:
      LOG(fatal) << "Invalid layout " << mLayout;
  }
}

void FTOFLayer::createLayer(TGeoVolume* motherVolume)
{
  std::string chipName = o2::iotof::GeometryTGeo::getFTOFChipPattern(),
              sensName = o2::iotof::GeometryTGeo::getFTOFSensorPattern();

  TGeoTube* sensor = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);
  TGeoTube* chip = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);
  TGeoTube* layer = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);

  TGeoMedium* medSi = gGeoManager->GetMedium("TF3_SILICON$");
  TGeoMedium* medAir = gGeoManager->GetMedium("TF3_AIR$");

  TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
  TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
  setSensorStyle(sensVol);
  setChipStyle(chipVol);
  setLayerStyle(layerVol);

  LOGP(info, "Inserting {} in {} ", sensVol->GetName(), chipVol->GetName());
  chipVol->AddNode(sensVol, 1, nullptr);

  LOGP(info, "Inserting {} in {} ", chipVol->GetName(), layerVol->GetName());
  layerVol->AddNode(chipVol, 1, nullptr);

  auto* fwdTOFRotation = new TGeoRotation("fwdTOFRotation", 0, 0, 180);
  auto* fwdTOFCombiTrans = new TGeoCombiTrans(0, 0, mZOffset, fwdTOFRotation);

  LOGP(info, "Inserting {} in {} ", layerVol->GetName(), motherVolume->GetName());
  motherVolume->AddNode(layerVol, 1, fwdTOFCombiTrans);
}

void BTOFLayer::createLayer(TGeoVolume* motherVolume)
{
  std::string chipName = o2::iotof::GeometryTGeo::getBTOFChipPattern(),
              sensName = o2::iotof::GeometryTGeo::getBTOFSensorPattern();

  TGeoTube* sensor = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);
  TGeoTube* chip = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);
  TGeoTube* layer = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);

  TGeoMedium* medSi = gGeoManager->GetMedium("TF3_SILICON$");
  TGeoMedium* medAir = gGeoManager->GetMedium("TF3_AIR$");

  TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
  TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
  setSensorStyle(sensVol);
  setChipStyle(chipVol);
  setLayerStyle(layerVol);

  LOGP(info, "Inserting {} in {} ", sensVol->GetName(), chipVol->GetName());
  chipVol->AddNode(sensVol, 1, nullptr);

  LOGP(info, "Inserting {} in {} ", chipVol->GetName(), layerVol->GetName());
  layerVol->AddNode(chipVol, 1, nullptr);

  auto* bwdTOFRotation = new TGeoRotation("bwdTOFRotation", 0, 0, 180);
  auto* fwdTOFCombiTrans = new TGeoCombiTrans(0, 0, mZOffset, bwdTOFRotation);

  LOGP(info, "Inserting {} in {} ", layerVol->GetName(), motherVolume->GetName());
  motherVolume->AddNode(layerVol, 1, fwdTOFCombiTrans);
}

} // namespace iotof
} // namespace o2
