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
Layer::Layer(std::string layerName, float rInn, float rOut, float zLength, float zOffset, float layerX2X0, int layout, int nSegments, float segmentSize, int nSensorsPerSegment, double tiltAngle)
  : mLayerName(layerName),
    mInnerRadius(rInn),
    mOuterRadius(rOut),
    mZLength(zLength),
    mZOffset(zOffset),
    mX2X0(layerX2X0),
    mLayout(layout),
    mSegments(nSegments, segmentSize),
    mSensorsPerSegment(nSensorsPerSegment),
    mTiltAngle(tiltAngle)
{
  float Si_X0 = 9.5f;
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
  if (1) { // Sanity checks
    if (mInnerRadius > mOuterRadius) {
      LOG(fatal) << "Invalid layer dimensions: rInner " << mInnerRadius << " cm is larger than rOuter " << mOuterRadius << " cm";
    }
    if ((mSegments.first != 0 || mSegments.second != 0.0f) && (layout != kBarrelSegmented && layout != kDiskSegmented)) {
      LOG(fatal) << "Invalid configuration: number of segments " << mSegments.first << " is set for non-segmented layout " << layout;
    }
    if ((mSegments.first <= 1 || mSegments.second <= 0.0f) && (layout == kBarrelSegmented || layout == kDiskSegmented)) {
      LOG(fatal) << "Invalid configuration: number of segments " << mSegments.first << " must be positive for segmented layout " << layout;
    }
    if (mSensorsPerSegment <= 0 && (layout == kBarrelSegmented || layout == kDiskSegmented)) {
      LOG(fatal) << "Invalid configuration: number of sensors per segment " << mSensorsPerSegment << " must be positive for segmented layout " << layout;
    }
    if (std::abs(mTiltAngle) > 0.1 && (layout != kBarrelSegmented && layout != kDiskSegmented)) {
      LOG(fatal) << "Invalid configuration: tilt angle " << mTiltAngle << " is set for non-segmented layout " << layout;
    }
  }

  LOGP(info, "TOF: Creating {} layer: rInner: {} (cm) rOuter: {} (cm) zLength: {} (cm) zOffset: {} x2X0: {}", name.c_str(), mInnerRadius, mOuterRadius, mZLength, mZOffset, mX2X0);
}

std::vector<std::string> ITOFLayer::mRegister;
void ITOFLayer::createLayer(TGeoVolume* motherVolume)
{
  const std::string chipName = o2::iotof::GeometryTGeo::getITOFChipPattern();
  const std::string sensName = o2::iotof::GeometryTGeo::getITOFSensorPattern();

  TGeoMedium* medSi = gGeoManager->GetMedium("TF3_SILICON$");
  TGeoMedium* medAir = gGeoManager->GetMedium("TF3_AIR$");
  LOGP(info, "Media: {} {}", (void*)medSi, (void*)medAir);

  switch (mLayout) {
    case kBarrel: {
      TGeoTube* sensor = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);
      TGeoTube* chip = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);
      TGeoTube* layer = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);

      TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
      TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
      TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
      sensVol->SetLineColor(kRed + 3);
      chipVol->SetLineColor(kRed + 3);
      layerVol->SetLineColor(kRed + 3);

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
      const double circumference = TMath::TwoPi() * 0.5 * (mInnerRadius + mOuterRadius);
      const double segmentSize = mSegments.second; // cm circumference / mSegments;
      const double avgRadius = 0.5 * (mInnerRadius + mOuterRadius);
      TGeoTube* layer = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);
      TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
      layerVol->SetLineColor(kRed + 3);

      for (int i = 0; i < mSegments.first; ++i) {
        LOGP(info, "iTOF: Creating segment {}/{} with size {} and thickness {}cm", i + 1, mSegments.first, segmentSize, (mOuterRadius - mInnerRadius));
        const double hx = 0.5 * segmentSize;
        const double hy = 0.5 * (mOuterRadius - mInnerRadius);
        const double hz = 0.5 * mZLength;
        TGeoBBox* sensor = new TGeoBBox(hy, hx, hz);
        TGeoBBox* chip = new TGeoBBox(hy, hx, hz);
        const std::string segmentTag = Form("segment%d", i + 1);
        TGeoVolume* sensVol = new TGeoVolume(Form("%s_%s", sensName.c_str(), segmentTag.c_str()), sensor, medSi);
        TGeoVolume* chipVol = new TGeoVolume(Form("%s_%s", chipName.c_str(), segmentTag.c_str()), chip, medSi);
        sensVol->SetLineColor(kRed + 3);
        chipVol->SetLineColor(kRed + 3);

        LOGP(info, "  Inserting Barrel {} in {} ", sensVol->GetName(), chipVol->GetName());
        ITOFLayer::mRegister.push_back(sensVol->GetName());
        chipVol->AddNode(sensVol, 1, nullptr);

        const double phi = TMath::TwoPi() * i / mSegments.first;

        LOG(info) << "  Tilting angle for segment " << i + 1 << ": " << phi * TMath::RadToDeg() << " degrees";
        const double x = avgRadius * TMath::Cos(phi);
        const double y = avgRadius * TMath::Sin(phi);
        auto* rotation = new TGeoRotation(Form("segmentRot%d", i + 1), phi * TMath::RadToDeg() + mTiltAngle, 0, 0);
        auto* transformation = new TGeoCombiTrans(x, y, 0, rotation);

        LOGP(info, "Inserting Barrel {} in {} ", chipVol->GetName(), layerVol->GetName());
        layerVol->AddNode(chipVol, 1 + i, transformation);
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
  std::string chipName = o2::iotof::GeometryTGeo::getOTOFChipPattern(),
              sensName = o2::iotof::GeometryTGeo::getOTOFSensorPattern();

  TGeoMedium* medSi = gGeoManager->GetMedium("TF3_SILICON$");
  TGeoMedium* medAir = gGeoManager->GetMedium("TF3_AIR$");
  LOGP(info, "Media: {} {}", (void*)medSi, (void*)medAir);

  switch (mLayout) {
    case kBarrel: {
      TGeoTube* sensor = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);
      TGeoTube* chip = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);
      TGeoTube* layer = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);

      TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
      TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
      TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
      sensVol->SetLineColor(kRed + 3);
      chipVol->SetLineColor(kRed + 3);
      layerVol->SetLineColor(kRed + 3);

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
      const double circumference = TMath::TwoPi() * 0.5 * (mInnerRadius + mOuterRadius);
      const double segmentSize = mSegments.second; // cm circumference / mSegments;
      const double avgRadius = 0.5 * (mInnerRadius + mOuterRadius);
      TGeoTube* layer = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);
      TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
      layerVol->SetLineColor(kRed + 3);

      for (int i = 0; i < mSegments.first; ++i) {
        LOGP(info, "oTOF: Creating segment {}/{} with size {} and thickness {}cm", i + 1, mSegments.first, segmentSize, (mOuterRadius - mInnerRadius));
        const double hx = 0.5 * segmentSize;
        const double hy = 0.5 * (mOuterRadius - mInnerRadius);
        const double hz = 0.5 * mZLength;
        TGeoBBox* sensor = new TGeoBBox(hy, hx, hz);
        TGeoBBox* chip = new TGeoBBox(hy, hx, hz);
        const std::string segmentTag = Form("segment%d", i + 1);
        TGeoVolume* sensVol = new TGeoVolume(Form("%s_%s", sensName.c_str(), segmentTag.c_str()), sensor, medSi);
        TGeoVolume* chipVol = new TGeoVolume(Form("%s_%s", chipName.c_str(), segmentTag.c_str()), chip, medSi);
        sensVol->SetLineColor(kRed + 3);
        chipVol->SetLineColor(kRed + 3);

        LOGP(info, "  Inserting Barrel {} in {} ", sensVol->GetName(), chipVol->GetName());
        OTOFLayer::mRegister.push_back(sensVol->GetName());
        chipVol->AddNode(sensVol, 1, nullptr);

        const double phi = TMath::TwoPi() * i / mSegments.first;

        LOG(info) << "  Tilting angle for segment " << i + 1 << ": " << phi * TMath::RadToDeg() << " degrees";
        const double x = avgRadius * TMath::Cos(phi);
        const double y = avgRadius * TMath::Sin(phi);
        auto* rotation = new TGeoRotation(Form("segmentRot%d", i + 1), phi * TMath::RadToDeg() + mTiltAngle, 0, 0);
        auto* transformation = new TGeoCombiTrans(x, y, 0, rotation);

        LOGP(info, "Inserting Barrel {} in {} ", chipVol->GetName(), layerVol->GetName());
        layerVol->AddNode(chipVol, 1 + i, transformation);
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
  sensVol->SetLineColor(kRed + 3);
  chipVol->SetLineColor(kRed + 3);
  layerVol->SetLineColor(kRed + 3);

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
  sensVol->SetLineColor(kRed + 3);
  chipVol->SetLineColor(kRed + 3);
  layerVol->SetLineColor(kRed + 3);

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