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

#include <TGeoTube.h>
#include <TGeoVolume.h>

namespace o2
{
namespace iotof
{
Layer::Layer(std::string layerName, float rInn, float rOut, float zLength, float zOffset, float layerX2X0, bool isBarrel)
  : mLayerName(layerName), mInnerRadius(rInn), mOuterRadius(rOut), mZLength(zLength), mZOffset(zOffset), mX2X0(layerX2X0), mIsBarrel(isBarrel)
{
  float Si_X0 = 9.5f;
  mChipThickness = mX2X0 * Si_X0;
  if (isBarrel) {
    mOuterRadius = mInnerRadius + mChipThickness;
  } else {
    mZLength = mChipThickness;
  }
  LOGP(info, "TOF: Creating {} layer: rInner: {} (cm) rOuter: {} (cm) zLength: {} (cm) zOffset: {} x2X0: {}", isBarrel ? std::string("barrel") : std::string("forward"), mInnerRadius, mOuterRadius, mZLength, mZOffset, mX2X0);
}

void ITOFLayer::createLayer(TGeoVolume* motherVolume)
{
  std::string chipName = o2::iotof::GeometryTGeo::getITOFChipPattern(),
              sensName = o2::iotof::GeometryTGeo::getITOFSensorPattern();

  TGeoTube* sensor = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);
  TGeoTube* chip = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);
  TGeoTube* layer = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);

  TGeoMedium* medSi = gGeoManager->GetMedium("TF3_SI$");
  TGeoMedium* medAir = gGeoManager->GetMedium("TF3_AIR$");

  LOGP(info, "Media: {} {}", (void*)medSi, (void*)medAir);

  TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
  TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
  sensVol->SetLineColor(kMagenta - 7);
  chipVol->SetLineColor(kMagenta - 7);
  layerVol->SetLineColor(kMagenta - 7);

  LOGP(info, "Inserting {} in {} ", sensVol->GetName(), chipVol->GetName());
  chipVol->AddNode(sensVol, 1, nullptr);

  LOGP(info, "Inserting {} in {} ", chipVol->GetName(), layerVol->GetName());
  layerVol->AddNode(chipVol, 1, nullptr);

  LOGP(info, "Inserting {} in {} ", layerVol->GetName(), motherVolume->GetName());
  motherVolume->AddNode(layerVol, 1, nullptr);
}

void OTOFLayer::createLayer(TGeoVolume* motherVolume)
{
  std::string chipName = o2::iotof::GeometryTGeo::getOTOFChipPattern(),
              sensName = o2::iotof::GeometryTGeo::getOTOFSensorPattern();

  TGeoTube* sensor = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);
  TGeoTube* chip = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);
  TGeoTube* layer = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);

  TGeoMedium* medSi = gGeoManager->GetMedium("TF3_SI$");
  TGeoMedium* medAir = gGeoManager->GetMedium("TF3_AIR$");

  TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
  TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
  sensVol->SetLineColor(kMagenta - 7);
  chipVol->SetLineColor(kMagenta - 7);
  layerVol->SetLineColor(kMagenta - 7);

  LOGP(info, "Inserting {} in {} ", sensVol->GetName(), chipVol->GetName());
  chipVol->AddNode(sensVol, 1, nullptr);

  LOGP(info, "Inserting {} in {} ", chipVol->GetName(), layerVol->GetName());
  layerVol->AddNode(chipVol, 1, nullptr);

  LOGP(info, "Inserting {} in {} ", layerVol->GetName(), motherVolume->GetName());
  motherVolume->AddNode(layerVol, 1, nullptr);
}

void FTOFLayer::createLayer(TGeoVolume* motherVolume)
{
  std::string chipName = o2::iotof::GeometryTGeo::getFTOFChipPattern(),
              sensName = o2::iotof::GeometryTGeo::getFTOFSensorPattern();

  TGeoTube* sensor = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);
  TGeoTube* chip = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);
  TGeoTube* layer = new TGeoTube(mInnerRadius, mOuterRadius, mZLength / 2);

  TGeoMedium* medSi = gGeoManager->GetMedium("TF3_SI$");
  TGeoMedium* medAir = gGeoManager->GetMedium("TF3_AIR$");

  TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
  TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
  sensVol->SetLineColor(kMagenta - 7);
  chipVol->SetLineColor(kMagenta - 7);
  layerVol->SetLineColor(kMagenta - 7);

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

  TGeoMedium* medSi = gGeoManager->GetMedium("TF3_SI$");
  TGeoMedium* medAir = gGeoManager->GetMedium("TF3_AIR$");

  TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
  TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
  sensVol->SetLineColor(kMagenta - 7);
  chipVol->SetLineColor(kMagenta - 7);
  layerVol->SetLineColor(kMagenta - 7);

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