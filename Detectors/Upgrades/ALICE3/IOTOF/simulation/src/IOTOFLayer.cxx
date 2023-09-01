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

#include "IOTOFSimulation/IOTOFLayer.h"
#include "IOTOFBase/GeometryTGeo.h"

#include "Framework/Logger.h"

#include <TGeoTube.h>
#include <TGeoVolume.h>

namespace o2
{
namespace iotof
{
Layer::Layer(std::string layerName, float rInn, float zLength, float layerX2X0)
  : mLayerName(layerName), mInnerRadius(rInn), mZ(zLength), mX2X0(layerX2X0)
{
  float Si_X0 = 9.5f;
  mChipThickness = mX2X0 * Si_X0;
  mOuterRadius = mInnerRadius + mChipThickness;
  LOGP(info, "TOF: Creating layer: rInner: {} rOuter: {} zLength: {} x2X0: {}", mInnerRadius, mOuterRadius, mZ, mX2X0);
}

// Layer::Layer(std::string layerName, float rInn, float zLength, float thick)
//   : mLayerName(layerName), mInnerRadius(rInn), mZ(zLength), mChipThickness(thick)
// {
//   float Si_X0 = 9.5f;
//   mOuterRadius = rInn + thick;
//   mX2X0 = mChipThickness / Si_X0;
//   LOGP(info, "TOF: Creating layer: rInner: {} rOuter: {} zLength: {} x2X0: {}", mInnerRadius, mOuterRadius, mZ, mX2X0);
// }

void ITOFLayer::createLayer(TGeoVolume* motherVolume)
{
  std::string chipName = o2::iotof::GeometryTGeo::getITOFChipPattern(),
              sensName = o2::iotof::GeometryTGeo::getITOFSensorPattern();

  TGeoTube* sensor = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2);
  TGeoTube* chip = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2);
  TGeoTube* layer = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2);

  TGeoMedium* medSi = gGeoManager->GetMedium("IOTOF_SI$");
  TGeoMedium* medAir = gGeoManager->GetMedium("IOTOF_AIR$");

  TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
  TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
  layerVol->SetLineColor(kAzure + 4);

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

  TGeoTube* sensor = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2);
  TGeoTube* chip = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2);
  TGeoTube* layer = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2);

  TGeoMedium* medSi = gGeoManager->GetMedium("IOTOF_SI$");
  TGeoMedium* medAir = gGeoManager->GetMedium("IOTOF_AIR$");

  TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
  TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
  layerVol->SetLineColor(kRed + 1);

  LOGP(info, "Inserting {} in {} ", sensVol->GetName(), chipVol->GetName());
  chipVol->AddNode(sensVol, 1, nullptr);

  LOGP(info, "Inserting {} in {} ", chipVol->GetName(), layerVol->GetName());
  layerVol->AddNode(chipVol, 1, nullptr);

  LOGP(info, "Inserting {} in {} ", layerVol->GetName(), motherVolume->GetName());
  motherVolume->AddNode(layerVol, 1, nullptr);
}

} // namespace iotof
} // namespace o2