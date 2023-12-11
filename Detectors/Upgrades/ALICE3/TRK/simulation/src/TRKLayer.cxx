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
#include <TGeoVolume.h>

namespace o2
{
namespace trk
{
TRKLayer::TRKLayer(int layerNumber, std::string layerName, float rInn, float rOut, float zLength, float layerX2X0)
  : mLayerNumber(layerNumber), mLayerName(layerName), mInnerRadius(rInn), mOuterRadius(rOut), mZ(zLength), mX2X0(layerX2X0)
{
  float Si_X0 = 9.5f;
  mChipThickness = mX2X0 * Si_X0;
  LOGP(info, "Creating layer: id: {} rInner: {} rOuter: {} zLength: {} x2X0: {}", mLayerNumber, mInnerRadius, mOuterRadius, mZ, mX2X0);
}

TRKLayer::TRKLayer(int layerNumber, std::string layerName, float rInn, float zLength, float thick)
  : mLayerNumber(layerNumber), mLayerName(layerName), mInnerRadius(rInn), mZ(zLength), mChipThickness(thick)
{
  float Si_X0 = 9.5f;
  mOuterRadius = rInn + thick;
  mX2X0 = mChipThickness / Si_X0;
  LOGP(info, "Creating layer: id: {} rInner: {} rOuter: {} zLength: {} x2X0: {}", mLayerNumber, mInnerRadius, mOuterRadius, mZ, mX2X0);
}

void TRKLayer::createLayer(TGeoVolume* motherVolume)
{
  std::string chipName = o2::trk::GeometryTGeo::getTRKChipPattern() + std::to_string(mLayerNumber),
              sensName = Form("%s%d", GeometryTGeo::getTRKSensorPattern(), mLayerNumber);

  TGeoTube* sensor = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2);
  TGeoTube* chip = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2);
  TGeoTube* layer = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2);

  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SI$");
  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");

  TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
  sensVol->SetLineColor(kBlue - 7);
  TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
  chipVol->SetLineColor(kBlue - 7);
  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
  layerVol->SetLineColor(kBlue - 7);

  LOGP(info, "Inserting {} in {} ", sensVol->GetName(), chipVol->GetName());
  chipVol->AddNode(sensVol, 1, nullptr);

  LOGP(info, "Inserting {} in {} ", chipVol->GetName(), layerVol->GetName());
  layerVol->AddNode(chipVol, 1, nullptr);

  LOGP(info, "Inserting {} in {} ", layerVol->GetName(), motherVolume->GetName());
  motherVolume->AddNode(layerVol, 1, nullptr);
}
// ClassImp(TRKLayer);

} // namespace trk
} // namespace o2