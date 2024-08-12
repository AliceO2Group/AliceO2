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
#include <TGeoBBox.h>
#include <TGeoVolume.h>

#include <TMath.h>

namespace o2
{
namespace trk
{
TRKLayer::TRKLayer(int layerNumber, std::string layerName, float rInn, float rOut, float zLength, float layerX2X0)
  : mLayerNumber(layerNumber), mLayerName(layerName), mInnerRadius(rInn), mOuterRadius(rOut), mZ(zLength), mX2X0(layerX2X0), mModuleWidth(4.54), mLayout(kCylinder)
{
  float Si_X0 = 9.5f;
  mChipThickness = mX2X0 * Si_X0;
  LOGP(info, "Creating layer: id: {} rInner: {} rOuter: {} zLength: {} x2X0: {}", mLayerNumber, mInnerRadius, mOuterRadius, mZ, mX2X0);
}

TRKLayer::TRKLayer(int layerNumber, std::string layerName, float rInn, float zLength, float thick)
  : mLayerNumber(layerNumber), mLayerName(layerName), mInnerRadius(rInn), mZ(zLength), mChipThickness(thick), mModuleWidth(4.54), mLayout(kCylinder)
{
  float Si_X0 = 9.5f;
  mOuterRadius = rInn + thick;
  mX2X0 = mChipThickness / Si_X0;
  LOGP(info, "Creating layer: id: {} rInner: {} rOuter: {} zLength: {} x2X0: {}", mLayerNumber, mInnerRadius, mOuterRadius, mZ, mX2X0);
}

void TRKLayer::createLayer(TGeoVolume* motherVolume)
{
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");

  std::string staveName = o2::trk::GeometryTGeo::getTRKStavePattern() + std::to_string(mLayerNumber),
              chipName = o2::trk::GeometryTGeo::getTRKChipPattern() + std::to_string(mLayerNumber),
              sensName = Form("%s%d", GeometryTGeo::getTRKSensorPattern(), mLayerNumber);

  TGeoTube* layer = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2);

  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
  layerVol->SetLineColor(kYellow);

  if (mLayout == eLayout::kCylinder) {
    TGeoTube* stave = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2);
    TGeoTube* chip = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2);
    TGeoTube* sensor = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2);

    TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
    sensVol->SetLineColor(kYellow);
    TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
    chipVol->SetLineColor(kYellow);
    TGeoVolume* staveVol = new TGeoVolume(staveName.c_str(), stave, medSi);
    staveVol->SetLineColor(kYellow);

    LOGP(info, "Inserting {} in {} ", sensVol->GetName(), chipVol->GetName());
    chipVol->AddNode(sensVol, 1, nullptr);

    LOGP(info, "Inserting {} in {} ", chipVol->GetName(), staveVol->GetName());
    staveVol->AddNode(chipVol, 1, nullptr);

    LOGP(info, "Inserting {} in {} ", staveVol->GetName(), layerVol->GetName());
    layerVol->AddNode(staveVol, 1, nullptr);
  } else if (mLayout == eLayout::kTurboStaves) {
    // Compute the number of staves
    double width = mModuleWidth * 2; // Each stave has two modules (based on the LOI design)
    int nStaves = (int)std::ceil(mInnerRadius * 2 * TMath::Pi() / width);
    nStaves += nStaves % 2; // Require an even number of staves

    // Compute the size of the overlap region
    double theta = 2 * TMath::Pi() / nStaves;
    double theta1 = std::atan(width / 2 / mInnerRadius);
    double st = std::sin(theta);
    double ct = std::cos(theta);
    double theta2 = std::atan((mInnerRadius * st - width / 2 * ct) / (mInnerRadius * ct + width / 2 * st));
    double overlap = (theta1 - theta2) * mInnerRadius;
    LOGP(info, "Creating a layer with {} staves and {} mm overlap", nStaves, overlap * 10);

    for (int iStave = 0; iStave < nStaves; iStave++) {
      TGeoBBox* sensor = new TGeoBBox(width / 2, mChipThickness / 2, mZ / 2);
      TGeoBBox* chip = new TGeoBBox(width / 2, mChipThickness / 2, mZ / 2);
      TGeoBBox* stave = new TGeoBBox(width / 2, mChipThickness / 2, mZ / 2);

      TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
      sensVol->SetLineColor(kYellow);
      TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
      chipVol->SetLineColor(kYellow);
      TGeoVolume* staveVol = new TGeoVolume(staveName.c_str(), stave, medSi);
      staveVol->SetLineColor(kYellow);

      // Put the staves in the correct position and orientation
      TGeoCombiTrans* trans = new TGeoCombiTrans();
      double theta = 360. * iStave / nStaves;
      TGeoRotation* rot = new TGeoRotation("rot", theta + 90 + 2, 0, 0);
      trans->SetRotation(rot);
      trans->SetTranslation(mInnerRadius * std::cos(2. * TMath::Pi() * iStave / nStaves), mInnerRadius * std::sin(2 * TMath::Pi() * iStave / nStaves), 0);

      LOGP(info, "Inserting {} in {} ", sensVol->GetName(), chipVol->GetName());
      chipVol->AddNode(sensVol, 1, nullptr);

      LOGP(info, "Inserting {} in {} ", chipVol->GetName(), staveVol->GetName());
      staveVol->AddNode(chipVol, 1, nullptr);

      LOGP(info, "Inserting {} in {} ", staveVol->GetName(), layerVol->GetName());
      layerVol->AddNode(staveVol, iStave, trans);
    }
  } else {
    LOGP(fatal, "Layout not implemented");
  }
  LOGP(info, "Inserting {} in {} ", layerVol->GetName(), motherVolume->GetName());
  motherVolume->AddNode(layerVol, 1, nullptr);
}
// ClassImp(TRKLayer);

} // namespace trk
} // namespace o2