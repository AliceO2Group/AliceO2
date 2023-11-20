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

#include "RICHSimulation/RICHRing.h"
#include "RICHBase/GeometryTGeo.h"

#include "Framework/Logger.h"

#include <TGeoTube.h>
#include <TGeoVolume.h>

namespace o2
{
namespace rich
{
RICHRing::RICHRing(int ringNumber, std::string ringName, float rInn, float rOut, float zLength, float ringX2X0)
  : mRingNumber(ringNumber), mRingName(ringName), mInnerRadius(rInn), mOuterRadius(rOut), mZ(zLength), mX2X0(ringX2X0)
{
  float Si_X0 = 9.5f;
  mChipThickness = mX2X0 * Si_X0;
  LOGP(info, "Creating ring: id: {} rInner: {} rOuter: {} zLength: {} x2X0: {}", mRingNumber, mInnerRadius, mOuterRadius, mZ, mX2X0);
}

RICHRing::RICHRing(int ringNumber, std::string ringName, float rInn, float zLength, float thick)
  : mRingNumber(ringNumber), mRingName(ringName), mInnerRadius(rInn), mZ(zLength), mChipThickness(thick)
{
  float Si_X0 = 9.5f;
  mOuterRadius = rInn + thick;
  mX2X0 = mChipThickness / Si_X0;
  LOGP(info, "Creating ring: id: {} rInner: {} rOuter: {} zLength: {} x2X0: {}", mRingNumber, mInnerRadius, mOuterRadius, mZ, mX2X0);
}

void RICHRing::createRing(TGeoVolume* motherVolume)
{
  std::string chipName = o2::rich::GeometryTGeo::getRICHChipPattern() + std::to_string(mRingNumber),
              sensName = Form("%s%d", GeometryTGeo::getRICHSensorPattern(), mRingNumber);

  TGeoTube* sensor = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2);
  TGeoTube* chip = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2);
  TGeoTube* ring = new TGeoTube(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2);

  TGeoMedium* medSi = gGeoManager->GetMedium("RICH_SI$");
  TGeoMedium* medAir = gGeoManager->GetMedium("RICH_AIR$");

  TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
  sensVol->SetLineColor(kBlue - 4);
  TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
  chipVol->SetLineColor(kBlue - 4);
  TGeoVolume* ringVol = new TGeoVolume(mRingName.c_str(), ring, medAir);
  ringVol->SetLineColor(kBlue - 4);

  LOGP(info, "Inserting {} in {} ", sensVol->GetName(), chipVol->GetName());
  chipVol->AddNode(sensVol, 1, nullptr);

  LOGP(info, "Inserting {} in {} ", chipVol->GetName(), ringVol->GetName());
  ringVol->AddNode(chipVol, 1, nullptr);

  LOGP(info, "Inserting {} in {} ", ringVol->GetName(), motherVolume->GetName());
  motherVolume->AddNode(ringVol, 1, nullptr);
}

} // namespace rich
} // namespace o2