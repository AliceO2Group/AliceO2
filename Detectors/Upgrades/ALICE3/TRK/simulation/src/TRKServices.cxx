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

#include <TRKSimulation/TRKServices.h>
#include <DetectorsBase/MaterialManager.h>
#include <TRKBase/GeometryTGeo.h>
#include <TGeoVolume.h>
#include <TGeoNode.h>
#include <TGeoTube.h>
#include <TColor.h>
#include <Rtypes.h>
#include <numeric>

#include <Framework/Logger.h>

namespace o2
{
namespace trk
{
TRKServices::TRKServices(float rMin, float zLength, float thickness)
{
  mColdPlateRMin = rMin;
  mColdPlateZLength = zLength;
  mColdPlateThickness = thickness;
}

void TRKServices::createMaterials()
{
  int ifield = 2;      // ?
  float fieldm = 10.0; // ?

  // Defines tracking media parameters.
  float epsil = .1;     // Tracking precision,
  float stemax = -0.01; // Maximum displacement for multiple scat
  float tmaxfd = -20.;  // Maximum angle due to field deflection
  float deemax = -.3;   // Maximum fractional energy loss, DLS
  float stmin = -.8;

  auto& matmgr = o2::base::MaterialManager::Instance();

  // Ceramic (Aluminium Oxide)
  float aCer[2] = {26.981538, 15.9994};
  float zCer[2] = {13., 8.};
  float wCer[2] = {0.5294, 0.4706}; // Mass %, which makes sense. TODO: check if Mixture needs mass% or comp%
  float dCer = 3.97;

  // AIR
  float aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  float zAir[4] = {6., 7., 8., 18.};
  float wAir[4] = {0.000124, 0.755267, 0.231781, 0.012827};
  float dAir = 1.20479E-3;

  matmgr.Mixture("TRKSERVICES", 66, "CER$", aCer, zCer, dCer, 2, wCer); // Ceramic for cold plate
  matmgr.Material("TRKSERVICES", 67, "COP", 63.546, 29, 8.96, 1, 1.);   // Copper for cables
  matmgr.Mixture("TRKSERVICES", 68, "VAC", aAir, zAir, dAir, 4, wAir);  // Vacuum for placeholding cables

  matmgr.Medium("TRKSERVICES", 1, "CER", 66, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin); // Ceramic for cold plate
  matmgr.Medium("TRKSERVICES", 2, "COP", 67, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin); // Copper for cables
  matmgr.Medium("TRKSERVICES", 3, "VAC", 68, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin); // Vacuum for placeholding cables
}

void TRKServices::createServices(TGeoVolume* motherVolume)
{
  createMaterials();
  createColdplate(motherVolume);
  createCables(motherVolume);
}

void TRKServices::createColdplate(TGeoVolume* motherVolume)
{
  auto& matmgr = o2::base::MaterialManager::Instance();
  const TGeoMedium* medCeramic = matmgr.getTGeoMedium("TRKSERVICES_CER");

  TGeoTube* coldPlate = new TGeoTube("TRK_COLDPLATEsh", mColdPlateRMin, mColdPlateRMin + mColdPlateThickness, mColdPlateZLength / 2.);
  TGeoVolume* coldPlateVolume = new TGeoVolume("TRK_COLDPLATE", coldPlate, medCeramic);
  coldPlateVolume->SetVisibility(1);
  coldPlateVolume->SetLineColor(kYellow);

  LOGP(info, "Creating cold plate service");

  LOGP(info, "Inserting {} in {} ", coldPlateVolume->GetName(), motherVolume->GetName());
  motherVolume->AddNode(coldPlateVolume, 1, nullptr);
}

void TRKServices::createCables(TGeoVolume* motherVolume)
{
  auto& matmgr = o2::base::MaterialManager::Instance();
  const TGeoMedium* medCopper = matmgr.getTGeoMedium("TRKSERVICES_COP");

  // Inner Tracker Services
  // Get geometry information from TRK which is already present
  float rMinInnerCables = ((TGeoTube*)motherVolume->GetNode(Form("%s3_1", GeometryTGeo::getTRKLayerPattern()))->GetVolume()->GetShape())->GetRmin();
  float rMaxInnerCables = 35.0; // ((TGeoTube*)motherVolume->GetNode(Form("%s7_1", GeometryTGeo::getTRKLayerPattern()))->GetVolume()->GetShape())->GetRmax();
  float zLengthInnerCables = ((TGeoTube*)motherVolume->GetNode(Form("%s7_1", GeometryTGeo::getTRKLayerPattern()))->GetVolume()->GetShape())->GetDz();
  LOGP(info, "Building service disk for Inner Tracker rminInnerCables is: {} rmaxInnerCables is {} Dz is {}", rMinInnerCables, rMaxInnerCables, zLengthInnerCables);

  TGeoMedium* medVac = matmgr.getTGeoMedium("TRKSERVICES_VAC");
  TGeoMedium* medCop = matmgr.getTGeoMedium("TRKSERVICES_COP");

  for (size_t iCableFan{0}; iCableFan < mCableFanWeights.size(); ++iCableFan) {
    TGeoTube* cableFan = new TGeoTube("TRK_CABLEFAN_MIDsh", rMinInnerCables, rMaxInnerCables, mMiddleDiskThickness * mCableFanWeights[iCableFan]);
    TGeoVolume* cableFanVolume = new TGeoVolume("TRK_CABLEFAN_MID", cableFan, !(iCableFan % 2) ? medVac : medCop);
    cableFanVolume->SetLineColor(!(iCableFan % 2) ? kGray : kBlack);
    auto* rot = new TGeoRotation(Form("TRK_CABLEFAN_MID_ROT_%d", iCableFan), 0, 0, 180);
    float incrShiftW = std::accumulate(mCableFanWeights.begin(), mCableFanWeights.begin() + iCableFan + 1, -mCableFanWeights[0]);
    auto* combiTrans = new TGeoCombiTrans(0, 0, zLengthInnerCables + 0.5 * mMiddleDiskThickness + incrShiftW * mMiddleDiskThickness, rot);

    LOGP(info, "Inserting {} in {} at Z={} position which is {} + {} ", cableFanVolume->GetName(), motherVolume->GetName(), zLengthInnerCables + incrShiftW * mMiddleDiskThickness, zLengthInnerCables, incrShiftW * mMiddleDiskThickness);
    motherVolume->AddNode(cableFanVolume, 1, combiTrans);
  }
  // Outer Tracker Services
  // Get geometry information from TRK which is already present
  float rMinOuterCables = 35.0 + mMiddleDiskThickness; // ((TGeoTube*)motherVolume->GetNode(Form("%s8_1", GeometryTGeo::getTRKLayerPattern()))->GetVolume()->GetShape())->GetRmin();
  float rMaxOuterCables = ((TGeoTube*)motherVolume->GetNode(Form("%s10_1", GeometryTGeo::getTRKLayerPattern()))->GetVolume()->GetShape())->GetRmax();
  float zLengthOuterCables = ((TGeoTube*)motherVolume->GetNode(Form("%s10_1", GeometryTGeo::getTRKLayerPattern()))->GetVolume()->GetShape())->GetDz();
  LOGP(info, "Building service disk for Outer Tracker rminOuterCables is: {} rmaxOuterCables is {} Dz is {}", rMinOuterCables, rMaxOuterCables, zLengthOuterCables);

  for (size_t iCableFan{0}; iCableFan < mCableFanWeights.size(); ++iCableFan) {
    TGeoTube* cableFan = new TGeoTube("TRK_CABLEFAN_EXTsh", rMinOuterCables, rMaxOuterCables, mMiddleDiskThickness * mCableFanWeights[iCableFan]);
    TGeoVolume* cableFanVolume = new TGeoVolume("TRK_CABLEFAN_EXT", cableFan, !(iCableFan % 2) ? medVac : medCop);
    cableFanVolume->SetLineColor(!(iCableFan % 2) ? kGray : kBlack);
    auto* rot = new TGeoRotation(Form("TRK_CABLEFAN_EXT_ROT_%d", iCableFan), 0, 0, 180);
    float incrShiftW = std::accumulate(mCableFanWeights.begin(), mCableFanWeights.begin() + iCableFan + 1, -mCableFanWeights[0]);
    auto* combiTrans = new TGeoCombiTrans(0, 0, zLengthOuterCables + 0.5 * mMiddleDiskThickness + incrShiftW * mMiddleDiskThickness, rot);

    LOGP(info, "Inserting {} in {} at Z={} position which is {} + {} ", cableFanVolume->GetName(), motherVolume->GetName(), zLengthOuterCables + incrShiftW * mMiddleDiskThickness, zLengthOuterCables, incrShiftW * mMiddleDiskThickness);
    motherVolume->AddNode(cableFanVolume, 1, combiTrans);
  }

  // FWD disks services, middle triplet
  // I put them here for convenience, ideally one would eventually have TRK and FT3 merged

  float rMinFwdCables = 35.0;
  float rMaxFwdCables = rMinFwdCables + mMiddleDiskThickness;
  float zLengthFwdCables = ((TGeoTube*)motherVolume->GetNode(Form("%s8_1", GeometryTGeo::getTRKLayerPattern()))->GetVolume()->GetShape())->GetDz() -
                           ((TGeoTube*)motherVolume->GetNode(Form("%s7_1", GeometryTGeo::getTRKLayerPattern()))->GetVolume()->GetShape())->GetDz() + mMiddleDiskThickness;

  LOGP(info, "Building service disk for FWD rminFwdCables is: {} rmaxFwdCables is {} Dz is {}", rMinFwdCables, rMaxFwdCables, zLengthFwdCables);
  for (size_t iCableLayer{0}; iCableLayer < mCableFanWeights.size(); ++iCableLayer) {
    TGeoTube* middleCableLayer = new TGeoTube("TRK_CABLELAYER_MIDsh", rMinFwdCables, rMaxFwdCables + mMiddleDiskThickness * mCableFanWeights[iCableLayer], zLengthFwdCables / 2.);
    TGeoVolume* middleCableLayerVolume = new TGeoVolume("TRK_CABLELAYER_MID", middleCableLayer, !(iCableLayer % 2) ? medVac : medCop);
    middleCableLayerVolume->SetLineColor(!(iCableLayer % 2) ? kGray : kBlack);
    auto* combiTrans = new TGeoCombiTrans(0, 0, zLengthInnerCables + (zLengthFwdCables) / 2, nullptr);

    LOGP(info, "Inserting {} in {} at Z={} position which is {} + {} ", middleCableLayerVolume->GetName(), motherVolume->GetName(), zLengthInnerCables + zLengthFwdCables / 2, zLengthInnerCables, zLengthFwdCables / 2);
    motherVolume->AddNode(middleCableLayerVolume, 1, combiTrans);
  }

  // FWD disks services, outer triplet
  // I put them here for convenience, ideally one would eventually have TRK and FT3 merged

  rMinFwdCables = rMaxOuterCables;
  rMaxFwdCables = rMinFwdCables + mMiddleDiskThickness;
  zLengthFwdCables = 400.0 - ((TGeoTube*)motherVolume->GetNode(Form("%s10_1", GeometryTGeo::getTRKLayerPattern()))->GetVolume()->GetShape())->GetDz() + mMiddleDiskThickness;

  LOGP(info, "Building service disk for FWD rminFwdCables is: {} rmaxFwdCables is {} Dz is {}", rMinFwdCables, rMaxFwdCables, zLengthFwdCables);
  for (size_t iCableLayer{0}; iCableLayer < mCableFanWeights.size(); ++iCableLayer) {
    TGeoTube* outerCableLayer = new TGeoTube("TRK_CABLELAYER_EXTsh", rMinFwdCables, rMaxFwdCables + mMiddleDiskThickness * mCableFanWeights[iCableLayer], zLengthFwdCables / 2.);
    TGeoVolume* outerCableLayerVolume = new TGeoVolume("TRK_CABLELAYER_EXT", outerCableLayer, !(iCableLayer % 2) ? medVac : medCop);
    outerCableLayerVolume->SetLineColor(!(iCableLayer % 2) ? kGray : kBlack);
    auto* combiTrans = new TGeoCombiTrans(0, 0, zLengthOuterCables + (zLengthFwdCables) / 2, nullptr);

    LOGP(info, "Inserting {} in {} at Z={} position which is {} + {} ", outerCableLayerVolume->GetName(), motherVolume->GetName(), zLengthOuterCables + zLengthFwdCables / 2, zLengthOuterCables, zLengthFwdCables / 2);
    motherVolume->AddNode(outerCableLayerVolume, 1, combiTrans);
  }
}
} // namespace trk
} // namespace o2