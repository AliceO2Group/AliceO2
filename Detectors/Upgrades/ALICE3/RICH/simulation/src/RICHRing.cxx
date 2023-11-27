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
#include "RICHBase/RICHBaseParam.h"
#include "Framework/Logger.h"

#include <TGeoManager.h>
#include <TGeoTube.h>
#include <TGeoVolume.h>
#include <TGeoArb8.h>

namespace o2
{
namespace rich
{

Ring::Ring(int rPosId,
           int nTilesPhi,
           float rMin,
           float rMax,
           float radThick,
           float radYmin,
           float radYmax,
           float radZ,
           float photThick,
           float photYmin,
           float photYmax,
           float photZ,
           float radTransZ,
           float photTransZ,
           float thetaB,
           const std::string motherName)
  : mNTiles{nTilesPhi}, mPosId{rPosId}, mRadThickness{radThick}
{
  TGeoManager* geoManager = gGeoManager;
  TGeoVolume* motherVolume = geoManager->GetVolume(motherName.c_str());
  TGeoMedium* medAerogel = gGeoManager->GetMedium("RCH_AEROGEL$");
  if (!medAerogel) {
    LOGP(fatal, "RICH: Aerogel medium not found");
  }
  TGeoMedium* medSi = gGeoManager->GetMedium("RCH_SI$");
  if (!medSi) {
    LOGP(fatal, "RICH: Silicon medium not found");
  }
  std::vector<TGeoArb8*> radiatorTiles(nTilesPhi), photoTiles(nTilesPhi);
  LOGP(info, "Creating ring: id: {} with {} tiles. ", rPosId, nTilesPhi);
  LOGP(info, "Rmin: {} Rmax: {} RadThick: {} RadYmin: {} RadYmax: {} RadZ: {} PhotThick: {} PhotYmin: {} PhotYmax: {} PhotZ: {}, zTransRad: {}, zTransPhot: {}, ThetaB: {}",
       rMin, rMax, radThick, radYmin, radYmax, radZ, photThick, photYmin, photYmax, photZ, radTransZ, photTransZ, thetaB);

  float deltaPhiDeg = 360.0 / nTilesPhi; // Transformation are constructed in degrees...
  float thetaBDeg = thetaB * 180.0 / TMath::Pi();
  size_t radTileCount{0}, photTileCount{0};
  // Radiator tiles
  for (auto& radiatorTile : radiatorTiles) {
    radiatorTile = new TGeoArb8(radZ / 2);
    radiatorTile->SetVertex(0, -radThick / 2, -radYmin / 2);
    radiatorTile->SetVertex(1, -radThick / 2, radYmin / 2);
    radiatorTile->SetVertex(2, radThick / 2, radYmin / 2);
    radiatorTile->SetVertex(3, radThick / 2, -radYmin / 2);
    radiatorTile->SetVertex(4, -radThick / 2, -radYmax / 2);
    radiatorTile->SetVertex(5, -radThick / 2, radYmax / 2);
    radiatorTile->SetVertex(6, radThick / 2, radYmax / 2);
    radiatorTile->SetVertex(7, radThick / 2, -radYmax / 2);

    TGeoVolume* radiatorTileVol = new TGeoVolume(Form("radTile_%d_%d", rPosId, radTileCount), radiatorTile, medAerogel);
    radiatorTileVol->SetLineColor(kBlue - 9);
    radiatorTileVol->SetFillColor(kBlue - 9);
    radiatorTileVol->SetTransparency(50);
    radiatorTileVol->SetLineWidth(1);

    auto* rotRadiator = new TGeoRotation(Form("radTileRotation_%d_%d", radTileCount, rPosId));
    rotRadiator->RotateY(-thetaBDeg);
    rotRadiator->RotateZ(radTileCount * deltaPhiDeg);

    auto* rotTransRadiator = new TGeoCombiTrans(radTransZ * TMath::Cos(radTileCount * TMath::Pi() / (nTilesPhi / 2)),
                                                radTransZ * TMath::Sin(radTileCount * TMath::Pi() / (nTilesPhi / 2)),
                                                radTransZ * TMath::Tan(thetaB),
                                                rotRadiator);

    motherVolume->AddNode(radiatorTileVol, 1, rotTransRadiator);
    radTileCount++;
  }
  // Photosensor tiles
  for (auto& photoTile : photoTiles) {
    photoTile = new TGeoArb8(photZ / 2);
    photoTile->SetVertex(0, -photThick / 2, -photYmin / 2);
    photoTile->SetVertex(1, -photThick / 2, photYmin / 2);
    photoTile->SetVertex(2, photThick / 2, photYmin / 2);
    photoTile->SetVertex(3, photThick / 2, -photYmin / 2);
    photoTile->SetVertex(4, -photThick / 2, -photYmax / 2);
    photoTile->SetVertex(5, -photThick / 2, photYmax / 2);
    photoTile->SetVertex(6, photThick / 2, photYmax / 2);
    photoTile->SetVertex(7, photThick / 2, -photYmax / 2);

    TGeoVolume* photoTileVol = new TGeoVolume(Form("photoTile_%d_%d", rPosId, photTileCount), photoTile, medSi);
    photoTileVol->SetLineColor(kOrange);
    photoTileVol->SetFillColor(kOrange);
    photoTileVol->SetLineWidth(1);

    auto* rotPhoto = new TGeoRotation(Form("photoTileRotation_%d_%d", photTileCount, rPosId));
    rotPhoto->RotateY(-thetaBDeg);
    rotPhoto->RotateZ(photTileCount * deltaPhiDeg);
    auto* rotTransPhoto = new TGeoCombiTrans(photTransZ * TMath::Cos(photTileCount * TMath::Pi() / (nTilesPhi / 2)),
                                             photTransZ * TMath::Sin(photTileCount * TMath::Pi() / (nTilesPhi / 2)),
                                             photTransZ * TMath::Tan(thetaB),
                                             rotPhoto);

    motherVolume->AddNode(photoTileVol, 1, rotTransPhoto);
    photTileCount++;
  }
}

} // namespace rich
} // namespace o2