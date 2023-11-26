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
           const std::string motherName)
  : mNTiles{nTilesPhi}, mPosId{rPosId}, mRadThickness{radThick}
{
  TGeoManager* geoManager = gGeoManager;
  TGeoVolume* motherVolume = geoManager->GetVolume(motherName.c_str());
  TGeoMedium* medAerogel = gGeoManager->GetMedium("RCH_AEROGEL$");
  if (!medAerogel) {
    LOGP(fatal, "Aerogel medium not found");
  }
  std::vector<TGeoArb8*> aeroTiles(nTilesPhi);
  LOGP(info, "Creating ring: id: {} with {} tiles. ", rPosId, nTilesPhi);

  float deltaPhiRad = 360.0 / nTilesPhi; // Transformation are constructed in degrees...
  size_t tileCount{0};
  for (auto& aeroTile : aeroTiles) {
    aeroTile = new TGeoArb8(radZ / 2);
    aeroTile->SetVertex(0, -radThick / 2, -radYmax / 2);
    aeroTile->SetVertex(1, -radThick / 2, radYmax / 2);
    aeroTile->SetVertex(2, radThick / 2, radYmax / 2);
    aeroTile->SetVertex(3, radThick / 2, -radYmax / 2);
    aeroTile->SetVertex(4, -radThick / 2, -radYmin / 2);
    aeroTile->SetVertex(5, -radThick / 2, radYmin / 2);
    aeroTile->SetVertex(6, radThick / 2, radYmin / 2);
    aeroTile->SetVertex(7, radThick / 2, -radYmin / 2);

    TGeoVolume* aeroTileVol = new TGeoVolume(Form("aeroTile_%d_%d", rPosId, tileCount), aeroTile, medAerogel);
    aeroTileVol->SetLineColor(kBlue - 9);
    aeroTileVol->SetFillColor(kBlue - 9);
    aeroTileVol->SetTransparency(50);
    aeroTileVol->SetLineWidth(1);

    auto* rotAero = new TGeoRotation(Form("aeroTileRotation%d_%d", tileCount, rPosId), 0, 0, tileCount * deltaPhiRad);
    auto* rotoTransAero = new TGeoCombiTrans(rMin * TMath::Cos(tileCount * TMath::Pi() / (nTilesPhi / 2)), rMin * TMath::Sin(tileCount * TMath::Pi() / (nTilesPhi / 2)), 0, rotAero);

    motherVolume->AddNode(aeroTileVol, 1, rotoTransAero);
    tileCount++;
  }
}

} // namespace rich
} // namespace o2