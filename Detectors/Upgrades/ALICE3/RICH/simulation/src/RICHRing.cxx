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
           float radThick,
           float photThick,
           const string& motherName)
  : mNTiles{nTiles}, mPosId{rPosId}, mRadThickness{radThick}
{
  TGeoManager* geoManager = gGeoManager;
  geoManager->GetVolume(motherName.c_str());
  TGeoMedium* medAerogel = gGeoManager->GetMedium("RCH_AEROGEL$");
  if (!medAerogel) {
    LOGP(fatal, "Aerogel medium not found");
  }
  std::vector<TGeoArb8*> aeroTiles(nTilesPhi);
  LOGP(info, "Creating ring: id: {} with {} tiles. ", rPosId, nTilesPhi);

  const float deltaPhi = TMath::TwoPi() / nTilesPhi;
  for (auto& aeroTile : aeroTiles) {
    tile = new TGeoArb8(mRadThickness);
    tile->SetVertex(0, 0, 0);
    tile->SetVertex(1, 0, 0);
    tile->SetVertex(2, 0, 0);
    tile->SetVertex(3, 0, 0);
    tile->SetVertex(4, 0, 0);
    tile->SetVertex(5, 0, 0);
    tile->SetVertex(6, 0, 0);
    tile->SetVertex(7, 0, 0);
  }
}

} // namespace rich
} // namespace o2