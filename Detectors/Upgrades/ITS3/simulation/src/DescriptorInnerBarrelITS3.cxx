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

#include "ITS3Simulation/DescriptorInnerBarrelITS3.h"
#include "ITSBase/GeometryTGeo.h"
#include "Framework/Logger.h"

using namespace o2::its3;

ClassImp(DescriptorInnerBarrelITS3);

void DescriptorInnerBarrelITS3::createLayer(int iLayer, TGeoVolume* dest)
{
  mIBLayers[iLayer] = std::make_unique<ITS3Layer>(iLayer);
  mIBLayers[iLayer]->createLayer(dest);
}

void DescriptorInnerBarrelITS3::createServices(TGeoVolume* dest)
{
  mServices = std::make_unique<ITS3Services>();
  mServices->createCYSSAssembly(dest);
}

void DescriptorInnerBarrelITS3::addAlignableVolumesLayer(int idLayer, int wrapperLayerId, TString& parentPath, int& lastUID) const
{
  TString wrpV = wrapperLayerId != -1 ? Form("%s%d_1", its::GeometryTGeo::getITSWrapVolPattern(), wrapperLayerId) : "";
  TString path = Form("%s/%s/%s%d_0", parentPath.Data(), wrpV.Data(), its::GeometryTGeo::getITS3LayerPattern(), idLayer);
  TString sname = its::GeometryTGeo::composeSymNameLayer(idLayer, true);

  for (int iHalfBarrel{0}; iHalfBarrel < 2; ++iHalfBarrel) {
    addAlignableVolumesHalfBarrel(idLayer, iHalfBarrel, path, lastUID);
  }
}

void DescriptorInnerBarrelITS3::addAlignableVolumesHalfBarrel(int idLayer, int iHB, TString& parentPath, int& lastUID) const
{
  // for ITS3 smallest alignable volume is the half-barrel (e.g., the carbon-form composite structure with the sensors)
  TString path = Form("%s/%s%d_%d", parentPath.Data(), its::GeometryTGeo::getITS3HalfBarrelPattern(), idLayer, iHB);
  TString sname = its::GeometryTGeo::composeSymNameHalfBarrel(idLayer, iHB, true);
  if (!gGeoManager->SetAlignableEntry(sname.Data(), path.Data())) {
    LOG(fatal) << "Unable to set alignable entry ! " << sname << " : " << path;
  }
  addAlignableVolumesChips(idLayer, iHB, path, lastUID);
}

void DescriptorInnerBarrelITS3::addAlignableVolumesChips(int idLayer, int iHB, TString& parentPath, int& lastUID) const
{
  for (int seg{0}; seg < constants::nSegments[idLayer]; ++seg) {
    for (int rsu{0}; rsu < constants::segment::nRSUs; ++rsu) {
      for (int tile{0}; tile < constants::rsu::nTiles; ++tile) {
        TString path = parentPath;
        path += Form("/%s_0/", its::GeometryTGeo::getITS3ChipPattern(idLayer));
        path += Form("%s_%d/", its::GeometryTGeo::getITS3SegmentPattern(idLayer), seg);
        path += Form("%s_%d/", its::GeometryTGeo::getITS3RSUPattern(idLayer), rsu);
        path += Form("%s_%d/", its::GeometryTGeo::getITS3TilePattern(idLayer), tile);
        TString sname = its::GeometryTGeo::composeSymNameChip(idLayer, iHB, 0, seg, rsu, tile, true);
        if (!gGeoManager->SetAlignableEntry(sname.Data(), path.Data())) {
          LOG(fatal) << "Unable to set alignable entry ! " << sname << " : " << path;
        }
        ++lastUID;
      }
    }
  }
}
