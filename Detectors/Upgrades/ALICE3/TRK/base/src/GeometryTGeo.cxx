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

#include <TRKBase/GeometryTGeo.h>
#include <TGeoManager.h>

namespace o2
{
namespace trk
{
std::unique_ptr<o2::trk::GeometryTGeo> GeometryTGeo::sInstance;

// Names
std::string GeometryTGeo::sVolumeName = "TRKV";
std::string GeometryTGeo::sLayerName = "TRKLayer";
std::string GeometryTGeo::sChipName = "TRKChip";
std::string GeometryTGeo::sSensorName = "TRKSensor";

GeometryTGeo::GeometryTGeo(bool build, int loadTrans) : DetMatrixCache()
{
  if (sInstance) {
    LOGP(fatal, "Invalid use of public constructor: o2::trk::GeometryTGeo instance exists");
  }
  if (build) {
    Build(loadTrans);
  }
}

void GeometryTGeo::Build(int loadTrans)
{
  if (isBuilt()) {
    LOGP(warning, "Already built");
    return; // already initialized
  }

  if (!gGeoManager) {
    LOGP(fatal, "Geometry is not loaded");
  }

  fillMatrixCache(loadTrans);
}

void GeometryTGeo::fillMatrixCache(int mask)
{
}

GeometryTGeo* GeometryTGeo::Instance()
{
  if (!sInstance) {
    sInstance = std::unique_ptr<GeometryTGeo>(new GeometryTGeo(true, 0));
  }
  return sInstance.get();
}

const char* GeometryTGeo::composeSymNameLayer(int d, int lr)
{
  return Form("%s/%s%d", composeSymNameTRK(d), getTRKLayerPattern(), lr);
}

const char* GeometryTGeo::composeSymNameChip(int d, int lr)
{
  return Form("%s/%s%d", composeSymNameLayer(d, lr), getTRKChipPattern(), lr);
}

const char* GeometryTGeo::composeSymNameSensor(int d, int lr)
{
  return Form("%s/%s%d", composeSymNameChip(d, lr), getTRKSensorPattern(), lr);
}

} // namespace trk
} // namespace o2