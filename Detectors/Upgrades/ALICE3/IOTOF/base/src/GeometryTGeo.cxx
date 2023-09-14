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

#include <IOTOFBase/GeometryTGeo.h>
#include <TGeoManager.h>

namespace o2
{
namespace iotof
{
std::unique_ptr<o2::iotof::GeometryTGeo> GeometryTGeo::sInstance;

// Common i/oTOF
std::string GeometryTGeo::sIOTOFVolumeName = "IOTOFV";

// Inner TOF
std::string GeometryTGeo::sITOFLayerName = "ITOFLayer";
std::string GeometryTGeo::sITOFChipName = "ITOFChip";
std::string GeometryTGeo::sITOFSensorName = "ITOFSensor";

// Outer TOF
std::string GeometryTGeo::sOTOFLayerName = "OTOFLayer";
std::string GeometryTGeo::sOTOFChipName = "OTOFChip";
std::string GeometryTGeo::sOTOFSensorName = "OTOFSensor";

// Forward TOF
std::string GeometryTGeo::sFTOFLayerName = "FTOFLayer";
std::string GeometryTGeo::sFTOFChipName = "FTOFChip";
std::string GeometryTGeo::sFTOFSensorName = "FTOFSensor";

// Backward TOF
std::string GeometryTGeo::sBTOFLayerName = "BTOFLayer";
std::string GeometryTGeo::sBTOFChipName = "BTOFChip";
std::string GeometryTGeo::sBTOFSensorName = "BTOFSensor";

GeometryTGeo::GeometryTGeo(bool build, int loadTrans) : DetMatrixCache()
{
  if (sInstance) {
    LOGP(fatal, "Invalid use of public constructor: o2::iotof::GeometryTGeo instance exists");
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

// // Inner TOF
// const char* GeometryTGeo::composeITOFSymNameLayer(int d, int lr)
// {
//   return Form("%s/%s%d", composeSymNameIOTOF(d), getITOFLayerPattern(), lr);
// }

// const char* GeometryTGeo::composeITOFSymNameChip(int d, int lr)
// {
//   return Form("%s/%s%d", composeITOFSymNameLayer(d, lr), getITOFChipPattern(), lr);
// }

// const char* GeometryTGeo::composeITOFSymNameSensor(int d, int lr)
// {
//   return Form("%s/%s%d", composeITOFSymNameChip(d, lr), getITOFSensorPattern(), lr);
// }

// // Outer TOF
// const char* GeometryTGeo::composeOTOFSymNameLayer(int d, int lr)
// {
//   return Form("%s/%s%d", composeSymNameIOTOF(d), getOTOFLayerPattern(), lr);
// }

// const char* GeometryTGeo::composeOTOFSymNameChip(int d, int lr)
// {
//   return Form("%s/%s%d", composeOTOFSymNameLayer(d, lr), getOTOFChipPattern(), lr);
// }

// const char* GeometryTGeo::composeOTOFSymNameSensor(int d, int lr)
// {
//   return Form("%s/%s%d", composeOTOFSymNameChip(d, lr), getOTOFSensorPattern(), lr);
// }

// // Forward TOF
// const char* GeometryTGeo::composeFTOFSymNameLayer(int d, int lr)
// {
//   return Form("%s/%s%d", composeSymNameIOTOF(d), getFTOFLayerPattern(), lr);
// }

// const char* GeometryTGeo::composeFTOFSymNameChip(int d, int lr)
// {
//   return Form("%s/%s%d", composeFTOFSymNameLayer(d, lr), getFTOFChipPattern(), lr);
// }

// const char* GeometryTGeo::composeFTOFSymNameSensor(int d, int lr)
// {
//   return Form("%s/%s%d", composeFTOFSymNameChip(d, lr), getFTOFSensorPattern(), lr);
// }

// // Backward TOF
// const char* GeometryTGeo::composeBTOFSymNameLayer(int d, int lr)
// {
//   return Form("%s/%s%d", composeSymNameIOTOF(d), getBTOFLayerPattern(), lr);
// }

// const char* GeometryTGeo::composeBTOFSymNameChip(int d, int lr)
// {
//   return Form("%s/%s%d", composeBTOFSymNameLayer(d, lr), getBTOFChipPattern(), lr);
// }

// const char* GeometryTGeo::composeBTOFSymNameSensor(int d, int lr)
// {
//   return Form("%s/%s%d", composeBTOFSymNameChip(d, lr), getBTOFSensorPattern(), lr);
// }

} // namespace iotof
} // namespace o2