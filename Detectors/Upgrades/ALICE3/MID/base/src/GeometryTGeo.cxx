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

#include <MI3Base/GeometryTGeo.h>
#include <TGeoManager.h>

namespace o2::mi3
{
std::unique_ptr<o2::mi3::GeometryTGeo> GeometryTGeo::sInstance;

std::string GeometryTGeo::sVolumeName = "MIDV";
std::string GeometryTGeo::sLayerName = "MIDLayer";
std::string GeometryTGeo::sStaveName = "MIDStave";
std::string GeometryTGeo::sModuleName = "MIDModule";
std::string GeometryTGeo::sSensorName = "MIDSensor";

GeometryTGeo::GeometryTGeo(bool build, int loadTrans) : DetMatrixCache()
{
  if (sInstance) {
    LOGP(fatal, "Invalid use of public constructor: o2::mi3::GeometryTGeo instance exists");
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

const char* GeometryTGeo::composeSymNameLayer(const int layer)
{
  return Form("%s/%s%d", composeSymNameMID(0), getMIDLayerPattern(), layer);
}

const char* GeometryTGeo::composeSymNameStave(const int layer,
                                              const int stave)
{
  return Form("%s/%s%d", composeSymNameLayer(layer), getMIDStavePattern(), stave);
}

const char* GeometryTGeo::composeSymNameModule(const int layer,
                                               const int stave,
                                               const int module)
{
  return Form("%s/%s%d", composeSymNameStave(layer, stave), getMIDModulePattern(), module);
}

const char* GeometryTGeo::composeSymNameSensor(const int layer,
                                               const int stave,
                                               const int module,
                                               const int sensor)
{
  return Form("%s/%s%d", composeSymNameModule(layer, stave, module), getMIDSensorPattern(), sensor);
}
} // namespace o2::mi3