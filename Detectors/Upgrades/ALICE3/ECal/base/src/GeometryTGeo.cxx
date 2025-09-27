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

/// \file GeometryTGeo.cxx
/// \brief Class containing ECal volume naming patterns
///
/// \author Evgeny Kryshen <evgeny.kryshen@cern.ch>

#include <ECalBase/GeometryTGeo.h>
#include <TGeoManager.h>

namespace o2
{
namespace ecal
{
std::unique_ptr<o2::ecal::GeometryTGeo> GeometryTGeo::sInstance;

std::string GeometryTGeo::sVolumeName = "ECALV";
std::string GeometryTGeo::sSectorName = "ECALSector";
std::string GeometryTGeo::sModuleName = "ECALModule";

GeometryTGeo::GeometryTGeo(bool build, int loadTrans) : DetMatrixCache()
{
  if (sInstance) {
    LOGP(fatal, "Invalid use of public constructor: o2::ecal::GeometryTGeo instance exists");
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

const char* GeometryTGeo::composeSymNameSector(int s)
{
  return Form("%s/%s_%d", composeSymNameECal(), getECalSectorPattern(), s);
}

const char* GeometryTGeo::composeSymNameModule(int s, int m)
{
  return Form("%s/%s_%d", composeSymNameSector(s), getECalModulePattern(), m);
}

} // namespace ecal
} // namespace o2
