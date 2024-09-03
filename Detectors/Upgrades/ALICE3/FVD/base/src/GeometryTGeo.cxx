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

#include "FVDBase/GeometryTGeo.h"
#include "FVDBase/FVDBaseParam.h"

#include <cmath>

#include <fairlogger/Logger.h>

using namespace o2::fvd;
namespace o2 
{
namespace fvd
{

std::unique_ptr<o2::fvd::GeometryTGeo> GeometryTGeo::sInstance;

GeometryTGeo::GeometryTGeo(bool build, int loadTrans) : DetMatrixCache()
{
  if (sInstance) {
    LOGP(fatal, "Invalid use of public constructor: o2::fvd::GeometryTGeo instance exists");
  }
  if (build) {
    Build(loadTrans);
  }
}

GeometryTGeo::~GeometryTGeo() {}

GeometryTGeo* GeometryTGeo::Instance()
{
  if (!sInstance) {
    sInstance = std::unique_ptr<GeometryTGeo>(new GeometryTGeo(true, 0));
  }
  return sInstance.get();
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

int GeometryTGeo::getCellId(int nmod, int nring, int nsec) const
{
   return nmod * FVDBaseParam::nCellA + 8 * nring +  nsec;
}

int GeometryTGeo::getCurrentCellId(const TVirtualMC* fMC) const
{
  int moduleId = -1;
  int sectorId = -1;
  int ringId = -1;

  fMC->CurrentVolOffID(2, moduleId);
  fMC->CurrentVolOffID(1, sectorId);
  fMC->CurrentVolOffID(0, ringId);
  int cellId = getCellId(moduleId, ringId, sectorId); 

  return cellId;
}

} // namespace fvd
} //namespace o2
