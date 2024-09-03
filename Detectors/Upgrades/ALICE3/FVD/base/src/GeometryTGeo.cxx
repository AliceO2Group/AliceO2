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

#include <cmath>

#include <fairlogger/Logger.h>

#include <TGeoBBox.h>
#include <TGeoCompositeShape.h>
#include <TGeoCone.h>
#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <TGeoMedium.h>
#include <TGeoTube.h>
#include <TGeoVolume.h>

using namespace o2::fvd;
namespace o2 
{
namespace fvd
{

std::unique_ptr<o2::fvd::GeometryTGeo> GeometryTGeo::sInstance;

GeometryTGeo::GeometryTGeo() : DetMatrixCache()
{
  if (sInstance) {
    LOGP(fatal, "Invalid use of public constructor: o2::fvd::GeometryTGeo instance exists");
  }
  Build();
}

GeometryTGeo::~GeometryTGeo() {}

void GeometryTGeo::Build() const
{
  // Top volume of FVD detector
  // A side

  TGeoVolume* vCave = gGeoManager->GetVolume("cave");
  if (!vCave) {
     LOG(fatal) << "Could not find the top volume for C-side";
  }

  // A side
  TGeoVolumeAssembly *vFVDA = buildModuleA();
  LOG(info) << "FVD: building geometry. The name of the volume is '" << vFVDA->GetName() << "'";

  vCave->AddNode(vFVDA, 0, new TGeoTranslation(sXGlobal, sYGlobal, sZGlobalA));

  // C side

  TGeoVolumeAssembly *vFVDC =  buildModuleC();
  LOG(info) << "FVD: building geometry. The name of the volume is '" << vFVDC->GetName() << "'";

  vCave->AddNode(vFVDC, 1, new TGeoTranslation(sXGlobal, sYGlobal, sZGlobalC));
}

TGeoVolumeAssembly* GeometryTGeo::buildModuleA() const
{
  TGeoVolumeAssembly* mod = new TGeoVolumeAssembly("FVDA"); // A or C

  const TGeoMedium* medium = gGeoManager->GetMedium("FVD_Scintillator"); 

  const float dphiDeg = 45.;

  for (int ir = 0; ir < sNumberOfCellRingsA; ir++) {
     for (int ic = 0; ic < sNumberOfCellSectors; ic ++) {
	int cellId = ic + ir;
	std::string tbsName = "tbs" + std::to_string(cellId);
	std::string nodeName = "node" + std::to_string(cellId);
	float rmin = sCellRingRadiiA[ir];
	float rmax = sCellRingRadiiA[ir+1];
	float phimin = dphiDeg;
	float phimax = dphiDeg;
        auto tbs = new TGeoTubeSeg(tbsName.c_str(), rmin, rmax, sDzScintillator, phimin, phimax);
	auto nod = new TGeoVolume(nodeName.c_str(), tbs, medium);
	mod->AddNode(nod, cellId);
     }
  }

  return mod;
}

TGeoVolumeAssembly* GeometryTGeo::buildModuleC() const
{
  TGeoVolumeAssembly* mod = new TGeoVolumeAssembly("FVDC"); // A or C

  const TGeoMedium* medium = gGeoManager->GetMedium("FVD_Scintillator"); 

  const float dphiDeg = 45.;

  for (int ir = 0; ir < sNumberOfCellRingsC; ir++) {
     for (int ic = 0; ic < sNumberOfCellSectors; ic ++) {
	int cellId = ic + ir + sNumberOfCellsA;
	std::string tbsName = "tbs" + std::to_string(cellId);
	std::string nodeName = "node" + std::to_string(cellId);
	float rmin = sCellRingRadiiC[ir];
	float rmax = sCellRingRadiiC[ir+1];
	float phimin = dphiDeg;
	float phimax = dphiDeg;
        auto tbs = new TGeoTubeSeg(tbsName.c_str(), rmin, rmax, sDzScintillator, phimin, phimax);
	auto nod = new TGeoVolume(nodeName.c_str(), tbs, medium);
	mod->AddNode(nod, cellId);
     }
  }

  return mod;
}

int GeometryTGeo::getCellId(int nmod, int nring, int nsec) const
{
   return nmod * sNumberOfCellRingsA + 8 * nring +  nsec;
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

void GeometryTGeo::fillMatrixCache(int mask)
{
}



GeometryTGeo* GeometryTGeo::Instance()
{
  if (!sInstance) {
    sInstance = std::unique_ptr<GeometryTGeo>(new GeometryTGeo());
  }
  return sInstance.get();
}

} // namespace fvd
} //namespace o2
