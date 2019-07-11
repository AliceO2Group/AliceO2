// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GeometryBuilder.cxx
/// \brief Class describing MFT Geometry Builder
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "MFTBase/Geometry.h"
#include "MFTBase/GeometryTGeo.h"
#include "MFTBase/GeometryBuilder.h"
#include "MFTBase/Segmentation.h"
#include "MFTBase/HalfSegmentation.h"
#include "MFTBase/HalfDetector.h"
#include "MFTBase/HalfCone.h"

#include "TGeoVolume.h"
#include "TGeoManager.h"

#include "FairLogger.h"

using namespace o2::mft;

ClassImp(o2::mft::GeometryBuilder);

//_____________________________________________________________________________
/// \brief Build the MFT Geometry
void GeometryBuilder::buildGeometry()
{

  Geometry* mftGeo = Geometry::instance();

  TGeoVolume* volMFT = new TGeoVolumeAssembly(GeometryTGeo::getMFTVolPattern());

  LOG(INFO) << "GeometryBuilder::buildGeometry volume name = " << GeometryTGeo::getMFTVolPattern();

  TGeoVolume* vALIC = gGeoManager->GetVolume("cave");
  if (!vALIC) {
    LOG(FATAL) << "Could not find the top volume";
  }

  LOG(DEBUG) << "buildGeometry: "
             << Form("gGeoManager name is %s title is %s", gGeoManager->GetName(), gGeoManager->GetTitle());

  Segmentation* seg = mftGeo->getSegmentation();

  for (int iHalf = 0; iHalf < 2; iHalf++) {
    HalfSegmentation* halfSeg = seg->getHalf(iHalf);
    auto* halfMFT = new HalfDetector(halfSeg);
    volMFT->AddNode(halfMFT->getVolume(), iHalf, halfSeg->getTransformation());
    delete halfMFT;
  }

  /// \todo Add the service, Barrel, etc Those objects will probably be defined into the COMMON ITSMFT area.

  auto* halfCone = new HalfCone();
  TGeoVolumeAssembly* halfCone1 = halfCone->createHalfCone(0);
  TGeoVolumeAssembly* halfCone2 = halfCone->createHalfCone(1);
  volMFT->AddNode(halfCone1, 1);
  volMFT->AddNode(halfCone2, 1);

  vALIC->AddNode(volMFT, 0);
}
