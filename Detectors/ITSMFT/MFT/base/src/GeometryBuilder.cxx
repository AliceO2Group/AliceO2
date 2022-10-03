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
#include "MFTBase/Barrel.h"
#include "MFTBase/PatchPanel.h"
#include "MFTBase/MFTBaseParam.h"
#include "MFTBase/PowerSupplyUnit.h"

#include "TGeoVolume.h"
#include "TGeoManager.h"

#include <fairlogger/Logger.h>

using namespace o2::mft;

ClassImp(o2::mft::GeometryBuilder);

//_____________________________________________________________________________
/// \brief Build the MFT Geometry
void GeometryBuilder::buildGeometry()
{

  Geometry* mftGeo = Geometry::instance();
  auto& mftBaseParam = MFTBaseParam::Instance();
  TGeoVolume* volMFT = new TGeoVolumeAssembly(GeometryTGeo::getMFTVolPattern());

  LOG(info) << "GeometryBuilder::buildGeometry volume name = " << GeometryTGeo::getMFTVolPattern();

  TGeoVolume* vALIC = gGeoManager->GetVolume("barrel");
  if (!vALIC) {
    LOG(fatal) << "Could not find the top volume";
  }

  LOG(debug) << "buildGeometry: "
             << Form("gGeoManager name is %s title is %s", gGeoManager->GetName(), gGeoManager->GetTitle());

  Segmentation* seg = mftGeo->getSegmentation();

  for (int iHalf = 0; iHalf < 2; iHalf++) {
    HalfSegmentation* halfSeg = seg->getHalf(iHalf);
    auto* halfMFT = new HalfDetector(halfSeg);
    volMFT->AddNode(halfMFT->getVolume(), iHalf, halfSeg->getTransformation());
    delete halfMFT;
  }

  // Add the service, Barrel, etc
  if (!mftBaseParam.minimal && mftBaseParam.buildPSU) {
    auto* mPSU = new PowerSupplyUnit();
    TGeoVolumeAssembly* mHalfPSU = mPSU->create();
    TGeoTranslation* t_PSU = new TGeoTranslation("t_PSU", 0, 0.0, -72.6);
    t_PSU->RegisterYourself();
    auto* r_PSU = new TGeoRotation("rotation_PSU", 0.0, 0.0, 180.0);
    auto* p_PSU = new TGeoCombiTrans(*t_PSU, *r_PSU);
    volMFT->AddNode(mHalfPSU, 1, t_PSU);
    volMFT->AddNode(mHalfPSU, 2, p_PSU);
  }

  if (!mftBaseParam.minimal && mftBaseParam.buildCone) {
    auto* halfCone = new HalfCone();
    TGeoVolumeAssembly* halfCone1 = halfCone->createHalfCone(0);
    TGeoVolumeAssembly* halfCone2 = halfCone->createHalfCone(1);
    volMFT->AddNode(halfCone1, 1);
    volMFT->AddNode(halfCone2, 1);
  }

  if (!mftBaseParam.minimal && mftBaseParam.buildBarrel) {
    //barrel services
    auto* t_barrel0 = new TGeoTranslation("translation_barrel", 0.0, 0.7, -80.17);
    auto* r_barrel0 = new TGeoRotation("rotation_barrel", 0.0, 0.0, 0.0);
    auto* p_barrel0 = new TGeoCombiTrans(*t_barrel0, *r_barrel0);
    auto* t_barrel1 = new TGeoTranslation("translation_barrel", 0.0, 0.7, -80.17);
    auto* r_barrel1 = new TGeoRotation("rotation_barrel", 0.0, 0.0, 180.0);
    auto* p_barrel1 = new TGeoCombiTrans(*t_barrel1, *r_barrel1);

    auto* halfBarrel = new Barrel();
    TGeoVolumeAssembly* halfBarrel0 = halfBarrel->createBarrel();
    volMFT->AddNode(halfBarrel0, 1, p_barrel0);
    TGeoVolumeAssembly* halfBarrel1 = halfBarrel->createBarrel();
    volMFT->AddNode(halfBarrel1, 1, p_barrel1);
  }

  if (!mftBaseParam.minimal && mftBaseParam.buildPatchPanel) {
    auto* t_patchpanel0 = new TGeoTranslation("translation_patchpanel", 0.0, 0., -81.5); //z (0,0.7, -81.5 -1.3; 0..81.7 --1.5
    auto* r_patchpanel0 = new TGeoRotation("rotation_patchpanel", 0.0, 0.0, 0.0);
    auto* p_patchpanel0 = new TGeoCombiTrans(*t_patchpanel0, *r_patchpanel0);
    auto* t_patchpanel1 = new TGeoTranslation("translation_patchpanel", 0.0, 0., -81.5); //z( 0, 0.7, -81.5-1.3; 0.. 81.7 --1.5
    auto* r_patchpanel1 = new TGeoRotation("rotation_patchpanel", 0.0, 0.0, 180.0);
    auto* p_patchpanel1 = new TGeoCombiTrans(*t_patchpanel1, *r_patchpanel1);

    auto* halfpatchpanel = new PatchPanel();
    TGeoVolumeAssembly* halfpatchpanel0 = halfpatchpanel->createPatchPanel();
    TGeoVolumeAssembly* halfpatchpanel1 = halfpatchpanel->createPatchPanel();
    volMFT->AddNode(halfpatchpanel0, 1, p_patchpanel0);
    volMFT->AddNode(halfpatchpanel1, 1, p_patchpanel1);
  }

  vALIC->AddNode(volMFT, 0, new TGeoTranslation(0., 30., 0.));
}
