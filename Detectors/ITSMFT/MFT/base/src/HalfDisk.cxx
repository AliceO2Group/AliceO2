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

/// \file HalfDisk.cxx
/// \brief Class describing geometry of one half of a MFT disk
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "TGeoMatrix.h"
#include "TGeoManager.h"
#include "TGeoBBox.h"

#include <fairlogger/Logger.h>

#include "MFTBase/HalfDiskSegmentation.h"
#include "MFTBase/Ladder.h"
#include "MFTBase/HalfDisk.h"
#include "MFTBase/Geometry.h"
#include "MFTBase/HeatExchanger.h"
#include "MFTBase/Support.h"
#include "MFTBase/PCBSupport.h"
#include "MFTBase/MFTBaseParam.h"

using namespace o2::mft;

ClassImp(o2::mft::HalfDisk);

/// \brief Default constructor

//_____________________________________________________________________________
HalfDisk::HalfDisk()
  : TNamed(), mPCBSupport(nullptr), mSupport(nullptr), mHeatExchanger(nullptr), mHalfDiskVolume(nullptr), mSegmentation(nullptr)
{
}

/// \brief Constructor

//_____________________________________________________________________________
HalfDisk::HalfDisk(HalfDiskSegmentation* segmentation)
  : TNamed(segmentation->GetName(), segmentation->GetName()),
    mPCBSupport(nullptr),
    mSupport(nullptr),
    mHeatExchanger(nullptr),
    mSegmentation(segmentation)
{
  Geometry* mftGeom = Geometry::instance();
  SetUniqueID(mSegmentation->GetUniqueID());
  auto& mftBaseParam = MFTBaseParam::Instance();

  LOG(debug1) << "HalfDisk " << Form("creating half-disk: %s Unique ID = %d ", GetName(), mSegmentation->GetUniqueID());

  mHalfDiskVolume = new TGeoVolumeAssembly(GetName());

  // Building Heat Exchanger Between faces
  if (mftBaseParam.buildHeatExchanger) {
    TGeoVolumeAssembly* heatExchangerVol = createHeatExchanger();
    mHalfDiskVolume->AddNode(heatExchangerVol, 1);
  }

  if (!mftBaseParam.minimal && mftBaseParam.buildPCBSupports) {
    // Building Support
    TGeoVolumeAssembly* supportVol = createSupport();
    mHalfDiskVolume->AddNode(supportVol, 1);
  }

  if (!mftBaseParam.minimal && mftBaseParam.buildPCBs) {
    // Building PCB
    TGeoVolumeAssembly* PCBVol = createPCBSupport();
    mHalfDiskVolume->AddNode(PCBVol, 1);
  }
  // Building Front Face of the Half Disk
  createLadders();
}

//_____________________________________________________________________________
HalfDisk::~HalfDisk()
{

  delete mSupport;
  delete mHeatExchanger;
}

/// \brief Build Heat exchanger
/// \return Pointer to the volume assembly holding the heat exchanger

//_____________________________________________________________________________
TGeoVolumeAssembly* HalfDisk::createHeatExchanger()
{

  Geometry* mftGeom = Geometry::instance();

  mHeatExchanger = new HeatExchanger();

  TGeoVolumeAssembly* vol =
    mHeatExchanger->create(mftGeom->getHalfID(GetUniqueID()), mftGeom->getDiskID(GetUniqueID()));

  return vol;
}

//_____________________________________________________________________________
TGeoVolumeAssembly* HalfDisk::createSupport()
{

  Geometry* mftGeom = Geometry::instance();

  mSupport = new Support();

  TGeoVolumeAssembly* vol =
    mSupport->create(mftGeom->getHalfID(GetUniqueID()), mftGeom->getDiskID(GetUniqueID()));

  return vol;
}

//_____________________________________________________________________________
TGeoVolumeAssembly* HalfDisk::createPCBSupport()
{

  Geometry* mftGeom = Geometry::instance();

  mPCBSupport = new PCBSupport();

  TGeoVolumeAssembly* vol =
    mPCBSupport->create(mftGeom->getHalfID(GetUniqueID()), mftGeom->getDiskID(GetUniqueID()));

  return vol;
}

/// \brief Build Ladders on the Half-disk

//_____________________________________________________________________________
void HalfDisk::createLadders()
{

  LOG(debug1) << "CreateLadders: start building ladders";
  for (Int_t iLadder = 0; iLadder < mSegmentation->getNLadders(); iLadder++) {

    LadderSegmentation* ladderSeg = mSegmentation->getLadder(iLadder);
    if (!ladderSeg) {
      Fatal("CreateLadders", Form("No Segmentation found for ladder %d ", iLadder), 0, 0);
    }
    auto* ladder = new Ladder(ladderSeg);
    TGeoVolume* ladVol = ladder->createVolume();

    // Position of the center on the ladder volume in the ladder coordinate system
    TGeoBBox* shape = (TGeoBBox*)ladVol->GetShape();
    Double_t center[3];
    center[0] = shape->GetDX();
    center[1] = shape->GetDY();
    center[2] = shape->GetDZ();

    Double_t master[3];
    ladderSeg->getTransformation()->LocalToMaster(center, master);
    Int_t ladderId = Geometry::instance()->getLadderID(ladderSeg->GetUniqueID());

    mHalfDiskVolume->AddNode(ladVol, ladderId, new TGeoCombiTrans(master[0], master[1], master[2], ladderSeg->getTransformation()->GetRotation()));

    delete ladder;
  }
}
