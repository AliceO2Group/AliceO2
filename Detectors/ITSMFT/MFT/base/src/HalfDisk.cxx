// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include "FairLogger.h"

#include "MFTBase/HalfDiskSegmentation.h"
#include "MFTBase/Ladder.h"
#include "MFTBase/HalfDisk.h"
#include "MFTBase/Geometry.h"
#include "MFTBase/HeatExchanger.h"
#include "MFTBase/Support.h"

using namespace o2::MFT;

ClassImp(o2::MFT::HalfDisk)

/// \brief Default constructor

//_____________________________________________________________________________
HalfDisk::HalfDisk():
TNamed(), 
mSupport(nullptr),
mHeatExchanger(nullptr),
mHalfDiskVolume(nullptr),
mSegmentation(nullptr)
{
  
}

/// \brief Constructor

//_____________________________________________________________________________
HalfDisk::HalfDisk(HalfDiskSegmentation *segmentation):TNamed(segmentation->GetName(),segmentation->GetName()),
  mSupport(nullptr),
  mHeatExchanger(nullptr),
  mSegmentation(segmentation)
{
  Geometry * mftGeom = Geometry::instance();
  SetUniqueID(mSegmentation->GetUniqueID());

  LOG(DEBUG1) << "HalfDisk " << Form("creating half-disk: %s Unique ID = %d ", GetName(), mSegmentation->GetUniqueID()) << FairLogger::endl;

  mHalfDiskVolume = new TGeoVolumeAssembly(GetName());
  /*  
  // Building MFT Support and PCBs
  mSupport = new Support();
  TGeoVolumeAssembly * mftSupport = mSupport->createVolume(mftGeom->getHalfID(GetUniqueID()),mftGeom->getDiskID(GetUniqueID()));  
  mHalfDiskVolume->AddNode(mftSupport,1);
  */
  // Building Heat Exchanger Between faces
  TGeoVolumeAssembly * heatExchangerVol = createHeatExchanger();
  mHalfDiskVolume->AddNode(heatExchangerVol,1);
        
  // Building Front Face of the Half Disk
  createLadders();
  
}

//_____________________________________________________________________________
HalfDisk::~HalfDisk() {

  delete mSupport;
  delete mHeatExchanger;

}

/// \brief Build Heat exchanger
/// \return Pointer to the volume assembly holding the heat exchanger

//_____________________________________________________________________________
TGeoVolumeAssembly * HalfDisk::createHeatExchanger()
{
  
  Geometry * mftGeom = Geometry::instance();

  mHeatExchanger = new HeatExchanger();
  
  TGeoVolumeAssembly * vol = mHeatExchanger->create(mftGeom->getHalfID(GetUniqueID()), mftGeom->getDiskID(GetUniqueID()));
  
  return vol;
  
}

/// \brief Build Ladders on the Half-disk

//_____________________________________________________________________________
void HalfDisk::createLadders()
{

  LOG(DEBUG1) << "CreateLadders: start building ladders" << FairLogger::endl;
  for (Int_t iLadder=0; iLadder<mSegmentation->getNLadders(); iLadder++) {
    
    LadderSegmentation * ladderSeg = mSegmentation->getLadder(iLadder);
    if(!ladderSeg) Fatal("CreateLadders",Form("No Segmentation found for ladder %d ",iLadder),0,0);
    auto * ladder = new Ladder(ladderSeg);
    TGeoVolume * ladVol = ladder->createVolume();
    
    // Position of the center on the ladder volume in the ladder coordinate system
    TGeoBBox* shape = (TGeoBBox*)ladVol->GetShape();
    Double_t center[3];
    center[0] = shape->GetDX();
    center[1] = shape->GetDY();
    center[2] = shape->GetDZ();

    Double_t master[3];
    ladderSeg->getTransformation()->LocalToMaster(center, master);
    Int_t ladderId = Geometry::instance()->getLadderID(ladderSeg->GetUniqueID());
    
    mHalfDiskVolume->AddNode(ladVol,ladderId,new TGeoCombiTrans(master[0],master[1],master[2],ladderSeg->getTransformation()->GetRotation()));
    
    delete ladder;
  }

}
