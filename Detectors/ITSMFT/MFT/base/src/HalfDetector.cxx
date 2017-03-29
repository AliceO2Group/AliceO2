/// \file HalfDetector.cxx
/// \brief Class Building the geometry of one half of the ALICE Muon Forward Tracker
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "TGeoMatrix.h"

#include "FairLogger.h"

#include "MFTBase/HalfDiskSegmentation.h"
#include "MFTBase/HalfSegmentation.h"
#include "MFTBase/HalfDisk.h"
#include "MFTBase/Geometry.h"
#include "MFTBase/HalfDetector.h"

using namespace AliceO2::MFT;

/// \cond CLASSIMP
ClassImp(AliceO2::MFT::HalfDetector)
/// \endcond

/// \brief Default constructor

//_____________________________________________________________________________
HalfDetector::HalfDetector():
TNamed(),
mHalfVolume(NULL),
mSegmentation(NULL)
{
  
}

/// \brief Constructor

//_____________________________________________________________________________
HalfDetector::HalfDetector(HalfSegmentation *seg):
TNamed(),
mHalfVolume(NULL),
mSegmentation(seg)
{
  
  Geometry * mftGeom = Geometry::Instance();
  
  SetUniqueID(mSegmentation->GetUniqueID());
  
  SetName(Form("MFT_H_%d",mftGeom->GetHalfMFTID(GetUniqueID())));
    
  Info("HalfDetector",Form("Creating : %s ",GetName()),0,0);

  mHalfVolume = new TGeoVolumeAssembly(GetName());
  
  CreateHalfDisks();

}

//_____________________________________________________________________________
HalfDetector::~HalfDetector() 
= default;

/// \brief Creates the Half-disks composing the Half-MFT 

//_____________________________________________________________________________
void HalfDetector::CreateHalfDisks()
{

  Info("CreateHalfDisks",Form("Creating  %d Half-Disk ",mSegmentation->GetNHalfDisks()),0,0);
  
  for (Int_t iDisk = 0 ; iDisk < mSegmentation->GetNHalfDisks(); iDisk++) {
    HalfDiskSegmentation * halfDiskSeg = mSegmentation->GetHalfDisk(iDisk);    
    auto * halfDisk = new HalfDisk(halfDiskSeg);
    Int_t halfDiskId = Geometry::Instance()->GetHalfDiskID(halfDiskSeg->GetUniqueID());
    mHalfVolume->AddNode(halfDisk->GetVolume(),halfDiskId,halfDiskSeg->GetTransformation());
    delete halfDisk;
  }
  
}
