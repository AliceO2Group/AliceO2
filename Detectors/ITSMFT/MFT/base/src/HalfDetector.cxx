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
fHalfVolume(NULL),
fSegmentation(NULL)
{
  
}

/// \brief Constructor

//_____________________________________________________________________________
HalfDetector::HalfDetector(HalfSegmentation *seg):
TNamed(),
fHalfVolume(NULL),
fSegmentation(seg)
{
  
  Geometry * mftGeom = Geometry::Instance();
  
  SetUniqueID(fSegmentation->GetUniqueID());
  
  SetName(Form("MFT_H_%d",mftGeom->GetHalfMFTID(GetUniqueID())));
    
  Info("HalfDetector",Form("Creating : %s ",GetName()),0,0);

  fHalfVolume = new TGeoVolumeAssembly(GetName());
  
  CreateHalfDisks();

}

//_____________________________________________________________________________
HalfDetector::~HalfDetector() 
{

  
}

/// \brief Creates the Half-disks composing the Half-MFT 

//_____________________________________________________________________________
void HalfDetector::CreateHalfDisks()
{

  Info("CreateHalfDisks",Form("Creating  %d Half-Disk ",fSegmentation->GetNHalfDisks()),0,0);
  
  for (Int_t iDisk = 0 ; iDisk < fSegmentation->GetNHalfDisks(); iDisk++) {
    HalfDiskSegmentation * halfDiskSeg = fSegmentation->GetHalfDisk(iDisk);    
    HalfDisk * halfDisk = new HalfDisk(halfDiskSeg);
    Int_t halfDiskId = Geometry::Instance()->GetHalfDiskID(halfDiskSeg->GetUniqueID());
    fHalfVolume->AddNode(halfDisk->GetVolume(),halfDiskId,halfDiskSeg->GetTransformation());
    delete halfDisk;
  }
  
}
