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

using namespace AliceO2::MFT;

/// \cond CLASSIMP
ClassImp(AliceO2::MFT::HalfDisk)
/// \endcond

/// \brief Default constructor

//_____________________________________________________________________________
HalfDisk::HalfDisk():
TNamed(), 
mSupport(NULL),
mHeatExchanger(NULL),
mHalfDiskVolume(NULL),
mSegmentation(NULL)
{
  
}

/// \brief Constructor

//_____________________________________________________________________________
HalfDisk::HalfDisk(HalfDiskSegmentation *segmentation):TNamed(segmentation->GetName(),segmentation->GetName()),
  mSupport(NULL),
  mHeatExchanger(NULL),
  mSegmentation(segmentation)
{
  Geometry * mftGeom = Geometry::Instance();
  SetUniqueID(mSegmentation->GetUniqueID());

  LOG(DEBUG1) << "HalfDisk " << Form("creating half-disk: %s Unique ID = %d ", GetName()) << FairLogger::endl;

  mHalfDiskVolume = new TGeoVolumeAssembly(GetName());
  
  // Building MFT Support and PCBs
  /*  
  fSupport = new Support();
  TGeoVolumeAssembly * mftSupport = fSupport->CreateVolume(mftGeom->GetHalfMFTID(GetUniqueID()),mftGeom->GetHalfDiskID(GetUniqueID()));  
  fHalfDiskVolume->AddNode(mftSupport,1);
  */
  // Building Heat Exchanger Between faces
  TGeoVolumeAssembly * heatExchangerVol = CreateHeatExchanger();
  mHalfDiskVolume->AddNode(heatExchangerVol,1);
  	
  // Building Front Face of the Half Disk
  CreateLadders();
  
}

//_____________________________________________________________________________
HalfDisk::~HalfDisk() {

  delete mSupport;
  delete mHeatExchanger;

}

/// \brief Build Heat exchanger
/// \return Pointer to the volume assembly holding the heat exchanger

//_____________________________________________________________________________
TGeoVolumeAssembly * HalfDisk::CreateHeatExchanger()
{
  
  Geometry * mftGeom = Geometry::Instance();

  mHeatExchanger = new HeatExchanger();
  
  TGeoVolumeAssembly * vol = mHeatExchanger->Create(mftGeom->GetHalfMFTID(GetUniqueID()), mftGeom->GetHalfDiskID(GetUniqueID()));
  
  return vol;
  
}

/// \brief Build Ladders on the Half-disk

//_____________________________________________________________________________
void HalfDisk::CreateLadders()
{

  LOG(DEBUG1) << "CreateLadders: start building ladders" << FairLogger::endl;
  for (Int_t iLadder=0; iLadder<mSegmentation->GetNLadders(); iLadder++) {
    
    LadderSegmentation * ladderSeg = mSegmentation->GetLadder(iLadder);
    if(!ladderSeg) Fatal("CreateLadders",Form("No Segmentation found for ladder %d ",iLadder),0,0);
    auto * ladder = new Ladder(ladderSeg);
    TGeoVolume * ladVol = ladder->CreateVolume();
    
    // Position of the center on the ladder volume in the ladder coordinate system
    TGeoBBox* shape = (TGeoBBox*)ladVol->GetShape();
    Double_t center[3];
    center[0] = shape->GetDX();
    center[1] = shape->GetDY();
    center[2] = shape->GetDZ();

    Double_t master[3];
    ladderSeg->GetTransformation()->LocalToMaster(center, master);
    Int_t ladderId = Geometry::Instance()->GetLadderID(ladderSeg->GetUniqueID());
    
    mHalfDiskVolume->AddNode(ladVol,ladderId,new TGeoCombiTrans(master[0],master[1],master[2],ladderSeg->GetTransformation()->GetRotation()));
    
    delete ladder;
  }

}
