/// \file Segmentation.cxx
/// \brief Class for the virtual segmentation of the ALICE Muon Forward Tracker
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "FairLogger.h"

#include "MFTBase/LadderSegmentation.h"
#include "MFTBase/HalfDiskSegmentation.h"
#include "MFTBase/HalfSegmentation.h"
#include "MFTBase/Segmentation.h"

using namespace o2::MFT;

ClassImp(o2::MFT::Segmentation)

//_____________________________________________________________________________
Segmentation::Segmentation():
  TNamed(),
  mHalves(nullptr)
{ 


}

//_____________________________________________________________________________
Segmentation::Segmentation(const Char_t *nameGeomFile): 
  TNamed(),
  mHalves(nullptr)
{ 

  // constructor
  
  mHalves = new TClonesArray("o2::MFT::HalfSegmentation", 2);
  mHalves->SetOwner(kTRUE);
  
  auto *halfBottom = new HalfSegmentation(nameGeomFile, Bottom);
  auto *halfTop    = new HalfSegmentation(nameGeomFile, Top);

  new ((*mHalves)[Bottom]) HalfSegmentation(*halfBottom);
  new ((*mHalves)[Top])    HalfSegmentation(*halfTop);

  delete halfBottom;
  delete halfTop;
  
  LOG(DEBUG1) << "MFT segmentation set!" << FairLogger::endl;

}

//_____________________________________________________________________________
Segmentation::~Segmentation() {

  if (mHalves) mHalves->Delete();
  delete mHalves; 
  
}

/// \brief Returns pointer to the segmentation of the half-MFT
/// \param iHalf Integer : 0 = Bottom; 1 = Top
/// \return Pointer to a HalfSegmentation

//_____________________________________________________________________________
HalfSegmentation* Segmentation::getHalf(Int_t iHalf) const 
{ 

  Info("GetHalf",Form("Ask for half %d (of %d and %d)",iHalf,Bottom,Top),0,0);

  return ((iHalf==Top || iHalf==Bottom) ? ( (HalfSegmentation*) mHalves->At(iHalf)) :  nullptr); 

}

///Clear the TClonesArray holding the HalfSegmentation objects

//_____________________________________________________________________________
void Segmentation::Clear(const Option_t* /*opt*/) {

  if (mHalves) mHalves->Delete();
  delete mHalves; 
  mHalves = nullptr;
  
}

/// Returns the pixel ID corresponding to a hit at (x,y,z) in the ALICE global frame
///
/// \param [in] xHit Double_t : x Position of the Hit
/// \param [in] yHit Double_t : y Position of the Hit
/// \param [in] zHit Double_t : z Position of the Hit
/// \param [in] sensor Int_t : Sensor ID in which the hit occured
/// \param [in] ladder Int_t : Ladder ID holding the sensor
/// \param [in] disk Int_t : Half-Disk ID holding the ladder
/// \param [in] half Int_t : Half-MFT  ID holding the half-disk
///
/// \param [out] xPixel Int_t : x position of the pixel hit on the sensor matrix
/// \param [out] yPixel Int_t : y position of the pixel hit on the sensor matrix
/// \retval <kTRUE> if hit into the active part of the sensor
/// \retval <kFALSE> if hit outside the active part

//_____________________________________________________________________________
Bool_t Segmentation::hitToPixelID(Double_t xHit, Double_t yHit, Double_t zHit, Int_t half, Int_t disk, Int_t ladder, Int_t sensor, Int_t &xPixel, Int_t &yPixel){

  Double_t master[3] = {xHit, yHit, zHit};
  Double_t local[3];
  HalfSegmentation * halfSeg = ((HalfSegmentation*)mHalves->At(half));
  if(!halfSeg) return kFALSE;
  HalfDiskSegmentation * diskSeg = halfSeg->getHalfDisk(disk);
  if(!diskSeg) return kFALSE;
  LadderSegmentation * ladderSeg = diskSeg->getLadder(ladder);
  if(!ladderSeg) return kFALSE;
  ChipSegmentation * chipSeg = ladderSeg->getSensor(sensor);
  if(!chipSeg) return kFALSE;

  halfSeg->getTransformation()->MasterToLocal(master, local);
  for (int i=0; i<3; i++) master[i] = local[i];
  diskSeg->getTransformation()->MasterToLocal(master, local);
  for (int i=0; i<3; i++) master[i] = local[i];
  ladderSeg->getTransformation()->MasterToLocal(master, local);
  for (int i=0; i<3; i++) master[i] = local[i];
  chipSeg->getTransformation()->MasterToLocal(master, local);
  
  return (chipSeg->hitToPixelID(local[0], local[1], xPixel, yPixel));

}

/// Returns the local ID of the sensor on the entire disk specified
///
/// \param sensor Int_t : Sensor ID
/// \param ladder Int_t : Ladder ID holding the Sensor
/// \param disk Int_t : Half-Disk ID holding the Sensor
/// \param half Int_t : Half-MFT  ID holding the Sensor
///
/// \return A fixed number that represents the ID of the sensor on the disk. It goes from 0 to the max number of sensor on the disk

//_____________________________________________________________________________
Int_t Segmentation::getDetElemLocalID(Int_t half, Int_t disk, Int_t ladder, Int_t sensor) const {

  Int_t localId =0;
  
  
  if (half==1) localId += getHalf(0)->getHalfDisk(disk)->getNChips();
  
  for (int iLad=0; iLad<getHalf(half)->getHalfDisk(disk)->getNLadders(); iLad++) {
    if (iLad<ladder) localId += getHalf(half)->getHalfDisk(disk)->getLadder(iLad)->getNSensors();
    else{
      for (int iSens=0; iSens<getHalf(half)->getHalfDisk(disk)->getLadder(iLad)->getNSensors(); iSens++) {
        if(iSens==sensor) return localId;
        localId++;
     }
    }
  }
  return -1;
}
