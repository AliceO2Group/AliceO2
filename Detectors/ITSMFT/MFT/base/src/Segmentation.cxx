/// \file Segmentation.cxx
/// \brief Class for the virtual segmentation of the ALICE Muon Forward Tracker
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "FairLogger.h"

#include "MFTBase/LadderSegmentation.h"
#include "MFTBase/HalfDiskSegmentation.h"
#include "MFTBase/HalfSegmentation.h"
#include "MFTBase/Segmentation.h"

using namespace AliceO2::MFT;

/// \cond CLASSIMP
ClassImp(AliceO2::MFT::Segmentation)
/// \endcond

//_____________________________________________________________________________
Segmentation::Segmentation():
  TNamed(),
  fHalves(NULL)
{ 


}

//_____________________________________________________________________________
Segmentation::Segmentation(const Char_t *nameGeomFile): 
  TNamed(),
  fHalves(NULL)
{ 

  // constructor
  
  fHalves = new TClonesArray("AliceO2::MFT::HalfSegmentation", 2);
  fHalves->SetOwner(kTRUE);
  
  HalfSegmentation *halfBottom = new HalfSegmentation(nameGeomFile, kBottom);
  HalfSegmentation *halfTop    = new HalfSegmentation(nameGeomFile, kTop);

  new ((*fHalves)[kBottom]) HalfSegmentation(*halfBottom);
  new ((*fHalves)[kTop])    HalfSegmentation(*halfTop);

  delete halfBottom;
  delete halfTop;
  
  LOG(DEBUG1) << "MFT segmentation set!" << FairLogger::endl;

}

//_____________________________________________________________________________
Segmentation::~Segmentation() {

  if (fHalves) fHalves->Delete();
  delete fHalves; 
  
}

/// \brief Returns pointer to the segmentation of the half-MFT
/// \param iHalf Integer : 0 = Bottom; 1 = Top
/// \return Pointer to a HalfSegmentation

//_____________________________________________________________________________
HalfSegmentation* Segmentation::GetHalf(Int_t iHalf) const 
{ 

  Info("GetHalf",Form("Ask for half %d (of %d and %d)",iHalf,kBottom,kTop),0,0);

  return ((iHalf==kTop || iHalf==kBottom) ? ( (HalfSegmentation*) fHalves->At(iHalf)) :  NULL); 

}

///Clear the TClonesArray holding the HalfSegmentation objects

//_____________________________________________________________________________
void Segmentation::Clear(const Option_t* /*opt*/) {

  if (fHalves) fHalves->Delete();
  delete fHalves; 
  fHalves = NULL;
  
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
Bool_t Segmentation::Hit2PixelID(Double_t xHit, Double_t yHit, Double_t zHit, Int_t half, Int_t disk, Int_t ladder, Int_t sensor, Int_t &xPixel, Int_t &yPixel){

  Double_t master[3] = {xHit, yHit, zHit};
  Double_t local[3];
  HalfSegmentation * halfSeg = ((HalfSegmentation*)fHalves->At(half));
  if(!halfSeg) return kFALSE;
  HalfDiskSegmentation * diskSeg = halfSeg->GetHalfDisk(disk);
  if(!diskSeg) return kFALSE;
  LadderSegmentation * ladderSeg = diskSeg->GetLadder(ladder);
  if(!ladderSeg) return kFALSE;
  ChipSegmentation * chipSeg = ladderSeg->GetSensor(sensor);
  if(!chipSeg) return kFALSE;

  //AliDebug(2,Form(" ->  Global %f %f %f",master[0],master[1],master[2]));
  halfSeg->GetTransformation()->MasterToLocal(master, local);
  //AliDebug(2,Form(" ->  Half %f %f %f",local[0],local[1],local[2]));
  for (int i=0; i<3; i++) master[i] = local[i];
  diskSeg->GetTransformation()->MasterToLocal(master, local);
  //AliDebug(2,Form(" ->  Disk %f %f %f",local[0],local[1],local[2]));
  for (int i=0; i<3; i++) master[i] = local[i];
  ladderSeg->GetTransformation()->MasterToLocal(master, local);
  //AliDebug(2,Form(" ->  Ladder %f %f %f",local[0],local[1],local[2]));
  for (int i=0; i<3; i++) master[i] = local[i];
  chipSeg->GetTransformation()->MasterToLocal(master, local);
  //AliDebug(2,Form(" ->  Chip Pos %f %f %f",local[0],local[1],local[2]));
  
  
   return (chipSeg->Hit2PixelID(local[0], local[1], xPixel, yPixel));

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
Int_t Segmentation::GetDetElemLocalID(Int_t half, Int_t disk, Int_t ladder, Int_t sensor) const {

  Int_t localId =0;
  
  
  if (half==1) localId += GetHalf(0)->GetHalfDisk(disk)->GetNChips();
  
  for (int iLad=0; iLad<GetHalf(half)->GetHalfDisk(disk)->GetNLadders(); iLad++) {
    if (iLad<ladder) localId += GetHalf(half)->GetHalfDisk(disk)->GetLadder(iLad)->GetNSensors();
    else{
      for (int iSens=0; iSens<GetHalf(half)->GetHalfDisk(disk)->GetLadder(iLad)->GetNSensors(); iSens++) {
        if(iSens==sensor) return localId;
        localId++;
     }
    }
  }
  return -1;
}
