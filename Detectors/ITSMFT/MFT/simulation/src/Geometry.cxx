/// \file Geometry.cxx
/// \brief Implementation of the Geometry class
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "TSystem.h"

#include "MFTBase/Constants.h"
#include "MFTSimulation/Geometry.h"
#include "MFTSimulation/GeometryBuilder.h"
#include "MFTSimulation/Segmentation.h"
#include "MFTSimulation/HalfSegmentation.h"
#include "MFTSimulation/HalfDiskSegmentation.h"
#include "MFTSimulation/LadderSegmentation.h"
#include "MFTSimulation/ChipSegmentation.h"

using namespace AliceO2::MFT;

/// \cond CLASSIMP
ClassImp(AliceO2::MFT::Geometry)
/// \endcond

Geometry* Geometry::fgInstance = 0;

/// \brief Singleton access

//____________________________________________________________________
Geometry* Geometry::Instance()
{

  if (!fgInstance) fgInstance = new Geometry();
  return fgInstance;

}

/// \bried Constructor

//_____________________________________________________________________________
Geometry::Geometry():
TNamed("MFT", "Muon Forward Tracker"),
fBuilder(0),
fSegmentation(0),
fSensorVolumeID(0)
{
  // default constructor

}

//_____________________________________________________________________________
Geometry::~Geometry()
{
  // destructor

  delete fBuilder;
  delete fSegmentation;

}

//_____________________________________________________________________________
void Geometry::Build()
{

  // load the detector segmentation
  if(!fSegmentation) fSegmentation = new Segmentation(gSystem->ExpandPathName("$(ALICE_ROOT)/ITSMFT/MFT/data/AliMFTGeometry.xml" ));

  // build the geometry
  if (!fBuilder) fBuilder = new GeometryBuilder();
  fBuilder->BuildGeometry();
  delete fBuilder;

}

/// \brief Returns the object Unique ID
/// \param [in] type: Type of the object (see Geometry::ObjectTypes)
/// \param [in] half: Half-MFT ID
/// \param [in] disk: Half-Disk ID
/// \param [in] ladder: Ladder ID
/// \param [in] chip: Sensor ID

//_____________________________________________________________________________
UInt_t Geometry::GetObjectID(ObjectTypes type, Int_t half, Int_t disk, Int_t ladder, Int_t chip) const
{

  UInt_t uniqueID = (type<<14) +  (half<<13) + (disk<<10) + (ladder<<4) + chip;

  return uniqueID;

}

/// \brief Returns the pixel ID corresponding to a hit at (x,y,z) in the ALICE global frame
///
/// \param [in] xHit Double_t : x Position of the Hit
/// \param [in] yHit Double_t : y Position of the Hit
/// \param [in] zHit Double_t : z Position of the Hit
/// \param [in] detElemID Int_t : Sensor Unique ID in which the hit occured
///
/// \param [out] xPixel Int_t : x position of the pixel hit on the sensor matrix
/// \param [out] yPixel Int_t : y position of the pixel hit on the sensor matrix
/// \retval <kTRUE> if hit into the active part of the sensor
/// \retval <kFALSE> if hit outside the active part

//_____________________________________________________________________________
Bool_t Geometry::Hit2PixelID(Double_t xHit, Double_t yHit, Double_t zHit, Int_t detElemID, Int_t &xPixel, Int_t &yPixel) const
{

  return (fSegmentation->Hit2PixelID(xHit, yHit, zHit, GetHalfID(detElemID), GetHalfDiskID(detElemID), GetLadderID(detElemID), GetSensorID(detElemID), xPixel, yPixel));

}

/// \brief Returns the center of the pixel position in the ALICE global frame
///
/// \param [in] xPixel Int_t : x position of the pixel hit on the sensor matrix
/// \param [in] yPixel Int_t : y position of the pixel hit on the sensor matrix
/// \param [in] detElemID Int_t : Sensor Unique ID in which the hit occured
/// \param [out] xCenter,yCenter,zCenter Double_t : (x,y,z) Position of the Hit in ALICE global frame

//_____________________________________________________________________________
void Geometry::GetPixelCenter(Int_t xPixel, Int_t yPixel, Int_t detElemID, Double_t &xCenter, Double_t &yCenter, Double_t &zCenter ) const
{

  Double_t local[3];
  local[0] = (0.5+xPixel) * Constants::kXPixelPitch + Constants::kSensorMargin;
  local[1] = (0.5+yPixel) * Constants::kYPixelPitch + (Constants::kSensorHeight-Constants::kSensorActiveHeight+ Constants::kSensorMargin);
  local[2] = Constants::kSensorThickness/2.;

  Double_t master[3];
  
  HalfSegmentation * halfSeg = fSegmentation->GetHalf(GetHalfID(detElemID));
  HalfDiskSegmentation * diskSeg = halfSeg->GetHalfDisk(GetHalfDiskID(detElemID));
  LadderSegmentation * ladderSeg = diskSeg->GetLadder(GetLadderID(detElemID));
  ChipSegmentation * chipSeg = ladderSeg->GetSensor(GetSensorID(detElemID));

  chipSeg->GetTransformation()->LocalToMaster(local, master);
  for (int i=0; i<3; i++) local[i] = master[i];
  ladderSeg->GetTransformation()->LocalToMaster(local, master);
  for (int i=0; i<3; i++) local[i] = master[i];
  diskSeg->GetTransformation()->LocalToMaster(local, master);
  for (int i=0; i<3; i++) local[i] = master[i];
  halfSeg->GetTransformation()->LocalToMaster(local, master);

  xCenter = master[0];
  yCenter = master[1];
  zCenter = master[2];

}

/// \brief Returns the number of sensors on the entire disk (top+bottom)
/// \param [in] diskId Int_t: Disk ID = [0,4]

//_____________________________________________________________________________
Int_t Geometry::GetDiskNSensors(Int_t diskId) const
{

  Int_t nSensors = 0;
  for (int iHalf=0; iHalf<2; iHalf++) {
    HalfDiskSegmentation * diskSeg = fSegmentation->GetHalf(iHalf)->GetHalfDisk(diskId);
    if(diskSeg) nSensors += diskSeg->GetNChips();

  }
  return nSensors;
}

/// \brief Returns the local ID of the sensor on the disk
/// \param [in] detElemID Int_t: Sensor Unique ID

//_____________________________________________________________________________
Int_t Geometry::GetDetElemLocalID(Int_t detElemID) const
{
  
  return  fSegmentation->GetDetElemLocalID(GetHalfID(detElemID), GetHalfDiskID(detElemID), GetLadderID(detElemID), GetSensorID(detElemID));
  
}

