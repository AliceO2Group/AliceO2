/// \file Geometry.cxx
/// \brief Implementation of the Geometry class
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "TSystem.h"

#include "MFTBase/Constants.h"
#include "MFTBase/Geometry.h"
#include "MFTBase/GeometryBuilder.h"
#include "MFTBase/Segmentation.h"
#include "MFTBase/HalfSegmentation.h"
#include "MFTBase/HalfDiskSegmentation.h"
#include "MFTBase/LadderSegmentation.h"
#include "MFTBase/ChipSegmentation.h"

using namespace AliceO2::MFT;

/// \cond CLASSIMP
ClassImp(AliceO2::MFT::Geometry)
/// \endcond

const Double_t Geometry::kSensorLength=3.; //[cm]
const Double_t Geometry::kSensorHeight=1.5; //[cm]
const Double_t Geometry::kXPixelPitch=29.250e-4; // 29.15 micron // TODO : Check that
const Double_t Geometry::kYPixelPitch=26.880e-4; // 26.88 micron // TODO : Check that
const Double_t Geometry::kSensorMargin=29.120e-4; // 29.12 micron // TODO : Check that

const Double_t Geometry::kSensorActiveWidth  = kNPixelX * kXPixelPitch; //[cm]
const Double_t Geometry::kSensorActiveHeight = kNPixelY * kYPixelPitch; //[cm]

const Double_t Geometry::kSensorInterspace = 0.01; //[cm]  Offset between two adjacent chip on a ladder
const Double_t Geometry::kSensorSideOffset = 0.04; // [cm] Side Offset between the ladder edge and the chip edge
const Double_t Geometry::kSensorTopOffset = 0.04; // [cm] Top Offset between the ladder edge and the chip edge
const Double_t Geometry::kLadderOffsetToEnd = 4.7; // [cm] Offset between the last Chip and the end of the ladder toward the DAQ connector
const Double_t Geometry::kSensorThickness = 50.e-4; // 50 microns

const Double_t Geometry::fHeightActive = 1.3;
const Double_t Geometry::fHeightReadout = 0.2;

// Allmost everything you wanted to know about the FPC
const Double_t Geometry::kLineWidth= 100.e-4;         // line width, 100 microns
const Double_t Geometry::kVarnishThickness= 45.e-4;   // 20 micron FPC + 25 microns of glue for encapsulation
const Double_t Geometry::kAluThickness = 25.e-4;      // 25 microns
const Double_t Geometry::kKaptonThickness = 88.e-4;   // 75 microns FPC + 13 microns of kapton for encapsulation
const Double_t Geometry::kFlexThickness = kKaptonThickness + 2*kAluThickness + 2*kVarnishThickness; // total thickness of a FPC
const Double_t Geometry::kFlexHeight = 1.68;
const Double_t Geometry::kClearance=300.e-4;      // 300 microns clearance without any conducting metal all around the FPC
const Double_t Geometry::kRadiusHole1=0.125;      // diameter of the FPC crew, closest to the FPC electric connector
const Double_t Geometry::kRadiusHole2=0.1;        // diameter of the FPC pin locator, after the previous hole crew
const Double_t Geometry::kHoleShift1=2.8;        // shift of the FPC crew
const Double_t Geometry::kHoleShift2=3.6;        // shift of the FPC pin locator
const Double_t Geometry::kConnectorOffset=0.4;    // distance between the connector and the start of the FPC
const Double_t Geometry::kCapacitorDx=0.05;
const Double_t Geometry::kCapacitorDy=0.1;
const Double_t Geometry::kCapacitorDz=0.05;
const Double_t Geometry::kConnectorLength=0.1; 
const Double_t Geometry::kConnectorWidth=0.025;
const Double_t Geometry::kConnectorHeight=0.1;
const Double_t Geometry::kConnectorThickness=0.01;
const Double_t Geometry::kShiftDDGNDline=0.4; // positionning of the line to separate AVDD/DVDD et AGND/DGND on the FPC
const Double_t Geometry::kShiftline=0.025; // positionning of the line along the FPC side
const Double_t Geometry::kEpsilon=0.0001; // to see the removed volumes produced by TGeoSubtraction
const Double_t Geometry::kRohacell=-0.001; // to modify the thickness of the rohacell 
const Double_t Geometry::kShift=-0.0013; // to be checked


const Double_t Geometry::kGlueThickness=100.e-4; // 100 microns of SE4445 to be confirmed
const Double_t Geometry::kGlueEdge=300.e-4; // in case the glue is not spreaded on the whole surface of the sensor

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
mBuilder(0),
mSegmentation(0),
mSensorVolumeID(0)
{
  // default constructor

}

//_____________________________________________________________________________
Geometry::~Geometry()
{
  // destructor

  delete mBuilder;
  delete mSegmentation;

}

//_____________________________________________________________________________
void Geometry::Build()
{

  // load the detector segmentation
  if(!mSegmentation) mSegmentation = new Segmentation(gSystem->ExpandPathName("$(ALICE_ROOT)/ITSMFT/MFT/data/AliMFTGeometry.xml" ));

  // build the geometry
  if (!mBuilder) mBuilder = new GeometryBuilder();
  mBuilder->BuildGeometry();
  delete mBuilder;

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

  return (mSegmentation->Hit2PixelID(xHit, yHit, zHit, GetHalfMFTID(detElemID), GetHalfDiskID(detElemID), GetLadderID(detElemID), GetSensorID(detElemID), xPixel, yPixel));

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
  local[0] = (0.5+xPixel) * Geometry::kXPixelPitch + Geometry::kSensorMargin;
  local[1] = (0.5+yPixel) * Geometry::kYPixelPitch + (Geometry::kSensorHeight-Geometry::kSensorActiveHeight+ Geometry::kSensorMargin);
  local[2] = Geometry::kSensorThickness/2.;

  Double_t master[3];
  
  HalfSegmentation * halfSeg = mSegmentation->GetHalf(GetHalfMFTID(detElemID));
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
    HalfDiskSegmentation * diskSeg = mSegmentation->GetHalf(iHalf)->GetHalfDisk(diskId);
    if(diskSeg) nSensors += diskSeg->GetNChips();

  }
  return nSensors;
}

/// \brief Returns the local ID of the sensor on the disk
/// \param [in] detElemID Int_t: Sensor Unique ID

//_____________________________________________________________________________
Int_t Geometry::GetDetElemLocalID(Int_t detElemID) const
{
  
  return  mSegmentation->GetDetElemLocalID(GetHalfMFTID(detElemID), GetHalfDiskID(detElemID), GetLadderID(detElemID), GetSensorID(detElemID));
  
}

