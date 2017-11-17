// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Geometry.cxx
/// \brief Implementation of the Geometry class
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "TSystem.h"

#include "FairLogger.h"

#include "MFTBase/Constants.h"
#include "MFTBase/Geometry.h"
#include "MFTBase/GeometryBuilder.h"
#include "MFTBase/Segmentation.h"
#include "MFTBase/HalfSegmentation.h"
#include "MFTBase/HalfDiskSegmentation.h"
#include "MFTBase/LadderSegmentation.h"
#include "MFTBase/ChipSegmentation.h"

using namespace o2::MFT;

ClassImp(o2::MFT::Geometry)


const Double_t Geometry::sSensorInterspace = 0.01; //[cm]  Offset between two adjacent chip on a ladder
const Double_t Geometry::sSensorSideOffset = 0.04; // [cm] Side Offset between the ladder edge and the chip edge
const Double_t Geometry::sSensorTopOffset = 0.04; // [cm] Top Offset between the ladder edge and the chip edge
const Double_t Geometry::sLadderOffsetToEnd = 4.7; // [cm] Offset between the last Chip and the end of the ladder toward the DAQ connector
const Double_t Geometry::sSensorThickness = 30.e-4; // 50 microns
const Double_t Geometry::sChipThickness = 50.e-4; // 50 microns

// Allmost everything you wanted to know about the FPC
const Double_t Geometry::sLineWidth= 100.e-4;         // line width, 100 microns
const Double_t Geometry::sVarnishThickness= 20.e-4;   // 20 micron FPC
const Double_t Geometry::sAluThickness = 25.e-4;      // 25 microns
const Double_t Geometry::sKaptonThickness = 75.e-4;   // 75 microns FPC
const Double_t Geometry::sFlexThickness = sKaptonThickness + 2*sAluThickness + 2*sVarnishThickness; // total thickness of a FPC
const Double_t Geometry::sFlexHeight = 1.68;
const Double_t Geometry::sClearance=300.e-4;      // 300 microns clearance without any conducting metal all around the FPC
const Double_t Geometry::sRadiusHole1=0.125;      // diameter of the FPC crew, closest to the FPC electric connector
const Double_t Geometry::sRadiusHole2=0.1;        // diameter of the FPC pin locator, after the previous hole crew
const Double_t Geometry::sHoleShift1=2.8;        // shift of the FPC crew
const Double_t Geometry::sHoleShift2=3.6;        // shift of the FPC pin locator
const Double_t Geometry::sConnectorOffset=0.4;    // distance between the connector and the start of the FPC
const Double_t Geometry::sCapacitorDx=0.05;
const Double_t Geometry::sCapacitorDy=0.1;
const Double_t Geometry::sCapacitorDz=0.05;
const Double_t Geometry::sConnectorLength=0.1; 
const Double_t Geometry::sConnectorWidth=0.025;
const Double_t Geometry::sConnectorHeight=0.1;
const Double_t Geometry::sConnectorThickness=0.01;
const Double_t Geometry::sShiftDDGNDline=0.4; // positionning of the line to separate AVDD/DVDD et AGND/DGND on the FPC
const Double_t Geometry::sShiftline=0.025; // positionning of the line along the FPC side
const Double_t Geometry::sEpsilon=0.0001; // to see the removed volumes produced by TGeoSubtraction
const Double_t Geometry::sRohacell=-0.001; // to modify the thickness of the rohacell 

const Double_t Geometry::sGlueThickness=100.e-4; // 100 microns of SE4445 to be confirmed
const Double_t Geometry::sGlueEdge=300.e-4; // in case the glue is not spreaded on the whole surface of the sensor

// need to do this, because of the different conventions between 
// ITS and MFT in placing the chips (rows, cols) in the geometry 
// at construction time
//
// xITS    0  +1   0   xMFT
// yITS =  0   0  +1 * yMFT
// zITS   +1   0   0   zMFT
//

TGeoHMatrix Geometry::sTransMFT2ITS = [] {
  TGeoHMatrix tmp;
  Double_t rot[9] = { 0., 1., 0., 
                      0., 0., 1., 
                      1., 0., 0.};
  tmp.SetRotation(rot);
  // equivalent to
  //tmp.RotateY(-90.);
  //tmp.RotateZ(-90.);
  return tmp;
}();

Geometry* Geometry::sInstance = nullptr;

/// \brief Singleton access

//____________________________________________________________________
Geometry* Geometry::instance()
{

  if (!sInstance) sInstance = new Geometry();
  return sInstance;

}

/// \brief Constructor

//_____________________________________________________________________________
Geometry::Geometry():
TNamed("MFT", "Muon Forward Tracker"),
mBuilder(nullptr),
mSegmentation(nullptr),
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
void Geometry::build()
{

  // load the detector segmentation
  if(!mSegmentation) mSegmentation = new Segmentation(gSystem->ExpandPathName("$(VMCWORKDIR)/Detectors/Geometry/MFT/data/Geometry.xml" ));

  // build the geometry
  if (!mBuilder) mBuilder = new GeometryBuilder();
  mBuilder->buildGeometry();
  delete mBuilder;

}

/// \brief Returns the object Unique ID
/// \param [in] type: Type of the object (see Geometry::ObjectTypes)
/// \param [in] half: Half-MFT ID
/// \param [in] disk: Half-Disk ID
/// \param [in] ladder: Ladder ID
/// \param [in] chip: Sensor ID

//_____________________________________________________________________________
UInt_t Geometry::getObjectID(ObjectTypes type, Int_t half, Int_t disk, Int_t plane, Int_t ladder, Int_t chip) const
{

  UInt_t uniqueID = 
    ((type  +1) << 16) + 
    ((half  +1) << 14) + 
    ((disk  +1) << 11) + 
    ((plane +1) <<  9) + 
    ((ladder+1) <<  3) + 
     (chip  +1);

  return uniqueID;

}

/// \brief Returns the number of sensors on the entire disk (top+bottom)
/// \param [in] diskId Int_t: Disk ID = [0,4]

//_____________________________________________________________________________
Int_t Geometry::getDiskNSensors(Int_t diskId) const
{

  Int_t nSensors = 0;
  for (int iHalf=0; iHalf<2; iHalf++) {
    HalfDiskSegmentation * diskSeg = mSegmentation->getHalf(iHalf)->getHalfDisk(diskId);
    if(diskSeg) nSensors += diskSeg->getNChips();

  }
  return nSensors;
}

/// \brief Returns the local ID of the sensor on the disk
/// \param [in] detElemID Int_t: Sensor Unique ID

//_____________________________________________________________________________
Int_t Geometry::getDetElemLocalID(Int_t detElemID) const
{
  
  return  mSegmentation->getDetElemLocalID(getHalfID(detElemID), getDiskID(detElemID), getLadderID(detElemID), getSensorID(detElemID));
  
}

