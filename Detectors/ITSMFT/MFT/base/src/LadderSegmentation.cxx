// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file LadderSegmentation.cxx
/// \brief Description of the virtual segmentation of a ladder
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "FairLogger.h"

#include "MFTBase/Constants.h"
#include "MFTBase/LadderSegmentation.h"
#include "MFTBase/ChipSegmentation.h"
#include "MFTBase/Geometry.h"
#include "MFTBase/GeometryTGeo.h"

using namespace o2::MFT;

ClassImp(LadderSegmentation);

/// Default constructor

//_____________________________________________________________________________
LadderSegmentation::LadderSegmentation():
  VSegmentation(),
  mChips(nullptr)
{


}

/// Constructor
/// \param [in] uniqueID UInt_t: Unique ID of the Ladder to build

//_____________________________________________________________________________
LadderSegmentation::LadderSegmentation(UInt_t uniqueID):
  VSegmentation(),
  mChips(nullptr)
{

  SetUniqueID(uniqueID);

  Geometry * mftGeom = Geometry::instance();
  
  SetName(Form("%s_%d_%d_%d",GeometryTGeo::getMFTLadderPattern(),
               mftGeom->getHalfID(GetUniqueID()),
               mftGeom->getDiskID(GetUniqueID()),
               mftGeom->getLadderID(GetUniqueID()) ));

  // constructor
  
}

/// Copy Constructor

//_____________________________________________________________________________
LadderSegmentation::LadderSegmentation(const LadderSegmentation& ladder):
  VSegmentation(ladder),
  mNSensors(ladder.mNSensors)
{
  // copy constructor
  
  if (ladder.mChips) mChips = new TClonesArray(*(ladder.mChips));
  else mChips = new TClonesArray("o2::MFT::ChipSegmentation",mNSensors);

  mChips->SetOwner(kTRUE);
        
}

/// Creates the Sensors Segmentation array on the Ladder

//_____________________________________________________________________________
void LadderSegmentation::createSensors() {
  
  if (!mChips) {
    mChips = new TClonesArray("o2::MFT::ChipSegmentation",mNSensors);
    mChips -> SetOwner(kTRUE);
  }

  Geometry * mftGeom = Geometry::instance();

  for (Int_t iSensor=0; iSensor<mNSensors; iSensor++) {
    UInt_t sensorUniqueID = mftGeom->getObjectID(Geometry::SensorType,
                           mftGeom->getHalfID(GetUniqueID()),
                           mftGeom->getDiskID(GetUniqueID()),
                           mftGeom->getPlaneID(GetUniqueID()),
                           mftGeom->getLadderID(GetUniqueID()),
                           iSensor);
    
    auto *chip = new ChipSegmentation(sensorUniqueID);

    new ((*mChips)[iSensor]) ChipSegmentation(*chip);
    delete chip;
  }

}

/// Returns pointer to a sensor segmentation
/// \param [in] sensorID Int_t: ID of the sensor on the ladder

//_____________________________________________________________________________
ChipSegmentation* LadderSegmentation::getSensor(Int_t sensorID) const {
  
  if (sensorID<0 || sensorID>=mNSensors) return nullptr;
  
  ChipSegmentation *chip = (ChipSegmentation*) mChips->At(sensorID);
  
  return chip;
  
}

/// Print out Ladder information (position, orientation, # of sensors)
/// \param [in] opt "s" or "sensor" -> The individual sensor information will be printed out as well

//_____________________________________________________________________________
void LadderSegmentation::print(Option_t* opt){
  
  getTransformation()->Print();
  if(opt && (strstr(opt,"sensor")||strstr(opt,"s"))){
    for (int i=0; i<getNSensors(); i++)  getSensor(i)->Print("");
  }
  
}

