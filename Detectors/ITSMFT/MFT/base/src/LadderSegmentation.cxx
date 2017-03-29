/// \file LadderSegmentation.cxx
/// \brief Description of the virtual segmentation of a ladder
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "FairLogger.h"

#include "MFTBase/Constants.h"
#include "MFTBase/LadderSegmentation.h"
#include "MFTBase/ChipSegmentation.h"
#include "MFTBase/Geometry.h"
#include "MFTBase/GeometryTGeo.h"

using namespace AliceO2::MFT;

/// \cond CLASSIMP
ClassImp(LadderSegmentation);
/// \endcond

/// Default constructor

//_____________________________________________________________________________
LadderSegmentation::LadderSegmentation():
  VSegmentation(),
  mChips(NULL)
{


}

/// Constructor
/// \param [in] uniqueID UInt_t: Unique ID of the Ladder to build

//_____________________________________________________________________________
LadderSegmentation::LadderSegmentation(UInt_t uniqueID):
  VSegmentation(),
  mChips(NULL)
{

  SetUniqueID(uniqueID);

  Geometry * mftGeom = Geometry::Instance();
  
  SetName(Form("%s_%d_%d_%d",GeometryTGeo::GetLadderName(),
               mftGeom->GetHalfMFTID(GetUniqueID()),
               mftGeom->GetHalfDiskID(GetUniqueID()),
               mftGeom->GetLadderID(GetUniqueID()) ));

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
  else mChips = new TClonesArray("AliceO2::MFT::ChipSegmentation",mNSensors);

  mChips->SetOwner(kTRUE);
	
}

/// Creates the Sensors Segmentation array on the Ladder

//_____________________________________________________________________________
void LadderSegmentation::CreateSensors() {
  
  if (!mChips) {
    mChips = new TClonesArray("AliceO2::MFT::ChipSegmentation",mNSensors);
    mChips -> SetOwner(kTRUE);
  }

  Geometry * mftGeom = Geometry::Instance();

  for (Int_t iSensor=0; iSensor<mNSensors; iSensor++) {
    UInt_t sensorUniqueID = mftGeom->GetObjectID(Geometry::kSensorType,
                                                 mftGeom->GetHalfMFTID(GetUniqueID()),
                                                 mftGeom->GetHalfDiskID(GetUniqueID()),
                                                 mftGeom->GetLadderID(GetUniqueID()),
                                                 iSensor);
    
    auto *chip = new ChipSegmentation(sensorUniqueID);

    new ((*mChips)[iSensor]) ChipSegmentation(*chip);
    delete chip;
  }

}

/// Returns pointer to a sensor segmentation
/// \param [in] sensorID Int_t: ID of the sensor on the ladder

//_____________________________________________________________________________
ChipSegmentation* LadderSegmentation::GetSensor(Int_t sensorID) const {
  
  if (sensorID<0 || sensorID>=mNSensors) return NULL;
  
  ChipSegmentation *chip = (ChipSegmentation*) mChips->At(sensorID);
  
  return chip;
  
}

/// Print out Ladder information (position, orientation, # of sensors)
/// \param [in] opt "s" or "sensor" -> The individual sensor information will be printed out as well

//_____________________________________________________________________________
void LadderSegmentation::Print(Option_t* opt){
  
  //AliInfo(Form("Ladder %s (Unique ID = %d)",GetName(),GetUniqueID()));
  GetTransformation()->Print();
  //AliInfo(Form("N Sensors = %d",GetNSensors()));
  if(opt && (strstr(opt,"sensor")||strstr(opt,"s"))){
    for (int i=0; i<GetNSensors(); i++)  GetSensor(i)->Print("");

  }
  
}

