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
  fChips(NULL)
{


}

/// Constructor
/// \param [in] uniqueID UInt_t: Unique ID of the Ladder to build

//_____________________________________________________________________________
LadderSegmentation::LadderSegmentation(UInt_t uniqueID):
  VSegmentation(),
  fChips(NULL)
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
  fNSensors(ladder.fNSensors)
{
  // copy constructor
  
  if (ladder.fChips) fChips = new TClonesArray(*(ladder.fChips));
  else fChips = new TClonesArray("AliceO2::MFT::ChipSegmentation",fNSensors);

  fChips->SetOwner(kTRUE);
	
}

/// Creates the Sensors Segmentation array on the Ladder

//_____________________________________________________________________________
void LadderSegmentation::CreateSensors() {
  
  if (!fChips) {
    fChips = new TClonesArray("AliceO2::MFT::ChipSegmentation",fNSensors);
    fChips -> SetOwner(kTRUE);
  }

  Geometry * mftGeom = Geometry::Instance();

  for (Int_t iSensor=0; iSensor<fNSensors; iSensor++) {
    UInt_t sensorUniqueID = mftGeom->GetObjectID(Geometry::kSensorType,
                                                 mftGeom->GetHalfMFTID(GetUniqueID()),
                                                 mftGeom->GetHalfDiskID(GetUniqueID()),
                                                 mftGeom->GetLadderID(GetUniqueID()),
                                                 iSensor);
    
    ChipSegmentation *chip = new ChipSegmentation(sensorUniqueID);

    new ((*fChips)[iSensor]) ChipSegmentation(*chip);
    delete chip;
  }

}

/// Returns pointer to a sensor segmentation
/// \param [in] sensorID Int_t: ID of the sensor on the ladder

//_____________________________________________________________________________
ChipSegmentation* LadderSegmentation::GetSensor(Int_t sensorID) const {
  
  if (sensorID<0 || sensorID>=fNSensors) return NULL;
  
  ChipSegmentation *chip = (ChipSegmentation*) fChips->At(sensorID);
  
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

