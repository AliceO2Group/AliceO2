/// \file Ladder.cxx
/// \brief Ladder builder
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "TGeoMatrix.h"
#include "TGeoManager.h"
#include "TGeoBBox.h"
#include "TGeoTube.h"
#include "TGeoCompositeShape.h"
#include "TGeoBoolNode.h"

#include "FairLogger.h"

#include "MFTBase/Constants.h"
#include "MFTSimulation/LadderSegmentation.h"
#include "MFTSimulation/ChipSegmentation.h"
#include "MFTSimulation/Flex.h"
#include "MFTSimulation/Chip.h"
#include "MFTSimulation/Ladder.h"
#include "MFTSimulation/Geometry.h"

using namespace AliceO2::MFT;

/// \cond CLASSIMP
ClassImp(AliceO2::MFT::Ladder)
/// \endcond

// Units are cm
const Double_t Ladder::kLadderDeltaY = Constants::kSensorHeight + 2.*Constants::kSensorTopOffset;
const Double_t Ladder::kLadderDeltaZ = Constants::kFlexThickness + Constants::kSensorThickness; // TODO: Adjust that value when adding glue layer

/// \brief Default constructor

//_____________________________________________________________________________
Ladder::Ladder():
TNamed(), 
fSegmentation(NULL),
fFlex(NULL),
fLadderVolume(NULL)
{
    
}

/// \brief Constructor

//_____________________________________________________________________________
Ladder::Ladder(LadderSegmentation *segmentation):
TNamed(segmentation->GetName(),segmentation->GetName()),
fSegmentation(segmentation), 
fFlex(NULL)
{

  LOG(DEBUG1) << "Ladder " << Form("creating : %s", GetName()) << FairLogger::endl;
  fLadderVolume = new TGeoVolumeAssembly(GetName());
  
}


//_____________________________________________________________________________
Ladder::~Ladder() 
{

  delete fFlex;
  
}

/// \brief Build the ladder

//_____________________________________________________________________________
TGeoVolume * Ladder::CreateVolume() 
{

  Int_t nChips = fSegmentation->GetNSensors();

  // Create the flex
  fFlex = new Flex(fSegmentation);     
  Double_t kFlexLength = nChips*(Constants::kSensorLength+Constants::kSensorInterspace)+Constants::kLadderOffsetToEnd + Constants::kSensorSideOffset;
  Double_t kShiftY = 2*Constants::kSensorTopOffset+Constants::kSensorHeight-Constants::kFlexHeight/2; // strange
  TGeoVolumeAssembly * flexVol = fFlex->MakeFlex(fSegmentation->GetNSensors(), kFlexLength);                               
  fLadderVolume->AddNode(flexVol, 1, new TGeoTranslation(kFlexLength/2+Constants::kSensorSideOffset/2, kShiftY, Constants::kFlexThickness/2));     

  // Create the CMOS Sensors
  CreateSensors();

  return fLadderVolume;
  
}

/// \brief Build the sensors

//_____________________________________________________________________________
void Ladder::CreateSensors() 
{

  // Create Shapes
  
  // The sensor part
  TGeoBBox *sensor = new TGeoBBox(Constants::kSensorLength/2., Constants::kSensorActiveHeight/2., Constants::kSensorThickness/2.);
  
  // The readout part
  TGeoBBox *readout = new TGeoBBox(Constants::kSensorLength/2.,(Constants::kSensorHeight-Constants::kSensorActiveHeight)/2.,  Constants::kSensorThickness/2.);
  
  // Get Mediums
  TGeoMedium *medSensorSi  = gGeoManager->GetMedium("MFT_Si$");
  TGeoMedium *medReadoutSi = gGeoManager->GetMedium("MFT_Readout$");
  TGeoMedium *medAir  = gGeoManager->GetMedium("MFT_Air$");
  //TGeoMedium *kMedGlue = gGeoManager->GetMedium("MFT_Epoxy$"); 
  TGeoMedium *kMedGlue = gGeoManager->GetMedium("MFT_SE4445$"); 
  
  Geometry * mftGeom = Geometry::Instance();
  
  TString namePrefix = Form("MFT_S_%d_%d_%d",
	  mftGeom->GetHalfID(fSegmentation->GetUniqueID()),
	  mftGeom->GetHalfDiskID(fSegmentation->GetUniqueID()),
	  mftGeom->GetLadderID(fSegmentation->GetUniqueID()) );
  
  TGeoVolume * chipVol = gGeoManager->MakeBox(namePrefix.Data(), medAir,Constants::kSensorLength/2.,Constants::kSensorHeight/2., Constants::kSensorThickness/2.);
  TGeoVolume * glue = gGeoManager->MakeBox(namePrefix.Data(), kMedGlue, (Constants::kSensorLength-Constants::kGlueEdge)/2., (Constants::kSensorHeight-Constants::kGlueEdge)/2., Constants::kGlueThickness/2.);
  glue->SetVisibility(kTRUE);
  glue->SetLineColor(kRed-10);
  glue->SetLineWidth(1);
  glue->SetFillColor(glue->GetLineColor());
  glue->SetFillStyle(4000); // 0% transparent

  // Create Volumes
  // Chip Volume
  chipVol->SetVisibility(kTRUE);

  // The sensor Volume
  TGeoVolume *sensorVol = new TGeoVolume("MFTSensor", sensor, medSensorSi);
  sensorVol->SetVisibility(kTRUE);
  
  sensorVol->SetLineColor(kGreen+1);
  sensorVol->SetLineWidth(1);
  sensorVol->SetFillColor(sensorVol->GetLineColor());
  sensorVol->SetFillStyle(4000); // 0% transparent
  
  if(!mftGeom->GetSensorVolumeID()){
    mftGeom->SetSensorVolumeID(sensorVol->GetNumber());
  } else if (mftGeom->GetSensorVolumeID() != sensorVol->GetNumber()){
    Fatal("CreateSensors",Form("Different Sensor VOLUME ID in TGeo !!!!"),0,0);
  }
  
  // The Readout Volume
  TGeoVolume *readoutVol = new TGeoVolume("Readout", readout, medReadoutSi);
  readoutVol->SetVisibility(kTRUE);
  readoutVol->SetLineColor(kRed-6);
  readoutVol->SetLineWidth(1);
  readoutVol->SetFillColor(readoutVol->GetLineColor());
  readoutVol->SetFillStyle(4000); // 0% transparent

  // Building up the chip
  chipVol->AddNode(readoutVol, 1, new TGeoTranslation(0.,-Constants::kSensorHeight/2.+readout->GetDY(),  0.));
  chipVol->AddNode(sensorVol, 1, new TGeoTranslation( 0., Constants::kSensorHeight/2.-sensor->GetDY(),0.));

  for (int ichip =0; ichip<fSegmentation->GetNSensors(); ichip++) {
    ChipSegmentation * chipSeg = fSegmentation->GetSensor(ichip);
    TGeoCombiTrans * chipPos = chipSeg->GetTransformation();
    TGeoCombiTrans * chipPosGlue = chipSeg->GetTransformation();
    // Position of the center on the chip in the chip coordinate system
    Double_t pos[3] ={Constants::kSensorLength/2., Constants::kSensorHeight/2., Constants::kSensorThickness/2. - Constants::kGlueThickness};
    Double_t posglue[3] ={Constants::kSensorLength/2., Constants::kSensorHeight/2., Constants::kGlueThickness/2-Constants::kSensorThickness};
    Double_t master[3];
    Double_t masterglue[3];
    chipPos->LocalToMaster(pos, master);
    chipPosGlue->LocalToMaster(posglue, masterglue);
    
    TGeoBBox* shape = (TGeoBBox*)fLadderVolume->GetShape();
    master[0] -= shape->GetDX();
    master[1] -= shape->GetDY();
    master[2] -= shape->GetDZ();

    masterglue[0] -= shape->GetDX();
    masterglue[1] -= shape->GetDY();
    masterglue[2] -= shape->GetDZ();

    LOG(DEBUG1) << "CreateSensors " << Form("adding chip %s_%d ",namePrefix.Data(),ichip) << FairLogger::endl;
    fLadderVolume->AddNode(chipVol, ichip, new TGeoTranslation(master[0],master[1],master[2]));
    fLadderVolume->AddNode(glue, ichip, new TGeoTranslation(masterglue[0],masterglue[1],masterglue[2]));

  }

}
