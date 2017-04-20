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
#include "MFTBase/LadderSegmentation.h"
#include "MFTBase/ChipSegmentation.h"
#include "MFTBase/Flex.h"
#include "MFTBase/Chip.h"
#include "MFTBase/Ladder.h"
#include "MFTBase/Geometry.h"

using namespace o2::MFT;

ClassImp(o2::MFT::Ladder)

// Units are cm
const Double_t Ladder::sLadderDeltaY = Geometry::sSensorHeight + 2.*Geometry::sSensorTopOffset;
const Double_t Ladder::sLadderDeltaZ = Geometry::sFlexThickness + Geometry::sSensorThickness; // TODO: Adjust that value when adding glue layer

/// \brief Default constructor

//_____________________________________________________________________________
Ladder::Ladder():
TNamed(), 
mSegmentation(nullptr),
mFlex(nullptr),
mLadderVolume(nullptr)
{
    
}

/// \brief Constructor

//_____________________________________________________________________________
Ladder::Ladder(LadderSegmentation *segmentation):
TNamed(segmentation->GetName(),segmentation->GetName()),
mSegmentation(segmentation), 
mFlex(nullptr)
{

  LOG(DEBUG1) << "Ladder " << Form("creating : %s", GetName()) << FairLogger::endl;
  mLadderVolume = new TGeoVolumeAssembly(GetName());
  
}


//_____________________________________________________________________________
Ladder::~Ladder() 
{

  delete mFlex;
  
}

/// \brief Build the ladder

//_____________________________________________________________________________
TGeoVolume * Ladder::createVolume() 
{

  Int_t nChips = mSegmentation->getNSensors();
  
  // Create the flex
  mFlex = new Flex(mSegmentation);     
  Double_t flexLength = nChips*(Geometry::sSensorLength+Geometry::sSensorInterspace)+Geometry::sLadderOffsetToEnd + Geometry::sSensorSideOffset;
  Double_t shiftY = 2*Geometry::sSensorTopOffset+Geometry::sSensorHeight-Geometry::sFlexHeight/2; // strange
  TGeoVolumeAssembly * flexVol = mFlex->makeFlex(mSegmentation->getNSensors(), flexLength);                               
  mLadderVolume->AddNode(flexVol, 1, new TGeoTranslation(flexLength/2+Geometry::sSensorSideOffset/2, shiftY, Geometry::sFlexThickness/2-Geometry::sRohacell));     
  
  // Create the CMOS Sensors
  createSensors();

  return mLadderVolume;
  
}

/// \brief Build the sensors

//_____________________________________________________________________________
void Ladder::createSensors() 
{

  // Create Shapes
  
  // The sensor part
  auto *sensor = new TGeoBBox(Geometry::sSensorLength/2., Geometry::sSensorActiveHeight/2., Geometry::sSensorThickness/2.);
  
  // The readout part
  auto *readout = new TGeoBBox(Geometry::sSensorLength/2.,(Geometry::sSensorHeight-Geometry::sSensorActiveHeight)/2.,  Geometry::sSensorThickness/2.);
  
  // Get Mediums
  TGeoMedium *medSensorSi  = gGeoManager->GetMedium("MFT_Si$");
  TGeoMedium *medReadoutSi = gGeoManager->GetMedium("MFT_Readout$");
  TGeoMedium *medAir  = gGeoManager->GetMedium("MFT_Air$");
  //TGeoMedium *kMedGlue = gGeoManager->GetMedium("MFT_Epoxy$"); 
  TGeoMedium *kMedGlue = gGeoManager->GetMedium("MFT_SE4445$"); 
  
  Geometry * mftGeom = Geometry::instance();
  
  TString namePrefix = Form("MFT_S_%d_%d_%d",
	  mftGeom->getHalfMFTID(mSegmentation->GetUniqueID()),
	  mftGeom->getHalfDiskID(mSegmentation->GetUniqueID()),
	  mftGeom->getLadderID(mSegmentation->GetUniqueID()) );
  
  TGeoVolume * chipVol = gGeoManager->MakeBox(namePrefix.Data(), medAir,Geometry::sSensorLength/2.,Geometry::sSensorHeight/2., Geometry::sSensorThickness/2.);
  TGeoVolume * glue = gGeoManager->MakeBox(namePrefix.Data(), kMedGlue, (Geometry::sSensorLength-Geometry::sGlueEdge)/2., (Geometry::sSensorHeight-Geometry::sGlueEdge)/2., Geometry::sGlueThickness/2.);
  glue->SetVisibility(kTRUE);
  glue->SetLineColor(kRed-10);
  glue->SetLineWidth(1);
  glue->SetFillColor(glue->GetLineColor());
  glue->SetFillStyle(4000); // 0% transparent

  // Create Volumes
  // Chip Volume
  chipVol->SetVisibility(kTRUE);

  // The sensor Volume
  auto *sensorVol = new TGeoVolume("MFTSensor", sensor, medSensorSi);
  sensorVol->SetVisibility(kTRUE);
  
  sensorVol->SetLineColor(kGreen+1);
  sensorVol->SetLineWidth(1);
  sensorVol->SetFillColor(sensorVol->GetLineColor());
  sensorVol->SetFillStyle(4000); // 0% transparent
  
  if(!mftGeom->getSensorVolumeID()){
    mftGeom->setSensorVolumeID(sensorVol->GetNumber());
  } else if (mftGeom->getSensorVolumeID() != sensorVol->GetNumber()){
    Fatal("CreateSensors",Form("Different Sensor VOLUME ID in TGeo !!!!"),0,0);
  }
  
  // The Readout Volume
  auto *readoutVol = new TGeoVolume("Readout", readout, medReadoutSi);
  readoutVol->SetVisibility(kTRUE);
  readoutVol->SetLineColor(kRed-6);
  readoutVol->SetLineWidth(1);
  readoutVol->SetFillColor(readoutVol->GetLineColor());
  readoutVol->SetFillStyle(4000); // 0% transparent

  // Building up the chip
  chipVol->AddNode(readoutVol, 1, new TGeoTranslation(0.,-Geometry::sSensorHeight/2.+readout->GetDY(),  0.));
  chipVol->AddNode(sensorVol, 1, new TGeoTranslation( 0., Geometry::sSensorHeight/2.-sensor->GetDY(),0.));

  for (int ichip = 0; ichip < mSegmentation->getNSensors(); ichip++) {
    ChipSegmentation * chipSeg = mSegmentation->getSensor(ichip);
    TGeoCombiTrans * chipPos = chipSeg->getTransformation();
    TGeoCombiTrans * chipPosGlue = chipSeg->getTransformation();
    // Position of the center on the chip in the chip coordinate system
    Double_t pos[3] ={Geometry::sSensorLength/2., Geometry::sSensorHeight/2., Geometry::sSensorThickness/2. - Geometry::sGlueThickness - Geometry::sRohacell};
    Double_t posglue[3] ={Geometry::sSensorLength/2., Geometry::sSensorHeight/2., Geometry::sGlueThickness/2-Geometry::sSensorThickness-Geometry::sRohacell};
    Double_t master[3];
    Double_t masterglue[3];
    chipPos->LocalToMaster(pos, master);
    chipPosGlue->LocalToMaster(posglue, masterglue);
    
    TGeoBBox* shape = (TGeoBBox*)mLadderVolume->GetShape();
    master[0] -= shape->GetDX();
    master[1] -= shape->GetDY();
    master[2] -= shape->GetDZ();

    masterglue[0] -= shape->GetDX();
    masterglue[1] -= shape->GetDY();
    masterglue[2] -= shape->GetDZ();

    LOG(DEBUG1) << "CreateSensors " << Form("adding chip %s_%d ",namePrefix.Data(),ichip) << FairLogger::endl;
    mLadderVolume->AddNode(chipVol, ichip, new TGeoTranslation(master[0],master[1],master[2]));
    mLadderVolume->AddNode(glue, ichip, new TGeoTranslation(masterglue[0],masterglue[1],masterglue[2]));

  }

}
