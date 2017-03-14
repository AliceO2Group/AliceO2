/// \file Chip.cxx
/// \brief Class describing geometry of MFT CMOS MAP Chip
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "TGeoManager.h"
#include "TGeoBBox.h"

#include "MFTBase/Constants.h"
#include "MFTBase/LadderSegmentation.h"
#include "MFTBase/ChipSegmentation.h"
#include "MFTBase/Chip.h"

using namespace AliceO2::MFT;

/// \cond CLASSIMP
ClassImp(AliceO2::MFT::Chip)
/// \endcond

//_____________________________________________________________________________
Chip::Chip():
TNamed()
{
  
  // default constructor
  
}

//_____________________________________________________________________________
Chip::Chip(ChipSegmentation *segmentation, const char * ladderName):TNamed(ladderName,ladderName)
{
 
}

//_____________________________________________________________________________
Chip::~Chip() 
{
  
}

//_____________________________________________________________________________
void Chip::GetPosition(LadderSegmentation * ladderSeg, Int_t iChip, Double_t *pos)
{

//  Double_t * fFlexDimensions = new Double_t[3];
//  ladderSeg->GetFlexLength(fFlexDimensions);
//  
//  Info("GetPosition",Form("fFlexDimensions %f %f %f",fFlexDimensions[0],fFlexDimensions[1], fFlexDimensions[2]),0,0);
//  
//  pos[0] = Constants::fChipSideOffset + Constants::fChipWidth/2. + iChip*(Constants::fChipWidth+Constants::fChipInterspace);
//  pos[1] = -(Constants::fChipTopOffset + fChipHeight/2.) ;
//  pos[2] =  fFlexDimensions[2] + Constants::fChipThickness/2.;
//  Warning ("GetPosition","---- Z position of Chip to be worked out --- ",0,0);
//  if (!ladderSeg->IsLeftType()) pos[0]  *= -1.;
    
}

//_____________________________________________________________________________
TGeoVolume * Chip::CreateVolume()
{
  
//  // Create Shapes
//  
//  // The sensor part
//  TGeoBBox *sensor = new TGeoBBox(Constants::kSensorLength/2.,Constants::kSensorHeight/2.,  Constants::fChipThickness/2.);
//  
//  // The readout part
//  TGeoBBox *readout = new TGeoBBox(Constants::fChipWidth/2.,(fChipHeight-fSensorHeight)/2.,  Constants::fChipThickness/2.);
//  
//  // Get Mediums
//  TGeoMedium *medSensorSi  = gGeoManager->GetMedium("MFT_Si");
//  TGeoMedium *medReadoutSi = gGeoManager->GetMedium("MFT_Readout");
//
//  // Create Volumes
//  // Chip Volume
//  TGeoVolumeAssembly *chipVol = new TGeoVolumeAssembly("Chip");
//  chipVol->SetVisibility(kTRUE);
//
//  // The sensor Volume
//  TGeoVolume *sensorVol = new TGeoVolume("Sensor", sensor, medSensorSi);
//  sensorVol->SetVisibility(kTRUE);
//  sensorVol->SetLineColor(kGreen+1);
//
//  // The Readout Volume
//  TGeoVolume *readoutVol = new TGeoVolume("Readout", readout, medReadoutSi);
//  readoutVol->SetVisibility(kTRUE);
//  readoutVol->SetLineColor(kRed+2);
//
//  // Building up the chip
//  chipVol->AddNode(readoutVol, 1, new TGeoTranslation(0.,-fChipHeight/2.+readout->GetDY(),  0.));
//  chipVol->AddNode(sensorVol, 1, new TGeoTranslation( 0., fChipHeight/2.-sensor->GetDY(),0.));
//
//  
//  
//  return chipVol;
  
}
