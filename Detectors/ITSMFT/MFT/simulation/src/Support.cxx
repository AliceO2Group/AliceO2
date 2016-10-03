/// \file Support.cxx
/// \brief Class describing geometry of one MFT half-disk support + PCBs
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "TGeoManager.h"
#include "TGeoTube.h"

#include "MFTBase/Constants.h"
#include "MFTSimulation/Support.h"

using namespace AliceO2::MFT;

/// \cond CLASSIMP
ClassImp(AliceO2::MFT::Support)
/// \endcond

//_____________________________________________________________________________
Support::Support():
TNamed(),
fSupportVolume(NULL),
fSupportThickness(1.4), // cm
fPCBThickness(0.1) // cm
{
  
  // default constructor
  
}

//_____________________________________________________________________________
Support::~Support() 
{
  
  delete fSupportVolume;

}

//_____________________________________________________________________________
TGeoVolumeAssembly* Support::CreateVolume(Int_t half, Int_t disk)
{

  Info("CreateVolume",Form("Creating support and PCB for half %d and disk %d (to be done!)",half, disk),0,0);
  /*
  fSupportVolume = new TGeoVolumeAssembly(Form("SupportPCB_%d_%d", half, disk));
  
  TGeoVolume * supportVolume =  CreateSupport(half, disk);
  TGeoVolumeAssembly * pcbVolume =  CreatePCBs(half, disk);
  
  // Place the core of the support
  fSupportVolume->AddNode(supportVolume, 1);
  
  
  // Place the front PCB
  fSupportVolume->AddNode(pcbVolume, 1,new TGeoTranslation(0.,0.,(fSupportThickness+ fPCBThickness)/2.));
  // Place the back PCB (supposing both fron and back are the same shapes)
  fSupportVolume->AddNode(pcbVolume, 2,new TGeoCombiTrans (0.,0.,-(fSupportThickness+ fPCBThickness)/2., new TGeoRotation("rot",0.,180.,0.)));
  */
  return fSupportVolume;

}

//_____________________________________________________________________________
TGeoVolumeAssembly* Support::CreatePCBs(Int_t half, Int_t disk)
{
  
  Info("CreatePCBs",Form("Creating PCB for half %d and disk %d ",half, disk),0,0);
  
  TGeoVolumeAssembly * pcbVolume = new TGeoVolumeAssembly(Form("PCB_%d_%d", half, disk));
  
  // Create Shapes
  Double_t phiMin =0., phiMax=180.;
  Double_t rMin =20., rMax=40.; // units are cm
  Double_t copperThickness = 0.05; //units are cm
  Double_t fr4Thickness = fPCBThickness - copperThickness;

  TGeoTubeSeg *varnishShape = new TGeoTubeSeg(rMin, rMax, fr4Thickness/2., phiMin, phiMax);
  TGeoTubeSeg *copperShape = new TGeoTubeSeg(rMin, rMax, copperThickness/2., phiMin, phiMax);
  
  // Get Mediums
  TGeoMedium *medFR4  = gGeoManager->GetMedium("MFT_FR4$");
  TGeoMedium *medCu  = gGeoManager->GetMedium("MFT_Cu$");
  
  // Create Volumes
  TGeoVolume *varnishVol = new TGeoVolume(Form("Varnish_%d_%d", half, disk), varnishShape, medFR4);
  varnishVol->SetVisibility(kTRUE);
  varnishVol->SetLineColor(kGreen);
  varnishVol->SetLineWidth(1);
  varnishVol->SetFillColor(varnishVol->GetLineColor());
  varnishVol->SetFillStyle(4000); // 0% transparent
  
  TGeoVolume *copperVol = new TGeoVolume(Form("Copper_%d_%d", half, disk), copperShape, medCu);
  copperVol->SetVisibility(kTRUE);
  copperVol->SetLineColor(kOrange);
  copperVol->SetLineWidth(1);
  copperVol->SetFillColor(copperVol->GetLineColor());
  copperVol->SetFillStyle(4000); // 0% transparent
  
  // Position Volumes in the mother PCB Volume
  pcbVolume->AddNode(varnishVol, 1,new TGeoTranslation(0.,0.,fr4Thickness/2.));
  pcbVolume->AddNode(copperVol, 1,new TGeoTranslation(0.,0.,-copperThickness/2.));

  return pcbVolume;

}

//_____________________________________________________________________________
TGeoVolume* Support::CreateSupport(Int_t half, Int_t disk)
{
  
  Info("CreateSupport",Form("Creating PCB for half %d and disk %d ",half, disk),0,0);
  
  // Create Shapes
  Double_t phiMin =0., phiMax=180.;
  Double_t rMin =20., rMax=40.; // units are cm
  
  TGeoTubeSeg *supportShape = new TGeoTubeSeg(rMin, rMax, fSupportThickness/2., phiMin, phiMax);
  
  // Get Mediums
  TGeoMedium *medPeek  = gGeoManager->GetMedium("MFT_PEEK$");
  
  // Create Volumes
  TGeoVolume *supportVol = new TGeoVolume(Form("Support_%d_%d", half, disk), supportShape, medPeek);
  supportVol->SetVisibility(kTRUE);
  supportVol->SetLineColor(kYellow-6);
  supportVol->SetLineWidth(1);
  supportVol->SetFillColor(supportVol->GetLineColor());
  supportVol->SetFillStyle(4000); // 0% transparent
  
  return supportVol;

}
