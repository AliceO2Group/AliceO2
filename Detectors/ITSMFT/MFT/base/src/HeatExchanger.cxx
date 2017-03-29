/// \file HeatExchanger.cxx
/// \brief Class building the MFT heat exchanger
/// \author P. Demongodin, Raphael Tieulent <raphael.tieulent@cern.ch>

#include "TMath.h"
#include "TGeoManager.h"
#include "TGeoCompositeShape.h"
#include "TGeoTube.h"
#include "TGeoBoolNode.h"
#include "TGeoBBox.h"
#include "TGeoVolume.h"

#include "FairLogger.h"

#include "MFTBase/Constants.h"
#include "MFTBase/HeatExchanger.h"
#include "MFTBase/Geometry.h"

using namespace AliceO2::MFT;

/// \cond CLASSIMP
ClassImp(AliceO2::MFT::HeatExchanger)
/// \endcond

//_____________________________________________________________________________
HeatExchanger::HeatExchanger() : 
TNamed(),
mHalfDisk(NULL),
mHalfDiskRotation(NULL),
mHalfDiskTransformation(NULL),
mRWater(0.),
mDRPipe(0.),
mHeatExchangerThickness(0.),
mCarbonThickness(0.),
mHalfDiskGap(0.),
mRohacellThickness(0.),
fnPart(),
mRMin(),
mZPlan(),
mSupportXDimensions(),
mSupportYDimensions(),
mLWater(0.),
mXPosition0(),
fangle0(0.),
fradius0(0.),
mLpartial0(0.),
mLWater3(),
mXPosition3(),
fangle3(),
fradius3(),
mLpartial3(),
fradius3fourth(),
fangle3fourth(),
fbeta3fourth(),
mLwater4(), 
mXposition4(), 
fangle4(), 
fradius4(), 
mLpartial4(), 
fangle4fifth(), 
fradius4fifth()
{

  mRWater = 0.1/2.;
  mDRPipe = 0.005;
  //fHeatExchangerThickness = 1.398; // to get a 13.4 mm thickness for the rohacell... but the water pipes are not inside it, then its density must be increased
  mHeatExchangerThickness = 1.4 + 2*Geometry::kRohacell; // kRohacell is used to link the rohacell thickness and the ladder positionning
  mCarbonThickness = (0.0290)/2.;  // half thickness of the carbon plate

  InitParameters();

}

//_____________________________________________________________________________
HeatExchanger::HeatExchanger(Double_t rWater,
			     Double_t dRPipe,
			     Double_t heatExchangerThickness,
			     Double_t carbonThickness) : 
TNamed(), 
mHalfDisk(NULL),
mHalfDiskRotation(NULL),
mHalfDiskTransformation(NULL),
mRWater(rWater),
mDRPipe(dRPipe),
mHeatExchangerThickness(heatExchangerThickness),
mCarbonThickness(carbonThickness),
mHalfDiskGap(0.),
mRohacellThickness(0.),
fnPart(),
mRMin(),
mZPlan(),
mSupportXDimensions(),
mSupportYDimensions(),
mLWater(0.),
mXPosition0(),
fangle0(0.),
fradius0(0.),
mLpartial0(0.),
mLWater3(),
mXPosition3(),
fangle3(),
fradius3(),
mLpartial3(),
fradius3fourth(),
fangle3fourth(),
fbeta3fourth(),
mLwater4(), 
mXposition4(), 
fangle4(), 
fradius4(), 
mLpartial4(), 
fangle4fifth(), 
fradius4fifth()
{

  InitParameters();
  
}

//_____________________________________________________________________________
TGeoVolumeAssembly* HeatExchanger::Create(Int_t half, Int_t disk) 
{
	
  Info("Create",Form("Creating HeatExchanger_%d_%d", disk, half),0,0);
  
  mHalfDisk = new TGeoVolumeAssembly(Form("HeatExchanger_%d_%d", disk, half));
    switch (disk) {
      case 0: CreateHalfDisk0(half);
        break;
      case 1: CreateHalfDisk1(half);
        break;
      case 2: CreateHalfDisk2(half);
        break;
      case 3: CreateHalfDisk3(half);
        break;
      case 4: CreateHalfDisk4(half);
        break;
    }
  
  Info("Create",Form("... done HeatExchanger_%d_%d", disk, half),0,0);
  
  return mHalfDisk;
  
}

//_____________________________________________________________________________
void HeatExchanger::CreateManyfold(Int_t disk)
{

  TGeoCombiTrans  *transformation1 = 0;
  TGeoCombiTrans  *transformation2 = 0;
  TGeoRotation    *rotation       = 0;
  TGeoCombiTrans  *transformation = 0;

  // **************************************** Manyfolds right and left, fm  ****************************************
  
  Double_t deltay = 0.2;  // shift respect to the median plan of the MFT
  Double_t mfX = 2.2;     // width
  Double_t mfY = 6.8;     // height
  Double_t mfZ = 1.7;     // thickness
  Double_t fShift=0;
  if(disk==3 || disk==4)fShift=0.015; // to avoid overlap with the 2 curved water pipes on the 2 upstream chambers

  TGeoMedium *kMedPeek    = gGeoManager->GetMedium("MFT_PEEK$");
  
  TGeoBBox *boxmanyfold = new TGeoBBox("boxmanyfold", mfX/2, mfY/2, mfZ/2);
  TGeoBBox *remove = new TGeoBBox("remove", 0.45/2 + Geometry::kEpsilon, mfY/2 + Geometry::kEpsilon, 0.6/2 + Geometry::kEpsilon);
  TGeoTranslation *tL= new TGeoTranslation ("tL", mfX/2-0.45/2, 0., -mfZ/2+0.6/2);
  TGeoSubtraction *boxManyFold = new TGeoSubtraction(boxmanyfold, remove, NULL, tL);
  TGeoCompositeShape *BoxManyFold = new TGeoCompositeShape("BoxManyFold", boxManyFold);

  TGeoTranslation *tR= new TGeoTranslation ("tR", -mfX/2+0.45/2, 0., -mfZ/2+0.6/2);
  TGeoSubtraction *boxManyFold1 = new TGeoSubtraction(BoxManyFold, remove, NULL, tR);
  TGeoCompositeShape *BoxManyFold1 = new TGeoCompositeShape("BoxManyFold1", boxManyFold1);

  TGeoVolume *MF01 = new TGeoVolume(Form("MF%d1",disk), BoxManyFold1, kMedPeek);

  rotation = new TGeoRotation ("rotation", 90., 90., 90.);
  transformation1 = new TGeoCombiTrans(mSupportXDimensions[disk][0]/2+mfZ/2+fShift, mfY/2+deltay, mZPlan[disk], rotation);

  mHalfDisk->AddNode(MF01, 1, transformation1);
    
  TGeoVolume *MF02 = new TGeoVolume(Form("MF%d2",disk), BoxManyFold1, kMedPeek);
  transformation2 = new TGeoCombiTrans(mSupportXDimensions[disk][0]/2+mfZ/2+fShift, -mfY/2-deltay, mZPlan[disk], rotation);

  mHalfDisk->AddNode(MF02, 1, transformation2);
    
  // ********************************************************************************************************
}

//_____________________________________________________________________________
void HeatExchanger::CreateHalfDisk0(Int_t half) {
  
  Int_t disk = 0;
  
  if      (half == kTop)    printf("Creating MFT heat exchanger for disk0 top\n");
  else if (half == kBottom) printf("Creating MFT heat exchanger for disk0 bottom\n");
  else     printf("No valid option for MFT heat exchanger on disk0\n");
  
  //carbon   = gGeoManager->GetMedium("MFT_Carbon$");
  carbon   = gGeoManager->GetMedium("MFT_CarbonFiber$");

  water    = gGeoManager->GetMedium("MFT_Water$");
  rohacell = gGeoManager->GetMedium("MFT_Rohacell");
  pipe     = gGeoManager->GetMedium("MFT_Polyimide");
  
  TGeoVolumeAssembly *cooling = new TGeoVolumeAssembly(Form("cooling_D0_H%d",half));
  
  Float_t lMiddle = mSupportXDimensions[disk][0] - 2.*mLWater;  // length of central part
  
  TGeoTranslation *translation    = 0;
  TGeoRotation    *rotation       = 0;
  TGeoCombiTrans  *transformation = 0;

  TGeoCombiTrans  *transformation1 = 0;
  TGeoCombiTrans  *transformation2 = 0;


  // **************************************** Water part ****************************************
  
  // ------------------- Tube shape -------------------
  
  TGeoVolume *waterTube1 = gGeoManager->MakeTube(Form("waterTube1_D0_H%d",half), water, 0., mRWater, mLWater/2.);
  waterTube1->SetLineColor(kBlue);
  
  for (Int_t itube=0; itube<3; itube++) {
    translation = new TGeoTranslation(mXPosition0[itube], 0.,  mLWater/2. + lMiddle/2.);
    cooling->AddNode (waterTube1, itube, translation);
    translation = new TGeoTranslation(mXPosition0[itube], 0., -mLWater/2. - lMiddle/2.);
    cooling->AddNode (waterTube1, itube+3, translation);
  }
  
  Double_t angle0rad = fangle0*(TMath::DegToRad());
  
  TGeoVolume *waterTube2 = gGeoManager->MakeTube(Form("waterTube2_D0_H%d",half), water, 0., mRWater, mLpartial0/2.);
  waterTube2->SetLineColor(kBlue);
  
  for (Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", -90., -fangle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + fradius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., - lMiddle/2. +
                                        fradius0*(TMath::Sin(angle0rad)) + (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (waterTube2, itube, transformation);
    rotation = new TGeoRotation ("rotation", -90., fangle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + fradius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., lMiddle/2. -
                                        fradius0*(TMath::Sin(angle0rad)) - (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (waterTube2, itube+3, transformation);
  }
  
  // ------------------- Torus shape -------------------
  
  // Sides torus
  
  TGeoVolume *waterTorus1 = gGeoManager->MakeTorus(Form("waterTorus1_D0_H%d",half), water, fradius0, 0., mRWater, 0., fangle0);
  waterTorus1->SetLineColor(kBlue);
  Double_t radius0mid = (lMiddle - 2.*(fradius0*(TMath::Sin(angle0rad)) + mLpartial0*(TMath::Cos(angle0rad))))/(2*(TMath::Sin(angle0rad)));
  
  for (Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(fradius0 + mXPosition0[itube], 0., - lMiddle/2., rotation);
    cooling->AddNode (waterTorus1, itube, transformation);
    rotation = new TGeoRotation ("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(fradius0 + mXPosition0[itube], 0.,  lMiddle/2., rotation);
    cooling->AddNode (waterTorus1, itube+3, transformation);
  }
  
  // Central Torus
  
  TGeoVolume *waterTorus2 = gGeoManager->MakeTorus(Form("waterTorus2_D0_H%d",half), water, radius0mid, 0., mRWater, - fangle0 , 2.*fangle0);
  waterTorus2->SetLineColor(kBlue);
  
  for(Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube]  + fradius0*(1-(TMath::Cos(angle0rad))) + mLpartial0*TMath::Sin(angle0rad) - radius0mid*TMath::Cos(angle0rad), 0., 0., rotation);
    cooling->AddNode (waterTorus2, itube, transformation);
  }
  
  // **************************************** Pipe part ****************************************
  
  // ------------------- Tube shape -------------------
  
  TGeoVolume *pipeTube1 = gGeoManager->MakeTube(Form("pipeTube1_D0_H%d",half), pipe, mRWater, mRWater + mDRPipe, mLWater/2.);
  pipeTube1->SetLineColor(10);
  
  for (Int_t itube=0; itube<3; itube++){
    translation = new TGeoTranslation(mXPosition0[itube], 0.,  mLWater/2. + lMiddle/2.);
    cooling->AddNode (pipeTube1, itube, translation);
    translation = new TGeoTranslation(mXPosition0[itube], 0., -mLWater/2. - lMiddle/2.);
    cooling->AddNode (pipeTube1, itube+3, translation);
  }
  
  TGeoVolume *pipeTube2 = gGeoManager->MakeTube(Form("pipeTube2_D0_H%d",half), pipe, mRWater, mRWater + mDRPipe, mLpartial0/2.);
  waterTube2->SetLineColor(10);
  
  for(Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", -90., -fangle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + fradius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., - lMiddle/2. +
                                        fradius0*(TMath::Sin(angle0rad)) + (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (pipeTube2, itube, transformation);
    
    rotation = new TGeoRotation ("rotation", -90., fangle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + fradius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., lMiddle/2. -
                                        fradius0*(TMath::Sin(angle0rad)) - (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (pipeTube2, itube+3, transformation);
  }
  
  // ------------------- Torus shape -------------------
  
  // Sides Torus
  TGeoVolume *pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorus1_D0_H%d",half), pipe, fradius0, mRWater, mRWater + mDRPipe, 0., fangle0);
  pipeTorus1->SetLineColor(10);
  
  for (Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(fradius0 + mXPosition0[itube], 0., - lMiddle/2., rotation);
    cooling->AddNode (pipeTorus1, itube, transformation);
    rotation = new TGeoRotation ("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(fradius0 + mXPosition0[itube], 0.,   lMiddle/2., rotation);
    cooling->AddNode (pipeTorus1, itube+3, transformation);
  }
  
  // Central Torus
  TGeoVolume *pipeTorus2 = gGeoManager->MakeTorus(Form("pipeTorus2_D0_H%d",half), pipe, radius0mid, mRWater, mRWater + mDRPipe, - fangle0 , 2.*fangle0);
  pipeTorus2->SetLineColor(10);
  
  for(Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube]  + fradius0*(1-(TMath::Cos(angle0rad))) + mLpartial0*TMath::Sin(angle0rad) - radius0mid*TMath::Cos(angle0rad), 0., 0., rotation);
    cooling->AddNode (pipeTorus2, itube, transformation);
  }
  
  Double_t deltaz = mHeatExchangerThickness - mCarbonThickness*2;  // distance between pair of carbon plates
  
//  if (half == kTop) {
//    rotation = new TGeoRotation ("rotation", 90., 90., 0.);
//    transformation = new TGeoCombiTrans(0., 0., fZPlan[disk] + deltaz/2. - fCarbonThickness - fRWater - fDRPipe, rotation);
//    fHalfDisk->AddNode(cooling, 1, transformation);
//    transformation = new TGeoCombiTrans(0., 0., fZPlan[disk] - deltaz/2. + fCarbonThickness + fRWater + fDRPipe, rotation);
//    fHalfDisk->AddNode(cooling, 2, transformation);
//  }
//  else if (half == kBottom) {
    rotation = new TGeoRotation ("rotation", -90., 90., 0.);
    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz/2. - mCarbonThickness - mRWater - mDRPipe, rotation);
    mHalfDisk->AddNode(cooling, 3, transformation);
    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz/2. + mCarbonThickness + mRWater + mDRPipe, rotation);
    mHalfDisk->AddNode(cooling, 4, transformation);
//  }
  



  // **************************************** Carbon Plates ****************************************
  
  TGeoVolumeAssembly *carbonPlate = new TGeoVolumeAssembly(Form("carbonPlate_D0_H%d",half));
  
  TGeoBBox *carbonBase0 = new TGeoBBox (Form("carbonBase0_D0_H%d",half),  (mSupportXDimensions[disk][0])/2., (mSupportYDimensions[disk][0])/2., mCarbonThickness);
  TGeoTranslation *t01= new TGeoTranslation ("t01",0., (mSupportYDimensions[disk][0])/2. + mHalfDiskGap , 0.);
  t01-> RegisterYourself();
  
  TGeoTubeSeg *holeCarbon0 = new TGeoTubeSeg(Form("holeCarbon0_D0_H%d",half), 0., mRMin[disk], mCarbonThickness + 0.000001, 0, 180.);
  TGeoTranslation *t02= new TGeoTranslation ("t02",0., - mHalfDiskGap , 0.);
  t02-> RegisterYourself();
  
  ///TGeoCompositeShape *cs0 = new TGeoCompositeShape(Form("cs0_D0_H%d",half), Form("(carbonBase0_D0_H%d:t01)-(holeCarbon0_D0_H%d:t02)",half,half));
  TGeoSubtraction    *carbonhole0 = new TGeoSubtraction(carbonBase0, holeCarbon0, t01, t02);
  TGeoCompositeShape *ch0 = new TGeoCompositeShape(Form("Carbon0_D0_H%d",half), carbonhole0);
  TGeoVolume *carbonBaseWithHole0 = new TGeoVolume(Form("carbonBaseWithHole_D0_H%d", half), ch0, carbon);
  

  carbonBaseWithHole0->SetLineColor(kGray+3);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  carbonPlate->AddNode(carbonBaseWithHole0, 0, new TGeoTranslation(0., 0., mZPlan[disk]));
  
  Double_t ty = mSupportYDimensions[disk][0];
  
  for (Int_t ipart=1; ipart<fnPart[disk]; ipart ++) {
    ty += mSupportYDimensions[disk][ipart]/2.;
    TGeoVolume *partCarbon = gGeoManager->MakeBox(Form("partCarbon_D0_H%d_%d", half,ipart), carbon, mSupportXDimensions[disk][ipart]/2., mSupportYDimensions[disk][ipart]/2., mCarbonThickness);
    partCarbon->SetLineColor(kGray+3);
    TGeoTranslation *t = new TGeoTranslation ("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    carbonPlate -> AddNode(partCarbon, ipart, t);
    ty += mSupportYDimensions[disk][ipart]/2.;
  }
  
//  if (half == kTop) {
//    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
//    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
//    fHalfDisk->AddNode(carbonPlate, 1, transformation);
//    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
//    fHalfDisk->AddNode(carbonPlate, 2, transformation);
//  }
//  else if (half == kBottom) {
    rotation = new TGeoRotation ("rotation", 180., 0., 0.);
    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
    mHalfDisk->AddNode(carbonPlate, 3, transformation);
    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
    mHalfDisk->AddNode(carbonPlate, 4, transformation);
//  }



  // **************************************** Rohacell Plate ****************************************
  
  TGeoVolumeAssembly *rohacellPlate = new TGeoVolumeAssembly(Form("rohacellPlate_D0_H%d",half));
  
  TGeoBBox *rohacellBase0 = new TGeoBBox (Form("rohacellBase0_D0_H%d",half),  (mSupportXDimensions[disk][0])/2., (mSupportYDimensions[disk][0])/2., mRohacellThickness);
  // TGeoTranslation *t3 = new TGeoTranslation ("t3",0., (fSupportYDimensions[disk][0])/2. + fHalfDiskGap , 0.);
  // t3 -> RegisterYourself();
  
  TGeoTubeSeg *holeRohacell0 = new TGeoTubeSeg(Form("holeRohacell0_D0_H%d",half), 0., mRMin[disk], mRohacellThickness + 0.000001, 0, 180.);
  // TGeoTranslation *t4= new TGeoTranslation ("t4", 0., - fHalfDiskGap , 0.);
  // t4-> RegisterYourself();
  
  ///cs0 = new TGeoCompositeShape("cs0", Form("(rohacellBase0_D0_H%d:t01)-(holeRohacell0_D0_H%d:t02)",half,half));
  TGeoSubtraction    *rohacellhole0 = new TGeoSubtraction(rohacellBase0, holeRohacell0, t01, t02);
  TGeoCompositeShape *rh0 = new TGeoCompositeShape(Form("rohacellBase0_D0_H%d",half), rohacellhole0);
  TGeoVolume *rohacellBaseWithHole = new TGeoVolume(Form("rohacellBaseWithHole_D0_H%d",half), rh0, rohacell);


  rohacellBaseWithHole->SetLineColor(kGray);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation =  new TGeoCombiTrans(0., 0., 0., rotation);
  rohacellPlate -> AddNode(rohacellBaseWithHole, 0, new TGeoTranslation(0., 0., mZPlan[disk]));
  
  ty = mSupportYDimensions[disk][0];
  
  for (Int_t ipart=1; ipart<fnPart[disk]; ipart ++) {
    ty += mSupportYDimensions[disk][ipart]/2.;
    TGeoVolume *partRohacell = gGeoManager->MakeBox(Form("partRohacelli_D0_H%d_%d", half,ipart), rohacell, mSupportXDimensions[disk][ipart]/2., mSupportYDimensions[disk][ipart]/2., mRohacellThickness);
    partRohacell->SetLineColor(kGray);
    TGeoTranslation *t = new TGeoTranslation ("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    rohacellPlate -> AddNode(partRohacell, ipart, t);
    ty += mSupportYDimensions[disk][ipart]/2.;
  }
  
//  if (half == kTop) {
//    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
//    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
//    fHalfDisk->AddNode(rohacellPlate, 1, transformation);
//  }
//  if (half == kBottom) {
    rotation = new TGeoRotation ("rotation", 180., 0., 0.);
    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
    mHalfDisk->AddNode(rohacellPlate, 1, transformation);
//  }
 
    CreateManyfold(disk);
 
}

//_____________________________________________________________________________
void HeatExchanger::CreateHalfDisk1(Int_t half) {
	
  Int_t disk = 1;
  
  if      (half == kTop)    printf("Creating MFT heat exchanger for disk1 top\n");
  else if (half == kBottom) printf("Creating MFT heat exchanger for disk1 bottom\n");
  else     printf("No valid option for MFT heat exchanger on disk1\n");
  
  //carbon   = gGeoManager->GetMedium("MFT_Carbon$");
  carbon   = gGeoManager->GetMedium("MFT_CarbonFiber$");

  water    = gGeoManager->GetMedium("MFT_Water$");
  rohacell = gGeoManager->GetMedium("MFT_Rohacell");
  pipe     = gGeoManager->GetMedium("MFT_Polyimide");
    
  TGeoVolumeAssembly *cooling = new TGeoVolumeAssembly(Form("cooling_D1_H%d",half));
  
  Float_t lMiddle = mSupportXDimensions[disk][0] - 2.*mLWater;  // length of central part
  
  TGeoTranslation *translation    = 0;
  TGeoRotation    *rotation       = 0;
  TGeoCombiTrans  *transformation = 0;
  
  
  // **************************************** Water part ****************************************
  
  // ------------------- Tube shape -------------------
  
  TGeoVolume *waterTube1 = gGeoManager->MakeTube(Form("waterTube1_D1_H%d",half), water, 0., mRWater, mLWater/2.);
  waterTube1->SetLineColor(kBlue);
  
  for (Int_t itube=0; itube<3; itube++) {
    translation = new TGeoTranslation(mXPosition0[itube], 0.,  mLWater/2. + lMiddle/2.);
    cooling->AddNode (waterTube1, itube, translation);
    translation = new TGeoTranslation(mXPosition0[itube], 0., -mLWater/2. - lMiddle/2.);
    cooling->AddNode (waterTube1, itube+3, translation);
  }
  
  Double_t angle0rad = fangle0*(TMath::DegToRad());
  
  TGeoVolume *waterTube2 = gGeoManager->MakeTube(Form("waterTube2_D1_H%d",half), water, 0., mRWater, mLpartial0/2.);
  waterTube2->SetLineColor(kBlue);
  
  for (Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", -90., -fangle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + fradius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., - lMiddle/2. +
                                        fradius0*(TMath::Sin(angle0rad)) + (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (waterTube2, itube, transformation);
    rotation = new TGeoRotation ("rotation", -90., fangle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + fradius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., lMiddle/2. -
                                        fradius0*(TMath::Sin(angle0rad)) - (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (waterTube2, itube+3, transformation);
  }
  
  // ------------------- Torus shape -------------------
  
  // Sides torus
  
  TGeoVolume *waterTorus1 = gGeoManager->MakeTorus(Form("waterTorus1_D1_H%d",half), water, fradius0, 0., mRWater, 0., fangle0);
  waterTorus1->SetLineColor(kBlue);
  Double_t radius0mid = (lMiddle - 2.*(fradius0*(TMath::Sin(angle0rad)) + mLpartial0*(TMath::Cos(angle0rad))))/(2*(TMath::Sin(angle0rad)));
  
  for (Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(fradius0 + mXPosition0[itube], 0., - lMiddle/2., rotation);
    cooling->AddNode (waterTorus1, itube, transformation);
    rotation = new TGeoRotation ("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(fradius0 + mXPosition0[itube], 0.,  lMiddle/2., rotation);
    cooling->AddNode (waterTorus1, itube+3, transformation);
  }
  
  // Central Torus
  
  TGeoVolume *waterTorus2 = gGeoManager->MakeTorus(Form("waterTorus2_D1_H%d",half), water, radius0mid, 0., mRWater, - fangle0 , 2.*fangle0);
  waterTorus2->SetLineColor(kBlue);
  
  for(Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube]  + fradius0*(1-(TMath::Cos(angle0rad))) + mLpartial0*TMath::Sin(angle0rad) - radius0mid*TMath::Cos(angle0rad), 0., 0., rotation);
    cooling->AddNode (waterTorus2, itube, transformation);
  }
  
  // **************************************** Pipe part ****************************************
  
  // ------------------- Tube shape -------------------
  
  TGeoVolume *pipeTube1 = gGeoManager->MakeTube(Form("pipeTube1_D1_H%d",half), pipe, mRWater, mRWater + mDRPipe, mLWater/2.);
  pipeTube1->SetLineColor(10);
  
  for (Int_t itube=0; itube<3; itube++){
    translation = new TGeoTranslation(mXPosition0[itube], 0.,  mLWater/2. + lMiddle/2.);
    cooling->AddNode (pipeTube1, itube, translation);
    translation = new TGeoTranslation(mXPosition0[itube], 0., -mLWater/2. - lMiddle/2.);
    cooling->AddNode (pipeTube1, itube+3, translation);
  }

  TGeoVolume *pipeTube2 = gGeoManager->MakeTube(Form("pipeTube2_D1_H%d",half), pipe, mRWater, mRWater + mDRPipe, mLpartial0/2.);
  waterTube2->SetLineColor(10);
  
  for(Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", -90., -fangle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + fradius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., - lMiddle/2. +
                                        fradius0*(TMath::Sin(angle0rad)) + (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (pipeTube2, itube, transformation);
    
    rotation = new TGeoRotation ("rotation", -90., fangle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + fradius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., lMiddle/2. -
                                        fradius0*(TMath::Sin(angle0rad)) - (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (pipeTube2, itube+3, transformation);
  }
  
  // ------------------- Torus shape -------------------
  
  // Sides Torus
  TGeoVolume *pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorus1_D1_H%d",half), pipe, fradius0, mRWater, mRWater + mDRPipe, 0., fangle0);
  pipeTorus1->SetLineColor(10);
  
  for (Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(fradius0 + mXPosition0[itube], 0., - lMiddle/2., rotation);
    cooling->AddNode (pipeTorus1, itube, transformation);
    rotation = new TGeoRotation ("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(fradius0 + mXPosition0[itube], 0.,   lMiddle/2., rotation);
    cooling->AddNode (pipeTorus1, itube+3, transformation);
  }
  
  // Central Torus
  TGeoVolume *pipeTorus2 = gGeoManager->MakeTorus(Form("pipeTorus2_D1_H%d",half), pipe, radius0mid, mRWater, mRWater + mDRPipe, - fangle0 , 2.*fangle0);
  pipeTorus2->SetLineColor(10);
  
  for(Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube]  + fradius0*(1-(TMath::Cos(angle0rad))) + mLpartial0*TMath::Sin(angle0rad) - radius0mid*TMath::Cos(angle0rad), 0., 0., rotation);
    cooling->AddNode (pipeTorus2, itube, transformation);
  }
  
  Double_t deltaz = mHeatExchangerThickness - mCarbonThickness*2;  // distance between pair of carbon plates
  
//  if (half == kTop) {
//    rotation = new TGeoRotation ("rotation", 90., 90., 0.);
//    transformation = new TGeoCombiTrans(0., 0., fZPlan[disk] + deltaz/2. - fCarbonThickness - fRWater - fDRPipe, rotation);
//    fHalfDisk->AddNode(cooling, 1, transformation);
//    transformation = new TGeoCombiTrans(0., 0., fZPlan[disk] - deltaz/2. + fCarbonThickness + fRWater + fDRPipe, rotation);
//    fHalfDisk->AddNode(cooling, 2, transformation);
//  }
//  else if (half == kBottom) {
    rotation = new TGeoRotation ("rotation", -90., 90., 0.);
    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz/2. - mCarbonThickness - mRWater - mDRPipe, rotation);
    mHalfDisk->AddNode(cooling, 0, transformation);
    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz/2. + mCarbonThickness + mRWater + mDRPipe, rotation);
    mHalfDisk->AddNode(cooling, 1, transformation);
//  }
  
  
  // **************************************** Carbon Plates ****************************************
  
   

  TGeoVolumeAssembly *carbonPlate = new TGeoVolumeAssembly(Form("carbonPlate_D1_H%d",half));
  
  TGeoBBox *carbonBase1 = new TGeoBBox (Form("carbonBase1_D1_H%d",half),  (mSupportXDimensions[disk][0])/2., (mSupportYDimensions[disk][0])/2., mCarbonThickness);
  TGeoTranslation *t11= new TGeoTranslation ("t11",0., (mSupportYDimensions[disk][0])/2. + mHalfDiskGap , 0.);
  t11-> RegisterYourself();
  
  TGeoTubeSeg *holeCarbon1 = new TGeoTubeSeg(Form("holeCarbon1_D1_H%d",half), 0., mRMin[disk], mCarbonThickness + 0.000001, 0, 180.);
  TGeoTranslation *t12= new TGeoTranslation ("t12",0., - mHalfDiskGap , 0.);
  t12-> RegisterYourself();
  
  
  ////TGeoCompositeShape *cs1 = new TGeoCompositeShape(Form("Carbon1_D1_H%d",half), Form("(carbonBase1_D1_H%d:t11)-(holeCarbon1_D1_H%d:t12)",half,half));
  TGeoSubtraction    *carbonhole1 = new TGeoSubtraction(carbonBase1, holeCarbon1, t11, t12);
  TGeoCompositeShape *ch1 = new TGeoCompositeShape(Form("Carbon1_D1_H%d",half), carbonhole1);
  TGeoVolume *carbonBaseWithHole1 = new TGeoVolume(Form("carbonBaseWithHole_D1_H%d",half), ch1, carbon);


  carbonBaseWithHole1->SetLineColor(kGray+3);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  carbonPlate->AddNode(carbonBaseWithHole1, 0, new TGeoTranslation(0., 0., mZPlan[disk]));
  
  Double_t ty = mSupportYDimensions[disk][0];
  
  for (Int_t ipart=1; ipart<fnPart[disk]; ipart ++) {
    ty += mSupportYDimensions[disk][ipart]/2.;
    TGeoVolume *partCarbon = gGeoManager->MakeBox(Form("partCarbon_D1_H%d_%d", half,ipart), carbon, mSupportXDimensions[disk][ipart]/2.,
                                                  mSupportYDimensions[disk][ipart]/2., mCarbonThickness);
    partCarbon->SetLineColor(kGray+3);
    TGeoTranslation *t = new TGeoTranslation ("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    carbonPlate -> AddNode(partCarbon, ipart, t);
    ty += mSupportYDimensions[disk][ipart]/2.;
  }
  
//  if (half == kTop) {
//    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
//    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
//    fHalfDisk->AddNode(carbonPlate, 1, transformation);
//    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
//    fHalfDisk->AddNode(carbonPlate, 2, transformation);
//  }
//  else if (half == kBottom) {
    rotation = new TGeoRotation ("rotation", 180., 0., 0.);
    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
    mHalfDisk->AddNode(carbonPlate, 0, transformation);
    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
    mHalfDisk->AddNode(carbonPlate, 1, transformation);
//  }

  
  // **************************************** Rohacell Plate ****************************************
  
  TGeoVolumeAssembly *rohacellPlate = new TGeoVolumeAssembly(Form("rohacellPlate_D1_H%d",half));
  
  TGeoBBox *rohacellBase1 = new TGeoBBox ("rohacellBase1",  (mSupportXDimensions[disk][0])/2., (mSupportYDimensions[disk][0])/2., mRohacellThickness);
  // TGeoTranslation *t3 = new TGeoTranslation ("t3",0., (fSupportYDimensions[disk][0])/2. + fHalfDiskGap , 0.);
  // t3 -> RegisterYourself();
  
  TGeoTubeSeg *holeRohacell1 = new TGeoTubeSeg("holeRohacell1", 0., mRMin[disk], mRohacellThickness + 0.000001, 0, 180.);
  // TGeoTranslation *t4= new TGeoTranslation ("t4", 0., - fHalfDiskGap , 0.);
  // t4-> RegisterYourself();
  
  //////cs1 = new TGeoCompositeShape(Form("rohacell_D1_H%d",half), "(rohacellBase1:t11)-(holeRohacell1:t12)");
  TGeoSubtraction    *rohacellhole1 = new TGeoSubtraction(rohacellBase1, holeRohacell1, t11, t12);
  TGeoCompositeShape *rh1 = new TGeoCompositeShape(Form("rohacellBase1_D1_H%d",half), rohacellhole1);
  TGeoVolume *rohacellBaseWithHole = new TGeoVolume(Form("rohacellBaseWithHole_D1_H%d",half), rh1, rohacell);


  rohacellBaseWithHole->SetLineColor(kGray);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation =  new TGeoCombiTrans(0., 0., 0., rotation);
  rohacellPlate -> AddNode(rohacellBaseWithHole, 0, new TGeoTranslation(0., 0., mZPlan[disk]));
  
  ty = mSupportYDimensions[disk][0];
  
  for (Int_t ipart=1; ipart<fnPart[disk]; ipart ++) {
    ty += mSupportYDimensions[disk][ipart]/2.;
    TGeoVolume *partRohacell = gGeoManager->MakeBox(Form("partRohacelli_D1_H%d_%d",half, ipart), rohacell, mSupportXDimensions[disk][ipart]/2.,
                                                    mSupportYDimensions[disk][ipart]/2., mRohacellThickness);
    partRohacell->SetLineColor(kGray);
    TGeoTranslation *t = new TGeoTranslation ("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    rohacellPlate -> AddNode(partRohacell, ipart, t);
    ty += mSupportYDimensions[disk][ipart]/2.;
  }
  
//  if (half == kTop) {
//    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
//    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
//    fHalfDisk->AddNode(rohacellPlate, 1, transformation);
//  }
//  if (half == kBottom) {
    rotation = new TGeoRotation ("rotation", 180., 0., 0.);
    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
    mHalfDisk->AddNode(rohacellPlate, 2, transformation);
//  }
  


}

//_____________________________________________________________________________
void HeatExchanger::CreateHalfDisk2(Int_t half) {
  
  Int_t disk = 2;
  
  if      (half == kTop)    printf("Creating MFT heat exchanger for disk2 top\n");
  else if (half == kBottom) printf("Creating MFT heat exchanger for disk2 bottom\n");
  else     printf("No valid option for MFT heat exchanger on disk2\n");
  
  
  //carbon   = gGeoManager->GetMedium("MFT_Carbon$");
  carbon   = gGeoManager->GetMedium("MFT_CarbonFiber$");

  water    = gGeoManager->GetMedium("MFT_Water$");
  rohacell = gGeoManager->GetMedium("MFT_Rohacell");
  pipe     = gGeoManager->GetMedium("MFT_Polyimide");
 
  TGeoVolumeAssembly *cooling = new TGeoVolumeAssembly(Form("cooling_D2_H%d",half));
  
  Float_t lMiddle = mSupportXDimensions[disk][0] - 2.*mLWater;  // length of central part
  
  TGeoTranslation *translation    = 0;
  TGeoRotation    *rotation       = 0;
  TGeoCombiTrans  *transformation = 0;
  
  // **************************************** Water part ****************************************
  
  // ------------------- Tube shape -------------------
  
  TGeoVolume *waterTube1 = gGeoManager->MakeTube(Form("waterTube1_D2_H%d",half), water, 0., mRWater, mLWater/2.);
  waterTube1->SetLineColor(kBlue);
  
  for (Int_t itube=0; itube<3; itube++) {
    translation = new TGeoTranslation(mXPosition0[itube], 0.,  mLWater/2. + lMiddle/2.);
    cooling->AddNode (waterTube1, itube, translation);
    translation = new TGeoTranslation(mXPosition0[itube], 0., -mLWater/2. - lMiddle/2.);
    cooling->AddNode (waterTube1, itube+3, translation);
  }
  
  Double_t angle0rad = fangle0*(TMath::DegToRad());
  
  TGeoVolume *waterTube2 = gGeoManager->MakeTube(Form("waterTube2_D2_H%d",half), water, 0., mRWater, mLpartial0/2.);
  waterTube2->SetLineColor(kBlue);
  
  for (Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", -90., -fangle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + fradius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., - lMiddle/2. +
                                        fradius0*(TMath::Sin(angle0rad)) + (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (waterTube2, itube, transformation);
    rotation = new TGeoRotation ("rotation", -90., fangle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + fradius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., lMiddle/2. -
                                        fradius0*(TMath::Sin(angle0rad)) - (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (waterTube2, itube+3, transformation);
  }
  
  // ------------------- Torus shape -------------------
  
  // Sides torus
  
  TGeoVolume *waterTorus1 = gGeoManager->MakeTorus(Form("waterTorus1_D2_H%d",half), water, fradius0, 0., mRWater, 0., fangle0);
  waterTorus1->SetLineColor(kBlue);
  Double_t radius0mid = (lMiddle - 2.*(fradius0*(TMath::Sin(angle0rad)) + mLpartial0*(TMath::Cos(angle0rad))))/(2*(TMath::Sin(angle0rad)));
  
  for (Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(fradius0 + mXPosition0[itube], 0., - lMiddle/2., rotation);
    cooling->AddNode (waterTorus1, itube, transformation);
    rotation = new TGeoRotation ("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(fradius0 + mXPosition0[itube], 0.,  lMiddle/2., rotation);
    cooling->AddNode (waterTorus1, itube+3, transformation);
  }
  
  // Central Torus
  
  TGeoVolume *waterTorus2 = gGeoManager->MakeTorus(Form("waterTorus2_D2_H%d",half), water, radius0mid, 0., mRWater, - fangle0 , 2.*fangle0);
  waterTorus2->SetLineColor(kBlue);
  
  for(Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube]  + fradius0*(1-(TMath::Cos(angle0rad))) +
                                        mLpartial0*TMath::Sin(angle0rad) - radius0mid*TMath::Cos(angle0rad), 0., 0., rotation);
    cooling->AddNode (waterTorus2, itube, transformation);
  }
  
  // **************************************** Pipe part ****************************************
  
  // ------------------- Tube shape -------------------
  
  TGeoVolume *pipeTube1 = gGeoManager->MakeTube(Form("pipeTube1_D2_H%d",half), pipe, mRWater, mRWater + mDRPipe, mLWater/2.);
  pipeTube1->SetLineColor(10);
  
  for (Int_t itube=0; itube<3; itube++){
    translation = new TGeoTranslation(mXPosition0[itube], 0.,  mLWater/2. + lMiddle/2.);
    cooling->AddNode (pipeTube1, itube, translation);
    translation = new TGeoTranslation(mXPosition0[itube], 0., -mLWater/2. - lMiddle/2.);
    cooling->AddNode (pipeTube1, itube+3, translation);
  }
  
  TGeoVolume *pipeTube2 = gGeoManager->MakeTube(Form("pipeTube2_D2_H%d",half), pipe, mRWater, mRWater + mDRPipe, mLpartial0/2.);
  waterTube2->SetLineColor(10);
  
  for(Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", -90., -fangle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + fradius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., - lMiddle/2. +
                                        fradius0*(TMath::Sin(angle0rad)) + (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (pipeTube2, itube, transformation);
    
    rotation = new TGeoRotation ("rotation", -90., fangle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + fradius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., lMiddle/2. -
                                        fradius0*(TMath::Sin(angle0rad)) - (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (pipeTube2, itube+3, transformation);
  }
  
  // ------------------- Torus shape -------------------
  
  // Sides Torus
  TGeoVolume *pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorus1_D2_H%d",half), pipe, fradius0, mRWater, mRWater + mDRPipe, 0., fangle0);
  pipeTorus1->SetLineColor(10);
  
  for (Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(fradius0 + mXPosition0[itube], 0., - lMiddle/2., rotation);
    cooling->AddNode (pipeTorus1, itube, transformation);
    rotation = new TGeoRotation ("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(fradius0 + mXPosition0[itube], 0.,   lMiddle/2., rotation);
    cooling->AddNode (pipeTorus1, itube+3, transformation);
  }
  
  // Central Torus
  TGeoVolume *pipeTorus2 = gGeoManager->MakeTorus(Form("pipeTorus2_D2_H%d",half), pipe, radius0mid, mRWater, mRWater + mDRPipe, - fangle0 , 2.*fangle0);
  pipeTorus2->SetLineColor(10);
  
  for(Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube]  + fradius0*(1-(TMath::Cos(angle0rad))) +
                                        mLpartial0*TMath::Sin(angle0rad) - radius0mid*TMath::Cos(angle0rad), 0., 0., rotation);
    cooling->AddNode (pipeTorus2, itube, transformation);
  }
  
  Double_t deltaz = mHeatExchangerThickness - mCarbonThickness*2;  // distance between pair of carbon plates
  
//  if (half == kTop) {
//    rotation = new TGeoRotation ("rotation", 90., 90., 0.);
//    transformation = new TGeoCombiTrans(0., 0., fZPlan[disk] + deltaz/2. - fCarbonThickness - fRWater - fDRPipe, rotation);
//    fHalfDisk->AddNode(cooling, 1, transformation);
//    transformation = new TGeoCombiTrans(0., 0., fZPlan[disk] - deltaz/2. + fCarbonThickness + fRWater + fDRPipe, rotation);
//    fHalfDisk->AddNode(cooling, 2, transformation);
//  }
//  else if (half == kBottom) {
    rotation = new TGeoRotation ("rotation", -90., 90., 0.);
    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz/2. - mCarbonThickness - mRWater - mDRPipe, rotation);
    mHalfDisk->AddNode(cooling, 3, transformation);
    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz/2. + mCarbonThickness + mRWater + mDRPipe, rotation);
    mHalfDisk->AddNode(cooling, 4, transformation);
//  }
  
  // **************************************** Carbon Plates ****************************************
  
  TGeoVolumeAssembly *carbonPlate = new TGeoVolumeAssembly(Form("carbonPlate_D2_H%d",half));
  
  TGeoBBox *carbonBase2 = new TGeoBBox (Form("carbonBase2_D2_H%d",half),  (mSupportXDimensions[disk][0])/2., (mSupportYDimensions[disk][0])/2., mCarbonThickness);
  TGeoTranslation *t21= new TGeoTranslation ("t21",0., (mSupportYDimensions[disk][0])/2. + mHalfDiskGap , 0.);
  t21-> RegisterYourself();
  
  TGeoTubeSeg *holeCarbon2 = new TGeoTubeSeg(Form("holeCarbon2_D2_H%d",half), 0., mRMin[disk], mCarbonThickness + 0.000001, 0, 180.);
  TGeoTranslation *t22= new TGeoTranslation ("t22",0., - mHalfDiskGap , 0.);
  t22-> RegisterYourself();
 

  ////TGeoCompositeShape *cs2 = new TGeoCompositeShape(Form("carbon2_D2_H%d",half),Form("(carbonBase2_D2_H%d:t21)-(holeCarbon2_D2_H%d:t22)",half,half));
  TGeoSubtraction    *carbonhole2 = new TGeoSubtraction(carbonBase2, holeCarbon2, t21, t22);
  TGeoCompositeShape *cs2 = new TGeoCompositeShape(Form("Carbon2_D2_H%d",half), carbonhole2);
  TGeoVolume *carbonBaseWithHole2 = new TGeoVolume(Form("carbonBaseWithHole_D2_H%d", half), cs2, carbon);

  carbonBaseWithHole2->SetLineColor(kGray+3);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  carbonPlate->AddNode(carbonBaseWithHole2, 0, new TGeoTranslation(0., 0., mZPlan[disk]));
  
  Double_t ty = mSupportYDimensions[disk][0];
  
  for (Int_t ipart=1; ipart<fnPart[disk]; ipart ++) {
    ty += mSupportYDimensions[disk][ipart]/2.;
    TGeoVolume *partCarbon = gGeoManager->MakeBox(Form("partCarbon_D2_H%d_%d", half, ipart), carbon, mSupportXDimensions[disk][ipart]/2., mSupportYDimensions[disk][ipart]/2., mCarbonThickness);
    partCarbon->SetLineColor(kGray+3);
    TGeoTranslation *t = new TGeoTranslation ("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    carbonPlate -> AddNode(partCarbon, ipart, t);
    ty += mSupportYDimensions[disk][ipart]/2.;
  }
  
//  if (half == kTop) {
//    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
//    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
//    fHalfDisk->AddNode(carbonPlate, 1, transformation);
//    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
//    fHalfDisk->AddNode(carbonPlate, 2, transformation);
//  }
//  else if (half == kBottom) {
    rotation = new TGeoRotation ("rotation", 180., 0., 0.);
    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
    mHalfDisk->AddNode(carbonPlate, 3, transformation);
    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
    mHalfDisk->AddNode(carbonPlate, 4, transformation);
//  }


  // **************************************** Rohacell Plate ****************************************
  
  TGeoVolumeAssembly *rohacellPlate = new TGeoVolumeAssembly(Form("rohacellPlate_D2_H%d",half));
  
  TGeoBBox *rohacellBase2 = new TGeoBBox (Form("rohacellBase2_D2_H%d",half),  (mSupportXDimensions[disk][0])/2., (mSupportYDimensions[disk][0])/2., mRohacellThickness);
  // TGeoTranslation *t3 = new TGeoTranslation ("t3",0., (fSupportYDimensions[disk][0])/2. + fHalfDiskGap , 0.);
  // t3 -> RegisterYourself();
  
  TGeoTubeSeg *holeRohacell2 = new TGeoTubeSeg(Form("holeRohacell2_D2_H%d",half), 0., mRMin[disk], mRohacellThickness + 0.000001, 0, 180.);
  // TGeoTranslation *t4= new TGeoTranslation ("t4", 0., - fHalfDiskGap , 0.);
  // t4-> RegisterYourself()
  ;
  
  ///cs2 = new TGeoCompositeShape(Form("rohacell_D2_H%d",half), Form("(rohacellBase2_D2_H%d:t21)-(holeRohacell2_D2_H%d:t22)",half,half));
  TGeoSubtraction    *rohacellhole2 = new TGeoSubtraction(rohacellBase2, holeRohacell2, t21, t22);
  TGeoCompositeShape *rh2 = new TGeoCompositeShape(Form("rohacellBase2_D2_H%d",half), rohacellhole2);
  TGeoVolume *rohacellBaseWithHole = new TGeoVolume(Form("rohacellBaseWithHole_D2_H%d",half), rh2, rohacell);



  rohacellBaseWithHole->SetLineColor(kGray);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation =  new TGeoCombiTrans(0., 0., 0., rotation);
  rohacellPlate -> AddNode(rohacellBaseWithHole, 0, new TGeoTranslation(0., 0., mZPlan[disk]));
  
  ty = mSupportYDimensions[disk][0];
  
  for (Int_t ipart=1; ipart<fnPart[disk]; ipart ++) {
    ty += mSupportYDimensions[disk][ipart]/2.;
    TGeoVolume *partRohacell = gGeoManager->MakeBox(Form("partRohacelli_D2_H%d_%d", half,ipart), rohacell, mSupportXDimensions[disk][ipart]/2., mSupportYDimensions[disk][ipart]/2., mRohacellThickness);
    partRohacell->SetLineColor(kGray);
    TGeoTranslation *t = new TGeoTranslation ("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    rohacellPlate -> AddNode(partRohacell, ipart, t);
    ty += mSupportYDimensions[disk][ipart]/2.;
  }
  
//  if (half == kTop) {
//    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
//    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
//    fHalfDisk->AddNode(rohacellPlate, 1, transformation);
//  }
//  if (half == kBottom) {
    rotation = new TGeoRotation ("rotation", 180., 0., 0.);
    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
    mHalfDisk->AddNode(rohacellPlate, 2, transformation);
//  }
  
}

//_____________________________________________________________________________
void HeatExchanger::CreateHalfDisk3(Int_t half)  {
  
  Int_t disk = 3;
  
  if      (half == kTop)    printf("Creating MFT heat exchanger for disk3 top\n");
  else if (half == kBottom) printf("Creating MFT heat exchanger for disk3 bottom\n");
  else     printf("No valid option for MFT heat exchanger on disk3\n");
  
  //carbon   = gGeoManager->GetMedium("MFT_Carbon$");
  carbon   = gGeoManager->GetMedium("MFT_CarbonFiber$");

  water    = gGeoManager->GetMedium("MFT_Water$");
  rohacell = gGeoManager->GetMedium("MFT_Rohacell");
  pipe     = gGeoManager->GetMedium("MFT_Polyimide");
  
  TGeoVolumeAssembly *cooling = new TGeoVolumeAssembly(Form("cooling_D3_H%d",half));
  
  Double_t deltaz= mHeatExchangerThickness - mCarbonThickness*2; //distance between pair of carbon plans
  Double_t lMiddle3[3] = {mSupportXDimensions[3][0] - 2.*mLWater3[0], mSupportXDimensions[3][0] - 2.*mLWater3[0], 0.};//distance between tube part
  
  TGeoTranslation *translation    = 0;
  TGeoRotation    *rotation       = 0;
  TGeoCombiTrans  *transformation = 0;
  
  Double_t beta3rad[3] = {0., 0., 0.};
  for (Int_t i=0; i<3; i++) {
    beta3rad[i] = fangle3[i]*(TMath::DegToRad());
  }
  Double_t fangleThirdPipe3rad=  fangleThirdPipe3*(TMath::DegToRad());
  
  Double_t radius3mid[2] = {((lMiddle3[0]) - 2.*(fradius3[0]*(TMath::Sin(beta3rad[0])) +
                                                 mLpartial3[0]*(TMath::Cos(beta3rad[0]))))/ (2*(TMath::Sin(beta3rad[0]))), 0.};//radius of central torus
  radius3mid[1] = (mSupportXDimensions[3][0]/2. - mLWater3[2]*TMath::Cos(fangleThirdPipe3rad) -
                   fradius3[2]*(TMath::Sin(beta3rad[2] + fangleThirdPipe3rad) - TMath::Sin(fangleThirdPipe3rad)))/(TMath::Sin(fangleThirdPipe3rad + beta3rad[2]));
  
  lMiddle3[2] = mSupportXDimensions[3][0] - 2.*mLWater3[2]*(TMath::Cos(fangleThirdPipe3rad));
  
  
  // **************************************** Water part ****************************************
  
  // ------------------- First and second pipe -------------------
  
  for (Int_t itube= 0; itube < 2; itube ++){
    
    
    // -------- Tube shape --------
    
    TGeoVolume *waterTube1 = gGeoManager->MakeTube(Form("waterTubeone%d_D3_H%d", itube,half), water, 0., mRWater, mLWater3[itube]/2.);
    waterTube1->SetLineColor(kBlue);
    translation = new TGeoTranslation (mXPosition3[itube], 0., mLWater3[itube]/2. + lMiddle3[itube]/2.);
    cooling->AddNode (waterTube1, 1, translation);
    
    TGeoVolume *waterTube2 = gGeoManager->MakeTube(Form("waterTubetwo%d_D3_H%d", itube,half), water, 0., mRWater, mLWater3[itube]/2.);
    waterTube2->SetLineColor(kBlue);
    translation = new TGeoTranslation (mXPosition3[itube], 0., -mLWater3[itube]/2. - lMiddle3[itube]/2.);
    cooling->AddNode (waterTube2, 2, translation);
    
    TGeoVolume *waterTube3 = gGeoManager->MakeTube(Form("waterTubethree%d_D3_H%d", itube,half), water, 0., mRWater, mLpartial3[itube]/2.);
    waterTube3->SetLineColor(kBlue);
    rotation = new TGeoRotation ("rotation", -90., 0 - fangle3[itube], 0.);
    
    transformation = new TGeoCombiTrans(mXPosition3[itube] + fradius3[itube]*(1-(TMath::Cos(beta3rad[0]))) +
                                        (mLpartial3[itube]/2.)*(TMath::Sin(beta3rad[0])), 0., (fradius3[itube])*(TMath::Sin(beta3rad[0])) +
                                        (mLpartial3[itube]/2.)*(TMath::Cos(beta3rad[0])) - lMiddle3[itube]/2., rotation);
    cooling->AddNode (waterTube3, 3, transformation);
    
    rotation = new TGeoRotation ("rotation", 90., 180 - fangle3[itube], 0.);
    transformation = new TGeoCombiTrans( mXPosition3[itube] + fradius3[itube]*(1-(TMath::Cos(beta3rad[0]))) +
                                        (mLpartial3[itube]/2.)*(TMath::Sin(beta3rad[0])), 0., lMiddle3[itube]/2. - (fradius3[itube])*(TMath::Sin(beta3rad[0])) -
                                        (mLpartial3[itube]/2.)*(TMath::Cos(beta3rad[0])), rotation);
    cooling->AddNode (waterTube3, 4, transformation);
    
    // -------- Torus shape --------
    
    //Sides torus
    TGeoVolume *waterTorus1 = gGeoManager->MakeTorus(Form("waterTorusone%d_D3_H%d", itube,half), water, fradius3[itube], 0., mRWater, 0., fangle3[itube]);
    waterTorus1->SetLineColor(kBlue);
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(fradius3[itube] + mXPosition3[itube], 0., - lMiddle3[itube]/2., rotation);
    cooling->AddNode (waterTorus1, 4, transformation);
    rotation = new TGeoRotation ("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(fradius3[itube] + mXPosition3[itube], 0., lMiddle3[itube]/2., rotation);
    cooling->AddNode (waterTorus1, 5, transformation);
    
    //Central torus
    TGeoVolume *waterTorus2 = gGeoManager->MakeTorus(Form("waterTorustwo%d_D3_H%d", itube,half), water, radius3mid[0], 0., mRWater, -fangle3[itube], 2.*fangle3[itube]);
    waterTorus2->SetLineColor(kBlue);
    rotation = new TGeoRotation ("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(mXPosition3[itube] + fradius3[0]*(1-(TMath::Cos(beta3rad[0])))+mLpartial3[0]*TMath::Sin(beta3rad[0]) -
                                        radius3mid[0]*TMath::Cos(beta3rad[0]) , 0., 0., rotation);
    cooling->AddNode (waterTorus2, 6, transformation);
  }
  
  // ------------------- Third pipe -------------------
  
  // -------- Tube shape --------
  
  TGeoVolume *waterTube1 = gGeoManager->MakeTube(Form("waterTubeone2_D3_H%d",half), water, 0., mRWater, mLWater3[2]/2.);
  waterTube1->SetLineColor(kBlue);
  rotation = new TGeoRotation ("rotation", 90., -fangleThirdPipe3, 90.);
  transformation = new TGeoCombiTrans (mXPosition3[2] + mLWater3[2]*TMath::Sin(fangleThirdPipe3rad)/2., 0.,
                                       mSupportXDimensions[3][0]/2. - mLWater3[2]*(TMath::Cos(fangleThirdPipe3rad))/2., rotation);
  cooling->AddNode (waterTube1, 3, transformation);
  
  rotation = new TGeoRotation ("rotation", 90., fangleThirdPipe3, 90.);
  transformation = new TGeoCombiTrans (mXPosition3[2] + mLWater3[2]*TMath::Sin(fangleThirdPipe3rad)/2., 0.,
                                       -mSupportXDimensions[3][0]/2. + mLWater3[2]*(TMath::Cos(fangleThirdPipe3rad))/2., rotation);
  cooling->AddNode (waterTube1, 4, transformation);
  
  // -------- Torus shape --------
  
  TGeoVolume *waterTorus1 = gGeoManager->MakeTorus(Form("waterTorusone2_D3_H%d",half), water, fradius3[2], 0., mRWater, fangleThirdPipe3, fangle3[2]);
  waterTorus1->SetLineColor(kBlue);
  rotation = new TGeoRotation ("rotation", 180., 90., 0.);
  transformation = new TGeoCombiTrans(mXPosition3[2] + mLWater3[2]*TMath::Sin(fangleThirdPipe3rad) +
                                      fradius3[2]*(TMath::Cos(fangleThirdPipe3rad)), 0., -lMiddle3[2]/2. - fradius3[2]*(TMath::Sin(fangleThirdPipe3rad)) , rotation);
  cooling->AddNode (waterTorus1, 4, transformation);
  
  rotation = new TGeoRotation ("rotation", 180., -90., 0.);
  transformation = new TGeoCombiTrans( mXPosition3[2] + mLWater3[2]*TMath::Sin(fangleThirdPipe3rad) +
                                      fradius3[2]*(TMath::Cos(fangleThirdPipe3rad)), 0.,  lMiddle3[2]/2. + fradius3[2]*(TMath::Sin(fangleThirdPipe3rad)), rotation);
  cooling->AddNode (waterTorus1, 5, transformation);
  
  TGeoVolume *waterTorus2 = gGeoManager->MakeTorus(Form("waterTorustwo2_D3_H%d",half), water, radius3mid[1], 0., mRWater, -(fangle3[2] + fangleThirdPipe3),
                                                   2.*(fangle3[2] + fangleThirdPipe3));
  waterTorus2->SetLineColor(kBlue);
  rotation = new TGeoRotation ("rotation", 0., 90., 0.);
  transformation = new TGeoCombiTrans( mXPosition3[2] + mLWater3[2]*TMath::Sin(fangleThirdPipe3rad) + fradius3[2]*(TMath::Cos(fangleThirdPipe3rad)) -
                                      (fradius3[2] + radius3mid[1])*(TMath::Cos(beta3rad[2] + fangleThirdPipe3rad)), 0., 0., rotation);
  cooling->AddNode (waterTorus2, 6, transformation);
  
  // ------------------- Fourth pipe -------------------
  
  Double_t radius3fourth[4] = {9.6, 2.9, 2.9, 0.};
  Double_t alpha3fourth[4] = { 40.8, 50, 60, 0};  //size angle
  alpha3fourth[3] = 8 + alpha3fourth[0] - alpha3fourth[1] + alpha3fourth[2];
  Double_t alpha3fourthrad[4] = {};
  for(Int_t i=0; i<4; i++){
    alpha3fourthrad[i] = (TMath::Pi())*(alpha3fourth[i])/180.;
  }
  Double_t beta3fourth[3] = {8, 8 + alpha3fourth[0], -(-8 - alpha3fourth[0] + alpha3fourth[1])};  //shift angle
  Double_t beta3fourthrad[3] = {0., 0., 0.};
  for(Int_t i=0; i<3; i++){
    beta3fourthrad[i] = (TMath::Pi())*(beta3fourth[i])/180.;
  }
  
  radius3fourth[3] = ((-(-(mLWater3[0] + lMiddle3[0]/2.) -
                         radius3fourth[0]*(TMath::Sin(beta3fourthrad[0])) +
                         radius3fourth[0]*(TMath::Sin(beta3fourthrad[0] + alpha3fourthrad[0])) +
                         radius3fourth[1]*(TMath::Cos(TMath::Pi()/2 - beta3fourthrad[0] - alpha3fourthrad[0])) +
                         radius3fourth[1]*(TMath::Cos(TMath::Pi()/2. - alpha3fourthrad[1] + alpha3fourthrad[0] + beta3fourthrad[0])) +
                         radius3fourth[2]*(TMath::Sin(alpha3fourthrad[1] - alpha3fourthrad[0] - beta3fourthrad[0])))) -
                      radius3fourth[2]*TMath::Cos(TMath::Pi()/2 - alpha3fourthrad[3]))/(TMath::Sin(alpha3fourthrad[3]));
  
  Double_t translation3x[4] = { mXPosition3[3] + radius3fourth[0]*(TMath::Cos(beta3fourthrad[0])),
				mXPosition3[3] + radius3fourth[0]*((TMath::Cos(beta3fourthrad[0])) - TMath::Cos(beta3fourthrad[0] + alpha3fourthrad[0])) -
				radius3fourth[1]*(TMath::Cos(beta3fourthrad[0] + alpha3fourthrad[0])),
				mXPosition3[3] + radius3fourth[0]*((TMath::Cos(beta3fourthrad[0])) - TMath::Cos(beta3fourthrad[0] + alpha3fourthrad[0])) -
				radius3fourth[1]*(TMath::Cos(beta3fourthrad[0] + alpha3fourthrad[0])) +
				radius3fourth[1]*(TMath::Sin(TMath::Pi()/2. - alpha3fourthrad[1] + alpha3fourthrad[0] + beta3fourthrad[0])) +
				radius3fourth[2]*(TMath::Cos(alpha3fourthrad[1] - alpha3fourthrad[0] - beta3fourthrad[0])),
				mXPosition3[3] + radius3fourth[0]*((TMath::Cos(beta3fourthrad[0])) - TMath::Cos(beta3fourthrad[0] + alpha3fourthrad[0])) -
				radius3fourth[1]*(TMath::Cos(beta3fourthrad[0] + alpha3fourthrad[0])) +
				radius3fourth[1]*(TMath::Sin(TMath::Pi()/2. - alpha3fourthrad[1] + alpha3fourthrad[0] + beta3fourthrad[0])) +
				radius3fourth[2]*(TMath::Cos(alpha3fourthrad[1] - alpha3fourthrad[0] - beta3fourthrad[0])) -
				radius3fourth[2]*(TMath::Sin((TMath::Pi()/2.) - alpha3fourthrad[3])) - radius3fourth[3]*(TMath::Cos(alpha3fourthrad[3]))};
  
  Double_t translation3y[3] = {0., 0., 0.};
  
  Double_t translation3z[3] = {-(mLWater3[0] + lMiddle3[0]/2.) - radius3fourth[0]*(TMath::Sin(beta3fourthrad[0])),
    -(mLWater3[0] + lMiddle3[0]/2.) - radius3fourth[0]*(TMath::Sin(beta3fourthrad[0])) +
    radius3fourth[0]*(TMath::Sin(beta3fourthrad[0] + alpha3fourthrad[0])) + radius3fourth[1]*(TMath::Sin(beta3fourthrad[0] + alpha3fourthrad[0])),
    -(mLWater3[0] + lMiddle3[0]/2.) - radius3fourth[0]*(TMath::Sin(beta3fourthrad[0])) +
    radius3fourth[0]*(TMath::Sin(beta3fourthrad[0] + alpha3fourthrad[0])) +
    radius3fourth[1]*(TMath::Cos(TMath::Pi()/2 - beta3fourthrad[0] - alpha3fourthrad[0])) +
    radius3fourth[1]*(TMath::Cos(TMath::Pi()/2. - alpha3fourthrad[1] + alpha3fourthrad[0] +
                                 beta3fourthrad[0])) + radius3fourth[2]*(TMath::Sin(alpha3fourthrad[1] - alpha3fourthrad[0] - beta3fourthrad[0]))};
  
  Double_t rotation3x[3] = {180., 180., 180.};
  Double_t rotation3y[3] = {90., 90., 90.};
  Double_t rotation3z[3] = {0., 180 - alpha3fourth[1]  , 0.};
  
  for (Int_t i= 0; i<3; i++) {
    waterTorus1 = gGeoManager->MakeTorus(Form("waterTorusone%d_D3_H%d", i,half), water, radius3fourth[i], 0., mRWater, beta3fourth[i],  alpha3fourth[i]);
    waterTorus1->SetLineColor(kBlue);
    rotation = new TGeoRotation ("rotation", rotation3x[i], rotation3y[i], rotation3z[i]);
    transformation = new TGeoCombiTrans(translation3x[i], translation3y[i], translation3z[i], rotation);
    cooling->AddNode (waterTorus1, 7, transformation);
    rotation = new TGeoRotation ("rotation", rotation3x[i] , rotation3y[i] - 180, rotation3z[i]);
    transformation = new TGeoCombiTrans(translation3x[i], translation3y[i], - translation3z[i], rotation);
    cooling->AddNode (waterTorus1, 8, transformation);
  }
  
  waterTorus2 = gGeoManager->MakeTorus(Form("waterTorusone3_D3_H%d",half), water, radius3fourth[3], 0., mRWater, -alpha3fourth[3], 2*alpha3fourth[3]);
  waterTorus2->SetLineColor(kBlue);
  rotation = new TGeoRotation ("rotation", 180., 90., 180);
  transformation = new TGeoCombiTrans(translation3x[3], 0., 0., rotation);
  cooling->AddNode(waterTorus2, 9, transformation);
  
  // **************************************** Pipe part ****************************************
  
  // ------------------- First and second pipe -------------------
  
  for (Int_t itube= 0; itube < 2; itube ++){
    
    // -------- Tube shape --------
    
    TGeoVolume *pipeTube1 = gGeoManager->MakeTube(Form("pipeTubeone%d_D3_H%d", itube,half), pipe, mRWater, mRWater + mDRPipe, mLWater3[itube]/2.);
    pipeTube1->SetLineColor(10);
    translation = new TGeoTranslation (mXPosition3[itube], 0., mLWater3[itube]/2. + lMiddle3[itube]/2.);
    cooling->AddNode (pipeTube1, 1, translation);
    
    TGeoVolume *pipeTube2 = gGeoManager->MakeTube(Form("pipeTubetwo%d_D3_H%d", itube,half), pipe, mRWater, mRWater + mDRPipe, mLWater3[itube]/2.);
    pipeTube2->SetLineColor(10);
    translation = new TGeoTranslation (mXPosition3[itube], 0., -mLWater3[itube]/2. - lMiddle3[itube]/2.);
    cooling->AddNode (pipeTube2, 2, translation);
    
    TGeoVolume *pipeTube3 = gGeoManager->MakeTube(Form("pipeTubethree%d_D3_H%d", itube,half), pipe, mRWater, mRWater + mDRPipe, mLpartial3[itube]/2.);
    pipeTube3->SetLineColor(10);
    rotation = new TGeoRotation ("rotation", -90., 0 - fangle3[itube], 0.);
    
    transformation = new TGeoCombiTrans(mXPosition3[itube] + fradius3[itube]*(1-(TMath::Cos(beta3rad[0]))) +
                                        (mLpartial3[itube]/2.)*(TMath::Sin(beta3rad[0])), 0., (fradius3[itube])*(TMath::Sin(beta3rad[0])) +
                                        (mLpartial3[itube]/2.)*(TMath::Cos(beta3rad[0])) - lMiddle3[itube]/2., rotation);
    cooling->AddNode (pipeTube3, 3, transformation);
    
    rotation = new TGeoRotation ("rotation", 90., 180 - fangle3[itube], 0.);
    transformation = new TGeoCombiTrans( mXPosition3[itube] + fradius3[itube]*(1-(TMath::Cos(beta3rad[0]))) +
                                        (mLpartial3[itube]/2.)*(TMath::Sin(beta3rad[0])), 0., lMiddle3[itube]/2. -
                                        (fradius3[itube])*(TMath::Sin(beta3rad[0])) - (mLpartial3[itube]/2.)*(TMath::Cos(beta3rad[0])), rotation);
    cooling->AddNode (pipeTube3, 4, transformation);
    
    // -------- Torus shape --------
    
    //Sides torus
    TGeoVolume *pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorusone%d_D3_H%d", itube,half), pipe, fradius3[itube], mRWater, mRWater + mDRPipe, 0., fangle3[itube]);
    pipeTorus1->SetLineColor(10);
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(fradius3[itube] + mXPosition3[itube], 0., - lMiddle3[itube]/2., rotation);
    cooling->AddNode (pipeTorus1, 4, transformation);
    rotation = new TGeoRotation ("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(fradius3[itube] + mXPosition3[itube], 0., lMiddle3[itube]/2., rotation);
    cooling->AddNode (pipeTorus1, 5, transformation);
    
    
    //Central torus
    TGeoVolume *pipeTorus2 = gGeoManager->MakeTorus(Form("pipeTorustwo%d_D3_H%d", itube,half), pipe, radius3mid[0], mRWater, mRWater + mDRPipe, -fangle3[itube], 2.*fangle3[itube]);
    pipeTorus2->SetLineColor(10);
    rotation = new TGeoRotation ("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(mXPosition3[itube] + fradius3[0]*(1-(TMath::Cos(beta3rad[0])))+mLpartial3[0]*TMath::Sin(beta3rad[0]) - radius3mid[0]*TMath::Cos(beta3rad[0]) , 0., 0., rotation);
    cooling->AddNode (pipeTorus2, 6, transformation);
  }
  
  // ------------------- Third pipe -------------------
  
  // -------- Tube shape --------
  
  TGeoVolume *pipeTube1 = gGeoManager->MakeTube(Form("pipeTubeone2_D3_H%d",half), pipe, mRWater, mRWater + mDRPipe, mLWater3[2]/2.);
  pipeTube1->SetLineColor(10);
  rotation = new TGeoRotation ("rotation", 90., -fangleThirdPipe3, 90.);
  transformation = new TGeoCombiTrans (mXPosition3[2] + mLWater3[2]*TMath::Sin(fangleThirdPipe3rad)/2., 0.,
                                       mSupportXDimensions[3][0]/2. - mLWater3[2]*(TMath::Cos(fangleThirdPipe3rad))/2., rotation);
  cooling->AddNode (pipeTube1, 3, transformation);
  
  rotation = new TGeoRotation ("rotation", 90., fangleThirdPipe3, 90.);
  transformation = new TGeoCombiTrans (mXPosition3[2] + mLWater3[2]*TMath::Sin(fangleThirdPipe3rad)/2., 0.,
                                       -mSupportXDimensions[3][0]/2. + mLWater3[2]*(TMath::Cos(fangleThirdPipe3rad))/2., rotation);
  cooling->AddNode (pipeTube1, 4, transformation);
  
  // -------- Torus shape --------
  
  TGeoVolume *pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorusone2_D3_H%d",half), pipe, fradius3[2], mRWater, mRWater + mDRPipe, fangleThirdPipe3, fangle3[2]);
  pipeTorus1->SetLineColor(10);
  rotation = new TGeoRotation ("rotation", 180., 90., 0.);
  transformation = new TGeoCombiTrans(mXPosition3[2] + mLWater3[2]*TMath::Sin(fangleThirdPipe3rad) +
                                      fradius3[2]*(TMath::Cos(fangleThirdPipe3rad)), 0., -lMiddle3[2]/2. - fradius3[2]*(TMath::Sin(fangleThirdPipe3rad)), rotation);
  cooling->AddNode (pipeTorus1, 4, transformation);
  rotation = new TGeoRotation ("rotation", 180., -90., 0.);
  transformation = new TGeoCombiTrans( mXPosition3[2] + mLWater3[2]*TMath::Sin(fangleThirdPipe3rad) +
                                      fradius3[2]*(TMath::Cos(fangleThirdPipe3rad)), 0.,  lMiddle3[2]/2. + fradius3[2]*(TMath::Sin(fangleThirdPipe3rad)), rotation);
  cooling->AddNode (pipeTorus1, 5, transformation);
  
  TGeoVolume *pipeTorus2 = gGeoManager->MakeTorus(Form("pipeTorustwo2_D3_H%d",half), pipe, radius3mid[1], mRWater, mRWater + mDRPipe,
                                                  -(fangle3[2] + fangleThirdPipe3), 2.*(fangle3[2] + fangleThirdPipe3));
  pipeTorus2->SetLineColor(10);
  rotation = new TGeoRotation ("rotation", 0., 90., 0.);
  transformation = new TGeoCombiTrans( mXPosition3[2] + mLWater3[2]*TMath::Sin(fangleThirdPipe3rad) +
                                      fradius3[2]*(TMath::Cos(fangleThirdPipe3rad)) -
                                      (fradius3[2] + radius3mid[1])*(TMath::Cos(beta3rad[2] + fangleThirdPipe3rad)), 0., 0., rotation);
  cooling->AddNode (pipeTorus2, 6, transformation);
  
  // ------------------- Fourth pipe -------------------
  
  for(Int_t i= 0; i<3; i++){
    pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorusone%d_D3_H%d", i,half), pipe, radius3fourth[i], mRWater, mRWater + mDRPipe, beta3fourth[i],  alpha3fourth[i]);
    pipeTorus1->SetLineColor(10);
    rotation = new TGeoRotation ("rotation", rotation3x[i], rotation3y[i], rotation3z[i]);
    transformation = new TGeoCombiTrans(translation3x[i], translation3y[i], translation3z[i], rotation);
    cooling->AddNode (pipeTorus1, 7, transformation);
    rotation = new TGeoRotation ("rotation", rotation3x[i] , rotation3y[i] - 180, rotation3z[i]);
    transformation = new TGeoCombiTrans(translation3x[i], translation3y[i], - translation3z[i], rotation);
    cooling->AddNode (pipeTorus1, 8, transformation);
  }
  
  pipeTorus2 = gGeoManager->MakeTorus(Form("pipeTorusone3_D3_H%d",half), pipe, radius3fourth[3], mRWater, mRWater + mDRPipe, -alpha3fourth[3], 2*alpha3fourth[3]);
  pipeTorus2->SetLineColor(10);
  rotation = new TGeoRotation ("rotation", 180., 90., 180);
  transformation = new TGeoCombiTrans(translation3x[3], 0., 0., rotation);
  cooling->AddNode(pipeTorus2, 9, transformation);
  
//  if (half == kTop) {
//    rotation = new TGeoRotation ("rotation", 90., 90., 0.);
//    transformation = new TGeoCombiTrans(0., 0., fZPlan[disk] + deltaz/2. - fCarbonThickness - fRWater - fDRPipe, rotation);
//    fHalfDisk->AddNode(cooling, 1, transformation);
//    transformation = new TGeoCombiTrans(0., 0., fZPlan[disk] - deltaz/2. + fCarbonThickness + fRWater + fDRPipe, rotation);
//    fHalfDisk->AddNode(cooling, 2, transformation);
//  }
//  else if (half == kBottom) {
    rotation = new TGeoRotation ("rotation", -90., 90., 0.);
    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz/2. - mCarbonThickness - mRWater - mDRPipe, rotation);
    mHalfDisk->AddNode(cooling, 3, transformation);
    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz/2. + mCarbonThickness + mRWater + mDRPipe, rotation);
    mHalfDisk->AddNode(cooling, 4, transformation);
//  }
	
  // **************************************** Carbon Plates ****************************************
  
  TGeoVolumeAssembly *carbonPlate = new TGeoVolumeAssembly(Form("carbonPlate_D3_H%d",half));
  
  TGeoBBox *carbonBase3 = new TGeoBBox (Form("carbonBase3_D3_H%d",half),  (mSupportXDimensions[disk][0])/2., (mSupportYDimensions[disk][0])/2., mCarbonThickness);
  TGeoTranslation *t31= new TGeoTranslation ("t31",0., (mSupportYDimensions[disk][0])/2.+ mHalfDiskGap , 0.);
  t31-> RegisterYourself();
  
  TGeoTubeSeg *holeCarbon3 = new TGeoTubeSeg(Form("holeCarbon3_D3_H%d",half), 0., mRMin[disk], mCarbonThickness + 0.000001, 0, 180.);
  TGeoTranslation *t32= new TGeoTranslation ("t32",0., - mHalfDiskGap , 0.);
  t32-> RegisterYourself();
  

  ///TGeoCompositeShape *cs3 = new TGeoCompositeShape(Form("Carbon3_D3_H%d",half),Form("(carbonBase3_D3_H%d:t31)-(holeCarbon3_D3_H%d:t32)",half,half) );
  TGeoSubtraction    *carbonhole3 = new TGeoSubtraction(carbonBase3, holeCarbon3, t31, t32);
  TGeoCompositeShape *cs3 = new TGeoCompositeShape(Form("Carbon3_D3_H%d",half), carbonhole3);
  TGeoVolume *carbonBaseWithHole3 = new TGeoVolume(Form("carbonBaseWithHole_D3_H%d",half), cs3, carbon);


  carbonBaseWithHole3->SetLineColor(kGray+3);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  carbonPlate->AddNode(carbonBaseWithHole3, 0, new TGeoTranslation(0., 0., mZPlan[disk]));
  
  Double_t ty = mSupportYDimensions[disk][0];
  
  for (Int_t ipart=1; ipart<fnPart[disk]; ipart ++) {
    ty += mSupportYDimensions[disk][ipart]/2.;
    TGeoVolume *partCarbon = gGeoManager->MakeBox(Form("partCarbon_D3_H%d_%d", half,ipart), carbon, mSupportXDimensions[disk][ipart]/2.,
                                                  mSupportYDimensions[disk][ipart]/2., mCarbonThickness);
    partCarbon->SetLineColor(kGray+3);
    TGeoTranslation *t = new TGeoTranslation ("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    carbonPlate -> AddNode(partCarbon, ipart, t);
    ty += mSupportYDimensions[disk][ipart]/2.;
  }
  
//  if (half == kTop) {
//    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
//    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
//    fHalfDisk->AddNode(carbonPlate, 1, transformation);
//    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
//    fHalfDisk->AddNode(carbonPlate, 2, transformation);
//  }
//  else if (half == kBottom) {
    rotation = new TGeoRotation ("rotation", 180., 0., 0.);
    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
    mHalfDisk->AddNode(carbonPlate, 3, transformation);
    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
    mHalfDisk->AddNode(carbonPlate, 4, transformation);
//  }
	

  // **************************************** Rohacell Plate ****************************************
  
  TGeoVolumeAssembly *rohacellPlate = new TGeoVolumeAssembly(Form("rohacellPlate_D3_H%d",half));
  
  TGeoBBox *rohacellBase3 = new TGeoBBox (Form("rohacellBase3_D3_H%d",half),  (mSupportXDimensions[disk][0])/2., (mSupportYDimensions[disk][0])/2., mRohacellThickness);
  // TGeoTranslation *t3 = new TGeoTranslation ("t3",0., (fSupportYDimensions[disk][0])/2. + fHalfDiskGap , 0.);
  // t3 -> RegisterYourself();
  
  TGeoTubeSeg *holeRohacell3 = new TGeoTubeSeg(Form("holeRohacell3_D3_H%d",half), 0., mRMin[disk], mRohacellThickness + 0.000001, 0, 180.);
  // TGeoTranslation *t4= new TGeoTranslation ("t4", 0., - fHalfDiskGap , 0.);
  // t4-> RegisterYourself();
  
  ///cs3 = new TGeoCompositeShape(Form("rohacell_D3_H%d",half), Form("(rohacellBase3_D3_H%d:t31)-(holeRohacell3_D3_H%d:t32)",half,half));
  TGeoSubtraction    *rohacellhole3 = new TGeoSubtraction(rohacellBase3, holeRohacell3, t31, t32);
  TGeoCompositeShape *rh3 = new TGeoCompositeShape(Form("rohacellBase3_D3_H%d",half), rohacellhole3);
  TGeoVolume *rohacellBaseWithHole = new TGeoVolume(Form("rohacellBaseWithHole_D3_H%d",half), rh3, rohacell);


  rohacellBaseWithHole->SetLineColor(kGray);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation =  new TGeoCombiTrans(0., 0., 0., rotation);
  rohacellPlate -> AddNode(rohacellBaseWithHole, 0, new TGeoTranslation(0., 0., mZPlan[disk]));
  
  ty = mSupportYDimensions[disk][0];
  
  for (Int_t ipart=1; ipart<fnPart[disk]; ipart ++) {
    ty += mSupportYDimensions[disk][ipart]/2.;
    TGeoVolume *partRohacell = gGeoManager->MakeBox(Form("partRohacelli_D3_H%d_%d", half, ipart), rohacell, mSupportXDimensions[disk][ipart]/2.,
                                                    mSupportYDimensions[disk][ipart]/2., mRohacellThickness);
    partRohacell->SetLineColor(kGray);
    TGeoTranslation *t = new TGeoTranslation ("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    rohacellPlate -> AddNode(partRohacell, ipart, t);
    ty += mSupportYDimensions[disk][ipart]/2.;
  }
  
//  if (half == kTop) {
//    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
//    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
//    fHalfDisk->AddNode(rohacellPlate, 1, transformation);
//  }
//  if (half == kBottom) {
    rotation = new TGeoRotation ("rotation", 180., 0., 0.);
    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
    mHalfDisk->AddNode(rohacellPlate, 2, transformation);
//  }
	
  
}

//_____________________________________________________________________________
void HeatExchanger::CreateHalfDisk4(Int_t half) {
  
  Int_t disk = 4;
  
  if      (half == kTop)    printf("Creating MFT heat exchanger for disk4 top\n");
  else if (half == kBottom) printf("Creating MFT heat exchanger for disk4 bottom\n");
  else     printf("No valid option for MFT heat exchanger on disk4\n");
  
  
  //carbon   = gGeoManager->GetMedium("MFT_Carbon$");
  carbon   = gGeoManager->GetMedium("MFT_CarbonFiber$");

  water    = gGeoManager->GetMedium("MFT_Water$");
  rohacell = gGeoManager->GetMedium("MFT_Rohacell");
  pipe     = gGeoManager->GetMedium("MFT_Polyimide");
    
  TGeoVolumeAssembly *cooling = new TGeoVolumeAssembly(Form("cooling_D4_H%d",half));
  Double_t deltaz= mHeatExchangerThickness - mCarbonThickness*2; //distance between pair of carbon plans
  
  TGeoTranslation *translation    = 0;
  TGeoRotation    *rotation       = 0;
  TGeoCombiTrans  *transformation = 0;
  
  Double_t lMiddle4[3] = {mSupportXDimensions[4][0] - 2*mLwater4[0], mSupportXDimensions[4][0] - 2*mLwater4[1], mSupportXDimensions[4][0] - 2*mLwater4[2]};                 //distance between tube part
  fangle4[5] = (fangle4[3] - fangle4[4]);
  Double_t anglerad[6]= {0.};  //angle of the sides torus
  for(Int_t i=0; i<6; i++){
    anglerad[i] = fangle4[i]*(TMath::DegToRad());
  }
  Double_t fradius4mid[3] = { (lMiddle4[0]-2.*(fradius4[0]*(TMath::Sin(anglerad[0])) + mLpartial4[0]*(TMath::Cos(anglerad[0]))))/(2*(TMath::Sin(anglerad[0]))) ,
    (lMiddle4[1]-2.*(fradius4[1]*(TMath::Sin(anglerad[1])) + mLpartial4[1]*(TMath::Cos(anglerad[1]))))/(2*(TMath::Sin(anglerad[1]))), 0. };                // radius of the central torus
  fradius4mid[2] = (mSupportXDimensions[4][0]/2. - fradius4[3]*TMath::Sin(anglerad[3]) - fradius4[4]*(TMath::Sin(anglerad[3]) -
                                                                                                      TMath::Sin(anglerad[5])))/(TMath::Sin(anglerad[5]));
  
  // **************************************** Water part ****************************************
  
  // ------------------- First and second pipe -------------------
  
  for (Int_t i=0; i<2; i++){
    
    // -------- Tube shape --------
    
    TGeoVolume *waterTube1 = gGeoManager->MakeTube(Form("waterTubeone%d_D4_H%d", i,half), water, 0., mRWater, mLwater4[i]/2.);
    waterTube1->SetLineColor(kBlue);
    translation = new TGeoTranslation (mXposition4[i], 0., mLwater4[i]/2. + lMiddle4[i]/2.);
    cooling->AddNode (waterTube1, 1, translation);
    translation = new TGeoTranslation (mXposition4[i], 0., -mLwater4[i]/2. - lMiddle4[i]/2.);
    cooling->AddNode (waterTube1, 2, translation);
    
    TGeoVolume *waterTube2 = gGeoManager->MakeTube(Form("waterTubetwo%d_D4_H%d", i,half), water, 0., mRWater, mLpartial4[i]/2.);
    waterTube2->SetLineColor(kBlue);
    rotation = new TGeoRotation ("rotation", -90., - fangle4[i], 0.);
    transformation = new TGeoCombiTrans( mXposition4[i]+fradius4[i]*(1-(TMath::Cos(anglerad[i])))+mLpartial4[i]*TMath::Sin(anglerad[i])/2., 0.,
                                        -mSupportXDimensions[4][0]/2. + mLwater4[i] + fradius4[i]*(TMath::Sin(anglerad[i])) +
                                        mLpartial4[i]*(TMath::Cos(anglerad[i]))/2., rotation);
    cooling->AddNode (waterTube2, 3, transformation);
    rotation = new TGeoRotation ("rotation", -90.,  fangle4[i], 0.);
    transformation = new TGeoCombiTrans( mXposition4[i]+fradius4[i]*(1-(TMath::Cos(anglerad[i])))+mLpartial4[i]*TMath::Sin(anglerad[i])/2., 0.,
                                        mSupportXDimensions[4][0]/2. - mLwater4[i] - fradius4[i]*(TMath::Sin(anglerad[i])) -
                                        mLpartial4[i]*(TMath::Cos(anglerad[i]))/2. , rotation);
    cooling->AddNode (waterTube2, 4, transformation);
    
    // -------- Torus shape --------
    
    TGeoVolume *waterTorus1 = gGeoManager->MakeTorus(Form("waterTorusone%d_D4_H%d", i,half), water, fradius4[i], 0., mRWater, 0., fangle4[i]);
    waterTorus1->SetLineColor(kBlue);
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mXposition4[i] + fradius4[i], 0., -mSupportXDimensions[4][0]/2. + mLwater4[i], rotation);
    cooling->AddNode (waterTorus1, 1, transformation);
    rotation = new TGeoRotation ("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mXposition4[i] + fradius4[i], 0., mSupportXDimensions[4][0]/2. - mLwater4[i], rotation);
    cooling->AddNode (waterTorus1, 2, transformation);
    
    TGeoVolume *waterTorus2 = gGeoManager->MakeTorus(Form("waterTorustwo%d_D4_H%d", i,half), water, fradius4mid[i], 0., mRWater, 180 - fangle4[i] ,2*fangle4[i]);
    waterTorus2->SetLineColor(kBlue);
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mXposition4[i]  + fradius4[i]*(1-(TMath::Cos(anglerad[i])))+mLpartial4[i]*TMath::Sin(anglerad[i]) -
                                        fradius4mid[i]*TMath::Cos(anglerad[i]), 0., 0., rotation);
    cooling->AddNode (waterTorus2, 3, transformation);
    
  }
  
  // ------------------- Third pipe -------------------
  
  // -------- Tube shape --------
  
  TGeoVolume *waterTube1 = gGeoManager->MakeTube(Form("waterTubeone2_D4_H%d", half), water, 0., mRWater, mLwater4[2]/2.);
  waterTube1->SetLineColor(kBlue);
  translation = new TGeoTranslation (mXposition4[2], 0., mLwater4[2]/2. + lMiddle4[2]/2.);
  cooling->AddNode (waterTube1, 1, translation);
  translation = new TGeoTranslation (mXposition4[2], 0., -mLwater4[2]/2. - lMiddle4[2]/2.);
  cooling->AddNode (waterTube1, 2, translation);
  
  TGeoVolume *waterTube2 = gGeoManager->MakeTube(Form("waterTubetwo2_D4_H%d", half), water, 0., mRWater,  lMiddle4[2]/2. - 2.*fradius4[2]*TMath::Sin(anglerad[2]));
  waterTube2->SetLineColor(kBlue);
  translation = new TGeoTranslation (mXposition4[2] + 2.*fradius4[2]*(1-TMath::Cos(anglerad[2])), 0., 0.);
  cooling->AddNode (waterTube2, 3, translation);
  
  // -------- Torus shape --------
  
  TGeoVolume *waterTorus1 = gGeoManager->MakeTorus(Form("waterTorusone2_D4_H%d", half), water, fradius4[2], 0., mRWater, 0., fangle4[2]);
  waterTorus1->SetLineColor(kBlue);
  rotation = new TGeoRotation ("rotation", 180., 90., 0.);
  transformation = new TGeoCombiTrans(mXposition4[2] + fradius4[2], 0., -mSupportXDimensions[4][0]/2. + mLwater4[2], rotation);
  cooling->AddNode (waterTorus1, 1, transformation);
  rotation = new TGeoRotation ("rotation", 180., -90., 180 - fangle4[2]);
  transformation = new TGeoCombiTrans(mXposition4[2] + fradius4[2] - 2*fradius4[2]*TMath::Cos(anglerad[2]), 0.,
                                      -mSupportXDimensions[4][0]/2. + mLwater4[2] + 2*fradius4[2]*TMath::Sin(anglerad[2]), rotation);
  cooling->AddNode (waterTorus1, 2, transformation);
  rotation = new TGeoRotation ("rotation", 180., -90., 0.);
  transformation = new TGeoCombiTrans(mXposition4[2] + fradius4[2], 0., mSupportXDimensions[4][0]/2. - mLwater4[2], rotation);
  cooling->AddNode (waterTorus1, 3, transformation);
  rotation = new TGeoRotation ("rotation", 180., 90., 180 - fangle4[2]);
  transformation = new TGeoCombiTrans(mXposition4[2] + fradius4[2] - 2*fradius4[2]*TMath::Cos(anglerad[2]), 0.,
                                      mSupportXDimensions[4][0]/2. - mLwater4[2] - 2*fradius4[2]*TMath::Sin(anglerad[2]), rotation);
  cooling->AddNode (waterTorus1, 4, transformation);
  
  // ------------------- Fourth pipe -------------------
  
  waterTorus1 = gGeoManager->MakeTorus(Form("waterTorusone3_D4_H%d", half), water, fradius4[3], 0., mRWater, 0., fangle4[3]);
  waterTorus1->SetLineColor(kBlue);
  rotation = new TGeoRotation ("rotation", 180., 90., 0.);
  transformation = new TGeoCombiTrans(mXposition4[3] + fradius4[3], 0., -mSupportXDimensions[4][0]/2., rotation);
  cooling->AddNode (waterTorus1, 1, transformation);
  
  TGeoVolume *waterTorus2 = gGeoManager->MakeTorus(Form("waterTorustwo3_D4_H%d", half), water,  fradius4[4] , 0., mRWater, 0., fangle4[4]);
  waterTorus2->SetLineColor(kBlue);
  rotation = new TGeoRotation ("rotation", 180., -90., 180 - fangle4[3]);
  transformation = new TGeoCombiTrans( mXposition4[3] + fradius4[3] - fradius4[3]*TMath::Cos(anglerad[3]) -
                                      fradius4[4]*TMath::Cos(anglerad[3]), 0., -mSupportXDimensions[4][0]/2. +
                                      fradius4[3]*TMath::Sin(anglerad[3]) + fradius4[4]*TMath::Sin(anglerad[3]), rotation);
  cooling->AddNode (waterTorus2, 1, transformation);
  rotation = new TGeoRotation ("rotation", 180., -90., 0.);
  transformation = new TGeoCombiTrans(mXposition4[3] + fradius4[3], 0., mSupportXDimensions[4][0]/2., rotation);
  cooling->AddNode (waterTorus1, 2, transformation);
  rotation = new TGeoRotation ("rotation", 180., 90., 180 - fangle4[3]);
  transformation = new TGeoCombiTrans( mXposition4[3] + fradius4[3] - fradius4[3]*TMath::Cos(anglerad[3]) -
                                      fradius4[4]*TMath::Cos(anglerad[3]), 0., mSupportXDimensions[4][0]/2. -
                                      fradius4[3]*TMath::Sin(anglerad[3]) - fradius4[4]*TMath::Sin(anglerad[3]), rotation);
  cooling->AddNode (waterTorus2, 2, transformation);
  
  TGeoVolume *waterTorus3 =  gGeoManager->MakeTorus(Form("waterTorusthree3_D4_H%d", half), water,  fradius4mid[2] , 0., mRWater, -fangle4[5], 2.*fangle4[5]);
  waterTorus3->SetLineColor(kBlue);
  rotation = new TGeoRotation ("rotation", 0., 90., 0.);
  transformation = new TGeoCombiTrans( mXposition4[3] + fradius4[3] - fradius4[3]*TMath::Cos(anglerad[3]) -
                                      fradius4[4]*TMath::Cos(anglerad[3]) - ((fradius4mid[2] - fradius4[4])*TMath::Cos(anglerad[5])), 0., 0., rotation);
  cooling->AddNode (waterTorus3, 1, transformation);
  
  // ------------------- Fifth pipe -------------------
  
  fangle4fifth[3] = fangle4fifth[0] - fangle4fifth[1] + fangle4fifth[2];
  Double_t angle4fifthrad[4] = {0., 0., 0., 0.};
  for(Int_t i=0; i<4; i++){
    angle4fifthrad[i] = (TMath::Pi())*(fangle4fifth[i])/180.;
  }
  Double_t beta4fourth[4] = {0, fangle4fifth[0], fangle4fifth[0] - fangle4fifth[1], 180};  //shift angle
  Double_t beta4fourthrad[4] = {};
  for(Int_t i=0; i<4; i++){
    beta4fourthrad[i] = (TMath::Pi())*(beta4fourth[i])/180.;
  }
  Double_t translation4x[4] = { mXposition4[4] + fradius4fifth[0]*(TMath::Cos(beta4fourthrad[0])),
				mXposition4[4] + fradius4fifth[0]*((TMath::Cos(beta4fourthrad[0])) - TMath::Cos(beta4fourthrad[0] + angle4fifthrad[0])) -
				fradius4fifth[1]*(TMath::Cos(beta4fourthrad[0] + angle4fifthrad[0])),
				
				mXposition4[4] + fradius4fifth[0]*((TMath::Cos(beta4fourthrad[0])) - TMath::Cos(beta4fourthrad[0] + angle4fifthrad[0])) -
				fradius4fifth[1]*(TMath::Cos(beta4fourthrad[0] + angle4fifthrad[0])) +
				fradius4fifth[1]*(TMath::Sin(TMath::Pi()/2. - angle4fifthrad[1] + angle4fifthrad[0] + beta4fourthrad[0])) +
				fradius4fifth[2]*(TMath::Cos(angle4fifthrad[1] - angle4fifthrad[0] - beta4fourthrad[0])),
    
				mXposition4[4] + fradius4fifth[0]*((TMath::Cos(beta4fourthrad[0])) - TMath::Cos(beta4fourthrad[0] + angle4fifthrad[0])) -
				fradius4fifth[1]*(TMath::Cos(beta4fourthrad[0] + angle4fifthrad[0])) +
				fradius4fifth[1]*(TMath::Sin(TMath::Pi()/2. - angle4fifthrad[1] + angle4fifthrad[0] + beta4fourthrad[0])) +
				fradius4fifth[2]*(TMath::Cos(angle4fifthrad[1] - angle4fifthrad[0] - beta4fourthrad[0])) -
				fradius4fifth[2]*(TMath::Sin((TMath::Pi()/2.) - angle4fifthrad[3])) - fradius4fifth[3]*(TMath::Cos(angle4fifthrad[3]))};
  
  Double_t translation4y[4] = {0., 0., 0., 0.};
  
  Double_t translation4z[4] = {-(mLwater4[0] + lMiddle4[0]/2.) - fradius4fifth[0]*(TMath::Sin(beta4fourthrad[0])),
    
    -(mLwater4[0] + lMiddle4[0]/2.) - fradius4fifth[0]*(TMath::Sin(beta4fourthrad[0])) +
    fradius4fifth[0]*(TMath::Sin(beta4fourthrad[0] + angle4fifthrad[0])) +
    fradius4fifth[1]*(TMath::Sin(beta4fourthrad[0] + angle4fifthrad[0])),
    
    -(mLwater4[0] + lMiddle4[0]/2.) - fradius4fifth[0]*(TMath::Sin(beta4fourthrad[0])) +
    fradius4fifth[0]*(TMath::Sin(beta4fourthrad[0] + angle4fifthrad[0])) +
    fradius4fifth[1]*(TMath::Cos(TMath::Pi()/2 - beta4fourthrad[0] - angle4fifthrad[0])) +
    fradius4fifth[1]*(TMath::Cos(TMath::Pi()/2. - angle4fifthrad[1] + angle4fifthrad[0] + beta4fourthrad[0])) +
    fradius4fifth[2]*(TMath::Sin(angle4fifthrad[1] - angle4fifthrad[0] - beta4fourthrad[0])),
    
    -(mLwater4[0] + lMiddle4[0]/2.) - fradius4fifth[0]*(TMath::Sin(beta4fourthrad[0])) +
    fradius4fifth[0]*(TMath::Sin(beta4fourthrad[0] + angle4fifthrad[0])) +
    fradius4fifth[1]*(TMath::Cos(TMath::Pi()/2 - beta4fourthrad[0] - angle4fifthrad[0])) +
    fradius4fifth[1]*(TMath::Cos(TMath::Pi()/2. - angle4fifthrad[1] + angle4fifthrad[0] + beta4fourthrad[0])) +
    fradius4fifth[2]*(TMath::Sin(angle4fifthrad[1] - angle4fifthrad[0] - beta4fourthrad[0])) +
    (fradius4fifth[3] + fradius4fifth[2])*(TMath::Sin(angle4fifthrad[3]))
  };
  
  Double_t rotation4x[4] = {180., 180., 180., 180};
  Double_t rotation4y[4] = {90., 90., 90., 90};
  Double_t rotation4z[4] = {0., 180 - fangle4fifth[1]  , 0., 0.};
  
  for (Int_t i= 0; i<4; i++){
    waterTorus1 = gGeoManager->MakeTorus(Form("waterTorusone%d_D4_H%d", i,half), water, fradius4fifth[i], 0., mRWater, beta4fourth[i],  fangle4fifth[i]);
    waterTorus1->SetLineColor(kBlue);
    rotation = new TGeoRotation ("rotation", rotation4x[i], rotation4y[i], rotation4z[i]);
    transformation = new TGeoCombiTrans(translation4x[i], translation4y[i], translation4z[i], rotation);
    cooling->AddNode (waterTorus1, 7, transformation);
    rotation = new TGeoRotation ("rotation", rotation4x[i] , rotation4y[i] - 180, rotation4z[i]);
    transformation = new TGeoCombiTrans(translation4x[i], translation4y[i], - translation4z[i], rotation);
    cooling->AddNode (waterTorus1, 8, transformation);
  }
  
  TGeoVolume *waterTubeFive = gGeoManager->MakeTube(Form("waterTubeFive1_D4_H%d",half), water, 0., mRWater, -translation4z[3]);
  waterTubeFive->SetLineColor(kBlue);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(translation4x[3] + fradius4fifth[3], 0., 0., rotation);
  cooling->AddNode(waterTubeFive, 1, transformation);
  
  
  // **************************************** Pipe part ****************************************
  
  // ------------------- First and second pipe -------------------
  
  for(Int_t i=0; i<2; i++){
    
    // -------- Tube shape --------
    
    TGeoVolume *pipeTube1 = gGeoManager->MakeTube(Form("pipeTubeone%d_D4_H%d", i,half), pipe, mRWater, mRWater + mDRPipe, mLwater4[i]/2.);
    pipeTube1->SetLineColor(10);
    translation = new TGeoTranslation (mXposition4[i], 0., mLwater4[i]/2. + lMiddle4[i]/2.);
    cooling->AddNode (pipeTube1, 1, translation);
    translation = new TGeoTranslation (mXposition4[i], 0., -mLwater4[i]/2. - lMiddle4[i]/2.);
    cooling->AddNode (pipeTube1, 2, translation);
    
    TGeoVolume *pipeTube2 = gGeoManager->MakeTube(Form("pipeTubetwo%d_D4_H%d", i,half), pipe, mRWater, mRWater + mDRPipe, mLpartial4[i]/2.);
    pipeTube2->SetLineColor(10);
    rotation = new TGeoRotation ("rotation", -90., - fangle4[i], 0.);
    transformation = new TGeoCombiTrans( mXposition4[i]+fradius4[i]*(1-(TMath::Cos(anglerad[i])))+mLpartial4[i]*TMath::Sin(anglerad[i])/2., 0.,
                                        -mSupportXDimensions[4][0]/2. + mLwater4[i] + fradius4[i]*(TMath::Sin(anglerad[i])) +
                                        mLpartial4[i]*(TMath::Cos(anglerad[i]))/2., rotation);
    cooling->AddNode (pipeTube2, 3, transformation);
    rotation = new TGeoRotation ("rotation", -90.,  fangle4[i], 0.);
    transformation = new TGeoCombiTrans( mXposition4[i]+fradius4[i]*(1-(TMath::Cos(anglerad[i])))+mLpartial4[i]*TMath::Sin(anglerad[i])/2., 0.,
                                        mSupportXDimensions[4][0]/2. - mLwater4[i] - fradius4[i]*(TMath::Sin(anglerad[i])) -
                                        mLpartial4[i]*(TMath::Cos(anglerad[i]))/2. , rotation);
    cooling->AddNode (pipeTube2, 4, transformation);
    
    // -------- Torus shape --------
    
    TGeoVolume *pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorusone%d_D4_H%d", i,half), pipe, fradius4[i], mRWater, mRWater + mDRPipe, 0., fangle4[i]);
    pipeTorus1->SetLineColor(10);
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mXposition4[i] + fradius4[i], 0., -mSupportXDimensions[4][0]/2. + mLwater4[i], rotation);
    cooling->AddNode (pipeTorus1, 1, transformation);
    rotation = new TGeoRotation ("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mXposition4[i] + fradius4[i], 0., mSupportXDimensions[4][0]/2. - mLwater4[i], rotation);
    cooling->AddNode (pipeTorus1, 2, transformation);
    
    TGeoVolume *pipeTorus2 = gGeoManager->MakeTorus(Form("pipeTorustwo%d_D4_H%d", i,half), pipe, fradius4mid[i], mRWater, mRWater + mDRPipe, 180 - fangle4[i] ,2*fangle4[i]);
    pipeTorus2->SetLineColor(10);
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mXposition4[i]  + fradius4[i]*(1-(TMath::Cos(anglerad[i])))+mLpartial4[i]*TMath::Sin(anglerad[i]) -
                                        fradius4mid[i]*TMath::Cos(anglerad[i]), 0., 0., rotation);
    cooling->AddNode (pipeTorus2, 3, transformation);
    
  }
  
  // ------------------- Third pipe -------------------
  
  // -------- Tube shape --------
  
  TGeoVolume *pipeTube1 = gGeoManager->MakeTube(Form("pipeTubeone2_D4_H%d",half), pipe, mRWater, mRWater + mDRPipe, mLwater4[2]/2.);
  pipeTube1->SetLineColor(10);
  translation = new TGeoTranslation (mXposition4[2], 0., mLwater4[2]/2. + lMiddle4[2]/2.);
  cooling->AddNode (pipeTube1, 1, translation);
  translation = new TGeoTranslation (mXposition4[2], 0., -mLwater4[2]/2. - lMiddle4[2]/2.);
  cooling->AddNode (pipeTube1, 2, translation);
  
  TGeoVolume *pipeTube2 = gGeoManager->MakeTube(Form("pipeTubetwo2_D4_H%d",half), pipe, mRWater, mRWater + mDRPipe,  lMiddle4[2]/2. - 2.*fradius4[2]*TMath::Sin(anglerad[2]));
  pipeTube2->SetLineColor(10);
  translation = new TGeoTranslation (mXposition4[2] + 2.*fradius4[2]*(1-TMath::Cos(anglerad[2])), 0., 0.);
  cooling->AddNode (pipeTube2, 3, translation);
  
  // -------- Torus shape --------
  
  TGeoVolume *pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorusone2_D4_H%d",half), pipe, fradius4[2], mRWater, mRWater + mDRPipe, 0., fangle4[2]);
  pipeTorus1->SetLineColor(10);
  rotation = new TGeoRotation ("rotation", 180., 90., 0.);
  transformation = new TGeoCombiTrans(mXposition4[2] + fradius4[2], 0., -mSupportXDimensions[4][0]/2. + mLwater4[2], rotation);
  cooling->AddNode (pipeTorus1, 1, transformation);
  rotation = new TGeoRotation ("rotation", 180., -90., 180 - fangle4[2]);
  transformation = new TGeoCombiTrans(mXposition4[2] + fradius4[2] - 2*fradius4[2]*TMath::Cos(anglerad[2]), 0.,
                                      -mSupportXDimensions[4][0]/2. + mLwater4[2] + 2*fradius4[2]*TMath::Sin(anglerad[2]), rotation);
  cooling->AddNode (pipeTorus1, 2, transformation);
  rotation = new TGeoRotation ("rotation", 180., -90., 0.);
  transformation = new TGeoCombiTrans(mXposition4[2] + fradius4[2], 0., mSupportXDimensions[4][0]/2. - mLwater4[2], rotation);
  cooling->AddNode (pipeTorus1, 3, transformation);
  rotation = new TGeoRotation ("rotation", 180., 90., 180 - fangle4[2]);
  transformation = new TGeoCombiTrans(mXposition4[2] + fradius4[2] - 2*fradius4[2]*TMath::Cos(anglerad[2]), 0.,
                                      mSupportXDimensions[4][0]/2. - mLwater4[2] - 2*fradius4[2]*TMath::Sin(anglerad[2]), rotation);
  cooling->AddNode (pipeTorus1, 4, transformation);
  
  // ------------------- Fourth pipe -------------------
  
  pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorusone3_D4_H%d",half), pipe, fradius4[3], mRWater, mRWater + mDRPipe, 0., fangle4[3]);
  pipeTorus1->SetLineColor(10);
  rotation = new TGeoRotation ("rotation", 180., 90., 0.);
  transformation = new TGeoCombiTrans(mXposition4[3] + fradius4[3], 0., -mSupportXDimensions[4][0]/2., rotation);
  cooling->AddNode (pipeTorus1, 1, transformation);
  
  TGeoVolume *pipeTorus2 = gGeoManager->MakeTorus(Form("pipeTorustwo3_D4_H%d",half), pipe,  fradius4[4] , mRWater, mRWater + mDRPipe, 0., fangle4[4]);
  pipeTorus2->SetLineColor(10);
  rotation = new TGeoRotation ("rotation", 180., -90., 180 - fangle4[3]);
  transformation = new TGeoCombiTrans( mXposition4[3] + fradius4[3] - fradius4[3]*TMath::Cos(anglerad[3]) -
                                      fradius4[4]*TMath::Cos(anglerad[3]), 0., -mSupportXDimensions[4][0]/2. +
                                      fradius4[3]*TMath::Sin(anglerad[3]) + fradius4[4]*TMath::Sin(anglerad[3]), rotation);
  cooling->AddNode (pipeTorus2, 1, transformation);
  rotation = new TGeoRotation ("rotation", 180., -90., 0.);
  transformation = new TGeoCombiTrans(mXposition4[3] + fradius4[3], 0., mSupportXDimensions[4][0]/2. , rotation);
  cooling->AddNode (pipeTorus1, 2, transformation);
  rotation = new TGeoRotation ("rotation", 180., 90., 180 - fangle4[3]);
  transformation = new TGeoCombiTrans( mXposition4[3] + fradius4[3] - fradius4[3]*TMath::Cos(anglerad[3]) -
                                      fradius4[4]*TMath::Cos(anglerad[3]), 0., mSupportXDimensions[4][0]/2. -
                                      fradius4[3]*TMath::Sin(anglerad[3]) - fradius4[4]*TMath::Sin(anglerad[3]), rotation);
  cooling->AddNode (pipeTorus2, 2, transformation);
  
  TGeoVolume *pipeTorus3 =  gGeoManager->MakeTorus(Form("pipeTorusthree3_D4_H%d",half), pipe,  fradius4mid[2] , mRWater, mRWater + mDRPipe, -fangle4[5], 2.*fangle4[5]);
  pipeTorus3->SetLineColor(10);
  rotation = new TGeoRotation ("rotation", 0., 90., 0.);
  transformation = new TGeoCombiTrans( mXposition4[3] + fradius4[3] - fradius4[3]*TMath::Cos(anglerad[3]) -
                                      fradius4[4]*TMath::Cos(anglerad[3]) - ((fradius4mid[2] - fradius4[4])*TMath::Cos(anglerad[5])), 0., 0., rotation);
  cooling->AddNode (pipeTorus3, 1, transformation);
  
  // ------------------- Fifth pipe -------------------
  
  for(Int_t i= 0; i<4; i++){
    pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorusone%d_D4_H%d", i,half), pipe, fradius4fifth[i], mRWater, mRWater + mDRPipe, beta4fourth[i],  fangle4fifth[i]);
    pipeTorus1->SetLineColor(10);
    rotation = new TGeoRotation ("rotation", rotation4x[i], rotation4y[i], rotation4z[i]);
    transformation = new TGeoCombiTrans(translation4x[i], translation4y[i], translation4z[i], rotation);
    cooling->AddNode (pipeTorus1, 7, transformation);
    rotation = new TGeoRotation ("rotation", rotation4x[i] , rotation4y[i] - 180, rotation4z[i]);
    transformation = new TGeoCombiTrans(translation4x[i], translation4y[i], - translation4z[i], rotation);
    cooling->AddNode (pipeTorus1, 8, transformation);
  }
  
  TGeoVolume *pipeTubeFive = gGeoManager->MakeTube(Form("pipeTubeFive1_D4_H%d", half), pipe, mRWater, mRWater + mDRPipe, -translation4z[3]);
  pipeTubeFive->SetLineColor(10);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(translation4x[3] + fradius4fifth[3], 0., 0., rotation);
  cooling->AddNode(pipeTubeFive, 1, transformation);
  
  
//  if (half == kTop) {
//    rotation = new TGeoRotation ("rotation", 90., 90., 0.);
//    transformation = new TGeoCombiTrans(0., 0., fZPlan[disk] + deltaz/2. - fCarbonThickness - fRWater - fDRPipe, rotation);
//    fHalfDisk->AddNode(cooling, 1, transformation);
//    transformation = new TGeoCombiTrans(0., 0., fZPlan[disk] - deltaz/2. + fCarbonThickness + fRWater + fDRPipe, rotation);
//    fHalfDisk->AddNode(cooling, 2, transformation);
//  }
//  else if (half == kBottom) {
    rotation = new TGeoRotation ("rotation", -90., 90., 0.);
    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz/2. - mCarbonThickness - mRWater - mDRPipe, rotation);
    mHalfDisk->AddNode(cooling, 3, transformation);
    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz/2. + mCarbonThickness + mRWater + mDRPipe, rotation);
    mHalfDisk->AddNode(cooling, 4, transformation);
//  }
	
  // **************************************** Carbon Plates ****************************************
  
  TGeoVolumeAssembly *carbonPlate = new TGeoVolumeAssembly(Form("carbonPlate_D4_H%d",half));
  
  TGeoBBox *carbonBase4 = new TGeoBBox (Form("carbonBase4_D4_H%d",half),  (mSupportXDimensions[disk][0])/2., (mSupportYDimensions[disk][0])/2., mCarbonThickness);
  TGeoTranslation *t41= new TGeoTranslation ("t41",0., (mSupportYDimensions[disk][0])/2. + mHalfDiskGap, 0.);
  t41-> RegisterYourself();
  
  TGeoTubeSeg *holeCarbon4 = new TGeoTubeSeg(Form("holeCarbon4_D4_H%d",half), 0., mRMin[disk], mCarbonThickness + 0.000001, 0, 180.);
  TGeoTranslation *t42= new TGeoTranslation ("t42",0., - mHalfDiskGap , 0.);
  t42-> RegisterYourself();
  
  
  ///TGeoCompositeShape *cs4 = new TGeoCompositeShape(Form("Carbon4_D4_H%d",half),Form("(carbonBase4_D4_H%d:t41)-(holeCarbon4_D4_H%d:t42)",half,half));
  TGeoSubtraction    *carbonhole4 = new TGeoSubtraction(carbonBase4, holeCarbon4, t41, t42);
  TGeoCompositeShape *cs4 = new TGeoCompositeShape(Form("Carbon4_D4_H%d",half), carbonhole4);
  TGeoVolume *carbonBaseWithHole4 = new TGeoVolume(Form("carbonBaseWithHole_D4_H%d",half), cs4, carbon);

  carbonBaseWithHole4->SetLineColor(kGray+3);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  carbonPlate->AddNode(carbonBaseWithHole4, 0, new TGeoTranslation(0., 0., mZPlan[disk]));
  
  Double_t ty = mSupportYDimensions[disk][0];
  
  for (Int_t ipart=1; ipart<fnPart[disk]; ipart ++) {
    ty += mSupportYDimensions[disk][ipart]/2.;
    TGeoVolume *partCarbon = gGeoManager->MakeBox(Form("partCarbon_D4_H%d_%d", half,ipart), carbon, mSupportXDimensions[disk][ipart]/2.,
                                                  mSupportYDimensions[disk][ipart]/2., mCarbonThickness);
    partCarbon->SetLineColor(kGray+3);
    TGeoTranslation *t = new TGeoTranslation ("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    carbonPlate -> AddNode(partCarbon, ipart, t);
    ty += mSupportYDimensions[disk][ipart]/2.;
  }
  
//  if (half == kTop) {
//    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
//    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
//    fHalfDisk->AddNode(carbonPlate, 1, transformation);
//    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
//    fHalfDisk->AddNode(carbonPlate, 2, transformation);
//  }
//  else if (half == kBottom) {
    rotation = new TGeoRotation ("rotation", 180., 0., 0.);
    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
    mHalfDisk->AddNode(carbonPlate, 3, transformation);
    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
    mHalfDisk->AddNode(carbonPlate, 4, transformation);
//  }

	
  // **************************************** Rohacell Plate ****************************************
  
  TGeoVolumeAssembly *rohacellPlate = new TGeoVolumeAssembly(Form("rohacellPlate_D4_H%d",half));
  
  TGeoBBox *rohacellBase4 = new TGeoBBox (Form("rohacellBase4_D4_H%d",half),  (mSupportXDimensions[disk][0])/2., (mSupportYDimensions[disk][0])/2., mRohacellThickness);
  // TGeoTranslation *t3 = new TGeoTranslation ("t3",0., (fSupportYDimensions[disk][0])/2. + fHalfDiskGap , 0.);
  // t3 -> RegisterYourself();
  
  TGeoTubeSeg *holeRohacell4 = new TGeoTubeSeg(Form("holeRohacell4_D4_H%d",half), 0., mRMin[disk], mRohacellThickness + 0.000001, 0, 180.);
  // TGeoTranslation *t4= new TGeoTranslation ("t4", 0., - fHalfDiskGap , 0.);
  // t4-> RegisterYourself();
  
  ///cs4 = new TGeoCompositeShape(Form("rohacell_D4_H%d",half), Form("(rohacellBase4_D4_H%d:t41)-(holeRohacell4_D4_H%d:t42)",half,half));
  TGeoSubtraction    *rohacellhole4 = new TGeoSubtraction(rohacellBase4, holeRohacell4, t41, t42);
  TGeoCompositeShape *rh4 = new TGeoCompositeShape(Form("rohacellBase4_D4_H%d",half), rohacellhole4);
  TGeoVolume *rohacellBaseWithHole = new TGeoVolume(Form("rohacellBaseWithHole_D4_H%d",half), rh4, rohacell);

  rohacellBaseWithHole->SetLineColor(kGray);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation =  new TGeoCombiTrans(0., 0., 0., rotation);
  rohacellPlate -> AddNode(rohacellBaseWithHole, 0, new TGeoTranslation(0., 0., mZPlan[disk]));
  
  ty = mSupportYDimensions[disk][0];
  
  for (Int_t ipart=1; ipart<fnPart[disk]; ipart ++) {
    ty += mSupportYDimensions[disk][ipart]/2.;
    TGeoVolume *partRohacell = gGeoManager->MakeBox(Form("partRohacelli_D4_H%d_%d", half, ipart), rohacell, mSupportXDimensions[disk][ipart]/2.,
                                                    mSupportYDimensions[disk][ipart]/2., mRohacellThickness);
    partRohacell->SetLineColor(kGray);
    TGeoTranslation *t = new TGeoTranslation ("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    rohacellPlate -> AddNode(partRohacell, ipart, t);
    ty += mSupportYDimensions[disk][ipart]/2.;
  }
  
//  if (half == kTop) {
//    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
//    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
//    fHalfDisk->AddNode(rohacellPlate, 1, transformation);
//  }
//  if (half == kBottom) {
    rotation = new TGeoRotation ("rotation", 180., 0., 0.);
    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
    mHalfDisk->AddNode(rohacellPlate, 2, transformation);
//  }
	
}

//_____________________________________________________________________________
void HeatExchanger::InitParameters() 
{
  
  mHalfDiskRotation = new TGeoRotation**[Geometry::kNDisks];
  mHalfDiskTransformation = new TGeoCombiTrans**[Geometry::kNDisks];
  for (Int_t idisk = 0; idisk < Geometry::kNDisks; idisk++) {
    mHalfDiskRotation[idisk] = new TGeoRotation*[kNHalves];
    mHalfDiskTransformation[idisk] = new TGeoCombiTrans*[kNHalves];
    for (Int_t ihalf = 0; ihalf < kNHalves; ihalf++) {
      mHalfDiskRotation[idisk][ihalf] = new TGeoRotation(Form("rotation%d%d", idisk, ihalf), 0., 0., 0.);
      mHalfDiskTransformation[idisk][ihalf] = new TGeoCombiTrans(Form("transformation%d%d", idisk, ihalf), 0., 0., 0., mHalfDiskRotation[idisk][ihalf]);
    }
  } 

  mRohacellThickness = mHeatExchangerThickness/2. - 2.*mCarbonThickness - 2*(mRWater + mDRPipe);//thickness of Rohacell plate over 2
  printf("Rohacell thickness %f \n",mRohacellThickness);
  
  mHalfDiskGap = 0.2;
  
  fnPart[0] = 3;
  fnPart[1] = 3;
  fnPart[2] = 3;
  fnPart[3] = 5;
  fnPart[4] = 4;
  
  mRMin[0] = 2.35;
  mRMin[1] = 2.35;
  mRMin[2] = 2.35;
  mRMin[3] = 3.35;
  mRMin[4] = 3.75;
  
  // fZPlan[0] = 46;
  // fZPlan[1] = 49.3;
  // fZPlan[2] = 53.1;
  // fZPlan[3] = 68.7;
  // fZPlan[4] = 76.8;
  
  mZPlan[0] = 0;
  mZPlan[1] = 0;
  mZPlan[2] = 0;
  mZPlan[3] = 0;
  mZPlan[4] = 0;
    
  mSupportXDimensions= new Double_t*[Geometry::kNDisks];
  mSupportYDimensions= new Double_t*[Geometry::kNDisks];
  
  for(Int_t i = 0; i < Geometry::kNDisks; i++) {
    mSupportXDimensions[i]= new double[fnPart[i]];
    mSupportYDimensions[i]= new double[fnPart[i]];
  }
  
  mSupportXDimensions[0][0]=21.;  mSupportXDimensions[0][1]=14.8;   mSupportXDimensions[0][2]=4.4;
  mSupportXDimensions[1][0]=21.;  mSupportXDimensions[1][1]=14.8;   mSupportXDimensions[1][2]=4.4;
  mSupportXDimensions[2][0]=22.6; mSupportXDimensions[2][1]=16.1;   mSupportXDimensions[2][2]=9.3;
  mSupportXDimensions[3][0]=28.4; mSupportXDimensions[3][1]=22.9;   mSupportXDimensions[3][2]=18.5 ;mSupportXDimensions[3][3]=8.3; mSupportXDimensions[3][4]=4.9;
  mSupportXDimensions[4][0]=28.4; mSupportXDimensions[4][1]=25.204; mSupportXDimensions[4][2]=21.9 ;mSupportXDimensions[4][3]=15.1;
  
  mSupportYDimensions[0][0]=6.2;  mSupportYDimensions[0][1]=3.5;   mSupportYDimensions[0][2]=1.4;
  mSupportYDimensions[1][0]=6.2;  mSupportYDimensions[1][1]=3.5;   mSupportYDimensions[1][2]=1.4;
  mSupportYDimensions[2][0]=6.61; mSupportYDimensions[2][1]=3.01;  mSupportYDimensions[2][2]=1.83;
  mSupportYDimensions[3][0]=6.61; mSupportYDimensions[3][1]=3.01;  mSupportYDimensions[3][2]=3.01 ;mSupportYDimensions[3][3]=1.8; mSupportYDimensions[3][4]=1.15;
  mSupportYDimensions[4][0]=6.61; mSupportYDimensions[4][1]=3.01;  mSupportYDimensions[4][2]=3.01 ;mSupportYDimensions[4][3]=2.42;
  
  //Paramteters for disks 0, 1, 2
  
  mLWater = 6.759;
  
  mXPosition0[0] = 1.7;
  mXPosition0[1] = 4.61;
  mXPosition0[2] = 7.72;
  
  fangle0 = 44.6;
  fradius0 = 2.5;
  mLpartial0 = 1.;
  
  //Parameters for disk 3
  
  mLWater3[0] = 8.032;
  mLWater3[1] = 8.032;
  mLWater3[2] = 8.2;
  
  mXPosition3[0] = 1.7;
  mXPosition3[1] = 4.61;
  mXPosition3[2] = 5.5;
  mXPosition3[3] = 6.81;
  
  fangle3[0] = 41.3;
  fangle3[1] = 41.3;
  fangle3[2] = 28;
  
  fradius3[0] = 4.3;
  fradius3[1] = 4.3;
  fradius3[2] = 7.4;
  
  fangleThirdPipe3 = 15.;
  mLpartial3[0] = 2.3;
  mLpartial3[1] = 2.3;
  
  fradius3fourth[0] = 9.6;
  fradius3fourth[1] = 2.9;
  fradius3fourth[2] = 2.9;
  fradius3fourth[3] = 0.;
  
  fangle3fourth[0] = 40.8;
  fangle3fourth[1] = 50.;
  fangle3fourth[2] = 60.;
  fangle3fourth[3] =  8 + fangle3fourth[0] - fangle3fourth[1] + fangle3fourth[2];
  
  // Parameters for disk 4
  
  mLwater4[0] = 5.911;
  mLwater4[1] = 3.697;
  mLwater4[2] = 3.038;
  
  mXposition4[0] = 1.7;
  mXposition4[1] = 3.492;
  mXposition4[2] = 4.61;
  mXposition4[3] = 5.5;
  mXposition4[4] = 6.5;
  
  fangle4[0] = 35.5;
  fangle4[1] = 30.;
  fangle4[2] = 54.;
  fangle4[3] = 53.;
  fangle4[4] = 40;
  fangle4[5] = (fangle4[3] - fangle4[4]);
  
  fradius4[0] = 6.6;
  fradius4[1] = 7.2;
  fradius4[2] = 4.6;
  fradius4[3] = 6.2;
  fradius4[4] = 6.;
  
  mLpartial4[0] = 2.5;
  mLpartial4[1] = 3.6;
  
  fangle4fifth[0] = 64.;
  fangle4fifth[1] = 30.;
  fangle4fifth[2] = 27.;
  fangle4fifth[3] = fangle4fifth[0] - fangle4fifth[1] + fangle4fifth[2];
  
  fradius4fifth[0] = 2.7;
  fradius4fifth[1] = 5.;
  fradius4fifth[2] = 5.1;
  fradius4fifth[3] = 4.3;    
  
}


