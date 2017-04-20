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

using namespace o2::MFT;

/// \cond CLASSIMP
ClassImp(o2::MFT::HeatExchanger)
/// \endcond

//_____________________________________________________________________________
HeatExchanger::HeatExchanger() : 
TNamed(),
mHalfDisk(nullptr),
mHalfDiskRotation(nullptr),
mHalfDiskTransformation(nullptr),
mRWater(0.),
mDRPipe(0.),
mHeatExchangerThickness(0.),
mCarbonThickness(0.),
mHalfDiskGap(0.),
mRohacellThickness(0.),
mNPart(),
mRMin(),
mZPlan(),
mSupportXDimensions(),
mSupportYDimensions(),
mLWater(0.),
mXPosition0(),
mAngle0(0.),
mRadius0(0.),
mLpartial0(0.),
mLWater3(),
mXPosition3(),
mAngle3(),
mRadius3(),
mLpartial3(),
mRadius3fourth(),
mAngle3fourth(),
mBeta3fourth(),
mLWater4(), 
mXPosition4(), 
mAngle4(), 
mRadius4(), 
mLpartial4(), 
mAngle4fifth(), 
mRadius4fifth()
{

  mRWater = 0.1/2.;
  mDRPipe = 0.005;
  //mHeatExchangerThickness = 1.398; // to get a 13.4 mm thickness for the rohacell... but the water pipes are not inside it, then its density must be increased
  mHeatExchangerThickness = 1.4 + 2*Geometry::sRohacell; // mRohacell is used to link the rohacell thickness and the ladder positionning
  mCarbonThickness = (0.0290)/2.;  // half thickness of the carbon plate

  initParameters();

}

//_____________________________________________________________________________
HeatExchanger::HeatExchanger(Double_t rWater,
			     Double_t dRPipe,
			     Double_t heatExchangerThickness,
			     Double_t carbonThickness) : 
TNamed(), 
mHalfDisk(nullptr),
mHalfDiskRotation(nullptr),
mHalfDiskTransformation(nullptr),
mRWater(rWater),
mDRPipe(dRPipe),
mHeatExchangerThickness(heatExchangerThickness),
mCarbonThickness(carbonThickness),
mHalfDiskGap(0.),
mRohacellThickness(0.),
mNPart(),
mRMin(),
mZPlan(),
mSupportXDimensions(),
mSupportYDimensions(),
mLWater(0.),
mXPosition0(),
mAngle0(0.),
mRadius0(0.),
mLpartial0(0.),
mLWater3(),
mXPosition3(),
mAngle3(),
mRadius3(),
mLpartial3(),
mRadius3fourth(),
mAngle3fourth(),
mBeta3fourth(),
mLWater4(), 
mXPosition4(), 
mAngle4(), 
mRadius4(), 
mLpartial4(), 
mAngle4fifth(), 
mRadius4fifth()
{

  initParameters();
  
}

//_____________________________________________________________________________
TGeoVolumeAssembly* HeatExchanger::create(Int_t half, Int_t disk) 
{
	
  Info("Create",Form("Creating HeatExchanger_%d_%d", disk, half),0,0);
  
  mHalfDisk = new TGeoVolumeAssembly(Form("HeatExchanger_%d_%d", disk, half));
    switch (disk) {
      case 0: createHalfDisk0(half);
        break;
      case 1: createHalfDisk1(half);
        break;
      case 2: createHalfDisk2(half);
        break;
      case 3: createHalfDisk3(half);
        break;
      case 4: createHalfDisk4(half);
        break;
    }
  
  Info("Create",Form("... done HeatExchanger_%d_%d", disk, half),0,0);
  
  return mHalfDisk;
  
}

//_____________________________________________________________________________
void HeatExchanger::createManyfold(Int_t disk)
{

  TGeoCombiTrans  *transformation1 = nullptr;
  TGeoCombiTrans  *transformation2 = nullptr;
  TGeoRotation    *rotation       = nullptr;
  TGeoCombiTrans  *transformation = nullptr;

  // **************************************** Manyfolds right and left, fm  ****************************************
  
  Double_t deltay = 0.2;  // shift respect to the median plan of the MFT
  Double_t mfX = 2.2;     // width
  Double_t mfY = 6.8-0.1;     // height, decrease to avoid overlap with support, to be solved, fm
  Double_t mfZ = 1.7-0.85;     // thickness, decrease to avoid overlap with support, to be solved, fm
  Double_t fShift=0;
  if(disk==3 || disk==4)fShift=0.015; // to avoid overlap with the 2 curved water pipes on the 2 upstream chambers

  TGeoMedium *kMedPeek    = gGeoManager->GetMedium("MFT_PEEK$");
  
  auto *boxmanyfold = new TGeoBBox("boxmanyfold", mfX/2, mfY/2, mfZ/2);
  auto *remove = new TGeoBBox("remove", 0.45/2 + Geometry::sEpsilon, mfY/2 + Geometry::sEpsilon, 0.6/2 + Geometry::sEpsilon);
  auto *tL= new TGeoTranslation ("tL", mfX/2-0.45/2, 0., -mfZ/2+0.6/2);
  auto *boxManyFold = new TGeoSubtraction(boxmanyfold, remove, nullptr, tL);
  auto *BoxManyFold = new TGeoCompositeShape("BoxManyFold", boxManyFold);

  auto *tR= new TGeoTranslation ("tR", -mfX/2+0.45/2, 0., -mfZ/2+0.6/2);
  auto *boxManyFold1 = new TGeoSubtraction(BoxManyFold, remove, nullptr, tR);
  auto *BoxManyFold1 = new TGeoCompositeShape("BoxManyFold1", boxManyFold1);

  auto *MF01 = new TGeoVolume(Form("MF%d1",disk), BoxManyFold1, kMedPeek);

  rotation = new TGeoRotation ("rotation", 90., 90., 90.);
  transformation1 = new TGeoCombiTrans(mSupportXDimensions[disk][0]/2+mfZ/2+fShift, mfY/2+deltay, mZPlan[disk], rotation);

  mHalfDisk->AddNode(MF01, 1, transformation1);
    
  auto *MF02 = new TGeoVolume(Form("MF%d2",disk), BoxManyFold1, kMedPeek);
  transformation2 = new TGeoCombiTrans(mSupportXDimensions[disk][0]/2+mfZ/2+fShift, -mfY/2-deltay, mZPlan[disk], rotation);

  mHalfDisk->AddNode(MF02, 1, transformation2);
    
  // ********************************************************************************************************
}

//_____________________________________________________________________________
void HeatExchanger::createHalfDisk0(Int_t half) {
  
  Int_t disk = 0;
  
  if      (half == Top)    printf("Creating MFT heat exchanger for disk0 top\n");
  else if (half == Bottom) printf("Creating MFT heat exchanger for disk0 bottom\n");
  else     printf("No valid option for MFT heat exchanger on disk0\n");
  
  //mCarbon   = gGeoManager->GetMedium("MFT_Carbon$");
  mCarbon   = gGeoManager->GetMedium("MFT_CarbonFiber$");

  mWater    = gGeoManager->GetMedium("MFT_Water$");
  mRohacell = gGeoManager->GetMedium("MFT_Rohacell");
  mPipe     = gGeoManager->GetMedium("MFT_Polyimide");
  
  auto *cooling = new TGeoVolumeAssembly(Form("cooling_D0_H%d",half));
  
  Float_t lMiddle = mSupportXDimensions[disk][0] - 2.*mLWater;  // length of central part
  
  TGeoTranslation *translation    = nullptr;
  TGeoRotation    *rotation       = nullptr;
  TGeoCombiTrans  *transformation = nullptr;

  TGeoCombiTrans  *transformation1 = nullptr;
  TGeoCombiTrans  *transformation2 = nullptr;


  // **************************************** Water part ****************************************
  
  // ------------------- Tube shape -------------------
  
  TGeoVolume *waterTube1 = gGeoManager->MakeTube(Form("waterTube1_D0_H%d",half), mWater, 0., mRWater, mLWater/2.);
  waterTube1->SetLineColor(kBlue);
  
  for (Int_t itube=0; itube<3; itube++) {
    translation = new TGeoTranslation(mXPosition0[itube], 0.,  mLWater/2. + lMiddle/2.);
    cooling->AddNode (waterTube1, itube, translation);
    translation = new TGeoTranslation(mXPosition0[itube], 0., -mLWater/2. - lMiddle/2.);
    cooling->AddNode (waterTube1, itube+3, translation);
  }
  
  Double_t angle0rad = mAngle0*(TMath::DegToRad());
  
  TGeoVolume *waterTube2 = gGeoManager->MakeTube(Form("waterTube2_D0_H%d",half), mWater, 0., mRWater, mLpartial0/2.);
  waterTube2->SetLineColor(kBlue);
  
  for (Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", -90., -mAngle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + mRadius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., - lMiddle/2. +
                                        mRadius0*(TMath::Sin(angle0rad)) + (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (waterTube2, itube, transformation);
    rotation = new TGeoRotation ("rotation", -90., mAngle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + mRadius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., lMiddle/2. -
                                        mRadius0*(TMath::Sin(angle0rad)) - (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (waterTube2, itube+3, transformation);
  }
  
  // ------------------- Torus shape -------------------
  
  // Sides torus
  
  TGeoVolume *waterTorus1 = gGeoManager->MakeTorus(Form("waterTorus1_D0_H%d",half), mWater, mRadius0, 0., mRWater, 0., mAngle0);
  waterTorus1->SetLineColor(kBlue);
  Double_t radius0mid = (lMiddle - 2.*(mRadius0*(TMath::Sin(angle0rad)) + mLpartial0*(TMath::Cos(angle0rad))))/(2*(TMath::Sin(angle0rad)));
  
  for (Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mRadius0 + mXPosition0[itube], 0., - lMiddle/2., rotation);
    cooling->AddNode (waterTorus1, itube, transformation);
    rotation = new TGeoRotation ("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mRadius0 + mXPosition0[itube], 0.,  lMiddle/2., rotation);
    cooling->AddNode (waterTorus1, itube+3, transformation);
  }
  
  // Central Torus
  
  TGeoVolume *waterTorus2 = gGeoManager->MakeTorus(Form("waterTorus2_D0_H%d",half), mWater, radius0mid, 0., mRWater, - mAngle0 , 2.*mAngle0);
  waterTorus2->SetLineColor(kBlue);
  
  for(Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube]  + mRadius0*(1-(TMath::Cos(angle0rad))) + mLpartial0*TMath::Sin(angle0rad) - radius0mid*TMath::Cos(angle0rad), 0., 0., rotation);
    cooling->AddNode (waterTorus2, itube, transformation);
  }
  
  // **************************************** Pipe part ****************************************
  
  // ------------------- Tube shape -------------------
  
  TGeoVolume *pipeTube1 = gGeoManager->MakeTube(Form("pipeTube1_D0_H%d",half), mPipe, mRWater, mRWater + mDRPipe, mLWater/2.);
  pipeTube1->SetLineColor(10);
  
  for (Int_t itube=0; itube<3; itube++){
    translation = new TGeoTranslation(mXPosition0[itube], 0.,  mLWater/2. + lMiddle/2.);
    cooling->AddNode (pipeTube1, itube, translation);
    translation = new TGeoTranslation(mXPosition0[itube], 0., -mLWater/2. - lMiddle/2.);
    cooling->AddNode (pipeTube1, itube+3, translation);
  }
  
  TGeoVolume *pipeTube2 = gGeoManager->MakeTube(Form("pipeTube2_D0_H%d",half), mPipe, mRWater, mRWater + mDRPipe, mLpartial0/2.);
  waterTube2->SetLineColor(10);
  
  for(Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", -90., -mAngle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + mRadius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., - lMiddle/2. +
                                        mRadius0*(TMath::Sin(angle0rad)) + (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (pipeTube2, itube, transformation);
    
    rotation = new TGeoRotation ("rotation", -90., mAngle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + mRadius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., lMiddle/2. -
                                        mRadius0*(TMath::Sin(angle0rad)) - (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (pipeTube2, itube+3, transformation);
  }
  
  // ------------------- Torus shape -------------------
  
  // Sides Torus
  TGeoVolume *pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorus1_D0_H%d",half), mPipe, mRadius0, mRWater, mRWater + mDRPipe, 0., mAngle0);
  pipeTorus1->SetLineColor(10);
  
  for (Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mRadius0 + mXPosition0[itube], 0., - lMiddle/2., rotation);
    cooling->AddNode (pipeTorus1, itube, transformation);
    rotation = new TGeoRotation ("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mRadius0 + mXPosition0[itube], 0.,   lMiddle/2., rotation);
    cooling->AddNode (pipeTorus1, itube+3, transformation);
  }
  
  // Central Torus
  TGeoVolume *pipeTorus2 = gGeoManager->MakeTorus(Form("pipeTorus2_D0_H%d",half), mPipe, radius0mid, mRWater, mRWater + mDRPipe, - mAngle0 , 2.*mAngle0);
  pipeTorus2->SetLineColor(10);
  
  for(Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube]  + mRadius0*(1-(TMath::Cos(angle0rad))) + mLpartial0*TMath::Sin(angle0rad) - radius0mid*TMath::Cos(angle0rad), 0., 0., rotation);
    cooling->AddNode (pipeTorus2, itube, transformation);
  }
  
  Double_t deltaz = mHeatExchangerThickness - mCarbonThickness*2;  // distance between pair of carbon plates
  
//  if (half == Top) {
//    rotation = new TGeoRotation ("rotation", 90., 90., 0.);
//    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz/2. - mCarbonThickness - mRWater - mDRPipe, rotation);
//    mHalfDisk->AddNode(cooling, 1, transformation);
//    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz/2. + mCarbonThickness + mRWater + mDRPipe, rotation);
//    mHalfDisk->AddNode(cooling, 2, transformation);
//  }
//  else if (half == Bottom) {
    rotation = new TGeoRotation ("rotation", -90., 90., 0.);
    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz/2. - mCarbonThickness - mRWater - mDRPipe, rotation);
    mHalfDisk->AddNode(cooling, 3, transformation);
    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz/2. + mCarbonThickness + mRWater + mDRPipe, rotation);
    mHalfDisk->AddNode(cooling, 4, transformation);
//  }
  



  // **************************************** Carbon Plates ****************************************
  
  auto *carbonPlate = new TGeoVolumeAssembly(Form("carbonPlate_D0_H%d",half));
  
  auto *carbonBase0 = new TGeoBBox (Form("carbonBase0_D0_H%d",half),  (mSupportXDimensions[disk][0])/2., (mSupportYDimensions[disk][0])/2., mCarbonThickness);
  auto *t01= new TGeoTranslation ("t01",0., (mSupportYDimensions[disk][0])/2. + mHalfDiskGap , 0.);
  t01-> RegisterYourself();
  
  auto *holeCarbon0 = new TGeoTubeSeg(Form("holeCarbon0_D0_H%d",half), 0., mRMin[disk], mCarbonThickness + 0.000001, 0, 180.);
  auto *t02= new TGeoTranslation ("t02",0., - mHalfDiskGap , 0.);
  t02-> RegisterYourself();
  
  ///TGeoCompositeShape *cs0 = new TGeoCompositeShape(Form("cs0_D0_H%d",half), Form("(carbonBase0_D0_H%d:t01)-(holeCarbon0_D0_H%d:t02)",half,half));
  auto    *carbonhole0 = new TGeoSubtraction(carbonBase0, holeCarbon0, t01, t02);
  auto *ch0 = new TGeoCompositeShape(Form("Carbon0_D0_H%d",half), carbonhole0);
  auto *carbonBaseWithHole0 = new TGeoVolume(Form("carbonBaseWithHole_D0_H%d", half), ch0, mCarbon);
  

  carbonBaseWithHole0->SetLineColor(kGray+3);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  carbonPlate->AddNode(carbonBaseWithHole0, 0, new TGeoTranslation(0., 0., mZPlan[disk]));
  
  Double_t ty = mSupportYDimensions[disk][0];
  
  for (Int_t ipart=1; ipart<mNPart[disk]; ipart ++) {
    ty += mSupportYDimensions[disk][ipart]/2.;
    TGeoVolume *partCarbon = gGeoManager->MakeBox(Form("partCarbon_D0_H%d_%d", half,ipart), mCarbon, mSupportXDimensions[disk][ipart]/2., mSupportYDimensions[disk][ipart]/2., mCarbonThickness);
    partCarbon->SetLineColor(kGray+3);
    auto *t = new TGeoTranslation ("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    carbonPlate -> AddNode(partCarbon, ipart, t);
    ty += mSupportYDimensions[disk][ipart]/2.;
  }
  
//  if (half == Top) {
//    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
//    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
//    mHalfDisk->AddNode(carbonPlate, 1, transformation);
//    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
//    mHalfDisk->AddNode(carbonPlate, 2, transformation);
//  }
//  else if (half == Bottom) {
    rotation = new TGeoRotation ("rotation", 180., 0., 0.);
    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
    mHalfDisk->AddNode(carbonPlate, 3, transformation);
    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
    mHalfDisk->AddNode(carbonPlate, 4, transformation);
//  }



  // **************************************** Rohacell Plate ****************************************
  
  auto *rohacellPlate = new TGeoVolumeAssembly(Form("rohacellPlate_D0_H%d",half));
  
  auto *rohacellBase0 = new TGeoBBox (Form("rohacellBase0_D0_H%d",half),  (mSupportXDimensions[disk][0])/2., (mSupportYDimensions[disk][0])/2., mRohacellThickness);
  // TGeoTranslation *t3 = new TGeoTranslation ("t3",0., (fSupportYDimensions[disk][0])/2. + fHalfDiskGap , 0.);
  // t3 -> RegisterYourself();
  
  auto *holeRohacell0 = new TGeoTubeSeg(Form("holeRohacell0_D0_H%d",half), 0., mRMin[disk], mRohacellThickness + 0.000001, 0, 180.);
  // TGeoTranslation *t4= new TGeoTranslation ("t4", 0., - fHalfDiskGap , 0.);
  // t4-> RegisterYourself();
  
  ///cs0 = new TGeoCompositeShape("cs0", Form("(rohacellBase0_D0_H%d:t01)-(holeRohacell0_D0_H%d:t02)",half,half));
  auto    *rohacellhole0 = new TGeoSubtraction(rohacellBase0, holeRohacell0, t01, t02);
  auto *rh0 = new TGeoCompositeShape(Form("rohacellBase0_D0_H%d",half), rohacellhole0);
  auto *rohacellBaseWithHole = new TGeoVolume(Form("rohacellBaseWithHole_D0_H%d",half), rh0, mRohacell);


  rohacellBaseWithHole->SetLineColor(kGray);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation =  new TGeoCombiTrans(0., 0., 0., rotation);
  rohacellPlate -> AddNode(rohacellBaseWithHole, 0, new TGeoTranslation(0., 0., mZPlan[disk]));
  
  ty = mSupportYDimensions[disk][0];
  
  for (Int_t ipart=1; ipart<mNPart[disk]; ipart ++) {
    ty += mSupportYDimensions[disk][ipart]/2.;
    TGeoVolume *partRohacell = gGeoManager->MakeBox(Form("partRohacelli_D0_H%d_%d", half,ipart), mRohacell, mSupportXDimensions[disk][ipart]/2., mSupportYDimensions[disk][ipart]/2., mRohacellThickness);
    partRohacell->SetLineColor(kGray);
    auto *t = new TGeoTranslation ("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    rohacellPlate -> AddNode(partRohacell, ipart, t);
    ty += mSupportYDimensions[disk][ipart]/2.;
  }
  
//  if (half == Top) {
//    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
//    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
//    mHalfDisk->AddNode(rohacellPlate, 1, transformation);
//  }
//  if (half == Bottom) {
    rotation = new TGeoRotation ("rotation", 180., 0., 0.);
    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
    mHalfDisk->AddNode(rohacellPlate, 1, transformation);
//  }
 
    createManyfold(disk);
 
}

//_____________________________________________________________________________
void HeatExchanger::createHalfDisk1(Int_t half) {
	
  Int_t disk = 1;
  
  if      (half == Top)    printf("Creating MFT heat exchanger for disk1 top\n");
  else if (half == Bottom) printf("Creating MFT heat exchanger for disk1 bottom\n");
  else     printf("No valid option for MFT heat exchanger on disk1\n");
  
  //mCarbon   = gGeoManager->GetMedium("MFT_Carbon$");
  mCarbon   = gGeoManager->GetMedium("MFT_CarbonFiber$");

  mWater    = gGeoManager->GetMedium("MFT_Water$");
  mRohacell = gGeoManager->GetMedium("MFT_Rohacell");
  mPipe     = gGeoManager->GetMedium("MFT_Polyimide");
    
  auto *cooling = new TGeoVolumeAssembly(Form("cooling_D1_H%d",half));
  
  Float_t lMiddle = mSupportXDimensions[disk][0] - 2.*mLWater;  // length of central part
  
  TGeoTranslation *translation    = nullptr;
  TGeoRotation    *rotation       = nullptr;
  TGeoCombiTrans  *transformation = nullptr;
  
  
  // **************************************** Water part ****************************************
  
  // ------------------- Tube shape -------------------
  
  TGeoVolume *waterTube1 = gGeoManager->MakeTube(Form("waterTube1_D1_H%d",half), mWater, 0., mRWater, mLWater/2.);
  waterTube1->SetLineColor(kBlue);
  
  for (Int_t itube=0; itube<3; itube++) {
    translation = new TGeoTranslation(mXPosition0[itube], 0.,  mLWater/2. + lMiddle/2.);
    cooling->AddNode (waterTube1, itube, translation);
    translation = new TGeoTranslation(mXPosition0[itube], 0., -mLWater/2. - lMiddle/2.);
    cooling->AddNode (waterTube1, itube+3, translation);
  }
  
  Double_t angle0rad = mAngle0*(TMath::DegToRad());
  
  TGeoVolume *waterTube2 = gGeoManager->MakeTube(Form("waterTube2_D1_H%d",half), mWater, 0., mRWater, mLpartial0/2.);
  waterTube2->SetLineColor(kBlue);
  
  for (Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", -90., -mAngle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + mRadius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., - lMiddle/2. +
                                        mRadius0*(TMath::Sin(angle0rad)) + (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (waterTube2, itube, transformation);
    rotation = new TGeoRotation ("rotation", -90., mAngle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + mRadius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., lMiddle/2. -
                                        mRadius0*(TMath::Sin(angle0rad)) - (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (waterTube2, itube+3, transformation);
  }
  
  // ------------------- Torus shape -------------------
  
  // Sides torus
  
  TGeoVolume *waterTorus1 = gGeoManager->MakeTorus(Form("waterTorus1_D1_H%d",half), mWater, mRadius0, 0., mRWater, 0., mAngle0);
  waterTorus1->SetLineColor(kBlue);
  Double_t radius0mid = (lMiddle - 2.*(mRadius0*(TMath::Sin(angle0rad)) + mLpartial0*(TMath::Cos(angle0rad))))/(2*(TMath::Sin(angle0rad)));
  
  for (Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mRadius0 + mXPosition0[itube], 0., - lMiddle/2., rotation);
    cooling->AddNode (waterTorus1, itube, transformation);
    rotation = new TGeoRotation ("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mRadius0 + mXPosition0[itube], 0.,  lMiddle/2., rotation);
    cooling->AddNode (waterTorus1, itube+3, transformation);
  }
  
  // Central Torus
  
  TGeoVolume *waterTorus2 = gGeoManager->MakeTorus(Form("waterTorus2_D1_H%d",half), mWater, radius0mid, 0., mRWater, - mAngle0 , 2.*mAngle0);
  waterTorus2->SetLineColor(kBlue);
  
  for(Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube]  + mRadius0*(1-(TMath::Cos(angle0rad))) + mLpartial0*TMath::Sin(angle0rad) - radius0mid*TMath::Cos(angle0rad), 0., 0., rotation);
    cooling->AddNode (waterTorus2, itube, transformation);
  }
  
  // **************************************** Pipe part ****************************************
  
  // ------------------- Tube shape -------------------
  
  TGeoVolume *pipeTube1 = gGeoManager->MakeTube(Form("pipeTube1_D1_H%d",half), mPipe, mRWater, mRWater + mDRPipe, mLWater/2.);
  pipeTube1->SetLineColor(10);
  
  for (Int_t itube=0; itube<3; itube++){
    translation = new TGeoTranslation(mXPosition0[itube], 0.,  mLWater/2. + lMiddle/2.);
    cooling->AddNode (pipeTube1, itube, translation);
    translation = new TGeoTranslation(mXPosition0[itube], 0., -mLWater/2. - lMiddle/2.);
    cooling->AddNode (pipeTube1, itube+3, translation);
  }

  TGeoVolume *pipeTube2 = gGeoManager->MakeTube(Form("pipeTube2_D1_H%d",half), mPipe, mRWater, mRWater + mDRPipe, mLpartial0/2.);
  waterTube2->SetLineColor(10);
  
  for(Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", -90., -mAngle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + mRadius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., - lMiddle/2. +
                                        mRadius0*(TMath::Sin(angle0rad)) + (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (pipeTube2, itube, transformation);
    
    rotation = new TGeoRotation ("rotation", -90., mAngle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + mRadius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., lMiddle/2. -
                                        mRadius0*(TMath::Sin(angle0rad)) - (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (pipeTube2, itube+3, transformation);
  }
  
  // ------------------- Torus shape -------------------
  
  // Sides Torus
  TGeoVolume *pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorus1_D1_H%d",half), mPipe, mRadius0, mRWater, mRWater + mDRPipe, 0., mAngle0);
  pipeTorus1->SetLineColor(10);
  
  for (Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mRadius0 + mXPosition0[itube], 0., - lMiddle/2., rotation);
    cooling->AddNode (pipeTorus1, itube, transformation);
    rotation = new TGeoRotation ("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mRadius0 + mXPosition0[itube], 0.,   lMiddle/2., rotation);
    cooling->AddNode (pipeTorus1, itube+3, transformation);
  }
  
  // Central Torus
  TGeoVolume *pipeTorus2 = gGeoManager->MakeTorus(Form("pipeTorus2_D1_H%d",half), mPipe, radius0mid, mRWater, mRWater + mDRPipe, - mAngle0 , 2.*mAngle0);
  pipeTorus2->SetLineColor(10);
  
  for(Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube]  + mRadius0*(1-(TMath::Cos(angle0rad))) + mLpartial0*TMath::Sin(angle0rad) - radius0mid*TMath::Cos(angle0rad), 0., 0., rotation);
    cooling->AddNode (pipeTorus2, itube, transformation);
  }
  
  Double_t deltaz = mHeatExchangerThickness - mCarbonThickness*2;  // distance between pair of carbon plates
  
//  if (half == Top) {
//    rotation = new TGeoRotation ("rotation", 90., 90., 0.);
//    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz/2. - mCarbonThickness - mRWater - mDRPipe, rotation);
//    mHalfDisk->AddNode(cooling, 1, transformation);
//    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz/2. + mCarbonThickness + mRWater + mDRPipe, rotation);
//    mHalfDisk->AddNode(cooling, 2, transformation);
//  }
//  else if (half == Bottom) {
    rotation = new TGeoRotation ("rotation", -90., 90., 0.);
    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz/2. - mCarbonThickness - mRWater - mDRPipe, rotation);
    mHalfDisk->AddNode(cooling, 0, transformation);
    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz/2. + mCarbonThickness + mRWater + mDRPipe, rotation);
    mHalfDisk->AddNode(cooling, 1, transformation);
//  }
  
  
  // **************************************** Carbon Plates ****************************************
  
   

  auto *carbonPlate = new TGeoVolumeAssembly(Form("carbonPlate_D1_H%d",half));
  
  auto *carbonBase1 = new TGeoBBox (Form("carbonBase1_D1_H%d",half),  (mSupportXDimensions[disk][0])/2., (mSupportYDimensions[disk][0])/2., mCarbonThickness);
  auto *t11= new TGeoTranslation ("t11",0., (mSupportYDimensions[disk][0])/2. + mHalfDiskGap , 0.);
  t11-> RegisterYourself();
  
  auto *holeCarbon1 = new TGeoTubeSeg(Form("holeCarbon1_D1_H%d",half), 0., mRMin[disk], mCarbonThickness + 0.000001, 0, 180.);
  auto *t12= new TGeoTranslation ("t12",0., - mHalfDiskGap , 0.);
  t12-> RegisterYourself();
  
  
  ////TGeoCompositeShape *cs1 = new TGeoCompositeShape(Form("Carbon1_D1_H%d",half), Form("(carbonBase1_D1_H%d:t11)-(holeCarbon1_D1_H%d:t12)",half,half));
  auto    *carbonhole1 = new TGeoSubtraction(carbonBase1, holeCarbon1, t11, t12);
  auto *ch1 = new TGeoCompositeShape(Form("Carbon1_D1_H%d",half), carbonhole1);
  auto *carbonBaseWithHole1 = new TGeoVolume(Form("carbonBaseWithHole_D1_H%d",half), ch1, mCarbon);


  carbonBaseWithHole1->SetLineColor(kGray+3);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  carbonPlate->AddNode(carbonBaseWithHole1, 0, new TGeoTranslation(0., 0., mZPlan[disk]));
  
  Double_t ty = mSupportYDimensions[disk][0];
  
  for (Int_t ipart=1; ipart<mNPart[disk]; ipart ++) {
    ty += mSupportYDimensions[disk][ipart]/2.;
    TGeoVolume *partCarbon = gGeoManager->MakeBox(Form("partCarbon_D1_H%d_%d", half,ipart), mCarbon, mSupportXDimensions[disk][ipart]/2.,
                                                  mSupportYDimensions[disk][ipart]/2., mCarbonThickness);
    partCarbon->SetLineColor(kGray+3);
    auto *t = new TGeoTranslation ("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    carbonPlate -> AddNode(partCarbon, ipart, t);
    ty += mSupportYDimensions[disk][ipart]/2.;
  }
  
//  if (half == Top) {
//    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
//    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
//    mHalfDisk->AddNode(carbonPlate, 1, transformation);
//    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
//    mHalfDisk->AddNode(carbonPlate, 2, transformation);
//  }
//  else if (half == Bottom) {
    rotation = new TGeoRotation ("rotation", 180., 0., 0.);
    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
    mHalfDisk->AddNode(carbonPlate, 0, transformation);
    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
    mHalfDisk->AddNode(carbonPlate, 1, transformation);
//  }

  
  // **************************************** Rohacell Plate ****************************************
  
  auto *rohacellPlate = new TGeoVolumeAssembly(Form("rohacellPlate_D1_H%d",half));
  
  auto *rohacellBase1 = new TGeoBBox ("rohacellBase1",  (mSupportXDimensions[disk][0])/2., (mSupportYDimensions[disk][0])/2., mRohacellThickness);
  // TGeoTranslation *t3 = new TGeoTranslation ("t3",0., (fSupportYDimensions[disk][0])/2. + fHalfDiskGap , 0.);
  // t3 -> RegisterYourself();
  
  auto *holeRohacell1 = new TGeoTubeSeg("holeRohacell1", 0., mRMin[disk], mRohacellThickness + 0.000001, 0, 180.);
  // TGeoTranslation *t4= new TGeoTranslation ("t4", 0., - fHalfDiskGap , 0.);
  // t4-> RegisterYourself();
  
  //////cs1 = new TGeoCompositeShape(Form("rohacell_D1_H%d",half), "(rohacellBase1:t11)-(holeRohacell1:t12)");
  auto    *rohacellhole1 = new TGeoSubtraction(rohacellBase1, holeRohacell1, t11, t12);
  auto *rh1 = new TGeoCompositeShape(Form("rohacellBase1_D1_H%d",half), rohacellhole1);
  auto *rohacellBaseWithHole = new TGeoVolume(Form("rohacellBaseWithHole_D1_H%d",half), rh1, mRohacell);


  rohacellBaseWithHole->SetLineColor(kGray);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation =  new TGeoCombiTrans(0., 0., 0., rotation);
  rohacellPlate -> AddNode(rohacellBaseWithHole, 0, new TGeoTranslation(0., 0., mZPlan[disk]));
  
  ty = mSupportYDimensions[disk][0];
  
  for (Int_t ipart=1; ipart<mNPart[disk]; ipart ++) {
    ty += mSupportYDimensions[disk][ipart]/2.;
    TGeoVolume *partRohacell = gGeoManager->MakeBox(Form("partRohacelli_D1_H%d_%d",half, ipart), mRohacell, mSupportXDimensions[disk][ipart]/2.,
                                                    mSupportYDimensions[disk][ipart]/2., mRohacellThickness);
    partRohacell->SetLineColor(kGray);
    auto *t = new TGeoTranslation ("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    rohacellPlate -> AddNode(partRohacell, ipart, t);
    ty += mSupportYDimensions[disk][ipart]/2.;
  }
  
//  if (half == Top) {
//    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
//    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
//    mHalfDisk->AddNode(rohacellPlate, 1, transformation);
//  }
//  if (half == Bottom) {
    rotation = new TGeoRotation ("rotation", 180., 0., 0.);
    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
    mHalfDisk->AddNode(rohacellPlate, 2, transformation);
//  }
  
    createManyfold(disk);

}

//_____________________________________________________________________________
void HeatExchanger::createHalfDisk2(Int_t half) {
  
  Int_t disk = 2;
  
  if      (half == Top)    printf("Creating MFT heat exchanger for disk2 top\n");
  else if (half == Bottom) printf("Creating MFT heat exchanger for disk2 bottom\n");
  else     printf("No valid option for MFT heat exchanger on disk2\n");
  
  
  //mCarbon   = gGeoManager->GetMedium("MFT_Carbon$");
  mCarbon   = gGeoManager->GetMedium("MFT_CarbonFiber$");

  mWater    = gGeoManager->GetMedium("MFT_Water$");
  mRohacell = gGeoManager->GetMedium("MFT_Rohacell");
  mPipe     = gGeoManager->GetMedium("MFT_Polyimide");
 
  auto *cooling = new TGeoVolumeAssembly(Form("cooling_D2_H%d",half));
  
  Float_t lMiddle = mSupportXDimensions[disk][0] - 2.*mLWater;  // length of central part
  
  TGeoTranslation *translation    = nullptr;
  TGeoRotation    *rotation       = nullptr;
  TGeoCombiTrans  *transformation = nullptr;
  
  // **************************************** Water part ****************************************
  
  // ------------------- Tube shape -------------------
  
  TGeoVolume *waterTube1 = gGeoManager->MakeTube(Form("waterTube1_D2_H%d",half), mWater, 0., mRWater, mLWater/2.);
  waterTube1->SetLineColor(kBlue);
  
  for (Int_t itube=0; itube<3; itube++) {
    translation = new TGeoTranslation(mXPosition0[itube], 0.,  mLWater/2. + lMiddle/2.);
    cooling->AddNode (waterTube1, itube, translation);
    translation = new TGeoTranslation(mXPosition0[itube], 0., -mLWater/2. - lMiddle/2.);
    cooling->AddNode (waterTube1, itube+3, translation);
  }
  
  Double_t angle0rad = mAngle0*(TMath::DegToRad());
  
  TGeoVolume *waterTube2 = gGeoManager->MakeTube(Form("waterTube2_D2_H%d",half), mWater, 0., mRWater, mLpartial0/2.);
  waterTube2->SetLineColor(kBlue);
  
  for (Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", -90., -mAngle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + mRadius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., - lMiddle/2. +
                                        mRadius0*(TMath::Sin(angle0rad)) + (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (waterTube2, itube, transformation);
    rotation = new TGeoRotation ("rotation", -90., mAngle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + mRadius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., lMiddle/2. -
                                        mRadius0*(TMath::Sin(angle0rad)) - (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (waterTube2, itube+3, transformation);
  }
  
  // ------------------- Torus shape -------------------
  
  // Sides torus
  
  TGeoVolume *waterTorus1 = gGeoManager->MakeTorus(Form("waterTorus1_D2_H%d",half), mWater, mRadius0, 0., mRWater, 0., mAngle0);
  waterTorus1->SetLineColor(kBlue);
  Double_t radius0mid = (lMiddle - 2.*(mRadius0*(TMath::Sin(angle0rad)) + mLpartial0*(TMath::Cos(angle0rad))))/(2*(TMath::Sin(angle0rad)));
  
  for (Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mRadius0 + mXPosition0[itube], 0., - lMiddle/2., rotation);
    cooling->AddNode (waterTorus1, itube, transformation);
    rotation = new TGeoRotation ("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mRadius0 + mXPosition0[itube], 0.,  lMiddle/2., rotation);
    cooling->AddNode (waterTorus1, itube+3, transformation);
  }
  
  // Central Torus
  
  TGeoVolume *waterTorus2 = gGeoManager->MakeTorus(Form("waterTorus2_D2_H%d",half), mWater, radius0mid, 0., mRWater, - mAngle0 , 2.*mAngle0);
  waterTorus2->SetLineColor(kBlue);
  
  for(Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube]  + mRadius0*(1-(TMath::Cos(angle0rad))) +
                                        mLpartial0*TMath::Sin(angle0rad) - radius0mid*TMath::Cos(angle0rad), 0., 0., rotation);
    cooling->AddNode (waterTorus2, itube, transformation);
  }
  
  // **************************************** Pipe part ****************************************
  
  // ------------------- Tube shape -------------------
  
  TGeoVolume *pipeTube1 = gGeoManager->MakeTube(Form("pipeTube1_D2_H%d",half), mPipe, mRWater, mRWater + mDRPipe, mLWater/2.);
  pipeTube1->SetLineColor(10);
  
  for (Int_t itube=0; itube<3; itube++){
    translation = new TGeoTranslation(mXPosition0[itube], 0.,  mLWater/2. + lMiddle/2.);
    cooling->AddNode (pipeTube1, itube, translation);
    translation = new TGeoTranslation(mXPosition0[itube], 0., -mLWater/2. - lMiddle/2.);
    cooling->AddNode (pipeTube1, itube+3, translation);
  }
  
  TGeoVolume *pipeTube2 = gGeoManager->MakeTube(Form("pipeTube2_D2_H%d",half), mPipe, mRWater, mRWater + mDRPipe, mLpartial0/2.);
  waterTube2->SetLineColor(10);
  
  for(Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", -90., -mAngle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + mRadius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., - lMiddle/2. +
                                        mRadius0*(TMath::Sin(angle0rad)) + (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (pipeTube2, itube, transformation);
    
    rotation = new TGeoRotation ("rotation", -90., mAngle0, 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube] + mRadius0*(1-(TMath::Cos(angle0rad))) +
                                        (mLpartial0/2.)*(TMath::Sin(angle0rad)), 0., lMiddle/2. -
                                        mRadius0*(TMath::Sin(angle0rad)) - (mLpartial0/2.)*TMath::Cos(angle0rad), rotation);
    cooling->AddNode (pipeTube2, itube+3, transformation);
  }
  
  // ------------------- Torus shape -------------------
  
  // Sides Torus
  TGeoVolume *pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorus1_D2_H%d",half), mPipe, mRadius0, mRWater, mRWater + mDRPipe, 0., mAngle0);
  pipeTorus1->SetLineColor(10);
  
  for (Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mRadius0 + mXPosition0[itube], 0., - lMiddle/2., rotation);
    cooling->AddNode (pipeTorus1, itube, transformation);
    rotation = new TGeoRotation ("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mRadius0 + mXPosition0[itube], 0.,   lMiddle/2., rotation);
    cooling->AddNode (pipeTorus1, itube+3, transformation);
  }
  
  // Central Torus
  TGeoVolume *pipeTorus2 = gGeoManager->MakeTorus(Form("pipeTorus2_D2_H%d",half), mPipe, radius0mid, mRWater, mRWater + mDRPipe, - mAngle0 , 2.*mAngle0);
  pipeTorus2->SetLineColor(10);
  
  for(Int_t itube=0; itube<3; itube++) {
    rotation = new TGeoRotation ("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(mXPosition0[itube]  + mRadius0*(1-(TMath::Cos(angle0rad))) +
                                        mLpartial0*TMath::Sin(angle0rad) - radius0mid*TMath::Cos(angle0rad), 0., 0., rotation);
    cooling->AddNode (pipeTorus2, itube, transformation);
  }
  
  Double_t deltaz = mHeatExchangerThickness - mCarbonThickness*2;  // distance between pair of carbon plates
  
//  if (half == Top) {
//    rotation = new TGeoRotation ("rotation", 90., 90., 0.);
//    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz/2. - mCarbonThickness - mRWater - mDRPipe, rotation);
//    mHalfDisk->AddNode(cooling, 1, transformation);
//    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz/2. + mCarbonThickness + mRWater + mDRPipe, rotation);
//    mHalfDisk->AddNode(cooling, 2, transformation);
//  }
//  else if (half == Bottom) {
    rotation = new TGeoRotation ("rotation", -90., 90., 0.);
    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz/2. - mCarbonThickness - mRWater - mDRPipe, rotation);
    mHalfDisk->AddNode(cooling, 3, transformation);
    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz/2. + mCarbonThickness + mRWater + mDRPipe, rotation);
    mHalfDisk->AddNode(cooling, 4, transformation);
//  }
  
  // **************************************** Carbon Plates ****************************************
  
  auto *carbonPlate = new TGeoVolumeAssembly(Form("carbonPlate_D2_H%d",half));
  
  auto *carbonBase2 = new TGeoBBox (Form("carbonBase2_D2_H%d",half),  (mSupportXDimensions[disk][0])/2., (mSupportYDimensions[disk][0])/2., mCarbonThickness);
  auto *t21= new TGeoTranslation ("t21",0., (mSupportYDimensions[disk][0])/2. + mHalfDiskGap , 0.);
  t21-> RegisterYourself();
  
  auto *holeCarbon2 = new TGeoTubeSeg(Form("holeCarbon2_D2_H%d",half), 0., mRMin[disk], mCarbonThickness + 0.000001, 0, 180.);
  auto *t22= new TGeoTranslation ("t22",0., - mHalfDiskGap , 0.);
  t22-> RegisterYourself();
 

  ////TGeoCompositeShape *cs2 = new TGeoCompositeShape(Form("carbon2_D2_H%d",half),Form("(carbonBase2_D2_H%d:t21)-(holeCarbon2_D2_H%d:t22)",half,half));
  auto    *carbonhole2 = new TGeoSubtraction(carbonBase2, holeCarbon2, t21, t22);
  auto *cs2 = new TGeoCompositeShape(Form("Carbon2_D2_H%d",half), carbonhole2);
  auto *carbonBaseWithHole2 = new TGeoVolume(Form("carbonBaseWithHole_D2_H%d", half), cs2, mCarbon);

  carbonBaseWithHole2->SetLineColor(kGray+3);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  carbonPlate->AddNode(carbonBaseWithHole2, 0, new TGeoTranslation(0., 0., mZPlan[disk]));
  
  Double_t ty = mSupportYDimensions[disk][0];
  
  for (Int_t ipart=1; ipart<mNPart[disk]; ipart ++) {
    ty += mSupportYDimensions[disk][ipart]/2.;
    TGeoVolume *partCarbon = gGeoManager->MakeBox(Form("partCarbon_D2_H%d_%d", half, ipart), mCarbon, mSupportXDimensions[disk][ipart]/2., mSupportYDimensions[disk][ipart]/2., mCarbonThickness);
    partCarbon->SetLineColor(kGray+3);
    auto *t = new TGeoTranslation ("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    carbonPlate -> AddNode(partCarbon, ipart, t);
    ty += mSupportYDimensions[disk][ipart]/2.;
  }
  
//  if (half == Top) {
//    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
//    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
//    mHalfDisk->AddNode(carbonPlate, 1, transformation);
//    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
//    mHalfDisk->AddNode(carbonPlate, 2, transformation);
//  }
//  else if (half == Bottom) {
    rotation = new TGeoRotation ("rotation", 180., 0., 0.);
    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
    mHalfDisk->AddNode(carbonPlate, 3, transformation);
    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
    mHalfDisk->AddNode(carbonPlate, 4, transformation);
//  }


  // **************************************** Rohacell Plate ****************************************
  
  auto *rohacellPlate = new TGeoVolumeAssembly(Form("rohacellPlate_D2_H%d",half));
  
  auto *rohacellBase2 = new TGeoBBox (Form("rohacellBase2_D2_H%d",half),  (mSupportXDimensions[disk][0])/2., (mSupportYDimensions[disk][0])/2., mRohacellThickness);
  // TGeoTranslation *t3 = new TGeoTranslation ("t3",0., (fSupportYDimensions[disk][0])/2. + fHalfDiskGap , 0.);
  // t3 -> RegisterYourself();
  
  auto *holeRohacell2 = new TGeoTubeSeg(Form("holeRohacell2_D2_H%d",half), 0., mRMin[disk], mRohacellThickness + 0.000001, 0, 180.);
  // TGeoTranslation *t4= new TGeoTranslation ("t4", 0., - fHalfDiskGap , 0.);
  // t4-> RegisterYourself();
  
  ///cs2 = new TGeoCompositeShape(Form("rohacell_D2_H%d",half), Form("(rohacellBase2_D2_H%d:t21)-(holeRohacell2_D2_H%d:t22)",half,half));
  auto    *rohacellhole2 = new TGeoSubtraction(rohacellBase2, holeRohacell2, t21, t22);
  auto *rh2 = new TGeoCompositeShape(Form("rohacellBase2_D2_H%d",half), rohacellhole2);
  auto *rohacellBaseWithHole = new TGeoVolume(Form("rohacellBaseWithHole_D2_H%d",half), rh2, mRohacell);



  rohacellBaseWithHole->SetLineColor(kGray);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation =  new TGeoCombiTrans(0., 0., 0., rotation);
  rohacellPlate -> AddNode(rohacellBaseWithHole, 0, new TGeoTranslation(0., 0., mZPlan[disk]));
  
  ty = mSupportYDimensions[disk][0];
  
  for (Int_t ipart=1; ipart<mNPart[disk]; ipart ++) {
    ty += mSupportYDimensions[disk][ipart]/2.;
    TGeoVolume *partRohacell = gGeoManager->MakeBox(Form("partRohacelli_D2_H%d_%d", half,ipart), mRohacell, mSupportXDimensions[disk][ipart]/2., mSupportYDimensions[disk][ipart]/2., mRohacellThickness);
    partRohacell->SetLineColor(kGray);
    auto *t = new TGeoTranslation ("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    rohacellPlate -> AddNode(partRohacell, ipart, t);
    ty += mSupportYDimensions[disk][ipart]/2.;
  }
  
//  if (half == Top) {
//    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
//    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
//    mHalfDisk->AddNode(rohacellPlate, 1, transformation);
//  }
//  if (half == Bottom) {
    rotation = new TGeoRotation ("rotation", 180., 0., 0.);
    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
    mHalfDisk->AddNode(rohacellPlate, 2, transformation);
//  }
  
    createManyfold(disk);
	
}

//_____________________________________________________________________________
void HeatExchanger::createHalfDisk3(Int_t half)  {
  
  Int_t disk = 3;
  
  if      (half == Top)    printf("Creating MFT heat exchanger for disk3 top\n");
  else if (half == Bottom) printf("Creating MFT heat exchanger for disk3 bottom\n");
  else     printf("No valid option for MFT heat exchanger on disk3\n");
  
  //mCarbon   = gGeoManager->GetMedium("MFT_Carbon$");
  mCarbon   = gGeoManager->GetMedium("MFT_CarbonFiber$");

  mWater    = gGeoManager->GetMedium("MFT_Water$");
  mRohacell = gGeoManager->GetMedium("MFT_Rohacell");
  mPipe     = gGeoManager->GetMedium("MFT_Polyimide");
  
  auto *cooling = new TGeoVolumeAssembly(Form("cooling_D3_H%d",half));
  
  Double_t deltaz= mHeatExchangerThickness - mCarbonThickness*2; //distance between pair of carbon plans
  Double_t lMiddle3[3] = {mSupportXDimensions[3][0] - 2.*mLWater3[0], mSupportXDimensions[3][0] - 2.*mLWater3[0], 0.};//distance between tube part
  
  TGeoTranslation *translation    = nullptr;
  TGeoRotation    *rotation       = nullptr;
  TGeoCombiTrans  *transformation = nullptr;
  
  Double_t beta3rad[3] = {0., 0., 0.};
  for (Int_t i=0; i<3; i++) {
    beta3rad[i] = mAngle3[i]*(TMath::DegToRad());
  }
  Double_t mAngleThirdPipe3rad=  mAngleThirdPipe3*(TMath::DegToRad());
  
  Double_t radius3mid[2] = {((lMiddle3[0]) - 2.*(mRadius3[0]*(TMath::Sin(beta3rad[0])) +
                                                 mLpartial3[0]*(TMath::Cos(beta3rad[0]))))/ (2*(TMath::Sin(beta3rad[0]))), 0.};//radius of central torus
  radius3mid[1] = (mSupportXDimensions[3][0]/2. - mLWater3[2]*TMath::Cos(mAngleThirdPipe3rad) -
                   mRadius3[2]*(TMath::Sin(beta3rad[2] + mAngleThirdPipe3rad) - TMath::Sin(mAngleThirdPipe3rad)))/(TMath::Sin(mAngleThirdPipe3rad + beta3rad[2]));
  
  lMiddle3[2] = mSupportXDimensions[3][0] - 2.*mLWater3[2]*(TMath::Cos(mAngleThirdPipe3rad));
  
  
  // **************************************** Water part ****************************************
  
  // ------------------- First and second pipe -------------------
  
  for (Int_t itube= 0; itube < 2; itube ++){
    
    
    // -------- Tube shape --------
    
    TGeoVolume *waterTube1 = gGeoManager->MakeTube(Form("waterTubeone%d_D3_H%d", itube,half), mWater, 0., mRWater, mLWater3[itube]/2.);
    waterTube1->SetLineColor(kBlue);
    translation = new TGeoTranslation (mXPosition3[itube], 0., mLWater3[itube]/2. + lMiddle3[itube]/2.);
    cooling->AddNode (waterTube1, 1, translation);
    
    TGeoVolume *waterTube2 = gGeoManager->MakeTube(Form("waterTubetwo%d_D3_H%d", itube,half), mWater, 0., mRWater, mLWater3[itube]/2.);
    waterTube2->SetLineColor(kBlue);
    translation = new TGeoTranslation (mXPosition3[itube], 0., -mLWater3[itube]/2. - lMiddle3[itube]/2.);
    cooling->AddNode (waterTube2, 2, translation);
    
    TGeoVolume *waterTube3 = gGeoManager->MakeTube(Form("waterTubethree%d_D3_H%d", itube,half), mWater, 0., mRWater, mLpartial3[itube]/2.);
    waterTube3->SetLineColor(kBlue);
    rotation = new TGeoRotation ("rotation", -90., 0 - mAngle3[itube], 0.);
    
    transformation = new TGeoCombiTrans(mXPosition3[itube] + mRadius3[itube]*(1-(TMath::Cos(beta3rad[0]))) +
                                        (mLpartial3[itube]/2.)*(TMath::Sin(beta3rad[0])), 0., (mRadius3[itube])*(TMath::Sin(beta3rad[0])) +
                                        (mLpartial3[itube]/2.)*(TMath::Cos(beta3rad[0])) - lMiddle3[itube]/2., rotation);
    cooling->AddNode (waterTube3, 3, transformation);
    
    rotation = new TGeoRotation ("rotation", 90., 180 - mAngle3[itube], 0.);
    transformation = new TGeoCombiTrans( mXPosition3[itube] + mRadius3[itube]*(1-(TMath::Cos(beta3rad[0]))) +
                                        (mLpartial3[itube]/2.)*(TMath::Sin(beta3rad[0])), 0., lMiddle3[itube]/2. - (mRadius3[itube])*(TMath::Sin(beta3rad[0])) -
                                        (mLpartial3[itube]/2.)*(TMath::Cos(beta3rad[0])), rotation);
    cooling->AddNode (waterTube3, 4, transformation);
    
    // -------- Torus shape --------
    
    //Sides torus
    TGeoVolume *waterTorus1 = gGeoManager->MakeTorus(Form("waterTorusone%d_D3_H%d", itube,half), mWater, mRadius3[itube], 0., mRWater, 0., mAngle3[itube]);
    waterTorus1->SetLineColor(kBlue);
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mRadius3[itube] + mXPosition3[itube], 0., - lMiddle3[itube]/2., rotation);
    cooling->AddNode (waterTorus1, 4, transformation);
    rotation = new TGeoRotation ("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mRadius3[itube] + mXPosition3[itube], 0., lMiddle3[itube]/2., rotation);
    cooling->AddNode (waterTorus1, 5, transformation);
    
    //Central torus
    TGeoVolume *waterTorus2 = gGeoManager->MakeTorus(Form("waterTorustwo%d_D3_H%d", itube,half), mWater, radius3mid[0], 0., mRWater, -mAngle3[itube], 2.*mAngle3[itube]);
    waterTorus2->SetLineColor(kBlue);
    rotation = new TGeoRotation ("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(mXPosition3[itube] + mRadius3[0]*(1-(TMath::Cos(beta3rad[0])))+mLpartial3[0]*TMath::Sin(beta3rad[0]) -
                                        radius3mid[0]*TMath::Cos(beta3rad[0]) , 0., 0., rotation);
    cooling->AddNode (waterTorus2, 6, transformation);
  }
  
  // ------------------- Third pipe -------------------
  
  // -------- Tube shape --------
  
  TGeoVolume *waterTube1 = gGeoManager->MakeTube(Form("waterTubeone2_D3_H%d",half), mWater, 0., mRWater, mLWater3[2]/2.);
  waterTube1->SetLineColor(kBlue);
  rotation = new TGeoRotation ("rotation", 90., -mAngleThirdPipe3, 90.);
  transformation = new TGeoCombiTrans (mXPosition3[2] + mLWater3[2]*TMath::Sin(mAngleThirdPipe3rad)/2., 0.,
                                       mSupportXDimensions[3][0]/2. - mLWater3[2]*(TMath::Cos(mAngleThirdPipe3rad))/2., rotation);
  cooling->AddNode (waterTube1, 3, transformation);
  
  rotation = new TGeoRotation ("rotation", 90., mAngleThirdPipe3, 90.);
  transformation = new TGeoCombiTrans (mXPosition3[2] + mLWater3[2]*TMath::Sin(mAngleThirdPipe3rad)/2., 0.,
                                       -mSupportXDimensions[3][0]/2. + mLWater3[2]*(TMath::Cos(mAngleThirdPipe3rad))/2., rotation);
  cooling->AddNode (waterTube1, 4, transformation);
  
  // -------- Torus shape --------
  
  TGeoVolume *waterTorus1 = gGeoManager->MakeTorus(Form("waterTorusone2_D3_H%d",half), mWater, mRadius3[2], 0., mRWater, mAngleThirdPipe3, mAngle3[2]);
  waterTorus1->SetLineColor(kBlue);
  rotation = new TGeoRotation ("rotation", 180., 90., 0.);
  transformation = new TGeoCombiTrans(mXPosition3[2] + mLWater3[2]*TMath::Sin(mAngleThirdPipe3rad) +
                                      mRadius3[2]*(TMath::Cos(mAngleThirdPipe3rad)), 0., -lMiddle3[2]/2. - mRadius3[2]*(TMath::Sin(mAngleThirdPipe3rad)) , rotation);
  cooling->AddNode (waterTorus1, 4, transformation);
  
  rotation = new TGeoRotation ("rotation", 180., -90., 0.);
  transformation = new TGeoCombiTrans( mXPosition3[2] + mLWater3[2]*TMath::Sin(mAngleThirdPipe3rad) +
                                      mRadius3[2]*(TMath::Cos(mAngleThirdPipe3rad)), 0.,  lMiddle3[2]/2. + mRadius3[2]*(TMath::Sin(mAngleThirdPipe3rad)), rotation);
  cooling->AddNode (waterTorus1, 5, transformation);
  
  TGeoVolume *waterTorus2 = gGeoManager->MakeTorus(Form("waterTorustwo2_D3_H%d",half), mWater, radius3mid[1], 0., mRWater, -(mAngle3[2] + mAngleThirdPipe3),
                                                   2.*(mAngle3[2] + mAngleThirdPipe3));
  waterTorus2->SetLineColor(kBlue);
  rotation = new TGeoRotation ("rotation", 0., 90., 0.);
  transformation = new TGeoCombiTrans( mXPosition3[2] + mLWater3[2]*TMath::Sin(mAngleThirdPipe3rad) + mRadius3[2]*(TMath::Cos(mAngleThirdPipe3rad)) -
                                      (mRadius3[2] + radius3mid[1])*(TMath::Cos(beta3rad[2] + mAngleThirdPipe3rad)), 0., 0., rotation);
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
    waterTorus1 = gGeoManager->MakeTorus(Form("waterTorusone%d_D3_H%d", i,half), mWater, radius3fourth[i], 0., mRWater, beta3fourth[i],  alpha3fourth[i]);
    waterTorus1->SetLineColor(kBlue);
    rotation = new TGeoRotation ("rotation", rotation3x[i], rotation3y[i], rotation3z[i]);
    transformation = new TGeoCombiTrans(translation3x[i], translation3y[i], translation3z[i], rotation);
    cooling->AddNode (waterTorus1, 7, transformation);
    rotation = new TGeoRotation ("rotation", rotation3x[i] , rotation3y[i] - 180, rotation3z[i]);
    transformation = new TGeoCombiTrans(translation3x[i], translation3y[i], - translation3z[i], rotation);
    cooling->AddNode (waterTorus1, 8, transformation);
  }
  
  waterTorus2 = gGeoManager->MakeTorus(Form("waterTorusone3_D3_H%d",half), mWater, radius3fourth[3], 0., mRWater, -alpha3fourth[3], 2*alpha3fourth[3]);
  waterTorus2->SetLineColor(kBlue);
  rotation = new TGeoRotation ("rotation", 180., 90., 180);
  transformation = new TGeoCombiTrans(translation3x[3], 0., 0., rotation);
  cooling->AddNode(waterTorus2, 9, transformation);
  
  // **************************************** Pipe part ****************************************
  
  // ------------------- First and second pipe -------------------
  
  for (Int_t itube= 0; itube < 2; itube ++){
    
    // -------- Tube shape --------
    
    TGeoVolume *pipeTube1 = gGeoManager->MakeTube(Form("pipeTubeone%d_D3_H%d", itube,half), mPipe, mRWater, mRWater + mDRPipe, mLWater3[itube]/2.);
    pipeTube1->SetLineColor(10);
    translation = new TGeoTranslation (mXPosition3[itube], 0., mLWater3[itube]/2. + lMiddle3[itube]/2.);
    cooling->AddNode (pipeTube1, 1, translation);
    
    TGeoVolume *pipeTube2 = gGeoManager->MakeTube(Form("pipeTubetwo%d_D3_H%d", itube,half), mPipe, mRWater, mRWater + mDRPipe, mLWater3[itube]/2.);
    pipeTube2->SetLineColor(10);
    translation = new TGeoTranslation (mXPosition3[itube], 0., -mLWater3[itube]/2. - lMiddle3[itube]/2.);
    cooling->AddNode (pipeTube2, 2, translation);
    
    TGeoVolume *pipeTube3 = gGeoManager->MakeTube(Form("pipeTubethree%d_D3_H%d", itube,half), mPipe, mRWater, mRWater + mDRPipe, mLpartial3[itube]/2.);
    pipeTube3->SetLineColor(10);
    rotation = new TGeoRotation ("rotation", -90., 0 - mAngle3[itube], 0.);
    
    transformation = new TGeoCombiTrans(mXPosition3[itube] + mRadius3[itube]*(1-(TMath::Cos(beta3rad[0]))) +
                                        (mLpartial3[itube]/2.)*(TMath::Sin(beta3rad[0])), 0., (mRadius3[itube])*(TMath::Sin(beta3rad[0])) +
                                        (mLpartial3[itube]/2.)*(TMath::Cos(beta3rad[0])) - lMiddle3[itube]/2., rotation);
    cooling->AddNode (pipeTube3, 3, transformation);
    
    rotation = new TGeoRotation ("rotation", 90., 180 - mAngle3[itube], 0.);
    transformation = new TGeoCombiTrans( mXPosition3[itube] + mRadius3[itube]*(1-(TMath::Cos(beta3rad[0]))) +
                                        (mLpartial3[itube]/2.)*(TMath::Sin(beta3rad[0])), 0., lMiddle3[itube]/2. -
                                        (mRadius3[itube])*(TMath::Sin(beta3rad[0])) - (mLpartial3[itube]/2.)*(TMath::Cos(beta3rad[0])), rotation);
    cooling->AddNode (pipeTube3, 4, transformation);
    
    // -------- Torus shape --------
    
    //Sides torus
    TGeoVolume *pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorusone%d_D3_H%d", itube,half), mPipe, mRadius3[itube], mRWater, mRWater + mDRPipe, 0., mAngle3[itube]);
    pipeTorus1->SetLineColor(10);
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mRadius3[itube] + mXPosition3[itube], 0., - lMiddle3[itube]/2., rotation);
    cooling->AddNode (pipeTorus1, 4, transformation);
    rotation = new TGeoRotation ("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mRadius3[itube] + mXPosition3[itube], 0., lMiddle3[itube]/2., rotation);
    cooling->AddNode (pipeTorus1, 5, transformation);
    
    
    //Central torus
    TGeoVolume *pipeTorus2 = gGeoManager->MakeTorus(Form("pipeTorustwo%d_D3_H%d", itube,half), mPipe, radius3mid[0], mRWater, mRWater + mDRPipe, -mAngle3[itube], 2.*mAngle3[itube]);
    pipeTorus2->SetLineColor(10);
    rotation = new TGeoRotation ("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(mXPosition3[itube] + mRadius3[0]*(1-(TMath::Cos(beta3rad[0])))+mLpartial3[0]*TMath::Sin(beta3rad[0]) - radius3mid[0]*TMath::Cos(beta3rad[0]) , 0., 0., rotation);
    cooling->AddNode (pipeTorus2, 6, transformation);
  }
  
  // ------------------- Third pipe -------------------
  
  // -------- Tube shape --------
  
  TGeoVolume *pipeTube1 = gGeoManager->MakeTube(Form("pipeTubeone2_D3_H%d",half), mPipe, mRWater, mRWater + mDRPipe, mLWater3[2]/2.);
  pipeTube1->SetLineColor(10);
  rotation = new TGeoRotation ("rotation", 90., -mAngleThirdPipe3, 90.);
  transformation = new TGeoCombiTrans (mXPosition3[2] + mLWater3[2]*TMath::Sin(mAngleThirdPipe3rad)/2., 0.,
                                       mSupportXDimensions[3][0]/2. - mLWater3[2]*(TMath::Cos(mAngleThirdPipe3rad))/2., rotation);
  cooling->AddNode (pipeTube1, 3, transformation);
  
  rotation = new TGeoRotation ("rotation", 90., mAngleThirdPipe3, 90.);
  transformation = new TGeoCombiTrans (mXPosition3[2] + mLWater3[2]*TMath::Sin(mAngleThirdPipe3rad)/2., 0.,
                                       -mSupportXDimensions[3][0]/2. + mLWater3[2]*(TMath::Cos(mAngleThirdPipe3rad))/2., rotation);
  cooling->AddNode (pipeTube1, 4, transformation);
  
  // -------- Torus shape --------
  
  TGeoVolume *pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorusone2_D3_H%d",half), mPipe, mRadius3[2], mRWater, mRWater + mDRPipe, mAngleThirdPipe3, mAngle3[2]);
  pipeTorus1->SetLineColor(10);
  rotation = new TGeoRotation ("rotation", 180., 90., 0.);
  transformation = new TGeoCombiTrans(mXPosition3[2] + mLWater3[2]*TMath::Sin(mAngleThirdPipe3rad) +
                                      mRadius3[2]*(TMath::Cos(mAngleThirdPipe3rad)), 0., -lMiddle3[2]/2. - mRadius3[2]*(TMath::Sin(mAngleThirdPipe3rad)), rotation);
  cooling->AddNode (pipeTorus1, 4, transformation);
  rotation = new TGeoRotation ("rotation", 180., -90., 0.);
  transformation = new TGeoCombiTrans( mXPosition3[2] + mLWater3[2]*TMath::Sin(mAngleThirdPipe3rad) +
                                      mRadius3[2]*(TMath::Cos(mAngleThirdPipe3rad)), 0.,  lMiddle3[2]/2. + mRadius3[2]*(TMath::Sin(mAngleThirdPipe3rad)), rotation);
  cooling->AddNode (pipeTorus1, 5, transformation);
  
  TGeoVolume *pipeTorus2 = gGeoManager->MakeTorus(Form("pipeTorustwo2_D3_H%d",half), mPipe, radius3mid[1], mRWater, mRWater + mDRPipe,
                                                  -(mAngle3[2] + mAngleThirdPipe3), 2.*(mAngle3[2] + mAngleThirdPipe3));
  pipeTorus2->SetLineColor(10);
  rotation = new TGeoRotation ("rotation", 0., 90., 0.);
  transformation = new TGeoCombiTrans( mXPosition3[2] + mLWater3[2]*TMath::Sin(mAngleThirdPipe3rad) +
                                      mRadius3[2]*(TMath::Cos(mAngleThirdPipe3rad)) -
                                      (mRadius3[2] + radius3mid[1])*(TMath::Cos(beta3rad[2] + mAngleThirdPipe3rad)), 0., 0., rotation);
  cooling->AddNode (pipeTorus2, 6, transformation);
  
  // ------------------- Fourth pipe -------------------
  
  for(Int_t i= 0; i<3; i++){
    pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorusone%d_D3_H%d", i,half), mPipe, radius3fourth[i], mRWater, mRWater + mDRPipe, beta3fourth[i],  alpha3fourth[i]);
    pipeTorus1->SetLineColor(10);
    rotation = new TGeoRotation ("rotation", rotation3x[i], rotation3y[i], rotation3z[i]);
    transformation = new TGeoCombiTrans(translation3x[i], translation3y[i], translation3z[i], rotation);
    cooling->AddNode (pipeTorus1, 7, transformation);
    rotation = new TGeoRotation ("rotation", rotation3x[i] , rotation3y[i] - 180, rotation3z[i]);
    transformation = new TGeoCombiTrans(translation3x[i], translation3y[i], - translation3z[i], rotation);
    cooling->AddNode (pipeTorus1, 8, transformation);
  }
  
  pipeTorus2 = gGeoManager->MakeTorus(Form("pipeTorusone3_D3_H%d",half), mPipe, radius3fourth[3], mRWater, mRWater + mDRPipe, -alpha3fourth[3], 2*alpha3fourth[3]);
  pipeTorus2->SetLineColor(10);
  rotation = new TGeoRotation ("rotation", 180., 90., 180);
  transformation = new TGeoCombiTrans(translation3x[3], 0., 0., rotation);
  cooling->AddNode(pipeTorus2, 9, transformation);
  
//  if (half == Top) {
//    rotation = new TGeoRotation ("rotation", 90., 90., 0.);
//    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz/2. - mCarbonThickness - mRWater - mDRPipe, rotation);
//    mHalfDisk->AddNode(cooling, 1, transformation);
//    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz/2. + mCarbonThickness + mRWater + mDRPipe, rotation);
//    mHalfDisk->AddNode(cooling, 2, transformation);
//  }
//  else if (half == Bottom) {
    rotation = new TGeoRotation ("rotation", -90., 90., 0.);
    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz/2. - mCarbonThickness - mRWater - mDRPipe, rotation);
    mHalfDisk->AddNode(cooling, 3, transformation);
    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz/2. + mCarbonThickness + mRWater + mDRPipe, rotation);
    mHalfDisk->AddNode(cooling, 4, transformation);
//  }
	
  // **************************************** Carbon Plates ****************************************
  
  auto *carbonPlate = new TGeoVolumeAssembly(Form("carbonPlate_D3_H%d",half));
  
  auto *carbonBase3 = new TGeoBBox (Form("carbonBase3_D3_H%d",half),  (mSupportXDimensions[disk][0])/2., (mSupportYDimensions[disk][0])/2., mCarbonThickness);
  auto *t31= new TGeoTranslation ("t31",0., (mSupportYDimensions[disk][0])/2.+ mHalfDiskGap , 0.);
  t31-> RegisterYourself();
  
  auto *holeCarbon3 = new TGeoTubeSeg(Form("holeCarbon3_D3_H%d",half), 0., mRMin[disk], mCarbonThickness + 0.000001, 0, 180.);
  auto *t32= new TGeoTranslation ("t32",0., - mHalfDiskGap , 0.);
  t32-> RegisterYourself();
  

  ///TGeoCompositeShape *cs3 = new TGeoCompositeShape(Form("Carbon3_D3_H%d",half),Form("(carbonBase3_D3_H%d:t31)-(holeCarbon3_D3_H%d:t32)",half,half) );
  auto    *carbonhole3 = new TGeoSubtraction(carbonBase3, holeCarbon3, t31, t32);
  auto *cs3 = new TGeoCompositeShape(Form("Carbon3_D3_H%d",half), carbonhole3);
  auto *carbonBaseWithHole3 = new TGeoVolume(Form("carbonBaseWithHole_D3_H%d",half), cs3, mCarbon);


  carbonBaseWithHole3->SetLineColor(kGray+3);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  carbonPlate->AddNode(carbonBaseWithHole3, 0, new TGeoTranslation(0., 0., mZPlan[disk]));
  
  Double_t ty = mSupportYDimensions[disk][0];
  
  for (Int_t ipart=1; ipart<mNPart[disk]; ipart ++) {
    ty += mSupportYDimensions[disk][ipart]/2.;
    TGeoVolume *partCarbon = gGeoManager->MakeBox(Form("partCarbon_D3_H%d_%d", half,ipart), mCarbon, mSupportXDimensions[disk][ipart]/2.,
                                                  mSupportYDimensions[disk][ipart]/2., mCarbonThickness);
    partCarbon->SetLineColor(kGray+3);
    auto *t = new TGeoTranslation ("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    carbonPlate -> AddNode(partCarbon, ipart, t);
    ty += mSupportYDimensions[disk][ipart]/2.;
  }
  
//  if (half == Top) {
//    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
//    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
//    mHalfDisk->AddNode(carbonPlate, 1, transformation);
//    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
//    mHalfDisk->AddNode(carbonPlate, 2, transformation);
//  }
//  else if (half == Bottom) {
    rotation = new TGeoRotation ("rotation", 180., 0., 0.);
    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
    mHalfDisk->AddNode(carbonPlate, 3, transformation);
    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
    mHalfDisk->AddNode(carbonPlate, 4, transformation);
//  }
	

  // **************************************** Rohacell Plate ****************************************
  
  auto *rohacellPlate = new TGeoVolumeAssembly(Form("rohacellPlate_D3_H%d",half));
  
  auto *rohacellBase3 = new TGeoBBox (Form("rohacellBase3_D3_H%d",half),  (mSupportXDimensions[disk][0])/2., (mSupportYDimensions[disk][0])/2., mRohacellThickness);
  // TGeoTranslation *t3 = new TGeoTranslation ("t3",0., (fSupportYDimensions[disk][0])/2. + fHalfDiskGap , 0.);
  // t3 -> RegisterYourself();
  
  auto *holeRohacell3 = new TGeoTubeSeg(Form("holeRohacell3_D3_H%d",half), 0., mRMin[disk], mRohacellThickness + 0.000001, 0, 180.);
  // TGeoTranslation *t4= new TGeoTranslation ("t4", 0., - fHalfDiskGap , 0.);
  // t4-> RegisterYourself();
  
  ///cs3 = new TGeoCompositeShape(Form("rohacell_D3_H%d",half), Form("(rohacellBase3_D3_H%d:t31)-(holeRohacell3_D3_H%d:t32)",half,half));
  auto    *rohacellhole3 = new TGeoSubtraction(rohacellBase3, holeRohacell3, t31, t32);
  auto *rh3 = new TGeoCompositeShape(Form("rohacellBase3_D3_H%d",half), rohacellhole3);
  auto *rohacellBaseWithHole = new TGeoVolume(Form("rohacellBaseWithHole_D3_H%d",half), rh3, mRohacell);


  rohacellBaseWithHole->SetLineColor(kGray);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation =  new TGeoCombiTrans(0., 0., 0., rotation);
  rohacellPlate -> AddNode(rohacellBaseWithHole, 0, new TGeoTranslation(0., 0., mZPlan[disk]));
  
  ty = mSupportYDimensions[disk][0];
  
  for (Int_t ipart=1; ipart<mNPart[disk]; ipart ++) {
    ty += mSupportYDimensions[disk][ipart]/2.;
    TGeoVolume *partRohacell = gGeoManager->MakeBox(Form("partRohacelli_D3_H%d_%d", half, ipart), mRohacell, mSupportXDimensions[disk][ipart]/2.,
                                                    mSupportYDimensions[disk][ipart]/2., mRohacellThickness);
    partRohacell->SetLineColor(kGray);
    auto *t = new TGeoTranslation ("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    rohacellPlate -> AddNode(partRohacell, ipart, t);
    ty += mSupportYDimensions[disk][ipart]/2.;
  }
  
//  if (half == Top) {
//    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
//    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
//    mHalfDisk->AddNode(rohacellPlate, 1, transformation);
//  }
//  if (half == Bottom) {
    rotation = new TGeoRotation ("rotation", 180., 0., 0.);
    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
    mHalfDisk->AddNode(rohacellPlate, 2, transformation);
//  }
	
  
    createManyfold(disk);
  
}

//_____________________________________________________________________________
void HeatExchanger::createHalfDisk4(Int_t half) {
  
  Int_t disk = 4;
  
  if      (half == Top)    printf("Creating MFT heat exchanger for disk4 top\n");
  else if (half == Bottom) printf("Creating MFT heat exchanger for disk4 bottom\n");
  else     printf("No valid option for MFT heat exchanger on disk4\n");
  
  
  //mCarbon   = gGeoManager->GetMedium("MFT_Carbon$");
  mCarbon   = gGeoManager->GetMedium("MFT_CarbonFiber$");

  mWater    = gGeoManager->GetMedium("MFT_Water$");
  mRohacell = gGeoManager->GetMedium("MFT_Rohacell");
  mPipe     = gGeoManager->GetMedium("MFT_Polyimide");
    
  auto *cooling = new TGeoVolumeAssembly(Form("cooling_D4_H%d",half));
  Double_t deltaz= mHeatExchangerThickness - mCarbonThickness*2; //distance between pair of carbon plans
  
  TGeoTranslation *translation    = nullptr;
  TGeoRotation    *rotation       = nullptr;
  TGeoCombiTrans  *transformation = nullptr;
  
  Double_t lMiddle4[3] = {mSupportXDimensions[4][0] - 2*mLWater4[0], mSupportXDimensions[4][0] - 2*mLWater4[1], mSupportXDimensions[4][0] - 2*mLWater4[2]};                 //distance between tube part
  mAngle4[5] = (mAngle4[3] - mAngle4[4]);
  Double_t anglerad[6]= {0.};  //angle of the sides torus
  for(Int_t i=0; i<6; i++){
    anglerad[i] = mAngle4[i]*(TMath::DegToRad());
  }
  Double_t mRadius4mid[3] = { (lMiddle4[0]-2.*(mRadius4[0]*(TMath::Sin(anglerad[0])) + mLpartial4[0]*(TMath::Cos(anglerad[0]))))/(2*(TMath::Sin(anglerad[0]))) ,
    (lMiddle4[1]-2.*(mRadius4[1]*(TMath::Sin(anglerad[1])) + mLpartial4[1]*(TMath::Cos(anglerad[1]))))/(2*(TMath::Sin(anglerad[1]))), 0. };                // radius of the central torus
  mRadius4mid[2] = (mSupportXDimensions[4][0]/2. - mRadius4[3]*TMath::Sin(anglerad[3]) - mRadius4[4]*(TMath::Sin(anglerad[3]) -
                                                                                                      TMath::Sin(anglerad[5])))/(TMath::Sin(anglerad[5]));
  
  // **************************************** Water part ****************************************
  
  // ------------------- First and second pipe -------------------
  
  for (Int_t i=0; i<2; i++){
    
    // -------- Tube shape --------
    
    TGeoVolume *waterTube1 = gGeoManager->MakeTube(Form("waterTubeone%d_D4_H%d", i,half), mWater, 0., mRWater, mLWater4[i]/2.);
    waterTube1->SetLineColor(kBlue);
    translation = new TGeoTranslation (mXPosition4[i], 0., mLWater4[i]/2. + lMiddle4[i]/2.);
    cooling->AddNode (waterTube1, 1, translation);
    translation = new TGeoTranslation (mXPosition4[i], 0., -mLWater4[i]/2. - lMiddle4[i]/2.);
    cooling->AddNode (waterTube1, 2, translation);
    
    TGeoVolume *waterTube2 = gGeoManager->MakeTube(Form("waterTubetwo%d_D4_H%d", i,half), mWater, 0., mRWater, mLpartial4[i]/2.);
    waterTube2->SetLineColor(kBlue);
    rotation = new TGeoRotation ("rotation", -90., - mAngle4[i], 0.);
    transformation = new TGeoCombiTrans( mXPosition4[i]+mRadius4[i]*(1-(TMath::Cos(anglerad[i])))+mLpartial4[i]*TMath::Sin(anglerad[i])/2., 0.,
                                        -mSupportXDimensions[4][0]/2. + mLWater4[i] + mRadius4[i]*(TMath::Sin(anglerad[i])) +
                                        mLpartial4[i]*(TMath::Cos(anglerad[i]))/2., rotation);
    cooling->AddNode (waterTube2, 3, transformation);
    rotation = new TGeoRotation ("rotation", -90.,  mAngle4[i], 0.);
    transformation = new TGeoCombiTrans( mXPosition4[i]+mRadius4[i]*(1-(TMath::Cos(anglerad[i])))+mLpartial4[i]*TMath::Sin(anglerad[i])/2., 0.,
                                        mSupportXDimensions[4][0]/2. - mLWater4[i] - mRadius4[i]*(TMath::Sin(anglerad[i])) -
                                        mLpartial4[i]*(TMath::Cos(anglerad[i]))/2. , rotation);
    cooling->AddNode (waterTube2, 4, transformation);
    
    // -------- Torus shape --------
    
    TGeoVolume *waterTorus1 = gGeoManager->MakeTorus(Form("waterTorusone%d_D4_H%d", i,half), mWater, mRadius4[i], 0., mRWater, 0., mAngle4[i]);
    waterTorus1->SetLineColor(kBlue);
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mXPosition4[i] + mRadius4[i], 0., -mSupportXDimensions[4][0]/2. + mLWater4[i], rotation);
    cooling->AddNode (waterTorus1, 1, transformation);
    rotation = new TGeoRotation ("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mXPosition4[i] + mRadius4[i], 0., mSupportXDimensions[4][0]/2. - mLWater4[i], rotation);
    cooling->AddNode (waterTorus1, 2, transformation);
    
    TGeoVolume *waterTorus2 = gGeoManager->MakeTorus(Form("waterTorustwo%d_D4_H%d", i,half), mWater, mRadius4mid[i], 0., mRWater, 180 - mAngle4[i] ,2*mAngle4[i]);
    waterTorus2->SetLineColor(kBlue);
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mXPosition4[i]  + mRadius4[i]*(1-(TMath::Cos(anglerad[i])))+mLpartial4[i]*TMath::Sin(anglerad[i]) -
                                        mRadius4mid[i]*TMath::Cos(anglerad[i]), 0., 0., rotation);
    cooling->AddNode (waterTorus2, 3, transformation);
    
  }
  
  // ------------------- Third pipe -------------------
  
  // -------- Tube shape --------
  
  TGeoVolume *waterTube1 = gGeoManager->MakeTube(Form("waterTubeone2_D4_H%d", half), mWater, 0., mRWater, mLWater4[2]/2.);
  waterTube1->SetLineColor(kBlue);
  translation = new TGeoTranslation (mXPosition4[2], 0., mLWater4[2]/2. + lMiddle4[2]/2.);
  cooling->AddNode (waterTube1, 1, translation);
  translation = new TGeoTranslation (mXPosition4[2], 0., -mLWater4[2]/2. - lMiddle4[2]/2.);
  cooling->AddNode (waterTube1, 2, translation);
  
  TGeoVolume *waterTube2 = gGeoManager->MakeTube(Form("waterTubetwo2_D4_H%d", half), mWater, 0., mRWater,  lMiddle4[2]/2. - 2.*mRadius4[2]*TMath::Sin(anglerad[2]));
  waterTube2->SetLineColor(kBlue);
  translation = new TGeoTranslation (mXPosition4[2] + 2.*mRadius4[2]*(1-TMath::Cos(anglerad[2])), 0., 0.);
  cooling->AddNode (waterTube2, 3, translation);
  
  // -------- Torus shape --------
  
  TGeoVolume *waterTorus1 = gGeoManager->MakeTorus(Form("waterTorusone2_D4_H%d", half), mWater, mRadius4[2], 0., mRWater, 0., mAngle4[2]);
  waterTorus1->SetLineColor(kBlue);
  rotation = new TGeoRotation ("rotation", 180., 90., 0.);
  transformation = new TGeoCombiTrans(mXPosition4[2] + mRadius4[2], 0., -mSupportXDimensions[4][0]/2. + mLWater4[2], rotation);
  cooling->AddNode (waterTorus1, 1, transformation);
  rotation = new TGeoRotation ("rotation", 180., -90., 180 - mAngle4[2]);
  transformation = new TGeoCombiTrans(mXPosition4[2] + mRadius4[2] - 2*mRadius4[2]*TMath::Cos(anglerad[2]), 0.,
                                      -mSupportXDimensions[4][0]/2. + mLWater4[2] + 2*mRadius4[2]*TMath::Sin(anglerad[2]), rotation);
  cooling->AddNode (waterTorus1, 2, transformation);
  rotation = new TGeoRotation ("rotation", 180., -90., 0.);
  transformation = new TGeoCombiTrans(mXPosition4[2] + mRadius4[2], 0., mSupportXDimensions[4][0]/2. - mLWater4[2], rotation);
  cooling->AddNode (waterTorus1, 3, transformation);
  rotation = new TGeoRotation ("rotation", 180., 90., 180 - mAngle4[2]);
  transformation = new TGeoCombiTrans(mXPosition4[2] + mRadius4[2] - 2*mRadius4[2]*TMath::Cos(anglerad[2]), 0.,
                                      mSupportXDimensions[4][0]/2. - mLWater4[2] - 2*mRadius4[2]*TMath::Sin(anglerad[2]), rotation);
  cooling->AddNode (waterTorus1, 4, transformation);
  
  // ------------------- Fourth pipe -------------------
  
  waterTorus1 = gGeoManager->MakeTorus(Form("waterTorusone3_D4_H%d", half), mWater, mRadius4[3], 0., mRWater, 0., mAngle4[3]);
  waterTorus1->SetLineColor(kBlue);
  rotation = new TGeoRotation ("rotation", 180., 90., 0.);
  transformation = new TGeoCombiTrans(mXPosition4[3] + mRadius4[3], 0., -mSupportXDimensions[4][0]/2., rotation);
  cooling->AddNode (waterTorus1, 1, transformation);
  
  TGeoVolume *waterTorus2 = gGeoManager->MakeTorus(Form("waterTorustwo3_D4_H%d", half), mWater,  mRadius4[4] , 0., mRWater, 0., mAngle4[4]);
  waterTorus2->SetLineColor(kBlue);
  rotation = new TGeoRotation ("rotation", 180., -90., 180 - mAngle4[3]);
  transformation = new TGeoCombiTrans( mXPosition4[3] + mRadius4[3] - mRadius4[3]*TMath::Cos(anglerad[3]) -
                                      mRadius4[4]*TMath::Cos(anglerad[3]), 0., -mSupportXDimensions[4][0]/2. +
                                      mRadius4[3]*TMath::Sin(anglerad[3]) + mRadius4[4]*TMath::Sin(anglerad[3]), rotation);
  cooling->AddNode (waterTorus2, 1, transformation);
  rotation = new TGeoRotation ("rotation", 180., -90., 0.);
  transformation = new TGeoCombiTrans(mXPosition4[3] + mRadius4[3], 0., mSupportXDimensions[4][0]/2., rotation);
  cooling->AddNode (waterTorus1, 2, transformation);
  rotation = new TGeoRotation ("rotation", 180., 90., 180 - mAngle4[3]);
  transformation = new TGeoCombiTrans( mXPosition4[3] + mRadius4[3] - mRadius4[3]*TMath::Cos(anglerad[3]) -
                                      mRadius4[4]*TMath::Cos(anglerad[3]), 0., mSupportXDimensions[4][0]/2. -
                                      mRadius4[3]*TMath::Sin(anglerad[3]) - mRadius4[4]*TMath::Sin(anglerad[3]), rotation);
  cooling->AddNode (waterTorus2, 2, transformation);
  
  TGeoVolume *waterTorus3 =  gGeoManager->MakeTorus(Form("waterTorusthree3_D4_H%d", half), mWater,  mRadius4mid[2] , 0., mRWater, -mAngle4[5], 2.*mAngle4[5]);
  waterTorus3->SetLineColor(kBlue);
  rotation = new TGeoRotation ("rotation", 0., 90., 0.);
  transformation = new TGeoCombiTrans( mXPosition4[3] + mRadius4[3] - mRadius4[3]*TMath::Cos(anglerad[3]) -
                                      mRadius4[4]*TMath::Cos(anglerad[3]) - ((mRadius4mid[2] - mRadius4[4])*TMath::Cos(anglerad[5])), 0., 0., rotation);
  cooling->AddNode (waterTorus3, 1, transformation);
  
  // ------------------- Fifth pipe -------------------
  
  mAngle4fifth[3] = mAngle4fifth[0] - mAngle4fifth[1] + mAngle4fifth[2];
  Double_t angle4fifthrad[4] = {0., 0., 0., 0.};
  for(Int_t i=0; i<4; i++){
    angle4fifthrad[i] = (TMath::Pi())*(mAngle4fifth[i])/180.;
  }
  Double_t beta4fourth[4] = {0, mAngle4fifth[0], mAngle4fifth[0] - mAngle4fifth[1], 180};  //shift angle
  Double_t beta4fourthrad[4] = {};
  for(Int_t i=0; i<4; i++){
    beta4fourthrad[i] = (TMath::Pi())*(beta4fourth[i])/180.;
  }
  Double_t translation4x[4] = { mXPosition4[4] + mRadius4fifth[0]*(TMath::Cos(beta4fourthrad[0])),
				mXPosition4[4] + mRadius4fifth[0]*((TMath::Cos(beta4fourthrad[0])) - TMath::Cos(beta4fourthrad[0] + angle4fifthrad[0])) -
				mRadius4fifth[1]*(TMath::Cos(beta4fourthrad[0] + angle4fifthrad[0])),
				
				mXPosition4[4] + mRadius4fifth[0]*((TMath::Cos(beta4fourthrad[0])) - TMath::Cos(beta4fourthrad[0] + angle4fifthrad[0])) -
				mRadius4fifth[1]*(TMath::Cos(beta4fourthrad[0] + angle4fifthrad[0])) +
				mRadius4fifth[1]*(TMath::Sin(TMath::Pi()/2. - angle4fifthrad[1] + angle4fifthrad[0] + beta4fourthrad[0])) +
				mRadius4fifth[2]*(TMath::Cos(angle4fifthrad[1] - angle4fifthrad[0] - beta4fourthrad[0])),
    
				mXPosition4[4] + mRadius4fifth[0]*((TMath::Cos(beta4fourthrad[0])) - TMath::Cos(beta4fourthrad[0] + angle4fifthrad[0])) -
				mRadius4fifth[1]*(TMath::Cos(beta4fourthrad[0] + angle4fifthrad[0])) +
				mRadius4fifth[1]*(TMath::Sin(TMath::Pi()/2. - angle4fifthrad[1] + angle4fifthrad[0] + beta4fourthrad[0])) +
				mRadius4fifth[2]*(TMath::Cos(angle4fifthrad[1] - angle4fifthrad[0] - beta4fourthrad[0])) -
				mRadius4fifth[2]*(TMath::Sin((TMath::Pi()/2.) - angle4fifthrad[3])) - mRadius4fifth[3]*(TMath::Cos(angle4fifthrad[3]))};
  
  Double_t translation4y[4] = {0., 0., 0., 0.};
  
  Double_t translation4z[4] = {-(mLWater4[0] + lMiddle4[0]/2.) - mRadius4fifth[0]*(TMath::Sin(beta4fourthrad[0])),
    
    -(mLWater4[0] + lMiddle4[0]/2.) - mRadius4fifth[0]*(TMath::Sin(beta4fourthrad[0])) +
    mRadius4fifth[0]*(TMath::Sin(beta4fourthrad[0] + angle4fifthrad[0])) +
    mRadius4fifth[1]*(TMath::Sin(beta4fourthrad[0] + angle4fifthrad[0])),
    
    -(mLWater4[0] + lMiddle4[0]/2.) - mRadius4fifth[0]*(TMath::Sin(beta4fourthrad[0])) +
    mRadius4fifth[0]*(TMath::Sin(beta4fourthrad[0] + angle4fifthrad[0])) +
    mRadius4fifth[1]*(TMath::Cos(TMath::Pi()/2 - beta4fourthrad[0] - angle4fifthrad[0])) +
    mRadius4fifth[1]*(TMath::Cos(TMath::Pi()/2. - angle4fifthrad[1] + angle4fifthrad[0] + beta4fourthrad[0])) +
    mRadius4fifth[2]*(TMath::Sin(angle4fifthrad[1] - angle4fifthrad[0] - beta4fourthrad[0])),
    
    -(mLWater4[0] + lMiddle4[0]/2.) - mRadius4fifth[0]*(TMath::Sin(beta4fourthrad[0])) +
    mRadius4fifth[0]*(TMath::Sin(beta4fourthrad[0] + angle4fifthrad[0])) +
    mRadius4fifth[1]*(TMath::Cos(TMath::Pi()/2 - beta4fourthrad[0] - angle4fifthrad[0])) +
    mRadius4fifth[1]*(TMath::Cos(TMath::Pi()/2. - angle4fifthrad[1] + angle4fifthrad[0] + beta4fourthrad[0])) +
    mRadius4fifth[2]*(TMath::Sin(angle4fifthrad[1] - angle4fifthrad[0] - beta4fourthrad[0])) +
    (mRadius4fifth[3] + mRadius4fifth[2])*(TMath::Sin(angle4fifthrad[3]))
  };
  
  Double_t rotation4x[4] = {180., 180., 180., 180};
  Double_t rotation4y[4] = {90., 90., 90., 90};
  Double_t rotation4z[4] = {0., 180 - mAngle4fifth[1]  , 0., 0.};
  
  for (Int_t i= 0; i<4; i++){
    waterTorus1 = gGeoManager->MakeTorus(Form("waterTorusone%d_D4_H%d", i,half), mWater, mRadius4fifth[i], 0., mRWater, beta4fourth[i],  mAngle4fifth[i]);
    waterTorus1->SetLineColor(kBlue);
    rotation = new TGeoRotation ("rotation", rotation4x[i], rotation4y[i], rotation4z[i]);
    transformation = new TGeoCombiTrans(translation4x[i], translation4y[i], translation4z[i], rotation);
    cooling->AddNode (waterTorus1, 7, transformation);
    rotation = new TGeoRotation ("rotation", rotation4x[i] , rotation4y[i] - 180, rotation4z[i]);
    transformation = new TGeoCombiTrans(translation4x[i], translation4y[i], - translation4z[i], rotation);
    cooling->AddNode (waterTorus1, 8, transformation);
  }
  
  TGeoVolume *waterTubeFive = gGeoManager->MakeTube(Form("waterTubeFive1_D4_H%d",half), mWater, 0., mRWater, -translation4z[3]);
  waterTubeFive->SetLineColor(kBlue);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(translation4x[3] + mRadius4fifth[3], 0., 0., rotation);
  cooling->AddNode(waterTubeFive, 1, transformation);
  
  
  // **************************************** Pipe part ****************************************
  
  // ------------------- First and second pipe -------------------
  
  for(Int_t i=0; i<2; i++){
    
    // -------- Tube shape --------
    
    TGeoVolume *pipeTube1 = gGeoManager->MakeTube(Form("pipeTubeone%d_D4_H%d", i,half), mPipe, mRWater, mRWater + mDRPipe, mLWater4[i]/2.);
    pipeTube1->SetLineColor(10);
    translation = new TGeoTranslation (mXPosition4[i], 0., mLWater4[i]/2. + lMiddle4[i]/2.);
    cooling->AddNode (pipeTube1, 1, translation);
    translation = new TGeoTranslation (mXPosition4[i], 0., -mLWater4[i]/2. - lMiddle4[i]/2.);
    cooling->AddNode (pipeTube1, 2, translation);
    
    TGeoVolume *pipeTube2 = gGeoManager->MakeTube(Form("pipeTubetwo%d_D4_H%d", i,half), mPipe, mRWater, mRWater + mDRPipe, mLpartial4[i]/2.);
    pipeTube2->SetLineColor(10);
    rotation = new TGeoRotation ("rotation", -90., - mAngle4[i], 0.);
    transformation = new TGeoCombiTrans( mXPosition4[i]+mRadius4[i]*(1-(TMath::Cos(anglerad[i])))+mLpartial4[i]*TMath::Sin(anglerad[i])/2., 0.,
                                        -mSupportXDimensions[4][0]/2. + mLWater4[i] + mRadius4[i]*(TMath::Sin(anglerad[i])) +
                                        mLpartial4[i]*(TMath::Cos(anglerad[i]))/2., rotation);
    cooling->AddNode (pipeTube2, 3, transformation);
    rotation = new TGeoRotation ("rotation", -90.,  mAngle4[i], 0.);
    transformation = new TGeoCombiTrans( mXPosition4[i]+mRadius4[i]*(1-(TMath::Cos(anglerad[i])))+mLpartial4[i]*TMath::Sin(anglerad[i])/2., 0.,
                                        mSupportXDimensions[4][0]/2. - mLWater4[i] - mRadius4[i]*(TMath::Sin(anglerad[i])) -
                                        mLpartial4[i]*(TMath::Cos(anglerad[i]))/2. , rotation);
    cooling->AddNode (pipeTube2, 4, transformation);
    
    // -------- Torus shape --------
    
    TGeoVolume *pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorusone%d_D4_H%d", i,half), mPipe, mRadius4[i], mRWater, mRWater + mDRPipe, 0., mAngle4[i]);
    pipeTorus1->SetLineColor(10);
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mXPosition4[i] + mRadius4[i], 0., -mSupportXDimensions[4][0]/2. + mLWater4[i], rotation);
    cooling->AddNode (pipeTorus1, 1, transformation);
    rotation = new TGeoRotation ("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mXPosition4[i] + mRadius4[i], 0., mSupportXDimensions[4][0]/2. - mLWater4[i], rotation);
    cooling->AddNode (pipeTorus1, 2, transformation);
    
    TGeoVolume *pipeTorus2 = gGeoManager->MakeTorus(Form("pipeTorustwo%d_D4_H%d", i,half), mPipe, mRadius4mid[i], mRWater, mRWater + mDRPipe, 180 - mAngle4[i] ,2*mAngle4[i]);
    pipeTorus2->SetLineColor(10);
    rotation = new TGeoRotation ("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mXPosition4[i]  + mRadius4[i]*(1-(TMath::Cos(anglerad[i])))+mLpartial4[i]*TMath::Sin(anglerad[i]) -
                                        mRadius4mid[i]*TMath::Cos(anglerad[i]), 0., 0., rotation);
    cooling->AddNode (pipeTorus2, 3, transformation);
    
  }
  
  // ------------------- Third pipe -------------------
  
  // -------- Tube shape --------
  
  TGeoVolume *pipeTube1 = gGeoManager->MakeTube(Form("pipeTubeone2_D4_H%d",half), mPipe, mRWater, mRWater + mDRPipe, mLWater4[2]/2.);
  pipeTube1->SetLineColor(10);
  translation = new TGeoTranslation (mXPosition4[2], 0., mLWater4[2]/2. + lMiddle4[2]/2.);
  cooling->AddNode (pipeTube1, 1, translation);
  translation = new TGeoTranslation (mXPosition4[2], 0., -mLWater4[2]/2. - lMiddle4[2]/2.);
  cooling->AddNode (pipeTube1, 2, translation);
  
  TGeoVolume *pipeTube2 = gGeoManager->MakeTube(Form("pipeTubetwo2_D4_H%d",half), mPipe, mRWater, mRWater + mDRPipe,  lMiddle4[2]/2. - 2.*mRadius4[2]*TMath::Sin(anglerad[2]));
  pipeTube2->SetLineColor(10);
  translation = new TGeoTranslation (mXPosition4[2] + 2.*mRadius4[2]*(1-TMath::Cos(anglerad[2])), 0., 0.);
  cooling->AddNode (pipeTube2, 3, translation);
  
  // -------- Torus shape --------
  
  TGeoVolume *pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorusone2_D4_H%d",half), mPipe, mRadius4[2], mRWater, mRWater + mDRPipe, 0., mAngle4[2]);
  pipeTorus1->SetLineColor(10);
  rotation = new TGeoRotation ("rotation", 180., 90., 0.);
  transformation = new TGeoCombiTrans(mXPosition4[2] + mRadius4[2], 0., -mSupportXDimensions[4][0]/2. + mLWater4[2], rotation);
  cooling->AddNode (pipeTorus1, 1, transformation);
  rotation = new TGeoRotation ("rotation", 180., -90., 180 - mAngle4[2]);
  transformation = new TGeoCombiTrans(mXPosition4[2] + mRadius4[2] - 2*mRadius4[2]*TMath::Cos(anglerad[2]), 0.,
                                      -mSupportXDimensions[4][0]/2. + mLWater4[2] + 2*mRadius4[2]*TMath::Sin(anglerad[2]), rotation);
  cooling->AddNode (pipeTorus1, 2, transformation);
  rotation = new TGeoRotation ("rotation", 180., -90., 0.);
  transformation = new TGeoCombiTrans(mXPosition4[2] + mRadius4[2], 0., mSupportXDimensions[4][0]/2. - mLWater4[2], rotation);
  cooling->AddNode (pipeTorus1, 3, transformation);
  rotation = new TGeoRotation ("rotation", 180., 90., 180 - mAngle4[2]);
  transformation = new TGeoCombiTrans(mXPosition4[2] + mRadius4[2] - 2*mRadius4[2]*TMath::Cos(anglerad[2]), 0.,
                                      mSupportXDimensions[4][0]/2. - mLWater4[2] - 2*mRadius4[2]*TMath::Sin(anglerad[2]), rotation);
  cooling->AddNode (pipeTorus1, 4, transformation);
  
  // ------------------- Fourth pipe -------------------
  
  pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorusone3_D4_H%d",half), mPipe, mRadius4[3], mRWater, mRWater + mDRPipe, 0., mAngle4[3]);
  pipeTorus1->SetLineColor(10);
  rotation = new TGeoRotation ("rotation", 180., 90., 0.);
  transformation = new TGeoCombiTrans(mXPosition4[3] + mRadius4[3], 0., -mSupportXDimensions[4][0]/2., rotation);
  cooling->AddNode (pipeTorus1, 1, transformation);
  
  TGeoVolume *pipeTorus2 = gGeoManager->MakeTorus(Form("pipeTorustwo3_D4_H%d",half), mPipe,  mRadius4[4] , mRWater, mRWater + mDRPipe, 0., mAngle4[4]);
  pipeTorus2->SetLineColor(10);
  rotation = new TGeoRotation ("rotation", 180., -90., 180 - mAngle4[3]);
  transformation = new TGeoCombiTrans( mXPosition4[3] + mRadius4[3] - mRadius4[3]*TMath::Cos(anglerad[3]) -
                                      mRadius4[4]*TMath::Cos(anglerad[3]), 0., -mSupportXDimensions[4][0]/2. +
                                      mRadius4[3]*TMath::Sin(anglerad[3]) + mRadius4[4]*TMath::Sin(anglerad[3]), rotation);
  cooling->AddNode (pipeTorus2, 1, transformation);
  rotation = new TGeoRotation ("rotation", 180., -90., 0.);
  transformation = new TGeoCombiTrans(mXPosition4[3] + mRadius4[3], 0., mSupportXDimensions[4][0]/2. , rotation);
  cooling->AddNode (pipeTorus1, 2, transformation);
  rotation = new TGeoRotation ("rotation", 180., 90., 180 - mAngle4[3]);
  transformation = new TGeoCombiTrans( mXPosition4[3] + mRadius4[3] - mRadius4[3]*TMath::Cos(anglerad[3]) -
                                      mRadius4[4]*TMath::Cos(anglerad[3]), 0., mSupportXDimensions[4][0]/2. -
                                      mRadius4[3]*TMath::Sin(anglerad[3]) - mRadius4[4]*TMath::Sin(anglerad[3]), rotation);
  cooling->AddNode (pipeTorus2, 2, transformation);
  
  TGeoVolume *pipeTorus3 =  gGeoManager->MakeTorus(Form("pipeTorusthree3_D4_H%d",half), mPipe,  mRadius4mid[2] , mRWater, mRWater + mDRPipe, -mAngle4[5], 2.*mAngle4[5]);
  pipeTorus3->SetLineColor(10);
  rotation = new TGeoRotation ("rotation", 0., 90., 0.);
  transformation = new TGeoCombiTrans( mXPosition4[3] + mRadius4[3] - mRadius4[3]*TMath::Cos(anglerad[3]) -
                                      mRadius4[4]*TMath::Cos(anglerad[3]) - ((mRadius4mid[2] - mRadius4[4])*TMath::Cos(anglerad[5])), 0., 0., rotation);
  cooling->AddNode (pipeTorus3, 1, transformation);
  
  // ------------------- Fifth pipe -------------------
  
  for(Int_t i= 0; i<4; i++){
    pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorusone%d_D4_H%d", i,half), mPipe, mRadius4fifth[i], mRWater, mRWater + mDRPipe, beta4fourth[i],  mAngle4fifth[i]);
    pipeTorus1->SetLineColor(10);
    rotation = new TGeoRotation ("rotation", rotation4x[i], rotation4y[i], rotation4z[i]);
    transformation = new TGeoCombiTrans(translation4x[i], translation4y[i], translation4z[i], rotation);
    cooling->AddNode (pipeTorus1, 7, transformation);
    rotation = new TGeoRotation ("rotation", rotation4x[i] , rotation4y[i] - 180, rotation4z[i]);
    transformation = new TGeoCombiTrans(translation4x[i], translation4y[i], - translation4z[i], rotation);
    cooling->AddNode (pipeTorus1, 8, transformation);
  }
  
  TGeoVolume *pipeTubeFive = gGeoManager->MakeTube(Form("pipeTubeFive1_D4_H%d", half), mPipe, mRWater, mRWater + mDRPipe, -translation4z[3]);
  pipeTubeFive->SetLineColor(10);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(translation4x[3] + mRadius4fifth[3], 0., 0., rotation);
  cooling->AddNode(pipeTubeFive, 1, transformation);
  
  
//  if (half == Top) {
//    rotation = new TGeoRotation ("rotation", 90., 90., 0.);
//    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz/2. - mCarbonThickness - mRWater - mDRPipe, rotation);
//    mHalfDisk->AddNode(cooling, 1, transformation);
//    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz/2. + mCarbonThickness + mRWater + mDRPipe, rotation);
//    mHalfDisk->AddNode(cooling, 2, transformation);
//  }
//  else if (half == Bottom) {
    rotation = new TGeoRotation ("rotation", -90., 90., 0.);
    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz/2. - mCarbonThickness - mRWater - mDRPipe, rotation);
    mHalfDisk->AddNode(cooling, 3, transformation);
    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz/2. + mCarbonThickness + mRWater + mDRPipe, rotation);
    mHalfDisk->AddNode(cooling, 4, transformation);
//  }
	
  // **************************************** Carbon Plates ****************************************
  
  auto *carbonPlate = new TGeoVolumeAssembly(Form("carbonPlate_D4_H%d",half));
  
  auto *carbonBase4 = new TGeoBBox (Form("carbonBase4_D4_H%d",half),  (mSupportXDimensions[disk][0])/2., (mSupportYDimensions[disk][0])/2., mCarbonThickness);
  auto *t41= new TGeoTranslation ("t41",0., (mSupportYDimensions[disk][0])/2. + mHalfDiskGap, 0.);
  t41-> RegisterYourself();
  
  auto *holeCarbon4 = new TGeoTubeSeg(Form("holeCarbon4_D4_H%d",half), 0., mRMin[disk], mCarbonThickness + 0.000001, 0, 180.);
  auto *t42= new TGeoTranslation ("t42",0., - mHalfDiskGap , 0.);
  t42-> RegisterYourself();
  
  
  ///TGeoCompositeShape *cs4 = new TGeoCompositeShape(Form("Carbon4_D4_H%d",half),Form("(carbonBase4_D4_H%d:t41)-(holeCarbon4_D4_H%d:t42)",half,half));
  auto    *carbonhole4 = new TGeoSubtraction(carbonBase4, holeCarbon4, t41, t42);
  auto *cs4 = new TGeoCompositeShape(Form("Carbon4_D4_H%d",half), carbonhole4);
  auto *carbonBaseWithHole4 = new TGeoVolume(Form("carbonBaseWithHole_D4_H%d",half), cs4, mCarbon);

  carbonBaseWithHole4->SetLineColor(kGray+3);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  carbonPlate->AddNode(carbonBaseWithHole4, 0, new TGeoTranslation(0., 0., mZPlan[disk]));
  
  Double_t ty = mSupportYDimensions[disk][0];
  
  for (Int_t ipart=1; ipart<mNPart[disk]; ipart ++) {
    ty += mSupportYDimensions[disk][ipart]/2.;
    TGeoVolume *partCarbon = gGeoManager->MakeBox(Form("partCarbon_D4_H%d_%d", half,ipart), mCarbon, mSupportXDimensions[disk][ipart]/2.,
                                                  mSupportYDimensions[disk][ipart]/2., mCarbonThickness);
    partCarbon->SetLineColor(kGray+3);
    auto *t = new TGeoTranslation ("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    carbonPlate -> AddNode(partCarbon, ipart, t);
    ty += mSupportYDimensions[disk][ipart]/2.;
  }
  
//  if (half == Top) {
//    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
//    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
//    mHalfDisk->AddNode(carbonPlate, 1, transformation);
//    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
//    mHalfDisk->AddNode(carbonPlate, 2, transformation);
//  }
//  else if (half == Bottom) {
    rotation = new TGeoRotation ("rotation", 180., 0., 0.);
    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
    mHalfDisk->AddNode(carbonPlate, 3, transformation);
    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
    mHalfDisk->AddNode(carbonPlate, 4, transformation);
//  }

	
  // **************************************** Rohacell Plate ****************************************
  
  auto *rohacellPlate = new TGeoVolumeAssembly(Form("rohacellPlate_D4_H%d",half));
  
  auto *rohacellBase4 = new TGeoBBox (Form("rohacellBase4_D4_H%d",half),  (mSupportXDimensions[disk][0])/2., (mSupportYDimensions[disk][0])/2., mRohacellThickness);
  // TGeoTranslation *t3 = new TGeoTranslation ("t3",0., (fSupportYDimensions[disk][0])/2. + fHalfDiskGap , 0.);
  // t3 -> RegisterYourself();
  
  auto *holeRohacell4 = new TGeoTubeSeg(Form("holeRohacell4_D4_H%d",half), 0., mRMin[disk], mRohacellThickness + 0.000001, 0, 180.);
  // TGeoTranslation *t4= new TGeoTranslation ("t4", 0., - fHalfDiskGap , 0.);
  // t4-> RegisterYourself();
  
  ///cs4 = new TGeoCompositeShape(Form("rohacell_D4_H%d",half), Form("(rohacellBase4_D4_H%d:t41)-(holeRohacell4_D4_H%d:t42)",half,half));
  auto    *rohacellhole4 = new TGeoSubtraction(rohacellBase4, holeRohacell4, t41, t42);
  auto *rh4 = new TGeoCompositeShape(Form("rohacellBase4_D4_H%d",half), rohacellhole4);
  auto *rohacellBaseWithHole = new TGeoVolume(Form("rohacellBaseWithHole_D4_H%d",half), rh4, mRohacell);

  rohacellBaseWithHole->SetLineColor(kGray);
  rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  transformation =  new TGeoCombiTrans(0., 0., 0., rotation);
  rohacellPlate -> AddNode(rohacellBaseWithHole, 0, new TGeoTranslation(0., 0., mZPlan[disk]));
  
  ty = mSupportYDimensions[disk][0];
  
  for (Int_t ipart=1; ipart<mNPart[disk]; ipart ++) {
    ty += mSupportYDimensions[disk][ipart]/2.;
    TGeoVolume *partRohacell = gGeoManager->MakeBox(Form("partRohacelli_D4_H%d_%d", half, ipart), mRohacell, mSupportXDimensions[disk][ipart]/2.,
                                                    mSupportYDimensions[disk][ipart]/2., mRohacellThickness);
    partRohacell->SetLineColor(kGray);
    auto *t = new TGeoTranslation ("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    rohacellPlate -> AddNode(partRohacell, ipart, t);
    ty += mSupportYDimensions[disk][ipart]/2.;
  }
  
//  if (half == Top) {
//    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
//    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
//    mHalfDisk->AddNode(rohacellPlate, 1, transformation);
//  }
//  if (half == Bottom) {
    rotation = new TGeoRotation ("rotation", 180., 0., 0.);
    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
    mHalfDisk->AddNode(rohacellPlate, 2, transformation);
//  }
	
    createManyfold(disk);
	
}

//_____________________________________________________________________________
void HeatExchanger::initParameters() 
{
  
  mHalfDiskRotation = new TGeoRotation**[Geometry::sNDisks];
  mHalfDiskTransformation = new TGeoCombiTrans**[Geometry::sNDisks];
  for (Int_t idisk = 0; idisk < Geometry::sNDisks; idisk++) {
    mHalfDiskRotation[idisk] = new TGeoRotation*[NHalves];
    mHalfDiskTransformation[idisk] = new TGeoCombiTrans*[NHalves];
    for (Int_t ihalf = 0; ihalf < NHalves; ihalf++) {
      mHalfDiskRotation[idisk][ihalf] = new TGeoRotation(Form("rotation%d%d", idisk, ihalf), 0., 0., 0.);
      mHalfDiskTransformation[idisk][ihalf] = new TGeoCombiTrans(Form("transformation%d%d", idisk, ihalf), 0., 0., 0., mHalfDiskRotation[idisk][ihalf]);
    }
  } 

  mRohacellThickness = mHeatExchangerThickness/2. - 2.*mCarbonThickness - 2*(mRWater + mDRPipe);//thickness of Rohacell plate over 2
  printf("Rohacell thickness %f \n",mRohacellThickness);
  
  mHalfDiskGap = 0.2;
  
  mNPart[0] = 3;
  mNPart[1] = 3;
  mNPart[2] = 3;
  mNPart[3] = 5;
  mNPart[4] = 4;
  
  mRMin[0] = 2.35;
  mRMin[1] = 2.35;
  mRMin[2] = 2.35;
  mRMin[3] = 3.35;
  mRMin[4] = 3.75;
  
  // mZPlan[0] = 46;
  // mZPlan[1] = 49.3;
  // mZPlan[2] = 53.1;
  // mZPlan[3] = 68.7;
  // mZPlan[4] = 76.8;
  
  mZPlan[0] = 0;
  mZPlan[1] = 0;
  mZPlan[2] = 0;
  mZPlan[3] = 0;
  mZPlan[4] = 0;
    
  mSupportXDimensions= new Double_t*[Geometry::sNDisks];
  mSupportYDimensions= new Double_t*[Geometry::sNDisks];
  
  for(Int_t i = 0; i < Geometry::sNDisks; i++) {
    mSupportXDimensions[i]= new double[mNPart[i]];
    mSupportYDimensions[i]= new double[mNPart[i]];
  }
  
  mSupportXDimensions[0][0]=23.;  mSupportXDimensions[0][1]=14.;   mSupportXDimensions[0][2]=5.1;
  mSupportXDimensions[1][0]=23.;  mSupportXDimensions[1][1]=14.;   mSupportXDimensions[1][2]=5.1;
  mSupportXDimensions[2][0]=23.; mSupportXDimensions[2][1]=14.;   mSupportXDimensions[2][2]=5.1;
  mSupportXDimensions[3][0]=28.4; mSupportXDimensions[3][1]=21.9;   mSupportXDimensions[3][2]=18.5 ;mSupportXDimensions[3][3]=8.3; mSupportXDimensions[3][4]=4.9;
  mSupportXDimensions[4][0]=28.4; mSupportXDimensions[4][1]=24.204; mSupportXDimensions[4][2]=21.9 ;mSupportXDimensions[4][3]=15.1;
  
  mSupportYDimensions[0][0]=6.7;  mSupportYDimensions[0][1]=2.6;   mSupportYDimensions[0][2]=2.0;
  mSupportYDimensions[1][0]=6.7;  mSupportYDimensions[1][1]=2.6;   mSupportYDimensions[1][2]=2.0;
  mSupportYDimensions[2][0]=6.7; mSupportYDimensions[2][1]=2.6;  mSupportYDimensions[2][2]=2.0;
  mSupportYDimensions[3][0]=6.61; mSupportYDimensions[3][1]=3.01;  mSupportYDimensions[3][2]=3.01 ;mSupportYDimensions[3][3]=1.8; mSupportYDimensions[3][4]=1.15;
  mSupportYDimensions[4][0]=6.61; mSupportYDimensions[4][1]=3.01;  mSupportYDimensions[4][2]=3.01 ;mSupportYDimensions[4][3]=2.42;
  
  //Paramteters for disks 0, 1, 2
  
  mLWater = 6.759;
  
  mXPosition0[0] = 1.7;
  mXPosition0[1] = 4.61;
  mXPosition0[2] = 6.41;
  
  mAngle0 = 44.6;
  mRadius0 = 2.5;
  mLpartial0 = 1.;
  
  //Parameters for disk 3
  
  mLWater3[0] = 8.032;
  mLWater3[1] = 8.032;
  mLWater3[2] = 8.2;
  
  mXPosition3[0] = 1.7;
  mXPosition3[1] = 4.61;
  mXPosition3[2] = 5.5;
  mXPosition3[3] = 5.81;
  
  mAngle3[0] = 41.3;
  mAngle3[1] = 41.3;
  mAngle3[2] = 28;
  
  mRadius3[0] = 4.3;
  mRadius3[1] = 4.3;
  mRadius3[2] = 7.4;
  
  mAngleThirdPipe3 = 15.;
  mLpartial3[0] = 2.3;
  mLpartial3[1] = 2.3;
  
  mRadius3fourth[0] = 9.6;
  mRadius3fourth[1] = 2.9;
  mRadius3fourth[2] = 2.9;
  mRadius3fourth[3] = 0.;
  
  mAngle3fourth[0] = 40.8;
  mAngle3fourth[1] = 50.;
  mAngle3fourth[2] = 60.;
  mAngle3fourth[3] =  8 + mAngle3fourth[0] - mAngle3fourth[1] + mAngle3fourth[2];
  
  // Parameters for disk 4
  
  mLWater4[0] = 5.911;
  mLWater4[1] = 3.697;
  mLWater4[2] = 3.038;
  
  mXPosition4[0] = 1.7;
  mXPosition4[1] = 3.492;
  mXPosition4[2] = 4.61;
  mXPosition4[3] = 5.5;
  mXPosition4[4] = 5.8;
  
  mAngle4[0] = 35.5;
  mAngle4[1] = 30.;
  mAngle4[2] = 54.;
  mAngle4[3] = 53.;
  mAngle4[4] = 40;
  mAngle4[5] = (mAngle4[3] - mAngle4[4]);
  
  mRadius4[0] = 6.6;
  mRadius4[1] = 7.2;
  mRadius4[2] = 4.6;
  mRadius4[3] = 6.2;
  mRadius4[4] = 6.;
  
  mLpartial4[0] = 2.5;
  mLpartial4[1] = 3.6;
  
  mAngle4fifth[0] = 64.;
  mAngle4fifth[1] = 30.;
  mAngle4fifth[2] = 27.;
  mAngle4fifth[3] = mAngle4fifth[0] - mAngle4fifth[1] + mAngle4fifth[2];
  
  mRadius4fifth[0] = 2.7;
  mRadius4fifth[1] = 5.;
  mRadius4fifth[2] = 5.1;
  mRadius4fifth[3] = 4.3;    
  
}


