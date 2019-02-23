//Info// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Support.cxx
/// \brief Class building the MFT PCB Supports
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>, Rafael Pezzi <rafael.pezzi@cern.ch>

#include "MFTBase/Constants.h"
#include "MFTBase/PCBSupport.h"
#include "MFTBase/Geometry.h"

using namespace o2::MFT;

/// \cond CLASSIMP
ClassImp(o2::MFT::PCBSupport)
/// \endcond

//_____________________________________________________________________________
PCBSupport::PCBSupport() :
TNamed(),
mHalfDisk(nullptr),
mDiskGap(1.4),
mPCBRad{17.5,17.5,17.5,23.0,23.0},
mCuThickness(.05),
mFR4Thickness(.1),
mPhi0(0.),
mPhi1(180.),
mT_delta(0.001),
mOuterCut{15.5,15.5,16.9,20.5,21.9}
{

  initParameters();

}

//_____________________________________________________________________________
TGeoVolumeAssembly* PCBSupport::create(Int_t half, Int_t disk)
{

  Info("Create",Form("Creating PCB_H%d_D%d", half,disk),0,0);
  mHalfDisk = new TGeoVolumeAssembly(Form("PCB_H%d_D%d", half,disk));
  //  auto *PCB = new TGeoVolumeAssembly(Form("PCB_VA_H%d_D%d",half,disk));
  auto *PCBCu = new TGeoTubeSeg(Form("PCBCu_H%d_D%d",half,disk), 0, mPCBRad[disk], mCuThickness, mPhi0, mPhi1);
  auto *PCBFR4 = new TGeoTubeSeg(Form("PCBFR4_H%d_D%d",half,disk), 0, mPCBRad[disk], mFR4Thickness, mPhi0, mPhi1);

  
  // Cutting boxes
  //Info("Create",Form("Cutting Boxes PCB_H%d_D%d", half,disk),0,0);
  for(Int_t cut = 0 ; cut<mNumberOfBoxCuts[disk]; cut++){
    auto *boxName =  Form("BoxCut_%d_H%d_D%d",cut, half, disk);
    auto *boxCSName = Form("BoxCS_%d_H%d_D%d",cut, half, disk);
    mSomeBox = new TGeoBBox(boxName,mBoxCuts[disk][cut][0],mBoxCuts[disk][cut][1], mFR4Thickness+20*mT_delta); // TODO: Adjust thickness
    mSomeTranslation = new TGeoTranslation(mBoxCuts[disk][cut][2],mBoxCuts[disk][cut][3], 0.);
    //The first subtraction needs a shape, the base tube
    if (cut == 0)  mSomeSubtraction = new TGeoSubtraction(PCBCu, mSomeBox, NULL,mSomeTranslation);
      else    mSomeSubtraction = new TGeoSubtraction(mPCBCu, mSomeBox, NULL,mSomeTranslation);
    mPCBCu = new TGeoCompositeShape(boxCSName, mSomeSubtraction);

    if (cut == 0)  mSomeSubtraction = new TGeoSubtraction(PCBFR4, mSomeBox, NULL,mSomeTranslation);
      else    mSomeSubtraction = new TGeoSubtraction(mPCBFR4, mSomeBox, NULL,mSomeTranslation);
    mPCBFR4 = new TGeoCompositeShape(boxCSName, mSomeSubtraction);

  }

  // =================  Holes ==================

  // ======= Prepare PCB volume and add to HalfDisk =========

  auto *PCB_Cu_vol = new TGeoVolume(Form("PCBCu_H%d_D%d",half,disk), mPCBCu, mPCBMediumCu);
  auto *PCB_FR4_vol = new TGeoVolume(Form("PCBFR4_H%d_D%d",half,disk), mPCBFR4, mPCBMediumFR4);
 // auto *tr1 = new TGeoTranslation(0., 0., 10.);
  auto *rot1 = new TGeoRotation("rot",0,0,180.);
  auto *rot2 = new TGeoRotation("rot",180.,0.,0.);
  auto *tr_rot1_Cu = new TGeoCombiTrans (0., 0., 1., rot1);
  auto *tr_rot1_FR4 = new TGeoCombiTrans (0., 0., 1.5, rot1);
  auto *tr_rot2_Cu = new TGeoCombiTrans (0., 0., -1., rot2);
  auto *tr_rot2_FR4 = new TGeoCombiTrans (0., 0., -1.5, rot2);
  mHalfDisk->AddNode(PCB_Cu_vol, 0,tr_rot1_Cu);
  mHalfDisk->AddNode(PCB_FR4_vol, 0,tr_rot1_FR4);
  mHalfDisk->AddNode(PCB_Cu_vol, 0,tr_rot2_Cu);
  mHalfDisk->AddNode(PCB_FR4_vol, 0,tr_rot2_FR4);
  return mHalfDisk;

}

//_____________________________________________________________________________
void PCBSupport::initParameters()
{

  mPCBMediumCu  = gGeoManager->GetMedium("MFT_Cu$");
  mPCBMediumFR4  = gGeoManager->GetMedium("MFT_FR4$");

  // # PCB parametrization =====
  // ================================================
  // ## Cut boxes (squares)
  // ### halfDisks 00
  mNumberOfBoxCuts[0]=7;
  // Cut boxes {Width, Height, x_center, y_center}
  mBoxCuts[00] = new Double_t[mNumberOfBoxCuts[0]][4]{
    {mPCBRad[0]+mT_delta, mDiskGap, 0., 0.},
    {sqrt(pow(mPCBRad[0],2.)-pow(mOuterCut[0],2.)),
      (mPCBRad[0]-mOuterCut[0])/2.,
      0.,
      (mPCBRad[0]+mOuterCut[0])/2.}, //External cut width: 2*sqrt(R²-x²)
    {12.4,   6.91,   0.,     0. },
    { 7.95,  9.4,    0.,     0. },
    { 2.9,  11.885,  0.,     0. },
    {1.3875, 1.45,  16.1875, 7.9},
    {1.3875, 1.45, -16.1875, 7.9}
  };

  // ### halfDisks 01
  mNumberOfBoxCuts[1]=mNumberOfBoxCuts[0];
  mBoxCuts[01]=mBoxCuts[00];

  // ### halfDisk 02
  mNumberOfBoxCuts[2]=9;
  mBoxCuts[02] = new Double_t[mNumberOfBoxCuts[2]][4]{
    {mPCBRad[2]+mT_delta, mDiskGap, 0., 0.},
    {sqrt(pow(mPCBRad[2],2.)-pow(mOuterCut[2],2.)),
      (mPCBRad[2]-mOuterCut[2])/2.,
      0.,
      (mPCBRad[2]+mOuterCut[2])/2.}, //External cut width: 2*sqrt(R²-x²)
    {12.8,   6.91,   0.,    0.},
    { 9.7 , 9.4,    0.,    0.},
    {(6.3-2.2)/2, 12.4,  (6.3+2.2)/2, 0},
    { 2.2+mT_delta,  11.9,  0.,    0.},
    {(6.3-2.2)/2, 12.4, - (6.3+2.2)/2, 0},
    {(mPCBRad[2]-14.8)/2, (10.0-6.5)/2, (mPCBRad[2]+14.8)/2, (10.0+6.5)/2},
    {(mPCBRad[2]-14.8)/2, (10.0-6.5)/2, -(mPCBRad[2]+14.8)/2, (10.0+6.5)/2}
  };

  // ### halfDisk 03
  mNumberOfBoxCuts[3]=8;
  mBoxCuts[03] = new Double_t[mNumberOfBoxCuts[3]][4]{
    {mPCBRad[3]+mT_delta, mDiskGap, 0., 0.},
    {sqrt(pow(mPCBRad[3],2.)-pow(mOuterCut[3],2.)),
      (mPCBRad[3]-mOuterCut[3])/2.,
      0.,
      (mPCBRad[3]+mOuterCut[3])/2.}, //External cut width: 2*sqrt(R²-x²)
    {15.7,  9.4, 0., 0.},
    { 9.7, 12.4, 0., 0.},
    { 4.6, 14.73, 0., 0.},
    { 2.9, 16.0 , 0. ,0.},
    {(mPCBRad[3]-18.3)/2., 4.2,(mPCBRad[3]+18.3)/2 , 0},
    {(mPCBRad[3]-18.3)/2., 4.2,-(mPCBRad[3]+18.3)/2 , 0}
  };


  // ### halfDisk 04
  mNumberOfBoxCuts[4]=8;
  mBoxCuts[04] = new Double_t[mNumberOfBoxCuts[4]][4]{
    {mPCBRad[4]+mT_delta, mDiskGap, 0., 0.},
    {sqrt(pow(mPCBRad[4],2.)-pow(mOuterCut[4],2.)),
      (mPCBRad[4]-mOuterCut[4])/2.,
      0.,
      (mPCBRad[4]+mOuterCut[4])/2.}, //External cut width: 2*sqrt(R²-x²)
    {16.2 , 9.4,   0.,   0.},
    {11.4 , 12.4,  0.,   0.},
    { 8.0 , 15.35, 0.,   0.},
    { 2.9 , 16.4,  0.,   0.},
    { 2.35, 4.2, -20.65, 0.},
    { 2.35, 4.2,  20.65, 0.}
  };


}
