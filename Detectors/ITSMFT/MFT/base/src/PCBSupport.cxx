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
mPCBThickness(.8),
mPhi0(0.),
mPhi1(180.),
mT_delta(0.001),
mOuterCut{15.5,15.5,16.9,20.5,21.9}
//mRaisedBoxHeight(0.305),
//mFixBoxHeight(1.41),
//mRad_M2(.156657/2.), // TODO: Check Radius of M2 holes
//mHeight_M2(.6/2), // Height of M2 holes on raised boxes
//mRad_D2_h(.2/2.),
//mHeight_D2_h(.4/2),
//mTwoHoles(2), // Number of D6.5 mm Holes in each halfDisk support
//mD65(.65/2.), //Radius
//mD6(.6/2.), // Radius
//mD8(.8/2.), // Radius
//mD3(.3/2.), // Radius
//mM3(.3/2.), // Radius   TODO: Verify this!
//mD45(.45/2.), // Radius
//mD2(.2/2.), // Radius
//mHeight_D2(.4/2.)
{
  // Support dimentions obtained from Meriadeg Guillamet's blueprints
  // from https://twiki.cern.ch/twiki/bin/viewauth/ALICE/MFTWP3

  initParameters();

}

//_____________________________________________________________________________
TGeoVolumeAssembly* PCBSupport::create(Int_t half, Int_t disk)
{

  Info("Create",Form("Creating PCB_H%d_D%d", half,disk),0,0);
  mHalfDisk = new TGeoVolumeAssembly(Form("Support_H%d_D%d", half,disk));
  auto *support = new TGeoVolumeAssembly(Form("PCB_VA_H%d_D%d",half,disk));
  auto *base = new TGeoTubeSeg(Form("Base_H%d_D%d",half,disk), 0, mPCBRad[disk], mPCBThickness/10., mPhi0, mPhi1);

  // Cutting boxes
  //Info("Create",Form("Cutting Boxes PCB_H%d_D%d", half,disk),0,0);
  for(Int_t cut = 0 ; cut<mNumberOfBoxCuts[disk]; cut++){
    auto *boxName =  Form("BoxCut_%d_H%d_D%d",cut, half, disk);
    auto *boxCSName = Form("BoxCS_%d_H%d_D%d",cut, half, disk);
    mSomeBox = new TGeoBBox(boxName,mBoxCuts[disk][cut][0],mBoxCuts[disk][cut][1],  mPCBThickness/2.+20*mT_delta);
    mSomeTranslation = new TGeoTranslation(mBoxCuts[disk][cut][2],mBoxCuts[disk][cut][3], 0.);
    //The first subtraction needs a shape, the base tube
    if (cut ==0)  mSomeSubtraction = new TGeoSubtraction(base, mSomeBox, NULL,mSomeTranslation);
      else    mSomeSubtraction = new TGeoSubtraction(mSomeCS, mSomeBox, NULL,mSomeTranslation);
    mSomeCS = new TGeoCompositeShape(boxCSName, mSomeSubtraction);
  }

  // =================  Holes ==================

  // ======= Prepare PCB volume and add to HalfDisk =========

  auto *PCB_vol = new TGeoVolume(Form("PCB_H%d_D%d",half,disk), mSomeCS, mSupportMedium);
 // auto *tr1 = new TGeoTranslation(0., 0., 10.);
  auto *rot = new TGeoRotation("rot",0,0,180);
  auto *tr_rot = new TGeoCombiTrans (0., 0., 1., rot);
  mHalfDisk->AddNode(PCB_vol, 0,tr_rot);
  return mHalfDisk;

}

//_____________________________________________________________________________
void PCBSupport::initParameters()
{

  mSupportMedium  = gGeoManager->GetMedium("MFT_PEEK$");

  // # PCB parametrization =====
  // TODO: Add real values for halfDisks 02 to 04

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
