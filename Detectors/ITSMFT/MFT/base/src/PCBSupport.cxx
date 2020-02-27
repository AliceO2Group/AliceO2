// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PCBSupport.cxx
/// \brief Class building the MFT PCB Supports
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \author Rafael Pezzi <rafael.pezzi@cern.ch>

#include "MFTBase/Constants.h"
#include "MFTBase/PCBSupport.h"
#include "MFTBase/Geometry.h"

using namespace o2::mft;

/// \cond CLASSIMP
ClassImp(o2::mft::PCBSupport)
  /// \endcond

  //_____________________________________________________________________________
  PCBSupport::PCBSupport() : mHalfDisk(nullptr),
                             mDiskGap(1.4),
                             mPCBRad{17.5, 17.5, 17.5, 23.0, 23.0},
                             mCuThickness(0.0124),
                             mFR4Thickness(.113),
                             mPhi0(0.),
                             mPhi1(180.),
                             mT_delta(0.01)
//mOuterCut{15.5,15.5,16.9,20.5,21.9}
{

  initParameters();
}

//_____________________________________________________________________________
TGeoVolumeAssembly* PCBSupport::create(Int_t half, Int_t disk)
{

  //Info("Create", Form("Creating PCB_H%d_D%d", half, disk), 0, 0);
  mHalfDisk = new TGeoVolumeAssembly(Form("PCB_H%d_D%d", half, disk));
  //  auto *PCB = new TGeoVolumeAssembly(Form("PCB_VA_H%d_D%d",half,disk));
  auto* PCBCu = new TGeoTubeSeg(Form("PCBCu_H%d_D%d", half, disk), 0, mPCBRad[disk], mCuThickness / 2., mPhi0, mPhi1);
  auto* PCBFR4 = new TGeoTubeSeg(Form("PCBFR4_H%d_D%d", half, disk), 0, mPCBRad[disk], mFR4Thickness / 2., mPhi0, mPhi1);

  // Cutting boxes
  //Info("Create",Form("Cutting Boxes PCB_H%d_D%d", half,disk),0,0);
  for (auto cut = 0; cut < mNumberOfBoxCuts[disk]; cut++) {
    auto* boxName = Form("PCBBoxCut_%d_H%d_D%d", cut, half, disk);
    auto* boxCSName = Form("PCBBoxCS_%d_H%d_D%d", cut, half, disk);
    mSomeBox = new TGeoBBox(boxName, mBoxCuts[disk][cut][0] / 2. + 2 * mT_delta, mBoxCuts[disk][cut][1] / 2. + 2 * mT_delta, mFR4Thickness + mT_delta);
    mSomeTranslation = new TGeoTranslation(mBoxCuts[disk][cut][2], mBoxCuts[disk][cut][3], 0.);
    //The first subtraction needs a shape, the base tube
    if (cut == 0)
      mSomeSubtraction = new TGeoSubtraction(PCBCu, mSomeBox, nullptr, mSomeTranslation);
    else
      mSomeSubtraction = new TGeoSubtraction(mPCBCu, mSomeBox, nullptr, mSomeTranslation);
    mPCBCu = new TGeoCompositeShape(boxCSName, mSomeSubtraction);

    if (cut == 0)
      mSomeSubtraction = new TGeoSubtraction(PCBFR4, mSomeBox, nullptr, mSomeTranslation);
    else
      mSomeSubtraction = new TGeoSubtraction(mPCBFR4, mSomeBox, nullptr, mSomeTranslation);
    mPCBFR4 = new TGeoCompositeShape(boxCSName, mSomeSubtraction);
  }

  //Info("Create",Form("Cutting Boxes PCB_H%d_D%d", half,disk),0,0);
  if (mNumberOfBoxAdd[disk] != 0)
    for (auto iBox = 0; iBox < mNumberOfBoxAdd[disk]; iBox++) {
      auto* boxName = Form("PCBBoxAdd_%d_H%d_D%d", iBox, half, disk);
      auto* boxCSName = Form("PCBBoxAddCS_%d_H%d_D%d", iBox, half, disk);
      mSomeBox = new TGeoBBox(boxName, mBoxAdd[disk][iBox][0] / 2., mBoxAdd[disk][iBox][1] / 2., mFR4Thickness / 2.);
      mSomeTranslation = new TGeoTranslation(mBoxAdd[disk][iBox][2], mBoxAdd[disk][iBox][3], 0.);
      mSomeUnion = new TGeoUnion(mPCBFR4, mSomeBox, nullptr, mSomeTranslation);
      mPCBFR4 = new TGeoCompositeShape(boxCSName, mSomeUnion);
      mSomeBox = new TGeoBBox(boxName, mBoxAdd[disk][iBox][0] / 2., mBoxAdd[disk][iBox][1] / 2., mCuThickness / 2.);
      mSomeUnion = new TGeoUnion(mPCBCu, mSomeBox, nullptr, mSomeTranslation);
      mPCBCu = new TGeoCompositeShape(boxCSName, mSomeUnion);
    }

  // =================  Holes ==================

  // Digging Holes
  //Info("Create",Form("Cutting holes PCB_H%d_D%d", half,disk),0,0);
  for (auto iHole = 0; iHole < mNumberOfHoles[disk]; iHole++) {
    auto* tubeName = Form("PCBHole_%d_H%d_D%d", iHole, half, disk);
    auto* tubeCSName = Form("PCBHoleCS_%d_H%d_D%d", iHole, half, disk);
    mSomeTube = new TGeoTube(tubeName, 0, mHoles[disk][iHole][0] / 2.0, mFR4Thickness + 10 * mT_delta); // TODO: Adjust thickness
    mSomeTranslation = new TGeoTranslation(mHoles[disk][iHole][1], mHoles[disk][iHole][2], 0.);
    mSomeSubtraction = new TGeoSubtraction(mPCBCu, mSomeTube, nullptr, mSomeTranslation);
    mPCBCu = new TGeoCompositeShape(tubeCSName, mSomeSubtraction);
    mSomeSubtraction = new TGeoSubtraction(mPCBFR4, mSomeTube, nullptr, mSomeTranslation);
    mPCBFR4 = new TGeoCompositeShape(tubeCSName, mSomeSubtraction);
  }

  // ======= Prepare PCB volume and add to HalfDisk =========

  auto* PCB_Cu_vol = new TGeoVolume(Form("PCBCu_H%d_D%d", half, disk), mPCBCu, mPCBMediumCu);
  auto* PCB_FR4_vol = new TGeoVolume(Form("PCBFR4_H%d_D%d", half, disk), mPCBFR4, mPCBMediumFR4);
  // auto *tr1 = new TGeoTranslation(0., 0., 10.);
  auto* rot1 = new TGeoRotation("rot", 180, 0, 0);
  auto* rot2 = new TGeoRotation("rot", 180., 180., 180.);
  auto* tr_rot1_Cu = new TGeoCombiTrans(0., 0., 0.4 + mFR4Thickness + mCuThickness / 2., rot1);
  auto* tr_rot1_FR4 = new TGeoCombiTrans(0., 0., 0.4 + mFR4Thickness / 2., rot1);
  auto* tr_rot2_Cu = new TGeoCombiTrans(0., 0., -(0.4 + mFR4Thickness + mCuThickness / 2.), rot2);
  auto* tr_rot2_FR4 = new TGeoCombiTrans(0., 0., -(0.4 + mFR4Thickness / 2.), rot2);
  mHalfDisk->AddNode(PCB_Cu_vol, 0, tr_rot1_Cu);
  mHalfDisk->AddNode(PCB_FR4_vol, 0, tr_rot1_FR4);
  mHalfDisk->AddNode(PCB_Cu_vol, 0, tr_rot2_Cu);
  mHalfDisk->AddNode(PCB_FR4_vol, 0, tr_rot2_FR4);
  return mHalfDisk;
}

//_____________________________________________________________________________
void PCBSupport::initParameters()
{

  mPCBMediumCu = gGeoManager->GetMedium("MFT_Cu$");
  mPCBMediumFR4 = gGeoManager->GetMedium("MFT_FR4$");

  // # PCB parametrization =====
  // ================================================
  // ## Cut boxes (squares)
  // ### PCB 00
  mNumberOfBoxCuts[0] = 10;
  // Cut boxes {Width, Height, x_center, y_center}
  mBoxCuts[00] = new Double_t[mNumberOfBoxCuts[0]][4]{
    {35.0, 8.91, 0., 4.455},
    {15.9, 0.49, 0., 9.155},
    {15.0, 2.52, 0.25, 10.66},
    {4.8, 2.1, 0.25, 12.97},
    {8.445, 1.4, 0., 16.8},
    {4.2775, 2., -6.36125, 16.5},
    {4.2775, 2., 6.36125, 16.5},
    {1.0, 1.0, -15.2, 9.41},
    {1.0, 1.0, 15.2, 9.41},
    {0.3, 0.3, -14.0, 9.5}};

  // ### PCB 01
  mNumberOfBoxCuts[1] = mNumberOfBoxCuts[0];
  mBoxCuts[01] = mBoxCuts[00];

  // ### PCB 02
  mNumberOfBoxCuts[2] = 9;
  mBoxCuts[02] = new Double_t[mNumberOfBoxCuts[2]][4]{
    {35, 8.91, 0.0, 4.455},
    {19.4, 0.49, 0.0, 9.155},
    {18.4, 2.52, .25, 10.66},
    {12.6, 0.48, 0.0, 12.16},
    {11.6, 1.62, 0.25, 13.21},
    {3.1, 0.91, 4.5, 14.475},
    {3.1, 0.91, -4.0, 14.475},
    {0.5, 0.69, 14.95, 9.255},
    {0.5, 0.69, -14.95, 9.255}};

  // ### PCB 03
  mNumberOfBoxCuts[3] = 14;
  mBoxCuts[03] = new Double_t[mNumberOfBoxCuts[3]][4]{
    {46, 5.0, 0, 2.5},
    {32, 5.0, 0, 7.5},
    {26.9, 1.92, -0.6, 10.96},
    {19.4, 0.48, 0, 12.16},
    {18.4, 2.53, 0.25, 13.665},
    {8.2, 1.8, 0.25, 15.83},
    {4.8, 1.4, 0.25, 17.43},
    {1.7, 0.46, -3, 16.96},
    {1.6, 0.6, 9.8, 20.8},
    {1.6, 0.6, -9.8, 20.8},
    {18.0, 1.9, 0, 22.05},
    {1.0, 2, 22.5, 6},
    {1.0, 2, -22.5, 6},
    {0.4, 0.3, -19.5, 10}};

  // ### PCB 04
  mNumberOfBoxCuts[4] = 14;
  mBoxCuts[04] = new Double_t[mNumberOfBoxCuts[4]][4]{
    {46.0, 5.0, 0.0, 2.5},
    {33.0, 5.0, 0.0, 7.5},
    {28.55, 1.92, .225, 10.96},
    {22.8, 0.48, 0.0, 12.16},
    {21.8, 2.53, 0.25, 13.665},
    {16.0, 0.42, 0.0, 15.14},
    {15.0, 2.01, 0.25, 16.355},
    {3.1, 0.58, 6.2, 17.65},
    {4.8, 1.17, 0.25, 17.945},
    {5.1, 0.58, -4.7, 17.65},
    {9.8, 0.5, 0.0, 22.75},
    {1.0, 1.8, 22.5, 5.9},
    {1.0, 1.8, -22.5, 5.9},
    {0.4, 0.3, -19.5, 10}};

  // ## Add boxes (squares)
  // Add boxes {Width, Height, x_center, y_center}
  // ### PCB 00
  mNumberOfBoxAdd[0] = 0;
  mBoxAdd[00] = nullptr;

  // ### PCB 01
  mNumberOfBoxAdd[1] = mNumberOfBoxAdd[0];
  mBoxAdd[01] = mBoxAdd[00];

  // ### PCB 02
  mNumberOfBoxAdd[2] = 4;
  mBoxAdd[02] = new Double_t[mNumberOfBoxAdd[2]][4]{
    {1.51, 2.59, 13.945, 10.205},
    {1.51, 2.59, -13.945, 10.205},
    {13.5, 0.9, 0.0, 16.45},
    {10.2, 0.6, 0.0, 17.2}};

  // ### PCB 03
  mNumberOfBoxAdd[3] = 0;
  mBoxAdd[03] = nullptr;

  // ### PCB 04
  mNumberOfBoxAdd[4] = 1;
  mBoxAdd[04] = new Double_t[mNumberOfBoxAdd[4]][4]{
    {22, 2.339, 0.0, 21.3305}};

  // Holes {Radius, x_center, y_center}
  // ### PCB 00
  mNumberOfHoles[0] = 7;
  mHoles[00] = new Double_t[mNumberOfHoles[0]][3]{
    {.3, 14.0, 9.5},
    {.3, -13.85, 9.5},
    {.3, -14.15, 9.5},
    {1., -11.0, 11.5},
    {1., 11.0, 11.5},
    {.35, -9.0, 11.5},
    {.35, 9.0, 11.5}};

  // ### PCB 01
  mNumberOfHoles[1] = mNumberOfHoles[0];
  mHoles[01] = mHoles[00];

  // ### PCB 02
  mNumberOfHoles[2] = 7;
  mHoles[02] = new Double_t[mNumberOfHoles[2]][3]{
    {.35, 12.5, 11.5},
    {.35, -12.5, 11.5},
    {1.0, -11.0, 11.5},
    {1.0, 11.0, 11.5},
    {0.3, 7.5, 14.5},
    {0.3, -7.5, 14.5},
    {0.3, -7.35, 14.5}};

  // ### PCB 03
  mNumberOfHoles[3] = 13;
  mHoles[03] = new Double_t[mNumberOfHoles[3]][3]{
    {0.3, 19.5, 10.0},
    {0.3, -19.7, 10.0},
    {0.3, -19.3, 10.0},
    {0.35, 16.5, 10.0},
    {0.35, -16.5, 10.0},
    {0.35, 6.0, 18.0},
    {0.35, -6.0, 18.0},
    {1.0, 10.0, 19.0},
    {1.0, -10.0, 19.0},
    {.424, 7.899, 20.415},
    {.424, -7.899, 20.415},
    {.424, 0.5, 20.415},
    {.424, -0.5, 20.415}};

  // ### PCB 04
  mNumberOfHoles[4] = 15;
  mHoles[04] = new Double_t[mNumberOfHoles[4]][3]{
    {0.3, 19.5, 10},
    {0.3, -19.7, 10},
    {0.3, -19.3, 10},
    {0.35, 17, 10},
    {0.35, -17, 10},
    {0.35, 9, 17.5},
    {0.35, -9, 17.5},
    {1, 10, 19},
    {1, -10, 19},
    {0.424, 10.098, 21.815},
    {0.424, -10.098, 21.815},
    {0.424, 2.699, 21.815},
    {0.424, -2.699, 21.815},
    {0.424, 1.699, 21.815},
    {0.424, -1.699, 21.815}};
}
