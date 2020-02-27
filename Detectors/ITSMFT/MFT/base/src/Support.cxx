// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Support.cxx
/// \brief Class building the MFT PCB Supports
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \author Rafael Pezzi <rafael.pezzi@cern.ch>

#include "MFTBase/Constants.h"
#include "MFTBase/Support.h"
#include "MFTBase/Geometry.h"

using namespace o2::mft;

/// \cond CLASSIMP
ClassImp(o2::mft::Support)
  /// \endcond

  //_____________________________________________________________________________
  Support::Support() : mHalfDisk(nullptr),
                       mDiskGap(1.4),
                       mSupRad{17.5, 17.5, 17.5, 23.0, 23.0},
                       mSupThickness(.8),
                       mPhi0(0.),
                       mPhi1(180.),
                       mT_delta(0.01),
                       mOuterCut{15.5, 15.5, 16.9, 20.5, 21.9},
                       mRaisedBoxHeight(0.305),
                       mFixBoxHeight(1.41),
                       mRad_M2(.156657 / 2.), // TODO: Check Radius of M2 holes
                       mHeight_M2(.6 / 2),    // Height of M2 holes on raised boxes
                       mRad_D2_h(.2 / 2.),
                       mHeight_D2_h(.4 / 2),
                       mTwoHoles(2),   // Number of D6.5 mm Holes in each halfDisk support
                       mD65(.65 / 2.), //Radius
                       mD6(.6 / 2.),   // Radius
                       mD8(.8 / 2.),   // Radius
                       mD3(.3 / 2.),   // Radius
                       mM3(.3 / 2.),   // Radius   TODO: Verify this!
                       mD45(.45 / 2.), // Radius
                       mD2(.2 / 2.),   // Radius
                       mHeight_D2(.4 / 2.)
{
  // Support dimentions obtained from Meriadeg Guillamet's blueprints
  // from https://twiki.cern.ch/twiki/bin/viewauth/ALICE/MFTWP3

  initParameters();
}

//_____________________________________________________________________________
TGeoVolumeAssembly* Support::create(Int_t half, Int_t disk)
{
  TGeoSubtraction* localSubtraction;
  TGeoBBox* localBox;
  TGeoTube* localTube;
  TGeoUnion* localUnion;
  TGeoTranslation* localTranslation;
  TGeoCompositeShape* localCS = nullptr;

  //Info("Create",Form("Creating Support_H%d_D%d", half,disk),0,0);
  mHalfDisk = new TGeoVolumeAssembly(Form("Support_H%d_D%d", half, disk));
  auto* base = new TGeoTubeSeg(Form("Base_H%d_D%d", half, disk), 0, mSupRad[disk], mSupThickness / 2., mPhi0, mPhi1);

  // Cutting boxes
  //Info("Create",Form("Cutting Boxes Support_H%d_D%d", half,disk),0,0);
  // Using template function to remove boxes
  localCS = serialBoolOperation(base, TGeoBBox(), mDiskBoxCuts[disk], TGeoSubtraction());

  // Adding raisedBoxes
  //Info("Create",Form("Adding raised boxes Support_H%d_D%d", half,disk),0,0);
  localCS = serialBoolOperation(localCS, TGeoBBox(), mDiskRaisedBoxes[disk], TGeoUnion());

  // Adding fixationBoxes
  localCS = serialBoolOperation(localCS, TGeoBBox(), mDiskFixBoxes[disk], TGeoUnion());

  // =================  Holes ==================
  //TODO: Holes pointing the y axis

  // ======= Creating big holes =========
  //Info("Create",Form("Cutting Voids Support_H%d_D%d", half,disk),0,0);
  for (auto iVoid = 0; iVoid < mNumberOfVoids[disk]; iVoid++) {
    TGeoArb8* localArb;
    localArb = new TGeoArb8(Form("sc_void_%d_H%d_D%d", iVoid, half, disk), mSupThickness + 20. * mT_delta);
    for (auto iVertex = 0; iVertex < 4; iVertex++) {
      Double_t* vertex = &mVoidVert[disk][iVoid][iVertex][0];
      localArb->SetVertex(iVertex, vertex[0], mOuterCut[disk] - vertex[1]);
      localArb->SetVertex(iVertex + 4, vertex[0], mOuterCut[disk] - vertex[1]); //Vertexes 4..7 = 0..3
    }
    localCS = new TGeoCompositeShape(nullptr, compositeOperation(localCS, localArb, nullptr, TGeoSubtraction()));
  }

  // ==== M2 6mm deep holes)
  //Info("Create",Form("Cutting M2 6 mm deep holes Support_H%d_D%d", half,disk),0,0);
  localTube = new TGeoTube(Form("sc_tube1_a_H%d_D%d", half, disk), 0, mRad_M2, mHeight_M2 + 6. * mT_delta);
  for (auto iHole = 0; iHole < mNumberOfM2Holes[disk]; iHole++) {
    localTranslation = new TGeoTranslation(-mM2Holes[disk][iHole][0],
                                           mOuterCut[disk] - mM2Holes[disk][iHole][1],
                                           mRaisedBoxHeight + mSupThickness / 2. - mHeight_M2);
    localCS = new TGeoCompositeShape(nullptr, compositeOperation(localCS, localTube, localTranslation, TGeoSubtraction()));

    //For the backside
    localTranslation = new TGeoTranslation(mM2Holes[disk][iHole][0],
                                           mOuterCut[disk] - mM2Holes[disk][iHole][1],
                                           -(mRaisedBoxHeight + mSupThickness / 2. - mHeight_M2));
    localCS = new TGeoCompositeShape(nullptr, compositeOperation(localCS, localTube, localTranslation, TGeoSubtraction()));
  }

  // ==== D2 H7 - 4 mm deep (on raisedBoxes)
  //Info("Create",Form("Cutting D2 mm holes on raisedboxes Support_H%d_D%d", half,disk),0,0);
  localTube = new TGeoTube(Form("sc_tube1_a_H%d_D%d", half, disk), 0, mRad_D2_h, mHeight_D2_h + 6. * mT_delta);

  for (auto iHole = 0; iHole < mNumberOfD2_hHoles[disk]; iHole++) {
    localTranslation = new TGeoTranslation(-mD2_hHoles[disk][iHole][0],
                                           mOuterCut[disk] - mD2_hHoles[disk][iHole][1],
                                           mRaisedBoxHeight + mSupThickness / 2. - mHeight_D2_h);
    localCS = new TGeoCompositeShape(nullptr, compositeOperation(localCS, localTube, localTranslation, TGeoSubtraction()));

    //For the backside
    localTranslation = new TGeoTranslation(mD2_hHoles[disk][iHole][0],
                                           mOuterCut[disk] - mD2_hHoles[disk][iHole][1],
                                           -(mRaisedBoxHeight + mSupThickness / 2. - mHeight_D2_h));
    localCS = new TGeoCompositeShape(nullptr, compositeOperation(localCS, localTube, localTranslation, TGeoSubtraction()));
  }

  // ==== D6.5 H7 (6.5 mm diameter holes)
  //Info("Create",Form("Cutting 6.5 holes Support_H%d_D%d", half,disk),0,0);
  localTube = new TGeoTube(Form("D65tube_H%d_D%d", half, disk), 0, mD65, mSupThickness / 2. + 20. * mT_delta);
  for (auto iHole = 0; iHole < mTwoHoles; iHole++) {
    localTranslation = new TGeoTranslation(-mD65Holes[disk][iHole][0],
                                           mOuterCut[disk] - mD65Holes[disk][iHole][1],
                                           0.);
    localCS = new TGeoCompositeShape(nullptr, compositeOperation(localCS, localTube, localTranslation, TGeoSubtraction()));
  }

  // ==== D6 H7 (6 mm diameter holes)
  //Info("Create",Form("Cutting 6 mm holes Support_H%d_D%d", half,disk),0,0);
  localTube = new TGeoTube(Form("D6tube_H%d_D%d", half, disk), 0, mD6, mSupThickness / 2. + mT_delta);
  for (auto iHole = 0; iHole < mTwoHoles; iHole++) {
    localTranslation = new TGeoTranslation(-mD6Holes[disk][iHole][0],
                                           mOuterCut[disk] - mD6Holes[disk][iHole][1],
                                           0.);
    localCS = new TGeoCompositeShape(nullptr, compositeOperation(localCS, localTube, localTranslation, TGeoSubtraction()));
  }

  // ==== D8 H7 (8 mm diameter holes)
  //Info("Create",Form("Cutting 8 mm holes Support_H%d_D%d", half,disk),0,0);
  localTube = new TGeoTube(Form("D8tube_H%d_D%d", half, disk), 0, mD8, mSupThickness / 2. + mRaisedBoxHeight + 20 * mT_delta);
  for (auto iHole = 0; iHole < mNumberOfD8_Holes[disk]; iHole++) {
    localTranslation = new TGeoTranslation(-mD8Holes[disk][iHole][0],
                                           mOuterCut[disk] - mD8Holes[disk][iHole][1],
                                           0.);
    localCS = new TGeoCompositeShape(nullptr, compositeOperation(localCS, localTube, localTranslation, TGeoSubtraction()));
  }

  // ==== D3 H7 (3 mm diameter holes)
  //Info("Create",Form("Cutting 3 mm holes Support_H%d_D%d", half,disk),0,0);
  localTube = new TGeoTube(Form("D3tube_H%d_D%d", half, disk), 0, mD3, mSupThickness / 2. + mT_delta);
  for (auto iHole = 0; iHole < mTwoHoles; iHole++) {
    localTranslation = new TGeoTranslation(-mD3Holes[disk][iHole][0],
                                           mOuterCut[disk] - mD3Holes[disk][iHole][1],
                                           0.);
    localCS = new TGeoCompositeShape(nullptr, compositeOperation(localCS, localTube, localTranslation, TGeoSubtraction()));
  }

  // ==== M3 H7 (?? mm diameter holes)
  //Info("Create",Form("Cutting M3 H7 holes Support_H%d_D%d", half,disk),0,0);
  localTube = new TGeoTube(Form("M3tube_H%d_D%d", half, disk), 0, mM3, mSupThickness / 2. + mT_delta);
  for (auto iHole = 0; iHole < mNumberOfM3Holes[disk]; iHole++) {
    localTranslation = new TGeoTranslation(-mM3Holes[disk][iHole][0],
                                           mOuterCut[disk] - mM3Holes[disk][iHole][1],
                                           0.);
    localCS = new TGeoCompositeShape(nullptr, compositeOperation(localCS, localTube, localTranslation, TGeoSubtraction()));
  }

  // ==== D4.5 H9
  //Info("Create",Form("Cutting 4.5 mm holes Support_H%d_D%d", half,disk),0,0);
  localTube = new TGeoTube(Form("D45tube_H%d_D%d", half, disk), 0, mD45, mSupThickness / 2. + mT_delta);
  for (auto iHole = 0; iHole < mTwoHoles; iHole++) {
    localTranslation = new TGeoTranslation(-mD45Holes[disk][iHole][0],
                                           mOuterCut[disk] - mD45Holes[disk][iHole][1],
                                           0.);
    localCS = new TGeoCompositeShape(nullptr, compositeOperation(localCS, localTube, localTranslation, TGeoSubtraction()));
  }

  // ==== D2 H7 - 4 mm deep (on lower surface)
  //Info("Create",Form("Cutting D2 holes Support_H%d_D%d", half,disk),0,0);
  localTube = new TGeoTube(Form("D2tube_H%d_D%d", half, disk), 0, mD2, .4 / 2. + 6 * mT_delta);
  for (auto iHole = 0; iHole < mTwoHoles; iHole++) {
    localTranslation = new TGeoTranslation(-mD2Holes[disk][iHole][0],
                                           mOuterCut[disk] - mD2Holes[disk][iHole][1],
                                           mSupThickness / 2. - .4 / 2);
    localCS = new TGeoCompositeShape(nullptr, compositeOperation(localCS, localTube, localTranslation, TGeoSubtraction()));
  }

  // ======= Prepare support volume and add to HalfDisk =========

  auto* support_vol = new TGeoVolume(Form("Support_H%d_D%d", half, disk), localCS, mSupportMedium);

  auto* rot = new TGeoRotation("rot", 0, 0, 180);
  mHalfDisk->AddNode(support_vol, 0, rot);
  return mHalfDisk;
}

//_____________________________________________________________________________
void Support::initParameters()
{

  mSupportMedium = gGeoManager->GetMedium("MFT_PEEK$");
  auto th = mSupThickness;
  // # Support parametrization =====

  // ================================================
  // ## Cut boxes (squares)
  // ### halfDisks 00

  mDiskBoxCuts[0] =
    {
      {mSupRad[0] + mT_delta, mDiskGap, th, 0., 0., 0},
      {sqrt(pow(mSupRad[0], 2.) - pow(mOuterCut[0], 2.)), (mSupRad[0] - mOuterCut[0]) / 2., th, 0., (mSupRad[0] + mOuterCut[0]) / 2., 0}, //External cut width: 2*sqrt(R²-x²)
      {12.4, 6.91, th, 0., 0., 0},
      {7.95, 9.4, th, 0., 0., 0},
      {2.9, 11.885, th, 0., 0., 0},
      {1.3875, 1.45, th, 16.1875, 7.9, 0},
      {1.3875, 1.45, th, -16.1875, 7.9, 0}};

  // ### halfDisks 01
  mDiskBoxCuts[1] = mDiskBoxCuts[0];

  // ### halfDisk 02
  mDiskBoxCuts[2] =
    {
      {mSupRad[2] + mT_delta, mDiskGap, th, 0., 0., 0},
      {sqrt(pow(mSupRad[2], 2.) - pow(mOuterCut[2], 2.)),
       (mSupRad[2] - mOuterCut[2]) / 2., th,
       0.,
       (mSupRad[2] + mOuterCut[2]) / 2., 0}, //External cut width: 2*sqrt(R²-x²)
      {12.8, 6.91, th, 0., 0., 0},
      {9.7, 9.4, th, 0., 0., 0},
      {(6.3 - 2.2) / 2, 12.4, th, (6.3 + 2.2) / 2, 0, 0},
      {2.2 + mT_delta, 11.9, th, 0., 0., 0},
      {(6.3 - 2.2) / 2, 12.4, th, -(6.3 + 2.2) / 2, 0, 0},
      {(mSupRad[2] - 14.8) / 2, (10.0 - 6.5) / 2, th, (mSupRad[2] + 14.8) / 2, (10.0 + 6.5) / 2, 0},
      {(mSupRad[2] - 14.8) / 2, (10.0 - 6.5) / 2, th, -(mSupRad[2] + 14.8) / 2, (10.0 + 6.5) / 2, 0}};

  // ### halfDisk 03
  mDiskBoxCuts[3] =
    {
      {mSupRad[3] + mT_delta, mDiskGap, th, 0., 0., 0},
      {sqrt(pow(mSupRad[3], 2.) - pow(mOuterCut[3], 2.)),
       (mSupRad[3] - mOuterCut[3]) / 2., th,
       0.,
       (mSupRad[3] + mOuterCut[3]) / 2., 0}, //External cut width: 2*sqrt(R²-x²)
      {15.7, 9.4, th, 0., 0., 0},
      {9.7, 12.4, th, 0., 0., 0},
      {4.6, 14.73, th, 0., 0., 0},
      {2.9, 16.0, th, 0., 0., 0},
      {(mSupRad[3] - 18.3) / 2., 4.2, th, (mSupRad[3] + 18.3) / 2, 0, 0},
      {(mSupRad[3] - 18.3) / 2., 4.2, th, -(mSupRad[3] + 18.3) / 2, 0, 0}};

  // ### halfDisk 04
  mDiskBoxCuts[4] =
    {
      {mSupRad[4] + mT_delta, mDiskGap, th, 0., 0., 0},
      {sqrt(pow(mSupRad[4], 2.) - pow(mOuterCut[4], 2.)),
       (mSupRad[4] - mOuterCut[4]) / 2., th,
       0.,
       (mSupRad[4] + mOuterCut[4]) / 2., 0}, //External cut width: 2*sqrt(R²-x²)
      {16.2, 9.4, th, 0., 0., 0},
      {11.4, 12.4, th, 0., 0., 0},
      {8.0, 15.35, th, 0., 0., 0},
      {2.9, 16.4, th, 0., 0., 0},
      {2.35, 4.2, th, -20.65, 0., 0},
      {2.35, 4.2, th, 20.65, 0., 0}};

  // ================================================
  // ## Raised boxes
  // ### halfDisks 00

  auto rBT = mRaisedBoxHeight / 2.;                      // Raised boxes thickness
  auto rBS = mRaisedBoxHeight / 2. + mSupThickness / 2.; //Raised boxes z shift
  // Raised Boxes {Width, Height, Thickness, x_center, y_center, z_center}
  mDiskRaisedBoxes[0] =
    {
      {(9.35 - 7.95) / 2., (8.81 - 6.91) / 2., rBT, (9.35 + 7.95) / 2., (8.81 + 6.91) / 2., rBS},
      {(7.65 - 2.9) / 2., (11.82 - 9.4) / 2., rBT, (7.65 + 2.9) / 2., (11.82 + 9.4) / 2., rBS},
      {(2.55 + 2.05) / 2., (13.92 - 11.885) / 2., rBT, (2.55 - 2.05) / 2., (13.92 + 11.885) / 2., rBS},
      {(7.15 - 2.9) / 2., (11.82 - 9.4) / 2., rBT, (-7.152 - 2.92) / 2., (11.82 + 9.4) / 2, rBS},
      {(10.55 - 7.95) / 2., (8.81 - 6.91) / 2., rBT, (-10.55 - 7.95) / 2., (8.81 + 6.91) / 2., rBS},
      //Back side:
      {(9.35 - 7.95) / 2., (8.81 - 6.91) / 2., rBT, -(9.35 + 7.95) / 2., (8.81 + 6.91) / 2., -rBS},
      {(7.65 - 2.9) / 2., (11.82 - 9.4) / 2., rBT, -(7.65 + 2.9) / 2., (11.82 + 9.4) / 2., -rBS},
      {(2.55 + 2.05) / 2., (13.92 - 11.885) / 2., rBT, -(2.55 - 2.05) / 2., (13.92 + 11.885) / 2., -rBS},
      {(7.15 - 2.9) / 2., (11.82 - 9.4) / 2., rBT, -(-7.152 - 2.92) / 2., (11.82 + 9.4) / 2, -rBS},
      {(10.55 - 7.95) / 2., (8.81 - 6.91) / 2., rBT, -(-10.55 - 7.95) / 2., (8.81 + 6.91) / 2., -rBS}};

  // ### halfDisks 01
  mDiskRaisedBoxes[1] = mDiskRaisedBoxes[0];

  // ### halfDisk 02
  mDiskRaisedBoxes[2] =
    {
      {(10.55 - 9.7) / 2., (8.81 - 6.91) / 2., rBT, (10.55 + 9.7) / 2., (8.81 + 6.91) / 2., rBS},
      {(8.85 - 6.3) / 2., (11.82 - 9.4) / 2., rBT, (8.85 + 6.3) / 2., (11.82 + 9.4) / 2., rBS},
      {(5.45 - 2.55) / 2., (14.83 - 12.4) / 2., rBT, (5.45 + 2.55) / 2., (14.83 + 12.4) / 2., rBS},
      {(2.2 + 2.2) / 2., (13.92 - 11.9) / 2., rBT, (2.2 - 2.2) / 2., (13.92 + 11.9) / 2, rBS},
      {(5.95 - 3.05) / 2., (14.83 - 12.4) / 2., rBT, -(5.95 + 3.05) / 2., (14.83 + 12.4) / 2., rBS},
      {(9.35 - 6.3) / 2., (11.82 - 9.4) / 2., rBT, -(9.35 + 6.3) / 2., (11.82 + 9.4) / 2, rBS},
      {(11.05 - 9.7) / 2., (8.81 - 6.91) / 2., rBT, -(11.05 + 9.7) / 2., (8.81 + 6.91) / 2., rBS},
      //Back side:
      {(10.55 - 9.7) / 2., (8.81 - 6.91) / 2., rBT, -(10.55 + 9.7) / 2., (8.81 + 6.91) / 2., -rBS},
      {(8.85 - 6.3) / 2., (11.82 - 9.4) / 2., rBT, -(8.85 + 6.3) / 2., (11.82 + 9.4) / 2., -rBS},
      {(5.45 - 2.55) / 2., (14.83 - 12.4) / 2., rBT, -(5.45 + 2.55) / 2., (14.83 + 12.4) / 2., -rBS},
      {(2.2 + 2.2) / 2., (13.92 - 11.9) / 2., rBT, -(2.2 - 2.2) / 2., (13.92 + 11.9) / 2, -rBS},
      {(5.95 - 3.05) / 2., (14.83 - 12.4) / 2., rBT, (5.95 + 3.05) / 2., (14.83 + 12.4) / 2., -rBS},
      {(9.35 - 6.3) / 2., (11.82 - 9.4) / 2., rBT, (9.35 + 6.3) / 2., (11.82 + 9.4) / 2, -rBS},
      {(11.05 - 9.7) / 2., (8.81 - 6.91) / 2., rBT, (11.05 + 9.7) / 2., (8.81 + 6.91) / 2., -rBS}};

  // ### halfDisk 03
  mDiskRaisedBoxes[3] =
    {
      {(12.75 - 9.7) / 2., (11.82 - 9.4) / 2., rBT, (12.75 + 9.7) / 2., (11.82 + 9.4) / 2., rBS},
      {(9.35 - 4.6) / 2., (14.83 - 12.4) / 2., rBT, (9.35 + 4.6) / 2., (14.83 + 12.4) / 2., rBS},
      {(4.25 - 2.9) / 2., (16.63 - 14.73) / 2., rBT, (4.25 + 2.9) / 2., (16.63 + 14.73) / 2., rBS},
      {(2.55 + 2.05) / 2., (18.03 - 16.0) / 2., rBT, (2.55 - 2.05) / 2., (18.03 + 16.0) / 2, rBS},
      {(3.75 - 2.9) / 2., (17.09 - 14.73) / 2., rBT, -(3.75 + 2.9) / 2., (17.09 + 14.73) / 2., rBS},
      {(8.85 - 4.6) / 2., (14.83 - 12.4) / 2., rBT, -(8.85 + 4.6) / 2., (14.83 + 12.4) / 2., rBS},
      {(13.95 - 9.7) / 2., (11.82 - 9.4) / 2., rBT, -(13.95 + 9.7) / 2., (11.82 + 9.4) / 2., rBS},
      //for backside:
      {(12.75 - 9.7) / 2., (11.82 - 9.4) / 2., rBT, -(12.75 + 9.7) / 2., (11.82 + 9.4) / 2., -rBS},
      {(9.35 - 4.6) / 2., (14.83 - 12.4) / 2., rBT, -(9.35 + 4.6) / 2., (14.83 + 12.4) / 2., -rBS},
      {(4.25 - 2.9) / 2., (16.63 - 14.73) / 2., rBT, -(4.25 + 2.9) / 2., (16.63 + 14.73) / 2., -rBS},
      {(2.55 + 2.05) / 2., (18.03 - 16.0) / 2., rBT, -(2.55 - 2.05) / 2., (18.03 + 16.0) / 2, -rBS},
      {(3.75 - 2.9) / 2., (17.09 - 14.73) / 2., rBT, (3.75 + 2.9) / 2., (17.09 + 14.73) / 2., -rBS},
      {(8.85 - 4.6) / 2., (14.83 - 12.4) / 2., rBT, (8.85 + 4.6) / 2., (14.83 + 12.4) / 2., -rBS},
      {(13.95 - 9.7) / 2., (11.82 - 9.4) / 2., rBT, (13.95 + 9.7) / 2., (11.82 + 9.4) / 2., -rBS}};

  // ### halfDisk 04
  mDiskRaisedBoxes[4] =
    {
      {(13.9 - 11.4) / 2., (11.82 - 9.4) / 2., rBT, -(13.9 + 11.4) / 2., (11.82 + 9.4) / 2., rBS},        // RB0
      {(10.55 - 8.0) / 2., (14.83 - 12.4) / 2., rBT, -(10.55 + 8.0) / 2., (14.83 + 12.4) / 2., rBS},      // RB1
      {(7.15 - 2.9) / 2., (17.84 - 15.35) / 2., rBT, -(7.15 + 2.9) / 2., (17.84 + 15.35) / 2., rBS},      // RB2
      {(2.05 + 2.55) / 2., (18.45 - 16.4) / 2., rBT, -(2.05 - 2.55) / 2., (18.45 + 16.4) / 2, rBS},       // RB3
      {-(-4.75 + 2.9) / 2., (17.26 - 15.35) / 2., rBT, -(-4.75 - 2.9) / 2., (17.26 + 15.35) / 2., rBS},   // RB4
      {-(-7.65 + 4.75) / 2., (17.85 - 15.35) / 2., rBT, -(-7.65 - 4.75) / 2., (17.85 + 15.35) / 2., rBS}, // RB5
      {-(-11.05 + 8.0) / 2., (14.83 - 12.4) / 2., rBT, -(-11.05 - 8.0) / 2., (14.83 + 12.4) / 2., rBS},   // RB6
      {-(-14.45 + 11.4) / 2., (11.82 - 9.4) / 2., rBT, -(-14.45 - 11.4) / 2., (11.82 + 9.4) / 2., rBS},   // RB7
      //For backside:
      {(13.9 - 11.4) / 2., (11.82 - 9.4) / 2., rBT, (13.9 + 11.4) / 2., (11.82 + 9.4) / 2., -rBS},        // RB0
      {(10.55 - 8.0) / 2., (14.83 - 12.4) / 2., rBT, (10.55 + 8.0) / 2., (14.83 + 12.4) / 2., -rBS},      // RB1
      {(7.15 - 2.9) / 2., (17.84 - 15.35) / 2., rBT, (7.15 + 2.9) / 2., (17.84 + 15.35) / 2., -rBS},      // RB2
      {(2.05 + 2.55) / 2., (18.45 - 16.4) / 2., rBT, (2.05 - 2.55) / 2., (18.45 + 16.4) / 2, -rBS},       // RB3
      {-(-4.75 + 2.9) / 2., (17.26 - 15.35) / 2., rBT, (-4.75 - 2.9) / 2., (17.26 + 15.35) / 2., -rBS},   // RB4
      {-(-7.65 + 4.75) / 2., (17.85 - 15.35) / 2., rBT, (-7.65 - 4.75) / 2., (17.85 + 15.35) / 2., -rBS}, // RB5
      {-(-11.05 + 8.0) / 2., (14.83 - 12.4) / 2., rBT, (-11.05 - 8.0) / 2., (14.83 + 12.4) / 2., -rBS},   // RB6
      {-(-14.45 + 11.4) / 2., (11.82 - 9.4) / 2., rBT, (-14.45 - 11.4) / 2., (11.82 + 9.4) / 2., -rBS}    // RB7
    };

  // ================================================
  // ## Fixation boxes
  // ### halfDisks 00
  // Fixation Boxes {Width, Height, Thickness, x_center, y_center, z_center = 0}
  mDiskFixBoxes[0] = {
    {(16.8 - 14.8) / 2., (6.5 - 4.6) / 2., mFixBoxHeight / 2., (16.8 + 14.8) / 2., (6.5 + 4.6) / 2., 0},
    //Other side:
    {(16.8 - 14.8) / 2., (6.5 - 4.6) / 2., mFixBoxHeight / 2., -(16.8 + 14.8) / 2., (6.5 + 4.6) / 2., 0}};

  // ### halfDisks 01
  mDiskFixBoxes[1] = mDiskFixBoxes[0];

  // ### halfDisk 02
  mDiskFixBoxes[2] = {
    {(16.8 - 14.8) / 2., (6.5 - 4.6) / 2., mFixBoxHeight / 2., (16.8 + 14.8) / 2., (6.5 + 4.6) / 2., 0},
    //Other side:
    {(16.8 - 14.8) / 2., (6.5 - 4.6) / 2., mFixBoxHeight / 2., -(16.8 + 14.8) / 2., (6.5 + 4.6) / 2., 0}};

  // ### halfDisk 03
  mDiskFixBoxes[3] = {
    {(25.6 - 24.5) / 2., (6.5 - 5.2) / 2., mFixBoxHeight / 2., (25.6 + 24.5) / 2., (6.5 + 5.2) / 2., 0},
    {(24.5 - 23.6) / 2., (6.5 - 4.2) / 2., mFixBoxHeight / 2., (24.5 + 23.6) / 2., (6.5 + 4.2) / 2., 0},
    {(23.6 - 22.0) / 2., (6.5 - 4.2) / 2., mSupThickness / 2., (23.6 + 22.0) / 2., (6.5 + 4.2) / 2., 0},
    //Other side:
    {(25.6 - 24.5) / 2., (6.5 - 5.2) / 2., mFixBoxHeight / 2., -(25.6 + 24.5) / 2., (6.5 + 5.2) / 2., 0},
    {(24.5 - 23.6) / 2., (6.5 - 4.2) / 2., mFixBoxHeight / 2., -(24.5 + 23.6) / 2., (6.5 + 4.2) / 2., 0},
    {(23.6 - 22.0) / 2., (6.5 - 4.2) / 2., mSupThickness / 2., -(23.6 + 22.0) / 2., (6.5 + 4.2) / 2., 0}};

  // ### halfDisk 04
  mDiskFixBoxes[4] = {
    {(25.6 - 24.5) / 2., (6.5 - 5.2) / 2., mFixBoxHeight / 2., (25.6 + 24.5) / 2., (6.5 + 5.2) / 2., 0},
    {(24.5 - 23.6) / 2., (6.5 - 4.2) / 2., mFixBoxHeight / 2., (24.5 + 23.6) / 2., (6.5 + 4.2) / 2., 0},
    {(23.6 - 22.0) / 2., (6.5 - 4.2) / 2., mSupThickness / 2., (23.6 + 22.0) / 2., (6.5 + 4.2) / 2., 0},
    //Other side:
    {(25.6 - 24.5) / 2., (6.5 - 5.2) / 2., mFixBoxHeight / 2., -(25.6 + 24.5) / 2., (6.5 + 5.2) / 2., 0},
    {(24.5 - 23.6) / 2., (6.5 - 4.2) / 2., mFixBoxHeight / 2., -(24.5 + 23.6) / 2., (6.5 + 4.2) / 2., 0},
    {(23.6 - 22.0) / 2., (6.5 - 4.2) / 2., mSupThickness / 2., -(23.6 + 22.0) / 2., (6.5 + 4.2) / 2., 0}};

  // ================================================
  // ## Big holes (Voids)
  //  Description only needed for one side of the disk.
  //   The other side is taken as a reflection on the y axis.
  //   4 vertices are required for each hole that will be used
  //   on TGeoArb8 shapes.

  // ### halfdisk 00
  mNumberOfVoids[0] = 3; //Number of Voids (big holes) in each halfDisk support
  mVoidVert[0] = new Double_t[mNumberOfVoids[0]][4][2]{
    {{.01, .01}, {-.01, .01}, {-.01, -.01}, {.01, -.01}}, //
    {{.01, .01}, {-.01, .01}, {-.01, -.01}, {.01, -.01}}, //
    {{.01, .01}, {-.01, .01}, {-.01, -.01}, {.01, -.01}}  //
  };

  // ### halfdisk 01
  mNumberOfVoids[1] = mNumberOfVoids[0];
  mVoidVert[1] = mVoidVert[0];

  // ### halfdisk 02
  mNumberOfVoids[2] = 3;
  mVoidVert[2] = new Double_t[mNumberOfVoids[2]][4][2]{
    {{.01, .01}, {-.01, .01}, {-.01, -.01}, {.01, -.01}}, //
    {{.01, .01}, {-.01, .01}, {-.01, -.01}, {.01, -.01}}, //
    {{.01, .01}, {-.01, .01}, {-.01, -.01}, {.01, -.01}}  //
  };

  // ### halfdisk 03
  mNumberOfVoids[3] = 6; //Number of Voids (big holes) in each halfDisk support
  mVoidVert[3] = new Double_t[mNumberOfVoids[3]][4][2]{
    {{-21.377, 15.3}, {-20.075, 11.5}, {-17.0, 11.5}, {-17.0, 15.3}}, // a,b,c,d
    {{-19.053, 9.5}, {-13.964, 3.5}, {-14.0, 7.5}, {-14.0, 9.5}},     // e,f,i,j
    {{-13.964, 3.5}, {-10.0, 3.5}, {-10.0, 7.5}, {-14.0, 7.5}},       // f,g,h,i,
    {{21.377, 15.3}, {17.0, 15.3}, {17.0, 11.5}, {20.075, 11.5}},     // s,t,q,r
    {{19.053, 9.5}, {14.0, 9.5}, {14.0, 7.5}, {13.964, 3.5}},         // m,n,o,l
    {{13.964, 3.5}, {14.0, 7.5}, {10.0, 7.5}, {10.0, 3.5}}            // l,o,p,k
  };

  // ### halfdisk 04
  mNumberOfVoids[4] = 6; //Number of Voids (big holes) in each halfDisk support
  mVoidVert[4] = new Double_t[mNumberOfVoids[4]][4][2]{
    {{-21.377, 16.7}, {-20.075, 12.9}, {-17.0, 12.9}, {-17.0, 16.7}},                         // a,b,c,d
    {{-19.053, 10.9}, {-13.964, 4.9}, {-15.0, 8.9}, {-15.0, 10.9}},                           // e,f,i,j
    {{-13.964 - 6 * mT_delta, 4.9}, {-11.5, 4.9}, {-11.5, 8.9}, {-15.0 - 6 * mT_delta, 8.9}}, // f,g,h,i,
    {{21.377, 16.7}, {17.0, 16.7}, {17.0, 12.9}, {20.075, 12.9}},                             // s,t,q,r
    {{19.053, 10.9}, {15.0, 10.9}, {15.0, 8.9}, {13.964, 4.9}},                               // m,n,o,l
    {{13.964 + 6 * mT_delta, 4.9}, {15.0 + 6 * mT_delta, 8.9}, {11.5, 8.9}, {11.5, 4.9}}      // l,o,p,k,
  };

  // ================================================
  // ## M2 6mm deep
  // ### halfDisk00
  mNumberOfM2Holes[0] = 12;
  mM2Holes[0] = new Double_t[mNumberOfM2Holes[0]][2]{
    {-8.75, 7.29},  // #16
    {-7.05, 4.275}, // #18
    {-5.35, 4.275}, // #20
    {-3.65, 4.275}, // #22
    {-1.95, 2.429}, // #24
    {-0.25, 2.175}, // #26
    {1.45, 2.269},  // #28
    {3.15, 4.275},  // #30
    {4.85, 4.275},  // #32
    {6.55, 4.275},  // #34
    {8.25, 7.29},   // #36
    {9.95, 7.29}    // #38
  };

  // ### halfDisk01
  mNumberOfM2Holes[1] = mNumberOfM2Holes[0];
  mM2Holes[1] = mM2Holes[0];

  // ### halfDisk02
  mNumberOfM2Holes[2] = 13;
  mM2Holes[2] = new Double_t[mNumberOfM2Holes[2]][2]{
    {-10.45, 8.69}, // 16
    {-8.75, 5.675}, // 18
    {-7.05, 5.675}, // 20
    {-5.35, 2.66},  // 22
    {-3.65, 2.66},  // 24
    {-1.95, 3.825}, // 26
    {-0.25, 3.575}, // 28
    {1.45, 3.665},  // 30
    {3.15, 2.66},   // 32
    {4.85, 2.66},   // 34
    {6.55, 5.675},  // 36
    {8.25, 5.675},  // 38
    {9.95, 8.69}    // 40
  };

  // ### halfDisk03
  mNumberOfM2Holes[3] = 16;
  mM2Holes[3] = new Double_t[mNumberOfM2Holes[3]][2]{
    {-12.15, 9.275}, // P
    {-10.45, 9.275}, // R
    {-8.75, 6.26},   // T
    {-7.05, 6.26},   // V
    {-5.35, 6.26},   // X
    {-3.65, 4.46},   // Z
    {-1.95, 3.225},  // BB
    {-0.25, 3.06},   // DD
    {1.45, 3.12},    // FF
    {3.15, 3.99},    // HH
    {4.85, 6.62},    // JJ
    {6.65, 6.62},    // LL
    {8.25, 6.62},    // NN
    {9.95, 9.275},   // PP
    {11.65, 9.275},  // RR
    {13.35, 9.275}   // TT
  };

  // ### halfDisk04
  mNumberOfM2Holes[4] = 17;
  mM2Holes[4] = new Double_t[mNumberOfM2Holes[4]][2]{
    {-13.85, 10.675}, // P
    {-12.15, 10.675}, // R
    {-10.45, 7.66},   // T
    {-8.75, 7.66},    // V
    {-7.05, 4.645},   // X
    {-5.35, 4.645},   // Z
    {-3.65, 5.235},   // BB
    {-1.95, 4.205},   // DD
    {-0.25, 4.06},    // FF
    {1.45, 4.115},    // HH
    {3.15, 4.86},     // JJ
    {4.85, 4.645},    // LL
    {6.55, 4.645},    // NN
    {8.25, 7.66},     // PP
    {9.95, 7.66},     // RR
    {11.65, 10.675},  // TT
    {13.35, 10.675}   // VV
  };

  // ================================================
  // ## D2 H7 - 4 mm deep (on raisedBoxes)
  // ### halfDisk00
  mNumberOfD2_hHoles[0] = 12;
  mD2_hHoles[0] = new Double_t[mNumberOfD2_hHoles[0]][2]{{-8.75, 8.09},  // #15
                                                         {-7.05, 5.075}, // #17
                                                         {-5.35, 5.075}, // #19
                                                         {-3.65, 5.075}, // #21
                                                         {-1.95, 3.229}, // #23
                                                         {-0.25, 2.975}, // #25
                                                         {1.45, 3.069},  // #27
                                                         {3.15, 5.075},  // #29
                                                         {4.85, 5.075},  // #31
                                                         {6.55, 5.075},  // #33
                                                         {8.25, 8.09},   // #35
                                                         {9.95, 8.09}};

  // ### halfDisk01
  mNumberOfD2_hHoles[1] = mNumberOfD2_hHoles[0];
  mD2_hHoles[1] = mD2_hHoles[0];

  // ### halfDisk02
  mNumberOfD2_hHoles[2] = 13;
  mD2_hHoles[2] = new Double_t[mNumberOfD2_hHoles[2]][2]{
    {-10.45, 9.49}, // 15
    {-8.75, 6.475}, // 17
    {-7.05, 6.475}, // 19
    {-5.35, 3.46},  // 21
    {-3.65, 3.46},  // 23
    {-1.95, 4.625}, // 25
    {-0.25, 4.375}, // 27
    {1.45, 4.465},  // 29
    {3.15, 3.46},   // 31
    {4.85, 3.46},   // 33
    {6.55, 6.475},  // 35
    {8.25, 6.475},  // 37
    {9.95, 9.49}    // 39
  };

  // ### halfDisk03
  mNumberOfD2_hHoles[3] = 16;
  mD2_hHoles[3] = new Double_t[mNumberOfD2_hHoles[3]][2]{
    {-12.15, 10.075}, // O
    {-10.45, 10.075}, // Q
    {-8.75, 7.06},    // S
    {-7.05, 7.06},    // T
    {-5.35, 7.06},    // W
    {-3.65, 5.26},    // Y
    {-1.95, 4.025},   // AA
    {-0.25, 3.86},    // CC
    {1.45, 3.92},     // EE
    {3.15, 4.79},     // GG
    {4.85, 7.06},     // II
    {6.65, 7.06},     // KK
    {8.25, 7.06},     // MM
    {9.95, 10.075},   // OO
    {11.65, 10.075},  // QQ
    {13.35, 10.075}   // SS
  };

  // ### halfDisk04
  mNumberOfD2_hHoles[4] = 17;
  mD2_hHoles[4] = new Double_t[mNumberOfD2_hHoles[4]][2]{
    {-13.85, 11.475}, // O
    {-12.15, 11.475}, // Q
    {-10.45, 8.46},   // S
    {-8.75, 8.46},    // U
    {-7.05, 5.445},   // W
    {-5.35, 5.445},   // Y
    {-3.65, 6.035},   // AA
    {-1.95, 5.005},   // CC
    {-.25, 4.86},     // EE
    {1.45, 4.915},    // GG
    {3.15, 5.66},     // II
    {4.85, 5.445},    // KK
    {6.55, 5.445},    // MM
    {8.25, 8.46},     // OO
    {9.95, 8.46},     // QQ
    {11.65, 11.475},  // SS
    {13.35, 11.475}   // UU
  };

  // ================================================
  // ## D6.5 mm holes
  // ### halfDisk00
  mD65Holes[0] = new Double_t[mTwoHoles][2]{
    {-16.6, 13.5},
    {16.6, 13.5}};

  // ### halfDisk01
  mD65Holes[1] = mD65Holes[0];

  // ### halfDisk02
  mD65Holes[2] = new Double_t[mTwoHoles][2]{
    {-16.6, 14.9}, // 1
    {16.6, 14.9}   // 2
  };

  // ### halfDisk03
  mD65Holes[3] = new Double_t[mTwoHoles][2]{
    {-17.4, 18.5}, // A
    {17.4, 18.5}   // B
  };

  // ### halfDisk02
  mD65Holes[4] = new Double_t[mTwoHoles][2]{
    {-17.4, 19.9}, //
    {17.4, 19.9}   // 2
  };

  // ================================================
  // ## D6 mm holes
  // ### halfDisk00
  mD6Holes[0] = new Double_t[mTwoHoles][2]{
    {-16.6, 12.5}, // 3
    {16.6, 12.5}   // 4
  };

  // ### halfDisk01
  mD6Holes[1] = mD6Holes[0];

  // ### halfDisk02
  mD6Holes[2] = new Double_t[mTwoHoles][2]{
    {-16.6, 13.9}, // 3
    {16.6, 13.9}   // 4
  };

  // ### halfDisk03
  mD6Holes[3] = new Double_t[mTwoHoles][2]{
    {-17.4, 17.5}, // C
    {17.4, 17.5}   // D
  };

  // ### halfDisk04
  mD6Holes[4] = new Double_t[mTwoHoles][2]{
    {-17.4, 18.9}, // #
    {17.4, 18.9}   // #
  };

  // ================================================
  // ## D8 mm holes
  // ### halfDisk00
  mNumberOfD8_Holes[0] = 2;
  mD8Holes[0] = new Double_t[mNumberOfD8_Holes[0]][2]{
    {-16.1, 10.3}, // 5
    {16.1, 10.3}   // 6
  };

  // ### halfDisk01
  mNumberOfD8_Holes[1] = mNumberOfD8_Holes[0];
  mD8Holes[1] = mD8Holes[0];

  // ### halfDisk02
  mNumberOfD8_Holes[2] = 2;
  mD8Holes[2] = new Double_t[mNumberOfD8_Holes[1]][2]{
    {-16.1, 11.7}, // 5
    {16.1, 11.7}   // 6
  };

  // ### halfDisk03
  mNumberOfD8_Holes[3] = 0;

  // ### halfDisk04
  mNumberOfD8_Holes[4] = 0;

  // ================================================
  // ## D3 mm holes
  // ### halfDisk00
  mD3Holes[0] = new Double_t[mTwoHoles][2]{
    {-14.0, 6.0}, // 7
    {14.0, 6.0}   // 8
  };

  // ### halfDisk01
  mD3Holes[1] = mD3Holes[0];

  // ### halfDisk02
  mD3Holes[2] = new Double_t[mTwoHoles][2]{
    {-14.0, 7.4}, // 7
    {14.0, 7.4}   // 8
  };

  // ### halfDisk03
  mD3Holes[3] = new Double_t[mTwoHoles][2]{
    {-19.5, 10.5}, // E
    {19.5, 10.5}   // F
  };

  // ### halfDisk04
  mD3Holes[4] = new Double_t[mTwoHoles][2]{
    {-19.5, 11.9}, // E
    {19.5, 11.9}   // F
  };

  // ================================================
  // ## M3 H7 mm holes
  // ### halfDisk00
  mNumberOfM3Holes[0] = 2;
  mM3Holes[0] = new Double_t[mNumberOfM3Holes[0]][2]{
    {-11.2, 6.0}, // 11
    {11.2, 6.0}   // 12
  };

  // ### halfDisk01
  mNumberOfM3Holes[1] = mNumberOfM3Holes[0];
  mM3Holes[1] = mM3Holes[0];

  // ### halfDisk02
  mNumberOfM3Holes[2] = 2;
  mM3Holes[2] = new Double_t[mNumberOfM3Holes[2]][2]{
    {-12.0, 7.4}, // 7
    {12.0, 7.4}   // 8
  };

  // ### halfDisk03
  mNumberOfM3Holes[3] = 4;
  mM3Holes[3] = new Double_t[mNumberOfM3Holes[3]][2]{
    {-16.5, 10.5}, // G
    {16.5, 10.5},  // H
    {-6.0, 2.5},   // M
    {6.0, 2.5}     // N
  };

  // ### halfDisk04
  mNumberOfM3Holes[4] = 4;
  mM3Holes[4] = new Double_t[mNumberOfM3Holes[4]][2]{
    {-17.0, 11.9}, // G
    {17.0, 11.9},  // H
    {-9.0, 4.4},   // K
    {9.0, 4.4}     // L
  };

  // ================================================
  // ## D45 H7 mm holes
  // ### halfDisk00
  mD45Holes[0] = new Double_t[mTwoHoles][2]{
    {-11.0, 4.0}, // 13
    {11.0, 4.0}   // 14
  };

  // ### halfDisk01
  mD45Holes[1] = mD45Holes[0];

  // ### halfDisk02
  mD45Holes[2] = new Double_t[mTwoHoles][2]{
    {-11.0, 5.4}, // 13
    {11.0, 5.4}   // 14
  };

  // ### halfDisk03
  mD45Holes[3] = new Double_t[mTwoHoles][2]{
    {-10., 1.5}, //  K
    {10., 1.5}   //  L
  };

  // ### halfDisk04
  mD45Holes[4] = new Double_t[mTwoHoles][2]{
    {-10.0, 2.9}, // M
    {10.0, 2.9}   // N
  };

  // ================================================
  // ## D2 H7 mm holes - 4 mm deep (on lower surface)
  // ### halfDisk00
  mD2Holes[0] = new Double_t[mTwoHoles][2]{
    {-12.2, 8.295}, // 9
    {12.2, 8.295}   // 10
  };

  // ### halfDisk01
  mD2Holes[1] = mD2Holes[0];

  // ### halfDisk02
  mD2Holes[2] = new Double_t[mTwoHoles][2]{
    {-12.6, 9.695}, // 9
    {12.6, 9.695}   // 10
  };

  // ### halfDisk03
  mD2Holes[3] = new Double_t[mTwoHoles][2]{
    {-15.5, 10.805}, // I
    {15.5, 10.805}   // J
  };

  // ### halfDisk04
  mD2Holes[4] = new Double_t[mTwoHoles][2]{
    {-16.0, 12.205}, // I
    {16.0, 12.205}   // J
  };
}
