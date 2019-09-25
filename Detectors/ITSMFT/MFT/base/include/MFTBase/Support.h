// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Support.h
/// \brief Class describing geometry of one MFT half-disk support
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#ifndef ALICEO2_MFT_SUPPORT_H_
#define ALICEO2_MFT_SUPPORT_H_

#include "TGeoVolume.h"
#include "TGeoMatrix.h"
#include "TGeoBBox.h"
#include "TGeoCompositeShape.h"
#include "TGeoTube.h"
#include "TGeoCone.h"
#include "TGeoArb8.h"
#include "TGeoBoolNode.h"
#include "TMath.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "FairLogger.h"

class TGeoVolume;
class TGeoCompositeShape;

namespace o2
{
namespace mft
{

class Support
{

 public:
  Support();
  ~Support() = default;
  TGeoVolumeAssembly* create(Int_t kHalf, Int_t disk);

 private:
  void initParameters();
  TGeoVolumeAssembly* mHalfDisk;
  TGeoMedium* mSupportMedium;
  TGeoBBox* mSomeBox;
  TGeoTube* mSomeTube;
  TGeoArb8* mSomeArb;

  TGeoSubtraction* mSomeSubtraction;
  TGeoUnion* mSomeUnion;
  TGeoTranslation* mSomeTranslation;
  TGeoCompositeShape* mSomeCS;

  Double_t mSupThickness; //Support Thickness
  Double_t mSupRad[5];    // Radius of each support disk
  Double_t mDiskGap;      //gap between half disks
  Double_t mPhi0;
  Double_t mPhi1;
  Double_t mT_delta; //Excess to remove to avoid coplanar surfaces that causes visualization glitches
  Double_t mRaisedBoxHeight;
  Double_t mFixBoxHeight;
  Double_t mOuterCut[5]; //Distance of external disk cuts (oposite to beam pipe)
                         // this is the y origin on Guillamet's PDF blueprints

  Int_t mNumberOfBoxCuts[5];  // Number of box cuts in each half disk support
  Double_t (*mBoxCuts[5])[4]; // Box cuts on each disk

  Int_t mNumberOfRaixedBoxes[5]; //Number of Raised boxes in each halfDisk support
  Double_t (*mBRaised[5])[4];    //Raised boxes for each halfDisk

  Int_t mNumberOfFixBoxes[5]; //Number of Fixation boxes in each halfDisk support
  Double_t (*mBFix[5])[5];    //Fixation boxes for each halfDisk

  Int_t mNumberOfVoids[5];        //Number of Voids (big holes) in each halfDisk support
  Double_t (*mVoidVert[5])[4][2]; //Vertexes of Voids

  Int_t mNumberOfM2Holes[5];  // Number of M2 Holes in each halfDisk support
  Double_t (*mM2Holes[5])[2]; // M2 holes on halfdisk 00 and 01
  Double_t mRad_M2;
  Double_t mHeight_M2;

  // ==== D2 H7 - 4 mm deep (on higher surface)
  Int_t mNumberOfD2_hHoles[5];
  Double_t (*mD2_hHoles[5])[2]; // D2 holes on raisedBoxes on each disk
  Double_t mRad_D2_h;
  Double_t mHeight_D2_h;

  // ==== D 6.5 mm holes
  Double_t (*mD65Holes[5])[2]; // Positions of D6.5 mm holes on disk
  Int_t mTwoHoles;             // Number of D6.5 mm Holes in each halfDisk support
  Double_t mD65;               //Radius

  // ==== D6 H7 (6 mm diameter holes)
  Double_t (*mD6Holes[5])[2]; // Positions of D6 mm holes on disk
  Double_t mD6;               // Radius

  // ==== D8 H7 (8 mm diameter holes)
  Int_t mNumberOfD8_Holes[5];
  Double_t mD8;               // Radius
  Double_t (*mD8Holes[5])[2]; // Positions of D8 mm holes on disk

  // ==== D3 H7 (3 mm diameter holes)
  Double_t mD3;               // Radius
  Double_t (*mD3Holes[5])[2]; // Positions of D8 mm holes on disk

  // ==== M3 H7 (?? mm diameter holes)
  Int_t mNumberOfM3Holes[5];  // Number of M2 Holes in each halfDisk support
  Double_t mM3;               // Radius   TODO: Verify this!
  Double_t (*mM3Holes[5])[2]; // Positions of M3 holes on disk

  // ==== D4.5 H9
  Double_t mD45;               // Radius
  Double_t (*mD45Holes[5])[2]; // Positions of D4.5 mm holes on disk

  // ==== D2 H7 - 4 mm deep (on lower surface)
  Double_t mD2; // Radius
  Double_t mHeight_D2;
  Double_t (*mD2Holes[5])[2]; // Positions of D2 mm holes on disk

  ClassDef(Support, 1);
};
} // namespace mft
} // namespace o2

#endif
