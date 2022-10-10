// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Support.h
/// \brief Class describing geometry of one MFT half-disk support
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \author Rafael Pezzi <rafael.pezzi@cern.ch>

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
#include <fairlogger/Logger.h>

namespace o2
{
namespace mft
{

class Support
{
  using shapeParam = std::vector<Double_t>;
  using diskBoxCuts = std::vector<shapeParam>;

 public:
  Support();
  ~Support() = default;
  TGeoVolumeAssembly* create(Int_t kHalf, Int_t disk);

 private:
  void initParameters();
  TGeoVolumeAssembly* mHalfDisk;
  TGeoMedium* mSupportMedium;

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

  std::vector<shapeParam> mDiskBoxCuts[5];     // Box cuts on each disk
  std::vector<shapeParam> mDiskRaisedBoxes[5]; //Raised boxes for each halfDisk
  std::vector<shapeParam> mDiskFixBoxes[5];    //Fix boxes for each halfDisk (can be merged with mDiskRaisedBoxes)

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

  ClassDefNV(Support, 1);
};

//Template to reduce boilerplate for TGeo boolean operations
template <class L, class R, class T, class OP>
auto compositeOperation(L&& left, R&& right, T&& translation, OP&& op)
{
  auto result = new OP(std::forward<L&&>(left), std::forward<R&&>(right), nullptr, std::forward<T&&>(translation));
  return result;
}

//Template function to perform serial TGeo boolean operations
template <class L, class SHAPE, class EL, class OP>
auto serialBoolOperation(L&& base, SHAPE&& shape, EL&& elements, OP&& op)
{
  TGeoCompositeShape* localCS = nullptr;
  SHAPE* localshape;
  TGeoTranslation* localTranslation;

  for (auto par : elements) {
    //Info("SerialBoolOperation", Form("params: %f %f %f %f %f %f,", par[0], par[1], par[2], par[3], par[4], par[5]), 0, 0);

    localshape = new SHAPE(par[0], par[1], par[2], nullptr);
    localTranslation = new TGeoTranslation(par[3], par[4], par[5]);

    //The first subtraction needs a shape, the base shape
    if (!localCS) {
      localCS = new TGeoCompositeShape(nullptr, compositeOperation(base, localshape, localTranslation, std::forward<OP>(op)));
    } else {
      localCS = new TGeoCompositeShape(nullptr, compositeOperation(localCS, localshape, localTranslation, std::forward<OP>(op)));
    }
  }
  return localCS;
}

} // namespace mft
} // namespace o2

#endif
