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

/// \file PCBSupport.h
/// \brief Class describing geometry of one MFT half-disk PCBsupport
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#ifndef ALICEO2_MFT_PCBSUPPORT_H_
#define ALICEO2_MFT_PCBSUPPORT_H_

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

class TGeoVolume;
class TGeoCompositeShape;

namespace o2
{
namespace mft
{

class PCBSupport
{

 public:
  PCBSupport();
  ~PCBSupport() = default;
  TGeoVolumeAssembly* create(Int_t kHalf, Int_t disk);

 private:
  void initParameters();
  TGeoVolumeAssembly* mHalfDisk;
  TGeoMedium* mPCBMediumCu;
  TGeoMedium* mPCBMediumFR4;
  TGeoBBox* mSomeBox;
  TGeoTube* mSomeTube;
  TGeoArb8* mSomeArb;

  TGeoSubtraction* mSomeSubtraction;
  TGeoUnion* mSomeUnion;
  TGeoTranslation* mSomeTranslation;
  TGeoCompositeShape* mPCBCu;
  TGeoCompositeShape* mPCBFR4;

  Double_t mCuThickness;  //Cu layer thickness
  Double_t mFR4Thickness; //FR4 layer thickness
  Double_t mPCBRad[5];    // Radius of each PCB disk
  Double_t mDiskGap;      //gap between half disks
  Double_t mPhi0;
  Double_t mPhi1;
  Double_t mT_delta;          //Excess to remove to avoid coplanar surfaces that causes visualization glitches and overlaps
  Int_t mNumberOfBoxCuts[5];  // Number of box cuts in each PCB
  Double_t (*mBoxCuts[5])[4]; // Box cuts on each PCB
  Int_t mNumberOfBoxAdd[5];   // Number of box added to each PCB
  Double_t (*mBoxAdd[5])[4];  // Box added to each PCB
  Int_t mNumberOfHoles[5];    // Number of Holes in each PCB
  Double_t (*mHoles[5])[3];   // Holes on each PCB

  ClassDefNV(PCBSupport, 1);
};
} // namespace mft
} // namespace o2

#endif
