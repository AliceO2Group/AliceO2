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

#ifndef ALICEO2_MFT_BARREL_H_
#define ALICEO2_MFT_BARREL_H_

#include "TGeoVolume.h"
#include "TGeoMatrix.h"

class TGeoVolumeAssembly;

namespace o2
{
namespace mft
{

class Barrel
{

 public:
  Barrel();
  ~Barrel() = default;

  Float_t GetFixationServicesThickness() { return mFSThickness; };
  void SetFixationServicesThickness(Float_t& tFSThickness) { mFSThickness = tFSThickness; };
  TGeoVolumeAssembly* createBarrel();

  // protected:
  //  TGeoVolumeAssembly* mBarrel;

 private:
  Float_t mFSThickness; //fixation services thickness

  ClassDefNV(Barrel, 2);
};
} // namespace mft
} // namespace o2

#endif
