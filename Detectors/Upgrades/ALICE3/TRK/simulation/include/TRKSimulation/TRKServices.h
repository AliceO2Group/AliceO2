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

#ifndef O2_TRK_SERVICES_H
#define O2_TRK_SERVICES_H

////// Inputs from F. Reidt, 11-2023
//                         Material 1    Fraction   X_0 (cm)  Material 2  Fraction  X_0 (cm)
// Fiber                   SiO2          0,5        12,29     PE          0,5       45
// Power bundle, no jacket Cu            0,09       1,44      PE          0,91      45
// Power bundle            Cu            0,06       1,44      PE          0,94      45
// Water bundle            PU            0,56       19        H2O         0,44      36,08
// Water bundle disk       PU            0,44       19        H2O         0,56      36,08

#include <TGeoManager.h>
#include <FairModule.h>

namespace o2
{
namespace trk
{

class TRKServices : public FairModule
{
  enum class Orientation { kASide = 1,
                           kCSide = -1 };
  // TRK services overview: three componenets
  //
  // ==================================================
  // ============||      Outer           ||============
  // =========|| ||       Tracker        || ||=========
  //          || ||======================|| ||
  //          ||      Inner + Middle        ||
  //          ||         Tracker            ||
  //          || ||======================|| ||
  // =========|| ||                      || ||=========  ---> createDisksServices
  // ============||                      ||============  ---> createMiddleBarrelServices
  // ==================================================  ---> createOuterServices
 public:
  TRKServices() = default;
  TRKServices(float rMin, float zLength, float thickness);
  void createMaterials();
  void createServices(TGeoVolume* motherVolume);
  void createColdplate(TGeoVolume* motherVolume);
  void createMiddleServices(TGeoVolume* motherVolume);
  void createOuterDisksServices(TGeoVolume* motherVolume);
  void createOuterBarrelServices(TGeoVolume* motherVolume);

 protected:
  // Coldplate
  float mColdPlateRMin;
  float mColdPlateZLength;
  float mColdPlateThickness;
  float mColdPlateX0;

  // Services
  float mFiberComposition[2] = {0.5, 0.5};               // SiO2, PE
  float mPowerBundleComposition[2] = {0.09, 0.91};       // Cu, PE
  float mPowerBundleJacketComposition[2] = {0.06, 0.94}; // Cu, PE
  float mWaterBundleComposition[2] = {0.56, 0.44};       // PU, H2O
  float mWaterBundleDiskComposition[2] = {0.44, 0.56};   // PU, H2O
  float mMiddleDiskThickness = 1.0;                      // cm
  std::vector<float> mCableFanWeights = {0.5, 0.3, 0.2}; // relative weights of the fan layers

  ClassDefOverride(TRKServices, 1);
};
} // namespace trk
} // namespace o2
#endif // O2_TRK_SERVICES_H