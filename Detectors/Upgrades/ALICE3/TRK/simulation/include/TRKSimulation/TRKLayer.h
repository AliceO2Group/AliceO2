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

#ifndef ALICEO2_TRK_LAYER_H
#define ALICEO2_TRK_LAYER_H

#include <TGeoManager.h>
#include <Rtypes.h>

#include "TRKBase/TRKBaseParam.h"
#include "TRKBase/Specs.h"

namespace o2
{
namespace trk
{
class TRKLayer
{
 public:
  TRKLayer() = default;
  TRKLayer(int layerNumber, std::string layerName, float rInn, float rOut, int numberOfModules, float layerX2X0);
  TRKLayer(int layerNumber, std::string layerName, float rInn, int numberOfModules, float thick);
  ~TRKLayer() = default;

  void setLayout(eLayout layout) { mLayout = layout; };

  auto getInnerRadius() const { return mInnerRadius; }
  auto getOuterRadius() const { return mOuterRadius; }
  auto getZ() const { return constants::moduleMLOT::length * mNumberOfModules; }
  auto getx2X0() const { return mX2X0; }
  auto getChipThickness() const { return mChipThickness; }
  auto getNumber() const { return mLayerNumber; }
  auto getName() const { return mLayerName; }

  TGeoVolume* createSensor(std::string type);
  TGeoVolume* createDeadzone(std::string type);
  TGeoVolume* createMetalStack(std::string type);
  TGeoVolume* createChip(std::string type);
  TGeoVolume* createModule(std::string type);
  TGeoVolume* createStave(std::string type);
  TGeoVolume* createHalfStave(std::string type);
  void createLayer(TGeoVolume* motherVolume);

 private:
  // TGeo objects outside logical volumes can cause errors. Only used in case of kStaggered and kTurboStaves layouts
  static constexpr float mLogicalVolumeThickness = 1.3;

  // User defined parameters for the layer, to be set in the constructor
  int mLayerNumber;
  std::string mLayerName;
  float mInnerRadius;
  float mOuterRadius;
  int mNumberOfModules;
  float mX2X0;
  float mChipThickness;

  // Fixed parameters for the layer, to be set based on the specifications of the chip and module
  eLayout mLayout = kCylinder;
  float mChipWidth = constants::moduleMLOT::chip::width;
  float mChipLength = constants::moduleMLOT::chip::length;
  float mDeadzoneWidth = constants::moduleMLOT::chip::passiveEdgeReadOut;
  float mSensorThickness = constants::moduleMLOT::silicon::thickness;
  int mHalfNumberOfChips = 4;

  static constexpr float Si_X0 = 9.5f;

  ClassDef(TRKLayer, 2);
};

} // namespace trk
} // namespace o2
#endif // ALICEO2_TRK_LAYER_H