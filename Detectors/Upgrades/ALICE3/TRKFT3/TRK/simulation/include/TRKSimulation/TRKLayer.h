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

#include "TRKBase/Specs.h"
#include "TRKBase/TRKBaseParam.h"
#include <TGeoManager.h>

#include <Rtypes.h>

#include <string>
#include <utility>

namespace o2
{
namespace trk
{
enum class MatBudgetParamMode {
  Thickness,
  X2X0
};

class TRKCylindricalLayer
{
 public:
  TRKCylindricalLayer() = default;
  TRKCylindricalLayer(int layerNumber, std::string layerName, float rInn, float length, float thickOrX2X0, MatBudgetParamMode mode);
  virtual ~TRKCylindricalLayer() = default;

  auto getInnerRadius() const { return mInnerRadius; }
  auto getOuterRadius() const { return mOuterRadius; }
  auto getZ() const { return mLength; }
  auto getx2X0() const { return mX2X0; }
  auto getChipThickness() const { return mChipThickness; }
  auto getNumber() const { return mLayerNumber; }
  auto getName() const { return mLayerName; }

  virtual TGeoVolume* createSensor();
  virtual TGeoVolume* createMetalStack();
  virtual void createLayer(TGeoVolume* motherVolume);

 protected:
  // User defined parameters for the layer, to be set in the constructor
  int mLayerNumber;
  std::string mLayerName;
  float mInnerRadius;
  float mOuterRadius;
  float mLength;
  float mX2X0;
  float mChipThickness;

  // Fixed parameters for the layer, to be set based on the specifications of the chip and module
  static constexpr double sSensorThickness = constants::moduleMLOT::silicon::thickness;

  static constexpr float Si_X0 = 9.5f;

  ClassDef(TRKCylindricalLayer, 0);
};

class TRKSegmentedLayer : public TRKCylindricalLayer
{
 public:
  TRKSegmentedLayer() = default;
  TRKSegmentedLayer(int layerNumber, std::string layerName, float rInn, float tiltAngle, int numberOfStaves, int numberOfModules, float thickOrX2X0, MatBudgetParamMode mode);
  ~TRKSegmentedLayer() override = default;

  TGeoVolume* createSensor() override;
  TGeoVolume* createDeadzone();
  TGeoVolume* createMetalStack() override;
  TGeoVolume* createChip();
  TGeoVolume* createModule();
  virtual TGeoVolume* createStave() = 0;
  void createLayer(TGeoVolume* motherVolume) override = 0;

 protected:
  float mTiltAngle;
  int mNumberOfModules;
  int mNumberOfStaves;
  bool mIsFlipped = false;

  // Fixed parameters for the layer, to be set based on the specifications of the chip and module
  static constexpr double sChipWidth = constants::moduleMLOT::chip::width;
  static constexpr double sChipLength = constants::moduleMLOT::chip::length;
  static constexpr double sDeadzoneWidth = constants::moduleMLOT::chip::passiveEdgeReadOut;
  static constexpr double sModuleLength = constants::moduleMLOT::length;
  static constexpr double sModuleWidth = constants::moduleMLOT::width;
  static constexpr int sHalfNumberOfChips = 4;

  // TGeo objects outside logical volumes can cause errors
  static constexpr float sLogicalVolumeThickness = 1.3;

  // For the segmented layers, because of tilting and staggering the bounding radii can be different
  // from the inner radius and inner radius + thickness.
  // This function calculates the bounding radii based on the geometry of the stave and the tilt angle,
  // to ensure that the layer volume is large enough to contain all the staves without overlaps.
  virtual std::pair<float, float> getBoundingRadii(double staveWidth) const;

  ClassDefOverride(TRKSegmentedLayer, 0);
};

class TRKMLLayer : public TRKSegmentedLayer
{
 public:
  TRKMLLayer() = default;
  TRKMLLayer(int layerNumber, std::string layerName, float rInn, float staggerOffset, float tiltAngle, int numberOfStaves, int numberOfModules, float thickOrX2X0, MatBudgetParamMode mode);
  ~TRKMLLayer() override = default;

  TGeoVolume* createStave() override;
  void createLayer(TGeoVolume* motherVolume) override;

 private:
  float mStaggerOffset;

  static constexpr double sStaveWidth = constants::ML::width;
  static constexpr int sFlippedLayerNumber = 3;

  // Override to account for the staggering offset present in specific ML layers
  std::pair<float, float> getBoundingRadii(double staveWidth) const override;

  ClassDefOverride(TRKMLLayer, 0);
};

class TRKOTLayer : public TRKSegmentedLayer
{
 public:
  TRKOTLayer() = default;
  TRKOTLayer(int layerNumber, std::string layerName, float rInn, float tiltAngle, int numberOfStaves, int numberOfModules, float thickOrX2X0, MatBudgetParamMode mode);
  ~TRKOTLayer() override = default;

  TGeoVolume* createStave() override;
  TGeoVolume* createHalfStave();
  void createLayer(TGeoVolume* motherVolume) override;

 protected:
  static constexpr float sGapBetweenOuterTrackerBarrelHalves = 0.8; // cm, gap between the two halves of the OT barrel

 private:
  static constexpr double sHalfStaveWidth = constants::OT::halfstave::width;
  static constexpr double sInStaveOverlap = constants::moduleMLOT::gaps::outerEdgeLongSide + constants::moduleMLOT::chip::passiveEdgeReadOut + 0.1; // 1.5mm outer-edge + 1mm deadzone + 1mm (true) overlap
  static constexpr double sStaveWidth = constants::OT::width - sInStaveOverlap;

  // Override to account for the staggering offset present in OT layers
  std::pair<float, float> getBoundingRadii(double staveWidth) const override;

  ClassDefOverride(TRKOTLayer, 0);
};

} // namespace trk
} // namespace o2
#endif // ALICEO2_TRK_LAYER_H
