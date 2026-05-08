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

#ifndef ALICEO2_IOTOF_LAYER_H
#define ALICEO2_IOTOF_LAYER_H

#include <TGeoManager.h>
#include <Rtypes.h>
#include <string>
#include <vector>

namespace o2
{
namespace iotof
{
class Layer
{
 public:
  Layer() = default;
  Layer(std::string layerName, float rInn, float rOut, float zLength, float zOffset, float layerX2X0,
        int layout = kBarrel, int nStaves = 0, float staveSize = 0.0, double staveTiltAngle = 0.0, int modulesPerStave = 0, float sensorThickness = 0.0f);
  ~Layer() = default;

  auto getInnerRadius() const { return mInnerRadius; }
  auto getOuterRadius() const { return mOuterRadius; }
  auto getZLength() const { return mZLength; }
  auto getZOffset() const { return mZOffset; }
  auto getx2X0() const { return mX2X0; }
  auto getChipThickness() const { return mChipThickness; }
  auto getName() const { return mLayerName; }
  auto getLayout() const { return mLayout; }
  auto getSegments() const { return mStaves; }
  static constexpr int kBarrel = 0;
  static constexpr int kDisk = 1;
  static constexpr int kBarrelSegmented = 2;
  static constexpr int kDiskSegmented = 3;

  virtual void createLayer(TGeoVolume* motherVolume) {};

 protected:
  std::string mLayerName;
  float mInnerRadius;
  float mOuterRadius;
  float mZLength;
  float mZOffset{0.f}; // Of use when fwd layers
  float mX2X0;
  float mChipThickness;   // Thickness of the chip in cm, derived from mX2X0 and the radiation length of silicon
  float mSensorThickness; // Thickness of the sensor in cm, to be subtracted from the chip thickness to get the total module thickness
  int mLayout{kBarrel};   // Identifier of the type of layer layout (barrel, disk, barrel segmented, disk segmented)
  // To be used only in case of the segmented layout, to define the number of staves in phi (for barrel) or in r (for disk)
  std::pair<int, float> mStaves{0, 0.0f}; // Number and size of staves in phi (for barrel) or in r (for disk) in case of segmented layout
  int mModulesPerStave{0};                // Number of modules along a stave
  double mTiltAngle{0.0};                 // Tilt angle in degrees to be applied as a rotation around the local center of the stave
};

class ITOFLayer : public Layer
{
 public:
  using Layer::Layer;
  virtual void createLayer(TGeoVolume* motherVolume) override;
  static std::vector<std::string> mRegister;
};

class OTOFLayer : public Layer
{
 public:
  using Layer::Layer;
  virtual void createLayer(TGeoVolume* motherVolume) override;
  static std::vector<std::string> mRegister;
};

class FTOFLayer : public Layer
{
 public:
  using Layer::Layer;
  virtual void createLayer(TGeoVolume* motherVolume) override;
};

class BTOFLayer : public Layer
{
 public:
  using Layer::Layer;
  virtual void createLayer(TGeoVolume* motherVolume) override;
};

} // namespace iotof
} // namespace o2
#endif // ALICEO2_IOTOF_LAYER_H
