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

namespace o2
{
namespace iotof
{
class Layer
{
 public:
  Layer() = default;
  Layer(std::string layerName, float rInn, float rOut, float zLength, float zOffset, float layerX2X0, bool isBarrel = true);
  ~Layer() = default;

  auto getInnerRadius() const { return mInnerRadius; }
  auto getOuterRadius() const { return mOuterRadius; }
  auto getZLength() const { return mZLength; }
  auto getZOffset() const { return mZOffset; }
  auto getx2X0() const { return mX2X0; }
  auto getChipThickness() const { return mChipThickness; }
  auto getName() const { return mLayerName; }
  auto getIsBarrel() const { return mIsBarrel; }

  virtual void createLayer(TGeoVolume* motherVolume){};

 protected:
  std::string mLayerName;
  float mInnerRadius;
  float mOuterRadius;
  float mZLength;
  float mZOffset{0.f}; // Of use when fwd layers
  float mX2X0;
  float mChipThickness;
  bool mIsBarrel{true};
};

class ITOFLayer : public Layer
{
 public:
  using Layer::Layer;
  virtual void createLayer(TGeoVolume* motherVolume) override;
};

class OTOFLayer : public Layer
{
 public:
  using Layer::Layer;
  virtual void createLayer(TGeoVolume* motherVolume) override;
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