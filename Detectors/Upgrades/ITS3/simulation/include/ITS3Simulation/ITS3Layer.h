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

/// \file ITS3Layer.h
/// \brief Definition of the ITS3Layer class
/// \author felix.schlepper@cern.ch

#ifndef ALICEO2_ITS3_ITS3LAYER_H
#define ALICEO2_ITS3_ITS3LAYER_H

#include <TGeoCompositeShape.h>
#include <TGeoTube.h>
#include <TGeoVolume.h>

#include "ITS3Base/SpecsV2.h"

namespace o2::its3
{

/// This class defines the geometry for the ITS3 IB layers.
class ITS3Layer final
{
  // The hierarchy will be the following:
  // ITS2          ->       ITS3
  // ---------------------------------
  // Sensor                 PixelArray
  // Chip                   Tile
  // Module                 RSU
  // HalfStave              Segment
  // Stave                  Chip
  // HalfBarrel             CarbonForm
  // Layer                  Layer
 public:
  enum class BuildLevel : uint8_t {
    kPixelArray = 0,
    kTile,
    kRSU,
    kSegment,
    kChip,
    kCarbonForm,
    kLayer,
    kAll,
  };
  static constexpr std::array<std::string_view, static_cast<size_t>(BuildLevel::kAll)> mNames{"PixelArray", "Tile", "RSU", "Segment", "CarbonForm", "Chip", "Layer"};
  static std::string_view getName(BuildLevel b)
  {
    return mNames[static_cast<size_t>((b == BuildLevel::kAll) ? BuildLevel::kLayer : b)];
  }

  explicit ITS3Layer(int layer = 0) : mNLayer(layer),
                                      mR(o2::its3::constants::radii[mNLayer]),
                                      mRmin(o2::its3::constants::radiiInner[mNLayer]),
                                      mRmax(o2::its3::constants::radiiOuter[mNLayer]) {}

  explicit ITS3Layer(TGeoVolume* motherVolume, int layer = 0) : ITS3Layer(layer)
  {
    createLayer(motherVolume);
  }

  explicit ITS3Layer(int layer, TGeoVolume* motherVolume, TGeoMatrix* mat = nullptr, BuildLevel level = BuildLevel::kAll, bool createMaterials = false) : ITS3Layer(layer)
  {
    buildPartial(motherVolume, mat, level, createMaterials);
  }

  // Create one layer of ITS3 and attach it to the motherVolume.
  void createLayer(TGeoVolume* motherVolume);
  // Build a partial Version of the detector.
  void buildPartial(TGeoVolume* motherVolume, TGeoMatrix* mat = nullptr, BuildLevel level = BuildLevel::kAll, bool createMaterials = false);

 private:
  TGeoMedium* mSilicon{nullptr};
  TGeoMedium* mAir{nullptr};
  TGeoMedium* mCarbon{nullptr};
  TGeoMedium* mCopper{nullptr};
  void getMaterials(bool create = false);
  TGeoMedium* getMaterial(const char* matName, bool create = false);

  void init();
  void createPixelArray();
  void createTile();
  void createRSU();
  void createSegment();
  void createChip();
  void createCarbonForm();
  TGeoCompositeShape* getHringShape(TGeoTubeSeg* Hring) const;
  void createLayerImpl();

  uint8_t mNLayer{0}; // Layer number
  double mR{0};       // Middle Radius
  double mRmin{0};    // Minimum Radius
  double mRmax{0};    // Maximum Radius

  // Individual pieces
  // since TGeo manages the resources itself one should not use these pointers
  // after initializition anymore!
  TGeoVolume* mPixelArray{nullptr};
  TGeoVolumeAssembly* mTile{nullptr};
  TGeoVolumeAssembly* mRSU{nullptr};
  TGeoVolumeAssembly* mSegment{nullptr};
  TGeoVolumeAssembly* mChip{nullptr};
  TGeoVolumeAssembly* mCarbonForm{nullptr};
  TGeoVolumeAssembly* mLayer{nullptr};

  ClassDef(ITS3Layer, 3);
};
} // namespace o2::its3

#endif
