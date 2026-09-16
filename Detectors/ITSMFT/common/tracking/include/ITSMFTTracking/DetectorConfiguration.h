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

#ifndef ALICEO2_ITSMFT_TRACKING_DETECTORCONFIGURATION_H_
#define ALICEO2_ITSMFT_TRACKING_DETECTORCONFIGURATION_H_

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include <gsl/span>

#include "ITSMFTTracking/SurfaceDescriptor.h"
#include "ITSMFTTracking/LayerMask.h"
#include "ITSMFTTracking/IndexTableConfigurationSet.h"

namespace o2::itsmft::tracking
{

enum class DetectorConfigurationError : uint8_t {
  None,
  EmptyCatalog,
  TooManySurfaces,
  InvalidComponentBoundary,
  HoleLayersOutsideLayout
};

// TimeFrame-owned detector geometry and prepared settings, shared by every
// iteration. LayerId is the dense descriptor position. SurfaceCatalogView
// borrows only the geometry; iteration topology and event state live elsewhere.
class DetectorConfiguration
{
 public:
  DetectorConfiguration() = default;
  DetectorConfiguration(gsl::span<const SurfaceDescriptor> layers, std::vector<uint16_t> componentOffsets = {0}, LayerMask holeLayers = {})
    : mLayers{layers.begin(), layers.end()}, mComponentOffsets{std::move(componentOffsets)}, mHoleLayers{holeLayers}
  {
    validate();
  }

  bool valid() const noexcept { return mError == DetectorConfigurationError::None; }
  DetectorConfigurationError getError() const noexcept { return mError; }
  bool empty() const noexcept { return mLayers.empty(); }
  std::size_t size() const noexcept { return mLayers.size(); }
  gsl::span<const SurfaceDescriptor> getLayers() const noexcept { return mLayers; }
  const SurfaceDescriptor& operator[](LayerId id) const { return mLayers.at(id.value()); }
  gsl::span<const uint16_t> getComponentOffsets() const noexcept { return mComponentOffsets; }
  LayerMask getHoleLayers() const noexcept { return mHoleLayers; }
  SurfaceCatalogView getSurfaceCatalog() const noexcept { return {mLayers.data(), static_cast<uint32_t>(mLayers.size())}; }

  // Cylinders have one radius; disks use the midpoint of their radial chart.
  float getRepresentativeRadius(LayerId id) const
  {
    const auto& surface = (*this)[id];
    return surface.kind == SurfaceKind::Cylinder ? surface.referenceCoordinate
                                                 : 0.5f * (surface.chartRange.min + surface.chartRange.max);
  }

  bool sameComponent(uint16_t first, uint16_t second) const noexcept
  {
    if (first >= mLayers.size() || second >= mLayers.size()) {
      return false;
    }
    const auto component = [this](uint16_t position) {
      return std::upper_bound(mComponentOffsets.begin(), mComponentOffsets.end(), position) - mComponentOffsets.begin();
    };
    return component(first) == component(second);
  }

  // Prepared once by Tracker before the configuration is installed in a frame.
  IndexTableConfigurationSet indexTableConfigs;
  std::vector<float> positionResolutions;
  std::vector<uint32_t> addTimeError;
  std::vector<float> layerResolution;
  std::vector<float> systError2Row;
  std::vector<float> systError2Col;

 private:
  void validate() noexcept
  {
    if (mLayers.empty()) {
      mError = DetectorConfigurationError::EmptyCatalog;
      return;
    }
    if (mLayers.size() > MaxLayoutSurfaces) {
      mError = DetectorConfigurationError::TooManySurfaces;
      return;
    }
    if (mComponentOffsets.empty() || mComponentOffsets.front() != 0 || mComponentOffsets.back() >= mLayers.size() ||
        !std::is_sorted(mComponentOffsets.begin(), mComponentOffsets.end()) ||
        std::adjacent_find(mComponentOffsets.begin(), mComponentOffsets.end()) != mComponentOffsets.end()) {
      mError = DetectorConfigurationError::InvalidComponentBoundary;
      return;
    }
    if (!mHoleLayers.isSubsetOf(LayerMask::span(0, static_cast<int>(mLayers.size()) - 1))) {
      mError = DetectorConfigurationError::HoleLayersOutsideLayout;
      return;
    }
    mError = DetectorConfigurationError::None;
  }

  std::vector<SurfaceDescriptor> mLayers;
  std::vector<uint16_t> mComponentOffsets;
  LayerMask mHoleLayers{};
  DetectorConfigurationError mError{DetectorConfigurationError::EmptyCatalog};
};

} // namespace o2::itsmft::tracking

#endif
