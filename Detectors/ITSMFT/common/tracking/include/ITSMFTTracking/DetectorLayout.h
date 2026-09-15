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

#ifndef ALICEO2_ITSMFT_TRACKING_DETECTORLAYOUT_H_
#define ALICEO2_ITSMFT_TRACKING_DETECTORLAYOUT_H_

#include <algorithm>
#include <cstdint>
#include <vector>

#include <gsl/span>

#include "ITSMFTTracking/SurfaceDescriptor.h"
#include "ITSMFTTracking/LayerMask.h"

namespace o2::itsmft::tracking
{

enum class DetectorLayoutError : uint8_t {
  None,
  EmptyCatalog,
  TooManySurfaces,
  InvalidComponentBoundary,
  HoleLayersOutsideLayout
};

struct DetectorLayoutDefinition {
  // First position of each component. Position zero is always required.
  std::vector<uint16_t> componentOffsets{0};
  LayerMask holeLayers{};
};

inline DetectorLayoutDefinition makeDetectorLayout(LayerMask holeLayers = {})
{
  DetectorLayoutDefinition definition;
  definition.holeLayers = holeLayers;
  return definition;
}

// Immutable detector layout. LayerId is exactly the dense position of a layer
// descriptor in this container. Iteration-expanded topology belongs to the
// Tracker's IterationConfiguration; this type intentionally owns no edges,
// paths, adjacency, schedules, or mutable pass state.
class DetectorLayout
{
 public:
  DetectorLayout() = default;
  DetectorLayout(gsl::span<const SurfaceDescriptor> layers, DetectorLayoutDefinition definition = {})
    : mLayers{layers.begin(), layers.end()}, mComponentOffsets{std::move(definition.componentOffsets)}, mHoleLayers{definition.holeLayers}
  {
    validate();
  }

  bool valid() const noexcept { return mError == DetectorLayoutError::None; }
  DetectorLayoutError getError() const noexcept { return mError; }
  bool empty() const noexcept { return mLayers.empty(); }
  std::size_t size() const noexcept { return mLayers.size(); }
  gsl::span<const SurfaceDescriptor> getLayers() const noexcept { return mLayers; }
  const SurfaceDescriptor& operator[](LayerId id) const { return mLayers.at(id.value()); }
  gsl::span<const uint16_t> getComponentOffsets() const noexcept { return mComponentOffsets; }
  LayerMask getHoleLayers() const noexcept { return mHoleLayers; }
  SurfaceCatalogView getSurfaceCatalog() const noexcept { return {mLayers.data(), static_cast<uint32_t>(mLayers.size())}; }

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

 private:
  void validate() noexcept
  {
    if (mLayers.empty()) {
      mError = DetectorLayoutError::EmptyCatalog;
      return;
    }
    if (mLayers.size() > MaxLayoutSurfaces) {
      mError = DetectorLayoutError::TooManySurfaces;
      return;
    }
    if (mComponentOffsets.empty() || mComponentOffsets.front() != 0 || mComponentOffsets.back() >= mLayers.size() ||
        !std::is_sorted(mComponentOffsets.begin(), mComponentOffsets.end()) ||
        std::adjacent_find(mComponentOffsets.begin(), mComponentOffsets.end()) != mComponentOffsets.end()) {
      mError = DetectorLayoutError::InvalidComponentBoundary;
      return;
    }
    if (!mHoleLayers.isSubsetOf(LayerMask::span(0, static_cast<int>(mLayers.size()) - 1))) {
      mError = DetectorLayoutError::HoleLayersOutsideLayout;
      return;
    }
    mError = DetectorLayoutError::None;
  }

  std::vector<SurfaceDescriptor> mLayers;
  std::vector<uint16_t> mComponentOffsets;
  LayerMask mHoleLayers{};
  DetectorLayoutError mError{DetectorLayoutError::EmptyCatalog};
};

} // namespace o2::itsmft::tracking

#endif
