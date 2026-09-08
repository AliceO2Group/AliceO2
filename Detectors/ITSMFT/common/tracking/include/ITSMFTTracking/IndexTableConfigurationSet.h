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

#ifndef ALICEO2_ITSMFT_TRACKING_INDEXTABLECONFIGURATIONSET_H_
#define ALICEO2_ITSMFT_TRACKING_INDEXTABLECONFIGURATIONSET_H_

#include <array>
#include <cassert>
#include "ITSMFTTracking/IndexTableUtils.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"

namespace o2::itsmft::tracking
{
// Owning cache: one all-layer lookup configuration for each coordinate kind,
// plus a compact surface-to-kind mapping. Copies never borrow another owner.
class IndexTableConfigurationSet
{
 public:
  bool reset(SurfaceCatalogView catalog) noexcept
  {
    *this = {};
    if (catalog.nSurfaces > MaxLayoutSurfaces || (catalog.nSurfaces && !catalog.surfaces)) {
      return false;
    }
    for (uint32_t layer = 0; layer < catalog.nSurfaces; ++layer) {
      const auto kind = catalog.surfaces[layer].kind;
      if (kind != SurfaceKind::Cylinder && kind != SurfaceKind::Disk) {
        *this = {};
        return false;
      }
      const auto slot = kind == SurfaceKind::Cylinder ? 0 : 1;
      mKindByLayer[layer] = slot;
      mPresent[slot] = true;
    }
    mLayers = catalog.nSurfaces;
    return true;
  }
  void clear() noexcept { *this = {}; }
  size_t size() const noexcept { return mLayers; }
  size_t configurationCount() const noexcept { return size_t(mPresent[0]) + size_t(mPresent[1]); }
  bool hasKind(SurfaceKind kind) const noexcept { return (kind == SurfaceKind::Cylinder || kind == SurfaceKind::Disk) && mPresent[kind == SurfaceKind::Cylinder ? 0 : 1]; }
  IndexTableUtilsCore& forKind(SurfaceKind kind) noexcept
  {
    assert(hasKind(kind));
    return mByKind[kind == SurfaceKind::Cylinder ? 0 : 1];
  }
  const IndexTableUtilsCore& operator[](size_t layer) const noexcept
  {
    assert(layer < mLayers);
    return mByKind[mKindByLayer[layer]];
  }

 private:
  std::array<IndexTableUtilsCore, 2> mByKind;
  std::array<uint8_t, MaxLayoutSurfaces> mKindByLayer{};
  std::array<bool, 2> mPresent{};
  uint32_t mLayers = 0;
};
} // namespace o2::itsmft::tracking
#endif
