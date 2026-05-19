// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef TRACKINGITSU_INCLUDE_LAYERMASK_H_
#define TRACKINGITSU_INCLUDE_LAYERMASK_H_

#include <cstdint>
#include <type_traits>

#ifndef GPUCA_GPUCODE
#include <fmt/format.h>
#include <string>
#endif

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "ITStracking/Constants.h"

namespace o2::its
{

struct LayerMask {
  GPUhdDefault() constexpr LayerMask() noexcept = default;
  GPUhdDefault() constexpr LayerMask(uint16_t mask) noexcept : mBits{mask} {}
  GPUhdDefault() constexpr LayerMask(int layer0, int layer1, int layer2) noexcept
    : mBits{static_cast<uint16_t>((uint16_t(1) << layer0) | (uint16_t(1) << layer1) | (uint16_t(1) << layer2))}
  {
  }
  GPUhdi() constexpr operator uint16_t() const noexcept { return mBits; }
  GPUhdi() constexpr uint16_t value() const noexcept { return mBits; }
  GPUhdi() constexpr void set(int layer) noexcept { mBits |= (uint16_t(1) << layer); }

  GPUhdi() LayerMask operator~() const noexcept { return LayerMask{static_cast<uint16_t>(~mBits)}; }
  GPUhdi() LayerMask operator&(LayerMask other) const noexcept { return LayerMask{static_cast<uint16_t>(mBits & other.mBits)}; }
  GPUhdi() LayerMask operator|(LayerMask other) const noexcept { return LayerMask{static_cast<uint16_t>(mBits | other.mBits)}; }
  GPUhdi() LayerMask& operator&=(LayerMask other) noexcept
  {
    mBits &= other.mBits;
    return *this;
  }
  GPUhdi() LayerMask& operator|=(LayerMask other) noexcept
  {
    mBits |= other.mBits;
    return *this;
  }

  GPUhdi() bool empty() const noexcept { return mBits == 0; }
  GPUhdi() bool has(int layer) const noexcept { return mBits & (uint16_t(1) << layer); }
  GPUhdi() bool isSubsetOf(LayerMask allowed) const noexcept { return (*this & ~allowed).empty(); }
  GPUhdi() bool isAllowedHoleMask(int maxHoles, LayerMask allowedHoleMask) const noexcept
  {
    const int allowedHoles = maxHoles > 0 ? maxHoles : 0;
    return count() <= allowedHoles && isSubsetOf(allowedHoleMask);
  }
  GPUhdi() bool isAllowed(int maxHoles, LayerMask allowedHoleMask) const noexcept
  {
    return holeMask().isAllowedHoleMask(maxHoles, allowedHoleMask);
  }
  GPUhdi() int length() const noexcept { return empty() ? 0 : last() - first() + 1; }
  GPUhdi() int count() const noexcept { return static_cast<int>(o2::gpu::GPUCommonMath::Popcount(mBits)); }
  GPUhdi() int first() const noexcept { return mBits ? static_cast<int>(o2::gpu::GPUCommonMath::Ctz(mBits)) : constants::UnusedIndex; }
  GPUhdi() int last() const noexcept { return mBits ? 31 - static_cast<int>(o2::gpu::GPUCommonMath::Clz(mBits)) : constants::UnusedIndex; }
  GPUhdi() LayerMask holeMask() const noexcept
  {
    return empty() ? LayerMask{0} : (span(first(), last()) & ~(*this));
  }

  GPUhdi() int slot(int layer) const noexcept
  {
    if (!has(layer)) {
      return constants::UnusedIndex;
    }
    const uint32_t lowerLayers = (uint32_t(1) << layer) - 1;
    return static_cast<int>(o2::gpu::GPUCommonMath::Popcount(static_cast<uint32_t>(mBits) & lowerLayers));
  }

  static GPUhdi() LayerMask span(int fromLayer, int toLayer) noexcept
  {
    if (fromLayer > toLayer) {
      return 0;
    }
    const uint32_t upper = (uint32_t(1) << (toLayer + 1)) - 1;
    const uint32_t lower = (uint32_t(1) << fromLayer) - 1;
    return static_cast<uint16_t>(upper & ~lower);
  }

  static GPUhdi() LayerMask skipped(int fromLayer, int toLayer) noexcept
  {
    return (toLayer - fromLayer <= 1) ? LayerMask{0} : span(fromLayer + 1, toLayer - 1);
  }

#ifndef GPUCA_GPUCODE
  std::string asString() const { return fmt::format("{:016b}", mBits); }
#endif

 private:
  uint16_t mBits{0};
};

static_assert(std::is_standard_layout_v<LayerMask>);
static_assert(std::is_trivially_copyable_v<LayerMask>);
static_assert(sizeof(LayerMask) == sizeof(uint16_t));
static_assert(alignof(LayerMask) == alignof(uint16_t));

} // namespace o2::its

#endif
