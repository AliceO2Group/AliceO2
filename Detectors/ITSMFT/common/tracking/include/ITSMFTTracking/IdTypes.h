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

#ifndef ALICEO2_ITSMFT_TRACKING_IDTYPES_H_
#define ALICEO2_ITSMFT_TRACKING_IDTYPES_H_

#include <cstdint>
#include <limits>
#include <type_traits>

#include "GPUCommonDef.h"

namespace o2::itsmft::tracking
{

// The coordinate convention of a surface and every state defined on it.
enum class SurfaceKind : uint8_t {
  Undefined,
  Cylinder,
  Disk
};

static_assert(std::is_same_v<std::underlying_type_t<SurfaceKind>, uint8_t>);
static_assert(sizeof(SurfaceKind) == sizeof(uint8_t));

namespace detail
{
template <typename Tag, typename ValueType>
class Identifier
{
 public:
  static constexpr ValueType InvalidValue = std::numeric_limits<ValueType>::max();

  GPUhdDefault() constexpr Identifier() noexcept = default;
  GPUhdDefault() explicit constexpr Identifier(ValueType value) noexcept : mValue{value} {}

  GPUhdi() constexpr ValueType value() const noexcept { return mValue; }
  GPUhdi() constexpr bool isValid() const noexcept { return mValue != InvalidValue; }
  GPUhdi() static constexpr Identifier invalid() noexcept { return Identifier{InvalidValue}; }

  GPUhdi() friend constexpr bool operator==(Identifier lhs, Identifier rhs) noexcept { return lhs.mValue == rhs.mValue; }
  GPUhdi() friend constexpr bool operator!=(Identifier lhs, Identifier rhs) noexcept { return !(lhs == rhs); }
  GPUhdi() friend constexpr bool operator<(Identifier lhs, Identifier rhs) noexcept { return lhs.mValue < rhs.mValue; }

 private:
  ValueType mValue{InvalidValue};
};
} // namespace detail

struct LayerIdTag;
struct EdgeIdTag;
struct CellPathIdTag;
struct ClusterSourceIdTag;

using LayerId = detail::Identifier<LayerIdTag, uint16_t>;
using EdgeId = detail::Identifier<EdgeIdTag, uint16_t>;
using CellPathId = detail::Identifier<CellPathIdTag, uint16_t>;
using ClusterSourceId = detail::Identifier<ClusterSourceIdTag, uint16_t>;

GPUhdi() constexpr bool isRecognizedSurfaceKind(SurfaceKind kind) noexcept
{
  return kind == SurfaceKind::Cylinder || kind == SurfaceKind::Disk;
}

inline constexpr uint32_t MaxLayoutSurfaces = 32;
inline constexpr uint32_t MaxLayoutEdges = MaxLayoutSurfaces * (MaxLayoutSurfaces - 1);
inline constexpr uint32_t MaxLayoutPaths = MaxLayoutSurfaces * (MaxLayoutSurfaces - 1) * (MaxLayoutSurfaces - 1);

static_assert(MaxLayoutEdges < EdgeId::InvalidValue);
static_assert(MaxLayoutPaths < CellPathId::InvalidValue);

} // namespace o2::itsmft::tracking

#endif
