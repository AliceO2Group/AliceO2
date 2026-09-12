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

#ifndef ALICEO2_ITSMFT_TRACKING_GLOBALMEASUREMENT_H_
#define ALICEO2_ITSMFT_TRACKING_GLOBALMEASUREMENT_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "GPUCommonDef.h"
#include "ITSMFTTracking/IdTypes.h"

namespace o2::itsmft::tracking
{

struct GlobalPoint3F {
  float x;
  float y;
  float z;
};

struct GlobalCovariance3F {
  float xx{0.f};
  float xy{0.f};
  float xz{0.f};
  float yy{0.f};
  float yz{0.f};
  float zz{0.f};

  GPUhdi() float& operator[](std::size_t index) noexcept { return (&xx)[index]; }
  GPUhdi() const float& operator[](std::size_t index) const noexcept { return (&xx)[index]; }
};

struct GlobalMeasurement {
  enum CovarianceIndex : uint8_t {
    XX,
    XY,
    XZ,
    YY,
    YZ,
    ZZ
  };

  union {
    struct {
      float x;
      float y;
      float z;
    };
    GlobalPoint3F position;
  };
  GlobalCovariance3F covariance{};
  float radius{0.f};
  float phi{0.f};
  uint32_t clusterId{std::numeric_limits<uint32_t>::max()};

  GPUhdi() bool hasValidClusterId() const noexcept { return clusterId != std::numeric_limits<uint32_t>::max(); }
};

#define O2_ITSMFT_ASSERT_GLOBAL_TYPE(Type, Size)     \
  static_assert(std::is_standard_layout_v<Type>);    \
  static_assert(std::is_trivially_copyable_v<Type>); \
  static_assert(sizeof(Type) == Size)

O2_ITSMFT_ASSERT_GLOBAL_TYPE(GlobalMeasurement, 48);
O2_ITSMFT_ASSERT_GLOBAL_TYPE(GlobalPoint3F, 12);
O2_ITSMFT_ASSERT_GLOBAL_TYPE(GlobalCovariance3F, 24);

#undef O2_ITSMFT_ASSERT_GLOBAL_TYPE

static_assert(alignof(GlobalMeasurement) == 4);
static_assert(offsetof(GlobalMeasurement, x) == 0);
static_assert(offsetof(GlobalMeasurement, covariance) == 12);
static_assert(offsetof(GlobalMeasurement, radius) == 36);
static_assert(offsetof(GlobalMeasurement, phi) == 40);
static_assert(offsetof(GlobalMeasurement, clusterId) == 44);

} // namespace o2::itsmft::tracking

#endif // ALICEO2_ITSMFT_TRACKING_GLOBALMEASUREMENT_H_
