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

#ifndef ALICEO2_ITSMFT_TRACKING_SURFACEMEASUREMENT_H_
#define ALICEO2_ITSMFT_TRACKING_SURFACEMEASUREMENT_H_

#include <type_traits>

#include "GPUCommonDef.h"
namespace o2::itsmft::tracking
{

// q is normal to the surface. The measured coordinates are always (u, v).
struct SurfaceFramePoint {
  float q{0.f};
  float u{0.f};
  float v{0.f};
  float frameAngle{0.f};
};

// Packed symmetric covariance of the measured (u, v) coordinates.
struct SurfaceCovariance2F {
  float uu{0.f};
  float uv{0.f};
  float vv{0.f};
};

struct SurfaceMeasurement {
  SurfaceFramePoint frame{};
  SurfaceCovariance2F covariance{};
};

#define O2_ITSMFT_ASSERT_DEVICE_TYPE(Type, Size)     \
  static_assert(std::is_standard_layout_v<Type>);    \
  static_assert(std::is_trivially_copyable_v<Type>); \
  static_assert(sizeof(Type) == Size)

O2_ITSMFT_ASSERT_DEVICE_TYPE(SurfaceFramePoint, 16);
O2_ITSMFT_ASSERT_DEVICE_TYPE(SurfaceCovariance2F, 12);
O2_ITSMFT_ASSERT_DEVICE_TYPE(SurfaceMeasurement, 28);
static_assert(alignof(SurfaceMeasurement) == 4);
static_assert(offsetof(SurfaceMeasurement, frame) == 0);
static_assert(offsetof(SurfaceMeasurement, covariance) == 16);

#undef O2_ITSMFT_ASSERT_DEVICE_TYPE

} // namespace o2::itsmft::tracking

#endif /* ALICEO2_ITSMFT_TRACKING_SURFACEMEASUREMENT_H_ */
