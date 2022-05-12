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

/// \file CartesianGPU.h
/// @author David Rohr

#ifndef ALICEO2_CARTESIANGPU_H
#define ALICEO2_CARTESIANGPU_H

#include "GPUCommonDef.h"

namespace o2::math_utils
{

namespace detail
{
template <typename T, int I>
struct GPUPoint2D {
  GPUdDefault() GPUPoint2D() = default;
  GPUhd() GPUPoint2D(T a, T b) : xx(a), yy(b) {}
  GPUhd() float X() const { return xx; }
  GPUhd() float Y() const { return yy; }
  GPUd() float R() const { return o2::gpu::CAMath::Sqrt(xx * xx + yy * yy); }
  GPUd() void SetX(float v) { xx = v; }
  GPUd() void SetY(float v) { yy = v; }
  T xx;
  T yy;
};

template <typename T, int I>
struct GPUPoint3D : public GPUPoint2D<T, I> {
  GPUdDefault() GPUPoint3D() = default;
  GPUhd() GPUPoint3D(T a, T b, T c) : GPUPoint2D<T, I>(a, b), zz(c) {}
  GPUhd() float Z() const { return zz; }
  GPUd() float R() const { return o2::gpu::CAMath::Sqrt(GPUPoint2D<T, I>::xx * GPUPoint2D<T, I>::xx + GPUPoint2D<T, I>::yy * GPUPoint2D<T, I>::yy + zz * zz); }
  GPUd() void SetZ(float v) { zz = v; }
  T zz;
};
} // namespace detail

} // end namespace o2::math_utils

#endif
