// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  GPUd() GPUPoint2D(T a, T b) : xx(a), yy(b) {}
  GPUd() float X() const { return xx; }
  GPUd() float Y() const { return yy; }
  GPUd() float R() const { return o2::gpu::CAMath::Sqrt(xx * xx + yy * yy); }
  GPUd() void SetX(float v) { xx = v; }
  GPUd() void SetY(float v) { yy = v; }
  T xx;
  T yy;
};

template <typename T, int I>
struct GPUPoint3D : public GPUPoint2D<T, I> {
  GPUdDefault() GPUPoint3D() = default;
  GPUd() GPUPoint3D(T a, T b, T c) : GPUPoint2D<T, I>(a, b), zz(c) {}
  GPUd() float Z() const { return zz; }
  GPUd() float R() const { return o2::gpu::CAMath::Sqrt(GPUPoint2D<T, I>::xx * GPUPoint2D<T, I>::xx + GPUPoint2D<T, I>::yy * GPUPoint2D<T, I>::yy + zz * zz); }
  GPUd() void SetZ(float v) { zz = v; }
  T zz;
};
} // namespace detail

} // end namespace o2::math_utils

#endif
