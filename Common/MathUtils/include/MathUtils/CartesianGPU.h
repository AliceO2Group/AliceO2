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

namespace o2::math_utils
{

namespace detail
{
template <typename T, int I>
struct GPUPoint2D {
  GPUPoint2D() = default;
  GPUPoint2D(T a, T b) : xx(a), yy(b) {}
  T xx;
  T yy;
  float X() const { return xx; }
  float Y() const { return yy; }
  float R() const { return o2::gpu::CAMath::Sqrt(xx * xx + yy * yy); }
  void SetX(float v) { xx = v; }
  void SetY(float v) { yy = v; }
};

template <typename T, int I>
struct GPUPoint3D : public GPUPoint2D<T, I> {
  GPUPoint3D() = default;
  GPUPoint3D(T a, T b, T c) : GPUPoint2D<T, I>(a, b), zz(c) {}
  T zz;
  float Z() const { return zz; }
  float R() const { return o2::gpu::CAMath::Sqrt(GPUPoint2D<T, I>::xx * GPUPoint2D<T, I>::xx + GPUPoint2D<T, I>::yy * GPUPoint2D<T, I>::yy + zz * zz); }
  void SetZ(float v) { zz = v; }
};
} // namespace detail

} // end namespace o2::math_utils

#endif
