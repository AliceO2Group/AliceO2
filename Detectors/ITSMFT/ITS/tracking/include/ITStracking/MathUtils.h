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
///
/// \file MathUtils.h
/// \brief
///

#ifndef O2_ITS_TRACKING_MATHUTILS_H_
#define O2_ITS_TRACKING_MATHUTILS_H_

#include "CommonConstants/MathConstants.h"
#include "ITStracking/Constants.h"
#include "MathUtils/Utils.h"
#include "GPUCommonMath.h"
#include "GPUCommonDef.h"

namespace o2::its::math_utils
{

GPUhdi() float computePhi(float x, float y)
{
  return o2::math_utils::fastATan2(-y, -x) + o2::constants::math::PI;
}

GPUhdi() constexpr float hypot(float x, float y)
{
  return o2::gpu::CAMath::Hypot(x, y);
}

GPUhdi() constexpr float getNormalizedPhi(float phi)
{
  phi -= o2::constants::math::TwoPI * o2::gpu::CAMath::Floor(phi * (1.f / o2::constants::math::TwoPI));
  return phi;
}

GPUhdi() float computeCurvature(float x1, float y1, float x2, float y2, float x3, float y3)
{
  // in case the triangle is degenerate we return infinite curvature.
  const float area = ((x2 - x1) * (y3 - y1)) - ((x3 - x1) * (y2 - y1));
  if (o2::gpu::CAMath::Abs(area) < constants::Tolerance) {
    return o2::constants::math::Almost0;
  }
  const float dx1 = x2 - x1, dy1 = y2 - y1;
  const float dx2 = x3 - x2, dy2 = y3 - y2;
  const float dx3 = x1 - x3, dy3 = y1 - y3;
  const float d1 = o2::gpu::CAMath::Sqrt((dx1 * dx1) + (dy1 * dy1));
  const float d2 = o2::gpu::CAMath::Sqrt((dx2 * dx2) + (dy2 * dy2));
  const float d3 = o2::gpu::CAMath::Sqrt((dx3 * dx3) + (dy3 * dy3));
  return -2.f * area / (d1 * d2 * d3);
}

GPUhdi() float computeCurvatureCentreX(float x1, float y1, float x2, float y2, float x3, float y3)
{
  // in case the triangle is degenerate we return set the centre to infinity.
  float dx21 = x2 - x1, dx32 = x3 - x2;
  if (o2::gpu::CAMath::Abs(dx21) < o2::its::constants::Tolerance ||
      o2::gpu::CAMath::Abs(dx32) < o2::its::constants::Tolerance) { // add small offset
    x2 += 1e-4;
    dx21 = x2 - x1;
    dx32 = x3 - x2;
  }
  const float k1 = (y2 - y1) / dx21, k2 = (y3 - y2) / dx32;
  if (o2::gpu::CAMath::Abs(k2 - k1) < o2::its::constants::Tolerance) {
    return o2::constants::math::VeryBig;
  }
  return 0.5f * (k1 * k2 * (y1 - y3) + k2 * (x1 + x2) - k1 * (x2 + x3)) / (k2 - k1);
}

GPUhdi() float computeTanDipAngle(float x1, float y1, float x2, float y2, float z1, float z2)
{
  // in case the points vertically align we go to pos/neg infinity.
  const float d = o2::gpu::CAMath::Hypot(x1 - x2, y1 - y2);
  if (o2::gpu::CAMath::Abs(d) < o2::its::constants::Tolerance) {
    return ((z1 > z2) ? -1.f : 1.f) * o2::constants::math::VeryBig;
  }
  return (z1 - z2) / d;
}

GPUhdi() float smallestAngleDifference(float a, float b)
{
  return o2::gpu::CAMath::Remainderf(b - a, o2::constants::math::TwoPI);
}

GPUhdi() constexpr float Sq(float v)
{
  return v * v;
}

GPUhdi() constexpr float SqSum(float v, float w)
{
  return Sq(v) + Sq(w);
}

GPUhdi() constexpr float SqSum(float u, float v, float w)
{
  return Sq(u) + SqSum(v, w);
}

GPUhdi() constexpr float SqDiff(float x, float y)
{
  return Sq(x - y);
}

GPUhdi() float MSangle(float mass, float p, float xX0)
{
  float beta = p / o2::gpu::CAMath::Hypot(mass, p);
  return 0.0136f * o2::gpu::CAMath::Sqrt(xX0) * (1.f + 0.038f * o2::gpu::CAMath::Log(xX0)) / (beta * p);
}

} // namespace o2::its::math_utils

#endif
