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
///
/// \file MathUtils.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_CAUTILS_H_
#define TRACKINGITSU_INCLUDE_CAUTILS_H_

#ifndef GPUCA_GPUCODE_DEVICE
#include <array>
#include <cmath>
#include <cassert>
#include <iostream>
#endif

#include "MathUtils/Utils.h"
#include "ITStracking/Constants.h"
#include "GPUCommonMath.h"
#include "GPUCommonDef.h"

namespace o2
{
namespace its
{

namespace math_utils
{
GPUhdni() float computePhi(const float, const float);
GPUhdni() float hypot(const float, const float);
GPUhdni() constexpr float getNormalizedPhi(const float);
GPUhdni() constexpr float3 crossProduct(const float3&, const float3&);
GPUhdni() float computeCurvature(float x1, float y1, float x2, float y2, float x3, float y3);
GPUhdni() float computeCurvatureCentreX(float x1, float y1, float x2, float y2, float x3, float y3);
GPUhdni() float computeTanDipAngle(float x1, float y1, float x2, float y2, float z1, float z2);

} // namespace math_utils

GPUhdi() float math_utils::computePhi(const float x, const float y)
{
  //return o2::gpu::CAMath::ATan2(-yCoordinate, -xCoordinate) + constants::math::Pi;
  return o2::math_utils::fastATan2(-y, -x) + constants::math::Pi;
}

GPUhdi() float math_utils::hypot(const float x, const float y)
{
  return o2::gpu::CAMath::Sqrt(x * x + y * y);
}

GPUhdi() constexpr float math_utils::getNormalizedPhi(const float phi)
{
  return (phi < 0) ? phi + constants::math::TwoPi : (phi > constants::math::TwoPi) ? phi - constants::math::TwoPi
                                                                                   : phi;
}

GPUhdi() constexpr float3 math_utils::crossProduct(const float3& firstVector, const float3& secondVector)
{

  return float3{(firstVector.y * secondVector.z) - (firstVector.z * secondVector.y),
                (firstVector.z * secondVector.x) - (firstVector.x * secondVector.z),
                (firstVector.x * secondVector.y) - (firstVector.y * secondVector.x)};
}

GPUhdi() float math_utils::computeCurvature(float x1, float y1, float x2, float y2, float x3, float y3)
{
  const float d = (x2 - x1) * (y3 - y2) - (x3 - x2) * (y2 - y1);
  const float a =
    0.5f * ((y3 - y2) * (y2 * y2 - y1 * y1 + x2 * x2 - x1 * x1) - (y2 - y1) * (y3 * y3 - y2 * y2 + x3 * x3 - x2 * x2));
  const float b =
    0.5f * ((x2 - x1) * (y3 * y3 - y2 * y2 + x3 * x3 - x2 * x2) - (x3 - x2) * (y2 * y2 - y1 * y1 + x2 * x2 - x1 * x1));
  const float den2 = (d * x1 - a) * (d * x1 - a) + (d * y1 - b) * (d * y1 - b);
  return den2 > 0.f ? -1.f * d / o2::gpu::CAMath::Sqrt(den2) : 0.f;
}

GPUhdi() float math_utils::computeCurvatureCentreX(float x1, float y1, float x2, float y2, float x3, float y3)
{
  float dx21 = x2 - x1, dx32 = x3 - x2;
  if (dx21 == 0.f || dx32 == 0.f) { // add small offset
    x2 += 1e-4;
    dx21 = x2 - x1;
    dx32 = x3 - x2;
  }
  float k1 = (y2 - y1) / dx21, k2 = (y3 - y2) / dx32;
  return (k1 != k2) ? 0.5f * (k1 * k2 * (y1 - y3) + k2 * (x1 + x2) - k1 * (x2 + x3)) / (k2 - k1) : 1e5;
}

GPUhdi() float math_utils::computeTanDipAngle(float x1, float y1, float x2, float y2, float z1, float z2)
{
  return (z1 - z2) / o2::gpu::CAMath::Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_CAUTILS_H_ */
