// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#ifndef __OPENCL__
#include <array>
#include <cmath>
#endif

#include "ITStracking/Constants.h"
#include "GPUCommonMath.h"
#include "GPUCommonDef.h"
#include <cassert>
#include <iostream>

namespace o2
{
namespace its
{

namespace math_utils
{
GPUhdni() float calculatePhiCoordinate(const float, const float);
GPUhdni() float calculateRCoordinate(const float, const float);
GPUhdni() constexpr float getNormalizedPhiCoordinate(const float);
GPUhdni() constexpr float3 crossProduct(const float3&, const float3&);
GPUhdni() float computeCurvature(float x1, float y1, float x2, float y2, float x3, float y3);
GPUhdni() float computeCurvatureCentreX(float x1, float y1, float x2, float y2, float x3, float y3);
GPUhdni() float computeTanDipAngle(float x1, float y1, float x2, float y2, float z1, float z2);
GPUhdni() float fastATan2(float y, float x);
GPUhdni() float fastATanP(float x);

} // namespace math_utils

GPUhdi() float math_utils::calculatePhiCoordinate(const float xCoordinate, const float yCoordinate)
{
  //return o2::gpu::CAMath::ATan2(-yCoordinate, -xCoordinate) + constants::math::Pi;
  return fastATan2(-yCoordinate, -xCoordinate) + constants::math::Pi;
}

GPUhdi() float math_utils::calculateRCoordinate(const float xCoordinate, const float yCoordinate)
{
  return o2::gpu::CAMath::Sqrt(xCoordinate * xCoordinate + yCoordinate * yCoordinate);
}

GPUhdi() constexpr float math_utils::getNormalizedPhiCoordinate(const float phiCoordinate)
{
  return (phiCoordinate < 0)
           ? phiCoordinate + constants::math::TwoPi
           : (phiCoordinate > constants::math::TwoPi) ? phiCoordinate - constants::math::TwoPi : phiCoordinate;
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

  return -1.f * d / o2::gpu::CAMath::Sqrt((d * x1 - a) * (d * x1 - a) + (d * y1 - b) * (d * y1 - b));
}

GPUhdi() float math_utils::computeCurvatureCentreX(float x1, float y1, float x2, float y2, float x3, float y3)
{
  const float k1 = (y2 - y1) / (x2 - x1), k2 = (y3 - y2) / (x3 - x2);
  return 0.5f * (k1 * k2 * (y1 - y3) + k2 * (x1 + x2) - k1 * (x2 + x3)) / (k2 - k1);
}

GPUhdi() float math_utils::computeTanDipAngle(float x1, float y1, float x2, float y2, float z1, float z2)
{
  return (z1 - z2) / o2::gpu::CAMath::Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

GPUhdi() float math_utils::fastATan2(float y, float x)
{
  constexpr float pi05 = 0.5 * constants::math::Pi;

  float sign = 1.f;

  if (y < 0.f) {
    y = -y;
    sign = -1.f;
  }

  // y >= 0

  float add = 0.f;
  if (x < 0.f) {
    add = pi05;
    float tmp = y;
    y = -x;
    x = tmp;
  }

  // x>=0, y>=0
  // use atan(t)+atan(1/t)=pi/2 for t>0
  float atanp = (x > y) ? fastATanP(y / x) : pi05 - fastATanP(x / y);
  return sign * (atanp + add);
}

GPUhdi() float math_utils::fastATanP(float x)
{
  // x >= 0
  const float kThresh = 0.315f; // parametrization switch here
  constexpr float kC0 = 7.85398163397448279e-01f;
  return x > kThresh ? x * (kC0 + 0.273f * (1.f - x)) : x / (1.f + x * x * 0.2815f);
}

} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_CAUTILS_H_ */
