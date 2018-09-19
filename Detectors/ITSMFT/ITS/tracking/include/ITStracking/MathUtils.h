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

#include <array>
#include <cmath>

#include "ITStracking/Constants.h"

namespace o2
{
namespace ITS
{
namespace CA
{

namespace MathUtils
{
float calculatePhiCoordinate(const float, const float);
float calculateRCoordinate(const float, const float);
GPU_HOST_DEVICE constexpr float getNormalizedPhiCoordinate(const float);
GPU_HOST_DEVICE constexpr float3 crossProduct(const float3&, const float3&);
float computeCurvature(float x1, float y1, float x2, float y2, float x3, float y3);
float computeCurvatureCentreX(float x1, float y1, float x2, float y2, float x3, float y3);
float computeTanDipAngle(float x1, float y1, float x2, float y2, float z1, float z2);

} // namespace MathUtils

inline float MathUtils::calculatePhiCoordinate(const float xCoordinate, const float yCoordinate)
{
  return std::atan2(-yCoordinate, -xCoordinate) + Constants::Math::Pi;
}

inline float MathUtils::calculateRCoordinate(const float xCoordinate, const float yCoordinate)
{
  return std::sqrt(xCoordinate * xCoordinate + yCoordinate * yCoordinate);
}

GPU_HOST_DEVICE constexpr float MathUtils::getNormalizedPhiCoordinate(const float phiCoordinate)
{
  return (phiCoordinate < 0)
           ? phiCoordinate + Constants::Math::TwoPi
           : (phiCoordinate > Constants::Math::TwoPi) ? phiCoordinate - Constants::Math::TwoPi : phiCoordinate;
}

GPU_HOST_DEVICE constexpr float3 MathUtils::crossProduct(const float3& firstVector, const float3& secondVector)
{

  return float3{ (firstVector.y * secondVector.z) - (firstVector.z * secondVector.y),
                 (firstVector.z * secondVector.x) - (firstVector.x * secondVector.z),
                 (firstVector.x * secondVector.y) - (firstVector.y * secondVector.x) };
}

inline float MathUtils::computeCurvature(float x1, float y1, float x2, float y2, float x3, float y3)
{
  const float d = (x2 - x1) * (y3 - y2) - (x3 - x2) * (y2 - y1);
  const float a =
    0.5f * ((y3 - y2) * (y2 * y2 - y1 * y1 + x2 * x2 - x1 * x1) - (y2 - y1) * (y3 * y3 - y2 * y2 + x3 * x3 - x2 * x2));
  const float b =
    0.5f * ((x2 - x1) * (y3 * y3 - y2 * y2 + x3 * x3 - x2 * x2) - (x3 - x2) * (y2 * y2 - y1 * y1 + x2 * x2 - x1 * x1));

  return -1.f * d / std::sqrt((d * x1 - a) * (d * x1 - a) + (d * y1 - b) * (d * y1 - b));
}

inline float MathUtils::computeCurvatureCentreX(float x1, float y1, float x2, float y2, float x3, float y3)
{
  const float k1 = (y2 - y1) / (x2 - x1), k2 = (y3 - y2) / (x3 - x2);
  return 0.5f * (k1 * k2 * (y1 - y3) + k2 * (x1 + x2) - k1 * (x2 + x3)) / (k2 - k1);
}

inline float MathUtils::computeTanDipAngle(float x1, float y1, float x2, float y2, float z1, float z2)
{
  return (z1 - z2) / std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

} // namespace CA
} // namespace ITS
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_CAUTILS_H_ */
