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

#include "ITSReconstruction/CA/Constants.h"

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
}

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
}
}
}

#endif /* TRACKINGITSU_INCLUDE_CAUTILS_H_ */
