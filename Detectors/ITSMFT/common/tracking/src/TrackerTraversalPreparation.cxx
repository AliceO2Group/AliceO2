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

#include "ITSMFTTracking/detail/TrackerTraversalPreparation.h"

#include <cmath>

#include "CommonConstants/MathConstants.h"
#include "GPUCommonMath.h"
#include "ITSMFTTracking/MathUtils.h"

namespace o2::itsmft::tracking
{

float cylinderLayerMultipleScatteringAngle(
  const CylinderLayerScatteringInputs& inputs, float trackletMinPt)
{
  return o2::its::math_utils::MSangle(0.14f, trackletMinPt, inputs.layerxX0);
}

float diskLayerMultipleScatteringAngle(const DiskLayerScatteringInputs& inputs, float trackletMinPt)
{
  const float invP = 1.f / trackletMinPt;
  const float tanlRef = (std::abs(inputs.layerRadius) > 1e-6f)
                          ? inputs.referenceCoordinate / inputs.layerRadius
                          : 0.f;
  const float absTanl = std::abs(tanlRef);
  const float cscLambda = (absTanl > 1e-6f)
                            ? std::sqrt(1.f + tanlRef * tanlRef) / absTanl
                            : 1e6f;
  return 0.0136f * invP * std::sqrt(inputs.layerxX0 * cscLambda);
}

float clampEdgeCurvature(float oneOverR, float outerRadius) noexcept
{
  return (outerRadius > 0.f && 0.5f * oneOverR >= 1.f / outerRadius)
           ? (2.f / outerRadius) - o2::constants::math::Almost0
           : oneOverR;
}

EdgeScatteringBendingPrep prepareEdgeScatteringAndBending(
  gsl::span<const float> perLayerMSAngle, int fromLayer, int toLayer,
  float r1, float r2, float clampedOneOverR, float res1, float res2) noexcept
{
  float ms2 = 0.f;
  for (int layer = fromLayer; layer < toLayer; ++layer) {
    ms2 += o2::its::math_utils::Sq(perLayerMSAngle[layer]);
  }
  const float msAngle = o2::gpu::CAMath::Sqrt(ms2);
  const float cosTheta1half = o2::gpu::CAMath::Sqrt(1.f - o2::its::math_utils::Sq(0.5f * r1 * clampedOneOverR));
  const float cosTheta2half = o2::gpu::CAMath::Sqrt(1.f - o2::its::math_utils::Sq(0.5f * r2 * clampedOneOverR));
  const float x = (r2 * cosTheta1half) - (r1 * cosTheta2half);
  const float delta = o2::gpu::CAMath::Sqrt(1.f / (1.f - 0.25f * o2::its::math_utils::Sq(x * clampedOneOverR)) *
                                            (o2::its::math_utils::Sq((0.25f * r1 * r2 * o2::its::math_utils::Sq(clampedOneOverR) / cosTheta2half) + cosTheta1half) * o2::its::math_utils::Sq(res1) +
                                             o2::its::math_utils::Sq((0.25f * r1 * r2 * o2::its::math_utils::Sq(clampedOneOverR) / cosTheta1half) + cosTheta2half) * o2::its::math_utils::Sq(res2)));
  const float phiCut = o2::gpu::CAMath::Min(o2::gpu::CAMath::ASin(0.5f * x * clampedOneOverR) + 2.f * msAngle + delta, o2::constants::math::PI * 0.5f);
  return {msAngle, phiCut};
}

} // namespace o2::itsmft::tracking
