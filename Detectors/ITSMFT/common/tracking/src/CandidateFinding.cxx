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

#include "ITSMFTTracking/detail/CandidateFinding.h"

#include "DataFormatsITS/Vertex.h"
#include "ITSMFTTracking/IndexTableUtils.h"
#include "ITSMFTTracking/Constants.h"
#include "ITSMFTTracking/MathUtils.h"

namespace o2::itsmft::tracking
{

bool projectTrackletSearchWindow(
  const GlobalMeasurement& sourceMeasurement,
  const o2::its::Vertex& vertex,
  float beamPositionVariance,
  SurfaceKind kind,
  const TrackletProjectionCache& edgeCache,
  const o2::itsmft::IndexTableUtilsCore& indexUtils,
  float nSigmaCut,
  TrackletSearchWindow& out)
{
  const bool disk = kind == SurfaceKind::Disk;
  const float referenceCoordinate = disk ? sourceMeasurement.z : sourceMeasurement.radius;
  const float referenceOrigin = disk ? vertex.getZ() : 0.f;
  const float projectedCoordinate = disk ? sourceMeasurement.radius : sourceMeasurement.z;
  const float projectedOrigin = disk ? 0.f : vertex.getZ();
  const float targetMin = disk ? edgeCache.targetMinZ : edgeCache.targetMinR;
  const float targetMax = disk ? edgeCache.targetMaxZ : edgeCache.targetMaxR;
  const float referenceDelta = referenceCoordinate - referenceOrigin;
  const float projectedDelta = projectedCoordinate - projectedOrigin;
  if (!(targetMin <= targetMax) ||
      !(o2::gpu::CAMath::Abs(referenceDelta) > o2::its::constants::Tolerance) ||
      (disk && !(projectedDelta > o2::its::constants::Tolerance))) {
    return false;
  }

  const float slope = projectedDelta / referenceDelta; // tan(lambda) for cylinders, 1/tan(lambda) for disks
  const float targetCoordinate = 0.5f * (targetMin + targetMax);
  const float referenceToTarget = targetCoordinate - referenceCoordinate;
  const float prediction = projectedCoordinate + slope * referenceToTarget;
  if (disk && !(prediction > 0.f)) {
    return false;
  }

  const float sourceCoordinateVariance = o2::its::math_utils::Sq(edgeCache.sourcePositionResolution);
  const float referenceOriginVariance = disk ? vertex.getSigmaZ2() : beamPositionVariance;
  const float projectedOriginVariance = disk ? beamPositionVariance : vertex.getSigmaZ2();
  const float inverseReferenceDelta = 1.f / referenceDelta;
  const float sourceVarianceScale = (1.f + o2::its::math_utils::Sq(slope)) * sourceCoordinateVariance;
  const float originVarianceScale = projectedOriginVariance + o2::its::math_utils::Sq(slope) * referenceOriginVariance;
  const float edgeMSVarianceScale = o2::its::math_utils::Sq(edgeCache.edgeMSAngle);
  const float varianceConstant = sourceVarianceScale;
  const float varianceLinear = 2.f * inverseReferenceDelta * sourceVarianceScale;
  const float varianceQuadratic = o2::its::math_utils::Sq(inverseReferenceDelta) *
                                    (sourceVarianceScale + originVarianceScale) +
                                  edgeMSVarianceScale;
  const float minDelta = targetMin - referenceCoordinate;
  const float minPrediction = projectedCoordinate + slope * minDelta;
  const float minVariance = varianceConstant + minDelta * (varianceLinear + minDelta * varianceQuadratic);
  const float maxDelta = targetMax - referenceCoordinate;
  const float maxPrediction = projectedCoordinate + slope * maxDelta;
  const float maxVariance = varianceConstant + maxDelta * (varianceLinear + maxDelta * varianceQuadratic);
  const float lowerBound = o2::gpu::CAMath::Min(minPrediction - nSigmaCut * o2::gpu::CAMath::Sqrt(minVariance),
                                                maxPrediction - nSigmaCut * o2::gpu::CAMath::Sqrt(maxVariance));
  const float upperBound = o2::gpu::CAMath::Max(minPrediction + nSigmaCut * o2::gpu::CAMath::Sqrt(minVariance),
                                                maxPrediction + nSigmaCut * o2::gpu::CAMath::Sqrt(maxVariance));
  const float searchPrediction = 0.5f * (lowerBound + upperBound);
  const float searchHalfWidth = 0.5f * (upperBound - lowerBound);

  const auto bins = o2::itsmft::getBinsPhiColumn(sourceMeasurement.phi, edgeCache.toLayer,
                                                 searchPrediction, searchHalfWidth,
                                                 edgeCache.edgePhiCut, indexUtils);
  if (bins.x < 0) {
    return false;
  }
  out = {bins,
         referenceCoordinate,
         projectedCoordinate,
         slope,
         varianceConstant,
         varianceLinear,
         varianceQuadratic,
         sourceMeasurement.phi,
         o2::its::math_utils::Sq(edgeCache.edgePhiCut / nSigmaCut)};
  return true;
}

} // namespace o2::itsmft::tracking
