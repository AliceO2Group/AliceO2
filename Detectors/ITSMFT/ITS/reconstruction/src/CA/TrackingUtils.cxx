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
/// \file TrackingUtils.cxx
/// \brief
///

#include "ITSReconstruction/CA/TrackingUtils.h"

#include <cmath>

#include "ITSReconstruction/CA/Constants.h"
#include "ITSReconstruction/CA/IndexTableUtils.h"
#include "ITSReconstruction/CA/MathUtils.h"

namespace o2
{
namespace ITS
{
namespace CA
{

GPU_DEVICE const int4 TrackingUtils::getBinsRect(const Cluster& currentCluster, const int layerIndex,
                                                 const float directionZIntersection)
{
  const float zRangeMin = directionZIntersection - 2 * Constants::Thresholds::ZCoordinateCut;
  const float phiRangeMin = currentCluster.phiCoordinate - Constants::Thresholds::PhiCoordinateCut;
  const float zRangeMax = directionZIntersection + 2 * Constants::Thresholds::ZCoordinateCut;
  const float phiRangeMax = currentCluster.phiCoordinate + Constants::Thresholds::PhiCoordinateCut;

  if (zRangeMax < -Constants::ITS::LayersZCoordinate()[layerIndex + 1] ||
      zRangeMin > Constants::ITS::LayersZCoordinate()[layerIndex + 1] || zRangeMin > zRangeMax) {

    return getEmptyBinsRect();
  }

  return int4{ MATH_MAX(0, IndexTableUtils::getZBinIndex(layerIndex + 1, zRangeMin)),
               IndexTableUtils::getPhiBinIndex(MathUtils::getNormalizedPhiCoordinate(phiRangeMin)),
               MATH_MIN(Constants::IndexTable::ZBins - 1, IndexTableUtils::getZBinIndex(layerIndex + 1, zRangeMax)),
               IndexTableUtils::getPhiBinIndex(MathUtils::getNormalizedPhiCoordinate(phiRangeMax)) };
}

float TrackingUtils::computeCurvature(float x1, float y1, float x2, float y2, float x3, float y3)
{
  const float d = (x2 - x1) * (y3 - y2) - (x3 - x2) * (y2 - y1);
  const float a =
    0.5f * ((y3 - y2) * (y2 * y2 - y1 * y1 + x2 * x2 - x1 * x1) - (y2 - y1) * (y3 * y3 - y2 * y2 + x3 * x3 - x2 * x2));
  const float b =
    0.5f * ((x2 - x1) * (y3 * y3 - y2 * y2 + x3 * x3 - x2 * x2) - (x3 - x2) * (y2 * y2 - y1 * y1 + x2 * x2 - x1 * x1));

  return -1.f * d / std::sqrt((d * x1 - a) * (d * x1 - a) + (d * y1 - b) * (d * y1 - b));
}

float TrackingUtils::computeCurvatureCentreX(float x1, float y1, float x2, float y2, float x3, float y3)
{
  const float k1 = (y2 - y1) / (x2 - x1), k2 = (y3 - y2) / (x3 - x2);
  return 0.5f * (k1 * k2 * (y1 - y3) + k2 * (x1 + x2) - k1 * (x2 + x3)) / (k2 - k1);
}

float TrackingUtils::computeTanDipAngle(float x1, float y1, float x2, float y2, float z1, float z2)
{
  return (z1 - z2) / std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}
}
}
}
