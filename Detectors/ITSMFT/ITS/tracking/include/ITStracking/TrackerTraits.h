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
/// \file TrackerTraits.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_TRACKERTRAITS_H_
#define TRACKINGITSU_INCLUDE_TRACKERTRAITS_H_

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <utility>

#include "ITStracking/Configuration.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/MathUtils.h"
#include "ITStracking/PrimaryVertexContext.h"
#include "ITStracking/Road.h"

namespace o2
{
namespace ITS
{

class TrackerTraits
{
 public:
  virtual ~TrackerTraits() {}

  GPU_HOST_DEVICE static constexpr int4 getEmptyBinsRect() { return int4{ 0, 0, 0, 0 }; }
  GPU_DEVICE static const int4 getBinsRect(const Cluster&, const int, const float, float maxdeltaz, float maxdeltaphi);

  virtual void computeLayerTracklets() {};
  virtual void computeLayerCells() {};

  void UpdateTrackingParameters(const TrackingParameters& trkPar);
  PrimaryVertexContext* getPrimaryVertexContext() { return mPrimaryVertexContext; }

 protected:
  PrimaryVertexContext* mPrimaryVertexContext;
  TrackingParameters mTrkParams;
};

inline void TrackerTraits::UpdateTrackingParameters(const TrackingParameters& trkPar) {
  mTrkParams = trkPar;
}

inline GPU_DEVICE const int4 TrackerTraits::getBinsRect(const Cluster& currentCluster, const int layerIndex,
                                                        const float directionZIntersection, float maxdeltaz, float maxdeltaphi)
{
  const float zRangeMin = directionZIntersection - 2 * maxdeltaz;
  const float phiRangeMin = currentCluster.phiCoordinate - maxdeltaphi;
  const float zRangeMax = directionZIntersection + 2 * maxdeltaz;
  const float phiRangeMax = currentCluster.phiCoordinate + maxdeltaphi;

  if (zRangeMax < -Constants::ITS::LayersZCoordinate()[layerIndex + 1] ||
      zRangeMin > Constants::ITS::LayersZCoordinate()[layerIndex + 1] || zRangeMin > zRangeMax) {

    return getEmptyBinsRect();
  }

  return int4{ MATH_MAX(0, IndexTableUtils::getZBinIndex(layerIndex + 1, zRangeMin)),
               IndexTableUtils::getPhiBinIndex(MathUtils::getNormalizedPhiCoordinate(phiRangeMin)),
               MATH_MIN(Constants::IndexTable::ZBins - 1, IndexTableUtils::getZBinIndex(layerIndex + 1, zRangeMax)),
               IndexTableUtils::getPhiBinIndex(MathUtils::getNormalizedPhiCoordinate(phiRangeMax)) };
}

}
}

#endif /* TRACKINGITSU_INCLUDE_TRACKERTRAITS_H_ */
