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
/// \file TrackerTraitsEC0.h
/// \brief
///

#ifndef TRACKINGEC0__INCLUDE_TRACKERTRAITS_H_
#define TRACKINGEC0__INCLUDE_TRACKERTRAITS_H_

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <utility>
#include <functional>

#include "EC0tracking/Configuration.h"
#include "EC0tracking/Definitions.h"
#include "EC0tracking/MathUtils.h"
#include "EC0tracking/PrimaryVertexContext.h"
#include "EC0tracking/Road.h"


namespace o2::its
{
class TrackITSExt;
} // namespace o2::its


namespace o2
{
namespace gpu
{
class GPUChainEC0;
}
namespace ecl
{

class TrackITSExt;
typedef std::function<int(o2::gpu::GPUChainEC0&, std::vector<o2::ecl::Road>& roads, std::array<const o2::ecl::Cluster*, 7> clusters, std::array<const o2::ecl::Cell*, 5> cells, const std::array<std::vector<o2::ecl::TrackingFrameInfo>, 7>& tf, std::vector<o2::its::TrackITSExt>& tracks)> FuncRunEC0TrackFit_t;



class TrackerTraitsEC0
{
 public:
  virtual ~TrackerTraitsEC0() = default;

  GPU_HOST_DEVICE static constexpr int4 getEmptyBinsRect() { return int4{0, 0, 0, 0}; }
  GPU_DEVICE static const int4 getBinsRect(const Cluster&, const int, const float, float maxdeltaz, float maxdeltaphi);

  void SetRecoChain(o2::gpu::GPUChainEC0* chain, FuncRunEC0TrackFit_t&& funcRunEC0TrackFit)
  {
    mChainRunEC0TrackFit = funcRunEC0TrackFit;
    mChain = chain;
  }

  virtual void computeLayerTracklets(){};
  virtual void computeLayerCells(){};
  virtual void refitTracks(const std::array<std::vector<TrackingFrameInfo>, 7>&, std::vector<o2::its::TrackITSExt>&){};

  void UpdateTrackingParameters(const TrackingParameters& trkPar);
  PrimaryVertexContext* getPrimaryVertexContext() { return mPrimaryVertexContext; }

 protected:
  PrimaryVertexContext* mPrimaryVertexContext;
  TrackingParameters mTrkParams;

  o2::gpu::GPUChainEC0* mChain = nullptr;
  FuncRunEC0TrackFit_t mChainRunEC0TrackFit;
};

inline void TrackerTraitsEC0::UpdateTrackingParameters(const TrackingParameters& trkPar)
{
  mTrkParams = trkPar;
}

inline GPU_DEVICE const int4 TrackerTraitsEC0::getBinsRect(const Cluster& currentCluster, const int layerIndex,
                                                        const float directionZIntersection, float maxdeltaz, float maxdeltaphi)
{
  const float zRangeMin = directionZIntersection - 2 * maxdeltaz;
  const float phiRangeMin = currentCluster.phiCoordinate - maxdeltaphi;
  const float zRangeMax = directionZIntersection + 2 * maxdeltaz;
  const float phiRangeMax = currentCluster.phiCoordinate + maxdeltaphi;

  if (zRangeMax < -constants::ecl::LayersZCoordinate()[layerIndex + 1] ||
      zRangeMin > constants::ecl::LayersZCoordinate()[layerIndex + 1] || zRangeMin > zRangeMax) {

    return getEmptyBinsRect();
  }

  return int4{gpu::GPUCommonMath::Max(0, index_table_utils::getZBinIndex(layerIndex + 1, zRangeMin)),
              index_table_utils::getPhiBinIndex(math_utils::getNormalizedPhiCoordinate(phiRangeMin)),
              gpu::GPUCommonMath::Min(constants::index_table::ZBins - 1, index_table_utils::getZBinIndex(layerIndex + 1, zRangeMax)),
              index_table_utils::getPhiBinIndex(math_utils::getNormalizedPhiCoordinate(phiRangeMax))};
}
} // namespace ecl
} // namespace o2

#endif /* TRACKINGEC0__INCLUDE_TRACKERTRAITS_H_ */
