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
#include <functional>

#include "ITStracking/Configuration.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/MathUtils.h"
#include "ITStracking/PrimaryVertexContext.h"
#include "ITStracking/Road.h"

namespace o2
{
namespace gpu
{
class GPUChainITS;
}
namespace its
{

class TrackITSExt;
typedef std::function<int(o2::gpu::GPUChainITS&, std::vector<Road>& roads, std::vector<const Cluster*>&, std::vector<const Cell*>&, const std::vector<std::vector<TrackingFrameInfo>>&, std::vector<TrackITSExt>&)> FuncRunITSTrackFit_t;

class TrackerTraits
{
 public:
  virtual ~TrackerTraits() = default;

  GPU_HOST_DEVICE static constexpr int4 getEmptyBinsRect() { return int4{0, 0, 0, 0}; }
  GPU_DEVICE const int4 getBinsRect(const Cluster&, const int, const float, const float, float maxdeltaz, float maxdeltaphi);

  void SetRecoChain(o2::gpu::GPUChainITS* chain, FuncRunITSTrackFit_t&& funcRunITSTrackFit)
  {
    mChainRunITSTrackFit = funcRunITSTrackFit;
    mChain = chain;
  }

  virtual void computeLayerTracklets(){};
  virtual void computeLayerCells(){};
  virtual void refitTracks(const std::vector<std::vector<TrackingFrameInfo>>&, std::vector<TrackITSExt>&){};

  void UpdateTrackingParameters(const TrackingParameters& trkPar);
  PrimaryVertexContext* getPrimaryVertexContext() { return mPrimaryVertexContext; }

 protected:
  PrimaryVertexContext* mPrimaryVertexContext;
  TrackingParameters mTrkParams;

  o2::gpu::GPUChainITS* mChain = nullptr;
  FuncRunITSTrackFit_t mChainRunITSTrackFit;
};

inline void TrackerTraits::UpdateTrackingParameters(const TrackingParameters& trkPar)
{
  mTrkParams = trkPar;
}

inline GPU_DEVICE const int4 TrackerTraits::getBinsRect(const Cluster& currentCluster, const int layerIndex,
                                                        const float z1, const float z2, float maxdeltaz, float maxdeltaphi)
{
  const float zRangeMin = o2::gpu::GPUCommonMath::Min(z1, z2) - maxdeltaz;
  const float phiRangeMin = currentCluster.phiCoordinate - maxdeltaphi;
  const float zRangeMax = o2::gpu::GPUCommonMath::Max(z1, z2) + maxdeltaz;
  const float phiRangeMax = currentCluster.phiCoordinate + maxdeltaphi;

  if (zRangeMax < -mTrkParams.LayerZ[layerIndex + 1] ||
      zRangeMin > mTrkParams.LayerZ[layerIndex + 1] || zRangeMin > zRangeMax) {

    return getEmptyBinsRect();
  }

  const IndexTableUtils& utils{mPrimaryVertexContext->mIndexTableUtils};
  return int4{o2::gpu::GPUCommonMath::Max(0, utils.getZBinIndex(layerIndex + 1, zRangeMin)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhiCoordinate(phiRangeMin)),
              o2::gpu::GPUCommonMath::Min(mTrkParams.ZBins - 1, utils.getZBinIndex(layerIndex + 1, zRangeMax)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhiCoordinate(phiRangeMax))};
}
} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_TRACKERTRAITS_H_ */
