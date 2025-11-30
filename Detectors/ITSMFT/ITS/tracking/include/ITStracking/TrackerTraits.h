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
/// \file TrackerTraits.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_TRACKERTRAITS_H_
#define TRACKINGITSU_INCLUDE_TRACKERTRAITS_H_

#include <oneapi/tbb.h>

#include "DetectorsBase/Propagator.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/MathUtils.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/TimeFrame.h"
#include "ITStracking/Cell.h"
#include "ITStracking/BoundedAllocator.h"

// #define OPTIMISATION_OUTPUT

namespace o2
{
namespace gpu
{
class GPUChainITS;
}
namespace its
{
class TrackITSExt;

template <int nLayers = 7>
class TrackerTraits
{
 public:
  using IndexTableUtilsN = IndexTableUtils<nLayers>;
  using CellSeedN = CellSeed<nLayers>;

  virtual ~TrackerTraits() = default;
  virtual void adoptTimeFrame(TimeFrame<nLayers>* tf) { mTimeFrame = tf; }
  virtual void initialiseTimeFrame(const int iteration) { mTimeFrame->initialise(iteration, mTrkParams[iteration], mTrkParams[iteration].NLayers); }

  virtual void computeLayerTracklets(const int iteration, int iROFslice, int iVertex);
  virtual void computeLayerCells(const int iteration);
  virtual void findCellsNeighbours(const int iteration);
  virtual void findRoads(const int iteration);

  virtual bool supportsExtendTracks() const noexcept { return true; }
  virtual void extendTracks(const int iteration);
  virtual bool supportsFindShortPrimaries() const noexcept { return true; }
  virtual void findShortPrimaries();

  virtual bool trackFollowing(TrackITSExt* track, int rof, bool outward, const int iteration);
  virtual void processNeighbours(int iLayer, int iLevel, const bounded_vector<CellSeedN>& currentCellSeed, const bounded_vector<int>& currentCellId, bounded_vector<CellSeedN>& updatedCellSeed, bounded_vector<int>& updatedCellId);

  void updateTrackingParameters(const std::vector<TrackingParameters>& trkPars) { mTrkParams = trkPars; }
  TimeFrame<nLayers>* getTimeFrame() { return mTimeFrame; }

  virtual void setBz(float bz);
  float getBz() const { return mBz; }
  bool isMatLUT() const;
  virtual const char* getName() const noexcept { return "CPU"; }
  virtual bool isGPU() const noexcept { return false; }
  void setMemoryPool(std::shared_ptr<BoundedMemoryResource> pool) noexcept { mMemoryPool = pool; }
  auto getMemoryPool() const noexcept { return mMemoryPool; }

  // Others
  GPUhd() static consteval int4 getEmptyBinsRect() { return int4{0, 0, 0, 0}; }
  const int4 getBinsRect(int layer, float phi, float maxdeltaphi, float z, float maxdeltaz) const noexcept { return getBinsRect(layer, phi, maxdeltaphi, z, z, maxdeltaz); }
  const int4 getBinsRect(const Cluster& cls, int layer, float z1, float z2, float maxdeltaz, float maxdeltaphi) const noexcept { return getBinsRect(layer, cls.phi, maxdeltaphi, z1, z2, maxdeltaz); }
  const int4 getBinsRect(int layer, float phi, float maxdeltaphi, float z1, float z2, float maxdeltaz) const noexcept;
  void SetRecoChain(o2::gpu::GPUChainITS* chain) { mChain = chain; }
  void setSmoothing(bool v) { mApplySmoothing = v; }
  bool getSmoothing() const { return mApplySmoothing; }
  void setNThreads(int n, std::shared_ptr<tbb::task_arena>& arena);
  int getNThreads() { return mTaskArena->max_concurrency(); }

  o2::gpu::GPUChainITS* getChain() const { return mChain; }

  // TimeFrame information forwarding
  virtual int getTFNumberOfClusters() const { return mTimeFrame->getNumberOfClusters(); }
  virtual int getTFNumberOfTracklets() const { return mTimeFrame->getNumberOfTracklets(); }
  virtual int getTFNumberOfCells() const { return mTimeFrame->getNumberOfCells(); }

 private:
  track::TrackParCov buildTrackSeed(const Cluster& cluster1, const Cluster& cluster2, const TrackingFrameInfo& tf3);
  TrackITSExt seedTrackForRefit(const CellSeedN& seed);
  bool fitTrack(TrackITSExt& track, int start, int end, int step, float chi2clcut = o2::constants::math::VeryBig, float chi2ndfcut = o2::constants::math::VeryBig, float maxQoverPt = o2::constants::math::VeryBig, int nCl = 0, o2::track::TrackPar* refLin = nullptr);

  bool mApplySmoothing = false;
  std::shared_ptr<BoundedMemoryResource> mMemoryPool;
  std::shared_ptr<tbb::task_arena> mTaskArena;

 protected:
  o2::gpu::GPUChainITS* mChain = nullptr;
  TimeFrame<nLayers>* mTimeFrame;
  std::vector<TrackingParameters> mTrkParams;

  float mBz{-999.f};
  bool mIsZeroField{false};
};

template <int nLayers>
inline const int4 TrackerTraits<nLayers>::getBinsRect(const int layerIndex, float phi, float maxdeltaphi, float z1, float z2, float maxdeltaz) const noexcept
{
  const float zRangeMin = o2::gpu::GPUCommonMath::Min(z1, z2) - maxdeltaz;
  const float phiRangeMin = (maxdeltaphi > o2::constants::math::PI) ? 0.f : phi - maxdeltaphi;
  const float zRangeMax = o2::gpu::GPUCommonMath::Max(z1, z2) + maxdeltaz;
  const float phiRangeMax = (maxdeltaphi > o2::constants::math::PI) ? o2::constants::math::TwoPI : phi + maxdeltaphi;

  if (zRangeMax < -mTrkParams[0].LayerZ[layerIndex] ||
      zRangeMin > mTrkParams[0].LayerZ[layerIndex] || zRangeMin > zRangeMax) {
    return getEmptyBinsRect();
  }

  const IndexTableUtilsN& utils{mTimeFrame->mIndexTableUtils};
  return int4{o2::gpu::GPUCommonMath::Max(0, utils.getZBinIndex(layerIndex, zRangeMin)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMin)),
              o2::gpu::GPUCommonMath::Min(mTrkParams[0].ZBins - 1, utils.getZBinIndex(layerIndex, zRangeMax)), // /!\ trkParams can potentially change across iterations
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMax))};
}

} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_TRACKERTRAITS_H_ */
