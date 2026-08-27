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
#include <utility>
#include <vector>

#include "DetectorsBase/Propagator.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITSMFTTracking/CapacityEstimator.h"
#include "ITStracking/TimeFrame.h"
#include "ITStracking/Cell.h"
#include "ITStracking/BoundedAllocator.h"
#include "ITStracking/TrackExtensionHypothesis.h"
#include "ITStracking/TrackFollower.h"
#include "ITStracking/TrackITSInternal.h"

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

template <int NLayers>
struct RoadSeed {
  TrackSeed<NLayers> seed;
  int cellId{constants::UnusedIndex};
  int cellTopologyId{constants::UnusedIndex};

  RoadSeed() = default;
  RoadSeed(TrackSeed<NLayers>&& inputSeed, int inputCellId, int inputCellTopologyId)
    : seed{std::move(inputSeed)}, cellId{inputCellId}, cellTopologyId{inputCellTopologyId} {}
};

template <int NLayers>
class TrackerTraits
{
 public:
  using IndexTableUtilsN = IndexTableUtils<NLayers>;
  using TrackSeedN = TrackSeed<NLayers>;
  using RoadSeedN = RoadSeed<NLayers>;

  virtual ~TrackerTraits() = default;
  virtual void adoptTimeFrame(TimeFrame<NLayers>* tf) { mTimeFrame = tf; }
  virtual void initialiseTimeFrame(const int iteration);

  virtual void computeLayerTracklets(const int iteration, int iVertex);
  virtual void computeLayerCells(const int iteration);
  virtual void findCellsNeighbours(const int iteration);
  virtual void findRoads(const int iteration);

  template <typename InputSeed>
  void processNeighbours(int iteration, int defaultCellTopologyId, int iLevel, uint64_t capacityKey, const bounded_vector<InputSeed>& currentSeeds, bounded_vector<RoadSeedN>& updatedSeeds);

  void acceptTracks(int iteration, bounded_vector<TrackITSExt>& tracks, const bounded_vector<int>& trackIndices, bounded_vector<bounded_vector<int>>& firstClusters);
  void markTracks(int iteration);

  void updateTrackingParameters(const std::vector<TrackingParameters>& trkPars)
  {
    mTrkParams = trkPars;
  }

  virtual void setBz(float bz);
  float getBz() const { return mBz; }
  virtual const char* getName() const noexcept { return "CPU"; }
  virtual bool isGPU() const noexcept { return false; }
  void setMemoryPool(std::shared_ptr<BoundedMemoryResource> pool) noexcept { mMemoryPool = pool; }
  auto getMemoryPool() const noexcept { return mMemoryPool; }

  // Others
  void setNThreads(int n, std::shared_ptr<tbb::task_arena>& arena);
  int getNThreads() { return mTaskArena->max_concurrency(); }

  // TimeFrame information forwarding
  virtual int getTFNumberOfClusters() const { return mTimeFrame->getNumberOfClusters(); }
  virtual int getTFNumberOfTracklets() const { return mTimeFrame->getNumberOfTracklets(); }
  virtual int getTFNumberOfCells() const { return mTimeFrame->getNumberOfCells(); }

 private:
  std::shared_ptr<BoundedMemoryResource> mMemoryPool;

 protected:
  std::shared_ptr<tbb::task_arena> mTaskArena;

  struct TrackFollowerScratch {
    explicit TrackFollowerScratch(std::pmr::memory_resource* memoryResource)
      : activeHypotheses(memoryResource), nextHypotheses(memoryResource)
    {
    }

    bounded_vector<TrackExtensionHypothesis<NLayers>> activeHypotheses;
    bounded_vector<TrackExtensionHypothesis<NLayers>> nextHypotheses;
  };

  bool finaliseTrackSeed(const TrackSeedN& seed,
                         TrackITSExt& track,
                         const int iteration,
                         const TrackingFrameInfo* const* tfInfos,
                         const Cluster* const* unsortedClusters,
                         const o2::base::Propagator* propagator,
                         const TrackFollowContext<NLayers>& followCtx,
                         TrackFollowerScratch& scratch);

  o2::gpu::GPUChainITS* mChain = nullptr;
  TimeFrame<NLayers>* mTimeFrame;
  std::vector<TrackingParameters> mTrkParams;

  float mBz{-999.f};
};

} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_TRACKERTRAITS_H_ */
