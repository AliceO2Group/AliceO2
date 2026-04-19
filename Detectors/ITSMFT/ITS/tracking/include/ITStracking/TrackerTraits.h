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

template <int NLayers>
class TrackerTraits
{
 public:
  using IndexTableUtilsN = IndexTableUtils<NLayers>;
  using TrackSeedN = TrackSeed<NLayers>;

  virtual ~TrackerTraits() = default;
  virtual void adoptTimeFrame(TimeFrame<NLayers>* tf) { mTimeFrame = tf; }
  virtual void initialiseTimeFrame(const int iteration) { mTimeFrame->initialise(iteration, mTrkParams[iteration], mTrkParams[iteration].NLayers, false); }

  virtual void computeLayerTracklets(const int iteration, int iVertex);
  virtual void computeLayerCells(const int iteration);
  virtual void findCellsNeighbours(const int iteration);
  virtual void findRoads(const int iteration);

  template <typename InputSeed>
  void processNeighbours(int iteration, int iLayer, int iLevel, const bounded_vector<InputSeed>& currentCellSeed, const bounded_vector<int>& currentCellId, bounded_vector<TrackSeedN>& updatedCellSeed, bounded_vector<int>& updatedCellId);

  void acceptTracks(int iteration, bounded_vector<TrackITSExt>& tracks, bounded_vector<bounded_vector<int>>& firstClusters, bounded_vector<bounded_vector<int>>& sharedFirstClusters);
  void markTracks(int iteration, bounded_vector<bounded_vector<int>>& sharedFirstClusters);

  void updateTrackingParameters(const std::vector<TrackingParameters>& trkPars)
  {
    mTrkParams = trkPars;
  }
  TimeFrame<NLayers>* getTimeFrame() { return mTimeFrame; }

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
  std::shared_ptr<tbb::task_arena> mTaskArena;

 protected:
  o2::gpu::GPUChainITS* mChain = nullptr;
  TimeFrame<NLayers>* mTimeFrame;
  std::vector<TrackingParameters> mTrkParams;

  float mBz{-999.f};
};

} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_TRACKERTRAITS_H_ */
