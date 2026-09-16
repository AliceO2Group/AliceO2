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
/// \brief Shared CA tracker traits: same ITS-style tracklet/cell/road logic; MFT uses x-y LUT and forward refit
///

#ifndef ALICEO2_ITSMFT_TRACKING_TRACKERTRAITS_H_
#define ALICEO2_ITSMFT_TRACKING_TRACKERTRAITS_H_

#include <array>
#include <optional>
#include <utility>
#include <vector>

#include <gsl/span>
#include <oneapi/tbb.h>

#include "ITSMFTTracking/Configuration.h"
#include "ITSMFTTracking/GenericTrack.h"
#include "ITSMFTTracking/IterationConfiguration.h"
#include "ITSMFTTracking/detail/TimeFrameScratch.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"
#include "ITSMFTTracking/SurfaceMeasurement.h"
#include "ITSMFTTracking/TimeFrame.h"
#include "ITSMFTTracking/detail/TrackingKernelParameters.h"
#include "ITSMFTTracking/BoundedAllocator.h"

namespace o2::itsmft::tracking
{

struct TrackerTestAccess;

struct IterationContext {
  int iteration{-1};
  TimeFrame& frame;
  TimeFrameScratch& scratch;
  TraversalTopologyView topology{};
  const DetectorConfiguration& detectorConfiguration;
  const IterationConfiguration& configuration;
  // Borrows the caller's span sequence and the frame's measurements. Both must
  // outlive this synchronous traversal; loading/resetting the frame invalidates it.
  gsl::span<const gsl::span<const GlobalMeasurement>> layerGlobalMeasurements;
  float bz{0.f};

  IterationContext(int iterationValue, TimeFrame& frameValue, TimeFrameScratch& scratchValue,
                   TraversalTopologyView topologyValue, const IterationConfiguration& configurationValue,
                   gsl::span<const gsl::span<const GlobalMeasurement>> layerGlobalMeasurementsValue,
                   float bzValue)
    : iteration{iterationValue}, frame{frameValue}, scratch{scratchValue}, topology{topologyValue}, detectorConfiguration{frameValue.getDetectorConfiguration()}, configuration{configurationValue}, layerGlobalMeasurements{layerGlobalMeasurementsValue}, bz{bzValue}
  {
  }
};

// Backend implementation of a traversal supplied explicitly by Tracker.
class TrackerTraits
{
 public:
  virtual ~TrackerTraits() = default;
  // The production caller supplies all event and iteration state explicitly.
  void runTraversal(IterationContext& view);

  virtual const char* getName() const noexcept { return "CPU"; }
  virtual bool isGPU() const noexcept { return false; }
  void setNThreads(int n, std::shared_ptr<tbb::task_arena>& arena);
  int getNThreads() { return mTaskArena->max_concurrency(); }

 private:
  friend struct TrackerTestAccess;

  void acceptTracks(IterationContext& context, int iteration,
                    bounded_vector<TrackingCandidate>& tracks,
                    bounded_vector<bounded_vector<int>>& firstClusters);

  // Tracklet and cell enumeration are common; coordinate selection is owned
  // by their operation leaves.
  void computeLayerTracklets(IterationContext& context, int iteration, int iVertex);
  void computeLayerCells(IterationContext& context, int iteration);
  void findCellsNeighbours(IterationContext& context, int iteration);

  void findRoads(IterationContext& context, int iteration);

  bool buildTrackSeed(IterationContext& context, int cellPathId,
                      const Triplet& cell, TrackSeed& output) const;

  struct RoadSeedEmission;

  // Neighbour processing helper; it does not encode a detector layer count.
  template <typename InputSeed>
  void processNeighbours(IterationContext& context, int iteration, CellPathId startingPath,
                         int defaultCellPathId, int startLevel, int currentLevel,
                         const bounded_vector<InputSeed>& currentSeeds,
                         bounded_vector<RoadSeedEmission>& updatedCells,
                         const TrackingKernelParameters& params);

  std::shared_ptr<tbb::task_arena> mTaskArena;
};

} // namespace o2::itsmft::tracking

#endif /* ALICEO2_ITSMFT_TRACKING_TRACKERTRAITS_H_ */
