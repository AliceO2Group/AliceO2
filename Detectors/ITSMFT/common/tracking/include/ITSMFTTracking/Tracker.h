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
/// \file Tracker.h
/// \brief Tracker orchestrator.
///

#ifndef ALICEO2_ITSMFT_TRACKING_TRACKER_H_
#define ALICEO2_ITSMFT_TRACKING_TRACKER_H_

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <gsl/span>

#include <oneapi/tbb/task_arena.h>

#include "ITSMFTTracking/Configuration.h"
#include "ITSMFTTracking/IterationConfiguration.h"
#include "ITSMFTTracking/DetectorConfiguration.h"
#include "ITSMFTTracking/detail/TimeFrameScratch.h"
#include "ITSMFTTracking/TimeFrame.h"
#include "ITSMFTTracking/TrackerTraits.h"

namespace o2::itsmft::tracking
{

struct TrackerTestAccess;

/// Statistics for the last successful run. Reset at the start of every run;
/// remain empty with zero elapsed time if that run fails.
struct TrackingStatistics {
  float elapsedMs{0.f};
  // Accepted-result counts are indexed by configured iteration.
  std::vector<std::size_t> acceptedTrackCounts;
};

struct TrackerInitialization {
  SurfaceCatalogView catalog;
  // First position of each component; zero is always required.
  std::vector<uint16_t> componentOffsets{0};
  LayerMask holeLayers{};
  TrackingPlan plan;
  std::shared_ptr<BoundedMemoryResource> memoryPool;
};

class Tracker
{
 public:
  /// Returns true after installing the complete configuration; logs the reason
  /// and returns false for invalid input, leaving existing configuration intact.
  bool initialize(TimeFrame& frame, const TrackerInitialization& configuration);

  gsl::span<const IterationConfiguration> getIterationConfigurations() const noexcept { return mIterations; }
  const TrackingExecutionPolicy& getExecutionPolicy() const noexcept { return mExecutionPolicy; }
  const IterationConfiguration* getIterationConfiguration(std::size_t iteration) const noexcept
  {
    return iteration < mIterations.size() ? &mIterations[iteration] : nullptr;
  }
  bool isConfiguredFor(const TimeFrame& frame) const noexcept;

  /// Run all configured iterations. Returns true on success, false when a
  /// per-TF resource failure (MemoryLimitExceeded or std::bad_alloc) is dropped
  /// with DropTFUponFailure enabled. Other failures propagate as exceptions.
  /// The event is reset after a failure during tracking.
  bool run(TimeFrame& frame, TrackerTraits& traits);
  const TrackingStatistics& getRunStatistics() const noexcept { return mRunStatistics; }

 private:
  friend struct TrackerTestAccess;
  gsl::span<const gsl::span<const GlobalMeasurement>> prepareTimeFrame(
    TimeFrame& frame, std::array<gsl::span<const GlobalMeasurement>, MaxLayoutSurfaces>& measurements) const;
  void configureBeamPosition(TimeFrame& frame) const;
  void initializeIteration(IterationContext& context) const;
  void computeTracksMClabels(TimeFrame& frame) const;
  TrackingExecutionPolicy mExecutionPolicy;
  std::vector<IterationConfiguration> mIterations;
  TrackingStatistics mRunStatistics;
  const TimeFrame* mFrame = nullptr;
};
} // namespace o2::itsmft::tracking

#endif /* ALICEO2_ITSMFT_TRACKING_TRACKER_H_ */
