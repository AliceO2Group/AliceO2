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

#ifndef ALICEO2_ITSMFT_TRACKING_WORKFLOWSESSION_H_
#define ALICEO2_ITSMFT_TRACKING_WORKFLOWSESSION_H_

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <gsl/span>
#include "CommonConstants/LHCConstants.h"
#include "Framework/Logger.h"
#include "ITSMFTTracking/TrackPublicationHelpers.h"
#include "ITSMFTTracking/IOUtils.h"
#include "ITSMFTTracking/ROFLookupTables.h"
#include "ITSMFTTracking/Tracker.h"

namespace o2::itsmft::tracking
{
enum class CATrackerPublicationAction {
  PublishInactiveEmpty,
  PublishActiveResult,
  SkipDroppedTimeFrame,
};
inline CATrackerPublicationAction decideCATrackerPublicationAction(bool active, bool success) noexcept
{
  if (!active) {
    return CATrackerPublicationAction::PublishInactiveEmpty;
  }
  return success ? CATrackerPublicationAction::PublishActiveResult : CATrackerPublicationAction::SkipDroppedTimeFrame;
}

// Validate actual source records against the unsigned BC range used by the
// legacy timing classes before passing them into the tracking workflow.
inline void validateSourceROFTiming(const ClusterSourceInput& source, const o2::InteractionRecord& origin,
                                    const o2::its::LayerTiming& timing)
{
  for (size_t rof = 0; rof < source.rofs.size(); ++rof) {
    const int64_t begin = source.rofs[rof].getBCData().differenceInBC(origin) +
                          static_cast<int64_t>(timing.mROFDelay) + timing.mROFBias;
    const int64_t end = begin + timing.mROFLength;
    if (timing.mROFLength == 0 || begin < 0 || end > std::numeric_limits<o2::its::TimeStampType>::max()) {
      throw std::runtime_error(std::format("Invalid ROF timing: source={} rof={}", source.id.value(), rof));
    }
  }
}

// The common columns are copied into framework-owned output storage before the
// session is reset. Detector-specific columns (MFT seed patterns, MC) stay explicit.
template <typename Allocator, typename Output, typename Staged>
void copyTrackingOutputColumns(Allocator& outputs, Output rofs, Output tracks, Output indices, const Staged& staged)
{
  outputs.template make<std::decay_t<decltype(staged.trackROFs)>>(rofs, staged.trackROFs.begin(), staged.trackROFs.end());
  outputs.template make<std::decay_t<decltype(staged.tracks)>>(tracks, staged.tracks.begin(), staged.tracks.end());
  outputs.template make<std::decay_t<decltype(staged.clusterIndices)>>(indices, staged.clusterIndices.begin(), staged.clusterIndices.end());
}

// Own every backing store borrowed by a single detector's workflow views.
// Detector-specific selection, truth vertices and output formats stay in the task.
class WorkflowSession
{
 public:
  WorkflowSession(const char* detectorName, int nLayers)
    : overlap(nLayers), vertices(nLayers), mask(nLayers), upcMask(nLayers), mDetectorName(detectorName) {}

  TimeFrame frame;
  std::vector<std::vector<uint32_t>> externalIndices;
  std::vector<std::vector<uint32_t>> clusterSizes;
  ROFOverlapTable overlap;
  ROFVertexLookupTable vertices;
  ROFMaskTable mask;
  ROFMaskTable upcMask;
  std::optional<ClockTimingPublicationView> publicationClock;

  class Cleanup
  {
   public:
    explicit Cleanup(WorkflowSession& session) : mSession(session) {}
    Cleanup(const Cleanup&) = delete;
    Cleanup& operator=(const Cleanup&) = delete;
    ~Cleanup() noexcept
    {
      if (mResetFrame) {
        mSession.reset();
      }
      mSession.invalidatePublication();
    }
    // Both the loader recovery and Tracker::run have already reset a dropped TF.
    void frameAlreadyReset() noexcept { mResetFrame = false; }

   private:
    WorkflowSession& mSession;
    bool mResetFrame = true;
  };
  Cleanup cleanupOnExit() { return Cleanup{*this}; }

  void reset() noexcept
  {
    externalIndices.clear();
    clusterSizes.clear();
    frame.resetTimeFrame();
  }
  void invalidatePublication() noexcept
  {
    publicationClock.reset();
    externalIndices.clear();
    clusterSizes.clear();
    frame.setROFViews({});
  }

  template <typename AlpideParameters>
  std::vector<o2::its::LayerTiming> layerTimings(const AlpideParameters& alpide, int nOrbits,
                                                 const std::vector<uint32_t>& addTimeError) const
  {
    const int nLayers = overlap.getEntries();
    if (addTimeError.size() != nLayers) {
      throw std::runtime_error{std::string(mDetectorName) + " CA timing-error layer count differs from the workflow layout"};
    }
    std::vector<o2::its::LayerTiming> timings(nLayers);
    for (int layer = 0; layer < nLayers; ++layer) {
      const auto length = alpide.getROFLengthInBC(layer);
      if (length <= 0) {
        throw std::runtime_error{std::string(mDetectorName) + " CA per-layer ROF timing has a non-positive ROF length"};
      }
      const auto rofsPerOrbit = o2::constants::lhc::LHCMaxBunches / static_cast<unsigned int>(length);
      timings[layer] = {.mNROFsTF = rofsPerOrbit * static_cast<unsigned int>(nOrbits),
                        .mROFLength = static_cast<uint32_t>(length),
                        .mROFDelay = static_cast<uint32_t>(alpide.getROFDelayInBC(layer)),
                        .mROFBias = static_cast<uint32_t>(alpide.getROFBiasInBC(layer)),
                        .mROFAddTimeErr = addTimeError[layer]};
      if (timings[layer].mNROFsTF == 0) {
        throw std::runtime_error{std::string(mDetectorName) + " CA per-layer ROF timing yields zero ROFs per TimeFrame"};
      }
    }
    return timings;
  }

  template <typename AcceptROF>
  void configureTiming(gsl::span<const o2::its::LayerTiming> timings, AcceptROF&& accept)
  {
    const int nLayers = overlap.getEntries();
    if (timings.size() != nLayers || timings.empty() ||
        !std::all_of(timings.begin(), timings.end(), [&](const auto& timing) {
          const auto& first = timings.front();
          return timing.mROFLength == first.mROFLength && timing.mROFDelay == first.mROFDelay &&
                 timing.mROFBias == first.mROFBias && timing.mROFAddTimeErr == first.mROFAddTimeErr;
        })) {
      throw std::runtime_error{std::string(mDetectorName) + " CA per-layer ROF timing configuration has an unexpected layer count or is not uniform"};
    }
    // Only owned timing structure survives between TFs. The key includes every
    // layer's extent and timing fields, so readout/CCDB changes rebuild it.
    publicationClock.reset();
    frame.setROFViews({});
    if (!matchesTiming(timings)) {
      ROFOverlapTable nextOverlap{nLayers};
      ROFVertexLookupTable nextVertices{nLayers};
      for (int layer = 0; layer < nLayers; ++layer) {
        nextOverlap.defineLayer(layer, timings[layer]);
        nextVertices.defineLayer(layer, timings[layer]);
      }
      nextOverlap.init();
      nextVertices.init();
      ROFMaskTable nextMask{nextOverlap};
      std::vector<o2::its::LayerTiming> nextTimingKey(timings.begin(), timings.end());
      overlap = std::move(nextOverlap);
      vertices = std::move(nextVertices);
      mask = std::move(nextMask);
      mTimingKey = std::move(nextTimingKey);
    }
    // Vertex contents and selection are event-local even on a cache hit. Views
    // are rebound only after refresh succeeds; a throwing filter leaves no
    // partially refreshed event published and the next call can reuse the key.
    vertices.update(nullptr, 0);
    mask.resetMask();
    for (int rof = 0; rof < static_cast<int>(timings[0].mNROFsTF); ++rof) {
      if (accept(rof)) {
        for (int layer = 0; layer < nLayers; ++layer) {
          mask.setROFEnabled(layer, rof, 1);
        }
      }
    }
    frame.setROFViews({overlap.getView(), vertices.getView(), mask.getView(), upcMask.getView()});
  }

  template <typename Load>
  bool loadWithRecovery(bool dropOnFailure, Load&& load)
  {
    try {
      load();
      return true;
    } catch (const BoundedMemoryResource::MemoryLimitExceeded& error) {
      LOGP(error, "{} CA loading exceeded memory limit: {}", mDetectorName, error.what());
      reset();
      if (!dropOnFailure) {
        throw;
      }
    } catch (const std::bad_alloc& error) {
      LOGP(error, "{} CA loading allocation failed: {}", mDetectorName, error.what());
      reset();
      if (!dropOnFailure) {
        throw;
      }
    } catch (const std::exception& error) {
      LOGP(error, "{} CA loading failed: {}", mDetectorName, error.what());
      reset();
      throw;
    }
    return false;
  }

  template <typename AfterLoad, typename Complete>
  bool process(Tracker& tracker, TrackerTraits& traits, ClusterSourceInput source,
               AfterLoad&& afterLoad, Complete&& complete)
  {
    const auto views = frame.getROFViews();
    if (views.overlap.mLayerCount > 0 && source.rofs.size() != views.overlap.getLayer(0).mNROFsTF) {
      LOGP(warn, "{} CA ROF count differs from continuous timing expectation: received {} expected {}",
           mDetectorName, source.rofs.size(), views.overlap.getLayer(0).mNROFsTF);
    }
    const auto origin = source.rofs.empty() ? o2::InteractionRecord{} : source.rofs.front().getBCData();
    if (!loadWithRecovery(tracker.getExecutionPolicy().DropTFUponFailure, [&] {
          if (!source.dictionary) {
            throw std::runtime_error{std::string(mDetectorName) + " CA tracker cluster dictionary is not available"};
          }
          if (views.overlap.mLayerCount <= 0) {
            throw std::runtime_error{std::string(mDetectorName) + " CA tracker received no adapter-owned runtime ROF timing view"};
          }
          const auto& clock = views.overlap.getLayer(0);
          validateSourceROFTiming(source, origin, clock);
          loadTimeFrameSources(frame, gsl::span<const ClusterSourceInput>{&source, 1},
                               frame.getDetectorConfiguration().getSurfaceCatalog(), &externalIndices, &clusterSizes);
          frame.setROFViews(views);
          for (uint16_t layer = 0; layer < source.layerToSurface.size(); ++layer) {
            frame.setROFViews(source.layerToSurface[layer].value(), views, layer);
          }
          afterLoad(origin);
        })) {
      return false;
    }
    if (!tracker.run(frame, traits)) {
      LOGP(warn, "{} CA tracking failed for this TF", mDetectorName);
      return false;
    }
    const auto& statistics = tracker.getRunStatistics();
    complete(statistics);
    LOGP(info, "{} CA tracking produced {} tracks in {:.2f} ms", mDetectorName, frame.getGenericTracks().size(), statistics.elapsedMs);
    return true;
  }

 private:
  bool matchesTiming(gsl::span<const o2::its::LayerTiming> timings) const noexcept
  {
    if (mTimingKey.size() != timings.size()) {
      return false;
    }
    for (std::size_t layer = 0; layer < timings.size(); ++layer) {
      const auto& cached = mTimingKey[layer];
      const auto& next = timings[layer];
      if (cached.mNROFsTF != next.mNROFsTF || cached.mROFLength != next.mROFLength ||
          cached.mROFDelay != next.mROFDelay || cached.mROFBias != next.mROFBias ||
          cached.mROFAddTimeErr != next.mROFAddTimeErr) {
        return false;
      }
    }
    return true;
  }

  const char* mDetectorName;
  std::vector<o2::its::LayerTiming> mTimingKey;
};
} // namespace o2::itsmft::tracking
#endif
