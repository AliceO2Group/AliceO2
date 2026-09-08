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

#ifndef ALICEO2_ITSMFT_TRACKING_GENERICTRACKOUTPUTADAPTER_H_
#define ALICEO2_ITSMFT_TRACKING_GENERICTRACKOUTPUTADAPTER_H_

// Pure host-side boundary for DPL adapters. It consumes immutable owner data
// plus workflow-owned ROF context and returns fully staged vectors.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

#include <gsl/span>

#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "ITSMFTTracking/detail/ITSSharedClusterCompatibility.h"
#include "ITSMFTTracking/detail/SurfaceTrackStateLegacyAdapters.h"
#include "ITSMFTTracking/SurfaceTiming.h"
#include "ITSMFTTracking/TimeFrame.h"
#include "ITSMFTTracking/ROFLookupTables.h"

namespace o2::itsmft::tracking
{

#ifndef GPUCA_GPUCODE

// Host-only immutable output view around the established clock-layer
// implementation. Symmetry, clamping, and ROF lookup stay in LayerTiming.
class ClockTimingPublicationView
{
 public:
  explicit ClockTimingPublicationView(const o2::its::LayerTiming& clock) : mClock{clock} {}

  std::optional<o2::its::TimeEstBC> makeTimeEstBC(const GenericTrackTimestamp& timestamp) const noexcept
  {
    if (!timestamp.isValid() || timestamp.begin < 0 || timestamp.end < 0 ||
        timestamp.begin > std::numeric_limits<uint32_t>::max() || timestamp.end > std::numeric_limits<uint32_t>::max()) {
      return std::nullopt;
    }
    const auto width = static_cast<uint64_t>(timestamp.end) - static_cast<uint64_t>(timestamp.begin);
    if (width > std::numeric_limits<uint16_t>::max()) {
      return std::nullopt;
    }
    return o2::its::TimeEstBC{static_cast<uint32_t>(timestamp.begin), static_cast<uint16_t>(width)};
  }

  std::optional<o2::its::TimeStamp> makeOutputTimestamp(const GenericTrackTimestamp& timestamp) const noexcept
  {
    const auto asymmetric = makeTimeEstBC(timestamp);
    if (!asymmetric) {
      return std::nullopt;
    }
    auto symmetric = asymmetric->makeSymmetrical();
    const float clamp = mClock.mROFLength * 0.5f;
    if (symmetric.getTimeStampError() > clamp) {
      symmetric.setTimeStampError(clamp);
    }
    return symmetric;
  }

  int getROF(const o2::its::TimeStamp& timestamp) const noexcept { return mClock.getROF(timestamp); }
  uint32_t getROFCount() const noexcept { return mClock.mNROFsTF; }
  const o2::its::LayerTiming& getLegacyClockLayer() const noexcept { return mClock; }

 private:
  o2::its::LayerTiming mClock;
};

#endif // !GPUCA_GPUCODE

enum class GenericTrackOutputAdapterError : uint8_t {
  None,
  TooManyGenericTracks,
  InvalidTrackRange,
  UnresolvedReference,
  MixedDetector,
  MixedSources,
  InvalidExternalClusterIndex,
  InvalidLayerLayout,
  InvalidTimestamp,
  InvalidROF,
  InvalidState,
  MissingCompatibility,
  MissingMCLabels
};

struct GenericTrackOutputAdapterSelection {
  std::vector<uint32_t> globalIndices;
};

struct GenericTrackOutputOrderEntry {
  uint32_t globalIndex{};
  o2::its::TimeStamp timestamp{};
};

// This context is intentionally source-local.  ROFRecord payload is copied
// only into the returned publication product, never into TimeFrame.
struct GenericTrackOutputTimingContext {
  gsl::span<const o2::itsmft::ROFRecord> inputROFs;
  ClockTimingPublicationView clock;
};

struct GenericTrackPublicationContext {
  o2::detectors::DetID::ID detector{};
  ClusterSourceId source{}; // Publication provenance; selection uses the bound layer mapping.
  gsl::span<const o2::itsmft::ROFRecord> inputROFs;
  ClockTimingPublicationView clock;
  gsl::span<const LayerId> layerMapping;
  const std::vector<std::vector<uint32_t>>* externalIndicesBySurface{nullptr};
  const std::vector<std::vector<uint32_t>>* clusterSizesBySurface{nullptr};
};

struct ITSGenericTrackOutput {
  std::vector<o2::its::TrackITS> tracks;
  std::vector<int> clusterIndices;
  std::vector<o2::itsmft::ROFRecord> trackROFs;
  std::vector<o2::MCCompLabel> labels;
};

struct MFTGenericTrackOutput {
  std::vector<o2::mft::TrackMFT> tracks;
  std::vector<int> clusterIndices;
  std::vector<o2::itsmft::ROFRecord> trackROFs;
  std::vector<uint16_t> seedPatterns;
  std::vector<o2::MCCompLabel> labels;
};

inline std::optional<GenericTrackOutputAdapterSelection> selectGenericTracksForSurfaces(
  const TimeFrame& frame,
  gsl::span<const LayerId> sourceSurfaces,
  GenericTrackOutputAdapterError& error)
{
  error = GenericTrackOutputAdapterError::None;
  const auto& tracks = frame.getGenericTracks();
  if (tracks.size() > std::numeric_limits<uint32_t>::max()) {
    error = GenericTrackOutputAdapterError::TooManyGenericTracks;
    return std::nullopt;
  }
  GenericTrackOutputAdapterSelection selection;
  const auto& references = frame.getTrackClusterIndices();
  selection.globalIndices.reserve(tracks.size());
  for (uint32_t globalIndex = 0; globalIndex < tracks.size(); ++globalIndex) {
    const auto& track = tracks[globalIndex];
    if (!isValidTrackRange(track, static_cast<uint32_t>(references.size()))) {
      error = GenericTrackOutputAdapterError::InvalidTrackRange;
      return std::nullopt;
    }
    bool requested = false;
    bool foreign = false;
    for (uint32_t i = track.firstClusterRef; i < track.clusterRefEnd; ++i) {
      const auto& reference = references[i];
      if (!reference.isValid()) {
        error = GenericTrackOutputAdapterError::UnresolvedReference;
        return std::nullopt;
      }
      const bool match = std::find(sourceSurfaces.begin(), sourceSurfaces.end(), reference.layer) != sourceSurfaces.end();
      requested |= match;
      foreign |= !match;
    }
    if (requested && foreign) {
      error = GenericTrackOutputAdapterError::MixedDetector;
      return std::nullopt;
    }
    if (requested) {
      selection.globalIndices.push_back(globalIndex);
    }
  }
  return selection;
}

inline std::optional<o2::its::TimeStamp> makeOutputTimestamp(const GenericTrackTimestamp& timestamp,
                                                             const ClockTimingPublicationView& clock,
                                                             GenericTrackOutputAdapterError& error)
{
  const auto result = clock.makeOutputTimestamp(timestamp);
  if (!result) {
    error = GenericTrackOutputAdapterError::InvalidTimestamp;
    return std::nullopt;
  }
  return result;
}

inline std::optional<std::vector<GenericTrackOutputOrderEntry>> makeLegacyOutputOrder(
  const TimeFrame& frame, const GenericTrackOutputAdapterSelection& selection,
  const ClockTimingPublicationView& clock, GenericTrackOutputAdapterError& error)
{
  std::vector<GenericTrackOutputOrderEntry> ordered;
  ordered.reserve(selection.globalIndices.size());
  for (const auto index : selection.globalIndices) {
    const auto timestamp = makeOutputTimestamp(frame.getGenericTracks()[index].timestamp, clock, error);
    if (!timestamp) {
      return std::nullopt;
    }
    ordered.push_back({index, *timestamp});
  }
  // Match Tracker::sortTracks(): lower timestamp edge, then chi2.
  std::sort(ordered.begin(), ordered.end(), [&frame](const auto& left, const auto& right) {
    const auto& leftTrack = frame.getGenericTracks()[left.globalIndex];
    const auto& rightTrack = frame.getGenericTracks()[right.globalIndex];
    const auto leftLower = left.timestamp.getTimeStamp() - left.timestamp.getTimeStampError();
    const auto rightLower = right.timestamp.getTimeStamp() - right.timestamp.getTimeStampError();
    if (leftLower != rightLower) {
      return leftLower < rightLower;
    }
    return leftTrack.chi2 < rightTrack.chi2;
  });
  return ordered;
}

inline void finalizeROFs(std::vector<o2::itsmft::ROFRecord>& rofs, const std::vector<o2::its::TimeStamp>& times,
                         const GenericTrackOutputTimingContext& context)
{
  for (auto& rof : rofs) {
    rof.setFirstEntry(0);
    rof.setNEntries(0);
  }
  for (const auto& time : times) {
    const int rof = context.clock.getROF(time);
    if (rof < 0 || static_cast<size_t>(rof) >= rofs.size()) {
      // Keep the track; omit only its TrackROF entry.
      continue;
    }
    rofs[rof].setNEntries(rofs[rof].getNEntries() + 1);
  }
  std::vector<int> counts(rofs.size());
  for (size_t i = 0; i < rofs.size(); ++i) {
    counts[i] = rofs[i].getNEntries();
  }
  std::exclusive_scan(counts.begin(), counts.end(), counts.begin(), 0);
  for (size_t i = 0; i < rofs.size(); ++i) {
    rofs[i].setFirstEntry(counts[i]);
  }
}

inline void setOutputClusterRange(o2::its::TrackITS& track, int first, int count)
{
  track.setClusterRefs(first, count);
}

inline void setOutputClusterRange(o2::mft::TrackMFT& track, int first, int count)
{
  track.setExternalClusterIndexOffset(first);
  track.setNumberOfPoints(count);
}

template <typename OutputTrack>
inline bool collectReferences(const TimeFrame& frame, const GenericTrack& common, gsl::span<const LayerId> layerMapping,
                              uint32_t maxLayers, std::vector<int>& outputIndices, OutputTrack& output,
                              uint32_t& pattern, GenericTrackOutputAdapterError& error,
                              const std::vector<std::vector<uint32_t>>* externalIndicesBySurface,
                              const std::vector<std::vector<uint32_t>>* clusterSizesBySurface)
{
  const auto& references = frame.getTrackClusterIndices();
  std::vector<const TrackClusterReference*> byLayer(maxLayers, nullptr);
  for (uint32_t ref = common.firstClusterRef; ref < common.clusterRefEnd; ++ref) {
    const auto& key = references[ref];
    if (!key.isValid()) {
      error = GenericTrackOutputAdapterError::UnresolvedReference;
      return false;
    }
    const auto where = std::find(layerMapping.begin(), layerMapping.end(), key.layer);
    if (where == layerMapping.end() || static_cast<uint32_t>(where - layerMapping.begin()) >= maxLayers) {
      error = GenericTrackOutputAdapterError::InvalidLayerLayout;
      return false;
    }
    const auto layer = static_cast<uint32_t>(where - layerMapping.begin());
    if (byLayer[layer] != nullptr) {
      error = GenericTrackOutputAdapterError::InvalidLayerLayout;
      return false;
    }
    byLayer[layer] = &key;
  }
  const int first = static_cast<int>(outputIndices.size());
  uint32_t count = 0;
  for (uint32_t layer = maxLayers; layer-- > 0;) {
    const auto* reference = byLayer[layer];
    if (reference == nullptr) {
      continue;
    }
    uint32_t externalIndex = reference->clusterId;
    if (externalIndicesBySurface != nullptr) {
      if (reference->layer.value() >= externalIndicesBySurface->size() ||
          reference->clusterId >= (*externalIndicesBySurface)[reference->layer.value()].size()) {
        error = GenericTrackOutputAdapterError::InvalidExternalClusterIndex;
        return false;
      }
      externalIndex = (*externalIndicesBySurface)[reference->layer.value()][reference->clusterId];
    }
    if (externalIndex > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
      error = GenericTrackOutputAdapterError::InvalidExternalClusterIndex;
      return false;
    }
    if (clusterSizesBySurface == nullptr ||
        reference->layer.value() >= clusterSizesBySurface->size() ||
        reference->clusterId >= (*clusterSizesBySurface)[reference->layer.value()].size()) {
      error = GenericTrackOutputAdapterError::UnresolvedReference;
      return false;
    }
    outputIndices.push_back(static_cast<int>(externalIndex));
    output.setClusterSize(layer, (*clusterSizesBySurface)[reference->layer.value()][reference->clusterId]);
    pattern |= 1u << layer;
    ++count;
  }
  setOutputClusterRange(output, first, static_cast<int>(count));
  return true;
}

inline std::optional<ITSGenericTrackOutput> stageITSGenericTrackOutput(const TimeFrame& frame,
                                                                       gsl::span<const LayerId> surfaces,
                                                                       const GenericTrackOutputTimingContext& context,
                                                                       const ITSSharedClusterCompatibility& compatibility,
                                                                       bool withMC, GenericTrackOutputAdapterError& error,
                                                                       const std::vector<std::vector<uint32_t>>* externalIndicesBySurface = nullptr,
                                                                       const std::vector<std::vector<uint32_t>>* clusterSizesBySurface = nullptr)
{
  const auto selection = selectGenericTracksForSurfaces(frame, surfaces, error);
  if (!selection || (!selection->globalIndices.empty() && !compatibility.isSealed())) {
    if (error == GenericTrackOutputAdapterError::None)
      error = GenericTrackOutputAdapterError::MissingCompatibility;
    return std::nullopt;
  }
  if (withMC && frame.getTrackLabels().size() != frame.getGenericTracks().size()) {
    error = GenericTrackOutputAdapterError::MissingMCLabels;
    return std::nullopt;
  }
  const auto ordered = makeLegacyOutputOrder(frame, *selection, context.clock, error);
  if (!ordered) {
    return std::nullopt;
  }
  ITSGenericTrackOutput staged;
  staged.trackROFs.assign(context.inputROFs.begin(), context.inputROFs.end());
  staged.tracks.reserve(ordered->size());
  staged.labels.reserve(withMC ? ordered->size() : 0);
  std::vector<o2::its::TimeStamp> times;
  times.reserve(ordered->size());
  for (const auto& orderedTrack : *ordered) {
    const auto index = orderedTrack.globalIndex;
    o2::track::TrackParCovF inner, outer;
    const auto& common = frame.getGenericTracks()[index];
    if (!legacy::exportBarrelTrackParCov(common.innerState, inner) || !legacy::exportBarrelTrackParCov(common.outerState, outer)) {
      error = GenericTrackOutputAdapterError::InvalidState;
      return std::nullopt;
    }
    const auto it = std::lower_bound(compatibility.entries().begin(), compatibility.entries().end(), index,
                                     [](const auto& entry, uint32_t value) { return entry.genericTrackIndex < value; });
    if (it == compatibility.entries().end() || it->genericTrackIndex != index) {
      error = GenericTrackOutputAdapterError::MissingCompatibility;
      return std::nullopt;
    }
    o2::its::TrackITS output{inner, common.chi2, outer};
    uint32_t pattern = 0;
    if (!collectReferences(frame, common, surfaces, 7, staged.clusterIndices, output, pattern, error,
                           externalIndicesBySurface, clusterSizesBySurface))
      return std::nullopt;
    output.setPattern(pattern);
    output.setSharedClusters(it->hasSharedClusters);
    output.getTimeStamp() = orderedTrack.timestamp;
    staged.tracks.push_back(std::move(output));
    times.push_back(orderedTrack.timestamp);
    if (withMC)
      staged.labels.push_back(frame.getTrackLabels()[index]);
  }
  finalizeROFs(staged.trackROFs, times, context);
  return staged;
}

inline std::optional<MFTGenericTrackOutput> stageMFTGenericTrackOutput(const TimeFrame& frame,
                                                                       gsl::span<const LayerId> surfaces,
                                                                       const GenericTrackOutputTimingContext& context,
                                                                       bool withMC, GenericTrackOutputAdapterError& error,
                                                                       const std::vector<std::vector<uint32_t>>* externalIndicesBySurface = nullptr,
                                                                       const std::vector<std::vector<uint32_t>>* clusterSizesBySurface = nullptr)
{
  const auto selection = selectGenericTracksForSurfaces(frame, surfaces, error);
  if (!selection)
    return std::nullopt;
  if (withMC && frame.getTrackLabels().size() != frame.getGenericTracks().size()) {
    error = GenericTrackOutputAdapterError::MissingMCLabels;
    return std::nullopt;
  }
  const auto ordered = makeLegacyOutputOrder(frame, *selection, context.clock, error);
  if (!ordered) {
    return std::nullopt;
  }
  MFTGenericTrackOutput staged;
  staged.trackROFs.assign(context.inputROFs.begin(), context.inputROFs.end());
  staged.tracks.reserve(ordered->size());
  staged.seedPatterns.reserve(ordered->size());
  std::vector<o2::its::TimeStamp> times;
  times.reserve(ordered->size());
  for (const auto& orderedTrack : *ordered) {
    const auto index = orderedTrack.globalIndex;
    const auto& common = frame.getGenericTracks()[index];
    o2::track::TrackParCovFwd inner, outer;
    if (!legacy::exportLegacyForwardTrackParCov(common.innerState, inner) || !legacy::exportLegacyForwardTrackParCov(common.outerState, outer)) {
      error = GenericTrackOutputAdapterError::InvalidState;
      return std::nullopt;
    }
    // Preserve the legacy TrackMFT object shape without claiming a seed-pT
    // estimate from this tracker. TrackMFT does not initialize mInvQPtSeed.
    outer.setTrackChi2(0.f);
    o2::mft::TrackMFT output;
    static_cast<o2::track::TrackParCovFwd&>(output) = inner;
    output.setOutParam(outer);
    output.setTrackChi2(common.chi2);
    output.setCA(true);
    output.setInvQPtSeed(0.);
    output.setChi2QPtSeed(0.);
    uint32_t pattern = 0;
    if (!collectReferences(frame, common, surfaces, 10, staged.clusterIndices, output, pattern, error,
                           externalIndicesBySurface, clusterSizesBySurface))
      return std::nullopt;
    staged.tracks.push_back(std::move(output));
    staged.seedPatterns.push_back(static_cast<uint16_t>(pattern));
    times.push_back(orderedTrack.timestamp);
    if (withMC)
      staged.labels.push_back(frame.getTrackLabels()[index]);
  }
  finalizeROFs(staged.trackROFs, times, context);
  return staged;
}

inline std::optional<ITSGenericTrackOutput> stageITSGenericTrackOutput(const TimeFrame& frame, const GenericTrackPublicationContext& context,
                                                                       const ITSSharedClusterCompatibility& compatibility, bool withMC,
                                                                       GenericTrackOutputAdapterError& error)
{
  if (context.detector != o2::detectors::DetID::ITS) {
    error = GenericTrackOutputAdapterError::MixedDetector;
    return std::nullopt;
  }
  return stageITSGenericTrackOutput(frame, context.layerMapping, {context.inputROFs, context.clock}, compatibility, withMC, error,
                                    context.externalIndicesBySurface, context.clusterSizesBySurface);
}

inline std::optional<MFTGenericTrackOutput> stageMFTGenericTrackOutput(const TimeFrame& frame, const GenericTrackPublicationContext& context,
                                                                       bool withMC, GenericTrackOutputAdapterError& error)
{
  if (context.detector != o2::detectors::DetID::MFT) {
    error = GenericTrackOutputAdapterError::MixedDetector;
    return std::nullopt;
  }
  return stageMFTGenericTrackOutput(frame, context.layerMapping, {context.inputROFs, context.clock}, withMC, error,
                                    context.externalIndicesBySurface, context.clusterSizesBySurface);
}

} // namespace o2::itsmft::tracking

#endif // ALICEO2_ITSMFT_TRACKING_GENERICTRACKOUTPUTADAPTER_H_
