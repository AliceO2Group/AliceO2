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

#ifndef ALICEO2_ITSMFT_TRACKING_TRACKPUBLICATIONHELPERS_H_
#define ALICEO2_ITSMFT_TRACKING_TRACKPUBLICATIONHELPERS_H_

// Shared host-side track selection, ordering and ROF assignment for publication.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

#include <gsl/span>

#include "DataFormatsITSMFT/ROFRecord.h"
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

struct TrackPublicationSelection {
  std::vector<uint32_t> globalIndices;
};

struct TrackPublicationOrderEntry {
  uint32_t globalIndex{};
  o2::its::TimeStamp timestamp{};
};

// This context is intentionally source-local.  ROFRecord payload is copied
// only into the returned publication product, never into TimeFrame.
struct TrackPublicationTimingContext {
  gsl::span<const o2::itsmft::ROFRecord> inputROFs;
  ClockTimingPublicationView clock;
};

inline std::optional<TrackPublicationSelection> selectGenericTracksForSurfaces(
  const TimeFrame& frame,
  gsl::span<const LayerId> sourceSurfaces)
{
  const auto& tracks = frame.getGenericTracks();
  if (tracks.size() > std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }
  TrackPublicationSelection selection;
  const auto& references = frame.getTrackClusterIndices();
  selection.globalIndices.reserve(tracks.size());
  for (uint32_t globalIndex = 0; globalIndex < tracks.size(); ++globalIndex) {
    const auto& track = tracks[globalIndex];
    if (!isValidTrackRange(track, static_cast<uint32_t>(references.size()))) {
      return std::nullopt;
    }
    bool requested = false;
    bool foreign = false;
    for (uint32_t i = track.firstClusterRef; i < track.clusterRefEnd; ++i) {
      const auto& reference = references[i];
      if (!reference.isValid()) {
        return std::nullopt;
      }
      const bool match = std::find(sourceSurfaces.begin(), sourceSurfaces.end(), reference.layer) != sourceSurfaces.end();
      requested |= match;
      foreign |= !match;
    }
    if (requested && foreign) {
      return std::nullopt;
    }
    if (requested) {
      selection.globalIndices.push_back(globalIndex);
    }
  }
  return selection;
}

inline std::optional<std::vector<TrackPublicationOrderEntry>> makeLegacyOutputOrder(
  const TimeFrame& frame, const TrackPublicationSelection& selection,
  const ClockTimingPublicationView& clock)
{
  std::vector<TrackPublicationOrderEntry> ordered;
  ordered.reserve(selection.globalIndices.size());
  for (const auto index : selection.globalIndices) {
    const auto timestamp = clock.makeOutputTimestamp(frame.getGenericTracks()[index].timestamp);
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
                         const TrackPublicationTimingContext& context)
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

} // namespace o2::itsmft::tracking

#endif // !GPUCA_GPUCODE

#endif // ALICEO2_ITSMFT_TRACKING_TRACKPUBLICATIONHELPERS_H_
