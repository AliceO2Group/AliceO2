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
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

#include <gsl/span>

#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTTracking/TimeFrame.h"
#include "ITSMFTTracking/ROFLookupTables.h"

namespace o2::itsmft::tracking
{

#ifndef GPUCA_GPUCODE

// Tracks already carry a symmetric timestamp. Apply the publication clock's
// uncertainty limit without modifying the frame-owned track.
inline o2::its::TimeStamp makeOutputTimestamp(o2::its::TimeStamp timestamp, const o2::its::LayerTiming& clock) noexcept
{
  timestamp.setTimeStampError(std::min(timestamp.getTimeStampError(), clock.mROFLength * 0.5f));
  return timestamp;
}

// This context is intentionally source-local.  ROFRecord payload is copied
// only into the returned publication product, never into TimeFrame.
struct TrackPublicationTimingContext {
  gsl::span<const o2::itsmft::ROFRecord> inputROFs;
  o2::its::LayerTiming clock;
};

inline std::optional<std::vector<uint32_t>> selectGenericTracksForSurfaces(
  const TimeFrame& frame,
  gsl::span<const LayerId> sourceSurfaces)
{
  const auto& tracks = frame.getGenericTracks();
  if (tracks.size() > std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }
  std::vector<uint32_t> selection;
  const auto& references = frame.getTrackClusterIndices();
  selection.reserve(tracks.size());
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
      selection.push_back(globalIndex);
    }
  }
  return selection;
}

inline std::optional<std::vector<uint32_t>> makeLegacyOutputOrder(
  const TimeFrame& frame, std::vector<uint32_t> selection,
  const o2::its::LayerTiming& clock)
{
  const auto& tracks = frame.getGenericTracks();
  for (const auto index : selection) {
    const auto& timestamp = tracks[index].timestamp;
    if (!std::isfinite(timestamp.getTimeStamp()) || !std::isfinite(timestamp.getTimeStampError()) ||
        timestamp.getTimeStampError() <= 0.f) {
      return std::nullopt;
    }
  }
  // Sort only indices, using the same clamped timestamp that will be published.
  // Match Tracker::sortTracks(): lower timestamp edge, then chi2.
  std::sort(selection.begin(), selection.end(), [&](uint32_t left, uint32_t right) {
    const auto& leftTrack = tracks[left];
    const auto& rightTrack = tracks[right];
    const auto leftTime = makeOutputTimestamp(leftTrack.timestamp, clock);
    const auto rightTime = makeOutputTimestamp(rightTrack.timestamp, clock);
    const auto leftLower = leftTime.getTimeStamp() - leftTime.getTimeStampError();
    const auto rightLower = rightTime.getTimeStamp() - rightTime.getTimeStampError();
    if (leftLower != rightLower) {
      return leftLower < rightLower;
    }
    return leftTrack.chi2 < rightTrack.chi2;
  });
  return selection;
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
