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

#ifndef ALICEO2_ITSMFT_TRACKING_GENERICTRACK_H_
#define ALICEO2_ITSMFT_TRACKING_GENERICTRACK_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "GPUCommonDef.h"
#ifndef GPUCA_GPUCODE
#include "ITSMFTTracking/Cell.h"
#endif
#include "ITSMFTTracking/IdTypes.h"
#include "ITSMFTTracking/SurfaceTrackState.h"
#include "ITSMFTTracking/LayerMask.h"
#include "ITSMFTTracking/SurfaceTiming.h"

namespace o2::itsmft::tracking
{

// Stable TimeFrame identity. clusterId is the pre-sort position in the
// per-surface measurement arrays; publication adapters translate it to any
// external index space.
struct TrackClusterReference {
  LayerId layer{};
  uint16_t reserved{0};
  uint32_t clusterId{std::numeric_limits<uint32_t>::max()};

  GPUhdi() bool isValid() const noexcept { return layer.isValid() && clusterId != std::numeric_limits<uint32_t>::max(); }
};

static_assert(std::is_standard_layout_v<TrackClusterReference>);
static_assert(std::is_trivially_copyable_v<TrackClusterReference>);
static_assert(sizeof(TrackClusterReference) == 8);
static_assert(alignof(TrackClusterReference) == 4);
static_assert(offsetof(TrackClusterReference, layer) == 0);
static_assert(offsetof(TrackClusterReference, clusterId) == 4);

// Frame-owned result; [firstClusterRef, clusterRefEnd) is inner-to-outer and
// valid only with the same normalized event.
struct GenericTrack {
  SurfaceTrackState innerState{};
  SurfaceTrackState outerState{};
  float chi2{0.f};
  GenericTrackTimestamp timestamp{};
  LayerMask hitLayers{};
  uint32_t firstClusterRef{0};
  uint32_t clusterRefEnd{0};
};

#ifndef GPUCA_GPUCODE

// Successful refit result; typed output remains adapter-owned.
struct TrackingCandidate {
  TrackSeed seed;
  GenericTrack track{};
  float phi{0.f};
  float eta{0.f};
  double charge{0.};

  int getNumberOfClusters() const noexcept { return seed.getActiveLayerCount(); }
  int getClusterIndex(int position) const noexcept { return seed.getCluster(position); }
  int getFirstClusterLayer() const noexcept { return seed.getHitLayerMask().first(); }
};

#endif

// Device-facing layout requirements.
static_assert(std::is_standard_layout_v<GenericTrack>);
static_assert(std::is_trivially_copyable_v<GenericTrack>);
static_assert(sizeof(GenericTrack) == 224);
static_assert(alignof(GenericTrack) == alignof(GenericTrackTimestamp));

// The caller supplies the current frame-owned reference-array size; do not
// infer validity from the track itself.
GPUhdi() constexpr bool isValidTrackRange(const GenericTrack& track, uint32_t trackClusterIndicesSize) noexcept
{
  return track.firstClusterRef <= track.clusterRefEnd && track.clusterRefEnd <= trackClusterIndicesSize;
}

GPUhdi() constexpr uint32_t trackClusterRefCount(const GenericTrack& track) noexcept
{
  return track.clusterRefEnd - track.firstClusterRef;
}

} // namespace o2::itsmft::tracking

#endif /* ALICEO2_ITSMFT_TRACKING_GENERICTRACK_H_ */
