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
/// \file TrackSeed.h
/// \brief GPU-portable whole-track seed for common CA tracking
///

#ifndef ALICEO2_ITSMFT_TRACKING_TRACKSEED_H_
#define ALICEO2_ITSMFT_TRACKING_TRACKSEED_H_

#include <array>

#include "DataFormatsITS/TimeEstBC.h"
#include "GPUCommonDef.h"
#include "ITSMFTTracking/Constants.h"
#include "ITSMFTTracking/IdTypes.h"
#include "ITSMFTTracking/LayerMask.h"
#include "ITSMFTTracking/SurfaceTrackState.h"
#include "ITSMFTTracking/Triplet.h"

namespace o2::itsmft::tracking
{

/// GPU-portable, non-templated whole-track seed with one cluster slot per
/// adopted-plan position. Fixed MaxLayoutSurfaces capacity is required for
/// device use, where heap allocation is unavailable.
///
/// This fixed-capacity value is the sole common-CA whole-track seed
/// representation.
class TrackSeed final
{
 public:
  static constexpr int MaxSurfaces = static_cast<int>(MaxLayoutSurfaces);

  GPUhdDefault() TrackSeed() = default;
  GPUhdDefault() TrackSeed(const TrackSeed&) = default;
  GPUhdDefault() ~TrackSeed() = default;
  GPUhdDefault() TrackSeed(TrackSeed&&) = default;
  GPUhdDefault() TrackSeed& operator=(const TrackSeed&) = default;
  GPUhdDefault() TrackSeed& operator=(TrackSeed&&) = default;

  // Triplet's hit mask is positional in the same fixed-capacity domain.
  GPUhd() TrackSeed(const Triplet& cs, const SurfaceTrackState& state, float chi2)
    : mState(state), mChi2(chi2), mLevel(cs.getLevel()), mTracklets{cs.getFirstTrackletIndex(), cs.getSecondTrackletIndex()}, mTime(cs.getTimeStamp())
  {
    const auto hitMask = cs.getHitLayerMask();
    int slot = 0;
    for (int position = 0; position < MaxSurfaces; ++position) {
      if (hitMask.has(position)) {
        mClusters[position] = cs.getClusters()[slot++];
        mHitLayerMask.set(position);
      }
    }
  }

  GPUhd() int getActiveLayerCount() const noexcept { return mHitLayerMask.count(); }
  GPUhd() int getInnerLayer() const noexcept { return mHitLayerMask.first(); }
  GPUhd() bool hasCluster(int position) const noexcept
  {
    return position >= 0 && position < MaxSurfaces && mHitLayerMask.has(position);
  }

  // Bounds-checked: an out-of-[0, MaxSurfaces) position safely
  // returns UnusedIndex instead of indexing out of bounds.
  GPUhd() int getCluster(int position) const noexcept
  {
    return (position >= 0 && position < MaxSurfaces) ? mClusters[position] : o2::its::constants::UnusedIndex;
  }

  GPUhd() LayerMask getHitLayerMask() const noexcept { return mHitLayerMask; }
  GPUhd() void setHitLayerMask(LayerMask mask) noexcept { mHitLayerMask = mask; }
  GPUhd() void setCluster(int position, int clusterIndex) noexcept
  {
    if (position >= 0 && position < MaxSurfaces) {
      mClusters[position] = clusterIndex;
    }
  }

  GPUhd() int getFirstClusterIndex() const noexcept { return getClusterBySlot(0); }
  GPUhd() int getSecondClusterIndex() const noexcept { return getClusterBySlot(1); }
  GPUhd() int getThirdClusterIndex() const noexcept { return getClusterBySlot(2); }

  GPUhd() auto& getClusters() noexcept { return mClusters; }
  GPUhd() const auto& getClusters() const noexcept { return mClusters; }

  GPUhd() int getFirstTrackletIndex() const noexcept { return mTracklets[0]; }
  GPUhd() void setFirstTrackletIndex(int trkl) noexcept { mTracklets[0] = trkl; }
  GPUhd() int getSecondTrackletIndex() const noexcept { return mTracklets[1]; }
  GPUhd() void setSecondTrackletIndex(int trkl) noexcept { mTracklets[1] = trkl; }

  GPUhd() float getChi2() const noexcept { return mChi2; }
  GPUhd() void setChi2(float chi2) noexcept { mChi2 = chi2; }
  GPUhd() int getLevel() const noexcept { return mLevel; }
  GPUhd() void setLevel(int level) noexcept { mLevel = level; }

  GPUhd() auto& getTimeStamp() noexcept { return mTime; }
  GPUhd() const auto& getTimeStamp() const noexcept { return mTime; }

  GPUhd() SurfaceTrackState& state() noexcept { return mState; }
  GPUhd() const SurfaceTrackState& state() const noexcept { return mState; }
  // Raw signed q/pT in slot 4 for cylinder and disk states; never squared.
  GPUhd() float getQOverPt() const noexcept { return mState.parameters[4]; }

 private:
  GPUhd() int getClusterBySlot(int requestedSlot) const noexcept
  {
    int slot = 0;
    for (int position = 0; position < MaxSurfaces; ++position) {
      if (hasCluster(position)) {
        if (slot++ == requestedSlot) {
          return mClusters[position];
        }
      }
    }
    return o2::its::constants::UnusedIndex;
  }

  SurfaceTrackState mState{};
  LayerMask mHitLayerMask{};
  float mChi2{o2::its::constants::UnsetValue};
  int mLevel{o2::its::constants::UnusedIndex};
  std::array<int, 2> mTracklets = o2::its::constants::helpers::initArray<int, 2, o2::its::constants::UnusedIndex>();
  std::array<int, MaxSurfaces> mClusters = o2::its::constants::helpers::initArray<int, MaxSurfaces, o2::its::constants::UnusedIndex>();
  o2::its::TimeEstBC mTime;
};

} // namespace o2::itsmft::tracking

#endif // ALICEO2_ITSMFT_TRACKING_TRACKSEED_H_
