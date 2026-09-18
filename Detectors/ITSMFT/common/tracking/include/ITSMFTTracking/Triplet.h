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
/// \file Triplet.h
/// \brief CA geometric triplet types with hole-layer support
///

#ifndef ALICEO2_ITSMFT_TRACKING_TRIPLET_H_
#define ALICEO2_ITSMFT_TRACKING_TRIPLET_H_

#include <array>
#include <cstdint>

#include "DataFormatsITS/TimeEstBC.h"
#include "ITSMFTTracking/LayerMask.h"
#include "ITSMFTTracking/TripletFitting.h"
#include "ITSMFTTracking/Constants.h"
#include "GPUCommonDef.h"

namespace o2::itsmft::tracking
{

struct TripletNeighbour {
  int cellTopology{-1};
  int cell{-1};
  int nextCellTopology{-1};
  int nextCell{-1};
  int level{-1};
};

struct TripleClusterReference {
  int surfacePosition{o2::its::constants::UnusedIndex};
  int clusterIndex{o2::its::constants::UnusedIndex};
};

/// Common non-`SurfaceKind`-templated CA cell/geometric-triplet value.
/// A Triplet deliberately has no kinematic state or fit chi2; those first
/// exist after TrackerTraits materializes a TrackSeed.
class Triplet final
{
 public:
  GPUhdDefault() Triplet() = default;
  GPUhd() Triplet(int innerL, int cl0, int cl1, int cl2, int trkl0, int trkl1, const o2::its::TimeEstBC& time)
    : Triplet(LayerMask(innerL, innerL + 1, innerL + 2), cl0, cl1, cl2, trkl0, trkl1, time)
  {
  }
  GPUhd() Triplet(LayerMask hitLayerMask, int cl0, int cl1, int cl2, int trkl0, int trkl1, const o2::its::TimeEstBC& time)
    : mLevel(1), mTime(time)
  {
    setHitLayerMask(hitLayerMask);
    auto& clusters = mClusters;
    clusters[0] = cl0;
    clusters[1] = cl1;
    clusters[2] = cl2;
    setFirstTrackletIndex(trkl0);
    setSecondTrackletIndex(trkl1);
  }
  GPUhdDefault() Triplet(const Triplet&) = default;
  GPUhdDefault() ~Triplet() = default;
  GPUhdDefault() Triplet(Triplet&&) = default;
  GPUhdDefault() Triplet& operator=(const Triplet&) = default;
  GPUhdDefault() Triplet& operator=(Triplet&&) = default;

  GPUhd() LayerMask getHitLayerMask() const { return LayerMask{mHitLayerMask}; }
  GPUhd() void setHitLayerMask(LayerMask mask) { mHitLayerMask = mask.value(); }
  GPUhd() int getInnerLayer() const { return getHitLayerMask().first(); }
  GPUhd() int getFirstTrackletIndex() const { return mTracklets[0]; }
  GPUhd() void setFirstTrackletIndex(int trkl) { mTracklets[0] = trkl; }
  GPUhd() int getSecondTrackletIndex() const { return mTracklets[1]; }
  GPUhd() void setSecondTrackletIndex(int trkl) { mTracklets[1] = trkl; }
  GPUhd() int getLevel() const { return mLevel; }
  GPUhd() void setLevel(int level) { mLevel = level; }
  GPUhd() int* getLevelPtr() { return &mLevel; }
  GPUhd() auto& getTimeStamp() noexcept { return mTime; }
  GPUhd() const auto& getTimeStamp() const noexcept { return mTime; }
  GPUhd() int getFirstClusterIndex() const { return mClusters[0]; }
  GPUhd() int getSecondClusterIndex() const { return mClusters[1]; }
  GPUhd() int getThirdClusterIndex() const { return mClusters[2]; }
  GPUhd() auto& getClusters() { return mClusters; }
  GPUhd() const auto& getClusters() const { return mClusters; }
  GPUhd() TripletFitFactor& tripletFactor() noexcept { return mTripletFactor; }
  GPUhd() const TripletFitFactor& tripletFactor() const noexcept { return mTripletFactor; }
  GPUhd() TripleClusterReference getClusterReference(int requestedSlot) const noexcept
  {
    if (requestedSlot < 0 || requestedSlot >= o2::its::constants::ClustersPerCell) {
      return {};
    }
    const auto mask = getHitLayerMask();
    int slot = 0;
    for (int position = 0; position < 32; ++position) {
      if (mask.has(position) && slot++ == requestedSlot) {
        return {position, mClusters[requestedSlot]};
      }
    }
    return {};
  }
  GPUhd() int getCluster(int layer) const
  {
    const int slot = getHitLayerMask().slot(layer);
    return (slot >= 0 && slot < o2::its::constants::ClustersPerCell) ? mClusters[slot] : o2::its::constants::UnusedIndex;
  }

 private:
  uint32_t mHitLayerMask{0};
  int mLevel{o2::its::constants::UnusedIndex};
  std::array<int, 2> mTracklets = o2::its::constants::helpers::initArray<int, 2, o2::its::constants::UnusedIndex>();
  std::array<int, o2::its::constants::ClustersPerCell> mClusters =
    o2::its::constants::helpers::initArray<int, o2::its::constants::ClustersPerCell, o2::its::constants::UnusedIndex>();
  o2::its::TimeEstBC mTime;
  TripletFitFactor mTripletFactor{};
};

} // namespace o2::itsmft::tracking

#endif /* ALICEO2_ITSMFT_TRACKING_TRIPLET_H_ */
