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
/// \file Cell.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_CACELL_H_
#define TRACKINGITSU_INCLUDE_CACELL_H_

#include "ITStracking/Constants.h"
#include "DataFormatsITS/TimeEstBC.h"
#include "ReconstructionDataFormats/Track.h"
#include "GPUCommonDef.h"

namespace o2::its
{

template <int NClusters>
class SeedBase : public o2::track::TrackParCovF
{
 public:
  GPUhd() int getInnerLayer() const { return getUserField(); }
  GPUhd() int getFirstTrackletIndex() const { return mTracklets[0]; };
  GPUhd() void setFirstTrackletIndex(int trkl) { mTracklets[0] = trkl; };
  GPUhd() int getSecondTrackletIndex() const { return mTracklets[1]; };
  GPUhd() void setSecondTrackletIndex(int trkl) { mTracklets[1] = trkl; };
  GPUhd() float getChi2() const { return mChi2; };
  GPUhd() void setChi2(float chi2) { mChi2 = chi2; };
  GPUhd() int getLevel() const { return mLevel; };
  GPUhd() void setLevel(int level) { mLevel = level; };
  GPUhd() int* getLevelPtr() { return &mLevel; }
  GPUhd() auto& getTimeStamp() noexcept { return mTime; }
  GPUhd() const auto& getTimeStamp() const noexcept { return mTime; }

 protected:
  GPUhdDefault() SeedBase() = default;
  GPUhdDefault() SeedBase(const SeedBase&) = default;
  GPUhdDefault() ~SeedBase() = default;
  GPUhdDefault() SeedBase(SeedBase&&) = default;
  GPUhdDefault() SeedBase& operator=(const SeedBase&) = default;
  GPUhdDefault() SeedBase& operator=(SeedBase&&) = default;
  GPUhd() SeedBase(const o2::track::TrackParCovF& tpc, float chi2, int level, const TimeEstBC& time)
    : o2::track::TrackParCovF(tpc), mChi2(chi2), mLevel(level), mTime(time)
  {
  }
  GPUhd() auto& clustersRaw() { return mClusters; }
  GPUhd() const auto& clustersRaw() const { return mClusters; }

 private:
  float mChi2{constants::UnsetValue};
  int mLevel{constants::UnusedIndex};
  std::array<int, 2> mTracklets = constants::helpers::initArray<int, 2, constants::UnusedIndex>();
  std::array<int, NClusters> mClusters = constants::helpers::initArray<int, NClusters, constants::UnusedIndex>();
  TimeEstBC mTime;
};

/// CellSeed: connections of three clusters
class CellSeed final : public SeedBase<constants::ClustersPerCell>
{
  using Base = SeedBase<constants::ClustersPerCell>;

 public:
  GPUhdDefault() CellSeed() = default;
  GPUhd() CellSeed(int innerL, int cl0, int cl1, int cl2, int trkl0, int trkl1, const o2::track::TrackParCovF& tpc, float chi2, const TimeEstBC& time)
    : Base(tpc, chi2, 1, time)
  {
    setUserField(innerL);
    auto& clusters = this->clustersRaw();
    clusters[0] = cl0;
    clusters[1] = cl1;
    clusters[2] = cl2;
    setFirstTrackletIndex(trkl0);
    setSecondTrackletIndex(trkl1);
  }
  GPUhdDefault() CellSeed(const CellSeed&) = default;
  GPUhdDefault() ~CellSeed() = default;
  GPUhdDefault() CellSeed(CellSeed&&) = default;
  GPUhdDefault() CellSeed& operator=(const CellSeed&) = default;
  GPUhdDefault() CellSeed& operator=(CellSeed&&) = default;

  GPUhd() int getFirstClusterIndex() const { return this->clustersRaw()[0]; };
  GPUhd() int getSecondClusterIndex() const { return this->clustersRaw()[1]; };
  GPUhd() int getThirdClusterIndex() const { return this->clustersRaw()[2]; };
  GPUhd() auto& getClusters() { return this->clustersRaw(); }
  GPUhd() const auto& getClusters() const { return this->clustersRaw(); }
  /// getCluster takes an ABSOLUTE layer index and returns UnusedIndex if the
  /// layer is outside the 3 stored slots (innerL, innerL+1, innerL+2).
  GPUhd() int getCluster(int layer) const
  {
    const int rel = layer - getInnerLayer();
    return (rel >= 0 && rel < constants::ClustersPerCell) ? this->clustersRaw()[rel] : constants::UnusedIndex;
  }
};

/// TrackSeed: full-width working representation used during road finding.
/// processNeighbours extends the cluster list inward, so we need NLayers
/// absolute-indexed slots here.
template <int NLayers>
class TrackSeed final : public SeedBase<NLayers>
{
  using Base = SeedBase<NLayers>;

 public:
  GPUhdDefault() TrackSeed() = default;
  GPUhd() TrackSeed(const CellSeed& cs)
    : Base(static_cast<const o2::track::TrackParCovF&>(cs), cs.getChi2(), cs.getLevel(), cs.getTimeStamp())
  {
    this->setUserField(cs.getInnerLayer());
    this->setFirstTrackletIndex(cs.getFirstTrackletIndex());
    this->setSecondTrackletIndex(cs.getSecondTrackletIndex());
    const int innerL = cs.getInnerLayer();
    auto& clusters = this->clustersRaw();
    clusters[innerL + 0] = cs.getFirstClusterIndex();
    clusters[innerL + 1] = cs.getSecondClusterIndex();
    clusters[innerL + 2] = cs.getThirdClusterIndex();
  }
  GPUhdDefault() TrackSeed(const TrackSeed&) = default;
  GPUhdDefault() ~TrackSeed() = default;
  GPUhdDefault() TrackSeed(TrackSeed&&) = default;
  GPUhdDefault() TrackSeed& operator=(const TrackSeed&) = default;
  GPUhdDefault() TrackSeed& operator=(TrackSeed&&) = default;

  /// Three-cluster view of the original cell — note: innerL (UserField) is not
  /// updated when processNeighbours extends the cluster list leftward.
  GPUhd() int getFirstClusterIndex() const { return this->clustersRaw()[this->getUserField()]; }
  GPUhd() int getSecondClusterIndex() const { return this->clustersRaw()[this->getUserField() + 1]; }
  GPUhd() int getThirdClusterIndex() const { return this->clustersRaw()[this->getUserField() + 2]; }
  GPUhd() auto& getClusters() { return this->clustersRaw(); }
  GPUhd() const auto& getClusters() const { return this->clustersRaw(); }
  GPUhd() int getCluster(int layer) const { return this->clustersRaw()[layer]; }
};

} // namespace o2::its

#endif /* TRACKINGITSU_INCLUDE_CACELL_H_ */
