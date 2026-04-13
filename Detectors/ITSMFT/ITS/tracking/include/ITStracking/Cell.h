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

template <int NLayers>
class CellSeed final : public o2::track::TrackParCovF
{
 public:
  GPUhdDefault() CellSeed() = default;
  GPUhd() CellSeed(int innerL, int cl0, int cl1, int cl2, int trkl0, int trkl1, o2::track::TrackParCovF& tpc, float chi2, const TimeEstBC& time) : o2::track::TrackParCovF(tpc), mChi2(chi2), mLevel(1), mTime(time)
  {
    mClusters.fill(constants::UnusedIndex);
    setUserField(innerL);
    mClusters[innerL + 0] = cl0;
    mClusters[innerL + 1] = cl1;
    mClusters[innerL + 2] = cl2;
    mTracklets[0] = trkl0;
    mTracklets[1] = trkl1;
  }
  GPUhdDefault() CellSeed(const CellSeed&) = default;
  GPUhdDefault() ~CellSeed() = default;
  // GPUhdDefault() CellSeed(CellSeed&&) = default; TODO cannot use this yet since TrackPar only has device
  GPUhdDefault() CellSeed& operator=(const CellSeed&) = default;
  GPUhdDefault() CellSeed& operator=(CellSeed&&) = default;

  GPUhd() int getFirstClusterIndex() const { return mClusters[getUserField()]; };
  GPUhd() int getSecondClusterIndex() const { return mClusters[getUserField() + 1]; };
  GPUhd() int getThirdClusterIndex() const { return mClusters[getUserField() + 2]; };
  GPUhd() int getFirstTrackletIndex() const { return mTracklets[0]; };
  GPUhd() void setFirstTrackletIndex(int trkl) { mTracklets[0] = trkl; };
  GPUhd() int getSecondTrackletIndex() const { return mTracklets[1]; };
  GPUhd() void setSecondTrackletIndex(int trkl) { mTracklets[1] = trkl; };
  GPUhd() float getChi2() const { return mChi2; };
  GPUhd() void setChi2(float chi2) { mChi2 = chi2; };
  GPUhd() int getLevel() const { return mLevel; };
  GPUhd() void setLevel(int level) { mLevel = level; };
  GPUhd() int* getLevelPtr() { return &mLevel; }
  GPUhd() auto& getClusters() { return mClusters; }
  GPUhd() int getCluster(int i) const { return mClusters[i]; }
  GPUhd() void printCell() const
  {
    printf("cell: %d, %d\t lvl: %d\t chi2: %f\tcls: [", mTracklets[0], mTracklets[1], mLevel, mChi2);
    for (int i = 0; i < NLayers; ++i) {
      printf("%d", mClusters[i]);
      if (i < NLayers - 1) {
        printf(" | ");
      }
    }
    printf("]");
    printf(" ts: %u +/- %u\n", mTime.getTimeStamp(), mTime.getTimeStampError());
  }
  GPUhd() auto& getTimeStamp() noexcept { return mTime; }
  GPUhd() const auto& getTimeStamp() const noexcept { return mTime; }

 private:
  float mChi2 = -999.f;
  int mLevel = constants::UnusedIndex;
  std::array<int, 2> mTracklets = constants::helpers::initArray<int, 2, constants::UnusedIndex>();
  std::array<int, NLayers> mClusters = constants::helpers::initArray<int, NLayers, constants::UnusedIndex>();
  TimeEstBC mTime;
};

} // namespace o2::its

#endif /* TRACKINGITSU_INCLUDE_CACELL_H_ */
