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

#ifndef GPUCA_GPUCODE_DEVICE
#include <array>
#include <vector>
#endif

#include "GPUCommonDef.h"

namespace o2
{
namespace its
{

class Cell final
{
 public:
  GPUhd() Cell();
  GPUd() Cell(const int, const int, const int, const int, const int);

  GPUhd() int getFirstClusterIndex() const { return mFirstClusterIndex; };
  GPUhd() int getSecondClusterIndex() const { return mSecondClusterIndex; };
  GPUhd() int getThirdClusterIndex() const { return mThirdClusterIndex; };
  GPUhd() int getFirstTrackletIndex() const { return mFirstTrackletIndex; };
  GPUhd() int getSecondTrackletIndex() const { return mSecondTrackletIndex; };
  GPUhd() int getLevel() const { return mLevel; };
  GPUhd() void setLevel(const int level) { mLevel = level; };
  GPUhd() int* getLevelPtr() { return &mLevel; }

 private:
  const int mFirstClusterIndex;
  const int mSecondClusterIndex;
  const int mThirdClusterIndex;
  const int mFirstTrackletIndex;
  const int mSecondTrackletIndex;
  int mLevel;
};

GPUhdi() Cell::Cell()
  : mFirstClusterIndex{0},
    mSecondClusterIndex{0},
    mThirdClusterIndex{0},
    mFirstTrackletIndex{0},
    mSecondTrackletIndex{0},
    mLevel{0}
{
  // Nothing to do
}

GPUdi() Cell::Cell(const int firstClusterIndex, const int secondClusterIndex, const int thirdClusterIndex,
                   const int firstTrackletIndex, const int secondTrackletIndex)
  : mFirstClusterIndex{firstClusterIndex},
    mSecondClusterIndex{secondClusterIndex},
    mThirdClusterIndex{thirdClusterIndex},
    mFirstTrackletIndex{firstTrackletIndex},
    mSecondTrackletIndex{secondTrackletIndex},
    mLevel{1}
{
  // Nothing to do
}

class CellSeed final : public o2::track::TrackParCovF
{
 public:
  GPUhdDefault() CellSeed() = default;
  GPUhdDefault() CellSeed(const CellSeed&) = default;
  GPUd() CellSeed(int innerL, int cl0, int cl1, int cl2, int trkl0, int trkl1, o2::track::TrackParCovF& tpc, float chi2) : o2::track::TrackParCovF{tpc}, mChi2{chi2}, mLevel{1}
  {
    setUserField(innerL);
    mClusters[innerL + 0] = cl0;
    mClusters[innerL + 1] = cl1;
    mClusters[innerL + 2] = cl2;
    mTracklets[0] = trkl0;
    mTracklets[1] = trkl1;
  }
  GPUhd() int getFirstClusterIndex() const { return mClusters[getUserField()]; };
  GPUhd() int getSecondClusterIndex() const { return mClusters[getUserField() + 1]; };
  GPUhd() int getThirdClusterIndex() const { return mClusters[getUserField() + 2]; };
  GPUhd() int getFirstTrackletIndex() const { return mTracklets[0]; };
  GPUhd() void setFirstTrackletIndex(int trkl) { mTracklets[0] = trkl; };
  GPUhd() int getSecondTrackletIndex() const { return mTracklets[1]; };
  GPUhd() void setSecondTrackletIndex(int trkl) { mTracklets[1] = trkl; };
  GPUhd() int getChi2() const { return mChi2; };
  GPUhd() void setChi2(float chi2) { mChi2 = chi2; };
  GPUhd() int getLevel() const { return mLevel; };
  GPUhd() void setLevel(int level) { mLevel = level; };
  GPUhd() int* getLevelPtr() { return &mLevel; }
  GPUhd() int* getClusters() { return mClusters; }
  GPUhd() int getCluster(int i) const { return mClusters[i]; }

 private:
  int mClusters[7] = {-1, -1, -1, -1, -1, -1, -1};
  int mTracklets[2] = {-1, -1};
  int mLevel = 0;
  float mChi2 = 0.f;
};

} // namespace its
} // namespace o2
#endif /* TRACKINGITSU_INCLUDE_CACELL_H_ */
