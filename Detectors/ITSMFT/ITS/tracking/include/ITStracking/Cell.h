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
  GPUd() Cell(const int, const int, const int, const int, const int, const float);

  GPUhd() int getFirstClusterIndex() const { return mFirstClusterIndex; };
  GPUhd() int getSecondClusterIndex() const { return mSecondClusterIndex; };
  GPUhd() int getThirdClusterIndex() const { return mThirdClusterIndex; };
  GPUhd() int getFirstTrackletIndex() const { return mFirstTrackletIndex; };
  GPUhd() int getSecondTrackletIndex() const { return mSecondTrackletIndex; };
  GPUhd() float getTanLambda() const { return mTanLambda; };
  GPUhd() int getLevel() const { return mLevel; };
  GPUhd() void setLevel(const int level) { mLevel = level; };
  GPUhd() int* getLevelPtr() { return &mLevel; }

 private:
  const int mFirstClusterIndex;
  const int mSecondClusterIndex;
  const int mThirdClusterIndex;
  const int mFirstTrackletIndex;
  const int mSecondTrackletIndex;
  const float mTanLambda;
  int mLevel;
};

GPUhdi() Cell::Cell()
  : mFirstClusterIndex{0},
    mSecondClusterIndex{0},
    mThirdClusterIndex{0},
    mFirstTrackletIndex{0},
    mSecondTrackletIndex{0},
    mTanLambda{0},
    mLevel{0}
{
  // Nothing to do
}

GPUdi() Cell::Cell(const int firstClusterIndex, const int secondClusterIndex, const int thirdClusterIndex,
                   const int firstTrackletIndex, const int secondTrackletIndex, const float tanL)
  : mFirstClusterIndex{firstClusterIndex},
    mSecondClusterIndex{secondClusterIndex},
    mThirdClusterIndex{thirdClusterIndex},
    mFirstTrackletIndex{firstTrackletIndex},
    mSecondTrackletIndex{secondTrackletIndex},
    mTanLambda{tanL},
    mLevel{1}
{
  // Nothing to do
}

} // namespace its
} // namespace o2
#endif /* TRACKINGITSU_INCLUDE_CACELL_H_ */
