// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#ifndef __OPENCL__
#include <array>
#include <vector>
#endif

#include "ITStracking/Definitions.h"
#include "GPUCommonDef.h"

namespace o2
{
namespace its
{

class Cell final
{
 public:
  GPU_DEVICE Cell(const int, const int, const int, const int, const int, const float3&, const float);

  GPUhdni() int getFirstClusterIndex() const;
  GPUhdni() int getSecondClusterIndex() const;
  GPUhdni() int getThirdClusterIndex() const;
  GPU_HOST_DEVICE int getFirstTrackletIndex() const;
  int getSecondTrackletIndex() const;
  int getLevel() const;
  float getCurvature() const;
  const float3& getNormalVectorCoordinates() const;
  void setLevel(const int level);

 private:
  const int mFirstClusterIndex;
  const int mSecondClusterIndex;
  const int mThirdClusterIndex;
  const int mFirstTrackletIndex;
  const int mSecondTrackletIndex;
  const float3 mNormalVectorCoordinates;
  const float mCurvature;
  int mLevel;
};

inline GPU_DEVICE Cell::Cell(const int firstClusterIndex, const int secondClusterIndex, const int thirdClusterIndex,
                             const int firstTrackletIndex, const int secondTrackletIndex,
                             const float3& normalVectorCoordinates, const float curvature)
  : mFirstClusterIndex{firstClusterIndex},
    mSecondClusterIndex{secondClusterIndex},
    mThirdClusterIndex{thirdClusterIndex},
    mFirstTrackletIndex(firstTrackletIndex),
    mSecondTrackletIndex(secondTrackletIndex),
    mNormalVectorCoordinates(normalVectorCoordinates),
    mCurvature{curvature},
    mLevel{1}
{
  // Nothing to do
}

GPUhdi() int Cell::getFirstClusterIndex() const { return mFirstClusterIndex; }

GPUhdi() int Cell::getSecondClusterIndex() const { return mSecondClusterIndex; }

GPUhdi() int Cell::getThirdClusterIndex() const { return mThirdClusterIndex; }

GPU_HOST_DEVICE inline int Cell::getFirstTrackletIndex() const { return mFirstTrackletIndex; }

inline int Cell::getSecondTrackletIndex() const { return mSecondTrackletIndex; }

inline int Cell::getLevel() const { return mLevel; }

inline float Cell::getCurvature() const { return mCurvature; }

inline const float3& Cell::getNormalVectorCoordinates() const { return mNormalVectorCoordinates; }

inline void Cell::setLevel(const int level) { mLevel = level; }
} // namespace its
} // namespace o2
#endif /* TRACKINGITSU_INCLUDE_CACELL_H_ */
