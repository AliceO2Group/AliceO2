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

#include <array>
#include <vector>

#include "ITStracking/Definitions.h"

namespace o2
{
namespace ITS
{

class Cell final
{
 public:
  GPU_DEVICE Cell(const int, const int, const int, const int, const int, const float3&, const float);

  int getFirstClusterIndex() const;
  int getSecondClusterIndex() const;
  int getThirdClusterIndex() const;
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

inline int Cell::getFirstClusterIndex() const { return mFirstClusterIndex; }

inline int Cell::getSecondClusterIndex() const { return mSecondClusterIndex; }

inline int Cell::getThirdClusterIndex() const { return mThirdClusterIndex; }

GPU_HOST_DEVICE inline int Cell::getFirstTrackletIndex() const { return mFirstTrackletIndex; }

inline int Cell::getSecondTrackletIndex() const { return mSecondTrackletIndex; }

inline int Cell::getLevel() const { return mLevel; }

inline float Cell::getCurvature() const { return mCurvature; }

inline const float3& Cell::getNormalVectorCoordinates() const { return mNormalVectorCoordinates; }

inline void Cell::setLevel(const int level) { mLevel = level; }
} // namespace ITS
} // namespace o2
#endif /* TRACKINGITSU_INCLUDE_CACELL_H_ */
