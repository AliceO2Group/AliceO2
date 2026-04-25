// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
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
/// \file Tracklet.h
/// \brief
///

#ifndef TRACKINGITS_INCLUDE_TRACKLET_H_
#define TRACKINGITS_INCLUDE_TRACKLET_H_

#include "ITStracking/Constants.h"
#include "DataFormatsITS/TimeEstBC.h"
#include "ITStracking/Cluster.h"
#include "GPUCommonRtypes.h"
#include "GPUCommonMath.h"
#include "GPUCommonDef.h"
#include "GPUCommonLogger.h"

namespace o2::its
{

// tracklets are entirely determined by their two cluster idx
struct Tracklet final {
  GPUhdDefault() Tracklet() = default;
  GPUhdi() Tracklet(const int firstClusterOrderingIndex, const int secondClusterOrderingIndex,
                    const Cluster& firstCluster, const Cluster& secondCluster, const TimeEstBC& t)
    : firstClusterIndex(firstClusterOrderingIndex),
      secondClusterIndex(secondClusterOrderingIndex),
      tanLambda((firstCluster.zCoordinate - secondCluster.zCoordinate) / (firstCluster.radius - secondCluster.radius)),
      phi(o2::gpu::GPUCommonMath::ATan2(firstCluster.yCoordinate - secondCluster.yCoordinate, firstCluster.xCoordinate - secondCluster.xCoordinate)),
      mTime(t) {}

  GPUhdi() Tracklet(const int idx0, const int idx1, float tanL, float phi, const TimeEstBC& t)
    : firstClusterIndex(idx0),
      secondClusterIndex(idx1),
      tanLambda(tanL),
      phi(phi),
      mTime(t) {}
  GPUhdi() bool operator<(const Tracklet& o) const noexcept
  {
    return (firstClusterIndex != o.firstClusterIndex) ? firstClusterIndex < o.firstClusterIndex : secondClusterIndex < o.secondClusterIndex;
  }
  GPUhdi() bool operator==(const Tracklet& o) const noexcept
  {
    return firstClusterIndex == o.firstClusterIndex && secondClusterIndex == o.secondClusterIndex;
  }
  GPUhdi() bool isCompatible(const Tracklet& o) const { return mTime.isCompatible(o.mTime); }
  GPUhd() void print() const
  {
    LOGP(info, "TRKLT: fClIdx:{} sClIdx:{} ts:{}+/-{} TgL={} Phi={}", firstClusterIndex, secondClusterIndex, mTime.getTimeStamp(), mTime.getTimeStampError(), tanLambda, phi);
  }
  GPUhd() auto& getTimeStamp() noexcept { return mTime; }
  GPUhd() const auto& getTimeStamp() const noexcept { return mTime; }

  int firstClusterIndex{constants::UnusedIndex};
  int secondClusterIndex{constants::UnusedIndex};
  float tanLambda{constants::UnsetValue};
  float phi{constants::UnsetValue};
  TimeEstBC mTime;

  ClassDefNV(Tracklet, 1);
};

} // namespace o2::its

#endif /* TRACKINGITS_INCLUDE_TRACKLET_H_ */
