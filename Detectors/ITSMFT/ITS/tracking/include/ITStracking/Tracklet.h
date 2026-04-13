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

#ifndef GPUCA_GPUCODE_DEVICE
#ifndef GPU_NO_FMT
#include <string>
#include <fmt/format.h>
#endif
#endif

namespace o2::its
{

struct Tracklet final {
  GPUhdDefault() Tracklet() = default;
  GPUhdi() Tracklet(const int, const int, const Cluster&, const Cluster&, const TimeEstBC& t);
  GPUhdi() Tracklet(const int, const int, float tanL, float phi, const TimeEstBC& t);
  GPUhdDefault() bool operator==(const Tracklet&) const = default;
  GPUhdi() unsigned char isEmpty() const
  {
    return firstClusterIndex < 0 || secondClusterIndex < 0;
  }
  GPUhdi() bool isCompatible(const Tracklet& o) const { return mTime.isCompatible(o.mTime); }
  GPUhdi() unsigned char operator<(const Tracklet&) const;
  GPUhd() void print() const
  {
    LOGP(info, "TRKLT: fClIdx:{} sClIdx:{} ts:{}+/-{} TgL={} Phi={}", firstClusterIndex, secondClusterIndex, mTime.getTimeStamp(), mTime.getTimeStampError(), tanLambda, phi);
  }
  GPUhd() auto& getTimeStamp() noexcept { return mTime; }
  GPUhd() const auto& getTimeStamp() const noexcept { return mTime; }

  int firstClusterIndex{constants::UnusedIndex};
  int secondClusterIndex{constants::UnusedIndex};
  float tanLambda{-999};
  float phi{-999};
  TimeEstBC mTime;

  ClassDefNV(Tracklet, 1);
};

GPUhdi() Tracklet::Tracklet(const int firstClusterOrderingIndex, const int secondClusterOrderingIndex,
                            const Cluster& firstCluster, const Cluster& secondCluster, const TimeEstBC& t)
  : firstClusterIndex(firstClusterOrderingIndex),
    secondClusterIndex(secondClusterOrderingIndex),
    tanLambda((firstCluster.zCoordinate - secondCluster.zCoordinate) /
              (firstCluster.radius - secondCluster.radius)),
    phi(o2::gpu::GPUCommonMath::ATan2(firstCluster.yCoordinate - secondCluster.yCoordinate,
                                      firstCluster.xCoordinate - secondCluster.xCoordinate)),
    mTime(t)
{
  // Nothing to do
}

GPUhdi() Tracklet::Tracklet(const int idx0, const int idx1, float tanL, float phi, const TimeEstBC& t)
  : firstClusterIndex(idx0),
    secondClusterIndex(idx1),
    tanLambda(tanL),
    phi(phi),
    mTime(t)
{
  // Nothing to do
}

GPUhdi() unsigned char Tracklet::operator<(const Tracklet& t) const
{
  if (isEmpty()) {
    return false;
  }
  return true;
}

} // namespace o2::its

#endif /* TRACKINGITS_INCLUDE_TRACKLET_H_ */
