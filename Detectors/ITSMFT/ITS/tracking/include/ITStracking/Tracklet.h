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
/// \file Tracklet.h
/// \brief
///

#ifndef TRACKINGITS_INCLUDE_TRACKLET_H_
#define TRACKINGITS_INCLUDE_TRACKLET_H_

#include "ITStracking/Constants.h"
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
  GPUhdi() Tracklet(const int, const int, const Cluster&, const Cluster&, short rof0, short rof1);
  GPUhdi() Tracklet(const int, const int, float tanL, float phi, short rof0, short rof1);
  GPUhdDefault() bool operator==(const Tracklet&) const = default;
  GPUhdi() unsigned char isEmpty() const
  {
    return firstClusterIndex < 0 || secondClusterIndex < 0;
  }
  GPUhdi() auto getMinRof() const noexcept { return o2::gpu::CAMath::Min(rof[0], rof[1]); }
  GPUhdi() auto getMaxRof() const noexcept { return o2::gpu::CAMath::Max(rof[0], rof[1]); }
  GPUhdi() auto getDeltaRof() const { return rof[1] - rof[0]; }
  GPUhdi() auto getSpanRof(const Tracklet& o) const noexcept { return o2::gpu::CAMath::Max(getMaxRof(), o.getMaxRof()) - o2::gpu::CAMath::Min(getMinRof(), o.getMinRof()); }
  GPUhdi() unsigned char operator<(const Tracklet&) const;
  GPUhd() void print() const
  {
    printf("TRKLT: fClIdx:%d fROF:%d sClIdx:%d sROF:%d (DROF:%d) tgl=%f phi=%f\n", firstClusterIndex, rof[0], secondClusterIndex, rof[1], getDeltaRof(), tanLambda, phi);
  }

  int firstClusterIndex{constants::UnusedIndex};
  int secondClusterIndex{constants::UnusedIndex};
  float tanLambda{-999};
  float phi{-999};
  short rof[2] = {constants::UnusedIndex, constants::UnusedIndex};

  ClassDefNV(Tracklet, 1);
};

GPUhdi() Tracklet::Tracklet(const int firstClusterOrderingIndex, const int secondClusterOrderingIndex,
                            const Cluster& firstCluster, const Cluster& secondCluster, short rof0 = -1, short rof1 = -1)
  : firstClusterIndex{firstClusterOrderingIndex},
    secondClusterIndex{secondClusterOrderingIndex},
    tanLambda{(firstCluster.zCoordinate - secondCluster.zCoordinate) /
              (firstCluster.radius - secondCluster.radius)},
    phi{o2::gpu::GPUCommonMath::ATan2(firstCluster.yCoordinate - secondCluster.yCoordinate,
                                      firstCluster.xCoordinate - secondCluster.xCoordinate)},
    rof{static_cast<short>(rof0), static_cast<short>(rof1)}
{
  // Nothing to do
}

GPUhdi() Tracklet::Tracklet(const int idx0, const int idx1, float tanL, float phi, short rof0, short rof1)
  : firstClusterIndex{idx0},
    secondClusterIndex{idx1},
    tanLambda{tanL},
    phi{phi},
    rof{static_cast<short>(rof0), static_cast<short>(rof1)}
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
