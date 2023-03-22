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

#ifndef TRACKINGITSU_INCLUDE_TRACKLET_H_
#define TRACKINGITSU_INCLUDE_TRACKLET_H_

#include "ITStracking/Cluster.h"
#include <iostream>
#include "GPUCommonMath.h"
#include "GPUCommonDef.h"

namespace o2
{
namespace its
{

struct Tracklet final {
  GPUhdi() Tracklet();
  GPUhdi() Tracklet(const int, const int, const Cluster&, const Cluster&, int rof0, int rof1);
  GPUhdi() Tracklet(const int, const int, float tanL, float phi, int rof0, int rof1);
  GPUhdi() bool operator==(const Tracklet&) const;
  GPUhdi() bool operator!=(const Tracklet&) const;
  GPUhdi() unsigned char isEmpty() const
  {
    return firstClusterIndex < 0 || secondClusterIndex < 0;
  }
  GPUhdi() void dump();
  GPUhdi() void dump() const;
  GPUhdi() void dump(const int, const int);
  GPUhdi() void dump(const int, const int) const;
  GPUhdi() unsigned char operator<(const Tracklet&) const;

  int firstClusterIndex;
  int secondClusterIndex;
  float tanLambda;
  float phi;
  unsigned short rof[2];
};

GPUhdi() Tracklet::Tracklet() : firstClusterIndex{-1}, secondClusterIndex{-1}, tanLambda{0.0f}, phi{0.0f}
{
  rof[0] = 0;
  rof[1] = 0;
}

GPUhdi() Tracklet::Tracklet(const int firstClusterOrderingIndex, const int secondClusterOrderingIndex,
                            const Cluster& firstCluster, const Cluster& secondCluster, int rof0 = -1, int rof1 = -1)
  : firstClusterIndex{firstClusterOrderingIndex},
    secondClusterIndex{secondClusterOrderingIndex},
    tanLambda{(firstCluster.zCoordinate - secondCluster.zCoordinate) /
              (firstCluster.radius - secondCluster.radius)},
    phi{o2::gpu::GPUCommonMath::ATan2(firstCluster.yCoordinate - secondCluster.yCoordinate,
                                      firstCluster.xCoordinate - secondCluster.xCoordinate)},
    rof{static_cast<unsigned short>(rof0), static_cast<unsigned short>(rof1)}
{
  // Nothing to do
}

GPUhdi() Tracklet::Tracklet(const int idx0, const int idx1, float tanL, float phi, int rof0, int rof1)
  : firstClusterIndex{idx0},
    secondClusterIndex{idx1},
    tanLambda{tanL},
    phi{phi},
    rof{static_cast<unsigned short>(rof0), static_cast<unsigned short>(rof1)}
{
  // Nothing to do
}

GPUhdi() bool Tracklet::operator==(const Tracklet& rhs) const
{
  return this->firstClusterIndex == rhs.firstClusterIndex &&
         this->secondClusterIndex == rhs.secondClusterIndex &&
         this->tanLambda == rhs.tanLambda &&
         this->phi == rhs.phi &&
         this->rof[0] == rhs.rof[0] &&
         this->rof[1] == rhs.rof[1];
}

GPUhdi() bool Tracklet::operator!=(const Tracklet& rhs) const
{
  return this->firstClusterIndex != rhs.firstClusterIndex ||
         this->secondClusterIndex != rhs.secondClusterIndex ||
         this->tanLambda != rhs.tanLambda ||
         this->phi != rhs.phi;
}

GPUhdi() unsigned char Tracklet::operator<(const Tracklet& t) const
{
  if (isEmpty()) {
    return false;
  }
  return true;
}

// GPUhdi() void Tracklet::dump()
// {
//   printf("fClIdx: %d sClIdx: %d  rof1: %hu rof2: %hu phi: %f tl: %f \n", firstClusterIndex, secondClusterIndex, rof[0], rof[1], phi, tanLambda);
// }

// GPUhdi() void Tracklet::dump() const
// {
//   printf("fClIdx: %d sClIdx: %d  rof1: %hu rof2: %hu phi: %f tl: %f \n", firstClusterIndex, secondClusterIndex, rof[0], rof[1], phi, tanLambda);
// }

// GPUhdi() void Tracklet::dump(const int offsetFirst, const int offsetSecond)
// {
//   printf("fClIdx: %d sClIdx: %d  rof1: %hu rof2: %hu phi: %f tl: %f \n", firstClusterIndex + offsetFirst, secondClusterIndex + offsetSecond, rof[0], rof[1], phi, tanLambda);
// }

// GPUhdi() void Tracklet::dump(const int offsetFirst, const int offsetSecond) const
// {
//   printf("fClIdx: %d sClIdx: %d  rof1: %hu rof2: %hu phi: %f tl: %f \n", firstClusterIndex + offsetFirst, secondClusterIndex + offsetSecond, rof[0], rof[1], phi, tanLambda);
// }

GPUhdi() void Tracklet::dump(const int offsetFirst, const int offsetSecond)
{
  printf("fClIdx: %d sClIdx: %d  rof1: %hu rof2: %hu\n", firstClusterIndex + offsetFirst, secondClusterIndex + offsetSecond, rof[0], rof[1]);
}

GPUhdi() void Tracklet::dump(const int offsetFirst, const int offsetSecond) const
{
  printf("fClIdx: %d sClIdx: %d  rof1: %hu rof2: %hu\n", firstClusterIndex + offsetFirst, secondClusterIndex + offsetSecond, rof[0], rof[1]);
}

GPUhdi() void Tracklet::dump()
{
  printf("fClIdx: %d sClIdx: %d  rof1: %hu rof2: %hu\n", firstClusterIndex, secondClusterIndex, rof[0], rof[1]);
}

GPUhdi() void Tracklet::dump() const
{
  printf("fClIdx: %d sClIdx: %d  rof1: %hu rof2: %hu\n", firstClusterIndex, secondClusterIndex, rof[0], rof[1]);
}

} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_TRACKLET_H_ */
