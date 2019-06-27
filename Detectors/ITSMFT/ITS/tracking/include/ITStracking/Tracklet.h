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
/// \file Tracklet.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_TRACKLET_H_
#define TRACKINGITSU_INCLUDE_TRACKLET_H_

#include "ITStracking/Cluster.h"
#include <iostream>
#include "GPUCommonMath.h"

namespace o2
{
namespace its
{

struct Tracklet final {
  Tracklet();
  GPU_DEVICE Tracklet(const int, const int, const Cluster&, const Cluster&);
  void dump();

  const int firstClusterIndex;
  const int secondClusterIndex;
  const float tanLambda;
  const float phiCoordinate;
};

inline Tracklet::Tracklet() : firstClusterIndex{ 0 }, secondClusterIndex{ 0 }, tanLambda{ 0.0f }, phiCoordinate{ 0.0f }
{
  // Nothing to do
}

inline GPU_DEVICE Tracklet::Tracklet(const int firstClusterOrderingIndex, const int secondClusterOrderingIndex,
                                     const Cluster& firstCluster, const Cluster& secondCluster)
  : firstClusterIndex{ firstClusterOrderingIndex },
    secondClusterIndex{ secondClusterOrderingIndex },
    tanLambda{ (firstCluster.zCoordinate - secondCluster.zCoordinate) /
               (firstCluster.rCoordinate - secondCluster.rCoordinate) },
    phiCoordinate{ gpu::GPUCommonMath::ATan2(firstCluster.yCoordinate - secondCluster.yCoordinate,
                                             firstCluster.xCoordinate - secondCluster.xCoordinate) }
{
  // Nothing to do
}

inline void Tracklet::dump()
{
  std::cout << "firstClusterIndex: " << firstClusterIndex << std::endl;
  std::cout << "secondClusterIndex: " << secondClusterIndex << std::endl;
  std::cout << "tanLambda: " << tanLambda << std::endl;
  std::cout << "phiCoordinate: " << phiCoordinate << std::endl;
}

} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_TRACKLET_H_ */
