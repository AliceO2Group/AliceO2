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
/// \file Cluster.cxx
/// \brief
///

#include "ITStracking/Cluster.h"
#include "ITStracking/MathUtils.h"
#include "ITStracking/IndexTableUtils.h"

namespace o2
{
namespace its
{

using math_utils::computePhi;
using math_utils::getNormalizedPhi;
using math_utils::hypot;

Cluster::Cluster(const float x, const float y, const float z, const int index)
  : xCoordinate{x},
    yCoordinate{y},
    zCoordinate{z},
    phi{getNormalizedPhi(computePhi(x, y))},
    radius{hypot(x, y)},
    clusterId{index},
    indexTableBinIndex{0}
{
  // Nothing to do
}

Cluster::Cluster(const int layerIndex, const IndexTableUtils& utils, const Cluster& other)
  : xCoordinate{other.xCoordinate},
    yCoordinate{other.yCoordinate},
    zCoordinate{other.zCoordinate},
    phi{getNormalizedPhi(computePhi(other.xCoordinate, other.yCoordinate))},
    radius{hypot(other.xCoordinate, other.yCoordinate)},
    clusterId{other.clusterId},
    indexTableBinIndex{utils.getBinIndex(utils.getZBinIndex(layerIndex, zCoordinate),
                                         utils.getPhiBinIndex(phi))}
//, montecarloId{ other.montecarloId }
{
  // Nothing to do
}

Cluster::Cluster(const int layerIndex, const float3& primaryVertex, const IndexTableUtils& utils, const Cluster& other)
  : xCoordinate{other.xCoordinate},
    yCoordinate{other.yCoordinate},
    zCoordinate{other.zCoordinate},
    phi{getNormalizedPhi(
      computePhi(xCoordinate - primaryVertex.x, yCoordinate - primaryVertex.y))},
    radius{hypot(xCoordinate - primaryVertex.x, yCoordinate - primaryVertex.y)},
    clusterId{other.clusterId},
    indexTableBinIndex{utils.getBinIndex(utils.getZBinIndex(layerIndex, zCoordinate),
                                         utils.getPhiBinIndex(phi))}
{
  // Nothing to do
}

void Cluster::Init(const int layerIndex, const float3& primaryVertex, const IndexTableUtils& utils, const Cluster& other)
{
  xCoordinate = other.xCoordinate;
  yCoordinate = other.yCoordinate;
  zCoordinate = other.zCoordinate;
  phi = getNormalizedPhi(
    computePhi(xCoordinate - primaryVertex.x, yCoordinate - primaryVertex.y));
  radius = hypot(xCoordinate - primaryVertex.x, yCoordinate - primaryVertex.y);
  clusterId = other.clusterId;
  indexTableBinIndex = utils.getBinIndex(utils.getZBinIndex(layerIndex, zCoordinate),
                                         utils.getPhiBinIndex(phi));
}

bool Cluster::operator==(const Cluster& rhs) const
{
  return this->xCoordinate == rhs.xCoordinate &&
         this->yCoordinate == rhs.yCoordinate &&
         this->zCoordinate == rhs.zCoordinate &&
         this->phi == rhs.phi &&
         this->radius == rhs.radius &&
         this->clusterId == rhs.clusterId &&
         this->indexTableBinIndex == rhs.indexTableBinIndex;
}

void Cluster::print() const {
  printf("Cluster: %f %f %f %f %f %d %d\n", xCoordinate, yCoordinate, zCoordinate, phi, radius, clusterId, indexTableBinIndex);
}

TrackingFrameInfo::TrackingFrameInfo(float x, float y, float z, float xTF, float alpha, GPUArray<float, 2>&& posTF,
                                     GPUArray<float, 3>&& covTF)
  : xCoordinate{x}, yCoordinate{y}, zCoordinate{z}, xTrackingFrame{xTF}, alphaTrackingFrame{alpha}, positionTrackingFrame{posTF}, covarianceTrackingFrame{covTF}
{
  // Nothing to do
}

} // namespace its
} // namespace o2
