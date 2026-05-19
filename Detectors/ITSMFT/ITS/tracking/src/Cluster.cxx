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
#include "GPUCommonMath.h"
#include "GPUCommonArray.h"

#include "ITStracking/Cluster.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/MathUtils.h"
#include "ITStracking/IndexTableUtils.h"

using namespace o2::its;

using math_utils::computePhi;
using math_utils::getNormalizedPhi;

Cluster::Cluster(const float x, const float y, const float z, const int index)
  : xCoordinate{x},
    yCoordinate{y},
    zCoordinate{z},
    phi{getNormalizedPhi(computePhi(x, y))},
    radius{o2::gpu::GPUCommonMath::Hypot(x, y)},
    clusterId{index},
    indexTableBinIndex{0}
{
  // Nothing to do
}

template <int nLayers>
Cluster::Cluster(const int layerIndex, const IndexTableUtils<nLayers>& utils, const Cluster& other)
  : xCoordinate{other.xCoordinate},
    yCoordinate{other.yCoordinate},
    zCoordinate{other.zCoordinate},
    phi{getNormalizedPhi(computePhi(other.xCoordinate, other.yCoordinate))},
    radius{o2::gpu::GPUCommonMath::Hypot(other.xCoordinate, other.yCoordinate)},
    clusterId{other.clusterId},
    indexTableBinIndex{utils.getBinIndex(utils.getZBinIndex(layerIndex, zCoordinate),
                                         utils.getPhiBinIndex(phi))}
//, montecarloId{ other.montecarloId }
{
  // Nothing to do
}

template <int nLayers>
Cluster::Cluster(const int layerIndex, const float3& primaryVertex, const IndexTableUtils<nLayers>& utils, const Cluster& other)
  : xCoordinate{other.xCoordinate},
    yCoordinate{other.yCoordinate},
    zCoordinate{other.zCoordinate},
    phi{getNormalizedPhi(
      computePhi(xCoordinate - primaryVertex.x, yCoordinate - primaryVertex.y))},
    radius{o2::gpu::GPUCommonMath::Hypot(xCoordinate - primaryVertex.x, yCoordinate - primaryVertex.y)},
    clusterId{other.clusterId},
    indexTableBinIndex{utils.getBinIndex(utils.getZBinIndex(layerIndex, zCoordinate),
                                         utils.getPhiBinIndex(phi))}
{
  // Nothing to do
}

GPUhd() void Cluster::print() const
{
  printf("Cluster: %f %f %f %f %f %d %d\n", xCoordinate, yCoordinate, zCoordinate, phi, radius, clusterId, indexTableBinIndex);
}

TrackingFrameInfo::TrackingFrameInfo(float x, float y, float z, float xTF, float alpha, std::array<float, 2>&& posTF,
                                     std::array<float, 3>&& covTF)
  : xCoordinate{x}, yCoordinate{y}, zCoordinate{z}, xTrackingFrame{xTF}, alphaTrackingFrame{alpha}, positionTrackingFrame{posTF}, covarianceTrackingFrame{covTF}
{
  // Nothing to do
}

GPUhd() void TrackingFrameInfo::print() const
{
  printf("x: %f y: %f z: %f xTF: %f alphaTF: %f posTF: %f %f covTF: %f %f %f\n",
         xCoordinate, yCoordinate, zCoordinate, xTrackingFrame, alphaTrackingFrame,
         positionTrackingFrame[0], positionTrackingFrame[1],
         covarianceTrackingFrame[0], covarianceTrackingFrame[1], covarianceTrackingFrame[2]);
}
