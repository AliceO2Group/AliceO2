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
/// \file Cluster.cxx
/// \brief
///

#include "ITStracking/Cluster.h"

#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/MathUtils.h"

namespace o2
{
namespace its
{

using MathUtils::calculatePhiCoordinate;
using MathUtils::calculateRCoordinate;
using MathUtils::getNormalizedPhiCoordinate;

Cluster::Cluster(const float x, const float y, const float z, const int index)
  : xCoordinate{ x },
    yCoordinate{ y },
    zCoordinate{ z },
    phiCoordinate{ getNormalizedPhiCoordinate(calculatePhiCoordinate(x, y)) },
    rCoordinate{ calculateRCoordinate(x, y) },
    clusterId{ index },
    indexTableBinIndex{ 0 }
{
  // Nothing to do
}

Cluster::Cluster(const int layerIndex, const Cluster& other)
  : xCoordinate{ other.xCoordinate },
    yCoordinate{ other.yCoordinate },
    zCoordinate{ other.zCoordinate },
    phiCoordinate{ getNormalizedPhiCoordinate(calculatePhiCoordinate(other.xCoordinate, other.yCoordinate)) },
    rCoordinate{ calculateRCoordinate(other.xCoordinate, other.yCoordinate) },
    clusterId{ other.clusterId },
    indexTableBinIndex{ IndexTableUtils::getBinIndex(IndexTableUtils::getZBinIndex(layerIndex, zCoordinate),
                                                     IndexTableUtils::getPhiBinIndex(phiCoordinate)) }
//, montecarloId{ other.montecarloId }
{
  // Nothing to do
}

Cluster::Cluster(const int layerIndex, const float3& primaryVertex, const Cluster& other)
  : xCoordinate{ other.xCoordinate },
    yCoordinate{ other.yCoordinate },
    zCoordinate{ other.zCoordinate },
    phiCoordinate{ getNormalizedPhiCoordinate(
      calculatePhiCoordinate(xCoordinate - primaryVertex.x, yCoordinate - primaryVertex.y)) },
    rCoordinate{ calculateRCoordinate(xCoordinate - primaryVertex.x, yCoordinate - primaryVertex.y) },
    clusterId{ other.clusterId },
    indexTableBinIndex{ IndexTableUtils::getBinIndex(IndexTableUtils::getZBinIndex(layerIndex, zCoordinate),
                                                     IndexTableUtils::getPhiBinIndex(phiCoordinate)) }
{
  // Nothing to do
}

TrackingFrameInfo::TrackingFrameInfo(float x, float y, float z, float xTF, float alpha, GPUArray<float, 2>&& posTF,
                                     GPUArray<float, 3>&& covTF)
  : xCoordinate{ x }, yCoordinate{ y }, zCoordinate{ z }, xTrackingFrame{ xTF }, alphaTrackingFrame{ alpha },
#ifdef __OPENCL__
    positionTrackingFrame{ posTF },
    covarianceTrackingFrame
{
  covTF
}
#else
    positionTrackingFrame{},
    covarianceTrackingFrame
{
}
#endif
{
  // Nothing to do
#ifdef __OPENCL__
  positionTrackingFrame.copy(posTF);
  covarianceTrackingFrame.copy(covTF);
#endif
}
} // namespace its
} // namespace o2
