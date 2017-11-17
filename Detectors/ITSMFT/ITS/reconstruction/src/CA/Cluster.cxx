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

#include "ITSReconstruction/CA/Cluster.h"

#include "ITSReconstruction/CA/IndexTableUtils.h"
#include "ITSReconstruction/CA/MathUtils.h"

namespace o2
{
namespace ITS
{
namespace CA
{

Cluster::Cluster(const float x, const float y, const float z, const int index)
    : xCoordinate { x }, yCoordinate { y }, zCoordinate { z }, phiCoordinate { 0 }, rCoordinate {
        0 }, clusterId { index }, indexTableBinIndex { 0 }
{
  // Nothing to do
}

Cluster::Cluster(const int layerIndex, const Cluster& other)
    : xCoordinate { other.xCoordinate }, yCoordinate { other.yCoordinate }, zCoordinate { other.zCoordinate }, 
      phiCoordinate{ MathUtils::getNormalizedPhiCoordinate(MathUtils::calculatePhiCoordinate(other.xCoordinate, other.yCoordinate)) },
      rCoordinate{ MathUtils::calculateRCoordinate(other.xCoordinate, other.yCoordinate) },
      clusterId{ other.clusterId },
      indexTableBinIndex{ IndexTableUtils::getBinIndex(IndexTableUtils::getZBinIndex(layerIndex, zCoordinate),
      IndexTableUtils::getPhiBinIndex(phiCoordinate)) } //, montecarloId{ other.montecarloId }
{
  // Nothing to do
}

Cluster::Cluster(const int layerIndex, const float3 &primaryVertex, const Cluster& other)
    : xCoordinate { other.xCoordinate }, yCoordinate { other.yCoordinate }, zCoordinate { other.zCoordinate }, phiCoordinate {
        MathUtils::getNormalizedPhiCoordinate(
            MathUtils::calculatePhiCoordinate(xCoordinate - primaryVertex.x, yCoordinate - primaryVertex.y)) }, rCoordinate {
        MathUtils::calculateRCoordinate(xCoordinate - primaryVertex.x, yCoordinate - primaryVertex.y) }, clusterId {
        other.clusterId }, indexTableBinIndex {
        IndexTableUtils::getBinIndex(IndexTableUtils::getZBinIndex(layerIndex, zCoordinate),
            IndexTableUtils::getPhiBinIndex(phiCoordinate)) }
{
  // Nothing to do
}

TrackingFrameInfo::TrackingFrameInfo(float xTF, float alpha, std::array<float, 2>&& posTF, std::array<float, 3>&& covTF)
  : xTrackingFrame{ xTF }, alphaTrackingFrame{ alpha }, positionTrackingFrame{ posTF }, covarianceTrackingFrame{ covTF }
{
  // Nothing to do
}

}
}
}