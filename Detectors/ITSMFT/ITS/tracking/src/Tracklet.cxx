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
/// \file Tracklet.cxx
/// \brief
///

#include <cmath>

#include "ITStracking/Tracklet.h"

namespace o2
{
namespace ITS
{

Tracklet::Tracklet() : firstClusterIndex{ 0 }, secondClusterIndex{ 0 }, tanLambda{ 0.0f }, phiCoordinate{ 0.0f }
{
  // Nothing to do
}

GPU_DEVICE Tracklet::Tracklet(const int firstClusterOrderingIndex, const int secondClusterOrderingIndex,
                              const Cluster& firstCluster, const Cluster& secondCluster)
  : firstClusterIndex{ firstClusterOrderingIndex },
    secondClusterIndex{ secondClusterOrderingIndex },
    tanLambda{ (firstCluster.zCoordinate - secondCluster.zCoordinate) /
               (firstCluster.rCoordinate - secondCluster.rCoordinate) },
    phiCoordinate{ MATH_ATAN2(firstCluster.yCoordinate - secondCluster.yCoordinate,
                              firstCluster.xCoordinate - secondCluster.xCoordinate) }
{
  // Nothing to do
}
} // namespace ITS
} // namespace o2
