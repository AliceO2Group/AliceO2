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
/// \file IndexTableUtils.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_INDEXTABLEUTILS_H_
#define TRACKINGITSU_INCLUDE_INDEXTABLEUTILS_H_

#ifndef __OPENCL__
#include <array>
#include <utility>
#include <vector>
#endif

#include "ITStracking/Constants.h"
#include "ITStracking/Definitions.h"
#include "GPUCommonMath.h"

namespace o2
{
namespace its
{

namespace index_table_utils
{
float getInverseZBinSize(const int);
GPU_HOST_DEVICE int getZBinIndex(const int, const float);
GPU_HOST_DEVICE int getPhiBinIndex(const float);
GPU_HOST_DEVICE int getBinIndex(const int, const int);
GPU_HOST_DEVICE int countRowSelectedBins(
  const GPUArray<int, constants::index_table::ZBins * constants::index_table::PhiBins + 1>&, const int, const int,
  const int);
} // namespace index_table_utils

inline float getInverseZCoordinate(const int layerIndex)
{
  return 0.5f * constants::index_table::ZBins / constants::its::LayersZCoordinate()[layerIndex];
}

GPU_HOST_DEVICE inline int index_table_utils::getZBinIndex(const int layerIndex, const float zCoordinate)
{
  return (zCoordinate + constants::its::LayersZCoordinate()[layerIndex]) *
         constants::index_table::InverseZBinSize()[layerIndex];
}

GPU_HOST_DEVICE inline int index_table_utils::getPhiBinIndex(const float currentPhi)
{
  return (currentPhi * constants::index_table::InversePhiBinSize);
}

GPU_HOST_DEVICE inline int index_table_utils::getBinIndex(const int zIndex, const int phiIndex)
{
  return gpu::GPUCommonMath::Min(phiIndex * constants::index_table::ZBins + zIndex,
                                 constants::index_table::ZBins * constants::index_table::PhiBins - 1);
}

GPU_HOST_DEVICE inline int index_table_utils::countRowSelectedBins(
  const GPUArray<int, constants::index_table::ZBins * constants::index_table::PhiBins + 1>& indexTable,
  const int phiBinIndex, const int minZBinIndex, const int maxZBinIndex)
{
  const int firstBinIndex{getBinIndex(minZBinIndex, phiBinIndex)};
  const int maxBinIndex{firstBinIndex + maxZBinIndex - minZBinIndex + 1};

  return indexTable[maxBinIndex] - indexTable[firstBinIndex];
}
} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_INDEXTABLEUTILS_H_ */
