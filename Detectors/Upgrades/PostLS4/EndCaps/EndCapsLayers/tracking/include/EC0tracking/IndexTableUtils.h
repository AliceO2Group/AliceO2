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

#ifndef TRACKINGEC0__INCLUDE_INDEXTABLEUTILS_H_
#define TRACKINGEC0__INCLUDE_INDEXTABLEUTILS_H_

#ifndef __OPENCL__
#include <array>
#include <utility>
#include <vector>
#endif

#include "EC0tracking/Constants.h"
#include "EC0tracking/Definitions.h"
#include "GPUCommonMath.h"
#include "GPUCommonDef.h"

namespace o2
{
namespace ecl
{

namespace index_table_utils
{
float getInverseZBinSize(const int);
GPUhdi() int getZBinIndex(const int, const float);
GPUhdi() int getPhiBinIndex(const float);
GPUhdi() int getBinIndex(const int, const int);
GPUhdi() int countRowSelectedBins(
  const GPUArrayEC0<int, constants::index_table::ZBins * constants::index_table::PhiBins + 1>&, const int, const int,
  const int);
} // namespace index_table_utils

inline float getInverseZCoordinate(const int layerIndex)
{
  return 0.5f * constants::index_table::ZBins / constants::ecl::LayersZCoordinate()[layerIndex];
}

GPUhdi() int index_table_utils::getZBinIndex(const int layerIndex, const float zCoordinate)
{
  return (zCoordinate + constants::ecl::LayersZCoordinate()[layerIndex]) *
         constants::index_table::InverseZBinSize()[layerIndex];
}

GPUhdi() int index_table_utils::getPhiBinIndex(const float currentPhi)
{
  return (currentPhi * constants::index_table::InversePhiBinSize);
}

GPUhdi() int index_table_utils::getBinIndex(const int zIndex, const int phiIndex)
{
  return gpu::GPUCommonMath::Min(phiIndex * constants::index_table::ZBins + zIndex,
                                 constants::index_table::ZBins * constants::index_table::PhiBins - 1);
}

GPUhdi() int index_table_utils::countRowSelectedBins(
  const GPUArrayEC0<int, constants::index_table::ZBins * constants::index_table::PhiBins + 1>& indexTable,
  const int phiBinIndex, const int minZBinIndex, const int maxZBinIndex)
{
  const int firstBinIndex{getBinIndex(minZBinIndex, phiBinIndex)};
  const int maxBinIndex{firstBinIndex + maxZBinIndex - minZBinIndex + 1};

  return indexTable[maxBinIndex] - indexTable[firstBinIndex];
}
} // namespace ecl
} // namespace o2

#endif /* TRACKINGEC0__INCLUDE_INDEXTABLEUTILS_H_ */
