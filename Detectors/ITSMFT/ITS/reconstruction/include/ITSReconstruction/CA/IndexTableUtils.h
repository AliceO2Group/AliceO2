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

#include <array>
#include <utility>
#include <vector>

#include "ITSReconstruction/CA/Constants.h"
#include "ITSReconstruction/CA/Definitions.h"

namespace o2
{
namespace ITS
{
namespace CA
{

namespace IndexTableUtils
{
float getInverseZBinSize(const int);
GPU_HOST_DEVICE int getZBinIndex(const int, const float);
GPU_HOST_DEVICE int getPhiBinIndex(const float);
GPU_HOST_DEVICE int getBinIndex(const int, const int);
GPU_HOST_DEVICE int countRowSelectedBins(
  const GPUArray<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1>&, const int, const int,
  const int);
} // namespace IndexTableUtils

inline float getInverseZCoordinate(const int layerIndex)
{
  return 0.5f * Constants::IndexTable::ZBins / Constants::ITS::LayersZCoordinate()[layerIndex];
}

GPU_HOST_DEVICE inline int IndexTableUtils::getZBinIndex(const int layerIndex, const float zCoordinate)
{
  return (zCoordinate + Constants::ITS::LayersZCoordinate()[layerIndex]) *
         Constants::IndexTable::InverseZBinSize()[layerIndex];
}

GPU_HOST_DEVICE inline int IndexTableUtils::getPhiBinIndex(const float currentPhi)
{
  return (currentPhi * Constants::IndexTable::InversePhiBinSize);
}

GPU_HOST_DEVICE inline int IndexTableUtils::getBinIndex(const int zIndex, const int phiIndex)
{
  return MATH_MIN(phiIndex * Constants::IndexTable::PhiBins + zIndex,
                  Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins);
}

GPU_HOST_DEVICE inline int IndexTableUtils::countRowSelectedBins(
  const GPUArray<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1>& indexTable,
  const int phiBinIndex, const int minZBinIndex, const int maxZBinIndex)
{
  const int firstBinIndex{ getBinIndex(minZBinIndex, phiBinIndex) };
  const int maxBinIndex{ firstBinIndex + maxZBinIndex - minZBinIndex + 1 };

  return indexTable[maxBinIndex] - indexTable[firstBinIndex];
}
} // namespace CA
} // namespace ITS
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_INDEXTABLEUTILS_H_ */
