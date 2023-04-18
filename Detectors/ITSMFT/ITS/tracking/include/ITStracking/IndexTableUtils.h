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
/// \file IndexTableUtils.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_INDEXTABLEUTILS_H_
#define TRACKINGITSU_INCLUDE_INDEXTABLEUTILS_H_

#include "ITStracking/Constants.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/Definitions.h"
#include "GPUCommonMath.h"
#include "GPUCommonDef.h"

namespace o2
{
namespace its
{
class IndexTableUtils
{
 public:
  template <class T>
  void setTrackingParameters(const T& params);
  float getInverseZCoordinate(const int layerIndex) const;
  GPUhdi() int getZBinIndex(const int, const float) const;
  GPUhdi() int getPhiBinIndex(const float) const;
  GPUhdi() int getBinIndex(const int, const int) const;
  GPUhdi() int countRowSelectedBins(const int*, const int, const int, const int) const;
  GPUhdi() void print() const;

  GPUhdi() int getNzBins() const { return mNzBins; }
  GPUhdi() int getNphiBins() const { return mNphiBins; }
  GPUhdi() float getLayerZ(int i) const { return mLayerZ[i]; }
  GPUhdi() void setNzBins(const int zBins) { mNzBins = zBins; }
  GPUhdi() void setNphiBins(const int phiBins) { mNphiBins = phiBins; }

 private:
  int mNzBins = 0;
  int mNphiBins = 0;
  float mInversePhiBinSize = 0.f;
  float mLayerZ[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  float mInverseZBinSize[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
};

template <class T>
inline void IndexTableUtils::setTrackingParameters(const T& params)
{
  mInversePhiBinSize = params.PhiBins / constants::math::TwoPi;
  mNzBins = params.ZBins;
  mNphiBins = params.PhiBins;
  for (int iLayer{0}; iLayer < params.LayerZ.size(); ++iLayer) {
    mLayerZ[iLayer] = params.LayerZ[iLayer];
  }
  for (unsigned int iLayer{0}; iLayer < params.LayerZ.size(); ++iLayer) {
    mInverseZBinSize[iLayer] = 0.5f * params.ZBins / params.LayerZ[iLayer];
  }
}

inline float IndexTableUtils::getInverseZCoordinate(const int layerIndex) const
{
  return 0.5f * mNzBins / mLayerZ[layerIndex];
}

GPUhdi() int IndexTableUtils::getZBinIndex(const int layerIndex, const float zCoordinate) const
{
  return (zCoordinate + mLayerZ[layerIndex]) * mInverseZBinSize[layerIndex];
}

GPUhdi() int IndexTableUtils::getPhiBinIndex(const float currentPhi) const
{
  return (currentPhi * mInversePhiBinSize);
}

GPUhdi() int IndexTableUtils::getBinIndex(const int zIndex, const int phiIndex) const
{
  return o2::gpu::GPUCommonMath::Min(phiIndex * mNzBins + zIndex, mNzBins * mNphiBins - 1);
}

GPUhdi() int IndexTableUtils::countRowSelectedBins(const int* indexTable, const int phiBinIndex,
                                                   const int minZBinIndex, const int maxZBinIndex) const
{
  const int firstBinIndex{getBinIndex(minZBinIndex, phiBinIndex)};
  const int maxBinIndex{firstBinIndex + maxZBinIndex - minZBinIndex + 1};

  return indexTable[maxBinIndex] - indexTable[firstBinIndex];
}

GPUhdi() void IndexTableUtils::print() const
{
  printf("NzBins: %d, NphiBins: %d, InversePhiBinSize: %f\n", mNzBins, mNphiBins, mInversePhiBinSize);
  for (int iLayer{0}; iLayer < 7; ++iLayer) {
    printf("Layer %d: Z: %f, InverseZBinSize: %f\n", iLayer, mLayerZ[iLayer], mInverseZBinSize[iLayer]);
  }
}
} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_INDEXTABLEUTILS_H_ */
