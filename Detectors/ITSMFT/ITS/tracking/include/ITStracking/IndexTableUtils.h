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

#include <array>

#include "ITStracking/Cluster.h"
#include "ITStracking/MathUtils.h"
#include "CommonConstants/MathConstants.h"
#include "GPUCommonMath.h"
#include "GPUCommonDef.h"

namespace o2::its
{

template <int nLayers>
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
  std::array<float, nLayers> mLayerZ{};
  std::array<float, nLayers> mInverseZBinSize{};
};

template <int nLayers>
template <class T>
inline void IndexTableUtils<nLayers>::setTrackingParameters(const T& params)
{
  mInversePhiBinSize = params.PhiBins / o2::constants::math::TwoPI;
  mNzBins = params.ZBins;
  mNphiBins = params.PhiBins;
  for (int iLayer{0}; iLayer < params.LayerZ.size(); ++iLayer) {
    mLayerZ[iLayer] = params.LayerZ[iLayer];
  }
  for (unsigned int iLayer{0}; iLayer < params.LayerZ.size(); ++iLayer) {
    mInverseZBinSize[iLayer] = 0.5f * params.ZBins / params.LayerZ[iLayer];
  }
}

template <int nLayers>
inline float IndexTableUtils<nLayers>::getInverseZCoordinate(const int layerIndex) const
{
  return 0.5f * mNzBins / mLayerZ[layerIndex];
}

template <int nLayers>
GPUhdi() int IndexTableUtils<nLayers>::getZBinIndex(const int layerIndex, const float zCoordinate) const
{
  return (zCoordinate + mLayerZ[layerIndex]) * mInverseZBinSize[layerIndex];
}

template <int nLayers>
GPUhdi() int IndexTableUtils<nLayers>::getPhiBinIndex(const float currentPhi) const
{
  return (currentPhi * mInversePhiBinSize);
}

template <int nLayers>
GPUhdi() int IndexTableUtils<nLayers>::getBinIndex(const int zIndex, const int phiIndex) const
{
  return o2::gpu::GPUCommonMath::Min(phiIndex * mNzBins + zIndex, (mNzBins * mNphiBins) - 1);
}

template <int nLayers>
GPUhdi() int IndexTableUtils<nLayers>::countRowSelectedBins(const int* indexTable, const int phiBinIndex,
                                                            const int minZBinIndex, const int maxZBinIndex) const
{
  const int firstBinIndex{getBinIndex(minZBinIndex, phiBinIndex)};
  const int maxBinIndex{firstBinIndex + maxZBinIndex - minZBinIndex + 1};

  return indexTable[maxBinIndex] - indexTable[firstBinIndex];
}

template <int nLayers>
GPUhdi() void IndexTableUtils<nLayers>::print() const
{
  printf("NzBins: %d, NphiBins: %d, InversePhiBinSize: %f\n", mNzBins, mNphiBins, mInversePhiBinSize);
  for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
    printf("Layer %d: Z: %f, InverseZBinSize: %f\n", iLayer, mLayerZ[iLayer], mInverseZBinSize[iLayer]);
  }
}

template <int nLayers>
GPUhdi() int4 getBinsRect(const Cluster& currentCluster, const int layerIndex,
                          const float z1, const float z2, const float maxdeltaz, const float maxdeltaphi,
                          const IndexTableUtils<nLayers>& utils)
{
  const float zRangeMin = o2::gpu::GPUCommonMath::Min(z1, z2) - maxdeltaz;
  const float phiRangeMin = (maxdeltaphi > o2::constants::math::PI) ? 0.f : currentCluster.phi - maxdeltaphi;
  const float zRangeMax = o2::gpu::GPUCommonMath::Max(z1, z2) + maxdeltaz;
  const float phiRangeMax = (maxdeltaphi > o2::constants::math::PI) ? o2::constants::math::TwoPI : currentCluster.phi + maxdeltaphi;

  if (zRangeMax < -utils.getLayerZ(layerIndex) ||
      zRangeMin > utils.getLayerZ(layerIndex) || zRangeMin > zRangeMax) {
    return int4{-1, -1, -1, -1};
  }

  return int4{o2::gpu::GPUCommonMath::Max(0, utils.getZBinIndex(layerIndex, zRangeMin)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMin)),
              o2::gpu::GPUCommonMath::Min(utils.getNzBins() - 1, utils.getZBinIndex(layerIndex, zRangeMax)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMax))};
}

} // namespace o2::its
#endif /* TRACKINGITSU_INCLUDE_INDEXTABLEUTILS_H_ */
