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

#ifndef GPUCA_GPUCODE_DEVICE
#include <array>
#include <utility>
#include <vector>
#endif

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

  GPUhdi() int getNzBins() const { return mNzBins; }
  GPUhdi() int getNphiBins() const { return mNphiBins; }
  GPUhdi() float getLayerZ(int i) const { return mLayerZ[i]; }

 private:
  int mNzBins = 0;
  int mNphiBins = 0;
  float mInversePhiBinSize = 0.f;
  std::vector<float> mLayerZ;
  std::vector<float> mInverseZBinSize;
};

template <class T>
inline void IndexTableUtils::setTrackingParameters(const T& params)
{
  mInversePhiBinSize = params.PhiBins / constants::math::TwoPi;
  mInverseZBinSize.resize(params.LayerZ.size());
  mNzBins = params.ZBins;
  mNphiBins = params.PhiBins;
  mLayerZ = params.LayerZ;
  for (unsigned int iL{0}; iL < mInverseZBinSize.size(); ++iL) {
    mInverseZBinSize[iL] = 0.5f * params.ZBins / params.LayerZ[iL];
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
  return gpu::GPUCommonMath::Min(phiIndex * mNzBins + zIndex, mNzBins * mNphiBins - 1);
}

GPUhdi() int IndexTableUtils::countRowSelectedBins(const int* indexTable, const int phiBinIndex,
                                                   const int minZBinIndex, const int maxZBinIndex) const
{
  const int firstBinIndex{getBinIndex(minZBinIndex, phiBinIndex)};
  const int maxBinIndex{firstBinIndex + maxZBinIndex - minZBinIndex + 1};

  return indexTable[maxBinIndex] - indexTable[firstBinIndex];
}
} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_INDEXTABLEUTILS_H_ */
