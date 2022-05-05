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
/// \file Constants.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_CONSTANTS_H_
#define TRACKINGITSU_INCLUDE_CONSTANTS_H_

#ifndef GPUCA_GPUCODE_DEVICE
#include <climits>
#include <vector>
#endif

#include "ITStracking/Definitions.h"
#include "CommonConstants/MathConstants.h"
#include "GPUCommonMath.h"
#include "GPUCommonDef.h"

namespace o2
{
namespace its
{

namespace constants
{
constexpr float MB = 1024.f * 1024.f;
constexpr float GB = 1024.f * 1024.f * 1024.f;
constexpr bool DoTimeBenchmarks = true;

namespace math
{
constexpr float Pi{3.14159265359f};
constexpr float TwoPi{2.0f * Pi};
constexpr float FloatMinThreshold{1e-20f};
} // namespace math

namespace its
{
constexpr int LayersNumberVertexer{3};
constexpr int ClustersPerCell{3};
constexpr int UnusedIndex{-1};
constexpr float Resolution{0.0005f};

GPUhdi() constexpr GPUArray<float, 3> VertexerHistogramVolume()
{
  return GPUArray<float, 3>{{1.98, 1.98, 40.f}};
}
} // namespace its

namespace its2
{
constexpr int LayersNumber{7};
constexpr int TrackletsPerRoad{LayersNumber - 1};
constexpr int CellsPerRoad{LayersNumber - 2};

GPUhdi() constexpr GPUArray<float, LayersNumber> LayersZCoordinate()
{
  constexpr double s = 1.; // safety margin
  return GPUArray<float, LayersNumber>{{16.333f + s, 16.333f + s, 16.333f + s, 42.140f + s, 42.140f + s, 73.745f + s, 73.745f + s}};
}
GPUhdi() constexpr GPUArray<float, LayersNumber> LayersRCoordinate()
{
  return GPUArray<float, LayersNumber>{{2.33959f, 3.14076f, 3.91924f, 19.6213f, 24.5597f, 34.388f, 39.3329f}};
}

constexpr int ZBins{256};
constexpr int PhiBins{128};
constexpr float InversePhiBinSize{PhiBins / constants::math::TwoPi};
GPUhdi() constexpr GPUArray<float, LayersNumber> InverseZBinSize()
{
  constexpr auto zSize = LayersZCoordinate();
  return GPUArray<float, LayersNumber>{{0.5f * ZBins / (zSize[0]), 0.5f * ZBins / (zSize[1]), 0.5f * ZBins / (zSize[2]),
                                        0.5f * ZBins / (zSize[3]), 0.5f * ZBins / (zSize[4]), 0.5f * ZBins / (zSize[5]),
                                        0.5f * ZBins / (zSize[6])}};
}
inline float getInverseZCoordinate(const int layerIndex)
{
  return 0.5f * ZBins / LayersZCoordinate()[layerIndex];
}

GPUhdi() int getZBinIndex(const int layerIndex, const float zCoordinate)
{
  return (zCoordinate + LayersZCoordinate()[layerIndex]) *
         InverseZBinSize()[layerIndex];
}

GPUhdi() int getPhiBinIndex(const float currentPhi)
{
  return (currentPhi * InversePhiBinSize);
}

GPUhdi() int getBinIndex(const int zIndex, const int phiIndex)
{
  return o2::gpu::GPUCommonMath::Min(phiIndex * ZBins + zIndex,
                                     ZBins * PhiBins - 1);
}

GPUhdi() constexpr int4 getEmptyBinsRect() { return int4{0, 0, 0, 0}; }

} // namespace its2

namespace pdgcodes
{
constexpr int PionCode{211};
}
} // namespace constants
#ifndef __OPENCL__ /// FIXME: this is for compatibility with OCL
typedef std::vector<std::vector<int>> index_table_t;
#endif
} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_CONSTANTS_H_ */
