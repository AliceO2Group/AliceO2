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
/// \file Constants.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_CONSTANTS_H_
#define TRACKINGITSU_INCLUDE_CONSTANTS_H_

#ifndef __OPENCL__
#include <climits>
#include <vector>
#endif

#include "ITStracking/Definitions.h"

namespace o2
{
namespace its
{

namespace constants
{

constexpr bool DoTimeBenchmarks = true;

namespace math
{
constexpr float Pi{ 3.14159265359f };
constexpr float TwoPi{ 2.0f * Pi };
constexpr float FloatMinThreshold{ 1e-20f };
} // namespace Math

namespace its
{
constexpr int LayersNumber{ 7 };
constexpr int LayersNumberVertexer{ 3 };
constexpr int TrackletsPerRoad{ LayersNumber - 1 };
constexpr int CellsPerRoad{ LayersNumber - 2 };
constexpr int ClustersPerCell{ 3 };
constexpr int UnusedIndex{ -1 };
constexpr float Resolution{ 0.0005f };

GPU_HOST_DEVICE constexpr GPUArray<float, LayersNumber> LayersZCoordinate()
{
  return GPUArray<float, LayersNumber>{ { 16.333f, 16.333f, 16.333f, 42.140f, 42.140f, 73.745f, 73.745f } };
}
GPU_HOST_DEVICE constexpr GPUArray<float, LayersNumber> LayersRCoordinate()
{
  return GPUArray<float, LayersNumber>{ { 2.33959f, 3.14076f, 3.91924f, 19.6213f, 24.5597f, 34.388f, 39.3329f } };
}
} // namespace its

namespace index_table
{
constexpr int ZBins{ 20 };
constexpr int PhiBins{ 20 };
constexpr float InversePhiBinSize{ constants::index_table::PhiBins / constants::math::TwoPi };
GPU_HOST_DEVICE constexpr GPUArray<float, its::LayersNumber> InverseZBinSize()
{
  return GPUArray<float, its::LayersNumber>{ { 0.5 * ZBins / 16.333f, 0.5 * ZBins / 16.333f, 0.5 * ZBins / 16.333f,
                                               0.5 * ZBins / 42.140f, 0.5 * ZBins / 42.140f, 0.5 * ZBins / 73.745f,
                                               0.5 * ZBins / 73.745f } };
}
} // namespace IndexTable

namespace pdgcodes
{
constexpr int PionCode{ 211 };
}
} // namespace Constants
} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_CONSTANTS_H_ */
