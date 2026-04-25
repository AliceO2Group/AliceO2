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

#include <array>
#include <utility>

namespace o2::its::constants
{

constexpr float KB = 1024.f;
constexpr float MB = KB * KB;
constexpr float GB = MB * KB;
constexpr bool DoTimeBenchmarks = true;
constexpr bool SaveTimeBenchmarks = false;
constexpr float Tolerance = 1e-12;                  // numerical tolerance
constexpr int ClustersPerCell = 3;                  // number of clusters for a cell
constexpr int UnusedIndex = -1;                     // global unused flag
constexpr float UnsetValue = -999.f;                // global unset value
constexpr float Radl = 9.36f;                       // Radiation length of Si [cm]
constexpr float Rho = 2.33f;                        // Density of Si [g/cm^3]
constexpr int MaxIter = 4;                          // Max. supported iterations
constexpr int MaxSelectedTrackletsPerCluster = 100; // vertexer: max lines per cluster

namespace helpers
{

// initialize a std::array at compile time fully with T
template <typename T, std::size_t N, T Value>
constexpr std::array<T, N> initArray()
{
  return []<std::size_t... Is>(std::index_sequence<Is...>) { return std::array<T, N>{(static_cast<void>(Is), Value)...}; }(std::make_index_sequence<N>{});
}

} // namespace helpers
} // namespace o2::its::constants

#endif /* TRACKINGITSU_INCLUDE_CONSTANTS_H_ */
