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

#include "GPUCommonDef.h"
#ifndef __METAL__
#include <array>
#include <utility>
#endif

namespace o2::its::constants
{

GPUglobalconstexpr() float KB = 1024.f;
GPUglobalconstexpr() float MB = KB * KB;
GPUglobalconstexpr() float GB = MB * KB;
GPUglobalconstexpr() bool DoTimeBenchmarks = true;
GPUglobalconstexpr() bool SaveTimeBenchmarks = false;
GPUglobalconstexpr() float Tolerance = 1e-12;                    // numerical tolerance
GPUglobalconstexpr() int ClustersPerCell = 3;                    // number of clusters for a cell
GPUglobalconstexpr() int UnusedIndex = -1;                       // global unused flag
GPUglobalconstexpr() float UnsetValue = -999.f;                  // global unset value
GPUglobalconstexpr() float Radl = 9.36f;                         // Radiation length of Si [cm]
GPUglobalconstexpr() float Rho = 2.33f;                          // Density of Si [g/cm^3]
GPUglobalconstexpr() int MaxIter = 4;                            // Max. supported iterations
GPUglobalconstexpr() int MaxSeedingPasses = 2;                   // seeding + UPC passes optionally appended after the tracking iterations
GPUglobalconstexpr() int MaxSelectedTrackletsPerCluster = 100;   // vertexer: max lines per cluster
GPUglobalconstexpr() int NumberOfConcurrentSeeds = 16;           // default split per worker for the final track fit/extraploation step
GPUglobalconstexpr() int MinNumberOfConcurrentSeeds = (1 << 8);  // minimum chunk size for a worker for the final track fit/extraploation step
GPUglobalconstexpr() int MaxNumberOfConcurrentSeeds = (1 << 12); // maximum chunk size for a worker for the final track fit/extraploation step
GPUglobalconstexpr() float MaxTrackSeedQ2Pt = 1.e3f;             // maximum q/pt for track seeds

GPUglobalconstexpr() int MaxBootstrapPasses = 5;                    // beam bootstrap: cap on the re-trackleting passes
GPUglobalconstexpr() float BeamConvergence2 = 5.e-3f * 5.e-3f;      // beam bootstrap: stop below a (50 um)^2 beam shift
GPUglobalconstexpr() float VtxMinGoodThreshold = 2.f;               // seeding emit: floor of the k*sqrt(ROF load) debris threshold (bounds included)
GPUglobalconstexpr() float VtxOverpopulatedRofTrimFraction = 0.02f; // overpopulated-ROF pruning: busiest fraction of ROFs left out of the per-ROF mean

#ifndef __METAL__
namespace helpers
{

// initialize a std::array at compile time fully with T
template <typename T, std::size_t N, T Value>
constexpr std::array<T, N> initArray()
{
  return []<std::size_t... Is>(std::index_sequence<Is...>) { return std::array<T, N>{(static_cast<void>(Is), Value)...}; }(std::make_index_sequence<N>{});
}

} // namespace helpers
#endif
} // namespace o2::its::constants

#endif /* TRACKINGITSU_INCLUDE_CONSTANTS_H_ */
