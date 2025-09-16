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

#ifndef ITSTRACKINGGPU_VERTEXINGKERNELS_H_
#define ITSTRACKINGGPU_VERTEXINGKERNELS_H_

#include <cstdint>
#include <gsl/span>
#include <array>
#include "ITStracking/Tracklet.h"
#include "ITStracking/Cluster.h"
#include "ITStracking/ClusterLines.h"
#include "ITStrackingGPU/Utils.h"

namespace o2::its
{

/// Trackleting
template <int32_t nLayers>
void countTrackletsInROFsHandler(const IndexTableUtils<nLayers>* GPUrestrict() utils,
                                 const uint8_t* GPUrestrict() multMask,
                                 const int32_t nRofs,
                                 const int32_t deltaROF,
                                 const int32_t* GPUrestrict() rofPV,
                                 const int32_t vertPerRofThreshold,
                                 const Cluster** GPUrestrict() clusters,
                                 const uint32_t nClusters,
                                 const int32_t** GPUrestrict() ROFClusters,
                                 const uint8_t** GPUrestrict() usedClusters,
                                 const int32_t** GPUrestrict() clustersIndexTables,
                                 int32_t** trackletsPerClusterLUTs,
                                 int32_t** trackletsPerClusterSumLUTs,
                                 int32_t** trackletsPerROF,
                                 const std::array<int32_t*, 2>& trackletsPerClusterLUTsHost,
                                 const std::array<int32_t*, 2>& trackletsPerClusterSumLUTsHost,
                                 const int32_t iteration,
                                 const float phiCut,
                                 const int32_t maxTrackletsPerCluster,
                                 const int32_t nBlocks,
                                 const int32_t nThreads,
                                 gpu::Streams& streams);

template <int32_t nLayers>
void computeTrackletsInROFsHandler(const IndexTableUtils<nLayers>* GPUrestrict() utils,
                                   const uint8_t* GPUrestrict() multMask,
                                   const int32_t nRofs,
                                   const int32_t deltaROF,
                                   const int32_t* GPUrestrict() rofPV,
                                   const int vertPerRofThreshold,
                                   const Cluster** GPUrestrict() clusters,
                                   const uint32_t nClusters,
                                   const int32_t** GPUrestrict() ROFClusters,
                                   const uint8_t** GPUrestrict() usedClusters,
                                   const int32_t** GPUrestrict() clustersIndexTables,
                                   Tracklet** GPUrestrict() foundTracklets,
                                   const int32_t** GPUrestrict() trackletsPerClusterLUTs,
                                   const int32_t** GPUrestrict() trackletsPerClusterSumLUTs,
                                   const int32_t** GPUrestrict() trackletsPerROF,
                                   const int32_t iteration,
                                   const float phiCut,
                                   const int32_t maxTrackletsPerCluster,
                                   const int32_t nBlocks,
                                   const int32_t nThreads,
                                   gpu::Streams& streams);

/// Selection
void countTrackletsMatchingInROFsHandler(const int32_t nRofs,
                                         const int32_t deltaROF,
                                         const uint32_t nClusters,
                                         const int32_t** GPUrestrict() ROFClusters,
                                         const Cluster** GPUrestrict() clusters,
                                         uint8_t** GPUrestrict() usedClusters,
                                         const Tracklet** GPUrestrict() foundTracklets,
                                         uint8_t* GPUrestrict() usedTracklets,
                                         const int32_t** GPUrestrict() trackletsPerClusterLUTs,
                                         const int32_t** GPUrestrict() trackletsPerClusterSumLUTs,
                                         int32_t* GPUrestrict() linesPerClusterLUT,
                                         int32_t* GPUrestrict() linesPerClusterSumLUT,
                                         const int32_t iteration,
                                         const float phiCut,
                                         const float tanLambdaCut,
                                         const int32_t nBlocks,
                                         const int32_t nThreads,
                                         gpu::Streams& streams);

void computeTrackletsMatchingInROFsHandler(const int32_t nRofs,
                                           const int32_t deltaROF,
                                           const uint32_t nClusters,
                                           const int32_t** GPUrestrict() ROFClusters,
                                           const Cluster** GPUrestrict() clusters,
                                           const uint8_t** GPUrestrict() usedClusters,
                                           const Tracklet** GPUrestrict() foundTracklets,
                                           uint8_t* GPUrestrict() usedTracklets,
                                           const int32_t** GPUrestrict() trackletsPerClusterLUTs,
                                           const int32_t** GPUrestrict() trackletsPerClusterSumLUTs,
                                           const int32_t* GPUrestrict() linesPerClusterSumLUT,
                                           Line* GPUrestrict() lines,
                                           const int32_t iteration,
                                           const float phiCut,
                                           const float tanLambdaCut,
                                           const int32_t nBlocks,
                                           const int32_t nThreads,
                                           gpu::Streams& streams);

} // namespace o2::its
#endif
