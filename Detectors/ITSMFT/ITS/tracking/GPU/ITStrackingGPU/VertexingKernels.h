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
#include "ITStracking/MathUtils.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/Tracklet.h"

#include "ITStrackingGPU/Utils.h"
#include "ITStrackingGPU/ClusterLinesGPU.h"
#include "ITStrackingGPU/VertexerTraitsGPU.h"
#include "ITStrackingGPU/TracerGPU.h"

#include "GPUCommonArray.h"

namespace o2::its::gpu
{
#ifdef GPUCA_GPUCODE // GPUg() global kernels must only when compiled by GPU compiler
template <TrackletMode Mode>
GPUg() void trackleterKernelMultipleRof(
  const Cluster* clustersNextLayer,    // 0 2
  const Cluster* clustersCurrentLayer, // 1 1
  const int* sizeNextLClusters,
  const int* sizeCurrentLClusters,
  const int* nextIndexTables,
  Tracklet* Tracklets,
  int* foundTracklets,
  const IndexTableUtils* utils,
  const unsigned int startRofId,
  const unsigned int rofSize,
  const float phiCut,
  const size_t maxTrackletsPerCluster);
#endif
template <TrackletMode Mode>
void trackletFinderHandler(const Cluster* clustersNextLayer,    // 0 2
                           const Cluster* clustersCurrentLayer, // 1 1
                           const int* sizeNextLClusters,
                           const int* sizeCurrentLClusters,
                           const int* nextIndexTables,
                           Tracklet* Tracklets,
                           int* foundTracklets,
                           const IndexTableUtils* utils,
                           const unsigned int startRofId,
                           const unsigned int rofSize,
                           const float phiCut,
                           const size_t maxTrackletsPerCluster = 1e2);
} // namespace o2::its::gpu
#endif