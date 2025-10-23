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

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "ITStrackingGPU/VertexingKernels.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/ClusterLines.h"

#include "GPUCommonMath.h"
#include "GPUCommonHelpers.h"
#include "GPUCommonDef.h"

namespace o2::its
{

namespace gpu
{

template <int nLayers, TrackletMode Mode, bool dryRun>
GPUg() void computeLayerTrackletMutliROFKernel(const Cluster** GPUrestrict() clusters,
                                               const int32_t** GPUrestrict() rofClusters,
                                               const uint8_t** GPUrestrict() usedClusters,
                                               const int32_t** GPUrestrict() clusterIndexTables,
                                               const float phiCut,
                                               maybe_const<dryRun, Tracklet>** GPUrestrict() tracklets,
                                               maybe_const<!dryRun, int32_t>** GPUrestrict() trackletOffsets,
                                               const IndexTableUtils<nLayers>* GPUrestrict() utils,
                                               const int32_t nRofs,
                                               const int32_t deltaRof,
                                               const int32_t* GPUrestrict() rofPV,
                                               const int32_t iteration,
                                               const int32_t verPerRofThreshold,
                                               const int32_t maxTrackletsPerCluster)
{
  constexpr int32_t iMode = (Mode == TrackletMode::Layer0Layer1) ? 0 : 1;
  const int32_t phiBins(utils->getNphiBins());
  const int32_t zBins(utils->getNzBins());
  const int32_t tableSize{phiBins * zBins + 1};
  extern __shared__ uint16_t storedTrackletsShared[]; // each deltaROF needs its own counters
  uint16_t* storedTrackletsLocal = storedTrackletsShared + threadIdx.x * (2 * deltaRof + 1);
  for (uint32_t pivotRofId{blockIdx.x}; pivotRofId < (uint32_t)nRofs; pivotRofId += gridDim.x) {
    if (iteration && rofPV[pivotRofId] > verPerRofThreshold) {
      continue;
    }
    const uint16_t startROF = o2::gpu::CAMath::Max(0, (int)pivotRofId - deltaRof);
    const uint16_t endROF = o2::gpu::CAMath::Min(nRofs, (int)pivotRofId + deltaRof + 1);
    const auto clustersCurrentLayer = getClustersOnLayer((int32_t)pivotRofId, nRofs, 1, rofClusters, clusters);
    if (clustersCurrentLayer.empty()) {
      continue;
    }
    auto trackletsPerCluster = getNTrackletsPerCluster(pivotRofId, nRofs, iMode, rofClusters, trackletOffsets);
    for (uint32_t iCurrentLayerClusterIndex{threadIdx.x}; iCurrentLayerClusterIndex < (uint32_t)clustersCurrentLayer.size(); iCurrentLayerClusterIndex += blockDim.x) {
      for (int16_t i{0}; i < (int16_t)((2 * deltaRof) + 1); ++i) {
        storedTrackletsLocal[i] = 0;
      }
      const Cluster& GPUrestrict() currentCluster { clustersCurrentLayer[iCurrentLayerClusterIndex] };
      const int4 selectedBinsRect{getBinsRect(currentCluster, (int)Mode, utils, 0.f, 0.f, 50.f, phiCut / 2)};
      if (selectedBinsRect.x != 0 || selectedBinsRect.y != 0 || selectedBinsRect.z != 0 || selectedBinsRect.w != 0) {
        int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};
        if (phiBinsNum < 0) {
          phiBinsNum += phiBins;
        }
        for (int32_t iPhiBin{selectedBinsRect.y}, iPhiCount{0}; iPhiCount < phiBinsNum; iPhiBin = ++iPhiBin == phiBins ? 0 : iPhiBin, iPhiCount++) {
          for (uint16_t targetRofId{startROF}; targetRofId < endROF; ++targetRofId) {
            uint16_t& storedTracklets = storedTrackletsLocal[pivotRofId - targetRofId + deltaRof];
            const int32_t firstBinIndex{utils->getBinIndex(selectedBinsRect.x, iPhiBin)};
            const int32_t maxBinIndex{firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1};
            const int32_t firstRowClusterIndex{clusterIndexTables[(int)Mode][(targetRofId)*tableSize + firstBinIndex]};
            const int32_t maxRowClusterIndex{clusterIndexTables[(int)Mode][(targetRofId)*tableSize + maxBinIndex]};
            auto clustersNextLayer = getClustersOnLayer((int32_t)targetRofId, nRofs, (int32_t)Mode, rofClusters, clusters);
            if (clustersNextLayer.empty()) {
              continue;
            }
            for (int32_t iNextLayerClusterIndex{firstRowClusterIndex}; iNextLayerClusterIndex < maxRowClusterIndex && iNextLayerClusterIndex < (int32_t)clustersNextLayer.size(); ++iNextLayerClusterIndex) {
              if (iteration && usedClusters[(int32_t)Mode][iNextLayerClusterIndex]) {
                continue;
              }
              const Cluster& GPUrestrict() nextCluster { clustersNextLayer[iNextLayerClusterIndex] };
              if (o2::gpu::GPUCommonMath::Abs(math_utils::smallestAngleDifference(currentCluster.phi, nextCluster.phi)) < phiCut) {
                if (storedTracklets < maxTrackletsPerCluster) {
                  if constexpr (!dryRun) {
                    if constexpr (Mode == TrackletMode::Layer0Layer1) {
                      tracklets[0][trackletsPerCluster[iCurrentLayerClusterIndex] + storedTracklets] = Tracklet{iNextLayerClusterIndex, (int)iCurrentLayerClusterIndex, nextCluster, currentCluster, (short)targetRofId, (short)pivotRofId};
                    } else {
                      tracklets[1][trackletsPerCluster[iCurrentLayerClusterIndex] + storedTracklets] = Tracklet{(int)iCurrentLayerClusterIndex, iNextLayerClusterIndex, currentCluster, nextCluster, (short)pivotRofId, (short)targetRofId};
                    }
                  }
                  ++storedTracklets;
                }
              }
            }
          }
        }
      }
      if constexpr (dryRun) {
        for (int32_t i{0}; i < (int32_t)((2 * deltaRof) + 1); ++i) {
          trackletsPerCluster[iCurrentLayerClusterIndex] += storedTrackletsLocal[i];
        }
      }
    }
  }
}

template <bool dryRun>
GPUg() void computeTrackletSelectionMutliROFKernel(const Cluster** GPUrestrict() clusters,
                                                   maybe_const<!dryRun, uint8_t>** GPUrestrict() usedClusters,
                                                   const int32_t** GPUrestrict() rofClusters,
                                                   const float phiCut,
                                                   const float tanLambdaCut,
                                                   const Tracklet** GPUrestrict() tracklets,
                                                   uint8_t* GPUrestrict() usedTracklets,
                                                   const int32_t** GPUrestrict() trackletOffsets,
                                                   const int32_t** GPUrestrict() trackletLUTs,
                                                   maybe_const<!dryRun, int32_t>* lineOffsets,
                                                   maybe_const<dryRun, Line>* GPUrestrict() lines,
                                                   const int32_t nRofs,
                                                   const int32_t deltaRof,
                                                   const int32_t maxTracklets)
{
  for (uint32_t pivotRofId{blockIdx.x}; pivotRofId < nRofs; pivotRofId += gridDim.x) {
    const int16_t startROF = o2::gpu::CAMath::Max(0, (int32_t)pivotRofId - deltaRof);
    const int16_t endROF = o2::gpu::CAMath::Min(nRofs, (int32_t)pivotRofId + deltaRof + 1);

    const uint32_t clusterOffset = rofClusters[1][pivotRofId];
    const uint32_t nClustersCurrentLayer = rofClusters[1][pivotRofId + 1] - clusterOffset;
    if (nClustersCurrentLayer <= 0) {
      continue;
    }

    auto linesPerCluster = getNLinesPerCluster(pivotRofId, nRofs, rofClusters, lineOffsets);
    auto nTrackletsPerCluster01 = getNTrackletsPerCluster(pivotRofId, nRofs, 0, rofClusters, trackletOffsets);
    auto nTrackletsPerCluster12 = getNTrackletsPerCluster(pivotRofId, nRofs, 1, rofClusters, trackletOffsets);

    for (uint32_t iCurrentLayerClusterIndex{threadIdx.x}; iCurrentLayerClusterIndex < nClustersCurrentLayer; iCurrentLayerClusterIndex += blockDim.x) {
      int32_t validTracklets{0};
      const int32_t nTracklets01 = nTrackletsPerCluster01[iCurrentLayerClusterIndex];
      const int32_t nTracklets12 = nTrackletsPerCluster12[iCurrentLayerClusterIndex];
      for (int32_t iTracklet12{0}; iTracklet12 < nTracklets12; ++iTracklet12) {
        for (int32_t iTracklet01{0}; iTracklet01 < nTracklets01; ++iTracklet01) {

          if (usedTracklets[trackletLUTs[0][clusterOffset + iCurrentLayerClusterIndex] + iTracklet01]) {
            continue;
          }

          const auto& GPUrestrict() tracklet01 { tracklets[0][trackletLUTs[0][clusterOffset + iCurrentLayerClusterIndex] + iTracklet01] };
          const auto& GPUrestrict() tracklet12 { tracklets[1][trackletLUTs[1][clusterOffset + iCurrentLayerClusterIndex] + iTracklet12] };
          const int16_t rof0 = tracklet01.rof[0];
          const int16_t rof2 = tracklet12.rof[1];
          if (deltaRof > 0 && ((rof0 < startROF) || (rof0 >= endROF) || (rof2 < startROF) || (rof2 >= endROF) || (o2::gpu::CAMath::Abs(rof0 - rof2) > deltaRof))) {
            continue;
          }

          const float deltaTanLambda{o2::gpu::GPUCommonMath::Abs(tracklet01.tanLambda - tracklet12.tanLambda)};
          const float deltaPhi{o2::gpu::GPUCommonMath::Abs(math_utils::smallestAngleDifference(tracklet01.phi, tracklet12.phi))};
          //
          if (deltaTanLambda < tanLambdaCut && deltaPhi < phiCut && validTracklets < maxTracklets) {
            // TODO use atomics to avoid race conditions for torn writes but is it needed here?
            usedTracklets[trackletLUTs[0][clusterOffset + iCurrentLayerClusterIndex] + iTracklet01] = 1;
            if constexpr (dryRun) {
              usedClusters[0][rofClusters[0][rof0] + tracklet01.firstClusterIndex] = 1;
              usedClusters[2][rofClusters[2][rof2] + tracklet12.secondClusterIndex] = 1;
            } else {
              const Cluster* clusters0 = clusters[0] + rofClusters[0][tracklet01.rof[0]];
              const Cluster* clusters1 = clusters[1] + rofClusters[1][tracklet01.rof[1]];
              lines[lineOffsets[iCurrentLayerClusterIndex] + validTracklets] = Line(tracklet01, clusters0, clusters1);
            }
            ++validTracklets;
          }
        }
      }

      if constexpr (dryRun) {
        linesPerCluster[iCurrentLayerClusterIndex] = validTracklets;
      }
    }
  }
}

template <TrackletMode Mode>
GPUg() void compileTrackletsPerROFKernel(const int32_t nRofs,
                                         int** GPUrestrict() nTrackletsPerROF,
                                         const int32_t** GPUrestrict() rofClusters,
                                         const int32_t** GPUrestrict() nTrackletsPerCluster)
{
  // TODO is this the best reduction kernel?
  constexpr int32_t iMode = (Mode == TrackletMode::Layer0Layer1) ? 0 : 1;
  extern __shared__ int32_t ssum[];
  for (uint32_t rof = blockIdx.x; rof < (uint32_t)nRofs; rof += gridDim.x) {
    const auto& GPUrestrict() currentNTracklets = getNTrackletsPerCluster(rof, nRofs, iMode, rofClusters, nTrackletsPerCluster);
    int32_t localSum = 0;
    for (uint32_t ci = threadIdx.x; ci < (uint32_t)currentNTracklets.size(); ci += blockDim.x) {
      localSum += currentNTracklets[ci];
    }
    ssum[threadIdx.x] = localSum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        ssum[threadIdx.x] += ssum[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      nTrackletsPerROF[iMode][rof] = ssum[0];
    }
  }
}

template <typename T>
GPUhi() void cubExclusiveScan(const T* GPUrestrict() in, T* GPUrestrict() out, int32_t num_items, cudaStream_t stream)
{
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  GPUChkErrS(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, in, out + 1, num_items, stream));
  GPUChkErrS(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
  GPUChkErrS(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, in, out + 1, num_items, stream));
  GPUChkErrS(cudaFreeAsync(d_temp_storage, stream));
}

} // namespace gpu

template <int nLayers>
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
                                 int32_t** GPUrestrict() trackletsPerClusterLUTs,
                                 int32_t** GPUrestrict() trackletsPerClusterSumLUTs,
                                 int32_t** GPUrestrict() trackletsPerROF,
                                 const std::array<int32_t*, 2>& trackletsPerClusterLUTsHost,
                                 const std::array<int32_t*, 2>& trackletsPerClusterSumLUTsHost,
                                 const int32_t iteration,
                                 const float phiCut,
                                 const int32_t maxTrackletsPerCluster,
                                 const int32_t nBlocks,
                                 const int32_t nThreads,
                                 gpu::Streams& streams)
{
  const uint32_t sharedBytes = nThreads * (2 * deltaROF + 1) * sizeof(uint16_t);
  gpu::computeLayerTrackletMutliROFKernel<nLayers, TrackletMode::Layer0Layer1, true><<<nBlocks, nThreads, sharedBytes, streams[0].get()>>>(clusters,
                                                                                                                                           ROFClusters,
                                                                                                                                           usedClusters,
                                                                                                                                           clustersIndexTables,
                                                                                                                                           phiCut,
                                                                                                                                           nullptr,
                                                                                                                                           trackletsPerClusterLUTs,
                                                                                                                                           utils,
                                                                                                                                           nRofs,
                                                                                                                                           deltaROF,
                                                                                                                                           rofPV,
                                                                                                                                           iteration,
                                                                                                                                           vertPerRofThreshold,
                                                                                                                                           maxTrackletsPerCluster);
  gpu::compileTrackletsPerROFKernel<TrackletMode::Layer0Layer1><<<nBlocks, nThreads, nThreads * sizeof(int32_t), streams[0].get()>>>(nRofs, trackletsPerROF, ROFClusters, (const int32_t**)trackletsPerClusterLUTs);
  gpu::cubExclusiveScan(trackletsPerClusterLUTsHost[0], trackletsPerClusterSumLUTsHost[0], nClusters, streams[0].get());

  gpu::computeLayerTrackletMutliROFKernel<nLayers, TrackletMode::Layer1Layer2, true><<<nBlocks, nThreads, sharedBytes, streams[1].get()>>>(clusters,
                                                                                                                                           ROFClusters,
                                                                                                                                           usedClusters,
                                                                                                                                           clustersIndexTables,
                                                                                                                                           phiCut,
                                                                                                                                           nullptr,
                                                                                                                                           trackletsPerClusterLUTs,
                                                                                                                                           utils,
                                                                                                                                           nRofs,
                                                                                                                                           deltaROF,
                                                                                                                                           rofPV,
                                                                                                                                           iteration,
                                                                                                                                           vertPerRofThreshold,
                                                                                                                                           maxTrackletsPerCluster);
  gpu::compileTrackletsPerROFKernel<TrackletMode::Layer1Layer2><<<nBlocks, nThreads, nThreads * sizeof(int), streams[1].get()>>>(nRofs, trackletsPerROF, ROFClusters, (const int**)trackletsPerClusterLUTs);
  gpu::cubExclusiveScan(trackletsPerClusterLUTsHost[1], trackletsPerClusterSumLUTsHost[1], nClusters, streams[1].get());
}

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
                                   gpu::Streams& streams)
{
  const uint32_t sharedBytes = nThreads * (2 * deltaROF + 1) * sizeof(uint16_t);
  gpu::computeLayerTrackletMutliROFKernel<nLayers, TrackletMode::Layer0Layer1, false><<<nBlocks, nThreads, sharedBytes, streams[0].get()>>>(clusters,
                                                                                                                                            ROFClusters,
                                                                                                                                            usedClusters,
                                                                                                                                            clustersIndexTables,
                                                                                                                                            phiCut,
                                                                                                                                            foundTracklets,
                                                                                                                                            trackletsPerClusterSumLUTs,
                                                                                                                                            utils,
                                                                                                                                            nRofs,
                                                                                                                                            deltaROF,
                                                                                                                                            rofPV,
                                                                                                                                            iteration,
                                                                                                                                            vertPerRofThreshold,
                                                                                                                                            maxTrackletsPerCluster);
  gpu::computeLayerTrackletMutliROFKernel<nLayers, TrackletMode::Layer1Layer2, false><<<nBlocks, nThreads, sharedBytes, streams[1].get()>>>(clusters,
                                                                                                                                            ROFClusters,
                                                                                                                                            usedClusters,
                                                                                                                                            clustersIndexTables,
                                                                                                                                            phiCut,
                                                                                                                                            foundTracklets,
                                                                                                                                            trackletsPerClusterSumLUTs,
                                                                                                                                            utils,
                                                                                                                                            nRofs,
                                                                                                                                            deltaROF,
                                                                                                                                            rofPV,
                                                                                                                                            iteration,
                                                                                                                                            vertPerRofThreshold,
                                                                                                                                            maxTrackletsPerCluster);
}

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
                                         gpu::Streams& streams)
{
  streams[1].sync(); // need to make sure that all tracklets are done, since this placed in 0 tracklet01 will be done but tracklet12 needs to be guaranteed
  gpu::computeTrackletSelectionMutliROFKernel<true><<<nBlocks, nThreads, 0, streams[0].get()>>>(nullptr,
                                                                                                usedClusters,
                                                                                                ROFClusters,
                                                                                                phiCut,
                                                                                                tanLambdaCut,
                                                                                                foundTracklets,
                                                                                                usedTracklets,
                                                                                                trackletsPerClusterLUTs,
                                                                                                trackletsPerClusterSumLUTs,
                                                                                                linesPerClusterLUT,
                                                                                                nullptr,
                                                                                                nRofs,
                                                                                                deltaROF,
                                                                                                100);
  gpu::cubExclusiveScan(linesPerClusterLUT, linesPerClusterSumLUT, nClusters, streams[0].get());
}

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
                                           gpu::Streams& streams)
{
  gpu::computeTrackletSelectionMutliROFKernel<false><<<nBlocks, nThreads, 0, streams[0].get()>>>(clusters,
                                                                                                 nullptr,
                                                                                                 ROFClusters,
                                                                                                 phiCut,
                                                                                                 tanLambdaCut,
                                                                                                 foundTracklets,
                                                                                                 usedTracklets,
                                                                                                 trackletsPerClusterLUTs,
                                                                                                 trackletsPerClusterSumLUTs,
                                                                                                 linesPerClusterSumLUT,
                                                                                                 lines,
                                                                                                 nRofs,
                                                                                                 deltaROF,
                                                                                                 100);
}

/// Explicit instantiation of ITS2 handlers
template void countTrackletsInROFsHandler<7>(const IndexTableUtils<7>* GPUrestrict() utils,
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

template void computeTrackletsInROFsHandler<7>(const IndexTableUtils<7>* GPUrestrict() utils,
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
/*
GPUg() void lineClustererMultipleRof(
  const int* sizeClustersL1,     // Number of clusters on layer 1 per ROF
  Line* lines,                   // Lines
  int* nFoundLines,              // Number of found lines
  int* nExclusiveFoundLines,     // Number of found lines exclusive scan
  int* clusteredLines,           // Clustered lines
  const unsigned int startRofId, // Starting ROF ID
  const unsigned int rofSize,    // Number of ROFs to consider // Number of found lines exclusive scan
  const float pairCut)           // Selection on line pairs
{
  for (unsigned int iRof{threadIdx.x}; iRof < rofSize; iRof += blockDim.x) {
    auto rof = iRof + startRofId;
    auto clustersL1offsetRof = sizeClustersL1[rof] - sizeClustersL1[startRofId]; // starting cluster offset for this ROF
    auto nClustersL1Rof = sizeClustersL1[rof + 1] - sizeClustersL1[rof];         // number of clusters for this ROF
    auto linesOffsetRof = nExclusiveFoundLines[clustersL1offsetRof];             // starting line offset for this ROF
    // auto* foundLinesRof = nFoundLines + clustersL1offsetRof;
    auto nLinesRof = nExclusiveFoundLines[clustersL1offsetRof + nClustersL1Rof] - linesOffsetRof;
    // printf("rof: %d -> %d lines.\n", rof, nLinesRof);
    for (int iLine1 = 0; iLine1 < nLinesRof; ++iLine1) {
      auto absLine1Index = nExclusiveFoundLines[clustersL1offsetRof] + iLine1;
      if (clusteredLines[absLine1Index] > -1) {
        continue;
      }
      for (int iLine2 = iLine1 + 1; iLine2 < nLinesRof; ++iLine2) {
        auto absLine2Index = nExclusiveFoundLines[clustersL1offsetRof] + iLine2;
        if (clusteredLines[absLine2Index] > -1) {
          continue;
        }

        if (Line::getDCA(lines[absLine1Index], lines[absLine2Index]) < pairCut) {
          ClusterLinesGPU tmpClus{lines[absLine1Index], lines[absLine2Index]};
          float tmpVertex[3];
          tmpVertex[0] = tmpClus.getVertex()[0];
          tmpVertex[1] = tmpClus.getVertex()[1];
          tmpVertex[2] = tmpClus.getVertex()[2];
          if (tmpVertex[0] * tmpVertex[0] + tmpVertex[1] * tmpVertex[1] > 4.f) { // outside the beampipe, skip it
            break;
          }
          clusteredLines[absLine1Index] = iLine1; // We set local index of first line to contribute, so we can retrieve the cluster later
          clusteredLines[absLine2Index] = iLine1;
          for (int iLine3 = 0; iLine3 < nLinesRof; ++iLine3) {
            auto absLine3Index = nExclusiveFoundLines[clustersL1offsetRof] + iLine3;
            if (clusteredLines[absLine3Index] > -1) {
              continue;
            }
            if (Line::getDistanceFromPoint(lines[absLine3Index], tmpVertex) < pairCut) {
              clusteredLines[absLine3Index] = iLine1;
            }
          }
          break;
        }
      }
    }
  } // rof loop
}

GPUg() void computeCentroidsKernel(
  Line* lines,
  int* nFoundLines,
  int* nExclusiveFoundLines,
  const unsigned int nClustersMiddleLayer,
  float* centroids,
  const float lowHistX,
  const float highHistX,
  const float lowHistY,
  const float highHistY,
  const float pairCut)
{
  const int nLines = nExclusiveFoundLines[nClustersMiddleLayer - 1] + nFoundLines[nClustersMiddleLayer - 1];
  const int maxIterations{nLines * (nLines - 1) / 2};
  for (unsigned int currentThreadIndex = blockIdx.x * blockDim.x + threadIdx.x; currentThreadIndex < maxIterations; currentThreadIndex += blockDim.x * gridDim.x) {
    int iFirstLine = currentThreadIndex / nLines;
    int iSecondLine = currentThreadIndex % nLines;
    // All unique pairs
    if (iSecondLine <= iFirstLine) {
      iFirstLine = nLines - iFirstLine - 2;
      iSecondLine = nLines - iSecondLine - 1;
    }
    if (Line::getDCA(lines[iFirstLine], lines[iSecondLine]) < pairCut) {
      ClusterLinesGPU cluster{lines[iFirstLine], lines[iSecondLine]};
      if (cluster.getVertex()[0] * cluster.getVertex()[0] + cluster.getVertex()[1] * cluster.getVertex()[1] < 1.98f * 1.98f) {
        // printOnThread(0, "xCentr: %f, yCentr: %f \n", cluster.getVertex()[0], cluster.getVertex()[1]);
        centroids[2 * currentThreadIndex] = cluster.getVertex()[0];
        centroids[2 * currentThreadIndex + 1] = cluster.getVertex()[1];
      } else {
        // write values outside the histogram boundaries,
        // default behaviour is not to have them added to histogram later
        // (writing zeroes would be problematic)
        centroids[2 * currentThreadIndex] = 2 * lowHistX;
        centroids[2 * currentThreadIndex + 1] = 2 * lowHistY;
      }
    } else {
      // write values outside the histogram boundaries,
      // default behaviour is not to have them added to histogram later
      // (writing zeroes would be problematic)
      centroids[2 * currentThreadIndex] = 2 * highHistX;
      centroids[2 * currentThreadIndex + 1] = 2 * highHistY;
    }
  }
}

GPUg() void computeZCentroidsKernel(
  const int nLines,
  const cub::KeyValuePair<int, int>* tmpVtX,
  float* beamPosition,
  Line* lines,
  float* centroids,
  const int* histX, // X
  const float lowHistX,
  const float binSizeHistX,
  const int nBinsHistX,
  const int* histY, // Y
  const float lowHistY,
  const float binSizeHistY,
  const int nBinsHistY,
  const float lowHistZ, // Z
  const float pairCut,
  const int binOpeningX,
  const int binOpeningY)
{
  for (unsigned int currentThreadIndex = blockIdx.x * blockDim.x + threadIdx.x; currentThreadIndex < nLines; currentThreadIndex += blockDim.x * gridDim.x) {
    if (tmpVtX[0].value || tmpVtX[1].value) {
      float tmpX{lowHistX + tmpVtX[0].key * binSizeHistX + binSizeHistX / 2};
      int sumWX{tmpVtX[0].value};
      float wX{tmpX * tmpVtX[0].value};
      for (int iBin{o2::gpu::GPUCommonMath::Max(0, tmpVtX[0].key - binOpeningX)}; iBin < o2::gpu::GPUCommonMath::Min(tmpVtX[0].key + binOpeningX + 1, nBinsHistX - 1); ++iBin) {
        if (iBin != tmpVtX[0].key) {
          wX += (lowHistX + iBin * binSizeHistX + binSizeHistX / 2) * histX[iBin];
          sumWX += histX[iBin];
        }
      }
      float tmpY{lowHistY + tmpVtX[1].key * binSizeHistY + binSizeHistY / 2};
      int sumWY{tmpVtX[1].value};
      float wY{tmpY * tmpVtX[1].value};
      for (int iBin{o2::gpu::GPUCommonMath::Max(0, tmpVtX[1].key - binOpeningY)}; iBin < o2::gpu::GPUCommonMath::Min(tmpVtX[1].key + binOpeningY + 1, nBinsHistY - 1); ++iBin) {
        if (iBin != tmpVtX[1].key) {
          wY += (lowHistY + iBin * binSizeHistY + binSizeHistY / 2) * histY[iBin];
          sumWY += histY[iBin];
        }
      }
      beamPosition[0] = wX / sumWX;
      beamPosition[1] = wY / sumWY;
      float mockBeamPoint1[3] = {beamPosition[0], beamPosition[1], -1}; // get two points laying at different z, to create line object
      float mockBeamPoint2[3] = {beamPosition[0], beamPosition[1], 1};
      Line pseudoBeam = {mockBeamPoint1, mockBeamPoint2};
      if (Line::getDCA(lines[currentThreadIndex], pseudoBeam) < pairCut) {
        ClusterLinesGPU cluster{lines[currentThreadIndex], pseudoBeam};
        centroids[currentThreadIndex] = cluster.getVertex()[2];
      } else {
        centroids[currentThreadIndex] = 2 * lowHistZ;
      }
    }
  }
}

GPUg() void computeVertexKernel(
  cub::KeyValuePair<int, int>* tmpVertexBins,
  int* histZ, // Z
  const float lowHistZ,
  const float binSizeHistZ,
  const int nBinsHistZ,
  Vertex* vertices,
  float* beamPosition,
  const int vertIndex,
  const int minContributors,
  const int binOpeningZ)
{
  for (unsigned int currentThreadIndex = blockIdx.x * blockDim.x + threadIdx.x; currentThreadIndex < binOpeningZ; currentThreadIndex += blockDim.x * gridDim.x) {
    if (currentThreadIndex == 0) {
      if (tmpVertexBins[2].value > 1 && (tmpVertexBins[0].value || tmpVertexBins[1].value)) {
        float z{lowHistZ + tmpVertexBins[2].key * binSizeHistZ + binSizeHistZ / 2};
        float ex{0.f};
        float ey{0.f};
        float ez{0.f};
        int sumWZ{tmpVertexBins[2].value};
        float wZ{z * tmpVertexBins[2].value};
        for (int iBin{o2::gpu::GPUCommonMath::Max(0, tmpVertexBins[2].key - binOpeningZ)}; iBin < o2::gpu::GPUCommonMath::Min(tmpVertexBins[2].key + binOpeningZ + 1, nBinsHistZ - 1); ++iBin) {
          if (iBin != tmpVertexBins[2].key) {
            wZ += (lowHistZ + iBin * binSizeHistZ + binSizeHistZ / 2) * histZ[iBin];
            sumWZ += histZ[iBin];
          }
          histZ[iBin] = 0;
        }
        if (sumWZ > minContributors || vertIndex == 0) {
          new (vertices + vertIndex) Vertex{o2::math_utils::Point3D<float>(beamPosition[0], beamPosition[1], wZ / sumWZ), std::array<float, 6>{ex, 0, ey, 0, 0, ez}, static_cast<ushort>(sumWZ), 0};
        } else {
          new (vertices + vertIndex) Vertex{};
        }
      } else {
        new (vertices + vertIndex) Vertex{};
      }
    }
  }
}
*/
} // namespace o2::its
