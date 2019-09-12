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
/// \file VertexerTraitsGPU.cu.cu
/// \brief
/// \author matteo.concas@cern.ch

#include <iostream>
#include <sstream>
#include <array>
#include <assert.h>

#include "ITStracking/MathUtils.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/ClusterLines.h"

#include "ITStrackingCUDA/Utils.h"
#include "ITStrackingCUDA/Context.h"
#include "ITStrackingCUDA/Stream.h"
#include "ITStrackingCUDA/VertexerTraitsGPU.h"

// #define TrackletingLayerOrder::fromInnermostToMiddleLayer 0
// #define TrackletingLayerOrder::fromMiddleToOuterLayer 1

namespace o2
{
namespace its
{

using constants::index_table::PhiBins;
using constants::index_table::ZBins;
using constants::its::LayersRCoordinate;
using constants::its::LayersZCoordinate;
using constants::math::TwoPi;
using index_table_utils::getPhiBinIndex;
using index_table_utils::getZBinIndex;
using math_utils::getNormalizedPhiCoordinate;

VertexerTraitsGPU::VertexerTraitsGPU()
{
  setIsGPU(true);
}

VertexerTraitsGPU::~VertexerTraitsGPU()
{
}

void VertexerTraitsGPU::initialise(ROframe* event)
{
  reset();
  arrangeClusters(event);
  mStoreVertexerGPUPtr = mStoreVertexerGPU.initialise(mClusters, mIndexTables);
}

namespace GPU
{
__device__ int getGlobalIdx()
{
  int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}

__global__ void trackleterKernel(
  DeviceStoreVertexerGPU& store,
  const TrackletingLayerOrder layerOrder,
  const float phiCut)
{
  const size_t nClustersMiddleLayer = store.getClusters()[1].size();
  for (size_t currentClusterIndex = blockIdx.x * blockDim.x + threadIdx.x; currentClusterIndex < nClustersMiddleLayer; currentClusterIndex += blockDim.x * gridDim.x) {
    if (currentClusterIndex < nClustersMiddleLayer) {
      int storedTracklets{0};
      const size_t stride{currentClusterIndex * store.getConfig().maxTrackletsPerCluster};
      const Cluster& currentCluster = store.getClusters()[1][currentClusterIndex]; // assign-constructor may be a problem, check
      const VertexerLayerName adjacentLayerIndex{layerOrder == TrackletingLayerOrder::fromInnermostToMiddleLayer ? VertexerLayerName::innermostLayer : VertexerLayerName::outerLayer};
      const int4 selectedBinsRect{VertexerTraits::getBinsRect(currentCluster, static_cast<int>(adjacentLayerIndex), 0.f, 50.f, phiCut)};
      if (selectedBinsRect.x != 0 || selectedBinsRect.y != 0 || selectedBinsRect.z != 0 || selectedBinsRect.w != 0) {
        int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};
        if (phiBinsNum < 0) {
          phiBinsNum += PhiBins;
        }
        const size_t nClustersAdjacentLayer = store.getClusters()[static_cast<int>(adjacentLayerIndex)].size();
        for (size_t iPhiBin{selectedBinsRect.y}, iPhiCount{0}; iPhiCount < phiBinsNum; iPhiBin = ++iPhiBin == PhiBins ? 0 : iPhiBin, iPhiCount++) {
          const int firstBinIndex{index_table_utils::getBinIndex(selectedBinsRect.x, iPhiBin)};
          const int firstRowClusterIndex{store.getIndexTable(adjacentLayerIndex)[firstBinIndex]};
          const int maxRowClusterIndex{store.getIndexTable(adjacentLayerIndex)[firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1]};
          for (size_t iAdjacentCluster{firstRowClusterIndex}; iAdjacentCluster < maxRowClusterIndex && iAdjacentCluster < nClustersAdjacentLayer; ++iAdjacentCluster) {
            const Cluster& adjacentCluster = store.getClusters()[static_cast<int>(adjacentLayerIndex)][iAdjacentCluster]; // assign-constructor may be a problem, check
            if (gpu::GPUCommonMath::Abs(currentCluster.phiCoordinate - adjacentCluster.phiCoordinate) < phiCut) {
              if (storedTracklets < store.getConfig().maxTrackletsPerCluster) {
                if (layerOrder == TrackletingLayerOrder::fromInnermostToMiddleLayer) {
                  store.getDuplets01().emplace(stride + storedTracklets, iAdjacentCluster, currentClusterIndex, adjacentCluster, currentCluster);
                } else {
                  store.getDuplets12().emplace(stride + storedTracklets, currentClusterIndex, iAdjacentCluster, currentCluster, adjacentCluster);
                }
                ++storedTracklets;
              } else {
                printf("debug: leaving tracklet behind\n");
              }
            }
          }
        }
      }
      store.getNFoundTracklets(layerOrder).emplace(currentClusterIndex, storedTracklets);
    }
  }
}

__global__ void trackletSelectionKernel(
  DeviceStoreVertexerGPU& store,
  const float tanLambdaCut = 0.025f)
{
  const int currentClusterIndex{getGlobalIdx()};
  if (currentClusterIndex < store.getClusters()[1].size()) {
    const int stride{currentClusterIndex * store.getConfig().maxTrackletsPerCluster};
    int validTracklets{0};
    for (int iTracklet12{0}; iTracklet12 < store.getNFoundTracklets(TrackletingLayerOrder::fromMiddleToOuterLayer)[currentClusterIndex]; ++iTracklet12) {
      for (int iTracklet01{0}; iTracklet01 < store.getNFoundTracklets(TrackletingLayerOrder::fromInnermostToMiddleLayer)[currentClusterIndex] && validTracklets < store.getConfig().maxTrackletsPerCluster; ++iTracklet01) {
        const float deltaTanLambda{gpu::GPUCommonMath::Abs(store.getDuplets01()[stride + iTracklet01].tanLambda - store.getDuplets12()[stride + iTracklet12].tanLambda)};
        if (deltaTanLambda < tanLambdaCut) {
          store.getLines().emplace(stride + validTracklets, store.getDuplets01()[stride + iTracklet01], store.getClusters()[0].get(), store.getClusters()[1].get());
          ++validTracklets;
        }
      }
    }
    if (validTracklets != store.getConfig().maxTrackletsPerCluster) {
      store.getLines().emplace(stride + validTracklets);
    } else {
      printf("info: fulfilled all the space with tracklets\n");
    }
  }
}

__global__ void debugSumKernel(const int* arrayToSum, const int size)
{
  const int currentClusterIndex{getGlobalIdx()};
  if (currentClusterIndex == 0) {
    int sum{0};
    printf("on device:\n");
    for (int i{0}; i < size; ++i) {
      printf("\t%d, ", arrayToSum[i]);
      sum += arrayToSum[i];
    }
    printf("\nSum is: %d\n", sum);
  }
}
} // namespace GPU

void VertexerTraitsGPU::computeTracklets()
{
  const dim3 threadsPerBlock{GPU::Utils::Host::getBlockSize(mClusters[1].capacity())};
  const dim3 blocksGrid{GPU::Utils::Host::getBlocksGrid(threadsPerBlock, mClusters[1].capacity())};

  GPU::trackleterKernel<<<1, threadsPerBlock>>>(
    getDeviceContext(),
    GPU::TrackletingLayerOrder::fromInnermostToMiddleLayer,
    mVrtParams.phiCut);

  GPU::trackleterKernel<<<1, threadsPerBlock>>>(
    getDeviceContext(),
    GPU::TrackletingLayerOrder::fromMiddleToOuterLayer,
    mVrtParams.phiCut);

  cudaError_t error = cudaGetLastError();

  if (error != cudaSuccess) {
    std::ostringstream errorString{};
    errorString << "CUDA API returned error  [" << cudaGetErrorString(error) << "] (code " << error << ")" << std::endl;
    throw std::runtime_error{errorString.str()};
  }
}

void VertexerTraitsGPU::computeTrackletMatching()
{
  const dim3 threadsPerBlock{GPU::Utils::Host::getBlockSize(mClusters[1].capacity())};
  const dim3 blocksGrid{GPU::Utils::Host::getBlocksGrid(threadsPerBlock, mClusters[1].capacity())};

  GPU::trackletSelectionKernel<<<1, threadsPerBlock>>>(
    getDeviceContext(),
    mVrtParams.tanLambdaCut);

  cudaError_t error = cudaGetLastError();

  if (error != cudaSuccess) {
    std::ostringstream errorString{};
    errorString << "CUDA API returned error  [" << cudaGetErrorString(error) << "] (code " << error << ")" << std::endl;
    throw std::runtime_error{errorString.str()};
  }
}
//
// // DEBUG section
// Line* lines = new Line[static_cast<int>(80e6)];
// Tracklet* comb01 = new Tracklet[static_cast<int>(80e6)];
// Tracklet* comb12 = new Tracklet[static_cast<int>(80e6)];
// int* foundTracklets01_h = new int[clusterSize1];
// int* foundTracklets12_h = new int[clusterSize1];
// int* cartellino = new int[2];
//
// cudaMemcpy(comb01, mGPURefTracklet01, static_cast<int>(80e6) * sizeof(Tracklet), cudaMemcpyDeviceToHost);
// cudaMemcpy(comb12, mGPURefTracklet12, static_cast<int>(80e6) * sizeof(Tracklet), cudaMemcpyDeviceToHost);
// cudaMemcpy(foundTracklets01_h, numTracks01, clusterSize1 * sizeof(int), cudaMemcpyDeviceToHost);
// cudaMemcpy(foundTracklets12_h, numTracks12, clusterSize1 * sizeof(int), cudaMemcpyDeviceToHost);
// cudaMemcpy(lines, mGPUtracklets, static_cast<int>(80e6) * sizeof(Line), cudaMemcpyDeviceToHost);
// cudaMemcpy(cartellino, debugArray, sizeof(int) * 2, cudaMemcpyDeviceToHost);
// cudaDeviceSynchronize();
//

//
// for (int i{0}; i < clusterSize1; ++i) {
//   const int stride{i * static_cast<int>(2e3)};
//   for (int j{0}; j < foundTracklets12_h[i]; ++j) {
//     mComb12.push_back(comb12[stride + j]);
//   }
// }
//
// for (int i{0}; i < clusterSize1; ++i) {
//   const int stride{i * static_cast<int>(2e3)};
//   for (int j{0}; j < foundTracklets12_h[i]; ++j) {
//     for (int k{0}; k < foundTracklets01_h[i]; ++k) {
//       assert(comb01[stride + k].secondClusterIndex == comb12[stride + j].firstClusterIndex);
//       const float deltaTanLambda{gpu::GPUCommonMath::Abs(comb01[stride + k].tanLambda - comb12[stride + j].tanLambda)};
//     }
//   }
// }
//
// for (int i{0}; i < clusterSize1; ++i) {
//   const int stride{i * static_cast<int>(2e3)};
//   int counter{0};
//   while (!lines[stride + counter].isEmpty) {
//     if (counter == static_cast<int>(2e3))
//       break;
//     mTracklets.push_back(lines[stride + counter]);
//     ++counter;
//   }
// }

VertexerTraits* createVertexerTraitsGPU()
{
  return new VertexerTraitsGPU;
}

} // namespace its
} // namespace o2
