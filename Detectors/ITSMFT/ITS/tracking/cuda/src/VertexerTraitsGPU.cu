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
#include "cub/cub.cuh"
#include <cooperative_groups.h>
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

#define LAYER0_TO_LAYER1 0
#define LAYER1_TO_LAYER2 1

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
  // GPU memory is not being reset at the moment
  mStoreVertexerGPU.initialise(mClusters);
}

namespace GPU
{

GPU_DEVICE int getGlobalIdx()
{
  int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}

GPU_GLOBAL void trackleterKernel(
  const Cluster* GPUclustersNext,
  const Cluster* GPUclustersCurrent,
  const int GPUclusterSizeNext,
  const int GPUclusterSize1,
  const int* indexTableNext,
  const char layerOrder,
  const float phiCut,
  Tracklet* GPUtracklets,
  int* foundTracklets,
  int* counter,
  // int* debugarray,
  const char isMc = false,
  const int* MClabelsNext = nullptr,
  const int* MClabelsCurrent = nullptr,
  const int maxTrackletsPerCluster = static_cast<int>(2e3) // also the stride
)
{
  for (int currentClusterIndex = blockIdx.x * blockDim.x + threadIdx.x; currentClusterIndex < GPUclusterSize1; currentClusterIndex += blockDim.x * gridDim.x) {
    if (layerOrder == LAYER0_TO_LAYER1) {
      atomicAdd(&counter[0], 1);
    } else {
      atomicAdd(&counter[1], 1);
    }
    if (currentClusterIndex < GPUclusterSize1) {
      int storedTracklets{0};
      const int stride{currentClusterIndex * maxTrackletsPerCluster};
      const Cluster currentCluster{GPUclustersCurrent[currentClusterIndex]};
      const int layerIndex{layerOrder == LAYER0_TO_LAYER1 ? 0 : 2};
      const int4 selectedBinsRect{VertexerTraits::getBinsRect(currentCluster, layerIndex, 0.f, 50.f, phiCut)};
      if (selectedBinsRect.x != 0 || selectedBinsRect.y != 0 || selectedBinsRect.z != 0 || selectedBinsRect.w != 0) {
        int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};
        if (phiBinsNum < 0) {
          phiBinsNum += PhiBins;
        }
        for (int iPhiBin{selectedBinsRect.y}, iPhiCount{0}; iPhiCount < phiBinsNum; iPhiBin = ++iPhiBin == PhiBins ? 0 : iPhiBin, iPhiCount++) {
          const int firstBinIndex{index_table_utils::getBinIndex(selectedBinsRect.x, iPhiBin)};
          const int firstRowClusterIndex{indexTableNext[firstBinIndex]};
          const int maxRowClusterIndex{indexTableNext[firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1]};
          for (int iNextLayerCluster{firstRowClusterIndex}; iNextLayerCluster < maxRowClusterIndex && iNextLayerCluster < GPUclusterSizeNext; ++iNextLayerCluster) {
            const Cluster& nextCluster{GPUclustersNext[iNextLayerCluster]};
            const char testMC{!isMc || MClabelsNext[iNextLayerCluster] == MClabelsCurrent[currentClusterIndex] && MClabelsNext[iNextLayerCluster] != -1};
            if (gpu::GPUCommonMath::Abs(currentCluster.phiCoordinate - nextCluster.phiCoordinate) < phiCut && testMC) {
              if (storedTracklets < maxTrackletsPerCluster) {
                if (layerOrder == LAYER0_TO_LAYER1) {
                  new (GPUtracklets + stride + storedTracklets) Tracklet(iNextLayerCluster, currentClusterIndex, nextCluster, currentCluster);
                } else {
                  new (GPUtracklets + stride + storedTracklets) Tracklet(currentClusterIndex, iNextLayerCluster, currentCluster, nextCluster);
                }
                ++storedTracklets;
              } else {
                printf("debug: leaving tracklet behind\n");
              }
            }
          }
        }
      }
      foundTracklets[currentClusterIndex] = storedTracklets;
    }
  }
}

GPU_GLOBAL void trackletSelectionKernel(
  const Cluster* GPUclusters0,
  const Cluster* GPUclusters1,
  const Tracklet* GPUtracklets01,
  const Tracklet* GPUtracklets12,
  const int GPUclusterSize1,
  const int* foundTracklets01,
  const int* foundTracklets12,
  Line* destTracklets,
  const float tanLambdaCut = 0.025f,
  const int maxTracklets = static_cast<int>(2e3))
{
  const int currentClusterIndex{getGlobalIdx()};
  if (currentClusterIndex < GPUclusterSize1) {
    const int stride{currentClusterIndex * maxTracklets};
    int validTracklets{0};
    for (int iTracklet12{0}; iTracklet12 < foundTracklets12[currentClusterIndex]; ++iTracklet12) {
      for (int iTracklet01{0}; iTracklet01 < foundTracklets01[currentClusterIndex] && validTracklets < maxTracklets; ++iTracklet01) {
        const float deltaTanLambda{gpu::GPUCommonMath::Abs(GPUtracklets01[stride + iTracklet01].tanLambda - GPUtracklets12[stride + iTracklet12].tanLambda)};
        if (/*deltaTanLambda < tanLambdaCut*/ true) {
          new (destTracklets + stride + validTracklets) Line(GPUtracklets01[stride + iTracklet01], GPUclusters0, GPUclusters1);
          ++validTracklets;
        }
      }
    }
    if (validTracklets != maxTracklets) {
      new (destTracklets + stride + validTracklets) Line(); // always complete line with empty one unless all spaces taken
    } else {
      printf("[INFO]: Fulfilled all the space with tracklets.\n");
    }
  }
}

GPU_GLOBAL void debugSumKernel(const int* arrayToSum, const int size)
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

using GPU::Utils::Host::getBlocksGrid;
using GPU::Utils::Host::getBlockSize;

void VertexerTraitsGPU::computeTracklets()
{
  const unsigned char useMCLabel = false;
  if (useMCLabel)
    std::cout << "info: running trackleter in Montecarlo check mode." << std::endl;
  const GPU::DeviceProperties& deviceProperties = GPU::Context::getInstance().getDeviceProperties();

  const int clusterSize0 = static_cast<int>(mClusters[0].size());
  const int clusterSize1 = static_cast<int>(mClusters[1].size());
  const int clusterSize2 = static_cast<int>(mClusters[2].size());

  dim3 threadsPerBlock{GPU::Utils::Host::getBlockSize(clusterSize1)};
  dim3 blocksGrid{GPU::Utils::Host::getBlocksGrid(threadsPerBlock, clusterSize1)};

  int* numTracks01;
  int* numTracks12;

  // DEBUG
  // int* debugArray;
  // cudaMalloc(reinterpret_cast<void**>(&debugArray), 2 * sizeof(int));
  // cudaMemset(debugArray, 0, 2 * sizeof(int));
  //
  // cudaMalloc(reinterpret_cast<void**>(&numTracks01), clusterSize1 * sizeof(int));
  // cudaMalloc(reinterpret_cast<void**>(&numTracks12), clusterSize1 * sizeof(int));
  // cudaMemset(numTracks01, 0, clusterSize1 * sizeof(int));
  // cudaMemset(numTracks12, 0, clusterSize1 * sizeof(int));
  //
  // // cudaMalloc(reinterpret_cast<void**>(&mGPUclusters0), clusterSize0 * sizeof(Cluster));
  // // cudaMalloc(reinterpret_cast<void**>(&mGPUclusters1), clusterSize1 * sizeof(Cluster));
  // // cudaMalloc(reinterpret_cast<void**>(&mGPUclusters2), clusterSize2 * sizeof(Cluster));
  //
  // cudaMemcpy(mGPUindexTable0, mIndexTables[0].data(), mIndexTables[0].size() * sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(mGPUindexTable2, mIndexTables[2].data(), mIndexTables[2].size() * sizeof(int), cudaMemcpyHostToDevice);
  //
  // cudaMemcpy(mGPUclusters0, mClusters[0].data(), mClusters[0].size() * sizeof(Cluster), cudaMemcpyHostToDevice);
  // cudaMemcpy(mGPUclusters1, mClusters[1].data(), mClusters[1].size() * sizeof(Cluster), cudaMemcpyHostToDevice);
  // cudaMemcpy(mGPUclusters2, mClusters[2].data(), mClusters[2].size() * sizeof(Cluster), cudaMemcpyHostToDevice);

  // if (useMCLabel) {

  //   cudaMalloc(reinterpret_cast<void**>(&mGPUMClabels0), clusterSize0 * sizeof(int));
  //   cudaMalloc(reinterpret_cast<void**>(&mGPUMClabels1), clusterSize1 * sizeof(int));
  //   cudaMalloc(reinterpret_cast<void**>(&mGPUMClabels2), clusterSize2 * sizeof(int));

  //   // cudaMemcpy(mGPUMClabels0, getMClabelsLayer(0).data(), clusterSize0 * sizeof(int), cudaMemcpyHostToDevice);
  //   // cudaMemcpy(mGPUMClabels1, getMClabelsLayer(1).data(), clusterSize1 * sizeof(int), cudaMemcpyHostToDevice);
  //   // cudaMemcpy(mGPUMClabels2, getMClabelsLayer(2).data(), clusterSize2 * sizeof(int), cudaMemcpyHostToDevice);
  // }

  // std::cout << "clusters on L0: " << clusterSize0 << " clusters on L1: " << clusterSize1 << " clusters on L2: " << clusterSize2 << std::endl;

  // GPU::trackleterKernel<<<blocksGrid, threadsPerBlock>>>(
  //   mGPUclusters0,
  //   mGPUclusters1,
  //   mClusters[0].size(),
  //   mClusters[1].size(),
  //   mGPUindexTable0,
  //   LAYER0_TO_LAYER1,
  //   mVrtParams.phiCut,
  //   mGPURefTracklet01,
  //   numTracks01,
  //   debugArray,
  //   useMCLabel,
  //   mGPUMClabels0,
  //   mGPUMClabels1);
  //
  // GPU::trackleterKernel<<<blocksGrid, threadsPerBlock>>>(
  //   mGPUclusters2,
  //   mGPUclusters1,
  //   mClusters[2].size(),
  //   mClusters[1].size(),
  //   mGPUindexTable2,
  //   LAYER1_TO_LAYER2,
  //   mVrtParams.phiCut,
  //   mGPURefTracklet12,
  //   numTracks12,
  //   debugArray,
  //   useMCLabel,
  //   mGPUMClabels2,
  //   mGPUMClabels1);
  //
  // GPU::trackletSelectionKernel<<<blocksGrid, threadsPerBlock>>>(
  //   mGPUclusters0,
  //   mGPUclusters1,
  //   mGPURefTracklet01,
  //   mGPURefTracklet12,
  //   mClusters[1].size(),
  //   numTracks01,
  //   numTracks12,
  //   mGPUtracklets);
  //
  // cudaError_t error = cudaGetLastError();
  //
  // if (error != cudaSuccess) {
  //   std::ostringstream errorString{};
  //   errorString << "CUDA API returned error  [" << cudaGetErrorString(error) << "] (code " << error << ")" << std::endl;
  //   throw std::runtime_error{errorString.str()};
  // }
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
  // // Dump for debug
  // for (int i{0}; i < clusterSize1; ++i) {
  //   const int stride{i * static_cast<int>(2e3)};
  //   for (int j{0}; j < foundTracklets01_h[i]; ++j) {
  //     mComb01.push_back(comb01[stride + j]);
  //   }
  // }
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
  //
  // delete[] lines;
  // delete[] comb01;
  // delete[] comb12;
  // delete[] foundTracklets01_h;
  // delete[] foundTracklets12_h;
  // delete[] cartellino;
  //
  // // \DEBUG section
  //
  // cudaFree(debugArray);
  // cudaFree(numTracks01);
  // cudaFree(numTracks12);
  //
  // cudaFree(mGPUclusters0);
  // cudaFree(mGPUclusters1);
  // cudaFree(mGPUclusters2);
  //
  // if (useMCLabel) {
  //   cudaFree(mGPUMClabels0);
  //   cudaFree(mGPUMClabels1);
  //   cudaFree(mGPUMClabels2);
  // }
}

VertexerTraits* createVertexerTraitsGPU()
{
  return new VertexerTraitsGPU;
}

} // namespace its
} // namespace o2
