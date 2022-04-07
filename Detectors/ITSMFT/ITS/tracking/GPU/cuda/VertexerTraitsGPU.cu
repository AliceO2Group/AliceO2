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
//
/// \author matteo.concas@cern.ch

#include <iostream>
#include <sstream>
#include <fstream>
#include <array>
#include <cassert>

#ifndef GPUCA_GPUCODE_GENRTC
#include <cub/cub.cuh>
#endif

#include "ITStracking/MathUtils.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/Tracklet.h"

#include "ITStrackingGPU/Utils.h"
#include "ITStrackingGPU/ClusterLinesGPU.h"
#include "ITStrackingGPU/VertexerTraitsGPU.h"

#include <fairlogger/Logger.h>

namespace o2
{
namespace its
{

using constants::its::VertexerHistogramVolume;
using constants::math::TwoPi;
using math_utils::getNormalizedPhi;

using namespace constants::its2;
GPUd() const int4 getBinsRect(const Cluster& currentCluster, const int layerIndex,
                              const float z1, float maxdeltaz, float maxdeltaphi)
{
  const float zRangeMin = z1 - maxdeltaz;
  const float phiRangeMin = currentCluster.phi - maxdeltaphi;
  const float zRangeMax = z1 + maxdeltaz;
  const float phiRangeMax = currentCluster.phi + maxdeltaphi;

  if (zRangeMax < -LayersZCoordinate()[layerIndex + 1] ||
      zRangeMin > LayersZCoordinate()[layerIndex + 1] || zRangeMin > zRangeMax) {

    return getEmptyBinsRect();
  }

  return int4{o2::gpu::GPUCommonMath::Max(0, getZBinIndex(layerIndex + 1, zRangeMin)),
              getPhiBinIndex(phiRangeMin),
              o2::gpu::GPUCommonMath::Min(ZBins - 1, getZBinIndex(layerIndex + 1, zRangeMax)),
              getPhiBinIndex(phiRangeMax)};
}

GPUh() void gpuThrowOnError()
{
  cudaError_t error = cudaGetLastError();

  if (error != cudaSuccess) {
    std::ostringstream errorString{};
    errorString << GPU_ARCH << " API returned error  [" << cudaGetErrorString(error) << "] (code " << error << ")" << std::endl;
    throw std::runtime_error{errorString.str()};
  }
}

VertexerTraitsGPU::VertexerTraitsGPU()
{
  gpu::utils::host::gpuMalloc((void**)&mDeviceIndexTableUtils, sizeof(IndexTableUtils));
  setIsGPU(true);
}

VertexerTraitsGPU::~VertexerTraitsGPU()
{
  gpu::utils::host::gpuFree(mDeviceIndexTableUtils);
}

void VertexerTraitsGPU::initialise(const MemoryParameters& memParams, const TrackingParameters& trackingParams)
{
  if (!mIndexTableUtils.getNzBins()) {
    updateVertexingParameters(mVrtParams);
  }
  gpu::utils::host::gpuMemcpyHostToDevice(mDeviceIndexTableUtils, &mIndexTableUtils, sizeof(mIndexTableUtils));
  mTimeFrameGPU->initialise(0, memParams, trackingParams, 3);
  setIsGPU(true);
}

namespace gpu
{

template <typename... Args>
GPUd() void printOnThread(const unsigned int tId, const char* str, Args... args)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == tId) {
    printf(str, args...);
  }
}

GPUd() void printVectorOnThread(const char* name, Vector<int>& vector, size_t size, const unsigned int tId = 0)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == tId) {
    printf("vector %s :", name);
    for (int i{0}; i < size; ++i) {
      printf("%d ", vector[i]);
    }
    printf("\n");
  }
}

GPUg() void printVectorKernel(DeviceStoreVertexerGPU& store, const unsigned int threadId)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == threadId) {
    for (int i{0}; i < store.getConfig().histConf.nBinsXYZ[0] - 1; ++i) {
      printf("%d: %d\n", i, store.getHistogramXYZ()[0].get()[i]);
    }
    printf("\n");
    for (int i{0}; i < store.getConfig().histConf.nBinsXYZ[1] - 1; ++i) {
      printf("%d: %d\n", i, store.getHistogramXYZ()[1].get()[i]);
    }
    printf("\n");
    for (int i{0}; i < store.getConfig().histConf.nBinsXYZ[2] - 1; ++i) {
      printf("%d: %d\n", i, store.getHistogramXYZ()[2].get()[i]);
    }
    printf("\n");
  }
}

GPUg() void dumpMaximaKernel(DeviceStoreVertexerGPU& store, const unsigned int threadId)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == threadId) {
    printf("XmaxBin: %d at index: %d | YmaxBin: %d at index: %d | ZmaxBin: %d at index: %d\n",
           store.getTmpVertexPositionBins()[0].value, store.getTmpVertexPositionBins()[0].key,
           store.getTmpVertexPositionBins()[1].value, store.getTmpVertexPositionBins()[1].key,
           store.getTmpVertexPositionBins()[2].value, store.getTmpVertexPositionBins()[2].key);
  }
}

template <TrackletMode Mode>
GPUg() void trackleterKernel(
  const Cluster* clustersNextLayer,    // 0 2
  const Cluster* clustersCurrentLayer, // 1 1
  const int sizeNextLClusters,
  const int sizeCurrentLClusters,
  const int* indexTableNext,
  const float phiCut,
  Tracklet* Tracklets,
  int* foundTracklets,
  const IndexTableUtils* utils,
  const int maxTrackletsPerCluster = 10)
{
  const int phiBins{utils->getNphiBins()};
  const int zBins{utils->getNzBins()};
  // loop on layer1 clusters
  for (int iCurrentLayerClusterIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentLayerClusterIndex < sizeCurrentLClusters; iCurrentLayerClusterIndex += blockDim.x * gridDim.x) {
    if (iCurrentLayerClusterIndex < sizeCurrentLClusters) {
      unsigned int storedTracklets{0};
      const int stride{iCurrentLayerClusterIndex * maxTrackletsPerCluster};
      const Cluster& currentCluster = clustersCurrentLayer[iCurrentLayerClusterIndex];
      const int4 selectedBinsRect{VertexerTraits::getBinsRect(currentCluster, (int)Mode, 0.f, 50.f, phiCut / 2, *utils)};
      if (selectedBinsRect.x != 0 || selectedBinsRect.y != 0 || selectedBinsRect.z != 0 || selectedBinsRect.w != 0) {
        int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};
        if (phiBinsNum < 0) {
          phiBinsNum += phiBins;
        }
        // loop on phi bins next layer
        for (unsigned int iPhiBin{(unsigned int)selectedBinsRect.y}, iPhiCount{0}; iPhiCount < (unsigned int)phiBinsNum; iPhiBin = ++iPhiBin == phiBins ? 0 : iPhiBin, iPhiCount++) {
          const int firstBinIndex{utils->getBinIndex(selectedBinsRect.x, iPhiBin)};
          const int firstRowClusterIndex{indexTableNext[firstBinIndex]};
          const int maxRowClusterIndex{indexTableNext[firstBinIndex + zBins]};
          // loop on clusters next layer
          for (int iNextLayerClusterIndex{firstRowClusterIndex}; iNextLayerClusterIndex < maxRowClusterIndex && iNextLayerClusterIndex < sizeNextLClusters; ++iNextLayerClusterIndex) {
            const Cluster& nextCluster = clustersNextLayer[iNextLayerClusterIndex];
            if (o2::gpu::GPUCommonMath::Abs(currentCluster.phi - nextCluster.phi) < phiCut) {
              if (storedTracklets < maxTrackletsPerCluster) {
                if constexpr (Mode == TrackletMode::Layer0Layer1) {
                  new (Tracklets + stride + storedTracklets) Tracklet{iNextLayerClusterIndex, iCurrentLayerClusterIndex, nextCluster, currentCluster};
                } else {
                  new (Tracklets + stride + storedTracklets) Tracklet{iCurrentLayerClusterIndex, iNextLayerClusterIndex, currentCluster, nextCluster};
                }
                ++storedTracklets;
              }
            }
          }
        }
      }
      foundTracklets[iCurrentLayerClusterIndex] = storedTracklets;
    }
  }
}

GPUg() void trackletSelectionKernel(
  DeviceStoreVertexerGPU& store,
  const unsigned char isInitRun = false,
  const float tanLambdaCut = 0.025f,
  const float phiCut = 0.002f)
{
  const size_t nClustersMiddleLayer = store.getClusters()[1].size();
  for (size_t iCurrentLayerClusterIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentLayerClusterIndex < nClustersMiddleLayer; iCurrentLayerClusterIndex += blockDim.x * gridDim.x) {
    const int stride{static_cast<int>(iCurrentLayerClusterIndex * store.getConfig().maxTrackletsPerCluster)};
    int validTracklets{0};
    for (int iTracklet12{0}; iTracklet12 < store.getNFoundTracklets(TrackletingLayerOrder::fromMiddleToOuterLayer)[iCurrentLayerClusterIndex]; ++iTracklet12) {
      for (int iTracklet01{0}; iTracklet01 < store.getNFoundTracklets(TrackletingLayerOrder::fromInnermostToMiddleLayer)[iCurrentLayerClusterIndex] && validTracklets < store.getConfig().maxTrackletsPerCluster; ++iTracklet01) {
        const float deltaTanLambda{o2::gpu::GPUCommonMath::Abs(store.getDuplets01()[stride + iTracklet01].tanLambda - store.getDuplets12()[stride + iTracklet12].tanLambda)};
        const float deltaPhi{o2::gpu::GPUCommonMath::Abs(store.getDuplets01()[stride + iTracklet01].phi - store.getDuplets12()[stride + iTracklet12].phi)};
        if (deltaTanLambda < tanLambdaCut && deltaPhi < phiCut && validTracklets != store.getConfig().maxTrackletsPerCluster) {
          assert(store.getDuplets01()[stride + iTracklet01].secondClusterIndex == store.getDuplets12()[stride + iTracklet12].firstClusterIndex);
          if (!isInitRun) {
            store.getLines().emplace(store.getNExclusiveFoundLines()[iCurrentLayerClusterIndex] + validTracklets, store.getDuplets01()[stride + iTracklet01], store.getClusters()[0].get(), store.getClusters()[1].get());
#ifdef _ALLOW_DEBUG_TREES_ITS_
            store.getDupletIndices()[0].emplace(store.getNExclusiveFoundLines()[iCurrentLayerClusterIndex] + validTracklets, stride + iTracklet01);
            store.getDupletIndices()[1].emplace(store.getNExclusiveFoundLines()[iCurrentLayerClusterIndex] + validTracklets, stride + iTracklet12);
#endif
          }
          ++validTracklets;
        }
      }
    }
    if (isInitRun) {
      store.getNFoundLines().emplace(iCurrentLayerClusterIndex, validTracklets);
      if (validTracklets >= store.getConfig().maxTrackletsPerCluster) {
        printf("Warning: not enough space for tracklet selection, some lines will be left behind\n");
      }
    }
  }
}

GPUg() void computeCentroidsKernel(DeviceStoreVertexerGPU& store,
                                   const float pairCut)
{
  const size_t nLines = store.getNExclusiveFoundLines()[store.getClusters()[1].size() - 1] + store.getNFoundLines()[store.getClusters()[1].size() - 1];
  const size_t maxIterations{nLines * (nLines - 1) / 2};
  for (size_t currentThreadIndex = blockIdx.x * blockDim.x + threadIdx.x; currentThreadIndex < maxIterations; currentThreadIndex += blockDim.x * gridDim.x) {
    int iFirstLine = currentThreadIndex / nLines;
    int iSecondLine = currentThreadIndex % nLines;
    if (iSecondLine <= iFirstLine) {
      iFirstLine = nLines - iFirstLine - 2;
      iSecondLine = nLines - iSecondLine - 1;
    }
    if (Line::getDCA(store.getLines()[iFirstLine], store.getLines()[iSecondLine]) < pairCut) {
      ClusterLinesGPU cluster{store.getLines()[iFirstLine], store.getLines()[iSecondLine]};
      if (cluster.getVertex()[0] * cluster.getVertex()[0] + cluster.getVertex()[1] * cluster.getVertex()[1] < 1.98f * 1.98f) {
        // printOnThread(0, "xCentr: %f, yCentr: %f \n", cluster.getVertex()[0], cluster.getVertex()[1]);
        store.getXYCentroids().emplace(2 * currentThreadIndex, cluster.getVertex()[0]);
        store.getXYCentroids().emplace(2 * currentThreadIndex + 1, cluster.getVertex()[1]);
      } else {
        // writing some data anyway outside the histogram, they will not be put in the histogram, by construction.
        store.getXYCentroids().emplace(2 * currentThreadIndex, 2 * store.getConfig().histConf.lowHistBoundariesXYZ[0]);
        store.getXYCentroids().emplace(2 * currentThreadIndex + 1, 2 * store.getConfig().histConf.lowHistBoundariesXYZ[1]);
      }
    } else {
      // writing some data anyway outside the histogram, they will not be put in the histogram, by construction.
      store.getXYCentroids().emplace(2 * currentThreadIndex, 2 * store.getConfig().histConf.lowHistBoundariesXYZ[0]);
      store.getXYCentroids().emplace(2 * currentThreadIndex + 1, 2 * store.getConfig().histConf.lowHistBoundariesXYZ[1]);
    }
  }
}

GPUg() void computeZCentroidsKernel(DeviceStoreVertexerGPU& store,
                                    const float pairCut, const int binOpeningX, const int binOpeningY)
{
  const int nLines = store.getNExclusiveFoundLines()[store.getClusters()[1].size() - 1] + store.getNFoundLines()[store.getClusters()[1].size() - 1];
  for (size_t currentThreadIndex = blockIdx.x * blockDim.x + threadIdx.x; currentThreadIndex < nLines; currentThreadIndex += blockDim.x * gridDim.x) {
    if (store.getTmpVertexPositionBins()[0].value || store.getTmpVertexPositionBins()[1].value) {
      float tmpX{store.getConfig().histConf.lowHistBoundariesXYZ[0] + store.getTmpVertexPositionBins()[0].key * store.getConfig().histConf.binSizeHistX + store.getConfig().histConf.binSizeHistX / 2};
      int sumWX{store.getTmpVertexPositionBins()[0].value};
      float wX{tmpX * store.getTmpVertexPositionBins()[0].value};
      for (int iBin{o2::gpu::GPUCommonMath::Max(0, store.getTmpVertexPositionBins()[0].key - binOpeningX)}; iBin < o2::gpu::GPUCommonMath::Min(store.getTmpVertexPositionBins()[0].key + binOpeningX + 1, store.getConfig().histConf.nBinsXYZ[0] - 1); ++iBin) {
        if (iBin != store.getTmpVertexPositionBins()[0].key) {
          wX += (store.getConfig().histConf.lowHistBoundariesXYZ[0] + iBin * store.getConfig().histConf.binSizeHistX + store.getConfig().histConf.binSizeHistX / 2) * store.getHistogramXYZ()[0].get()[iBin];
          sumWX += store.getHistogramXYZ()[0].get()[iBin];
        }
      }
      float tmpY{store.getConfig().histConf.lowHistBoundariesXYZ[1] + store.getTmpVertexPositionBins()[1].key * store.getConfig().histConf.binSizeHistY + store.getConfig().histConf.binSizeHistY / 2};
      int sumWY{store.getTmpVertexPositionBins()[1].value};
      float wY{tmpY * store.getTmpVertexPositionBins()[1].value};
      for (int iBin{o2::gpu::GPUCommonMath::Max(0, store.getTmpVertexPositionBins()[1].key - binOpeningY)}; iBin < o2::gpu::GPUCommonMath::Min(store.getTmpVertexPositionBins()[1].key + binOpeningY + 1, store.getConfig().histConf.nBinsXYZ[1] - 1); ++iBin) {
        if (iBin != store.getTmpVertexPositionBins()[1].key) {
          wY += (store.getConfig().histConf.lowHistBoundariesXYZ[1] + iBin * store.getConfig().histConf.binSizeHistY + store.getConfig().histConf.binSizeHistY / 2) * store.getHistogramXYZ()[1].get()[iBin];
          sumWY += store.getHistogramXYZ()[1].get()[iBin];
        }
      }
      store.getBeamPosition().emplace(0, wX / sumWX);
      store.getBeamPosition().emplace(1, wY / sumWY);
      float fakeBeamPoint1[3] = {store.getBeamPosition()[0], store.getBeamPosition()[1], -1}; // get two points laying at different z, to create line object
      float fakeBeamPoint2[3] = {store.getBeamPosition()[0], store.getBeamPosition()[1], 1};
      Line pseudoBeam = {fakeBeamPoint1, fakeBeamPoint2};
      if (Line::getDCA(store.getLines()[currentThreadIndex], pseudoBeam) < pairCut) {
        ClusterLinesGPU cluster{store.getLines()[currentThreadIndex], pseudoBeam};
        store.getZCentroids().emplace(currentThreadIndex, cluster.getVertex()[2]);
      } else {
        store.getZCentroids().emplace(currentThreadIndex, 2 * store.getConfig().histConf.lowHistBoundariesXYZ[2]);
      }
    }
  }
}

GPUg() void computeVertexKernel(DeviceStoreVertexerGPU& store, const int vertIndex, const int minContributors, const int binOpeningZ)
{
  for (size_t currentThreadIndex = blockIdx.x * blockDim.x + threadIdx.x; currentThreadIndex < binOpeningZ; currentThreadIndex += blockDim.x * gridDim.x) {
    if (currentThreadIndex == 0) {
      if (store.getTmpVertexPositionBins()[2].value > 1 && (store.getTmpVertexPositionBins()[0].value || store.getTmpVertexPositionBins()[1].value)) {
        float z{store.getConfig().histConf.lowHistBoundariesXYZ[2] + store.getTmpVertexPositionBins()[2].key * store.getConfig().histConf.binSizeHistZ + store.getConfig().histConf.binSizeHistZ / 2};
        float ex{0.f};
        float ey{0.f};
        float ez{0.f};
        int sumWZ{store.getTmpVertexPositionBins()[2].value};
        float wZ{z * store.getTmpVertexPositionBins()[2].value};
        for (int iBin{o2::gpu::GPUCommonMath::Max(0, store.getTmpVertexPositionBins()[2].key - binOpeningZ)}; iBin < o2::gpu::GPUCommonMath::Min(store.getTmpVertexPositionBins()[2].key + binOpeningZ + 1, store.getConfig().histConf.nBinsXYZ[2] - 1); ++iBin) {
          if (iBin != store.getTmpVertexPositionBins()[2].key) {
            wZ += (store.getConfig().histConf.lowHistBoundariesXYZ[2] + iBin * store.getConfig().histConf.binSizeHistZ + store.getConfig().histConf.binSizeHistZ / 2) * store.getHistogramXYZ()[2].get()[iBin];
            sumWZ += store.getHistogramXYZ()[2].get()[iBin];
          }
          store.getHistogramXYZ()[2].get()[iBin] = 0;
        }
        if (sumWZ > minContributors || vertIndex == 0) {
          store.getVertices().emplace(vertIndex, store.getBeamPosition()[0], store.getBeamPosition()[1], wZ / sumWZ, ex, ey, ez, sumWZ);
        } else {
          store.getVertices().emplace(vertIndex);
        }
      } else {
        store.getVertices().emplace(vertIndex);
      }
    }
  }
}
} // namespace gpu

void VertexerTraitsGPU::computeTracklets()
{
  for (int rofId{0}; rofId < mTimeFrameGPU->getNrof(); ++rofId) {
    // const dim3 threadsPerBlock{gpu::utils::host::getBlockSize(mTimeFrameGPU->getNClustersLayer(rofId, 1))};
    // const dim3 blocksGrid{gpu::utils::host::getBlocksGrid(threadsPerBlock, mTimeFrameGPU->getNClustersLayer(rofId, 1))};
    gpu::trackleterKernel<TrackletMode::Layer0Layer1><<<1, 1>>>(
      mTimeFrameGPU->getDeviceClustersOnLayer(rofId, 0),
      mTimeFrameGPU->getDeviceClustersOnLayer(rofId, 1),
      mTimeFrameGPU->getNClustersLayer(rofId, 0),
      mTimeFrameGPU->getNClustersLayer(rofId, 1),
      mTimeFrameGPU->getDeviceIndexTableL0(rofId),
      mVrtParams.phiCut,
      mTimeFrameGPU->getDeviceTracklets()[0].get(),
      mTimeFrameGPU->getDeviceNTrackletsCluster(rofId, 0),
      mDeviceIndexTableUtils,
      mTimeFrameGPU->getConfig().maxTrackletsPerCluster);

    gpu::trackleterKernel<TrackletMode::Layer1Layer2><<<1, 1>>>(
      mTimeFrameGPU->getDeviceClustersOnLayer(rofId, 2),
      mTimeFrameGPU->getDeviceClustersOnLayer(rofId, 1),
      mTimeFrameGPU->getNClustersLayer(rofId, 2),
      mTimeFrameGPU->getNClustersLayer(rofId, 1),
      mTimeFrameGPU->getDeviceIndexTableL2(rofId),
      mVrtParams.phiCut,
      mTimeFrameGPU->getDeviceTracklets()[1].get(),
      mTimeFrameGPU->getDeviceNTrackletsCluster(rofId, 1),
      mDeviceIndexTableUtils,
      mTimeFrameGPU->getConfig().maxTrackletsPerCluster);
  }
  // #ifdef VTX_DEBUG
  std::ofstream out01("NTC01.txt"), out12("NTC12.txt");
  std::vector<std::vector<int>> NtrackletsClusters01(mTimeFrameGPU->getNrof());
  std::vector<std::vector<int>> NtrackletsClusters12(mTimeFrameGPU->getNrof());
  for (int iRof{0}; iRof < mTimeFrameGPU->getNrof(); ++iRof) {
    NtrackletsClusters01[iRof].resize(mTimeFrameGPU->getClustersOnLayer(iRof, 1).size());
    NtrackletsClusters12[iRof].resize(mTimeFrameGPU->getClustersOnLayer(iRof, 1).size());
    cudaMemcpy(NtrackletsClusters01[iRof].data(), mTimeFrameGPU->getDeviceNTrackletsCluster(iRof, 0), sizeof(int) * mTimeFrameGPU->getClustersOnLayer(iRof, 1).size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(NtrackletsClusters12[iRof].data(), mTimeFrameGPU->getDeviceNTrackletsCluster(iRof, 1), sizeof(int) * mTimeFrameGPU->getClustersOnLayer(iRof, 1).size(), cudaMemcpyDeviceToHost);

    std::copy(NtrackletsClusters01[iRof].begin(), NtrackletsClusters01[iRof].end(), std::ostream_iterator<double>(out01, "\t"));
    std::copy(NtrackletsClusters12[iRof].begin(), NtrackletsClusters12[iRof].end(), std::ostream_iterator<double>(out12, "\t"));
    out01 << std::endl;
    out12 << std::endl;
  }
  out01.close();
  out12.close();
  // #endif
}

// void VertexerTraitsGPU::computeTrackletMatching()
// {
//   if (!mClusters[1].size()) {
//     return;
//   }
//   const dim3 threadsPerBlock{gpu::utils::host::getBlockSize(mClusters[1].capacity())};
//   const dim3 blocksGrid{gpu::utils::host::getBlocksGrid(threadsPerBlock, mClusters[1].capacity())};
//   size_t bufferSize = mStoreVertexerGPU.getConfig().tmpCUBBufferSize * sizeof;

//   gpu::trackletSelectionKernel<<<blocksGrid, threadsPerBlock>>>(
//     getDeviceContext(),
//     true, // isInitRun
//     mVrtParams.tanLambdaCut,
//     mVrtParams.phiCut);

//   discardResult(cub::DeviceScan::ExclusiveSum(reinterpret_cast<void*>(mStoreVertexerGPU.getCUBTmpBuffer().get()),
//                                               bufferSize,
//                                               mStoreVertexerGPU.getNFoundLines().get(),
//                                               mStoreVertexerGPU.getNExclusiveFoundLines().get(),
//                                               mClusters[1].size()));

//   gpu::trackletSelectionKernel<<<blocksGrid, threadsPerBlock>>>(
//     getDeviceContext(),
//     false, // isInitRun
//     mVrtParams.tanLambdaCut,
//     mVrtParams.phiCut);

//   gpuThrowOnError();

// #ifdef _ALLOW_DEBUG_TREES_ITS_
//   if (isDebugFlag(VertexerDebug::TrackletTreeAll)) {
//     mDebugger->fillTrackletSelectionTree(mClusters,
//                                          mStoreVertexerGPU.getRawDupletsFromGPU(gpu::TrackletingLayerOrder::fromInnermostToMiddleLayer),
//                                          mStoreVertexerGPU.getRawDupletsFromGPU(gpu::TrackletingLayerOrder::fromMiddleToOuterLayer),
//                                          mStoreVertexerGPU.getDupletIndicesFromGPU(),
//                                          mEvent);
//   }
//   mTracklets = mStoreVertexerGPU.getLinesFromGPU();
//   if (isDebugFlag(VertexerDebug::LineTreeAll)) {
//     mDebugger->fillPairsInfoTree(mTracklets, mEvent);
//   }
//   if (isDebugFlag(VertexerDebug::LineSummaryAll)) {
//     mDebugger->fillLinesSummaryTree(mTracklets, mEvent);
//   }
// #endif
// }

// void VertexerTraitsGPU::computeVertices()
// {
//   if (!mClusters[1].size()) {
//     return;
//   }
//   const dim3 threadsPerBlock{gpu::utils::host::getBlockSize(mClusters[1].capacity())};
//   const dim3 blocksGrid{gpu::utils::host::getBlocksGrid(threadsPerBlock, mClusters[1].capacity())};
//   size_t bufferSize = mStoreVertexerGPU.getConfig().tmpCUBBufferSize * sizeof(int);
//   int nLines = mStoreVertexerGPU.getNExclusiveFoundLines().getElementFromDevice(mClusters[1].size() - 1) + mStoreVertexerGPU.getNFoundLines().getElementFromDevice(mClusters[1].size() - 1);
//   int nCentroids{static_cast<int>(nLines * (nLines - 1) / 2)};
//   int* histogramXY[2] = {mStoreVertexerGPU.getHistogramXYZ()[0].get(), mStoreVertexerGPU.getHistogramXYZ()[1].get()};
//   float tmpArrayLow[2] = {mStoreVertexerGPU.getConfig().histConf.lowHistBoundariesXYZ[0], mStoreVertexerGPU.getConfig().histConf.lowHistBoundariesXYZ[1]};
//   float tmpArrayHigh[2] = {mStoreVertexerGPU.getConfig().histConf.highHistBoundariesXYZ[0], mStoreVertexerGPU.getConfig().histConf.highHistBoundariesXYZ[1]};
//   gpu::computeCentroidsKernel<<<blocksGrid, threadsPerBlock>>>(getDeviceContext(),
//                                                                mVrtParams.histPairCut);

//   discardResult(cub::DeviceHistogram::MultiHistogramEven<2, 2>(reinterpret_cast<void*>(mStoreVertexerGPU.getCUBTmpBuffer().get()), // d_temp_storage
//                                                                bufferSize,                                                         // temp_storage_bytes
//                                                                mStoreVertexerGPU.getXYCentroids().get(),                           // d_samples
//                                                                histogramXY,                                                        // d_histogram
//                                                                mStoreVertexerGPU.getConfig().histConf.nBinsXYZ,                    // num_levels
//                                                                tmpArrayLow,                                                        // lower_level
//                                                                tmpArrayHigh,                                                       // fupper_level
//                                                                nCentroids));                                                       // num_row_pixels
//   discardResult(cub::DeviceReduce::ArgMax(reinterpret_cast<void*>(mStoreVertexerGPU.getCUBTmpBuffer().get()),
//                                           bufferSize,
//                                           histogramXY[0],
//                                           mStoreVertexerGPU.getTmpVertexPositionBins().get(),
//                                           mStoreVertexerGPU.getConfig().histConf.nBinsXYZ[0]));
//   discardResult(cub::DeviceReduce::ArgMax(reinterpret_cast<void*>(mStoreVertexerGPU.getCUBTmpBuffer().get()),
//                                           bufferSize,
//                                           histogramXY[1],
//                                           mStoreVertexerGPU.getTmpVertexPositionBins().get() + 1,
//                                           mStoreVertexerGPU.getConfig().histConf.nBinsXYZ[0]));
//   gpu::computeZCentroidsKernel<<<blocksGrid, threadsPerBlock>>>(getDeviceContext(), mVrtParams.histPairCut, mStoreVertexerGPU.getConfig().histConf.binSpanXYZ[0], mStoreVertexerGPU.getConfig().histConf.binSpanXYZ[1]);
//   discardResult(cub::DeviceHistogram::HistogramEven(reinterpret_cast<void*>(mStoreVertexerGPU.getCUBTmpBuffer().get()), // d_temp_storage
//                                                     bufferSize,                                                         // temp_storage_bytes
//                                                     mStoreVertexerGPU.getZCentroids().get(),                            // d_samples
//                                                     mStoreVertexerGPU.getHistogramXYZ()[2].get(),                       // d_histogram
//                                                     mStoreVertexerGPU.getConfig().histConf.nBinsXYZ[2],                 // num_levels
//                                                     mStoreVertexerGPU.getConfig().histConf.lowHistBoundariesXYZ[2],     // lower_level
//                                                     mStoreVertexerGPU.getConfig().histConf.highHistBoundariesXYZ[2],    // fupper_level
//                                                     nLines));                                                           // num_row_pixels
//   for (int iVertex{0}; iVertex < mStoreVertexerGPU.getConfig().nMaxVertices; ++iVertex) {
//     discardResult(cub::DeviceReduce::ArgMax(reinterpret_cast<void*>(mStoreVertexerGPU.getCUBTmpBuffer().get()),
//                                             bufferSize,
//                                             mStoreVertexerGPU.getHistogramXYZ()[2].get(),
//                                             mStoreVertexerGPU.getTmpVertexPositionBins().get() + 2,
//                                             mStoreVertexerGPU.getConfig().histConf.nBinsXYZ[2]));
// #ifdef _ALLOW_DEBUG_TREES_ITS_
//     if (isDebugFlag(VertexerDebug::HistCentroids) && !iVertex) {
//       mDebugger->fillXYZHistogramTree(std::array<std::vector<int>, 3>{mStoreVertexerGPU.getHistogramXYFromGPU()[0],
//                                                                       mStoreVertexerGPU.getHistogramXYFromGPU()[1], mStoreVertexerGPU.getHistogramZFromGPU()},
//                                       std::array<int, 3>{mStoreVertexerGPU.getConfig().histConf.nBinsXYZ[0] - 1,
//                                                          mStoreVertexerGPU.getConfig().histConf.nBinsXYZ[1] - 1,
//                                                          mStoreVertexerGPU.getConfig().histConf.nBinsXYZ[2] - 1});
//     }
// #endif
//     gpu::computeVertexKernel<<<blocksGrid, 5>>>(getDeviceContext(), iVertex, mVrtParams.clusterContributorsCut, mStoreVertexerGPU.getConfig().histConf.binSpanXYZ[2]);
//   }
//   std::vector<gpu::GPUVertex> vertices;
//   vertices.resize(mStoreVertexerGPU.getConfig().nMaxVertices);
//   mStoreVertexerGPU.getVertices().copyIntoSizedVector(vertices);

//   for (auto& vertex : vertices) {
//     if (vertex.realVertex) {
//       mVertices.emplace_back(vertex.xCoord, vertex.yCoord, vertex.zCoord, std::array<float, 6>{0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, vertex.contributors, 0.f, -9);
//     }
//   }

//   gpuThrowOnError();
// }

// #ifdef _ALLOW_DEBUG_TREES_ITS_
// void VertexerTraitsGPU::computeMCFiltering()
// {
//   std::vector<Tracklet> tracklets01 = mStoreVertexerGPU.getRawDupletsFromGPU(gpu::TrackletingLayerOrder::fromInnermostToMiddleLayer);
//   std::vector<Tracklet> tracklets12 = mStoreVertexerGPU.getRawDupletsFromGPU(gpu::TrackletingLayerOrder::fromMiddleToOuterLayer);
//   std::vector<int> labels01 = mStoreVertexerGPU.getNFoundTrackletsFromGPU(gpu::TrackletingLayerOrder::fromInnermostToMiddleLayer);
//   std::vector<int> labels12 = mStoreVertexerGPU.getNFoundTrackletsFromGPU(gpu::TrackletingLayerOrder::fromMiddleToOuterLayer);
//   VertexerStoreConfigurationGPU tmpGPUConf;
//   const int stride = tmpGPUConf.maxTrackletsPerCluster;

//   filterTrackletsWithMC(tracklets01, tracklets12, labels01, labels12, stride);
//   mStoreVertexerGPU.updateFoundDuplets(gpu::TrackletingLayerOrder::fromInnermostToMiddleLayer, labels01);
//   mStoreVertexerGPU.updateDuplets(gpu::TrackletingLayerOrder::fromInnermostToMiddleLayer, tracklets01);
//   mStoreVertexerGPU.updateFoundDuplets(gpu::TrackletingLayerOrder::fromMiddleToOuterLayer, labels12);
//   mStoreVertexerGPU.updateDuplets(gpu::TrackletingLayerOrder::fromMiddleToOuterLayer, tracklets12);

//   if (isDebugFlag(VertexerDebug::CombinatoricsTreeAll)) {
//     mDebugger->fillCombinatoricsTree(mClusters,
//                                      mStoreVertexerGPU.getDupletsFromGPU(gpu::TrackletingLayerOrder::fromInnermostToMiddleLayer),
//                                      mStoreVertexerGPU.getDupletsFromGPU(gpu::TrackletingLayerOrder::fromMiddleToOuterLayer),
//                                      mEvent);
//   }
// }
// #endif

VertexerTraits* createVertexerTraitsGPU()
{
  return new VertexerTraitsGPU;
}

// template GPUg() gpu::trackleterKernel<TrackletMode::Layer0Layer1>;
} // namespace its
} // namespace o2