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
#include <cub/cub.cuh>

#include "ITStracking/MathUtils.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/Tracklet.h"

#include "ITStrackingCUDA/Utils.h"
#include "ITStrackingCUDA/ClusterLinesGPU.h"
#include "ITStrackingCUDA/Context.h"
#include "ITStrackingCUDA/Stream.h"
#include "ITStrackingCUDA/VertexerTraitsGPU.h"

namespace o2
{
namespace its
{

using constants::index_table::PhiBins;
using constants::index_table::ZBins;
using constants::its::LayersRCoordinate;
using constants::its::LayersZCoordinate;
using constants::its::VertexerHistogramVolume;
using constants::math::TwoPi;
using index_table_utils::getPhiBinIndex;
using index_table_utils::getZBinIndex;
using math_utils::getNormalizedPhiCoordinate;

GPUh() void gpuThrowOnError()
{
  cudaError_t error = cudaGetLastError();

  if (error != cudaSuccess) {
    std::ostringstream errorString{};
    errorString << "CUDA API returned error  [" << cudaGetErrorString(error) << "] (code " << error << ")" << std::endl;
    throw std::runtime_error{errorString.str()};
  }
}

#ifdef _ALLOW_DEBUG_TREES_ITS_
VertexerTraitsGPU::VertexerTraitsGPU()
{
  setIsGPU(true);
  std::cout << "[DEBUG] Creating file: dbg_ITSVertexerGPU.root" << std::endl;
  mDebugger = new StandaloneDebugger::StandaloneDebugger("dbg_ITSVertexerGPU.root");
}

VertexerTraitsGPU::~VertexerTraitsGPU()
{
  delete mDebugger;
}
#else
VertexerTraitsGPU::VertexerTraitsGPU()
{
  setIsGPU(true);
}

#endif

void VertexerTraitsGPU::initialise(ROframe* event)
{
  reset();
  arrangeClusters(event);
  mStoreVertexerGPUPtr = mStoreVertexerGPU.initialise(mClusters, mIndexTables);
}

namespace GPU
{

template <typename... Args>
GPUd() void printOnThread(const int tId, const char* str, Args... args)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == tId) {
    printf(str, args...);
  }
}

GPUd() void printVectorOnThread(const char* name, Vector<int>& vector, size_t size, const int tId = 0)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == tId) {
    printf("vector %s :", name);
    for (int i{0}; i < size; ++i) {
      printf("%d ", vector[i]);
    }
    printf("\n");
  }
}

GPUg() void printVectorKernel(DeviceStoreVertexerGPU& store, const int threadId)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == threadId) {
    for (int i{0}; i < 100; ++i) {
      printf("%d ", store.getHistogramXYZ()[0].get()[i]);
    }
    printf("\n");
    for (int i{0}; i < 100; ++i) {
      printf("%d ", store.getHistogramXYZ()[1].get()[i]);
    }
    printf("\n");
    for (int i{0}; i < 800; ++i) {
      printf("%d ", store.getHistogramXYZ()[2].get()[i]);
    }
    printf("\n");
  }
}

GPUg() void dumpMaximaKernel(DeviceStoreVertexerGPU& store, const int threadId)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == threadId) {
    printf("XmaxBin: %d at index: %d | YmaxBin: %d at index: %d | ZmaxBin: %d at index: %d\n",
           store.getTmpVertexPositionBins()[0].value, store.getTmpVertexPositionBins()[0].key,
           store.getTmpVertexPositionBins()[1].value, store.getTmpVertexPositionBins()[1].key,
           store.getTmpVertexPositionBins()[2].value, store.getTmpVertexPositionBins()[2].key);
  }
}

GPUg() void trackleterKernel(
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

GPUg() void trackletSelectionKernel(
  DeviceStoreVertexerGPU& store,
  const unsigned char isInitRun = false,
  const float tanLambdaCut = 0.025f,
  const float phiCut = 0.002f)
{
  const size_t nClustersMiddleLayer = store.getClusters()[1].size();
  for (size_t currentClusterIndex = blockIdx.x * blockDim.x + threadIdx.x; currentClusterIndex < nClustersMiddleLayer; currentClusterIndex += blockDim.x * gridDim.x) {
    const int stride{static_cast<int>(currentClusterIndex * store.getConfig().maxTrackletsPerCluster)};
    int validTracklets{0};
    for (int iTracklet12{0}; iTracklet12 < store.getNFoundTracklets(TrackletingLayerOrder::fromMiddleToOuterLayer)[currentClusterIndex]; ++iTracklet12) {
      for (int iTracklet01{0}; iTracklet01 < store.getNFoundTracklets(TrackletingLayerOrder::fromInnermostToMiddleLayer)[currentClusterIndex] && validTracklets < store.getConfig().maxTrackletsPerCluster; ++iTracklet01) {
        const float deltaTanLambda{gpu::GPUCommonMath::Abs(store.getDuplets01()[stride + iTracklet01].tanLambda - store.getDuplets12()[stride + iTracklet12].tanLambda)};
        const float deltaPhi{gpu::GPUCommonMath::Abs(store.getDuplets01()[stride + iTracklet01].phiCoordinate - store.getDuplets12()[stride + iTracklet12].phiCoordinate)};
        if (deltaTanLambda < tanLambdaCut && deltaPhi < phiCut && validTracklets != store.getConfig().maxTrackletsPerCluster) {
          assert(store.getDuplets01()[stride + iTracklet01].secondClusterIndex == store.getDuplets12()[stride + iTracklet12].firstClusterIndex);
          if (!isInitRun) {
            store.getLines().emplace(store.getNExclusiveFoundLines()[currentClusterIndex] + validTracklets, store.getDuplets01()[stride + iTracklet01], store.getClusters()[0].get(), store.getClusters()[1].get());
#ifdef _ALLOW_DEBUG_TREES_ITS_
            store.getDupletIndices()[0].emplace(store.getNExclusiveFoundLines()[currentClusterIndex] + validTracklets, stride + iTracklet01);
            store.getDupletIndices()[1].emplace(store.getNExclusiveFoundLines()[currentClusterIndex] + validTracklets, stride + iTracklet12);
#endif
          }
          ++validTracklets;
        }
      }
    }
    if (isInitRun) {
      store.getNFoundLines().emplace(currentClusterIndex, validTracklets);
      if (validTracklets >= store.getConfig().maxTrackletsPerCluster) {
        printf("Warning: not enough space for tracklet selection, some lines will be left behind\n");
      }
    }
  }
  if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
    // first thread I want to write an empty line after last found, as adebug flag. Might delete later
    store.getLines().emplace(store.getNExclusiveFoundLines()[store.getClusters()[1].size() - 1] + store.getNFoundLines()[store.getClusters()[1].size() - 1]);
  }
}

GPUg() void computeCentroidsKernel(DeviceStoreVertexerGPU& store,
                                   const float pairCut)
{
  const int nLines = store.getNExclusiveFoundLines()[store.getClusters()[1].size() - 1] + store.getNFoundLines()[store.getClusters()[1].size() - 1];
  const int maxIterations{nLines * (nLines - 1) / 2};
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

        return;
      }
    }
    // writing some data anyway outside the histogram, they will not be put in the histogram, by construction.
    store.getXYCentroids().emplace(2 * currentThreadIndex, 2 * store.getConfig().lowHistBoundariesXYZ[0]);
    store.getXYCentroids().emplace(2 * currentThreadIndex + 1, 2 * store.getConfig().lowHistBoundariesXYZ[1]);
  }
}

GPUg() void computeZCentroidsKernel(DeviceStoreVertexerGPU& store,
                                    const float pairCut, const int binOpeningX, const int binOpeningY)
{
  const int nLines = store.getNExclusiveFoundLines()[store.getClusters()[1].size() - 1] + store.getNFoundLines()[store.getClusters()[1].size() - 1];
  for (size_t currentThreadIndex = blockIdx.x * blockDim.x + threadIdx.x; currentThreadIndex < nLines; currentThreadIndex += blockDim.x * gridDim.x) {
    if (store.getTmpVertexPositionBins()[0].value || store.getTmpVertexPositionBins()[1].value) {
      float tmpX{store.getConfig().lowHistBoundariesXYZ[0] + store.getTmpVertexPositionBins()[0].key * store.getConfig().binSizeHistX + store.getConfig().binSizeHistX / 2};
      int sumWX{store.getTmpVertexPositionBins()[0].value};
      float wX{tmpX * store.getTmpVertexPositionBins()[0].value};
      for (int iBin{gpu::GPUCommonMath::Max(0, store.getTmpVertexPositionBins()[0].key - binOpeningX)}; iBin < gpu::GPUCommonMath::Min(store.getTmpVertexPositionBins()[0].key + binOpeningX + 1, store.getConfig().nBinsXYZ[0] - 1); ++iBin) {
        if (iBin != store.getTmpVertexPositionBins()[0].key) {
          wX += (store.getConfig().lowHistBoundariesXYZ[0] + iBin * store.getConfig().binSizeHistX + store.getConfig().binSizeHistX / 2) * store.getHistogramXYZ()[0].get()[iBin];
          sumWX += store.getHistogramXYZ()[0].get()[iBin];
        }
      }
      float tmpY{store.getConfig().lowHistBoundariesXYZ[1] + store.getTmpVertexPositionBins()[1].key * store.getConfig().binSizeHistY + store.getConfig().binSizeHistY / 2};
      int sumWY{store.getTmpVertexPositionBins()[1].value};
      float wY{tmpY * store.getTmpVertexPositionBins()[1].value};
      for (int iBin{gpu::GPUCommonMath::Max(0, store.getTmpVertexPositionBins()[1].key - binOpeningY)}; iBin < gpu::GPUCommonMath::Min(store.getTmpVertexPositionBins()[1].key + binOpeningY + 1, store.getConfig().nBinsXYZ[1] - 1); ++iBin) {
        if (iBin != store.getTmpVertexPositionBins()[1].key) {
          wY += (store.getConfig().lowHistBoundariesXYZ[1] + iBin * store.getConfig().binSizeHistY + store.getConfig().binSizeHistY / 2) * store.getHistogramXYZ()[1].get()[iBin];
          sumWY += store.getHistogramXYZ()[1].get()[iBin];
        }
      }
      store.getBeamPosition().emplace(0, wX / sumWX);
      store.getBeamPosition().emplace(1, wY / sumWY);
      float fakeBeamPoint1[3] = {store.getBeamPosition()[0], store.getBeamPosition()[1], -1}; // get two points laying at different z, to create line object
      float fakeBeamPoint2[3] = {store.getBeamPosition()[0], store.getBeamPosition()[1], 1};
      Line pseudoBeam = Line::Line(fakeBeamPoint1, fakeBeamPoint2);
      if (Line::getDCA(store.getLines()[currentThreadIndex], pseudoBeam) < pairCut) {
        ClusterLinesGPU cluster{store.getLines()[currentThreadIndex], pseudoBeam};
        store.getZCentroids().emplace(currentThreadIndex, cluster.getVertex()[2]);
      } else {
        store.getZCentroids().emplace(currentThreadIndex, 2 * store.getConfig().lowHistBoundariesXYZ[2]);
      }
    }
  }
}

GPUg() void computeVertexKernel(DeviceStoreVertexerGPU& store, const int vertIndex, const int minContributors, const int binOpeningZ)
{
  for (size_t currentThreadIndex = blockIdx.x * blockDim.x + threadIdx.x; currentThreadIndex < binOpeningZ; currentThreadIndex += blockDim.x * gridDim.x) {
    if (currentThreadIndex == 0) {
      if (store.getTmpVertexPositionBins()[2].value > 1) {
        float z{store.getConfig().lowHistBoundariesXYZ[2] + store.getTmpVertexPositionBins()[2].key * store.getConfig().binSizeHistZ + store.getConfig().binSizeHistZ / 2};
        float ex{0.f};
        float ey{0.f};
        float ez{0.f};
        int sumWZ{store.getTmpVertexPositionBins()[2].value};
        float wZ{z * store.getTmpVertexPositionBins()[2].value};
        for (int iBin{gpu::GPUCommonMath::Max(0, store.getTmpVertexPositionBins()[2].key - binOpeningZ)}; iBin < gpu::GPUCommonMath::Min(store.getTmpVertexPositionBins()[2].key + binOpeningZ + 1, store.getConfig().nBinsXYZ[2] - 1); ++iBin) {
          if (iBin != store.getTmpVertexPositionBins()[2].key) {
            wZ += (store.getConfig().lowHistBoundariesXYZ[2] + iBin * store.getConfig().binSizeHistZ + store.getConfig().binSizeHistZ / 2) * store.getHistogramXYZ()[2].get()[iBin];
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
} // namespace GPU

void VertexerTraitsGPU::computeTracklets()
{
  const dim3 threadsPerBlock{GPU::Utils::Host::getBlockSize(mClusters[1].capacity())};
  const dim3 blocksGrid{GPU::Utils::Host::getBlocksGrid(threadsPerBlock, mClusters[1].capacity())};

  GPU::trackleterKernel<<<blocksGrid, threadsPerBlock>>>(
    getDeviceContext(),
    GPU::TrackletingLayerOrder::fromInnermostToMiddleLayer,
    mVrtParams.phiCut);

  GPU::trackleterKernel<<<blocksGrid, threadsPerBlock>>>(
    getDeviceContext(),
    GPU::TrackletingLayerOrder::fromMiddleToOuterLayer,
    mVrtParams.phiCut);

  gpuThrowOnError();

#ifdef _ALLOW_DEBUG_TREES_ITS_
  if (isDebugFlag(VertexerDebug::CombinatoricsTreeAll)) {
    mDebugger->fillCombinatoricsTree(mStoreVertexerGPU.getDupletsFromGPU(GPU::TrackletingLayerOrder::fromInnermostToMiddleLayer),
                                     mStoreVertexerGPU.getDupletsFromGPU(GPU::TrackletingLayerOrder::fromMiddleToOuterLayer));
  }
#endif
}

void VertexerTraitsGPU::computeTrackletMatching()
{
  const dim3 threadsPerBlock{GPU::Utils::Host::getBlockSize(mClusters[1].capacity())};
  const dim3 blocksGrid{GPU::Utils::Host::getBlocksGrid(threadsPerBlock, mClusters[1].capacity())};
  size_t bufferSize = mStoreVertexerGPU.getConfig().tmpCUBBufferSize * sizeof(int);

  GPU::trackletSelectionKernel<<<blocksGrid, threadsPerBlock>>>(
    getDeviceContext(),
    true, // isInitRun
    mVrtParams.tanLambdaCut,
    mVrtParams.phiCut);

  cub::DeviceScan::ExclusiveSum(reinterpret_cast<void*>(mStoreVertexerGPU.getCUBTmpBuffer().get()),
                                bufferSize,
                                mStoreVertexerGPU.getNFoundLines().get(),
                                mStoreVertexerGPU.getNExclusiveFoundLines().get(),
                                mClusters[1].size());

  GPU::trackletSelectionKernel<<<blocksGrid, threadsPerBlock>>>(
    getDeviceContext(),
    false, // isInitRun
    mVrtParams.tanLambdaCut,
    mVrtParams.phiCut);

  gpuThrowOnError();

#ifdef _ALLOW_DEBUG_TREES_ITS_
  if (isDebugFlag(VertexerDebug::TrackletTreeAll)) {
    mDebugger->fillTrackletSelectionTree(mClusters,
                                         mStoreVertexerGPU.getRawDupletsFromGPU(GPU::TrackletingLayerOrder::fromInnermostToMiddleLayer),
                                         mStoreVertexerGPU.getRawDupletsFromGPU(GPU::TrackletingLayerOrder::fromMiddleToOuterLayer),
                                         mStoreVertexerGPU.getDupletIndicesFromGPU(),
                                         mEvent);
  }
  mTracklets = mStoreVertexerGPU.getLinesFromGPU();
  if (isDebugFlag(VertexerDebug::LineTreeAll)) {
    mDebugger->fillPairsInfoTree(mTracklets, mEvent);
  }
  if (isDebugFlag(VertexerDebug::LineSummaryAll)) {
    mDebugger->fillLinesSummaryTree(mTracklets, mEvent);
  }
#endif
}

void VertexerTraitsGPU::computeVertices()
{
  const dim3 threadsPerBlock{GPU::Utils::Host::getBlockSize(mClusters[1].capacity())};
  const dim3 blocksGrid{GPU::Utils::Host::getBlocksGrid(threadsPerBlock, mClusters[1].capacity())};
  size_t bufferSize = mStoreVertexerGPU.getConfig().tmpCUBBufferSize * sizeof(int);
  int nLines = mStoreVertexerGPU.getNExclusiveFoundLines().getElementFromDevice(mClusters[1].size() - 1) + mStoreVertexerGPU.getNFoundLines().getElementFromDevice(mClusters[1].size() - 1);
  int nCentroids{static_cast<int>(nLines * (nLines - 1) / 2)};
  int* histogramXY[2] = {mStoreVertexerGPU.getHistogramXYZ()[0].get(), mStoreVertexerGPU.getHistogramXYZ()[1].get()};

  GPU::computeCentroidsKernel<<<blocksGrid, threadsPerBlock>>>(getDeviceContext(),
                                                               mVrtParams.histPairCut);

  cub::DeviceHistogram::MultiHistogramEven<2, 2>(reinterpret_cast<void*>(mStoreVertexerGPU.getCUBTmpBuffer().get()), // d_temp_storage
                                                 bufferSize,                                                         // temp_storage_bytes
                                                 mStoreVertexerGPU.getXYCentroids().get(),                           // d_samples
                                                 histogramXY,                                                        // d_histogram
                                                 mStoreVertexerGPU.getConfig().nBinsXYZ,                             // num_levels
                                                 mStoreVertexerGPU.getConfig().lowHistBoundariesXYZ,                 // lower_level
                                                 mStoreVertexerGPU.getConfig().highHistBoundariesXYZ,                // fupper_level
                                                 nCentroids);                                                        // num_row_pixels

  cub::DeviceReduce::ArgMax(reinterpret_cast<void*>(mStoreVertexerGPU.getCUBTmpBuffer().get()),
                            bufferSize,
                            histogramXY[0],
                            mStoreVertexerGPU.getTmpVertexPositionBins().get(),
                            mStoreVertexerGPU.getConfig().nBinsXYZ[0]);
  cub::DeviceReduce::ArgMax(reinterpret_cast<void*>(mStoreVertexerGPU.getCUBTmpBuffer().get()),
                            bufferSize,
                            histogramXY[1],
                            mStoreVertexerGPU.getTmpVertexPositionBins().get() + 1,
                            mStoreVertexerGPU.getConfig().nBinsXYZ[0]);

  GPU::computeZCentroidsKernel<<<blocksGrid, threadsPerBlock>>>(getDeviceContext(), mVrtParams.histPairCut, mStoreVertexerGPU.getConfig().binSpanXYZ[0], mStoreVertexerGPU.getConfig().binSpanXYZ[1]);

  cub::DeviceHistogram::HistogramEven(reinterpret_cast<void*>(mStoreVertexerGPU.getCUBTmpBuffer().get()), // d_temp_storage
                                      bufferSize,                                                         // temp_storage_bytes
                                      mStoreVertexerGPU.getZCentroids().get(),                            // d_samples
                                      mStoreVertexerGPU.getHistogramXYZ()[2].get(),                       // d_histogram
                                      mStoreVertexerGPU.getConfig().nBinsXYZ[2],                          // num_levels
                                      mStoreVertexerGPU.getConfig().lowHistBoundariesXYZ[2],              // lower_level
                                      mStoreVertexerGPU.getConfig().highHistBoundariesXYZ[2],             // fupper_level
                                      nLines);                                                            // num_row_pixels

  for (int iVertex{0}; iVertex < mStoreVertexerGPU.getConfig().nMaxVertices; ++iVertex) {
    cub::DeviceReduce::ArgMax(reinterpret_cast<void*>(mStoreVertexerGPU.getCUBTmpBuffer().get()),
                              bufferSize,
                              mStoreVertexerGPU.getHistogramXYZ()[2].get(),
                              mStoreVertexerGPU.getTmpVertexPositionBins().get() + 2,
                              mStoreVertexerGPU.getConfig().nBinsXYZ[2]);
    // GPU::dumpMaximaKernel<<<blocksGrid, threadsPerBlock>>>(getDeviceContext(), 0);
#ifdef _ALLOW_DEBUG_TREES_ITS_
    if (isDebugFlag(VertexerDebug::HistCentroids) && !iVertex) {
      mDebugger->fillXYZHistogramTree(std::array<std::vector<int>, 3>{mStoreVertexerGPU.getHistogramXYFromGPU()[0],
                                                                      mStoreVertexerGPU.getHistogramXYFromGPU()[1], mStoreVertexerGPU.getHistogramZFromGPU()},
                                      std::array<int, 3>{mStoreVertexerGPU.getConfig().nBinsXYZ[0] - 1,
                                                         mStoreVertexerGPU.getConfig().nBinsXYZ[1] - 1,
                                                         mStoreVertexerGPU.getConfig().nBinsXYZ[2] - 1});
    }
#endif
    GPU::computeVertexKernel<<<blocksGrid, 5>>>(getDeviceContext(), iVertex, mVrtParams.clusterContributorsCut, mStoreVertexerGPU.getConfig().binSpanXYZ[2]);
  }
  std::vector<GPU::GPUVertex> vertices;
  vertices.resize(mStoreVertexerGPU.getConfig().nMaxVertices);
  mStoreVertexerGPU.getVertices().copyIntoSizedVector(vertices);

  for (auto& vertex : vertices) {
    if (vertex.realVertex) {
      mVertices.emplace_back(vertex.xCoord, vertex.yCoord, vertex.zCoord, std::array<float, 6>{0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, vertex.contributors, 0.f, -9);
    }
  }

  gpuThrowOnError();
}

#ifdef _ALLOW_DEBUG_TREES_ITS_
void VertexerTraitsGPU::computeMCFiltering()
{
  std::vector<Tracklet> tracklets01 = mStoreVertexerGPU.getRawDupletsFromGPU(GPU::TrackletingLayerOrder::fromInnermostToMiddleLayer);
  std::vector<Tracklet> tracklets12 = mStoreVertexerGPU.getRawDupletsFromGPU(GPU::TrackletingLayerOrder::fromMiddleToOuterLayer);
  std::vector<int> labels01 = mStoreVertexerGPU.getNFoundTrackletsFromGPU(GPU::TrackletingLayerOrder::fromInnermostToMiddleLayer);
  std::vector<int> labels12 = mStoreVertexerGPU.getNFoundTrackletsFromGPU(GPU::TrackletingLayerOrder::fromMiddleToOuterLayer);
  VertexerStoreConfigurationGPU tmpGPUConf;
  const int stride = tmpGPUConf.maxTrackletsPerCluster;

  filterTrackletsWithMC(tracklets01, tracklets12, labels01, labels12, stride);
  mStoreVertexerGPU.updateFoundDuplets(GPU::TrackletingLayerOrder::fromInnermostToMiddleLayer, labels01);
  mStoreVertexerGPU.updateDuplets(GPU::TrackletingLayerOrder::fromInnermostToMiddleLayer, tracklets01);
  mStoreVertexerGPU.updateFoundDuplets(GPU::TrackletingLayerOrder::fromMiddleToOuterLayer, labels12);
  mStoreVertexerGPU.updateDuplets(GPU::TrackletingLayerOrder::fromMiddleToOuterLayer, tracklets12);

  if (isDebugFlag(VertexerDebug::CombinatoricsTreeAll)) {
    mDebugger->fillCombinatoricsMCTree(mStoreVertexerGPU.getDupletsFromGPU(GPU::TrackletingLayerOrder::fromInnermostToMiddleLayer),
                                       mStoreVertexerGPU.getDupletsFromGPU(GPU::TrackletingLayerOrder::fromMiddleToOuterLayer));
  }
}
#endif

VertexerTraits* createVertexerTraitsGPU()
{
  return new VertexerTraitsGPU;
}

} // namespace its
} // namespace o2