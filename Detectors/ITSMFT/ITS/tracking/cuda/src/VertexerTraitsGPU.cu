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
#ifdef _ALLOW_DEBUG_TREES_ITS_
  mDebugger = new StandaloneDebugger::StandaloneDebugger("dbg_ITSVertexerGPU.root");
#endif
}

#ifdef _ALLOW_DEBUG_TREES_ITS_
VertexerTraitsGPU::~VertexerTraitsGPU()
{
  delete mDebugger;
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
GPUd() int getGlobalIdx()
{
  int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
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
  const float tanLambdaCut = 0.025f,
  const float phiCut = 0.002f)
{
  const int currentClusterIndex{getGlobalIdx()};
  if (currentClusterIndex < store.getClusters()[1].size()) {
    const int stride{currentClusterIndex * store.getConfig().maxTrackletsPerCluster};
    int validTracklets{0};
    for (int iTracklet12{0}; iTracklet12 < store.getNFoundTracklets(TrackletingLayerOrder::fromMiddleToOuterLayer)[currentClusterIndex]; ++iTracklet12) {
      for (int iTracklet01{0}; iTracklet01 < store.getNFoundTracklets(TrackletingLayerOrder::fromInnermostToMiddleLayer)[currentClusterIndex] && validTracklets < store.getConfig().maxTrackletsPerCluster; ++iTracklet01) {
        const float deltaTanLambda{gpu::GPUCommonMath::Abs(store.getDuplets01()[stride + iTracklet01].tanLambda - store.getDuplets12()[stride + iTracklet12].tanLambda)};
        const float deltaPhi{gpu::GPUCommonMath::Abs(store.getDuplets01()[stride + iTracklet01].phiCoordinate - store.getDuplets12()[stride + iTracklet12].phiCoordinate)};
        if (deltaTanLambda < tanLambdaCut && deltaPhi < phiCut && validTracklets != store.getConfig().maxTrackletsPerCluster) {
          store.getLines().emplace(stride + validTracklets, store.getDuplets01()[stride + iTracklet01], store.getClusters()[0].get(), store.getClusters()[1].get());
#ifdef _ALLOW_DEBUG_TREES_ITS_
          store.getDupletIndices()[0].emplace(stride + validTracklets, stride + iTracklet01);
          store.getDupletIndices()[1].emplace(stride + validTracklets, stride + iTracklet12);
#endif
          ++validTracklets;
        }
      }
    }
    store.getNFoundLines().emplace(currentClusterIndex, validTracklets);
    if (validTracklets != store.getConfig().maxTrackletsPerCluster) {
      store.getLines().emplace(stride + validTracklets);
    } else {
      printf("info: fulfilled all the space with tracklets\n");
    }
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

  GPU::trackletSelectionKernel<<<1, threadsPerBlock>>>(
    getDeviceContext(),
    mVrtParams.tanLambdaCut,
    mVrtParams.phiCut);

  cudaError_t error = cudaGetLastError();

  if (error != cudaSuccess) {
    std::ostringstream errorString{};
    errorString << "CUDA API returned error  [" << cudaGetErrorString(error) << "] (code " << error << ")" << std::endl;
    throw std::runtime_error{errorString.str()};
  }
#ifdef _ALLOW_DEBUG_TREES_ITS_
  if (isDebugFlag(VertexerDebug::TrackletTreeAll)) {
    mDebugger->fillStridedTrackletSelectionTree(mClusters,
                                                mStoreVertexerGPU.getRawDupletsFromGPU(GPU::TrackletingLayerOrder::fromInnermostToMiddleLayer),
                                                mStoreVertexerGPU.getRawDupletsFromGPU(GPU::TrackletingLayerOrder::fromMiddleToOuterLayer),
                                                mStoreVertexerGPU.getDupletIndicesFromGPU(),
                                                mEvent);
  }
  mTracklets = mStoreVertexerGPU.getLinesFromGPU();
  if (isDebugFlag(VertexerDebug::LineTreeAll)) {
    mDebugger->fillLinesInfoTree(mTracklets, mEvent);
  }
  if (isDebugFlag(VertexerDebug::LineSummaryAll)) {
    mDebugger->fillLinesSummaryTree(mTracklets, mEvent);
  }
}

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
