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
/// \file DeviceStoreVertexerGPU.cu
/// \brief
/// \author matteo.concas@cern.ch

#include <iostream>

#include "ITStrackingCUDA/DeviceStoreVertexerGPU.h"
#include "ITStracking/Configuration.h"

namespace o2
{
namespace its
{
namespace gpu
{
GPUg() void defaultInitArrayKernel(int* array, const size_t arraySize, const int initValue = 0)
{
  for (size_t i{blockIdx.x * blockDim.x + threadIdx.x}; i < arraySize; i += blockDim.x * gridDim.x) {
    if (i < arraySize) {
      array[i] = initValue;
    }
  }
}

DeviceStoreVertexerGPU::DeviceStoreVertexerGPU()
{
  mDuplets01 = Vector<Tracklet>{mGPUConf.dupletsCapacity, mGPUConf.dupletsCapacity};                         // 200 * 4e4 * sizeof(Tracklet) = 128MB
  mDuplets12 = Vector<Tracklet>{mGPUConf.dupletsCapacity, mGPUConf.dupletsCapacity};                         // 200 * 4e4 * sizeof(Tracklet) = 128MB
  mTracklets = Vector<Line>{mGPUConf.processedTrackletsCapacity, mGPUConf.processedTrackletsCapacity};       // 200 * 4e4 * sizeof(Line) = 296MB
  mCUBTmpBuffer = Vector<int>{mGPUConf.tmpCUBBufferSize, mGPUConf.tmpCUBBufferSize};                         // 5e3 * sizeof(int) = 20KB
  mXYCentroids = Vector<float>{2 * mGPUConf.maxCentroidsXYCapacity, 2 * mGPUConf.maxCentroidsXYCapacity};    //
  mZCentroids = Vector<float>{mGPUConf.processedTrackletsCapacity, mGPUConf.processedTrackletsCapacity};     //
  mNFoundLines = Vector<int>{mGPUConf.clustersPerLayerCapacity, mGPUConf.clustersPerLayerCapacity};          // 4e4 * sizeof(int) = 160KB
  mNExclusiveFoundLines = Vector<int>{mGPUConf.clustersPerLayerCapacity, mGPUConf.clustersPerLayerCapacity}; // 4e4 * sizeof(int) = 160KB, tot = <10MB
  mTmpVertexPositionBins = Vector<cub::KeyValuePair<int, int>>{3, 3};
  mGPUVertices = Vector<GPUVertex>{mGPUConf.nMaxVertices, mGPUConf.nMaxVertices};
  mBeamPosition = Vector<float>{2, 2};

  for (int iTable{0}; iTable < 2; ++iTable) {
    mIndexTables[iTable] = Vector<int>{constants::its2::ZBins * constants::its2::PhiBins + 1}; // 2*20*20+1 * sizeof(int) = 802B
  }
  for (int iLayer{0}; iLayer < constants::its::LayersNumberVertexer; ++iLayer) { // 4e4 * 3 * sizof(Cluster) = 3.36MB
    mClusters[iLayer] = Vector<Cluster>{mGPUConf.clustersPerLayerCapacity, mGPUConf.clustersPerLayerCapacity};
  }
  for (int iPair{0}; iPair < constants::its::LayersNumberVertexer - 1; ++iPair) {
    mNFoundDuplets[iPair] = Vector<int>{mGPUConf.clustersPerLayerCapacity, mGPUConf.clustersPerLayerCapacity}; // 4e4 * 2 * sizeof(int) = 320KB
  }
  for (int iHisto{0}; iHisto < 3; ++iHisto) {
    mHistogramXYZ[iHisto] = Vector<int>{mGPUConf.histConf.nBinsXYZ[iHisto], mGPUConf.histConf.nBinsXYZ[iHisto]};
  }

#ifdef _ALLOW_DEBUG_TREES_ITS_
  for (int iLayersCouple{0}; iLayersCouple < 2; ++iLayersCouple) {
    mDupletIndices[iLayersCouple] = Vector<int>{mGPUConf.processedTrackletsCapacity, mGPUConf.processedTrackletsCapacity};
  }
  mSizes = Vector<int>{constants::its::LayersNumberVertexer};
#endif
} // namespace gpu

UniquePointer<DeviceStoreVertexerGPU> DeviceStoreVertexerGPU::initialise(const std::array<std::vector<Cluster>, constants::its::LayersNumberVertexer>& clusters,
                                                                         const std::array<std::array<int, constants::its2::ZBins * constants::its2::PhiBins + 1>,
                                                                                          constants::its::LayersNumberVertexer>& indexTables)
{
#ifdef _ALLOW_DEBUG_TREES_ITS_
  std::array<int, constants::its::LayersNumberVertexer> tmpSizes = {static_cast<int>(clusters[0].size()),
                                                                    static_cast<int>(clusters[1].size()),
                                                                    static_cast<int>(clusters[2].size())};
  mSizes.reset(tmpSizes.data(), static_cast<int>(3));
#endif
  for (int iLayer{0}; iLayer < constants::its::LayersNumberVertexer; ++iLayer) {
    mClusters[iLayer].reset(clusters[iLayer].data(), static_cast<int>(clusters[iLayer].size()));
  }
  mIndexTables[0].reset(indexTables[0].data(), static_cast<int>(indexTables[0].size()));
  mIndexTables[1].reset(indexTables[2].data(), static_cast<int>(indexTables[2].size()));

  const dim3 threadsPerBlock{utils::host::getBlockSize(mClusters[1].capacity())};
  const dim3 blocksGrid{utils::host::getBlocksGrid(threadsPerBlock, mClusters[1].capacity())};

  UniquePointer<DeviceStoreVertexerGPU> deviceStoreVertexerPtr{*this};

  defaultInitArrayKernel<<<blocksGrid, threadsPerBlock>>>(getNFoundTracklets(TrackletingLayerOrder::fromInnermostToMiddleLayer).get(),
                                                          getNFoundTracklets(TrackletingLayerOrder::fromInnermostToMiddleLayer).capacity());
  defaultInitArrayKernel<<<blocksGrid, threadsPerBlock>>>(getNFoundTracklets(TrackletingLayerOrder::fromMiddleToOuterLayer).get(),
                                                          getNFoundTracklets(TrackletingLayerOrder::fromMiddleToOuterLayer).capacity());

  return deviceStoreVertexerPtr;
}

GPUd() const Vector<int>& DeviceStoreVertexerGPU::getIndexTable(const VertexerLayerName layer)
{
  if (layer == VertexerLayerName::innermostLayer) {
    return mIndexTables[0];
  }
  return mIndexTables[1];
}

} // namespace gpu
} // namespace its
} // namespace o2
