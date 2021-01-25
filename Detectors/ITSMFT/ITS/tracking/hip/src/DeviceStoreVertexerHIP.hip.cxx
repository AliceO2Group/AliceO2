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
/// \file DeviceStoreVertexerHIP.hip.cxx
/// \brief
/// \author matteo.concas@cern.ch

#include <iostream>

#include "ITStrackingHIP/DeviceStoreVertexerHIP.h"
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

DeviceStoreVertexerHIP::DeviceStoreVertexerHIP()
{
  mDuplets01 = VectorHIP<Tracklet>{mGPUConf.dupletsCapacity, mGPUConf.dupletsCapacity};                         // 200 * 4e4 * sizeof(Tracklet) = 128MB
  mDuplets12 = VectorHIP<Tracklet>{mGPUConf.dupletsCapacity, mGPUConf.dupletsCapacity};                         // 200 * 4e4 * sizeof(Tracklet) = 128MB
  mTracklets = VectorHIP<Line>{mGPUConf.processedTrackletsCapacity, mGPUConf.processedTrackletsCapacity};       // 200 * 4e4 * sizeof(Line) = 296MB
  mCUBTmpBuffer = VectorHIP<int>{mGPUConf.tmpCUBBufferSize, mGPUConf.tmpCUBBufferSize};                         // 5e3 * sizeof(int) = 20KB
  mXYCentroids = VectorHIP<float>{2 * mGPUConf.maxCentroidsXYCapacity, 2 * mGPUConf.maxCentroidsXYCapacity};    //
  mZCentroids = VectorHIP<float>{mGPUConf.processedTrackletsCapacity, mGPUConf.processedTrackletsCapacity};     //
  mNFoundLines = VectorHIP<int>{mGPUConf.clustersPerLayerCapacity, mGPUConf.clustersPerLayerCapacity};          // 4e4 * sizeof(int) = 160KB
  mNExclusiveFoundLines = VectorHIP<int>{mGPUConf.clustersPerLayerCapacity, mGPUConf.clustersPerLayerCapacity}; // 4e4 * sizeof(int) = 160KB, tot = <10MB
  mTmpVertexPositionBins = VectorHIP<hipcub::KeyValuePair<int, int>>{3, 3};
  mGPUVertices = VectorHIP<GPUVertex>{mGPUConf.nMaxVertices, mGPUConf.nMaxVertices};
  mBeamPosition = VectorHIP<float>{2, 2};

  for (int iTable{0}; iTable < 2; ++iTable) {
    mIndexTables[iTable] = VectorHIP<int>{constants::its2::ZBins * constants::its2::PhiBins + 1}; // 2*20*20+1 * sizeof(int) = 802B
  }
  for (int iLayer{0}; iLayer < constants::its::LayersNumberVertexer; ++iLayer) { // 4e4 * 3 * sizof(Cluster) = 3.36MB
    mClusters[iLayer] = VectorHIP<Cluster>{mGPUConf.clustersPerLayerCapacity, mGPUConf.clustersPerLayerCapacity};
  }
  for (int iPair{0}; iPair < constants::its::LayersNumberVertexer - 1; ++iPair) {
    mNFoundDuplets[iPair] = VectorHIP<int>{mGPUConf.clustersPerLayerCapacity, mGPUConf.clustersPerLayerCapacity}; // 4e4 * 2 * sizeof(int) = 320KB
  }
  for (int iHisto{0}; iHisto < 3; ++iHisto) {
    mHistogramXYZ[iHisto] = VectorHIP<int>{mGPUConf.histConf.nBinsXYZ[iHisto], mGPUConf.histConf.nBinsXYZ[iHisto]};
  }

#ifdef _ALLOW_DEBUG_TREES_ITS_
  for (int iLayersCouple{0}; iLayersCouple < 2; ++iLayersCouple) {
    mDupletIndices[iLayersCouple] = VectorHIP<int>{mGPUConf.processedTrackletsCapacity, mGPUConf.processedTrackletsCapacity};
  }
  mSizes = VectorHIP<int>{constants::its::LayersNumberVertexer};
#endif
} // namespace gpu

UniquePointer<DeviceStoreVertexerHIP> DeviceStoreVertexerHIP::initialise(const std::array<std::vector<Cluster>, constants::its::LayersNumberVertexer>& clusters,
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

  const dim3 threadsPerBlock{utils::host_hip::getBlockSize(mClusters[1].capacity())};
  const dim3 blocksGrid{utils::host_hip::getBlocksGrid(threadsPerBlock, mClusters[1].capacity())};

  UniquePointer<DeviceStoreVertexerHIP> deviceStoreVertexerPtr{*this};

  hipLaunchKernelGGL((defaultInitArrayKernel), dim3(blocksGrid), dim3(threadsPerBlock), 0, 0, getNFoundTracklets(TrackletingLayerOrder::fromInnermostToMiddleLayer).get(),
                     getNFoundTracklets(TrackletingLayerOrder::fromInnermostToMiddleLayer).capacity(), 0);
  hipLaunchKernelGGL((defaultInitArrayKernel), dim3(blocksGrid), dim3(threadsPerBlock), 0, 0, getNFoundTracklets(TrackletingLayerOrder::fromMiddleToOuterLayer).get(),
                     getNFoundTracklets(TrackletingLayerOrder::fromMiddleToOuterLayer).capacity(), 0);

  return deviceStoreVertexerPtr;
}

GPUd() const VectorHIP<int>& DeviceStoreVertexerHIP::getIndexTable(const VertexerLayerName layer)
{
  if (layer == VertexerLayerName::innermostLayer) {
    return mIndexTables[0];
  }
  return mIndexTables[1];
}

} // namespace gpu
} // namespace its
} // namespace o2
