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

#include "ITStrackingCUDA/DeviceStoreVertexerGPU.h"
#include "ITStracking/Configuration.h"
#include <iostream>

namespace o2
{
namespace its
{
namespace GPU
{
template <class T>
GPUg() void defaultInitArrayKernel(T* array, const size_t arraySize)
{
  for (size_t i{blockIdx.x * blockDim.x + threadIdx.x}; i < arraySize; i += blockDim.x * gridDim.x) {
    if (i < arraySize) {
      array[i] = T{};
    }
  }
}

DeviceStoreVertexerGPU::DeviceStoreVertexerGPU()
{
  mDuplets01 = Vector<Tracklet>{mGPUConf.dupletsCapacity, mGPUConf.dupletsCapacity};
  mDuplets12 = Vector<Tracklet>{mGPUConf.dupletsCapacity, mGPUConf.dupletsCapacity};
  mTracklets = Vector<Line>{mGPUConf.processedTrackletsCapacity, mGPUConf.processedTrackletsCapacity};
#ifdef _ALLOW_DEBUG_TREES_ITS_
  for (int iLayersCouple{0}; iLayersCouple < 2; ++iLayersCouple) {
    mDupletIndices[iLayersCouple] = Vector<int>{mGPUConf.processedTrackletsCapacity, mGPUConf.processedTrackletsCapacity};
  }
#endif
  for (int iTable{0}; iTable < 2; ++iTable) {
    mIndexTables[iTable] = Vector<int>{constants::index_table::ZBins * constants::index_table::PhiBins + 1};
  }
  for (int iLayer{0}; iLayer < constants::its::LayersNumberVertexer; ++iLayer) {
    mClusters[iLayer] = Vector<Cluster>{mGPUConf.clustersPerLayerCapacity};
  }
  for (int iPair{0}; iPair < constants::its::LayersNumberVertexer - 1; ++iPair) {
    mNFoundDuplets[iPair] = Vector<int>{mGPUConf.clustersPerLayerCapacity};
  }
  mNFoundLines = Vector<int>{mGPUConf.clustersPerLayerCapacity};
  mSizes = Vector<int>{constants::its::LayersNumberVertexer};
}

UniquePointer<DeviceStoreVertexerGPU> DeviceStoreVertexerGPU::initialise(const std::array<std::vector<Cluster>, constants::its::LayersNumberVertexer>& clusters,
                                                                         const std::array<std::array<int, constants::index_table::ZBins * constants::index_table::PhiBins + 1>,
                                                                                          constants::its::LayersNumberVertexer>& indexTables)
{
  std::array<int, constants::its::LayersNumberVertexer> tmpSizes = {static_cast<int>(clusters[0].size()),
                                                                    static_cast<int>(clusters[1].size()),
                                                                    static_cast<int>(clusters[2].size())};
  mSizes.reset(tmpSizes.data(), constants::its::LayersNumberVertexer);
  for (int iLayer{0}; iLayer < constants::its::LayersNumberVertexer; ++iLayer) {
    mClusters[iLayer].reset(clusters[iLayer].data(), static_cast<int>(clusters[iLayer].size()));
  }
  mIndexTables[0].reset(indexTables[0].data(), static_cast<int>(indexTables[0].size()));
  mIndexTables[1].reset(indexTables[2].data(), static_cast<int>(indexTables[2].size()));

  const dim3 threadsPerBlock{Utils::Host::getBlockSize(mClusters[1].capacity())};
  const dim3 blocksGrid{Utils::Host::getBlocksGrid(threadsPerBlock, mClusters[1].capacity())};

  UniquePointer<DeviceStoreVertexerGPU> deviceStoreVertexerPtr{*this};

  defaultInitArrayKernel<int><<<blocksGrid, threadsPerBlock>>>(getNFoundTracklets(TrackletingLayerOrder::fromInnermostToMiddleLayer).get(),
                                                               getNFoundTracklets(TrackletingLayerOrder::fromInnermostToMiddleLayer).capacity());
  defaultInitArrayKernel<int><<<blocksGrid, threadsPerBlock>>>(getNFoundTracklets(TrackletingLayerOrder::fromMiddleToOuterLayer).get(),
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

} // namespace GPU
} // namespace its
} // namespace o2
