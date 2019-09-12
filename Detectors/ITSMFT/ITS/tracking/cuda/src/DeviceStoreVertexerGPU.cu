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
  for (int iTable{0}; iTable < 2; ++iTable) {
    mIndexTables[iTable] = Vector<int>{constants::index_table::ZBins * constants::index_table::PhiBins + 1};
  }
  for (int iLayer{0}; iLayer < constants::its::LayersNumberVertexer; ++iLayer) {
    mClusters[iLayer] = Vector<Cluster>{mGPUConf.clustersPerLayerCapacity};
  }
  for (int iPair{0}; iPair < constants::its::LayersNumberVertexer - 1; ++iPair) {
    mNFoundTracklets[iPair] = Vector<int>{mGPUConf.clustersPerLayerCapacity};
  }
  mNFoundLines = Vector<int>{mGPUConf.clustersPerLayerCapacity};
}

UniquePointer<DeviceStoreVertexerGPU> DeviceStoreVertexerGPU::initialise(const std::array<std::vector<Cluster>, constants::its::LayersNumberVertexer>& clusters,
                                                                         const std::array<std::array<int, constants::index_table::ZBins * constants::index_table::PhiBins + 1>,
                                                                                          constants::its::LayersNumberVertexer>& indexTables)
{
  for (int iLayer{0}; iLayer < constants::its::LayersNumberVertexer; ++iLayer) {
    mSizes[iLayer] = clusters[iLayer].size();
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

std::vector<Tracklet> DeviceStoreVertexerGPU::getDupletsfromGPU(const TrackletingLayerOrder order)
{
  // Danger, really large allocations, use debug-purpose only.
  std::vector<Tracklet> tmpDuplets{static_cast<size_t>(mGPUConf.dupletsCapacity)};
  std::vector<int> nFoundDuplets{mSizes[1]};
  std::vector<Tracklet> shrinkedDuplets{2000};

  if (order == GPU::TrackletingLayerOrder::fromInnermostToMiddleLayer) {
    mNFoundTracklets[0].copyIntoVector(nFoundDuplets, mSizes[1]);
    mDuplets01.copyIntoVector(tmpDuplets, tmpDuplets.size());
  } else {
    mDuplets12.copyIntoVector(tmpDuplets, tmpDuplets.size());
    mNFoundTracklets[1].copyIntoVector(nFoundDuplets, mSizes[1]);
  }

  for (int iCluster{0}; iCluster < mSizes[1]; ++iCluster) {
    const int stride{iCluster * mGPUConf.maxTrackletsPerCluster};
    for (int iDuplet{0}; iDuplet < nFoundDuplets[iCluster]; ++iDuplet) {
      shrinkedDuplets.push_back(tmpDuplets[stride + iDuplet]);
    }
  }
  return shrinkedDuplets;
}

std::vector<Line> DeviceStoreVertexerGPU::getLinesfromGPU()
{
  // Danger, really large allocations, use debug-purpose only.
  std::vector<Line> tmpLines{static_cast<size_t>(mGPUConf.processedTrackletsCapacity)};
  std::vector<int> nFoundLines{mSizes[1]};
  std::vector<Line> shrinkedLines{1000};

  mNFoundLines.copyIntoVector(nFoundLines, mSizes[1]);

  for (int iCluster{0}; iCluster < mSizes[1]; ++iCluster) {
    const int stride{iCluster * mGPUConf.maxTrackletsPerCluster};
    for (int iLine{0}; iLine < nFoundLines[iCluster]; ++iLine) {
      shrinkedLines.push_back(tmpLines[stride + iLine]);
    }
  }
  return shrinkedLines;
}

} // namespace GPU
} // namespace its
} // namespace o2
