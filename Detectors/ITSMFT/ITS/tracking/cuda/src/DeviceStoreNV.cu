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
/// \file DeviceStoreNV.cxx
/// \brief
///

#include "ITStrackingCUDA/DeviceStoreNV.h"

#include <sstream>

#include "ITStrackingCUDA/Stream.h"

namespace {

using namespace o2::its;

__device__ void fillIndexTables(GPU::DeviceStoreNV &primaryVertexContext, const int layerIndex)
{

  const int currentClusterIndex { static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) };
  const int nextLayerClustersNum { static_cast<int>(primaryVertexContext.getClusters()[layerIndex + 1].size()) };

  if (currentClusterIndex < nextLayerClustersNum) {

    const int currentBinIndex {
        primaryVertexContext.getClusters()[layerIndex + 1][currentClusterIndex].indexTableBinIndex };
    int previousBinIndex;

    if (currentClusterIndex == 0) {

      primaryVertexContext.getIndexTables()[layerIndex][0] = 0;
      previousBinIndex = 0;

    } else {

      previousBinIndex = primaryVertexContext.getClusters()[layerIndex + 1][currentClusterIndex - 1].indexTableBinIndex;
    }

    if (currentBinIndex > previousBinIndex) {

      for (int iBin { previousBinIndex + 1 }; iBin <= currentBinIndex; ++iBin) {

        primaryVertexContext.getIndexTables()[layerIndex][iBin] = currentClusterIndex;
      }

      previousBinIndex = currentBinIndex;
    }

    if (currentClusterIndex == nextLayerClustersNum - 1) {

      for (int iBin{ currentBinIndex + 1 }; iBin <= o2::its::constants::index_table::ZBins * o2::its::constants::index_table::PhiBins;
           iBin++) {

        primaryVertexContext.getIndexTables()[layerIndex][iBin] = nextLayerClustersNum;
      }
    }
  }
}

__device__ void fillTrackletsPerClusterTables(GPU::DeviceStoreNV &primaryVertexContext, const int layerIndex)
{
  const int currentClusterIndex { static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) };
  const int clustersSize { static_cast<int>(primaryVertexContext.getClusters()[layerIndex + 1].size()) };

  if (currentClusterIndex < clustersSize) {

    primaryVertexContext.getTrackletsPerClusterTable()[layerIndex][currentClusterIndex] = 0;
  }
}

__device__ void fillCellsPerClusterTables(GPU::DeviceStoreNV &primaryVertexContext, const int layerIndex)
{
  const int totalThreadNum { static_cast<int>(primaryVertexContext.getClusters()[layerIndex + 1].size()) };
  const int trackletsSize { static_cast<int>(primaryVertexContext.getTracklets()[layerIndex + 1].capacity()) };
  const int trackletsPerThread { 1 + (trackletsSize - 1) / totalThreadNum };
  const int firstTrackletIndex { static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) * trackletsPerThread };

  if (firstTrackletIndex < trackletsSize) {

    const int trackletsToSet { min(trackletsSize, firstTrackletIndex + trackletsPerThread) - firstTrackletIndex };
    memset(&primaryVertexContext.getCellsPerTrackletTable()[layerIndex][firstTrackletIndex], 0,
        trackletsToSet * sizeof(int));
  }
}

__global__ void fillDeviceStructures(GPU::DeviceStoreNV &primaryVertexContext, const int layerIndex)
{
  fillIndexTables(primaryVertexContext, layerIndex);

  if (layerIndex < o2::its::constants::its::CellsPerRoad) {

    fillTrackletsPerClusterTables(primaryVertexContext, layerIndex);
  }

  if (layerIndex < o2::its::constants::its::CellsPerRoad - 1) {

    fillCellsPerClusterTables(primaryVertexContext, layerIndex);
  }
}
}

namespace o2
{
namespace its
{
namespace GPU
{

DeviceStoreNV::DeviceStoreNV()
{
  // Nothing to do
}

UniquePointer<DeviceStoreNV> DeviceStoreNV::initialise(const float3& primaryVertex,
                                                       const std::array<std::vector<Cluster>, constants::its::LayersNumber>& clusters,
                                                       const std::array<std::vector<Tracklet>, constants::its::TrackletsPerRoad>& tracklets,
                                                       const std::array<std::vector<Cell>, constants::its::CellsPerRoad>& cells,
                                                       const std::array<std::vector<int>, constants::its::CellsPerRoad - 1>& cellsLookupTable)
{
  mPrimaryVertex = UniquePointer<float3>{ primaryVertex };

  for (int iLayer{ 0 }; iLayer < constants::its::LayersNumber; ++iLayer) {

    this->mClusters[iLayer] =
        Vector<Cluster> { &clusters[iLayer][0], static_cast<int>(clusters[iLayer].size()) };

    if (iLayer < constants::its::TrackletsPerRoad) {
      this->mTracklets[iLayer].reset(tracklets[iLayer].capacity());
    }

    if (iLayer < constants::its::CellsPerRoad) {

      this->mTrackletsLookupTable[iLayer].reset(static_cast<int>(clusters[iLayer + 1].size()));
      this->mTrackletsPerClusterTable[iLayer].reset(static_cast<int>(clusters[iLayer + 1].size()));
      this->mCells[iLayer].reset(static_cast<int>(cells[iLayer].capacity()));
    }

    if (iLayer < constants::its::CellsPerRoad - 1) {

      this->mCellsLookupTable[iLayer].reset(static_cast<int>(cellsLookupTable[iLayer].size()));
      this->mCellsPerTrackletTable[iLayer].reset(static_cast<int>(cellsLookupTable[iLayer].size()));
    }
  }

  UniquePointer<DeviceStoreNV> gpuContextDevicePointer { *this };

  std::array<Stream, constants::its::LayersNumber> streamArray;

  for (int iLayer{ 0 }; iLayer < constants::its::TrackletsPerRoad; ++iLayer) {

    const int nextLayerClustersNum = static_cast<int>(clusters[iLayer + 1].size());

    dim3 threadsPerBlock { Utils::Host::getBlockSize(nextLayerClustersNum) };
    dim3 blocksGrid { Utils::Host::getBlocksGrid(threadsPerBlock, nextLayerClustersNum) };

    fillDeviceStructures<<< blocksGrid, threadsPerBlock, 0, streamArray[iLayer].get() >>>(*gpuContextDevicePointer, iLayer);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {

      std::ostringstream errorString { };
      errorString << __FILE__ << ":" << __LINE__ << " CUDA API returned error [" << cudaGetErrorString(error)
          << "] (code " << error << ")" << std::endl;

      throw std::runtime_error { errorString.str() };
    }
  }

  return gpuContextDevicePointer;
}

}
}
}
