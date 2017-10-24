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
/// \file PrimaryVertexContext.cxx
/// \brief
///

#include "ITSReconstruction/CA/gpu/PrimaryVertexContext.h"

#include <sstream>

#include "ITSReconstruction/CA/gpu/Stream.h"

namespace {

using namespace o2::ITS::CA;

__device__ void fillIndexTables(GPU::PrimaryVertexContext &primaryVertexContext, const int layerIndex)
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

      for (int iBin { currentBinIndex + 1 }; iBin <= Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins;
          iBin++) {

        primaryVertexContext.getIndexTables()[layerIndex][iBin] = nextLayerClustersNum;
      }
    }
  }
}

__device__ void fillTrackletsPerClusterTables(GPU::PrimaryVertexContext &primaryVertexContext, const int layerIndex)
{
  const int currentClusterIndex { static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) };
  const int clustersSize { static_cast<int>(primaryVertexContext.getClusters()[layerIndex + 1].size()) };

  if (currentClusterIndex < clustersSize) {

    primaryVertexContext.getTrackletsPerClusterTable()[layerIndex][currentClusterIndex] = 0;
  }
}

__device__ void fillCellsPerClusterTables(GPU::PrimaryVertexContext &primaryVertexContext, const int layerIndex)
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

__global__ void fillDeviceStructures(GPU::PrimaryVertexContext &primaryVertexContext, const int layerIndex)
{
  fillIndexTables(primaryVertexContext, layerIndex);

  if (layerIndex < Constants::ITS::CellsPerRoad) {

    fillTrackletsPerClusterTables(primaryVertexContext, layerIndex);
  }

  if (layerIndex < Constants::ITS::CellsPerRoad - 1) {

    fillCellsPerClusterTables(primaryVertexContext, layerIndex);
  }
}
}

namespace o2
{
namespace ITS
{
namespace CA
{
namespace GPU
{

PrimaryVertexContext::PrimaryVertexContext()
{
  // Nothing to do
}

UniquePointer<PrimaryVertexContext> PrimaryVertexContext::initialize(const float3 &primaryVertex,
    const std::array<std::vector<Cluster>, Constants::ITS::LayersNumber> &clusters,
    const std::array<std::vector<Cell>, Constants::ITS::CellsPerRoad> &cells,
    const std::array<std::vector<int>, Constants::ITS::CellsPerRoad - 1> &cellsLookupTable)
{
  mPrimaryVertex = UniquePointer<float3>{ primaryVertex };

  for (int iLayer { 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer) {

    this->mClusters[iLayer] =
        Vector<Cluster> { &clusters[iLayer][0], static_cast<int>(clusters[iLayer].size()) };

    if (iLayer < Constants::ITS::TrackletsPerRoad) {

      this->mTracklets[iLayer].reset(static_cast<int>(std::ceil(
          (Constants::Memory::TrackletsMemoryCoefficients[iLayer] * clusters[iLayer].size())
              * clusters[iLayer + 1].size())));
    }

    if (iLayer < Constants::ITS::CellsPerRoad) {

      this->mTrackletsLookupTable[iLayer].reset(static_cast<int>(clusters[iLayer + 1].size()));
      this->mTrackletsPerClusterTable[iLayer].reset(static_cast<int>(clusters[iLayer + 1].size()));
      this->mCells[iLayer].reset(static_cast<int>(cells[iLayer].capacity()));
    }

    if (iLayer < Constants::ITS::CellsPerRoad - 1) {

      this->mCellsLookupTable[iLayer].reset(static_cast<int>(cellsLookupTable[iLayer].size()));
      this->mCellsPerTrackletTable[iLayer].reset(static_cast<int>(cellsLookupTable[iLayer].size()));
    }
  }

  UniquePointer<PrimaryVertexContext> gpuContextDevicePointer { *this };

  std::array<Stream, Constants::ITS::LayersNumber> streamArray;

  for (int iLayer { 0 }; iLayer < Constants::ITS::TrackletsPerRoad; ++iLayer) {

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
}
