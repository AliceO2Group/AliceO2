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
/// \file TrackerTraitsNV.cu
/// \brief
///

#include "ITStrackingCUDA/TrackerTraitsNV.h"

#include <array>
#include <sstream>
#include <iostream>

#include <cooperative_groups.h>

#include "cub/cub.cuh"

#include "ITStracking/Constants.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/MathUtils.h"
#include "ITStrackingCUDA/Context.h"
#include "ITStrackingCUDA/DeviceStoreNV.h"
#include "ITStrackingCUDA/PrimaryVertexContextNV.h"
#include "ITStrackingCUDA/Stream.h"
#include "ITStrackingCUDA/Vector.h"

namespace o2
{
namespace its
{
namespace GPU
{

__constant__ TrackingParameters kTrkPar;

__device__ void computeLayerTracklets(DeviceStoreNV& devStore, const int layerIndex,
    Vector<Tracklet>& trackletsVector)
{
  const int currentClusterIndex = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  int clusterTrackletsNum = 0;

  if (currentClusterIndex < devStore.getClusters()[layerIndex].size()) {

    Vector<Cluster> nextLayerClusters { devStore.getClusters()[layerIndex + 1].getWeakCopy() };
    const Cluster currentCluster { devStore.getClusters()[layerIndex][currentClusterIndex] };

    /*if (mUsedClustersTable[currentCluster.clusterId] != constants::its::UnusedIndex) {

     continue;
     }*/

    const float tanLambda { (currentCluster.zCoordinate - devStore.getPrimaryVertex().z)
        / currentCluster.rCoordinate };
    const float directionZIntersection{ tanLambda * ((constants::its::LayersRCoordinate())[layerIndex + 1] - currentCluster.rCoordinate) + currentCluster.zCoordinate };

    const int4 selectedBinsRect { TrackerTraits::getBinsRect(currentCluster, layerIndex, directionZIntersection,
                                                             kTrkPar.TrackletMaxDeltaZ[layerIndex], kTrkPar.TrackletMaxDeltaPhi) };

    if (selectedBinsRect.x != 0 || selectedBinsRect.y != 0 || selectedBinsRect.z != 0 || selectedBinsRect.w != 0) {

      const int nextLayerClustersNum { static_cast<int>(nextLayerClusters.size()) };
      int phiBinsNum { selectedBinsRect.w - selectedBinsRect.y + 1 };

      if (phiBinsNum < 0) {

        phiBinsNum += constants::IndexTable::PhiBins;
      }

      for (int iPhiBin{ selectedBinsRect.y }, iPhiCount{ 0 }; iPhiCount < phiBinsNum;
           iPhiBin = ++iPhiBin == constants::IndexTable::PhiBins ? 0 : iPhiBin, iPhiCount++) {

        const int firstBinIndex { IndexTableUtils::getBinIndex(selectedBinsRect.x, iPhiBin) };
        const int firstRowClusterIndex = devStore.getIndexTables()[layerIndex][firstBinIndex];
        const int maxRowClusterIndex = devStore.getIndexTables()[layerIndex][ { firstBinIndex
            + selectedBinsRect.z - selectedBinsRect.x + 1 }];

        for (int iNextLayerCluster { firstRowClusterIndex };
            iNextLayerCluster <= maxRowClusterIndex && iNextLayerCluster < nextLayerClustersNum; ++iNextLayerCluster) {

          const Cluster& nextCluster { nextLayerClusters[iNextLayerCluster] };

          const float deltaZ{ gpu::GPUCommonMath::Abs(
            tanLambda * (nextCluster.rCoordinate - currentCluster.rCoordinate) + currentCluster.zCoordinate - nextCluster.zCoordinate) };
          const float deltaPhi{ gpu::GPUCommonMath::Abs(currentCluster.phiCoordinate - nextCluster.phiCoordinate) };

          if (deltaZ < kTrkPar.TrackletMaxDeltaZ[layerIndex] && (deltaPhi < kTrkPar.TrackletMaxDeltaPhi || gpu::GPUCommonMath::Abs(deltaPhi - constants::Math::TwoPi) < kTrkPar.TrackletMaxDeltaPhi)) {

            cooperative_groups::coalesced_group threadGroup = cooperative_groups::coalesced_threads();
            int currentIndex{};

            if (threadGroup.thread_rank() == 0) {

              currentIndex = trackletsVector.extend(threadGroup.size());
            }

            currentIndex = threadGroup.shfl(currentIndex, 0) + threadGroup.thread_rank();

            trackletsVector.emplace(currentIndex, currentClusterIndex, iNextLayerCluster, currentCluster, nextCluster);
            ++clusterTrackletsNum;
          }
        }
      }

      if (layerIndex > 0) {

        devStore.getTrackletsPerClusterTable()[layerIndex - 1][currentClusterIndex] = clusterTrackletsNum;
      }
    }
  }
}

__device__ void computeLayerCells(DeviceStoreNV& devStore, const int layerIndex,
    Vector<Cell>& cellsVector)
{
  const int currentTrackletIndex = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const float3 &primaryVertex = devStore.getPrimaryVertex();
  int trackletCellsNum = 0;

  if (currentTrackletIndex < devStore.getTracklets()[layerIndex].size()) {

    const Tracklet& currentTracklet { devStore.getTracklets()[layerIndex][currentTrackletIndex] };
    const int nextLayerClusterIndex { currentTracklet.secondClusterIndex };
    const int nextLayerFirstTrackletIndex {
        devStore.getTrackletsLookupTable()[layerIndex][nextLayerClusterIndex] };
    const int nextLayerTrackletsNum { static_cast<int>(devStore.getTracklets()[layerIndex + 1].size()) };

    if (devStore.getTracklets()[layerIndex + 1][nextLayerFirstTrackletIndex].firstClusterIndex
        == nextLayerClusterIndex) {

      const Cluster& firstCellCluster {
          devStore.getClusters()[layerIndex][currentTracklet.firstClusterIndex] };
      const Cluster& secondCellCluster {
          devStore.getClusters()[layerIndex + 1][currentTracklet.secondClusterIndex] };
      const float firstCellClusterQuadraticRCoordinate { firstCellCluster.rCoordinate * firstCellCluster.rCoordinate };
      const float secondCellClusterQuadraticRCoordinate { secondCellCluster.rCoordinate * secondCellCluster.rCoordinate };
      const float3 firstDeltaVector { secondCellCluster.xCoordinate - firstCellCluster.xCoordinate,
          secondCellCluster.yCoordinate - firstCellCluster.yCoordinate, secondCellClusterQuadraticRCoordinate
              - firstCellClusterQuadraticRCoordinate };

      for (int iNextLayerTracklet { nextLayerFirstTrackletIndex };
          iNextLayerTracklet < nextLayerTrackletsNum
              && devStore.getTracklets()[layerIndex + 1][iNextLayerTracklet].firstClusterIndex
                  == nextLayerClusterIndex; ++iNextLayerTracklet) {

        const Tracklet& nextTracklet { devStore.getTracklets()[layerIndex + 1][iNextLayerTracklet] };
        const float deltaTanLambda{ gpu::GPUCommonMath::Abs(currentTracklet.tanLambda - nextTracklet.tanLambda) };
        const float deltaPhi{ gpu::GPUCommonMath::Abs(currentTracklet.phiCoordinate - nextTracklet.phiCoordinate) };

        if (deltaTanLambda < kTrkPar.CellMaxDeltaTanLambda && (deltaPhi < kTrkPar.CellMaxDeltaPhi || gpu::GPUCommonMath::Abs(deltaPhi - constants::Math::TwoPi) < kTrkPar.CellMaxDeltaPhi)) {

          const float averageTanLambda { 0.5f * (currentTracklet.tanLambda + nextTracklet.tanLambda) };
          const float directionZIntersection { -averageTanLambda * firstCellCluster.rCoordinate
              + firstCellCluster.zCoordinate };
          const float deltaZ{ gpu::GPUCommonMath::Abs(directionZIntersection - primaryVertex.z) };

          if (deltaZ < kTrkPar.CellMaxDeltaZ[layerIndex]) {

            const Cluster& thirdCellCluster {
                devStore.getClusters()[layerIndex + 2][nextTracklet.secondClusterIndex] };

            const float thirdCellClusterQuadraticRCoordinate { thirdCellCluster.rCoordinate
                * thirdCellCluster.rCoordinate };

            const float3 secondDeltaVector { thirdCellCluster.xCoordinate - firstCellCluster.xCoordinate,
                thirdCellCluster.yCoordinate - firstCellCluster.yCoordinate, thirdCellClusterQuadraticRCoordinate
                    - firstCellClusterQuadraticRCoordinate };

            float3 cellPlaneNormalVector { MathUtils::crossProduct(firstDeltaVector, secondDeltaVector) };

            const float vectorNorm{ gpu::GPUCommonMath::Sqrt(
              cellPlaneNormalVector.x * cellPlaneNormalVector.x + cellPlaneNormalVector.y * cellPlaneNormalVector.y + cellPlaneNormalVector.z * cellPlaneNormalVector.z) };

            if (!(vectorNorm < constants::Math::FloatMinThreshold || gpu::GPUCommonMath::Abs(cellPlaneNormalVector.z) < constants::Math::FloatMinThreshold)) {

              const float inverseVectorNorm { 1.0f / vectorNorm };
              const float3 normalizedPlaneVector { cellPlaneNormalVector.x * inverseVectorNorm, cellPlaneNormalVector.y
                  * inverseVectorNorm, cellPlaneNormalVector.z * inverseVectorNorm };
              const float planeDistance { -normalizedPlaneVector.x * (secondCellCluster.xCoordinate - primaryVertex.x)
                  - (normalizedPlaneVector.y * secondCellCluster.yCoordinate - primaryVertex.y)
                  - normalizedPlaneVector.z * secondCellClusterQuadraticRCoordinate };
              const float normalizedPlaneVectorQuadraticZCoordinate { normalizedPlaneVector.z * normalizedPlaneVector.z };
              const float cellTrajectoryRadius{ gpu::GPUCommonMath::Sqrt(
                (1.0f - normalizedPlaneVectorQuadraticZCoordinate - 4.0f * planeDistance * normalizedPlaneVector.z) / (4.0f * normalizedPlaneVectorQuadraticZCoordinate)) };
              const float2 circleCenter { -0.5f * normalizedPlaneVector.x / normalizedPlaneVector.z, -0.5f
                  * normalizedPlaneVector.y / normalizedPlaneVector.z };
              const float distanceOfClosestApproach{ gpu::GPUCommonMath::Abs(
                cellTrajectoryRadius - gpu::GPUCommonMath::Sqrt(circleCenter.x * circleCenter.x + circleCenter.y * circleCenter.y)) };

              if (distanceOfClosestApproach
                  <= kTrkPar.CellMaxDCA[layerIndex]) {

            	cooperative_groups::coalesced_group threadGroup = cooperative_groups::coalesced_threads();
                int currentIndex { };

                if (threadGroup.thread_rank() == 0) {

                  currentIndex = cellsVector.extend(threadGroup.size());
                }

                currentIndex = threadGroup.shfl(currentIndex, 0) + threadGroup.thread_rank();

                cellsVector.emplace(currentIndex, currentTracklet.firstClusterIndex,
                    nextTracklet.firstClusterIndex, nextTracklet.secondClusterIndex, currentTrackletIndex,
                    iNextLayerTracklet, normalizedPlaneVector, 1.0f / cellTrajectoryRadius);
                ++trackletCellsNum;
              }
            }
          }
        }
      }

      if (layerIndex > 0) {

        devStore.getCellsPerTrackletTable()[layerIndex - 1][currentTrackletIndex] = trackletCellsNum;
      }
    }
  }
}

__global__ void layerTrackletsKernel(DeviceStoreNV& devStore, const int layerIndex,
    Vector<Tracklet> trackletsVector)
{
  computeLayerTracklets(devStore, layerIndex, trackletsVector);
}

__global__ void sortTrackletsKernel(DeviceStoreNV& devStore, const int layerIndex,
    Vector<Tracklet> tempTrackletArray)
{
  const int currentTrackletIndex { static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) };

  if (currentTrackletIndex < tempTrackletArray.size()) {

    const int firstClusterIndex = tempTrackletArray[currentTrackletIndex].firstClusterIndex;
    const int offset = atomicAdd(&devStore.getTrackletsPerClusterTable()[layerIndex - 1][firstClusterIndex],
        -1) - 1;
    const int startIndex = devStore.getTrackletsLookupTable()[layerIndex - 1][firstClusterIndex];

    memcpy(&devStore.getTracklets()[layerIndex][startIndex + offset],
        &tempTrackletArray[currentTrackletIndex], sizeof(Tracklet));
  }
}

__global__ void layerCellsKernel(DeviceStoreNV& devStore, const int layerIndex,
    Vector<Cell> cellsVector)
{
  computeLayerCells(devStore, layerIndex, cellsVector);
}

__global__ void sortCellsKernel(DeviceStoreNV& devStore, const int layerIndex,
    Vector<Cell> tempCellsArray)
{
  const int currentCellIndex = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);

  if (currentCellIndex < tempCellsArray.size()) {

    const int firstTrackletIndex = tempCellsArray[currentCellIndex].getFirstTrackletIndex();
    const int offset = atomicAdd(&devStore.getCellsPerTrackletTable()[layerIndex - 1][firstTrackletIndex],
        -1) - 1;
    const int startIndex = devStore.getCellsLookupTable()[layerIndex - 1][firstTrackletIndex];

    memcpy(&devStore.getCells()[layerIndex][startIndex + offset], &tempCellsArray[currentCellIndex],
        sizeof(Cell));
  }
}

} /// End of GPU namespace

TrackerTraits* createTrackerTraitsNV() {
  return new TrackerTraitsNV;
}

TrackerTraitsNV::TrackerTraitsNV() {
  mPrimaryVertexContext = new PrimaryVertexContextNV;
}

TrackerTraitsNV::~TrackerTraitsNV() {
  delete mPrimaryVertexContext;
}

void TrackerTraitsNV::computeLayerTracklets()
{
  PrimaryVertexContextNV* primaryVertexContext = static_cast<PrimaryVertexContextNV*>(mPrimaryVertexContext);

  cudaMemcpyToSymbol(GPU::kTrkPar, &mTrkParams, sizeof(TrackingParameters));
  std::array<size_t, constants::its::CellsPerRoad> tempSize;
  std::array<int, constants::its::CellsPerRoad> trackletsNum;
  std::array<GPU::Stream, constants::its::TrackletsPerRoad> streamArray;

  for (int iLayer{ 0 }; iLayer < constants::its::CellsPerRoad; ++iLayer) {

    tempSize[iLayer] = 0;
    primaryVertexContext->getTempTrackletArray()[iLayer].reset(
		static_cast<int>(primaryVertexContext->getDeviceTracklets()[iLayer + 1].capacity()));

    cub::DeviceScan::ExclusiveSum(static_cast<void *>(NULL), tempSize[iLayer],
        primaryVertexContext->getDeviceTrackletsPerClustersTable()[iLayer].get(),
        primaryVertexContext->getDeviceTrackletsLookupTable()[iLayer].get(),
        primaryVertexContext->getClusters()[iLayer + 1].size());

    primaryVertexContext->getTempTableArray()[iLayer].reset(static_cast<int>(tempSize[iLayer]));
  }

  cudaDeviceSynchronize();

  for (int iLayer{ 0 }; iLayer < constants::its::TrackletsPerRoad; ++iLayer) {

    const GPU::DeviceProperties& deviceProperties = GPU::Context::getInstance().getDeviceProperties();
    const int clustersNum { static_cast<int>(primaryVertexContext->getClusters()[iLayer].size()) };
    dim3 threadsPerBlock { GPU::Utils::Host::getBlockSize(clustersNum, 1, 192) };
    dim3 blocksGrid { GPU::Utils::Host::getBlocksGrid(threadsPerBlock, clustersNum) };

    if (iLayer == 0) {

      GPU::layerTrackletsKernel<<< blocksGrid, threadsPerBlock, 0, streamArray[iLayer].get() >>>(primaryVertexContext->getDeviceContext(),
          iLayer, primaryVertexContext->getDeviceTracklets()[iLayer].getWeakCopy());

    } else {

      GPU::layerTrackletsKernel<<< blocksGrid, threadsPerBlock, 0, streamArray[iLayer].get() >>>(primaryVertexContext->getDeviceContext(),
          iLayer, primaryVertexContext->getTempTrackletArray()[iLayer - 1].getWeakCopy());
    }

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {

      std::ostringstream errorString { };
      errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << ")"
          << std::endl;

      throw std::runtime_error { errorString.str() };
    }
  }

  cudaDeviceSynchronize();

  for (int iLayer{ 0 }; iLayer < constants::its::CellsPerRoad; ++iLayer) {

    trackletsNum[iLayer] = primaryVertexContext->getTempTrackletArray()[iLayer].getSizeFromDevice();
    if (trackletsNum[iLayer] == 0) {
      continue;
    }
    primaryVertexContext->getDeviceTracklets()[iLayer + 1].resize(trackletsNum[iLayer]);

    cub::DeviceScan::ExclusiveSum(static_cast<void *>(primaryVertexContext->getTempTableArray()[iLayer].get()), tempSize[iLayer],
        primaryVertexContext->getDeviceTrackletsPerClustersTable()[iLayer].get(),
        primaryVertexContext->getDeviceTrackletsLookupTable()[iLayer].get(),
        primaryVertexContext->getClusters()[iLayer + 1].size(), streamArray[iLayer + 1].get());

    dim3 threadsPerBlock { GPU::Utils::Host::getBlockSize(trackletsNum[iLayer]) };
    dim3 blocksGrid { GPU::Utils::Host::getBlocksGrid(threadsPerBlock, trackletsNum[iLayer]) };

    GPU::sortTrackletsKernel<<< blocksGrid, threadsPerBlock, 0, streamArray[iLayer + 1].get() >>>(primaryVertexContext->getDeviceContext(),
        iLayer + 1, primaryVertexContext->getTempTrackletArray()[iLayer].getWeakCopy());

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {

      std::ostringstream errorString { };
      errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << ")"
          << std::endl;

      throw std::runtime_error { errorString.str() };
    }
  }
}

void TrackerTraitsNV::computeLayerCells()
{

  PrimaryVertexContextNV* primaryVertexContext = static_cast<PrimaryVertexContextNV*>(mPrimaryVertexContext);
  std::array<size_t, constants::its::CellsPerRoad - 1> tempSize;
  std::array<int, constants::its::CellsPerRoad - 1> trackletsNum;
  std::array<int, constants::its::CellsPerRoad - 1> cellsNum;
  std::array<GPU::Stream, constants::its::CellsPerRoad> streamArray;

  for (int iLayer{ 0 }; iLayer < constants::its::CellsPerRoad - 1; ++iLayer) {

    tempSize[iLayer] = 0;
    trackletsNum[iLayer] = primaryVertexContext->getDeviceTracklets()[iLayer + 1].getSizeFromDevice();
    primaryVertexContext->getTempCellArray()[iLayer].reset(
		static_cast<int>(primaryVertexContext->getDeviceCells()[iLayer + 1].capacity()));
    if (trackletsNum[iLayer] == 0) {
      continue;
    }
    cub::DeviceScan::ExclusiveSum(static_cast<void *>(NULL), tempSize[iLayer],
        primaryVertexContext->getDeviceCellsPerTrackletTable()[iLayer].get(),
        primaryVertexContext->getDeviceCellsLookupTable()[iLayer].get(), trackletsNum[iLayer]);

    primaryVertexContext->getTempTableArray()[iLayer].reset(static_cast<int>(tempSize[iLayer]));
  }

  cudaDeviceSynchronize();

  for (int iLayer{ 0 }; iLayer < constants::its::CellsPerRoad; ++iLayer) {
    const GPU::DeviceProperties& deviceProperties = GPU::Context::getInstance().getDeviceProperties();
    const int trackletsSize = primaryVertexContext->getDeviceTracklets()[iLayer].getSizeFromDevice();
    if (trackletsSize == 0) {
      continue;
    }
    dim3 threadsPerBlock { GPU::Utils::Host::getBlockSize(trackletsSize) };
    dim3 blocksGrid { GPU::Utils::Host::getBlocksGrid(threadsPerBlock, trackletsSize) };

    if(iLayer == 0) {

      GPU::layerCellsKernel<<< blocksGrid, threadsPerBlock, 0, streamArray[iLayer].get() >>>(primaryVertexContext->getDeviceContext(),
          iLayer, primaryVertexContext->getDeviceCells()[iLayer].getWeakCopy());

    } else {

      GPU::layerCellsKernel<<< blocksGrid, threadsPerBlock, 0, streamArray[iLayer].get() >>>(primaryVertexContext->getDeviceContext(),
          iLayer, primaryVertexContext->getTempCellArray()[iLayer - 1].getWeakCopy());
    }

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {

      std::ostringstream errorString { };
      errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << ")"
          << std::endl;

      throw std::runtime_error { errorString.str() };
    }
  }

  cudaDeviceSynchronize();

  for (int iLayer{ 0 }; iLayer < constants::its::CellsPerRoad - 1; ++iLayer) {
    cellsNum[iLayer] = primaryVertexContext->getTempCellArray()[iLayer].getSizeFromDevice();
    if (cellsNum[iLayer] == 0) {
      continue;
    }
    primaryVertexContext->getDeviceCells()[iLayer + 1].resize(cellsNum[iLayer]);

    cub::DeviceScan::ExclusiveSum(static_cast<void *>(primaryVertexContext->getTempTableArray()[iLayer].get()), tempSize[iLayer],
        primaryVertexContext->getDeviceCellsPerTrackletTable()[iLayer].get(),
        primaryVertexContext->getDeviceCellsLookupTable()[iLayer].get(), trackletsNum[iLayer],
        streamArray[iLayer + 1].get());

    dim3 threadsPerBlock { GPU::Utils::Host::getBlockSize(trackletsNum[iLayer]) };
    dim3 blocksGrid { GPU::Utils::Host::getBlocksGrid(threadsPerBlock, trackletsNum[iLayer]) };

    GPU::sortCellsKernel<<< blocksGrid, threadsPerBlock, 0, streamArray[iLayer + 1].get() >>>(primaryVertexContext->getDeviceContext(),
        iLayer + 1, primaryVertexContext->getTempCellArray()[iLayer].getWeakCopy());

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {

      std::ostringstream errorString { };
      errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << ")"
          << std::endl;

      throw std::runtime_error { errorString.str() };
    }
  }

  cudaDeviceSynchronize();

  for (int iLayer{ 0 }; iLayer < constants::its::CellsPerRoad; ++iLayer) {

    int cellsSize = 0;
    if (iLayer == 0) {

      cellsSize = primaryVertexContext->getDeviceCells()[iLayer].getSizeFromDevice();
      if (cellsSize == 0)
        continue;
    } else {

      cellsSize = cellsNum[iLayer - 1];
      if (cellsSize == 0)
        continue;
      primaryVertexContext->getDeviceCellsLookupTable()[iLayer - 1].copyIntoVector(
          primaryVertexContext->getCellsLookupTable()[iLayer - 1], trackletsNum[iLayer - 1]);
    }

    primaryVertexContext->getDeviceCells()[iLayer].copyIntoVector(primaryVertexContext->getCells()[iLayer], cellsSize);
  }
}

void TrackerTraitsNV::refitTracks(const std::array<std::vector<TrackingFrameInfo>, 7>& tf, std::vector<TrackITSExt>& tracks)
{
  PrimaryVertexContextNV* pvctx = static_cast<PrimaryVertexContextNV*>(mPrimaryVertexContext);

  std::array<const Cell*, 5> cells;
  for (int iLayer = 0; iLayer < 5; iLayer++) {
    cells[iLayer] = pvctx->getDeviceCells()[iLayer].get();
  }
  std::array<const Cluster*, 7> clusters;
  for (int iLayer = 0; iLayer < 7; iLayer++) {
    clusters[iLayer] = pvctx->getDeviceClusters()[iLayer].get();
  }
  mChainRunITSTrackFit(*mChain, mPrimaryVertexContext->getRoads(), clusters, cells, tf, tracks);
}
}
}
