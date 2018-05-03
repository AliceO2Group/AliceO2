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
/// \file Tracker.cu
/// \brief
///

#include "ITSReconstruction/CA/Tracker.h"

#include <array>
#include <sstream>
#include <iostream>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "cub/cub.cuh"

#include "ITSReconstruction/CA/Constants.h"
#include "ITSReconstruction/CA/IndexTableUtils.h"
#include "ITSReconstruction/CA/MathUtils.h"
#include "ITSReconstruction/CA/PrimaryVertexContext.h"
#include "ITSReconstruction/CA/TrackingUtils.h"
#include "ITSReconstruction/CA/gpu/Context.h"
#include "ITSReconstruction/CA/gpu/Stream.h"
#include "ITSReconstruction/CA/gpu/Vector.h"

namespace o2
{
namespace ITS
{
namespace CA
{
namespace GPU
{

__device__ void computeLayerTracklets(PrimaryVertexContext &primaryVertexContext, const int layerIndex,
    Vector<Tracklet>& trackletsVector)
{
  const int currentClusterIndex = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  int clusterTrackletsNum = 0;

  if (currentClusterIndex < primaryVertexContext.getClusters()[layerIndex].size()) {

    Vector<Cluster> nextLayerClusters { primaryVertexContext.getClusters()[layerIndex + 1].getWeakCopy() };
    const Cluster currentCluster { primaryVertexContext.getClusters()[layerIndex][currentClusterIndex] };

    /*if (mUsedClustersTable[currentCluster.clusterId] != Constants::ITS::UnusedIndex) {

     continue;
     }*/

    const float tanLambda { (currentCluster.zCoordinate - primaryVertexContext.getPrimaryVertex().z)
        / currentCluster.rCoordinate };
    const float directionZIntersection { tanLambda
        * ((Constants::ITS::LayersRCoordinate())[layerIndex + 1] - currentCluster.rCoordinate)
        + currentCluster.zCoordinate };

    const int4 selectedBinsRect { TrackingUtils::getBinsRect(currentCluster, layerIndex, directionZIntersection) };

    if (selectedBinsRect.x != 0 || selectedBinsRect.y != 0 || selectedBinsRect.z != 0 || selectedBinsRect.w != 0) {

      const int nextLayerClustersNum { static_cast<int>(nextLayerClusters.size()) };
      int phiBinsNum { selectedBinsRect.w - selectedBinsRect.y + 1 };

      if (phiBinsNum < 0) {

        phiBinsNum += Constants::IndexTable::PhiBins;
      }

      for (int iPhiBin { selectedBinsRect.y }, iPhiCount { 0 }; iPhiCount < phiBinsNum;
          iPhiBin = ++iPhiBin == Constants::IndexTable::PhiBins ? 0 : iPhiBin, iPhiCount++) {

        const int firstBinIndex { IndexTableUtils::getBinIndex(selectedBinsRect.x, iPhiBin) };
        const int firstRowClusterIndex = primaryVertexContext.getIndexTables()[layerIndex][firstBinIndex];
        const int maxRowClusterIndex = primaryVertexContext.getIndexTables()[layerIndex][ { firstBinIndex
            + selectedBinsRect.z - selectedBinsRect.x + 1 }];

        for (int iNextLayerCluster { firstRowClusterIndex };
            iNextLayerCluster <= maxRowClusterIndex && iNextLayerCluster < nextLayerClustersNum; ++iNextLayerCluster) {

          const Cluster& nextCluster { nextLayerClusters[iNextLayerCluster] };

          const float deltaZ { MATH_ABS(
              tanLambda * (nextCluster.rCoordinate - currentCluster.rCoordinate) + currentCluster.zCoordinate
                  - nextCluster.zCoordinate) };
          const float deltaPhi { MATH_ABS(currentCluster.phiCoordinate - nextCluster.phiCoordinate) };

          if (deltaZ < Constants::Thresholds::TrackletMaxDeltaZThreshold()[layerIndex]
              && (deltaPhi < Constants::Thresholds::PhiCoordinateCut
                  || MATH_ABS(deltaPhi - Constants::Math::TwoPi) < Constants::Thresholds::PhiCoordinateCut)) {

        	  cooperative_groups::coalesced_group threadGroup = cooperative_groups::coalesced_threads();
			  int currentIndex { };

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

        primaryVertexContext.getTrackletsPerClusterTable()[layerIndex - 1][currentClusterIndex] = clusterTrackletsNum;
      }
    }
  }
}

__device__ void computeLayerCells(PrimaryVertexContext& primaryVertexContext, const int layerIndex,
    Vector<Cell>& cellsVector)
{
  const int currentTrackletIndex = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const float3 &primaryVertex = primaryVertexContext.getPrimaryVertex();
  int trackletCellsNum = 0;

  if (currentTrackletIndex < primaryVertexContext.getTracklets()[layerIndex].size()) {

    const Tracklet& currentTracklet { primaryVertexContext.getTracklets()[layerIndex][currentTrackletIndex] };
    const int nextLayerClusterIndex { currentTracklet.secondClusterIndex };
    const int nextLayerFirstTrackletIndex {
        primaryVertexContext.getTrackletsLookupTable()[layerIndex][nextLayerClusterIndex] };
    const int nextLayerTrackletsNum { static_cast<int>(primaryVertexContext.getTracklets()[layerIndex + 1].size()) };

    if (primaryVertexContext.getTracklets()[layerIndex + 1][nextLayerFirstTrackletIndex].firstClusterIndex
        == nextLayerClusterIndex) {

      const Cluster& firstCellCluster {
          primaryVertexContext.getClusters()[layerIndex][currentTracklet.firstClusterIndex] };
      const Cluster& secondCellCluster {
          primaryVertexContext.getClusters()[layerIndex + 1][currentTracklet.secondClusterIndex] };
      const float firstCellClusterQuadraticRCoordinate { firstCellCluster.rCoordinate * firstCellCluster.rCoordinate };
      const float secondCellClusterQuadraticRCoordinate { secondCellCluster.rCoordinate * secondCellCluster.rCoordinate };
      const float3 firstDeltaVector { secondCellCluster.xCoordinate - firstCellCluster.xCoordinate,
          secondCellCluster.yCoordinate - firstCellCluster.yCoordinate, secondCellClusterQuadraticRCoordinate
              - firstCellClusterQuadraticRCoordinate };

      for (int iNextLayerTracklet { nextLayerFirstTrackletIndex };
          iNextLayerTracklet < nextLayerTrackletsNum
              && primaryVertexContext.getTracklets()[layerIndex + 1][iNextLayerTracklet].firstClusterIndex
                  == nextLayerClusterIndex; ++iNextLayerTracklet) {

        const Tracklet& nextTracklet { primaryVertexContext.getTracklets()[layerIndex + 1][iNextLayerTracklet] };
        const float deltaTanLambda { MATH_ABS(currentTracklet.tanLambda - nextTracklet.tanLambda) };
        const float deltaPhi { MATH_ABS(currentTracklet.phiCoordinate - nextTracklet.phiCoordinate) };

        if (deltaTanLambda < Constants::Thresholds::CellMaxDeltaTanLambdaThreshold
            && (deltaPhi < Constants::Thresholds::CellMaxDeltaPhiThreshold
                || MATH_ABS(deltaPhi - Constants::Math::TwoPi) < Constants::Thresholds::CellMaxDeltaPhiThreshold)) {

          const float averageTanLambda { 0.5f * (currentTracklet.tanLambda + nextTracklet.tanLambda) };
          const float directionZIntersection { -averageTanLambda * firstCellCluster.rCoordinate
              + firstCellCluster.zCoordinate };
          const float deltaZ { MATH_ABS(directionZIntersection - primaryVertex.z) };

          if (deltaZ < Constants::Thresholds::CellMaxDeltaZThreshold()[layerIndex]) {

            const Cluster& thirdCellCluster {
                primaryVertexContext.getClusters()[layerIndex + 2][nextTracklet.secondClusterIndex] };

            const float thirdCellClusterQuadraticRCoordinate { thirdCellCluster.rCoordinate
                * thirdCellCluster.rCoordinate };

            const float3 secondDeltaVector { thirdCellCluster.xCoordinate - firstCellCluster.xCoordinate,
                thirdCellCluster.yCoordinate - firstCellCluster.yCoordinate, thirdCellClusterQuadraticRCoordinate
                    - firstCellClusterQuadraticRCoordinate };

            float3 cellPlaneNormalVector { MathUtils::crossProduct(firstDeltaVector, secondDeltaVector) };

            const float vectorNorm { std::sqrt(
                cellPlaneNormalVector.x * cellPlaneNormalVector.x + cellPlaneNormalVector.y * cellPlaneNormalVector.y
                    + cellPlaneNormalVector.z * cellPlaneNormalVector.z) };

            if (!(vectorNorm < Constants::Math::FloatMinThreshold
                || MATH_ABS(cellPlaneNormalVector.z) < Constants::Math::FloatMinThreshold)) {

              const float inverseVectorNorm { 1.0f / vectorNorm };
              const float3 normalizedPlaneVector { cellPlaneNormalVector.x * inverseVectorNorm, cellPlaneNormalVector.y
                  * inverseVectorNorm, cellPlaneNormalVector.z * inverseVectorNorm };
              const float planeDistance { -normalizedPlaneVector.x * (secondCellCluster.xCoordinate - primaryVertex.x)
                  - (normalizedPlaneVector.y * secondCellCluster.yCoordinate - primaryVertex.y)
                  - normalizedPlaneVector.z * secondCellClusterQuadraticRCoordinate };
              const float normalizedPlaneVectorQuadraticZCoordinate { normalizedPlaneVector.z * normalizedPlaneVector.z };
              const float cellTrajectoryRadius { MATH_SQRT(
                  (1.0f - normalizedPlaneVectorQuadraticZCoordinate - 4.0f * planeDistance * normalizedPlaneVector.z)
                      / (4.0f * normalizedPlaneVectorQuadraticZCoordinate)) };
              const float2 circleCenter { -0.5f * normalizedPlaneVector.x / normalizedPlaneVector.z, -0.5f
                  * normalizedPlaneVector.y / normalizedPlaneVector.z };
              const float distanceOfClosestApproach { MATH_ABS(
                  cellTrajectoryRadius - MATH_SQRT(circleCenter.x * circleCenter.x + circleCenter.y * circleCenter.y)) };

              if (distanceOfClosestApproach
                  <= Constants::Thresholds::CellMaxDistanceOfClosestApproachThreshold()[layerIndex]) {

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

        primaryVertexContext.getCellsPerTrackletTable()[layerIndex - 1][currentTrackletIndex] = trackletCellsNum;
      }
    }
  }
}

__global__ void layerTrackletsKernel(PrimaryVertexContext& primaryVertexContext, const int layerIndex,
    Vector<Tracklet> trackletsVector)
{
  computeLayerTracklets(primaryVertexContext, layerIndex, trackletsVector);
}

__global__ void sortTrackletsKernel(PrimaryVertexContext& primaryVertexContext, const int layerIndex,
    Vector<Tracklet> tempTrackletArray)
{
  const int currentTrackletIndex { static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) };

  if (currentTrackletIndex < tempTrackletArray.size()) {

    const int firstClusterIndex = tempTrackletArray[currentTrackletIndex].firstClusterIndex;
    const int offset = atomicAdd(&primaryVertexContext.getTrackletsPerClusterTable()[layerIndex - 1][firstClusterIndex],
        -1) - 1;
    const int startIndex = primaryVertexContext.getTrackletsLookupTable()[layerIndex - 1][firstClusterIndex];

    memcpy(&primaryVertexContext.getTracklets()[layerIndex][startIndex + offset],
        &tempTrackletArray[currentTrackletIndex], sizeof(Tracklet));
  }
}

__global__ void layerCellsKernel(PrimaryVertexContext& primaryVertexContext, const int layerIndex,
    Vector<Cell> cellsVector)
{
  computeLayerCells(primaryVertexContext, layerIndex, cellsVector);
}

__global__ void sortCellsKernel(PrimaryVertexContext& primaryVertexContext, const int layerIndex,
    Vector<Cell> tempCellsArray)
{
  const int currentCellIndex = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);

  if (currentCellIndex < tempCellsArray.size()) {

    const int firstTrackletIndex = tempCellsArray[currentCellIndex].getFirstTrackletIndex();
    const int offset = atomicAdd(&primaryVertexContext.getCellsPerTrackletTable()[layerIndex - 1][firstTrackletIndex],
        -1) - 1;
    const int startIndex = primaryVertexContext.getCellsLookupTable()[layerIndex - 1][firstTrackletIndex];

    memcpy(&primaryVertexContext.getCells()[layerIndex][startIndex + offset], &tempCellsArray[currentCellIndex],
        sizeof(Cell));
  }
}

} /// End of GPU namespace

template<>
void TrackerTraits<true>::computeLayerTracklets(CA::PrimaryVertexContext& primaryVertexContext)
{
  std::array<size_t, Constants::ITS::CellsPerRoad> tempSize;
  std::array<int, Constants::ITS::CellsPerRoad> trackletsNum;
  std::array<GPU::Stream, Constants::ITS::TrackletsPerRoad> streamArray;

  for (int iLayer { 0 }; iLayer < Constants::ITS::CellsPerRoad; ++iLayer) {

    tempSize[iLayer] = 0;
    primaryVertexContext.getTempTrackletArray()[iLayer].reset(
		static_cast<int>(primaryVertexContext.getDeviceTracklets()[iLayer + 1].capacity()));

    cub::DeviceScan::ExclusiveSum(static_cast<void *>(NULL), tempSize[iLayer],
        primaryVertexContext.getDeviceTrackletsPerClustersTable()[iLayer].get(),
        primaryVertexContext.getDeviceTrackletsLookupTable()[iLayer].get(),
        primaryVertexContext.getClusters()[iLayer + 1].size());

    primaryVertexContext.getTempTableArray()[iLayer].reset(static_cast<int>(tempSize[iLayer]));
  }

  cudaDeviceSynchronize();

  for (int iLayer { 0 }; iLayer < Constants::ITS::TrackletsPerRoad; ++iLayer) {

    const GPU::DeviceProperties& deviceProperties = GPU::Context::getInstance().getDeviceProperties();
    const int clustersNum { static_cast<int>(primaryVertexContext.getClusters()[iLayer].size()) };
    dim3 threadsPerBlock { GPU::Utils::Host::getBlockSize(clustersNum, 1, 192) };
    dim3 blocksGrid { GPU::Utils::Host::getBlocksGrid(threadsPerBlock, clustersNum) };

    if (iLayer == 0) {

      GPU::layerTrackletsKernel<<< blocksGrid, threadsPerBlock, 0, streamArray[iLayer].get() >>>(primaryVertexContext.getDeviceContext(),
          iLayer, primaryVertexContext.getDeviceTracklets()[iLayer].getWeakCopy());

    } else {

      GPU::layerTrackletsKernel<<< blocksGrid, threadsPerBlock, 0, streamArray[iLayer].get() >>>(primaryVertexContext.getDeviceContext(),
          iLayer, primaryVertexContext.getTempTrackletArray()[iLayer - 1].getWeakCopy());
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

  for (int iLayer { 0 }; iLayer < Constants::ITS::CellsPerRoad; ++iLayer) {

    trackletsNum[iLayer] = primaryVertexContext.getTempTrackletArray()[iLayer].getSizeFromDevice();
    primaryVertexContext.getDeviceTracklets()[iLayer + 1].resize(trackletsNum[iLayer]);

    cub::DeviceScan::ExclusiveSum(static_cast<void *>(primaryVertexContext.getTempTableArray()[iLayer].get()), tempSize[iLayer],
        primaryVertexContext.getDeviceTrackletsPerClustersTable()[iLayer].get(),
        primaryVertexContext.getDeviceTrackletsLookupTable()[iLayer].get(),
        primaryVertexContext.getClusters()[iLayer + 1].size(), streamArray[iLayer + 1].get());

    dim3 threadsPerBlock { GPU::Utils::Host::getBlockSize(trackletsNum[iLayer]) };
    dim3 blocksGrid { GPU::Utils::Host::getBlocksGrid(threadsPerBlock, trackletsNum[iLayer]) };

    GPU::sortTrackletsKernel<<< blocksGrid, threadsPerBlock, 0, streamArray[iLayer + 1].get() >>>(primaryVertexContext.getDeviceContext(),
        iLayer + 1, primaryVertexContext.getTempTrackletArray()[iLayer].getWeakCopy());

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {

      std::ostringstream errorString { };
      errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << ")"
          << std::endl;

      throw std::runtime_error { errorString.str() };
    }
  }
}

template<>
void TrackerTraits<true>::computeLayerCells(CA::PrimaryVertexContext& primaryVertexContext)
{
  std::array<size_t, Constants::ITS::CellsPerRoad - 1> tempSize;
  std::array<int, Constants::ITS::CellsPerRoad - 1> trackletsNum;
  std::array<int, Constants::ITS::CellsPerRoad - 1> cellsNum;
  std::array<GPU::Stream, Constants::ITS::CellsPerRoad> streamArray;

  for (int iLayer { 0 }; iLayer < Constants::ITS::CellsPerRoad - 1; ++iLayer) {

    tempSize[iLayer] = 0;
    trackletsNum[iLayer] = primaryVertexContext.getDeviceTracklets()[iLayer + 1].getSizeFromDevice();
    primaryVertexContext.getTempCellArray()[iLayer].reset(
		static_cast<int>(primaryVertexContext.getDeviceCells()[iLayer + 1].capacity()));

    cub::DeviceScan::ExclusiveSum(static_cast<void *>(NULL), tempSize[iLayer],
        primaryVertexContext.getDeviceCellsPerTrackletTable()[iLayer].get(),
        primaryVertexContext.getDeviceCellsLookupTable()[iLayer].get(), trackletsNum[iLayer]);

    primaryVertexContext.getTempTableArray()[iLayer].reset(static_cast<int>(tempSize[iLayer]));
  }

  cudaDeviceSynchronize();

  for (int iLayer { 0 }; iLayer < Constants::ITS::CellsPerRoad; ++iLayer) {

    const GPU::DeviceProperties& deviceProperties = GPU::Context::getInstance().getDeviceProperties();
    const int trackletsSize = primaryVertexContext.getDeviceTracklets()[iLayer].getSizeFromDevice();
    dim3 threadsPerBlock { GPU::Utils::Host::getBlockSize(trackletsSize) };
    dim3 blocksGrid { GPU::Utils::Host::getBlocksGrid(threadsPerBlock, trackletsSize) };

    if(iLayer == 0) {

      GPU::layerCellsKernel<<< blocksGrid, threadsPerBlock, 0, streamArray[iLayer].get() >>>(primaryVertexContext.getDeviceContext(),
          iLayer, primaryVertexContext.getDeviceCells()[iLayer].getWeakCopy());

    } else {

      GPU::layerCellsKernel<<< blocksGrid, threadsPerBlock, 0, streamArray[iLayer].get() >>>(primaryVertexContext.getDeviceContext(),
          iLayer, primaryVertexContext.getTempCellArray()[iLayer - 1].getWeakCopy());
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

  for (int iLayer { 0 }; iLayer < Constants::ITS::CellsPerRoad - 1; ++iLayer) {

    cellsNum[iLayer] = primaryVertexContext.getTempCellArray()[iLayer].getSizeFromDevice();
    primaryVertexContext.getDeviceCells()[iLayer + 1].resize(cellsNum[iLayer]);

    cub::DeviceScan::ExclusiveSum(static_cast<void *>(primaryVertexContext.getTempTableArray()[iLayer].get()), tempSize[iLayer],
        primaryVertexContext.getDeviceCellsPerTrackletTable()[iLayer].get(),
        primaryVertexContext.getDeviceCellsLookupTable()[iLayer].get(), trackletsNum[iLayer],
        streamArray[iLayer + 1].get());

    dim3 threadsPerBlock { GPU::Utils::Host::getBlockSize(trackletsNum[iLayer]) };
    dim3 blocksGrid { GPU::Utils::Host::getBlocksGrid(threadsPerBlock, trackletsNum[iLayer]) };

    GPU::sortCellsKernel<<< blocksGrid, threadsPerBlock, 0, streamArray[iLayer + 1].get() >>>(primaryVertexContext.getDeviceContext(),
        iLayer + 1, primaryVertexContext.getTempCellArray()[iLayer].getWeakCopy());

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {

      std::ostringstream errorString { };
      errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << ")"
          << std::endl;

      throw std::runtime_error { errorString.str() };
    }
  }

  cudaDeviceSynchronize();

  for (int iLayer { 0 }; iLayer < Constants::ITS::CellsPerRoad; ++iLayer) {

    int cellsSize;

    if (iLayer == 0) {

      cellsSize = primaryVertexContext.getDeviceCells()[iLayer].getSizeFromDevice();

    } else {

      cellsSize = cellsNum[iLayer - 1];

      primaryVertexContext.getDeviceCellsLookupTable()[iLayer - 1].copyIntoVector(
          primaryVertexContext.getCellsLookupTable()[iLayer - 1], trackletsNum[iLayer - 1]);
    }

    primaryVertexContext.getDeviceCells()[iLayer].copyIntoVector(primaryVertexContext.getCells()[iLayer], cellsSize);
  }
}

}
}
}
