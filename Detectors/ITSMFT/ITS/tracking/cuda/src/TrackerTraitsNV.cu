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

#ifndef GPUCA_GPUCODE_GENRTC
#include <cooperative_groups.h>
#include "cub/cub.cuh"
#endif

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

using namespace constants::its2;
GPU_DEVICE const int4 getBinsRect(const Cluster& currentCluster, const int layerIndex,
                                  const float z1, const float z2, float maxdeltaz, float maxdeltaphi)
{
  const float zRangeMin = o2::gpu::GPUCommonMath::Min(z1, z2) - maxdeltaz;
  const float phiRangeMin = currentCluster.phiCoordinate - maxdeltaphi;
  const float zRangeMax = o2::gpu::GPUCommonMath::Max(z1, z2) + maxdeltaz;
  const float phiRangeMax = currentCluster.phiCoordinate + maxdeltaphi;

  if (zRangeMax < -LayersZCoordinate()[layerIndex + 1] ||
      zRangeMin > LayersZCoordinate()[layerIndex + 1] || zRangeMin > zRangeMax) {

    return getEmptyBinsRect();
  }

  return int4{o2::gpu::GPUCommonMath::Max(0, getZBinIndex(layerIndex + 1, zRangeMin)),
              getPhiBinIndex(phiRangeMin),
              o2::gpu::GPUCommonMath::Min(ZBins - 1, getZBinIndex(layerIndex + 1, zRangeMax)),
              getPhiBinIndex(phiRangeMax)};
}

namespace gpu
{

struct StaticTrackingParameters {
  StaticTrackingParameters& operator=(const StaticTrackingParameters& t);

  int CellMinimumLevel();

  /// General parameters
  int ClusterSharing = 0;
  int MinTrackLength = 7;
  /// Trackleting cuts
  float TrackletMaxDeltaPhi = 0.3f;
  float TrackletMaxDeltaZ[constants::its2::TrackletsPerRoad] = {0.1f, 0.1f, 0.3f, 0.3f, 0.3f, 0.3f};
  /// Cell finding cuts
  float CellMaxDeltaTanLambda = 0.025f;
  float CellMaxDCA[constants::its2::CellsPerRoad] = {0.05f, 0.04f, 0.05f, 0.2f, 0.4f};
  float CellMaxDeltaPhi = 0.14f;
  float CellMaxDeltaZ[constants::its2::CellsPerRoad] = {0.2f, 0.4f, 0.5f, 0.6f, 3.0f};
  /// Neighbour finding cuts
  float NeighbourMaxDeltaCurvature[constants::its2::CellsPerRoad - 1] = {0.008f, 0.0025f, 0.003f, 0.0035f};
  float NeighbourMaxDeltaN[constants::its2::CellsPerRoad - 1] = {0.002f, 0.0090f, 0.002f, 0.005f};
};
__constant__ StaticTrackingParameters kTrkPar;

__device__ void computeLayerTracklets(DeviceStoreNV& devStore, const int layerIndex,
                                      Vector<Tracklet>& trackletsVector)
{
  const int currentClusterIndex = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  int clusterTrackletsNum = 0;

  if (currentClusterIndex < devStore.getClusters()[layerIndex].size()) {

    Vector<Cluster> nextLayerClusters{devStore.getClusters()[layerIndex + 1].getWeakCopy()};
    const Cluster currentCluster{devStore.getClusters()[layerIndex][currentClusterIndex]};

    /*if (mUsedClustersTable[currentCluster.clusterId] != constants::its::UnusedIndex) {

     continue;
     }*/

    const float tanLambda{(currentCluster.zCoordinate - devStore.getPrimaryVertex().z) / currentCluster.rCoordinate};
    const float zAtRmin{tanLambda * (devStore.getRmin(layerIndex + 1) - currentCluster.rCoordinate) + currentCluster.zCoordinate};
    const float zAtRmax{tanLambda * (devStore.getRmax(layerIndex + 1) - currentCluster.rCoordinate) + currentCluster.zCoordinate};

    const int4 selectedBinsRect{getBinsRect(currentCluster, layerIndex, zAtRmin, zAtRmax,
                                            kTrkPar.TrackletMaxDeltaZ[layerIndex], kTrkPar.TrackletMaxDeltaPhi)};

    if (selectedBinsRect.x != 0 || selectedBinsRect.y != 0 || selectedBinsRect.z != 0 || selectedBinsRect.w != 0) {

      const int nextLayerClustersNum{static_cast<int>(nextLayerClusters.size())};
      int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};

      if (phiBinsNum < 0) {

        phiBinsNum += constants::its2::PhiBins;
      }

      for (int iPhiBin{selectedBinsRect.y}, iPhiCount{0}; iPhiCount < phiBinsNum;
           iPhiBin = ++iPhiBin == constants::its2::PhiBins ? 0 : iPhiBin, iPhiCount++) {

        const int firstBinIndex{constants::its2::getBinIndex(selectedBinsRect.x, iPhiBin)};
        const int firstRowClusterIndex = devStore.getIndexTables()[layerIndex][firstBinIndex];
        const int maxRowClusterIndex = devStore.getIndexTables()[layerIndex][{firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1}];

        for (int iNextLayerCluster{firstRowClusterIndex};
             iNextLayerCluster <= maxRowClusterIndex && iNextLayerCluster < nextLayerClustersNum; ++iNextLayerCluster) {

          const Cluster& nextCluster{nextLayerClusters[iNextLayerCluster]};

          const float deltaZ{o2::gpu::GPUCommonMath::Abs(
            tanLambda * (nextCluster.rCoordinate - currentCluster.rCoordinate) + currentCluster.zCoordinate - nextCluster.zCoordinate)};
          const float deltaPhi{o2::gpu::GPUCommonMath::Abs(currentCluster.phiCoordinate - nextCluster.phiCoordinate)};

          if (deltaZ < kTrkPar.TrackletMaxDeltaZ[layerIndex] && (deltaPhi < kTrkPar.TrackletMaxDeltaPhi || o2::gpu::GPUCommonMath::Abs(deltaPhi - constants::math::TwoPi) < kTrkPar.TrackletMaxDeltaPhi)) {

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
  const float3& primaryVertex = devStore.getPrimaryVertex();
  int trackletCellsNum = 0;

  if (currentTrackletIndex < devStore.getTracklets()[layerIndex].size()) {

    const Tracklet& currentTracklet{devStore.getTracklets()[layerIndex][currentTrackletIndex]};
    const int nextLayerClusterIndex{currentTracklet.secondClusterIndex};
    const int nextLayerFirstTrackletIndex{
      devStore.getTrackletsLookupTable()[layerIndex][nextLayerClusterIndex]};
    const int nextLayerTrackletsNum{static_cast<int>(devStore.getTracklets()[layerIndex + 1].size())};

    if (devStore.getTracklets()[layerIndex + 1][nextLayerFirstTrackletIndex].firstClusterIndex == nextLayerClusterIndex) {

      const Cluster& firstCellCluster{
        devStore.getClusters()[layerIndex][currentTracklet.firstClusterIndex]};
      const Cluster& secondCellCluster{
        devStore.getClusters()[layerIndex + 1][currentTracklet.secondClusterIndex]};
      const float firstCellClusterQuadraticRCoordinate{firstCellCluster.rCoordinate * firstCellCluster.rCoordinate};
      const float secondCellClusterQuadraticRCoordinate{secondCellCluster.rCoordinate * secondCellCluster.rCoordinate};
      const float3 firstDeltaVector{secondCellCluster.xCoordinate - firstCellCluster.xCoordinate,
                                    secondCellCluster.yCoordinate - firstCellCluster.yCoordinate, secondCellClusterQuadraticRCoordinate - firstCellClusterQuadraticRCoordinate};

      for (int iNextLayerTracklet{nextLayerFirstTrackletIndex};
           iNextLayerTracklet < nextLayerTrackletsNum && devStore.getTracklets()[layerIndex + 1][iNextLayerTracklet].firstClusterIndex == nextLayerClusterIndex; ++iNextLayerTracklet) {

        const Tracklet& nextTracklet{devStore.getTracklets()[layerIndex + 1][iNextLayerTracklet]};
        const float deltaTanLambda{o2::gpu::GPUCommonMath::Abs(currentTracklet.tanLambda - nextTracklet.tanLambda)};
        const float deltaPhi{o2::gpu::GPUCommonMath::Abs(currentTracklet.phiCoordinate - nextTracklet.phiCoordinate)};

        if (deltaTanLambda < kTrkPar.CellMaxDeltaTanLambda && (deltaPhi < kTrkPar.CellMaxDeltaPhi || o2::gpu::GPUCommonMath::Abs(deltaPhi - constants::math::TwoPi) < kTrkPar.CellMaxDeltaPhi)) {

          const float averageTanLambda{0.5f * (currentTracklet.tanLambda + nextTracklet.tanLambda)};
          const float directionZIntersection{-averageTanLambda * firstCellCluster.rCoordinate + firstCellCluster.zCoordinate};
          const float deltaZ{o2::gpu::GPUCommonMath::Abs(directionZIntersection - primaryVertex.z)};

          if (deltaZ < kTrkPar.CellMaxDeltaZ[layerIndex]) {

            const Cluster& thirdCellCluster{
              devStore.getClusters()[layerIndex + 2][nextTracklet.secondClusterIndex]};

            const float thirdCellClusterQuadraticRCoordinate{thirdCellCluster.rCoordinate * thirdCellCluster.rCoordinate};

            const float3 secondDeltaVector{thirdCellCluster.xCoordinate - firstCellCluster.xCoordinate,
                                           thirdCellCluster.yCoordinate - firstCellCluster.yCoordinate, thirdCellClusterQuadraticRCoordinate - firstCellClusterQuadraticRCoordinate};

            float3 cellPlaneNormalVector{math_utils::crossProduct(firstDeltaVector, secondDeltaVector)};

            const float vectorNorm{o2::gpu::GPUCommonMath::Sqrt(
              cellPlaneNormalVector.x * cellPlaneNormalVector.x + cellPlaneNormalVector.y * cellPlaneNormalVector.y + cellPlaneNormalVector.z * cellPlaneNormalVector.z)};

            if (!(vectorNorm < constants::math::FloatMinThreshold || o2::gpu::GPUCommonMath::Abs(cellPlaneNormalVector.z) < constants::math::FloatMinThreshold)) {

              const float inverseVectorNorm{1.0f / vectorNorm};
              const float3 normalizedPlaneVector{cellPlaneNormalVector.x * inverseVectorNorm, cellPlaneNormalVector.y * inverseVectorNorm, cellPlaneNormalVector.z * inverseVectorNorm};
              const float planeDistance{-normalizedPlaneVector.x * (secondCellCluster.xCoordinate - primaryVertex.x) - (normalizedPlaneVector.y * secondCellCluster.yCoordinate - primaryVertex.y) - normalizedPlaneVector.z * secondCellClusterQuadraticRCoordinate};
              const float normalizedPlaneVectorQuadraticZCoordinate{normalizedPlaneVector.z * normalizedPlaneVector.z};
              const float cellTrajectoryRadius{o2::gpu::GPUCommonMath::Sqrt(
                (1.0f - normalizedPlaneVectorQuadraticZCoordinate - 4.0f * planeDistance * normalizedPlaneVector.z) / (4.0f * normalizedPlaneVectorQuadraticZCoordinate))};
              const float2 circleCenter{-0.5f * normalizedPlaneVector.x / normalizedPlaneVector.z, -0.5f * normalizedPlaneVector.y / normalizedPlaneVector.z};
              const float distanceOfClosestApproach{o2::gpu::GPUCommonMath::Abs(
                cellTrajectoryRadius - o2::gpu::GPUCommonMath::Sqrt(circleCenter.x * circleCenter.x + circleCenter.y * circleCenter.y))};

              if (distanceOfClosestApproach <= kTrkPar.CellMaxDCA[layerIndex]) {

                cooperative_groups::coalesced_group threadGroup = cooperative_groups::coalesced_threads();
                int currentIndex{};

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
  const int currentTrackletIndex{static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x)};

  if (currentTrackletIndex < tempTrackletArray.size()) {

    const int firstClusterIndex = tempTrackletArray[currentTrackletIndex].firstClusterIndex;
    const int offset = atomicAdd(&devStore.getTrackletsPerClusterTable()[layerIndex - 1][firstClusterIndex],
                                 -1) -
                       1;
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
                                 -1) -
                       1;
    const int startIndex = devStore.getCellsLookupTable()[layerIndex - 1][firstTrackletIndex];

    memcpy(&devStore.getCells()[layerIndex][startIndex + offset], &tempCellsArray[currentCellIndex],
           sizeof(Cell));
  }
}

} // namespace gpu

TrackerTraits* createTrackerTraitsNV()
{
  return new TrackerTraitsNV;
}

TrackerTraitsNV::TrackerTraitsNV()
{
  mPrimaryVertexContext = new PrimaryVertexContextNV;
}

TrackerTraitsNV::~TrackerTraitsNV()
{
  delete mPrimaryVertexContext;
}

void TrackerTraitsNV::computeLayerTracklets()
{
  PrimaryVertexContextNV* primaryVertexContext = static_cast<PrimaryVertexContextNV*>(mPrimaryVertexContext);

  // cudaMemcpyToSymbol(gpu::kTrkPar, &mTrkParams, sizeof(TrackingParameters));
  std::array<size_t, constants::its2::CellsPerRoad> tempSize;
  std::array<int, constants::its2::CellsPerRoad> trackletsNum;
  std::array<gpu::Stream, constants::its2::TrackletsPerRoad> streamArray;

  for (int iLayer{0}; iLayer < constants::its2::CellsPerRoad; ++iLayer) {

    tempSize[iLayer] = 0;
    primaryVertexContext->getTempTrackletArray()[iLayer].reset(
      static_cast<int>(primaryVertexContext->getDeviceTracklets()[iLayer + 1].capacity()));

    cub::DeviceScan::ExclusiveSum(static_cast<void*>(NULL), tempSize[iLayer],
                                  primaryVertexContext->getDeviceTrackletsPerClustersTable()[iLayer].get(),
                                  primaryVertexContext->getDeviceTrackletsLookupTable()[iLayer].get(),
                                  primaryVertexContext->getClusters()[iLayer + 1].size());

    primaryVertexContext->getTempTableArray()[iLayer].reset(static_cast<int>(tempSize[iLayer]));
  }

  cudaDeviceSynchronize();

  for (int iLayer{0}; iLayer < constants::its2::TrackletsPerRoad; ++iLayer) {

    const gpu::DeviceProperties& deviceProperties = gpu::Context::getInstance().getDeviceProperties();
    const int clustersNum{static_cast<int>(primaryVertexContext->getClusters()[iLayer].size())};
    dim3 threadsPerBlock{gpu::utils::host::getBlockSize(clustersNum, 1, 192)};
    dim3 blocksGrid{gpu::utils::host::getBlocksGrid(threadsPerBlock, clustersNum)};

    if (iLayer == 0) {

      gpu::layerTrackletsKernel<<<blocksGrid, threadsPerBlock, 0, streamArray[iLayer].get()>>>(primaryVertexContext->getDeviceContext(),
                                                                                               iLayer, primaryVertexContext->getDeviceTracklets()[iLayer].getWeakCopy());

    } else {

      gpu::layerTrackletsKernel<<<blocksGrid, threadsPerBlock, 0, streamArray[iLayer].get()>>>(primaryVertexContext->getDeviceContext(),
                                                                                               iLayer, primaryVertexContext->getTempTrackletArray()[iLayer - 1].getWeakCopy());
    }

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {

      std::ostringstream errorString{};
      errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << ")"
                  << std::endl;

      throw std::runtime_error{errorString.str()};
    }
  }

  cudaDeviceSynchronize();

  for (int iLayer{0}; iLayer < constants::its2::CellsPerRoad; ++iLayer) {

    trackletsNum[iLayer] = primaryVertexContext->getTempTrackletArray()[iLayer].getSizeFromDevice();
    if (trackletsNum[iLayer] == 0) {
      continue;
    }
    primaryVertexContext->getDeviceTracklets()[iLayer + 1].resize(trackletsNum[iLayer]);

    cub::DeviceScan::ExclusiveSum(static_cast<void*>(primaryVertexContext->getTempTableArray()[iLayer].get()), tempSize[iLayer],
                                  primaryVertexContext->getDeviceTrackletsPerClustersTable()[iLayer].get(),
                                  primaryVertexContext->getDeviceTrackletsLookupTable()[iLayer].get(),
                                  primaryVertexContext->getClusters()[iLayer + 1].size(), streamArray[iLayer + 1].get());

    dim3 threadsPerBlock{gpu::utils::host::getBlockSize(trackletsNum[iLayer])};
    dim3 blocksGrid{gpu::utils::host::getBlocksGrid(threadsPerBlock, trackletsNum[iLayer])};

    gpu::sortTrackletsKernel<<<blocksGrid, threadsPerBlock, 0, streamArray[iLayer + 1].get()>>>(primaryVertexContext->getDeviceContext(),
                                                                                                iLayer + 1, primaryVertexContext->getTempTrackletArray()[iLayer].getWeakCopy());

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {

      std::ostringstream errorString{};
      errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << ")"
                  << std::endl;

      throw std::runtime_error{errorString.str()};
    }
  }
}

void TrackerTraitsNV::computeLayerCells()
{

  PrimaryVertexContextNV* primaryVertexContext = static_cast<PrimaryVertexContextNV*>(mPrimaryVertexContext);
  std::array<size_t, constants::its2::CellsPerRoad - 1> tempSize;
  std::array<int, constants::its2::CellsPerRoad - 1> trackletsNum;
  std::array<int, constants::its2::CellsPerRoad - 1> cellsNum;
  std::array<gpu::Stream, constants::its2::CellsPerRoad> streamArray;

  for (int iLayer{0}; iLayer < constants::its2::CellsPerRoad - 1; ++iLayer) {

    tempSize[iLayer] = 0;
    trackletsNum[iLayer] = primaryVertexContext->getDeviceTracklets()[iLayer + 1].getSizeFromDevice();
    primaryVertexContext->getTempCellArray()[iLayer].reset(
      static_cast<int>(primaryVertexContext->getDeviceCells()[iLayer + 1].capacity()));
    if (trackletsNum[iLayer] == 0) {
      continue;
    }
    cub::DeviceScan::ExclusiveSum(static_cast<void*>(NULL), tempSize[iLayer],
                                  primaryVertexContext->getDeviceCellsPerTrackletTable()[iLayer].get(),
                                  primaryVertexContext->getDeviceCellsLookupTable()[iLayer].get(), trackletsNum[iLayer]);

    primaryVertexContext->getTempTableArray()[iLayer].reset(static_cast<int>(tempSize[iLayer]));
  }

  cudaDeviceSynchronize();

  for (int iLayer{0}; iLayer < constants::its2::CellsPerRoad; ++iLayer) {
    const gpu::DeviceProperties& deviceProperties = gpu::Context::getInstance().getDeviceProperties();
    const int trackletsSize = primaryVertexContext->getDeviceTracklets()[iLayer].getSizeFromDevice();
    if (trackletsSize == 0) {
      continue;
    }
    dim3 threadsPerBlock{gpu::utils::host::getBlockSize(trackletsSize)};
    dim3 blocksGrid{gpu::utils::host::getBlocksGrid(threadsPerBlock, trackletsSize)};

    if (iLayer == 0) {

      gpu::layerCellsKernel<<<blocksGrid, threadsPerBlock, 0, streamArray[iLayer].get()>>>(primaryVertexContext->getDeviceContext(),
                                                                                           iLayer, primaryVertexContext->getDeviceCells()[iLayer].getWeakCopy());

    } else {

      gpu::layerCellsKernel<<<blocksGrid, threadsPerBlock, 0, streamArray[iLayer].get()>>>(primaryVertexContext->getDeviceContext(),
                                                                                           iLayer, primaryVertexContext->getTempCellArray()[iLayer - 1].getWeakCopy());
    }

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {

      std::ostringstream errorString{};
      errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << ")"
                  << std::endl;

      throw std::runtime_error{errorString.str()};
    }
  }

  cudaDeviceSynchronize();

  for (int iLayer{0}; iLayer < constants::its2::CellsPerRoad - 1; ++iLayer) {
    cellsNum[iLayer] = primaryVertexContext->getTempCellArray()[iLayer].getSizeFromDevice();
    if (cellsNum[iLayer] == 0) {
      continue;
    }
    primaryVertexContext->getDeviceCells()[iLayer + 1].resize(cellsNum[iLayer]);

    cub::DeviceScan::ExclusiveSum(static_cast<void*>(primaryVertexContext->getTempTableArray()[iLayer].get()), tempSize[iLayer],
                                  primaryVertexContext->getDeviceCellsPerTrackletTable()[iLayer].get(),
                                  primaryVertexContext->getDeviceCellsLookupTable()[iLayer].get(), trackletsNum[iLayer],
                                  streamArray[iLayer + 1].get());

    dim3 threadsPerBlock{gpu::utils::host::getBlockSize(trackletsNum[iLayer])};
    dim3 blocksGrid{gpu::utils::host::getBlocksGrid(threadsPerBlock, trackletsNum[iLayer])};

    gpu::sortCellsKernel<<<blocksGrid, threadsPerBlock, 0, streamArray[iLayer + 1].get()>>>(primaryVertexContext->getDeviceContext(),
                                                                                            iLayer + 1, primaryVertexContext->getTempCellArray()[iLayer].getWeakCopy());

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {

      std::ostringstream errorString{};
      errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << ")"
                  << std::endl;

      throw std::runtime_error{errorString.str()};
    }
  }

  cudaDeviceSynchronize();

  for (int iLayer{0}; iLayer < constants::its2::CellsPerRoad; ++iLayer) {

    int cellsSize = 0;
    if (iLayer == 0) {

      cellsSize = primaryVertexContext->getDeviceCells()[iLayer].getSizeFromDevice();
      if (cellsSize == 0) {
        continue;
      }
    } else {

      cellsSize = cellsNum[iLayer - 1];
      if (cellsSize == 0) {
        continue;
      }
      primaryVertexContext->getDeviceCellsLookupTable()[iLayer - 1].copyIntoVector(
        primaryVertexContext->getCellsLookupTable()[iLayer - 1], trackletsNum[iLayer - 1]);
    }

    primaryVertexContext->getDeviceCells()[iLayer].copyIntoVector(primaryVertexContext->getCells()[iLayer], cellsSize);
  }
}

void TrackerTraitsNV::refitTracks(const std::vector<std::vector<TrackingFrameInfo>>& tf, std::vector<TrackITSExt>& tracks)
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
  //TODO: restore this
  // mChainRunITSTrackFit(*mChain, mPrimaryVertexContext->getRoads(), clusters, cells, tf, tracks);
}
} // namespace its
} // namespace o2
