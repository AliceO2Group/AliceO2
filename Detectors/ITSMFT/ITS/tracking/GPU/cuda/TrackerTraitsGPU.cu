// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///

#include <array>
#include <sstream>
#include <iostream>

#include "ITStracking/Constants.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/MathUtils.h"

#include "ITStrackingGPU/Stream.h"
#include "ITStrackingGPU/TrackerTraitsGPU.h"

namespace o2
{
namespace its
{
using gpu::utils::host::checkGPUError;
using namespace constants::its2;

namespace gpu
{
__constant__ StaticTrackingParameters<7> kTrkPar;

GPUg() void computeLayerTrackletsKernel(const int layerIndex,
                                        const Cluster* clustersNextLayer,
                                        const Cluster* clustersCurrentLayer,
                                        unsigned char* usedClusters,
                                        Tracklet* trackletsVector,
                                        const int nVertices,
                                        const int startSortedIndex,
                                        const int sizeCurrentLClusters,
                                        const float minR,
                                        const float maxR,
                                        const float meanDeltaR)
{
  int clusterTrackletsNum = 0;
  for (int currentClusterIndex = blockIdx.x * blockDim.x + threadIdx.x; currentClusterIndex < sizeCurrentLClusters; currentClusterIndex += blockDim.x * gridDim.x) {
    const Cluster& currentCluster{clustersCurrentLayer[currentClusterIndex]};
    const int currentSortedIndex{startSortedIndex + currentClusterIndex};
    if (usedClusters[currentSortedIndex]) {
      continue;
    }
    const float inverseR0{1.f / currentCluster.radius};
    for (int iPrimaryVertex{0}; iPrimaryVertex < nVertices; iPrimaryVertex++) {
    }
    // const float tanLambda{(currentCluster.zCoordinate - devStore.getPrimaryVertex().z) / currentCluster.radius};
    // const float zAtRmin{tanLambda * (devStore.getRmin(layerIndex + 1) - currentCluster.radius) + currentCluster.zCoordinate};
    // const float zAtRmax{tanLambda * (devStore.getRmax(layerIndex + 1) - currentCluster.radius) + currentCluster.zCoordinate};

    // const int4 selectedBinsRect{getBinsRect(currentCluster, layerIndex, zAtRmin, zAtRmax,
    //                                         kTrkPar.TrackletMaxDeltaZ[layerIndex], kTrkPar.TrackletMaxDeltaPhi)};
    // if (selectedBinsRect.x != 0 || selectedBinsRect.y != 0 || selectedBinsRect.z != 0 || selectedBinsRect.w != 0) {
    //   const int nextLayerClustersNum{static_cast<int>(nextLayerClusters.size())};
    //   int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};
    //   if (phiBinsNum < 0) {
    //     phiBinsNum += constants::its2::PhiBins;
    //   }
    //   for (int iPhiBin{selectedBinsRect.y}, iPhiCount{0}; iPhiCount < phiBinsNum;
    //        iPhiBin = ++iPhiBin == constants::its2::PhiBins ? 0 : iPhiBin, iPhiCount++) {
    //     const int firstBinIndex{constants::its2::getBinIndex(selectedBinsRect.x, iPhiBin)};
    //     const int firstRowClusterIndex = devStore.getIndexTables()[layerIndex][firstBinIndex];
    //     const int maxRowClusterIndex = devStore.getIndexTables()[layerIndex][{firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1}];
    //     for (int iNextLayerCluster{firstRowClusterIndex};
    //          iNextLayerCluster <= maxRowClusterIndex && iNextLayerCluster < nextLayerClustersNum; ++iNextLayerCluster) {
    //       const Cluster& nextCluster{nextLayerClusters[iNextLayerCluster]};
    //       const float deltaZ{o2::gpu::GPUCommonMath::Abs(
    //         tanLambda * (nextCluster.radius - currentCluster.radius) + currentCluster.zCoordinate - nextCluster.zCoordinate)};
    //       const float deltaPhi{o2::gpu::GPUCommonMath::Abs(currentCluster.phi - nextCluster.phi)};
    //       if (deltaZ < kTrkPar.TrackletMaxDeltaZ[layerIndex] && (deltaPhi < kTrkPar.TrackletMaxDeltaPhi || o2::gpu::GPUCommonMath::Abs(deltaPhi - constants::math::TwoPi) < kTrkPar.TrackletMaxDeltaPhi)) {
    //         cooperative_groups::coalesced_group threadGroup = cooperative_groups::coalesced_threads();
    //         int currentIndex{};
    //         if (threadGroup.thread_rank() == 0) {
    //           currentIndex = trackletsVector.extend(threadGroup.size());
    //         }
    //         currentIndex = threadGroup.shfl(currentIndex, 0) + threadGroup.thread_rank();
    //         trackletsVector.emplace(currentIndex, currentClusterIndex, iNextLayerCluster, currentCluster, nextCluster);
    //         ++clusterTrackletsNum;
    //       }
    //     }
    //   }
    //   if (layerIndex > 0) {
    //     devStore.getTrackletsPerClusterTable()[layerIndex - 1][currentClusterIndex] = clusterTrackletsNum;
    //   }
    // }
  }
}

// GPUd() void computeLayerCells(DeviceStoreNV& devStore, const int layerIndex,
//                               Vector<Cell>& cellsVector)
// {
//   const int currentTrackletIndex = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
//   const float3& primaryVertex = devStore.getPrimaryVertex();
//   int trackletCellsNum = 0;
//   if (currentTrackletIndex < devStore.getTracklets()[layerIndex].size()) {
//     const Tracklet& currentTracklet{devStore.getTracklets()[layerIndex][currentTrackletIndex]};
//     const int nextLayerClusterIndex{currentTracklet.secondClusterIndex};
//     const int nextLayerFirstTrackletIndex{
//       devStore.getTrackletsLookupTable()[layerIndex][nextLayerClusterIndex]};
//     const int nextLayerTrackletsNum{static_cast<int>(devStore.getTracklets()[layerIndex + 1].size())};
//     if (devStore.getTracklets()[layerIndex + 1][nextLayerFirstTrackletIndex].firstClusterIndex == nextLayerClusterIndex) {
//       const Cluster& firstCellCluster{
//         devStore.getClusters()[layerIndex][currentTracklet.firstClusterIndex]};
//       const Cluster& secondCellCluster{
//         devStore.getClusters()[layerIndex + 1][currentTracklet.secondClusterIndex]};
//       const float firstCellClusterQuadraticRCoordinate{firstCellCluster.radius * firstCellCluster.radius};
//       const float secondCellClusterQuadraticRCoordinate{secondCellCluster.radius * secondCellCluster.radius};
//       const float3 firstDeltaVector{secondCellCluster.xCoordinate - firstCellCluster.xCoordinate,
//                                     secondCellCluster.yCoordinate - firstCellCluster.yCoordinate, secondCellClusterQuadraticRCoordinate - firstCellClusterQuadraticRCoordinate};
//       for (int iNextLayerTracklet{nextLayerFirstTrackletIndex};
//            iNextLayerTracklet < nextLayerTrackletsNum && devStore.getTracklets()[layerIndex + 1][iNextLayerTracklet].firstClusterIndex == nextLayerClusterIndex; ++iNextLayerTracklet) {
//         const Tracklet& nextTracklet{devStore.getTracklets()[layerIndex + 1][iNextLayerTracklet]};
//         const float deltaTanLambda{o2::gpu::GPUCommonMath::Abs(currentTracklet.tanLambda - nextTracklet.tanLambda)};
//         const float deltaPhi{o2::gpu::GPUCommonMath::Abs(currentTracklet.phi - nextTracklet.phi)};
//         if (deltaTanLambda < kTrkPar.CellMaxDeltaTanLambda && (deltaPhi < kTrkPar.CellMaxDeltaPhi || o2::gpu::GPUCommonMath::Abs(deltaPhi - constants::math::TwoPi) < kTrkPar.CellMaxDeltaPhi)) {
//           const float averageTanLambda{0.5f * (currentTracklet.tanLambda + nextTracklet.tanLambda)};
//           const float directionZIntersection{-averageTanLambda * firstCellCluster.radius + firstCellCluster.zCoordinate};
//           const float deltaZ{o2::gpu::GPUCommonMath::Abs(directionZIntersection - primaryVertex.z)};
//           if (deltaZ < kTrkPar.CellMaxDeltaZ[layerIndex]) {
//             const Cluster& thirdCellCluster{
//               devStore.getClusters()[layerIndex + 2][nextTracklet.secondClusterIndex]};
//             const float thirdCellClusterQuadraticRCoordinate{thirdCellCluster.radius * thirdCellCluster.radius};
//             const float3 secondDeltaVector{thirdCellCluster.xCoordinate - firstCellCluster.xCoordinate,
//                                            thirdCellCluster.yCoordinate - firstCellCluster.yCoordinate, thirdCellClusterQuadraticRCoordinate - firstCellClusterQuadraticRCoordinate};
//             float3 cellPlaneNormalVector{math_utils::crossProduct(firstDeltaVector, secondDeltaVector)};
//             const float vectorNorm{o2::gpu::GPUCommonMath::Sqrt(
//               cellPlaneNormalVector.x * cellPlaneNormalVector.x + cellPlaneNormalVector.y * cellPlaneNormalVector.y + cellPlaneNormalVector.z * cellPlaneNormalVector.z)};
//             if (!(vectorNorm < constants::math::FloatMinThreshold || o2::gpu::GPUCommonMath::Abs(cellPlaneNormalVector.z) < constants::math::FloatMinThreshold)) {
//               const float inverseVectorNorm{1.0f / vectorNorm};
//               const float3 normalizedPlaneVector{cellPlaneNormalVector.x * inverseVectorNorm, cellPlaneNormalVector.y * inverseVectorNorm, cellPlaneNormalVector.z * inverseVectorNorm};
//               const float planeDistance{-normalizedPlaneVector.x * (secondCellCluster.xCoordinate - primaryVertex.x) - (normalizedPlaneVector.y * secondCellCluster.yCoordinate - primaryVertex.y) - normalizedPlaneVector.z * secondCellClusterQuadraticRCoordinate};
//               const float normalizedPlaneVectorQuadraticZCoordinate{normalizedPlaneVector.z * normalizedPlaneVector.z};
//               const float cellTrajectoryRadius{o2::gpu::GPUCommonMath::Sqrt(
//                 (1.0f - normalizedPlaneVectorQuadraticZCoordinate - 4.0f * planeDistance * normalizedPlaneVector.z) / (4.0f * normalizedPlaneVectorQuadraticZCoordinate))};
//               const float2 circleCenter{-0.5f * normalizedPlaneVector.x / normalizedPlaneVector.z, -0.5f * normalizedPlaneVector.y / normalizedPlaneVector.z};
//               const float distanceOfClosestApproach{o2::gpu::GPUCommonMath::Abs(
//                 cellTrajectoryRadius - o2::gpu::GPUCommonMath::Sqrt(circleCenter.x * circleCenter.x + circleCenter.y * circleCenter.y))};
//               if (distanceOfClosestApproach <= kTrkPar.CellMaxDCA[layerIndex]) {
//                 cooperative_groups::coalesced_group threadGroup = cooperative_groups::coalesced_threads();
//                 int currentIndex{};
//                 if (threadGroup.thread_rank() == 0) {
//                   currentIndex = cellsVector.extend(threadGroup.size());
//                 }
//                 currentIndex = threadGroup.shfl(currentIndex, 0) + threadGroup.thread_rank();
//                 cellsVector.emplace(currentIndex, currentTracklet.firstClusterIndex,
//                                     nextTracklet.firstClusterIndex, nextTracklet.secondClusterIndex, currentTrackletIndex,
//                                     iNextLayerTracklet, averageTanLambda);
//                 ++trackletCellsNum;
//               }
//             }
//           }
//         }
//       }
//       if (layerIndex > 0) {
//         devStore.getCellsPerTrackletTable()[layerIndex - 1][currentTrackletIndex] = trackletCellsNum;
//       }
//     }
//   }
// }

// GPUg() void sortTrackletsKernel(DeviceStoreNV& devStore, const int layerIndex,
//                                 Vector<Tracklet> tempTrackletArray)
// {
//   const int currentTrackletIndex{static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x)};
//   if (currentTrackletIndex < tempTrackletArray.size()) {
//     const int firstClusterIndex = tempTrackletArray[currentTrackletIndex].firstClusterIndex;
//     const int offset = atomicAdd(&devStore.getTrackletsPerClusterTable()[layerIndex - 1][firstClusterIndex], -1) - 1;
//     const int startIndex = devStore.getTrackletsLookupTable()[layerIndex - 1][firstClusterIndex];
//     memcpy(&devStore.getTracklets()[layerIndex][startIndex + offset],
//            &tempTrackletArray[currentTrackletIndex], sizeof(Tracklet));
//   }
// }

// GPUg() void layerCellsKernel(DeviceStoreNV& devStore, const int layerIndex,
//                              Vector<Cell> cellsVector)
// {
//   computeLayerCells(devStore, layerIndex, cellsVector);
// }

// GPUg() void sortCellsKernel(DeviceStoreNV& devStore, const int layerIndex,
//                             Vector<Cell> tempCellsArray)
// {
//   const int currentCellIndex = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
//   if (currentCellIndex < tempCellsArray.size()) {
//     const int firstTrackletIndex = tempCellsArray[currentCellIndex].getFirstTrackletIndex();
//     const int offset = atomicAdd(&devStore.getCellsPerTrackletTable()[layerIndex - 1][firstTrackletIndex],
//                                  -1) -
//                        1;
//     const int startIndex = devStore.getCellsLookupTable()[layerIndex - 1][firstTrackletIndex];
//     memcpy(&devStore.getCells()[layerIndex][startIndex + offset], &tempCellsArray[currentCellIndex],
//            sizeof(Cell));
//   }
// }

} // namespace gpu

template <int NLayers>
void TrackerTraitsGPU<NLayers>::initialiseTimeFrame(const int iteration, const MemoryParameters& memParams, const TrackingParameters& trackingParams)
{
  mTimeFrameGPU->initialise(iteration, memParams, trackingParams, NLayers);
  setIsGPU(true);
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::computeLayerTracklets(const int iteration)
{
  checkGPUError(cudaMemcpyToSymbol(gpu::kTrkPar, &mTrkParams, sizeof(gpu::StaticTrackingParameters<NLayers>)), __FILE__, __LINE__);
  std::array<unsigned int, NLayers - 2> tempSize;
  // std::array<int, NLayers - 2> trackletsNum;
  std::array<gpu::Stream, NLayers - 1> streamArray;

  // for (int iLayer{0}; iLayer < constants::its2::CellsPerRoad; ++iLayer) {
  //   // primaryVertexContext->getTempTrackletArray()[iLayer].reset(
  //   //   static_cast<int>(primaryVertexContext->getDeviceTracklets()[iLayer + 1].capacity()));
  //   cub::DeviceScan::ExclusiveSum(static_cast<void*>(NULL), tempSize[iLayer],
  //                                 primaryVertexContext->getDeviceTrackletsPerClustersTable()[iLayer].get(),
  //                                 primaryVertexContext->getDeviceTrackletsLookupTable()[iLayer].get(),
  //                                 primaryVertexContext->getClusters()[iLayer + 1].size()); // N clusters layer+1
  //   primaryVertexContext->getTempTableArray()[iLayer].reset(static_cast<int>(tempSize[iLayer]));
  // }

  // cudaDeviceSynchronize();
  // const Vertex diamondVert({mTrkParams.Diamond[0], mTrkParams.Diamond[1], mTrkParams.Diamond[2]}, {25.e-6f, 0.f, 0.f, 25.e-6f, 0.f, 36.f}, 1, 1.f);
  // gsl::span<const Vertex> diamondSpan(&diamondVert, 1);
  for (int rof0{0}; rof0 < mTimeFrameGPU->getNrof(); ++rof0) {
    for (int iLayer{0}; iLayer < NLayers - 1; ++iLayer) {
      // gsl::span<const Vertex> primaryVertices = mTrkParams.UseDiamond ? diamondSpan : mTimeFrameGPU->getPrimaryVertices(rof0);// replace with GPU one
      int minRof = (rof0 >= mTrkParams.DeltaROF) ? rof0 - mTrkParams.DeltaROF : 0;
      int maxRof = (rof0 == mTimeFrameGPU->getNrof() - mTrkParams.DeltaROF) ? rof0 : rof0 + mTrkParams.DeltaROF;
      const dim3 threadsPerBlock{gpu::utils::host::getBlockSize(mTimeFrameGPU->getNClustersLayer(rof0, iLayer))};
      const dim3 blocksGrid{gpu::utils::host::getBlocksGrid(threadsPerBlock, mTimeFrameGPU->getNClustersLayer(rof0, iLayer))};
      const float meanDeltaR{mTrkParams.LayerRadii[iLayer + 1] - mTrkParams.LayerRadii[iLayer]};
      if (!iLayer) {
        if (!mTimeFrameGPU->getClustersOnLayer(rof0, iLayer).size()) {
          LOGP(info, "Skipping ROF0: {}, no clusters found on layer {}", rof0, iLayer);
          continue;
        }
        // Reset used tracklets
        checkGPUError(cudaMemset(mTimeFrameGPU->getDeviceUsedClustersOnLayer(rof0, iLayer), false, sizeof(unsigned char) * mTimeFrameGPU->getNClustersLayer(rof0, iLayer)), __FILE__, __LINE__);
        gpu::computeLayerTrackletsKernel<<<blocksGrid, threadsPerBlock, 0, streamArray[iLayer].get()>>>(iLayer,
                                                                                                        mTimeFrameGPU->getDeviceClustersOnLayer(rof0, iLayer),
                                                                                                        mTimeFrameGPU->getDeviceClustersOnLayer(rof0, iLayer + 1),
                                                                                                        mTimeFrameGPU->getDeviceUsedClustersOnLayer(rof0, iLayer),
                                                                                                        mTimeFrameGPU->getDeviceTracklets(rof0, iLayer),
                                                                                                        mTimeFrameGPU->getConfig().maxVerticesCapacity,
                                                                                                        mTimeFrameGPU->getSortedStartIndex(rof0, iLayer),
                                                                                                        mTimeFrameGPU->getNClustersLayer(rof0, iLayer),
                                                                                                        mTimeFrameGPU->getMinR(iLayer + 1),
                                                                                                        mTimeFrameGPU->getMaxR(iLayer + 1),
                                                                                                        meanDeltaR);
      } /*else {
        gpu::layerTrackletsKernel<<<blocksGrid, threadsPerBlock , 0, streamArray[iLayer]>>>(primaryVertexContext->getDeviceContext(),
                                                                                                iLayer, primaryVertexContext->getTempTrackletArray()[iLayer - 1].getWeakCopy());
      }*/
    }
  }

  //   cudaDeviceSynchronize();

  //   for (int iLayer{0}; iLayer < constants::its2::CellsPerRoad; ++iLayer) {

  //     trackletsNum[iLayer] = primaryVertexContext->getTempTrackletArray()[iLayer].getSizeFromDevice();
  //     if (trackletsNum[iLayer] == 0) {
  //       continue;
  //     }
  //     primaryVertexContext->getDeviceTracklets()[iLayer + 1].resize(trackletsNum[iLayer]);
  //     cub::DeviceScan::ExclusiveSum(static_cast<void*>(primaryVertexContext->getTempTableArray()[iLayer].get()), tempSize[iLayer],
  //                                   primaryVertexContext->getDeviceTrackletsPerClustersTable()[iLayer].get(),
  //                                   primaryVertexContext->getDeviceTrackletsLookupTable()[iLayer].get(),
  //                                   primaryVertexContext->getClusters()[iLayer + 1].size(), streamArray[iLayer + 1].get());
  //     dim3 threadsPerBlock{gpu::utils::host::getBlockSize(trackletsNum[iLayer])};
  //     dim3 blocksGrid{gpu::utils::host::getBlocksGrid(threadsPerBlock, trackletsNum[iLayer])};
  //     gpu::sortTrackletsKernel<<<blocksGrid, threadsPerBlock, 0, streamArray[iLayer + 1].get()>>>(primaryVertexContext->getDeviceContext(),
  //                                                                                                 iLayer + 1, primaryVertexContext->getTempTrackletArray()[iLayer].getWeakCopy());
  //     cudaError_t error = cudaGetLastError();
  //     if (error != cudaSuccess) {
  //       std::ostringstream errorString{};
  //       errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << ")"
  //                   << std::endl;
  //       throw std::runtime_error{errorString.str()};
  //     }
  //   }
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::computeLayerCells()
{
  //   PrimaryVertexContextNV* primaryVertexContext = static_cast<PrimaryVertexContextNV*>(nullptr); //TODO: FIX THIS with Time Frames
  //   std::array<size_t, constants::its2::CellsPerRoad - 1> tempSize;
  //   std::array<int, constants::its2::CellsPerRoad - 1> trackletsNum;
  //   std::array<int, constants::its2::CellsPerRoad - 1> cellsNum;
  //   std::array<gpu::Stream, constants::its2::CellsPerRoad> streamArray;

  //   for (int iLayer{0}; iLayer < constants::its2::CellsPerRoad - 1; ++iLayer) {
  //     tempSize[iLayer] = 0;
  //     trackletsNum[iLayer] = primaryVertexContext->getDeviceTracklets()[iLayer + 1].getSizeFromDevice();
  //     primaryVertexContext->getTempCellArray()[iLayer].reset(
  //       static_cast<int>(primaryVertexContext->getDeviceCells()[iLayer + 1].capacity()));
  //     if (trackletsNum[iLayer] == 0) {
  //       continue;
  //     }
  //     cub::DeviceScan::ExclusiveSum(static_cast<void*>(NULL), tempSize[iLayer],
  //                                   primaryVertexContext->getDeviceCellsPerTrackletTable()[iLayer].get(),
  //                                   primaryVertexContext->getDeviceCellsLookupTable()[iLayer].get(), trackletsNum[iLayer]);
  //     primaryVertexContext->getTempTableArray()[iLayer].reset(static_cast<int>(tempSize[iLayer]));
  //   }
  //   cudaDeviceSynchronize();
  //   for (int iLayer{0}; iLayer < constants::its2::CellsPerRoad; ++iLayer) {
  //     const gpu::DeviceProperties& deviceProperties = gpu::Context::getInstance().getDeviceProperties();
  //     const int trackletsSize = primaryVertexContext->getDeviceTracklets()[iLayer].getSizeFromDevice();
  //     if (trackletsSize == 0) {
  //       continue;
  //     }
  //     dim3 threadsPerBlock{gpu::utils::host::getBlockSize(trackletsSize)};
  //     dim3 blocksGrid{gpu::utils::host::getBlocksGrid(threadsPerBlock, trackletsSize)};
  //     if (iLayer == 0) {
  //       gpu::layerCellsKernel<<<blocksGrid, threadsPerBlock, 0, streamArray[iLayer].get()>>>(primaryVertexContext->getDeviceContext(),
  //                                                                                            iLayer, primaryVertexContext->getDeviceCells()[iLayer].getWeakCopy());
  //     } else {
  //       gpu::layerCellsKernel<<<blocksGrid, threadsPerBlock, 0, streamArray[iLayer].get()>>>(primaryVertexContext->getDeviceContext(),
  //                                                                                            iLayer, primaryVertexContext->getTempCellArray()[iLayer - 1].getWeakCopy());
  //     }
  //     cudaError_t error = cudaGetLastError();
  //     if (error != cudaSuccess) {
  //       std::ostringstream errorString{};
  //       errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << ")"
  //                   << std::endl;
  //       throw std::runtime_error{errorString.str()};
  //     }
  //   }
  //   cudaDeviceSynchronize();
  //   for (int iLayer{0}; iLayer < constants::its2::CellsPerRoad - 1; ++iLayer) {
  //     cellsNum[iLayer] = primaryVertexContext->getTempCellArray()[iLayer].getSizeFromDevice();
  //     if (cellsNum[iLayer] == 0) {
  //       continue;
  //     }
  //     primaryVertexContext->getDeviceCells()[iLayer + 1].resize(cellsNum[iLayer]);
  //     cub::DeviceScan::ExclusiveSum(static_cast<void*>(primaryVertexContext->getTempTableArray()[iLayer].get()), tempSize[iLayer],
  //                                   primaryVertexContext->getDeviceCellsPerTrackletTable()[iLayer].get(),
  //                                   primaryVertexContext->getDeviceCellsLookupTable()[iLayer].get(), trackletsNum[iLayer],
  //                                   streamArray[iLayer + 1].get());
  //     dim3 threadsPerBlock{gpu::utils::host::getBlockSize(trackletsNum[iLayer])};
  //     dim3 blocksGrid{gpu::utils::host::getBlocksGrid(threadsPerBlock, trackletsNum[iLayer])};
  //     gpu::sortCellsKernel<<<blocksGrid, threadsPerBlock, 0, streamArray[iLayer + 1].get()>>>(primaryVertexContext->getDeviceContext(),
  //                                                                                             iLayer + 1, primaryVertexContext->getTempCellArray()[iLayer].getWeakCopy());
  //     cudaError_t error = cudaGetLastError();
  //     if (error != cudaSuccess) {
  //       std::ostringstream errorString{};
  //       errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << ")"
  //                   << std::endl;
  //       throw std::runtime_error{errorString.str()};
  //     }
  //   }
  //   cudaDeviceSynchronize();
  //   for (int iLayer{0}; iLayer < constants::its2::CellsPerRoad; ++iLayer) {
  //     int cellsSize = 0;
  //     if (iLayer == 0) {
  //       cellsSize = primaryVertexContext->getDeviceCells()[iLayer].getSizeFromDevice();
  //       if (cellsSize == 0) {
  //         continue;
  //       }
  //     } else {
  //       cellsSize = cellsNum[iLayer - 1];
  //       if (cellsSize == 0) {
  //         continue;
  //       }
  //       primaryVertexContext->getDeviceCellsLookupTable()[iLayer - 1].copyIntoVector(
  //         primaryVertexContext->getCellsLookupTable()[iLayer - 1], trackletsNum[iLayer - 1]);
  //     }
  //     primaryVertexContext->getDeviceCells()[iLayer].copyIntoVector(primaryVertexContext->getCells()[iLayer], cellsSize);
  //   }
}

// void TrackerTraitsGPU::refitTracks(const std::vector<std::vector<TrackingFrameInfo>>& tf, std::vector<TrackITSExt>& tracks)
// {
//   PrimaryVertexContextNV* pvctx = static_cast<PrimaryVertexContextNV*>(nullptr); //TODO: FIX THIS with Time Frames
//   std::array<const Cell*, 5> cells;
//   for (int iLayer = 0; iLayer < 5; iLayer++) {
//     cells[iLayer] = pvctx->getDeviceCells()[iLayer].get();
//   }
//   std::array<const Cluster*, 7> clusters;
//   for (int iLayer = 0; iLayer < 7; iLayer++) {
//     clusters[iLayer] = pvctx->getDeviceClusters()[iLayer].get();
//   }
//   //TODO: restore this
//   // mChainRunITSTrackFit(*mChain, mPrimaryVertexContext->getRoads(), clusters, cells, tf, tracks);
// }
template class TrackerTraitsGPU<7>;
} // namespace its
} // namespace o2
