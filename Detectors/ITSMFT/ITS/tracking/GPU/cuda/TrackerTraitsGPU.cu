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

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/unique.h>

#include "ITStracking/Constants.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/MathUtils.h"

#include "ITStrackingGPU/Stream.h"
#include "ITStrackingGPU/TrackerTraitsGPU.h"

#include "GPUCommonLogger.h"
namespace o2
{
namespace its
{
using gpu::utils::host::checkGPUError;
using namespace constants::its2;

namespace gpu
{

GPUd() const int4 getBinsRect(const Cluster& currentCluster, const int layerIndex,
                              const o2::its::IndexTableUtils& utils,
                              const float z1, const float z2, float maxdeltaz, float maxdeltaphi)
{
  const float zRangeMin = o2::gpu::GPUCommonMath::Min(z1, z2) - maxdeltaz;
  const float phiRangeMin = currentCluster.phi - maxdeltaphi;
  const float zRangeMax = o2::gpu::GPUCommonMath::Max(z1, z2) + maxdeltaz;
  const float phiRangeMax = currentCluster.phi + maxdeltaphi;

  if (zRangeMax < -LayersZCoordinate()[layerIndex + 1] ||
      zRangeMin > LayersZCoordinate()[layerIndex + 1] || zRangeMin > zRangeMax) {

    return getEmptyBinsRect();
  }

  return int4{o2::gpu::GPUCommonMath::Max(0, getZBinIndex(layerIndex + 1, zRangeMin)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMin)),
              o2::gpu::GPUCommonMath::Min(ZBins - 1, getZBinIndex(layerIndex + 1, zRangeMax)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMax))};
}

GPUd() float Sq(float q)
{
  return q * q;
}

template <int NLayers = 7>
GPUg() void computeLayerTrackletsKernel(
  const int rof0,
  const int maxRofs,
  const int layerIndex,
  const Cluster* clustersNextLayer,           // input data rof0-delta <rof0< rof0+delta (up to 3 rofs)
  const Cluster* clustersCurrentLayer,        // input data rof0
  const int* indexTable,                      // input data rof0-delta <rof0< rof0+delta (up to 3 rofs)
  const int* roFrameClusters,                 // input data O(1)
  const int* roFrameClustersNext,             // input data O(1)
  const unsigned char* usedClustersLayer,     // input data rof0
  const unsigned char* usedClustersNextLayer, // input data rof1
  const Vertex* vertices,                     // input data
  int* trackletsLookUpTable,                  // output data
  Tracklet* tracklets,                        // output data
  const int nVertices,
  const int currentLayerClustersSize,
  const float phiCut,
  const float minR,
  const float maxR,
  const float meanDeltaR,
  const float positionResolution,
  const float mSAngle,
  const StaticTrackingParameters<NLayers>* trkPars,
  const IndexTableUtils* utils,
  const unsigned int maxTrackletsPerCluster = 10)
{
  // int clusterTrackletsNum = 0;
  for (int currentClusterIndex = blockIdx.x * blockDim.x + threadIdx.x; currentClusterIndex < currentLayerClustersSize; currentClusterIndex += blockDim.x * gridDim.x) {
    unsigned int storedTracklets{0};
    const Cluster& currentCluster{clustersCurrentLayer[currentClusterIndex]};
    const int currentSortedIndex{roFrameClusters[rof0] + currentClusterIndex};
    if (usedClustersLayer[currentSortedIndex]) {
      continue;
    }
    int minRof = (rof0 >= trkPars->DeltaROF) ? rof0 - trkPars->DeltaROF : 0;
    int maxRof = (rof0 == maxRofs - trkPars->DeltaROF) ? rof0 : rof0 + trkPars->DeltaROF;
    const float inverseR0{1.f / currentCluster.radius};
    for (int iPrimaryVertex{0}; iPrimaryVertex < nVertices; iPrimaryVertex++) {
      const auto& primaryVertex{vertices[iPrimaryVertex]};
      if (primaryVertex.getX() == 0.f && primaryVertex.getY() == 0.f && primaryVertex.getZ() == 0.f) {
        continue;
      }
      // if (!currentClusterIndex) {
      //   printf("rof0 %d: Nv: %d -> x: %lf, y: %lf, z: %lf\n", rof0, iPrimaryVertex, primaryVertex.getX(), primaryVertex.getY(), primaryVertex.getZ());
      // }

      const float resolution{o2::gpu::GPUCommonMath::Sqrt(Sq(trkPars->PVres) / primaryVertex.getNContributors() + Sq(positionResolution))};
      const float tanLambda{(currentCluster.zCoordinate - primaryVertex.getZ()) * inverseR0};
      const float zAtRmin{tanLambda * (minR - currentCluster.radius) + currentCluster.zCoordinate};
      const float zAtRmax{tanLambda * (maxR - currentCluster.radius) + currentCluster.zCoordinate};
      const float sqInverseDeltaZ0{1.f / (Sq(currentCluster.zCoordinate - primaryVertex.getZ()) + 2.e-8f)}; /// protecting from overflows adding the detector resolution
      const float sigmaZ{std::sqrt(Sq(resolution) * Sq(tanLambda) * ((Sq(inverseR0) + sqInverseDeltaZ0) * Sq(meanDeltaR) + 1.f) + Sq(meanDeltaR * mSAngle))};

      const int4 selectedBinsRect{getBinsRect(currentCluster, layerIndex, *utils, zAtRmin, zAtRmax, sigmaZ * trkPars->NSigmaCut, phiCut)};
      // if (!currentClusterIndex) {
      //   printf("%d %d %d %d\n", selectedBinsRect.x, selectedBinsRect.y, selectedBinsRect.z, selectedBinsRect.w);
      // }
      if (selectedBinsRect.x == 0 && selectedBinsRect.y == 0 && selectedBinsRect.z == 0 && selectedBinsRect.w == 0) {
        continue;
      }
      int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};
      if (phiBinsNum < 0) {
        phiBinsNum += trkPars->PhiBins;
      }
      constexpr int tableSize{256 * 128 + 1}; // hardcoded for the time being

      for (int rof1{minRof}; rof1 <= maxRof; ++rof1) {
        // printf("%d %d %d \n", minRof, maxRof, roFrameClustersNext[rof1 + 1] - roFrameClustersNext[rof1]);
        if (!(roFrameClustersNext[rof1 + 1] - roFrameClustersNext[rof1])) { // number of clusters on next layer > 0
          continue;
        }
        for (int iPhiCount{0}; iPhiCount < phiBinsNum; iPhiCount++) {
          int iPhiBin = (selectedBinsRect.y + iPhiCount) % trkPars->PhiBins;
          const int firstBinIndex{utils->getBinIndex(selectedBinsRect.x, iPhiBin)};
          const int maxBinIndex{firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1};
          const int firstRowClusterIndex = indexTable[rof1 * tableSize + firstBinIndex];
          const int maxRowClusterIndex = indexTable[rof1 * tableSize + maxBinIndex];
          // if (!currentClusterIndex) {
          //   printf("%d %d %d %d\n", firstBinIndex, maxBinIndex, firstRowClusterIndex, maxRowClusterIndex);
          // }
          for (int iNextCluster{firstRowClusterIndex}; iNextCluster < maxRowClusterIndex; ++iNextCluster) {
            // printf("%d %d\n", iNextCluster, roFrameClustersNext[rof1 + 1] - roFrameClustersNext[rof1]);
            if (iNextCluster >= (roFrameClustersNext[rof1 + 1] - roFrameClustersNext[rof1])) {
              break;
            }

            const Cluster& nextCluster{getPtrFromRuler<Cluster>(rof1, clustersNextLayer, roFrameClustersNext)[iNextCluster]};
            if (usedClustersNextLayer[nextCluster.clusterId]) {
              continue;
            }
            const float deltaPhi{o2::gpu::GPUCommonMath::Abs(currentCluster.phi - nextCluster.phi)};
            const float deltaZ{o2::gpu::GPUCommonMath::Abs(tanLambda * (nextCluster.radius - currentCluster.radius) + currentCluster.zCoordinate - nextCluster.zCoordinate)};

            if (deltaZ / sigmaZ < trkPars->NSigmaCut && (deltaPhi < phiCut || o2::gpu::GPUCommonMath::Abs(deltaPhi - constants::math::TwoPi) < phiCut)) {
              trackletsLookUpTable[currentSortedIndex]++; // Should be race-condition safe
              // printf("%d %d %d %d %d %d %d %d %d %d %f %f %f %f %f \n", maxBinIndex, firstBinIndex, iPhiBin, iPhiCount, phiBinsNum, rof1, rof0, firstRowClusterIndex, maxRowClusterIndex, iNextCluster, nextCluster.xCoordinate, nextCluster.yCoordinate, nextCluster.zCoordinate, deltaPhi, deltaZ);
              const float phi{o2::gpu::GPUCommonMath::ATan2(currentCluster.yCoordinate - nextCluster.yCoordinate,
                                                            currentCluster.xCoordinate - nextCluster.xCoordinate)};
              const float tanL{(currentCluster.zCoordinate - nextCluster.zCoordinate) /
                               (currentCluster.radius - nextCluster.radius)};
              const size_t stride{currentClusterIndex * maxTrackletsPerCluster};
              new (tracklets + stride + storedTracklets) Tracklet{currentSortedIndex, roFrameClustersNext[rof1] + iNextCluster, tanL, phi, rof0, rof1};
              ++storedTracklets;
              // printf("%d %d %lf %lf %hu %hu\n", t.firstClusterIndex, t.secondClusterIndex, t.tanLambda, t.phi, t.rof[0], t.rof[1]);
            }
          }
        }
      }
    }
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
//         if (deltaTanLambda < trkPars->CellMaxDeltaTanLambda && (deltaPhi < trkPars->CellMaxDeltaPhi || o2::gpu::GPUCommonMath::Abs(deltaPhi - constants::math::TwoPi) < trkPars->CellMaxDeltaPhi)) {
//           const float averageTanLambda{0.5f * (currentTracklet.tanLambda + nextTracklet.tanLambda)};
//           const float directionZIntersection{-averageTanLambda * firstCellCluster.radius + firstCellCluster.zCoordinate};
//           const float deltaZ{o2::gpu::GPUCommonMath::Abs(directionZIntersection - primaryVertex.z)};
//           if (deltaZ < trkPars->CellMaxDeltaZ[layerIndex]) {
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
//               if (distanceOfClosestApproach <= trkPars->CellMaxDCA[layerIndex]) {
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
  std::array<gpu::Stream, NLayers - 1> streamArray;

  const Vertex diamondVert({mTrkParams.Diamond[0], mTrkParams.Diamond[1], mTrkParams.Diamond[2]}, {25.e-6f, 0.f, 0.f, 25.e-6f, 0.f, 36.f}, 1, 1.f);
  gsl::span<const Vertex> diamondSpan(&diamondVert, 1);
  for (int rof0{0}; rof0 < mTimeFrameGPU->getNrof(); ++rof0) {
    gsl::span<const Vertex> primaryVertices = mTrkParams.UseDiamond ? diamondSpan : mTimeFrameGPU->getPrimaryVertices(rof0); // replace with GPU one
    std::vector<Vertex> paddedVertices;
    for (int iVertex{0}; iVertex < mTimeFrameGPU->getConfig().maxVerticesCapacity; ++iVertex) {
      if (iVertex < primaryVertices.size()) {
        paddedVertices.emplace_back(primaryVertices[iVertex]);
      } else {
        paddedVertices.emplace_back(Vertex());
      }
    }
    checkGPUError(cudaMemcpy(mTimeFrameGPU->getDeviceVertices(rof0), paddedVertices.data(), mTimeFrameGPU->getConfig().maxVerticesCapacity * sizeof(Vertex), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    for (int iLayer{0}; iLayer < NLayers - 1; ++iLayer) {
      const dim3 threadsPerBlock{gpu::utils::host::getBlockSize(mTimeFrameGPU->getNClustersLayer(rof0, iLayer))};
      const dim3 blocksGrid{gpu::utils::host::getBlocksGrid(threadsPerBlock, mTimeFrameGPU->getNClustersLayer(rof0, iLayer))};
      const float meanDeltaR{mTrkParams.LayerRadii[iLayer + 1] - mTrkParams.LayerRadii[iLayer]};

      if (!mTimeFrameGPU->getClustersOnLayer(rof0, iLayer).size()) {
        LOGP(info, "Skipping ROF0: {}, no clusters found on layer {}", rof0, iLayer);
        continue;
      }
      gpu::computeLayerTrackletsKernel<<<blocksGrid, threadsPerBlock, 0, streamArray[iLayer].get()>>>(
        rof0,
        mTimeFrameGPU->getNrof(),
        iLayer,
        mTimeFrameGPU->getDeviceClustersOnLayer(0, iLayer + 1),      // :check:
        mTimeFrameGPU->getDeviceClustersOnLayer(rof0, iLayer),       // :check:
        mTimeFrameGPU->getDeviceIndexTables(iLayer + 1),             // :check:
        mTimeFrameGPU->getDeviceROframesClustersOnLayer(iLayer),     // :check:
        mTimeFrameGPU->getDeviceROframesClustersOnLayer(iLayer + 1), // :check:
        mTimeFrameGPU->getDeviceUsedClustersOnLayer(0, iLayer),      // :check:
        mTimeFrameGPU->getDeviceUsedClustersOnLayer(0, iLayer + 1),  // :check:
        mTimeFrameGPU->getDeviceVertices(rof0),
        mTimeFrameGPU->getDeviceTrackletsLookupTable(0, iLayer),
        mTimeFrameGPU->getDeviceTracklets(rof0, iLayer),
        mTimeFrameGPU->getConfig().maxVerticesCapacity,
        mTimeFrameGPU->getNClustersLayer(rof0, iLayer),
        mTimeFrameGPU->getPhiCut(iLayer),
        mTimeFrameGPU->getMinR(iLayer + 1),
        mTimeFrameGPU->getMaxR(iLayer + 1),
        meanDeltaR,
        mTimeFrameGPU->getPositionResolution(iLayer),
        mTimeFrameGPU->getMSangle(iLayer),
        mTimeFrameGPU->getDeviceTrackingParameters(),
        mTimeFrameGPU->getDeviceIndexTableUtils(),
        mTimeFrameGPU->getConfig().maxTrackletsPerCluster);
    }
  }
  checkGPUError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  // std::vector<std::vector<int>> tables(NLayers - 1);
  // for (int i{1}; i < /*NLayers - 1*/ 2; ++i) {
  //   tables[i].resize(mTimeFrameGPU->mClusters[i].size());
  //   checkGPUError(cudaMemcpy(tables[i].data(), mTimeFrameGPU->getDeviceTrackletsLookupTable(0, i), mTimeFrameGPU->mClusters[i].size() * sizeof(int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
  //   std::exclusive_scan(tables[i].begin(), tables[i].end(), tables[i].begin(), 0);
  //   std::cout << " === table " << i << " ===" << std::endl;
  //   for (auto j : tables[i]) {
  //     std::cout << j << "\n";
  //   }
  //   std::cout << std::endl;
  // }
  // std::vector<std::vector<Tracklet>> trackletsHost(NLayers - 1, std::vector<Tracklet>(mTimeFrameGPU->getConfig().trackletsCapacity));
  // for (int iLayer{0}; iLayer < NLayers - 1; ++iLayer) {
  //   checkGPUError(cudaMemcpy(trackletsHost[iLayer].data(), mTimeFrameGPU->getDeviceTracklets(0, iLayer), mTimeFrameGPU->getConfig().trackletsCapacity * sizeof(Tracklet), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
  //   std::sort(trackletsHost[iLayer].begin(), trackletsHost[iLayer].end(), [](const Tracklet& a, const Tracklet& b) { return !a.isEmpty() && b.isEmpty(); });
  //   // std::sort(trackletsHost[iLayer].begin(), trackletsHost[iLayer].begin() +)
  //   int count{0};
  //   for (auto& t : trackletsHost[iLayer]) {
  //     t.dump();
  //     if (++count > 10) {
  //       break;
  //     }
  //   }
  //   std::cout << iLayer << " ===" << std::endl;
  // }

  // int* trackletSizesD;
  // int trackletSizeH[6];
  // checkGPUError(cudaMalloc(reinterpret_cast<void**>(&trackletSizesD), (NLayers - 1) * sizeof(int)), __FILE__, __LINE__);
  for (int iLayer{0}; iLayer < NLayers - 1; ++iLayer) {
    size_t bufferSize = mTimeFrameGPU->getConfig().tmpCUBBufferSize;
    auto begin = thrust::device_ptr<o2::its::Tracklet>(mTimeFrameGPU->getDeviceTracklets(0, iLayer));
    auto end = thrust::device_ptr<o2::its::Tracklet>(mTimeFrameGPU->getDeviceTracklets(0, iLayer) + mTimeFrameGPU->mClusters[iLayer].size());
    thrust::sort(begin, end);
    //   discardResult(cub::DeviceReduce::Sum(reinterpret_cast<void*>(mTimeFrameGPU->getDeviceCUBBuffer(iLayer)), // d_temp_storage
    //                                        bufferSize,                                                         // temp_storage_bytes
    //                                        mTimeFrameGPU->getDeviceTrackletsLookupTable(0, iLayer),            // d_in
    //                                        trackletSizesD + iLayer,                                            // d_out
    //                                        mTimeFrameGPU->mClusters[iLayer + 1].size()));                      // num_items
    //   // printf("buffer size: %zu\n", bufferSize);
    //   checkGPUError(cudaMemcpy(trackletSizeH + iLayer, trackletSizesD + iLayer, sizeof(int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    //   // printf("size: %d\n", trackletSizeH[iLayer]);
    //   if (trackletSizeH[iLayer] == 0) {
    //     continue;
    //   }
    //   discardResult(cub::DeviceScan::ExclusiveSum(reinterpret_cast<void*>(mTimeFrameGPU->getDeviceCUBBuffer(iLayer)),                // d_temp_storage
    //                                               bufferSize,                                                                        // temp_storage_bytes
    //                                               mTimeFrameGPU->getDeviceTrackletsLookupTable(0, iLayer),                           // d_in
    //                                               mTimeFrameGPU->getDeviceTrackletsLookupTable(0, iLayer),                           // d_out
    //                                               mTimeFrameGPU->mClusters[iLayer + 1].size() /*, streamArray[iLayer + 1].get()*/)); // num_items
  }
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
