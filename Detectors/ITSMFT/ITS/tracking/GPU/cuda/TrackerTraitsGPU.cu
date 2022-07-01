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
#include <unistd.h>

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/unique.h>

#include "ITStracking/Constants.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/MathUtils.h"

#include "ITStrackingGPU/TrackerTraitsGPU.h"

#include "GPUCommonLogger.h"
#include "GPUCommonAlgorithmThrust.h"
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

GPUhd() float Sq(float q)
{
  return q * q;
}

template <typename T>
struct trackletSortEmptyFunctor : public thrust::binary_function<T, T, bool> {
  GPUhd() bool operator()(const T& lhs, const T& rhs) const
  {
    return lhs.firstClusterIndex > rhs.firstClusterIndex;
  }
};

template <typename T>
struct trackletSortIndexFunctor : public thrust::binary_function<T, T, bool> {
  GPUhd() bool operator()(const T& lhs, const T& rhs) const
  {
    return lhs.firstClusterIndex < rhs.firstClusterIndex || (lhs.firstClusterIndex == rhs.firstClusterIndex && lhs.secondClusterIndex < rhs.secondClusterIndex);
  }
};

// Compute the tracklets for a given layer
template <int NLayers = 7>
GPUg() void computeLayerTrackletsKernel(
  const int rof0,
  const int maxRofs,
  const int layerIndex,
  const Cluster* clustersCurrentLayer,        // input data rof0
  const Cluster* clustersNextLayer,           // input data rof0-delta <rof0< rof0+delta (up to 3 rofs)
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
  const unsigned int maxTrackletsPerCluster = 50)
{
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
      const float resolution{o2::gpu::GPUCommonMath::Sqrt(Sq(trkPars->PVres) / primaryVertex.getNContributors() + Sq(positionResolution))};
      const float tanLambda{(currentCluster.zCoordinate - primaryVertex.getZ()) * inverseR0};
      const float zAtRmin{tanLambda * (minR - currentCluster.radius) + currentCluster.zCoordinate};
      const float zAtRmax{tanLambda * (maxR - currentCluster.radius) + currentCluster.zCoordinate};
      const float sqInverseDeltaZ0{1.f / (Sq(currentCluster.zCoordinate - primaryVertex.getZ()) + 2.e-8f)}; /// protecting from overflows adding the detector resolution
      const float sigmaZ{std::sqrt(Sq(resolution) * Sq(tanLambda) * ((Sq(inverseR0) + sqInverseDeltaZ0) * Sq(meanDeltaR) + 1.f) + Sq(meanDeltaR * mSAngle))};

      const int4 selectedBinsRect{getBinsRect(currentCluster, layerIndex, *utils, zAtRmin, zAtRmax, sigmaZ * trkPars->NSigmaCut, phiCut)};
      if (selectedBinsRect.x == 0 && selectedBinsRect.y == 0 && selectedBinsRect.z == 0 && selectedBinsRect.w == 0) {
        continue;
      }
      int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};
      if (phiBinsNum < 0) {
        phiBinsNum += trkPars->PhiBins;
      }
      constexpr int tableSize{256 * 128 + 1}; // hardcoded for the time being

      for (int rof1{minRof}; rof1 <= maxRof; ++rof1) {
        if (!(roFrameClustersNext[rof1 + 1] - roFrameClustersNext[rof1])) { // number of clusters on next layer > 0
          continue;
        }
        for (int iPhiCount{0}; iPhiCount < phiBinsNum; iPhiCount++) {
          int iPhiBin = (selectedBinsRect.y + iPhiCount) % trkPars->PhiBins;
          const int firstBinIndex{utils->getBinIndex(selectedBinsRect.x, iPhiBin)};
          const int maxBinIndex{firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1};
          const int firstRowClusterIndex = indexTable[rof1 * tableSize + firstBinIndex];
          const int maxRowClusterIndex = indexTable[rof1 * tableSize + maxBinIndex];
          for (int iNextCluster{firstRowClusterIndex}; iNextCluster < maxRowClusterIndex; ++iNextCluster) {
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
              trackletsLookUpTable[currentSortedIndex]++; // Race-condition safe
              const float phi{o2::gpu::GPUCommonMath::ATan2(currentCluster.yCoordinate - nextCluster.yCoordinate, currentCluster.xCoordinate - nextCluster.xCoordinate)};
              const float tanL{(currentCluster.zCoordinate - nextCluster.zCoordinate) / (currentCluster.radius - nextCluster.radius)};
              const size_t stride{currentClusterIndex * maxTrackletsPerCluster};
              new (tracklets + stride + storedTracklets) Tracklet{currentSortedIndex, roFrameClustersNext[rof1] + iNextCluster, tanL, phi, rof0, rof1};
              ++storedTracklets;
            }
          }
        }
      }
    }
    if (storedTracklets > maxTrackletsPerCluster) {
      printf("its-gpu-tracklet finder: found more tracklets per clusters (%d) than maximum set (%d), check the configuration!\n", maxTrackletsPerCluster, storedTracklets);
    }
  }
}

// Decrease LUT entries corresponding to duplicated tracklets. NB: duplicate tracklets are removed separately (see const Tracklets*).
GPUg() void removeDuplicateTrackletsEntriesLUTKernel(
  int* trackletsLookUpTable,
  const Tracklet* tracklets,
  const int* nTracklets,
  const int layerIndex)
{
  int id0{-1}, id1{-1};
  for (int iTracklet{0}; iTracklet < nTracklets[layerIndex]; ++iTracklet) {
    auto& trk = tracklets[iTracklet];
    if (trk.firstClusterIndex == id0 && trk.secondClusterIndex == id1) {
      trackletsLookUpTable[id0]--;
    } else {
      id0 = trk.firstClusterIndex;
      id1 = trk.secondClusterIndex;
    }
  }
}

// Compute cells kernel
template <int NLayers = 7>
GPUg() void computeLayerCellsKernel(
  const Tracklet* trackletsCurrentLayer,
  const Tracklet* trackletsNextLayer,
  const int* trackletsCurrentLayerLUT,
  const int nTracklets)
{
  for (int currentTrackletIndex = blockIdx.x * blockDim.x + threadIdx.x; currentTrackletIndex < nTracklets; currentTrackletIndex += blockDim.x * gridDim.x) {
    const Tracklet& currentTracklet = trackletsCurrentLayer[currentTrackletIndex];
    const int nextLayerClusterIndex{currentTracklet.secondClusterIndex};
    const int nextLayerFirstTrackletIndex{trackletsCurrentLayerLUT[nextLayerClusterIndex]};
    const int nextLayerLastTrackletIndex{trackletsCurrentLayerLUT[nextLayerClusterIndex + 1]};

    if (nextLayerFirstTrackletIndex == nextLayerLastTrackletIndex) {
      continue;
    }
    for (int iNextTracklet{nextLayerFirstTrackletIndex}; iNextTracklet < nextLayerLastTrackletIndex; ++iNextTracklet) {
      if (trackletsNextLayer[iNextTracklet].firstClusterIndex != nextLayerClusterIndex) {
        break;
      }
      const Tracklet& nextTracklet = trackletsNextLayer[iNextTracklet];
      const float deltaTanLambda{o2::gpu::GPUCommonMath::Abs(currentTracklet.tanLambda - nextTracklet.tanLambda)};
      const float tanLambda{(currentTracklet.tanLambda + nextTracklet.tanLambda) * 0.5f};
    }
  }
}
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

      gpu::computeLayerTrackletsKernel<<<blocksGrid, threadsPerBlock, 0, mTimeFrameGPU->getStream(iLayer).get()>>>(
        rof0,
        mTimeFrameGPU->getNrof(),
        iLayer,
        mTimeFrameGPU->getDeviceClustersOnLayer(rof0, iLayer),       // :check:
        mTimeFrameGPU->getDeviceClustersOnLayer(0, iLayer + 1),      // :check:
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
  // std::vector<std::vector<Tracklet>> trackletsHost(NLayers - 1, std::vector<Tracklet>(mTimeFrameGPU->getConfig().trackletsCapacity));
  size_t bufferSize = mTimeFrameGPU->getConfig().tmpCUBBufferSize;
  for (int iLayer{0}; iLayer < NLayers - 1; ++iLayer) {
    // Sort tracklets to put empty ones on the right side of the array.
    auto thrustTrackletsBegin = thrust::device_ptr<o2::its::Tracklet>(mTimeFrameGPU->getDeviceTrackletsAll(iLayer));
    auto thrustTrackletsEnd = thrust::device_ptr<o2::its::Tracklet>(mTimeFrameGPU->getDeviceTrackletsAll(iLayer) + mTimeFrameGPU->getConfig().trackletsCapacity);
    thrust::sort(thrustTrackletsBegin, thrustTrackletsEnd, gpu::trackletSortEmptyFunctor<o2::its::Tracklet>());

    // Get number of found tracklets
    // With thrust:
    // auto thrustTrackletsLUTbegin = thrust::device_ptr<int>(mTimeFrameGPU->getDeviceTrackletsLookupTable(0, iLayer));
    // auto thrustTrackletsLUTend = thrust::device_ptr<int>(mTimeFrameGPU->getDeviceTrackletsLookupTable(0, iLayer) + mTimeFrameGPU->mClusters[iLayer].size());
    // mTimeFrameGPU->getTrackletSizeHost()[iLayer] = thrust::reduce(thrustTrackletsLUTbegin, thrustTrackletsLUTend, 0);

    discardResult(cub::DeviceReduce::Sum(reinterpret_cast<void*>(mTimeFrameGPU->getDeviceCUBBuffer(iLayer)), // d_temp_storage
                                         bufferSize,                                                         // temp_storage_bytes
                                         mTimeFrameGPU->getDeviceTrackletsLookupTable(0, iLayer),            // d_in
                                         mTimeFrameGPU->getDeviceNFoundTracklets() + iLayer,                 // d_out
                                         mTimeFrameGPU->mClusters[iLayer].size(),                            // num_items
                                         mTimeFrameGPU->getStream(iLayer).get()));
  }
  discardResult(cudaDeviceSynchronize());
  checkGPUError(cudaMemcpy(mTimeFrameGPU->getTrackletSizeHost().data(), mTimeFrameGPU->getDeviceNFoundTracklets(), (NLayers - 1) * sizeof(int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
  for (int iLayer{0}; iLayer < NLayers - 1; ++iLayer) {

    // Sort tracklets according to cluster ids
    auto thrustTrackletsBegin = thrust::device_ptr<o2::its::Tracklet>(mTimeFrameGPU->getDeviceTrackletsAll(iLayer));
    auto thrustTrackletsEnd = thrust::device_ptr<o2::its::Tracklet>(mTimeFrameGPU->getDeviceTrackletsAll(iLayer) + mTimeFrameGPU->getTrackletSizeHost()[iLayer]);
    thrust::sort(thrustTrackletsBegin, thrustTrackletsEnd, gpu::trackletSortIndexFunctor<o2::its::Tracklet>());
  }
  discardResult(cudaDeviceSynchronize());
  for (int iLayer{0}; iLayer < NLayers - 1; ++iLayer) {
    // Remove duplicate entries in LUTs, done by single thread so far
    gpu::removeDuplicateTrackletsEntriesLUTKernel<<<1, 1, 0, mTimeFrameGPU->getStream(iLayer).get()>>>(
      mTimeFrameGPU->getDeviceTrackletsLookupTable(0, iLayer),
      mTimeFrameGPU->getDeviceTrackletsAll(iLayer),
      mTimeFrameGPU->getDeviceNFoundTracklets(),
      iLayer);
  }
  // Remove actual tracklet duplicates
  for (int iLayer{0}; iLayer < NLayers - 1; ++iLayer) {
    auto begin = thrust::device_ptr<o2::its::Tracklet>(mTimeFrameGPU->getDeviceTrackletsAll(iLayer));
    auto end = thrust::device_ptr<o2::its::Tracklet>(mTimeFrameGPU->getDeviceTrackletsAll(iLayer) + mTimeFrameGPU->getTrackletSizeHost()[iLayer]);

    auto new_end = thrust::unique(begin, end);
    mTimeFrameGPU->getTrackletSizeHost()[iLayer] = new_end - begin;
  }
  discardResult(cudaDeviceSynchronize());
  for (int iLayer{0}; iLayer < NLayers - 1; ++iLayer) {

    // Compute LUT
    discardResult(cub::DeviceScan::ExclusiveSum(reinterpret_cast<void*>(mTimeFrameGPU->getDeviceCUBBuffer(iLayer)), // d_temp_storage
                                                bufferSize,                                                         // temp_storage_bytes
                                                mTimeFrameGPU->getDeviceTrackletsLookupTable(0, iLayer),            // d_in
                                                mTimeFrameGPU->getDeviceTrackletsLookupTable(0, iLayer),            // d_out
                                                mTimeFrameGPU->mClusters[iLayer].size(),                            // num_items
                                                mTimeFrameGPU->getStream(iLayer).get()));
  }
  discardResult(cudaDeviceSynchronize());

  // Create tracklets labels, at the moment on the host
  if (mTimeFrameGPU->hasMCinformation()) {
    for (int iLayer{0}; iLayer < mTrkParams.TrackletsPerRoad(); ++iLayer) {
      std::vector<o2::its::Tracklet> tracklets(mTimeFrameGPU->getTrackletSizeHost()[iLayer]);
      checkGPUError(cudaMemcpy(tracklets.data(), mTimeFrameGPU->getDeviceTrackletsAll(iLayer), mTimeFrameGPU->getTrackletSizeHost()[iLayer] * sizeof(o2::its::Tracklet), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
      for (auto& trk : tracklets) {
        MCCompLabel label;
        int currentId{mTimeFrameGPU->mClusters[iLayer][trk.firstClusterIndex].clusterId};
        int nextId{mTimeFrameGPU->mClusters[iLayer + 1][trk.secondClusterIndex].clusterId};
        for (auto& lab1 : mTimeFrameGPU->getClusterLabels(iLayer, currentId)) {
          for (auto& lab2 : mTimeFrameGPU->getClusterLabels(iLayer + 1, nextId)) {
            if (lab1 == lab2 && lab1.isValid()) {
              label = lab1;
              break;
            }
          }
          if (label.isValid()) {
            break;
          }
        }
        mTimeFrameGPU->getTrackletsLabel(iLayer).emplace_back(label);
      }
    }
  }
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::computeLayerCells()
{
  for (int iLayer{0}; iLayer < NLayers - 2 - 1; ++iLayer) {
    if (!mTimeFrameGPU->getTrackletSizeHost()[iLayer + 1] ||
        !mTimeFrameGPU->getTrackletSizeHost()[iLayer]) {
      continue;
    }
    float resolution{std::sqrt(gpu::Sq(mTrkParams.LayerMisalignment[iLayer]) + gpu::Sq(mTrkParams.LayerMisalignment[iLayer + 1]) + gpu::Sq(mTrkParams.LayerMisalignment[iLayer + 2])) / mTrkParams.LayerResolution[iLayer]};
    resolution = resolution > 1.e-12 ? resolution : 1.f;

    gpu::computeLayerCellsKernel<<<1, 1, 0, mTimeFrameGPU->getStream(iLayer).get()>>>(
      mTimeFrameGPU->getDeviceTrackletsAll(iLayer),
      mTimeFrameGPU->getDeviceTrackletsAll(iLayer + 1),
      mTimeFrameGPU->getDeviceTrackletsLookupTable(0, iLayer),
      mTimeFrameGPU->getTrackletSizeHost()[iLayer]);
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
    //       gpu::layerCellsKernel<<<blocksGrid, threadsPerBlock, 0, mTimeFrameGPU->getStream(iLayer).get()>>>(primaryVertexContext->getDeviceContext(),
    //                                                                                            iLayer, primaryVertexContext->getDeviceCells()[iLayer].getWeakCopy());
    //     } else {
    //       gpu::layerCellsKernel<<<blocksGrid, threadsPerBlock, 0, mTimeFrameGPU->getStream(iLayer).get()>>>(primaryVertexContext->getDeviceContext(),
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
  }
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
