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
//
/// \author matteo.concas@cern.ch

#include <iostream>
#include <sstream>
#include <fstream>
#include <array>
#include <cassert>

#ifndef GPUCA_GPUCODE_GENRTC
#include <cub/cub.cuh>
#endif

#include "ITStracking/MathUtils.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/Tracklet.h"

#include "ITStrackingGPU/Utils.h"
#include "ITStrackingGPU/ClusterLinesGPU.h"
#include "ITStrackingGPU/VertexerTraitsGPU.h"

namespace o2
{
namespace its
{

using constants::its::VertexerHistogramVolume;
using constants::math::TwoPi;
using gpu::utils::host::checkGPUError;
using math_utils::getNormalizedPhi;

using namespace constants::its2;
GPUd() const int4 getBinsRect(const Cluster& currentCluster, const int layerIndex,
                              const float z1, float maxdeltaz, float maxdeltaphi)
{
  const float zRangeMin = z1 - maxdeltaz;
  const float phiRangeMin = currentCluster.phi - maxdeltaphi;
  const float zRangeMax = z1 + maxdeltaz;
  const float phiRangeMax = currentCluster.phi + maxdeltaphi;

  if (zRangeMax < -LayersZCoordinate()[layerIndex + 1] ||
      zRangeMin > LayersZCoordinate()[layerIndex + 1] || zRangeMin > zRangeMax) {

    return getEmptyBinsRect();
  }

  return int4{o2::gpu::GPUCommonMath::Max(0, getZBinIndex(layerIndex + 1, zRangeMin)),
              getPhiBinIndex(phiRangeMin),
              o2::gpu::GPUCommonMath::Min(ZBins - 1, getZBinIndex(layerIndex + 1, zRangeMax)),
              getPhiBinIndex(phiRangeMax)};
}

GPUh() void gpuThrowOnError()
{
  cudaError_t error = cudaGetLastError();

  if (error != cudaSuccess) {
    std::ostringstream errorString{};
    errorString << GPU_ARCH << " API returned error  [" << cudaGetErrorString(error) << "] (code " << error << ")" << std::endl;
    throw std::runtime_error{errorString.str()};
  }
}

VertexerTraitsGPU::VertexerTraitsGPU()
{
  gpu::utils::host::gpuMalloc((void**)&mDeviceIndexTableUtils, sizeof(IndexTableUtils));
  setIsGPU(true);
}

VertexerTraitsGPU::~VertexerTraitsGPU()
{
  gpu::utils::host::gpuFree(mDeviceIndexTableUtils);
}

void VertexerTraitsGPU::initialise(const MemoryParameters& memParams, const TrackingParameters& trackingParams)
{
  if (!mIndexTableUtils.getNzBins()) {
    updateVertexingParameters(mVrtParams);
  }
  gpu::utils::host::gpuMemcpyHostToDevice(mDeviceIndexTableUtils, &mIndexTableUtils, sizeof(mIndexTableUtils));
  mTimeFrameGPU->initialise(0, memParams, trackingParams, 3);
  setIsGPU(true);
}

namespace gpu
{

template <typename... Args>
GPUd() void printOnThread(const unsigned int tId, const char* str, Args... args)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == tId) {
    printf(str, args...);
  }
}

GPUd() void printVectorOnThread(const char* name, Vector<int>& vector, size_t size, const unsigned int tId = 0)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == tId) {
    printf("vector %s :", name);
    for (int i{0}; i < size; ++i) {
      printf("%d ", vector[i]);
    }
    printf("\n");
  }
}

// GPUg() void dumpMaximaKernel(DeviceStoreVertexerGPU& store, const int threadId)
// {
//   if (blockIdx.x * blockDim.x + threadIdx.x == threadId) {
//     printf("XmaxBin: %d at index: %d | YmaxBin: %d at index: %d | ZmaxBin: %d at index: %d\n",
//            store.getTmpVertexPositionBins()[0].value, store.getTmpVertexPositionBins()[0].key,
//            store.getTmpVertexPositionBins()[1].value, store.getTmpVertexPositionBins()[1].key,
//            store.getTmpVertexPositionBins()[2].value, store.getTmpVertexPositionBins()[2].key);
//   }
// }

template <TrackletMode Mode>
GPUg() void trackleterKernel(
  const Cluster* clustersNextLayer,    // 0 2
  const Cluster* clustersCurrentLayer, // 1 1
  const int sizeNextLClusters,
  const int sizeCurrentLClusters,
  const int* indexTableNext,
  const float phiCut,
  Tracklet* Tracklets,
  int* foundTracklets,
  const IndexTableUtils* utils,
  const size_t maxTrackletsPerCluster = 10)
{
  const int phiBins{utils->getNphiBins()};
  const int zBins{utils->getNzBins()};
  // loop on layer1 clusters
  for (int iCurrentLayerClusterIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentLayerClusterIndex < sizeCurrentLClusters; iCurrentLayerClusterIndex += blockDim.x * gridDim.x) {
    if (iCurrentLayerClusterIndex < sizeCurrentLClusters) {
      unsigned int storedTracklets{0};
      const size_t stride{iCurrentLayerClusterIndex * maxTrackletsPerCluster};
      const Cluster& currentCluster = clustersCurrentLayer[iCurrentLayerClusterIndex];
      const int4 selectedBinsRect{VertexerTraits::getBinsRect(currentCluster, (int)Mode, 0.f, 50.f, phiCut / 2, *utils)};
      if (selectedBinsRect.x != 0 || selectedBinsRect.y != 0 || selectedBinsRect.z != 0 || selectedBinsRect.w != 0) {
        int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};
        if (phiBinsNum < 0) {
          phiBinsNum += phiBins;
        }
        // loop on phi bins next layer
        for (unsigned int iPhiBin{(unsigned int)selectedBinsRect.y}, iPhiCount{0}; iPhiCount < (unsigned int)phiBinsNum; iPhiBin = ++iPhiBin == phiBins ? 0 : iPhiBin, iPhiCount++) {
          const int firstBinIndex{utils->getBinIndex(selectedBinsRect.x, iPhiBin)};
          const int firstRowClusterIndex{indexTableNext[firstBinIndex]};
          const int maxRowClusterIndex{indexTableNext[firstBinIndex + zBins]};
          // loop on clusters next layer
          for (int iNextLayerClusterIndex{firstRowClusterIndex}; iNextLayerClusterIndex < maxRowClusterIndex && iNextLayerClusterIndex < sizeNextLClusters; ++iNextLayerClusterIndex) {
            const Cluster& nextCluster = clustersNextLayer[iNextLayerClusterIndex];
            if (o2::gpu::GPUCommonMath::Abs(currentCluster.phi - nextCluster.phi) < phiCut) {
              if (storedTracklets < maxTrackletsPerCluster) {
                if constexpr (Mode == TrackletMode::Layer0Layer1) {
                  new (Tracklets + stride + storedTracklets) Tracklet{iNextLayerClusterIndex, iCurrentLayerClusterIndex, nextCluster, currentCluster};
                } else {
                  new (Tracklets + stride + storedTracklets) Tracklet{iCurrentLayerClusterIndex, iNextLayerClusterIndex, currentCluster, nextCluster};
                }
                ++storedTracklets;
              }
            }
          }
        }
      }
      foundTracklets[iCurrentLayerClusterIndex] = storedTracklets;
    }
  }
}

template <bool initRun>
GPUg() void trackletSelectionKernel(
  const Cluster* clusters0,
  const Cluster* clusters1,
  const size_t nClustersMiddleLayer,
  Tracklet* tracklets01,
  Tracklet* tracklets12,
  const int* nFoundTracklet01,
  const int* nFoundTracklet12,
  unsigned char* usedTracklets,
  Line* lines,
  int* nFoundLines,
  int* nExclusiveFoundLines,
  const int maxTrackletsPerCluster = 10,
  const float tanLambdaCut = 0.025f,
  const float phiCut = 0.002f)
{
  for (int iCurrentLayerClusterIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentLayerClusterIndex < nClustersMiddleLayer; iCurrentLayerClusterIndex += blockDim.x * gridDim.x) {
    const int stride{static_cast<int>(iCurrentLayerClusterIndex * maxTrackletsPerCluster)};
    int validTracklets{0};
    for (int iTracklet12{0}; iTracklet12 < nFoundTracklet12[iCurrentLayerClusterIndex]; ++iTracklet12) {
      for (int iTracklet01{0}; iTracklet01 < nFoundTracklet01[iCurrentLayerClusterIndex] && validTracklets < maxTrackletsPerCluster; ++iTracklet01) {
        const float deltaTanLambda{o2::gpu::GPUCommonMath::Abs(tracklets01[stride + iTracklet01].tanLambda - tracklets12[stride + iTracklet12].tanLambda)};
        const float deltaPhi{o2::gpu::GPUCommonMath::Abs(tracklets01[stride + iTracklet01].phi - tracklets12[stride + iTracklet12].phi)};
        if (!usedTracklets[iTracklet01] && deltaTanLambda < tanLambdaCut && deltaPhi < phiCut && validTracklets != maxTrackletsPerCluster) {
          if constexpr (!initRun) {
            usedTracklets[iTracklet01] = true;
            new (lines + nExclusiveFoundLines[iCurrentLayerClusterIndex] + validTracklets) Line{tracklets01[stride + iTracklet01], clusters0, clusters1};
          }
          ++validTracklets;
        }
      }
    }
    if constexpr (initRun) {
      nFoundLines[iCurrentLayerClusterIndex] = validTracklets;
      if (validTracklets >= maxTrackletsPerCluster) {
        printf("gpu tracklet selection: some lines will be left behind for cluster %d. valid: %d max: %d\n", iCurrentLayerClusterIndex, validTracklets, maxTrackletsPerCluster);
      }
    }
  }
}

GPUg() void computeCentroidsKernel(Line* lines,
                                   int* nFoundLines,
                                   int* nExclusiveFoundLines,
                                   const size_t nClustersMiddleLayer,
                                   float* centroids,
                                   const float lowHistX,
                                   const float highHistX,
                                   const float lowHistY,
                                   const float highHistY,
                                   const float pairCut)
{
  const int nLines = nExclusiveFoundLines[nClustersMiddleLayer - 1] + nFoundLines[nClustersMiddleLayer - 1];
  const int maxIterations{nLines * (nLines - 1) / 2};
  for (size_t currentThreadIndex = blockIdx.x * blockDim.x + threadIdx.x; currentThreadIndex < maxIterations; currentThreadIndex += blockDim.x * gridDim.x) {
    int iFirstLine = currentThreadIndex / nLines;
    int iSecondLine = currentThreadIndex % nLines;
    // All unique pairs
    if (iSecondLine <= iFirstLine) {
      iFirstLine = nLines - iFirstLine - 2;
      iSecondLine = nLines - iSecondLine - 1;
    }
    if (Line::getDCA(lines[iFirstLine], lines[iSecondLine]) < pairCut) {
      ClusterLinesGPU cluster{lines[iFirstLine], lines[iSecondLine]};
      if (cluster.getVertex()[0] * cluster.getVertex()[0] + cluster.getVertex()[1] * cluster.getVertex()[1] < 1.98f * 1.98f) {
        // printOnThread(0, "xCentr: %f, yCentr: %f \n", cluster.getVertex()[0], cluster.getVertex()[1]);
        centroids[2 * currentThreadIndex] = cluster.getVertex()[0];
        centroids[2 * currentThreadIndex + 1] = cluster.getVertex()[1];
      } else {
        // writing some data anyway outside the histogram, they will not be put in the histogram, by construction.
        centroids[2 * currentThreadIndex] = 2 * lowHistX;
        centroids[2 * currentThreadIndex + 1] = 2 * lowHistY;
      }
    } else {
      // writing some data anyway outside the histogram, they will not be put in the histogram, by construction.
      centroids[2 * currentThreadIndex] = 2 * highHistX;
      centroids[2 * currentThreadIndex + 1] = 2 * highHistY;
    }
  }
}

GPUg() void computeZCentroidsKernel(const int nLines,
                                    const cub::KeyValuePair<int, int>* tmpVtX,
                                    float* beamPosition,
                                    Line* lines,
                                    float* centroids,
                                    const int* histX, // X
                                    const float lowHistX,
                                    const float binSizeHistX,
                                    const int nBinsHistX,
                                    const int* histY, // Y
                                    const float lowHistY,
                                    const float binSizeHistY,
                                    const int nBinsHistY,
                                    const float lowHistZ, // Z
                                    const float pairCut,
                                    const int binOpeningX,
                                    const int binOpeningY)
{
  for (size_t currentThreadIndex = blockIdx.x * blockDim.x + threadIdx.x; currentThreadIndex < nLines; currentThreadIndex += blockDim.x * gridDim.x) {
    if (tmpVtX[0].value || tmpVtX[1].value) {
      float tmpX{lowHistX + tmpVtX[0].key * binSizeHistX + binSizeHistX / 2};
      int sumWX{tmpVtX[0].value};
      float wX{tmpX * tmpVtX[0].value};
      for (int iBin{o2::gpu::GPUCommonMath::Max(0, tmpVtX[0].key - binOpeningX)}; iBin < o2::gpu::GPUCommonMath::Min(tmpVtX[0].key + binOpeningX + 1, nBinsHistX - 1); ++iBin) {
        if (iBin != tmpVtX[0].key) {
          wX += (lowHistX + iBin * binSizeHistX + binSizeHistX / 2) * histX[iBin];
          sumWX += histX[iBin];
        }
      }
      float tmpY{lowHistY + tmpVtX[1].key * binSizeHistY + binSizeHistY / 2};
      int sumWY{tmpVtX[1].value};
      float wY{tmpY * tmpVtX[1].value};
      for (int iBin{o2::gpu::GPUCommonMath::Max(0, tmpVtX[1].key - binOpeningY)}; iBin < o2::gpu::GPUCommonMath::Min(tmpVtX[1].key + binOpeningY + 1, nBinsHistY - 1); ++iBin) {
        if (iBin != tmpVtX[1].key) {
          wY += (lowHistY + iBin * binSizeHistY + binSizeHistY / 2) * histY[iBin];
          sumWY += histY[iBin];
        }
      }
      beamPosition[0] = wX / sumWX;
      beamPosition[1] = wY / sumWY;
      float mockBeamPoint1[3] = {beamPosition[0], beamPosition[1], -1}; // get two points laying at different z, to create line object
      float mockBeamPoint2[3] = {beamPosition[0], beamPosition[1], 1};
      Line pseudoBeam = {mockBeamPoint1, mockBeamPoint2};
      if (Line::getDCA(lines[currentThreadIndex], pseudoBeam) < pairCut) {
        ClusterLinesGPU cluster{lines[currentThreadIndex], pseudoBeam};
        centroids[currentThreadIndex] = cluster.getVertex()[2];
      } else {
        centroids[currentThreadIndex] = 2 * lowHistZ;
      }
    }
  }
}

// GPUg() void computeVertexKernel(DeviceStoreVertexerGPU& store, const int vertIndex, const int minContributors, const int binOpeningZ)
// {
//   for (size_t currentThreadIndex = blockIdx.x * blockDim.x + threadIdx.x; currentThreadIndex < binOpeningZ; currentThreadIndex += blockDim.x * gridDim.x) {
//     if (currentThreadIndex == 0) {
//       if (store.getTmpVertexPositionBins()[2].value > 1 && (store.getTmpVertexPositionBins()[0].value || store.getTmpVertexPositionBins()[1].value)) {
//         float z{store.getConfig().histConf.lowHistBoundariesXYZ[2] + store.getTmpVertexPositionBins()[2].key * store.getConfig().histConf.binSizeHistZ + store.getConfig().histConf.binSizeHistZ / 2};
//         float ex{0.f};
//         float ey{0.f};
//         float ez{0.f};
//         int sumWZ{store.getTmpVertexPositionBins()[2].value};
//         float wZ{z * store.getTmpVertexPositionBins()[2].value};
//         for (int iBin{o2::gpu::GPUCommonMath::Max(0, store.getTmpVertexPositionBins()[2].key - binOpeningZ)}; iBin < o2::gpu::GPUCommonMath::Min(store.getTmpVertexPositionBins()[2].key + binOpeningZ + 1, store.getConfig().histConf.nBinsXYZ[2] - 1); ++iBin) {
//           if (iBin != store.getTmpVertexPositionBins()[2].key) {
//             wZ += (store.getConfig().histConf.lowHistBoundariesXYZ[2] + iBin * store.getConfig().histConf.binSizeHistZ + store.getConfig().histConf.binSizeHistZ / 2) * store.getHistogramXYZ()[2].get()[iBin];
//             sumWZ += store.getHistogramXYZ()[2].get()[iBin];
//           }
//           store.getHistogramXYZ()[2].get()[iBin] = 0;
//         }
//         if (sumWZ > minContributors || vertIndex == 0) {
//           store.getVertices().emplace(vertIndex, store.getBeamPosition()[0], store.getBeamPosition()[1], wZ / sumWZ, ex, ey, ez, sumWZ);
//         } else {
//           store.getVertices().emplace(vertIndex);
//         }
//       } else {
//         store.getVertices().emplace(vertIndex);
//       }
//     }
//   }
// }
} // namespace gpu

void VertexerTraitsGPU::computeTracklets()
{
  if (!mTimeFrameGPU->getClusters().size()) {
    return;
  }
  for (int rofId{0}; rofId < mTimeFrameGPU->getNrof(); ++rofId) {
    const dim3 threadsPerBlock{gpu::utils::host::getBlockSize(mTimeFrameGPU->getNClustersLayer(rofId, 1))};
    const dim3 blocksGrid{gpu::utils::host::getBlocksGrid(threadsPerBlock, mTimeFrameGPU->getNClustersLayer(rofId, 1))};
    gpu::trackleterKernel<TrackletMode::Layer0Layer1><<<1, 1>>>(
      mTimeFrameGPU->getDeviceClustersOnLayer(rofId, 0),
      mTimeFrameGPU->getDeviceClustersOnLayer(rofId, 1),
      mTimeFrameGPU->getNClustersLayer(rofId, 0),
      mTimeFrameGPU->getNClustersLayer(rofId, 1),
      mTimeFrameGPU->getDeviceIndexTableL0(rofId),
      mVrtParams.phiCut,
      mTimeFrameGPU->getDeviceTracklets()[0].get(),
      mTimeFrameGPU->getDeviceNTrackletsCluster(rofId, 0),
      mDeviceIndexTableUtils,
      mTimeFrameGPU->getConfig().maxTrackletsPerCluster);

    gpu::trackleterKernel<TrackletMode::Layer1Layer2><<<1, 1>>>(
      mTimeFrameGPU->getDeviceClustersOnLayer(rofId, 2),
      mTimeFrameGPU->getDeviceClustersOnLayer(rofId, 1),
      mTimeFrameGPU->getNClustersLayer(rofId, 2),
      mTimeFrameGPU->getNClustersLayer(rofId, 1),
      mTimeFrameGPU->getDeviceIndexTableL2(rofId),
      mVrtParams.phiCut,
      mTimeFrameGPU->getDeviceTracklets()[1].get(),
      mTimeFrameGPU->getDeviceNTrackletsCluster(rofId, 1),
      mDeviceIndexTableUtils,
      mTimeFrameGPU->getConfig().maxTrackletsPerCluster);
  }
#ifdef VTX_DEBUG
  std::ofstream out01("NTC01.txt"), out12("NTC12.txt");
  std::vector<std::vector<int>> NtrackletsClusters01(mTimeFrameGPU->getNrof());
  std::vector<std::vector<int>> NtrackletsClusters12(mTimeFrameGPU->getNrof());
  for (int iRof{0}; iRof < mTimeFrameGPU->getNrof(); ++iRof) {
    NtrackletsClusters01[iRof].resize(mTimeFrameGPU->getClustersOnLayer(iRof, 1).size());
    NtrackletsClusters12[iRof].resize(mTimeFrameGPU->getClustersOnLayer(iRof, 1).size());
    checkGPUError(cudaMemcpy(NtrackletsClusters01[iRof].data(), mTimeFrameGPU->getDeviceNTrackletsCluster(iRof, 0), sizeof(int) * mTimeFrameGPU->getClustersOnLayer(iRof, 1).size(), cudaMemcpyDeviceToHost));
    checkGPUError(cudaMemcpy(NtrackletsClusters12[iRof].data(), mTimeFrameGPU->getDeviceNTrackletsCluster(iRof, 1), sizeof(int) * mTimeFrameGPU->getClustersOnLayer(iRof, 1).size(), cudaMemcpyDeviceToHost));

    std::copy(NtrackletsClusters01[iRof].begin(), NtrackletsClusters01[iRof].end(), std::ostream_iterator<double>(out01, "\t"));
    std::copy(NtrackletsClusters12[iRof].begin(), NtrackletsClusters12[iRof].end(), std::ostream_iterator<double>(out12, "\t"));
    out01 << std::endl;
    out12 << std::endl;
  }
  out01.close();
  out12.close();
#endif
}

void VertexerTraitsGPU::computeTrackletMatching()
{
  if (!mTimeFrameGPU->getClusters().size()) {
    return;
  }
  for (int rofId{0}; rofId < mTimeFrameGPU->getNrof(); ++rofId) {
    const dim3 threadsPerBlock{gpu::utils::host::getBlockSize(mTimeFrameGPU->getNClustersLayer(rofId, 1))};
    const dim3 blocksGrid{gpu::utils::host::getBlocksGrid(threadsPerBlock, mTimeFrameGPU->getNClustersLayer(rofId, 1))};

    size_t bufferSize = mTimeFrameGPU->getConfig().tmpCUBBufferSize;

    gpu::trackletSelectionKernel<true><<<blocksGrid, threadsPerBlock>>>(
      mTimeFrameGPU->getDeviceClustersOnLayer(rofId, 0),
      mTimeFrameGPU->getDeviceClustersOnLayer(rofId, 1),
      mTimeFrameGPU->getNClustersLayer(rofId, 1),
      mTimeFrameGPU->getDeviceTracklets()[0].get(),
      mTimeFrameGPU->getDeviceTracklets()[1].get(),
      mTimeFrameGPU->getDeviceNTrackletsCluster(rofId, 0),
      mTimeFrameGPU->getDeviceNTrackletsCluster(rofId, 1),
      mTimeFrameGPU->getDeviceUsedTracklets(rofId),
      mTimeFrameGPU->getDeviceLines(rofId),
      mTimeFrameGPU->getDeviceNFoundLines(rofId),
      mTimeFrameGPU->getDeviceExclusiveNFoundLines(rofId),
      mTimeFrameGPU->getConfig().maxTrackletsPerCluster);

    discardResult(cub::DeviceScan::ExclusiveSum(reinterpret_cast<void*>(mTimeFrameGPU->getDeviceCUBBuffer(rofId)),
                                                bufferSize,
                                                mTimeFrameGPU->getDeviceNFoundLines(rofId),
                                                mTimeFrameGPU->getDeviceExclusiveNFoundLines(rofId),
                                                mTimeFrameGPU->getNClustersLayer(rofId, 1)));

    gpu::trackletSelectionKernel<false><<<blocksGrid, threadsPerBlock>>>(
      mTimeFrameGPU->getDeviceClustersOnLayer(rofId, 0),
      mTimeFrameGPU->getDeviceClustersOnLayer(rofId, 1),
      mTimeFrameGPU->getNClustersLayer(rofId, 1),
      mTimeFrameGPU->getDeviceTracklets()[0].get(),
      mTimeFrameGPU->getDeviceTracklets()[1].get(),
      mTimeFrameGPU->getDeviceNTrackletsCluster(rofId, 0),
      mTimeFrameGPU->getDeviceNTrackletsCluster(rofId, 1),
      mTimeFrameGPU->getDeviceUsedTracklets(rofId),
      mTimeFrameGPU->getDeviceLines(rofId),
      mTimeFrameGPU->getDeviceNFoundLines(rofId),
      mTimeFrameGPU->getDeviceExclusiveNFoundLines(rofId),
      mTimeFrameGPU->getConfig().maxTrackletsPerCluster);

    gpuThrowOnError();
  }
#ifdef VTX_DEBUG
  std::vector<std::vector<int>> NFoundLines(mTimeFrameGPU->getNrof()), ExcNFoundLines(mTimeFrameGPU->getNrof());
  std::ofstream nlines_out("N_lines_gpu.txt");
  for (size_t rofId{0}; rofId < mTimeFrameGPU->getNrof(); ++rofId) {
    NFoundLines[rofId].resize(mTimeFrameGPU->getNClustersLayer(rofId, 1));
    ExcNFoundLines[rofId].resize(mTimeFrameGPU->getNClustersLayer(rofId, 1));
    checkGPUError(cudaMemcpy(NFoundLines[rofId].data(), mTimeFrameGPU->getDeviceNFoundLines(rofId), sizeof(int) * mTimeFrameGPU->getNClustersLayer(rofId, 1), cudaMemcpyDeviceToHost));
    checkGPUError(cudaMemcpy(ExcNFoundLines[rofId].data(), mTimeFrameGPU->getDeviceExclusiveNFoundLines(rofId), sizeof(int) * mTimeFrameGPU->getNClustersLayer(rofId, 1), cudaMemcpyDeviceToHost));
    std::copy(NFoundLines[rofId].begin(), NFoundLines[rofId].end(), std::ostream_iterator<double>(nlines_out, "\t"));
    nlines_out << std::endl;
    std::copy(ExcNFoundLines[rofId].begin(), ExcNFoundLines[rofId].end(), std::ostream_iterator<double>(nlines_out, "\t"));
    nlines_out << std::endl
               << " ---\n";
  }
  nlines_out.close();
#endif
}

void VertexerTraitsGPU::computeVertices()
{
  if (!mTimeFrameGPU->getClusters().size()) {
    return;
  }
  for (int rofId{0}; rofId < mTimeFrameGPU->getNrof(); ++rofId) {
    const dim3 threadsPerBlock{gpu::utils::host::getBlockSize(mTimeFrameGPU->getNClustersLayer(rofId, 1))};
    const dim3 blocksGrid{gpu::utils::host::getBlocksGrid(threadsPerBlock, mTimeFrameGPU->getNClustersLayer(rofId, 1))};

    size_t bufferSize = mTimeFrameGPU->getConfig().tmpCUBBufferSize;
    int excLas, nLas, nLines;
    checkGPUError(cudaMemcpy(&excLas, mTimeFrameGPU->getDeviceExclusiveNFoundLines(rofId) + mTimeFrameGPU->getNClustersLayer(rofId, 1) - 1, sizeof(int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    checkGPUError(cudaMemcpy(&nLas, mTimeFrameGPU->getDeviceNFoundLines(rofId) + mTimeFrameGPU->getNClustersLayer(rofId, 1) - 1, sizeof(int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    nLines = excLas + nLas;
    int nCentroids{nLines * (nLines - 1) / 2};
    int* histogramXY[2] = {mTimeFrameGPU->getDeviceXHistograms(rofId), mTimeFrameGPU->getDeviceYHistograms(rofId)};
    float tmpArrayLow[2] = {mTimeFrameGPU->getConfig().histConf.lowHistBoundariesXYZ[0], mTimeFrameGPU->getConfig().histConf.lowHistBoundariesXYZ[1]};
    float tmpArrayHigh[2] = {mTimeFrameGPU->getConfig().histConf.highHistBoundariesXYZ[0], mTimeFrameGPU->getConfig().histConf.highHistBoundariesXYZ[1]};

    gpu::computeCentroidsKernel<<<blocksGrid, threadsPerBlock>>>(
      mTimeFrameGPU->getDeviceLines(rofId),
      mTimeFrameGPU->getDeviceNFoundLines(rofId),
      mTimeFrameGPU->getDeviceExclusiveNFoundLines(rofId),
      mTimeFrameGPU->getNClustersLayer(rofId, 1),
      mTimeFrameGPU->getDeviceXYCentroids(rofId),
      mTimeFrameGPU->getConfig().histConf.lowHistBoundariesXYZ[0],
      mTimeFrameGPU->getConfig().histConf.highHistBoundariesXYZ[0],
      mTimeFrameGPU->getConfig().histConf.lowHistBoundariesXYZ[1],
      mTimeFrameGPU->getConfig().histConf.highHistBoundariesXYZ[1],
      mVrtParams.histPairCut);

    discardResult(cub::DeviceHistogram::MultiHistogramEven<2, 2>(reinterpret_cast<void*>(mTimeFrameGPU->getDeviceCUBBuffer(rofId)), // d_temp_storage
                                                                 bufferSize,                                                        // temp_storage_bytes
                                                                 mTimeFrameGPU->getDeviceXYCentroids(rofId),                        // d_samples
                                                                 histogramXY,                                                       // d_histogram
                                                                 mTimeFrameGPU->getConfig().histConf.nBinsXYZ,                      // num_levels
                                                                 tmpArrayLow,                                                       // lower_level
                                                                 tmpArrayHigh,                                                      // fupper_level
                                                                 nCentroids));                                                      // num_row_pixels
    discardResult(cub::DeviceReduce::ArgMax(reinterpret_cast<void*>(mTimeFrameGPU->getDeviceCUBBuffer(rofId)),
                                            bufferSize,
                                            histogramXY[0],
                                            mTimeFrameGPU->getTmpVertexPositionBins(rofId),
                                            mTimeFrameGPU->getConfig().histConf.nBinsXYZ[0]));

    discardResult(cub::DeviceReduce::ArgMax(reinterpret_cast<void*>(mTimeFrameGPU->getDeviceCUBBuffer(rofId)),
                                            bufferSize,
                                            histogramXY[1],
                                            mTimeFrameGPU->getTmpVertexPositionBins(rofId) + 1,
                                            mTimeFrameGPU->getConfig().histConf.nBinsXYZ[1]));

    gpu::computeZCentroidsKernel<<<blocksGrid, threadsPerBlock>>>(nLines,
                                                                  mTimeFrameGPU->getTmpVertexPositionBins(rofId),
                                                                  mTimeFrameGPU->getDeviceBeamPosition(rofId),
                                                                  mTimeFrameGPU->getDeviceLines(rofId),
                                                                  mTimeFrameGPU->getDeviceZCentroids(rofId),
                                                                  mTimeFrameGPU->getDeviceXHistograms(rofId),
                                                                  mTimeFrameGPU->getConfig().histConf.lowHistBoundariesXYZ[0],
                                                                  mTimeFrameGPU->getConfig().histConf.binSizeHistX,
                                                                  mTimeFrameGPU->getConfig().histConf.nBinsXYZ[0],
                                                                  mTimeFrameGPU->getDeviceYHistograms(rofId),
                                                                  mTimeFrameGPU->getConfig().histConf.lowHistBoundariesXYZ[1],
                                                                  mTimeFrameGPU->getConfig().histConf.binSizeHistY,
                                                                  mTimeFrameGPU->getConfig().histConf.nBinsXYZ[1],
                                                                  mTimeFrameGPU->getConfig().histConf.lowHistBoundariesXYZ[2],
                                                                  mVrtParams.histPairCut,
                                                                  mTimeFrameGPU->getConfig().histConf.binSpanXYZ[0],
                                                                  mTimeFrameGPU->getConfig().histConf.binSpanXYZ[1]);

    discardResult(cub::DeviceHistogram::HistogramEven(reinterpret_cast<void*>(mTimeFrameGPU->getDeviceCUBBuffer(rofId)), // d_temp_storage
                                                      bufferSize,                                                        // temp_storage_bytes
                                                      mTimeFrameGPU->getDeviceZCentroids(rofId),                         // d_samples
                                                      mTimeFrameGPU->getDeviceZHistograms(rofId),                        // d_histogram
                                                      mTimeFrameGPU->getConfig().histConf.nBinsXYZ[2],                   // num_levels
                                                      mTimeFrameGPU->getConfig().histConf.lowHistBoundariesXYZ[2],       // lower_level
                                                      mTimeFrameGPU->getConfig().histConf.highHistBoundariesXYZ[2],      // fupper_level
                                                      nLines));                                                          // num_row_pixels
    //   for (int iVertex{0}; iVertex < mStoreVertexerGPU.getConfig().nMaxVertices; ++iVertex) {
    //     discardResult(cub::DeviceReduce::ArgMax(reinterpret_cast<void*>(mStoreVertexerGPU.getCUBTmpBuffer().get()),
    //                                             bufferSize,
    //                                             mStoreVertexerGPU.getHistogramXYZ()[2].get(),
    //                                             mStoreVertexerGPU.getTmpVertexPositionBins().get() + 2,
    //                                             mStoreVertexerGPU.getConfig().histConf.nBinsXYZ[2]));
    //     gpu::computeVertexKernel<<<blocksGrid, 5>>>(getDeviceContext(), iVertex, mVrtParams.clusterContributorsCut, mStoreVertexerGPU.getConfig().histConf.binSpanXYZ[2]);
    //   }
    //   std::vector<gpu::GPUVertex> vertices;
    //   vertices.resize(mStoreVertexerGPU.getConfig().nMaxVertices);
    //   mStoreVertexerGPU.getVertices().copyIntoSizedVector(vertices);

    //   for (auto& vertex : vertices) {
    //     if (vertex.realVertex) {
    //       mVertices.emplace_back(vertex.xCoord, vertex.yCoord, vertex.zCoord, std::array<float, 6>{0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, vertex.contributors, 0.f, -9);
    //     }
    //   }
  }
  gpuThrowOnError();
}

VertexerTraits* createVertexerTraitsGPU()
{
  return new VertexerTraitsGPU;
}

} // namespace its
} // namespace o2