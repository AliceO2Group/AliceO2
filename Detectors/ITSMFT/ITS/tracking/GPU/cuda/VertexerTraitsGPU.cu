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

#include "GPUCommonArray.h"

#ifdef VTX_DEBUG
#include "TTree.h"
#include "TFile.h"
#endif

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

GPUg() void printBufferOnThread(const int* v, size_t size, const unsigned int tId = 0)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == tId) {
    printf("vector :");
    for (int i{0}; i < size; ++i) {
      printf("%d\t", v[i]);
    }
    printf("\n");
  }
}

GPUg() void printBufferOnThreadF(const float* v, size_t size, const unsigned int tId = 0)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == tId) {
    printf("vector :");
    for (int i{0}; i < size; ++i) {
      printf("%.9f\t", v[i]);
    }
    printf("\n");
  }
}

GPUg() void dumpMaximaKernel(const cub::KeyValuePair<int, int>* tmpVertexBins, const int threadId)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == threadId) {
    printf("XmaxBin: %d at index: %d | YmaxBin: %d at index: %d | ZmaxBin: %d at index: %d\n",
           tmpVertexBins[0].value, tmpVertexBins[0].key,
           tmpVertexBins[1].value, tmpVertexBins[1].key,
           tmpVertexBins[2].value, tmpVertexBins[2].key);
  }
}

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
    const int stride{iCurrentLayerClusterIndex * maxTrackletsPerCluster};
    int validTracklets{0};
    for (int iTracklet12{0}; iTracklet12 < nFoundTracklet12[iCurrentLayerClusterIndex]; ++iTracklet12) {
      for (int iTracklet01{0}; iTracklet01 < nFoundTracklet01[iCurrentLayerClusterIndex] && validTracklets < maxTrackletsPerCluster; ++iTracklet01) {
        const float deltaTanLambda{o2::gpu::GPUCommonMath::Abs(tracklets01[stride + iTracklet01].tanLambda - tracklets12[stride + iTracklet12].tanLambda)};
        const float deltaPhi{o2::gpu::GPUCommonMath::Abs(tracklets01[stride + iTracklet01].phi - tracklets12[stride + iTracklet12].phi)};
        if (!usedTracklets[stride + iTracklet01] && deltaTanLambda < tanLambdaCut && deltaPhi < phiCut && validTracklets != maxTrackletsPerCluster) {
          usedTracklets[stride + iTracklet01] = true;
          if constexpr (!initRun) {
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
        // write values outside the histogram boundaries,
        // default behaviour is not to have them added to histogram later
        // (writing zeroes would be problematic)
        centroids[2 * currentThreadIndex] = 2 * lowHistX;
        centroids[2 * currentThreadIndex + 1] = 2 * lowHistY;
      }
    } else {
      // write values outside the histogram boundaries,
      // default behaviour is not to have them added to histogram later
      // (writing zeroes would be problematic)
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

GPUg() void computeVertexKernel(cub::KeyValuePair<int, int>* tmpVertexBins,
                                int* histZ, // Z
                                const float lowHistZ,
                                const float binSizeHistZ,
                                const int nBinsHistZ,
                                Vertex* vertices,
                                float* beamPosition,
                                const int vertIndex,
                                const int minContributors,
                                const int binOpeningZ)
{
  for (size_t currentThreadIndex = blockIdx.x * blockDim.x + threadIdx.x; currentThreadIndex < binOpeningZ; currentThreadIndex += blockDim.x * gridDim.x) {
    if (currentThreadIndex == 0) {
      if (tmpVertexBins[2].value > 1 && (tmpVertexBins[0].value || tmpVertexBins[1].value)) {
        float z{lowHistZ + tmpVertexBins[2].key * binSizeHistZ + binSizeHistZ / 2};
        float ex{0.f};
        float ey{0.f};
        float ez{0.f};
        int sumWZ{tmpVertexBins[2].value};
        float wZ{z * tmpVertexBins[2].value};
        for (int iBin{o2::gpu::GPUCommonMath::Max(0, tmpVertexBins[2].key - binOpeningZ)}; iBin < o2::gpu::GPUCommonMath::Min(tmpVertexBins[2].key + binOpeningZ + 1, nBinsHistZ - 1); ++iBin) {
          if (iBin != tmpVertexBins[2].key) {
            wZ += (lowHistZ + iBin * binSizeHistZ + binSizeHistZ / 2) * histZ[iBin];
            sumWZ += histZ[iBin];
          }
          histZ[iBin] = 0;
        }
        if (sumWZ > minContributors || vertIndex == 0) {
          new (vertices + vertIndex) Vertex{o2::math_utils::Point3D<float>(beamPosition[0], beamPosition[1], wZ / sumWZ), o2::gpu::gpustd::array<float, 6>{ex, 0, ey, 0, 0, ez}, static_cast<ushort>(sumWZ), 0};
        } else {
          new (vertices + vertIndex) Vertex{};
        }
      } else {
        new (vertices + vertIndex) Vertex{};
      }
    }
  }
}
} // namespace gpu

void VertexerTraitsGPU::computeTracklets()
{
  if (!mTimeFrameGPU->getClusters().size()) {
    return;
  }
  for (int rofId{0}; rofId < mTimeFrameGPU->getNrof(); ++rofId) {
    const dim3 threadsPerBlock{gpu::utils::host::getBlockSize(mTimeFrameGPU->getNClustersLayer(rofId, 1))};
    const dim3 blocksGrid{gpu::utils::host::getBlocksGrid(threadsPerBlock, mTimeFrameGPU->getNClustersLayer(rofId, 1))};
    gpu::trackleterKernel<TrackletMode::Layer0Layer1><<<blocksGrid, threadsPerBlock>>>(
      mTimeFrameGPU->getDeviceClustersOnLayer(rofId, 0),
      mTimeFrameGPU->getDeviceClustersOnLayer(rofId, 1),
      mTimeFrameGPU->getNClustersLayer(rofId, 0),
      mTimeFrameGPU->getNClustersLayer(rofId, 1),
      mTimeFrameGPU->getDeviceIndexTableL0(rofId),
      mVrtParams.phiCut,
      mTimeFrameGPU->getDeviceTracklets(rofId, 0),
      mTimeFrameGPU->getDeviceNTrackletsCluster(rofId, 0),
      mDeviceIndexTableUtils,
      mTimeFrameGPU->getConfig().maxTrackletsPerCluster);

    gpu::trackleterKernel<TrackletMode::Layer1Layer2><<<blocksGrid, threadsPerBlock>>>(
      mTimeFrameGPU->getDeviceClustersOnLayer(rofId, 2),
      mTimeFrameGPU->getDeviceClustersOnLayer(rofId, 1),
      mTimeFrameGPU->getNClustersLayer(rofId, 2),
      mTimeFrameGPU->getNClustersLayer(rofId, 1),
      mTimeFrameGPU->getDeviceIndexTableL2(rofId),
      mVrtParams.phiCut,
      mTimeFrameGPU->getDeviceTracklets(rofId, 1),
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
    checkGPUError(cudaMemcpy(NtrackletsClusters01[iRof].data(), mTimeFrameGPU->getDeviceNTrackletsCluster(iRof, 0), sizeof(int) * mTimeFrameGPU->getClustersOnLayer(iRof, 1).size(), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    checkGPUError(cudaMemcpy(NtrackletsClusters12[iRof].data(), mTimeFrameGPU->getDeviceNTrackletsCluster(iRof, 1), sizeof(int) * mTimeFrameGPU->getClustersOnLayer(iRof, 1).size(), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    std::copy(NtrackletsClusters01[iRof].begin(), NtrackletsClusters01[iRof].end(), std::ostream_iterator<double>(out01, "\t"));
    std::copy(NtrackletsClusters12[iRof].begin(), NtrackletsClusters12[iRof].end(), std::ostream_iterator<double>(out12, "\t"));
    out01 << std::endl;
    out12 << std::endl;
  }
  out01.close();
  out12.close();
  // Dump lines on root file
  TFile* trackletFile = TFile::Open("artefacts_tf_gpu.root", "recreate");
  TTree* tr_tre = new TTree("tracklets", "tf");
  std::vector<o2::its::Tracklet> tracklets_vec01(0);
  std::vector<o2::its::Tracklet> tracklets_vec12(0);
  std::vector<o2::its::Tracklet> tracklets_clean01(0);
  std::vector<o2::its::Tracklet> tracklets_clean12(0);
  tr_tre->Branch("Tracklets0", &tracklets_clean01);
  tr_tre->Branch("Tracklets1", &tracklets_clean12);
  for (int iRof{0}; iRof < mTimeFrameGPU->getNrof(); ++iRof) {
    tracklets_clean01.clear();
    tracklets_clean12.clear();
    tracklets_vec01.resize(mTimeFrameGPU->getClustersOnLayer(iRof, 1).size() * mTimeFrameGPU->getConfig().maxTrackletsPerCluster); // Nclusters * TrackletsPerCluster
    tracklets_vec12.resize(mTimeFrameGPU->getClustersOnLayer(iRof, 1).size() * mTimeFrameGPU->getConfig().maxTrackletsPerCluster);
    checkGPUError(cudaMemcpy(tracklets_vec01.data(),
                             mTimeFrameGPU->getDeviceTracklets(iRof, 0),
                             mTimeFrameGPU->getClustersOnLayer(iRof, 1).size() * sizeof(Tracklet) * mTimeFrameGPU->getConfig().maxTrackletsPerCluster,
                             cudaMemcpyDeviceToHost),
                  __FILE__, __LINE__);
    checkGPUError(cudaMemcpy(tracklets_vec12.data(),
                             mTimeFrameGPU->getDeviceTracklets(iRof, 1),
                             mTimeFrameGPU->getClustersOnLayer(iRof, 1).size() * sizeof(Tracklet) * mTimeFrameGPU->getConfig().maxTrackletsPerCluster,
                             cudaMemcpyDeviceToHost),
                  __FILE__, __LINE__);
    for (auto iCluster{0}; iCluster < NtrackletsClusters01[iRof].size(); ++iCluster) {
      auto nTracklets{NtrackletsClusters01[iRof][iCluster]};
      for (auto iTracklet{0}; iTracklet < nTracklets; ++iTracklet) {
        tracklets_clean01.push_back(tracklets_vec01[iCluster * mTimeFrameGPU->getConfig().maxTrackletsPerCluster + iTracklet]);
      }
    }
    for (auto iCluster{0}; iCluster < NtrackletsClusters12[iRof].size(); ++iCluster) {
      auto nTracklets{NtrackletsClusters12[iRof][iCluster]};
      for (auto iTracklet{0}; iTracklet < nTracklets; ++iTracklet) {
        tracklets_clean12.push_back(tracklets_vec12[iCluster * mTimeFrameGPU->getConfig().maxTrackletsPerCluster + iTracklet]);
      }
    }
    tr_tre->Fill();
  }
  trackletFile->cd();
  tr_tre->Write();
  trackletFile->Close();
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

    // Reset used tracklets
    checkGPUError(cudaMemset(mTimeFrameGPU->getDeviceUsedTracklets(rofId), false, sizeof(unsigned char) * mTimeFrameGPU->getConfig().maxTrackletsPerCluster * mTimeFrameGPU->getNClustersLayer(rofId, 1)), __FILE__, __LINE__);

    gpu::trackletSelectionKernel<true><<<blocksGrid, threadsPerBlock>>>(
      mTimeFrameGPU->getDeviceClustersOnLayer(rofId, 0),
      mTimeFrameGPU->getDeviceClustersOnLayer(rofId, 1),
      mTimeFrameGPU->getNClustersLayer(rofId, 1),
      mTimeFrameGPU->getDeviceTracklets(rofId, 0),
      mTimeFrameGPU->getDeviceTracklets(rofId, 1),
      mTimeFrameGPU->getDeviceNTrackletsCluster(rofId, 0),
      mTimeFrameGPU->getDeviceNTrackletsCluster(rofId, 1),
      mTimeFrameGPU->getDeviceUsedTracklets(rofId),
      mTimeFrameGPU->getDeviceLines(rofId),
      mTimeFrameGPU->getDeviceNFoundLines(rofId),
      mTimeFrameGPU->getDeviceExclusiveNFoundLines(rofId),
      mTimeFrameGPU->getConfig().maxTrackletsPerCluster,
      mVrtParams.tanLambdaCut,
      mVrtParams.phiCut);

    discardResult(cub::DeviceScan::ExclusiveSum(reinterpret_cast<void*>(mTimeFrameGPU->getDeviceCUBBuffer(rofId)),
                                                bufferSize,
                                                mTimeFrameGPU->getDeviceNFoundLines(rofId),
                                                mTimeFrameGPU->getDeviceExclusiveNFoundLines(rofId),
                                                mTimeFrameGPU->getNClustersLayer(rofId, 1)));
    // Reset used tracklets
    checkGPUError(cudaMemset(mTimeFrameGPU->getDeviceUsedTracklets(rofId), false, sizeof(unsigned char) * mTimeFrameGPU->getConfig().maxTrackletsPerCluster * mTimeFrameGPU->getNClustersLayer(rofId, 1)), __FILE__, __LINE__);

    gpu::trackletSelectionKernel<false><<<blocksGrid, threadsPerBlock>>>(
      mTimeFrameGPU->getDeviceClustersOnLayer(rofId, 0),
      mTimeFrameGPU->getDeviceClustersOnLayer(rofId, 1),
      mTimeFrameGPU->getNClustersLayer(rofId, 1),
      mTimeFrameGPU->getDeviceTracklets(rofId, 0),
      mTimeFrameGPU->getDeviceTracklets(rofId, 1),
      mTimeFrameGPU->getDeviceNTrackletsCluster(rofId, 0),
      mTimeFrameGPU->getDeviceNTrackletsCluster(rofId, 1),
      mTimeFrameGPU->getDeviceUsedTracklets(rofId),
      mTimeFrameGPU->getDeviceLines(rofId),
      mTimeFrameGPU->getDeviceNFoundLines(rofId),
      mTimeFrameGPU->getDeviceExclusiveNFoundLines(rofId),
      mTimeFrameGPU->getConfig().maxTrackletsPerCluster,
      mVrtParams.tanLambdaCut,
      mVrtParams.phiCut);
    gpuThrowOnError();
  }
#ifdef VTX_DEBUG
  std::vector<std::vector<int>> NFoundLines(mTimeFrameGPU->getNrof()), ExcNFoundLines(mTimeFrameGPU->getNrof());
  std::ofstream nlines_out("N_lines_gpu.txt");
  for (size_t rofId{0}; rofId < mTimeFrameGPU->getNrof(); ++rofId) {
    NFoundLines[rofId].resize(mTimeFrameGPU->getNClustersLayer(rofId, 1));
    ExcNFoundLines[rofId].resize(mTimeFrameGPU->getNClustersLayer(rofId, 1));
    checkGPUError(cudaMemcpy(NFoundLines[rofId].data(), mTimeFrameGPU->getDeviceNFoundLines(rofId), sizeof(int) * mTimeFrameGPU->getNClustersLayer(rofId, 1), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    checkGPUError(cudaMemcpy(ExcNFoundLines[rofId].data(), mTimeFrameGPU->getDeviceExclusiveNFoundLines(rofId), sizeof(int) * mTimeFrameGPU->getNClustersLayer(rofId, 1), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    std::copy(NFoundLines[rofId].begin(), NFoundLines[rofId].end(), std::ostream_iterator<double>(nlines_out, "\t"));
    nlines_out << std::endl;
    std::copy(ExcNFoundLines[rofId].begin(), ExcNFoundLines[rofId].end(), std::ostream_iterator<double>(nlines_out, "\t"));
    nlines_out << std::endl
               << " ---\n";
  }
  nlines_out.close();

  // Dump lines on root file
  TFile* trackletFile = TFile::Open("artefacts_tf_gpu.root", "update");
  TTree* ln_tre = new TTree("lines", "tf");
  std::vector<o2::its::Line> lines_vec(0);
  ln_tre->Branch("Lines", &lines_vec);
  for (int rofId{0}; rofId < mTimeFrameGPU->getNrof(); ++rofId) {
    lines_vec.clear();
    auto sum = std::accumulate(NFoundLines[rofId].begin(), NFoundLines[rofId].end(), 0);
    lines_vec.resize(sum);
    checkGPUError(cudaMemcpy(lines_vec.data(), mTimeFrameGPU->getDeviceLines(rofId), sizeof(Line) * sum, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    ln_tre->Fill();
  }
  trackletFile->cd();
  ln_tre->Write();
  trackletFile->Close();
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
      mTimeFrameGPU->getDeviceXYCentroids(rofId), // output
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
                                                                  mTimeFrameGPU->getDeviceZCentroids(rofId), // output
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

    for (int iVertex{0}; iVertex < mTimeFrameGPU->getConfig().maxVerticesCapacity; ++iVertex) {
      discardResult(cub::DeviceReduce::ArgMax(reinterpret_cast<void*>(mTimeFrameGPU->getDeviceCUBBuffer(rofId)),
                                              bufferSize,
                                              mTimeFrameGPU->getDeviceZHistograms(rofId),
                                              mTimeFrameGPU->getTmpVertexPositionBins(rofId) + 2,
                                              mTimeFrameGPU->getConfig().histConf.nBinsXYZ[2]));

      gpu::computeVertexKernel<<<blocksGrid, 5>>>(mTimeFrameGPU->getTmpVertexPositionBins(rofId),
                                                  mTimeFrameGPU->getDeviceZHistograms(rofId),
                                                  mTimeFrameGPU->getConfig().histConf.lowHistBoundariesXYZ[2],
                                                  mTimeFrameGPU->getConfig().histConf.binSizeHistZ,
                                                  mTimeFrameGPU->getConfig().histConf.nBinsXYZ[2],
                                                  mTimeFrameGPU->getDeviceVertices(rofId),
                                                  mTimeFrameGPU->getDeviceBeamPosition(rofId),
                                                  iVertex,
                                                  mVrtParams.clusterContributorsCut,
                                                  mTimeFrameGPU->getConfig().histConf.binSpanXYZ[2]);
    }
    std::vector<Vertex> GPUvertices;
    GPUvertices.resize(mTimeFrameGPU->getConfig().maxVerticesCapacity);
    checkGPUError(cudaMemcpy(GPUvertices.data(), mTimeFrameGPU->getDeviceVertices(rofId), sizeof(gpu::GPUVertex) * mTimeFrameGPU->getConfig().maxVerticesCapacity, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    int nRealVertices{0};
    for (auto& vertex : GPUvertices) {
      if (vertex.getX() || vertex.getY() || vertex.getZ()) {
        ++nRealVertices;
      }
    }
    mTimeFrameGPU->addPrimaryVertices(gsl::span<const Vertex>{GPUvertices.data(), static_cast<gsl::span<const Vertex>::size_type>(nRealVertices)});
  }
  gpuThrowOnError();
}

VertexerTraits* createVertexerTraitsGPU()
{
  return new VertexerTraitsGPU;
}

} // namespace its
} // namespace o2