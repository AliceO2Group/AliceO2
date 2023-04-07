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
#include <thread>

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/unique.h>
#include <thrust/remove.h>

#include "ITStracking/Constants.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/MathUtils.h"

#include "ITStrackingGPU/TrackerTraitsGPU.h"
#include "ITStrackingGPU/TracerGPU.h"

#include "GPUCommonLogger.h"
#include "GPUCommonAlgorithmThrust.h"

#ifndef __HIPCC__
#define THRUST_NAMESPACE thrust::cuda
#else
#define THRUST_NAMESPACE thrust::hip
#endif

namespace o2
{
namespace its
{
using gpu::utils::checkGPUError;
using namespace constants::its2;

namespace gpu
{

GPUg() void printBufferLayerOnThread(const int layer, const int* v, size_t size, const int len = 150, const unsigned int tId = 0)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == tId) {
    for (int i{0}; i < size; ++i) {
      if (!(i % len)) {
        printf("\n layer %d: ===>%d/%d\t", layer, i, (int)size);
      }
      printf("%d\t", v[i]);
    }
    printf("\n");
  }
}

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

  return int4{o2::gpu::GPUCommonMath::Max(0, utils.getZBinIndex(layerIndex + 1, zRangeMin)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMin)),
              o2::gpu::GPUCommonMath::Min(ZBins - 1, utils.getZBinIndex(layerIndex + 1, zRangeMax)),
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

// Dump vertices
GPUg() void printVertices(const Vertex* v, size_t size, const unsigned int tId = 0)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == tId) {
    printf("vertices: ");
    for (int i{0}; i < size; ++i) {
      printf("x=%f y=%f z=%f\n", v[i].getX(), v[i].getY(), v[i].getZ());
    }
  }
}

// Dump tracklets
GPUg() void printTracklets(const Tracklet* t,
                           const int offset,
                           const int startRof,
                           const int nrof,
                           const int* roFrameClustersCurrentLayer, // Number of clusters on layer 0 per ROF
                           const int* roFrameClustersNextLayer,    // Number of clusters on layer 1 per ROF
                           const int maxClustersPerRof = 5e2,
                           const int maxTrackletsPerCluster = 50,
                           const unsigned int tId = 0)
{
  if (threadIdx.x == tId) {
    auto offsetCurrent{roFrameClustersCurrentLayer[offset]};
    auto offsetNext{roFrameClustersNextLayer[offset]};
    auto offsetChunk{(startRof - offset) * maxClustersPerRof * maxTrackletsPerCluster};
    for (int i{offsetChunk}; i < offsetChunk + nrof * maxClustersPerRof * maxTrackletsPerCluster; ++i) {
      if (t[i].firstClusterIndex != -1) {
        t[i].dump(offsetCurrent, offsetNext);
      }
    }
  }
}

GPUg() void printTrackletsNotStrided(const Tracklet* t,
                                     const int offset,
                                     const int* roFrameClustersCurrentLayer, // Number of clusters on layer 0 per ROF
                                     const int* roFrameClustersNextLayer,    // Number of clusters on layer 1 per ROF
                                     const int ntracklets,
                                     const unsigned int tId = 0)
{
  if (threadIdx.x == tId) {
    auto offsetCurrent{roFrameClustersCurrentLayer[offset]};
    auto offsetNext{roFrameClustersNextLayer[offset]};
    for (int i{0}; i < ntracklets; ++i) {
      t[i].dump(offsetCurrent, offsetNext);
    }
  }
}

// Compute the tracklets for a given layer
template <int nLayers = 7>
GPUg() void computeLayerTrackletsKernelSingleRof(
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
  const StaticTrackingParameters<nLayers>* trkPars,
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

template <int nLayers = 7>
GPUg() void compileTrackletsLookupTableKernel(const Tracklet* tracklets,
                                              int* trackletsLookUpTable,
                                              const int nTracklets)
{
  for (int currentTrackletIndex = blockIdx.x * blockDim.x + threadIdx.x; currentTrackletIndex < nTracklets; currentTrackletIndex += blockDim.x * gridDim.x) {
    auto& tracklet{tracklets[currentTrackletIndex]};
    if (tracklet.firstClusterIndex >= 0) {
      atomicAdd(trackletsLookUpTable + tracklet.firstClusterIndex, 1);
    }
  }
}

template <int nLayers = 7>
GPUg() void computeLayerTrackletsKernelMultipleRof(
  const int layerIndex,
  const int iteration,
  const unsigned int startRofId,
  const unsigned int rofSize,
  const int maxRofs,
  const Cluster* clustersCurrentLayer,        // input data rof0
  const Cluster* clustersNextLayer,           // input data rof0-delta <rof0< rof0+delta (up to 3 rofs)
  const int* roFrameClustersCurrentLayer,     // Number of clusters on layer 0 per ROF
  const int* roFrameClustersNextLayer,        // Number of clusters on layer 1 per ROF
  const int* indexTablesNext,                 // input data rof0-delta <rof0< rof0+delta (up to 3 rofs)
  const unsigned char* usedClustersLayer,     // input data rof0
  const unsigned char* usedClustersNextLayer, // input data rof1
  Tracklet* tracklets,                        // output data
  const Vertex* vertices,
  const int* nVertices,
  const float phiCut,
  const float minR,
  const float maxR,
  const float meanDeltaR,
  const float positionResolution,
  const float mSAngle,
  const StaticTrackingParameters<nLayers>* trkPars,
  const IndexTableUtils* utils,
  const unsigned int maxClustersPerRof = 5e2,
  const unsigned int maxTrackletsPerCluster = 50)
{
  const int phiBins{utils->getNphiBins()};
  const int zBins{utils->getNzBins()};
  for (unsigned int iRof{blockIdx.x}; iRof < rofSize; iRof += gridDim.x) {
    auto rof0 = iRof + startRofId;
    auto nClustersCurrentLayerRof = roFrameClustersCurrentLayer[rof0 + 1] - roFrameClustersCurrentLayer[rof0];
    auto* clustersCurrentLayerRof = clustersCurrentLayer + (roFrameClustersCurrentLayer[rof0] - roFrameClustersCurrentLayer[startRofId]);
    auto nVerticesRof0 = nVertices[rof0 + 1] - nVertices[rof0];
    auto trackletsRof0 = tracklets + maxTrackletsPerCluster * maxClustersPerRof * iRof;
    for (int currentClusterIndex = threadIdx.x; currentClusterIndex < nClustersCurrentLayerRof; currentClusterIndex += blockDim.x) {
      if (nClustersCurrentLayerRof > maxClustersPerRof) {
        printf("its-gpu-tracklet finder: on layer %d found more clusters per ROF (%d) than maximum set (%d), check the configuration!\n", layerIndex, nClustersCurrentLayerRof, maxClustersPerRof);
      }
      unsigned int storedTracklets{0};
      const Cluster& currentCluster{clustersCurrentLayerRof[currentClusterIndex]};
      const int currentSortedIndex{roFrameClustersCurrentLayer[rof0] + currentClusterIndex};
      const int currentSortedIndexChunk{currentSortedIndex - roFrameClustersCurrentLayer[startRofId]};
      if (usedClustersLayer[currentSortedIndex]) {
        continue;
      }

      int minRof = (rof0 >= trkPars->DeltaROF) ? rof0 - trkPars->DeltaROF : 0;
      int maxRof = (rof0 == maxRofs - trkPars->DeltaROF) ? rof0 : rof0 + trkPars->DeltaROF; // works with delta = {0, 1}
      const float inverseR0{1.f / currentCluster.radius};

      for (int iPrimaryVertex{0}; iPrimaryVertex < nVerticesRof0; iPrimaryVertex++) {
        const auto& primaryVertex{vertices[nVertices[rof0] + iPrimaryVertex]};
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
        const int tableSize{phiBins * zBins + 1};
        for (int rof1{minRof}; rof1 <= maxRof; ++rof1) {
          auto nClustersNext{roFrameClustersNextLayer[rof1 + 1] - roFrameClustersNextLayer[rof1]};
          if (!nClustersNext) { // number of clusters on next layer > 0
            continue;
          }
          for (int iPhiCount{0}; iPhiCount < phiBinsNum; iPhiCount++) {
            int iPhiBin = (selectedBinsRect.y + iPhiCount) % trkPars->PhiBins;
            const int firstBinIndex{utils->getBinIndex(selectedBinsRect.x, iPhiBin)};
            const int maxBinIndex{firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1};
            const int firstRowClusterIndex = indexTablesNext[(rof1 - startRofId) * tableSize + firstBinIndex];
            const int maxRowClusterIndex = indexTablesNext[(rof1 - startRofId) * tableSize + maxBinIndex];
            for (int iNextCluster{firstRowClusterIndex}; iNextCluster < maxRowClusterIndex; ++iNextCluster) {
              if (iNextCluster >= nClustersNext) {
                break;
              }
              auto nextClusterIndex{roFrameClustersNextLayer[rof1] - roFrameClustersNextLayer[startRofId] + iNextCluster};
              const Cluster& nextCluster{clustersNextLayer[nextClusterIndex]};
              if (usedClustersNextLayer[nextCluster.clusterId]) {
                continue;
              }
              const float deltaPhi{o2::gpu::GPUCommonMath::Abs(currentCluster.phi - nextCluster.phi)};
              const float deltaZ{o2::gpu::GPUCommonMath::Abs(tanLambda * (nextCluster.radius - currentCluster.radius) + currentCluster.zCoordinate - nextCluster.zCoordinate)};

              if ((deltaZ / sigmaZ < trkPars->NSigmaCut && (deltaPhi < phiCut || o2::gpu::GPUCommonMath::Abs(deltaPhi - constants::math::TwoPi) < phiCut))) {
                const float phi{o2::gpu::GPUCommonMath::ATan2(currentCluster.yCoordinate - nextCluster.yCoordinate, currentCluster.xCoordinate - nextCluster.xCoordinate)};
                const float tanL{(currentCluster.zCoordinate - nextCluster.zCoordinate) / (currentCluster.radius - nextCluster.radius)};
                const size_t stride{currentClusterIndex * maxTrackletsPerCluster};
                if (storedTracklets < maxTrackletsPerCluster) {
                  new (trackletsRof0 + stride + storedTracklets) Tracklet{currentSortedIndexChunk, nextClusterIndex, tanL, phi, static_cast<ushort>(rof0), static_cast<ushort>(rof1)};
                }
                // else {
                // printf("its-gpu-tracklet-finder: on rof %d layer: %d: found more tracklets (%d) than maximum allowed per cluster. This is lossy!\n", rof0, layerIndex, storedTracklets);
                // }
                ++storedTracklets;
              }
            }
          }
        }
      }
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
template <bool initRun, int nLayers = 7>
GPUg() void computeLayerCellsKernel(
  const Tracklet* trackletsCurrentLayer,
  const Tracklet* trackletsNextLayer,
  const int* trackletsCurrentLayerLUT,
  const int nTrackletsCurrent,
  Cell* cells,
  int* cellsLUT,
  const StaticTrackingParameters<nLayers>* trkPars)
{
  for (int iCurrentTrackletIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentTrackletIndex < nTrackletsCurrent; iCurrentTrackletIndex += blockDim.x * gridDim.x) {
    const Tracklet& currentTracklet = trackletsCurrentLayer[iCurrentTrackletIndex];
    const int nextLayerClusterIndex{currentTracklet.secondClusterIndex};
    const int nextLayerFirstTrackletIndex{trackletsCurrentLayerLUT[nextLayerClusterIndex]};
    const int nextLayerLastTrackletIndex{trackletsCurrentLayerLUT[nextLayerClusterIndex + 1]};
    if (nextLayerFirstTrackletIndex == nextLayerLastTrackletIndex) {
      continue;
    }
    int foundCells{0};
    for (int iNextTrackletIndex{nextLayerFirstTrackletIndex}; iNextTrackletIndex < nextLayerLastTrackletIndex; ++iNextTrackletIndex) {
      if (trackletsNextLayer[iNextTrackletIndex].firstClusterIndex != nextLayerClusterIndex) {
        break;
      }
      const Tracklet& nextTracklet = trackletsNextLayer[iNextTrackletIndex];
      const float deltaTanLambda{o2::gpu::GPUCommonMath::Abs(currentTracklet.tanLambda - nextTracklet.tanLambda)};
      const float tanLambda{(currentTracklet.tanLambda + nextTracklet.tanLambda) * 0.5f};

      if (deltaTanLambda / trkPars->CellDeltaTanLambdaSigma < trkPars->NSigmaCut) {
        if constexpr (!initRun) {
          new (cells + cellsLUT[iCurrentTrackletIndex] + foundCells) Cell{currentTracklet.firstClusterIndex, nextTracklet.firstClusterIndex,
                                                                          nextTracklet.secondClusterIndex,
                                                                          iCurrentTrackletIndex,
                                                                          iNextTrackletIndex,
                                                                          tanLambda};
        }
        ++foundCells;
      }
    }
    if constexpr (initRun) {
      // Fill cell Lookup table
      cellsLUT[iCurrentTrackletIndex] = foundCells;
    }
  }
}

template <bool initRun, int nLayers = 7>
GPUg() void computeLayerCellNeighboursKernel(Cell* cellsCurrentLayer,
                                             Cell* cellsNextLayer,
                                             const int layerIndex,
                                             const int* cellsNextLayerLUT,
                                             int* neighboursLUT,
                                             int* cellNeighbours,
                                             const int* nCells,
                                             const int maxCellNeighbours = 1e2)
{
  for (int iCurrentCellIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentCellIndex < nCells[layerIndex]; iCurrentCellIndex += blockDim.x * gridDim.x) {
    const Cell& currentCell = cellsCurrentLayer[iCurrentCellIndex];
    const int nextLayerTrackletIndex{currentCell.getSecondTrackletIndex()};
    const int nextLayerFirstCellIndex{cellsNextLayerLUT[nextLayerTrackletIndex]};
    const int nextLayerLastCellIndex{cellsNextLayerLUT[nextLayerTrackletIndex + 1]};
    int foundNeighbours{0};
    for (int iNextCell{nextLayerFirstCellIndex}; iNextCell < nextLayerLastCellIndex; ++iNextCell) {
      Cell& nextCell = cellsNextLayer[iNextCell];
      if (nextCell.getFirstTrackletIndex() != nextLayerTrackletIndex) { // Check if cells share the same tracklet
        break;
      }
      if constexpr (initRun) {
        atomicAdd(neighboursLUT + iNextCell, 1);
      } else {
        if (foundNeighbours >= maxCellNeighbours) {
          printf("its-gpu-neighbours-finder: on layer: %d: found more neighbours (%d) than maximum allowed per cell, skipping writing. This is lossy!\n", layerIndex, neighboursLUT[iNextCell]);
          continue;
        }
        cellNeighbours[neighboursLUT[iNextCell] + foundNeighbours++] = iCurrentCellIndex;

        const int currentCellLevel{currentCell.getLevel()};
        if (currentCellLevel >= nextCell.getLevel()) {
          atomicExch(nextCell.getLevelPtr(), currentCellLevel + 1);
        }
      }
    }
  }
}

} // namespace gpu

template <int nLayers>
void TrackerTraitsGPU<nLayers>::initialiseTimeFrame(const int iteration)
{
  mTimeFrameGPU->initialise(iteration, mTrkParams[iteration], nLayers);
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::computeLayerTracklets(const int iteration)
{
  if (!mTimeFrameGPU->getClusters().size()) {
    return;
  }
  const Vertex diamondVert({mTrkParams[iteration].Diamond[0], mTrkParams[iteration].Diamond[1], mTrkParams[iteration].Diamond[2]}, {25.e-6f, 0.f, 0.f, 25.e-6f, 0.f, 36.f}, 1, 1.f);
  gsl::span<const Vertex> diamondSpan(&diamondVert, 1);
  std::vector<std::thread> threads(mTimeFrameGPU->getNChunks());
  // std::array<std::array<int, 3>, nLayers - 1> totTrackletsChunk{std::array<int, 3>{0, 0, 0}, std::array<int, 3>{0, 0, 0}, std::array<int, 3>{0, 0, 0}, std::array<int, 3>{0, 0, 0}, std::array<int, 3>{0, 0, 0}};
  for (int chunkId{0}; chunkId < mTimeFrameGPU->getNChunks(); ++chunkId) {
    int maxTracklets{static_cast<int>(mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->clustersPerROfCapacity) *
                     static_cast<int>(mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->maxTrackletsPerCluster)};
    int maxRofPerChunk{mTimeFrameGPU->mNrof / (int)mTimeFrameGPU->getNChunks()};
    // Define workload
    auto doTrackReconstruction = [&, chunkId, maxRofPerChunk, iteration]() -> void {
      auto offset = chunkId * maxRofPerChunk;
      auto maxROF = offset + maxRofPerChunk;
      while (offset < maxROF) {
        auto rofs = mTimeFrameGPU->loadChunkData<gpu::Task::Tracker>(chunkId, offset, maxROF);
        RANGE("chunk_gpu_tracking", 1);
        for (int iLayer{0}; iLayer < nLayers - 1; ++iLayer) {
          auto nclus = mTimeFrameGPU->getTotalClustersPerROFrange(offset, rofs, iLayer);
          const float meanDeltaR{mTrkParams[iteration].LayerRadii[iLayer + 1] - mTrkParams[iteration].LayerRadii[iLayer]};
          gpu::computeLayerTrackletsKernelMultipleRof<<<rofs, 1024, 0, mTimeFrameGPU->getStream(chunkId).get()>>>(
            iLayer,                                                                                // const int layerIndex,
            iteration,                                                                             // const int iteration,
            offset,                                                                                // const unsigned int startRofId,
            rofs,                                                                                  // const unsigned int rofSize,
            0,                                                                                     // const unsigned int deltaRof,
            mTimeFrameGPU->getChunk(chunkId).getDeviceClusters(iLayer),                            // const Cluster* clustersCurrentLayer,
            mTimeFrameGPU->getChunk(chunkId).getDeviceClusters(iLayer + 1),                        // const Cluster* clustersNextLayer,
            mTimeFrameGPU->getDeviceROframesClusters(iLayer),                                      // const int* roFrameClustersCurrentLayer, // Number of clusters on layer 0 per ROF
            mTimeFrameGPU->getDeviceROframesClusters(iLayer + 1),                                  // const int* roFrameClustersNextLayer,    // Number of clusters on layer 1 per ROF
            mTimeFrameGPU->getChunk(chunkId).getDeviceIndexTables(iLayer + 1),                     // const int* indexTableNextLayer,
            mTimeFrameGPU->getDeviceUsedClusters(iLayer),                                          // const int* usedClustersCurrentLayer,
            mTimeFrameGPU->getDeviceUsedClusters(iLayer + 1),                                      // const int* usedClustersNextLayer,
            mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(iLayer),                           // Tracklet* tracklets,       // output data
            mTimeFrameGPU->getDeviceVertices(),                                                    // const Vertex* vertices,
            mTimeFrameGPU->getDeviceROframesPV(),                                                  // const int* pvROFrame,
            mTimeFrameGPU->getPhiCut(iLayer),                                                      // const float phiCut,
            mTimeFrameGPU->getMinR(iLayer + 1),                                                    // const float minR,
            mTimeFrameGPU->getMaxR(iLayer + 1),                                                    // const float maxR,
            meanDeltaR,                                                                            // const float meanDeltaR,
            mTimeFrameGPU->getPositionResolution(iLayer),                                          // const float positionResolution,
            mTimeFrameGPU->getMSangle(iLayer),                                                     // const float mSAngle,
            mTimeFrameGPU->getDeviceTrackingParameters(),                                          // const StaticTrackingParameters<nLayers>* trkPars,
            mTimeFrameGPU->getDeviceIndexTableUtils(),                                             // const IndexTableUtils* utils
            mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->clustersPerROfCapacity,  // const int clustersPerROfCapacity,
            mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->maxTrackletsPerCluster); // const int maxTrackletsPerCluster

          // Remove empty tracklets due to striding.
          auto nulltracklet = o2::its::Tracklet{};
          auto thrustTrackletsBegin = thrust::device_ptr<o2::its::Tracklet>(mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(iLayer));
          auto thrustTrackletsEnd = thrust::device_ptr<o2::its::Tracklet>(mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(iLayer) + (int)rofs * maxTracklets);
          auto thrustTrackletsAfterEraseEnd = thrust::remove(THRUST_NAMESPACE::par.on(mTimeFrameGPU->getStream(chunkId).get()),
                                                             thrustTrackletsBegin,
                                                             thrustTrackletsEnd,
                                                             nulltracklet);
          // Sort tracklets by first cluster index.
          thrust::sort(THRUST_NAMESPACE::par.on(mTimeFrameGPU->getStream(chunkId).get()),
                       thrustTrackletsBegin,
                       thrustTrackletsAfterEraseEnd,
                       gpu::trackletSortIndexFunctor<o2::its::Tracklet>());

          // Remove duplicates.
          auto thrustTrackletsAfterUniqueEnd = thrust::unique(THRUST_NAMESPACE::par.on(mTimeFrameGPU->getStream(chunkId).get()), thrustTrackletsBegin, thrustTrackletsAfterEraseEnd);

          discardResult(cudaStreamSynchronize(mTimeFrameGPU->getStream(chunkId).get()));
          mTimeFrameGPU->getHostNTracklets(chunkId)[iLayer] = thrustTrackletsAfterUniqueEnd - thrustTrackletsBegin;
          // Compute tracklet lookup table.
          gpu::compileTrackletsLookupTableKernel<<<rofs, 1024, 0, mTimeFrameGPU->getStream(chunkId).get()>>>(mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(iLayer),
                                                                                                             mTimeFrameGPU->getChunk(chunkId).getDeviceTrackletsLookupTables(iLayer),
                                                                                                             mTimeFrameGPU->getHostNTracklets(chunkId)[iLayer]);
          discardResult(cub::DeviceScan::ExclusiveSum(mTimeFrameGPU->getChunk(chunkId).getDeviceCUBTmpBuffer(),                       // d_temp_storage
                                                      mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->tmpCUBBufferSize, // temp_storage_bytes
                                                      mTimeFrameGPU->getChunk(chunkId).getDeviceTrackletsLookupTables(iLayer),        // d_in
                                                      mTimeFrameGPU->getChunk(chunkId).getDeviceTrackletsLookupTables(iLayer),        // d_out
                                                      nclus,                                                                          // num_items
                                                      mTimeFrameGPU->getStream(chunkId).get()));

          // Create tracklets labels, at the moment on the host
          if (mTimeFrameGPU->hasMCinformation()) {
            std::vector<o2::its::Tracklet> tracklets(mTimeFrameGPU->getHostNTracklets(chunkId)[iLayer]);
            checkGPUError(cudaHostRegister(tracklets.data(), tracklets.size() * sizeof(o2::its::Tracklet), cudaHostRegisterDefault));
            checkGPUError(cudaMemcpyAsync(tracklets.data(), mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(iLayer), tracklets.size() * sizeof(o2::its::Tracklet), cudaMemcpyDeviceToHost, mTimeFrameGPU->getStream(chunkId).get()));
            for (auto& trk : tracklets) {
              MCCompLabel label;
              int currentId{mTimeFrameGPU->mClusters[iLayer][trk.firstClusterIndex].clusterId};   // This is not yet offsetted to the index of the first cluster of the chunk
              int nextId{mTimeFrameGPU->mClusters[iLayer + 1][trk.secondClusterIndex].clusterId}; // This is not yet offsetted to the index of the first cluster of the chunk
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
              // TODO: implment label merging.
              // mTimeFrameGPU->getTrackletsLabel(iLayer).emplace_back(label);
            }
            checkGPUError(cudaHostUnregister(tracklets.data()));
          }
        }
        for (int iLayer{0}; iLayer < nLayers - 2; ++iLayer) {
          // Compute layer cells.
          gpu::computeLayerCellsKernel<true><<<10, 1024, 0, mTimeFrameGPU->getStream(chunkId).get()>>>(
            mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(iLayer),
            mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(iLayer + 1),
            mTimeFrameGPU->getChunk(chunkId).getDeviceTrackletsLookupTables(iLayer + 1),
            mTimeFrameGPU->getHostNTracklets(chunkId)[iLayer],
            nullptr,
            mTimeFrameGPU->getChunk(chunkId).getDeviceCellsLookupTables(iLayer),
            mTimeFrameGPU->getDeviceTrackingParameters());

          // Compute number of found Cells
          checkGPUError(cub::DeviceReduce::Sum(mTimeFrameGPU->getChunk(chunkId).getDeviceCUBTmpBuffer(),                       // d_temp_storage
                                               mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->tmpCUBBufferSize, // temp_storage_bytes
                                               mTimeFrameGPU->getChunk(chunkId).getDeviceCellsLookupTables(iLayer),            // d_in
                                               mTimeFrameGPU->getChunk(chunkId).getDeviceNFoundCells() + iLayer,               // d_out
                                               mTimeFrameGPU->getHostNTracklets(chunkId)[iLayer],                              // num_items
                                               mTimeFrameGPU->getStream(chunkId).get()));
          // Compute LUT
          discardResult(cub::DeviceScan::ExclusiveSum(mTimeFrameGPU->getChunk(chunkId).getDeviceCUBTmpBuffer(),                       // d_temp_storage
                                                      mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->tmpCUBBufferSize, // temp_storage_bytes
                                                      mTimeFrameGPU->getChunk(chunkId).getDeviceCellsLookupTables(iLayer),            // d_in
                                                      mTimeFrameGPU->getChunk(chunkId).getDeviceCellsLookupTables(iLayer),            // d_out
                                                      mTimeFrameGPU->getHostNTracklets(chunkId)[iLayer],                              // num_items
                                                      mTimeFrameGPU->getStream(chunkId).get()));

          gpu::computeLayerCellsKernel<false><<<10, 1024, 0, mTimeFrameGPU->getStream(chunkId).get()>>>(
            mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(iLayer),
            mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(iLayer + 1),
            mTimeFrameGPU->getChunk(chunkId).getDeviceTrackletsLookupTables(iLayer + 1),
            mTimeFrameGPU->getHostNTracklets(chunkId)[iLayer],
            mTimeFrameGPU->getChunk(chunkId).getDeviceCells(iLayer),
            mTimeFrameGPU->getChunk(chunkId).getDeviceCellsLookupTables(iLayer),
            mTimeFrameGPU->getDeviceTrackingParameters());
        }
        checkGPUError(cudaMemcpyAsync(mTimeFrameGPU->getHostNCells(chunkId).data(),
                                      mTimeFrameGPU->getChunk(chunkId).getDeviceNFoundCells(),
                                      (nLayers - 2) * sizeof(int),
                                      cudaMemcpyDeviceToHost,
                                      mTimeFrameGPU->getStream(chunkId).get()));
        // Create cells labels TODO: make it work after fixing the tracklets labels
        if (mTimeFrameGPU->hasMCinformation()) {
          for (int iLayer{0}; iLayer < nLayers - 2; ++iLayer) {
            std::vector<o2::its::Cell> cells(mTimeFrameGPU->getHostNCells(chunkId)[iLayer]);
            // Async with not registered memory?
            checkGPUError(cudaMemcpyAsync(cells.data(), mTimeFrameGPU->getChunk(chunkId).getDeviceCells(iLayer), mTimeFrameGPU->getHostNCells(chunkId)[iLayer] * sizeof(o2::its::Cell), cudaMemcpyDeviceToHost));
            for (auto& cell : cells) {
              MCCompLabel currentLab{mTimeFrameGPU->getTrackletsLabel(iLayer)[cell.getFirstTrackletIndex()]};
              MCCompLabel nextLab{mTimeFrameGPU->getTrackletsLabel(iLayer + 1)[cell.getSecondTrackletIndex()]};
              mTimeFrameGPU->getCellsLabel(iLayer).emplace_back(currentLab == nextLab ? currentLab : MCCompLabel());
            }
          }
        }

        for (int iLayer{0}; iLayer < nLayers - 3; ++iLayer) {
          gpu::computeLayerCellNeighboursKernel<true><<<10, 1024, 0, mTimeFrameGPU->getStream(chunkId).get()>>>(
            mTimeFrameGPU->getChunk(chunkId).getDeviceCells(iLayer),
            mTimeFrameGPU->getChunk(chunkId).getDeviceCells(iLayer + 1),
            iLayer,
            mTimeFrameGPU->getChunk(chunkId).getDeviceCellsLookupTables(iLayer + 1),
            mTimeFrameGPU->getChunk(chunkId).getDeviceCellNeigboursLookupTables(iLayer),
            nullptr,
            mTimeFrameGPU->getChunk(chunkId).getDeviceNFoundCells(),
            mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->maxNeighboursSize);

          // Compute Cell Neighbours LUT
          checkGPUError(cub::DeviceScan::ExclusiveSum(mTimeFrameGPU->getChunk(chunkId).getDeviceCUBTmpBuffer(),                       // d_temp_storage
                                                      mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->tmpCUBBufferSize, // temp_storage_bytes
                                                      mTimeFrameGPU->getChunk(chunkId).getDeviceCellNeigboursLookupTables(iLayer),    // d_in
                                                      mTimeFrameGPU->getChunk(chunkId).getDeviceCellNeigboursLookupTables(iLayer),    // d_out
                                                      mTimeFrameGPU->getHostNCells(chunkId)[iLayer + 1],                              // num_items
                                                      mTimeFrameGPU->getStream(chunkId).get()));

          gpu::computeLayerCellNeighboursKernel<false><<<10, 1024, 0, mTimeFrameGPU->getStream(chunkId).get()>>>(
            mTimeFrameGPU->getChunk(chunkId).getDeviceCells(iLayer),
            mTimeFrameGPU->getChunk(chunkId).getDeviceCells(iLayer + 1),
            iLayer,
            mTimeFrameGPU->getChunk(chunkId).getDeviceCellsLookupTables(iLayer + 1),
            mTimeFrameGPU->getChunk(chunkId).getDeviceCellNeigboursLookupTables(iLayer),
            mTimeFrameGPU->getChunk(chunkId).getDeviceCellNeighbours(iLayer),
            mTimeFrameGPU->getChunk(chunkId).getDeviceNFoundCells(),
            mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->maxNeighboursSize);
          // if (!chunkId) {
          //   gpu::printBufferLayerOnThread<<<1, 1, 0, mTimeFrameGPU->getStream(chunkId).get()>>>(iLayer,
          //                                                                                       mTimeFrameGPU->getChunk(chunkId).getDeviceCellNeighbours(iLayer),
          //                                                                                       mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->maxNeighboursSize * rofs);
          // }
        }

        // End of tracking for this chunk
        offset += rofs;
      }
    };
    threads[chunkId] = std::thread(doTrackReconstruction);
  }
  for (auto& thread : threads) {
    thread.join();
  }

  mTimeFrameGPU->wipe(nLayers);
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::computeLayerCells(const int iteration)
{
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

template <int nLayers>
void TrackerTraitsGPU<nLayers>::findCellsNeighbours(const int iteration){};

template <int nLayers>
void TrackerTraitsGPU<nLayers>::findRoads(const int iteration){};

template <int nLayers>
void TrackerTraitsGPU<nLayers>::findTracks(){};

template <int nLayers>
void TrackerTraitsGPU<nLayers>::extendTracks(const int iteration){};

template <int nLayers>
void TrackerTraitsGPU<nLayers>::setBz(float bz)
{
  mBz = bz;
  mTimeFrameGPU->setBz(bz);
}

template <int nLayers>
int TrackerTraitsGPU<nLayers>::getTFNumberOfClusters() const
{
  return mTimeFrameGPU->getNumberOfClusters();
}

template <int nLayers>
int TrackerTraitsGPU<nLayers>::getTFNumberOfTracklets() const
{
  return mTimeFrameGPU->getNumberOfTracklets();
}

template <int nLayers>
int TrackerTraitsGPU<nLayers>::getTFNumberOfCells() const
{
  return mTimeFrameGPU->getNumberOfCells();
}

template class TrackerTraitsGPU<7>;
} // namespace its
} // namespace o2
