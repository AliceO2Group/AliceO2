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

#include <cuda_runtime.h>
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
#include "DataFormatsITS/TrackITS.h"

#include "ITStrackingGPU/TrackerTraitsGPU.h"
#include "ITStrackingGPU/TrackingKernels.h"

#ifndef __HIPCC__
#define THRUST_NAMESPACE thrust::cuda
#else
#define THRUST_NAMESPACE thrust::hip
#endif

// O2 track model
#include "ReconstructionDataFormats/Track.h"
#include "DetectorsBase/Propagator.h"
using namespace o2::track;

#define gpuCheckError(x)                \
  {                                     \
    gpuAssert((x), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
  if (code != cudaSuccess) {
    LOGF(error, "GPUassert: %s %s %d", cudaGetErrorString(code), file, line);
    if (abort) {
      throw std::runtime_error("GPU assert failed.");
    }
  }
}

namespace o2::its

{
using namespace constants::its2;

namespace gpu
{
GPUd() bool fitTrack(TrackITSExt& track,
                     int start,
                     int end,
                     int step,
                     float chi2clcut,
                     float chi2ndfcut,
                     float maxQoverPt,
                     int nCl,
                     float Bz,
                     const TrackingFrameInfo** tfInfos,
                     const o2::base::Propagator* prop,
                     o2::base::PropagatorF::MatCorrType matCorrType)
{
  for (int iLayer{start}; iLayer != end; iLayer += step) {
    if (track.getClusterIndex(iLayer) == constants::its::UnusedIndex) {
      continue;
    }
    const TrackingFrameInfo& trackingHit = tfInfos[iLayer][track.getClusterIndex(iLayer)];
    if (!track.o2::track::TrackParCovF::rotate(trackingHit.alphaTrackingFrame)) {
      return false;
    }

    if (!prop->propagateToX(track,
                            trackingHit.xTrackingFrame,
                            Bz,
                            o2::base::PropagatorImpl<float>::MAX_SIN_PHI,
                            o2::base::PropagatorImpl<float>::MAX_STEP,
                            matCorrType)) {
      return false;
    }

    if (matCorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE) {
      const float xx0 = (iLayer > 2) ? 1.e-2f : 5.e-3f; // Rough layer thickness
      constexpr float radiationLength = 9.36f;          // Radiation length of Si [cm]
      constexpr float density = 2.33f;                  // Density of Si [g/cm^3]
      if (!track.correctForMaterial(xx0, xx0 * radiationLength * density, true)) {
        return false;
      }
    }

    auto predChi2{track.getPredictedChi2(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)};

    if ((nCl >= 3 && predChi2 > chi2clcut) || predChi2 < 0.f) {
      return false;
    }
    track.setChi2(track.getChi2() + predChi2);
    if (!track.o2::track::TrackParCov::update(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)) {
      return false;
    }
    nCl++;
  }
  return o2::gpu::GPUCommonMath::Abs(track.getQ2Pt()) < maxQoverPt && track.getChi2() < chi2ndfcut * (nCl * 2 - 5);
}

template <int nLayers>
GPUg() void fitTrackSeedsKernel(
  CellSeed* trackSeeds,
  const TrackingFrameInfo** foundTrackingFrameInfo,
  o2::its::TrackITSExt* tracks,
  const unsigned int nSeeds,
  const float Bz,
  const int startLevel,
  float maxChi2ClusterAttachment,
  float maxChi2NDF,
  const o2::base::Propagator* propagator,
  const o2::base::PropagatorF::MatCorrType matCorrType)
{
  for (int iCurrentTrackSeedIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentTrackSeedIndex < nSeeds; iCurrentTrackSeedIndex += blockDim.x * gridDim.x) {
    auto& seed = trackSeeds[iCurrentTrackSeedIndex];

    TrackITSExt temporaryTrack{seed};

    temporaryTrack.resetCovariance();
    temporaryTrack.setChi2(0);
    int* clusters = seed.getClusters();
    for (int iL{0}; iL < 7; ++iL) {
      temporaryTrack.setExternalClusterIndex(iL, clusters[iL], clusters[iL] != constants::its::UnusedIndex);
    }
    bool fitSuccess = fitTrack(temporaryTrack,               // TrackITSExt& track,
                               0,                            // int lastLayer,
                               nLayers,                      // int firstLayer,
                               1,                            // int firstCluster,
                               maxChi2ClusterAttachment,     // float maxChi2ClusterAttachment,
                               maxChi2NDF,                   // float maxChi2NDF,
                               o2::constants::math::VeryBig, // float maxQoverPt,
                               0,                            // nCl,
                               Bz,                           // float Bz,
                               foundTrackingFrameInfo,       // TrackingFrameInfo** trackingFrameInfo,
                               propagator,                   // const o2::base::Propagator* propagator,
                               matCorrType);                 // o2::base::PropagatorF::MatCorrType matCorrType
    if (!fitSuccess) {
      continue;
    }
    temporaryTrack.getParamOut() = temporaryTrack.getParamIn();
    temporaryTrack.resetCovariance();
    temporaryTrack.setChi2(0);

    fitSuccess = fitTrack(temporaryTrack,           // TrackITSExt& track,
                          nLayers - 1,              // int lastLayer,
                          -1,                       // int firstLayer,
                          -1,                       // int firstCluster,
                          maxChi2ClusterAttachment, // float maxChi2ClusterAttachment,
                          maxChi2NDF,               // float maxChi2NDF,
                          50.f,                     // float maxQoverPt,
                          0,                        // nCl,
                          Bz,                       // float Bz,
                          foundTrackingFrameInfo,   // TrackingFrameInfo** trackingFrameInfo,
                          propagator,               // const o2::base::Propagator* propagator,
                          matCorrType);             // o2::base::PropagatorF::MatCorrType matCorrType
    if (!fitSuccess) {
      continue;
    }
    tracks[iCurrentTrackSeedIndex] = temporaryTrack;
  }
}

template <bool initRun, int nLayers = 7> // Version for new tracker to supersede the old one
GPUg() void computeLayerCellNeighboursKernel(
  CellSeed* cellsCurrentLayer,
  CellSeed* cellsNextLayer,
  int* neighboursLUT,
  const int* cellsNextLayerLUT,
  gpuPair<int, int>* cellNeighbours,
  const float maxChi2ClusterAttachment,
  const float bz,
  const int layerIndex,
  const int* nCells,
  const int maxCellNeighbours = 1e2)
{
  for (int iCurrentCellIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentCellIndex < nCells[layerIndex]; iCurrentCellIndex += blockDim.x * gridDim.x) {
    const auto& currentCellSeed{cellsCurrentLayer[iCurrentCellIndex]};
    const int nextLayerTrackletIndex{currentCellSeed.getSecondTrackletIndex()};
    const int nextLayerFirstCellIndex{cellsNextLayerLUT[nextLayerTrackletIndex]};
    const int nextLayerLastCellIndex{cellsNextLayerLUT[nextLayerTrackletIndex + 1]};
    int foundNeighbours{0};
    for (int iNextCell{nextLayerFirstCellIndex}; iNextCell < nextLayerLastCellIndex; ++iNextCell) {
      CellSeed nextCellSeed{cellsNextLayer[iNextCell]};                     // Copy
      if (nextCellSeed.getFirstTrackletIndex() != nextLayerTrackletIndex) { // Check if cells share the same tracklet
        break;
      }
      if (!nextCellSeed.rotate(currentCellSeed.getAlpha()) ||
          !nextCellSeed.propagateTo(currentCellSeed.getX(), bz)) {
        continue;
      }
      float chi2 = currentCellSeed.getPredictedChi2(nextCellSeed);
      if (chi2 > maxChi2ClusterAttachment) /// TODO: switch to the chi2 wrt cluster to avoid correlation
      {
        continue;
      }
      if constexpr (initRun) {
        atomicAdd(neighboursLUT + iNextCell, 1);
      } else {
        if (foundNeighbours >= maxCellNeighbours) {
          printf("its-gpu-neighbours-finder: data loss on layer: %d: number of neightbours exceeded the threshold!\n");
          continue;
        }
        cellNeighbours[neighboursLUT[iNextCell] + foundNeighbours++] = {iCurrentCellIndex, iNextCell};

        // FIXME: this is prone to race conditions: check on level is not atomic
        const int currentCellLevel{currentCellSeed.getLevel()};
        if (currentCellLevel >= nextCellSeed.getLevel()) {
          atomicExch(cellsNextLayer[iNextCell].getLevelPtr(), currentCellLevel + 1); // Update level on corresponding cell
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Legacy Kernels, to possibly take inspiration from
////////////////////////////////////////////////////////////////////////////////
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

// Functors to sort tracklets
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

// Print layer buffer
GPUg() void printBufferLayerOnThread(const int layer, const int* v, unsigned int size, const int len = 150, const unsigned int tId = 0)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == tId) {
    for (int i{0}; i < size; ++i) {
      if (!(i % len)) {
        printf("\n layer %d: ===> %d/%d\t", layer, i, (int)size);
      }
      printf("%d\t", v[i]);
    }
    printf("\n");
  }
}

// Dump vertices
GPUg() void printVertices(const Vertex* v, unsigned int size, const unsigned int tId = 0)
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
  const short rof0,
  const short maxRofs,
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
    short minRof = (rof0 >= trkPars->DeltaROF) ? rof0 - trkPars->DeltaROF : 0;
    short maxRof = (rof0 == static_cast<short>(maxRofs - trkPars->DeltaROF)) ? rof0 : rof0 + trkPars->DeltaROF;
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
      const float sigmaZ{o2::gpu::CAMath::Sqrt(Sq(resolution) * Sq(tanLambda) * ((Sq(inverseR0) + sqInverseDeltaZ0) * Sq(meanDeltaR) + 1.f) + Sq(meanDeltaR * mSAngle))};

      const int4 selectedBinsRect{getBinsRect(currentCluster, layerIndex, *utils, zAtRmin, zAtRmax, sigmaZ * trkPars->NSigmaCut, phiCut)};
      if (selectedBinsRect.x == 0 && selectedBinsRect.y == 0 && selectedBinsRect.z == 0 && selectedBinsRect.w == 0) {
        continue;
      }
      int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};
      if (phiBinsNum < 0) {
        phiBinsNum += trkPars->PhiBins;
      }
      constexpr int tableSize{256 * 128 + 1}; // hardcoded for the time being

      for (short rof1{minRof}; rof1 <= maxRof; ++rof1) {
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
              const unsigned int stride{currentClusterIndex * maxTrackletsPerCluster};
              new (tracklets + stride + storedTracklets) Tracklet{currentSortedIndex, roFrameClustersNext[rof1] + iNextCluster, tanL, phi, rof0, rof1};
              ++storedTracklets;
            }
          }
        }
      }
    }
    // if (storedTracklets > maxTrackletsPerCluster) {
    //   printf("its-gpu-tracklet finder: found more tracklets per clusters (%d) than maximum set (%d), check the configuration!\n", maxTrackletsPerCluster, storedTracklets);
    // }
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
    auto nClustersCurrentLayerRof = o2::gpu::GPUCommonMath::Min(roFrameClustersCurrentLayer[rof0 + 1] - roFrameClustersCurrentLayer[rof0], (int)maxClustersPerRof);
    // if (nClustersCurrentLayerRof > maxClustersPerRof) {
    //   printf("its-gpu-tracklet finder: on layer %d found more clusters per ROF (%d) than maximum set (%d), check the configuration!\n", layerIndex, nClustersCurrentLayerRof, maxClustersPerRof);
    // }
    auto* clustersCurrentLayerRof = clustersCurrentLayer + (roFrameClustersCurrentLayer[rof0] - roFrameClustersCurrentLayer[startRofId]);
    auto nVerticesRof0 = nVertices[rof0 + 1] - nVertices[rof0];
    auto trackletsRof0 = tracklets + maxTrackletsPerCluster * maxClustersPerRof * iRof;
    for (int currentClusterIndex = threadIdx.x; currentClusterIndex < nClustersCurrentLayerRof; currentClusterIndex += blockDim.x) {
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
        const float sigmaZ{o2::gpu::CAMath::Sqrt(Sq(resolution) * Sq(tanLambda) * ((Sq(inverseR0) + sqInverseDeltaZ0) * Sq(meanDeltaR) + 1.f) + Sq(meanDeltaR * mSAngle))};

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
                const unsigned int stride{currentClusterIndex * maxTrackletsPerCluster};
                if (storedTracklets < maxTrackletsPerCluster) {
                  new (trackletsRof0 + stride + storedTracklets) Tracklet{currentSortedIndexChunk, nextClusterIndex, tanL, phi, static_cast<short>(rof0), static_cast<short>(rof1)};
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
  CellSeed* cells,
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

      if (deltaTanLambda / trkPars->CellDeltaTanLambdaSigma < trkPars->NSigmaCut) {
        if constexpr (!initRun) {
          new (cells + cellsLUT[iCurrentTrackletIndex] + foundCells) Cell{currentTracklet.firstClusterIndex, nextTracklet.firstClusterIndex,
                                                                          nextTracklet.secondClusterIndex,
                                                                          iCurrentTrackletIndex,
                                                                          iNextTrackletIndex};
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

template <bool dryRun, int nLayers = 7>
GPUg() void computeLayerRoadsKernel(
  const int level,
  const int layerIndex,
  CellSeed** cells,
  const int* nCells,
  int** neighbours,
  int** neighboursLUT,
  Road<nLayers - 2>* roads,
  int* roadsLookupTable)
{
  for (int iCurrentCellIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentCellIndex < nCells[layerIndex]; iCurrentCellIndex += blockDim.x * gridDim.x) {
    auto& currentCell{cells[layerIndex][iCurrentCellIndex]};
    if (currentCell.getLevel() != level) {
      continue;
    }
    int nRoadsCurrentCell{0};
    if constexpr (dryRun) {
      roadsLookupTable[iCurrentCellIndex]++;
    } else {
      roads[roadsLookupTable[iCurrentCellIndex] + nRoadsCurrentCell++] = Road<nLayers - 2>{layerIndex, iCurrentCellIndex};
    }
    if (level == 1) {
      continue;
    }

    const auto currentCellNeighOffset{neighboursLUT[layerIndex - 1][iCurrentCellIndex]};
    const int cellNeighboursNum{neighboursLUT[layerIndex - 1][iCurrentCellIndex + 1] - currentCellNeighOffset};
    bool isFirstValidNeighbour{true};
    for (int iNeighbourCell{0}; iNeighbourCell < cellNeighboursNum; ++iNeighbourCell) {
      const int neighbourCellId = neighbours[layerIndex - 1][currentCellNeighOffset + iNeighbourCell];
      const CellSeed& neighbourCell = cells[layerIndex - 1][neighbourCellId];
      if (level - 1 != neighbourCell.getLevel()) {
        continue;
      }
      if (isFirstValidNeighbour) {
        isFirstValidNeighbour = false;
      } else {
        if constexpr (dryRun) {
          roadsLookupTable[iCurrentCellIndex]++; // dry run we just count the number of roads
        } else {
          roads[roadsLookupTable[iCurrentCellIndex] + nRoadsCurrentCell++] = Road<nLayers - 2>{layerIndex, iCurrentCellIndex};
        }
      }
      // traverseCellsTreeDevice<dryRun>(neighbourCellId, layerIndex - 1, iCurrentCellIndex, nRoadsCurrentCell, roadsLookupTable, cells, roads);
    }
  }
}
} // namespace gpu

template <bool isInit>
void cellNeighboursHandler(CellSeed* cellsCurrentLayer,
                           CellSeed* cellsNextLayer,
                           int* neighboursLUT,
                           const int* cellsNextLayerLUT,
                           gpuPair<int, int>* cellNeighbours,
                           const float maxChi2ClusterAttachment,
                           const float bz,
                           const int layerIndex,
                           const int* nCells,
                           const int maxCellNeighbours = 1e2)
{
  gpu::computeLayerCellNeighboursKernel<isInit><<<20, 512>>>(
    cellsCurrentLayer,        // CellSeed* cellsCurrentLayer,
    cellsNextLayer,           // CellSeed* cellsNextLayer,
    neighboursLUT,            // int* neighboursLUT,
    cellsNextLayerLUT,        // const int* cellsNextLayerLUT,
    cellNeighbours,           // gpuPair<int, int>* cellNeighbours,
    maxChi2ClusterAttachment, // const float maxChi2ClusterAttachment,
    bz,                       // const float bz,
    layerIndex,               // const int layerIndex,
    nCells,                   // const int* nCells,
    maxCellNeighbours);       // const int maxCellNeighbours = 1e2
}

void trackSeedHandler(CellSeed* trackSeeds,
                      const TrackingFrameInfo** foundTrackingFrameInfo,
                      o2::its::TrackITSExt* tracks,
                      const unsigned int nSeeds,
                      const float Bz,
                      const int startLevel,
                      float maxChi2ClusterAttachment,
                      float maxChi2NDF,
                      const o2::base::Propagator* propagator,
                      const o2::base::PropagatorF::MatCorrType matCorrType)
{
  gpu::fitTrackSeedsKernel<<<20, 256>>>(
    trackSeeds,               // CellSeed* trackSeeds,
    foundTrackingFrameInfo,   // TrackingFrameInfo** foundTrackingFrameInfo,
    tracks,                   // o2::its::TrackITSExt* tracks,
    nSeeds,                   // const unsigned int nSeeds,
    Bz,                       // const float Bz,
    startLevel,               // const int startLevel,
    maxChi2ClusterAttachment, // float maxChi2ClusterAttachment,
    maxChi2NDF,               // float maxChi2NDF,
    propagator,               // const o2::base::Propagator* propagator
    matCorrType);             // o2::base::PropagatorF::MatCorrType matCorrType

  gpuCheckError(cudaPeekAtLastError());
  gpuCheckError(cudaDeviceSynchronize());
}
} // namespace o2::its
