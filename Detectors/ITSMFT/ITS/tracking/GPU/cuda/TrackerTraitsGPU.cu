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
#include "DetectorsBase/Propagator.h"
#include "DataFormatsITS/TrackITS.h"

#define GPUCA_TPC_GEOMETRY_O2 // To set working switch in GPUTPCGeometry whose else statement is bugged
#define GPUCA_O2_INTERFACE    // To suppress errors related to the weird dependency between itsgputracking and GPUTracking

#include "ITStrackingGPU/TrackerTraitsGPU.h"
#include "ITStrackingGPU/TracerGPU.h"

#include "GPUCommonLogger.h"
#include "GPUCommonAlgorithmThrust.h"

#ifndef __HIPCC__
#define THRUST_NAMESPACE thrust::cuda
// #include "GPUReconstructionCUDADef.h"
#else
#define THRUST_NAMESPACE thrust::hip
// clang-format off
// #ifndef GPUCA_NO_CONSTANT_MEMORY
//   #ifdef GPUCA_CONSTANT_AS_ARGUMENT
//     #define GPUCA_CONSMEM_PTR const GPUConstantMemCopyable gGPUConstantMemBufferByValue,
//     #define GPUCA_CONSMEM_CALL gGPUConstantMemBufferHost,
//     #define GPUCA_CONSMEM (const_cast<GPUConstantMem&>(gGPUConstantMemBufferByValue.v))
//   #else
//     #define GPUCA_CONSMEM_PTR
//     #define GPUCA_CONSMEM_CALL
//     #define GPUCA_CONSMEM (gGPUConstantMemBuffer.v)
//   #endif
// #else
//   #define GPUCA_CONSMEM_PTR const GPUConstantMem *gGPUConstantMemBuffer,
//   #define GPUCA_CONSMEM_CALL me->mDeviceConstantMem,
//   #define GPUCA_CONSMEM const_cast<GPUConstantMem&>(*gGPUConstantMemBuffer)
// #endif
// #define GPUCA_KRNL_BACKEND_CLASS GPUReconstructionHIPBackend
// clang-format on
#endif
// #include "GPUConstantMem.h"

// Files for propagation with material
#include "Ray.cxx"
#include "MatLayerCylSet.cxx"
#include "MatLayerCyl.cxx"

// O2 track model
#include "TrackParametrization.cxx"
#include "TrackParametrizationWithError.cxx"
// #include "Propagator.cxx"

namespace o2
{
namespace its
{
using gpu::utils::checkGPUError;
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

  return int4{o2::gpu::GPUCommonMath::Max(0, utils.getZBinIndex(layerIndex + 1, zRangeMin)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMin)),
              o2::gpu::GPUCommonMath::Min(ZBins - 1, utils.getZBinIndex(layerIndex + 1, zRangeMax)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMax))};
}

// template <bool dryRun, int nLayers = 7>
// GPUd() void traverseCellsTreeDevice(
//   const int currentCellId,
//   const int currentLayerId,
//   const int startingCellId, // used to compile LUT
//   int& nRoadsStartingCell,
//   int* roadsLookUpTable,
//   CellSeed** cells,
//   int** cellNeighbours,
//   int** cellNeighboursLUT,
//   Road<nLayers - 2>* roads)
// {
//   CellSeed& currentCell{cells[currentLayerId][currentCellId]};
//   const int currentCellLevel = currentCell.getLevel();
//   if constexpr (dryRun) {
//     ++nRoadsStartingCell; // in dry run I just want to count the total number of roads
//   } else {
//     // mTimeFrame->getRoads().back().addCell(currentLayerId, currentCellId);
//   }
//   if (currentLayerId > 0 && currentCellLevel > 1) {
//     const int cellNeighboursNum{cellNeighboursLUT[currentLayerId - 1][currentCellId + 1] - cellNeighboursLUT[currentLayerId - 1][currentCellId]}; // careful!
//     bool isFirstValidNeighbour{true};

//     for (int iNeighbourCell{0}; iNeighbourCell < cellNeighboursNum; ++iNeighbourCell) {
//       const int neighbourCellId =cellNeighbours[currentLayerId - 1][currentCellId][iNeighbourCell];
//         const CellSeed& neighbourCell = mTimeFrame->getCells()[currentLayerId - 1][neighbourCellId];

//       if (currentCellLevel - 1 != neighbourCell.getLevel()) {
//         continue;
//       }

//       if (isFirstValidNeighbour) {
//         isFirstValidNeighbour = false;
//       } else {
//         // mTimeFrame->getRoads().push_back(mTimeFrame->getRoads().back());
//       }
//       traverseCellsTree<dryRun>(neighbourCellId, currentLayerId - 1, cells, cellNeighbours, cellNeighboursLUT, roads);
//       ++nRoadsStartingCell;
//     }
//   }
// }

GPUhd() float Sq(float q)
{
  return q * q;
}

GPUd() bool fitTrack(TrackITSExt& track,
                     int start,
                     int end,
                     int step,
                     float chi2clcut,
                     float chi2ndfcut,
                     float maxQoverPt,
                     int nCl,
                     float Bz,
                     TrackingFrameInfo** tfInfos,
                     const o2::base::Propagator* prop,
                     o2::base::PropagatorF::MatCorrType matCorrType = o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrNONE)
{
  for (int iLayer{start}; iLayer != end; iLayer += step) {
    if (track.getClusterIndex(iLayer) == constants::its::UnusedIndex) {
      continue;
    }
    const TrackingFrameInfo& trackingHit = tfInfos[iLayer][track.getClusterIndex(iLayer)];
    if (!track.o2::track::TrackParCovF::rotate(trackingHit.alphaTrackingFrame)) {
      return false;
    }
    if (matCorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE) {
      if (!track.propagateTo(trackingHit.xTrackingFrame, Bz)) {
        return false;
      }
    } else {
      // FIXME
      // if (!prop->propagateToX(track, trackingHit.xTrackingFrame,
      //                         prop->getNominalBz(),
      //                         o2::base::PropagatorImpl<float>::MAX_SIN_PHI,
      //                         o2::base::PropagatorImpl<float>::MAX_STEP,
      //                         matCorrType)) {
      //   return false;
      // }
    }
    track.setChi2(track.getChi2() + track.getPredictedChi2Unchecked(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame));
    if (!track.TrackParCov::update(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)) {
      return false;
    }

    const float xx0 = (iLayer > 2) ? 0.008f : 0.003f; // Rough layer thickness
    constexpr float radiationLength = 9.36f;          // Radiation length of Si [cm]
    constexpr float density = 2.33f;                  // Density of Si [g/cm^3]
    if (!track.correctForMaterial(xx0, xx0 * radiationLength * density, true)) {
      return false;
    }

    auto predChi2{track.getPredictedChi2Unchecked(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)};
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
GPUg() void printBufferLayerOnThread(const int layer, const int* v, size_t size, const int len = 150, const unsigned int tId = 0)
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

template <bool initRun, int nLayers = 7>
GPUg() void computeLayerCellNeighboursKernel(CellSeed* cellsCurrentLayer,
                                             CellSeed* cellsNextLayer,
                                             const int layerIndex,
                                             const int* cellsNextLayerLUT,
                                             int* neighboursLUT,
                                             int* cellNeighbours,
                                             const int* nCells,
                                             const int maxCellNeighbours = 1e2)
{
  for (int iCurrentCellIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentCellIndex < nCells[layerIndex]; iCurrentCellIndex += blockDim.x * gridDim.x) {
    const CellSeed& currentCell = cellsCurrentLayer[iCurrentCellIndex];
    const int nextLayerTrackletIndex{currentCell.getSecondTrackletIndex()};
    const int nextLayerFirstCellIndex{cellsNextLayerLUT[nextLayerTrackletIndex]};
    const int nextLayerLastCellIndex{cellsNextLayerLUT[nextLayerTrackletIndex + 1]};
    int foundNeighbours{0};
    for (int iNextCell{nextLayerFirstCellIndex}; iNextCell < nextLayerLastCellIndex; ++iNextCell) {
      CellSeed& nextCell = cellsNextLayer[iNextCell];
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

template <int nLayers = 7>
GPUg() void fitTrackSeedsKernel(
  CellSeed* trackSeeds,
  TrackingFrameInfo** foundTrackingFrameInfo,
  o2::its::TrackITSExt* tracks,
  const size_t nSeeds,
  const float Bz,
  const int startLevel,
  float maxChi2ClusterAttachment,
  float maxChi2NDF,
  const o2::base::Propagator* propagator)
{
  for (int iCurrentTrackSeedIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentTrackSeedIndex < nSeeds; iCurrentTrackSeedIndex += blockDim.x * gridDim.x) {
    auto& seed = trackSeeds[iCurrentTrackSeedIndex];
    if (seed.getQ2Pt() > 1.e3 || seed.getChi2() > maxChi2NDF * ((startLevel + 2) * 2 - (nLayers - 2))) {
      continue;
    }
    TrackITSExt temporaryTrack{seed};
    temporaryTrack.resetCovariance();
    temporaryTrack.setChi2(0);
    int* clusters = seed.getClusters();
    for (int iL{0}; iL < 7; ++iL) {
      temporaryTrack.setExternalClusterIndex(iL, clusters[iL], clusters[iL] != constants::its::UnusedIndex);
    }
    bool fitSuccess = fitTrack(temporaryTrack,                                                // TrackITSExt& track,
                               0,                                                             // int lastLayer,
                               nLayers,                                                       // int firstLayer,
                               1,                                                             // int firstCluster,
                               maxChi2ClusterAttachment,                                      // float maxChi2ClusterAttachment,
                               maxChi2NDF,                                                    // float maxChi2NDF,
                               o2::constants::math::VeryBig,                                  // float maxQoverPt,
                               0,                                                             // nCl,
                               Bz,                                                            // float Bz,
                               foundTrackingFrameInfo,                                        // TrackingFrameInfo** trackingFrameInfo,
                               propagator,                                                    // const o2::base::Propagator* propagator,
                               o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrNONE); // o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrLUT
    if (!fitSuccess) {
      continue;
    }
    temporaryTrack.getParamOut() = temporaryTrack.getParamIn();
    temporaryTrack.resetCovariance();
    temporaryTrack.setChi2(0);

    fitSuccess = fitTrack(temporaryTrack,                                                // TrackITSExt& track,
                          nLayers - 1,                                                   // int lastLayer,
                          -1,                                                            // int firstLayer,
                          -1,                                                            // int firstCluster,
                          maxChi2ClusterAttachment,                                      // float maxChi2ClusterAttachment,
                          maxChi2NDF,                                                    // float maxChi2NDF,
                          50.f,                                                          // float maxQoverPt,
                          0,                                                             // nCl,
                          Bz,                                                            // float Bz,
                          foundTrackingFrameInfo,                                        // TrackingFrameInfo** trackingFrameInfo,
                          propagator,                                                    // const o2::base::Propagator* propagator,
                          o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrNONE); // o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrLUT
    if (!fitSuccess) {
      continue;
    }
    tracks[iCurrentTrackSeedIndex] = temporaryTrack;
  }
}

template <int nLayers = 7>
GPUg() void fitTracksKernel(
  Cluster** foundClusters,
  Cluster** foundUnsortedClusters,
  TrackingFrameInfo** foundTrackingFrameInfo,
  Tracklet** foundTracklets,
  CellSeed** foundCellsSeeds,
  o2::track::TrackParCovF** trackSeeds,
  float** trackSeedsChi2,
  const Road<nLayers - 2>* roads,
  o2::its::TrackITSExt* tracks,
  const size_t nRoads,
  const float Bz,
  float maxChi2ClusterAttachment,
  float maxChi2NDF,
  const o2::base::Propagator* propagator)
{
  for (int iCurrentRoadIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentRoadIndex < nRoads; iCurrentRoadIndex += blockDim.x * gridDim.x) {
    auto& currentRoad{roads[iCurrentRoadIndex]};
    int clusters[nLayers];
    int tracklets[nLayers - 1];
    memset(clusters, constants::its::UnusedIndex, sizeof(clusters));
    memset(tracklets, constants::its::UnusedIndex, sizeof(tracklets));
    int lastCellLevel{constants::its::UnusedIndex}, firstTracklet{constants::its::UnusedIndex}, lastCellIndex{constants::its::UnusedIndex};

    for (int iCell{0}; iCell < nLayers - 2; ++iCell) {
      const int cellIndex = currentRoad[iCell];
      if (cellIndex == constants::its::UnusedIndex) {
        continue;
      } else {
        if (firstTracklet == constants::its::UnusedIndex) {
          firstTracklet = iCell;
        }
        tracklets[iCell] = foundCellsSeeds[iCell][cellIndex].getFirstTrackletIndex();
        tracklets[iCell + 1] = foundCellsSeeds[iCell][cellIndex].getSecondTrackletIndex();
        clusters[iCell] = foundCellsSeeds[iCell][cellIndex].getFirstClusterIndex();
        clusters[iCell + 1] = foundCellsSeeds[iCell][cellIndex].getSecondClusterIndex();
        clusters[iCell + 2] = foundCellsSeeds[iCell][cellIndex].getThirdClusterIndex();
        lastCellLevel = iCell;
        lastCellIndex = cellIndex;
      }
    }

    int count{1};
    unsigned short rof{foundTracklets[firstTracklet][tracklets[firstTracklet]].rof[0]};

    for (int iT = firstTracklet; iT < nLayers - 1; ++iT) {
      if (tracklets[iT] == constants::its::UnusedIndex) {
        continue;
      }
      if (rof == foundTracklets[iT][tracklets[iT]].rof[1]) {
        count++;
      } else {
        if (count == 1) {
          rof = foundTracklets[iT][tracklets[iT]].rof[1];
        } else {
          count--;
        }
      }
    }
    if (lastCellLevel == constants::its::UnusedIndex) {
      continue;
    }
    for (size_t iC{0}; iC < nLayers; iC++) {
      if (clusters[iC] != constants::its::UnusedIndex) {
        clusters[iC] = foundClusters[iC][clusters[iC]].clusterId;
      }
    }

    TrackITSExt temporaryTrack{foundCellsSeeds[lastCellLevel][lastCellIndex]};
    temporaryTrack.setChi2(foundCellsSeeds[lastCellLevel][lastCellIndex].getChi2());
    for (size_t iC = 0; iC < nLayers; ++iC) {
      temporaryTrack.setExternalClusterIndex(iC, clusters[iC], clusters[iC] != constants::its::UnusedIndex);
    }
    bool fitSuccess = fitTrack(temporaryTrack,                                                // TrackITSExt& track,
                               lastCellLevel - 1,                                             // int start,
                               -1,                                                            // int end,
                               -1,                                                            // int step,
                               maxChi2ClusterAttachment,                                      // float maxChi2ClusterAttachment,
                               maxChi2NDF,                                                    // float maxChi2NDF,
                               1.e3,                                                          // float maxQoverPt,
                               3,                                                             // int nCl,
                               Bz,                                                            // float Bz,
                               foundTrackingFrameInfo,                                        // TrackingFrameInfo** trackingFrameInfo,
                               propagator,                                                    // const o2::base::Propagator* propagator,
                               o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrNONE); // o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrLUT

    if (!fitSuccess) {
      continue;
    }
    temporaryTrack.resetCovariance();
    temporaryTrack.setChi2(0);
    fitSuccess = fitTrack(temporaryTrack,                                                // TrackITSExt& track,
                          0,                                                             // int lastLayer,
                          7,                                                             // int firstLayer,
                          1,                                                             // int firstCluster,
                          maxChi2ClusterAttachment,                                      // float maxChi2ClusterAttachment,
                          maxChi2NDF,                                                    // float maxChi2NDF,
                          o2::constants::math::VeryBig,                                  // float maxQoverPt,
                          0,                                                             // nCl,
                          Bz,                                                            // float Bz,
                          foundTrackingFrameInfo,                                        // TrackingFrameInfo** trackingFrameInfo,
                          propagator,                                                    // const o2::base::Propagator* propagator,
                          o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrNONE); // o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrLUT

    if (!fitSuccess) {
      continue;
    }
    temporaryTrack.getParamOut() = temporaryTrack;
    temporaryTrack.resetCovariance();
    temporaryTrack.setChi2(0);
    fitSuccess = fitTrack(temporaryTrack,                                                // TrackITSExt& track,
                          6 /* NL - 1 */,                                                // int lastLayer,
                          -1,                                                            // int firstLayer,
                          -1,                                                            // int firstCluster,
                          maxChi2ClusterAttachment,                                      // float maxChi2ClusterAttachment,
                          maxChi2NDF,                                                    // float maxChi2NDF,
                          50.,                                                           // float maxQoverPt,
                          0,                                                             // nCl,
                          Bz,                                                            // float Bz,
                          foundTrackingFrameInfo,                                        // TrackingFrameInfo** trackingFrameInfo,
                          propagator,                                                    // const o2::base::Propagator* propagator,
                          o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrNONE); // o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrLUT

    if (!fitSuccess) {
      continue;
    }
    tracks[iCurrentRoadIndex] = temporaryTrack;
  }
}

} // namespace gpu

template <int nLayers>
void TrackerTraitsGPU<nLayers>::initialiseTimeFrame(const int iteration)
{
  mTimeFrameGPU->initialiseHybrid(iteration, mTrkParams[iteration], nLayers);
  mTimeFrameGPU->loadClustersDevice();
  mTimeFrameGPU->loadUnsortedClustersDevice();
  mTimeFrameGPU->loadTrackingFrameInfoDevice();
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
        ////////////////////
        /// Tracklet finding

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

        ////////////////
        /// Cell finding
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

        // Create cells labels
        // TODO: make it work after fixing the tracklets labels
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

        /////////////////////
        /// Neighbour finding
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
        // Download cells into vectors

        for (int iLevel{nLayers - 2}; iLevel >= mTrkParams[iteration].CellMinimumLevel(); --iLevel) {
          const int minimumLevel{iLevel - 1};
          for (int iLayer{nLayers - 3}; iLayer >= minimumLevel; --iLayer) {
            // gpu::computeLayerRoadsKernel<true><<<1, 1, 0, mTimeFrameGPU->getStream(chunkId).get()>>>(iLevel,                                                               // const int level,
            //  iLayer,                                                               // const int layerIndex,
            //  mTimeFrameGPU->getChunk(chunkId).getDeviceArrayCells(),               // const CellSeed** cells,
            //  mTimeFrameGPU->getChunk(chunkId).getDeviceNFoundCells(),              // const int* nCells,
            //  mTimeFrameGPU->getChunk(chunkId).getDeviceArrayNeighboursCell(),      // const int** neighbours,
            //  mTimeFrameGPU->getChunk(chunkId).getDeviceArrayNeighboursCellLUT(),   // const int** neighboursLUT,
            //  mTimeFrameGPU->getChunk(chunkId).getDeviceRoads(),                    // Road* roads,
            //  mTimeFrameGPU->getChunk(chunkId).getDeviceRoadsLookupTables(iLayer)); // int* roadsLookupTable
          }
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

template <int nLayers>
void TrackerTraitsGPU<nLayers>::findCellsNeighbours(const int iteration)
{
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::findRoads(const int iteration)
{
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::findTracks()
{
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::extendTracks(const int iteration)
{
}

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

////////////////////////////////////////////////////////////////////////////////
// Hybrid tracking
template <int nLayers>
void TrackerTraitsGPU<nLayers>::computeTrackletsHybrid(const int iteration)
{
  TrackerTraits::computeLayerTracklets(iteration);
  mTimeFrameGPU->loadTrackletsDevice();
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::computeCellsHybrid(const int iteration)
{
  TrackerTraits::computeLayerCells(iteration);
  mTimeFrameGPU->loadCellsDevice();
};

template <int nLayers>
void TrackerTraitsGPU<nLayers>::findCellsNeighboursHybrid(const int iteration)
{
  TrackerTraits::findCellsNeighbours(iteration);
};

template <int nLayers>
void TrackerTraitsGPU<nLayers>::findRoadsHybrid(const int iteration)
{
  for (int startLevel{mTrkParams[iteration].CellsPerRoad()}; startLevel >= mTrkParams[iteration].CellMinimumLevel(); --startLevel) {
    const int minimumLayer{startLevel - 1};
    std::vector<CellSeed> trackSeeds;
    for (int startLayer{mTrkParams[iteration].CellsPerRoad() - 1}; startLayer >= minimumLayer; --startLayer) {
      std::vector<int> lastCellId, updatedCellId;
      std::vector<CellSeed> lastCellSeed, updatedCellSeed;

      processNeighbours(startLayer, startLevel, mTimeFrame->getCells()[startLayer], lastCellId, updatedCellSeed, updatedCellId);

      int level = startLevel;
      for (int iLayer{startLayer - 1}; iLayer > 0 && level > 2; --iLayer) {
        lastCellSeed.swap(updatedCellSeed);
        lastCellId.swap(updatedCellId);
        updatedCellSeed.clear();
        updatedCellId.clear();
        processNeighbours(iLayer, --level, lastCellSeed, lastCellId, updatedCellSeed, updatedCellId);
      }
      trackSeeds.insert(trackSeeds.end(), updatedCellSeed.begin(), updatedCellSeed.end());
    }
    mTimeFrameGPU->createTrackITSExtDevice(trackSeeds);
    mTimeFrameGPU->loadTrackSeedsDevice(trackSeeds);

    gpu::fitTrackSeedsKernel<<<20, 512>>>(
      mTimeFrameGPU->getDeviceTrackSeeds(),             // CellSeed* trackSeeds,
      mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(), // TrackingFrameInfo** foundTrackingFrameInfo,
      mTimeFrameGPU->getDeviceTrackITSExt(),            // o2::its::TrackITSExt* tracks,
      trackSeeds.size(),                                // const size_t nSeeds,
      mBz,                                              // const float Bz,
      startLevel,                                       // const int startLevel,
      mTrkParams[0].MaxChi2ClusterAttachment,           // float maxChi2ClusterAttachment,
      mTrkParams[0].MaxChi2NDF,                         // float maxChi2NDF,
      mTimeFrameGPU->getDevicePropagator());            // const o2::base::Propagator* propagator

    checkGPUError(cudaHostUnregister(trackSeeds.data()));
    mTimeFrameGPU->downloadTrackITSExtDevice();

    auto& tracks = mTimeFrameGPU->getTrackITSExt();
    std::sort(tracks.begin(), tracks.end(), [](const TrackITSExt& a, const TrackITSExt& b) {
      return a.getChi2() < b.getChi2();
    });

    for (auto& track : tracks) {
      int nShared = 0;
      bool isFirstShared{false};
      for (int iLayer{0}; iLayer < mTrkParams[0].NLayers; ++iLayer) {
        if (track.getClusterIndex(iLayer) == constants::its::UnusedIndex) {
          continue;
        }
        nShared += int(mTimeFrame->isClusterUsed(iLayer, track.getClusterIndex(iLayer)));
        isFirstShared |= !iLayer && mTimeFrame->isClusterUsed(iLayer, track.getClusterIndex(iLayer));
      }

      if (nShared > mTrkParams[0].ClusterSharing) {
        continue;
      }

      std::array<int, 3> rofs{INT_MAX, INT_MAX, INT_MAX};
      for (int iLayer{0}; iLayer < mTrkParams[0].NLayers; ++iLayer) {
        if (track.getClusterIndex(iLayer) == constants::its::UnusedIndex) {
          continue;
        }
        mTimeFrame->markUsedCluster(iLayer, track.getClusterIndex(iLayer));
        int currentROF = mTimeFrame->getClusterROF(iLayer, track.getClusterIndex(iLayer));
        for (int iR{0}; iR < 3; ++iR) {
          if (rofs[iR] == INT_MAX) {
            rofs[iR] = currentROF;
          }
          if (rofs[iR] == currentROF) {
            break;
          }
        }
      }
      if (rofs[2] != INT_MAX) {
        continue;
      }
      if (rofs[1] != INT_MAX) {
        track.setNextROFbit();
      }
      mTimeFrame->getTracks(std::min(rofs[0], rofs[1])).emplace_back(track);
    }
  }
};

template <int nLayers>
void TrackerTraitsGPU<nLayers>::findTracksHybrid(const int iteration)
{
  // LOGP(info, "propagator device pointer: {}", (void*)mTimeFrameGPU->getDevicePropagator());
  mTimeFrameGPU->createTrackITSExtDevice();
  gpu::fitTracksKernel<<<20, 512>>>(mTimeFrameGPU->getDeviceArrayClusters(),          // Cluster** foundClusters,
                                    mTimeFrameGPU->getDeviceArrayUnsortedClusters(),  // Cluster** foundUnsortedClusters,
                                    mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(), // TrackingFrameInfo** foundTrackingFrameInfo,
                                    mTimeFrameGPU->getDeviceArrayTracklets(),         // Tracklet** foundTracklets,
                                    mTimeFrameGPU->getDeviceArrayCells(),             // CellSeed** foundCells,
                                    mTimeFrameGPU->getDeviceArrayTrackSeeds(),        // o2::track::TrackParCovF** trackSeeds,
                                    mTimeFrameGPU->getDeviceArrayTrackSeedsChi2(),    // float** trackSeedsChi2,
                                    mTimeFrameGPU->getDeviceRoads(),                  // const Road<nLayers - 2>* roads,
                                    mTimeFrameGPU->getDeviceTrackITSExt(),            // o2::its::TrackITSExt* tracks,
                                    mTimeFrameGPU->getRoads().size(),                 // const size_t nRoads,
                                    mBz,                                              // const float Bz,
                                    mTrkParams[0].MaxChi2ClusterAttachment,           // float maxChi2ClusterAttachment,
                                    mTrkParams[0].MaxChi2NDF,                         // float maxChi2NDF,
                                    mTimeFrameGPU->getDevicePropagator());            // const o2::base::Propagator* propagator
  mTimeFrameGPU->downloadTrackITSExtDevice();
  discardResult(cudaDeviceSynchronize());
  auto& tracks = mTimeFrameGPU->getTrackITSExt();
  std::sort(tracks.begin(), tracks.end(),
            [](TrackITSExt& track1, TrackITSExt& track2) { return track1.isBetter(track2, 1.e6f); });
  for (auto& track : tracks) {
    if (!track.getNumberOfClusters()) {
      continue;
    }
    int nShared = 0;
    for (int iLayer{0}; iLayer < mTrkParams[0].NLayers; ++iLayer) {
      if (track.getClusterIndex(iLayer) == constants::its::UnusedIndex) {
        continue;
      }
      nShared += int(mTimeFrameGPU->isClusterUsed(iLayer, track.getClusterIndex(iLayer)));
    }

    if (nShared > mTrkParams[0].ClusterSharing) {
      continue;
    }

    std::array<int, 3> rofs{INT_MAX, INT_MAX, INT_MAX};
    for (int iLayer{0}; iLayer < mTrkParams[0].NLayers; ++iLayer) {
      if (track.getClusterIndex(iLayer) == constants::its::UnusedIndex) {
        continue;
      }
      mTimeFrameGPU->markUsedCluster(iLayer, track.getClusterIndex(iLayer));
      int currentROF = mTimeFrameGPU->getClusterROF(iLayer, track.getClusterIndex(iLayer));
      for (int iR{0}; iR < 3; ++iR) {
        if (rofs[iR] == INT_MAX) {
          rofs[iR] = currentROF;
        }
        if (rofs[iR] == currentROF) {
          break;
        }
      }
    }
    if (rofs[2] != INT_MAX) {
      continue;
    }
    if (rofs[1] != INT_MAX) {
      track.setNextROFbit();
    }
    mTimeFrameGPU->getTracks(std::min(rofs[0], rofs[1])).emplace_back(track);
  }
}

template class TrackerTraitsGPU<7>;
} // namespace its
} // namespace o2
