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
#include <unistd.h>

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/unique.h>
#include <thrust/remove.h>

#include "ITStracking/Constants.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/MathUtils.h"
#include "ITStracking/ExternalAllocator.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/Cluster.h"
#include "ITStracking/Cell.h"
#include "ITStracking/TrackHelpers.h"
#include "ITStracking/TrackFollower.h"
#include "DataFormatsITS/TrackITS.h"
#include "ITStrackingGPU/TrackingKernels.h"
#include "ITStrackingGPU/Utils.h"
#include "utils/strtag.h"

// O2 track model
#include "ReconstructionDataFormats/Track.h"
#include "DetectorsBase/Propagator.h"
using namespace o2::track;

namespace o2::its
{
namespace gpu
{

template <typename T1, typename T2>
struct sort_by_second {
  GPUhd() bool operator()(const gpuPair<T1, T2>& a, const gpuPair<T1, T2>& b) const { return a.second < b.second; }
};

template <typename T1, typename T2>
struct pair_to_first {
  GPUhd() int operator()(const gpuPair<T1, T2>& a) const
  {
    return a.first;
  }
};

template <typename T1, typename T2>
struct pair_to_second {
  GPUhd() int operator()(const gpuPair<T1, T2>& a) const
  {
    return a.second;
  }
};

template <typename T1, typename T2>
struct is_invalid_pair {
  GPUhd() bool operator()(const gpuPair<T1, T2>& p) const
  {
    return p.first == -1 && p.second == -1;
  }
};

template <typename T1, typename T2>
struct is_valid_pair {
  GPUhd() bool operator()(const gpuPair<T1, T2>& p) const
  {
    return !(p.first == -1 && p.second == -1);
  }
};

struct compare_track_index_chi2 {
  const TrackITSExt* tracks;

  GPUhd() bool operator()(const int a, const int b) const
  {
    return o2::its::track::isBetter(tracks[a], tracks[b]);
  }
};

template <int NLayers>
struct TrackExtensionDirectionFollowerDevice {
  GPUdi() bool operator()(TrackITSInternal<NLayers>& candidate, bool outward) const
  {
    const TrackExtensionHypothesis<NLayers> startHypothesis{candidate, outward};
    TrackExtensionHypothesis<NLayers> bestHypothesis;
    if (!followTrackExtensionDirection<NLayers>(startHypothesis, *fitCtx, *followCtx, outward,
                                                activeHypotheses, nextHypotheses, bestHypothesis)) {
      return false;
    }
    updateTrackFromExtensionHypothesis(bestHypothesis, outward, fitCtx->nLayers, candidate);
    return true;
  }

  const o2::its::track::TrackFitContext<NLayers>* fitCtx{nullptr};
  const TrackFollowContext<NLayers>* followCtx{nullptr};
  TrackExtensionHypothesis<NLayers>* activeHypotheses{nullptr};
  TrackExtensionHypothesis<NLayers>* nextHypotheses{nullptr};
};

template <int NLayers>
GPUg() void __launch_bounds__(constants::GPUThreads, 1) countTrackSeedsKernel(
  TrackSeed<NLayers>* trackSeeds,
  const TrackingFrameInfo** foundTrackingFrameInfo,
  const Cluster** unsortedClusters,
  int* seedLUT,
  const float* layerRadii,
  const float* minPts,
  const float* layerxX0,
  const unsigned int nSeeds,
  const float bz,
  const float maxChi2ClusterAttachment,
  const float maxChi2NDF,
  const int reseedIfShorter,
  const bool repeatRefitOut,
  const bool shiftRefToCluster,
  const o2::base::Propagator* propagator,
  const o2::base::PropagatorF::MatCorrType matCorrType)
{
  const o2::its::track::TrackFitContext<NLayers> fitCtx{
    foundTrackingFrameInfo, layerxX0, NLayers, bz,
    maxChi2ClusterAttachment, maxChi2NDF,
    propagator, matCorrType, shiftRefToCluster, repeatRefitOut};
  for (int iCurrentTrackSeedIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentTrackSeedIndex < nSeeds; iCurrentTrackSeedIndex += blockDim.x * gridDim.x) {
    TrackITSInternal<NLayers> temporaryTrack;
    if (o2::its::track::refitTrackSeed(trackSeeds[iCurrentTrackSeedIndex],
                                       temporaryTrack,
                                       fitCtx,
                                       unsortedClusters,
                                       layerRadii,
                                       minPts,
                                       reseedIfShorter)) {
      seedLUT[iCurrentTrackSeedIndex] = 1;
    }
  }
}

template <int NLayers>
GPUg() void __launch_bounds__(constants::GPUThreads, 1) fitTrackSeedsKernel(
  TrackSeed<NLayers>* trackSeeds,
  const TrackingFrameInfo** foundTrackingFrameInfo,
  const Cluster** unsortedClusters,
  const IndexTableUtils<NLayers>* utils,
  const typename ROFMaskTable<NLayers>::View rofMask,
  const typename ROFOverlapTable<NLayers>::View rofOverlaps,
  const Cluster** clusters,
  const unsigned char** usedClusters,
  const int** clustersIndexTables,
  const int** ROFClusters,
  o2::its::TrackITSExt* tracks,
  const int* seedLUT,
  TrackExtensionHypothesis<NLayers>* activeHypothesesScratch,
  TrackExtensionHypothesis<NLayers>* nextHypothesesScratch,
  const float* layerRadii,
  const float* minPts,
  const float* layerxX0,
  const unsigned int nSeeds,
  const float bz,
  const float maxChi2ClusterAttachment,
  const float maxChi2NDF,
  const int reseedIfShorter,
  const bool repeatRefitOut,
  const bool shiftRefToCluster,
  const int nLayers,
  const int phiBins,
  const int maxHypothesesConfig,
  const bool extendTop,
  const bool extendBot,
  const float nSigmaCutPhi,
  const float nSigmaCutZ,
  const o2::base::Propagator* propagator,
  const o2::base::PropagatorF::MatCorrType matCorrType)
{
  const o2::its::track::TrackFitContext<NLayers> fitCtx{
    foundTrackingFrameInfo, layerxX0, nLayers, bz,
    maxChi2ClusterAttachment, maxChi2NDF,
    propagator, matCorrType, shiftRefToCluster, repeatRefitOut};
  const TrackFollowContext<NLayers> followCtx{
    utils, rofMask, rofOverlaps,
    clusters, usedClusters, clustersIndexTables, ROFClusters,
    layerRadii, phiBins, maxHypothesesConfig, nSigmaCutPhi, nSigmaCutZ};
  for (int iCurrentTrackSeedIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentTrackSeedIndex < nSeeds; iCurrentTrackSeedIndex += blockDim.x * gridDim.x) {
    if (seedLUT[iCurrentTrackSeedIndex] == seedLUT[iCurrentTrackSeedIndex + 1]) {
      continue;
    }
    TrackITSInternal<NLayers> temporaryTrack;
    bool refitSuccess = o2::its::track::refitTrackSeed(trackSeeds[iCurrentTrackSeedIndex],
                                                       temporaryTrack,
                                                       fitCtx,
                                                       unsortedClusters,
                                                       layerRadii,
                                                       minPts,
                                                       reseedIfShorter);
    if (refitSuccess) {
      if ((extendTop || extendBot) && activeHypothesesScratch && nextHypothesesScratch) {
        const int maxHypotheses = o2::gpu::CAMath::Max(maxHypothesesConfig, 1);
        const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
        auto* activeHypotheses = activeHypothesesScratch + threadIndex * maxHypotheses;
        auto* nextHypotheses = nextHypothesesScratch + threadIndex * maxHypotheses;
        const auto backup = temporaryTrack;
        auto best = temporaryTrack;
        uint32_t bestDiff{0};
        TrackExtensionDirectionFollowerDevice<NLayers> followDirection{&fitCtx, &followCtx, activeHypotheses, nextHypotheses};
        TrackExtensionBestTrial<NLayers> bestTrial{backup.getPattern(), fitCtx};
        followTrackExtensionBranches(backup, extendTop, extendBot, nLayers, followDirection, bestTrial, best, bestDiff);
        temporaryTrack = best;
        tracks[seedLUT[iCurrentTrackSeedIndex]] = makeTrackITSExt(temporaryTrack);
        if (bestDiff) {
          tracks[seedLUT[iCurrentTrackSeedIndex]].setExtendedLayerPattern<NLayers>(bestDiff);
        }
        continue;
      }
      tracks[seedLUT[iCurrentTrackSeedIndex]] = makeTrackITSExt(temporaryTrack);
    }
  }
}

template <bool initRun, int NLayers>
GPUg() void __launch_bounds__(constants::GPUThreads, 1) computeLayerCellNeighboursKernel(
  CellSeed** cellSeedArray,
  int* neighboursCursor,
  int** cellsLUTs,
  CellNeighbour* cellNeighbours,
  const int sourceCellTopologyId,
  const int targetCellTopologyId,
  const float maxChi2ClusterAttachment,
  const float bz,
  const unsigned int nCells)
{
  for (int iCurrentCellIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentCellIndex < nCells; iCurrentCellIndex += blockDim.x * gridDim.x) {
    const auto& currentCellSeed{cellSeedArray[sourceCellTopologyId][iCurrentCellIndex]};
    const int nextLayerTrackletIndex{currentCellSeed.getSecondTrackletIndex()};
    const int nextLayerFirstCellIndex{cellsLUTs[targetCellTopologyId][nextLayerTrackletIndex]};
    const int nextLayerLastCellIndex{cellsLUTs[targetCellTopologyId][nextLayerTrackletIndex + 1]};
    for (int iNextCell{nextLayerFirstCellIndex}; iNextCell < nextLayerLastCellIndex; ++iNextCell) {
      auto nextCellSeed{cellSeedArray[targetCellTopologyId][iNextCell]}; // Copy
      if (nextCellSeed.getFirstTrackletIndex() != nextLayerTrackletIndex || !currentCellSeed.getTimeStamp().isCompatible(nextCellSeed.getTimeStamp())) {
        break;
      }

      if (!nextCellSeed.rotate(currentCellSeed.getAlpha()) ||
          !nextCellSeed.propagateTo(currentCellSeed.getX(), bz)) {
        continue;
      }

      float chi2 = currentCellSeed.getPredictedChi2(nextCellSeed);
      if (chi2 > maxChi2ClusterAttachment) {
        continue;
      }

      if constexpr (initRun) {
        atomicAdd(neighboursCursor + iNextCell, 1);
      } else {
        const int offset = atomicAdd(neighboursCursor + iNextCell, 1);
        cellNeighbours[offset] = {sourceCellTopologyId, iCurrentCellIndex, targetCellTopologyId, iNextCell, currentCellSeed.getLevel() + 1};
        const int currentCellLevel{currentCellSeed.getLevel()};
        if (currentCellLevel >= nextCellSeed.getLevel()) {
          atomicMax(cellSeedArray[targetCellTopologyId][iNextCell].getLevelPtr(), currentCellLevel + 1);
        }
      }
    }
  }
}

template <bool initRun, int NLayers>
GPUg() void __launch_bounds__(constants::GPUThreads, 1) computeLayerCellsKernel(
  const Cluster** sortedClusters,
  const Cluster** unsortedClusters,
  const TrackingFrameInfo** tfInfo,
  Tracklet** tracklets,
  int** trackletsLUT,
  const int nTrackletsCurrent,
  const int cellTopologyId,
  const typename TrackingTopology<NLayers>::View topology,
  CellSeed* cells,
  int** cellsLUTs,
  const float* layerxX0,
  const float bz,
  const float maxChi2ClusterAttachment,
  const float cellDeltaTanLambdaSigma,
  const float nSigmaCut)
{
  const auto cellTopology = topology.getCell(cellTopologyId);
  const auto first = topology.getLink(cellTopology.firstLink);
  const auto second = topology.getLink(cellTopology.secondLink);
  const int layers[3] = {first.fromLayer, first.toLayer, second.toLayer};
  for (int iCurrentTrackletIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentTrackletIndex < nTrackletsCurrent; iCurrentTrackletIndex += blockDim.x * gridDim.x) {
    if constexpr (!initRun) {
      if (cellsLUTs[cellTopologyId][iCurrentTrackletIndex] == cellsLUTs[cellTopologyId][iCurrentTrackletIndex + 1]) {
        continue;
      }
    }
    const Tracklet& currentTracklet = tracklets[cellTopology.firstLink][iCurrentTrackletIndex];
    const int nextLayerClusterIndex{currentTracklet.secondClusterIndex};
    const int nextLayerFirstTrackletIndex{trackletsLUT[cellTopology.secondLink][nextLayerClusterIndex]};
    const int nextLayerLastTrackletIndex{trackletsLUT[cellTopology.secondLink][nextLayerClusterIndex + 1]};
    if (nextLayerFirstTrackletIndex == nextLayerLastTrackletIndex) {
      continue;
    }
    int foundCells{0};
    for (int iNextTrackletIndex{nextLayerFirstTrackletIndex}; iNextTrackletIndex < nextLayerLastTrackletIndex; ++iNextTrackletIndex) {
      if (tracklets[cellTopology.secondLink][iNextTrackletIndex].firstClusterIndex != nextLayerClusterIndex) {
        break;
      }
      const Tracklet& nextTracklet = tracklets[cellTopology.secondLink][iNextTrackletIndex];
      if (!currentTracklet.getTimeStamp().isCompatible(nextTracklet.getTimeStamp())) {
        continue;
      }
      const float deltaTanLambda{o2::gpu::CAMath::Abs(currentTracklet.tanLambda - nextTracklet.tanLambda)};

      if (deltaTanLambda / cellDeltaTanLambdaSigma < nSigmaCut) {
        const int clusId[3]{
          sortedClusters[layers[0]][currentTracklet.firstClusterIndex].clusterId,
          sortedClusters[layers[1]][nextTracklet.firstClusterIndex].clusterId,
          sortedClusters[layers[2]][nextTracklet.secondClusterIndex].clusterId};

        const auto& cluster1_glo = unsortedClusters[layers[0]][clusId[0]];
        const auto& cluster2_glo = unsortedClusters[layers[1]][clusId[1]];
        const auto& cluster3_tf = tfInfo[layers[2]][clusId[2]];
        auto track{o2::its::track::buildTrackSeed(cluster1_glo, cluster2_glo, cluster3_tf, bz)};
        float chi2{0.f};
        bool good{false};
        for (int iC{2}; iC--;) {
          const TrackingFrameInfo& trackingHit = tfInfo[layers[iC]][clusId[iC]];
          if (!track.rotate(trackingHit.alphaTrackingFrame)) {
            break;
          }
          if (!track.propagateTo(trackingHit.xTrackingFrame, bz)) {
            break;
          }

          if (!track.correctForMaterial(layerxX0[layers[iC]], layerxX0[layers[iC]] * constants::Radl * constants::Rho, true)) {
            break;
          }

          const auto predChi2{track.getPredictedChi2Quiet(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)};
          if (!track.o2::track::TrackParCov::update(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)) {
            break;
          }
          if (!iC && predChi2 > maxChi2ClusterAttachment) {
            break;
          }
          good = !iC;
          chi2 += predChi2;
        }
        if (!good) {
          continue;
        }
        if constexpr (!initRun) {
          TimeEstBC ts = currentTracklet.getTimeStamp();
          ts += nextTracklet.getTimeStamp();
          new (cells + cellsLUTs[cellTopologyId][iCurrentTrackletIndex] + foundCells) CellSeed{cellTopology.hitLayerMask, clusId[0], clusId[1], clusId[2], iCurrentTrackletIndex, iNextTrackletIndex, track, chi2, ts};
        }
        ++foundCells;
      }
    }
    if constexpr (initRun) {
      cellsLUTs[cellTopologyId][iCurrentTrackletIndex] = foundCells;
    }
  }
}

template <bool initRun, int NLayers>
GPUg() void __launch_bounds__(constants::GPUThreads, 1) computeLayerTrackletsMultiROFKernel(
  const IndexTableUtils<NLayers>* utils,
  const typename ROFMaskTable<NLayers>::View rofMask,
  const int linkId,
  const typename TrackingTopology<NLayers>::View topology,
  const typename ROFOverlapTable<NLayers>::View rofOverlaps,
  const typename ROFVertexLookupTable<NLayers>::View vertexLUT,
  const Vertex* vertices,
  const int* rofPV,
  const int vertexId,
  const Cluster** clusters,
  const int** ROFClusters,
  const unsigned char** usedClusters,
  const int** indexTables,
  Tracklet** tracklets,
  int** trackletsLUT,
  const bool selectUPCVertices,
  const float NSigmaCut,
  const float phiCut,
  const float resolutionPV,
  const float minR,
  const float maxR,
  const float positionResolution,
  const float meanDeltaR,
  const float MSAngle)
{
  const auto link = topology.getLink(linkId);
  const int fromLayer = link.fromLayer;
  const int toLayer = link.toLayer;
  const int phiBins{utils->getNphiBins()};
  const int zBins{utils->getNzBins()};
  const int tableSize{phiBins * zBins + 1};
  const int totalROFs0 = rofOverlaps.getLayer(fromLayer).mNROFsTF;
  const int totalROFs1 = rofOverlaps.getLayer(toLayer).mNROFsTF;
  for (unsigned int pivotROF{blockIdx.x}; pivotROF < totalROFs0; pivotROF += gridDim.x) {
    if (!rofMask.isROFEnabled(fromLayer, pivotROF)) {
      continue;
    }

    const auto& pvs = vertexLUT.getVertices(fromLayer, pivotROF);
    auto primaryVertices = gpuSpan<const Vertex>(&vertices[pvs.getFirstEntry()], pvs.getEntries());
    if (primaryVertices.empty()) {
      continue;
    }
    const auto startVtx{vertexId >= 0 ? vertexId : 0};
    const auto endVtx{vertexId >= 0 ? o2::gpu::CAMath::Min(vertexId + 1, static_cast<int>(primaryVertices.size())) : static_cast<int>(primaryVertices.size())};
    if (endVtx <= startVtx || (vertexId + 1) > primaryVertices.size()) {
      continue;
    }

    const auto& rofOverlap = rofOverlaps.getOverlap(fromLayer, toLayer, pivotROF);
    if (!rofOverlap.getEntries()) {
      continue;
    }

    auto clustersCurrentLayer = getClustersOnLayer(pivotROF, totalROFs0, fromLayer, ROFClusters, clusters);
    if (clustersCurrentLayer.empty()) {
      continue;
    }

    for (int currentClusterIndex = threadIdx.x; currentClusterIndex < clustersCurrentLayer.size(); currentClusterIndex += blockDim.x) {

      unsigned int storedTracklets{0};
      const auto& currentCluster{clustersCurrentLayer[currentClusterIndex]};
      const int currentSortedIndex{ROFClusters[fromLayer][pivotROF] + currentClusterIndex};
      if (usedClusters[fromLayer][currentCluster.clusterId]) {
        continue;
      }
      if constexpr (!initRun) {
        if (trackletsLUT[linkId][currentSortedIndex] == trackletsLUT[linkId][currentSortedIndex + 1]) {
          continue;
        }
      }

      const float inverseR0{1.f / currentCluster.radius};
      for (int iV{startVtx}; iV < endVtx; ++iV) {
        auto& primaryVertex{primaryVertices[iV]};
        if (!vertexLUT.isVertexCompatible(fromLayer, pivotROF, primaryVertex)) {
          continue;
        }
        if (primaryVertex.isFlagSet(Vertex::Flags::UPCMode) != selectUPCVertices) {
          continue;
        }

        const float resolution = o2::gpu::CAMath::Sqrt(math_utils::Sq(resolutionPV) / primaryVertex.getNContributors() + math_utils::Sq(positionResolution));
        const float tanLambda{(currentCluster.zCoordinate - primaryVertex.getZ()) * inverseR0};
        const float zAtRmin{tanLambda * (minR - currentCluster.radius) + currentCluster.zCoordinate};
        const float zAtRmax{tanLambda * (maxR - currentCluster.radius) + currentCluster.zCoordinate};
        const float sqInverseDeltaZ0{1.f / (math_utils::Sq(currentCluster.zCoordinate - primaryVertex.getZ()) + constants::Tolerance)}; /// protecting from overflows adding the detector resolution
        const float sigmaZ{o2::gpu::CAMath::Sqrt(math_utils::Sq(resolution) * math_utils::Sq(tanLambda) * ((math_utils::Sq(inverseR0) + sqInverseDeltaZ0) * math_utils::Sq(meanDeltaR) + 1.f) + math_utils::Sq(meanDeltaR * MSAngle))};
        const int4 selectedBinsRect{o2::its::getBinsRect(currentCluster, toLayer, zAtRmin, zAtRmax, sigmaZ * NSigmaCut, phiCut, *utils)};
        if (selectedBinsRect.x < 0) {
          continue;
        }
        int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};

        if (phiBinsNum < 0) {
          phiBinsNum += phiBins;
        }

        for (short targetROF = rofOverlap.getFirstEntry(); targetROF < rofOverlap.getEntriesBound(); ++targetROF) {
          if (!rofMask.isROFEnabled(toLayer, targetROF)) {
            continue;
          }
          auto clustersNextLayer = getClustersOnLayer(targetROF, totalROFs1, toLayer, ROFClusters, clusters);
          if (clustersNextLayer.empty()) {
            continue;
          }
          const auto ts = rofOverlaps.getTimeStamp(fromLayer, pivotROF, toLayer, targetROF);
          if (!ts.isCompatible(primaryVertex.getTimeStamp())) {
            continue;
          }
          for (int iPhiCount{0}; iPhiCount < phiBinsNum; iPhiCount++) {
            int iPhiBin = (selectedBinsRect.y + iPhiCount) % phiBins;
            const int firstBinIndex{utils->getBinIndex(selectedBinsRect.x, iPhiBin)};
            const int maxBinIndex{firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1};
            const int firstRowClusterIndex = indexTables[toLayer][(targetROF)*tableSize + firstBinIndex];
            const int maxRowClusterIndex = indexTables[toLayer][(targetROF)*tableSize + maxBinIndex];
            for (int nextClusterIndex{firstRowClusterIndex}; nextClusterIndex < maxRowClusterIndex; ++nextClusterIndex) {
              if (nextClusterIndex >= clustersNextLayer.size()) {
                break;
              }
              const Cluster& nextCluster{clustersNextLayer[nextClusterIndex]};
              if (usedClusters[toLayer][nextCluster.clusterId]) {
                continue;
              }
              const float deltaPhi{o2::gpu::CAMath::Abs(currentCluster.phi - nextCluster.phi)};
              const float deltaZ{o2::gpu::CAMath::Abs(tanLambda * (nextCluster.radius - currentCluster.radius) + currentCluster.zCoordinate - nextCluster.zCoordinate)};
              if (deltaZ / sigmaZ < NSigmaCut && (deltaPhi < phiCut || o2::gpu::CAMath::Abs(deltaPhi - o2::constants::math::TwoPI) < phiCut)) {
                if constexpr (initRun) {
                  trackletsLUT[linkId][currentSortedIndex]++; // we need l0 as well for usual exclusive sums.
                } else {
                  const float phi{o2::gpu::CAMath::ATan2(currentCluster.yCoordinate - nextCluster.yCoordinate, currentCluster.xCoordinate - nextCluster.xCoordinate)};
                  const float tanL{(currentCluster.zCoordinate - nextCluster.zCoordinate) / (currentCluster.radius - nextCluster.radius)};
                  const int nextSortedIndex{ROFClusters[toLayer][targetROF] + nextClusterIndex};
                  new (tracklets[linkId] + trackletsLUT[linkId][currentSortedIndex] + storedTracklets) Tracklet{currentSortedIndex, nextSortedIndex, tanL, phi, ts};
                }
                ++storedTracklets;
              }
            }
          }
        }
      }
    }
  }
}

GPUg() void __launch_bounds__(constants::GPUThreads, 1) compileTrackletsLookupTableKernel(
  const Tracklet* tracklets,
  int* trackletsLookUpTable,
  const int nTracklets)
{
  for (int currentTrackletIndex = blockIdx.x * blockDim.x + threadIdx.x; currentTrackletIndex < nTracklets; currentTrackletIndex += blockDim.x * gridDim.x) {
    atomicAdd(&trackletsLookUpTable[tracklets[currentTrackletIndex].firstClusterIndex], 1);
  }
}

template <bool dryRun, int NLayers, typename CurrentSeed>
GPUg() void __launch_bounds__(constants::GPUThreads, 1) processNeighboursKernel(
  const int defaultCellTopologyId,
  const int level,
  CellSeed** allCellSeeds,
  CurrentSeed* currentCellSeeds,
  const int* currentCellIds,
  const int* currentCellTopologyIds,
  const unsigned int nCurrentCells,
  TrackSeed<NLayers>* updatedCellSeeds,
  int* updatedCellsIds,
  int* updatedCellTopologyIds,
  int* foundSeedsTable,               // auxiliary only in GPU code to compute the number of cells per iteration
  const unsigned char** usedClusters, // Used clusters
  CellNeighbour** neighbours,
  int** neighboursLUT,
  const TrackingFrameInfo** foundTrackingFrameInfo,
  const float* layerxX0,
  const float bz,
  const float maxChi2ClusterAttachment,
  const o2::base::Propagator* propagator,
  const o2::base::PropagatorF::MatCorrType matCorrType)
{
  for (unsigned int iCurrentCell = blockIdx.x * blockDim.x + threadIdx.x; iCurrentCell < nCurrentCells; iCurrentCell += blockDim.x * gridDim.x) {
    if constexpr (!dryRun) {
      if (foundSeedsTable[iCurrentCell] == foundSeedsTable[iCurrentCell + 1]) {
        continue;
      }
    }
    int foundSeeds{0};
    const auto& currentCell{currentCellSeeds[iCurrentCell]};
    const int cellTopologyId = currentCellTopologyIds == nullptr ? defaultCellTopologyId : currentCellTopologyIds[iCurrentCell];
    if (currentCell.getLevel() != level) {
      continue;
    }
    if (currentCellIds == nullptr) {
      bool used = false;
      for (int layer = 0; layer < NLayers; ++layer) {
        const int clusterIndex = currentCell.getCluster(layer);
        used |= clusterIndex != constants::UnusedIndex && usedClusters[layer][clusterIndex];
      }
      if (used) {
        continue;
      }
    }
    const int cellId = currentCellIds == nullptr ? iCurrentCell : currentCellIds[iCurrentCell];
    if (cellTopologyId < 0 || neighboursLUT[cellTopologyId] == nullptr || neighbours[cellTopologyId] == nullptr) {
      continue;
    }

    const int startNeighbourId{neighboursLUT[cellTopologyId][cellId]};
    const int endNeighbourId{neighboursLUT[cellTopologyId][cellId + 1]};

    for (int iNeighbourCell{startNeighbourId}; iNeighbourCell < endNeighbourId; ++iNeighbourCell) {
      const auto& neighbourRef = neighbours[cellTopologyId][iNeighbourCell];
      const int neighbourCellTopologyId = neighbourRef.cellTopology;
      const int neighbourCellId = neighbourRef.cell;
      const auto& neighbourCell = allCellSeeds[neighbourCellTopologyId][neighbourCellId];

      if (neighbourCell.getSecondTrackletIndex() != currentCell.getFirstTrackletIndex()) {
        continue;
      }
      if (!currentCell.getTimeStamp().isCompatible(neighbourCell.getTimeStamp())) {
        continue;
      }
      if (currentCell.getLevel() - 1 != neighbourCell.getLevel()) {
        continue;
      }
      const int neighbourLayer = neighbourCell.getInnerLayer();
      const int neighbourCluster = neighbourCell.getFirstClusterIndex();
      if (usedClusters[neighbourLayer][neighbourCluster]) {
        continue;
      }
      TrackSeed<NLayers> seed{currentCell};
      auto& trHit = foundTrackingFrameInfo[neighbourLayer][neighbourCluster];

      if (!seed.rotate(trHit.alphaTrackingFrame)) {
        continue;
      }

      if (!propagator->propagateToX(seed, trHit.xTrackingFrame, bz, o2::base::PropagatorImpl<float>::MAX_SIN_PHI, o2::base::PropagatorImpl<float>::MAX_STEP, matCorrType)) {
        continue;
      }

      if (matCorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE) {
        if (!seed.correctForMaterial(layerxX0[neighbourLayer], layerxX0[neighbourLayer] * constants::Radl * constants::Rho, true)) {
          continue;
        }
      }

      auto predChi2{seed.getPredictedChi2Quiet(trHit.positionTrackingFrame, trHit.covarianceTrackingFrame)};
      if ((predChi2 > maxChi2ClusterAttachment) || predChi2 < 0.f) {
        continue;
      }
      seed.setChi2(seed.getChi2() + predChi2);
      if (!seed.o2::track::TrackParCov::update(trHit.positionTrackingFrame, trHit.covarianceTrackingFrame)) {
        continue;
      }
      if constexpr (dryRun) {
        foundSeedsTable[iCurrentCell]++;
      } else {
        seed.getClusters()[neighbourLayer] = neighbourCluster;
        auto mask = seed.getHitLayerMask();
        mask.set(neighbourLayer);
        seed.setHitLayerMask(mask);
        seed.setLevel(neighbourCell.getLevel());
        seed.setFirstTrackletIndex(neighbourCell.getFirstTrackletIndex());
        seed.setSecondTrackletIndex(neighbourCell.getSecondTrackletIndex());
        updatedCellsIds[foundSeedsTable[iCurrentCell] + foundSeeds] = neighbourCellId;
        updatedCellTopologyIds[foundSeedsTable[iCurrentCell] + foundSeeds] = neighbourCellTopologyId;
        updatedCellSeeds[foundSeedsTable[iCurrentCell] + foundSeeds] = seed;
      }
      foundSeeds++;
    }
  }
}

} // namespace gpu

template <int NLayers>
void countTrackletsInROFsHandler(const IndexTableUtils<NLayers>* utils,
                                 const typename ROFMaskTable<NLayers>::View& rofMask,
                                 const int linkId,
                                 const int fromLayer,
                                 const int toLayer,
                                 const typename ROFOverlapTable<NLayers>::View& rofOverlaps,
                                 const typename ROFVertexLookupTable<NLayers>::View& vertexLUT,
                                 const int vertexId,
                                 const Vertex* vertices,
                                 const int* rofPV,
                                 const Cluster** clusters,
                                 std::vector<unsigned int> nClusters,
                                 const int** ROFClusters,
                                 const unsigned char** usedClusters,
                                 const int** clustersIndexTables,
                                 int** trackletsLUTs,
                                 gsl::span<int*> trackletsLUTsHost,
                                 const bool selectUPCVertices,
                                 const float NSigmaCut,
                                 const typename TrackingTopology<NLayers>::View topology,
                                 bounded_vector<float>& linkPhiCuts,
                                 const float resolutionPV,
                                 std::array<float, NLayers>& minRs,
                                 std::array<float, NLayers>& maxRs,
                                 bounded_vector<float>& resolutions,
                                 std::vector<float>& radii,
                                 bounded_vector<float>& linkMSAngles,
                                 o2::its::ExternalAllocator* alloc,
                                 gpu::Streams& streams)
{
  gpu::computeLayerTrackletsMultiROFKernel<true><<<constants::GPUBlocks, constants::GPUThreads, 0, streams[linkId].get()>>>(
    utils,
    rofMask,
    linkId,
    topology,
    rofOverlaps,
    vertexLUT,
    vertices,
    rofPV,
    vertexId,
    clusters,
    ROFClusters,
    usedClusters,
    clustersIndexTables,
    nullptr,
    trackletsLUTs,
    selectUPCVertices,
    NSigmaCut,
    linkPhiCuts[linkId],
    resolutionPV,
    minRs[toLayer],
    maxRs[toLayer],
    resolutions[fromLayer],
    radii[toLayer] - radii[fromLayer],
    linkMSAngles[linkId]);
  auto nosync_policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(streams[linkId].get());
  thrust::exclusive_scan(nosync_policy, trackletsLUTsHost[linkId], trackletsLUTsHost[linkId] + nClusters[fromLayer] + 1, trackletsLUTsHost[linkId]);
}

template <int NLayers>
void computeTrackletsInROFsHandler(const IndexTableUtils<NLayers>* utils,
                                   const typename ROFMaskTable<NLayers>::View& rofMask,
                                   const int linkId,
                                   const int fromLayer,
                                   const int toLayer,
                                   const typename ROFOverlapTable<NLayers>::View& rofOverlaps,
                                   const typename ROFVertexLookupTable<NLayers>::View& vertexLUT,
                                   const int vertexId,
                                   const Vertex* vertices,
                                   const int* rofPV,
                                   const Cluster** clusters,
                                   std::vector<unsigned int> nClusters,
                                   const int** ROFClusters,
                                   const unsigned char** usedClusters,
                                   const int** clustersIndexTables,
                                   Tracklet** tracklets,
                                   gsl::span<Tracklet*> spanTracklets,
                                   gsl::span<int> nTracklets,
                                   int** trackletsLUTs,
                                   gsl::span<int*> trackletsLUTsHost,
                                   const bool selectUPCVertices,
                                   const float NSigmaCut,
                                   const typename TrackingTopology<NLayers>::View topology,
                                   bounded_vector<float>& linkPhiCuts,
                                   const float resolutionPV,
                                   std::array<float, NLayers>& minRs,
                                   std::array<float, NLayers>& maxRs,
                                   bounded_vector<float>& resolutions,
                                   std::vector<float>& radii,
                                   bounded_vector<float>& linkMSAngles,
                                   o2::its::ExternalAllocator* alloc,
                                   gpu::Streams& streams)
{
  gpu::computeLayerTrackletsMultiROFKernel<false><<<constants::GPUBlocks, constants::GPUThreads, 0, streams[linkId].get()>>>(
    utils,
    rofMask,
    linkId,
    topology,
    rofOverlaps,
    vertexLUT,
    vertices,
    rofPV,
    vertexId,
    clusters,
    ROFClusters,
    usedClusters,
    clustersIndexTables,
    tracklets,
    trackletsLUTs,
    selectUPCVertices,
    NSigmaCut,
    linkPhiCuts[linkId],
    resolutionPV,
    minRs[toLayer],
    maxRs[toLayer],
    resolutions[fromLayer],
    radii[toLayer] - radii[fromLayer],
    linkMSAngles[linkId]);
  thrust::device_ptr<Tracklet> tracklets_ptr(spanTracklets[linkId]);
  auto nosync_policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(streams[linkId].get());
  thrust::sort(nosync_policy, tracklets_ptr, tracklets_ptr + nTracklets[linkId]);
  auto unique_end = thrust::unique(nosync_policy, tracklets_ptr, tracklets_ptr + nTracklets[linkId]);
  nTracklets[linkId] = unique_end - tracklets_ptr;
  if (fromLayer > 0) {
    GPUChkErrS(cudaMemsetAsync(trackletsLUTsHost[linkId], 0, (nClusters[fromLayer] + 1) * sizeof(int), streams[linkId].get()));
    gpu::compileTrackletsLookupTableKernel<<<constants::GPUBlocks, constants::GPUThreads, 0, streams[linkId].get()>>>(
      spanTracklets[linkId],
      trackletsLUTsHost[linkId],
      nTracklets[linkId]);
    thrust::exclusive_scan(nosync_policy, trackletsLUTsHost[linkId], trackletsLUTsHost[linkId] + nClusters[fromLayer] + 1, trackletsLUTsHost[linkId]);
  }
}

template <int NLayers>
void countCellsHandler(
  const Cluster** sortedClusters,
  const Cluster** unsortedClusters,
  const TrackingFrameInfo** tfInfo,
  Tracklet** tracklets,
  int** trackletsLUT,
  const int nTracklets,
  const int cellTopologyId,
  const typename TrackingTopology<NLayers>::View topology,
  CellSeed* cells,
  int** cellsLUTsArrayDevice,
  int* cellsLUTsHost,
  const float bz,
  const float maxChi2ClusterAttachment,
  const float cellDeltaTanLambdaSigma,
  const float nSigmaCut,
  const std::vector<float>& layerxX0Host,
  o2::its::ExternalAllocator* alloc,
  gpu::Streams& streams)
{
  thrust::device_vector<float> layerxX0(layerxX0Host);
  gpu::computeLayerCellsKernel<true, NLayers><<<constants::GPUBlocks, constants::GPUThreads, 0, streams[cellTopologyId].get()>>>(
    sortedClusters,   // const Cluster**
    unsortedClusters, // const Cluster**
    tfInfo,           // const TrackingFrameInfo**
    tracklets,        // const Tracklets**
    trackletsLUT,     // const int**
    nTracklets,       // const int
    cellTopologyId,   // const int
    topology,
    cells,                // CellSeed*
    cellsLUTsArrayDevice, // int**
    thrust::raw_pointer_cast(&layerxX0[0]),
    bz,                       // const float
    maxChi2ClusterAttachment, // const float
    cellDeltaTanLambdaSigma,  // const float
    nSigmaCut);               // const float
  auto nosync_policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(streams[cellTopologyId].get());
  thrust::exclusive_scan(nosync_policy, cellsLUTsHost, cellsLUTsHost + nTracklets + 1, cellsLUTsHost);
}

template <int NLayers>
void computeCellsHandler(
  const Cluster** sortedClusters,
  const Cluster** unsortedClusters,
  const TrackingFrameInfo** tfInfo,
  Tracklet** tracklets,
  int** trackletsLUT,
  const int nTracklets,
  const int cellTopologyId,
  const typename TrackingTopology<NLayers>::View topology,
  CellSeed* cells,
  int** cellsLUTsArrayDevice,
  int* cellsLUTsHost,
  const float bz,
  const float maxChi2ClusterAttachment,
  const float cellDeltaTanLambdaSigma,
  const float nSigmaCut,
  const std::vector<float>& layerxX0Host,
  gpu::Streams& streams)
{
  thrust::device_vector<float> layerxX0(layerxX0Host);
  gpu::computeLayerCellsKernel<false, NLayers><<<constants::GPUBlocks, constants::GPUThreads, 0, streams[cellTopologyId].get()>>>(
    sortedClusters,   // const Cluster**
    unsortedClusters, // const Cluster**
    tfInfo,           // const TrackingFrameInfo**
    tracklets,        // const Tracklets**
    trackletsLUT,     // const int**
    nTracklets,       // const int
    cellTopologyId,   // const int
    topology,
    cells,                // CellSeed*
    cellsLUTsArrayDevice, // int**
    thrust::raw_pointer_cast(&layerxX0[0]),
    bz,                       // const float
    maxChi2ClusterAttachment, // const float
    cellDeltaTanLambdaSigma,  // const float
    nSigmaCut);               // const float
}

template <int NLayers>
void countCellNeighboursHandler(CellSeed** cellsLayersDevice,
                                int* neighboursCursor,
                                int** cellsLUTs,
                                const int sourceCellTopologyId,
                                const int targetCellTopologyId,
                                const float maxChi2ClusterAttachment,
                                const float bz,
                                const unsigned int nCells,
                                gpu::Stream& stream)
{
  gpu::computeLayerCellNeighboursKernel<true, NLayers><<<constants::GPUBlocks, constants::GPUThreads, 0, stream.get()>>>(
    cellsLayersDevice,
    neighboursCursor,
    cellsLUTs,
    nullptr,
    sourceCellTopologyId,
    targetCellTopologyId,
    maxChi2ClusterAttachment,
    bz,
    nCells);
}

void scanCellNeighboursHandler(int* neighboursCursor,
                               int* neighboursLUT,
                               const unsigned int nCells,
                               o2::its::ExternalAllocator* alloc,
                               gpu::Stream& stream)
{
  auto nosync_policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(stream.get());
  thrust::exclusive_scan(nosync_policy, neighboursCursor, neighboursCursor + nCells + 1, neighboursCursor);
  GPUChkErrS(cudaMemcpyAsync(neighboursLUT, neighboursCursor, (nCells + 1) * sizeof(int), cudaMemcpyDeviceToDevice, stream.get()));
}

template <int NLayers>
void computeCellNeighboursHandler(CellSeed** cellsLayersDevice,
                                  int* neighboursCursor,
                                  int** cellsLUTs,
                                  CellNeighbour* cellNeighbours,
                                  const int sourceCellTopologyId,
                                  const int targetCellTopologyId,
                                  const float maxChi2ClusterAttachment,
                                  const float bz,
                                  const unsigned int nCells,
                                  gpu::Stream& stream)
{
  gpu::computeLayerCellNeighboursKernel<false, NLayers><<<constants::GPUBlocks, constants::GPUThreads, 0, stream.get()>>>(
    cellsLayersDevice,
    neighboursCursor,
    cellsLUTs,
    cellNeighbours,
    sourceCellTopologyId,
    targetCellTopologyId,
    maxChi2ClusterAttachment,
    bz,
    nCells);
}

int filterCellNeighboursHandler(gpuPair<int, int>* cellNeighbourPairs,
                                int* cellNeighbours,
                                unsigned int nNeigh,
                                gpu::Stream& stream,
                                o2::its::ExternalAllocator* allocator)
{
  auto nosync_policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(allocator)).on(stream.get());
  thrust::device_ptr<gpuPair<int, int>> neighVectorPairs(cellNeighbourPairs);
  thrust::device_ptr<int> validNeighs(cellNeighbours);
  auto updatedEnd = thrust::remove_if(nosync_policy, neighVectorPairs, neighVectorPairs + nNeigh, gpu::is_invalid_pair<int, int>());
  size_t newSize = updatedEnd - neighVectorPairs;
  thrust::stable_sort(nosync_policy, neighVectorPairs, neighVectorPairs + newSize, gpu::sort_by_second<int, int>());
  thrust::transform(nosync_policy, neighVectorPairs, neighVectorPairs + newSize, validNeighs, gpu::pair_to_first<int, int>());
  return newSize;
}

template <int NLayers>
void processNeighboursHandler(const int startLevel,
                              const int defaultCellTopologyId,
                              CellSeed** allCellSeeds,
                              CellSeed* currentCellSeeds,
                              const int* currentCellTopologyIds,
                              const int* currentCellIds,
                              const int* nCells,
                              const unsigned char** usedClusters,
                              CellNeighbour** neighbours,
                              int** neighboursDeviceLUTs,
                              const TrackingFrameInfo** foundTrackingFrameInfo,
                              bounded_vector<TrackSeed<NLayers>>& seedsHost,
                              const float bz,
                              const float maxChi2ClusterAttachment,
                              const float maxChi2NDF,
                              const int maxHoles,
                              const int minSeedingClusters,
                              const LayerMask holeLayerMask,
                              const LayerMask nonSeedingLayerMask,
                              const std::vector<float>& layerxX0Host,
                              const o2::base::Propagator* propagator,
                              const o2::base::PropagatorF::MatCorrType matCorrType,
                              o2::its::ExternalAllocator* alloc)
{
  constexpr uint64_t Tag = qStr2Tag("ITS_PNH1");
  alloc->pushTagOnStack(Tag);
  auto allocInt = gpu::TypedAllocator<int>(alloc);
  auto allocTrackSeed = gpu::TypedAllocator<TrackSeed<NLayers>>(alloc);
  thrust::device_vector<float> layerxX0(layerxX0Host);
  thrust::device_vector<int, gpu::TypedAllocator<int>> foundSeedsTable(nCells[defaultCellTopologyId] + 1, 0, allocInt);
  auto nosync_policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(gpu::Stream::DefaultStream);

  gpu::processNeighboursKernel<true, NLayers, CellSeed><<<constants::GPUBlocks, constants::GPUThreads>>>(
    defaultCellTopologyId,
    startLevel,
    allCellSeeds,
    currentCellSeeds,
    nullptr,
    nullptr,
    nCells[defaultCellTopologyId],
    nullptr,
    nullptr,
    nullptr,
    thrust::raw_pointer_cast(&foundSeedsTable[0]),
    usedClusters,
    neighbours,
    neighboursDeviceLUTs,
    foundTrackingFrameInfo,
    thrust::raw_pointer_cast(&layerxX0[0]),
    bz,
    maxChi2ClusterAttachment,
    propagator,
    matCorrType);
  thrust::exclusive_scan(nosync_policy, foundSeedsTable.begin(), foundSeedsTable.end(), foundSeedsTable.begin());

  thrust::device_vector<int, gpu::TypedAllocator<int>> updatedCellId(foundSeedsTable.back(), 0, allocInt);
  thrust::device_vector<int, gpu::TypedAllocator<int>> updatedCellTopologyId(foundSeedsTable.back(), 0, allocInt);
  thrust::device_vector<TrackSeed<NLayers>, gpu::TypedAllocator<TrackSeed<NLayers>>> updatedCellSeed(foundSeedsTable.back(), allocTrackSeed);
  gpu::processNeighboursKernel<false, NLayers, CellSeed><<<constants::GPUBlocks, constants::GPUThreads>>>(
    defaultCellTopologyId,
    startLevel,
    allCellSeeds,
    currentCellSeeds,
    nullptr,
    nullptr,
    nCells[defaultCellTopologyId],
    thrust::raw_pointer_cast(&updatedCellSeed[0]),
    thrust::raw_pointer_cast(&updatedCellId[0]),
    thrust::raw_pointer_cast(&updatedCellTopologyId[0]),
    thrust::raw_pointer_cast(&foundSeedsTable[0]),
    usedClusters,
    neighbours,
    neighboursDeviceLUTs,
    foundTrackingFrameInfo,
    thrust::raw_pointer_cast(&layerxX0[0]),
    bz,
    maxChi2ClusterAttachment,
    propagator,
    matCorrType);
  GPUChkErrS(cudaStreamSynchronize(gpu::Stream::DefaultStream));

  int level = startLevel;
  thrust::device_vector<int, gpu::TypedAllocator<int>> lastCellId(allocInt);
  thrust::device_vector<int, gpu::TypedAllocator<int>> lastCellTopologyId(allocInt);
  thrust::device_vector<TrackSeed<NLayers>, gpu::TypedAllocator<TrackSeed<NLayers>>> lastCellSeed(allocTrackSeed);
  while (level > 2 && !updatedCellSeed.empty()) {
    lastCellSeed.swap(updatedCellSeed);
    lastCellId.swap(updatedCellId);
    lastCellTopologyId.swap(updatedCellTopologyId);
    thrust::device_vector<TrackSeed<NLayers>, gpu::TypedAllocator<TrackSeed<NLayers>>>(allocTrackSeed).swap(updatedCellSeed);
    thrust::device_vector<int, gpu::TypedAllocator<int>>(allocInt).swap(updatedCellId);
    thrust::device_vector<int, gpu::TypedAllocator<int>>(allocInt).swap(updatedCellTopologyId);
    auto lastCellSeedSize{lastCellSeed.size()};
    foundSeedsTable.resize(lastCellSeedSize + 1);
    thrust::fill(nosync_policy, foundSeedsTable.begin(), foundSeedsTable.end(), 0);

    --level;
    gpu::processNeighboursKernel<true, NLayers, TrackSeed<NLayers>><<<constants::GPUBlocks, constants::GPUThreads>>>(
      constants::UnusedIndex,
      level,
      allCellSeeds,
      thrust::raw_pointer_cast(&lastCellSeed[0]),
      thrust::raw_pointer_cast(&lastCellId[0]),
      thrust::raw_pointer_cast(&lastCellTopologyId[0]),
      lastCellSeedSize,
      nullptr,
      nullptr,
      nullptr,
      thrust::raw_pointer_cast(&foundSeedsTable[0]),
      usedClusters,
      neighbours,
      neighboursDeviceLUTs,
      foundTrackingFrameInfo,
      thrust::raw_pointer_cast(&layerxX0[0]),
      bz,
      maxChi2ClusterAttachment,
      propagator,
      matCorrType);
    thrust::exclusive_scan(nosync_policy, foundSeedsTable.begin(), foundSeedsTable.end(), foundSeedsTable.begin());

    auto foundSeeds{foundSeedsTable.back()};
    updatedCellId.resize(foundSeeds);
    thrust::fill(nosync_policy, updatedCellId.begin(), updatedCellId.end(), 0);
    updatedCellTopologyId.resize(foundSeeds);
    thrust::fill(nosync_policy, updatedCellTopologyId.begin(), updatedCellTopologyId.end(), 0);
    updatedCellSeed.resize(foundSeeds);
    thrust::fill(nosync_policy, updatedCellSeed.begin(), updatedCellSeed.end(), TrackSeed<NLayers>());

    gpu::processNeighboursKernel<false, NLayers, TrackSeed<NLayers>><<<constants::GPUBlocks, constants::GPUThreads>>>(
      constants::UnusedIndex,
      level,
      allCellSeeds,
      thrust::raw_pointer_cast(&lastCellSeed[0]),
      thrust::raw_pointer_cast(&lastCellId[0]),
      thrust::raw_pointer_cast(&lastCellTopologyId[0]),
      lastCellSeedSize,
      thrust::raw_pointer_cast(&updatedCellSeed[0]),
      thrust::raw_pointer_cast(&updatedCellId[0]),
      thrust::raw_pointer_cast(&updatedCellTopologyId[0]),
      thrust::raw_pointer_cast(&foundSeedsTable[0]),
      usedClusters,
      neighbours,
      neighboursDeviceLUTs,
      foundTrackingFrameInfo,
      thrust::raw_pointer_cast(&layerxX0[0]),
      bz,
      maxChi2ClusterAttachment,
      propagator,
      matCorrType);
  }
  GPUChkErrS(cudaStreamSynchronize(gpu::Stream::DefaultStream));
  thrust::device_vector<TrackSeed<NLayers>, gpu::TypedAllocator<TrackSeed<NLayers>>> outSeeds(updatedCellSeed.size(), allocTrackSeed);
  auto end = thrust::copy_if(nosync_policy, updatedCellSeed.begin(), updatedCellSeed.end(), outSeeds.begin(), track::TrackSeedSelector<NLayers>{constants::MaxTrackSeedQ2Pt, maxChi2NDF, startLevel, maxHoles, minSeedingClusters, holeLayerMask, nonSeedingLayerMask});
  auto s{end - outSeeds.begin()};
  seedsHost.reserve(seedsHost.size() + s);
  thrust::copy(outSeeds.begin(), outSeeds.begin() + s, std::back_inserter(seedsHost));
  alloc->popTagOffStack(Tag);
}

template <int NLayers>
void countTrackSeedHandler(TrackSeed<NLayers>* trackSeeds,
                           const TrackingFrameInfo** foundTrackingFrameInfo,
                           const Cluster** unsortedClusters,
                           int* seedLUT,
                           const std::vector<float>& layerRadiiHost,
                           const std::vector<float>& minPtsHost,
                           const std::vector<float>& layerxX0Host,
                           const unsigned int nSeeds,
                           const float bz,
                           const float maxChi2ClusterAttachment,
                           const float maxChi2NDF,
                           const int reseedIfShorter,
                           const bool repeatRefitOut,
                           const bool shiftRefToCluster,
                           const o2::base::Propagator* propagator,
                           const o2::base::PropagatorF::MatCorrType matCorrType,
                           o2::its::ExternalAllocator* alloc)
{
  // TODO: the minPts&layerRadii is transfered twice
  // we should allocate this in constant memory and stop these
  // small transferes!
  thrust::device_vector<float> minPts(minPtsHost);
  thrust::device_vector<float> layerRadii(layerRadiiHost);
  thrust::device_vector<float> layerxX0(layerxX0Host);
  gpu::countTrackSeedsKernel<NLayers><<<constants::GPUBlocks, constants::GPUThreads>>>(
    trackSeeds,                               // CellSeed*
    foundTrackingFrameInfo,                   // TrackingFrameInfo**
    unsortedClusters,                         // Cluster**
    seedLUT,                                  // int*
    thrust::raw_pointer_cast(&layerRadii[0]), // const float*
    thrust::raw_pointer_cast(&minPts[0]),     // const float*
    thrust::raw_pointer_cast(&layerxX0[0]),   // const float*
    nSeeds,                                   // const unsigned int
    bz,                                       // const float
    maxChi2ClusterAttachment,                 // float
    maxChi2NDF,                               // float
    reseedIfShorter,                          // int
    repeatRefitOut,                           // bool
    shiftRefToCluster,                        // bool
    propagator,                               // const o2::base::Propagator*
    matCorrType);                             // o2::base::PropagatorF::MatCorrType
  auto sync_policy = THRUST_NAMESPACE::par(gpu::TypedAllocator<char>(alloc));
  thrust::exclusive_scan(sync_policy, seedLUT, seedLUT + nSeeds + 1, seedLUT);
}

template <int NLayers>
void computeTrackSeedHandler(TrackSeed<NLayers>* trackSeeds,
                             const TrackingFrameInfo** foundTrackingFrameInfo,
                             const Cluster** unsortedClusters,
                             const IndexTableUtils<NLayers>* utils,
                             const typename ROFMaskTable<NLayers>::View& rofMask,
                             const typename ROFOverlapTable<NLayers>::View& rofOverlaps,
                             const Cluster** clusters,
                             const unsigned char** usedClusters,
                             const int** clustersIndexTables,
                             const int** ROFClusters,
                             o2::its::TrackITSExt* tracks,
                             int* trackIndices,
                             const int* seedLUT,
                             TrackExtensionHypothesis<NLayers>* activeHypotheses,
                             TrackExtensionHypothesis<NLayers>* nextHypotheses,
                             const std::vector<float>& layerRadiiHost,
                             const std::vector<float>& minPtsHost,
                             const std::vector<float>& layerxX0Host,
                             const unsigned int nSeeds,
                             const unsigned int nTracks,
                             const float bz,
                             const float maxChi2ClusterAttachment,
                             const float maxChi2NDF,
                             const int reseedIfShorter,
                             const bool repeatRefitOut,
                             const bool shiftRefToCluster,
                             const int nLayers,
                             const int phiBins,
                             const int maxHypotheses,
                             const bool extendTop,
                             const bool extendBot,
                             const float nSigmaCutPhi,
                             const float nSigmaCutZ,
                             const o2::base::Propagator* propagator,
                             const o2::base::PropagatorF::MatCorrType matCorrType,
                             o2::its::ExternalAllocator* alloc)
{
  thrust::device_vector<float> minPts(minPtsHost);
  thrust::device_vector<float> layerRadii(layerRadiiHost);
  thrust::device_vector<float> layerxX0(layerxX0Host);
  gpu::fitTrackSeedsKernel<NLayers><<<constants::GPUBlocks, constants::GPUThreads>>>(
    trackSeeds,                               // CellSeed*
    foundTrackingFrameInfo,                   // TrackingFrameInfo**
    unsortedClusters,                         // Cluster**
    utils,                                    // IndexTableUtils*
    rofMask,                                  // ROFMaskTable::View
    rofOverlaps,                              // ROFOverlapTable::View
    clusters,                                 // Cluster**
    usedClusters,                             // unsigned char**
    clustersIndexTables,                      // int**
    ROFClusters,                              // int**
    tracks,                                   // TrackITSExt*
    seedLUT,                                  // const int*
    activeHypotheses,                         // TrackExtensionHypothesis*
    nextHypotheses,                           // TrackExtensionHypothesis*
    thrust::raw_pointer_cast(&layerRadii[0]), // const float*
    thrust::raw_pointer_cast(&minPts[0]),     // const float*
    thrust::raw_pointer_cast(&layerxX0[0]),   // const float*
    nSeeds,                                   // const unsigned int
    bz,                                       // const float
    maxChi2ClusterAttachment,                 // float
    maxChi2NDF,                               // float
    reseedIfShorter,                          // int
    repeatRefitOut,                           // bool
    shiftRefToCluster,                        // bool
    nLayers,                                  // int
    phiBins,                                  // int
    maxHypotheses,                            // int
    extendTop,                                // bool
    extendBot,                                // bool
    nSigmaCutPhi,                             // float
    nSigmaCutZ,                               // float
    propagator,                               // const o2::base::Propagator*
    matCorrType);                             // o2::base::PropagatorF::MatCorrType
  auto sync_policy = THRUST_NAMESPACE::par(gpu::TypedAllocator<char>(alloc));
  thrust::device_ptr<int> trackIndicesPtr(trackIndices);
  thrust::sequence(sync_policy, trackIndicesPtr, trackIndicesPtr + nTracks);
  thrust::sort(sync_policy, trackIndicesPtr, trackIndicesPtr + nTracks, gpu::compare_track_index_chi2{tracks});
}

/// Explicit instantiation of ITS2 handlers
template void countTrackletsInROFsHandler<7>(const IndexTableUtils<7>* utils,
                                             const ROFMaskTable<7>::View& rofMask,
                                             const int linkId,
                                             const int fromLayer,
                                             const int toLayer,
                                             const ROFOverlapTable<7>::View& rofOverlaps,
                                             const ROFVertexLookupTable<7>::View& vertexLUT,
                                             const int vertexId,
                                             const Vertex* vertices,
                                             const int* rofPV,
                                             const Cluster** clusters,
                                             std::vector<unsigned int> nClusters,
                                             const int** ROFClusters,
                                             const unsigned char** usedClusters,
                                             const int** clustersIndexTables,
                                             int** trackletsLUTs,
                                             gsl::span<int*> trackletsLUTsHost,
                                             const bool selectUPCVertices,
                                             const float NSigmaCut,
                                             const TrackingTopology<7>::View topology,
                                             bounded_vector<float>& linkPhiCuts,
                                             const float resolutionPV,
                                             std::array<float, 7>& minRs,
                                             std::array<float, 7>& maxRs,
                                             bounded_vector<float>& resolutions,
                                             std::vector<float>& radii,
                                             bounded_vector<float>& linkMSAngles,
                                             o2::its::ExternalAllocator* alloc,
                                             gpu::Streams& streams);

template void computeTrackletsInROFsHandler<7>(const IndexTableUtils<7>* utils,
                                               const ROFMaskTable<7>::View& rofMask,
                                               const int linkId,
                                               const int fromLayer,
                                               const int toLayer,
                                               const ROFOverlapTable<7>::View& rofOverlaps,
                                               const ROFVertexLookupTable<7>::View& vertexLUT,
                                               const int vertexId,
                                               const Vertex* vertices,
                                               const int* rofPV,
                                               const Cluster** clusters,
                                               std::vector<unsigned int> nClusters,
                                               const int** ROFClusters,
                                               const unsigned char** usedClusters,
                                               const int** clustersIndexTables,
                                               Tracklet** tracklets,
                                               gsl::span<Tracklet*> spanTracklets,
                                               gsl::span<int> nTracklets,
                                               int** trackletsLUTs,
                                               gsl::span<int*> trackletsLUTsHost,
                                               const bool selectUPCVertices,
                                               const float NSigmaCut,
                                               const TrackingTopology<7>::View topology,
                                               bounded_vector<float>& linkPhiCuts,
                                               const float resolutionPV,
                                               std::array<float, 7>& minRs,
                                               std::array<float, 7>& maxRs,
                                               bounded_vector<float>& resolutions,
                                               std::vector<float>& radii,
                                               bounded_vector<float>& linkMSAngles,
                                               o2::its::ExternalAllocator* alloc,
                                               gpu::Streams& streams);

template void countCellsHandler<7>(const Cluster** sortedClusters,
                                   const Cluster** unsortedClusters,
                                   const TrackingFrameInfo** tfInfo,
                                   Tracklet** tracklets,
                                   int** trackletsLUT,
                                   const int nTracklets,
                                   const int cellTopologyId,
                                   const TrackingTopology<7>::View topology,
                                   CellSeed* cells,
                                   int** cellsLUTsArrayDevice,
                                   int* cellsLUTsHost,
                                   const float bz,
                                   const float maxChi2ClusterAttachment,
                                   const float cellDeltaTanLambdaSigma,
                                   const float nSigmaCut,
                                   const std::vector<float>& layerxX0Host,
                                   o2::its::ExternalAllocator* alloc,
                                   gpu::Streams& streams);

template void computeCellsHandler<7>(const Cluster** sortedClusters,
                                     const Cluster** unsortedClusters,
                                     const TrackingFrameInfo** tfInfo,
                                     Tracklet** tracklets,
                                     int** trackletsLUT,
                                     const int nTracklets,
                                     const int cellTopologyId,
                                     const TrackingTopology<7>::View topology,
                                     CellSeed* cells,
                                     int** cellsLUTsArrayDevice,
                                     int* cellsLUTsHost,
                                     const float bz,
                                     const float maxChi2ClusterAttachment,
                                     const float cellDeltaTanLambdaSigma,
                                     const float nSigmaCut,
                                     const std::vector<float>& layerxX0Host,
                                     gpu::Streams& streams);

template void countCellNeighboursHandler<7>(CellSeed** cellsLayersDevice,
                                            int* neighboursCursor,
                                            int** cellsLUTs,
                                            const int sourceCellTopologyId,
                                            const int targetCellTopologyId,
                                            const float maxChi2ClusterAttachment,
                                            const float bz,
                                            const unsigned int nCells,
                                            gpu::Stream& stream);

template void computeCellNeighboursHandler<7>(CellSeed** cellsLayersDevice,
                                              int* neighboursCursor,
                                              int** cellsLUTs,
                                              CellNeighbour* cellNeighbours,
                                              const int sourceCellTopologyId,
                                              const int targetCellTopologyId,
                                              const float maxChi2ClusterAttachment,
                                              const float bz,
                                              const unsigned int nCells,
                                              gpu::Stream& stream);

template void processNeighboursHandler<7>(const int startLevel,
                                          const int defaultCellTopologyId,
                                          CellSeed** allCellSeeds,
                                          CellSeed* currentCellSeeds,
                                          const int* currentCellTopologyIds,
                                          const int* currentCellIds,
                                          const int* nCells,
                                          const unsigned char** usedClusters,
                                          CellNeighbour** neighbours,
                                          int** neighboursDeviceLUTs,
                                          const TrackingFrameInfo** foundTrackingFrameInfo,
                                          bounded_vector<TrackSeed<7>>& seedsHost,
                                          const float bz,
                                          const float maxChi2ClusterAttachment,
                                          const float maxChi2NDF,
                                          const int maxHoles,
                                          const int minSeedingClusters,
                                          const LayerMask holeLayerMask,
                                          const LayerMask nonSeedingLayerMask,
                                          const std::vector<float>& layerxX0Host,
                                          const o2::base::Propagator* propagator,
                                          const o2::base::PropagatorF::MatCorrType matCorrType,
                                          o2::its::ExternalAllocator* alloc);

template void countTrackSeedHandler(TrackSeed<7>* trackSeeds,
                                    const TrackingFrameInfo** foundTrackingFrameInfo,
                                    const Cluster** unsortedClusters,
                                    int* seedLUT,
                                    const std::vector<float>& layerRadiiHost,
                                    const std::vector<float>& minPtsHost,
                                    const std::vector<float>& layerxX0Host,
                                    const unsigned int nSeeds,
                                    const float bz,
                                    const float maxChi2ClusterAttachment,
                                    const float maxChi2NDF,
                                    const int reseedIfShorter,
                                    const bool repeatRefitOut,
                                    const bool shiftRefToCluster,
                                    const o2::base::Propagator* propagator,
                                    const o2::base::PropagatorF::MatCorrType matCorrType,
                                    o2::its::ExternalAllocator* alloc);

template void computeTrackSeedHandler(TrackSeed<7>* trackSeeds,
                                      const TrackingFrameInfo** foundTrackingFrameInfo,
                                      const Cluster** unsortedClusters,
                                      const IndexTableUtils<7>* utils,
                                      const ROFMaskTable<7>::View& rofMask,
                                      const ROFOverlapTable<7>::View& rofOverlaps,
                                      const Cluster** clusters,
                                      const unsigned char** usedClusters,
                                      const int** clustersIndexTables,
                                      const int** ROFClusters,
                                      o2::its::TrackITSExt* tracks,
                                      int* trackIndices,
                                      const int* seedLUT,
                                      TrackExtensionHypothesis<7>* activeHypotheses,
                                      TrackExtensionHypothesis<7>* nextHypotheses,
                                      const std::vector<float>& layerRadiiHost,
                                      const std::vector<float>& minPtsHost,
                                      const std::vector<float>& layerxX0Host,
                                      const unsigned int nSeeds,
                                      const unsigned int nTracks,
                                      const float bz,
                                      const float maxChi2ClusterAttachment,
                                      const float maxChi2NDF,
                                      const int reseedIfShorter,
                                      const bool repeatRefitOut,
                                      const bool shiftRefToCluster,
                                      const int nLayers,
                                      const int phiBins,
                                      const int maxHypotheses,
                                      const bool extendTop,
                                      const bool extendBot,
                                      const float nSigmaCutPhi,
                                      const float nSigmaCutZ,
                                      const o2::base::Propagator* propagator,
                                      const o2::base::PropagatorF::MatCorrType matCorrType,
                                      o2::its::ExternalAllocator* alloc);

/// Explicit instantiation of ALICE3 handlers
#ifdef ENABLE_UPGRADES
template void countTrackletsInROFsHandler<11>(const IndexTableUtils<11>* utils,
                                              const ROFMaskTable<11>::View& rofMask,
                                              const int linkId,
                                              const int fromLayer,
                                              const int toLayer,
                                              const ROFOverlapTable<11>::View& rofOverlaps,
                                              const ROFVertexLookupTable<11>::View& vertexLUT,
                                              const int vertexId,
                                              const Vertex* vertices,
                                              const int* rofPV,
                                              const Cluster** clusters,
                                              std::vector<unsigned int> nClusters,
                                              const int** ROFClusters,
                                              const unsigned char** usedClusters,
                                              const int** clustersIndexTables,
                                              int** trackletsLUTs,
                                              gsl::span<int*> trackletsLUTsHost,
                                              const bool selectUPCVertices,
                                              const float NSigmaCut,
                                              const TrackingTopology<11>::View topology,
                                              bounded_vector<float>& linkPhiCuts,
                                              const float resolutionPV,
                                              std::array<float, 11>& minRs,
                                              std::array<float, 11>& maxRs,
                                              bounded_vector<float>& resolutions,
                                              std::vector<float>& radii,
                                              bounded_vector<float>& linkMSAngles,
                                              o2::its::ExternalAllocator* alloc,
                                              gpu::Streams& streams);

template void computeTrackletsInROFsHandler<11>(const IndexTableUtils<11>* utils,
                                                const ROFMaskTable<11>::View& rofMask,
                                                const int linkId,
                                                const int fromLayer,
                                                const int toLayer,
                                                const ROFOverlapTable<11>::View& rofOverlaps,
                                                const ROFVertexLookupTable<11>::View& vertexLUT,
                                                const int vertexId,
                                                const Vertex* vertices,
                                                const int* rofPV,
                                                const Cluster** clusters,
                                                std::vector<unsigned int> nClusters,
                                                const int** ROFClusters,
                                                const unsigned char** usedClusters,
                                                const int** clustersIndexTables,
                                                Tracklet** tracklets,
                                                gsl::span<Tracklet*> spanTracklets,
                                                gsl::span<int> nTracklets,
                                                int** trackletsLUTs,
                                                gsl::span<int*> trackletsLUTsHost,
                                                const bool selectUPCVertices,
                                                const float NSigmaCut,
                                                const TrackingTopology<11>::View topology,
                                                bounded_vector<float>& linkPhiCuts,
                                                const float resolutionPV,
                                                std::array<float, 11>& minRs,
                                                std::array<float, 11>& maxRs,
                                                bounded_vector<float>& resolutions,
                                                std::vector<float>& radii,
                                                bounded_vector<float>& linkMSAngles,
                                                o2::its::ExternalAllocator* alloc,
                                                gpu::Streams& streams);

template void countCellsHandler<11>(const Cluster** sortedClusters,
                                    const Cluster** unsortedClusters,
                                    const TrackingFrameInfo** tfInfo,
                                    Tracklet** tracklets,
                                    int** trackletsLUT,
                                    const int nTracklets,
                                    const int cellTopologyId,
                                    const TrackingTopology<11>::View topology,
                                    CellSeed* cells,
                                    int** cellsLUTsArrayDevice,
                                    int* cellsLUTsHost,
                                    const float bz,
                                    const float maxChi2ClusterAttachment,
                                    const float cellDeltaTanLambdaSigma,
                                    const float nSigmaCut,
                                    const std::vector<float>& layerxX0Host,
                                    o2::its::ExternalAllocator* alloc,
                                    gpu::Streams& streams);

template void computeCellsHandler<11>(const Cluster** sortedClusters,
                                      const Cluster** unsortedClusters,
                                      const TrackingFrameInfo** tfInfo,
                                      Tracklet** tracklets,
                                      int** trackletsLUT,
                                      const int nTracklets,
                                      const int cellTopologyId,
                                      const TrackingTopology<11>::View topology,
                                      CellSeed* cells,
                                      int** cellsLUTsArrayDevice,
                                      int* cellsLUTsHost,
                                      const float bz,
                                      const float maxChi2ClusterAttachment,
                                      const float cellDeltaTanLambdaSigma,
                                      const float nSigmaCut,
                                      const std::vector<float>& layerxX0Host,
                                      gpu::Streams& streams);

template void countCellNeighboursHandler<11>(CellSeed** cellsLayersDevice,
                                             int* neighboursCursor,
                                             int** cellsLUTs,
                                             const int sourceCellTopologyId,
                                             const int targetCellTopologyId,
                                             const float maxChi2ClusterAttachment,
                                             const float bz,
                                             const unsigned int nCells,
                                             gpu::Stream& stream);

template void computeCellNeighboursHandler<11>(CellSeed** cellsLayersDevice,
                                               int* neighboursCursor,
                                               int** cellsLUTs,
                                               CellNeighbour* cellNeighbours,
                                               const int sourceCellTopologyId,
                                               const int targetCellTopologyId,
                                               const float maxChi2ClusterAttachment,
                                               const float bz,
                                               const unsigned int nCells,
                                               gpu::Stream& stream);

template void processNeighboursHandler<11>(const int startLevel,
                                           const int defaultCellTopologyId,
                                           CellSeed** allCellSeeds,
                                           CellSeed* currentCellSeeds,
                                           const int* currentCellTopologyIds,
                                           const int* currentCellIds,
                                           const int* nCells,
                                           const unsigned char** usedClusters,
                                           CellNeighbour** neighbours,
                                           int** neighboursDeviceLUTs,
                                           const TrackingFrameInfo** foundTrackingFrameInfo,
                                           bounded_vector<TrackSeed<11>>& seedsHost,
                                           const float bz,
                                           const float maxChi2ClusterAttachment,
                                           const float maxChi2NDF,
                                           const int maxHoles,
                                           const int minSeedingClusters,
                                           const LayerMask holeLayerMask,
                                           const LayerMask nonSeedingLayerMask,
                                           const std::vector<float>& layerxX0Host,
                                           const o2::base::Propagator* propagator,
                                           const o2::base::PropagatorF::MatCorrType matCorrType,
                                           o2::its::ExternalAllocator* alloc);

template void countTrackSeedHandler(TrackSeed<11>* trackSeeds,
                                    const TrackingFrameInfo** foundTrackingFrameInfo,
                                    const Cluster** unsortedClusters,
                                    int* seedLUT,
                                    const std::vector<float>& layerRadiiHost,
                                    const std::vector<float>& minPtsHost,
                                    const std::vector<float>& layerxX0Host,
                                    const unsigned int nSeeds,
                                    const float bz,
                                    const float maxChi2ClusterAttachment,
                                    const float maxChi2NDF,
                                    const int reseedIfShorter,
                                    const bool repeatRefitOut,
                                    const bool shiftRefToCluster,
                                    const o2::base::Propagator* propagator,
                                    const o2::base::PropagatorF::MatCorrType matCorrType,
                                    o2::its::ExternalAllocator* alloc);

template void computeTrackSeedHandler(TrackSeed<11>* trackSeeds,
                                      const TrackingFrameInfo** foundTrackingFrameInfo,
                                      const Cluster** unsortedClusters,
                                      const IndexTableUtils<11>* utils,
                                      const ROFMaskTable<11>::View& rofMask,
                                      const ROFOverlapTable<11>::View& rofOverlaps,
                                      const Cluster** clusters,
                                      const unsigned char** usedClusters,
                                      const int** clustersIndexTables,
                                      const int** ROFClusters,
                                      o2::its::TrackITSExt* tracks,
                                      int* trackIndices,
                                      const int* seedLUT,
                                      TrackExtensionHypothesis<11>* activeHypotheses,
                                      TrackExtensionHypothesis<11>* nextHypotheses,
                                      const std::vector<float>& layerRadiiHost,
                                      const std::vector<float>& minPtsHost,
                                      const std::vector<float>& layerxX0Host,
                                      const unsigned int nSeeds,
                                      const unsigned int nTracks,
                                      const float bz,
                                      const float maxChi2ClusterAttachment,
                                      const float maxChi2NDF,
                                      const int reseedIfShorter,
                                      const bool repeatRefitOut,
                                      const bool shiftRefToCluster,
                                      const int nLayers,
                                      const int phiBins,
                                      const int maxHypotheses,
                                      const bool extendTop,
                                      const bool extendBot,
                                      const float nSigmaCutPhi,
                                      const float nSigmaCutZ,
                                      const o2::base::Propagator* propagator,
                                      const o2::base::PropagatorF::MatCorrType matCorrType,
                                      o2::its::ExternalAllocator* alloc);
#endif
} // namespace o2::its
