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
#include <type_traits>
#include <cmath>
#include <limits>
#include <unistd.h>

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/binary_search.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "DataFormatsITS/TrackITS.h"
#include "ITSMFTTracking/Constants.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStrackingGPU/LaunchGeometry.h"
#include "ITSMFTTracking/MathUtils.h"
#include "ITStracking/ExternalAllocator.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/Cluster.h"
#include "ITStracking/Cell.h"
#include "ITStracking/TrackHelpers.h"
#include "ITStracking/TrackFollower.h"
#include "ITStrackingGPU/TrackingKernels.h"
#include "ITStrackingGPU/Utils.h"
#include "MathUtils/Utils.h"
#include "ITStrackingGPU/ClusterLinesGPU.h"
#include "utils/strtag.h"

// O2 track model
#include "ReconstructionDataFormats/Track.h"
#include "DetectorsBase/Propagator.h"
using namespace o2::track;

namespace o2::its
{

using o2::itsmft::tracking::runOnSlab;
using o2::itsmft::tracking::SlabSite;
namespace gpu
{

struct compare_track_index_chi2 {
  const TrackITSExt* tracks;
  const int* seedIndices;

  GPUhd() bool operator()(const int a, const int b) const
  {
    if (o2::its::track::isBetter(tracks[a], tracks[b])) {
      return true;
    }
    if (o2::its::track::isBetter(tracks[b], tracks[a])) {
      return false;
    }
    return seedIndices[a] < seedIndices[b];
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

template <int NLayers, bool ExtendTracks>
GPUg() void __launch_bounds__(GPUThreads, (ExtendTracks ? MinBlocks.fitTrackSeedsExtended : MinBlocks.fitTrackSeeds)) fitTrackSeedsKernel(
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
  int* trackSeedIndices,
  int* outputCounter,
  const int trackCapacity,
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
    TrackITSInternal<NLayers> temporaryTrack;
    bool refitSuccess = o2::its::track::refitTrackSeed(trackSeeds[iCurrentTrackSeedIndex],
                                                       temporaryTrack,
                                                       fitCtx,
                                                       unsortedClusters,
                                                       layerRadii,
                                                       minPts,
                                                       reseedIfShorter);
    if (!refitSuccess) {
      continue;
    }
    uint32_t bestDiff{0};
    if constexpr (ExtendTracks) {
      if ((extendTop || extendBot) && activeHypothesesScratch && nextHypothesesScratch) {
        const int maxHypotheses = o2::gpu::CAMath::Max(maxHypothesesConfig, 1);
        const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
        auto* activeHypotheses = activeHypothesesScratch + threadIndex * maxHypotheses;
        auto* nextHypotheses = nextHypothesesScratch + threadIndex * maxHypotheses;
        const auto backup = temporaryTrack;
        auto best = temporaryTrack;
        TrackExtensionDirectionFollowerDevice<NLayers> followDirection{&fitCtx, &followCtx, activeHypotheses, nextHypotheses};
        TrackExtensionBestTrial<NLayers> bestTrial{backup.getPattern(), fitCtx};
        followTrackExtensionBranches(backup, extendTop, extendBot, nLayers, followDirection, bestTrial, best, bestDiff);
        temporaryTrack = best;
      }
    }
    const int slot = atomicAdd(outputCounter, 1);
    if (slot >= trackCapacity) {
      continue;
    }
    tracks[slot] = makeTrackITSExt(temporaryTrack);
    if (bestDiff) {
      tracks[slot].setExtendedLayerPattern<NLayers>(bestDiff);
    }
    trackSeedIndices[slot] = iCurrentTrackSeedIndex;
  }
}

/// A (source cell, target cell) pair that passed the index and time-stamp cuts and is worth fitting.
struct CellNeighbourCandidate {
  int currentCell;
  int nextCell;
};
template <int NLayers>
GPUg() void __launch_bounds__(GPUThreads, MinBlocks.computeLayerCellNeighbours) computeLayerCellNeighbourCandidatesKernel(
  CellSeed** cellSeedArray,
  int** cellsLUTs,
  const int sourceCellTopologyId,
  const int targetCellTopologyId,
  CellNeighbourCandidate* candidates, // nullptr on the counting pass
  int* candidateCounter,
  const int candidateCapacity, // 0 on the counting pass, so nothing is written
  const unsigned int nCells)
{
  for (int iCurrentCellIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentCellIndex < nCells; iCurrentCellIndex += blockDim.x * gridDim.x) {
    const auto& currentCellSeed{cellSeedArray[sourceCellTopologyId][iCurrentCellIndex]};
    const int nextLayerTrackletIndex{currentCellSeed.getSecondTrackletIndex()};
    const int nextLayerFirstCellIndex{cellsLUTs[targetCellTopologyId][nextLayerTrackletIndex]};
    const int nextLayerLastCellIndex{cellsLUTs[targetCellTopologyId][nextLayerTrackletIndex + 1]};
    const auto currentTimeStamp{currentCellSeed.getTimeStamp()};
    for (int iNextCell{nextLayerFirstCellIndex}; iNextCell < nextLayerLastCellIndex; ++iNextCell) {
      const auto& nextCellSeed{cellSeedArray[targetCellTopologyId][iNextCell]}; // No copy: only two accessors are read.
      if (nextCellSeed.getFirstTrackletIndex() != nextLayerTrackletIndex || !currentTimeStamp.isCompatible(nextCellSeed.getTimeStamp())) {
        break;
      }
      const int outputIndex = atomicAdd(candidateCounter, 1);
      if (outputIndex < candidateCapacity) {
        candidates[outputIndex] = {iCurrentCellIndex, iNextCell};
      }
    }
  }
}

template <int NLayers>
GPUg() void __launch_bounds__(GPUThreads, MinBlocks.computeLayerCellNeighbours) fitCellNeighboursKernel(
  CellSeed** cellSeedArray,
  const CellNeighbourCandidate* candidates,
  const int nCandidates,
  CellNeighbour* cellNeighbours,
  int* outputCounter,
  const int outputCapacity,
  const int sourceCellTopologyId,
  const int targetCellTopologyId,
  const float maxChi2ClusterAttachment,
  const float bz)
{
  for (int iCandidate = blockIdx.x * blockDim.x + threadIdx.x; iCandidate < nCandidates; iCandidate += blockDim.x * gridDim.x) {
    const CellNeighbourCandidate candidate = candidates[iCandidate];
    const auto& currentCellSeed{cellSeedArray[sourceCellTopologyId][candidate.currentCell]};
    auto nextCellSeed{cellSeedArray[targetCellTopologyId][candidate.nextCell]}; // Copy

    if (!nextCellSeed.rotate(currentCellSeed.getAlpha()) ||
        !nextCellSeed.propagateTo(currentCellSeed.getX(), bz)) {
      continue;
    }

    float chi2 = currentCellSeed.getPredictedChi2Fast(nextCellSeed);
    if (chi2 > maxChi2ClusterAttachment) {
      continue;
    }

    const int currentCellLevel{currentCellSeed.getLevel()};
    const int outputIndex = atomicAdd(outputCounter, 1);
    if (outputIndex < outputCapacity) {
      cellNeighbours[outputIndex] = {sourceCellTopologyId, candidate.currentCell, targetCellTopologyId, candidate.nextCell, currentCellLevel + 1};
    }
    if (currentCellLevel >= nextCellSeed.getLevel()) {
      atomicMax(cellSeedArray[targetCellTopologyId][candidate.nextCell].getLevelPtr(), currentCellLevel + 1);
    }
  }
}

/// A tracklet pair that passed the cheap cuts and is worth fitting.
struct CellCandidate {
  int firstTrackletIndex;
  int secondTrackletIndex;
};
template <int NLayers, bool Emit>
GPUg() void __launch_bounds__(GPUThreads, MinBlocks.computeLayerCells) computeLayerCellCandidatesKernel(
  Tracklet** tracklets,
  int** trackletsLUT,
  const int nTrackletsCurrent,
  const int cellTopologyId,
  const typename TrackingTopology<NLayers>::View topology,
  const Cluster** sortedClusters,
  const Cluster** unsortedClusters,
  const TrackingFrameInfo** tfInfo,
  const float* layerxX0,
  const float bz,
  CellCandidate* candidates,
  unsigned int* candidateKeys,
  int* outputCounter,
  const int outputCapacity,
  const float cellDeltaTanLambdaSigma,
  const float cellDeltaPhiCut,
  const float nSigmaCut)
{
  const auto cellTopology = topology.getCell(cellTopologyId);
  for (int iCurrentTrackletIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentTrackletIndex < nTrackletsCurrent; iCurrentTrackletIndex += blockDim.x * gridDim.x) {
    const Tracklet& currentTracklet = tracklets[cellTopology.firstLink][iCurrentTrackletIndex];
    const int nextLayerClusterIndex{currentTracklet.secondClusterIndex};
    const int nextLayerFirstTrackletIndex{trackletsLUT[cellTopology.secondLink][nextLayerClusterIndex]};
    const int nextLayerLastTrackletIndex{trackletsLUT[cellTopology.secondLink][nextLayerClusterIndex + 1]};
    if (nextLayerFirstTrackletIndex == nextLayerLastTrackletIndex) {
      continue;
    }
    for (int iNextTrackletIndex{nextLayerFirstTrackletIndex}; iNextTrackletIndex < nextLayerLastTrackletIndex; ++iNextTrackletIndex) {
      if (tracklets[cellTopology.secondLink][iNextTrackletIndex].firstClusterIndex != nextLayerClusterIndex) {
        break;
      }
      const Tracklet& nextTracklet = tracklets[cellTopology.secondLink][iNextTrackletIndex];
      if (!currentTracklet.getTimeStamp().isCompatible(nextTracklet.getTimeStamp())) {
        continue;
      }
      if (cellDeltaPhiCut > 0.f &&
          !math_utils::isPhiDifferenceBelow(currentTracklet.phi, nextTracklet.phi, cellDeltaPhiCut)) {
        continue;
      }
      const float deltaTanLambda{o2::gpu::CAMath::Abs(currentTracklet.tanLambda - nextTracklet.tanLambda)};
      if (deltaTanLambda / cellDeltaTanLambdaSigma < nSigmaCut) {
        if constexpr (Emit) {
          const int outputIndex = atomicAdd(outputCounter, 1);
          if (outputIndex < outputCapacity) {
            candidates[outputIndex] = CellCandidate{iCurrentTrackletIndex, iNextTrackletIndex};
            const auto firstLink = topology.getLink(cellTopology.firstLink);
            const auto secondLink = topology.getLink(cellTopology.secondLink);
            const int layers[3] = {firstLink.fromLayer, firstLink.toLayer, secondLink.toLayer};
            const int clusId[3]{
              sortedClusters[layers[0]][currentTracklet.firstClusterIndex].clusterId,
              sortedClusters[layers[1]][nextTracklet.firstClusterIndex].clusterId,
              sortedClusters[layers[2]][nextTracklet.secondClusterIndex].clusterId};
            const auto seed{o2::its::track::buildTrackSeed(unsortedClusters[layers[0]][clusId[0]], unsortedClusters[layers[1]][clusId[1]], tfInfo[layers[2]][clusId[2]], bz)};
            candidateKeys[outputIndex] = static_cast<unsigned int>(
              seed.getELossSteps(layerxX0[layers[1]] * constants::Radl * constants::Rho, true));
          }
        } else {
          atomicAdd(outputCounter, 1);
        }
      }
    }
  }
}

/// Fit one tracklet pair per thread, emitting a cell for each pair that survives the fit.
template <int NLayers>
GPUg() void __launch_bounds__(GPUThreads, MinBlocks.computeLayerCells) fitLayerCellsKernel(
  const Cluster** sortedClusters,
  const Cluster** unsortedClusters,
  const TrackingFrameInfo** tfInfo,
  Tracklet** tracklets,
  const CellCandidate* candidates,
  const int nCandidates,
  const int cellTopologyId,
  const typename TrackingTopology<NLayers>::View topology,
  CellSeed* cells,
  int* outputCounter,
  const int outputCapacity,
  const float* layerxX0,
  const float bz,
  const float maxChi2ClusterAttachment)
{
  const auto cellTopology = topology.getCell(cellTopologyId);
  const auto first = topology.getLink(cellTopology.firstLink);
  const auto second = topology.getLink(cellTopology.secondLink);
  const int layers[3] = {first.fromLayer, first.toLayer, second.toLayer};
  for (int iCandidate = blockIdx.x * blockDim.x + threadIdx.x; iCandidate < nCandidates; iCandidate += blockDim.x * gridDim.x) {
    const CellCandidate candidate = candidates[iCandidate];
    const Tracklet& currentTracklet = tracklets[cellTopology.firstLink][candidate.firstTrackletIndex];
    const Tracklet& nextTracklet = tracklets[cellTopology.secondLink][candidate.secondTrackletIndex];
    const int clusId[3]{
      sortedClusters[layers[0]][currentTracklet.firstClusterIndex].clusterId,
      sortedClusters[layers[1]][nextTracklet.firstClusterIndex].clusterId,
      sortedClusters[layers[2]][nextTracklet.secondClusterIndex].clusterId};

    const auto& cluster1Glo = unsortedClusters[layers[0]][clusId[0]];
    const auto& cluster2Glo = unsortedClusters[layers[1]][clusId[1]];
    const auto& cluster3Tf = tfInfo[layers[2]][clusId[2]];
    auto track{o2::its::track::buildTrackSeed(cluster1Glo, cluster2Glo, cluster3Tf, bz)};
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
    TimeEstBC ts = currentTracklet.getTimeStamp();
    ts += nextTracklet.getTimeStamp();
    const int outputIndex = atomicAdd(outputCounter, 1);
    if (outputIndex < outputCapacity) {
      new (cells + outputIndex) CellSeed{cellTopology.hitLayerMask, clusId[0], clusId[1], clusId[2], candidate.firstTrackletIndex, candidate.secondTrackletIndex, track, chi2, ts};
    }
  }
}

template <int NLayers>
GPUg() void __launch_bounds__(GPUThreads, MinBlocks.computeLayerTracklets) computeLayerTrackletsMultiROFKernel(
  const IndexTableUtils<NLayers>* utils,
  const typename ROFMaskTable<NLayers>::View rofMask,
  const int linkId,
  const typename TrackingTopology<NLayers>::View topology,
  const typename ROFOverlapTable<NLayers>::View rofOverlaps,
  const typename ROFVertexLookupTable<NLayers>::View vertexLUT,
  const Vertex* vertices,
  const bool vtxMode,
  const int vertexId,
  const Cluster** clusters,
  const int** ROFClusters,
  const unsigned char** usedClusters,
  const int** indexTables,
  Tracklet** tracklets,
  int* outputCounter,
  const int outputCapacity,
  const bool selectUPCVertices,
  const float NSigmaCut,
  const float phiCut,
  const float resolutionPV,
  const float* minRs,
  const float* maxRs,
  const float positionResolution,
  const float meanDeltaR,
  const float MSAngle)
{
  const auto link = topology.getLink(linkId);
  const int fromLayer = link.fromLayer;
  const int toLayer = link.toLayer;
  const float minR = minRs[toLayer];
  const float maxR = maxRs[toLayer];
  const int phiBins{utils->getNphiBins()};
  const int zBins{utils->getNzBins()};
  const int tableSize{phiBins * zBins + 1};
  const int totalROFs0 = rofOverlaps.getLayer(fromLayer).mNROFsTF;
  const int totalROFs1 = rofOverlaps.getLayer(toLayer).mNROFsTF;
  if (totalROFs0 <= 0) {
    return;
  }

  const int* const rofOffsets = ROFClusters[fromLayer];
  const int totalClusters = rofOffsets[totalROFs0];
  for (int currentSortedIndex = blockIdx.x * blockDim.x + threadIdx.x;
       currentSortedIndex < totalClusters;
       currentSortedIndex += blockDim.x * gridDim.x) {
    // last ROF whose first cluster is at or before this one
    int lo{0}, hi{totalROFs0 - 1};
    while (lo < hi) {
      const int mid{(lo + hi + 1) >> 1};
      if (rofOffsets[mid] <= currentSortedIndex) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }
    const unsigned int pivotROF = static_cast<unsigned int>(lo);
    if (!rofMask.isROFEnabled(fromLayer, pivotROF)) {
      continue;
    }

    // The diamond is a single PV-independent vertex: the lookup table is not consulted at all, since during the seeding-vertex pass it holds no vertices yet
    gpuSpan<const Vertex> primaryVertices;
    if (vtxMode) {
      primaryVertices = gpuSpan<const Vertex>(vertices, 1);
    } else {
      const auto& pvs = vertexLUT.getVertices(fromLayer, pivotROF);
      primaryVertices = gpuSpan<const Vertex>(&vertices[pvs.getFirstEntry()], pvs.getEntries());
    }
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

    {
      const auto& currentCluster{clusters[fromLayer][currentSortedIndex]};
      if (usedClusters[fromLayer][currentCluster.clusterId]) {
        continue;
      }

      const float inverseR0{1.f / currentCluster.radius};
      for (int iV{startVtx}; iV < endVtx; ++iV) {
        auto& primaryVertex{primaryVertices[iV]};
        if (!vtxMode && !vertexLUT.isVertexCompatible(fromLayer, pivotROF, primaryVertex)) {
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
          if (!vtxMode && !ts.isCompatible(primaryVertex.getTimeStamp())) {
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
                const float phi{o2::math_utils::fastATan2(currentCluster.yCoordinate - nextCluster.yCoordinate, currentCluster.xCoordinate - nextCluster.xCoordinate)};
                const float tanL{(currentCluster.zCoordinate - nextCluster.zCoordinate) / (currentCluster.radius - nextCluster.radius)};
                const int nextSortedIndex{ROFClusters[toLayer][targetROF] + nextClusterIndex};
                const int outputIndex = atomicAdd(outputCounter, 1); // the optimizer turns this into a wave ballot vote
                if (outputIndex < outputCapacity) {
                  new (tracklets[linkId] + outputIndex) Tracklet{currentSortedIndex, nextSortedIndex, tanL, phi, ts};
                }
              }
            }
          }
        }
      }
    }
  }
}

GPUg() void __launch_bounds__(GPUThreads, MinBlocks.compileLookupTable) compileTrackletsLookupTableKernel(
  const Tracklet* tracklets,
  int* trackletsLookUpTable,
  const int nTracklets)
{
  for (int currentTrackletIndex = blockIdx.x * blockDim.x + threadIdx.x; currentTrackletIndex < nTracklets; currentTrackletIndex += blockDim.x * gridDim.x) {
    atomicAdd(&trackletsLookUpTable[tracklets[currentTrackletIndex].firstClusterIndex], 1);
  }
}

GPUg() void __launch_bounds__(GPUThreads, MinBlocks.compileLookupTable) compileLookupTableKernel(
  const int* keys,
  int* lookUpTable,
  const int nEntries)
{
  for (int currentEntry = blockIdx.x * blockDim.x + threadIdx.x; currentEntry < nEntries; currentEntry += blockDim.x * gridDim.x) {
    atomicAdd(&lookUpTable[keys[currentEntry]], 1);
  }
}

GPUg() void __launch_bounds__(GPUThreads, MinBlocks.compileLookupTable) compileCellNeighboursLookupTableKernel(
  const CellNeighbour* neighbours,
  int* neighboursLookUpTable,
  const int nNeighbours)
{
  for (int currentNeighbourIndex = blockIdx.x * blockDim.x + threadIdx.x; currentNeighbourIndex < nNeighbours; currentNeighbourIndex += blockDim.x * gridDim.x) {
    atomicAdd(&neighboursLookUpTable[neighbours[currentNeighbourIndex].nextCell], 1);
  }
}

struct trackletClusterKey {
  GPUhd() uint64_t operator()(const Tracklet& tracklet) const
  {
    return (static_cast<uint64_t>(tracklet.firstClusterIndex) << 32) | static_cast<uint32_t>(tracklet.secondClusterIndex);
  }
};

struct cellTrackletKey {
  GPUhd() uint64_t operator()(const CellSeed& cell) const
  {
    return (static_cast<uint64_t>(cell.getFirstTrackletIndex()) << 32) | static_cast<uint32_t>(cell.getSecondTrackletIndex());
  }
};

/// The first tracklet index recovered from a cellTrackletKey, for building the lookup table.
struct cellKeyFirstTracklet {
  GPUhd() int operator()(const uint64_t key) const { return static_cast<int>(key >> 32); }
};

struct cellNeighbourNextCell {
  GPUhd() int operator()(const CellNeighbour& neighbour) const { return neighbour.nextCell; }
};

struct cellNeighbourLess {
  GPUhd() bool operator()(const CellNeighbour& a, const CellNeighbour& b) const
  {
    if (a.nextCellTopology != b.nextCellTopology) {
      return a.nextCellTopology < b.nextCellTopology;
    }
    if (a.nextCell != b.nextCell) {
      return a.nextCell < b.nextCell;
    }
    if (a.cellTopology != b.cellTopology) {
      return a.cellTopology < b.cellTopology;
    }
    return a.cell < b.cell;
  }
};

/// A (current cell, neighbour-list entry) pair that passed every integer cut and is worth fitting.
struct NeighbourCandidate {
  int currentCell;
  int neighbourEntry;
};
template <int NLayers, typename CurrentSeed>
GPUg() void __launch_bounds__(GPUThreads, (std::is_same_v<CurrentSeed, CellSeed> ? MinBlocks.processNeighboursCellSeed : MinBlocks.processNeighboursTrackSeed)) processNeighbourCandidatesKernel(
  const int defaultCellTopologyId,
  const int level,
  CellSeed** allCellSeeds,
  CurrentSeed* currentCellSeeds,
  const int* currentCellIds,
  const int* currentCellTopologyIds,
  const unsigned int nCurrentCells,
  const unsigned char** usedClusters,
  CellNeighbour** neighbours,
  int** neighboursLUT,
  NeighbourCandidate* candidates, // nullptr on the counting pass
  int* candidateCounter,
  const int candidateCapacity) // 0 on the counting pass, so nothing is written
{
  for (unsigned int iCurrentCell = blockIdx.x * blockDim.x + threadIdx.x; iCurrentCell < nCurrentCells; iCurrentCell += blockDim.x * gridDim.x) {
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
      const auto& neighbourCell = allCellSeeds[neighbourRef.cellTopology][neighbourRef.cell];

      if (neighbourCell.getSecondTrackletIndex() != currentCell.getFirstTrackletIndex()) {
        continue;
      }
      if (!currentCell.getTimeStamp().isCompatible(neighbourCell.getTimeStamp())) {
        continue;
      }
      if (currentCell.getLevel() - 1 != neighbourCell.getLevel()) {
        continue;
      }
      if (usedClusters[neighbourCell.getInnerLayer()][neighbourCell.getFirstClusterIndex()]) {
        continue;
      }
      const int outputIndex = atomicAdd(candidateCounter, 1);
      if (outputIndex < candidateCapacity) {
        candidates[outputIndex] = {static_cast<int>(iCurrentCell), iNeighbourCell};
      }
    }
  }
}

template <int NLayers, typename CurrentSeed>
GPUg() void __launch_bounds__(GPUThreads, (std::is_same_v<CurrentSeed, CellSeed> ? MinBlocks.processNeighboursCellSeed : MinBlocks.processNeighboursTrackSeed)) fitNeighbourCandidatesKernel(
  const int defaultCellTopologyId,
  CellSeed** allCellSeeds,
  CurrentSeed* currentCellSeeds,
  const int* currentCellTopologyIds,
  const NeighbourCandidate* candidates,
  const int* candidateCounter,
  const int candidateCapacity,
  CellNeighbour** neighbours,
  TrackSeed<NLayers>* updatedCellSeeds,
  int* updatedCellsIds,
  int* updatedCellTopologyIds,
  int* updatedSourceSeeds,
  int* outputCounter,
  const int outputCapacity,
  const TrackingFrameInfo** foundTrackingFrameInfo,
  const float* layerxX0,
  const float bz,
  const float maxChi2ClusterAttachment,
  const o2::base::Propagator* propagator,
  const o2::base::PropagatorF::MatCorrType matCorrType)
{
  const int filled = *candidateCounter < candidateCapacity ? *candidateCounter : candidateCapacity;
  for (int iCandidate = blockIdx.x * blockDim.x + threadIdx.x; iCandidate < filled; iCandidate += blockDim.x * gridDim.x) {
    const NeighbourCandidate candidate = candidates[iCandidate];
    const unsigned int iCurrentCell = static_cast<unsigned int>(candidate.currentCell);
    const auto& currentCell{currentCellSeeds[iCurrentCell]};
    const int cellTopologyId = currentCellTopologyIds == nullptr ? defaultCellTopologyId : currentCellTopologyIds[iCurrentCell];
    const auto& neighbourRef = neighbours[cellTopologyId][candidate.neighbourEntry];
    const int neighbourCellTopologyId = neighbourRef.cellTopology;
    const int neighbourCellId = neighbourRef.cell;
    const auto& neighbourCell = allCellSeeds[neighbourCellTopologyId][neighbourCellId];
    const int neighbourLayer = neighbourCell.getInnerLayer();
    const int neighbourCluster = neighbourCell.getFirstClusterIndex();

    {
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
      seed.getClusters()[neighbourLayer] = neighbourCluster;
      auto mask = seed.getHitLayerMask();
      mask.set(neighbourLayer);
      seed.setHitLayerMask(mask);
      seed.setLevel(neighbourCell.getLevel());
      seed.setFirstTrackletIndex(neighbourCell.getFirstTrackletIndex());
      seed.setSecondTrackletIndex(neighbourCell.getSecondTrackletIndex());
      const int outputIndex = atomicAdd(outputCounter, 1);
      if (outputIndex < outputCapacity) {
        updatedCellsIds[outputIndex] = neighbourCellId;
        updatedCellTopologyIds[outputIndex] = neighbourCellTopologyId;
        updatedCellSeeds[outputIndex] = seed;
        if (updatedSourceSeeds != nullptr) {
          updatedSourceSeeds[outputIndex] = static_cast<int>(iCurrentCell);
        }
      }
    }
  }
}

/// Sort key that orders seeds by azimuth without mixing hit-layer patterns.
template <int NLayers>
GPUg() void __launch_bounds__(GPUThreads, MinBlocks.compileLookupTable) computeTrackSeedSortKeysKernel(
  const TrackSeed<NLayers>* trackSeeds,
  const unsigned int nSeeds,
  unsigned int* keys)
{
  static_assert(NLayers < 32, "the hit-layer pattern must leave room for the azimuth bits");
  constexpr int PhiShift{32 - NLayers};
  constexpr unsigned int PhiMask{(1u << PhiShift) - 1u};
  for (unsigned int iSeed = blockIdx.x * blockDim.x + threadIdx.x; iSeed < nSeeds; iSeed += blockDim.x * gridDim.x) {
    const auto& seed = trackSeeds[iSeed];
    const unsigned int hitPattern = seed.getHitLayerMask().value();
    const float phi = seed.getPhiPos(); // [0, 2pi)
    const unsigned int phiBin = static_cast<unsigned int>(phi * (static_cast<float>(PhiMask) / o2::constants::math::TwoPI)) & PhiMask;
    keys[iSeed] = (hitPattern << PhiShift) | phiBin;
  }
}

/// Sort key that orders neighbour candidates by the cluster their fit will read.
GPUg() void __launch_bounds__(GPUThreads, MinBlocks.compileLookupTable) computeNeighbourCandidateSortKeysKernel(
  const int defaultCellTopologyId,
  const int* currentCellTopologyIds,
  CellSeed** allCellSeeds,
  CellNeighbour** neighbours,
  const NeighbourCandidate* candidates,
  const int nCandidates,
  unsigned int* keys)
{
  constexpr unsigned int ClusterMask{0x07FFFFFFu};
  for (int iCandidate = blockIdx.x * blockDim.x + threadIdx.x; iCandidate < nCandidates; iCandidate += blockDim.x * gridDim.x) {
    const NeighbourCandidate candidate = candidates[iCandidate];
    const int cellTopologyId = currentCellTopologyIds == nullptr ? defaultCellTopologyId : currentCellTopologyIds[candidate.currentCell];
    const auto& neighbourRef = neighbours[cellTopologyId][candidate.neighbourEntry];
    const auto& neighbourCell = allCellSeeds[neighbourRef.cellTopology][neighbourRef.cell];
    keys[iCandidate] = (static_cast<unsigned int>(neighbourCell.getInnerLayer()) << 27) |
                       (static_cast<unsigned int>(neighbourCell.getFirstClusterIndex()) & ClusterMask);
  }
}

/// Order a candidate list in place by the hit each fit will read.
void sortNeighbourCandidates(const int defaultCellTopologyId,
                             const int* currentCellTopologyIds,
                             CellSeed** allCellSeeds,
                             CellNeighbour** neighbours,
                             NeighbourCandidate* candidates,
                             const int nCandidates,
                             o2::its::ExternalAllocator* alloc)
{
  auto keys = TypedAllocator<unsigned int>(alloc).allocate(nCandidates);
  auto policy = THRUST_NAMESPACE::par_nosync(TypedAllocator<char>(alloc)).on(Stream::DefaultStream);
  computeNeighbourCandidateSortKeysKernel<<<gridBlocks(ResidentBlocks.compileLookupTable), GPUThreads, 0, Stream::DefaultStream>>>(
    defaultCellTopologyId, currentCellTopologyIds, allCellSeeds, neighbours, candidates, nCandidates,
    thrust::raw_pointer_cast(keys));
  thrust::stable_sort_by_key(policy, keys, keys + nCandidates, thrust::device_ptr<NeighbourCandidate>(candidates));
}

GPUg() void vertexingRegisterCellClustersOwnership(
  const CellSeed* cells,
  const int nCells,
  unsigned long long** clusterOwners)
{
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < nCells; k += blockDim.x * gridDim.x) {
    const CellSeed& cell = cells[k];
    if (o2::gpu::CAMath::Abs(cell.getQ2Pt()) < o2::constants::math::Almost0 ||
        o2::gpu::CAMath::Abs(cell.getSnp()) > o2::constants::math::Almost1) {
      continue;
    }
    const float pt = cell.getPt();
    const float rank = pt > 1.e-6f ? 1.f / pt : 1.e9f;
    const unsigned long long key = (static_cast<unsigned long long>(__float_as_uint(rank)) << 32) | static_cast<unsigned long long>(k);
    o2::gpu::GPUCommonMath::AtomicMin(&clusterOwners[0][cell.getFirstClusterIndex()], key);
    o2::gpu::GPUCommonMath::AtomicMin(&clusterOwners[1][cell.getSecondClusterIndex()], key);
    o2::gpu::GPUCommonMath::AtomicMin(&clusterOwners[2][cell.getThirdClusterIndex()], key);
  }
}

GPUdi() int clusterROF(const int* rofArr, const int nRofs, const int clusterIdx)
{
  const int key = clusterIdx + 1;
  int lo = 0, hi = nRofs + 1;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (rofArr[mid] < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo - 1;
}

template <int NLayers>
GPUg() void dedupCellsKernel(
  const int nCells,
  const CellSeed* cells,
  const unsigned long long* const* clusterOwners,
  const int ownedClustersCut,
  const float beamX,
  const float beamY,
  const float maxZ,
  const float minPt,
  int* cellAccepted)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nCells; i += blockDim.x * gridDim.x) {
    const CellSeed& cell = cells[i];
    std::array<float, 3> origin, direction;
    if (!cell.getPxPyPzGlo(direction)) {
      cellAccepted[i] = 0;
      continue;
    }
    const bool owned0 = static_cast<uint32_t>(clusterOwners[0][cell.getFirstClusterIndex()]) == static_cast<uint32_t>(i);
    const bool owned1 = static_cast<uint32_t>(clusterOwners[1][cell.getSecondClusterIndex()]) == static_cast<uint32_t>(i);
    const bool owned2 = static_cast<uint32_t>(clusterOwners[2][cell.getThirdClusterIndex()]) == static_cast<uint32_t>(i);
    const bool keepCell = (static_cast<int>(owned0) + static_cast<int>(owned1) + static_cast<int>(owned2)) >= 3 - ownedClustersCut;
    cell.getXYZGlo(origin);
    const float dx = origin[0] - beamX;
    const float dy = origin[1] - beamY;
    const float den = direction[0] * direction[0] + direction[1] * direction[1];
    const bool projOk = den >= constants::Tolerance && o2::gpu::CAMath::Abs(origin[2] - (dx * direction[0] + dy * direction[1]) / den * direction[2]) < maxZ;
    const bool ptOk = minPt <= 0.f || cell.getPt() >= minPt;
    cellAccepted[i] = keepCell && projOk && ptOk ? 1 : 0;
  }
}

template <int NLayers>
GPUg() void linearizeCellsKernel(
  const int nCells,
  const CellSeed* cells,
  const int* rofFramesClustersL1, // layer-1 ROF boundaries, size nRofsL1 + 1
  const int nRofsL1,
  const int* lineSlots, // exclusive-scanned accept flags, size nCells + 1
  GPULine* lines,
  int* lineRof,
  const float beamX,
  const float beamY,
  float* lineZs,
  o2::its::TimeEstBC* lineTimes,
  int* lineClusters, // 3 per line (L0,L1,L2 cluster ids), for the host-side MC label derivation
  float* lineChi2,
  float* linePt)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nCells; i += blockDim.x * gridDim.x) {
    const int slot = lineSlots[i];
    if (slot == lineSlots[i + 1]) {
      continue;
    }
    const CellSeed& cell = cells[i];
    std::array<float, 3> origin, direction;
    cell.getXYZGlo(origin);
    cell.getPxPyPzGlo(direction);
    lines[slot] = GPULine{origin.data(), direction.data(), cell.getTimeStamp()};
    lineRof[slot] = clusterROF(rofFramesClustersL1, nRofsL1, cell.getSecondClusterIndex());
    const float dx = origin[0] - beamX;
    const float dy = origin[1] - beamY;
    const float den = direction[0] * direction[0] + direction[1] * direction[1];
    const float s0 = -(dx * direction[0] + dy * direction[1]) / den;
    lineZs[slot] = origin[2] + s0 * direction[2];
    lineTimes[slot] = cell.getTimeStamp();
    lineClusters[3 * slot + 0] = cell.getFirstClusterIndex();
    lineClusters[3 * slot + 1] = cell.getSecondClusterIndex();
    lineClusters[3 * slot + 2] = cell.getThirdClusterIndex();
    lineChi2[slot] = cell.getChi2();
    linePt[slot] = cell.getPt();
  }
}

template <int NLayers>
GPUg() void gatherSortedLinesKernel(const int nLines, LineProjSoA lineProj, LineProjSoA lineProjSorted)
{
  const int* sortedIdx = lineProj.idx;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nLines; i += blockDim.x * gridDim.x) {
    lineProjSorted.z[i] = lineProj.z[sortedIdx[i]];
    lineProjSorted.t[i] = lineProj.t[sortedIdx[i]];
    lineProjSorted.rof[i] = lineProj.rof[sortedIdx[i]];
  }
}

template <int NLayers>
GPUg() void scanDensityKernel(int* zDensity, LineWindow* win, const int nLines, const int* offsets, const LineProjSoA lineProjSorted, const float zWindow)
{
  const float* z = lineProjSorted.z;
  const o2::its::TimeEstBC* t = lineProjSorted.t;
  const int* rof = lineProjSorted.rof;
  for (int iLine = blockIdx.x * blockDim.x + threadIdx.x; iLine < nLines; iLine += blockDim.x * gridDim.x) {
    const int rofId = rof[iLine];
    const int rofOffset = offsets[rofId];
    const int nextRofOffset = offsets[rofId + 1];
    const float zk = z[iLine];
    const int lo = deviceLowerBound(z, rofOffset, nextRofOffset, zk - zWindow); // first line with z >= zk - zWindow
    const int hi = deviceUpperBound(z, rofOffset, nextRofOffset, zk + zWindow); // first line with z  > zk + zWindow
    win[iLine] = LineWindow{lo, hi};
    const auto ti = t[iLine];
    int count = 0;
    for (int j = lo; j < hi; ++j) {
      const auto tj = t[j];
      if (ti.isCompatible(tj)) { // count only if time compatible (includes self)
        ++count;
      }
    }
    zDensity[iLine] = count;
  }
}

template <int NLayers>
GPUg() void fitPeaksKernel(const int* nPeaksDevice,
                           const int* peakLineIdx,
                           const gpu::LineWindow* win,
                           const LineProjSoA lineProjSorted,
                           const GPULine* lines,
                           const float* lineChi2,       // global-indexed, same indexing as lines[]
                           const float* linePt,         // idem
                           const float goodLineChi2Cut, // a contributor counts towards nGood only if its own
                           const float goodLinePtCut,   // cell passes both; <= 0 disables that half
                           const float pairCut2,
                           const float nSigmaCut,
                           const int minContributors,
                           const float beamX,
                           const float beamY,
                           const uint8_t* isZPeakFine, // null when the fine pass is off
                           const float fineMaxDrift,   // <= 0 disables; see VertexerParamConfig::fineMaxDrift
                           VertexCand* cands)
{
  const int nPeaks = *nPeaksDevice;
  const o2::its::TimeEstBC* t = lineProjSorted.t;
  const int* idx = lineProjSorted.idx;
  for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < nPeaks; p += blockDim.x * gridDim.x) {
    cands[p].ok = 0;
    cands[p].nGood = 0;
    const int k = peakLineIdx[p];
    cands[p].fine = isZPeakFine != nullptr ? isZPeakFine[k] : 0;
    const auto tk = t[k];
    const LineWindow wk = win[k];

    GPUClusterLinesFit seed;
    int nMembers = 0;
    for (int j = wk.lo; j < wk.hi; ++j) {
      const auto tj = t[j];
      if (tk.isCompatible(tj)) {
        seed.add(lines[idx[j]]);
        ++nMembers;
      }
    }
    float seedVertex[3];
    if (nMembers < 2 || !seed.solve(seedVertex)) {
      continue;
    }

    GPUClusterLinesFit fit;
    int nKept = 0;
    int nGood = 0;
    for (int j = wk.lo; j < wk.hi; ++j) {
      const auto tj = t[j];
      if (tk.isCompatible(tj)) {
        const GPULine& line = lines[idx[j]];
        if (GPULine::getDistance2FromPoint(line, seedVertex) < pairCut2) {
          fit.add(line);
          const float c = lineChi2[idx[j]];
          const float pt = linePt[idx[j]];
          const bool okChi2 = (goodLineChi2Cut <= 0.f || c <= goodLineChi2Cut);
          const bool okPt = (goodLinePtCut <= 0.f || pt >= goodLinePtCut);
          nGood += okChi2 && okPt;
          ++nKept;
        }
      }
    }
    float vertex[3];
    if (nKept < 2 || !fit.solve(vertex)) {
      continue;
    }
    cands[p].seed[0] = seedVertex[0];
    cands[p].seed[1] = seedVertex[1];
    cands[p].seed[2] = seedVertex[2];
    const float bd2 = (beamX - vertex[0]) * (beamX - vertex[0]) + (beamY - vertex[1]) * (beamY - vertex[1]);
    if (nKept < minContributors || !(bd2 < nSigmaCut)) {
      continue;
    }
    if (fineMaxDrift > 0.f && cands[p].fine &&
        o2::gpu::GPUCommonMath::Abs(vertex[2] - seedVertex[2]) > fineMaxDrift) {
      continue;
    }

    for (int j = wk.lo; j < wk.hi; ++j) {
      const auto tj = t[j];
      if (tk.isCompatible(tj)) {
        const GPULine& line = lines[idx[j]];
        if (GPULine::getDistance2FromPoint(line, seedVertex) < pairCut2) {
          fit.addResidual(line, vertex);
        }
      }
    }

    cands[p].x = vertex[0];
    cands[p].y = vertex[1];
    cands[p].z = vertex[2];
    for (int i = 0; i < 6; ++i) {
      cands[p].rms2[i] = fit.getRMS2()[i];
    }
    cands[p].avgDist2 = fit.getAvgDistance2();
    cands[p].nGood = nGood;
    cands[p].time = fit.getTimeStamp();
    cands[p].size = nKept;
    cands[p].ok = 1;
  }
}

// Strict local maximum of the density over a line's own z-window, ties broken by smaller z
GPUdi() bool isDensityPeak(const int* density, const float* z, const LineWindow w, const int iLine)
{
  const int di = density[iLine];
  const float zi = z[iLine];
  for (int j = w.lo; j < w.hi; ++j) {
    const int dj = density[j];
    if (dj > di || (dj == di && z[j] < zi)) {
      return false;
    }
  }
  return true;
}

template <int NLayers>
GPUg() void findPeaksKernel(const int* zDensity, const LineWindow* win, const int nLines, const LineProjSoA lineProjSorted, uint8_t* isZPeak,
                            const int* zDensityFine, const LineWindow* winFine,
                            const int fineMinDensity, uint8_t* isZPeakFine)
{
  const float* z = lineProjSorted.z;
  for (int iLine = blockIdx.x * blockDim.x + threadIdx.x; iLine < nLines; iLine += blockDim.x * gridDim.x) {
    uint8_t peak = zDensity[iLine] >= 2 && isDensityPeak(zDensity, z, win[iLine], iLine);

    // fine pass: if the coarse pass did not find a peak, check if the fine density is above threshold and is a peak
    uint8_t fine = 0;
    if (!peak && zDensityFine != nullptr) {
      fine = zDensityFine[iLine] >= fineMinDensity && isDensityPeak(zDensityFine, z, winFine[iLine], iLine);
      peak = fine;
    }
    isZPeak[iLine] = peak;
    if (isZPeakFine != nullptr) {
      isZPeakFine[iLine] = fine;
    }
  }
}

template <int NLayers>
GPUg() void dedupVertexCandidatesKernel(const int* nPeaksDevice,
                                        const int* peakLineIdx,
                                        const int* peakOffsets,
                                        const LineProjSoA lineProjSorted,
                                        const float duplicateZCut,
                                        const float duplicateZScale,
                                        VertexCand* cands)
{
  const int nPeaks = *nPeaksDevice;
  for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < nPeaks; p += blockDim.x * gridDim.x) {
    cands[p].keep = 0; // every visited slot must be written: this array is never memset
    if (!cands[p].ok) {
      continue;
    }
    const int r = lineProjSorted.rof[peakLineIdx[p]];
    const float zp = cands[p].z;
    const int sp = cands[p].size;
    float radius = duplicateZCut;
    if (duplicateZScale > 0.f && sp > 0) {
      radius = duplicateZScale / o2::gpu::GPUCommonMath::Sqrt((float)sp);
    }
    const auto tp = cands[p].time;
    uint8_t survive = 1;
    for (int q = peakOffsets[r]; q < peakOffsets[r + 1] && survive; ++q) {
      if (q == p || !cands[q].ok) {
        continue;
      }
      if (!tp.isCompatible(cands[q].time)) {
        continue;
      }
      if (o2::gpu::GPUCommonMath::Abs(zp - cands[q].z) >= radius) {
        continue;
      }
      const int sq = cands[q].size;
      if (sq > sp || (sq == sp && q < p)) {
        survive = 0;
      }
    }
    cands[p].keep = survive;
  }
}

template <int NLayers>
GPUg() void emitKeysForClusterSortingKernel(const Cluster* unsorted,
                                            const int* clusterOffsets, // this layer, size nRofs+1
                                            const IndexTableUtils<NLayers>* utils,
                                            const typename ROFMaskTable<NLayers>::View rofMask,
                                            float beamX, float beamY,
                                            int zBins, int phiBins, int nRofs, int iLayer,
                                            float* minRadiusLayer, float* maxRadiusLayer,
                                            int* keys)
{
  const int numBins = zBins * phiBins;
  for (int iROF = blockIdx.x; iROF < nRofs; iROF += gridDim.x) {
    const bool enabled = rofMask.isROFEnabled(iLayer, iROF);
    const int start = clusterOffsets[iROF];
    const int n = clusterOffsets[iROF + 1] - start;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const Cluster& c = unsorted[start + i];
      const float x = c.xCoordinate - beamX, y = c.yCoordinate - beamY;
      const float phi = math_utils::computePhi(x, y);
      int zBin = utils->getZBinIndex(iLayer, c.zCoordinate);
      zBin = o2::gpu::GPUCommonMath::Max(0, o2::gpu::GPUCommonMath::Min(zBin, zBins - 1)); // TODO: count bogus (clamped) if hasBogusClusters() is ever needed
      const int bin = utils->getBinIndex(zBin, utils->getPhiBinIndex(phi));
      if (enabled) {
        const float r = math_utils::hypot(x, y);
        o2::gpu::GPUCommonMath::AtomicMin(&minRadiusLayer[iLayer], r);
        o2::gpu::GPUCommonMath::AtomicMax(&maxRadiusLayer[iLayer], r);
      }
      keys[start + i] = iROF * numBins + bin;
    }
  }
}

template <int NLayers>
GPUg() void gatherSortedClustersKernel(const Cluster* unsorted,
                                       Cluster* sorted,
                                       const int* perm,
                                       const IndexTableUtils<NLayers>* utils,
                                       float beamX, float beamY,
                                       int zBins, int nClustersLayer, int iLayer)
{
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < nClustersLayer; j += blockDim.x * gridDim.x) {
    Cluster c = unsorted[perm[j]];
    const float x = c.xCoordinate - beamX, y = c.yCoordinate - beamY;
    const float phi = math_utils::computePhi(x, y);
    int zBin = utils->getZBinIndex(iLayer, c.zCoordinate);
    zBin = o2::gpu::GPUCommonMath::Max(0, o2::gpu::GPUCommonMath::Min(zBin, zBins - 1));
    c.phi = phi;
    c.radius = math_utils::hypot(x, y);
    c.indexTableBinIndex = utils->getBinIndex(zBin, utils->getPhiBinIndex(phi));
    sorted[j] = c;
  }
}

template <int NLayers>
GPUg() void buildClusterIndexTableKernel(const int* sortedKeys,
                                         const int* clusterOffsets, // this layer, size nRofs+1
                                         int* indexTable,           // output, size nRofs*(numBins+1)
                                         int numBins, int nRofs)
{
  const int stride = numBins + 1;
  for (int iROF = blockIdx.x; iROF < nRofs; iROF += gridDim.x) {
    const int rofStart = clusterOffsets[iROF];
    const int rofEnd = clusterOffsets[iROF + 1];
    int* base = indexTable + iROF * stride;
    for (int b = threadIdx.x; b <= numBins; b += blockDim.x) {
      const int keyB = iROF * numBins + b;
      base[b] = deviceLowerBound(sortedKeys, rofStart, rofEnd, keyB) - rofStart; // ROF-local
    }
  }
}

} // namespace gpu

template <int NLayers>
int TrackingKernels<NLayers>::computeTrackletsInROFsHandler(const IndexTableUtils<NLayers>* utils,
                                                            const typename ROFMaskTable<NLayers>::View& rofMask,
                                                            const int linkId,
                                                            const int fromLayer,
                                                            const int toLayer,
                                                            const typename ROFOverlapTable<NLayers>::View& rofOverlaps,
                                                            const typename ROFVertexLookupTable<NLayers>::View& vertexLUT,
                                                            const int vertexId,
                                                            const Vertex* vertices,
                                                            const bool vtxMode,
                                                            const Cluster** clusters,
                                                            const std::vector<unsigned int>& nClusters,
                                                            const int** ROFClusters,
                                                            const unsigned char** usedClusters,
                                                            const int** clustersIndexTables,
                                                            Tracklet** tracklets,
                                                            gsl::span<Tracklet*> spanTracklets,
                                                            gsl::span<int> nTracklets,
                                                            const int capacity,
                                                            gsl::span<int*> trackletsLUTsHost,
                                                            const bool selectUPCVertices,
                                                            const float NSigmaCut,
                                                            const typename TrackingTopology<NLayers>::View topology,
                                                            bounded_vector<float>& linkPhiCuts,
                                                            const float resolutionPV,
                                                            const float* minRs,
                                                            const float* maxRs,
                                                            bounded_vector<float>& resolutions,
                                                            std::vector<float>& radii,
                                                            bounded_vector<float>& linkMSAngles,
                                                            o2::its::ExternalAllocator* alloc,
                                                            gpu::Streams& streams)
{
  int emitted = 0;
  int* outputCounter = trackletsLUTsHost[linkId] + nClusters[fromLayer];
  GPUChkErrS(cudaMemsetAsync(outputCounter, 0, sizeof(int), streams[linkId].get()));
  gpu::computeLayerTrackletsMultiROFKernel<NLayers><<<gpu::gridBlocks(gpu::ResidentBlocks.computeLayerTracklets), gpu::GPUThreads, 0, streams[linkId].get()>>>(
    utils,
    rofMask,
    linkId,
    topology,
    rofOverlaps,
    vertexLUT,
    vertices,
    vtxMode,
    vertexId,
    clusters,
    ROFClusters,
    usedClusters,
    clustersIndexTables,
    tracklets,
    outputCounter,
    capacity,
    selectUPCVertices,
    NSigmaCut,
    linkPhiCuts[linkId],
    resolutionPV,
    minRs,
    maxRs,
    resolutions[fromLayer],
    radii[toLayer] - radii[fromLayer],
    linkMSAngles[linkId]);
  GPUChkErrS(cudaMemcpyAsync(&emitted, outputCounter, sizeof(int), cudaMemcpyDeviceToHost, streams[linkId].get()));
  streams[linkId].sync();
  if (emitted > capacity) {
    return emitted;
  }
  nTracklets[linkId] = emitted;
  auto nosync_policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(streams[linkId].get());
  if (emitted > 0) {
    thrust::device_ptr<Tracklet> trackletsPtr(spanTracklets[linkId]);
    constexpr uint64_t SortTag = qStr2Tag("ITSTRKSR");
    alloc->pushTagOnStack(SortTag);
    auto keys = gpu::TypedAllocator<uint64_t>(alloc).allocate(emitted);
    thrust::transform(nosync_policy, trackletsPtr, trackletsPtr + emitted, keys, gpu::trackletClusterKey{});
    thrust::sort_by_key(nosync_policy, keys, keys + emitted, trackletsPtr);
    if (vertexId < 0) {
      auto uniqueEnd = thrust::unique_by_key(nosync_policy, keys, keys + emitted, trackletsPtr);
      nTracklets[linkId] = uniqueEnd.first - keys;
    }
    streams[linkId].sync();
    alloc->popTagOffStack(SortTag);
  }
  GPUChkErrS(cudaMemsetAsync(trackletsLUTsHost[linkId], 0, (nClusters[fromLayer] + 1) * sizeof(int), streams[linkId].get()));
  if (nTracklets[linkId] == 0) {
    return emitted;
  }
  gpu::compileTrackletsLookupTableKernel<<<gpu::gridBlocks(gpu::ResidentBlocks.compileLookupTable), gpu::GPUThreads, 0, streams[linkId].get()>>>(
    spanTracklets[linkId],
    trackletsLUTsHost[linkId],
    nTracklets[linkId]);
  thrust::exclusive_scan(nosync_policy, trackletsLUTsHost[linkId], trackletsLUTsHost[linkId] + nClusters[fromLayer] + 1, trackletsLUTsHost[linkId]);
  return emitted;
}

template <int NLayers>
int TrackingKernels<NLayers>::computeCellsHandler(
  const Cluster** sortedClusters,
  const Cluster** unsortedClusters,
  const TrackingFrameInfo** tfInfo,
  Tracklet** tracklets,
  int** trackletsLUT,
  const int nTracklets,
  const int cellTopologyId,
  const typename TrackingTopology<NLayers>::View topology,
  CellSeed* cells,
  const int capacity,
  int* cellsLUTsHost,
  const float bz,
  const float maxChi2ClusterAttachment,
  const float cellDeltaTanLambdaSigma,
  const float cellDeltaPhiCut,
  const float nSigmaCut,
  const float* layerxX0,
  o2::its::ExternalAllocator* alloc,
  gpu::Streams& streams)
{
  int emitted = 0;
  auto& stream = streams[cellTopologyId];
  int* outputCounter = cellsLUTsHost + nTracklets;

  constexpr uint64_t CandidateTag = qStr2Tag("ITSCELCA");
  alloc->pushTagOnStack(CandidateTag);
  gpu::TypedAllocator<gpu::CellCandidate> candidateAllocator(alloc);

  const int candidateBlocks = gpu::gridBlocks(gpu::ResidentBlocks.computeLayerCells);
  GPUChkErrS(cudaMemsetAsync(outputCounter, 0, sizeof(int), stream.get()));
  gpu::computeLayerCellCandidatesKernel<NLayers, false><<<candidateBlocks, gpu::GPUThreads, 0, stream.get()>>>(
    tracklets, trackletsLUT, nTracklets, cellTopologyId, topology,
    sortedClusters, unsortedClusters, tfInfo, layerxX0, bz,
    nullptr, nullptr, outputCounter, 0, cellDeltaTanLambdaSigma, cellDeltaPhiCut, nSigmaCut);
  int nCandidates = 0;
  GPUChkErrS(cudaMemcpyAsync(&nCandidates, outputCounter, sizeof(int), cudaMemcpyDeviceToHost, stream.get()));
  stream.sync();

  if (nCandidates == 0) {
    GPUChkErrS(cudaMemsetAsync(cellsLUTsHost, 0, (nTracklets + 1) * sizeof(int), stream.get()));
    stream.sync();
    alloc->popTagOffStack(CandidateTag);
    return 0;
  }

  auto candidates = candidateAllocator.allocate(nCandidates);
  gpu::TypedAllocator<unsigned int> candidateKeyAllocator(alloc);
  auto candidateKeys = candidateKeyAllocator.allocate(nCandidates);
  GPUChkErrS(cudaMemsetAsync(outputCounter, 0, sizeof(int), stream.get()));
  gpu::computeLayerCellCandidatesKernel<NLayers, true><<<candidateBlocks, gpu::GPUThreads, 0, stream.get()>>>(
    tracklets, trackletsLUT, nTracklets, cellTopologyId, topology,
    sortedClusters, unsortedClusters, tfInfo, layerxX0, bz,
    thrust::raw_pointer_cast(candidates), thrust::raw_pointer_cast(candidateKeys),
    outputCounter, nCandidates, cellDeltaTanLambdaSigma, cellDeltaPhiCut, nSigmaCut);

  // order the candidates by momentum before fitting them, so that the ELoss iteration count inside is uniform
  {
    auto candidatePolicy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(stream.get());
    thrust::sort_by_key(candidatePolicy, candidateKeys, candidateKeys + nCandidates, candidates);
  }

  GPUChkErrS(cudaMemsetAsync(outputCounter, 0, sizeof(int), stream.get()));
  gpu::fitLayerCellsKernel<NLayers><<<candidateBlocks, gpu::GPUThreads, 0, stream.get()>>>(
    sortedClusters, unsortedClusters, tfInfo, tracklets,
    thrust::raw_pointer_cast(candidates), nCandidates,
    cellTopologyId, topology, cells, outputCounter, capacity,
    layerxX0, bz, maxChi2ClusterAttachment);
  GPUChkErrS(cudaMemcpyAsync(&emitted, outputCounter, sizeof(int), cudaMemcpyDeviceToHost, stream.get()));
  stream.sync();
  alloc->popTagOffStack(CandidateTag);

  if (emitted > capacity) {
    return emitted;
  }

  auto nosync_policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(stream.get());
  GPUChkErrS(cudaMemsetAsync(cellsLUTsHost, 0, (nTracklets + 1) * sizeof(int), stream.get()));
  if (emitted == 0) {
    return emitted;
  }
  constexpr uint64_t SortTag = qStr2Tag("ITSCELSR");
  alloc->pushTagOnStack(SortTag);
  gpu::TypedAllocator<int> keyAllocator(alloc);
  gpu::TypedAllocator<uint64_t> sortKeyAllocator(alloc);
  gpu::TypedAllocator<CellSeed> cellAllocator(alloc);
  auto keys = sortKeyAllocator.allocate(emitted);
  auto permutation = keyAllocator.allocate(emitted);
  thrust::device_ptr<CellSeed> cellsPtr(cells);
  thrust::transform(nosync_policy, cellsPtr, cellsPtr + emitted, keys, gpu::cellTrackletKey{});
  thrust::sequence(nosync_policy, permutation, permutation + emitted);
  thrust::stable_sort_by_key(nosync_policy, keys, keys + emitted, permutation);
  auto sortedCells = cellAllocator.allocate(emitted);
  thrust::gather(nosync_policy, permutation, permutation + emitted, cellsPtr, sortedCells);
  auto lutKeys = keyAllocator.allocate(emitted);
  thrust::transform(nosync_policy, keys, keys + emitted, lutKeys, gpu::cellKeyFirstTracklet{});
  gpu::compileLookupTableKernel<<<gpu::gridBlocks(gpu::ResidentBlocks.compileLookupTable), gpu::GPUThreads, 0, stream.get()>>>(thrust::raw_pointer_cast(lutKeys),
                                                                                                                               cellsLUTsHost,
                                                                                                                               emitted);
  thrust::exclusive_scan(nosync_policy, cellsLUTsHost, cellsLUTsHost + nTracklets + 1, cellsLUTsHost);
  GPUChkErrS(cudaMemcpyAsync(cells, thrust::raw_pointer_cast(sortedCells), emitted * sizeof(CellSeed), cudaMemcpyDeviceToDevice, stream.get()));
  stream.sync();
  alloc->popTagOffStack(SortTag);
  return emitted;
}

void resetOutputCounterHandler(int* outputCounter, gpu::Stream& stream)
{
  GPUChkErrS(cudaMemsetAsync(outputCounter, 0, sizeof(int), stream.get()));
}

template <int NLayers>
void TrackingKernels<NLayers>::computeCellNeighboursHandler(CellSeed** cellsLayersDevice,
                                                            int** cellsLUTs,
                                                            CellNeighbour* cellNeighbours,
                                                            int* outputCounter,
                                                            const int capacity,
                                                            const int sourceCellTopologyId,
                                                            const int targetCellTopologyId,
                                                            const float maxChi2ClusterAttachment,
                                                            const float bz,
                                                            const unsigned int nCells,
                                                            o2::its::ExternalAllocator* alloc,
                                                            gpu::Stream& stream)
{
  const int neighbourBlocks = gpu::gridBlocks(gpu::ResidentBlocks.computeLayerCellNeighbours);

  constexpr uint64_t CandidateTag = qStr2Tag("ITSNGHCA");
  alloc->pushTagOnStack(CandidateTag);
  gpu::TypedAllocator<gpu::CellNeighbourCandidate> candidateAllocator(alloc);
  gpu::TypedAllocator<int> counterAllocator(alloc);

  auto candidateCounter = counterAllocator.allocate(1);
  int* candidateCounterPtr = thrust::raw_pointer_cast(candidateCounter);

  GPUChkErrS(cudaMemsetAsync(candidateCounterPtr, 0, sizeof(int), stream.get()));
  gpu::computeLayerCellNeighbourCandidatesKernel<NLayers><<<neighbourBlocks, gpu::GPUThreads, 0, stream.get()>>>(
    cellsLayersDevice, cellsLUTs, sourceCellTopologyId, targetCellTopologyId,
    nullptr, // counting pass: capacity 0, so nothing is written
    candidateCounterPtr, 0, nCells);
  int nCandidates = 0;
  GPUChkErrS(cudaMemcpyAsync(&nCandidates, candidateCounterPtr, sizeof(int), cudaMemcpyDeviceToHost, stream.get()));
  stream.sync();

  if (nCandidates == 0) {
    alloc->popTagOffStack(CandidateTag);
    return;
  }

  auto candidates = candidateAllocator.allocate(nCandidates);
  GPUChkErrS(cudaMemsetAsync(candidateCounterPtr, 0, sizeof(int), stream.get()));
  gpu::computeLayerCellNeighbourCandidatesKernel<NLayers><<<neighbourBlocks, gpu::GPUThreads, 0, stream.get()>>>(
    cellsLayersDevice, cellsLUTs, sourceCellTopologyId, targetCellTopologyId,
    thrust::raw_pointer_cast(candidates), candidateCounterPtr, nCandidates, nCells);

  gpu::fitCellNeighboursKernel<NLayers><<<neighbourBlocks, gpu::GPUThreads, 0, stream.get()>>>(
    cellsLayersDevice,
    thrust::raw_pointer_cast(candidates),
    nCandidates,
    cellNeighbours,
    outputCounter,
    capacity,
    sourceCellTopologyId,
    targetCellTopologyId,
    maxChi2ClusterAttachment,
    bz);

  stream.sync(); // the candidate slab must outlive the kernels reading it
  alloc->popTagOffStack(CandidateTag);
}

template <int NLayers>
void TrackingKernels<NLayers>::sortClustersHandler(const Cluster* unsorted,   // this layer (resident unsorted)
                                                   Cluster* sorted,           // this layer (output)
                                                   const int* clusterOffsets, // this layer ROF boundaries, size nRofs+1
                                                   int* indexTable,           // this layer (output), size nRofs*(zBins*phiBins+1)
                                                   const IndexTableUtils<NLayers>* utils,
                                                   const typename ROFMaskTable<NLayers>::View& rofMask,
                                                   float beamX, float beamY,
                                                   int zBins, int phiBins, int nRofs, int nClustersLayer, int iLayer,
                                                   float* minRadiusLayer, float* maxRadiusLayer, // per-layer device arrays
                                                   int* keys,                                    // scratch, size nClustersLayer
                                                   int* perm,                                    // scratch, size nClustersLayer
                                                   o2::its::ExternalAllocator* alloc,
                                                   gpu::Stream& stream)
{
  if (nClustersLayer == 0) {
    return;
  }
  const int numBins = zBins * phiBins;
  auto policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(stream.get());
  thrust::fill_n(policy, minRadiusLayer + iLayer, 1, std::numeric_limits<float>::max());
  thrust::fill_n(policy, maxRadiusLayer + iLayer, 1, std::numeric_limits<float>::min());

  gpu::emitKeysForClusterSortingKernel<NLayers><<<gpu::gridBlocks(gpu::DefaultBlocksPerComputeUnit), gpu::GPUThreads, 0, stream.get()>>>(
    unsorted, clusterOffsets, utils, rofMask, beamX, beamY, zBins, phiBins, nRofs, iLayer,
    minRadiusLayer, maxRadiusLayer, keys);

  thrust::sequence(policy, perm, perm + nClustersLayer);
  thrust::stable_sort_by_key(policy, keys, keys + nClustersLayer, perm);

  gpu::gatherSortedClustersKernel<NLayers><<<gpu::gridBlocks(gpu::DefaultBlocksPerComputeUnit), gpu::GPUThreads, 0, stream.get()>>>(
    unsorted, sorted, perm, utils, beamX, beamY, zBins, nClustersLayer, iLayer);

  gpu::buildClusterIndexTableKernel<NLayers><<<gpu::gridBlocks(gpu::DefaultBlocksPerComputeUnit), gpu::GPUThreads, 0, stream.get()>>>(
    keys, clusterOffsets, indexTable, numBins, nRofs);
}

template <int NLayers>
void TrackingKernels<NLayers>::registerClusterOwnershipHandler(const CellSeed* cells,
                                                               const int nCells,
                                                               unsigned long long** clusterOwnersDeviceArray,
                                                               gpu::Stream& stream)
{

  gpu::vertexingRegisterCellClustersOwnership<<<gpu::gridBlocks(gpu::DefaultBlocksPerComputeUnit), gpu::GPUThreads, 0, stream.get()>>>(
    cells,
    nCells,
    clusterOwnersDeviceArray);
}

template <int NLayers>
void TrackingKernels<NLayers>::linearizeCellsToLinesHandler(const int nCells,
                                                            const CellSeed* cells,
                                                            const unsigned long long* const* clusterOwners,
                                                            const int* rofFramesClustersL1,
                                                            const int nRofsL1,
                                                            const int ownedClustersCut,
                                                            gpu::GPULine* lines,
                                                            int* lineRof,
                                                            int* lineClusters,
                                                            int* lineSlots, // nCells + 1 scratch: accept flags, scanned in place into slots
                                                            const float beamX,
                                                            const float beamY,
                                                            const float maxZ,
                                                            const float minPt,
                                                            float* linesZs,
                                                            o2::its::TimeEstBC* lineTimes,
                                                            float* lineChi2,
                                                            float* linePt,
                                                            o2::its::ExternalAllocator* alloc,
                                                            gpu::Stream& stream)
{
  gpu::dedupCellsKernel<NLayers><<<gpu::gridBlocks(gpu::DefaultBlocksPerComputeUnit), gpu::GPUThreads, 0, stream.get()>>>(
    nCells,
    cells,
    clusterOwners,
    ownedClustersCut,
    beamX,
    beamY,
    maxZ,
    minPt,
    lineSlots);
  auto nosync_policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(stream.get());
  thrust::exclusive_scan(nosync_policy, lineSlots, lineSlots + nCells + 1, lineSlots);
  gpu::linearizeCellsKernel<NLayers><<<gpu::gridBlocks(gpu::DefaultBlocksPerComputeUnit), gpu::GPUThreads, 0, stream.get()>>>(
    nCells,
    cells,
    rofFramesClustersL1,
    nRofsL1,
    lineSlots,
    lines,
    lineRof,
    beamX,
    beamY,
    linesZs,
    lineTimes,
    lineClusters,
    lineChi2,
    linePt);
}

// Orders lines by (ROF, z): primary key the ROF, secondary the projected z within the ROF.
struct RofZLess {
  const int* rof;
  const float* z;
  GPUhdi() bool operator()(const int a, const int b) const
  {
    return rof[a] != rof[b] ? rof[a] < rof[b] : z[a] < z[b];
  }
};

template <int NLayers>
void TrackingKernels<NLayers>::sortLinesHandler(const int nLines,
                                                const int nRofs,
                                                const gpu::LineProjSoA soa,
                                                const gpu::LineProjSoA sortedSoa,
                                                const int* lineRof,
                                                int* rofOffsets,
                                                o2::its::ExternalAllocator* alloc,
                                                gpu::Stream& stream)
{
  if (nLines < 2) {
    return;
  }
  auto policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(stream.get());
  thrust::sequence(policy, soa.idx, soa.idx + nLines);
  thrust::sort(policy, soa.idx, soa.idx + nLines, RofZLess{lineRof, soa.z});
  gpu::gatherSortedLinesKernel<NLayers><<<gpu::gridBlocks(gpu::DefaultBlocksPerComputeUnit), gpu::GPUThreads, 0, stream.get()>>>(nLines, soa, sortedSoa);
  auto rofSorted = thrust::make_permutation_iterator(lineRof, soa.idx);
  thrust::lower_bound(policy, rofSorted, rofSorted + nLines,
                      thrust::make_counting_iterator(0), thrust::make_counting_iterator(nRofs + 1),
                      rofOffsets);
}

template <int NLayers>
void TrackingKernels<NLayers>::scanDensityHandler(const int nLines,
                                                  const gpu::LineProjSoA sortedSoa,
                                                  const int* rofOffsets,
                                                  int* density,
                                                  gpu::LineWindow* win,
                                                  const float zWindow,
                                                  gpu::Stream& stream)
{
  if (nLines < 2) {
    return;
  }
  gpu::scanDensityKernel<NLayers><<<gpu::gridBlocks(gpu::DefaultBlocksPerComputeUnit), gpu::GPUThreads, 0, stream.get()>>>(density, win, nLines, rofOffsets, sortedSoa, zWindow);
}

template <int NLayers>
void TrackingKernels<NLayers>::findPeaksHandler(const int nLines,
                                                const int nRofs,
                                                const gpu::LineProjSoA sortedSoa,
                                                const int* rofOffsets,
                                                const int* density,
                                                const gpu::LineWindow* win,
                                                uint8_t* isPeak,
                                                const int* densityFine,
                                                const gpu::LineWindow* winFine,
                                                const int fineMinDensity,
                                                uint8_t* isPeakFine,
                                                int* peakScan,
                                                int* peakLineIdx,
                                                int* peakOffsets,
                                                o2::its::ExternalAllocator* alloc,
                                                gpu::Stream& stream)
{
  if (nLines < 2) {
    return;
  }
  gpu::findPeaksKernel<NLayers><<<gpu::gridBlocks(gpu::DefaultBlocksPerComputeUnit), gpu::GPUThreads, 0, stream.get()>>>(density, win, nLines, sortedSoa, isPeak,
                                                                                                                         densityFine, winFine, fineMinDensity, isPeakFine);
  auto nosync_policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(stream.get());
  thrust::exclusive_scan(nosync_policy, isPeak, isPeak + nLines + 1, peakScan, 0, thrust::plus<int>());
  thrust::scatter_if(nosync_policy, thrust::make_counting_iterator(0), thrust::make_counting_iterator(nLines),
                     peakScan, isPeak, peakLineIdx);
  thrust::gather(nosync_policy, rofOffsets, rofOffsets + nRofs + 1, peakScan, peakOffsets);
}

template <int NLayers>
void TrackingKernels<NLayers>::fitPeaksHandler(const int* nPeaksDevice,
                                               const int* peakLineIdx,
                                               const gpu::LineWindow* win,
                                               const gpu::LineProjSoA sortedSoa,
                                               const gpu::GPULine* lines,
                                               const float* lineChi2,
                                               const float* linePt,
                                               const float goodLineChi2Cut,
                                               const float goodLinePtCut,
                                               const float pairCut2,
                                               const float nSigmaCut,
                                               const int minContributors,
                                               const float beamX,
                                               const float beamY,
                                               const uint8_t* isPeakFine,
                                               const float fineMaxDrift,
                                               gpu::VertexCand* cands,
                                               gpu::Stream& stream)
{
  gpu::fitPeaksKernel<NLayers><<<gpu::gridBlocks(gpu::DefaultBlocksPerComputeUnit), gpu::GPUThreads, 0, stream.get()>>>(
    nPeaksDevice, peakLineIdx, win, sortedSoa, lines, lineChi2, linePt, goodLineChi2Cut, goodLinePtCut, pairCut2, nSigmaCut, minContributors, beamX, beamY, isPeakFine, fineMaxDrift, cands);
}

template <int NLayers>
void TrackingKernels<NLayers>::dedupVertexCandidatesHandler(const int* nPeaksDevice,
                                                            const int* peakLineIdx,
                                                            const int* peakOffsets,
                                                            const gpu::LineProjSoA sortedSoa,
                                                            const float duplicateZCut,
                                                            const float duplicateZScale,
                                                            gpu::VertexCand* cands,
                                                            gpu::Stream& stream)
{
  gpu::dedupVertexCandidatesKernel<NLayers><<<gpu::gridBlocks(gpu::DefaultBlocksPerComputeUnit), gpu::GPUThreads, 0, stream.get()>>>(
    nPeaksDevice, peakLineIdx, peakOffsets, sortedSoa, duplicateZCut, duplicateZScale, cands);
}

int finalizeCellNeighboursHandler(CellNeighbour* cellNeighbours,
                                  int* neighboursLUT,
                                  const int nTargetCells,
                                  const int capacity,
                                  o2::its::ExternalAllocator* alloc,
                                  gpu::Stream& stream)
{
  int emitted = 0;
  int* outputCounter = neighboursLUT + nTargetCells;
  GPUChkErrS(cudaMemcpyAsync(&emitted, outputCounter, sizeof(int), cudaMemcpyDeviceToHost, stream.get()));
  stream.sync();
  if (emitted > capacity) {
    return emitted;
  }
  auto nosync_policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(stream.get());
  if (emitted > 0) {
    thrust::device_ptr<CellNeighbour> neighboursPtr(cellNeighbours);
    constexpr uint64_t SortTag = qStr2Tag("ITSNGHSR");
    alloc->pushTagOnStack(SortTag);
#ifdef GPUCA_DETERMINISTIC_MODE
    thrust::sort(nosync_policy, neighboursPtr, neighboursPtr + emitted, gpu::cellNeighbourLess{});
#else
    auto keys = gpu::TypedAllocator<int>(alloc).allocate(emitted);
    thrust::transform(nosync_policy, neighboursPtr, neighboursPtr + emitted, keys, gpu::cellNeighbourNextCell{});
    thrust::sort_by_key(nosync_policy, keys, keys + emitted, neighboursPtr);
#endif
    stream.sync();
    alloc->popTagOffStack(SortTag);
  }
  GPUChkErrS(cudaMemsetAsync(neighboursLUT, 0, (nTargetCells + 1) * sizeof(int), stream.get()));
  if (emitted == 0) {
    return emitted;
  }
  gpu::compileCellNeighboursLookupTableKernel<<<gpu::gridBlocks(gpu::ResidentBlocks.compileLookupTable), gpu::GPUThreads, 0, stream.get()>>>(
    cellNeighbours,
    neighboursLUT,
    emitted);
  thrust::exclusive_scan(nosync_policy, neighboursLUT, neighboursLUT + nTargetCells + 1, neighboursLUT);
  return emitted;
}

template <int NLayers>
void TrackingKernels<NLayers>::processNeighboursHandler(const int startLevel,
                                                        const int startCellTopologyId,
                                                        CellSeed** allCellSeeds,
                                                        CellSeed* currentCellSeeds,
                                                        const int* currentCellTopologyIds,
                                                        const int* currentCellIds,
                                                        const int* nCells,
                                                        const unsigned char** usedClusters,
                                                        CellNeighbour** neighbours,
                                                        int** neighboursDeviceLUTs,
                                                        const TrackingFrameInfo** foundTrackingFrameInfo,
                                                        TrackSeed<NLayers>* seedsDevice,
                                                        const int seedsCapacity,
                                                        int& seedsCursor,
                                                        CapacityEstimator& estimator,
                                                        const int iteration,
                                                        const float bz,
                                                        const float maxChi2ClusterAttachment,
                                                        const float maxChi2NDF,
                                                        const int maxHoles,
                                                        const int minSeedingClusters,
                                                        const LayerMask holeLayerMask,
                                                        const LayerMask nonSeedingLayerMask,
                                                        const float* layerxX0,
                                                        const o2::base::Propagator* propagator,
                                                        const o2::base::PropagatorF::MatCorrType matCorrType,
                                                        o2::its::ExternalAllocator* alloc)
{
  constexpr uint64_t Tag = qStr2Tag("ITS_PNH1");
  alloc->pushTagOnStack(Tag);
  auto allocInt = gpu::TypedAllocator<int>(alloc);
  auto allocTrackSeed = gpu::TypedAllocator<TrackSeed<NLayers>>(alloc);
  auto nosync_policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(gpu::Stream::DefaultStream);
  auto outputCounter = allocInt.allocate(1);

  auto roadKey = [&](const int level) {
    return CapacityEstimator::makeKey(SlabSite::Roads, iteration, CapacityEstimator::makeVariant(startLevel, level), startCellTopologyId);
  };
  auto candidateKey = [&](const int level) {
    return CapacityEstimator::makeKey(SlabSite::RoadCandidates, iteration, CapacityEstimator::makeVariant(startLevel, level), startCellTopologyId);
  };

  struct Slab {
    thrust::device_ptr<TrackSeed<NLayers>> seeds{};
    thrust::device_ptr<int> cellIds{};
    thrust::device_ptr<int> cellTopologyIds{};
    int capacity{0};
  };
  Slab slabs[2];
  auto ensureCapacity = [&](Slab& slab, const int capacity) {
    if (slab.capacity >= capacity) {
      return;
    }
    slab.seeds = allocTrackSeed.allocate(capacity);
    slab.cellIds = allocInt.allocate(capacity);
    slab.cellTopologyIds = allocInt.allocate(capacity);
    slab.capacity = capacity;
  };

  constexpr double SlabHeadroom = 1.3; // deliberately tighter than the estimator's adaptive margin
  size_t peak = 0;
  double waveScale = static_cast<double>(nCells[startCellTopologyId]);
  for (int level = startLevel; level >= 2 && waveScale > 0.; --level) {
    const double expected = estimator.expected(roadKey(level), waveScale);
    peak = std::max(peak, static_cast<size_t>(std::ceil(expected * SlabHeadroom)));
    waveScale = expected;
  }
  if (peak == 0) {
    peak = estimator.peakCapacity(roadKey(startLevel));
  }
  const int slabCapacity = static_cast<int>(std::min(peak, static_cast<size_t>(std::numeric_limits<int>::max())));
  ensureCapacity(slabs[0], slabCapacity);
  ensureCapacity(slabs[1], slabCapacity);

  int filled = -1; // slab holding the wave that was produced last
  int nWaveSeeds = 0;

  auto processLevel = [&](auto* levelSeeds, const int* levelCellIds, const int* levelCellTopologyIds,
                          const unsigned int nLevelSeeds, const int level, const int topologyId) {
    const int outIdx = filled == 0 ? 1 : 0;
    Slab& out = slabs[outIdx];
    thrust::device_ptr<TrackSeed<NLayers>> staged{};
    thrust::device_ptr<int> stagedCellIds{}, stagedCellTopologyIds{}, sourceSeeds{};
    const int emitted = runOnSlab(
      estimator, roadKey(level), static_cast<double>(nLevelSeeds), [&](const int capacity) {
        ensureCapacity(out, capacity);
#ifdef GPUCA_DETERMINISTIC_MODE
        staged = allocTrackSeed.allocate(out.capacity);
        stagedCellIds = allocInt.allocate(out.capacity);
        stagedCellTopologyIds = allocInt.allocate(out.capacity);
        sourceSeeds = allocInt.allocate(out.capacity);
#else
        staged = out.seeds;
        stagedCellIds = out.cellIds;
        stagedCellTopologyIds = out.cellTopologyIds;
#endif
        using LevelSeed = std::remove_pointer_t<decltype(levelSeeds)>;
        const int neighbourGrid = gpu::gridBlocks(std::is_same_v<LevelSeed, CellSeed>
                                                    ? gpu::ResidentBlocks.processNeighboursCellSeed
                                                    : gpu::ResidentBlocks.processNeighboursTrackSeed);
        GPUChkErrS(cudaMemsetAsync(thrust::raw_pointer_cast(outputCounter), 0, sizeof(int), gpu::Stream::DefaultStream));

        constexpr uint64_t CandidateTag = qStr2Tag("ITS_PNCA");
        alloc->pushTagOnStack(CandidateTag);
        auto allocCandidate = gpu::TypedAllocator<gpu::NeighbourCandidate>(alloc);
        auto candidateCounter = allocInt.allocate(1);
        int* candidateCounterPtr = thrust::raw_pointer_cast(candidateCounter);

        gpu::NeighbourCandidate* candidatePtr = nullptr;
        int candidateCapacity = 0;
        const int nCandidates = runOnSlab(
          estimator, candidateKey(level), static_cast<double>(nLevelSeeds), [&](const int attemptCapacity) {
            if (attemptCapacity > candidateCapacity) {
              candidatePtr = thrust::raw_pointer_cast(allocCandidate.allocate(attemptCapacity));
              candidateCapacity = attemptCapacity;
            }
            GPUChkErrS(cudaMemsetAsync(candidateCounterPtr, 0, sizeof(int), gpu::Stream::DefaultStream));
            gpu::processNeighbourCandidatesKernel<NLayers, LevelSeed><<<neighbourGrid, gpu::GPUThreads>>>(
              topologyId, level, allCellSeeds, levelSeeds, levelCellIds, levelCellTopologyIds, nLevelSeeds,
              usedClusters, neighbours, neighboursDeviceLUTs,
              candidatePtr, candidateCounterPtr, attemptCapacity);
            int produced{0};
            GPUChkErrS(cudaMemcpyAsync(&produced, candidateCounterPtr, sizeof(int), cudaMemcpyDeviceToHost, gpu::Stream::DefaultStream));
            GPUChkErrS(cudaStreamSynchronize(gpu::Stream::DefaultStream));
            return produced;
          });

        if (nCandidates > 0) {
          gpu::sortNeighbourCandidates(topologyId, levelCellTopologyIds, allCellSeeds, neighbours, candidatePtr, nCandidates, alloc);
          gpu::fitNeighbourCandidatesKernel<NLayers, LevelSeed><<<neighbourGrid, gpu::GPUThreads>>>(
            topologyId, allCellSeeds, levelSeeds, levelCellTopologyIds,
            candidatePtr, candidateCounterPtr, candidateCapacity, neighbours,
            thrust::raw_pointer_cast(staged),
            thrust::raw_pointer_cast(stagedCellIds),
            thrust::raw_pointer_cast(stagedCellTopologyIds),
            thrust::raw_pointer_cast(sourceSeeds),
            thrust::raw_pointer_cast(outputCounter),
            out.capacity,
            foundTrackingFrameInfo, layerxX0, bz, maxChi2ClusterAttachment, propagator, matCorrType);
        }
        int wanted{0};
        GPUChkErrS(cudaMemcpyAsync(&wanted, thrust::raw_pointer_cast(outputCounter), sizeof(int), cudaMemcpyDeviceToHost, gpu::Stream::DefaultStream));
        GPUChkErrS(cudaStreamSynchronize(gpu::Stream::DefaultStream));
        alloc->popTagOffStack(CandidateTag);
        return wanted;
      },
      static_cast<size_t>(out.capacity));

    nWaveSeeds = emitted;
    filled = outIdx;
#ifdef GPUCA_DETERMINISTIC_MODE
    if (emitted > 0) {
      auto permutation = allocInt.allocate(emitted);
      thrust::sequence(nosync_policy, permutation, permutation + emitted);
      thrust::stable_sort_by_key(nosync_policy, sourceSeeds, sourceSeeds + emitted, permutation);
      thrust::gather(nosync_policy, permutation, permutation + emitted, staged, out.seeds);
      thrust::gather(nosync_policy, permutation, permutation + emitted, stagedCellIds, out.cellIds);
      thrust::gather(nosync_policy, permutation, permutation + emitted, stagedCellTopologyIds, out.cellTopologyIds);
      GPUChkErrS(cudaStreamSynchronize(gpu::Stream::DefaultStream));
    }
#endif
  };

  processLevel(currentCellSeeds, currentCellIds, currentCellTopologyIds, nCells[startCellTopologyId], startLevel, startCellTopologyId);

  int level = startLevel;
  while (level > 2 && nWaveSeeds > 0) {
    const Slab& in = slabs[filled];
    const int nLastSeeds = nWaveSeeds;
    --level;
    processLevel(thrust::raw_pointer_cast(in.seeds), thrust::raw_pointer_cast(in.cellIds), thrust::raw_pointer_cast(in.cellTopologyIds),
                 nLastSeeds, level, constants::UnusedIndex);
  }

  if (nWaveSeeds > 0) {
    Slab& spare = slabs[filled == 0 ? 1 : 0];
    ensureCapacity(spare, nWaveSeeds);
    const auto& last = slabs[filled];
    auto end = thrust::copy_if(nosync_policy, last.seeds, last.seeds + nWaveSeeds, spare.seeds, track::TrackSeedSelector<NLayers>{constants::MaxTrackSeedQ2Pt, maxChi2NDF, startLevel, maxHoles, minSeedingClusters, holeLayerMask, nonSeedingLayerMask});
    const int nSelected = static_cast<int>(end - spare.seeds);
    if (nSelected > 0 && seedsCursor + nSelected <= seedsCapacity) {
      GPUChkErrS(cudaMemcpyAsync(seedsDevice + seedsCursor, thrust::raw_pointer_cast(spare.seeds),
                                 nSelected * sizeof(TrackSeed<NLayers>), cudaMemcpyDeviceToDevice,
                                 gpu::Stream::DefaultStream));
      GPUChkErrS(cudaStreamSynchronize(gpu::Stream::DefaultStream));
    }
    seedsCursor += nSelected;
  }
  alloc->popTagOffStack(Tag);
}

template <int NLayers>
int TrackingKernels<NLayers>::computeTrackSeedHandler(TrackSeed<NLayers>* trackSeeds,
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
                                                      int* trackSeedIndices,
                                                      int* outputCounter,
                                                      const int trackCapacity,
                                                      TrackExtensionHypothesis<NLayers>* activeHypotheses,
                                                      TrackExtensionHypothesis<NLayers>* nextHypotheses,
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
                                                      const int maxHypotheses,
                                                      const bool extendTop,
                                                      const bool extendBot,
                                                      const float nSigmaCutPhi,
                                                      const float nSigmaCutZ,
                                                      const o2::base::Propagator* propagator,
                                                      const o2::base::PropagatorF::MatCorrType matCorrType,
                                                      o2::its::ExternalAllocator* alloc)
{
  GPUChkErrS(cudaMemsetAsync(outputCounter, 0, sizeof(int), gpu::Stream::DefaultStream));

  if (nSeeds > 1) { // Group the seeds by hit-layer pattern before fitting them
    constexpr uint64_t SortTag = qStr2Tag("ITS_TSSK");
    alloc->pushTagOnStack(SortTag);
    auto allocKey = gpu::TypedAllocator<unsigned int>(alloc);
    auto allocIndex = gpu::TypedAllocator<int>(alloc);
    auto allocSeed = gpu::TypedAllocator<TrackSeed<NLayers>>(alloc);
    auto keys = allocKey.allocate(nSeeds);
    auto order = allocIndex.allocate(nSeeds);
    auto sortedSeeds = allocSeed.allocate(nSeeds);
    auto sort_policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(gpu::Stream::DefaultStream);
    gpu::computeTrackSeedSortKeysKernel<NLayers><<<gpu::gridBlocks(gpu::ResidentBlocks.compileLookupTable), gpu::GPUThreads, 0, gpu::Stream::DefaultStream>>>(
      trackSeeds, nSeeds, thrust::raw_pointer_cast(keys));
    thrust::sequence(sort_policy, order, order + nSeeds);
    thrust::stable_sort_by_key(sort_policy, keys, keys + nSeeds, order);
    thrust::gather(sort_policy, order, order + nSeeds, thrust::device_ptr<TrackSeed<NLayers>>(trackSeeds), sortedSeeds);
    GPUChkErrS(cudaMemcpyAsync(trackSeeds, thrust::raw_pointer_cast(sortedSeeds), nSeeds * sizeof(TrackSeed<NLayers>), cudaMemcpyDeviceToDevice, gpu::Stream::DefaultStream));
    GPUChkErrS(cudaStreamSynchronize(gpu::Stream::DefaultStream));
    alloc->popTagOffStack(SortTag);
  }

  // track follower is compiled out of the kernel when no iteration asks for it
  const auto launchFit = [&](auto extendTracks) {
    gpu::fitTrackSeedsKernel<NLayers, decltype(extendTracks)::value><<<gpu::gridBlocks(decltype(extendTracks)::value ? gpu::ResidentBlocks.fitTrackSeedsExtended
                                                                                                                     : gpu::ResidentBlocks.fitTrackSeeds),
                                                                       gpu::GPUThreads>>>(trackSeeds,               // CellSeed*
                                                                                          foundTrackingFrameInfo,   // TrackingFrameInfo**
                                                                                          unsortedClusters,         // Cluster**
                                                                                          utils,                    // IndexTableUtils*
                                                                                          rofMask,                  // ROFMaskTable::View
                                                                                          rofOverlaps,              // ROFOverlapTable::View
                                                                                          clusters,                 // Cluster**
                                                                                          usedClusters,             // unsigned char**
                                                                                          clustersIndexTables,      // int**
                                                                                          ROFClusters,              // int**
                                                                                          tracks,                   // TrackITSExt*
                                                                                          trackSeedIndices,         // int*
                                                                                          outputCounter,            // int*
                                                                                          trackCapacity,            // const int
                                                                                          activeHypotheses,         // TrackExtensionHypothesis*
                                                                                          nextHypotheses,           // TrackExtensionHypothesis*
                                                                                          layerRadii,               // const float*
                                                                                          minPts,                   // const float*
                                                                                          layerxX0,                 // const float*
                                                                                          nSeeds,                   // const unsigned int
                                                                                          bz,                       // const float
                                                                                          maxChi2ClusterAttachment, // float
                                                                                          maxChi2NDF,               // float
                                                                                          reseedIfShorter,          // int
                                                                                          repeatRefitOut,           // bool
                                                                                          shiftRefToCluster,        // bool
                                                                                          nLayers,                  // int
                                                                                          phiBins,                  // int
                                                                                          maxHypotheses,            // int
                                                                                          extendTop,                // bool
                                                                                          extendBot,                // bool
                                                                                          nSigmaCutPhi,             // float
                                                                                          nSigmaCutZ,               // float
                                                                                          propagator,               // const o2::base::Propagator*
                                                                                          matCorrType);             // o2::base::PropagatorF::MatCorrType
  };
  if (extendTop || extendBot) {
    launchFit(std::true_type{});
  } else {
    launchFit(std::false_type{});
  }
  int emitted{0};
  GPUChkErrS(cudaMemcpyAsync(&emitted, outputCounter, sizeof(int), cudaMemcpyDeviceToHost, gpu::Stream::DefaultStream));
  GPUChkErrS(cudaStreamSynchronize(gpu::Stream::DefaultStream));
  if (emitted > trackCapacity) { // the slab was too small, the caller resizes and calls again
    return emitted;
  }
  constexpr uint64_t Tag = qStr2Tag("ITS_CTSH");
  alloc->pushTagOnStack(Tag);
  auto sync_policy = THRUST_NAMESPACE::par(gpu::TypedAllocator<char>(alloc));
  thrust::device_ptr<int> trackIndicesPtr(trackIndices);
  thrust::sequence(sync_policy, trackIndicesPtr, trackIndicesPtr + emitted);
  thrust::sort(sync_policy, trackIndicesPtr, trackIndicesPtr + emitted, gpu::compare_track_index_chi2{tracks, trackSeedIndices});

  if (emitted > 0) {
    auto allocTrack = gpu::TypedAllocator<o2::its::TrackITSExt>(alloc);
    auto sorted = allocTrack.allocate(emitted);
    thrust::device_ptr<o2::its::TrackITSExt> tracksPtr(tracks);
    thrust::gather(sync_policy, trackIndicesPtr, trackIndicesPtr + emitted, tracksPtr, sorted);
    GPUChkErrS(cudaMemcpyAsync(tracks, thrust::raw_pointer_cast(sorted),
                               emitted * sizeof(o2::its::TrackITSExt), cudaMemcpyDeviceToDevice,
                               gpu::Stream::DefaultStream));
    GPUChkErrS(cudaStreamSynchronize(gpu::Stream::DefaultStream));
  }
  alloc->popTagOffStack(Tag);
  return emitted;
}

/// One instantiation per detector layout emits every handler above.
template struct TrackingKernels<7>;
#ifdef ENABLE_UPGRADES
template struct TrackingKernels<11>;
template struct TrackingKernels<13>;
#endif

} // namespace o2::its
