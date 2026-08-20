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
#include <unistd.h>

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include "DataFormatsITS/TrackITS.h"
#include "ITStracking/Constants.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStrackingGPU/LaunchGeometry.h"
#include "ITStracking/MathUtils.h"
#include "ITStracking/ExternalAllocator.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/Cluster.h"
#include "ITStracking/Cell.h"
#include "ITStracking/TrackHelpers.h"
#include "ITStracking/TrackFollower.h"
#include "ITStrackingGPU/TrackingKernels.h"
#include "ITStrackingGPU/Utils.h"
#include "MathUtils/Utils.h"
#include "utils/strtag.h"

// O2 track model
#include "ReconstructionDataFormats/Track.h"
#include "DetectorsBase/Propagator.h"
using namespace o2::track;

namespace o2::its
{
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

template <int NLayers>
GPUg() void __launch_bounds__(GPUThreads, MinBlocks.computeLayerCellNeighbours) computeLayerCellNeighboursKernel(
  CellSeed** cellSeedArray,
  int** cellsLUTs,
  CellNeighbour* cellNeighbours,
  int* outputCounter,
  const int outputCapacity,
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

      float chi2 = currentCellSeed.getPredictedChi2Fast(nextCellSeed);
      if (chi2 > maxChi2ClusterAttachment) {
        continue;
      }

      const int currentCellLevel{currentCellSeed.getLevel()};
      const int outputIndex = atomicAdd(outputCounter, 1);
      if (outputIndex < outputCapacity) {
        cellNeighbours[outputIndex] = {sourceCellTopologyId, iCurrentCellIndex, targetCellTopologyId, iNextCell, currentCellLevel + 1};
      }
      if (currentCellLevel >= nextCellSeed.getLevel()) {
        atomicMax(cellSeedArray[targetCellTopologyId][iNextCell].getLevelPtr(), currentCellLevel + 1);
      }
    }
  }
}

/// A tracklet pair that passed the cheap cuts and is worth fitting.
struct CellCandidate {
  int firstTrackletIndex;
  int secondTrackletIndex;
};

template <int NLayers>
GPUg() void __launch_bounds__(GPUThreads, MinBlocks.computeLayerCells) computeLayerCellCandidatesKernel(
  Tracklet** tracklets,
  int** trackletsLUT,
  const int nTrackletsCurrent,
  const int cellTopologyId,
  const typename TrackingTopology<NLayers>::View topology,
  CellCandidate* candidates,
  int* outputCounter,
  const int outputCapacity,
  const float cellDeltaTanLambdaSigma,
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
      const float deltaTanLambda{o2::gpu::CAMath::Abs(currentTracklet.tanLambda - nextTracklet.tanLambda)};
      if (deltaTanLambda / cellDeltaTanLambdaSigma < nSigmaCut) {
        const int outputIndex = atomicAdd(outputCounter, 1);
        if (outputIndex < outputCapacity) {
          candidates[outputIndex] = CellCandidate{iCurrentTrackletIndex, iNextTrackletIndex};
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

    {
      const auto& currentCluster{clusters[fromLayer][currentSortedIndex]};
      if (usedClusters[fromLayer][currentCluster.clusterId]) {
        continue;
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

template <int NLayers, typename CurrentSeed>
GPUg() void __launch_bounds__(GPUThreads, (std::is_same_v<CurrentSeed, CellSeed> ? MinBlocks.processNeighboursCellSeed : MinBlocks.processNeighboursTrackSeed)) processNeighboursKernel(
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
  int* updatedSourceSeeds,
  int* outputCounter,
  const int outputCapacity,
  const unsigned char** usedClusters,
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
                                                            std::array<float, NLayers>& minRs,
                                                            std::array<float, NLayers>& maxRs,
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
    minRs[toLayer],
    maxRs[toLayer],
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
  gpu::computeLayerCellCandidatesKernel<NLayers><<<candidateBlocks, gpu::GPUThreads, 0, stream.get()>>>(
    tracklets, trackletsLUT, nTracklets, cellTopologyId, topology,
    nullptr, // counting pass: capacity 0, so nothing is written
    outputCounter, 0, cellDeltaTanLambdaSigma, nSigmaCut);
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
  GPUChkErrS(cudaMemsetAsync(outputCounter, 0, sizeof(int), stream.get()));
  gpu::computeLayerCellCandidatesKernel<NLayers><<<candidateBlocks, gpu::GPUThreads, 0, stream.get()>>>(
    tracklets, trackletsLUT, nTracklets, cellTopologyId, topology,
    thrust::raw_pointer_cast(candidates), outputCounter, nCandidates,
    cellDeltaTanLambdaSigma, nSigmaCut);

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
                                                            gpu::Stream& stream)
{
  gpu::computeLayerCellNeighboursKernel<NLayers><<<gpu::gridBlocks(gpu::ResidentBlocks.computeLayerCellNeighbours), gpu::GPUThreads, 0, stream.get()>>>(
    cellsLayersDevice,
    cellsLUTs,
    cellNeighbours,
    outputCounter,
    capacity,
    sourceCellTopologyId,
    targetCellTopologyId,
    maxChi2ClusterAttachment,
    bz,
    nCells);
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
        GPUChkErrS(cudaMemsetAsync(thrust::raw_pointer_cast(outputCounter), 0, sizeof(int), gpu::Stream::DefaultStream));
        gpu::processNeighboursKernel<NLayers, std::remove_pointer_t<decltype(levelSeeds)>><<<gpu::gridBlocks(std::is_same_v<std::remove_pointer_t<decltype(levelSeeds)>, CellSeed>
                                                                                                               ? gpu::ResidentBlocks.processNeighboursCellSeed
                                                                                                               : gpu::ResidentBlocks.processNeighboursTrackSeed),
                                                                                             gpu::GPUThreads>>>(topologyId,
                                                                                                                level,
                                                                                                                allCellSeeds,
                                                                                                                levelSeeds,
                                                                                                                levelCellIds,
                                                                                                                levelCellTopologyIds,
                                                                                                                nLevelSeeds,
                                                                                                                thrust::raw_pointer_cast(staged),
                                                                                                                thrust::raw_pointer_cast(stagedCellIds),
                                                                                                                thrust::raw_pointer_cast(stagedCellTopologyIds),
                                                                                                                thrust::raw_pointer_cast(sourceSeeds),
                                                                                                                thrust::raw_pointer_cast(outputCounter),
                                                                                                                out.capacity,
                                                                                                                usedClusters,
                                                                                                                neighbours,
                                                                                                                neighboursDeviceLUTs,
                                                                                                                foundTrackingFrameInfo,
                                                                                                                layerxX0,
                                                                                                                bz,
                                                                                                                maxChi2ClusterAttachment,
                                                                                                                propagator,
                                                                                                                matCorrType);
        int wanted{0};
        GPUChkErrS(cudaMemcpyAsync(&wanted, thrust::raw_pointer_cast(outputCounter), sizeof(int), cudaMemcpyDeviceToHost, gpu::Stream::DefaultStream));
        GPUChkErrS(cudaStreamSynchronize(gpu::Stream::DefaultStream));
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
