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
/// \file TrackerTraits.cxx
/// \brief
///

#include <algorithm>
#include <array>
#include <iterator>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <oneapi/tbb/blocked_range.h>

#include "CommonConstants/MathConstants.h"
#include "Framework/Logger.h"
#include "GPUCommonMath.h"
#include "ITSMFTTracking/BoundedAllocator.h"
#include "ITSMFTTracking/Cell.h"
#include "ITSMFTTracking/CapacityEstimator.h"
#include "ITSMFTTracking/SlabBumpAllocator.h"
#include "ITSMFTTracking/Constants.h"
#include "ITSMFTTracking/MathUtils.h"
#include "ITSMFTTracking/Configuration.h"
#include "ITSMFTTracking/IndexTableConfiguration.h"
#include "ITSMFTTracking/RefitDriver.h"
#include "ITSMFTTracking/Propagator.h"
#include "ITSMFTTracking/MaterialPhysics.h"
#include "ITSMFTTracking/detail/MFTFwdTrackHelpers.h"
#include "ITSMFTTracking/IndexTableUtils.h"
#include "ITSMFTTracking/LayerMask.h"
#include "ITSMFTTracking/TripletFitting.h"
#include "ITSMFTTracking/detail/TimeFrameScratch.h"
#include "ITSMFTTracking/TrackerTraits.h"
#include "ITSMFTTracking/detail/CandidateFinding.h"
#include "ReconstructionDataFormats/TrackParametrization.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2::itsmft::tracking
{

namespace math_utils = o2::its::math_utils;
using o2::its::TimeEstBC;

namespace
{
constexpr uint8_t kCompatibilityAbsCharge = 1;
const o2::track::PID kCompatibilityPID = o2::track::PID::Pion;

struct RoadSeedEmission {
  TrackSeed seed;
  int cellId{-1};
  int cellPathId{-1};
};

void reserveGenericTrackPublication(TimeFrame& frame, std::size_t candidateCount, std::size_t maxReferencesPerTrack)
{
  auto& tracks = frame.getGenericTracks();
  auto& references = frame.getTrackClusterIndices();
  if (candidateCount > tracks.max_size() - tracks.size() ||
      (maxReferencesPerTrack != 0 && candidateCount > (references.max_size() - references.size()) / maxReferencesPerTrack)) {
    throw std::length_error{"GenericTrack publication exceeds the output container capacity"};
  }
  tracks.reserve(tracks.size() + candidateCount);
  references.reserve(references.size() + candidateCount * maxReferencesPerTrack);
}

bool appendGenericTrack(TimeFrame& frame,
                        const TrackingCandidate& candidate,
                        gsl::span<const gsl::span<const GlobalMeasurement>> layerMeasurements)
{
  GenericTrack track = candidate.track;
  track.hitLayers = {};
  std::vector<TrackClusterReference> resolvedReferences;
  resolvedReferences.reserve(layerMeasurements.size());
  for (std::size_t position = 0; position < layerMeasurements.size(); ++position) {
    const int localIndex = candidate.getClusterIndex(static_cast<int>(position));
    if (localIndex == o2::its::constants::UnusedIndex) {
      continue;
    }
    if (localIndex < 0 || static_cast<std::size_t>(localIndex) >= layerMeasurements[position].size()) {
      return false;
    }
    const auto& measurement = layerMeasurements[position][localIndex];
    const TrackClusterReference reference{LayerId{static_cast<uint16_t>(position)}, 0, measurement.clusterId};
    if (!reference.isValid()) {
      return false;
    }
    resolvedReferences.push_back(reference);
    track.hitLayers.set(static_cast<int>(position));
  }
  if (!track.innerState.hasRecognizedKind() || !track.outerState.hasRecognizedKind() ||
      !track.timestamp.isValid() || resolvedReferences.empty()) {
    return false;
  }

  auto& tracks = frame.getGenericTracks();
  auto& references = frame.getTrackClusterIndices();
  const auto oldTrackSize = tracks.size();
  const auto oldReferenceSize = references.size();
  if (oldTrackSize > std::numeric_limits<uint32_t>::max() || oldReferenceSize > std::numeric_limits<uint32_t>::max() ||
      resolvedReferences.size() > std::numeric_limits<uint32_t>::max() - oldReferenceSize) {
    return false;
  }

  try {
    for (const auto& reference : resolvedReferences) {
      references.push_back(reference);
    }
    track.firstClusterRef = static_cast<uint32_t>(oldReferenceSize);
    track.clusterRefEnd = static_cast<uint32_t>(references.size());
    tracks.push_back(track);
  } catch (...) {
    references.resize(oldReferenceSize);
    tracks.resize(oldTrackSize);
    throw;
  }
  return true;
}

// A static diamond vertex represents all primary vertices and has no event
// timestamp. Derive its envelope from the tested ROF's configured bounds;
// TimeEstBC cannot represent a full TimeFrame. The resulting timestamp is
// compatible by construction with that ROF.
template <typename ROFOverlapView>
Vertex diamondVertexForROF(const Vertex& base, const ROFOverlapView& rofOverlapView, int layer, int rofId)
{
  Vertex v = base;
  v.setTimeStamp(rofOverlapView.getLayer(layer).getROFTimeBounds(rofId, true));
  return v;
}

// Convert ROOT-visible parameters to the device-portable record once per iteration.
} // namespace

void TrackerTraits::runTraversal(IterationContext& view)
{
  if (view.iteration < 0) {
    throw TraversalException{view.iteration, TraversalFailureReason::IterationOutOfRange};
  }
  int maxNvertices{-1};
  if (view.configuration.parameters.PerPrimaryVertexProcessing) {
    maxNvertices = view.frame.getMaxVerticesPerROF();
  }
  int iVertex = std::min(maxNvertices, 0);
  do {
    computeLayerTracklets(view, view.iteration, iVertex);
    computeLayerCells(view, view.iteration);
    findCellsNeighbours(view, view.iteration);
    findRoads(view, view.iteration);
  } while (++iVertex < maxNvertices);
}

void TrackerTraits::computeLayerTracklets(IterationContext& context, const int iteration, int iVertex)
{
  auto& scratch = context.scratch;
  const auto scratchEdgeCount = scratch.getTracklets().size();
  for (size_t edgeId = 0; edgeId < scratchEdgeCount; ++edgeId) {
    scratch.getTracklets()[edgeId].clear();
    scratch.getTrackletsLabel(edgeId).clear();
    std::fill(scratch.getTrackletsLookupTable()[edgeId].begin(), scratch.getTrackletsLookupTable()[edgeId].end(), 0);
  }

  const auto edgeIds = context.configuration.edgeIds();
  const auto& mMemoryPool = scratch.getMemoryPool();
  auto* mFrame = &context.frame;
  const auto& trkParam = context.configuration.parameters;
  const auto& mTraversalGraph = context.topology;
  const auto& mKernelParameters = context.configuration.kernelParameters;
  const auto& mLayerGlobalMeasurements = context.layerGlobalMeasurements;
  const auto& topology = mTraversalGraph;
  const Vertex diamondVert(trkParam.Diamond, trkParam.DiamondCov, 1, 1.f);

  mTaskArena->execute([&] {
    auto forTracklets = [&](int fromLayer, int toLayer, SurfaceKind kind,
                            const TrackletProjectionCache& edgeCache, int pivotROF, auto&& emit) {
      if (!mFrame->isROFEnabled(fromLayer, pivotROF)) {
        return;
      }
      // Derive a diamond vertex for this pivot ROF; each invocation owns its
      // stack frame, so this is safe inside the parallel dispatch.
      Vertex diamondForROF{};
      gsl::span<const Vertex> primaryVertices;
      if (trkParam.UseDiamond) {
        diamondForROF = diamondVertexForROF(diamondVert, mFrame->getROFViews(fromLayer).overlap,
                                            mFrame->getROFLocalLayer(fromLayer), pivotROF);
        primaryVertices = gsl::span<const Vertex>(&diamondForROF, 1);
      } else {
        primaryVertices = mFrame->getPrimaryVertices(fromLayer, pivotROF);
      }
      if (primaryVertices.empty()) {
        return;
      }
      const int startVtx = iVertex >= 0 ? iVertex : 0;
      const int endVtx = iVertex >= 0 ? o2::gpu::CAMath::Min(iVertex + 1, int(primaryVertices.size())) : int(primaryVertices.size());
      if (endVtx <= startVtx || (iVertex + 1) > primaryVertices.size()) {
        return;
      }

      const auto& rofOverlap = mFrame->getROFOverlap(fromLayer, toLayer, pivotROF);
      if (!rofOverlap.getEntries()) {
        return;
      }

      auto layer0 = mFrame->getClustersOnLayer(pivotROF, fromLayer);
      if (layer0.empty()) {
        return;
      }

      for (int iCluster = 0; iCluster < int(layer0.size()); ++iCluster) {
        const GlobalMeasurement& sourceMeasurement = layer0[iCluster];
        const int currentSortedIndex = mFrame->getSortedIndex(pivotROF, fromLayer, iCluster);
        if (mFrame->isClusterUsed(fromLayer, sourceMeasurement.clusterId)) {
          continue;
        }

        for (int iV = startVtx; iV < endVtx; ++iV) {
          const auto& pv = primaryVertices[iV];
          if (!mFrame->isVertexCompatible(fromLayer, pivotROF, pv)) {
            continue;
          }
          if (pv.isFlagSet(Vertex::Flags::UPCMode) != trkParam.PassFlags[IterationStep::SelectUPCVertices]) {
            continue;
          }
          const auto& indexTableUtils = mFrame->getIndexTableUtils(toLayer);
          TrackletSearchWindow window{};
          if (!projectTrackletSearchWindow(sourceMeasurement, pv, mFrame->getBeamPositionVariance(),
                                           kind, edgeCache, indexTableUtils,
                                           mKernelParameters.nSigmaCut, window)) {
            continue;
          }
          const auto bins = window.bins;
          int rowBinsNum = bins.w - bins.y + 1;
          if (rowBinsNum < 0) {
            rowBinsNum += indexTableUtils.getNrowBins();
          }
          rowBinsNum = std::max(0, rowBinsNum);

          for (int targetROF = rofOverlap.getFirstEntry(); targetROF < rofOverlap.getEntriesBound(); ++targetROF) {
            if (!mFrame->isROFEnabled(toLayer, targetROF)) {
              continue;
            }
            auto layer1 = mFrame->getClustersOnLayer(targetROF, toLayer);
            if (layer1.empty()) {
              continue;
            }
            const auto ts = mFrame->getROFTimeStamp(fromLayer, pivotROF, toLayer, targetROF);
            if (!ts.isCompatible(pv.getTimeStamp())) {
              continue;
            }
            const auto& targetIndexTable = mFrame->getIndexTable(targetROF, toLayer);
            const int colBinRange = (bins.z - bins.x) + 1;
            for (int iRow = 0; iRow < rowBinsNum; ++iRow) {
              int iRowBin = bins.y + iRow;
              iRowBin %= indexTableUtils.getNrowBins();
              if (iRowBin < 0 || iRowBin >= indexTableUtils.getNrowBins()) {
                break;
              }
              const int firstBinIdx = indexTableUtils.getBinIndex(bins.x, iRowBin);
              const int maxBinIdx = firstBinIdx + colBinRange;
              const int firstRow = targetIndexTable[firstBinIdx];
              const int lastRow = targetIndexTable[maxBinIdx];
              for (int iNext = firstRow; iNext < lastRow; ++iNext) {
                if (iNext >= int(layer1.size())) {
                  break;
                }
                const GlobalMeasurement& targetMeasurement = layer1[iNext];
                if (mFrame->isClusterUsed(toLayer, targetMeasurement.clusterId)) {
                  continue;
                }

                const float targetReferenceCoordinate = kind == SurfaceKind::Cylinder ? targetMeasurement.radius : targetMeasurement.z;
                const float targetProjectedCoordinate = kind == SurfaceKind::Cylinder ? targetMeasurement.z : targetMeasurement.radius;
                const float referenceDelta = targetReferenceCoordinate - window.sourceReferenceCoordinate;
                const float candidatePrediction = window.sourceProjectedCoordinate + window.slope * referenceDelta;
                const float candidateVariance = window.varianceConstant +
                                                referenceDelta * (window.varianceLinear + referenceDelta * window.varianceQuadratic);
                const float projectedResidual = candidatePrediction - targetProjectedCoordinate;
                const float phiResidual = std::remainder(window.phiPrediction - targetMeasurement.phi, o2::constants::math::TwoPI);

                if (!(candidateVariance > 0.f && window.phiVariance > 0.f)) {
                  continue;
                }
                const float chi2 = o2::its::math_utils::Sq(projectedResidual) / candidateVariance +
                                   o2::its::math_utils::Sq(phiResidual) / window.phiVariance;
                if (chi2 >= o2::its::math_utils::Sq(mKernelParameters.nSigmaCut)) {
                  continue;
                }
                const float deltaR = sourceMeasurement.radius - targetMeasurement.radius;
                const float deltaZ = sourceMeasurement.z - targetMeasurement.z;
                const float tanL = o2::its::math_utils::Sq(deltaR) > o2::constants::math::Almost0 ? deltaZ / deltaR : std::copysign(o2::constants::math::VeryBig, deltaZ);
                const float phi{o2::gpu::GPUCommonMath::ATan2(sourceMeasurement.y - targetMeasurement.y,
                                                              sourceMeasurement.x - targetMeasurement.x)};
                emit(currentSortedIndex, mFrame->getSortedIndex(targetROF, toLayer, iNext), tanL, phi, ts);
              }
            }
          }
        }
      }
    };

    const int maxConcurrency = std::max(1, mTaskArena->max_concurrency());
    const int nConcurrentSinks = std::min(static_cast<int>(edgeIds.size()), maxConcurrency);
    tbb::parallel_for(0, static_cast<int>(edgeIds.size()), [&](const int edgeIndex) {
      const auto edgeId = edgeIds[edgeIndex];
      const auto& edge = topology.getEdge(edgeId);
      const int fromLayer = edge.from.value();
      const int toLayer = edge.to.value();
      const auto kind = topology.getSurface(edge.from).kind;
      const auto& layerRadii = context.detectorConfiguration.layerRadii;
      const TrackletProjectionCache edgeCache{
        fromLayer, toLayer, layerRadii[fromLayer], layerRadii[toLayer],
        mFrame->getMinR(toLayer), mFrame->getMaxR(toLayer),
        mFrame->getMinZ(toLayer), mFrame->getMaxZ(toLayer),
        context.detectorConfiguration.positionResolutions[fromLayer],
        scratch.getEdgeMSAngle(edgeId.value()), scratch.getEdgePhiCut(edgeId.value())};
      const int endROF = mFrame->getROFTiming(fromLayer).mNROFsTF;
      const auto key = CapacityEstimator::makeKey(SlabSite::Tracklets, iteration, iVertex + 1, edgeId);
      const auto scale = static_cast<double>(mFrame->getClusters()[fromLayer].size());
      const auto capacity = mFrame->getCapacityEstimator().capacity(key, scale);
      UnorderedSlabSink<Tracklet> sink{{.capacity = capacity, .nThreads = maxConcurrency, .nConcurrentSinks = nConcurrentSinks}, mMemoryPool.get()};
      tbb::parallel_for(0, endROF, [&](const int pivotROF) {
        auto& handle = sink.local();
        forTracklets(fromLayer, toLayer, kind, edgeCache, pivotROF,
                     [&handle](auto&&... args) { handle.emplace(std::forward<decltype(args)>(args)...); });
      });
      const auto stats = sink.stats();
      sink.finalizeUnordered(scratch.getTracklets()[edgeId.value()]);
      mFrame->getCapacityEstimator().update(key, scale, stats.requested, stats.capacity, stats.emitted,
                                            stats.spilled, stats.overflowed, stats.memoryLimited);
    });

    tbb::parallel_for(0, static_cast<int>(edgeIds.size()), [&](const int edgeIndex) {
      const auto edgeId = edgeIds[edgeIndex];
      /// Sort tracklets & remove duplicates
      // duplicates can exist simply since we evaluate per vertex
      auto& trkl{scratch.getTracklets()[edgeId.value()]};
      std::sort(trkl.begin(), trkl.end());
      trkl.erase(std::unique(trkl.begin(), trkl.end()), trkl.end());
      trkl.shrink_to_fit();
      auto& lut{scratch.getTrackletsLookupTable()[edgeId.value()]};
      if (!trkl.empty()) {
        for (const auto& tkl : trkl) {
          lut[tkl.firstClusterIndex + 1]++;
        }
        std::inclusive_scan(lut.begin(), lut.end(), lut.begin());
      }
    });

    /// Create tracklets labels
    if (mFrame->hasMCinformation() && trkParam.CreateArtefactLabels) {
      tbb::parallel_for(0, static_cast<int>(edgeIds.size()), [&](const int edgeIndex) {
        const auto edgeId = edgeIds[edgeIndex];
        const auto& edge = topology.getEdge(edgeId);
        const int fromLayer = edge.from.value();
        const int toLayer = edge.to.value();
        for (auto& trk : scratch.getTracklets()[edgeId.value()]) {
          MCCompLabel label;
          const auto currentId = mFrame->getClusters()[fromLayer][trk.firstClusterIndex].clusterId;
          const auto nextId = mFrame->getClusters()[toLayer][trk.secondClusterIndex].clusterId;
          for (const auto& lab1 : mFrame->getLabels(LayerId{static_cast<uint16_t>(fromLayer)}, currentId)) {
            for (const auto& lab2 : mFrame->getLabels(LayerId{static_cast<uint16_t>(toLayer)}, nextId)) {
              if (lab1 == lab2 && lab1.isValid()) {
                label = lab1;
                break;
              }
            }
            if (label.isValid()) {
              break;
            }
          }
          scratch.getTrackletsLabel(edgeId.value()).emplace_back(label);
        }
      });
    }
  });
}

void TrackerTraits::computeLayerCells(IterationContext& context, const int iteration)
{
  auto& scratch = context.scratch;
  const auto scratchCellCount = scratch.getCells().size();
  for (size_t cellPathId = 0; cellPathId < scratchCellCount; ++cellPathId) {
    deepVectorClear(scratch.getCells()[cellPathId]);
    deepVectorClear(scratch.getCellsLookupTable()[cellPathId]);
    if (context.frame.hasMCinformation() && context.configuration.parameters.CreateArtefactLabels) {
      deepVectorClear(scratch.getCellsLabel(cellPathId));
    }
  }

  const auto cellIds = context.configuration.cellIds();
  const auto& mMemoryPool = scratch.getMemoryPool();
  const auto& trkParam = context.configuration.parameters;
  const auto mBz = context.bz;
  const auto& mTraversalGraph = context.topology;
  const auto& mKernelParameters = context.configuration.kernelParameters;
  const auto& mLayerGlobalMeasurements = context.layerGlobalMeasurements;
  const auto& topology = mTraversalGraph;

  mTaskArena->execute([&] {
    auto forTrackletCells = [&](int firstEdgeId, int secondEdgeId, const std::array<int, 3>& hitLayers, int iTracklet, auto&& emit) {
      const Tracklet& currentTracklet{scratch.getTracklets()[firstEdgeId][iTracklet]};
      const int nextLayerClusterIndex{currentTracklet.secondClusterIndex};
      const int nextLayerFirstTrackletIndex{scratch.getTrackletsLookupTable()[secondEdgeId][nextLayerClusterIndex]};
      const int nextLayerLastTrackletIndex{scratch.getTrackletsLookupTable()[secondEdgeId][nextLayerClusterIndex + 1]};
      for (int iNextTracklet{nextLayerFirstTrackletIndex}; iNextTracklet < nextLayerLastTrackletIndex; ++iNextTracklet) {
        const Tracklet& nextTracklet{scratch.getTracklets()[secondEdgeId][iNextTracklet]};
        if (nextTracklet.firstClusterIndex != nextLayerClusterIndex) {
          break;
        }
        if (!currentTracklet.getTimeStamp().isCompatible(nextTracklet.getTimeStamp())) {
          continue;
        }

        /// Prepare the track seed; clusters are numbered from inner to outer.
        const int sortedId[3]{currentTracklet.firstClusterIndex, nextTracklet.firstClusterIndex, nextTracklet.secondClusterIndex};

        const float edgeMSAngle = scratch.getEdgeMSAngle(secondEdgeId);
        const float angularTolerance = mKernelParameters.nSigmaCut * edgeMSAngle;
        const float lambda01 = std::atan(currentTracklet.tanLambda);
        const float lambda12 = std::atan(nextTracklet.tanLambda);
        const float deltaLambda = std::abs(lambda01 - lambda12);
        if (deltaLambda > angularTolerance) {
          continue;
        }

        const auto& inner = mLayerGlobalMeasurements[hitLayers[0]][sortedId[0]];
        const auto& middle = mLayerGlobalMeasurements[hitLayers[1]][sortedId[1]];
        const auto& outer = mLayerGlobalMeasurements[hitLayers[2]][sortedId[2]];
        const float length01 = std::hypot(inner.x - middle.x, inner.y - middle.y);
        const float length12 = std::hypot(middle.x - outer.x, middle.y - outer.y);
        const float maximumCurvature = std::min({std::abs(o2::constants::math::B2C * mBz) /
                                                   mKernelParameters.trackletMinPt,
                                                 2.f / length01,
                                                 2.f / length12});
        const float maximumBending =
          std::asin(std::clamp(0.5f * maximumCurvature * length01, 0.f, 1.f)) +
          std::asin(std::clamp(0.5f * maximumCurvature * length12, 0.f, 1.f));
        const float deltaPhi = std::abs(std::remainder(currentTracklet.phi - nextTracklet.phi,
                                                       o2::constants::math::TwoPI));
        const float sinTheta = std::max(std::abs(std::cos(0.5f * (lambda01 + lambda12))),
                                        o2::constants::math::Almost0);
        const float azimuthalTolerance = angularTolerance / sinTheta;
        if (deltaPhi > maximumBending + azimuthalTolerance) {
          continue;
        }

        const std::array<GlobalMeasurement, 3> measurements{inner, middle, outer};
        TripletFitFactor tripletFactor{};
        if (makeTripletFitFactor(measurements, tripletFactor)) {
          TimeEstBC ts = currentTracklet.getTimeStamp();
          ts += nextTracklet.getTimeStamp();
          // Build directly from the resolved plan positions; plan validation
          // already checked them against the cell's hit-surface mask.
          const LayerMask hitLayerMask{hitLayers[0], hitLayers[1], hitLayers[2]};
          CellSeed seed{hitLayerMask, sortedId[0], sortedId[1], sortedId[2], iTracklet, iNextTracklet, ts};
          seed.tripletFactor() = tripletFactor;
          emit(std::move(seed));
        }
      }
    };

    const int maxConcurrency = std::max(1, mTaskArena->max_concurrency());
    for (const auto cellId : cellIds) {
      const auto& cellTopology = topology.getPath(cellId);
      const auto firstEdgeId = cellTopology.first;
      const auto secondEdgeId = cellTopology.second;
      if (scratch.getTracklets()[firstEdgeId.value()].empty() ||
          scratch.getTracklets()[secondEdgeId.value()].empty()) {
        continue;
      }

      const auto& firstEdge = topology.getEdge(cellTopology.first);
      const auto& secondEdge = topology.getEdge(cellTopology.second);
      const std::array<int, 3> layers{firstEdge.from.value(), firstEdge.to.value(), secondEdge.to.value()};

      auto& layerCells = scratch.getCells()[cellId.value()];
      auto& lut = scratch.getCellsLookupTable()[cellId.value()];
      const int currentLayerTrackletsNum{static_cast<int>(scratch.getTracklets()[firstEdgeId.value()].size())};
      const auto key = CapacityEstimator::makeKey(SlabSite::Cells, iteration, 0, cellId);
      const auto scale = static_cast<double>(currentLayerTrackletsNum);
      const auto capacity = context.frame.getCapacityEstimator().capacity(key, scale);
      GroupedSlabSink<CellSeed> sink{{.capacity = capacity, .nThreads = maxConcurrency}, mMemoryPool.get()};
      tbb::parallel_for(0, currentLayerTrackletsNum, [&](const int iTracklet) {
        auto& handle = sink.local();
        handle.beginProducer(iTracklet);
        forTrackletCells(firstEdgeId.value(), secondEdgeId.value(), layers, iTracklet,
                         [&handle](CellSeed seed) { handle.emplace(std::move(seed)); });
      });
      const auto stats = sink.stats();
      sink.finalizeGrouped(static_cast<size_t>(currentLayerTrackletsNum), lut, layerCells);
      context.frame.getCapacityEstimator().update(key, scale, stats.requested, stats.capacity, stats.emitted,
                                                  stats.spilled, stats.overflowed, stats.memoryLimited);

      if (context.frame.hasMCinformation() && trkParam.CreateArtefactLabels) {
        auto& labels = scratch.getCellsLabel(cellId.value());
        labels.reserve(layerCells.size());
        for (const auto& cell : layerCells) {
          MCCompLabel currentLab{scratch.getTrackletsLabel(firstEdgeId.value())[cell.getFirstTrackletIndex()]};
          MCCompLabel nextLab{scratch.getTrackletsLabel(secondEdgeId.value())[cell.getSecondTrackletIndex()]};
          labels.emplace_back(currentLab == nextLab ? currentLab : MCCompLabel());
        }
      }
    }
  });

  const auto scratchEdgeCount = scratch.getTracklets().size();
  for (size_t edgeId = 0; edgeId < scratchEdgeCount; ++edgeId) {
    deepVectorClear(scratch.getTracklets()[edgeId]);
    deepVectorClear(scratch.getTrackletsLabel(edgeId));
  }
}

void TrackerTraits::findCellsNeighbours(IterationContext& context, const int iteration)
{
  auto& scratch = context.scratch;
  const auto& memoryPool = scratch.getMemoryPool();
  const auto& topology = context.topology;
  const auto& globalMeasurements = context.layerGlobalMeasurements;
  const auto& params = context.configuration.kernelParameters;
  for (std::size_t slot = 0; slot < scratch.getCellsNeighbours().size(); ++slot) {
    deepVectorClear(scratch.getCellsNeighbours()[slot]);
    deepVectorClear(scratch.getCellsNeighboursTopology()[slot]);
    deepVectorClear(scratch.getCellsNeighboursLUT()[slot]);
  }
  const auto& scheduledCells = context.configuration.topology.scheduledPaths;
  const auto scratchCellCount = scratch.getCells().size();
  if (scratch.getCellsLookupTable().size() != scratchCellCount ||
      scratch.getCellsNeighbours().size() != scratchCellCount ||
      scratch.getCellsNeighboursTopology().size() != scratchCellCount ||
      scratch.getCellsNeighboursLUT().size() != scratchCellCount) {
    throw TraversalException{iteration, TraversalFailureReason::SparseTopologyMismatch};
  }
  mTaskArena->execute([&] {
    std::vector<bounded_vector<CellNeighbour>> cellsNeighboursByTarget;
    cellsNeighboursByTarget.reserve(scratchCellCount);
    for (size_t cellPathId = 0; cellPathId < scratchCellCount; ++cellPathId) {
      cellsNeighboursByTarget.emplace_back(memoryPool.get());
    }

    for (const auto cellId : scheduledCells) {
      if (static_cast<size_t>(cellId.value()) >= scratchCellCount ||
          static_cast<size_t>(cellId.value()) >= scratch.getCellsLookupTable().size()) {
        throw TraversalException{iteration, TraversalFailureReason::SparseTopologyMismatch};
      }
      const auto& cellTopology = topology.getPath(cellId);
      const float currentMSAngle = scratch.getEdgeMSAngle(cellTopology.second.value());
      const float currentAngularVariance = currentMSAngle * currentMSAngle;
      if (scratch.getCells()[cellId.value()].empty()) {
        continue;
      }
      const auto successors = topology.getPathsStartingWithEdge(cellTopology.second);
      if (!successors.getEntries()) {
        continue;
      }

      struct SuccessorBinding {
        CellPathId cellId;
        float angularVariance;
      };
      std::array<SuccessorBinding, MaxLayoutSurfaces> successorBindings{};
      size_t successorBindingCount = 0;
      if (successors.getEntries() > successorBindings.size()) {
        throw TraversalException{iteration, TraversalFailureReason::SparseTopologyMismatch};
      }
      for (uint32_t iSuccessor = 0; iSuccessor < successors.getEntries(); ++iSuccessor) {
        const auto nextCellId = topology.pathsByFirstEdge[successors.getFirstEntry() + iSuccessor];
        if (static_cast<size_t>(nextCellId.value()) >= scratch.getCells().size() ||
            static_cast<size_t>(nextCellId.value()) >= scratch.getCellsLookupTable().size()) {
          throw TraversalException{iteration, TraversalFailureReason::SparseTopologyMismatch};
        }
        if (scratch.getCells()[nextCellId.value()].empty() ||
            scratch.getCellsLookupTable()[nextCellId.value()].empty()) {
          continue;
        }
        const auto& nextCellTopology = topology.getPath(nextCellId);
        const float nextMSAngle = scratch.getEdgeMSAngle(nextCellTopology.second.value());
        successorBindings[successorBindingCount++] = {nextCellId, nextMSAngle * nextMSAngle};
      }

      const int maxConcurrency = std::max(1, mTaskArena->max_concurrency());
      const auto key = CapacityEstimator::makeKey(SlabSite::Neighbours, iteration, 0, cellId);
      const auto scale = static_cast<double>(scratch.getCells()[cellId.value()].size());
      const auto capacity = context.frame.getCapacityEstimator().capacity(key, scale);
      UnorderedSlabSink<CellNeighbour> sink{{.capacity = capacity, .nThreads = maxConcurrency}, memoryPool.get()};
      tbb::parallel_for(0, static_cast<int>(scratch.getCells()[cellId.value()].size()), [&](const int iCell) {
        auto& handle = sink.local();
        const auto& currentCellSeed{scratch.getCells()[cellId.value()][iCell]};
        const int nextLayerTrackletIndex{currentCellSeed.getSecondTrackletIndex()};
        for (size_t iSuccessor = 0; iSuccessor < successorBindingCount; ++iSuccessor) {
          const auto& successor = successorBindings[iSuccessor];
          const auto& nextCellLUT = scratch.getCellsLookupTable()[successor.cellId.value()];
          if (nextLayerTrackletIndex < 0 || nextLayerTrackletIndex + 1 >= static_cast<int>(nextCellLUT.size())) {
            continue;
          }
          const int nextLayerFirstCellIndex{nextCellLUT[nextLayerTrackletIndex]};
          const int nextLayerLastCellIndex{nextCellLUT[nextLayerTrackletIndex + 1]};
          if (nextLayerFirstCellIndex < 0 || nextLayerLastCellIndex < nextLayerFirstCellIndex ||
              nextLayerLastCellIndex > static_cast<int>(scratch.getCells()[successor.cellId.value()].size())) {
            throw TraversalException{iteration, TraversalFailureReason::SparseTopologyMismatch};
          }
          for (int iNextCell{nextLayerFirstCellIndex}; iNextCell < nextLayerLastCellIndex; ++iNextCell) {
            const auto& nextCellSeedRef{scratch.getCells()[successor.cellId.value()][iNextCell]};
            if (nextCellSeedRef.getFirstTrackletIndex() != nextLayerTrackletIndex || !currentCellSeed.getTimeStamp().isCompatible(nextCellSeedRef.getTimeStamp())) {
              break;
            }

            const auto currentMiddle = currentCellSeed.getClusterReference(1);
            const auto currentOuter = currentCellSeed.getClusterReference(2);
            const auto nextInner = nextCellSeedRef.getClusterReference(0);
            const auto nextMiddle = nextCellSeedRef.getClusterReference(1);
            if (currentMiddle.surfacePosition != nextInner.surfacePosition ||
                currentMiddle.clusterIndex != nextInner.clusterIndex ||
                currentOuter.surfacePosition != nextMiddle.surfacePosition ||
                currentOuter.clusterIndex != nextMiddle.clusterIndex) {
              continue;
            }

            const std::array<CellClusterReference, 4> references{
              currentCellSeed.getClusterReference(0), currentMiddle,
              currentOuter, nextCellSeedRef.getClusterReference(2)};
            std::array<GlobalMeasurement, 4> measurements{};
            bool measurementsValid = true;
            for (std::size_t hit = 0; hit < references.size(); ++hit) {
              const auto reference = references[hit];
              if (reference.surfacePosition < 0 ||
                  static_cast<std::size_t>(reference.surfacePosition) >= globalMeasurements.size() ||
                  reference.clusterIndex < 0 ||
                  static_cast<std::size_t>(reference.clusterIndex) >= globalMeasurements[reference.surfacePosition].size()) {
                measurementsValid = false;
                break;
              }
              measurements[hit] = globalMeasurements[reference.surfacePosition][reference.clusterIndex];
            }
            AdjacentTripletFitResult adjacentFit{};
            const bool fitValid = measurementsValid &&
                                  fitAdjacentTripletFactors(
                                    currentCellSeed.tripletFactor(), nextCellSeedRef.tripletFactor(), measurements,
                                    {currentAngularVariance, successor.angularVariance}, adjacentFit);
            if (!fitValid || adjacentFit.chi2 > params.maxChi2ClusterAttachment) {
              continue;
            }

            const int nextLevel = currentCellSeed.getLevel() + 1;
            handle.emplace(cellId.value(), iCell, successor.cellId.value(), iNextCell, nextLevel);
          }
        }
      });

      const auto stats = sink.stats();
      bounded_vector<CellNeighbour> sourceNeighbours{memoryPool.get()};
      sink.finalizeUnordered(sourceNeighbours);
      context.frame.getCapacityEstimator().update(key, scale, stats.requested, stats.capacity, stats.emitted,
                                                  stats.spilled, stats.overflowed, stats.memoryLimited);
      std::sort(sourceNeighbours.begin(), sourceNeighbours.end(), [](const auto& a, const auto& b) {
        return std::tie(a.nextCellTopology, a.nextCell, a.cellTopology, a.cell) <
               std::tie(b.nextCellTopology, b.nextCell, b.cellTopology, b.cell);
      });
      for (const auto& neighbour : sourceNeighbours) {
        cellsNeighboursByTarget[neighbour.nextCellTopology].push_back(neighbour);
        if (neighbour.level > scratch.getCells()[neighbour.nextCellTopology][neighbour.nextCell].getLevel()) {
          scratch.getCells()[neighbour.nextCellTopology][neighbour.nextCell].setLevel(neighbour.level);
        }
      }
    }

    for (size_t cellPathId = 0; cellPathId < scratchCellCount; ++cellPathId) {
      auto& cellsNeighbours = cellsNeighboursByTarget[cellPathId];
      if (cellsNeighbours.empty()) {
        continue;
      }

      std::sort(cellsNeighbours.begin(), cellsNeighbours.end(), [](const auto& a, const auto& b) {
        return std::tie(a.nextCell, a.cellTopology, a.cell) < std::tie(b.nextCell, b.cellTopology, b.cell);
      });

      auto& cellsNeighbourLUT = scratch.getCellsNeighboursLUT()[cellPathId];
      cellsNeighbourLUT.assign(scratch.getCells()[cellPathId].size(), 0);
      for (const auto& neigh : cellsNeighbours) {
        ++cellsNeighbourLUT[neigh.nextCell];
      }
      std::inclusive_scan(cellsNeighbourLUT.begin(), cellsNeighbourLUT.end(), cellsNeighbourLUT.begin());

      scratch.getCellsNeighbours()[cellPathId].reserve(cellsNeighbours.size());
      scratch.getCellsNeighboursTopology()[cellPathId].reserve(cellsNeighbours.size());
      std::ranges::transform(cellsNeighbours, std::back_inserter(scratch.getCellsNeighbours()[cellPathId]), [](const auto& neigh) { return neigh.cell; });
      std::ranges::transform(cellsNeighbours, std::back_inserter(scratch.getCellsNeighboursTopology()[cellPathId]), [](const auto& neigh) { return neigh.cellTopology; });
    }
  });
  for (auto& cellLUT : scratch.getCellsLookupTable()) {
    deepVectorClear(cellLUT);
  }
}

bool TrackerTraits::buildTrackSeed(IterationContext& context, int,
                                   const CellSeed& cell, TrackSeed& output,
                                   OperationFailureReason& reason) const
{
  std::array<const GlobalMeasurement*, 3> globals{};
  std::array<const SurfaceMeasurement*, 3> measurements{};
  std::array<const SurfaceDescriptor*, 3> surfaces{};
  for (int hit = 0; hit < 3; ++hit) {
    const auto reference = cell.getClusterReference(hit);
    const auto surface = LayerId{static_cast<uint16_t>(reference.surfacePosition)};
    globals[hit] = &context.layerGlobalMeasurements[reference.surfacePosition][reference.clusterIndex];
    measurements[hit] = context.frame.getSurfaceMeasurement(surface, globals[hit]->clusterId);
    surfaces[hit] = &context.topology.getSurface(surface);
  }

  SurfaceTrackState state{};
  float chi2{0.f};
  const auto& outer = *measurements[2];
  const auto kind = surfaces[2]->kind;

  float sinPhi = 0.f, cosPhi = 0.f, tanLambda = 0.f, qOverPt = 1.f / o2::track::kMostProbablePt;
  float curvatureSquared = 1.f;

  state.referenceCoordinate = outer.frame.q;
  state.alpha = (kind == SurfaceKind::Cylinder) ? outer.frame.frameAngle : 0.f;
  state.parameters[0] = outer.frame.u;
  state.parameters[1] = outer.frame.v;

  float cosAlpha, sinAlpha, x[3], y[3];
  o2::math_utils::detail::sincos(state.alpha, sinAlpha, cosAlpha);
  for (int i{0}; i < 3; ++i) {
    const auto& pos = globals[i]->position;
    x[i] = pos.x * cosAlpha + pos.y * sinAlpha;
    y[i] = -pos.x * sinAlpha + pos.y * cosAlpha;
  }
  const float dx = x[2] - x[1];
  const float dy = y[2] - y[1];
  const float chordLength = std::hypot(dx, dy);
  const float inverseLength = 1.f / chordLength;

  const float chordCos = dx * inverseLength;
  const float chordSin = dy * inverseLength;
  tanLambda = -0.5f *
              (math_utils::computeTanDipAngle(x[0], y[0], x[1], y[1], globals[0]->position.z, globals[1]->position.z) +
               math_utils::computeTanDipAngle(x[1], y[1], x[2], y[2], globals[1]->position.z, globals[2]->position.z));

  if (std::abs(context.bz) < 0.01f) {
    cosPhi = chordCos;
    sinPhi = chordSin;
  } else {
    const float curvature =
      math_utils::computeCurvature(
        x[2], y[2], x[1], y[1], x[0], y[0]);

    const float halfSin = 0.5f * curvature * chordLength;
    const float halfCos =
      std::sqrt((1.f - halfSin) * (1.f + halfSin));

    cosPhi = chordCos * halfCos - chordSin * halfSin;
    sinPhi = chordSin * halfCos + chordCos * halfSin;
    qOverPt = curvature /
              (context.bz * o2::constants::math::B2C);
    curvatureSquared = curvature * curvature;
  }

  float phi = o2::gpu::GPUCommonMath::ASin(sinPhi);
  if (cosPhi < 0.f) {
    phi = o2::constants::math::PI - phi;
  } else if (phi < 0.f) {
    phi += o2::constants::math::TwoPI;
  }

  state.parameters[2] = (kind == SurfaceKind::Cylinder) ? sinPhi : phi;
  state.parameters[3] = tanLambda;
  state.parameters[4] = qOverPt;
  state.covariance[packedCovarianceIndex(0, 0)] = outer.covariance.uu;
  state.covariance[packedCovarianceIndex(1, 0)] = outer.covariance.uv;
  state.covariance[packedCovarianceIndex(1, 1)] = outer.covariance.vv;
  state.covariance[packedCovarianceIndex(2, 2)] = (kind == SurfaceKind::Cylinder) ? o2::track::kCSnp2max : o2::track::kCSnp2max / (cosPhi * cosPhi);
  state.covariance[packedCovarianceIndex(3, 3)] = o2::track::kCTgl2max;
  state.covariance[packedCovarianceIndex(4, 4)] = o2::track::kC1Pt2max * std::clamp(curvatureSquared, 0.0005f, 1.f);

  state.kind = kind;
  state.flags = 0;
  state.absCharge = kCompatibilityAbsCharge;
  state.pid = kCompatibilityPID;

  const std::array<const SurfaceMeasurement*, 2> attachmentMeasurements{measurements[1], measurements[0]};
  const std::array<const SurfaceDescriptor*, 2> attachmentSurfaces{surfaces[1], surfaces[0]};
  for (int step = 0; step < 2; ++step) {
    const auto& targetSurface = *attachmentSurfaces[step];
    if (!Propagator::attachMeasurement(
          state, targetSurface, *attachmentMeasurements[step], context.bz,
          material::MaterialTraversalDirection::OppositeMomentum,
          step == 1,
          context.configuration.kernelParameters.maxChi2ClusterAttachment,
          chi2, reason)) {
      return false;
    }
  }

  output = TrackSeed{cell, state, chi2};
  return true;
}

template <typename InputSeed>
void TrackerTraits::processNeighbours(IterationContext& context, int iteration, CellPathId startingPath,
                                      int defaultCellPathId, int startLevel, int currentLevel,
                                      const bounded_vector<InputSeed>& currentCellSeed,
                                      const bounded_vector<int>& currentCellId,
                                      const bounded_vector<int>& currentCellPathId,
                                      bounded_vector<TrackSeed>& updatedCellSeeds,
                                      bounded_vector<int>& updatedCellsIds,
                                      bounded_vector<int>& updatedCellsPathIds,
                                      const TrackingKernelParameters& params)
{
  auto* scratch = &context.scratch;
  const auto& mMemoryPool = scratch->getMemoryPool();
  const auto mBz = context.bz;
  const auto& mLayerGlobalMeasurements = context.layerGlobalMeasurements;
  const int activeSurfaceCount = context.configuration.topology.nLayers;

  mTaskArena->execute([&] {
    auto forCellNeighbours = [&](int iCell, auto&& emit) {
      const auto& currentCell{currentCellSeed[iCell]};
      const int cellPathId = currentCellPathId.empty() ? defaultCellPathId : currentCellPathId[iCell];

      if (currentCell.getLevel() != currentLevel) {
        return;
      }
      if (currentCellId.empty()) {
        for (int layer = 0; layer < activeSurfaceCount; ++layer) {
          const int clusterIndex = currentCell.getCluster(layer);
          if (clusterIndex != o2::its::constants::UnusedIndex &&
              context.frame.isClusterUsed(layer, mLayerGlobalMeasurements[layer][clusterIndex].clusterId)) {
            return;
          }
        }
      }

      const int cellId = currentCellId.empty() ? iCell : currentCellId[iCell];
      if (cellPathId < 0 || scratch->getCellsNeighboursLUT()[cellPathId].empty()) {
        return;
      }
      const int startNeighbourId{cellId ? scratch->getCellsNeighboursLUT()[cellPathId][cellId - 1] : 0};
      const int endNeighbourId{scratch->getCellsNeighboursLUT()[cellPathId][cellId]};
      TrackSeed baseSeed{};
      if constexpr (std::is_same_v<InputSeed, CellSeed>) {
        OperationFailureReason buildReason{};
        if (!buildTrackSeed(context, cellPathId, currentCell, baseSeed, buildReason)) {
          return;
        }
      } else {
        baseSeed = currentCell;
      }
      for (int iNeighbourCell{startNeighbourId}; iNeighbourCell < endNeighbourId; ++iNeighbourCell) {
        const int neighbourCellPathId = scratch->getCellsNeighboursTopology()[cellPathId][iNeighbourCell];
        const int neighbourCellId = scratch->getCellsNeighbours()[cellPathId][iNeighbourCell];
        const auto& neighbourCell = scratch->getCells()[neighbourCellPathId][neighbourCellId];
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
        if (neighbourLayer < 0 || neighbourLayer >= activeSurfaceCount) {
          throw TraversalException{iteration, TraversalFailureReason::SparseTopologyMismatch};
        }
        const int neighbourCluster = neighbourCell.getFirstClusterIndex();
        const auto& neighbourGlobal = mLayerGlobalMeasurements[neighbourLayer][neighbourCluster];
        if (context.frame.isClusterUsed(neighbourLayer, neighbourGlobal.clusterId)) {
          continue;
        }

        /// Let's start the fitting procedure
        TrackSeed seed{baseSeed};
        seed.getTimeStamp() = currentCell.getTimeStamp();
        seed.getTimeStamp() += neighbourCell.getTimeStamp();

        const auto* measurement = context.frame.getSurfaceMeasurement(LayerId{static_cast<uint16_t>(neighbourLayer)}, neighbourGlobal.clusterId);
        if (measurement == nullptr) {
          continue;
        }
        float chi2 = seed.getChi2();
        OperationFailureReason attachReason{};
        const bool attached = Propagator::attachMeasurement(seed.state(), context.topology.getSurface(LayerId{static_cast<uint16_t>(neighbourLayer)}), *measurement, mBz,
                                                            material::MaterialTraversalDirection::OppositeMomentum, true,
                                                            params.maxChi2ClusterAttachment, chi2, attachReason);
        if (!attached) {
          continue;
        }
        seed.setChi2(chi2);

        seed.setCluster(neighbourLayer, neighbourCluster);
        auto hitLayerMask = seed.getHitLayerMask();
        hitLayerMask.set(neighbourLayer);
        seed.setHitLayerMask(hitLayerMask);
        seed.setLevel(neighbourCell.getLevel());
        seed.setFirstTrackletIndex(neighbourCell.getFirstTrackletIndex());
        seed.setSecondTrackletIndex(neighbourCell.getSecondTrackletIndex());
        emit(RoadSeedEmission{std::move(seed), neighbourCellId, neighbourCellPathId});
      }
    };

    const int nCells = static_cast<int>(currentCellSeed.size());
    const auto key = CapacityEstimator::makeKey(SlabSite::Roads, iteration,
                                                CapacityEstimator::makeVariant(startLevel, currentLevel),
                                                startingPath);
    const auto scale = static_cast<double>(nCells);
    const auto capacity = context.frame.getCapacityEstimator().capacity(key, scale);
    GroupedSlabSink<RoadSeedEmission> sink{{.capacity = capacity, .nThreads = std::max(1, mTaskArena->max_concurrency())}, mMemoryPool.get()};
    tbb::parallel_for(0, nCells, [&](const int iCell) {
      auto& handle = sink.local();
      handle.beginProducer(iCell);
      forCellNeighbours(iCell, [&handle](RoadSeedEmission emission) { handle.emplace(std::move(emission)); });
    });
    const auto stats = sink.stats();
    bounded_vector<int> lut{mMemoryPool.get()};
    bounded_vector<RoadSeedEmission> emissions{mMemoryPool.get()};
    sink.finalizeGrouped(static_cast<size_t>(nCells), lut, emissions);
    context.frame.getCapacityEstimator().update(key, scale, stats.requested, stats.capacity, stats.emitted,
                                                stats.spilled, stats.overflowed, stats.memoryLimited);
    updatedCellSeeds.reserve(emissions.size());
    updatedCellsIds.reserve(emissions.size());
    updatedCellsPathIds.reserve(emissions.size());
    for (auto& emission : emissions) {
      updatedCellSeeds.push_back(std::move(emission.seed));
      updatedCellsIds.push_back(emission.cellId);
      updatedCellsPathIds.push_back(emission.cellPathId);
    }
  });
}

void TrackerTraits::findRoads(IterationContext& context, const int iteration)
{
  auto* scratch = &context.scratch;
  const auto& mMemoryPool = scratch->getMemoryPool();
  const auto& trkParam = context.configuration.parameters;
  const auto mBz = context.bz;
  const auto& mTraversalGraph = context.topology;
  const auto& mKernelParameters = context.configuration.kernelParameters;
  const auto& mLayerGlobalMeasurements = context.layerGlobalMeasurements;
  const gsl::span<const CellPathId> roadStartCells = context.configuration.topology.roadStartPaths;
  const int activeSurfaceCount = context.configuration.topology.nLayers;
  bounded_vector<bounded_vector<int>> firstClusters(activeSurfaceCount, bounded_vector<int>(mMemoryPool.get()), mMemoryPool.get());
  firstClusters.resize(activeSurfaceCount);
  // Road starts are the binding's seeding-eligible sparse-plan subsequence.
  // CellPathId values use compact slots; LayerId directly indexes layout-owned
  // layer data.
  // Filter roads by absolute q/pT in parameters[4]'s units, identically for
  // both families. Non-finite values fail the finite-bound comparison.
  constexpr float maxAbsQOverPt = 1.e3f;
  const auto seedingLayerMask = context.topology.seedingLayers;
  const auto nonSeedingLayerMask = ~seedingLayerMask;
  const int cellsPerRoad = seedingLayerMask.count() - 2;
  const auto& componentOffsets = context.configuration.topology.roadStartComponentOffsets;
  const auto holeLayerMask = context.frame.getLayout().getHoleLayers();
  if (componentOffsets.empty() || componentOffsets.front() != 0 || componentOffsets.back() != roadStartCells.size()) {
    throw TraversalException{iteration, TraversalFailureReason::SparseTopologyMismatch};
  }
  for (size_t component = 0; component + 1 < componentOffsets.size(); ++component) {
    const auto componentRoadStarts = roadStartCells.subspan(componentOffsets[component],
                                                            componentOffsets[component + 1] - componentOffsets[component]);
    for (int startLevel{cellsPerRoad}; startLevel >= trkParam.CellMinimumLevel(); --startLevel) {

      auto seedFilter = [&](const auto& seed) {
        const auto hitLayerMask = seed.getHitLayerMask();
        const int effectiveTrackLength = hitLayerMask.empty()
                                           ? 0
                                           : hitLayerMask.length() - (LayerMask::span(hitLayerMask.first(), hitLayerMask.last()) & nonSeedingLayerMask).count();
        const auto effectiveHoleMask = hitLayerMask.holeMask() & ~nonSeedingLayerMask;
        return effectiveHoleMask.isAllowedHoleMask(trkParam.MaxHoles, holeLayerMask) &&
               effectiveTrackLength >= trkParam.getMinSeedingClusters() &&
               std::abs(seed.getQOverPt()) <= maxAbsQOverPt && seed.getChi2() <= trkParam.MaxChi2NDF * ((startLevel + 2) * 2 - 5);
      };

      bounded_vector<TrackSeed> trackSeeds(mMemoryPool.get());
      // The binding supplies the ownership-filtered road-start span.
      for (const auto startId : componentRoadStarts) {
        // Cell population is per-event/per-vertex data, so check it against
        // the current vertex rather than caching it in the pass plan.
        if (scratch->getCells()[startId.value()].empty()) {
          continue;
        }

        bounded_vector<int> lastCellId(mMemoryPool.get()), updatedCellId(mMemoryPool.get());
        bounded_vector<int> lastCellPathId(mMemoryPool.get()), updatedCellPathId(mMemoryPool.get());
        bounded_vector<TrackSeed> lastCellSeed(mMemoryPool.get()), updatedCellSeed(mMemoryPool.get());

        processNeighbours(context, iteration, startId, startId.value(), startLevel, startLevel,
                          scratch->getCells()[startId.value()], lastCellId, lastCellPathId,
                          updatedCellSeed, updatedCellId, updatedCellPathId, mKernelParameters);

        int level = startLevel;
        while (level > 2 && !updatedCellSeed.empty()) {
          lastCellSeed.swap(updatedCellSeed);
          lastCellId.swap(updatedCellId);
          lastCellPathId.swap(updatedCellPathId);
          deepVectorClear(updatedCellSeed); /// tame the memory peaks
          deepVectorClear(updatedCellId);   /// tame the memory peaks
          deepVectorClear(updatedCellPathId);
          --level;
          processNeighbours(context, iteration, startId, o2::its::constants::UnusedIndex, startLevel, level,
                            lastCellSeed, lastCellId, lastCellPathId,
                            updatedCellSeed, updatedCellId, updatedCellPathId, mKernelParameters);
        }
        deepVectorClear(lastCellId);     /// tame the memory peaks
        deepVectorClear(lastCellPathId); /// tame the memory peaks
        deepVectorClear(lastCellSeed);   /// tame the memory peaks

        if (!updatedCellSeed.empty()) {
          trackSeeds.reserve(trackSeeds.size() + std::count_if(updatedCellSeed.begin(), updatedCellSeed.end(), seedFilter));
          std::copy_if(updatedCellSeed.begin(), updatedCellSeed.end(), std::back_inserter(trackSeeds), seedFilter);
        }
      }

      if (trackSeeds.empty()) {
        continue;
      }

      bounded_vector<TrackingCandidate> tracks(mMemoryPool.get());
      mTaskArena->execute([&] {
        const int nSeeds = static_cast<int>(trackSeeds.size());
        const auto key = CapacityEstimator::makeKey(SlabSite::Tracks, iteration,
                                                    CapacityEstimator::makeVariant(startLevel, static_cast<int>(component)), 0);
        const auto scale = static_cast<double>(nSeeds);
        const auto capacity = context.frame.getCapacityEstimator().capacity(key, scale);
        GroupedSlabSink<TrackingCandidate> sink{{.capacity = capacity, .nThreads = std::max(1, mTaskArena->max_concurrency())}, mMemoryPool.get()};
        tbb::parallel_for(0, nSeeds, [&](const int iSeed) {
          SurfaceTrackState innerState{};
          SurfaceTrackState outerState{};
          float chi2 = 0.f;
          OperationFailureReason reason{};
          if (!fitTrackSeedLegs(trackSeeds[iSeed], context.frame, mLayerGlobalMeasurements,
                                mTraversalGraph.getSurfaceCatalogView(), mBz,
                                trkParam.ShiftRefToCluster, trkParam.MaxChi2ClusterAttachment, trkParam.MaxChi2NDF,
                                trkParam.RepeatRefitOut, gsl::span<const float>(trkParam.MinPt),
                                innerState, outerState, chi2, reason)) {
            return;
          }
          TrackingCandidate temporaryTrack;
          temporaryTrack.seed = trackSeeds[iSeed];
          temporaryTrack.track.innerState = innerState;
          temporaryTrack.track.outerState = outerState;
          temporaryTrack.track.chi2 = chi2;
          temporaryTrack.charge = innerState.parameters[4] < 0.f ? -1 : 1;
          temporaryTrack.phi = innerState.kind == SurfaceKind::Cylinder ? std::asin(innerState.parameters[2]) + innerState.alpha : innerState.parameters[2];
          temporaryTrack.eta = std::asinh(innerState.parameters[3]);
          auto& handle = sink.local();
          handle.beginProducer(iSeed);
          handle.emplace(std::move(temporaryTrack));
        });
        const auto stats = sink.stats();
        bounded_vector<int> lut{mMemoryPool.get()};
        sink.finalizeGrouped(static_cast<size_t>(nSeeds), lut, tracks);
        context.frame.getCapacityEstimator().update(key, scale, stats.requested, stats.capacity, stats.emitted,
                                                    stats.spilled, stats.overflowed, stats.memoryLimited);
        deepVectorClear(trackSeeds);
      });

      // Same ordering as o2::its::track::isBetter (longer track, then lower chi2).
      std::sort(tracks.begin(), tracks.end(), [](const TrackingCandidate& a, const TrackingCandidate& b) {
        const auto ncla = a.getNumberOfClusters();
        const auto nclb = b.getNumberOfClusters();
        return (ncla == nclb) ? (a.track.chi2 < b.track.chi2) : ncla > nclb;
      });
      acceptTracks(context, iteration, tracks, firstClusters);
    }
  }
}

void TrackerTraits::acceptTracks(IterationContext& context, int iteration,
                                 bounded_vector<TrackingCandidate>& tracks,
                                 bounded_vector<bounded_vector<int>>& firstClusters)
{
  auto* scratch = &context.scratch;
  auto* mFrame = &context.frame;
  const auto& trkParam = context.configuration.parameters;
  const auto& mLayerGlobalMeasurements = context.layerGlobalMeasurements;
  const int activeSurfaceCount = context.configuration.topology.nLayers;
  reserveGenericTrackPublication(*mFrame, tracks.size(), static_cast<std::size_t>(activeSurfaceCount));
  for (auto& track : tracks) {
    int nShared = 0;
    bool isFirstShared{false};
    int firstLayer{-1}, firstCluster{-1};
    for (int iLayer{0}; iLayer < activeSurfaceCount; ++iLayer) {
      if (track.getClusterIndex(iLayer) == o2::its::constants::UnusedIndex) {
        continue;
      }
      const auto clusterId = mLayerGlobalMeasurements[iLayer][track.getClusterIndex(iLayer)].clusterId;
      bool isShared = mFrame->isClusterUsed(iLayer, clusterId);
      nShared += int(isShared);
      if (firstLayer < 0) {
        firstCluster = track.getClusterIndex(iLayer);
        isFirstShared = isShared && trkParam.AllowSharingFirstCluster && std::find(firstClusters[iLayer].begin(), firstClusters[iLayer].end(), firstCluster) != firstClusters[iLayer].end();
        firstLayer = iLayer;
      }
    }

    /// do not account for the first cluster in the shared clusters number if it is allowed
    if (nShared - int(isFirstShared && trkParam.AllowSharingFirstCluster) > trkParam.SharedMaxClusters) {
      continue;
    }

    bool firstCls{true}, nominalCompatible{true};
    TimeEstBC nominalTS, expandedTS;
    float smallestROFHalf = std::numeric_limits<float>::max();
    for (int iLayer{0}; iLayer < activeSurfaceCount; ++iLayer) {
      if (track.getClusterIndex(iLayer) == o2::its::constants::UnusedIndex) {
        continue;
      }
      smallestROFHalf = std::min(smallestROFHalf, mFrame->getROFTiming(iLayer).mROFLength * 0.5f);
      const auto clusterId = mLayerGlobalMeasurements[iLayer][track.getClusterIndex(iLayer)].clusterId;
      mFrame->markUsedCluster(iLayer, clusterId);
      int currentROF = mFrame->getClusterROF(iLayer, track.getClusterIndex(iLayer));
      const auto nominalROFTS = mFrame->getROFTiming(iLayer).getROFTimeBounds(currentROF);
      const auto expandedROFTS = mFrame->getROFTiming(iLayer).getROFTimeBounds(currentROF, true);
      if (firstCls) {
        firstCls = false;
        nominalTS = nominalROFTS;
        expandedTS = expandedROFTS;
      } else {
        if (nominalCompatible) {
          if (nominalTS.isCompatible(nominalROFTS)) {
            nominalTS += nominalROFTS;
          } else {
            nominalCompatible = false;
          }
        }
        if (!expandedTS.isCompatible(expandedROFTS)) {
          LOGP(fatal, "TS {}+/-{} are incompatible with {}+/-{}, this should not happen!", expandedROFTS.getTimeStamp(), expandedROFTS.getTimeStampError(), expandedTS.getTimeStamp(), expandedTS.getTimeStampError());
        }
        expandedTS += expandedROFTS;
      }
    }
    const auto selectedTimestamp = nominalCompatible ? nominalTS : expandedTS;
    const auto selectedTimestampSymmetric = selectedTimestamp.makeSymmetrical();
    // This is the same sanity clamp as the legacy symmetric timestamp, but
    // committed directly to the detector-neutral GenericTrack interval.
    const float selectedTimestampError = std::min(selectedTimestampSymmetric.getTimeStampError(), smallestROFHalf);
    track.track.timestamp = {static_cast<TFBC>(selectedTimestampSymmetric.getTimeStamp() - selectedTimestampError),
                             static_cast<TFBC>(selectedTimestampSymmetric.getTimeStamp() + selectedTimestampError)};
    if (!appendGenericTrack(*mFrame, track, mLayerGlobalMeasurements)) {
      LOGP(fatal, "GenericTrack publication failed for an accepted CA track");
    }

    if (trkParam.AllowSharingFirstCluster) {
      firstClusters[firstLayer].push_back(firstCluster);
    }
  }
}

void TrackerTraits::setNThreads(int n, std::shared_ptr<tbb::task_arena>& arena)
{
#if defined(OPTIMISATION_OUTPUT)
  mTaskArena = std::make_shared<tbb::task_arena>(1);
#else
  if (arena == nullptr) {
    mTaskArena = std::make_shared<tbb::task_arena>(std::abs(n));
    LOGP(info, "Setting tracker with {} threads.", n);
  } else {
    mTaskArena = arena;
  }
#endif
}

} // namespace o2::itsmft::tracking
