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
#include "ITStracking/TrackFollower.h"
#include "ITStracking/TrackHelpers.h"
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

template <int NLayers>
struct seed_selector {
  float mMaxQ2Pt;
  float mMaxChi2;
  int mMaxHoles;
  int mMinTrackLength;
  LayerMask mHoleLayerMask;

  GPUhd() seed_selector(float maxQ2Pt, float maxChi2, int maxHoles, int minTrackLength, LayerMask holeLayerMask) : mMaxQ2Pt(maxQ2Pt), mMaxChi2(maxChi2), mMaxHoles(maxHoles), mMinTrackLength(minTrackLength), mHoleLayerMask(holeLayerMask) {}
  GPUhd() bool operator()(const TrackSeed<NLayers>& seed) const
  {
    return !(seed.getQ2Pt() > mMaxQ2Pt || seed.getChi2() > mMaxChi2) &&
           seed.getHitLayerMask().length() >= mMinTrackLength &&
           seed.getHitLayerMask().isAllowed(mMaxHoles, mHoleLayerMask);
  }
};

struct compare_track_chi2 {
  GPUhd() bool operator()(const TrackITSExt& a, const TrackITSExt& b) const
  {
    return o2::its::track::isBetter(a, b);
  }
};

template <int NLayers>
GPUdi() void writeTrackExtensionCandidate(const int trackIndex,
                                          const TrackITSExt& original,
                                          const TrackITSExt& updated,
                                          TrackExtensionCandidate<NLayers>* candidates,
                                          int& slot)
{
  if (slot >= MaxTrackExtensionCandidatesPerTrack) {
    return;
  }
  auto& candidate = candidates[getFlatTrackExtensionCandidateIndex(trackIndex, slot)];
  candidate.reset();
  candidate.trackIndex = trackIndex;
  for (int iLayer{0}; iLayer < NLayers; ++iLayer) {
    if (original.getClusterIndex(iLayer) == constants::UnusedIndex && updated.getClusterIndex(iLayer) != constants::UnusedIndex) {
      candidate.addedClusters[iLayer] = updated.getClusterIndex(iLayer);
      ++candidate.nAddedClusters;
    }
  }
  if (!candidate.nAddedClusters) {
    candidate.reset();
    return;
  }
  candidate.chi2 = updated.getChi2();
  ++slot;
}

template <int NLayers>
GPUg() void __launch_bounds__(256, 1) computeTrackExtensionCandidatesKernel(const TrackITSExt* tracks,
                                                                            const IndexTableUtils<NLayers>* utils,
                                                                            const typename ROFMaskTable<NLayers>::View rofMask,
                                                                            const typename ROFOverlapTable<NLayers>::View rofOverlaps,
                                                                            const Cluster** clusters,
                                                                            const unsigned char** usedClusters,
                                                                            const int** clustersIndexTables,
                                                                            const int** ROFClusters,
                                                                            const TrackingFrameInfo** trackingFrameInfo,
                                                                            TrackExtensionCandidate<NLayers>* candidates,
                                                                            int* candidateOffsets,
                                                                            TrackExtensionHypothesis<NLayers>* activeHypothesesScratch,
                                                                            TrackExtensionHypothesis<NLayers>* nextHypothesesScratch,
                                                                            const std::array<float, NLayers> layerRadii,
                                                                            const std::array<float, NLayers> layerxX0,
                                                                            const int nTracks,
                                                                            const int nLayers,
                                                                            const int phiBins,
                                                                            const int beamWidth,
                                                                            const bool extendTop,
                                                                            const bool extendBot,
                                                                            const float bz,
                                                                            const float maxChi2ClusterAttachment,
                                                                            const float maxChi2NDF,
                                                                            const float nSigmaCutPhi,
                                                                            const float nSigmaCutZ,
                                                                            const o2::base::Propagator* propagator,
                                                                            const o2::base::PropagatorF::MatCorrType matCorrType)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    candidateOffsets[nTracks] = 0;
  }
  const int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  auto* const threadActiveHypotheses = activeHypothesesScratch + (globalThreadId * beamWidth);
  auto* const threadNextHypotheses = nextHypothesesScratch + (globalThreadId * beamWidth);
  for (int iTrack = globalThreadId; iTrack < nTracks; iTrack += blockDim.x * gridDim.x) {
    for (int iCandidate{0}; iCandidate < MaxTrackExtensionCandidatesPerTrack; ++iCandidate) {
      candidates[getFlatTrackExtensionCandidateIndex(iTrack, iCandidate)].reset();
    }
    const auto& track = tracks[iTrack];
    auto* activeHypotheses = threadActiveHypotheses;
    auto* nextHypotheses = threadNextHypotheses;
    int slot{0};
    if (extendTop && getTrackExtensionLastClusterLayer<NLayers>(track) != nLayers - 1) {
      TrackITSExt topCandidate;
      if (followTrackExtensionDirection(track, *utils, rofMask, rofOverlaps, clusters, usedClusters, clustersIndexTables, ROFClusters, trackingFrameInfo, layerRadii.data(), layerxX0.data(), nLayers, phiBins, beamWidth, bz, maxChi2ClusterAttachment, maxChi2NDF, nSigmaCutPhi, nSigmaCutZ, true, propagator, matCorrType, activeHypotheses, nextHypotheses, topCandidate)) {
        writeTrackExtensionCandidate(iTrack, track, topCandidate, candidates, slot);
        if (extendBot && getTrackExtensionFirstClusterLayer<NLayers>(topCandidate) != 0) {
          TrackITSExt topBottomCandidate;
          if (followTrackExtensionDirection(topCandidate, *utils, rofMask, rofOverlaps, clusters, usedClusters, clustersIndexTables, ROFClusters, trackingFrameInfo, layerRadii.data(), layerxX0.data(), nLayers, phiBins, beamWidth, bz, maxChi2ClusterAttachment, maxChi2NDF, nSigmaCutPhi, nSigmaCutZ, false, propagator, matCorrType, activeHypotheses, nextHypotheses, topBottomCandidate)) {
            writeTrackExtensionCandidate(iTrack, track, topBottomCandidate, candidates, slot);
          }
        }
      }
    }
    if (extendBot && getTrackExtensionFirstClusterLayer<NLayers>(track) != 0) {
      TrackITSExt bottomCandidate;
      if (followTrackExtensionDirection(track, *utils, rofMask, rofOverlaps, clusters, usedClusters, clustersIndexTables, ROFClusters, trackingFrameInfo, layerRadii.data(), layerxX0.data(), nLayers, phiBins, beamWidth, bz, maxChi2ClusterAttachment, maxChi2NDF, nSigmaCutPhi, nSigmaCutZ, false, propagator, matCorrType, activeHypotheses, nextHypotheses, bottomCandidate)) {
        writeTrackExtensionCandidate(iTrack, track, bottomCandidate, candidates, slot);
        if (extendTop && getTrackExtensionLastClusterLayer<NLayers>(bottomCandidate) != nLayers - 1) {
          TrackITSExt bottomTopCandidate;
          if (followTrackExtensionDirection(bottomCandidate, *utils, rofMask, rofOverlaps, clusters, usedClusters, clustersIndexTables, ROFClusters, trackingFrameInfo, layerRadii.data(), layerxX0.data(), nLayers, phiBins, beamWidth, bz, maxChi2ClusterAttachment, maxChi2NDF, nSigmaCutPhi, nSigmaCutZ, true, propagator, matCorrType, activeHypotheses, nextHypotheses, bottomTopCandidate)) {
            writeTrackExtensionCandidate(iTrack, track, bottomTopCandidate, candidates, slot);
          }
        }
      }
    }
    candidateOffsets[iTrack] = slot;
  }
}

template <int NLayers>
GPUdi() bool fitTrackExtensionResult(const TrackITSExt& startTrack,
                                     const TrackExtensionCandidate<NLayers>& candidate,
                                     const TrackingFrameInfo* const* trackingFrameInfo,
                                     const float* layerxX0,
                                     const int nLayers,
                                     const float bz,
                                     const float maxChi2ClusterAttachment,
                                     const float maxChi2NDF,
                                     const o2::base::Propagator* propagator,
                                     const o2::base::PropagatorF::MatCorrType matCorrType,
                                     const bool shiftRefToCluster,
                                     TrackITSExt& track)
{
  track = startTrack;
  for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
    if (candidate.addedClusters[iLayer] != constants::UnusedIndex) {
      track.setExternalClusterIndex(iLayer, candidate.addedClusters[iLayer], true);
    }
  }

  o2::track::TrackPar linRef{track};
  o2::its::track::resetTrackCovariance(track);
  track.setChi2(0);
  bool fitSuccess = o2::its::track::fitTrack(track,
                                             0,
                                             nLayers,
                                             1,
                                             maxChi2ClusterAttachment,
                                             maxChi2NDF,
                                             o2::constants::math::VeryBig,
                                             0,
                                             bz,
                                             trackingFrameInfo,
                                             layerxX0,
                                             propagator,
                                             matCorrType,
                                             &linRef,
                                             shiftRefToCluster);
  if (!fitSuccess) {
    return false;
  }

  track.getParamOut() = track.getParamIn();
  linRef = track.getParamOut();
  o2::its::track::resetTrackCovariance(track);
  track.setChi2(0);
  fitSuccess = o2::its::track::fitTrack(track,
                                        nLayers - 1,
                                        -1,
                                        -1,
                                        maxChi2ClusterAttachment,
                                        maxChi2NDF,
                                        50.f,
                                        0,
                                        bz,
                                        trackingFrameInfo,
                                        layerxX0,
                                        propagator,
                                        matCorrType,
                                        &linRef,
                                        shiftRefToCluster);
  if (!fitSuccess) {
    return false;
  }

  uint32_t diff{0};
  for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
    if (candidate.addedClusters[iLayer] != constants::UnusedIndex) {
      diff |= (0x1u << iLayer);
    }
  }
  applyExtendedClustersPattern<NLayers>(track, diff);
  return true;
}

template <int NLayers>
GPUdi() bool refitTrackExtensionResult(TrackITSExt& track,
                                       const TrackingFrameInfo* const* trackingFrameInfo,
                                       const float* layerxX0,
                                       const int nLayers,
                                       const float bz,
                                       const float maxChi2ClusterAttachment,
                                       const float maxChi2NDF,
                                       const o2::base::Propagator* propagator,
                                       const o2::base::PropagatorF::MatCorrType matCorrType,
                                       const bool shiftRefToCluster)
{
  o2::track::TrackPar linRef{track};
  o2::its::track::resetTrackCovariance(track);
  track.setChi2(0);
  bool fitSuccess = o2::its::track::fitTrack(track,
                                             0,
                                             nLayers,
                                             1,
                                             maxChi2ClusterAttachment,
                                             maxChi2NDF,
                                             o2::constants::math::VeryBig,
                                             0,
                                             bz,
                                             trackingFrameInfo,
                                             layerxX0,
                                             propagator,
                                             matCorrType,
                                             &linRef,
                                             shiftRefToCluster);
  if (!fitSuccess) {
    return false;
  }

  track.getParamOut() = track.getParamIn();
  linRef = track.getParamOut();
  o2::its::track::resetTrackCovariance(track);
  track.setChi2(0);
  return o2::its::track::fitTrack(track,
                                  nLayers - 1,
                                  -1,
                                  -1,
                                  maxChi2ClusterAttachment,
                                  maxChi2NDF,
                                  50.f,
                                  0,
                                  bz,
                                  trackingFrameInfo,
                                  layerxX0,
                                  propagator,
                                  matCorrType,
                                  &linRef,
                                  shiftRefToCluster);
}

template <int NLayers>
GPUdi() void finaliseTrackExtensionCandidate(const uint32_t backupPattern,
                                             TrackITSExt& candidate,
                                             const TrackingFrameInfo* const* trackingFrameInfo,
                                             const float* layerxX0,
                                             const int nLayers,
                                             const float bz,
                                             const float maxChi2ClusterAttachment,
                                             const float maxChi2NDF,
                                             const o2::base::Propagator* propagator,
                                             const o2::base::PropagatorF::MatCorrType matCorrType,
                                             const bool shiftRefToCluster,
                                             TrackITSExt& best)
{
  const auto diff = (candidate.getPattern() & ~backupPattern) & makeAddedClustersPatternMask<NLayers>();
  if (!diff || !refitTrackExtensionResult<NLayers>(candidate, trackingFrameInfo, layerxX0, nLayers, bz, maxChi2ClusterAttachment, maxChi2NDF, propagator, matCorrType, shiftRefToCluster)) {
    return;
  }
  applyExtendedClustersPattern<NLayers>(candidate, diff);
  if (o2::its::track::isBetter(candidate, best)) {
    best = candidate;
  }
}

template <int NLayers>
GPUg() void __launch_bounds__(256, 1) computeTrackExtensionResultsKernel(const TrackITSExt* tracks,
                                                                         const TrackExtensionCandidate<NLayers>* candidates,
                                                                         const int* candidateOffsets,
                                                                         TrackExtensionResult<NLayers>* results,
                                                                         const TrackingFrameInfo** trackingFrameInfo,
                                                                         const std::array<float, NLayers> layerxX0,
                                                                         const int nTracks,
                                                                         const int nLayers,
                                                                         const float bz,
                                                                         const float maxChi2ClusterAttachment,
                                                                         const float maxChi2NDF,
                                                                         const o2::base::Propagator* propagator,
                                                                         const o2::base::PropagatorF::MatCorrType matCorrType,
                                                                         const bool shiftRefToCluster)
{
  for (int iTrack = blockIdx.x * blockDim.x + threadIdx.x; iTrack < nTracks; iTrack += blockDim.x * gridDim.x) {
    const int firstResult = candidateOffsets[iTrack];
    const int nResults = candidateOffsets[iTrack + 1] - firstResult;
    const auto& startTrack = tracks[iTrack];
    for (int iCandidate{0}; iCandidate < nResults; ++iCandidate) {
      const auto& candidate = candidates[getFlatTrackExtensionCandidateIndex(iTrack, iCandidate)];
      auto& result = results[firstResult + iCandidate];
      result.reset();
      if (!candidate.isValidForTrack(iTrack)) {
        continue;
      }
      result.candidate = candidate;
      if (!fitTrackExtensionResult(startTrack,
                                   candidate,
                                   trackingFrameInfo,
                                   layerxX0.data(),
                                   nLayers,
                                   bz,
                                   maxChi2ClusterAttachment,
                                   maxChi2NDF,
                                   propagator,
                                   matCorrType,
                                   shiftRefToCluster,
                                   result.track)) {
        result.reset();
        continue;
      }
      result.candidate.chi2 = result.track.getChi2();
    }
  }
}

template <int NLayers>
GPUg() void __launch_bounds__(256, 1) countTrackSeedsKernel(
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
  for (int iCurrentTrackSeedIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentTrackSeedIndex < nSeeds; iCurrentTrackSeedIndex += blockDim.x * gridDim.x) {
    TrackITSExt temporaryTrack;
    if (o2::its::track::refitTrack(trackSeeds[iCurrentTrackSeedIndex],
                                   temporaryTrack,
                                   maxChi2ClusterAttachment,
                                   maxChi2NDF,
                                   bz,
                                   foundTrackingFrameInfo,
                                   unsortedClusters,
                                   layerxX0,
                                   layerRadii,
                                   minPts,
                                   propagator,
                                   matCorrType,
                                   reseedIfShorter,
                                   shiftRefToCluster,
                                   repeatRefitOut)) {
      seedLUT[iCurrentTrackSeedIndex] = 1;
    }
  }
}

template <int NLayers>
GPUg() void __launch_bounds__(256, 1) fitTrackSeedsKernel(
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
  const int beamWidthConfig,
  const bool extendTop,
  const bool extendBot,
  const float nSigmaCutPhi,
  const float nSigmaCutZ,
  const o2::base::Propagator* propagator,
  const o2::base::PropagatorF::MatCorrType matCorrType)
{
  for (int iCurrentTrackSeedIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentTrackSeedIndex < nSeeds; iCurrentTrackSeedIndex += blockDim.x * gridDim.x) {
    if (seedLUT[iCurrentTrackSeedIndex] == seedLUT[iCurrentTrackSeedIndex + 1]) {
      continue;
    }
    TrackITSExt temporaryTrack;
    bool refitSuccess = o2::its::track::refitTrack(trackSeeds[iCurrentTrackSeedIndex],
                                                   temporaryTrack,
                                                   maxChi2ClusterAttachment,
                                                   maxChi2NDF,
                                                   bz,
                                                   foundTrackingFrameInfo,
                                                   unsortedClusters,
                                                   layerxX0,
                                                   layerRadii,
                                                   minPts,
                                                   propagator,
                                                   matCorrType,
                                                   reseedIfShorter,
                                                   shiftRefToCluster,
                                                   repeatRefitOut);
    if (refitSuccess) {
      if ((extendTop || extendBot) && activeHypothesesScratch && nextHypothesesScratch) {
        const int beamWidth = o2::gpu::CAMath::Max(beamWidthConfig, 1);
        const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
        auto* activeHypotheses = activeHypothesesScratch + threadIndex * beamWidth;
        auto* nextHypotheses = nextHypothesesScratch + threadIndex * beamWidth;
        const auto backupPattern = temporaryTrack.getPattern();
        auto best = temporaryTrack;
        TrackITSExt topResult;
        TrackITSExt botResult;
        bool hasTopResult{false};
        bool hasBotResult{false};
        const uint32_t lastLayer = static_cast<uint32_t>(nLayers - 1);

        if (extendTop && getTrackExtensionLastClusterLayer<NLayers>(temporaryTrack) != lastLayer) {
          auto candidate = temporaryTrack;
          if (followTrackExtensionDirection<NLayers>(temporaryTrack,
                                                     *utils,
                                                     rofMask,
                                                     rofOverlaps,
                                                     clusters,
                                                     usedClusters,
                                                     clustersIndexTables,
                                                     ROFClusters,
                                                     foundTrackingFrameInfo,
                                                     layerRadii,
                                                     layerxX0,
                                                     nLayers,
                                                     phiBins,
                                                     beamWidth,
                                                     bz,
                                                     maxChi2ClusterAttachment,
                                                     maxChi2NDF,
                                                     nSigmaCutPhi,
                                                     nSigmaCutZ,
                                                     true,
                                                     propagator,
                                                     matCorrType,
                                                     activeHypotheses,
                                                     nextHypotheses,
                                                     candidate)) {
            topResult = candidate;
            hasTopResult = true;
            finaliseTrackExtensionCandidate<NLayers>(backupPattern, candidate, foundTrackingFrameInfo, layerxX0, nLayers, bz, maxChi2ClusterAttachment, maxChi2NDF, propagator, matCorrType, shiftRefToCluster, best);
          }
        }
        if (extendBot && getTrackExtensionFirstClusterLayer<NLayers>(temporaryTrack) != 0) {
          auto candidate = temporaryTrack;
          if (followTrackExtensionDirection<NLayers>(temporaryTrack,
                                                     *utils,
                                                     rofMask,
                                                     rofOverlaps,
                                                     clusters,
                                                     usedClusters,
                                                     clustersIndexTables,
                                                     ROFClusters,
                                                     foundTrackingFrameInfo,
                                                     layerRadii,
                                                     layerxX0,
                                                     nLayers,
                                                     phiBins,
                                                     beamWidth,
                                                     bz,
                                                     maxChi2ClusterAttachment,
                                                     maxChi2NDF,
                                                     nSigmaCutPhi,
                                                     nSigmaCutZ,
                                                     false,
                                                     propagator,
                                                     matCorrType,
                                                     activeHypotheses,
                                                     nextHypotheses,
                                                     candidate)) {
            botResult = candidate;
            hasBotResult = true;
            finaliseTrackExtensionCandidate<NLayers>(backupPattern, candidate, foundTrackingFrameInfo, layerxX0, nLayers, bz, maxChi2ClusterAttachment, maxChi2NDF, propagator, matCorrType, shiftRefToCluster, best);
          }
        }
        if (extendTop && extendBot) {
          if (hasTopResult && getTrackExtensionFirstClusterLayer<NLayers>(topResult) != 0) {
            auto candidate = topResult;
            if (followTrackExtensionDirection<NLayers>(topResult,
                                                       *utils,
                                                       rofMask,
                                                       rofOverlaps,
                                                       clusters,
                                                       usedClusters,
                                                       clustersIndexTables,
                                                       ROFClusters,
                                                       foundTrackingFrameInfo,
                                                       layerRadii,
                                                       layerxX0,
                                                       nLayers,
                                                       phiBins,
                                                       beamWidth,
                                                       bz,
                                                       maxChi2ClusterAttachment,
                                                       maxChi2NDF,
                                                       nSigmaCutPhi,
                                                       nSigmaCutZ,
                                                       false,
                                                       propagator,
                                                       matCorrType,
                                                       activeHypotheses,
                                                       nextHypotheses,
                                                       candidate)) {
              finaliseTrackExtensionCandidate<NLayers>(backupPattern, candidate, foundTrackingFrameInfo, layerxX0, nLayers, bz, maxChi2ClusterAttachment, maxChi2NDF, propagator, matCorrType, shiftRefToCluster, best);
            }
          }
          if (hasBotResult && getTrackExtensionLastClusterLayer<NLayers>(botResult) != lastLayer) {
            auto candidate = botResult;
            if (followTrackExtensionDirection<NLayers>(botResult,
                                                       *utils,
                                                       rofMask,
                                                       rofOverlaps,
                                                       clusters,
                                                       usedClusters,
                                                       clustersIndexTables,
                                                       ROFClusters,
                                                       foundTrackingFrameInfo,
                                                       layerRadii,
                                                       layerxX0,
                                                       nLayers,
                                                       phiBins,
                                                       beamWidth,
                                                       bz,
                                                       maxChi2ClusterAttachment,
                                                       maxChi2NDF,
                                                       nSigmaCutPhi,
                                                       nSigmaCutZ,
                                                       true,
                                                       propagator,
                                                       matCorrType,
                                                       activeHypotheses,
                                                       nextHypotheses,
                                                       candidate)) {
              finaliseTrackExtensionCandidate<NLayers>(backupPattern, candidate, foundTrackingFrameInfo, layerxX0, nLayers, bz, maxChi2ClusterAttachment, maxChi2NDF, propagator, matCorrType, shiftRefToCluster, best);
            }
          }
        }
        temporaryTrack = best;
      }
      tracks[seedLUT[iCurrentTrackSeedIndex]] = temporaryTrack;
    }
  }
}

template <bool initRun, int NLayers>
GPUg() void __launch_bounds__(256, 1) computeLayerCellNeighboursKernel(
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
GPUg() void __launch_bounds__(256, 1) computeLayerCellsKernel(
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
  const auto first = topology.getTransition(cellTopology.firstTransition);
  const auto second = topology.getTransition(cellTopology.secondTransition);
  const int layers[3] = {first.fromLayer, first.toLayer, second.toLayer};
  for (int iCurrentTrackletIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentTrackletIndex < nTrackletsCurrent; iCurrentTrackletIndex += blockDim.x * gridDim.x) {
    if constexpr (!initRun) {
      if (cellsLUTs[cellTopologyId][iCurrentTrackletIndex] == cellsLUTs[cellTopologyId][iCurrentTrackletIndex + 1]) {
        continue;
      }
    }
    const Tracklet& currentTracklet = tracklets[cellTopology.firstTransition][iCurrentTrackletIndex];
    const int nextLayerClusterIndex{currentTracklet.secondClusterIndex};
    const int nextLayerFirstTrackletIndex{trackletsLUT[cellTopology.secondTransition][nextLayerClusterIndex]};
    const int nextLayerLastTrackletIndex{trackletsLUT[cellTopology.secondTransition][nextLayerClusterIndex + 1]};
    if (nextLayerFirstTrackletIndex == nextLayerLastTrackletIndex) {
      continue;
    }
    int foundCells{0};
    for (int iNextTrackletIndex{nextLayerFirstTrackletIndex}; iNextTrackletIndex < nextLayerLastTrackletIndex; ++iNextTrackletIndex) {
      if (tracklets[cellTopology.secondTransition][iNextTrackletIndex].firstClusterIndex != nextLayerClusterIndex) {
        break;
      }
      const Tracklet& nextTracklet = tracklets[cellTopology.secondTransition][iNextTrackletIndex];
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
GPUg() void __launch_bounds__(256, 1) computeLayerTrackletsMultiROFKernel(
  const IndexTableUtils<NLayers>* utils,
  const typename ROFMaskTable<NLayers>::View rofMask,
  const int transitionId,
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
  const auto transition = topology.getTransition(transitionId);
  const int fromLayer = transition.fromLayer;
  const int toLayer = transition.toLayer;
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
        if (trackletsLUT[transitionId][currentSortedIndex] == trackletsLUT[transitionId][currentSortedIndex + 1]) {
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
                  trackletsLUT[transitionId][currentSortedIndex]++; // we need l0 as well for usual exclusive sums.
                } else {
                  const float phi{o2::gpu::CAMath::ATan2(currentCluster.yCoordinate - nextCluster.yCoordinate, currentCluster.xCoordinate - nextCluster.xCoordinate)};
                  const float tanL{(currentCluster.zCoordinate - nextCluster.zCoordinate) / (currentCluster.radius - nextCluster.radius)};
                  const int nextSortedIndex{ROFClusters[toLayer][targetROF] + nextClusterIndex};
                  new (tracklets[transitionId] + trackletsLUT[transitionId][currentSortedIndex] + storedTracklets) Tracklet{currentSortedIndex, nextSortedIndex, tanL, phi, ts};
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

GPUg() void __launch_bounds__(256, 1) compileTrackletsLookupTableKernel(
  const Tracklet* tracklets,
  int* trackletsLookUpTable,
  const int nTracklets)
{
  for (int currentTrackletIndex = blockIdx.x * blockDim.x + threadIdx.x; currentTrackletIndex < nTracklets; currentTrackletIndex += blockDim.x * gridDim.x) {
    atomicAdd(&trackletsLookUpTable[tracklets[currentTrackletIndex].firstClusterIndex], 1);
  }
}

template <bool dryRun, int NLayers, typename CurrentSeed>
GPUg() void __launch_bounds__(256, 1) processNeighboursKernel(
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
void computeTrackExtensionCandidatesHandler(const TrackITSExt* tracks,
                                            const IndexTableUtils<NLayers>* utils,
                                            const typename ROFMaskTable<NLayers>::View& rofMask,
                                            const typename ROFOverlapTable<NLayers>::View& rofOverlaps,
                                            const Cluster** clusters,
                                            const unsigned char** usedClusters,
                                            const int** clustersIndexTables,
                                            const int** ROFClusters,
                                            const TrackingFrameInfo** trackingFrameInfo,
                                            TrackExtensionCandidate<NLayers>* candidates,
                                            int* candidateOffsets,
                                            TrackExtensionHypothesis<NLayers>* activeHypotheses,
                                            TrackExtensionHypothesis<NLayers>* nextHypotheses,
                                            const std::array<float, NLayers> layerRadii,
                                            const std::array<float, NLayers> layerxX0,
                                            const int nTracks,
                                            const int nLayers,
                                            const int phiBins,
                                            const int beamWidth,
                                            const bool extendTop,
                                            const bool extendBot,
                                            const float bz,
                                            const float maxChi2ClusterAttachment,
                                            const float maxChi2NDF,
                                            const float nSigmaCutPhi,
                                            const float nSigmaCutZ,
                                            const o2::base::Propagator* propagator,
                                            const o2::base::PropagatorF::MatCorrType matCorrType,
                                            gpu::Stream& stream)
{
  if (nTracks <= 0 || candidates == nullptr || candidateOffsets == nullptr || activeHypotheses == nullptr || nextHypotheses == nullptr) {
    return;
  }
  gpu::computeTrackExtensionCandidatesKernel<NLayers><<<kTrackExtensionLaunchBlocks, kTrackExtensionLaunchThreadsPerBlock, 0, stream.get()>>>(
    tracks,
    utils,
    rofMask,
    rofOverlaps,
    clusters,
    usedClusters,
    clustersIndexTables,
    ROFClusters,
    trackingFrameInfo,
    candidates,
    candidateOffsets,
    activeHypotheses,
    nextHypotheses,
    layerRadii,
    layerxX0,
    nTracks,
    nLayers,
    phiBins,
    beamWidth,
    extendTop,
    extendBot,
    bz,
    maxChi2ClusterAttachment,
    maxChi2NDF,
    nSigmaCutPhi,
    nSigmaCutZ,
    propagator,
    matCorrType);
  GPUChkErrS(cudaGetLastError());
  GPUChkErrS(cudaStreamSynchronize(stream.get()));
  thrust::device_ptr<int> offsets(candidateOffsets);
  thrust::exclusive_scan(offsets, offsets + nTracks + 1, offsets);
}

template <int NLayers>
void computeTrackExtensionResultsHandler(const TrackITSExt* tracks,
                                         const TrackExtensionCandidate<NLayers>* candidates,
                                         const int* candidateOffsets,
                                         TrackExtensionResult<NLayers>* results,
                                         const TrackingFrameInfo** trackingFrameInfo,
                                         const std::array<float, NLayers> layerxX0,
                                         const int nTracks,
                                         const int nLayers,
                                         const float bz,
                                         const float maxChi2ClusterAttachment,
                                         const float maxChi2NDF,
                                         const o2::base::Propagator* propagator,
                                         const o2::base::PropagatorF::MatCorrType matCorrType,
                                         const bool shiftRefToCluster,
                                         gpu::Stream& stream)
{
  if (nTracks <= 0 || tracks == nullptr || candidates == nullptr || candidateOffsets == nullptr || results == nullptr) {
    return;
  }
  gpu::computeTrackExtensionResultsKernel<NLayers><<<kTrackExtensionLaunchBlocks, kTrackExtensionLaunchThreadsPerBlock, 0, stream.get()>>>(
    tracks,
    candidates,
    candidateOffsets,
    results,
    trackingFrameInfo,
    layerxX0,
    nTracks,
    nLayers,
    bz,
    maxChi2ClusterAttachment,
    maxChi2NDF,
    propagator,
    matCorrType,
    shiftRefToCluster);
  GPUChkErrS(cudaGetLastError());
  GPUChkErrS(cudaStreamSynchronize(stream.get()));
}

template <int NLayers>
void countTrackletsInROFsHandler(const IndexTableUtils<NLayers>* utils,
                                 const typename ROFMaskTable<NLayers>::View& rofMask,
                                 const int transitionId,
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
                                 bounded_vector<float>& transitionPhiCuts,
                                 const float resolutionPV,
                                 std::array<float, NLayers>& minRs,
                                 std::array<float, NLayers>& maxRs,
                                 bounded_vector<float>& resolutions,
                                 std::vector<float>& radii,
                                 bounded_vector<float>& transitionMSAngles,
                                 o2::its::ExternalAllocator* alloc,
                                 gpu::Streams& streams)
{
  gpu::computeLayerTrackletsMultiROFKernel<true><<<60, 256, 0, streams[transitionId].get()>>>(
    utils,
    rofMask,
    transitionId,
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
    transitionPhiCuts[transitionId],
    resolutionPV,
    minRs[toLayer],
    maxRs[toLayer],
    resolutions[fromLayer],
    radii[toLayer] - radii[fromLayer],
    transitionMSAngles[transitionId]);
  auto nosync_policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(streams[transitionId].get());
  thrust::exclusive_scan(nosync_policy, trackletsLUTsHost[transitionId], trackletsLUTsHost[transitionId] + nClusters[fromLayer] + 1, trackletsLUTsHost[transitionId]);
}

template <int NLayers>
void computeTrackletsInROFsHandler(const IndexTableUtils<NLayers>* utils,
                                   const typename ROFMaskTable<NLayers>::View& rofMask,
                                   const int transitionId,
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
                                   bounded_vector<float>& transitionPhiCuts,
                                   const float resolutionPV,
                                   std::array<float, NLayers>& minRs,
                                   std::array<float, NLayers>& maxRs,
                                   bounded_vector<float>& resolutions,
                                   std::vector<float>& radii,
                                   bounded_vector<float>& transitionMSAngles,
                                   o2::its::ExternalAllocator* alloc,
                                   gpu::Streams& streams)
{
  gpu::computeLayerTrackletsMultiROFKernel<false><<<60, 256, 0, streams[transitionId].get()>>>(
    utils,
    rofMask,
    transitionId,
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
    transitionPhiCuts[transitionId],
    resolutionPV,
    minRs[toLayer],
    maxRs[toLayer],
    resolutions[fromLayer],
    radii[toLayer] - radii[fromLayer],
    transitionMSAngles[transitionId]);
  thrust::device_ptr<Tracklet> tracklets_ptr(spanTracklets[transitionId]);
  auto nosync_policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(streams[transitionId].get());
  thrust::sort(nosync_policy, tracklets_ptr, tracklets_ptr + nTracklets[transitionId]);
  auto unique_end = thrust::unique(nosync_policy, tracklets_ptr, tracklets_ptr + nTracklets[transitionId]);
  nTracklets[transitionId] = unique_end - tracklets_ptr;
  if (fromLayer > 0) {
    GPUChkErrS(cudaMemsetAsync(trackletsLUTsHost[transitionId], 0, (nClusters[fromLayer] + 1) * sizeof(int), streams[transitionId].get()));
    gpu::compileTrackletsLookupTableKernel<<<60, 256, 0, streams[transitionId].get()>>>(
      spanTracklets[transitionId],
      trackletsLUTsHost[transitionId],
      nTracklets[transitionId]);
    thrust::exclusive_scan(nosync_policy, trackletsLUTsHost[transitionId], trackletsLUTsHost[transitionId] + nClusters[fromLayer] + 1, trackletsLUTsHost[transitionId]);
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
  gpu::computeLayerCellsKernel<true, NLayers><<<60, 256, 0, streams[cellTopologyId].get()>>>(
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
  gpu::computeLayerCellsKernel<false, NLayers><<<60, 256, 0, streams[cellTopologyId].get()>>>(
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
  gpu::computeLayerCellNeighboursKernel<true, NLayers><<<60, 256, 0, stream.get()>>>(
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
  gpu::computeLayerCellNeighboursKernel<false, NLayers><<<60, 256, 0, stream.get()>>>(
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
                              const int minTrackLength,
                              const LayerMask holeLayerMask,
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

  gpu::processNeighboursKernel<true, NLayers, CellSeed><<<60, 256>>>(
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
  gpu::processNeighboursKernel<false, NLayers, CellSeed><<<60, 256>>>(
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
    gpu::processNeighboursKernel<true, NLayers, TrackSeed<NLayers>><<<60, 256>>>(
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

    gpu::processNeighboursKernel<false, NLayers, TrackSeed<NLayers>><<<60, 256>>>(
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
  auto end = thrust::copy_if(nosync_policy, updatedCellSeed.begin(), updatedCellSeed.end(), outSeeds.begin(), gpu::seed_selector<NLayers>(1.e3, maxChi2NDF * ((startLevel + 2) * 2 - 5), maxHoles, minTrackLength, holeLayerMask));
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
  gpu::countTrackSeedsKernel<NLayers><<<60, 256>>>(
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
                             const int beamWidth,
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
  gpu::fitTrackSeedsKernel<NLayers><<<60, 256>>>(
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
    beamWidth,                                // int
    extendTop,                                // bool
    extendBot,                                // bool
    nSigmaCutPhi,                             // float
    nSigmaCutZ,                               // float
    propagator,                               // const o2::base::Propagator*
    matCorrType);                             // o2::base::PropagatorF::MatCorrType
  auto sync_policy = THRUST_NAMESPACE::par(gpu::TypedAllocator<char>(alloc));
  thrust::device_ptr<o2::its::TrackITSExt> tr_ptr(tracks);
  thrust::sort(sync_policy, tr_ptr, tr_ptr + nTracks, gpu::compare_track_chi2());
}

/// Explicit instantiation of ITS2 handlers
template void computeTrackExtensionCandidatesHandler<7>(const TrackITSExt* tracks,
                                                        const IndexTableUtils<7>* utils,
                                                        const ROFMaskTable<7>::View& rofMask,
                                                        const ROFOverlapTable<7>::View& rofOverlaps,
                                                        const Cluster** clusters,
                                                        const unsigned char** usedClusters,
                                                        const int** clustersIndexTables,
                                                        const int** ROFClusters,
                                                        const TrackingFrameInfo** trackingFrameInfo,
                                                        TrackExtensionCandidate<7>* candidates,
                                                        int* candidateOffsets,
                                                        TrackExtensionHypothesis<7>* activeHypotheses,
                                                        TrackExtensionHypothesis<7>* nextHypotheses,
                                                        const std::array<float, 7> layerRadii,
                                                        const std::array<float, 7> layerxX0,
                                                        const int nTracks,
                                                        const int nLayers,
                                                        const int phiBins,
                                                        const int beamWidth,
                                                        const bool extendTop,
                                                        const bool extendBot,
                                                        const float bz,
                                                        const float maxChi2ClusterAttachment,
                                                        const float maxChi2NDF,
                                                        const float nSigmaCutPhi,
                                                        const float nSigmaCutZ,
                                                        const o2::base::Propagator* propagator,
                                                        const o2::base::PropagatorF::MatCorrType matCorrType,
                                                        gpu::Stream& stream);

template void computeTrackExtensionResultsHandler<7>(const TrackITSExt* tracks,
                                                     const TrackExtensionCandidate<7>* candidates,
                                                     const int* candidateOffsets,
                                                     TrackExtensionResult<7>* results,
                                                     const TrackingFrameInfo** trackingFrameInfo,
                                                     const std::array<float, 7> layerxX0,
                                                     const int nTracks,
                                                     const int nLayers,
                                                     const float bz,
                                                     const float maxChi2ClusterAttachment,
                                                     const float maxChi2NDF,
                                                     const o2::base::Propagator* propagator,
                                                     const o2::base::PropagatorF::MatCorrType matCorrType,
                                                     const bool shiftRefToCluster,
                                                     gpu::Stream& stream);

template void countTrackletsInROFsHandler<7>(const IndexTableUtils<7>* utils,
                                             const ROFMaskTable<7>::View& rofMask,
                                             const int transitionId,
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
                                             bounded_vector<float>& transitionPhiCuts,
                                             const float resolutionPV,
                                             std::array<float, 7>& minRs,
                                             std::array<float, 7>& maxRs,
                                             bounded_vector<float>& resolutions,
                                             std::vector<float>& radii,
                                             bounded_vector<float>& transitionMSAngles,
                                             o2::its::ExternalAllocator* alloc,
                                             gpu::Streams& streams);

template void computeTrackletsInROFsHandler<7>(const IndexTableUtils<7>* utils,
                                               const ROFMaskTable<7>::View& rofMask,
                                               const int transitionId,
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
                                               bounded_vector<float>& transitionPhiCuts,
                                               const float resolutionPV,
                                               std::array<float, 7>& minRs,
                                               std::array<float, 7>& maxRs,
                                               bounded_vector<float>& resolutions,
                                               std::vector<float>& radii,
                                               bounded_vector<float>& transitionMSAngles,
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
                                          const int minTrackLength,
                                          const LayerMask holeLayerMask,
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
                                      const int beamWidth,
                                      const bool extendTop,
                                      const bool extendBot,
                                      const float nSigmaCutPhi,
                                      const float nSigmaCutZ,
                                      const o2::base::Propagator* propagator,
                                      const o2::base::PropagatorF::MatCorrType matCorrType,
                                      o2::its::ExternalAllocator* alloc);

/// Explicit instantiation of ALICE3 handlers
#ifdef ENABLE_UPGRADES
template void computeTrackExtensionCandidatesHandler<11>(const TrackITSExt* tracks,
                                                         const IndexTableUtils<11>* utils,
                                                         const ROFMaskTable<11>::View& rofMask,
                                                         const ROFOverlapTable<11>::View& rofOverlaps,
                                                         const Cluster** clusters,
                                                         const unsigned char** usedClusters,
                                                         const int** clustersIndexTables,
                                                         const int** ROFClusters,
                                                         const TrackingFrameInfo** trackingFrameInfo,
                                                         TrackExtensionCandidate<11>* candidates,
                                                         int* candidateOffsets,
                                                         TrackExtensionHypothesis<11>* activeHypotheses,
                                                         TrackExtensionHypothesis<11>* nextHypotheses,
                                                         const std::array<float, 11> layerRadii,
                                                         const std::array<float, 11> layerxX0,
                                                         const int nTracks,
                                                         const int nLayers,
                                                         const int phiBins,
                                                         const int beamWidth,
                                                         const bool extendTop,
                                                         const bool extendBot,
                                                         const float bz,
                                                         const float maxChi2ClusterAttachment,
                                                         const float maxChi2NDF,
                                                         const float nSigmaCutPhi,
                                                         const float nSigmaCutZ,
                                                         const o2::base::Propagator* propagator,
                                                         const o2::base::PropagatorF::MatCorrType matCorrType,
                                                         gpu::Stream& stream);

template void computeTrackExtensionResultsHandler<11>(const TrackITSExt* tracks,
                                                      const TrackExtensionCandidate<11>* candidates,
                                                      const int* candidateOffsets,
                                                      TrackExtensionResult<11>* results,
                                                      const TrackingFrameInfo** trackingFrameInfo,
                                                      const std::array<float, 11> layerxX0,
                                                      const int nTracks,
                                                      const int nLayers,
                                                      const float bz,
                                                      const float maxChi2ClusterAttachment,
                                                      const float maxChi2NDF,
                                                      const o2::base::Propagator* propagator,
                                                      const o2::base::PropagatorF::MatCorrType matCorrType,
                                                      const bool shiftRefToCluster,
                                                      gpu::Stream& stream);

template void countTrackletsInROFsHandler<11>(const IndexTableUtils<11>* utils,
                                              const ROFMaskTable<11>::View& rofMask,
                                              const int transitionId,
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
                                              bounded_vector<float>& transitionPhiCuts,
                                              const float resolutionPV,
                                              std::array<float, 11>& minRs,
                                              std::array<float, 11>& maxRs,
                                              bounded_vector<float>& resolutions,
                                              std::vector<float>& radii,
                                              bounded_vector<float>& transitionMSAngles,
                                              o2::its::ExternalAllocator* alloc,
                                              gpu::Streams& streams);

template void computeTrackletsInROFsHandler<11>(const IndexTableUtils<11>* utils,
                                                const ROFMaskTable<11>::View& rofMask,
                                                const int transitionId,
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
                                                bounded_vector<float>& transitionPhiCuts,
                                                const float resolutionPV,
                                                std::array<float, 11>& minRs,
                                                std::array<float, 11>& maxRs,
                                                bounded_vector<float>& resolutions,
                                                std::vector<float>& radii,
                                                bounded_vector<float>& transitionMSAngles,
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
                                           const int minTrackLength,
                                           const LayerMask holeLayerMask,
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
                                      const int beamWidth,
                                      const bool extendTop,
                                      const bool extendBot,
                                      const float nSigmaCutPhi,
                                      const float nSigmaCutZ,
                                      const o2::base::Propagator* propagator,
                                      const o2::base::PropagatorF::MatCorrType matCorrType,
                                      o2::its::ExternalAllocator* alloc);
#endif
} // namespace o2::its
