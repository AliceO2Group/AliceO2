// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackFollower.h
/// \brief Hypothesis search used by CPU and GPU track extension.

#ifndef TRACKINGITSU_INCLUDE_TRACKFOLLOWER_H_
#define TRACKINGITSU_INCLUDE_TRACKFOLLOWER_H_

#include <cstdint>

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "DetectorsBase/Propagator.h"

#include "ITStracking/Cluster.h"
#include "ITStracking/Constants.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/MathUtils.h"
#include "ITStracking/ROFLookupTables.h"
#include "ITStracking/TrackExtensionHypothesis.h"
#include "ITStracking/TrackHelpers.h"

namespace o2::its
{

template <int NLayers>
GPUhdi() void keepTrackExtensionHypothesis(const TrackExtensionHypothesis<NLayers>& hypo,
                                           TrackExtensionHypothesis<NLayers>* keptHypotheses,
                                           int& nKeptHypotheses,
                                           const int maxHypotheses)
{
  if (nKeptHypotheses < maxHypotheses) {
    keptHypotheses[nKeptHypotheses++] = hypo;
    return;
  }

  int worst{0};
  for (int i{1}; i < nKeptHypotheses; ++i) {
    if (track::isBetter(keptHypotheses[worst].nClusters, keptHypotheses[worst].chi2, keptHypotheses[i].nClusters, keptHypotheses[i].chi2)) {
      worst = i;
    }
  }
  if (track::isBetter(hypo.nClusters, hypo.chi2, keptHypotheses[worst].nClusters, keptHypotheses[worst].chi2)) {
    keptHypotheses[worst] = hypo;
  }
}

template <int NLayers>
GPUhdi() void updateTrackFromExtensionHypothesis(const TrackExtensionHypothesis<NLayers>& hypo,
                                                 const bool outward,
                                                 const int nLayers,
                                                 TrackITSInternal<NLayers>& track)
{
  if (outward) {
    track.paramOut = hypo.param;
  } else {
    track.paramIn = hypo.param;
  }
  track.time = hypo.time;
  track.setChi2(hypo.chi2);
  for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
    if (track.getClusterIndex(iLayer) == constants::UnusedIndex && hypo.clusters[iLayer] != constants::UnusedIndex) {
      track.setClusterIndex(iLayer, hypo.clusters[iLayer]);
    }
  }
}

// Search-specific inputs for track extension: cluster/ROF/index tables, layer
// radii, and phi/z search cuts. Kept separate from the fit context so that
// refit-only callers don't have to carry these fields.
template <int NLayers>
struct TrackFollowContext {
  const IndexTableUtils<NLayers>* utils{nullptr};
  typename ROFMaskTable<NLayers>::View rofMask;
  typename ROFOverlapTable<NLayers>::View rofOverlaps;
  const Cluster* const* clusters{nullptr};
  const unsigned char* const* usedClusters{nullptr};
  const int* const* clustersIndexTables{nullptr};
  const int* const* ROFClusters{nullptr};
  const float* layerRadii{nullptr};
  int phiBins{0};
  int maxHypotheses{0};
  float nSigmaCutPhi{0.f};
  float nSigmaCutZ{0.f};
};

template <int NLayers>
struct TrackExtensionBestTrial {
  GPUdi() TrackExtensionBestTrial(uint32_t backupPattern, const track::TrackFitContext<NLayers>& fit)
    : backupPattern{backupPattern}, fit{fit}
  {
  }

  GPUdi() void update(TrackITSInternal<NLayers>& trial, TrackITSInternal<NLayers>& best, uint32_t& bestDiff) const
  {
    const auto diff = (trial.getPattern() & ~backupPattern) & TrackITS::getLayerPatternMask<NLayers>();
    if (!diff || !track::refitTrack(trial, fit)) {
      return;
    }
    if (track::isBetter(trial, best)) {
      best = trial;
      bestDiff = diff;
    }
  }

  uint32_t backupPattern{0};
  const track::TrackFitContext<NLayers>& fit;
};

template <int NLayers, typename FollowDirection, typename BestTrial>
GPUdi() void followTrackExtensionBranches(const TrackITSInternal<NLayers>& backup,
                                          const bool extendTop,
                                          const bool extendBot,
                                          const int nLayers,
                                          FollowDirection& followDirection,
                                          BestTrial& bestTrial,
                                          TrackITSInternal<NLayers>& best,
                                          uint32_t& bestDiff)
{
  const uint32_t lastLayer = static_cast<uint32_t>(nLayers - 1);
  TrackITSInternal<NLayers> topResult;
  TrackITSInternal<NLayers> botResult;
  bool hasTopResult{false};
  bool hasBotResult{false};

  if (extendTop && backup.getLastClusterLayer() != lastLayer) {
    auto candidate = backup;
    if (followDirection(candidate, true)) {
      topResult = candidate;
      hasTopResult = true;
      bestTrial.update(candidate, best, bestDiff);
    }
  }
  if (extendBot && backup.getFirstClusterLayer() != 0) {
    auto candidate = backup;
    if (followDirection(candidate, false)) {
      botResult = candidate;
      hasBotResult = true;
      bestTrial.update(candidate, best, bestDiff);
    }
  }
  if (extendTop && extendBot) {
    if (hasTopResult && topResult.getFirstClusterLayer() != 0) {
      auto candidate = topResult;
      if (followDirection(candidate, false)) {
        bestTrial.update(candidate, best, bestDiff);
      }
    }
    if (hasBotResult && botResult.getLastClusterLayer() != lastLayer) {
      auto candidate = botResult;
      if (followDirection(candidate, true)) {
        bestTrial.update(candidate, best, bestDiff);
      }
    }
  }
}

template <int NLayers>
GPUhdi() bool followTrackExtensionDirection(const TrackExtensionHypothesis<NLayers>& startHypothesis,
                                            const track::TrackFitContext<NLayers>& fit,
                                            const TrackFollowContext<NLayers>& ctx,
                                            const bool outward,
                                            TrackExtensionHypothesis<NLayers>* activeHypotheses,
                                            TrackExtensionHypothesis<NLayers>* nextHypotheses,
                                            TrackExtensionHypothesis<NLayers>& bestHypothesis)
{
  const auto& utils = *ctx.utils;
  const int step = outward ? 1 : -1;
  const int end = outward ? fit.nLayers - 1 : 0;
  const int maxHypotheses = o2::gpu::CAMath::Max(ctx.maxHypotheses, 1);
  int nActive{1};
  int nNext{0};
  activeHypotheses[0] = startHypothesis;

  const int tableSize = utils.getNphiBins() * utils.getNzBins() + 1;
  for (int iLayer = activeHypotheses[0].edgeLayer + step; nActive > 0; iLayer += step) {
    if ((step > 0 && iLayer > end) || (step < 0 && iLayer < end)) {
      break;
    }
    nNext = 0;
    for (int iHypo{0}; iHypo < nActive; ++iHypo) {
      auto hypo = activeHypotheses[iHypo];
      const float r = ctx.layerRadii[iLayer];
      float x{-999.f};
      if (!hypo.param.getXatLabR(r, x, fit.bz, o2::track::DirAuto) || x <= 0.f) {
        continue;
      }

      if (!fit.propagator->propagateToX(hypo.param, x, fit.bz, o2::base::PropagatorF::MAX_SIN_PHI,
                                        o2::base::PropagatorF::MAX_STEP, fit.matCorrType)) {
        continue;
      }
      if (fit.matCorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE &&
          !hypo.param.correctForMaterial(fit.layerxX0[iLayer], fit.layerxX0[iLayer] * constants::Radl * constants::Rho, true)) {
        continue;
      }

      const float ePhi{o2::gpu::CAMath::Sqrt(hypo.param.getSigmaSnp2() / hypo.param.getCsp2())};
      const float eZ{o2::gpu::CAMath::Sqrt(hypo.param.getSigmaZ2())};
      const int4 selectedBins = getBinsRect(iLayer, hypo.param.getPhi(), hypo.param.getZ(), ctx.nSigmaCutZ * eZ, ctx.nSigmaCutPhi * ePhi, utils);
      if (selectedBins.x < 0) {
        continue;
      }

      int phiBinsNum = selectedBins.w - selectedBins.y + 1;
      if (phiBinsNum < 0) {
        phiBinsNum += ctx.phiBins;
      }

      const auto rofRange = ctx.rofOverlaps.getLayer(iLayer).getROFRange(hypo.time);
      for (int rof = rofRange.getFirstEntry(); rof < rofRange.getEntriesBound(); ++rof) {
        if (!ctx.rofMask.isROFEnabled(iLayer, rof)) {
          continue;
        }
        const int rofStart = ctx.ROFClusters[iLayer][rof];
        const int nLayerClusters = ctx.ROFClusters[iLayer][rof + 1] - rofStart;
        if (nLayerClusters <= 0) {
          continue;
        }
        const Cluster* layerClusters = ctx.clusters[iLayer] + rofStart;
        const int* indexTable = ctx.clustersIndexTables[iLayer] + rof * tableSize;
        const int zBinRange = selectedBins.z - selectedBins.x + 1;
        for (int iPhiCount = 0; iPhiCount < phiBinsNum; ++iPhiCount) {
          const int iPhiBin = (selectedBins.y + iPhiCount) % ctx.phiBins;
          const int firstBinIndex = utils.getBinIndex(selectedBins.x, iPhiBin);
          const int maxBinIndex = firstBinIndex + zBinRange;
          const int firstRowClusterIndex = indexTable[firstBinIndex];
          const int maxRowClusterIndex = indexTable[maxBinIndex];
          for (int iNextCluster{firstRowClusterIndex}; iNextCluster < maxRowClusterIndex; ++iNextCluster) {
            if (iNextCluster >= nLayerClusters) {
              break;
            }
            const Cluster& nextCluster = layerClusters[iNextCluster];
            if (ctx.usedClusters[iLayer][nextCluster.clusterId]) {
              continue;
            }

            const TrackingFrameInfo& trackingHit = fit.tfInfos[iLayer][nextCluster.clusterId];
            auto updated = hypo;
            if (!updated.param.rotate(trackingHit.alphaTrackingFrame) ||
                !fit.propagator->propagateToX(updated.param, trackingHit.xTrackingFrame, fit.bz,
                                              o2::base::PropagatorF::MAX_SIN_PHI,
                                              o2::base::PropagatorF::MAX_STEP,
                                              fit.matCorrType)) {
              continue;
            }

            const auto predChi2 = updated.param.getPredictedChi2Quiet(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame);
            if (predChi2 < 0.f || predChi2 > fit.maxChi2ClusterAttachment) {
              continue;
            }
            if (!updated.param.o2::track::TrackParCov::update(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)) {
              continue;
            }
            updated.chi2 += predChi2;
            updated.clusters[iLayer] = nextCluster.clusterId;
            ++updated.nClusters;
            updated.edgeLayer = iLayer;
            updated.time += ctx.rofOverlaps.getLayer(iLayer).getROFTimeBounds(rof, true);
            keepTrackExtensionHypothesis(updated, nextHypotheses, nNext, maxHypotheses);
          }
        }
      }
      keepTrackExtensionHypothesis(hypo, nextHypotheses, nNext, maxHypotheses);
    }
    if (nNext == 0) {
      break;
    }
    for (int iHypo{0}; iHypo < nNext; ++iHypo) {
      activeHypotheses[iHypo] = nextHypotheses[iHypo];
    }
    nActive = nNext;
  }

  const TrackExtensionHypothesis<NLayers>* bestHypo{nullptr};
  for (int iHypo{0}; iHypo < nActive; ++iHypo) {
    const auto& hypo = activeHypotheses[iHypo];
    if (hypo.nClusters == startHypothesis.nClusters) {
      continue;
    }
    const float maxChi2 = fit.maxChi2NDF * static_cast<float>(hypo.nClusters * 2 - 5);
    if (hypo.chi2 >= maxChi2) {
      continue;
    }
    if (!bestHypo || track::isBetter(hypo.nClusters, hypo.chi2, bestHypo->nClusters, bestHypo->chi2)) {
      bestHypo = &hypo;
    }
  }
  if (!bestHypo) {
    return false;
  }

  bestHypothesis = *bestHypo;
  return true;
}

} // namespace o2::its

#endif // TRACKINGITSU_INCLUDE_TRACKFOLLOWER_H_
