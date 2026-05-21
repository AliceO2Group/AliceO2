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

/// \file TrackFollower.h
/// \brief Beam search used by CPU and GPU track extension.

#ifndef TRACKINGITSU_INCLUDE_TRACKFOLLOWER_H_
#define TRACKINGITSU_INCLUDE_TRACKFOLLOWER_H_

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "CommonConstants/MathConstants.h"
#include "DetectorsBase/Propagator.h"

#include "ITStracking/Cluster.h"
#include "ITStracking/Constants.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/MathUtils.h"
#include "ITStracking/ROFLookupTables.h"
#include "ITStracking/TrackExtensionCandidate.h"

namespace o2::its
{

template <int NLayers>
GPUhdi() bool isBetterTrackExtensionHypothesis(const TrackExtensionHypothesis<NLayers>& a, const TrackExtensionHypothesis<NLayers>& b)
{
  return (a.nClusters > b.nClusters) || (a.nClusters == b.nClusters && a.chi2 < b.chi2);
}

template <int NLayers>
GPUhdi() void addTrackExtensionHypothesisToBeam(const TrackExtensionHypothesis<NLayers>& hypo,
                                                TrackExtensionHypothesis<NLayers>* beam,
                                                int& nBeam,
                                                const int beamWidth)
{
  if (nBeam < beamWidth) {
    beam[nBeam++] = hypo;
    return;
  }

  int worst{0};
  for (int i{1}; i < nBeam; ++i) {
    if (isBetterTrackExtensionHypothesis(beam[worst], beam[i])) {
      worst = i;
    }
  }
  if (isBetterTrackExtensionHypothesis(hypo, beam[worst])) {
    beam[worst] = hypo;
  }
}

template <int NLayers>
GPUhdi() int4 getTrackExtensionBinsAt(const IndexTableUtils<NLayers>& utils,
                                      const int layer,
                                      const float phi,
                                      const float deltaPhi,
                                      const float z,
                                      const float deltaZ)
{
  const float zRangeMin = z - deltaZ;
  const float zRangeMax = z + deltaZ;
  if (zRangeMax < -utils.getLayerZ(layer) || zRangeMin > utils.getLayerZ(layer) || zRangeMin > zRangeMax) {
    return {-1, -1, -1, -1};
  }
  const float phiRangeMin = (deltaPhi > o2::constants::math::PI) ? 0.f : phi - deltaPhi;
  const float phiRangeMax = (deltaPhi > o2::constants::math::PI) ? o2::constants::math::TwoPI : phi + deltaPhi;
  return {o2::gpu::CAMath::Max(0, utils.getZBinIndex(layer, zRangeMin)),
          utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMin)),
          o2::gpu::CAMath::Min(utils.getNzBins() - 1, utils.getZBinIndex(layer, zRangeMax)),
          utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMax))};
}

template <int NLayers>
GPUhdi() int getTrackExtensionFirstClusterLayer(const TrackITSExt& track)
{
  const uint32_t pattern = track.getPattern();
  for (int iLayer{0}; iLayer < NLayers; ++iLayer) {
    if (pattern & (0x1u << iLayer)) {
      return iLayer;
    }
  }
  return constants::UnusedIndex;
}

template <int NLayers>
GPUhdi() int getTrackExtensionLastClusterLayer(const TrackITSExt& track)
{
  const uint32_t pattern = track.getPattern();
  for (int iLayer{NLayers}; iLayer-- > 0;) {
    if (pattern & (0x1u << iLayer)) {
      return iLayer;
    }
  }
  return constants::UnusedIndex;
}

template <int NLayers>
GPUhdi() void initialiseTrackExtensionHypothesis(const TrackITSExt& track,
                                                 const bool outward,
                                                 TrackExtensionHypothesis<NLayers>& hypo)
{
  hypo.param = outward ? track.getParamOut() : track.getParamIn();
  hypo.time = track.getTimeStamp();
  hypo.chi2 = track.getChi2();
  hypo.nClusters = track.getNClusters();
  hypo.edgeLayer = outward ? getTrackExtensionLastClusterLayer<NLayers>(track) : getTrackExtensionFirstClusterLayer<NLayers>(track);
  for (int iLayer{0}; iLayer < NLayers; ++iLayer) {
    hypo.clusters[iLayer] = track.getClusterIndex(iLayer);
  }
}

template <int NLayers>
GPUhdi() bool followTrackExtensionDirection(const TrackITSExt& track,
                                            const IndexTableUtils<NLayers>& utils,
                                            const typename ROFMaskTable<NLayers>::View& rofMask,
                                            const typename ROFOverlapTable<NLayers>::View& rofOverlaps,
                                            const Cluster* const* clusters,
                                            const unsigned char* const* usedClusters,
                                            const int* const* clustersIndexTables,
                                            const int* const* ROFClusters,
                                            const TrackingFrameInfo* const* trackingFrameInfo,
                                            const float* layerRadii,
                                            const float* layerxX0,
                                            const int nLayers,
                                            const int phiBins,
                                            const int beamWidthConfig,
                                            const float bz,
                                            const float maxChi2ClusterAttachment,
                                            const float maxChi2NDF,
                                            const float nSigmaCutPhi,
                                            const float nSigmaCutZ,
                                            const bool outward,
                                            const o2::base::Propagator* propagator,
                                            const o2::base::PropagatorF::MatCorrType matCorrType,
                                            TrackExtensionHypothesis<NLayers>* activeHypotheses,
                                            TrackExtensionHypothesis<NLayers>* nextHypotheses,
                                            TrackITSExt& updatedTrack)
{
  const int step = outward ? 1 : -1;
  const int end = outward ? nLayers - 1 : 0;
  const int beamWidth = o2::gpu::CAMath::Max(beamWidthConfig, 1);
  int nActive{1};
  int nNext{0};
  initialiseTrackExtensionHypothesis(track, outward, activeHypotheses[0]);

  const int tableSize = utils.getNphiBins() * utils.getNzBins() + 1;
  for (int iLayer = activeHypotheses[0].edgeLayer + step; nActive > 0; iLayer += step) {
    if ((step > 0 && iLayer > end) || (step < 0 && iLayer < end)) {
      break;
    }
    nNext = 0;
    for (int iHypo{0}; iHypo < nActive; ++iHypo) {
      auto hypo = activeHypotheses[iHypo];
      const float r = layerRadii[iLayer];
      float x{-999.f};
      if (!hypo.param.getXatLabR(r, x, bz, o2::track::DirAuto) || x <= 0.f) {
        continue;
      }

      if (!propagator->propagateToX(hypo.param, x, bz, o2::base::PropagatorF::MAX_SIN_PHI,
                                    o2::base::PropagatorF::MAX_STEP, matCorrType)) {
        continue;
      }
      if (matCorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE &&
          !hypo.param.correctForMaterial(layerxX0[iLayer], layerxX0[iLayer] * constants::Radl * constants::Rho, true)) {
        continue;
      }

      const float ePhi{o2::gpu::CAMath::Sqrt(hypo.param.getSigmaSnp2() / hypo.param.getCsp2())};
      const float eZ{o2::gpu::CAMath::Sqrt(hypo.param.getSigmaZ2())};
      const int4 selectedBins = getTrackExtensionBinsAt(utils,
                                                        iLayer,
                                                        hypo.param.getPhi(),
                                                        nSigmaCutPhi * ePhi,
                                                        hypo.param.getZ(),
                                                        nSigmaCutZ * eZ);
      if (selectedBins.x < 0) {
        continue;
      }

      int phiBinsNum = selectedBins.w - selectedBins.y + 1;
      if (phiBinsNum < 0) {
        phiBinsNum += phiBins;
      }

      const auto rofRange = rofOverlaps.getLayer(iLayer).getROFRange(hypo.time);
      for (int rof = rofRange.x; rof <= rofRange.y; ++rof) {
        if (!rofMask.isROFEnabled(iLayer, rof)) {
          continue;
        }
        const int rofStart = ROFClusters[iLayer][rof];
        const int nLayerClusters = ROFClusters[iLayer][rof + 1] - rofStart;
        if (nLayerClusters <= 0) {
          continue;
        }
        const Cluster* layerClusters = clusters[iLayer] + rofStart;
        const int* indexTable = clustersIndexTables[iLayer] + rof * tableSize;
        const int zBinRange = selectedBins.z - selectedBins.x + 1;
        for (int iPhiCount = 0; iPhiCount < phiBinsNum; ++iPhiCount) {
          const int iPhiBin = (selectedBins.y + iPhiCount) % phiBins;
          const int firstBinIndex = utils.getBinIndex(selectedBins.x, iPhiBin);
          const int maxBinIndex = firstBinIndex + zBinRange;
          const int firstRowClusterIndex = indexTable[firstBinIndex];
          const int maxRowClusterIndex = indexTable[maxBinIndex];
          for (int iNextCluster{firstRowClusterIndex}; iNextCluster < maxRowClusterIndex; ++iNextCluster) {
            if (iNextCluster >= nLayerClusters) {
              break;
            }
            const Cluster& nextCluster = layerClusters[iNextCluster];
            if (usedClusters[iLayer][nextCluster.clusterId]) {
              continue;
            }

            const TrackingFrameInfo& trackingHit = trackingFrameInfo[iLayer][nextCluster.clusterId];
            auto updated = hypo;
            if (!updated.param.rotate(trackingHit.alphaTrackingFrame) ||
                !propagator->propagateToX(updated.param, trackingHit.xTrackingFrame, bz,
                                          o2::base::PropagatorF::MAX_SIN_PHI,
                                          o2::base::PropagatorF::MAX_STEP,
                                          matCorrType)) {
              continue;
            }

            const auto predChi2 = updated.param.getPredictedChi2Quiet(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame);
            if (predChi2 < 0.f || predChi2 > maxChi2ClusterAttachment) {
              continue;
            }
            if (!updated.param.o2::track::TrackParCov::update(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)) {
              continue;
            }
            updated.chi2 += predChi2;
            updated.clusters[iLayer] = nextCluster.clusterId;
            ++updated.nClusters;
            updated.edgeLayer = iLayer;
            const auto rofTS = rofOverlaps.getLayer(iLayer).getROFTimeBounds(rof, true);
            const auto& ts = updated.time;
            const float lower = o2::gpu::CAMath::Max(ts.getTimeStamp() - ts.getTimeStampError(), static_cast<float>(rofTS.lower()));
            const float upper = o2::gpu::CAMath::Min(ts.getTimeStamp() + ts.getTimeStampError(), static_cast<float>(rofTS.upper()));
            updated.time.setTimeStamp(0.5f * (lower + upper));
            updated.time.setTimeStampError(0.5f * (upper - lower));
            addTrackExtensionHypothesisToBeam(updated, nextHypotheses, nNext, beamWidth);
          }
        }
      }
      addTrackExtensionHypothesisToBeam(hypo, nextHypotheses, nNext, beamWidth);
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
    if (hypo.nClusters == track.getNClusters()) {
      continue;
    }
    const float maxChi2 = maxChi2NDF * static_cast<float>(hypo.nClusters * 2 - 5);
    if (hypo.chi2 >= maxChi2) {
      continue;
    }
    if (!bestHypo || isBetterTrackExtensionHypothesis(hypo, *bestHypo)) {
      bestHypo = &hypo;
    }
  }
  if (!bestHypo) {
    return false;
  }

  updatedTrack = track;
  if (outward) {
    updatedTrack.getParamOut() = bestHypo->param;
  } else {
    updatedTrack.getParamIn() = bestHypo->param;
  }
  updatedTrack.getTimeStamp() = bestHypo->time;
  updatedTrack.setChi2(bestHypo->chi2);
  for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
    if (updatedTrack.getClusterIndex(iLayer) == constants::UnusedIndex && bestHypo->clusters[iLayer] != constants::UnusedIndex) {
      updatedTrack.setExternalClusterIndex(iLayer, bestHypo->clusters[iLayer], true);
    }
  }
  return true;
}

} // namespace o2::its

#endif // TRACKINGITSU_INCLUDE_TRACKFOLLOWER_H_
