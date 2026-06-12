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
///
/// \file TrackHelpers.h
/// \brief Shared host/device helpers for ITS tracker trait implementations
///

#ifndef O2_ITS_TRACKING_TRACKHELPERS_H_
#define O2_ITS_TRACKING_TRACKHELPERS_H_

#include "CommonConstants/MathConstants.h"
#include "DataFormatsITS/TrackITS.h"
#include "ITStracking/Cell.h"
#include "ITStracking/Cluster.h"
#include "ITStracking/Constants.h"
#include "ITStracking/LayerMask.h"
#include "ITStracking/MathUtils.h"
#include "ITStracking/TrackITSInternal.h"
#include "DetectorsBase/Propagator.h"
#include "ReconstructionDataFormats/Track.h"

namespace o2::its::track
{

GPUhdi() bool isBetter(const int nClustersA, const float chi2A, const int nClustersB, const float chi2B)
{
  return (nClustersA > nClustersB) || (nClustersA == nClustersB && chi2A < chi2B);
}

GPUhdi() bool isBetter(const auto& a, const auto& b)
{
  return isBetter(a.getNumberOfClusters(), a.getChi2(), b.getNumberOfClusters(), b.getChi2());
}

template <int NLayers>
struct TrackSeedSelector {
  float maxQ2Pt;
  float maxChi2;
  int maxHoles;
  int minTrackLength;
  LayerMask holeLayerMask;
  LayerMask nonSeedingLayerMask;

  GPUhd() TrackSeedSelector(float maxQ2Pt, float maxChi2NDF, int startLevel, int maxHoles, int minTrackLength, LayerMask holeLayerMask, LayerMask nonSeedingLayerMask)
    : maxQ2Pt{maxQ2Pt}, maxChi2{maxChi2NDF * ((startLevel + 2) * 2 - 5)}, maxHoles{maxHoles}, minTrackLength{minTrackLength}, holeLayerMask{holeLayerMask}, nonSeedingLayerMask{nonSeedingLayerMask}
  {
  }

  static GPUhdi() int getEffectiveTrackLength(LayerMask hitLayerMask, LayerMask excludedLayerMask)
  {
    if (hitLayerMask.empty()) {
      return 0;
    }
    return hitLayerMask.length() - (LayerMask::span(hitLayerMask.first(), hitLayerMask.last()) & excludedLayerMask).count();
  }

  static GPUhdi() LayerMask getEffectiveHoleMask(LayerMask hitLayerMask, LayerMask excludedLayerMask)
  {
    return hitLayerMask.holeMask() & ~excludedLayerMask;
  }

  GPUhd() bool operator()(const TrackSeed<NLayers>& seed) const
  {
    const auto hitLayerMask = seed.getHitLayerMask();
    return !(seed.getQ2Pt() > maxQ2Pt || seed.getChi2() > maxChi2) &&
           getEffectiveTrackLength(hitLayerMask, nonSeedingLayerMask) >= minTrackLength &&
           getEffectiveHoleMask(hitLayerMask, nonSeedingLayerMask).isAllowedHoleMask(maxHoles, holeLayerMask);
  }
};

// Find the populated interior layer closest to the radial midpoint.
// If no layer can be found, return constants::UnusedIndex.
// Should minimize the sagitta bias.
template <int NLayers>
GPUdi() int selectReseedMidLayer(int minLayer, int maxLayer, const float* layerRadii, const TrackSeed<NLayers>& seed)
{
  int midLayer = constants::UnusedIndex;
  float distanceToMidR = layerRadii[NLayers - 1]; // midpoint cannot be last layer
  const float midR = 0.5f * (layerRadii[maxLayer] + layerRadii[minLayer]);
  for (int iLayer = minLayer + 1; iLayer < maxLayer; ++iLayer) {
    if (seed.getCluster(iLayer) != constants::UnusedIndex) {
      const float distance = o2::gpu::CAMath::Abs(midR - layerRadii[iLayer]);
      if (distance < distanceToMidR) { // keep the smaller-radius layer on ties
        midLayer = iLayer;
        distanceToMidR = distance;
      }
    }
  }
  return midLayer;
}

GPUdi() void resetTrackCovariance(o2::track::TrackParCov& track)
{
  track.resetCovariance();
  track.setCov(track.getQ2Pt() * track.getQ2Pt() * track.getCov()[o2::track::CovLabels::kSigQ2Pt2], o2::track::CovLabels::kSigQ2Pt2);
}

GPUdi() o2::track::TrackParCov buildTrackSeed(const Cluster& cluster1,
                                              const Cluster& cluster2,
                                              const TrackingFrameInfo& tf3,
                                              const float bz,
                                              const bool reverse = false)
{
  float ca = constants::UnsetValue, sa = constants::UnsetValue, snp = constants::UnsetValue, q2pt = constants::UnsetValue, q2pt2 = constants::UnsetValue;
  o2::gpu::CAMath::SinCos(tf3.alphaTrackingFrame, sa, ca);
  const float sign = reverse ? -1.f : 1.f;
  const float x1 = (cluster1.xCoordinate * ca) + (cluster1.yCoordinate * sa);
  const float y1 = (-cluster1.xCoordinate * sa) + (cluster1.yCoordinate * ca);
  const float x2 = (cluster2.xCoordinate * ca) + (cluster2.yCoordinate * sa);
  const float y2 = (-cluster2.xCoordinate * sa) + (cluster2.yCoordinate * ca);
  const float x3 = tf3.xTrackingFrame;
  const float y3 = tf3.positionTrackingFrame[0];
  if (o2::gpu::CAMath::Abs(bz) < 0.01f) { // zero field
    const float dx = x3 - x1;
    const float dy = y3 - y1;
    snp = sign * dy / o2::gpu::CAMath::Hypot(dx, dy);
    q2pt = 1.f / o2::track::kMostProbablePt;
    q2pt2 = 1.f;
  } else {
    const float crv = math_utils::computeCurvature(x3, y3, x2, y2, x1, y1);
    snp = sign * crv * (x3 - math_utils::computeCurvatureCentreX(x3, y3, x2, y2, x1, y1));
    q2pt = sign * crv / (bz * o2::constants::math::B2C);
    q2pt2 = crv * crv;
  }
  const float tgl = -0.5f * sign * (math_utils::computeTanDipAngle(x1, y1, x2, y2, cluster1.zCoordinate, cluster2.zCoordinate) + math_utils::computeTanDipAngle(x2, y2, x3, y3, cluster2.zCoordinate, tf3.positionTrackingFrame[1]));
  const float sg2q2pt = o2::track::kC1Pt2max * o2::gpu::CAMath::Clamp(q2pt2, 0.0005f, 1.0f);
  return {x3, tf3.alphaTrackingFrame, {y3, tf3.positionTrackingFrame[1], snp, tgl, q2pt}, {tf3.covarianceTrackingFrame[0], tf3.covarianceTrackingFrame[1], tf3.covarianceTrackingFrame[2], 0.f, 0.f, o2::track::kCSnp2max, 0.f, 0.f, 0.f, o2::track::kCTgl2max, 0.f, 0.f, 0.f, 0.f, sg2q2pt}};
}

template <int NLayers>
GPUdi() TrackITSInternal<NLayers> seedTrackForRefit(const TrackSeed<NLayers>& seed,
                                                    const TrackingFrameInfo* const* foundTrackingFrameInfo,
                                                    const Cluster* const* unsortedClusters,
                                                    const float* layerRadii,
                                                    const float bz,
                                                    const int reseedIfShorter)
{
  TrackITSInternal<NLayers> temporaryTrack;
  temporaryTrack.paramIn = static_cast<const o2::track::TrackParCov&>(seed);
  int lrMin = NLayers;
  int lrMax = 0;
  for (int iL{0}; iL < NLayers; ++iL) {
    const int idx = seed.getCluster(iL);
    temporaryTrack.setClusterIndex(iL, idx);
    if (idx != constants::UnusedIndex) {
      lrMin = o2::gpu::CAMath::Min(lrMin, iL);
      lrMax = o2::gpu::CAMath::Max(lrMax, iL);
    }
  }

  const int ncl = temporaryTrack.getNClusters();
  if (ncl < reseedIfShorter && ncl > 2) {
    const int lrMid = selectReseedMidLayer<NLayers>(lrMin, lrMax, layerRadii, seed);
    if (lrMid != constants::UnusedIndex) {
      const auto& cluster0TF = foundTrackingFrameInfo[lrMin][seed.getCluster(lrMin)];
      const auto& cluster1GL = unsortedClusters[lrMid][seed.getCluster(lrMid)];
      const auto& cluster2GL = unsortedClusters[lrMax][seed.getCluster(lrMax)];
      temporaryTrack.paramIn = buildTrackSeed(cluster2GL, cluster1GL, cluster0TF, bz, true);
    }
  }

  resetTrackCovariance(temporaryTrack.paramIn);
  return temporaryTrack;
}

// Inputs shared by fit/refit calls within a tracking pass.
template <int NLayers>
struct TrackFitContext {
  const TrackingFrameInfo* const* tfInfos{nullptr};
  const float* layerxX0{nullptr};
  int nLayers{0};
  float bz{0.f};
  float maxChi2ClusterAttachment{0.f};
  float maxChi2NDF{0.f};
  const o2::base::Propagator* propagator{nullptr};
  o2::base::PropagatorF::MatCorrType matCorrType{o2::base::PropagatorF::MatCorrType::USEMatCorrNONE};
  bool shiftRefToCluster{false};
  bool repeatRefitOut{false};
};

template <int NLayers>
GPUdi() bool fitTrack(TrackITSInternal<NLayers>& trk,
                      o2::track::TrackParCov& param,
                      int start,
                      int end,
                      int step,
                      float maxQoverPt,
                      int nCl,
                      const TrackFitContext<NLayers>& ctx,
                      o2::track::TrackPar* linRef = nullptr)
{
  for (int iLayer{start}; iLayer != end; iLayer += step) {
    if (trk.getClusterIndex(iLayer) == constants::UnusedIndex) {
      continue;
    }

    const TrackingFrameInfo& trackingHit = ctx.tfInfos[iLayer][trk.getClusterIndex(iLayer)];
    if (linRef) {
      if (!param.o2::track::TrackParCovF::rotate(trackingHit.alphaTrackingFrame, *linRef, ctx.bz)) {
        return false;
      }
      if (!ctx.propagator->propagateToX(param, *linRef, trackingHit.xTrackingFrame, ctx.bz,
                                        o2::base::PropagatorImpl<float>::MAX_SIN_PHI,
                                        o2::base::PropagatorImpl<float>::MAX_STEP,
                                        ctx.matCorrType)) {
        return false;
      }
      if (ctx.matCorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE) {
        if (!param.correctForMaterial(*linRef, ctx.layerxX0[iLayer], ctx.layerxX0[iLayer] * constants::Radl * constants::Rho, true)) {
          continue;
        }
      }
    } else {
      if (!param.o2::track::TrackParCovF::rotate(trackingHit.alphaTrackingFrame)) {
        return false;
      }
      if (!ctx.propagator->propagateToX(param, trackingHit.xTrackingFrame, ctx.bz,
                                        o2::base::PropagatorImpl<float>::MAX_SIN_PHI,
                                        o2::base::PropagatorImpl<float>::MAX_STEP,
                                        ctx.matCorrType)) {
        return false;
      }
      if (ctx.matCorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE) {
        if (!param.correctForMaterial(ctx.layerxX0[iLayer], ctx.layerxX0[iLayer] * constants::Radl * constants::Rho, true)) {
          continue;
        }
      }
    }

    const auto predChi2{param.getPredictedChi2Quiet(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)};
    if ((nCl >= 3 && predChi2 > ctx.maxChi2ClusterAttachment) || predChi2 < 0.f) {
      return false;
    }
    trk.setChi2(trk.getChi2() + predChi2);
    if (!param.o2::track::TrackParCov::update(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)) {
      return false;
    }
    if (linRef && ctx.shiftRefToCluster) {
      linRef->setY(trackingHit.positionTrackingFrame[0]);
      linRef->setZ(trackingHit.positionTrackingFrame[1]);
    }
    nCl++;
  }

  return o2::gpu::CAMath::Abs(param.getQ2Pt()) < maxQoverPt && trk.getChi2() < ctx.maxChi2NDF * (float)((nCl * 2) - 5);
}

template <int NLayers>
GPUdi() bool refitTrack(TrackITSInternal<NLayers>& track,
                        const TrackFitContext<NLayers>& ctx,
                        const float minPt = -1.f)
{
  o2::track::TrackPar linRef{track.paramIn};
  resetTrackCovariance(track.paramIn);
  track.setChi2(0);
  bool fitSuccess = fitTrack(track, track.paramIn, 0, ctx.nLayers, 1,
                             o2::constants::math::VeryBig, 0, ctx, &linRef);
  if (!fitSuccess) {
    return false;
  }

  track.paramOut = track.paramIn;
  linRef = track.paramOut;
  resetTrackCovariance(track.paramIn);
  track.setChi2(0);
  fitSuccess = fitTrack(track, track.paramIn, ctx.nLayers - 1, -1, -1,
                        50.f, 0, ctx, &linRef);
  if (!fitSuccess) {
    return false;
  }
  if (minPt > 0.f && track.getPt() < minPt) {
    return false;
  }
  if (ctx.repeatRefitOut) { // repeat outward refit seeding and linearizing with the stable inward fit result
    o2::track::TrackParCov saveInw{track.paramIn};
    linRef = saveInw; // use refitted track as lin.reference
    float saveChi2 = track.getChi2();
    track.paramOut = saveInw;
    track::resetTrackCovariance(track.paramOut);
    track.setChi2(0);
    fitSuccess = fitTrack(track, track.paramOut, 0, ctx.nLayers, 1,
                          o2::constants::math::VeryBig, 0, ctx, &linRef);
    if (!fitSuccess) {
      return false;
    }
    track.paramIn = saveInw;
    track.setChi2(saveChi2);
  }
  return true;
}

template <int NLayers>
GPUdi() bool refitTrackSeed(const TrackSeed<NLayers>& trackSeed,
                            TrackITSInternal<NLayers>& temporaryTrack,
                            const TrackFitContext<NLayers>& ctx,
                            const Cluster* const* clusters,
                            const float* layerRadii,
                            const float* minPt,
                            const int reseedIfShorter)
{
  temporaryTrack = seedTrackForRefit(trackSeed,
                                     ctx.tfInfos,
                                     clusters,
                                     layerRadii,
                                     ctx.bz,
                                     reseedIfShorter);
  return refitTrack(temporaryTrack, ctx, minPt[NLayers - temporaryTrack.getNClusters()]);
}

} // namespace o2::its::track

#endif
