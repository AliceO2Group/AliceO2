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

#include <cmath>

#include "DataFormatsITS/TrackITS.h"
#include "ITStracking/Cell.h"
#include "ITStracking/Cluster.h"
#include "ITStracking/Constants.h"
#include "ITStracking/MathUtils.h"
#include "DetectorsBase/Propagator.h"
#include "ReconstructionDataFormats/Track.h"

namespace o2::its::track
{

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

GPUdi() void resetTrackCovariance(TrackITSExt& track)
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
GPUdi() TrackITSExt seedTrackForRefit(const TrackSeed<NLayers>& seed,
                                      const TrackingFrameInfo* const* foundTrackingFrameInfo,
                                      const Cluster* const* unsortedClusters,
                                      const float* layerRadii,
                                      const float bz,
                                      const int reseedIfShorter)
{
  TrackITSExt temporaryTrack(seed);
  int lrMin = NLayers;
  int lrMax = 0;
  for (int iL{0}; iL < NLayers; ++iL) {
    const int idx = seed.getCluster(iL);
    temporaryTrack.setExternalClusterIndex(iL, idx, idx != constants::UnusedIndex);
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
      temporaryTrack.getParamIn() = buildTrackSeed(cluster2GL, cluster1GL, cluster0TF, bz, true);
    }
  }

  resetTrackCovariance(temporaryTrack);
  return temporaryTrack;
}

GPUdi() bool fitTrack(TrackITSExt& trk,
                      int start,
                      int end,
                      int step,
                      float chi2clcut,
                      float chi2ndfcut,
                      float maxQoverPt,
                      int nCl,
                      const float bz,
                      const TrackingFrameInfo* const* tfInfos,
                      const float* layerxX0,
                      const o2::base::Propagator* propagator,
                      const o2::base::PropagatorF::MatCorrType matCorrType,
                      o2::track::TrackPar* linRef = nullptr,
                      const bool shiftRefToCluster = false)
{
  for (int iLayer{start}; iLayer != end; iLayer += step) {
    if (trk.getClusterIndex(iLayer) == constants::UnusedIndex) {
      continue;
    }

    const TrackingFrameInfo& trackingHit = tfInfos[iLayer][trk.getClusterIndex(iLayer)];
    if (linRef) {
      if (!trk.o2::track::TrackParCovF::rotate(trackingHit.alphaTrackingFrame, *linRef, bz)) {
        return false;
      }
      if (!propagator->propagateToX(trk, *linRef, trackingHit.xTrackingFrame, bz,
                                    o2::base::PropagatorImpl<float>::MAX_SIN_PHI,
                                    o2::base::PropagatorImpl<float>::MAX_STEP,
                                    matCorrType)) {
        return false;
      }
      if (matCorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE) {
        if (!trk.correctForMaterial(*linRef, layerxX0[iLayer], layerxX0[iLayer] * constants::Radl * constants::Rho, true)) {
          continue;
        }
      }
    } else {
      if (!trk.o2::track::TrackParCovF::rotate(trackingHit.alphaTrackingFrame)) {
        return false;
      }
      if (!propagator->propagateToX(trk, trackingHit.xTrackingFrame, bz,
                                    o2::base::PropagatorImpl<float>::MAX_SIN_PHI,
                                    o2::base::PropagatorImpl<float>::MAX_STEP,
                                    matCorrType)) {
        return false;
      }
      if (matCorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE) {
        if (!trk.correctForMaterial(layerxX0[iLayer], layerxX0[iLayer] * constants::Radl * constants::Rho, true)) {
          continue;
        }
      }
    }

    const auto predChi2{trk.getPredictedChi2Quiet(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)};
    if ((nCl >= 3 && predChi2 > chi2clcut) || predChi2 < 0.f) {
      return false;
    }
    trk.setChi2(trk.getChi2() + predChi2);
    if (!trk.o2::track::TrackParCov::update(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)) {
      return false;
    }
    if (linRef && shiftRefToCluster) {
      linRef->setY(trackingHit.positionTrackingFrame[0]);
      linRef->setZ(trackingHit.positionTrackingFrame[1]);
    }
    nCl++;
  }

  return o2::gpu::CAMath::Abs(trk.getQ2Pt()) < maxQoverPt && trk.getChi2() < chi2ndfcut * (float)((nCl * 2) - 5);
}

template <int NLayers>
GPUdi() bool refitTrack(const TrackSeed<NLayers>& trackSeed,
                        TrackITSExt& temporaryTrack,
                        float chi2clcut,
                        float chi2ndfcut,
                        const float bz,
                        const TrackingFrameInfo* const* tfInfos,
                        const Cluster* const* clusters,
                        const float* layerxX0,
                        const float* layerRadii,
                        const float* minPt,
                        const o2::base::Propagator* propagator,
                        const o2::base::PropagatorF::MatCorrType matCorrType,
                        const int reseedIfShorter,
                        const bool shiftRefToCluster,
                        const bool repeatRefitOut)
{
  temporaryTrack = seedTrackForRefit(trackSeed,
                                     tfInfos,
                                     clusters,
                                     layerRadii,
                                     bz,
                                     reseedIfShorter);
  o2::track::TrackPar linRef{temporaryTrack};
  bool fitSuccess = fitTrack(temporaryTrack,
                             0,
                             NLayers,
                             1,
                             chi2clcut,
                             chi2ndfcut,
                             o2::constants::math::VeryBig,
                             0,
                             bz,
                             tfInfos,
                             layerxX0,
                             propagator,
                             matCorrType,
                             &linRef,
                             shiftRefToCluster);
  if (!fitSuccess) {
    return false;
  }
  temporaryTrack.getParamOut() = temporaryTrack.getParamIn();
  linRef = temporaryTrack.getParamOut(); // use refitted track as lin.reference
  resetTrackCovariance(temporaryTrack);
  temporaryTrack.setChi2(0);
  fitSuccess = fitTrack(temporaryTrack,
                        NLayers - 1,
                        -1,
                        -1,
                        chi2clcut,
                        chi2ndfcut,
                        50.f,
                        0,
                        bz,
                        tfInfos,
                        layerxX0,
                        propagator,
                        matCorrType,
                        &linRef,
                        shiftRefToCluster);
  if (!fitSuccess || temporaryTrack.getPt() < minPt[NLayers - temporaryTrack.getNClusters()]) {
    return false;
  }
  if (repeatRefitOut) { // repeat outward refit seeding and linearizing with the stable inward fit result
    o2::track::TrackParCov saveInw{temporaryTrack};
    linRef = saveInw; // use refitted track as lin.reference
    float saveChi2 = temporaryTrack.getChi2();
    track::resetTrackCovariance(temporaryTrack);
    temporaryTrack.setChi2(0);
    fitSuccess = o2::its::track::fitTrack(temporaryTrack,
                                          0,
                                          NLayers,
                                          1,
                                          chi2clcut,
                                          chi2ndfcut,
                                          o2::constants::math::VeryBig,
                                          0,
                                          bz,
                                          tfInfos,
                                          layerxX0,
                                          propagator,
                                          matCorrType,
                                          &linRef,
                                          shiftRefToCluster);
    if (!fitSuccess) {
      return false;
    }
    temporaryTrack.getParamOut() = temporaryTrack.getParamIn();
    temporaryTrack.getParamIn() = saveInw;
    temporaryTrack.setChi2(saveChi2);
  }
  return true;
}

} // namespace o2::its::track

#endif
