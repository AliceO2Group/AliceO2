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

#ifndef O2_ITS3_ALIGN_TRACKFIT
#define O2_ITS3_ALIGN_TRACKFIT

#include <Eigen/Dense>

#include "ITSBase/GeometryTGeo.h"
#include "DetectorsBase/Propagator.h"
#include "ReconstructionDataFormats/Track.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"

namespace o2::its3::align
{
using Mat51 = Eigen::Matrix<double, 5, 1>;
using Mat55 = Eigen::Matrix<double, 5, 5>;
using TrackD = o2::track::TrackParCovD;

template <typename T>
struct TrackingCluster : public o2::BaseCluster<T> {
  using o2::BaseCluster<T>::BaseCluster;
  T alpha{};
};

template <typename T, typename F>
track::TrackParametrizationWithError<T> convertTrack(const track::TrackParametrizationWithError<F>& trk)
{
  if constexpr (std::is_same_v<T, F>) {
    return trk;
  }
  track::TrackParametrizationWithError<T> dst;
  dst.setX(trk.getX());
  dst.setAlpha(trk.getAlpha());
  for (int iPar{0}; iPar < track::kNParams; ++iPar) {
    dst.setParam(trk.getParam(iPar), iPar);
  }
  dst.setAbsCharge(trk.getAbsCharge());
  dst.setPID(trk.getPID());
  dst.setUserField(trk.getUserField());
  for (int iCov{0}; iCov < track::kCovMatSize; ++iCov) {
    dst.setCov(trk.getCov()[iCov], iCov);
  }
  return dst;
}

// Both tracks must be at the same (alpha, x).
// Returns the interpolated track.
template <typename T>
o2::track::TrackParametrizationWithError<T> interpolateTrackParCov(
  const o2::track::TrackParametrizationWithError<T>& tA,
  const o2::track::TrackParametrizationWithError<T>& tB)
{
  auto res = tA;
  if (!tA.isValid() || !tB.isValid() || tA.getAlpha() != tB.getAlpha() || tA.getX() != tB.getX()) {
    res.invalidate();
    return res;
  }
  auto unpack = [](const std::array<T, track::kCovMatSize>& c) {
    Mat55 m;
    for (int i = 0, k = 0; i < 5; ++i) {
      for (int j = 0; j <= i; ++j, ++k) {
        m(i, j) = m(j, i) = (double)c[k];
      }
    }
    return m;
  };
  Mat55 cA = unpack(tA.getCov());
  Mat55 cB = unpack(tB.getCov());
  Mat55 wA = cA.inverse();
  Mat55 wB = cB.inverse();
  Mat55 wTot = wA + wB;
  Mat55 cTot = wTot.inverse();
  Mat51 pA, pB;
  for (int i = 0; i < 5; ++i) {
    pA(i) = tA.getParam(i);
    pB(i) = tB.getParam(i);
  }
  Mat51 pTot = cTot * (wA * pA + wB * pB);
  // build result - same alpha/x as inputs
  for (int i = 0; i < 5; ++i) {
    res.setParam(pTot(i), i);
  }
  for (int i = 0, k = 0; i < 5; ++i) {
    for (int j = 0; j <= i; ++j, ++k) {
      res.setCov(static_cast<T>(cTot(i, j)), k);
    }
  }
  return res;
}

// Performs an outward (0->7) and inward (7->0) Kalman refit storing the
// extrapolation *before* the cluster update at each layer.
// cluster array clArr[0] = PV (optional), clArr[1..7] = layers 0-6.
// chi2 is accumulated only for the outward direction
template <typename T>
bool doBidirRefit(
  const o2::its::TrackITS& iTrack,
  std::array<const TrackingCluster<T>*, 8>& clArr,
  std::array<o2::track::TrackParametrizationWithError<T>, 8>& extrapOut,
  std::array<o2::track::TrackParametrizationWithError<T>, 8>& extrapInw,
  T& chi2,
  bool useStableRef,
  typename o2::base::PropagatorImpl<T>::MatCorrType corrType)
{
  const auto prop = o2::base::PropagatorImpl<T>::Instance();
  const auto geom = o2::its::GeometryTGeo::Instance();
  const auto bz = prop->getNominalBz();

  auto rotateTrack = [bz](o2::track::TrackParametrizationWithError<T>& tr, T alpha, o2::track::TrackParametrization<T>* refLin) {
    return refLin ? tr.rotate(alpha, *refLin, bz) : tr.rotate(alpha);
  };
  auto accountCluster = [&](int i, std::array<o2::track::TrackParametrizationWithError<T>, 8>& extrapDest, o2::track::TrackParametrizationWithError<T>& tr, o2::track::TrackParametrization<T>* refLin) -> int {
    if (clArr[i]) {
      bool outward = tr.getX() < clArr[i]->getX();
      if (!rotateTrack(tr, clArr[i]->alpha, refLin) || !prop->propagateTo(tr, refLin, clArr[i]->getX(), false, base::PropagatorImpl<T>::MAX_SIN_PHI, base::PropagatorImpl<T>::MAX_STEP, corrType)) {
        return 0;
      }
      if (outward) {
        chi2 += tr.getPredictedChi2Quiet(*clArr[i]);
      }
      extrapDest[i] = tr; // before update
      if (!tr.update(*clArr[i])) {
        return 0;
      }
    } else {
      extrapDest[i].invalidate();
      return -1;
    }
    return 1;
  };
  auto trFitInw = convertTrack<T>(iTrack.getParamOut());
  auto trFitOut = convertTrack<T>(iTrack.getParamIn());
  if (clArr[0]) { // propagate outward seed to PV cluster's tracking frame
    if (!trFitOut.rotate(clArr[0]->alpha) || !prop->propagateToX(trFitOut, clArr[0]->getX(), bz, base::PropagatorImpl<T>::MAX_SIN_PHI, base::PropagatorImpl<T>::MAX_STEP, corrType)) {
      return false;
    }
  }
  // linearization references
  o2::track::TrackParametrization<T> refLinInw0, refLinOut0, *refLinOut = nullptr, *refLinInw = nullptr;
  if (useStableRef) {
    refLinOut = &(refLinOut0 = trFitOut);
    refLinInw = &(refLinInw0 = trFitInw);
  }

  auto resetTrackCov = [bz](auto& trk) {
    trk.resetCovariance();
    float qptB5Scale = std::abs(bz) > 0.1f ? std::abs(bz) / 5.006680f : 1.f;
    float q2pt2 = trk.getQ2Pt() * trk.getQ2Pt(), q2pt2Wgh = q2pt2 * qptB5Scale * qptB5Scale;
    float err2 = (100.f + q2pt2Wgh) / (1.f + q2pt2Wgh) * q2pt2; // -> 100 for high pTs, -> 1 for low pTs.
    trk.setCov(err2, 14);                                       // 100% error
  };
  resetTrackCov(trFitOut);
  resetTrackCov(trFitInw);

  for (int i = 0; i <= 7; i++) {
    if (!accountCluster(i, extrapOut, trFitOut, refLinOut) || !accountCluster(7 - i, extrapInw, trFitInw, refLinInw)) {
      return false;
    }
  }
  return true;
}

} // namespace o2::its3::align

#endif
