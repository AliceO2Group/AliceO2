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
//
// \author matteo.concas@cern.ch

#include "ITStracking/Smoother.h"

namespace o2
{
namespace its
{

constexpr std::array<double, 3> getInverseSymm2D(const std::array<double, 3>& mat)
{
  const double det = mat[0] * mat[2] - mat[1] * mat[1];
  return std::array<double, 3>{mat[2] / det, -mat[1] / det, mat[0] / det};
}

// Smoother
template <unsigned int D>
Smoother<D>::Smoother(TrackITSExt& track, size_t smoothingLayer, const ROframe& event, float bZ, o2::base::PropagatorF::MatCorrType corr) : mLayerToSmooth{smoothingLayer},
                                                                                                                                            mBz(bZ),
                                                                                                                                            mCorr(corr)
{

  auto propInstance = o2::base::Propagator::Instance();
  const TrackingFrameInfo& originalTf = event.getTrackingFrameInfoOnLayer(mLayerToSmooth).at(track.getClusterIndex(mLayerToSmooth));

  mOutwardsTrack = track;               // This track will be propagated outwards inside the smoother! (as last step of fitting did inward propagation)
  mInwardsTrack = {track.getParamOut(), // This track will be propagated inwards inside the smoother!
                   static_cast<short>(mOutwardsTrack.getNumberOfClusters()), -999, static_cast<std::uint32_t>(event.getROFrameId()),
                   mOutwardsTrack.getParamOut(), mOutwardsTrack.getClusterIndexes()};

  mOutwardsTrack.resetCovariance();
  mOutwardsTrack.setChi2(0);
  mInwardsTrack.resetCovariance();
  mInwardsTrack.setChi2(0);

  bool statusOutw{false};
  bool statusInw{false};

  //////////////////////
  // Outward propagation
  for (size_t iLayer{0}; iLayer < mLayerToSmooth; ++iLayer) {
    if (mOutwardsTrack.getClusterIndex(iLayer) == constants::its::UnusedIndex) { // Shorter tracks
      continue;
    }
    const TrackingFrameInfo& tF = event.getTrackingFrameInfoOnLayer(iLayer).at(mOutwardsTrack.getClusterIndex(iLayer));
    statusOutw = mOutwardsTrack.rotate(tF.alphaTrackingFrame);
    statusOutw &= propInstance->propagateToX(mOutwardsTrack,
                                             tF.xTrackingFrame,
                                             mBz,
                                             o2::base::PropagatorImpl<float>::MAX_SIN_PHI,
                                             o2::base::PropagatorImpl<float>::MAX_STEP,
                                             mCorr);
    mOutwardsTrack.setChi2(mOutwardsTrack.getChi2() + mOutwardsTrack.getPredictedChi2(tF.positionTrackingFrame, tF.covarianceTrackingFrame));
    statusOutw &= mOutwardsTrack.o2::track::TrackParCov::update(tF.positionTrackingFrame, tF.covarianceTrackingFrame);
    // LOG(info) << "Outwards loop on inwards track, layer: " << iLayer << " x: " << mOutwardsTrack.getX();
  }

  // Prediction on the previously outwards-propagated track is done on a copy, as the process seems to be not reversible
  auto outwardsClone = mOutwardsTrack;
  statusOutw = outwardsClone.rotate(originalTf.alphaTrackingFrame);
  statusOutw &= propInstance->propagateToX(outwardsClone,
                                           originalTf.xTrackingFrame,
                                           mBz,
                                           o2::base::PropagatorImpl<float>::MAX_SIN_PHI,
                                           o2::base::PropagatorImpl<float>::MAX_STEP,
                                           mCorr);
  /////////////////////
  // Inward propagation
  for (size_t iLayer{D - 1}; iLayer > mLayerToSmooth; --iLayer) {
    if (mInwardsTrack.getClusterIndex(iLayer) == constants::its::UnusedIndex) { // Shorter tracks
      continue;
    }
    const TrackingFrameInfo& tF = event.getTrackingFrameInfoOnLayer(iLayer).at(mInwardsTrack.getClusterIndex(iLayer));
    statusInw = mInwardsTrack.rotate(tF.alphaTrackingFrame);
    statusInw &= propInstance->propagateToX(mInwardsTrack,
                                            tF.xTrackingFrame,
                                            mBz,
                                            o2::base::PropagatorImpl<float>::MAX_SIN_PHI,
                                            o2::base::PropagatorImpl<float>::MAX_STEP,
                                            mCorr);
    mInwardsTrack.setChi2(mInwardsTrack.getChi2() + mInwardsTrack.getPredictedChi2(tF.positionTrackingFrame, tF.covarianceTrackingFrame));
    statusInw &= mInwardsTrack.o2::track::TrackParCov::update(tF.positionTrackingFrame, tF.covarianceTrackingFrame);
    // LOG(info) << "Inwards loop on outwards track, layer: " << iLayer << " x: " << mInwardsTrack.getX();
  }

  // Prediction on the previously inwards-propagated track is done on a copy, as the process seems to be not revesible
  auto inwardsClone = mInwardsTrack;
  statusInw = inwardsClone.rotate(originalTf.alphaTrackingFrame);
  statusInw &= propInstance->propagateToX(inwardsClone,
                                          originalTf.xTrackingFrame,
                                          mBz,
                                          o2::base::PropagatorImpl<float>::MAX_SIN_PHI,
                                          o2::base::PropagatorImpl<float>::MAX_STEP,
                                          mCorr);
  // Compute weighted local chi2
  mInitStatus = statusInw && statusOutw;
  if (mInitStatus) {
    mBestChi2 = computeSmoothedPredictedChi2(inwardsClone, outwardsClone, originalTf.positionTrackingFrame, originalTf.covarianceTrackingFrame);
    mLastChi2 = mBestChi2;
    LOG(info) << "Smoothed chi2 on original cluster: " << mBestChi2;
  }
}

template <unsigned int D>
Smoother<D>::~Smoother() = default;

template <unsigned int D>
float Smoother<D>::computeSmoothedPredictedChi2(const o2::track::TrackParCov& firstTrack,  // outwards track: from innermost cluster to outermost
                                                const o2::track::TrackParCov& secondTrack, // inwards track: from outermost cluster to innermost
                                                const std::array<float, 2>& cls,
                                                const std::array<float, 3>& clCov)
{
  // Tracks need to be already propagated, compute only chi2
  // Symmetric covariances assumed

  if (firstTrack.getX() != secondTrack.getX()) {
    LOG(fatal) << "Tracks need to be propagated to the same point! secondTrack.X=" << secondTrack.getX() << " firstTrack.X=" << firstTrack.getX();
  }

  std::array<double, 2> pp1 = {static_cast<double>(firstTrack.getY()), static_cast<double>(firstTrack.getZ())};   // P1: predicted Y,Z points
  std::array<double, 2> pp2 = {static_cast<double>(secondTrack.getY()), static_cast<double>(secondTrack.getZ())}; // P2: predicted Y,Z points

  std::array<double, 3> c1 = {static_cast<double>(firstTrack.getSigmaY2()),
                              static_cast<double>(firstTrack.getSigmaZY()),
                              static_cast<double>(firstTrack.getSigmaZ2())}; // Cov. track 1

  std::array<double, 3> c2 = {static_cast<double>(secondTrack.getSigmaY2()),
                              static_cast<double>(secondTrack.getSigmaZY()),
                              static_cast<double>(secondTrack.getSigmaZ2())}; // Cov. track 2

  std::array<double, 3> w1 = getInverseSymm2D(c1); // weight matrices
  std::array<double, 3> w2 = getInverseSymm2D(c2);

  std::array<double, 3> w1w2 = {w1[0] + w2[0], w1[1] + w2[1], w1[2] + w2[2]}; // (W1 + W2)
  std::array<double, 3> C = getInverseSymm2D(w1w2);                           // C = (W1+W2)^-1

  std::array<double, 2> w1pp1 = {w1[0] * pp1[0] + w1[1] * pp1[1], w1[1] * pp1[0] + w1[2] * pp1[1]}; // W1 * P1
  std::array<double, 2> w2pp2 = {w2[0] * pp2[0] + w2[1] * pp2[1], w2[1] * pp2[0] + w2[2] * pp2[1]}; // W2 * P2

  double Y = C[0] * (w1pp1[0] + w2pp2[0]) + C[1] * (w1pp1[1] + w2pp2[1]); // Pp: weighted normalized combination of the predictions:
  double Z = C[1] * (w1pp1[0] + w2pp2[0]) + C[2] * (w1pp1[1] + w2pp2[1]); // Pp = [(W1 * P1) + (W2 * P2)] / (W1 + W2)

  std::array<double, 2> delta = {Y - cls[0], Z - cls[1]};                                                                                         // Δ = Pp - X, X: space point of cluster (Y,Z)
  std::array<double, 3> CCp = {C[0] + static_cast<double>(clCov[0]), C[1] + static_cast<double>(clCov[1]), C[2] + static_cast<double>(clCov[2])}; // Transformation of cluster covmat: CCp = C + Cov
  std::array<double, 3> Wp = getInverseSymm2D(CCp);                                                                                               // Get weight matrix: Wp = CCp^-1

  float chi2 = static_cast<float>(delta[0] * (Wp[0] * delta[0] + Wp[1] * delta[1]) + delta[1] * (Wp[1] * delta[0] + Wp[2] * delta[1])); // chi2 = tΔ * (Wp * Δ)

  // #ifdef CA_DEBUG
  LOG(info) << "Cluster_y: " << cls[0] << " Cluster_z: " << cls[1];
  LOG(info) << "\t\t- Covariance cluster: Y2: " << clCov[0] << " YZ: " << clCov[1] << " Z2: " << clCov[2];
  LOG(info) << "\t\t- Propagated t1_y: " << pp1[0] << " t1_z: " << pp1[1];
  LOG(info) << "\t\t- Propagated t2_y: " << pp2[0] << " t2_z: " << pp2[1];
  LOG(info) << "\t\t- Covariance t1: sY2: " << c1[0] << " sYZ: " << c1[1] << " sZ2: " << c1[2];
  LOG(info) << "\t\t- Covariance t2: sY2: " << c2[0] << " sYZ: " << c2[1] << " sZ2: " << c2[2];
  LOG(info) << "Smoother prediction Y: " << Y << " Z: " << Z;
  LOG(info) << "\t\t- Delta_y: " << delta[0] << " Delta_z: " << delta[1];
  LOG(info) << "\t\t- Covariance Pr: Y2: " << C[0] << " YZ: " << C[1] << " Z2: " << C[2];
  LOG(info) << "\t\t- predicted chi2 t1: " << firstTrack.getPredictedChi2(cls, clCov);
  LOG(info) << "\t\t- predicted chi2 t2: " << secondTrack.getPredictedChi2(cls, clCov);
  // #endif
  return chi2;
}

template <unsigned int D>
bool Smoother<D>::testCluster(const int clusterId, const ROframe& event)
{
  if (!mInitStatus) {
    return false;
  }
  auto propInstance = o2::base::Propagator::Instance();
  const TrackingFrameInfo& testTf = event.getTrackingFrameInfoOnLayer(mLayerToSmooth).at(clusterId);

  bool statusOutw{false};
  bool statusInw{false};

  // Prediction on the previously outwards-propagated track is done on a copy, as the process seems to be not reversible
  auto outwardsClone = mOutwardsTrack;
  statusOutw = outwardsClone.rotate(testTf.alphaTrackingFrame);
  statusOutw &= propInstance->propagateToX(outwardsClone,
                                           testTf.xTrackingFrame,
                                           mBz,
                                           o2::base::PropagatorImpl<float>::MAX_SIN_PHI,
                                           o2::base::PropagatorImpl<float>::MAX_STEP,
                                           mCorr);

  // Prediction on the previously inwards-propagated track is done on a copy, as the process seems to be not reversible
  auto inwardsClone = mInwardsTrack;
  statusInw = inwardsClone.rotate(testTf.alphaTrackingFrame);
  statusInw &= propInstance->propagateToX(inwardsClone,
                                          testTf.xTrackingFrame,
                                          mBz,
                                          o2::base::PropagatorImpl<float>::MAX_SIN_PHI,
                                          o2::base::PropagatorImpl<float>::MAX_STEP,
                                          mCorr);
  if (!(statusOutw && statusInw)) {
    LOG(warning) << "Failed propagation in smoother!";
    return false;
  }

  // Compute weighted local chi2
  mLastChi2 = computeSmoothedPredictedChi2(inwardsClone, outwardsClone, testTf.positionTrackingFrame, testTf.covarianceTrackingFrame);
  LOG(info) << "Smoothed chi2 on tested cluster: " << mLastChi2;

  return true;
}

template class Smoother<7>;

} // namespace its
} // namespace o2
