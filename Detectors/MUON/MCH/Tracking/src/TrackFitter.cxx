// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackFitter.cxx
/// \brief Implementation of a class to fit a track to a set of clusters
///
/// \author Philippe Pillot, Subatech

#include "MCHTracking/TrackFitter.h"

#include <stdexcept>

#include <TGeoGlobalMagField.h>
#include <TMatrixD.h>

#include "Field/MagneticField.h"
#include "MCHTracking/TrackExtrap.h"

namespace o2
{
namespace mch
{

using namespace std;

constexpr double TrackFitter::SDefaultChamberZ[10];
constexpr double TrackFitter::SChamberThicknessInX0[10];

//_________________________________________________________________________________________________
void TrackFitter::initField(float l3Current, float dipoleCurrent)
{
  /// Create the magnetic field map if not already done

  if (TGeoGlobalMagField::Instance()->GetField()) {
    return;
  }

  auto field =
    o2::field::MagneticField::createFieldMap(l3Current, dipoleCurrent, o2::field::MagneticField::kConvLHC, false, 3500.,
                                             "A-A", "$(O2_ROOT)/share/Common/maps/mfchebKGI_sym.root");
  TGeoGlobalMagField::Instance()->SetField(field);
  TGeoGlobalMagField::Instance()->Lock();

  TrackExtrap::setField();
}

//_________________________________________________________________________________________________
void TrackFitter::fit(Track& track, bool smooth, bool finalize,
                      std::list<TrackParam>::reverse_iterator* itStartingParam)
{
  /// Fit a track to its attached clusters
  /// Smooth the track if requested and the smoother enabled
  /// If finalize = true: copy the smoothed parameters, if any, into the regular ones
  /// Fit the entire track or only the part upstream itStartingParam
  /// Throw an exception in case of failure

  // initialize the starting track parameters and cluster
  auto itParam(track.rbegin());
  if (itStartingParam != nullptr) {
    // use the ones pointed to by itStartingParam
    if (*itStartingParam == track.rend()) {
      throw runtime_error("invalid starting parameters");
    }
    itParam = *itStartingParam;
  } else {
    // or start from the last cluster and compute the track parameters from its position
    // and the one of the first previous cluster found on a different chamber
    auto itPreviousParam(itParam);
    do {
      ++itPreviousParam;
      if (itPreviousParam == track.rend()) {
        throw runtime_error("A track is made of at least 2 clusters in 2 different chambers");
      }
    } while (itPreviousParam->getClusterPtr()->getChamberId() == itParam->getClusterPtr()->getChamberId());
    initTrack(*itPreviousParam->getClusterPtr(), *itParam->getClusterPtr(), *itParam);
  }

  // recusively add the upstream clusters and update the track parameters
  TrackParam* startingParam = &*itParam;
  while (++itParam != track.rend()) {
    addCluster(*startingParam, *itParam->getClusterPtr(), *itParam);
    startingParam = &*itParam;
  }

  // smooth the track if requested and the smoother enabled
  if (smooth && mSmooth) {
    smoothTrack(track, finalize);
  }
}

//_________________________________________________________________________________________________
void TrackFitter::initTrack(const Cluster& cl1, const Cluster& cl2, TrackParam& param)
{
  /// Compute the initial track parameters at the z position of the last cluster (cl2)
  /// The covariance matrix is computed such that the last cluster is the only constraint
  /// (by assigning an infinite dispersion to the other cluster)
  /// These parameters are the seed for the Kalman filter

  // compute the track parameters at the last cluster
  double dZ = cl1.getZ() - cl2.getZ();
  param.setNonBendingCoor(cl2.getX());
  param.setBendingCoor(cl2.getY());
  param.setZ(cl2.getZ());
  param.setNonBendingSlope((cl1.getX() - cl2.getX()) / dZ);
  param.setBendingSlope((cl1.getY() - cl2.getY()) / dZ);
  double bendingImpact = cl2.getY() - cl2.getZ() * param.getBendingSlope();
  double inverseBendingMomentum = 1. / TrackExtrap::getBendingMomentumFromImpactParam(bendingImpact);
  param.setInverseBendingMomentum(inverseBendingMomentum);

  // compute the track parameter covariances at the last cluster (as if the other clusters did not exist)
  TMatrixD lastParamCov(5, 5);
  lastParamCov.Zero();
  double cl1Ey2(0.);
  if (mUseChamberResolution) {
    // Non bending plane
    lastParamCov(0, 0) = mChamberResolutionX2;
    lastParamCov(1, 1) = (1000. * mChamberResolutionX2 + lastParamCov(0, 0)) / dZ / dZ;
    // Bending plane
    lastParamCov(2, 2) = mChamberResolutionY2;
    cl1Ey2 = mChamberResolutionY2;
  } else {
    // Non bending plane
    lastParamCov(0, 0) = cl2.getEx2();
    lastParamCov(1, 1) = (1000. * cl1.getEx2() + lastParamCov(0, 0)) / dZ / dZ;
    // Bending plane
    lastParamCov(2, 2) = cl2.getEy2();
    cl1Ey2 = cl1.getEy2();
  }
  // Non bending plane
  lastParamCov(0, 1) = -lastParamCov(0, 0) / dZ;
  lastParamCov(1, 0) = lastParamCov(0, 1);
  // Bending plane
  lastParamCov(2, 3) = -lastParamCov(2, 2) / dZ;
  lastParamCov(3, 2) = lastParamCov(2, 3);
  lastParamCov(3, 3) = (1000. * cl1Ey2 + lastParamCov(2, 2)) / dZ / dZ;
  // Inverse bending momentum (vertex resolution + bending slope resolution + 10% error on dipole parameters+field)
  if (TrackExtrap::isFieldON()) {
    lastParamCov(4, 4) =
      ((mBendingVertexDispersion2 +
        (cl1.getZ() * cl1.getZ() * lastParamCov(2, 2) + cl2.getZ() * cl2.getZ() * 1000. * cl1Ey2) / dZ / dZ) /
         bendingImpact / bendingImpact +
       0.1 * 0.1) *
      inverseBendingMomentum * inverseBendingMomentum;
    lastParamCov(2, 4) = cl1.getZ() * lastParamCov(2, 2) * inverseBendingMomentum / bendingImpact / dZ;
    lastParamCov(4, 2) = lastParamCov(2, 4);
    lastParamCov(3, 4) = -(cl1.getZ() * lastParamCov(2, 2) + cl2.getZ() * 1000. * cl1Ey2) * inverseBendingMomentum /
                         bendingImpact / dZ / dZ;
    lastParamCov(4, 3) = lastParamCov(3, 4);
  } else {
    lastParamCov(4, 4) = inverseBendingMomentum * inverseBendingMomentum;
  }
  param.setCovariances(lastParamCov);

  // set other parameters
  param.setClusterPtr(&cl2);
  param.setTrackChi2(0.);
}

//_________________________________________________________________________________________________
void TrackFitter::addCluster(const TrackParam& startingParam, const Cluster& cl, TrackParam& param)
{
  /// Extrapolate the starting track parameters to the z position of the new cluster
  /// accounting for MCS dispersion in the current chamber and the other(s) crossed
  /// Recompute the parameters adding the cluster constraint with the Kalman filter
  /// Throw an exception in case of failure

  if (cl.getZ() <= startingParam.getZ()) {
    throw runtime_error("The new cluster must be upstream");
  }

  // copy the current parameters into the new ones
  param.setParameters(startingParam.getParameters());
  param.setZ(startingParam.getZ());
  param.setCovariances(startingParam.getCovariances());
  param.setTrackChi2(startingParam.getTrackChi2());

  // add MCS effect in the current chamber
  int currentChamber(startingParam.getClusterPtr()->getChamberId());
  TrackExtrap::addMCSEffect(&param, SChamberThicknessInX0[currentChamber], -1.);

  // reset propagator for smoother
  if (mSmooth) {
    param.resetPropagator();
  }

  // add MCS in missing chambers if any
  int expectedChamber(currentChamber - 1);
  currentChamber = cl.getChamberId();
  while (currentChamber < expectedChamber) {
    if (!TrackExtrap::extrapToZCov(&param, SDefaultChamberZ[expectedChamber], mSmooth)) {
      throw runtime_error("Track extrapolation failed");
    }
    TrackExtrap::addMCSEffect(&param, SChamberThicknessInX0[expectedChamber], -1.);
    expectedChamber--;
  }

  // extrapolate to the z position of the new cluster
  if (!TrackExtrap::extrapToZCov(&param, cl.getZ(), mSmooth)) {
    throw runtime_error("Track extrapolation failed");
  }

  // save extrapolated parameters and covariances for smoother
  if (mSmooth) {
    param.setExtrapParameters(param.getParameters());
    param.setExtrapCovariances(param.getCovariances());
  }

  // recompute the parameters
  param.setClusterPtr(&cl);
  runKalmanFilter(param);
}

//_________________________________________________________________________________________________
void TrackFitter::smoothTrack(Track& track, bool finalize)
{
  /// Recompute the track parameters at each cluster using the Smoother
  /// Smoothed parameters are stored in dedicated data members
  /// If finalize, they are copied in the regular parameters in case of success
  /// Throw an exception in case of failure

  auto itCurrentParam(track.begin());
  auto itPreviousParam(itCurrentParam);
  ++itCurrentParam;

  // smoothed parameters and covariances at first cluster = filtered parameters and covariances
  itPreviousParam->setSmoothParameters(itPreviousParam->getParameters());
  itPreviousParam->setSmoothCovariances(itPreviousParam->getCovariances());

  // local chi2 at first cluster = last additional chi2 provided by Kalman
  itPreviousParam->setLocalChi2(itPreviousParam->getTrackChi2() - itCurrentParam->getTrackChi2());

  // recursively smooth the next parameters and covariances
  do {
    runSmoother(*itPreviousParam, *itCurrentParam);
    ++itPreviousParam;
  } while (++itCurrentParam != track.end());

  // update the regular parameters and covariances if requested
  if (finalize) {
    for (auto& param : track) {
      param.setParameters(param.getSmoothParameters());
      param.setCovariances(param.getSmoothCovariances());
    }
  }
}

//_________________________________________________________________________________________________
void TrackFitter::runKalmanFilter(TrackParam& trackParam)
{
  /// Compute the new track parameters including the attached cluster with the Kalman filter
  /// The current parameters are supposed to have been extrapolated to the cluster z position
  /// Throw an exception in case of failure

  // get actual track parameters (p)
  TMatrixD param(trackParam.getParameters());

  // get new cluster parameters (m)
  const Cluster* cluster = trackParam.getClusterPtr();
  TMatrixD clusterParam(5, 1);
  clusterParam.Zero();
  clusterParam(0, 0) = cluster->getX();
  clusterParam(2, 0) = cluster->getY();

  // compute the actual parameter weight (W)
  TMatrixD paramWeight(trackParam.getCovariances());
  if (paramWeight.Determinant() != 0) {
    paramWeight.Invert();
  } else {
    throw runtime_error("Determinant = 0");
  }

  // compute the new cluster weight (U)
  TMatrixD clusterWeight(5, 5);
  clusterWeight.Zero();
  if (mUseChamberResolution) {
    clusterWeight(0, 0) = 1. / mChamberResolutionX2;
    clusterWeight(2, 2) = 1. / mChamberResolutionY2;
  } else {
    clusterWeight(0, 0) = 1. / cluster->getEx2();
    clusterWeight(2, 2) = 1. / cluster->getEy2();
  }

  // compute the new parameters covariance matrix ((W+U)^-1)
  TMatrixD newParamCov(paramWeight, TMatrixD::kPlus, clusterWeight);
  if (newParamCov.Determinant() != 0) {
    newParamCov.Invert();
  } else {
    throw runtime_error("Determinant = 0");
  }
  trackParam.setCovariances(newParamCov);

  // compute the new parameters (p' = ((W+U)^-1)U(m-p) + p)
  TMatrixD tmp(clusterParam, TMatrixD::kMinus, param);   // m-p
  TMatrixD tmp2(clusterWeight, TMatrixD::kMult, tmp);    // U(m-p)
  TMatrixD newParam(newParamCov, TMatrixD::kMult, tmp2); // ((W+U)^-1)U(m-p)
  newParam += param;                                     // ((W+U)^-1)U(m-p) + p
  trackParam.setParameters(newParam);

  // compute the additional chi2 (= ((p'-p)^-1)W(p'-p) + ((p'-m)^-1)U(p'-m))
  tmp = newParam;                                                // p'
  tmp -= param;                                                  // (p'-p)
  TMatrixD tmp3(paramWeight, TMatrixD::kMult, tmp);              // W(p'-p)
  TMatrixD addChi2Track(tmp, TMatrixD::kTransposeMult, tmp3);    // ((p'-p)^-1)W(p'-p)
  tmp = newParam;                                                // p'
  tmp -= clusterParam;                                           // (p'-m)
  TMatrixD tmp4(clusterWeight, TMatrixD::kMult, tmp);            // U(p'-m)
  addChi2Track += TMatrixD(tmp, TMatrixD::kTransposeMult, tmp4); // ((p'-p)^-1)W(p'-p) + ((p'-m)^-1)U(p'-m)
  trackParam.setTrackChi2(trackParam.getTrackChi2() + addChi2Track(0, 0));
}

//_________________________________________________________________________________________________
void TrackFitter::runSmoother(const TrackParam& previousParam, TrackParam& param)
{
  /// Recompute the track parameters starting from the previous ones
  /// Throw an exception in case of failure

  // get variables
  const TMatrixD& extrapParameters = previousParam.getExtrapParameters();           // X(k+1 k)
  const TMatrixD& filteredParameters = param.getParameters();                       // X(k k)
  const TMatrixD& previousSmoothParameters = previousParam.getSmoothParameters();   // X(k+1 n)
  const TMatrixD& propagator = previousParam.getPropagator();                       // F(k)
  const TMatrixD& extrapCovariances = previousParam.getExtrapCovariances();         // C(k+1 k)
  const TMatrixD& filteredCovariances = param.getCovariances();                     // C(k k)
  const TMatrixD& previousSmoothCovariances = previousParam.getSmoothCovariances(); // C(k+1 n)

  // compute smoother gain: A(k) = C(kk) * F(k)^t * (C(k+1 k))^-1
  TMatrixD extrapWeight(extrapCovariances);
  if (extrapWeight.Determinant() != 0) {
    extrapWeight.Invert(); // (C(k+1 k))^-1
  } else {
    throw runtime_error("Determinant = 0");
  }
  TMatrixD smootherGain(filteredCovariances, TMatrixD::kMultTranspose, propagator); // C(kk) * F(k)^t
  smootherGain *= extrapWeight;                                                     // C(kk) * F(k)^t * (C(k+1 k))^-1

  // compute smoothed parameters: X(k n) = X(k k) + A(k) * (X(k+1 n) - X(k+1 k))
  TMatrixD tmpParam(previousSmoothParameters, TMatrixD::kMinus, extrapParameters); // X(k+1 n) - X(k+1 k)
  TMatrixD smoothParameters(smootherGain, TMatrixD::kMult, tmpParam);              // A(k) * (X(k+1 n) - X(k+1 k))
  smoothParameters += filteredParameters;                                          // X(k k) + A(k) * (X(k+1 n) - X(k+1 k))
  param.setSmoothParameters(smoothParameters);

  // compute smoothed covariances: C(k n) = C(k k) + A(k) * (C(k+1 n) - C(k+1 k)) * (A(k))^t
  TMatrixD tmpCov(previousSmoothCovariances, TMatrixD::kMinus, extrapCovariances); // C(k+1 n) - C(k+1 k)
  TMatrixD tmpCov2(tmpCov, TMatrixD::kMultTranspose, smootherGain);                // (C(k+1 n) - C(k+1 k)) * (A(k))^t
  TMatrixD smoothCovariances(smootherGain, TMatrixD::kMult, tmpCov2);              // A(k) * (C(k+1 n) - C(k+1 k)) * (A(k))^t
  smoothCovariances += filteredCovariances;                                        // C(k k) + A(k) * (C(k+1 n) - C(k+1 k)) * (A(k))^t
  param.setSmoothCovariances(smoothCovariances);

  // compute smoothed residual: r(k n) = cluster - X(k n)
  const Cluster* cluster = param.getClusterPtr();
  TMatrixD smoothResidual(2, 1);
  smoothResidual.Zero();
  smoothResidual(0, 0) = cluster->getX() - smoothParameters(0, 0);
  smoothResidual(1, 0) = cluster->getY() - smoothParameters(2, 0);

  // compute weight of smoothed residual: W(k n) = (clusterCov - C(k n))^-1
  TMatrixD smoothResidualWeight(2, 2);
  if (mUseChamberResolution) {
    smoothResidualWeight(0, 0) = mChamberResolutionX2 - smoothCovariances(0, 0);
    smoothResidualWeight(1, 1) = mChamberResolutionY2 - smoothCovariances(2, 2);
  } else {
    smoothResidualWeight(0, 0) = cluster->getEx2() - smoothCovariances(0, 0);
    smoothResidualWeight(1, 1) = cluster->getEy2() - smoothCovariances(2, 2);
  }
  smoothResidualWeight(0, 1) = -smoothCovariances(0, 2);
  smoothResidualWeight(1, 0) = -smoothCovariances(2, 0);
  if (smoothResidualWeight.Determinant() != 0) {
    smoothResidualWeight.Invert();
  } else {
    throw runtime_error("Determinant = 0");
  }

  // compute local chi2 = (r(k n))^t * W(k n) * r(k n)
  TMatrixD tmpChi2(smoothResidual, TMatrixD::kTransposeMult, smoothResidualWeight); // (r(k n))^t * W(k n)
  TMatrixD localChi2(tmpChi2, TMatrixD::kMult, smoothResidual);                     // (r(k n))^t * W(k n) * r(k n)
  param.setLocalChi2(localChi2(0, 0));
}

} // namespace mch
} // namespace o2
