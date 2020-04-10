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
/// \author Philippe Pillot, Subatech; adapted by Rafael Pezzi, UFRGS

#include "MFTBase/Constants.h"
#include "MFTTracking/TrackFitter.h"
#include "MFTTracking/TrackCA.h"
#include "MFTTracking/TrackExtrap.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "ITSMFTReconstruction/ChipMappingMFT.h"
#include <stdexcept>
#include <TMath.h>
#include <TMatrixD.h>
#include <TF1.h>
#include <TF2.h>
#include "CommonConstants/MathConstants.h"
#include "MathUtils/MathBase.h"
#include "MathUtils/Utils.h"

using o2::math_utils::math_base::fitGaus;

namespace o2
{
namespace mft
{

//_________________________________________________________________________________________________
void TrackFitter::setBz(float bZ)
{
  /// Set the magnetic field for the MFT
  mBZField = bZ;
  mTrackExtrap.setBz(bZ);
}

//_________________________________________________________________________________________________
void TrackFitter::fit(FitterTrackMFT& track, bool smooth, bool finalize,
                      std::list<TrackParamMFT>::reverse_iterator* itStartingParam)
{
  /// Fit a track to its attached clusters
  /// Smooth the track if requested and the smoother enabled
  /// If finalize = true: copy the smoothed parameters, if any, into the regular ones
  /// Fit the entire track or only the part upstream itStartingParam
  /// Throw an exception in case of failure

  std::cout << "\n ***************************** Start Fitting new track ***************************** \n";

  // initialize the starting track parameters and cluster
  auto itParam(track.rbegin());
  if (itStartingParam != nullptr) {
    // use the ones pointed to by itStartingParam
    if (*itStartingParam == track.rend()) {
      throw std::runtime_error("invalid starting parameters");
    }
    itParam = *itStartingParam;
  } else {
    // or start from the last cluster and compute the track parameters from its position
    // and the one of the first previous cluster found on a different layer
    auto itPreviousParam(itParam);
    ++itPreviousParam;
    // TODO: Refine criteria for initTrack: cluster from next MFT layer or next disk?
    //(*itParam).setInvPx(1.0); // TODO: Improve initial momentum estimate
    //(*itParam).setInvPy(1.0); // TODO: Improve initial momentum estimate
    //(*itParam).setInvQPz(1.0); // TODO: Improve initial momentum estimate
    initTrack(*itParam->getClusterPtr(), *itParam);
  }

  std::cout << "Seed covariances:";
  itParam->getCovariances().Print();

  // recusively add the upstream clusters and update the track parameters
  TrackParamMFT* startingParam = &*itParam;
  while (++itParam != track.rend()) {
    try {
      addCluster(*startingParam, *itParam->getClusterPtr(), *itParam);
      startingParam = &*itParam;
    } catch (std::exception const&) {
      throw;
    }
  }

  std::cout << "Track covariances:";
  itParam--;
  itParam->getCovariances().Print();
  std::cout << "Track Chi2 = " << itParam->getTrackChi2() << std::endl;
  std::cout << " ***************************** Done fitting *****************************\n";
  // smooth the track if requested and the smoother enabled
  if (smooth && mSmooth) {
    try {
      smoothTrack(track, finalize);
    } catch (std::exception const&) {
      throw;
    }
  }
}

//_________________________________________________________________________________________________
void TrackFitter::initTrack(const o2::itsmft::Cluster& cl, TrackParamMFT& param)
{

  /// Compute the initial track parameters at the z position of the last cluster (cl)
  /// The covariance matrix is computed such that the last cluster is the only constraint
  /// (by assigning an infinite dispersion to the other cluster)
  /// These parameters are the seed for the Kalman filter

  // compute the track parameters at the last cluster

  //double k = -mBZField * o2::constants::math::B2C;
  double x0 = cl.getX();
  double y0 = cl.getY();
  double dZ = cl.getZ();
  double pt = TMath::Sqrt(x0 * x0 + y0 * y0) * 1;
  double pz = dZ * 1;
  //double phi1 = TMath::ATan2(cl1.getY(), cl1.getX());
  double phi2 = TMath::ATan2(cl.getY(), cl.getX());
  double tanl = pz / pt;
  double r0sq = cl.getX() * cl.getX() + cl.getY() * cl.getY();
  double r0cu = r0sq * TMath::Sqrt(r0sq);
  double sigmax0sq = cl.getSigmaZ2();       // FIXME: from cluster
  double sigmay0sq = cl.getSigmaY2();       // FIXME: from cluster
  double sigmaDeltaZsq = 5;        // Primary vertex distribution: beam interaction diamond
  double sigmaboost = 5e3;         // Boost q/pt seed covariances
  double sigmainvqptsq = sigmaboost / r0cu / r0cu * (sigmax0sq * x0 * x0 + sigmay0sq * y0 * y0); // .25 / pt; // Very large uncertainty on pt

  std::cout << "initTrack: pt = " << pt << std::endl;
  std::cout << " -iniTrack: cluster    X =  " << cl.getX() << " Y = " << cl.getY() << " Z = " << cl.getZ() << " phi2 = " << phi2 << std::endl;

  //std::cout << " mBZField = " << mBZField << std::endl;

  param.setX(cl.getX());
  param.setY(cl.getY());
  param.setZ(cl.getZ());
  param.setPhi(phi2);
  param.setInvQPt(1.0 / pt);
  param.setTanl(tanl);
  std::cout << "  seed Phi, Tanl, InvQpt = " << param.getPhi() << " " << param.getTanl() << " " << param.getInvQPt() << std::endl;

  // compute the track parameter covariances at the last cluster (as if the other clusters did not exist)
  TMatrixD lastParamCov(5, 5);
  lastParamCov.Zero();
  lastParamCov(0, 0) = sigmax0sq;                   // <X,X>
  lastParamCov(0, 1) = 0;                           // <Y,X>
  lastParamCov(0, 2) = -sigmax0sq * y0 / r0sq;      // <PHI,X>
  lastParamCov(0, 3) = -dZ * sigmax0sq * x0 / r0cu; // <TANL,X>
  lastParamCov(0, 4) = sigmaboost * -x0 * sigmax0sq / r0cu; // <INVQPT,X>

  lastParamCov(1, 1) = sigmay0sq;                   // <Y,Y>
  lastParamCov(1, 2) = sigmay0sq * x0 / r0sq;       // <PHI,Y>
  lastParamCov(1, 3) = -dZ * sigmay0sq * y0 / r0cu; // <TANL,Y>
  lastParamCov(1, 4) = sigmaboost * y0 * sigmay0sq / r0cu; //1e-2; // <INVQPT,Y>

  lastParamCov(2, 2) = (sigmax0sq * y0 * y0 + sigmay0sq * x0 * x0) / r0sq / r0sq;    // <PHI,PHI>
  lastParamCov(2, 3) = dZ * x0 * y0 * (sigmax0sq - sigmay0sq) / r0sq / r0cu; //  <TANL,PHI>
  lastParamCov(2, 4) = sigmaboost *  y0 * x0 / r0cu / r0sq * (sigmax0sq - sigmay0sq); //  <INVQPT,PHI>

  lastParamCov(3, 3) = dZ * dZ * (sigmax0sq * x0 * x0 + sigmay0sq * y0 * y0) / r0cu / r0cu + sigmaDeltaZsq / r0sq; // <TANL,TANL>
  lastParamCov(3, 4) = sigmaboost * dZ / r0cu / r0cu * (sigmax0sq * x0 * x0 + sigmay0sq * y0 * y0);         // <INVQPT,TANL>

  lastParamCov(4, 4) = sigmainvqptsq; // <INVQPT,INVQPT>

  lastParamCov(1, 0) = lastParamCov(0, 1); //
  lastParamCov(2, 0) = lastParamCov(0, 2); //
  lastParamCov(2, 1) = lastParamCov(1, 2); //
  lastParamCov(3, 0) = lastParamCov(0, 3); //
  lastParamCov(3, 1) = lastParamCov(1, 3); //
  lastParamCov(3, 2) = lastParamCov(2, 3); //
  lastParamCov(4, 0) = lastParamCov(0, 4); //
  lastParamCov(4, 1) = lastParamCov(1, 4); //
  lastParamCov(4, 2) = lastParamCov(2, 4); //
  lastParamCov(4, 3) = lastParamCov(3, 4); //

  param.setCovariances(lastParamCov);

  // set other parameters
  param.setClusterPtr(&cl);
  param.setTrackChi2(0.);
}

//_________________________________________________________________________________________________
void TrackFitter::addCluster(const TrackParamMFT& startingParam, const o2::itsmft::Cluster& cl, TrackParamMFT& param)
{
  /// Extrapolate the starting track parameters to the z position of the new cluster
  /// accounting for MCS dispersion in the current layer and the other(s) crossed
  /// Recompute the parameters adding the cluster constraint with the Kalman filter
  /// Throw an exception in case of failure

  if (cl.getZ() <= startingParam.getZ()) {
    LOG(ERROR) << "AddCluster ERROR: The new cluster must be upstream! ********************* ";
    // FIXME! This should throw an error. Skiping due to bug on track finding.
    //throw std::runtime_error("The new cluster must be upstream");
  }
  std::cout << "addCluster:     X = " << cl.getX() << " Y = " << cl.getY() << " Z = " << cl.getZ() << std::endl;
  // copy the current parameters into the new ones
  param.setParameters(startingParam.getParameters());
  param.setZ(startingParam.getZ());
  param.setCovariances(startingParam.getCovariances());
  param.setTrackChi2(startingParam.getTrackChi2());

  // add MCS effect in the current layer
  o2::itsmft::ChipMappingMFT mftChipMapper;
  int currentLayer(mftChipMapper.chip2Layer(startingParam.getClusterPtr()->getSensorID()));
  //mTrackExtrap.addMCSEffect(&param, SLayerThicknessInX0[currentLayer], -1.);

  // reset propagator for smoother
  if (mSmooth) {
    param.resetPropagator();
  }

  std::cout << "  BeforeExtrap: X = " << param.getX() << " Y = " << param.getY() << " Z = " << param.getZ() << " Tgl = " << param.getTanl() << "  Phi = " << param.getPhi() << " pz = " << param.getPz() << " pt = " << param.getPt() << std::endl;
  //param.getCovariances().Print();

  // add MCS in missing layers if any
  int expectedLayer(currentLayer - 1);
  currentLayer = mftChipMapper.chip2Layer(cl.getSensorID());
  while (currentLayer < expectedLayer) {
    if (!mTrackExtrap.extrapToZCov(&param, o2::mft::constants::LayerZPosition[expectedLayer], mSmooth)) {
      throw std::runtime_error("1. Track extrapolation failed");
    }
    //mTrackExtrap.addMCSEffect(&param, SLayerThicknessInX0[expectedLayer], -1.);
    expectedLayer--;
  }

  // extrapolate to the z position of the new cluster
  if (!mTrackExtrap.extrapToZCov(&param, cl.getZ(), mSmooth)) {
    throw std::runtime_error("2. Track extrapolation failed");
  }

  std::cout << "   AfterExtrap: X = " << param.getX() << " Y = " << param.getY() << " Z = " << param.getZ() << " Tgl = " << param.getTanl() << "  Phi = " << param.getPhi() << " pz = " << param.getPz() << " pt = " << param.getPt() << std::endl;
  //param.getCovariances().Print();

  // save extrapolated parameters and covariances for smoother
  if (mSmooth) {
    param.setExtrapParameters(param.getParameters());
    param.setExtrapCovariances(param.getCovariances());
  }

  // recompute the parameters
  param.setClusterPtr(&cl);
  try {
    runKalmanFilter(param);
  } catch (std::exception const&) {
    throw;
  }

  std::cout << "   New Cluster: X = " << cl.getX() << " Y = " << cl.getY() << " Z = " << cl.getZ() << std::endl;

  std::cout << "   AfterKalman: X = " << param.getX() << " Y = " << param.getY() << " Z = " << param.getZ() << " Tgl = " << param.getTanl() << "  Phi = " << param.getPhi() << " pz = " << param.getPz() << " pt = " << param.getPt() << std::endl;
  //param.getCovariances().Print();
  std::cout << std::endl;
}

//_________________________________________________________________________________________________
void TrackFitter::smoothTrack(FitterTrackMFT& track, bool finalize)
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
    try {
      runSmoother(*itPreviousParam, *itCurrentParam);
    } catch (std::exception const&) {
      throw;
    }
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
void TrackFitter::runKalmanFilter(TrackParamMFT& trackParam)
{
  /// Compute the new track parameters including the attached cluster with the Kalman filter
  /// The current parameters are supposed to have been extrapolated to the cluster z position
  /// Throw an exception in case of failure

  // get actual track parameters (p)
  TMatrixD param(trackParam.getParameters());

  // get new cluster parameters (m)
  const o2::itsmft::Cluster* cluster = trackParam.getClusterPtr();
  TMatrixD clusterParam(5, 1);
  clusterParam.Zero();
  clusterParam(0, 0) = cluster->getX();
  clusterParam(1, 0) = cluster->getY();

  // compute the actual parameter weight (W)
  TMatrixD paramWeight(trackParam.getCovariances());
  if (paramWeight.Determinant() != 0) {
    paramWeight.Invert();
  } else {
    throw std::runtime_error("Determinant = 0");
  }

  // compute the new cluster weight (U)
  TMatrixD clusterWeight(5, 5);
  clusterWeight.Zero();
  clusterWeight(0, 0) = 1. / cluster->getSigmaZ2(); // 1. / cluster->getEx2();
  clusterWeight(1, 1) = 1. / cluster->getSigmaY2(); //  1. / cluster->getEy2();

  // compute the new parameters covariance matrix ((W+U)^-1)
  TMatrixD newParamCov(paramWeight, TMatrixD::kPlus, clusterWeight);
  if (newParamCov.Determinant() != 0) {
    newParamCov.Invert();
  } else {
    throw std::runtime_error("Determinant = 0");
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
void TrackFitter::runSmoother(const TrackParamMFT& previousParam, TrackParamMFT& param)
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
    throw std::runtime_error("Determinant = 0");
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
  const o2::itsmft::Cluster* cluster = param.getClusterPtr();
  TMatrixD smoothResidual(2, 1);
  smoothResidual.Zero();
  smoothResidual(0, 0) = cluster->getX() - smoothParameters(0, 0);
  smoothResidual(1, 0) = cluster->getY() - smoothParameters(1, 0);

  // compute weight of smoothed residual: W(k n) = (clusterCov - C(k n))^-1
  TMatrixD smoothResidualWeight(2, 2);
  smoothResidualWeight(0, 0) = cluster->getSigmaZ2() - smoothCovariances(0, 0); // cluster->getEx2() - smoothCovariances(0, 0);
  smoothResidualWeight(0, 1) = -smoothCovariances(0, 2);
  smoothResidualWeight(1, 0) = -smoothCovariances(2, 0);
  smoothResidualWeight(1, 1) = cluster->getSigmaY2() - smoothCovariances(2, 2); // cluster->getEy2() - smoothCovariances(2, 2);
  if (smoothResidualWeight.Determinant() != 0) {
    smoothResidualWeight.Invert();
  } else {
    throw std::runtime_error("Determinant = 0");
  }

  // compute local chi2 = (r(k n))^t * W(k n) * r(k n)
  TMatrixD tmpChi2(smoothResidual, TMatrixD::kTransposeMult, smoothResidualWeight); // (r(k n))^t * W(k n)
  TMatrixD localChi2(tmpChi2, TMatrixD::kMult, smoothResidual);                     // (r(k n))^t * W(k n) * r(k n)
  param.setLocalChi2(localChi2(0, 0));
}

} // namespace mft
} // namespace o2
