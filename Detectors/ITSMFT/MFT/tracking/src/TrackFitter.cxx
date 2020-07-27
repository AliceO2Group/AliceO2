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
#include "MFTTracking/Cluster.h"
#include "DataFormatsMFT/TrackMFT.h"
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
  auto& mftTrackingParam = MFTTrackingParam::Instance();

  /// Set the magnetic field for the MFT
  mBZField = bZ;

  if (mftTrackingParam.verbose) {
    LOG(INFO) << "Setting Fitter field = " << bZ;
  }
}

//_________________________________________________________________________________________________
bool TrackFitter::fit(TrackLTF& track, bool outward)
{

  /// Fit a track to its attached clusters
  /// Fit the entire track or only the part upstream itStartingParam
  /// Returns false in case of failure

  auto& mftTrackingParam = MFTTrackingParam::Instance();
  auto nClusters = track.getNPoints();

  if (mftTrackingParam.verbose) {
    std::cout << "\n ***************************** Start Fitting new track ***************************** \n";
    std::cout << "N Clusters = " << nClusters << std::endl;
  }


  if (mftTrackingParam.verbose) {
    std::cout << "Seed covariances: \n";
    track.getCovariances().Print(std::cout);
    std::cout << std::endl;
  }

  // recursively add clusters and update the track parameters
  if (!outward) { // Inward
    nClusters--;
    while (nClusters-- > 0) {
      if (!addCluster(track, nClusters)) {
        return false;
      }
    }
  } else { // Outward for MCH matching
    int ncl = 1;
    while (ncl < nClusters) {
      if (!addCluster(track, ncl)) {
        return false;
      }
      ncl++;
    }
  }
  if (mftTrackingParam.verbose) {
    //  std::cout << "Track covariances:";
    //  trac->getCovariances().Print();
    std::cout << "Track Chi2 = " << track.getTrackChi2() << std::endl;
    std::cout << " ***************************** Done fitting *****************************\n";
  }

  return true;
}

//_________________________________________________________________________________________________
bool TrackFitter::initTrack(TrackLTF& track, bool outward)
{
  // initialize the starting track parameters and cluster
  double chi2invqptquad;
  double invpqtquad;
  auto nPoints = track.getNPoints();
  invpqtquad = invQPtFromFCF(track, mBZField, chi2invqptquad);
  track.setInvQPtQuadtratic(invpqtquad);
  track.setChi2QPtQuadtratic(chi2invqptquad);
  track.setInvQPt(invpqtquad);

  /// Compute the initial track parameters
  /// The covariance matrix is computed such that the last cluster is the only constraint
  /// (by assigning an infinite dispersion to the other cluster)
  /// These parameters are the seed for the Kalman filter

  auto& mftTrackingParam = MFTTrackingParam::Instance();

  // compute track parameters
  int where; // First or last cluster?
  if (outward)
    where = 0;
  else
    where = nPoints - 1;

  double x0 = track.getXCoordinates()[where];
  double y0 = track.getYCoordinates()[where];
  double z0 = track.getZCoordinates()[where];
  double pt = TMath::Sqrt(x0 * x0 + y0 * y0);
  double pz = z0;
  double phi0 = TMath::ATan2(y0, x0);
  double tanl = pz / pt;
  double r0sq = x0 * x0 + y0 * y0;
  double r0cu = r0sq * TMath::Sqrt(r0sq);
  double invr0sq = 1.0 / r0sq;
  double invr0cu = 1.0 / r0cu;
  double sigmax0sq = 5e-4;
  double sigmay0sq = 5.43e-4;
  double sigmaDeltaZsq = 5.0;                      // Primary vertex distribution: beam interaction diamond
  double sigmaboost = mftTrackingParam.sigmaboost; // Boost q/pt seed covariances
  double seedH_k = mftTrackingParam.seedH_k;       // SeedH constant

  track.setX(x0);
  track.setY(y0);
  track.setZ(z0);
  track.setPhi(phi0);
  track.setTanl(tanl);

  // Configure the track seed
  switch (mftTrackingParam.seed) {
    case AB:
      if (mftTrackingParam.verbose)
        std::cout << " Init track with Seed A / B; sigmaboost = " << sigmaboost << ".\n";
      track.setInvQPt(1.0 / pt); // Seeds A & B
      break;
    case CE:
      if (mftTrackingParam.verbose)
        std::cout << " Init track with Seed C / E; sigmaboost = " << sigmaboost << ".\n";
      track.setInvQPt(std::copysign(1.0, track.getInvQPt()) / pt); // Seeds C & E
      break;
    case DH:
      if (mftTrackingParam.verbose)
        std::cout << " Init track with Seed H; (k = " << seedH_k << "); sigmaboost = " << sigmaboost << ".\n";
      track.setInvQPt(track.getInvQPt() / seedH_k); // SeedH
      break;
    default:
      if (mftTrackingParam.verbose)
        std::cout << " Init track with Seed D.\n";
      break;
  }
  if (mftTrackingParam.verbose) {
    auto model = (mftTrackingParam.trackmodel == Helix) ? "Helix" : (mftTrackingParam.trackmodel == Quadratic) ? "Quadratic" : "Linear";
    std::cout << "Track Model: " << model << std::endl;
    std::cout << "  initTrack: X = " << x0 << " Y = " << y0 << " Z = " << z0 << " Tgl = " << track.getTanl() << "  Phi = " << track.getPhi() << " pz = " << track.getPz() << " qpt = " << 1.0 / track.getInvQPt() << std::endl;
  }

  // compute the track parameter covariances at the last cluster (as if the other clusters did not exist)
  SMatrix55 lastParamCov;
  lastParamCov(0, 0) = sigmax0sq;                                   // <X,X>
  lastParamCov(0, 1) = 0;                                           // <Y,X>
  lastParamCov(0, 2) = sigmaboost * -sigmax0sq * y0 * invr0sq;      // <PHI,X>
  lastParamCov(0, 3) = sigmaboost * -z0 * sigmax0sq * x0 * invr0cu; // <TANL,X>
  lastParamCov(0, 4) = sigmaboost * -x0 * sigmax0sq * invr0cu;      // <INVQPT,X>

  lastParamCov(1, 1) = sigmay0sq;                                   // <Y,Y>
  lastParamCov(1, 2) = sigmaboost * sigmay0sq * x0 * invr0sq;       // <PHI,Y>
  lastParamCov(1, 3) = sigmaboost * -z0 * sigmay0sq * y0 * invr0cu; // <TANL,Y>
  lastParamCov(1, 4) = sigmaboost * y0 * sigmay0sq * invr0cu;       //1e-2; // <INVQPT,Y>

  lastParamCov(2, 2) = sigmaboost * (sigmax0sq * y0 * y0 + sigmay0sq * x0 * x0) * invr0sq * invr0sq; // <PHI,PHI>
  lastParamCov(2, 3) = sigmaboost * z0 * x0 * y0 * (sigmax0sq - sigmay0sq) * invr0sq * invr0cu;      //  <TANL,PHI>
  lastParamCov(2, 4) = sigmaboost * y0 * x0 * invr0cu * invr0sq * (sigmax0sq - sigmay0sq);           //  <INVQPT,PHI>

  lastParamCov(3, 3) = sigmaboost * z0 * z0 * (sigmax0sq * x0 * x0 + sigmay0sq * y0 * y0) * invr0cu * invr0cu + sigmaDeltaZsq * invr0sq; // <TANL,TANL>
  lastParamCov(3, 4) = sigmaboost * z0 * invr0cu * invr0cu * (sigmax0sq * x0 * x0 + sigmay0sq * y0 * y0);                                // <INVQPT,TANL>

  lastParamCov(4, 4) = sigmaboost * sigmaboost * (sigmax0sq * x0 * x0 + sigmay0sq * y0 * y0) * invr0cu * invr0cu; // <INVQPT,INVQPT>

  track.setCovariances(lastParamCov);
  track.setTrackChi2(0.);

  return true;
}

bool TrackFitter::addCluster(TrackLTF& track, int cluster)
{
  /// Propagate track to the z position of the new cluster
  /// accounting for MCS dispersion in the current layer and the other(s) crossed
  /// Recompute the parameters adding the cluster constraint with the Kalman filter
  /// Returns false in case of failure

  auto& mftTrackingParam = MFTTrackingParam::Instance();
  const auto& clx = track.getXCoordinates()[cluster];
  const auto& cly = track.getYCoordinates()[cluster];
  const auto& clz = track.getZCoordinates()[cluster];

  if (track.getZ() == clz) {
    LOG(INFO) << "AddCluster ERROR: The new cluster must be upstream! Bug on TrackFinder. " << (track.isCA() ? " CATrack" : "LTFTrack");
    LOG(INFO) << "track.getZ() = " << track.getZ() << " ; newClusterZ = " << clz << " ==> Skipping point.";
    return true;
  }
  if (mftTrackingParam.verbose)
    std::cout << "addCluster:     X = " << clx << " Y = " << cly << " Z = " << clz << " nCluster = " << cluster << std::endl;

  // add MCS effects for the new cluster
  using o2::mft::constants::LayerZPosition;
  int startingLayerID, newLayerID;

  double dZ = clz - track.getZ();
  //LayerID of each cluster from ZPosition // TODO: Use ChipMapping
  for (auto layer = 10; layer--;)
    if (track.getZ() < LayerZPosition[layer] + .3 & track.getZ() > LayerZPosition[layer] - .3)
      startingLayerID = layer;
  for (auto layer = 10; layer--;)
    if (clz<LayerZPosition[layer] + .3 & clz> LayerZPosition[layer] - .3)
      newLayerID = layer;
  // Number of disks crossed by this tracklet
  int NDisksMS;
  if (clz - track.getZ() > 0)
    NDisksMS = (startingLayerID % 2 == 0) ? (startingLayerID - newLayerID) / 2 : (startingLayerID - newLayerID + 1) / 2;
  else
    NDisksMS = (startingLayerID % 2 == 0) ? (newLayerID - startingLayerID + 1) / 2 : (newLayerID - startingLayerID) / 2;

  double MFTDiskThicknessInX0 = mftTrackingParam.MFTRadLenghts / 5.0;
  if (mftTrackingParam.verbose) {
    std::cout << "startingLayerID = " << startingLayerID << " ; "
              << "newLayerID = " << newLayerID << " ; ";
    std::cout << "cl.getZ() = " << clz << " ; ";
    std::cout << "startingParam.getZ() = " << track.getZ() << " ; ";
    std::cout << "NDisksMS = " << NDisksMS << std::endl;
  }

  if ((NDisksMS * MFTDiskThicknessInX0) != 0)
    track.addMCSEffect(-1, NDisksMS * MFTDiskThicknessInX0);

  if (mftTrackingParam.verbose)
    std::cout << "  BeforeExtrap: X = " << track.getX() << " Y = " << track.getY() << " Z = " << track.getZ() << " Tgl = " << track.getTanl() << "  Phi = " << track.getPhi() << " pz = " << track.getPz() << " qpt = " << 1.0 / track.getInvQPt() << std::endl;

  // Propagate track to the z position of the new cluster
  switch (mftTrackingParam.trackmodel) {
    case Linear:
      track.propagateToZlinear(clz);
      break;
    case Quadratic:
      track.propagateToZquadratic(clz, mBZField);
      break;
    case Helix:
      track.propagateToZhelix(clz, mBZField);
      break;
    default:
      std::cout << " Invalid track model.\n";
      return false;
      break;
  }

  if (mftTrackingParam.verbose)
    std::cout << "   AfterExtrap: X = " << track.getX() << " Y = " << track.getY() << " Z = " << track.getZ() << " Tgl = " << track.getTanl() << "  Phi = " << track.getPhi() << " pz = " << track.getPz() << " qpt = " << 1.0 / track.getInvQPt() << std::endl;

  // recompute the parameters
  if (runKalmanFilter(track, cluster)) {
    if (mftTrackingParam.verbose) {
      std::cout << "   New Cluster: X = " << clx << " Y = " << cly << " Z = " << clz << std::endl;
      std::cout << "   AfterKalman: X = " << track.getX() << " Y = " << track.getY() << " Z = " << track.getZ() << " Tgl = " << track.getTanl() << "  Phi = " << track.getPhi() << " pz = " << track.getPz() << " qpt = " << 1.0 / track.getInvQPt() << std::endl;
      std::cout << std::endl;
      // Outputs track covariance matrix:
      // param.getCovariances().Print();
    }
    return true;
  }
  return false;
}

//_________________________________________________________________________________________________
bool TrackFitter::runKalmanFilter(TrackLTF& track, int cluster)
{
  /// Compute the new track parameters including the attached cluster with the Kalman filter
  /// The current track is expected to have been propagated to the cluster z position
  /// Retruns false in case of failure

  // get propagated track parameters (p)
  auto param = track.getParameters();

  // get new cluster parameters (m)
  SMatrix5 clusterParam;
  clusterParam(0) = track.getXCoordinates()[cluster];
  clusterParam(1) = track.getYCoordinates()[cluster];

  // compute the actual parameter weight (W)
  SMatrix55 paramWeight(track.getCovariances());
  if (!(paramWeight.Invert())) {
    LOG(INFO) << "runKalmanFilter ERROR: Determinant = 0";
    return false;
  }

  // compute the new cluster weight (U)
  SMatrix55 clusterWeight;
  clusterWeight(0, 0) = 1. / 5e-4; // FIXME
  clusterWeight(1, 1) = 1. / 5.43e-4;

  // compute the new parameters covariance matrix ((W+U)^-1)
  SMatrix55 newParamCov(paramWeight + clusterWeight);
  if (!newParamCov.Invert()) {
    LOG(INFO) << "runKalmanFilter ERROR: Determinant = 0";
    return false;
  }
  track.setCovariances(newParamCov);

  // compute the new parameters: (p' = ((W+U)^-1)U(m-p) + p)
  // Parameters increment: p' - p = ((W+U)^-1)U(m-p)
  SMatrix5 predict_residuals(clusterParam - param); // m-p   -> residuals of prediction
  SMatrix5 tmp(clusterWeight * predict_residuals);  // U(m-p)
  SMatrix5 newParamDelta(newParamCov * tmp);        // ((W+U)^-1)U(m-p)
  SMatrix5 newParam = newParamDelta + param;        // ((W+U)^-1)U(m-p) + p
  track.setParameters(newParam);

  // compute the additional addChi2 = ((p'-p)^t)W(p'-p) + ((p'-m)^t)U(p'-m)
  SMatrix5 tmp2(clusterParam - newParam); // (m-p)
  auto addChi2Track(ROOT::Math::Similarity(newParamDelta, paramWeight) + ROOT::Math::Similarity(tmp2, clusterWeight));
  track.setTrackChi2(track.getTrackChi2() + addChi2Track);

  return true;
}

//__________________________________________________________________________
Double_t invQPtFromParabola2(const TrackLTF& track, Double_t bFieldZ, Double_t& chi2)
{
  auto nPoints = track.getNPoints();

  const std::array<Float_t, constants::mft::LayersNumber>& xPositons = track.getXCoordinates();
  const std::array<Float_t, constants::mft::LayersNumber>& yPositons = track.getYCoordinates();

  //rotate track to stabilize quadratic fitting
  auto deltax = xPositons[0] - xPositons[nPoints - 1];
  auto deltay = yPositons[0] - yPositons[nPoints - 1];
  //auto x_m = (xPositons[nPoints - 1] + xPositons[0]) / 2;
  //auto y_m = (yPositons[nPoints - 1] + yPositons[0]) / 2;
  auto theta = -TMath::ATan2(deltay, deltax);
  auto costheta = TMath::Cos(theta), sintheta = TMath::Sin(theta);

  bool verbose = false;
  if (verbose) {
    std::cout << "First and last cluster X,Y => " << xPositons[0] << " , " << yPositons[0] << "     /  " << xPositons[nPoints - 1] << " , " << yPositons[nPoints - 1] << std::endl;
    std::cout << " Angle to rotate: " << theta << " ( " << theta * TMath::RadToDeg() << " deg ) ; nPoints = " << nPoints << std::endl;
  }

  Double_t* x = new Double_t[nPoints];
  Double_t* y = new Double_t[nPoints];
  for (auto n = 0; n < nPoints; n++) {
    auto x_0 = xPositons[n]; // - x_m;
    auto y_0 = yPositons[n]; // - y_m;
    x[n] = x_0 * costheta - y_0 * sintheta;
    y[n] = x_0 * sintheta + y_0 * costheta;
    if (verbose)
      std::cout << "    adding rotated point to fit at z = " << track.getZCoordinates()[n] << " (" << x[n] << "," << y[n] << ") " << std::endl;
  }

  Double_t q0, q1, q2;
  chi2 = QuadraticRegression2(nPoints, x, y, q0, q1, q2);
  Double_t radiusParabola = 0.5 / q2;
  auto invqpt_parabola = q2 / (o2::constants::math::B2C * bFieldZ * 0.5); // radiusParabola; // radius = 0.5/q2

  if (verbose) {
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << " Fit QuadraticRegression: " << std::endl;
    std::cout << " Fit Parameters [0] = " << q0 << " [1] =  " << q1 << " [2] = " << q2 << std::endl;
    std::cout << " Radius from QuadraticRegression = " << 0.5 / q2 << std::endl;
    std::cout << " Seed qpt = " << 1.0 / invqpt_parabola << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
  }

  return invqpt_parabola;
}

//__________________________________________________________________________
Double_t QuadraticRegression2(Int_t nVal, Double_t* xVal, Double_t* yVal, Double_t& p0, Double_t& p1, Double_t& p2)
{
  /// Perform a Quadratic Regression
  /// Assume same error on all clusters = 1
  /// Return ~ Chi2

  TMatrixD y(nVal, 1);
  TMatrixD x(nVal, 3);
  TMatrixD xtrans(3, nVal);

  for (int i = 0; i < nVal; i++) {
    y(i, 0) = yVal[i];
    x(i, 0) = 1.;
    x(i, 1) = xVal[i];
    x(i, 2) = xVal[i] * xVal[i];
    xtrans(0, i) = 1.;
    xtrans(1, i) = xVal[i];
    xtrans(2, i) = xVal[i] * xVal[i];
  }
  TMatrixD tmp(xtrans, TMatrixD::kMult, x);
  tmp.Invert();

  TMatrixD tmp2(xtrans, TMatrixD::kMult, y);
  TMatrixD b(tmp, TMatrixD::kMult, tmp2);

  p0 = b(0, 0);
  p1 = b(1, 0);
  p2 = b(2, 0);

  // chi2 = (y-xb)^t . W . (y-xb)
  TMatrixD tmp3(x, TMatrixD::kMult, b);
  TMatrixD tmp4(y, TMatrixD::kMinus, tmp3);
  TMatrixD chi2(tmp4, TMatrixD::kTransposeMult, tmp4);

  return chi2(0, 0);
}

Double_t invQPtFromFCF(const TrackLTF& track, Double_t bFieldZ, Double_t& chi2)
{

  const std::array<Float_t, constants::mft::LayersNumber>& xPositons = track.getXCoordinates();
  const std::array<Float_t, constants::mft::LayersNumber>& yPositons = track.getYCoordinates();
  const std::array<Float_t, constants::mft::LayersNumber>& zPositons = track.getZCoordinates();

  // Fast Circle Fit (Hansroul, Jeremie, Savard, 1987)
  auto nPoints = track.getNPoints();
  Double_t* xVal = new Double_t[nPoints];
  Double_t* yVal = new Double_t[nPoints];
  Double_t* zVal = new Double_t[nPoints];
  Double_t* xErr = new Double_t[nPoints];
  Double_t* yErr = new Double_t[nPoints];
  Double_t* uVal = new Double_t[nPoints - 1];
  Double_t* vVal = new Double_t[nPoints - 1];
  Double_t* vErr = new Double_t[nPoints - 1];
  Double_t a, ae, b, be, x2, y2, invx2y2, rx, ry, r;

  for (auto np = 0; np < nPoints; np++) {
    //printf("BV track %d  %f  %f  %f \n", np, trackparam->getClusterPtr()->getX(), trackparam->getClusterPtr()->getY(), trackparam->getClusterPtr()->getZ());
    xErr[np] = 5e-4; // FIXME
    yErr[np] = 5e-4; // FIXME
    if (np > 0) {
      xVal[np] = xPositons[np] - xVal[0];
      yVal[np] = yPositons[np] - yVal[0];
      xErr[np] *= std::sqrt(2.);
      yErr[np] *= std::sqrt(2.);
    } else {
      xVal[np] = 0.;
      yVal[np] = 0.;
    }
    zVal[np] = zPositons[np];
  }
  for (int i = 0; i < (nPoints - 1); i++) {
    x2 = xVal[i + 1] * xVal[i + 1];
    y2 = yVal[i + 1] * yVal[i + 1];
    invx2y2 = 1. / (x2 + y2);
    uVal[i] = xVal[i + 1] * invx2y2;
    vVal[i] = yVal[i + 1] * invx2y2;
    vErr[i] = std::sqrt(8. * xErr[i + 1] * xErr[i + 1] * x2 * y2 + 2. * yErr[i + 1] * yErr[i + 1] * (x2 - y2)) * invx2y2 * invx2y2;
  }

  Double_t invqpt_fcf;
  Int_t qfcf;
  chi2 = 0.;
  if (LinearRegression((nPoints - 1), uVal, vVal, yErr, a, ae, b, be)) {
    // v = a * u + b
    // circle passing through (0,0):
    // (x - rx)^2 + (y - ry)^2 = r^2
    // ---> a = - rx / ry;
    // ---> b = 1 / (2 * ry)
    ry = 1. / (2. * b);
    rx = -a * ry;
    r = std::sqrt(rx * rx + ry * ry);

    // pt --->
    Double_t invpt = 1. / (o2::constants::math::B2C * bFieldZ * r);

    // sign(q) --->
    // rotate around the first point (0,0) to bring the last point
    // on the x axis (y = 0) and check the y sign of the rotated
    // center of the circle
    Double_t x = xVal[nPoints - 1], y = yVal[nPoints - 1], z = zVal[nPoints - 1];
    Double_t slope = TMath::ATan2(y, x);
    Double_t cosSlope = TMath::Cos(slope);
    Double_t sinSlope = TMath::Sin(slope);
    Double_t rxRot = rx * cosSlope + ry * sinSlope;
    Double_t ryRot = rx * sinSlope - ry * cosSlope;
    qfcf = (ryRot < 0.) ? -1 : +1;
    //printf("BV r-quad %f ,  r-fcf %f  q-fcf %f \n", 0.5 / q2, r, qfcf);

    //Double_t xRot = x * cosSlope + y * sinSlope;
    //printf("BV check %f %f \n", xRot, 2.0 * rxRot);

    Double_t alpha = 2.0 * std::abs(TMath::ATan2(rxRot, ryRot));
    Double_t x0 = xVal[0], y0 = yVal[0], z0 = zVal[0];
    Double_t dxyz2 = (x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0);
    Double_t cst = 1000.;
    Double_t c_alpha = cst * alpha;
    Double_t p, pt, pz;
    pt = 1. / invpt;
    p = std::sqrt(dxyz2) / c_alpha;
    pz = std::sqrt(p * p - pt * pt);
    //    tanl = pz / pt;
    invqpt_fcf = qfcf * invpt;
  } else { // the linear regression failed...
    printf("BV LinearRegression failed!\n");
    invqpt_fcf = 1. / 100.;
  }

  return invqpt_fcf;
}

//__________________________________________________________________________
Bool_t LinearRegression(Int_t nVal, Double_t* xVal, Double_t* yVal, Double_t* yErr, Double_t& a, Double_t& ae, Double_t& b, Double_t& be)
{
  // linear regression y = a * x + b

  Double_t S1, SXY, SX, SY, SXX, SsXY, SsXX, SsYY, Xm, Ym, s, delta, difx;
  Double_t invYErr2;

  S1 = SXY = SX = SY = SXX = 0.0;
  SsXX = SsYY = SsXY = Xm = Ym = 0.;
  difx = 0.;
  for (Int_t i = 0; i < nVal; i++) {
    //printf("BV LinFit %d  %f  %f  %f  \n", i, xVal[i], yVal[i], yErr[i]);
    invYErr2 = 1. / (yErr[i] * yErr[i]);
    S1 += invYErr2;
    SXY += xVal[i] * yVal[i] * invYErr2;
    SX += xVal[i] * invYErr2;
    SY += yVal[i] * invYErr2;
    SXX += xVal[i] * xVal[i] * invYErr2;
    if (i > 0)
      difx += TMath::Abs(xVal[i] - xVal[i - 1]);
    Xm += xVal[i];
    Ym += yVal[i];
    SsXX += xVal[i] * xVal[i];
    SsYY += yVal[i] * yVal[i];
    SsXY += xVal[i] * yVal[i];
  }
  delta = SXX * S1 - SX * SX;
  if (delta == 0.) {
    return kFALSE;
  }
  a = (SXY * S1 - SX * SY) / delta;
  b = (SY * SXX - SX * SXY) / delta;

  Ym /= (Double_t)nVal;
  Xm /= (Double_t)nVal;
  SsYY -= (Double_t)nVal * (Ym * Ym);
  SsXX -= (Double_t)nVal * (Xm * Xm);
  SsXY -= (Double_t)nVal * (Ym * Xm);
  Double_t eps = 1.E-24;
  if ((nVal > 2) && (TMath::Abs(difx) > eps) && ((SsYY - (SsXY * SsXY) / SsXX) > 0.)) {
    s = TMath::Sqrt((SsYY - (SsXY * SsXY) / SsXX) / (nVal - 2));
    be = s * TMath::Sqrt(1. / (Double_t)nVal + (Xm * Xm) / SsXX);
    ae = s / TMath::Sqrt(SsXX);
  } else {
    be = 0.;
    ae = 0.;
  }
  return kTRUE;
}

} // namespace mft
} // namespace o2
