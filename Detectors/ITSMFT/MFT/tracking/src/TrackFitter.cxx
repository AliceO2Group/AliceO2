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
    std::cout << "Seed covariances: \n"
              << track.getCovariances() << std::endl
              << std::endl;
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

  auto& mftTrackingParam = MFTTrackingParam::Instance();

  // initialize the starting track parameters and cluster
  double chi2invqptquad;
  double invQPtSeed;
  auto nPoints = track.getNPoints();
  auto k = TMath::Abs(o2::constants::math::B2C * mBZField);
  auto Hz = std::copysign(1, mBZField);
  //invQPtSeed = invQPtFromParabola(track, mBZField, chi2invqptquad);
  invQPtSeed = invQPtFromFCF(track, mBZField, chi2invqptquad);

  if (mftTrackingParam.verbose) {
    std::cout << "\n ***************************** Start Fitting new track ***************************** \n";
    std::cout << "N Clusters = " << nPoints << std::endl;
  }

  track.setInvQPtSeed(invQPtSeed);
  track.setChi2QPtSeed(chi2invqptquad);
  track.setInvQPt(invQPtSeed);

  /// Compute the initial track parameters
  /// The covariance matrix is computed such that the last cluster is the only constraint
  /// (by assigning an infinite dispersion to the other cluster)
  /// These parameters are the seed for the Kalman filter

  // compute track parameters
  int where; // First or last cluster?
  if (outward)
    where = 0;
  else
    where = nPoints - 1;

  double x0 = track.getXCoordinates()[where];
  double y0 = track.getYCoordinates()[where];
  double z0 = track.getZCoordinates()[where];
  double deltaX = track.getXCoordinates()[nPoints - 1] - track.getXCoordinates()[0];
  double deltaY = track.getYCoordinates()[nPoints - 1] - track.getYCoordinates()[0];
  double deltaZ = track.getZCoordinates()[nPoints - 1] - track.getZCoordinates()[0];
  double deltaR = TMath::Sqrt(deltaX * deltaX + deltaY * deltaY);
  double tanl = 0.5 * TMath::Sqrt(2) * (deltaZ / deltaR) *
                TMath::Sqrt(TMath::Sqrt((invQPtSeed * deltaR * k) * (invQPtSeed * deltaR * k) + 1) + 1);
  double phi0 = TMath::ATan2(y0, x0) + 0.5 * Hz * invQPtSeed * deltaZ * k / tanl;
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
        std::cout << " Init track with Seed A / B; sigmaboost = " << sigmaboost << (track.isCA() ? " CA Track " : " LTF Track") << std::endl;
      track.setInvQPt(1.0 / TMath::Sqrt(x0 * x0 + y0 * y0)); // Seeds A & B
      break;
    case DH:
      if (mftTrackingParam.verbose)
        std::cout << " Init track with Seed H; (k = " << seedH_k << "); sigmaboost = " << sigmaboost << (track.isCA() ? " CA Track " : " LTF Track") << std::endl;
      track.setInvQPt(track.getInvQPt() / seedH_k); // SeedH
      break;
    default:
      LOG(ERROR) << "Invalid MFT tracking seed";
      return false;
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

//_________________________________________________________________________________________________
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

//_________________________________________________________________________________________________
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
    xErr[np] = 5e-4; // FIXME -> errors from clusters
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
    qfcf = (ryRot > 0.) ? -1 : +1;

    Double_t alpha = 2.0 * std::abs(TMath::ATan2(rxRot, ryRot));
    Double_t x0 = xVal[0], y0 = yVal[0], z0 = zVal[0];
    Double_t dxyz2 = (x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0);
    Double_t cst = 1000.;
    Double_t c_alpha = cst * alpha;
    Double_t p, pt, pz;
    pt = 1. / invpt;
    p = std::sqrt(dxyz2) / c_alpha;
    pz = std::sqrt(p * p - pt * pt);

    invqpt_fcf = qfcf * invpt;
  } else { // the linear regression failed...
    printf("BV LinearRegression failed!\n");
    invqpt_fcf = 1. / 100.;
  }

  return invqpt_fcf;
}

////_________________________________________________________________________________________________
Bool_t LinearRegression(Int_t nVal, Double_t* xVal, Double_t* yVal, Double_t* yErr, Double_t& a, Double_t& ae, Double_t& b, Double_t& be)
{
  // linear regression y = a * x + b

  Double_t S1, SXY, SX, SY, SXX, SsXY, SsXX, SsYY, Xm, Ym, s, delta, difx;
  Double_t invYErr2;

  S1 = SXY = SX = SY = SXX = 0.0;
  SsXX = SsYY = SsXY = Xm = Ym = 0.;
  difx = 0.;
  for (Int_t i = 0; i < nVal; i++) {
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
