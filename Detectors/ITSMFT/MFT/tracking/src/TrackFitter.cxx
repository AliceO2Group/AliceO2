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

  /// Fit a track using its attached clusters
  /// Returns false in case of failure

  auto& mftTrackingParam = MFTTrackingParam::Instance();
  auto nClusters = track.getNumberOfPoints();

  if (mftTrackingParam.verbose) {
    std::cout << "Seed covariances: \n"
              << track.getCovariances() << std::endl
              << std::endl;
  }

  // recursively compute clusters, updating the track parameters
  if (!outward) { // Inward for vertexing
    nClusters--;
    while (nClusters-- > 0) {
      if (!computeCluster(track, nClusters)) {
        return false;
      }
    }
  } else { // Outward for MCH matching
    int ncl = 1;
    while (ncl < nClusters) {
      if (!computeCluster(track, ncl)) {
        return false;
      }
      ncl++;
    }
  }
  if (mftTrackingParam.verbose) {
    //  Print final covariances? std::cout << "Track covariances:"; track->getCovariances().Print();
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
	double sigmainvQPtsq;
  double chi2invqptquad;
  double invQPtSeed;
  auto nPoints = track.getNumberOfPoints();
  auto k = TMath::Abs(o2::constants::math::B2C * mBZField);
  auto Hz = std::copysign(1, mBZField);
  invQPtSeed = invQPtFromFCF(track, mBZField, sigmainvQPtsq);

  if (mftTrackingParam.verbose) {
    std::cout << "\n ***************************** Start Fitting new track ***************************** \n";
    std::cout << "N Clusters = " << nPoints << std::endl;
  }

  track.setInvQPtSeed(invQPtSeed);
  track.setChi2QPtSeed(chi2invqptquad);
  track.setInvQPt(invQPtSeed);

  /// Compute the initial track parameters to seed the Kalman filter

  int start; // Start fitting by the first or the last cluster
  if (outward)
    start = 0;
  else
    start = nPoints - 1;

  double x0 = track.getXCoordinates()[start];
  double y0 = track.getYCoordinates()[start];
  double z0 = track.getZCoordinates()[start];
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
  double sigmax0sq = track.getSigmasX2()[start];
  double sigmay0sq = track.getSigmasY2()[start];
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
    std::cout << " Variances: sigma_x0 = " << TMath::Sqrt(sigmax0sq) << " sigma_y0 = " << TMath::Sqrt(sigmay0sq) << " sigma_q/pt = " << TMath::Sqrt(sigmainvQPtsq) << std::endl;
  }

  auto model = (mftTrackingParam.trackmodel == Helix) ? "Helix" : (mftTrackingParam.trackmodel == Quadratic) ? "Quadratic" : "Linear";


  double C1 = track.getInvQPt() * k * deltaR;
  double D1 = TMath::Sqrt(C1 * C1 + 1);
  double invD1 = 1. / D1;
  double D2 = 1. / TMath::Sqrt(D1 + 1);
  double E1 = deltaR * D2 * D2 * invD1;
  double F1 = Hz * k * D2;
  double G1 = deltaZ * k * D2 / D1;
  double invDr2 = 1. / (deltaR * deltaR);
  double J1 = G1 * invDr2 / (k*deltaR);
  double L1 = C1 * C1 - 2 * D1 * (D1 + 1);
  double inv2rt2 = 0.25 * TMath::Sqrt(2);

  // compute the track parameter covariances at the last cluster (as if the other clusters did not exist)
  SMatrix55 lastParamCov;
  lastParamCov(0, 0) = sigmax0sq;                   // <X,X>
  lastParamCov(0, 1) = 0;                           // <Y,X>
  lastParamCov(0, 2) = sigmax0sq * invDr2 * (inv2rt2 * track.getInvQPt() * F1 * E1 * L1 * deltaX - deltaY);	// <PHI,X>
  lastParamCov(0, 3) = sigmax0sq * inv2rt2 * J1 * L1 * deltaX; 	// <TANL,X>
  lastParamCov(0, 4) = 0;    	// <INVQPT,X>

  lastParamCov(1, 1) = sigmay0sq;	// <Y,Y>
  lastParamCov(1, 2) = sigmay0sq * invDr2 * (inv2rt2 * track.getInvQPt() * F1 * E1 * L1 * deltaY - deltaX);       // <PHI,Y>
  lastParamCov(1, 3) = sigmay0sq * inv2rt2 * J1 * L1 * deltaY; 	// <TANL,Y>
  lastParamCov(1, 4) = 0;       // <INVQPT,Y>

  lastParamCov(2, 2) = sigmainvQPtsq * 0.125 * E1 * E1 * F1 * F1 * L1 * L1; // <PHI,PHI>

  lastParamCov(2, 3) = sigmainvQPtsq * 0.125 * C1 * E1 * F1 * G1 * L1;      //  <TANL,PHI>

  lastParamCov(2, 4) = sigmainvQPtsq * inv2rt2 * E1 * F1 * L1;           //  <INVQPT,PHI>

  lastParamCov(3, 3) = sigmainvQPtsq * 0.125 * C1 * C1 * G1 * G1; // <TANL,TANL>

  lastParamCov(3, 4) = sigmainvQPtsq * inv2rt2 * C1 * G1;                                // <INVQPT,TANL>

  lastParamCov(4, 4) = sigmainvQPtsq; // <INVQPT,INVQPT>
  track.setCovariances(lastParamCov);
  track.setTrackChi2(0.);

  return true;
}

//_________________________________________________________________________________________________
bool TrackFitter::computeCluster(TrackLTF& track, int cluster)
{
  /// Propagate track to the z position of the new cluster
  /// accounting for MCS dispersion in the current layer and the other(s) crossed
  /// Recompute the parameters adding the cluster constraint with the Kalman filter
  /// Returns false in case of failure

  auto& mftTrackingParam = MFTTrackingParam::Instance();
  const auto& clx = track.getXCoordinates()[cluster];
  const auto& cly = track.getYCoordinates()[cluster];
  const auto& clz = track.getZCoordinates()[cluster];
  const auto& sigmaX2 = track.getSigmasX2()[cluster];
  const auto& sigmaY2 = track.getSigmasY2()[cluster];

  if (track.getZ() == clz) {
    LOG(INFO) << "AddCluster ERROR: The new cluster must be upstream! Bug on TrackFinder. " << (track.isCA() ? " CATrack" : "LTFTrack");
    LOG(INFO) << "track.getZ() = " << track.getZ() << " ; newClusterZ = " << clz << " ==> Skipping point.";
    return true;
  }
  if (mftTrackingParam.verbose)
    std::cout << "computeCluster:     X = " << clx << " Y = " << cly << " Z = " << clz << " nCluster = " << cluster << std::endl;

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

  // recompute parameters
  const std::array<float, 2>& pos = {clx, cly};
  const std::array<float, 2>& cov = {sigmaX2, sigmaY2};

  if (track.update(pos, cov)) {
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
Double_t invQPtFromFCF(const TrackLTF& track, Double_t bFieldZ, Double_t& sigmainvqptsq)
{

  const std::array<Float_t, constants::mft::LayersNumber>& xPositions = track.getXCoordinates();
  const std::array<Float_t, constants::mft::LayersNumber>& yPositions = track.getYCoordinates();
  const std::array<Float_t, constants::mft::LayersNumber>& zPositions = track.getZCoordinates();
  const std::array<Float_t, constants::mft::LayersNumber>& SigmasX2 = track.getSigmasX2();
  const std::array<Float_t, constants::mft::LayersNumber>& SigmasY2 = track.getSigmasY2();

  // Fast Circle Fit (Hansroul, Jeremie, Savard, 1987)
  auto nPoints = track.getNumberOfPoints();
  Double_t* xVal = new Double_t[nPoints];
  Double_t* yVal = new Double_t[nPoints];
  Double_t* zVal = new Double_t[nPoints];
  Double_t* xErr = new Double_t[nPoints];
  Double_t* yErr = new Double_t[nPoints];
  Double_t* uVal = new Double_t[nPoints - 1];
  Double_t* vVal = new Double_t[nPoints - 1];
  Double_t* vErr = new Double_t[nPoints - 1];
  Double_t* fweight = new Double_t[nPoints - 1];
  Double_t* Rn = new Double_t[nPoints - 1];
  Double_t* Pn = new Double_t[nPoints - 1];
  Double_t A, Aerr, B, Berr, x2, y2, invx2y2, a, b, r, sigmaRsq, u2, sigma;
  Double_t F0,F1,F2,F3,F4, SumSRn, SumSPn, SumRn, SumUPn, SumRP;

	SumSRn = SumSPn = SumRn = SumUPn = SumRP = 0.0;
  F0 = F1 = F2 = F3 = F4 = 0.0;

  for (auto np = 0; np < nPoints; np++) {
    xErr[np] = SigmasX2[np];
    yErr[np] = SigmasY2[np];
    if (np > 0) {
      xVal[np] = xPositions[np] - xVal[0];
      yVal[np] = yPositions[np] - yVal[0];
      xErr[np] *= std::sqrt(2.);
      yErr[np] *= std::sqrt(2.);
    } else {
      xVal[np] = 0.;
      yVal[np] = 0.;
    }
    zVal[np] = zPositions[np];
  }
	for (int i = 0; i < (nPoints - 1); i++) {
    x2 = xVal[i + 1] * xVal[i + 1];
    y2 = yVal[i + 1] * yVal[i + 1];
    invx2y2 = 1. / (x2 + y2);
    uVal[i] = xVal[i + 1] * invx2y2;
    vVal[i] = yVal[i + 1] * invx2y2;
    vErr[i] = std::sqrt(8. * xErr[i + 1] * xErr[i + 1] * x2 * y2 + 2. * yErr[i + 1] * yErr[i + 1] * (x2 - y2) * (x2 - y2)) * invx2y2 * invx2y2;
    u2 = uVal[i] * uVal[i];
    fweight[i] = 1;//. / vErr[i];
    F0 += fweight[i];   // f = fn(Hansroul) que é o peso de cada ponto Vn...inverso da incerteza? 
    F1 += fweight[i] * uVal[i];
    F2 += fweight[i] * u2;
    F3 += fweight[i] * uVal[i] * u2;
    F4 += fweight[i] * u2 * u2;
  }

	double Rn_det1 = F2 * F4 - F3 * F3; 
  double Rn_det2 = F1 * F4 - F2 * F3;
  double Rn_det3 = F1 * F3 - F2 * F2;
  double Pn_det1 = Rn_det2;
  double Pn_det2 = F0 * F4 - F2 * F2;
  double Pn_det3 = F0 * F3 - F1 * F2;

	for (int j = 0; j < (nPoints - 1); j++) {
    Rn[j] = fweight[j] * (Rn_det1 - uVal[j]*Rn_det2 + uVal[j]*uVal[j]*Rn_det3);
    SumSRn += Rn[j] * Rn[j] * vErr[j];
    SumRn += Rn[j];

    Pn[j] = fweight[j] * (-Pn_det1 + uVal[j]*Pn_det2 - uVal[j]*uVal[j]*Pn_det3);
    SumSPn += Pn[j] * Pn[j] * vErr[j];
    SumUPn += uVal[j] * Pn[j];
  
    SumRP += Rn[j] * Pn[j] * vErr[j] * vErr[j]; //falta um vErr?
  }
 
  Double_t invqpt_fcf;
  Int_t qfcf;
//  chi2 = 0.;
  if (LinearRegression((nPoints - 1), uVal, vVal, vErr, B, Berr, A, Aerr)) {
    // v = a * u + b
    // circle passing through (0,0):
    // (x - rx)^2 + (y - ry)^2 = r^2
    // ---> a = - rx / ry;
    // ---> b = 1 / (2 * ry)
    b =  1. / (2. * A);
    a = -B * b;
    r = std::sqrt(a * a + b * b);
    double_t invR = 1./r;

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
    Double_t rxRot = a * cosSlope + b * sinSlope;
    Double_t ryRot = a * sinSlope - b * cosSlope;
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

    //error calculations:
    double invA2 = 1. / (A * A);

    double sigmaAsq = SumSRn / (SumRn * SumRn);
    double sigmaBsq = SumSPn / (SumUPn * SumUPn);
    double sigmaAB = SumRP / (SumRn * SumUPn);

		double sigmaasq_FCF = TMath::Abs (0.25 * invA2 * invA2 * (B * B * sigmaAsq + A * A * sigmaBsq - A * B * sigmaAB));
    double sigmabsq_FCF = TMath::Abs (0.25 * invA2 * invA2 * sigmaAsq);
    double sigma2R = invR * invR * (b * b * sigmaasq_FCF + a * a * sigmabsq_FCF + 2 * a * b * TMath::Sqrt(sigmaasq_FCF) * TMath::Sqrt(sigmabsq_FCF));

		sigmainvqptsq = sigma2R * invpt * invpt * invR * invR;

		std::cout << " Sigma^2_A " << sigmaAsq <<" Sigma^2_B " << sigmaBsq <<" Sigma_AB " << sigmaAB << std::endl;
		std::cout << " Sigma^2_a " << sigmaasq_FCF <<" Sigma^2_b " << sigmabsq_FCF <<" Sigma^2_R " << sigma2R << std::endl;
		

  } else { // the linear regression failed...
    LOG(WARN) << "LinearRegression failed!";
    invqpt_fcf = 1. / 100.;
  }

  return invqpt_fcf;
}

////_________________________________________________________________________________________________
Bool_t LinearRegression(Int_t nVal, Double_t* xVal, Double_t* yVal, Double_t* yErr, Double_t& B, Double_t& Berr, Double_t& A, Double_t& Aerr)
{
  // linear regression y = B * x + A

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
  B = (SXY * S1 - SX * SY) / delta;
  A = (SY * SXX - SX * SXY) / delta;

  Ym /= (Double_t)nVal;
  Xm /= (Double_t)nVal;
  SsYY -= (Double_t)nVal * (Ym * Ym);
  SsXX -= (Double_t)nVal * (Xm * Xm);
  SsXY -= (Double_t)nVal * (Ym * Xm);
  Double_t eps = 1.E-24;
  if ((nVal > 2) && (TMath::Abs(difx) > eps) && ((SsYY - (SsXY * SsXY) / SsXX) > 0.)) {
    s = TMath::Sqrt((SsYY - (SsXY * SsXY) / SsXX) / (nVal - 2));
    Aerr = s * TMath::Sqrt(1. / (Double_t)nVal + (Xm * Xm) / SsXX);
    Berr = s / TMath::Sqrt(SsXX);
  } else {
    Aerr = 0.;
    Berr = 0.;
  }
  return kTRUE;
}


} // namespace mft
} // namespace o2
