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
#include "MathUtils/fit.h"
#include "MathUtils/Utils.h"

using o2::math_utils::fitGaus;

namespace o2
{
namespace mft
{

//_________________________________________________________________________________________________
bool TrackFitter::fit(TrackLTF& track, bool outward)
{
  /// Fit a track using its attached clusters
  /// Returns false in case of failure

  auto nClusters = track.getNumberOfPoints();

  if (mVerbose) {
    std::cout << "Seed covariances: \n"
              << track.getCovariances() << std::endl
              << std::endl;
  }

  // recursively compute clusters, updating the track parameters
  if (!outward) { // Inward for vertexing
    while (nClusters-- > 0) {
      if (!computeCluster(track, nClusters)) {
        return false;
      }
    }
  } else { // Outward for MCH matching
    int ncl = 0;
    while (ncl < nClusters) {
      if (!computeCluster(track, ncl)) {
        return false;
      }
      ncl++;
    }
  }
  if (mVerbose) {
    std::cout << "Track Chi2 = " << track.getTrackChi2() << std::endl;
    std::cout << " ***************************** Done fitting *****************************\n";
  }

  return true;
}

//_________________________________________________________________________________________________
bool TrackFitter::initTrack(TrackLTF& track, bool outward)
{

  // initialize the starting track parameters and cluster
  double sigmainvQPtsq;
  double chi2invqptquad;
  auto invQPt0 = invQPtFromFCF(track, mBZField, sigmainvQPtsq);
  auto nPoints = track.getNumberOfPoints();
  auto k = TMath::Abs(o2::constants::math::B2C * mBZField);
  auto Hz = std::copysign(1, mBZField);

  if (mVerbose) {
    std::cout << "\n ***************************** Start Fitting new track ***************************** \n";
    std::cout << "N Clusters = " << nPoints << std::endl;
  }

  track.setInvQPtSeed(invQPt0);
  track.setChi2QPtSeed(chi2invqptquad);
  track.setInvQPt(invQPt0);

  /// Compute the initial track parameters to seed the Kalman filter
  int first_cls, last_cls;
  if (outward) { // MCH matching
    first_cls = 1;
    last_cls = 0;
  } else { // Vertexing
    first_cls = nPoints - 1;
    last_cls = nPoints - 2;
  }

  auto x0 = track.getXCoordinates()[first_cls];
  auto y0 = track.getYCoordinates()[first_cls];
  auto z0 = track.getZCoordinates()[first_cls];

  //Compute tanl using first two clusters
  auto deltaX = track.getXCoordinates()[1] - track.getXCoordinates()[0];
  auto deltaY = track.getYCoordinates()[1] - track.getYCoordinates()[0];
  auto deltaZ = track.getZCoordinates()[1] - track.getZCoordinates()[0];
  auto deltaR = TMath::Sqrt(deltaX * deltaX + deltaY * deltaY);
  auto tanl0 = 0.5 * TMath::Sqrt2() * (deltaZ / deltaR) *
               TMath::Sqrt(TMath::Sqrt((invQPt0 * deltaR * k) * (invQPt0 * deltaR * k) + 1) + 1);

  // Compute phi at the last cluster using two last clusters
  deltaX = track.getXCoordinates()[first_cls] - track.getXCoordinates()[last_cls];
  deltaY = track.getYCoordinates()[first_cls] - track.getYCoordinates()[last_cls];
  deltaZ = track.getZCoordinates()[first_cls] - track.getZCoordinates()[last_cls];
  deltaR = TMath::Sqrt(deltaX * deltaX + deltaY * deltaY);
  auto phi0 = TMath::ATan2(deltaY, deltaX) - 0.5 * Hz * invQPt0 * deltaZ * k / tanl0;
  auto sigmax0sq = track.getSigmasX2()[first_cls];
  auto sigmay0sq = track.getSigmasY2()[first_cls];
  auto sigmax1sq = track.getSigmasX2()[last_cls];
  auto sigmay1sq = track.getSigmasY2()[last_cls];
  auto sigmaDeltaXsq = sigmax0sq + sigmax1sq;
  auto sigmaDeltaYsq = sigmay0sq + sigmay1sq;

  track.setX(x0);
  track.setY(y0);
  track.setZ(z0);
  track.setPhi(phi0);
  track.setTanl(tanl0);

  if (mVerbose) {
    std::cout << " Init " << (track.isCA() ? "CA Track " : "LTF Track") << std::endl;
    auto model = (mTrackModel == Helix) ? "Helix" : (mTrackModel == Quadratic) ? "Quadratic"
                                                  : (mTrackModel == Optimized) ? "Optimized"
                                                                               : "Linear";
    std::cout << "Track Model: " << model << std::endl;
    std::cout << "  initTrack: X = " << x0 << " Y = " << y0 << " Z = " << z0 << " Tgl = " << tanl0 << "  Phi = " << phi0 << " pz = " << track.getPz() << " q/pt = " << track.getInvQPt() << std::endl;
  }

  SMatrix55Sym lastParamCov;
  float qptsigma = TMath::Max(std::abs(track.getInvQPt()), .5);
  float tanlsigma = TMath::Max(std::abs(track.getTanl()), .5);

  lastParamCov(0, 0) = 1;                              // <X,X>
  lastParamCov(1, 1) = 1;                              // <Y,X>
  lastParamCov(2, 2) = TMath::Pi() * TMath::Pi() / 16; // <PHI,X>
  lastParamCov(3, 3) = 10 * tanlsigma * tanlsigma;     // <TANL,X>
  lastParamCov(4, 4) = 10 * qptsigma * qptsigma;       // <INVQPT,X>

  track.setCovariances(lastParamCov);
  track.setTrackChi2(0.);

  return true;
}

//_________________________________________________________________________________________________
bool TrackFitter::propagateToZ(TrackLTF& track, double z)
{
  // Propagate track to the z position of the new cluster
  switch (mTrackModel) {
    case Linear:
      track.propagateToZlinear(z);
      break;
    case Quadratic:
      track.propagateToZquadratic(z, mBZField);
      break;
    case Helix:
      track.propagateToZhelix(z, mBZField);
      break;
    case Optimized:
      track.propagateToZ(z, mBZField);
      break;
    default:
      std::cout << " Invalid track model.\n";
      return false;
      break;
  }
  return true;
}

//_________________________________________________________________________________________________
bool TrackFitter::propagateToNextClusterWithMCS(TrackLTF& track, double z)
{

  // Propagate track to the next cluster z position, adding angular MCS effects at the center of
  // each disk crossed by the track

  using o2::mft::constants::LayerZPosition;
  int startingLayerID, newLayerID;
  auto startingZ = track.getZ();

  //LayerID of each cluster from ZPosition // TODO: Use ChipMapping
  for (auto layer = 10; layer--;) {
    if (startingZ<LayerZPosition[layer] + .3 & startingZ> LayerZPosition[layer] - .3) {
      startingLayerID = layer;
    }
  }
  for (auto layer = 10; layer--;) {
    if (z<LayerZPosition[layer] + .3 & z> LayerZPosition[layer] - .3) {
      newLayerID = layer;
    }
  }

  int direction = (newLayerID - startingLayerID) / std::abs(newLayerID - startingLayerID);
  auto currentLayer = startingLayerID;

  if (mVerbose) {
    std::cout << " => Propagate to next cluster with MCS : startingLayerID = " << startingLayerID << " = > "
              << " newLayerID = " << newLayerID << " (NLayers = " << std::abs(newLayerID - startingLayerID);
    std::cout << ") ; track.getZ() = " << track.getZ() << " => ";
    std::cout << "destination cluster z = " << z << " ; " << std::endl;
  }

  // Number of disks crossed by this track segment
  while (currentLayer != newLayerID) {
    auto nextlayer = currentLayer + direction;
    auto nextZ = LayerZPosition[nextlayer];

    int NDisksMS;
    if (nextZ - track.getZ() > 0) {
      NDisksMS = (currentLayer % 2 == 0) ? (currentLayer - nextlayer) / 2 : (currentLayer - nextlayer + 1) / 2;
    } else {
      NDisksMS = (currentLayer % 2 == 0) ? (nextlayer - currentLayer + 1) / 2 : (nextlayer - currentLayer) / 2;
    }

    if (mVerbose) {
      std::cout << "currentLayer = " << currentLayer << " ; "
                << "nextlayer = " << nextlayer << " ; ";
      std::cout << "track.getZ() = " << track.getZ() << " ; ";
      std::cout << "nextZ = " << nextZ << " ; ";
      std::cout << "NDisksMS = " << NDisksMS << std::endl;
    }

    if ((NDisksMS * mMFTDiskThicknessInX0) != 0) {
      track.addMCSEffect(NDisksMS * mMFTDiskThicknessInX0);
      if (mVerbose) {
        std::cout << "Track covariances after MCS effects: \n"
                  << track.getCovariances() << std::endl
                  << std::endl;
      }
    }

    if (mVerbose) {
      std::cout << "  BeforeExtrap: X = " << track.getX() << " Y = " << track.getY() << " Z = " << track.getZ() << " Tgl = " << track.getTanl() << "  Phi = " << track.getPhi() << " pz = " << track.getPz() << " q/pt = " << track.getInvQPt() << std::endl;
    }

    propagateToZ(track, nextZ);

    currentLayer = nextlayer;
  }
  if (z != track.getZ()) {
    propagateToZ(track, z);
  }
  return true;
}

//_________________________________________________________________________________________________
bool TrackFitter::computeCluster(TrackLTF& track, int cluster)
{
  /// Propagate track to the z position of the new cluster
  /// accounting for MCS dispersion in the current layer and the other(s) crossed
  /// Recompute the parameters adding the cluster constraint with the Kalman filter
  /// Returns false in case of failure

  const auto& clx = track.getXCoordinates()[cluster];
  const auto& cly = track.getYCoordinates()[cluster];
  const auto& clz = track.getZCoordinates()[cluster];
  const auto& sigmaX2 = track.getSigmasX2()[cluster];
  const auto& sigmaY2 = track.getSigmasY2()[cluster];

  if (mVerbose) {
    std::cout << "computeCluster:     X = " << clx << " Y = " << cly << " Z = " << clz << " nCluster = " << cluster << std::endl;
  }

  if (!propagateToNextClusterWithMCS(track, clz)) {
    return false;
  }

  if (mVerbose) {
    std::cout << "   AfterExtrap: X = " << track.getX() << " Y = " << track.getY() << " Z = " << track.getZ() << " Tgl = " << track.getTanl() << "  Phi = " << track.getPhi() << " pz = " << track.getPz() << " q/pt = " << track.getInvQPt() << std::endl;
    std::cout << "Track covariances after extrap: \n"
              << track.getCovariances() << std::endl
              << std::endl;
  }

  // recompute parameters
  const std::array<float, 2>& pos = {clx, cly};
  const std::array<float, 2>& cov = {sigmaX2, sigmaY2};

  if (track.update(pos, cov)) {
    if (mVerbose) {
      std::cout << "   New Cluster: X = " << clx << " Y = " << cly << " Z = " << clz << std::endl;
      std::cout << "   AfterKalman: X = " << track.getX() << " Y = " << track.getY() << " Z = " << track.getZ() << " Tgl = " << track.getTanl() << "  Phi = " << track.getPhi() << " pz = " << track.getPz() << " q/pt = " << track.getInvQPt() << std::endl;
      std::cout << std::endl;
      std::cout << "Track covariances after Kalman update: \n"
                << track.getCovariances() << std::endl
                << std::endl;
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
  Double_t F0, F1, F2, F3, F4, SumSRn, SumSPn, SumRn, SumUPn, SumRP;

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
    fweight[i] = 1. / vErr[i];
    F0 += fweight[i];
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
    Rn[j] = fweight[j] * (Rn_det1 - uVal[j] * Rn_det2 + uVal[j] * uVal[j] * Rn_det3);
    SumSRn += Rn[j] * Rn[j] * vErr[j] * vErr[j];
    SumRn += Rn[j];

    Pn[j] = fweight[j] * (-Pn_det1 + uVal[j] * Pn_det2 - uVal[j] * uVal[j] * Pn_det3);
    SumSPn += Pn[j] * Pn[j] * vErr[j] * vErr[j];
    SumUPn += uVal[j] * Pn[j];

    SumRP += Rn[j] * Pn[j] * vErr[j] * vErr[j] * vErr[j];
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
    b = 1. / (2. * A);
    a = -B * b;
    r = std::sqrt(a * a + b * b);
    double_t invR = 1. / r;

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

    double sigmaasq_FCF = TMath::Abs(0.25 * invA2 * invA2 * (B * B * sigmaAsq + A * A * sigmaBsq - A * B * sigmaAB));
    double sigmabsq_FCF = TMath::Abs(0.25 * invA2 * invA2 * sigmaAsq);
    double sigma2R = invR * invR * (b * b * sigmaasq_FCF + a * a * sigmabsq_FCF + 2 * a * b * TMath::Sqrt(sigmaasq_FCF) * TMath::Sqrt(sigmabsq_FCF));

    sigmainvqptsq = sigma2R * invpt * invpt * invR * invR;

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
    if (i > 0) {
      difx += TMath::Abs(xVal[i] - xVal[i - 1]);
    }
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
