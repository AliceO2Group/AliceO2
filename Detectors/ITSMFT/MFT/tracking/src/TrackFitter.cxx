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
template <typename T>
bool TrackFitter<T>::fit(T& track, bool outward)
{
  /// Fit a track using its attached clusters
  /// Returns false in case of failure

  auto nClusters = track.getNumberOfPoints();
  auto lastLayer = track.getLayers()[(outward ? 0 : nClusters - 1)];

  if (mVerbose) {
    std::cout << "Seed covariances: \n"
              << track.getCovariances() << std::endl
              << std::endl;
  }

  // recursively compute clusters, updating the track parameters
  if (!outward) { // Inward for vertexing
    while (nClusters-- > 0) {
      if (!computeCluster(track, nClusters, lastLayer)) {
        return false;
      }
    }
  } else { // Outward for MCH matching
    int ncl = 0;
    while (ncl < nClusters) {
      if (!computeCluster(track, ncl, lastLayer)) {
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
template <typename T>
bool TrackFitter<T>::initTrack(T& track, bool outward)
{

  if constexpr (std::is_same<T, o2::mft::TrackLTF>::value) { // Magnet on
    // initialize the starting track parameters
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
    int first_cls, next_cls;
    if (outward) { // MCH matching
      first_cls = 0;
      next_cls = nPoints - 1;
    } else { // Vertexing
      first_cls = nPoints - 1;
      next_cls = 0;
    }

    auto x0 = track.getXCoordinates()[first_cls];
    auto y0 = track.getYCoordinates()[first_cls];
    auto z0 = track.getZCoordinates()[first_cls];

    // Compute tanl using first two clusters
    auto deltaX = track.getXCoordinates()[1] - track.getXCoordinates()[0];
    auto deltaY = track.getYCoordinates()[1] - track.getYCoordinates()[0];
    auto deltaZ = track.getZCoordinates()[1] - track.getZCoordinates()[0];
    auto deltaR = TMath::Sqrt(deltaX * deltaX + deltaY * deltaY);
    auto tanl0 = -std::abs(deltaZ / deltaR);

    // Compute phi at the last cluster using two last clusters
    deltaX = track.getXCoordinates()[first_cls] - track.getXCoordinates()[next_cls];
    deltaY = track.getYCoordinates()[first_cls] - track.getYCoordinates()[next_cls];
    deltaZ = track.getZCoordinates()[first_cls] - track.getZCoordinates()[next_cls];
    deltaR = TMath::Sqrt(deltaX * deltaX + deltaY * deltaY);
    double phi0;
    if (outward) {
      phi0 = TMath::ATan2(-deltaY, -deltaX) - 0.5 * Hz * invQPt0 * deltaZ * k / tanl0;
    } else {
      phi0 = TMath::ATan2(deltaY, deltaX) - 0.5 * Hz * invQPt0 * deltaZ * k / tanl0;
    }
    // The low momentum phi0 correction may be irrelevant and may require a call to o2::math_utils::bringToPMPiGend(phi0);

    track.setX(x0);
    track.setY(y0);
    track.setZ(z0);
    track.setPhi(phi0);
    track.setTanl(tanl0);

    if (mVerbose) {
      std::cout << " Init " << (track.isCA() ? "CA Track " : "LTF Track") << (outward ? " (outward)" : " (inward)") << std::endl;
      auto model = (mTrackModel == Helix) ? "Helix" : (mTrackModel == Quadratic) ? "Quadratic"
                                                    : (mTrackModel == Optimized) ? "Optimized"
                                                                                 : "Linear";
      std::cout << "Track Model: " << model << std::endl;
      std::cout << "  initTrack: X = " << x0 << " Y = " << y0 << " Z = " << z0 << " Tgl = " << tanl0 << "  Phi = " << phi0 << " pz = " << track.getPz() << " q/pt = " << track.getInvQPt() << std::endl;
    }

    SMatrix55Sym lastParamCov;
    double qptsigma = TMath::Range(1., 10., std::abs(track.getInvQPt()));

    lastParamCov(0, 0) = 1.;       // <X,X>
    lastParamCov(1, 1) = 1.;       // <Y,Y>
    lastParamCov(2, 2) = 1.;       // <PHI,PHI>
    lastParamCov(3, 3) = 1.;       // <TANL,TANL>
    lastParamCov(4, 4) = qptsigma; // <INVQPT,INVQPT>

    track.setCovariances(lastParamCov);
    track.setTrackChi2(0.);

  } else { // Magnet off: linear tracks

    // initialize the starting track parameters
    double chi2invqptquad;
    auto invQPt0 = 0.;

    auto nPoints = track.getNumberOfPoints();

    if (mVerbose) {
      std::cout << "\n ***************************** Start Fitting new track ***************************** \n";
      std::cout << "N Clusters = " << nPoints << std::endl;
    }

    track.setInvQPtSeed(0);
    track.setChi2QPtSeed(0);
    track.setInvQPt(invQPt0);

    /// Compute the initial track parameters to seed the Kalman filter
    int first_cls, next_cls;
    if (outward) { // MCH matching
      first_cls = 0;
      next_cls = nPoints - 1;
    } else { // Vertexing
      first_cls = nPoints - 1;
      next_cls = 0;
    }

    auto x0 = track.getXCoordinates()[first_cls];
    auto y0 = track.getYCoordinates()[first_cls];
    auto z0 = track.getZCoordinates()[first_cls];

    auto deltaX = track.getXCoordinates()[first_cls] - track.getXCoordinates()[next_cls];
    auto deltaY = track.getYCoordinates()[first_cls] - track.getYCoordinates()[next_cls];
    auto deltaZ = track.getZCoordinates()[first_cls] - track.getZCoordinates()[next_cls];
    auto deltaR = TMath::Sqrt(deltaX * deltaX + deltaY * deltaY);
    auto tanl0 = -std::abs(deltaZ) / deltaR;
    double phi0;
    if (outward) {
      phi0 = TMath::ATan2(-deltaY, -deltaX);
    } else {
      phi0 = TMath::ATan2(deltaY, deltaX);
    }

    track.setX(x0);
    track.setY(y0);
    track.setZ(z0);
    track.setPhi(phi0);
    track.setTanl(tanl0);

    if (mVerbose) {
      std::cout << " Init " << (track.isCA() ? "CA Track " : "LTF Track") << (outward ? " (outward)" : " (inward)") << std::endl;
      auto model = "Linear";
      std::cout << "Track Model: " << model << std::endl;
      std::cout << "  initTrack: X = " << x0 << " Y = " << y0 << " Z = " << z0 << " Tgl = " << tanl0 << "  Phi = " << phi0 << " pz = " << track.getPz() << " q/pt = " << track.getInvQPt() << std::endl;
    }

    SMatrix55Sym lastParamCov;
    double qptsigma = TMath::Range(1., 10., std::abs(track.getInvQPt()));

    lastParamCov(0, 0) = 1.; // <X,X>
    lastParamCov(1, 1) = 1.; // <Y,Y>
    lastParamCov(2, 2) = 1.; // <PHI,PHI>
    lastParamCov(3, 3) = 1.; // <TANL,TANL>
    lastParamCov(4, 4) = 0.; // <INVQPT,INVQPT>

    track.setCovariances(lastParamCov);
    track.setTrackChi2(0.);
  }

  return true;
}

//_________________________________________________________________________________________________
template <typename T>
bool TrackFitter<T>::propagateToZ(T& track, double z)
{
  // Propagate track to the z position of the new cluster
  if constexpr (std::is_same<T, o2::mft::TrackLTF>::value) { // Magnet on

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
  } else {
    track.propagateToZlinear(z);
  }
  return true;
}

//_________________________________________________________________________________________________
template <typename T>
bool TrackFitter<T>::propagateToNextClusterWithMCS(T& track, double z, int& startingLayerID, const int& newLayerID)
{

  // Propagate track to the next cluster z position, adding angular MCS effects at the center of
  // each disk crossed by the track. This method is valid only for track propagation between
  // clusters at MFT layers positions.

  if (startingLayerID == newLayerID) { // Same layer, nothing to do.
    if (mVerbose) {
      std::cout << " => Propagate to next cluster with MCS : startingLayerID = " << startingLayerID << " = > "
                << " newLayerID = " << newLayerID << " (NLayers = " << std::abs(newLayerID - startingLayerID);
      std::cout << ") ; track.getZ() = " << track.getZ() << " => ";
      std::cout << "destination cluster z = " << z << " ; => Same layer: no MCS effects." << std::endl;
    }
    if (z != track.getZ()) {
      propagateToZ(track, z);
    }
    return true;
  }

  using o2::mft::constants::LayerZPosition;
  auto startingZ = track.getZ();

  // https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
  auto signum = [](auto a) {
    return (0 < a) - (a < 0);
  };
  int direction = signum(newLayerID - startingLayerID); // takes values +1, 0, -1
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
  startingLayerID = newLayerID;
  return true;
}

//_________________________________________________________________________________________________
template <typename T>
bool TrackFitter<T>::computeCluster(T& track, int cluster, int& startingLayerID)
{
  /// Propagate track to the z position of the new cluster
  /// accounting for MCS dispersion in the current layer and the other(s) crossed
  /// Recompute the parameters adding the cluster constraint with the Kalman filter
  /// Returns false in case of failure

  const auto& clx = track.getXCoordinates()[cluster];
  const auto& cly = track.getYCoordinates()[cluster];
  const auto& clz = track.getZCoordinates()[cluster];
  const auto& sigmaX2 = track.getSigmasX2()[cluster] + mAlignResidual;
  const auto& sigmaY2 = track.getSigmasY2()[cluster] + mAlignResidual;
  const auto& newLayerID = track.getLayers()[cluster];

  if (mVerbose) {
    std::cout << "computeCluster:     X = " << clx << " Y = " << cly << " Z = " << clz << " nCluster = " << cluster << " Layer = " << newLayerID << std::endl;
  }

  if (!propagateToNextClusterWithMCS(track, clz, startingLayerID, newLayerID)) {
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
      std::cout << "   New Cluster: X = " << clx << " Y = " << cly << " Z = " << clz << " sigmaX2 = " << sigmaX2 << " sigmaY2 = " << sigmaY2 << std::endl;
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
template <typename T>
Double_t invQPtFromFCF(const T& track, Double_t bFieldZ, Double_t& sigmainvqptsq)
{

  const std::array<Float_t, constants::mft::LayersNumber>& xPositions = track.getXCoordinates();
  const std::array<Float_t, constants::mft::LayersNumber>& yPositions = track.getYCoordinates();
  const std::array<Float_t, constants::mft::LayersNumber>& zPositions = track.getZCoordinates();
  const std::array<Float_t, constants::mft::LayersNumber>& SigmasX2 = track.getSigmasX2();
  const std::array<Float_t, constants::mft::LayersNumber>& SigmasY2 = track.getSigmasY2();

  // Fast Circle Fit (Hansroul, Jeremie, Savard, 1987)
  auto nPoints = track.getNumberOfPoints();
  std::vector<double> xVal(nPoints);
  std::vector<double> yVal(nPoints);
  std::vector<double> zVal(nPoints);
  std::vector<double> xErr(nPoints);
  std::vector<double> yErr(nPoints);
  std::vector<double> uVal(nPoints);
  std::vector<double> vVal(nPoints);
  std::vector<double> vErr(nPoints);
  std::vector<double> fweight(nPoints);
  std::vector<double> Rn(nPoints);
  std::vector<double> Pn(nPoints);
  Double_t A, Aerr, B, Berr, x2, y2, invx2y2, a, b, r, sigmaRsq, u2, sigma;
  Double_t F0, F1, F2, F3, F4, SumSRn, SumSPn, SumRn, SumUPn, SumRP;

  SumSRn = SumSPn = SumRn = SumUPn = SumRP = 0.0;
  F0 = F1 = F2 = F3 = F4 = 0.0;

  for (auto np = 0; np < nPoints; np++) {
    xErr[np] = SigmasX2[np];
    yErr[np] = SigmasY2[np];
    if (np > 0) {
      xVal[np] = xPositions[np] - xPositions[0] + xVal[0];
      yVal[np] = yPositions[np] - yPositions[0] + yVal[0];
    } else {
      xVal[np] = .00001;
      yVal[np] = 0.;
    }
    zVal[np] = zPositions[np];
  }
  for (int i = 0; i < nPoints; i++) {
    x2 = xVal[i] * xVal[i];
    y2 = yVal[i] * yVal[i];
    invx2y2 = 1. / (x2 + y2);
    uVal[i] = xVal[i] * invx2y2;
    vVal[i] = yVal[i] * invx2y2;
    vErr[i] = std::sqrt(8. * xErr[i] * xErr[i] * x2 * y2 + 2. * yErr[i] * yErr[i] * (x2 - y2) * (x2 - y2)) * invx2y2 * invx2y2;
  }

  Double_t invqpt_fcf;
  Int_t qfcf;
  //  chi2 = 0.;
  if (LinearRegression(nPoints, uVal, vVal, vErr, B, Berr, A, Aerr)) {
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

    invqpt_fcf = qfcf * invpt;
  } else { // the linear regression failed...
    LOG(warn) << "LinearRegression failed!";
    invqpt_fcf = 1. / 100.;
  }

  return invqpt_fcf;
}

////_________________________________________________________________________________________________
Bool_t LinearRegression(Int_t nVal, std::vector<double>& xVal, std::vector<double>& yVal, std::vector<double>& yErr, Double_t& B, Double_t& Berr, Double_t& A, Double_t& Aerr)
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

template class TrackFitter<o2::mft::TrackLTF>;
template class TrackFitter<o2::mft::TrackLTFL>;

} // namespace mft
} // namespace o2
