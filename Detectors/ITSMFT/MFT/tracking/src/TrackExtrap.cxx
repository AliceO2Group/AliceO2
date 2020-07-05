// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackExtrap.cxx
/// \brief Implementation of tools for track extrapolation
///
/// \author Philippe Pillot, Subatech; adapted by Rafael Pezzi, UFRGS

#include "MFTTracking/TrackExtrap.h"
#include "MFTTracking/TrackParamMFT.h"

#include "CommonConstants/MathConstants.h"
#include "MathUtils/Utils.h"
#include <TGeoGlobalMagField.h>
#include <TGeoManager.h>
#include <TGeoMaterial.h>
#include <TGeoNode.h>
#include <TGeoShape.h>
#include <TMath.h>

#include <FairMQLogger.h>

namespace o2
{
namespace mft
{

//__________________________________________________________________________
void TrackExtrap::linearExtrapToZ(TrackParamMFT* trackParam, double zEnd)
{
  /// Track parameters linearly extrapolated to the plane at "zEnd".
  /// On return, results from the extrapolation are updated in trackParam.

  if (trackParam->getZ() == zEnd) {
    return; // nothing to be done if same z
  }

  // Compute track parameters
  double dZ = (zEnd - trackParam->getZ());
  double x0 = trackParam->getX();
  double y0 = trackParam->getY();
  double phi0 = trackParam->getPhi();
  double cosphi0, sinphi0;
  o2::utils::sincos(phi0, sinphi0, cosphi0);
  double invtanl0 = 1.0 / trackParam->getTanl();
  double n = dZ * invtanl0;
  double x = x0 + n * cosphi0;
  double y = y0 + n * sinphi0;
  trackParam->setX(x);
  trackParam->setY(y);
  trackParam->setZ(zEnd);
}

//__________________________________________________________________________
void TrackExtrap::linearExtrapToZCov(TrackParamMFT* trackParam, double zEnd, bool updatePropagator)
{
  /// Track parameters and their covariances linearly extrapolated to the plane at "zEnd".
  /// On return, results from the extrapolation are updated in trackParam.

  // Calculate the jacobian related to the track parameters extrapolated to "zEnd"
  double dZ = (zEnd - trackParam->getZ());
  double phi0 = trackParam->getPhi();
  double tanl0 = trackParam->getTanl();
  double invtanl0 = 1.0 / tanl0;
  //  double invqpt0 = trackParam->getInvQPt();
  double cosphi0, sinphi0;
  o2::utils::sincos(phi0, sinphi0, cosphi0);
  // double k = TMath::Abs(o2::constants::math::B2C * getBz());
  double n = dZ * invtanl0;
  double m = n * invtanl0;
  //  double theta = -invqpt0 * dZ * k * invtanl0;
  //  auto Hz = getSignBz();

  linearExtrapToZ(trackParam, zEnd);

  // Calculate Jacobian
  TMatrixD jacob(5, 5);
  jacob.UnitMatrix();
  jacob(0, 2) = -n * sinphi0;
  jacob(0, 3) = -m * cosphi0;
  jacob(1, 2) = n * cosphi0;
  jacob(1, 3) = -m * sinphi0;

  // Extrapolate track parameter covariances to "zEnd"
  TMatrixD tmp(trackParam->getCovariances(), TMatrixD::kMultTranspose, jacob);
  TMatrixD tmp2(jacob, TMatrixD::kMult, tmp);
  trackParam->setCovariances(tmp2);

  // Update the propagator if required
  if (updatePropagator) {
    trackParam->updatePropagator(jacob);
  }
}

//__________________________________________________________________________
void TrackExtrap::quadraticExtrapToZ(TrackParamMFT* trackParam, double zEnd)
{
  /// Track parameters extrapolated to the plane at "zEnd" considering a helix
  /// On return, results from the extrapolation are updated in trackParam.

  if (trackParam->getZ() == zEnd) {
    return; // nothing to be done if same z
  }

  // Compute track parameters
  double dZ = (zEnd - trackParam->getZ());
  double x0 = trackParam->getX();
  double y0 = trackParam->getY();
  double phi0 = trackParam->getPhi();
  double cosphi0, sinphi0;
  o2::utils::sincos(phi0, sinphi0, cosphi0);
  double invtanl0 = 1.0 / trackParam->getTanl();
  double invqpt0 = trackParam->getInvQPt();
  auto Hz = getSignBz();
  double k = TMath::Abs(o2::constants::math::B2C * getBz());
  double n = dZ * invtanl0;
  double theta = -invqpt0 * dZ * k * invtanl0;
  double deltax = n * cosphi0 - 0.5 * n * theta * Hz * sinphi0;
  double deltay = n * sinphi0 + 0.5 * n * theta * Hz * cosphi0;

  double x = x0 + deltax;
  double y = y0 + deltay;
  double phi = phi0 + Hz * theta;

  trackParam->setX(x);
  trackParam->setY(y);
  trackParam->setZ(zEnd);
  trackParam->setPhi(phi);
}

//__________________________________________________________________________
void TrackExtrap::helixExtrapToZ(TrackParamMFT* trackParam, double zEnd)
{
  /// Track parameters extrapolated to the plane at "zEnd" considering a helix
  /// On return, results from the extrapolation are updated in trackParam.

  if (trackParam->getZ() == zEnd) {
    return; // nothing to be done if same z
  }

  // Compute track parameters
  double dZ = (zEnd - trackParam->getZ());
  double x0 = trackParam->getX();
  double y0 = trackParam->getY();
  double px0 = trackParam->getPx();
  double py0 = trackParam->getPy();
  double invtanl0 = 1.0 / trackParam->getTanl();
  double invqpt0 = trackParam->getInvQPt();
  auto q = trackParam->getCharge();
  auto Hz = getSignBz();
  double k = TMath::Abs(o2::constants::math::B2C * getBz());
  auto invk = 1.0 / k;
  double theta = -invqpt0 * dZ * k * invtanl0;
  double costheta, sintheta;
  o2::utils::sincos(theta, sintheta, costheta);
  double deltax = Hz * py0 * invk * (1.0 - costheta) - px0 * q * invk * sintheta;
  double deltay = -Hz * px0 * invk * (1.0 - costheta) - py0 * q * invk * sintheta;

  double x = x0 + deltax;
  double y = y0 + deltay;
  double phi = trackParam->getPhi() + Hz * theta;

  trackParam->setX(x);
  trackParam->setY(y);
  trackParam->setZ(zEnd);
  trackParam->setPhi(phi);
}

//__________________________________________________________________________
void TrackExtrap::quadraticExtrapToZCov(TrackParamMFT* trackParam, double zEnd, bool updatePropagator)
{

  // Calculate the jacobian related to the track parameters extrapolated to "zEnd"
  double dZ = (zEnd - trackParam->getZ());
  double phi0 = trackParam->getPhi();
  double tanl0 = trackParam->getTanl();
  double invtanl0 = 1.0 / tanl0;
  double invqpt0 = trackParam->getInvQPt();
  double cosphi0, sinphi0;
  o2::utils::sincos(phi0, sinphi0, cosphi0);
  double k = TMath::Abs(o2::constants::math::B2C * getBz());
  double n = dZ * invtanl0;
  double m = n * invtanl0;
  double theta = -invqpt0 * dZ * k * invtanl0;
  auto Hz = getSignBz();

  quadraticExtrapToZ(trackParam, zEnd);

  // Calculate Jacobian
  TMatrixD jacob(5, 5);
  jacob.UnitMatrix();
  jacob(0, 2) = -n * theta * 0.5 * Hz * cosphi0 - n * sinphi0;
  jacob(0, 3) = Hz * m * theta * sinphi0 - m * cosphi0;
  jacob(0, 4) = k * m * 0.5 * Hz * dZ * sinphi0;
  jacob(1, 2) = -n * theta * 0.5 * Hz * sinphi0 + n * cosphi0;
  jacob(1, 3) = -Hz * m * theta * cosphi0 - m * sinphi0;
  jacob(1, 4) = -k * m * 0.5 * Hz * dZ * cosphi0;
  jacob(2, 3) = -Hz * theta * invtanl0;
  jacob(2, 4) = -Hz * k * n;

  // Extrapolate track parameter covariances to "zEnd"
  TMatrixD tmp(trackParam->getCovariances(), TMatrixD::kMultTranspose, jacob);
  TMatrixD tmp2(jacob, TMatrixD::kMult, tmp);
  trackParam->setCovariances(tmp2);

  // Update the propagator if required
  if (updatePropagator) {
    trackParam->updatePropagator(jacob);
  }
}

//__________________________________________________________________________
void TrackExtrap::helixExtrapToZCov(TrackParamMFT* trackParam, double zEnd, bool updatePropagator)
{

  // Calculate the jacobian related to the track parameters extrapolated to "zEnd"
  double dZ = (zEnd - trackParam->getZ());
  double phi0 = trackParam->getPhi();
  double tanl0 = trackParam->getTanl();
  double invtanl0 = 1.0 / tanl0;
  double invqpt0 = trackParam->getInvQPt();
  auto qpt0 = 1.0 / invqpt0;
  double cosphi0, sinphi0;
  o2::utils::sincos(phi0, sinphi0, cosphi0);
  double k = TMath::Abs(o2::constants::math::B2C * getBz());
  double invk = 1.0 / k;
  double theta = -invqpt0 * dZ * k * invtanl0;
  double costheta, sintheta;
  o2::utils::sincos(theta, sintheta, costheta);
  auto Hz = getSignBz();
  auto L = qpt0 * qpt0 * invk;
  auto N = dZ * invtanl0 * qpt0;
  auto O = sintheta * cosphi0;
  auto P = sinphi0 * costheta;
  auto R = sinphi0 * sintheta;
  auto S = cosphi0 * costheta;
  auto Y = sinphi0 * qpt0 * invk;
  auto X = cosphi0 * qpt0 * invk;
  auto YC = Y * costheta;
  auto YS = Y * sintheta;
  auto XC = X * costheta;
  auto XS = X * sintheta;
  auto T = qpt0 * costheta;
  auto U = qpt0 * sintheta;
  auto V = qpt0;
  double n = dZ * invtanl0;
  double m = n * invtanl0;

  // Extrapolate track parameters to "zEnd"
  helixExtrapToZ(trackParam, zEnd);

  // Calculate Jacobian
  TMatrixD jacob(5, 5);
  jacob.UnitMatrix();
  jacob(0, 2) = Hz * X - Hz * XC + YS;
  jacob(0, 3) = Hz * R * m - S * m;
  jacob(0, 4) = -Hz * N * R + Hz * T * Y - Hz * V * Y + N * S + U * X;
  jacob(1, 2) = Hz * Y - Hz * YC - XS;
  jacob(1, 3) = -Hz * O * m - P * m;
  jacob(1, 4) = Hz * N * O - Hz * T * X + Hz * V * X + N * P + U * Y;
  jacob(2, 3) = -Hz * theta * invtanl0;
  jacob(2, 4) = -Hz * k * n;

  // Extrapolate track parameter covariances to "zEnd"
  TMatrixD tmp(trackParam->getCovariances(), TMatrixD::kMultTranspose, jacob);
  TMatrixD tmp2(jacob, TMatrixD::kMult, tmp);
  trackParam->setCovariances(tmp2);

  // Update the propagator if required
  if (updatePropagator) {
    trackParam->updatePropagator(jacob);
  }
}

//__________________________________________________________________________
bool TrackExtrap::extrapToZ(TrackParamMFT* trackParam, double zEnd, bool isFieldON)
{
  /// Interface to track parameter extrapolation to the plane at "Z".
  /// On return, the track parameters resulting from the extrapolation are updated in trackParam.
  if (!isFieldON) {
    linearExtrapToZ(trackParam, zEnd);
    return true;
  } else {

    quadraticExtrapToZ(trackParam, zEnd);
    return true;
  }
}

//__________________________________________________________________________
void TrackExtrap::extrapToZCov(TrackParamMFT* trackParam, double zEnd, bool updatePropagator, bool isFieldON)
{
  /// Track parameters and their covariances extrapolated to the plane at "zEnd".
  /// On return, results from the extrapolation are updated in trackParam.

  auto& mftTrackingParam = MFTTrackingParam::Instance();

  if (!isFieldON) { // linear extrapolation if no magnetic field
    linearExtrapToZCov(trackParam, zEnd, updatePropagator);
    return;
  } else {
    // Extrapolate track parameters to "zEnd"

    switch (mftTrackingParam.trackmodel) {
      case Helix:
        helixExtrapToZCov(trackParam, zEnd, updatePropagator);
        return;
        break;
      case Quadratic:
        quadraticExtrapToZCov(trackParam, zEnd, updatePropagator);
        return;
        break;
      case Linear:
        linearExtrapToZCov(trackParam, zEnd, updatePropagator);
        return;
        break;
    }
  }
}

//__________________________________________________________________________
void TrackExtrap::addMCSEffect(TrackParamMFT* trackParam, double dZ, double x0)
{
  /// Add to the track parameter covariances the effects of multiple Coulomb scattering
  /// through a material of thickness "abs(dZ)" and of radiation length "x0"
  /// assuming linear propagation and using the small angle approximation.
  /// All scattering evaluated happens at the position of the first cluster

  bool debug = false;
  double phi0 = trackParam->getPhi();
  double tanl0 = trackParam->getTanl();
  double invtanl0 = 1.0 / tanl0;
  double invqpt0 = trackParam->getInvQPt();
  double p = trackParam->getP();

  double cosphi0, sinphi0;
  o2::utils::sincos(phi0, sinphi0, cosphi0);

  double csclambda = TMath::Abs(TMath::Sqrt(1 + tanl0 * tanl0) * invtanl0);
  double pathLengthOverX0 = x0 * csclambda;

  // Angular dispersion square of the track (variance) in a plane perpendicular to the trajectory
  double sigmathetasq = 0.0136 * invqpt0 * (1 + 0.038 * TMath::Log(pathLengthOverX0));
  sigmathetasq *= sigmathetasq * pathLengthOverX0;

  // Get covariance matrix
  TMatrixD newParamCov(trackParam->getCovariances());
  if (debug) {
    std::cout << "Track covariances before MCS:";
    newParamCov.Print();
  }

  if (dZ > 0) {
    double A = tanl0 * tanl0 + 1;
    double B = dZ * cosphi0 * invtanl0;
    double C = dZ * sinphi0 * invtanl0;
    double D = A * B * invtanl0;
    double E = -A * C * invtanl0;
    double F = -C - D;
    double G = B + E;
    double H = -invqpt0 * tanl0;

    newParamCov(0, 0) += sigmathetasq * F * F;

    newParamCov(0, 1) += sigmathetasq * F * G;
    newParamCov(1, 0) += sigmathetasq * F * G;

    newParamCov(1, 1) += sigmathetasq * G * G;

    newParamCov(2, 0) += sigmathetasq * F;
    newParamCov(0, 2) += sigmathetasq * F;

    newParamCov(2, 1) += sigmathetasq * G;
    newParamCov(1, 2) += sigmathetasq * G;

    newParamCov(2, 2) += sigmathetasq;

    newParamCov(3, 0) += sigmathetasq * A * F;
    newParamCov(0, 3) += sigmathetasq * A * F;

    newParamCov(3, 1) += sigmathetasq * A * G;
    newParamCov(1, 3) += sigmathetasq * A * G;

    newParamCov(3, 2) += sigmathetasq * A;
    newParamCov(2, 3) += sigmathetasq * A;

    newParamCov(3, 3) += sigmathetasq * A * A;

    newParamCov(4, 0) += sigmathetasq * F * H;
    newParamCov(0, 4) += sigmathetasq * F * H;

    newParamCov(4, 1) += sigmathetasq * G * H;
    newParamCov(1, 4) += sigmathetasq * G * H;

    newParamCov(4, 2) += sigmathetasq * H;
    newParamCov(2, 4) += sigmathetasq * H;

    newParamCov(4, 3) += sigmathetasq * A * H;
    newParamCov(3, 4) += sigmathetasq * A * H;

    newParamCov(4, 4) += sigmathetasq * tanl0 * tanl0 * invqpt0 * invqpt0;
  } else {

    double A = tanl0 * tanl0 + 1;
    double H = -invqpt0 * tanl0;

    newParamCov(2, 2) += sigmathetasq;

    newParamCov(3, 2) += sigmathetasq * A;
    newParamCov(2, 3) += sigmathetasq * A;

    newParamCov(3, 3) += sigmathetasq * A * A;

    newParamCov(4, 2) += sigmathetasq * H;
    newParamCov(2, 4) += sigmathetasq * H;

    newParamCov(4, 3) += sigmathetasq * A * H;
    newParamCov(3, 4) += sigmathetasq * A * H;

    newParamCov(4, 4) += sigmathetasq * tanl0 * tanl0 * invqpt0 * invqpt0;
  }

  if (debug) {
    std::cout << "Track covariances after MCS:";
    newParamCov.Print();
    std::cout << " **********************************************************\n";
  }

  // Set new covariances
  trackParam->setCovariances(newParamCov);
}

} // namespace mft
} // namespace o2
