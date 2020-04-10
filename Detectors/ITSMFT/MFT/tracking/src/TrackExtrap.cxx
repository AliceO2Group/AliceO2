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
  double dZ = zEnd - trackParam->getZ();
  double xSlope = trackParam->getPx() / trackParam->getPz();
  double ySlope = trackParam->getPy() / trackParam->getPz();
  trackParam->setX(trackParam->getX() + xSlope * dZ);
  trackParam->setY(trackParam->getY() + ySlope * dZ);
  trackParam->setZ(zEnd);
}

//__________________________________________________________________________
void TrackExtrap::linearExtrapToZCov(TrackParamMFT* trackParam, double zEnd, bool updatePropagator)
{
  /// Track parameters and their covariances linearly extrapolated to the plane at "zEnd".
  /// On return, results from the extrapolation are updated in trackParam.

  if (trackParam->getZ() == zEnd) {
    return; // nothing to be done if same z
  }

  // No need to propagate the covariance matrix if it does not exist
  if (!trackParam->hasCovariances()) {
    LOG(WARNING) << "Covariance matrix does not exist";
    // Extrapolate linearly track parameters to "zEnd"
    linearExtrapToZ(trackParam, zEnd);
    return;
  }

  // Compute track parameters
  double dZ = zEnd - trackParam->getZ();
  double xSlope = trackParam->getPx() / trackParam->getPz();
  double ySlope = trackParam->getPy() / trackParam->getPz();
  trackParam->setX(trackParam->getX() + xSlope * dZ);
  trackParam->setY(trackParam->getY() + ySlope * dZ);
  trackParam->setZ(zEnd);

  // Calculate the jacobian related to the track parameters linear extrapolation to "zEnd"
  TMatrixD jacob(5, 5);
  jacob.UnitMatrix();
  jacob(0, 1) = dZ;
  jacob(2, 3) = dZ;

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
  double phi0 = trackParam->getPhi();
  double cosphi0 = TMath::Cos(phi0);
  double sinphi0 = TMath::Sin(phi0);
  double tanl0 = trackParam->getTanl();
  double invqpt0 = trackParam->getInvQPt();

  double k = getBz() * o2::constants::math::B2C;
  double deltax = (dZ * cosphi0 / tanl0 - dZ * dZ * k * invqpt0 * sinphi0 / (2. * tanl0 * tanl0));
  double deltay = (dZ * sinphi0 / tanl0 + dZ * dZ * k * invqpt0 * cosphi0 / (2. * tanl0 * tanl0));

  double x = x0 + deltax;
  double y = y0 + deltay;
  double deltaphi = +dZ * k * invqpt0 / tanl0;
  //std::cout << "    Deltaphi extrap = " << deltaphi << " dZ = " << dZ << std::endl;
  //std::cout << "      Deltax extrap = " << deltax << " = " << (dZ * cosphi0 / tanl0)*100 << " + " << (- dZ * dZ * k * invqpt0 * sinphi0 / (2. * tanl0 * tanl0))*100 << std::endl;
  //std::cout << "      Deltay extrap = " << deltay << std::endl;

  float phi = phi0 + deltaphi;
  //o2::utils::BringToPMPi(phi);
  double tanl = tanl0;
  double invqpt = invqpt0;
  trackParam->setX(x);
  trackParam->setY(y);
  trackParam->setZ(zEnd);
  trackParam->setPhi(phi);
  trackParam->setTanl(tanl);
  trackParam->setInvQPt(invqpt);
}

//__________________________________________________________________________
void TrackExtrap::helixExtrapToZCov(TrackParamMFT* trackParam, double zEnd, bool updatePropagator)
{
  helixExtrapToZ(trackParam, zEnd);

  // Calculate the jacobian related to the track parameters linear extrapolation to "zEnd"
  double dZ = (zEnd - trackParam->getZ()); // Propagate in meters
  double phi0 = trackParam->getPhi();
  double tanl0 = trackParam->getTanl();
  double invqpt0 = trackParam->getInvQPt();
  double dZ2 = dZ * dZ;
  double cosphi0 = TMath::Cos(phi0);
  double sinphi0 = TMath::Sin(phi0);
  double tanl0sq = tanl0 * tanl0;
  double k = getBz() * o2::constants::math::B2C;

  TMatrixD jacob(5, 5);
  jacob.UnitMatrix();
  jacob(0, 2) = -dZ2 * k * invqpt0 * cosphi0 / 2. / tanl0sq - dZ * sinphi0 / tanl0;
  jacob(0, 3) = dZ2 * k * invqpt0 * sinphi0 / tanl0sq / tanl0 - dZ * cosphi0 / tanl0sq;
  jacob(0, 4) = -dZ2 * k * sinphi0 / 2. / tanl0sq;
  jacob(1, 2) = -dZ2 * k * invqpt0 * sinphi0 / 2. / tanl0sq + dZ * cosphi0 / tanl0;
  jacob(1, 3) = -dZ2 * k * invqpt0 * cosphi0 / tanl0sq / tanl0 - dZ * sinphi0 / tanl0sq;
  jacob(1, 4) = dZ2 * k * cosphi0 / 2. / tanl0sq;
  jacob(2, 3) = -dZ * k * invqpt0 / tanl0sq;
  jacob(2, 4) = dZ * k / tanl0;

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
    helixExtrapToZ(trackParam, zEnd);
    return true;
  }
}

//__________________________________________________________________________
bool TrackExtrap::extrapToZCov(TrackParamMFT* trackParam, double zEnd, bool updatePropagator, bool isFieldON)
{
  /// Track parameters and their covariances extrapolated to the plane at "zEnd".
  /// On return, results from the extrapolation are updated in trackParam.

  if (!isFieldON) { // linear extrapolation if no magnetic field
    linearExtrapToZCov(trackParam, zEnd, updatePropagator);
    return true;
  } else {
    helixExtrapToZCov(trackParam, zEnd, updatePropagator);
    return true;
  }
}

//__________________________________________________________________________
void TrackExtrap::addMCSEffect(TrackParamMFT* trackParam, double dZ, double x0, bool isFieldON)
{
  /// Add to the track parameter covariances the effects of multiple Coulomb scattering
  /// through a material of thickness "abs(dZ)" and of radiation length "x0"
  /// assuming linear propagation and using the small angle approximation.
  /// dZ = zOut - zIn (sign is important) and "param" is assumed to be given zOut.
  /// If x0 <= 0., assume dZ = pathLength/x0 and consider the material thickness as negligible.

  double xSlope = trackParam->getPx() / trackParam->getPz();
  double ySlope = trackParam->getPy() / trackParam->getPz();

  double inverseMomentum = trackParam->getInverseMomentum();
  double inverseTotalMomentum2 = inverseMomentum * inverseMomentum * (1.0 + ySlope * ySlope) /
                                 (1.0 + ySlope * ySlope + xSlope * xSlope);
  // Path length in the material
  double signedPathLength = dZ * TMath::Sqrt(1.0 + ySlope * ySlope + xSlope * xSlope);
  double pathLengthOverX0 = (x0 > 0.) ? TMath::Abs(signedPathLength) / x0 : TMath::Abs(signedPathLength);
  // relativistic velocity
  double velo = 1.;
  // Angular dispersion square of the track (variance) in a plane perpendicular to the trajectory
  double theta02 = 0.0136 / velo * (1 + 0.038 * TMath::Log(pathLengthOverX0));
  theta02 *= theta02 * inverseTotalMomentum2 * pathLengthOverX0;

  double varCoor = (x0 > 0.) ? signedPathLength * signedPathLength * theta02 / 3. : 0.;
  double varSlop = theta02;
  double covCorrSlope = (x0 > 0.) ? signedPathLength * theta02 / 2. : 0.;

  // Set MCS covariance matrix
  TMatrixD newParamCov(trackParam->getCovariances());
  // Non bending plane    // FIXME: Update param coordinate system
  newParamCov(0, 0) += varCoor;
  newParamCov(0, 1) += covCorrSlope;
  newParamCov(1, 0) += covCorrSlope;
  newParamCov(1, 1) += varCoor;
  // Bending plane
  newParamCov(2, 2) += varSlop;
  newParamCov(2, 3) += covCorrSlope;
  newParamCov(3, 2) += covCorrSlope;
  newParamCov(3, 3) += varSlop;

  // Set momentum related covariances if B!=0
  if (isFieldON) {
    // compute derivative d(q/Pxy) / dSlopeX and d(q/Pxy) / dSlopeY
    double dqPxydSlopeX =
      inverseMomentum * xSlope / (1. + xSlope * xSlope + ySlope * ySlope);
    double dqPxydSlopeY = -inverseMomentum * xSlope * xSlope * ySlope /
                          (1. + ySlope * ySlope) /
                          (1. + xSlope * xSlope + ySlope * ySlope);
    // Inverse bending momentum (due to dependences with bending and non bending slopes)
    newParamCov(4, 0) += dqPxydSlopeX * covCorrSlope;
    newParamCov(0, 4) += dqPxydSlopeX * covCorrSlope;
    newParamCov(4, 1) += dqPxydSlopeX * varSlop;
    newParamCov(1, 4) += dqPxydSlopeX * varSlop;
    newParamCov(4, 2) += dqPxydSlopeY * covCorrSlope;
    newParamCov(2, 4) += dqPxydSlopeY * covCorrSlope;
    newParamCov(4, 3) += dqPxydSlopeY * varSlop;
    newParamCov(3, 4) += dqPxydSlopeY * varSlop;
    newParamCov(4, 4) += (dqPxydSlopeX * dqPxydSlopeX + dqPxydSlopeY * dqPxydSlopeY) * varSlop;
  }

  // Set new covariances
  trackParam->setCovariances(newParamCov);
}

} // namespace mft
} // namespace o2
