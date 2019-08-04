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
/// \author Philippe Pillot, Subatech

#include "TrackExtrap.h"

#include <TGeoGlobalMagField.h>
#include <TMath.h>
#include <TMatrixD.h>

#include <FairMQLogger.h>

#include "TrackParam.h"

namespace o2
{
namespace mch
{

double TrackExtrap::sSimpleBValue = 0.;
bool TrackExtrap::sFieldON = false;

//__________________________________________________________________________
void TrackExtrap::setField()
{
  /// Set field on/off flag.
  /// Set field at the centre of the dipole
  const double x[3] = {50., 50., SSimpleBPosition};
  double b[3] = {0., 0., 0.};
  TGeoGlobalMagField::Instance()->Field(x, b);
  sSimpleBValue = b[0];
  sFieldON = (TMath::Abs(sSimpleBValue) > 1.e-10) ? true : false;
  LOG(INFO) << "Track extrapolation with magnetic field " << (sFieldON ? "ON" : "OFF");
}

//__________________________________________________________________________
double TrackExtrap::getImpactParamFromBendingMomentum(double bendingMomentum)
{
  /// Returns impact parameter at vertex in bending plane (cm),
  /// from the signed bending momentum "BendingMomentum" in bending plane (GeV/c),
  /// using simple values for dipole magnetic field.
  /// The sign of "BendingMomentum" is the sign of the charge.
  const double correctionFactor = 1.1; // impact parameter is 10% underestimated
  if (bendingMomentum == 0.) {
    return 1.e10;
  }
  return correctionFactor * (-0.0003 * sSimpleBValue * SSimpleBLength * SSimpleBPosition / bendingMomentum);
}

//__________________________________________________________________________
double TrackExtrap::getBendingMomentumFromImpactParam(double impactParam)
{
  /// Returns signed bending momentum in bending plane (GeV/c),
  /// the sign being the sign of the charge for particles moving forward in Z,
  /// from the impact parameter "ImpactParam" at vertex in bending plane (cm),
  /// using simple values for dipole magnetic field.
  const double correctionFactor = 1.1; // bending momentum is 10% underestimated
  if (impactParam == 0.) {
    return 1.e10;
  }
  if (sFieldON) {
    return correctionFactor * (-0.0003 * sSimpleBValue * SSimpleBLength * SSimpleBPosition / impactParam);
  } else {
    return SMostProbBendingMomentum;
  }
}

//__________________________________________________________________________
void TrackExtrap::linearExtrapToZ(TrackParam* trackParam, double zEnd)
{
  /// Track parameters linearly extrapolated to the plane at "zEnd".
  /// On return, results from the extrapolation are updated in trackParam.

  if (trackParam->getZ() == zEnd) {
    return; // nothing to be done if same z
  }

  // Compute track parameters
  double dZ = zEnd - trackParam->getZ();
  trackParam->setNonBendingCoor(trackParam->getNonBendingCoor() + trackParam->getNonBendingSlope() * dZ);
  trackParam->setBendingCoor(trackParam->getBendingCoor() + trackParam->getBendingSlope() * dZ);
  trackParam->setZ(zEnd);
}

//__________________________________________________________________________
void TrackExtrap::linearExtrapToZCov(TrackParam* trackParam, double zEnd, bool updatePropagator)
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
  trackParam->setNonBendingCoor(trackParam->getNonBendingCoor() + trackParam->getNonBendingSlope() * dZ);
  trackParam->setBendingCoor(trackParam->getBendingCoor() + trackParam->getBendingSlope() * dZ);
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
bool TrackExtrap::extrapToZ(TrackParam* trackParam, double zEnd)
{
  /// Interface to track parameter extrapolation to the plane at "Z" using Helix or Rungekutta algorithm.
  /// On return, the track parameters resulting from the extrapolation are updated in trackParam.
  if (!sFieldON) {
    linearExtrapToZ(trackParam, zEnd);
    return true;
  } else {
    return extrapToZRungekutta(trackParam, zEnd);
  }
}

//__________________________________________________________________________
bool TrackExtrap::extrapToZCov(TrackParam* trackParam, double zEnd, bool updatePropagator)
{
  /// Track parameters and their covariances extrapolated to the plane at "zEnd".
  /// On return, results from the extrapolation are updated in trackParam.

  if (trackParam->getZ() == zEnd) {
    return true; // nothing to be done if same z
  }

  if (!sFieldON) { // linear extrapolation if no magnetic field
    linearExtrapToZCov(trackParam, zEnd, updatePropagator);
    return true;
  }

  // No need to propagate the covariance matrix if it does not exist
  if (!trackParam->hasCovariances()) {
    LOG(WARNING) << "Covariance matrix does not exist";
    // Extrapolate track parameters to "zEnd"
    return extrapToZ(trackParam, zEnd);
  }

  // Save the actual track parameters
  TrackParam trackParamSave(*trackParam);
  TMatrixD paramSave(trackParamSave.getParameters());
  double zBegin = trackParamSave.getZ();

  // Get reference to the parameter covariance matrix
  const TMatrixD& kParamCov = trackParam->getCovariances();

  // Extrapolate track parameters to "zEnd"
  // Do not update the covariance matrix if the extrapolation failed
  if (!extrapToZ(trackParam, zEnd)) {
    return false;
  }

  // Get reference to the extrapolated parameters
  const TMatrixD& extrapParam = trackParam->getParameters();

  // Calculate the jacobian related to the track parameters extrapolation to "zEnd"
  bool extrapStatus = true;
  TMatrixD jacob(5, 5);
  jacob.Zero();
  TMatrixD dParam(5, 1);
  double direction[5] = {-1., -1., 1., 1., -1.};
  for (int i = 0; i < 5; i++) {
    // Skip jacobian calculation for parameters with no associated error
    if (kParamCov(i, i) <= 0.) {
      continue;
    }

    // Small variation of parameter i only
    for (int j = 0; j < 5; j++) {
      if (j == i) {
        dParam(j, 0) = TMath::Sqrt(kParamCov(i, i));
        dParam(j, 0) *= TMath::Sign(1., direction[j] * paramSave(j, 0)); // variation always in the same direction
      } else {
        dParam(j, 0) = 0.;
      }
    }

    // Set new parameters
    trackParamSave.setParameters(paramSave);
    trackParamSave.addParameters(dParam);
    trackParamSave.setZ(zBegin);

    // Extrapolate new track parameters to "zEnd"
    if (!extrapToZ(&trackParamSave, zEnd)) {
      LOG(WARNING) << "Bad covariance matrix";
      extrapStatus = false;
    }

    // Calculate the jacobian
    TMatrixD jacobji(trackParamSave.getParameters(), TMatrixD::kMinus, extrapParam);
    jacobji *= 1. / dParam(i, 0);
    jacob.SetSub(0, i, jacobji);
  }

  // Extrapolate track parameter covariances to "zEnd"
  TMatrixD tmp(kParamCov, TMatrixD::kMultTranspose, jacob);
  TMatrixD tmp2(jacob, TMatrixD::kMult, tmp);
  trackParam->setCovariances(tmp2);

  // Update the propagator if required
  if (updatePropagator) {
    trackParam->updatePropagator(jacob);
  }

  return extrapStatus;
}

//__________________________________________________________________________
double TrackExtrap::getMCSAngle2(const TrackParam& param, double dZ, double x0)
{
  /// Return the angular dispersion square due to multiple Coulomb scattering
  /// through a material of thickness "abs(dZ)" and of radiation length "x0"
  /// assuming linear propagation and using the small angle approximation.
  double bendingSlope = param.getBendingSlope();
  double nonBendingSlope = param.getNonBendingSlope();
  double inverseTotalMomentum2 = param.getInverseBendingMomentum() * param.getInverseBendingMomentum() *
                                 (1.0 + bendingSlope * bendingSlope) /
                                 (1.0 + bendingSlope * bendingSlope + nonBendingSlope * nonBendingSlope);
  // Path length in the material
  double pathLength =
    TMath::Abs(dZ) * TMath::Sqrt(1.0 + bendingSlope * bendingSlope + nonBendingSlope * nonBendingSlope);
  // relativistic velocity
  double velo = 1.;
  // Angular dispersion square of the track (variance) in a plane perpendicular to the trajectory
  double theta02 = 0.0136 / velo * (1 + 0.038 * TMath::Log(pathLength / x0));
  return theta02 * theta02 * inverseTotalMomentum2 * pathLength / x0;
}

//__________________________________________________________________________
void TrackExtrap::addMCSEffect(TrackParam* trackParam, double dZ, double x0)
{
  /// Add to the track parameter covariances the effects of multiple Coulomb scattering
  /// through a material of thickness "abs(dZ)" and of radiation length "x0"
  /// assuming linear propagation and using the small angle approximation.
  /// dZ = zOut - zIn (sign is important) and "param" is assumed to be given zOut.
  /// If x0 <= 0., assume dZ = pathLength/x0 and consider the material thickness as negligible.

  double bendingSlope = trackParam->getBendingSlope();
  double nonBendingSlope = trackParam->getNonBendingSlope();
  double inverseBendingMomentum = trackParam->getInverseBendingMomentum();
  double inverseTotalMomentum2 = inverseBendingMomentum * inverseBendingMomentum * (1.0 + bendingSlope * bendingSlope) /
                                 (1.0 + bendingSlope * bendingSlope + nonBendingSlope * nonBendingSlope);
  // Path length in the material
  double signedPathLength = dZ * TMath::Sqrt(1.0 + bendingSlope * bendingSlope + nonBendingSlope * nonBendingSlope);
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
  // Non bending plane
  newParamCov(0, 0) += varCoor;
  newParamCov(0, 1) += covCorrSlope;
  newParamCov(1, 0) += covCorrSlope;
  newParamCov(1, 1) += varSlop;
  // Bending plane
  newParamCov(2, 2) += varCoor;
  newParamCov(2, 3) += covCorrSlope;
  newParamCov(3, 2) += covCorrSlope;
  newParamCov(3, 3) += varSlop;

  // Set momentum related covariances if B!=0
  if (sFieldON) {
    // compute derivative d(q/Pxy) / dSlopeX and d(q/Pxy) / dSlopeY
    double dqPxydSlopeX =
      inverseBendingMomentum * nonBendingSlope / (1. + nonBendingSlope * nonBendingSlope + bendingSlope * bendingSlope);
    double dqPxydSlopeY = -inverseBendingMomentum * nonBendingSlope * nonBendingSlope * bendingSlope /
                          (1. + bendingSlope * bendingSlope) /
                          (1. + nonBendingSlope * nonBendingSlope + bendingSlope * bendingSlope);
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

//__________________________________________________________________________
void TrackExtrap::convertTrackParamForExtrap(TrackParam* trackParam, double forwardBackward, double* v3)
{
  /// Set vector of Geant3 parameters pointed to by "v3" from track parameters in trackParam.
  /// Since TrackParam is only geometry, one uses "forwardBackward"
  /// to know whether the particle is going forward (+1) or backward (-1).
  v3[0] = trackParam->getNonBendingCoor(); // X
  v3[1] = trackParam->getBendingCoor();    // Y
  v3[2] = trackParam->getZ();              // Z
  double pYZ = TMath::Abs(1.0 / trackParam->getInverseBendingMomentum());
  double pZ = pYZ / TMath::Sqrt(1.0 + trackParam->getBendingSlope() * trackParam->getBendingSlope());
  v3[6] = TMath::Sqrt(pYZ * pYZ + pZ * pZ * trackParam->getNonBendingSlope() * trackParam->getNonBendingSlope()); // P
  v3[5] = -forwardBackward * pZ / v3[6];                                                                          // PZ/P spectro. z<0
  v3[3] = trackParam->getNonBendingSlope() * v3[5];                                                               // PX/P
  v3[4] = trackParam->getBendingSlope() * v3[5];                                                                  // PY/P
}

//__________________________________________________________________________
void TrackExtrap::recoverTrackParam(double* v3, double charge, TrackParam* trackParam)
{
  /// Set track parameters in trackParam from Geant3 parameters pointed to by "v3",
  /// assumed to be calculated for forward motion in Z.
  /// "InverseBendingMomentum" is signed with "charge".
  trackParam->setNonBendingCoor(v3[0]); // X
  trackParam->setBendingCoor(v3[1]);    // Y
  trackParam->setZ(v3[2]);              // Z
  double pYZ = v3[6] * TMath::Sqrt((1. - v3[3]) * (1. + v3[3]));
  trackParam->setInverseBendingMomentum(charge / pYZ);
  trackParam->setBendingSlope(v3[4] / v3[5]);
  trackParam->setNonBendingSlope(v3[3] / v3[5]);
}

//__________________________________________________________________________
bool TrackExtrap::extrapToZRungekutta(TrackParam* trackParam, double zEnd)
{
  /// Track parameter extrapolation to the plane at "Z" using Rungekutta algorithm.
  /// On return, the track parameters resulting from the extrapolation are updated in trackParam.
  if (trackParam->getZ() == zEnd) {
    return true; // nothing to be done if same Z
  }
  double forwardBackward; // +1 if forward, -1 if backward
  if (zEnd < trackParam->getZ()) {
    forwardBackward = 1.0; // spectro. z<0
  } else {
    forwardBackward = -1.0;
  }
  // sign of charge (sign of fInverseBendingMomentum if forward motion)
  // must be changed if backward extrapolation
  double chargeExtrap = forwardBackward * TMath::Sign(double(1.0), trackParam->getInverseBendingMomentum());
  double v3[7] = {0.}, v3New[7] = {0.};
  double dZ(0.), step(0.);
  int stepNumber = 0;

  // Extrapolation loop (until within tolerance or the track turn around)
  double residue = zEnd - trackParam->getZ();
  bool uturn = false;
  bool trackingFailed = false;
  bool tooManyStep = false;
  while (TMath::Abs(residue) > SRungeKuttaMaxResidue && stepNumber <= SMaxStepNumber) {

    dZ = zEnd - trackParam->getZ();
    // step lenght assuming linear trajectory
    step = dZ * TMath::Sqrt(1.0 + trackParam->getBendingSlope() * trackParam->getBendingSlope() +
                            trackParam->getNonBendingSlope() * trackParam->getNonBendingSlope());
    convertTrackParamForExtrap(trackParam, forwardBackward, v3);

    do { // reduce step lenght while zEnd oversteped
      if (stepNumber > SMaxStepNumber) {
        LOG(WARNING) << "Too many trials: " << stepNumber;
        tooManyStep = true;
        break;
      }
      stepNumber++;
      step = TMath::Abs(step);
      if (!extrapOneStepRungekutta(chargeExtrap, step, v3, v3New)) {
        trackingFailed = true;
        break;
      }
      residue = zEnd - v3New[2];
      step *= dZ / (v3New[2] - trackParam->getZ());
    } while (residue * dZ < 0 && TMath::Abs(residue) > SRungeKuttaMaxResidue);

    if (trackingFailed) {
      break;
    } else if (v3New[5] * v3[5] < 0) { // the track turned around
      LOG(WARNING) << "The track turned around";
      uturn = true;
      break;
    } else {
      recoverTrackParam(v3New, chargeExtrap * forwardBackward, trackParam);
    }
  }

  // terminate the extropolation with a straight line up to the exact "zEnd" value
  if (trackingFailed || uturn) {

    // track ends +-100 meters away in the bending direction
    dZ = zEnd - v3[2];
    double bendingSlope = TMath::Sign(1.e4, -sSimpleBValue * trackParam->getInverseBendingMomentum()) / dZ;
    double pZ =
      TMath::Abs(1. / trackParam->getInverseBendingMomentum()) / TMath::Sqrt(1.0 + bendingSlope * bendingSlope);
    double nonBendingSlope = TMath::Sign(TMath::Abs(v3[3]) * v3[6] / pZ, trackParam->getNonBendingSlope());
    trackParam->setNonBendingCoor(trackParam->getNonBendingCoor() + dZ * nonBendingSlope);
    trackParam->setNonBendingSlope(nonBendingSlope);
    trackParam->setBendingCoor(trackParam->getBendingCoor() + dZ * bendingSlope);
    trackParam->setBendingSlope(bendingSlope);
    trackParam->setZ(zEnd);

    return false;

  } else {

    // track extrapolated normally
    trackParam->setNonBendingCoor(trackParam->getNonBendingCoor() + residue * trackParam->getNonBendingSlope());
    trackParam->setBendingCoor(trackParam->getBendingCoor() + residue * trackParam->getBendingSlope());
    trackParam->setZ(zEnd);

    return !tooManyStep;
  }
}

//__________________________________________________________________________
bool TrackExtrap::extrapOneStepRungekutta(double charge, double step, const double* vect, double* vout)
{
  /// <pre>
  ///  ******************************************************************
  ///  *                                                                *
  ///  *  Runge-Kutta method for tracking a particle through a magnetic *
  ///  *  field. Uses Nystroem algorithm (See Handbook Nat. Bur. of     *
  ///  *  Standards, procedure 25.5.20)                                 *
  ///  *                                                                *
  ///  *  Input parameters                                              *
  ///  *  CHARGE    Particle charge                                     *
  ///  *  STEP    Step size                                             *
  ///  *  VECT    Initial co-ords,direction cosines,momentum            *
  ///  *  Output parameters                                             *
  ///  *  VOUT    Output co-ords,direction cosines,momentum             *
  ///  *  User routine called                                           *
  ///  *  CALL GUFLD(X,F)                                               *
  ///  *                                                                *
  ///  *    ==>Called by : USER, GUSWIM                                 *
  ///  *  Authors    R.Brun, M.Hansroul  *********                      *
  ///  *       V.Perevoztchikov (CUT STEP implementation)               *
  ///  *                                                                *
  ///  ******************************************************************
  /// </pre>

  double h2(0.), h4(0.), f[4] = {0.};
  double xyzt[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
  double a(0.), b(0.), c(0.), ph(0.), ph2(0.);
  double secxs[4] = {0.}, secys[4] = {0.}, seczs[4] = {0.}, hxp[3] = {0.};
  double g1(0.), g2(0.), g3(0.), g4(0.), g5(0.), g6(0.), ang2(0.), dxt(0.), dyt(0.), dzt(0.);
  double est(0.), at(0.), bt(0.), ct(0.), cba(0.);
  double f1(0.), f2(0.), f3(0.), f4(0.), rho(0.), tet(0.), hnorm(0.), hp(0.), rho1(0.), sint(0.), cost(0.);

  double x(0.);
  double y(0.);
  double z(0.);

  double xt(0.);
  double yt(0.);
  double zt(0.);

  double maxit = 1992;
  double maxcut = 11;

  const double kdlt = 1e-4;
  const double kdlt32 = kdlt / 32.;
  const double kthird = 1. / 3.;
  const double khalf = 0.5;
  const double kec = 2.9979251e-4;

  const double kpisqua = 9.86960440109;
  const int kix = 0;
  const int kiy = 1;
  const int kiz = 2;
  const int kipx = 3;
  const int kipy = 4;
  const int kipz = 5;

  // *.
  // *.    ------------------------------------------------------------------
  // *.
  // *             this constant is for units cm,gev/c and kgauss
  // *
  int iter = 0;
  int ncut = 0;
  for (int j = 0; j < 7; j++) {
    vout[j] = vect[j];
  }

  double pinv = kec * charge / vect[6];
  double tl = 0.;
  double h = step;
  double rest(0.);

  do {
    rest = step - tl;
    if (TMath::Abs(h) > TMath::Abs(rest)) {
      h = rest;
    }
    // cmodif: call gufld(vout,f) changed into:
    TGeoGlobalMagField::Instance()->Field(vout, f);

    // *
    // *             start of integration
    // *
    x = vout[0];
    y = vout[1];
    z = vout[2];
    a = vout[3];
    b = vout[4];
    c = vout[5];

    h2 = khalf * h;
    h4 = khalf * h2;
    ph = pinv * h;
    ph2 = khalf * ph;
    secxs[0] = (b * f[2] - c * f[1]) * ph2;
    secys[0] = (c * f[0] - a * f[2]) * ph2;
    seczs[0] = (a * f[1] - b * f[0]) * ph2;
    ang2 = (secxs[0] * secxs[0] + secys[0] * secys[0] + seczs[0] * seczs[0]);
    if (ang2 > kpisqua) {
      break;
    }

    dxt = h2 * a + h4 * secxs[0];
    dyt = h2 * b + h4 * secys[0];
    dzt = h2 * c + h4 * seczs[0];
    xt = x + dxt;
    yt = y + dyt;
    zt = z + dzt;
    // *
    // *              second intermediate point
    // *

    est = TMath::Abs(dxt) + TMath::Abs(dyt) + TMath::Abs(dzt);
    if (est > h) {
      if (ncut++ > maxcut) {
        break;
      }
      h *= khalf;
      continue;
    }

    xyzt[0] = xt;
    xyzt[1] = yt;
    xyzt[2] = zt;

    // cmodif: call gufld(xyzt,f) changed into:
    TGeoGlobalMagField::Instance()->Field(xyzt, f);

    at = a + secxs[0];
    bt = b + secys[0];
    ct = c + seczs[0];

    secxs[1] = (bt * f[2] - ct * f[1]) * ph2;
    secys[1] = (ct * f[0] - at * f[2]) * ph2;
    seczs[1] = (at * f[1] - bt * f[0]) * ph2;
    at = a + secxs[1];
    bt = b + secys[1];
    ct = c + seczs[1];
    secxs[2] = (bt * f[2] - ct * f[1]) * ph2;
    secys[2] = (ct * f[0] - at * f[2]) * ph2;
    seczs[2] = (at * f[1] - bt * f[0]) * ph2;
    dxt = h * (a + secxs[2]);
    dyt = h * (b + secys[2]);
    dzt = h * (c + seczs[2]);
    xt = x + dxt;
    yt = y + dyt;
    zt = z + dzt;
    at = a + 2. * secxs[2];
    bt = b + 2. * secys[2];
    ct = c + 2. * seczs[2];

    est = TMath::Abs(dxt) + TMath::Abs(dyt) + TMath::Abs(dzt);
    if (est > 2. * TMath::Abs(h)) {
      if (ncut++ > maxcut) {
        break;
      }
      h *= khalf;
      continue;
    }

    xyzt[0] = xt;
    xyzt[1] = yt;
    xyzt[2] = zt;

    // cmodif: call gufld(xyzt,f) changed into:
    TGeoGlobalMagField::Instance()->Field(xyzt, f);

    z = z + (c + (seczs[0] + seczs[1] + seczs[2]) * kthird) * h;
    y = y + (b + (secys[0] + secys[1] + secys[2]) * kthird) * h;
    x = x + (a + (secxs[0] + secxs[1] + secxs[2]) * kthird) * h;

    secxs[3] = (bt * f[2] - ct * f[1]) * ph2;
    secys[3] = (ct * f[0] - at * f[2]) * ph2;
    seczs[3] = (at * f[1] - bt * f[0]) * ph2;
    a = a + (secxs[0] + secxs[3] + 2. * (secxs[1] + secxs[2])) * kthird;
    b = b + (secys[0] + secys[3] + 2. * (secys[1] + secys[2])) * kthird;
    c = c + (seczs[0] + seczs[3] + 2. * (seczs[1] + seczs[2])) * kthird;

    est = TMath::Abs(secxs[0] + secxs[3] - (secxs[1] + secxs[2])) +
          TMath::Abs(secys[0] + secys[3] - (secys[1] + secys[2])) +
          TMath::Abs(seczs[0] + seczs[3] - (seczs[1] + seczs[2]));

    if (est > kdlt && TMath::Abs(h) > 1.e-4) {
      if (ncut++ > maxcut) {
        break;
      }
      h *= khalf;
      continue;
    }

    ncut = 0;
    // *               if too many iterations, go to helix
    if (iter++ > maxit) {
      break;
    }

    tl += h;
    if (est < kdlt32) {
      h *= 2.;
    }
    cba = 1. / TMath::Sqrt(a * a + b * b + c * c);
    vout[0] = x;
    vout[1] = y;
    vout[2] = z;
    vout[3] = cba * a;
    vout[4] = cba * b;
    vout[5] = cba * c;
    rest = step - tl;
    if (step < 0.) {
      rest = -rest;
    }
    if (rest < 1.e-5 * TMath::Abs(step)) {
      return true;
    }

  } while (1);

  // angle too big, use helix
  LOG(WARNING) << "Ruge-Kutta failed: switch to helix";

  f1 = f[0];
  f2 = f[1];
  f3 = f[2];
  f4 = TMath::Sqrt(f1 * f1 + f2 * f2 + f3 * f3);
  if (f4 < 1.e-10) {
    LOG(ERROR) << "magnetic field at (" << xyzt[0] << ", " << xyzt[1] << ", " << xyzt[2] << ") = " << f4
               << ": giving up";
    return false;
  }
  rho = -f4 * pinv;
  tet = rho * step;

  hnorm = 1. / f4;
  f1 = f1 * hnorm;
  f2 = f2 * hnorm;
  f3 = f3 * hnorm;

  hxp[0] = f2 * vect[kipz] - f3 * vect[kipy];
  hxp[1] = f3 * vect[kipx] - f1 * vect[kipz];
  hxp[2] = f1 * vect[kipy] - f2 * vect[kipx];

  hp = f1 * vect[kipx] + f2 * vect[kipy] + f3 * vect[kipz];

  rho1 = 1. / rho;
  sint = TMath::Sin(tet);
  cost = 2. * TMath::Sin(khalf * tet) * TMath::Sin(khalf * tet);

  g1 = sint * rho1;
  g2 = cost * rho1;
  g3 = (tet - sint) * hp * rho1;
  g4 = -cost;
  g5 = sint;
  g6 = cost * hp;

  vout[kix] = vect[kix] + g1 * vect[kipx] + g2 * hxp[0] + g3 * f1;
  vout[kiy] = vect[kiy] + g1 * vect[kipy] + g2 * hxp[1] + g3 * f2;
  vout[kiz] = vect[kiz] + g1 * vect[kipz] + g2 * hxp[2] + g3 * f3;

  vout[kipx] = vect[kipx] + g4 * vect[kipx] + g5 * hxp[0] + g6 * f1;
  vout[kipy] = vect[kipy] + g4 * vect[kipy] + g5 * hxp[1] + g6 * f2;
  vout[kipz] = vect[kipz] + g4 * vect[kipz] + g5 * hxp[2] + g6 * f3;

  return true;
}

} // namespace mch
} // namespace o2
