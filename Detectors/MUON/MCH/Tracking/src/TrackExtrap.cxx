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
#include <TGeoManager.h>
#include <TGeoMaterial.h>
#include <TGeoNode.h>
#include <TGeoShape.h>
#include <TMath.h>

#include <FairMQLogger.h>

#include "TrackParam.h"

namespace o2
{
namespace mch
{

bool TrackExtrap::sExtrapV2 = false;
double TrackExtrap::sSimpleBValue = 0.;
bool TrackExtrap::sFieldON = false;
std::size_t TrackExtrap::sNCallExtrapToZCov = 0;
std::size_t TrackExtrap::sNCallField = 0;

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
  } else if (sExtrapV2) {
    return extrapToZRungekuttaV2(trackParam, zEnd);
  } else {
    return extrapToZRungekutta(trackParam, zEnd);
  }
}

//__________________________________________________________________________
bool TrackExtrap::extrapToZCov(TrackParam* trackParam, double zEnd, bool updatePropagator)
{
  /// Track parameters and their covariances extrapolated to the plane at "zEnd".
  /// On return, results from the extrapolation are updated in trackParam.

  ++sNCallExtrapToZCov;

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
      return false;
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

  return true;
}

//__________________________________________________________________________
bool TrackExtrap::extrapToVertex(TrackParam* trackParam, double xVtx, double yVtx, double zVtx,
                                 double errXVtx, double errYVtx, bool correctForMCS, bool correctForEnergyLoss)
{
  /// Main method for extrapolation to the vertex:
  /// Returns the track parameters and covariances resulting from the extrapolation of the current trackParam
  /// Changes parameters and covariances according to multiple scattering and energy loss corrections:
  /// if correctForMCS=true:  compute parameters using Branson correction and add correction resolution to covariances
  /// if correctForMCS=false: add parameter dispersion due to MCS in parameter covariances
  /// if correctForEnergyLoss=true:  correct parameters for energy loss and add energy loss fluctuation to covariances
  /// if correctForEnergyLoss=false: do nothing about energy loss

  if (trackParam->getZ() == zVtx) {
    return true; // nothing to be done if already at vertex
  }

  // Check the vertex position with respect to the absorber (spectro z<0)
  if (zVtx < SAbsZBeg) {
    if (zVtx < SAbsZEnd) {
      LOG(WARNING) << "Ending Z (" << zVtx << ") downstream the front absorber (zAbsorberEnd = " << SAbsZEnd << ")";
      return false;
    } else {
      LOG(WARNING) << "Ending Z (" << zVtx << ") inside the front absorber (" << SAbsZBeg << ", " << SAbsZEnd << ")";
      return false;
    }
  }

  // Check the track position with respect to the vertex and the absorber (spectro z<0)
  if (trackParam->getZ() > SAbsZEnd) {
    if (trackParam->getZ() > zVtx) {
      LOG(WARNING) << "Starting Z (" << trackParam->getZ() << ") upstream the vertex (zVtx = " << zVtx << ")";
      return false;
    } else if (trackParam->getZ() > SAbsZBeg) {
      LOG(WARNING) << "Starting Z (" << trackParam->getZ() << ") upstream the front absorber (zAbsorberBegin = " << SAbsZBeg << ")";
      return false;
    } else {
      LOG(WARNING) << "Starting Z (" << trackParam->getZ() << ") inside the front absorber (" << SAbsZBeg << ", " << SAbsZEnd << ")";
      return false;
    }
  }

  // Extrapolate track parameters (and covariances if any) to the end of the absorber
  if ((trackParam->hasCovariances() && !extrapToZCov(trackParam, SAbsZEnd)) ||
      (!trackParam->hasCovariances() && !extrapToZ(trackParam, SAbsZEnd))) {
    return false;
  }

  // Get absorber correction parameters assuming linear propagation in absorber
  double trackXYZOut[3] = {trackParam->getNonBendingCoor(), trackParam->getBendingCoor(), trackParam->getZ()};
  double trackXYZIn[3] = {0., 0., 0.};
  if (correctForMCS) { // assume linear propagation to the vertex
    trackXYZIn[2] = SAbsZBeg;
    trackXYZIn[0] = trackXYZOut[0] + (xVtx - trackXYZOut[0]) / (zVtx - trackXYZOut[2]) * (trackXYZIn[2] - trackXYZOut[2]);
    trackXYZIn[1] = trackXYZOut[1] + (yVtx - trackXYZOut[1]) / (zVtx - trackXYZOut[2]) * (trackXYZIn[2] - trackXYZOut[2]);
  } else { // or standard propagation without vertex constraint
    TrackParam trackParamIn(*trackParam);
    if (!extrapToZ(&trackParamIn, SAbsZBeg)) {
      return false;
    }
    trackXYZIn[0] = trackParamIn.getNonBendingCoor();
    trackXYZIn[1] = trackParamIn.getBendingCoor();
    trackXYZIn[2] = trackParamIn.getZ();
  }
  double pTot = trackParam->p();
  double pathLength(0.), f0(0.), f1(0.), f2(0.), meanRho(0.), totalELoss(0.), sigmaELoss2(0.);
  if (!getAbsorberCorrectionParam(trackXYZIn, trackXYZOut, pTot, pathLength, f0, f1, f2, meanRho, totalELoss, sigmaELoss2)) {
    return false;
  }

  // Compute track parameters and covariances at vertex according to correctForMCS and correctForEnergyLoss flags
  if (correctForMCS) {
    if (correctForEnergyLoss) {
      // Correct for multiple scattering and energy loss
      correctELossEffectInAbsorber(trackParam, 0.5 * totalELoss, 0.5 * sigmaELoss2);
      if (!correctMCSEffectInAbsorber(trackParam, xVtx, yVtx, zVtx, errXVtx, errYVtx, trackXYZIn[2], pathLength, f0, f1, f2)) {
        return false;
      }
      correctELossEffectInAbsorber(trackParam, 0.5 * totalELoss, 0.5 * sigmaELoss2);
    } else {
      // Correct for multiple scattering
      if (!correctMCSEffectInAbsorber(trackParam, xVtx, yVtx, zVtx, errXVtx, errYVtx, trackXYZIn[2], pathLength, f0, f1, f2)) {
        return false;
      }
    }
  } else {
    if (correctForEnergyLoss) {
      // Correct for energy loss add multiple scattering dispersion in covariance matrix
      correctELossEffectInAbsorber(trackParam, 0.5 * totalELoss, 0.5 * sigmaELoss2);
      addMCSEffectInAbsorber(trackParam, -pathLength, f0, f1, f2); // (spectro. (z<0))
      if (!extrapToZCov(trackParam, trackXYZIn[2])) {
        return false;
      }
      correctELossEffectInAbsorber(trackParam, 0.5 * totalELoss, 0.5 * sigmaELoss2);
      if (!extrapToZCov(trackParam, zVtx)) {
        return false;
      }
    } else {
      // add multiple scattering dispersion in covariance matrix
      addMCSEffectInAbsorber(trackParam, -pathLength, f0, f1, f2); // (spectro. (z<0))
      if (!extrapToZCov(trackParam, zVtx)) {
        return false;
      }
    }
  }

  return true;
}

//__________________________________________________________________________
bool TrackExtrap::getAbsorberCorrectionParam(double trackXYZIn[3], double trackXYZOut[3], double pTotal,
                                             double& pathLength, double& f0, double& f1, double& f2,
                                             double& meanRho, double& totalELoss, double& sigmaELoss2)
{
  /// Parameters used to correct for Multiple Coulomb Scattering and energy loss in absorber
  /// Calculated assuming a linear propagation from trackXYZIn to trackXYZOut (order is important)
  /// pathLength:  path length between trackXYZIn and trackXYZOut (cm)
  /// f0:          0th moment of z calculated with the inverse radiation-length distribution
  /// f1:          1st moment of z calculated with the inverse radiation-length distribution
  /// f2:          2nd moment of z calculated with the inverse radiation-length distribution
  /// meanRho:     average density of crossed material (g/cm3)
  /// totalELoss:  total energy loss in absorber
  /// sigmaELoss2: square of energy loss fluctuation in absorber

  // Check whether the geometry is available
  if (!gGeoManager) {
    LOG(WARNING) << "geometry is missing";
    return false;
  }

  // Initialize starting point and direction
  pathLength = TMath::Sqrt((trackXYZOut[0] - trackXYZIn[0]) * (trackXYZOut[0] - trackXYZIn[0]) +
                           (trackXYZOut[1] - trackXYZIn[1]) * (trackXYZOut[1] - trackXYZIn[1]) +
                           (trackXYZOut[2] - trackXYZIn[2]) * (trackXYZOut[2] - trackXYZIn[2]));
  if (pathLength < TGeoShape::Tolerance()) {
    LOG(WARNING) << "path length is too small";
    return false;
  }
  double b[3] = {(trackXYZOut[0] - trackXYZIn[0]) / pathLength, (trackXYZOut[1] - trackXYZIn[1]) / pathLength, (trackXYZOut[2] - trackXYZIn[2]) / pathLength};
  TGeoNode* currentnode = gGeoManager->InitTrack(trackXYZIn, b);
  if (!currentnode) {
    LOG(WARNING) << "starting point out of geometry";
    return false;
  }

  // loop over absorber slices and calculate absorber's parameters
  f0 = f1 = f2 = meanRho = totalELoss = 0.;
  double sigmaELoss(0.);
  double zB = trackXYZIn[2];
  double remainingPathLength = pathLength;
  do {

    // Get material properties
    TGeoMaterial* material = currentnode->GetVolume()->GetMedium()->GetMaterial();
    double rho = material->GetDensity(); // material density (g/cm3)
    double x0 = material->GetRadLen();   // radiation-length (cm-1)
    double atomicZ = material->GetZ();   // Z of material
    double atomicZoverA(0.);             // Z/A of material
    if (material->IsMixture()) {
      TGeoMixture* mixture = static_cast<TGeoMixture*>(material);
      double sum(0.);
      for (int iel = 0; iel < mixture->GetNelements(); ++iel) {
        sum += mixture->GetWmixt()[iel];
        atomicZoverA += mixture->GetWmixt()[iel] * mixture->GetZmixt()[iel] / mixture->GetAmixt()[iel];
      }
      atomicZoverA /= sum;
    } else {
      atomicZoverA = atomicZ / material->GetA();
    }

    // Get path length within this material
    gGeoManager->FindNextBoundary(remainingPathLength);
    double localPathLength = gGeoManager->GetStep() + 1.e-6;
    // Check if boundary within remaining path length. If so, make sure to cross the boundary to prepare the next step
    if (localPathLength >= remainingPathLength) {
      localPathLength = remainingPathLength;
    } else {
      currentnode = gGeoManager->Step();
      if (!currentnode) {
        LOG(WARNING) << "navigation failed";
        return false;
      }
      if (!gGeoManager->IsEntering()) {
        // make another small step to try to enter in new absorber slice
        gGeoManager->SetStep(0.001);
        currentnode = gGeoManager->Step();
        if (!gGeoManager->IsEntering() || !currentnode) {
          LOG(WARNING) << "navigation failed";
          return false;
        }
        localPathLength += 0.001;
      }
    }

    // calculate absorber's parameters
    double zE = b[2] * localPathLength + zB;
    double dzB = zB - trackXYZIn[2];
    double dzE = zE - trackXYZIn[2];
    f0 += localPathLength / x0;
    f1 += (dzE * dzE - dzB * dzB) / b[2] / b[2] / x0 / 2.;
    f2 += (dzE * dzE * dzE - dzB * dzB * dzB) / b[2] / b[2] / b[2] / x0 / 3.;
    meanRho += localPathLength * rho;
    totalELoss += betheBloch(pTotal, localPathLength, rho, atomicZ, atomicZoverA);
    sigmaELoss += energyLossFluctuation(pTotal, localPathLength, rho, atomicZoverA);

    // prepare next step
    zB = zE;
    remainingPathLength -= localPathLength;
  } while (remainingPathLength > TGeoShape::Tolerance());

  meanRho /= pathLength;
  sigmaELoss2 = sigmaELoss * sigmaELoss;

  return true;
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
  double pathLength = TMath::Abs(dZ) * TMath::Sqrt(1.0 + bendingSlope * bendingSlope + nonBendingSlope * nonBendingSlope);
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
void TrackExtrap::addMCSEffectInAbsorber(TrackParam* param, double signedPathLength, double f0, double f1, double f2)
{
  /// Add to the track parameter covariances the effects of multiple Coulomb scattering
  /// signedPathLength must have the sign of (zOut - zIn) where all other parameters are assumed to be given at zOut.

  // absorber related covariance parameters
  double bendingSlope = param->getBendingSlope();
  double nonBendingSlope = param->getNonBendingSlope();
  double inverseBendingMomentum = param->getInverseBendingMomentum();
  double alpha2 = 0.0136 * 0.0136 * inverseBendingMomentum * inverseBendingMomentum * (1.0 + bendingSlope * bendingSlope) /
                  (1.0 + bendingSlope * bendingSlope + nonBendingSlope * nonBendingSlope); // velocity = 1
  double pathLength = TMath::Abs(signedPathLength);
  double varCoor = alpha2 * (pathLength * pathLength * f0 - 2. * pathLength * f1 + f2);
  double covCorrSlope = TMath::Sign(1., signedPathLength) * alpha2 * (pathLength * f0 - f1);
  double varSlop = alpha2 * f0;

  // Set MCS covariance matrix
  TMatrixD newParamCov(param->getCovariances());
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
    double dqPxydSlopeX = inverseBendingMomentum * nonBendingSlope / (1. + nonBendingSlope * nonBendingSlope + bendingSlope * bendingSlope);
    double dqPxydSlopeY = -inverseBendingMomentum * nonBendingSlope * nonBendingSlope * bendingSlope /
                          (1. + bendingSlope * bendingSlope) / (1. + nonBendingSlope * nonBendingSlope + bendingSlope * bendingSlope);
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
  param->setCovariances(newParamCov);
}

//__________________________________________________________________________
double TrackExtrap::betheBloch(double pTotal, double pathLength, double rho, double atomicZ, double atomicZoverA)
{
  /// Returns the mean total momentum energy loss of muon with total momentum='pTotal'
  /// in the absorber layer of lenght='pathLength', density='rho', Z='atomicZ' and mean Z/A='atomicZoverA'
  /// This is the parameterization of the Bethe-Bloch formula inspired by Geant.

  static constexpr double mK = 0.307075e-3; // [GeV*cm^2/g]
  static constexpr double me = 0.511e-3;    // [GeV/c^2]
  static constexpr double x0 = 0.2 * 2.303; // density effect first junction point * 2.303
  static constexpr double x1 = 3. * 2.303;  // density effect second junction point * 2.303

  double bg = pTotal / SMuMass; // beta*gamma
  double bg2 = bg * bg;
  double maxT = 2 * me * bg2; // neglecting the electron mass

  // mean exitation energy (GeV)
  double mI = (atomicZ < 13) ? (12. * atomicZ + 7.) * 1.e-9 : (9.76 * atomicZ + 58.8 * TMath::Power(atomicZ, -0.19)) * 1.e-9;

  // density effect
  double x = TMath::Log(bg);
  double lhwI = TMath::Log(28.816 * 1e-9 * TMath::Sqrt(rho * atomicZoverA) / mI);
  double d2(0.);
  if (x > x1) {
    d2 = lhwI + x - 0.5;
  } else if (x > x0) {
    double r = (x1 - x) / (x1 - x0);
    d2 = lhwI + x - 0.5 + (0.5 - lhwI - x0) * r * r * r;
  }

  return pathLength * rho * (mK * atomicZoverA * (1 + bg2) / bg2 * (0.5 * TMath::Log(2 * me * bg2 * maxT / (mI * mI)) - bg2 / (1 + bg2) - d2));
}

//__________________________________________________________________________
double TrackExtrap::energyLossFluctuation(double pTotal, double pathLength, double rho, double atomicZoverA)
{
  /// Returns the total momentum energy loss fluctuation of muon with total momentum='pTotal'
  /// in the absorber layer of lenght='pathLength', density='rho', A='atomicA' and Z='atomicZ'

  static constexpr double mK = 0.307075e-3; // GeV.g^-1.cm^2

  double p2 = pTotal * pTotal;
  double beta2 = p2 / (p2 + SMuMass * SMuMass);

  double fwhm = 2. * mK * rho * pathLength * atomicZoverA / beta2; // FWHM of the energy loss Landau distribution

  return fwhm / TMath::Sqrt(8. * log(2.)); // gaussian: fwmh = 2 * srqt(2*ln(2)) * sigma (i.e. fwmh = 2.35 * sigma)
}

//__________________________________________________________________________
bool TrackExtrap::correctMCSEffectInAbsorber(TrackParam* param, double xVtx, double yVtx, double zVtx, double errXVtx, double errYVtx,
                                             double absZBeg, double pathLength, double f0, double f1, double f2)
{
  /// Correct parameters and corresponding covariances using Branson correction
  /// - input param are parameters and covariances at the end of absorber
  /// - output param are parameters and covariances at vertex
  /// Absorber correction parameters are supposed to be calculated at the current track z-position

  // Position of the Branson plane (spectro. (z<0))
  double zB = (f1 > 0.) ? absZBeg - f2 / f1 : 0.;

  // Add MCS effects to current parameter covariances (spectro. (z<0))
  addMCSEffectInAbsorber(param, -pathLength, f0, f1, f2);

  // Get track parameters and covariances in the Branson plane corrected for magnetic field effect
  if (!extrapToZCov(param, zVtx)) {
    return false;
  }
  linearExtrapToZCov(param, zB);

  // compute track parameters at vertex
  TMatrixD newParam(5, 1);
  newParam(0, 0) = xVtx;
  newParam(1, 0) = (param->getNonBendingCoor() - xVtx) / (zB - zVtx);
  newParam(2, 0) = yVtx;
  newParam(3, 0) = (param->getBendingCoor() - yVtx) / (zB - zVtx);
  newParam(4, 0) = param->getCharge() / param->p() *
                   TMath::Sqrt(1.0 + newParam(1, 0) * newParam(1, 0) + newParam(3, 0) * newParam(3, 0)) /
                   TMath::Sqrt(1.0 + newParam(3, 0) * newParam(3, 0));

  // Get covariances in (X, SlopeX, Y, SlopeY, q*PTot) coordinate system
  TMatrixD paramCovP(param->getCovariances());
  cov2CovP(param->getParameters(), paramCovP);

  // Get the covariance matrix in the (XVtx, X, YVtx, Y, q*PTot) coordinate system
  TMatrixD paramCovVtx(5, 5);
  paramCovVtx.Zero();
  paramCovVtx(0, 0) = errXVtx * errXVtx;
  paramCovVtx(1, 1) = paramCovP(0, 0);
  paramCovVtx(2, 2) = errYVtx * errYVtx;
  paramCovVtx(3, 3) = paramCovP(2, 2);
  paramCovVtx(4, 4) = paramCovP(4, 4);
  paramCovVtx(1, 3) = paramCovP(0, 2);
  paramCovVtx(3, 1) = paramCovP(2, 0);
  paramCovVtx(1, 4) = paramCovP(0, 4);
  paramCovVtx(4, 1) = paramCovP(4, 0);
  paramCovVtx(3, 4) = paramCovP(2, 4);
  paramCovVtx(4, 3) = paramCovP(4, 2);

  // Jacobian of the transformation (XVtx, X, YVtx, Y, q*PTot) -> (XVtx, SlopeXVtx, YVtx, SlopeYVtx, q*PTotVtx)
  TMatrixD jacob(5, 5);
  jacob.UnitMatrix();
  jacob(1, 0) = -1. / (zB - zVtx);
  jacob(1, 1) = 1. / (zB - zVtx);
  jacob(3, 2) = -1. / (zB - zVtx);
  jacob(3, 3) = 1. / (zB - zVtx);

  // Compute covariances at vertex in the (XVtx, SlopeXVtx, YVtx, SlopeYVtx, q*PTotVtx) coordinate system
  TMatrixD tmp(paramCovVtx, TMatrixD::kMultTranspose, jacob);
  TMatrixD newParamCov(jacob, TMatrixD::kMult, tmp);

  // Compute covariances at vertex in the (XVtx, SlopeXVtx, YVtx, SlopeYVtx, q/PyzVtx) coordinate system
  covP2Cov(newParam, newParamCov);

  // Set parameters and covariances at vertex
  param->setParameters(newParam);
  param->setZ(zVtx);
  param->setCovariances(newParamCov);

  return true;
}

//__________________________________________________________________________
void TrackExtrap::correctELossEffectInAbsorber(TrackParam* param, double eLoss, double sigmaELoss2)
{
  /// Correct parameters for energy loss and add energy loss fluctuation effect to covariances

  // Get parameter covariances in (X, SlopeX, Y, SlopeY, q*PTot) coordinate system
  TMatrixD newParamCov(param->getCovariances());
  cov2CovP(param->getParameters(), newParamCov);

  // Compute new parameters corrected for energy loss
  double p = param->p();
  double e = TMath::Sqrt(p * p + SMuMass * SMuMass);
  double eCorr = e + eLoss;
  double pCorr = TMath::Sqrt(eCorr * eCorr - SMuMass * SMuMass);
  double nonBendingSlope = param->getNonBendingSlope();
  double bendingSlope = param->getBendingSlope();
  param->setInverseBendingMomentum(param->getCharge() / pCorr *
                                   TMath::Sqrt(1.0 + nonBendingSlope * nonBendingSlope + bendingSlope * bendingSlope) /
                                   TMath::Sqrt(1.0 + bendingSlope * bendingSlope));

  // Add effects of energy loss fluctuation to covariances
  newParamCov(4, 4) += eCorr * eCorr / pCorr / pCorr * sigmaELoss2;

  // Get new parameter covariances in (X, SlopeX, Y, SlopeY, q/Pyz) coordinate system
  covP2Cov(param->getParameters(), newParamCov);

  // Set new parameter covariances
  param->setCovariances(newParamCov);
}

//__________________________________________________________________________
void TrackExtrap::cov2CovP(const TMatrixD& param, TMatrixD& cov)
{
  /// change coordinate system: (X, SlopeX, Y, SlopeY, q/Pyz) -> (X, SlopeX, Y, SlopeY, q*PTot)
  /// parameters (param) are given in the (X, SlopeX, Y, SlopeY, q/Pyz) coordinate system

  // charge * total momentum
  double qPTot = TMath::Sqrt(1. + param(1, 0) * param(1, 0) + param(3, 0) * param(3, 0)) /
                 TMath::Sqrt(1. + param(3, 0) * param(3, 0)) / param(4, 0);

  // Jacobian of the opposite transformation
  TMatrixD jacob(5, 5);
  jacob.UnitMatrix();
  jacob(4, 1) = qPTot * param(1, 0) / (1. + param(1, 0) * param(1, 0) + param(3, 0) * param(3, 0));
  jacob(4, 3) = -qPTot * param(1, 0) * param(1, 0) * param(3, 0) /
                (1. + param(3, 0) * param(3, 0)) / (1. + param(1, 0) * param(1, 0) + param(3, 0) * param(3, 0));
  jacob(4, 4) = -qPTot / param(4, 0);

  // compute covariances in new coordinate system
  TMatrixD tmp(cov, TMatrixD::kMultTranspose, jacob);
  cov.Mult(jacob, tmp);
}

//__________________________________________________________________________
void TrackExtrap::covP2Cov(const TMatrixD& param, TMatrixD& covP)
{
  /// change coordinate system: (X, SlopeX, Y, SlopeY, q*PTot) -> (X, SlopeX, Y, SlopeY, q/Pyz)
  /// parameters (param) are given in the (X, SlopeX, Y, SlopeY, q/Pyz) coordinate system

  // charge * total momentum
  double qPTot = TMath::Sqrt(1. + param(1, 0) * param(1, 0) + param(3, 0) * param(3, 0)) /
                 TMath::Sqrt(1. + param(3, 0) * param(3, 0)) / param(4, 0);

  // Jacobian of the transformation
  TMatrixD jacob(5, 5);
  jacob.UnitMatrix();
  jacob(4, 1) = param(4, 0) * param(1, 0) / (1. + param(1, 0) * param(1, 0) + param(3, 0) * param(3, 0));
  jacob(4, 3) = -param(4, 0) * param(1, 0) * param(1, 0) * param(3, 0) /
                (1. + param(3, 0) * param(3, 0)) / (1. + param(1, 0) * param(1, 0) + param(3, 0) * param(3, 0));
  jacob(4, 4) = -param(4, 0) / qPTot;

  // compute covariances in new coordinate system
  TMatrixD tmp(covP, TMatrixD::kMultTranspose, jacob);
  covP.Mult(jacob, tmp);
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
  /// Return false in case of failure and let the trackParam as they were when that happened
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
  while (TMath::Abs(residue) > SRungeKuttaMaxResidue) {

    dZ = zEnd - trackParam->getZ();
    // step length assuming linear trajectory
    step = dZ * TMath::Sqrt(1.0 + trackParam->getBendingSlope() * trackParam->getBendingSlope() +
                            trackParam->getNonBendingSlope() * trackParam->getNonBendingSlope());
    convertTrackParamForExtrap(trackParam, forwardBackward, v3);

    do { // reduce step length while zEnd overstepped
      if (++stepNumber > SMaxStepNumber) {
        LOG(WARNING) << "Too many trials";
        return false;
      }
      step = TMath::Abs(step);
      if (!extrapOneStepRungekutta(chargeExtrap, step, v3, v3New)) {
        return false;
      }
      residue = zEnd - v3New[2];
      step *= dZ / (v3New[2] - trackParam->getZ());
    } while (residue * dZ < 0 && TMath::Abs(residue) > SRungeKuttaMaxResidue);

    if (v3New[5] * v3[5] < 0) { // the track turned around
      LOG(WARNING) << "The track turned around";
      return false;
    }

    recoverTrackParam(v3New, chargeExtrap * forwardBackward, trackParam);
  }

  // terminate the extropolation with a straight line up to the exact "zEnd" value
  trackParam->setNonBendingCoor(trackParam->getNonBendingCoor() + residue * trackParam->getNonBendingSlope());
  trackParam->setBendingCoor(trackParam->getBendingCoor() + residue * trackParam->getBendingSlope());
  trackParam->setZ(zEnd);

  return true;
}

//__________________________________________________________________________
bool TrackExtrap::extrapToZRungekuttaV2(TrackParam* trackParam, double zEnd)
{
  /// Track parameter extrapolation to the plane at "Z" using Rungekutta algorithm.
  /// On return, the track parameters resulting from the extrapolation are updated in trackParam.
  /// Return false in case of failure and let the trackParam as they were when that happened

  if (trackParam->getZ() == zEnd) {
    return true; // nothing to be done if same Z
  }

  double residue = zEnd - trackParam->getZ();
  double forwardBackward = (residue < 0) ? 1. : -1.; // +1 if forward, -1 if backward
  double v3[7] = {0.};
  convertTrackParamForExtrap(trackParam, forwardBackward, v3);
  double charge = TMath::Sign(double(1.), trackParam->getInverseBendingMomentum());

  // Extrapolation loop (until within tolerance or the track turn around)
  double v3New[7] = {0.};
  int stepNumber = 0;
  while (true) {

    if (++stepNumber > SMaxStepNumber) {
      LOG(WARNING) << "Too many trials";
      return false;
    }

    // step length assuming linear trajectory
    double slopeX = v3[3] / v3[5];
    double slopeY = v3[4] / v3[5];
    double step = TMath::Abs(residue) * TMath::Sqrt(1.0 + slopeX * slopeX + slopeY * slopeY);

    if (!extrapOneStepRungekutta(forwardBackward * charge, step, v3, v3New)) {
      return false;
    }

    if (v3New[5] * v3[5] < 0) {
      LOG(WARNING) << "The track turned around";
      return false;
    }

    residue = zEnd - v3New[2];
    if (TMath::Abs(residue) < SRungeKuttaMaxResidueV2) {
      break;
    }

    for (int i = 0; i < 7; ++i) {
      v3[i] = v3New[i];
    }

    // invert the sens of propagation if the track went too far
    if (forwardBackward * residue > 0) {
      forwardBackward = -forwardBackward;
      v3[3] = -v3[3];
      v3[4] = -v3[4];
      v3[5] = -v3[5];
    }
  }

  recoverTrackParam(v3New, charge, trackParam);

  // terminate the extrapolation with a straight line up to the exact "zEnd" value
  trackParam->setNonBendingCoor(trackParam->getNonBendingCoor() + residue * trackParam->getNonBendingSlope());
  trackParam->setBendingCoor(trackParam->getBendingCoor() + residue * trackParam->getBendingSlope());
  trackParam->setZ(zEnd);

  return true;
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
    ++sNCallField;

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
    ++sNCallField;

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
    ++sNCallField;

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
    LOG(WARNING) << "magnetic field at (" << xyzt[0] << ", " << xyzt[1] << ", " << xyzt[2] << ") = " << f4 << ": giving up";
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

//__________________________________________________________________________
void TrackExtrap::printNCalls()
{
  /// Print the number of times some methods are called
  LOG(INFO) << "number of times extrapToZCov() is called = " << sNCallExtrapToZCov;
  LOG(INFO) << "number of times Field() is called = " << sNCallField;
}

} // namespace mch
} // namespace o2
