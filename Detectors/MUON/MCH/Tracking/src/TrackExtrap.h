// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackExtrap.h
/// \brief Definition of tools for track extrapolation
///
/// \author Philippe Pillot, Subatech

#ifndef ALICEO2_MCH_TRACKEXTRAP_H_
#define ALICEO2_MCH_TRACKEXTRAP_H_

#include <cstddef>

#include <TMatrixD.h>

namespace o2
{
namespace mch
{

class TrackParam;

/// Class holding tools for track extrapolation
class TrackExtrap
{
 public:
  // static class
  TrackExtrap() = delete;
  ~TrackExtrap() = delete;

  TrackExtrap(const TrackExtrap&) = delete;
  TrackExtrap& operator=(const TrackExtrap&) = delete;
  TrackExtrap(TrackExtrap&&) = delete;
  TrackExtrap& operator=(TrackExtrap&&) = delete;

  static void setField();

  /// Return true if the field is switched ON
  static bool isFieldON() { return sFieldON; }

  /// Switch to Runge-Kutta extrapolation v2
  static void useExtrapV2() { sExtrapV2 = true; }

  static double getImpactParamFromBendingMomentum(double bendingMomentum);
  static double getBendingMomentumFromImpactParam(double impactParam);

  static void linearExtrapToZ(TrackParam* trackParam, double zEnd);
  static void linearExtrapToZCov(TrackParam* trackParam, double zEnd, bool updatePropagator = false);

  static bool extrapToZ(TrackParam* trackParam, double zEnd);
  static bool extrapToZCov(TrackParam* trackParam, double zEnd, bool updatePropagator = false);

  static bool extrapToVertex(TrackParam* trackParam, double xVtx, double yVtx, double zVtx, double errXVtx, double errYVtx)
  {
    /// Extrapolate track parameters to vertex, corrected for multiple scattering and energy loss effects
    /// Add branson correction resolution and energy loss fluctuation to parameter covariances
    return extrapToVertex(trackParam, xVtx, yVtx, zVtx, errXVtx, errYVtx, true, true);
  }
  static bool extrapToVertexWithoutELoss(TrackParam* trackParam, double xVtx, double yVtx, double zVtx, double errXVtx, double errYVtx)
  {
    /// Extrapolate track parameters to vertex, corrected for multiple scattering effects only
    /// Add branson correction resolution to parameter covariances
    return extrapToVertex(trackParam, xVtx, yVtx, zVtx, errXVtx, errYVtx, true, false);
  }
  static bool extrapToVertexWithoutBranson(TrackParam* trackParam, double zVtx)
  {
    /// Extrapolate track parameters to vertex, corrected for energy loss effects only
    /// Add dispersion due to multiple scattering and energy loss fluctuation to parameter covariances
    return extrapToVertex(trackParam, 0., 0., zVtx, 0., 0., false, true);
  }
  static bool extrapToVertexUncorrected(TrackParam* trackParam, double zVtx)
  {
    /// Extrapolate track parameters to vertex without multiple scattering and energy loss corrections
    /// Add dispersion due to multiple scattering to parameter covariances
    return extrapToVertex(trackParam, 0., 0., zVtx, 0., 0., false, false);
  }

  static double getMCSAngle2(const TrackParam& param, double dZ, double x0);
  static void addMCSEffect(TrackParam* trackParam, double dZ, double x0);

  static void printNCalls();

 private:
  static bool extrapToVertex(TrackParam* trackParam, double xVtx, double yVtx, double zVtx,
                             double errXVtx, double errYVtx, bool correctForMCS, bool correctForEnergyLoss);

  static bool getAbsorberCorrectionParam(double trackXYZIn[3], double trackXYZOut[3], double pTotal,
                                         double& pathLength, double& f0, double& f1, double& f2,
                                         double& meanRho, double& totalELoss, double& sigmaELoss2);

  static void addMCSEffectInAbsorber(TrackParam* param, double signedPathLength, double f0, double f1, double f2);

  static double betheBloch(double pTotal, double pathLength, double rho, double atomicZ, double atomicZoverA);
  static double energyLossFluctuation(double pTotal, double pathLength, double rho, double atomicZoverA);

  static bool correctMCSEffectInAbsorber(TrackParam* param, double xVtx, double yVtx, double zVtx, double errXVtx, double errYVtx,
                                         double absZBeg, double pathLength, double f0, double f1, double f2);
  static void correctELossEffectInAbsorber(TrackParam* param, double eLoss, double sigmaELoss2);

  static void cov2CovP(const TMatrixD& param, TMatrixD& cov);
  static void covP2Cov(const TMatrixD& param, TMatrixD& covP);

  static void convertTrackParamForExtrap(TrackParam* trackParam, double forwardBackward, double* v3);
  static void recoverTrackParam(double* v3, double Charge, TrackParam* trackParam);

  static bool extrapToZRungekutta(TrackParam* trackParam, double zEnd);
  static bool extrapToZRungekuttaV2(TrackParam* trackParam, double zEnd);
  static bool extrapOneStepRungekutta(double charge, double step, const double* vect, double* vout);

  static constexpr double SMuMass = 0.105658;                         ///< Muon mass (GeV/c2)
  static constexpr double SAbsZBeg = -90.;                            ///< Position of the begining of the absorber (cm)
  static constexpr double SAbsZEnd = -505.;                           ///< Position of the end of the absorber (cm)
  static constexpr double SSimpleBPosition = -0.5 * (994.05 + 986.6); ///< Position of the dipole (cm)
  static constexpr double SSimpleBLength = 0.5 * (502.1 + 309.4);     ///< Length of the dipole (cm)
  static constexpr int SMaxStepNumber = 5000;                         ///< Maximum number of steps for track extrapolation
  static constexpr double SRungeKuttaMaxResidue = 0.002;              ///< Max z-distance to destination to stop the track extrap (cm)
  static constexpr double SRungeKuttaMaxResidueV2 = 0.01;             ///< Max z-distance to destination to stop the track extrap v2 (cm)
  /// Most probable value (GeV/c) of muon momentum in bending plane (used when B = 0)
  /// Needed to get some "reasonable" corrections for MCS and E loss even if B = 0
  static constexpr double SMostProbBendingMomentum = 2.;

  static bool sExtrapV2; ///< switch to Runge-Kutta extrapolation v2

  static double sSimpleBValue; ///< Magnetic field value at the centre
  static bool sFieldON;        ///< true if the field is switched ON

  static std::size_t sNCallExtrapToZCov; ///< number of times the method extrapToZCov(...) is called
  static std::size_t sNCallField;        ///< number of times the method Field(...) is called
};

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_TRACKEXTRAP_H_
