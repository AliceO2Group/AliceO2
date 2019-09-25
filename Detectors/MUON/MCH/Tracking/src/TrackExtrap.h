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

  static double getImpactParamFromBendingMomentum(double bendingMomentum);
  static double getBendingMomentumFromImpactParam(double impactParam);

  static void linearExtrapToZ(TrackParam* trackParam, double zEnd);
  static void linearExtrapToZCov(TrackParam* trackParam, double zEnd, bool updatePropagator = false);

  static bool extrapToZ(TrackParam* trackParam, double zEnd);
  static bool extrapToZCov(TrackParam* trackParam, double zEnd, bool updatePropagator = false);

  static double getMCSAngle2(const TrackParam& param, double dZ, double x0);
  static void addMCSEffect(TrackParam* trackParam, double dZ, double x0);

 private:
  static void convertTrackParamForExtrap(TrackParam* trackParam, double forwardBackward, double* v3);
  static void recoverTrackParam(double* v3, double Charge, TrackParam* trackParam);

  static bool extrapToZRungekutta(TrackParam* trackParam, double Z);
  static bool extrapOneStepRungekutta(double charge, double step, const double* vect, double* vout);

  static constexpr double SSimpleBPosition = -0.5 * (994.05 + 986.6); ///< Position of the dipole
  static constexpr double SSimpleBLength = 0.5 * (502.1 + 309.4);     ///< Length of the dipole
  static constexpr int SMaxStepNumber = 5000;                         ///< Maximum number of steps for track extrapolation
  static constexpr double SRungeKuttaMaxResidue = 0.002;              ///< Max z-distance to destination to stop the track extrap
  /// Most probable value (GeV/c) of muon momentum in bending plane (used when B = 0)
  /// Needed to get some "reasonable" corrections for MCS and E loss even if B = 0
  static constexpr double SMostProbBendingMomentum = 2.;

  static double sSimpleBValue; ///< Magnetic field value at the centre
  static bool sFieldON;        ///< true if the field is switched ON
};

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_TRACKEXTRAP_H_
