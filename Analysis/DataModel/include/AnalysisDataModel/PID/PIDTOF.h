// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   PIDTOF.h
/// \author Nicolo' Jacazio
/// \since  02/07/2020
/// \brief  Implementation of the TOF detector response for PID
///

#ifndef O2_FRAMEWORK_PIDTOF_H_
#define O2_FRAMEWORK_PIDTOF_H_

// ROOT includes
#include "Rtypes.h"
#include "TMath.h"

// O2 includes
#include "Framework/Logger.h"
#include "ReconstructionDataFormats/PID.h"
#include "AnalysisDataModel/PID/DetectorResponse.h"

namespace o2::pid::tof
{

// Utility values
static constexpr float kCSPEED = TMath::C() * 1.0e2f * 1.0e-12f; /// Speed of light in TOF units (cm/ps)

/// \brief Class to handle the the TOF detector response for the TOF beta measurement
template <typename Coll, typename Trck, o2::track::PID::ID id>
class Beta
{
 public:
  Beta() = default;
  ~Beta() = default;

  /// Computes the beta of a track given a length, a time measurement and an event time (in ps)
  static float GetBeta(const float length, const float tofSignal, const float collisionTime);

  /// Gets the beta for the track of interest
  float GetBeta(const Coll& col, const Trck& trk) const { return GetBeta(trk.length(), trk.tofSignal(), col.collisionTime() * 1000.f); }

  /// Computes the expected uncertainty on the beta measurement
  static float GetExpectedSigma(const float& length, const float& tofSignal, const float& collisionTime, const float& time_reso);

  /// Gets the expected uncertainty on the beta measurement of the track of interest
  float GetExpectedSigma(const Coll& col, const Trck& trk) const { return GetExpectedSigma(trk.length(), trk.tofSignal(), col.collisionTime() * 1000.f, mExpectedResolution); }

  /// Gets the expected beta for a given mass hypothesis (no energy loss taken into account)
  static float GetExpectedSignal(const float& mom, const float& mass);

  /// Gets the expected beta given the particle index (no energy loss taken into account)
  float GetExpectedSignal(const Coll& col, const Trck& trk) const { return GetExpectedSignal(trk.p(), o2::track::PID::getMass2Z(id)); }

  /// Gets the number of sigmas with respect the approximate beta (no energy loss taken into account)
  float GetSeparation(const Coll& col, const Trck& trk) const { return (GetBeta(col, trk) - GetExpectedSignal(col, trk)) / GetExpectedSigma(col, trk); }

  float mExpectedResolution = 80; /// Expected time resolution
};

//_________________________________________________________________________
template <typename Coll, typename Trck, o2::track::PID::ID id>
float Beta<Coll, Trck, id>::GetBeta(const float length, const float tofSignal, const float collisionTime)
{
  if (tofSignal <= 0) {
    return -999.f;
  }
  return length / (tofSignal - collisionTime) / kCSPEED;
}

//_________________________________________________________________________
template <typename Coll, typename Trck, o2::track::PID::ID id>
float Beta<Coll, Trck, id>::GetExpectedSigma(const float& length, const float& tofSignal, const float& collisionTime, const float& time_reso)
{
  if (tofSignal <= 0) {
    return -999.f;
  }
  return GetBeta(length, tofSignal, collisionTime) / (tofSignal - collisionTime) * time_reso;
}

//_________________________________________________________________________
template <typename Coll, typename Trck, o2::track::PID::ID id>
float Beta<Coll, Trck, id>::GetExpectedSignal(const float& mom, const float& mass)
{
  if (mom > 0) {
    return mom / TMath::Sqrt(mom * mom + mass * mass);
  }
  return 0;
}

/// \brief Class to handle the the TOF detector response for the expected time
template <typename Coll, typename Trck, o2::track::PID::ID id>
class ExpTimes
{
 public:
  ExpTimes() = default;
  ~ExpTimes() = default;

  /// Computes the expected time of a track, given it TOF expected momentum
  static float ComputeExpectedTime(const float& tofExpMom, const float& length, const float& massZ);

  /// Gets the expected signal of the track of interest under the PID assumption
  float GetExpectedSignal(const Coll& col, const Trck& trk) const { return ComputeExpectedTime(trk.tofExpMom() / kCSPEED, trk.length(), o2::track::PID::getMass2Z(id)); }

  /// Gets the expected resolution of the measurement
  float GetExpectedSigma(const DetectorResponse& response, const Coll& col, const Trck& trk) const;

  /// Gets the number of sigmas with respect the expected time
  float GetSeparation(const DetectorResponse& response, const Coll& col, const Trck& trk) const { return (trk.tofSignal() - col.collisionTime() * 1000.f - GetExpectedSignal(col, trk)) / GetExpectedSigma(response, col, trk); }
};

//_________________________________________________________________________
template <typename Coll, typename Trck, o2::track::PID::ID id>
float ExpTimes<Coll, Trck, id>::ComputeExpectedTime(const float& tofExpMom, const float& length, const float& massZ)
{
  const float energy = sqrt((massZ * massZ) + (tofExpMom * tofExpMom));
  return length * energy / (kCSPEED * tofExpMom);
}

//_________________________________________________________________________
template <typename Coll, typename Trck, o2::track::PID::ID id>
float ExpTimes<Coll, Trck, id>::GetExpectedSigma(const DetectorResponse& response, const Coll& col, const Trck& trk) const
{
  if (trk.tofSignal() <= 0) {
    return -999.f;
  }
  const float x[4] = {trk.p(), trk.tofSignal(), col.collisionTimeRes() * 1000.f, o2::track::PID::getMass2Z(id)};
  return response(response.kSigma, x);
  // return response(response.kSigma, const Coll& col, const Trck& trk, id);
}

} // namespace o2::pid::tof

#endif // O2_FRAMEWORK_PIDTOF_H_
