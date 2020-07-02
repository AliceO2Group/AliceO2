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
///

#ifndef O2_FRAMEWORK_PIDTOF_H_
#define O2_FRAMEWORK_PIDTOF_H_

// ROOT includes
#include "Rtypes.h"
#include "TMath.h"

// O2 includes
#include "Framework/Logger.h"
#include "ReconstructionDataFormats/PID.h"

namespace o2::pid::tof
{

/// \brief Class to handle the event times available for PID
class EventTime
{
 public:
  EventTime() = default;
  ~EventTime() = default;

  /// Setter for the event time in momentum bin i
  void SetEvTime(int i, float evtime) { mEvTime[i] = evtime; }
  /// Setter for the event time resolution in momentum bin i
  void SetEvTimeReso(int i, float evtimereso) { mEvTimeReso[i] = evtimereso; }
  /// Setter for the event time mask in momentum bin i
  void SetEvTimeMask(int i, int mask) { mEvTimeMask[i] = mask; }
  /// Getter for the event time in momentum bin i
  float GetEvTime(int i) const { return mEvTime[i]; }
  /// Getter for the event time resolution in momentum bin i
  float GetEvTimeReso(int i) const { return mEvTimeReso[i]; }
  /// Getter for the momentum bin index
  uint GetMomBin(float mom) const;
  /// Getter for the event time for the momentum requested
  float GetEvTime(float mom) const { return mEvTime[GetMomBin(mom)]; }
  /// Getter for the event time resolution for the momentum requested
  float GetEvTimeReso(float mom) const { return mEvTimeReso[GetMomBin(mom)]; }

 private:
  static const uint mNmomBins = 1;                           /// Number of momentum bin
  static constexpr float mMomBins[mNmomBins + 1] = {0, 100}; /// Momentum bins
  float mEvTime[mNmomBins];                                  /// Evtime (best, T0, T0-TOF, ...) of the event as a function of p
  float mEvTimeReso[mNmomBins];                              /// Evtime (best, T0, T0-TOF, ...) resolution as a function of p
  int mEvTimeMask[mNmomBins];                                /// Mask withthe T0 used (0x1=T0-TOF,0x2=T0A,0x3=TOC) for p bins
};

/// \brief Class to handle the parametrization of the detector response
class Param
{
 public:
  Param() = default;
  ~Param() = default;

  /// Setter for expected time resolution
  void SetTimeResolution(float res) { mSigma = res; }
  /// Getter for expected time resolution
  float GetTimeResolution() const { return mSigma; }

  // Tracking resolution for expected times
  /// Setter for resolution parametrization
  void SetTrackParameter(uint ip, float value) { ip < mParDim ? mPar[ip] = value : 0.f; };
  /// Getter for resolution parametrization
  float GetTrackParameter(uint ip) { return ip < mParDim ? mPar[ip] : -999.f; };

  /// Getter for the expected resolution.
  /// Returns the expected sigma of the PID signal for the specified
  /// particle mass/Z.
  /// If the operation is not possible, return a negative value.
  double GetExpectedSigma(float mom, float tof, float evtimereso, float massZ) const;

 private:
  double mSigma; /// intrinsic TOF resolution
  static constexpr uint mParDim = 4;
  float mPar[mParDim] = {0.008, 0.008, 0.002, 40.0}; /// parameter for expected time resolutions
};

/// \brief Class to handle the the TOF detector response
class Response
{
 public:
  Response() = default;
  ~Response() = default;

  /// Updater for the TOF response to setup the track parameters
  /// i.e. sets the track of interest
  void UpdateTrack(float mom, float tofexpmom, float length, float tofsignal)
  {
    mMomentum = mom;
    mTOFExpMomentum = tofexpmom;
    mLength = length;
    mTOFSignal = tofsignal;
  };

  /// Setter for the event time information in the parametrization
  void SetEventTime(EventTime& evtime) { mEventTime = evtime; };
  /// Getter for the event time information in the parametrization
  EventTime GetEventTime() const { return mEventTime; };
  /// Setter for the resolution parametrization
  void SetParam(Param& evtime) { mParam = evtime; };
  /// Getter for the resolution parametrization
  Param GetParam() const { return mParam; };

  // TOF beta
  /// Computes the beta of a track given a length, a time measurement and an event time
  static float GetBeta(float length, float time, float evtime);
  /// Gets the beta for the track of interest
  float GetBeta() const { return GetBeta(mLength, mTOFSignal, mEventTime.GetEvTime(mMomentum)); }
  /// Computes the expected uncertainty on the beta measurement
  static float GetBetaExpectedSigma(float length, float time, float evtime, float sigmat = 80);
  /// Gets the expected uncertainty on the beta measurement of the track of interest
  float GetBetaExpectedSigma(float sigmat = 80) const { return GetBetaExpectedSigma(mLength, mTOFSignal, mEventTime.GetEvTime(mMomentum), sigmat); }
  /// Gets the expected beta for a given mass hypothesis (no energy loss taken into account)
  static float GetExpectedBeta(float mom, float mass);
  /// Gets the expected beta given the particle index (no energy loss taken into account)
  float GetExpectedBeta(o2::track::PID::ID id) const { return GetExpectedBeta(mMomentum, o2::track::PID::getMass2Z(id)); }
  /// Gets the number of sigmas with respect the approximate beta (no energy loss taken into account)
  float GetBetaNumberOfSigmas(o2::track::PID::ID id, float sigmat = 80) { return (GetBeta() - GetExpectedBeta(id)) / GetBetaExpectedSigma(sigmat); }

  // TOF expected times
  /// Computes the expected time of a track, given it TOF expected momentum
  static float ComputeExpectedTime(float tofexpmom, float length, float massZ);
  /// Gets the expected signal of the track of interest under the PID assumption
  float GetExpectedSignal(o2::track::PID::ID id) const { return ComputeExpectedTime(mTOFExpMomentum, mLength, o2::track::PID::getMass2Z(id)); }

  // Expected resolution
  /// Gets the expected resolution of the measurement
  float GetExpectedSigma(o2::track::PID::ID id) const { return mParam.GetExpectedSigma(mMomentum, mTOFSignal, mEventTime.GetEvTimeReso(mMomentum), o2::track::PID::getMass2Z(id)); }

  // Nsigma
  float GetNumberOfSigmas(o2::track::PID::ID id) const { return (mTOFSignal - mEventTime.GetEvTime(mMomentum) - GetExpectedSignal(id)) / GetExpectedSigma(id); }

  // double GetMismatchProbability(double time, double eta) const;
  // Utility values
  static constexpr float kCSPEED = TMath::C() * 1.0e2f * 1.0e-12f; /// Speed of light in TOF units (cm/ps)
 private:
  Param mParam; /// Parametrization of the TOF signal
  // Event of interest information
  EventTime mEventTime; /// Event time object
  // Track of interest information
  float mMomentum;       /// Momentum of the track of interest
  float mTOFExpMomentum; /// TOF expected momentum of the track of interest
  float mLength;         /// Track of interest integrated length
  float mTOFSignal;      /// Track of interest integrated length
};

} // namespace o2::pid::tof

#endif // O2_FRAMEWORK_PIDTOF_H_
