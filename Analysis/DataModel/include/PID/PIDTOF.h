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
// #include "PIDParamBase.h"

// O2 includes
#include "ReconstructionDataFormats/PID.h"

namespace o2::pid::tof
{
class EventTime
{
 public:
  EventTime() = default;
  ~EventTime() = default;

  void SetEvTime(float evtime, int i) { mEvTime[i] = evtime; }
  void SetEvTimeReso(float evtimereso, int i) { mEvTimeReso[i] = evtimereso; }
  void SetEvTimeMask(int mask, int i) { mEvTimeMask[i] = mask; }
  float GetEvTime(int i) const { return mEvTime[i]; }
  float GetEvTimeReso(int i) const { return mEvTimeReso[i]; }
  int GetMomBin(float mom) const
  {
    for (int i = 0; i < fNmomBins; i++)
      if (abs(mom) < fmomBins[i + 1])
        return i;
    return fNmomBins;
  }
  float GetEvTime(float mom) const { return mEvTime[GetMomBin(mom)]; }
  float EvTimeReso(float mom) const { return mEvTimeReso[GetMomBin(mom)]; }

 private:
  static const int fNmomBins = 10;      /// Number of momentum bin
  static float fmomBins[fNmomBins + 1]; /// Momentum bins
  float mEvTime[fNmomBins];             /// Evtime (best, T0, T0-TOF, ...) of the event as a function of p
  float mEvTimeReso[fNmomBins];         /// Evtime (best, T0, T0-TOF, ...) resolution as a function of p
  int mEvTimeMask[fNmomBins];           /// Mask withthe T0 used (0x1=T0-TOF,0x2=T0A,0x3=TOC) for p bins
};

class Param
{
 public:
  Param() = default;
  ~Param() = default;

  void SetTimeResolution(float res) { mSigma = res; }
  float GetTimeResolution() const { return mSigma; }

  // Tracking resolution for expected times
  void SetTrackParameter(Int_t ip, float value)
  {
    if (ip >= 0 && ip < 4)
      mPar[ip] = value;
  };
  float GetTrackParameter(Int_t ip)
  {
    if (ip >= 0 && ip < 4)
      return mPar[ip];
    else
      return -1.0;
  };

  double GetExpectedSigma(float mom, float tof, float evtimereso, float massZ) const;
  // double GetNSigma(float mom, float time, float exptime, float evtime, float evtimereso, float mass) const;

 private:
  double mSigma; /// intrinsic TOF resolution
                 // float mPar[4]; /// parameter for expected time resolutions
  float mPar[4] = {0.008, 0.008, 0.002, 40.0};
};

class Response
{
 public:
  Response() = default;
  ~Response() = default;

  // void SetMaxMismatchProbability(double p) { fPmax = p; }
  // double GetMaxMismatchProbability() const { return fPmax; }

  void InitResponse(float mom, float tofexpmom, float length, float tofsignal)
  {
    mMomentum = mom;
    mTOFExpMomentum = tofexpmom;
    mLength = length;
    mTOFSignal = tofsignal;
  };
  inline double GetExpectedSigma(o2::track::PID::ID id) const { return mParam.GetExpectedSigma(mMomentum, mTOFSignal, mEventTime.EvTimeReso(mMomentum), o2::track::PID::getMass2Z(id)); };

  // double GetExpectedSigma(float mom, float tof, AliPID::EParticleType type) const;
  // double GetExpectedSignal(const AliVTrack* track, AliPID::EParticleType type) const;
  // double GetMismatchProbability(double time, double eta) const;

  void SetEventTime(EventTime& evtime) { mEventTime = evtime; }; /// To set a particular event time object
  EventTime GetEventTime() const { return mEventTime; };         /// To get the event time object

  void SetParam(Param& evtime) { mParam = evtime; }; /// To set a particular parametrization time object
  Param GetParam() const { return mParam; };         /// To get the parametrization time object

  // TOF beta
  static float GetBeta(float length, float time, float evtime);
  inline float GetBeta() const { return GetBeta(mLength, mTOFSignal, mEventTime.GetEvTime(mMomentum)); };
  static float GetBetaExpectedSigma(float length, float time, float evtime, float sigmat = 80);
  inline float GetBetaExpectedSigma(float sigmat = 80) const { return GetBetaExpectedSigma(mLength, mTOFSignal, mEventTime.GetEvTime(mMomentum), sigmat); };
  static float GetExpectedBeta(float mom, float mass);
  inline float GetExpectedBeta(o2::track::PID::ID id) const { return GetExpectedBeta(mMomentum, o2::track::PID::getMass2Z(id)); };

  // TOF expected times
  static float ComputeExpectedTime(float tofexpmom, float length, float massZ);
  float GetExpectedSignal() const;

 private:
  Param mParam; /// Parametrization of the TOF signal
  // Event of interest information
  EventTime mEventTime; /// Event time object
  // Track of interest information
  float mMomentum;       /// Momentum of the track of interest
  float mTOFExpMomentum; /// TOF expected momentum of the track of interest
  float mLength;         /// Track of interest integrated length
  float mTOFSignal;      /// Track of interest integrated length

  static constexpr float kCSPEED = TMath::C() * 1.0e2f / 1.0e-12f; /// Speed of light in TOF units (cm/ps)

  // static TF1* fTOFtailResponse;   // function to generate a TOF tail
  // static TH1F* fHmismTOF;         // TOF mismatch distribution
  // static TH1D* fHchannelTOFdistr; // TOF channel distance distribution
  // static TH1D* fHTOFtailResponse; // histogram to generate a TOF tail

  // ClassDef(TOF, 6) // TOF PID class
};

} // namespace o2::pid::tof

#endif // O2_FRAMEWORK_PIDTOF_H_
