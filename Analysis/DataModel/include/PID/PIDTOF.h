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

#include "Rtypes.h"
// #include "PIDParamBase.h"

namespace o2::pid::tof
{
class EventTime
{
 public:
  EventTime() = default;
  ~EventTime() = default;

  void SetEvTime(Double_t evtime, Int_t i) { fT0event[i] = evtime; }
  float GetEvTime(Int_t i) const { return fT0event[i]; }
  float EvTimeReso(Int_t i) const { return fT0resolution[i]; }
  // void SetEvTime(Double_t evtime) { mEvTime = evtime; }
  // float GetEvTime(Float_t mom) const { return mEvTime; }
  // float EvTimeReso(Float_t mom) const { return mEvTimeReso; }

 private:
  float mEvTime = 0;
  float mEvTimeReso = 0;
  static const Int_t fNmomBins = 10;      /// Number of momentum bin
  static Float_t fmomBins[fNmomBins + 1]; /// Momentum bins
  Float_t fT0event[fNmomBins];            /// Evtime (best, T0, T0-TOF, ...) of the event as a function of p
  Float_t fT0resolution[fNmomBins];       /// Evtime (best, T0, T0-TOF, ...) resolution as a function of p
  // Float_t fPCutMin[fNmomBins + 1];        /// Min values for p bins
  Int_t fMaskT0[fNmomBins]; /// Mask withthe T0 used (0x1=T0-TOF,0x2=T0A,0x3=TOC) for p bins
};

class Param
{
 public:
  Param() = default;
  ~Param() = default;

  void SetTimeResolution(Float_t res) { mSigma = res; }
  Float_t GetTimeResolution() const { return mSigma; }

  // Tracking resolution for expected times
  void SetTrackParameter(Int_t ip, Float_t value)
  {
    if (ip >= 0 && ip < 4)
      mPar[ip] = value;
  };
  Float_t GetTrackParameter(Int_t ip)
  {
    if (ip >= 0 && ip < 4)
      return mPar[ip];
    else
      return -1.0;
  };

  Double_t GetExpectedSigma(Float_t mom, Float_t tof, Float_t evtimereso, Float_t massZ) const;
  Double_t GetNSigma(Float_t mom, Float_t time, Float_t exptime, Float_t evtime, Float_t evtimereso, Float_t mass) const;

 private:
  Double_t mSigma; /// intrinsic TOF resolution
                   // Float_t mPar[4]; /// parameter for expected time resolutions
  Float_t mPar[4] = {0.008, 0.008, 0.002, 40.0};
};

class Response
{
 public:
  Response() = default;
  ~Response() = default;

  // void SetMaxMismatchProbability(Double_t p) { fPmax = p; }
  // Double_t GetMaxMismatchProbability() const { return fPmax; }

  inline Double_t GetExpectedSigma(Float_t mom, Float_t tof, Float_t massZ) const { return mParam.GetExpectedSigma(mom, tof, mEventTime.EvTimeReso(mom), massZ); };

  // Double_t GetExpectedSigma(Float_t mom, Float_t tof, AliPID::EParticleType type) const;
  // Double_t GetExpectedSignal(const AliVTrack* track, AliPID::EParticleType type) const;

  // Double_t GetMismatchProbability(Double_t time, Double_t eta) const;

  void SetEventTime(EventTime& evtime) { mEventTime = evtime; }; // To set a particular event time object
  EventTime& GetEventTime() { return mEventTime; };              // To get the event time object

  //  private:
  EventTime mEventTime; /// Event time object
  Param mParam;         /// Parametrization of the TOF signal

  // static TF1* fTOFtailResponse;   // function to generate a TOF tail
  // static TH1F* fHmismTOF;         // TOF mismatch distribution
  // static TH1D* fHchannelTOFdistr; // TOF channel distance distribution
  // static TH1D* fHTOFtailResponse; // histogram to generate a TOF tail

  // ClassDef(TOF, 6) // TOF PID class
};

constexpr float kElectronMass = 5.10998909999999971e-04;
constexpr float kPionMass = 1.39569997787475586e-01f;
constexpr float kKaonMass = 4.93676990270614624e-01;
constexpr float kProtonMass = 9.38271999359130859e-01f;

float beta(float l, float t, float t0);
float betaerror(float l, float t, float t0, float sigmat = 80);
float expbeta(float p, float m);
float p(float eta, float signed1Pt);

} // namespace o2::pid::tof

#endif // O2_FRAMEWORK_PIDTOF_H_