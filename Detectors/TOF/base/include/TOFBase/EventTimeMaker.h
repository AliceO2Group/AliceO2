// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   EventTimeMaker.h
/// \author Francesca Ercolessi francesca.ercolessi@cern.ch
/// \author Francesco Noferini francesco.noferini@cern.ch
/// \author Nicolò Jacazio nicolo.jacazio@cern.ch
/// \brief  Definition of the TOF event time maker

#ifndef ALICEO2_TOF_EVENTTIMEMAKER_H
#define ALICEO2_TOF_EVENTTIMEMAKER_H

#include "TRandom.h"
#include "TMath.h"
#include "ReconstructionDataFormats/PID.h"
#include "Framework/Logger.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{

namespace tof
{
struct EventTimeTOFParams : public o2::conf::ConfigurableParamHelper<EventTimeTOFParams> {
  float maxNsigma = 3.0;
  int maxNtracksInSet = 10;
  O2ParamDef(EventTimeTOFParams, "EventTimeTOF");
};

struct eventTimeContainer {
  eventTimeContainer(const float& e, const float& err, const float& diamond) : mEventTime{e}, mEventTimeError{err}, mDiamondSpread{diamond} {};
  double mEventTime = 0.f;                   /// Value of the event time
  double mEventTimeError = 0.f;              /// Uncertainty on the computed event time
  unsigned short mEventTimeMultiplicity = 0; /// Track multiplicity used to compute the event time

  double mSumOfWeights = 0.f;     /// sum of weights of all track contributors
  std::vector<float> mWeights;    /// weights (1/sigma^2) associated to a track in event time computation, 0 if track not used
  std::vector<double> mTrackTimes; /// eventtime provided by a single track
  float mDiamondSpread = 6.f;     /// spread of primary verdex in cm. Used when resetting the container to the default value

  // Aliases
  const double& eventTime = mEventTime;
  const double& eventTimeError = mEventTimeError;
  const unsigned short& eventTimeMultiplicity = mEventTimeMultiplicity;
  const double& sumweights = mSumOfWeights;
  const std::vector<float>& weights = mWeights;
  const std::vector<double>& tracktime = mTrackTimes;

  void reset()
  {
    // reset info
    mSumOfWeights = 0.;
    mWeights.clear();
    mTrackTimes.clear();
    mEventTime = 0.;
    mEventTimeError = mDiamondSpread * 33.356409f; // move from diamond spread (cm) to spread on event time (ps)
    mEventTimeMultiplicity = 0;
  }

  template <typename trackType,
            bool (*trackFilter)(const trackType&)>
  void removeBias(const trackType& track,
                  int& nTrackIndex /* index of the track to remove the bias */,
                  float& eventTimeValue,
                  float& eventTimeError,
                  const unsigned short& minimumMultiplicity = 2) const
  {
    double evTime = eventTimeValue;
    double evTimeRes = eventTimeError;
    removeBias<trackType, trackFilter>(track, nTrackIndex, evTime, evTimeRes);
    eventTimeValue = evTime;
    eventTimeError = evTimeRes;
  }

  template <typename trackType,
            bool (*trackFilter)(const trackType&)>
  void removeBias(const trackType& track,
                  int& nTrackIndex /* index of the track to remove the bias */,
                  double& eventTimeValue,
                  double& eventTimeError,
                  const unsigned short& minimumMultiplicity = 2) const
  {
    eventTimeValue = mEventTime;
    eventTimeError = mEventTimeError;
    if (!trackFilter(track)) { // Check if the track was usable for the event time
      nTrackIndex++;
      return;
    }
    if (mEventTimeMultiplicity <= minimumMultiplicity && mWeights[nTrackIndex] > 1E-6f) { // Check if a track was used for the event time and if the multiplicity is low
      eventTimeValue = 0.f;
      eventTimeError = mDiamondSpread * 33.356409f; // move from diamond (cm) to spread on event time (ps)
      LOG(debug) << mEventTimeMultiplicity << " <= " << minimumMultiplicity << " and " << mWeights[nTrackIndex] << ": track was used, setting " << eventTimeValue << " " << eventTimeError;
      nTrackIndex++;
      return;
    }
    // Remove the bias
    double sumw = 1.f / eventTimeError / eventTimeError;
    LOG(debug) << "sumw " << sumw;
    eventTimeValue *= sumw;
    eventTimeValue -= mWeights[nTrackIndex] * mTrackTimes[nTrackIndex];
    sumw -= mWeights[nTrackIndex];
    eventTimeValue /= sumw;
    eventTimeError = sqrt(1.f / sumw);
    nTrackIndex++;
  }

  void print()
  {
    LOG(info) << "eventTimeContainer " << mEventTime << " +- " << mEventTimeError << " sum of weights " << mSumOfWeights << " tracks used " << mEventTimeMultiplicity;
  }
};

struct eventTimeTrack {
  eventTimeTrack() = default;
  eventTimeTrack(double tof, float expt[3], float expsigma[3]) : mSignal(tof)
  {
    for (int i = 0; i < 3; i++) {
      expTimes[i] = expt[i];
      expSigma[i] = expsigma[i];
    }
  }
  double tofSignal() const { return mSignal; }
  float tofExpSignalPi() const { return expTimes[0]; }
  float tofExpSignalKa() const { return expTimes[1]; }
  float tofExpSignalPr() const { return expTimes[2]; }
  float tofExpSigmaPi() const { return expSigma[0]; }
  float tofExpSigmaKa() const { return expSigma[1]; }
  float tofExpSigmaPr() const { return expSigma[2]; }
  double mSignal = 0.f;
  float expTimes[3] = {0.f, 0.f, 0.f};
  float expSigma[3] = {999.f, 999.f, 999.f};
};

struct eventTimeTrackTest : eventTimeTrack {
  float tofChi2() const { return mTOFChi2; }
  float pt() const { return mPt; }
  float p() const { return mP; }
  float length() const { return mLength; }
  int masshypo() const { return mHypo; }
  float mTOFChi2 = -1.f;
  float mPt = 0.f;
  float mP = 0.f;
  float mLength = 0.f;
  int mHypo = 0;
};

void generateEvTimeTracks(std::vector<eventTimeTrackTest>& tracks, int ntracks, float evTime = 0.0);

template <typename trackType>
bool filterDummy(const trackType& tr)
{
  return (tr.tofChi2() >= 0 && tr.p() < 2.0);
} // accept all

void computeEvTime(const std::vector<eventTimeTrack>& tracks, const std::vector<int>& trkIndex, eventTimeContainer& evtime);
void computeEvTimeFast(const std::vector<eventTimeTrack>& tracks, const std::vector<int>& trkIndex, eventTimeContainer& evtime);
int getStartTimeInSet(const std::vector<eventTimeTrack>& tracks, std::vector<int>& trackInSet, unsigned long& bestComb, double refT0 = 0);
int getStartTimeInSetFast(const std::vector<eventTimeTrack>& tracks, std::vector<int>& trackInSet, unsigned long& bestComb);

template <typename trackTypeContainer,
          typename trackType,
          bool (*trackFilter)(const trackType&)>
eventTimeContainer evTimeMaker(const trackTypeContainer& tracks,
                               const float& diamond = 6.0 /* spread of primary verdex in cm */,
                               bool isFast = false)
{
  static std::vector<eventTimeTrack> trkWork;
  trkWork.clear();
  static std::vector<int> trkIndex; // indexes of working tracks in the track original array
  trkIndex.clear();

  static float expt[3], expsigma[3];

  static eventTimeContainer result = {0, 0, diamond};

  // reset info
  result.reset();

  for (auto track : tracks) { // Loop on tracks
    if (trackFilter(track)) { // Select tracks good for T0 computation
      expt[0] = track.tofExpSignalPi();
      expt[1] = track.tofExpSignalKa();
      expt[2] = track.tofExpSignalPr();
      expsigma[0] = track.tofExpSigmaPi();
      expsigma[1] = track.tofExpSigmaKa();
      expsigma[2] = track.tofExpSigmaPr();
      trkWork.emplace_back(track.tofSignal(), expt, expsigma);
      trkIndex.push_back(result.mWeights.size());
    }
    result.mWeights.push_back(0.);
    result.mTrackTimes.push_back(0.);
  }
  if (!isFast) {
    computeEvTime(trkWork, trkIndex, result);
  } else {
    computeEvTimeFast(trkWork, trkIndex, result);
  }
  return result;
}

template <typename trackTypeContainer,
          typename trackType,
          bool (*trackFilter)(const trackType&),
          template <typename T, o2::track::PID::ID> typename response,
          typename responseParametersType>
eventTimeContainer evTimeMakerFromParam(const trackTypeContainer& tracks,
                                        const responseParametersType& responseParameters,
                                        const float& diamond = 6.0 /* spread of primary verdex in cm */,
                                        bool isFast = false)
{
  static std::vector<eventTimeTrack> trkWork;
  trkWork.clear();
  static std::vector<int> trkIndex; // indexes of working tracks in the track original array
  trkIndex.clear();

  constexpr auto responsePi = response<trackType, o2::track::PID::Pion>();
  constexpr auto responseKa = response<trackType, o2::track::PID::Kaon>();
  constexpr auto responsePr = response<trackType, o2::track::PID::Proton>();

  static float expt[3], expsigma[3];

  static eventTimeContainer result = {0, 0, diamond};

  // reset info
  result.reset();

  for (auto track : tracks) { // Loop on tracks
    if (trackFilter(track)) { // Select tracks good for T0 computation
      expt[0] = responsePi.GetExpectedSignal(track);
      expt[1] = responseKa.GetExpectedSignal(track);
      expt[2] = responsePr.GetExpectedSignal(track);
      expsigma[0] = responsePi.GetExpectedSigmaTracking(responseParameters, track);
      expsigma[1] = responseKa.GetExpectedSigmaTracking(responseParameters, track);
      expsigma[2] = responsePr.GetExpectedSigmaTracking(responseParameters, track);
      trkWork.emplace_back(track.tofSignal(), expt, expsigma);
      trkIndex.push_back(result.mWeights.size());
    }
    result.mWeights.push_back(0.);
    result.mTrackTimes.push_back(0.);
  }
  if (!isFast) {
    computeEvTime(trkWork, trkIndex, result);
  } else {
    computeEvTimeFast(trkWork, trkIndex, result);
  }
  return result;
}
} // namespace tof
} // namespace o2

#endif /* ALICEO2_TOF_EVENTTIMEMAKER_H */
