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

namespace o2
{

namespace tof
{

struct eventTimeContainer {
  eventTimeContainer(const float& e, const float& err) : eventTime{e}, eventTimeError{err} {};
  float eventTime = 0.f;                      /// Value of the event time
  float eventTimeError = 0.f;                 /// Uncertainty on the computed event time
  unsigned short eventTimeMultiplicity = 0.f; /// Track multiplicity used to compute the event time

  float sumweights = 0.f;       /// sum of weights of all track contributors
  std::vector<float> weights;   /// weights (1/sigma^2) associated to a track in event time computation, 0 if track not used
  std::vector<float> tracktime; /// eventtime provided by a single track
  void print()
  {
    LOG(info) << "eventTimeContainer " << eventTime << " +- " << eventTimeError << " sum of weights " << sumweights << " tracks used " << eventTimeMultiplicity;
  }
};

struct eventTimeTrack {
  eventTimeTrack() = default;
  eventTimeTrack(float tof, float expt[3], float expsigma[3]) : mSignal(tof)
  {
    for (int i = 0; i < 3; i++) {
      expTimes[i] = expt[i];
      expSigma[i] = expsigma[i];
    }
  }
  float tofSignal() const { return mSignal; }
  float tofExpSignalPi() const { return expTimes[0]; }
  float tofExpSignalKa() const { return expTimes[1]; }
  float tofExpSignalPr() const { return expTimes[2]; }
  float tofExpSigmaPi() const { return expSigma[0]; }
  float tofExpSigmaKa() const { return expSigma[1]; }
  float tofExpSigmaPr() const { return expSigma[2]; }
  float mSignal = 0.f;
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
int getStartTimeInSet(const std::vector<eventTimeTrack>& tracks, std::vector<int>& trackInSet, unsigned long& bestComb);
int getStartTimeInSetFast(const std::vector<eventTimeTrack>& tracks, std::vector<int>& trackInSet, unsigned long& bestComb);

template <typename trackTypeContainer,
          typename trackType,
          bool (*trackFilter)(const trackType&)>
eventTimeContainer evTimeMaker(const trackTypeContainer& tracks,
                               float diamond = 6.0 /* spread of primary verdex in cm */, bool isFast = false)
{
  static std::vector<eventTimeTrack> trkWork;
  trkWork.clear();
  static std::vector<int> trkIndex; // indexes of working tracks in the track original array
  trkIndex.clear();

  static float expt[3], expsigma[3];

  static eventTimeContainer result = {0, 0};

  // reset info
  float sigmaFill = diamond * 33.356409; // move from diamond (cm) to spread on event time (ps)
  result.weights.clear();
  result.tracktime.clear();
  result.eventTime = 0.;
  result.eventTimeError = sigmaFill;
  result.sumweights = 0.;

  for (auto track : tracks) { // Loop on tracks
    if (trackFilter(track)) { // Select tracks good for T0 computation
      expt[0] = track.tofExpSignalPi();
      expt[1] = track.tofExpSignalKa();
      expt[2] = track.tofExpSignalPr();
      expsigma[0] = track.tofExpSigmaPi();
      expsigma[1] = track.tofExpSigmaKa();
      expsigma[2] = track.tofExpSigmaPr();
      trkWork.emplace_back(track.tofSignal(), expt, expsigma);
      trkIndex.push_back(result.weights.size());
    }
    result.weights.push_back(0.);
    result.tracktime.push_back(0.);
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
                                        float diamond = 6.0 /* spread of primary verdex in cm */, bool isFast = false)
{
  static std::vector<eventTimeTrack> trkWork;
  trkWork.clear();
  static std::vector<int> trkIndex; // indexes of working tracks in the track original array
  trkIndex.clear();

  constexpr auto responsePi = response<trackType, o2::track::PID::Pion>();
  constexpr auto responseKa = response<trackType, o2::track::PID::Kaon>();
  constexpr auto responsePr = response<trackType, o2::track::PID::Proton>();

  static float expt[3], expsigma[3];

  static eventTimeContainer result = {0, 0};

  // reset info
  float sigmaFill = diamond * 33.356409; // move from diamond (cm) to spread on event time (ps)
  result.weights.clear();
  result.tracktime.clear();
  result.eventTime = 0.;
  result.eventTimeError = sigmaFill;
  result.sumweights = 0.;

  for (auto track : tracks) { // Loop on tracks
    if (trackFilter(track)) { // Select tracks good for T0 computation
      expt[0] = responsePi.GetExpectedSignal(track);
      expt[1] = responseKa.GetExpectedSignal(track);
      expt[2] = responsePr.GetExpectedSignal(track);
      expsigma[0] = responsePi.GetExpectedSigmaTracking(responseParameters, track);
      expsigma[1] = responseKa.GetExpectedSigmaTracking(responseParameters, track);
      expsigma[2] = responsePr.GetExpectedSigmaTracking(responseParameters, track);
      trkWork.emplace_back(track.tofSignal(), expt, expsigma);
      trkIndex.push_back(result.weights.size());
    }
    result.weights.push_back(0.);
    result.tracktime.push_back(0.);
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
