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

///
/// \file   EventTimeMaker.cxx
/// \author Francesca Ercolessi francesca.ercolessi@cern.ch
/// \author Francesco Noferini francesco.noferini@cern.ch
/// \author Nicolò Jacazio nicolo.jacazio@cern.ch
/// \brief  Implementation of the TOF event time maker
///

#include "TRandom.h"
#include "TMath.h"
#include "TOFBase/EventTimeMaker.h"
#include "TOFBase/Geo.h"

O2ParamImpl(o2::tof::EventTimeTOFParams);

namespace o2
{

namespace tof
{

int eventTimeContainer::maxNtracksInSet = -1;
// usefull constants
constexpr unsigned long combinatorial[20] = {1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147, 531441, 1594323, 4782969, 14348907, 43046721, 129140163, 387420489, 1162261467};
//---------------

void computeEvTime(const std::vector<eventTimeTrack>& tracks, const std::vector<int>& trkIndex, eventTimeContainer& evtime)
{
  static constexpr int maxNumberOfSets = 200;
  static constexpr float weightLimit = 1E-6; // Limit in the weights

  const int ntracks = tracks.size();
  LOG(debug) << "For the collision time using " << ntracks << " tracks";

  // find the int BC as the more frequent BC
  std::map<int, int> bcCand;
  for (const auto& trk : tracks) {
    int bc = int(trk.tofSignal() * o2::tof::Geo::BC_TIME_INPS_INV);
    bcCand[bc]++;
  }

  int maxcount = 0;
  int maxbc = 0;
  for (const auto& [bc, count] : bcCand) {
    if (count > maxcount) {
      maxbc = bc;
      maxcount = count;
    }
  }

  double t0fill = maxbc * o2::tof::Geo::BC_TIME_INPS;

  //  LOG(info) << "Int bc candidate from TOF : " << maxbc << " - time_in_ps = " << t0fill;

  if (ntracks < 2) { // at least 2 tracks required
    LOG(debug) << "Skipping event because at least 2 tracks are required";
    return;
  }

  const int maxNtrackInSet = evtime.getMaxNtracksInSet();
  int hypo[maxNtrackInSet];

  int nmaxtracksinset = ntracks > 22 ? 6 : maxNtrackInSet; // max number of tracks in a set for event time computation
  // int nmaxtracksinset = maxNtrackInSet;

  LOG(debug) << "nmaxtracksinset " << nmaxtracksinset;
  int ntracksinset = std::min(ntracks, nmaxtracksinset);
  LOG(debug) << "ntracksinset " << ntracksinset;

  int nset = ((ntracks - 1) / ntracksinset) + 1;
  LOG(debug) << "nset " << nset;
  int ntrackUsable = ntracks;
  LOG(debug) << "ntrackUsable " << ntrackUsable;

  if (nset > maxNumberOfSets) {
    nset = maxNumberOfSets;
    LOG(debug) << "resetting nset " << nset;
    ntrackUsable = nmaxtracksinset * nset;
    LOG(debug) << "resetting ntrackUsable " << ntrackUsable;
  }

  // list of tracks in set
  std::vector<int> trackInSet[maxNumberOfSets];

  for (int i = 0; i < ntrackUsable; i++) {
    int iset = i % nset;

    trackInSet[iset].push_back(i);
  }

  int status;
  // compute event time for each set
  for (int iset = 0; iset < nset; iset++) {
    LOG(debug) << "iset " << iset << " has size " << trackInSet[iset].size();
    unsigned long bestComb = 0;
    while (!(status = getStartTimeInSet(tracks, trackInSet[iset], bestComb, t0fill))) {
      ;
    }
    if (status == 1) {
      int ntracks = trackInSet[iset].size();
      // set the best in set
      for (int itrk = 0; itrk < ntracks; itrk++) {
        hypo[itrk] = bestComb % 3;
        bestComb /= 3;

        int index = trkIndex[trackInSet[iset][itrk]];
        const eventTimeTrack& ctrack = tracks[trackInSet[iset][itrk]];
        LOG(debug) << "Using hypothesis: " << hypo[itrk] << " tofSignal: " << ctrack.mSignal << " exp. time: " << ctrack.expTimes[hypo[itrk]] << " exp. sigma: " << ctrack.expSigma[hypo[itrk]];
        LOG(debug) << "0= " << ctrack.expTimes[0] << " +- " << ctrack.expSigma[0] << " 1= " << ctrack.expTimes[1] << " +- " << ctrack.expSigma[1] << " 2= " << ctrack.expTimes[2] << " +- " << ctrack.expSigma[2];

        evtime.mWeights[index] = 1. / (ctrack.expSigma[hypo[itrk]] * ctrack.expSigma[hypo[itrk]]);
        evtime.mTrackTimes[index] = ctrack.mSignal - ctrack.expTimes[hypo[itrk]];
      }
    }
    LOG(debug) << "iset " << iset << " did not have good status";
  } // end loop in set

  // do average among all tracks
  int worse = 0;
  int nRemoved = 0;
  double finalTime = 0, allweights = 0;
  int ntrackUsed = 0;
  while (worse > -1) {
    float errworse = EventTimeTOFParams::Instance().maxNsigma;
    finalTime = 0, allweights = 0;
    ntrackUsed = 0;
    worse = -1;
    for (int i = 0; i < evtime.mWeights.size(); i++) {
      if (evtime.mWeights[i] < weightLimit) {
        continue;
      }
      ntrackUsed++;
      allweights += evtime.mWeights[i];
      finalTime += evtime.mTrackTimes[i] * evtime.mWeights[i];
    }

    double averageBest = finalTime / allweights;
    for (int i = 0; i < evtime.mWeights.size(); i++) {
      if (evtime.mWeights[i] < weightLimit) {
        continue;
      }
      float err = evtime.mTrackTimes[i] - averageBest;
      err *= sqrt(evtime.mWeights[i]);
      err = fabs(err);
      if (err > errworse) {
        errworse = err;
        worse = i;
      }
    }
    if (worse > -1) { // remove track
                      //      LOG(info) << "err " << errworse;
      evtime.mWeights[worse] = 0;
      nRemoved++;
    }
  }

  LOG(debug) << "Removed " << nRemoved << " tracks from start time calculation";

  if (allweights < weightLimit) {
    LOG(debug) << "Skipping because allweights " << allweights << " are lower than " << weightLimit;
    return;
  }

  evtime.mEventTime = finalTime / allweights;
  evtime.mEventTimeError = sqrt(1. / allweights);
  evtime.mEventTimeMultiplicity = ntrackUsed;
}

void computeEvTimeFast(const std::vector<eventTimeTrack>& tracks, const std::vector<int>& trkIndex, eventTimeContainer& evtime)
{
  static constexpr int maxNumberOfSets = 200;
  static constexpr float weightLimit = 1E-6; // Limit in the weights

  const int ntracks = tracks.size();
  LOG(debug) << "For the collision time using " << ntracks;

  if (ntracks < 2) { // at least 2 tracks required
    LOG(debug) << "Skipping event because at least 2 tracks are required";
    return;
  }
  const int maxNtrackInSet = evtime.getMaxNtracksInSet();

  int hypo[maxNtrackInSet];

  int nmaxtracksinset = ntracks > 22 ? 6 : maxNtrackInSet; // max number of tracks in a set for event time computation
  int ntracksinset = std::min(ntracks, nmaxtracksinset);

  int nset = ((ntracks - 1) / ntracksinset) + 1;
  int ntrackUsable = ntracks;

  if (nset > maxNumberOfSets) {
    nset = maxNumberOfSets;
    ntrackUsable = nmaxtracksinset * nset;
  }

  // list of tracks in set
  std::vector<int> trackInSet[maxNumberOfSets];

  for (int i = 0; i < ntrackUsable; i++) {
    int iset = i % nset;

    trackInSet[iset].push_back(i);
  }

  int status;
  // compute event time for each set
  for (int iset = 0; iset < nset; iset++) {
    unsigned long bestComb = 0;
    while (!(status = getStartTimeInSetFast(tracks, trackInSet[iset], bestComb))) {
      ;
    }
    if (status == 1) {
      int ntracks = trackInSet[iset].size();
      // set the best in set
      for (int itrk = 0; itrk < ntracks; itrk++) {
        hypo[itrk] = bestComb % 3;
        bestComb /= 3;

        int index = trkIndex[trackInSet[iset][itrk]];
        const eventTimeTrack& ctrack = tracks[trackInSet[iset][itrk]];
        LOG(debug) << "Using hypothesis: " << hypo[itrk] << " tofSignal: " << ctrack.mSignal << " exp. time: " << ctrack.expTimes[hypo[itrk]] << " exp. sigma: " << ctrack.expSigma[hypo[itrk]];
        LOG(debug) << "0= " << ctrack.expTimes[0] << " +- " << ctrack.expSigma[0] << " 1= " << ctrack.expTimes[1] << " +- " << ctrack.expSigma[1] << " 2= " << ctrack.expTimes[2] << " +- " << ctrack.expSigma[2];

        evtime.mWeights[index] = 1. / (ctrack.expSigma[hypo[itrk]] * ctrack.expSigma[hypo[itrk]]);
        evtime.mTrackTimes[index] = ctrack.mSignal - ctrack.expTimes[hypo[itrk]];
      }
    }
  } // end loop in set

  // do average among all tracks
  double finalTime = 0, allweights = 0;
  int ntrackUsed = 0;
  for (int i = 0; i < evtime.mWeights.size(); i++) {
    if (evtime.mWeights[i] < weightLimit) {
      continue;
    }
    ntrackUsed++;
    allweights += evtime.mWeights[i];
    finalTime += evtime.mTrackTimes[i] * evtime.mWeights[i];
  }

  if (allweights < weightLimit) {
    LOG(debug) << "Skipping because allweights " << allweights << " are lower than " << weightLimit;
    return;
  }

  evtime.mEventTime = finalTime / allweights;
  evtime.mEventTimeError = sqrt(1. / allweights);
  evtime.mEventTimeMultiplicity = ntrackUsed;
}

int getStartTimeInSet(const std::vector<eventTimeTrack>& tracks, std::vector<int>& trackInSet, unsigned long& bestComb, double refT0)
{
  const auto& params = EventTimeTOFParams::Instance();
  float errworse = params.maxNsigma;
  const int maxNtrackInSet = eventTimeContainer::getMaxNtracksInSet();
  float chi2, chi2best;
  double averageBest = 0;
  int hypo[maxNtrackInSet];
  double starttime[maxNtrackInSet], weighttime[maxNtrackInSet];

  chi2best = 10000;
  int ntracks = trackInSet.size();
  LOG(debug) << "Computing the start time in set with " << ntracks << " tracks";

  if (ntracks < 3) {
    LOG(debug) << "Not enough tracks!";
    return 2; // no event time in the set
  }

  unsigned long ncomb = combinatorial[ntracks];
  for (unsigned long comb = 0; comb < ncomb; comb++) {
    unsigned long curr = comb;

    int ngood = 0;
    double average = 0;
    double sumweights = 0;
    // get track info in the set for current combination
    for (int itrk = 0; itrk < ntracks; itrk++) {
      hypo[itrk] = curr % 3;
      curr /= 3;
      const eventTimeTrack& ctrack = tracks[trackInSet[itrk]];
      starttime[itrk] = ctrack.mSignal - ctrack.expTimes[hypo[itrk]];

      if (fabs(starttime[itrk] - refT0) < 2000) { // otherwise time inconsistent with the int BC
        weighttime[itrk] = 1. / (ctrack.expSigma[hypo[itrk]] * ctrack.expSigma[hypo[itrk]]);
        average += starttime[itrk] * weighttime[itrk];
        sumweights += weighttime[itrk];
        ngood++;
      }
    }

    average /= sumweights;

    // compute chi2
    chi2 = 0;
    double deltat;
    for (int itrk = 0; itrk < ntracks; itrk++) {
      deltat = starttime[itrk] - average;
      chi2 += deltat * deltat * weighttime[itrk];
    }
    chi2 /= (ngood - 1);

    if (chi2 < chi2best) {
      bestComb = comb;
      chi2best = chi2;
      averageBest = average;
    }
  } // end loop in combinations

  int worse = -1;
  // check the best combination
  unsigned long curr = bestComb;
  for (int itrk = 0; itrk < ntracks; itrk++) {
    hypo[itrk] = curr % 3;
    curr /= 3;

    const eventTimeTrack& ctrack = tracks[trackInSet[itrk]];
    float err = ctrack.mSignal - ctrack.expTimes[hypo[itrk]] - averageBest;
    err /= ctrack.expSigma[hypo[itrk]];
    err = fabs(err);
    if (err > errworse) {
      errworse = err;
      worse = itrk;
    }
  }

  if (worse > -1) {
    const eventTimeTrack& ctrack = tracks[trackInSet[worse]];
    // remove the track and try again
    trackInSet.erase(trackInSet.begin() + worse);
    return 0;
  }

  return 1; // good event time in the set
}

int getStartTimeInSetFast(const std::vector<eventTimeTrack>& tracks, std::vector<int>& trackInSet, unsigned long& bestComb)
{
  const int maxNtrackInSet = eventTimeContainer::getMaxNtracksInSet();
  float chi2, chi2best;
  double averageBest = 0;
  int hypo[maxNtrackInSet];
  double starttime[maxNtrackInSet], weighttime[maxNtrackInSet];

  chi2best = 10000;
  int ntracks = trackInSet.size();

  if (ntracks < 3) {
    return 2; // no event time in the set
  }

  unsigned long ncomb = combinatorial[ntracks];

  unsigned long comb = 0; // use only pions

  int ngood = 0;
  double average = 0;
  double sumweights = 0;

  // get track info in the set for current combination
  for (int itrk = 0; itrk < ntracks; itrk++) {
    hypo[itrk] = 0;
    const eventTimeTrack& ctrack = tracks[trackInSet[itrk]];
    starttime[itrk] = ctrack.mSignal - ctrack.expTimes[hypo[itrk]];
    weighttime[itrk] = 1. / (ctrack.expSigma[hypo[itrk]] * ctrack.expSigma[hypo[itrk]]);

    average += starttime[itrk] * weighttime[itrk];
    sumweights += weighttime[itrk];
    ngood++;
  }

  average /= sumweights;

  // compute chi2
  chi2 = 0;
  double deltat;
  for (int itrk = 0; itrk < ntracks; itrk++) {
    deltat = starttime[itrk] - average;
    chi2 += deltat * deltat * weighttime[itrk];
  }
  chi2 /= (ngood - 1);

  bestComb = comb;
  chi2best = chi2;
  averageBest = average;

  int worse = -1;
  float errworse = 4;

  // check the best combination
  for (int itrk = 0; itrk < ntracks; itrk++) {
    const eventTimeTrack& ctrack = tracks[trackInSet[itrk]];
    float err = ctrack.mSignal - ctrack.expTimes[hypo[itrk]] - averageBest;
    err /= ctrack.expSigma[hypo[itrk]];
    err = fabs(err);
    if (err > errworse) {
      errworse = err;
      worse = itrk;
    }
  }

  if (worse > -1) {
    const eventTimeTrack& ctrack = tracks[trackInSet[worse]];
    // remove the track and try again
    trackInSet.erase(trackInSet.begin() + worse);
    return 0;
  }

  return 1; // good event time in the set
}

void generateEvTimeTracks(std::vector<eventTimeTrackTest>& tracks, int ntracks, float evTime)
{
  eventTimeTrackTest track;
  constexpr float masses[3] = {0.13957000, 0.49367700, 0.93827200};
  constexpr float kCSPEED = TMath::C() * 1.0e2f * 1.0e-12f; /// Speed of light in TOF units (cm/ps)
  float energy = 0.f;
  float betas[3] = {0.f};

  float pMismatch = ntracks * 0.00005;

  for (int i = 0; i < ntracks; i++) {
    track.mTOFChi2 = 1.f;
    track.mP = gRandom->Exp(1);
    track.mPt = track.mP;
    track.mLength = 400.;
    track.mHypo = gRandom->Exp(1);
    if (track.mHypo > 2) {
      track.mHypo = 2;
    }
    for (int j = 0; j < 3; j++) {
      energy = sqrt(masses[j] * masses[j] + track.mP * track.mP);
      betas[j] = track.mP / energy;
      track.expTimes[j] = track.mLength / (betas[j] * kCSPEED);
      track.expSigma[j] = 100.f;
      if (j == track.mHypo) {
        track.mSignal = track.expTimes[j] + gRandom->Gaus(0.f, track.expSigma[j]);

        if (gRandom->Rndm() < pMismatch) { // assign time from a different particle
          float p = gRandom->Exp(1);
          float l = 400;
          int hypo = gRandom->Exp(1);
          if (hypo > 2) {
            hypo = 2;
          }
          energy = sqrt(masses[hypo] * masses[hypo] + p * p);
          float beta = p / energy;
          track.mSignal = l / (beta * kCSPEED) + gRandom->Gaus(0.f, track.expSigma[j]);
        }
      }
    }
    tracks.push_back(track);
  }
}

} // namespace tof
} // namespace o2
