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
#include "TOFReconstruction/EventTimeMaker.h"

namespace o2
{

namespace tof
{

constexpr int MAXNTRACKINSET = 10;
// usefull constants
constexpr unsigned long combinatorial[MAXNTRACKINSET + 1] = {1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049};
//---------------

void computeEvTime(const std::vector<eventTimeTrack>& tracks, const std::vector<int>& trkIndex, eventTimeContainer& evtime)
{
  static constexpr int maxNumberOfSets = 200;
  static constexpr float weightLimit = 1E-6; // Limit in the weights

  const int ntracks = tracks.size();
  LOG(debug) << "For the collision time using " << ntracks;

  if (ntracks < 2) { // at least 2 tracks required
    LOG(debug) << "Skipping event because at least 2 tracks are required";
    return;
  }

  int hypo[MAXNTRACKINSET];

  int nmaxtracksinset = ntracks > 22 ? 6 : MAXNTRACKINSET; // max number of tracks in a set for event time computation
  int ntracksinset = std::min(ntracks, nmaxtracksinset);

  int nset = ((ntracks - 1) / ntracksinset) + 1;
  int ntrackUsed = ntracks;

  if (nset > maxNumberOfSets) {
    nset = maxNumberOfSets;
    ntrackUsed = nmaxtracksinset * nset;
  }

  // list of tracks in set
  std::vector<int> trackInSet[maxNumberOfSets];

  for (int i = 0; i < ntrackUsed; i++) {
    int iset = i % nset;

    trackInSet[iset].push_back(i);
  }

  int status;
  // compute event time for each set
  for (int iset = 0; iset < nset; iset++) {
    unsigned long bestComb = 0;
    while (!(status = getStartTimeInSet(tracks, trackInSet[iset], bestComb))) {
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

        evtime.weights[index] = 1. / (ctrack.expSigma[hypo[itrk]] * ctrack.expSigma[hypo[itrk]]);
        evtime.tracktime[index] = ctrack.mSignal - ctrack.expTimes[hypo[itrk]];
      }
    }
  } // end loop in set

  // do average among all tracks
  float finalTime = 0, allweights = 0;
  for (int i = 0; i < evtime.weights.size(); i++) {
    if (evtime.weights[i] < weightLimit) {
      continue;
    }
    allweights += evtime.weights[i];
    finalTime += evtime.tracktime[i] * evtime.weights[i];
  }

  if (allweights < weightLimit) {
    LOG(debug) << "Skipping because allweights " << allweights << " are lower than " << weightLimit;
    return;
  }

  evtime.eventTime = finalTime / allweights;
  evtime.eventTimeError = sqrt(1. / allweights);
  evtime.eventTimeMultiplicity = ntracks;
}

int getStartTimeInSet(const std::vector<eventTimeTrack>& tracks, std::vector<int>& trackInSet, unsigned long& bestComb)
{
  float chi2, chi2best, averageBest = 0;
  int hypo[MAXNTRACKINSET];
  float starttime[MAXNTRACKINSET], weighttime[MAXNTRACKINSET];

  chi2best = 10000;
  int ntracks = trackInSet.size();

  if (ntracks < 3) {
    return 2; // no event time in the set
  }

  unsigned long ncomb = combinatorial[ntracks];
  for (unsigned long comb = 0; comb < ncomb; comb++) {
    unsigned long curr = comb;

    int ngood = 0;
    float average = 0;
    float sumweights = 0;
    // get track info in the set for current combination
    for (int itrk = 0; itrk < ntracks; itrk++) {
      hypo[itrk] = curr % 3;
      curr /= 3;
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
    float deltat;
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
  float errworse = 4;
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
