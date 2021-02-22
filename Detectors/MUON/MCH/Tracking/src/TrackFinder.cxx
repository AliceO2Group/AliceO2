// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackFinder.cxx
/// \brief Implementation of a class to reconstruct tracks
///
/// \author Philippe Pillot, Subatech

#include "MCHTracking/TrackFinder.h"

#include <cassert>
#include <iostream>
#include <stdexcept>

#include <TGeoGlobalMagField.h>
#include <TMatrixD.h>
#include <TMath.h>

#include "Field/MagneticField.h"
#include "MCHTracking/TrackExtrap.h"

namespace o2
{
namespace mch
{

using namespace std;

constexpr double TrackFinder::SDefaultChamberZ[10];
constexpr double TrackFinder::SChamberThicknessInX0[10];
constexpr bool TrackFinder::SRequestStation[5];
constexpr int TrackFinder::SNDE[10];

//_________________________________________________________________________________________________
void TrackFinder::init(float l3Current, float dipoleCurrent)
{
  /// Prepare to run the algorithm

  // create the magnetic field map if not already done
  mTrackFitter.initField(l3Current, dipoleCurrent);

  // enable the track smoother
  mTrackFitter.smoothTracks(true);

  // set the chamber resolution used for fitting the tracks during the tracking
  mTrackFitter.setChamberResolution(SChamberResolutionX, SChamberResolutionY);

  // use the Runge-Kutta extrapolation v2
  TrackExtrap::useExtrapV2();

  // set the maximum MCS angle in chamber from the minimum acceptable momentum
  TrackParam param{};
  double inverseBendingP = (SMinBendingMomentum > 0.) ? 1. / SMinBendingMomentum : 1.;
  param.setInverseBendingMomentum(inverseBendingP);
  for (int iCh = 0; iCh < 10; ++iCh) {
    mMaxMCSAngle2[iCh] = TrackExtrap::getMCSAngle2(param, SChamberThicknessInX0[iCh], 1.);
  }

  // prepare the internal array of list of vector
  // grouping DEs in z-planes (2 for chambers 1-4 and 4 for chambers 5-10)
  for (int iCh = 0; iCh < 4; ++iCh) {
    mClusters[2 * iCh].reserve(2);
    mClusters[2 * iCh].emplace_back(100 * (iCh + 1) + 1, nullptr);
    mClusters[2 * iCh].emplace_back(100 * (iCh + 1) + 3, nullptr);
    mClusters[2 * iCh + 1].reserve(2);
    mClusters[2 * iCh + 1].emplace_back(100 * (iCh + 1), nullptr);
    mClusters[2 * iCh + 1].emplace_back(100 * (iCh + 1) + 2, nullptr);
  }
  for (int iCh = 4; iCh < 6; ++iCh) {
    mClusters[8 + 4 * (iCh - 4)].reserve(5);
    mClusters[8 + 4 * (iCh - 4)].emplace_back(100 * (iCh + 1), nullptr);
    mClusters[8 + 4 * (iCh - 4)].emplace_back(100 * (iCh + 1) + 2, nullptr);
    mClusters[8 + 4 * (iCh - 4)].emplace_back(100 * (iCh + 1) + 4, nullptr);
    mClusters[8 + 4 * (iCh - 4)].emplace_back(100 * (iCh + 1) + 14, nullptr);
    mClusters[8 + 4 * (iCh - 4)].emplace_back(100 * (iCh + 1) + 16, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 1].reserve(4);
    mClusters[8 + 4 * (iCh - 4) + 1].emplace_back(100 * (iCh + 1) + 1, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 1].emplace_back(100 * (iCh + 1) + 3, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 1].emplace_back(100 * (iCh + 1) + 15, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 1].emplace_back(100 * (iCh + 1) + 17, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 2].reserve(4);
    mClusters[8 + 4 * (iCh - 4) + 2].emplace_back(100 * (iCh + 1) + 6, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 2].emplace_back(100 * (iCh + 1) + 8, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 2].emplace_back(100 * (iCh + 1) + 10, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 2].emplace_back(100 * (iCh + 1) + 12, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 3].reserve(5);
    mClusters[8 + 4 * (iCh - 4) + 3].emplace_back(100 * (iCh + 1) + 5, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 3].emplace_back(100 * (iCh + 1) + 7, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 3].emplace_back(100 * (iCh + 1) + 9, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 3].emplace_back(100 * (iCh + 1) + 11, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 3].emplace_back(100 * (iCh + 1) + 13, nullptr);
  }
  for (int iCh = 6; iCh < 10; ++iCh) {
    mClusters[8 + 4 * (iCh - 4)].reserve(7);
    mClusters[8 + 4 * (iCh - 4)].emplace_back(100 * (iCh + 1), nullptr);
    mClusters[8 + 4 * (iCh - 4)].emplace_back(100 * (iCh + 1) + 2, nullptr);
    mClusters[8 + 4 * (iCh - 4)].emplace_back(100 * (iCh + 1) + 4, nullptr);
    mClusters[8 + 4 * (iCh - 4)].emplace_back(100 * (iCh + 1) + 6, nullptr);
    mClusters[8 + 4 * (iCh - 4)].emplace_back(100 * (iCh + 1) + 20, nullptr);
    mClusters[8 + 4 * (iCh - 4)].emplace_back(100 * (iCh + 1) + 22, nullptr);
    mClusters[8 + 4 * (iCh - 4)].emplace_back(100 * (iCh + 1) + 24, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 1].reserve(6);
    mClusters[8 + 4 * (iCh - 4) + 1].emplace_back(100 * (iCh + 1) + 1, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 1].emplace_back(100 * (iCh + 1) + 3, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 1].emplace_back(100 * (iCh + 1) + 5, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 1].emplace_back(100 * (iCh + 1) + 21, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 1].emplace_back(100 * (iCh + 1) + 23, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 1].emplace_back(100 * (iCh + 1) + 25, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 2].reserve(6);
    mClusters[8 + 4 * (iCh - 4) + 2].emplace_back(100 * (iCh + 1) + 8, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 2].emplace_back(100 * (iCh + 1) + 10, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 2].emplace_back(100 * (iCh + 1) + 12, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 2].emplace_back(100 * (iCh + 1) + 14, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 2].emplace_back(100 * (iCh + 1) + 16, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 2].emplace_back(100 * (iCh + 1) + 18, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 3].reserve(7);
    mClusters[8 + 4 * (iCh - 4) + 3].emplace_back(100 * (iCh + 1) + 7, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 3].emplace_back(100 * (iCh + 1) + 9, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 3].emplace_back(100 * (iCh + 1) + 11, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 3].emplace_back(100 * (iCh + 1) + 13, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 3].emplace_back(100 * (iCh + 1) + 15, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 3].emplace_back(100 * (iCh + 1) + 17, nullptr);
    mClusters[8 + 4 * (iCh - 4) + 3].emplace_back(100 * (iCh + 1) + 19, nullptr);
  }
}

//_________________________________________________________________________________________________
const std::list<Track>& TrackFinder::findTracks(const std::unordered_map<int, std::list<Cluster>>& clusters)
{
  /// Run the track finder algorithm

  mTracks.clear();

  // fill the internal array of pointers to the list of clusters per DE
  for (auto& plane : mClusters) {
    for (auto& de : plane) {
      auto itDE = clusters.find(de.first);
      if (itDE == clusters.end()) {
        de.second = nullptr;
      } else {
        de.second = &(itDE->second);
      }
    }
  }

  // use the chamber resolution when fitting the tracks during the tracking
  mTrackFitter.useChamberResolution();

  // find track candidates on stations 4 and 5
  auto tStart = std::chrono::high_resolution_clock::now();
  findTrackCandidates();
  auto tEnd = std::chrono::high_resolution_clock::now();
  mTimeFindCandidates += tEnd - tStart;
  if (mMoreCandidates) {
    tStart = std::chrono::high_resolution_clock::now();
    findMoreTrackCandidates();
    tEnd = std::chrono::high_resolution_clock::now();
    mTimeFindMoreCandidates += tEnd - tStart;
  }
  mNCandidates += mTracks.size();
  print("------ list of track candidates ------");
  printTracks();

  // track each candidate down to chamber 1 and remove it
  tStart = std::chrono::high_resolution_clock::now();
  for (auto itTrack = mTracks.begin(); itTrack != mTracks.end();) {
    std::unordered_map<int, std::unordered_set<uint32_t>> excludedClusters{};
    followTrackInChamber(itTrack, 5, 0, false, excludedClusters);
    print("findTracks: removing candidate at position #", getTrackIndex(itTrack));
    itTrack = mTracks.erase(itTrack);
  }
  tEnd = std::chrono::high_resolution_clock::now();
  mTimeFollowTracks += tEnd - tStart;
  print("------ list of tracks before improvement and cleaning ------");
  printTracks();

  // improve the reconstructed tracks
  tStart = std::chrono::high_resolution_clock::now();
  improveTracks();
  tEnd = std::chrono::high_resolution_clock::now();
  mTimeImproveTracks += tEnd - tStart;

  // remove connected tracks in stations(1..) 3, 4 and 5
  tStart = std::chrono::high_resolution_clock::now();
  removeConnectedTracks(2, 4);
  tEnd = std::chrono::high_resolution_clock::now();
  mTimeCleanTracks += tEnd - tStart;

  // refine the tracks using cluster resolution or just finalize them
  if (mRefineTracks) {
    tStart = std::chrono::high_resolution_clock::now();
    mTrackFitter.useClusterResolution();
    refineTracks();
    tEnd = std::chrono::high_resolution_clock::now();
    mTimeRefineTracks += tEnd - tStart;
  } else {
    finalize();
  }

  return mTracks;
}

//_________________________________________________________________________________________________
void TrackFinder::findTrackCandidates()
{
  /// Find track candidates, made of at least 1 cluster in each chamber of station(1..) 5(4)
  /// and at least one compatible cluster in station(1..) 4(5) if requested
  /// The current parameters of every candidates are set to continue the tracking in the backward direction

  // start by looking for candidates on station 5
  findTrackCandidatesInSt5();

  for (auto itTrack = mTracks.begin(); itTrack != mTracks.end();) {

    // prepare backward tracking if not already done
    if (!itTrack->hasCurrentParam()) {
      prepareBackwardTracking(itTrack, false);
    }

    // look for compatible clusters on station 4
    std::unordered_map<int, std::unordered_set<uint32_t>> excludedClusters{};
    auto itNewTrack = followTrackInChamber(itTrack, 7, 6, false, excludedClusters);

    // keep the current candidate only if no compatible cluster is found and the station is not requested
    if (!SRequestStation[3] && excludedClusters.empty() && itTrack->areCurrentParamValid()) {
      ++itTrack;
    } else {
      print("findTrackCandidates: removing candidate at position #", getTrackIndex(itTrack));
      itTrack = mTracks.erase(itTrack);
      // prepare backward tracking for the new tracks
      for (; itNewTrack != mTracks.end() && itNewTrack != itTrack; ++itNewTrack) {
        prepareBackwardTracking(itNewTrack, false);
      }
    }
  }

  auto itLastCandidateFromSt5 = mTracks.empty() ? mTracks.end() : std::prev(mTracks.end());

  // then look for candidates on station 4
  findTrackCandidatesInSt4();

  auto itFirstCandidateOnSt4 = (itLastCandidateFromSt5 == mTracks.end()) ? mTracks.begin() : std::next(itLastCandidateFromSt5);
  for (auto itTrack = itFirstCandidateOnSt4; itTrack != mTracks.end();) {

    // prepare forward tracking if not already done
    if (!itTrack->hasCurrentParam()) {
      try {
        prepareForwardTracking(itTrack, true);
      } catch (exception const&) {
        print("findTrackCandidates: removing candidate at position #", getTrackIndex(itTrack));
        itTrack = mTracks.erase(itTrack);
        continue;
      }
    }

    // look for compatible clusters on each chamber of station 5 separately,
    // exluding those already attached to an identical candidate on station 4
    // (cases where both chambers of station 5 are fired should have been found in the first step)
    std::unordered_map<int, std::unordered_set<uint32_t>> excludedClusters{};
    if (itLastCandidateFromSt5 != mTracks.end()) {
      excludeClustersFromIdenticalTracks(itTrack, excludedClusters, std::next(itLastCandidateFromSt5));
    }
    auto itFirstNewTrack = followTrackInChamber(itTrack, 8, 8, false, excludedClusters);
    auto itNewTrack = followTrackInChamber(itTrack, 9, 9, false, excludedClusters);
    if (itFirstNewTrack == mTracks.end()) {
      itFirstNewTrack = itNewTrack;
    }

    // keep the current candidate only if no compatible cluster is found and the station is not requested
    if (!SRequestStation[4] && excludedClusters.empty()) {
      itFirstNewTrack = itTrack;
      ++itTrack;
    } else {
      print("findTrackCandidates: removing candidate at position #", getTrackIndex(itTrack));
      itTrack = mTracks.erase(itTrack);
    }

    // refit the track(s) and prepare to continue the tracking in the backward direction
    while (itFirstNewTrack != mTracks.end() && itFirstNewTrack != itTrack) {
      try {
        prepareBackwardTracking(itFirstNewTrack, true);
        ++itFirstNewTrack;
      } catch (exception const&) {
        print("findTrackCandidates: removing candidate at position #", getTrackIndex(itFirstNewTrack));
        itFirstNewTrack = mTracks.erase(itFirstNewTrack);
      }
    }
  }
}

//_________________________________________________________________________________________________
void TrackFinder::findTrackCandidatesInSt5()
{
  /// Find all combinations of clusters between the 2 chambers that could belong to a valid track
  /// New candidates are added in the track list, which must be empty at this stage

  print("--- find candidates in station 5 ---");

  for (int iPlaneCh9 = 27; iPlaneCh9 > 23; --iPlaneCh9) {

    for (int iPlaneCh10 = 28; iPlaneCh10 < 32; ++iPlaneCh10) {

      // skip candidates that have already been found starting from a previous combination
      bool skipUsedPairs = (iPlaneCh9 == 24 || iPlaneCh9 == 26 || iPlaneCh10 == 29 || iPlaneCh10 == 31);

      // find all valid candidates between these 2 planes
      auto itTrack = findTrackCandidates(iPlaneCh9, iPlaneCh10, skipUsedPairs, mTracks.begin());

      // stop here if overlaps have already been checked on both chambers
      if ((iPlaneCh9 == 24 || iPlaneCh9 == 26) && (iPlaneCh10 == 29 || iPlaneCh10 == 31)) {
        continue;
      }

      while (itTrack != mTracks.end()) {

        auto itNextTrack = std::next(itTrack);

        if (iPlaneCh10 == 28 || iPlaneCh10 == 30) {

          // if not already done, look for compatible clusters in the overlapping regions of chamber 10
          try {
            prepareForwardTracking(itTrack, true);
          } catch (exception const&) {
            print("findTrackCandidatesInSt5: removing candidate at position #", getTrackIndex(itTrack));
            itTrack = mTracks.erase(itTrack);
            continue;
          }
          auto itNewTrack = followTrackInOverlapDE(itTrack, itTrack->last().getClusterPtr()->getDEId(), iPlaneCh10 + 1);

          if (itNewTrack != mTracks.end()) {

            // remove the initial candidate if compatible cluster(s) are found
            print("findTrackCandidatesInSt5: removing candidate at position #", getTrackIndex(itTrack));
            itTrack = mTracks.erase(itTrack);

            // refit the track(s) with new attached cluster(s) and prepare to continue the tracking in the backward direction
            bool stop(false);
            while (!stop) {
              itTrack = std::prev(itTrack);
              if (itTrack == itNewTrack) {
                stop = true;
              }
              try {
                prepareBackwardTracking(itTrack, true);
              } catch (exception const&) {
                print("findTrackCandidatesInSt5: removing candidate at position #", getTrackIndex(itTrack));
                itTrack = mTracks.erase(itTrack);
              }
            }
          } else {
            // prepare to continue the tracking in the backward direction with the initial candidate
            prepareBackwardTracking(itTrack, false);
          }
        }

        if (iPlaneCh9 == 25 || iPlaneCh9 == 27) {

          while (itTrack != itNextTrack) {

            // if not already done, look for compatible clusters in the overlapping regions of chamber 9
            if (!itTrack->hasCurrentParam()) {
              prepareBackwardTracking(itTrack, false);
            }
            auto itNewTrack = followTrackInOverlapDE(itTrack, itTrack->first().getClusterPtr()->getDEId(), iPlaneCh9 - 1);

            // keep the initial candidate only if no compatible cluster is found
            if (itNewTrack == mTracks.end()) {
              ++itTrack;
            } else {
              print("findTrackCandidatesInSt5: removing candidate at position #", getTrackIndex(itTrack));
              itTrack = mTracks.erase(itTrack);
            }
          }
        }

        itTrack = itNextTrack;
      }
    }
  }

  // remove tracks out of limits now that overlaps have been checked
  for (auto itTrack = mTracks.begin(); itTrack != mTracks.end();) {
    if (itTrack->isRemovable()) {
      itTrack = mTracks.erase(itTrack);
    } else {
      ++itTrack;
    }
  }
}

//_________________________________________________________________________________________________
void TrackFinder::findTrackCandidatesInSt4()
{
  /// Find all combinations of clusters between the 2 chambers that could belong to a valid track
  /// New candidates are added at the end of the track list, after the candidates from station 5

  print("--- find candidates in station 4 ---");

  auto itLastCandidateFromSt5 = mTracks.empty() ? mTracks.end() : std::prev(mTracks.end());

  for (int iPlaneCh8 = 20; iPlaneCh8 < 24; ++iPlaneCh8) {

    for (int iPlaneCh7 = 19; iPlaneCh7 > 15; --iPlaneCh7) {

      // skip candidates that have already been found starting from a previous combination
      bool skipUsedPairs = (iPlaneCh7 == 18 || iPlaneCh7 == 16 || iPlaneCh8 == 21 || iPlaneCh8 == 23);

      // find all valid candidates between these 2 planes
      auto itFirstCandidateOnSt4 = (itLastCandidateFromSt5 == mTracks.end()) ? mTracks.begin() : std::next(itLastCandidateFromSt5);
      auto itTrack = findTrackCandidates(iPlaneCh7, iPlaneCh8, skipUsedPairs, itFirstCandidateOnSt4);

      // stop here if overlaps have already been checked on both chambers
      if ((iPlaneCh7 == 18 || iPlaneCh7 == 16) && (iPlaneCh8 == 21 || iPlaneCh8 == 23)) {
        continue;
      }

      while (itTrack != mTracks.end()) {

        auto itNextTrack = std::next(itTrack);

        if (iPlaneCh7 == 19 || iPlaneCh7 == 17) {

          // if not already done, look for compatible clusters in the overlapping regions of chamber 7
          prepareBackwardTracking(itTrack, false);
          auto itNewTrack = followTrackInOverlapDE(itTrack, itTrack->first().getClusterPtr()->getDEId(), iPlaneCh7 - 1);

          // keep the initial candidate only if no compatible cluster is found
          if (itNewTrack != mTracks.end()) {
            print("findTrackCandidatesInSt4: removing candidate at position #", getTrackIndex(itTrack));
            mTracks.erase(itTrack);
            itTrack = itNewTrack;
          }
        }

        while (itTrack != itNextTrack) {

          // for every tracks, prepare to continue the tracking in the forward direction
          try {
            prepareForwardTracking(itTrack, true);
          } catch (exception const&) {
            print("findTrackCandidatesInSt4: removing candidate at position #", getTrackIndex(itTrack));
            itTrack = mTracks.erase(itTrack);
            continue;
          }

          if (iPlaneCh8 == 20 || iPlaneCh8 == 22) {

            // if not already done, look for compatible clusters in the overlapping regions of chamber 8
            auto itNewTrack = followTrackInOverlapDE(itTrack, itTrack->last().getClusterPtr()->getDEId(), iPlaneCh8 + 1);

            // keep the initial candidate only if no compatible cluster is found
            if (itNewTrack == mTracks.end()) {
              ++itTrack;
            } else {
              // prepare to continue the tracking of the new tracks from the last attached cluster
              for (; itNewTrack != itTrack; ++itNewTrack) {
                prepareForwardTracking(itNewTrack, false);
              }
              print("findTrackCandidatesInSt4: removing candidate at position #", getTrackIndex(itTrack));
              itTrack = mTracks.erase(itTrack);
            }
          } else {
            ++itTrack;
          }
        }
      }
    }
  }

  // remove tracks out of limits now that overlaps have been checked
  auto itTrack = (itLastCandidateFromSt5 == mTracks.end()) ? mTracks.begin() : ++itLastCandidateFromSt5;
  while (itTrack != mTracks.end()) {
    if (itTrack->isRemovable()) {
      itTrack = mTracks.erase(itTrack);
    } else {
      ++itTrack;
    }
  }
}

//_________________________________________________________________________________________________
void TrackFinder::findMoreTrackCandidates()
{
  /// Find all combinations of clusters between one chamber of station 4 and one chamber of station 5
  /// that could belong to a valid track and that are not already part of track previously found
  /// New track candidates are added at the end of the track list
  /// Their current parameters are set to continue the tracking in the backward direction

  print("--- find more candidates ---");

  auto itLastCandidate = mTracks.empty() ? mTracks.end() : std::prev(mTracks.end());

  for (int iPlaneSt4 = 23; iPlaneSt4 > 15; --iPlaneSt4) {

    for (int iPlaneSt5 = 24; iPlaneSt5 < 32; ++iPlaneSt5) {

      // find all valid candidates between these 2 planes
      auto itTrack = findTrackCandidates(iPlaneSt4, iPlaneSt5, true, mTracks.begin());

      // stop here if overlaps have already been checked on both chambers
      if ((iPlaneSt4 % 2 == 0) && (iPlaneSt5 % 2 == 1)) {
        continue;
      }

      while (itTrack != mTracks.end()) {

        auto itNextTrack = std::next(itTrack);

        if (iPlaneSt5 % 2 == 0) {

          // if not already done, look for compatible clusters in the overlapping regions of that chamber in station 5
          try {
            prepareForwardTracking(itTrack, true);
          } catch (exception const&) {
            print("findMoreTrackCandidates: removing candidate at position #", getTrackIndex(itTrack));
            itTrack = mTracks.erase(itTrack);
            continue;
          }
          auto itNewTrack = followTrackInOverlapDE(itTrack, itTrack->last().getClusterPtr()->getDEId(), iPlaneSt5 + 1);

          if (itNewTrack != mTracks.end()) {

            // remove the initial candidate if compatible cluster(s) are found
            print("findMoreTrackCandidates: removing candidate at position #", getTrackIndex(itTrack));
            itTrack = mTracks.erase(itTrack);

            // refit the track(s) with new cluster(s) and prepare to continue the tracking in the backward direction
            bool stop(false);
            while (!stop) {
              itTrack = std::prev(itTrack);
              if (itTrack == itNewTrack) {
                stop = true;
              }
              try {
                prepareBackwardTracking(itTrack, true);
              } catch (exception const&) {
                print("findMoreTrackCandidates: removing candidate at position #", getTrackIndex(itTrack));
                itTrack = mTracks.erase(itTrack);
              }
            }
          } else {
            // prepare to continue the tracking in the backward direction with the initial candidate
            prepareBackwardTracking(itTrack, false);
          }
        }

        if (iPlaneSt4 % 2 == 1) {

          while (itTrack != itNextTrack) {

            // if not already done, look for compatible clusters in the overlapping regions of that chamber in station 4
            if (!itTrack->hasCurrentParam()) {
              prepareBackwardTracking(itTrack, false);
            }
            auto itNewTrack = followTrackInOverlapDE(itTrack, itTrack->first().getClusterPtr()->getDEId(), iPlaneSt4 - 1);

            // keep the initial candidate only if no compatible cluster is found
            if (itNewTrack == mTracks.end()) {
              ++itTrack;
            } else {
              print("findMoreTrackCandidates: removing candidate at position #", getTrackIndex(itTrack));
              itTrack = mTracks.erase(itTrack);
            }
          }
        }

        itTrack = itNextTrack;
      }
    }
  }

  // remove tracks out of limits now that overlaps have been checked
  // and make sure every new tracks are prepared to continue the tracking in the backward direction
  auto itTrack = (itLastCandidate == mTracks.end()) ? mTracks.begin() : ++itLastCandidate;
  while (itTrack != mTracks.end()) {
    if (itTrack->isRemovable()) {
      itTrack = mTracks.erase(itTrack);
    } else {
      if (!itTrack->hasCurrentParam()) {
        prepareBackwardTracking(itTrack, false);
      }
      ++itTrack;
    }
  }
}

//_________________________________________________________________________________________________
std::list<Track>::iterator TrackFinder::findTrackCandidates(int plane1, int plane2, bool skipUsedPairs, const std::list<Track>::iterator& itFirstTrack)
{
  /// Find all combinations of clusters between the 2 planes that could belong to a valid track
  /// If skipUsedPairs == true: skip combinations of clusters already part of a track starting from itFirstTrack
  /// New candidates are added at the end of the track list
  /// Return an iterator to the first candidate found

  static const double bendingVertexDispersion2 = SBendingVertexDispersion * SBendingVertexDispersion;

  // maximum impact parameter dispersion**2 due to MCS in chambers
  double impactMCS2(0.);
  int chamber1 = getChamberId(plane1);
  for (int iCh = 0; iCh <= chamber1; ++iCh) {
    impactMCS2 += SDefaultChamberZ[iCh] * SDefaultChamberZ[iCh] * mMaxMCSAngle2[iCh];
  }

  // create an iterator to the last track of the list before adding new ones
  auto itTrack = mTracks.empty() ? mTracks.end() : std::prev(mTracks.end());

  for (auto& de1 : mClusters[plane1]) {

    // skip DE without cluster
    if (de1.second == nullptr) {
      continue;
    }

    for (const auto& cluster1 : *de1.second) {

      double z1 = cluster1.getZ();

      for (auto& de2 : mClusters[plane2]) {

        // skip DE without cluster
        if (de2.second == nullptr) {
          continue;
        }

        for (const auto& cluster2 : *de2.second) {

          // skip combinations of clusters already part of a track if requested
          if (skipUsedPairs && itTrack != mTracks.end() && areUsed(cluster1, cluster2, itFirstTrack, std::next(itTrack))) {
            continue;
          }

          double z2 = cluster2.getZ();
          double dZ = z1 - z2;

          // check if non bending impact parameter is within tolerances
          double nonBendingSlope = (cluster1.getX() - cluster2.getX()) / dZ;
          double nonBendingImpactParam = TMath::Abs(cluster1.getX() - cluster1.getZ() * nonBendingSlope);
          double nonBendingImpactParamErr = TMath::Sqrt((z1 * z1 * chamberResolutionX2() + z2 * z2 * chamberResolutionX2()) / dZ / dZ + impactMCS2);
          if ((nonBendingImpactParam - SSigmaCutForTracking * nonBendingImpactParamErr) > (3. * SNonBendingVertexDispersion)) {
            continue;
          }

          double bendingSlope = (cluster1.getY() - cluster2.getY()) / dZ;
          if (TrackExtrap::isFieldON()) { // depending whether the field is ON or OFF
            // check if bending momentum is within tolerances
            double bendingImpactParam = cluster1.getY() - cluster1.getZ() * bendingSlope;
            double bendingImpactParamErr2 = (z1 * z1 * chamberResolutionY2() + z2 * z2 * chamberResolutionY2()) / dZ / dZ + impactMCS2;
            double bendingMomentum = TMath::Abs(TrackExtrap::getBendingMomentumFromImpactParam(bendingImpactParam));
            double bendingMomentumErr = TMath::Sqrt((bendingVertexDispersion2 + bendingImpactParamErr2) / bendingImpactParam / bendingImpactParam + 0.01) * bendingMomentum;
            if ((bendingMomentum + 3. * bendingMomentumErr) < SMinBendingMomentum) {
              continue;
            }
          } else {
            // or check if bending impact parameter is within tolerances
            double bendingImpactParam = TMath::Abs(cluster1.getY() - cluster1.getZ() * bendingSlope);
            double bendingImpactParamErr = TMath::Sqrt((z1 * z1 * chamberResolutionY2() + z2 * z2 * chamberResolutionY2()) / dZ / dZ + impactMCS2);
            if ((bendingImpactParam - SSigmaCutForTracking * bendingImpactParamErr) > (3. * SBendingVertexDispersion)) {
              continue;
            }
          }

          // create a new track candidate
          createTrack(cluster1, cluster2);
        }
      }
    }
  }

  return (itTrack == mTracks.end()) ? mTracks.begin() : ++itTrack;
}

//_________________________________________________________________________________________________
std::list<Track>::iterator TrackFinder::followTrackInOverlapDE(const std::list<Track>::iterator& itTrack, int currentDE, int plane)
{
  /// Follow the track candidate "itTrack" in the DE of the "plane" overlapping "currentDE" and look for compatible clusters
  /// The tracking starts from the current parameters, which are supposed to be at a cluster on the same chamber
  /// The track is duplicated to consider all possibilities and new candidates are added before "itTrack"
  /// Tracks going out of limits with the new cluster are added anyway and tagged as removable
  /// The method returns an iterator to the first new candidate, or mTracks.end() if none is found
  /// The initial candidate "itTrack" is not modified and the new tracks don't have their current parameters set

  print("followTrackInOverlapDE: follow track #", getTrackIndex(itTrack), " currently at DE ", currentDE, " to plane ", plane);
  printTrack(*itTrack);

  // the current track parameters must be set
  assert(itTrack->hasCurrentParam());

  auto itNewTrack(itTrack);

  const TrackParam& currentParam = itTrack->getCurrentParam();
  int currentChamber = itTrack->getCurrentChamber();

  // loop over all DEs of plane
  TrackParam paramAtCluster{};
  for (auto& de : mClusters[plane]) {

    // skip DE without cluster
    if (de.second == nullptr) {
      continue;
    }

    // skip DE that do not overlap with the current DE
    if (de.first % 100 != (currentDE % 100 + 1) % SNDE[currentChamber] && de.first % 100 != (currentDE % 100 - 1 + SNDE[currentChamber]) % SNDE[currentChamber]) {
      continue;
    }

    // look for cluster candidate in this DE
    for (const auto& cluster : *de.second) {

      // try to add the current cluster
      if (!isCompatible(currentParam, cluster, paramAtCluster)) {
        continue;
      }

      // duplicate the track and add the new cluster
      itNewTrack = mTracks.emplace(itNewTrack, *itTrack);
      print("followTrackInOverlapDE: duplicating candidate at position #", getTrackIndex(itNewTrack), " to add cluster ", cluster.getIdAsString());
      itNewTrack->addParamAtCluster(paramAtCluster);

      // tag the track as removable (if it is not already the case) if it is out of limits
      if (!itNewTrack->isRemovable() && !isAcceptable(paramAtCluster)) {
        itNewTrack->removable();
      }
    }
  }

  return (itNewTrack == itTrack) ? mTracks.end() : itNewTrack;
}

//_________________________________________________________________________________________________
std::list<Track>::iterator TrackFinder::followTrackInChamber(std::list<Track>::iterator& itTrack,
                                                             int chamber, int lastChamber, bool canSkip,
                                                             std::unordered_map<int, std::unordered_set<uint32_t>>& excludedClusters)
{
  /// Follow the track candidate pointed to by "itTrack" to the given "chamber"
  /// The tracking starts from the current parameters, which must have already been set
  /// The direction of propagation is supposed to be forward if "chamber" is on station 5 and backward otherwise
  /// Look for compatible cluster(s), excluding those in the "excludedClusters" list, which
  /// correspond to compatible clusters already associated to this candidate in a previous step
  /// For each (pair of) cluster(s) found, continue the tracking to the next chamber, up to "lastChamber"
  /// This is a recursive procedure. Once reaching the last requested chamber, every valid tracks found
  /// are added before "itTrack" and the associated clusters from this chamber onward are attached to them
  /// The method returns an iterator to the first new candidate, or mTracks.end() if none is found
  /// Every compatible clusters found in the process are added to the "excludedClusters" list
  /// The initial candidate "itTrack" is not modified, with the exception of its current parameters,
  /// which are set to the parameters at "chamber" or invalidated in case of propagation issue

  // list of (half-)planes, 2 or 4 per chamber, ordered according to the direction of propagation,
  // which is forward when going to station 5 and backward otherwise with the present algorithm
  static constexpr int plane[10][4] = {{1, 0, -1, -1}, {3, 2, -1, -1}, {5, 4, -1, -1}, {7, 6, -1, -1}, {11, 10, 9, 8}, {15, 14, 13, 12}, {19, 18, 17, 16}, {23, 22, 21, 20}, {24, 25, 26, 27}, {28, 29, 30, 31}};

  print("followTrackInChamber: follow track #", getTrackIndex(itTrack), " to chamber ", chamber + 1, " up to chamber ", lastChamber + 1);

  // the current track parameters must be set at a different chamber and valid
  if (!itTrack->areCurrentParamValid() || chamber == itTrack->getCurrentChamber()) {
    return mTracks.end();
  }

  // determine whether the chamber is the first one reached on the station
  int currentChamber = itTrack->getCurrentChamber();
  bool isFirstOnStation = ((chamber < currentChamber && chamber % 2 == 1) || (chamber > currentChamber && chamber % 2 == 0));

  // follow the track in the 2 planes or 4 half-planes of the chamber
  auto itFirstNewTrack = followTrackInChamber(itTrack, plane[chamber][0], plane[chamber][1], lastChamber, excludedClusters);
  if (chamber > 3) {
    auto itNewTrack = followTrackInChamber(itTrack, plane[chamber][2], plane[chamber][3], lastChamber, excludedClusters);
    if (itFirstNewTrack == mTracks.end()) {
      itFirstNewTrack = itNewTrack;
    }
  }

  // add MCS effects in that chamber before going further with this track or stop here if the track could not reach that chamber
  if (itTrack->areCurrentParamValid()) {
    TrackExtrap::addMCSEffect(&(itTrack->getCurrentParam()), SChamberThicknessInX0[chamber], -1.);
  } else {
    return itFirstNewTrack;
  }

  if (chamber != lastChamber) {

    // save the current track parameters before going to the next chamber
    TrackParam currentParam = itTrack->getCurrentParam();

    // consider the possibility to skip the chamber if it is the first one of the station or if we know we can skip it,
    // i.e. if a compatible cluster has been found on the first chamber and none has been found on the second
    if (isFirstOnStation || (canSkip && excludedClusters.empty())) {
      int nextChamber = (chamber > lastChamber) ? chamber - 1 : chamber + 1;
      auto itNewTrack = followTrackInChamber(itTrack, nextChamber, lastChamber, false, excludedClusters);
      if (itFirstNewTrack == mTracks.end()) {
        itFirstNewTrack = itNewTrack;
      }
    }

    // consider the possibility to skip the entire station if not requested and not the last one
    if (isFirstOnStation && !SRequestStation[chamber / 2] && chamber / 2 != lastChamber / 2) {
      int nextChamber = (chamber > lastChamber) ? chamber - 2 : chamber + 2;
      auto itNewTrack = followTrackInChamber(itTrack, nextChamber, lastChamber, false, excludedClusters);
      if (itFirstNewTrack == mTracks.end()) {
        itFirstNewTrack = itNewTrack;
      }
    }

    // reset the current track parameters to the ones at that chamber if needed
    // (not sure it is needed at all but that way it is clear what the current track parameters are at the end of this function)
    if (itTrack->getCurrentChamber() != chamber) {
      setCurrentParam(*itTrack, currentParam, chamber);
    }
  } else {

    // add a new track if a cluster has been found on the first chamber of the station but not on the second and last one
    // or if one reaches station 1 and it is not requested, whether a cluster has been found on it or not
    if ((!isFirstOnStation && canSkip && excludedClusters.empty()) ||
        (chamber / 2 == 0 && !SRequestStation[0] && (isFirstOnStation || !canSkip))) {
      itFirstNewTrack = mTracks.emplace(itTrack, *itTrack);
      print("followTrackInChamber: duplicating candidate at position #", getTrackIndex(itFirstNewTrack));
    }
  }

  return itFirstNewTrack;
}

//_________________________________________________________________________________________________
std::list<Track>::iterator TrackFinder::followTrackInChamber(std::list<Track>::iterator& itTrack,
                                                             int plane1, int plane2, int lastChamber,
                                                             std::unordered_map<int, std::unordered_set<uint32_t>>& excludedClusters)
{
  /// Follow the track candidate pointed to by "itTrack" to the (half)chamber formed by "plane1" and "plane2"
  /// The tracking starts from the current parameters, which must have already been set
  /// Look for compatible cluster(s), excluding those in the "excludedClusters" list, which
  /// correspond to compatible clusters already associated to this candidate in a previous step
  /// For each (pair of) cluster(s) found, continue the tracking to the next chamber, up to "lastChamber"
  /// This is a recursive procedure. Once reaching the last requested chamber, every valid tracks found
  /// are added before "itTrack" and the associated clusters from this chamber onward are attached to them
  /// Only the tracks with at least one compatible cluster found on plane1 or plane2 are considered
  /// The method returns an iterator to the first new candidate, or mTracks.end() if none is found
  /// Every compatible clusters found in the process are added to the "excludedClusters" list
  /// The initial candidate "itTrack" is not modified, with the exception of its current parameters,
  /// which are set to the parameters at that chamber without adding MCS effects, or invalidated in case of issue

  print("followTrackInChamber: follow track #", getTrackIndex(itTrack), " to planes ", plane1, " and ", plane2, " up to chamber ", lastChamber + 1);
  printTrack(*itTrack);

  // the current track parameters must be set and valid
  if (!itTrack->areCurrentParamValid()) {
    return mTracks.end();
  }

  auto itFirstNewTrack(mTracks.end());

  // add MCS effects in the missing chambers if any. Update the current parameters in the process
  int chamber = getChamberId(plane1);
  if ((chamber < itTrack->getCurrentChamber() - 1 || chamber > itTrack->getCurrentChamber() + 1) &&
      !propagateCurrentParam(*itTrack, (chamber < itTrack->getCurrentChamber()) ? chamber + 1 : chamber - 1)) {
    return mTracks.end();
  }

  // extrapolate the candidate to the chamber if not already there
  TrackParam paramAtChamber = itTrack->getCurrentParam();
  if (itTrack->getCurrentChamber() != chamber && !TrackExtrap::extrapToZCov(&paramAtChamber, SDefaultChamberZ[chamber], true)) {
    itTrack->invalidateCurrentParam();
    return mTracks.end();
  }

  // determine the next chamber to go to, if lastChamber is not yet reached
  int nextChamber(-1);
  if (chamber > lastChamber) {
    nextChamber = chamber - 1;
  } else if (chamber < lastChamber) {
    nextChamber = chamber + 1;
  }

  // loop over all DEs of plane1
  TrackParam paramAtCluster1{};
  TrackParam currentParamAtCluster1{};
  TrackParam paramAtCluster2{};
  std::unordered_map<int, std::unordered_set<uint32_t>> newExcludedClusters{};
  for (auto& de1 : mClusters[plane1]) {

    // skip DE without cluster
    if (de1.second == nullptr) {
      continue;
    }

    // get the list of excluded clusters for the DE
    auto itExcludedClusters = excludedClusters.find(de1.first);
    bool hasExcludedClusters = (itExcludedClusters != excludedClusters.end());

    // look for cluster candidate in this DE
    for (const auto& cluster1 : *de1.second) {

      // skip excluded clusters
      if (hasExcludedClusters && itExcludedClusters->second.count(cluster1.getUniqueId()) > 0) {
        continue;
      }

      // try to add the current cluster
      if (!isCompatible(paramAtChamber, cluster1, paramAtCluster1)) {
        continue;
      }

      // add it to the list of excluded clusters for this candidate
      excludedClusters[de1.first].emplace(cluster1.getUniqueId());

      // skip tracks out of limits, but after checking for overlaps
      bool isAcceptableAtCluster1 = isAcceptable(paramAtCluster1);

      // save the current parameters at cluster1, reset the propagator and add MCS effects before going to plane2
      currentParamAtCluster1 = paramAtCluster1;
      currentParamAtCluster1.resetPropagator();
      TrackExtrap::addMCSEffect(&currentParamAtCluster1, SChamberThicknessInX0[chamber], -1.);

      // loop over all DEs of plane2
      bool cluster2Found(false);
      for (auto& de2 : mClusters[plane2]) {

        // skip DE without cluster
        if (de2.second == nullptr) {
          continue;
        }

        // skip DE that do not overlap with the DE of plane1
        if (de2.first % 100 != (de1.first % 100 + 1) % SNDE[chamber] && de2.first % 100 != (de1.first % 100 - 1 + SNDE[chamber]) % SNDE[chamber]) {
          continue;
        }

        // look for cluster candidate in this DE
        for (const auto& cluster2 : *de2.second) {

          // try to add the current cluster
          if (!isCompatible(currentParamAtCluster1, cluster2, paramAtCluster2)) {
            continue;
          }

          cluster2Found = true;

          // add it to the list of excluded clusters for this candidate
          excludedClusters[de2.first].emplace(cluster2.getUniqueId());

          // skip tracks out of limits
          if (!isAcceptableAtCluster1 || !isAcceptable(paramAtCluster2)) {
            continue;
          }

          // continue the tracking to the next chambers and attach the 2 clusters to the new tracks if any
          auto itNewTrack = addClustersAndFollowTrack(itTrack, paramAtCluster1, &paramAtCluster2, nextChamber, lastChamber, newExcludedClusters);
          if (itFirstNewTrack == mTracks.end()) {
            itFirstNewTrack = itNewTrack;
          }

          // transfert the list of new excluded clusters to the full list for the initial candidate
          moveClusters(newExcludedClusters, excludedClusters);
        }
      }

      if (!cluster2Found && isAcceptableAtCluster1) {

        // continue the tracking with only cluster1 if no compatible cluster is found on plane2 and the track stays within limits
        auto itNewTrack = addClustersAndFollowTrack(itTrack, paramAtCluster1, nullptr, nextChamber, lastChamber, newExcludedClusters);
        if (itFirstNewTrack == mTracks.end()) {
          itFirstNewTrack = itNewTrack;
        }

        // transfert the list of new excluded clusters to the full list for the initial candidate
        moveClusters(newExcludedClusters, excludedClusters);
      }
    }
  }

  // loop over all DEs of plane2
  for (auto& de2 : mClusters[plane2]) {

    // skip DE without cluster
    if (de2.second == nullptr) {
      continue;
    }

    // get the list of excluded clusters for the DE
    auto itExcludedClusters = excludedClusters.find(de2.first);
    bool hasExcludedClusters = (itExcludedClusters != excludedClusters.end());

    // look for cluster candidate in this DE
    for (const auto& cluster2 : *de2.second) {

      // skip excluded clusters (in particular the ones already attached together with a cluster on plane1)
      if (hasExcludedClusters && itExcludedClusters->second.count(cluster2.getUniqueId()) > 0) {
        continue;
      }

      // try to add the current cluster
      if (!isCompatible(paramAtChamber, cluster2, paramAtCluster2)) {
        continue;
      }

      // add it to the list of excluded clusters for this candidate
      excludedClusters[de2.first].emplace(cluster2.getUniqueId());

      // skip tracks out of limits
      if (!isAcceptable(paramAtCluster2)) {
        continue;
      }

      // continue the tracking to the next chambers and attach the cluster to the new tracks if any
      auto itNewTrack = addClustersAndFollowTrack(itTrack, paramAtCluster2, nullptr, nextChamber, lastChamber, newExcludedClusters);
      if (itFirstNewTrack == mTracks.end()) {
        itFirstNewTrack = itNewTrack;
      }

      // transfert the list of new excluded clusters to the full list for the initial candidate
      moveClusters(newExcludedClusters, excludedClusters);
    }
  }

  // reset the current parameters to the ones at that chamber if needed, not adding MCS effects yet
  if (itTrack->getCurrentChamber() != chamber) {
    setCurrentParam(*itTrack, paramAtChamber, chamber);
  }

  return itFirstNewTrack;
}

//_________________________________________________________________________________________________
std::list<Track>::iterator TrackFinder::addClustersAndFollowTrack(std::list<Track>::iterator& itTrack, const TrackParam& paramAtCluster1,
                                                                  const TrackParam* paramAtCluster2, int nextChamber, int lastChamber,
                                                                  std::unordered_map<int, std::unordered_set<uint32_t>>& excludedClusters)
{
  /// If "nextChamber" >= 0: continue the tracking of "itTrack" up to "lastChamber", attach the two clusters
  /// to every new tracks found and return an iterator to the first of them (or mTracks.end() if none is found)
  /// Every compatible clusters found in the process is added to the "excludedClusters" list of this candidate
  /// If nextChamber < 0: duplicate itTrack, attach the clusters and return an iterator to the new track
  /// The initial candidate "itTrack" is not modified, with the exception of its current parameters

  // the list of excluded clusters must be empty here as new cluster(s) are being attached to the candidate
  assert(excludedClusters.empty());

  auto itFirstNewTrack(mTracks.end());

  if (nextChamber >= 0) {

    // the tracking continues from paramAtCluster2, if any, or from paramAtCluster1
    if (paramAtCluster2) {
      print("addClustersAndFollowTrack: 2 clusters found (", paramAtCluster1.getClusterPtr()->getIdAsString(), " and ",
            paramAtCluster2->getClusterPtr()->getIdAsString(), "). Continuing the tracking of candidate #", getTrackIndex(itTrack));
      setCurrentParam(*itTrack, *paramAtCluster2, paramAtCluster2->getClusterPtr()->getChamberId());
    } else {
      print("addClustersAndFollowTrack: 1 cluster found (", paramAtCluster1.getClusterPtr()->getIdAsString(),
            "). Continuing the tracking of candidate #", getTrackIndex(itTrack));
      setCurrentParam(*itTrack, paramAtCluster1, paramAtCluster1.getClusterPtr()->getChamberId());
    }

    // follow the track to the next chamber, which can be skipped if it is on the same station
    bool canSkip = (nextChamber / 2 == paramAtCluster1.getClusterPtr()->getChamberId() / 2);
    auto itNewTrack = followTrackInChamber(itTrack, nextChamber, lastChamber, canSkip, excludedClusters);
    itFirstNewTrack = itNewTrack;

    // attach the current cluster(s) to every new tracks found
    if (itNewTrack != mTracks.end()) {
      while (itNewTrack != itTrack) {
        itNewTrack->addParamAtCluster(paramAtCluster1);
        if (paramAtCluster2) {
          itNewTrack->addParamAtCluster(*paramAtCluster2);
          print("addClustersAndFollowTrack: add to the candidate at position #", getTrackIndex(itNewTrack),
                " clusters ", paramAtCluster1.getClusterPtr()->getIdAsString(), " and ", paramAtCluster2->getClusterPtr()->getIdAsString());
        } else {
          print("addClustersAndFollowTrack: add to the candidate at position #", getTrackIndex(itNewTrack),
                " cluster ", paramAtCluster1.getClusterPtr()->getIdAsString());
        }
        ++itNewTrack;
      }
    }

  } else {

    // or duplicate the track and add the new cluster(s)
    itFirstNewTrack = mTracks.emplace(itTrack, *itTrack);
    itFirstNewTrack->addParamAtCluster(paramAtCluster1);
    if (paramAtCluster2) {
      itFirstNewTrack->addParamAtCluster(*paramAtCluster2);
      print("addClustersAndFollowTrack: duplicating candidate at position #", getTrackIndex(itFirstNewTrack), " to add 2 clusters (",
            paramAtCluster1.getClusterPtr()->getIdAsString(), " and ", paramAtCluster2->getClusterPtr()->getIdAsString(), ")");
    } else {
      print("addClustersAndFollowTrack: duplicating candidate at position #", getTrackIndex(itFirstNewTrack),
            " to add 1 cluster (", paramAtCluster1.getClusterPtr()->getIdAsString(), ")");
    }
  }

  return itFirstNewTrack;
}

//_________________________________________________________________________________________________
void TrackFinder::improveTracks()
{
  /// Improve tracks by removing removable clusters with local chi2 higher than the defined cut
  /// Removable clusters are identified by the method Track::tagRemovableClusters()
  /// Recompute track parameters and covariances at the remaining clusters
  /// Remove the track if it cannot be improved or in case of failure

  // Maximum chi2 to keep a cluster (the factor 2 is for the 2 degrees of freedom: x and y)
  static const double maxChi2OfCluster = 2. * SSigmaCutForImprovement * SSigmaCutForImprovement;

  for (auto itTrack = mTracks.begin(); itTrack != mTracks.end();) {

    bool removeTrack(false);

    // At the first step, only run the smoother
    auto itStartingParam = std::prev(itTrack->rend());

    while (true) {

      // Refit the part of the track affected by the cluster removal, run the smoother, but do not finalize
      try {
        mTrackFitter.fit(*itTrack, true, false, (itStartingParam == itTrack->rbegin()) ? nullptr : &itStartingParam);
      } catch (exception const&) {
        removeTrack = true;
        break;
      }

      // Identify removable clusters
      itTrack->tagRemovableClusters(requestedStationMask(), !mMoreCandidates);

      // Look for the cluster with the worst local chi2
      double worstLocalChi2(-1.);
      auto itWorstParam(itTrack->end());
      for (auto itParam = itTrack->begin(); itParam != itTrack->end(); ++itParam) {
        if (itParam->getLocalChi2() > worstLocalChi2) {
          worstLocalChi2 = itParam->getLocalChi2();
          itWorstParam = itParam;
        }
      }

      // If the worst chi2 is under requirement then the track is improved
      if (worstLocalChi2 < maxChi2OfCluster) {
        break;
      }

      // If the worst cluster is not removable then the track cannot be improved
      if (!itWorstParam->isRemovable()) {
        removeTrack = true;
        break;
      }

      // Remove the worst cluster
      auto itNextParam = itTrack->removeParamAtCluster(itWorstParam);

      // Decide from where to refit the track: from the cluster next the one suppressed or
      // from scratch if the removed cluster was used to compute the tracking seed
      itStartingParam = itTrack->rbegin();
      auto itNextToNextParam = (itNextParam == itTrack->end()) ? itNextParam : std::next(itNextParam);
      while (itNextToNextParam != itTrack->end()) {
        if (itNextToNextParam->getClusterPtr()->getChamberId() != itNextParam->getClusterPtr()->getChamberId()) {
          itStartingParam = std::make_reverse_iterator(++itNextParam);
          break;
        }
        ++itNextToNextParam;
      }
    }

    // Remove the track if it couldn't be improved
    if (removeTrack) {
      print("improveTracks: removing candidate at position #", getTrackIndex(itTrack));
      itTrack = mTracks.erase(itTrack);
    } else {
      ++itTrack;
    }
  }
}

//_________________________________________________________________________________________________
void TrackFinder::removeConnectedTracks(int stMin, int stMax)
{
  /// Find and remove tracks sharing 1 cluster or more in station(s) [stMin, stMax]
  /// For each couple of connected tracks, one removes the one with the smallest
  /// number of fired chambers or with the highest chi2/(ndf-1) value in case of equality

  if (mTracks.size() < 2) {
    return;
  }

  int chMin = 2 * stMin;
  int chMax = 2 * stMax + 1;
  int nPlane = 2 * (chMax - chMin + 1);

  // first loop to fill the arrays of cluster Ids and number of fired chambers
  std::vector<uint32_t> ClIds(nPlane * mTracks.size());
  std::vector<uint8_t> nFiredCh(mTracks.size());
  int previousCh(-1);
  int iTrack(0);
  for (auto itTrack = mTracks.begin(); itTrack != mTracks.end(); ++itTrack, ++iTrack) {
    for (auto itParam = itTrack->rbegin(); itParam != itTrack->rend(); ++itParam) {
      int ch = itParam->getClusterPtr()->getChamberId();
      if (ch != previousCh) {
        ++nFiredCh[iTrack];
        previousCh = ch;
      }
      if (ch >= chMin && ch <= chMax) {
        ClIds[nPlane * iTrack + 2 * (ch - chMin) + itParam->getClusterPtr()->getDEId() % 2] = itParam->getClusterPtr()->getUniqueId();
      }
    }
  }

  // second loop to tag the tracks to remove
  int iTrack1 = mTracks.size() - 1;
  int iindex = ClIds.size() - 1;
  for (auto itTrack1 = mTracks.rbegin(); itTrack1 != mTracks.rend(); ++itTrack1, iindex -= nPlane, --iTrack1) {
    int iTrack2 = iTrack1 - 1;
    int jindex = iindex - nPlane;
    for (auto itTrack2 = std::next(itTrack1); itTrack2 != mTracks.rend(); ++itTrack2, --iTrack2) {
      for (int iPlane = nPlane; iPlane > 0; --iPlane) {
        if (ClIds[iindex] > 0 && ClIds[iindex] == ClIds[jindex]) {
          if ((nFiredCh[iTrack2] > nFiredCh[iTrack1]) ||
              ((nFiredCh[iTrack2] == nFiredCh[iTrack1]) &&
               (itTrack2->first().getTrackChi2() / (itTrack2->getNDF() - 1) < itTrack1->first().getTrackChi2() / (itTrack1->getNDF() - 1)))) {
            itTrack1->connected();
          } else {
            itTrack2->connected();
          }
          iindex -= iPlane;
          jindex -= iPlane;
          break;
        }
        --iindex;
        --jindex;
      }
      iindex += nPlane;
    }
  }

  // third loop to remove them. That way all combinations are tested.
  for (auto itTrack = mTracks.begin(); itTrack != mTracks.end();) {
    if (itTrack->isConnected()) {
      print("removeConnectedTracks: removing candidate at position #", getTrackIndex(itTrack));
      itTrack = mTracks.erase(itTrack);
    } else {
      ++itTrack;
    }
  }
}

//_________________________________________________________________________________________________
void TrackFinder::refineTracks()
{
  /// Refit, smooth and finalize the reconstructed tracks

  for (auto itTrack = mTracks.begin(); itTrack != mTracks.end();) {
    try {
      mTrackFitter.fit(*itTrack);
      ++itTrack;
    } catch (exception const&) {
      print("refineTracks: removing candidate at position #", getTrackIndex(itTrack));
      itTrack = mTracks.erase(itTrack);
    }
  }
}

//_________________________________________________________________________________________________
void TrackFinder::finalize()
{
  /// Copy the smoothed parameters and covariances into the regular ones
  for (auto& track : mTracks) {
    for (auto& param : track) {
      param.setParameters(param.getSmoothParameters());
      param.setCovariances(param.getSmoothCovariances());
    }
  }
}

//_________________________________________________________________________________________________
void TrackFinder::createTrack(const Cluster& cl1, const Cluster& cl2)
{
  /// Create a new track with these 2 clusters and store it at the end of the list of tracks
  /// Compute the track parameters and covariance matrices at the 2 clusters

  // create the track and the trackParam at each cluster
  Track& track = mTracks.emplace_back();
  track.createParamAtCluster(cl2);
  track.createParamAtCluster(cl1);
  print("createTrack: creating candidate at position #", getTrackIndex(std::prev(mTracks.end())),
        " with clusters ", cl1.getIdAsString(), " and ", cl2.getIdAsString());

  // fit the track using the Kalman filter
  try {
    mTrackFitter.fit(track, false);
  } catch (exception const&) {
    print("... fit failed --> removing it");
    mTracks.erase(std::prev(mTracks.end()));
  }
}

//_________________________________________________________________________________________________
bool TrackFinder::isAcceptable(const TrackParam& param) const
{
  /// Return true if the track is within given limits on momentum/angle/origin

  // impact parameter dispersion**2 due to MCS in chambers
  int chamber = param.getClusterPtr()->getChamberId();
  double impactMCS2(0.);
  if (TrackExtrap::isFieldON() && chamber < 6) {
    // track momentum is known
    for (int iCh = 0; iCh <= chamber; ++iCh) {
      impactMCS2 += SDefaultChamberZ[iCh] * SDefaultChamberZ[iCh] * TrackExtrap::getMCSAngle2(param, SChamberThicknessInX0[iCh], 1.);
    }
  } else {
    // track momentum is unknown
    for (Int_t iCh = 0; iCh <= chamber; ++iCh) {
      impactMCS2 += SDefaultChamberZ[iCh] * SDefaultChamberZ[iCh] * mMaxMCSAngle2[iCh];
    }
  }

  const TMatrixD& paramCov = param.getCovariances();
  double z = param.getZ();

  // check if non bending impact parameter is within tolerances
  double nonBendingImpactParam = TMath::Abs(param.getNonBendingCoor() - z * param.getNonBendingSlope());
  double nonBendingImpactParamErr = TMath::Sqrt(paramCov(0, 0) + z * z * paramCov(1, 1) - 2. * z * paramCov(0, 1) + impactMCS2);
  if ((nonBendingImpactParam - SSigmaCutForTracking * nonBendingImpactParamErr) > (3. * SNonBendingVertexDispersion)) {
    return false;
  }

  if (TrackExtrap::isFieldON()) { // depending whether the field is ON or OFF
    // check if bending momentum is within tolerances
    double bendingMomentum = TMath::Abs(1. / param.getInverseBendingMomentum());
    double bendingMomentumErr = TMath::Sqrt(paramCov(4, 4)) * bendingMomentum * bendingMomentum;
    if (chamber < 6 && (bendingMomentum + SSigmaCutForTracking * bendingMomentumErr) < SMinBendingMomentum) {
      return false;
    } else if ((bendingMomentum + 3. * bendingMomentumErr) < SMinBendingMomentum) {
      return false;
    }
  } else {
    // or check if bending impact parameter is within tolerances
    double bendingImpactParam = TMath::Abs(param.getBendingCoor() - z * param.getBendingSlope());
    double bendingImpactParamErr = TMath::Sqrt(paramCov(2, 2) + z * z * paramCov(3, 3) - 2. * z * paramCov(2, 3) + impactMCS2);
    if ((bendingImpactParam - SSigmaCutForTracking * bendingImpactParamErr) > (3. * SBendingVertexDispersion)) {
      return false;
    }
  }

  return true;
}

//_________________________________________________________________________________________________
void TrackFinder::prepareForwardTracking(std::list<Track>::iterator& itTrack, bool runSmoother)
{
  /// Prepare the current track parameters in view of continuing the tracking in the forward chambers
  /// Run the smoother to recompute the parameters at last cluster if requested
  /// Throw an exception in case of failure while running the smoother

  if (runSmoother) {
    auto itStartingParam = std::prev(itTrack->rend());
    mTrackFitter.fit(*itTrack, true, false, &itStartingParam);
  }

  setCurrentParam(*itTrack, itTrack->last(), itTrack->last().getClusterPtr()->getChamberId(), runSmoother);
}

//_________________________________________________________________________________________________
void TrackFinder::prepareBackwardTracking(std::list<Track>::iterator& itTrack, bool refit)
{
  /// Prepare the current track parameters in view of continuing the tracking in the backward chambers
  /// Refit the track to recompute the parameters at first cluster if requested
  /// Throw an exception in case of failure during the refit

  if (refit) {
    mTrackFitter.fit(*itTrack, false);
  }

  setCurrentParam(*itTrack, itTrack->first(), itTrack->first().getClusterPtr()->getChamberId());
}

//_________________________________________________________________________________________________
void TrackFinder::setCurrentParam(Track& track, const TrackParam& param, int chamber, bool smoothed)
{
  /// Set the current track parameters and the associated chamber, using smoothed ones if requested
  /// Add MCS effects and reset the propagator if they are associated with a cluster

  track.setCurrentParam(param, chamber);

  if (smoothed) {
    TrackParam& currentParam = track.getCurrentParam();
    currentParam.setParameters(param.getSmoothParameters());
    currentParam.setCovariances(param.getSmoothCovariances());
  }

  if (param.getClusterPtr()) {
    TrackParam& currentParam = track.getCurrentParam();
    currentParam.resetPropagator();
    TrackExtrap::addMCSEffect(&currentParam, SChamberThicknessInX0[chamber], -1.);
  }
}

//_________________________________________________________________________________________________
bool TrackFinder::propagateCurrentParam(Track& track, int chamber)
{
  /// Propagate the current track parameters to the chamber
  /// Adding MCS effects in every chambers crossed, including this one
  /// Return false and invalidate the current parameters in case of failure during extrapolation

  TrackParam& currentParam = track.getCurrentParam();
  int& currentChamber = track.getCurrentChamber();
  while (currentChamber != chamber) {

    currentChamber += (chamber < currentChamber) ? -1 : 1;

    if (!TrackExtrap::extrapToZCov(&currentParam, SDefaultChamberZ[currentChamber], true)) {
      track.invalidateCurrentParam();
      return false;
    }

    TrackExtrap::addMCSEffect(&currentParam, SChamberThicknessInX0[currentChamber], -1.);
  }

  return true;
}

//_________________________________________________________________________________________________
bool TrackFinder::areUsed(const Cluster& cl1, const Cluster& cl2, const std::list<Track>::iterator& itFirstTrack, const std::list<Track>::iterator& itLastTrack)
{
  /// Return true if the 2 clusters are already part of a track between itFirstTrack and mTracks.end()

  if (itFirstTrack == mTracks.end()) {
    return false;
  }

  for (auto itTrack = itFirstTrack; itTrack != itLastTrack; ++itTrack) {

    bool cl1Used(false), cl2Used(false);

    for (auto itParam = itTrack->rbegin(); itParam != itTrack->rend(); ++itParam) {

      if (itParam->getClusterPtr() == &cl1) {
        cl1Used = true;
      } else if (itParam->getClusterPtr() == &cl2) {
        cl2Used = true;
      }

      if (cl1Used && cl2Used) {
        return true;
      }
    }
  }

  return false;
}

//_________________________________________________________________________________________________
void TrackFinder::excludeClustersFromIdenticalTracks(const std::list<Track>::iterator& itTrack,
                                                     std::unordered_map<int, std::unordered_set<uint32_t>>& excludedClusters,
                                                     const std::list<Track>::iterator& itEndTrack)
{
  /// Find tracks in the range [mTracks.begin(), itEndTrack[ that contain all the clusters of itTrack
  /// and add the clusters that these tracks have on station 5 in the excludedClusters list
  for (auto itTrack2 = mTracks.begin(); itTrack2 != itEndTrack; ++itTrack2) {
    if (itTrack->getNClustersInCommon(*itTrack2) == itTrack->getNClusters()) {
      for (auto itParam = itTrack2->rbegin(); itParam != itTrack2->rend(); ++itParam) {
        const Cluster* cluster = itParam->getClusterPtr();
        if (cluster->getChamberId() > 7) {
          excludedClusters[cluster->getDEId()].emplace(cluster->getUniqueId());
        } else {
          break;
        }
      }
    }
  }
}

//_________________________________________________________________________________________________
void TrackFinder::moveClusters(std::unordered_map<int, std::unordered_set<uint32_t>>& source, std::unordered_map<int, std::unordered_set<uint32_t>>& destination)
{
  /// Move cluster Ids listed in source into destination then clear source
  for (auto& sourceDE : source) {
    destination[sourceDE.first].insert(sourceDE.second.begin(), sourceDE.second.end());
  }
  source.clear();
}

//_________________________________________________________________________________________________
bool TrackFinder::isCompatible(const TrackParam& param, const Cluster& cluster, TrackParam& paramAtCluster)
{
  /// Test the compatibility between the track and the cluster
  /// If compatible, paramAtCluster contains the new track parameters at this cluster

  // maximum chi2 to accept a cluster candidate (the factor 2 is for the 2 degrees of freedom: x and y)
  static const double maxChi2OfCluster = 2. * SSigmaCutForTracking * SSigmaCutForTracking;

  // fast try to add the current cluster
  if (!tryOneClusterFast(param, cluster)) {
    return false;
  }

  // try to add the current cluster accurately
  if (tryOneCluster(param, cluster, paramAtCluster) >= maxChi2OfCluster) {
    return false;
  }

  // save the extrapolated parameters and covariances for the smoother
  paramAtCluster.setExtrapParameters(paramAtCluster.getParameters());
  paramAtCluster.setExtrapCovariances(paramAtCluster.getCovariances());

  // compute the new track parameters including the cluster using the Kalman filter
  try {
    mTrackFitter.runKalmanFilter(paramAtCluster);
  } catch (exception const&) {
    return false;
  }

  return true;
}

//_________________________________________________________________________________________________
bool TrackFinder::tryOneClusterFast(const TrackParam& param, const Cluster& cluster)
{
  /// Quickly test the compatibility between the track and the cluster
  /// given the track and cluster resolutions + the maximum-distance-to-track value
  /// and assuming linear propagation of the track to the z position of the cluster
  /// Return true if they are compatibles

  ++mNCallTryOneClusterFast;

  double dZ = cluster.getZ() - param.getZ();
  double dX = cluster.getX() - (param.getNonBendingCoor() + param.getNonBendingSlope() * dZ);
  double dY = cluster.getY() - (param.getBendingCoor() + param.getBendingSlope() * dZ);
  const TMatrixD& paramCov = param.getCovariances();
  double errX2 = paramCov(0, 0) + dZ * dZ * paramCov(1, 1) + 2. * dZ * paramCov(0, 1) + chamberResolutionX2();
  double errY2 = paramCov(2, 2) + dZ * dZ * paramCov(3, 3) + 2. * dZ * paramCov(2, 3) + chamberResolutionY2();

  double dXmax = SSigmaCutForTracking * TMath::Sqrt(2. * errX2) + SMaxNonBendingDistanceToTrack;
  double dYmax = SSigmaCutForTracking * TMath::Sqrt(2. * errY2) + SMaxBendingDistanceToTrack;

  if (TMath::Abs(dX) > dXmax || TMath::Abs(dY) > dYmax) {
    return false;
  }
  return true;
}

//_________________________________________________________________________________________________
double TrackFinder::tryOneCluster(const TrackParam& param, const Cluster& cluster, TrackParam& paramAtCluster)
{
  /// Test the compatibility between the track and the cluster
  /// given the track covariance matrix and the cluster resolution
  /// and propagating properly the track to the z position of the cluster
  /// Return the matching chi2 and the track parameters at the cluster

  ++mNCallTryOneCluster;

  // Extrapolate the track parameters and covariances at the z position of the cluster
  paramAtCluster = param;
  paramAtCluster.setClusterPtr(&cluster);
  if (!TrackExtrap::extrapToZCov(&paramAtCluster, cluster.getZ(), true)) {
    return mTrackFitter.getMaxChi2();
  }

  // Compute the cluster-track residuals in bending and non bending directions
  double dX = cluster.getX() - paramAtCluster.getNonBendingCoor();
  double dY = cluster.getY() - paramAtCluster.getBendingCoor();

  // Combine the cluster and track resolutions and covariances
  const TMatrixD& paramCov = paramAtCluster.getCovariances();
  double sigmaX2 = paramCov(0, 0) + chamberResolutionX2();
  double sigmaY2 = paramCov(2, 2) + chamberResolutionY2();
  double covXY = paramCov(0, 2);
  double det = sigmaX2 * sigmaY2 - covXY * covXY;

  // Compute and return the matching chi2
  if (det == 0.) {
    return mTrackFitter.getMaxChi2();
  }
  return (dX * dX * sigmaY2 + dY * dY * sigmaX2 - 2. * dX * dY * covXY) / det;
}

//_________________________________________________________________________________________________
uint8_t TrackFinder::requestedStationMask() const
{
  /// Get the mask of the requested station, i.e. an integer where
  /// bit n is set to 1 if the station n was requested
  uint8_t mask(0);
  for (int i = 0; i < 5; ++i) {
    if (SRequestStation[i]) {
      mask |= (1 << i);
    }
  }
  return mask;
}

//_________________________________________________________________________________________________
int TrackFinder::getTrackIndex(const std::list<Track>::iterator& itCurrentTrack) const
{
  /// return the index of the track pointed to by the given iterator in the list of tracks
  /// return -1 if it points to mTracks.end()
  /// return -2 if it points to nothing in the list
  /// return -3 if the debug level is < 1 as this function is supposed to be used for debug only

  if (mDebugLevel < 1) {
    return -3;
  }

  if (itCurrentTrack == mTracks.end()) {
    return -1;
  }

  int index(0);
  for (auto itTrack = mTracks.begin(); itTrack != itCurrentTrack; ++itTrack) {
    if (itTrack == mTracks.end()) {
      return -2;
    }
    ++index;
  }

  return index;
}

//_________________________________________________________________________________________________
void TrackFinder::printTracks() const
{
  /// print all the tracks currently in the list if the debug level is > 1
  if (mDebugLevel > 1) {
    for (const auto& track : mTracks) {
      track.print();
    }
  }
}

//_________________________________________________________________________________________________
void TrackFinder::printTrack(const Track& track) const
{
  /// print the track if the debug level is > 1
  if (mDebugLevel > 1) {
    track.print();
  }
}

//_________________________________________________________________________________________________
void TrackFinder::printTrackParam(const TrackParam& trackParam) const
{
  /// print the track parameters if the debug level is > 1
  if (mDebugLevel > 1) {
    trackParam.print();
  }
}

//_________________________________________________________________________________________________
template <class... Args>
void TrackFinder::print(Args... args) const
{
  /// print a debug message if the debug level is > 0
  if (mDebugLevel > 0) {
    (cout << ... << args) << "\n";
  }
}

//_________________________________________________________________________________________________
void TrackFinder::printStats() const
{
  /// print the timers
  LOG(INFO) << "number of candidates tracked = " << mNCandidates;
  TrackExtrap::printNCalls();
  LOG(INFO) << "number of times tryOneClusterFast() is called = " << mNCallTryOneClusterFast;
  LOG(INFO) << "number of times tryOneCluster() is called = " << mNCallTryOneCluster;
}

//_________________________________________________________________________________________________
void TrackFinder::printTimers() const
{
  /// print the timers
  LOG(INFO) << "findTrackCandidates duration = " << mTimeFindCandidates.count() << " s";
  LOG(INFO) << "findMoreTrackCandidates duration = " << mTimeFindMoreCandidates.count() << " s";
  LOG(INFO) << "followTracks duration = " << mTimeFollowTracks.count() << " s";
  LOG(INFO) << "improveTracks duration = " << mTimeImproveTracks.count() << " s";
  LOG(INFO) << "removeConnectedTracks duration = " << mTimeCleanTracks.count() << " s";
  LOG(INFO) << "refineTracks duration = " << mTimeRefineTracks.count() << " s";
}

} // namespace mch
} // namespace o2
