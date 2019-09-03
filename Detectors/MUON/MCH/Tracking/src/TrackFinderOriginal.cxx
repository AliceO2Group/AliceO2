// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackFinderOriginal.cxx
/// \brief Implementation of a class to reconstruct tracks with the original algorithm
///
/// \author Philippe Pillot, Subatech

#include "TrackFinderOriginal.h"

#include <iostream>
#include <stdexcept>

#include <TGeoGlobalMagField.h>
#include <TMatrixD.h>
#include <TMath.h>

#include "Field/MagneticField.h"
#include "TrackExtrap.h"

namespace o2
{
namespace mch
{

using namespace std;

constexpr float TrackFinderOriginal::SDefaultChamberZ[10];
constexpr double TrackFinderOriginal::SChamberThicknessInX0[10];
constexpr bool TrackFinderOriginal::SRequestStation[5];

//_________________________________________________________________________________________________
void TrackFinderOriginal::init(float l3Current, float dipoleCurrent)
{
  /// Prepare to run the algorithm

  // Create the magnetic field map if not already done
  mTrackFitter.initField(l3Current, dipoleCurrent);

  // Enable the track smoother
  mTrackFitter.smoothTracks(true);

  // Set the maximum MCS angle in chamber from the minimum acceptable momentum
  TrackParam param{};
  double inverseBendingP = (SMinBendingMomentum > 0.) ? 1. / SMinBendingMomentum : 1.;
  param.setInverseBendingMomentum(inverseBendingP);
  for (int iCh = 0; iCh < 10; ++iCh) {
    mMaxMCSAngle2[iCh] = TrackExtrap::getMCSAngle2(param, SChamberThicknessInX0[iCh], 1.);
  }
}

//_________________________________________________________________________________________________
const std::list<Track>& TrackFinderOriginal::findTracks(const std::array<std::list<Cluster>, 10>* clusters)
{
  /// Run the orginal track finder algorithm

  print("\n------------------ Start the original track finder ------------------");
  mClusters = clusters;
  mTracks.clear();

  // Look for candidates from clusters in stations(1..) 4 and 5
  print("\n--> Step 1: find track candidates\n");
  findTrackCandidates();
  if (mMoreCandidates) {
    findMoreTrackCandidates();
  }

  // Stop tracking if no candidate found
  if (mTracks.empty()) {
    return mTracks;
  }

  // Follow tracks in stations(1..) 3, 2 then 1
  print("\n--> Step 2: Follow track candidates\n");
  followTracks(mTracks.begin(), mTracks.end(), 2);

  // Complete the reconstructed tracks
  if (completeTracks()) {
    printTracks();
    removeDuplicateTracks();
  }
  print("Currently ", mTracks.size(), " candidates");
  printTracks();

  // Improve the reconstructed tracks
  improveTracks();
  print("Currently ", mTracks.size(), " candidates");
  printTracks();

  // Remove connected tracks in stations(1..) 3, 4 and 5
  removeConnectedTracks(3, 4);
  removeConnectedTracks(2, 2);

  // Set the final track parameters and covariances
  finalize();
  printTracks();
  /*
  // Refit the reconstructed tracks with a different resolution for mono-cathod clusters
  discardMonoCathodClusters();
*/
  return mTracks;
}

//_________________________________________________________________________________________________
void TrackFinderOriginal::findTrackCandidates()
{
  /// Track candidates are made of 2 clusters in station(1..) 5(4) and
  /// at least one compatible cluster in station(1..) 4(5) if requested

  for (int iSt = 4; iSt >= 3; --iSt) {

    // Find track candidates in the station
    auto itTrack = findTrackCandidates(2 * iSt, 2 * iSt + 1);

    for (; itTrack != mTracks.end();) {

      // Look for compatible clusters in the other station
      auto itNewTrack = followTrackInStation(itTrack, 7 - iSt);
      print("new candidates starting at position #", getTrackIndex(itNewTrack));

      // Keep the current candidate only if no compatible cluster is found and the station is not requested
      if (!SRequestStation[7 - iSt] && itNewTrack == mTracks.end()) {
        ++itTrack;
      } else {
        print("Removing original candidate now at position #", getTrackIndex(itTrack));
        printTrackParam(itTrack->first());
        itTrack = mTracks.erase(itTrack);
      }
    }

    print("Currently ", mTracks.size(), " candidates");
  }

  // Make sure each candidate is unique
  printTracks();
  removeDuplicateTracks();
  printTracks();

  print("Refitting tracks");
  auto itTrack(mTracks.begin());
  while (itTrack != mTracks.end()) {
    try {

      // Refit tracks using the Kalman filter
      mTrackFitter.fit(*itTrack, false);

      // Make sure they pass the selections
      if (!isAcceptable(itTrack->first())) {
        print("Removing candidate at position #", getTrackIndex(itTrack));
        itTrack = mTracks.erase(itTrack);
      } else {
        ++itTrack;
      }
    } catch (exception const&) {
      print("Removing candidate at position #", getTrackIndex(itTrack));
      itTrack = mTracks.erase(itTrack);
    }
  }

  print("Currently ", mTracks.size(), " candidates");
  printTracks();
}

//_________________________________________________________________________________________________
void TrackFinderOriginal::findMoreTrackCandidates()
{
  /// Track candidates are made of 1 cluster in station(1..) 4 and 1 in station 5
  /// possibly completed by a cluster in the second chamber of each station
  /// Candidates are formed by making linear propagation and refitted afterward

  // create an iterator to the last track of the list before adding new ones
  auto itCurrentTrack = mTracks.empty() ? mTracks.end() : std::prev(mTracks.end());

  for (int iCh1 = 6; iCh1 <= 7; ++iCh1) {

    for (int iCh2 = 8; iCh2 <= 9; ++iCh2) {

      // Find track candidates between these 2 chambers
      auto itTrack = findTrackCandidates(iCh1, iCh2, true);

      for (; itTrack != mTracks.end();) {

        // Look for compatible clusters in the second chamber of station (1..) 5
        auto itNewTrack = followLinearTrackInChamber(itTrack, 17 - iCh2);
        print("new candidates starting at position #", getTrackIndex(itNewTrack));

        // Keep the current candidate only if no compatible cluster is found
        if (itNewTrack == mTracks.end()) {
          itNewTrack = itTrack;
          ++itTrack;
        } else {
          print("Removing original candidate now at position #", getTrackIndex(itTrack));
          printTrackParam(itTrack->first());
          itTrack = mTracks.erase(itTrack);
        }

        for (; itNewTrack != itTrack;) {

          // Look for compatible clusters in the second chamber of station (1..) 4
          auto itNewNewTrack = followLinearTrackInChamber(itNewTrack, 13 - iCh1);
          print("new candidates starting at position #", getTrackIndex(itNewNewTrack));

          // Keep the current candidate only if no compatible cluster is found
          if (itNewNewTrack == mTracks.end()) {
            ++itNewTrack;
          } else {
            print("Removing original candidate now at position #", getTrackIndex(itNewTrack));
            printTrackParam(itTrack->first());
            itNewTrack = mTracks.erase(itNewTrack);
          }
        }
      }

      print("Currently ", mTracks.size(), " candidates");
    }
  }

  print("Refitting tracks");
  itCurrentTrack = (itCurrentTrack == mTracks.end()) ? mTracks.begin() : ++itCurrentTrack;
  while (itCurrentTrack != mTracks.end()) {
    try {

      // Refit tracks using the Kalman filter
      mTrackFitter.fit(*itCurrentTrack, false);

      // Make sure they pass the selections
      if (!isAcceptable(itCurrentTrack->first())) {
        print("Removing candidate at position #", getTrackIndex(itCurrentTrack));
        itCurrentTrack = mTracks.erase(itCurrentTrack);
      } else {
        ++itCurrentTrack;
      }
    } catch (exception const&) {
      print("Removing candidate at position #", getTrackIndex(itCurrentTrack));
      itCurrentTrack = mTracks.erase(itCurrentTrack);
    }
  }

  print("Currently ", mTracks.size(), " candidates");
  printTracks();
}

//_________________________________________________________________________________________________
std::list<Track>::iterator TrackFinderOriginal::findTrackCandidates(int ch1, int ch2, bool skipUsedPairs)
{
  /// Find all combinations of clusters between the 2 chambers that could belong to a valid track
  /// If skipUsedPairs == true: skip combinations of clusters already part of a track
  /// New candidates are added at the end of the track list
  /// Return an iterator to the first candidate found

  print("Looking for candidates between chamber ", ch1, " and ", ch2);

  double bendingVertexDispersion2 = SBendingVertexDispersion * SBendingVertexDispersion;

  // maximum impact parameter dispersion**2 due to MCS in chambers
  double impactMCS2(0.);
  for (int iCh = 0; iCh <= ch1; ++iCh) {
    impactMCS2 += SDefaultChamberZ[iCh] * SDefaultChamberZ[iCh] * mMaxMCSAngle2[iCh];
  }

  // create an iterator to the last track of the list before adding new ones
  auto itTrack = mTracks.empty() ? mTracks.end() : std::prev(mTracks.end());

  for (const auto& cluster1 : mClusters->at(ch1)) {

    double z1 = cluster1.getZ();

    for (const auto& cluster2 : mClusters->at(ch2)) {

      // skip combinations of clusters already part of a track if requested
      if (skipUsedPairs && areUsed(cluster1, cluster2)) {
        continue;
      }

      double z2 = cluster2.getZ();
      double dZ = z1 - z2;

      // check if non bending impact parameter is within tolerances
      double nonBendingSlope = (cluster1.getX() - cluster2.getX()) / dZ;
      double nonBendingImpactParam = TMath::Abs(cluster1.getX() - cluster1.getZ() * nonBendingSlope);
      double nonBendingImpactParamErr = TMath::Sqrt((z1 * z1 * cluster2.getEx2() + z2 * z2 * cluster1.getEx2()) / dZ / dZ + impactMCS2);
      if ((nonBendingImpactParam - SSigmaCutForTracking * nonBendingImpactParamErr) > (3. * SNonBendingVertexDispersion)) {
        continue;
      }

      double bendingSlope = (cluster1.getY() - cluster2.getY()) / dZ;
      if (TrackExtrap::isFieldON()) { // depending whether the field is ON or OFF
        // check if bending momentum is within tolerances
        double bendingImpactParam = cluster1.getY() - cluster1.getZ() * bendingSlope;
        double bendingImpactParamErr2 = (z1 * z1 * cluster2.getEy2() + z2 * z2 * cluster1.getEy2()) / dZ / dZ + impactMCS2;
        double bendingMomentum = TMath::Abs(TrackExtrap::getBendingMomentumFromImpactParam(bendingImpactParam));
        double bendingMomentumErr = TMath::Sqrt((bendingVertexDispersion2 + bendingImpactParamErr2) / bendingImpactParam / bendingImpactParam + 0.01) * bendingMomentum;
        if ((bendingMomentum + 3. * bendingMomentumErr) < SMinBendingMomentum) {
          continue;
        }
      } else {
        // or check if bending impact parameter is within tolerances
        double bendingImpactParam = TMath::Abs(cluster1.getY() - cluster1.getZ() * bendingSlope);
        double bendingImpactParamErr = TMath::Sqrt((z1 * z1 * cluster2.getEy2() + z2 * z2 * cluster1.getEy2()) / dZ / dZ + impactMCS2);
        if ((bendingImpactParam - SSigmaCutForTracking * bendingImpactParamErr) > (3. * SBendingVertexDispersion)) {
          continue;
        }
      }

      // create a new track candidate
      createTrack(cluster1, cluster2);
    }
  }

  print("Currently ", mTracks.size(), " candidates");

  return (itTrack == mTracks.end()) ? mTracks.begin() : ++itTrack;
}

//_________________________________________________________________________________________________
bool TrackFinderOriginal::areUsed(const Cluster& cl1, const Cluster& cl2)
{
  /// Return true if the 2 clusters are already part of a track

  for (const auto& track : mTracks) {

    bool cl1Used(false), cl2Used(false);

    for (auto itParam = track.rbegin(); itParam != track.rend(); ++itParam) {

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
void TrackFinderOriginal::createTrack(const Cluster& cl1, const Cluster& cl2)
{
  /// Create a new track with these 2 clusters and store it at the end of the list of tracks
  /// Compute the track parameters and covariance matrices at the 2 clusters

  print("Creating a new candidate");

  // create the track and the trackParam at each cluster
  mTracks.emplace_back();
  TrackParam& param2 = mTracks.back().createParamAtCluster(cl2);
  TrackParam& param1 = mTracks.back().createParamAtCluster(cl1);

  // Compute track parameters
  double dZ = cl1.getZ() - cl2.getZ();
  double nonBendingSlope = (cl1.getX() - cl2.getX()) / dZ;
  double bendingSlope = (cl1.getY() - cl2.getY()) / dZ;
  double bendingImpact = cl1.getY() - cl1.getZ() * bendingSlope;
  double inverseBendingMomentum = 1. / TrackExtrap::getBendingMomentumFromImpactParam(bendingImpact);

  // Set track parameters at first cluster
  param1.setNonBendingCoor(cl1.getX());
  param1.setNonBendingSlope(nonBendingSlope);
  param1.setBendingCoor(cl1.getY());
  param1.setBendingSlope(bendingSlope);
  param1.setInverseBendingMomentum(inverseBendingMomentum);

  // Set track parameters at last cluster
  param2.setNonBendingCoor(cl2.getX());
  param2.setNonBendingSlope(nonBendingSlope);
  param2.setBendingCoor(cl2.getY());
  param2.setBendingSlope(bendingSlope);
  param2.setInverseBendingMomentum(inverseBendingMomentum);

  // Compute and set track parameters covariances at first cluster
  TMatrixD paramCov(5, 5);
  paramCov.Zero();
  // Non bending plane
  double cl1Ex2 = cl1.getEx2();
  double cl2Ex2 = cl2.getEx2();
  paramCov(0, 0) = cl1Ex2;
  paramCov(0, 1) = cl1Ex2 / dZ;
  paramCov(1, 0) = paramCov(0, 1);
  paramCov(1, 1) = (cl1Ex2 + cl2Ex2) / dZ / dZ;
  // Bending plane
  double cl1Ey2 = cl1.getEy2();
  double cl2Ey2 = cl2.getEy2();
  paramCov(2, 2) = cl1Ey2;
  paramCov(2, 3) = cl1Ey2 / dZ;
  paramCov(3, 2) = paramCov(2, 3);
  paramCov(3, 3) = (cl1Ey2 + cl2Ey2) / dZ / dZ;
  // Inverse bending momentum (vertex resolution + bending slope resolution + 10% error on dipole parameters+field)
  if (TrackExtrap::isFieldON()) {
    paramCov(4, 4) = ((SBendingVertexDispersion * SBendingVertexDispersion +
                       (cl1.getZ() * cl1.getZ() * cl2Ey2 + cl2.getZ() * cl2.getZ() * cl1Ey2) / dZ / dZ) /
                        bendingImpact / bendingImpact +
                      0.1 * 0.1) *
                     inverseBendingMomentum * inverseBendingMomentum;
    paramCov(2, 4) = -cl2.getZ() * cl1Ey2 * inverseBendingMomentum / bendingImpact / dZ;
    paramCov(4, 2) = paramCov(2, 4);
    paramCov(3, 4) = -(cl1.getZ() * cl2Ey2 + cl2.getZ() * cl1Ey2) * inverseBendingMomentum / bendingImpact / dZ / dZ;
    paramCov(4, 3) = paramCov(3, 4);
  } else {
    paramCov(4, 4) = inverseBendingMomentum * inverseBendingMomentum;
  }
  param1.setCovariances(paramCov);

  // Compute and set track parameters covariances at last cluster
  // Non bending plane
  paramCov(0, 0) = cl2Ex2;
  paramCov(0, 1) = -cl2Ex2 / dZ;
  paramCov(1, 0) = paramCov(0, 1);
  // Bending plane
  paramCov(2, 2) = cl2Ey2;
  paramCov(2, 3) = -cl2Ey2 / dZ;
  paramCov(3, 2) = paramCov(2, 3);
  // Inverse bending momentum (vertex resolution + bending slope resolution + 10% error on dipole parameters+field)
  if (TrackExtrap::isFieldON()) {
    paramCov(2, 4) = cl1.getZ() * cl2Ey2 * inverseBendingMomentum / bendingImpact / dZ;
    paramCov(4, 2) = paramCov(2, 4);
  }
  param2.setCovariances(paramCov);

  printTrackParam(param1);
}

//_________________________________________________________________________________________________
bool TrackFinderOriginal::isAcceptable(const TrackParam& param) const
{
  /// Return true if the track is within given limits on momentum/angle/origin

  const TMatrixD& paramCov = param.getCovariances();
  int chamber = param.getClusterPtr()->getChamberId();
  double z = param.getZ();

  // impact parameter dispersion**2 due to MCS in chambers
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
void TrackFinderOriginal::removeDuplicateTracks()
{
  /// If two tracks contain exactly the same clusters, keep only the first one
  /// If a track contains all the clusters of another, remove that other track

  print("Remove duplicated tracks");

  for (auto itTrack1 = mTracks.begin(); itTrack1 != mTracks.end();) {

    int nClusters1 = itTrack1->getNClusters();
    bool track1Removed(false);

    for (auto itTrack2 = std::next(itTrack1); itTrack2 != mTracks.end();) {

      int nClusters2 = itTrack2->getNClusters();
      int nClustersInCommon = itTrack2->getNClustersInCommon(*itTrack1);

      if (nClustersInCommon == nClusters2) {
        print("Removing candidate at position #", getTrackIndex(itTrack2));
        itTrack2 = mTracks.erase(itTrack2);
      } else if (nClustersInCommon == nClusters1) {
        print("Removing candidate at position #", getTrackIndex(itTrack1));
        itTrack1 = mTracks.erase(itTrack1);
        track1Removed = true;
        break;
      } else {
        ++itTrack2;
      }
    }

    if (!track1Removed) {
      ++itTrack1;
    }
  }

  print("Currently ", mTracks.size(), " candidates");
}

//_________________________________________________________________________________________________
void TrackFinderOriginal::removeConnectedTracks(int stMin, int stMax)
{
  /// Find and remove tracks sharing 1 cluster or more in station(s) [stMin, stMax].
  /// For each couple of connected tracks, one removes the one with the smallest
  /// number of clusters or with the highest chi2 value in case of equality.

  print("Remove connected tracks in stations [", stMin, ", ", stMax, "]");

  // first loop to tag the tracks to remove...
  for (auto itTrack1 = mTracks.begin(); itTrack1 != mTracks.end(); ++itTrack1) {
    for (auto itTrack2 = std::next(itTrack1); itTrack2 != mTracks.end(); ++itTrack2) {
      if (itTrack2->getNClustersInCommon(*itTrack1, stMin, stMax) > 0) {
        if (itTrack2->isBetter(*itTrack1)) {
          itTrack1->connected();
        } else {
          itTrack2->connected();
        }
      }
    }
  }

  // ...then remove them. That way all combinations are tested.
  for (auto itTrack = mTracks.begin(); itTrack != mTracks.end();) {
    if (itTrack->isConnected()) {
      print("Removing candidate at position #", getTrackIndex(itTrack));
      itTrack = mTracks.erase(itTrack);
    } else {
      ++itTrack;
    }
  }

  print("Currently ", mTracks.size(), " candidates");
}

//_________________________________________________________________________________________________
void TrackFinderOriginal::followTracks(const std::list<Track>::iterator& itTrackBegin,
                                       const std::list<Track>::iterator& itTrackEnd, int nextStation)
{
  /// Follow the track candidates in the range [itTrackBegin, itTrackEnd[ down to the first station,
  /// starting from the station "nextStation", and look for compatible clusters.
  /// At least one cluster per requested station is needed to continue with the candidate
  /// If several clusters are found the candidate is duplicated to consider all possibilities

  print("Follow track candidates #", getTrackIndex(itTrackBegin), " to #", getTrackIndex(std::prev(itTrackEnd)), " in station ", nextStation);

  for (auto itTrack = itTrackBegin; itTrack != itTrackEnd;) {

    // Look for compatible clusters in the next station
    auto itNewTrack = followTrackInStation(itTrack, nextStation);

    // Try to recover the track if no compatible cluster is found
    if (itNewTrack == mTracks.end()) {

      // Keep the case where no cluster is found as a possible candidate if the next station is not requested
      if (!SRequestStation[nextStation]) {
        print("Duplicate original candidate");
        itTrack = mTracks.emplace(itTrack, *itTrack);
      }

      // Try to recover
      itNewTrack = recoverTrack(itTrack, nextStation);

      // Remove the initial candidate or its copy
      print("Removing original candidate or its copy at position #", getTrackIndex(itTrack));
      printTrackParam(itTrack->first());
      itTrack = mTracks.erase(itTrack);

      // If the next station is not requested, we can at minimum continue with the initial candidate
      if (!SRequestStation[nextStation]) {
        if (itNewTrack == mTracks.end()) {
          itNewTrack = itTrack;
        }
        ++itTrack;
      }

    } else {

      // Or remove the initial candidate if new candidates have been produced
      print("Removing original candidate now at position #", getTrackIndex(itTrack));
      printTrackParam(itTrack->first());
      itTrack = mTracks.erase(itTrack);
    }

    print("Currently ", mTracks.size(), " candidates");
    printTracks();

    // follow the new candidates, if any, down to the first station
    if (itNewTrack != mTracks.end() && nextStation != 0) {
      followTracks(itNewTrack, itTrack, nextStation - 1);
    }
  }
}

//_________________________________________________________________________________________________
std::list<Track>::iterator TrackFinderOriginal::followTrackInStation(const std::list<Track>::iterator& itTrack, int nextStation)
{
  /// Follow the track candidate pointed to by "itTrack" to the given station and look for compatible cluster(s)
  /// A new candidate is produced for each (pair of) compatible cluster(s) found and inserted just before "itTrack"
  /// The method returns an iterator to the first new candidate, or mTracks.end() if no compatible cluster is found
  /// In any case, the initial candidate pointed to by "itTrack" remains untouched

  print("Follow track candidate #", getTrackIndex(itTrack), " in station ", nextStation);
  printTrackParam(itTrack->first());

  auto itNewTrack(itTrack);
  TrackParam extrapTrackParamAtCluster1{};
  TrackParam extrapTrackParamAtCluster2{};
  TrackParam extrapTrackParam{};

  // Order the chambers according to the propagation direction (tracking starts with ch2):
  // - nextStation == station(1...) 5 => forward propagation
  // - nextStation < station(1...) 5 => backward propagation
  int ch1(0), ch2(0);
  if (nextStation == 4) {
    ch1 = 2 * nextStation + 1;
    ch2 = 2 * nextStation;
  } else {
    ch1 = 2 * nextStation;
    ch2 = 2 * nextStation + 1;
  }

  // Maximum chi2 to accept a cluster candidate (the factor 2 is for the 2 degrees of freedom: x and y)
  double maxChi2OfCluster = 2. * SSigmaCutForTracking * SSigmaCutForTracking;

  // Get the current track parameters according to the propagation direction
  TrackParam extrapTrackParamAtCh = (nextStation == 4) ? itTrack->last() : itTrack->first();

  // Add MCS effects in the current chamber
  int currentChamber(extrapTrackParamAtCh.getClusterPtr()->getChamberId());
  TrackExtrap::addMCSEffect(&extrapTrackParamAtCh, SChamberThicknessInX0[currentChamber], -1.);

  // Reset the propagator for the smoother
  if (mTrackFitter.isSmootherEnabled()) {
    extrapTrackParamAtCh.resetPropagator();
  }

  // Add MCS effects in the missing chamber(s) if any
  while (ch1 < ch2 && currentChamber > ch2 + 1) {
    --currentChamber;
    if (!TrackExtrap::extrapToZCov(&extrapTrackParamAtCh, SDefaultChamberZ[currentChamber],
                                   mTrackFitter.isSmootherEnabled())) {
      return mTracks.end();
    }
    TrackExtrap::addMCSEffect(&extrapTrackParamAtCh, SChamberThicknessInX0[currentChamber], -1.);
  }

  //Extrapolate the track candidate to chamber 2
  if (!TrackExtrap::extrapToZCov(&extrapTrackParamAtCh, SDefaultChamberZ[ch2], mTrackFitter.isSmootherEnabled())) {
    return mTracks.end();
  }

  // Prepare to remember the clusters used in ch1 in combination with a cluster in ch2
  std::vector<bool> clusterCh1Used(mClusters->at(ch1).size(), false);

  // Look for cluster candidates in chamber 2
  for (const auto& clusterCh2 : mClusters->at(ch2)) {

    // Fast try to add the current cluster
    if (!tryOneClusterFast(extrapTrackParamAtCh, clusterCh2)) {
      continue;
    }

    // Try to add the current cluster accurately
    if (tryOneCluster(extrapTrackParamAtCh, clusterCh2, extrapTrackParamAtCluster2,
                      mTrackFitter.isSmootherEnabled()) >= maxChi2OfCluster) {
      continue;
    }

    // Save the extrapolated parameters and covariances for the smoother
    if (mTrackFitter.isSmootherEnabled()) {
      extrapTrackParamAtCluster2.setExtrapParameters(extrapTrackParamAtCluster2.getParameters());
      extrapTrackParamAtCluster2.setExtrapCovariances(extrapTrackParamAtCluster2.getCovariances());
    }

    // Compute the new track parameters including clusterCh2 using the Kalman filter
    try {
      mTrackFitter.runKalmanFilter(extrapTrackParamAtCluster2);
    } catch (exception const&) {
      continue;
    }

    // skip tracks out of limits
    if (!isAcceptable(extrapTrackParamAtCluster2)) {
      continue;
    }

    // copy the new track parameters for the next step and add MCS effects in chamber 2
    extrapTrackParam = extrapTrackParamAtCluster2;
    TrackExtrap::addMCSEffect(&extrapTrackParam, SChamberThicknessInX0[ch2], -1.);

    // Reset the propagator for the smoother
    if (mTrackFitter.isSmootherEnabled()) {
      extrapTrackParam.resetPropagator();
    }

    //Extrapolate the track candidate to chamber 1
    bool foundSecondCluster(false);
    if (TrackExtrap::extrapToZCov(&extrapTrackParam, SDefaultChamberZ[ch1], mTrackFitter.isSmootherEnabled())) {

      // look for second cluster candidates in chamber 1
      int iCluster1(-1);
      for (const auto& clusterCh1 : mClusters->at(ch1)) {

        ++iCluster1;

        // Fast try to add the current cluster
        if (!tryOneClusterFast(extrapTrackParam, clusterCh1)) {
          continue;
        }

        // Try to add the current cluster accurately
        if (tryOneCluster(extrapTrackParam, clusterCh1, extrapTrackParamAtCluster1,
                          mTrackFitter.isSmootherEnabled()) >= maxChi2OfCluster) {
          continue;
        }

        // Save the extrapolated parameters and covariances for the smoother
        if (mTrackFitter.isSmootherEnabled()) {
          extrapTrackParamAtCluster1.setExtrapParameters(extrapTrackParamAtCluster1.getParameters());
          extrapTrackParamAtCluster1.setExtrapCovariances(extrapTrackParamAtCluster1.getCovariances());
        }

        // Compute the new track parameters including clusterCh1 using the Kalman filter
        try {
          mTrackFitter.runKalmanFilter(extrapTrackParamAtCluster1);
        } catch (exception const&) {
          continue;
        }

        // Skip tracks out of limits
        if (!isAcceptable(extrapTrackParamAtCluster1)) {
          continue;
        }

        // Copy the initial candidate into a new track with these 2 clusters added
        print("Duplicate the candidate");
        itNewTrack = mTracks.emplace(itNewTrack, *itTrack);
        updateTrack(*itNewTrack, extrapTrackParamAtCluster1, extrapTrackParamAtCluster2);

        // Tag clusterCh1 as used
        clusterCh1Used[iCluster1] = true;
        foundSecondCluster = true;
      }
    }

    // If no clusterCh1 found then copy the initial candidate into a new track with only clusterCh2 added
    if (!foundSecondCluster) {
      print("Duplicate the candidate");
      itNewTrack = mTracks.emplace(itNewTrack, *itTrack);
      updateTrack(*itNewTrack, extrapTrackParamAtCluster2);
    }
  }

  // Add MCS effects in chamber 2
  TrackExtrap::addMCSEffect(&extrapTrackParamAtCh, SChamberThicknessInX0[ch2], -1.);

  //Extrapolate the track candidate to chamber 1
  if (!TrackExtrap::extrapToZCov(&extrapTrackParamAtCh, SDefaultChamberZ[ch1], mTrackFitter.isSmootherEnabled())) {
    return (itNewTrack == itTrack) ? mTracks.end() : itNewTrack;
  }

  // look for cluster candidates not already used in chamber 1
  int iCluster1(-1);
  for (const auto& clusterCh1 : mClusters->at(ch1)) {

    ++iCluster1;
    if (clusterCh1Used[iCluster1]) {
      continue;
    }

    // Fast try to add the current cluster
    if (!tryOneClusterFast(extrapTrackParamAtCh, clusterCh1)) {
      continue;
    }

    // Try to add the current cluster accurately
    if (tryOneCluster(extrapTrackParamAtCh, clusterCh1, extrapTrackParamAtCluster1,
                      mTrackFitter.isSmootherEnabled()) >= maxChi2OfCluster) {
      continue;
    }

    // Save the extrapolated parameters and covariances for the smoother
    if (mTrackFitter.isSmootherEnabled()) {
      extrapTrackParamAtCluster1.setExtrapParameters(extrapTrackParamAtCluster1.getParameters());
      extrapTrackParamAtCluster1.setExtrapCovariances(extrapTrackParamAtCluster1.getCovariances());
    }

    // Compute the new track parameters including clusterCh1 using the Kalman filter
    try {
      mTrackFitter.runKalmanFilter(extrapTrackParamAtCluster1);
    } catch (exception const&) {
      continue;
    }

    // Skip tracks out of limits
    if (!isAcceptable(extrapTrackParamAtCluster1)) {
      continue;
    }

    // Copy the initial candidate into a new track with clusterCh1 added
    print("Duplicate the candidate");
    itNewTrack = mTracks.emplace(itNewTrack, *itTrack);
    updateTrack(*itNewTrack, extrapTrackParamAtCluster1);
  }

  return (itNewTrack == itTrack) ? mTracks.end() : itNewTrack;
}

//_________________________________________________________________________________________________
std::list<Track>::iterator TrackFinderOriginal::followLinearTrackInChamber(const std::list<Track>::iterator& itTrack, int nextChamber)
{
  /// Follow linearly the track candidate pointed to by "itTrack" to the given chamber and look for compatible cluster(s)
  /// The first (or last) cluster attached to the candidate is supposed to be on the other chamber of the same station
  /// A new candidate is produced for each compatible cluster found and inserted just before "itTrack"
  /// The method returns an iterator to the first new candidate, or mTracks.end() if no compatible cluster is found
  /// In any case, the initial candidate pointed to by "itTrack" remains untouched

  print("Follow track candidate #", getTrackIndex(itTrack), " in chamber ", nextChamber);
  printTrackParam(itTrack->first());

  auto itNewTrack(itTrack);
  TrackParam extrapTrackParamAtCluster{};

  // Maximum chi2 to accept a cluster candidate (the factor 2 is for the 2 degrees of freedom: x and y)
  double maxChi2OfCluster = 2. * SSigmaCutForTracking * SSigmaCutForTracking;

  // Get the current track parameters according to the propagation direction
  TrackParam trackParam = (nextChamber > 7) ? itTrack->last() : itTrack->first();

  // Add MCS effects in the current chamber
  TrackExtrap::addMCSEffect(&trackParam, SChamberThicknessInX0[trackParam.getClusterPtr()->getChamberId()], -1.);

  // Look for cluster candidates in the next chamber
  for (const auto& cluster : mClusters->at(nextChamber)) {

    // Fast try to add the current cluster
    if (!tryOneClusterFast(trackParam, cluster)) {
      continue;
    }

    // propagate linearly the track to the z position of the current cluster
    extrapTrackParamAtCluster = trackParam;
    TrackExtrap::linearExtrapToZCov(&extrapTrackParamAtCluster, cluster.getZ());

    // Try to add the current cluster accurately
    if (tryOneCluster(extrapTrackParamAtCluster, cluster, extrapTrackParamAtCluster, false) >= maxChi2OfCluster) {
      continue;
    }

    // Copy the initial candidate into a new track with cluster added
    print("Duplicate the candidate");
    itNewTrack = mTracks.emplace(itNewTrack, *itTrack);
    updateTrack(*itNewTrack, extrapTrackParamAtCluster);
  }

  return (itNewTrack == itTrack) ? mTracks.end() : itNewTrack;
}

//_________________________________________________________________________________________________
bool TrackFinderOriginal::tryOneClusterFast(const TrackParam& param, const Cluster& cluster)
{
  /// Quickly test the compatibility between the track and the cluster
  /// given the track and cluster resolutions + the maximum-distance-to-track value
  /// and assuming linear propagation of the track to the z position of the cluster
  /// Return true if they are compatibles

  double dZ = cluster.getZ() - param.getZ();
  double dX = cluster.getX() - (param.getNonBendingCoor() + param.getNonBendingSlope() * dZ);
  double dY = cluster.getY() - (param.getBendingCoor() + param.getBendingSlope() * dZ);
  const TMatrixD& paramCov = param.getCovariances();
  double errX2 = paramCov(0, 0) + dZ * dZ * paramCov(1, 1) + 2. * dZ * paramCov(0, 1) + cluster.getEx2();
  double errY2 = paramCov(2, 2) + dZ * dZ * paramCov(3, 3) + 2. * dZ * paramCov(2, 3) + cluster.getEy2();

  double dXmax = SSigmaCutForTracking * TMath::Sqrt(2. * errX2) + SMaxNonBendingDistanceToTrack;
  double dYmax = SSigmaCutForTracking * TMath::Sqrt(2. * errY2) + SMaxBendingDistanceToTrack;

  if (TMath::Abs(dX) > dXmax || TMath::Abs(dY) > dYmax) {
    return false;
  }
  return true;
}

//_________________________________________________________________________________________________
double TrackFinderOriginal::tryOneCluster(const TrackParam& param, const Cluster& cluster, TrackParam& paramAtCluster,
                                          bool updatePropagator)
{
  /// Test the compatibility between the track and the cluster
  /// given the track covariance matrix and the cluster resolution
  /// and propagating properly the track to the z position of the cluster
  /// Return the matching chi2 and the track parameters at the cluster

  // Extrapolate the track parameters and covariances at the z position of the cluster
  paramAtCluster = param;
  paramAtCluster.setClusterPtr(&cluster);
  if (!TrackExtrap::extrapToZCov(&paramAtCluster, cluster.getZ(), updatePropagator)) {
    return mTrackFitter.getMaxChi2();
  }

  // Compute the cluster-track residuals in bending and non bending directions
  double dX = cluster.getX() - paramAtCluster.getNonBendingCoor();
  double dY = cluster.getY() - paramAtCluster.getBendingCoor();

  // Combine the cluster and track resolutions and covariances
  const TMatrixD& paramCov = paramAtCluster.getCovariances();
  double sigmaX2 = paramCov(0, 0) + cluster.getEx2();
  double sigmaY2 = paramCov(2, 2) + cluster.getEy2();
  double covXY = paramCov(0, 2);
  double det = sigmaX2 * sigmaY2 - covXY * covXY;

  // Compute and return the matching chi2
  if (det == 0.) {
    return mTrackFitter.getMaxChi2();
  }
  return (dX * dX * sigmaY2 + dY * dY * sigmaX2 - 2. * dX * dY * covXY) / det;
}

//_________________________________________________________________________________________________
void TrackFinderOriginal::updateTrack(Track& track, TrackParam& trackParamAtCluster)
{
  /// Add to the track candidate the track parameters computed at the new cluster found in the station

  // Flag the cluster as being (not) removable
  if (SRequestStation[trackParamAtCluster.getClusterPtr()->getChamberId() / 2]) {
    trackParamAtCluster.setRemovable(false);
  } else {
    trackParamAtCluster.setRemovable(true);
  }

  // No need to compute the chi2 in this case, just set it to minimum possible value
  trackParamAtCluster.setLocalChi2(0.);

  // Add the parameters at the new cluster
  track.addParamAtCluster(trackParamAtCluster);
}

//_________________________________________________________________________________________________
void TrackFinderOriginal::updateTrack(Track& track, TrackParam& trackParamAtCluster1, TrackParam& trackParamAtCluster2)
{
  /// Add to the track candidate the track parameters computed at the 2 new clusters found in the station
  /// trackParamAtCluster1 are supposed to have been computed after cluster2 has already been attached to the track

  // Flag the clusters as being removable
  trackParamAtCluster1.setRemovable(true);
  trackParamAtCluster2.setRemovable(true);

  // Compute the local chi2 at cluster1
  const Cluster* cluster1(trackParamAtCluster1.getClusterPtr());
  double deltaX = trackParamAtCluster1.getNonBendingCoor() - cluster1->getX();
  double deltaY = trackParamAtCluster1.getBendingCoor() - cluster1->getY();
  double localChi2 = deltaX * deltaX / cluster1->getEx2() + deltaY * deltaY / cluster1->getEy2();
  trackParamAtCluster1.setLocalChi2(localChi2);

  // Compute the local chi2 at cluster2
  const Cluster* cluster2(trackParamAtCluster2.getClusterPtr());
  TrackParam extrapTrackParamAtCluster2(trackParamAtCluster1);
  TrackExtrap::extrapToZ(&extrapTrackParamAtCluster2, trackParamAtCluster2.getZ());
  deltaX = extrapTrackParamAtCluster2.getNonBendingCoor() - cluster2->getX();
  deltaY = extrapTrackParamAtCluster2.getBendingCoor() - cluster2->getY();
  localChi2 = deltaX * deltaX / cluster2->getEx2() + deltaY * deltaY / cluster2->getEy2();
  trackParamAtCluster2.setLocalChi2(localChi2);

  // Add the parameters at the new clusters
  track.addParamAtCluster(trackParamAtCluster2);
  track.addParamAtCluster(trackParamAtCluster1);
}

//_________________________________________________________________________________________________
std::list<Track>::iterator TrackFinderOriginal::recoverTrack(std::list<Track>::iterator& itTrack, int nextStation)
{
  /// Try to recover the track candidate pointed to by "itTrack" in the next station
  /// by removing the worst of the clusters attached in the current station and resuming the tracking
  /// The method returns an iterator to the first new candidate, or mTracks.end() if no compatible cluster is found
  /// The initial candidate pointed to by "itTrack" is not removed but it is potentially modified

  // Do not try to recover track until we have attached cluster(s) on station(1..) 3
  if (nextStation > 1) {
    return mTracks.end();
  }

  print("Try to recover the track candidate #", getTrackIndex(itTrack), " in station ", nextStation);

  // Look for the cluster to remove
  auto itWorstParam = itTrack->end();
  double worstLocalChi2(-1.);
  for (auto itParam = itTrack->begin(); itParam != itTrack->end(); ++itParam) {

    // Check if the current cluster is in the previous station
    if (itParam->getClusterPtr()->getChamberId() / 2 != nextStation + 1) {
      break;
    }

    // Check if the current cluster is removable
    if (!itParam->isRemovable()) {
      return mTracks.end();
    }

    // Reset the current cluster as being not removable if it is on a requested station
    if (SRequestStation[nextStation + 1]) {
      itParam->setRemovable(false);
    }

    // Pick up the cluster with the worst chi2
    if (itParam->getLocalChi2() > worstLocalChi2) {
      worstLocalChi2 = itParam->getLocalChi2();
      itWorstParam = itParam;
    }
  }

  // Check if the cluster to remove has been found
  if (itWorstParam == itTrack->end()) {
    return mTracks.end();
  }

  // Remove the worst cluster
  auto itParam = itTrack->removeParamAtCluster(itWorstParam);

  // refit the track from the second cluster as currently done in AliRoot
  //itParam = std::next(itTrack->begin());

  // In case the cluster removed was not the last attached:
  if (itParam != itTrack->begin()) {

    // Recompute the track parameters at the clusters upstream the one removed, starting from the parameters downstream
    try {
      auto ritParam = std::make_reverse_iterator(++itParam);
      mTrackFitter.fit(*itTrack, false, false, &ritParam);
    } catch (exception const&) {
      return mTracks.end();
    }

    // Skip tracks out of limits
    if (!isAcceptable(itTrack->first())) {
      return mTracks.end();
    }
  }

  // Look for new cluster(s) in the next station
  return followTrackInStation(itTrack, nextStation);
}

//_________________________________________________________________________________________________
bool TrackFinderOriginal::completeTracks()
{
  /// Complete tracks by adding missing clusters (if there is an overlap between
  /// two detection elements, the track may have two clusters in the same chamber)
  /// If several compatible clusters are found on a chamber only the best one is attached
  /// Refit the entire track if it has been completed and remove it if the fit fails
  /// Return true if one or more tracks have been completed

  print("Complete tracks");

  bool completionOccurred(false);
  TrackParam paramAtCluster{};
  TrackParam bestParamAtCluster{};

  // Maximum chi2 to accept a cluster candidate (the factor 2 is for the 2 degrees of freedom: x and y)
  double maxChi2OfCluster = 2. * SSigmaCutForTracking * SSigmaCutForTracking;

  for (auto itTrack = mTracks.begin(); itTrack != mTracks.end();) {

    bool trackCompleted(false);

    for (auto itParam = itTrack->begin(), itPrevParam = itTrack->begin(); itParam != itTrack->end();) {

      // Prepare access to the next trackParam before adding new cluster as it can be added in-between
      auto itNextParam = std::next(itParam);

      // Never test new clusters starting from parameters at last cluster because the covariance matrix is meaningless
      TrackParam* param = (itNextParam == itTrack->end()) ? &*itPrevParam : &*itParam;

      // Look for a second cluster candidate in the same chamber
      int deId = itParam->getClusterPtr()->getDEId();
      double bestChi2AtCluster = mTrackFitter.getMaxChi2();
      for (const auto& cluster : mClusters->at(itParam->getClusterPtr()->getChamberId())) {

        // In another detection element
        if (cluster.getDEId() == deId) {
          continue;
        }

        // Fast try to add the current cluster
        if (!tryOneClusterFast(*param, cluster)) {
          continue;
        }

        // Try to add the current cluster accurately
        if (tryOneCluster(*param, cluster, paramAtCluster, false) >= maxChi2OfCluster) {
          continue;
        }

        // Compute the new track parameters including the cluster using the Kalman filter
        try {
          mTrackFitter.runKalmanFilter(paramAtCluster);
        } catch (exception const&) {
          continue;
        }

        // Keep the best cluster
        if (paramAtCluster.getTrackChi2() < bestChi2AtCluster) {
          bestChi2AtCluster = paramAtCluster.getTrackChi2();
          bestParamAtCluster = paramAtCluster;
        }
      }

      // Add the new cluster if any (should be added either just before or just after itParam)
      if (bestChi2AtCluster < mTrackFitter.getMaxChi2()) {
        itParam->setRemovable(true);
        bestParamAtCluster.setRemovable(true);
        itTrack->addParamAtCluster(bestParamAtCluster);
        trackCompleted = kTRUE;
        completionOccurred = kTRUE;
      }

      itPrevParam = itParam;
      itParam = itNextParam;
    }

    // If the track has been completed, refit it using the Kalman filter and remove it in case of failure
    if (trackCompleted) {
      try {
        mTrackFitter.fit(*itTrack, false);
        print("Candidate at position #", getTrackIndex(itTrack), " completed");
        ++itTrack;
      } catch (exception const&) {
        print("Removing candidate at position #", getTrackIndex(itTrack));
        itTrack = mTracks.erase(itTrack);
      }
    } else {
      ++itTrack;
    }
  }

  return completionOccurred;
}

//_________________________________________________________________________________________________
void TrackFinderOriginal::improveTracks()
{
  /// Improve tracks by removing removable clusters with local chi2 higher than the defined cut
  /// Removable clusters are identified by the method Track::tagRemovableClusters()
  /// Recompute track parameters and covariances at the remaining clusters
  /// Remove the track if it cannot be improved or in case of failure

  print("Improve tracks");

  // The smoother must be enabled to compute the local chi2 at each cluster
  if (!mTrackFitter.isSmootherEnabled()) {
    LOG(ERROR) << "Smoother disabled --> tracks cannot be improved";
    return;
  }

  // Maximum chi2 to keep a cluster (the factor 2 is for the 2 degrees of freedom: x and y)
  double maxChi2OfCluster = 2. * SSigmaCutForImprovement * SSigmaCutForImprovement;

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
      itTrack->tagRemovableClusters(requestedStationMask());

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
      print("Removing candidate at position #", getTrackIndex(itTrack));
      itTrack = mTracks.erase(itTrack);
    } else {
      print("Candidate at position #", getTrackIndex(itTrack), " is improved");
      ++itTrack;
    }
  }
}

//_________________________________________________________________________________________________
void TrackFinderOriginal::finalize()
{
  /// Copy the smoothed parameters and covariances into the regular ones

  print("Finalize tracks");

  // The smoother must be enabled to compute the final parameters at each cluster
  if (!mTrackFitter.isSmootherEnabled()) {
    LOG(ERROR) << "Smoother disabled --> tracks cannot be finalized";
    return;
  }

  for (auto& track : mTracks) {
    for (auto& param : track) {
      param.setParameters(param.getSmoothParameters());
      param.setCovariances(param.getSmoothCovariances());
    }
  }
}

//_________________________________________________________________________________________________
uint8_t TrackFinderOriginal::requestedStationMask() const
{
  /// Get the mask of the requested station, i.e. an integer where
  /// bit n is set to one if the station n was requested

  uint8_t mask(0);

  for (int i = 0; i < 5; ++i) {
    if (SRequestStation[i]) {
      mask |= (1 << i);
    }
  }

  return mask;
}

//_________________________________________________________________________________________________
int TrackFinderOriginal::getTrackIndex(const std::list<Track>::iterator& itCurrentTrack) const
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
void TrackFinderOriginal::printTracks() const
{
  /// print all the tracks currently in the list if the debug level is > 1
  if (mDebugLevel > 1) {
    for (const auto& track : mTracks) {
      track.print();
    }
  }
}

//_________________________________________________________________________________________________
void TrackFinderOriginal::printTrackParam(const TrackParam& trackParam) const
{
  /// print the track parameters if the debug level is > 1
  if (mDebugLevel > 1) {
    trackParam.print();
  }
}

//_________________________________________________________________________________________________
template <class... Args>
void TrackFinderOriginal::print(Args... args) const
{
  /// print a debug message if the debug level is > 0
  if (mDebugLevel > 0) {
    (cout << ... << args) << "\n";
  }
}

} // namespace mch
} // namespace o2
