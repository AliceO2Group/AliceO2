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

/// \file   MID/Tracking/src/Tracker.cxx
/// \brief  Implementation of the tracker algorithm for the MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   09 May 2017
#include "MIDTracking/Tracker.h"

#include <cmath>
#include <functional>
#include <stdexcept>

#include "Framework/Logger.h"
#include "MIDBase/DetectorParameters.h"
#include "MIDTracking/TrackerParam.h"

namespace o2
{
namespace mid
{

//______________________________________________________________________________
Tracker::Tracker(const GeometryTransformer& geoTrans) : mTransformer(geoTrans)
{
  /// Constructor
}

//______________________________________________________________________________
bool Tracker::init(bool keepAll)
{
  /// Initializes the tracker

  if (keepAll) {
    mFollowTrack = &Tracker::followTrackKeepAll;
  } else {
    mFollowTrack = &Tracker::followTrackKeepBest;
  }

  mImpactParamCut = TrackerParam::Instance().impactParamCut;
  mSigmaCut = TrackerParam::Instance().sigmaCut;
  mMaxChi2 = 2. * mSigmaCut * mSigmaCut;

  return true;
}

//______________________________________________________________________________
int Tracker::getFirstNeighbourRPC(int rpc) const
{
  /// Gets first neighbour RPC
  return (rpc == 0) ? rpc : rpc - 1;
}

//______________________________________________________________________________
int Tracker::getLastNeighbourRPC(int rpc) const
{
  /// Gets first neighbour RPC
  return (rpc == 8) ? rpc : rpc + 1;
}

//______________________________________________________________________________
bool Tracker::loadClusters(gsl::span<const Cluster>& clusters)
{
  /// Fills the array of clusters per detection element

  for (auto& cl : clusters) {
    int deId = cl.deId;
    // This needs to be done before adding the element to mClusters
    mClusterIndexes[deId].emplace_back(mClusters.size());
    const auto& position = mTransformer.localToGlobal(deId, cl.xCoor, cl.yCoor);
    mClusters.emplace_back(cl);
    mClusters.back().xCoor = position.x();
    mClusters.back().yCoor = position.y();
    mClusters.back().zCoor = position.z();
  }

  return (clusters.size() > 0);
}

//______________________________________________________________________________
void Tracker::process(gsl::span<const Cluster> clusters, gsl::span<const ROFRecord> rofRecords)
{
  /// Main function: runs on a data containing the clusters in timeframe
  /// and builds the tracks
  mClusters.clear();
  mTracks.clear();
  mTrackROFRecords.clear();
  mClusterROFRecords.clear();
  for (auto& rofRecord : rofRecords) {
    auto firstTrackEntry = mTracks.size();
    auto firstClusterEntry = mClusters.size();
    process(clusters.subspan(rofRecord.firstEntry, rofRecord.nEntries), true);
    auto nTrackEntries = mTracks.size() - firstTrackEntry;
    mTrackROFRecords.emplace_back(rofRecord, firstTrackEntry, nTrackEntries);
    auto nClusterEntries = mClusters.size() - firstClusterEntry;
    mClusterROFRecords.emplace_back(rofRecord, firstClusterEntry, nClusterEntries);
  }
}

//______________________________________________________________________________
void Tracker::process(gsl::span<const Cluster> clusters, bool accumulate)
{
  /// Main function: runs on a data containing the clusters per event
  /// and builds the tracks

  // Reset cluster and tracks information
  for (auto& clIdx : mClusterIndexes) {
    clIdx.clear();
  }

  if (!accumulate) {
    mClusters.clear();
    mTracks.clear();
  }

  // Load the digits to get the fired pads
  if (loadClusters(clusters)) {
    mFirstTrackOffset = mTracks.size();
    try {
      // Right and left side can be processed in parallel
      // Right inward
      mTrackOffset = mTracks.size();
      mNTracksStep1 = 0;
      processSide(true, true);
      mNTracksStep1 = mTracks.size() - mTrackOffset;
      // Right outward
      processSide(true, false);
      // Left inward
      mTrackOffset = mTracks.size();
      mNTracksStep1 = 0;
      processSide(false, true);
      mNTracksStep1 = mTracks.size() - mTrackOffset;
      // left outward
      processSide(false, false);
    } catch (std::exception const& e) {
      LOG(error) << e.what() << " --> abort";
      mTracks.erase(mTracks.begin() + mFirstTrackOffset, mTracks.end());
    }
  }
}

//______________________________________________________________________________
bool Tracker::processSide(bool isRight, bool isInward)
{
  /// Make tracks on one side of the detector
  int firstCh = (isInward) ? 3 : 0;
  int secondCh = (isInward) ? 2 : 1;
  int rpcOffset1 = detparams::getDEId(isRight, firstCh, 0);
  int rpcOffset2 = detparams::getDEId(isRight, secondCh, 0);

  // loop on RPCs in first plane
  Track track;
  for (int irpc = 0; irpc < 9; ++irpc) {
    int deId1 = rpcOffset1 + irpc;
    for (auto clIdx1 : mClusterIndexes[deId1]) {
      // loop on clusters of the RPC in the first plane
      auto& cl1 = mClusters[clIdx1];
      int firstRpc = getFirstNeighbourRPC(irpc);
      int lastRpc = getLastNeighbourRPC(irpc);
      for (int irpc2 = firstRpc; irpc2 <= lastRpc; ++irpc2) {
        // loop on (neighbour) RPCs in second plane
        int deId2 = rpcOffset2 + irpc2;
        for (auto clIdx2 : mClusterIndexes[deId2]) {
          // loop on clusters of the RPC in the second plane
          auto& cl2 = mClusters[clIdx2];

          if (!makeTrackSeed(track, cl1, cl2)) {
            continue;
          }

          track.setClusterMatchedUnchecked(firstCh, clIdx1);
          track.setClusterMatchedUnchecked(secondCh, clIdx2);
          track.setClusterMatchedUnchecked(3 - firstCh, -1);
          track.setClusterMatchedUnchecked(3 - secondCh, -1);
          std::invoke(mFollowTrack, this, track, isRight, isInward);
        } // loop on clusters in second plane
      }   // loop on RPCs in second plane
    }     // loop on clusters in first plane
  }       // loop on RPCs in first plane
  return true;
}

//______________________________________________________________________________
bool Tracker::makeTrackSeed(Track& track, const Cluster& cl1, const Cluster& cl2) const
{
  /// Make a track seed from two clusters

  // First check if the delta_x between the two clusters is not too large
  double dZ = cl2.zCoor - cl1.zCoor;
  double dZ2 = dZ * dZ;
  double nonBendingSlope = (cl2.xCoor - cl1.xCoor) / dZ;
  double nonBendingImpactParam = std::abs(cl2.xCoor - cl2.zCoor * nonBendingSlope);
  double nonBendingImpactParamErr = std::sqrt(
    (cl1.zCoor * cl1.zCoor * cl2.getEX2() + cl2.zCoor * cl2.zCoor * cl1.getEX2()) / dZ2);
  if ((nonBendingImpactParam - mSigmaCut * nonBendingImpactParamErr) > mImpactParamCut) {
    return false;
  }

  // Then start making the track (from 2 points)
  track.setPosition(cl2.xCoor, cl2.yCoor, cl2.zCoor);
  track.setDirection(nonBendingSlope, (cl2.yCoor - cl1.yCoor) / dZ, 1.);
  track.setCovarianceParameters(cl2.getEX2(),                        // x-x
                                cl2.getEY2(),                        // y-y
                                (cl1.getEX2() + cl2.getEX2()) / dZ2, // slopeX-slopeX
                                (cl1.getEY2() + cl2.getEY2()) / dZ2, // slopeY-slopeY
                                cl2.getEX2() / dZ,                   // x-slopeX
                                cl2.getEY2() / dZ);                  // y-slopeY
  track.setChi2(0.);
  track.setNDF(0);

  return true;
}

//______________________________________________________________________________
bool Tracker::followTrackKeepAll(const Track& track, bool isRight, bool isInward)
{
  /// Follows the track segment in the other station and adds all possible matches
  /// It assumes that the inward tracking is performed before the outward tracking
  ///
  /// This algorithm keeps all of the tracks that can be obtained by pairing
  /// all of the clusters in the two stations (if they pass the chi2 cut).
  /// Tests show that this method allows to correctly reconstruct all real tracks
  /// even when the track multiplicity is large.
  /// However, the method is of course more time consuming and also reconstruct
  /// a larger number of fake tracks.

  std::unordered_set<int> excludedClusters;

  if (isInward) {

    // look for clusters in both chambers or in first chamber only
    findAllClusters(track, isRight, 1, 0, 8, 0, excludedClusters, true);

    // look for clusters in the second chamber only
    findAllClusters(track, isRight, 0, 0, 8, -1, excludedClusters, true);

  } else {

    // exclude clusters from tracks already found by inward tracking (with 4/4 clusters)
    excludeUsedClusters(track, 1, 0, excludedClusters);

    // look for clusters in first chamber only
    findAllClusters(track, isRight, 2, 0, 8, -1, excludedClusters, true);

    // look for clusters in the second chamber only
    findAllClusters(track, isRight, 3, 0, 8, -1, excludedClusters, true);
  }

  return true;
}

//______________________________________________________________________________
bool Tracker::findAllClusters(const Track& track, bool isRight, int chamber, int firstRPC, int lastRPC, int nextChamber,
                              std::unordered_set<int>& excludedClusters, bool excludeClusters)
{
  /// Find all compatible clusters in these RPCs and attach them to a copy of the track
  /// For each of them look for further compatible clusters in the next chamber, if any
  /// Either exclude the excludedClusters from the search or add the new clusters found in the list
  /// Throw an exception if the maximum number of tracks is exceeded

  int rpcOffset = detparams::getDEId(isRight, chamber, 0);
  bool clusterFound = false;
  Track newTrack;

  for (int irpc = firstRPC; irpc <= lastRPC; ++irpc) {
    int deId = rpcOffset + irpc;
    for (auto clIdx : mClusterIndexes[deId]) {

      // skip excluded clusters
      if (excludeClusters && excludedClusters.count(clIdx) > 0) {
        continue;
      }

      // try to attach this cluster
      if (!tryOneCluster(track, chamber, clIdx, newTrack)) {
        continue;
      }
      clusterFound = true;

      // add this cluster to the list to be excluded later on
      if (!excludeClusters) {
        excludedClusters.emplace(clIdx);
      }

      // store this track extrapolated to MT11 unless compatible clusters are found in the next chamber (if any)
      if (nextChamber < 0 || !findAllClusters(newTrack, isRight, nextChamber, getFirstNeighbourRPC(irpc),
                                              getLastNeighbourRPC(irpc), -1, excludedClusters, false)) {
        if (mTracks.size() - mFirstTrackOffset >= TrackerParam::Instance().maxCandidates) {
          throw std::length_error(std::string("Too many track candidates (") +
                                  (mTracks.size() - mFirstTrackOffset) + ")");
        }
        newTrack.propagateToZ(SMT11Z);
        mTracks.emplace_back(newTrack);
      }
    }
  }

  return clusterFound;
}

//______________________________________________________________________________
bool Tracker::followTrackKeepBest(const Track& track, bool isRight, bool isInward)
{
  /// Follows the track segment in the other station
  /// Adds only the best track

  // This algorithm is works generally well,
  // but tests show that it can fail reconstructing the generated track
  // in case of two tracks close in space.
  // The main issue is that we do not have charge distribution information, and the strips are large,
  // so it can happen that the track obtained by combining the hits of the first track in MT1
  // and of the second track in MT2 can sometimes have a better chi2 of the real tracks.
  // Since this algorithm only keeps the best track, it might chose the fake instead of the real track.
  // The algorithm is however fast, so it should be preferred in case of low tracks multiplicity

  std::array<int, 2> chamberOrder;
  chamberOrder[0] = isInward ? 1 : 2;
  chamberOrder[1] = isInward ? 0 : 3;

  Track bestTrack;

  // loop on next two chambers
  for (int ich = 0; ich < 2; ++ich) {
    findNextCluster(track, isRight, isInward, chamberOrder[ich], 0, 8, bestTrack);
    if (bestTrack.getNDF() == 4) {
      // We already have a track with 4 clusters: no need to search for a track with only one cluster
      // in the next chamber
      break;
    }
  }

  if (bestTrack.getNDF() > 0) {
    // Extrapolate to MT11
    bestTrack.propagateToZ(SMT11Z);

    // Add the track if it is not compatible or better than the ones we already have
    if (tryAddTrack(bestTrack)) {
      return true;
    }
  }

  return false;
}

//______________________________________________________________________________
bool Tracker::findNextCluster(const Track& track, bool isRight, bool isInward, int chamber, int firstRPC, int lastRPC,
                              Track& bestTrack) const
{
  /// Find next best cluster
  int nextChamber = (isInward) ? chamber - 1 : chamber + 1;
  int rpcOffset = detparams::getDEId(isRight, chamber, 0);
  Track newTrack;
  for (int irpc = firstRPC; irpc <= lastRPC; ++irpc) {
    int deId = rpcOffset + irpc;
    for (auto clIdx : mClusterIndexes[deId]) {
      if (!tryOneCluster(track, chamber, clIdx, newTrack)) {
        continue;
      }
      if (nextChamber >= 0 && nextChamber <= 3) {
        // We found a cluster in the first chamber of the station
        // We search for a cluster in the last chamber, this time limiting to the RPC above and below this one
        findNextCluster(newTrack, isRight, isInward, nextChamber,
                        getFirstNeighbourRPC(irpc), getLastNeighbourRPC(irpc), bestTrack);
      }
      if (newTrack.getNDF() > bestTrack.getNDF() ||
          (newTrack.getNDF() == bestTrack.getNDF() && newTrack.getChi2() < bestTrack.getChi2())) {
        // Prefer tracks with a larger number of attached clusters, even if the chi2 is worse
        bestTrack = newTrack;
      }
    } // loop on clusters
  }   // loop on RPC

  return (bestTrack.getNDF() > 0);
}

//______________________________________________________________________________
bool Tracker::tryOneCluster(const Track& track, int chamber, int clIdx, Track& newTrack) const
{
  /// Tests the compatibility between the track and the cluster given the track and cluster resolutions
  /// If the cluster is compatible, it propagates a copy of the track to the z of the cluster and runs the kalman filter

  newTrack = track;
  auto& cl = mClusters[clIdx];
  newTrack.propagateToZ(cl.zCoor);

  double diff[2] = {cl.xCoor - newTrack.getPositionX(), cl.yCoor - newTrack.getPositionY()};
  double err2[2] = {newTrack.getCovarianceParameter(Track::CovarianceParamIndex::VarX) + cl.getEX2(),
                    newTrack.getCovarianceParameter(Track::CovarianceParamIndex::VarY) + cl.getEY2()};
  double chi2 = diff[0] * diff[0] / err2[0] + diff[1] * diff[1] / err2[1];
  if (chi2 > mMaxChi2) {
    return false;
  }

  runKalmanFilter(newTrack, cl);
  newTrack.setChi2(track.getChi2() + chi2);
  newTrack.setNDF(track.getNDF() + 2);
  newTrack.setClusterMatchedUnchecked(chamber, clIdx);

  return true;
}

//__________________________________________________________________________
void Tracker::runKalmanFilter(Track& track, const Cluster& cluster) const
{
  /// Computes new track parameters and their covariances including new cluster using kalman filter.
  /// Returns the additional track chi2

  double pos[2] = {track.getPositionX(), track.getPositionY()};
  double dir[2] = {track.getDirectionX(), track.getDirectionY()};
  const std::array<float, 6>& covParams = track.getCovarianceParameters();
  double clusPos[2] = {cluster.xCoor, cluster.yCoor};
  double clusterSigma[2] = {cluster.getEX2(), cluster.getEY2()};

  std::array<float, 6> newCovParams;
  double newPos[2], newDir[2];
  for (int idx = 0; idx < 2; ++idx) {
    int slopeIdx = idx + 2;
    int covIdx = idx + 4;

    double den = clusterSigma[idx] + covParams[idx];
    assert(den != 0.);
    // if ( den == 0. ) return 2.*mMaxChi2;

    // Compute the new covariance
    // cov -> (W+U)^-1
    // where W = (old_cov)^-1
    // and U = inverse of uncertainties of cluster
    // s_x^2 -> s_x^2 * cl_s_x^2 / (s_x^2 + cl_s_x^2)
    newCovParams[idx] = covParams[idx] * clusterSigma[idx] / den;
    // cov(x,slopeX) -> cov(x,slopeX) * cl_s_x^2 / (s_x^2 + cl_s_x^2)
    newCovParams[covIdx] = covParams[covIdx] * clusterSigma[idx] / den;
    // s_slopeX^2 -> s_slopeX^2 - cov(x,slopeX)^2 / (s_x^2 + cl_s_x^2)
    newCovParams[slopeIdx] = covParams[slopeIdx] - covParams[covIdx] * covParams[covIdx] / den;

    double diff = clusPos[idx] - pos[idx];

    // New parameters: p' = ((W+U)^-1)U(m-p) + p
    // x -> x + ( cl_x - x ) * s_x^2 / (s_x^2 + cl_s_x^2)
    newPos[idx] = pos[idx] + diff * covParams[idx] / den;
    // slopeX -> slopeX + ( cl_x - x ) * cov(x,slopeX) / (s_x^2 + cl_s_x^2)
    newDir[idx] = dir[idx] + diff * covParams[covIdx] / den;
  }

  // Save the new parameters
  track.setPosition(newPos[0], newPos[1], cluster.zCoor);
  track.setDirection(newDir[0], newDir[1], 1.);

  // Save the new parameters covariance matrix
  track.setCovarianceParameters(newCovParams);
}

//______________________________________________________________________________
bool Tracker::tryAddTrack(const Track& track)
{
  /// Checks if the track is duplicated.
  /// If it is identical to another track (same clusters), reject it.
  /// If track parameters are compatible, selects the track with the
  /// smallest chi2
  /// Otherwise add the track to the list

  // We divide the chi2 by two since we want to consider only the uncertainty
  // on one of the two tracks. We further reduce to 0.4 since we want to account
  // for the case where one of the two reconstructed tracks has a much better precision
  // of the other
  float chi2Cut = 0.4 * mSigmaCut * mSigmaCut;
  for (auto checkTrack = mTracks.begin() + mTrackOffset; checkTrack != mTracks.end(); ++checkTrack) {
    int nCommonClusters = 0;
    for (int ich = 0; ich < 4; ++ich) {
      if (track.getClusterMatchedUnchecked(ich) == checkTrack->getClusterMatchedUnchecked(ich)) {
        ++nCommonClusters;
      }
    }
    if (nCommonClusters == 4) {
      return false;
    }
    if (nCommonClusters == 3 && track.isCompatible(*checkTrack, chi2Cut)) {
      // The new track is compatible with an existing one
      if (track.getNDF() > checkTrack->getNDF() ||
          (track.getNDF() == checkTrack->getNDF() && track.getChi2() < checkTrack->getChi2())) {
        // The new track has more cluster or is more precise than the old one: replace it!
        *checkTrack = track;
      }
      return false;
    }
  }

  // The new track is not compatible with the previous ones: keep it
  mTracks.emplace_back(track);
  return true;
}

//______________________________________________________________________________
void Tracker::excludeUsedClusters(const Track& track, int ch1, int ch2, std::unordered_set<int>& excludedClusters)
{
  /// Find tracks that contain the same clusters as this track in chambers ch1 and ch2,
  /// and add the clusters that these tracks have on the other chambers in the excludedClusters list

  int nTracks = mTrackOffset + mNTracksStep1;
  for (int i = mTrackOffset; i < nTracks; ++i) {
    const auto& tr = mTracks[i];
    if (track.getClusterMatchedUnchecked(ch1) == tr.getClusterMatchedUnchecked(ch1) &&
        track.getClusterMatchedUnchecked(ch2) == tr.getClusterMatchedUnchecked(ch2)) {
      excludedClusters.emplace(tr.getClusterMatchedUnchecked(3 - ch1));
      excludedClusters.emplace(tr.getClusterMatchedUnchecked(3 - ch2));
    }
  }
}

} // namespace mid
} // namespace o2
