// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "FairLogger.h"
#include "MIDBase/Constants.h"

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
bool Tracker::init()
{
  /// Initializes the task

  // Sets the proper size
  for (auto& clIdx : mClusterIndexes) {
    clIdx.reserve(20);
  }

  // Prepare storage of clusters
  mClusters.reserve(100);

  // Prepare storage of tracks
  mTracks.reserve(30);

  return true;
}

//______________________________________________________________________________
void Tracker::reset()
{
  /// Resets clusters and the number of tracks

  for (auto& clIdx : mClusterIndexes) {
    clIdx.clear();
  }

  mClusters.clear();
  mTracks.clear();
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
bool Tracker::loadClusters(gsl::span<const Cluster2D>& clusters)
{
  /// Fills the array of clusters per detection element

  LOG(DEBUG) << "Loading clusters:";
  for (auto& currData : clusters) {
    int deId = currData.deId;
    // This needs to be done before adding the element to mClusters
    mClusterIndexes[deId].emplace_back(mClusters.size());
    const auto& position = mTransformer.localToGlobal(deId, currData.xCoor, currData.yCoor);
    mClusters.push_back({ currData.deId,
                          position.x(), position.y(), position.z(),
                          currData.sigmaX2, currData.sigmaY2 });

    LOG(DEBUG) << "deId " << deId << " pos: (" << currData.xCoor << ", " << currData.yCoor << ") err2: ("
               << currData.sigmaX2 << ", " << currData.sigmaY2 << ") => (" << mClusters.back().xCoor << "," << mClusters.back().yCoor
               << "," << mClusters.back().zCoor << ")";
  }

  return (clusters.size() > 0);
}

//______________________________________________________________________________
bool Tracker::process(gsl::span<const Cluster2D> clusters)
{
  /// Main function: runs on a data containing the clusters
  /// and builds the tracks

  // Reset cluster and tracks information
  reset();

  // Load the digits to get the fired pads
  if (loadClusters(clusters)) {
    // Right and left side can be processed in parallel
    // Right inward
    processSide(true, true);
    // Right outward
    processSide(true, false);
    // Left inward
    processSide(false, true);
    // left outward
    processSide(false, false);
  }

  return true;
}

//______________________________________________________________________________
bool Tracker::processSide(bool isRight, bool isInward)
{
  /// Make tracks on one side of the detector
  int firstCh = (isInward) ? 3 : 0;
  int secondCh = (isInward) ? 2 : 1;
  int rpcOffset1 = Constants::getDEId(isRight, firstCh, 0);
  int rpcOffset2 = Constants::getDEId(isRight, secondCh, 0);

  // loop on RPCs in first plane
  Track track;
  track.setNDF(2);
  for (int irpc = 0; irpc < 9; ++irpc) {
    int deId1 = rpcOffset1 + irpc;
    for (size_t icl1 = 0; icl1 < mClusterIndexes[deId1].size(); ++icl1) {
      // loop on clusters of the RPC in the first plane
      auto& cl1 = mClusters[mClusterIndexes[deId1][icl1]];
      int firstRpc = getFirstNeighbourRPC(irpc);
      int lastRpc = getLastNeighbourRPC(irpc);
      for (int irpc2 = firstRpc; irpc2 <= lastRpc; ++irpc2) {
        // loop on (neighbour) RPCs in second plane
        int deId2 = rpcOffset2 + irpc2;
        for (size_t icl2 = 0; icl2 < mClusterIndexes[deId2].size(); ++icl2) {
          // loop on clusters of the RPC in the second plane
          auto& cl2 = mClusters[mClusterIndexes[deId2][icl2]];

          if (!makeTrackSeed(track, cl1, cl2)) {
            continue;
          }

          track.setClusterMatched(firstCh, mClusterIndexes[deId1][icl1]);
          track.setClusterMatched(secondCh, mClusterIndexes[deId2][icl2]);
          track.setClusterMatched(3 - firstCh, -1);
          track.setClusterMatched(3 - secondCh, -1);
          LOG(DEBUG) << "Seed: ";
          LOG(DEBUG) << "Track seed: " << deId1 << " - " << deId2 << "  Position: (" << track.getPositionX() << ", " << track.getPositionY() << ", " << track.getPositionZ() << ")";
          // LOG(DEBUG) << "Covariance: " << track.getCovarianceParameters();
          followTrack(track, isRight, isInward);
        } // loop on clusters in second plane
      }   // loop on RPCs in second plane
    }     // loop on clusters in first plane
  }       // loop on RPCs in first plane
  return true;
}

//______________________________________________________________________________
bool Tracker::makeTrackSeed(Track& track, const Cluster3D& cl1, const Cluster3D& cl2) const
{
  /// Make a track seed from two clusters

  // First check if the delta_x between the two clusters is not too large
  double dZ = cl2.zCoor - cl1.zCoor;
  double dZ2 = dZ * dZ;
  double nonBendingSlope = (cl2.xCoor - cl1.xCoor) / dZ;
  double nonBendingImpactParam = std::abs(cl2.xCoor - cl2.zCoor * nonBendingSlope);
  double nonBendingImpactParamErr = std::sqrt(
    (cl1.zCoor * cl1.zCoor * cl2.sigmaX2 + cl2.zCoor * cl2.zCoor * cl1.sigmaX2) / dZ2);
  if ((nonBendingImpactParam - mSigmaCut * nonBendingImpactParamErr) > mImpactParamCut) {
    LOG(DEBUG) << "NB slope: " << nonBendingSlope << " NB impact param: " << nonBendingImpactParam << " - " << mSigmaCut
               << " * " << nonBendingImpactParamErr << " > " << mImpactParamCut;
    return false;
  }

  // Then start making the track (from 2 points)
  track.setPosition(cl2.xCoor, cl2.yCoor, cl2.zCoor);
  track.setDirection(nonBendingSlope, (cl2.yCoor - cl1.yCoor) / dZ, 1.);
  track.setCovarianceParameters(cl2.sigmaX2,                       // x-x
                                cl2.sigmaY2,                       // y-y
                                (cl1.sigmaX2 + cl2.sigmaX2) / dZ2, // slopeX-slopeX
                                (cl1.sigmaY2 + cl2.sigmaY2) / dZ2, // slopeY-slopeY
                                cl2.sigmaX2 / dZ,                  // x-slopeX
                                cl2.sigmaY2 / dZ);                 // y-slopeY

  return true;
}

//______________________________________________________________________________
bool Tracker::followTrack(const Track& track, bool isRight, bool isInward)
{
  /// Follows the track segment in the other station
  std::array<int, 2> chamberOrder;
  chamberOrder[0] = isInward ? 1 : 2;
  chamberOrder[1] = isInward ? 0 : 3;

  Track bestTrack;
  bestTrack.setChi2(2. * mSigmaCut * mSigmaCut);
  bestTrack.setNDF(2);

  // loop on next two chambers
  for (int ich = 0; ich < 2; ++ich) {
    findNextCluster(track, isRight, isInward, chamberOrder[ich], 0, 8, bestTrack);
    if (bestTrack.getNDF() == 4) {
      // We already have a track with 4 clusters: no need to search for a track with only one cluster
      // in the next chamber
      break;
    }
  }

  if (bestTrack.getNDF() > 2) {
    // Extrapolate to first cluster in MT11 and compute the chi2
    finalizeTrack(bestTrack);

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
  int rpcOffset = Constants::getDEId(isRight, chamber, 0);
  Track newTrack;
  for (int irpc = firstRPC; irpc <= lastRPC; ++irpc) {
    int deId = rpcOffset + irpc;
    for (size_t icl = 0; icl < mClusterIndexes[deId].size(); ++icl) {
      auto& cl = mClusters[mClusterIndexes[deId][icl]];
      double addChi2AtCluster = tryOneCluster(track, cl, newTrack);
      newTrack.setChi2(track.getChi2() + addChi2AtCluster);
      if (newTrack.getChi2() > bestTrack.getChi2()) {
        continue;
      }
      newTrack.setClusterMatched(Constants::getChamber(cl.deId), mClusterIndexes[deId][icl]);
      newTrack.setNDF(track.getNDF() + 1);
      LOG(DEBUG) << "Attach cluster number " << newTrack.getNDF() << ": DeId " << deId << "  cluster " << icl;
      if (nextChamber >= 0 && nextChamber <= 3) {
        // We found a cluster in the first chamber of the station
        // We search for a cluster in the last chamber, this time limiting to the RPC above and below this one
        findNextCluster(newTrack, isRight, isInward, nextChamber, getFirstNeighbourRPC(irpc), getLastNeighbourRPC(irpc), bestTrack);
      }
      if (newTrack.getNDF() >= bestTrack.getNDF() && newTrack.getChi2OverNDF() < bestTrack.getChi2OverNDF()) {
        // Prefer tracks with a larger number of attached clusters, even if the chi2 is worse
        bestTrack = newTrack;
        LOG(DEBUG) << "Selected track with clusters: " << newTrack.getClusterMatched(0) << " " << newTrack.getClusterMatched(1) << " " << newTrack.getClusterMatched(2) << " " << newTrack.getClusterMatched(3);
      }
    } // loop on clusters
  }   // loop on RPC

  return (bestTrack.getNDF() > 2);
}

//______________________________________________________________________________
double Tracker::tryOneCluster(const Track& track, const Cluster3D& cluster, Track& newTrack) const
{
  /// Tests the compatibility between the track and the cluster
  /// given the track and cluster resolutions + the maximum-distance-to-track value
  /// If the cluster is compatible, it propagates a copy of the track to the z of the cluster,
  /// runs the kalman filter and returns the additional chi2.
  /// It returns twice the maximum allowd chi2 otherwise
  double dZ = cluster.zCoor - track.getPositionZ();
  double dZ2 = dZ * dZ;
  double cpos[2] = { cluster.xCoor, cluster.yCoor };
  double cerr2[2] = { cluster.sigmaX2, cluster.sigmaY2 };
  double pos[2] = { track.getPositionX(), track.getPositionY() };
  double newPos[2] = { 0., 0. };
  double dist[2] = { 0., 0. };
  double dir[2] = { track.getDirectionX(), track.getDirectionY() };
  const std::array<float, 6> covParams = track.getCovarianceParameters();
  for (int icoor = 0; icoor < 2; ++icoor) {
    newPos[icoor] = pos[icoor] + dir[icoor] * dZ;
    dist[icoor] = cpos[icoor] - newPos[icoor];
    double err2 = covParams[icoor] + dZ2 * covParams[icoor + 2] + 2. * dZ * covParams[icoor + 4] + cerr2[icoor];
    double distMax = mSigmaCut * std::sqrt(2. * err2) + 4.;
    if (std::abs(dist[icoor]) > distMax) {
      LOG(DEBUG) << "Reject cluster: coordinate " << icoor << "  cl " << cpos[icoor] << " tr " << newPos[icoor] << " err "
                 << std::sqrt(err2);
      return 2. * mMaxChi2;
    }
  }

  newTrack = track;
  newTrack.propagateToZ(cluster.zCoor);

  return runKalmanFilter(newTrack, cluster);
}

//__________________________________________________________________________
double Tracker::runKalmanFilter(Track& track, const Cluster3D& cluster) const
{
  /// Computes new track parameters and their covariances including new cluster using kalman filter.
  /// Returns the additional track chi2

  double pos[2] = { track.getPositionX(), track.getPositionY() };
  double dir[2] = { track.getDirectionX(), track.getDirectionY() };
  double clusPos[2] = { cluster.xCoor, cluster.yCoor };

  std::array<float, 6> newCovParams;
  const std::array<float, 6> covParams = track.getCovarianceParameters();
  double newPos[2], newDir[2];
  double clusterSigma[2] = { cluster.sigmaX2, cluster.sigmaY2 };
  double chi2 = 0.;
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
    // slopeX -> slopeX + ( cl_x - x ) * s_slopeX^2 / (s_x^2 + cl_s_x^2)
    newDir[idx] = dir[idx] + diff * covParams[covIdx] / den;

    chi2 += diff * diff / den;
  }

  // Save the new parameters
  track.setPosition(newPos[0], newPos[1], cluster.zCoor);
  track.setDirection(newDir[0], newDir[1], 1.);

  // Save the new parameters covariance matrix
  track.setCovarianceParameters(newCovParams);

  return chi2;
}

//______________________________________________________________________________
void Tracker::finalizeTrack(Track& track)
{
  /// Computes the chi2 of the track
  /// and extrapolate it to the first cluster
  int ndf = 0;
  double chi2 = 0.;
  for (int ich = 3; ich >= 0; --ich) {
    int matchedClusterIdx = track.getClusterMatched(ich);
    if (matchedClusterIdx < 0) {
      continue;
    }
    ++ndf;
    Cluster3D& cl(mClusters[matchedClusterIdx]);
    track.propagateToZ(cl.zCoor);
    double clPos[2] = { cl.xCoor, cl.yCoor };
    double clErr2[2] = { cl.sigmaX2, cl.sigmaY2 };
    double trackPos[2] = { track.getPositionX(), track.getPositionY() };
    double trackCov[2] = { track.getCovarianceParameter(Track::CovarianceParamIndex::VarX),
                           track.getCovarianceParameter(Track::CovarianceParamIndex::VarY) };
    for (int icoor = 0; icoor < 2; ++icoor) {
      double diff = trackPos[icoor] - clPos[icoor];
      chi2 += diff * diff / (trackCov[icoor] + clErr2[icoor]);
    }
  }
  track.setChi2(chi2);
  track.setNDF(ndf);
  LOG(DEBUG) << "Finalize track: " << track;
}

//______________________________________________________________________________
bool Tracker::tryAddTrack(const Track& track)
{
  /// Checks if the track is duplicated.
  /// If it is identical to another track (same clusters), reject it.
  /// If track parameters are compatible, selects the track with the
  /// smallest chi2
  /// Otherwise add the track to the list

  float chi2OverNDF = track.getChi2OverNDF();
  // We divide the chi2 by two since we want to consider only the uncertainty
  // on one of the two tracks. We further reduce to 0.4 since we want to account
  // for the case where one of the two reconstructed tracks has a much better precision
  // of the other
  float chi2Cut = 0.4 * mSigmaCut * mSigmaCut;
  for (auto& checkTrack : mTracks) {
    int nCommonClusters = 0;
    for (int ich = 0; ich < 4; ++ich) {
      if (track.getClusterMatched(ich) == checkTrack.getClusterMatched(ich)) {
        ++nCommonClusters;
      }
    }
    if (nCommonClusters == 4) {
      return false;
    }
    if (track.isCompatible(checkTrack, chi2Cut)) {
      // The new track is compatible with an existing one
      if (chi2OverNDF < checkTrack.getChi2OverNDF()) {
        // The new track is more precise than the old one: replace it!
        LOG(DEBUG) << "Replacing track " << checkTrack << "    with " << track;
        checkTrack = track;
      } else {
        LOG(DEBUG) << "Rejecting track " << track << "     compatible with " << checkTrack;
        // The new track is less precise than the old one: reject it!
      }
      return false;
    }
  }

  // The new track is not compatible with the previous ones: keep it
  mTracks.emplace_back(track);
  return true;
}

} // namespace mid
} // namespace o2
