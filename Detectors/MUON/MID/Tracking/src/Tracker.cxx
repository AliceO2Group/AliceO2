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
#include "Tracker.h"

#include <cmath>
#include "FairLogger.h"
#include "MIDBase/Constants.h"

namespace o2
{
namespace mid
{

//______________________________________________________________________________
Tracker::Tracker() : mImpactParamCut(210.), mSigmaCut(5.), mMaxChi2(1.e6), mNTracks(0)
{
  /// Default constructor
}

//______________________________________________________________________________
bool Tracker::init()
{
  /// Initializes the task

  // Reset the ordered array of clusters
  for (int deId = 0; deId < 72; ++deId) {
    mClusters[deId].reserve(20);
    mNClusters[deId] = 0;
  }

  // Prepare storage of tracks
  mTracks.reserve(30);

  return true;
}

//______________________________________________________________________________
void Tracker::reset()
{
  /// Resets clusters and the number of tracks
  for (int deId = 0; deId < 72; ++deId) {
    mNClusters[deId] = 0;
  }

  mNTracks = 0;
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
int Tracker::getClusterId(int id, int deId) const
{
  /// Gets first neighbour RPC
  return 1000 * deId + id;
}

//______________________________________________________________________________
bool Tracker::loadClusters(const std::vector<Cluster2D>& clusters)
{
  /// Fills the array of clusters per detection element

  for (auto& currData : clusters) {
    int deId = currData.deId;
    if (mNClusters[deId] >= static_cast<unsigned long int>(mClusters[deId].size())) {
      mClusters[deId].emplace_back(Cluster3D());
    }
    Cluster3D& cl(mClusters[deId][mNClusters[deId]]);
    ++mNClusters[deId];
    cl.deId = currData.deId;
    cl.id = mNClusters[deId];
    cl.position = mTransformer.localToGlobal(deId, currData.xCoor, currData.yCoor);
    cl.sigmaX2 = currData.sigmaX2;
    cl.sigmaY2 = currData.sigmaY2;

    LOG(DEBUG) << "deId " << deId << " pos: (" << currData.xCoor << ", " << currData.yCoor << ") err2: ("
               << currData.sigmaX2 << ", " << currData.sigmaY2 << ") => (" << cl.position.x() << "," << cl.position.y()
               << "," << cl.position.z() << ")";
  }

  return (clusters.size() > 0);
}

//______________________________________________________________________________
bool Tracker::process(const std::vector<Cluster2D>& clusters)
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
  for (int irpc = 0; irpc < 9; ++irpc) {
    int deId1 = rpcOffset1 + irpc;
    for (int icl1 = 0; icl1 < mNClusters[deId1]; ++icl1) {
      // loop on clusters of the RPC in the first plane
      auto& cl1 = mClusters[deId1][icl1];
      int firstRpc = getFirstNeighbourRPC(irpc);
      int lastRpc = getLastNeighbourRPC(irpc);
      for (int irpc2 = firstRpc; irpc2 <= lastRpc; ++irpc2) {
        // loop on (neighbour) RPCs in second plane
        int deId2 = rpcOffset2 + irpc2;
        for (int icl2 = 0; icl2 < mNClusters[deId2]; ++icl2) {
          // loop on clusters of the RPC in the second plane
          auto& cl2 = mClusters[deId2][icl2];

          if (!makeTrackSeed(track, cl1, cl2)) {
            continue;
          }

          track.setClusterMatched(firstCh, getClusterId(cl1.id, deId1));
          track.setClusterMatched(secondCh, getClusterId(cl2.id, deId2));
          track.setClusterMatched(3 - firstCh, 0);
          track.setClusterMatched(3 - secondCh, 0);
          LOG(DEBUG) << deId1 << " - " << deId2;
          LOG(DEBUG) << "Position: " << track.getPosition();
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
  double dZ = cl2.position.z() - cl1.position.z();
  double dZ2 = dZ * dZ;
  double nonBendingSlope = (cl2.position.x() - cl1.position.x()) / dZ;
  double nonBendingImpactParam = std::abs(cl2.position.x() - cl2.position.z() * nonBendingSlope);
  double nonBendingImpactParamErr = std::sqrt(
    (cl1.position.z() * cl1.position.z() * cl2.sigmaX2 + cl2.position.z() * cl2.position.z() * cl1.sigmaX2) / dZ2);
  if ((nonBendingImpactParam - mSigmaCut * nonBendingImpactParamErr) > mImpactParamCut) {
    LOG(DEBUG) << "NB slope: " << nonBendingSlope << " NB impact param: " << nonBendingImpactParam << " - " << mSigmaCut
               << " * " << nonBendingImpactParamErr << " > " << mImpactParamCut;
    return false;
  }

  // Then start making the track (from 2 points)
  track.setPosition(cl2.position.x(), cl2.position.y(), cl2.position.z());
  track.setDirection(nonBendingSlope, (cl2.position.y() - cl1.position.y()) / dZ, 1.);
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
  double bestChi2 = 2. * mSigmaCut * mSigmaCut;
  int nFiredChambers = 0;
  int chamberOrder[2];
  chamberOrder[0] = isInward ? 1 : 2;
  chamberOrder[1] = isInward ? 0 : 3;

  // Add the track to the list
  if (mNTracks >= static_cast<unsigned long int>(mTracks.size())) {
    mTracks.emplace_back(Track());
  }
  Track& bestTrack(mTracks[mNTracks]);

  // loop on next two chambers
  for (int ich = 0; ich < 2; ++ich) {
    findNextCluster(track, isRight, isInward, chamberOrder[ich], 0, 8, nFiredChambers, bestChi2, bestTrack);
    if (nFiredChambers == 2) {
      // We already found all clusters: no need to search for mono-cathodic clusters
      // in the next chamber
      break;
    }
  }

  if (nFiredChambers == 0) {
    // No track found
    return false;
  }

  // Extrapolate to first cluster in MT11 and compute the chi2
  finalizeTrack(bestTrack);

  // Add the track if it is not compatible or better than the ones we already have
  return addTrack(bestTrack);
}

//______________________________________________________________________________
bool Tracker::findNextCluster(const Track& track, bool isRight, bool isInward, int chamber, int firstRPC, int lastRPC,
                              int& nFiredChambers, double& bestChi2, Track& bestTrack, double chi2, int depth) const
{
  /// Find next best cluster
  int nextChamber = (isInward) ? chamber - 1 : chamber + 1;
  int rpcOffset = Constants::getDEId(isRight, chamber, 0);
  Track newTrack;
  for (int irpc = firstRPC; irpc <= lastRPC; ++irpc) {
    int deId = rpcOffset + irpc;
    for (int icl = 0; icl < mNClusters[deId]; ++icl) {
      auto& cl = mClusters[deId][icl];
      double addChi2AtCluster = tryOneCluster(track, cl, newTrack);
      double sumChi2 = chi2 + addChi2AtCluster;
      if (sumChi2 > bestChi2) {
        continue;
      }
      if (nextChamber >= 0 && nextChamber <= 3) {
        findNextCluster(newTrack, isRight, isInward, nextChamber, getFirstNeighbourRPC(irpc), getLastNeighbourRPC(irpc),
                        nFiredChambers, bestChi2, bestTrack, sumChi2, depth + 1);
      }
      if (depth >= nFiredChambers && sumChi2 < bestChi2) {
        nFiredChambers = depth;
        bestChi2 = sumChi2;
        bestTrack = newTrack;
        LOG(DEBUG) << "DeId " << deId << "  cluster " << icl << "  nFiredChambers " << nFiredChambers << "  chi2 "
                   << bestChi2;
      }
    } // loop on clusters
  }   // loop on RPC

  return (nFiredChambers > 0);
}

//______________________________________________________________________________
double Tracker::tryOneCluster(const Track& track, const Cluster3D& cluster, Track& newTrack) const
{
  /// Tests the compatibility between the track and the cluster
  /// given the track and cluster resolutions + the maximum-distance-to-track value
  /// If the cluster is compatible, it propagates a copy of the track to the z of the cluster,
  /// runs the kalman filter and returns the additional chi2.
  /// It returns twice the maximum allowd chi2 otherwise
  double dZ = cluster.position.z() - track.getPosition().z();
  double dZ2 = dZ * dZ;
  double cpos[2] = { cluster.position.x(), cluster.position.y() };
  double cerr2[2] = { cluster.sigmaX2, cluster.sigmaY2 };
  double pos[2] = { track.getPosition().x(), track.getPosition().y() };
  double newPos[2] = { 0., 0. };
  double dist[2] = { 0., 0. };
  double dir[2] = { track.getDirection().x(), track.getDirection().y() };
  const std::array<float, 6> covParams = track.getCovarianceParameters();
  for (int icoor = 0; icoor < 2; ++icoor) {
    newPos[icoor] = pos[icoor] + dir[icoor] * dZ;
    dist[icoor] = cpos[icoor] - newPos[icoor];
    double err2 = covParams[icoor] + dZ2 * covParams[icoor + 2] + 2. * dZ * covParams[icoor + 4] + cerr2[icoor];
    double distMax = mSigmaCut * std::sqrt(2. * err2) + 4.;
    if (std::abs(dist[icoor]) > distMax) {
      LOG(DEBUG) << "Coordinate " << icoor << "  cl " << cpos[icoor] << " tr " << newPos[icoor] << " err "
                 << std::sqrt(err2);
      return 2. * mMaxChi2;
    }
  }

  newTrack = track;
  newTrack.propagateToZ(cluster.position.z());
  newTrack.setClusterMatched(Constants::getChamber(cluster.deId), getClusterId(cluster.id, cluster.deId));

  return runKalmanFilter(newTrack, cluster);
}

//__________________________________________________________________________
double Tracker::runKalmanFilter(Track& track, const Cluster3D& cluster) const
{
  /// Computes new track parameters and their covariances including new cluster using kalman filter.
  /// Returns the additional track chi2

  double pos[2] = { track.getPosition().x(), track.getPosition().y() };
  double dir[2] = { track.getDirection().x(), track.getDirection().y() };
  double clusPos[2] = { cluster.position.x(), cluster.position.y() };

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
  track.setPosition(newPos[0], newPos[1], cluster.position.z());
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
    if (matchedClusterIdx == 0) {
      continue;
    }
    ++ndf;
    int deId = matchedClusterIdx / 1000;
    int icl = matchedClusterIdx % 1000 - 1;
    Cluster3D& cl(mClusters[deId][icl]);
    track.propagateToZ(cl.position.z());
    double clPos[2] = { cl.position.x(), cl.position.y() };
    double clErr2[2] = { cl.sigmaX2, cl.sigmaY2 };
    double trackPos[2] = { track.getPosition().x(), track.getPosition().y() };
    double trackCov[2] = { track.getCovarianceParameter(Track::CovarianceParamIndex::VarX),
                           track.getCovarianceParameter(Track::CovarianceParamIndex::VarY) };
    for (int icoor = 0; icoor < 2; ++icoor) {
      double diff = trackPos[icoor] - clPos[icoor];
      chi2 += diff * diff / (trackCov[icoor] + clErr2[icoor]);
    }
  }
  track.setChi2(chi2);
  track.setNDF(ndf);
  LOG(DEBUG) << track;
}

//______________________________________________________________________________
bool Tracker::addTrack(const Track& track)
{
  /// Adds the track to the list
  /// Does not add identical tracks (i.e. sharing the same clusters)
  /// If track parameters are compatible, selects the track with the
  /// smallest chi2

  float chi2OverNDF = track.getChi2OverNDF();
  // We divide the chi2 by two since we want to consider only the uncertainty
  // on one of the two tracks. We further reduce to 0.4 since we want to account
  // for the case where one of the two reconstructed tracks has a much better precision
  // of the other
  float chi2Cut = 0.4 * mSigmaCut * mSigmaCut;
  for (int itrack = 0; itrack < mNTracks; ++itrack) {
    int nCommonClusters = 0;
    auto& checkTrack = mTracks[itrack];
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
        LOG(DEBUG) << "Replacing track " << checkTrack << "\n with " << track;
        checkTrack = track;
        return true;
      } else {
        LOG(DEBUG) << "Rejecting track " << track << "\n compatible with " << checkTrack;
        // The new track is less precise than the old one: reject it
        return false;
      }
    }
  }

  // The new track is not compatible with the previous ones: add the track to the list
  ++mNTracks;
  return true;
}

} // namespace mid
} // namespace o2
