// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackFitter.cxx
/// \brief Implementation of a class to fit a track to a set of clusters
///
/// \author Philippe Pillot, Subatech

#include "MFTTracking/TrackFitter.h"
#include "MFTTracking/TrackCA.h"
#include "DataFormatsMFT/TrackMFT.h"



#include <stdexcept>

#include <TMatrixD.h>

namespace o2
{
namespace mft
{

using namespace std;
using Track = o2::mft::TrackCA;
using o2::mft::TrackCA;


//_________________________________________________________________________________________________
void TrackFitter::initField(float l3Current)
{
  /// Set the magnetic field for the MFT
}

//_________________________________________________________________________________________________
template <typename T, typename C>
TrackMFTExt TrackFitter::fit(T&& track, C&& clusters)
{
  TrackMFTExt fittedTrack;
  TMatrixD covariances(5,5);
  auto xpos = track.getXCoordinates();
  auto ypos = track.getYCoordinates();
  auto zpos = track.getZCoordinates();
  auto clusterIDs = track.getClustersId();
  auto nClusters = track.getNPoints();
  auto clusOffset = clusters.size();
  static int ntrack = 0;

  std::cout << "** Fitting new track at clusOffset =  " << clusOffset << std::endl;
  // Add clusters to Tracker's cluster vector & set fittedTrack cluster range
  for (auto cls = 0 ; cls < nClusters; cls++) {
    clusters.emplace_back(Cluster(xpos[cls], ypos[cls], zpos[cls], clusterIDs[cls]));
    std::cout << "Adding cluster " << cls << " to track " << ntrack << " with clusterID " << clusterIDs[cls] << " at z = " <<  zpos[cls] << std::endl;
  }
  auto lastTrackCluster = (clusters.end())--; lastTrackCluster--;
  //auto firstTrackCluster = lastTrackCluster;
  //std::advance(firstTrackCluster,-nClusters);

  fittedTrack.setFirstClusterEntry(clusOffset);
  fittedTrack.setNumberOfClusters(nClusters);

  // Initialize track parameters
  initTrack(*lastTrackCluster,covariances);
  std::cout << "Initializing track with first (last) cluster -> clusterId =  " << (*lastTrackCluster).clusterId << " at z = " <<  (*lastTrackCluster).zCoordinate << std::endl;

  // Add remaining clusters
  int nclu = nClusters-1;
  auto currCluster = lastTrackCluster;
  while (nclu > 0 ) {
   currCluster--;
   addCluster(*currCluster, covariances);
   std::cout << nclu << " -> clusterId =  " << (*currCluster).clusterId << " at z = " <<  (*currCluster).zCoordinate << std::endl;
   nclu--;
  }
  // Smooth track
  ntrack++;
  return fittedTrack;
}



//_________________________________________________________________________________________________
void TrackFitter::initTrack(const Cluster& cl, TMatrixD& covariances)
{
  /// Compute the initial track parameters at the z position of the last cluster (cl)
  /// The covariance matrix is computed such that the last cluster is the only constraint
  /// These parameters are the seed for the Kalman filter



}

//_________________________________________________________________________________________________
void TrackFitter::addCluster(const Cluster& newcl, TMatrixD& covariances)
{
  /// Extrapolate the starting track parameters to the z position of the new cluster
  /// accounting for MCS dispersion in the current chamber and the other(s) crossed
  /// Recompute the parameters adding the cluster constraint with the Kalman filter
  /// Throw an exception in case of failure
}

//_________________________________________________________________________________________________
void TrackFitter::smoothTrack(Track& track, bool finalize)
{
  /// Recompute the track parameters at each cluster using the Smoother
  /// Smoothed parameters are stored in dedicated data members
  /// If finalize, they are copied in the regular parameters in case of success
  /// Throw an exception in case of failure
}

//_________________________________________________________________________________________________
void TrackFitter::runKalmanFilter(TrackParam& trackParam)
{
  /// Compute the new track parameters including the attached cluster with the Kalman filter
  /// The current parameters are supposed to have been extrapolated to the cluster z position
  /// Throw an exception in case of failure

  // get actual track parameters (p)
}

//_________________________________________________________________________________________________
void TrackFitter::runSmoother(const TrackParam& previousParam, TrackParam& param)
{
  /// Recompute the track parameters starting from the previous ones
  /// Throw an exception in case of failure
}

// Define template specializations
template TrackMFTExt TrackFitter::fit<TrackCA&, std::vector<Cluster>&>(TrackCA&, std::vector<Cluster>&);
template TrackMFTExt TrackFitter::fit<TrackLTF&, std::vector<Cluster>&>(TrackLTF&, std::vector<Cluster>&);

} // namespace mft
} // namespace o2
