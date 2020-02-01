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

#include <stdexcept>

#include <TMatrixD.h>

namespace o2
{
namespace mft
{

using namespace std;
using Track = o2::mft::TrackCA;

//_________________________________________________________________________________________________
void TrackFitter::initField(float l3Current)
{
  /// Create the magnetic field map if not
}

//_________________________________________________________________________________________________
void TrackFitter::fit(Track& track, bool smooth, bool finalize,
                      std::list<TrackParam>::reverse_iterator* itStartingParam)
{
  /// Fit a track to its attached clusters
  /// Smooth the track if requested and the smoother enabled
  /// If finalize = true: copy the smoothed parameters, if any, into the regular ones
  /// Fit the entire track or only the part upstream itStartingParam
  /// Throw an exception in case of failure
}

//_________________________________________________________________________________________________
void TrackFitter::initTrack(const Cluster& cl1, const Cluster& cl2, TrackParam& param)
{
  /// Compute the initial track parameters at the z position of the last cluster (cl2)
  /// The covariance matrix is computed such that the last cluster is the only constraint
  /// (by assigning an infinite dispersion to the other cluster)
  /// These parameters are the seed for the Kalman filter

  // compute the track parameters at the last cluster
}

//_________________________________________________________________________________________________
void TrackFitter::addCluster(const TrackParam& startingParam, const Cluster& cl, TrackParam& param)
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

} // namespace mft
} // namespace o2
