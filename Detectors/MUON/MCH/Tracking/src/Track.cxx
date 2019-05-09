// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Track.cxx
/// \brief Implementation of the MCH track for internal use
///
/// \author Philippe Pillot, Subatech

#include "Track.h"

#include <FairMQLogger.h>

namespace o2
{
namespace mch
{

//__________________________________________________________________________
TrackParam& Track::createParamAtCluster(const Cluster& cluster)
{
  /// Create the object to hold the track parameters at the given cluster
  /// Only the z position of the track is set to the cluster z position
  /// Keep the internal list of track parameters at clusters sorted in z
  /// Return a reference to the newly created parameters

  // find the iterator before which the new element will be constructed
  auto itParam = mParamAtClusters.begin();
  for (; itParam != mParamAtClusters.end(); ++itParam) {
    if (cluster.getZ() >= itParam->getZ()) {
      break;
    }
  }

  // add the new track parameters
  mParamAtClusters.emplace(itParam);
  --itParam;
  itParam->setZ(cluster.getZ());
  itParam->setClusterPtr(&cluster);

  return *itParam;
}

} // namespace mch
} // namespace o2
