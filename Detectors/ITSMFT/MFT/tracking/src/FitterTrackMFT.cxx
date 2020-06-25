// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file FitterTrackMFT.cxx
/// \brief Implementation of the MFT track for internal use
///
/// \author Philippe Pillot, Subatech; adapted by Rafael Pezzi, UFRGS

#include "MFTTracking/FitterTrackMFT.h"

#include <iostream>

#include <FairMQLogger.h>

namespace o2
{
namespace mft
{

using namespace std;

//__________________________________________________________________________
FitterTrackMFT::FitterTrackMFT(const FitterTrackMFT& track)
  : mParamAtVertex(track.mParamAtVertex),
    mParamAtClusters(track.mParamAtClusters),
    mCurrentParam(nullptr),
    mCurrentLayer(-1),
    mConnected(track.mConnected),
    mRemovable(track.mRemovable),
    mMCCompLabels(track.mMCCompLabels),
    mNPoints(track.mNPoints)

{
}

//__________________________________________________________________________
TrackParamMFT& FitterTrackMFT::createParamAtCluster(const Cluster& cluster)
{
  /// Create the object to hold the track parameters at the given cluster
  /// Only the z position of the track is set to the cluster z position
  /// Keep the internal list of track parameters at clusters sorted in z
  /// Return a reference to the newly created parameters

  // find the iterator before which the new element will be constructed
  auto itParam = mParamAtClusters.begin();
  for (; itParam != mParamAtClusters.end(); ++itParam) {
    if (cluster.zCoordinate >= itParam->getZ()) {
      break;
    }
  }

  // add the new track parameters
  mParamAtClusters.emplace(itParam);
  --itParam;
  itParam->setX(cluster.xCoordinate);
  itParam->setY(cluster.yCoordinate);
  itParam->setZ(cluster.zCoordinate);
  itParam->setClusterPtr(&cluster);
  mNPoints++;
  return *itParam;
}

//__________________________________________________________________________
void FitterTrackMFT::addParamAtCluster(const TrackParamMFT& param)
{
  /// Add a copy of the given track parameters in the internal list
  /// The parameters must be associated with a cluster
  /// Keep the internal list of track parameters sorted in clusters z

  const Cluster* cluster = param.getClusterPtr();
  if (cluster == nullptr) {
    LOG(ERROR) << "The TrackParamMFT must be associated with a cluster --> not added";
    return;
  }

  // find the iterator before which the new element will be constructed
  auto itParam = mParamAtClusters.begin();
  for (; itParam != mParamAtClusters.end(); ++itParam) {
    if (cluster->zCoordinate >= itParam->getZ()) {
      break;
    }
  }

  // add the new track parameters
  mParamAtClusters.emplace(itParam, param);
}

//__________________________________________________________________________
bool FitterTrackMFT::isBetter(const FitterTrackMFT& track) const
{
  /// Return true if this track is better than the one given as parameter
  /// It is better if it has more clusters or a better chi2 in case of equality
  int nCl1 = this->getNClusters();
  int nCl2 = track.getNClusters();
  return ((nCl1 > nCl2) || ((nCl1 == nCl2) && (this->first().getTrackChi2() < track.first().getTrackChi2())));
}

//__________________________________________________________________________
void FitterTrackMFT::tagRemovableClusters(uint8_t requestedStationMask)
{
  /// Identify clusters that can be removed from the track,
  /// with the only requirements to have at least 1 cluster per requested station
  /// and at least 2 chambers over 4 in stations 4 & 5 that contain cluster(s)
}

//__________________________________________________________________________
void FitterTrackMFT::setCurrentParam(const TrackParamMFT& param, int chamber)
{
  /// set the current track parameters and the associated chamber
  if (mCurrentParam) {
    *mCurrentParam = param;
  } else {
    mCurrentParam = std::make_unique<TrackParamMFT>(param);
  }
  mCurrentParam->setClusterPtr(nullptr);
  mCurrentLayer = chamber;
}

//__________________________________________________________________________
TrackParamMFT& FitterTrackMFT::getCurrentParam()
{
  /// get a reference to the current track parameters. Create dummy parameters if needed
  if (!mCurrentParam) {
    mCurrentParam = std::make_unique<TrackParamMFT>();
  }
  return *mCurrentParam;
}

} // namespace mft
} // namespace o2
