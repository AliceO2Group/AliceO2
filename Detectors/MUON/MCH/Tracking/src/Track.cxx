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

#include "MCHTracking/Track.h"

#include <iostream>

#include <FairMQLogger.h>

namespace o2
{
namespace mch
{

using namespace std;

//__________________________________________________________________________
Track::Track(const Track& track)
  : mParamAtClusters(track.mParamAtClusters),
    mCurrentParam(nullptr),
    mCurrentChamber(-1),
    mConnected(track.mConnected),
    mRemovable(track.mRemovable)
{
  /// Copy the track, except the current parameters and chamber, which are reset
}

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

//__________________________________________________________________________
void Track::addParamAtCluster(const TrackParam& param)
{
  /// Add a copy of the given track parameters in the internal list
  /// The parameters must be associated with a cluster
  /// Keep the internal list of track parameters sorted in clusters z

  const Cluster* cluster = param.getClusterPtr();
  if (cluster == nullptr) {
    LOG(ERROR) << "The TrackParam must be associated with a cluster --> not added";
    return;
  }

  // find the iterator before which the new element will be constructed
  auto itParam = mParamAtClusters.begin();
  for (; itParam != mParamAtClusters.end(); ++itParam) {
    if (cluster->getZ() >= itParam->getZ()) {
      break;
    }
  }

  // add the new track parameters
  mParamAtClusters.emplace(itParam, param);
}

//__________________________________________________________________________
int Track::getNClustersInCommon(const Track& track, int stMin, int stMax) const
{
  /// Return the number of clusters in common on stations [stMin, stMax]
  /// between this track and the one given as parameter

  int chMin = 2 * stMin;
  int chMax = 2 * stMax + 1;
  int nClustersInCommon(0);

  for (const auto& param1 : *this) {

    int ch1 = param1.getClusterPtr()->getChamberId();
    if (ch1 < chMin || ch1 > chMax) {
      continue;
    }

    for (const auto& param2 : track) {

      int ch2 = param2.getClusterPtr()->getChamberId();
      if (ch2 < chMin || ch2 > chMax) {
        continue;
      }

      if (param1.getClusterPtr()->getUniqueId() == param2.getClusterPtr()->getUniqueId()) {
        ++nClustersInCommon;
        break;
      }
    }
  }

  return nClustersInCommon;
}

//__________________________________________________________________________
bool Track::isBetter(const Track& track) const
{
  /// Return true if this track is better than the one given as parameter
  /// It is better if it has more clusters or a better chi2 in case of equality
  int nCl1 = this->getNClusters();
  int nCl2 = track.getNClusters();
  return ((nCl1 > nCl2) || ((nCl1 == nCl2) && (this->first().getTrackChi2() < track.first().getTrackChi2())));
}

//__________________________________________________________________________
void Track::tagRemovableClusters(uint8_t requestedStationMask, bool request2ChInSameSt45)
{
  /// Identify clusters that can be removed from the track, with the requirement
  /// to have enough chambers fired to fulfill the tracking criteria

  // count the number of clusters in each chamber and the number of chambers fired on stations 4 and 5
  int nClusters[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (auto& param : *this) {
    ++nClusters[param.getClusterPtr()->getChamberId()];
  }
  int nChFiredInSt4 = (nClusters[6] > 0) ? 1 : 0;
  if (nClusters[7] > 0) {
    ++nChFiredInSt4;
  }
  int nChFiredInSt5 = (nClusters[8] > 0) ? 1 : 0;
  if (nClusters[9] > 0) {
    ++nChFiredInSt5;
  }
  int nChFiredInSt45 = nChFiredInSt4 + nChFiredInSt5;

  bool removable[10] = {false, false, false, false, false, false, false, false, false, false};

  // for station 1, 2 and 3, there must be at least one cluster per requested station
  for (int iCh = 0; iCh < 6; iCh += 2) {
    if (nClusters[iCh] + nClusters[iCh + 1] > 1 || (requestedStationMask & (1 << (iCh / 2))) == 0) {
      removable[iCh] = removable[iCh + 1] = true;
    }
  }

  // for station 4 and 5, there must be at least one cluster per requested station and
  // at least 2 chambers fired (on the same station or not depending on the requirement)
  if (nChFiredInSt45 == 4) {
    removable[6] = removable[7] = removable[8] = removable[9] = true;
  } else if (nChFiredInSt45 == 3) {
    if (nChFiredInSt4 == 2 && request2ChInSameSt45) {
      removable[6] = (nClusters[6] > 1);
      removable[7] = (nClusters[7] > 1);
    } else if (nClusters[6] + nClusters[7] > 1 || (requestedStationMask & 0x8) == 0) {
      removable[6] = removable[7] = true;
    }
    if (nChFiredInSt5 == 2 && request2ChInSameSt45) {
      removable[8] = (nClusters[8] > 1);
      removable[9] = (nClusters[9] > 1);
    } else if (nClusters[8] + nClusters[9] > 1 || (requestedStationMask & 0x10) == 0) {
      removable[8] = removable[9] = true;
    }
  } else {
    for (int iCh = 6; iCh < 10; ++iCh) {
      removable[iCh] = (nClusters[iCh] > 1);
    }
  }

  // tag the removable clusters
  for (auto& param : *this) {
    param.setRemovable(removable[param.getClusterPtr()->getChamberId()]);
  }
}

//__________________________________________________________________________
void Track::setCurrentParam(const TrackParam& param, int chamber)
{
  /// set the current track parameters and the associated chamber
  if (mCurrentParam) {
    *mCurrentParam = param;
  } else {
    mCurrentParam = std::make_unique<TrackParam>(param);
  }
  mCurrentParam->setClusterPtr(nullptr);
  mCurrentChamber = chamber;
}

//__________________________________________________________________________
TrackParam& Track::getCurrentParam()
{
  /// get a reference to the current track parameters. Create dummy parameters if needed
  if (!mCurrentParam) {
    mCurrentParam = std::make_unique<TrackParam>();
  }
  return *mCurrentParam;
}

//__________________________________________________________________________
void Track::print() const
{
  /// Print the track parameters at first cluster and the Id of all associated clusters
  mParamAtClusters.front().print();
  cout << "\tcurrent chamber = " << mCurrentChamber + 1 << " ; clusters = {";
  for (const auto& param : *this) {
    cout << param.getClusterPtr()->getIdAsString() << ", ";
  }
  cout << "}" << endl;
}

} // namespace mch
} // namespace o2
