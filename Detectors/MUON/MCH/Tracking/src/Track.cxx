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

#include <iostream>

#include <FairMQLogger.h>

namespace o2
{
namespace mch
{

using namespace std;

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
TrackParam& Track::addParamAtCluster(TrackParam& param)
{
  /// Add a copy of the given track parameters in the internal list
  /// The parameters must be associated with a cluster
  /// Keep the internal list of track parameters sorted in clusters z
  /// Return a reference to the newly created parameters
  /// or to the original ones in case of error

  const Cluster* cluster = param.getClusterPtr();
  if (cluster == nullptr) {
    LOG(ERROR) << "The TrackParam must be associated with a cluster --> not added";
    return param;
  }

  // find the iterator before which the new element will be constructed
  auto itParam = mParamAtClusters.begin();
  for (; itParam != mParamAtClusters.end(); ++itParam) {
    if (cluster->getZ() >= itParam->getZ()) {
      break;
    }
  }

  // add the new track parameters
  itParam = mParamAtClusters.emplace(itParam, param);

  return *itParam;
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
void Track::tagRemovableClusters(uint8_t requestedStationMask)
{
  /// Identify clusters that can be removed from the track,
  /// with the only requirements to have at least 1 cluster per requested station
  /// and at least 2 chambers over 4 in stations 4 & 5 that contain cluster(s)

  int previousCh(-1), previousSt(-1), nChHitInSt45(0);
  TrackParam* previousParam(nullptr);

  for (auto& param : *this) {

    int currentCh = param.getClusterPtr()->getChamberId();
    int currentSt = currentCh / 2;

    // set the cluster as removable if the station is not requested or if it is not alone in the station
    if (((1 << currentSt) & requestedStationMask) == 0) {
      param.setRemovable(true);
    } else if (currentSt == previousSt) {
      previousParam->setRemovable(true);
      param.setRemovable(true);
    } else {
      param.setRemovable(false);
      previousSt = currentSt;
      previousParam = &param;
    }

    // count the number of chambers in station 4 & 5 that contain cluster(s)
    if (currentCh > 5 && currentCh != previousCh) {
      ++nChHitInSt45;
      previousCh = currentCh;
    }
  }

  // if there are less than 3 chambers containing cluster(s) in station 4 & 5
  if (nChHitInSt45 < 3) {

    previousCh = -1;
    previousParam = nullptr;

    for (auto itParam = this->rbegin(); itParam != this->rend(); ++itParam) {

      int currentCh = itParam->getClusterPtr()->getChamberId();

      if (currentCh < 6) {
        break;
      }

      // set the cluster as not removable unless it is not alone in the chamber
      if (currentCh == previousCh) {
        previousParam->setRemovable(true);
        itParam->setRemovable(true);
      } else {
        itParam->setRemovable(false);
        previousCh = currentCh;
        previousParam = &*itParam;
      }
    }
  }
}

//__________________________________________________________________________
void Track::print() const
{
  /// Print the track parameters at first cluster and the Id of all associated clusters
  mParamAtClusters.front().print();
  cout << "\tclusters = {";
  for (const auto& param : *this) {
    cout << param.getClusterPtr()->getUniqueId() << ", ";
  }
  cout << "}" << endl;
}

} // namespace mch
} // namespace o2
