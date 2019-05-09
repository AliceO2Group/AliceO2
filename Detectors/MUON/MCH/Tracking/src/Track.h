// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Track.h
/// \brief Definition of the MCH track for internal use
///
/// \author Philippe Pillot, Subatech

#ifndef ALICEO2_MCH_TRACK_H_
#define ALICEO2_MCH_TRACK_H_

#include <list>

#include "Cluster.h"
#include "TrackParam.h"

namespace o2
{
namespace mch
{

/// track for internal use
class Track
{
 public:
  Track() = default;
  ~Track() = default;

  Track(const Track& track) = default;
  Track& operator=(const Track& track) = default;
  Track(Track&&) = delete;
  Track& operator=(Track&&) = delete;

  /// Return a reference to the track parameters at vertex
  const TrackParam& getParamAtVertex() const { return mParamAtVertex; }

  /// Return the number of attached clusters
  int getNClusters() const { return mParamAtClusters.size(); }

  /// Return a reference to the track parameters at first cluster
  const TrackParam& first() const { return mParamAtClusters.front(); }
  /// Return a reference to the track parameters at last cluster
  const TrackParam& last() const { return mParamAtClusters.back(); }

  /// Return an iterator to the track parameters at clusters (point to the first one)
  auto begin() { return mParamAtClusters.begin(); }
  auto begin() const { return mParamAtClusters.begin(); }
  /// Return an iterator passing the track parameters at last cluster
  auto end() { return mParamAtClusters.end(); }
  auto end() const { return mParamAtClusters.end(); }
  /// Return a reverse iterator to the track parameters at clusters (point to the last one)
  auto rbegin() { return mParamAtClusters.rbegin(); }
  auto rbegin() const { return mParamAtClusters.rbegin(); }
  /// Return a reverse iterator passing the track parameters at first cluster
  auto rend() { return mParamAtClusters.rend(); }
  auto rend() const { return mParamAtClusters.rend(); }

  TrackParam& createParamAtCluster(const Cluster& cluster);

 private:
  TrackParam mParamAtVertex{};              ///< track parameters at vertex
  std::list<TrackParam> mParamAtClusters{}; ///< list of track parameters at each cluster
};

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_TRACK_H_
