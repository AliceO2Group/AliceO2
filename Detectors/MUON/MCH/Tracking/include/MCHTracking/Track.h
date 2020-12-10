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
#include <memory>

#include "MCHTracking/Cluster.h"
#include "MCHTracking/TrackParam.h"

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

  Track(const Track& track);
  Track& operator=(const Track& track) = delete;
  Track(Track&&) = delete;
  Track& operator=(Track&&) = delete;

  /// Return the number of attached clusters
  int getNClusters() const { return mParamAtClusters.size(); }

  /// Return the number of degrees of freedom of the track
  int getNDF() const { return 2 * getNClusters() - 5; }

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
  void addParamAtCluster(const TrackParam& param);
  /// Remove the given track parameters from the internal list and return an iterator to the parameters that follow
  auto removeParamAtCluster(std::list<TrackParam>::iterator& itParam) { return mParamAtClusters.erase(itParam); }

  int getNClustersInCommon(const Track& track, int stMin = 0, int stMax = 4) const;

  bool isBetter(const Track& track) const;

  void tagRemovableClusters(uint8_t requestedStationMask, bool request2ChInSameSt45);

  void setCurrentParam(const TrackParam& param, int chamber);
  TrackParam& getCurrentParam();
  /// get a reference to the current chamber on which the current parameters are given
  int& getCurrentChamber() { return mCurrentChamber; }
  /// check whether the current track parameters exist
  bool hasCurrentParam() const { return mCurrentParam ? true : false; }
  /// check if the current parameters are valid
  bool areCurrentParamValid() const { return (mCurrentChamber > -1); }
  /// invalidate the current parameters
  void invalidateCurrentParam() { mCurrentChamber = -1; }

  /// set the flag telling if this track shares cluster(s) with another
  void connected(bool connected = true) { mConnected = connected; }
  /// return the flag telling if this track shares cluster(s) with another
  bool isConnected() const { return mConnected; }

  /// set the flag telling if this track should be deleted
  void removable(bool removable = true) { mRemovable = removable; }
  /// return the flag telling if this track should be deleted
  bool isRemovable() const { return mRemovable; }

  void print() const;

 private:
  std::list<TrackParam> mParamAtClusters{};    ///< list of track parameters at each cluster
  std::unique_ptr<TrackParam> mCurrentParam{}; ///< current track parameters used during tracking
  int mCurrentChamber = -1;                    ///< current chamber on which the current parameters are given
  bool mConnected = false;                     ///< flag telling if this track shares cluster(s) with another
  bool mRemovable = false;                     ///< flag telling if this track should be deleted
};

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_TRACK_H_
