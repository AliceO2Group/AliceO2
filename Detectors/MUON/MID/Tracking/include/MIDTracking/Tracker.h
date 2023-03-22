// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDTracking/Tracker.h
/// \brief  Track reconstruction algorithm for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   09 May 2017

#ifndef O2_MID_TRACKER_H
#define O2_MID_TRACKER_H

#include <vector>
#include <unordered_set>
#include <gsl/gsl>
#include "DataFormatsMID/Cluster.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/Track.h"
#include "MIDBase/GeometryTransformer.h"

namespace o2
{
namespace mid
{
/// Tracking algorithm for MID
class Tracker
{
 public:
  Tracker(const GeometryTransformer& geoTrans);

  /// Gets the impact parameter cut
  inline float getImpactParamCut() const { return mImpactParamCut; }
  /// Gets number of sigmas for cuts
  inline float getSigmaCut() const { return mSigmaCut; }

  void process(gsl::span<const Cluster> clusters, bool accumulate = false);
  void process(gsl::span<const Cluster> clusters, gsl::span<const ROFRecord> rofRecords);
  bool init(bool keepAll = false);

  /// Gets the array of reconstructes tracks
  const std::vector<Track>& getTracks() { return mTracks; }

  /// Gets the array of associated clusters
  const std::vector<Cluster>& getClusters() { return mClusters; }

  /// Gets the vector of tracks RO frame records
  const std::vector<ROFRecord>& getTrackROFRecords() { return mTrackROFRecords; }

  /// Gets the vector of cluster RO frame records
  const std::vector<ROFRecord>& getClusterROFRecords() { return mClusterROFRecords; }

 private:
  void processSide(bool isRight, bool isInward);
  void tryAddTrack(const Track& track);
  void followTrackKeepAll(Track& track, bool isRight, bool isInward);
  bool findAllClusters(const Track& track, bool isRight, int chamber, int firstRPC, int lastRPC, int nextChamber,
                       std::unordered_set<int>& excludedClusters, bool excludeClusters);
  void followTrackKeepBest(Track& track, bool isRight, bool isInward);
  void findNextCluster(const Track& track, bool isRight, bool isInward, int chamber, int firstRPC, int lastRPC, Track& bestTrack) const;
  int getFirstNeighbourRPC(int rpc) const;
  int getLastNeighbourRPC(int rpc) const;
  bool loadClusters(gsl::span<const Cluster>& clusters);
  bool makeTrackSeed(Track& track, const Cluster& cl1, const Cluster& cl2) const;
  void runKalmanFilter(Track& track, const Cluster& cluster) const;
  bool tryOneCluster(const Track& track, int chamber, int clIdx, Track& newTrack) const;
  void excludeUsedClusters(const Track& track, int ch1, int ch2, std::unordered_set<int>& excludedClusters) const;
  bool skipOneChamber(Track& track) const;

  static constexpr float SMT11Z = -1603.5; ///< Position of the first MID chamber (cm)

  float mImpactParamCut = 210.; ///< Cut on impact parameter
  float mSigmaCut = 5.;         ///< Number of sigmas cut
  float mMaxChi2 = 50.;         ///< Maximum cut on chi2 to attach a cluster (= 2 * mSigmaCut^2)

  std::vector<int> mClusterIndexes[72]; ///< Ordered arrays of clusters indexes
  std::vector<Cluster> mClusters{};     ///< 3D clusters

  std::vector<Track> mTracks{};                ///< Vector of tracks
  std::vector<ROFRecord> mTrackROFRecords{};   ///< List of track RO frame records
  std::vector<ROFRecord> mClusterROFRecords{}; ///< List of cluster RO frame records
  size_t mFirstTrackOffset{0};                 ///! Offset for the first track in the current event
  size_t mTrackOffset{0};                      ///! Offset for the track in the current event
  int mNTracksStep1{0};                        ///! Number of tracks found in the first tracking step

  GeometryTransformer mTransformer{}; ///< Geometry transformer

  typedef void (Tracker::*TrackerMemFn)(Track&, bool, bool);
  TrackerMemFn mFollowTrack{&Tracker::followTrackKeepAll}; ///! Choice of the function to follow the track
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_TRACKER_H */
