// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <gsl/gsl>
#include "DataFormatsMID/Cluster2D.h"
#include "DataFormatsMID/Cluster3D.h"
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

  /// Sets impact parameter cut
  void setImpactParamCut(float impactParamCut) { mImpactParamCut = impactParamCut; }
  /// Gets the impact parameter cut
  inline float getImpactParamCut() const { return mImpactParamCut; }
  /// Sets number of sigmas for cuts
  void setSigmaCut(float sigmaCut) { mImpactParamCut = mSigmaCut; }
  /// Gets number of sigmas for cuts
  inline float getSigmaCut() const { return mSigmaCut; }

  bool process(gsl::span<const Cluster2D> clusters);
  bool init(bool keepAll = false);

  /// Gets the array of reconstructes tracks
  const std::vector<Track>& getTracks() { return mTracks; }

  /// Gets the array of associated clusters
  const std::vector<Cluster3D>& getClusters() { return mClusters; }

 private:
  bool processSide(bool isRight, bool isInward);
  bool tryAddTrack(const Track& track);
  bool followTrackKeepAll(const Track& track, bool isRight, bool isInward);
  bool followTrackKeepBest(const Track& track, bool isRight, bool isInward);
  bool findAllClusters(Track& track, int clIdx, bool isRight, bool isInward, int chamber, int irpc);
  bool findNextCluster(const Track& track, bool isRight, bool isInward, int chamber, int firstRPC, int lastRPC, Track& bestTrack) const;
  int getFirstNeighbourRPC(int rpc) const;
  int getLastNeighbourRPC(int rpc) const;
  bool loadClusters(gsl::span<const Cluster2D>& clusters);
  bool makeTrackSeed(Track& track, const Cluster3D& cl1, const Cluster3D& cl2) const;
  void reset();
  double runKalmanFilter(Track& track, const Cluster3D& cluster) const;
  double tryOneCluster(const Track& track, const Cluster3D& cluster, Track& newTrack) const;
  void finalizeTrack(Track& track);

  float mImpactParamCut = 210.; ///< Cut on impact parameter
  float mSigmaCut = 5.;         ///< Number of sigmas cut
  float mMaxChi2 = 1.e6;        ///< Maximum cut on chi2

  std::vector<std::pair<int, bool>> mClusterIndexes[72]; ///< Ordered arrays of clusters indexes
  std::vector<Cluster3D> mClusters;                      ///< 3D clusters

  std::vector<Track> mTracks; ///< Array of tracks

  GeometryTransformer mTransformer; ///< Geometry transformer

  typedef bool (Tracker::*TrackerMemFn)(const Track&, bool, bool);
  TrackerMemFn mFollowTrack{nullptr}; ///! Choice of the function to follow the track
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_TRACKER_H */
