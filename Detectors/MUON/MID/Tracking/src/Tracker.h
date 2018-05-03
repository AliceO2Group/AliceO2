// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Tracking/src/Tracker.h
/// \brief  Track reconstruction algorithm for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   09 May 2017

#ifndef O2_MID_TRACKER_H
#define O2_MID_TRACKER_H

#include <vector>
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
  Tracker();
  virtual ~Tracker() = default;

  Tracker(const Tracker&) = delete;
  Tracker& operator=(const Tracker&) = delete;
  Tracker(Tracker&&) = delete;
  Tracker& operator=(Tracker&&) = delete;

  /// Sets impact parameter cut
  void setImpactParamCut(float impactParamCut) { mImpactParamCut = impactParamCut; }
  /// Gets the impact parameter cut
  inline float getImpactParamCut() const { return mImpactParamCut; }
  /// Sets number of sigmas for cuts
  void setSigmaCut(float sigmaCut) { mImpactParamCut = mSigmaCut; }
  /// Gets number of sigmas for cuts
  inline float getSigmaCut() const { return mSigmaCut; }

  bool process(const std::vector<Cluster2D>& clusters);
  bool init();

  /// Gets the array of reconstructes tracks
  const std::vector<Track>& getTracks() { return mTracks; }

  /// Gets the number of reconstructed tracks
  unsigned long int getNTracks() { return mNTracks; }

 private:
  bool processSide(bool isRight, bool isInward);
  bool addTrack(const Track& track);
  bool followTrack(const Track& track, bool isRight, bool isInward);
  bool findNextCluster(const Track& track, bool isRight, bool isInward, int chamber, int firstRPC, int lastRPC,
                       int& nFiredChambers, double& bestChi2, Track& bestTrack, double chi2 = 0., int depth = 1) const;
  int getClusterId(int id, int deId) const;
  int getFirstNeighbourRPC(int rpc) const;
  int getLastNeighbourRPC(int rpc) const;
  bool loadClusters(const std::vector<Cluster2D>& clusters);
  bool makeTrackSeed(Track& track, const Cluster3D& cl1, const Cluster3D& cl2) const;
  void reset();
  double runKalmanFilter(Track& track, const Cluster3D& cluster) const;
  double tryOneCluster(const Track& track, const Cluster3D& cluster, Track& newTrack) const;
  void finalizeTrack(Track& track);

  float mImpactParamCut; ///< Cut on impact parameter
  float mSigmaCut;       ///< Number of sigmas cut
  float mMaxChi2;        ///< Maximum cut on chi2

  std::vector<Cluster3D> mClusters[72]; ///< Ordered arrays of clusters
  unsigned long int mNClusters[72];     ///< Number of clusters per RPC

  std::vector<Track> mTracks; ///< Array of tracks
  unsigned long int mNTracks; ///< Number of tracks

  GeometryTransformer mTransformer; ///< Geometry transformer
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_TRACKER_H */
