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

/// \file TrackFinderOriginal.h
/// \brief Definition of a class to reconstruct tracks with the original algorithm
///
/// \author Philippe Pillot, Subatech

#ifndef O2_MCH_TRACKFINDERORIGINAL_H_
#define O2_MCH_TRACKFINDERORIGINAL_H_

#include <chrono>

#include "DataFormatsMCH/Cluster.h"
#include "MCHTracking/Track.h"
#include "MCHTracking/TrackFitter.h"

namespace o2
{
namespace mch
{

/// Class to reconstruct tracks with the original algorithm
class TrackFinderOriginal
{
 public:
  TrackFinderOriginal() = default;
  ~TrackFinderOriginal() = default;

  TrackFinderOriginal(const TrackFinderOriginal&) = delete;
  TrackFinderOriginal& operator=(const TrackFinderOriginal&) = delete;
  TrackFinderOriginal(TrackFinderOriginal&&) = delete;
  TrackFinderOriginal& operator=(TrackFinderOriginal&&) = delete;

  void init();
  void initField(float l3Current, float dipoleCurrent);

  const std::list<Track>& findTracks(const std::array<std::list<const Cluster*>, 10>& clusters);

  /// set the debug level defining the verbosity
  void debug(int debugLevel) { mDebugLevel = debugLevel; }

  void printStats() const;
  void printTimers() const;

 private:
  void findTrackCandidates();
  void findMoreTrackCandidates();
  std::list<Track>::iterator findTrackCandidates(int ch1, int ch2, bool skipUsedPairs = false);
  bool areUsed(const Cluster& cl1, const Cluster& cl2);
  void createTrack(const Cluster& cl1, const Cluster& cl2);
  std::list<Track>::iterator addTrack(const std::list<Track>::iterator& pos, const Track& track);
  bool isAcceptable(const TrackParam& param) const;
  void removeDuplicateTracks();
  void removeConnectedTracks(int stMin, int stMax);
  void followTracks(const std::list<Track>::iterator& itTrackBegin, const std::list<Track>::iterator& itTrackEnd, int nextStation);
  std::list<Track>::iterator followTrackInStation(const std::list<Track>::iterator& itTrack, int nextStation);
  std::list<Track>::iterator followLinearTrackInChamber(const std::list<Track>::iterator& itTrack, int nextChamber);
  bool tryOneClusterFast(const TrackParam& param, const Cluster& cluster);
  double tryOneCluster(const TrackParam& param, const Cluster& cluster, TrackParam& paramAtCluster, bool updatePropagator);
  void updateTrack(Track& track, TrackParam& trackParamAtCluster);
  void updateTrack(Track& track, TrackParam& trackParamAtCluster1, TrackParam& trackParamAtCluster2);
  std::list<Track>::iterator recoverTrack(std::list<Track>::iterator& itTrack, int nextStation);
  bool completeTracks();
  void improveTracks();
  void refineTracks();
  void finalize();
  uint8_t requestedStationMask() const;
  int getTrackIndex(const std::list<Track>::iterator& itCurrentTrack) const;
  void printTracks() const;
  void printTrackParam(const TrackParam& trackParam) const;
  template <class... Args>
  void print(Args... args) const;

  /// maximum distance to the track to search for compatible cluster(s) in non bending direction
  static constexpr double SMaxNonBendingDistanceToTrack = 1.;
  /// maximum distance to the track to search for compatible cluster(s) in bending direction
  static constexpr double SMaxBendingDistanceToTrack = 1.;
  static constexpr double SMinBendingMomentum = 0.8; ///< minimum value (GeV/c) of momentum in bending plane
  /// z position of the chambers
  static constexpr double SDefaultChamberZ[10] = {-526.16, -545.24, -676.4, -695.4, -967.5,
                                                  -998.5, -1276.5, -1307.5, -1406.6, -1437.6};
  /// default chamber thickness in X0 for reconstruction
  static constexpr double SChamberThicknessInX0[10] = {0.065, 0.065, 0.075, 0.075, 0.035,
                                                       0.035, 0.035, 0.035, 0.035, 0.035};

  TrackFitter mTrackFitter{}; /// track fitter

  const std::array<std::list<const Cluster*>, 10>* mClusters = nullptr; ///< pointer to the lists of clusters

  std::list<Track> mTracks{}; ///< list of reconstructed tracks

  double mChamberResolutionX2 = 0.;      ///< chamber resolution square (cm^2) in x direction
  double mChamberResolutionY2 = 0.;      ///< chamber resolution square (cm^2) in y direction
  double mBendingVertexDispersion2 = 0.; ///< vertex dispersion square (cm^2) in y direction
  double mMaxChi2ForTracking = 0.;       ///< maximum chi2 to accept a cluster candidate during tracking
  double mMaxChi2ForImprovement = 0.;    ///< maximum chi2 to accept a cluster candidate during improvement

  double mMaxMCSAngle2[10]{}; ///< maximum angle dispersion due to MCS

  int mDebugLevel = 0; ///< debug level defining the verbosity

  std::size_t mNCandidates = 0;            ///< counter
  std::size_t mNCallTryOneCluster = 0;     ///< counter
  std::size_t mNCallTryOneClusterFast = 0; ///< counter

  std::chrono::duration<double> mTimeFindCandidates{};     ///< timer
  std::chrono::duration<double> mTimeFindMoreCandidates{}; ///< timer
  std::chrono::duration<double> mTimeFollowTracks{};       ///< timer
  std::chrono::duration<double> mTimeCompleteTracks{};     ///< timer
  std::chrono::duration<double> mTimeImproveTracks{};      ///< timer
  std::chrono::duration<double> mTimeCleanTracks{};        ///< timer
  std::chrono::duration<double> mTimeRefineTracks{};       ///< timer
};

} // namespace mch
} // namespace o2

#endif // O2_MCH_TRACKFINDERORIGINAL_H_
