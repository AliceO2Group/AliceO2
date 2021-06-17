// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackFinder.h
/// \brief Definition of a class to reconstruct tracks
///
/// \author Philippe Pillot, Subatech

#ifndef ALICEO2_MCH_TRACKFINDER_H_
#define ALICEO2_MCH_TRACKFINDER_H_

#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <array>
#include <vector>
#include <utility>

#include "MCHTracking/Cluster.h"
#include "MCHTracking/Track.h"
#include "MCHTracking/TrackFitter.h"

namespace o2
{
namespace mch
{

/// Class to reconstruct tracks
class TrackFinder
{
 public:
  TrackFinder() = default;
  ~TrackFinder() = default;

  TrackFinder(const TrackFinder&) = delete;
  TrackFinder& operator=(const TrackFinder&) = delete;
  TrackFinder(TrackFinder&&) = delete;
  TrackFinder& operator=(TrackFinder&&) = delete;

  void init(float l3Current, float dipoleCurrent);

  const std::list<Track>& findTracks(const std::unordered_map<int, std::list<Cluster>>& clusters);

  /// set the debug level defining the verbosity
  void debug(int debugLevel) { mDebugLevel = debugLevel; }

  void printStats() const;
  void printTimers() const;

 private:
  void findTrackCandidates();
  void findTrackCandidatesInSt5();
  void findTrackCandidatesInSt4();
  void findMoreTrackCandidates();
  std::list<Track>::iterator findTrackCandidates(int plane1, int plane2, bool skipUsedPairs, const std::list<Track>::iterator& itFirstTrack);

  std::list<Track>::iterator followTrackInOverlapDE(const std::list<Track>::iterator& itTrack, int currentDE, int plane);
  std::list<Track>::iterator followTrackInChamber(std::list<Track>::iterator& itTrack,
                                                  int chamber, int lastChamber, bool canSkip,
                                                  std::unordered_map<int, std::unordered_set<uint32_t>>& excludedClusters);
  std::list<Track>::iterator followTrackInChamber(std::list<Track>::iterator& itTrack,
                                                  int plane1, int plane2, int lastChamber,
                                                  std::unordered_map<int, std::unordered_set<uint32_t>>& excludedClusters);
  std::list<Track>::iterator addClustersAndFollowTrack(std::list<Track>::iterator& itTrack, const TrackParam& paramAtCluster1,
                                                       const TrackParam* paramAtCluster2, int nextChamber, int lastChamber,
                                                       std::unordered_map<int, std::unordered_set<uint32_t>>& excludedClusters);

  void improveTracks();

  void removeConnectedTracks(int stMin, int stMax);

  void refineTracks();

  void finalize();

  void createTrack(const Cluster& cl1, const Cluster& cl2);

  bool isAcceptable(const TrackParam& param) const;

  void prepareForwardTracking(std::list<Track>::iterator& itTrack, bool runSmoother);
  void prepareBackwardTracking(std::list<Track>::iterator& itTrack, bool refit);
  void setCurrentParam(Track& track, const TrackParam& param, int chamber, bool smoothed = false);
  bool propagateCurrentParam(Track& track, int chamber);

  bool areUsed(const Cluster& cl1, const Cluster& cl2, const std::list<Track>::iterator& itFirstTrack, const std::list<Track>::iterator& itLastTrack);
  void excludeClustersFromIdenticalTracks(const std::list<Track>::iterator& itTrack,
                                          std::unordered_map<int, std::unordered_set<uint32_t>>& excludedClusters,
                                          const std::list<Track>::iterator& itEndTrack);
  void moveClusters(std::unordered_map<int, std::unordered_set<uint32_t>>& source, std::unordered_map<int, std::unordered_set<uint32_t>>& destination);

  bool isCompatible(const TrackParam& param, const Cluster& cluster, TrackParam& paramAtCluster);
  bool tryOneClusterFast(const TrackParam& param, const Cluster& cluster);
  double tryOneCluster(const TrackParam& param, const Cluster& cluster, TrackParam& paramAtCluster);

  uint8_t requestedStationMask() const;

  int getTrackIndex(const std::list<Track>::iterator& itCurrentTrack) const;
  void printTracks() const;
  void printTrack(const Track& track) const;
  void printTrackParam(const TrackParam& trackParam) const;
  template <class... Args>
  void print(Args... args) const;

  /// return the chamber to which this plane belong to
  int getChamberId(int plane) { return (plane < 8) ? plane / 2 : 4 + (plane - 8) / 4; }

  ///< maximum distance to the track to search for compatible cluster(s) in non bending direction
  static constexpr double SMaxNonBendingDistanceToTrack = 1.;
  ///< maximum distance to the track to search for compatible cluster(s) in bending direction
  static constexpr double SMaxBendingDistanceToTrack = 1.;
  static constexpr double SMinBendingMomentum = 0.8; ///< minimum value (GeV/c) of momentum in bending plane
  /// z position of the chambers
  static constexpr double SDefaultChamberZ[10] = {-526.16, -545.24, -676.4, -695.4, -967.5,
                                                  -998.5, -1276.5, -1307.5, -1406.6, -1437.6};
  /// default chamber thickness in X0 for reconstruction
  static constexpr double SChamberThicknessInX0[10] = {0.065, 0.065, 0.075, 0.075, 0.035,
                                                       0.035, 0.035, 0.035, 0.035, 0.035};
  static constexpr int SNDE[10] = {4, 4, 4, 4, 18, 18, 26, 26, 26, 26}; ///< number of DE per chamber

  TrackFitter mTrackFitter{}; /// track fitter

  std::array<std::vector<std::pair<const int, const std::list<Cluster>*>>, 32> mClusters{}; ///< array of pointers to the lists of clusters per DE

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
  std::chrono::duration<double> mTimeImproveTracks{};      ///< timer
  std::chrono::duration<double> mTimeCleanTracks{};        ///< timer
  std::chrono::duration<double> mTimeRefineTracks{};       ///< timer
};

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_TRACKFINDER_H_
