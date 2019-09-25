// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackFinderOriginal.h
/// \brief Definition of a class to reconstruct tracks with the original algorithm
///
/// \author Philippe Pillot, Subatech

#ifndef ALICEO2_MCH_TRACKFINDERORIGINAL_H_
#define ALICEO2_MCH_TRACKFINDERORIGINAL_H_

#include "Cluster.h"
#include "Track.h"
#include "TrackFitter.h"

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

  void init(float l3Current, float dipoleCurrent);
  const std::list<Track>& findTracks(const std::array<std::list<Cluster>, 10>* clusters);

  /// set the flag to try to find more track candidates starting from 1 cluster in each of station (1..) 4 and 5
  void findMoreTrackCandidates(bool moreCandidates) { mMoreCandidates = moreCandidates; }

  /// set the debug level defining the verbosity
  void debug(int debugLevel) { mDebugLevel = debugLevel; }

 private:
  void findTrackCandidates();
  void findMoreTrackCandidates();
  std::list<Track>::iterator findTrackCandidates(int ch1, int ch2, bool skipUsedPairs = false);
  bool areUsed(const Cluster& cl1, const Cluster& cl2);
  void createTrack(const Cluster& cl1, const Cluster& cl2);
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
  void finalize();
  uint8_t requestedStationMask() const;
  int getTrackIndex(const std::list<Track>::iterator& itCurrentTrack) const;
  void printTracks() const;
  void printTrackParam(const TrackParam& trackParam) const;
  template <class... Args>
  void print(Args... args) const;

  /// sigma cut to select clusters (local chi2) and tracks (global chi2) during tracking
  static constexpr double SSigmaCutForTracking = 5.;
  /// sigma cut to select clusters (local chi2) and tracks (global chi2) during improvement
  static constexpr double SSigmaCutForImprovement = 4.;
  ///< maximum distance to the track to search for compatible cluster(s) in non bending direction
  static constexpr double SMaxNonBendingDistanceToTrack = 1.;
  ///< maximum distance to the track to search for compatible cluster(s) in bending direction
  static constexpr double SMaxBendingDistanceToTrack = 1.;
  static constexpr double SNonBendingVertexDispersion = 70.; ///< vertex dispersion (cm) in non bending plane
  static constexpr double SBendingVertexDispersion = 70.;    ///< vertex dispersion (cm) in bending plane
  static constexpr double SMinBendingMomentum = 0.8;         ///< minimum value (GeV/c) of momentum in bending plane
  /// z position of the chambers
  static constexpr float SDefaultChamberZ[10] = {-526.16, -545.24, -676.4, -695.4, -967.5,
                                                 -998.5, -1276.5, -1307.5, -1406.6, -1437.6};
  /// default chamber thickness in X0 for reconstruction
  static constexpr double SChamberThicknessInX0[10] = {0.065, 0.065, 0.075, 0.075, 0.035,
                                                       0.035, 0.035, 0.035, 0.035, 0.035};
  ///< if true, at least one cluster in the station is requested to validate the track
  static constexpr bool SRequestStation[5] = {true, true, true, true, true};

  TrackFitter mTrackFitter{}; /// track fitter

  const std::array<std::list<Cluster>, 10>* mClusters = nullptr; ///< pointer to the lists of clusters
  std::list<Track> mTracks{};                                    ///< list of reconstructed tracks

  double mMaxMCSAngle2[10]{}; ///< maximum angle dispersion due to MCS

  bool mMoreCandidates = false; ///< try to find more track candidates starting from 1 cluster in each of station (1..) 4 and 5

  int mDebugLevel = 0; ///< debug level defining the verbosity
};

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_TRACKFINDERORIGINAL_H_
