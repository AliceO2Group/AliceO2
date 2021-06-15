// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackFitter.h
/// \brief Definition of a class to fit a track to a set of clusters
///
/// \author Philippe Pillot, Subatech

#ifndef ALICEO2_MCH_TRACKFITTER_H_
#define ALICEO2_MCH_TRACKFITTER_H_

#include "MCHTracking/Cluster.h"
#include "MCHTracking/Track.h"
#include "MCHTracking/TrackParam.h"

namespace o2
{
namespace mch
{

/// Class to fit a track to a set of clusters
class TrackFitter
{
 public:
  TrackFitter() = default;
  ~TrackFitter() = default;

  TrackFitter(const TrackFitter&) = delete;
  TrackFitter& operator=(const TrackFitter&) = delete;
  TrackFitter(TrackFitter&&) = delete;
  TrackFitter& operator=(TrackFitter&&) = delete;

  void initField(float l3Current, float dipoleCurrent);

  /// Set the vertex dispersion in y direction used for the track covariance seed
  void setBendingVertexDispersion(double ey) { mBendingVertexDispersion2 = ey * ey; }

  /// Enable/disable the smoother (and the saving of related parameters)
  void smoothTracks(bool smooth) { mSmooth = smooth; }
  /// Return the smoother enable/disable flag
  bool isSmootherEnabled() { return mSmooth; }

  /// Use the own resolution of each cluster during the fit (default)
  void useClusterResolution() { mUseChamberResolution = false; }
  /// Use the chamber resolution instead of cluster resolution during the fit
  void useChamberResolution() { mUseChamberResolution = true; }
  // Set the chamber resolution in x and y directions
  void setChamberResolution(double ex, double ey);

  void fit(Track& track, bool smooth = true, bool finalize = true,
           std::list<TrackParam>::reverse_iterator* itStartingParam = nullptr);

  void runKalmanFilter(TrackParam& trackParam);

  /// Return the maximum chi2 above which the track can be considered as abnormal
  static constexpr double getMaxChi2() { return SMaxChi2; }

 private:
  void initTrack(const Cluster& cl1, const Cluster& cl2, TrackParam& param);
  void addCluster(const TrackParam& startingParam, const Cluster& cl, TrackParam& param);
  void smoothTrack(Track& track, bool finalize);
  void runSmoother(const TrackParam& previousParam, TrackParam& param);

  static constexpr double SMaxChi2 = 2.e10; ///< maximum chi2 above which the track can be considered as abnormal
  /// z position of the chambers
  static constexpr double SDefaultChamberZ[10] = {-526.16, -545.24, -676.4, -695.4, -967.5,
                                                  -998.5, -1276.5, -1307.5, -1406.6, -1437.6};
  /// default chamber thickness in X0 for reconstruction
  static constexpr double SChamberThicknessInX0[10] = {0.065, 0.065, 0.075, 0.075, 0.035,
                                                       0.035, 0.035, 0.035, 0.035, 0.035};

  double mBendingVertexDispersion2 = 4900.; ///< vertex dispersion square (cm^2) in y direction

  bool mSmooth = false; ///< switch ON/OFF the smoother

  bool mUseChamberResolution = false; ///< switch between using chamber or cluster resolution
  double mChamberResolutionX2 = 0.04; ///< chamber resolution square in x direction
  double mChamberResolutionY2 = 0.04; ///< chamber resolution square in y direction
};

//_________________________________________________________________________________________________
inline void TrackFitter::setChamberResolution(double ex, double ey)
{
  /// Set the chamber resolution in x and y directions
  mChamberResolutionX2 = ex * ex;
  mChamberResolutionY2 = ey * ey;
}

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_TRACKFITTER_H_
