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

#ifndef ALICEO2_MFT_TRACKFITTER_H_
#define ALICEO2_MFT_TRACKFITTER_H_

#include "MFTTracking/Cluster.h"
#include "MFTTracking/TrackCA.h"
#include "MFTTracking/TrackParam.h"
#include "DataFormatsMFT/TrackMFT.h"

#include <list>

namespace o2
{
namespace mft
{
using Track = o2::mft::TrackCA;

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

  void initField(float l3Current);

  /// Enable/disable the smoother (and the saving of related parameters)
  void smoothTracks(bool smooth) { mSmooth = smooth; }
  /// Return the smoother enable/disable flag
  bool isSmootherEnabled() { return mSmooth; }

  //void fit(Track& track, bool smooth = true, bool finalize = true,
  //         std::list<TrackParam>::reverse_iterator* itStartingParam = nullptr);

  void runKalmanFilter(TrackParam& trackParam);

  template <typename T, typename C>
  TrackMFTExt fit(T&& track, C&& clusters);


  /// Return the maximum chi2 above which the track can be considered as abnormal
  static constexpr double getMaxChi2() { return SMaxChi2; }

 private:
  void initTrack(const Cluster& cl, TMatrixD& covariances);
  void addCluster(const Cluster& newcl, TMatrixD& covariances);
  void smoothTrack(Track& track, bool finalize);
  void runSmoother(const TrackParam& previousParam, TrackParam& param);
  Float_t bField = 0.5; // Tesla. TODO: calculate value according to the solenoid current
  static constexpr double SMaxChi2 = 2.e10; ///< maximum chi2 above which the track can be considered as abnormal

  bool mSmooth = false; ///< switch ON/OFF the smoother
};

} // namespace mft
} // namespace o2

#endif // ALICEO2_MFT_TRACKFITTER_H_
