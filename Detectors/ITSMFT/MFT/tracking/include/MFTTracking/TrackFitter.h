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
/// \author Philippe Pillot, Subatech; adapted by Rafael Pezzi, UFRGS

#ifndef ALICEO2_MFT_TRACKFITTER_H_
#define ALICEO2_MFT_TRACKFITTER_H_

//#include "MFTTracking/Cluster.h"
#include "MFTTracking/TrackCA.h"
#include "MFTTracking/FitterTrackMFT.h"
#include "MFTTracking/TrackParam.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsITSMFT/Cluster.h"

#include <TLinearFitter.h>
#include <list>

namespace o2
{
namespace mft
{
//using Track = o2::mft::FitterTrackMFT;
//using Cluster = o2::itsmft::Cluster;

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

  void fit(FitterTrackMFT& track, bool smooth = true, bool finalize = true,
           std::list<TrackParam>::reverse_iterator* itStartingParam = nullptr);

  void runKalmanFilter(TrackParam& trackParam);

  /// Return the maximum chi2 above which the track can be considered as abnormal
  static constexpr double getMaxChi2() { return SMaxChi2; }

 private:
  void initTrack(const o2::itsmft::Cluster& cl, const o2::itsmft::Cluster& cl2, TrackParam& param);
  void addCluster(const TrackParam& startingParam, const o2::itsmft::Cluster& cl, TrackParam& param);
  void smoothTrack(FitterTrackMFT& track, bool finalize);
  void runSmoother(const TrackParam& previousParam, TrackParam& param);
  Float_t bField = 0.5;                     // Tesla. TODO: calculate value according to the solenoid current
  static constexpr double SMaxChi2 = 2.e10; ///< maximum chi2 above which the track can be considered as abnormal
  /// z position of the layers
  static constexpr float SDefaultLayerZ[10] = {-45.3, -46.7, -48.6, -50.0, -52.4, -53.8, -67.7, -69.1, -76.1, -77.5};
  /// default layer thickness in X0 for reconstruction  //FIXME: set values for the MFT
  static constexpr double SLayerThicknessInX0[10] = {0.065, 0.065, 0.075, 0.075, 0.035,
                                                     0.035, 0.035, 0.035, 0.035, 0.035};

  bool mSmooth = false; ///< switch ON/OFF the smoother
  bool mFieldON = true;
};

static bool extrapToZ(TrackParam* trackParam, double zEnd, bool isFieldON = true);
static bool extrapToZCov(TrackParam* trackParam, double zEnd, bool updatePropagator = false, bool isFieldON = true);
static void linearExtrapToZ(TrackParam* trackParam, double zEnd);
static void linearExtrapToZCov(TrackParam* trackParam, double zEnd, bool updatePropagator);
static void addMCSEffect(TrackParam* trackParam, double dZ, double x0, bool isFieldON = true);
static Double_t momentumFromSagitta(FitterTrackMFT& track);

} // namespace mft
} // namespace o2

#endif // ALICEO2_MFT_TRACKFITTER_H_
