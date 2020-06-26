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

#include "MFTTracking/Cluster.h"
#include "MFTTracking/TrackCA.h"
#include "MFTTracking/FitterTrackMFT.h"
#include "MFTTracking/TrackParamMFT.h"
#include "MFTTracking/TrackExtrap.h"
#include "MFTTracking/MFTTrackingParam.h"
#include "DataFormatsMFT/TrackMFT.h"

#include <TLinearFitter.h>
#include <list>

namespace o2
{
namespace mft
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

  void setBz(float bZ);

  /// Enable/disable the smoother (and the saving of related parameters)
  void smoothTracks(bool smooth) { mSmooth = smooth; }
  /// Return the smoother enable/disable flag
  bool isSmootherEnabled() { return mSmooth; }

  bool fit(FitterTrackMFT& track, bool smooth = true, bool finalize = true,
           std::list<TrackParamMFT>::reverse_iterator* itStartingParam = nullptr);

  bool runKalmanFilter(TrackParamMFT& trackParam);

  /// Return the maximum chi2 above which the track can be considered as abnormal
  static constexpr double getMaxChi2() { return SMaxChi2; }

 private:
  void initTrack(const Cluster& cl, TrackParamMFT& param);
  bool addCluster(const TrackParamMFT& startingParam, const Cluster& cl, TrackParamMFT& param);
  bool smoothTrack(FitterTrackMFT& track, bool finalize);
  bool runSmoother(const TrackParamMFT& previousParam, TrackParamMFT& param);
  Float_t mBZField;                         // kiloGauss.
  static constexpr double SMaxChi2 = 2.e10; ///< maximum chi2 above which the track can be considered as abnormal
  /// default layer thickness in X0 for reconstruction  //FIXME: set values for the MFT
  static constexpr double SLayerThicknessInX0[10] = {0.065, 0.065, 0.075, 0.075, 0.035,
                                                     0.035, 0.035, 0.035, 0.035, 0.035};

  bool mSmooth = false; ///< switch ON/OFF the smoother
  bool mFieldON = true;
  o2::mft::TrackExtrap mTrackExtrap;
};

// Functions to estimate momentum and charge from track curvature
Double_t invQPtFromParabola(const FitterTrackMFT& track, double bFieldZ, Double_t& chi2);
Double_t QuadraticRegression(Int_t nVal, Double_t* xVal, Double_t* yVal, Double_t& p0, Double_t& p1, Double_t& p2);

} // namespace mft
} // namespace o2

#endif // ALICEO2_MFT_TRACKFITTER_H_
