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

#ifndef ALICEO2_MFT_TRACKFITTER2_H_
#define ALICEO2_MFT_TRACKFITTER2_H_

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
class TrackFitter2
{

  using SMatrix55 = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;
  using SMatrix5 = ROOT::Math::SVector<Double_t, 5>;

 public:
  TrackFitter2() = default;
  ~TrackFitter2() = default;

  TrackFitter2(const TrackFitter2&) = delete;
  TrackFitter2& operator=(const TrackFitter2&) = delete;
  TrackFitter2(TrackFitter2&&) = delete;
  TrackFitter2& operator=(TrackFitter2&&) = delete;

  void setBz(float bZ);

  bool fit(TrackLTF& track);

  bool fit(FitterTrackMFT& track, std::list<TrackParamMFT>::reverse_iterator* itStartingParam = nullptr);

  bool runKalmanFilter(TrackLTF& track, int cluster);

  /// Return the maximum chi2 above which the track can be considered as abnormal
  static constexpr double getMaxChi2() { return SMaxChi2; }

 private:
  bool initTrack(TrackLTF& track);
  bool addCluster(TrackLTF& track, int cluster);

  Float_t mBZField;                         // kiloGauss.
  static constexpr double SMaxChi2 = 2.e10; ///< maximum chi2 above which the track can be considered as abnormal
  /// default layer thickness in X0 for reconstruction  //FIXME: set values for the MFT
  static constexpr double SLayerThicknessInX0[10] = {0.065, 0.065, 0.075, 0.075, 0.035,
                                                     0.035, 0.035, 0.035, 0.035, 0.035};

  bool mFieldON = true;
  o2::mft::TrackExtrap mTrackExtrap;
};

// Functions to estimate momentum and charge from track curvature
Double_t invQPtFromParabola2(const FitterTrackMFT& track, double bFieldZ, Double_t& chi2);
Double_t invQPtFromParabola2(const TrackLTF& track, double bFieldZ, Double_t& chi2);
Double_t QuadraticRegression2(Int_t nVal, Double_t* xVal, Double_t* yVal, Double_t& p0, Double_t& p1, Double_t& p2);

} // namespace mft
} // namespace o2

#endif // ALICEO2_MFT_TrackFitter2_H_
