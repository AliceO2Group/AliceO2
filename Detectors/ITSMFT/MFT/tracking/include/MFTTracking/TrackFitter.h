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

/// \file TrackFitter.h
/// \brief Definition of a class to fit a track to a set of clusters
///
/// \author Philippe Pillot, Subatech; adapted by Rafael Pezzi, UFRGS

#ifndef ALICEO2_MFT_TrackFitter_H_
#define ALICEO2_MFT_TrackFitter_H_

#include "MFTTracking/Cluster.h"
#include "MFTTracking/TrackCA.h"
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

  using SMatrix55Sym = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;
  using SMatrix5 = ROOT::Math::SVector<Double_t, 5>;

 public:
  TrackFitter() = default;
  ~TrackFitter() = default;

  TrackFitter(const TrackFitter&) = delete;
  TrackFitter& operator=(const TrackFitter&) = delete;
  TrackFitter(TrackFitter&&) = delete;
  TrackFitter& operator=(TrackFitter&&) = delete;

  void setBz(float bZ) { mBZField = bZ; }
  void setMFTRadLength(float MFT_x2X0) { mMFTDiskThicknessInX0 = MFT_x2X0 / 5.0; }
  void setVerbosity(float v) { mVerbose = v; }
  void setTrackModel(float m) { mTrackModel = m; }

  bool initTrack(TrackLTF& track, bool outward = false);
  bool fit(TrackLTF& track, bool outward = false);

  /// Return the maximum chi2 above which the track can be considered as abnormal
  static constexpr double getMaxChi2() { return SMaxChi2; }

 private:
  bool propagateToZ(TrackLTF& track, double z);
  bool propagateToNextClusterWithMCS(TrackLTF& track, double z);
  bool computeCluster(TrackLTF& track, int cluster);

  bool mFieldON = true;
  Float_t mBZField; // kiloGauss.
  Float_t mMFTDiskThicknessInX0 = 0.042 / 5;
  Int_t mTrackModel = MFTTrackModel::Optimized;

  static constexpr double SMaxChi2 = 2.e10; ///< maximum chi2 above which the track can be considered as abnormal
  bool mVerbose = false;
};

// Functions to estimate momentum and charge from track curvature
Double_t invQPtFromFCF(const TrackLTF& track, Double_t bFieldZ, Double_t& chi2);
Bool_t LinearRegression(Int_t nVal, Double_t* xVal, Double_t* yVal, Double_t* yErr, Double_t& a, Double_t& ae, Double_t& b, Double_t& be);

} // namespace mft
} // namespace o2

#endif // ALICEO2_MFT_TrackFitter_H_
