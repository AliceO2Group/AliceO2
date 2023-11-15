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

/// \file Tracking.h
/// \brief Check the performance of the TRD in global tracking
/// \author Ole Schmidt

#ifndef O2_TRD_TRACKINGQC_H
#define O2_TRD_TRACKINGQC_H

#include "DataFormatsTRD/TrackTRD.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "TRDBase/PadCalibrationsAliases.h"
#include "DataFormatsTRD/CalibratedTracklet.h"
#include "DataFormatsTRD/Constants.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DetectorsBase/Propagator.h"
#include "TRDBase/RecoParam.h"

#include "Rtypes.h"
#include "TH1.h"

#include <gsl/span>
#include <bitset>

using GTrackID = o2::dataformats::GlobalTrackID;

namespace o2
{

namespace globaltracking
{
class RecoContainer;
}

namespace trd
{

struct TrackQC {

  GTrackID refGlobalTrackId;         ///< GlobalTrackID of the seeding track (either ITS-TPC or TPC)
  TrackTRD trackTRD;                 ///< the found TRD track
  o2::track::TrackParCov trackSeed;  ///< outer param of the seeding track
  float dEdxTotTPC;                  ///< raw total dEdx information for seeding track in TPC

  std::array<o2::track::TrackPar, constants::NLAYER> trackProp{}; ///< the track parameters stored at the radius where the track is updated with TRD info
  std::array<Tracklet64, constants::NLAYER> trklt64{};            ///< the raw tracklet used for the update (includes uncorrected charges)
  std::array<CalibratedTracklet, constants::NLAYER> trkltCalib{}; ///< the TRD space point used for the update (not yet tilt-corrected and z-shift corrected)

  std::array<float, constants::NLAYER> trackletY{};            ///< y-position of tracklet used for track update (including correction)
  std::array<float, constants::NLAYER> trackletZ{};            ///< z-position of tracklet used for track update (including correction)
  std::array<float, constants::NLAYER> trackletChi2{};         ///< estimated chi2 for the update of the track with the given tracklet
  std::array<std::array<float, constants::NCHARGES>, constants::NLAYER> trackletCorCharges{}; ///< corrected charges of tracklets

  ClassDefNV(TrackQC, 6);
};

class Tracking
{
  using MatCorrType = o2::base::Propagator::MatCorrType;

 public:
  Tracking() = default;
  Tracking(const Tracking&) = delete;
  ~Tracking() = default;

  /// Load geometry and apply magnetic field setting
  void init();

  /// Initialize the input arrays
  void setInput(const o2::globaltracking::RecoContainer& input);

  /// Main processing function
  void run();

  /// Reset the output vector
  void reset() { mTrackQC.clear(); }

  /// Check track QC
  void checkTrack(const TrackTRD& trk, bool isTPCTRD);

  /// Disable TPC dEdx information
  void disablePID() { mPID = false; }

  // Make output accessible to DPL processor
  std::vector<TrackQC>& getTrackQC() { return mTrackQC; }

  // Set the local gain factors with values from the ccdb
  void setLocalGainFactors(const o2::trd::LocalGainFactor& localGain)
  {
    mLocalGain = localGain;
  }

 private:
  float mMaxSnp{o2::base::Propagator::MAX_SIN_PHI};  ///< max snp when propagating tracks
  float mMaxStep{o2::base::Propagator::MAX_STEP};    ///< maximum step for propagation
  MatCorrType mMatCorr{MatCorrType::USEMatCorrNONE}; ///< if material correction should be done
  RecoParam mRecoParam;                              ///< parameters required for TRD reconstruction
  bool mPID{true};                                   ///< if TPC only tracks are not available we don't fill PID info

  // QA results
  std::vector<TrackQC> mTrackQC;

  // input from DPL
  gsl::span<const o2::dataformats::TrackTPCITS> mTracksITSTPC; ///< ITS-TPC seeding tracks
  gsl::span<const o2::tpc::TrackTPC> mTracksTPC;               ///< TPC seeding tracks
  gsl::span<const TrackTRD> mTracksITSTPCTRD;                  ///< TRD tracks reconstructed from TPC or ITS-TPC seeds
  gsl::span<const TrackTRD> mTracksTPCTRD;                     ///< TRD tracks reconstructed from TPC or TPC seeds
  gsl::span<const Tracklet64> mTrackletsRaw;                   ///< array of raw tracklets needed for TRD refit
  gsl::span<const CalibratedTracklet> mTrackletsCalib;         ///< array of calibrated tracklets needed for TRD refit

  // corrections from ccdb, some need to be loaded only once hence an init flag
  o2::trd::LocalGainFactor mLocalGain; ///< local gain factors from krypton calibration

  ClassDefNV(Tracking, 2);
};

} // namespace trd

namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::trd::TrackQC> : std::true_type {
};
} // namespace framework

} // namespace o2

#endif // O2_TRD_TRACKINGQC_H
