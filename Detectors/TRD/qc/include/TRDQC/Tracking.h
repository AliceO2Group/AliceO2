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
#include "DataFormatsTRD/CalibratedTracklet.h"
#include "DataFormatsTRD/Constants.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DetectorsBase/Propagator.h"
#include "TRDBase/RecoParam.h"
#include "TH1.h"

#include "Rtypes.h"

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
  int type;                          ///< 0 TPC-TRD track; 1 ITS-TPC-TRD track
  GTrackID refGlobalTrackId;         ///< GlobalTrackID of the seeding track (either ITS-TPC or TPC)
  int nTracklets;                    ///< number of attached TRD tracklets
  int nLayers;                       //< Number of Layers of a Track in which the track extrapolation was in geometrical acceptance of the TRD
  float chi2;                        ///< total chi2 value for the track
  float reducedChi2;                 ///< chi2 total divided by number of layers in which track is inside TRD geometrical acceptance
  float p;                           ///< the total momentum of the track at the point of the innermost ITS cluster (ITS-TPC-TRD) or at the inner TPC radius (TPC-TRD)
  float pt;                          ///< the transverse momentum of the track at the point of the innermost ITS cluster (ITS-TPC-TRD) or at the inner TPC radius (TPC-TRD)
  float ptSigma2;                    ///< Sigma2 of pt
  float dEdxTotTPC;                  ///< raw total dEdx information for seeding track in TPC
  std::bitset<6> isCrossingNeighbor; ///< indicate if track crossed a padrow and/or had a neighboring tracklet in that layer
  bool hasNeighbor;                  ///< indicate if a track had a tracklet with a neighboring one e.g. potentailly split tracklet
  bool hasPadrowCrossing;            ///< indicate if track crossed a padrow

  // layer-wise information for seeding track and assigned tracklet (if available)
  std::array<bool, constants::NLAYER> findable{};  ///< flag if track was in geometrical acceptance
  std::array<float, constants::NLAYER> trackX{};   ///< x-position of seeding track (sector coordinates)
  std::array<float, constants::NLAYER> trackY{};   ///< y-position of seeding track (sector coordinates)
  std::array<float, constants::NLAYER> trackZ{};   ///< z-position of seeding track (sector coordinates)
  std::array<float, constants::NLAYER> trackSnp{}; ///< sin(phi) of seeding track (sector coordinates -> local inclination in r-phi)
  std::array<float, constants::NLAYER> trackTgl{}; ///< tan(lambda) of seeding track (inclination in s_xy-z plane)
  std::array<float, constants::NLAYER> trackQpt{}; ///< q/pt of seeding track
  std::array<float, constants::NLAYER> trackPhi{}; //< Phi 0:2Pi value of Track
  std::array<float, constants::NLAYER> trackEta{}; //< Eta value of Track

  // tracklet position is also given in sector coordinates
  std::array<float, constants::NLAYER> trackletYraw{};         ///< y-position of tracklet without tilt correction
  std::array<float, constants::NLAYER> trackletZraw{};         ///< z-position of tracklet without tilt correction
  std::array<float, constants::NLAYER> trackletY{};            ///< y-position of tracklet used for track update (including correction)
  std::array<float, constants::NLAYER> trackletZ{};            ///< z-position of tracklet used for track update (including correction)
  std::array<float, constants::NLAYER> trackletDy{};           ///< tracklet deflection over drift length obtained from CalibratedTracklet
  std::array<int, constants::NLAYER> trackletSlope{};          ///< the raw slope from Tracklet64
  std::array<int, constants::NLAYER> trackletSlopeSigned{};    ///< the raw slope from Tracklet64 (signed integer)
  std::array<int, constants::NLAYER> trackletPosition{};       ///< the raw position from Tracklet64
  std::array<int, constants::NLAYER> trackletPositionSigned{}; ///< the raw position from Tracklet64 (signed integer)
  std::array<int, constants::NLAYER> trackletDet{};            ///< the chamber of the tracklet
  // some tracklet details to identify its global MCM number to check if it is from noisy MCM
  std::array<int, constants::NLAYER> trackletHCId{};                                     ///< the half-chamber ID of the tracklet
  std::array<int, constants::NLAYER> trackletRob{};                                      ///< the ROB number of the tracklet
  std::array<int, constants::NLAYER> trackletMcm{};                                      ///< the MCM number of the tracklet
  std::array<float, constants::NLAYER> trackletChi2{};                                   ///< estimated chi2 for the update of the track with the given tracklet
  std::array<std::array<int, constants::NCHARGES>, constants::NLAYER> trackletCharges{}; ///< charges of tracklets
  ClassDefNV(TrackQC, 4);
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

  // Make output accessible to DPL processor
  std::vector<TrackQC>& getTrackQC() { return mTrackQC; }

 private:
  float mMaxSnp{o2::base::Propagator::MAX_SIN_PHI};  ///< max snp when propagating tracks
  float mMaxStep{o2::base::Propagator::MAX_STEP};    ///< maximum step for propagation
  MatCorrType mMatCorr{MatCorrType::USEMatCorrNONE}; ///< if material correction should be done
  RecoParam mRecoParam;                              ///< parameters required for TRD reconstruction
  // QA results
  std::vector<TrackQC> mTrackQC;
  // input from DPL
  gsl::span<const o2::dataformats::TrackTPCITS> mTracksITSTPC; ///< ITS-TPC seeding tracks
  gsl::span<const o2::tpc::TrackTPC> mTracksTPC;               ///< TPC seeding tracks
  gsl::span<const TrackTRD> mTracksITSTPCTRD;                  ///< TRD tracks reconstructed from TPC or ITS-TPC seeds
  gsl::span<const TrackTRD> mTracksTPCTRD;                     ///< TRD tracks reconstructed from TPC or TPC seeds
  gsl::span<const Tracklet64> mTrackletsRaw;                   ///< array of raw tracklets needed for TRD refit
  gsl::span<const CalibratedTracklet> mTrackletsCalib;         ///< array of calibrated tracklets needed for TRD refit

  ClassDefNV(Tracking, 1);
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_TRACKINGQC_H
