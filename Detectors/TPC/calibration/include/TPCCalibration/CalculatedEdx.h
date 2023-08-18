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

///
/// @file   CalculatedEdx.h
/// @author Tuba Gündem, tuba.gundem@cern.ch
///

#ifndef AliceO2_TPC_CalculatedEdx_H
#define AliceO2_TPC_CalculatedEdx_H

// o2 includes
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/dEdxInfo.h"
#include "GPUO2InterfaceRefit.h"
#include "CalibdEdxContainer.h"
#include "CorrectionMapsHelper.h"
#include "CommonUtils/TreeStreamRedirector.h"

#include <vector>

namespace o2::tpc
{

/// \brief dEdx calculation class
///
/// This class is used to calculate dEdx of reconstructed tracks.
/// Calibration objects are loaded from CCDB with the run number.
/// For the calculation of dEdx:
///   different corrections (track topology correction, gain map and residual dEdx correction) can be switched on and off
///   truncation range can be set for truncated mean calculation
///
/// How to use:
/// Example:
/// CalculatedEdx c{};
/// c.loadCalibsFromCCDB(runNumberOrTimeStamp);
/// start looping over the data
/// c.setMembers(tpcTrackClIdxVecInput, clusterIndex, tpcTracks); // set the member variables: TrackTPC, TPCClRefElem, o2::tpc::ClusterNativeAccess
/// c.setRefit(); // set the refit pointer to perform refitting of tracks, otherwise setPropagateTrack to true
/// start looping over the tracks
/// c.calculatedEdx(track, output, 0.01, 0.6, CorrectionFlags::TopologyPol | CorrectionFlags::GainFull | CorrectionFlags::GainResidual | CorrectionFlags::dEdxResidual) // this will fill the dEdxInfo output for given track

enum class CorrectionFlags : unsigned short {
  TopologySimple = 1 << 0, ///< flag for simple analytical topology correction
  TopologyPol = 1 << 1,    ///< flag for topology correction from polynomials
  GainFull = 1 << 2,       ///< flag for full gain map from calibration container
  GainResidual = 1 << 3,   ///< flag for residuals gain map from calibration container
  dEdxResidual = 1 << 4,   ///< flag for residual dEdx correction
};

inline CorrectionFlags operator&(CorrectionFlags a, CorrectionFlags b) { return static_cast<CorrectionFlags>(static_cast<unsigned short>(a) & static_cast<unsigned short>(b)); }
inline CorrectionFlags operator~(CorrectionFlags a) { return static_cast<CorrectionFlags>(~static_cast<unsigned short>(a)); }
inline CorrectionFlags operator|(CorrectionFlags a, CorrectionFlags b) { return static_cast<CorrectionFlags>(static_cast<unsigned short>(a) | static_cast<unsigned short>(b)); }

class CalculatedEdx
{
 public:
  CalculatedEdx();

  ~CalculatedEdx() = default;

  /// set the member variables
  /// \param tpcTrackClIdxVecInput TPCClRefElem member variable
  /// \param clIndex ClusterNativeAccess member variable
  /// \param vTPCTracksArrayInp vector of tpc tracks
  void setMembers(std::vector<o2::tpc::TPCClRefElem>* tpcTrackClIdxVecInput, const o2::tpc::ClusterNativeAccess& clIndex, std::vector<o2::tpc::TrackTPC>* vTPCTracksArrayInp);

  /// set the refitter
  void setRefit();

  /// \param propagate propagate the tracks to extract the track parameters instead of performing a refit
  void setPropagateTrack(const bool propagate) { mPropagateTrack = propagate; }

  /// \param debug use debug streamer and set debug vectors
  void setDebug(const bool debug) { mDebug = debug; }

  /// \param field magnetic field in kG, used for track propagation
  void setField(const float field) { mField = field; }

  /// \param maxMissingCl maximum number of missing clusters for subthreshold check
  void setMaxMissingCl(int maxMissingCl) { mMaxMissingCl = maxMissingCl; }

  /// set the debug streamer
  void setStreamer() { mStreamer = std::make_unique<o2::utils::TreeStreamRedirector>("dEdxDebug.root", "recreate"); };

  /// \return returns magnetic field in kG
  float getField() { return mField; }

  /// \return returns maxMissingCl for subthreshold cluster treatment
  int getMaxMissingCl() { return mMaxMissingCl; }

  /// fill missing clusters with minimum charge (method=0) or minimum charge/2 (method=1)
  void fillMissingClusters(int missingClusters[4], float minChargeTot, float minChargeMax, int method);

  /// get the truncated mean for the input track with the truncation range, charge type, region and corrections
  /// the cluster charge is normalized by effective length*gain, you can turn off the normalization by setting all corrections to false
  /// \param track input track
  /// \param output output dEdxInfo
  /// \param low lower cluster cut
  /// \param high higher cluster cut
  /// \param mask to apply different corrections: TopologySimple = simple analytical topology correction, TopologyPol = topology correction from polynomials, GainFull = full gain map from calibration container,
  ///                                                      GainResidual = residuals gain map from calibration container, dEdxResidual = residual dEdx correction
  void calculatedEdx(TrackTPC& track, dEdxInfo& output, float low = 0.05f, float high = 0.6f, CorrectionFlags mask = CorrectionFlags::TopologyPol | CorrectionFlags::GainFull | CorrectionFlags::GainResidual | CorrectionFlags::dEdxResidual);

  /// get the truncated mean for the input charge vector and the truncation range low*nCl<nCl<high*nCl
  /// \param charge input vector
  /// \param low lower cluster cut (e.g. 0.05)
  /// \param high higher cluster cut (e.g. 0.6)
  float getTruncMean(std::vector<float>& charge, float low, float high) const;

  /// get effective track length using simple analytical topology correction
  /// \param track input track
  /// \param region pad region
  /// \return returns simple analytical topology correction
  float getTrackTopologyCorrection(const o2::tpc::TrackTPC& track, const unsigned int region) const;

  /// get effective track length using topology correction from polynomials
  /// \param track input track
  /// \param cl cluster
  /// \param region pad region
  /// \param charge total or maximum charge of the cluster, cl
  /// \param chargeType total or maximum
  /// \param threshold zero supression threshold
  /// \return returns topology correction from polynomials
  float getTrackTopologyCorrectionPol(const o2::tpc::TrackTPC& track, const o2::tpc::ClusterNative& cl, const unsigned int region, const float charge, ChargeType chargeType, const float threshold) const;

  /// load calibration objects from CCDB
  /// \param runNumberOrTimeStamp run number or time stamp
  void loadCalibsFromCCDB(long runNumberOrTimeStamp);

 private:
  std::vector<TrackTPC>* mTracks{nullptr};                       ///< vector containing the tpc tracks which will be processed.
  std::vector<TPCClRefElem>* mTPCTrackClIdxVecInput{nullptr};    ///< input vector with TPC tracks cluster indicies
  const o2::tpc::ClusterNativeAccess* mClusterIndex{nullptr};    ///< needed to access clusternative with tpctracks
  o2::gpu::CorrectionMapsHelper mTPCCorrMapsHelper;              ///< cluster corrections map helper
  std::unique_ptr<o2::gpu::GPUO2InterfaceRefit> mRefit{nullptr}; ///< TPC refitter used for TPC tracks refit during the reconstruction

  int mMaxMissingCl{2};                                                ///< maximum number of missing clusters for subthreshold check
  float mField{5};                                                     ///< magnetic field in kG, used for track propagation
  bool mPropagateTrack{false};                                         ///< propagating the track instead of performing a refit
  bool mDebug{false};                                                  ///< use the debug streamer
  CalibdEdxContainer mCalibCont;                                       ///< calibration container
  std::unique_ptr<o2::utils::TreeStreamRedirector> mStreamer{nullptr}; ///< debug streamer

  std::array<std::vector<float>, 5> mChargeTotROC;
  std::array<std::vector<float>, 5> mChargeMaxROC;
};

} // namespace o2::tpc

#endif
