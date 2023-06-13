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
#include "GPUO2InterfaceRefit.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "TPCCalibration/CalibPadGainTracksBase.h"
#include "CalibdEdxTrackTopologyPol.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "CalibdEdxContainer.h"
#include "CorrectionMapsHelper.h"
#include "CommonUtils/TreeStreamRedirector.h"

#include <vector>

namespace o2::tpc
{

/// \brief dEdx calculation class
///
/// This class is used to calculate dEdx of reconstructed tracks.
/// Calibration objects are loaded from CCDB with a timestamp.
/// For the calculation of dEdx:
///   charge type and the region can be set
///   different corrections (track topology correction, gain map and residual dEdx correction) can be switched on and off
///   truncation range can be set for truncated mean calculation
///
/// How to use:
/// Example:
/// CalculatedEdx calc{};
/// calc.loadCalibsFromCCDB(timeStamp);
/// start looping over the data
/// calc.setMembers(tpcTrackClIdxVecInput, clusterIndex, tpcTracks); // set the member variables: TrackTPC, TPCClRefElem, o2::tpc::ClusterNativeAccess
/// calc.setRefit(); // set the refit pointer to perform refitting of tracks, otherwise setPropagateTrack to true
/// start looping over the tracks
/// calc.getTruncMean(track, 0.01, 0.6, ChargeType::Tot, CalculatedEdx::RegionType::entire, 0b11111) // this will return the calculated dEdx value for the given track

class CalculatedEdx
{
 public:
  enum RegionType : unsigned char {
    entire,
    iroc,
    oroc1,
    oroc2,
    oroc3
  };

  /// default constructor
  CalculatedEdx()
  {
    mTPCCorrMapsHelper.setOwner(true);
    mTPCCorrMapsHelper.setCorrMap(TPCFastTransformHelperO2::instance()->create(0));
  }

  /// default destructor
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

  /// \param chargeType type of charge which is used for the dE/dx calculations
  void setChargeType(const ChargeType chargeType) { mChargeType = chargeType; }

  /// \param maxMissingCl maximum number of missing clusters for subthreshold check
  void setMaxMissingCl(int maxMissingCl) { mMaxMissingCl = maxMissingCl; }

  /// set the debug streamer
  void setStreamer() { mStreamer = std::make_unique<o2::utils::TreeStreamRedirector>("dEdxDebug.root", "recreate"); };

  /// fill missing clusters with minimum charge (method=0) or minimum charge/2 (method=1)
  void fillMissingClusters(std::vector<float>& charge, int missingClusters, float minCharge, int method);

  /// get the truncated mean for the input track with the truncation range, charge type, region and corrections
  /// the cluster charge is normalized by effective length*gain, you can turn off the normalization by setting all corrections to false
  /// \param track input track
  /// \param low lower cluster cut
  /// \param high higher cluster cut
  /// \param chargeType cluster charge type that will be used for truncation, options: ChargeType::Tot, ChargeType::Max
  /// \param regionType region that will be used for truncation, options: RegionType::entire, RegionType::iroc, RegionType::oroc1, RegionType::oroc2, RegionType::oroc3
  /// \param mask bit mask to apply different corrections: 1 bit = simple analytical topology correction, 2 bit = topology correction from polynomials, 3 = full gain map from calibration container,
  ///                                                      4 bit = residuals gain map from calibration container, 5 bit = residual dEdx correction
  float getTruncMean(TrackTPC& track, float low = 0.05f, float high = 0.6f, ChargeType chargeType = ChargeType::Tot, RegionType regionType = RegionType::entire, unsigned short mask = 0b11110);

  /// get the truncated mean for the input charge vector and the truncation range low*nCl<nCl<high*nCl
  /// \param charge input vector
  /// \param low lower cluster cut (e.g. 0.05)
  /// \param high higher cluster cut (e.g. 0.6)
  float getTruncMean(std::vector<float>& charge, float low, float high);

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
  /// \param threshold zero supression threshold
  /// \return returns topology correction from polynomials
  float getTrackTopologyCorrectionPol(const o2::tpc::TrackTPC& track, const o2::tpc::ClusterNative& cl, const unsigned int region, const float charge, const float threshold) const;

  /// load calibration objects from CCDB
  /// \param timestampCCDB timestamp
  void loadCalibsFromCCDB(long timestampCCDB = 0);

 private:
  std::vector<TrackTPC>* mTracks{nullptr};                       ///< vector containing the tpc tracks which will be processed.
  std::vector<TPCClRefElem>* mTPCTrackClIdxVecInput{nullptr};    ///< input vector with TPC tracks cluster indicies
  const o2::tpc::ClusterNativeAccess* mClusterIndex{nullptr};    ///< needed to access clusternative with tpctracks
  o2::gpu::CorrectionMapsHelper mTPCCorrMapsHelper;              ///< cluster corrections map helper
  std::unique_ptr<o2::gpu::GPUO2InterfaceRefit> mRefit{nullptr}; ///< TPC refitter used for TPC tracks refit during the reconstruction

  int mMaxMissingCl{2};                    ///< maximum number of missing clusters for subthreshold check
  float mField{-5};                        ///< magnetic field in kG, used for track propagation
  bool mPropagateTrack{false};             ///< propagating the track instead of performing a refit
  bool mDebug{false};                      ///< use the debug streamer
  ChargeType mChargeType{ChargeType::Tot}; ///< charge type to calculate truncated mean

  std::vector<float> mChargeTmp; ///< memory for truncated mean calculation
  CalibdEdxContainer mCalibCont; ///< calibration container

  std::unique_ptr<o2::utils::TreeStreamRedirector> mStreamer{nullptr}; ///< debug streamer
  std::vector<int> regionVector;                                       ///< debug streamer vector for region
  std::vector<unsigned char> rowIndexVector;                           ///< debug streamer vector for row index
  std::vector<unsigned char> padVector;                                ///< debug streamer vector for pad
  std::vector<int> stackVector;                                        ///< debug streamer vector for stack
  std::vector<unsigned char> sectorVector;                             ///< debug streamer vector for sector

  std::vector<float> effectiveLengthVector; ///< debug streamer vector for efective length
  std::vector<float> gainVector;            ///< debug streamer vector for gain
  std::vector<float> residualCorrVector;    ///< debug streamer vector for residual dEdx correction

  std::vector<float> chargeVector;     ///< debug streamer vector for charge
  std::vector<float> chargeNormVector; ///< debug streamer vector for normalized charge with effectiveLength, gain and residual dEdx correction
};
} // namespace o2::tpc

#endif
