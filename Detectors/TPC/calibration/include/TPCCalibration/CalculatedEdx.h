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
/// c.calculatedEdx(track, output, 0.015, 0.60, CorrectionFlags::TopologyPol | CorrectionFlags::dEdxResidual, ClusterFlags::ExcludeEdgeCl) // this will fill the dEdxInfo output for given track

enum class CorrectionFlags : unsigned short {
  None = 0,
  TopologySimple = 1 << 0, ///< flag for simple analytical topology correction
  TopologyPol = 1 << 1,    ///< flag for topology correction from polynomials
  GainFull = 1 << 2,       ///< flag for full gain map from calibration container
  GainResidual = 1 << 3,   ///< flag for residuals gain map from calibration container
  dEdxResidual = 1 << 4,   ///< flag for residual dEdx correction
};

enum class ClusterFlags : unsigned short {
  None = 0,
  ExcludeSingleCl = 1 << 0,         ///< flag to exclude single clusters in dEdx calculation
  ExcludeSplitCl = 1 << 1,          ///< flag to exclude split clusters in dEdx calculation
  ExcludeEdgeCl = 1 << 2,           ///< flag to exclude sector edge clusters in dEdx calculation
  ExcludeSubthresholdCl = 1 << 3,   ///< flag to exclude subthreshold clusters in dEdx calculation
  ExcludeSectorBoundaries = 1 << 4, ///< flag to exclude sector boundary clusters in subthreshold cluster treatment
};

inline CorrectionFlags operator&(CorrectionFlags a, CorrectionFlags b) { return static_cast<CorrectionFlags>(static_cast<unsigned short>(a) & static_cast<unsigned short>(b)); }
inline CorrectionFlags operator~(CorrectionFlags a) { return static_cast<CorrectionFlags>(~static_cast<unsigned short>(a)); }
inline CorrectionFlags operator|(CorrectionFlags a, CorrectionFlags b) { return static_cast<CorrectionFlags>(static_cast<unsigned short>(a) | static_cast<unsigned short>(b)); }

inline ClusterFlags operator&(ClusterFlags a, ClusterFlags b) { return static_cast<ClusterFlags>(static_cast<unsigned short>(a) & static_cast<unsigned short>(b)); }
inline ClusterFlags operator~(ClusterFlags a) { return static_cast<ClusterFlags>(~static_cast<unsigned short>(a)); }
inline ClusterFlags operator|(ClusterFlags a, ClusterFlags b) { return static_cast<ClusterFlags>(static_cast<unsigned short>(a) | static_cast<unsigned short>(b)); }

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
  void setFieldNominalGPUBz(const float field) { mFieldNominalGPUBz = field; }

  /// \param maxMissingCl maximum number of missing clusters for subthreshold check
  void setMaxMissingCl(int maxMissingCl) { mMaxMissingCl = maxMissingCl; }

  /// \param minChargeTotThreshold upper limit for the possible minimum charge tot in subthreshold treatment
  void setMinChargeTotThreshold(float minChargeTotThreshold) { mMinChargeTotThreshold = minChargeTotThreshold; }

  /// \param minChargeMaxThreshold upper limit for the possible minimum charge max in subthreshold treatment
  void setMinChargeMaxThreshold(float minChargeMaxThreshold) { mMinChargeMaxThreshold = minChargeMaxThreshold; }

  /// set the debug streamer
  void setStreamer(const char* debugRootFile) { mStreamer = std::make_unique<o2::utils::TreeStreamRedirector>(debugRootFile, "recreate"); };

  /// \return returns magnetic field in kG
  float getFieldNominalGPUBz() { return mFieldNominalGPUBz; }

  /// \return returns maxMissingCl for subthreshold cluster treatment
  int getMaxMissingCl() { return mMaxMissingCl; }

  /// \return returns the upper limit for the possible minimum charge tot in subthreshold treatment
  float getMinChargeTotThreshold() { return mMinChargeTotThreshold; }

  /// \return returns the upper limit for the possible minimum charge max in subthreshold treatment
  float getMinChargeMaxThreshold() { return mMinChargeMaxThreshold; }

  /// fill missing clusters with minimum charge (method=0) or minimum charge/2 (method=1) or Landau (method=2)
  void fillMissingClusters(int missingClusters[4], float minChargeTot, float minChargeMax, int method);

  /// get the truncated mean for the input track with the truncation range, charge type, region and corrections
  /// the cluster charge is normalized by effective length*gain, you can turn off the normalization by setting all corrections to false
  /// \param track input track
  /// \param output output dEdxInfo
  /// \param low lower cluster cut
  /// \param high higher cluster cut
  /// \param mask to apply different corrections: TopologySimple = simple analytical topology correction, TopologyPol = topology correction from polynomials, GainFull = full gain map from calibration container,
  ///                                                      GainResidual = residuals gain map from calibration container, dEdxResidual = residual dEdx correction
  void calculatedEdx(TrackTPC& track, dEdxInfo& output, float low = 0.015f, float high = 0.6f, CorrectionFlags correctionMask = CorrectionFlags::TopologyPol | CorrectionFlags::dEdxResidual, ClusterFlags clusterMask = ClusterFlags::None, int subthresholdMethod = 0, const char* debugRootFile = "dEdxDebug.root");

  /// get the truncated mean for the input charge vector and the truncation range low*nCl<nCl<high*nCl
  /// \param charge input vector
  /// \param low lower cluster cut (e.g. 0.015)
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

  /// load calibration objects from local CCDB folder
  /// \param localCCDBFolder local CCDB folder
  void loadCalibsFromLocalCCDBFolder(const char* localCCDBFolder);

  /// load track topology correction from a local file
  /// \param folder folder path without a trailing /
  /// \param file file path starting with /
  /// \param object name of the object to load
  void setTrackTopologyCorrectionFromFile(const char* folder, const char* file, const char* object);

  /// load gain map from a local file
  /// \param folder folder path without a trailing /
  /// \param file file path starting with /
  /// \param object name of the object to load
  void setGainMapFromFile(const char* folder, const char* file, const char* object);

  /// load gain map residual from a local file
  /// \param folder folder path without a trailing /
  /// \param file file path starting with /
  /// \param object name of the object to load
  void setGainMapResidualFromFile(const char* folder, const char* file, const char* object);

  /// load dEdx residual correction from a local file
  /// \param folder folder path without a trailing /
  /// \param file file path starting with /
  /// \param object name of the object to load
  void setResidualCorrectionFromFile(const char* folder, const char* file, const char* object);

  /// load zero suppression threshold from a local file
  /// \param folder folder path without a trailing /
  /// \param file file path starting with /
  /// \param object name of the object to load
  void setZeroSuppressionThresholdFromFile(const char* folder, const char* file, const char* object);

  /// load magnetic field from a local file
  /// \param folder folder path without a trailing /
  /// \param file file path starting with /
  /// \param object name of the object to load
  void setMagneticFieldFromFile(const char* folder, const char* file, const char* object);

  /// load propagator from a local file
  /// \param folder folder path without a trailing /
  /// \param file file path starting with /
  /// \param object name of the object to load
  void setPropagatorFromFile(const char* folder, const char* file, const char* object);

 private:
  std::vector<TrackTPC>* mTracks{nullptr};                       ///< vector containing the tpc tracks which will be processed
  std::vector<TPCClRefElem>* mTPCTrackClIdxVecInput{nullptr};    ///< input vector with TPC tracks cluster indicies
  const o2::tpc::ClusterNativeAccess* mClusterIndex{nullptr};    ///< needed to access clusternative with tpctracks
  o2::gpu::CorrectionMapsHelper mTPCCorrMapsHelper;              ///< cluster correction maps helper
  std::vector<unsigned char> mTPCRefitterShMap;                  ///< externally set TPC clusters sharing map
  std::vector<unsigned int> mTPCRefitterOccMap;                  ///< externally set TPC clusters occupancy map
  std::unique_ptr<o2::gpu::GPUO2InterfaceRefit> mRefit{nullptr}; ///< TPC refitter used for TPC tracks refit during the reconstruction

  int mMaxMissingCl{1};                                                ///< maximum number of missing clusters for subthreshold check
  float mMinChargeTotThreshold{50};                                    ///< upper limit for minimum charge tot value in subthreshold treatment, i.e for a high dEdx track adding a minimum value of 500 to track as a virtual charge doesn't make sense
  float mMinChargeMaxThreshold{50};                                    ///< upper limit for minimum charge max value in subthreshold treatment, i.e for a high dEdx track adding a minimum value of 500 to track as a virtual charge doesn't make sense
  float mFieldNominalGPUBz{5};                                         ///< magnetic field in kG, used for track propagation
  bool mPropagateTrack{false};                                         ///< propagating the track instead of performing a refit
  bool mDebug{false};                                                  ///< use the debug streamer
  CalibdEdxContainer mCalibCont;                                       ///< calibration container
  std::unique_ptr<o2::utils::TreeStreamRedirector> mStreamer{nullptr}; ///< debug streamer

  std::array<std::vector<float>, 5> mChargeTotROC;
  std::array<std::vector<float>, 5> mChargeMaxROC;
};

} // namespace o2::tpc

#endif