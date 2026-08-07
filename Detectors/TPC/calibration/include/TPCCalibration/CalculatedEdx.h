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
#include "TPCBase/Mapper.h"
#include "GPUO2InterfaceRefit.h"
#include "CalibdEdxContainer.h"
#include "CorrectionMapsHelper.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "TPCCalibration/CorrectdEdxDistortions.h"
#include "TPCFastTransformPOD.h"
#include "GPUCommonRtypes.h"
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <utility>

namespace o2::tpc
{

/// \brief average cluster occupancy of a track, per TPC region
struct AverageOccupancy {
  double IROC = 0.;
  double OROC1 = 0.;
  double OROC2 = 0.;
  double OROC3 = 0.;
  ClassDefNV(AverageOccupancy, 1);
};

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
/// c.calculatedEdx(track, output, averageOcc, 0.015, 0.60, CorrectionFlags::TopologyPol | CorrectionFlags::dEdxResidual, ClusterFlags::ExcludeEdgeCl) // this will fill the dEdxInfo output and per-region average track occupancy averageOcc for given track

enum class CorrectionFlags : unsigned short {
  None = 0,
  TopologySimple = 1 << 0, ///< flag for simple analytical topology correction
  TopologyPol = 1 << 1,    ///< flag for topology correction from polynomials
  GainFull = 1 << 2,       ///< flag for full gain map from calibration container
  GainResidual = 1 << 3,   ///< flag for residuals gain map from calibration container
  dEdxResidual = 1 << 4,   ///< flag for residual dEdx correction
  dEdxSC = 1 << 5,         ///< flag for space-charge dEdx correction
};

enum class ClusterFlags : unsigned short {
  None = 0,
  ExcludeSingleCl = 1 << 0,         ///< flag to exclude single clusters in dEdx calculation
  ExcludeSplitPadCl = 1 << 1,       ///< flag to exclude split pad clusters in dEdx calculation
  ExcludeSplitTimeCl = 1 << 2,      ///< flag to exclude split time clusters in dEdx calculation
  ExcludeSplitCl = 1 << 3,          ///< flag to exclude split pad or time clusters in dEdx calculation
  ExcludeEdgeCl = 1 << 4,           ///< flag to exclude sector edge clusters in dEdx calculation
  ExcludeSubthresholdCl = 1 << 5,   ///< flag to exclude subthreshold clusters in dEdx calculation
  ExcludeSectorBoundaries = 1 << 6, ///< flag to exclude sector boundary clusters in subthreshold cluster treatment
  ExcludeSharedCl = 1 << 7,         ///< flag to exclude clusters shared between tracks in dEdx calculation
  ExcludeSamePadRowCl = 1 << 8,     ///< flag to exclude clusters in the same pad row in dEdx calculation
};

inline CorrectionFlags operator&(CorrectionFlags a, CorrectionFlags b) { return static_cast<CorrectionFlags>(static_cast<unsigned short>(a) & static_cast<unsigned short>(b)); }
inline CorrectionFlags operator~(CorrectionFlags a) { return static_cast<CorrectionFlags>(~static_cast<unsigned short>(a)); }
inline CorrectionFlags operator|(CorrectionFlags a, CorrectionFlags b) { return static_cast<CorrectionFlags>(static_cast<unsigned short>(a) | static_cast<unsigned short>(b)); }

inline ClusterFlags operator&(ClusterFlags a, ClusterFlags b) { return static_cast<ClusterFlags>(static_cast<unsigned short>(a) & static_cast<unsigned short>(b)); }
inline ClusterFlags operator~(ClusterFlags a) { return static_cast<ClusterFlags>(~static_cast<unsigned short>(a)); }
inline ClusterFlags operator|(ClusterFlags a, ClusterFlags b) { return static_cast<ClusterFlags>(static_cast<unsigned short>(a) | static_cast<unsigned short>(b)); }

/// \brief bundles the settings of one calculatedEdx() call (everything except the track/output/averageOcc)
/// used by calculatedEdxMultipleSettings() to evaluate several settings for the same track without repeating
/// the track refit/propagation for every setting
struct dEdxSettings {
  float low = 0.015f;                                                                            ///< lower cluster cut
  float high = 0.6f;                                                                             ///< higher cluster cut
  CorrectionFlags correctionMask = CorrectionFlags::TopologyPol | CorrectionFlags::dEdxResidual; ///< corrections to apply
  ClusterFlags clusterMask = ClusterFlags::None;                                                 ///< clusters to exclude
  int subthresholdMethod = 0;                                                                    ///< subthreshold cluster charge filling method
  int stackBoundaryMethod = 0;                                                                   ///< stack boundary cluster exclusion method
  std::string debugRootFile = "dEdxDebug.root";                                                  ///< debug streamer output file used if mDebug is set
  float maxSubthresholdChargeTot = 100000.f;                                                     ///< upper limit for the per-region minimum qTot used as the virtual charge of a subthreshold cluster (default effectively disables the cap)
  float maxSubthresholdChargeMax = 100000.f;                                                     ///< upper limit for the per-region minimum qMax used as the virtual charge of a subthreshold cluster (default effectively disables the cap)
};

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
  void setRefit(const unsigned int nHbfPerTf = 32);

  /// \param propagate propagate the tracks to extract the track parameters instead of performing a refit
  void setPropagateTrack(const bool propagate) { mPropagateTrack = propagate; }

  /// \param propagate propagate the tracks to extract the track parameters instead of performing a refit
  void setPropagateParams(const bool propagate) { mPropagateParams = propagate; }

  /// \param debug use debug streamer and set debug vectors
  void setDebug(const bool debug) { mDebug = debug; }

  /// \param field magnetic field in kG, used for track propagation
  void setFieldNominalGPUBz(const float field) { mFieldNominalGPUBz = field; }

  /// \param maxMissingCl maximum number of missing clusters for subthreshold check
  void setMaxMissingCl(int maxMissingCl) { mMaxMissingCl = maxMissingCl; }

  /// set the debug streamer for a given output file; a new streamer is only created the first time a given debugRootFile is seen,
  /// so different calculatedEdx() calls using different debugRootFile names each get their own independent debug file
  void setStreamer(const char* debugRootFile)
  {
    auto& streamer = mStreamers[debugRootFile];
    if (!streamer) {
      streamer = std::make_unique<o2::utils::TreeStreamRedirector>(debugRootFile, "recreate");
    }
  };

  /// set the debug streamer of the space-charge dedx correction
  void setSCStreamer(const char* debugRootFile = "debug_sc_corrections.root") { mSCdEdxCorrection.setStreamer(debugRootFile); }

  /// \param lumi set luminosity for space-charge correction map scaling
  void setLumi(const float lumi) { mSCdEdxCorrection.setLumi(lumi); }

  /// \return returns magnetic field in kG
  float getFieldNominalGPUBz() { return mFieldNominalGPUBz; }

  /// \return returns maxMissingCl for subthreshold cluster treatment
  int getMaxMissingCl() { return mMaxMissingCl; }

  /// \return returns the number of rows where refit/propagation failed (row.propagationFailed) since the last resetDebugCounters()
  long getNPropagationFailed() const { return mNPropagationFailed; }

  /// \return returns the number of rows gathered by gatherRowClusterData() (processed for refit/propagation) since the last resetDebugCounters()
  long getNRowsProcessed() const { return mNRowsProcessed; }

  /// \return returns the number of row gaps filled as subthreshold clusters by calculatedEdxFromRowData() since the last resetDebugCounters() per setting
  const std::vector<long>& getNSubThresholdFilledPerSettings() const { return mNSubThresholdFilledPerSettings; }

  /// reset the running counters returned by getNPropagationFailed()/getNRowsProcessed()/getNSubThresholdFilledPerSettings()
  void resetDebugCounters()
  {
    mNPropagationFailed = 0;
    mNRowsProcessed = 0;
    mNSubThresholdFilledPerSettings.clear();
  }

  /// fill missing clusters per region with that region's running minimum charge (method=0) or half of it (method=1),
  /// \param missingClusters number of row gaps to fill, per region (IROC, OROC1, OROC2, OROC3)
  /// \param minChargeTot per-region running minimum qTot among the accepted clusters of that region
  /// \param minChargeMax per-region running minimum qMax among the accepted clusters of that region
  void fillMissingClusters(int missingClusters[4], const float minChargeTot[4], const float minChargeMax[4], int method, std::array<std::vector<float>, 5>& chargeTotROC, std::array<std::vector<float>, 5>& chargeMaxROC);

  /// \param rowOrder (sector, row) keys in the order they are first encountered while scanning the track's native cluster references (0..nClusterReferences-1), i.e. the track's true physical row-traversal order
  void handleSameRowClusters(o2::tpc::TrackTPC& track, std::vector<std::pair<unsigned char, unsigned char>>& rowOrder, std::map<std::pair<unsigned char, unsigned char>, std::vector<int>>& clustersByRow, std::map<std::pair<unsigned char, unsigned char>, o2::tpc::ClusterNative>& combinedClustersByRow, std::map<int, std::tuple<unsigned char, unsigned char, unsigned int>>& clusterReferencesByIndex);

  /// get the truncated mean for the input track with the truncation range, charge type, region and corrections
  /// the cluster charge is normalized by effective length*gain, you can turn off the normalization by setting all corrections to false
  /// \param track input track
  /// \param output output dEdxInfo
  /// \param averageOcc output average cluster occupancy of the track, per TPC region
  /// \param low lower cluster cut
  /// \param high higher cluster cut
  /// \param correctionMask to apply different corrections: TopologySimple = simple analytical topology correction, TopologyPol = topology correction from polynomials, GainFull = full gain map from calibration container,
  ///                                                      GainResidual = residuals gain map from calibration container, dEdxResidual = residual dEdx correction
  /// \param maxSubthresholdChargeTot upper limit for the per-region minimum qTot used as the virtual charge of a subthreshold cluster
  /// \param maxSubthresholdChargeMax upper limit for the per-region minimum qMax used as the virtual charge of a subthreshold cluster
  void calculatedEdx(TrackTPC& track, dEdxInfo& output, AverageOccupancy& averageOcc, float low = 0.015f, float high = 0.6f, CorrectionFlags correctionMask = CorrectionFlags::TopologyPol | CorrectionFlags::dEdxResidual, ClusterFlags clusterMask = ClusterFlags::None, int subthresholdMethod = 0, int stackBoundaryMethod = 0, const char* debugRootFile = "dEdxDebug.root", float maxSubthresholdChargeTot = 100000.f, float maxSubthresholdChargeMax = 100000.f);

  /// evaluate several dEdx settings for the same track while performing the track refit/propagation to each cluster row only once
  /// \param track input track
  /// \param outputs output dEdxInfo, filled with one entry per entry in settingsList, in the same order
  /// \param averageOcc output average cluster occupancy of the track, per TPC region; a single value, since occupancy does not depend on the dEdx settings and is therefore the same for every entry in settingsList
  /// \param settingsList list of dEdx settings to evaluate for this track
  void calculatedEdxMultipleSettings(TrackTPC& track, std::vector<dEdxInfo>& outputs, AverageOccupancy& averageOcc, const std::vector<dEdxSettings>& settingsList);

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
  /// \param clPad cluster pad
  /// \param clTime cluster time
  /// \param region pad region
  /// \param charge total or maximum charge of the cluster, cl
  /// \param chargeType total or maximum
  /// \param threshold zero supression threshold
  /// \return returns topology correction from polynomials
  float getTrackTopologyCorrectionPol(const o2::tpc::TrackTPC& track, const o2::tpc::ClusterNative& cl, const unsigned int region, const float charge, ChargeType chargeType, const float threshold) const;

  /// \return returns space-charge dedx correctin
  auto& getSCCorrection() { return mSCdEdxCorrection; }

  /// \return returns cluster occupancy for given cluster time; only valid (non-sentinel) when the refit method is used, since the occupancy map is only filled by setRefit()
  unsigned int getOccupancy(float clTime) const;

  /// \return returns true if given row index is in a stack boundary
  bool isInStackBoundaries(int stackNumber, unsigned char rowIndex, int stackBoundaryMethod);

  /// load calibration objects from CCDB
  /// \param runNumberOrTimeStamp run number or time stamp
  /// \param isMC set if dEdx residual and space-charge corrections will be loaded for MC or real data
  /// \param loadSCCorrMap set to false to skip loading the space-charge correction maps
  void loadCalibsFromCCDB(long runNumberOrTimeStamp, const bool isMC = false, const bool loadSCCorrMap = true);

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
  /// \brief per (sector,row) cluster/track data gathered once per track by gatherRowClusterData(), independent of the dEdx settings reused by calculatedEdxFromRowData() for each entry in a settingsList so the track refit/propagation done in gatherRowClusterData() is not repeated per setting
  struct RowClusterData {
    o2::tpc::ClusterNative cl;       ///< cluster (combined if isCombined)
    o2::tpc::TrackTPC trackSnapshot; ///< track state after refit/propagation to this row's cluster
    unsigned char sectorIndex;
    unsigned char rowIndex;
    unsigned int region;
    unsigned char pad;
    GEMstack stack;
    int stackNumber;
    StackID stackID;
    float chargeTot;
    float chargeMax;
    float clPad;
    float clTime;
    float threshold;
    float gain;
    float gainResidual;
    unsigned int occupancy;
    bool isShared;
    bool isCombined;
    bool isDeadRegion;
    bool propagationFailed;           ///< true if refit/propagation to this row failed, or the resulting track param is NaN
    int missingClusters;              ///< number of skipped rows since the previous entry in rowData (i.e. rowIndex - previous rowIndex - 1); same for every settings entry since rowOrder does not depend on the settings
    bool sameSectorAsPrevRow;         ///< true if this row's sector equals the previous entry in rowData's sector
    bool missingClusterGapDeadOrEdge; ///< true if any of the missingClusters skipped row(s) would land on a dead channel or off the padrow edge
  };

  /// gather, for every (sector, row) of the track's row-traversal order, performing the refit/propagation to each cluster row exactly once
  /// \param track input track, mutated in place by refit/propagation
  /// \param rowData output per-row data
  /// \param averageOcc output average cluster occupancy of the track, per TPC region
  void gatherRowClusterData(o2::tpc::TrackTPC& track, std::vector<RowClusterData>& rowData, AverageOccupancy& averageOcc);

  /// compute the dEdx output for one dEdx settings entry from the row data previously gathered by gatherRowClusterData()
  /// \param rowData per row data gathered by gatherRowClusterData() for the track being processed
  /// \param settings dEdx settings to apply
  /// \param settingsIndex index of settings within its settingsList
  /// \param trackTime0 track.getTime0() of the track being processed, captured before refit/propagation (unaffected by it)
  /// \param trackOrig pristine track (before refit/propagation mutated it), used for the debug "dEdxDebugTrack" row; ignored if mDebug is false
  /// \param averageOcc average cluster occupancy of the track as computed by gatherRowClusterData(), only used for the debug "dEdxDebugTrack" row; ignored if mDebug is false
  /// \param output output dEdxInfo
  void calculatedEdxFromRowData(const std::vector<RowClusterData>& rowData, const dEdxSettings& settings, size_t settingsIndex, float trackTime0, const o2::tpc::TrackTPC& trackOrig, const AverageOccupancy& averageOcc, dEdxInfo& output);

  std::vector<TrackTPC>* mTracks{nullptr};                    ///< vector containing the tpc tracks which will be processed
  std::vector<TPCClRefElem>* mTPCTrackClIdxVecInput{nullptr}; ///< input vector with TPC tracks cluster indicies
  const o2::tpc::ClusterNativeAccess* mClusterIndex{nullptr}; ///< needed to access clusternative with tpctracks
  const o2::gpu::TPCFastTransformPOD* mTPCCorrMap{nullptr};   ///< cluster correction maps helper
  o2::gpu::aligned_unique_buffer_ptr<o2::gpu::TPCFastTransformPOD> mTPCCorrMapBuffer;
  std::vector<unsigned char> mTPCRefitterShMap;                  ///< externally set TPC clusters sharing map
  std::vector<unsigned int> mTPCRefitterOccMap;                  ///< externally set TPC clusters occupancy map
  std::unique_ptr<o2::gpu::GPUO2InterfaceRefit> mRefit{nullptr}; ///< TPC refitter used for TPC tracks refit during the reconstruction

  int mMaxMissingCl{1};                                                                         ///< maximum number of missing clusters for subthreshold check
  float mFieldNominalGPUBz{5};                                                                  ///< magnetic field in kG, used for track propagation
  bool mPropagateTrack{false};                                                                  ///< propagating the track instead of performing a refit (faster than refit)
  bool mPropagateParams{false};                                                                 ///< propagating the parameters instead of full propagation (faster than track propagation)
  bool mDebug{false};                                                                           ///< use the debug streamer
  CalibdEdxContainer mCalibCont;                                                                ///< calibration container
  std::unordered_map<std::string, std::unique_ptr<o2::utils::TreeStreamRedirector>> mStreamers; ///< debug streamers, keyed by output file name so each debugRootFile gets its own tree
  long mDebugTrackIndex{-1};                                                                    ///< running index of the track being processed, written to the debug trees so per-cluster rows can be grouped back into tracks
  long mNPropagationFailed{0};                                                                  ///< number of rows where refit/propagation failed since the last resetDebugCounters()
  long mNRowsProcessed{0};                                                                      ///< number of rows gathered by gatherRowClusterData() since the last resetDebugCounters()
  std::vector<long> mNSubThresholdFilledPerSettings;                                            ///< number of row gaps filled as subthreshold clusters, per dEdxSettings list index, since the last resetDebugCounters()

  CorrectdEdxDistortions mSCdEdxCorrection; ///< for space-charge correction of dE/dx

  std::array<std::vector<unsigned char>, 4> mStackBoundaries = {{{0, 62}, {63, 96}, {97, 126}, {127, 151}}}; // for excluding stack boundaries in dEdx calculation
};

} // namespace o2::tpc

#endif