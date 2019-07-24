// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackInterpolation.h
/// \brief Definition of the TrackInterpolation class
///
/// \author Ole Schmidt, ole.schmidt@cern.ch

#ifndef ALICEO2_TPC_TRACKINTERPOLATION_H_
#define ALICEO2_TPC_TRACKINTERPOLATION_H_

#include "CommonDataFormat/EvIndex.h"
#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/Constants.h"
#include "DataFormatsTOF/Cluster.h"
#include "SpacePoints/SpacePointsCalibParam.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"

class TTree;

namespace o2
{
namespace tpc
{

/// Structure used to store the TPC cluster residuals
struct TPCClusterResiduals {
  short dy{};           ///< residual in Y
  short dz{};           ///< residual in Z
  short y{};            ///< Y position of track
  short z{};            ///< Z position of track
  short phi{};          ///< phi angle of track
  short tgl{};          ///< dip angle of track
  unsigned char sec{};  ///< sector number 0..17
  unsigned char dRow{}; ///< distance to previous row in units of pad rows
  short row{};          ///< TPC pad row (absolute units)
  void setDY(float val) { dy = fabs(val) < param::MaxResid ? static_cast<short>(val * 0x7fff / param::MaxResid) : static_cast<short>(std::copysign(1., val) * 0x7fff); }
  void setDZ(float val) { dz = fabs(val) < param::MaxResid ? static_cast<short>(val * 0x7fff / param::MaxResid) : static_cast<short>(std::copysign(1., val) * 0x7fff); }
  void setY(float val) { y = fabs(val) < param::MaxY ? static_cast<short>(val * 0x7fff / param::MaxY) : static_cast<short>(std::copysign(1., val) * 0x7fff); }
  void setZ(float val) { z = fabs(val) < param::MaxZ ? static_cast<short>(val * 0x7fff / param::MaxZ) : static_cast<short>(std::copysign(1., val) * 0x7fff); }
  void setPhi(float val) { phi = fabs(val) < param::MaxTgSlp ? static_cast<short>(val * 0x7fff / param::MaxTgSlp) : static_cast<short>(std::copysign(1., val) * 0x7fff); }
  void setTgl(float val) { tgl = fabs(val) < param::MaxTgSlp ? static_cast<short>(val * 0x7fff / param::MaxTgSlp) : static_cast<short>(std::copysign(1., val) * 0x7fff); }
};

/// Structure filled for each track with track quality information and a vector with TPCClusterResiduals
struct TrackData {
  int trkId{};                 ///< track ID for debugging
  float eta{};                 ///< track dip angle
  float phi{};                 ///< track azimuthal angle
  float qPt{};                 ///< track q/pT
  float chi2TPC{};             ///< chi2 of TPC track
  float chi2ITS{};             ///< chi2 of ITS track
  unsigned short nClsTPC{};    ///< number of attached TPC clusters
  unsigned short nClsITS{};    ///< number of attached ITS clusters
  unsigned short nTrkltsTRD{}; ///< number of attached TRD tracklets
  // TODO: Add an additional structure with event information and reference to the tracks for given event?
  int nTracksInEvent{};                      ///< total number of tracks in given event / timeframe ? Used for multiplicity estimate
  o2::dataformats::RangeReference<> clIdx{}; ///< index of first cluster residual and total number of cluster residuals of this track
};

/// \class TrackInterpolation
/// This class is retrieving the TPC space point residuals by interpolating ITS/TRD/TOF tracks.
/// The residuals are stored in the specified vectors of TPCClusterResiduals
/// It has been ported from the AliTPCcalibAlignInterpolation class from AliRoot.
class TrackInterpolation
{
 public:
  /// Default constructor
  TrackInterpolation() = default;

  /// Enumeration for indexing the arrays of the CacheStruct
  enum {
    ExtOut = 0, ///< extrapolation outwards of ITS track
    ExtIn,      ///< extrapolation inwards of TRD/TOF track
    Int,        ///< interpolation (mean positions of both extrapolations)
    NIndices    ///< total number of indices (3)
  };

  /// Structure for caching positions, covariances and angles for extrapolations from ITS and TRD/TOF and for interpolation
  struct CacheStruct {
    std::array<float, NIndices> y{};
    std::array<float, NIndices> z{};
    std::array<float, NIndices> sy2{};
    std::array<float, NIndices> szy{};
    std::array<float, NIndices> sz2{};
    std::array<float, NIndices> phi{};
    std::array<float, NIndices> tgl{};
    float clY{ 0.f };
    float clZ{ 0.f };
    float clAngle{ 0.f };
    unsigned short clAvailable{ 0 };
  };

  /// Initialize everything
  void init();

  /// Main processing function
  void process();

  /// Extrapolate ITS-only track through TPC and store residuals to TPC clusters along the way
  /// \param trkITS ITS only track to be extrapolated
  /// \param trkTPC TPC track matched to trkITS used for TPC cluster access
  /// \param trkTime time assigned to TPC track (needed for cluster coordinate transformation)
  /// \param trkIdTPC TPC track ID
  /// \return flag if track could successfully be extrapolated to all TPC clusters
  bool extrapolateTrackITS(const o2::its::TrackITS& trkITS, const TrackTPC& trkTPC, float trkTime, int trkIdTPC);

  /// Interpolate ITS-TPC-TOF tracks in the TPC and store residuals to TPC clusters
  /// \param matchTOF
  /// \return flag if track could be processed successfully
  bool interpolateTrackITSTOF(const std::pair<o2::dataformats::EvIndex<>, o2::dataformats::MatchInfoTOF>& matchTOF);

  /// Check if given ITS-TPC track fullfills quality criteria
  /// \param matchITSTPC ITS-TPC track to be checked
  /// \param hasOuterPoint Flag if track has an attached cluster in TRD or TOF
  /// \return Flag if quality criteria are met
  bool trackPassesQualityCuts(const o2::dataformats::TrackTPCITS& matchITSTPC, bool hasOuterPoint = true) const;

  /// Load specified entry for all input trees
  /// \param iEntry number of tree entry
  void loadEntryForTrees(int iEntry);

  /// Attach all input trees
  void attachInputTrees();

  /// Add branches to the output trees
  void prepareOutputTrees();

  /// Reset cache and output vectors
  void reset();

  // -------------------------------------- settings --------------------------------------------------

  /// Sets the tree/chain containing ITS-TPC matched tracks
  void setInputTreeITSTPCTracks(TTree* tree) { mTreeITSTPCTracks = tree; }
  /// Sets the tree/chain containing TPC tracks
  void setInputTreeTPCTracks(TTree* tree) { mTreeTPCTracks = tree; }
  ///< Sets the reader for TPC clusters
  void setInputTPCClustersReader(ClusterNativeHelper::Reader* reader) { mTPCClusterReader = reader; }
  /// Sets the tree/chain containing ITS tracks
  void setInputTreeITSTracks(TTree* tree) { mTreeITSTracks = tree; }
  /// Sets the tree/chain containing ITS clusters
  void setInputTreeITSClusters(TTree* tree) { mTreeITSClusters = tree; }
  /// Sets the tree/chain containing TOF matching information
  void setInputTreeTOFMatches(TTree* tree) { mTreeTOFMatches = tree; }
  ///< Sets the tree/chain containing TOF clusters
  void setInputTreeTOFClusters(TTree* tree) { mTreeTOFClusters = tree; }

  /// Sets the output tree for track information
  void setOutputTreeTracks(TTree* tree) { mTreeOutTrackData = tree; }
  /// Sets the output tree for TPC cluster residuals
  void setOutputTreeResiduals(TTree* tree) { mTreeOutClusterRes = tree; }

  /// Sets the maximum phi angle at which track extrapolation is aborted
  void setMaxSnp(float snp) { mMaxSnp = snp; }
  /// Sets the maximum step length for track extrapolation
  void setMaxStep(float step) { mMaxStep = step; }
  /// Sets the flag if material correction should be applied when extrapolating the tracks
  void setMatCorr(int matCorr) { mMatCorr = matCorr; }
  /// Sets whether ITS tracks without match in TRD or TOF should be processed as well
  void setDoITSOnlyTracks(bool flag) { mDoITSOnlyTracks = flag; }

  /// Setters and Getters for the input branch/file names
  void setITSTPCTrackBranchName(const std::string& name) { mITSTPCTrackBranchName = name; }
  void setTPCTrackBranchName(const std::string& name) { mTPCTrackBranchName = name; }
  void setTPCClusterFileName(const std::string& name) { mTPCClusterFileName = name; }
  void setITSTrackBranchName(const std::string& name) { mITSTrackBranchName = name; }
  void setITSClusterBranchName(const std::string& name) { mITSClusterBranchName = name; }
  void setTOFMatchingBranchName(const std::string& name) { mTOFMatchingBranchName = name; }
  void setTOFClusterBranchName(const std::string& name) { mTOFClusterBranchName = name; }

  const std::string& getITSTPCTrackBranchName() const { return mITSTPCTrackBranchName; }
  const std::string& getTPCTrackBranchName() const { return mTPCTrackBranchName; }
  const std::string& getTPCClusterFileName() const { return mTPCClusterFileName; }
  const std::string& getITSTrackBranchName() const { return mITSTrackBranchName; }
  const std::string& getITSClusterBranchName() const { return mITSClusterBranchName; }
  const std::string& getTOFMatchingBranchName() const { return mTOFMatchingBranchName; }
  const std::string& getTOFClusterBranchName() const { return mTOFClusterBranchName; }

 private:
  // names of input files / branches
  std::string mITSTPCTrackBranchName{ "TPCITS" };                ///< name of branch containing ITS-TPC matched tracks
  std::string mTPCTrackBranchName{ "Tracks" };                   ///< name of branch containing TPC tracks (needed for TPC cluster access)
  std::string mTPCClusterFileName{ "tpc-native-clusters.root" }; ///< name of file containing TPC native clusters
  std::string mITSTrackBranchName{ "ITSTrack" };                 ///< name of branch containing input ITS tracks
  std::string mITSClusterBranchName{ "ITSCluster" };             ///< name of branch containing input ITS clusters
  std::string mTOFMatchingBranchName{ "TOFMatchInfo" };          ///< name of branch containing matching info for ITS-TPC-TOF tracks
  std::string mTOFClusterBranchName{ "TOFCluster" };             ///< name of branch containing TOF clusters

  // parameters
  float mTPCTimeBinMUS{ .2f }; ///< TPC time bin duration in us
  float mSigYZ2TOF{ .75f };    ///< for now assume cluster error for TOF equal for all clusters in both Y and Z

  // settings
  float mMaxSnp{ .85f };          ///< max snp when propagating ITS tracks
  float mMaxStep{ 2.f };          ///< maximum step for propagation
  int mMatCorr{ 1 };              ///< if material correction should be done
  bool mDoITSOnlyTracks{ false }; ///< if ITS only tracks should be processed or not

  // input
  TTree* mTreeITSTPCTracks{ nullptr };                                                                             ///< input tree for ITS-TPC matched tracks
  TTree* mTreeTPCTracks{ nullptr };                                                                                ///< input tree for TPC tracks
  TTree* mTreeITSTracks{ nullptr };                                                                                ///< input tree for ITS tracks
  TTree* mTreeITSClusters{ nullptr };                                                                              ///< input tree for ITS clusters
  TTree* mTreeTOFMatches{ nullptr };                                                                               ///< input tree for ITS-TPC-TOF matches
  TTree* mTreeTOFClusters{ nullptr };                                                                              ///< input tree for TOF clusters
  std::vector<o2::dataformats::TrackTPCITS>* mITSTPCTrackVecInput{ nullptr };                                      ///< input vector with ITS-TPC matched tracks
  std::vector<TrackTPC>* mTPCTrackVecInput{ nullptr };                                                             ///< input vector with TPC tracks
  std::vector<o2::its::TrackITS>* mITSTrackVecInput{ nullptr };                                                    ///< input vector with ITS tracks
  std::vector<o2::itsmft::Cluster>* mITSClusterVecInput{ nullptr };                                                ///< input vector with ITS clusters
  ClusterNativeHelper::Reader* mTPCClusterReader = nullptr;                                                        ///< TPC cluster reader
  std::unique_ptr<ClusterNativeAccess> mTPCClusterIdxStructOwn;                                                    ///< struct holding the TPC cluster indices (for tree-based I/O)
  std::unique_ptr<ClusterNative[]> mTPCClusterBufferOwn;                                                           ///< buffer for clusters in mTPCClusterIdxStructOwn
  MCLabelContainer mTPCClusterMCBufferOwn;                                                                         ///< buffer for mc labels
  std::vector<std::pair<o2::dataformats::EvIndex<>, o2::dataformats::MatchInfoTOF>>* mTOFMatchVecInput{ nullptr }; ///< input vector for ITS-TPC-TOF matching information
  std::vector<o2::tof::Cluster>* mTOFClusterVecInput{ nullptr };                                                   ///< input vector with TOF clusters

  const ClusterNativeAccess* mTPCClusterIdxStruct = nullptr; ///< struct holding the TPC cluster indices

  // output
  TTree* mTreeOutTrackData{ nullptr };                    ///< output tree with track quality information
  std::vector<TrackData> mTrackData{};                    ///< this vector is used to store the track quality information on a per track basis
  std::vector<TrackData>* mTrackDataPtr{ &mTrackData };   ///< pointer to vector with track data
  TTree* mTreeOutClusterRes{ nullptr };                   ///< output tree with TPC cluster residuals
  std::vector<TPCClusterResiduals> mClRes{};              ///< residuals for each available TPC cluster of all tracks
  std::vector<TPCClusterResiduals>* mClResPtr{ &mClRes }; ///< pointer to vector with residuals

  // cache
  std::array<CacheStruct, Constants::MAXGLOBALPADROW> mCache{ {} }; ///< caching positions, covariances and angles for track extrapolations and interpolation

  // helpers
  std::unique_ptr<TPCFastTransform> mFastTransform{}; ///< TPC cluster transformation
};

} // namespace tpc

} // namespace o2

#endif
