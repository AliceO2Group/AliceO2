// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MatchTOF.h
/// \brief Class to perform TOF matching to global tracks
/// \author Francesco.Noferini@cern.ch, Chiara.Zampolli@cern.ch

#ifndef ALICEO2_GLOBTRACKING_MATCHTOF_
#define ALICEO2_GLOBTRACKING_MATCHTOF_

#include <Rtypes.h>
#include <array>
#include <vector>
#include <string>
#include <gsl/span>
#include <TStopwatch.h>
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "CommonDataFormat/EvIndex.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "TOFBase/Geo.h"
#include "DataFormatsTOF/Cluster.h"
#include "GlobalTracking/MatchTPCITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "ReconstructionDataFormats/PID.h"
#include <gsl/span>

// from FIT
#include "DataFormatsFT0/RecPoints.h"

#ifdef _ALLOW_DEBUG_TREES_
//#define _ALLOW_TOF_DEBUG_
#endif

class TTree;

namespace o2
{

namespace dataformats
{
template <typename TruthElement>
class MCTruthContainer;
}

namespace globaltracking
{

///< original track in the currently loaded TPC-ITS reco output
struct TrackLocTPCITS : public o2::track::TrackParCov {
  o2::dataformats::EvIndex<int, int> source; ///< track origin id
  o2::math_utils::Bracket<float> timeBins;   ///< bracketing time-bins
  float zMin = 0;                            // min possible Z of this track
  float zMax = 0;                            // max possible Z of this track
  int matchID = MinusOne;                    ///< entry (none if MinusOne) of TOF matchTOF struct in the mMatchesTOF
  TrackLocTPCITS(const o2::track::TrackParCov& src, int tch, int tid) : o2::track::TrackParCov(src), source(tch, tid) {}
  TrackLocTPCITS() = default;
  ClassDefNV(TrackLocTPCITS, 1); // RS TODO: is this class needed?
};

class MatchTOF
{
  using Geo = o2::tof::Geo;
  using Cluster = o2::tof::Cluster;
  using evGIdx = o2::dataformats::EvIndex<int, o2::dataformats::GlobalTrackID>;
  using evIdx = o2::dataformats::EvIndex<int, int>;
  using timeEst = o2::dataformats::TimeStampWithError<float, float>;
  using matchTrack = std::pair<o2::track::TrackParCov, timeEst>;

 public:
  ///< perform matching for provided input
  void run();

  ///< fill output tree
  void fill();

  ///< perform all initializations
  void init();
  void initTPConly();

  ///< attach DPL data and run
  void run(const gsl::span<const o2::dataformats::TrackTPCITS>& trackArray, const gsl::span<const Cluster>& clusterArray, const o2::dataformats::MCTruthContainer<o2::MCCompLabel>& toflab, const gsl::span<const o2::MCCompLabel>& itslab, const gsl::span<const o2::MCCompLabel>& tpclab);
  void run(const gsl::span<const o2::tpc::TrackTPC>& trackArray, const gsl::span<const Cluster>& clusterArray, const o2::dataformats::MCTruthContainer<o2::MCCompLabel>& toflab, const gsl::span<const o2::MCCompLabel>& tpclab);

  ///< set tree/chain containing tracks
  void setInputTreeTracks(TTree* tree) { mInputTreeTracks = tree; }

  ///< set tree/chain containing TPC tracks
  void setInputTreeTPCTracks(TTree* tree) { mTreeTPCTracks = tree; }

  ///< set tree/chain containing TOF clusters
  void setInputTreeTOFClusters(TTree* tree) { mTreeTOFClusters = tree; }

  ///< set output tree to write matched tracks
  void setOutputTree(TTree* tr) { mOutputTree = tr; }

  ///< set output tree to write calibration infos
  void setOutputTreeCalib(TTree* tr) { mOutputTreeCalib = tr; }

  ///< set input branch names for the input from the tree
  void setTrackBranchName(const std::string& nm) { mTracksBranchName = nm; }
  void setTPCTrackBranchName(const std::string& nm) { mTPCTracksBranchName = nm; }
  void setTPCMCTruthBranchName(const std::string& nm) { mTPCMCTruthBranchName = nm; }
  void setITSMCTruthBranchName(const std::string& nm) { mITSMCTruthBranchName = nm; }
  void setTOFMCTruthBranchName(const std::string& nm) { mTOFMCTruthBranchName = nm; }
  void setTOFClusterBranchName(const std::string& nm) { mTOFClusterBranchName = nm; }
  void setOutTOFMCTruthBranchName(const std::string& nm) { mOutTOFMCTruthBranchName = nm; }
  void setOutTPCMCTruthBranchName(const std::string& nm) { mOutTPCMCTruthBranchName = nm; }
  void setOutITSMCTruthBranchName(const std::string& nm) { mOutITSMCTruthBranchName = nm; }
  void setOutTracksBranchName(const std::string& nm) { mOutTracksBranchName = nm; }
  void setOutCalibBranchName(const std::string& nm) { mOutCalibBranchName = nm; }

  ///< get input branch names for the input from the tree
  const std::string& getTracksBranchName() const { return mTracksBranchName; }
  const std::string& getTPCTracksBranchName() const { return mTPCTracksBranchName; }
  const std::string& getTPCMCTruthBranchName() const { return mTPCMCTruthBranchName; }
  const std::string& getITSMCTruthBranchName() const { return mITSMCTruthBranchName; }
  const std::string& getTOFMCTruthBranchName() const { return mTOFMCTruthBranchName; }
  const std::string& getTOFClusterBranchName() const { return mTOFClusterBranchName; }
  const std::string& getOutTOFMCTruthBranchName() const { return mOutTOFMCTruthBranchName; }
  const std::string& getOutTPCMCTruthBranchName() const { return mOutTPCMCTruthBranchName; }
  const std::string& getOutITSMCTruthBranchName() const { return mOutITSMCTruthBranchName; }

  ///< print settings
  void print() const;
  void printCandidatesTOF() const;

  ///< set time tolerance on track-TOF times comparison
  void setTimeTolerance(float val) { mTimeTolerance = val; }
  ///< get tolerance on track-TOF times comparison
  float getTimeTolerance() const { return mTimeTolerance; }

  ///< set space tolerance on track-TOF times comparison // this in the old AliRoot was the TOF matching window
  void setSpaceTolerance(float val) { mSpaceTolerance = val; }
  ///< get tolerance on track-TOF times comparison
  float getSpaceTolerance() const { return mSpaceTolerance; }

  ///< set number of sigma used to do the matching
  void setSigmaTimeCut(float val) { mSigmaTimeCut = val; }
  ///< get number of sigma used to do the matching
  float getSigmaTimeCut() const { return mSigmaTimeCut; }

  enum DebugFlagTypes : UInt_t {
    MatchTreeAll = 0x1 << 1, ///< produce matching candidates tree for all candidates
  };
  ///< check if partucular flags are set
  bool isDebugFlag(UInt_t flags) const { return mDBGFlags & flags; }

  ///< get debug trees flags
  UInt_t getDebugFlags() const { return mDBGFlags; }

  ///< set or unset debug stream flag
  void setDebugFlag(UInt_t flag, bool on = true);

  ///< set the name of output debug file
  void setDebugTreeFileName(std::string name)
  {
    if (!name.empty()) {
      mDebugTreeFileName = name;
    }
  }

  ///< get the name of output debug file
  const std::string& getDebugTreeFileName() const { return mDebugTreeFileName; }

  ///< fill matching debug tree
  void fillTOFmatchTree(const char* tname, int cacheTOF, int sectTOF, int plateTOF, int stripTOF, int padXTOF, int padZTOF, int cacheeTrk, int crossedStrip, int sectPropagation, int platePropagation, int stripPropagation, int padXPropagation, int padZPropagation, float resX, float resZ, float res, matchTrack& trk, float intLength, float intTimePion, float timeTOF);
  void fillTOFmatchTreeWithLabels(const char* tname, int cacheTOF, int sectTOF, int plateTOF, int stripTOF, int padXTOF, int padZTOF, int cacheeTrk, int crossedStrip, int sectPropagation, int platePropagation, int stripPropagation, int padXPropagation, int padZPropagation, float resX, float resZ, float res, matchTrack& trk, int TPClabelTrackID, int TPClabelEventID, int TPClabelSourceID, int ITSlabelTrackID, int ITSlabelEventID, int ITSlabelSourceID, int TOFlabelTrackID0, int TOFlabelEventID0, int TOFlabelSourceID0, int TOFlabelTrackID1, int TOFlabelEventID1, int TOFlabelSourceID1, int TOFlabelTrackID2, int TOFlabelEventID2, int TOFlabelSourceID2, float intLength, float intTimePion, float timeTOF);
  void dumpWinnerMatches();

  std::vector<o2::dataformats::MatchInfoTOF>& getMatchedTrackVector() { return mMatchedTracks; }
  std::vector<o2::dataformats::CalibInfoTOF>& getCalibVector() { return mCalibInfoTOF; }

  std::vector<o2::MCCompLabel>& getMatchedTOFLabelsVector() { return mOutTOFLabels; } ///< get vector of TOF label of matched tracks
  std::vector<o2::MCCompLabel>& getMatchedTPCLabelsVector() { return mOutTPCLabels; } ///< get vector of TPC label of matched tracks
  std::vector<o2::MCCompLabel>& getMatchedITSLabelsVector() { return mOutITSLabels; } ///< get vector of ITS label of matched tracks

  // this method is deprecated
  void setFITRecPoints(const std::vector<o2::ft0::RecPoints>* recpoints)
  {
    if (recpoints) {
      // need explicit cast because the gsl index_type is signed
      mFITRecPoints = {recpoints->data(), static_cast<decltype(mFITRecPoints)::index_type>(recpoints->size())};
    }
  }
  void setFITRecPoints(gsl::span<o2::ft0::RecPoints const> recpoints)
  {
    mFITRecPoints = recpoints;
  }

  int findFITIndex(int bc);

 private:
  void attachInputTrees();
  void attachInputTreesTPConly();
  bool prepareTracks();
  bool prepareTPCTracks();
  bool prepareTOFClusters();
  bool loadTracksNextChunk();
  bool loadTPCTracksNextChunk();
  bool loadTOFClustersNextChunk();

  void doMatching(int sec);
  void doMatchingForTPC(int sec);
  void selectBestMatches();
  bool propagateToRefX(o2::track::TrackParCov& trc, float xRef /*in cm*/, float stepInCm /*in cm*/, o2::track::TrackLTIntegral& intLT);
  bool propagateToRefXWithoutCov(o2::track::TrackParCov& trc, float xRef /*in cm*/, float stepInCm /*in cm*/, float bz);

  //================================================================

  // Data members

  bool mSAInitDone = false;      ///< flag that standalone init already done
  bool mWFInputAttached = false; ///< flag that the standalone input is attached

  float mXRef = Geo::RMIN; ///< reference radius to propage tracks for matching

  int mCurrTracksTreeEntry = -1;      ///< current tracks tree entry loaded to memory
  int mCurrTOFClustersTreeEntry = -1; ///< current TOF clusters tree entry loaded to memory

  bool mMCTruthON = false; ///< flag availability of MC truth

  ///========== Parameters to be set externally, e.g. from CCDB ====================

  // to be done later

  float mTimeTolerance = 1e3; ///<tolerance in ns for track-TOF time bracket matching
  float mSpaceTolerance = 10; ///<tolerance in cm for track-TOF time bracket matching
  int mSigmaTimeCut = 30.;    ///< number of sigmas to cut on time when matching the track to the TOF cluster

  TTree* mInputTreeTracks = nullptr; ///< input tree for tracks
  TTree* mTreeTPCTracks = nullptr;   ///< input tree for TPC tracks
  TTree* mTreeTOFClusters = nullptr; ///< input tree for TOF clusters

  bool mIsITSused = true;

  TTree* mOutputTree = nullptr; ///< output tree for matched tracks

  TTree* mOutputTreeCalib = nullptr; ///< output tree for calibration infos

  ///>>>------ these are input arrays which should not be modified by the matching code
  //           since this info is provided by external device
  gsl::span<const o2::dataformats::TrackTPCITS> mTracksArrayInp;  ///< input tracks
  std::vector<o2::dataformats::TrackTPCITS>* mTracksArrayInpVect; ///< input tracks (vector to read from tree)
  gsl::span<const o2::tpc::TrackTPC> mTPCTracksArrayInp;          ///< input TPC tracks
  std::vector<o2::tpc::TrackTPC>* mTPCTracksArrayInpVect;         ///< input tracks (vector to read from tree)
  gsl::span<const Cluster> mTOFClustersArrayInp;                  ///< input TOF clusters
  std::vector<Cluster>* mTOFClustersArrayInpVect;                 ///< input TOF clusters (vector to read from tree)

  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mTOFClusLabels;     ///< input TOF clusters MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mTOFClusLabelsPtr; ///< input TOF clusters MC labels (pointer to read from tree)
  std::vector<o2::MCCompLabel> mTracksLblWork;                           ///<TPCITS track labels

  gsl::span<const o2::MCCompLabel> mTPCLabels;  ///< TPC label of input tracks
  gsl::span<const o2::MCCompLabel> mITSLabels;  ///< ITS label of input tracks
  std::vector<o2::MCCompLabel>* mTPCLabelsVect; ///< TPC label of input tracks (vector to read from tree)
  std::vector<o2::MCCompLabel>* mITSLabelsVect; ///< ITS label of input tracks (vector to read from tree)

  gsl::span<o2::ft0::RecPoints const> mFITRecPoints; ///< FIT recpoints

  /// <<<-----

  ///<working copy of the input tracks
  std::vector<matchTrack> mTracksWork;              ///<track params prepared for matching + time value
  std::vector<float> mExtraTPCFwdTime;              ///<track extra params for TPC tracks: Fws Max time
  std::vector<o2::track::TrackLTIntegral> mLTinfos; ///<expected times and others
  std::vector<Cluster> mTOFClusWork;                ///<track params prepared for matching
  std::vector<int8_t> mSideTPC;                     ///<track side for TPC tracks

  ///< per sector indices of track entry in mTracksWork
  std::array<std::vector<int>, o2::constants::math::NSectors> mTracksSectIndexCache;
  ///< per sector indices of track entry in mTPCTracksWork
  std::array<std::vector<int>, o2::constants::math::NSectors> mTPCTracksSectIndexCache;
  ///< per sector indices of TOF cluster entry in mTOFClusWork
  std::array<std::vector<int>, o2::constants::math::NSectors> mTOFClusSectIndexCache;

  ///<array of track-TOFCluster pairs from the matching
  std::vector<o2::dataformats::MatchInfoTOF> mMatchedTracksPairs;

  ///<array of TOFChannel calibration info
  std::vector<o2::dataformats::CalibInfoTOF> mCalibInfoTOF;

  ///<array of matched TOFCluster with matching information (residuals, expected times...) with the corresponding vector of indices
  //std::vector<o2::dataformats::MatchInfoTOF> mMatchedTracks;
  std::vector<o2::dataformats::MatchInfoTOF> mMatchedTracks; // this is the output of the matching
  std::vector<o2::MCCompLabel> mOutTOFLabels;                ///< TOF label of matched tracks
  std::vector<o2::MCCompLabel> mOutTPCLabels;                ///< TPC label of matched tracks
  std::vector<o2::MCCompLabel> mOutITSLabels;                ///< ITS label of matched tracks

  int mNumOfTracks;                     // number of tracks to be matched
  std::vector<int> mMatchedTracksIndex; // vector of indexes of the tracks to be matched
  int mNumOfClusters;                   // number of clusters to be matched
  int* mMatchedClustersIndex = nullptr; //[mNumOfClusters]

  std::string mTracksBranchName = "TPCITS";                    ///< name of branch containing input matched tracks
  std::string mTPCTracksBranchName = "TPCTracks";              ///< name of branch containing actual TPC tracks
  std::string mTPCMCTruthBranchName = "MatchTPCMCTruth";       ///< name of branch containing TPC labels
  std::string mITSMCTruthBranchName = "MatchITSMCTruth";       ///< name of branch containing ITS labels
  std::string mTOFMCTruthBranchName = "TOFClusterMCTruth";     ///< name of branch containing TOF clusters labels
  std::string mTOFClusterBranchName = "TOFCluster";            ///< name of branch containing input ITS clusters
  std::string mOutTracksBranchName = "TOFMatchInfo";           ///< name of branch containing output matched tracks
  std::string mOutCalibBranchName = "TOFCalibInfo";            ///< name of branch containing output calibration infos
  std::string mOutTOFMCTruthBranchName = "MatchTOFMCTruth";    ///< name of branch containing TOF labels for output matched tracks
  std::string mOutTPCMCTruthBranchName = "MatchTPCMCTruth";    ///< name of branch containing TOF labels for output matched tracks
  std::string mOutITSMCTruthBranchName = "MatchITSMCTruth";    ///< name of branch containing TOF labels for output matched tracks
  std::string mOutTPCTrackMCTruthBranchName = "TPCTracksMCTruth"; ///< name of branch containing TPC labels for input TPC tracks

  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  UInt_t mDBGFlags = 0;
  std::string mDebugTreeFileName = "dbg_matchTOF.root"; ///< name for the debug tree file

  ///----------- aux stuff --------------///
  static constexpr float MAXSNP = 0.85; // max snp of ITS or TPC track at xRef to be matched

  Bool_t mIsworkflowON = kFALSE;

  TStopwatch mTimerTot;
  TStopwatch mTimerDBG;
  ClassDefNV(MatchTOF, 3);
};
} // namespace globaltracking
} // namespace o2

#endif
