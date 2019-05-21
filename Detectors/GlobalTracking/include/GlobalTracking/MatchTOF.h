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
#include <TStopwatch.h>
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "ReconstructionDataFormats/CalibInfoTOF.h"
#include "CommonDataFormat/EvIndex.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "TOFBase/Geo.h"
#include "DataFormatsTOF/Cluster.h"
#include "GlobalTracking/MatchTPCITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "ReconstructionDataFormats/PID.h"

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
  TimeBracket timeBins;                      ///< bracketing time-bins
  float zMin = 0;                            // min possible Z of this track
  float zMax = 0;                            // max possible Z of this track
  int matchID = MinusOne;                    ///< entry (none if MinusOne) of TOF matchTOF struct in the mMatchesTOF
  TrackLocTPCITS(const o2::track::TrackParCov& src, int tch, int tid) : o2::track::TrackParCov(src), source(tch, tid) {}
  TrackLocTPCITS() = default;
  ClassDefNV(TrackLocTPCITS, 1);
};

class MatchTOF
{
  using Geo = o2::tof::Geo;
  using Cluster = o2::tof::Cluster;
  using evIdx = o2::dataformats::EvIndex<int, int>;

 public:
  ///< perform matching for provided input
  void run();

  ///< fill output tree
  void fill();

  ///< perform all initializations
  void init();

  ///< perform all initializations
  void initWorkflow(const std::vector<o2::dataformats::TrackTPCITS>* trackArray, const std::vector<Cluster>* clusterArray);

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

#ifdef _ALLOW_DEBUG_TREES_
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
  void fillTOFmatchTree(const char* tname, int cacheTOF, int sectTOF, int plateTOF, int stripTOF, int padXTOF, int padZTOF, int cacheeTrk, int crossedStrip, int sectPropagation, int platePropagation, int stripPropagation, int padXPropagation, int padZPropagation, float resX, float resZ, float res, o2::dataformats::TrackTPCITS& trk, float intLength, float intTimePion, float timeTOF);
  void fillTOFmatchTreeWithLabels(const char* tname, int cacheTOF, int sectTOF, int plateTOF, int stripTOF, int padXTOF, int padZTOF, int cacheeTrk, int crossedStrip, int sectPropagation, int platePropagation, int stripPropagation, int padXPropagation, int padZPropagation, float resX, float resZ, float res, o2::dataformats::TrackTPCITS& trk, int TPClabelTrackID, int TPClabelEventID, int TPClabelSourceID, int ITSlabelTrackID, int ITSlabelEventID, int ITSlabelSourceID, int TOFlabelTrackID0, int TOFlabelEventID0, int TOFlabelSourceID0, int TOFlabelTrackID1, int TOFlabelEventID1, int TOFlabelSourceID1, int TOFlabelTrackID2, int TOFlabelEventID2, int TOFlabelSourceID2, float intLength, float intTimePion, float timeTOF);
  void dumpWinnerMatches();
#endif

 private:
  void attachInputTrees();
  bool prepareTracks();
  bool prepareTOFClusters();
  bool loadTracksNextChunk();
  bool loadTOFClustersNextChunk();

  void doMatching(int sec);
  void selectBestMatches();
  bool propagateToRefX(o2::track::TrackParCov& trc, float xRef /*in cm*/, float stepInCm /*in cm*/, o2::track::TrackLTIntegral& intLT);
  bool propagateToRefXWithoutCov(o2::track::TrackParCov& trc, float xRef /*in cm*/, float stepInCm /*in cm*/, float bz);

  //================================================================

  // Data members

  bool mInitDone = false; ///< flag init already done

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

  TTree* mOutputTree = nullptr; ///< output tree for matched tracks

  TTree* mOutputTreeCalib = nullptr; ///< output tree for calibration infos

  ///>>>------ these are input arrays which should not be modified by the matching code
  //           since this info is provided by external device
  std::vector<o2::dataformats::TrackTPCITS>* mTracksArrayInp = nullptr; ///< input tracks
  std::vector<o2::tpc::TrackTPC>* mTPCTracksArrayInp = nullptr;         ///< input TPC tracks
  std::vector<Cluster>* mTOFClustersArrayInp = nullptr;                 ///< input TOF clusters

  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mTOFClusLabels = nullptr; ///< input TOF clusters MC labels
  std::vector<o2::MCCompLabel> mTracksLblWork;                                  ///<TPCITS track labels

  std::vector<o2::MCCompLabel>* mTPCLabels = nullptr; ///< TPC label of input tracks
  std::vector<o2::MCCompLabel>* mITSLabels = nullptr; ///< ITS label of input tracks

  /// <<<-----

  ///<working copy of the input tracks
  std::vector<o2::dataformats::TrackTPCITS> mTracksWork; ///<track params prepared for matching
  std::vector<Cluster> mTOFClusWork;                     ///<track params prepared for matching

  ///< per sector indices of track entry in mTracksWork
  std::array<std::vector<int>, o2::constants::math::NSectors> mTracksSectIndexCache;
  ///< per sector indices of TOF cluster entry in mTOFClusWork
  std::array<std::vector<int>, o2::constants::math::NSectors> mTOFClusSectIndexCache;

  ///<array of track-TOFCluster pairs from the matching
  std::vector<std::pair<evIdx, o2::dataformats::MatchInfoTOF>> mMatchedTracksPairs;

  ///<array of TOFChannel calibration info
  std::vector<o2::dataformats::CalibInfoTOF> mCalibInfoTOF;

  ///<array of matched TOFCluster with matching information (residuals, expected times...) with the corresponding vector of indices
  //std::vector<o2::dataformats::MatchInfoTOF> mMatchedTracks;
  std::vector<std::pair<evIdx, o2::dataformats::MatchInfoTOF>> mMatchedTracks; // this is the output of the matching
  std::vector<o2::MCCompLabel> mOutTOFLabels;                                ///< TOF label of matched tracks
  std::vector<o2::MCCompLabel> mOutTPCLabels;                                ///< TPC label of matched tracks
  std::vector<o2::MCCompLabel> mOutITSLabels;                                ///< ITS label of matched tracks

  int mNumOfTracks;                     // number of tracks to be matched
  std::vector<int> mMatchedTracksIndex; // vector of indexes of the tracks to be matched
  int mNumOfClusters;                   // number of clusters to be matched
  int* mMatchedClustersIndex = nullptr; //[mNumOfClusters]

  std::string mTracksBranchName = "TPCITS";                 ///< name of branch containing input matched tracks
  std::string mTPCTracksBranchName = "Tracks";              ///< name of branch containing actual TPC tracks
  std::string mTPCMCTruthBranchName = "MatchTPCMCTruth";    ///< name of branch containing TPC labels
  std::string mITSMCTruthBranchName = "MatchITSMCTruth";    ///< name of branch containing ITS labels
  std::string mTOFMCTruthBranchName = "TOFClusterMCTruth";  ///< name of branch containing TOF clusters labels
  std::string mTOFClusterBranchName = "TOFCluster";         ///< name of branch containing input ITS clusters
  std::string mOutTracksBranchName = "TOFMatchInfo";        ///< name of branch containing output matched tracks
  std::string mOutCalibBranchName = "TOFCalibInfo";         ///< name of branch containing output calibration infos
  std::string mOutTOFMCTruthBranchName = "MatchTOFMCTruth"; ///< name of branch containing TOF labels for output matched tracks
  std::string mOutTPCMCTruthBranchName = "MatchTPCMCTruth"; ///< name of branch containing TOF labels for output matched tracks
  std::string mOutITSMCTruthBranchName = "MatchITSMCTruth"; ///< name of branch containing TOF labels for output matched tracks

#ifdef _ALLOW_DEBUG_TREES_
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  UInt_t mDBGFlags = 0;
  std::string mDebugTreeFileName = "dbg_matchTOF.root"; ///< name for the debug tree file
#endif

  ///----------- aux stuff --------------///
  static constexpr float MAXSNP = 0.85; // max snp of ITS or TPC track at xRef to be matched

  Bool_t mIsworkflowON = kFALSE;

  TStopwatch mTimerTot;
  TStopwatch mTimerDBG;
  ClassDefNV(MatchTOF, 2);
};
} // namespace globaltracking
} // namespace o2

#endif
