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

#define _ALLOW_DEBUG_TREES_ // to allow debug and control tree output

#include <Rtypes.h>
#include <array>
#include <vector>
#include <string>
#include <TStopwatch.h>
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "CommonDataFormat/EvIndex.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "TOFBase/Geo.h"
#include "DataFormatsTOF/Cluster.h"
#include "GlobalTracking/MatchTPCITS.h"
#include "DataFormatsTPC/TrackTPC.h"

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
  timeBracket timeBins;                      ///< bracketing time-bins
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
  
 public:
  ///< perform matching for provided input
  void run();

  ///< perform all initializations
  void init();

  ///< set tree/chain containing tracks
  void setInputTreeTracks(TTree* tree) { mInputTreeTracks = tree; }

  ///< set tree/chain containing TPC tracks
  void setInputTreeTPCTracks(TTree* tree) { mTreeTPCTracks = tree; }

  ///< set tree/chain containing TOF clusters
  void setInputTreeTOFClusters(TTree* tree) { mTreeTOFClusters = tree; }

  ///< set output tree to write matched tracks
  void setOutputTree(TTree* tr) { mOutputTree = tr; }

  ///< set input branch names for the input from the tree
  void setTrackBranchName(const std::string& nm) { mTracksBranchName = nm; }
  void setTPCTrackBranchName(const std::string& nm) { mTPCTracksBranchName = nm; }
  void setTOFClusterBranchName(const std::string& nm) { mTOFClusterBranchName = nm; }
  void setTOFMCTruthBranchName(const std::string& nm) { mTOFMCTruthBranchName = nm; }
  void setOutTracksBranchName(const std::string& nm) { mOutTracksBranchName = nm; }

  ///< get input branch names for the input from the tree
  const std::string& getTracksBranchName() const { return mTracksBranchName; }
  const std::string& getTPCTracksBranchName() const { return mTPCTracksBranchName;}
  const std::string& getTOFClusterBranchName() const { return mTOFClusterBranchName; }
  const std::string& getTOFMCTruthBranchName() const { return mTOFMCTruthBranchName; }

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

 private:
  void attachInputTrees();
  bool prepareTracks();
  bool prepareTOFClusters();
  bool loadTracksNextChunk();
  void loadTracksChunk(int chunk);
  bool loadTOFClustersNextChunk();
  void loadTOFClustersChunk(int chunk);
  
  void doMatching(int sec);
  void selectBestMatches();
  bool propagateToRefX(o2::track::TrackParCov& trc, float xRef /*in cm*/, float stepInCm /*in cm*/);

  //================================================================

  // Data members

  bool mInitDone = false; ///< flag init already done

  float mXRef = Geo::RMIN;                 ///< reference radius to propage tracks for matching
  
  int mCurrTracksTreeEntry = -1;      ///< current tracks tree entry loaded to memory
  int mCurrTOFClustersTreeEntry = -1; ///< current TOF clusters tree entry loaded to memory

  bool mMCTruthON = false; ///< flag availability of MC truth

  ///========== Parameters to be set externally, e.g. from CCDB ====================

  // to be done later

  float mTimeTolerance = 1e3; ///<tolerance in ns for track-TOF time bracket matching
  float mSpaceTolerance = 10; ///<tolerance in cm for track-TOF time bracket matching
  int   mSigmaTimeCut = 3; ///< number of sigmas to cut on time when matching the track to the TOF cluster
  
  TTree* mInputTreeTracks = nullptr; ///< input tree for tracks
  TTree* mTreeTPCTracks = nullptr;   ///< input tree for TPC tracks
  TTree* mTreeTOFClusters = nullptr; ///< input tree for TOF clusters

  TTree* mOutputTree = nullptr; ///< output tree for matched tracks

  ///>>>------ these are input arrays which should not be modified by the matching code
  //           since this info is provided by external device
  std::vector<o2::dataformats::TrackTPCITS>* mTracksArrayInp = nullptr; ///< input tracks
  std::vector<o2::TPC::TrackTPC>* mTPCTracksArrayInp = nullptr; ///< input TPC tracks
  std::vector<Cluster>* mTOFClustersArrayInp = nullptr; ///< input TOF clusters

  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mTracksLabels = nullptr; ///< input TPC Track MC labels

  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mTOFClusLabels = nullptr; ///< input TOF clusters MC labels
  std::vector<o2::MCCompLabel> mTracksLblWork; ///<TPCITS track labels
  std::vector<o2::MCCompLabel> mTTOFClusLblWork; ///<TOF cluster labels


  /// <<<-----

  ///<working copy of the input tracks
  std::vector<o2::dataformats::TrackTPCITS> mTracksWork;        ///<track params prepared for matching
  std::vector<Cluster> mTOFClusWork;        ///<track params prepared for matching

  ///< per sector indices of track entry in mTracksWork
  std::array<std::vector<int>, o2::constants::math::NSectors> mTracksSectIndexCache;
  ///< per sector indices of TOF cluster entry in mTOFClusWork
  std::array<std::vector<int>, o2::constants::math::NSectors> mTOFClusSectIndexCache;

  

  ///<array of track-TOFCluster pairs from the matching
  std::vector<std::pair<int, o2::dataformats::MatchInfoTOF>> mMatchedTracksPairs;

  ///<array of matched TOFCluster with matching information (residuals, expected times...) with the corresponding vector of indices
  std::vector<o2::dataformats::MatchInfoTOF> mMatchedTracks;
  int mNumOfTracks;  // number of tracks to be matched
  int* mMatchedTracksIndex = nullptr;  //[mNumOfTracks]
  int mNumOfClusters;  // number of clusters to be matched
  int* mMatchedClustersIndex = nullptr;  //[mNumOfClusters]

  std::string mTracksBranchName = "TPCITS";                ///< name of branch containing input matched tracks
  std::string mTPCTracksBranchName = "TPC";                ///< name of branch containing actual TPC tracks
  std::string mTOFClusterBranchName = "TOFCluster";        ///< name of branch containing input ITS clusters
  std::string mTOFMCTruthBranchName = "TOFClusterMCTruth"; ///< name of branch containing ITS MC labels
  std::string mOutTracksBranchName = "TOFMatchInfo";       ///< name of branch containing output matched tracks

  ///----------- aux stuff --------------///
  static constexpr float MaxSnp = 0.85;                // max snp of ITS or TPC track at xRef to be matched
  
  TStopwatch mTimerTot;
  ClassDefNV(MatchTOF, 1);
};
} // namespace globaltracking
} // namespace o2

#endif
