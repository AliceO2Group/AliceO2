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
#include "DataFormatsTOF/Cluster.h"

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

class MatchTOF
{
 public:
  ///< perform matching for provided input
  void run();

  ///< perform all initializations
  void init();

  ///< set tree/chain containing tracks
  void setInputTreeTracks(TTree* tree) { mInputTreeTracks = tree; }

  ///< set tree/chain containing TOF clusters
  void setInputTreeTOFClusters(TTree* tree) { mTreeTOFClusters = tree; }

  ///< set output tree to write matched tracks
  void setOutputTree(TTree* tr) { mOutputTree = tr; }

  ///< set input branch names for the input from the tree
  void setTrackBranchName(const std::string& nm) { mTracksBranchName = nm; }
  void setTOFClusterBranchName(const std::string& nm) { mTOFClusterBranchName = nm; }
  void setTOFMCTruthBranchName(const std::string& nm) { mTOFMCTruthBranchName = nm; }
  void setOutTracksBranchName(const std::string& nm) { mOutTracksBranchName = nm; }

  ///< get input branch names for the input from the tree
  const std::string& getTracksBranchName() const { return mTracksBranchName; }
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

 private:
  void attachInputTrees();
  bool prepareTracks();
  bool loadTracksNextChunk();
  void loadTracksChunk(int chunk);
  bool loadTOFClustersNextChunk();
  void loadTOFClustersChunk(int chunk);

  void doMatching(int sec);
  void selectBestMatches();

  //================================================================

  // Data members

  bool mInitDone = false; ///< flag init already done

  int mCurrTracksTreeEntry = -1;      ///< current tracks tree entry loaded to memory
  int mCurrTOFClustersTreeEntry = -1; ///< current TOF clusters tree entry loaded to memory

  bool mMCTruthON = false; ///< flag availability of MC truth

  ///========== Parameters to be set externally, e.g. from CCDB ====================

  // to be done later

  float mTimeTolerance = 1e3; ///<tolerance in ns for track-TOF time bracket matching
  float mSpaceTolerance = 10; ///<tolerance in cm for track-TOF time bracket matching

  TTree* mInputTreeTracks = nullptr; ///< input tree for tracks
  TTree* mTreeTOFClusters = nullptr; ///< input tree for TOF clusters

  TTree* mOutputTree = nullptr; ///< output tree for matched tracks

  ///>>>------ these are input arrays which should not be modified by the matching code
  //           since this info is provided by external device
  std::vector<o2::dataformats::TrackTPCITS>* mTracksArrayInp = nullptr; ///< input tracks

  std::vector<o2::tof::Cluster>* mTOFClustersArrayInp = nullptr; ///< input TOF clusters

  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mTOFTrkLabels = nullptr; ///< input TOF Track MC labels

  /// <<<-----

  ///<outputs tracks container
  std::vector<o2::dataformats::MatchInfoTOF> mMatchedTracks;

  std::string mTracksBranchName = "TPCITS";                ///< name of branch containing input matched tracks
  std::string mTOFClusterBranchName = "TOFCluster";        ///< name of branch containing input ITS clusters
  std::string mTOFMCTruthBranchName = "TOFClusterMCTruth"; ///< name of branch containing ITS MC labels
  std::string mOutTracksBranchName = "TOFMatchInfo";       ///< name of branch containing output matched tracks

  TStopwatch mTimerTot;
  ClassDefNV(MatchTOF, 1);
};
} // namespace globaltracking
} // namespace o2

#endif
