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

#include "GlobalTracking/MatchTOFBase.h"

class TTree;

namespace o2
{
namespace globaltracking
{

class MatchTOF : public MatchTOFBase
{
 public:
  ///< perform matching for provided input
  void run();

  ///< fill output tree
  void fill();

  ///< perform all initializations
  void init();
  void initTPConly();

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
  void setTOFMCTruthBranchName(const std::string& nm) { mTOFMCTruthBranchName = nm; }
  void setTOFClusterBranchName(const std::string& nm) { mTOFClusterBranchName = nm; }
  void setOutTOFMCTruthBranchName(const std::string& nm) { mOutTOFMCTruthBranchName = nm; }
  void setOutTracksBranchName(const std::string& nm) { mOutTracksBranchName = nm; }
  void setOutCalibBranchName(const std::string& nm) { mOutCalibBranchName = nm; }

  ///< get input branch names for the input from the tree
  const std::string& getTracksBranchName() const { return mTracksBranchName; }
  const std::string& getTPCTracksBranchName() const { return mTPCTracksBranchName; }
  const std::string& getTPCMCTruthBranchName() const { return mTPCMCTruthBranchName; }
  const std::string& getTOFMCTruthBranchName() const { return mTOFMCTruthBranchName; }
  const std::string& getTOFClusterBranchName() const { return mTOFClusterBranchName; }
  const std::string& getOutTOFMCTruthBranchName() const { return mOutTOFMCTruthBranchName; }

 private:
  void attachInputTrees();
  void attachInputTreesTPConly();
  bool prepareTracks();
  bool prepareTPCTracks();
  bool prepareTOFClusters();
  bool loadTracksNextChunk();
  bool loadTPCTracksNextChunk();
  bool loadTOFClustersNextChunk();

  //================================================================
  // Data members

  //  int mCurrTracksTreeEntry = 0;      defined in base class now
  int mCurrTOFClustersTreeEntry = 0; ///< current TOF clusters tree entry loaded to memory

  TTree* mInputTreeTracks = nullptr; ///< input tree for tracks
  TTree* mTreeTPCTracks = nullptr;   ///< input tree for TPC tracks
  TTree* mTreeTOFClusters = nullptr; ///< input tree for TOF clusters

  TTree* mOutputTree = nullptr; ///< output tree for matched tracks

  TTree* mOutputTreeCalib = nullptr; ///< output tree for calibration infos

  std::string mTracksBranchName = "TPCITS";                       ///< name of branch containing input matched tracks
  std::string mTPCTracksBranchName = "TPCTracks";                 ///< name of branch containing actual TPC tracks
  std::string mTPCMCTruthBranchName = "MatchMCTruth";             ///< name of branch containing TPC labels
  std::string mTOFMCTruthBranchName = "TOFClusterMCTruth";        ///< name of branch containing TOF clusters labels
  std::string mTOFClusterBranchName = "TOFCluster";               ///< name of branch containing input ITS clusters
  std::string mOutTracksBranchName = "TOFMatchInfo";              ///< name of branch containing output matched tracks
  std::string mOutCalibBranchName = "TOFCalibInfo";               ///< name of branch containing output calibration infos
  std::string mOutTOFMCTruthBranchName = "MatchTOFMCTruth";       ///< name of branch containing TOF labels for output matched tracks
  std::string mOutTPCTrackMCTruthBranchName = "TPCTracksMCTruth"; ///< name of branch containing TPC labels for input TPC tracks

  ClassDefNV(MatchTOF, 1);
};
} // namespace globaltracking
} // namespace o2

#endif
