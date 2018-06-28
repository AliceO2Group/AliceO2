// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <TTree.h>
#include <cassert>

#include "FairLogger.h"
#include "Field/MagneticField.h"
#include "Field/MagFieldFast.h"
#include "TOFBase/Geo.h"

#include "SimulationDataFormat/MCTruthContainer.h"

#include "DetectorsBase/Propagator.h"

#include "MathUtils/Cartesian3D.h"
#include "MathUtils/Utils.h"
#include "CommonConstants/MathConstants.h"
#include "CommonConstants/PhysicsConstants.h"
#include "DetectorsBase/GeometryManager.h"

#include <Math/SMatrix.h>
#include <Math/SVector.h>
#include <TFile.h>
#include <TGeoGlobalMagField.h>
#include "DataFormatsParameters/GRPObject.h"

#include "GlobalTracking/MatchTOF.h"

using namespace o2::globaltracking;

ClassImp(MatchTOF);

//______________________________________________
void MatchTOF::run()
{
  ///< running the matching

  if (!mInitDone) {
    LOG(FATAL) << "init() was not done yet" << FairLogger::endl;
  }

  mTimerTot.Start();

  prepareTracks();
  for (int sec = o2::constants::math::NSectors; sec--;) {
    doMatching(sec);
  }
  if (0) { // enabling this creates very verbose output
    mTimerTot.Stop();
    printCandidatesTOF();
    mTimerTot.Start(false);
  }

  selectBestMatches();

  mTimerTot.Stop();

  printf("Timing:\n");
  printf("Total:        ");
  mTimerTot.Print();
}

//______________________________________________
void MatchTOF::init()
{
  ///< initizalizations

  if (mInitDone) {
    LOG(ERROR) << "Initialization was already done" << FairLogger::endl;
    return;
  }

  attachInputTrees();

  // create output branch
  if (mOutputTree) {
    mOutputTree->Branch(mOutTracksBranchName.data(), &mMatchedTracks);
    LOG(INFO) << "Matched tracks will be stored in " << mOutTracksBranchName << " branch of tree "
              << mOutputTree->GetName() << FairLogger::endl;
  } else {
    LOG(ERROR) << "Output tree is not attached, matched tracks will not be stored" << FairLogger::endl;
  }

  mInitDone = true;

  {
    mTimerTot.Stop();
    mTimerTot.Reset();
  }

  print();
}

//______________________________________________
void MatchTOF::print() const
{
  ///< print the settings

  printf("\n****** component for the matching of tracks to TF clusters ******\n");
  if (!mInitDone) {
    printf("init is not done yet\n");
    return;
  }

  printf("MC truth: %s\n", mMCTruthON ? "on" : "off");
  printf("Time tolerance: %.3f\n", mTimeTolerance);
  printf("Space tolerance: %.3f\n", mSpaceTolerance);

  printf("**********************************************************************\n");
}

//______________________________________________
void MatchTOF::printCandidatesTOF() const
{
  ///< print the candidates for the matching
}

//______________________________________________
void MatchTOF::attachInputTrees()
{
  ///< attaching the input tree

  if (!mInputTreeTracks) {
    LOG(FATAL) << "Input tree with tracks is not set" << FairLogger::endl;
  }

  if (!mTreeTOFClusters) {
    LOG(FATAL) << "TOF clusters data input tree is not set" << FairLogger::endl;
  }

  // input tracks

  if (!mInputTreeTracks->GetBranch(mTracksBranchName.data())) {
    LOG(FATAL) << "Did not find tracks branch " << mTracksBranchName << " in the input tree" << FairLogger::endl;
  }
  mInputTreeTracks->SetBranchAddress(mTracksBranchName.data(), &mTracksArrayInp);
  LOG(INFO) << "Attached tracks " << mTracksBranchName << " branch with " << mInputTreeTracks->GetEntries()
            << " entries" << FairLogger::endl;

  // input TOF clusters

  if (!mTreeTOFClusters->GetBranch(mTOFClusterBranchName.data())) {
    LOG(FATAL) << "Did not find TOF clusters branch " << mTOFClusterBranchName << " in the input tree"
               << FairLogger::endl;
  }
  mTreeTOFClusters->SetBranchAddress(mTOFClusterBranchName.data(), &mTOFClustersArrayInp);
  LOG(INFO) << "Attached TOF clusters " << mTOFClusterBranchName << " branch with " << mTreeTOFClusters->GetEntries()
            << " entries" << FairLogger::endl;

  // is there MC info available ?
  if (mTreeTOFClusters->GetBranch(mTOFMCTruthBranchName.data())) {
    mTreeTOFClusters->SetBranchAddress(mTOFMCTruthBranchName.data(), &mTOFTrkLabels);
    LOG(INFO) << "Found TOF Clusters MCLabels branch " << mTOFMCTruthBranchName << FairLogger::endl;
  }

  mMCTruthON = mTOFTrkLabels;
  mCurrTracksTreeEntry = -1;
  mCurrTOFClustersTreeEntry = -1;
}

//______________________________________________
bool MatchTOF::prepareTracks()
{
  ///< prepare the tracks that we want to match to TOF

  pipo if (!loadTracksNextChunk())
  {
    return false;
  }

  int ntr = mTracksArrayInp->size();

  for (int it = 0; it < ntr; it++) {
    o2::dataformats::TrackTPCITS& trcOrig = (*mTracksArrayInp)[it];
  }
}

//_____________________________________________________
bool MatchTOF::loadTracksNextChunk()
{
  ///< load next chunk of tracks to be matched to TOF
  while (++mCurrTracksTreeEntry < mInputTreeTracks->GetEntries()) {
    mInputTreeTracks->GetEntry(mCurrTracksTreeEntry);
    LOG(DEBUG) << "Loading tracks entry " << mCurrTracksTreeEntry << " -> " << mTracksArrayInp->size()
               << " tracks" << FairLogger::endl;
    if (!mTracksArrayInp->size()) {
      continue;
    }
    return true;
  }
  --mCurrTracksTreeEntry;
  return false;
}
//______________________________________________
void MatchTOF::loadTracksChunk(int chunk)
{
  ///< load the next chunk of tracks to be matched to TOF (chunk = timeframe? to be checked)

  // load single entry from tracks tree
  if (mCurrTracksTreeEntry != chunk) {
    mInputTreeTracks->GetEntry(mCurrTracksTreeEntry = chunk);
  }
}

//______________________________________________
bool MatchTOF::loadTOFClustersNextChunk()
{
  ///< load next chunk of clusters to be matched to TOF
  while (++mCurrTOFClustersTreeEntry < mTreeTOFClusters->GetEntries()) {
    mTreeTOFClusters->GetEntry(mCurrTOFClustersTreeEntry);
    LOG(DEBUG) << "Loading TOF clusters entry " << mCurrTOFClustersTreeEntry << " -> " << mTOFClustersArrayInp->size()
               << " tracks" << FairLogger::endl;
    if (!mTOFClustersArrayInp->size()) {
      continue;
    }
    return true;
  }
  --mCurrTOFClustersTreeEntry;
  return false;
}
//______________________________________________
void MatchTOF::loadTOFClustersChunk(int chunk)
{
  ///< load the next chunk of TOF clusters for the matching (chunk = timeframe? to be checked)

  // load single entry from TOF clusters tree
  if (mCurrTOFClustersTreeEntry != chunk) {
    mTreeTOFClusters->GetEntry(mCurrTOFClustersTreeEntry = chunk);
  }
}
//______________________________________________
void MatchTOF::doMatching(int sec)
{
  ///< do the real matching
}

//______________________________________________
void MatchTOF::selectBestMatches()
{
  ///< define the track-TOFcluster pair
}
