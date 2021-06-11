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

#include "MathUtils/Cartesian.h"
#include "MathUtils/Utils.h"
#include "CommonConstants/MathConstants.h"
#include "CommonConstants/PhysicsConstants.h"
#include "CommonConstants/GeomConstants.h"
#include "DetectorsBase/GeometryManager.h"

#include <Math/SMatrix.h>
#include <Math/SVector.h>
#include <TFile.h>
#include <TGeoGlobalMagField.h>
#include "DataFormatsParameters/GRPObject.h"
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/TrackLTIntegral.h"

#include "GlobalTracking/MatchTOF.h"

#include "TPCBase/ParameterGas.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"

using namespace o2::globaltracking;
using trkType = o2::dataformats::MatchInfoTOF::TrackType;

ClassImp(MatchTOF);

//______________________________________________
void MatchTOF::run()
{
  ///< running the matching

  //  if (!mWFInputAttached && !mSAInitDone) {
  if (!mWFInputAttached) {
    // LOG(ERROR) << "run called with mSAInitDone=" << mSAInitDone << " and mWFInputAttached=" << mWFInputAttached;
    // throw std::runtime_error("standalone init was not done or workflow input was not yet attached");
    LOG(ERROR) << "run called with mWFInputAttached=" << mWFInputAttached;
    throw std::runtime_error("workflow input was not yet attached");
  }

  updateTimeDependentParams();

  mTimerTot.Start();

  // we load all TOF clusters (to be checked if we need to split per time frame)
  prepareTOFClusters();

  mTimerTot.Stop();
  LOGF(INFO, "Timing prepareTOFCluster: Cpu: %.3e s Real: %.3e s in %d slots", mTimerTot.CpuTime(), mTimerTot.RealTime(), mTimerTot.Counter() - 1);
  mTimerTot.Start();

  for (int i = 0; i < trkType::SIZE; i++) {
    mNumOfTracks[i] = 0;
    mMatchedTracks[i].clear();
    mTracksWork[i].clear();
    mOutTOFLabels[i].clear();
  }

  if (mIsworkflowON) {
    LOG(DEBUG) << "Number of entries in track tree = " << mCurrTracksTreeEntry;

    if (mIsITSTPCused) {
      prepareTracks();
    }
    if (mIsTPCused) {
      prepareTPCTracks();
    }

    mTimerTot.Stop();
    LOGF(INFO, "Timing prepare tracks: Cpu: %.3e s Real: %.3e s in %d slots", mTimerTot.CpuTime(), mTimerTot.RealTime(), mTimerTot.Counter() - 1);
    mTimerTot.Start();

    for (int sec = o2::constants::math::NSectors; sec--;) {
      LOG(INFO) << "Doing matching for sector " << sec << "...";
      if (mIsITSTPCused) {
        doMatching(sec, trkType::ITSTPC);
      }
      if (mIsTPCused) {
        doMatchingForTPC(sec);
      }
      LOG(INFO) << "...done. Now check the best matches";
      selectBestMatches();
    }
  }

  // we do the matching per entry of the TPCITS matched tracks tree
  while (!mIsworkflowON && mCurrTracksTreeEntry + 1 < mInputTreeTracks->GetEntries()) { // we add "+1" because mCurrTracksTreeEntry starts from -1, and it is incremented in loadTracksNextChunk which is called by prepareTracks
    LOG(DEBUG) << "Number of entries in track tree = " << mCurrTracksTreeEntry;

    if (mIsITSTPCused) {
      prepareTracks();
    }
    if (mIsTPCused) {
      prepareTPCTracks();
    }

    mTimerTot.Stop();
    LOGF(INFO, "Timing prepare tracks: Cpu: %.3e s Real: %.3e s in %d slots", mTimerTot.CpuTime(), mTimerTot.RealTime(), mTimerTot.Counter() - 1);
    mTimerTot.Start();

    for (int sec = o2::constants::math::NSectors; sec--;) {
      mMatchedTracksPairs.clear();
      LOG(INFO) << "Doing matching for sector " << sec << "...";
      if (mIsITSTPCused) {
        doMatching(sec, trkType::ITSTPC);
      }
      if (mIsTPCused) {
        doMatchingForTPC(sec);
      }
      LOG(INFO) << "...done. Now check the best matches";
      selectBestMatches();
    }

    mTimerTot.Stop();
    LOGF(INFO, "Timing Do Matching: Cpu: %.3e s Real: %.3e s in %d slots", mTimerTot.CpuTime(), mTimerTot.RealTime(), mTimerTot.Counter() - 1);
    mTimerTot.Start();

    //    fill();
  }

  mWFInputAttached = false;

  mTimerTot.Stop();
  LOGF(INFO, "Timing Do Matching: Cpu: %.3e s Real: %.3e s in %d slots", mTimerTot.CpuTime(), mTimerTot.RealTime(), mTimerTot.Counter() - 1);
}

//______________________________________________
void MatchTOF::fill()
{
  mOutputTree->Fill();
  if (mOutputTreeCalib) {
    mOutputTreeCalib->Fill();
  }
}
//______________________________________________
void MatchTOF::init()
{
  ///< initizalizations
  mIsITSTPCused = true;
  mIsTPCused = false;

  if (mSAInitDone) {
    LOG(ERROR) << "Initialization was already done";
    return;
  }
  attachInputTrees();

  // create output branch with track-tof matching
  if (mOutputTree) {
    mOutputTree->Branch(mOutTracksBranchName.data(), &(mMatchedTracks[trkType::ITSTPC]));
    LOG(INFO) << "Matched tracks will be stored in " << mOutTracksBranchName << " branch of tree "
              << mOutputTree->GetName();
    if (mMCTruthON) {
      mOutputTree->Branch(mOutTOFMCTruthBranchName.data(), &(mOutTOFLabels[trkType::ITSTPC]));
      LOG(INFO) << "TOF Tracks Labels branch: " << mOutTOFMCTruthBranchName;
    }

  } else {
    LOG(INFO) << "Output tree is not attached, matched tracks will not be stored";
  }

  // create output branch for calibration info
  if (mOutputTreeCalib) {
    mOutputTreeCalib->Branch(mOutCalibBranchName.data(), &mCalibInfoTOF);
    LOG(INFO) << "Calib infos will be stored in " << mOutCalibBranchName << " branch of tree "
              << mOutputTreeCalib->GetName();
  } else {
    LOG(INFO) << "Calib Output tree is not attached, calib infos will not be stored";
  }

  mSAInitDone = true;

  {
    mTimerTot.Stop();
    mTimerTot.Reset();
  }

  print();
}
//______________________________________________
void MatchTOF::initTPConly()
{
  ///< initizalizations

  mIsITSTPCused = false;
  mIsTPCused = true;

  if (mSAInitDone) {
    LOG(ERROR) << "Initialization was already done";
    return;
  }
  attachInputTreesTPConly();

  // create output branch with track-tof matching
  if (mOutputTree) {
    mOutputTree->Branch(mOutTracksBranchName.data(), &(mMatchedTracks[trkType::TPC]));
    LOG(INFO) << "Matched tracks will be stored in " << mOutTracksBranchName << " branch of tree "
              << mOutputTree->GetName();
    if (mMCTruthON) {
      mOutputTree->Branch(mOutTOFMCTruthBranchName.data(), &(mOutTOFLabels[trkType::TPC]));
      LOG(INFO) << "TOF Tracks Labels branch: " << mOutTOFMCTruthBranchName;
    }

  } else {
    LOG(INFO) << "Output tree is not attached, matched tracks will not be stored";
  }

  // create output branch for calibration info
  if (mOutputTreeCalib) {
    mOutputTreeCalib->Branch(mOutCalibBranchName.data(), &mCalibInfoTOF);
    LOG(INFO) << "Calib infos will be stored in " << mOutCalibBranchName << " branch of tree "
              << mOutputTreeCalib->GetName();
  } else {
    LOG(INFO) << "Calib Output tree is not attached, calib infos will not be stored";
  }

  mSAInitDone = true;

  {
    mTimerTot.Stop();
    mTimerTot.Reset();
  }

  print();
}
//______________________________________________
void MatchTOF::attachInputTrees()
{
  ///< attaching the input tree
  LOG(DEBUG) << "attachInputTrees";
  if (!mInputTreeTracks) {
    LOG(FATAL) << "Input tree with tracks is not set";
  }

  if (!mTreeTOFClusters) {
    LOG(FATAL) << "TOF clusters data input tree is not set";
  }

  // input tracks (this is the pairs of ITS-TPC matches)

  if (!mInputTreeTracks->GetBranch(mTracksBranchName.data())) {
    LOG(FATAL) << "Did not find tracks branch " << mTracksBranchName << " in the input tree";
  }
  mInputTreeTracks->SetBranchAddress(mTracksBranchName.data(), &mITSTPCTracksArrayInpVect);
  LOG(INFO) << "Attached tracks " << mTracksBranchName << " branch with " << mInputTreeTracks->GetEntries()
            << " entries";

  // input TOF clusters

  if (!mTreeTOFClusters->GetBranch(mTOFClusterBranchName.data())) {
    LOG(FATAL) << "Did not find TOF clusters branch " << mTOFClusterBranchName << " in the input tree";
  }
  mTreeTOFClusters->SetBranchAddress(mTOFClusterBranchName.data(), &mTOFClustersArrayInpVect);
  LOG(INFO) << "Attached TOF clusters " << mTOFClusterBranchName << " branch with " << mTreeTOFClusters->GetEntries()
            << " entries";
  // is there MC info available ?
  mMCTruthON = true;
  if (mTreeTOFClusters->GetBranch(mTOFMCTruthBranchName.data())) {
    mTOFClusLabelsPtr = &mTOFClusLabels;
    mTreeTOFClusters->SetBranchAddress(mTOFMCTruthBranchName.data(), &mTOFClusLabelsPtr);
    LOG(INFO) << "Found TOF Clusters MCLabels branch " << mTOFMCTruthBranchName;
  } else {
    mMCTruthON = false;
  }
  if (mInputTreeTracks->GetBranch(mTPCMCTruthBranchName.data())) {
    mInputTreeTracks->SetBranchAddress(mTPCMCTruthBranchName.data(), &(mTPCLabelsVect[trkType::ITSTPC]));
    LOG(INFO) << "Found TPC tracks MCLabels branch " << mTPCMCTruthBranchName.data();
  } else {
    mMCTruthON = false;
  }

  mCurrTracksTreeEntry = -1;
  mCurrTOFClustersTreeEntry = -1;
}
//______________________________________________
void MatchTOF::attachInputTreesTPConly()
{
  ///< attaching the input tree
  LOG(DEBUG) << "attachInputTrees";

  if (!mTreeTPCTracks) {
    LOG(FATAL) << "TPC tracks data input tree is not set";
  }

  if (!mTreeTOFClusters) {
    LOG(FATAL) << "TOF clusters data input tree is not set";
  }

  // input tracks (this is the TPC tracks)

  if (!mTreeTPCTracks->GetBranch(mTPCTracksBranchName.data())) {
    LOG(FATAL) << "Did not find tracks branch " << mTPCTracksBranchName << " in the input tree";
  }
  mTreeTPCTracks->SetBranchAddress(mTPCTracksBranchName.data(), &mTPCTracksArrayInpVect);
  LOG(INFO) << "Attached tracks " << mTPCTracksBranchName << " branch with " << mTreeTPCTracks->GetEntries()
            << " entries";

  // input TOF clusters

  if (!mTreeTOFClusters->GetBranch(mTOFClusterBranchName.data())) {
    LOG(FATAL) << "Did not find TOF clusters branch " << mTOFClusterBranchName << " in the input tree";
  }
  mTreeTOFClusters->SetBranchAddress(mTOFClusterBranchName.data(), &mTOFClustersArrayInpVect);
  LOG(INFO) << "Attached TOF clusters " << mTOFClusterBranchName << " branch with " << mTreeTOFClusters->GetEntries()
            << " entries";
  // is there MC info available ?
  mMCTruthON = true;
  if (mTreeTOFClusters->GetBranch(mTOFMCTruthBranchName.data())) {
    mTOFClusLabelsPtr = &mTOFClusLabels;
    mTreeTOFClusters->SetBranchAddress(mTOFMCTruthBranchName.data(), &mTOFClusLabelsPtr);
    LOG(INFO) << "Found TOF Clusters MCLabels branch " << mTOFMCTruthBranchName;
  } else {
    mMCTruthON = false;
  }
  if (mTreeTPCTracks->GetBranch(mOutTPCTrackMCTruthBranchName.data())) {
    mTreeTPCTracks->SetBranchAddress(mOutTPCTrackMCTruthBranchName.data(), &(mTPCLabelsVect[trkType::TPC]));
    LOG(INFO) << "Found TPC tracks MCLabels branch " << mOutTPCTrackMCTruthBranchName;
  } else {
    mMCTruthON = false;
  }

  mCurrTracksTreeEntry = -1;
  mCurrTOFClustersTreeEntry = -1;
}
//______________________________________________
bool MatchTOF::prepareTracks()
{
  ///< prepare the tracks that we want to match to TOF
  if (!mIsworkflowON && !loadTracksNextChunk()) {
    return false;
  }

  return MatchTOFBase::prepareTracks();
}
//______________________________________________
bool MatchTOF::prepareTPCTracks()
{
  ///< prepare the tracks that we want to match to TOF

  if (!mIsworkflowON && !loadTPCTracksNextChunk()) {
    return false;
  }

  return MatchTOFBase::prepareTPCTracks();
}
//______________________________________________
bool MatchTOF::prepareTOFClusters()
{
  ///< prepare the tracks that we want to match to TOF

  // copy the track params, propagate to reference X and build sector tables
  mTOFClusWork.clear();
  //  mTOFClusWork.reserve(mNumOfClusters); // we cannot do this, we don't have mNumOfClusters yet
  //  if (mMCTruthON) {
  //    mTOFClusLblWork.clear();
  //    mTOFClusLblWork.reserve(mNumOfClusters);
  //  }

  for (int sec = o2::constants::math::NSectors; sec--;) {
    mTOFClusSectIndexCache[sec].clear();
    //mTOFClusSectIndexCache[sec].reserve(100 + 1.2 * mNumOfClusters / o2::constants::math::NSectors); // we cannot do this, we don't have mNumOfClusters yet
  }

  mNumOfClusters = 0;
  while (!mIsworkflowON && loadTOFClustersNextChunk()) {
    int nClusterInCurrentChunk = mTOFClustersArrayInp.size();
    LOG(DEBUG) << "nClusterInCurrentChunk = " << nClusterInCurrentChunk;
    mNumOfClusters += nClusterInCurrentChunk;
    for (int it = 0; it < nClusterInCurrentChunk; it++) {
      const o2::tof::Cluster& clOrig = mTOFClustersArrayInp[it];
      // create working copy of track param
      mTOFClusWork.emplace_back(clOrig);
      auto& cl = mTOFClusWork.back();
      cl.setEntryInTree(mCurrTOFClustersTreeEntry);
      // cache work track index
      mTOFClusSectIndexCache[cl.getSector()].push_back(mTOFClusWork.size() - 1);
    }
  }

  if (mIsworkflowON) {
    int nClusterInCurrentChunk = mTOFClustersArrayInp.size();
    LOG(DEBUG) << "nClusterInCurrentChunk = " << nClusterInCurrentChunk;
    mNumOfClusters += nClusterInCurrentChunk;
    for (int it = 0; it < nClusterInCurrentChunk; it++) {
      const o2::tof::Cluster& clOrig = mTOFClustersArrayInp[it];
      // create working copy of track param
      mTOFClusWork.emplace_back(clOrig);
      auto& cl = mTOFClusWork.back();
      cl.setEntryInTree(mCurrTOFClustersTreeEntry);
      // cache work track index
      mTOFClusSectIndexCache[cl.getSector()].push_back(mTOFClusWork.size() - 1);
    }
  }

  // sort clusters in each sector according to their time (increasing in time)
  for (int sec = o2::constants::math::NSectors; sec--;) {
    auto& indexCache = mTOFClusSectIndexCache[sec];
    LOG(INFO) << "Sorting sector" << sec << " | " << indexCache.size() << " TOF clusters";
    if (!indexCache.size()) {
      continue;
    }
    std::sort(indexCache.begin(), indexCache.end(), [this](int a, int b) {
      auto& clA = mTOFClusWork[a];
      auto& clB = mTOFClusWork[b];
      return (clA.getTime() - clB.getTime()) < 0.;
    });
  } // loop over TOF clusters of single sector

  if (mMatchedClustersIndex) {
    delete[] mMatchedClustersIndex;
  }
  mMatchedClustersIndex = new int[mNumOfClusters];
  std::fill_n(mMatchedClustersIndex, mNumOfClusters, -1); // initializing all to -1

  return true;
}

//_____________________________________________________
bool MatchTOF::loadTracksNextChunk()
{
  ///< load next chunk of tracks to be matched to TOF
  while (++mCurrTracksTreeEntry < mInputTreeTracks->GetEntries()) {
    mInputTreeTracks->GetEntry(mCurrTracksTreeEntry);
    mITSTPCTracksArrayInp = gsl::span<const o2::dataformats::TrackTPCITS>{*mITSTPCTracksArrayInpVect};
    LOG(INFO) << "Loading tracks entry " << mCurrTracksTreeEntry << " -> " << mITSTPCTracksArrayInp.size()
              << " tracks";
    if (!mITSTPCTracksArrayInp.size()) {
      continue;
    }
    if (mMCTruthON) {
      mTPCLabels[trkType::ITSTPC] = gsl::span<const o2::MCCompLabel>{*(mTPCLabelsVect[trkType::ITSTPC])};
    }
    return true;
  }
  --mCurrTracksTreeEntry;
  return false;
}
//_____________________________________________________
bool MatchTOF::loadTPCTracksNextChunk()
{
  ///< load next chunk of tracks to be matched to TOF
  while (++mCurrTracksTreeEntry < mTreeTPCTracks->GetEntries()) {
    mTreeTPCTracks->GetEntry(mCurrTracksTreeEntry);
    mTPCTracksArrayInp = gsl::span<const o2::tpc::TrackTPC>{*mTPCTracksArrayInpVect};
    LOG(INFO) << "Loading TPC tracks entry " << mCurrTracksTreeEntry << " -> " << mTPCTracksArrayInp.size()
              << " tracks";
    if (!mTPCTracksArrayInp.size()) {
      continue;
    }
    return true;
  }
  --mCurrTracksTreeEntry;
  return false;
}
//______________________________________________
bool MatchTOF::loadTOFClustersNextChunk()
{
  LOG(DEBUG) << "Loat clusters next chunck";
  ///< load next chunk of clusters to be matched to TOF
  LOG(DEBUG) << "Loading TOF clusters: number of entries in tree = " << mTreeTOFClusters->GetEntries();
  while (++mCurrTOFClustersTreeEntry < mTreeTOFClusters->GetEntries()) {
    mTreeTOFClusters->GetEntry(mCurrTOFClustersTreeEntry);
    mTOFClustersArrayInp = gsl::span<const o2::tof::Cluster>{*mTOFClustersArrayInpVect};
    LOG(DEBUG) << "Loading TOF clusters entry " << mCurrTOFClustersTreeEntry << " -> " << mTOFClustersArrayInp.size()
               << " clusters";
    LOG(INFO) << "Loading TOF clusters entry " << mCurrTOFClustersTreeEntry << " -> " << mTOFClustersArrayInp.size()
              << " clusters";
    if (!mTOFClustersArrayInp.size()) {
      continue;
    }
    return true;
  }
  --mCurrTOFClustersTreeEntry;
  return false;
}
