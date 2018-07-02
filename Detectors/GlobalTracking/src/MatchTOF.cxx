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
using timeEst = o2::dataformats::TimeStampWithError<float, float>;

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
  prepareTOFClusters();
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
    mTreeTOFClusters->SetBranchAddress(mTOFMCTruthBranchName.data(), &mTOFClusLabels);
    LOG(INFO) << "Found TOF Clusters MCLabels branch " << mTOFMCTruthBranchName << FairLogger::endl;
  }

  mMCTruthON = mTOFClusLabels;
  mCurrTracksTreeEntry = -1;
  mCurrTOFClustersTreeEntry = -1;
}

//______________________________________________
bool MatchTOF::prepareTracks()
{
  ///< prepare the tracks that we want to match to TOF

  if (!loadTracksNextChunk())
  {
    return false;
  }

  int ntr = mTracksArrayInp->size();

  // copy the track params, propagate to reference X and build sector tables
  mTracksWork.clear();
  mTracksWork.reserve(ntr);
  if (mMCTruthON) {
    mTracksLblWork.clear();
    mTracksLblWork.reserve(ntr);
  }
  
  for (int it = 0; it < ntr; it++) {
    o2::dataformats::TrackTPCITS& trcOrig = (*mTracksArrayInp)[it];

    // create working copy of track param
    mTracksWork.emplace_back(static_cast<o2::track::TrackParCov&>(trcOrig), mCurrTPCTracksTreeEntry, it);
    auto& trc = mTracksWork.back();
    // propagate to matching Xref
    if (!propagateToRefX(trc)) {
      mTracksWork.pop_back(); // discard track whose propagation to mXRef failed
      continue;
    }
    if (mMCTruthON) {
      mTracksLblWork.emplace_back(mTracksLabels->getLabels(it)[0]);
    }
    // cache work track index
    mTracksSectIndexCache[o2::utils::Angle2Sector(trc.getAlpha())].push_back(mTracksWork.size() - 1);
  }

  // sort tracks in each sector according to their time (increasing in time)
  for (int sec = o2::constants::math::NSectors; sec--;) {
    auto& indexCache = mTracksSectIndexCache[sec];
    LOG(INFO) << "Sorting sector" << sec << " | " << indexCache.size() << " tracks" << FairLogger::endl;
    if (!indexCache.size())
      continue;
    std::sort(indexCache.begin(), indexCache.end(), [this](int a, int b) {
	auto& trcA = mTracksWork[a];
	auto& trcB = mTracksWork[b];
	return (trcA.getTimeMUS() - trcB.getTimeMUS()) < 0.;
      });
  } // loop over tracks of single sector

  return true;
}
//______________________________________________
bool MatchTOF::prepareTOFClusters()
{
  ///< prepare the tracks that we want to match to TOF

  if (!loadTOFClustersNextChunk())
  {
    return false;
  }

  int ntr = mTOFClustersArrayInp->size();

  // copy the track params, propagate to reference X and build sector tables
  mTOFClusWork.clear();
  mTOFClusWork.reserve(ntr);
  if (mMCTruthON) {
    mTOFClusLblWork.clear();
    mTOFClusLblWork.reserve(ntr);
  }
  
  for (int it = 0; it < ntr; it++) {
    Cluster& clOrig = (*mTOFClustersArrayInp)[it];

    // create working copy of track param
    mTOFClusWork.emplace_back(clOrig), mCurrTOFClustersTreeEntry, it);
    auto& cl = mTOFClusWork.back();
    if (mMCTruthON) {
      mTOFClusLblWork.emplace_back(mTOFClusLabels->getLabels(it)[0]);
    }
    // cache work track index
    mTOFClusSectIndexCache[o2::utils::Angle2Sector(cl.getAlpha())].push_back(mTOFClusWork.size() - 1);
  }

  // sort tracks in each sector according to their time (increasing in time)
  for (int sec = o2::constants::math::NSectors; sec--;) {
    auto& indexCache = mTOFClusSectIndexCache[sec];
    LOG(INFO) << "Sorting sector" << sec << " | " << indexCache.size() << " TOF clusters" << FairLogger::endl;
    if (!indexCache.size())
      continue;
    std::sort(indexCache.begin(), indexCache.end(), [this](int a, int b) {
	auto& clA = mTOFClusWork[a];
	auto& clB = mTOFClusWork[b];
	return (clA.getTime() - clB.getTime()) < 0.;
      });
  } // loop over tracks of single sector

  return true;
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

//______________________________________________
bool MatchTPCITS::propagateToRefX(o2::track::TrackParCov& trc)
{
  // propagate track to matching reference X
  bool refReached = false;
  
  while (o2::Base::Propagator::Instance()->PropagateToXBxByBz(trc, mXRef, o2::constants::physics::MassPionCharged, MaxSnp, 2., 0.)) {
    if (refReached)
      break; // RS: tmp
    // make sure the track is indeed within the sector defined by alpha
    if (fabs(trc.getY()) < mXRef * tan(o2::constants::math::SectorSpanRad / 2)) {
      refReached = true;
      break; // ok, within
    }
    auto alphaNew = o2::utils::Angle2Alpha(trc.getPhiPos());
    if (!trc.rotate(alphaNew) != 0) {
      break; // failed (RS: check effect on matching tracks to neighbouring sector)
    }
  }
  return refReached && std::abs(trc.getSnp()) < MaxSnp;
}
