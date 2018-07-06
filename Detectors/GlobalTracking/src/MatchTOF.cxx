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
using evIdx = o2::dataformats::EvIndex<int, int>;

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
    selectBestMatches();
  }
  if (0) { // enabling this creates very verbose output
    mTimerTot.Stop();
    printCandidatesTOF();
    mTimerTot.Start(false);
  }

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

  if (!mTreeTPCTracks) {
    LOG(FATAL) << "TPC tracks data input tree is not set" << FairLogger::endl;
  }

  if (!mTreeTOFClusters) {
    LOG(FATAL) << "TOF clusters data input tree is not set" << FairLogger::endl;
  }

  // input tracks (this is the pais of ITS-TPC matches)

  if (!mInputTreeTracks->GetBranch(mTracksBranchName.data())) {
    LOG(FATAL) << "Did not find tracks branch " << mTracksBranchName << " in the input tree" << FairLogger::endl;
  }
  mInputTreeTracks->SetBranchAddress(mTracksBranchName.data(), &mTracksArrayInp);
  LOG(INFO) << "Attached tracks " << mTracksBranchName << " branch with " << mInputTreeTracks->GetEntries()
            << " entries" << FairLogger::endl;

  // actual TPC tracks
  
  if (!mTreeTPCTracks->GetBranch(mTPCTrackBranchName.data())) {
    LOG(FATAL) << "Did not find TPC tracks branch " << mTPCTrackBranchName << " in the input tree" << FairLogger::endl;
  }
  mTreeTPCTracks->SetBranchAddress(mTPCTrackBranchName.data(), &mTPCTracksArrayInp);
  LOG(INFO) << "Attached TPC tracks " << mTPCTrackBranchName << " branch with " << mTreeTPCTracks->GetEntries()
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

  mNumOfTracks = mTracksArrayInp->size();
  if (mNumOfTracks == 0) return; // no tracks to be matched
  if (mMatchedTracksIndex) delete[] mMatchedTracksIndex;
  mMatchedTracksIndex = new int[mNumOfTracks];
  std::fill_n(mMatchedTracksIndex, mNumOfTracks, -1); // initializing all to -1
  
  // copy the track params, propagate to reference X and build sector tables
  mTracksWork.clear();
  mTracksWork.reserve(mNumOfTracks);
  if (mMCTruthON) {
    mTracksLblWork.clear();
    mTracksLblWork.reserve(mNumOfTracks);
  }
  for (int sec = o2::constants::math::NSectors; sec--;) {
    mTracksSectIndexCache[sec].clear();
    mTracksSectIndexCache[sec].reserve(100 + 1.2 * mNumOfTracks / o2::constants::math::NSectors);
  }

  for (int it = 0; it < mNumOfTracks; it++) {
    o2::dataformats::TrackTPCITS& trcOrig = (*mTracksArrayInp)[it]; // TODO: check if we cannot directly use the o2::track::TrackParCov class instead of o2::dataformats::TrackTPCITS, and then avoid the casting below; this is the track at the vertex

    evIdx evIdxTPC = trcOrig.getRefTPC();
    int indTPC = evIdx.getIndex();
    o2::TPC::TrackTPC& trcTPCOrig = (*mTPCTracksArrayInp)[indTPC]; // we take the track when it is propagated out

    // create working copy of track param
    mTracksWork.emplace_back(static_cast<o2::track::TrackParCov&>(trcTPCOrig), mCurrTracksTreeEntry, it);
    auto& trc = mTracksWork.back();
    // propagate to matching Xref
    if (!propagateToRefX(trc, mXRef, 2)) {
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
	return ((trcA.getTimeMUS().getTimeStamp() - mSigmaTimeCut*trcA.getTimeMUS().getTimeStampError()) - (trcB.getTimeMUS().getTimeStamp() - mSigmaTimeCut*trcB.getTimeMUS().getTimeStampError()) < 0.;
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

  mNumOfClusters = mTOFClustersArrayInp->size();
  if (mNumOfClusters == 0) return; // no clusters to be matched
  if (mMatchedClustersIndex) delete[] mMatchedClustersIndex;
  mMatchedClustersIndex = new int[mNumOfClusters];
  std::fill_n(mMatchedClustersIndex, mNumOfClusters, -1); // initializing all to -1

  // copy the track params, propagate to reference X and build sector tables
  mTOFClusWork.clear();
  mTOFClusWork.reserve(mNumOfClusters);
  if (mMCTruthON) {
    mTOFClusLblWork.clear();
    mTOFClusLblWork.reserve(mNumOfClusters);
  }
  for (int sec = o2::constants::math::NSectors; sec--;) {
    mTOFClusSectIndexCache[sec].clear();
    mTOFClusSectIndexCache[sec].reserve(100 + 1.2 * mNumOfClusters / o2::constants::math::NSectors);
  }
  
  for (int it = 0; it < mNumOfClusters; it++) {
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
  ///< do the real matching per sector

  mMatchedTracksPairs.clear(); // new sector
  
  auto& cacheTOF = mTOFClusSectIndexCache[sec];   // array of cached TOF cluster indices for this sector; reminder: they are ordered in time!
  auto& cacheTrk = mTracksSectIndexCache[sec];   // array of cached tracks indices for this sector; reminder: they are ordered in time!
  int nTracks = cacheTrk.size(), nTOFCls = cacheTOF.size();
  if (!nTracks || !nTOFCls) {
    LOG(INFO) << "Matching sector " << sec << " : N tracks:" << nTracks << " TOF:" << nTOFCls << " in sector "
              << sec << FairLogger::endl;
    return;
  }
  int itof0 = 0; // starting index in TOF clusters for matching of the track
  int   detId[2][5]; // at maximum one track can fall in 2 strips during the propagation; the second dimention of the array is the TOF det index
  float deltaPos[2][3]; // at maximum one track can fall in 2 strips during the propagation; the second dimention of the array is the residuals
  for (int itrk = 0; itrk < cacheTrk.size(); itrk++) {
    int nPropagatedInStrip = 0; // how many strips were hit during the propagation
    auto& trefTrk = mTracksWork[cacheTrk[itrk]];
    float minTrkTime = (trefTrk.getTimeMUS().getTimeStamp() - mSigmaTimeCut*trefTrk.getTimeMUS().getTimeStampError())*1.E6; // minimum time in ps
    float maxTrkTime = (trefTrk.getTimeMUS().getTimeStamp() + mSigmaTimeCut*trefTrk.getTimeMUS().getTimeStampError())*1.E6; // maximum time in ps
    int istep = 1; // number of steps
    float step = 0.1; // step size in cm
    while (propagateToRefX(trefTrk, mXRef+istep*step, step) && nPropagatedInStrip <2 && mXRef+istep*step < Geo::RMAX){
      float pos[3] = {trefTrk.getX(), trefTrk.getY(), trefTrk.getZ()};
      Geo::getPadDxDyDz(pos, detId[nPropagatedInStrip], deltaPos[nPropagatedInStrip]);
      // check if after the propagation we are in a TOF strip
      if (detId[2] != -1) { // we ended in a TOF strip
	nPropagatedInStrip++;
      }
    }
    if (nPropagatedInStrip == 0) continue; // the track never hit a TOF strip during the propagation
    for (auto itof = itof0; itof < nTOFCls; itof++) {
      auto& trefTOF = mTOFClusWork[cacheTOF[itof]];
      // compare the times of the track and the TOF clusters - remember that they both are ordered in time!
      if (trefTOF.getTime() < minTrkTime) {
	itof0 = itof;
	continue;
      }
      if (trefTOF.getTime() > maxTrkTime) { // no more TOF clusters can be matched to this track
	break;
      }
      int mainChannel = trefTOF.getMainContributingChannel();
      int indices[5];
      Geo::getVolumeIndices(mainChannel, indices);
      for (auto iPropagation = 0; iPropagation < nPropagatedInStrip; iPropagation++){
	if (indices[0] != detId[iPropagation][0]) continue;
	if (indices[1] != detId[iPropagation][1]) continue;
	if (indices[2] != detId[iPropagation][2]) continue;
	float resX = deltaPos[iPropagation][0] - (indices[4] - detId[iPropagation][4])*Geo::XPAD; // readjusting the residuals due to the fact that the propagation fell in a pad that was not exactly the one of the cluster
	float resZ = deltaPos[iPropagation][2] - (indices[3] - detId[iPropagation][3])*Geo::ZPAD; // readjusting the residuals due to the fact that the propagation fell in a pad that was not exactly the one of the cluster
	float res = TMath::Sqrt(resX*resX + resZ*resZ);
	float chi2 = res; // TODO: take into account also the time!
	if (res < mSpaceTolerance) { // matching ok!
	  mMatchedTracksPairs.push_back(std::make_pair(itrk, MatchInfoTOF(itof, chi2))); // TODO: check if this is correct!
	}
      }
    }
  }
  return;
}

//______________________________________________
void MatchTOF::selectBestMatches()
{
  ///< define the track-TOFcluster pair per sector

  // first, we sort according to the chi2
  std::sort(mMatchedTracksPairs.begin(), mMatchedTracksPairs.end(), [this](std::pair<int, o2::dataformats::MatchInfoTOF> a, std::pair<int, o2::dataformats::MatchInfoTOF> b) {
      return (a.second.getChi2() < b.second.getChi2());});
  for (auto matchingPair = mMatchedTracksPairs.begin(); matchingPair!= mMatchedTracksPairs.end(); matchingPair++) {
    if (mMatchedTracksIndex[matchingPair.first] != -1) { // the track was already filled
      continue;
    }
    if (mMatchedClustersIndex[matchingPair.second.getTOFClIndex()] != -1) { // the track was already filled
      continue;
    }
    mMatchedTracksIndex[matchingPair.first] = mMatchedTracks.size(); // index of the MatchInfoTOF correspoding to this track
    mMatchedClustersIndex[matchingPair.second.getTOFClIndex()] = mMatchedTracksIndex[matchingPair.first]; // index of the track that was matched to this cluster
    mMatchedTracks.pushBack(matchingPair.second); // array of MatchInfoTOF
  }
 
}

//______________________________________________
    bool MatchTPCITS::propagateToRefX(o2::track::TrackParCov& trc, float xRef, float stepInCm)
{
  // propagate track to matching reference X
  bool refReached = false;
  
  while (o2::Base::Propagator::Instance()->PropagateToXBxByBz(trc, xRef, o2::constants::physics::MassPionCharged, MaxSnp, stepInCm, 0.)) {
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
