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
    LOG(FATAL) << "init() was not done yet";
  }

  mTimerTot.Start();

  // we load all TOF clusters (to be checked if we need to split per time frame)
  prepareTOFClusters();

  // we do the matching per entry of the TPCITS matched tracks tree
  while (mCurrTracksTreeEntry+1 < mInputTreeTracks->GetEntries()) { // we add "+1" because mCurrTracksTreeEntry starts from -1, and it is incremented in loadTracksNextChunk which is called by prepareTracks
    Printf("mCurrTracksTreeEntry = %d", mCurrTracksTreeEntry);
    mMatchedTracks.clear();
    mOutTOFLabels.clear();
    mOutTPCLabels.clear();
    mOutITSLabels.clear();
    prepareTracks();

    Printf("*************** Printing the tracks before starting the matching");

    // printing the tracks    
    std::array<float, 3> globalPosTmp;
    int totTracks = 0; 
    /*
    for (int sec = o2::constants::math::NSectors; sec--;) {
      Printf("\nsector %d", sec);
      auto& cacheTrkTmp = mTracksSectIndexCache[sec];   // array of cached tracks indices for this sector; reminder: they are ordered in time!
      for (int itrk = 0; itrk < cacheTrkTmp.size(); itrk++){
	auto& trc = mTracksWork[cacheTrkTmp[itrk]];
	trc.getXYZGlo(globalPosTmp);
	printf("Track %d [in this sector it is the %d]: Global coordinates After propagating to 371 cm: globalPos[0] = %f, globalPos[1] = %f, globalPos[2] = %f\n", totTracks, itrk, globalPosTmp[0], globalPosTmp[1], globalPosTmp[2]);
	Printf("The phi angle is %f", TMath::ATan2(globalPosTmp[1], globalPosTmp[0]));
	totTracks++;
      }
    }
    */
    for (int sec = o2::constants::math::NSectors; sec--;) {
      printf("\n\ndoing matching for sector %i...\n", sec);
      doMatching(sec);
      printf("...done. Now check the best matches\n");
      selectBestMatches();
    }
    if (0) { // enabling this creates very verbose output
      mTimerTot.Stop();
      printCandidatesTOF();
      mTimerTot.Start(false);
    }
    mOutputTree->Fill();
  }
  
#ifdef _ALLOW_DEBUG_TREES_
  mDBGOut.reset();
#endif

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
    LOG(ERROR) << "Initialization was already done";
    return;
  }

  attachInputTrees();

  // create output branch
  if (mOutputTree) {
    mOutputTree->Branch(mOutTracksBranchName.data(), &mMatchedTracks);
    LOG(INFO) << "Matched tracks will be stored in " << mOutTracksBranchName << " branch of tree "
              << mOutputTree->GetName();
    if (mMCTruthON) {
      mOutputTree->Branch(mOutTPCMCTruthBranchName.data(), &mOutITSLabels);
      LOG(INFO) << "ITS Tracks Labels branch: " << mOutITSMCTruthBranchName;
      mOutputTree->Branch(mOutITSMCTruthBranchName.data(), &mOutTPCLabels);
      LOG(INFO) << "TPC Tracks Labels branch: " << mOutTPCMCTruthBranchName;
      mOutputTree->Branch(mOutTOFMCTruthBranchName.data(), &mOutTOFLabels);
      LOG(INFO) << "TOF Tracks Labels branch: " << mOutTOFMCTruthBranchName;
    }

  } else {
    LOG(ERROR) << "Output tree is not attached, matched tracks will not be stored";
  }

#ifdef _ALLOW_DEBUG_TREES_
  // debug streamer
  if (mDBGFlags) {
    mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(mDebugTreeFileName.data(), "recreate");
  }
#endif
  
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
  printf("SigmaTimeCut: %d\n", mSigmaTimeCut);

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
    LOG(FATAL) << "Input tree with tracks is not set";
  }

  if (!mTreeTPCTracks) {
    LOG(FATAL) << "TPC tracks data input tree is not set";
  }

  if (!mTreeTOFClusters) {
    LOG(FATAL) << "TOF clusters data input tree is not set";
  }

  // input tracks (this is the pais of ITS-TPC matches)

  if (!mInputTreeTracks->GetBranch(mTracksBranchName.data())) {
    LOG(FATAL) << "Did not find tracks branch " << mTracksBranchName << " in the input tree";
  }
  mInputTreeTracks->SetBranchAddress(mTracksBranchName.data(), &mTracksArrayInp);
  LOG(INFO) << "Attached tracks " << mTracksBranchName << " branch with " << mInputTreeTracks->GetEntries()
            << " entries";

  // actual TPC tracks
  
  /*  if (!mTreeTPCTracks->GetBranch(mTPCTracksBranchName.data())) {
    LOG(FATAL) << "Did not find TPC tracks branch " << mTPCTracksBranchName << " in the input tree";
  }
  mTreeTPCTracks->SetBranchAddress(mTPCTracksBranchName.data(), &mTPCTracksArrayInp);
  LOG(INFO) << "Attached TPC tracks " << mTPCTracksBranchName << " branch with " << mTreeTPCTracks->GetEntries()
            << " entries";
  */
  // input TOF clusters

  if (!mTreeTOFClusters->GetBranch(mTOFClusterBranchName.data())) {
    LOG(FATAL) << "Did not find TOF clusters branch " << mTOFClusterBranchName << " in the input tree"
              ;
  }
  mTreeTOFClusters->SetBranchAddress(mTOFClusterBranchName.data(), &mTOFClustersArrayInp);
  LOG(INFO) << "Attached TOF clusters " << mTOFClusterBranchName << " branch with " << mTreeTOFClusters->GetEntries()
            << " entries";

  // is there MC info available ?
  if (mTreeTOFClusters->GetBranch(mTOFMCTruthBranchName.data())) {
    mTreeTOFClusters->SetBranchAddress(mTOFMCTruthBranchName.data(), &mTOFClusLabels);
    LOG(INFO) << "Found TOF Clusters MCLabels branch " << mTOFMCTruthBranchName;
  }
  if (mInputTreeTracks->GetBranch(mTPCMCTruthBranchName.data())) {
    mInputTreeTracks->SetBranchAddress(mTPCMCTruthBranchName.data(), &mTPCLabels);
    LOG(INFO) << "Found TPC tracks MCLabels branch " << mTPCMCTruthBranchName.data();
  }
  if (mInputTreeTracks->GetBranch(mITSMCTruthBranchName.data())) {
    mInputTreeTracks->SetBranchAddress(mITSMCTruthBranchName.data(), &mITSLabels);
    LOG(INFO) << "Found ITS tracks MCLabels branch " << mITSMCTruthBranchName.data();
  }
  
  mMCTruthON = (mTOFClusLabels && mTPCLabels && mITSLabels);
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
  if (mNumOfTracks == 0) return false; // no tracks to be matched
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

  Printf("\n\nWe have %d tracks to try to match to TOF", mNumOfTracks);
  int nNotPropagatedToTOF = 0;
  for (int it = 0; it < mNumOfTracks; it++) {
    o2::dataformats::TrackTPCITS& trcOrig = (*mTracksArrayInp)[it]; // TODO: check if we cannot directly use the o2::track::TrackParCov class instead of o2::dataformats::TrackTPCITS, and then avoid the casting below; this is the track at the vertex
    //o2::track::TrackParCov& trcOrig = (*mTracksArrayInp)[it]; // TODO: check if we cannot directly use the o2::track::TrackParCov class instead of o2::dataformats::TrackTPCITS, and then avoid the casting below; this is the track at the vertex
    /*
    evIdx evIdxTPC = trcOrig.getRefTPC();
    int indTPC = evIdxTPC.getIndex();
    o2::TPC::TrackTPC& trcTPCOrig = (*mTPCTracksArrayInp)[indTPC]; // we take the track when it is propagated out
    */
    std::array<float, 3> globalPos;
    Printf("\nOriginal Track %d: getTimeMUS().getTimeStamp() = %f, getTimeMUS().getTimeStampError() = %f", it, (*mTracksArrayInp)[it].getTimeMUS().getTimeStamp() , (*mTracksArrayInp)[it].getTimeMUS().getTimeStampError()); 
    // create working copy of track param
    mTracksWork.emplace_back(trcOrig);//, mCurrTracksTreeEntry, it);
    Printf("Before checking propagation: mTracksWork size = %d", mTracksWork.size());
    // make a copy of the TPC track that we have to propagate
    //o2::TPC::TrackTPC* trc = new o2::TPC::TrackTPC(trcTPCOrig); // this would take the TPCout track
    auto& trc = mTracksWork.back(); // with this we take the TPCITS track propagated to the vertex
    Printf("Copied Track %d: getTimeMUS().getTimeStamp() = %f, getTimeMUS().getTimeStampError() = %f", it, trc.getTimeMUS().getTimeStamp() , trc.getTimeMUS().getTimeStampError()); 
    // propagate to matching Xref
    trc.getXYZGlo(globalPos);
    printf("Global coordinates Before propagating to 371 cm: globalPos[0] = %f, globalPos[1] = %f, globalPos[2] = %f\n", globalPos[0], globalPos[1], globalPos[2]);
    Printf("Radius xy Before propagating to 371 cm = %f", TMath::Sqrt(globalPos[0]*globalPos[0] + globalPos[1]*globalPos[1]));
    Printf("Radius xyz Before propagating to 371 cm = %f", TMath::Sqrt(globalPos[0]*globalPos[0] + globalPos[1]*globalPos[1] + globalPos[2]*globalPos[2]));  
    //    mTracksSectIndexCache[o2::utils::Angle2Sector( TMath::ATan2(globalPos[1], globalPos[0]))].push_back(mTracksWork.size() - 1);
    if (!propagateToRefX(trc, mXRef, 2) || TMath::Abs(trc.getZ()) > Geo::MAXHZTOF) {
      Printf("The track failed propagation");
      Printf("After checking propagation: mTracksWork size = %d", mTracksWork.size());
      //mTracksWork.pop_back(); // discard track whose propagation to mXRef failed, or those that go beyond TOF in z
      nNotPropagatedToTOF++;
      continue;
    }
    else {
      Printf("The track succeeded propagation");
    }      
    Printf("After checking propagation: mTracksWork size = %d", mTracksWork.size());

    //    if (mMCTruthON) {
    //      mTracksLblWork.emplace_back(mTracksLabels->getLabels(it)[0]);
    //    }
    // cache work track index
    trc.getXYZGlo(globalPos);
    printf("Global coordinates After propagating to 371 cm: globalPos[0] = %f, globalPos[1] = %f, globalPos[2] = %f\n", globalPos[0], globalPos[1], globalPos[2]);
    Printf("Radius xy After propagating to 371 cm = %f", TMath::Sqrt(globalPos[0]*globalPos[0] + globalPos[1]*globalPos[1]));
    Printf("Radius xyz After propagating to 371 cm = %f", TMath::Sqrt(globalPos[0]*globalPos[0] + globalPos[1]*globalPos[1] + globalPos[2]*globalPos[2]));  
    Printf("Before pushing: mTracksWork size = %d", mTracksWork.size());
    Printf("The track will go to sector %d", o2::utils::Angle2Sector(TMath::ATan2(globalPos[1], globalPos[0])));
    //mTracksSectIndexCache[o2::utils::Angle2Sector(TMath::ATan2(globalPos[1], globalPos[0]))].push_back(mTracksWork.size() - 1);
    mTracksSectIndexCache[o2::utils::Angle2Sector(TMath::ATan2(globalPos[1], globalPos[0]))].push_back(it);
    int labelTPC = mTPCLabels->at(it);    
    int labelITS = mITSLabels->at(it);    
    Printf("TPC label of the track = %d", labelTPC);
    Printf("ITS label of the track = %d", labelITS);
    //delete trc; // Check: is this needed?
  }

  LOG(INFO) << "Total number of tracks = " << mNumOfTracks << ", Number of tracks that failed to be propagated to TOF = " << nNotPropagatedToTOF;
  
  // sort tracks in each sector according to their time (increasing in time) 
  for (int sec = o2::constants::math::NSectors; sec--;) {
    auto& indexCache = mTracksSectIndexCache[sec];
    LOG(INFO) << "Sorting sector" << sec << " | " << indexCache.size() << " tracks";
    if (!indexCache.size())
      continue;
    std::sort(indexCache.begin(), indexCache.end(), [this](int a, int b) {
	auto& trcA = mTracksWork[a];
	auto& trcB = mTracksWork[b];
	return ((trcA.getTimeMUS().getTimeStamp() - mSigmaTimeCut*trcA.getTimeMUS().getTimeStampError()) - (trcB.getTimeMUS().getTimeStamp() - mSigmaTimeCut*trcB.getTimeMUS().getTimeStampError()) < 0.);
      });
  } // loop over tracks of single sector

  /*
  // printing the tracks
  std::array<float, 3> globalPos;
  int itmp = 0;
  for (int sec = o2::constants::math::NSectors; sec--;) {
    Printf("sector %d", sec);
    auto& cacheTrk = mTracksSectIndexCache[sec];   // array of cached tracks indices for this sector; reminder: they are ordered in time!
    for (int itrk = 0; itrk < cacheTrk.size(); itrk++){
      itmp++; 
      auto& trc = mTracksWork[cacheTrk[itrk]];
      trc.getXYZGlo(globalPos);
      printf("Track %d: Global coordinates After propagating to 371 cm: globalPos[0] = %f, globalPos[1] = %f, globalPos[2] = %f\n", itrk, globalPos[0], globalPos[1], globalPos[2]);
      //      Printf("The phi angle is %f", TMath::ATan2(globalPos[1], globalPos[0]));
    }
  }
  Printf("we have %d tracks",itmp);      
  */
  return true;
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
    //mTOFClusSectIndexCache[sec].reserve(100 + 1.2 * mNumOfClusters / o2::constants::math::NSectors);
  }
  
  mNumOfClusters = 0; 
  while (loadTOFClustersNextChunk())
  {
    int nClusterInCurrentChunk = mTOFClustersArrayInp->size();
    printf("nClusterInCurrentChunk = %d\n", nClusterInCurrentChunk);
    mNumOfClusters += nClusterInCurrentChunk;
    for (int it = 0; it < nClusterInCurrentChunk; it++) {
      Cluster& clOrig = (*mTOFClustersArrayInp)[it];
      
      // create working copy of track param
      mTOFClusWork.emplace_back(clOrig);//, mCurrTOFClustersTreeEntry, it);
      auto& cl = mTOFClusWork.back();
      //   if (mMCTruthON) {
      //      mTOFClusLblWork.emplace_back(mTOFClusLabels->getLabels(it)[0]);
      //    }
      // cache work track index
      mTOFClusSectIndexCache[o2::utils::Angle2Sector(cl.getPhi())].push_back(mTOFClusWork.size() - 1);
    }
    
  }

  // sort clusters in each sector according to their time (increasing in time)
  for (int sec = o2::constants::math::NSectors; sec--;) {
    auto& indexCache = mTOFClusSectIndexCache[sec];
    LOG(INFO) << "Sorting sector" << sec << " | " << indexCache.size() << " TOF clusters";
    if (!indexCache.size())
      continue;
    std::sort(indexCache.begin(), indexCache.end(), [this](int a, int b) {
	auto& clA = mTOFClusWork[a];
	auto& clB = mTOFClusWork[b];
	return (clA.getTime() - clB.getTime()) < 0.;
      });
  } // loop over TOF clusters of single sector

  if (mMatchedClustersIndex) delete[] mMatchedClustersIndex;
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
    LOG(INFO) << "Loading tracks entry " << mCurrTracksTreeEntry << " -> " << mTracksArrayInp->size()
               << " tracks";
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
  printf("Loading TOF clusters: number of entries in tree = %d\n", mTreeTOFClusters->GetEntries());
  while (++mCurrTOFClustersTreeEntry < mTreeTOFClusters->GetEntries()) {
    mTreeTOFClusters->GetEntry(mCurrTOFClustersTreeEntry);
    LOG(DEBUG) << "Loading TOF clusters entry " << mCurrTOFClustersTreeEntry << " -> " << mTOFClustersArrayInp->size()
               << " clusters";
    LOG(INFO) << "Loading TOF clusters entry " << mCurrTOFClustersTreeEntry << " -> " << mTOFClustersArrayInp->size()
               << " clusters";
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
  
  /*
  // printing the tracks
  std::array<float, 3> globalPosTmp;
  Printf("sector %d", sec);
  auto& cacheTrkTmp = mTracksSectIndexCache[sec];   // array of cached tracks indices for this sector; reminder: they are ordered in time!
  for (int itrk = 0; itrk < cacheTrkTmp.size(); itrk++){
    auto& trc = mTracksWork[cacheTrkTmp[itrk]];
    trc.getXYZGlo(globalPosTmp);
    printf("Track %d: Global coordinates After propagating to 371 cm: globalPos[0] = %f, globalPos[1] = %f, globalPos[2] = %f\n", itrk, globalPosTmp[0], globalPosTmp[1], globalPosTmp[2]);
    Printf("The phi angle is %f", TMath::ATan2(globalPosTmp[1], globalPosTmp[0]));
  }
  */
  auto& cacheTOF = mTOFClusSectIndexCache[sec];   // array of cached TOF cluster indices for this sector; reminder: they are ordered in time!
  auto& cacheTrk = mTracksSectIndexCache[sec];   // array of cached tracks indices for this sector; reminder: they are ordered in time!
  int nTracks = cacheTrk.size(), nTOFCls = cacheTOF.size();
  LOG(INFO) << "Matching sector " << sec << ": number of tracks: " << nTracks << ", number of TOF clusters: " << nTOFCls;
  if (!nTracks || !nTOFCls) {
    return;
  }
  int itof0 = 0; // starting index in TOF clusters for matching of the track
  int   detId[2][5]; // at maximum one track can fall in 2 strips during the propagation; the second dimention of the array is the TOF det index
  float deltaPos[2][3]; // at maximum one track can fall in 2 strips during the propagation; the second dimention of the array is the residuals
  int nStepsInsideSameStrip[2] = {0,0}; // number of propagation steps in the same strip (since we have maximum 2 strips, it has dimention = 2)
  float deltaPosTemp[3];
  std::array<float, 3> pos;
  std::array<float, 3> posBeforeProp;
  float posFloat[3]; 

  Printf("Trying to match %d tracks", cacheTrk.size());
  for (int itrk = 0; itrk < cacheTrk.size(); itrk++) {
    Printf("\n\n\n\n ************ track %d **********", itrk);
    for (int ii = 0; ii < 2; ii++) {
      detId[ii][2] = -1; // before trying to match, we need to inizialize the detId corresponding to the strip number to -1; this is the array that we will use to save the det id of the maximum 2 strips matched 
      nStepsInsideSameStrip[ii] = 0;
    }
    int nStripsCrossedInPropagation = 0; // how many strips were hit during the propagation
    auto& trefTrk = mTracksWork[cacheTrk[itrk]];
    float minTrkTime = (trefTrk.getTimeMUS().getTimeStamp() - mSigmaTimeCut*trefTrk.getTimeMUS().getTimeStampError())*1.E6; // minimum time in ps
    float maxTrkTime = (trefTrk.getTimeMUS().getTimeStamp() + mSigmaTimeCut*trefTrk.getTimeMUS().getTimeStampError())*1.E6; // maximum time in ps
    int istep = 1; // number of steps
    float step = 0.1; // step size in cm
    trefTrk.getXYZGlo(posBeforeProp);
    //float posBeforeProp[3] = {trefTrk.getX(), trefTrk.getY(), trefTrk.getZ()}; // in local ref system
    printf("Global coordinates: posBeforeProp[0] = %f, posBeforeProp[1] = %f, posBeforeProp[2] = %f\n", posBeforeProp[0], posBeforeProp[1], posBeforeProp[2]);
    Printf("Radius xy = %f", TMath::Sqrt(posBeforeProp[0]*posBeforeProp[0] + posBeforeProp[1]*posBeforeProp[1]));
    Printf("Radius xyz = %f", TMath::Sqrt(posBeforeProp[0]*posBeforeProp[0] + posBeforeProp[1]*posBeforeProp[1] + posBeforeProp[2]*posBeforeProp[2]));

    (*mDBGOut) << "propOK"
	       << "track=" << trefTrk << "\n";
    // initializing
    for (int ii = 0; ii < 2; ii++){
      for (int iii = 0; iii < 5; iii++){
	detId[ii][iii] = -1;
      }
    }
    int detIdTemp[5] = {-1, -1, -1, -1, -1}; // TOF detector id at the current propagation point
    while (propagateToRefX(trefTrk, mXRef+istep*step, step) && nStripsCrossedInPropagation <=2 && mXRef+istep*step < Geo::RMAX){
      if (0 && istep%100 == 0){
	printf("istep = %d, currentPosition = %f \n", istep, mXRef+istep*step);
      }
      //float pos[3] = {trefTrk.getX(), trefTrk.getY(), trefTrk.getZ()}; // these are the coordinates in the local ref system
      trefTrk.getXYZGlo(pos);
      //printf("getPadDxDyDz:\n");
      for (int ii = 0; ii < 3; ii++){ // we need to change the type... 
	posFloat[ii] = pos[ii];
      }
      //Printf("posFloat[0] = %f, posFloat[1] = %f, posFloat[2] = %f", posFloat[0], posFloat[1], posFloat[2]);
      //Printf("radius xy = %f", TMath::Sqrt(posFloat[0]*posFloat[0] + posFloat[1]*posFloat[1]));
      //Printf("radius xyz = %f", TMath::Sqrt(posFloat[0]*posFloat[0] + posFloat[1]*posFloat[1] + posFloat[2]*posFloat[2]));
      for (int idet = 0; idet < 5; idet++) detIdTemp[idet] = -1;
      Geo::getPadDxDyDz(posFloat, detIdTemp, deltaPosTemp);
      
      //Printf("detIdTemp[0] = %d, detIdTemp[1] = %d, detIdTemp[2] = %d, detIdTemp[3] = %d, detIdTemp[4] = %d", detIdTemp[0], detIdTemp[1], detIdTemp[2], detIdTemp[3], detIdTemp[4]);
      if (detIdTemp[2] != -1 && nStripsCrossedInPropagation == 0){ // print in case you have a useful propagation
	Printf("*********** We have crossed a strip during propagation!*********");
	//printf("Global coordinates: pos[0] = %f, pos[1] = %f, pos[2] = %f\n", pos[0], pos[1], pos[2]);
	//	printf("detIdTemp[0] = %d, detIdTemp[1] = %d, detIdTemp[2] = %d, detIdTemp[3] = %d, detIdTemp[4] = %d\n", detIdTemp[0], detIdTemp[1], detIdTemp[2], detIdTemp[3], detIdTemp[4]);
	//printf("deltaPosTemp[0] = %f, deltaPosTemp[1] = %f, deltaPosTemp[2] = %f\n", deltaPosTemp[0], deltaPosTemp[1], deltaPosTemp[2]);
      }
      else {
	Printf("*********** We have NOT crossed a strip during propagation!*********");
	printf("Global coordinates: pos[0] = %f, pos[1] = %f, pos[2] = %f\n", pos[0], pos[1], pos[2]);
	printf("detIdTemp[0] = %d, detIdTemp[1] = %d, detIdTemp[2] = %d, detIdTemp[3] = %d, detIdTemp[4] = %d\n", detIdTemp[0], detIdTemp[1], detIdTemp[2], detIdTemp[3], detIdTemp[4]);
	printf("deltaPosTemp[0] = %f, deltaPosTemp[1] = %f, deltaPosTemp[2] = %f\n", deltaPosTemp[0], deltaPosTemp[1], deltaPosTemp[2]);
      }	
	//printf("getPadDxDyDz done\n");
      istep++;
      // check if after the propagation we are in a TOF strip
      if (detIdTemp[2] != -1) { // we ended in a TOF strip
	//	Printf("nStripsCrossedInPropagation = %d, detId[nStripsCrossedInPropagation-1][0] = %d, detIdTemp[0] = %d, detId[nStripsCrossedInPropagation-1][1] = %d, detIdTemp[1] =%d, detId[nStripsCrossedInPropagation-1][2] = %d, detIdTemp[2] = %d", nStripsCrossedInPropagation, detId[nStripsCrossedInPropagation-1][0], detIdTemp[0], detId[nStripsCrossedInPropagation-1][1], detIdTemp[1], detId[nStripsCrossedInPropagation-1][2], detIdTemp[2]);
	if(nStripsCrossedInPropagation == 0 || // we are crossing a strip for the first time...
	   (nStripsCrossedInPropagation >= 1 && (detId[nStripsCrossedInPropagation-1][0] != detIdTemp[0] || detId[nStripsCrossedInPropagation-1][1] != detIdTemp[1] || detId[nStripsCrossedInPropagation-1][2] != detIdTemp[2]))) { // ...or we are crossing a new strip
	  //	  if (nStripsCrossedInPropagation == 0) Printf("We cross a strip for the first time"); 
	  Printf("We are in a TOF strip! %d", detIdTemp[2]);
	   if(nStripsCrossedInPropagation == 2) {
	     break; // we have already matched 2 strips, we cannot match more
	   }
	   nStripsCrossedInPropagation++;
	}
	//Printf("nStepsInsideSameStrip[nStripsCrossedInPropagation-1] = %d", nStepsInsideSameStrip[nStripsCrossedInPropagation-1]);
	if(nStepsInsideSameStrip[nStripsCrossedInPropagation-1] == 0){
	  detId[nStripsCrossedInPropagation-1][0] = detIdTemp[0];
	  detId[nStripsCrossedInPropagation-1][1] = detIdTemp[1];
	  detId[nStripsCrossedInPropagation-1][2] = detIdTemp[2];
	  detId[nStripsCrossedInPropagation-1][3] = detIdTemp[3];
	  detId[nStripsCrossedInPropagation-1][4] = detIdTemp[4];
	  deltaPos[nStripsCrossedInPropagation-1][0] = deltaPosTemp[0];
	  deltaPos[nStripsCrossedInPropagation-1][1] = deltaPosTemp[1];
	  deltaPos[nStripsCrossedInPropagation-1][2] = deltaPosTemp[2];
	  nStepsInsideSameStrip[nStripsCrossedInPropagation-1]++;
	}
	else{ // a further propagation step in the same strip -> update info (we sum up on all matching with strip - we will divide for the number of steps a bit below)
	  deltaPos[nStripsCrossedInPropagation-1][0] += deltaPosTemp[0] + (detIdTemp[4] - detId[nStripsCrossedInPropagation-1][4])*Geo::XPAD; // residual in x
	  deltaPos[nStripsCrossedInPropagation-1][1] += deltaPosTemp[1]; // residual in y
	  deltaPos[nStripsCrossedInPropagation-1][2] += deltaPosTemp[2] + (detIdTemp[3] - detId[nStripsCrossedInPropagation-1][3])*Geo::ZPAD; // residual in z
	  nStepsInsideSameStrip[nStripsCrossedInPropagation-1]++;
	}
      }
    }    
    printf("while done, we propagated track %d in %d strips\n", itrk, nStripsCrossedInPropagation);
    if (nStripsCrossedInPropagation == 0) {
      auto labelTPCNoStripsCrossed = mTPCLabels->at(mTracksSectIndexCache[sec][itrk]);    
      Printf("The current track (index = %d) never crossed a strip", cacheTrk[itrk]);
      Printf("TrackID = %d, EventID = %d, SourceID = %d", labelTPCNoStripsCrossed.getTrackID(), labelTPCNoStripsCrossed.getEventID(), labelTPCNoStripsCrossed.getSourceID());
      printf("Global coordinates: pos[0] = %f, pos[1] = %f, pos[2] = %f\n", pos[0], pos[1], pos[2]);
      printf("detIdTemp[0] = %d, detIdTemp[1] = %d, detIdTemp[2] = %d, detIdTemp[3] = %d, detIdTemp[4] = %d\n", detIdTemp[0], detIdTemp[1], detIdTemp[2], detIdTemp[3], detIdTemp[4]);
      printf("deltaPosTemp[0] = %f, deltaPosTemp[1] = %f, deltaPosTemp[2] = %f\n", deltaPosTemp[0], deltaPosTemp[1], deltaPosTemp[2]);
    }
    Printf("We will check now the %d TOF clusters", nTOFCls);
   
    for(Int_t imatch = 0; imatch < nStripsCrossedInPropagation; imatch++){
      //printf("imatch = %d\n", imatch);
      // we take as residual the average of the residuals along the propagation in the same strip
      deltaPos[imatch][0] /= nStepsInsideSameStrip[imatch]; 
      deltaPos[imatch][1] /= nStepsInsideSameStrip[imatch];
      deltaPos[imatch][2] /= nStepsInsideSameStrip[imatch];
      printf("matched strip %d: deltaPos[0] = %f, deltaPos[1] = %f, deltaPos[2] = %f, residual (x, z) = %f\n", imatch, deltaPos[imatch][0], deltaPos[imatch][1], deltaPos[imatch][2], TMath::Sqrt(deltaPos[imatch][0]*deltaPos[imatch][0] + deltaPos[imatch][2]*deltaPos[imatch][2]));
    }
  
    if (nStripsCrossedInPropagation == 0) {
      continue; // the track never hit a TOF strip during the propagation
    }
    Printf("We will check now the %d TOF clusters", nTOFCls);
    bool foundCluster = false;
    for (auto itof = itof0; itof < nTOFCls; itof++) {
      //      printf("itof = %d\n", itof);
      auto& trefTOF = mTOFClusWork[cacheTOF[itof]];
      // compare the times of the track and the TOF clusters - remember that they both are ordered in time!
      Printf("trefTOF.getTime() = %f, maxTrkTime = %f, minTrkTime = %f", trefTOF.getTime(), maxTrkTime, minTrkTime);
      /* This part is commented out for now, as we don't want to have any check on the time enabled
      if (trefTOF.getTime() < minTrkTime) { // this cluster has a time that is too small for the current track, we will get to the next one
	Printf("In trefTOF.getTime() < minTrkTime");
	itof0 = itof+1; // but for the next track that we will check, we will ignore this cluster (the time is anyway too small)
	//continue;
      }
      if (trefTOF.getTime() > maxTrkTime) { // no more TOF clusters can be matched to this track
	//	break;
      }
      */
      int mainChannel = trefTOF.getMainContributingChannel();
      int indices[5];
      Geo::getVolumeIndices(mainChannel, indices);
      const auto& labelsTOF = mTOFClusLabels->getLabels(mTOFClusSectIndexCache[indices[0]][itof]);
      int trackIdTOF;
      int eventIdTOF;
      int sourceIdTOF;
      for (auto iPropagation = 0; iPropagation < nStripsCrossedInPropagation; iPropagation++){
	printf("iPropagation = %d\n", iPropagation);
	Printf("TOF Cluster [%d, %d]:      indices   = %d, %d, %d, %d, %d", itof, cacheTOF[itof], indices[0], indices[1], indices[2], indices[3], indices[4]);
	Printf("Propagated Track [%d, %d]: detId[%d]  = %d, %d, %d, %d, %d", itrk, cacheTrk[itrk], iPropagation, detId[iPropagation][0], detId[iPropagation][1], detId[iPropagation][2], detId[iPropagation][3], detId[iPropagation][4]);
	float resX = deltaPos[iPropagation][0] - (indices[4] - detId[iPropagation][4])*Geo::XPAD; // readjusting the residuals due to the fact that the propagation fell in a pad that was not exactly the one of the cluster
	float resZ = deltaPos[iPropagation][2] - (indices[3] - detId[iPropagation][3])*Geo::ZPAD; // readjusting the residuals due to the fact that the propagation fell in a pad that was not exactly the one of the cluster
	float res = TMath::Sqrt(resX*resX + resZ*resZ);
	Printf("resX = %f, resZ = %f, res = %f", resX, resZ, res);
#ifdef _ALLOW_DEBUG_TREES_
	fillTOFmatchTree("match0", cacheTOF[itof], indices[0], indices[1], indices[2], indices[3], indices[4], cacheTrk[itrk], iPropagation, detId[iPropagation][0], detId[iPropagation][1], detId[iPropagation][2], detId[iPropagation][3], detId[iPropagation][4], resX, resZ, res, trefTrk); 
#endif
	int tofLabelTrackID[3] = {-1, -1, -1};
	int tofLabelEventID[3] = {-1, -1, -1};
	int tofLabelSourceID[3] = {-1, -1, -1};
	for (int ilabel = 0; ilabel < labelsTOF.size(); ilabel++){
	  tofLabelTrackID[ilabel] = labelsTOF[ilabel].getTrackID();
	  tofLabelEventID[ilabel] = labelsTOF[ilabel].getEventID();
	  tofLabelSourceID[ilabel] = labelsTOF[ilabel].getSourceID();
	}
	auto labelTPC = mTPCLabels->at(mTracksSectIndexCache[indices[0]][itrk]);    
	auto labelITS = mITSLabels->at(mTracksSectIndexCache[indices[0]][itrk]);    
	fillTOFmatchTreeWithLabels("matchPossibleWithLabels", cacheTOF[itof], indices[0], indices[1], indices[2], indices[3], indices[4], cacheTrk[itrk], iPropagation, detId[iPropagation][0], detId[iPropagation][1], detId[iPropagation][2], detId[iPropagation][3], detId[iPropagation][4], resX, resZ, res, trefTrk, labelTPC.getTrackID(), labelTPC.getEventID(), labelTPC.getSourceID(), labelITS.getTrackID(), labelITS.getEventID(), labelITS.getSourceID(), tofLabelTrackID[0], tofLabelEventID[0], tofLabelSourceID[0], tofLabelTrackID[1], tofLabelEventID[1], tofLabelSourceID[1], tofLabelTrackID[2], tofLabelEventID[2], tofLabelSourceID[2]); 
	if (indices[0] != detId[iPropagation][0]) continue;
	if (indices[1] != detId[iPropagation][1]) continue;
	if (indices[2] != detId[iPropagation][2]) continue;
	float chi2 = res; // TODO: take into account also the time!
	fillTOFmatchTree("match1", cacheTOF[itof], indices[0], indices[1], indices[2], indices[3], indices[4], cacheTrk[itrk], iPropagation, detId[iPropagation][0], detId[iPropagation][1], detId[iPropagation][2], detId[iPropagation][3], detId[iPropagation][4], resX, resZ, res, trefTrk); 

	fillTOFmatchTreeWithLabels("matchOkWithLabels", cacheTOF[itof], indices[0], indices[1], indices[2], indices[3], indices[4], cacheTrk[itrk], iPropagation, detId[iPropagation][0], detId[iPropagation][1], detId[iPropagation][2], detId[iPropagation][3], detId[iPropagation][4], resX, resZ, res, trefTrk, labelTPC.getTrackID(), labelTPC.getEventID(), labelTPC.getSourceID(), labelITS.getTrackID(), labelITS.getEventID(), labelITS.getSourceID(), tofLabelTrackID[0], tofLabelEventID[0], tofLabelSourceID[0], tofLabelTrackID[1], tofLabelEventID[1], tofLabelSourceID[1], tofLabelTrackID[2], tofLabelEventID[2], tofLabelSourceID[2]); 
	
	if (res < mSpaceTolerance) { // matching ok!
	  Printf("YUHUUUUUUUUU! We have a match! between track %d and TOF cluster %d", mTracksSectIndexCache[indices[0]][itrk], mTOFClusSectIndexCache[indices[0]][itof]);
	  foundCluster = true;
	  mMatchedTracksPairs.push_back(std::make_pair(mTracksSectIndexCache[indices[0]][itrk], o2::dataformats::MatchInfoTOF(mTOFClusSectIndexCache[indices[0]][itof], chi2))); // TODO: check if this is correct!
	  for (int ilabel = 0; ilabel < labelsTOF.size(); ilabel++){
	    Printf("TOF label %d: trackID = %d, eventID = %d, sourceID = %d", ilabel, labelsTOF[ilabel].getTrackID(), labelsTOF[ilabel].getEventID(), labelsTOF[ilabel].getSourceID());
	  }
	  Printf("TPC label of the track: trackID = %d, eventID = %d, sourceID = %d", labelTPC.getTrackID(), labelTPC.getEventID(), labelTPC.getSourceID());
	  Printf("ITS label of the track: trackID = %d, eventID = %d, sourceID = %d", labelITS.getTrackID(), labelITS.getEventID(), labelITS.getSourceID());
	  fillTOFmatchTreeWithLabels("matchOkWithLabelsInSpaceTolerance", cacheTOF[itof], indices[0], indices[1], indices[2], indices[3], indices[4], cacheTrk[itrk], iPropagation, detId[iPropagation][0], detId[iPropagation][1], detId[iPropagation][2], detId[iPropagation][3], detId[iPropagation][4], resX, resZ, res, trefTrk, labelTPC.getTrackID(), labelTPC.getEventID(), labelTPC.getSourceID(), labelITS.getTrackID(), labelITS.getEventID(), labelITS.getSourceID(), tofLabelTrackID[0], tofLabelEventID[0], tofLabelSourceID[0], tofLabelTrackID[1], tofLabelEventID[1], tofLabelSourceID[1], tofLabelTrackID[2], tofLabelEventID[2], tofLabelSourceID[2]); 
	}
	else {
	  Printf("We have not matched with any TOF cluster");
	}
      }
    }
    if (!foundCluster) Printf("We did not find any TOF cluster for track %d", cacheTrk[itrk]);
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
  int i = 0;
  // then we take discard the pairs if their track or cluster was already matched (since they are ordered in chi2, we will take the best matching)
  for (const std::pair<int,o2::dataformats::MatchInfoTOF> &matchingPair : mMatchedTracksPairs){
    Printf("selectBestMatches: i = %d", i);
    if (mMatchedTracksIndex[matchingPair.first] != -1) { // the track was already filled
      continue;
    }
    if (mMatchedClustersIndex[matchingPair.second.getTOFClIndex()] != -1) { // the track was already filled
      continue;
    }
    mMatchedTracksIndex[matchingPair.first] = mMatchedTracks.size(); // index of the MatchInfoTOF correspoding to this track
    mMatchedClustersIndex[matchingPair.second.getTOFClIndex()] = mMatchedTracksIndex[matchingPair.first]; // index of the track that was matched to this cluster
    //mMatchedTracks.push_back(matchingPair.second); // array of MatchInfoTOF
    mMatchedTracks.push_back(matchingPair); // array of MatchInfoTOF
    Printf("size of mMatchedTracks = %d", mMatchedTracks.size());
    int labelTPCint = mTPCLabels->at(matchingPair.first);    
    int labelITSint = mITSLabels->at(matchingPair.first);    
    Printf("TPC label of the track = %d", labelTPCint);
    Printf("ITS label of the track = %d", labelITSint);
    const auto& labelTPC = mTPCLabels->at(matchingPair.first);
    Printf("labelTPC: trackID = %d, eventID = %d, sourceID = %d", labelTPC.getTrackID(), labelTPC.getEventID(), labelTPC.getSourceID());
    const auto& labelITS = mITSLabels->at(matchingPair.first);
    Printf("labelITS: trackID = %d, eventID = %d, sourceID = %d", labelITS.getTrackID(), labelITS.getEventID(), labelITS.getSourceID());
    const auto& labelsTOF = mTOFClusLabels->getLabels(matchingPair.second.getTOFClIndex());
    bool labelOk = false; // whether we have found or not the same TPC label of the track among the labels of the TOF cluster
    for (int ilabel = 0; ilabel < labelsTOF.size(); ilabel++){
      Printf("TOF label %d: trackID = %d, eventID = %d, sourceID = %d", ilabel, labelsTOF[ilabel].getTrackID(), labelsTOF[ilabel].getEventID(), labelsTOF[ilabel].getSourceID());
      if (labelsTOF[ilabel].getTrackID() == labelTPC.getTrackID() && labelsTOF[ilabel].getEventID() == labelTPC.getEventID() && labelsTOF[ilabel].getSourceID() == labelTPC.getSourceID() && !labelOk) { // if we find one TOF cluster label that is the same as the TPC one, we are happy - even if it is not the first one
	mOutTOFLabels.push_back(labelsTOF[ilabel]);
	Printf("Adding label for good match %d", mOutTOFLabels.size() );
	labelOk = true;
      }
    }
    if (!labelOk) {
      // we have not found the track label among those associated to the TOF cluster --> fake match! We will associate the label of the main channel, but negative
      o2::MCCompLabel fakeTOFlabel;
      fakeTOFlabel.set(-labelsTOF[0].getTrackID(), labelsTOF[0].getEventID(), labelsTOF[0].getSourceID());
      mOutTOFLabels.push_back(fakeTOFlabel);
      Printf("Adding label for fake match %d", mOutTOFLabels.size());
    }
    mOutTPCLabels.push_back(labelTPC);
    mOutITSLabels.push_back(labelITS);
    i++;
  }
  Printf("size of mOutTPCLabels = %d", mOutTPCLabels.size());
  Printf("size of mOutITSLabels = %d", mOutITSLabels.size());
  Printf("size of mOutTOFLabels = %d", mOutTOFLabels.size());
 
}

//______________________________________________
bool MatchTOF::propagateToRefX(o2::track::TrackParCov& trc, float xRef, float stepInCm)
{
  //  printf("in propagateToRefX\n");
  // propagate track to matching reference X
  bool refReached = false;
  float xStart = trc.getX();
  // the first propagation will be from 2m, if the track is not at least at 2m
  if (xStart < 50.) xStart = 50.;
  int istep = 1;
  bool hasPropagated = o2::Base::Propagator::Instance()->PropagateToXBxByBz(trc, xStart + istep*stepInCm, o2::constants::physics::MassPionCharged, MAXSNP, stepInCm, 0.);
  while (hasPropagated) {
    //Printf("propagateToRefX: istep = %d", istep);
    if (trc.getX() > xRef) {
      refReached = true; // we reached the 371cm reference
      //      Printf("propagateToRefX: trc.getX() > xRef --> refReached = true");
    }
    istep++;
    if (fabs(trc.getY()) > trc.getX() * tan(o2::constants::math::SectorSpanRad / 2)) { // we are still in the same sector
      // we need to rotate the track to go to the new sector
      //Printf("propagateToRefX: changing sector");
      auto alphaNew = o2::utils::Angle2Alpha(trc.getPhiPos());
      if (!trc.rotate(alphaNew) != 0) {
	//	Printf("propagateToRefX: failed to rotate");
	break; // failed (RS: check effect on matching tracks to neighbouring sector)
      }
    }
    if (refReached) break;
    hasPropagated = o2::Base::Propagator::Instance()->PropagateToXBxByBz(trc, xStart + istep*stepInCm, o2::constants::physics::MassPionCharged, MAXSNP, stepInCm, 0.);
  }
  //  Printf("propagateToXRef: hasPropagate(%d) = %d", istep, hasPropagated);
  if (std::abs(trc.getSnp()) > MAXSNP) Printf("propagateToRefX: condition on snp not ok, returning false");
  //Printf("propagateToRefX: final x of the track = %f, refReached at the end is %d", trc.getX(), (int)refReached);
  //Printf("propagateToRefX: snp of teh track is %f (--> %f grad)", trc.getSnp(), TMath::ASin(trc.getSnp())*TMath::RadToDeg());
  return refReached && std::abs(trc.getSnp()) < 0.95;
  //return refReached;
}
      
#ifdef _ALLOW_DEBUG_TREES_
//______________________________________________
void MatchTOF::setDebugFlag(UInt_t flag, bool on)
{
  ///< set debug stream flag
  if (on) {
    mDBGFlags |= flag;
  } else {
    mDBGFlags &= ~flag;
  }
}

//_________________________________________________________
void MatchTOF::fillTOFmatchTree(const char* trname, int cacheTOF, int sectTOF, int plateTOF, int stripTOF, int padXTOF, int padZTOF, int cacheeTrk, int crossedStrip, int sectPropagation, int platePropagation, int stripPropagation, int padXPropagation, int padZPropagation, float resX, float resZ, float res, o2::dataformats::TrackTPCITS &trk)
{
  ///< fill debug tree for TOF tracks matching check

  mTimerDBG.Start(false);

//  Printf("************** Filling the debug tree with %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %f, %f, %f", cacheTOF, sectTOF, plateTOF, stripTOF, padXTOF, padZTOF, cacheeTrk, crossedStrip, sectPropagation, platePropagation, stripPropagation, padXPropagation, padZPropagation, resX, resZ, res);
  (*mDBGOut) << trname
             << "clusterTOF=" << cacheTOF << "sectTOF=" << sectTOF << "plateTOF=" << plateTOF << "stripTOF=" << stripTOF << "padXTOF=" << padXTOF << "padZTOF=" << padZTOF
             << "crossedStrip=" << crossedStrip << "sectPropagation=" << sectPropagation << "platePropagation=" << platePropagation << "stripPropagation=" << stripPropagation << "padXPropagation=" << padXPropagation
             << "resX=" << resX << "resZ=" << resZ << "res=" << res << "track=" << trk << "\n";
  mTimerDBG.Stop();
}

//_________________________________________________________
void MatchTOF::fillTOFmatchTreeWithLabels(const char* trname, int cacheTOF, int sectTOF, int plateTOF, int stripTOF, int padXTOF, int padZTOF, int cacheeTrk, int crossedStrip, int sectPropagation, int platePropagation, int stripPropagation, int padXPropagation, int padZPropagation, float resX, float resZ, float res, o2::dataformats::TrackTPCITS &trk, int TPClabelTrackID, int TPClabelEventID, int TPClabelSourceID, int ITSlabelTrackID, int ITSlabelEventID, int ITSlabelSourceID, int TOFlabelTrackID0, int TOFlabelEventID0, int TOFlabelSourceID0, int TOFlabelTrackID1, int TOFlabelEventID1, int TOFlabelSourceID1, int TOFlabelTrackID2, int TOFlabelEventID2, int TOFlabelSourceID2)
{
  ///< fill debug tree for TOF tracks matching check

  mTimerDBG.Start(false);

  (*mDBGOut) << trname
             << "clusterTOF=" << cacheTOF << "sectTOF=" << sectTOF << "plateTOF=" << plateTOF << "stripTOF=" << stripTOF << "padXTOF=" << padXTOF << "padZTOF=" << padZTOF
             << "crossedStrip=" << crossedStrip << "sectPropagation=" << sectPropagation << "platePropagation=" << platePropagation << "stripPropagation=" << stripPropagation << "padXPropagation=" << padXPropagation
             << "resX=" << resX << "resZ=" << resZ << "res=" << res << "track=" << trk
             << "TPClabelTrackID=" << TPClabelTrackID << "TPClabelEventID=" << TPClabelEventID << "TPClabelSourceID=" << TPClabelSourceID
             << "ITSlabelTrackID=" << ITSlabelTrackID << "ITSlabelEventID=" << ITSlabelEventID << "ITSlabelSourceID=" << ITSlabelSourceID
             << "TOFlabelTrackID0=" << TOFlabelTrackID0 << "TOFlabelEventID0=" << TOFlabelEventID0 << "TOFlabelSourceID0=" << TOFlabelSourceID0
             << "TOFlabelTrackID1=" << TOFlabelTrackID1 << "TOFlabelEventID1=" << TOFlabelEventID1 << "TOFlabelSourceID1=" << TOFlabelSourceID1
             << "TOFlabelTrackID2=" << TOFlabelTrackID2 << "TOFlabelEventID2=" << TOFlabelEventID2 << "TOFlabelSourceID2=" << TOFlabelSourceID2
             << "\n";
  mTimerDBG.Stop();
}
#endif
