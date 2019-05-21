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
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/TrackLTIntegral.h"

#include "GlobalTracking/MatchTOF.h"
#include "GlobalTracking/MatchTPCITS.h"

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

  if(mIsworkflowON){
    LOG(DEBUG) << "Number of entries in track tree = " << mCurrTracksTreeEntry;
    mMatchedTracks.clear();
    mOutTOFLabels.clear();
    mOutTPCLabels.clear();
    mOutITSLabels.clear();
    prepareTracks();

    for (int sec = o2::constants::math::NSectors; sec--;) {
      LOG(INFO) << "Doing matching for sector " << sec << "...";
      doMatching(sec);
      LOG(INFO) << "...done. Now check the best matches";
      selectBestMatches();
    }
    if (0) { // enabling this creates very verbose output
      mTimerTot.Stop();
      printCandidatesTOF();
      mTimerTot.Start(false);
    }
  }

  // we do the matching per entry of the TPCITS matched tracks tree
  while (!mIsworkflowON && mCurrTracksTreeEntry + 1 < mInputTreeTracks->GetEntries()) { // we add "+1" because mCurrTracksTreeEntry starts from -1, and it is incremented in loadTracksNextChunk which is called by prepareTracks
    LOG(DEBUG) << "Number of entries in track tree = " << mCurrTracksTreeEntry;
    mMatchedTracks.clear();
    mOutTOFLabels.clear();
    mOutTPCLabels.clear();
    mOutITSLabels.clear();
    prepareTracks();

    /* Uncomment for local debug 
    Printf("*************** Printing the tracks before starting the matching");

    // printing the tracks    
    std::array<float, 3> globalPosTmp;
    int totTracks = 0; 
    
    for (int sec = o2::constants::math::NSectors; sec--;) {
     Printf("\nsector %d", sec);
      auto& cacheTrkTmp = mTracksSectIndexCache[sec];   // array of cached tracks indices for this sector; reminder: they are ordered in time!
      for (int itrk = 0; itrk < cacheTrkTmp.size(); itrk++){
	auto& trc = mTracksWork[cacheTrkTmp[itrk]];
        trc.getXYZGlo(globalPosTmp);
	LOG(INFO) << "Track" << totTracks << " [in this sector it is the " << itrk << "]: Global coordinates After propagating to 371 cm: globalPos[0] = " << globalPosTmp[0] << ", globalPos[1] = " << globalPosTmp[1] << ", globalPos[2] = " << globalPosTmp[2];
	LOG(INFO) << "The phi angle is " << TMath::ATan2(globalPosTmp[1], globalPosTmp[0]);
	totTracks++;
      }
    }
    */

    for (int sec = o2::constants::math::NSectors; sec--;) {
      LOG(INFO) << "Doing matching for sector " << sec << "...";
      doMatching(sec);
      LOG(INFO) << "...done. Now check the best matches";
      selectBestMatches();
    }
    if (0) { // enabling this creates very verbose output
      mTimerTot.Stop();
      printCandidatesTOF();
      mTimerTot.Start(false);
    }
  }

#ifdef _ALLOW_DEBUG_TREES_
  if (mDBGFlags)
    mDBGOut.reset();
#endif

  mTimerTot.Stop();
  printf("Timing:\n");
  printf("Total:        ");
  mTimerTot.Print();
}

//______________________________________________
void MatchTOF::fill()
{
    mOutputTree->Fill();
    if (mOutputTreeCalib)
      mOutputTreeCalib->Fill();
}

//______________________________________________
void MatchTOF::initWorkflow(const std::vector<o2::dataformats::TrackTPCITS>* trackArray, const std::vector<Cluster>* clusterArray){

  if (mInitDone) {
    LOG(ERROR) << "Initialization was already done";
    return;
  }

  mTracksArrayInp = trackArray;
  mTOFClustersArrayInp = clusterArray;
  mIsworkflowON = kTRUE;
  mInitDone = true;
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

  // create output branch with track-tof matching
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

  // create output branch for calibration info
  if (mOutputTreeCalib) {
    mOutputTreeCalib->Branch(mOutCalibBranchName.data(), &mCalibInfoTOF);
    LOG(INFO) << "Calib infos will be stored in " << mOutCalibBranchName << " branch of tree "
              << mOutputTreeCalib->GetName();
  } else {
    LOG(INFO) << "Calib Output tree is not attached, calib infos will not be stored";
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

  LOG(INFO) << "****** component for the matching of tracks to TOF clusters ******";
  if (!mInitDone) {
    LOG(INFO) << "init is not done yet - nothing to print";
    return;
  }

  LOG(INFO) << "MC truth: " << (mMCTruthON ? "on" : "off");
  LOG(INFO) << "Time tolerance: " << mTimeTolerance;
  LOG(INFO) << "Space tolerance: " << mSpaceTolerance;
  LOG(INFO) << "SigmaTimeCut: " << mSigmaTimeCut;

  LOG(INFO) << "**********************************************************************";
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

  // input TOF clusters

  if (!mTreeTOFClusters->GetBranch(mTOFClusterBranchName.data())) {
    LOG(FATAL) << "Did not find TOF clusters branch " << mTOFClusterBranchName << " in the input tree";
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

  if (!mIsworkflowON && !loadTracksNextChunk()) {
    return false;
  }

  mNumOfTracks = mTracksArrayInp->size();
  if (mNumOfTracks == 0)
    return false; // no tracks to be matched
  mMatchedTracksIndex.resize(mNumOfTracks);
  std::fill(mMatchedTracksIndex.begin(), mMatchedTracksIndex.end(), -1); // initializing all to -1

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

  // getting Bz (mag field)
  auto o2field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
  float bzField = o2field->solenoidField(); // magnetic field in kGauss

  Printf("\n\nWe have %d tracks to try to match to TOF", mNumOfTracks);
  int nNotPropagatedToTOF = 0;
  for (int it = 0; it < mNumOfTracks; it++) {
    const o2::dataformats::TrackTPCITS& trcOrig = (*mTracksArrayInp)[it]; // TODO: check if we cannot directly use the o2::track::TrackParCov class instead of o2::dataformats::TrackTPCITS, and then avoid the casting below; this is the track at the vertex
    std::array<float, 3> globalPos;

    // create working copy of track param
    mTracksWork.emplace_back(trcOrig); //, mCurrTracksTreeEntry, it);
    // make a copy of the TPC track that we have to propagate
    //o2::tpc::TrackTPC* trc = new o2::tpc::TrackTPC(trcTPCOrig); // this would take the TPCout track
    //auto& trc = mTracksWork.back(); // with this we take the TPCITS track propagated to the vertex
    auto& trc = mTracksWork.back().getParamOut();        // with this we take the TPCITS track propagated to the vertex
    auto& intLT = mTracksWork.back().getLTIntegralOut(); // we get the integrated length from TPC-ITC outward propagation

    if (trc.getX() < o2::globaltracking::MatchTPCITS::XTPCOuterRef - 1.) { // tpc-its track outward propagation did not reach outer ref.radius, skip this track
      nNotPropagatedToTOF++;
      continue;
    }

    // propagate to matching Xref
    trc.getXYZGlo(globalPos);
    LOG(DEBUG) << "Global coordinates Before propagating to 371 cm: globalPos[0] = " << globalPos[0] << ", globalPos[1] = " << globalPos[1] << ", globalPos[2] = " << globalPos[2];
    LOG(DEBUG) << "Radius xy Before propagating to 371 cm = " << TMath::Sqrt(globalPos[0] * globalPos[0] + globalPos[1] * globalPos[1]);
    LOG(DEBUG) << "Radius xyz Before propagating to 371 cm = " << TMath::Sqrt(globalPos[0] * globalPos[0] + globalPos[1] * globalPos[1] + globalPos[2] * globalPos[2]);
    if (!propagateToRefXWithoutCov(trc, mXRef, 2, bzField)) { // we first propagate to 371 cm without considering the covariance matrix
      nNotPropagatedToTOF++;
      continue;
    }

    // the "rough" propagation worked; now we can propagate considering also the cov matrix
    if (!propagateToRefX(trc, mXRef, 2, intLT) || TMath::Abs(trc.getZ()) > Geo::MAXHZTOF) { // we check that the propagation with the cov matrix worked; CHECK: can it happen that it does not if the propagation without the errors succeeded?
      nNotPropagatedToTOF++;
      continue;
    }

    trc.getXYZGlo(globalPos);

    LOG(DEBUG) << "Global coordinates After propagating to 371 cm: globalPos[0] = " << globalPos[0] << ", globalPos[1] = " << globalPos[1] << ", globalPos[2] = " << globalPos[2];
    LOG(DEBUG) << "Radius xy After propagating to 371 cm = " << TMath::Sqrt(globalPos[0] * globalPos[0] + globalPos[1] * globalPos[1]);
    LOG(DEBUG) << "Radius xyz After propagating to 371 cm = " << TMath::Sqrt(globalPos[0] * globalPos[0] + globalPos[1] * globalPos[1] + globalPos[2] * globalPos[2]);
    LOG(DEBUG) << "The track will go to sector " << o2::utils::Angle2Sector(TMath::ATan2(globalPos[1], globalPos[0]));

    mTracksSectIndexCache[o2::utils::Angle2Sector(TMath::ATan2(globalPos[1], globalPos[0]))].push_back(it);
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
      return ((trcA.getTimeMUS().getTimeStamp() - mSigmaTimeCut * trcA.getTimeMUS().getTimeStampError()) - (trcB.getTimeMUS().getTimeStamp() - mSigmaTimeCut * trcB.getTimeMUS().getTimeStampError()) < 0.);
    });
  } // loop over tracks of single sector

  // Uncomment for local debug
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
    //mTOFClusSectIndexCache[sec].reserve(100 + 1.2 * mNumOfClusters / o2::constants::math::NSectors); // we cannot do this, we don't have mNumOfClusters yet
  }

  mNumOfClusters = 0;
  while (!mIsworkflowON && loadTOFClustersNextChunk()) {
    int nClusterInCurrentChunk = mTOFClustersArrayInp->size();
    LOG(DEBUG) << "nClusterInCurrentChunk = " << nClusterInCurrentChunk;
    mNumOfClusters += nClusterInCurrentChunk;
    for (int it = 0; it < nClusterInCurrentChunk; it++) {
      const Cluster& clOrig = (*mTOFClustersArrayInp)[it];
      // create working copy of track param
      mTOFClusWork.emplace_back(clOrig);
      auto& cl = mTOFClusWork.back();
      cl.setEntryInTree(mCurrTOFClustersTreeEntry);
      // cache work track index
      mTOFClusSectIndexCache[cl.getSector()].push_back(mTOFClusWork.size() - 1);
    }
  }

  if(mIsworkflowON){
    int nClusterInCurrentChunk = mTOFClustersArrayInp->size();
    LOG(DEBUG) << "nClusterInCurrentChunk = " << nClusterInCurrentChunk;
    mNumOfClusters += nClusterInCurrentChunk;
    for (int it = 0; it < nClusterInCurrentChunk; it++) {
      const Cluster& clOrig = (*mTOFClustersArrayInp)[it];
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
    if (!indexCache.size())
      continue;
    std::sort(indexCache.begin(), indexCache.end(), [this](int a, int b) {
      auto& clA = mTOFClusWork[a];
      auto& clB = mTOFClusWork[b];
      return (clA.getTime() - clB.getTime()) < 0.;
    });
  } // loop over TOF clusters of single sector

  if (mMatchedClustersIndex)
    delete[] mMatchedClustersIndex;
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
bool MatchTOF::loadTOFClustersNextChunk()
{
  ///< load next chunk of clusters to be matched to TOF
  printf("Loading TOF clusters: number of entries in tree = %lld\n", mTreeTOFClusters->GetEntries());
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
void MatchTOF::doMatching(int sec)
{
  ///< do the real matching per sector
  mMatchedTracksPairs.clear(); // new sector

  //uncomment for local debug
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
  auto& cacheTOF = mTOFClusSectIndexCache[sec]; // array of cached TOF cluster indices for this sector; reminder: they are ordered in time!
  auto& cacheTrk = mTracksSectIndexCache[sec];  // array of cached tracks indices for this sector; reminder: they are ordered in time!
  int nTracks = cacheTrk.size(), nTOFCls = cacheTOF.size();
  LOG(INFO) << "Matching sector " << sec << ": number of tracks: " << nTracks << ", number of TOF clusters: " << nTOFCls;
  if (!nTracks || !nTOFCls) {
    return;
  }
  int itof0 = 0;                           // starting index in TOF clusters for matching of the track
  int detId[2][5];                         // at maximum one track can fall in 2 strips during the propagation; the second dimention of the array is the TOF det index
  float deltaPos[2][3];                    // at maximum one track can fall in 2 strips during the propagation; the second dimention of the array is the residuals
  o2::track::TrackLTIntegral trkLTInt[2];  // Here we store the integrated track length and time for the (max 2) matched strips
  int nStepsInsideSameStrip[2] = { 0, 0 }; // number of propagation steps in the same strip (since we have maximum 2 strips, it has dimention = 2)
  float deltaPosTemp[3];
  std::array<float, 3> pos;
  std::array<float, 3> posBeforeProp;
  float posFloat[3];

  LOG(DEBUG) << "Trying to match %d tracks" << cacheTrk.size();
  for (int itrk = 0; itrk < cacheTrk.size(); itrk++) {
    for (int ii = 0; ii < 2; ii++) {
      detId[ii][2] = -1; // before trying to match, we need to inizialize the detId corresponding to the strip number to -1; this is the array that we will use to save the det id of the maximum 2 strips matched
      nStepsInsideSameStrip[ii] = 0;
    }
    int nStripsCrossedInPropagation = 0; // how many strips were hit during the propagation
    auto& trackWork = mTracksWork[cacheTrk[itrk]];
    auto& trefTrk = trackWork.getParamOut();
    auto& intLT = trackWork.getLTIntegralOut();
    Printf("intLT (before doing anything): length = %f, time (Pion) = %f", intLT.getL(), intLT.getTOF(o2::track::PID::Pion));
    float minTrkTime = (trackWork.getTimeMUS().getTimeStamp() - mSigmaTimeCut * trackWork.getTimeMUS().getTimeStampError()) * 1.E6; // minimum time in ps
    float maxTrkTime = (trackWork.getTimeMUS().getTimeStamp() + mSigmaTimeCut * trackWork.getTimeMUS().getTimeStampError()) * 1.E6; // maximum time in ps
    int istep = 1;                                                                                                              // number of steps
    float step = 0.1;                                                                                                           // step size in cm
                                                                                                                                //uncomment for local debug
                                                                                                                                /*
																//trefTrk.getXYZGlo(posBeforeProp);
																//float posBeforeProp[3] = {trefTrk.getX(), trefTrk.getY(), trefTrk.getZ()}; // in local ref system
																//printf("Global coordinates: posBeforeProp[0] = %f, posBeforeProp[1] = %f, posBeforeProp[2] = %f\n", posBeforeProp[0], posBeforeProp[1], posBeforeProp[2]);
																//Printf("Radius xy = %f", TMath::Sqrt(posBeforeProp[0]*posBeforeProp[0] + posBeforeProp[1]*posBeforeProp[1]));
																//Printf("Radius xyz = %f", TMath::Sqrt(posBeforeProp[0]*posBeforeProp[0] + posBeforeProp[1]*posBeforeProp[1] + posBeforeProp[2]*posBeforeProp[2]));
																*/

#ifdef _ALLOW_DEBUG_TREES_
    if (mDBGFlags) {
      (*mDBGOut) << "propOK"
                 << "track=" << trefTrk << "\n";
    }
#endif

    // initializing
    for (int ii = 0; ii < 2; ii++) {
      for (int iii = 0; iii < 5; iii++) {
        detId[ii][iii] = -1;
      }
    }
    int detIdTemp[5] = { -1, -1, -1, -1, -1 }; // TOF detector id at the current propagation point
    while (propagateToRefX(trefTrk, mXRef + istep * step, step, intLT) && nStripsCrossedInPropagation <= 2 && mXRef + istep * step < Geo::RMAX) {
      if (0 && istep % 100 == 0) {
        printf("istep = %d, currentPosition = %f \n", istep, mXRef + istep * step);
      }
      trefTrk.getXYZGlo(pos);
      for (int ii = 0; ii < 3; ii++) { // we need to change the type...
        posFloat[ii] = pos[ii];
      }
      // uncomment below only for local debug; this will produce A LOT of output - one print per propagation step
      /*
      Printf("posFloat[0] = %f, posFloat[1] = %f, posFloat[2] = %f", posFloat[0], posFloat[1], posFloat[2]);
      Printf("radius xy = %f", TMath::Sqrt(posFloat[0]*posFloat[0] + posFloat[1]*posFloat[1]));
      Printf("radius xyz = %f", TMath::Sqrt(posFloat[0]*posFloat[0] + posFloat[1]*posFloat[1] + posFloat[2]*posFloat[2]));
      */
      for (int idet = 0; idet < 5; idet++)
        detIdTemp[idet] = -1;
      Geo::getPadDxDyDz(posFloat, detIdTemp, deltaPosTemp);

      // uncomment below only for local debug; this will produce A LOT of output - one print per propagation step
      //Printf("detIdTemp[0] = %d, detIdTemp[1] = %d, detIdTemp[2] = %d, detIdTemp[3] = %d, detIdTemp[4] = %d", detIdTemp[0], detIdTemp[1], detIdTemp[2], detIdTemp[3], detIdTemp[4]);
      if (detIdTemp[2] != -1 && nStripsCrossedInPropagation == 0) { // print in case you have a useful propagation
        LOG(DEBUG) << "*********** We have crossed a strip during propagation!*********";
        LOG(DEBUG) << "Global coordinates: pos[0] = " << pos[0] << ", pos[1] = " << pos[1] << ", pos[2] = " << pos[2];
        LOG(DEBUG) << "detIdTemp[0] = " << detIdTemp[0] << ", detIdTemp[1] = " << detIdTemp[1] << ", detIdTemp[2] = " << detIdTemp[2] << ", detIdTemp[3] = " << detIdTemp[3] << ", detIdTemp[4] = " << detIdTemp[4];
        LOG(DEBUG) << "deltaPosTemp[0] = " << deltaPosTemp[0] << ", deltaPosTemp[1] = " << deltaPosTemp[1] << " deltaPosTemp[2] = " << deltaPosTemp[2];
      } else {
        LOG(DEBUG) << "*********** We have NOT crossed a strip during propagation!*********";
        LOG(DEBUG) << "Global coordinates: pos[0] = " << pos[0] << ", pos[1] = " << pos[1] << ", pos[2] = " << pos[2];
        LOG(DEBUG) << "detIdTemp[0] = " << detIdTemp[0] << ", detIdTemp[1] = " << detIdTemp[1] << ", detIdTemp[2] = " << detIdTemp[2] << ", detIdTemp[3] = " << detIdTemp[3] << ", detIdTemp[4] = " << detIdTemp[4];
        LOG(DEBUG) << "deltaPosTemp[0] = " << deltaPosTemp[0] << ", deltaPosTemp[1] = " << deltaPosTemp[1] << " deltaPosTemp[2] = " << deltaPosTemp[2];
      }
      istep++;
      // check if after the propagation we are in a TOF strip
      if (detIdTemp[2] != -1) { // we ended in a TOF strip
        LOG(DEBUG) << "nStripsCrossedInPropagation = " << nStripsCrossedInPropagation << ", detId[nStripsCrossedInPropagation][0] = " << detId[nStripsCrossedInPropagation][0] << ", detIdTemp[0] = " << detIdTemp[0] << ", detId[nStripsCrossedInPropagation][1] = " << detId[nStripsCrossedInPropagation][1] << ", detIdTemp[1] = " << detIdTemp[1] << ", detId[nStripsCrossedInPropagation][2] = " << detId[nStripsCrossedInPropagation][2] << ", detIdTemp[2] = " << detIdTemp[2];
        if (nStripsCrossedInPropagation == 0 ||                                                                                                                                                                                            // we are crossing a strip for the first time...
            (nStripsCrossedInPropagation >= 1 && (detId[nStripsCrossedInPropagation - 1][0] != detIdTemp[0] || detId[nStripsCrossedInPropagation - 1][1] != detIdTemp[1] || detId[nStripsCrossedInPropagation - 1][2] != detIdTemp[2]))) { // ...or we are crossing a new strip
          if (nStripsCrossedInPropagation == 0)
            LOG(DEBUG) << "We cross a strip for the first time";
          if (nStripsCrossedInPropagation == 2) {
            break; // we have already matched 2 strips, we cannot match more
          }
          nStripsCrossedInPropagation++;
        }
        //Printf("nStepsInsideSameStrip[nStripsCrossedInPropagation-1] = %d", nStepsInsideSameStrip[nStripsCrossedInPropagation - 1]);
        if (nStepsInsideSameStrip[nStripsCrossedInPropagation - 1] == 0) {
          detId[nStripsCrossedInPropagation - 1][0] = detIdTemp[0];
          detId[nStripsCrossedInPropagation - 1][1] = detIdTemp[1];
          detId[nStripsCrossedInPropagation - 1][2] = detIdTemp[2];
          detId[nStripsCrossedInPropagation - 1][3] = detIdTemp[3];
          detId[nStripsCrossedInPropagation - 1][4] = detIdTemp[4];
          deltaPos[nStripsCrossedInPropagation - 1][0] = deltaPosTemp[0];
          deltaPos[nStripsCrossedInPropagation - 1][1] = deltaPosTemp[1];
          deltaPos[nStripsCrossedInPropagation - 1][2] = deltaPosTemp[2];
          trkLTInt[nStripsCrossedInPropagation - 1] = intLT;
          Printf("intLT (after matching to strip %d): length = %f, time (Pion) = %f", nStripsCrossedInPropagation - 1, trkLTInt[nStripsCrossedInPropagation - 1].getL(), trkLTInt[nStripsCrossedInPropagation - 1].getTOF(o2::track::PID::Pion));
          nStepsInsideSameStrip[nStripsCrossedInPropagation - 1]++;
        } else {                                                                                                                                    // a further propagation step in the same strip -> update info (we sum up on all matching with strip - we will divide for the number of steps a bit below)
                                                                                                                                                    // N.B. the integrated length and time are taken (at least for now) from the first time we crossed the strip, so here we do nothing with those
          deltaPos[nStripsCrossedInPropagation - 1][0] += deltaPosTemp[0] + (detIdTemp[4] - detId[nStripsCrossedInPropagation - 1][4]) * Geo::XPAD; // residual in x
          deltaPos[nStripsCrossedInPropagation - 1][1] += deltaPosTemp[1];                                                                          // residual in y
          deltaPos[nStripsCrossedInPropagation - 1][2] += deltaPosTemp[2] + (detIdTemp[3] - detId[nStripsCrossedInPropagation - 1][3]) * Geo::ZPAD; // residual in z
          nStepsInsideSameStrip[nStripsCrossedInPropagation - 1]++;
        }
      }
    }
    LOG(DEBUG) << "while done, we propagated track " << itrk << " in %d strips" << nStripsCrossedInPropagation;
    LOG(INFO) << "while done, we propagated track " << itrk << " in %d strips" << nStripsCrossedInPropagation;

    // uncomment for debug purposes, to check tracks that did not cross any strip
    /*
    if (nStripsCrossedInPropagation == 0) {
      auto labelTPCNoStripsCrossed = mTPCLabels->at(mTracksSectIndexCache[sec][itrk]);    
      Printf("The current track (index = %d) never crossed a strip", cacheTrk[itrk]);
      Printf("TrackID = %d, EventID = %d, SourceID = %d", labelTPCNoStripsCrossed.getTrackID(), labelTPCNoStripsCrossed.getEventID(), labelTPCNoStripsCrossed.getSourceID());
      printf("Global coordinates: pos[0] = %f, pos[1] = %f, pos[2] = %f\n", pos[0], pos[1], pos[2]);
      printf("detIdTemp[0] = %d, detIdTemp[1] = %d, detIdTemp[2] = %d, detIdTemp[3] = %d, detIdTemp[4] = %d\n", detIdTemp[0], detIdTemp[1], detIdTemp[2], detIdTemp[3], detIdTemp[4]);
      printf("deltaPosTemp[0] = %f, deltaPosTemp[1] = %f, deltaPosTemp[2] = %f\n", deltaPosTemp[0], deltaPosTemp[1], deltaPosTemp[2]);
    }
    */

    for (Int_t imatch = 0; imatch < nStripsCrossedInPropagation; imatch++) {
      // we take as residual the average of the residuals along the propagation in the same strip
      deltaPos[imatch][0] /= nStepsInsideSameStrip[imatch];
      deltaPos[imatch][1] /= nStepsInsideSameStrip[imatch];
      deltaPos[imatch][2] /= nStepsInsideSameStrip[imatch];
      LOG(DEBUG) << "matched strip " << imatch << ": deltaPos[0] = " << deltaPos[imatch][0] << ", deltaPos[1] = " << deltaPos[imatch][1] << ", deltaPos[2] = " << deltaPos[imatch][2] << ", residual (x, z) = " << TMath::Sqrt(deltaPos[imatch][0] * deltaPos[imatch][0] + deltaPos[imatch][2] * deltaPos[imatch][2]);
    }

    if (nStripsCrossedInPropagation == 0) {
      continue; // the track never hit a TOF strip during the propagation
    }
    bool foundCluster = false;
    auto labelTPC = (*mTPCLabels)[mTracksSectIndexCache[sec][itrk]];
    for (auto itof = itof0; itof < nTOFCls; itof++) {
      //      printf("itof = %d\n", itof);
      auto& trefTOF = mTOFClusWork[cacheTOF[itof]];
      // compare the times of the track and the TOF clusters - remember that they both are ordered in time!
      //Printf("trefTOF.getTime() = %f, maxTrkTime = %f, minTrkTime = %f", trefTOF.getTime(), maxTrkTime, minTrkTime);

      if (trefTOF.getTime() < minTrkTime) { // this cluster has a time that is too small for the current track, we will get to the next one
        //Printf("In trefTOF.getTime() < minTrkTime");
        itof0 = itof + 1; // but for the next track that we will check, we will ignore this cluster (the time is anyway too small)
        continue;
      }
      if (trefTOF.getTime() > maxTrkTime) { // no more TOF clusters can be matched to this track
        break;
      }

      int mainChannel = trefTOF.getMainContributingChannel();
      int indices[5];
      Geo::getVolumeIndices(mainChannel, indices);

      // TO be done
      // weighted average to be included in case of multipad clusters

      const auto& labelsTOF = mTOFClusLabels->getLabels(mTOFClusSectIndexCache[indices[0]][itof]);
      int trackIdTOF;
      int eventIdTOF;
      int sourceIdTOF;
      for (auto iPropagation = 0; iPropagation < nStripsCrossedInPropagation; iPropagation++) {
        LOG(DEBUG) << "TOF Cluster [" << itof << ", " << cacheTOF[itof] << "]:      indices   = " << indices[0] << ", " << indices[1] << ", " << indices[2] << ", " << indices[3] << ", " << indices[4];
        LOG(DEBUG) << "Propagated Track [" << itrk << ", " << cacheTrk[itrk] << "]: detId[" << iPropagation << "]  = " << detId[iPropagation][0] << ", " << detId[iPropagation][1] << ", " << detId[iPropagation][2] << ", " << detId[iPropagation][3] << ", " << detId[iPropagation][4];
        float resX = deltaPos[iPropagation][0] - (indices[4] - detId[iPropagation][4]) * Geo::XPAD; // readjusting the residuals due to the fact that the propagation fell in a pad that was not exactly the one of the cluster
        float resZ = deltaPos[iPropagation][2] - (indices[3] - detId[iPropagation][3]) * Geo::ZPAD; // readjusting the residuals due to the fact that the propagation fell in a pad that was not exactly the one of the cluster
        float res = TMath::Sqrt(resX * resX + resZ * resZ);
        LOG(DEBUG) << "resX = " << resX << ", resZ = " << resZ << ", res = " << res;
#ifdef _ALLOW_DEBUG_TREES_
        fillTOFmatchTree("match0", cacheTOF[itof], indices[0], indices[1], indices[2], indices[3], indices[4], cacheTrk[itrk], iPropagation, detId[iPropagation][0], detId[iPropagation][1], detId[iPropagation][2], detId[iPropagation][3], detId[iPropagation][4], resX, resZ, res, trackWork, trkLTInt[iPropagation].getL(), trkLTInt[iPropagation].getTOF(o2::track::PID::Pion), trefTOF.getTime());
#endif
        int tofLabelTrackID[3] = { -1, -1, -1 };
        int tofLabelEventID[3] = { -1, -1, -1 };
        int tofLabelSourceID[3] = { -1, -1, -1 };
        for (int ilabel = 0; ilabel < labelsTOF.size(); ilabel++) {
          tofLabelTrackID[ilabel] = labelsTOF[ilabel].getTrackID();
          tofLabelEventID[ilabel] = labelsTOF[ilabel].getEventID();
          tofLabelSourceID[ilabel] = labelsTOF[ilabel].getSourceID();
        }
        //auto labelTPC = mTPCLabels->at(mTracksSectIndexCache[indices[0]][itrk]);
        auto labelITS = (*mITSLabels)[mTracksSectIndexCache[indices[0]][itrk]];
        fillTOFmatchTreeWithLabels("matchPossibleWithLabels", cacheTOF[itof], indices[0], indices[1], indices[2], indices[3], indices[4], cacheTrk[itrk], iPropagation, detId[iPropagation][0], detId[iPropagation][1], detId[iPropagation][2], detId[iPropagation][3], detId[iPropagation][4], resX, resZ, res, trackWork, labelTPC.getTrackID(), labelTPC.getEventID(), labelTPC.getSourceID(), labelITS.getTrackID(), labelITS.getEventID(), labelITS.getSourceID(), tofLabelTrackID[0], tofLabelEventID[0], tofLabelSourceID[0], tofLabelTrackID[1], tofLabelEventID[1], tofLabelSourceID[1], tofLabelTrackID[2], tofLabelEventID[2], tofLabelSourceID[2], trkLTInt[iPropagation].getL(), trkLTInt[iPropagation].getTOF(o2::track::PID::Pion), trefTOF.getTime());
        if (indices[0] != detId[iPropagation][0])
          continue;
        if (indices[1] != detId[iPropagation][1])
          continue;
        if (indices[2] != detId[iPropagation][2])
          continue;
        float chi2 = res; // TODO: take into account also the time!
        fillTOFmatchTree("match1", cacheTOF[itof], indices[0], indices[1], indices[2], indices[3], indices[4], cacheTrk[itrk], iPropagation, detId[iPropagation][0], detId[iPropagation][1], detId[iPropagation][2], detId[iPropagation][3], detId[iPropagation][4], resX, resZ, res, trackWork, trkLTInt[iPropagation].getL(), trkLTInt[iPropagation].getTOF(o2::track::PID::Pion), trefTOF.getTime());

        fillTOFmatchTreeWithLabels("matchOkWithLabels", cacheTOF[itof], indices[0], indices[1], indices[2], indices[3], indices[4], cacheTrk[itrk], iPropagation, detId[iPropagation][0], detId[iPropagation][1], detId[iPropagation][2], detId[iPropagation][3], detId[iPropagation][4], resX, resZ, res, trackWork, labelTPC.getTrackID(), labelTPC.getEventID(), labelTPC.getSourceID(), labelITS.getTrackID(), labelITS.getEventID(), labelITS.getSourceID(), tofLabelTrackID[0], tofLabelEventID[0], tofLabelSourceID[0], tofLabelTrackID[1], tofLabelEventID[1], tofLabelSourceID[1], tofLabelTrackID[2], tofLabelEventID[2], tofLabelSourceID[2], trkLTInt[iPropagation].getL(), trkLTInt[iPropagation].getTOF(o2::track::PID::Pion), trefTOF.getTime());

        if (res < mSpaceTolerance) { // matching ok!
          LOG(DEBUG) << "MATCHING FOUND: We have a match! between track " << mTracksSectIndexCache[indices[0]][itrk] << " and TOF cluster " << mTOFClusSectIndexCache[indices[0]][itof];
          foundCluster = true;
          evIdx eventIndexTOFCluster(trefTOF.getEntryInTree(), mTOFClusSectIndexCache[indices[0]][itof]);
          evIdx eventIndexTracks(mCurrTracksTreeEntry, mTracksSectIndexCache[indices[0]][itrk]);
          mMatchedTracksPairs.emplace_back(std::make_pair(eventIndexTracks, o2::dataformats::MatchInfoTOF(eventIndexTOFCluster, chi2, trkLTInt[iPropagation]))); // TODO: check if this is correct!
          for (int ilabel = 0; ilabel < labelsTOF.size(); ilabel++) {
            LOG(DEBUG) << "TOF label " << ilabel << ": trackID = " << labelsTOF[ilabel].getTrackID() << ", eventID = " << labelsTOF[ilabel].getEventID() << ", sourceID = " << labelsTOF[ilabel].getSourceID();
          }
          LOG(DEBUG) << "TPC label of the track: trackID = " << labelTPC.getTrackID() << ", eventID = " << labelTPC.getEventID() << ", sourceID = " << labelTPC.getSourceID();
          LOG(DEBUG) << "ITS label of the track: trackID = " << labelITS.getTrackID() << ", eventID = " << labelITS.getEventID() << ", sourceID = " << labelITS.getSourceID();
          fillTOFmatchTreeWithLabels("matchOkWithLabelsInSpaceTolerance", cacheTOF[itof], indices[0], indices[1], indices[2], indices[3], indices[4], cacheTrk[itrk], iPropagation, detId[iPropagation][0], detId[iPropagation][1], detId[iPropagation][2], detId[iPropagation][3], detId[iPropagation][4], resX, resZ, res, trackWork, labelTPC.getTrackID(), labelTPC.getEventID(), labelTPC.getSourceID(), labelITS.getTrackID(), labelITS.getEventID(), labelITS.getSourceID(), tofLabelTrackID[0], tofLabelEventID[0], tofLabelSourceID[0], tofLabelTrackID[1], tofLabelEventID[1], tofLabelSourceID[1], tofLabelTrackID[2], tofLabelEventID[2], tofLabelSourceID[2], trkLTInt[iPropagation].getL(), trkLTInt[iPropagation].getTOF(o2::track::PID::Pion), trefTOF.getTime());
        }
      }
    }
    if (!foundCluster)
      LOG(DEBUG) << "We did not find any TOF cluster for track " << cacheTrk[itrk] << " (label = " << labelTPC.getTrackID() << ", pt = " << trefTrk.getPt();
  }
  return;
}

//______________________________________________
void MatchTOF::selectBestMatches()
{
  ///< define the track-TOFcluster pair per sector

  // first, we sort according to the chi2
  std::sort(mMatchedTracksPairs.begin(), mMatchedTracksPairs.end(), [this](std::pair<evIdx, o2::dataformats::MatchInfoTOF> a, std::pair<evIdx, o2::dataformats::MatchInfoTOF> b) { return (a.second.getChi2() < b.second.getChi2()); });
  int i = 0;
  // then we take discard the pairs if their track or cluster was already matched (since they are ordered in chi2, we will take the best matching)
  for (const std::pair<evIdx, o2::dataformats::MatchInfoTOF>& matchingPair : mMatchedTracksPairs) {
    if (mMatchedTracksIndex[matchingPair.first.getIndex()] != -1) { // the track was already filled
      continue;
    }
    if (mMatchedClustersIndex[matchingPair.second.getTOFClIndex()] != -1) { // the track was already filled
      continue;
    }
    mMatchedTracksIndex[matchingPair.first.getIndex()] = mMatchedTracks.size();                                      // index of the MatchInfoTOF correspoding to this track
    mMatchedClustersIndex[matchingPair.second.getTOFClIndex()] = mMatchedTracksIndex[matchingPair.first.getIndex()]; // index of the track that was matched to this cluster
    mMatchedTracks.push_back(matchingPair);                                                               // array of MatchInfoTOF

    // add also calibration infos ciao
    mCalibInfoTOF.emplace_back(mTOFClusWork[matchingPair.second.getTOFClIndex()].getMainContributingChannel(),
                               int(mTOFClusWork[matchingPair.second.getTOFClIndex()].getTimeRaw() * 1E12), // add time stamp
                               mTOFClusWork[matchingPair.second.getTOFClIndex()].getTimeRaw() - matchingPair.second.getLTIntegralOut().getTOF(o2::track::PID::Pion),
                               mTOFClusWork[matchingPair.second.getTOFClIndex()].getTot());

    const auto& labelTPC = (*mTPCLabels)[matchingPair.first.getIndex()];
    LOG(DEBUG) << "labelTPC: trackID = " << labelTPC.getTrackID() << ", eventID = " << labelTPC.getEventID() << ", sourceID = " << labelTPC.getSourceID();
    const auto& labelITS = (*mITSLabels)[matchingPair.first.getIndex()];
    LOG(DEBUG) << "labelITS: trackID = " << labelITS.getTrackID() << ", eventID = " << labelITS.getEventID() << ", sourceID = " << labelITS.getSourceID();
    const auto& labelsTOF = mTOFClusLabels->getLabels(matchingPair.second.getTOFClIndex());
    bool labelOk = false; // whether we have found or not the same TPC label of the track among the labels of the TOF cluster
    for (int ilabel = 0; ilabel < labelsTOF.size(); ilabel++) {
      LOG(DEBUG) << "TOF label " << ilabel << ": trackID = " << labelsTOF[ilabel].getTrackID() << ", eventID = " << labelsTOF[ilabel].getEventID() << ", sourceID = " << labelsTOF[ilabel].getSourceID();
      if (labelsTOF[ilabel].getTrackID() == labelTPC.getTrackID() && labelsTOF[ilabel].getEventID() == labelTPC.getEventID() && labelsTOF[ilabel].getSourceID() == labelTPC.getSourceID() && !labelOk) { // if we find one TOF cluster label that is the same as the TPC one, we are happy - even if it is not the first one
        mOutTOFLabels.push_back(labelsTOF[ilabel]);
        labelOk = true;
      }
    }
    if (!labelOk) {
      // we have not found the track label among those associated to the TOF cluster --> fake match! We will associate the label of the main channel, but negative
      mOutTOFLabels.emplace_back(labelsTOF[0].getTrackID(), labelsTOF[0].getEventID(), labelsTOF[0].getSourceID(), true);
    }
    mOutTPCLabels.push_back(labelTPC);
    mOutITSLabels.push_back(labelITS);
    i++;
  }
}

//______________________________________________
bool MatchTOF::propagateToRefX(o2::track::TrackParCov& trc, float xRef, float stepInCm, o2::track::TrackLTIntegral& intLT)
{
  // propagate track to matching reference X
  const int matCorr = 1; // material correction method
  const float tanHalfSector = tan(o2::constants::math::SectorSpanRad / 2);
  bool refReached = false;
  float xStart = trc.getX();
  // the first propagation will be from 2m, if the track is not at least at 2m
  if (xStart < 50.)
    xStart = 50.;
  int istep = 1;
  bool hasPropagated = o2::base::Propagator::Instance()->PropagateToXBxByBz(trc, xStart + istep * stepInCm, o2::constants::physics::MassPionCharged, MAXSNP, stepInCm, matCorr, &intLT);
  while (hasPropagated) {
    if (trc.getX() > xRef) {
      refReached = true; // we reached the 371cm reference
    }
    istep++;
    if (fabs(trc.getY()) > trc.getX() * tanHalfSector) { // we are still in the same sector
      // we need to rotate the track to go to the new sector
      //Printf("propagateToRefX: changing sector");
      auto alphaNew = o2::utils::Angle2Alpha(trc.getPhiPos());
      if (!trc.rotate(alphaNew) != 0) {
        //	Printf("propagateToRefX: failed to rotate");
        break; // failed (this line is taken from MatchTPCITS and the following comment too: RS: check effect on matching tracks to neighbouring sector)
      }
    }
    if (refReached)
      break;
    hasPropagated = o2::base::Propagator::Instance()->PropagateToXBxByBz(trc, xStart + istep * stepInCm, o2::constants::physics::MassPionCharged, MAXSNP, stepInCm, matCorr, &intLT);
  }
  //  if (std::abs(trc.getSnp()) > MAXSNP) Printf("propagateToRefX: condition on snp not ok, returning false");
  //Printf("propagateToRefX: snp of teh track is %f (--> %f grad)", trc.getSnp(), TMath::ASin(trc.getSnp())*TMath::RadToDeg());
  return refReached && std::abs(trc.getSnp()) < 0.95; // Here we need to put MAXSNP
}

//______________________________________________
bool MatchTOF::propagateToRefXWithoutCov(o2::track::TrackParCov& trc, float xRef, float stepInCm, float bzField)
{
  // propagate track to matching reference X without using the covariance matrix
  // we create the copy of the track in a TrackPar object (no cov matrix)
  o2::track::TrackPar trcNoCov(trc);
  const float tanHalfSector = tan(o2::constants::math::SectorSpanRad / 2);
  bool refReached = false;
  float xStart = trcNoCov.getX();
  // the first propagation will be from 2m, if the track is not at least at 2m
  if (xStart < 50.)
    xStart = 50.;
  int istep = 1;
  bool hasPropagated = trcNoCov.propagateParamTo(xStart + istep * stepInCm, bzField);
  while (hasPropagated) {
    if (trcNoCov.getX() > xRef) {
      refReached = true; // we reached the 371cm reference
    }
    istep++;
    if (fabs(trcNoCov.getY()) > trcNoCov.getX() * tanHalfSector) { // we are still in the same sector
      // we need to rotate the track to go to the new sector
      //Printf("propagateToRefX: changing sector");
      auto alphaNew = o2::utils::Angle2Alpha(trcNoCov.getPhiPos());
      if (!trcNoCov.rotateParam(alphaNew) != 0) {
        //	Printf("propagateToRefX: failed to rotate");
        break; // failed (this line is taken from MatchTPCITS and the following comment too: RS: check effect on matching tracks to neighbouring sector)
      }
    }
    if (refReached)
      break;
    hasPropagated = trcNoCov.propagateParamTo(xStart + istep * stepInCm, bzField);
  }
  //  if (std::abs(trc.getSnp()) > MAXSNP) Printf("propagateToRefX: condition on snp not ok, returning false");
  //Printf("propagateToRefX: snp of teh track is %f (--> %f grad)", trcNoCov.getSnp(), TMath::ASin(trcNoCov.getSnp())*TMath::RadToDeg());

  return refReached && std::abs(trcNoCov.getSnp()) < 0.95 && TMath::Abs(trcNoCov.getZ()) < Geo::MAXHZTOF; // Here we need to put MAXSNP
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
void MatchTOF::fillTOFmatchTree(const char* trname, int cacheTOF, int sectTOF, int plateTOF, int stripTOF, int padXTOF, int padZTOF, int cacheeTrk, int crossedStrip, int sectPropagation, int platePropagation, int stripPropagation, int padXPropagation, int padZPropagation, float resX, float resZ, float res, o2::dataformats::TrackTPCITS& trk, float intLength, float intTimePion, float timeTOF)
{
  ///< fill debug tree for TOF tracks matching check

  mTimerDBG.Start(false);

  //  Printf("************** Filling the debug tree with %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %f, %f, %f", cacheTOF, sectTOF, plateTOF, stripTOF, padXTOF, padZTOF, cacheeTrk, crossedStrip, sectPropagation, platePropagation, stripPropagation, padXPropagation, padZPropagation, resX, resZ, res);

  if (mDBGFlags) {
    (*mDBGOut) << trname
               << "clusterTOF=" << cacheTOF << "sectTOF=" << sectTOF << "plateTOF=" << plateTOF << "stripTOF=" << stripTOF << "padXTOF=" << padXTOF << "padZTOF=" << padZTOF
               << "crossedStrip=" << crossedStrip << "sectPropagation=" << sectPropagation << "platePropagation=" << platePropagation << "stripPropagation=" << stripPropagation << "padXPropagation=" << padXPropagation
               << "resX=" << resX << "resZ=" << resZ << "res=" << res << "track=" << trk << "intLength=" << intLength << "intTimePion=" << intTimePion << "timeTOF=" << timeTOF << "\n";
  }
  mTimerDBG.Stop();
}

//_________________________________________________________
void MatchTOF::fillTOFmatchTreeWithLabels(const char* trname, int cacheTOF, int sectTOF, int plateTOF, int stripTOF, int padXTOF, int padZTOF, int cacheeTrk, int crossedStrip, int sectPropagation, int platePropagation, int stripPropagation, int padXPropagation, int padZPropagation, float resX, float resZ, float res, o2::dataformats::TrackTPCITS& trk, int TPClabelTrackID, int TPClabelEventID, int TPClabelSourceID, int ITSlabelTrackID, int ITSlabelEventID, int ITSlabelSourceID, int TOFlabelTrackID0, int TOFlabelEventID0, int TOFlabelSourceID0, int TOFlabelTrackID1, int TOFlabelEventID1, int TOFlabelSourceID1, int TOFlabelTrackID2, int TOFlabelEventID2, int TOFlabelSourceID2, float intLength, float intTimePion, float timeTOF)
{
  ///< fill debug tree for TOF tracks matching check

  mTimerDBG.Start(false);

  if (mDBGFlags) {
    (*mDBGOut) << trname
               << "clusterTOF=" << cacheTOF << "sectTOF=" << sectTOF << "plateTOF=" << plateTOF << "stripTOF=" << stripTOF << "padXTOF=" << padXTOF << "padZTOF=" << padZTOF
               << "crossedStrip=" << crossedStrip << "sectPropagation=" << sectPropagation << "platePropagation=" << platePropagation << "stripPropagation=" << stripPropagation << "padXPropagation=" << padXPropagation
               << "resX=" << resX << "resZ=" << resZ << "res=" << res << "track=" << trk
               << "TPClabelTrackID=" << TPClabelTrackID << "TPClabelEventID=" << TPClabelEventID << "TPClabelSourceID=" << TPClabelSourceID
               << "ITSlabelTrackID=" << ITSlabelTrackID << "ITSlabelEventID=" << ITSlabelEventID << "ITSlabelSourceID=" << ITSlabelSourceID
               << "TOFlabelTrackID0=" << TOFlabelTrackID0 << "TOFlabelEventID0=" << TOFlabelEventID0 << "TOFlabelSourceID0=" << TOFlabelSourceID0
               << "TOFlabelTrackID1=" << TOFlabelTrackID1 << "TOFlabelEventID1=" << TOFlabelEventID1 << "TOFlabelSourceID1=" << TOFlabelSourceID1
               << "TOFlabelTrackID2=" << TOFlabelTrackID2 << "TOFlabelEventID2=" << TOFlabelEventID2 << "TOFlabelSourceID2=" << TOFlabelSourceID2
               << "intLength=" << intLength << "intTimePion=" << intTimePion << "timeTOF=" << timeTOF
               << "\n";
  }
  mTimerDBG.Stop();
}
#endif
