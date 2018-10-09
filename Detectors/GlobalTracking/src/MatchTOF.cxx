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
    printf("doing matching...\n");
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
  
  if (!mTreeTPCTracks->GetBranch(mTPCTracksBranchName.data())) {
    LOG(FATAL) << "Did not find TPC tracks branch " << mTPCTracksBranchName << " in the input tree" << FairLogger::endl;
  }
  mTreeTPCTracks->SetBranchAddress(mTPCTracksBranchName.data(), &mTPCTracksArrayInp);
  LOG(INFO) << "Attached TPC tracks " << mTPCTracksBranchName << " branch with " << mTreeTPCTracks->GetEntries()
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
  
  int nNotPropagatedToTOF = 0;
  for (int it = 0; it < mNumOfTracks; it++) {
    o2::dataformats::TrackTPCITS& trcOrig = (*mTracksArrayInp)[it]; // TODO: check if we cannot directly use the o2::track::TrackParCov class instead of o2::dataformats::TrackTPCITS, and then avoid the casting below; this is the track at the vertex
    //o2::track::TrackParCov& trcOrig = (*mTracksArrayInp)[it]; // TODO: check if we cannot directly use the o2::track::TrackParCov class instead of o2::dataformats::TrackTPCITS, and then avoid the casting below; this is the track at the vertex
    /*
    evIdx evIdxTPC = trcOrig.getRefTPC();
    int indTPC = evIdxTPC.getIndex();
    o2::TPC::TrackTPC& trcTPCOrig = (*mTPCTracksArrayInp)[indTPC]; // we take the track when it is propagated out
    */
    Printf("Original Track %d: getTimeMUS().getTimeStamp() = %f, getTimeMUS().getTimeStampError() = %f", it, (*mTracksArrayInp)[it].getTimeMUS().getTimeStamp() , (*mTracksArrayInp)[it].getTimeMUS().getTimeStampError()); 
    // create working copy of track param
    mTracksWork.emplace_back(trcOrig);//, mCurrTracksTreeEntry, it);
    // make a copy of the TPC track that we have to propagate
    //o2::TPC::TrackTPC* trc = new o2::TPC::TrackTPC(trcTPCOrig); // this would take the TPCout track
    auto& trc = mTracksWork.back(); // with this we take the TPCITS track propagated to the vertex
    Printf("Copied Track %d: getTimeMUS().getTimeStamp() = %f, getTimeMUS().getTimeStampError() = %f", it, trc.getTimeMUS().getTimeStamp() , trc.getTimeMUS().getTimeStampError()); 
    // propagate to matching Xref
    if (!propagateToRefX(trc, mXRef, 2) || TMath::Abs(trc.getZ()) > Geo::MAXHZTOF) {
      mTracksWork.pop_back(); // discard track whose propagation to mXRef failed, or those that go beyond TOF in z
      nNotPropagatedToTOF++;
      continue;
    }
    //    if (mMCTruthON) {
    //      mTracksLblWork.emplace_back(mTracksLabels->getLabels(it)[0]);
    //    }
    // cache work track index
    std::array<float, 3> globalPos;
    trc.getXYZGlo(globalPos);

    mTracksSectIndexCache[o2::utils::Angle2Sector( TMath::ATan2(globalPos[1], globalPos[0]))].push_back(mTracksWork.size() - 1);
    //delete trc; // Check: is this needed?
  }

  LOG(INFO) << "Total number of tracks = " << mNumOfTracks << ", Number of tracks that failed to be propagated to TOF = " << nNotPropagatedToTOF << FairLogger::endl;
  
  // sort tracks in each sector according to their time (increasing in time) 
  for (int sec = o2::constants::math::NSectors; sec--;) {
    auto& indexCache = mTracksSectIndexCache[sec];
    LOG(INFO) << "Sorting sector" << sec << " | " << indexCache.size() << " tracks" << FairLogger::endl;
    if (!indexCache.size())
      continue;
    std::sort(indexCache.begin(), indexCache.end(), [this](int a, int b) {
	auto& trcA = mTracksWork[a];
	auto& trcB = mTracksWork[b];
	return ((trcA.getTimeMUS().getTimeStamp() - mSigmaTimeCut*trcA.getTimeMUS().getTimeStampError()) - (trcB.getTimeMUS().getTimeStamp() - mSigmaTimeCut*trcB.getTimeMUS().getTimeStampError()) < 0.);
      });
  } // loop over tracks of single sector

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
    LOG(INFO) << "Sorting sector" << sec << " | " << indexCache.size() << " TOF clusters" << FairLogger::endl;
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
  printf("Loading TOF clusters: number of entries in tree = %d\n", mTreeTOFClusters->GetEntries());
  while (++mCurrTOFClustersTreeEntry < mTreeTOFClusters->GetEntries()) {
    mTreeTOFClusters->GetEntry(mCurrTOFClustersTreeEntry);
    LOG(DEBUG) << "Loading TOF clusters entry " << mCurrTOFClustersTreeEntry << " -> " << mTOFClustersArrayInp->size()
               << " clusters" << FairLogger::endl;
    LOG(INFO) << "Loading TOF clusters entry " << mCurrTOFClustersTreeEntry << " -> " << mTOFClustersArrayInp->size()
               << " clusters" << FairLogger::endl;
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
  LOG(INFO) << "Matching sector " << sec << ": number of tracks: " << nTracks << ", number of TOF clusters: " << nTOFCls << FairLogger::endl;
  if (!nTracks || !nTOFCls) {
    return;
  }
  int itof0 = 0; // starting index in TOF clusters for matching of the track
  int   detId[2][5]; // at maximum one track can fall in 2 strips during the propagation; the second dimention of the array is the TOF det index
  float deltaPos[2][3]; // at maximum one track can fall in 2 strips during the propagation; the second dimention of the array is the residuals
  int nStepsInsideSameStrip[2] = {0,0}; // number of propagation steps in the same strip (since we have maximum 2 strips, it has dimention = 2)
  int detIdTemp[5];
  float deltaPosTemp[3];
  std::array<float, 3> pos;
  std::array<float, 3> posBeforeProp;
  float posFloat[3]; 
  
  for (int itrk = 0; itrk < cacheTrk.size(); itrk++) {
    Printf("\n track %d", itrk);
    for (int ii = 0; ii < 2; ii++)
      detId[ii][2] = -1; // before trying to match, we need to inizialize the detId corresponding to the strip number to -1
    int nStripsCrossedInPropagation = 0; // how many strips were hit during the propagation
    auto& trefTrk = mTracksWork[cacheTrk[itrk]];
    float minTrkTime = (trefTrk.getTimeMUS().getTimeStamp() - mSigmaTimeCut*trefTrk.getTimeMUS().getTimeStampError())*1.E6; // minimum time in ps
    float maxTrkTime = (trefTrk.getTimeMUS().getTimeStamp() + mSigmaTimeCut*trefTrk.getTimeMUS().getTimeStampError())*1.E6; // maximum time in ps
    int istep = 1; // number of steps
    float step = 0.1; // step size in cm
    trefTrk.getXYZGlo(posBeforeProp);
    //float posBeforeProp[3] = {trefTrk.getX(), trefTrk.getY(), trefTrk.getZ()}; // in local ref system
    printf("Global coordinates: posBeforeProp[0] = %f, posBeforeProp[1] = %f, posBeforeProp[2] = %f\n", posBeforeProp[0], posBeforeProp[1], posBeforeProp[2]);
    while (propagateToRefX(trefTrk, mXRef+istep*step, step) && nStripsCrossedInPropagation <2 && mXRef+istep*step < Geo::RMAX){
      if (istep%100 == 0){
	printf("istep = %d, currentPosition = %f \n", istep, mXRef+istep*step);
      }
      //float pos[3] = {trefTrk.getX(), trefTrk.getY(), trefTrk.getZ()}; // these are the coordinates in the local ref system
      trefTrk.getXYZGlo(pos);
      //printf("getPadDxDyDz:\n");
      for (int ii = 0; ii < 3; ii++){ // we need to change the type... 
	posFloat[ii] = pos[ii];
      }
      Geo::getPadDxDyDz(posFloat, detIdTemp, deltaPosTemp);
      if (detIdTemp[2] != -1 && nStripsCrossedInPropagation == 0){ // print in case you have a useful propagation
	Printf("\n\n*********** We have crossed a strip during propagation!*********");
	printf("Global coordinates: pos[0] = %f, pos[1] = %f, pos[2] = %f\n", pos[0], pos[1], pos[2]);
	printf("detIdTemp[0] = %d, detIdTemp[1] = %d, detIdTemp[2] = %d, detIdTemp[3] = %d, detIdTemp[4] = %d\n", detIdTemp[0], detIdTemp[1], detIdTemp[2], detIdTemp[3], detIdTemp[4]);
	printf("deltaPosTemp[0] = %f, deltaPosTemp[1] = %f, deltaPosTemp[2] = %f\n", deltaPosTemp[0], deltaPosTemp[1], deltaPosTemp[2]);
      }
	//printf("getPadDxDyDz done\n");
      istep++;
      // check if after the propagation we are in a TOF strip
      if (detIdTemp[2] != -1) { // we ended in a TOF strip
	Printf("We are in a TOF strip!");
	if(nStripsCrossedInPropagation == 0 || detId[nStripsCrossedInPropagation-1][0] != detIdTemp[0] ||
	   detId[nStripsCrossedInPropagation-1][1] != detIdTemp[1] ||
	   detId[nStripsCrossedInPropagation-1][2] != detIdTemp[2]) {
	  if(nStripsCrossedInPropagation == 2) break; // we have already matched 2 strips, we cannot match more
	  nStripsCrossedInPropagation++;
	}	
	
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
	else{ // a further propagation step in the same strip -> update info (we average on all matching with strip)
	  deltaPos[nStripsCrossedInPropagation-1][0] += deltaPosTemp[0] + (detIdTemp[4] - detId[nStripsCrossedInPropagation-1][4])*Geo::XPAD;
	  deltaPos[nStripsCrossedInPropagation-1][1] += deltaPosTemp[1];
	  deltaPos[nStripsCrossedInPropagation-1][2] += deltaPosTemp[2] + (detIdTemp[3] - detId[nStripsCrossedInPropagation-1][3])*Geo::ZPAD;
	  nStepsInsideSameStrip[nStripsCrossedInPropagation-1]++;
	}
      }
    }    
    printf("while done, we propagated track %d in %d strips\n", itrk, nStripsCrossedInPropagation);
    
    for(Int_t imatch = 0; imatch < nStripsCrossedInPropagation;imatch++){
      //printf("imatch = %d\n", imatch);
      // we take as residual the average of the residuals along the propagation in the same strip
      deltaPos[imatch][0] /= nStepsInsideSameStrip[imatch]; 
      deltaPos[imatch][1] /= nStepsInsideSameStrip[imatch];
      deltaPos[imatch][2] /= nStepsInsideSameStrip[imatch];
      printf("matched strip %d: deltaPos[0] = %f, deltaPos[1] = %f, deltaPos[2] = %f\n", imatch, deltaPos[imatch][0], deltaPos[imatch][1], deltaPos[imatch][2]);
    }

    if (nStripsCrossedInPropagation == 0) continue; // the track never hit a TOF strip during the propagation
    Printf("We will check now the %d TOF clusters", nTOFCls);
    for (auto itof = itof0; itof < nTOFCls; itof++) {
      printf("itof = %d\n", itof);
      auto& trefTOF = mTOFClusWork[cacheTOF[itof]];
      // compare the times of the track and the TOF clusters - remember that they both are ordered in time!
      Printf("trefTOF.getTime() = %f, maxTrkTime = %f, minTrkTime = %f", trefTOF.getTime(), maxTrkTime, minTrkTime);
      if (trefTOF.getTime() < minTrkTime) {
	itof0 = itof;
	//	continue;
      }
      if (trefTOF.getTime() > maxTrkTime) { // no more TOF clusters can be matched to this track
	//	break;
      }
      int mainChannel = trefTOF.getMainContributingChannel();
      int indices[5];
      Geo::getVolumeIndices(mainChannel, indices);
      for (auto iPropagation = 0; iPropagation < nStripsCrossedInPropagation; iPropagation++){
	printf("iPropagation = %d\n", iPropagation);
	Printf("indices[0] = %d, indices[1] = %d, indices[2] = %d, detId[iPropagation][0] = %d, detId[iPropagation][1] = %d, detId[iPropagation][2] = %d", indices[0], indices[1], indices[2], detId[iPropagation][0], detId[iPropagation][1], detId[iPropagation][2]);
	if (indices[0] != detId[iPropagation][0]) continue;
	if (indices[1] != detId[iPropagation][1]) continue;
	if (indices[2] != detId[iPropagation][2]) continue;
	float resX = deltaPos[iPropagation][0] - (indices[4] - detId[iPropagation][4])*Geo::XPAD; // readjusting the residuals due to the fact that the propagation fell in a pad that was not exactly the one of the cluster
	float resZ = deltaPos[iPropagation][2] - (indices[3] - detId[iPropagation][3])*Geo::ZPAD; // readjusting the residuals due to the fact that the propagation fell in a pad that was not exactly the one of the cluster
	float res = TMath::Sqrt(resX*resX + resZ*resZ);
	Printf("resX = %f, resZ = %f, res = %f", resX, resZ, res);
	float chi2 = res; // TODO: take into account also the time!
	if (res < mSpaceTolerance) { // matching ok!
	  Printf("YUHUUUUUUUUU! We have a match!");
	  mMatchedTracksPairs.push_back(std::make_pair(mTracksSectIndexCache[indices[0]][itrk], o2::dataformats::MatchInfoTOF(mTOFClusSectIndexCache[indices[0]][itof], chi2))); // TODO: check if this is correct!
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
  int i = 0;
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
    i++;
  }
 
}

//______________________________________________
bool MatchTOF::propagateToRefX(o2::track::TrackParCov& trc, float xRef, float stepInCm)
{
  //  printf("in propagateToRefX\n");
  // propagate track to matching reference X
  bool refReached = o2::Base::Propagator::Instance()->PropagateToXBxByBz(trc, xRef, o2::constants::physics::MassPionCharged, MaxSnp, stepInCm, 0.);;
  //printf("refReached = %d\n", (int)refReached);
  return refReached;
  /*
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
  printf("returning from propagateToRefX\n");
  return refReached && std::abs(trc.getSnp()) < MaxSnp;
  */
}
