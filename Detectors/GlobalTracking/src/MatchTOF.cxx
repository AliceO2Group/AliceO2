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

#include "TPCBase/ParameterGas.h"
#include "TPCBase/ParameterElectronics.h"

using namespace o2::globaltracking;
using evGIdx = o2::dataformats::EvIndex<int, o2::dataformats::GlobalTrackID>;
using evIdx = o2::dataformats::EvIndex<int, int>;

ClassImp(MatchTOF);

//______________________________________________
void MatchTOF::run()
{
  ///< running the matching

  if (!mWFInputAttached && !mSAInitDone) {
    LOG(ERROR) << "run called with mSAInitDone=" << mSAInitDone << " and mWFInputAttached=" << mWFInputAttached;
    throw std::runtime_error("standalone init was not done or workflow input was not yet attached");
  }
  mTimerTot.Start();

  // we load all TOF clusters (to be checked if we need to split per time frame)
  prepareTOFClusters();

  mTimerTot.Stop();
  LOGF(INFO, "Timing prepareTOFCluster: Cpu: %.3e s Real: %.3e s in %d slots", mTimerTot.CpuTime(), mTimerTot.RealTime(), mTimerTot.Counter() - 1);
  mTimerTot.Start();

  if (mIsworkflowON) {
    LOG(DEBUG) << "Number of entries in track tree = " << mCurrTracksTreeEntry;

    if (mIsITSused) {
      prepareTracks();
    } else {
      prepareTPCTracks();
    }

    mMatchedTracks.clear();
    mOutTOFLabels.clear();
    mOutTPCLabels.clear();
    mOutITSLabels.clear();

    mTimerTot.Stop();
    LOGF(INFO, "Timing prepare tracks: Cpu: %.3e s Real: %.3e s in %d slots", mTimerTot.CpuTime(), mTimerTot.RealTime(), mTimerTot.Counter() - 1);
    mTimerTot.Start();

    for (int sec = o2::constants::math::NSectors; sec--;) {
      LOG(INFO) << "Doing matching for sector " << sec << "...";
      if (mIsITSused) {
        doMatching(sec);
      } else {
        doMatchingForTPC(sec);
      }
      LOG(INFO) << "...done. Now check the best matches";
      selectBestMatches();
    }
  }

  // we do the matching per entry of the TPCITS matched tracks tree
  while (!mIsworkflowON && mCurrTracksTreeEntry + 1 < mInputTreeTracks->GetEntries()) { // we add "+1" because mCurrTracksTreeEntry starts from -1, and it is incremented in loadTracksNextChunk which is called by prepareTracks
    LOG(DEBUG) << "Number of entries in track tree = " << mCurrTracksTreeEntry;

    if (mIsITSused) {
      prepareTracks();
    } else {
      prepareTPCTracks();
    }

    mMatchedTracks.clear();
    mOutTOFLabels.clear();
    mOutTPCLabels.clear();
    mOutITSLabels.clear();

    mTimerTot.Stop();
    LOGF(INFO, "Timing prepare tracks: Cpu: %.3e s Real: %.3e s in %d slots", mTimerTot.CpuTime(), mTimerTot.RealTime(), mTimerTot.Counter() - 1);
    mTimerTot.Start();

    for (int sec = o2::constants::math::NSectors; sec--;) {
      LOG(INFO) << "Doing matching for sector " << sec << "...";
      if (mIsITSused) {
        doMatching(sec);
      } else {
        doMatchingForTPC(sec);
      }
      LOG(INFO) << "...done. Now check the best matches";
      selectBestMatches();
    }

    mTimerTot.Stop();
    LOGF(INFO, "Timing Do Matching: Cpu: %.3e s Real: %.3e s in %d slots", mTimerTot.CpuTime(), mTimerTot.RealTime(), mTimerTot.Counter() - 1);
    mTimerTot.Start();

    fill();
  }

#ifdef _ALLOW_TOF_DEBUG_
  if (mDBGFlags)
    mDBGOut.reset();
#endif

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
void MatchTOF::run(const gsl::span<const o2::dataformats::TrackTPCITS>& trackArray, const gsl::span<const Cluster>& clusterArray, const o2::dataformats::MCTruthContainer<o2::MCCompLabel>& toflab, const gsl::span<const o2::MCCompLabel>& itslab, const gsl::span<const o2::MCCompLabel>& tpclab)
{
  mIsITSused = true;
  mTracksArrayInp = trackArray;
  mTOFClustersArrayInp = clusterArray;
  mIsworkflowON = kTRUE;
  mTOFClusLabels = toflab;
  mTPCLabels = tpclab;
  mITSLabels = itslab;

  mMCTruthON = (mTOFClusLabels.getNElements() && mTPCLabels.size() && mITSLabels.size());
  mWFInputAttached = true;
  mSAInitDone = true;
  run();
}
//______________________________________________
void MatchTOF::run(const gsl::span<const o2::tpc::TrackTPC>& trackArray, const gsl::span<const Cluster>& clusterArray, const o2::dataformats::MCTruthContainer<o2::MCCompLabel>& toflab, const gsl::span<const o2::MCCompLabel>& tpclab)
{
  mIsITSused = false;
  mTPCTracksArrayInp = trackArray;
  mTOFClustersArrayInp = clusterArray;
  mIsworkflowON = kTRUE;
  mTOFClusLabels = toflab;
  mTPCLabels = tpclab;

  mMCTruthON = (mTOFClusLabels.getNElements() && mTPCLabels.size());
  mWFInputAttached = true;
  mSAInitDone = true;

  run();
}
//______________________________________________
void MatchTOF::init()
{
  ///< initizalizations
  mIsITSused = true;

  if (mSAInitDone) {
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
      mOutputTree->Branch(mOutITSMCTruthBranchName.data(), &mOutITSLabels);
      LOG(INFO) << "ITS Tracks Labels branch: " << mOutITSMCTruthBranchName;
      mOutputTree->Branch(mOutTPCMCTruthBranchName.data(), &mOutTPCLabels);
      LOG(INFO) << "TPC Tracks Labels branch: " << mOutTPCMCTruthBranchName;
      mOutputTree->Branch(mOutTOFMCTruthBranchName.data(), &mOutTOFLabels);
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

#ifdef _ALLOW_TOF_DEBUG_
  // debug streamer
  if (mDBGFlags) {
    mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(mDebugTreeFileName.data(), "recreate");
  }
#endif

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

  mIsITSused = false;

  if (mSAInitDone) {
    LOG(ERROR) << "Initialization was already done";
    return;
  }
  attachInputTreesTPConly();

  // create output branch with track-tof matching
  if (mOutputTree) {
    mOutputTree->Branch(mOutTracksBranchName.data(), &mMatchedTracks);
    LOG(INFO) << "Matched tracks will be stored in " << mOutTracksBranchName << " branch of tree "
              << mOutputTree->GetName();
    if (mMCTruthON) {
      mOutputTree->Branch(mOutTPCMCTruthBranchName.data(), &mOutTPCLabels);
      LOG(INFO) << "TPC Tracks Labels branch: " << mOutTPCMCTruthBranchName;
      mOutputTree->Branch(mOutTOFMCTruthBranchName.data(), &mOutTOFLabels);
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

#ifdef _ALLOW_TOF_DEBUG_
  // debug streamer
  if (mDBGFlags) {
    mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(mDebugTreeFileName.data(), "recreate");
  }
#endif

  mSAInitDone = true;

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
  mInputTreeTracks->SetBranchAddress(mTracksBranchName.data(), &mTracksArrayInpVect);
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
    mInputTreeTracks->SetBranchAddress(mTPCMCTruthBranchName.data(), &mTPCLabelsVect);
    LOG(INFO) << "Found TPC tracks MCLabels branch " << mTPCMCTruthBranchName.data();
  } else {
    mMCTruthON = false;
  }
  if (mInputTreeTracks->GetBranch(mITSMCTruthBranchName.data())) {
    mInputTreeTracks->SetBranchAddress(mITSMCTruthBranchName.data(), &mITSLabelsVect);
    LOG(INFO) << "Found ITS tracks MCLabels branch " << mITSMCTruthBranchName.data();
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
    mTreeTPCTracks->SetBranchAddress(mOutTPCTrackMCTruthBranchName.data(), &mTPCLabelsVect);
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

  mNumOfTracks = mTracksArrayInp.size();
  if (mNumOfTracks == 0) {
    return false; // no tracks to be matched
  }
  mMatchedTracksIndex.resize(mNumOfTracks);
  std::fill(mMatchedTracksIndex.begin(), mMatchedTracksIndex.end(), -1); // initializing all to -1

  // copy the track params, propagate to reference X and build sector tables
  mTracksWork.clear();
  mLTinfos.clear();
  mTracksWork.reserve(mNumOfTracks);
  mLTinfos.reserve(mNumOfTracks);
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
  float maxInvPt = abs(bzField) > 0.1 ? 1. / (abs(bzField) * 0.05) : 999.;

  LOG(DEBUG) << "\n\nWe have %d tracks to try to match to TOF: " << mNumOfTracks;
  int nNotPropagatedToTOF = 0;
  for (int it = 0; it < mNumOfTracks; it++) {
    const o2::dataformats::TrackTPCITS& trcOrig = mTracksArrayInp[it]; // TODO: check if we cannot directly use the o2::track::TrackParCov class instead of o2::dataformats::TrackTPCITS, and then avoid the casting below; this is the track at the vertex
    std::array<float, 3> globalPos;

    // create working copy of track param
    mTracksWork.emplace_back(std::make_pair(trcOrig.getParamOut(), trcOrig.getTimeMUS())); //, mCurrTracksTreeEntry, it);
    mLTinfos.emplace_back(trcOrig.getLTIntegralOut());
    // make a copy of the TPC track that we have to propagate
    //o2::tpc::TrackTPC* trc = new o2::tpc::TrackTPC(trcTPCOrig); // this would take the TPCout track
    //auto& trc = mTracksWork.back(); // with this we take the TPCITS track propagated to the vertex
    auto& trc = mTracksWork.back().first; // with this we take the TPCITS track propagated to the vertex
    auto& intLT = mLTinfos.back();        // we get the integrated length from TPC-ITC outward propagation

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
    LOG(DEBUG) << "The track will go to sector " << o2::math_utils::angle2Sector(TMath::ATan2(globalPos[1], globalPos[0]));

    mTracksSectIndexCache[o2::math_utils::angle2Sector(TMath::ATan2(globalPos[1], globalPos[0]))].push_back(it);
    //delete trc; // Check: is this needed?
  }

  LOG(INFO) << "Total number of tracks = " << mNumOfTracks << ", Number of tracks that failed to be propagated to TOF = " << nNotPropagatedToTOF;

  // sort tracks in each sector according to their time (increasing in time)
  for (int sec = o2::constants::math::NSectors; sec--;) {
    auto& indexCache = mTracksSectIndexCache[sec];
    LOG(INFO) << "Sorting sector" << sec << " | " << indexCache.size() << " tracks";
    if (!indexCache.size()) {
      continue;
    }
    std::sort(indexCache.begin(), indexCache.end(), [this](int a, int b) {
      auto& trcA = mTracksWork[a].second;
      auto& trcB = mTracksWork[b].second;
      return ((trcA.getTimeStamp() - mSigmaTimeCut * trcA.getTimeStampError()) - (trcB.getTimeStamp() - mSigmaTimeCut * trcB.getTimeStampError()) < 0.);
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
bool MatchTOF::prepareTPCTracks()
{
  ///< prepare the tracks that we want to match to TOF

  if (!mIsworkflowON && !loadTPCTracksNextChunk()) {
    return false;
  }

  mNumOfTracks = mTPCTracksArrayInp.size();
  if (mNumOfTracks == 0) {
    return false; // no tracks to be matched
  }
  mMatchedTracksIndex.resize(mNumOfTracks);
  std::fill(mMatchedTracksIndex.begin(), mMatchedTracksIndex.end(), -1); // initializing all to -1

  // copy the track params, propagate to reference X and build sector tables
  mTracksWork.clear();
  mTracksWork.reserve(mNumOfTracks);
  mSideTPC.clear();
  mSideTPC.reserve(mNumOfTracks);
  mExtraTPCFwdTime.clear();
  mExtraTPCFwdTime.reserve(mNumOfTracks);

  for (int sec = o2::constants::math::NSectors; sec--;) {
    mTPCTracksSectIndexCache[sec].clear();
    mTPCTracksSectIndexCache[sec].reserve(100 + 1.2 * mNumOfTracks / o2::constants::math::NSectors);
  }

  // getting Bz (mag field)
  auto o2field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
  float bzField = o2field->solenoidField(); // magnetic field in kGauss
  float maxInvPt = abs(bzField) > 0.1 ? 1. / (abs(bzField) * 0.05) : 999.;
  int nclustersMin = 0;
  LOG(INFO) << "Max track Inv pT allowed = " << maxInvPt;
  LOG(INFO) << "Min track Nclusters allowed = " << nclustersMin;

  LOG(DEBUG) << "\n\nWe have %d tracks to try to match to TOF: " << mNumOfTracks;
  int nNotPropagatedToTOF = 0;
  for (int it = 0; it < mNumOfTracks; it++) {
    const o2::tpc::TrackTPC& trcOrig = mTPCTracksArrayInp[it]; // TODO: check if we cannot directly use the o2::track::TrackParCov class instead of o2::dataformats::TrackTPCITS, and then avoid the casting below; this is the track at the vertex
    std::array<float, 3> globalPos;

    // create working copy of track param
    timeEst timeInfo;
    // set
    timeInfo.setTimeStamp(trcOrig.getTime0() * o2::tpc::ParameterElectronics::Instance().ZbinWidth);
    timeInfo.setTimeStampError((trcOrig.getDeltaTBwd() + 5) * o2::tpc::ParameterElectronics::Instance().ZbinWidth);
    mSideTPC.push_back(trcOrig.hasASideClustersOnly() ? 1 : (trcOrig.hasCSideClustersOnly() ? -1 : 0));
    mExtraTPCFwdTime.push_back((trcOrig.getDeltaTFwd() + 5) * o2::tpc::ParameterElectronics::Instance().ZbinWidth);

    o2::track::TrackLTIntegral intLT0; //mTPCTracksWork.back().getLTIntegralOut(); // we get the integrated length from TPC-ITC outward propagation
    // make a copy of the TPC track that we have to propagate
    //o2::tpc::TrackTPC* trc = new o2::tpc::TrackTPC(trcTPCOrig); // this would take the TPCout track
    mTracksWork.emplace_back(std::make_pair(trcOrig.getOuterParam(), timeInfo));
    auto& trc = mTracksWork.back().first;
    auto& intLT = mLTinfos.emplace_back(intLT0);

    if (trcOrig.getNClusters() < nclustersMin) {
      nNotPropagatedToTOF++;
      continue;
    }

    if (std::abs(trc.getQ2Pt()) > maxInvPt) { // tpc-its track outward propagation did not reach outer ref.radius, skip this track
      nNotPropagatedToTOF++;
      continue;
    }

    //    printf("N clusters = %d\n",trcOrig.getNClusters());

#ifdef _ALLOW_TOF_DEBUG_
    // propagate to matching Xref
    trc.getXYZGlo(globalPos);
    LOG(INFO) << "Global coordinates Before propagating to 371 cm: globalPos[0] = " << globalPos[0] << ", globalPos[1] = " << globalPos[1] << ", globalPos[2] = " << globalPos[2];
    LOG(INFO) << "Radius xy Before propagating to 371 cm = " << TMath::Sqrt(globalPos[0] * globalPos[0] + globalPos[1] * globalPos[1]);
    LOG(INFO) << "Radius xyz Before propagating to 371 cm = " << TMath::Sqrt(globalPos[0] * globalPos[0] + globalPos[1] * globalPos[1] + globalPos[2] * globalPos[2]);
    // the "very rough" propagation worked; now we can propagate considering also the cov matrix
#endif

    if (!propagateToRefXWithoutCov(trc, mXRef, 10, bzField)) { // we first propagate to 371 cm without considering the covariance matrix
      nNotPropagatedToTOF++;
      continue;
    }

    if (trc.getX() < o2::globaltracking::MatchTPCITS::XTPCOuterRef - 1.) {
      if (!propagateToRefX(trc, o2::globaltracking::MatchTPCITS::XTPCOuterRef, 10, intLT) || TMath::Abs(trc.getZ()) > Geo::MAXHZTOF) { // we check that the propagation with the cov matrix worked; CHECK: can it happen that it does not if the propagation without the errors succeeded?
        nNotPropagatedToTOF++;
        continue;
      }
    }

    // the "rough" propagation worked; now we can propagate considering also the cov matrix
    if (!propagateToRefX(trc, mXRef, 2, intLT) || TMath::Abs(trc.getZ()) > Geo::MAXHZTOF) { // we check that the propagation with the cov matrix worked; CHECK: can it happen that it does not if the propagation without the errors succeeded?
      nNotPropagatedToTOF++;
      continue;
    }

    trc.getXYZGlo(globalPos);

#ifdef _ALLOW_TOF_DEBUG_
    LOG(INFO) << "Global coordinates After propagating to 371 cm: globalPos[0] = " << globalPos[0] << ", globalPos[1] = " << globalPos[1] << ", globalPos[2] = " << globalPos[2];
    LOG(INFO) << "Radius xy After propagating to 371 cm = " << TMath::Sqrt(globalPos[0] * globalPos[0] + globalPos[1] * globalPos[1]);
    LOG(INFO) << "Radius xyz After propagating to 371 cm = " << TMath::Sqrt(globalPos[0] * globalPos[0] + globalPos[1] * globalPos[1] + globalPos[2] * globalPos[2]);
    LOG(INFO) << "The track will go to sector " << o2::math_utils::angle2Sector(TMath::ATan2(globalPos[1], globalPos[0]));
#endif

    mTracksSectIndexCache[o2::math_utils::angle2Sector(TMath::ATan2(globalPos[1], globalPos[0]))].push_back(it);
    //delete trc; // Check: is this needed?
  }

  LOG(INFO) << "Total number of tracks = " << mNumOfTracks << ", Number of tracks that failed to be propagated to TOF = " << nNotPropagatedToTOF;

  // sort tracks in each sector according to their time (increasing in time)
  for (int sec = o2::constants::math::NSectors; sec--;) {
    auto& indexCache = mTracksSectIndexCache[sec];
    LOG(INFO) << "Sorting sector" << sec << " | " << indexCache.size() << " tracks";
    if (!indexCache.size()) {
      continue;
    }
    std::sort(indexCache.begin(), indexCache.end(), [this](int a, int b) {
      auto& trcA = mTracksWork[a].second;
      auto& trcB = mTracksWork[b].second;
      return ((trcA.getTimeStamp() - trcA.getTimeStampError()) - (trcB.getTimeStamp() - trcB.getTimeStampError()) < 0.);
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
    int nClusterInCurrentChunk = mTOFClustersArrayInp.size();
    LOG(DEBUG) << "nClusterInCurrentChunk = " << nClusterInCurrentChunk;
    mNumOfClusters += nClusterInCurrentChunk;
    for (int it = 0; it < nClusterInCurrentChunk; it++) {
      const Cluster& clOrig = mTOFClustersArrayInp[it];
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
      const Cluster& clOrig = mTOFClustersArrayInp[it];
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
    mTracksArrayInp = gsl::span<const o2::dataformats::TrackTPCITS>{*mTracksArrayInpVect};
    LOG(INFO) << "Loading tracks entry " << mCurrTracksTreeEntry << " -> " << mTracksArrayInp.size()
              << " tracks";
    if (!mTracksArrayInp.size()) {
      continue;
    }
    if (mMCTruthON) {
      mITSLabels = gsl::span<const o2::MCCompLabel>{*mITSLabelsVect};
      mTPCLabels = gsl::span<const o2::MCCompLabel>{*mTPCLabelsVect};
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
    mTOFClustersArrayInp = gsl::span<const Cluster>{*mTOFClustersArrayInpVect};
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
//______________________________________________
void MatchTOF::doMatching(int sec)
{

  ///< do the real matching per sector
  mMatchedTracksPairs.clear(); // new sector

  auto& cacheTOF = mTOFClusSectIndexCache[sec]; // array of cached TOF cluster indices for this sector; reminder: they are ordered in time!
  auto& cacheTrk = mTracksSectIndexCache[sec];  // array of cached tracks indices for this sector; reminder: they are ordered in time!
  int nTracks = cacheTrk.size(), nTOFCls = cacheTOF.size();
  LOG(INFO) << "Matching sector " << sec << ": number of tracks: " << nTracks << ", number of TOF clusters: " << nTOFCls;
  if (!nTracks || !nTOFCls) {
    return;
  }
  int itof0 = 0;                          // starting index in TOF clusters for matching of the track
  int detId[2][5];                        // at maximum one track can fall in 2 strips during the propagation; the second dimention of the array is the TOF det index
  float deltaPos[2][3];                   // at maximum one track can fall in 2 strips during the propagation; the second dimention of the array is the residuals
  o2::track::TrackLTIntegral trkLTInt[2]; // Here we store the integrated track length and time for the (max 2) matched strips
  int nStepsInsideSameStrip[2] = {0, 0};  // number of propagation steps in the same strip (since we have maximum 2 strips, it has dimention = 2)
  float deltaPosTemp[3];
  std::array<float, 3> pos;
  std::array<float, 3> posBeforeProp;
  float posFloat[3];

  // prematching for TPC only tracks (identify BC candidate to correct z for TPC track accordingly to v_drift)

  LOG(DEBUG) << "Trying to match %d tracks" << cacheTrk.size();
  for (int itrk = 0; itrk < cacheTrk.size(); itrk++) {
    for (int ii = 0; ii < 2; ii++) {
      detId[ii][2] = -1; // before trying to match, we need to inizialize the detId corresponding to the strip number to -1; this is the array that we will use to save the det id of the maximum 2 strips matched
      nStepsInsideSameStrip[ii] = 0;
    }
    int nStripsCrossedInPropagation = 0; // how many strips were hit during the propagation
    auto& trackWork = mTracksWork[cacheTrk[itrk]];
    auto& trefTrk = trackWork.first;
    auto& intLT = mLTinfos[cacheTrk[itrk]];

    //    Printf("intLT (before doing anything): length = %f, time (Pion) = %f", intLT.getL(), intLT.getTOF(o2::track::PID::Pion));
    float minTrkTime = (trackWork.second.getTimeStamp() - mSigmaTimeCut * trackWork.second.getTimeStampError()) * 1.E6;             // minimum time in ps
    float maxTrkTime = (trackWork.second.getTimeStamp() + mSigmaTimeCut * trackWork.second.getTimeStampError()) * 1.E6;             // maximum time in ps
    int istep = 1;                                                                                                                  // number of steps
    float step = 1.0;                                                                                                               // step size in cm
                                                                                                                                    //uncomment for local debug
                                                                                                                                    /*
																//trefTrk.getXYZGlo(posBeforeProp);
																//float posBeforeProp[3] = {trefTrk.getX(), trefTrk.getY(), trefTrk.getZ()}; // in local ref system
																//printf("Global coordinates: posBeforeProp[0] = %f, posBeforeProp[1] = %f, posBeforeProp[2] = %f\n", posBeforeProp[0], posBeforeProp[1], posBeforeProp[2]);
																//Printf("Radius xy = %f", TMath::Sqrt(posBeforeProp[0]*posBeforeProp[0] + posBeforeProp[1]*posBeforeProp[1]));
																//Printf("Radius xyz = %f", TMath::Sqrt(posBeforeProp[0]*posBeforeProp[0] + posBeforeProp[1]*posBeforeProp[1] + posBeforeProp[2]*posBeforeProp[2]));
																*/

#ifdef _ALLOW_TOF_DEBUG_
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

    int detIdTemp[5] = {-1, -1, -1, -1, -1}; // TOF detector id at the current propagation point

    double reachedPoint = mXRef + istep * step;

    while (propagateToRefX(trefTrk, reachedPoint, step, intLT) && nStripsCrossedInPropagation <= 2 && reachedPoint < Geo::RMAX) {
      // while (o2::base::Propagator::Instance()->PropagateToXBxByBz(trefTrk,  mXRef + istep * step, MAXSNP, step, 1, &intLT) && nStripsCrossedInPropagation <= 2 && mXRef + istep * step < Geo::RMAX) {

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

      for (int idet = 0; idet < 5; idet++) {
        detIdTemp[idet] = -1;
      }

      Geo::getPadDxDyDz(posFloat, detIdTemp, deltaPosTemp);

      reachedPoint += step;

      if (detIdTemp[2] == -1) {
        continue;
      }

      // to reduce the active region of the strip -> uncomment these lines
      // float yresidual = TMath::Abs(deltaPosTemp[1]);
      // if(yresidual > 0.55){
      // 	reachedPoint += step;
      // 	continue;
      // }

      //      printf("res %f %f %f -- %f %f %f (%d)\n",deltaPosTemp[0],deltaPosTemp[1],deltaPosTemp[2],pos[0],pos[1],pos[2],detIdTemp[2]);

      // if you want to exit from the strip matched uncomment this line
      //      reachedPoint += 3.0; // go out from the strip at the next step

      //      printf("idet: %d %d %d %d %d\n",detIdTemp[0],detIdTemp[1],detIdTemp[2],detIdTemp[3],detIdTemp[4]);

      // uncomment below only for local debug; this will produce A LOT of output - one print per propagation step
      //Printf("detIdTemp[0] = %d, detIdTemp[1] = %d, detIdTemp[2] = %d, detIdTemp[3] = %d, detIdTemp[4] = %d", detIdTemp[0], detIdTemp[1], detIdTemp[2], detIdTemp[3], detIdTemp[4]);
      // if (nStripsCrossedInPropagation == 0) { // print in case you have a useful propagation
      //   LOG(DEBUG) << "*********** We have crossed a strip during propagation!*********";
      //   LOG(DEBUG) << "Global coordinates: pos[0] = " << pos[0] << ", pos[1] = " << pos[1] << ", pos[2] = " << pos[2];
      //   LOG(DEBUG) << "detIdTemp[0] = " << detIdTemp[0] << ", detIdTemp[1] = " << detIdTemp[1] << ", detIdTemp[2] = " << detIdTemp[2] << ", detIdTemp[3] = " << detIdTemp[3] << ", detIdTemp[4] = " << detIdTemp[4];
      //   LOG(DEBUG) << "deltaPosTemp[0] = " << deltaPosTemp[0] << ", deltaPosTemp[1] = " << deltaPosTemp[1] << " deltaPosTemp[2] = " << deltaPosTemp[2];
      // } else {
      //   LOG(DEBUG) << "*********** We have NOT crossed a strip during propagation!*********";
      //   LOG(DEBUG) << "Global coordinates: pos[0] = " << pos[0] << ", pos[1] = " << pos[1] << ", pos[2] = " << pos[2];
      //   LOG(DEBUG) << "detIdTemp[0] = " << detIdTemp[0] << ", detIdTemp[1] = " << detIdTemp[1] << ", detIdTemp[2] = " << detIdTemp[2] << ", detIdTemp[3] = " << detIdTemp[3] << ", detIdTemp[4] = " << detIdTemp[4];
      //   LOG(DEBUG) << "deltaPosTemp[0] = " << deltaPosTemp[0] << ", deltaPosTemp[1] = " << deltaPosTemp[1] << " deltaPosTemp[2] = " << deltaPosTemp[2];
      // }

      // check if after the propagation we are in a TOF strip
      // we ended in a TOF strip
      // LOG(DEBUG) << "nStripsCrossedInPropagation = " << nStripsCrossedInPropagation << ", detId[nStripsCrossedInPropagation][0] = " << detId[nStripsCrossedInPropagation][0] << ", detIdTemp[0] = " << detIdTemp[0] << ", detId[nStripsCrossedInPropagation][1] = " << detId[nStripsCrossedInPropagation][1] << ", detIdTemp[1] = " << detIdTemp[1] << ", detId[nStripsCrossedInPropagation][2] = " << detId[nStripsCrossedInPropagation][2] << ", detIdTemp[2] = " << detIdTemp[2];
      if (nStripsCrossedInPropagation == 0 ||                                                                                                                                                                                            // we are crossing a strip for the first time...
          (nStripsCrossedInPropagation >= 1 && (detId[nStripsCrossedInPropagation - 1][0] != detIdTemp[0] || detId[nStripsCrossedInPropagation - 1][1] != detIdTemp[1] || detId[nStripsCrossedInPropagation - 1][2] != detIdTemp[2]))) { // ...or we are crossing a new strip
        if (nStripsCrossedInPropagation == 0) {
          LOG(DEBUG) << "We cross a strip for the first time";
        }
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
        //          Printf("intLT (after matching to strip %d): length = %f, time (Pion) = %f", nStripsCrossedInPropagation - 1, trkLTInt[nStripsCrossedInPropagation - 1].getL(), trkLTInt[nStripsCrossedInPropagation - 1].getTOF(o2::track::PID::Pion));
        nStepsInsideSameStrip[nStripsCrossedInPropagation - 1]++;
      } else { // a further propagation step in the same strip -> update info (we sum up on all matching with strip - we will divide for the number of steps a bit below)
        // N.B. the integrated length and time are taken (at least for now) from the first time we crossed the strip, so here we do nothing with those
        deltaPos[nStripsCrossedInPropagation - 1][0] += deltaPosTemp[0] + (detIdTemp[4] - detId[nStripsCrossedInPropagation - 1][4]) * Geo::XPAD; // residual in x
        deltaPos[nStripsCrossedInPropagation - 1][1] += deltaPosTemp[1];                                                                          // residual in y
        deltaPos[nStripsCrossedInPropagation - 1][2] += deltaPosTemp[2] + (detIdTemp[3] - detId[nStripsCrossedInPropagation - 1][3]) * Geo::ZPAD; // residual in z
        nStepsInsideSameStrip[nStripsCrossedInPropagation - 1]++;
      }
    }
    //    LOG(DEBUG) << "while done, we propagated track " << itrk << " in %d strips" << nStripsCrossedInPropagation;
    //    LOG(INFO) << "while done, we propagated track " << itrk << " in %d strips" << nStripsCrossedInPropagation;

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
      //   LOG(DEBUG) << "matched strip " << imatch << ": deltaPos[0] = " << deltaPos[imatch][0] << ", deltaPos[1] = " << deltaPos[imatch][1] << ", deltaPos[2] = " << deltaPos[imatch][2] << ", residual (x, z) = " << TMath::Sqrt(deltaPos[imatch][0] * deltaPos[imatch][0] + deltaPos[imatch][2] * deltaPos[imatch][2]);
    }

    if (nStripsCrossedInPropagation == 0) {
      continue; // the track never hit a TOF strip during the propagation
    }
    bool foundCluster = false;
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

      // compute fine correction using cluster position instead of pad center
      // this because in case of multiple-hit cluster position is averaged on all pads contributing to the cluster (then error position matrix can be used for Chi2 if nedeed)
      int ndigits = 1;
      float posCorr[3] = {0, 0, 0};

      if (trefTOF.isBitSet(Cluster::kLeft)) {
        posCorr[0] += Geo::XPAD, ndigits++;
      }
      if (trefTOF.isBitSet(Cluster::kUpLeft)) {
        posCorr[0] += Geo::XPAD, posCorr[2] -= Geo::ZPAD, ndigits++;
      }
      if (trefTOF.isBitSet(Cluster::kDownLeft)) {
        posCorr[0] += Geo::XPAD, posCorr[2] += Geo::ZPAD, ndigits++;
      }
      if (trefTOF.isBitSet(Cluster::kUp)) {
        posCorr[2] -= Geo::ZPAD, ndigits++;
      }
      if (trefTOF.isBitSet(Cluster::kDown)) {
        posCorr[2] += Geo::ZPAD, ndigits++;
      }
      if (trefTOF.isBitSet(Cluster::kRight)) {
        posCorr[0] -= Geo::XPAD, ndigits++;
      }
      if (trefTOF.isBitSet(Cluster::kUpRight)) {
        posCorr[0] -= Geo::XPAD, posCorr[2] -= Geo::ZPAD, ndigits++;
      }
      if (trefTOF.isBitSet(Cluster::kDownRight)) {
        posCorr[0] -= Geo::XPAD, posCorr[2] += Geo::ZPAD, ndigits++;
      }

      float ndifInv = 1. / ndigits;
      if (ndigits > 1) {
        posCorr[0] *= ndifInv;
        posCorr[1] *= ndifInv;
        posCorr[2] *= ndifInv;
      }

      int trackIdTOF;
      int eventIdTOF;
      int sourceIdTOF;
      for (auto iPropagation = 0; iPropagation < nStripsCrossedInPropagation; iPropagation++) {
        LOG(DEBUG) << "TOF Cluster [" << itof << ", " << cacheTOF[itof] << "]:      indices   = " << indices[0] << ", " << indices[1] << ", " << indices[2] << ", " << indices[3] << ", " << indices[4];
        LOG(DEBUG) << "Propagated Track [" << itrk << ", " << cacheTrk[itrk] << "]: detId[" << iPropagation << "]  = " << detId[iPropagation][0] << ", " << detId[iPropagation][1] << ", " << detId[iPropagation][2] << ", " << detId[iPropagation][3] << ", " << detId[iPropagation][4];
        float resX = deltaPos[iPropagation][0] - (indices[4] - detId[iPropagation][4]) * Geo::XPAD + posCorr[0]; // readjusting the residuals due to the fact that the propagation fell in a pad that was not exactly the one of the cluster
        float resZ = deltaPos[iPropagation][2] - (indices[3] - detId[iPropagation][3]) * Geo::ZPAD + posCorr[2]; // readjusting the residuals due to the fact that the propagation fell in a pad that was not exactly the one of the cluster
        float res = TMath::Sqrt(resX * resX + resZ * resZ);

        LOG(DEBUG) << "resX = " << resX << ", resZ = " << resZ << ", res = " << res;
#ifdef _ALLOW_TOF_DEBUG_
        fillTOFmatchTree("match0", cacheTOF[itof], indices[0], indices[1], indices[2], indices[3], indices[4], cacheTrk[itrk], iPropagation, detId[iPropagation][0], detId[iPropagation][1], detId[iPropagation][2], detId[iPropagation][3], detId[iPropagation][4], resX, resZ, res, trackWork, trkLTInt[iPropagation].getL(), trkLTInt[iPropagation].getTOF(o2::track::PID::Pion), trefTOF.getTime());
        int tofLabelTrackID[3] = {-1, -1, -1};
        int tofLabelEventID[3] = {-1, -1, -1};
        int tofLabelSourceID[3] = {-1, -1, -1};
        if (mMCTruthON) {
          const auto& labelsTOF = mTOFClusLabels.getLabels(mTOFClusSectIndexCache[indices[0]][itof]);
          for (int ilabel = 0; ilabel < labelsTOF.size(); ilabel++) {
            tofLabelTrackID[ilabel] = labelsTOF[ilabel].getTrackID();
            tofLabelEventID[ilabel] = labelsTOF[ilabel].getEventID();
            tofLabelSourceID[ilabel] = labelsTOF[ilabel].getSourceID();
          }
          auto labelTPC = mTPCLabels[mTracksSectIndexCache[sec][itrk]];
          auto labelITS = mITSLabels[mTracksSectIndexCache[indices[0]][itrk]];
          fillTOFmatchTreeWithLabels("matchPossibleWithLabels", cacheTOF[itof], indices[0], indices[1], indices[2], indices[3], indices[4], cacheTrk[itrk], iPropagation, detId[iPropagation][0], detId[iPropagation][1], detId[iPropagation][2], detId[iPropagation][3], detId[iPropagation][4], resX, resZ, res, trackWork, labelTPC.getTrackID(), labelTPC.getEventID(), labelTPC.getSourceID(), labelITS.getTrackID(), labelITS.getEventID(), labelITS.getSourceID(), tofLabelTrackID[0], tofLabelEventID[0], tofLabelSourceID[0], tofLabelTrackID[1], tofLabelEventID[1], tofLabelSourceID[1], tofLabelTrackID[2], tofLabelEventID[2], tofLabelSourceID[2], trkLTInt[iPropagation].getL(), trkLTInt[iPropagation].getTOF(o2::track::PID::Pion), trefTOF.getTime());
        }
#endif
        if (indices[0] != detId[iPropagation][0]) {
          continue;
        }
        if (indices[1] != detId[iPropagation][1]) {
          continue;
        }
        if (indices[2] != detId[iPropagation][2]) {
          continue;
        }
        float chi2 = res; // TODO: take into account also the time!
#ifdef _ALLOW_TOF_DEBUG_
        fillTOFmatchTree("match1", cacheTOF[itof], indices[0], indices[1], indices[2], indices[3], indices[4], cacheTrk[itrk], iPropagation, detId[iPropagation][0], detId[iPropagation][1], detId[iPropagation][2], detId[iPropagation][3], detId[iPropagation][4], resX, resZ, res, trackWork, trkLTInt[iPropagation].getL(), trkLTInt[iPropagation].getTOF(o2::track::PID::Pion), trefTOF.getTime());
        if (mMCTruthON) {
          auto labelTPC = mTPCLabels[mTracksSectIndexCache[sec][itrk]];
          auto labelITS = mITSLabels[mTracksSectIndexCache[indices[0]][itrk]];
          fillTOFmatchTreeWithLabels("matchOkWithLabels", cacheTOF[itof], indices[0], indices[1], indices[2], indices[3], indices[4], cacheTrk[itrk], iPropagation, detId[iPropagation][0], detId[iPropagation][1], detId[iPropagation][2], detId[iPropagation][3], detId[iPropagation][4], resX, resZ, res, trackWork, labelTPC.getTrackID(), labelTPC.getEventID(), labelTPC.getSourceID(), labelITS.getTrackID(), labelITS.getEventID(), labelITS.getSourceID(), tofLabelTrackID[0], tofLabelEventID[0], tofLabelSourceID[0], tofLabelTrackID[1], tofLabelEventID[1], tofLabelSourceID[1], tofLabelTrackID[2], tofLabelEventID[2], tofLabelSourceID[2], trkLTInt[iPropagation].getL(), trkLTInt[iPropagation].getTOF(o2::track::PID::Pion), trefTOF.getTime());
        }
#endif

        if (res < mSpaceTolerance) { // matching ok!
          LOG(DEBUG) << "MATCHING FOUND: We have a match! between track " << mTracksSectIndexCache[indices[0]][itrk] << " and TOF cluster " << mTOFClusSectIndexCache[indices[0]][itof];
          foundCluster = true;
          // set event indexes (to be checked)
          evIdx eventIndexTOFCluster(trefTOF.getEntryInTree(), mTOFClusSectIndexCache[indices[0]][itof]);
          evGIdx eventIndexTracks(mCurrTracksTreeEntry, {uint32_t(mTracksSectIndexCache[indices[0]][itrk]), o2::dataformats::GlobalTrackID::ITSTPC});
          mMatchedTracksPairs.emplace_back(eventIndexTOFCluster, chi2, trkLTInt[iPropagation], eventIndexTracks); // TODO: check if this is correct!

#ifdef _ALLOW_TOF_DEBUG_
          if (mMCTruthON) {
            const auto& labelsTOF = mTOFClusLabels.getLabels(mTOFClusSectIndexCache[indices[0]][itof]);
            auto labelTPC = mTPCLabels[mTracksSectIndexCache[sec][itrk]];
            auto labelITS = mITSLabels[mTracksSectIndexCache[indices[0]][itrk]];
            for (int ilabel = 0; ilabel < labelsTOF.size(); ilabel++) {
              LOG(DEBUG) << "TOF label " << ilabel << labelsTOF[ilabel];
            }
            LOG(DEBUG) << "TPC label " << labelTPC;
            LOG(DEBUG) << "ITS label " << labelITS;
            fillTOFmatchTreeWithLabels("matchOkWithLabelsInSpaceTolerance", cacheTOF[itof], indices[0], indices[1], indices[2], indices[3], indices[4], cacheTrk[itrk], iPropagation, detId[iPropagation][0], detId[iPropagation][1], detId[iPropagation][2], detId[iPropagation][3], detId[iPropagation][4], resX, resZ, res, trackWork, labelTPC.getTrackID(), labelTPC.getEventID(), labelTPC.getSourceID(), labelITS.getTrackID(), labelITS.getEventID(), labelITS.getSourceID(), tofLabelTrackID[0], tofLabelEventID[0], tofLabelSourceID[0], tofLabelTrackID[1], tofLabelEventID[1], tofLabelSourceID[1], tofLabelTrackID[2], tofLabelEventID[2], tofLabelSourceID[2], trkLTInt[iPropagation].getL(), trkLTInt[iPropagation].getTOF(o2::track::PID::Pion), trefTOF.getTime());
          }
#endif
        }
      }
    }
    if (!foundCluster && mMCTruthON) {
      auto labelTPC = mTPCLabels[mTracksSectIndexCache[sec][itrk]];
      LOG(DEBUG) << "We did not find any TOF cluster for track " << cacheTrk[itrk] << " (label = " << labelTPC << ", pt = " << trefTrk.getPt();
    }
  }
  return;
}
//______________________________________________
void MatchTOF::doMatchingForTPC(int sec)
{
  printf("here, DoMatch\n");
  auto& gasParam = o2::tpc::ParameterGas::Instance();
  float vdrift = gasParam.DriftV;

  ///< do the real matching per sector
  mMatchedTracksPairs.clear(); // new sector

  auto& cacheTOF = mTOFClusSectIndexCache[sec]; // array of cached TOF cluster indices for this sector; reminder: they are ordered in time!
  auto& cacheTrk = mTracksSectIndexCache[sec];  // array of cached tracks indices for this sector; reminder: they are ordered in time!
  int nTracks = cacheTrk.size(), nTOFCls = cacheTOF.size();
  LOG(INFO) << "Matching sector " << sec << ": number of tracks: " << nTracks << ", number of TOF clusters: " << nTOFCls;
  if (!nTracks || !nTOFCls) {
    return;
  }
  int itof0 = 0; // starting index in TOF clusters for matching of the track
  float deltaPosTemp[3];
  std::array<float, 3> pos;
  std::array<float, 3> posBeforeProp;
  float posFloat[3];

  // prematching for TPC only tracks (identify BC candidate to correct z for TPC track accordingly to v_drift)

  std::vector<unsigned long> BCcand;

  std::vector<int> nStripsCrossedInPropagation;
  std::vector<std::array<std::array<int, 5>, 2>> detId;
  std::vector<std::array<o2::track::TrackLTIntegral, 2>> trkLTInt;
  std::vector<std::array<std::array<float, 3>, 2>> deltaPos;
  std::vector<std::array<int, 2>> nStepsInsideSameStrip;

  LOG(DEBUG) << "Trying to match %d tracks" << cacheTrk.size();
  for (int itrk = 0; itrk < cacheTrk.size(); itrk++) {
    auto& trackWork = mTracksWork[cacheTrk[itrk]];
    auto& trefTrk = trackWork.first;
    auto& intLT = mLTinfos[cacheTrk[itrk]];

    int nBCcand = 1;
    BCcand.clear();
    nStripsCrossedInPropagation.clear();

    int side = mSideTPC[cacheTrk[itrk]];

    // look at BC candidates for the track
    itof0 = 0;
    double minTrkTime = (trackWork.second.getTimeStamp() - trackWork.second.getTimeStampError()) * 1.E6; // minimum time in ps
    double maxTrkTime = (trackWork.second.getTimeStamp() + mExtraTPCFwdTime[cacheTrk[itrk]]) * 1.E6;     // maximum time in ps

    for (auto itof = itof0; itof < nTOFCls; itof++) {
      auto& trefTOF = mTOFClusWork[cacheTOF[itof]];

      if (trefTOF.getTime() < minTrkTime) { // this cluster has a time that is too small for the current track, we will get to the next one
        itof0 = itof + 1;
        continue;
      }

      if (trefTOF.getTime() > maxTrkTime) { // this cluster has a time that is too large for the current track, close loop
        break;
      }

      if ((trefTOF.getZ() * side < 0) && ((side > 0) != (trackWork.first.getTgl() > 0))) {
        continue;
      }

      unsigned long bc = (unsigned long)(trefTOF.getTime() * Geo::BC_TIME_INPS_INV);

      bool isalreadyin = false;

      for (int k = 0; k < BCcand.size(); k++) {
        if (bc == BCcand[k]) {
          isalreadyin = true;
        }
      }

      if (!isalreadyin) {
        BCcand.emplace_back(bc);
        nStripsCrossedInPropagation.emplace_back(0);
      }
    }
    detId.clear();
    detId.reserve(BCcand.size());
    trkLTInt.clear();
    trkLTInt.reserve(BCcand.size());
    deltaPos.clear();
    deltaPos.reserve(BCcand.size());
    nStepsInsideSameStrip.clear();
    nStepsInsideSameStrip.reserve(BCcand.size());

    // printf("%d) ts_error = %f -- z_error = %f\n", itrk, trackWork.second.getTimeStampError(), trackWork.second.getTimeStampError() * vdrift);

    //    Printf("intLT (before doing anything): length = %f, time (Pion) = %f", intLT.getL(), intLT.getTOF(o2::track::PID::Pion));
    int istep = 1;    // number of steps
    float step = 1.0; // step size in cm
                      //uncomment for local debug
                      /*
    //trefTrk.getXYZGlo(posBeforeProp);
    //float posBeforeProp[3] = {trefTrk.getX(), trefTrk.getY(), trefTrk.getZ()}; // in local ref system
    //printf("Global coordinates: posBeforeProp[0] = %f, posBeforeProp[1] = %f, posBeforeProp[2] = %f\n", posBeforeProp[0], posBeforeProp[1], posBeforeProp[2]);
    //Printf("Radius xy = %f", TMath::Sqrt(posBeforeProp[0]*posBeforeProp[0] + posBeforeProp[1]*posBeforeProp[1]));
    //Printf("Radius xyz = %f", TMath::Sqrt(posBeforeProp[0]*posBeforeProp[0] + posBeforeProp[1]*posBeforeProp[1] + posBeforeProp[2]*posBeforeProp[2]));
    */

#ifdef _ALLOW_TOF_DEBUG_
    if (mDBGFlags) {
      (*mDBGOut) << "propOK"
                 << "track=" << trefTrk << "\n";
    }
#endif

    int detIdTemp[5] = {-1, -1, -1, -1, -1}; // TOF detector id at the current propagation point

    double reachedPoint = mXRef + istep * step;

    // initializing
    for (int ibc = 0; ibc < BCcand.size(); ibc++) {
      for (int ii = 0; ii < 2; ii++) {
        nStepsInsideSameStrip[ibc][ii] = 0;
        for (int iii = 0; iii < 5; iii++) {
          detId[ibc][ii][iii] = -1;
        }
      }
    }
    while (propagateToRefX(trefTrk, reachedPoint, step, intLT) && reachedPoint < Geo::RMAX) {
      // while (o2::base::Propagator::Instance()->PropagateToXBxByBz(trefTrk,  mXRef + istep * step, MAXSNP, step, 1, &intLT) && nStripsCrossedInPropagation <= 2 && mXRef + istep * step < Geo::RMAX) {

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

      reachedPoint += step;

      // check if you fall in a strip
      for (int ibc = 0; ibc < BCcand.size(); ibc++) {
        for (int idet = 0; idet < 5; idet++) {
          detIdTemp[idet] = -1;
        }

        if (side > 0) {
          posFloat[2] = pos[2] - vdrift * (trackWork.second.getTimeStamp() - BCcand[ibc] * Geo::BC_TIME_INPS * 1E-6);
        } else if (side < 0) {
          posFloat[2] = pos[2] + vdrift * (trackWork.second.getTimeStamp() - BCcand[ibc] * Geo::BC_TIME_INPS * 1E-6);
        } else {
          posFloat[2] = pos[2];
        }

        Geo::getPadDxDyDz(posFloat, detIdTemp, deltaPosTemp);

        if (detIdTemp[2] == -1) {
          continue;
        }

        if (nStripsCrossedInPropagation[ibc] == 0 ||                                                                                                                                                                                                                          // we are crossing a strip for the first time...
            (nStripsCrossedInPropagation[ibc] >= 1 && (detId[ibc][nStripsCrossedInPropagation[ibc] - 1][0] != detIdTemp[0] || detId[ibc][nStripsCrossedInPropagation[ibc] - 1][1] != detIdTemp[1] || detId[ibc][nStripsCrossedInPropagation[ibc] - 1][2] != detIdTemp[2]))) { // ...or we are crossing a new strip
          if (nStripsCrossedInPropagation[ibc] == 0) {
            LOG(DEBUG) << "We cross a strip for the first time";
          }
          if (nStripsCrossedInPropagation[ibc] == 2) {
            continue; // we have already matched 2 strips, we cannot match more
          }
          nStripsCrossedInPropagation[ibc]++;
        }

        //Printf("nStepsInsideSameStrip[nStripsCrossedInPropagation-1] = %d", nStepsInsideSameStrip[nStripsCrossedInPropagation - 1]);
        if (nStepsInsideSameStrip[ibc][nStripsCrossedInPropagation[ibc] - 1] == 0) {
          detId[ibc][nStripsCrossedInPropagation[ibc] - 1][0] = detIdTemp[0];
          detId[ibc][nStripsCrossedInPropagation[ibc] - 1][1] = detIdTemp[1];
          detId[ibc][nStripsCrossedInPropagation[ibc] - 1][2] = detIdTemp[2];
          detId[ibc][nStripsCrossedInPropagation[ibc] - 1][3] = detIdTemp[3];
          detId[ibc][nStripsCrossedInPropagation[ibc] - 1][4] = detIdTemp[4];
          deltaPos[ibc][nStripsCrossedInPropagation[ibc] - 1][0] = deltaPosTemp[0];
          deltaPos[ibc][nStripsCrossedInPropagation[ibc] - 1][1] = deltaPosTemp[1];
          deltaPos[ibc][nStripsCrossedInPropagation[ibc] - 1][2] = deltaPosTemp[2];

          trkLTInt[ibc][nStripsCrossedInPropagation[ibc] - 1] = intLT;
          //          Printf("intLT (after matching to strip %d): length = %f, time (Pion) = %f", nStripsCrossedInPropagation - 1, trkLTInt[nStripsCrossedInPropagation - 1].getL(), trkLTInt[nStripsCrossedInPropagation - 1].getTOF(o2::track::PID::Pion));
          nStepsInsideSameStrip[ibc][nStripsCrossedInPropagation[ibc] - 1]++;
        } else { // a further propagation step in the same strip -> update info (we sum up on all matching with strip - we will divide for the number of steps a bit below)
          // N.B. the integrated length and time are taken (at least for now) from the first time we crossed the strip, so here we do nothing with those
          deltaPos[ibc][nStripsCrossedInPropagation[ibc] - 1][0] += deltaPosTemp[0] + (detIdTemp[4] - detId[ibc][nStripsCrossedInPropagation[ibc] - 1][4]) * Geo::XPAD; // residual in x
          deltaPos[ibc][nStripsCrossedInPropagation[ibc] - 1][1] += deltaPosTemp[1];                                                                                    // residual in y
          deltaPos[ibc][nStripsCrossedInPropagation[ibc] - 1][2] += deltaPosTemp[2] + (detIdTemp[3] - detId[ibc][nStripsCrossedInPropagation[ibc] - 1][3]) * Geo::ZPAD; // residual in z
          nStepsInsideSameStrip[ibc][nStripsCrossedInPropagation[ibc] - 1]++;
        }
      }
    }
    for (int ibc = 0; ibc < BCcand.size(); ibc++) {
      float minTime = (BCcand[ibc] - 1) * Geo::BC_TIME_INPS;
      float maxTime = (BCcand[ibc] + 1) * Geo::BC_TIME_INPS;
      for (Int_t imatch = 0; imatch < nStripsCrossedInPropagation[ibc]; imatch++) {
        // we take as residual the average of the residuals along the propagation in the same strip
        deltaPos[ibc][imatch][0] /= nStepsInsideSameStrip[ibc][imatch];
        deltaPos[ibc][imatch][1] /= nStepsInsideSameStrip[ibc][imatch];
        deltaPos[ibc][imatch][2] /= nStepsInsideSameStrip[ibc][imatch];
        //   LOG(DEBUG) << "matched strip " << imatch << ": deltaPos[0] = " << deltaPos[imatch][0] << ", deltaPos[1] = " << deltaPos[imatch][1] << ", deltaPos[2] = " << deltaPos[imatch][2] << ", residual (x, z) = " << TMath::Sqrt(deltaPos[imatch][0] * deltaPos[imatch][0] + deltaPos[imatch][2] * deltaPos[imatch][2]);
      }

      if (nStripsCrossedInPropagation[ibc] == 0) {
        continue; // the track never hit a TOF strip during the propagation
      }

      bool foundCluster = false;
      itof0 = 0;
      for (auto itof = itof0; itof < nTOFCls; itof++) {
        //      printf("itof = %d\n", itof);
        auto& trefTOF = mTOFClusWork[cacheTOF[itof]];
        // compare the times of the track and the TOF clusters - remember that they both are ordered in time!
        //Printf("trefTOF.getTime() = %f, maxTrkTime = %f, minTrkTime = %f", trefTOF.getTime(), maxTrkTime, minTrkTime);

        if (trefTOF.getTime() < minTime) { // this cluster has a time that is too small for the current track, we will get to the next one
          //Printf("In trefTOF.getTime() < minTrkTime");
          itof0 = itof + 1; // but for the next track that we will check, we will ignore this cluster (the time is anyway too small)
          continue;
        }
        if (trefTOF.getTime() > maxTime) { // no more TOF clusters can be matched to this track
          break;
        }

        int mainChannel = trefTOF.getMainContributingChannel();
        int indices[5];
        Geo::getVolumeIndices(mainChannel, indices);

        // compute fine correction using cluster position instead of pad center
        // this because in case of multiple-hit cluster position is averaged on all pads contributing to the cluster (then error position matrix can be used for Chi2 if nedeed)
        int ndigits = 1;
        float posCorr[3] = {0, 0, 0};

        if (trefTOF.isBitSet(Cluster::kLeft)) {
          posCorr[0] += Geo::XPAD, ndigits++;
        }
        if (trefTOF.isBitSet(Cluster::kUpLeft)) {
          posCorr[0] += Geo::XPAD, posCorr[2] -= Geo::ZPAD, ndigits++;
        }
        if (trefTOF.isBitSet(Cluster::kDownLeft)) {
          posCorr[0] += Geo::XPAD, posCorr[2] += Geo::ZPAD, ndigits++;
        }
        if (trefTOF.isBitSet(Cluster::kUp)) {
          posCorr[2] -= Geo::ZPAD, ndigits++;
        }
        if (trefTOF.isBitSet(Cluster::kDown)) {
          posCorr[2] += Geo::ZPAD, ndigits++;
        }
        if (trefTOF.isBitSet(Cluster::kRight)) {
          posCorr[0] -= Geo::XPAD, ndigits++;
        }
        if (trefTOF.isBitSet(Cluster::kUpRight)) {
          posCorr[0] -= Geo::XPAD, posCorr[2] -= Geo::ZPAD, ndigits++;
        }
        if (trefTOF.isBitSet(Cluster::kDownRight)) {
          posCorr[0] -= Geo::XPAD, posCorr[2] += Geo::ZPAD, ndigits++;
        }

        float ndifInv = 1. / ndigits;
        if (ndigits > 1) {
          posCorr[0] *= ndifInv;
          posCorr[1] *= ndifInv;
          posCorr[2] *= ndifInv;
        }

        int trackIdTOF;
        int eventIdTOF;
        int sourceIdTOF;
        for (auto iPropagation = 0; iPropagation < nStripsCrossedInPropagation[ibc]; iPropagation++) {
          LOG(DEBUG) << "TOF Cluster [" << itof << ", " << cacheTOF[itof] << "]:      indices   = " << indices[0] << ", " << indices[1] << ", " << indices[2] << ", " << indices[3] << ", " << indices[4];
          LOG(DEBUG) << "Propagated Track [" << itrk << ", " << cacheTrk[itrk] << "]: detId[" << iPropagation << "]  = " << detId[ibc][iPropagation][0] << ", " << detId[ibc][iPropagation][1] << ", " << detId[ibc][iPropagation][2] << ", " << detId[ibc][iPropagation][3] << ", " << detId[ibc][iPropagation][4];
          float resX = deltaPos[ibc][iPropagation][0] - (indices[4] - detId[ibc][iPropagation][4]) * Geo::XPAD + posCorr[0]; // readjusting the residuals due to the fact that the propagation fell in a pad that was not exactly the one of the cluster
          float resZ = deltaPos[ibc][iPropagation][2] - (indices[3] - detId[ibc][iPropagation][3]) * Geo::ZPAD + posCorr[2]; // readjusting the residuals due to the fact that the propagation fell in a pad that was not exactly the one of the cluster
          float res = TMath::Sqrt(resX * resX + resZ * resZ);
          if (indices[0] != detId[ibc][iPropagation][0]) {
            continue;
          }
          if (indices[1] != detId[ibc][iPropagation][1]) {
            continue;
          }
          if (indices[2] != detId[ibc][iPropagation][2]) {
            continue;
          }
          LOG(DEBUG) << "resX = " << resX << ", resZ = " << resZ << ", res = " << res;
          float chi2 = res; // TODO: take into account also the time!

          if (res < mSpaceTolerance) { // matching ok!
            LOG(DEBUG) << "MATCHING FOUND: We have a match! between track " << mTracksSectIndexCache[indices[0]][itrk] << " and TOF cluster " << mTOFClusSectIndexCache[indices[0]][itof];
            foundCluster = true;
            // set event indexes (to be checked)
            evIdx eventIndexTOFCluster(trefTOF.getEntryInTree(), mTOFClusSectIndexCache[indices[0]][itof]);
            evGIdx eventIndexTracks(mCurrTracksTreeEntry, {uint32_t(mTracksSectIndexCache[indices[0]][itrk]), o2::dataformats::GlobalTrackID::TPC});
            mMatchedTracksPairs.emplace_back(eventIndexTOFCluster, chi2, trkLTInt[ibc][iPropagation], eventIndexTracks); // TODO: check if this is correct!
          }
        }
      }
      if (!foundCluster && mMCTruthON) {
        const auto& labelTPC = mTPCLabels[mTracksSectIndexCache[sec][itrk]];
        LOG(DEBUG) << "We did not find any TOF cluster for track " << cacheTrk[itrk] << " (label = " << labelTPC << ", pt = " << trefTrk.getPt();
      }
    }
  }
  return;
}
//______________________________________________
int MatchTOF::findFITIndex(int bc)
{
  if (mFITRecPoints.size() == 0) {
    return -1;
  }

  int index = -1;
  int distMax = 5; // require bc distance below 5

  for (int i = 0; i < mFITRecPoints.size(); i++) {
    const o2::InteractionRecord ir = mFITRecPoints[i].getInteractionRecord();
    int bct0 = ir.orbit * o2::constants::lhc::LHCMaxBunches + ir.bc;
    int dist = bc - bct0;

    if (dist < 0 || dist > distMax) {
      continue;
    }

    distMax = dist;
    index = i;
  }

  return index;
}
//______________________________________________
void MatchTOF::selectBestMatches()
{
  ///< define the track-TOFcluster pair per sector

  LOG(INFO) << "Number of pair matched = " << mMatchedTracksPairs.size();

  // first, we sort according to the chi2
  std::sort(mMatchedTracksPairs.begin(), mMatchedTracksPairs.end(), [this](o2::dataformats::MatchInfoTOF& a, o2::dataformats::MatchInfoTOF& b) { return (a.getChi2() < b.getChi2()); });
  int i = 0;
  // then we take discard the pairs if their track or cluster was already matched (since they are ordered in chi2, we will take the best matching)
  for (const o2::dataformats::MatchInfoTOF& matchingPair : mMatchedTracksPairs) {
    if (mMatchedTracksIndex[matchingPair.getTrackIndex()] != -1) { // the track was already filled
      continue;
    }
    if (mMatchedClustersIndex[matchingPair.getTOFClIndex()] != -1) { // the track was already filled
      continue;
    }
    mMatchedTracksIndex[matchingPair.getTrackIndex()] = mMatchedTracks.size();                               // index of the MatchInfoTOF correspoding to this track
    mMatchedClustersIndex[matchingPair.getTOFClIndex()] = mMatchedTracksIndex[matchingPair.getTrackIndex()]; // index of the track that was matched to this cluster
    mMatchedTracks.push_back(matchingPair);                                                                  // array of MatchInfoTOF

    // get fit info
    double t0info = 0;

    if (mFITRecPoints.size() > 0) {
      int index = findFITIndex(mTOFClusWork[matchingPair.getTOFClIndex()].getBC());

      if (index > -1) {
        o2::InteractionRecord ir = mFITRecPoints[index].getInteractionRecord();
        t0info = ir.bc2ns() * 1E3;
      }
    }

    // add also calibration infos
    mCalibInfoTOF.emplace_back(mTOFClusWork[matchingPair.getTOFClIndex()].getMainContributingChannel(),
                               int(mTOFClusWork[matchingPair.getTOFClIndex()].getTimeRaw() * 1E12), // add time stamp
                               mTOFClusWork[matchingPair.getTOFClIndex()].getTimeRaw() - mLTinfos[matchingPair.getTrackIndex()].getTOF(o2::track::PID::Pion) - t0info,
                               mTOFClusWork[matchingPair.getTOFClIndex()].getTot());
    if (mMCTruthON) {
      const auto& labelsTOF = mTOFClusLabels.getLabels(matchingPair.getTOFClIndex());
      const auto& labelTPC = mTPCLabels[matchingPair.getTrackIndex()];
      // we want to store positive labels independently of how they are flagged from TPC,ITS people
      //    o2::MCCompLabel labelTPC(abs(labelTPCor.getTrackID()), labelTPCor.getEventID(), labelTPCor.getSourceID());
      //    o2::MCCompLabel labelITS(abs(labelITSor.getTrackID()), labelITSor.getEventID(), labelITSor.getSourceID());
      LOG(DEBUG) << "TPC label" << labelTPC;
      bool labelOk = false; // whether we have found or not the same TPC label of the track among the labels of the TOF cluster

      for (int ilabel = 0; ilabel < labelsTOF.size(); ilabel++) {
        LOG(DEBUG) << "TOF label " << ilabel << labelsTOF[ilabel];
        if (labelsTOF[ilabel].getTrackID() == labelTPC.getTrackID() && labelsTOF[ilabel].getEventID() == labelTPC.getEventID() && labelsTOF[ilabel].getSourceID() == labelTPC.getSourceID() && !labelOk) { // if we find one TOF cluster label that is the same as the TPC one, we are happy - even if it is not the first one
          mOutTOFLabels.push_back(labelsTOF[ilabel]);
          labelOk = true;
        }
      }
      if (!labelOk) {
        // we have not found the track label among those associated to the TOF cluster --> fake match! We will associate the label of the main channel, but negative
        //assert(labelsTOF.size());
        if (!labelsTOF.size()) {
          throw std::runtime_error("TOF label not found since size of label is zero. This should not happen!!!!");
        }
        mOutTOFLabels.emplace_back(labelsTOF[0].getTrackID(), labelsTOF[0].getEventID(), labelsTOF[0].getSourceID(), true);
      }
      mOutTPCLabels.push_back(labelTPC);
      if (mIsITSused) {
        const auto& labelITS = mITSLabels[matchingPair.getTrackIndex()];
        LOG(DEBUG) << "ITS label" << labelITS;
        mOutITSLabels.push_back(labelITS);
      }
    }
    i++;
  }
}
//______________________________________________
bool MatchTOF::propagateToRefX(o2::track::TrackParCov& trc, float xRef, float stepInCm, o2::track::TrackLTIntegral& intLT)
{
  // propagate track to matching reference X
  o2::base::Propagator::MatCorrType matCorr = o2::base::Propagator::MatCorrType::USEMatCorrTGeo; // material correction method
  const float tanHalfSector = tan(o2::constants::math::SectorSpanRad / 2);
  bool refReached = false;
  float xStart = trc.getX();
  // the first propagation will be from 2m, if the track is not at least at 2m
  if (xStart < 50.) {
    xStart = 50.;
  }
  int istep = 1;
  bool hasPropagated = o2::base::Propagator::Instance()->PropagateToXBxByBz(trc, xStart + istep * stepInCm, MAXSNP, stepInCm, matCorr, &intLT);
  while (hasPropagated) {
    if (trc.getX() > xRef) {
      refReached = true; // we reached the 371cm reference
    }
    istep++;
    if (fabs(trc.getY()) > trc.getX() * tanHalfSector) { // we are still in the same sector
      // we need to rotate the track to go to the new sector
      //Printf("propagateToRefX: changing sector");
      auto alphaNew = o2::math_utils::angle2Alpha(trc.getPhiPos());
      if (!trc.rotate(alphaNew) != 0) {
        //	Printf("propagateToRefX: failed to rotate");
        break; // failed (this line is taken from MatchTPCITS and the following comment too: RS: check effect on matching tracks to neighbouring sector)
      }
    }
    if (refReached) {
      break;
    }
    hasPropagated = o2::base::Propagator::Instance()->PropagateToXBxByBz(trc, xStart + istep * stepInCm, MAXSNP, stepInCm, matCorr, &intLT);
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
  if (xStart < 50.) {
    xStart = 50.;
  }
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
      auto alphaNew = o2::math_utils::angle2Alpha(trcNoCov.getPhiPos());
      if (!trcNoCov.rotateParam(alphaNew) != 0) {
        //	Printf("propagateToRefX: failed to rotate");
        break; // failed (this line is taken from MatchTPCITS and the following comment too: RS: check effect on matching tracks to neighbouring sector)
      }
    }
    if (refReached) {
      break;
    }
    hasPropagated = trcNoCov.propagateParamTo(xStart + istep * stepInCm, bzField);
  }
  //  if (std::abs(trc.getSnp()) > MAXSNP) Printf("propagateToRefX: condition on snp not ok, returning false");
  //Printf("propagateToRefX: snp of teh track is %f (--> %f grad)", trcNoCov.getSnp(), TMath::ASin(trcNoCov.getSnp())*TMath::RadToDeg());

  return refReached && std::abs(trcNoCov.getSnp()) < 0.95 && TMath::Abs(trcNoCov.getZ()) < Geo::MAXHZTOF; // Here we need to put MAXSNP
}

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
void MatchTOF::fillTOFmatchTree(const char* trname, int cacheTOF, int sectTOF, int plateTOF, int stripTOF, int padXTOF, int padZTOF, int cacheeTrk, int crossedStrip, int sectPropagation, int platePropagation, int stripPropagation, int padXPropagation, int padZPropagation, float resX, float resZ, float res, matchTrack& trk, float intLength, float intTimePion, float timeTOF)
{
  ///< fill debug tree for TOF tracks matching check

  mTimerDBG.Start(false);

  //  Printf("************** Filling the debug tree with %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %f, %f, %f", cacheTOF, sectTOF, plateTOF, stripTOF, padXTOF, padZTOF, cacheeTrk, crossedStrip, sectPropagation, platePropagation, stripPropagation, padXPropagation, padZPropagation, resX, resZ, res);

  if (mDBGFlags) {
    (*mDBGOut) << trname
               << "clusterTOF=" << cacheTOF << "sectTOF=" << sectTOF << "plateTOF=" << plateTOF << "stripTOF=" << stripTOF << "padXTOF=" << padXTOF << "padZTOF=" << padZTOF
               << "crossedStrip=" << crossedStrip << "sectPropagation=" << sectPropagation << "platePropagation=" << platePropagation << "stripPropagation=" << stripPropagation << "padXPropagation=" << padXPropagation
               << "resX=" << resX << "resZ=" << resZ << "res=" << res << "track=" << trk.first << "intLength=" << intLength << "intTimePion=" << intTimePion << "timeTOF=" << timeTOF << "\n";
  }
  mTimerDBG.Stop();
}

//_________________________________________________________
void MatchTOF::fillTOFmatchTreeWithLabels(const char* trname, int cacheTOF, int sectTOF, int plateTOF, int stripTOF, int padXTOF, int padZTOF, int cacheeTrk, int crossedStrip, int sectPropagation, int platePropagation, int stripPropagation, int padXPropagation, int padZPropagation, float resX, float resZ, float res, matchTrack& trk, int TPClabelTrackID, int TPClabelEventID, int TPClabelSourceID, int ITSlabelTrackID, int ITSlabelEventID, int ITSlabelSourceID, int TOFlabelTrackID0, int TOFlabelEventID0, int TOFlabelSourceID0, int TOFlabelTrackID1, int TOFlabelEventID1, int TOFlabelSourceID1, int TOFlabelTrackID2, int TOFlabelEventID2, int TOFlabelSourceID2, float intLength, float intTimePion, float timeTOF)
{
  ///< fill debug tree for TOF tracks matching check

  mTimerDBG.Start(false);

  if (mDBGFlags) {
    (*mDBGOut) << trname
               << "clusterTOF=" << cacheTOF << "sectTOF=" << sectTOF << "plateTOF=" << plateTOF << "stripTOF=" << stripTOF << "padXTOF=" << padXTOF << "padZTOF=" << padZTOF
               << "crossedStrip=" << crossedStrip << "sectPropagation=" << sectPropagation << "platePropagation=" << platePropagation << "stripPropagation=" << stripPropagation << "padXPropagation=" << padXPropagation
               << "resX=" << resX << "resZ=" << resZ << "res=" << res << "track=" << trk.first
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
