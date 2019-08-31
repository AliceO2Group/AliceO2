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
#include <TSystem.h>
#include <cassert>

#include "FairLogger.h"
#include "Field/MagneticField.h"
#include "Field/MagFieldFast.h"
#include "ITSBase/GeometryTGeo.h"

#include "CommonUtils/TreeStream.h"

#include "DataFormatsTPC/Defs.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterGas.h"
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
#include "SimulationDataFormat/MCTruthContainer.h"
#include "GlobalTracking/MatchTPCITS.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"

#include "GPUO2Interface.h" // Needed for propper settings in GPUParam.h
#define GPUCA_O2_LIB        // Temporary workaround, must not be set globally, but needed right now for GPUParam.h
#include "GPUParam.h"       // Consider more universal access
#undef GPUCA_O2_LIB

using namespace o2::globaltracking;

using MatrixDSym4 = ROOT::Math::SMatrix<double, 4, 4, ROOT::Math::MatRepSym<double, 4>>;
using MatrixD4 = ROOT::Math::SMatrix<double, 4, 4, ROOT::Math::MatRepStd<double, 4>>;

constexpr float MatchTPCITS::XTPCInnerRef;
constexpr float MatchTPCITS::XTPCOuterRef;
constexpr float MatchTPCITS::XMatchingRef;
constexpr float MatchTPCITS::YMaxAtXMatchingRef;
constexpr float MatchTPCITS::Tan70, MatchTPCITS::Cos70I2, MatchTPCITS::MaxSnp, MatchTPCITS::MaxTgp;

//______________________________________________
void MatchTPCITS::printABTracksTree(const ABTrackLinksList& llist) const
{
  // dump all hypotheses
  int startLayer = 0, nextHyp;
  while ((nextHyp = llist.firstInLr[startLayer]) == MinusOne && startLayer < NITSLayers) {
    startLayer++;
  }
  if (startLayer == NITSLayers) {
    printf("No matches\n");
    return;
  }
  auto lblTrc = mTPCTrkLabels->getLabels(mTPCWork[llist.trackID].source.getIndex())[0]; // tmp
  LOG(INFO) << "Matches for track " << llist.trackID << " " << lblTrc;
  while (nextHyp > MinusOne) {
    printf("Lr/IC/ClID/Chi2/{MC}:%c ", lblTrc.getTrackID() > 0 ? 'C' : 'F');
    int parID = nextHyp; // print particular hypothesis
    int lr = startLayer;
    while (1) {
      const auto& lnk = mABTrackLinks[parID];
      int mcEv = -1, mcTr = -1;
      if (lnk.clID > MinusOne) {
        const auto lab = mITSClsLabels->getLabels(lnk.clID)[0];
        if (lab.isValid()) {
          mcEv = lab.getEventID();
          mcTr = lab.getTrackID();
        } else {
          mcEv = mcTr = -999; // noise
        }
      } else if (lnk.isDummyTop()) { // top layer, use TPC MC lbl
        mcEv = lblTrc.getEventID();
        mcTr = lblTrc.getTrackID();
      }
      printf("[%d/%3d/%5d/%6.2f/{%d/%d}]", lr++, lnk.icCandID, lnk.clID, lnk.chi2, mcEv, mcTr);
      if (lnk.isDummyTop()) { // reached dummy seed on the dummy layer above the last ITS layer
        break;
      }
      parID = lnk.parentID;
    }
    printf("\n");
    nextHyp = mABTrackLinks[nextHyp].nextOnLr;
  }
}

//______________________________________________
void MatchTPCITS::dumpABTracksDebugTree(const ABTrackLinksList& llist)
{
  // dump all hypotheses
  int startLayer = 0, nextHyp;
  while ((nextHyp = llist.firstInLr[startLayer]) == MinusOne && startLayer < NITSLayers) {
    startLayer++;
  }
  if (startLayer == NITSLayers) {
    return;
  }
  LOG(INFO) << "Dump AB Matches for track " << llist.trackID;
  auto lblTrc = mTPCTrkLabels->getLabels(mTPCWork[llist.trackID].source.getIndex())[0]; // tmp
  while (nextHyp > MinusOne) {
    // fill debug AB track
    ABDebugTrack dbgTrack;
    int parID = nextHyp; // print particular hypothesis
    int lr = startLayer;
    while (1) {
      const auto& lnk = mABTrackLinks[parID];
      if (lnk.clID>MinusOne) {
	auto& dbgLnk = dbgTrack.links.emplace_back();
#ifdef _ALLOW_DEBUG_AB_
	dbgLnk.seed = lnk.seed; // seed before update
#endif
	dbgLnk.clLabel = mITSClsLabels->getLabels(lnk.clID)[0];
	dbgLnk.chi2 = lnk.chi2;
	dbgLnk.lr = lr;
	(o2::BaseCluster<float>&)dbgLnk = (*mITSClustersArrayInp)[lnk.clID];
	dbgTrack.nClus++;
	if (lblTrc.getEventID()==dbgLnk.clLabel.getEventID() && std::abs(lblTrc.getTrackID())==dbgLnk.clLabel.getTrackID()) {
	  dbgTrack.nClusCorr++;
	}
      }
      if (lnk.isDummyTop()) { // reached dummy seed on the dummy layer above the last ITS layer
	dbgTrack.tpcSeed = lnk; // tpc seed
	dbgTrack.trackID = llist.trackID;
	dbgTrack.tpcLabel = lblTrc;
	dbgTrack.icCand = lnk.icCandID;
	dbgTrack.icTimeBin = mInteractions[lnk.icCandID].timeBins;
        break;
      }
      lr++;
      parID = lnk.parentID;
    }
    dbgTrack.chi2 = dbgTrack.links.front().chi2;
    // at the moment links contain cumulative chi2, convert to track-to-cluster chi2
    for (int i=0;i<dbgTrack.nClus-1;i++) {
      dbgTrack.links[i].chi2 -= dbgTrack.links[i+1].chi2;
    }
    // dump debug track
    (*mDBGOut) << "abtree" << "trc=" << dbgTrack << "\n";
      
    nextHyp = mABTrackLinks[nextHyp].nextOnLr;
  }
}

//______________________________________________
void MatchTPCITS::printABClusterUsage() const
{
  // print links info of clusters involved in AB tracks
  int ncl = mABClusterLinkIndex.size();
  for (int icl = 0; icl < ncl; icl++) {
    int lnkIdx = mABClusterLinkIndex[icl];
    if (lnkIdx <= MinusOne) { // not used or used in standard ITS tracks
      continue;
    }
    LOG(INFO) << "Links for cluster " << icl;
    int count = 0;
    while (lnkIdx > MinusOne) {
      const auto& linkCl = mABClusterLinks[lnkIdx];
      const auto& linkTrack = mABTrackLinks[linkCl.linkedABTrack];
      // find top track link on the dummy layer
      int topIdx = linkCl.linkedABTrack, nUp = 0;
      while (1) {
        if (mABTrackLinks[topIdx].isDummyTop()) {
          break;
        }
        nUp++;
        topIdx = mABTrackLinks[topIdx].parentID;
      }
      const auto& topTrack = mABTrackLinks[topIdx];
      printf("[#%d Tr:%d IC:%d Chi2:%.2f NP:%d]", count++, topTrack.parentID, linkTrack.icCandID, linkTrack.chi2, nUp);
      lnkIdx = linkCl.nextABClusterLink;
    }
    printf("\n");
  }
}

//______________________________________________
MatchTPCITS::MatchTPCITS() = default;

//______________________________________________
MatchTPCITS::~MatchTPCITS() = default;

//______________________________________________
void MatchTPCITS::setDPLIO(bool v)
{
  ///< set unput type
  assert(!mInitDone); // must be set before the init is called
  mDPLIO = v;
}

//______________________________________________
void MatchTPCITS::assertDPLIO(bool v)
{
  ///< make sure that the IO mode corresponds to requested
  if (mDPLIO != v) {
    LOG(FATAL) << "Requested operation is not allowed in " << (mDPLIO ? "DPL" : "Tree") << " mode";
  }
}

//______________________________________________
void MatchTPCITS::run()
{
  ///< perform matching for provided input
  if (!mInitDone) {
    LOG(FATAL) << "init() was not done yet";
  }
  ProcInfo_t procInfoStart, procInfoStop;
  gSystem->GetProcInfo(&procInfoStart);
  constexpr uint64_t kMB = 1024 * 1024;
  printf("Memory (GB) at entrance: RSS: %.3f VMem: %.3f\n", float(procInfoStart.fMemResident) / kMB, float(procInfoStart.fMemVirtual) / kMB);

  mTimerTot.Start();

  clear();

  if (!prepareITSClusters() || !prepareITSTracks() || !prepareTPCTracks() || !prepareFITInfo()) {
    return;
  }

  for (int sec = o2::constants::math::NSectors; sec--;) {
    doMatching(sec);
  }

  if (0) { // enabling this creates very verbose output
    mTimerTot.Stop();
    printCandidatesTPC();
    printCandidatesITS();
    mTimerTot.Start(false);
  }

  selectBestMatches();

  refitWinners();

  if (isRunAfterBurner()) {
    runAfterBurner();
  }

#ifdef _ALLOW_DEBUG_TREES_
  if (mDBGOut && isDebugFlag(WinnerMatchesTree)) {
    dumpWinnerMatches();
  }
  mDBGOut.reset();
#endif

  gSystem->GetProcInfo(&procInfoStop);
  mTimerTot.Stop();

  printf("Timing:\n");
  printf("Total:        ");
  mTimerTot.Print();
  printf("Data IO:      ");
  mTimerIO.Print();
  printf("Refits      : ");
  mTimerRefit.Print();
  printf("DBG trees:    ");
  mTimerDBG.Print();

  printf("Memory (GB) at exit: RSS: %.3f VMem: %.3f\n", float(procInfoStop.fMemResident) / kMB, float(procInfoStop.fMemVirtual) / kMB);
  printf("Memory increment: RSS: %.3f VMem: %.3f\n",
         float(procInfoStop.fMemResident - procInfoStart.fMemResident) / kMB,
         float(procInfoStop.fMemVirtual - procInfoStart.fMemVirtual) / kMB);
}

//______________________________________________
void MatchTPCITS::clear()
{
  ///< clear results of previous TF reconstruction
  mMatchRecordsTPC.clear();
  mMatchRecordsITS.clear();
  mWinnerChi2Refit.clear();
  mMatchedTracks.clear();
  if (mMCTruthON) {
    mOutITSLabels.clear();
    mOutTPCLabels.clear();
  }
}

//______________________________________________
void MatchTPCITS::init()
{
  ///< perform initizalizations, precalculate what is needed
  if (mInitDone) {
    LOG(ERROR) << "Initialization was already done";
    return;
  }

  // make sure T2GRot matrices are loaded into ITS geometry helper
  o2::its::GeometryTGeo::Instance()->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2GRot));

  mSectEdgeMargin2 = mCrudeAbsDiffCut[o2::track::kY] * mCrudeAbsDiffCut[o2::track::kY]; ///< precalculated ^2

  auto& gasParam = o2::tpc::ParameterGas::Instance();
  auto& elParam = o2::tpc::ParameterElectronics::Instance();
  auto& detParam = o2::tpc::ParameterDetector::Instance();
  mTPCTBinMUS = elParam.ZbinWidth;
  mTPCVDrift0 = gasParam.DriftV;
  mTPCZMax = detParam.TPClength;

  assert(mITSROFrameLengthMUS > 0.0f);
  mITSROFramePhaseOffset = mITSROFrameOffsetMUS / mITSROFrameLengthMUS;
  mITSROFrame2TPCBin = mITSROFrameLengthMUS / mTPCTBinMUS;
  mTPCBin2ITSROFrame = 1. / mITSROFrame2TPCBin;
  mTPCBin2Z = mTPCTBinMUS * mTPCVDrift0;
  mZ2TPCBin = 1. / mTPCBin2Z;
  mTPCVDrift0Inv = 1. / mTPCVDrift0;
  mNTPCBinsFullDrift = mTPCZMax * mZ2TPCBin;

  mTPCTimeEdgeTSafeMargin = z2TPCBin(mTPCTimeEdgeZSafeMargin);

  std::unique_ptr<TPCTransform> fastTransform = (o2::tpc::TPCFastTransformHelperO2::instance()->create(0));
  mTPCTransform = std::move(fastTransform);
  mTPCClusterParam = std::make_unique<o2::gpu::GPUParam>();
  mTPCClusterParam->SetDefaults(o2::base::Propagator::Instance()->getNominalBz()); // TODO this may change
  mFieldON = std::abs(o2::base::Propagator::Instance()->getNominalBz()) > 0.01;

  if (!mDPLIO) {
    attachInputTrees();

    // create output branch
    if (mOutputTree) {
      LOG(INFO) << "ITS-TPC Matching results will be stored in the tree " << mOutputTree->GetName();
      mOutputTree->Branch(mOutTPCITSTracksBranchName.data(), &mMatchedTracks);
      LOG(INFO) << "Matched tracks branch: " << mOutTPCITSTracksBranchName;
      if (mMCTruthON) {
        mOutputTree->Branch(mOutTPCMCTruthBranchName.data(), &mOutITSLabels);
        LOG(INFO) << "ITS Tracks Labels branch: " << mOutITSMCTruthBranchName;
        mOutputTree->Branch(mOutITSMCTruthBranchName.data(), &mOutTPCLabels);
        LOG(INFO) << "TPC Tracks Labels branch: " << mOutTPCMCTruthBranchName;
      }
    } else {
      LOG(ERROR) << "Output tree is not attached, matched tracks will not be stored";
    }
  }
#ifdef _ALLOW_DEBUG_TREES_
  // debug streamer
  if (mDBGFlags) {
    mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(mDebugTreeFileName.data(), "recreate");
  }
#endif

  mRGHelper.init(); // prepare helper for TPC track / ITS clusters matching
  const auto& zr = mRGHelper.layers.back().zRange;
  mITSFiducialZCut = std::max(std::abs(zr.min()), std::abs(zr.max())) + 20.;

  clear();

  mInitDone = true;

  {
    mTimerTot.Stop();
    mTimerIO.Stop();
    mTimerDBG.Stop();
    mTimerRefit.Stop();
    mTimerTot.Reset();
    mTimerIO.Reset();
    mTimerDBG.Reset();
    mTimerRefit.Reset();
  }

  print();
}

//______________________________________________
void MatchTPCITS::selectBestMatches()
{
  ///< loop over match records and select the ones with best chi2
  LOG(INFO) << "Selecting best matches";
  int nValidated = 0, iter = 0;

  do {
    nValidated = 0;
    int ntpc = mTPCWork.size(), nremaining = 0;
    ;
    for (int it = 0; it < ntpc; it++) {
      auto& tTPC = mTPCWork[it];
      if (isDisabledTPC(tTPC) || isValidatedTPC(tTPC)) {
        continue;
      }
      nremaining++;
      if (validateTPCMatch(it)) {
        nValidated++;
        continue;
      }
    }
    printf("iter %d Validated %d of %d remaining matches\n", iter, nValidated, nremaining);
    iter++;
  } while (nValidated);
}

//______________________________________________
bool MatchTPCITS::validateTPCMatch(int iTPC)
{
  const auto& tTPC = mTPCWork[iTPC];
  auto& rcTPC = mMatchRecordsTPC[tTPC.matchID]; // best TPC->ITS match
  if (rcTPC.nextRecID == Validated) {
    return false; // RS do we need this
  }
  // check if it is consistent with corresponding ITS->TPC match
  auto& tITS = mITSWork[rcTPC.partnerID];       //  partner ITS track
  auto& rcITS = mMatchRecordsITS[tITS.matchID]; // best ITS->TPC match record
  if (rcITS.nextRecID == Validated) {
    return false;              // RS do we need this ?
  }
  if (rcITS.partnerID == iTPC) { // is best matching TPC track for this ITS track actually iTPC?
    // unlink winner TPC track from all ITS candidates except winning one
    int nextTPC = rcTPC.nextRecID;
    while (nextTPC > MinusOne) {
      auto& rcTPCrem = mMatchRecordsTPC[nextTPC];
      removeTPCfromITS(iTPC, rcTPCrem.partnerID); // remove references on mtID from ITS match=rcTPCrem.partnerID
      nextTPC = rcTPCrem.nextRecID;
    }
    rcTPC.nextRecID = Validated;
    int itsWinID = rcTPC.partnerID;

    // unlink winner ITS match from all TPC matches using it
    int nextITS = rcITS.nextRecID;
    while (nextITS > MinusOne) {
      auto& rcITSrem = mMatchRecordsITS[nextITS];
      removeITSfromTPC(itsWinID, rcITSrem.partnerID); // remove references on itsWinID from TPC match=rcITSrem.partnerID
      nextITS = rcITSrem.nextRecID;
    }
    rcITS.nextRecID = Validated;
    return true;
  }
  return false;
}

//______________________________________________
int MatchTPCITS::getNMatchRecordsTPC(const TrackLocTPC& tTPC) const
{
  ///< get number of matching records for TPC track
  int count = 0, recID = tTPC.matchID;
  while (recID > MinusOne) {
    recID = mMatchRecordsTPC[recID].nextRecID;
    count++;
  }
  return count;
}

//______________________________________________
int MatchTPCITS::getNMatchRecordsITS(const TrackLocITS& tTPC) const
{
  ///< get number of matching records for ITS track
  int count = 0, recID = tTPC.matchID;
  while (recID > MinusOne) {
    auto& itsRecord = mMatchRecordsITS[recID];
    recID = itsRecord.nextRecID;
    count++;
  }
  return count;
}

//______________________________________________
void MatchTPCITS::attachInputTrees()
{
  if (!mTreeITSTracks) {
    LOG(FATAL) << "ITS tracks data input tree is not set";
  }

  if (!mTreeITSTrackROFRec) {
    LOG(FATAL) << "ITS ROFRec data input tree is not set";
  }

  if (!mTreeTPCTracks) {
    LOG(FATAL) << "TPC tracks data input tree is not set";
  }

  if (!mTreeITSClusters) {
    LOG(FATAL) << "ITS clusters data input tree is not set";
  }

  if (!mTreeITSTracks->GetBranch(mITSTrackBranchName.data())) {
    LOG(FATAL) << "Did not find ITS tracks branch " << mITSTrackBranchName << " in the input tree";
  }
  mTreeITSTracks->SetBranchAddress(mITSTrackBranchName.data(), &mITSTracksArrayInp);
  LOG(INFO) << "Attached ITS tracks " << mITSTrackBranchName << " branch with " << mTreeITSTracks->GetEntries()
            << " entries";

  if (!mTreeITSTracks->GetBranch(mITSTrackClusIdxBranchName.data())) {
    LOG(FATAL) << "Did not find ITS track cluster indices branch " << mITSTrackClusIdxBranchName << " in the input tree";
  }
  mTreeITSTracks->SetBranchAddress(mITSTrackClusIdxBranchName.data(), &mITSTrackClusIdxInp);
  LOG(INFO) << "Attached ITS track cluster indices " << mITSTrackClusIdxBranchName << " branch with "
            << mTreeITSTracks->GetEntries() << " entries";

  if (!mTreeITSTrackROFRec->GetBranch(mITSTrackROFRecBranchName.data())) {
    LOG(FATAL) << "Did not find ITS tracks ROFRecords branch " << mITSTrackROFRecBranchName << " in the input tree";
  }
  mTreeITSTrackROFRec->SetBranchAddress(mITSTrackROFRecBranchName.data(), &mITSTrackROFRec);
  LOG(INFO) << "Attached ITS tracks ROFRec " << mITSTrackROFRecBranchName << " branch with "
            << mTreeITSTrackROFRec->GetEntries() << " entries";

  if (!mTreeTPCTracks->GetBranch(mTPCTrackBranchName.data())) {
    LOG(FATAL) << "Did not find TPC tracks branch " << mTPCTrackBranchName << " in the input tree";
  }
  mTreeTPCTracks->SetBranchAddress(mTPCTrackBranchName.data(), &mTPCTracksArrayInp);
  LOG(INFO) << "Attached TPC tracks " << mTPCTrackBranchName << " branch with " << mTreeTPCTracks->GetEntries()
            << " entries";

  if (!mTreeITSClusters->GetBranch(mITSClusterBranchName.data())) {
    LOG(FATAL) << "Did not find ITS clusters branch " << mITSClusterBranchName << " in the input tree";
  }
  LOG(INFO) << "Will use ITS clusters " << mITSClusterBranchName << " branch with " << mTreeITSClusters->GetEntries()
            << " entries";

  if (!mTreeITSClusterROFRec->GetBranch(mITSClusterROFRecBranchName.data())) {
    LOG(FATAL) << "Did not find ITS clusters ROFRecords branch " << mITSClusterROFRecBranchName << " in the input tree";
  }
  LOG(INFO) << "Will use ITS clusters ROFRec " << mITSClusterROFRecBranchName << " branch with "
            << mTreeITSClusterROFRec->GetEntries() << " entries";

  if (!mTPCClusterReader) {
    LOG(FATAL) << "TPC clusters reader is not set";
  }
  LOG(INFO) << "Attached TPC clusters reader with " << mTPCClusterReader->getTreeSize();
  mTPCClusterIdxStructOwn = std::make_unique<o2::tpc::ClusterNativeAccess>();

  // is there FIT Info available?
  if (mTreeFITInfo) {
    mTreeFITInfo->SetBranchAddress(mFITInfoBranchName.data(), &mFITInfoInp);
    LOG(INFO) << "Attached FIT info " << mFITInfoBranchName << " branch with " << mTreeFITInfo->GetEntries() << " entries";
  } else {
    LOG(INFO) << "FIT info is not available";
  }

  // is there MC info available ?
  if (mMCTruthON && mTreeITSTracks->GetBranch(mITSMCTruthBranchName.data())) {
    mTreeITSTracks->SetBranchAddress(mITSMCTruthBranchName.data(), &mITSTrkLabels);
    LOG(INFO) << "Found ITS Track MCLabels branch " << mITSMCTruthBranchName;
  }
  // is there MC info available ?
  if (mMCTruthON && mTreeTPCTracks->GetBranch(mTPCMCTruthBranchName.data())) {
    mTreeTPCTracks->SetBranchAddress(mTPCMCTruthBranchName.data(), &mTPCTrkLabels);
    LOG(INFO) << "Found TPC Track MCLabels branch " << mTPCMCTruthBranchName;
  }

  mMCTruthON &= (mITSTrkLabels && mTPCTrkLabels && mTreeITSClusters->GetBranch(mITSClusMCTruthBranchName.data()));
}

//______________________________________________
bool MatchTPCITS::prepareTPCTracks()
{
  ///< load next chunk of TPC data and prepare for matching
  mMatchRecordsTPC.clear();
  int curTracksEntry = 0;

  if (!mDPLIO) { // in the DPL IO mode the input TPC tracks must be already attached
    if (!loadTPCTracksNextChunk()) {
      return false;
    }
    curTracksEntry = mTreeTPCTracks ? mTreeTPCTracks->GetReadEntry() : 0;
  }

  int ntr = mTPCTracksArrayInp->size();
  // number of records might be actually more than N tracks!
  mMatchRecordsTPC.reserve(mMatchRecordsTPC.size() + mMaxMatchCandidates * ntr);

  // copy the track params, propagate to reference X and build sector tables
  mTPCWork.clear();
  mTPCWork.reserve(ntr);
  if (mMCTruthON) {
    mTPCLblWork.clear();
    mTPCLblWork.reserve(ntr);
  }
  for (int sec = o2::constants::math::NSectors; sec--;) {
    mTPCSectIndexCache[sec].clear();
    mTPCSectIndexCache[sec].reserve(100 + 1.2 * ntr / o2::constants::math::NSectors);
    mTPCTimeBinStart[sec].clear();
  }

  for (int it = 0; it < ntr; it++) {

    const auto& trcOrig = (*mTPCTracksArrayInp)[it];

    // make sure the track was propagated to inner TPC radius at the ref. radius
    if (trcOrig.getX() > XTPCInnerRef + 0.1)
      continue; // failed propagation to inner TPC radius, cannot be matched

    // create working copy of track param
    mTPCWork.emplace_back(static_cast<const o2::track::TrackParCov&>(trcOrig), curTracksEntry, it);
    auto& trc = mTPCWork.back();
    // propagate to matching Xref
    if (!propagateToRefX(trc)) {
      mTPCWork.pop_back(); // discard track whose propagation to XMatchingRef failed
      continue;
    }
    if (mMCTruthON) {
      mTPCLblWork.emplace_back(mTPCTrkLabels->getLabels(it)[0]);
    }

    float time0 = trcOrig.getTime0() - mNTPCBinsFullDrift;
    trc.timeBins.set(time0 - trcOrig.getDeltaTBwd() - mTPCTimeEdgeTSafeMargin,
                     time0 + trcOrig.getDeltaTFwd() + mTPCTimeEdgeTSafeMargin);
    // assign min max possible Z for this track which still respects the clusters A/C side
    if (trcOrig.hasASideClustersOnly()) {
      trc.zMin = trc.getZ() - trcOrig.getDeltaTBwd() * mTPCBin2Z;
      trc.zMax = trc.getZ() + trcOrig.getDeltaTFwd() * mTPCBin2Z;
    } else if (trcOrig.hasCSideClustersOnly()) {
      trc.zMin = trc.getZ() - trcOrig.getDeltaTFwd() * mTPCBin2Z;
      trc.zMax = trc.getZ() + trcOrig.getDeltaTBwd() * mTPCBin2Z;
    }
    // TODO : special treatment of tracks crossing the CE

    // cache work track index
    mTPCSectIndexCache[o2::utils::Angle2Sector(trc.getAlpha())].push_back(mTPCWork.size() - 1);
  }
  float maxTimeBin = 0;
  int nITSROFs = mITSROFTimes.size();
  // sort tracks in each sector according to their timeMax
  for (int sec = o2::constants::math::NSectors; sec--;) {
    auto& indexCache = mTPCSectIndexCache[sec];
    LOG(INFO) << "Sorting sector" << sec << " | " << indexCache.size() << " TPC tracks";
    if (!indexCache.size())
      continue;
    std::sort(indexCache.begin(), indexCache.end(), [this](int a, int b) {
      auto& trcA = mTPCWork[a];
      auto& trcB = mTPCWork[b];
      return (trcA.timeBins.max() - trcB.timeBins.max()) < 0.;
    });

    // build array of 1st entries with tmax corresponding to each ITS ROF (or trigger)
    float tmax = mTPCWork[indexCache.back()].timeBins.max();
    if (maxTimeBin < tmax) {
      maxTimeBin = tmax;
    }
    int nbins = 1 + tpcTimeBin2ITSROFrame(tmax);
    auto& tbinStart = mTPCTimeBinStart[sec];
    tbinStart.resize(nbins, -1);
    int itsROF = 0;
    tbinStart[0] = itsROF;
    for (int itr = 0; itr < (int)indexCache.size(); itr++) {
      auto& trc = mTPCWork[indexCache[itr]];

      while (itsROF < nITSROFs && trc.timeBins < mITSROFTimes[itsROF]) {
        itsROF++;
      }
      if (tbinStart[itsROF] == -1) {
        tbinStart[itsROF] = itr;
      }
    }
    for (int i = 1; i < nbins; i++) {
      if (tbinStart[i] == -1) { // fill gaps with preceding indices
        tbinStart[i] = tbinStart[i - 1];
      }
    }
  } // loop over tracks of single sector

  // create mapping from TPC time-bins to ITS ROFs

  if (mITSROFTimes.back() < maxTimeBin) {
    maxTimeBin = mITSROFTimes.back().max();
  }
  int nb = int(maxTimeBin) + 1;
  mITSROFofTPCBin.resize(nb, -1);
  int itsROF = 0;
  for (int ib = 0; ib < nb; ib++) {
    while (itsROF < nITSROFs && ib < mITSROFTimes[itsROF].min()) {
      itsROF++;
    }
    mITSROFofTPCBin[ib] = itsROF;
  }
  return true;
}

//_____________________________________________________
bool MatchTPCITS::prepareITSClusters()
{
  if (!mDPLIO) { // for input from tree read ROFrecords vector, in DPL IO mode the vector should be already attached
    mTimerIO.Start(false);

    std::vector<o2::itsmft::Cluster> bufferCl, *buffClPtr = &bufferCl;
    MCLabCont bufferMC, *buffMCPtr = &bufferMC;
    auto rofrPtr = &mITSClusterROFRecBuffer;
    mTreeITSClusterROFRec->SetBranchAddress(mITSClusterROFRecBranchName.data(), &rofrPtr);
    mTreeITSClusters->SetBranchAddress(mITSClusterBranchName.data(), &buffClPtr);
    if (mMCTruthON) {
      mTreeITSClusters->SetBranchAddress(mITSClusMCTruthBranchName.data(), &buffMCPtr);
      mITSClsLabels = &mITSClsLabelsBuffer;
    }
    mTreeITSClusterROFRec->GetEntry(0); // keep clusters ROFRecs ready
    int nROFs = mITSClusterROFRecBuffer.size();

    // estimate total number of clusters and reserve the space
    size_t nclTot = 0;
    for (int ir = 0; ir < nROFs; ir++) {
      o2::itsmft::ROFRecord& clROF = mITSClusterROFRecBuffer[ir];
      clROF.getROFEntry().setIndex(nclTot); // linearize
      clROF.getROFEntry().setEvent(0);      // fix entry, though it is not used
      nclTot += clROF.getNROFEntries();
    }
    mITSClusterROFRec = &mITSClusterROFRecBuffer;
    mTreeITSClusterROFRec->SetBranchAddress(mITSClusterROFRecBranchName.data(), nullptr); // detach

    mITSClustersBuffer.reserve(nclTot);
    for (int iev = 0; iev < mTreeITSClusters->GetEntries(); iev++) {
      mTreeITSClusters->GetEntry(iev);
      std::copy(bufferCl.begin(), bufferCl.end(), std::inserter(mITSClustersBuffer, mITSClustersBuffer.end()));
      if (mMCTruthON) {
        mITSClsLabelsBuffer.mergeAtBack(bufferMC);
      }
    }
    LOG(INFO) << "Merged " << mTreeITSClusters->GetEntries() << " ITS cluster entries to single container";
    mITSClustersArrayInp = &mITSClustersBuffer;
    mTreeITSClusters->SetBranchAddress(mITSClusterBranchName.data(), nullptr); // detach input array
    if (mMCTruthON) {
      mTreeITSClusters->SetBranchAddress(mITSClusMCTruthBranchName.data(), nullptr); // detach input container
    }
    mTimerIO.Stop();
  }
  return true;
}

//_____________________________________________________
bool MatchTPCITS::prepareITSTracks()
{
  // In the standalone (tree-based) mode load next chunk of ITS data,
  // In the DPL-driven mode the input data containers are supposed to be already assigned
  // Do preparatory work for matching

  if (!mDPLIO) { // for input from tree read ROFrecords vector, in DPL IO mode the vector should be already attached
    mTreeITSTrackROFRec->GetEntry(0);
    mTreeITSClusterROFRec->GetEntry(0); // keep clusters ROFRecs ready
  }
  int nROFs = mITSTrackROFRec->size();

  if (!nROFs) {
    LOG(INFO) << "Empty TF";
    return false;
  }

  mITSWork.clear();
  mITSROFTimes.clear();
  // number of records might be actually more than N tracks!
  mMatchRecordsITS.clear(); // RS TODO reserve(mMatchRecordsITS.size() + mMaxMatchCandidates*ntr);
  if (mMCTruthON) {
    mITSLblWork.clear();
  }

  // total N ITS clusters in TF
  const auto& lastClROF = mITSClusterROFRec->back();
  int nITSClus = lastClROF.getROFEntry().getIndex() + lastClROF.getNROFEntries();
  mABClusterLinkIndex.clear();
  mABClusterLinkIndex.resize(nITSClus, MinusOne);

  for (int sec = o2::constants::math::NSectors; sec--;) {
    mITSSectIndexCache[sec].clear();
    mITSTimeBinStart[sec].clear();
    mITSTimeBinStart[sec].resize(nROFs, -1); // start of ITS work tracks in every sector
  }
  setStartIR((*mITSTrackROFRec)[0].getBCData());
  for (int irof = 0; irof < nROFs; irof++) {
    const auto& rofRec = (*mITSTrackROFRec)[irof];
    int cluROFOffset = (*mITSClusterROFRec)[irof].getROFEntry().getIndex(); // clusters of this ROF start at this offset
    // in case of the input from the tree make sure needed entry is loaded
    auto rEntry = rofRec.getROFEntry().getEvent();
    if (!mDPLIO && mCurrITSTracksTreeEntry != rEntry) { // in DPL IO mode the input tracks must be already attached
      mTreeITSTracks->GetEntry((mCurrITSTracksTreeEntry = rEntry));
    }
    float tmn = intRecord2TPCTimeBin(rofRec.getBCData());     // ITS track min time in TPC time-bins
    mITSROFTimes.emplace_back(tmn, tmn + mITSROFrame2TPCBin); // ITS track min/max time in TPC time-bins

    for (int sec = o2::constants::math::NSectors; sec--;) {         // start of sector's tracks for this ROF
      mITSTimeBinStart[sec][irof] = mITSSectIndexCache[sec].size(); // The sorting does not affect this
    }

    int trlim = rofRec.getROFEntry().getIndex() + rofRec.getNROFEntries();
    for (int it = rofRec.getROFEntry().getIndex(); it < trlim; it++) {
      auto& trcOrig = (*mITSTracksArrayInp)[it];

      if (isRunAfterBurner()) {
        flagUsedITSClusters(trcOrig, cluROFOffset);
      }

      if (trcOrig.getParamOut().getX() < 1.) {
        continue; // backward refit failed
      }
      int nWorkTracks = mITSWork.size();
      // working copy of outer track param
      auto& trc = mITSWork.emplace_back(static_cast<const o2::track::TrackParCov&>(trcOrig.getParamOut()), rEntry, it);
      
      if (!trc.rotate(o2::utils::Angle2Alpha(trc.getPhiPos()))) {
        mITSWork.pop_back(); // discard failed track
        continue;
      }
      // make sure the track is at the ref. radius
      if (!propagateToRefX(trc)) {
        mITSWork.pop_back(); // discard failed track
        continue;            // add to cache only those ITS tracks which reached ref.X and have reasonable snp
      }
      if (mMCTruthON) {
        mITSLblWork.emplace_back(mITSTrkLabels->getLabels(it)[0]);
      }
      trc.roFrame = irof;

      // cache work track index
      int sector = o2::utils::Angle2Sector(trc.getAlpha());
      mITSSectIndexCache[sector].push_back(nWorkTracks);

      // If the ITS track is very close to the sector edge, it may match also to a TPC track in the neighb. sector.
      // For a track with Yr and Phir at Xr the distance^2 between the poisition of this track in the neighb. sector
      // when propagated to Xr (in this neighbouring sector) and the edge will be (neglecting the curvature)
      // [(Xr*tg(10)-Yr)/(tgPhir+tg70)]^2  / cos(70)^2  // for the next sector
      // [(Xr*tg(10)+Yr)/(tgPhir-tg70)]^2  / cos(70)^2  // for the prev sector
      // Distances to the sector edges in neighbourings sectors (at Xref in theit proper frames)
      float tgp = trc.getSnp(), trcY = trc.getY();
      tgp /= std::sqrt((1.f - tgp) * (1.f + tgp)); // tan of track direction XY

      // sector up
      float dy2Up = (YMaxAtXMatchingRef - trcY) / (tgp + Tan70);
      if ((dy2Up * dy2Up * Cos70I2) < mSectEdgeMargin2) { // need to check this track for matching in sector up
        addLastTrackCloneForNeighbourSector(sector < (o2::constants::math::NSectors - 1) ? sector + 1 : 0);
      }
      // sector down
      float dy2Dn = (YMaxAtXMatchingRef + trcY) / (tgp - Tan70);
      if ((dy2Dn * dy2Dn * Cos70I2) < mSectEdgeMargin2) { // need to check this track for matching in sector down
        addLastTrackCloneForNeighbourSector(sector > 1 ? sector - 1 : o2::constants::math::NSectors - 1);
      }
    }
  }

  // sort tracks in each sector according to their time, then tgl
  for (int sec = o2::constants::math::NSectors; sec--;) {
    auto& indexCache = mITSSectIndexCache[sec];
    LOG(INFO) << "Sorting sector" << sec << " | " << indexCache.size() << " ITS tracks";
    if (!indexCache.size()) {
      continue;
    }
    std::sort(indexCache.begin(), indexCache.end(), [this](int a, int b) {
      auto& trackA = mITSWork[a];
      auto& trackB = mITSWork[b];
      if (trackA.roFrame < trackB.roFrame) { // ITS tracks have the same time coverage
        return true;
      } else if (trackA.roFrame > trackB.roFrame) {
        return false;
      }
      return trackA.getTgl() < trackB.getTgl();
    });

    /* TOREMOVE?
    // build array of 1st entries with of each ITS ROFrame
    int nbins = 1 + mITSWork[indexCache.back()].roFrame;
    auto& tbinStart = mITSTimeBinStart[sec];
    tbinStart.resize(nbins > 1 ? nbins : 1, -1);
    tbinStart[0] = 0;
    for (int itr = 0; itr < (int)indexCache.size(); itr++) {
      auto& trc = mITSWork[indexCache[itr]];
      if (tbinStart[trc.roFrame] == -1) {
        tbinStart[trc.roFrame] = itr;
      }
    }
    for (int i = 1; i < nbins; i++) {
      if (tbinStart[i] == -1) { // fill gaps with preceding indices. TODO ? Shall we do this for cont.readout in ITS only?
        tbinStart[i] = tbinStart[i - 1];
      }
    }
    */
  } // loop over tracks of single sector
  mMatchRecordsITS.reserve(mITSWork.size() * mMaxMatchCandidates);

  return true;
}

//_____________________________________________________
bool MatchTPCITS::prepareFITInfo()
{
  // If available, read FIT Info

  if (mFITInfoInp) { // for input from tree read ROFrecords vector, in DPL IO mode the vector should be already attached
    if (!mDPLIO) {
      mTreeFITInfo->GetEntry(0);
    }
    LOG(INFO) << "Loaded FIT Info with " << mFITInfoInp->size() << " entries";
  }

  return true;
}

//_____________________________________________________
bool MatchTPCITS::loadTPCTracksNextChunk()
{
  ///< load next chunk of TPC data
  mTimerIO.Start(false);
  while (++mCurrTPCTracksTreeEntry < mTreeTPCTracks->GetEntries()) {
    mTreeTPCTracks->GetEntry(mCurrTPCTracksTreeEntry);
    LOG(DEBUG) << "Loading TPC tracks entry " << mCurrTPCTracksTreeEntry << " -> " << mTPCTracksArrayInp->size()
               << " tracks";
    if (mTPCTracksArrayInp->size() < 1) {
      continue;
    }
    mTimerIO.Stop();
    return true;
  }
  mTimerIO.Stop();
  return false;
}

//_____________________________________________________
void MatchTPCITS::doMatching(int sec)
{
  ///< run matching for currently cached ITS data for given TPC sector
  auto& cacheITS = mITSSectIndexCache[sec];   // array of cached ITS track indices for this sector
  auto& cacheTPC = mTPCSectIndexCache[sec];   // array of cached ITS track indices for this sector
  auto& tbinStartTPC = mTPCTimeBinStart[sec]; // array of 1st TPC track with timeMax in ITS ROFrame
  auto& tbinStartITS = mITSTimeBinStart[sec];
  int nTracksTPC = cacheTPC.size(), nTracksITS = cacheITS.size();
  if (!nTracksTPC || !nTracksITS) {
    LOG(INFO) << "Matchng sector " << sec << " : N tracks TPC:" << nTracksTPC << " ITS:" << nTracksITS << " in sector "
              << sec;
    return;
  }

  /// full drift time + safety margin
  float maxTDriftSafe = (mNTPCBinsFullDrift + mTPCITSTimeBinSafeMargin + mTPCTimeEdgeTSafeMargin);

  // get min ROFrame (in TPC time-bins) of ITS tracks currently in cache
  auto minROFITS = mITSWork[cacheITS.front()].roFrame;

  if (minROFITS >= int(tbinStartTPC.size())) {
    LOG(INFO) << "ITS min ROFrame " << minROFITS << " exceeds all cached TPC track ROF eqiuvalent "
              << cacheTPC.size() - 1;
    return;
  }

  int nCheckTPCControl = 0, nCheckITSControl = 0, nMatchesControl = 0; // temporary

  int idxMinTPC = tbinStartTPC[minROFITS]; // index of 1st cached TPC track within cached ITS ROFrames
  for (int itpc = idxMinTPC; itpc < nTracksTPC; itpc++) {
    auto& trefTPC = mTPCWork[cacheTPC[itpc]];
    // estimate ITS 1st ROframe bin this track may match to: TPC track are sorted according to their
    // timeMax, hence the timeMax - MaxmNTPCBinsFullDrift are non-decreasing
    int itsROBin = tpcTimeBin2ITSROFrame(trefTPC.timeBins.max() - maxTDriftSafe);
    if (itsROBin >= int(tbinStartITS.size())) { // time of TPC track exceeds the max time of ITS in the cache
      break;
    }
    int iits0 = tbinStartITS[itsROBin];
    nCheckTPCControl++;
    for (auto iits = iits0; iits < nTracksITS; iits++) {
      auto& trefITS = mITSWork[cacheITS[iits]];
      const auto& timeITS = mITSROFTimes[trefITS.roFrame];
      // compare if the ITS and TPC tracks may overlap in time
      if (trefTPC.timeBins < timeITS) {
        // since TPC tracks are sorted in timeMax and ITS tracks are sorted in timeMin
        // all following ITS tracks also will not match
        break;
      }
      if (trefTPC.timeBins > timeITS) { // its bracket precedes TPC bracket
        continue;
      }
      nCheckITSControl++;
      float chi2 = -1;
      int rejFlag = compareTPCITSTracks(trefITS, trefTPC, chi2);

#ifdef _ALLOW_DEBUG_TREES_
      if (mDBGOut && ((rejFlag == Accept && isDebugFlag(MatchTreeAccOnly)) || isDebugFlag(MatchTreeAll))) {
        fillTPCITSmatchTree(cacheITS[iits], cacheTPC[itpc], rejFlag, chi2);
      }
#endif

      if (rejFlag == RejectOnTgl) {
        // ITS tracks in each ROFrame are ordered in Tgl, hence if this check failed on Tgl check
        // (i.e. tgl_its>tgl_tpc+tolerance), tnem all other ITS tracks in this ROFrame will also have tgl too large.
        // Jump on the 1st ITS track of the next ROFrame
        int rof = trefITS.roFrame;
        bool stop = false;
        do {
          if (++rof >= int(tbinStartITS.size())) {
            stop = true;
            break; // no more ITS ROFrames in cache
          }
          iits = tbinStartITS[rof] - 1;                  // next track to be checked -1
        } while (iits <= tbinStartITS[trefITS.roFrame]); // skip empty bins
        if (stop) {
          break;
        }
        continue;
      }
      if (rejFlag != Accept) {
        continue;
      }
      registerMatchRecordTPC(cacheITS[iits], cacheTPC[itpc], chi2); // register matching candidate
      nMatchesControl++;
    }
  }

  LOG(INFO) << "Match sector " << sec << " N tracks TPC:" << nTracksTPC << " ITS:" << nTracksITS
            << " N TPC tracks checked: " << nCheckTPCControl << " (starting from " << idxMinTPC
            << "), checks: " << nCheckITSControl << ", matches:" << nMatchesControl;
}

//______________________________________________
void MatchTPCITS::suppressMatchRecordITS(int itsID, int tpcID)
{
  ///< suppress the reference on the tpcID in the list of matches recorded for itsID
  auto& tITS = mITSWork[itsID];
  int topID = MinusOne, recordID = tITS.matchID;   // 1st entry in mMatchRecordsITS
  while (recordID > MinusOne) {                    // navigate over records for given ITS track
    if (mMatchRecordsITS[recordID].partnerID == tpcID) {
      // unlink this record, connecting its child to its parrent
      if (topID < 0) {
        tITS.matchID = mMatchRecordsITS[recordID].nextRecID;
      } else {
        mMatchRecordsITS[topID].nextRecID = mMatchRecordsITS[recordID].nextRecID;
      }
      return;
    }
    topID = recordID;
    recordID = mMatchRecordsITS[recordID].nextRecID; // check next record
  }
}

//______________________________________________
bool MatchTPCITS::registerMatchRecordTPC(int iITS, int iTPC, float chi2)
{
  ///< record matching candidate, making sure that number of ITS candidates per TPC track, sorted
  ///< in matching chi2 does not exceed allowed number
  auto& tTPC = mTPCWork[iTPC];                 // get matchRecord structure of this TPC track, create if none
  if (tTPC.matchID < 0) {                      // no matches yet, just add new record
    registerMatchRecordITS(iITS, iTPC, chi2);  // register TPC track in the ITS records
    tTPC.matchID = mMatchRecordsTPC.size();    // new record will be added in the end
    mMatchRecordsTPC.emplace_back(iITS, chi2); // create new record with empty reference on next match
    return true;
  }

  int count = 0, nextID = tTPC.matchID, topID = MinusOne;
  do {
    auto& nextMatchRec = mMatchRecordsTPC[nextID];
    count++;
    if (chi2 < nextMatchRec.chi2) { // need to insert new record before nextMatchRec?
      if (count < mMaxMatchCandidates) {
        break; // will insert in front of nextID
      } else { // max number of candidates reached, will overwrite the last one
        nextMatchRec.chi2 = chi2;
        suppressMatchRecordITS(nextMatchRec.partnerID, iTPC); // flag as disabled the overriden ITS match
        registerMatchRecordITS(iITS, tTPC.matchID, chi2);     // register TPC track entry in the ITS records
        nextMatchRec.partnerID = iITS;                        // reuse the record of suppressed ITS match to store better one
        return true;
      }
    }
    topID = nextID; // check next match record
    nextID = nextMatchRec.nextRecID;
  } while (nextID > MinusOne);

  // if count == mMaxMatchCandidates, the max number of candidates was already reached, and the
  // new candidated was either discarded (if its chi2 is worst one) or has overwritten worst
  // existing candidate. Otherwise, we need to add new entry
  if (count < mMaxMatchCandidates) {
    if (topID < 0) {                                                       // the new match is top candidate
      topID = tTPC.matchID = mMatchRecordsTPC.size();                      // register new record as top one
    } else {                                                               // there are better candidates
      topID = mMatchRecordsTPC[topID].nextRecID = mMatchRecordsTPC.size(); // register to his parent
    }
    // nextID==-1 will mean that the while loop run over all candidates->the new one is the worst (goes to the end)
    registerMatchRecordITS(iITS, tTPC.matchID, chi2);  // register TPC track in the ITS records
    mMatchRecordsTPC.emplace_back(iITS, chi2, nextID); // create new record with empty reference on next match
    // make sure that after addition the number of candidates don't exceed allowed number
    count++;
    while (nextID > MinusOne) {
      if (count > mMaxMatchCandidates) {
        suppressMatchRecordITS(mMatchRecordsTPC[nextID].partnerID, iTPC);
        // exclude nextID record, w/o changing topID (which becomes the last record)
        nextID = mMatchRecordsTPC[topID].nextRecID = mMatchRecordsTPC[nextID].nextRecID;
        continue;
      }
      count++;
      topID = nextID;
      nextID = mMatchRecordsTPC[nextID].nextRecID;
    }
    return true;
  } else {
    return false; // unless nextID was assigned OverrideExisting, new candidate was discarded
  }
}

//______________________________________________
void MatchTPCITS::registerMatchRecordITS(int iITS, int iTPC, float chi2)
{
  ///< register TPC match in ITS tracks match records, ordering then in chi2
  auto& tITS = mITSWork[iITS];
  int idnew = mMatchRecordsITS.size();
  mMatchRecordsITS.emplace_back(iTPC, chi2); // associate iTPC with this record
  if (tITS.matchID < 0) {
    tITS.matchID = idnew;
    return;
  }
  // there are other matches for this ITS track, insert the new record preserving chi2 order
  // navigate till last record or the one with worse chi2
  int topID = MinusOne, nextRecord = tITS.matchID;
  mMatchRecordsITS.emplace_back(iTPC, chi2); // associate iTPC with this record
  auto& newRecord = mMatchRecordsITS.back();
  do {
    auto& recITS = mMatchRecordsITS[nextRecord];
    if (chi2 < recITS.chi2) {           // insert before this one
      newRecord.nextRecID = nextRecord; // new one will refer to old one it overtook
      if (topID < 0) {
        tITS.matchID = idnew; // the new one is the best match, track will refer to it
      } else {
        mMatchRecordsITS[topID].nextRecID = idnew; // new record will follow existing better one
      }
      return;
    }
    topID = nextRecord;
    nextRecord = mMatchRecordsITS[nextRecord].nextRecID;
  } while (nextRecord > MinusOne);

  // if we reached here, the new record should be added in the end
  mMatchRecordsITS[topID].nextRecID = idnew; // register new link
}

//______________________________________________
int MatchTPCITS::compareTPCITSTracks(const TrackLocITS& tITS, const TrackLocTPC& tTPC, float& chi2) const
{
  ///< compare pair of ITS and TPC tracks
  auto& trackTPC = tTPC;
  auto& trackITS = tITS;
  chi2 = -1.f;
  int rejFlag = Accept;
  float diff; // make rough check differences and their nsigmas

  // start with check on Tgl, since rjection on it will allow to profit from sorting
  diff = trackITS.getParam(o2::track::kTgl) - trackTPC.getParam(o2::track::kTgl);
  if ((rejFlag = roughCheckDif(diff, mCrudeAbsDiffCut[o2::track::kTgl], RejectOnTgl))) {
    return rejFlag;
  }
  diff *= diff / (trackITS.getDiagError2(o2::track::kTgl) + trackTPC.getDiagError2(o2::track::kTgl));
  if ((rejFlag = roughCheckDif(diff, mCrudeNSigma2Cut[o2::track::kTgl], RejectOnTgl + NSigmaShift))) {
    return rejFlag;
  }

  diff = trackITS.getParam(o2::track::kY) - trackTPC.getParam(o2::track::kY);
  if ((rejFlag = roughCheckDif(diff, mCrudeAbsDiffCut[o2::track::kY], RejectOnY))) {
    return rejFlag;
  }
  diff *= diff / (trackITS.getDiagError2(o2::track::kY) + trackTPC.getDiagError2(o2::track::kY));
  if ((rejFlag = roughCheckDif(diff, mCrudeNSigma2Cut[o2::track::kY], RejectOnY + NSigmaShift))) {
    return rejFlag;
  }

  if (mCompareTracksDZ) { // in continuous mode we usually don't use DZ
    diff = trackITS.getParam(o2::track::kZ) - trackTPC.getParam(o2::track::kZ);
    if ((rejFlag = roughCheckDif(diff, mCrudeAbsDiffCut[o2::track::kZ], RejectOnZ))) {
      return rejFlag;
    }
    diff *= diff / (trackITS.getDiagError2(o2::track::kZ) + trackTPC.getDiagError2(o2::track::kZ));
    if ((rejFlag = roughCheckDif(diff, mCrudeNSigma2Cut[o2::track::kZ], RejectOnZ + NSigmaShift))) {
      return rejFlag;
    }
  } else { // in continuous mode we use special check on allowed Z range
    if (trackITS.getParam(o2::track::kZ) - tTPC.zMax > mCrudeAbsDiffCut[o2::track::kZ])
      return RejectOnZ;
    if (trackITS.getParam(o2::track::kZ) - tTPC.zMin < -mCrudeAbsDiffCut[o2::track::kZ])
      return -RejectOnZ;
  }

  diff = trackITS.getParam(o2::track::kSnp) - trackTPC.getParam(o2::track::kSnp);
  if ((rejFlag = roughCheckDif(diff, mCrudeAbsDiffCut[o2::track::kSnp], RejectOnSnp))) {
    return rejFlag;
  }
  diff *= diff / (trackITS.getDiagError2(o2::track::kSnp) + trackTPC.getDiagError2(o2::track::kSnp));
  if ((rejFlag = roughCheckDif(diff, mCrudeNSigma2Cut[o2::track::kSnp], RejectOnSnp + NSigmaShift))) {
    return rejFlag;
  }

  diff = trackITS.getParam(o2::track::kQ2Pt) - trackTPC.getParam(o2::track::kQ2Pt);
  if ((rejFlag = roughCheckDif(diff, mCrudeAbsDiffCut[o2::track::kQ2Pt], RejectOnQ2Pt))) {
    return rejFlag;
  }
  diff *= diff / (trackITS.getDiagError2(o2::track::kQ2Pt) + trackTPC.getDiagError2(o2::track::kQ2Pt));
  if ((rejFlag = roughCheckDif(diff, mCrudeNSigma2Cut[o2::track::kQ2Pt], RejectOnQ2Pt + NSigmaShift))) {
    return rejFlag;
  }
  // calculate mutual chi2 excluding Z in continuos mode
  chi2 = getPredictedChi2NoZ(tITS, tTPC);
  if (chi2 > mCutMatchingChi2)
    return RejectOnChi2;

  return Accept;
}

//______________________________________________
void MatchTPCITS::printCandidatesTPC() const
{
  ///< print mathing records
  int ntpc = mTPCWork.size();
  printf("\n\nPrinting all TPC -> ITS matches for %d TPC tracks\n", ntpc);
  for (int i = 0; i < ntpc; i++) {
    const auto& tTPC = mTPCWork[i];
    int nm = getNMatchRecordsTPC(tTPC);
    printf("*** trackTPC#%6d %6d(%4d) : Ncand = %d\n", i, tTPC.source.getIndex(), tTPC.source.getEvent(), nm);
    int count = 0, recID = tTPC.matchID;
    while (recID > MinusOne) {
      const auto& rcTPC = mMatchRecordsTPC[recID];
      const auto& tITS = mITSWork[rcTPC.partnerID];
      printf("  * cand %2d : ITS track %6d(%4d) Chi2: %.2f\n", count, tITS.source.getIndex(),
             tITS.source.getEvent(), rcTPC.chi2);
      count++;
      recID = rcTPC.nextRecID;
    }
  }
}

//______________________________________________
void MatchTPCITS::printCandidatesITS() const
{
  ///< print mathing records
  int nits = mITSWork.size();
  printf("\n\nPrinting all ITS -> TPC matches for %d ITS tracks\n", nits);

  for (int i = 0; i < nits; i++) {
    const auto& tITS = mITSWork[i];
    printf("*** trackITS#%6d %6d(%4d) : Ncand = %d\n", i, tITS.source.getIndex(), tITS.source.getEvent(),
           getNMatchRecordsITS(tITS));
    int count = 0, recID = tITS.matchID;
    while (recID > MinusOne) {
      const auto& rcITS = mMatchRecordsITS[recID];
      const auto& tTPC = mTPCWork[rcITS.partnerID];
      printf("  * cand %2d : TPC track %6d(%4d) Chi2: %.2f\n", count, tTPC.source.getIndex(),
             tTPC.source.getEvent(), rcITS.chi2);
      count++;
      recID = rcITS.nextRecID;
    }
  }
}

//______________________________________________
float MatchTPCITS::getPredictedChi2NoZ(const o2::track::TrackParCov& tr1, const o2::track::TrackParCov& tr2) const
{
  /// get chi2 between 2 tracks, neglecting Z parameter.
  /// 2 tracks must be defined at the same parameters X,alpha (check is currently commented)

  //  if (std::abs(tr1.getAlpha() - tr2.getAlpha()) > FLT_EPSILON) {
  //    LOG(ERROR) << "The reference Alpha of the tracks differ: "
  //	       << tr1.getAlpha() << " : " << tr2.getAlpha();
  //    return 2. * o2::track::HugeF;
  //  }
  //  if (std::abs(tr1.getX() - tr2.getX()) > FLT_EPSILON) {
  //    LOG(ERROR) << "The reference X of the tracks differ: "
  //	       << tr1.getX() << " : " << tr2.getX();
  //    return 2. * o2::track::HugeF;
  //  }
  MatrixDSym4 covMat;
  covMat(0, 0) = static_cast<double>(tr1.getSigmaY2()) + static_cast<double>(tr2.getSigmaY2());
  covMat(1, 0) = static_cast<double>(tr1.getSigmaSnpY()) + static_cast<double>(tr2.getSigmaSnpY());
  covMat(1, 1) = static_cast<double>(tr1.getSigmaSnp2()) + static_cast<double>(tr2.getSigmaSnp2());
  covMat(2, 0) = static_cast<double>(tr1.getSigmaTglY()) + static_cast<double>(tr2.getSigmaTglY());
  covMat(2, 1) = static_cast<double>(tr1.getSigmaTglSnp()) + static_cast<double>(tr2.getSigmaTglSnp());
  covMat(2, 2) = static_cast<double>(tr1.getSigmaTgl2()) + static_cast<double>(tr2.getSigmaTgl2());
  covMat(3, 0) = static_cast<double>(tr1.getSigma1PtY()) + static_cast<double>(tr2.getSigma1PtY());
  covMat(3, 1) = static_cast<double>(tr1.getSigma1PtSnp()) + static_cast<double>(tr2.getSigma1PtSnp());
  covMat(3, 2) = static_cast<double>(tr1.getSigma1PtTgl()) + static_cast<double>(tr2.getSigma1PtTgl());
  covMat(3, 3) = static_cast<double>(tr1.getSigma1Pt2()) + static_cast<double>(tr2.getSigma1Pt2());
  if (!covMat.Invert()) {
    LOG(ERROR) << "Cov.matrix inversion failed: " << covMat;
    return 2. * o2::track::HugeF;
  }
  double chi2diag = 0., chi2ndiag = 0.,
         diff[o2::track::kNParams - 1] = {tr1.getParam(o2::track::kY) - tr2.getParam(o2::track::kY),
                                          tr1.getParam(o2::track::kSnp) - tr2.getParam(o2::track::kSnp),
                                          tr1.getParam(o2::track::kTgl) - tr2.getParam(o2::track::kTgl),
                                          tr1.getParam(o2::track::kQ2Pt) - tr2.getParam(o2::track::kQ2Pt)};
  for (int i = o2::track::kNParams - 1; i--;) {
    chi2diag += diff[i] * diff[i] * covMat(i, i);
    for (int j = i; j--;) {
      chi2ndiag += diff[i] * diff[j] * covMat(i, j);
    }
  }
  return chi2diag + 2. * chi2ndiag;
}

//______________________________________________
void MatchTPCITS::addLastTrackCloneForNeighbourSector(int sector)
{
  // add clone of the src ITS track cashe, propagate it to ref.X in requested sector
  // and register its index in the sector cache. Used for ITS tracks which are so close
  // to their setctor edge that their matching should be checked also in the neighbouring sector
  mITSWork.reserve(mITSWork.size()+100);
  mITSWork.push_back(mITSWork.back()); // clone the last track defined in given sector
  auto& trc = mITSWork.back();
  if (trc.rotate(o2::utils::Sector2Angle(sector)) &&
      o2::base::Propagator::Instance()->PropagateToXBxByBz(trc, XMatchingRef, o2::constants::physics::MassPionCharged, MaxSnp,
                                                           2., o2::base::Propagator::USEMatCorrNONE)) {
    // TODO: use faster prop here, no 3d field, materials
    mITSSectIndexCache[sector].push_back(mITSWork.size() - 1); // register track CLONE
    if (mMCTruthON) {
      mITSLblWork.emplace_back(mITSTrkLabels->getLabels(trc.source.getIndex())[0]);
    }
  } else {
    mITSWork.pop_back(); // rotation / propagation failed
  }
}

//______________________________________________
bool MatchTPCITS::propagateToRefX(o2::track::TrackParCov& trc)
{
  // propagate track to matching reference X, making sure its assigned alpha
  // is consistent with TPC sector
  bool refReached = false;
  refReached = XMatchingRef < 10.; // RS: tmp, to cover XMatchingRef~0
  while (o2::base::Propagator::Instance()->PropagateToXBxByBz(trc, XMatchingRef, o2::constants::physics::MassPionCharged,
                                                              MaxSnp, 2., mUseMatCorrFlag)) {
    if (refReached) {
      break;
    }
    // make sure the track is indeed within the sector defined by alpha
    if (fabs(trc.getY()) < XMatchingRef * tan(o2::constants::math::SectorSpanRad / 2)) {
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

//______________________________________________
void MatchTPCITS::print() const
{
  ///< print all settings
  printf("\n******************** TPC-ITS matching component ********************\n");
  if (!mInitDone) {
    printf("init is not done yet\n");
    return;
  }

  printf("MC truth: %s\n", mMCTruthON ? "on" : "off");
  printf("Matching reference X: %.3f\n", XMatchingRef);
  printf("Account Z dimension: %s\n", mCompareTracksDZ ? "on" : "off");
  printf("Cut on matching chi2: %.3f\n", mCutMatchingChi2);
  printf("Max number ITS candidates per TPC track: %d\n", mMaxMatchCandidates);
  printf("Crude cut on track params: ");
  for (int i = 0; i < o2::track::kNParams; i++) {
    printf(" %.3e", mCrudeAbsDiffCut[i]);
  }
  printf("\n");

  printf("NSigma^2 cut on track params: ");
  for (int i = 0; i < o2::track::kNParams; i++) {
    printf(" %6.2f", mCrudeNSigma2Cut[i]);
  }
  printf("\n");

  printf("TPC-ITS time(bins) bracketing safety margin: %6.2f\n", mTimeBinTolerance);
  printf("TPC Z->time(bins) bracketing safety margin: %6.2f\n", mTPCTimeEdgeZSafeMargin);

#ifdef _ALLOW_DEBUG_TREES_

  printf("Output debug tree (%s) file: %s\n", mDBGFlags ? "on" : "off", mDebugTreeFileName.data());
  if (getDebugFlags()) {
    printf("Debug stream flags:\n");
    if (isDebugFlag(MatchTreeAll | MatchTreeAccOnly)) {
      printf("* matching canditate pairs: %s\n", isDebugFlag(MatchTreeAccOnly) ? "accepted" : "all");
    }
    if (isDebugFlag(WinnerMatchesTree)) {
      printf("* winner matches\n");
    }
  }
#endif

  printf("**********************************************************************\n");
}

//______________________________________________
void MatchTPCITS::refitWinners(bool loopInITS)
{
  ///< refit winning tracks

  mTimerRefit.Start(false);
  LOG(INFO) << "Refitting winner matches";
  mWinnerChi2Refit.resize(mITSWork.size(), -1.f);
  if (loopInITS) {
    int iTPC = 0; // will be ignored
    for (int iITS = 0; iITS < (int)mITSWork.size(); iITS++) {
      if (!refitTrackTPCITSloopITS(iITS, iTPC)) {
        continue;
      }
      mWinnerChi2Refit[iITS] = mMatchedTracks.back().getChi2Refit();
    }
  } else {
    int iITS;
    for (int iTPC = 0; iTPC < (int)mTPCWork.size(); iTPC++) {
      if (!refitTrackTPCITSloopTPC(iTPC, iITS)) {
        continue;
      }
      mWinnerChi2Refit[iITS] = mMatchedTracks.back().getChi2Refit();
    }
  }
  /*
  */
  // flush last tracks
  mTimerRefit.Stop();

  if (mMatchedTracks.size() && mOutputTree) {
    mTimerIO.Start(false);
    mOutputTree->Fill();
    mTimerIO.Stop();
  }
}

//______________________________________________
bool MatchTPCITS::refitTrackTPCITSloopITS(int iITS, int& iTPC)
{
  ///< refit in inward direction the pair of TPC and ITS tracks

  const float maxStep = 2.f; // max propagation step (TODO: tune)

  const auto& tITS = mITSWork[iITS];
  if (isDisabledITS(tITS)) {
    return false; // no match
  }
  const auto& itsMatchRec = mMatchRecordsITS[tITS.matchID];
  iTPC = itsMatchRec.partnerID;
  const auto& tTPC = mTPCWork[iTPC];

  if (!mDPLIO) {
    mTimerRefit.Stop(); // stop during IO
    loadITSTracksChunk(tITS.source.getEvent());
    loadTPCTracksChunk(tTPC.source.getEvent());
    loadTPCClustersChunk(tTPC.source.getEvent());
    mTimerRefit.Start(false);
  }
  auto itsTrOrig = (*mITSTracksArrayInp)[tITS.source.getIndex()]; // currently we store clusterIDs in the track

  mMatchedTracks.emplace_back(tTPC, tITS); // create a copy of TPC track at xRef
  auto& trfit = mMatchedTracks.back();
  // in continuos mode the Z of TPC track is meaningless, unless it is CE crossing
  // track (currently absent, TODO)
  if (!mCompareTracksDZ) {
    trfit.setZ(tITS.getZ()); // fix the seed Z
  }
  auto dzCorr = trfit.getZ() - tTPC.getZ();
  float deltaT = dzCorr * mZ2TPCBin; // time correction in time-bins

  // refit TPC track inward into the ITS
  int nclRefit = 0, ncl = itsTrOrig.getNumberOfClusters();
  float chi2 = 0.f;
  auto geom = o2::its::GeometryTGeo::Instance();
  auto propagator = o2::base::Propagator::Instance();
  // NOTE: the ITS cluster index is stored wrt 1st cluster of relevant ROF, while here we extract clusters from the
  // buffer for the whole TF. Therefore, we should shift the index by the entry of the ROF's 1st cluster in the global cluster buffer
  int clusIndOffs = (*mITSClusterROFRec)[tITS.roFrame].getROFEntry().getIndex();

  int clEntry = itsTrOrig.getFirstClusterEntry();
  for (int icl = 0; icl < ncl; icl++) {
    const auto& clus = (*mITSClustersArrayInp)[clusIndOffs + (*mITSTrackClusIdxInp)[clEntry++]];
    float alpha = geom->getSensorRefAlpha(clus.getSensorID()), x = clus.getX();
    if (!trfit.rotate(alpha) ||
        // note: here we also calculate the L,T integral (in the inward direction, but this is irrelevant)
        // note: we should eventually use TPC pid in the refit (TODO)
        // note: since we are at small R, we can use field BZ component at origin rather than 3D field
        // !propagator->PropagateToXBxByBz(trfit, x, o2::constants::physics::MassPionCharged,
        !propagator->propagateToX(trfit, x, propagator->getNominalBz(), o2::constants::physics::MassPionCharged,
                                  MaxSnp, maxStep, mUseMatCorrFlag, &trfit.getLTIntegralOut())) {
      break;
    }
    chi2 += trfit.getPredictedChi2(static_cast<const o2::BaseCluster<float>&>(clus));
    if (!trfit.update(static_cast<const o2::BaseCluster<float>&>(clus))) {
      break;
    }
    nclRefit++;
  }
  if (nclRefit != ncl) {
    printf("FAILED after ncl=%d\n", nclRefit);
    printf("its was:  ");
    tITS.print();
    printf("tpc was:  ");
    tTPC.print();
    mMatchedTracks.pop_back(); // destroy failed track
    return false;
  }

  // we need to update the LTOF integral by the distance to the "primary vertex"
  const Point3D<float> vtxDummy; // at the moment using dummy vertex: TODO use MeanVertex constraint instead
  if (!propagator->propagateToDCA(vtxDummy, trfit, propagator->getNominalBz(), o2::constants::physics::MassPionCharged,
                                  maxStep, mUseMatCorrFlag, &trfit.getLTIntegralOut())) {
    LOG(ERROR) << "LTOF integral might be incorrect";
  }

  /// precise time estimate
  auto tpcTrOrig = (*mTPCTracksArrayInp)[tTPC.source.getIndex()];
  float timeTB = tpcTrOrig.getTime0() - mNTPCBinsFullDrift;
  if (tpcTrOrig.hasASideClustersOnly()) {
    timeTB += deltaT;
  } else if (tpcTrOrig.hasCSideClustersOnly()) {
    timeTB -= deltaT;
  } else {
    // TODO : special treatment of tracks crossing the CE
  }
  // convert time in timebins to time in microseconds
  float time = timeTB * mTPCTBinMUS;
  // estimate the error on time
  float timeErr = std::sqrt(tITS.getSigmaZ2() + tTPC.getSigmaZ2()) * mTPCVDrift0Inv;

  // outward refit
  auto& tracOut = trfit.getParamOut(); // this track is already at the matching reference X
  {
    int icl = tpcTrOrig.getNClusterReferences() - 1;
    uint8_t sector, prevsector, row, prevrow;
    uint32_t clusterIndexInRow;
    std::array<float, 2> clsYZ;
    std::array<float, 3> clsCov = {};
    float clsX;

    const auto& cl = tpcTrOrig.getCluster(icl, *mTPCClusterIdxStruct, sector, row);
    mTPCTransform->Transform(sector, row, cl.getPad(), cl.getTime(), clsX, clsYZ[0], clsYZ[1], timeTB);
    // rotate to 1 cluster's sector
    if (!tracOut.rotate(o2::utils::Sector2Angle(sector % 18))) {
      LOG(WARNING) << "Rotation to sector " << int(sector % 18) << " failed";
      mMatchedTracks.pop_back(); // destroy failed track
      return false;
    }
    // TODO: consider propagating in empty space till TPC entrance in large step, and then in more detailed propagation with mat. corrections

    // propagate to 1st cluster X
    if (!propagator->PropagateToXBxByBz(tracOut, clsX, o2::constants::physics::MassPionCharged, MaxSnp, 10., mUseMatCorrFlag, &trfit.getLTIntegralOut())) {
      LOG(WARNING) << "Propagation to 1st cluster at X=" << clsX << " failed, Xtr=" << tracOut.getX() << " snp=" << tracOut.getSnp();
      mMatchedTracks.pop_back(); // destroy failed track
      return false;
    }
    //
    mTPCClusterParam->GetClusterErrors2(row, clsYZ[1], tracOut.getSnp(), tracOut.getTgl(), clsCov[0], clsCov[2]);
    //
    float chi2Out = tracOut.getPredictedChi2(clsYZ, clsCov);
    if (!tracOut.update(clsYZ, clsCov)) {
      LOG(WARNING) << "Update failed at 1st cluster, chi2 =" << chi2Out;
      mMatchedTracks.pop_back(); // destroy failed track
      return false;
    }
    prevrow = row;
    prevsector = sector;

    for (; icl--;) {
      const auto& cl = tpcTrOrig.getCluster(icl, *mTPCClusterIdxStruct, sector, row);
      if (row <= prevrow) {
        LOG(WARNING) << "New row/sect " << int(row) << '/' << int(sector) << " is <= the previous " << int(prevrow)
                     << '/' << int(prevsector) << " TrackID: " << tTPC.source.getIndex() << " Pt:" << tracOut.getPt();
        if (row < prevrow) {
          break;
        } else {
          continue; // just skip duplicate clusters
        }
      }
      prevrow = row;
      mTPCTransform->Transform(sector, row, cl.getPad(), cl.getTime(), clsX, clsYZ[0], clsYZ[1], timeTB);
      if (prevsector != sector) {
        prevsector = sector;
        if (!tracOut.rotate(o2::utils::Sector2Angle(sector % 18))) {
          LOG(WARNING) << "Rotation to sector " << int(sector % 18) << " failed";
          mMatchedTracks.pop_back(); // destroy failed track
          return false;
        }
      }
      if (!propagator->PropagateToXBxByBz(tracOut, clsX, o2::constants::physics::MassPionCharged, MaxSnp,
                                          10., o2::base::Propagator::USEMatCorrNONE, &trfit.getLTIntegralOut())) { // no material correction!
        LOG(INFO) << "Propagation to cluster " << icl << " (of " << tpcTrOrig.getNClusterReferences() << ") at X="
                  << clsX << " failed, Xtr=" << tracOut.getX() << " snp=" << tracOut.getSnp() << " pT=" << tracOut.getPt();
        mMatchedTracks.pop_back(); // destroy failed track
        return false;
      }
      chi2Out += tracOut.getPredictedChi2(clsYZ, clsCov);
      if (!tracOut.update(clsYZ, clsCov)) {
        LOG(WARNING) << "Update failed at cluster " << icl << ", chi2 =" << chi2Out;
        mMatchedTracks.pop_back(); // destroy failed track
        return false;
      }
    }
    // propagate to the outer edge of the TPC, TODO: check outer radius
    // Note: it is allowed to not reach the requested radius
    propagator->PropagateToXBxByBz(tracOut, XTPCOuterRef, o2::constants::physics::MassPionCharged, MaxSnp,
                                   10., mUseMatCorrFlag, &trfit.getLTIntegralOut());

    //    LOG(INFO) << "Refitted with chi2 = " << chi2Out;
  }

  trfit.setChi2Match(itsMatchRec.chi2);
  trfit.setChi2Refit(chi2);
  trfit.setTimeMUS(time, timeErr);
  trfit.setRefTPC(tTPC.source);
  trfit.setRefITS(tITS.source);

  if (mMCTruthON) { // store MC info
    mOutITSLabels.emplace_back(mITSLblWork[iITS]);
    mOutTPCLabels.emplace_back(mTPCLblWork[iTPC]);
  }

  //  trfit.print(); // DBG

  return true;
}

//______________________________________________
bool MatchTPCITS::refitTrackTPCITSloopTPC(int iTPC, int& iITS)
{
  ///< refit in inward direction the pair of TPC and ITS tracks

  const float maxStep = 2.f; // max propagation step (TODO: tune)

  const auto& tTPC = mTPCWork[iTPC];
  if (isDisabledTPC(tTPC)) {
    return false; // no match
  }
  const auto& tpcMatchRec = mMatchRecordsTPC[tTPC.matchID];
  iITS = tpcMatchRec.partnerID;
  const auto& tITS = mITSWork[iITS];

  if (!mDPLIO) {
    mTimerRefit.Stop(); // stop during IO
    loadITSTracksChunk(tITS.source.getEvent());
    loadTPCTracksChunk(tTPC.source.getEvent());
    loadTPCClustersChunk(tTPC.source.getEvent());
    mTimerRefit.Start(false);
  }
  auto itsTrOrig = (*mITSTracksArrayInp)[tITS.source.getIndex()]; // currently we store clusterIDs in the track

  mMatchedTracks.emplace_back(tTPC, tITS); // create a copy of TPC track at xRef
  auto& trfit = mMatchedTracks.back();
  // in continuos mode the Z of TPC track is meaningless, unless it is CE crossing
  // track (currently absent, TODO)
  if (!mCompareTracksDZ) {
    trfit.setZ(tITS.getZ()); // fix the seed Z
  }
  auto dzCorr = trfit.getZ() - tTPC.getZ();
  float deltaT = dzCorr * mZ2TPCBin; // time correction in time-bins

  // refit TPC track inward into the ITS
  int nclRefit = 0, ncl = itsTrOrig.getNumberOfClusters();
  float chi2 = 0.f;
  auto geom = o2::its::GeometryTGeo::Instance();
  auto propagator = o2::base::Propagator::Instance();
  // NOTE: the ITS cluster index is stored wrt 1st cluster of relevant ROF, while here we extract clusters from the
  // buffer for the whole TF. Therefore, we should shift the index by the entry of the ROF's 1st cluster in the global cluster buffer
  int clusIndOffs = (*mITSClusterROFRec)[tITS.roFrame].getROFEntry().getIndex();

  int clEntry = itsTrOrig.getFirstClusterEntry();
  for (int icl = 0; icl < ncl; icl++) {
    const auto& clus = (*mITSClustersArrayInp)[clusIndOffs + (*mITSTrackClusIdxInp)[clEntry++]];
    float alpha = geom->getSensorRefAlpha(clus.getSensorID()), x = clus.getX();
    if (!trfit.rotate(alpha) ||
        // note: here we also calculate the L,T integral (in the inward direction, but this is irrelevant)
        // note: we should eventually use TPC pid in the refit (TODO)
        // note: since we are at small R, we can use field BZ component at origin rather than 3D field
        // !propagator->PropagateToXBxByBz(trfit, x, o2::constants::physics::MassPionCharged,
        !propagator->propagateToX(trfit, x, propagator->getNominalBz(), o2::constants::physics::MassPionCharged,
                                  MaxSnp, maxStep, mUseMatCorrFlag, &trfit.getLTIntegralOut())) {
      break;
    }
    chi2 += trfit.getPredictedChi2(static_cast<const o2::BaseCluster<float>&>(clus));
    if (!trfit.update(static_cast<const o2::BaseCluster<float>&>(clus))) {
      break;
    }
    nclRefit++;
  }
  if (nclRefit != ncl) {
    printf("FAILED after ncl=%d\n", nclRefit);
    printf("its was:  ");
    tITS.print();
    printf("tpc was:  ");
    tTPC.print();
    mMatchedTracks.pop_back(); // destroy failed track
    return false;
  }

  // we need to update the LTOF integral by the distance to the "primary vertex"
  const Point3D<float> vtxDummy; // at the moment using dummy vertex: TODO use MeanVertex constraint instead
  if (!propagator->propagateToDCA(vtxDummy, trfit, propagator->getNominalBz(), o2::constants::physics::MassPionCharged,
                                  maxStep, mUseMatCorrFlag, &trfit.getLTIntegralOut())) {
    LOG(ERROR) << "LTOF integral might be incorrect";
  }

  /// precise time estimate
  auto tpcTrOrig = (*mTPCTracksArrayInp)[tTPC.source.getIndex()];
  float timeTB = tpcTrOrig.getTime0() - mNTPCBinsFullDrift;
  if (tpcTrOrig.hasASideClustersOnly()) {
    timeTB += deltaT;
  } else if (tpcTrOrig.hasCSideClustersOnly()) {
    timeTB -= deltaT;
  } else {
    // TODO : special treatment of tracks crossing the CE
  }
  // convert time in timebins to time in microseconds
  float time = timeTB * mTPCTBinMUS;
  // estimate the error on time
  float timeErr = std::sqrt(tITS.getSigmaZ2() + tTPC.getSigmaZ2()) * mTPCVDrift0Inv;

  // outward refit
  auto& tracOut = trfit.getParamOut(); // this track is already at the matching reference X
  {
    int icl = tpcTrOrig.getNClusterReferences() - 1;
    uint8_t sector, prevsector, row, prevrow;
    uint32_t clusterIndexInRow;
    std::array<float, 2> clsYZ;
    std::array<float, 3> clsCov = {};
    float clsX;

    const auto& cl = tpcTrOrig.getCluster(icl, *mTPCClusterIdxStruct, sector, row);
    mTPCTransform->Transform(sector, row, cl.getPad(), cl.getTime() - timeTB, clsX, clsYZ[0], clsYZ[1]);
    // rotate to 1 cluster's sector
    if (!tracOut.rotate(o2::utils::Sector2Angle(sector % 18))) {
      LOG(WARNING) << "Rotation to sector " << int(sector % 18) << " failed";
      mMatchedTracks.pop_back(); // destroy failed track
      return false;
    }
    // TODO: consider propagating in empty space till TPC entrance in large step, and then in more detailed propagation with mat. corrections

    // propagate to 1st cluster X
    if (!propagator->PropagateToXBxByBz(tracOut, clsX, o2::constants::physics::MassPionCharged, MaxSnp, 10., mUseMatCorrFlag, &trfit.getLTIntegralOut())) {
      LOG(WARNING) << "Propagation to 1st cluster at X=" << clsX << " failed, Xtr=" << tracOut.getX() << " snp=" << tracOut.getSnp();
      mMatchedTracks.pop_back(); // destroy failed track
      return false;
    }
    //
    mTPCClusterParam->GetClusterErrors2(row, clsYZ[1], tracOut.getSnp(), tracOut.getTgl(), clsCov[0], clsCov[2]);
    //
    float chi2Out = tracOut.getPredictedChi2(clsYZ, clsCov);
    if (!tracOut.update(clsYZ, clsCov)) {
      LOG(WARNING) << "Update failed at 1st cluster, chi2 =" << chi2Out;
      mMatchedTracks.pop_back(); // destroy failed track
      return false;
    }
    prevrow = row;
    prevsector = sector;

    for (; icl--;) {
      const auto& cl = tpcTrOrig.getCluster(icl, *mTPCClusterIdxStruct, sector, row);
      if (row <= prevrow) {
        LOG(WARNING) << "New row/sect " << int(row) << '/' << int(sector) << " is <= the previous " << int(prevrow)
                     << '/' << int(prevsector) << " TrackID: " << tTPC.source.getIndex() << " Pt:" << tracOut.getPt();
        if (row < prevrow) {
          break;
        } else {
          continue; // just skip duplicate clusters
        }
      }
      prevrow = row;
      mTPCTransform->Transform(sector, row, cl.getPad(), cl.getTime() - timeTB, clsX, clsYZ[0], clsYZ[1]);
      if (prevsector != sector) {
        prevsector = sector;
        if (!tracOut.rotate(o2::utils::Sector2Angle(sector % 18))) {
          LOG(WARNING) << "Rotation to sector " << int(sector % 18) << " failed";
          mMatchedTracks.pop_back(); // destroy failed track
          return false;
        }
      }
      if (!propagator->PropagateToXBxByBz(tracOut, clsX, o2::constants::physics::MassPionCharged, MaxSnp,
                                          10., o2::base::Propagator::USEMatCorrNONE, &trfit.getLTIntegralOut())) { // no material correction!
        LOG(INFO) << "Propagation to cluster " << icl << " (of " << tpcTrOrig.getNClusterReferences() << ") at X="
                  << clsX << " failed, Xtr=" << tracOut.getX() << " snp=" << tracOut.getSnp() << " pT=" << tracOut.getPt();
        mMatchedTracks.pop_back(); // destroy failed track
        return false;
      }
      chi2Out += tracOut.getPredictedChi2(clsYZ, clsCov);
      if (!tracOut.update(clsYZ, clsCov)) {
        LOG(WARNING) << "Update failed at cluster " << icl << ", chi2 =" << chi2Out;
        mMatchedTracks.pop_back(); // destroy failed track
        return false;
      }
    }
    // propagate to the outer edge of the TPC, TODO: check outer radius
    // Note: it is allowed to not reach the requested radius
    propagator->PropagateToXBxByBz(tracOut, XTPCOuterRef, o2::constants::physics::MassPionCharged, MaxSnp,
                                   10., mUseMatCorrFlag, &trfit.getLTIntegralOut());

    //    LOG(INFO) << "Refitted with chi2 = " << chi2Out;
  }

  trfit.setChi2Match(tpcMatchRec.chi2);
  trfit.setChi2Refit(chi2);
  trfit.setTimeMUS(time, timeErr);
  trfit.setRefTPC(tTPC.source);
  trfit.setRefITS(tITS.source);

  if (mMCTruthON) { // store MC info
    mOutITSLabels.emplace_back(mITSLblWork[iITS]);
    mOutTPCLabels.emplace_back(mTPCLblWork[iTPC]);
  }

  //  trfit.print(); // DBG

  return true;
}

//________________________________________________________
void MatchTPCITS::loadTPCClustersChunk(int chunk)
{
  // load single entry from ITS clusters tree
  if (mCurrTPCClustersTreeEntry != chunk) {
    mTimerIO.Start(false);
    mTPCClusterReader->read(mCurrTPCClustersTreeEntry = chunk);
    mTimerIO.Stop();
    mTPCClusterReader->fillIndex(*mTPCClusterIdxStructOwn.get(), mTPCClusterBufferOwn, mTPCClusterMCBufferOwn);
    mTPCClusterIdxStruct = mTPCClusterIdxStructOwn.get();
  }
}

//________________________________________________________
void MatchTPCITS::loadITSTracksChunk(int chunk)
{
  // load single entry from ITS tracks tree
  if (mTreeITSTracks && mTreeITSTracks->GetReadEntry() != chunk) {
    mTimerIO.Start(false);
    mTreeITSTracks->GetEntry(chunk);
    mTimerIO.Stop();
  }
}

//________________________________________________________
void MatchTPCITS::loadTPCTracksChunk(int chunk)
{
  // load single entry from TPC tracks tree
  if (mTreeTPCTracks && mTreeTPCTracks->GetReadEntry() != chunk) {
    mTimerIO.Start(false);
    mTreeTPCTracks->GetEntry(chunk);
    mTimerIO.Stop();
  }
}

//>>============================= AfterBurner for TPC-track / ITS cluster matching ===================>>
//______________________________________________
int MatchTPCITS::prepareTPCTracksAfterBurner()
{
  ///< select TPC tracks to be considered in afterburner
  mTPCABIndexCache.clear();
  mTPCABTimeBinStart.clear();
  const auto& outerLr = mRGHelper.layers.back();
  // to avoid difference between 3D field propagation and Bz-bazed getXatLabR we propagate RMax+margin
  const float ROuter = outerLr.rRange.max() + 0.5f;

  auto propagator = o2::base::Propagator::Instance();

  for (int iTPC = 0; iTPC < (int)mTPCWork.size(); iTPC++) {
    auto& tTPC = mTPCWork[iTPC];
    if (isDisabledTPC(tTPC)) {
      // Popagate to the vicinity of the out layer. Note: the Z of the track might be uncertain,
      // in this case the material corrections will be correct only in the limit of their uniformity in Z,
      // which should be good assumption....
      float xTgt;
      if (!tTPC.getXatLabR(ROuter, xTgt, propagator->getNominalBz(), o2::track::DirInward) ||
          !propagator->PropagateToXBxByBz(tTPC, xTgt, o2::constants::physics::MassPionCharged, MaxSnp, 2., mUseMatCorrFlag)) {
        continue;
      }
      mTPCABIndexCache.push_back(iTPC);
    }
  }
  // sort tracks according to their timeMin
  LOG(INFO) << "Sorting " << mTPCABIndexCache.size() << " selected TPC tracks for AfterBurner in tMin";
  std::sort(mTPCABIndexCache.begin(), mTPCABIndexCache.end(), [this](int a, int b) {
    auto& trcA = mTPCWork[a];
    auto& trcB = mTPCWork[b];
    return (trcA.timeBins.min() - trcB.timeBins.min()) < 0.;
  });

  return mTPCABIndexCache.size();
}

//______________________________________________
int MatchTPCITS::prepareInteractionTimes()
{
  // guess interaction times from various sources and relate with ITS rofs
  const float T0UncertaintyTB = 0.5 / (1e3 * mTPCTBinMUS); // assumed T0 time uncertainty (~0.5ns) in TPC timeBins
  mInteractions.clear();
  if (mFITInfoInp) {
    for (const auto& ft : *mFITInfoInp) {
      if (!ft.isValidTime(o2::ft0::RecPoints::TimeMean)) {
        continue;
      }
      auto fitTime = intRecord2TPCTimeBin(ft.getInteractionRecord()); // FIT time in TPC timebins
      // find corresponding ITS ROF, works both in cont. and trigg. modes (ignore T0 MeanTime within the BC)
      int nITSROFs = mITSROFTimes.size();
      for (int rof = 0; rof < nITSROFs; rof++) {
        if (mITSROFTimes[rof] < fitTime) {
          continue;
        }
        if (fitTime >= mITSROFTimes[rof].min()) { // belongs to this ROF
          mInteractions.emplace_back(ft.getInteractionRecord(), fitTime, T0UncertaintyTB, rof, o2::detectors::DetID::FT0);
        }
        break; // this or next ITSrof in time is > fitTime
      }
    }
  }
  return mInteractions.size();
}

//______________________________________________
void MatchTPCITS::runAfterBurner()
{
  mABTrackLinks.clear();

  int nIntCand = prepareInteractionTimes();
  int nTPCCand = prepareTPCTracksAfterBurner();
  LOG(INFO) << "AfterBurner will check " << nIntCand << " interaction candindates for " << nTPCCand << " TPC tracks";
  if (!nIntCand || !nTPCCand) {
    return;
  }
  int iC = 0;                                // interaction candindate to consider and result of its time-bracket comparison to TPC track
  int iCClean = iC;                          // id of the next candidate whose cache to be cleaned
  for (int itr = 0; itr < nTPCCand; itr++) { // TPC track indices are sorted in tMin
    const auto& tTPC = mTPCWork[mTPCABIndexCache[itr]];
    // find 1st interaction candidate compatible with time brackets of this track
    int iCRes;
    while ((iCRes = tTPC.timeBins.isOutside(mInteractions[iC].timeBins)) < 0 && ++iC < nIntCand) { // interaction precedes the track time-bracket
      cleanAfterBurnerClusRefCache(iC, iCClean);                                                   // if possible, clean unneeded cached cluster references
    }
    if (iCRes == 0) {
      int iCStart = iC, iCEnd = iC; // check all interaction candidates matching to this TPC track
      do {
        if (!mInteractions[iCEnd].clRefPtr) { // if not done yet, fill sorted cluster references for interaction candidate
          mInteractions[iCEnd].clRefPtr = &mITSChipClustersRefs.emplace_back();
          fillClustersForAfterBurner(mITSChipClustersRefs.back(), mInteractions[iCEnd].rofITS);
          // tst
          int ncl = mITSChipClustersRefs.back().clusterID.size();
          printf("loaded %d clusters at cache at %p\n", ncl, mInteractions[iCEnd].clRefPtr);
        }
      } while (++iCEnd < nIntCand && !tTPC.timeBins.isOutside(mInteractions[iCEnd].timeBins));

      auto lbl = mTPCLblWork[mTPCABIndexCache[itr]]; // tmp
      if (runAfterBurner(mTPCABIndexCache[itr], iCStart, iCEnd)) {
	lbl.print(); // tmp
	//tmp
	if (tTPC.matchID > MinusOne) {
	  printf("AB Matching tree for TPC WID %d and IC %d : %d\n", mTPCABIndexCache[itr], iCStart, iCEnd);
	  auto& llinks = mABTrackLinksList[tTPC.matchID];
	  printABTracksTree(llinks);
	  if (mDBGOut) {
	    dumpABTracksDebugTree(llinks);
	  } 
	}
      }
    } else if (iCRes > 0) {
      continue; // TPC track precedes the interaction (means orphan track?), no need to check it
    } else {
      LOG(INFO) << "All interaction candidates precede track " << itr << " [" << tTPC.timeBins.min() << ":" << tTPC.timeBins.max() << "]";
      break; // all interaction candidates precede TPC track
    }
  }
  buildABCluster2TracksLinks();
}

//______________________________________________
bool MatchTPCITS::runAfterBurner(int tpcWID, int iCStart, int iCEnd)
{
  // Try to match TPC tracks to ITS clusters, assuming that it comes from interaction candidate in the range [iCStart:iCEnd)
  // The track is already propagated to the outer R of the outermost layer

  LOG(INFO) << "AfterBurner for TPC track " << tpcWID << " with int.candidates " << iCStart << " " << iCEnd;

  auto& tTPC = mTPCWork[tpcWID];
  auto& abTrackLinksList = createABTrackLinksList(tpcWID);

  const int maxMissed = 0;

  for (int iCC = iCStart; iCC < iCEnd; iCC++) {
    const auto& iCCand = mInteractions[iCC];
    int topLinkID = registerABTrackLink(abTrackLinksList, tTPC, iCC, NITSLayers, tpcWID, MinusTen); // add track copy as a link on N+1 layer
    if (topLinkID == MinusOne) {
      continue; // link to be discarded, RS: do we need this for the fake layer?
    }
    auto& topLink = mABTrackLinks[topLinkID];

    correctTPCTrack(topLink, tTPC, iCCand); // correct track for assumed Z location calibration
    /*
    // tmp
    LOG(INFO) << "Check track TPC mtc=" << tTPC.matchID << " int.cand. " << iCC
              << " [" << tTPC.timeBins.min() << ":" << tTPC.timeBins.max() << "] for interaction "
              << " [" << iCCand.timeBins.min() << ":" << iCCand.timeBins.max() << "]";
    */
    if (std::abs(topLink.getZ()) > mITSFiducialZCut) { // we can discard this seed
      topLink.disable();
    }
  }
  int seedLrOK = NITSLayers;
  for (int ilr = NITSLayers; ilr > 0; ilr--) {
    int nextLinkID = abTrackLinksList.firstInLr[ilr];
    while (nextLinkID > MinusOne) {
      if (!mABTrackLinks[nextLinkID].isDisabled()) {
        if ( checkABSeedFromLr(ilr, nextLinkID, abTrackLinksList) ) {
	  if (seedLrOK>ilr) { // flag lowest layer from which the seed was prolongated
	    seedLrOK = ilr; 
	  }
	}
      }
      nextLinkID = mABTrackLinks[nextLinkID].nextOnLr;
    }
  }
  // disable link-list if neiher of seeds reached highest requested layer
  if (seedLrOK-mABRequireToReachLayer>1) {
    mABTrackLinksList.pop_back();
    tTPC.matchID = MinusTen;
    return false;
  }
  
  return true;
}

//______________________________________________
int MatchTPCITS::checkABSeedFromLr(int lrSeed, int seedID, ABTrackLinksList& llist)
{
  // check seed isd on layer lrSeed for prolongation to next layer
  int lrTgt = lrSeed - 1;
  auto& seedLink = mABTrackLinks[seedID];
  o2::track::TrackParCov seed(seedLink); // operate with copy
  auto propagator = o2::base::Propagator::Instance();
  float xTgt;
  const auto& lr = mRGHelper.layers[lrTgt];
  if (!seed.getXatLabR(lr.rRange.max(), xTgt, propagator->getNominalBz(), o2::track::DirInward) ||
      !propagator->PropagateToXBxByBz(seed, xTgt, o2::constants::physics::MassPionCharged, MaxSnp, 2., mUseMatCorrFlag)) {
    return 0;
  }
  auto icCandID = mABTrackLinks[seedID].icCandID;

  // fetch cluster reference object for the ITS ROF corresponding to interaction candidate
  const auto& clRefs = *static_cast<const ITSChipClustersRefs*>(mInteractions[icCandID].clRefPtr);
  const float nSigmaZ = 5., nSigmaY = 5.;
  const float YErr2Extra = 0.1 * 0.1;
  constexpr int NZSpan = 3;
  float sna, csa;                                     // circle parameters for B ON data
  float zDRStep = -seed.getTgl() * lr.rRange.delta(); // approximate Z span when going from layer rMin to rMax
  float errZ = std::sqrt(seed.getSigmaZ2());
  if (lr.zRange.isOutside(seed.getZ(), nSigmaZ * errZ + std::abs(zDRStep))) {
    // printf("Lr %d missed by Z = %.2f + %.3f\n", lrTgt, seed.getZ(), nSigmaZ * errZ + std::abs(zDRStep)); // tmp
    return 0;
  }
  std::vector<int> chipSelClusters; // preliminary cluster candidates //RS TODO do we keep this local / consider array instead of vector
  o2::utils::CircleXY trcCircle;
  o2::utils::IntervalXY trcLinPar; // line parameters fpr B OFF data
  // approximate errors
  float errY = std::sqrt(seed.getSigmaY2() + YErr2Extra), errYFrac = errY * mRGHelper.ladderWidthInv(), errPhi = errY * lr.rInv;
  if (mFieldON) {
    seed.getCircleParams(propagator->getNominalBz(), trcCircle, sna, csa);
  } else {
    seed.getLineParams(trcLinPar, sna, csa);
  }
  float xCurr, yCurr;
  o2::utils::rotateZ(seed.getX(), seed.getY(), xCurr, yCurr, sna, csa);
  float phi = std::atan2(yCurr, xCurr);
  // find approximate ladder and chip_in_ladder corresponding to this track extrapolation
  int nLad2Check = 0, ladIDguess = lr.getLadderID(phi), chipIDguess = lr.getChipID(seed.getZ() + 0.5 * zDRStep);
  std::array<int, MaxLadderCand> lad2Check;
  nLad2Check = mFieldON ? findLaddersToCheckBOn(lrTgt, ladIDguess, trcCircle, errYFrac, lad2Check) : findLaddersToCheckBOff(lrTgt, ladIDguess, trcLinPar, errYFrac, lad2Check);

  const auto& tTPC = mTPCWork[llist.trackID];                        // tmp
  auto lblTrc = mTPCTrkLabels->getLabels(tTPC.source.getIndex())[0]; // tmp
  for (int ilad = nLad2Check; ilad--;) {
    int ladID = lad2Check[ilad];
    const auto& lad = lr.ladders[ladID];

    // we assume that close chips on the same ladder with have close xyEdges, so it is enough to calculate track-chip crossing
    // coordinates xCross,yCross,zCross for this central chipIDguess, although we are going to check also neighbours
    float t = 1e9, xCross, yCross;
    const auto& chipC = lad.chips[chipIDguess];
    bool res = mFieldON ? chipC.xyEdges.circleCrossParam(trcCircle, t) : chipC.xyEdges.lineCrossParam(trcLinPar, t);
    chipC.xyEdges.eval(t, xCross, yCross);
    float dx = xCross - xCurr, dy = yCross - yCurr, dst2 = dx * dx + dy * dy, dst = sqrtf(dst2);
    // Z-step sign depends on radius decreasing or increasing during the propagation
    float zCross = seed.getZ() + seed.getTgl() * (dst2 < 2 * (dx * xCurr + dy * yCurr) ? dst : -dst);

    for (int ich = -1; ich < 2; ich++) {
      int chipID = chipIDguess + ich;
      if (chipID < 0 || chipID >= lad.chips.size()) {
        continue;
      }
      const auto& chip = lad.chips[chipID];
      if (chip.zRange.isOutside(zCross, nSigmaZ * errZ)) {
        continue;
      }
      int chipGID = chip.id;
      const auto& clRange = clRefs.chipRefs[chipGID];
      if (!clRange.getEntries()) {
        continue;
      }
      /*
      // tmp
      printf("Lr %d #%d/%d LadID: %d (phi:%+d) ChipID: %d [%d Ncl: %d from %d] (rRhi:%d Z:%+d[%+.1f:%+.1f]) | %+.3f %+.3f -> %+.3f %+.3f %+.3f (zErr: %.3f)\n",
             lrTgt, ilad, ich, ladID, lad.isPhiOutside(phi, errPhi), chipID,
             chipGID, clRange.getEntries(), clRange.getFirstEntry(),
             chip.xyEdges.seenByCircle(trcCircle, errYFrac), chip.zRange.isOutside(zCross, 3 * errZ), chip.zRange.min(), chip.zRange.max(),
             xCurr, yCurr, xCross, yCross, zCross, errZ);
      */
      // track Y error in chip frame
      float errYcalp = errY * (csa * chipC.csAlp + sna * chipC.snAlp); // sigY_rotate(from alpha0 to alpha1) = sigY * cos(alpha1 - alpha0);
      float tolerZ = errZ * nSigmaZ, tolerY = errYcalp * nSigmaY;
      float yTrack = -xCross * chipC.snAlp + yCross * chipC.csAlp; // track-chip crossing Y in chip frame
      // select candidate clusters for this chip
      if (!preselectChipClusters(chipSelClusters, clRange, clRefs, yTrack, zCross, tolerY, tolerZ, lblTrc)) {
        continue;
      }
      o2::track::TrackParCov trcLC = seed;
      if (!trcLC.rotate(chipC.alp) || !trcLC.propagateTo(chipC.xRef, propagator->getNominalBz())) {
        LOG(INFO) << " failed to rotate to alpha=" << chipC.alp << " or prop to X=" << chipC.xRef;
        trcLC.print();
        continue;
      }
      int cntc = 0;
      for (auto clID : chipSelClusters) {
        const auto& cls = (*mITSClustersArrayInp)[clID];
        auto chi2 = trcLC.getPredictedChi2(cls);
	/*
        const auto lab = mITSClsLabels->getLabels(clID)[0];                                           // tmp
        LOG(INFO) << "cl " << cntc++ << "ClLbl:" << lab << " TrcLbl" << lblTrc << " chi2 = " << chi2; // tmp
	*/
        if (chi2 > mCutABTrack2ClChi2) {
          continue;
        }
        int lnkID = registerABTrackLink(llist, trcLC, icCandID, lrTgt, seedID, clID); // add new link with track copy
        if (lnkID > MinusOne) {
          auto& link = mABTrackLinks[lnkID];
#ifdef _ALLOW_DEBUG_AB_
	  link.seed = link;
#endif
          link.update(cls);
          link.chi2 = chi2 + mABTrackLinks[seedID].chi2; // don't use seedLink since it may be changed are reallocation
          mABTrackLinks[seedID].nDaughters++; // idem, don't use seedLink.nDaughters++;
        }
      }
    }
  }
  return seedLink.nDaughters;
}

//______________________________________________
int MatchTPCITS::findLaddersToCheckBOn(int ilr, int lad0, const o2::utils::CircleXY& circle, float errYFrac,
                                       std::array<int, MatchTPCITS::MaxLadderCand>& lad2Check) const
{
  // check if ladder lad0 and at most +-MaxUpDnLadders around it are compatible with circular track of
  // r^2 = r2 and centered at xC,yC
  const auto& lr = mRGHelper.layers[ilr];
  int nacc = 0, jmp = 0;
  if (lr.ladders[lad0].xyEdges.seenByCircle(circle, errYFrac)) {
    lad2Check[nacc++] = lad0;
  }
  bool doUp = true, doDn = true;
  while ((doUp || doDn) && jmp++ < MaxUpDnLadders) {
    if (doUp) {
      int ldID = (lad0 + jmp) % lr.nLadders;
      if (lr.ladders[ldID].xyEdges.seenByCircle(circle, errYFrac)) {
        lad2Check[nacc++] = ldID;
      } else {
        doUp = false;
      }
    }
    if (doDn) {
      int ldID = lad0 - jmp;
      if (ldID < 0) {
        ldID += lr.nLadders;
      }
      if (lr.ladders[ldID].xyEdges.seenByCircle(circle, errYFrac)) {
        lad2Check[nacc++] = ldID;
      } else {
        doDn = false;
      }
    }
  }
  return nacc;
}

//______________________________________________
int MatchTPCITS::findLaddersToCheckBOff(int ilr, int lad0, const o2::utils::IntervalXY& trcLinPar, float errYFrac,
                                        std::array<int, MatchTPCITS::MaxLadderCand>& lad2Check) const
{
  // check if ladder lad0 and at most +-MaxUpDnLadders around it are compatible with linear track

  const auto& lr = mRGHelper.layers[ilr];
  int nacc = 0, jmp = 0;
  if (lr.ladders[lad0].xyEdges.seenByLine(trcLinPar, errYFrac)) {
    lad2Check[nacc++] = lad0;
  }
  bool doUp = true, doDn = true;
  while ((doUp || doDn) && jmp++ < MaxUpDnLadders) {
    if (doUp) {
      int ldID = (lad0 + jmp) % lr.nLadders;
      if (lr.ladders[ldID].xyEdges.seenByLine(trcLinPar, errYFrac)) {
        lad2Check[nacc++] = ldID;
      } else {
        doUp = false;
      }
    }
    if (doDn) {
      int ldID = lad0 - jmp;
      if (ldID < 0) {
        ldID += lr.nLadders;
      }
      if (lr.ladders[ldID].xyEdges.seenByLine(trcLinPar, errYFrac)) {
        lad2Check[nacc++] = ldID;
      } else {
        doDn = false;
      }
    }
  }
  return nacc;
}

//______________________________________________
void MatchTPCITS::buildABCluster2TracksLinks()
{
  // build links from clusters to tracks for afterburner
  int nTrackLinkList = mABTrackLinksList.size();
  for (int ils = 0; ils < nTrackLinkList; ils++) {
    auto& trList = mABTrackLinksList[ils];
    if (trList.trackID<=MinusOne) {
      LOG(ERROR) << "ABTrackLinksList does not point on tracks, impossible"; // tmp
      continue;
    }
    // register all clusters of all seeds starting from the innermost layer
    for (int lr = 0; lr < NITSLayers; lr++) {
      int finalTrackLinkIdx = trList.firstInLr[lr];
      while (finalTrackLinkIdx > MinusOne) { // loop over all links of this layer
        auto& finalTrackLink = mABTrackLinks[finalTrackLinkIdx];
        if (finalTrackLink.nDaughters) {
          finalTrackLinkIdx = finalTrackLink.nextOnLr; // pick next link on the layer
          continue;                                    // at this moment we need to find the end-point of the seed
        }
        // register links for clusters of this seed moving from lowest to upper layer
        int followLinkIdx = finalTrackLinkIdx;
        while (1) { //>> loop over links of the same seed
          const auto& followLink = mABTrackLinks[followLinkIdx];
          int clID = followLink.clID; // in principle, the cluster might be missing on particular layer
          if (clID > MinusOne) {      //>> register cluster usage
            int newClLinkIdx = mABClusterLinks.size();
            auto& newClLink = mABClusterLinks.emplace_back(finalTrackLinkIdx); // create new link

            //>> insert new link in the list of other links for this cluster ordering in track final quality
            int clLinkIdx = mABClusterLinkIndex[clID];
            int prevClLinkIdx = MinusOne;
            while (clLinkIdx > MinusOne) {
              auto& clLink = mABClusterLinks[clLinkIdx];
              const auto& competingTrackLink = mABTrackLinks[clLink.linkedABTrack];
              if (isBetter(finalTrackLink, competingTrackLink)) {
                newClLink.nextABClusterLink = clLinkIdx;
                break;
              }
              prevClLinkIdx = clLinkIdx; // check next link
              clLinkIdx = clLink.nextABClusterLink;
            }
            if (prevClLinkIdx > MinusOne) { // new link is not the best (1st) one, register it in its predecessor
              mABClusterLinks[prevClLinkIdx].nextABClusterLink = newClLinkIdx;
            } else { // new link is the 1st one, register it in the mABClusterLinkIndex
              mABClusterLinkIndex[clID] = newClLinkIdx;
            }
            //<< insert new link in the list of other links for this cluster ordering in track final quality

          }                                   //<< register cluster usage
          else if (followLink.isDummyTop()) { // we reached dummy seed on the dummy layer above the last ITS layer
            break;
          }

          followLinkIdx = followLink.parentID; // go upward
        }                                      //>> loop over links of the same seed

        finalTrackLinkIdx = finalTrackLink.nextOnLr; // pick next link on the layer
      }                                              // loop over all final seeds of this layer
    }
  }
  printABClusterUsage();
}

//______________________________________________
int MatchTPCITS::registerABTrackLink(ABTrackLinksList& llist, const o2::track::TrackParCov& src, int ic, int lr, int parentID, int clID)
{
  // registers new ABLink on the layer, assigning provided kinematics. The link will be registered in a
  // way preserving the quality ordering of the links on the layer
  int lnkID = mABTrackLinks.size();
  if (llist.firstInLr[lr] == MinusOne) { // no links on this layer yet
    llist.firstInLr[lr] = lnkID;
    mABTrackLinks.emplace_back(src, ic, parentID, clID);
    return lnkID;
  }
  // add new link sorting links of this layer in quality

  int count = 0, nextID = llist.firstInLr[lr], topID = MinusOne;
  do {
    auto& nextLink = mABTrackLinks[nextID];
    count++;
    if (isBetter(src, nextLink)) {      // need to insert new link before nexLink
      if (count < mMaxABLinksOnLayer) { // will insert in front of nextID
        auto& newLnk = mABTrackLinks.emplace_back(src, ic, parentID, clID);
        newLnk.nextOnLr = nextID; // point to the next one
        if (topID > MinusOne) {
          mABTrackLinks[topID].nextOnLr = lnkID; // point from previous one
        }
        return lnkID;
      } else {                                     // max number of candidates reached, will overwrite the last one
        ((o2::track::TrackParCov&)nextLink) = src; // NOTE: this makes sense only if the prolongation tree is filled from top to bottom
        return nextID;                             // i.e. there are no links on the lower layers pointing on overwritten one!!!
      }
    }
    topID = nextID;
    nextID = nextLink.nextOnLr;
  } while (nextID > MinusOne);
  // new link is worse than all others, add it only if there is a room to expand
  if (count < mMaxABLinksOnLayer) {
    mABTrackLinks.emplace_back(src, ic, parentID, clID);
    if (topID > MinusOne) {
      mABTrackLinks[topID].nextOnLr = lnkID; // point from previous one
    }
    return lnkID;
  }
  return MinusOne; // link to be ignored
}

//______________________________________________
ABTrackLinksList& MatchTPCITS::createABTrackLinksList(int tpcWID)
{
  // return existing or newly created AB links list for TPC track work copy ID
  auto& tTPC = mTPCWork[tpcWID];
  tTPC.matchID = mABTrackLinksList.size(); // register new list in the TPC track
  return mABTrackLinksList.emplace_back(tpcWID);
}
  
//______________________________________________
float MatchTPCITS::correctTPCTrack(o2::track::TrackParCov& trc, const TrackLocTPC& tTPC, const InteractionCandidate& cand) const
{
  // Correct the track copy trc of the working TPC track tTPC in continuous RO mode for the assumed interaction time
  // return extra uncertainty in Z due to the interaction time incertainty
  // TODO: at the moment, apply simple shift, but with Z-dependent calibration we may
  // need to do corrections on TPC cluster level and refit
  auto tpcTrOrig = (*mTPCTracksArrayInp)[tTPC.source.getIndex()];
  if (tpcTrOrig.hasBothSidesClusters()) {
    return 0.;
  }
  float timeIC = cand.timeBins.mean(), timeTrc = tpcTrOrig.getTime0() - mNTPCBinsFullDrift;
  // if interaction time precedes the initial assumption on t0 (i.e. timeIC < timeTrc),
  // the track actually was drifting longer, i.e. tracks should be shifted closer to the CE
  float dDrift = (timeIC - timeTrc) * mTPCBin2Z, driftErr = cand.timeBins.delta() * mTPCBin2Z;

  trc.setZ(tTPC.getZ() + (tpcTrOrig.hasASideClustersOnly() ? dDrift : -dDrift));
  trc.setCov(trc.getSigmaZ2() + driftErr * driftErr, o2::track::kSigZ2);
  // tmp
  /*
  printf("Ttrack A=%d: pT:%.1f Ncl:%2d T:%.2f  TIC: %.2f -> Z=%+.2f -> %+.2f +- %.3f [shift = %f]\n",
         tpcTrOrig.hasASideClustersOnly(), trc.getPt(), tpcTrOrig.getNClusterReferences(), timeTrc, timeIC, tTPC.getZ(), trc.getZ(),
         driftErr, (tpcTrOrig.hasASideClustersOnly() ? dDrift : -dDrift));
  */
  return driftErr;
}

//______________________________________________
void MatchTPCITS::fillClustersForAfterBurner(ITSChipClustersRefs& refCont, int rofStart, int nROFs)
{
  // Prepare unused clusters of given ROFs range for matching in the afterburner
  // Note: normally only 1 ROF needs to be filled (nROFs==1 ) unless we want
  // to account for interaction on the boundary of 2 rofs, which then may contribute to both ROFs.
  int first = (*mITSClusterROFRec)[rofStart].getROFEntry().getIndex(), last = first;
  for (int ir = nROFs; ir--;) {
    last += (*mITSClusterROFRec)[rofStart + ir].getNROFEntries();
  }
  refCont.clear();
  auto& idxSort = refCont.clusterID;
  for (int icl = first; icl < last; icl++) {
    if (mABClusterLinkIndex[icl] != MinusTen) { // clusters with MinusOne are used in main matching
      idxSort.push_back(icl);
    }
  }
  // sort in chip, Z
  sort(idxSort.begin(), idxSort.end(), [clusArr = mITSClustersArrayInp](int i, int j) {
    const auto &clI = (*clusArr)[i], &clJ = (*clusArr)[j];
    if (clI.getSensorID() < clJ.getSensorID()) {
      return true;
    }
    if (clI.getSensorID() == clJ.getSensorID()) {
      return clI.getZ() < clJ.getZ();
    }
    return false;
  });

  int ncl = idxSort.size();
  int lastSens = -1, nClInSens = 0;
  ClusRange* chipClRefs = nullptr;
  for (int icl = 0; icl < ncl; icl++) {
    const auto& clus = (*mITSClustersArrayInp)[idxSort[icl]];
    int sens = clus.getSensorID();
    if (sens != lastSens) {
      if (chipClRefs) { // finalize chip reference
        chipClRefs->setEntries(nClInSens);
        nClInSens = 0;
      }
      chipClRefs = &refCont.chipRefs[(lastSens = sens)];
      chipClRefs->setFirstEntry(icl);
    }
    nClInSens++;
  }
  if (chipClRefs) {
    chipClRefs->setEntries(nClInSens); // finalize last chip reference
  }
}

//______________________________________________
void MatchTPCITS::selectBestMatchesAB()
{
  ///< loop over After-Burner match records and select the ones with best quality
  LOG(INFO) << "Selecting best AfterBurner matches ";
  int nValidated = 0, iter = 0;

  int nTrackLinkList = mABTrackLinksList.size();

  /*  
  
  do {
    nValidated = 0;
    int ntpc = mTPCWork.size(), nremaining = 0;
    ;
    for (int it = 0; it < ntpc; it++) {
      auto& tTPC = mTPCWork[it];
      if (isDisabledTPC(tTPC) || isValidatedTPC(tTPC)) {
        continue;
      }
      nremaining++;
      if (validateTPCMatch(it)) {
        nValidated++;
        continue;
      }
    }
    printf("iter %d Validated %d of %d remaining matches\n", iter, nValidated, nremaining);
    iter++;
  } while (nValidated);
  */
}

/*
{
  auto rcen2 = xc*xc+yc*yc, rcen = sqrtf(rcen2); // radius^2 of the circle center
  auto rlPrt = rLr + rTra, rlMrt = rLr - rTra; // sum and differenc of layer and track radii
  auto dtPart = (rcen2-rlMrt*rlMrt)*(rcen2-rlPrt*rlPrt);
  if (dtPart>0.) {
    return false; // no intersection
  }

}
*/

//<<============================= AfterBurner for TPC-track / ITS cluster matching ===================<<

#ifdef _ALLOW_DEBUG_TREES_
//______________________________________________
void MatchTPCITS::setDebugFlag(UInt_t flag, bool on)
{
  ///< set debug stream flag
  if (on) {
    mDBGFlags |= flag;
  } else {
    mDBGFlags &= ~flag;
  }
}

//_________________________________________________________
void MatchTPCITS::fillTPCITSmatchTree(int itsID, int tpcID, int rejFlag, float chi2)
{
  ///< fill debug tree for ITS TPC tracks matching check

  mTimerDBG.Start(false);

  auto& trackITS = mITSWork[itsID];
  auto& trackTPC = mTPCWork[tpcID];
  if (chi2 < 0.) { // need to recalculate
    chi2 = getPredictedChi2NoZ(trackITS, trackTPC);
  }
  o2::MCCompLabel lblITS, lblTPC;
  (*mDBGOut) << "match"
             << "chi2Match=" << chi2 << "its=" << trackITS << "tpc=" << trackTPC;
  if (mMCTruthON) {
    lblITS = mITSLblWork[itsID];
    lblTPC = mTPCLblWork[tpcID];
    (*mDBGOut) << "match"
               << "itsLbl=" << lblITS << "tpcLbl=" << lblTPC;
  }
  (*mDBGOut) << "match"
             << "rejFlag=" << rejFlag << "\n";

  mTimerDBG.Stop();
}

//______________________________________________
void MatchTPCITS::dumpWinnerMatches()
{
  ///< write winner matches into debug tree

  mTimerDBG.Start(false);

  LOG(INFO) << "Dumping debug tree for winner matches";
  for (int iits = 0; iits < int(mITSWork.size()); iits++) {
    auto& tITS = mITSWork[iits];
    if (isDisabledITS(tITS)) {
      continue;
    }
    auto& itsMatchRec = mMatchRecordsITS[tITS.matchID];
    int itpc = itsMatchRec.partnerID;
    auto& tTPC = mTPCWork[itpc];

    (*mDBGOut) << "matchWin"
               << "chi2Match=" << itsMatchRec.chi2 << "chi2Refit=" << mWinnerChi2Refit[iits] << "its=" << tITS
               << "tpc=" << tTPC;

    o2::MCCompLabel lblITS, lblTPC;
    if (mMCTruthON) {
      lblITS = mITSLblWork[iits];
      lblTPC = mTPCLblWork[itpc];
      (*mDBGOut) << "matchWin"
                 << "itsLbl=" << lblITS << "tpcLbl=" << lblTPC;
    }
    (*mDBGOut) << "matchWin"
               << "\n";
  }
  mTimerDBG.Stop();
}

//_________________________________________________________
void MatchTPCITS::setUseMatCorrFlag(int f)
{
  ///< set flag to select material correction type
  mUseMatCorrFlag = f;
  if (f < o2::base::Propagator::USEMatCorrNONE || f > o2::base::Propagator::USEMatCorrLUT) {
    LOG(FATAL) << "Invalid MatCorr flag" << f;
  }
}

#endif
