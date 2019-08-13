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
#include "ITSBase/GeometryTGeo.h"

#include "DetectorsBase/Propagator.h"
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

  mTimerTot.Start();

  clear();

  if (!prepareITSTracks() || !prepareTPCTracks() || !prepareFITInfo()) {
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

  buildMatch2TrackTables();

  selectBestMatches();

  refitWinners();

#ifdef _ALLOW_DEBUG_TREES_
  if (mDBGOut && isDebugFlag(WinnerMatchesTree)) {
    dumpWinnerMatches();
  }
  mDBGOut.reset();
#endif

  mTimerTot.Stop();

  printf("Timing:\n");
  printf("Total:        ");
  mTimerTot.Print();
  printf("Data IO:      ");
  mTimerIO.Print();
  printf("Registration: ");
  mTimerReg.Print();
  printf("Refits      : ");
  mTimerRefit.Print();
  printf("DBG trees:    ");
  mTimerDBG.Print();
}

//______________________________________________
void MatchTPCITS::clear()
{
  ///< clear results of previous TF reconstruction
  mMatchRecordsTPC.clear();
  mMatchRecordsITS.clear();
  mMatchesTPC.clear();
  mMatchesITS.clear();
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

  clear();

  mInitDone = true;

  {
    mTimerTot.Stop();
    mTimerIO.Stop();
    mTimerDBG.Stop();
    mTimerReg.Stop();
    mTimerRefit.Stop();
    mTimerTot.Reset();
    mTimerIO.Reset();
    mTimerDBG.Reset();
    mTimerReg.Reset();
    mTimerRefit.Reset();
  }

  print();
}

//______________________________________________
void MatchTPCITS::selectBestMatches()
{
  ///< loop over match records and select the ones with best chi2
  LOG(INFO) << "Selecting best matches for " << mMatchesTPC.size() << " TPC tracks";

  int nValidated = 0, iter = 0;

  do {
    nValidated = 0;
    for (int imtTPC = 0; imtTPC < mMatchesTPC.size(); imtTPC++) {
      auto& tpcMatch = mMatchesTPC[imtTPC];
      if (isDisabledTPC(tpcMatch))
        continue;
      if (isValidatedTPC(tpcMatch))
        continue;
      if (validateTPCMatch(imtTPC)) {
        nValidated++;
        continue;
      }
    }
    printf("iter %d Validated %d of %d\n", iter, nValidated, int(mMatchesTPC.size()));
    iter++;
  } while (nValidated);
}

//______________________________________________
bool MatchTPCITS::validateTPCMatch(int mtID)
{
  auto& tpcMatch = mMatchesTPC[mtID];
  auto& rcTPC = mMatchRecordsTPC[tpcMatch.first]; // best TPC->ITS match
  if (rcTPC.nextRecID == Validated)
    return false; // RS do we need this
  // check if it is consistent with corresponding ITS->TPC match
  auto& itsMatch = mMatchesITS[rcTPC.matchID];    // matchCand of partner ITS track
  auto& rcITS = mMatchRecordsITS[itsMatch.first]; // best ITS->TPC match
  if (rcTPC.nextRecID == Validated)
    return false;              // RS do we need this ?
  if (rcITS.matchID == mtID) { // is best matching TPC track for this ITS track actually mtID?

    // unlink winner TPC track from all ITS candidates except winning one
    int nextTPC = rcTPC.nextRecID;
    while (nextTPC > MinusOne) {
      auto& rcTPCrem = mMatchRecordsTPC[nextTPC];
      removeTPCfromITS(mtID, rcTPCrem.matchID); // remove references on mtID from ITS match=rcTPCrem.matchID
      nextTPC = rcTPCrem.nextRecID;
    }
    rcTPC.nextRecID = Validated;
    int itsWinID = rcTPC.matchID;

    // unlink winner ITS match from all TPC matches using it
    int nextITS = rcITS.nextRecID;
    while (nextITS > MinusOne) {
      auto& rcITSrem = mMatchRecordsITS[nextITS];
      removeITSfromTPC(itsWinID, rcITSrem.matchID); // remove references on itsWinID from TPC match=rcITSrem.matchID
      nextITS = rcITSrem.nextRecID;
    }
    rcITS.nextRecID = Validated;
    return true;
  }
  return false;
}

//______________________________________________
int MatchTPCITS::getNMatchRecordsTPC(const matchCand& tpcMatch) const
{
  ///< get number of matching records for TPC track referring to this matchCand
  int count = 0, recID = tpcMatch.first;
  while (recID > MinusOne) {
    recID = mMatchRecordsTPC[recID].nextRecID;
    count++;
  }
  return count;
}

//______________________________________________
int MatchTPCITS::getNMatchRecordsITS(const matchCand& itsMatch) const
{
  ///< get number of matching records for ITS track referring to this matchCand
  int count = 0, recID = itsMatch.first;
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
  mTreeITSClusters->SetBranchAddress(mITSClusterBranchName.data(), &mITSClustersArrayInp);
  LOG(INFO) << "Attached ITS clusters " << mITSClusterBranchName << " branch with " << mTreeITSClusters->GetEntries()
            << " entries";

  if (!mTreeITSClusterROFRec->GetBranch(mITSClusterROFRecBranchName.data())) {
    LOG(FATAL) << "Did not find ITS clusters ROFRecords branch " << mITSClusterROFRecBranchName << " in the input tree";
  }
  mTreeITSClusterROFRec->SetBranchAddress(mITSClusterROFRecBranchName.data(), &mITSClusterROFRec);
  LOG(INFO) << "Attached ITS clusters ROFRec " << mITSClusterROFRecBranchName << " branch with "
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
  if (mTreeITSTracks->GetBranch(mITSMCTruthBranchName.data())) {
    mTreeITSTracks->SetBranchAddress(mITSMCTruthBranchName.data(), &mITSTrkLabels);
    LOG(INFO) << "Found ITS Track MCLabels branch " << mITSMCTruthBranchName;
  }
  // is there MC info available ?
  if (mTreeTPCTracks->GetBranch(mTPCMCTruthBranchName.data())) {
    mTreeTPCTracks->SetBranchAddress(mTPCMCTruthBranchName.data(), &mTPCTrkLabels);
    LOG(INFO) << "Found TPC Track MCLabels branch " << mTPCMCTruthBranchName;
  }

  mMCTruthON = (mITSTrkLabels && mTPCTrkLabels);
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
  mMatchesTPC.reserve(mMatchesTPC.size() + ntr);
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
      return (trcA.timeBins.tmax - trcB.timeBins.tmax) < 0.;
    });

    // build array of 1st entries with tmax corresponding to each ITS ROF (or trigger)
    float tmax = mTPCWork[indexCache.back()].timeBins.tmax;
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

      while (trc.timeBins.tmax < mITSROFTimes[itsROF].tmin && itsROF < nITSROFs) {
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

  if (maxTimeBin < mITSROFTimes.back().tmax) {
    maxTimeBin = mITSROFTimes.back().tmax;
  }
  int nb = int(maxTimeBin) + 1;
  mITSROFofTPCBin.resize(nb, -1);
  int itsROF = 0;
  for (int ib = 0; ib < nb; ib++) {
    while (ib < mITSROFTimes[itsROF].tmin && itsROF < nITSROFs) {
      itsROF++;
    }
    mITSROFofTPCBin[ib] = itsROF;
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

  mMatchesITS.clear();

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

  for (int sec = o2::constants::math::NSectors; sec--;) {
    mITSSectIndexCache[sec].clear();
    mITSTimeBinStart[sec].clear();
    mITSTimeBinStart[sec].resize(nROFs, -1); // start of ITS work tracks in every sector
  }
  setStartIR((*mITSTrackROFRec)[0].getBCData());
  for (int irof = 0; irof < nROFs; irof++) {
    const auto& rofRec = (*mITSTrackROFRec)[irof];
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
      if (trcOrig.getParamOut().getX() < 1.) {
        continue; // backward refit failed
      }
      int nWorkTracks = mITSWork.size();
      // working copy of outer track param
      mITSWork.emplace_back(static_cast<const o2::track::TrackParCov&>(trcOrig.getParamOut()), rEntry, it);
      auto& trc = mITSWork.back();

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
      float tgp = trc.getSnp();
      tgp /= std::sqrt((1.f - tgp) * (1.f + tgp)); // tan of track direction XY

      // sector up
      float dy2Up = (YMaxAtXMatchingRef - trc.getY()) / (tgp + Tan70);
      if ((dy2Up * dy2Up * Cos70I2) < mSectEdgeMargin2) { // need to check this track for matching in sector up
        addTrackCloneForNeighbourSector(trc, sector < (o2::constants::math::NSectors - 1) ? sector + 1 : 0);
      }
      // sector down
      float dy2Dn = (YMaxAtXMatchingRef + trc.getY()) / (tgp - Tan70);
      if ((dy2Dn * dy2Dn * Cos70I2) < mSectEdgeMargin2) { // need to check this track for matching in sector down
        addTrackCloneForNeighbourSector(trc, sector > 1 ? sector - 1 : o2::constants::math::NSectors - 1);
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
  mMatchesITS.reserve(mITSWork.size());
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
    int itsROBin = tpcTimeBin2ITSROFrame(trefTPC.timeBins.tmax - maxTDriftSafe);
    if (itsROBin >= int(tbinStartITS.size())) { // time of TPC track exceeds the max time of ITS in the cache
      break;
    }
    int iits0 = tbinStartITS[itsROBin];
    nCheckTPCControl++;
    for (auto iits = iits0; iits < nTracksITS; iits++) {
      auto& trefITS = mITSWork[cacheITS[iits]];
      const auto& timeITS = mITSROFTimes[trefITS.roFrame];
      // compare if the ITS and TPC tracks may overlap in time
      if (trefTPC.timeBins.tmax < timeITS.tmin) {
        // since TPC tracks are sorted in timeMax and ITS tracks are sorted in timeMin
        // all following ITS tracks also will not match
        break;
      }
      if (trefTPC.timeBins.tmin > timeITS.tmax) { // its bracket is fully before TPC bracket
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
      mTimerReg.Start(false);
      registerMatchRecordTPC(trefITS, trefTPC, chi2); // register matching candidate
      mTimerReg.Stop();
      nMatchesControl++;
    }
  }

  LOG(INFO) << "Match sector " << sec << " N tracks TPC:" << nTracksTPC << " ITS:" << nTracksITS
            << " N TPC tracks checked: " << nCheckTPCControl << " (starting from " << idxMinTPC
            << "), checks: " << nCheckITSControl << ", matches:" << nMatchesControl;
}

//______________________________________________
void MatchTPCITS::suppressMatchRecordITS(int matchITSID, int matchTPCID)
{
  ///< suppress the reference on the matchCand with id=matchTPCID in
  ///< the list of matches recorded by for matchCand with id matchITSID
  auto& itsMatch = mMatchesITS[matchITSID];
  int topID = MinusOne, recordID = itsMatch.first; // 1st entry in mMatchRecordsITS
  while (recordID > MinusOne) {                    // navigate over records for given ITS track
    if (mMatchRecordsITS[recordID].matchID == matchTPCID) {
      // unlink this record, connecting its child to its parrent
      if (topID < 0) {
        itsMatch.first = mMatchRecordsITS[recordID].nextRecID;
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
bool MatchTPCITS::registerMatchRecordTPC(TrackLocITS& tITS, TrackLocTPC& tTPC, float chi2)
{
  ///< record matching candidate, making sure that number of ITS candidates per TPC track, sorted
  ///< in matching chi2 does not exceed allowed number

  auto& mtcTPC = getTPCMatchEntry(tTPC);               // get matchCand structure of this TPC track, create if none
  int nextID = mtcTPC.first;                           // get 1st matchRecord this matchCand refers to
  if (nextID < 0) {                                    // no matches yet, just add new record
    registerMatchRecordITS(tITS, tTPC.matchID, chi2);  // register matchCand entry in the ITS records
    mtcTPC.first = mMatchRecordsTPC.size();            // new record will be added in the end
    mMatchRecordsTPC.emplace_back(tITS.matchID, chi2); // create new record with empty reference on next match
    return true;
  }

  int count = 0, topID = MinusOne;
  do {
    auto& nextMatchRec = mMatchRecordsTPC[nextID];
    count++;
    if (chi2 < nextMatchRec.chi2) { // need to insert new record before nextMatchRec?
      if (count < mMaxMatchCandidates) {
        break; // will insert in front of nextID
      } else { // max number of candidates reached, will overwrite the last one
        nextMatchRec.chi2 = chi2;
        suppressMatchRecordITS(nextMatchRec.matchID, tTPC.matchID); // flag as disabled the overriden ITS match
        registerMatchRecordITS(tITS, tTPC.matchID, chi2);           // register matchCand entry in the ITS records
        nextMatchRec.matchID = tITS.matchID;                        // reuse the record of suppressed ITS match to store better one
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
      topID = mtcTPC.first = mMatchRecordsTPC.size();                      // register new record as top one
    } else {                                                               // there are better candidates
      topID = mMatchRecordsTPC[topID].nextRecID = mMatchRecordsTPC.size(); // register to his parent
    }
    // nextID==-1 will mean that the while loop run over all candidates->the new one is the worst (goes to the end)
    registerMatchRecordITS(tITS, tTPC.matchID, chi2);          // register matchCand entry in the ITS records
    mMatchRecordsTPC.emplace_back(tITS.matchID, chi2, nextID); // create new record with empty reference on next match
    // make sure that after addition the number of candidates don't exceed allowed number
    count++;
    while (nextID > MinusOne) {
      if (count > mMaxMatchCandidates) {
        suppressMatchRecordITS(mMatchRecordsTPC[nextID].matchID, tTPC.matchID);
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
void MatchTPCITS::registerMatchRecordITS(TrackLocITS& tITS, int matchTPCID, float chi2)
{
  ///< register TPC match in ITS match records, ordering then in chi2
  auto& itsMatch = getITSMatchEntry(tITS); // if needed, create new entry
  int nextRecord = itsMatch.first;         // entry of 1st match record in mMatchRecordsITS
  int idnew = mMatchRecordsITS.size();
  mMatchRecordsITS.emplace_back(matchTPCID, chi2); // associate index of matchCand with this record
  if (nextRecord < 0) {                            // this is the 1st match for this TPC track
    itsMatch.first = idnew;
    return;
  }
  // there are other matches for this ITS track, insert the new record preserving chi2 order
  // navigate till last record or the one with worse chi2
  int topID = MinusOne;
  auto& newRecord = mMatchRecordsITS.back();
  do {
    auto& recITS = mMatchRecordsITS[nextRecord];
    if (chi2 < recITS.chi2) {           // insert before this one
      newRecord.nextRecID = nextRecord; // new one will refer to old one it overtook
      if (topID < 0) {
        itsMatch.first = idnew; // the new one is the best match, the matchCand will refer to it
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

  printf("\n\nPrinting all %zu TPC -> ITS matches\n", mMatchesTPC.size());
  for (const auto& tpcMatch : mMatchesTPC) {
    printf("*** trackTPC# %6d(%4d) : Ncand = %d\n", tpcMatch.source.getIndex(), tpcMatch.source.getEvent(),
           getNMatchRecordsTPC(tpcMatch));
    int count = 0, recID = tpcMatch.first;
    while (recID > MinusOne) {
      const auto& rcTPC = mMatchRecordsTPC[recID];
      const auto& itsMatch = mMatchesITS[rcTPC.matchID];
      printf("  * cand %2d : ITS track %6d(%4d) Chi2: %.2f\n", count, itsMatch.source.getIndex(),
             itsMatch.source.getEvent(), rcTPC.chi2);
      count++;
      recID = rcTPC.nextRecID;
    }
  }
}

//______________________________________________
void MatchTPCITS::printCandidatesITS() const
{
  ///< print mathing records

  printf("\n\nPrinting all %zu ITS -> TPC matches\n", mMatchesITS.size());
  for (const auto& itsMatch : mMatchesITS) {
    printf("*** trackITS# %6d(%4d) : Ncand = %d\n", itsMatch.source.getIndex(), itsMatch.source.getEvent(),
           getNMatchRecordsITS(itsMatch));
    int count = 0, recID = itsMatch.first;
    while (recID > MinusOne) {
      const auto& rcITS = mMatchRecordsITS[recID];
      const auto& tpcMatch = mMatchesTPC[rcITS.matchID];
      printf("  * cand %2d : TPC track %6d(%4d) Chi2: %.2f\n", count, tpcMatch.source.getIndex(),
             tpcMatch.source.getEvent(), rcITS.chi2);
      count++;
      recID = rcITS.nextRecID;
    }
  }
}

//______________________________________________
void MatchTPCITS::buildMatch2TrackTables()
{
  ///< refer each match to corrsponding track
  mITSMatch2Track.clear();
  mITSMatch2Track.resize(mMatchesITS.size(), MinusOne);
  for (int i = 0; i < int(mITSWork.size()); i++) {
    const auto& its = mITSWork[i];
    if (its.matchID > MinusOne) {
      mITSMatch2Track[its.matchID] = i;
    }
  }

  mTPCMatch2Track.clear();
  mTPCMatch2Track.resize(mMatchesTPC.size(), MinusOne);
  for (int i = 0; i < int(mTPCWork.size()); i++) {
    const auto& tpc = mTPCWork[i];
    if (tpc.matchID > MinusOne) {
      mTPCMatch2Track[tpc.matchID] = i;
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
void MatchTPCITS::addTrackCloneForNeighbourSector(const TrackLocITS& src, int sector)
{
  // add clone of the src ITS track cashe, propagate it to ref.X in requested sector
  // and register its index in the sector cache. Used for ITS tracks which are so close
  // to their setctor edge that their matching should be checked also in the neighbouring sector

  mITSWork.push_back(src); // clone the track defined in given sector
  auto& trc = mITSWork.back();
  if (trc.rotate(o2::utils::Sector2Angle(sector)) &&
      o2::base::Propagator::Instance()->PropagateToXBxByBz(trc, XMatchingRef, o2::constants::physics::MassPionCharged, MaxSnp,
                                                           2., 0)) {
    // TODO: use faster prop here, no 3d field, materials
    mITSSectIndexCache[sector].push_back(mITSWork.size() - 1); // register track CLONE
    if (mMCTruthON) {
      mITSLblWork.emplace_back(mITSTrkLabels->getLabels(src.source.getIndex())[0]);
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
                                                              MaxSnp, 2., 1)) {
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
void MatchTPCITS::refitWinners()
{
  ///< refit winning tracks

  mTimerRefit.Start(false);

  LOG(INFO) << "Refitting winner matches";
  mWinnerChi2Refit.resize(mITSWork.size(), -1.f);
  mCurrITSClustersTreeEntry = -1;
  for (int iITS = 0; iITS < mITSWork.size(); iITS++) {
    if (!refitTrackTPCITS(iITS)) {
      continue;
    }
    mWinnerChi2Refit[iITS] = mMatchedTracks.back().getChi2Refit();
  }
  // flush last tracks
  if (mMatchedTracks.size() && mOutputTree) {
    mOutputTree->Fill();
  }
  mTimerRefit.Stop();
}

//______________________________________________
bool MatchTPCITS::refitTrackTPCITS(int iITS)
{
  ///< refit in inward direction the pair of TPC and ITS tracks

  const float maxStep = 2.f; // max propagation step (TODO: tune)
  const int matCorr = 1;     // material correction method

  const auto& tITS = mITSWork[iITS];
  if (tITS.matchID < 0 || isDisabledITS(mMatchesITS[tITS.matchID])) {
    return false; // no match
  }
  const auto& itsMatch = mMatchesITS[tITS.matchID];
  const auto& itsMatchRec = mMatchRecordsITS[itsMatch.first];
  int iTPC = mTPCMatch2Track[itsMatchRec.matchID];
  const auto& tTPC = mTPCWork[iTPC];

  if (!mDPLIO) {
    loadITSClustersChunk(tITS.source.getEvent());
    loadITSTracksChunk(tITS.source.getEvent());
    loadTPCTracksChunk(tTPC.source.getEvent());
    loadTPCClustersChunk(tTPC.source.getEvent());
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
                                  MaxSnp, maxStep, matCorr, &trfit.getLTIntegralOut())) {
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
                                  maxStep, matCorr, &trfit.getLTIntegralOut())) {
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
    if (!propagator->PropagateToXBxByBz(tracOut, clsX, o2::constants::physics::MassPionCharged, MaxSnp, 10., 1, &trfit.getLTIntegralOut())) {
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
                                          10., 0, &trfit.getLTIntegralOut())) { // no material correction!
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
                                   10., 1, &trfit.getLTIntegralOut());

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

//________________________________________________________
void MatchTPCITS::loadITSClustersChunk(int chunk)
{
  // load single entry from ITS clusters tree
  if (mCurrITSClustersTreeEntry != chunk) {
    mTimerIO.Start(false);
    mTreeITSClusters->GetEntry(mCurrITSClustersTreeEntry = chunk);
    mTimerIO.Stop();
  }
}

//________________________________________________________
void MatchTPCITS::loadTPCClustersChunk(int chunk)
{
  // load single entry from ITS clusters tree
  if (mCurrTPCClustersTreeEntry != chunk) {
    mTimerIO.Start(false);
    mTPCClusterReader->read(mCurrTPCClustersTreeEntry = chunk);
    mTPCClusterReader->fillIndex(*mTPCClusterIdxStructOwn.get(), mTPCClusterBufferOwn, mTPCClusterMCBufferOwn);
    mTPCClusterIdxStruct = mTPCClusterIdxStructOwn.get();
    mTimerIO.Stop();
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
    auto& its = mITSWork[iits];
    if (its.matchID < 0 || isDisabledITS(mMatchesITS[its.matchID])) {
      continue;
    }
    auto& itsMatch = mMatchesITS[its.matchID];
    auto& itsMatchRec = mMatchRecordsITS[itsMatch.first];
    int itpc = mTPCMatch2Track[itsMatchRec.matchID];
    auto& tpc = mTPCWork[itpc];

    (*mDBGOut) << "matchWin"
               << "chi2Match=" << itsMatchRec.chi2 << "chi2Refit=" << mWinnerChi2Refit[iits] << "its=" << its
               << "tpc=" << tpc;

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

#endif
