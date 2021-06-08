// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <TSystem.h>
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
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "ReconstructionDataFormats/Vertex.h"
#include "GlobalTracking/MatchTPCITS.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "DataFormatsTPC/WorkflowHelper.h"

#include "ITStracking/IOUtils.h"

#include "GPUO2Interface.h" // Needed for propper settings in GPUParam.h

using namespace o2::globaltracking;

using MatrixDSym4 = ROOT::Math::SMatrix<double, 4, 4, ROOT::Math::MatRepSym<double, 4>>;
using MatrixD4 = ROOT::Math::SMatrix<double, 4, 4, ROOT::Math::MatRepStd<double, 4>>;
using NAMES = o2::base::NameConf;
using GTrackID = o2::dataformats::GlobalTrackID;

constexpr float MatchTPCITS::XMatchingRef;
constexpr float MatchTPCITS::YMaxAtXMatchingRef;
constexpr float MatchTPCITS::Tan70, MatchTPCITS::Cos70I2, MatchTPCITS::MaxSnp, MatchTPCITS::MaxTgp;

//______________________________________________
void MatchTPCITS::printABTracksTree(const ABTrackLinksList& llist) const
{
  // dump all hypotheses
  if (llist.lowestLayer == NITSLayers) {
    printf("No matches\n");
    return;
  }
  o2::MCCompLabel lblTrc;
  if (mMCTruthON) {
    lblTrc = mTPCTrkLabels[mTPCWork[llist.trackID].sourceID];
  }
  LOG(INFO) << "Matches for track " << llist.trackID << " lowest lr: " << int(llist.lowestLayer) << " " << lblTrc
            << " pT= " << mTPCWork[llist.trackID].getPt();

  auto printHyp = [this, &lblTrc](int nextHyp, int cnt) {
    printf("#%d Lr/IC/Lnk/ClID/Chi2/Chi2Nrm/{MC}:%c ", cnt++, mTPCTrkLabels.size() ? (lblTrc.isFake() ? 'F' : 'C') : '-');
    int nFakeClus = 0, nTotClus = 0, parID = nextHyp; // print particular hypothesis
    while (1) {
      const auto& lnk = mABTrackLinks[parID];
      int mcEv = -1, mcTr = -1;
      if (lnk.clID > MinusOne && mITSClsLabels) {
        nTotClus++;
        const auto lab = mITSClsLabels->getLabels(lnk.clID)[0];
        if (lab.isValid()) {
          mcEv = lab.getEventID();
          mcTr = lab.getTrackID();
          if (mcEv != lblTrc.getEventID() || mcTr != lblTrc.getTrackID()) {
            nFakeClus++;
          }
        } else {
          mcEv = mcTr = -999; // noise
          nFakeClus++;
        }
      } else if (lnk.isDummyTop() && mMCTruthON) { // top layer, use TPC MC lbl
        mcEv = lblTrc.getEventID();
        mcTr = lblTrc.getTrackID();
      }
      printf("[%d/%d/%d/%d/%6.2f/%6.2f/{%d/%d}]", lnk.layerID, lnk.icCandID, parID, lnk.clID, lnk.chi2, lnk.chi2Norm(), mcEv, mcTr);
      if (lnk.isDummyTop()) { // reached dummy seed on the dummy layer above the last ITS layer
        break;
      }
      parID = lnk.parentID;
    }
    printf(" NTot:%d NFakes:%d\n", nTotClus, nFakeClus);
  };

  int cnt = 0; // tmp
  for (int lowest = llist.lowestLayer; lowest <= mParams->requireToReachLayerAB; lowest++) {
    int nextHyp = llist.firstInLr[lowest];
    while (nextHyp > MinusOne) {
      if (mABTrackLinks[nextHyp].nDaughters == 0) { // select only head links, w/o daughters
        printHyp(nextHyp, cnt++);
      }
      nextHyp = mABTrackLinks[nextHyp].nextOnLr;
    }
  }

  if (llist.bestOrdLinkID > MinusOne) { // print best matches list
    int next = llist.bestOrdLinkID;
    LOG(INFO) << "Best matches:";
    int cnt = 0;
    while (next > MinusOne) {
      const auto& lnkOrd = mABBestLinks[next];
      int nextHyp = lnkOrd.trackLinkID;
      printHyp(nextHyp, cnt++);
      next = lnkOrd.nextLinkID;
      refitABTrack(nextHyp);
    }
  }
}

//______________________________________________
void MatchTPCITS::dumpABTracksDebugTree(const ABTrackLinksList& llist)
{
  // dump all hypotheses
  static int entCnt = 0;
  if (llist.lowestLayer == NITSLayers) {
    return;
  }
  LOG(INFO) << "Dump AB Matches for track " << llist.trackID;
  o2::MCCompLabel lblTrc;
  if (mMCTruthON) {
    lblTrc = mTPCTrkLabels[mTPCWork[llist.trackID].sourceID]; // tmp
  }
  int ord = 0;
  for (int lowest = llist.lowestLayer; lowest <= mParams->requireToReachLayerAB; lowest++) {
    int nextHyp = llist.firstInLr[lowest];
    while (nextHyp > MinusOne) {
      if (mABTrackLinks[nextHyp].nDaughters != 0) { // select only head links, w/o daughters
        nextHyp = mABTrackLinks[nextHyp].nextOnLr;
        continue;
      }
      // fill debug AB track
      ABDebugTrack dbgTrack;
      int parID = nextHyp; // print particular hypothesis
      while (1) {
        const auto& lnk = mABTrackLinks[parID];
        if (lnk.clID > MinusOne) {
          auto& dbgLnk = dbgTrack.links.emplace_back();
#ifdef _ALLOW_DEBUG_AB_
          dbgLnk.seed = lnk.seed; // seed before update
#endif
          dbgLnk.clLabel = mITSClsLabels->getLabels(lnk.clID)[0];
          dbgLnk.chi2 = lnk.chi2;
          dbgLnk.lr = lnk.layerID;
          (o2::BaseCluster<float>&)dbgLnk = mITSClustersArray[lnk.clID];
          dbgTrack.nClusITS++;
          if (lblTrc.getEventID() == dbgLnk.clLabel.getEventID() && std::abs(lblTrc.getTrackID()) == dbgLnk.clLabel.getTrackID()) {
            dbgTrack.nClusITSCorr++;
          }
        }
        if (lnk.isDummyTop()) {   // reached dummy seed on the dummy layer above the last ITS layer
          dbgTrack.tpcSeed = lnk; // tpc seed
          dbgTrack.trackID = llist.trackID;
          dbgTrack.tpcLabel = lblTrc;
          dbgTrack.icCand = lnk.icCandID;
          dbgTrack.icTimeBin = mInteractions[lnk.icCandID].tBracket;
          const auto& tpcTrOrig = mTPCTracksArray[mTPCWork[llist.trackID].sourceID];
          unsigned nclTPC = tpcTrOrig.getNClusterReferences();
          dbgTrack.nClusTPC = nclTPC > 0xff ? 0xff : nclTPC;
          dbgTrack.sideAC = tpcTrOrig.hasASideClusters() + (tpcTrOrig.hasCSideClusters() ? 2 : 0);
          break;
        }
        parID = lnk.parentID;
      }
      dbgTrack.chi2 = dbgTrack.links.front().chi2;
      // at the moment links contain cumulative chi2, convert to track-to-cluster chi2
      for (int i = 0; i < dbgTrack.nClusITS - 1; i++) {
        dbgTrack.links[i].chi2 -= dbgTrack.links[i + 1].chi2;
      }
      // at the moment the links are stored from inner to outer layers, invert this order
      for (int i = dbgTrack.nClusITS / 2; i--;) {
        std::swap(dbgTrack.links[i], dbgTrack.links[dbgTrack.nClusITS - i - 1]);
      }
      dbgTrack.valid = ord == 0 && llist.isValidated();
      dbgTrack.order = ord++;
      // dump debug track
      (*mDBGOut) << "abtree"
                 << "trc=" << dbgTrack << "\n";
      entCnt++;
      nextHyp = mABTrackLinks[nextHyp].nextOnLr;
    }
  } // loop over layers
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
void MatchTPCITS::run(const o2::globaltracking::RecoContainer& inp)
{
  ///< perform matching for provided input
  if (!mInitDone) {
    LOG(FATAL) << "init() was not done yet";
  }
  mRecoCont = &inp;
  mStartIR = inp.startIR;
  updateTimeDependentParams();

  ProcInfo_t procInfoStart, procInfoStop;
  gSystem->GetProcInfo(&procInfoStart);
  constexpr uint64_t kMB = 1024 * 1024;
  LOGF(info, "Memory (GB) at entrance: RSS: %.3f VMem: %.3f\n", float(procInfoStart.fMemResident) / kMB, float(procInfoStart.fMemVirtual) / kMB);

  mTimer[SWTot].Start(false);

  clear();

  if (!prepareITSData() || !prepareTPCData() || !prepareFITData()) {
    return;
  }

  mTimer[SWDoMatching].Start(false);
  for (int sec = o2::constants::math::NSectors; sec--;) {
    doMatching(sec);
  }
  mTimer[SWDoMatching].Stop();
  if (0) { // enabling this creates very verbose output
    mTimer[SWTot].Stop();
    printCandidatesTPC();
    printCandidatesITS();
    mTimer[SWTot].Start(false);
  }

  selectBestMatches();

  refitWinners();

  if (mUseFT0 && Params::Instance().runAfterBurner) {
    runAfterBurner();
  }

#ifdef _ALLOW_DEBUG_TREES_
  if (mDBGOut && isDebugFlag(WinnerMatchesTree)) {
    dumpWinnerMatches();
  }
#endif

  gSystem->GetProcInfo(&procInfoStop);
  mTimer[SWTot].Stop();

  for (int i = 0; i < NStopWatches; i++) {
    LOGF(INFO, "Timing for %15s: Cpu: %.3e Real: %.3e s in %d slots of TF#%d", TimerName[i], mTimer[i].CpuTime(), mTimer[i].RealTime(), mTimer[i].Counter() - 1, mTFCount);
  }
  LOGF(INFO, "Memory (GB) at exit: RSS: %.3f VMem: %.3f", float(procInfoStop.fMemResident) / kMB, float(procInfoStop.fMemVirtual) / kMB);
  LOGF(INFO, "Memory increment: RSS: %.3f VMem: %.3f", float(procInfoStop.fMemResident - procInfoStart.fMemResident) / kMB,
       float(procInfoStop.fMemVirtual - procInfoStart.fMemVirtual) / kMB);
  mTFCount++;
}

//______________________________________________
void MatchTPCITS::end()
{
#ifdef _ALLOW_DEBUG_TREES_
  mDBGOut.reset();
#endif
}

//______________________________________________
void MatchTPCITS::clear()
{
  ///< clear results of previous TF reconstruction
  mMatchRecordsTPC.clear();
  mMatchRecordsITS.clear();
  mWinnerChi2Refit.clear();
  mMatchedTracks.clear();
  mITSWork.clear();
  mTPCWork.clear();
  mInteractions.clear();
  mITSROFIntCandEntries.clear();
  mITSROFTimes.clear();
  mITSTrackROFContMapping.clear();
  mITSClustersArray.clear();
  mABClusterLinkIndex.clear();
  mITSChipClustersRefs.clear();

  mABTrackLinksList.clear();
  mABTrackLinks.clear();
  mABClusterLinks.clear();
  mABBestLinks.clear();
  mABClusterLinkIndex.clear();
  mTPCABIndexCache.clear();
  mTPCABTimeBinStart.clear();

  for (int sec = o2::constants::math::NSectors; sec--;) {
    mITSSectIndexCache[sec].clear();
    mITSTimeStart[sec].clear();
    mTPCSectIndexCache[sec].clear();
    mTPCTimeStart[sec].clear();
  }

  if (mMCTruthON) {
    mOutLabels.clear();
    mITSROFTimes.clear();
    mTPCLblWork.clear();
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
  for (int i = NStopWatches; i--;) {
    mTimer[i].Stop();
    mTimer[i].Reset();
  }
  mParams = &Params::Instance();
  mFT0Params = &o2::ft0::InteractionTag::Instance();
  setUseMatCorrFlag(mParams->matCorr);
  auto* prop = o2::base::Propagator::Instance();
  if (!prop->getMatLUT() && mParams->matCorr == o2::base::Propagator::MatCorrType::USEMatCorrLUT) {
    LOG(WARNING) << "Requested material LUT is not loaded, switching to TGeo usage";
    setUseMatCorrFlag(o2::base::Propagator::MatCorrType::USEMatCorrTGeo);
  }

  // make sure T2GRot matrices are loaded into ITS geometry helper
  o2::its::GeometryTGeo::Instance()->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2GRot) | o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L));

  mSectEdgeMargin2 = mParams->crudeAbsDiffCut[o2::track::kY] * mParams->crudeAbsDiffCut[o2::track::kY]; ///< precalculated ^2
  std::unique_ptr<TPCTransform> fastTransform = (o2::tpc::TPCFastTransformHelperO2::instance()->create(0));
  mTPCTransform = std::move(fastTransform);

  if (mVDriftCalibOn) {
    float maxDTgl = std::min(0.02f, mParams->maxVDriftUncertainty) * mParams->maxTglForVDriftCalib;
    mHistoDTgl = std::make_unique<o2::dataformats::FlatHisto2D_f>(mParams->nBinsTglVDriftCalib, -mParams->maxTglForVDriftCalib, mParams->maxTglForVDriftCalib,
                                                                  mParams->nBinsDTglVDriftCalib, -maxDTgl, maxDTgl);
  }

#ifdef _ALLOW_DEBUG_TREES_
  // debug streamer
  if (mDBGFlags) {
    mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(mDebugTreeFileName.data(), "recreate");
  }
#endif

  mRGHelper.init(); // prepare helper for TPC track / ITS clusters matching
  const auto& zr = mRGHelper.layers.back().zRange;
  mITSFiducialZCut = std::max(std::abs(zr.getMin()), std::abs(zr.getMax())) + 20.;

  clear();

  mInitDone = true;

  if (fair::Logger::Logging(fair::Severity::info)) {
    print();
  }
}

//______________________________________________
void MatchTPCITS::updateTimeDependentParams()
{
  ///< update parameters depending on time (once per TF)
  auto& gasParam = o2::tpc::ParameterGas::Instance();
  auto& elParam = o2::tpc::ParameterElectronics::Instance();
  auto& detParam = o2::tpc::ParameterDetector::Instance();
  mTPCTBinMUS = elParam.ZbinWidth;
  mTPCTBinNS = mTPCTBinMUS * 1e3;
  mTPCVDrift0 = gasParam.DriftV;
  mTPCZMax = detParam.TPClength;
  mTPCTBinMUSInv = 1. / mTPCTBinMUS;
  assert(mITSROFrameLengthMUS > 0.0f);
  if (mITSTriggered) {
    mITSROFrame2TPCBin = mITSROFrameLengthMUS * mTPCTBinMUSInv;
  } else {
    mITSROFrame2TPCBin = mITSROFrameLengthMUS * mTPCTBinMUSInv; // RSTODO use both ITS and TPC times BCs once will be available for TPC also
  }
  mTPCBin2ITSROFrame = 1. / mITSROFrame2TPCBin;
  mTPCBin2Z = mTPCTBinMUS * mTPCVDrift0;
  mZ2TPCBin = 1. / mTPCBin2Z;
  mTPCVDrift0Inv = 1. / mTPCVDrift0;
  mNTPCBinsFullDrift = mTPCZMax * mZ2TPCBin;
  mTPCTimeEdgeTSafeMargin = z2TPCBin(mParams->safeMarginTPCTimeEdge);
  mTPCExtConstrainedNSigmaInv = 1.f / mParams->tpcExtConstrainedNSigma;
  mBz = o2::base::Propagator::Instance()->getNominalBz();
  mFieldON = std::abs(mBz) > 0.01;

  mMinTPCTrackPtInv = (mFieldON && mParams->minTPCTrackR > 0) ? 1. / std::abs(mParams->minTPCTrackR * mBz * o2::constants::math::B2C) : 999.;
  mMinITSTrackPtInv = (mFieldON && mParams->minITSTrackR > 0) ? 1. / std::abs(mParams->minITSTrackR * mBz * o2::constants::math::B2C) : 999.;

  o2::math_utils::Point3D<float> p0(90., 1., 1), p1(90., 100., 100.);
  auto matbd = o2::base::Propagator::Instance()->getMatBudget(mParams->matCorr, p0, p1);
  mTPCmeanX0Inv = matbd.meanX2X0 / matbd.length;
}

//______________________________________________
void MatchTPCITS::selectBestMatches()
{
  ///< loop over match records and select the ones with best chi2
  mTimer[SWSelectBest].Start(false);
  LOG(INFO) << "Selecting best matches";
  int nValidated = 0, iter = 0;

  do {
    nValidated = 0;
    int ntpc = mTPCWork.size(), nremaining = 0;
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
    LOGF(INFO, "iter %d Validated %d of %d remaining matches", iter, nValidated, nremaining);
    iter++;
  } while (nValidated);
  mTimer[SWSelectBest].Stop();
}

//______________________________________________
bool MatchTPCITS::validateTPCMatch(int iTPC)
{
  const auto& tTPC = mTPCWork[iTPC];
  auto& rcTPC = mMatchRecordsTPC[tTPC.matchID]; // best TPC->ITS match
  /* // should never happen
  if (rcTPC.nextRecID == Validated) {
    LOG(WARNING) << "TPC->ITS was already validated";
    return false; // RS do we need this
  }
  */
  // check if it is consistent with corresponding ITS->TPC match
  auto& tITS = mITSWork[rcTPC.partnerID];       //  partner ITS track
  auto& rcITS = mMatchRecordsITS[tITS.matchID]; // best ITS->TPC match record
  if (rcITS.nextRecID == Validated) {
    return false;
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
void MatchTPCITS::addTPCSeed(const o2::track::TrackParCov& _tr, float t0, float terr, GTrackID srcGID, int tpcID)
{
  // account single TPC seed, can be from standalone TPC track or constrained track from match to TRD and/or TOF
  const float SQRT12DInv = 2. / sqrt(12.);
  if (_tr.getX() > o2::constants::geom::XTPCInnerRef + 0.1 || std::abs(_tr.getQ2Pt()) > mMinTPCTrackPtInv) {
    return;
  }
  const auto& tpcOrig = mTPCTracksArray[tpcID];
  // discard tracks w/o certain number of total or innermost pads (last cluster is innermost one)
  if (tpcOrig.getNClusterReferences() < mParams->minTPCClusters) {
    return;
  }
  uint8_t clSect = 0, clRow = 0;
  uint32_t clIdx = 0;
  tpcOrig.getClusterReference(mTPCTrackClusIdx, tpcOrig.getNClusterReferences() - 1, clSect, clRow, clIdx);
  if (clRow > mParams->askMinTPCRow) {
    return;
  }
  // create working copy of track param
  bool extConstrained = srcGID.getSource() != GTrackID::TPC;
  if (extConstrained) {
    terr *= mParams->tpcExtConstrainedNSigma;
  } else {
    terr += tpcTimeBin2MUS(tpcOrig.hasBothSidesClusters() ? mParams->safeMarginTPCITSTimeBin : mTPCTimeEdgeTSafeMargin);
  }
  auto& trc = mTPCWork.emplace_back(
    TrackLocTPC{_tr, {t0 - terr, t0 + terr}, extConstrained ? t0 : tpcTimeBin2MUS(tpcOrig.getTime0()),
                // for A/C constrained tracks the terr is half-interval, for externally constrained tracks it is sigma*Nsigma
                terr * (extConstrained ? mTPCExtConstrainedNSigmaInv : SQRT12DInv),
                tpcID,
                srcGID,
                MinusOne,
                (extConstrained || tpcOrig.hasBothSidesClusters()) ? TrackLocTPC::Constrained : (tpcOrig.hasASideClustersOnly() ? TrackLocTPC::ASide : TrackLocTPC::CSide)});
  // propagate to matching Xref
  if (!propagateToRefX(trc)) {
    mTPCWork.pop_back(); // discard track whose propagation to XMatchingRef failed
    return;
  }
  if (mMCTruthON) {
    mTPCLblWork.emplace_back(mTPCTrkLabels[tpcID]);
  }
  // cache work track index
  mTPCSectIndexCache[o2::math_utils::angle2Sector(trc.getAlpha())].push_back(mTPCWork.size() - 1);
}

//______________________________________________
bool MatchTPCITS::prepareTPCData()
{
  ///< load TPC data and prepare for matching
  mTimer[SWPrepTPC].Start(false);
  const auto& inp = *mRecoCont;

  mTPCTracksArray = inp.getTPCTracks();
  mTPCTrackClusIdx = inp.getTPCTracksClusterRefs();
  mTPCClusterIdxStruct = &inp.inputsTPCclusters->clusterIndex;
  mTPCRefitterShMap = inp.clusterShMapTPC;
  if (mMCTruthON) {
    mTPCTrkLabels = inp.getTPCTracksMCLabels();
  }

  int ntr = mTPCTracksArray.size();
  mMatchRecordsTPC.reserve(mParams->maxMatchCandidates * ntr); // number of records might be actually more than N tracks!
  mTPCWork.reserve(ntr);
  if (mMCTruthON) {
    mTPCLblWork.reserve(ntr);
  }
  for (int sec = o2::constants::math::NSectors; sec--;) {
    mTPCSectIndexCache[sec].reserve(100 + 1.2 * ntr / o2::constants::math::NSectors);
  }

  auto creator = [this](auto& trk, GTrackID gid, float time0, float terr) {
    if constexpr (isITSTrack<decltype(trk)>()) {
      // do nothing, ITS tracks will be processed in a direct loop over ROFs
    }
    if constexpr (isTPCTrack<decltype(trk)>()) {
      // unconstrained TPC track, with t0 = TrackTPC.getTime0+0.5*(DeltaFwd-DeltaBwd) and terr = 0.5*(DeltaFwd+DeltaBwd) in TimeBins
      if (!this->mSkipTPCOnly) {
        this->addTPCSeed(trk, this->tpcTimeBin2MUS(time0), this->tpcTimeBin2MUS(terr), gid, gid.getIndex());
      }
    }
    if constexpr (isTPCTOFTrack<decltype(trk)>()) {
      // TPC track constrained by TOF time, time and its error in \mus
      this->addTPCSeed(trk, time0, terr, gid, this->mRecoCont->getTPCContributorGID(gid));
    }
    if constexpr (isTRDTrack<decltype(trk)>()) {
      // TPC track constrained by TRD trigger time, time and its error in \mus
      LOG(ERROR) << "Not ready yet for TPC-TRD tracks";
    }
    if constexpr (isTPCTRDTOFTrack<decltype(trk)>()) {
      // TPC track constrained by TRD and TOF time, time and its error in \mus
      LOG(ERROR) << "Not ready yet for TPC-TRD-TOF tracks";
    }
    return true;
  };
  mRecoCont->createTracksVariadic(creator);

  float maxTime = 0;
  int nITSROFs = mITSROFTimes.size();
  // sort tracks in each sector according to their timeMax
  for (int sec = o2::constants::math::NSectors; sec--;) {
    auto& indexCache = mTPCSectIndexCache[sec];
    LOG(INFO) << "Sorting sector" << sec << " | " << indexCache.size() << " TPC tracks";
    if (!indexCache.size()) {
      continue;
    }
    std::sort(indexCache.begin(), indexCache.end(), [this](int a, int b) {
      auto& trcA = mTPCWork[a];
      auto& trcB = mTPCWork[b];
      return (trcA.tBracket.getMax() - trcB.tBracket.getMax()) < 0.;
    });

    // build array of 1st entries with tmax corresponding to each ITS ROF (or trigger),
    // TPC tracks below this entry cannot match to ITS tracks of this and higher ROFs

    float tmax = mTPCWork[indexCache.back()].tBracket.getMax();
    if (maxTime < tmax) {
      maxTime = tmax;
    }
    int nbins = 1 + time2ITSROFrame(tmax);
    auto& timeStart = mTPCTimeStart[sec];
    timeStart.resize(nbins, -1);
    int itsROF = 0;

    timeStart[0] = 0;
    for (int itr = 0; itr < (int)indexCache.size(); itr++) {
      auto& trc = mTPCWork[indexCache[itr]];
      while (itsROF < nITSROFs && !(trc.tBracket < mITSROFTimes[itsROF])) { // 1st ITS frame afte max allowed time for this TPC track
        itsROF++;
      }
      int itsROFMatch = itsROF;
      if (itsROFMatch && timeStart[--itsROFMatch] == -1) { // register ITSrof preceding the one which exceeds the TPC track tmax
        timeStart[itsROFMatch] = itr;
      }
    }
    for (int i = 1; i < nbins; i++) {
      if (timeStart[i] == -1) { // fill gaps with preceding indices
        timeStart[i] = timeStart[i - 1];
      }
    }
  } // loop over tracks of single sector

  // create mapping from TPC time-bins to ITS ROFs

  if (mITSROFTimes.back() < maxTime) {
    maxTime = mITSROFTimes.back().getMax();
  }

  // FIXME
  /*
  int nb = int(maxTime) + 1;
  mITSROFofTPCBin.resize(nb, -1);
  int itsROF = 0;
  for (int ib = 0; ib < nb; ib++) {
    while (itsROF < nITSROFs && ib < mITSROFTimes[itsROF].getMin()) {
      itsROF++;
    }
    mITSROFofTPCBin[ib] = itsROF;
  }
*/
  mTPCRefitter = std::make_unique<o2::gpu::GPUO2InterfaceRefit>(mTPCClusterIdxStruct, mTPCTransform.get(), mBz, mTPCTrackClusIdx.data(), mTPCRefitterShMap.data(), nullptr, o2::base::Propagator::Instance());

  mTimer[SWPrepTPC].Stop();
  return mTPCWork.size() > 0;
}

//_____________________________________________________
bool MatchTPCITS::prepareITSData()
{
  // Do preparatory work for matching
  mTimer[SWPrepITS].Start(false);
  const auto& inp = *mRecoCont;

  // ITS clusters
  mITSClusterROFRec = inp.getITSClustersROFRecords();
  const auto clusITS = inp.getITSClusters();
  if (mITSClusterROFRec.empty() || clusITS.empty()) {
    LOG(INFO) << "No ITS clusters";
    return false;
  }
  const auto patterns = inp.getITSClustersPatterns();
  auto pattIt = patterns.begin();
  mITSClustersArray.reserve(clusITS.size());
  o2::its::ioutils::convertCompactClusters(clusITS, pattIt, mITSClustersArray, *mITSDict);
  if (mMCTruthON) {
    mITSClsLabels = inp.mcITSClusters.get();
  }

  // ITS tracks
  mITSTracksArray = inp.getITSTracks();
  mITSTrackClusIdx = inp.getITSTracksClusterRefs();
  mITSTrackROFRec = inp.getITSTracksROFRecords();
  if (mMCTruthON) {
    mITSTrkLabels = inp.getITSTracksMCLabels();
  }
  int nROFs = mITSTrackROFRec.size();
  mITSWork.reserve(mITSTracksArray.size());

  // total N ITS clusters in TF
  const auto& lastClROF = mITSClusterROFRec[nROFs - 1]; //.back();
  int nITSClus = lastClROF.getFirstEntry() + lastClROF.getNEntries();
  mABClusterLinkIndex.resize(nITSClus, MinusOne);
  for (int sec = o2::constants::math::NSectors; sec--;) {
    mITSTimeStart[sec].resize(nROFs, -1); // start of ITS work tracks in every sector
  }

  for (int irof = 0; irof < nROFs; irof++) {
    const auto& rofRec = mITSTrackROFRec[irof];
    int nBC = rofRec.getBCData().differenceInBC(mStartIR);
    float tMin = nBC * o2::constants::lhc::LHCBunchSpacingMUS;
    float tMax = (nBC + mITSROFrameLengthInBC) * o2::constants::lhc::LHCBunchSpacingMUS;
    if (!mITSTriggered) {
      auto irofCont = nBC / mITSROFrameLengthInBC;
      if (mITSTrackROFContMapping.size() <= irofCont) { // there might be gaps in the non-empty rofs, this will map continuous ROFs index to non empty ones
        mITSTrackROFContMapping.resize((1 + irofCont / 128) * 128, 0);
      }
      mITSTrackROFContMapping[irofCont] = irof;
    }

    int cluROFOffset = mITSClusterROFRec[irof].getFirstEntry(); // clusters of this ROF start at this offset
    mITSROFTimes.emplace_back(tMin, tMax);                      // ITS ROF min/max time

    for (int sec = o2::constants::math::NSectors; sec--;) {         // start of sector's tracks for this ROF
      mITSTimeStart[sec][irof] = mITSSectIndexCache[sec].size();    // The sorting does not affect this
    }

    int trlim = rofRec.getFirstEntry() + rofRec.getNEntries();
    for (int it = rofRec.getFirstEntry(); it < trlim; it++) {
      const auto& trcOrig = mITSTracksArray[it];
      if (mParams->runAfterBurner) {
        flagUsedITSClusters(trcOrig, cluROFOffset);
      }
      if (trcOrig.getParamOut().getX() < 1.) {
        continue; // backward refit failed
      }
      if (std::abs(trcOrig.getQ2Pt()) > mMinITSTrackPtInv) {
        continue;
      }
      int nWorkTracks = mITSWork.size();
      // working copy of outer track param
      auto& trc = mITSWork.emplace_back(TrackLocITS{trcOrig.getParamOut(), {tMin, tMax}, it, irof, MinusOne});
      if (!trc.rotate(o2::math_utils::angle2Alpha(trc.getPhiPos()))) {
        mITSWork.pop_back(); // discard failed track
        continue;
      }
      // make sure the track is at the ref. radius
      if (!propagateToRefX(trc)) {
        mITSWork.pop_back(); // discard failed track
        continue;            // add to cache only those ITS tracks which reached ref.X and have reasonable snp
      }
      if (mMCTruthON) {
        mITSLblWork.emplace_back(mITSTrkLabels[it]);
      }
      // cache work track index
      int sector = o2::math_utils::angle2Sector(trc.getAlpha());
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
        addLastTrackCloneForNeighbourSector(sector < (o2::constants::math::NSectors - 1) ? sector + 1 : 0);
      }
      // sector down
      float dy2Dn = (YMaxAtXMatchingRef + trc.getY()) / (tgp - Tan70);
      if ((dy2Dn * dy2Dn * Cos70I2) < mSectEdgeMargin2) { // need to check this track for matching in sector down
        addLastTrackCloneForNeighbourSector(sector > 1 ? sector - 1 : o2::constants::math::NSectors - 1);
      }
    }
  }

  if (!mITSTriggered) { // fill the gaps;
    int nr = mITSTrackROFContMapping.size();
    for (int i = 1; i < nr; i++) {
      if (mITSTrackROFContMapping[i] < mITSTrackROFContMapping[i - 1]) {
        mITSTrackROFContMapping[i] = mITSTrackROFContMapping[i - 1];
      }
    }
  }

  // sort tracks in each sector according to their min time, then tgl
  // RSTODO: sorting in tgl will be dangerous once the tracks with different time uncertaincies will be added
  for (int sec = o2::constants::math::NSectors; sec--;) {
    auto& indexCache = mITSSectIndexCache[sec];
    LOG(INFO) << "Sorting sector" << sec << " | " << indexCache.size() << " ITS tracks";
    if (!indexCache.size()) {
      continue;
    }
    std::sort(indexCache.begin(), indexCache.end(), [this](int a, int b) {
      auto& trackA = mITSWork[a];
      auto& trackB = mITSWork[b];
      if (trackA.tBracket.getMin() < trackB.tBracket.getMin()) {
        return true;
      } else if (trackA.tBracket.getMin() > trackB.tBracket.getMin()) {
        return false;
      }
      return trackA.getTgl() < trackB.getTgl();
    });
  } // loop over tracks of single sector
  mMatchRecordsITS.reserve(mITSWork.size() * mParams->maxMatchCandidates);
  mTimer[SWPrepITS].Stop();

  return nITSClus > 0;
}

//_____________________________________________________
bool MatchTPCITS::prepareFITData()
{
  // If available, read FIT Info
  if (mUseFT0) {
    mFITInfo = mRecoCont->getFT0RecPoints();
    prepareInteractionTimes();
  }
  return true;
}

//_____________________________________________________
void MatchTPCITS::doMatching(int sec)
{
  ///< run matching for currently cached ITS data for given TPC sector
  auto& cacheITS = mITSSectIndexCache[sec];   // array of cached ITS track indices for this sector
  auto& cacheTPC = mTPCSectIndexCache[sec];   // array of cached ITS track indices for this sector
  auto& timeStartTPC = mTPCTimeStart[sec];    // array of 1st TPC track with timeMax in ITS ROFrame
  auto& timeStartITS = mITSTimeStart[sec];
  int nTracksTPC = cacheTPC.size(), nTracksITS = cacheITS.size();
  if (!nTracksTPC || !nTracksITS) {
    LOG(INFO) << "Matchng sector " << sec << " : N tracks TPC:" << nTracksTPC << " ITS:" << nTracksITS << " in sector " << sec;
    return;
  }

  /// full drift time + safety margin
  float maxTDriftSafe = tpcTimeBin2MUS(mNTPCBinsFullDrift + mParams->safeMarginTPCITSTimeBin + mTPCTimeEdgeTSafeMargin);
  float vdErrT = tpcTimeBin2MUS(mZ2TPCBin * mParams->maxVDriftUncertainty);

  // get min ROFrame of ITS tracks currently in cache
  auto minROFITS = mITSWork[cacheITS.front()].roFrame;

  if (minROFITS >= int(timeStartTPC.size())) {
    LOG(INFO) << "ITS min ROFrame " << minROFITS << " exceeds all cached TPC track ROF eqiuvalent " << cacheTPC.size() - 1;
    return;
  }

  int nCheckTPCControl = 0, nCheckITSControl = 0, nMatchesControl = 0; // temporary
  int idxMinTPC = timeStartTPC[minROFITS];                             // index of 1st cached TPC track within cached ITS ROFrames
  auto t2nbs = tpcTimeBin2MUS(mZ2TPCBin * mParams->tpcTimeICMatchingNSigma); // FIXME work directly with time in \mus
  bool checkInteractionCandidates = mUseFT0 && mParams->validateMatchByFIT != MatchTPCITSParams::Disable;

  for (int itpc = idxMinTPC; itpc < nTracksTPC; itpc++) {
    auto& trefTPC = mTPCWork[cacheTPC[itpc]];
    // estimate ITS 1st ROframe bin this track may match to: TPC track are sorted according to their
    // timeMax, hence the timeMax - MaxmNTPCBinsFullDrift are non-decreasing
    int itsROBin = time2ITSROFrame(trefTPC.tBracket.getMax() - maxTDriftSafe);

    if (itsROBin >= int(timeStartITS.size())) { // time of TPC track exceeds the max time of ITS in the cache
      break;
    }
    int iits0 = timeStartITS[itsROBin];
    nCheckTPCControl++;
    for (auto iits = iits0; iits < nTracksITS; iits++) {
      auto& trefITS = mITSWork[cacheITS[iits]];
      // compare if the ITS and TPC tracks may overlap in time
      if (trefTPC.tBracket < trefITS.tBracket) { // since TPC tracks are sorted in timeMax and ITS tracks are sorted in timeMin all following ITS tracks also will not match
        break;
      }
      if (trefTPC.tBracket > trefITS.tBracket) { // its bracket precedes TPC bracket
        continue;
      }

      // is corrected TPC track time compatible with ITS ROF expressed
      auto deltaT = (trefITS.getZ() - trefTPC.getZ()) * mTPCVDrift0Inv;                  // drift time difference corresponding to Z differences
      auto timeCorr = trefTPC.getCorrectedTime(deltaT);                                  // TPC time required to match to Z of ITS track
      auto timeCorrErr = std::sqrt(trefITS.getSigmaZ2() + trefTPC.getSigmaZ2()) * t2nbs; // nsigma*error
      if (mVDriftCalibOn) {
        timeCorrErr += vdErrT * (250. - abs(trefITS.getZ())); // account for the extra error from TPC VDrift uncertainty
      }
      o2::math_utils::Bracketf_t trange(timeCorr - timeCorrErr, timeCorr + timeCorrErr);
      if (trefITS.tBracket.isOutside(trange)) {
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
      /*
      // RS: this might be dangerous for ITS tracks with different time coverages.
      if (rejFlag == RejectOnTgl) {
        // ITS tracks in each ROFrame are ordered in Tgl, hence if this check failed on Tgl check
        // (i.e. tgl_its>tgl_tpc+tolerance), then all other ITS tracks in this ROFrame will also have tgl too large.
        // Jump on the 1st ITS track of the next ROFrame
        int rof = trefITS.roFrame;
        bool stop = false;
        do {
          if (++rof >= int(timeStartITS.size())) {
            stop = true;
            break; // no more ITS ROFrames in cache
          }
          iits = timeStartITS[rof] - 1;                  // next track to be checked -1
        } while (iits <= timeStartITS[trefITS.roFrame]); // skip empty bins
        if (stop) {
          break;
        }
        continue;
      }
      */
      if (rejFlag != Accept) {
        continue;
      }
      int matchedIC = MinusOne;
      if (!isCosmics()) {
        // validate by bunch filling scheme
        auto irBracket = tBracket2IRBracket(trange);
        if (irBracket.isInvalid()) {
          continue;
        }

        if (checkInteractionCandidates) {
          // check if corrected TPC track time is compatible with any of interaction times
          auto interactionRefs = mITSROFIntCandEntries[trefITS.roFrame]; // reference on interaction candidates compatible with this track
          int nic = interactionRefs.getEntries();
          if (nic) {
            int idIC = interactionRefs.getFirstEntry(), maxIC = idIC + nic;
            for (; idIC < maxIC; idIC++) {
              auto cmp = mInteractions[idIC].tBracket.isOutside(trange);
              if (cmp == o2::math_utils::Bracketf_t::Above) { // trange is above this interaction candidate, the following ones may match
                continue;
              }
              if (cmp == o2::math_utils::Bracketf_t::Inside) {
                matchedIC = idIC;
              }
              break; // we loop till 1st matching IC or the one above the trange (since IC are ordered, all others will be above too)
            }
          }
        }
        if (mParams->validateMatchByFIT == MatchTPCITSParams::Require && matchedIC == MinusOne) {
          continue;
        }
      }
      registerMatchRecordTPC(cacheITS[iits], cacheTPC[itpc], chi2, matchedIC); // register matching candidate
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
  int topID = MinusOne, recordID = tITS.matchID; // 1st entry in mMatchRecordsITS
  while (recordID > MinusOne) {                  // navigate over records for given ITS track
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
bool MatchTPCITS::registerMatchRecordTPC(int iITS, int iTPC, float chi2, int candIC)
{
  ///< record matching candidate, making sure that number of ITS candidates per TPC track, sorted
  ///< in matching chi2 does not exceed allowed number
  auto& tTPC = mTPCWork[iTPC];                                   // get MatchRecord structure of this TPC track, create if none
  if (tTPC.matchID < 0) {                                        // no matches yet, just add new record
    registerMatchRecordITS(iITS, iTPC, chi2, candIC);            // register TPC track in the ITS records
    tTPC.matchID = mMatchRecordsTPC.size();                      // new record will be added in the end
    mMatchRecordsTPC.emplace_back(iITS, chi2, MinusOne, candIC); // create new record with empty reference on next match
    return true;
  }

  int count = 0, nextID = tTPC.matchID, topID = MinusOne;
  do {
    auto& nextMatchRec = mMatchRecordsTPC[nextID];
    count++;
    if (!nextMatchRec.isBetter(chi2, candIC)) { // need to insert new record before nextMatchRec?
      if (count < mParams->maxMatchCandidates) {
        break;                                                // will insert in front of nextID
      } else {                                                // max number of candidates reached, will overwrite the last one
        suppressMatchRecordITS(nextMatchRec.partnerID, iTPC); // flag as disabled the overriden ITS match
        registerMatchRecordITS(iITS, iTPC, chi2, candIC);     // register TPC track entry in the ITS records
        // reuse the record of suppressed ITS match to store better one
        nextMatchRec.chi2 = chi2;
        nextMatchRec.matchedIC = candIC;
        nextMatchRec.partnerID = iITS;
        return true;
      }
    }
    topID = nextID; // check next match record
    nextID = nextMatchRec.nextRecID;
  } while (nextID > MinusOne);

  // if count == mParams->maxMatchCandidates, the max number of candidates was already reached, and the
  // new candidated was either discarded (if its chi2 is worst one) or has overwritten worst
  // existing candidate. Otherwise, we need to add new entry
  if (count < mParams->maxMatchCandidates) {
    if (topID < 0) {                                                       // the new match is top candidate
      topID = tTPC.matchID = mMatchRecordsTPC.size();                      // register new record as top one
    } else {                                                               // there are better candidates
      topID = mMatchRecordsTPC[topID].nextRecID = mMatchRecordsTPC.size(); // register to his parent
    }
    // nextID==-1 will mean that the while loop run over all candidates->the new one is the worst (goes to the end)
    registerMatchRecordITS(iITS, iTPC, chi2, candIC);          // register TPC track in the ITS records
    mMatchRecordsTPC.emplace_back(iITS, chi2, nextID, candIC); // create new record with empty reference on next match
    // make sure that after addition the number of candidates don't exceed allowed number
    count++;
    while (nextID > MinusOne) {
      if (count > mParams->maxMatchCandidates) {
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
void MatchTPCITS::registerMatchRecordITS(int iITS, int iTPC, float chi2, int candIC)
{
  ///< register TPC match in ITS tracks match records, ordering them in quality
  auto& tITS = mITSWork[iITS];
  int idnew = mMatchRecordsITS.size();
  auto& newRecord = mMatchRecordsITS.emplace_back(iTPC, chi2, MinusOne, candIC); // associate iTPC with this record
  if (tITS.matchID < 0) {
    tITS.matchID = idnew;
    return;
  }
  // there are other matches for this ITS track, insert the new record preserving quality order
  // navigate till last record or the one with worse chi2
  int topID = MinusOne, nextRecord = tITS.matchID;
  do {
    auto& nextMatchRec = mMatchRecordsITS[nextRecord];
    if (!nextMatchRec.isBetter(chi2, candIC)) { // need to insert new record before nextMatchRec?
      newRecord.nextRecID = nextRecord;         // new one will refer to old one it overtook
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
  chi2 = -1.f;
  int rejFlag = Accept;
  float diff; // make rough check differences and their nsigmas

  // start with check on Tgl, since rjection on it will allow to profit from sorting
  diff = tITS.getParam(o2::track::kTgl) - tTPC.getParam(o2::track::kTgl);
  if ((rejFlag = roughCheckDif(diff, mParams->crudeAbsDiffCut[o2::track::kTgl], RejectOnTgl))) {
    return rejFlag;
  }
  auto err2Tgl = tITS.getDiagError2(o2::track::kTgl) + tTPC.getDiagError2(o2::track::kTgl);
  if (mVDriftCalibOn) {
    auto addErr = tITS.getParam(o2::track::kTgl) * mParams->maxVDriftUncertainty;
    err2Tgl += addErr * addErr; // account for VDrift uncertainty
  }
  diff *= diff / err2Tgl;
  if ((rejFlag = roughCheckDif(diff, mParams->crudeNSigma2Cut[o2::track::kTgl], RejectOnTgl + NSigmaShift))) {
    return rejFlag;
  }
  diff = tITS.getParam(o2::track::kY) - tTPC.getParam(o2::track::kY);
  if ((rejFlag = roughCheckDif(diff, mParams->crudeAbsDiffCut[o2::track::kY], RejectOnY))) {
    return rejFlag;
  }
  diff *= diff / (tITS.getDiagError2(o2::track::kY) + tTPC.getDiagError2(o2::track::kY));
  if ((rejFlag = roughCheckDif(diff, mParams->crudeNSigma2Cut[o2::track::kY], RejectOnY + NSigmaShift))) {
    return rejFlag;
  }

  if (tTPC.constraint == TrackLocTPC::Constrained) { // in continuous only constrained tracks can be compared in Z
    diff = tITS.getParam(o2::track::kZ) - tTPC.getParam(o2::track::kZ);
    if ((rejFlag = roughCheckDif(diff, mParams->crudeAbsDiffCut[o2::track::kZ], RejectOnZ))) {
      return rejFlag;
    }
    diff *= diff / (tITS.getDiagError2(o2::track::kZ) + tTPC.getDiagError2(o2::track::kZ));
    if ((rejFlag = roughCheckDif(diff, mParams->crudeNSigma2Cut[o2::track::kZ], RejectOnZ + NSigmaShift))) {
      return rejFlag;
    }
  }

  diff = tITS.getParam(o2::track::kSnp) - tTPC.getParam(o2::track::kSnp);
  if ((rejFlag = roughCheckDif(diff, mParams->crudeAbsDiffCut[o2::track::kSnp], RejectOnSnp))) {
    return rejFlag;
  }
  diff *= diff / (tITS.getDiagError2(o2::track::kSnp) + tTPC.getDiagError2(o2::track::kSnp));
  if ((rejFlag = roughCheckDif(diff, mParams->crudeNSigma2Cut[o2::track::kSnp], RejectOnSnp + NSigmaShift))) {
    return rejFlag;
  }

  diff = tITS.getParam(o2::track::kQ2Pt) - tTPC.getParam(o2::track::kQ2Pt);
  if ((rejFlag = roughCheckDif(diff, mParams->crudeAbsDiffCut[o2::track::kQ2Pt], RejectOnQ2Pt))) {
    return rejFlag;
  }
  diff *= diff / (tITS.getDiagError2(o2::track::kQ2Pt) + tTPC.getDiagError2(o2::track::kQ2Pt));
  if ((rejFlag = roughCheckDif(diff, mParams->crudeNSigma2Cut[o2::track::kQ2Pt], RejectOnQ2Pt + NSigmaShift))) {
    return rejFlag;
  }
  // calculate mutual chi2 excluding Z in continuos mode
  chi2 = getPredictedChi2NoZ(tITS, tTPC);
  if (chi2 > mParams->cutMatchingChi2) {
    return RejectOnChi2;
  }

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
    printf("*** trackTPC#%6d %6d : Ncand = %d Best = %d\n", i, tTPC.sourceID, nm, tTPC.matchID);
    int count = 0, recID = tTPC.matchID;
    while (recID > MinusOne) {
      const auto& rcTPC = mMatchRecordsTPC[recID];
      const auto& tITS = mITSWork[rcTPC.partnerID];
      printf("  * cand %2d : ITS track %6d(src:%6d) Chi2: %.2f\n", count, rcTPC.partnerID, tITS.sourceID, rcTPC.chi2);
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
    printf("*** trackITS#%6d %6d : Ncand = %d Best = %d\n", i, tITS.sourceID, getNMatchRecordsITS(tITS), tITS.matchID);
    int count = 0, recID = tITS.matchID;
    while (recID > MinusOne) {
      const auto& rcITS = mMatchRecordsITS[recID];
      const auto& tTPC = mTPCWork[rcITS.partnerID];
      printf("  * cand %2d : TPC track %6d(src:%6d) Chi2: %.2f\n", count, rcITS.partnerID, tTPC.sourceID, rcITS.chi2);
      count++;
      recID = rcITS.nextRecID;
    }
  }
}

//______________________________________________
float MatchTPCITS::getPredictedChi2NoZ(const o2::track::TrackParCov& trITS, const o2::track::TrackParCov& trTPC) const
{
  /// get chi2 between 2 tracks, neglecting Z parameter.
  /// 2 tracks must be defined at the same parameters X,alpha (check is currently commented)

  //  if (std::abs(trITS.getAlpha() - trTPC.getAlpha()) > FLT_EPSILON) {
  //    LOG(ERROR) << "The reference Alpha of the tracks differ: "
  //        << trITS.getAlpha() << " : " << trTPC.getAlpha();
  //    return 2. * o2::track::HugeF;
  //  }
  //  if (std::abs(trITS.getX() - trTPC.getX()) > FLT_EPSILON) {
  //    LOG(ERROR) << "The reference X of the tracks differ: "
  //        << trITS.getX() << " : " << trTPC.getX();
  //    return 2. * o2::track::HugeF;
  //  }
  MatrixDSym4 covMat;
  covMat(0, 0) = static_cast<double>(trITS.getSigmaY2()) + static_cast<double>(trTPC.getSigmaY2());
  covMat(1, 0) = static_cast<double>(trITS.getSigmaSnpY()) + static_cast<double>(trTPC.getSigmaSnpY());
  covMat(1, 1) = static_cast<double>(trITS.getSigmaSnp2()) + static_cast<double>(trTPC.getSigmaSnp2());
  covMat(2, 0) = static_cast<double>(trITS.getSigmaTglY()) + static_cast<double>(trTPC.getSigmaTglY());
  covMat(2, 1) = static_cast<double>(trITS.getSigmaTglSnp()) + static_cast<double>(trTPC.getSigmaTglSnp());
  covMat(2, 2) = static_cast<double>(trITS.getSigmaTgl2()) + static_cast<double>(trTPC.getSigmaTgl2());
  if (mVDriftCalibOn) {
    auto addErr = trITS.getParam(o2::track::kTgl) * mParams->maxVDriftUncertainty;
    covMat(2, 2) += addErr * addErr;
  }
  covMat(3, 0) = static_cast<double>(trITS.getSigma1PtY()) + static_cast<double>(trTPC.getSigma1PtY());
  covMat(3, 1) = static_cast<double>(trITS.getSigma1PtSnp()) + static_cast<double>(trTPC.getSigma1PtSnp());
  covMat(3, 2) = static_cast<double>(trITS.getSigma1PtTgl()) + static_cast<double>(trTPC.getSigma1PtTgl());
  covMat(3, 3) = static_cast<double>(trITS.getSigma1Pt2()) + static_cast<double>(trTPC.getSigma1Pt2());
  if (!covMat.Invert()) {
    LOG(ERROR) << "Cov.matrix inversion failed: " << covMat;
    return 2. * o2::track::HugeF;
  }
  double chi2diag = 0., chi2ndiag = 0.,
         diff[o2::track::kNParams - 1] = {trITS.getParam(o2::track::kY) - trTPC.getParam(o2::track::kY),
                                          trITS.getParam(o2::track::kSnp) - trTPC.getParam(o2::track::kSnp),
                                          trITS.getParam(o2::track::kTgl) - trTPC.getParam(o2::track::kTgl),
                                          trITS.getParam(o2::track::kQ2Pt) - trTPC.getParam(o2::track::kQ2Pt)};
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
  // add clone of the src ITS track cache, propagate it to ref.X in requested sector
  // and register its index in the sector cache. Used for ITS tracks which are so close
  // to their setctor edge that their matching should be checked also in the neighbouring sector
  mITSWork.push_back(mITSWork.back()); // clone the last track defined in given sector
  auto& trc = mITSWork.back();
  if (trc.rotate(o2::math_utils::sector2Angle(sector)) &&
      o2::base::Propagator::Instance()->PropagateToXBxByBz(trc, XMatchingRef, MaxSnp, 2., MatCorrType::USEMatCorrNONE)) {
    // TODO: use faster prop here, no 3d field, materials
    mITSSectIndexCache[sector].push_back(mITSWork.size() - 1); // register track CLONE
    if (mMCTruthON) {
      mITSLblWork.emplace_back(mITSTrkLabels[trc.sourceID]);
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
  int trialsLeft = 2;
  while (o2::base::Propagator::Instance()->PropagateToXBxByBz(trc, XMatchingRef, MaxSnp, 2., mUseMatCorrFlag)) {
    if (refReached) {
      break;
    }
    // make sure the track is indeed within the sector defined by alpha
    if (fabs(trc.getY()) < XMatchingRef * tan(o2::constants::math::SectorSpanRad / 2)) {
      refReached = true;
      break; // ok, within
    }
    if (!trialsLeft--) {
      break;
    }
    auto alphaNew = o2::math_utils::angle2Alpha(trc.getPhiPos());
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
  printf("Cut on matching chi2: %.3f\n", mParams->cutMatchingChi2);
  printf("Max number ITS candidates per TPC track: %d\n", mParams->maxMatchCandidates);
  printf("Crude cut on track params: ");
  for (int i = 0; i < o2::track::kNParams; i++) {
    printf(" %.3e", mParams->crudeAbsDiffCut[i]);
  }
  printf("\n");

  printf("NSigma^2 cut on track params: ");
  for (int i = 0; i < o2::track::kNParams; i++) {
    printf(" %6.2f", mParams->crudeNSigma2Cut[i]);
  }
  printf("\n");

  printf("TPC-ITS time(bins) bracketing safety margin: %6.2f\n", mParams->timeBinTolerance);
  printf("TPC Z->time(bins) bracketing safety margin: %6.2f\n", mParams->safeMarginTPCTimeEdge);

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

  mTimer[SWRefit].Start(false);
  LOG(INFO) << "Refitting winner matches";
  mWinnerChi2Refit.resize(mITSWork.size(), -1.f);
  int iITS;
  for (int iTPC = 0; iTPC < (int)mTPCWork.size(); iTPC++) {
    if (!refitTrackTPCITS(iTPC, iITS)) {
      continue;
    }
    mWinnerChi2Refit[iITS] = mMatchedTracks.back().getChi2Refit();
  }
  mTimer[SWRefit].Stop();
}

//______________________________________________
bool MatchTPCITS::refitTrackTPCITS(int iTPC, int& iITS)
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
  const auto& itsTrOrig = mITSTracksArray[tITS.sourceID];

  mMatchedTracks.emplace_back(tTPC, tITS); // create a copy of TPC track at xRef
  auto& trfit = mMatchedTracks.back();
  // in continuos mode the Z of TPC track is meaningless, unless it is CE crossing
  // track (currently absent, TODO)
  if (!mCompareTracksDZ) {
    trfit.setZ(tITS.getZ()); // fix the seed Z
  }
  float deltaT = (trfit.getZ() - tTPC.getZ()) * mTPCVDrift0Inv; // time correction in \mus

  // refit TPC track inward into the ITS
  int nclRefit = 0, ncl = itsTrOrig.getNumberOfClusters();
  float chi2 = 0.f;
  auto geom = o2::its::GeometryTGeo::Instance();
  auto propagator = o2::base::Propagator::Instance();
  // NOTE: the ITS cluster index is stored wrt 1st cluster of relevant ROF, while here we extract clusters from the
  // buffer for the whole TF. Therefore, we should shift the index by the entry of the ROF's 1st cluster in the global cluster buffer
  int clusIndOffs = mITSClusterROFRec[tITS.roFrame].getFirstEntry();
  int clEntry = itsTrOrig.getFirstClusterEntry();

  float addErr2 = 0;
  // extra error on tgl due to the assumed vdrift uncertainty
  if (mVDriftCalibOn) {
    addErr2 = tITS.getParam(o2::track::kTgl) * mParams->maxVDriftUncertainty;
    addErr2 *= addErr2;
    trfit.updateCov(addErr2, o2::track::kSigTgl2);
  }

  for (int icl = 0; icl < ncl; icl++) {
    const auto& clus = mITSClustersArray[clusIndOffs + mITSTrackClusIdx[clEntry++]];
    float alpha = geom->getSensorRefAlpha(clus.getSensorID()), x = clus.getX();
    if (!trfit.rotate(alpha) ||
        // note: here we also calculate the L,T integral (in the inward direction, but this is irrelevant)
        // note: we should eventually use TPC pid in the refit (TODO)
        // note: since we are at small R, we can use field BZ component at origin rather than 3D field
        !propagator->propagateToX(trfit, x, propagator->getNominalBz(),
                                  MaxSnp, maxStep, mUseMatCorrFlag, &trfit.getLTIntegralOut())) {
      break;
    }
    chi2 += trfit.getPredictedChi2(clus);
    if (!trfit.update(clus)) {
      break;
    }
    nclRefit++;
  }
  if (nclRefit != ncl) {
    LOGP(WARNING, "Refit in ITS failed after ncl={}, match between TPC track #{} and ITS track #{}", nclRefit, tTPC.sourceID, tITS.sourceID);
    LOGP(WARNING, "{:s}", trfit.asString());
    mMatchedTracks.pop_back(); // destroy failed track
    return false;
  }

  // We need to update the LTOF integral by the distance to the "primary vertex"
  // We want to leave the track at the the position of its last update, so we do a fast propagation on the TrackPar copy of trfit,
  // and since for the LTOF calculation the material effects are irrelevant, we skip material corrections
  const o2::dataformats::VertexBase vtxDummy; // at the moment using dummy vertex: TODO use MeanVertex constraint instead
  o2::track::TrackPar trpar(trfit);
  if (!propagator->propagateToDCA(vtxDummy.getXYZ(), trpar, propagator->getNominalBz(),
                                  maxStep, MatCorrType::USEMatCorrNONE, nullptr, &trfit.getLTIntegralOut())) {
    LOG(ERROR) << "LTOF integral might be incorrect";
  }

  float timeC = tTPC.getCorrectedTime(deltaT);                                                                                                    /// precise time estimate
  float timeErr = tTPC.constraint == TrackLocTPC::Constrained ? tTPC.timeErr : std::sqrt(tITS.getSigmaZ2() + tTPC.getSigmaZ2()) * mTPCVDrift0Inv; // estimate the error on time

  // outward refit
  auto& tracOut = trfit.getParamOut(); // this is a clone of ITS outward track already at the matching reference X
  auto& tofL = trfit.getLTIntegralOut();
  {
    float xtogo = 0;
    if (!tracOut.getXatLabR(o2::constants::geom::XTPCInnerRef, xtogo, mBz, o2::track::DirOutward) ||
        !propagator->PropagateToXBxByBz(tracOut, xtogo, MaxSnp, 10., mUseMatCorrFlag, &tofL)) {
      LOG(DEBUG) << "Propagation to inner TPC boundary X=" << xtogo << " failed, Xtr=" << tracOut.getX() << " snp=" << tracOut.getSnp();
      mMatchedTracks.pop_back(); // destroy failed track
      return false;
    }
    if (mVDriftCalibOn) {
      tracOut.updateCov(addErr2, o2::track::kSigTgl2);
    }
    float chi2Out = 0;
    auto posStart = tracOut.getXYZGlo();
    int retVal = mTPCRefitter->RefitTrackAsTrackParCov(tracOut, mTPCTracksArray[tTPC.sourceID].getClusterRef(), timeC * mTPCTBinMUSInv, &chi2Out, true, false); // outward refit
    if (retVal < 0) {
      LOG(DEBUG) << "Refit failed";
      mMatchedTracks.pop_back(); // destroy failed track
      return false;
    }
    auto posEnd = tracOut.getXYZGlo();
    // account path integrals
    float dX = posEnd.x() - posStart.x(), dY = posEnd.y() - posStart.y(), dZ = posEnd.z() - posStart.z(), d2XY = dX * dX + dY * dY;
    if (mFieldON) { // circular arc = 2*R*asin(dXY/2R)
      float b[3];
      o2::math_utils::Point3D<float> posAv(0.5 * (posEnd.x() + posStart.x()), 0.5 * (posEnd.y() + posStart.y()), 0.5 * (posEnd.z() + posStart.z()));
      propagator->getFieldXYZ(posAv, b);
      float curvH = std::abs(0.5f * tracOut.getCurvature(b[2])), arcXY = 1. / curvH * std::asin(curvH * std::sqrt(d2XY));
      d2XY = arcXY * arcXY;
    }
    auto lInt = std::sqrt(d2XY + dZ * dZ);
    tofL.addStep(lInt, tracOut.getP2Inv());
    tofL.addX2X0(lInt * mTPCmeanX0Inv);
    propagator->PropagateToXBxByBz(tracOut, o2::constants::geom::XTPCOuterRef, MaxSnp, 10., mUseMatCorrFlag, &tofL);
    /*
    LOG(INFO) <<  "TPC " << iTPC << " ITS " << iITS << " Refitted with chi2 = " << chi2Out;
    tracOut.print();
    tofL.print();
    */
  }

  trfit.setChi2Match(tpcMatchRec.chi2);
  trfit.setChi2Refit(chi2);
  trfit.setTimeMUS(timeC, timeErr);
  trfit.setRefTPC({unsigned(tTPC.sourceID), o2::dataformats::GlobalTrackID::TPC});
  trfit.setRefITS({unsigned(tITS.sourceID), o2::dataformats::GlobalTrackID::ITS});

  if (mMCTruthON) { // store MC info: we assign TPC track label and declare the match fake if the ITS and TPC labels are different (their fake flag is ignored)
    auto& lbl = mOutLabels.emplace_back(mTPCLblWork[iTPC]);
    lbl.setFakeFlag(mITSLblWork[iITS] != mTPCLblWork[iTPC]);
  }

  // if requested, fill the difference of ITS and TPC tracks tgl for vdrift calibation
  if (mHistoDTgl) {
    auto tglITS = tITS.getTgl();
    if (std::abs(tglITS) < mHistoDTgl->getXMax()) {
      auto dTgl = tglITS - tTPC.getTgl();
      mHistoDTgl->fill(tglITS, dTgl);
    }
  }
  //  trfit.print(); // DBG

  return true;
}

//______________________________________________
bool MatchTPCITS::refitTPCInward(o2::track::TrackParCov& trcIn, float& chi2, float xTgt, int trcID, float timeTB) const
{
  // inward refit
  constexpr float TolSNP = 0.99;
  const auto& tpcTrOrig = mTPCTracksArray[trcID];

  trcIn = tpcTrOrig.getOuterParam();
  chi2 = 0;

  auto propagator = o2::base::Propagator::Instance();
  int retVal = mTPCRefitter->RefitTrackAsTrackParCov(trcIn, tpcTrOrig.getClusterRef(), timeTB, &chi2, false, true); // inward refit with matrix reset
  if (retVal < 0) {
    LOG(WARNING) << "Refit failed";
    LOG(WARNING) << trcIn.asString();
    return false;
  }
  //
  // propagate to the inner edge of the TPC
  // Note: it is allowed to not reach the requested radius
  if (!propagator->PropagateToXBxByBz(trcIn, xTgt, MaxSnp, 2., mUseMatCorrFlag)) {
    LOG(DEBUG) << "Propagation to target X=" << xTgt << " failed, Xtr=" << trcIn.getX() << " snp=" << trcIn.getSnp() << " pT=" << trcIn.getPt();
    LOG(DEBUG) << trcIn.asString();
    return false;
  }
  return true;
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
  const float ROuter = outerLr.rRange.getMax() + 0.5f;

  auto propagator = o2::base::Propagator::Instance();

  for (int iTPC = 0; iTPC < (int)mTPCWork.size(); iTPC++) {
    auto& tTPC = mTPCWork[iTPC];
    if (isDisabledTPC(tTPC)) {
      // Popagate to the vicinity of the out layer. Note: the Z of the track might be uncertain,
      // in this case the material corrections will be correct only in the limit of their uniformity in Z,
      // which should be good assumption....
      float xTgt;
      if (!tTPC.getXatLabR(ROuter, xTgt, propagator->getNominalBz(), o2::track::DirInward) ||
          !propagator->PropagateToXBxByBz(tTPC, xTgt, MaxSnp, 2., mUseMatCorrFlag)) {
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
    return (trcA.tBracket.getMin() - trcB.tBracket.getMin()) < 0.;
  });

  return mTPCABIndexCache.size();
}

//______________________________________________
int MatchTPCITS::prepareInteractionTimes()
{
  // guess interaction times from various sources and relate with ITS rofs
  const float ft0Uncertainty = 0.5e-3;
  int nITSROFs = mITSROFTimes.size();
  mITSROFIntCandEntries.resize(nITSROFs);

  if (mFITInfo.size()) {
    int rof = 0;
    for (const auto& ft : mFITInfo) {
      if (!mFT0Params->isSelected(ft)) {
        continue;
      }
      auto fitTime = ft.getInteractionRecord().differenceInBCMS(mStartIR);
      // find corresponding ITS ROF, works both in cont. and trigg. modes (ignore T0 MeanTime within the BC)
      for (; rof < nITSROFs; rof++) {
        if (mITSROFTimes[rof] < fitTime) {
          continue;
        }
        if (fitTime >= mITSROFTimes[rof].getMin()) { // belongs to this ROF
          auto& ref = mITSROFIntCandEntries[rof];
          if (!ref.getEntries()) {
            ref.setFirstEntry(mInteractions.size()); // register entry
          }
          ref.changeEntriesBy(1); // increment counter
          mInteractions.emplace_back(ft.getInteractionRecord(), fitTime, ft0Uncertainty, rof, o2::detectors::DetID::FT0);
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

  int nIntCand = mInteractions.size();
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
    while ((iCRes = tTPC.tBracket.isOutside(mInteractions[iC].tBracket)) < 0 && ++iC < nIntCand) { // interaction precedes the track time-bracket
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
        }
      } while (++iCEnd < nIntCand && !tTPC.tBracket.isOutside(mInteractions[iCEnd].tBracket));

      auto lbl = mTPCLblWork[mTPCABIndexCache[itr]]; // tmp
      if (runAfterBurner(mTPCABIndexCache[itr], iCStart, iCEnd)) {
        lbl.print(); // tmp
        //tmp
        if (tTPC.matchID > MinusOne) {
          printf("AB Matching tree for TPC WID %d and IC %d : %d\n", mTPCABIndexCache[itr], iCStart, iCEnd);
          auto& llinks = mABTrackLinksList[tTPC.matchID];
          printABTracksTree(llinks);
        }
      }
    } else if (iCRes > 0) {
      continue; // TPC track precedes the interaction (means orphan track?), no need to check it
    } else {
      LOG(INFO) << "All interaction candidates precede track " << itr << " [" << tTPC.tBracket.getMin() << ":" << tTPC.tBracket.getMax() << "]";
      break; // all interaction candidates precede TPC track
    }
  }
  buildABCluster2TracksLinks();
  selectBestMatchesAB(); // validate matches which are good in both ways: TPCtrack->ITSclusters and ITSclusters->TPCtrack

  // tmp
  if (mDBGOut) {
    for (const auto& llinks : mABTrackLinksList) {
      dumpABTracksDebugTree(llinks);
    }
  }
  // tmp
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

    if (correctTPCTrack(topLink, tTPC, iCCand) < 0) { // correct track for assumed Z location calibration
      topLink.disable();
      continue;
    }
    /*
    // tmp
    LOG(INFO) << "Check track TPC mtc=" << tTPC.matchID << " int.cand. " << iCC
              << " [" << tTPC.tBracket.getMin() << ":" << tTPC.tBracket.getMax() << "] for interaction "
              << " [" << iCCand.tBracket.getMin() << ":" << iCCand.tBracket.getMax() << "]";
    */
    if (std::abs(topLink.getZ()) > mITSFiducialZCut) { // we can discard this seed
      topLink.disable();
    }
  }
  for (int ilr = NITSLayers; ilr > 0; ilr--) {
    int nextLinkID = abTrackLinksList.firstInLr[ilr];
    if (nextLinkID < 0) {
      break;
    }
    while (nextLinkID > MinusOne) {
      if (!mABTrackLinks[nextLinkID].isDisabled()) {
        checkABSeedFromLr(ilr, nextLinkID, abTrackLinksList);
      }
      nextLinkID = mABTrackLinks[nextLinkID].nextOnLr;
    }
    accountForOverlapsAB(ilr - 1);
    //    printf("After seeds of Lr %d:\n",ilr);
    //    printABTracksTree(abTrackLinksList); // tmp tmp
  }
  // disable link-list if neiher of seeds reached highest requested layer
  if (abTrackLinksList.lowestLayer > mParams->requireToReachLayerAB) {
    destroyLastABTrackLinksList();
    tTPC.matchID = MinusTen;
    return false;
  }

  return true;
}

//______________________________________________
void MatchTPCITS::accountForOverlapsAB(int lrSeed)
{
  // TODO
  LOG(WARNING) << "TODO";
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
  if (!seed.getXatLabR(lr.rRange.getMax(), xTgt, propagator->getNominalBz(), o2::track::DirInward) ||
      !propagator->PropagateToXBxByBz(seed, xTgt, MaxSnp, 2., mUseMatCorrFlag)) {
    return 0;
  }
  auto icCandID = seedLink.icCandID;

  // fetch clusters reference object for the ITS ROF corresponding to interaction candidate
  const auto& clRefs = *static_cast<const ITSChipClustersRefs*>(mInteractions[icCandID].clRefPtr);
  const float nSigmaZ = 5., nSigmaY = 5.;             // RS TODO: convert to settable parameter
  const float YErr2Extra = 0.1 * 0.1;                 // // RS TODO: convert to settable parameter
  float sna, csa;                                     // circle parameters for B ON data
  float zDRStep = -seed.getTgl() * lr.rRange.delta(); // approximate Z span when going from layer rMin to rMax
  float errZ = std::sqrt(seed.getSigmaZ2());
  if (lr.zRange.isOutside(seed.getZ(), nSigmaZ * errZ + std::abs(zDRStep))) {
    // printf("Lr %d missed by Z = %.2f + %.3f\n", lrTgt, seed.getZ(), nSigmaZ * errZ + std::abs(zDRStep)); // tmp
    return 0;
  }
  std::vector<int> chipSelClusters; // preliminary cluster candidates //RS TODO do we keep this local / consider array instead of vector
  o2::math_utils::CircleXYf_t trcCircle;
  o2::math_utils::IntervalXYf_t trcLinPar; // line parameters for B OFF data
  // approximate errors
  float errY = std::sqrt(seed.getSigmaY2() + YErr2Extra), errYFrac = errY * mRGHelper.ladderWidthInv(), errPhi = errY * lr.rInv;
  if (mFieldON) {
    seed.getCircleParams(propagator->getNominalBz(), trcCircle, sna, csa);
  } else {
    seed.getLineParams(trcLinPar, sna, csa);
  }
  float xCurr, yCurr;
  o2::math_utils::rotateZ(seed.getX(), seed.getY(), xCurr, yCurr, sna, csa);
  float phi = std::atan2(yCurr, xCurr); // RS: TODO : can we use fast atan2 here?
  // find approximate ladder and chip_in_ladder corresponding to this track extrapolation
  int nLad2Check = 0, ladIDguess = lr.getLadderID(phi), chipIDguess = lr.getChipID(seed.getZ() + 0.5 * zDRStep);
  std::array<int, MaxLadderCand> lad2Check;
  nLad2Check = mFieldON ? findLaddersToCheckBOn(lrTgt, ladIDguess, trcCircle, errYFrac, lad2Check) : findLaddersToCheckBOff(lrTgt, ladIDguess, trcLinPar, errYFrac, lad2Check);

  const auto& tTPC = mTPCWork[llist.trackID]; // tmp
  o2::MCCompLabel lblTrc;
  if (mMCTruthON) {
    lblTrc = mTPCTrkLabels[tTPC.sourceID]; // tmp
  }
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
      if (lad.chips[chipID].zRange.isOutside(zCross, nSigmaZ * errZ)) {
        continue;
      }
      const auto& clRange = clRefs.chipRefs[lad.chips[chipID].id];
      if (!clRange.getEntries()) {
        continue;
      }
      /*
      // tmp
      printf("Lr %d #%d/%d LadID: %d (phi:%+d) ChipID: %d [%d Ncl: %d from %d] (rRhi:%d Z:%+d[%+.1f:%+.1f]) | %+.3f %+.3f -> %+.3f %+.3f %+.3f (zErr: %.3f)\n",
             lrTgt, ilad, ich, ladID, lad.isPhiOutside(phi, errPhi), chipID,
             chipGID, clRange.getEntries(), clRange.getFirstEntry(),
             lad.chips[chipID].xyEdges.seenByCircle(trcCircle, errYFrac), lad.chips[chipID].zRange.isOutside(zCross, 3 * errZ), lad.chips[chipID].zRange.getMin(), lad.chips[chipID].zRange.getMax(),
             xCurr, yCurr, xCross, yCross, zCross, errZ);
      */
      // track Y error in chip frame
      float errYcalp = errY * (csa * chipC.csAlp + sna * chipC.snAlp); // sigY_rotate(from alpha0 to alpha1) = sigY * cos(alpha1 - alpha0);
      float tolerZ = errZ * nSigmaZ, tolerY = errYcalp * nSigmaY;
      float yTrack = -xCross * chipC.snAlp + yCross * chipC.csAlp;                                            // track-chip crossing Y in chip frame
      if (!preselectChipClusters(chipSelClusters, clRange, clRefs, yTrack, zCross, tolerY, tolerZ, lblTrc)) { // select candidate clusters for this chip
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
        const auto& cls = mITSClustersArray[clID];
        auto chi2 = trcLC.getPredictedChi2(cls);
        /*
        const auto lab = mITSClsLabels->getLabels(clID)[0];                                           // tmp
        LOG(INFO) << "cl " << cntc++ << " ClLbl:" << lab << " TrcLbl" << lblTrc << " chi2 = " << chi2 << " chipGID: " << lad.chips[chipID].id; // tmp
 */
        if (chi2 > mParams->cutABTrack2ClChi2) {
          continue;
        }
        int lnkID = registerABTrackLink(llist, trcLC, icCandID, lrTgt, seedID, clID, chi2); // add new link with track copy
        if (lnkID > MinusOne) {
          auto& link = mABTrackLinks[lnkID];
          link.ladderID = ladID; // store ladderID for double hit check
#ifdef _ALLOW_DEBUG_AB_
          link.seed = link;
#endif
          link.update(cls);
          link.chi2 = chi2 + mABTrackLinks[seedID].chi2; // don't use seedLink since it may be changed are reallocation
          mABTrackLinks[seedID].nDaughters++;            // idem, don't use seedLink.nDaughters++;

          if (lrTgt < llist.lowestLayer) {
            llist.lowestLayer = lrTgt; // update lowest layer reached
          }
          //   printf("Added chi2 %.3f @ lr %d as %d\n",link.chi2, lrTgt, lnkID); // tmp tmp
        }
      }
    }
  }
  return mABTrackLinks[seedID].nDaughters;
}

//______________________________________________
void MatchTPCITS::mergeABSeedsOnOverlaps(int ilrPar, ABTrackLinksList& llist)
{
  // try to merge seeds which may come from double hit on the layer. Parent layer is provided,
  // The merged seeds will be added unconditionally (if they pass chi2 cut)
  int linksFilled = mABTrackLinks.size();
  int lrID = ilrPar - 1;
  int topLinkID = llist.firstInLr[lrID];
  while (topLinkID > MinusOne) {
    const auto& topLink = mABTrackLinks[topLinkID];
    int runLinkID = topLink.nextOnLr; // running link ID
    while (runLinkID > MinusOne) {
      const auto& runLink = mABTrackLinks[runLinkID];
      // to be considered as double hit candidate 2 links must have common parent and neighbouring ladders
      while (1) {
        if (topLink.parentID != runLink.parentID) {
          break;
        }
        int dLadder = topLink.ladderID - runLink.ladderID;
        if (dLadder == 0 || (dLadder > 1 && dLadder != mRGHelper.layers[lrID].ladders.size() - 1)) { // difference must be 1 or Nladders-1
          break;
        }
      }
    }
  }
}

//______________________________________________
int MatchTPCITS::findLaddersToCheckBOn(int ilr, int lad0, const o2::math_utils::CircleXYf_t& circle, float errYFrac,
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
int MatchTPCITS::findLaddersToCheckBOff(int ilr, int lad0, const o2::math_utils::IntervalXYf_t& trcLinPar, float errYFrac,
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
    if (trList.trackID <= MinusOne) {
      LOG(ERROR) << "ABTrackLinksList does not point on tracks, impossible"; // tmp
      continue;
    }
    // register all clusters of all seeds starting from the innermost layer
    for (int lr = trList.lowestLayer; lr <= mParams->requireToReachLayerAB; lr++) {
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
            auto& newClLink = mABClusterLinks.emplace_back(finalTrackLinkIdx, ils); // create new link

            //>> insert new link in the list of other links for this cluster ordering in track final quality
            int clLinkIdx = mABClusterLinkIndex[clID];
            int prevClLinkIdx = MinusOne;
            while (clLinkIdx > MinusOne) {
              auto& clLink = mABClusterLinks[clLinkIdx];
              const auto& competingTrackLink = mABTrackLinks[clLink.linkedABTrack];
              if (isBetter(finalTrackLink.chi2Norm(), competingTrackLink.chi2Norm())) {
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
int MatchTPCITS::registerABTrackLink(ABTrackLinksList& llist, const o2::track::TrackParCov& src, int ic, int lr, int parentID, int clID, float chi2Cl)
{
  // registers new ABLink on the layer, assigning provided kinematics. The link will be registered in a
  // way preserving the quality ordering of the links on the layer
  int lnkID = mABTrackLinks.size();
  if (llist.firstInLr[lr] == MinusOne) { // no links on this layer yet
    if (lr == NITSLayers) {
      llist.firstLinkID = lnkID; // register very 1st link
    }
    llist.firstInLr[lr] = lnkID;
    mABTrackLinks.emplace_back(src, ic, lr, parentID, clID);
    return lnkID;
  }
  // add new link sorting links of this layer in quality

  int count = 0, nextID = llist.firstInLr[lr], topID = MinusOne;
  do {
    auto& nextLink = mABTrackLinks[nextID];
    count++;
    // if clID==-10, this is a special link on the dummy layer, corresponding to particular Interaction Candidate, in this case
    // it does not matter if we add new link before or after the preceding link of the same dummy layer
    if (clID == MinusTen || isBetter(mABTrackLinks[parentID].chi2NormPredict(chi2Cl), nextLink.chi2Norm())) { // need to insert new link before nextLink
      if (count < mMaxABLinksOnLayer) {                                                                       // will insert in front of nextID
        auto& newLnk = mABTrackLinks.emplace_back(src, ic, lr, parentID, clID);
        newLnk.nextOnLr = nextID; // point to the next one
        if (topID > MinusOne) {
          mABTrackLinks[topID].nextOnLr = lnkID; // point from previous one
        } else {
          llist.firstInLr[lr] = lnkID; // flag as best on the layer
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
    mABTrackLinks.emplace_back(src, ic, lr, parentID, clID);
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
  if (tTPC.constraint == TrackLocTPC::Constrained) {
    return 0.f;
  }
  auto tpcTrOrig = mTPCTracksArray[tTPC.sourceID];
  float timeIC = cand.tBracket.mean();
  float driftErr = cand.tBracket.delta() * mTPCBin2Z;

  // we use this for refit
  /*
  {
    float r = std::sqrt(trc.getX()*trc.getX() + trc.getY()*trc.getY());
    float chi2 = 0;
    bool res = refitTPCInward(trc, chi2, r, tTPC.sourceID(), timeIC );
    if (!res) {
      return -1;
    }
    float xTgt;
    auto propagator = o2::base::Propagator::Instance();
    if (!trc.getXatLabR(r, xTgt, propagator->getNominalBz(), o2::track::DirInward) ||
 !propagator->PropagateToXBxByBz(trc, xTgt, MaxSnp, 2., mUseMatCorrFlag)) {
      return -1;
    }
  }
  */
  // if interaction time precedes the initial assumption on t0 (i.e. timeIC < timeTrc),
  // the track actually was drifting longer, i.e. tracks should be shifted closer to the CE
  float dDrift = (timeIC - tTPC.time0) * mTPCBin2Z;
  float zz = tTPC.getZ() + (tpcTrOrig.hasASideClustersOnly() ? dDrift : -dDrift);                                 // tmp
  LOG(INFO) << "CorrTrack Z=" << trc.getZ() << " (zold= " << zz << ") at TIC= " << timeIC << " Ttr= " << tTPC.time0; // tmp

  // we use this w/o refit
  //  /*
  {
    trc.setZ(tTPC.getZ() + (tTPC.constraint == TrackLocTPC::ASide ? dDrift : -dDrift));
  }
  //  */
  trc.setCov(trc.getSigmaZ2() + driftErr * driftErr, o2::track::kSigZ2);

  return driftErr;
}

//______________________________________________
void MatchTPCITS::fillClustersForAfterBurner(ITSChipClustersRefs& refCont, int rofStart, int nROFs)
{
  // Prepare unused clusters of given ROFs range for matching in the afterburner
  // Note: normally only 1 ROF needs to be filled (nROFs==1 ) unless we want
  // to account for interaction on the boundary of 2 rofs, which then may contribute to both ROFs.
  int first = mITSClusterROFRec[rofStart].getFirstEntry(), last = first;
  for (int ir = nROFs; ir--;) {
    last += mITSClusterROFRec[rofStart + ir].getNEntries();
  }
  refCont.clear();
  auto& idxSort = refCont.clusterID;
  for (int icl = first; icl < last; icl++) {
    if (mABClusterLinkIndex[icl] != MinusTen) { // clusters with MinusOne are used in main matching
      idxSort.push_back(icl);
    }
  }
  // sort in chip, Z
  sort(idxSort.begin(), idxSort.end(), [clusArr = mITSClustersArray](int i, int j) {
    const auto &clI = clusArr[i], &clJ = clusArr[j];
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
    const auto& clus = mITSClustersArray[idxSort[icl]];
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

  int nTrackLinkList = mABTrackLinksList.size(), nremaining = 0;
  do {
    nValidated = nremaining = 0;
    for (int ils = 0; ils < nTrackLinkList; ils++) {
      auto& trList = mABTrackLinksList[ils];
      if (trList.isValidated() || trList.isDisabled()) {
        continue;
      }
      nremaining++;
      if (validateABMatch(ils)) {
        nValidated++;
        continue;
      }
    }
    printf("iter %d Validated %d of %d remaining AB matches\n", iter, nValidated, nremaining);
    iter++;
  } while (nValidated);
}

//______________________________________________
bool MatchTPCITS::validateABMatch(int ilink)
{
  // make sure the preference of the set of cluster by this link is reciprocal

  buildBestLinksList(ilink);

  auto& trList = mABTrackLinksList[ilink];

  // pick the best TPC->ITS links branch
  int bestHypID = trList.firstInLr[trList.lowestLayer];
  const auto& bestHyp = mABTrackLinks[bestHypID]; // best hypothesis

  LOG(INFO) << "validateABMatch " << ilink;
  printABTracksTree(trList);

  int parID = bestHypID;
  int headID = parID;
  while (1) {
    const auto& lnk = mABTrackLinks[parID];
    LOG(INFO) << " *link " << parID << "(head " << headID << " cl: " << lnk.clID << " on Lr:" << int(lnk.layerID) << ")"; // tmp
    if (lnk.clID > MinusOne) {
      int clLnkIdx = mABClusterLinkIndex[lnk.clID]; // id of ITSclus->TPCtracks quality records
      if (clLnkIdx < Zero) {
        LOG(ERROR) << "AB-referred cluster " << lnk.clID << " does not have ABlinks record";
        continue;
      }

      // navigate to best *available* ABlTrackLinkList contributed by this cluster
      while (mABTrackLinksList[mABClusterLinks[clLnkIdx].linkedABTrackList].isValidated()) {
        clLnkIdx = mABClusterLinks[clLnkIdx].nextABClusterLink;
      }
      if (clLnkIdx < Zero) {
        LOG(ERROR) << "AB-referred cluster " << lnk.clID << " does not have ABlinks record, exhausted?";
        continue;
      }

      const auto& linkCl = mABClusterLinks[clLnkIdx];

      //<< tmp : for printout only
      std::stringstream mcStr;
      mcStr << " **cl" << lnk.clID << " -> seed:" << linkCl.linkedABTrack << '/' << linkCl.linkedABTrackList << " | ";
      // find corresponding TPC track
      int linkID = linkCl.linkedABTrack;
      while (1) {
        const auto& linkTrack = mABTrackLinks[linkID];
        linkID = linkTrack.parentID;
        if (linkTrack.isDummyTop()) {
          break;
        }
      }
      mcStr << "{TPCwrk: " << linkID << "} ";
      if (mITSClsLabels) {
        auto lbls = mITSClsLabels->getLabels(lnk.clID);
        for (const auto lbl : lbls) {
          if (lbl.isValid()) {
            mcStr << '[' << lbl.getSourceID() << '/' << lbl.getEventID() << '/' << (lbl.isFake() ? '-' : '+') << std::setw(6) << lbl.getTrackID() << ']';
          } else {
            mcStr << (lbl.isNoise() ? "[noise]" : "[unset]");
          }
        }
      }
      LOG(INFO) << mcStr.str();
      //>> tmp

      if (linkCl.linkedABTrack != headID) { // best link for this cluster differs from the link we are checking
        return false;
      }
    } else if (lnk.isDummyTop()) { // top layer reached, this is winner
      break;
    }
    parID = lnk.parentID; // check next cluster of the same link
  }
  LOG(INFO) << "OK validateABMatch " << ilink;
  trList.validate();
  return true;
}

//______________________________________________
void MatchTPCITS::buildBestLinksList(int ilink)
{
  ///< build ordered list of links for given ABTrackLinksList, including those finishing on higher layers
  auto& trList = mABTrackLinksList[ilink];

  // pick the best TPC->ITS links branch
  //  int bestHypID = trList.firstInLr[trList.lowestLayer];
  //

  int start = mABBestLinks.size(); // order links will be added started from here
  // 1) the links on the lowestLayer are already sorted, just copy them
  int lowestLayer = trList.lowestLayer;
  int nextID = trList.firstInLr[lowestLayer];
  int prevOrdLink = MinusOne, newOrdLinkID = MinusOne;
  int nHyp = 0;
  while (nextID > MinusOne && nHyp < mMaxABFinalHyp) {
    newOrdLinkID = mABBestLinks.size();
    auto& best = mABBestLinks.emplace_back(nextID);
    nHyp++;
    if (prevOrdLink != MinusOne) {
      mABBestLinks[prevOrdLink].nextLinkID = newOrdLinkID; // register reference on the new link from previous one
    }
    prevOrdLink = newOrdLinkID;
    nextID = mABTrackLinks[nextID].nextOnLr;
  }
  trList.bestOrdLinkID = start;

  // now check if shorter seed links from layers above need to be considered as final track candidates
  while (++lowestLayer <= mParams->requireToReachLayerAB) {
    nextID = trList.firstInLr[lowestLayer];

    while (nextID > MinusOne) {
      auto& candHyp = mABTrackLinks[nextID];
      // compare candHyp with already stored best hypotheses (keeping in mind that within a single layer they are already sorted in quality

      int nextBest = trList.bestOrdLinkID, prevBest = MinusOne;
      while (nextBest > MinusOne) {
        const auto& bestOrd = mABBestLinks[nextBest];
        const auto& bestHyp = mABTrackLinks[bestOrd.trackLinkID];
        if (isBetter(candHyp.chi2Norm(), bestHyp.chi2Norm())) { // need to insert new candidate before bestOrd
          break;
        }
        prevBest = nextBest;
        nextBest = bestOrd.nextLinkID;
      }

      bool reuseWorst = false, newIsWorst = (nextBest <= MinusOne);
      if (nHyp == mMaxABFinalHyp) { // max number of best hypotheses reached
        if (newIsWorst) {           // the bestHyp is worse than all already registered hypotheses,
          break;                    // ignore candHyp and all remaining hypotheses on this layer since they are worse
        }
        reuseWorst = true; // we don't add new ordLink slot to the pool but reuse the worst one
      }

      int newID = mABBestLinks.size();
      if (reuseWorst) { // navigate to worst hypothesis link in order to use it for registration of new ordLink
        int nextWorst = nextBest, prevWorst = prevBest;
        while (mABBestLinks[nextWorst].nextLinkID > MinusOne) { // navigate to worst hypothesis link
          prevWorst = nextWorst;
          nextWorst = mABBestLinks[nextWorst].nextLinkID;
        }
        newID = nextWorst;
        if (prevWorst > MinusOne) { // detach the reused slot from the superior slot refferring to suppressed one
          mABBestLinks[prevWorst].nextLinkID = MinusOne;
        }
      } else { // add new slot to the pool
        mABBestLinks.emplace_back();
        nHyp++;
      }
      mABBestLinks[newID].trackLinkID = nextID;
      if (newID != nextBest) { // if we did not reuse the worst link, register the worse one here
        mABBestLinks[newID].nextLinkID = nextBest;
      }
      if (prevBest > MinusOne) {
        mABBestLinks[prevBest].nextLinkID = newID; // register new ordLink in its superior link
      } else {                                     // register new ordLink as best link
        trList.bestOrdLinkID = newID;
      }

      nextID = candHyp.nextOnLr;
    }
  }
}

void MatchTPCITS::refitABTrack(int ibest) const
{
  auto propagator = o2::base::Propagator::Instance();
  const float maxStep = 2.f; // max propagation step (TODO: tune)

  int ncl = 0;
  const auto& lnk0 = mABTrackLinks[ibest];
  o2::track::TrackParCov trc = lnk0;
  trc.resetCovariance();
  const auto& cl0 = mITSClustersArray[lnk0.clID];
  trc.setY(cl0.getY());
  trc.setZ(cl0.getZ());
  trc.setCov(cl0.getSigmaY2(), o2::track::kSigY2);
  trc.setCov(cl0.getSigmaZ2(), o2::track::kSigZ2);
  trc.setCov(cl0.getSigmaYZ(), o2::track::kSigZY); // for the 1st point we don't need any fit
  ibest = lnk0.parentID;
  float chi2 = 0;
  while (ibest > MinusOne) {
    const auto& lnk = mABTrackLinks[ibest];
    if (!trc.rotate(lnk.getAlpha()) ||
        !propagator->propagateToX(trc, lnk.getX(), propagator->getNominalBz(), MaxSnp, maxStep, mUseMatCorrFlag, nullptr)) {
      LOG(WARNING) << "Failed to rotate to " << lnk.getAlpha() << " or propagate to " << lnk.getX();
      LOG(WARNING) << trc.asString();
      break;
    }
    if (lnk.clID > MinusOne) {
      const auto& cl = mITSClustersArray[lnk.clID];
      chi2 += trc.getPredictedChi2(cl);
      if (!trc.update(cl)) {
        LOG(WARNING) << "Failed to update by " << cl;
        LOG(WARNING) << trc.asString();
        break;
      }
      ncl++;
    } else if (lnk.isDummyTop()) { // dummy layer to which TPC track was propagated
      LOG(INFO) << "end of fit: chi2=" << chi2 << " ncl= " << ncl;
      LOG(INFO) << "TPCtrc: " << lnk.asString();
      LOG(INFO) << "ITStrc: " << trc.asString();
      break;
    }
    ibest = lnk.parentID;
  }
}

//______________________________________________
void MatchTPCITS::setITSROFrameLengthInBC(int nbc)
{
  mITSROFrameLengthInBC = nbc;
  mITSROFrameLengthMUS = nbc * o2::constants::lhc::LHCBunchSpacingNS * 1e-3;
  mITSROFrameLengthMUSInv = 1. / mITSROFrameLengthMUS;
}

//___________________________________________________________________
void MatchTPCITS::setBunchFilling(const o2::BunchFilling& bf)
{
  mBunchFilling = bf;
  // find closest (from above) filled bunch
  int minBC = bf.getFirstFilledBC(), maxBC = bf.getLastFilledBC();
  if (minBC < 0) {
    throw std::runtime_error("Bunch filling is not set in MatchTPCITS");
  }
  int bcAbove = minBC;
  for (int i = o2::constants::lhc::LHCMaxBunches; i--;) {
    if (bf.testBC(i)) {
      bcAbove = i;
    }
    mClosestBunchAbove[i] = bcAbove;
  }
  int bcBelow = maxBC;
  for (int i = 0; i < o2::constants::lhc::LHCMaxBunches; i++) {
    if (bf.testBC(i)) {
      bcBelow = i;
    }
    mClosestBunchBelow[i] = bcBelow;
  }
}

//___________________________________________________________________
MatchTPCITS::BracketIR MatchTPCITS::tBracket2IRBracket(const BracketF tbrange)
{
  // convert time bracket to IR bracket
  o2::InteractionRecord irMin(mStartIR), irMax(mStartIR);
  if (tbrange.getMin() > 0) {
    irMin += o2::InteractionRecord(tbrange.getMin() * 1000.f); // time in ns is needed
  }
  irMax += o2::InteractionRecord(tbrange.getMax() * 1000.f);
  irMax++; // to account for rounding
  int bc = mClosestBunchAbove[irMin.bc];
  if (bc < irMin.bc) {
    irMin.orbit++;
  }
  irMin.bc = bc;
  bc = mClosestBunchBelow[irMax.bc];
  if (bc > irMax.bc) {
    if (irMax.orbit == 0) {
      bc = 0;
    } else {
      irMax.orbit--;
    }
  }
  irMax.bc = bc;
  return {irMin, irMax};
}

//______________________________________________
void MatchTPCITS::removeTPCfromITS(int tpcID, int itsID)
{
  ///< remove reference to tpcID track from itsID track matches
  auto& tITS = mITSWork[itsID];
  if (isValidatedITS(tITS)) {
    return;
  }
  int topID = MinusOne, next = tITS.matchID; // ITS MatchRecord
  while (next > MinusOne) {
    auto& rcITS = mMatchRecordsITS[next];
    if (rcITS.partnerID == tpcID) {
      if (topID < 0) {
        tITS.matchID = rcITS.nextRecID;
      } else {
        mMatchRecordsITS[topID].nextRecID = rcITS.nextRecID;
      }
      return;
    }
    topID = next;
    next = rcITS.nextRecID;
  }
}

//______________________________________________
void MatchTPCITS::removeITSfromTPC(int itsID, int tpcID)
{
  ///< remove reference to itsID track from matches of tpcID track
  auto& tTPC = mTPCWork[tpcID];
  if (isValidatedTPC(tTPC)) {
    return;
  }
  int topID = MinusOne, next = tTPC.matchID;
  while (next > MinusOne) {
    auto& rcTPC = mMatchRecordsTPC[next];
    if (rcTPC.partnerID == itsID) {
      if (topID < 0) {
        tTPC.matchID = rcTPC.nextRecID;
      } else {
        mMatchRecordsTPC[topID].nextRecID = rcTPC.nextRecID;
      }
      return;
    }
    topID = next;
    next = rcTPC.nextRecID;
  }
}

//______________________________________________
void MatchTPCITS::flagUsedITSClusters(const o2::its::TrackITS& track, int rofOffset)
{
  // flag clusters used by this track
  int clEntry = track.getFirstClusterEntry();
  for (int icl = track.getNumberOfClusters(); icl--;) {
    mABClusterLinkIndex[rofOffset + mITSTrackClusIdx[clEntry++]] = MinusTen;
  }
}
//__________________________________________________________
int MatchTPCITS::preselectChipClusters(std::vector<int>& clVecOut, const ClusRange& clRange, const ITSChipClustersRefs& clRefs,
                                       float trackY, float trackZ, float tolerY, float tolerZ,
                                       const o2::MCCompLabel& lblTrc) const // TODO lbl is not needed
{
  clVecOut.clear();
  int icID = clRange.getFirstEntry();
  for (int icl = clRange.getEntries(); icl--;) { // note: clusters within a chip are sorted in Z
    int clID = clRefs.clusterID[icID++];         // so, we go in clusterID increasing direction
    const auto& cls = mITSClustersArray[clID];
    float dz = trackZ - cls.getZ();
    auto label = mITSClsLabels->getLabels(clID)[0]; // tmp
    //    if (!(label == lblTrc)) {
    //      continue; // tmp
    //    }
    LOG(DEBUG) << "cl" << icl << '/' << clID << " " << label
               << " dZ: " << dz << " [" << tolerZ << "| dY: " << trackY - cls.getY() << " [" << tolerY << "]";
    if (dz > tolerZ) {
      float clsZ = cls.getZ();
      LOG(DEBUG) << "Skip the rest since " << trackZ << " > " << clsZ << "\n";
      break;
    } else if (dz < -tolerZ) {
      LOG(DEBUG) << "Skip cluster dz=" << dz << " Ztr=" << trackZ << " zCl=" << cls.getZ();
      continue;
    }
    if (fabs(trackY - cls.getY()) > tolerY) {
      LOG(DEBUG) << "Skip cluster dy= " << trackY - cls.getY() << " Ytr=" << trackY << " yCl=" << cls.getY();
      continue;
    }
    clVecOut.push_back(clID);
  }
  return clVecOut.size();
}

//______________________________________________
void MatchTPCITS::cleanAfterBurnerClusRefCache(int currentIC, int& startIC)
{
  // check if some of cached cluster reference from tables startIC to currentIC can be released,
  // they will be necessarily in front slots of the mITSChipClustersRefs
  while (startIC < currentIC && mInteractions[currentIC].tBracket.getMin() - mInteractions[startIC].tBracket.getMax() > MinTBToCleanCache) {
    LOG(INFO) << "CAN REMOVE CACHE FOR " << startIC << " curent IC=" << currentIC;
    while (mInteractions[startIC].clRefPtr == &mITSChipClustersRefs.front()) {
      LOG(INFO) << "Reset cache pointer" << mInteractions[startIC].clRefPtr << " for IC=" << startIC;
      mInteractions[startIC++].clRefPtr = nullptr;
    }
    LOG(INFO) << "Reset cache slot " << &mITSChipClustersRefs.front();
    mITSChipClustersRefs.pop_front();
  }
}

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

  mTimer[SWDBG].Start(false);

  auto& trackITS = mITSWork[itsID];
  auto& trackTPC = mTPCWork[tpcID];
  if (chi2 < 0.) { // need to recalculate
    chi2 = getPredictedChi2NoZ(trackITS, trackTPC);
  }
  o2::MCCompLabel lblITS, lblTPC;
  (*mDBGOut) << "match"
             << "tf=" << mTFCount << "chi2Match=" << chi2 << "its=" << trackITS << "tpc=" << trackTPC;
  if (mMCTruthON) {
    lblITS = mITSLblWork[itsID];
    lblTPC = mTPCLblWork[tpcID];
    (*mDBGOut) << "match"
               << "itsLbl=" << lblITS << "tpcLbl=" << lblTPC;
  }
  (*mDBGOut) << "match"
             << "rejFlag=" << rejFlag << "\n";

  mTimer[SWDBG].Stop();
}

//______________________________________________
void MatchTPCITS::dumpWinnerMatches()
{
  ///< write winner matches into debug tree

  mTimer[SWDBG].Start(false);

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
               << "tf=" << mTFCount << "chi2Match=" << itsMatchRec.chi2 << "chi2Refit=" << mWinnerChi2Refit[iits] << "its=" << tITS << "tpc=" << tTPC;

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
  mTimer[SWDBG].Stop();
}

#endif
