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

using namespace o2::globaltracking;

using MatrixDSym4 = ROOT::Math::SMatrix<double, 4, 4, ROOT::Math::MatRepSym<double, 4>>;
using MatrixD4 = ROOT::Math::SMatrix<double, 4, 4, ROOT::Math::MatRepStd<double, 4>>;

//______________________________________________
void MatchTPCITS::run()
{
  ///< perform matching for provided input
  if (!mInitDone) {
    LOG(FATAL) << "init() was not done yet" << FairLogger::endl;
  }

  mTimerTot.Start();

  prepareTPCTracks();
  prepareITSTracks();
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
void MatchTPCITS::init()
{
  ///< perform initizalizations, precalculate what is needed
  if (mInitDone) {
    LOG(ERROR) << "Initialization was already done" << FairLogger::endl;
    return;
  }

  // make sure T2GRot matrices are loaded into ITS geometry helper
  o2::ITS::GeometryTGeo::Instance()->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2GRot));

  mYMaxAtXRef = mXRef * std::tan(o2::constants::math::SectorSpanRad * 0.5); ///< max Y in the sector at reference X
  mSectEdgeMargin2 = mCrudeAbsDiffCut[o2::track::kY] * mCrudeAbsDiffCut[o2::track::kY]; ///< precalculated ^2

  const auto& gasParam = o2::TPC::ParameterGas::defaultInstance();
  const auto& elParam = o2::TPC::ParameterElectronics::defaultInstance();
  const auto& detParam = o2::TPC::ParameterDetector::defaultInstance();
  mTPCTBinMUS = elParam.getZBinWidth();
  mTPCVDrift0 = gasParam.getVdrift();
  mTPCZMax = detParam.getTPClength();

  assert(mITSROFrameLengthMUS > 0.f);
  mITSROFramePhaseOffset = mITSROFrameOffsetMUS / mITSROFrameLengthMUS;
  mITSROFrame2TPCBin = mITSROFrameLengthMUS / mTPCTBinMUS;
  mTPCBin2ITSROFrame = 1. / mITSROFrame2TPCBin;
  mTPCBin2Z = mTPCTBinMUS * mTPCVDrift0;
  mZ2TPCBin = 1. / mTPCBin2Z;
  mTPCVDrift0Inv = 1. / mTPCVDrift0;
  mNTPCBinsFullDrift = mTPCZMax * mZ2TPCBin;

  mTPCTimeEdgeTSafeMargin = z2TPCBin(mTPCTimeEdgeZSafeMargin);

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
    LOG(ERROR) << "Output tree is not attached, matched tracks will not be stored" << FairLogger::endl;
  }

#ifdef _ALLOW_DEBUG_TREES_
  // debug streamer
  if (mDBGFlags) {
    mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(mDebugTreeFileName.data(), "recreate");
  }
#endif

  mMatchRecordsTPC.clear();
  mMatchRecordsITS.clear();
  mMatchesTPC.clear();
  mMatchesITS.clear();

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
  LOG(INFO) << "Selecting best matches for " << mMatchesTPC.size() << " TPC tracks" << FairLogger::endl;

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
    LOG(FATAL) << "ITS tracks data input tree is not set" << FairLogger::endl;
  }

  if (!mTreeTPCTracks) {
    LOG(FATAL) << "TPC tracks data input tree is not set" << FairLogger::endl;
  }

  if (!mTreeITSClusters) {
    LOG(FATAL) << "ITS clusters data input tree is not set" << FairLogger::endl;
  }

  if (!mTreeITSTracks->GetBranch(mITSTrackBranchName.data())) {
    LOG(FATAL) << "Did not find ITS tracks branch " << mITSTrackBranchName << " in the input tree" << FairLogger::endl;
  }
  mTreeITSTracks->SetBranchAddress(mITSTrackBranchName.data(), &mITSTracksArrayInp);
  LOG(INFO) << "Attached ITS tracks " << mITSTrackBranchName << " branch with " << mTreeITSTracks->GetEntries()
            << " entries" << FairLogger::endl;

  if (!mTreeTPCTracks->GetBranch(mTPCTrackBranchName.data())) {
    LOG(FATAL) << "Did not find TPC tracks branch " << mTPCTrackBranchName << " in the input tree" << FairLogger::endl;
  }
  mTreeTPCTracks->SetBranchAddress(mTPCTrackBranchName.data(), &mTPCTracksArrayInp);
  LOG(INFO) << "Attached TPC tracks " << mTPCTrackBranchName << " branch with " << mTreeTPCTracks->GetEntries()
            << " entries" << FairLogger::endl;

  if (!mTreeITSClusters->GetBranch(mITSClusterBranchName.data())) {
    LOG(FATAL) << "Did not find ITS clusters branch " << mITSClusterBranchName << " in the input tree"
               << FairLogger::endl;
  }
  mTreeITSClusters->SetBranchAddress(mITSClusterBranchName.data(), &mITSClustersArrayInp);
  LOG(INFO) << "Attached ITS clusters " << mITSClusterBranchName << " branch with " << mTreeITSClusters->GetEntries()
            << " entries" << FairLogger::endl;

  // is there MC info available ?
  if (mTreeITSTracks->GetBranch(mITSMCTruthBranchName.data())) {
    mTreeITSTracks->SetBranchAddress(mITSMCTruthBranchName.data(), &mITSTrkLabels);
    LOG(INFO) << "Found ITS Track MCLabels branch " << mITSMCTruthBranchName << FairLogger::endl;
  }
  // is there MC info available ?
  if (mTreeTPCTracks->GetBranch(mTPCMCTruthBranchName.data())) {
    mTreeTPCTracks->SetBranchAddress(mTPCMCTruthBranchName.data(), &mTPCTrkLabels);
    LOG(INFO) << "Found TPC Track MCLabels branch " << mTPCMCTruthBranchName << FairLogger::endl;
  }

  mMCTruthON = (mITSTrkLabels && mTPCTrkLabels);
  mCurrTPCTracksTreeEntry = -1;
  mCurrITSTracksTreeEntry = -1;
}

//______________________________________________
bool MatchTPCITS::prepareTPCTracks()
{
  ///< load next chunk of TPC data and prepare for matching
  mMatchRecordsTPC.clear();

  if (!loadTPCTracksNextChunk()) {
    return false;
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

    o2::TPC::TrackTPC& trcOrig = (*mTPCTracksArrayInp)[it];

    // make sure the track was propagated to inner TPC radius at the ref. radius
    if (trcOrig.getX() > mXTPCInnerRef + 0.1)
      continue; // failed propagation to inner TPC radius, cannot be matched

    // create working copy of track param
    mTPCWork.emplace_back(static_cast<o2::track::TrackParCov&>(trcOrig), mCurrTPCTracksTreeEntry, it);
    auto& trc = mTPCWork.back();
    // propagate to matching Xref
    if (!propagateToRefX(trc)) {
      mTPCWork.pop_back(); // discard track whose propagation to mXRef failed
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

  // sort tracks in each sector according to their timeMax
  for (int sec = o2::constants::math::NSectors; sec--;) {
    auto& indexCache = mTPCSectIndexCache[sec];
    LOG(INFO) << "Sorting sector" << sec << " | " << indexCache.size() << " TPC tracks" << FairLogger::endl;
    if (!indexCache.size())
      continue;
    std::sort(indexCache.begin(), indexCache.end(), [this](int a, int b) {
      auto& trcA = mTPCWork[a];
      auto& trcB = mTPCWork[b];
      return (trcA.timeBins.tmax - trcB.timeBins.tmax) < 0.;
    });

    // build array of 1st entries with tmax corresponding to each ITS RO cycle
    float tmax = mTPCWork[indexCache.back()].timeBins.tmax;
    int nbins = 1 + tpcTimeBin2ITSROFrame(tmax);
    auto& tbinStart = mTPCTimeBinStart[sec];
    tbinStart.resize(nbins > 1 ? nbins : 1, -1);
    tbinStart[0] = 0;
    for (int itr = 0; itr < (int)indexCache.size(); itr++) {
      auto& trc = mTPCWork[indexCache[itr]];
      int bTrc = tpcTimeBin2ITSROFrame(trc.timeBins.tmax);
      if (bTrc < 0) {
        continue;
      }
      if (tbinStart[bTrc] == -1) {
        tbinStart[bTrc] = itr;
      }
    }
    for (int i = 1; i < nbins; i++) {
      if (tbinStart[i] == -1) { // fill gaps with preceding indices
        tbinStart[i] = tbinStart[i - 1];
      }
    }
  } // loop over tracks of single sector

  return true;
}

//_____________________________________________________
bool MatchTPCITS::prepareITSTracks()
{
  // load next chunk of ITS data and prepare for matching
  mMatchesITS.clear();
  mITSWork.clear();
  // number of records might be actually more than N tracks!
  mMatchRecordsITS.clear(); // RS TODO reserve(mMatchRecordsITS.size() + mMaxMatchCandidates*ntr);
  if (mMCTruthON) {
    mITSLblWork.clear();
  }
  for (int sec = o2::constants::math::NSectors; sec--;) {
    mITSSectIndexCache[sec].clear();
  }

  while (loadITSTracksNextChunk()) {
    int ntr = mITSTracksArrayInp->size();
    for (int it = 0; it < ntr; it++) {
      auto& trcOrig = (*mITSTracksArrayInp)[it];

      if (trcOrig.getParamOut().getX() < 1.) {
        continue; // backward refit failed
      }
      // working copy of outer track param
      mITSWork.emplace_back(static_cast<o2::track::TrackParCov&>(trcOrig.getParamOut()), mCurrITSTracksTreeEntry, it);
      auto& trc = mITSWork.back();

      // TODO: why I did this?
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

      float tmn = itsROFrame2TPCTimeBin(trcOrig.getROFrame());
      trc.timeBins.set(tmn, tmn + mITSROFrame2TPCBin);
      trc.roFrame = trcOrig.getROFrame();

      // cache work track index
      int sector = o2::utils::Angle2Sector(trc.getAlpha());
      mITSSectIndexCache[sector].push_back(mITSWork.size() - 1);

      // If the ITS track is very close to the sector edge, it may match also to a TPC track in the neighb. sector.
      // For a track with Yr and Phir at Xr the distance^2 between the poisition of this track in the neighb. sector
      // when propagated to Xr (in this neighbouring sector) and the edge will be (neglecting the curvature)
      // [(Xr*tg(10)-Yr)/(tgPhir+tg70)]^2  / cos(70)^2  // for the next sector
      // [(Xr*tg(10)+Yr)/(tgPhir-tg70)]^2  / cos(70)^2  // for the prev sector
      // Distances to the sector edges in neighbourings sectors (at Xref in theit proper frames)
      float tgp = trc.getSnp();
      tgp /= std::sqrt((1.f - tgp) * (1.f + tgp)); // tan of track direction XY

      // sector up
      float dy2Up = (mYMaxAtXRef - trc.getY()) / (tgp + Tan70);
      if ((dy2Up * dy2Up * Cos70I2) < mSectEdgeMargin2) { // need to check this track for matching in sector up
        addTrackCloneForNeighbourSector(trc, sector < (o2::constants::math::NSectors - 1) ? sector + 1 : 0);
      }
      // sector down
      float dy2Dn = (mYMaxAtXRef + trc.getY()) / (tgp - Tan70);
      if ((dy2Dn * dy2Dn * Cos70I2) < mSectEdgeMargin2) { // need to check this track for matching in sector down
        addTrackCloneForNeighbourSector(trc, sector > 1 ? sector - 1 : o2::constants::math::NSectors - 1);
      }
    }
  }
  // sort tracks in each sector according to their time, then tgl
  for (int sec = o2::constants::math::NSectors; sec--;) {
    auto& indexCache = mITSSectIndexCache[sec];
    LOG(INFO) << "Sorting sector" << sec << " | " << indexCache.size() << " ITS tracks" << FairLogger::endl;
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

    // build array of 1st entries with of each ITS RO cycle
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
      if (tbinStart[i] == -1) { // fill gaps with preceding indices
        tbinStart[i] = tbinStart[i - 1];
      }
    }
  } // loop over tracks of single sector
  mMatchesITS.reserve(mITSWork.size());
  mMatchRecordsITS.reserve(mITSWork.size() * mMaxMatchCandidates);

  return true;
}

//_____________________________________________________
bool MatchTPCITS::loadITSTracksNextChunk()
{
  ///< load next chunk of ITS data
  mTimerIO.Start(false);

  while (++mCurrITSTracksTreeEntry < mTreeITSTracks->GetEntries()) {
    mTreeITSTracks->GetEntry(mCurrITSTracksTreeEntry);
    LOG(DEBUG) << "Loading ITS tracks entry " << mCurrITSTracksTreeEntry << " -> " << mITSTracksArrayInp->size()
               << " tracks" << FairLogger::endl;
    if (!mITSTracksArrayInp->size()) {
      continue;
    }
    mTimerIO.Stop();
    return true;
  }
  --mCurrITSTracksTreeEntry;
  mTimerIO.Stop();
  return false;
}

//_____________________________________________________
bool MatchTPCITS::loadTPCTracksNextChunk()
{
  ///< load next chunk of TPC data
  mTimerIO.Start(false);

  while (++mCurrTPCTracksTreeEntry < mTreeTPCTracks->GetEntries()) {
    mTreeTPCTracks->GetEntry(mCurrTPCTracksTreeEntry);
    LOG(DEBUG) << "Loading TPC tracks entry " << mCurrTPCTracksTreeEntry << " -> " << mTPCTracksArrayInp->size()
               << " tracks" << FairLogger::endl;
    if (mTPCTracksArrayInp->size() < 1) {
      continue;
    }
    mTimerIO.Stop();
    return true;
  }
  --mCurrTPCTracksTreeEntry;

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
              << sec << FairLogger::endl;
    return;
  }

  /// full drift time + safety margin
  float maxTDriftSafe = (mNTPCBinsFullDrift + mITSTPCTimeBinSafeMargin + mTPCTimeEdgeTSafeMargin);

  // get min ROFrame (in TPC time-bins) of ITS tracks currently in cache
  auto minROFITS = mITSWork[cacheITS.front()].roFrame;

  if (minROFITS >= int(tbinStartTPC.size())) {
    LOG(INFO) << "ITS min ROFrame " << minROFITS << " exceeds all cached TPC track ROF eqiuvalent "
              << cacheTPC.size() - 1 << FairLogger::endl;
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
      // compare if the ITS and TPC tracks may overlap in time
      if (trefTPC.timeBins.tmax < trefITS.timeBins.tmin) {
        // since TPC tracks are sorted in timeMax and ITS tracks are sorted in timeMin
        // all following ITS tracks also will not match
        break;
      }
      if (trefTPC.timeBins.tmin > trefITS.timeBins.tmax) { // its bracket is fully before TPC bracket
        continue;
      }
      nCheckITSControl++;
      float chi2 = -1;
      int rejFlag = compareITSTPCTracks(trefITS, trefTPC, chi2);

#ifdef _ALLOW_DEBUG_TREES_
      if (mDBGOut && ((rejFlag == Accept && isDebugFlag(MatchTreeAccOnly)) || isDebugFlag(MatchTreeAll))) {
        fillITSTPCmatchTree(cacheITS[iits], cacheTPC[itpc], rejFlag, chi2);
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
            << "), checks: " << nCheckITSControl << ", matches:" << nMatchesControl << FairLogger::endl;
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
        nextMatchRec.matchID = tITS.matchID; // reuse the record of suppressed ITS match to store better one
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
int MatchTPCITS::compareITSTPCTracks(const TrackLocITS& tITS, const TrackLocTPC& tTPC, float& chi2) const
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
  //	       << tr1.getAlpha() << " : " << tr2.getAlpha() << FairLogger::endl;
  //    return 2. * o2::track::HugeF;
  //  }
  //  if (std::abs(tr1.getX() - tr2.getX()) > FLT_EPSILON) {
  //    LOG(ERROR) << "The reference X of the tracks differ: "
  //	       << tr1.getX() << " : " << tr2.getX() << FairLogger::endl;
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
    LOG(ERROR) << "Cov.matrix inversion failed: " << covMat << FairLogger::endl;
    return 2. * o2::track::HugeF;
  }
  double chi2diag = 0., chi2ndiag = 0.,
         diff[o2::track::kNParams - 1] = { tr1.getParam(o2::track::kY) - tr2.getParam(o2::track::kY),
                                           tr1.getParam(o2::track::kSnp) - tr2.getParam(o2::track::kSnp),
                                           tr1.getParam(o2::track::kTgl) - tr2.getParam(o2::track::kTgl),
                                           tr1.getParam(o2::track::kQ2Pt) - tr2.getParam(o2::track::kQ2Pt) };
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
      o2::Base::Propagator::Instance()->PropagateToXBxByBz(trc, mXRef, o2::constants::physics::MassPionCharged, MaxSnp,
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
  refReached = mXRef < 10.; // RS: tmp, to cover mXRef~0
  while (o2::Base::Propagator::Instance()->PropagateToXBxByBz(trc, mXRef, o2::constants::physics::MassPionCharged,
                                                              MaxSnp, 2., 1)) {
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
  printf("Matching reference X: %.3f\n", mXRef);
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
  printf("Max.Number of matched tracks per output entry: %d\n", mMaxOutputTracksPerEntry);

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

  LOG(INFO) << "Refitting winner matches" << FairLogger::endl;
  mWinnerChi2Refit.resize(mITSWork.size(), -1.f);
  mCurrITSTracksTreeEntry = -1;
  mCurrITSClustersTreeEntry = -1;
  for (int iITS = 0; iITS < mITSWork.size(); iITS++) {
    if (!refitTrackITSTPC(iITS)) {
      continue;
    }
    mWinnerChi2Refit[iITS] = mMatchedTracks.back().getChi2Refit();
    if (mMatchedTracks.size() == mMaxOutputTracksPerEntry) {
      if (mOutputTree) {
        mTimerRefit.Stop();
        mOutputTree->Fill();
        mTimerRefit.Start(false);
      }
      mMatchedTracks.clear();
      if (mMCTruthON) {
        mOutITSLabels.clear();
        mOutTPCLabels.clear();
      }
    }
  }
  // flush last tracks
  if (mMatchedTracks.size() && mOutputTree) {
    mOutputTree->Fill();
  }
  mMatchedTracks.clear();
  if (mMCTruthON) {
    mOutITSLabels.clear();
    mOutTPCLabels.clear();
  }
  mTimerRefit.Stop();
}

//______________________________________________
bool MatchTPCITS::refitTrackITSTPC(int iITS)
{
  ///< refit in inward direction the pair of TPC and ITS tracks
  const auto& tITS = mITSWork[iITS];
  if (tITS.matchID < 0 || isDisabledITS(mMatchesITS[tITS.matchID])) {
    return false; // no match
  }
  const auto& itsMatch = mMatchesITS[tITS.matchID];
  const auto& itsMatchRec = mMatchRecordsITS[itsMatch.first];
  int iTPC = mTPCMatch2Track[itsMatchRec.matchID];
  const auto& tTPC = mTPCWork[iTPC];

  loadITSClustersChunk(tITS.source.getEvent());
  loadITSTracksChunk(tITS.source.getEvent());
  loadTPCTracksChunk(tTPC.source.getEvent());

  mMatchedTracks.emplace_back(tTPC); // create a copy of TPC track at xRef
  auto& trfit = mMatchedTracks.back();
  // in continuos mode the Z of TPC track is meaningless, unless it is CE crossing
  // track (currently absent, TODO)
  if (!mCompareTracksDZ) {
    trfit.setZ(tITS.getZ()); // fix the seed Z
  }
  float deltaT = (trfit.getZ() - tTPC.getZ()) * mZ2TPCBin; // time correction in time-bins

  auto itsTrOrig = (*mITSTracksArrayInp)[tITS.source.getIndex()]; // currently we store clusterIDs in the track
  int nclRefit = 0, ncl = itsTrOrig.getNumberOfClusters();
  float chi2 = 0.f;
  auto geom = o2::ITS::GeometryTGeo::Instance();
  auto propagator = o2::Base::Propagator::Instance();
  for (int icl = 0; icl < ncl; icl++) {
    const auto& clus = (*mITSClustersArrayInp)[itsTrOrig.getClusterIndex(icl)];
    float alpha = geom->getSensorRefAlpha(clus.getSensorID()), x = clus.getX();
    if (!trfit.rotate(alpha) ||
        !propagator->PropagateToXBxByBz(trfit, x, o2::constants::physics::MassPionCharged, MaxSnp, 2., 1)) {
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

  /// precise time estimate
  auto tpcTrOrig = (*mTPCTracksArrayInp)[tTPC.source.getIndex()];
  float time = tpcTrOrig.getTime0() - mNTPCBinsFullDrift;
  if (tpcTrOrig.hasASideClustersOnly()) {
    time += deltaT;
  } else if (tpcTrOrig.hasCSideClustersOnly()) {
    time -= deltaT;
  } else {
    // TODO : special treatment of tracks crossing the CE
  }
  // convert time to microseconds
  time *= mTPCTBinMUS;
  // estimate the error on time
  float timeErr = std::sqrt(tITS.getSigmaZ2() + tTPC.getSigmaZ2()) * mTPCVDrift0Inv;
  trfit.setChi2Match(itsMatchRec.chi2);
  trfit.setChi2Refit(chi2);
  trfit.setTimeMUS(time, timeErr);
  trfit.setRefTPC(tTPC.source);
  trfit.setRefITS(tITS.source);

  if (mMCTruthON) { // store MC info
    mOutITSLabels.emplace_back(mITSLblWork[iITS]);
    mOutTPCLabels.emplace_back(mTPCLblWork[iTPC]);
  }

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
void MatchTPCITS::loadITSTracksChunk(int chunk)
{
  // load single entry from ITS tracks tree
  if (mCurrITSTracksTreeEntry != chunk) {
    mTimerIO.Start(false);
    mTreeITSTracks->GetEntry(mCurrITSTracksTreeEntry = chunk);
    mTimerIO.Stop();
  }
}

//________________________________________________________
void MatchTPCITS::loadTPCTracksChunk(int chunk)
{
  // load single entry from TPC tracks tree
  if (mCurrTPCTracksTreeEntry != chunk) {
    mTimerIO.Start(false);
    mTreeITSTracks->GetEntry(mCurrTPCTracksTreeEntry = chunk);
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
void MatchTPCITS::fillITSTPCmatchTree(int itsID, int tpcID, int rejFlag, float chi2)
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

  LOG(INFO) << "Dumping debug tree for winner matches" << FairLogger::endl;
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
