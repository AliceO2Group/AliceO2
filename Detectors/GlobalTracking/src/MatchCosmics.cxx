// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "GlobalTracking/MatchCosmics.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "GPUO2InterfaceRefit.h"
#include "ReconstructionDataFormats/GlobalTrackAccessor.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTOF/Cluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsFT0/RecPoints.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/TrackTPCTOF.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "ITStracking/IOUtils.h"
#include "ITSBase/GeometryTGeo.h"
#include "TPCBase/ParameterElectronics.h"
#include "DetectorsBase/Propagator.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "GlobalTracking/MatchTPCITS.h"
#include "CommonConstants/GeomConstants.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include <algorithm>
#include <numeric>

using namespace o2::globaltracking;

using GTrackID = o2d::GlobalTrackID;
using MatCorrType = o2::base::Propagator::MatCorrType;

//________________________________________________________
void MatchCosmics::process(const o2::globaltracking::RecoContainer& data)
{
  updateTimeDependentParams();
  mRecords.clear();
  mWinners.clear();
  mCosmicTracks.clear();
  mCosmicTracksLbl.clear();

  createSeeds(data);
  int ntr = mSeeds.size();

  // propagate to DCA to origin
  const o2::math_utils::Point3D<float> v{0., 0., 0};
  for (int i = 0; i < ntr; i++) {
    auto& trc = mSeeds[i];
    if (!o2::base::Propagator::Instance()->propagateToDCABxByBz(v, trc, mMatchParams->maxStep, mMatchParams->matCorr)) {
      trc.matchID = Reject; // reject track
    }
  }

  // sort in time bracket lower edge
  std::vector<int> sortID(ntr);
  std::iota(sortID.begin(), sortID.end(), 0);
  std::sort(sortID.begin(), sortID.end(), [this](int a, int b) { return mSeeds[a].tBracket.getMin() < mSeeds[b].tBracket.getMin(); });

  for (int i = 0; i < ntr; i++) {
    for (int j = i + 1; j < ntr; j++) {
      if (checkPair(sortID[i], sortID[j]) == RejTime) {
        break;
      }
    }
  }

  selectWinners();
  refitWinners(data);

  mTFCount++;
}

//________________________________________________________
void MatchCosmics::refitWinners(const o2::globaltracking::RecoContainer& data)
{
  LOG(INFO) << "Refitting " << mWinners.size() << " winner matches";
  int count = 0;
  auto tpcTBinMUSInv = 1. / mTPCTBinMUS;
  if (!mTPCTransform) { // eventually, should be updated at every TF?
    mTPCTransform = o2::tpc::TPCFastTransformHelperO2::instance()->create(0);
  }
  const auto& tpcClusRefs = data.getTPCTracksClusterRefs();
  const auto& tpcClusShMap = data.clusterShMapTPC;
  std::unique_ptr<o2::gpu::GPUO2InterfaceRefit> tpcRefitter;
  if (data.inputsTPCclusters) {
    tpcRefitter = std::make_unique<o2::gpu::GPUO2InterfaceRefit>(&data.inputsTPCclusters->clusterIndex,
                                                                 mTPCTransform.get(), mBz,
                                                                 tpcClusRefs.data(), tpcClusShMap.data(),
                                                                 nullptr, o2::base::Propagator::Instance());
  }

  const auto& itsClusters = prepareITSClusters(data);
  // RS FIXME: this is probably a temporary solution, since ITS tracking over boundaries will likely change the TrackITS format
  std::vector<int> itsTracksROF;

  const auto& itsTracksROFRec = data.getITSTracksROFRecords();
  itsTracksROF.resize(data.getITSTracks().size());
  for (unsigned irf = 0, cnt = 0; irf < itsTracksROFRec.size(); irf++) {
    int ntr = itsTracksROFRec[irf].getNEntries();
    for (int itr = 0; itr < ntr; itr++) {
      itsTracksROF[cnt++] = irf;
    }
  }

  auto refitITSTrack = [this, &data, &itsTracksROF, &itsClusters](o2::track::TrackParCov& trFit, GTrackID gidx, float& chi2, bool inward = false) {
    const auto& itsTrOrig = data.getITSTrack(gidx);
    int nclRefit = 0, ncl = itsTrOrig.getNumberOfClusters(), rof = itsTracksROF[gidx.getIndex()];
    const auto& itsClustersROFRec = data.getITSClustersROFRecords();
    const auto& itsTrackClusRefs = data.getITSTracksClusterRefs();
    int clusIndOffs = itsClustersROFRec[rof].getFirstEntry(), clEntry = itsTrOrig.getFirstClusterEntry();
    const auto propagator = o2::base::Propagator::Instance();
    const auto geomITS = o2::its::GeometryTGeo::Instance();
    int from = ncl - 1, to = -1, step = -1;
    if (inward) {
      from = 0;
      to = ncl;
      step = 1;
    }
    for (int icl = from; icl != to; icl += step) { // ITS clusters are referred in layer decreasing order
      const auto& clus = itsClusters[clusIndOffs + itsTrackClusRefs[clEntry + icl]];
      float alpha = geomITS->getSensorRefAlpha(clus.getSensorID()), x = clus.getX();
      if (!trFit.rotate(alpha) || !propagator->propagateToX(trFit, x, propagator->getNominalBz(), this->mMatchParams->maxSnp, this->mMatchParams->maxStep, this->mMatchParams->matCorr)) {
        break;
      }
      chi2 += trFit.getPredictedChi2(clus);
      if (!trFit.update(clus)) {
        break;
      }
      nclRefit++;
    }
    return nclRefit == ncl ? ncl : -1;
  };

  for (auto winRID : mWinners) {
    const auto& rec = mRecords[winRID];
    int poolEntryID[2] = {rec.id0, rec.id1};
    const o2::track::TrackParCov outerLegs[2] = {data.getTrackParamOut(mSeeds[rec.id0].origID), data.getTrackParamOut(mSeeds[rec.id1].origID)};
    auto tOverlap = mSeeds[rec.id0].tBracket.getOverlap(mSeeds[rec.id1].tBracket);
    float t0 = tOverlap.mean(), dt = tOverlap.delta() * 0.5;
    auto pnt0 = outerLegs[0].getXYZGlo(), pnt1 = outerLegs[1].getXYZGlo();
    int btm = 0, top = 1;
    // we fit topward from bottom
    if (pnt0.Y() > pnt1.Y()) {
      btm = 1;
      top = 0;
    }
    LOG(DEBUG) << "Winner " << count++ << " Record " << winRID << " Partners:"
               << " B: " << mSeeds[poolEntryID[btm]].origID << "/" << mSeeds[poolEntryID[btm]].origID.getSourceName()
               << " U: " << mSeeds[poolEntryID[top]].origID << "/" << mSeeds[poolEntryID[top]].origID.getSourceName()
               << " | T:" << tOverlap.asString();

    float chi2 = 0;
    int nclTot = 0;

    // Start from bottom leg inward refit
    o2::track::TrackParCov trCosm(mSeeds[poolEntryID[btm]]); // copy of the btm track
    // The bottom leg needs refit only if it is an unconstrained TPC track, otherwise it is already refitted as inner param
    if (mSeeds[poolEntryID[btm]].origID.getSource() == GTrackID::TPC) {
      const auto& tpcTrOrig = data.getTPCTrack(mSeeds[poolEntryID[btm]].origID);
      trCosm = outerLegs[btm];
      int retVal = tpcRefitter->RefitTrackAsTrackParCov(trCosm, tpcTrOrig.getClusterRef(), t0 * tpcTBinMUSInv, &chi2, false, true); // inward refit, reset
      if (retVal < 0) {                                                                                                             // refit failed
        LOG(DEBUG) << "Inward refit of btm TPC track failed.";
        continue;
      }
      nclTot += retVal;
      LOG(DEBUG) << "chi2 after btm TPC refit with " << retVal << " clusters : " << chi2 << " orig.chi2 was " << tpcTrOrig.getChi2();
    } else { // just collect NClusters and chi2
      auto gidxListBtm = data.getSingleDetectorRefs(mSeeds[poolEntryID[btm]].origID);
      if (gidxListBtm[GTrackID::TPC].isIndexSet()) {
        const auto& tpcTrOrig = data.getTPCTrack(gidxListBtm[GTrackID::TPC]);
        nclTot += tpcTrOrig.getNClusters();
        chi2 += tpcTrOrig.getChi2();
      }
      if (gidxListBtm[GTrackID::ITS].isIndexSet()) {
        const auto& itsTrOrig = data.getITSTrack(gidxListBtm[GTrackID::ITS]);
        nclTot += itsTrOrig.getNClusters();
        chi2 += itsTrOrig.getChi2();
      }
    }
    trCosm.invert();
    if (!trCosm.rotate(mSeeds[poolEntryID[top]].getAlpha()) ||
        !o2::base::Propagator::Instance()->PropagateToXBxByBz(trCosm, mSeeds[poolEntryID[top]].getX(), mMatchParams->maxSnp, mMatchParams->maxStep, mMatchParams->matCorr)) {
      LOG(DEBUG) << "Rotation/propagation of btm-track to top-track frame failed.";
      continue;
    }
    // save bottom parameter at merging point
    auto trCosmBtm = trCosm;
    int nclBtm = nclTot;

    // Continue with top leg outward refit
    auto gidxListTop = data.getSingleDetectorRefs(mSeeds[poolEntryID[top]].origID);

    // is there ITS sub-track?
    if (gidxListTop[GTrackID::ITS].isIndexSet()) {
      auto nclfit = refitITSTrack(trCosm, gidxListTop[GTrackID::ITS], chi2, false);
      if (nclfit < 0) {
        continue;
      }
      LOG(DEBUG) << "chi2 after top ITS refit with " << nclfit << " clusters : " << chi2 << " orig.chi2 was " << data.getITSTrack(gidxListTop[GTrackID::ITS]).getChi2();
      nclTot += nclfit;
    } // ITS refit
    //
    if (gidxListTop[GTrackID::TPC].isIndexSet()) { // outward refit in TPC
      // go to TPC boundary, if needed
      if (trCosm.getX() * trCosm.getX() + trCosm.getY() * trCosm.getY() <= o2::constants::geom::XTPCInnerRef * o2::constants::geom::XTPCInnerRef) {
        float xtogo = 0;
        if (!trCosm.getXatLabR(o2::constants::geom::XTPCInnerRef, xtogo, mBz, o2::track::DirOutward) ||
            !o2::base::Propagator::Instance()->PropagateToXBxByBz(trCosm, xtogo, mMatchParams->maxSnp, mMatchParams->maxStep, mMatchParams->matCorr)) {
          LOG(DEBUG) << "Propagation to inner TPC boundary X=" << xtogo << " failed";
          continue;
        }
      }
      const auto& tpcTrOrig = data.getTPCTrack(gidxListTop[GTrackID::TPC]);
      int retVal = tpcRefitter->RefitTrackAsTrackParCov(trCosm, tpcTrOrig.getClusterRef(), t0 * tpcTBinMUSInv, &chi2, true, false); // outward refit, no reset
      if (retVal < 0) {                                                                                                             // refit failed
        LOG(DEBUG) << "Outward refit of top TPC track failed.";
        continue;
      } // outward refit in TPC
      LOG(DEBUG) << "chi2 after top TPC refit with " << retVal << " clusters : " << chi2 << " orig.chi2 was " << tpcTrOrig.getChi2();
      nclTot += retVal;
    }

    // inward refit of top leg for evaluation in DCA
    float chi2Dummy = 0;
    auto trCosmTop = outerLegs[top];
    if (gidxListTop[GTrackID::TPC].isIndexSet()) { // inward refit in TPC
      const auto& tpcTrOrig = data.getTPCTrack(gidxListTop[GTrackID::TPC]);
      int retVal = tpcRefitter->RefitTrackAsTrackParCov(trCosmTop, tpcTrOrig.getClusterRef(), t0 * tpcTBinMUSInv, &chi2Dummy, false, true); // inward refit, reset
      if (retVal < 0) {                                                                                                                     // refit failed
        LOG(DEBUG) << "Outward refit of top TPC track failed.";
        continue;
      } // inward refit in TPC
    }
    // is there ITS sub-track ?
    if (gidxListTop[GTrackID::ITS].isIndexSet()) {
      auto nclfit = refitITSTrack(trCosmTop, gidxListTop[GTrackID::ITS], chi2Dummy, true);
      if (nclfit < 0) {
        continue;
      }
      nclTot += nclfit;
    } // ITS refit
    // propagate to bottom param
    if (!trCosmTop.rotate(trCosmBtm.getAlpha()) ||
        !o2::base::Propagator::Instance()->PropagateToXBxByBz(trCosmTop, trCosmBtm.getX(), mMatchParams->maxSnp, mMatchParams->maxStep, mMatchParams->matCorr)) {
      LOG(DEBUG) << "Rotation/propagation of top-track to bottom-track frame failed.";
      continue;
    }
    // calculate weighted average of 2 legs and chi2
    o2::track::TrackParCov::MatrixDSym5 cov5;
    float chi2Match = trCosmBtm.getPredictedChi2(trCosmTop, cov5);
    if (!trCosmBtm.update(trCosmTop, cov5)) {
      LOG(DEBUG) << "Top/Bottom update failed";
      continue;
    }
    // create final track
    mCosmicTracks.emplace_back(mSeeds[poolEntryID[btm]].origID, mSeeds[poolEntryID[top]].origID, trCosmBtm, trCosmTop, chi2, chi2Match, nclTot, t0, dt);
    if (mUseMC) {
      o2::MCCompLabel lbl[2] = {data.getTrackMCLabel(mSeeds[poolEntryID[btm]].origID), data.getTrackMCLabel(mSeeds[poolEntryID[top]].origID)};
      auto& tlb = mCosmicTracksLbl.emplace_back((nclBtm > nclTot - nclBtm ? lbl[0] : lbl[1]));
      tlb.setFakeFlag(lbl[0] != lbl[1]);
    }
  }
  LOG(INFO) << "Validated " << mCosmicTracks.size() << " top-bottom tracks in TF# " << mTFCount;
}

//________________________________________________________
void MatchCosmics::selectWinners()
{
  // select mutually best matches
  int ntr = mSeeds.size(), iter = 0, nValidated = 0;
  mWinners.reserve(mRecords.size() / 2); // there are 2 records per match candidate
  do {
    nValidated = 0;
    int nRemaining = 0;
    for (int i = 0; i < ntr; i++) {
      if (mSeeds[i].matchID < 0 || mRecords[mSeeds[i].matchID].next == Validated) { // either have no match or already validated
        continue;
      }
      nRemaining++;
      if (validateMatch(i)) {
        mWinners.push_back(mSeeds[i].matchID);
        nValidated++;
        continue;
      }
    }
    LOGF(INFO, "iter %d Validated %d of %d remaining matches", iter, nValidated, nRemaining);
    iter++;
  } while (nValidated);
}

//________________________________________________________
bool MatchCosmics::validateMatch(int partner0)
{
  // make sure that the best partner of seed_i has also seed_i as a best partner
  auto& matchRec = mRecords[mSeeds[partner0].matchID];
  auto partner1 = matchRec.id1;
  auto& patnerRec = mRecords[mSeeds[partner1].matchID];
  if (patnerRec.next == Validated) { // partner1 was already validated with other partner0
    return false;
  }
  if (patnerRec.id1 == partner0) { // mutually best
    // unlink winner partner0 from all other mathes
    auto next0 = matchRec.next;
    while (next0 > MinusOne) {
      auto& nextRec = mRecords[next0];
      suppressMatch(partner0, nextRec.id1);
      next0 = nextRec.next;
    }
    matchRec.next = Validated;

    // unlink winner partner1 from all other matches
    auto next1 = patnerRec.next;
    while (next1 > MinusOne) {
      auto& nextRec = mRecords[next1];
      suppressMatch(partner1, nextRec.id1);
      next1 = nextRec.next;
    }
    patnerRec.next = Validated;
    return true;
  }
  return false;
}

//________________________________________________________
void MatchCosmics::suppressMatch(int partner0, int partner1)
{
  // suppress reference to partner0 from partner1 match record
  if (mSeeds[partner1].matchID < 0 || mRecords[mSeeds[partner1].matchID].next == Validated) {
    LOG(WARNING) << "Attempt to remove null or validated partner match " << mSeeds[partner1].matchID;
    return;
  }
  int topID = MinusOne, next = mSeeds[partner1].matchID;
  while (next > MinusOne) {
    auto& matchRec = mRecords[next];
    if (matchRec.id1 == partner0) {
      if (topID < 0) {                            // best match
        mSeeds[partner1].matchID = matchRec.next; // exclude best match link
      } else {                                    // not the 1st link in the chain
        mRecords[topID].next = matchRec.next;
      }
      return;
    }
    topID = next;
    next = matchRec.next;
  }
}

//________________________________________________________
MatchCosmics::RejFlag MatchCosmics::checkPair(int i, int j)
{
  // if validated with given chi2, register match
  RejFlag rej = RejOther;
  auto& seed0 = mSeeds[i];
  auto& seed1 = mSeeds[j];
  if (seed0.matchID == Reject) {
    return rej;
  }
  if (seed1.matchID == Reject) {
    return rej;
  }

  LOG(DEBUG) << "Seed " << i << " [" << seed0.tBracket.getMin() << " : " << seed0.tBracket.getMax() << "] | "
             << "Seed " << j << " [" << seed1.tBracket.getMin() << " : " << seed1.tBracket.getMax() << "] | ";
  LOG(DEBUG) << seed0.origID << " | " << seed0.o2::track::TrackPar::asString();
  LOG(DEBUG) << seed1.origID << " | " << seed1.o2::track::TrackPar::asString();

  if (seed1.tBracket > seed0.tBracket) {
    return (rej = RejTime); // since the brackets are sorted in tmin, all following tbj will also exceed tbi
  }
  float chi2 = 1.e9f;

  // check
  // 1) crude check on tgl and q/pt (if B!=0). Note: back-to-back tracks will have mutually params (see TrackPar::invertParam)
  while (1) {
    auto dTgl = seed0.getTgl() + seed1.getTgl();
    if (dTgl * dTgl > (mMatchParams->systSigma2[o2::track::kTgl] + seed0.getSigmaTgl2() + seed1.getSigmaTgl2()) * mMatchParams->crudeNSigma2Cut[o2::track::kTgl]) {
      rej = RejTgl;
      break;
    }
    if (mFieldON) {
      auto dQ2Pt = seed0.getQ2Pt() + seed1.getQ2Pt();
      if (dQ2Pt * dQ2Pt > (mMatchParams->systSigma2[o2::track::kQ2Pt] + seed0.getSigma1Pt2() + seed1.getSigma1Pt2()) * mMatchParams->crudeNSigma2Cut[o2::track::kQ2Pt]) {
        rej = RejQ2Pt;
        break;
      }
    }
    o2::track::TrackParCov seed1Inv = seed1;
    seed1Inv.invert();
    for (int i = 0; i < o2::track::kNParams; i++) { // add systematic error
      seed1Inv.updateCov(mMatchParams->systSigma2[i], o2::track::DiagMap[i]);
    }

    if (!seed1Inv.rotate(seed0.getAlpha()) ||
        !o2::base::Propagator::Instance()->PropagateToXBxByBz(seed1Inv, seed0.getX(), mMatchParams->maxSnp, mMatchParams->maxStep, mMatchParams->matCorr)) {
      rej = RejProp;
      break;
    }
    auto dSnp = seed0.getSnp() - seed1Inv.getSnp();
    if (dSnp * dSnp > (seed0.getSigmaSnp2() + seed1Inv.getSigmaSnp2()) * mMatchParams->crudeNSigma2Cut[o2::track::kSnp]) {
      rej = RejSnp;
      break;
    }
    auto dY = seed0.getY() - seed1Inv.getY();
    if (dY * dY > (seed0.getSigmaY2() + seed1Inv.getSigmaY2()) * mMatchParams->crudeNSigma2Cut[o2::track::kY]) {
      rej = RejY;
      break;
    }
    bool ignoreZ = seed0.origID.getSource() == o2d::GlobalTrackID::TPC || seed1.origID.getSource() == o2d::GlobalTrackID::TPC;
    if (!ignoreZ) { // cut on Z makes no sense for TPC only tracks
      auto dZ = seed0.getZ() - seed1Inv.getZ();
      if (dZ * dZ > (seed0.getSigmaZ2() + seed1Inv.getSigmaZ2()) * mMatchParams->crudeNSigma2Cut[o2::track::kZ]) {
        rej = RejZ;
        break;
      }
    } else { // inflate Z error
      seed1Inv.setCov(250. * 250., o2::track::DiagMap[o2::track::kZ]);
      seed1Inv.setCov(0., o2::track::CovarMap[o2::track::kZ][o2::track::kY]); // set all correlation terms for Z error to 0
      seed1Inv.setCov(0., o2::track::CovarMap[o2::track::kZ][o2::track::kSnp]);
      seed1Inv.setCov(0., o2::track::CovarMap[o2::track::kZ][o2::track::kTgl]);
      seed1Inv.setCov(0., o2::track::CovarMap[o2::track::kZ][o2::track::kQ2Pt]);
    }
    // calculate chi2 (expensive)
    chi2 = seed0.getPredictedChi2(seed1Inv);
    if (chi2 > mMatchParams->crudeChi2Cut) {
      rej = RejChi2;
      break;
    }
    rej = Accept;
    registerMatch(i, j, chi2);
    registerMatch(j, i, chi2); // the reverse reference can be also done in a separate loop
    LOG(DEBUG) << "Chi2 = " << chi2 << " NMatches " << mRecords.size();
    break;
  }

#ifdef _ALLOW_DEBUG_TREES_
  if (mDBGOut && ((rej == Accept && isDebugFlag(MatchTreeAccOnly)) || isDebugFlag(MatchTreeAll))) {
    auto seed1I = seed1;
    seed1I.invert();
    if (seed1I.rotate(seed0.getAlpha()) && o2::base::Propagator::Instance()->PropagateToXBxByBz(seed1I, seed0.getX(), mMatchParams->maxSnp, mMatchParams->maxStep, mMatchParams->matCorr)) {
      int rejI = int(rej);
      (*mDBGOut) << "match"
                 << "tf=" << mTFCount << "seed0=" << seed0 << "seed1=" << seed1I << "chi2Match=" << chi2 << "rej=" << rejI << "\n";
    }
  }
#endif

  return rej;
}

//________________________________________________________
void MatchCosmics::registerMatch(int i, int j, float chi2)
{
  /// register track index j as a match for track index i
  int newRef = mRecords.size();
  auto& matchRec = mRecords.emplace_back(MatchRecord{i, j, chi2, MinusOne});
  auto* best = &mSeeds[i].matchID;
  while (*best > MinusOne) {
    auto& oldMatchRec = mRecords[*best];
    if (oldMatchRec.chi2 > chi2) { // insert new match in front of the old one
      matchRec.next = *best;       // new record will refer to the one it is superseding
      *best = newRef;              // the reference on the superseded record should now refer to new one
      break;
    }
    best = &oldMatchRec.next;
  }
  if (matchRec.next == MinusOne) { // did not supersed any other record
    *best = newRef;
  }
}

//________________________________________________________
void MatchCosmics::createSeeds(const o2::globaltracking::RecoContainer& data)
{
  // Scan all inputs and create seeding tracks

  mSeeds.clear();

  auto creator = [this](auto& _tr, GTrackID _origID, float t0, float terr) {
    if (std::abs(_tr.getQ2Pt()) > this->mQ2PtCutoff) {
      return true;
    }
    if constexpr (isTPCTrack<decltype(_tr)>()) {
      // unconstrained TPC track, with t0 = TrackTPC.getTime0+0.5*(DeltaFwd-DeltaBwd) and terr = 0.5*(DeltaFwd+DeltaBwd) in TimeBins
      t0 *= this->mTPCTBinMUS;
      terr *= this->mTPCTBinMUS;
    } else if (isITSTrack<decltype(_tr)>()) {
      t0 += 0.5 * this->mITSROFrameLengthMUS; // time 0 is supplied as beginning of ROF in \mus
      terr *= this->mITSROFrameLengthMUS;     // error is supplied a half-ROF duration, convert to \mus
    } else {                                  // all other tracks are provided with time and its gaussian error in \mus
      terr *= this->mMatchParams->nSigmaTError;
    }
    terr += this->mMatchParams->timeToleranceMUS;
    mSeeds.emplace_back(TrackSeed{_tr, {t0 - terr, t0 + terr}, _origID, MinusOne});
    return true;
  };

  data.createTracksVariadic(creator);

  LOG(INFO) << "collected " << mSeeds.size() << " seeds";
}

//________________________________________________________
void MatchCosmics::updateTimeDependentParams()
{
  ///< update parameters depending on time (once per TF)
  auto& elParam = o2::tpc::ParameterElectronics::Instance();
  mTPCTBinMUS = elParam.ZbinWidth; // TPC bin in microseconds
  mBz = o2::base::Propagator::Instance()->getNominalBz();
  mFieldON = std::abs(mBz) > 0.01;
  mQ2PtCutoff = 1.f / std::max(0.05f, mMatchParams->minSeedPt);
  if (mFieldON) {
    mQ2PtCutoff *= 5.00668 / std::abs(mBz);
  } else {
    mQ2PtCutoff = 1e9;
  }
}

//________________________________________________________
void MatchCosmics::init()
{
  mMatchParams = &o2::globaltracking::MatchCosmicsParams::Instance();

#ifdef _ALLOW_DEBUG_TREES_COSM
  // debug streamer
  if (mDBGFlags) {
    mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(mDebugTreeFileName.data(), "recreate");
  }
#endif
}

//________________________________________________________
std::vector<o2::BaseCluster<float>> MatchCosmics::prepareITSClusters(const o2::globaltracking::RecoContainer& data) const
{
  std::vector<o2::BaseCluster<float>> itscl;
  const auto& clusITS = data.getITSClusters();
  if (clusITS.size()) {
    const auto& patterns = data.getITSClustersPatterns();
    itscl.reserve(clusITS.size());
    auto pattIt = patterns.begin();
    o2::its::ioutils::convertCompactClusters(clusITS, pattIt, itscl, *mITSDict);
  }
  return std::move(itscl);
}

//______________________________________________
void MatchCosmics::end()
{
#ifdef _ALLOW_DEBUG_TREES_COSM
  mDBGOut.reset();
#endif
}

#ifdef _ALLOW_DEBUG_TREES_
//______________________________________________
void MatchCosmics::setDebugFlag(UInt_t flag, bool on)
{
  ///< set debug stream flag
  if (on) {
    mDBGFlags |= flag;
  } else {
    mDBGFlags &= ~flag;
  }
}

#endif
