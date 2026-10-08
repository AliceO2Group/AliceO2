// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "ReconstructionDataFormats/DCA.h"
#include "ITStracking/IOUtils.h"
#include "ITSBase/GeometryTGeo.h"
#include "TPCBase/ParameterElectronics.h"
#include "DetectorsBase/Propagator.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "GlobalTracking/MatchTPCITS.h"
#include "CommonConstants/GeomConstants.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsTPC/VDriftCorrFact.h"
#include "TPCFastTransformPOD.h"
#include <algorithm>
#include <numeric>
#include <unordered_map>

using namespace o2::globaltracking;

using GTrackID = o2d::GlobalTrackID;
using MatCorrType = o2::base::Propagator::MatCorrType;

namespace
{
// energy-loss sign of the propagations: the propagator applies the loss along the direction of the track parameters, and the refit runs
// from the bottom leg up through the top leg (parameters of an upward-moving particle); the cosmic muon flies from the top leg to the
// bottom leg, so the sign is imposed along its flight
constexpr int ELossGain = 1;
constexpr int ELossLoss = -1;
} // namespace

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
  const auto prop = o2::base::Propagator::Instance();
  // propagate to DCA to origin. A VertexBase (origin, zero covariance) selects the TrackParCov overload of propagateToDCABxByBz:
  // with a Point3D the TrackPar_t overload is used and only the parameters are propagated, the covariance stays the one at the
  // track's reference X (TPC inner radius), which makes the y/snp cuts and the chi2 of checkPair far too tight.
  const o2::dataformats::VertexBase v;
  for (int i = 0; i < ntr; i++) {
    auto& trc = mSeeds[i];
    if (trc.matchID != Reject) {
      // a cosmic muon flies downward: along a leg whose outward direction points up (top leg) the inward propagation follows the flight,
      // so the energy is lost; along the bottom leg it goes back in the flight, so the energy is gained
      std::array<float, 3> momentum{};
      trc.getPxPyPzGlo(momentum);
      const int eLossSign = momentum[1] > 0.f ? ELossLoss : ELossGain;
      if (!prop->propagateToDCABxByBz(v, trc, mMatchParams->maxStep, mMatchParams->matCorr, nullptr, nullptr, eLossSign)) {
        trc.matchID = Reject; // reject track
        continue;
      }
      if (std::abs(trc.getY()) < mMatchParams->minSeedDCAxy || trc.getY() * trc.getY() < mMatchParams->minSeedDCAxyNSigma * mMatchParams->minSeedDCAxyNSigma * trc.getSigmaY2()) {
        // passes close to the beam line, absolutely or within its errors: indistinguishable from collision tracks
        trc.matchID = Reject;
        continue;
      }
      if (mMatchParams->dcaCutChi2[trc.origID.getSource()] > 0.f && mUsePVInfo && trc.vtIDMin >= 0 && (std::abs(trc.getY()) < mMatchParams->fiducialRIP && std::abs(trc.getZ()) < mMatchParams->fiducialZIP)) {
        // do the propagation only if we are in the fiducial IP range.
        for (int iv = trc.vtIDMin; iv <= trc.vtIDMax; iv++) { // vtIDMax is the last compatible vertex (inclusive); vtIDMin < 0: no compatible vertex
          const auto& pv = data.getPrimaryVertex(iv);
          o2::track::TrackParCov trcatPV(trc);
          o2::dataformats::DCA dca;
          if (!trcatPV.propagateToDCA(pv, mBz, &dca)) {
            trc.matchID = Reject;
            break;
          }
          if (trc.origID.getSource() == GTrackID::TPC) { // correct the track Z position for the vertex time
            const auto& trcTPC = data.getTPCTrack(trc.origID);
            float deltaZ = trcTPC.hasBothSidesClusters() ? 0.f : (pv.getTimeStamp().getTimeStamp() - trcTPC.getTime0() * 8 * o2::constants::lhc::LHCBunchSpacingMUS) * mTPCVDrift;
            dca.setZ(dca.getZ() + (trcTPC.hasASideClustersOnly() ? deltaZ : -deltaZ));
          }
          dca.addCov({mMatchParams->systSigma2[0], 0.f, mMatchParams->systSigma2[1]});
          if (dca.calcChi2() < mMatchParams->dcaCutChi2[trc.origID.getSource()]) {
            trc.matchID = Reject;
            break;
          }
        }
      }
    }
  }
  // TPC refitter of this TF: same-side TPC-only legs compared at a common time (checkPair) and the refit of the winners
  std::unique_ptr<o2::gpu::GPUO2InterfaceRefit> tpcRefitter;
  if (data.inputsTPCclusters) {
    tpcRefitter = std::make_unique<o2::gpu::GPUO2InterfaceRefit>(&data.inputsTPCclusters->clusterIndex, mTPCCorrMaps, mBz, data.getTPCTracksClusterRefs().data(), 0,
                                                                 data.clusterShMapTPC.data(), data.occupancyMapTPC.data(), data.occupancyMapTPC.size(), nullptr,
                                                                 o2::base::Propagator::Instance());
  }
  mTPCRefitter = tpcRefitter.get();
  mRecoData = &data;
  mNRefitsCommonTime = 0;

  // sort in time bracket lower edge, putting rejected tracks in the end
  std::vector<int> sortID(ntr);
  std::iota(sortID.begin(), sortID.end(), 0);
  std::sort(sortID.begin(), sortID.end(), [this](int a, int b) { return mSeeds[a].matchID == Reject ? false : (mSeeds[b].matchID == Reject ? true : (mSeeds[a].tBracket.getMin() < mSeeds[b].tBracket.getMin())); });
  int lastValid = ntr - 1;
  for (; lastValid >= 0; lastValid--) {
    if (mSeeds[sortID[lastValid]].matchID != Reject) {
      break;
    }
  }
  ntr = lastValid >= 0 ? lastValid + 1 : 0;
  LOGP(info, "Collected {} seeds, validated: {}", mSeeds.size(), ntr);
  sortID.resize(ntr);
  for (int i = 0; i < ntr; i++) {
    for (int j = i + 1; j < ntr; j++) {
      if (checkPair(sortID[i], sortID[j]) == RejTime) {
        break;
      }
    }
  }

  selectWinners();
  refitWinners(data);
  if (mNRefitsCommonTime) {
    LOGP(info, "{} seeds refitted at the common time of same-side pairs", mNRefitsCommonTime);
  }
  mTPCRefitter = nullptr;
  mRecoData = nullptr;

  mTFCount++;
}

//________________________________________________________
void MatchCosmics::refitWinners(const o2::globaltracking::RecoContainer& data)
{
  LOG(info) << "Refitting " << mWinners.size() << " winner matches";
  int count = 0;
  auto tpcTBinMUSInv = 1. / mTPCTBinMUS;
  auto* tpcRefitter = mTPCRefitter; // created in process()

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

  auto refitITSTrack = [this, &data, &itsTracksROF, &itsClusters](o2::track::TrackParCov& trFit, GTrackID gidx, float& chi2, bool inward, int eLossSign) {
    const auto& itsTrOrig = data.getITSTrack(gidx);
    int nclRefit = 0, ncl = itsTrOrig.getNumberOfClusters(), rof = itsTracksROF[gidx.getIndex()];
    const auto& itsTrackClusRefs = data.getITSTracksClusterRefs();
    int clEntry = itsTrOrig.getFirstClusterEntry();
    const auto propagator = o2::base::Propagator::Instance();
    const auto geomITS = o2::its::GeometryTGeo::Instance();
    int from = ncl - 1, to = -1, step = -1;
    if (inward) {
      from = 0;
      to = ncl;
      step = 1;
    }
    for (int icl = from; icl != to; icl += step) { // ITS clusters are referred in layer decreasing order
      const auto& clus = itsClusters[itsTrackClusRefs[clEntry + icl]];
      float alpha = geomITS->getSensorRefAlpha(clus.getSensorID()), x = clus.getX();
      if (!trFit.rotate(alpha) || !propagator->propagateToX(trFit, x, propagator->getNominalBz(), this->mMatchParams->maxSnp, this->mMatchParams->maxStep, this->mMatchParams->matCorr, nullptr, eLossSign)) {
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
    o2::track::TrackParCov outerLegs[2] = {data.getTrackParamOut(mSeeds[rec.id0].origID), data.getTrackParamOut(mSeeds[rec.id1].origID)};
    for (auto& leg : outerLegs) {
      leg.setPID(o2::track::PID::Muon, true); // as the seeds
    }
    auto tOverlap = mSeeds[rec.id0].tBracket.getOverlap(mSeeds[rec.id1].tBracket);
    float t0 = tOverlap.mean(), dt = tOverlap.delta() * 0.5;
    if (rec.tCommonErr >= 0.f) { // TPC-only legs on opposite sides: their z continuity fixes the time, refit both legs with it
      t0 = rec.tCommon;
      dt = rec.tCommonErr;
    }
    auto pnt0 = outerLegs[0].getXYZGlo(), pnt1 = outerLegs[1].getXYZGlo();
    int btm = 0, top = 1;
    // we fit topward from bottom
    if (pnt0.Y() > pnt1.Y()) {
      btm = 1;
      top = 0;
    }
    LOG(debug) << "Winner " << count++ << " Record " << winRID << " Partners:"
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
      trCosm.resetCovariance();
      // in case of cosmics, constrain the momentum
      if (!mFieldON) {
        trCosm.setQ2Pt(-o2::track::kMostProbablePt);
      }
      int retVal = tpcRefitter->RefitTrackAsTrackParCov(trCosm, tpcTrOrig.getClusterRef(), t0 * tpcTBinMUSInv, &chi2, false, false, ELossGain); // inward refit, reset
      if (retVal < 0) {                                                                                                             // refit failed
        LOG(debug) << "Inward refit of btm TPC track failed.";
        continue;
      }
      nclTot += retVal;
      LOG(debug) << "chi2 after btm TPC refit with " << retVal << " clusters : " << chi2 << " orig.chi2 was " << tpcTrOrig.getChi2();
    } else { // just collect NClusters and chi2
      // since we did not refit bottom track, we just invert its conventional q/pT in case of B=0, so that after the inversion it gets correct sign
      if (!mFieldON) {
        trCosm.setQ2Pt(-trCosm.getQ2Pt());
      }
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
        !o2::base::Propagator::Instance()->PropagateToXBxByBz(trCosm, mSeeds[poolEntryID[top]].getX(), mMatchParams->maxSnp, mMatchParams->maxStep, mMatchParams->matCorr, nullptr, ELossGain)) {
      LOG(debug) << "Rotation/propagation of btm-track to top-track frame failed.";
      continue;
    }
    // save bottom parameter at merging point
    auto trCosmBtm = trCosm;
    int nclBtm = nclTot;

    // Continue with top leg outward refit
    auto gidxListTop = data.getSingleDetectorRefs(mSeeds[poolEntryID[top]].origID);

    // is there ITS sub-track?
    if (gidxListTop[GTrackID::ITS].isIndexSet()) {
      auto nclfit = refitITSTrack(trCosm, gidxListTop[GTrackID::ITS], chi2, false, ELossGain);
      if (nclfit < 0) {
        continue;
      }
      LOG(debug) << "chi2 after top ITS refit with " << nclfit << " clusters : " << chi2 << " orig.chi2 was " << data.getITSTrack(gidxListTop[GTrackID::ITS]).getChi2();
      nclTot += nclfit;
    } // ITS refit
    //
    if (gidxListTop[GTrackID::TPC].isIndexSet()) { // outward refit in TPC
      // go to TPC boundary, if needed
      if (trCosm.getX() * trCosm.getX() + trCosm.getY() * trCosm.getY() <= o2::constants::geom::XTPCInnerRef * o2::constants::geom::XTPCInnerRef) {
        float xtogo = 0;
        if (!trCosm.getXatLabR(o2::constants::geom::XTPCInnerRef, xtogo, mBz, o2::track::DirOutward) ||
            !o2::base::Propagator::Instance()->PropagateToXBxByBz(trCosm, xtogo, mMatchParams->maxSnp, mMatchParams->maxStep, mMatchParams->matCorr, nullptr, ELossGain)) {
          LOG(debug) << "Propagation to inner TPC boundary X=" << xtogo << " failed";
          continue;
        }
      }
      const auto& tpcTrOrig = data.getTPCTrack(gidxListTop[GTrackID::TPC]);
      int retVal = tpcRefitter->RefitTrackAsTrackParCov(trCosm, tpcTrOrig.getClusterRef(), t0 * tpcTBinMUSInv, &chi2, true, false, ELossGain); // outward refit, no reset
      if (retVal < 0) {                                                                                                             // refit failed
        LOG(debug) << "Outward refit of top TPC track failed.";
        continue;
      } // outward refit in TPC
      LOG(debug) << "chi2 after top TPC refit with " << retVal << " clusters : " << chi2 << " orig.chi2 was " << tpcTrOrig.getChi2();
      nclTot += retVal;
    }

    // inward refit of top leg for evaluation in DCA
    float chi2Dummy = 0;
    auto trCosmTop = outerLegs[top];
    if (gidxListTop[GTrackID::TPC].isIndexSet()) { // inward refit in TPC
      const auto& tpcTrOrig = data.getTPCTrack(gidxListTop[GTrackID::TPC]);
      int retVal = tpcRefitter->RefitTrackAsTrackParCov(trCosmTop, tpcTrOrig.getClusterRef(), t0 * tpcTBinMUSInv, &chi2Dummy, false, true, ELossLoss); // inward refit, reset
      if (retVal < 0) {                                                                                                                     // refit failed
        LOG(debug) << "Inward refit of top TPC track failed.";
        continue;
      } // inward refit in TPC
    }
    // is there ITS sub-track ?
    if (gidxListTop[GTrackID::ITS].isIndexSet()) {
      auto nclfit = refitITSTrack(trCosmTop, gidxListTop[GTrackID::ITS], chi2Dummy, true, ELossLoss);
      if (nclfit < 0) {
        continue;
      }
      nclTot += nclfit;
    } // ITS refit
    // propagate to bottom param
    if (!trCosmTop.rotate(trCosmBtm.getAlpha()) ||
        !o2::base::Propagator::Instance()->PropagateToXBxByBz(trCosmTop, trCosmBtm.getX(), mMatchParams->maxSnp, mMatchParams->maxStep, mMatchParams->matCorr, nullptr, ELossLoss)) {
      LOG(debug) << "Rotation/propagation of top-track to bottom-track frame failed.";
      continue;
    }
    // calculate weighted average of 2 legs and chi2
    o2::track::TrackParCov::MatrixDSym5 cov5;
    float chi2Match = trCosmBtm.getPredictedChi2(trCosmTop, cov5);
    if (mMatchParams->maxChi2Match >= 0.f && chi2Match > mMatchParams->maxChi2Match) {
      LOG(debug) << "Top/Bottom refitted legs disagree, chi2Match " << chi2Match;
      continue;
    }
    if (!trCosmBtm.update(trCosmTop, cov5)) {
      LOG(debug) << "Top/Bottom update failed";
      continue;
    }
    // TPC-only legs on opposite sides: the legs' pT is required in checkPair, the refitted cosmic's here
    if (mSeeds[rec.id0].tpcSide * mSeeds[rec.id1].tpcSide < 0 && std::abs(trCosmBtm.getQ2Pt()) > mQ2PtCutoffOppositeSides) {
      LOG(debug) << "Cosmic with legs on opposite TPC sides below minPtOppositeSides";
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
  LOG(info) << "Validated " << mCosmicTracks.size() << " top-bottom tracks in TF# " << mTFCount;
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
    LOGF(info, "iter %d Validated %d of %d remaining matches", iter, nValidated, nRemaining);
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
    LOG(warning) << "Attempt to remove null or validated partner match " << mSeeds[partner1].matchID;
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

  LOG(debug) << "Seed " << i << " [" << seed0.tBracket.getMin() << " : " << seed0.tBracket.getMax() << "] | "
             << "Seed " << j << " [" << seed1.tBracket.getMin() << " : " << seed1.tBracket.getMax() << "] | ";
  LOG(debug) << seed0.origID << " | " << seed0.o2::track::TrackPar::asString();
  LOG(debug) << seed1.origID << " | " << seed1.o2::track::TrackPar::asString();

  if (seed1.tBracket > seed0.tBracket) {
    return (rej = RejTime); // since the brackets are sorted in tmin, all following tbj will also exceed tbi
  }
  float chi2 = 1.e9f;
  float tCommon = 0.f;     // time fixed by z continuity of TPC-only legs on opposite sides, used by the refit
  float tCommonErr = -1.f; // its error (< 0: not fixed)
  TrackSeed seed0Common;   // same-side TPC-only legs refitted at a common time (refitSameSideAtCommonTime)
  TrackSeed seed1Common;
  bool commonTime = false;

  // check
  // 1) crude check on tgl and q/pt (if B!=0). Note: back-to-back tracks will have mutually params (see TrackPar::invertParam)
  while (1) {
    // TPC-only legs on opposite sides: their z continuity defines the time, so z does not reject random pairs of collision tracks; require
    // the pT of a cosmic for both legs already here, so that such a pair cannot win against the true partner of one of its legs
    if (seed0.tpcSide * seed1.tpcSide < 0 && std::max(std::abs(seed0.getQ2Pt()), std::abs(seed1.getQ2Pt())) > mQ2PtCutoffOppositeSides) {
      rej = RejQ2Pt;
      break;
    }
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
    if (mMatchParams->vetoSameHalf) {
      // a cosmic has its two legs on opposite sides of its closest approach to the beam line: project the legs' reference points (before
      // the propagation to the DCA) on the transverse direction of seed0 at its DCA; pieces of one leg are on the same side. Transverse
      // only: the z of a TPC-only track refers to its own time0, so z differences between the legs are meaningless. Skipped for
      // reference points close to the DCA (e.g. ITS-containing tracks), where the sign is undefined.
      std::array<float, 3> pca{};
      seed0.getXYZGlo(pca);
      const float phi = seed0.getAlpha() + std::asin(seed0.getSnp());
      const float dir[2] = {std::cos(phi), std::sin(phi)};
      float proj0 = 0.f;
      float proj1 = 0.f;
      for (int k = 0; k < 2; k++) {
        proj0 += (seed0.xyzRef[k] - pca[k]) * dir[k];
        proj1 += (seed1.xyzRef[k] - pca[k]) * dir[k];
      }
      constexpr float MinDist = 20.f; // cm
      if (std::abs(proj0) > MinDist && std::abs(proj1) > MinDist && proj0 * proj1 > 0.f) {
        rej = RejSameHalf;
        break;
      }
    }
    // TPC-only legs on the same side: the tracker transformed each leg's clusters with its own time0, a guess (the two legs of a cosmic
    // get guesses tens of mus apart), so the legs were distortion-corrected at different z and disagree although they are one track;
    // compare them refitted at one common time, the centre of their brackets' overlap (the time the refit of the winners uses)
    if (mMatchParams->refitSameSideAtCommonTime && seed0.tpcSide != 0 && seed0.tpcSide == seed1.tpcSide) { // tpcSide != 0: TPC-only one-side legs
      const float tPair = seed0.tBracket.getOverlap(seed1.tBracket).mean();
      commonTime = refitSeedAtTime(seed0, tPair, seed0Common) && refitSeedAtTime(seed1, tPair, seed1Common);
    }
    const TrackSeed& leg0 = commonTime ? seed0Common : seed0;
    const TrackSeed& leg1 = commonTime ? seed1Common : seed1;
    o2::track::TrackParCov seed1Inv = leg1;
    seed1Inv.invert();
    for (int i = 0; i < o2::track::kNParams; i++) { // add systematic error
      seed1Inv.updateCov(mMatchParams->systSigma2[i], o2::track::DiagMap[i]);
    }

    if (!seed1Inv.rotate(leg0.getAlpha()) ||
        !o2::base::Propagator::Instance()->PropagateToXBxByBz(seed1Inv, leg0.getX(), mMatchParams->maxSnp, mMatchParams->maxStep, mMatchParams->matCorr)) {
      rej = RejProp;
      break;
    }
    auto dSnp = leg0.getSnp() - seed1Inv.getSnp();
    if (dSnp * dSnp > (leg0.getSigmaSnp2() + seed1Inv.getSigmaSnp2()) * mMatchParams->crudeNSigma2Cut[o2::track::kSnp]) {
      rej = RejSnp;
      break;
    }
    auto dY = leg0.getY() - seed1Inv.getY();
    if (dY * dY > (leg0.getSigmaY2() + seed1Inv.getSigmaY2()) * mMatchParams->crudeNSigma2Cut[o2::track::kY]) {
      rej = RejY;
      break;
    }
    bool ignoreZ = leg0.origID.getSource() == o2d::GlobalTrackID::TPC || leg1.origID.getSource() == o2d::GlobalTrackID::TPC;
    if (ignoreZ && mMatchParams->constrainTPCOnlyZ) {
      // a TPC-only track with clusters on one side has z relative to its time0: z(t) = z + side * vD * (t - tRef); a CE-crossing or
      // non-TPC-only track has an absolute z (side 0). Bring both legs to a common time where possible and test z; for legs on opposite
      // sides, z continuity fixes the common time, which must lie in both time brackets.
      const int side0 = leg0.tpcSide;
      const int side1 = leg1.tpcSide;
      const float sigZ2 = (leg0.getSigmaZ2() + seed1Inv.getSigmaZ2()) * mMatchParams->crudeNSigma2Cut[o2::track::kZ];
      if (side0 == 0 || side1 == 0 || side0 == side1) {
        float dZ = leg0.getZ() - seed1Inv.getZ();
        float dZTimeTol = 0.f;          // the time of a non-TPC absolute leg is only known within its bracket (tRef is the bracket centre)
        if (side0 != 0 && side1 != 0) { // same side: the z offset is fixed by the difference of the reference times
          dZ -= side0 * mTPCVDrift * (leg0.tRef - leg1.tRef);
        } else if (side1 != 0) { // leg0 absolute: move leg1 to the time of leg0
          dZ -= side1 * mTPCVDrift * (leg0.tRef - leg1.tRef);
          if (leg0.origID.getSource() != o2d::GlobalTrackID::TPC) {
            dZTimeTol = 0.5f * mTPCVDrift * leg0.tBracket.delta();
          }
        } else if (side0 != 0) { // leg1 absolute: move leg0 to the time of leg1
          dZ += side0 * mTPCVDrift * (leg1.tRef - leg0.tRef);
          if (leg1.origID.getSource() != o2d::GlobalTrackID::TPC) {
            dZTimeTol = 0.5f * mTPCVDrift * leg1.tBracket.delta();
          }
        }
        const float dZTol = std::sqrt(sigZ2) + dZTimeTol;
        if (dZ * dZ > dZTol * dZTol) {
          rej = RejZ;
          break;
        }
      } else { // opposite sides
        const float t = 0.5f * (side0 * (seed1Inv.getZ() - leg0.getZ()) / mTPCVDrift + leg0.tRef + leg1.tRef);
        const float tTol = std::sqrt(sigZ2) / (2.f * mTPCVDrift);
        if (t < std::max(leg0.tBracket.getMin(), leg1.tBracket.getMin()) - tTol || t > std::min(leg0.tBracket.getMax(), leg1.tBracket.getMax()) + tTol) {
          rej = RejZ;
          break;
        }
        tCommon = t;
        tCommonErr = std::sqrt(leg0.getSigmaZ2() + seed1Inv.getSigmaZ2()) / (2.f * mTPCVDrift);
      }
    }
    // the z of a TPC-only leg refers to its own time0: it is tested above at a common time with constrainTPCOnlyZ, otherwise ignored
    if (!ignoreZ) { // both legs have an absolute z
      auto dZ = leg0.getZ() - seed1Inv.getZ();
      if (dZ * dZ > (leg0.getSigmaZ2() + seed1Inv.getSigmaZ2()) * mMatchParams->crudeNSigma2Cut[o2::track::kZ]) {
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
    chi2 = leg0.getPredictedChi2(seed1Inv);
    if (chi2 > mMatchParams->crudeChi2Cut) {
      rej = RejChi2;
      break;
    }
    rej = Accept;
    registerMatch(i, j, chi2, tCommon, tCommonErr);
    registerMatch(j, i, chi2, tCommon, tCommonErr); // the reverse reference can be also done in a separate loop
    LOG(debug) << "Chi2 = " << chi2 << " NMatches " << mRecords.size();
    break;
  }

#ifdef _ALLOW_DEBUG_TREES_
  if (mDBGOut && ((rej == Accept && isDebugFlag(MatchTreeAccOnly)) || isDebugFlag(MatchTreeAll))) {
    const TrackSeed& dbgLeg0 = commonTime ? seed0Common : seed0; // the legs compared (refitted at the common time if they were)
    auto seed1I = commonTime ? seed1Common : seed1;
    seed1I.invert();
    if (seed1I.rotate(dbgLeg0.getAlpha()) && o2::base::Propagator::Instance()->PropagateToXBxByBz(seed1I, dbgLeg0.getX(), mMatchParams->maxSnp, mMatchParams->maxStep, mMatchParams->matCorr)) {
      int rejI = int(rej);
      int commonTimeI = commonTime;
      (*mDBGOut) << "match"
                 << "tf=" << mTFCount << "seed0=" << dbgLeg0 << "seed1=" << seed1I << "chi2Match=" << chi2 << "rej=" << rejI << "commonTime=" << commonTimeI
                 << "side0=" << int(seed0.tpcSide) << "side1=" << int(seed1.tpcSide) << "tCommon=" << tCommon << "tCommonErr=" << tCommonErr << "\n";
    }
  }
#endif

  return rej;
}

//________________________________________________________
bool MatchCosmics::refitSeedAtTime(const TrackSeed& seed, float timeMUS, TrackSeed& out)
{
  // TPC-only seed refitted with its clusters transformed at the time timeMUS, then treated as the seeds in createSeeds and process() (muon,
  // extra z error, propagation to the DCA to the beam line); energy loss along the muon's flight in the refit and the propagation, as in
  // refitWinners. Its z then refers to timeMUS
  if (!mTPCRefitter || !mRecoData) {
    return false;
  }
  const auto& tpcTrack = mRecoData->getTPCTrack(seed.origID);
  o2::track::TrackParCov trk = tpcTrack.getParamOut();
  trk.setPID(o2::track::PID::Muon, true);
  trk.resetCovariance();
  // the muon flies downward: inward along a leg pointing up (top leg) is along its flight (loss), along the bottom leg against it (gain)
  std::array<float, 3> momentum{};
  trk.getPxPyPzGlo(momentum);
  const int eLossSign = momentum[1] > 0.f ? ELossLoss : ELossGain;
  if (mTPCRefitter->RefitTrackAsTrackParCov(trk, tpcTrack.getClusterRef(), timeMUS / mTPCTBinMUS, nullptr, false, true, eLossSign) < 0) {
    return false;
  }
  trk.setCov(mMatchParams->tpcExtraZError2 + trk.getSigmaZ2(), o2::track::kSigZ2);
  const o2::dataformats::VertexBase v;
  if (!o2::base::Propagator::Instance()->propagateToDCABxByBz(v, trk, mMatchParams->maxStep, mMatchParams->matCorr, nullptr, nullptr, eLossSign)) {
    return false;
  }
  out = seed;
  static_cast<o2::track::TrackParCov&>(out) = trk;
  out.tRef = timeMUS;
  mNRefitsCommonTime++;
  return true;
}

//________________________________________________________
void MatchCosmics::registerMatch(int i, int j, float chi2, float tCommon, float tCommonErr)
{
  /// register track index j as a match for track index i
  int newRef = mRecords.size();
  auto& matchRec = mRecords.emplace_back(MatchRecord{i, j, chi2, MinusOne, tCommon, tCommonErr});
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
  std::unordered_map<GTrackID, int> trackEntry;

  auto creator = [this, &trackEntry](auto& _tr, GTrackID _origID, float t0, float terr) {
    if constexpr (std::is_base_of_v<o2::track::TrackParCov, std::decay_t<decltype(_tr)>>) {
      if (std::abs(_tr.getQ2Pt()) > this->mQ2PtCutoff) {
        return true;
      }
      if constexpr (isTPCTrack<decltype(_tr)>()) {
        if (!this->mMatchParams->allowTPCOnly || _tr.getNClusters() < this->mMatchParams->minSeedNClTPC) {
          return true;
        }
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
      trackEntry[_origID] = mSeeds.size();
      auto& seed = mSeeds.emplace_back(TrackSeed{_tr, {t0 - terr, t0 + terr}, _origID, MinusOne});
      seed.setPID(o2::track::PID::Muon, true); // muon mass and charge for the material corrections, whatever the leg's dE/dx PID
      seed.getXYZGlo(seed.xyzRef);
      seed.tRef = t0;
      if constexpr (isTPCTrack<decltype(_tr)>()) {
        seed.setCov(this->mMatchParams->tpcExtraZError2 + _tr.getCov()[o2::track::kSigZ2], o2::track::kSigZ2);
        seed.tRef = _tr.getTime0() * this->mTPCTBinMUS; // the z of a TPC-only track refers to its time0
        seed.tpcSide = _tr.hasASideClustersOnly() ? 1 : (_tr.hasCSideClustersOnly() ? -1 : 0);
      }
      return true;
    } else {
      return false;
    }
  };

  data.createTracksVariadic(creator);

  if (mUsePVInfo) {                                         // if needed, veto with the primary vertex info
    auto trackIndex = data.getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
    auto vtxRefs = data.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs
    int nv = vtxRefs.size() - 1;                            // The last entry is for unassigned tracks, no need to check them
    const auto propagator = o2::base::Propagator::Instance();
    for (int iv = 0; iv < nv; iv++) {
      const auto& vtref = vtxRefs[iv];
      int it = vtref.getFirstEntry(), itLim = it + vtref.getEntries();
      for (; it < itLim; it++) {
        auto tvid = trackIndex[it];
        auto entry = trackEntry.find(tvid);
        if (entry == trackEntry.end()) {
          continue;
        }
        auto& seed = mSeeds[entry->second];
        if (seed.matchID == Reject || (mMatchParams->discardPVContributors && tvid.isPVContributor())) {
          seed.matchID = Reject;
          continue;
        }
        if (seed.vtIDMin < 0) {
          seed.vtIDMin = iv;
        }
        seed.vtIDMax = std::max(short(iv), seed.vtIDMax);
      }
    }
  }
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
  mQ2PtCutoffOppositeSides = mMatchParams->minPtOppositeSides > 0.f ? 1.f / mMatchParams->minPtOppositeSides : 1e9;
  if (mFieldON) {
    mQ2PtCutoff *= 5.00668 / std::abs(mBz);
    mQ2PtCutoffOppositeSides *= 5.00668 / std::abs(mBz);
  } else {
    mQ2PtCutoff = 1e9;
    mQ2PtCutoffOppositeSides = 1e9;
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
o2::itsmft::ClustersPerLayer<o2::BaseCluster<float>> MatchCosmics::prepareITSClusters(const o2::globaltracking::RecoContainer& data) const
{
  o2::itsmft::ClustersPerLayer<o2::BaseCluster<float>> itscl;
  int nLr = data.getITSPerLayer() ? o2::globaltracking::MaxITSLayers : 1;
  itscl.init(nLr);
  for (int lr = 0; lr < nLr; lr++) {
    itscl.beginLayer(lr);
    const auto& clusITS = data.getITSClusters(lr);
    if (clusITS.size()) {
      auto pattIt = data.getITSClustersPatterns(lr).begin();
      itscl.getClusters().reserve(itscl.size() + clusITS.size());
      o2::its::ioutils::convertCompactClusters(clusITS, pattIt, itscl.getClusters(), mITSDict);
    }
  }
  itscl.finalize();
  return itscl;
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

//______________________________________________
void MatchCosmics::setTPCVDrift(const o2::tpc::VDriftCorrFact& v)
{
  mTPCVDrift = v.refVDrift * v.corrFact;
  mTPCVDriftCorrFact = v.corrFact;
  mTPCVDriftRef = v.refVDrift;
  mTPCDriftTimeOffset = v.getTimeOffset();
}

//______________________________________________
void MatchCosmics::setTPCCorrMaps(const o2::gpu::TPCFastTransformPOD* maph)
{
  mTPCCorrMaps = maph;
}

#endif
