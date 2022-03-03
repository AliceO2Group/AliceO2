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

/// \file SVertexer.cxx
/// \brief Secondary vertex finder
/// \author ruben.shahoyan@cern.ch

#include "DetectorsVertexing/SVertexer.h"
#include "DetectorsBase/Propagator.h"
#include "TPCBase/ParameterGas.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

using namespace o2::vertexing;

using PID = o2::track::PID;
using TrackTPCITS = o2::dataformats::TrackTPCITS;
using TrackITS = o2::its::TrackITS;
using TrackTPC = o2::tpc::TrackTPC;

//__________________________________________________________________
void SVertexer::process(const o2::globaltracking::RecoContainer& recoData) // accessor to various reconstrucred data types
{
  updateTimeDependentParams(); // TODO RS: strictly speaking, one should do this only in case of the CCDB objects update
  mPVertices = recoData.getPrimaryVertices();
  buildT2V(recoData); // build track->vertex refs from vertex->track (if other workflow will need this, consider producing a message in the VertexTrackMatcher)
  int ntrP = mTracksPool[POS].size(), ntrN = mTracksPool[NEG].size(), iThread = 0;
  mV0sTmp[0].clear();
  mCascadesTmp[0].clear();
#ifdef WITH_OPENMP
  int dynGrp = std::min(4, std::max(1, mNThreads / 2));
#pragma omp parallel for schedule(dynamic, dynGrp) num_threads(mNThreads)
#endif
  for (int itp = 0; itp < ntrP; itp++) {
    auto& seedP = mTracksPool[POS][itp];
    int firstN = mVtxFirstTrack[NEG][seedP.vBracket.getMin()];
    if (firstN < 0) {
      LOG(debug) << "No partner is found for pos.track " << itp << " out of " << ntrP;
      continue;
    }
    for (int itn = firstN; itn < ntrN; itn++) { // start from the 1st negative track of lowest-ID vertex of positive
      auto& seedN = mTracksPool[NEG][itn];
      if (seedN.vBracket > seedP.vBracket) { // all vertices compatible with seedN are in future wrt that of seedP
        LOG(debug) << "Brackets do not match";
        break;
      }
#ifdef WITH_OPENMP
      iThread = omp_get_thread_num();
#endif
      if (mSVParams->maxPVContributors < 2 && seedP.gid.isPVContributor() + seedN.gid.isPVContributor() > mSVParams->maxPVContributors) {
        continue;
      }
      checkV0(seedP, seedN, itp, itn, iThread);
    }
  }
  // sort V0s and Cascades in vertex id
  struct vid {
    int thrID;
    int entry;
    int vtxID;
  };
  size_t nv0 = 0, ncsc = 0;
  for (int i = 0; i < mNThreads; i++) {
    nv0 += mV0sTmp[0].size();
    ncsc += mCascadesTmp[i].size();
  }
  std::vector<vid> v0SortID, cascSortID;
  v0SortID.reserve(nv0);
  cascSortID.reserve(ncsc);
  for (int i = 0; i < mNThreads; i++) {
    for (int j = 0; j < (int)mV0sTmp[i].size(); j++) {
      v0SortID.emplace_back(vid{i, j, mV0sTmp[i][j].getVertexID()});
    }
    for (int j = 0; j < (int)mCascadesTmp[i].size(); j++) {
      cascSortID.emplace_back(vid{i, j, mCascadesTmp[i][j].getVertexID()});
    }
  }
  std::sort(v0SortID.begin(), v0SortID.end(), [](const vid& a, const vid& b) { return a.vtxID > b.vtxID; });
  std::sort(cascSortID.begin(), cascSortID.end(), [](const vid& a, const vid& b) { return a.vtxID > b.vtxID; });
  // sorted V0s
  std::vector<V0> bufV0;
  bufV0.reserve(nv0);
  for (const auto& id : v0SortID) {
    auto& v0 = mV0sTmp[id.thrID][id.entry];
    int pos = bufV0.size();
    bufV0.push_back(v0);
    v0.setVertexID(pos); // this v0 copy will be discarded, use its vertexID to store the new position of final V0
  }
  // since V0s were reshuffled, we need to correct the cascade -> V0 reference indices
  for (int i = 0; i < mNThreads; i++) {  // merge results of all threads
    for (auto& casc : mCascadesTmp[i]) { // before merging fix cascades references on v0
      casc.setV0ID(mV0sTmp[i][casc.getV0ID()].getVertexID());
    }
  }
  // sorted Cascades
  std::vector<Cascade> bufCasc;
  bufCasc.reserve(ncsc);
  for (const auto& id : cascSortID) {
    bufCasc.push_back(mCascadesTmp[id.thrID][id.entry]);
  }
  //
  mV0sTmp[0].swap(bufV0);               // the final result is fetched from here
  mCascadesTmp[0].swap(bufCasc);        // the final result is fetched from here
  for (int i = 1; i < mNThreads; i++) { // clean unneeded s.vertices
    mV0sTmp[i].clear();
    mCascadesTmp[i].clear();
  }
  LOG(debug) << "DONE : " << mV0sTmp[0].size() << " " << mCascadesTmp[0].size();
}

//__________________________________________________________________
void SVertexer::init()
{
}

//__________________________________________________________________
void SVertexer::updateTimeDependentParams()
{
  // TODO RS: strictly speaking, one should do this only in case of the CCDB objects update
  mSVParams = &SVertexerParams::Instance();
  // precalculated selection cuts
  mMinR2ToMeanVertex = mSVParams->minRToMeanVertex * mSVParams->minRToMeanVertex;
  mMaxR2ToMeanVertexCascV0 = mSVParams->maxRToMeanVertexCascV0 * mSVParams->maxRToMeanVertexCascV0;
  mMaxDCAXY2ToMeanVertex = mSVParams->maxDCAXYToMeanVertex * mSVParams->maxDCAXYToMeanVertex;
  mMaxDCAXY2ToMeanVertexV0Casc = mSVParams->maxDCAXYToMeanVertexV0Casc * mSVParams->maxDCAXYToMeanVertexV0Casc;
  mMinR2DiffV0Casc = mSVParams->minRDiffV0Casc * mSVParams->minRDiffV0Casc;
  mMinPt2V0 = mSVParams->minPtV0 * mSVParams->minPtV0;
  mMaxTgl2V0 = mSVParams->maxTglV0 * mSVParams->maxTglV0;
  mMinPt2Casc = mSVParams->minPtCasc * mSVParams->minPtCasc;
  mMaxTgl2Casc = mSVParams->maxTglCasc * mSVParams->maxTglCasc;

  auto bz = o2::base::Propagator::Instance()->getNominalBz();

  mV0Hyps[HypV0::Photon].set(PID::Photon, PID::Electron, PID::Electron, mSVParams->pidCutsPhoton, bz);
  mV0Hyps[HypV0::K0].set(PID::K0, PID::Pion, PID::Pion, mSVParams->pidCutsK0, bz);
  mV0Hyps[HypV0::Lambda].set(PID::Lambda, PID::Proton, PID::Pion, mSVParams->pidCutsLambda, bz);
  mV0Hyps[HypV0::AntiLambda].set(PID::Lambda, PID::Pion, PID::Proton, mSVParams->pidCutsLambda, bz);
  mV0Hyps[HypV0::HyperTriton].set(PID::HyperTriton, PID::Helium3, PID::Pion, mSVParams->pidCutsHTriton, bz);
  mV0Hyps[HypV0::AntiHyperTriton].set(PID::HyperTriton, PID::Pion, PID::Helium3, mSVParams->pidCutsHTriton, bz);

  mCascHyps[HypCascade::XiMinus].set(PID::XiMinus, PID::Lambda, PID::Pion, mSVParams->pidCutsXiMinus, bz);
  mCascHyps[HypCascade::OmegaMinus].set(PID::OmegaMinus, PID::Lambda, PID::Kaon, mSVParams->pidCutsOmegaMinus, bz);

  setupThreads();

  for (auto& ft : mFitterV0) {
    ft.setBz(bz);
  }
  for (auto& ft : mFitterCasc) {
    ft.setBz(bz);
  }

  auto& gasParam = o2::tpc::ParameterGas::Instance();
  mTPCBin2Z = gasParam.DriftV / mMUS2TPCBin;
}

//__________________________________________________________________
void SVertexer::setupThreads()
{
  if (!mV0sTmp.empty()) {
    return;
  }
  mV0sTmp.resize(mNThreads);
  mCascadesTmp.resize(mNThreads);
  mFitterV0.resize(mNThreads);
  auto bz = o2::base::Propagator::Instance()->getNominalBz();
  for (auto& fitter : mFitterV0) {
    fitter.setBz(bz);
    fitter.setUseAbsDCA(mSVParams->useAbsDCA);
    fitter.setPropagateToPCA(false);
    fitter.setMaxR(mSVParams->maxRIni);
    fitter.setMinParamChange(mSVParams->minParamChange);
    fitter.setMinRelChi2Change(mSVParams->minRelChi2Change);
    fitter.setMaxDZIni(mSVParams->maxDZIni);
    fitter.setMaxChi2(mSVParams->maxChi2);
    fitter.setMatCorrType(o2::base::Propagator::MatCorrType(mSVParams->matCorr));
    fitter.setUsePropagator(mSVParams->usePropagator);
    fitter.setRefitWithMatCorr(mSVParams->refitWithMatCorr);
    fitter.setMaxStep(mSVParams->maxStep);
    fitter.setMaxSnp(mSVParams->maxSnp);
    fitter.setMinXSeed(mSVParams->minXSeed);
  }
  mFitterCasc.resize(mNThreads);
  for (auto& fitter : mFitterCasc) {
    fitter.setBz(bz);
    fitter.setUseAbsDCA(mSVParams->useAbsDCA);
    fitter.setPropagateToPCA(false);
    fitter.setMaxR(mSVParams->maxRIniCasc);
    fitter.setMinParamChange(mSVParams->minParamChange);
    fitter.setMinRelChi2Change(mSVParams->minRelChi2Change);
    fitter.setMaxDZIni(mSVParams->maxDZIni);
    fitter.setMaxChi2(mSVParams->maxChi2);
    fitter.setMatCorrType(o2::base::Propagator::MatCorrType(mSVParams->matCorr));
    fitter.setUsePropagator(mSVParams->usePropagator);
    fitter.setRefitWithMatCorr(mSVParams->refitWithMatCorr);
    fitter.setMaxStep(mSVParams->maxStep);
    fitter.setMaxSnp(mSVParams->maxSnp);
    fitter.setMinXSeed(mSVParams->minXSeed);
  }
}

//__________________________________________________________________
bool SVertexer::acceptTrack(GIndex gid, const o2::track::TrackParCov& trc) const
{
  if (gid.isPVContributor() && mSVParams->maxPVContributors < 1) {
    return false;
  }
  // DCA to mean vertex
  if (mSVParams->minDCAToPV > 0.f) {
    o2::track::TrackPar trp(trc);
    std::array<float, 2> dca;
    auto* prop = o2::base::Propagator::Instance();
    if (mSVParams->usePropagator) {
      if (trp.getX() > mSVParams->minRFor3DField && !prop->PropagateToXBxByBz(trp, mSVParams->minRFor3DField, mSVParams->maxSnp, mSVParams->maxStep, o2::base::Propagator::MatCorrType(mSVParams->matCorr))) {
        return true; // we don't need actually to propagate to the beam-line
      }
      if (!prop->propagateToDCA(mMeanVertex.getXYZ(), trp, prop->getNominalBz(), mSVParams->maxStep, o2::base::Propagator::MatCorrType(mSVParams->matCorr), &dca)) {
        return true;
      }
    } else {
      if (!trp.propagateParamToDCA(mMeanVertex.getXYZ(), prop->getNominalBz(), &dca)) {
        return true;
      }
    }
    if (std::abs(dca[0]) < mSVParams->minDCAToPV) {
      return false;
    }
  }
  return true;
}

//__________________________________________________________________
void SVertexer::buildT2V(const o2::globaltracking::RecoContainer& recoData) // accessor to various tracks
{
  // build track->vertices from vertices->tracks, rejecting vertex contributors if requested
  auto trackIndex = recoData.getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
  auto vtxRefs = recoData.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs

  std::unordered_map<GIndex, std::pair<int, int>> tmap;
  std::unordered_map<GIndex, bool> rejmap;
  int nv = vtxRefs.size() - 1; // The last entry is for unassigned tracks, ignore them
  for (int i = 0; i < 2; i++) {
    mTracksPool[i].clear();
    mVtxFirstTrack[i].clear();
    mVtxFirstTrack[i].resize(nv, -1);
  }
  for (int iv = 0; iv < nv; iv++) {
    const auto& vtref = vtxRefs[iv];
    int it = vtref.getFirstEntry(), itLim = it + vtref.getEntries();
    for (; it < itLim; it++) {
      auto tvid = trackIndex[it];
      if (!recoData.isTrackSourceLoaded(tvid.getSource())) {
        continue;
      }
      // unconstrained TPC tracks require special treatment: there is no point in checking DCA to mean vertex since it is not precise,
      // but we need to create a clone of TPC track constrained to this particular vertex time.
      if (tvid.getSource() == GIndex::TPC && processTPCTrack(recoData.getTPCTrack(tvid), tvid, iv)) { // processTPCTrack may decide that this track does not need special treatment (e.g. it is constrained...)
        continue;
      }
      std::decay_t<decltype(tmap.find(tvid))> tref{};
      if (tvid.isAmbiguous()) { // was this track already processed?
        auto tref = tmap.find(tvid);
        if (tref != tmap.end()) {
          mTracksPool[tref->second.second][tref->second.first].vBracket.setMax(iv); // this track was already processed with other vertex, account the latter
          continue;
        }
        // was it already rejected?
        if (rejmap.find(tvid) != rejmap.end()) {
          continue;
        }
      }
      const auto& trc = recoData.getTrackParam(tvid);
      if (!acceptTrack(tvid, trc)) {
        if (tvid.isAmbiguous()) {
          rejmap[tvid] = true;
        }
        continue;
      }
      int posneg = trc.getSign() < 0 ? 1 : 0;
      float r = std::sqrt(trc.getX() * trc.getX() + trc.getY() * trc.getY());
      mTracksPool[posneg].emplace_back(TrackCand{trc, tvid, {iv, iv}, r});
      if (tvid.isAmbiguous()) { // track attached to >1 vertex, remember that it was already processed
        tmap[tvid] = {mTracksPool[posneg].size() - 1, posneg};
      }
    }
  }
  // register 1st track of each charge for each vertex

  for (int pn = 0; pn < 2; pn++) {
    auto& vtxFirstT = mVtxFirstTrack[pn];
    const auto& tracksPool = mTracksPool[pn];
    for (unsigned i = 0; i < tracksPool.size(); i++) {
      const auto& t = tracksPool[i];
      if (vtxFirstT[t.vBracket.getMin()] == -1) {
        vtxFirstT[t.vBracket.getMin()] = i;
      }
    }
  }

  LOG(info) << "Collected " << mTracksPool[POS].size() << " positive and " << mTracksPool[NEG].size() << " negative seeds";
}

//__________________________________________________________________
bool SVertexer::checkV0(const TrackCand& seedP, const TrackCand& seedN, int iP, int iN, int ithread)
{
  auto& fitterV0 = mFitterV0[ithread];
  int nCand = fitterV0.process(seedP, seedN);
  if (nCand == 0) { // discard this pair
    return false;
  }
  const auto& v0XYZ = fitterV0.getPCACandidate();
  // validate V0 radial position
  // check closeness to the beam-line
  float dxv0 = v0XYZ[0] - mMeanVertex.getX(), dyv0 = v0XYZ[1] - mMeanVertex.getY(), r2v0 = dxv0 * dxv0 + dyv0 * dyv0;
  if (r2v0 < mMinR2ToMeanVertex) {
    return false;
  }
  float rv0 = std::sqrt(r2v0), drv0P = rv0 - seedP.minR, drv0N = rv0 - seedN.minR;
  if (drv0P > mSVParams->causalityRTolerance || drv0P < -mSVParams->maxV0ToProngsRDiff ||
      drv0N > mSVParams->causalityRTolerance || drv0N < -mSVParams->maxV0ToProngsRDiff) {
    LOG(debug) << "RejCausality " << drv0P << " " << drv0N;
    return false;
  }

  if (!fitterV0.isPropagateTracksToVertexDone() && !fitterV0.propagateTracksToVertex()) {
    return false;
  }
  int cand = 0;
  auto& trPProp = fitterV0.getTrack(0, cand);
  auto& trNProp = fitterV0.getTrack(1, cand);
  std::array<float, 3> pP, pN;
  trPProp.getPxPyPzGlo(pP);
  trNProp.getPxPyPzGlo(pN);
  // estimate DCA of neutral V0 track to beamline: straight line with parametric equation
  // x = X0 + pV0[0]*t, y = Y0 + pV0[1]*t reaches DCA to beamline (Xv, Yv) at
  // t = -[ (x0-Xv)*pV0[0] + (y0-Yv)*pV0[1]) ] / ( pT(pV0)^2 )
  // Similar equation for 3D distance involving pV0[2]
  std::array<float, 3> pV0 = {pP[0] + pN[0], pP[1] + pN[1], pP[2] + pN[2]};
  float pt2V0 = pV0[0] * pV0[0] + pV0[1] * pV0[1], prodXYv0 = dxv0 * pV0[0] + dyv0 * pV0[1], tDCAXY = prodXYv0 / pt2V0;
  if (pt2V0 < mMinPt2V0) { // pt cut
    LOG(debug) << "RejPt2 " << pt2V0;
    return false;
  }
  if (pV0[2] * pV0[2] / pt2V0 > mMaxTgl2V0) { // tgLambda cut
    LOG(debug) << "RejTgL " << pV0[2] * pV0[2] / pt2V0;
    return false;
  }
  float p2V0 = pt2V0 + pV0[2] * pV0[2], ptV0 = std::sqrt(pt2V0);
  // apply mass selections
  float p2Pos = pP[0] * pP[0] + pP[1] * pP[1] + pP[2] * pP[2], p2Neg = pN[0] * pN[0] + pN[1] * pN[1] + pN[2] * pN[2];

  bool goodHyp = false;
  std::array<bool, NHypV0> hypCheckStatus{};
  for (int ipid = 0; ipid < NHypV0; ipid++) {
    if (mV0Hyps[ipid].check(p2Pos, p2Neg, p2V0, ptV0)) {
      goodHyp = hypCheckStatus[ipid] = true;
    }
  }
  if (!goodHyp && mSVParams->checkV0Hypothesis) {
    LOG(debug) << "RejHypo";
    return false;
  }

  bool checkForCascade = mEnableCascades && r2v0 < mMaxR2ToMeanVertexCascV0 && (!mSVParams->checkV0Hypothesis || (hypCheckStatus[HypV0::Lambda] || hypCheckStatus[HypV0::AntiLambda]));
  bool rejectIfNotCascade = false;
  float dcaX = dxv0 - pV0[0] * tDCAXY, dcaY = dyv0 - pV0[1] * tDCAXY, dca2 = dcaX * dcaX + dcaY * dcaY;
  float cosPAXY = prodXYv0 / std::sqrt(r2v0 * pt2V0);

  if (checkForCascade) { // use loser cuts for cascade v0 candidates
    if (dca2 > mMaxDCAXY2ToMeanVertexV0Casc || cosPAXY < mSVParams->minCosPAXYMeanVertexCascV0) {
      LOG(debug) << "Rej for cascade DCAXY2: " << dca2 << " << cosPAXY: " << cosPAXY;
      return false;
    }
  }
  if (dca2 > mMaxDCAXY2ToMeanVertex || cosPAXY < mSVParams->minCosPAXYMeanVertex) {
    if (checkForCascade) {
      rejectIfNotCascade = true;
    } else {
      return false;
    }
  }

  auto vlist = seedP.vBracket.getOverlap(seedN.vBracket); // indices of vertices shared by both seeds
  bool added = false;
  auto bestCosPA = checkForCascade ? mSVParams->minCosPACascV0 : mSVParams->minCosPA;
  for (int iv = vlist.getMin(); iv <= vlist.getMax(); iv++) {
    const auto& pv = mPVertices[iv];
    const auto v0XYZ = fitterV0.getPCACandidatePos(cand);
    // check cos of pointing angle
    float dx = v0XYZ[0] - pv.getX(), dy = v0XYZ[1] - pv.getY(), dz = v0XYZ[2] - pv.getZ(), prodXYZv0 = dx * pV0[0] + dy * pV0[1] + dz * pV0[2];
    float cosPA = prodXYZv0 / std::sqrt((dx * dx + dy * dy + dz * dz) * p2V0);
    if (cosPA < bestCosPA) {
      LOG(debug) << "Rej. cosPA: " << cosPA;
      continue;
    }
    if (!added) {
      auto& v0new = mV0sTmp[ithread].emplace_back(v0XYZ, pV0, fitterV0.calcPCACovMatrixFlat(cand), trPProp, trNProp, seedP.gid, seedN.gid);
      v0new.setDCA(fitterV0.getChi2AtPCACandidate());
      added = true;
    }
    auto& v0 = mV0sTmp[ithread].back();
    v0.setCosPA(cosPA);
    v0.setVertexID(iv);
    bestCosPA = cosPA;
  }
  if (!added) {
    return false;
  }
  auto& v0 = mV0sTmp[ithread].back();

  // check cascades
  if (checkForCascade) {
    int nCascAdded = 0;
    if (hypCheckStatus[HypV0::Lambda] || !mSVParams->checkCascadeHypothesis) {
      nCascAdded += checkCascades(rv0, pV0, p2V0, iN, NEG, ithread);
    }
    if (hypCheckStatus[HypV0::AntiLambda] || !mSVParams->checkCascadeHypothesis) {
      nCascAdded += checkCascades(rv0, pV0, p2V0, iP, POS, ithread);
    }
    if (!nCascAdded && rejectIfNotCascade) { // v0 would be accepted only if it creates a cascade
      mV0sTmp[ithread].pop_back();
      return false;
    }
  }

  return true;
}

//__________________________________________________________________
int SVertexer::checkCascades(float rv0, std::array<float, 3> pV0, float p2V0, int avoidTrackID, int posneg, int ithread)
{
  // check last added V0 for belonging to cascade
  auto& fitterCasc = mFitterCasc[ithread];
  const auto& v0 = mV0sTmp[ithread].back();
  auto& tracks = mTracksPool[posneg];
  const auto& pv = mPVertices[v0.getVertexID()];
  int nCascIni = mCascadesTmp[ithread].size();
  // start from the 1st track compatible with V0's primary vertex
  int firstTr = mVtxFirstTrack[posneg][v0.getVertexID()], nTr = tracks.size();
  if (firstTr < 0) {
    firstTr = nTr;
  }
  for (int it = firstTr; it < nTr; it++) {
    if (it == avoidTrackID) {
      continue; // skip the track used by V0
    }
    auto& bach = tracks[it];
    if (bach.vBracket > v0.getVertexID()) {
      break; // all other bachelor candidates will be also not compatible with this PV
    }
    if (bach.vBracket.isOutside(v0.getVertexID())) {
      LOG(error) << "Incompatible bachelor: PV " << bach.vBracket.asString() << " vs V0 " << v0.getVertexID();
    }
    if (bach.minR > rv0 + mSVParams->causalityRTolerance) {
      continue;
    }
    int nCandC = fitterCasc.process(v0, bach);
    if (nCandC == 0) { // discard this pair
      continue;
    }
    int candC = 0;
    const auto& cascXYZ = fitterCasc.getPCACandidatePos(candC);
    // make sure the cascade radius is smaller than that of the vertex
    float dxc = cascXYZ[0] - pv.getX(), dyc = cascXYZ[1] - pv.getY(), dzc = cascXYZ[2] - pv.getZ(), r2casc = dxc * dxc + dyc * dyc;
    if (rv0 * rv0 - r2casc < mMinR2DiffV0Casc || r2casc < mMinR2ToMeanVertex) {
      continue;
    }
    // do we want to apply mass cut ?
    //
    if (!fitterCasc.isPropagateTracksToVertexDone() && !fitterCasc.propagateTracksToVertex()) {
      continue;
    }
    auto& trNeut = fitterCasc.getTrack(0, candC);
    auto& trBach = fitterCasc.getTrack(1, candC);
    trNeut.setPID(o2::track::PID::Lambda);
    trBach.setPID(o2::track::PID::Pion);
    std::array<float, 3> pNeut, pBach;
    trNeut.getPxPyPzGlo(pNeut);
    trBach.getPxPyPzGlo(pBach);
    std::array<float, 3> pCasc = {pNeut[0] + pBach[0], pNeut[1] + pBach[1], pNeut[2] + pBach[2]};
    auto prodPPos = pV0[0] * dxc + pV0[1] * dyc + pV0[2] * dzc;
    if (prodPPos < 0.) { // causality cut
      continue;
    }
    float pt2Casc = pCasc[0] * pCasc[0] + pCasc[1] * pCasc[1], p2Casc = pt2Casc + pCasc[2] * pCasc[2];
    if (pt2Casc < mMinPt2Casc) { // pt cut
      continue;
    }
    if (pCasc[2] * pCasc[2] / pt2Casc > mMaxTgl2Casc) { // tgLambda cut
      continue;
    }
    //    LOG(info) << "ptcasc2 " << pt2Casc << " tglcasc2 " << pCasc[2]*pCasc[2] / pt2Casc << " cut " << mMaxTgl2Casc;
    float cosPA = (pCasc[0] * dxc + pCasc[1] * dyc + pCasc[2] * dzc) / std::sqrt(p2Casc * (r2casc + dzc * dzc));
    if (cosPA < mSVParams->minCosPACasc) {
      continue;
    }
    float p2Bach = pBach[0] * pBach[0] + pBach[1] * pBach[1] + pBach[2] * pBach[2];
    float ptCasc = std::sqrt(pt2Casc);
    bool goodHyp = false;
    for (int ipid = 0; ipid < NHypCascade; ipid++) {
      if (mCascHyps[ipid].check(p2V0, p2Bach, p2Casc, ptCasc)) {
        goodHyp = true;
        break;
      }
    }
    if (!goodHyp) {
      continue;
    }
    auto& casc = mCascadesTmp[ithread].emplace_back(cascXYZ, pCasc, fitterCasc.calcPCACovMatrixFlat(candC), trNeut, trBach, mV0sTmp[ithread].size() - 1, bach.gid);
    o2::track::TrackParCov trc = casc;
    o2::dataformats::DCA dca;
    if (!trc.propagateToDCA(pv, fitterCasc.getBz(), &dca, 5.) ||
        std::abs(dca.getY()) > mSVParams->maxDCAXYCasc || std::abs(dca.getZ()) > mSVParams->maxDCAZCasc) {
      mCascadesTmp[ithread].pop_back();
      continue;
    }
    casc.setCosPA(cosPA);
    casc.setVertexID(v0.getVertexID());
    casc.setDCA(fitterCasc.getChi2AtPCACandidate());
  }
  return mCascadesTmp[ithread].size() - nCascIni;
}

//__________________________________________________________________
void SVertexer::setNThreads(int n)
{
#ifdef WITH_OPENMP
  mNThreads = n > 0 ? n : 1;
#else
  mNThreads = 1;
#endif
}

//______________________________________________
bool SVertexer::processTPCTrack(const o2::tpc::TrackTPC& trTPC, GIndex gid, int vtxid)
{
  // if TPC trackis unconstrained, try to create in the tracks pool a clone constrained to vtxid vertex time.
  if (trTPC.hasBothSidesClusters()) { // this is effectively constrained track
    return false;                     // let it be processed as such
  }
  const auto& vtx = mPVertices[vtxid];
  auto twe = vtx.getTimeStamp();
  int posneg = trTPC.getSign() < 0 ? 1 : 0;
  auto trLoc = mTracksPool[posneg].emplace_back(TrackCand{trTPC, gid, {vtxid, vtxid}, 0.});
  auto err = correctTPCTrack(trLoc, trTPC, twe.getTimeStamp(), twe.getTimeStampError());
  if (err < 0) {
    mTracksPool[posneg].pop_back(); // discard
  }
  trLoc.minR = std::sqrt(trLoc.getX() * trLoc.getX() + trLoc.getY() * trLoc.getY());
  return true;
}

//______________________________________________
float SVertexer::correctTPCTrack(o2::track::TrackParCov& trc, const o2::tpc::TrackTPC tTPC, float tmus, float tmusErr) const
{
  // Correct the track copy trc of the TPC track for the assumed interaction time
  // return extra uncertainty in Z due to the interaction time uncertainty
  // TODO: at the moment, apply simple shift, but with Z-dependent calibration we may
  // need to do corrections on TPC cluster level and refit
  // This is a clone of MatchTPCITS::correctTPCTrack
  float dDrift = (tmus * mMUS2TPCBin - tTPC.getTime0()) * mTPCBin2Z;
  float driftErr = tmusErr * mMUS2TPCBin * mTPCBin2Z;
  // eventually should be refitted, at the moment we simply shift...
  trc.setZ(tTPC.getZ() + (tTPC.hasASideClustersOnly() ? dDrift : -dDrift));
  trc.setCov(trc.getSigmaZ2() + driftErr * driftErr, o2::track::kSigZ2);

  return driftErr;
}
