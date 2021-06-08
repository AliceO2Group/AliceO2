// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file SVertexer.cxx
/// \brief Secondary vertex finder
/// \author ruben.shahoyan@cern.ch

#include "DetectorsVertexing/SVertexer.h"
#include "DetectorsBase/Propagator.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsITS/TrackITS.h"

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
  omp_set_num_threads(mNThreads);
  int dynGrp = std::min(4, std::max(1, mNThreads / 2));
#pragma omp parallel for schedule(dynamic, dynGrp)
#endif
  for (int itp = 0; itp < ntrP; itp++) {
    auto& seedP = mTracksPool[POS][itp];
    for (int itn = mVtxFirstTrack[NEG][seedP.vBracket.getMin()]; itn < ntrN; itn++) { // start from the 1st negative track of lowest-ID vertex of positive
      auto& seedN = mTracksPool[NEG][itn];
      if (seedN.vBracket > seedP.vBracket) { // all vertices compatible with seedN are in future wrt that of seedP
        break;
      }
#ifdef WITH_OPENMP
      iThread = omp_get_thread_num();
#endif
      checkV0(seedP, seedN, itp, itn, iThread);
    }
  }
#ifdef WITH_OPENMP
  for (int i = 1; i < mNThreads; i++) { // merge results of all threads
    for (auto& casc : mCascadesTmp[i]) { // before merging fix cascades references on v0
      casc.setV0ID(casc.getV0ID() + mV0sTmp[0].size());
    }
    mV0sTmp[0].insert(mV0sTmp[0].end(), mV0sTmp[i].begin(), mV0sTmp[i].end());
    mCascadesTmp[0].insert(mCascadesTmp[0].end(), mCascadesTmp[i].begin(), mCascadesTmp[i].end());
    mV0sTmp[i].clear();
    mCascadesTmp[i].clear();
  }
#endif
  LOG(INFO) << "DONE : " << mV0sTmp[0].size() << " " << mCascadesTmp[0].size();
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
  }
}

//__________________________________________________________________
void SVertexer::buildT2V(const o2::globaltracking::RecoContainer& recoData) // accessor to various tracks
{
  // build track->vertices from vertices->tracks, rejecting vertex contributors
  auto trackIndex = recoData.getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
  auto vtxRefs = recoData.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs

  // track selector: at the moment reject prompt tracks contributing to vertex fit and unconstrained TPC tracks
  auto selTrack = [&](GIndex gid) {
    return (gid.isPVContributor() || !recoData.isTrackSourceLoaded(gid.getSource())) ? false : true;
  };

  std::unordered_map<GIndex, std::pair<int, int>> tmap;
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
      if (!selTrack(tvid)) {
        continue;
      }
      std::decay_t<decltype(tmap.find(tvid))> tref{};
      if (tvid.isAmbiguous()) {
        auto tref = tmap.find(tvid);
        if (tref != tmap.end()) {
          mTracksPool[tref->second.second][tref->second.first].vBracket.setMax(iv); // this track was already processed with other vertex, account the latter
          continue;
        }
      }
      const auto& trc = recoData.getTrackParam(tvid);
      int posneg = trc.getSign() < 0 ? 1 : 0;
      mTracksPool[posneg].emplace_back(TrackCand{trc, tvid, {iv, iv}});
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

  LOG(INFO) << "Collected " << mTracksPool[POS].size() << " positive and " << mTracksPool[NEG].size() << " negative seeds";
}

//__________________________________________________________________
bool SVertexer::checkV0(TrackCand& seedP, TrackCand& seedN, int iP, int iN, int ithread)
{
  auto& fitterV0 = mFitterV0[ithread];
  int nCand = fitterV0.process(seedP, seedN);
  if (nCand == 0) { // discard this pair
    return false;
  }
  const auto& v0XYZ = fitterV0.getPCACandidate();
  // check closeness to the beam-line
  float dxv0 = v0XYZ[0] - mMeanVertex.getX(), dyv0 = v0XYZ[1] - mMeanVertex.getY(), r2v0 = dxv0 * dxv0 + dyv0 * dyv0;
  if (r2v0 < mMinR2ToMeanVertex) {
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
  if (!goodHyp) {
    return false;
  }

  bool checkForCascade = mEnableCascades && r2v0 < mMaxR2ToMeanVertexCascV0 && (hypCheckStatus[HypV0::Lambda] || hypCheckStatus[HypV0::AntiLambda]);
  bool rejectIfNotCascade = false;
  float dcaX = dxv0 - pV0[0] * tDCAXY, dcaY = dyv0 - pV0[1] * tDCAXY, dca2 = dcaX * dcaX + dcaY * dcaY;
  float cosPAXY = prodXYv0 / std::sqrt(r2v0 * pt2V0);

  if (checkForCascade) { // use loser cuts for cascade v0 candidates
    if (dca2 > mMaxDCAXY2ToMeanVertexV0Casc || cosPAXY < mSVParams->minCosPAXYMeanVertexCascV0) {
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
  if (vlist.isInvalid()) {
    LOG(WARNING) << "Incompatible tracks: V0 " << seedP.vBracket.asString() << " | V1 " << seedN.vBracket.asString();
    return false;
  }

  bool added = false;
  auto bestCosPA = checkForCascade ? mSVParams->minCosPACascV0 : mSVParams->minCosPA;
  for (int iv = vlist.getMin(); iv <= vlist.getMax(); iv++) {
    const auto& pv = mPVertices[iv];
    const auto v0XYZ = fitterV0.getPCACandidatePos(cand);
    // check cos of pointing angle
    float dx = v0XYZ[0] - pv.getX(), dy = v0XYZ[1] - pv.getY(), dz = v0XYZ[2] - pv.getZ(), prodXYZv0 = dx * pV0[0] + dy * pV0[1] + dz * pV0[2];
    float cosPA = prodXYZv0 / std::sqrt((dx * dx + dy * dy + dz * dz) * p2V0);
    if (cosPA < bestCosPA) {
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
    if (hypCheckStatus[HypV0::Lambda]) {
      nCascAdded += checkCascades(r2v0, pV0, p2V0, iN, NEG, ithread);
    }
    if (hypCheckStatus[HypV0::AntiLambda]) {
      nCascAdded += checkCascades(r2v0, pV0, p2V0, iP, POS, ithread);
    }
    if (!nCascAdded && rejectIfNotCascade) { // v0 would be accepted only if it creates a cascade
      mV0sTmp[ithread].pop_back();
      return false;
    }
  }

  return true;
}

//__________________________________________________________________
int SVertexer::checkCascades(float r2v0, std::array<float, 3> pV0, float p2V0, int avoidTrackID, int posneg, int ithread)
{
  // check last added V0 for belonging to cascade
  auto& fitterCasc = mFitterCasc[ithread];
  const auto& v0 = mV0sTmp[ithread].back();
  auto& tracks = mTracksPool[posneg];
  const auto& pv = mPVertices[v0.getVertexID()];
  int nCascIni = mCascadesTmp[ithread].size();
  // start from the 1st track compatible with V0's primary vertex
  for (unsigned it = mVtxFirstTrack[posneg][v0.getVertexID()]; it < tracks.size(); it++) {
    if (it == avoidTrackID) {
      continue; // skip the track used by V0
    }
    auto& bach = tracks[it];
    if (bach.vBracket > v0.getVertexID()) {
      break; // all other bachelor candidates will be also not compatible with this PV
    }
    if (bach.vBracket.isOutside(v0.getVertexID())) {
      LOG(ERROR) << "Incompatible bachelor: PV " << bach.vBracket.asString() << " vs V0 " << v0.getVertexID();
    }
    int nCandC = fitterCasc.process(v0, bach);
    if (nCandC == 0) { // discard this pair
      continue;
    }
    int candC = 0;
    const auto& cascXYZ = fitterCasc.getPCACandidatePos(candC);
    // make sure the cascade radius is smaller than that of the vertex
    float dxc = cascXYZ[0] - pv.getX(), dyc = cascXYZ[1] - pv.getY(), dzc = cascXYZ[2] - pv.getZ(), r2casc = dxc * dxc + dyc * dyc;
    if (r2v0 - r2casc < mMinR2DiffV0Casc || r2casc < mMinR2ToMeanVertex) {
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
