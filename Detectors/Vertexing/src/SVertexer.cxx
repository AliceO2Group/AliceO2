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
void SVertexer::process(const gsl::span<const PVertex>& vertices,            // primary vertices
                        const gsl::span<const GIndex>& trackIndex,           // Global ID's for associated tracks
                        const gsl::span<const VRef>& vtxRefs,                // references from vertex to these track IDs
                        const o2::globaltracking::RecoContainer& recoTracks, // accessor to various tracks
                        std::vector<V0>& v0s,                                // found V0s
                        std::vector<RRef>& vtx2V0refs                        // references from PVertex to V0
)
{
  std::unordered_map<uint64_t, int> cache; // cache for tested combinations, the value >0 will give the entry of prevalidated V0 in the v0sTmp
  std::vector<V0> v0sTmp(1);               // 1st one is dummy!
  std::vector<int> v0sIdx;                 // id's in v0sTmp used attached to p.vertices
  std::vector<RRef> pv2v0sRefs;            // p.vertex to v0 index references
  std::vector<char> selQ(trackIndex.size(), 0);

  mPVertices = vertices;
  mV0sTmp[0].clear();
  auto tracksPool = buildT2V(trackIndex, vtxRefs, recoTracks);
  // V0 finding
  int ntr = tracksPool.size();
#ifdef WITH_OPENMP
  omp_set_num_threads(mNThreads);
  int dynGrp = std::min(4, std::max(1, mNThreads / 2));
#pragma omp parallel for schedule(dynamic, dynGrp)
#endif
  for (int it0 = 0; it0 < ntr; it0++) {
    auto& seed0 = tracksPool[it0];
    for (int it1 = it0 + 1; it1 < ntr; it1++) {
      auto& seed1 = tracksPool[it1];
      if (seed1.vBracket > seed0.vBracket) { // all vertices compatible with seed1 are in future wrt that of seed0
        break;
      }
      if (seed0.getSign() == seed1.getSign()) { // make sure the seeds have opPite charges
        continue;
      }
      auto seedP = &seed0, seedN = &seed1;
      if (seed0.getSign() < seed1.getSign()) { // order to + , -
        std::swap(seedP, seedN);
      }
#ifdef WITH_OPENMP
      int ith = omp_get_thread_num();
#else
      int ith = 0;
#endif
      checkV0Pair(*seedP, *seedN, ith);
    }
  }
#ifdef WITH_OPENMP
  for (int i = 1; i < mNThreads; i++) { // merge results of all threads
    mV0sTmp[0].insert(mV0sTmp[0].end(), mV0sTmp[i].begin(), mV0sTmp[i].end());
    mV0sTmp[i].clear();
  }
#endif

  vtx2V0refs.clear();
  vtx2V0refs.resize(vertices.size());
  finalizeV0s(v0s, vtx2V0refs);
}

//__________________________________________________________________
void SVertexer::init()
{
  mSVParams = &SVertexerParams::Instance();

  // precalculated selection cuts
  mMinR2ToMeanVertex = mSVParams->minRfromMeanVertex * mSVParams->minRfromMeanVertex;
  mMaxDCAXY2ToMeanVertex = mSVParams->maxDCAXYfromMeanVertex * mSVParams->maxDCAXYfromMeanVertex;
  mMinCosPointingAngle = mSVParams->minCosPointingAngle;
  //
  auto bz = o2::base::Propagator::Instance()->getNominalBz();
  //
  setupThreads();
  mV0Hyps[SVertexerParams::Photon].set(PID::Photon, PID::Electron, PID::Electron, mSVParams->pidCutsPhoton, bz);
  mV0Hyps[SVertexerParams::K0].set(PID::K0, PID::Pion, PID::Pion, mSVParams->pidCutsK0, bz);
  mV0Hyps[SVertexerParams::Lambda].set(PID::Lambda, PID::Proton, PID::Pion, mSVParams->pidCutsLambda, bz);
  mV0Hyps[SVertexerParams::AntiLambda].set(PID::Lambda, PID::Pion, PID::Proton, mSVParams->pidCutsLambda, bz);
  mV0Hyps[SVertexerParams::HyperTriton].set(PID::HyperTriton, PID::Helium3, PID::Pion, mSVParams->pidCutsHTriton, bz);
  mV0Hyps[SVertexerParams::AntiHyperTriton].set(PID::HyperTriton, PID::Pion, PID::Helium3, mSVParams->pidCutsHTriton, bz);
  //
}

//__________________________________________________________________
void SVertexer::setupThreads()
{
  mFitter2Prong.resize(mNThreads);
  mV0sTmp.resize(mNThreads);
  auto bz = o2::base::Propagator::Instance()->getNominalBz();
  for (auto& fitter : mFitter2Prong) {
    fitter.setBz(bz);
    fitter.setUseAbsDCA(mSVParams->useAbsDCA);
    fitter.setPropagateToPCA(false);
    fitter.setMaxR(mSVParams->maxRIni);
    fitter.setMinParamChange(mSVParams->minParamChange);
    fitter.setMinRelChi2Change(mSVParams->minRelChi2Change);
    fitter.setMaxDZIni(mSVParams->maxDZIni);
    fitter.setMaxChi2(mSVParams->maxChi2);
  }
}

//__________________________________________________________________
std::vector<SVertexer::TrackCand> SVertexer::buildT2V(const gsl::span<const GIndex>& trackIndex,           // Global ID's for associated tracks
                                                      const gsl::span<const VRef>& vtxRefs,                // references from vertex to these track IDs
                                                      const o2::globaltracking::RecoContainer& recoTracks) // accessor to various tracks
{
  // build track->vertices from vertices->tracks, rejecting vertex contributors

  // track selector: at the moment reject prompt tracks contributing to vertex fit and unconstrained TPC tracks
  auto selTrack = [&](GIndex gid) {
    return (gid.isPVContributor() || !recoTracks.isTrackSourceLoaded(gid.getSource())) ? false : true;
  };

  std::vector<TrackCand> tracksPool;
  std::unordered_map<GIndex, int> tmap;
  int nv = vtxRefs.size();
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
          tracksPool[tref->second].vBracket.setMax(iv); // this track was already processed with other vertex, account the latter
          continue;
        }
      }

      tracksPool.emplace_back(TrackCand{recoTracks.getTrack(tvid), tvid, {iv, iv}});
      if (tvid.isAmbiguous()) { // track attached to >1 vertex, remember that it was already processed
        tmap[tvid] = tracksPool.size() - 1;
      }
    }
  }
  LOG(INFO) << "Collected " << tracksPool.size() << " seeds";
  return std::move(tracksPool);
}

//__________________________________________________________________
void SVertexer::finalizeV0s(std::vector<V0>& v0s, std::vector<RRef>& vtx2V0refs)
{
  auto& tmpV0s = mV0sTmp[0];
  int nv0 = tmpV0s.size();
  std::vector<int> v0sortid(nv0);
  std::iota(v0sortid.begin(), v0sortid.end(), 0);
  std::sort(v0sortid.begin(), v0sortid.end(), [&](int i, int j) { return tmpV0s[i].getVertexID() < tmpV0s[j].getVertexID(); });
  int pvID = -1, nForPV = 0;
  for (int iv = 0; iv < nv0; iv++) {
    const auto& v0 = tmpV0s[iv];
    if (pvID < v0.getVertexID()) {
      if (pvID > -1) {
        vtx2V0refs[pvID].setEntries(nForPV);
      }
      pvID = v0.getVertexID();
      vtx2V0refs[pvID].setFirstEntry(v0s.size());
      nForPV = 0;
    }
    v0s.push_back(v0);
    nForPV++;
  }
  if (pvID != -1) { // finalize
    vtx2V0refs[pvID].setEntries(nForPV);
    // fill empty slots
    int ent = v0s.size();
    for (int ip = vtx2V0refs.size(); ip--;) {
      if (vtx2V0refs[ip].getEntries()) {
        ent = vtx2V0refs[ip].getFirstEntry();
      } else {
        vtx2V0refs[ip].setFirstEntry(ent);
      }
    }
  }
}

//__________________________________________________________________
bool SVertexer::checkV0Pair(TrackCand& seedP, TrackCand& seedN, int ithread)
{
  auto& fitter = mFitter2Prong[ithread];
  int nCand = fitter.process(seedP, seedN);
  if (nCand == 0) { // discard this pair
    return false;
  }
  const auto& v0XYZ = fitter.getPCACandidate();
  // check closeness to the beam-line
  auto r2 = (v0XYZ[0] - mMeanVertex.getX()) * (v0XYZ[0] - mMeanVertex.getX()) + (v0XYZ[1] - mMeanVertex.getY()) * (v0XYZ[1] - mMeanVertex.getY());
  if (r2 < mMinR2ToMeanVertex) {
    return false;
  }
  if (!fitter.isPropagateTracksToVertexDone() && !fitter.propagateTracksToVertex()) {
    return false;
  }
  auto& trPProp = fitter.getTrack(0);
  auto& trNProp = fitter.getTrack(1);
  std::array<float, 3> pP, pN;
  trPProp.getPxPyPzGlo(pP);
  trNProp.getPxPyPzGlo(pN);
  // estimate DCA of neutral V0 track to beamline: straight line with parametric equation
  // x = X0 + pV0[0]*t, y = Y0 + pV0[1]*t reaches DCA to beamline (Xv, Yv) at
  // t = -[ (x0-Xv)*pV0[0] + (y0-Yv)*pV0[1]) ] / ( pT(pV0)^2 )
  // Similar equation for 3D distance involving pV0[2]
  std::array<float, 3> pV0 = {pP[0] + pN[0], pP[1] + pN[1], pP[2] + pN[2]};
  float dx = v0XYZ[0] - mMeanVertex.getX(), dy = v0XYZ[1] - mMeanVertex.getY();
  float pt2V0 = pV0[0] * pV0[0] + pV0[1] * pV0[1], prodXY = dx * pV0[0] + dy * pV0[1], tDCAXY = -prodXY / pt2V0;
  float dcaX = dx + pV0[0] * tDCAXY, dcaY = dy + pV0[1] * tDCAXY, dca2 = dcaX * dcaX + dcaY * dcaY;
  if (dca2 > mMaxDCAXY2ToMeanVertex) {
    return false;
  }
  float p2V0 = pt2V0 + pV0[2] * pV0[2], ptV0 = std::sqrt(pt2V0);
  // apply mass selections
  float p2Pos = pP[0] * pP[0] + pP[1] * pP[1] + pP[2] * pP[2], p2Neg = pN[0] * pN[0] + pN[1] * pN[1] + pN[2] * pN[2];
  bool goodHyp = false;
  for (int ipid = 0; ipid < SVertexerParams::NPIDV0; ipid++) {
    if (mV0Hyps[ipid].check(p2Pos, p2Neg, p2V0, ptV0)) {
      goodHyp = true;
      break;
    }
  }
  if (!goodHyp) {
    return false;
  }
  auto vlist = seedP.vBracket.getOverlap(seedN.vBracket); // indices of vertices shared by both seeds
  if (vlist.isInvalid()) {
    LOG(WARNING) << "Incompatible tracks: V0 " << seedP.vBracket.asString() << " | V1 " << seedN.vBracket.asString();
    return false;
  }
  bool added = false;
  auto bestCosPA = mMinCosPointingAngle;
  for (int iv = vlist.getMin(); iv <= vlist.getMax(); iv++) {
    const auto& pv = mPVertices[iv];
    // check cos of pointing angle
    float dz = v0XYZ[2] - pv.getZ(), cosPointingAngle = (prodXY + dz * pV0[2]) / std::sqrt((dx * dx + dy * dy + dz * dz) * p2V0);
    if (cosPointingAngle < bestCosPA) {
      continue;
    }
    if (added) { // V0 was already added, just reassign to this vertex
      auto& v0 = mV0sTmp[ithread].back();
      v0.setCosPA(cosPointingAngle);
      v0.setVertexID(iv);
      bestCosPA = cosPointingAngle;
    } else {
      std::array<float, 3> v0XYZF = {float(v0XYZ[0]), float(v0XYZ[1]), float(v0XYZ[2])};
      auto& v0 = mV0sTmp[ithread].emplace_back(v0XYZF, pV0, trPProp, trNProp, seedP.gid, seedN.gid);
      v0.setCosPA(cosPointingAngle);
      v0.setVertexID(iv);
      v0.setDCA(fitter.getChi2AtPCACandidate());
      bestCosPA = cosPointingAngle;
      added = true;
    }
  }

  return true;
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