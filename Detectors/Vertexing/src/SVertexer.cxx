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

using namespace o2::vertexing;

using PID = o2::track::PID;
using TrackTPCITS = o2::dataformats::TrackTPCITS;
using TrackITS = o2::its::TrackITS;
using TrackTPC = o2::tpc::TrackTPC;

void SVertexer::init()
{
  mSVParams = &SVertexerParams::Instance();

  mFitter2Prong.setUseAbsDCA(mSVParams->useAbsDCA);
  mFitter2Prong.setPropagateToPCA(false);
  mFitter2Prong.setMaxR(mSVParams->maxRIni);
  mFitter2Prong.setMinParamChange(mSVParams->minParamChange);
  mFitter2Prong.setMinRelChi2Change(mSVParams->minRelChi2Change);
  mFitter2Prong.setMaxDZIni(mSVParams->maxDZIni);
  mFitter2Prong.setMaxChi2(mSVParams->maxChi2);

  // precalculated selection cuts
  mMinR2ToMeanVertex = mSVParams->minRfromMeanVertex * mSVParams->minRfromMeanVertex;
  mMaxDCAXY2ToMeanVertex = mSVParams->maxDCAXYfromMeanVertex * mSVParams->maxDCAXYfromMeanVertex;
  mMinCosPointingAngle = mSVParams->minCosPointingAngle;
  //
  auto bz = o2::base::Propagator::Instance()->getNominalBz();
  mFitter2Prong.setBz(bz);
  //
  mV0Hyps[SVertexerParams::Photon].set(PID::Photon, PID::Electron, PID::Electron, mSVParams->pidCutsPhoton, bz);
  mV0Hyps[SVertexerParams::K0].set(PID::K0, PID::Pion, PID::Pion, mSVParams->pidCutsK0, bz);
  mV0Hyps[SVertexerParams::Lambda].set(PID::Lambda, PID::Proton, PID::Pion, mSVParams->pidCutsLambda, bz);
  mV0Hyps[SVertexerParams::AntiLambda].set(PID::Lambda, PID::Pion, PID::Proton, mSVParams->pidCutsLambda, bz);
  mV0Hyps[SVertexerParams::HyperTriton].set(PID::HyperTriton, PID::Helium3, PID::Pion, mSVParams->pidCutsHTriton, bz);
  mV0Hyps[SVertexerParams::AntiHyperTriton].set(PID::HyperTriton, PID::Pion, PID::Helium3, mSVParams->pidCutsHTriton, bz);
  //
}

void SVertexer::process(const gsl::span<const PVertex>& vertices,   // primary vertices
                        const gsl::span<const GIndex>& trackIndex,  // Global ID's for associated tracks
                        const gsl::span<const VRef>& vtxRefs,       // references from vertex to these track IDs
                        const o2d::GlobalTrackAccessor& tracksPool, // accessor to various tracks
                        std::vector<V0>& v0s,                       // found V0s
                        std::vector<RRef>& vtx2V0refs               // references from PVertex to V0
)
{
  std::unordered_map<uint64_t, int> cache; // cache for tested combinations, the value >0 will give the entry of prevalidated V0 in the v0sTmp
  std::vector<V0> v0sTmp(1);               // 1st one is dummy!
  std::vector<int> v0sIdx;                 // id's in v0sTmp used attached to p.vertices
  std::vector<RRef> pv2v0sRefs;            // p.vertex to v0 index references
  std::vector<char> selQ(trackIndex.size(), 0);

  auto selTrack = [&tracksPool, &trackIndex](int i) -> char {
    auto id = trackIndex[i];
    return id.isPVContributor() || (id.getSource() != GIndex::ITSTPC && id.getSource() != GIndex::ITS) ? 0 : tracksPool.get(id).getCharge();
  };

  for (size_t i = 0; i < trackIndex.size(); i++) {
    selQ[i] = selTrack(i);
  }

  int nv = vertices.size();
  size_t countTot = 0, countTotUnique = 0;
  for (int iv = 0; iv < nv; iv++) {
    //
    // select vertices
    auto& pvrefs = pv2v0sRefs.emplace_back(v0sIdx.size(), 0);
    const auto& pv = vertices[iv];
    if (pv.getNContributors() < 5) {
      continue;
    }
    //
    const auto& vtref = vtxRefs[iv];
    //
    int first = vtref.getFirstEntry();
    int last = first + vtref.getEntries();
    size_t count = 0, countUnique = 0;
    for (int ipos = first; ipos < last; ipos++) {
      if (selQ[ipos] != 1) {
        continue;
      }
      auto idpos = trackIndex[ipos];
      bool ambiguousPos = idpos.isAmbiguous(); // is this track compatible also with other vertex?
      const auto& trPos = tracksPool.get(idpos);

      for (int ineg = first; ineg < last; ineg++) {
        if (selQ[ineg] != -1) {
          continue;
        }
        auto idneg = trackIndex[ineg];
        count++;
        bool accept = false, newPair = true, ambiguousV0 = ambiguousPos && idneg.isAmbiguous(); // V0 is ambiguous if both tracks are compatible with other vertices
        uint64_t idPosNeg = getPairIdx(idpos, idneg);
        int* resPair = nullptr;
        if (ambiguousV0) { // check if it was already processed
          resPair = &cache[idPosNeg];
          if ((*resPair) < 0) { // was already checked and rejected
            continue;
          } else if ((*resPair) == 0) { // was not checked yet
            countUnique++;
            (*resPair) = -1; // this is rejection flag
          } else {
            newPair = false;
          }
        }
        if (newPair) {
          auto trNeg = tracksPool.get(idneg);
          //          LOG(INFO) << "0: " << idpos << " " << tr0.o2::track::TrackPar::asString();
          //          LOG(INFO) << "1: " << idneg << " " << tr1.o2::track::TrackPar::asString();
          int nCand = mFitter2Prong.process(trPos, trNeg);
          if (nCand == 0) { // discard this pair
            continue;
          }
          const auto& v0pos = mFitter2Prong.getPCACandidate();
          // check closeness to the beam-line
          auto r2 = (v0pos[0] - mMeanVertex.getX()) * (v0pos[0] - mMeanVertex.getX()) + (v0pos[1] - mMeanVertex.getY()) * (v0pos[1] - mMeanVertex.getY());
          if (r2 < mMinR2ToMeanVertex) {
            continue;
          }
          if (!mFitter2Prong.isPropagateTracksToVertexDone() && !mFitter2Prong.propagateTracksToVertex()) {
            continue;
          }
          auto& trPosProp = mFitter2Prong.getTrack(0);
          auto& trNegProp = mFitter2Prong.getTrack(1);
          std::array<float, 3> pPos, pNeg;
          trPosProp.getPxPyPzGlo(pPos);
          trNegProp.getPxPyPzGlo(pNeg);
          // estimate DCA of neutral V0 track to beamline: straight line with parametric equation
          // x = X0 + pV0[0]*t, y = Y0 + pV0[1]*t reaches DCA to beamline (Xv, Yv) at
          // t = -[ (x0-Xv)*pV0[0] + (y0-Yv)*pV0[1]) ] / ( pT(pV0)^2 )
          // Similar equation for 3D distance involving pV0[2]
          std::array<float, 3> pV0 = {pPos[0] + pNeg[0], pPos[1] + pNeg[1], pPos[2] + pNeg[2]};
          float dx = v0pos[0] - mMeanVertex.getX(), dy = v0pos[1] - mMeanVertex.getY();
          float pt2V0 = pV0[0] * pV0[0] + pV0[1] * pV0[1], prodXY = dx * pV0[0] + dy * pV0[1], tDCAXY = -prodXY / pt2V0;
          float dcaX = dx + pV0[0] * tDCAXY, dcaY = dy + pV0[1] * tDCAXY, dca2 = dcaX * dcaX + dcaY * dcaY;
          if (dca2 > mMaxDCAXY2ToMeanVertex) {
            continue;
          }
          float p2V0 = pt2V0 + pV0[2] * pV0[2], ptV0 = std::sqrt(pt2V0);

          // apply mass selections
          float p2Pos = pPos[0] * pPos[0] + pPos[1] * pPos[1] + pPos[2] * pPos[2], p2Neg = pNeg[0] * pNeg[0] + pNeg[1] * pNeg[1] + pNeg[2] * pNeg[2];
          bool goodHyp = false;
          for (int ipid = 0; ipid < SVertexerParams::NPIDV0; ipid++) {
            if (mV0Hyps[ipid].check(p2Pos, p2Neg, p2V0, ptV0)) {
              goodHyp = true;
              break;
            }
          }
          if (!goodHyp) {
            continue;
          }
          // check cos of pointing angle
          float dz = v0pos[2] - pv.getZ(), cosPointingAngle = (prodXY + dz * pV0[2]) / std::sqrt((dx * dx + dy * dy + dz * dz) * p2V0);
          if (cosPointingAngle < mMinCosPointingAngle) {
            if (!ambiguousV0) {
              continue; // no need to check this pair wrt other vertices
            }
            cosPointingAngle = mMinCosPointingAngle - 1e-6;
          } else { // cuts passed, register v0
            accept = true;
          }
          // preliminary checks passed, cache V0 and proceed to specific vertex check
          int szV0tmp = v0sTmp.size();
          if (resPair) {
            *resPair = szV0tmp; // index of V0 to be created
          }
          // LOG(INFO) << "Adding new V0 " << szV0tmp;

          std::array<float, 3> v0posF = {float(v0pos[0]), float(v0pos[1]), float(v0pos[2])};
          auto& v0 = v0sTmp.emplace_back(v0posF, pV0, trPosProp, trNegProp, idpos, idneg);
          if (accept) {
            v0.setCosPA(cosPointingAngle);
            v0.setDCA(mFitter2Prong.getChi2AtPCACandidate());
            v0.setVertexID(iv);
            v0sIdx.push_back(szV0tmp);
            pvrefs.changeEntriesBy(1);
          }
        } else { // check if already created V0 is good for this vertex
          // LOG(INFO) << "Rechecking new V0 " << *resPair;
          auto& v0 = v0sTmp[*resPair];
          std::array<float, 3> posV0, momV0;
          v0.getXYZGlo(posV0);
          v0.getPxPyPzGlo(momV0);
          float cosPointingAngle = posV0[0] * momV0[0] + posV0[1] * momV0[1] + posV0[2] * momV0[2];
          cosPointingAngle /= std::sqrt((posV0[0] * posV0[0] + posV0[1] * posV0[1] + posV0[2] * posV0[2]) * (momV0[0] * momV0[0] + momV0[1] * momV0[1] + momV0[2] * momV0[2]));
          if (cosPointingAngle > v0.getCosPA()) { // reassign
            v0.setCosPA(cosPointingAngle);
            v0.setVertexID(iv);
            v0sIdx.push_back(*resPair);
            pvrefs.changeEntriesBy(1);
          }
        }
      }
    }
    countTot += count;
    countTotUnique += countUnique;
    // LOG(INFO) << "Tried " << count << " combs with " << countUnique << " uniques";
  }
  // finalize V0s
  for (int iv = 0; iv < nv; iv++) {
    const auto& pvtx = vertices[iv];
    const auto& pvrefsTmp = pv2v0sRefs[iv];
    auto& pvrefsFin = vtx2V0refs.emplace_back(v0s.size(), 0);
    int from = pvrefsTmp.getFirstEntry(), to = from + pvrefsTmp.getEntries(), nAdded = 0;
    for (int iv0 = from; iv0 < to; iv0++) {
      const auto& v0tmp = v0sTmp[v0sIdx[iv0]];
      if (v0tmp.getVertexID() != iv) { // this v0 - p.vertex association was reassigned
        continue;
      }
      v0s.push_back(v0tmp);
      nAdded++;
    }
    pvrefsFin.setEntries(nAdded);
    LOG(INFO) << nAdded << " V0s added for vertex " << iv << " with " << pvtx.getNContributors() << " tracks, out of initial " << pvrefsTmp.getEntries();
    pvtx.print();
  }

  LOG(INFO) << "Tried " << countTot << " combs in total " << countTotUnique << " uniques";
}
