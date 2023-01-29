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

/// \file SVertexer.h
/// \brief Secondary vertex finder
/// \author ruben.shahoyan@cern.ch
#ifndef O2_S_VERTEXER_H
#define O2_S_VERTEXER_H

#include "gsl/span"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/V0.h"
#include "ReconstructionDataFormats/Cascade.h"
#include "ReconstructionDataFormats/DecayNbody.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "CommonDataFormat/RangeReference.h"
#include "DCAFitter/DCAFitterN.h"
#include "DetectorsVertexing/SVertexerParams.h"
#include "DetectorsVertexing/SVertexHypothesis.h"
#include "DataFormatsTPC/TrackTPC.h"
#include <numeric>
#include <algorithm>
#include "GPUO2InterfaceRefit.h"
#include "TPCFastTransform.h"

namespace o2
{
namespace tpc
{
class VDriftCorrFact;
}
namespace gpu
{
class CorrectionMapsHelper;
}

namespace vertexing
{

namespace o2d = o2::dataformats;

class SVertexer
{
 public:
  using GIndex = o2::dataformats::VtxTrackIndex;
  using VRef = o2::dataformats::VtxTrackRef;
  using PVertex = const o2::dataformats::PrimaryVertex;
  using V0 = o2::dataformats::V0;
  using Cascade = o2::dataformats::Cascade;
  using DecayNbody = o2::dataformats::DecayNbody;
  using RRef = o2::dataformats::RangeReference<int, int>;
  using VBracket = o2::math_utils::Bracket<int>;

  enum HypV0 { Photon,
               K0,
               Lambda,
               AntiLambda,
               HyperTriton,
               AntiHyperTriton,
               Hyperhydrog4,
               AntiHyperhydrog4,
               NHypV0 };

  enum HypCascade {
    XiMinus,
    OmegaMinus,
    NHypCascade
  };

  enum Hyp3body {
    H3L3body,
    AntiH3L3body,
    He4L3body,
    AntiHe4L3body,
    He5L3body,
    AntiHe5L3body,
    NHyp3body
  };

  static constexpr int POS = 0, NEG = 1;
  struct TrackCand : o2::track::TrackParCov {
    GIndex gid{};
    VBracket vBracket{};
    float minR = 0; // track lowest point r
  };

  SVertexer(bool enabCascades = true, bool enab3body = false) : mEnableCascades{enabCascades}, mEnable3BodyDecays{enab3body} {}

  void setEnableCascades(bool v) { mEnableCascades = v; }
  void setEnable3BodyDecays(bool v) { mEnable3BodyDecays = v; }
  void init();
  void process(const o2::globaltracking::RecoContainer& recoTracks); // accessor to various tracks
  auto& getMeanVertex() const { return mMeanVertex; }
  void setMeanVertex(const o2d::VertexBase& v) { mMeanVertex = v; }
  void setNThreads(int n);
  int getNThreads() const { return mNThreads; }
  void setTPCTBin(int nbc)
  {
    // set TPC time bin in BCs
    mMUS2TPCBin = 1.f / (nbc * o2::constants::lhc::LHCBunchSpacingMUS);
  }
  void setTPCVDrift(const o2::tpc::VDriftCorrFact& v);
  void setTPCCorrMaps(o2::gpu::CorrectionMapsHelper* maph);

  template <typename V0CONT, typename V0REFCONT, typename CASCCONT, typename CASCREFCONT, typename VTX3BCONT, typename VTX3BREFCONT>
  void extractSecondaryVertices(V0CONT& v0s, V0REFCONT& vtx2V0Refs, CASCCONT& cascades, CASCREFCONT& vtx2CascRefs, VTX3BCONT& vtx3, VTX3BREFCONT& vtx3Refs);
  void initTPCTransform();

 private:
  bool checkV0(const TrackCand& seed0, const TrackCand& seed1, int iP, int iN, int ithread);
  int checkCascades(float rv0, std::array<float, 3> pV0, float p2V0, int avoidTrackID, int posneg, VBracket v0vlist, int ithread);
  int check3bodyDecays(float rv0, std::array<float, 3> pV0, float p2V0, int avoidTrackID, int posneg, VBracket v0vlist, int ithread);
  void setupThreads();
  void buildT2V(const o2::globaltracking::RecoContainer& recoTracks);
  void updateTimeDependentParams();
  bool acceptTrack(GIndex gid, const o2::track::TrackParCov& trc) const;
  bool processTPCTrack(const o2::tpc::TrackTPC& trTPC, GIndex gid, int vtxid);
  float correctTPCTrack(o2::track::TrackParCov& trc, const o2::tpc::TrackTPC tTPC, float tmus, float tmusErr) const;

  uint64_t getPairIdx(GIndex id1, GIndex id2) const
  {
    return (uint64_t(id1) << 32) | id2;
  }

  // at the moment not used
  o2::gpu::CorrectionMapsHelper* mTPCCorrMapsHelper = nullptr;
  std::unique_ptr<o2::gpu::GPUO2InterfaceRefit> mTPCRefitter; ///< TPC refitter used for TPC tracks refit during the reconstruction

  gsl::span<const PVertex> mPVertices;
  std::vector<std::vector<V0>> mV0sTmp;
  std::vector<std::vector<Cascade>> mCascadesTmp;
  std::vector<std::vector<DecayNbody>> m3bodyTmp;
  std::array<std::vector<TrackCand>, 2> mTracksPool{}; // pools of positive and negative seeds sorted in min VtxID
  std::array<std::vector<int>, 2> mVtxFirstTrack{};    // 1st pos. and neg. track of the pools for each vertex

  o2d::VertexBase mMeanVertex{{0., 0., 0.}, {0.1 * 0.1, 0., 0.1 * 0.1, 0., 0., 6. * 6.}};
  const SVertexerParams* mSVParams = nullptr;
  std::array<SVertexHypothesis, NHypV0> mV0Hyps;
  std::array<SVertexHypothesis, NHypCascade> mCascHyps;
  std::array<SVertex3Hypothesis, NHyp3body> m3bodyHyps;

  std::vector<DCAFitterN<2>> mFitterV0;
  std::vector<DCAFitterN<2>> mFitterCasc;
  std::vector<DCAFitterN<3>> mFitter3body;
  int mNThreads = 1;
  float mMinR2ToMeanVertex = 0;
  float mMaxDCAXY2ToMeanVertex = 0;
  float mMaxDCAXY2ToMeanVertexV0Casc = 0;
  float mMaxDCAXY2ToMeanVertex3bodyV0 = 0;
  float mMinR2DiffV0Casc = 0;
  float mMaxR2ToMeanVertexCascV0 = 0;
  float mMinPt2V0 = 1e-6;
  float mMaxTgl2V0 = 2. * 2.;
  float mMinPt2Casc = 1e-4;
  float mMaxTgl2Casc = 2. * 2.;
  float mMinPt23Body = 1e-4;
  float mMaxTgl23Body = 2.f * 2.f;
  float mMUS2TPCBin = 1.f / (8 * o2::constants::lhc::LHCBunchSpacingMUS);
  float mTPCBin2Z = 0;
  float mTPCVDrift = 0;
  float mTPCVDriftCorrFact = 1.; ///< TPC nominal correction factort (wrt ref)
  float mTPCVDriftRef = 0;
  float mTPCDriftTimeOffset = 0; ///< drift time offset in mus

  bool mEnableCascades = true;
  bool mEnable3BodyDecays = false;
};

// input containers can be std::vectors or pmr vectors
template <typename V0CONT, typename V0REFCONT, typename CASCCONT, typename CASCREFCONT, typename VTX3BCONT, typename VTX3BREFCONT>
void SVertexer::extractSecondaryVertices(V0CONT& v0s, V0REFCONT& vtx2V0Refs, CASCCONT& cascades, CASCREFCONT& vtx2CascRefs, VTX3BCONT& vtx3, VTX3BREFCONT& vtx3Refs)
{
  v0s.clear();
  vtx2V0Refs.clear();
  vtx2V0Refs.resize(mPVertices.size());
  cascades.clear();
  vtx2CascRefs.clear();
  vtx2CascRefs.resize(mPVertices.size());
  vtx3.clear();
  vtx3Refs.clear();
  vtx3Refs.resize(mPVertices.size());

  auto& tmpV0s = mV0sTmp[0];
  auto& tmpCascs = mCascadesTmp[0];
  auto& tmp3B = m3bodyTmp[0];
  int nv0 = tmpV0s.size(), nCasc = tmpCascs.size(), n3body = tmp3B.size();
  std::vector<int> v0SortID(nv0), v0NewInd(nv0), cascSortID(nCasc), vtx3SortID(n3body);
  std::iota(v0SortID.begin(), v0SortID.end(), 0);
  std::sort(v0SortID.begin(), v0SortID.end(), [&](int i, int j) { return tmpV0s[i].getVertexID() < tmpV0s[j].getVertexID(); });
  std::iota(cascSortID.begin(), cascSortID.end(), 0);
  std::sort(cascSortID.begin(), cascSortID.end(), [&](int i, int j) { return tmpCascs[i].getVertexID() < tmpCascs[j].getVertexID(); });
  std::iota(vtx3SortID.begin(), vtx3SortID.end(), 0);
  std::sort(vtx3SortID.begin(), vtx3SortID.end(), [&](int i, int j) { return tmp3B[i].getVertexID() < tmp3B[j].getVertexID(); });
  // relate V0s to primary vertices
  int pvID = -1, nForPV = 0;
  for (int iv = 0; iv < nv0; iv++) {
    const auto& v0 = tmpV0s[v0SortID[iv]];
    if (pvID < v0.getVertexID()) {
      if (pvID > -1) {
        vtx2V0Refs[pvID].setEntries(nForPV);
      }
      pvID = v0.getVertexID();
      vtx2V0Refs[pvID].setFirstEntry(v0s.size());
      nForPV = 0;
    }
    v0NewInd[v0SortID[iv]] = v0s.size(); // memorise updated v0 id to fix its reference in the cascade
    v0s.push_back(v0);
    nForPV++;
  }
  if (pvID != -1) { // finalize
    vtx2V0Refs[pvID].setEntries(nForPV);
    // fill empty slots
    int ent = v0s.size();
    for (int ip = vtx2V0Refs.size(); ip--;) {
      if (vtx2V0Refs[ip].getEntries()) {
        ent = vtx2V0Refs[ip].getFirstEntry();
      } else {
        vtx2V0Refs[ip].setFirstEntry(ent);
      }
    }
  }
  // update V0s references in cascades
  for (auto& casc : tmpCascs) {
    casc.setV0ID(v0NewInd[casc.getV0ID()]);
  }

  // relate Cascades to primary vertices
  pvID = -1;
  nForPV = 0;
  for (int iv = 0; iv < nCasc; iv++) {
    const auto& casc = tmpCascs[cascSortID[iv]];
    if (pvID < casc.getVertexID()) {
      if (pvID > -1) {
        vtx2CascRefs[pvID].setEntries(nForPV);
      }
      pvID = casc.getVertexID();
      vtx2CascRefs[pvID].setFirstEntry(cascades.size());
      nForPV = 0;
    }
    cascades.push_back(casc);
    nForPV++;
  }
  if (pvID != -1) { // finalize
    vtx2CascRefs[pvID].setEntries(nForPV);
    // fill empty slots
    int ent = cascades.size();
    for (int ip = vtx2CascRefs.size(); ip--;) {
      if (vtx2CascRefs[ip].getEntries()) {
        ent = vtx2CascRefs[ip].getFirstEntry();
      } else {
        vtx2CascRefs[ip].setFirstEntry(ent);
      }
    }
  }

  // relate 3 body decays to primary vertices
  pvID = -1;
  nForPV = 0;
  for (int iv = 0; iv < n3body; iv++) {
    const auto& vertex3body = tmp3B[vtx3SortID[iv]];
    if (pvID < vertex3body.getVertexID()) {
      if (pvID > -1) {
        vtx3Refs[pvID].setEntries(nForPV);
      }
      pvID = vertex3body.getVertexID();
      vtx3Refs[pvID].setFirstEntry(vtx3.size());
      nForPV = 0;
    }
    vtx3.push_back(vertex3body);
    nForPV++;
  }
  if (pvID != -1) { // finalize
    vtx3Refs[pvID].setEntries(nForPV);
    // fill empty slots
    int ent = vtx3.size();
    for (int ip = vtx3Refs.size(); ip--;) {
      if (vtx3Refs[ip].getEntries()) {
        ent = vtx3Refs[ip].getFirstEntry();
      } else {
        vtx3Refs[ip].setFirstEntry(ent);
      }
    }
  }
}

} // namespace vertexing
} // namespace o2

#endif
