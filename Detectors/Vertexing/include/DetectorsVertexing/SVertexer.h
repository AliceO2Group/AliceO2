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
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "CommonDataFormat/RangeReference.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "DetectorsVertexing/SVertexerParams.h"
#include "DetectorsVertexing/SVertexHypothesis.h"
#include <numeric>
#include <algorithm>

namespace o2
{
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
  using RRef = o2::dataformats::RangeReference<int, int>;
  using VBracket = o2::math_utils::Bracket<int>;

  enum HypV0 { Photon,
               K0,
               Lambda,
               AntiLambda,
               HyperTriton,
               AntiHyperTriton,
               NHypV0 };

  enum HypCascade {
    XiMinus,
    OmegaMinus,
    NHypCascade
  };

  static constexpr int POS = 0, NEG = 1;
  struct TrackCand : o2::track::TrackParCov {
    GIndex gid;
    VBracket vBracket;
  };

  SVertexer(bool enabCascades = true) : mEnableCascades(enabCascades) {}

  void setEnableCascades(bool v) { mEnableCascades = v; }
  void init();
  void process(const o2::globaltracking::RecoContainer& recoTracks); // accessor to various tracks

  auto& getMeanVertex() const { return mMeanVertex; }
  void setMeanVertex(const o2d::VertexBase& v) { mMeanVertex = v; }
  void setNThreads(int n);
  int getNThreads() const { return mNThreads; }

  template <typename V0CONT, typename V0REFCONT, typename CASCCONT, typename CASCREFCONT>
  void extractSecondaryVertices(V0CONT& v0s, V0REFCONT& vtx2V0Refs, CASCCONT& cascades, CASCREFCONT& vtx2CascRefs);

 private:
  bool checkV0(TrackCand& seed0, TrackCand& seed1, int iP, int iN, int ithread);
  int checkCascades(float r2v0, std::array<float, 3> pV0, float p2v0, int avoidTrackID, int posneg, int ithread);
  void setupThreads();
  void buildT2V(const o2::globaltracking::RecoContainer& recoTracks);
  void updateTimeDependentParams();

  uint64_t getPairIdx(GIndex id1, GIndex id2) const
  {
    return (uint64_t(id1) << 32) | id2;
  }

  gsl::span<const PVertex> mPVertices;
  std::vector<std::vector<V0>> mV0sTmp;
  std::vector<std::vector<Cascade>> mCascadesTmp;
  std::array<std::vector<TrackCand>, 2> mTracksPool{}; // pools of positive and negative seeds sorted in min VtxID
  std::array<std::vector<int>, 2> mVtxFirstTrack{};    // 1st pos. and neg. track of the pools for each vertex
  o2d::VertexBase mMeanVertex{{0., 0., 0.}, {0.1 * 0.1, 0., 0.1 * 0.1, 0., 0., 6. * 6.}};
  const SVertexerParams* mSVParams = nullptr;
  std::array<SVertexHypothesis, NHypV0> mV0Hyps;
  std::array<SVertexHypothesis, NHypCascade> mCascHyps;

  std::vector<DCAFitterN<2>> mFitterV0;
  std::vector<DCAFitterN<2>> mFitterCasc;
  int mNThreads = 1;
  float mMinR2ToMeanVertex = 0;
  float mMaxDCAXY2ToMeanVertex = 0;
  float mMaxDCAXY2ToMeanVertexV0Casc = 0;
  float mMinR2DiffV0Casc = 0;
  float mMaxR2ToMeanVertexCascV0 = 0;

  bool mEnableCascades = true;
};

// input containers can be std::vectors or pmr vectors
template <typename V0CONT, typename V0REFCONT, typename CASCCONT, typename CASCREFCONT>
void SVertexer::extractSecondaryVertices(V0CONT& v0s, V0REFCONT& vtx2V0Refs, CASCCONT& cascades, CASCREFCONT& vtx2CascRefs)
{
  v0s.clear();
  vtx2V0Refs.clear();
  vtx2V0Refs.resize(mPVertices.size());
  cascades.clear();
  vtx2CascRefs.clear();
  vtx2CascRefs.resize(mPVertices.size());

  auto& tmpV0s = mV0sTmp[0];
  auto& tmpCascs = mCascadesTmp[0];
  int nv0 = tmpV0s.size(), nCasc = tmpCascs.size();
  std::vector<int> v0SortID(nv0), v0NewInd(nv0), cascSortID(nCasc);
  std::iota(v0SortID.begin(), v0SortID.end(), 0);
  std::sort(v0SortID.begin(), v0SortID.end(), [&](int i, int j) { return tmpV0s[i].getVertexID() < tmpV0s[j].getVertexID(); });
  std::iota(cascSortID.begin(), cascSortID.end(), 0);
  std::sort(cascSortID.begin(), cascSortID.end(), [&](int i, int j) { return tmpCascs[i].getVertexID() < tmpCascs[j].getVertexID(); });

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
}

} // namespace vertexing
} // namespace o2

#endif
