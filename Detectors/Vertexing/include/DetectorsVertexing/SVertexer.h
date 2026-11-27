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
#include "DataFormatsCalibration/MeanVertexObject.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/V0.h"
#include "ReconstructionDataFormats/Cascade.h"
#include "ReconstructionDataFormats/Decay3Body.h"
#include "ReconstructionDataFormats/DecayNBodyIndex.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "CommonDataFormat/RangeReference.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "DCAFitter/DCAFitterN.h"
#include "DetectorsVertexing/SVertexerParams.h"
#include "DetectorsVertexing/SVertexHypothesis.h"
#include "StrangenessTracking/StrangenessTracker.h"
#include "DataFormatsTPC/TrackTPC.h"
#include <numeric>
#include <unordered_map>
#include <algorithm>
#include "GPUO2InterfaceRefit.h"
#include "DataFormatsTPC/PIDResponse.h"

// uncomment this to enable filling rejection records
// #define _FILL_SV_POOL_REJECTIONS_

namespace o2
{
namespace tpc
{
class VDriftCorrFact;
}
namespace gpu
{
class TPCFastTransformPOD;
}

namespace vertexing
{

class SVertexer
{
 public:
  using GIndex = o2::dataformats::VtxTrackIndex;
  using VRef = o2::dataformats::VtxTrackRef;
  using PVertex = const o2::dataformats::PrimaryVertex;
  using V0 = o2::dataformats::V0;
  using V0Index = o2::dataformats::V0Index;
  using Cascade = o2::dataformats::Cascade;
  using CascadeIndex = o2::dataformats::CascadeIndex;
  using Decay3Body = o2::dataformats::Decay3Body;
  using Decay3BodyIndex = o2::dataformats::Decay3BodyIndex;
  using RRef = o2::dataformats::RangeReference<int, int>;
  using VBracket = o2::math_utils::Bracket<int>;
  using PIDResponse = o2::tpc::PIDResponse;

  enum HypV0 { Photon,
               K0,
               Lambda,
               AntiLambda,
               HyperTriton,
               AntiHyperTriton,
               Hyperhydrog4,
               AntiHyperhydrog4,
               NHypV0 };

  // reasons for a track considered in buildT2V not to enter the POS/NEG pools (bit = 0x1 << reason)
  enum PoolRejection : int {
    RejTPCExcluded,   // TPC-only tracks excluded by mExcludeTPCtracks
    RejTPCMaxX,       // TPC track X > mTPCTrackMaxX
    RejTPCTimeCorr,   // correctTPCTrack failed for the unconstrained TPC track clone
    RejTPCPhotonNCls, // photon tune: too few TPC clusters
    RejTPCPhotonDPV,  // photon tune: Z too far from the vertex
    RejTPCPhotonRD2,  // photon tune: circle too far from the beam line
    RejPVContributor, // PV contributor while maxPVContributors < 1
    RejMinDCAToPV,    // |DCA| to mean vertex < minDCAToPV (and not a heavy ionizing particle)
    RejShortITSOnly,  // short ITS-only track (see mITSSAminNclu, mRejectITSonlyOBtrack)
    NPoolRejections
  };

  // key of the mPoolRejections map: (vertexID << 32) | GIndex.getRaw()
  static uint64_t getPoolRejectionKey(GIndex gid, int iv)
  {
    return (uint64_t(iv) << 32) | uint64_t(gid.getRaw());
  }

  // inverse of getPoolRejectionKey: returns the track GIndex and the vertex ID
  static std::pair<GIndex, int> rejectionKey2GidIv(uint64_t k)
  {
    GIndex gid;
    gid.setRaw(static_cast<GIndex::Base_t>(k & 0xffffffff));
    return {gid, int(k >> 32)};
  }

  enum HypCascade {
    XiMinus,
    OmegaMinus,
    NHypCascade
  };

  enum Hyp3body {
    H3L3body,
    AntiH3L3body,
    H4L3body,
    AntiH4L3body,
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
    bool hasTPC = false;
    int8_t nITSclu = -1;
    bool compatibleProton = false; // dE/dx compatibility with proton hypothesis (FIXME: use better, uint8_t compat mask?)
    bool hasITS() const
    {
      return nITSclu > 0;
    }
  };

  SVertexer(bool enabCascades = true, bool enab3body = false) : mEnableCascades{enabCascades}, mEnable3BodyDecays{enab3body}
  {
  }

  void setEnableCascades(bool v) { mEnableCascades = v; }
  void setEnable3BodyDecays(bool v) { mEnable3BodyDecays = v; }
  void init();
  void process(const o2::globaltracking::RecoContainer& recoTracks, o2::framework::ProcessingContext& pc);
  void produceOutput(o2::framework::ProcessingContext& pc);
  int getNV0s() const { return mNV0s; }
  int getNCascades() const { return mNCascades; }
  int getN3Bodies() const { return mN3Bodies; }
  int getNStrangeTracks() const { return mNStrangeTracks; }
  auto& getMeanVertex() const { return mMeanVertex; }
  const auto& getPoolRejections() const { return mPoolRejections; } // empty unless compiled with _FILL_SV_POOL_REJECTIONS_
  void setMeanVertex(const o2::dataformats::MeanVertexObject* v)
  {
    if (v == nullptr) {
      return;
    }
    mMeanVertex = v->getMeanVertex();
  }

  void setNThreads(int n);
  int getNThreads() const { return mNThreads; }
  void setUseMC(bool v) { mUseMC = v; }
  bool getUseMC() const { return mUseMC; }

  void setTPCTBin(int nbc)
  {
    // set TPC time bin in BCs
    mMUS2TPCBin = 1.f / (nbc * o2::constants::lhc::LHCBunchSpacingMUS);
  }
  void setTPCVDrift(const o2::tpc::VDriftCorrFact& v);
  void setTPCCorrMaps(const o2::gpu::TPCFastTransformPOD* maph);
  void setStrangenessTracker(o2::strangeness_tracking::StrangenessTracker* tracker) { mStrTracker = tracker; }
  o2::strangeness_tracking::StrangenessTracker* getStrangenessTracker() { return mStrTracker; }

  std::array<size_t, 3> getNFitterCalls() const;
  void setSources(GIndex::mask_t src) { mSrc = src; }

 private:
  template <class TVI, class TCI, class T3I, class TR>
  void extractPVReferences(const TVI& v0s, TR& vtx2V0Refs, const TCI& cascades, TR& vtx2CascRefs, const T3I& vtxs3, TR& vtx2body3Refs);
  bool checkV0(const TrackCand& seed0, const TrackCand& seed1, int iP, int iN, int ithread);
  int checkCascades(const V0Index& v0Idx, const V0& v0, float rv0, std::array<float, 3> pV0, float p2V0, int avoidTrackID, int posneg, VBracket v0vlist, int ithread);
  int check3bodyDecays(const V0Index& v0Idx, const V0& v0, float rv0, std::array<float, 3> pV0, float p2V0, int avoidTrackID, int posneg, VBracket v0vlist, int ithread);
  void setupThreads();
  void buildT2V(const o2::globaltracking::RecoContainer& recoTracks);
  void updateTimeDependentParams();
  bool acceptTrack(const GIndex gid, const o2::track::TrackParCov& trc, PoolRejection& reason) const;
  void flagPoolRejection(GIndex gid, int iv, PoolRejection reason)
  {
#ifdef _FILL_SV_POOL_REJECTIONS_
    mPoolRejections[getPoolRejectionKey(gid, iv)] |= uint16_t(0x1 << reason);
#endif
  }
  bool processTPCTrack(const o2::tpc::TrackTPC& trTPC, GIndex gid, int vtxid);
  float correctTPCTrack(TrackCand& trc, const o2::tpc::TrackTPC& tTPC, float tmus, float tmusErr) const;

  uint64_t getPairIdx(GIndex id1, GIndex id2) const
  {
    return (uint64_t(id1) << 32) | id2;
  }
  const o2::globaltracking::RecoContainer* mRecoCont = nullptr;
  GIndex::mask_t mSrc{};

  const o2::tpc::ClusterNativeAccess* mTPCClusterIdxStruct = nullptr; ///< struct holding the TPC cluster indices
  gsl::span<const o2::tpc::TrackTPC> mTPCTracksArray;                 ///< input TPC tracks span
  gsl::span<const o2::tpc::TPCClRefElem> mTPCTrackClusIdx;            ///< input TPC track cluster indices span
  gsl::span<const unsigned char> mTPCRefitterShMap;                   ///< externally set TPC clusters sharing map
  gsl::span<const unsigned int> mTPCRefitterOccMap;                   ///< externally set TPC clusters occupancy map
  const o2::gpu::TPCFastTransformPOD* mTPCCorrMaps = nullptr;
  std::unique_ptr<o2::gpu::GPUO2InterfaceRefit> mTPCRefitter; ///< TPC refitter used for TPC tracks refit during the reconstruction
  o2::strangeness_tracking::StrangenessTracker* mStrTracker = nullptr;
  gsl::span<const PVertex> mPVertices;
  std::vector<std::vector<V0>> mV0sTmp;
  std::vector<std::vector<Cascade>> mCascadesTmp;
  std::vector<std::vector<Decay3Body>> m3bodyTmp;
  std::vector<std::vector<V0Index>> mV0sIdxTmp;
  std::vector<std::vector<CascadeIndex>> mCascadesIdxTmp;
  std::vector<std::vector<Decay3BodyIndex>> m3bodyIdxTmp;
  std::array<std::vector<TrackCand>, 2> mTracksPool{}; // pools of positive and negative seeds sorted in min VtxID
  std::array<std::vector<int>, 2> mVtxFirstTrack{};    // 1st pos. and neg. track of the pools for each vertex
  std::unordered_map<uint64_t, uint16_t> mPoolRejections{}; // key: (vertexID << 32) | GIndex.getRaw(); value: bit pattern (0x1<<PoolRejection) of reasons for not entering the pools (filled only if _FILL_SV_POOL_REJECTIONS_ is defined)

  o2::dataformats::VertexBase mMeanVertex{{0., 0., 0.}, {0.1 * 0.1, 0., 0.1 * 0.1, 0., 0., 6. * 6.}};
  const SVertexerParams* mSVParams = nullptr;
  std::array<SVertexHypothesis, NHypV0> mV0Hyps;
  std::array<SVertexHypothesis, NHypCascade> mCascHyps;
  std::array<SVertex3Hypothesis, NHyp3body> m3bodyHyps;
  std::vector<DCAFitterN<2>> mFitterV0;
  std::vector<DCAFitterN<2>> mFitterCasc;
  std::vector<DCAFitterN<3>> mFitter3body;

  PIDResponse mPIDresponse;

  int mNThreads = 1;
  int mNV0s = 0, mNCascades = 0, mN3Bodies = 0, mNStrangeTracks = 0;
  float mBz = 0;
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
  bool mUseMC = false;
};

} // namespace vertexing
} // namespace o2

#endif
