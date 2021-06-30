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

/// \file PVertexer.h
/// \brief Primary vertex finder
/// \author ruben.shahoyan@cern.ch

#ifndef O2_PVERTEXER_H
#define O2_PVERTEXER_H

#include <array>
#include <utility>
#include "CommonConstants/LHCConstants.h"
#include "CommonDataFormat/TimeStamp.h"
#include "CommonDataFormat/BunchFilling.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "MathUtils/Utils.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/DCA.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "DetectorsVertexing/PVertexerHelpers.h"
#include "DetectorsVertexing/PVertexerParams.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "gsl/span"
#include <numeric>
#include <TTree.h>
#include <TFile.h>

//TODO: MeanVertex and parameters input from CCDB

//#define _PV_DEBUG_TREE_ // if enabled, produce dbscan and vertex comparison dump

namespace o2
{
namespace vertexing
{

namespace o2d = o2::dataformats;

class PVertexer
{

 public:
  enum class FitStatus : int { Failure,
                               PoolEmpty,
                               NotEnoughTracks,
                               IterateFurther,
                               OK };

  void init();
  void end();

  template <typename TR>
  int process(const TR& tracks, const gsl::span<o2d::GlobalTrackID> gids, const gsl::span<o2::InteractionRecord> bcData,
              std::vector<PVertex>& vertices, std::vector<o2d::VtxTrackIndex>& vertexTrackIDs, std::vector<V2TRef>& v2tRefs,
              const gsl::span<const o2::MCCompLabel> lblTracks, std::vector<o2::MCEventLabel>& lblVtx);

  bool findVertex(const VertexingInput& input, PVertex& vtx);

  void setStartIR(const o2::InteractionRecord& ir) { mStartIR = ir; } ///< set InteractionRecods for the beginning of the TF

  void setTukey(float t)
  {
    mTukey2I = t > 0.f ? 1.f / (t * t) : 1.f / (PVertexerParams::kDefTukey * PVertexerParams::kDefTukey);
  }
  float getTukey() const;

  bool setCompatibleIR(PVertex& vtx);

  void setBunchFilling(const o2::BunchFilling& bf);

  void setBz(float bz) { mBz = bz; }
  void setValidateWithIR(bool v) { mValidateWithIR = v; }
  bool getValidateWithIR() const { return mValidateWithIR; }

  auto& getTracksPool() const { return mTracksPool; }
  auto& getTimeZClusters() const { return mTimeZClusters; }

  auto& getMeanVertex() const { return mMeanVertex; }
  void setMeanVertex(const o2d::VertexBase& v)
  {
    mMeanVertex = v;
    initMeanVertexConstraint();
  }

  void setITSROFrameLength(float v)
  {
    mITSROFrameLengthMUS = v;
  }

 private:
  static constexpr int DBS_UNDEF = -2, DBS_NOISE = -1, DBS_INCHECK = -10;

  SeedHistoTZ buildHistoTZ(const VertexingInput& input);
  int runVertexing(gsl::span<o2d::GlobalTrackID> gids, const gsl::span<o2::InteractionRecord> bcData,
                   std::vector<PVertex>& vertices, std::vector<o2d::VtxTrackIndex>& vertexTrackIDs, std::vector<V2TRef>& v2tRefs,
                   gsl::span<const o2::MCCompLabel> lblTracks, std::vector<o2::MCEventLabel>& lblVtx);
  void createMCLabels(gsl::span<const o2::MCCompLabel> lblTracks, const std::vector<uint32_t>& trackIDs, const std::vector<V2TRef>& v2tRefs, std::vector<o2::MCEventLabel>& lblVtx);
  void reduceDebris(std::vector<PVertex>& vertices, std::vector<int>& timeSort, const std::vector<o2::MCEventLabel>& lblVtx);
  FitStatus fitIteration(const VertexingInput& input, VertexSeed& vtxSeed);
  void finalizeVertex(const VertexingInput& input, const PVertex& vtx, std::vector<PVertex>& vertices, std::vector<V2TRef>& v2tRefs, std::vector<uint32_t>& trackIDs, SeedHistoTZ* histo = nullptr);
  void accountTrack(TrackVF& trc, VertexSeed& vtxSeed) const;
  bool solveVertex(VertexSeed& vtxSeed) const;
  FitStatus evalIterations(VertexSeed& vtxSeed, PVertex& vtx) const;
  TimeEst timeEstimate(const VertexingInput& input) const;
  float findZSeedHistoPeak() const;
  void initMeanVertexConstraint();
  void applyConstraint(VertexSeed& vtxSeed) const;
  bool upscaleSigma(VertexSeed& vtxSeed) const;
  bool relateTrackToMeanVertex(o2::track::TrackParCov& trc, float vtxErr2) const;

  template <typename TR>
  void createTracksPool(const TR& tracks, gsl::span<const o2d::GlobalTrackID> gids);

  int findVertices(const VertexingInput& input, std::vector<PVertex>& vertices, std::vector<uint32_t>& trackIDs, std::vector<V2TRef>& v2tRefs);
  void reAttach(std::vector<PVertex>& vertices, std::vector<int>& timeSort, std::vector<uint32_t>& trackIDs, std::vector<V2TRef>& v2tRefs);

  std::pair<int, int> getBestIR(const PVertex& vtx, const gsl::span<o2::InteractionRecord> bcData, int& currEntry) const;

  int dbscan_RangeQuery(int idxs, std::vector<int>& cand, std::vector<int>& status);
  void dbscan_clusterize();
  void doDBScanDump(const VertexingInput& input, gsl::span<const o2::MCCompLabel> lblTracks);
  void doVtxDump(std::vector<PVertex>& vertices, std::vector<uint32_t> trackIDsLoc, std::vector<V2TRef>& v2tRefsLoc, gsl::span<const o2::MCCompLabel> lblTracks);

  o2::BunchFilling mBunchFilling;
  std::array<int16_t, o2::constants::lhc::LHCMaxBunches> mClosestBunchAbove; // closest filled bunch from above
  std::array<int16_t, o2::constants::lhc::LHCMaxBunches> mClosestBunchBelow; // closest filled bunch from below
  o2d::VertexBase mMeanVertex{{0., 0., 0.}, {0.1 * 0.1, 0., 0.1 * 0.1, 0., 0., 6. * 6.}};
  std::array<float, 3> mXYConstraintInvErr = {1.0f, 0.f, 1.0f}; ///< nominal vertex constraint inverted errors^2
  //
  std::vector<TrackVF> mTracksPool;         ///< tracks in internal representation used for vertexing, sorted in time
  std::vector<TimeZCluster> mTimeZClusters; ///< set of time clusters
  float mITSROFrameLengthMUS = 0;           ///< ITS readout time span in \mus
  float mBz = 0.;                          ///< mag.field at beam line
  bool mValidateWithIR = false;            ///< require vertex validation with InteractionRecords (if available)

  o2::InteractionRecord mStartIR{0, 0}; ///< IR corresponding to the start of the TF

  ///========== Parameters to be set externally, e.g. from CCDB ====================
  const PVertexerParams* mPVParams = nullptr;
  float mTukey2I = 0;                        ///< 1./[Tukey parameter]^2
  static constexpr float kDefTukey = 5.0f;   ///< def.value for tukey constant
  static constexpr float kHugeF = 1.e12;     ///< very large float
  static constexpr float kAlmost0F = 1e-12;  ///< tiny float
  static constexpr double kAlmost0D = 1e-16; ///< tiny double

#ifdef _PV_DEBUG_TREE_
  std::unique_ptr<TFile> mDebugDumpFile;
  std::unique_ptr<TTree> mDebugDBScanTree;
  std::unique_ptr<TTree> mDebugVtxTree;
  std::unique_ptr<TTree> mDebugVtxCompTree;

  std::vector<TrackVFDump> mDebugDumpDBSTrc;
  std::vector<GTrackID> mDebugDumpDBSGID;
  std::vector<o2::MCCompLabel> mDebugDumpDBSTrcMC;

  PVertex mDebugDumpVtx;
  std::vector<TrackVFDump> mDebugDumpVtxTrc;
  std::vector<GTrackID> mDebugDumpVtxGID;
  std::vector<o2::MCCompLabel> mDebugDumpVtxTrcMC;

  std::vector<PVtxCompDump> mDebugDumpPVComp;
  std::vector<o2::MCEventLabel> mDebugDumpPVCompLbl0; // for some reason the added as a class member
  std::vector<o2::MCEventLabel> mDebugDumpPVCompLbl1; // gets stored as simple uint
#endif
};

//___________________________________________________________________
inline void PVertexer::applyConstraint(VertexSeed& vtxSeed) const
{
  // impose meanVertex constraint, i.e. account terms
  // (V_i-Constrain_i)^2/sig2constr_i for i=X,Y in the fit chi2 definition
  vtxSeed.cxx += mXYConstraintInvErr[0];
  vtxSeed.cyy += mXYConstraintInvErr[1];
  vtxSeed.cx0 += mXYConstraintInvErr[0] * mMeanVertex.getX();
  vtxSeed.cy0 += mXYConstraintInvErr[1] * mMeanVertex.getY();
}

//___________________________________________________________________
inline bool PVertexer::upscaleSigma(VertexSeed& vtxSeed) const
{
  // scale upward the scaleSigma2 if needes
  if (vtxSeed.scaleSigma2 < mPVParams->maxScale2) {
    auto s = vtxSeed.scaleSigma2 * mPVParams->upscaleFactor;
    vtxSeed.setScale(s > mPVParams->maxScale2 ? mPVParams->maxScale2 : s, mTukey2I);
    return true;
  }
  return false;
}

//___________________________________________________________________
template <typename TR>
void PVertexer::createTracksPool(const TR& tracks, gsl::span<const o2d::GlobalTrackID> gids)
{
  // create pull of all candidate tracks in a global array ordered in time
  mTracksPool.clear();
  auto ntGlo = tracks.size();
  std::vector<int> sortedTrackID(ntGlo);
  mTracksPool.reserve(ntGlo);
  std::iota(sortedTrackID.begin(), sortedTrackID.end(), 0);
  std::sort(sortedTrackID.begin(), sortedTrackID.end(), [&tracks](int i, int j) {
    return tracks[i].timeEst.getTimeStamp() < tracks[j].timeEst.getTimeStamp();
  });

  // check all containers
  float vtxErr2 = 0.5 * (mMeanVertex.getSigmaX2() + mMeanVertex.getSigmaY2());
  o2d::DCA dca;

  for (uint32_t i = 0; i < ntGlo; i++) {
    int id = sortedTrackID[i];
    o2::track::TrackParCov trc = tracks[id];
    if (!relateTrackToMeanVertex(trc, vtxErr2)) {
      continue;
    }
    auto& tvf = mTracksPool.emplace_back(trc, tracks[id].getTimeMUS(), id, gids[id], mPVParams->addTimeSigma2, mPVParams->addZSigma2);
  }

  if (mTracksPool.empty()) {
    return;
  }
  //
  auto tMin = mTracksPool.front().timeEst.getTimeStamp();
  auto tMax = mTracksPool.back().timeEst.getTimeStamp();
}

//___________________________________________________________________
template <typename TR>
int PVertexer::process(const TR& tracks, const gsl::span<o2d::GlobalTrackID> gids, const gsl::span<o2::InteractionRecord> bcData,
                       std::vector<PVertex>& vertices, std::vector<o2d::VtxTrackIndex>& vertexTrackIDs, std::vector<V2TRef>& v2tRefs,
                       const gsl::span<const o2::MCCompLabel> lblTracks, std::vector<o2::MCEventLabel>& lblVtx)
{
  createTracksPool(tracks, gids);
  return runVertexing(gids, bcData, vertices, vertexTrackIDs, v2tRefs, lblTracks, lblVtx);
}

} // namespace vertexing
} // namespace o2
#endif
