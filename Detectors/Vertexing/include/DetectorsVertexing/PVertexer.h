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
#include "DataFormatsCalibration/MeanVertexObject.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "gsl/span"
#include <numeric>
#include <TTree.h>
#include <TFile.h>
#include <TStopwatch.h>

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
  PVertexer();
  void init();
  void end();
  template <typename TR>
  int process(const TR& tracks, const gsl::span<o2d::GlobalTrackID> gids, const gsl::span<InteractionCandidate> intCand,
              std::vector<PVertex>& vertices, std::vector<o2d::VtxTrackIndex>& vertexTrackIDs, std::vector<V2TRef>& v2tRefs,
              const gsl::span<const o2::MCCompLabel> lblTracks, std::vector<o2::MCEventLabel>& lblVtx);
  template <typename TR>
  int process(const TR& tracks, const gsl::span<o2d::GlobalTrackID> gids, const gsl::span<o2::InteractionRecord> intRec,
              std::vector<PVertex>& vertices, std::vector<o2d::VtxTrackIndex>& vertexTrackIDs, std::vector<V2TRef>& v2tRefs,
              const gsl::span<const o2::MCCompLabel> lblTracks, std::vector<o2::MCEventLabel>& lblVtx);

  int processFromExternalPool(const std::vector<TrackVF>& pool, std::vector<PVertex>& vertices, std::vector<o2d::VtxTrackIndex>& vertexTrackIDs, std::vector<V2TRef>& v2tRefs);

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
  void setTrackSources(GTrackID::mask_t s);

  auto& getTracksPool() const { return mTracksPool; }
  auto& getTimeZClusters() const { return mTimeZClusters; }

  auto& getMeanVertex() const { return mMeanVertex; }
  void setMeanVertex(const o2d::MeanVertexObject* v)
  {
    if (!v) {
      return;
    }
    mMeanVertex = *v;
    mMeanVertexSeed = *v;
    initMeanVertexConstraint();
  }

  void setITSROFrameLength(float v)
  {
    mITSROFrameLengthMUS = v;
  }

  // special methods used to refit already found vertex skipping certain number of tracks.
  template <typename TR>
  bool prepareVertexRefit(const TR& tracks, const o2d::VertexBase& vtxSeed);

  PVertex refitVertex(const std::vector<bool> useTrack, const o2d::VertexBase& vtxSeed);

  auto getNTZClusters() const { return mNTZClustersIni; }
  auto getTotTrials() const { return mTotTrials; }
  auto getMaxTrialsPerCluster() const { return mMaxTrialPerCluster; }
  auto getLongestClusterMult() const { return mLongestClusterMult; }
  auto getLongestClusterTimeMS() const { return mLongestClusterTimeMS; }

  TStopwatch& getTimeDBScan() { return mTimeDBScan; }
  TStopwatch& getTimeVertexing() { return mTimeVertexing; }
  TStopwatch& getTimeDebris() { return mTimeDebris; }
  TStopwatch& getTimeMADSel() { return mTimeMADSel; }
  TStopwatch& getTimeReAttach() { return mTimeReAttach; }

  void setPoolDumpDirectory(const std::string& d) { mPoolDumpDirectory = d; }

  void printInpuTracksStatus(const VertexingInput& input) const;

 private:
  static constexpr int DBS_UNDEF = -2, DBS_NOISE = -1, DBS_INCHECK = -10;

  SeedHistoTZ buildHistoTZ(const VertexingInput& input);
  int runVertexing(gsl::span<o2d::GlobalTrackID> gids, const gsl::span<InteractionCandidate> intCand,
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
  bool relateTrackToMeanVertex(o2::track::TrackParCov& trc, float vtxErr2);
  bool relateTrackToVertex(o2::track::TrackParCov& trc, const o2d::VertexBase& vtxSeed) const;
  void applyMADSelection(std::vector<PVertex>& vertices, std::vector<int>& timeSort, const std::vector<V2TRef>& v2tRefs, const std::vector<uint32_t>& trackIDs);
  void applITSOnlyFractionCut(std::vector<PVertex>& vertices, std::vector<int>& timeSort, const std::vector<V2TRef>& v2tRefs, const std::vector<uint32_t>& trackIDs);
  void applInteractionValidation(std::vector<PVertex>& vertices, std::vector<int>& timeSort, const gsl::span<InteractionCandidate> intCand, int minContrib);

  template <typename TR>
  void createTracksPool(const TR& tracks, gsl::span<const o2d::GlobalTrackID> gids);

  int findVertices(const VertexingInput& input, std::vector<PVertex>& vertices, std::vector<uint32_t>& trackIDs, std::vector<V2TRef>& v2tRefs);
  void reAttach(std::vector<PVertex>& vertices, std::vector<int>& timeSort, std::vector<uint32_t>& trackIDs, std::vector<V2TRef>& v2tRefs);

  std::pair<int, int> getBestIR(const PVertex& vtx, const gsl::span<InteractionCandidate> intCand, int& currEntry) const;

  int dbscan_RangeQuery(int idxs, std::vector<int>& cand, std::vector<int>& status);
  void dbscan_clusterize();
  void doDBScanDump(const VertexingInput& input, gsl::span<const o2::MCCompLabel> lblTracks);
  void doVtxDump(std::vector<PVertex>& vertices, std::vector<uint32_t> trackIDsLoc, std::vector<V2TRef>& v2tRefsLoc, gsl::span<const o2::MCCompLabel> lblTracks);
  void doDBGPoolDump(gsl::span<const o2::MCCompLabel> lblTracks);
  void dumpPool();

  o2::BunchFilling mBunchFilling;
  std::array<int16_t, o2::constants::lhc::LHCMaxBunches> mClosestBunchAbove{-1}; // closest filled bunch from above, 1st element -1 to disable usage by default
  std::array<int16_t, o2::constants::lhc::LHCMaxBunches> mClosestBunchBelow{-1}; // closest filled bunch from below, 1st element -1 to disable usage by default
  o2d::MeanVertexObject mMeanVertex{};                                           // calibrated mean vertex object
  o2d::VertexBase mMeanVertexSeed{};                                             // mean vertex at particular Z (accounting for slopes
  std::array<float, 3> mXYConstraintInvErr = {1.0f, 0.f, 1.0f}; ///< nominal vertex constraint inverted errors^2
  //
  std::vector<TrackVF> mTracksPool;         ///< tracks in internal representation used for vertexing, sorted in time
  std::vector<TimeZCluster> mTimeZClusters; ///< set of time clusters
  float mITSROFrameLengthMUS = 0;           ///< ITS readout time span in \mus
  float mBz = 0.;                           ///< mag.field at beam line
  float mDBScanDeltaT = 0.;                 ///< deltaT cut for DBScan check
  float mDBSMaxZ2InvCorePoint = 0;          ///< inverse of max sigZ^2 of the track which can be core point in the DBScan
  bool mValidateWithIR = false;             ///< require vertex validation with InteractionCandidates (if available)

  o2::InteractionRecord mStartIR{0, 0}; ///< IR corresponding to the start of the TF

  // structure for the vertex refit
  o2d::VertexBase mVtxRefitOrig{};   ///< original vertex whose tracks are refitted
  std::vector<int> mRefitTrackIDs{}; ///< dummy IDs for refitted tracks
  //

  ///========== Parameters to be set externally, e.g. from CCDB ====================
  GTrackID::mask_t mTrackSrc{};
  std::vector<int> mSrcVec{};
  const PVertexerParams* mPVParams = nullptr;
  float mTukey2I = 0;                        ///< 1./[Tukey parameter]^2
  static constexpr float kDefTukey = 5.0f;   ///< def.value for tukey constant
  static constexpr float kHugeF = 1.e12;     ///< very large float
  static constexpr float kAlmost0F = 1e-12;  ///< tiny float
  static constexpr double kAlmost0D = 1e-16; ///< tiny double
  size_t mNTZClustersIni = 0;
  size_t mTotTrials = 0;
  size_t mMaxTrialPerCluster = 0;
  float mMaxTDiffDebris = 0;      ///< when reducing debris, don't consider vertices separated by time > this value in \mus
  float mMaxTDiffDebrisExtra = 0; ///< when reducing debris, don't consider vertices separated by time > this value in \mus (optional additional cut
  float mMaxTDiffDebrisFiducial = 0;
  float mMaxZDiffDebrisFiducial = 0;
  float mMaxMultRatDebrisFiducial = 0;
  long mLongestClusterTimeMS = 0;
  int mLongestClusterMult = 0;
  bool mPoolDumpProduced = false;
  bool mITSOnly = false;
  TStopwatch mTimeDBScan;
  TStopwatch mTimeVertexing;
  TStopwatch mTimeDebris;
  TStopwatch mTimeMADSel;
  TStopwatch mTimeReAttach;
  std::string mPoolDumpDirectory{};
#ifdef _PV_DEBUG_TREE_
  std::unique_ptr<TFile> mDebugDumpFile;
  std::unique_ptr<TTree> mDebugDBScanTree;
  std::unique_ptr<TTree> mDebugPoolTree;
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
  vtxSeed.cxy += mXYConstraintInvErr[1];
  vtxSeed.cyy += mXYConstraintInvErr[2];
  float xv = mMeanVertex.getXAtZ(vtxSeed.getZ()), yv = mMeanVertex.getYAtZ(vtxSeed.getZ());
  vtxSeed.cx0 += mXYConstraintInvErr[0] * xv + mXYConstraintInvErr[1] * yv;
  vtxSeed.cy0 += mXYConstraintInvErr[1] * xv + mXYConstraintInvErr[2] * yv;
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
  float vtxErr2 = 0.5 * (mMeanVertex.getSigmaX2() + mMeanVertex.getSigmaY2()) + mPVParams->meanVertexExtraErrSelection * mPVParams->meanVertexExtraErrSelection;
  const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();

  for (uint32_t i = 0; i < ntGlo; i++) {
    int id = sortedTrackID[i];
    o2::track::TrackParCov trc = tracks[id];
    trc.updateCov(mPVParams->sysErrY2, o2::track::kSigY2);
    trc.updateCov(mPVParams->sysErrZ2, o2::track::kSigZ2);
    if (!relateTrackToMeanVertex(trc, vtxErr2)) {
      continue;
    }
    auto& tvf = mTracksPool.emplace_back(trc, tracks[id].getTimeMUS(), id, gids[id], mPVParams->addTimeSigma2, mPVParams->addZSigma2);
    if (tvf.wghHisto < 0) {
      mTracksPool.pop_back(); // discard bad track
      continue;
    }
    if (gids[id].getSource() == GTrackID::ITSTPC) { // if the track was adjusted to ITS ROF boundary, flag it
      float bcf = tvf.timeEst.getTimeStamp() / o2::constants::lhc::LHCBunchSpacingMUS + o2::constants::lhc::LHCMaxBunches;
      int bcWrtROF = int(bcf - alpParams.roFrameBiasInBC) % alpParams.roFrameLengthInBC;
      if (bcWrtROF == 0) {
        float dbc = bcf - (int(bcf / alpParams.roFrameBiasInBC)) * alpParams.roFrameBiasInBC;
        if (std::abs(dbc) < 1e-6) {
          tvf.setITSTPCAdjusted();
          LOGP(debug, "Adjusted t={} -> bcf={} dbc = {}", tvf.timeEst.getTimeStamp(), bcf, dbc);
        }
      }
    }
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
int PVertexer::process(const TR& tracks, const gsl::span<o2d::GlobalTrackID> gids, const gsl::span<InteractionCandidate> intCand,
                       std::vector<PVertex>& vertices, std::vector<o2d::VtxTrackIndex>& vertexTrackIDs, std::vector<V2TRef>& v2tRefs,
                       const gsl::span<const o2::MCCompLabel> lblTracks, std::vector<o2::MCEventLabel>& lblVtx)
{
  createTracksPool(tracks, gids);
  return runVertexing(gids, intCand, vertices, vertexTrackIDs, v2tRefs, lblTracks, lblVtx);
}

//___________________________________________________________________
template <typename TR>
int PVertexer::process(const TR& tracks, const gsl::span<o2d::GlobalTrackID> gids, const gsl::span<o2::InteractionRecord> intRec,
                       std::vector<PVertex>& vertices, std::vector<o2d::VtxTrackIndex>& vertexTrackIDs, std::vector<V2TRef>& v2tRefs,
                       const gsl::span<const o2::MCCompLabel> lblTracks, std::vector<o2::MCEventLabel>& lblVtx)
{
  createTracksPool(tracks, gids);
  static std::vector<InteractionCandidate> intCand;
  intCand.clear();
  intCand.reserve(intRec.size());
  for (const auto& ir : intRec) {
    intCand.emplace_back(InteractionCandidate{ir, float(ir.differenceInBC(mStartIR) * o2::constants::lhc::LHCBunchSpacingMUS), 0, 0});
  }
  return runVertexing(gids, intCand, vertices, vertexTrackIDs, v2tRefs, lblTracks, lblVtx);
}

//___________________________________________________________________
template <typename TR>
bool PVertexer::prepareVertexRefit(const TR& tracks, const o2d::VertexBase& vtxSeed)
{
  // Create pull of all tracks for the refitting starting from the tracks of already found vertex.
  // This method should be called once before calling refitVertex method for given vtxSeed (which can be
  // called multiple time with different useTrack request)
  mTracksPool.clear();
  mTracksPool.reserve(tracks.size());
  for (uint32_t i = 0; i < tracks.size(); i++) {
    o2::track::TrackParCov trc = tracks[i];
    if (!relateTrackToVertex(trc, vtxSeed)) {
      continue;
    }
    auto& tvf = mTracksPool.emplace_back(trc, TimeEst{0.f, 1.f}, i, GTrackID{}, 1., mPVParams->addZSigma2);
    tvf.entry = i;
  }
  if (int(mTracksPool.size()) < mPVParams->minTracksPerVtx) {
    return false;
  }
  mRefitTrackIDs.resize(tracks.size());
  std::iota(mRefitTrackIDs.begin(), mRefitTrackIDs.end(), 0);
  mVtxRefitOrig = vtxSeed;
  return true;
}

} // namespace vertexing
} // namespace o2
#endif
