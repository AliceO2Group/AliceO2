// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

  template <typename TR, typename BC>
  int process(const TR& tracks, const gsl::span<o2d::GlobalTrackID> gids, const BC& bcData,
              std::vector<PVertex>& vertices, std::vector<o2d::VtxTrackIndex>& vertexTrackIDs, std::vector<V2TRef>& v2tRefs,
              gsl::span<const o2::MCCompLabel> lblITS, gsl::span<const o2::MCCompLabel> lblTPC, std::vector<o2::MCEventLabel>& lblVtx)
  {
    auto nv = process(tracks, gids, bcData, vertices, vertexTrackIDs, v2tRefs);
    if (lblITS.size() && lblTPC.size()) {
      createMCLabels(lblITS, lblTPC, vertices, vertexTrackIDs, v2tRefs, lblVtx);
    }
    return nv;
  }

  template <typename TR, typename BC>
  int process(const TR& tracks, const gsl::span<o2d::GlobalTrackID> gids, const BC& bcData,
              std::vector<PVertex>& vertices, std::vector<o2d::VtxTrackIndex>& vertexTrackIDs, std::vector<V2TRef>& v2tRefs);

  static void createMCLabels(gsl::span<const o2::MCCompLabel> lblITS, gsl::span<const o2::MCCompLabel> lblTPC,
                             const std::vector<PVertex> vertices, const std::vector<o2d::VtxTrackIndex> vertexTrackIDs, const std::vector<V2TRef> v2tRefs,
                             std::vector<o2::MCEventLabel>& lblVtx);
  bool findVertex(const VertexingInput& input, PVertex& vtx);

  void setStartIR(const o2::InteractionRecord& ir) { mStartIR = ir; } ///< set InteractionRecods for the beginning of the TF

  void setTukey(float t)
  {
    mTukey2I = t > 0.f ? 1.f / (t * t) : 1.f / (PVertexerParams::kDefTukey * PVertexerParams::kDefTukey);
  }
  float getTukey() const;

  void finalizeVertex(const VertexingInput& input, const PVertex& vtx, std::vector<PVertex>& vertices, std::vector<V2TRef>& v2tRefs, std::vector<GTrackID>& vertexTrackIDs, SeedHisto& histo);
  bool setCompatibleIR(PVertex& vtx);

  void setBunchFilling(const o2::BunchFilling& bf);

  void setBz(float bz) { mBz = bz; }
  void setValidateWithIR(bool v) { mValidateWithIR = v; }
  bool getValidateWithIR() const { return mValidateWithIR; }

  auto& getTracksPool() const { return mTracksPool; }
  auto& getTimeZClusters() const { return mTimeZClusters; }
  auto& getSortedTrackIndices() const { return mSortedTrackID; }

  auto& getMeanVertex() const { return mMeanVertex; }
  void setMeanVertex(const o2d::VertexBase& v)
  {
    mMeanVertex = v;
    initMeanVertexConstraint();
  }

  float estimateScale2()
  {
    auto sc = mPVParams->zHistoBinSize * mPVParams->zHistoBinSize * mTukey2I / (mStatZErr.getMean() * mStatZErr.getMean());
    return sc;
  }

 private:
  FitStatus fitIteration(const VertexingInput& input, VertexSeed& vtxSeed);
  void accountTrack(TrackVF& trc, VertexSeed& vtxSeed) const;
  bool solveVertex(VertexSeed& vtxSeed) const;
  FitStatus evalIterations(VertexSeed& vtxSeed, PVertex& vtx) const;
  TimeEst timeEstimate(const VertexingInput& input) const;
  float findZSeedHistoPeak() const;
  void initMeanVertexConstraint();
  void applyConstraint(VertexSeed& vtxSeed) const;
  bool upscaleSigma(VertexSeed& vtxSeed) const;

  template <typename TR>
  void createTracksPool(const TR& tracks, gsl::span<const o2d::GlobalTrackID> gids);

  int findVertices(const VertexingInput& input, std::vector<PVertex>& vertices, std::vector<GTrackID>& vertexTrackIDs, std::vector<V2TRef>& v2tRefs);

  template <typename BC>
  std::pair<int, int> getBestIR(const PVertex& vtx, const BC& bcData, int& currEntry) const;

  int dbscan_RangeQuery(int idxs, std::vector<int>& cand, const std::vector<int>& status);
  void dbscan_clusterize();

  o2::BunchFilling mBunchFilling;
  std::array<int16_t, o2::constants::lhc::LHCMaxBunches> mClosestBunchAbove; // closest filled bunch from above
  std::array<int16_t, o2::constants::lhc::LHCMaxBunches> mClosestBunchBelow; // closest filled bunch from below
  o2d::VertexBase mMeanVertex{{0., 0., 0.}, {0.1 * 0.1, 0., 0.1 * 0.1, 0., 0., 6. * 6.}};
  std::array<float, 3> mXYConstraintInvErr = {1.0f, 0.f, 1.0f}; ///< nominal vertex constraint inverted errors^2
  //
  o2::math_utils::StatAccumulator mStatZErr;
  o2::math_utils::StatAccumulator mStatTErr;
  std::vector<TrackVF> mTracksPool;        ///< tracks in internal representation used for vertexing
  std::vector<int> mSortedTrackID;         ///< indices of tracks sorted in time
  std::vector<TimeZCluster> mTimeZClusters; ///< set of time clusters
  std::vector<int> mClusterTrackIDs;        ///< IDs of tracks making the clusters

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

  ClassDefNV(PVertexer, 1);
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
  mSortedTrackID.clear();

  auto ntGlo = tracks.size();
  mTracksPool.reserve(ntGlo);
  // check all containers
  float vtxErr2 = 0.5 * (mMeanVertex.getSigmaX2() + mMeanVertex.getSigmaY2());
  float pullIniCut = 9.; // RS FIXME  pullIniCut should be a parameter
  o2d::DCA dca;

  for (uint32_t i = 0; i < ntGlo; i++) {
    o2::track::TrackParCov trc = tracks[i];
    if (!trc.propagateToDCA(mMeanVertex, mBz, &dca, mPVParams->dcaTolerance) ||
        dca.getY() * dca.getY() / (dca.getSigmaY2() + vtxErr2) > mPVParams->pullIniCut) {
      continue;
    }
    auto& tvf = mTracksPool.emplace_back(trc, tracks[i].getTimeMUS(), gids[i]);
    mStatZErr.add(std::sqrt(trc.getSigmaZ2()));
    mStatTErr.add(tvf.timeEst.getTimeStampError());
  }
  // TODO: try to narrow timestamps using tof times
  auto [zerrMean, zerrRMS] = mStatZErr.getMeanRMS2<float>();

  auto [terrMean, terrRMS] = mStatTErr.getMeanRMS2<float>();

  if (mTracksPool.empty()) {
    return;
  }
  //
  mSortedTrackID.resize(mTracksPool.size());
  std::iota(mSortedTrackID.begin(), mSortedTrackID.end(), 0);

  std::sort(mSortedTrackID.begin(), mSortedTrackID.end(), [this](int i, int j) {
    return this->mTracksPool[i].timeEst.getTimeStamp() < this->mTracksPool[j].timeEst.getTimeStamp();
  });

  auto tMin = mTracksPool[mSortedTrackID.front()].timeEst.getTimeStamp();
  auto tMax = mTracksPool[mSortedTrackID.back()].timeEst.getTimeStamp();
}

//___________________________________________________________________
template <typename TR, typename BC>
int PVertexer::process(const TR& tracks, gsl::span<o2d::GlobalTrackID> gids, const BC& bcData,
                       std::vector<PVertex>& vertices, std::vector<o2d::VtxTrackIndex>& vertexTrackIDs, std::vector<V2TRef>& v2tRefs)
{
  createTracksPool(tracks, gids);
  dbscan_clusterize();

  std::vector<PVertex> verticesLoc;
  std::vector<GTrackID> vertexTrackIDsLoc;
  std::vector<V2TRef> v2tRefsLoc;
  std::vector<float> validationTimes;

  for (auto tc : mTimeZClusters) {
    VertexingInput inp;
    //    inp.idRange = gsl::span<int>((int*)&mSortedTrackID[tc.first], tc.count);
    inp.idRange = gsl::span<int>((int*)&mClusterTrackIDs[tc.first], tc.count);
    inp.scaleSigma2 = 3. * estimateScale2();
    inp.timeEst = tc.timeEst;
    findVertices(inp, verticesLoc, vertexTrackIDsLoc, v2tRefsLoc);
  }

  // sort in time
  std::vector<int> vtTimeSortID(verticesLoc.size());
  std::iota(vtTimeSortID.begin(), vtTimeSortID.end(), 0);
  std::sort(vtTimeSortID.begin(), vtTimeSortID.end(), [&verticesLoc](int i, int j) {
    return verticesLoc[i].getTimeStamp().getTimeStamp() < verticesLoc[j].getTimeStamp().getTimeStamp();
  });

  vertices.clear();
  v2tRefs.clear();
  vertexTrackIDs.clear();
  vertices.reserve(verticesLoc.size());
  v2tRefs.reserve(v2tRefsLoc.size());
  vertexTrackIDs.reserve(vertexTrackIDsLoc.size());

  int trCopied = 0, count = 0, vtimeID = 0;
  for (auto i : vtTimeSortID) {
    auto& vtx = verticesLoc[i];

    bool irSet = setCompatibleIR(vtx);
    if (!irSet) {
      continue;
    }
    // do we need to validate by Int. records ?
    if (mValidateWithIR) {
      auto bestMatch = getBestIR(vtx, bcData, vtimeID);
      if (bestMatch.first >= 0) {
        vtx.setFlags(PVertex::TimeValidated);
        if (bestMatch.second == 1) {
          vtx.setIR(bcData[bestMatch.first].getInteractionRecord());
        }
        LOG(DEBUG) << "Validated with t0 " << bestMatch.first << " with " << bestMatch.second << " candidates";
      } else if (vtx.getNContributors() >= mPVParams->minNContributorsForIRcut) {
        LOG(DEBUG) << "Discarding " << vtx;
        continue; // reject
      }
    }
    vertices.push_back(vtx);
    int it = v2tRefsLoc[i].getFirstEntry(), itEnd = it + v2tRefsLoc[i].getEntries(), dest0 = vertexTrackIDs.size();
    for (; it < itEnd; it++) {
      auto& gid = vertexTrackIDs.emplace_back(vertexTrackIDsLoc[it]);
      gid.setPVContributor();
    }
    v2tRefs.emplace_back(dest0, v2tRefsLoc[i].getEntries());
    LOG(DEBUG) << "#" << count++ << " " << vertices.back() << " | " << v2tRefs.back().getEntries() << " indices from " << v2tRefs.back().getFirstEntry(); // RS REM
  }

  return vertices.size();
}

//___________________________________________________________________
template <typename BC>
std::pair<int, int> PVertexer::getBestIR(const PVertex& vtx, const BC& bcData, int& currEntry) const
{
  // select best matching interaction record
  int best = -1, n = bcData.size();
  while (currEntry < n && bcData[currEntry].getInteractionRecord() < vtx.getIRMin()) {
    currEntry++; // skip all times which have no chance to be matched
  }
  int i = currEntry, nCompatible = 0;
  float bestDf = 1e12;
  auto tVtxNS = (vtx.getTimeStamp().getTimeStamp() + mPVParams->timeBiasMS) * 1e3; // time in ns
  while (i < n) {
    if (bcData[i].getInteractionRecord() > vtx.getIRMax()) {
      break;
    }
    nCompatible++;
    auto dfa = std::abs(bcData[i].getInteractionRecord().differenceInBCNS(mStartIR) - tVtxNS);
    if (dfa <= bestDf) {
      bestDf = dfa;
      best = i;
    }
    i++;
  }
  return {best, nCompatible};
}

} // namespace vertexing
} // namespace o2
#endif
