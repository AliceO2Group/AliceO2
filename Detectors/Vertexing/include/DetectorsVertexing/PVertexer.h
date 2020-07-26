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
#include "CommonDataFormat/TimeStamp.h"
#include "CommonDataFormat/RangeReference.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "MathUtils/Utils.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/Vertex.h"
#include "DetectorsVertexing/PVertexerHelpers.h"
#include "DetectorsVertexing/PVertexerParams.h"
#include "gsl/span"

namespace o2
{
namespace vertexing
{

namespace o2d = o2::dataformats;

class VertexSeed;
class TrackVF;
class VertexingInput;
class SeedHisto;
class TimeCluster;

class PVertexer
{

 public:
  enum class FitStatus : int { Failure,
                               PoolEmpty,
                               NotEnoughTracks,
                               IterateFurther,
                               OK };

  using TimeEst = o2::dataformats::TimeStampWithError<float, float>;
  using Vertex = o2::dataformats::Vertex<TimeEst>;
  using V2TRef = o2::dataformats::RangeReference<int, int>;

  void init();
  int process(gsl::span<const o2d::TrackTPCITS> tracksITSTPC, std::vector<Vertex>& vertices, std::vector<int>& vertexTrackIDs, std::vector<V2TRef>& v2tRefs,
              gsl::span<const o2::MCCompLabel> lblITS, gsl::span<const o2::MCCompLabel> lblTPC, std::vector<o2::MCEventLabel>& lblVtx);
  int process(gsl::span<const o2d::TrackTPCITS> tracksITSTPC, std::vector<Vertex>& vertices, std::vector<int>& vertexTrackIDs, std::vector<V2TRef>& v2tRefs);
  static void createMCLabels(gsl::span<const o2::MCCompLabel> lblITS, gsl::span<const o2::MCCompLabel> lblTPC,
                             const std::vector<Vertex> vertices, const std::vector<int> vertexTrackIDs, const std::vector<V2TRef> v2tRefs,
                             std::vector<o2::MCEventLabel>& lblVtx);
  bool findVertex(const VertexingInput& input, Vertex& vtx);

  void setTukey(float t)
  {
    mTukey2I = t > 0.f ? 1.f / (t * t) : 1.f / (PVertexerParams::kDefTukey * PVertexerParams::kDefTukey);
  }
  float getTukey() const;

  void finalizeVertex(const VertexingInput& input, const Vertex& vtx, std::vector<Vertex>& vertices, std::vector<V2TRef>& v2tRefs, std::vector<int>& vertexTrackIDs, SeedHisto& histo);

  void setBz(float bz) { mBz = bz; }

  auto& getTracksPool() const { return mTracksPool; }
  auto& getTimeClusters() const { return mTimesClusters; }
  auto& getSortedTrackIndices() const { return mSortedTrackID; }

  auto& getMeanVertex() const { return mMeanVertex; }
  void setMeanVertex(const o2d::VertexBase& v)
  {
    mMeanVertex = v;
    initMeanVertexConstraint();
  }

  float estimateScale2()
  {
    float minrange = std::min(mParams->zHistoBinSize, mParams->minZSeedRange);
    auto sc = mParams->zHistoBinSize * mParams->zHistoBinSize * mTukey2I / (mStatZErr.getMean() * mStatZErr.getMean());
    return sc;
  }

 private:
  FitStatus fitIteration(const VertexingInput& input, VertexSeed& vtxSeed);
  void accountTrack(TrackVF& trc, VertexSeed& vtxSeed) const;
  bool solveVertex(VertexSeed& vtxSeed) const;
  FitStatus evalIterations(VertexSeed& vtxSeed, Vertex& vtx) const;
  TimeEst timeEstimate(const VertexingInput& input) const;
  float findZSeedHistoPeak() const;
  void initMeanVertexConstraint();
  void applyConstraint(VertexSeed& vtxSeed) const;
  bool upscaleSigma(VertexSeed& vtxSeed) const;
  void createTracksPool(gsl::span<const o2d::TrackTPCITS> tracksITSTPC);
  void clusterizeTimeBruteForce(float margin = 0.1, float cut = 25);
  void clusterizeTime(float binSize = 0.1, float maxTDist = 0.6);
  int findVertices(const VertexingInput& input, std::vector<Vertex>& vertices, std::vector<int>& vertexTrackIDs, std::vector<V2TRef>& v2tRefs);

  o2d::VertexBase mMeanVertex{{0., 0., 0.}, {0.1 * 0.1, 0., 0.1 * 0.1, 0., 0., 6. * 6.}};
  std::array<float, 3> mXYConstraintInvErr = {1.0f, 0.f, 1.0f}; ///< nominal vertex constraint inverted errors^2
  //
  o2::utils::StatAccumulator mStatZErr;
  o2::utils::StatAccumulator mStatTErr;
  std::vector<TrackVF> mTracksPool;        ///< tracks in internal representation used for vertexing
  std::vector<int> mSortedTrackID;         ///< indices of tracks sorted in time
  std::vector<TimeCluster> mTimesClusters; ///< set of time clusters
  float mBz = 0.;                          ///< mag.field at beam line

  ///========== Parameters to be set externally, e.g. from CCDB ====================
  const PVertexerParams* mParams = nullptr;
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
  if (vtxSeed.scaleSigma2 < mParams->maxScale2) {
    auto s = vtxSeed.scaleSigma2 * mParams->upscaleFactor;
    vtxSeed.setScale(s > mParams->maxScale2 ? mParams->maxScale2 : s, mTukey2I);
    return true;
  }
  return false;
}

} // namespace vertexing
} // namespace o2
#endif
