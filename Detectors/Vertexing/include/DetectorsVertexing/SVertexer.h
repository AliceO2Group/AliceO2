// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "ReconstructionDataFormats/GlobalTrackAccessor.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/V0.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "CommonDataFormat/RangeReference.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "DetectorsVertexing/SVertexerParams.h"
#include "DetectorsVertexing/V0Hypothesis.h"

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
  using RRef = o2::dataformats::RangeReference<int, int>;

  void init();
  void process(const gsl::span<const PVertex>& vertices,   // primary vertices
               const gsl::span<const GIndex>& trackIndex,  // Global ID's for associated tracks
               const gsl::span<const VRef>& vtxRefs,       // references from vertex to these track IDs
               const o2d::GlobalTrackAccessor& tracksPool, // accessor to various tracks
               std::vector<V0>& v0s,                       // found V0s
               std::vector<RRef>& vtx2V0refs               // references from PVertex to V0
  );

  auto& getMeanVertex() const { return mMeanVertex; }
  void setMeanVertex(const o2d::VertexBase& v) { mMeanVertex = v; }

 private:
  uint64_t getPairIdx(GIndex id1, GIndex id2) const
  {
    return (uint64_t(id1) << 32) | id2;
  }
  o2d::VertexBase mMeanVertex{{0., 0., 0.}, {0.1 * 0.1, 0., 0.1 * 0.1, 0., 0., 6. * 6.}};
  const SVertexerParams* mSVParams = nullptr;
  std::array<V0Hypothesis, SVertexerParams::NPIDV0> mV0Hyps;
  DCAFitterN<2> mFitter2Prong;

  float mMinR2ToMeanVertex = 0;
  float mMaxDCAXY2ToMeanVertex = 0;
  float mMinCosPointingAngle = 0;
};

} // namespace vertexing
} // namespace o2

#endif
