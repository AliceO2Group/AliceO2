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
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/V0.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsITS/TrackITS.h"
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
  using TrackTPCITS = o2::dataformats::TrackTPCITS;
  using TrackITS = o2::its::TrackITS;
  using TrackTPC = o2::tpc::TrackTPC;
  using RRef = o2::dataformats::RangeReference<int, int>;

  // struct to access tracks and extra info from different sources
  struct TrackAccessor {
    static constexpr std::array<size_t, GIndex::NSources> sizes{sizeof(TrackTPCITS), sizeof(TrackITS), sizeof(TrackTPC)};
    std::array<const char*, GIndex::NSources> startOfSource{};
    std::array<std::vector<char>, GIndex::NSources> charges;

    TrackAccessor(const gsl::span<const TrackTPCITS>& tpcits, const gsl::span<const TrackITS>& its, const gsl::span<const TrackTPC>& tpc)
    {
      if (tpcits.size()) {
        startOfSource[GIndex::TPCITS] = reinterpret_cast<const char*>(tpcits.data());
        auto& ch = charges[GIndex::TPCITS];
        ch.resize(tpcits.size());
        for (uint32_t ic = 0; ic < tpcits.size(); ic++) {
          ch[ic] = tpcits[ic].getCharge();
        }
      }
      if (its.size()) {
        startOfSource[GIndex::ITS] = reinterpret_cast<const char*>(its.data());
        auto& ch = charges[GIndex::ITS];
        ch.resize(its.size());
        for (uint32_t ic = 0; ic < its.size(); ic++) {
          ch[ic] = its[ic].getCharge();
        }
      }
      if (tpc.size()) {
        startOfSource[GIndex::TPC] = reinterpret_cast<const char*>(tpc.data());
        auto& ch = charges[GIndex::TPC];
        ch.resize(tpc.size());
        for (uint32_t ic = 0; ic < tpc.size(); ic++) {
          ch[ic] = tpc[ic].getCharge();
        }
      }
    }
    char getCharge(GIndex id) const { return getCharge(id.getSource(), id.getIndex()); }
    char getCharge(int src, int idx) const { return charges[src][idx]; }
    const o2::track::TrackParCov& getTrack(GIndex id) const { return getTrack(id.getSource(), id.getIndex()); }
    const o2::track::TrackParCov& getTrack(int src, int idx) const { return *reinterpret_cast<const o2::track::TrackParCov*>(startOfSource[src] + sizes[src] * idx); }
  };

  void init();
  void process(const gsl::span<const PVertex>& vertices,   // primary vertices
               const gsl::span<const GIndex>& trackIndex,  // Global ID's for associated tracks
               const gsl::span<const VRef>& vtxRefs,       // references from vertex to these track IDs
               const gsl::span<const TrackTPCITS>& tpcits, // global tracks
               const gsl::span<const TrackITS>& its,       // ITS tracks
               const gsl::span<const TrackTPC>& tpc,       // TPC tracks
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
