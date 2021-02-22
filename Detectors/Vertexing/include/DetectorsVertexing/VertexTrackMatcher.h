// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file VertexTrackMatcher.h
/// \brief Class for vertex track association
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_VERTEX_TRACK_MATCHER_
#define ALICEO2_VERTEX_TRACK_MATCHER_

#include "gsl/span"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DetectorsVertexing/PVertexerParams.h"
#include "MathUtils/Primitive2D.h"

namespace o2
{
namespace vertexing
{

class VertexTrackMatcher
{
 public:
  using GIndex = o2::dataformats::VtxTrackIndex;
  using VRef = o2::dataformats::VtxTrackRef;
  using PVertex = const o2::dataformats::PrimaryVertex;
  using TrackTPCITS = o2::dataformats::TrackTPCITS;
  using TrackITS = o2::its::TrackITS;
  using ITSROFR = o2::itsmft::ROFRecord;
  using TrackTPC = o2::tpc::TrackTPC;
  using TmpMap = std::vector<std::vector<GIndex>>;
  using TimeEst = o2::dataformats::TimeStampWithError<float, float>;
  using TBracket = o2::math_utils::Bracketf_t;

  void init();
  void process(const gsl::span<const PVertex>& vertices,   // vertices
               const gsl::span<const GIndex>& v2tfitIDs,   // IDs of contributor tracks used in fit
               const gsl::span<const VRef>& v2tfitRefs,    // references on these tracks (we used special reference with multiple sources, but currently only TPCITS used)
               const gsl::span<const TrackTPCITS>& tpcits, // global tracks
               const gsl::span<const TrackITS>& its,       // ITS tracks
               const gsl::span<const ITSROFR>& itsROFR,    // ITS tracks ROFRecords
               const gsl::span<const TrackTPC>& tpc,       // TPC tracks
               std::vector<GIndex>& trackIndex,            // Global ID's for associated tracks
               std::vector<VRef>& vtxRefs);                // references on these tracks

  ///< set InteractionRecods for the beginning of the TF
  void setStartIR(const o2::InteractionRecord& ir) { mStartIR = ir; }
  void setITSROFrameLengthInBC(int nbc);
  int getITSROFrameLengthInBC() const { return mITSROFrameLengthInBC; }

 private:
  void attachTPCITS(TmpMap& tmpMap, const gsl::span<const TrackTPCITS>& tpcits, const std::vector<int>& idTPCITS, const gsl::span<const PVertex>& vertices);
  void attachITS(TmpMap& tmpMap, const gsl::span<const TrackITS>& its, const gsl::span<const ITSROFR>& itsROFR, const std::vector<int>& flITS,
                 const gsl::span<const PVertex>& vertices, std::vector<int>& idxVtx);
  void attachTPC(TmpMap& tmpMap, const std::vector<TBracket>& tpcTimes, const std::vector<int>& idTPC, const gsl::span<const PVertex>& vertices, std::vector<int>& idVtx);
  bool compatibleTimes(const TimeEst& vtxT, const TimeEst& trcT) const;
  void updateTPCTimeDependentParams();
  float tpcTimeBin2MUS(float t)
  { // convert TPC time bin to microseconds
    return t * mTPCBin2MUS;
  }

  o2::InteractionRecord mStartIR{0, 0}; ///< IR corresponding to the start of the TF
  int mITSROFrameLengthInBC = 0;        ///< ITS RO frame in BC (for ITS cont. mode only)
  float mMaxTPCDriftTimeMUS = 0;
  float mTPCBin2MUS = 0;
  const o2::vertexing::PVertexerParams* mPVParams = nullptr;

  ClassDefNV(VertexTrackMatcher, 1);
};

} // namespace vertexing
} // namespace o2

#endif
