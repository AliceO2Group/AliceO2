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

/// \file VertexTrackMatcher.h
/// \brief Class for vertex track association
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_VERTEX_TRACK_MATCHER_
#define ALICEO2_VERTEX_TRACK_MATCHER_

#include "gsl/span"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DetectorsVertexing/PVertexerParams.h"
#include "MathUtils/Primitive2D.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"

namespace o2
{
namespace vertexing
{

class VertexTrackMatcher
{
 public:
  using GIndex = o2::dataformats::GlobalTrackID;
  using VTIndex = o2::dataformats::VtxTrackIndex;
  using VRef = o2::dataformats::VtxTrackRef;
  using PVertex = const o2::dataformats::PrimaryVertex;
  using TmpMap = std::vector<std::vector<VTIndex>>;
  using TimeEst = o2::dataformats::TimeStampWithError<float, float>;
  using TBracket = o2::math_utils::Bracketf_t;

  struct TrackTBracket {
    TBracket tBracket{}; ///< bracketing time in \mus
    GIndex origID{};     ///< track origin id
  };
  struct VtxTBracket {
    TBracket tBracket{}; ///< bracketing time in \mus
    int origID = -1;     ///< vertex origin id
  };

  void process(const o2::globaltracking::RecoContainer& recoData,
               std::vector<VTIndex>& trackIndex, // Global ID's for associated tracks
               std::vector<VRef>& vtxRefs);      // references on these tracks

  void setITSROFrameLengthMUS(float v) { mITSROFrameLengthMUS = v; }
  void setMFTROFrameLengthMUS(float v) { mMFTROFrameLengthMUS = v; }
  void setMaxTPCDriftTimeMUS(float v) { mMaxTPCDriftTimeMUS = v; }
  void setTPCBin2MUS(float v) { mTPCBin2MUS = v; }
  void setPrescaleLogs(int n) { mPrescaleLogs = n; }

  float getITSROFrameLengthMUS() const { return mITSROFrameLengthMUS; }
  float getMFTROFrameLengthMUS() const { return mMFTROFrameLengthMUS; }
  float getMaxTPCDriftTimeMUS() const { return mMaxTPCDriftTimeMUS; }
  float getTPCBin2MUS() const { return mTPCBin2MUS; }

 private:
  void extractTracks(const o2::globaltracking::RecoContainer& data, const std::unordered_map<GIndex, bool>& vcont);
  std::vector<TrackTBracket> mTBrackets;
  float mITSROFrameLengthMUS = 0;       ///< ITS RO frame in mus
  float mMFTROFrameLengthMUS = 0;       ///< MFT RO frame in mus
  float mMaxTPCDriftTimeMUS = 0;
  float mTPCBin2MUS = 0;
  int mPrescaleLogs = 0;
};

} // namespace vertexing
} // namespace o2

#endif
