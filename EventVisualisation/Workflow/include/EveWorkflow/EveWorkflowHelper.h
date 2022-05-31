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

///
/// \file   EveWorkflowHelper.h
/// \author julian.myrcha@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_WORKFLOW_EVEWORKFLOWHELPER_H
#define ALICE_O2_EVENTVISUALISATION_WORKFLOW_EVEWORKFLOWHELPER_H

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "Framework/DataProcessingHeader.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "EveWorkflow/EveConfiguration.h"
#include "EventVisualisationDataConverter/VisualisationEvent.h"
#include "MFTBase/GeometryTGeo.h"
#include "ITSBase/GeometryTGeo.h"
#include "TPCFastTransform.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "Framework/AnalysisDataModel.h"
#include "DetectorsVertexing/PVertexerParams.h"

namespace o2::itsmft
{
class TopologyDictionary;
}

namespace o2::mft
{
class GeometryTGeo;
}

namespace o2::its
{
class GeometryTGeo;
}

namespace o2::phos
{
class Geometry;
}

namespace o2::event_visualisation
{
using GID = o2::dataformats::GlobalTrackID;
using PNT = std::array<float, 3>;

struct TracksSet {
  std::vector<GID> trackGID;
  std::vector<float> trackTime;
};

class EveWorkflowHelper
{
  static constexpr std::array<std::pair<float, float>, GID::NSources> minmaxR{{
    {1., 40.},   // ITS
    {85., 240.}, // TPC
    {-1, -1},    // TRD (never alone)
    {-1, -1},    // TOF
    {-1, -1},    // PHS
    {-1, -1},    // CPV
    {-1, -1},    // EMC
    {-1, -1},    // HMP
    {-1, -1},    // MFT
    {-1, -1},    // MCH
    {-1, -1},    // MID
    {-1, -1},    // ZDC
    {-1, -1},    // FT0
    {-1, -1},    // VF0
    {-1, -1},    // FDD
    {1., 240},   // ITSTPC
    {85., 405.}, // TPCTOF
    {85., 372.}, // TPCTRD
    {1., 372.},  // ITSTPCTRD
    {1., 405.},  // ITSTPCTOF,
    {85., 405.}, // TPCTRDTOF,
    {1., 405.},  // ITSTPCTRDTOF, // full barrel track
    {-1, -1},    // ITSAB,
  }};
  static constexpr std::array<std::pair<float, float>, GID::NSources> minmaxZ{{
    {-74., 74.},   // ITS
    {-260., 260.}, // TPC
    {-1, -1},      // TRD (never alone)
    {-1, -1},      // TOF
    {-1, -1},      // PHS
    {-1, -1},      // CPV
    {-1, -1},      // EMC
    {-1, -1},      // HMP
    {-1, -1},      // MFT
    {-1, -1},      // MCH
    {-1, -1},      // MID
    {-1, -1},      // ZDC
    {-1, -1},      // FT0
    {-1, -1},      // VF0
    {-1, -1},      // FDD
    {-260., 260.}, // ITSTPC
    {-375., 375.}, // TPCTOF
    {-375., 375.}, // TPCTRD
    {-375., 375.}, // ITSTPCTRD
    {-375., 375.}, // ITSTPCTOF,
    {-375., 375.}, // TPCTRDTOF,
    {-375., 375.}, // ITSTPCTRDTOF, // full barrel track
    {-1, -1},      // ITSAB,
  }};
  std::unique_ptr<gpu::TPCFastTransform> mTPCFastTransform;

 public:
  using AODBarrelTracks = soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra>;
  using AODBarrelTrack = AODBarrelTracks::iterator;

  using AODForwardTracks = soa::Join<aod::FwdTracks, aod::FwdTracksCov>;
  using AODForwardTrack = AODForwardTracks::iterator;

  using AODMFTTracks = aod::MFTTracks;
  using AODMFTTrack = AODMFTTracks::iterator;

  enum Filter : uint8_t {
    ITSROF,
    TimeBracket,
    EtaBracket,
    TotalNTracks,
    NFilters
  };

  using FilterSet = std::bitset<Filter::NFilters>;

  using Bracket = o2::math_utils::Bracketf_t;

  EveWorkflowHelper(const FilterSet& enabledFilters = {}, std::size_t maxNTracks = -1, const Bracket& timeBracket = {}, const Bracket& etaBracket = {});
  static std::vector<PNT> getTrackPoints(const o2::track::TrackPar& trc, float minR, float maxR, float maxStep, float minZ = -25000, float maxZ = 25000);
  void selectTracks(const CalibObjectsConst* calib, GID::mask_t maskCl,
                    GID::mask_t maskTrk, GID::mask_t maskMatch, bool trackSorting);
  void addTrackToEvent(const o2::track::TrackParCov& tr, GID gid, float trackTime, float dz, GID::Source source = GID::NSources, float maxStep = 4.f);
  void draw();
  void drawTPC(GID gid, float trackTime);
  void drawITS(GID gid, float trackTime);
  void drawMFT(GID gid, float trackTime);
  void drawMCH(GID gid, float trackTime);
  void drawMID(GID gid, float trackTime);
  void drawITSTPC(GID gid, float trackTime, GID::Source source = GID::ITSTPC);
  void drawITSTPCTOF(GID gid, float trackTime);
  void drawITSTPCTRD(GID gid, float trackTime);
  void drawITSTPCTRDTOF(GID gid, float trackTime);
  void drawTPCTRDTOF(GID gid, float trackTime);
  void drawTPCTRD(GID gid, float trackTime);
  void drawTPCTOF(GID gid, float trackTime);
  void drawPHOS();
  void drawAODBarrel(AODBarrelTrack const& track, float trackTime);
  void drawAODMFT(AODMFTTrack const& track, float trackTime);
  void drawITSClusters(GID gid, float trackTime);
  void drawTPCClusters(GID gid, float trackTime);
  void drawMFTClusters(GID gid, float trackTime);
  void drawMCHClusters(GID gid, float trackTime);
  void drawMIDClusters(GID gid, float trackTime);
  void drawTRDClusters(const o2::trd::TrackTRD& trc, float trackTime);
  void drawTOFClusters(GID gid, float trackTime);
  void drawPoint(float x, float y, float z, float trackTime) { mEvent.addCluster(x, y, z, trackTime); }
  void prepareITSClusters(const o2::itsmft::TopologyDictionary* dict); // fills mITSClustersArray
  void prepareMFTClusters(const o2::itsmft::TopologyDictionary* dict); // fills mMFTClustersArray
  void clear() { mEvent.clear(); }

  GID::Source detectorMapToGIDSource(uint8_t dm);

  void save(const std::string& jsonPath,
            int numberOfFiles,
            o2::dataformats::GlobalTrackID::mask_t trkMask,
            o2::dataformats::GlobalTrackID::mask_t clMask,
            o2::header::DataHeader::RunNumberType runNumber,
            o2::framework::DataProcessingHeader::CreationTime creationTime);

  FilterSet mEnabledFilters;
  std::size_t mMaxNTracks;
  Bracket mTimeBracket;
  Bracket mEtaBracket;
  o2::globaltracking::RecoContainer mRecoCont;
  o2::globaltracking::RecoContainer& getRecoContainer() { return mRecoCont; }
  TracksSet mTrackSet;
  o2::event_visualisation::VisualisationEvent mEvent;
  std::unordered_map<o2::dataformats::GlobalTrackID, std::size_t> mTotalTracks;
  std::vector<o2::BaseCluster<float>> mITSClustersArray;
  std::vector<o2::BaseCluster<float>> mMFTClustersArray;
  o2::mft::GeometryTGeo* mMFTGeom;
  o2::its::GeometryTGeo* mITSGeom;
  o2::phos::Geometry* mPHOSGeom;

  float mMUS2TPCTimeBins = 5.0098627;
  float mITSROFrameLengthMUS = 0; ///< ITS RO frame in mus
  float mMFTROFrameLengthMUS = 0; ///< MFT RO frame in mus
  float mTPCBin2MUS = 0;
  const o2::vertexing::PVertexerParams* mPVParams = nullptr;
};
} // namespace o2::event_visualisation

#endif // ALICE_O2_EVENTVISUALISATION_WORKFLOW_EVEWORKFLOWHELPER_H
