// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file RecoContainer.h
/// \brief Wrapper container for different reconstructed object types
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_RECO_CONTAINER
#define ALICEO2_RECO_CONTAINER

#include "Framework/ProcessingContext.h"
#include "Framework/InputSpec.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "ReconstructionDataFormats/GlobalTrackAccessor.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "CommonDataFormat/AbstractRefAccessor.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include <gsl/span>
#include <memory>

// We forward declare the internal structures, to reduce header dependencies.
// Please include headers for TPC Hits or TRD tracklets directly (DataFormatsTPC/WorkflowHelper.h / DataFormatsTRD/RecoInputContainer.h)
namespace o2::tpc
{
using TPCClRefElem = uint32_t;
struct ClusterNativeAccess;
namespace internal
{
struct getWorkflowTPCInput_ret;
} // namespace internal
} // namespace o2::tpc
namespace o2::trd
{
class Tracklet64;
class CalibratedTracklet;
class TriggerRecord;
struct RecoInputContainer;
} // namespace o2::trd

namespace o2
{
namespace globaltracking
{

struct DataRequest {
  std::vector<o2::framework::InputSpec> inputs;
  std::unordered_map<std::string, bool> requestMap;
  void addInput(const o2::framework::InputSpec&& isp);

  void requestTracks(o2::dataformats::GlobalTrackID::mask_t src, bool mc);
  void requestClusters(o2::dataformats::GlobalTrackID::mask_t src, bool useMC);

  void requestITSTracks(bool mc);
  void requestTPCTracks(bool mc);
  void requestITSTPCTracks(bool mc);
  void requestTPCTOFTracks(bool mc);
  void requestTOFMatches(bool mc);
  void requestFT0RecPoints(bool mc);

  void requestITSClusters(bool mc);
  void requestTPCClusters(bool mc);
  void requestTOFClusters(bool mc);
  void requestTRDTracklets();
};

struct RecoContainer {
  RecoContainer();
  ~RecoContainer();
  using TracksAccessor = o2::dataformats::GlobalTrackAccessor;
  using VariaAccessor = o2::dataformats::AbstractRefAccessor<int, o2::dataformats::GlobalTrackID::NSources>; // there is no unique <Varia> structure, so the default return type is dummy (int)
  using MCAccessor = o2::dataformats::AbstractRefAccessor<o2::MCCompLabel, o2::dataformats::GlobalTrackID::NSources>;
  using GTrackID = o2::dataformats::GlobalTrackID;

  using GlobalIDSet = std::array<GTrackID, GTrackID::NSources>;

  o2::InteractionRecord startIR; // TF start IR

  TracksAccessor tracksPool;    // container for tracks
  MCAccessor tracksMCPool;      // container for tracks MC info
  VariaAccessor clusRefPool;    // container for cluster references
  VariaAccessor tracksROFsPool; // container for tracks ROF records
  VariaAccessor clusROFPool;    // container for cluster ROF records
  VariaAccessor clustersPool;   // container for clusters
  VariaAccessor miscPool;       // container for misc info, e.g. patterns, match info w/o tracks etc.

  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mcITSClusters;
  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mcTOFClusters;

  gsl::span<const unsigned char> clusterShMapTPC; ///< externally set TPC clusters sharing map

  std::unique_ptr<o2::tpc::internal::getWorkflowTPCInput_ret> inputsTPCclusters; // special struct for TPC clusters access
  std::unique_ptr<o2::trd::RecoInputContainer> inputsTRD;                        // special struct for TRD tracklets, trigger records

  void collectData(o2::framework::ProcessingContext& pc, const DataRequest& request);
  void createTracks(std::function<bool(const o2::track::TrackParCov&, float, float, GTrackID)> const& creator) const;
  void fillTrackMCLabels(const gsl::span<GTrackID> gids, std::vector<o2::MCCompLabel>& mcinfo) const;

  void addITSTracks(o2::framework::ProcessingContext& pc, bool mc);
  void addTPCTracks(o2::framework::ProcessingContext& pc, bool mc);

  void addITSTPCTracks(o2::framework::ProcessingContext& pc, bool mc);
  void addTPCTOFTracks(o2::framework::ProcessingContext& pc, bool mc);
  void addTOFMatches(o2::framework::ProcessingContext& pc, bool mc);

  void addITSClusters(o2::framework::ProcessingContext& pc, bool mc);
  void addTPCClusters(o2::framework::ProcessingContext& pc, bool mc);
  void addTOFClusters(o2::framework::ProcessingContext& pc, bool mc);
  void addTRDTracklets(o2::framework::ProcessingContext& pc);

  void addFT0RecPoints(o2::framework::ProcessingContext& pc, bool mc);

  // custom getters

  // get contributors from single detectors: return array with sources set to all contributing GTrackIDs
  GlobalIDSet getSingleDetectorRefs(GTrackID gidx) const;

  // get contributing TPC GTrackID to the source. If source gidx is not contributed by TPC,
  // returned GTrackID.isSourceSet()==false
  GTrackID getTPCContributorGID(GTrackID source) const;

  // get contributing ITS GTrackID to the source. If source gidx is not contributed by ITS,
  // returned GTrackID.isSourceSet()==false
  GTrackID getITSContributorGID(GTrackID source) const;

  // check if track source attached
  bool isTrackSourceLoaded(int src) const;

  // check if match source attached
  bool isMatchSourceLoaded(int src) const
  {
    return miscPool.isLoaded(src);
  }

  // fetch track param
  const o2::track::TrackParCov& getTrack(GTrackID gidx) const;

  // fetch outer param
  const o2::track::TrackParCov& getTrackParamOut(GTrackID gidx) const;

  //--- getters: to avoid exposing all headers here, we use templates
  // ITS
  template <typename U> // o2::its::TrackITS
  auto getITSTracks() const
  {
    return tracksPool.getSpan<U>(GTrackID::ITS);
  }

  template <typename U> // o2::its::TrackITS or TrackParCov
  auto getITSTrack(GTrackID id) const
  {
    return tracksPool.get_as<U>(id);
  }

  template <typename U> // o2::itsmft::ROFRecord
  auto getITSTracksROFRecords() const
  {
    return tracksROFsPool.getSpan<U>(GTrackID::ITS);
  }

  auto getITSTracksClusterRefs() const { return clusRefPool.getSpan<int>(GTrackID::ITS); }
  auto getITSTracksMCLabels() const { return tracksMCPool.getSpan<o2::MCCompLabel>(GTrackID::ITS); }
  auto getITSTrackMCLabel(GTrackID id) const { return tracksMCPool.get_as<o2::MCCompLabel>(id); }

  // TPC
  template <typename U> // o2::tpc::TrackTPC
  auto getTPCTracks() const
  {
    return tracksPool.getSpan<U>(GTrackID::TPC);
  }

  template <typename U> // o2::tpc::TrackTPC or TrackParCov
  auto getTPCTrack(GTrackID id) const
  {
    return tracksPool.get_as<U>(id);
  }

  auto getTPCTracksClusterRefs() const { return clusRefPool.getSpan<o2::tpc::TPCClRefElem>(GTrackID::TPC); }
  auto getTPCTracksMCLabels() const { return tracksMCPool.getSpan<o2::MCCompLabel>(GTrackID::TPC); }
  auto getTPCTrackMCLabel(GTrackID id) const { return tracksMCPool.get_as<o2::MCCompLabel>(id); }

  // ITS-TPC
  template <typename U> // o2::dataformats::TrackTPCITS
  auto getTPCITSTracks() const
  {
    return tracksPool.getSpan<U>(GTrackID::ITSTPC);
  }

  template <typename U> // o2::dataformats::TrackTPCITS or TrackParCov
  auto getTPCITSTrack(GTrackID id) const
  {
    return tracksPool.get_as<U>(id);
  }

  auto getTPCITSTracksMCLabels() const { return tracksMCPool.getSpan<o2::MCCompLabel>(GTrackID::ITSTPC); }
  auto getTPCITSTrackMCLabel(GTrackID id) const { return tracksMCPool.get_as<o2::MCCompLabel>(id); }

  // TPC-TOF
  template <typename U> // o2::dataformats::TrackTPCTOF
  auto getTPCTOFTracks() const
  {
    return tracksPool.getSpan<U>(GTrackID::TPCTOF);
  }

  template <typename U> // o2::dataformats::MatchInfoTOF
  auto getTPCTOFMatches() const
  {
    return miscPool.getSpan<U>(GTrackID::TPCTOF);
  }

  template <typename U> // o2::dataformats::MatchInfoTOF
  auto getTPCTOFMatch(GTrackID id) const
  {
    return miscPool.get_as<U>(id);
  }

  template <typename U> // o2::dataformats::TrackTPCTOF or TrackParCov
  auto getTPCTOFTrack(GTrackID id) const
  {
    return tracksPool.get_as<U>(id);
  }

  auto getTPCTOFTracksMCLabels() const { return tracksMCPool.getSpan<o2::MCCompLabel>(GTrackID::TPCTOF); }
  auto getTPCTOFTrackMCLabel(GTrackID id) const { return tracksMCPool.get_as<o2::MCCompLabel>(id); }

  // global TOF matches
  template <typename U> // o2::dataformats::MatchInfoTOF
  auto getTOFMatches() const
  {
    return miscPool.getSpan<U>(GTrackID::ITSTPCTOF);
  }

  template <typename U> // o2::dataformats::MatchInfoTOF
  auto getTOFMatch(GTrackID id) const
  {
    return miscPool.get_as<U>(id);
  }

  // TPC clusters
  const o2::tpc::ClusterNativeAccess& getTPCClusters() const;

  // TRD tracklets
  gsl::span<const o2::trd::Tracklet64> getTRDTracklets() const;
  gsl::span<const o2::trd::CalibratedTracklet> getTRDCalibratedTracklets() const;
  gsl::span<const o2::trd::TriggerRecord> getTRDTriggerRecords() const;

  // ITS clusters
  template <typename U> // o2::itsmft::ROFRecord
  auto getITSClustersROFRecords() const
  {
    return clusROFPool.getSpan<U>(GTrackID::ITS);
  }

  template <typename U> // o2::itsmft::CompClusterExt
  auto getITSClusters() const
  {
    return clustersPool.getSpan<U>(GTrackID::ITS);
  }

  auto getITSClustersPatterns() const
  {
    return miscPool.getSpan<unsigned char>(GTrackID::ITS);
  }

  // TOF clusters
  template <typename U> // o2::tof::Cluster
  auto getTOFClusters() const
  {
    return clustersPool.getSpan<U>(GTrackID::TOF);
  }

  o2::MCCompLabel getTrackMCLabel(GTrackID id) const
  {
    return tracksMCPool.get(id);
  }

  // FT0
  template <typename U> // o2::ft0::RecPoints
  auto getFT0RecPoints() const
  {
    return miscPool.getSpan<U>(GTrackID::FT0);
  }
};

} // namespace globaltracking
} // namespace o2

#endif
