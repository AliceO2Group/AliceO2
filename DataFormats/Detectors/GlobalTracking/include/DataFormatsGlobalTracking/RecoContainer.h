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

#include "CommonDataFormat/InteractionRecord.h"
#include "ReconstructionDataFormats/GlobalTrackAccessor.h"
#include "CommonDataFormat/RangeReference.h"
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
class TrackTPC;
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
//class TrackTRD;
struct RecoInputContainer;
} // namespace o2::trd

namespace o2::framework
{
class ProcessingContext;
struct InputSpec;
} // namespace o2::framework

namespace o2::its
{
class TrackITS;
}

namespace o2::itsmft
{
class ROFRecord;
class CompClusterExt;
} // namespace o2::itsmft

namespace o2::tof
{
class Cluster;
}

namespace o2::ft0
{
class RecPoints;
}

namespace o2::dataformats
{
class TrackTPCITS;
class TrackTPCTOF;
class MatchInfoTOF;
class PrimaryVertex;
class VtxTrackIndex;
class VtxTrackRef;
class V0;
class Cascade;
} // namespace o2::dataformats

namespace o2
{
class MCEventLabel;
}

namespace o2
{
namespace globaltracking
{

// helper class to request DPL input data from the processor specs definition
struct DataRequest {
  std::vector<o2::framework::InputSpec> inputs;
  std::unordered_map<std::string, bool> requestMap;
  void addInput(const o2::framework::InputSpec&& isp);

  bool isRequested(const std::string& t) const { return !t.empty() && requestMap.find(t) != requestMap.end(); }
  void requestTracks(o2::dataformats::GlobalTrackID::mask_t src, bool mc);
  void requestClusters(o2::dataformats::GlobalTrackID::mask_t src, bool useMC);

  void requestITSTracks(bool mc);
  void requestTPCTracks(bool mc);
  void requestITSTPCTracks(bool mc);
  void requestTPCTOFTracks(bool mc);
  void requestITSTPCTRDTracks(bool mc);
  void requestTPCTRDTracks(bool mc);
  void requestTOFMatches(bool mc);
  void requestFT0RecPoints(bool mc);

  void requestITSClusters(bool mc);
  void requestTPCClusters(bool mc);
  void requestTOFClusters(bool mc);
  void requestTRDTracklets(bool mc);

  void requestPrimaryVertertices(bool mc);
  void requestPrimaryVerterticesTMP(bool mc);
  void requestSecondaryVertertices(bool mc);
};

// Helper class to requested data.
// Most common data have dedicated getters, some need to be called with returned typa as a template.
// In general on either gets a gsl::span<const Type> via getter like e.g. getITSTracks()
// of a reference to particular track, i.e. getITSTrack(GlobalTrackID id).
// Note that random access like getITSTracks()[i] has an overhead, since for every call a span is created.
// Therefore, for the random access better to use direct getter, i.e. auto& tr = getITSTrack(gid)
// while for looping over the whole span first create a span then iterate over it.

struct RecoContainer {
  RecoContainer();
  ~RecoContainer();

  enum CommonSlots {
    TRACKS,
    MATCHES,
    MATCHESEXTRA,
    TRACKREFS,
    CLUSREFS,
    CLUSTERS,
    PATTERNS,
    INDICES,
    MCLABELS,      // track labels
    MCLABELSEXTRA, // additonal labels, like TOF clusters matching label (not sure TOF really needs it)
    NCOMMONSLOTS
  };

  // slots to register primary vertex data
  enum PVTXSlots { PVTX,            // Primary vertices
                   PVTX_TRMTC,      // matched track indices
                   PVTX_TRMTCREFS,  // PV -> matched tracks referencing object
                   PVTX_CONTID,     // contributors indices
                   PVTX_CONTIDREFS, // PV -> contributors indices
                   PVTX_MCTR,       // PV MC label
                   NPVTXSLOTS };

  // slots to register secondary vertex data
  enum SVTXSlots { V0S,           // V0 objects
                   PVTX_V0REFS,   // PV -> V0 references
                   CASCS,         // Cascade objects
                   PVTX_CASCREFS, // PV -> Cascade reference
                   NSVTXSLOTS };

  using AccSlots = o2::dataformats::AbstractRefAccessor<int, NCOMMONSLOTS>; // int here is a dummy placeholder
  using PVertexAccessor = o2::dataformats::AbstractRefAccessor<int, NPVTXSLOTS>;
  using SVertexAccessor = o2::dataformats::AbstractRefAccessor<int, NSVTXSLOTS>;
  using GTrackID = o2::dataformats::GlobalTrackID;
  using GlobalIDSet = std::array<GTrackID, GTrackID::NSources>;

  o2::InteractionRecord startIR; // TF start IR

  std::array<AccSlots, GTrackID::NSources> commonPool;
  PVertexAccessor pvtxPool; // containers for primary vertex related objects
  SVertexAccessor svtxPool; // containers for secondary vertex related objects

  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mcITSClusters;
  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mcTOFClusters;

  gsl::span<const unsigned char> clusterShMapTPC; ///< externally set TPC clusters sharing map

  std::unique_ptr<o2::tpc::internal::getWorkflowTPCInput_ret> inputsTPCclusters; // special struct for TPC clusters access
  std::unique_ptr<o2::trd::RecoInputContainer> inputsTRD;                        // special struct for TRD tracklets, trigger records

  void collectData(o2::framework::ProcessingContext& pc, const DataRequest& request);
  void createTracks(std::function<bool(const o2::track::TrackParCov&, GTrackID)> const& creator) const;
  void createTracksWithMatchingTimeInfo(std::function<bool(const o2::track::TrackParCov&, GTrackID, float, float)> const& creator) const;
  template <class T>
  void createTracksVariadic(T creator) const;
  void fillTrackMCLabels(const gsl::span<GTrackID> gids, std::vector<o2::MCCompLabel>& mcinfo) const;

  void addITSTracks(o2::framework::ProcessingContext& pc, bool mc);
  void addTPCTracks(o2::framework::ProcessingContext& pc, bool mc);

  void addITSTPCTRDTracks(o2::framework::ProcessingContext& pc, bool mc);
  void addTPCTRDTracks(o2::framework::ProcessingContext& pc, bool mc);
  void addITSTPCTracks(o2::framework::ProcessingContext& pc, bool mc);
  void addTPCTOFTracks(o2::framework::ProcessingContext& pc, bool mc);
  void addTOFMatches(o2::framework::ProcessingContext& pc, bool mc);

  void addITSClusters(o2::framework::ProcessingContext& pc, bool mc);
  void addTPCClusters(o2::framework::ProcessingContext& pc, bool mc, bool shmap);
  void addTOFClusters(o2::framework::ProcessingContext& pc, bool mc);
  void addTRDTracklets(o2::framework::ProcessingContext& pc);

  void addFT0RecPoints(o2::framework::ProcessingContext& pc, bool mc);

  void addPVertices(o2::framework::ProcessingContext& pc, bool mc);
  void addPVerticesTMP(o2::framework::ProcessingContext& pc, bool mc);
  void addSVertices(o2::framework::ProcessingContext& pc, bool);

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
  bool isMatchSourceLoaded(int src) const { return commonPool[src].isLoaded(MATCHES); }

  // -------------------------------
  // generic getter for span
  template <typename U>
  gsl::span<const U> getSpan(int src, int slotID) const
  {
    return commonPool[src].getSpan<U>(slotID);
  }

  // generic getter for an object in the attaches span
  template <typename U>
  const U& getObject(int src, int index, int slotID) const
  {
    return commonPool[src].get_as<U>(slotID, index);
  }
  template <typename U>
  const U& getObject(GTrackID gid, int slotID) const
  {
    return getObject<U>(gid.getSource(), gid.getIndex(), slotID);
  }
  // <<TODO Make this private

  //-------------------------
  // in general, U can be a TrackParCov or particular detector track, e.g. o2::its::TrackITS
  template <typename U>
  gsl::span<const U> getTracks(int src) const
  {
    return getSpan<U>(src, TRACKS);
  }

  template <typename U>
  const U& getTrack(int src, int id) const
  {
    return getObject<U>(src, id, TRACKS);
  }

  template <typename U>
  const U& getTrack(GTrackID gid) const
  {
    return getObject<U>(gid, TRACKS);
  }

  o2::MCCompLabel getTrackMCLabel(GTrackID id) const { return getObject<o2::MCCompLabel>(id, MCLABELS); }

  //--------------------------------------------
  // fetch track param
  const o2::track::TrackParCov& getTrackParam(GTrackID gidx) const;
  // fetch outer param (not all track types might have it)
  const o2::track::TrackParCov& getTrackParamOut(GTrackID gidx) const;

  //--------------------------------------------
  // ITS
  const o2::its::TrackITS& getITSTrack(GTrackID gid) const { return getTrack<o2::its::TrackITS>(gid); }
  auto getITSTracks() const { return getTracks<o2::its::TrackITS>(GTrackID::ITS); }
  auto getITSTracksROFRecords() const { return getSpan<o2::itsmft::ROFRecord>(GTrackID::ITS, TRACKREFS); }
  auto getITSTracksClusterRefs() const { return getSpan<int>(GTrackID::ITS, INDICES); }
  auto getITSTracksMCLabels() const { return getSpan<o2::MCCompLabel>(GTrackID::ITS, MCLABELS); }
  // ITS clusters
  auto getITSClustersROFRecords() const { return getSpan<o2::itsmft::ROFRecord>(GTrackID::ITS, CLUSREFS); }
  auto getITSClusters() const { return getSpan<o2::itsmft::CompClusterExt>(GTrackID::ITS, CLUSTERS); }
  auto getITSClustersPatterns() const { return getSpan<unsigned char>(GTrackID::ITS, PATTERNS); }
  auto getITSClustersMCLabels() const { return mcITSClusters.get(); }

  // TPC
  const o2::tpc::TrackTPC& getTPCTrack(GTrackID id) const { return getTrack<o2::tpc::TrackTPC>(id); }
  auto getTPCTracks() const { return getTracks<o2::tpc::TrackTPC>(GTrackID::TPC); }
  auto getTPCTracksClusterRefs() const { return getSpan<o2::tpc::TPCClRefElem>(GTrackID::TPC, INDICES); }
  auto getTPCTracksMCLabels() const { return getSpan<o2::MCCompLabel>(GTrackID::TPC, MCLABELS); }
  auto getTPCTrackMCLabel(GTrackID id) const { return getObject<o2::MCCompLabel>(id, MCLABELS); }
  const o2::tpc::ClusterNativeAccess& getTPCClusters() const;
  const o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>* getTPCClustersMCLabels() const;

  // ITS-TPC
  const o2::dataformats::TrackTPCITS& getTPCITSTrack(GTrackID gid) const { return getTrack<o2::dataformats::TrackTPCITS>(gid); }
  auto getTPCITSTracks() const { return getTracks<o2::dataformats::TrackTPCITS>(GTrackID::ITSTPC); }
  auto getTPCITSTracksMCLabels() const { return getSpan<o2::MCCompLabel>(GTrackID::ITSTPC, MCLABELS); }
  auto getTPCITSTrackMCLabel(GTrackID id) const { return getObject<o2::MCCompLabel>(id, MCLABELS); }

  // ITS-TPC-TRD, since the TrackTRD track is just an alias, forward-declaring it does not work, need to keep template
  template <class U>
  auto getITSTPCTRDTrack(GTrackID id) const
  {
    return getTrack<U>(id);
  }
  template <class U>
  auto getITSTPCTRDTracks() const
  {
    return getTracks<U>(GTrackID::ITSTPCTRD);
  }
  // TPC-TRD
  template <class U>
  auto getTPCTRDTrack(GTrackID id) const
  {
    return getTrack<U>(id);
  }
  template <class U>
  auto getTPCTRDTracks() const
  {
    return getTracks<U>(GTrackID::TPCTRD);
  }
  // TRD tracklets
  gsl::span<const o2::trd::Tracklet64> getTRDTracklets() const;
  gsl::span<const o2::trd::CalibratedTracklet> getTRDCalibratedTracklets() const;
  gsl::span<const o2::trd::TriggerRecord> getTRDTriggerRecords() const;
  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* getTRDTrackletsMCLabels() const;

  // TPC-TOF, made of refitted TPC track and separate matchInfo
  const o2::dataformats::TrackTPCTOF& getTPCTOFTrack(GTrackID gid) const { return getTrack<o2::dataformats::TrackTPCTOF>(gid); }
  const o2::dataformats::MatchInfoTOF& getTPCTOFMatch(GTrackID id) const { return getObject<o2::dataformats::MatchInfoTOF>(id, MATCHES); }
  auto getTPCTOFTrackMCLabel(GTrackID id) const { return getObject<o2::MCCompLabel>(id, MCLABELS); }
  auto getTPCTOFTracks() const { return getTracks<o2::dataformats::TrackTPCTOF>(GTrackID::TPCTOF); }
  auto getTPCTOFMatches() const { return getSpan<o2::dataformats::MatchInfoTOF>(GTrackID::TPCTOF, MATCHES); }
  auto getTPCTOFTracksMCLabels() const { return getSpan<o2::MCCompLabel>(GTrackID::TPCTOF, MCLABELS); }
  // global ITS-TPC-TOF matches, TODO: add ITS-TPC-TRD-TOF and TPC-TRD-TOF
  const o2::dataformats::MatchInfoTOF& getTOFMatch(GTrackID id) const { return getObject<o2::dataformats::MatchInfoTOF>(id, MATCHES); }
  const o2::dataformats::TrackTPCITS& getITSTPCTOFTrack(GTrackID id) const; // this is special since global TOF track is just a reference on TPCITS
  auto getTOFMatches() const { return getSpan<o2::dataformats::MatchInfoTOF>(GTrackID::ITSTPCTOF, MATCHES); }
  auto getTOFMatchesMCLabels() const { return getSpan<o2::MCCompLabel>(GTrackID::ITSTPCTOF, MCLABELS); }
  // TOF clusters
  auto getTOFClusters() const { return getSpan<o2::tof::Cluster>(GTrackID::TOF, CLUSTERS); }
  auto getTOFClustersMCLabels() const { return mcTOFClusters.get(); }

  // FT0
  auto getFT0RecPoints() const { return getSpan<o2::ft0::RecPoints>(GTrackID::FT0, TRACKS); }

  // Primary vertices
  const o2::dataformats::PrimaryVertex& getPrimaryVertex(int i) const { return pvtxPool.get_as<o2::dataformats::PrimaryVertex>(PVTX, i); }
  const o2::MCEventLabel& getPrimaryVertexMCLabel(int i) const { return pvtxPool.get_as<o2::MCEventLabel>(PVTX_MCTR, i); }
  auto getPrimaryVertices() const { return pvtxPool.getSpan<o2::dataformats::PrimaryVertex>(PVTX); }
  auto getPrimaryVertexMatchedTracks() const { return pvtxPool.getSpan<o2::dataformats::VtxTrackIndex>(PVTX_TRMTC); }
  auto getPrimaryVertexContributors() const { return pvtxPool.getSpan<o2::dataformats::VtxTrackIndex>(PVTX_CONTID); }
  auto getPrimaryVertexMatchedTrackRefs() const { return pvtxPool.getSpan<o2::dataformats::VtxTrackRef>(PVTX_TRMTCREFS); }
  auto getPrimaryVertexContributorsRefs() const { return pvtxPool.getSpan<o2::dataformats::VtxTrackRef>(PVTX_CONTIDREFS); }
  auto getPrimaryVertexMCLabels() const { return pvtxPool.getSpan<o2::MCEventLabel>(PVTX_MCTR); }

  // Secondary vertices
  const o2::dataformats::V0& getV0(int i) const { return svtxPool.get_as<o2::dataformats::V0>(V0S, i); }
  const o2::dataformats::Cascade& getCascade(int i) const { return svtxPool.get_as<o2::dataformats::Cascade>(CASCS, i); }
  auto getV0s() const { return svtxPool.getSpan<o2::dataformats::V0>(V0S); }
  auto getPV2V0Refs() { return svtxPool.getSpan<o2::dataformats::RangeReference<int, int>>(PVTX_V0REFS); }
  auto getCascades() const { return svtxPool.getSpan<o2::dataformats::Cascade>(CASCS); }
  auto getPV2CascadesRefs() { return svtxPool.getSpan<o2::dataformats::RangeReference<int, int>>(PVTX_CASCREFS); }
};

} // namespace globaltracking
} // namespace o2

#endif
