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

/// \file RecoContainer.h
/// \brief Wrapper container for different reconstructed object types
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_RECO_CONTAINER
#define ALICEO2_RECO_CONTAINER

#include "CommonDataFormat/InteractionRecord.h"
#include "ReconstructionDataFormats/GlobalTrackAccessor.h"
#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/DecayNbody.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/MatchingType.h"
#include "CommonDataFormat/AbstractRefAccessor.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "DataFormatsCTP/LumiInfo.h"
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
class TrackTriggerRecord;
// class TrackTRD;
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

namespace o2::mft
{
class TrackMFT;
}

namespace o2::mch
{
class TrackMCH;
class ROFRecord;
class Cluster;
} // namespace o2::mch

namespace o2::mid
{
class Cluster;
class ROFRecord;
class Track;
class MCClusterLabel;
class MCLabel;
} // namespace o2::mid

namespace o2::itsmft
{
class ROFRecord;
class CompClusterExt;
class TrkClusRef;
} // namespace o2::itsmft

namespace o2::tof
{
class Cluster;
}

namespace o2::hmpid
{
class Cluster;
class Trigger;
} // namespace o2::hmpid

namespace o2::ft0
{
class RecPoints;
class ChannelDataFloat;
} // namespace o2::ft0

namespace o2::fv0
{
class RecPoints;
class ChannelDataFloat;
} // namespace o2::fv0

namespace o2::zdc
{
class BCRecData;
class ZDCEnergy;
class ZDCTDCData;
} // namespace o2::zdc

namespace o2::fdd
{
class RecPoint;
class ChannelDataFloat;
} // namespace o2::fdd

namespace o2::ctp
{
class CTPDigit;
} // namespace o2::ctp

namespace o2::phos
{
class Cell;
class TriggerRecord;
class MCLabel;
} // namespace o2::phos

namespace o2::cpv
{
class Cluster;
class TriggerRecord;
} // namespace o2::cpv

namespace o2::emcal
{
class Cell;
class TriggerRecord;
class MCLabel;
} // namespace o2::emcal

namespace o2::dataformats
{
class TrackTPCITS;
class TrackTPCTOF;
class MatchInfoTOF;
class MatchInfoHMP;
class PrimaryVertex;
class VtxTrackIndex;
class VtxTrackRef;
class V0;
class Cascade;
class StrangeTrack;
class TrackCosmics;
class GlobalFwdTrack;
class MatchInfoFwd;
class TrackMCHMID;
class IRFrame;
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
  MatchingType matchingInputType = MatchingType::Standard; // use subspec = 0 for inputs

  auto getMatchingInputType() const { return matchingInputType; }
  void setMatchingInputStrict() { matchingInputType = MatchingType::Strict; }
  void setMatchingInputFull() { matchingInputType = MatchingType::Full; }
  void setMatchingInputStandard() { matchingInputType = MatchingType::Standard; }
  uint32_t getMatchingInputSubSpec() const { return getSubSpec(matchingInputType); }

  void addInput(const o2::framework::InputSpec&& isp);

  bool isRequested(const std::string& t) const { return !t.empty() && requestMap.find(t) != requestMap.end(); }
  void requestTracks(o2::dataformats::GlobalTrackID::mask_t src, bool mc);
  void requestClusters(o2::dataformats::GlobalTrackID::mask_t src, bool useMC, o2::detectors::DetID::mask_t skipDetClusters = {});

  void requestITSTracks(bool mc);
  void requestMFTTracks(bool mc);
  void requestMCHTracks(bool mc);
  void requestMIDTracks(bool mc);
  void requestTPCTracks(bool mc);
  void requestITSTPCTracks(bool mc);
  void requestGlobalFwdTracks(bool mc);
  void requestMFTMCHMatches(bool mc);
  void requestMCHMIDMatches(bool mc);
  void requestTPCTOFTracks(bool mc);
  void requestITSTPCTRDTracks(bool mc);
  void requestTPCTRDTracks(bool mc);
  void requestTOFMatches(o2::dataformats::GlobalTrackID::mask_t src, bool mc);
  void requestFT0RecPoints(bool mc);
  void requestFV0RecPoints(bool mc);
  void requestFDDRecPoints(bool mc);
  void requestZDCRecEvents(bool mc);
  void requestITSClusters(bool mc);
  void requestMFTClusters(bool mc);
  void requestTPCClusters(bool mc);
  void requestTOFClusters(bool mc);
  void requestTRDTracklets(bool mc);
  void requestMCHClusters(bool mc);
  void requestMIDClusters(bool mc);
  void requestHMPClusters(bool mc);
  void requestHMPMatches(bool mc); // no input available at the moment

  void requestCTPDigits(bool mc);

  void requestPHOSCells(bool mc);
  void requestEMCALCells(bool mc);
  void requestCPVClusters(bool mc);

  void requestCoscmicTracks(bool mc);

  void requestPrimaryVertertices(bool mc);
  void requestPrimaryVerterticesTMP(bool mc);
  void requestSecondaryVertices(bool mc);
  void requestStrangeTracks(bool mc);

  void requestIRFramesITS();
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
    VARIA,         // misc data, which does not fit to other categories
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
  enum SVTXSlots { V0S,            // V0 objects
                   PVTX_V0REFS,    // PV -> V0 references
                   CASCS,          // Cascade objects
                   PVTX_CASCREFS,  // PV -> Cascade reference
                   DECAY3BODY,     // 3-body decay objects
                   PVTX_3BODYREFS, // PV -> 3-body decay references
                   NSVTXSLOTS };

  // slots to register strangeness tracking data
  enum STRKSlots {
    STRACK,
    STRACK_MC,
    NSTRKSLOTS
  };

  // slots for cosmics
  enum CosmicsSlots { COSM_TRACKS,
                      COSM_TRACKS_MC,
                      NCOSMSLOTS };

  using AccSlots = o2::dataformats::AbstractRefAccessor<int, NCOMMONSLOTS>; // int here is a dummy placeholder
  using PVertexAccessor = o2::dataformats::AbstractRefAccessor<int, NPVTXSLOTS>;
  using SVertexAccessor = o2::dataformats::AbstractRefAccessor<int, NSVTXSLOTS>;
  using STrackAccessor = o2::dataformats::AbstractRefAccessor<int, NSTRKSLOTS>;
  using CosmicsAccessor = o2::dataformats::AbstractRefAccessor<int, NCOSMSLOTS>;
  using GTrackID = o2::dataformats::GlobalTrackID;
  using GlobalIDSet = std::array<GTrackID, GTrackID::NSources>;

  static constexpr float PS2MUS = 1e-6;

  o2::InteractionRecord startIR; // TF start IR

  std::array<AccSlots, GTrackID::NSources> commonPool;
  PVertexAccessor pvtxPool; // containers for primary vertex related objects
  SVertexAccessor svtxPool; // containers for secondary vertex related objects
  STrackAccessor strkPool;  // containers for strangeness tracking related objects
  CosmicsAccessor cosmPool; // containers for cosmics track data

  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mcITSClusters;
  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mcTOFClusters;
  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mcHMPClusters;
  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mcCPVClusters;
  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mcMCHClusters;
  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::phos::MCLabel>> mcPHSCells;
  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::emcal::MCLabel>> mcEMCCells;
  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::mid::MCClusterLabel>> mcMIDTrackClusters;
  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::mid::MCClusterLabel>> mcMIDClusters;
  std::unique_ptr<const std::vector<o2::MCCompLabel>> mcMIDTracks;
  o2::ctp::LumiInfo mCTPLumi;

  gsl::span<const unsigned char> clusterShMapTPC; ///< externally set TPC clusters sharing map

  std::unique_ptr<o2::tpc::internal::getWorkflowTPCInput_ret> inputsTPCclusters; // special struct for TPC clusters access
  std::unique_ptr<o2::trd::RecoInputContainer> inputsTRD;                        // special struct for TRD tracklets, trigger records

  void collectData(o2::framework::ProcessingContext& pc, const DataRequest& request);
  void createTracks(std::function<bool(const o2::track::TrackParCov&, GTrackID)> const& creator) const;
  template <class T>
  void createTracksVariadic(T creator, GTrackID::mask_t srcSel = GTrackID::getSourcesMask("all")) const;
  void fillTrackMCLabels(const gsl::span<GTrackID> gids, std::vector<o2::MCCompLabel>& mcinfo) const;

  void addITSTracks(o2::framework::ProcessingContext& pc, bool mc);
  void addMFTTracks(o2::framework::ProcessingContext& pc, bool mc);
  void addMCHTracks(o2::framework::ProcessingContext& pc, bool mc);
  void addMIDTracks(o2::framework::ProcessingContext& pc, bool mc);
  void addTPCTracks(o2::framework::ProcessingContext& pc, bool mc);

  void addITSTPCTRDTracks(o2::framework::ProcessingContext& pc, bool mc);
  void addTPCTRDTracks(o2::framework::ProcessingContext& pc, bool mc);
  void addITSTPCTracks(o2::framework::ProcessingContext& pc, bool mc);
  void addGlobalFwdTracks(o2::framework::ProcessingContext& pc, bool mc);
  void addTPCTOFTracks(o2::framework::ProcessingContext& pc, bool mc);
  void addTOFMatchesITSTPC(o2::framework::ProcessingContext& pc, bool mc);
  void addTOFMatchesTPCTRD(o2::framework::ProcessingContext& pc, bool mc);
  void addTOFMatchesITSTPCTRD(o2::framework::ProcessingContext& pc, bool mc);

  void addHMPMatches(o2::framework::ProcessingContext& pc, bool mc);
  void addMFTMCHMatches(o2::framework::ProcessingContext& pc, bool mc);
  void addMCHMIDMatches(o2::framework::ProcessingContext& pc, bool mc);

  void addITSClusters(o2::framework::ProcessingContext& pc, bool mc);
  void addMFTClusters(o2::framework::ProcessingContext& pc, bool mc);
  void addTPCClusters(o2::framework::ProcessingContext& pc, bool mc, bool shmap);
  void addTOFClusters(o2::framework::ProcessingContext& pc, bool mc);
  void addHMPClusters(o2::framework::ProcessingContext& pc, bool mc);
  void addTRDTracklets(o2::framework::ProcessingContext& pc, bool mc);
  void addMCHClusters(o2::framework::ProcessingContext& pc, bool mc);
  void addMIDClusters(o2::framework::ProcessingContext& pc, bool mc);

  void addFT0RecPoints(o2::framework::ProcessingContext& pc, bool mc);
  void addFV0RecPoints(o2::framework::ProcessingContext& pc, bool mc);
  void addFDDRecPoints(o2::framework::ProcessingContext& pc, bool mc);

  void addZDCRecEvents(o2::framework::ProcessingContext& pc, bool mc);

  void addCTPDigits(o2::framework::ProcessingContext& pc, bool mc);

  void addCPVClusters(o2::framework::ProcessingContext& pc, bool mc);
  void addPHOSCells(o2::framework::ProcessingContext& pc, bool mc);
  void addEMCALCells(o2::framework::ProcessingContext& pc, bool mc);

  void addCosmicTracks(o2::framework::ProcessingContext& pc, bool mc);

  void addPVertices(o2::framework::ProcessingContext& pc, bool mc);
  void addPVerticesTMP(o2::framework::ProcessingContext& pc, bool mc);
  void addSVertices(o2::framework::ProcessingContext& pc, bool);

  void addStrangeTracks(o2::framework::ProcessingContext& pc, bool mc);

  void addIRFramesITS(o2::framework::ProcessingContext& pc);

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

  o2::MCCompLabel getTrackMCLabel(GTrackID id) const
  {
    // RS FIXME: THIS IS TEMPORARY: some labels are still not implemented: in this case return dummy label
    return commonPool[id.getSource()].getSize(MCLABELS) ? getObject<o2::MCCompLabel>(id, MCLABELS) : o2::MCCompLabel{};
    // return getObject<o2::MCCompLabel>(id, MCLABELS);
  }

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
  auto getITSABRefs() const { return getSpan<o2::itsmft::TrkClusRef>(GTrackID::ITSAB, TRACKREFS); }
  const o2::itsmft::TrkClusRef& getITSABRef(GTrackID gid) const { return getObject<o2::itsmft::TrkClusRef>(gid, TRACKREFS); }
  auto getITSABClusterRefs() const { return getSpan<int>(GTrackID::ITSAB, INDICES); }
  auto getITSABMCLabels() const { return getSpan<o2::MCCompLabel>(GTrackID::ITSAB, MCLABELS); }

  // ITS clusters
  auto getITSClustersROFRecords() const { return getSpan<o2::itsmft::ROFRecord>(GTrackID::ITS, CLUSREFS); }
  auto getITSClusters() const { return getSpan<o2::itsmft::CompClusterExt>(GTrackID::ITS, CLUSTERS); }
  auto getITSClustersPatterns() const { return getSpan<unsigned char>(GTrackID::ITS, PATTERNS); }
  auto getITSClustersMCLabels() const { return mcITSClusters.get(); }

  // MFT
  const o2::mft::TrackMFT& getMFTTrack(GTrackID gid) const { return getTrack<o2::mft::TrackMFT>(gid); }
  auto getMFTTracks() const { return getTracks<o2::mft::TrackMFT>(GTrackID::MFT); }
  auto getMFTTracksROFRecords() const { return getSpan<o2::itsmft::ROFRecord>(GTrackID::MFT, TRACKREFS); }
  auto getMFTTracksClusterRefs() const { return getSpan<int>(GTrackID::MFT, INDICES); }
  auto getMFTTracksMCLabels() const { return getSpan<o2::MCCompLabel>(GTrackID::MFT, MCLABELS); }

  // MFT clusters
  auto getMFTClustersROFRecords() const { return getSpan<o2::itsmft::ROFRecord>(GTrackID::MFT, CLUSREFS); }
  auto getMFTClusters() const { return getSpan<o2::itsmft::CompClusterExt>(GTrackID::MFT, CLUSTERS); }
  auto getMFTClustersPatterns() const { return getSpan<unsigned char>(GTrackID::MFT, PATTERNS); }

  // MCH
  const o2::mch::TrackMCH& getMCHTrack(GTrackID gid) const { return getTrack<o2::mch::TrackMCH>(gid); }
  auto getMCHTracks() const { return getTracks<o2::mch::TrackMCH>(GTrackID::MCH); }
  auto getMCHTracksROFRecords() const { return getSpan<o2::mch::ROFRecord>(GTrackID::MCH, TRACKREFS); }
  auto getMCHTrackClusters() const { return getSpan<o2::mch::Cluster>(GTrackID::MCH, INDICES); }
  auto getMCHTracksMCLabels() const { return getSpan<o2::MCCompLabel>(GTrackID::MCH, MCLABELS); }

  // MCH clusters
  auto getMCHClusterROFRecords() const { return getSpan<o2::mch::ROFRecord>(GTrackID::MCH, CLUSREFS); }
  auto getMCHClusters() const { return getSpan<o2::mch::Cluster>(GTrackID::MCH, CLUSTERS); }
  auto getMCHClustersMCLabels() const { return mcMCHClusters.get(); }

  // MID
  const o2::mid::Track& getMIDTrack(GTrackID gid) const { return getTrack<o2::mid::Track>(gid); }
  auto getMIDTracks() const { return getTracks<o2::mid::Track>(GTrackID::MID); }
  auto getMIDTracksROFRecords() const { return getSpan<o2::mid::ROFRecord>(GTrackID::MID, TRACKREFS); }
  auto getMIDTrackClusters() const { return getSpan<o2::mid::Cluster>(GTrackID::MID, INDICES); }
  auto getMIDTrackClustersROFRecords() const { return getSpan<o2::mid::ROFRecord>(GTrackID::MID, MATCHES); }
  auto getMIDTracksMCLabels() const { return getSpan<o2::MCCompLabel>(GTrackID::MID, MCLABELS); }
  const o2::dataformats::MCTruthContainer<o2::mid::MCClusterLabel>* getMIDTracksClusterMCLabels() const;

  // MID clusters
  auto getMIDClusterROFRecords() const { return getSpan<o2::mid::ROFRecord>(GTrackID::MID, CLUSREFS); }
  auto getMIDClusters() const { return getSpan<o2::mid::Cluster>(GTrackID::MID, CLUSTERS); }
  const o2::dataformats::MCTruthContainer<o2::mid::MCClusterLabel>* getMIDClustersMCLabels() const;

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

  // MFT-MCH
  const o2::dataformats::GlobalFwdTrack& getGlobalFwdTrack(GTrackID gid) const { return getTrack<o2::dataformats::GlobalFwdTrack>(gid); }
  auto getGlobalFwdTracks() const { return getTracks<o2::dataformats::GlobalFwdTrack>(GTrackID::MFTMCH); }
  auto getMFTMCHMatches() const { return getSpan<o2::dataformats::MatchInfoFwd>(GTrackID::MFTMCH, MATCHES); }
  auto getGlobalFwdTracksMCLabels() const { return getSpan<o2::MCCompLabel>(GTrackID::MFTMCH, MCLABELS); }

  // MCH-MID
  const o2::dataformats::TrackMCHMID& getMCHMIDMatch(GTrackID gid) const { return getObject<o2::dataformats::TrackMCHMID>(gid, MATCHES); }
  auto getMCHMIDMatches() const { return getSpan<o2::dataformats::TrackMCHMID>(GTrackID::MCHMID, MATCHES); }
  auto getMCHMIDMatchesMCLabels() const { return getSpan<o2::MCCompLabel>(GTrackID::MCHMID, MCLABELS); }

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
  auto getITSTPCTRDTriggers() const
  {
    return getSpan<o2::trd::TrackTriggerRecord>(GTrackID::ITSTPCTRD, TRACKREFS);
  }
  auto getITSTPCTRDTracksMCLabels() const { return getSpan<o2::MCCompLabel>(GTrackID::ITSTPCTRD, MCLABELS); }
  auto getITSTPCTRDTrackMCLabel(GTrackID id) const { return getObject<o2::MCCompLabel>(id, MCLABELS); }
  auto getITSTPCTRDSATracksMCLabels() const { return getSpan<o2::MCCompLabel>(GTrackID::ITSTPCTRD, MCLABELSEXTRA); }
  auto getITSTPCTRDSATrackMCLabel(GTrackID id) const { return getObject<o2::MCCompLabel>(id, MCLABELSEXTRA); }

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
  auto getTPCTRDTriggers() const
  {
    return getSpan<o2::trd::TrackTriggerRecord>(GTrackID::TPCTRD, TRACKREFS);
  }
  auto getTPCTRDTracksMCLabels() const { return getSpan<o2::MCCompLabel>(GTrackID::TPCTRD, MCLABELS); }
  auto getTPCTRDTrackMCLabel(GTrackID id) const { return getObject<o2::MCCompLabel>(id, MCLABELS); }
  auto getTPCTRDSATracksMCLabels() const { return getSpan<o2::MCCompLabel>(GTrackID::TPCTRD, MCLABELSEXTRA); }
  auto getTPCTRDSATrackMCLabel(GTrackID id) const { return getObject<o2::MCCompLabel>(id, MCLABELSEXTRA); }
  // TRD tracklets
  gsl::span<const o2::trd::Tracklet64> getTRDTracklets() const;
  gsl::span<const o2::trd::CalibratedTracklet> getTRDCalibratedTracklets() const;
  gsl::span<const o2::trd::TriggerRecord> getTRDTriggerRecords() const;
  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* getTRDTrackletsMCLabels() const;

  // TOF
  const o2::dataformats::MatchInfoTOF& getTOFMatch(GTrackID id) const { return getObject<o2::dataformats::MatchInfoTOF>(id, MATCHES); } // generic match getter
  // TPC-TOF, made of refitted TPC track and separate matchInfo
  const o2::dataformats::TrackTPCTOF& getTPCTOFTrack(GTrackID gid) const { return getTrack<o2::dataformats::TrackTPCTOF>(gid); }
  const o2::dataformats::MatchInfoTOF& getTPCTOFMatch(GTrackID id) const { return getObject<o2::dataformats::MatchInfoTOF>(id, MATCHES); }
  auto getTPCTOFTrackMCLabel(GTrackID id) const { return getObject<o2::MCCompLabel>(id, MCLABELS); }
  auto getTPCTOFTracks() const { return getTracks<o2::dataformats::TrackTPCTOF>(GTrackID::TPCTOF); }
  // TPC-TOF matches
  auto getTPCTOFMatches() const { return getSpan<o2::dataformats::MatchInfoTOF>(GTrackID::TPCTOF, MATCHES); }
  auto getTPCTOFTracksMCLabels() const { return getSpan<o2::MCCompLabel>(GTrackID::TPCTOF, MCLABELS); }
  // TPC-TRD-TOF matches
  auto getTPCTRDTOFMatches() const { return getSpan<o2::dataformats::MatchInfoTOF>(GTrackID::TPCTRDTOF, MATCHES); }
  auto getTPCTRDTOFTracksMCLabels() const { return getSpan<o2::MCCompLabel>(GTrackID::TPCTRDTOF, MCLABELS); }
  // global ITS-TPC-TOF matches
  const o2::dataformats::TrackTPCITS& getITSTPCTOFTrack(GTrackID id) const; // this is special since global TOF track is just a reference on TPCITS
  auto getITSTPCTOFMatches() const { return getSpan<o2::dataformats::MatchInfoTOF>(GTrackID::ITSTPCTOF, MATCHES); }
  auto getITSTPCTOFMatchesMCLabels() const { return getSpan<o2::MCCompLabel>(GTrackID::ITSTPCTOF, MCLABELS); }
  // global ITS-TPC-TRD-TOF matches
  //  const o2::dataformats::TrackTPCITS& getITSTPCTRDTOFTrack(GTrackID id) const; // TODO this is special since global TOF track is just a reference on TPCITS
  auto getITSTPCTRDTOFMatches() const { return getSpan<o2::dataformats::MatchInfoTOF>(GTrackID::ITSTPCTRDTOF, MATCHES); }
  auto getITSTPCTRDTOFMatchesMCLabels() const { return getSpan<o2::MCCompLabel>(GTrackID::ITSTPCTRDTOF, MCLABELS); }

  // HMPID matches
  auto getHMPMatches() const { return getSpan<o2::dataformats::MatchInfoHMP>(GTrackID::HMP, MATCHES); }
  auto getHMPMatchesMCLabels() const { return getSpan<o2::MCCompLabel>(GTrackID::HMP, MCLABELS); }

  // TOF clusters
  auto getTOFClusters() const { return getSpan<o2::tof::Cluster>(GTrackID::TOF, CLUSTERS); }
  auto getTOFClustersMCLabels() const { return mcTOFClusters.get(); }

  // HMPID clusters
  auto getHMPClusterTriggers() const { return getSpan<o2::hmpid::Trigger>(GTrackID::HMP, CLUSREFS); }
  auto getHMPClusters() const { return getSpan<o2::hmpid::Cluster>(GTrackID::HMP, CLUSTERS); }
  auto getHMPClustersMCLabels() const { return mcHMPClusters.get(); }

  // FT0
  auto getFT0RecPoints() const { return getSpan<o2::ft0::RecPoints>(GTrackID::FT0, TRACKS); }
  auto getFT0ChannelsData() const { return getSpan<o2::ft0::ChannelDataFloat>(GTrackID::FT0, CLUSTERS); }

  // FV0
  auto getFV0RecPoints() const { return getSpan<o2::fv0::RecPoints>(GTrackID::FV0, TRACKS); }
  auto getFV0ChannelsData() const { return getSpan<o2::fv0::ChannelDataFloat>(GTrackID::FV0, CLUSTERS); }

  // FDD
  auto getFDDRecPoints() const { return getSpan<o2::fdd::RecPoint>(GTrackID::FDD, TRACKS); }
  auto getFDDChannelsData() const { return getSpan<o2::fdd::ChannelDataFloat>(GTrackID::FDD, CLUSTERS); }

  // ZDC
  auto getZDCBCRecData() const { return getSpan<o2::zdc::BCRecData>(GTrackID::ZDC, MATCHES); }
  auto getZDCEnergy() const { return getSpan<o2::zdc::ZDCEnergy>(GTrackID::ZDC, TRACKS); }
  auto getZDCTDCData() const { return getSpan<o2::zdc::ZDCTDCData>(GTrackID::ZDC, CLUSTERS); }
  auto getZDCInfo() const { return getSpan<uint16_t>(GTrackID::ZDC, PATTERNS); }

  // CTP
  auto getCTPDigits() const { return getSpan<const o2::ctp::CTPDigit>(GTrackID::CTP, CLUSTERS); }
  const o2::ctp::LumiInfo& getCTPLumi() const { return mCTPLumi; }

  // CPV
  auto getCPVClusters() const { return getSpan<const o2::cpv::Cluster>(GTrackID::CPV, CLUSTERS); }
  auto getCPVTriggers() const { return getSpan<const o2::cpv::TriggerRecord>(GTrackID::CPV, CLUSREFS); }
  auto getCPVClustersMCLabels() const { return mcCPVClusters.get(); }

  // PHOS
  auto getPHOSCells() const { return getSpan<const o2::phos::Cell>(GTrackID::PHS, CLUSTERS); }
  auto getPHOSTriggers() const { return getSpan<const o2::phos::TriggerRecord>(GTrackID::PHS, CLUSREFS); }
  const o2::dataformats::MCTruthContainer<o2::phos::MCLabel>* getPHOSCellsMCLabels() const;

  // EMCAL
  auto getEMCALCells() const { return getSpan<const o2::emcal::Cell>(GTrackID::EMC, CLUSTERS); }
  auto getEMCALTriggers() const { return getSpan<const o2::emcal::TriggerRecord>(GTrackID::EMC, CLUSREFS); }
  const o2::dataformats::MCTruthContainer<o2::emcal::MCLabel>* getEMCALCellsMCLabels() const;

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
  auto getDecays3Body() const { return svtxPool.getSpan<o2::dataformats::DecayNbody>(DECAY3BODY); }
  auto getPV2Decays3BodyRefs() { return svtxPool.getSpan<o2::dataformats::RangeReference<int, int>>(PVTX_3BODYREFS); }

  // Strangeness track
  auto getStrangeTracks() const { return strkPool.getSpan<o2::dataformats::StrangeTrack>(STRACK); }
  auto getStrangeTracksMCLabels() const { return strkPool.getSpan<o2::MCCompLabel>(STRACK_MC); }
  const o2::dataformats::StrangeTrack& getStrangeTrack(int i) const { return strkPool.get_as<o2::dataformats::StrangeTrack>(STRACK, i); }

  // Cosmic tracks
  const o2::dataformats::TrackCosmics& getCosmicTrack(int i) const { return cosmPool.get_as<o2::dataformats::TrackCosmics>(COSM_TRACKS, i); }
  auto getCosmicTrackMCLabel(int i) const { return cosmPool.get_as<o2::MCCompLabel>(COSM_TRACKS_MC, i); }
  auto getCosmicTracks() const { return cosmPool.getSpan<o2::dataformats::TrackCosmics>(COSM_TRACKS); }
  auto getCosmicTrackMCLabels() const { return cosmPool.getSpan<o2::MCCompLabel>(COSM_TRACKS_MC); }

  // IRFrames where ITS was reconstructed and tracks were seen (e.g. sync.w-flow mult. selection)
  auto getIRFramesITS() const { return getSpan<o2::dataformats::IRFrame>(GTrackID::ITS, VARIA); }

  void getTrackTimeITSTPCTRDTOF(GTrackID gid, float& t, float& tErr) const;
  void getTrackTimeTPCTRDTOF(GTrackID gid, float& t, float& tErr) const;
  void getTrackTimeITSTPCTOF(GTrackID gid, float& t, float& tErr) const;
  void getTrackTimeITSTPCTRD(GTrackID gid, float& t, float& tErr) const;
  void getTrackTimeTPCTRD(GTrackID gid, float& t, float& tErr) const;
  void getTrackTimeITSTPC(GTrackID gid, float& t, float& tErr) const;
  void getTrackTimeTPCTOF(GTrackID gid, float& t, float& tErr) const;
  void getTrackTimeITS(GTrackID gid, float& t, float& tErr) const;
  void getTrackTimeTPC(GTrackID gid, float& t, float& tErr) const;

  void getTrackTime(GTrackID gid, float& t, float& tErr) const;
};

} // namespace globaltracking
} // namespace o2

#endif
