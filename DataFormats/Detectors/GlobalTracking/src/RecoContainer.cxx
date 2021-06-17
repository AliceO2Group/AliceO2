// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file RecoContainer.cxx
/// \brief Wrapper container for different reconstructed object types
/// \author ruben.shahoyan@cern.ch

#include <fmt/format.h>
#include <chrono>
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "CommonDataFormat/TimeStamp.h"
#include "CommonDataFormat/IRFrame.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "ReconstructionDataFormats/V0.h"
#include "ReconstructionDataFormats/Cascade.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "ReconstructionDataFormats/TrackCosmics.h"

using namespace o2::globaltracking;
using namespace o2::framework;
namespace o2d = o2::dataformats;

using GTrackID = o2d::GlobalTrackID;
using DetID = o2::detectors::DetID;

RecoContainer::RecoContainer() = default;
RecoContainer::~RecoContainer() = default;

void DataRequest::addInput(const InputSpec&& isp)
{
  if (std::find(inputs.begin(), inputs.end(), isp) == inputs.end()) {
    inputs.emplace_back(isp);
  }
}

void DataRequest::requestIRFramesITS()
{
  addInput({"IRFramesITS", "ITS", "IRFRAMES", 0, Lifetime::Timeframe});
  requestMap["IRFramesITS"] = false;
}

void DataRequest::requestITSTracks(bool mc)
{
  addInput({"trackITS", "ITS", "TRACKS", 0, Lifetime::Timeframe});
  addInput({"trackITSROF", "ITS", "ITSTrackROF", 0, Lifetime::Timeframe});
  addInput({"trackClIdx", "ITS", "TRACKCLSID", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"trackITSMCTR", "ITS", "TRACKSMCTR", 0, Lifetime::Timeframe});
  }
  requestMap["trackITS"] = mc;
}

void DataRequest::requestMFTTracks(bool mc)
{
  addInput({"trackMFT", "MFT", "TRACKS", 0, Lifetime::Timeframe});
  addInput({"trackMFTROF", "MFT", "MFTTrackROF", 0, Lifetime::Timeframe});
  addInput({"trackClIdx", "MFT", "TRACKCLSID", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"trackMFTMCTR", "MFT", "TRACKSMCTR", 0, Lifetime::Timeframe});
  }
  requestMap["trackMFT"] = mc;
}

void DataRequest::requestTPCTracks(bool mc)
{
  addInput({"trackTPC", "TPC", "TRACKS", 0, Lifetime::Timeframe});
  addInput({"trackTPCClRefs", "TPC", "CLUSREFS", 0, Lifetime::Timeframe});
  if (requestMap.find("clusTPC") != requestMap.end()) {
    addInput({"clusTPCshmap", "TPC", "CLSHAREDMAP", 0, Lifetime::Timeframe});
  }
  if (mc) {
    addInput({"trackTPCMCTR", "TPC", "TRACKSMCLBL", 0, Lifetime::Timeframe});
  }
  requestMap["trackTPC"] = mc;
}

void DataRequest::requestITSTPCTracks(bool mc)
{
  addInput({"trackITSTPC", "GLO", "TPCITS", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"trackITSTPCMCTR", "GLO", "TPCITS_MC", 0, Lifetime::Timeframe});
  }
  requestMap["trackITSTPC"] = mc;
}

void DataRequest::requestTPCTOFTracks(bool mc)
{
  addInput({"matchTPCTOF", "TOF", "MATCHINFOS_TPC", 0, Lifetime::Timeframe});
  addInput({"trackTPCTOF", "TOF", "TOFTRACKS_TPC", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"clsTOF_TPC_MCTR", "TOF", "MCMATCHTOF_TPC", 0, Lifetime::Timeframe});
  }
  requestMap["trackTPCTOF"] = mc;
}

void DataRequest::requestITSTPCTRDTracks(bool mc)
{
  addInput({"trackITSTPCTRD", "TRD", "MATCHTRD_GLO", 0, Lifetime::Timeframe});
  addInput({"trigITSTPCTRD", "TRD", "TRKTRG_GLO", 0, Lifetime::Timeframe});
  if (mc) {
    LOG(WARNING) << "TRD Tracks does not support MC truth, dummy label will be returned";
  }
  requestMap["trackITSTPCTRD"] = false;
}

void DataRequest::requestTPCTRDTracks(bool mc)
{
  addInput({"trackTPCTRD", "TRD", "MATCHTRD_TPC", 0, Lifetime::Timeframe});
  addInput({"trigTPCTRD", "TRD", "TRKTRG_TPC", 0, Lifetime::Timeframe});
  if (mc) {
    LOG(WARNING) << "TRD Tracks does not support MC truth, dummy label will be returned";
  }
  requestMap["trackTPCTRD"] = false;
}

void DataRequest::requestTOFMatches(bool mc)
{
  addInput({"matchITSTPCTOF", "TOF", "MATCHINFOS", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"clsTOF_GLO_MCTR", "TOF", "MCMATCHTOF", 0, Lifetime::Timeframe});
  }
  requestMap["matchTOF"] = mc;
}

void DataRequest::requestITSClusters(bool mc)
{
  addInput({"clusITS", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe});
  addInput({"clusITSPatt", "ITS", "PATTERNS", 0, Lifetime::Timeframe});
  addInput({"clusITSROF", "ITS", "CLUSTERSROF", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"clusITSMC", "ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe});
  }
  requestMap["clusITS"] = mc;
}

void DataRequest::requestTPCClusters(bool mc)
{
  addInput({"clusTPC", ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}, Lifetime::Timeframe});
  if (requestMap.find("trackTPC") != requestMap.end()) {
    addInput({"clusTPCshmap", "TPC", "CLSHAREDMAP", 0, Lifetime::Timeframe});
  }
  if (mc) {
    addInput({"clusTPCMC", ConcreteDataTypeMatcher{"TPC", "CLNATIVEMCLBL"}, Lifetime::Timeframe});
  }
  requestMap["clusTPC"] = mc;
}

void DataRequest::requestTOFClusters(bool mc)
{
  addInput({"tofcluster", "TOF", "CLUSTERS", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"tofclusterlabel", "TOF", "CLUSTERSMCTR", 0, Lifetime::Timeframe});
  }
  requestMap["clusTOF"] = mc;
}

void DataRequest::requestTRDTracklets(bool mc)
{
  addInput({"trdtracklets", o2::header::gDataOriginTRD, "TRACKLETS", 0, Lifetime::Timeframe});
  addInput({"trdctracklets", o2::header::gDataOriginTRD, "CTRACKLETS", 0, Lifetime::Timeframe});
  addInput({"trdtriggerrec", o2::header::gDataOriginTRD, "TRKTRGRD", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"trdtrackletlabels", o2::header::gDataOriginTRD, "TRKLABELS", 0, Lifetime::Timeframe});
  }
  requestMap["trackletTRD"] = mc;
}

void DataRequest::requestFT0RecPoints(bool mc)
{
  addInput({"ft0recpoints", "FT0", "RECPOINTS", 0, Lifetime::Timeframe});
  addInput({"ft0channels", "FT0", "RECCHDATA", 0, Lifetime::Timeframe});
  if (mc) {
    LOG(ERROR) << "FT0 RecPoint does not support MC truth";
  }
  requestMap["FT0"] = false;
}

void DataRequest::requestCoscmicTracks(bool mc)
{
  addInput({"cosmics", "GLO", "COSMICTRC", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"cosmicsMC", "GLO", "COSMICTRC_MC", 0, Lifetime::Timeframe});
  }
  requestMap["Cosmics"] = mc;
}

void DataRequest::requestPrimaryVertertices(bool mc)
{
  addInput({"pvtx", "GLO", "PVTX", 0, Lifetime::Timeframe});
  addInput({"pvtx_trmtc", "GLO", "PVTX_TRMTC", 0, Lifetime::Timeframe});    // global ids of associated tracks
  addInput({"pvtx_tref", "GLO", "PVTX_TRMTCREFS", 0, Lifetime::Timeframe}); // vertex - trackID refs
  if (mc) {
    addInput({"pvtx_mc", "GLO", "PVTX_MCTR", 0, Lifetime::Timeframe});
  }
  requestMap["PVertex"] = mc;
}

void DataRequest::requestPrimaryVerterticesTMP(bool mc) // primary vertices before global vertex-track matching
{
  addInput({"pvtx", "GLO", "PVTX", 0, Lifetime::Timeframe});
  addInput({"pvtx_cont", "GLO", "PVTX_CONTID", 0, Lifetime::Timeframe});        // global ids of contributors
  addInput({"pvtx_contref", "GLO", "PVTX_CONTIDREFS", 0, Lifetime::Timeframe}); // vertex - trackID refs of contributors
  if (mc) {
    addInput({"pvtx_mc", "GLO", "PVTX_MCTR", 0, Lifetime::Timeframe});
  }
  requestMap["PVertexTMP"] = mc;
}

void DataRequest::requestSecondaryVertertices(bool)
{
  addInput({"v0s", "GLO", "V0s", 0, Lifetime::Timeframe});
  addInput({"p2v0s", "GLO", "PVTX_V0REFS", 0, Lifetime::Timeframe});
  addInput({"cascs", "GLO", "CASCS", 0, Lifetime::Timeframe});
  addInput({"p2cascs", "GLO", "PVTX_CASCREFS", 0, Lifetime::Timeframe});
  requestMap["SVertex"] = false; // no MC provided for secondary vertices
}

void DataRequest::requestTracks(GTrackID::mask_t src, bool useMC)
{
  // request tracks for sources probided by the mask
  if (src[GTrackID::ITS]) {
    requestITSTracks(useMC);
  }
  if (src[GTrackID::MFT]) {
    requestMFTTracks(useMC);
  }
  if (src[GTrackID::TPC]) {
    requestTPCTracks(useMC);
  }
  if (src[GTrackID::ITSTPC] || src[GTrackID::ITSTPCTOF]) {
    requestITSTPCTracks(useMC);
  }
  if (src[GTrackID::TPCTOF]) {
    requestTPCTOFTracks(useMC);
  }
  if (src[GTrackID::ITSTPCTOF]) {
    requestTOFMatches(useMC);
    requestTOFClusters(false); // RSTODO Needed just to set the time of ITSTPC track, consider moving to MatchInfoTOF
                               // NOTE: Getting TOF Clusters is carried over to InputHelper::addInputSpecs. If changed here, please fix there.
  }
  if (src[GTrackID::ITSTPCTRD]) {
    requestITSTPCTRDTracks(useMC);
  }
  if (src[GTrackID::TPCTRD]) {
    requestTPCTRDTracks(useMC);
  }
}

void DataRequest::requestClusters(GTrackID::mask_t src, bool useMC)
{
  // request clusters for detectors of the sources probided by the mask

  // clusters needed for refits
  if (GTrackID::includesDet(DetID::ITS, src)) {
    requestITSClusters(useMC);
  }
  if (GTrackID::includesDet(DetID::TPC, src)) {
    requestTPCClusters(useMC);
  }
  if (GTrackID::includesDet(DetID::TOF, src)) {
    requestTOFClusters(useMC);
  }
  if (GTrackID::includesDet(DetID::TRD, src)) {
    requestTRDTracklets(useMC);
  }
}

//__________________________________________________________________
void RecoContainer::collectData(ProcessingContext& pc, const DataRequest& requests)
{
  auto& reqMap = requests.requestMap;

  const auto* dh = o2::header::get<o2::header::DataHeader*>(pc.inputs().getByPos(0).header);
  startIR = {0, dh->firstTForbit};

  auto req = reqMap.find("trackITS");
  if (req != reqMap.end()) {
    addITSTracks(pc, req->second);
  }

  req = reqMap.find("trackMFT");
  if (req != reqMap.end()) {
    addMFTTracks(pc, req->second);
  }

  req = reqMap.find("trackTPC");
  if (req != reqMap.end()) {
    addTPCTracks(pc, req->second);
  }

  req = reqMap.find("trackITSTPC");
  if (req != reqMap.end()) {
    addITSTPCTracks(pc, req->second);
  }

  req = reqMap.find("trackITSTPCTRD");
  if (req != reqMap.end()) {
    addITSTPCTRDTracks(pc, req->second);
  }

  req = reqMap.find("trackTPCTRD");
  if (req != reqMap.end()) {
    addTPCTRDTracks(pc, req->second);
  }

  req = reqMap.find("trackTPCTOF");
  if (req != reqMap.end()) {
    addTPCTOFTracks(pc, req->second);
  }

  req = reqMap.find("matchTOF");
  if (req != reqMap.end()) {
    addTOFMatches(pc, req->second);
  }

  req = reqMap.find("clusITS");
  if (req != reqMap.end()) {
    addITSClusters(pc, req->second);
  }

  req = reqMap.find("clusTPC");
  if (req != reqMap.end()) {
    addTPCClusters(pc, req->second, reqMap.find("trackTPC") != reqMap.end());
  }

  req = reqMap.find("clusTOF");
  if (req != reqMap.end()) {
    addTOFClusters(pc, req->second);
  }

  req = reqMap.find("FT0");
  if (req != reqMap.end()) {
    addFT0RecPoints(pc, req->second);
  }

  req = reqMap.find("trackletTRD");
  if (req != reqMap.end()) {
    addTRDTracklets(pc);
  }

  req = reqMap.find("Cosmics");
  if (req != reqMap.end()) {
    addCosmicTracks(pc, req->second);
  }

  req = reqMap.find("PVertex");
  if (req != reqMap.end()) {
    addPVertices(pc, req->second);
  }

  req = reqMap.find("PVertexTMP");
  if (req != reqMap.end()) {
    addPVerticesTMP(pc, req->second);
  }

  req = reqMap.find("SVertex");
  if (req != reqMap.end()) {
    addSVertices(pc, req->second);
  }

  req = reqMap.find("IRFramesITS");
  if (req != reqMap.end()) {
    addIRFramesITS(pc);
  }
}

//____________________________________________________________
void RecoContainer::addSVertices(ProcessingContext& pc, bool)
{
  svtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::V0>>("v0s"), V0S);
  svtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::RangeReference<int, int>>>("p2v0s"), PVTX_V0REFS);
  svtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::Cascade>>("cascs"), CASCS);
  svtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::RangeReference<int, int>>>("p2cascs"), PVTX_CASCREFS);
  // no mc
}

//____________________________________________________________
void RecoContainer::addPVertices(ProcessingContext& pc, bool mc)
{
  if (!pvtxPool.isLoaded(PVTX)) { // in case was loaded via addPVerticesTMP
    pvtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::PrimaryVertex>>("pvtx"), PVTX);
  }
  pvtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::VtxTrackIndex>>("pvtx_trmtc"), PVTX_TRMTC);
  pvtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::VtxTrackRef>>("pvtx_tref"), PVTX_TRMTCREFS);

  if (mc && !pvtxPool.isLoaded(PVTX_MCTR)) { // in case was loaded via addPVerticesTMP
    pvtxPool.registerContainer(pc.inputs().get<gsl::span<o2::MCEventLabel>>("pvtx_mc"), PVTX_MCTR);
  }
}

//____________________________________________________________
void RecoContainer::addPVerticesTMP(ProcessingContext& pc, bool mc)
{
  if (!pvtxPool.isLoaded(PVTX)) { // in case was loaded via addPVertices
    pvtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::PrimaryVertex>>("pvtx"), PVTX);
  }
  pvtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::VtxTrackIndex>>("pvtx_cont"), PVTX_CONTID);
  pvtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::VtxTrackRef>>("pvtx_contref"), PVTX_CONTIDREFS);

  if (mc && !pvtxPool.isLoaded(PVTX_MCTR)) { // in case was loaded via addPVertices
    pvtxPool.registerContainer(pc.inputs().get<gsl::span<o2::MCEventLabel>>("pvtx_mc"), PVTX_MCTR);
  }
}

//____________________________________________________________
void RecoContainer::addCosmicTracks(ProcessingContext& pc, bool mc)
{
  cosmPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::TrackCosmics>>("cosmics"), COSM_TRACKS);
  if (mc) {
    cosmPool.registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("cosmicsMC"), COSM_TRACKS_MC);
  }
}

//____________________________________________________________
void RecoContainer::addITSTracks(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::ITS].registerContainer(pc.inputs().get<gsl::span<o2::its::TrackITS>>("trackITS"), TRACKS);
  commonPool[GTrackID::ITS].registerContainer(pc.inputs().get<gsl::span<int>>("trackClIdx"), INDICES);
  commonPool[GTrackID::ITS].registerContainer(pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("trackITSROF"), TRACKREFS);
  if (mc) {
    commonPool[GTrackID::ITS].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackITSMCTR"), MCLABELS);
  }
}

//____________________________________________________________
void RecoContainer::addIRFramesITS(ProcessingContext& pc)
{
  commonPool[GTrackID::ITS].registerContainer(pc.inputs().get<gsl::span<o2::dataformats::IRFrame>>("IRFramesITS"), VARIA);
}

//____________________________________________________________
void RecoContainer::addMFTTracks(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::MFT].registerContainer(pc.inputs().get<gsl::span<o2::mft::TrackMFT>>("trackMFT"), TRACKS);
  commonPool[GTrackID::MFT].registerContainer(pc.inputs().get<gsl::span<int>>("trackClIdx"), INDICES);
  commonPool[GTrackID::MFT].registerContainer(pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("trackMFTROF"), TRACKREFS);
  if (mc) {
    commonPool[GTrackID::MFT].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackMFTMCTR"), MCLABELS);
  }
}

//____________________________________________________________
void RecoContainer::addTPCTracks(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::TPC].registerContainer(pc.inputs().get<gsl::span<o2::tpc::TrackTPC>>("trackTPC"), TRACKS);
  commonPool[GTrackID::TPC].registerContainer(pc.inputs().get<gsl::span<o2::tpc::TPCClRefElem>>("trackTPCClRefs"), INDICES);
  if (mc) {
    commonPool[GTrackID::TPC].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackTPCMCTR"), MCLABELS);
  }
}

//__________________________________________________________
void RecoContainer::addITSTPCTracks(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::ITSTPC].registerContainer(pc.inputs().get<gsl::span<o2d::TrackTPCITS>>("trackITSTPC"), TRACKS);
  if (mc) {
    commonPool[GTrackID::ITSTPC].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackITSTPCMCTR"), MCLABELS);
  }
}

//__________________________________________________________
void RecoContainer::addITSTPCTRDTracks(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::ITSTPCTRD].registerContainer(pc.inputs().get<gsl::span<o2::trd::TrackTRD>>("trackITSTPCTRD"), TRACKS);
  commonPool[GTrackID::ITSTPCTRD].registerContainer(pc.inputs().get<gsl::span<o2::trd::TrackTriggerRecord>>("trigITSTPCTRD"), TRACKREFS);
}

//__________________________________________________________
void RecoContainer::addTPCTRDTracks(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::TPCTRD].registerContainer(pc.inputs().get<gsl::span<o2::trd::TrackTRD>>("trackTPCTRD"), TRACKS);
  commonPool[GTrackID::TPCTRD].registerContainer(pc.inputs().get<gsl::span<o2::trd::TrackTriggerRecord>>("trigTPCTRD"), TRACKREFS);
}

//__________________________________________________________
void RecoContainer::addTPCTOFTracks(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::TPCTOF].registerContainer(pc.inputs().get<gsl::span<o2d::TrackTPCTOF>>("trackTPCTOF"), TRACKS);
  commonPool[GTrackID::TPCTOF].registerContainer(pc.inputs().get<gsl::span<o2d::MatchInfoTOF>>("matchTPCTOF"), MATCHES);
  if (mc) {
    commonPool[GTrackID::TPCTOF].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("clsTOF_TPC_MCTR"), MCLABELS);
  }
}

//__________________________________________________________
void RecoContainer::addTOFMatches(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::ITSTPCTOF].registerContainer(pc.inputs().get<gsl::span<o2d::MatchInfoTOF>>("matchITSTPCTOF"), MATCHES); //only ITS/TPC : TOF match info, no real tracks
  if (mc) {
    commonPool[GTrackID::ITSTPCTOF].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("clsTOF_GLO_MCTR"), MCLABELS);
  }
}

//__________________________________________________________
void RecoContainer::addITSClusters(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::ITS].registerContainer(pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("clusITSROF"), CLUSREFS);
  commonPool[GTrackID::ITS].registerContainer(pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("clusITS"), CLUSTERS);
  commonPool[GTrackID::ITS].registerContainer(pc.inputs().get<gsl::span<unsigned char>>("clusITSPatt"), PATTERNS);
  if (mc) {
    mcITSClusters = pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("clusITSMC");
  }
}

//__________________________________________________________
void RecoContainer::addTPCClusters(ProcessingContext& pc, bool mc, bool shmap)
{
  inputsTPCclusters = o2::tpc::getWorkflowTPCInput(pc, 0, mc);
  if (shmap) {
    clusterShMapTPC = pc.inputs().get<gsl::span<unsigned char>>("clusTPCshmap");
  }
}

//__________________________________________________________
void RecoContainer::addTRDTracklets(ProcessingContext& pc)
{
  inputsTRD = o2::trd::getRecoInputContainer(pc, nullptr, this);
}

//__________________________________________________________
void RecoContainer::addTOFClusters(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::TOF].registerContainer(pc.inputs().get<gsl::span<o2::tof::Cluster>>("tofcluster"), CLUSTERS);
  if (mc) {
    mcTOFClusters = pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("tofclusterlabel");
  }
}

//__________________________________________________________
void RecoContainer::addFT0RecPoints(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::FT0].registerContainer(pc.inputs().get<gsl::span<o2::ft0::RecPoints>>("ft0recpoints"), TRACKS);
  commonPool[GTrackID::FT0].registerContainer(pc.inputs().get<gsl::span<o2::ft0::ChannelDataFloat>>("ft0channels"), CLUSTERS);

  if (mc) {
    LOG(ERROR) << "FT0 RecPoint does not support MC truth";
  }
}

const o2::tpc::ClusterNativeAccess& RecoContainer::getTPCClusters() const
{
  return inputsTPCclusters->clusterIndex;
}

gsl::span<const o2::trd::Tracklet64> RecoContainer::getTRDTracklets() const
{
  return inputsTRD->mTracklets;
}

gsl::span<const o2::trd::CalibratedTracklet> RecoContainer::getTRDCalibratedTracklets() const
{
  return inputsTRD->mSpacePoints;
}

gsl::span<const o2::trd::TriggerRecord> RecoContainer::getTRDTriggerRecords() const
{
  return inputsTRD->mTriggerRecords;
}

const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* RecoContainer::getTRDTrackletsMCLabels() const
{
  return inputsTRD->mTrackletLabels.get();
}

const o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>* RecoContainer::getTPCClustersMCLabels() const
{
  return inputsTPCclusters->clusterIndex.clustersMCTruth;
}

//__________________________________________________________
const o2::track::TrackParCov& RecoContainer::getTrackParamOut(GTrackID gidx) const
{
  // get outer param of track
  auto trSrc = gidx.getSource();
  if (trSrc == GTrackID::ITSTPC) {
    return getTrack<o2d::TrackTPCITS>(gidx).getParamOut();
  } else if (trSrc == GTrackID::ITSTPCTOF) { // the physical tracks are in ITS-TPC, need to get reference from match info
    return getTrack<o2d::TrackTPCITS>(getTOFMatch(gidx).getEvIdxTrack().getIndex()).getParamOut();
  } else if (trSrc == GTrackID::TPCTOF) {
    return getTrack<o2d::TrackTPCTOF>(gidx).getParamOut();
  } else if (trSrc == GTrackID::ITS) {
    return getTrack<o2::its::TrackITS>(gidx).getParamOut();
  } else if (trSrc == GTrackID::TPC) {
    return getTrack<o2::tpc::TrackTPC>(gidx).getParamOut();
  } else {
    throw std::runtime_error(fmt::format("not defined for tracks of source {:d}", int(trSrc)));
  }
}

//__________________________________________________________
bool RecoContainer::isTrackSourceLoaded(int src) const
{
  if (src == GTrackID::ITSTPCTOF) {
    if (!isMatchSourceLoaded(src)) { // the physical tracks are in ITS-TPC, need to get reference from match info
      return false;
    }
    src = GTrackID::ITSTPC;
  }
  return commonPool[src].isLoaded(TRACKS);
}

//__________________________________________________________
const o2::track::TrackParCov& RecoContainer::getTrackParam(GTrackID gidx) const
{
  // get base track
  auto trSrc = gidx.getSource();
  if (trSrc == GTrackID::ITSTPCTOF) { // the physical tracks are in ITS-TPC, need to get reference from match info
    gidx = getTOFMatch(gidx).getEvIdxTrack().getIndex();
  }
  return getObject<o2::track::TrackParCov>(gidx, TRACKS);
}

//__________________________________________________________
const o2::dataformats::TrackTPCITS& RecoContainer::getITSTPCTOFTrack(GTrackID gidx) const
{
  // get ITS-TPC track pointed by global TOF match
  return getTPCITSTrack(getTOFMatch(gidx).getEvIdxTrack().getIndex());
}

//________________________________________________________
void RecoContainer::fillTrackMCLabels(const gsl::span<GTrackID> gids, std::vector<o2::MCCompLabel>& mcinfo) const
{
  // fills the MCLabels corresponding to gids to MC info
  mcinfo.clear();
  mcinfo.reserve(gids.size());
  for (auto gid : gids) {
    mcinfo.push_back(getTrackMCLabel(gid));
  }
}

void o2::globaltracking::RecoContainer::createTracks(std::function<bool(const o2::track::TrackParCov&, o2::dataformats::GlobalTrackID)> const& creator) const
{
  createTracksVariadic([&creator](const auto& _tr, GTrackID _origID, float t0, float terr) {
    if constexpr (std::is_base_of_v<o2::track::TrackParCov, std::decay_t<decltype(_tr)>>) {
      return creator(_tr, _origID);
    } else {
      return false;
    }
  });
}

//________________________________________________________
// get contributors from single detectors
RecoContainer::GlobalIDSet RecoContainer::getSingleDetectorRefs(GTrackID gidx) const
{
  GlobalIDSet table;
  auto src = gidx.getSource();
  table[src] = gidx;
  if (src == GTrackID::ITSTPCTOF) {
    const auto& parent0 = getTOFMatch(gidx); //ITS/TPC : TOF
    const auto& parent1 = getTPCITSTrack(parent0.getEvIdxTrack().getIndex());
    table[GTrackID::ITSTPC] = parent0.getEvIdxTrack().getIndex();
    table[GTrackID::TOF] = {unsigned(parent0.getEvIdxTOFCl().getIndex()), GTrackID::TOF};
    table[GTrackID::ITS] = parent1.getRefITS();
    table[GTrackID::TPC] = parent1.getRefTPC();
  } else if (src == GTrackID::TPCTOF) {
    const auto& parent0 = getTPCTOFMatch(gidx); //TPC : TOF
    table[GTrackID::TOF] = {unsigned(parent0.getEvIdxTOFCl().getIndex()), GTrackID::TOF};
    table[GTrackID::TPC] = parent0.getEvIdxTrack().getIndex();
  } else if (src == GTrackID::ITSTPC) {
    const auto& parent0 = getTPCITSTrack(gidx);
    table[GTrackID::ITS] = parent0.getRefITS();
    table[GTrackID::TPC] = parent0.getRefTPC();
  }
  return std::move(table);
}

// get contributing TPC GTrackID to the source. If source gidx is not contributed by TPC,
// returned GTrackID.isSourceSet()==false
GTrackID RecoContainer::getTPCContributorGID(GTrackID gidx) const
{
  auto src = gidx.getSource();
  if (src == GTrackID::ITSTPCTOF) {
    const auto& parent0 = getTOFMatch(gidx); //ITS/TPC : TOF
    const auto& parent1 = getTPCITSTrack(parent0.getEvIdxTrack().getIndex());
    return parent1.getRefTPC();
  } else if (src == GTrackID::TPCTOF) {
    const auto& parent0 = getTPCTOFMatch(gidx); //TPC : TOF
    return parent0.getEvIdxTrack().getIndex();
  } else if (src == GTrackID::ITSTPC) {
    const auto& parent0 = getTPCITSTrack(gidx);
    return parent0.getRefTPC();
  }
  return src == GTrackID::TPC ? gidx : GTrackID{};
}

// get contributing ITS GTrackID to the source. If source gidx is not contributed by TPC,
// returned GTrackID.isSourceSet()==false
GTrackID RecoContainer::getITSContributorGID(GTrackID gidx) const
{
  auto src = gidx.getSource();
  if (src == GTrackID::ITSTPCTOF) {
    const auto& parent0 = getTOFMatch(gidx); //ITS/TPC : TOF
    const auto& parent1 = getTPCITSTrack(parent0.getEvIdxTrack().getIndex());
    return parent1.getRefITS();
  } else if (src == GTrackID::ITSTPC) {
    const auto& parent0 = getTPCITSTrack(gidx);
    return parent0.getRefITS();
  }
  return src == GTrackID::ITS ? gidx : GTrackID{};
}
