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
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsTRD/RecoInputContainer.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTOF/Cluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsFT0/RecPoints.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/TrackTPCTOF.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"

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

void DataRequest::requestTPCTracks(bool mc)
{
  addInput({"trackTPC", "TPC", "TRACKS", 0, Lifetime::Timeframe});
  addInput({"trackTPCClRefs", "TPC", "CLUSREFS", 0, Lifetime::Timeframe});
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
  addInput({"clusTPCshmap", "TPC", "CLSHAREDMAP", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"clusITSMC", "ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe});
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

void DataRequest::requestTRDTracklets()
{
  addInput({"trdtracklets", o2::header::gDataOriginTRD, "TRACKLETS", 0, Lifetime::Timeframe});
  addInput({"trdctracklets", o2::header::gDataOriginTRD, "CTRACKLETS", 0, Lifetime::Timeframe});
  addInput({"trdtriggerrec", o2::header::gDataOriginTRD, "TRKTRGRD", 0, Lifetime::Timeframe});
}

void DataRequest::requestFT0RecPoints(bool mc)
{
  addInput({"ft0recpoints", "FT0", "RECPOINTS", 0, Lifetime::Timeframe});
  if (mc) {
    LOG(ERROR) << "FT0 RecPoint does not support MC truth";
  }
  requestMap["FT0"] = false;
}

void DataRequest::requestTracks(GTrackID::mask_t src, bool useMC)
{
  // request tracks for sources probided by the mask
  if (src[GTrackID::ITS]) {
    requestITSTracks(useMC);
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
    requestTRDTracklets();
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

  req = reqMap.find("trackTPC");
  if (req != reqMap.end()) {
    addTPCTracks(pc, req->second);
  }

  req = reqMap.find("trackITSTPC");
  if (req != reqMap.end()) {
    addITSTPCTracks(pc, req->second);
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
    addTPCClusters(pc, req->second);
  }

  req = reqMap.find("clusTOF");
  if (req != reqMap.end()) {
    addTOFClusters(pc, req->second);
  }

  req = reqMap.find("FT0");
  if (req != reqMap.end()) {
    addFT0RecPoints(pc, req->second);
  }
}

//____________________________________________________________
void RecoContainer::addITSTracks(ProcessingContext& pc, bool mc)
{
  tracksPool.registerContainer(pc.inputs().get<gsl::span<o2::its::TrackITS>>("trackITS"), GTrackID::ITS);
  clusRefPool.registerContainer(pc.inputs().get<gsl::span<int>>("trackClIdx"), GTrackID::ITS);
  tracksROFsPool.registerContainer(pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("trackITSROF"), GTrackID::ITS);
  if (mc) {
    tracksMCPool.registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackITSMCTR"), GTrackID::ITS);
  }
}

//____________________________________________________________
void RecoContainer::addTPCTracks(ProcessingContext& pc, bool mc)
{
  tracksPool.registerContainer(pc.inputs().get<gsl::span<o2::tpc::TrackTPC>>("trackTPC"), GTrackID::TPC);
  clusRefPool.registerContainer(pc.inputs().get<gsl::span<o2::tpc::TPCClRefElem>>("trackTPCClRefs"), GTrackID::TPC);
  if (mc) {
    tracksMCPool.registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackTPCMCTR"), GTrackID::TPC);
  }
}

//__________________________________________________________
void RecoContainer::addITSTPCTracks(ProcessingContext& pc, bool mc)
{
  tracksPool.registerContainer(pc.inputs().get<gsl::span<o2d::TrackTPCITS>>("trackITSTPC"), GTrackID::ITSTPC);
  if (mc) {
    tracksMCPool.registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackITSTPCMCTR"), GTrackID::ITSTPC);
  }
}

//__________________________________________________________
void RecoContainer::addTPCTOFTracks(ProcessingContext& pc, bool mc)
{
  tracksPool.registerContainer(pc.inputs().get<gsl::span<o2d::TrackTPCTOF>>("trackTPCTOF"), GTrackID::TPCTOF);
  miscPool.registerContainer(pc.inputs().get<gsl::span<o2d::MatchInfoTOF>>("matchTPCTOF"), GTrackID::TPCTOF);
  if (mc) {
    tracksMCPool.registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("clsTOF_TPC_MCTR"), GTrackID::TPCTOF);
  }
}

//__________________________________________________________
void RecoContainer::addTOFMatches(ProcessingContext& pc, bool mc)
{
  miscPool.registerContainer(pc.inputs().get<gsl::span<o2d::MatchInfoTOF>>("matchITSTPCTOF"), GTrackID::ITSTPCTOF); //only ITS/TPC : TOF match info, no real tracks
  if (mc) {
    tracksMCPool.registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("clsTOF_GLO_MCTR"), GTrackID::ITSTPCTOF);
  }
}

//__________________________________________________________
void RecoContainer::addITSClusters(ProcessingContext& pc, bool mc)
{
  clusROFPool.registerContainer(pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("clusITSROF"), GTrackID::ITS);
  clustersPool.registerContainer(pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("clusITS"), GTrackID::ITS);
  miscPool.registerContainer(pc.inputs().get<gsl::span<unsigned char>>("clusITSPatt"), GTrackID::ITS);
  if (mc) {
    mcITSClusters = pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("labels");
  }
}

//__________________________________________________________
void RecoContainer::addTPCClusters(ProcessingContext& pc, bool mc)
{
  inputsTPCclusters = o2::tpc::getWorkflowTPCInput(pc, 0, mc);
  clusterShMapTPC = pc.inputs().get<gsl::span<unsigned char>>("clusTPCshmap");
}

void RecoContainer::addTRDTracklets(ProcessingContext& pc)
{
  inputsTRD = o2::trd::getRecoInputContainer(pc, nullptr, this);
}

//__________________________________________________________
void RecoContainer::addTOFClusters(ProcessingContext& pc, bool mc)
{
  clustersPool.registerContainer(pc.inputs().get<gsl::span<o2::tof::Cluster>>("tofcluster"), GTrackID::TOF);
  if (mc) {
    mcTOFClusters = pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("tofclusterlabel");
  }
}

//__________________________________________________________
void RecoContainer::addFT0RecPoints(ProcessingContext& pc, bool mc)
{
  miscPool.registerContainer(pc.inputs().get<gsl::span<o2::ft0::RecPoints>>("ft0recpoints"), GTrackID::FT0);
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

//__________________________________________________________
const o2::track::TrackParCov& RecoContainer::getTrackParamOut(GTrackID gidx) const
{
  // get outer param of track
  auto trSrc = gidx.getSource();
  if (trSrc == GTrackID::ITSTPC) {
    return tracksPool.get_as<o2d::TrackTPCITS>(gidx).getParamOut();
  } else if (trSrc == GTrackID::ITSTPCTOF) { // the physical tracks are in ITS-TPC, need to get reference from match info
    return tracksPool.get_as<o2d::TrackTPCITS>(getTOFMatch<o2d::MatchInfoTOF>(gidx).getEvIdxTrack().getIndex()).getParamOut();
  } else if (trSrc == GTrackID::TPCTOF) {
    return tracksPool.get_as<o2d::TrackTPCTOF>(gidx).getParamOut();
  } else if (trSrc == GTrackID::ITS) {
    return tracksPool.get_as<o2::its::TrackITS>(gidx).getParamOut();
  } else if (trSrc == GTrackID::TPC) {
    return tracksPool.get_as<o2::tpc::TrackTPC>(gidx).getParamOut();
  } else {
    throw std::runtime_error(fmt::format("not defined for tracks of source {:d}", int(trSrc)));
  }
}

//__________________________________________________________
bool RecoContainer::isTrackSourceLoaded(int src) const
{
  if (src == GTrackID::ITSTPCTOF && isMatchSourceLoaded(src)) { // the physical tracks are in ITS-TPC, need to get reference from match info
    src = GTrackID::ITSTPC;
  }
  return tracksPool.isLoaded(src);
}

//__________________________________________________________
const o2::track::TrackParCov& RecoContainer::getTrack(GTrackID gidx) const
{
  // get base track
  auto trSrc = gidx.getSource();
  if (trSrc == GTrackID::ITSTPCTOF) { // the physical tracks are in ITS-TPC, need to get reference from match info
    gidx = getTOFMatch<o2d::MatchInfoTOF>(gidx).getEvIdxTrack().getIndex();
  }
  return tracksPool.get(gidx);
}

//________________________________________________________
void RecoContainer::fillTrackMCLabels(const gsl::span<GTrackID> gids, std::vector<o2::MCCompLabel>& mcinfo) const
{
  // fills the MCLabels corresponding to gids to MC info
  mcinfo.clear();
  mcinfo.reserve(gids.size());
  for (auto gid : gids) {
    mcinfo.push_back(tracksMCPool.get(gid));
  }
}

//________________________________________________________
void RecoContainer::createTracks(std::function<bool(const o2::track::TrackParCov&, float, float, GTrackID)> const& creator) const
{
  // We go from most complete tracks to least complete ones, taking into account that some track times
  // do not bear their own kinematics but just constrain the time
  // As we get more track types functional, this method should be completed
  // If user provided function creator returns true, then the track is considered as consumed and its contributing
  // simpler tracks will not be provided to the creator. If it returns false, the creator will be called also
  // with this simpler contrubutors.
  // The creator function is called with track kinematics, track GlobalTrackID and track timing information as 2 floats
  // which depends on the track time:
  // 1) For track types containing TimeStampWithError ts it is ts.getTimeStamp(), getTimeStampError()
  // 2) For tracks with asymmetric time uncertainty, e.g. TPC: as mean time of t0-errBwd,t+errFwd and 0.5(errBwd+errFwd), all in TPC time bins
  // 3) For tracks whose timing is provided as RO frame: as time in \mus for RO frame start since the start of TF, half-duration of RO window and 0.

  auto start_time = std::chrono::high_resolution_clock::now();
  constexpr float PS2MUS = 1e-6;
  std::array<std::vector<uint8_t>, GTrackID::NSources> usedData;
  auto flagUsed2 = [&usedData](int idx, int src) {
    if (!usedData[src].empty()) {
      usedData[src][idx] = 1;
    }
  };
  auto flagUsed = [&usedData, &flagUsed2](const GTrackID gidx) { flagUsed2(gidx.getIndex(), gidx.getSource()); };
  auto isUsed2 = [&usedData](int idx, int src) { return (!usedData[src].empty()) && (usedData[src][idx] != 0); };
  auto isUsed = [&usedData, isUsed2](const GTrackID gidx) { return isUsed2(gidx.getIndex(), gidx.getSource()); };

  // create only for those data types which are used
  const auto& tracksITS = getITSTracks<o2::its::TrackITS>();
  const auto& tracksTPC = getTPCTracks<o2::tpc::TrackTPC>();
  const auto& tracksTPCITS = getTPCITSTracks<o2d::TrackTPCITS>();
  const auto& tracksTPCTOF = getTPCTOFTracks<o2d::TrackTPCTOF>();
  const auto& matchesTPCTOF = getTPCTOFMatches<o2d::MatchInfoTOF>();

  usedData[GTrackID::ITS].resize(tracksITS.size());       // to flag used ITS tracks
  usedData[GTrackID::TPC].resize(tracksTPC.size());       // to flag used TPC tracks
  usedData[GTrackID::ITSTPC].resize(tracksTPCITS.size()); // to flag used ITSTPC tracks
  usedData[GTrackID::TOF].resize(getTOFMatches<o2d::MatchInfoTOF>().size()); // to flag used ITSTPC-TOF matches

  // ITS-TPC-TOF matches, may refer to ITS-TPC (TODO: something else?) tracks
  {
    auto matches = getTOFMatches<o2d::MatchInfoTOF>(); // thes are just MatchInfoTOF objects, pointing on ITS-TPC match and TOF cl.
    auto tofClusters = getTOFClusters<o2::tof::Cluster>();
    if (matches.size() && (!tofClusters.size() || !tracksTPCITS.size())) {
      throw std::runtime_error(fmt::format("Global-TOF tracks ({}) require ITS-TPC tracks ({}) and TOF clusters ({})",
                                           matches.size(), tracksTPCITS.size(), tofClusters.size()));
    }
    for (unsigned i = 0; i < matches.size(); i++) {
      const auto& match = matches[i];
      const auto& tofCl = tofClusters[match.getTOFClIndex()];
      float timeTOFMUS = (tofCl.getTime() - match.getLTIntegralOut().getTOF(o2::track::PID::Pion)) * PS2MUS; // tof time in \mus, FIXME: account for time of flight to R TOF
      const float timeErr = 0.010f;                                                                          // assume 10 ns error FIXME

      auto gidx = match.getEvIdxTrack().getIndex(); // this should be corresponding ITS-TPC track
      if (creator(tracksPool.get(gidx), timeTOFMUS, timeErr, {i, GTrackID::ITSTPCTOF})) {
        flagUsed2(i, GTrackID::TOF);
        flagUsed(gidx); // flag used ITS-TPC tracks
      }
    }
  }

  // ITS-TPC matches, may refer to ITS, TPC (TODO: something else?) tracks
  {
    for (unsigned i = 0; i < tracksTPCITS.size(); i++) {
      const auto& matchTr = tracksTPCITS[i];
      if (isUsed2(i, GTrackID::ITSTPC)) {
        flagUsed(matchTr.getRefITS()); // flag used ITS tracks
        flagUsed(matchTr.getRefTPC()); // flag used TPC tracks
        continue;
      }
      if (creator(matchTr, matchTr.getTimeMUS().getTimeStamp(), matchTr.getTimeMUS().getTimeStampError(), {i, GTrackID::ITSTPC})) {
        flagUsed2(i, GTrackID::ITSTPC);
        flagUsed(matchTr.getRefITS()); // flag used ITS tracks
        flagUsed(matchTr.getRefTPC()); // flag used TPC tracks
      }
    }
  }

  // TPC-TOF matches, may refer to TPC (TODO: something else?) tracks
  {
    if (matchesTPCTOF.size() && !tracksTPCTOF.size()) {
      throw std::runtime_error(fmt::format("TPC-TOF matched tracks ({}) require TPCTOF matches ({}) and TPCTOF tracks ({})",
                                           matchesTPCTOF.size(), tracksTPCTOF.size()));
    }
    for (unsigned i = 0; i < matchesTPCTOF.size(); i++) {
      const auto& match = matchesTPCTOF[i];
      const auto& gidx = match.getEvIdxTrack().getIndex(); // TPC (or other? but w/o ITS) track global idx (FIXME: TOF has to git rid of EvIndex stuff)
      if (isUsed(gidx)) {                                  // is TPC track already used
        continue;
      }
      const auto& trc = tracksTPCTOF[i];
      if (creator(trc, trc.getTimeMUS().getTimeStamp(), trc.getTimeMUS().getTimeStampError(), {i, GTrackID::TPCTOF})) {
        flagUsed(gidx); // flag used TPC tracks
      }
    }
  }

  // TPC only tracks
  {
    for (unsigned i = 0; i < tracksTPC.size(); i++) {
      if (isUsed2(i, GTrackID::TPC)) { // skip used tracks
        continue;
      }
      const auto& trc = tracksTPC[i];
      if (creator(trc, trc.getTime0() + 0.5 * (trc.getDeltaTFwd() - trc.getDeltaTBwd()), 0.5 * (trc.getDeltaTFwd() + trc.getDeltaTBwd()), {i, GTrackID::TPC})) {
        flagUsed2(i, GTrackID::TPC); // flag used TPC tracks
      }
    }
  }

  // ITS only tracks
  {
    const auto& rofrs = getITSTracksROFRecords<o2::itsmft::ROFRecord>();
    for (unsigned irof = 0; irof < rofrs.size(); irof++) {
      const auto& rofRec = rofrs[irof];
      float t0 = rofRec.getBCData().differenceInBC(startIR) * o2::constants::lhc::LHCBunchSpacingNS * 1e-3;
      int trlim = rofRec.getFirstEntry() + rofRec.getNEntries();
      for (int it = rofRec.getFirstEntry(); it < trlim; it++) {
        if (isUsed2(it, GTrackID::ITS)) { // skip used tracks
          continue;
        }
        GTrackID gidITS(it, GTrackID::ITS);
        const auto& trc = getITSTrack<o2::its::TrackITS>(gidITS);
        if (creator(trc, t0, 0.5, gidITS)) {
          flagUsed2(it, GTrackID::ITS);
        }
      }
    }
  }
  auto current_time = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "RecoContainer::createTracks took " << std::chrono::duration_cast<std::chrono::microseconds>(current_time - start_time).count() * 1e-6 << " CPU s.";
}

// get contributors from single detectors
RecoContainer::GlobalIDSet RecoContainer::getSingleDetectorRefs(GTrackID gidx) const
{
  GlobalIDSet table;
  auto src = gidx.getSource();
  table[src] = gidx;
  if (src == GTrackID::ITSTPCTOF) {
    const auto& parent0 = getTOFMatch<o2d::MatchInfoTOF>(gidx); //ITS/TPC : TOF
    const auto& parent1 = getTPCITSTrack<o2d::TrackTPCITS>(parent0.getEvIdxTrack().getIndex());
    table[GTrackID::ITSTPC] = parent0.getEvIdxTrack().getIndex();
    table[GTrackID::TOF] = {unsigned(parent0.getEvIdxTOFCl().getIndex()), GTrackID::TOF};
    table[GTrackID::ITS] = parent1.getRefITS();
    table[GTrackID::TPC] = parent1.getRefTPC();
  } else if (src == GTrackID::TPCTOF) {
    const auto& parent0 = getTPCTOFMatch<o2d::MatchInfoTOF>(gidx); //TPC : TOF
    table[GTrackID::TOF] = {unsigned(parent0.getEvIdxTOFCl().getIndex()), GTrackID::TOF};
    table[GTrackID::TPC] = parent0.getEvIdxTrack().getIndex();
  } else if (src == GTrackID::ITSTPC) {
    const auto& parent0 = getTPCITSTrack<o2d::TrackTPCITS>(gidx);
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
    const auto& parent0 = getTOFMatch<o2d::MatchInfoTOF>(gidx); //ITS/TPC : TOF
    const auto& parent1 = getTPCITSTrack<o2d::TrackTPCITS>(parent0.getEvIdxTrack().getIndex());
    return parent1.getRefTPC();
  } else if (src == GTrackID::TPCTOF) {
    const auto& parent0 = getTPCTOFMatch<o2d::MatchInfoTOF>(gidx); //TPC : TOF
    return parent0.getEvIdxTrack().getIndex();
  } else if (src == GTrackID::ITSTPC) {
    const auto& parent0 = getTPCITSTrack<o2d::TrackTPCITS>(gidx);
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
    const auto& parent0 = getTOFMatch<o2d::MatchInfoTOF>(gidx); //ITS/TPC : TOF
    const auto& parent1 = getTPCITSTrack<o2d::TrackTPCITS>(parent0.getEvIdxTrack().getIndex());
    return parent1.getRefITS();
  } else if (src == GTrackID::ITSTPC) {
    const auto& parent0 = getTPCITSTrack<o2d::TrackTPCITS>(gidx);
    return parent0.getRefITS();
  }
  return src == GTrackID::ITS ? gidx : GTrackID{};
}
