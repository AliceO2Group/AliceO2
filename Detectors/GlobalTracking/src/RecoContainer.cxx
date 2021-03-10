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
#include "GlobalTracking/RecoContainer.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTOF/Cluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsFT0/RecPoints.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/TrackTPCTOF.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
// RSTODO to remove once the framework will start propagating the header.firstTForbit
#include "DetectorsRaw/HBFUtils.h"

using namespace o2::globaltracking;
using namespace o2::framework;
namespace o2d = o2::dataformats;

using GTrackID = o2d::GlobalTrackID;

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

void DataRequest::requestFT0RecPoints(bool mc)
{
  addInput({"ft0recpoints", "FT0", "RECPOINTS", 0, Lifetime::Timeframe});
  if (mc) {
    LOG(ERROR) << "FT0 RecPoint does not support MC truth";
  }
  requestMap["FT0"] = false;
}

//__________________________________________________________________
void RecoContainer::collectData(ProcessingContext& pc, const DataRequest& requests)
{
  auto& reqMap = requests.requestMap;

  /// RS FIXME: this will not work until the framework does not propagate the dh->firstTForbit
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
  //RSTODO: below is a hack, to remove once the framework will start propagating the header.firstTForbit
  const auto tracksITSROF = getITSTracksROFRecords<o2::itsmft::ROFRecord>();
  if (tracksITSROF.size()) {
    startIR = o2::raw::HBFUtils::Instance().getFirstIRofTF(tracksITSROF[0].getBCData());
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
void RecoContainer::createTracks(std::function<void(const o2::track::TrackParCov&, float, float, GTrackID)> const& creator) const
{
  // We go from most complete tracks to least complete ones, taking into account that some track times
  // do not bear their own kinematics but just constrain the time
  // As we get more track types functional, this method should be completed
  constexpr float PS2MUS = 1e-6;
  std::array<std::vector<uint8_t>, GTrackID::NSources> usedData;
  auto flagUsed = [&usedData](const GTrackID gidx) { auto src = gidx.getSource();
                                                             if (!usedData[src].empty()) {
							       usedData[gidx.getSource()][gidx.getIndex()] = 1;
							     } };
  auto isUsed = [&usedData](const GTrackID gidx) { return (!usedData[gidx.getSource()].empty()) && (usedData[gidx.getSource()][gidx.getIndex()] != 0); };
  auto isUsed2 = [&usedData](int idx, int src) { return (!usedData[src].empty()) && (usedData[src][idx] != 0); };

  // create only for those data types which are used
  const auto& tracksITS = getITSTracks<o2::its::TrackITS>();
  const auto& tracksTPC = getTPCTracks<o2::tpc::TrackTPC>();
  const auto& tracksTPCITS = getTPCITSTracks<o2d::TrackTPCITS>();
  const auto& tracksTPCTOF = getTPCTOFTracks<o2d::TrackTPCTOF>();
  const auto& matchesTPCTOF = getTPCTOFMatches<o2d::MatchInfoTOF>();

  usedData[GTrackID::ITS].resize(tracksITS.size());       // to flag used ITS tracks
  usedData[GTrackID::TPC].resize(tracksTPC.size());       // to flag used TPC tracks
  usedData[GTrackID::ITSTPC].resize(tracksTPCITS.size()); // to flag used ITSTPC tracks

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
      creator(tracksPool.get(gidx), timeTOFMUS, timeErr, {i, GTrackID::ITSTPCTOF});
      flagUsed(gidx); // flag used ITS-TPC tracks
    }
  }

  // ITS-TPC matches, may refer to ITS, TPC (TODO: something else?) tracks
  {
    for (unsigned i = 0; i < tracksTPCITS.size(); i++) {
      const auto& matchTr = tracksTPCITS[i];
      flagUsed(matchTr.getRefITS()); // flag used ITS tracks
      flagUsed(matchTr.getRefTPC()); // flag used TPC tracks
      if (isUsed2(i, GTrackID::ITSTPC)) {
        continue;
      }
      creator(matchTr, matchTr.getTimeMUS().getTimeStamp(), matchTr.getTimeMUS().getTimeStampError(), {i, GTrackID::ITSTPC});
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
      const auto& gidx = match.getEvIdxTrack().getIndex(); // TPC (or other?) track global idx (FIXME: TOF has to git rid of EvIndex stuff)
      if (isUsed(gidx)) {                                  // is TPC track already used
        continue;
      }
      flagUsed(gidx); // flag used TPC tracks
      const auto& trc = tracksTPCTOF[i];
      creator(trc, trc.getTimeMUS().getTimeStamp(), trc.getTimeMUS().getTimeStampError(), {i, GTrackID::TPCTOF});
    }
  }

  // TPC only tracks
  {
    for (unsigned i = 0; i < tracksTPC.size(); i++) {
      if (isUsed2(i, GTrackID::TPC)) { // skip used tracks
        continue;
      }
      const auto& trc = tracksTPC[i];
      creator(trc, trc.getTime0(), 0.5 * (trc.getDeltaTBwd() + trc.getDeltaTFwd()), {i, GTrackID::TPC});
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
        creator(trc, t0, 0.5, gidITS);
      }
    }
  }
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
