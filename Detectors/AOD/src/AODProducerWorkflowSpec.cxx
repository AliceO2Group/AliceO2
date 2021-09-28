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

/// @file   AODProducerWorkflowSpec.cxx

#include "AODProducerWorkflow/AODProducerWorkflowSpec.h"
#include "DataFormatsFT0/RecPoints.h"
#include "DataFormatsFDD/RecPoint.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsBase/GeometryManager.h"
#include "CCDB/BasicCCDBManager.h"
#include "CommonConstants/PhysicsConstants.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataTypes.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Logger.h"
#include "Framework/TableBuilder.h"
#include "Framework/TableTreeHelpers.h"
#include "FDDBase/Constants.h"
#include "FT0Base/Geometry.h"
#include "FV0Base/Geometry.h"
#include "GlobalTracking/MatchTOF.h"
#include "ReconstructionDataFormats/Cascade.h"
#include "MCHTracking/TrackExtrap.h"
#include "MCHTracking/TrackParam.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/GlobalFwdTrack.h"
#include "ReconstructionDataFormats/V0.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TMath.h"
#include "MathUtils/Utils.h"
#include <map>
#include <unordered_map>
#include <vector>

using namespace o2::framework;
using namespace o2::math_utils::detail;
using PVertex = o2::dataformats::PrimaryVertex;
using GIndex = o2::dataformats::VtxTrackIndex;
using DataRequest = o2::globaltracking::DataRequest;
using GID = o2::dataformats::GlobalTrackID;

namespace o2::aodproducer
{

namespace
{
// takes a local vertex timing in NS and converts to a global BC information
// using the orbit offset from the simulation
uint64_t relativeTime_to_GlobalBC(double relativeTimeStampInNS)
{
  return std::round((o2::raw::HBFUtils::Instance().getFirstSampledTFIR().bc2ns() + relativeTimeStampInNS) / o2::constants::lhc::LHCBunchSpacingNS);
}
} // namespace

void AODProducerWorkflowDPL::collectBCs(gsl::span<const o2::fdd::RecPoint>& fddRecPoints,
                                        gsl::span<const o2::ft0::RecPoints>& ft0RecPoints,
                                        gsl::span<const o2::fv0::RecPoints>& fv0RecPoints,
                                        gsl::span<const o2::dataformats::PrimaryVertex>& primVertices,
                                        const std::vector<o2::InteractionTimeRecord>& mcRecords,
                                        std::map<uint64_t, int>& bcsMap)
{
  // collecting non-empty BCs and enumerating them
  for (auto& rec : mcRecords) {
    uint64_t globalBC = rec.toLong();
    bcsMap[globalBC] = 1;
  }

  for (auto& fddRecPoint : fddRecPoints) {
    uint64_t globalBC = fddRecPoint.getInteractionRecord().toLong();
    bcsMap[globalBC] = 1;
  }

  for (auto& ft0RecPoint : ft0RecPoints) {
    uint64_t globalBC = ft0RecPoint.getInteractionRecord().toLong();
    bcsMap[globalBC] = 1;
  }

  for (auto& fv0RecPoint : fv0RecPoints) {
    uint64_t globalBC = fv0RecPoint.getInteractionRecord().toLong();
    bcsMap[globalBC] = 1;
  }

  for (auto& vertex : primVertices) {
    auto& timeStamp = vertex.getTimeStamp();
    double tsTimeStamp = timeStamp.getTimeStamp() * 1E3; // mus to ns
    uint64_t globalBC = relativeTime_to_GlobalBC(tsTimeStamp);
    bcsMap[globalBC] = 1;
  }

  int bcID = 0;
  for (auto& item : bcsMap) {
    item.second = bcID;
    bcID++;
  }
}

uint64_t AODProducerWorkflowDPL::getTFNumber(const o2::InteractionRecord& tfStartIR, int runNumber)
{
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  o2::ccdb::CcdbApi ccdb_api;
  const std::string rct_path = "RCT/RunInformation/";
  const std::string start_orbit_path = "Trigger/StartOrbit";
  const std::string url = "http://ccdb-test.cern.ch:8080";

  mgr.setURL(url);
  ccdb_api.init(url);

  std::map<int, int>* mapStartOrbit = mgr.get<std::map<int, int>>(start_orbit_path);
  int64_t ts = 0;
  std::map<std::string, std::string> metadata;
  std::map<std::string, std::string> headers;
  const std::string run_path = Form("%s/%i", rct_path.data(), runNumber);
  headers = ccdb_api.retrieveHeaders(run_path, metadata, -1);
  ts = atol(headers["SOR"].c_str());

  // ccdb returns timestamp in mus
  // mus to ms
  ts = ts / 1000;

  uint32_t initialOrbit = mapStartOrbit->at(runNumber);
  uint16_t firstRecBC = tfStartIR.bc;
  uint32_t firstRecOrbit = tfStartIR.orbit;
  const o2::InteractionRecord firstRec(firstRecBC, firstRecOrbit);
  ts += firstRec.bc2ns() / 1000000;

  return ts;
};

template <typename TracksCursorType, typename TracksCovCursorType>
void AODProducerWorkflowDPL::addToTracksTable(TracksCursorType& tracksCursor, TracksCovCursorType& tracksCovCursor,
                                              const o2::track::TrackParCov& track, int collisionID)
{
  // tracks
  tracksCursor(0,
               collisionID,
               o2::aod::track::Track,
               truncateFloatFraction(track.getX(), mTrackX),
               truncateFloatFraction(track.getAlpha(), mTrackAlpha),
               track.getY(),
               track.getZ(),
               truncateFloatFraction(track.getSnp(), mTrackSnp),
               truncateFloatFraction(track.getTgl(), mTrackTgl),
               truncateFloatFraction(track.getQ2Pt(), mTrack1Pt));
  // trackscov
  tracksCovCursor(0,
                  truncateFloatFraction(TMath::Sqrt(track.getSigmaY2()), mTrackCovDiag),
                  truncateFloatFraction(TMath::Sqrt(track.getSigmaZ2()), mTrackCovDiag),
                  truncateFloatFraction(TMath::Sqrt(track.getSigmaSnp2()), mTrackCovDiag),
                  truncateFloatFraction(TMath::Sqrt(track.getSigmaTgl2()), mTrackCovDiag),
                  truncateFloatFraction(TMath::Sqrt(track.getSigma1Pt2()), mTrackCovDiag),
                  (Char_t)(128. * track.getSigmaZY() / track.getSigmaZ2() / track.getSigmaY2()),
                  (Char_t)(128. * track.getSigmaSnpY() / track.getSigmaSnp2() / track.getSigmaY2()),
                  (Char_t)(128. * track.getSigmaSnpZ() / track.getSigmaSnp2() / track.getSigmaZ2()),
                  (Char_t)(128. * track.getSigmaTglY() / track.getSigmaTgl2() / track.getSigmaY2()),
                  (Char_t)(128. * track.getSigmaTglZ() / track.getSigmaTgl2() / track.getSigmaZ2()),
                  (Char_t)(128. * track.getSigmaTglSnp() / track.getSigmaTgl2() / track.getSigmaSnp2()),
                  (Char_t)(128. * track.getSigma1PtY() / track.getSigma1Pt2() / track.getSigmaY2()),
                  (Char_t)(128. * track.getSigma1PtZ() / track.getSigma1Pt2() / track.getSigmaZ2()),
                  (Char_t)(128. * track.getSigma1PtSnp() / track.getSigma1Pt2() / track.getSigmaSnp2()),
                  (Char_t)(128. * track.getSigma1PtTgl() / track.getSigma1Pt2() / track.getSigmaTgl2()));
}

template <typename TracksExtraCursorType>
void AODProducerWorkflowDPL::addToTracksExtraTable(TracksExtraCursorType& tracksExtraCursor, TrackExtraInfo& extraInfoHolder)
{
  // extra
  tracksExtraCursor(0,
                    truncateFloatFraction(extraInfoHolder.tpcInnerParam, mTrack1Pt),
                    extraInfoHolder.flags,
                    extraInfoHolder.itsClusterMap,
                    extraInfoHolder.tpcNClsFindable,
                    extraInfoHolder.tpcNClsFindableMinusFound,
                    extraInfoHolder.tpcNClsFindableMinusCrossedRows,
                    extraInfoHolder.tpcNClsShared,
                    extraInfoHolder.trdPattern,
                    truncateFloatFraction(extraInfoHolder.itsChi2NCl, mTrackCovOffDiag),
                    truncateFloatFraction(extraInfoHolder.tpcChi2NCl, mTrackCovOffDiag),
                    truncateFloatFraction(extraInfoHolder.trdChi2, mTrackCovOffDiag),
                    truncateFloatFraction(extraInfoHolder.tofChi2, mTrackCovOffDiag),
                    truncateFloatFraction(extraInfoHolder.tpcSignal, mTrackSignal),
                    truncateFloatFraction(extraInfoHolder.trdSignal, mTrackSignal),
                    truncateFloatFraction(extraInfoHolder.length, mTrackSignal),
                    truncateFloatFraction(extraInfoHolder.tofExpMom, mTrack1Pt),
                    truncateFloatFraction(extraInfoHolder.trackEtaEMCAL, mTrackPosEMCAL),
                    truncateFloatFraction(extraInfoHolder.trackPhiEMCAL, mTrackPosEMCAL),
                    truncateFloatFraction(extraInfoHolder.trackTime, mTrackSignal),
                    truncateFloatFraction(extraInfoHolder.trackTimeRes, mTrackSignal));
}

template <typename mftTracksCursorType>
void AODProducerWorkflowDPL::addToMFTTracksTable(mftTracksCursorType& mftTracksCursor,
                                                 const o2::mft::TrackMFT& track, int collisionID)
{
  // mft tracks
  mftTracksCursor(0,
                  collisionID,
                  track.getX(),
                  track.getY(),
                  track.getZ(),
                  track.getPhi(),
                  track.getTanl(),
                  track.getInvQPt(),
                  track.getNumberOfPoints(),
                  track.getTrackChi2());
}

template <typename TracksCursorType, typename TracksCovCursorType, typename TracksExtraCursorType, typename MftTracksCursorType, typename FwdTracksCursorType>
void AODProducerWorkflowDPL::fillTrackTablesPerCollision(int collisionID,
                                                         double interactionTime,
                                                         const o2::dataformats::VtxTrackRef& trackRef,
                                                         gsl::span<const GIndex>& GIndices,
                                                         o2::globaltracking::RecoContainer& data,
                                                         TracksCursorType& tracksCursor,
                                                         TracksCovCursorType& tracksCovCursor,
                                                         TracksExtraCursorType& tracksExtraCursor,
                                                         MftTracksCursorType& mftTracksCursor,
                                                         FwdTracksCursorType& fwdTracksCursor,
                                                         const dataformats::PrimaryVertex& vertex)
{
  const auto& tpcClusRefs = data.getTPCTracksClusterRefs();
  const auto& tpcClusShMap = data.clusterShMapTPC;
  const auto& tpcClusAcc = data.getTPCClusters();
  const auto& tpcTracks = data.getTPCTracks();
  const auto& itsTracks = data.getITSTracks();
  const auto& itsABRefs = data.getITSABRefs();
  const auto& tofClus = data.getTOFClusters();

  for (int src = GIndex::NSources; src--;) {
    int start = trackRef.getFirstEntryOfSource(src);
    int end = start + trackRef.getEntriesOfSource(src);
    for (int ti = start; ti < end; ti++) {
      TrackExtraInfo extraInfoHolder;
      auto& trackIndex = GIndices[ti];
      if (GIndex::includesSource(src, mInputSources)) {
        if (src == GIndex::Source::MFT) { // MFT tracks are treated separately since they are stored in a different table
          const auto& track = data.getMFTTrack(trackIndex);
          addToMFTTracksTable(mftTracksCursor, track, collisionID);
        } else if (src == GIndex::Source::MCH) {
          // FwdTracks tracks are treated separately since they are stored in a different table
          const auto& track = data.getMCHTrack(trackIndex);
          if (collisionID < 0) {
            InteractionRecord meanIR;
            auto rofsMCH = data.getMCHTracksROFRecords();
            for (const auto& rof : rofsMCH) {
              if (trackIndex >= rof.getFirstIdx() && trackIndex <= rof.getLastIdx()) {
                meanIR = rof.getBCData() + rof.getBCWidth() / 2;
              }
              math_utils::Point3D<double> vertex{};
              // FIXME: should we get better
              // than {0,0,0} as vertex here ?
              addToFwdTracksTable(fwdTracksCursor, track, -1, vertex);
            }
          } else {
            math_utils::Point3D<double> vtx{vertex.getX(),
                                            vertex.getY(), vertex.getZ()};
            addToFwdTracksTable(fwdTracksCursor, track, collisionID, vtx);
          }
        } else if (src == GIndex::Source::MFTMCH) {
          const auto& track = data.getGlobalFwdTrack(trackIndex);
          addToFwdTracksTable(fwdTracksCursor, track, collisionID, {0, 0, 0});
        } else {
          auto contributorsGID = data.getSingleDetectorRefs(trackIndex);
          const auto& trackPar = data.getTrackParam(trackIndex);
          if (contributorsGID[GIndex::Source::ITS].isIndexSet()) {
            extraInfoHolder.itsClusterMap = itsTracks[contributorsGID[GIndex::ITS].getIndex()].getPattern();
          } else if (contributorsGID[GIndex::Source::ITSAB].isIndexSet()) { // this is an ITS-TPC afterburner contributor
            extraInfoHolder.itsClusterMap = itsABRefs[contributorsGID[GIndex::Source::ITSAB].getIndex()].pattern;
          }
          if (contributorsGID[GIndex::Source::TPC].isIndexSet()) {
            const auto& tpcOrig = tpcTracks[contributorsGID[GIndex::TPC].getIndex()];
            extraInfoHolder.tpcInnerParam = tpcOrig.getP();
            extraInfoHolder.tpcChi2NCl = tpcOrig.getNClusters() ? tpcOrig.getChi2() / tpcOrig.getNClusters() : 0;
            extraInfoHolder.tpcSignal = tpcOrig.getdEdx().dEdxTotTPC;
            uint8_t shared, found, crossed; // fixme: need to switch from these placeholders to something more reasonable
            countTPCClusters(tpcOrig, tpcClusRefs, tpcClusShMap, tpcClusAcc, shared, found, crossed);
            extraInfoHolder.tpcNClsFindable = tpcOrig.getNClusters();
            extraInfoHolder.tpcNClsFindableMinusFound = tpcOrig.getNClusters() - found;
            extraInfoHolder.tpcNClsFindableMinusCrossedRows = tpcOrig.getNClusters() - crossed;
            extraInfoHolder.tpcNClsShared = shared;
          }
          if (contributorsGID[GIndex::Source::ITSTPCTOF].isIndexSet()) {
            const auto& tofMatch = data.getTOFMatch(contributorsGID[GIndex::Source::ITSTPCTOF]);
            extraInfoHolder.tofChi2 = tofMatch.getChi2();
            const auto& tofInt = tofMatch.getLTIntegralOut();
            float intLen = tofInt.getL();
            extraInfoHolder.length = intLen;
            if (interactionTime > 0) {
              extraInfoHolder.tofSignal = static_cast<float>(tofMatch.getSignal() - interactionTime);
            }
            const float mass = o2::constants::physics::MassPionCharged; // default pid = pion
            if (tofInt.getTOF(o2::track::PID::Pion) > 0.f) {
              const float expBeta = (intLen / (tofInt.getTOF(o2::track::PID::Pion) * cSpeed));
              extraInfoHolder.tofExpMom = mass * expBeta / std::sqrt(1.f - expBeta * expBeta);
            }
            const auto& tofCl = tofClus[contributorsGID[GIndex::Source::TOF]];
            // correct the time of the track
            extraInfoHolder.trackTime = (tofCl.getTime() - tofInt.getTOF(trackPar.getPID())) * 1e-3; // tof time in \mus, FIXME: account for time of flight to R TOF
            extraInfoHolder.trackTimeRes = 200e-3;                                                   // FIXME: calculate actual resolution (if possible?)
          }
          if (src == GIndex::Source::TPCTRD || src == GIndex::Source::ITSTPCTRD) {
            const auto& trdOrig = data.getTrack<o2::trd::TrackTRD>(src, contributorsGID[src].getIndex());
            extraInfoHolder.trdChi2 = trdOrig.getChi2();
            extraInfoHolder.trdPattern = getTRDPattern(trdOrig);
          }
          addToTracksTable(tracksCursor, tracksCovCursor, trackPar, collisionID);
          addToTracksExtraTable(tracksExtraCursor, extraInfoHolder);
          // collecting table indices of barrel tracks for V0s table
          mGIDToTableID.emplace(trackIndex, mTableTrID);
          mTableTrID++;
        }
      }
    }
  }
}

template <typename FwdTracksCursorType, typename fwdTrackType>
void AODProducerWorkflowDPL::addToFwdTracksTable(FwdTracksCursorType& fwdTracksCursor,
                                                 const fwdTrackType& track, int collisionID,
                                                 const math_utils::Point3D<double>& vertex)

{

  // table columns must be floats, not double
  uint8_t trackTypeId;
  float x;
  float y;
  float z;
  float rabs;
  float phi;
  float tanl;
  float invqpt;
  float chi2;
  float pdca;
  int nClusters;
  float chi2matchmchmid = -1.0;
  float chi2matchmchmft = -1.0;
  float matchscoremchmft = -1.0;
  int matchmfttrackid = -1;
  int matchmchtrackid = -1;
  uint16_t mchBitMap = 0;
  uint8_t midBitMap = 0;
  uint32_t midBoards = 0;
  float trackTime = 0;
  float trackTimeRes = 0;

  if constexpr (!std::is_base_of_v<o2::track::TrackParCovFwd, std::decay_t<decltype(track)>>) {
    // This is a MCH track
    trackTypeId = o2::aod::fwdtrack::MCHStandaloneTrack;
    // mch standalone tracks extrapolated to vertex

    // compute 3 sets of tracks parameters :
    // - at vertex
    // - at DCA
    // - at the end of the absorber

    // extrapolate to vertex
    o2::mch::TrackParam trackParamAtVertex(track.getZ(), track.getParameters());
    double errVtx{0.0}; // FIXME: get errors associated with vertex if available
    double errVty{0.0};
    if (!o2::mch::TrackExtrap::extrapToVertex(trackParamAtVertex, vertex.x(), vertex.y(), vertex.z(), errVtx, errVty)) {
      return;
    }

    // extrapolate to DCA
    o2::mch::TrackParam trackParamAtDCA(track.getZ(), track.getParameters());
    if (!o2::mch::TrackExtrap::extrapToVertexWithoutBranson(trackParamAtDCA, vertex.z())) {
      return;
    }

    // extrapolate to the end of the absorber
    o2::mch::TrackParam trackParamAtRAbs(track.getZ(), track.getParameters());
    if (!o2::mch::TrackExtrap::extrapToZ(trackParamAtRAbs, -505.)) { // FIXME: replace hardcoded 505
      return;
    }

    double dcaX = trackParamAtDCA.getNonBendingCoor() - vertex.x();
    double dcaY = trackParamAtDCA.getBendingCoor() - vertex.y();
    double dca = std::sqrt(dcaX * dcaX + dcaY * dcaY);

    double xAbs = trackParamAtRAbs.getNonBendingCoor();
    double yAbs = trackParamAtRAbs.getBendingCoor();

    double px = trackParamAtVertex.px();
    double py = trackParamAtVertex.py();
    double pz = trackParamAtVertex.pz();

    double pt = std::sqrt(px * px + py * py);
    double dphi = std::asin(py / pt);
    double dtanl = pz / pt;
    double dinvqpt = 1.0 / (trackParamAtVertex.getCharge() * pt);
    double dpdca = trackParamAtVertex.p() * dca;
    double dchi2 = track.getChi2OverNDF();

    x = trackParamAtVertex.getNonBendingCoor();
    y = trackParamAtVertex.getBendingCoor();
    z = trackParamAtVertex.getZ();
    rabs = std::sqrt(xAbs * xAbs + yAbs * yAbs);
    phi = dphi;
    tanl = dtanl;
    invqpt = dinvqpt;
    chi2 = dchi2;
    pdca = dpdca;
    nClusters = track.getNClusters();

  } else {
    // This is a GlobalMuonTrack or a GlobalForwardTrack
    x = track.getX();
    y = track.getY();
    z = track.getZ();
    phi = track.getPhi();
    tanl = track.getTanl();
    invqpt = track.getInvQPt();
    chi2 = track.getTrackChi2();
    //nClusters = track.getNumberOfPoints();
    chi2matchmchmid = track.getMIDMatchingChi2();
    chi2matchmchmft = track.getMatchingChi2();
    matchmfttrackid = track.getMFTTrackID();
    matchmchtrackid = track.getMCHTrackID();
    trackTypeId = (chi2matchmchmid >= 0) ? o2::aod::fwdtrack::GlobalMuonTrack : o2::aod::fwdtrack::GlobalForwardTrack;
  }

  fwdTracksCursor(0,
                  collisionID,
                  trackTypeId,
                  x,
                  y,
                  z,
                  phi,
                  tanl,
                  invqpt,
                  nClusters,
                  pdca,
                  rabs,
                  chi2,
                  chi2matchmchmid,
                  chi2matchmchmft,
                  matchscoremchmft,
                  matchmfttrackid,
                  matchmchtrackid,
                  mchBitMap,
                  midBitMap,
                  midBoards,
                  trackTime,
                  trackTimeRes);
}

template <typename MCParticlesCursorType>
void AODProducerWorkflowDPL::fillMCParticlesTable(o2::steer::MCKinematicsReader& mcReader,
                                                  const MCParticlesCursorType& mcParticlesCursor,
                                                  gsl::span<const o2::dataformats::VtxTrackRef>& primVer2TRefs,
                                                  gsl::span<const GIndex>& GIndices,
                                                  o2::globaltracking::RecoContainer& data,
                                                  std::vector<std::pair<int, int>> const& mcColToEvSrc)
{
  // mark reconstructed MC particles to store them into the table
  for (auto& trackRef : primVer2TRefs) {
    for (int src = GIndex::NSources; src--;) {
      int start = trackRef.getFirstEntryOfSource(src);
      int end = start + trackRef.getEntriesOfSource(src);
      for (int ti = start; ti < end; ti++) {
        auto& trackIndex = GIndices[ti];
        if (GIndex::includesSource(src, mInputSources)) {
          auto mcTruth = data.getTrackMCLabel(trackIndex);
          if (!mcTruth.isValid()) {
            continue;
          }
          int source = mcTruth.getSourceID();
          int event = mcTruth.getEventID();
          int particle = mcTruth.getTrackID();
          mToStore[Triplet_t(source, event, particle)] = 1;
          // treating contributors of global tracks
          auto contributorsGID = data.getSingleDetectorRefs(trackIndex);
          if (contributorsGID[GIndex::Source::ITS].isIndexSet() && contributorsGID[GIndex::Source::TPC].isIndexSet()) {
            auto mcTruthITS = data.getTrackMCLabel(contributorsGID[GIndex::Source::ITS]);
            if (mcTruthITS.isValid()) {
              source = mcTruthITS.getSourceID();
              event = mcTruthITS.getEventID();
              particle = mcTruthITS.getTrackID();
              mToStore[Triplet_t(source, event, particle)] = 1;
            }
            auto mcTruthTPC = data.getTrackMCLabel(contributorsGID[GIndex::Source::TPC]);
            if (mcTruthTPC.isValid()) {
              source = mcTruthTPC.getSourceID();
              event = mcTruthTPC.getEventID();
              particle = mcTruthTPC.getTrackID();
              mToStore[Triplet_t(source, event, particle)] = 1;
            }
          }
        }
      }
    }
  }
  int tableIndex = 1;
  for (int mccolid = 0; mccolid < mcColToEvSrc.size(); ++mccolid) {
    auto event = mcColToEvSrc[mccolid].first;
    auto source = mcColToEvSrc[mccolid].second;
    std::vector<MCTrack> const& mcParticles = mcReader.getTracks(source, event);
    // mark tracks to be stored per event
    // loop over stack of MC particles from end to beginning: daughters are stored after mothers
    if (mRecoOnly) {
      for (int particle = mcParticles.size() - 1; particle >= 0; particle--) {
        int mother0 = mcParticles[particle].getMotherTrackId();
        if (mother0 == -1) {
          mToStore[Triplet_t(source, event, particle)] = 1;
        }
        if (mToStore.find(Triplet_t(source, event, particle)) == mToStore.end()) {
          continue;
        }
        if (mother0 != -1) {
          mToStore[Triplet_t(source, event, mother0)] = 1;
        }
        int mother1 = mcParticles[particle].getSecondMotherTrackId();
        if (mother1 != -1) {
          mToStore[Triplet_t(source, particle, mother1)] = 1;
        }
        int daughter0 = mcParticles[particle].getFirstDaughterTrackId();
        if (daughter0 != -1) {
          mToStore[Triplet_t(source, event, daughter0)] = 1;
        }
        int daughterL = mcParticles[particle].getLastDaughterTrackId();
        if (daughterL != -1) {
          mToStore[Triplet_t(source, event, daughterL)] = 1;
        }
      }
      // enumerate reconstructed mc particles and their relatives to get mother/daughter relations
      for (int particle = 0; particle < mcParticles.size(); particle++) {
        auto mapItem = mToStore.find(Triplet_t(source, event, particle));
        if (mapItem != mToStore.end()) {
          mapItem->second = tableIndex - 1;
          tableIndex++;
        }
      }
    }
    // if all mc particles are stored, all mc particles will be enumerated
    if (!mRecoOnly) {
      for (int particle = 0; particle < mcParticles.size(); particle++) {
        mToStore[Triplet_t(source, event, particle)] = tableIndex - 1;
        tableIndex++;
      }
    }
    // fill survived mc tracks into the table
    for (int particle = 0; particle < mcParticles.size(); particle++) {
      if (mToStore.find(Triplet_t(source, event, particle)) == mToStore.end()) {
        continue;
      }
      int statusCode = 0;
      uint8_t flags = 0;
      float weight = 0.f;
      int mcMother0 = mcParticles[particle].getMotherTrackId();
      auto item = mToStore.find(Triplet_t(source, event, mcMother0));
      int mother0 = -1;
      if (item != mToStore.end()) {
        mother0 = item->second;
      }
      int mcMother1 = mcParticles[particle].getSecondMotherTrackId();
      int mother1 = -1;
      item = mToStore.find(Triplet_t(source, event, mcMother1));
      if (item != mToStore.end()) {
        mother1 = item->second;
      }
      int mcDaughter0 = mcParticles[particle].getFirstDaughterTrackId();
      int daughter0 = -1;
      item = mToStore.find(Triplet_t(source, event, mcDaughter0));
      if (item != mToStore.end()) {
        daughter0 = item->second;
      }
      int mcDaughterL = mcParticles[particle].getLastDaughterTrackId();
      int daughterL = -1;
      item = mToStore.find(Triplet_t(source, event, mcDaughterL));
      if (item != mToStore.end()) {
        daughterL = item->second;
      }
      float pX = (float)mcParticles[particle].Px();
      float pY = (float)mcParticles[particle].Py();
      float pZ = (float)mcParticles[particle].Pz();
      float energy = (float)mcParticles[particle].GetEnergy();

      mcParticlesCursor(0,
                        mccolid,
                        mcParticles[particle].GetPdgCode(),
                        statusCode,
                        flags,
                        mother0,
                        mother1,
                        daughter0,
                        daughterL,
                        truncateFloatFraction(weight, mMcParticleW),
                        truncateFloatFraction(pX, mMcParticleMom),
                        truncateFloatFraction(pY, mMcParticleMom),
                        truncateFloatFraction(pZ, mMcParticleMom),
                        truncateFloatFraction(energy, mMcParticleMom),
                        truncateFloatFraction((float)mcParticles[particle].Vx(), mMcParticlePos),
                        truncateFloatFraction((float)mcParticles[particle].Vy(), mMcParticlePos),
                        truncateFloatFraction((float)mcParticles[particle].Vz(), mMcParticlePos),
                        truncateFloatFraction((float)mcParticles[particle].T(), mMcParticlePos));
    }
    mcReader.releaseTracksForSourceAndEvent(source, event);
  }
}

template <typename MCTrackLabelCursorType, typename MCMFTTrackLabelCursorType, typename MCFwdTrackLabelCursorType>
void AODProducerWorkflowDPL::fillMCTrackLabelsTable(const MCTrackLabelCursorType& mcTrackLabelCursor,
                                                    const MCMFTTrackLabelCursorType& mcMFTTrackLabelCursor,
                                                    const MCFwdTrackLabelCursorType& mcFwdTrackLabelCursor,
                                                    o2::dataformats::VtxTrackRef const& trackRef,
                                                    gsl::span<const GIndex>& primVerGIs,
                                                    o2::globaltracking::RecoContainer& data)
{
  // labelMask (temporary) usage:
  //   bit 13 -- ITS and TPC labels are not equal
  //   bit 14 -- isNoise() == true
  //   bit 15 -- isFake() == true
  // labelID = -1 -- label is not set

  for (int src = GIndex::NSources; src--;) {
    int start = trackRef.getFirstEntryOfSource(src);
    int end = start + trackRef.getEntriesOfSource(src);
    for (int ti = start; ti < end; ti++) {
      auto& trackIndex = primVerGIs[ti];
      if (GIndex::includesSource(src, mInputSources)) {
        auto mcTruth = data.getTrackMCLabel(trackIndex);
        MCLabels labelHolder;
        if ((src == GIndex::Source::MFT) || (src == GIndex::Source::MFTMCH) || (src == GIndex::Source::MCH)) { // treating mft and fwd labels separately
          if (mcTruth.isValid()) {                                                                             // if not set, -1 will be stored
            labelHolder.labelID = mToStore.at(Triplet_t(mcTruth.getSourceID(), mcTruth.getEventID(), mcTruth.getTrackID()));
          }
          if (mcTruth.isFake()) {
            labelHolder.fwdLabelMask |= (0x1 << 7);
          }
          if (mcTruth.isNoise()) {
            labelHolder.fwdLabelMask |= (0x1 << 6);
          }
          if (src == GIndex::Source::MFT) {
            mcMFTTrackLabelCursor(0,
                                  labelHolder.labelID,
                                  labelHolder.fwdLabelMask);

          } else {
            mcFwdTrackLabelCursor(0,
                                  labelHolder.labelID,
                                  labelHolder.fwdLabelMask);
          }
        } else {
          if (mcTruth.isValid()) { // if not set, -1 will be stored
            labelHolder.labelID = mToStore.at(Triplet_t(mcTruth.getSourceID(), mcTruth.getEventID(), mcTruth.getTrackID()));
          }
          // treating possible mismatches for global tracks
          auto contributorsGID = data.getSingleDetectorRefs(trackIndex);
          if (contributorsGID[GIndex::Source::ITS].isIndexSet() && contributorsGID[GIndex::Source::TPC].isIndexSet()) {
            auto mcTruthITS = data.getTrackMCLabel(contributorsGID[GIndex::Source::ITS]);
            if (mcTruthITS.isValid()) {
              labelHolder.labelITS = mToStore.at(Triplet_t(mcTruthITS.getSourceID(), mcTruthITS.getEventID(), mcTruthITS.getTrackID()));
            }
            auto mcTruthTPC = data.getTrackMCLabel(contributorsGID[GIndex::Source::TPC]);
            if (mcTruthTPC.isValid()) {
              labelHolder.labelTPC = mToStore.at(Triplet_t(mcTruthTPC.getSourceID(), mcTruthTPC.getEventID(), mcTruthTPC.getTrackID()));
              labelHolder.labelID = labelHolder.labelTPC;
            }
            if (labelHolder.labelITS != labelHolder.labelTPC) {
              LOG(DEBUG) << "ITS-TPC MCTruth: labelIDs do not match at " << trackIndex.getIndex() << ", src = " << src;
              labelHolder.labelMask |= (0x1 << 13);
            }
          }
          if (mcTruth.isFake()) {
            labelHolder.labelMask |= (0x1 << 15);
          }
          if (mcTruth.isNoise()) {
            labelHolder.labelMask |= (0x1 << 14);
          }
          mcTrackLabelCursor(0,
                             labelHolder.labelID,
                             labelHolder.labelMask);
        }
      }
    }
  }
}

void AODProducerWorkflowDPL::countTPCClusters(const o2::tpc::TrackTPC& track,
                                              const gsl::span<const o2::tpc::TPCClRefElem>& tpcClusRefs,
                                              const gsl::span<const unsigned char>& tpcClusShMap,
                                              const o2::tpc::ClusterNativeAccess& tpcClusAcc,
                                              uint8_t& shared, uint8_t& found, uint8_t& crossed)
{
  constexpr int maxRows = 152;
  constexpr int neighbour = 2;
  std::array<bool, maxRows> clMap{}, shMap{};
  uint8_t sectorIndex;
  uint8_t rowIndex;
  uint32_t clusterIndex;
  shared = 0;
  for (int i = 0; i < track.getNClusterReferences(); i++) {
    o2::tpc::TrackTPC::getClusterReference(tpcClusRefs, i, sectorIndex, rowIndex, clusterIndex, track.getClusterRef());
    unsigned int absoluteIndex = tpcClusAcc.clusterOffset[sectorIndex][rowIndex] + clusterIndex;
    clMap[rowIndex] = true;
    if (tpcClusShMap[absoluteIndex] > 1) {
      if (!shMap[rowIndex]) {
        shared++;
      }
      shMap[rowIndex] = true;
    }
  }

  crossed = 0;
  found = 0;
  int last = -1;
  for (int i = 0; i < maxRows; i++) {
    if (clMap[i]) {
      crossed++;
      found++;
      last = i;
    } else if ((i - last) <= neighbour) {
      crossed++;
    } else {
      int lim = std::min(i + 1 + neighbour, maxRows);
      for (int j = i + 1; j < lim; j++) {
        if (clMap[j]) {
          crossed++;
        }
      }
    }
  }
}

uint8_t AODProducerWorkflowDPL::getTRDPattern(const o2::trd::TrackTRD& track)
{
  uint8_t pattern = 0;
  for (int il = o2::trd::TrackTRD::EGPUTRDTrack::kNLayers; il >= 0; il--) {
    if (track.getTrackletIndex(il) != -1) {
      pattern |= 0x1 << il;
    }
  }
  return pattern;
}

void AODProducerWorkflowDPL::init(InitContext& ic)
{
  mTimer.Stop();
  mTFNumber = ic.options().get<int64_t>("aod-timeframe-id");
  mRecoOnly = ic.options().get<int>("reco-mctracks-only");
  mTruncate = ic.options().get<int>("enable-truncation");

  if (mTFNumber == -1L) {
    LOG(INFO) << "TFNumber will be obtained from CCDB";
  }

  if (mTruncate != 1) {
    LOG(INFO) << "Truncation is not used!";
    mCollisionPosition = 0xFFFFFFFF;
    mCollisionPositionCov = 0xFFFFFFFF;
    mTrackX = 0xFFFFFFFF;
    mTrackAlpha = 0xFFFFFFFF;
    mTrackSnp = 0xFFFFFFFF;
    mTrackTgl = 0xFFFFFFFF;
    mTrack1Pt = 0xFFFFFFFF;
    mTrackCovDiag = 0xFFFFFFFF;
    mTrackCovOffDiag = 0xFFFFFFFF;
    mTrackSignal = 0xFFFFFFFF;
    mTrackPosEMCAL = 0xFFFFFFFF;
    mTracklets = 0xFFFFFFFF;
    mMcParticleW = 0xFFFFFFFF;
    mMcParticlePos = 0xFFFFFFFF;
    mMcParticleMom = 0xFFFFFFFF;
    mCaloAmp = 0xFFFFFFFF;
    mCaloTime = 0xFFFFFFFF;
    mMuonTr1P = 0xFFFFFFFF;
    mMuonTrThetaX = 0xFFFFFFFF;
    mMuonTrThetaY = 0xFFFFFFFF;
    mMuonTrZmu = 0xFFFFFFFF;
    mMuonTrBend = 0xFFFFFFFF;
    mMuonTrNonBend = 0xFFFFFFFF;
    mMuonTrCov = 0xFFFFFFFF;
    mMuonCl = 0xFFFFFFFF;
    mMuonClErr = 0xFFFFFFFF;
    mV0Time = 0xFFFFFFFF;
    mFDDTime = 0xFFFFFFFF;
    mT0Time = 0xFFFFFFFF;
    mV0Amplitude = 0xFFFFFFFF;
    mFDDAmplitude = 0xFFFFFFFF;
    mT0Amplitude = 0xFFFFFFFF;
  }
  // Needed by MCH track extrapolation
  o2::base::GeometryManager::loadGeometry();

  mTimer.Reset();
}

void AODProducerWorkflowDPL::run(ProcessingContext& pc)
{
  mTimer.Start(false);

  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest);

  auto primVertices = recoData.getPrimaryVertices();
  auto primVer2TRefs = recoData.getPrimaryVertexMatchedTrackRefs();
  auto primVerGIs = recoData.getPrimaryVertexMatchedTracks();
  auto primVerLabels = recoData.getPrimaryVertexMCLabels();

  auto secVertices = recoData.getV0s();
  auto cascades = recoData.getCascades();

  auto fddChData = recoData.getFDDChannelsData();
  auto fddRecPoints = recoData.getFDDRecPoints();
  auto ft0ChData = recoData.getFT0ChannelsData();
  auto ft0RecPoints = recoData.getFT0RecPoints();
  auto fv0ChData = recoData.getFV0ChannelsData();
  auto fv0RecPoints = recoData.getFV0RecPoints();

  LOG(DEBUG) << "FOUND " << primVertices.size() << " primary vertices";
  LOG(DEBUG) << "FOUND " << ft0RecPoints.size() << " FT0 rec. points";
  LOG(DEBUG) << "FOUND " << fv0RecPoints.size() << " FV0 rec. points";
  LOG(DEBUG) << "FOUND " << fddRecPoints.size() << " FDD rec. points";

  auto& bcBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "BC"});
  auto& cascadesBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "CASCADE"});
  auto& collisionsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "COLLISION"});
  auto& fddBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FDD"});
  auto& ft0Builder = pc.outputs().make<TableBuilder>(Output{"AOD", "FT0"});
  auto& fv0aBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FV0A"});
  auto& fv0cBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FV0C"});
  auto& fwdTracksBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FWDTRACK"});
  auto& mcColLabelsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCCOLLISIONLABEL"});
  auto& mcCollisionsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCCOLLISION"});
  auto& mcMFTTrackLabelBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCMFTTRACKLABEL"});
  auto& mcFwdTrackLabelBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCFWDTRACKLABEL"});
  auto& mcParticlesBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCPARTICLE"});
  auto& mcTrackLabelBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCTRACKLABEL"});
  auto& mftTracksBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MFTTRACK"});
  auto& tracksBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "TRACK"});
  auto& tracksCovBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "TRACKCOV"});
  auto& tracksExtraBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "TRACKEXTRA"});
  auto& v0sBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "V0S"});
  auto& zdcBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "ZDC"});

  auto bcCursor = bcBuilder.cursor<o2::aod::BCs>();
  auto cascadesCursor = cascadesBuilder.cursor<o2::aod::StoredCascades>();
  auto collisionsCursor = collisionsBuilder.cursor<o2::aod::Collisions>();
  auto fddCursor = fddBuilder.cursor<o2::aod::FDDs>();
  auto ft0Cursor = ft0Builder.cursor<o2::aod::FT0s>();
  auto fv0aCursor = fv0aBuilder.cursor<o2::aod::FV0As>();
  auto fv0cCursor = fv0cBuilder.cursor<o2::aod::FV0Cs>();
  auto fwdTracksCursor = fwdTracksBuilder.cursor<o2::aodproducer::FwdTracksTable>();
  auto mcColLabelsCursor = mcColLabelsBuilder.cursor<o2::aod::McCollisionLabels>();
  auto mcCollisionsCursor = mcCollisionsBuilder.cursor<o2::aod::McCollisions>();
  auto mcMFTTrackLabelCursor = mcMFTTrackLabelBuilder.cursor<o2::aod::McMFTTrackLabels>();
  auto mcFwdTrackLabelCursor = mcFwdTrackLabelBuilder.cursor<o2::aod::McFwdTrackLabels>();
  auto mcParticlesCursor = mcParticlesBuilder.cursor<o2::aodproducer::MCParticlesTable>();
  auto mcTrackLabelCursor = mcTrackLabelBuilder.cursor<o2::aod::McTrackLabels>();
  auto mftTracksCursor = mftTracksBuilder.cursor<o2::aodproducer::MFTTracksTable>();
  auto tracksCovCursor = tracksCovBuilder.cursor<o2::aodproducer::TracksCovTable>();
  auto tracksCursor = tracksBuilder.cursor<o2::aodproducer::TracksTable>();
  auto tracksExtraCursor = tracksExtraBuilder.cursor<o2::aodproducer::TracksExtraTable>();
  auto v0sCursor = v0sBuilder.cursor<o2::aod::StoredV0s>();
  auto zdcCursor = zdcBuilder.cursor<o2::aod::Zdcs>();

  o2::steer::MCKinematicsReader mcReader("collisioncontext.root");
  const auto mcContext = mcReader.getDigitizationContext();
  const auto& mcRecords = mcContext->getEventRecords();
  const auto& mcParts = mcContext->getEventParts();

  LOG(DEBUG) << "FOUND " << mcRecords.size() << " records";
  LOG(DEBUG) << "FOUND " << mcParts.size() << " parts";

  std::map<uint64_t, int> bcsMap;
  collectBCs(fddRecPoints, ft0RecPoints, fv0RecPoints, primVertices, mcRecords, bcsMap);

  const auto* dh = o2::header::get<o2::header::DataHeader*>(pc.inputs().getFirstValid(true).header);
  o2::InteractionRecord startIR = {0, dh->firstTForbit};

  uint64_t tfNumber;
  // default dummy run number
  int runNumber = 244918; // TODO: get real run number
  if (mTFNumber == -1L) {
    tfNumber = getTFNumber(startIR, runNumber);
  } else {
    tfNumber = mTFNumber;
  }

  uint64_t dummyBC = 0;
  float dummyTime = 0.f;
  uint8_t dummyTriggerMask = 0;

  int nFV0ChannelsAside = o2::fv0::Geometry::getNumberOfReadoutChannels();
  std::vector<float> vFV0Amplitudes(nFV0ChannelsAside, 0.);
  for (auto& fv0RecPoint : fv0RecPoints) {
    const auto channelData = fv0RecPoint.getBunchChannelData(fv0ChData);
    for (auto& channel : channelData) {
      vFV0Amplitudes[channel.channel] = channel.charge; // amplitude, mV
    }
    float aAmplitudesA[nFV0ChannelsAside];
    for (int i = 0; i < nFV0ChannelsAside; i++) {
      aAmplitudesA[i] = truncateFloatFraction(vFV0Amplitudes[i], mV0Amplitude);
    }
    uint64_t bc = fv0RecPoint.getInteractionRecord().toLong();
    auto item = bcsMap.find(bc);
    int bcID = -1;
    if (item != bcsMap.end()) {
      bcID = item->second;
    } else {
      LOG(FATAL) << "Error: could not find a corresponding BC ID for a FV0 rec. point; BC = " << bc;
    }
    fv0aCursor(0,
               bcID,
               aAmplitudesA,
               truncateFloatFraction(fv0RecPoint.getCollisionGlobalMeanTime() * 1E-3, mV0Time), // ps to ns
               fv0RecPoint.getTrigger().triggerSignals);
  }

  float dummyFV0AmplC[32] = {0.};
  fv0cCursor(0,
             dummyBC,
             dummyFV0AmplC,
             dummyTime);

  float dummyEnergyZEM1 = 0;
  float dummyEnergyZEM2 = 0;
  float dummyEnergyCommonZNA = 0;
  float dummyEnergyCommonZNC = 0;
  float dummyEnergyCommonZPA = 0;
  float dummyEnergyCommonZPC = 0;
  float dummyEnergySectorZNA[4] = {0.};
  float dummyEnergySectorZNC[4] = {0.};
  float dummyEnergySectorZPA[4] = {0.};
  float dummyEnergySectorZPC[4] = {0.};
  zdcCursor(0,
            dummyBC,
            dummyEnergyZEM1,
            dummyEnergyZEM2,
            dummyEnergyCommonZNA,
            dummyEnergyCommonZNC,
            dummyEnergyCommonZPA,
            dummyEnergyCommonZPC,
            dummyEnergySectorZNA,
            dummyEnergySectorZNC,
            dummyEnergySectorZPA,
            dummyEnergySectorZPC,
            dummyTime,
            dummyTime,
            dummyTime,
            dummyTime,
            dummyTime,
            dummyTime);

  // TODO: figure out collision weight
  // keep track event/source id for each mc-collision
  std::vector<std::pair<int, int>> mcColToEvSrc;

  float mcColWeight = 1.;
  // filling mcCollision table
  int index = 0;
  for (auto& rec : mcRecords) {
    auto time = rec.getTimeNS();
    uint64_t globalBC = rec.toLong();
    auto item = bcsMap.find(globalBC);
    int bcID = -1;
    if (item != bcsMap.end()) {
      bcID = item->second;
    } else {
      LOG(FATAL) << "Error: could not find a corresponding BC ID for MC collision; BC = " << globalBC << ", index = " << index;
    }
    auto& colParts = mcParts[index];
    for (auto colPart : colParts) {
      auto eventID = colPart.entryID;
      auto sourceID = colPart.sourceID;
      // FIXME:
      // use generators' names for generatorIDs (?)
      short generatorID = sourceID;
      auto& header = mcReader.getMCEventHeader(sourceID, eventID);
      mcCollisionsCursor(0,
                         bcID,
                         generatorID,
                         truncateFloatFraction(header.GetX(), mCollisionPosition),
                         truncateFloatFraction(header.GetY(), mCollisionPosition),
                         truncateFloatFraction(header.GetZ(), mCollisionPosition),
                         truncateFloatFraction(time, mCollisionPosition),
                         truncateFloatFraction(mcColWeight, mCollisionPosition),
                         header.GetB());
      mcColToEvSrc.emplace_back(std::pair<int, int>(eventID, sourceID));
    }
    index++;
  }

  // vector of FDD amplitudes
  int nFDDChannels = o2::fdd::Nchannels;
  std::vector<float> vFDDAmplitudes(nFDDChannels, 0.);
  // filling FDD table
  for (const auto& fddRecPoint : fddRecPoints) {
    const auto channelData = fddRecPoint.getBunchChannelData(fddChData);
    // TODO: switch to calibrated amplitude
    for (const auto& channel : channelData) {
      vFDDAmplitudes[channel.mPMNumber] = channel.mChargeADC; // amplitude, mV
    }
    float aFDDAmplitudesA[int(nFDDChannels * 0.5)];
    float aFDDAmplitudesC[int(nFDDChannels * 0.5)];
    for (int i = 0; i < nFDDChannels; i++) {
      if (i < nFDDChannels * 0.5) {
        aFDDAmplitudesC[i] = truncateFloatFraction(vFDDAmplitudes[i], mFDDAmplitude);
      } else {
        aFDDAmplitudesA[i - int(nFDDChannels * 0.5)] = truncateFloatFraction(vFDDAmplitudes[i], mFDDAmplitude);
      }
    }
    uint64_t globalBC = fddRecPoint.getInteractionRecord().toLong();
    uint64_t bc = globalBC;
    auto item = bcsMap.find(bc);
    int bcID = -1;
    if (item != bcsMap.end()) {
      bcID = item->second;
    } else {
      LOG(FATAL) << "Error: could not find a corresponding BC ID for a FDD rec. point; BC = " << bc;
    }
    fddCursor(0,
              bcID,
              aFDDAmplitudesA,
              aFDDAmplitudesC,
              truncateFloatFraction(fddRecPoint.getCollisionTimeA() * 1E-3, mFDDTime), // ps to ns
              truncateFloatFraction(fddRecPoint.getCollisionTimeC() * 1E-3, mFDDTime), // ps to ns
              fddRecPoint.getTrigger().triggersignals);
  }

  // vector of FT0 amplitudes
  int nFT0Channels = o2::ft0::Geometry::Nsensors;
  int nFT0ChannelsAside = o2::ft0::Geometry::NCellsA * 4;
  std::vector<float> vAmplitudes(nFT0Channels, 0.);
  // filling FT0 table
  for (auto& ft0RecPoint : ft0RecPoints) {
    const auto channelData = ft0RecPoint.getBunchChannelData(ft0ChData);
    // TODO: switch to calibrated amplitude
    for (auto& channel : channelData) {
      vAmplitudes[channel.ChId] = channel.QTCAmpl; // amplitude, mV
    }
    float aAmplitudesA[nFT0ChannelsAside];
    float aAmplitudesC[nFT0Channels - nFT0ChannelsAside];
    for (int i = 0; i < nFT0Channels; i++) {
      if (i < nFT0ChannelsAside) {
        aAmplitudesA[i] = truncateFloatFraction(vAmplitudes[i], mT0Amplitude);
      } else {
        aAmplitudesC[i - nFT0ChannelsAside] = truncateFloatFraction(vAmplitudes[i], mT0Amplitude);
      }
    }
    uint64_t globalBC = ft0RecPoint.getInteractionRecord().toLong();
    uint64_t bc = globalBC;
    auto item = bcsMap.find(bc);
    int bcID = -1;
    if (item != bcsMap.end()) {
      bcID = item->second;
    } else {
      LOG(FATAL) << "Error: could not find a corresponding BC ID for a FT0 rec. point; BC = " << bc;
    }
    ft0Cursor(0,
              bcID,
              aAmplitudesA,
              aAmplitudesC,
              truncateFloatFraction(ft0RecPoint.getCollisionTimeA() * 1E-3, mT0Time), // ps to ns
              truncateFloatFraction(ft0RecPoint.getCollisionTimeC() * 1E-3, mT0Time), // ps to ns
              ft0RecPoint.getTrigger().triggersignals);
  }

  // filling MC collision labels
  for (auto& label : primVerLabels) {
    auto it = std::find_if(mcColToEvSrc.begin(), mcColToEvSrc.end(),
                           [&label](const std::pair<int, int>& item) { return (item.first == label.getEventID() && item.second == label.getSourceID()); });
    int32_t mcCollisionID = it - mcColToEvSrc.begin();
    uint16_t mcMask = 0; // todo: set mask using normalized weights?
    mcColLabelsCursor(0, mcCollisionID, mcMask);
  }

  // hash map for track indices of secondary vertices
  std::unordered_map<int, int> v0sIndices;

  // filling unassigned tracks first
  // so that all unassigned tracks are stored in the beginning of the table together
  auto& trackRef = primVer2TRefs.back(); // references to unassigned tracks are at the end
  // fixme: interaction time is undefined for unassigned tracks (?)
  fillTrackTablesPerCollision(-1, -1, trackRef, primVerGIs, recoData, tracksCursor, tracksCovCursor, tracksExtraCursor, mftTracksCursor, fwdTracksCursor, dataformats::PrimaryVertex{});

  // filling collisions and tracks into tables
  int collisionID = 0;
  for (auto& vertex : primVertices) {
    auto& cov = vertex.getCov();
    auto& timeStamp = vertex.getTimeStamp();
    const double interactionTime = timeStamp.getTimeStamp() * 1E3; // mus to ns
    uint64_t globalBC = relativeTime_to_GlobalBC(interactionTime);
    LOG(DEBUG) << globalBC << " " << interactionTime;
    // collision timestamp in ns wrt the beginning of collision BC
    const float relInteractionTime = static_cast<float>(globalBC * o2::constants::lhc::LHCBunchSpacingNS - interactionTime);
    auto item = bcsMap.find(globalBC);
    int bcID = -1;
    if (item != bcsMap.end()) {
      bcID = item->second;
    } else {
      LOG(FATAL) << "Error: could not find a corresponding BC ID for a collision; BC = " << globalBC << ", collisionID = " << collisionID;
    }

    collisionsCursor(0,
                     bcID,
                     truncateFloatFraction(vertex.getX(), mCollisionPosition),
                     truncateFloatFraction(vertex.getY(), mCollisionPosition),
                     truncateFloatFraction(vertex.getZ(), mCollisionPosition),
                     truncateFloatFraction(cov[0], mCollisionPositionCov),
                     truncateFloatFraction(cov[1], mCollisionPositionCov),
                     truncateFloatFraction(cov[2], mCollisionPositionCov),
                     truncateFloatFraction(cov[3], mCollisionPositionCov),
                     truncateFloatFraction(cov[4], mCollisionPositionCov),
                     truncateFloatFraction(cov[5], mCollisionPositionCov),
                     vertex.getFlags(),
                     truncateFloatFraction(vertex.getChi2(), mCollisionPositionCov),
                     vertex.getNContributors(),
                     truncateFloatFraction(relInteractionTime, mCollisionPosition),
                     truncateFloatFraction(timeStamp.getTimeStampError() * 1E3, mCollisionPositionCov));
    auto& trackRef = primVer2TRefs[collisionID];
    // passing interaction time in [ps]
    fillTrackTablesPerCollision(collisionID, interactionTime * 1E3, trackRef, primVerGIs, recoData, tracksCursor, tracksCovCursor, tracksExtraCursor, mftTracksCursor, fwdTracksCursor, vertex);
    collisionID++;
  }

  // filling v0s table
  for (auto& svertex : secVertices) {
    auto trPosID = svertex.getProngID(0);
    auto trNegID = svertex.getProngID(1);
    int posTableIdx = -1;
    int negTableIdx = -1;
    auto item = mGIDToTableID.find(trPosID);
    if (item != mGIDToTableID.end()) {
      posTableIdx = item->second;
    } else {
      LOG(FATAL) << "Could not find a positive track index";
    }
    item = mGIDToTableID.find(trNegID);
    if (item != mGIDToTableID.end()) {
      negTableIdx = item->second;
    } else {
      LOG(FATAL) << "Could not find a negative track index";
    }
    v0sCursor(0, posTableIdx, negTableIdx);
  }

  // filling cascades table
  for (auto& cascade : cascades) {
    auto bachelorID = cascade.getBachelorID();
    int bachTableIdx = -1;
    auto item = mGIDToTableID.find(bachelorID);
    if (item != mGIDToTableID.end()) {
      bachTableIdx = item->second;
    } else {
      LOG(FATAL) << "Could not find a bachelor track index";
    }
    cascadesCursor(0, cascade.getV0ID(), bachTableIdx);
  }

  mTableTrID = 0;
  mGIDToTableID.clear();

  // filling BC table
  // TODO: get real triggerMask
  uint64_t triggerMask = 1;
  for (auto& item : bcsMap) {
    uint64_t bc = item.first;
    bcCursor(0,
             runNumber,
             bc,
             triggerMask);
  }

  bcsMap.clear();

  // filling mc particles table
  fillMCParticlesTable(mcReader,
                       mcParticlesCursor,
                       primVer2TRefs,
                       primVerGIs,
                       recoData,
                       mcColToEvSrc);

  // ------------------------------------------------------
  // filling track labels

  // need to go through labels in the same order as for tracks
  fillMCTrackLabelsTable(mcTrackLabelCursor, mcMFTTrackLabelCursor, mcFwdTrackLabelCursor, primVer2TRefs.back(), primVerGIs, recoData);
  for (int iref = 0; iref < primVer2TRefs.size() - 1; iref++) {
    auto& trackRef = primVer2TRefs[iref];
    fillMCTrackLabelsTable(mcTrackLabelCursor, mcMFTTrackLabelCursor, mcFwdTrackLabelCursor, trackRef, primVerGIs, recoData);
  }

  mToStore.clear();

  pc.outputs().snapshot(Output{"TFN", "TFNumber", 0, Lifetime::Timeframe}, tfNumber);

  mTimer.Stop();
}

void AODProducerWorkflowDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "aod producer dpl total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getAODProducerWorkflowSpec(GID::mask_t src, bool useMC)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  dataRequest->requestTracks(src, useMC);
  dataRequest->requestPrimaryVertertices(useMC);
  dataRequest->requestSecondaryVertertices(useMC);
  dataRequest->requestFT0RecPoints(false);
  dataRequest->requestFV0RecPoints(false);
  dataRequest->requestFDDRecPoints(false);
  dataRequest->requestClusters(GIndex::getSourcesMask("TPC,TOF"), false);

  outputs.emplace_back(OutputLabel{"O2bc"}, "AOD", "BC", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2cascade"}, "AOD", "CASCADE", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2collision"}, "AOD", "COLLISION", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2fdd"}, "AOD", "FDD", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2ft0"}, "AOD", "FT0", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2fv0a"}, "AOD", "FV0A", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2fv0c"}, "AOD", "FV0C", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2fwdtrack"}, "AOD", "FWDTRACK", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mccollision"}, "AOD", "MCCOLLISION", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mccollisionlabel"}, "AOD", "MCCOLLISIONLABEL", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mcmfttracklabel"}, "AOD", "MCMFTTRACKLABEL", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mcfwdtracklabel"}, "AOD", "MCFWDTRACKLABEL", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mcparticle"}, "AOD", "MCPARTICLE", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mctracklabel"}, "AOD", "MCTRACKLABEL", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mfttrack"}, "AOD", "MFTTRACK", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2track"}, "AOD", "TRACK", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2trackcov"}, "AOD", "TRACKCOV", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2trackextra"}, "AOD", "TRACKEXTRA", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2v0"}, "AOD", "V0S", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2zdc"}, "AOD", "ZDC", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputSpec{"TFN", "TFNumber"});

  return DataProcessorSpec{
    "aod-producer-workflow",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<AODProducerWorkflowDPL>(src, dataRequest)},
    Options{
      ConfigParamSpec{"aod-timeframe-id", VariantType::Int64, -1L, {"Set timeframe number"}},
      ConfigParamSpec{"enable-truncation", VariantType::Int, 1, {"Truncation parameter: 1 -- on, != 1 -- off"}},
      ConfigParamSpec{"reco-mctracks-only", VariantType::Int, 0, {"Store only reconstructed MC tracks and their mothers/daughters. 0 -- off, != 0 -- on"}}}};
}

} // namespace o2::aodproducer
