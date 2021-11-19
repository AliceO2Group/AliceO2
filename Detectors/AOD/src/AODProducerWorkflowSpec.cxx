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
#include "O2Version.h"
#include "TMath.h"
#include "MathUtils/Utils.h"
#include "Math/SMatrix.h"
#include "TMatrixD.h"
#include "TString.h"
#include "TObjString.h"
#include <map>
#include <unordered_map>
#include <vector>

using namespace o2::framework;
using namespace o2::math_utils::detail;
using PVertex = o2::dataformats::PrimaryVertex;
using GIndex = o2::dataformats::VtxTrackIndex;
using DataRequest = o2::globaltracking::DataRequest;
using GID = o2::dataformats::GlobalTrackID;
using SMatrix55Sym = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;

namespace o2::aodproducer
{

void AODProducerWorkflowDPL::collectBCs(gsl::span<const o2::fdd::RecPoint>& fddRecPoints,
                                        gsl::span<const o2::ft0::RecPoints>& ft0RecPoints,
                                        gsl::span<const o2::fv0::RecPoints>& fv0RecPoints,
                                        gsl::span<const o2::dataformats::PrimaryVertex>& primVertices,
                                        gsl::span<const o2::emcal::TriggerRecord>& caloEMCCellsTRGR,
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

  for (auto& emcaltrg : caloEMCCellsTRGR) {
    uint64_t globalBC = emcaltrg.getBCData().toLong();
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
  float sY = TMath::Sqrt(track.getSigmaY2()), sZ = TMath::Sqrt(track.getSigmaZ2()), sSnp = TMath::Sqrt(track.getSigmaSnp2()),
        sTgl = TMath::Sqrt(track.getSigmaTgl2()), sQ2Pt = TMath::Sqrt(track.getSigma1Pt2());
  tracksCovCursor(0,
                  truncateFloatFraction(sY, mTrackCovDiag),
                  truncateFloatFraction(sZ, mTrackCovDiag),
                  truncateFloatFraction(sSnp, mTrackCovDiag),
                  truncateFloatFraction(sTgl, mTrackCovDiag),
                  truncateFloatFraction(sQ2Pt, mTrackCovDiag),
                  (Char_t)(128. * track.getSigmaZY() / (sZ * sY)),
                  (Char_t)(128. * track.getSigmaSnpY() / (sSnp * sY)),
                  (Char_t)(128. * track.getSigmaSnpZ() / (sSnp * sZ)),
                  (Char_t)(128. * track.getSigmaTglY() / (sTgl * sY)),
                  (Char_t)(128. * track.getSigmaTglZ() / (sTgl * sZ)),
                  (Char_t)(128. * track.getSigmaTglSnp() / (sTgl * sSnp)),
                  (Char_t)(128. * track.getSigma1PtY() / (sQ2Pt * sY)),
                  (Char_t)(128. * track.getSigma1PtZ() / (sQ2Pt * sZ)),
                  (Char_t)(128. * track.getSigma1PtSnp() / (sQ2Pt * sSnp)),
                  (Char_t)(128. * track.getSigma1PtTgl() / (sQ2Pt * sTgl)));
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

template <typename TracksCursorType, typename TracksCovCursorType, typename TracksExtraCursorType, typename MftTracksCursorType, typename FwdTracksCursorType, typename FwdTracksCovCursorType>
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
                                                         FwdTracksCovCursorType& fwdTracksCovCursor,
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
              addToFwdTracksTable(fwdTracksCursor, fwdTracksCovCursor, track, -1, vertex);
            }
          } else {
            math_utils::Point3D<double> vtx{vertex.getX(),
                                            vertex.getY(), vertex.getZ()};
            addToFwdTracksTable(fwdTracksCursor, fwdTracksCovCursor, track, collisionID, vtx);
          }
        } else if (src == GIndex::Source::MFTMCH) {
          const auto& track = data.getGlobalFwdTrack(trackIndex);
          addToFwdTracksTable(fwdTracksCursor, fwdTracksCovCursor, track, collisionID, {0, 0, 0});
        } else {
          // normal tracks table
          auto contributorsGID = data.getSingleDetectorRefs(trackIndex);
          const auto& trackPar = data.getTrackParam(trackIndex);
          if (contributorsGID[GIndex::Source::ITS].isIndexSet()) {
            int nClusters = itsTracks[contributorsGID[GIndex::ITS].getIndex()].getNClusters();
            float chi2 = itsTracks[contributorsGID[GIndex::ITS].getIndex()].getChi2();
            extraInfoHolder.itsChi2NCl = nClusters != 0 ? chi2 / (float)nClusters : 0;
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

          // set bit encoding for PVContributor property as part of the flag field
          if (trackIndex.isPVContributor()) {
            extraInfoHolder.flags |= o2::aod::track::PVContributor;
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

template <typename FwdTracksCursorType, typename FwdTracksCovCursorType, typename fwdTrackType>
void AODProducerWorkflowDPL::addToFwdTracksTable(FwdTracksCursorType& fwdTracksCursor, FwdTracksCovCursorType& fwdTracksCovCursor,
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

  float sigX = 0;
  float sigY = 0;
  float sigPhi = 0;
  float sigTgl = 0;
  float sig1Pt = 0;

  int8_t rhoXY = 0;
  int8_t rhoPhiX = 0;
  int8_t rhoPhiY = 0;
  int8_t rhoTglX = 0;
  int8_t rhoTglY = 0;
  int8_t rhoTglPhi = 0;
  int8_t rho1PtX = 0;
  int8_t rho1PtY = 0;
  int8_t rho1PtPhi = 0;
  int8_t rho1PtTgl = 0;

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

    sigX = TMath::Sqrt(trackParamAtVertex.getCovariances()(0, 0));
    sigY = TMath::Sqrt(trackParamAtVertex.getCovariances()(1, 1));
    sigPhi = TMath::Sqrt(trackParamAtVertex.getCovariances()(2, 2));
    sigTgl = TMath::Sqrt(trackParamAtVertex.getCovariances()(3, 3));
    sig1Pt = TMath::Sqrt(trackParamAtVertex.getCovariances()(4, 4));
    rhoXY = (Char_t)(128. * trackParamAtVertex.getCovariances()(0, 1) / (sigX * sigY));
    rhoPhiX = (Char_t)(128. * trackParamAtVertex.getCovariances()(0, 2) / (sigPhi * sigX));
    rhoPhiY = (Char_t)(128. * trackParamAtVertex.getCovariances()(1, 2) / (sigPhi * sigY));
    rhoTglX = (Char_t)(128. * trackParamAtVertex.getCovariances()(0, 3) / (sigTgl * sigX));
    rhoTglY = (Char_t)(128. * trackParamAtVertex.getCovariances()(1, 3) / (sigTgl * sigY));
    rhoTglPhi = (Char_t)(128. * trackParamAtVertex.getCovariances()(2, 3) / (sigTgl * sigPhi));
    rho1PtX = (Char_t)(128. * trackParamAtVertex.getCovariances()(0, 4) / (sig1Pt * sigX));
    rho1PtY = (Char_t)(128. * trackParamAtVertex.getCovariances()(1, 4) / (sig1Pt * sigY));
    rho1PtPhi = (Char_t)(128. * trackParamAtVertex.getCovariances()(2, 4) / (sig1Pt * sigPhi));
    rho1PtTgl = (Char_t)(128. * trackParamAtVertex.getCovariances()(3, 4) / (sig1Pt * sigTgl));

  } else {
    // This is a GlobalMuonTrack or a GlobalForwardTrack
    x = track.getX();
    y = track.getY();
    z = track.getZ();
    phi = track.getPhi();
    tanl = track.getTanl();
    invqpt = track.getInvQPt();
    chi2 = track.getTrackChi2();
    // nClusters = track.getNumberOfPoints();
    chi2matchmchmid = track.getMIDMatchingChi2();
    chi2matchmchmft = track.getMatchingChi2();
    matchmfttrackid = track.getMFTTrackID();
    matchmchtrackid = track.getMCHTrackID();

    sigX = TMath::Sqrt(track.getCovariances()(0, 0));
    sigY = TMath::Sqrt(track.getCovariances()(1, 1));
    sigPhi = TMath::Sqrt(track.getCovariances()(2, 2));
    sigTgl = TMath::Sqrt(track.getCovariances()(3, 3));
    sig1Pt = TMath::Sqrt(track.getCovariances()(4, 4));
    rhoXY = (Char_t)(128. * track.getCovariances()(0, 1) / (sigX * sigY));
    rhoPhiX = (Char_t)(128. * track.getCovariances()(0, 2) / (sigPhi * sigX));
    rhoPhiY = (Char_t)(128. * track.getCovariances()(1, 2) / (sigPhi * sigY));
    rhoTglX = (Char_t)(128. * track.getCovariances()(0, 3) / (sigTgl * sigX));
    rhoTglY = (Char_t)(128. * track.getCovariances()(1, 3) / (sigTgl * sigY));
    rhoTglPhi = (Char_t)(128. * track.getCovariances()(2, 3) / (sigTgl * sigPhi));
    rho1PtX = (Char_t)(128. * track.getCovariances()(0, 4) / (sig1Pt * sigX));
    rho1PtY = (Char_t)(128. * track.getCovariances()(1, 4) / (sig1Pt * sigY));
    rho1PtPhi = (Char_t)(128. * track.getCovariances()(2, 4) / (sig1Pt * sigPhi));
    rho1PtTgl = (Char_t)(128. * track.getCovariances()(3, 4) / (sig1Pt * sigTgl));

    trackTypeId = (chi2matchmchmid >= 0) ? o2::aod::fwdtrack::GlobalMuonTrack : o2::aod::fwdtrack::GlobalForwardTrack;
  }

  auto covmat = track.getCovariances();

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

  fwdTracksCovCursor(0,
                     sigX,
                     sigY,
                     sigPhi,
                     sigTgl,
                     sig1Pt,
                     rhoXY,
                     rhoPhiX,
                     rhoPhiY,
                     rhoTglX,
                     rhoTglY,
                     rhoTglPhi,
                     rho1PtX,
                     rho1PtY,
                     rho1PtPhi,
                     rho1PtTgl);
}

template <typename MCParticlesCursorType>
void AODProducerWorkflowDPL::fillMCParticlesTable(o2::steer::MCKinematicsReader& mcReader,
                                                  const MCParticlesCursorType& mcParticlesCursor,
                                                  gsl::span<const o2::dataformats::VtxTrackRef>& primVer2TRefs,
                                                  gsl::span<const GIndex>& GIndices,
                                                  o2::globaltracking::RecoContainer& data,
                                                  std::map<std::pair<int, int>, int> const& mcColToEvSrc)
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
  for (auto& colInfo : mcColToEvSrc) { // loop over "<eventID, sourceID> <-> combined MC col. ID" key pairs
    int event = colInfo.first.first;
    int source = colInfo.first.second;
    int mcColId = colInfo.second;
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
          mToStore[Triplet_t(source, event, mother1)] = 1;
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
      if (!mcParticles[particle].isPrimary()) {
        flags |= o2::aod::mcparticle::enums::ProducedByTransport; // mark as produced by transport
        statusCode = mcParticles[particle].getProcess();
      } else {
        statusCode = mcParticles[particle].getStatusCode();
      }
      if (source == 0) {
        flags |= o2::aod::mcparticle::enums::FromBackgroundEvent; // mark as particle from background event
      }
      if (mcParticles[particle].isPrimary()) {
        flags |= o2::aod::mcparticle::enums::PhysicalPrimary; // mark as physical primary
      }
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
      auto pX = (float)mcParticles[particle].Px();
      auto pY = (float)mcParticles[particle].Py();
      auto pZ = (float)mcParticles[particle].Pz();
      auto energy = (float)mcParticles[particle].GetEnergy();
      mcParticlesCursor(0,
                        mcColId,
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

// fill calo related tables (cells and calotrigger table)
// currently hardcoded for EMCal, can be expanded for PHOS
template <typename TCaloCells, typename TCaloTriggerRecord, typename TCaloCursor, typename TCaloTRGTableCursor>
void AODProducerWorkflowDPL::fillCaloTable(const TCaloCells& calocells, const TCaloTriggerRecord& caloCellTRGR, const TCaloCursor& caloCellCursor,
                                           const TCaloTRGTableCursor& caloCellTRGTableCursor, std::map<uint64_t, int>& bcsMap)
{
  uint64_t globalBC = 0;    // global BC ID
  uint64_t globalBCRel = 0; // BC id reltive to minGlBC (from FIT)

  // get cell belonging to an eveffillnt instead of timeframe
  mCaloEventHandler->reset();
  mCaloEventHandler->setCellData(calocells, caloCellTRGR);

  // loop over events
  for (int iev = 0; iev < mCaloEventHandler->getNumberOfEvents(); iev++) {
    o2::emcal::EventData inputEvent = mCaloEventHandler->buildEvent(iev);
    auto cellsInEvent = inputEvent.mCells;                  // get cells belonging to current event
    auto interactionRecord = inputEvent.mInteractionRecord; // get interaction records belonging to current event

    // Convert bc to global bc relative to min global BC found for all primary verteces in timeframe
    // minGlBC and maxGlBC are set in findMinMaxBc(...)
    globalBC = interactionRecord.toLong();

    // check with Markus if globalBC ID is needed or globalBC - minGlBC
    // in case of collision vertex what is used is
    // uint64_t globalBC = std::round(tsTimeStamp / o2::constants::lhc::LHCBunchSpacingNS);
    auto item = bcsMap.find(globalBC);
    int bcID = -1;
    if (item != bcsMap.end()) {
      bcID = item->second;
    } else {
      LOG(FATAL) << "Error: could not find a corresponding BC ID for a EMCal point; globalBC = " << globalBC;
    }

    // loop over all cells in collision
    for (auto& cell : cellsInEvent) {

      // fill table
      caloCellCursor(0,
                     bcID,
                     cell.getTower(),
                     truncateFloatFraction(cell.getAmplitude(), mCaloAmp),
                     truncateFloatFraction(cell.getTimeStamp(), mCaloTime),
                     cell.getType(),
                     1); // hard coded for emcal (-1 would be undefined, 0 phos)

      // once decided on final form, fill calotrigger table here:

      // ...
    }

    // todo: fill with actual values once decided
    caloCellTRGTableCursor(0,
                           bcID,
                           0,  // fastOrAbsId (dummy value)
                           0., // lnAmplitude (dummy value)
                           0,  // triggerBits (dummy value)
                           1); // caloType (dummy value)
  }
}

void AODProducerWorkflowDPL::init(InitContext& ic)
{
  mTimer.Stop();
  mLPMProdTag = ic.options().get<string>("lpmp-prod-tag");
  mAnchorPass = ic.options().get<string>("anchor-pass");
  mAnchorProd = ic.options().get<string>("anchor-prod");
  mRecoPass = ic.options().get<string>("reco-pass");
  mTFNumber = ic.options().get<int64_t>("aod-timeframe-id");
  mRecoOnly = ic.options().get<int>("reco-mctracks-only");
  mTruncate = ic.options().get<int>("enable-truncation");
  mRunNumber = ic.options().get<int>("run-number");

  if (mTFNumber == -1L) {
    LOG(INFO) << "TFNumber will be obtained from CCDB";
  }
  if (mRunNumber == -1L) {
    LOG(INFO) << "The Run number will be obtained from DPL headers";
  }

  // create EventHandler used for calo cells
  mCaloEventHandler = new o2::emcal::EventHandler<o2::emcal::Cell>();

  // set no truncation if selected by user
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
    mCaloAmp = 0xFFFFFFFF;  // todo check which truncation should actually be used
    mCaloTime = 0xFFFFFFFF; // todo check which truncation should actually be used
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

  // writing metadata if it's not yet in AOD file
  // note: `--aod-writer-resmode "UPDATE"` has to be used,
  //       so that metadata is not overwritten
  mResFile += ".root";
  auto* fResFile = TFile::Open(mResFile, "UPDATE");
  if (!fResFile) {
    LOGF(fatal, "Could not open file %s", mResFile);
  }
  if (!fResFile->FindObjectAny("metaData")) {
    // populating metadata map
    TString dataType = mUseMC ? "MC" : "RAW";
    mMetaData.Add(new TObjString("DataType"), new TObjString(dataType));
    mMetaData.Add(new TObjString("Run"), new TObjString("3"));
    TString O2Version = o2::fullVersion();
    TString ROOTVersion = ROOT_RELEASE;
    mMetaData.Add(new TObjString("O2Version"), new TObjString(O2Version));
    mMetaData.Add(new TObjString("ROOTVersion"), new TObjString(ROOTVersion));
    mMetaData.Add(new TObjString("RecoPassName"), new TObjString(mRecoPass));
    mMetaData.Add(new TObjString("AnchorProduction"), new TObjString(mAnchorProd));
    mMetaData.Add(new TObjString("AnchorPassName"), new TObjString(mAnchorPass));
    mMetaData.Add(new TObjString("LPMProductionTag"), new TObjString(mLPMProdTag));
    LOGF(info, "Metadata: writing into %s", mResFile);
    fResFile->WriteObject(&mMetaData, "metaData");
  } else {
    LOGF(fatal, "Metadata: target file %s already has metadata", mResFile);
  }
  fResFile->Close();

  mTimer.Reset();
}

void AODProducerWorkflowDPL::run(ProcessingContext& pc)
{
  mTimer.Start(false);

  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest);
  mStartIR = recoData.startIR;

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

  // get calo information
  auto caloEMCCells = recoData.getEMCALCells();
  auto caloEMCCellsTRGR = recoData.getEMCALTriggers();

  LOG(DEBUG) << "FOUND " << primVertices.size() << " primary vertices";
  LOG(DEBUG) << "FOUND " << ft0RecPoints.size() << " FT0 rec. points";
  LOG(DEBUG) << "FOUND " << fv0RecPoints.size() << " FV0 rec. points";
  LOG(DEBUG) << "FOUND " << fddRecPoints.size() << " FDD rec. points";
  LOG(DEBUG) << "FOUND " << caloEMCCells.size() << " EMC cells";
  LOG(DEBUG) << "FOUND " << caloEMCCellsTRGR.size() << " EMC Trigger Records";

  auto& bcBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "BC"});
  auto& cascadesBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "CASCADE"});
  auto& collisionsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "COLLISION"});
  auto& fddBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FDD"});
  auto& ft0Builder = pc.outputs().make<TableBuilder>(Output{"AOD", "FT0"});
  auto& fv0aBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FV0A"});
  auto& fv0cBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FV0C"});
  auto& fwdTracksBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FWDTRACK"});
  auto& fwdTracksCovBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FWDTRACKCOV"});
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
  auto& v0sBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "V0"});
  auto& zdcBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "ZDC"});
  auto& caloCellsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "CALO"});
  auto& caloCellsTRGTableBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "CALOTRIGGER"});

  auto bcCursor = bcBuilder.cursor<o2::aod::BCs>();
  auto cascadesCursor = cascadesBuilder.cursor<o2::aod::StoredCascades>();
  auto collisionsCursor = collisionsBuilder.cursor<o2::aod::Collisions>();
  auto fddCursor = fddBuilder.cursor<o2::aod::FDDs>();
  auto ft0Cursor = ft0Builder.cursor<o2::aod::FT0s>();
  auto fv0aCursor = fv0aBuilder.cursor<o2::aod::FV0As>();
  auto fv0cCursor = fv0cBuilder.cursor<o2::aod::FV0Cs>();
  auto fwdTracksCursor = fwdTracksBuilder.cursor<o2::aodproducer::FwdTracksTable>();
  auto fwdTracksCovCursor = fwdTracksCovBuilder.cursor<o2::aodproducer::FwdTracksCovTable>();
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
  auto caloCellsCursor = caloCellsBuilder.cursor<o2::aod::Calos>();
  auto caloCellsTRGTableCursor = caloCellsTRGTableBuilder.cursor<o2::aod::CaloTriggers>();

  std::unique_ptr<o2::steer::MCKinematicsReader> mcReader;
  if (mUseMC) {
    mcReader = std::make_unique<o2::steer::MCKinematicsReader>("collisioncontext.root");
    LOG(DEBUG) << "FOUND " << mcReader->getDigitizationContext()->getEventRecords().size()
               << " records" << mcReader->getDigitizationContext()->getEventParts().size() << " parts";
  }

  std::map<uint64_t, int> bcsMap;
  collectBCs(fddRecPoints, ft0RecPoints, fv0RecPoints, primVertices, caloEMCCellsTRGR, mUseMC ? mcReader->getDigitizationContext()->getEventRecords() : std::vector<o2::InteractionTimeRecord>{}, bcsMap);
  const auto* dh = o2::header::get<o2::header::DataHeader*>(pc.inputs().getFirstValid(true).header);

  uint64_t tfNumber;
  const int runNumber = (mRunNumber == -1) ? int(dh->runNumber) : mRunNumber;
  if (mTFNumber == -1L) {
    // TODO has to use absolute time of TF
    tfNumber = uint64_t(dh->firstTForbit) + (uint64_t(dh->runNumber) << 32); // getTFNumber(mStartIR, runNumber);
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

  // keep track event/source id for each mc-collision
  // using map and not unordered_map to ensure
  // correct ordering when iterating over container elements
  std::map<std::pair<int, int>, int> mcColToEvSrc;

  if (mUseMC) {
    // TODO: figure out collision weight
    float mcColWeight = 1.;
    // filling mcCollision table
    int nMCCollisions = mcReader->getDigitizationContext()->getNCollisions();
    const auto& mcRecords = mcReader->getDigitizationContext()->getEventRecords();
    const auto& mcParts = mcReader->getDigitizationContext()->getEventParts();
    for (int iCol = 0; iCol < nMCCollisions; iCol++) {
      auto time = mcRecords[iCol].getTimeNS();
      auto globalBC = mcRecords[iCol].toLong();
      auto item = bcsMap.find(globalBC);
      int bcID = -1;
      if (item != bcsMap.end()) {
        bcID = item->second;
      } else {
        LOG(FATAL) << "Error: could not find a corresponding BC ID for MC collision; BC = " << globalBC << ", mc collision = " << iCol;
      }
      auto& colParts = mcParts[iCol];
      for (auto colPart : colParts) {
        auto eventID = colPart.entryID;
        auto sourceID = colPart.sourceID;
        if (sourceID == 0) { // embedding: using background event info
          // FIXME:
          // use generators' names for generatorIDs (?)
          short generatorID = sourceID;
          auto& header = mcReader->getMCEventHeader(sourceID, eventID);
          mcCollisionsCursor(0,
                             bcID,
                             generatorID,
                             truncateFloatFraction(header.GetX(), mCollisionPosition),
                             truncateFloatFraction(header.GetY(), mCollisionPosition),
                             truncateFloatFraction(header.GetZ(), mCollisionPosition),
                             truncateFloatFraction(time, mCollisionPosition),
                             truncateFloatFraction(mcColWeight, mCollisionPosition),
                             header.GetB());
        }
        mcColToEvSrc.emplace(std::pair<int, int>(eventID, sourceID), iCol); // point background and injected signal events to one collision
      }
    }
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

  if (mUseMC) {
    // filling MC collision labels
    for (auto& label : primVerLabels) {
      auto it = mcColToEvSrc.find(std::pair<int, int>(label.getEventID(), label.getSourceID()));
      int32_t mcCollisionID = it != mcColToEvSrc.end() ? it->second : -1;
      uint16_t mcMask = 0; // todo: set mask using normalized weights?
      mcColLabelsCursor(0, mcCollisionID, mcMask);
    }
  }

  // hash map for track indices of secondary vertices
  std::unordered_map<int, int> v0sIndices;

  // filling unassigned tracks first
  // so that all unassigned tracks are stored in the beginning of the table together
  auto& trackRef = primVer2TRefs.back(); // references to unassigned tracks are at the end
  // fixme: interaction time is undefined for unassigned tracks (?)
  fillTrackTablesPerCollision(-1, -1, trackRef, primVerGIs, recoData, tracksCursor, tracksCovCursor, tracksExtraCursor, mftTracksCursor, fwdTracksCursor, fwdTracksCovCursor, dataformats::PrimaryVertex{});

  // filling collisions and tracks into tables
  int collisionID = 0;
  for (auto& vertex : primVertices) {
    auto& cov = vertex.getCov();
    auto& timeStamp = vertex.getTimeStamp();                       // this is a relative time
    const double interactionTime = timeStamp.getTimeStamp() * 1E3; // mus to ns
    uint64_t globalBC = relativeTime_to_GlobalBC(interactionTime);
    uint64_t localBC = relativeTime_to_LocalBC(interactionTime);
    LOG(DEBUG) << "global BC " << globalBC << " local BC " << localBC << " relative interaction time " << interactionTime;
    // collision timestamp in ns wrt the beginning of collision BC
    const float relInteractionTime = static_cast<float>(localBC * o2::constants::lhc::LHCBunchSpacingNS - interactionTime);
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
    fillTrackTablesPerCollision(collisionID, interactionTime * 1E3, trackRef, primVerGIs, recoData, tracksCursor, tracksCovCursor, tracksExtraCursor, mftTracksCursor, fwdTracksCursor, fwdTracksCovCursor, vertex);
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
      LOG(WARN) << "Could not find a positive track index for prong ID " << trPosID;
    }
    item = mGIDToTableID.find(trNegID);
    if (item != mGIDToTableID.end()) {
      negTableIdx = item->second;
    } else {
      LOG(WARN) << "Could not find a negative track index for prong ID " << trNegID;
    }
    if (posTableIdx != -1 and negTableIdx != -1) {
      v0sCursor(0, posTableIdx, negTableIdx);
    }
  }

  // filling cascades table
  for (auto& cascade : cascades) {
    auto bachelorID = cascade.getBachelorID();
    int bachTableIdx = -1;
    auto item = mGIDToTableID.find(bachelorID);
    if (item != mGIDToTableID.end()) {
      bachTableIdx = item->second;
    } else {
      LOG(WARN) << "Could not find a bachelor track index";
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

  if (mInputSources[GIndex::EMC]) {
    // fill EMC cells to tables
    // TODO handle MC info
    fillCaloTable(caloEMCCells, caloEMCCellsTRGR, caloCellsCursor, caloCellsTRGTableCursor, bcsMap);
  }

  bcsMap.clear();

  if (mUseMC) {
    // filling mc particles table
    fillMCParticlesTable(*mcReader,
                         mcParticlesCursor,
                         primVer2TRefs,
                         primVerGIs,
                         recoData,
                         mcColToEvSrc);

    mcColToEvSrc.clear();

    // ------------------------------------------------------
    // filling track labels

    // need to go through labels in the same order as for tracks
    fillMCTrackLabelsTable(mcTrackLabelCursor, mcMFTTrackLabelCursor, mcFwdTrackLabelCursor, primVer2TRefs.back(), primVerGIs, recoData);
    for (int iref = 0; iref < primVer2TRefs.size() - 1; iref++) {
      auto& trackRef = primVer2TRefs[iref];
      fillMCTrackLabelsTable(mcTrackLabelCursor, mcMFTTrackLabelCursor, mcFwdTrackLabelCursor, trackRef, primVerGIs, recoData);
    }
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

DataProcessorSpec getAODProducerWorkflowSpec(GID::mask_t src, bool enableSV, bool useMC, std::string resFile)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  dataRequest->requestTracks(src, useMC);
  dataRequest->requestPrimaryVertertices(useMC);
  if (enableSV) {
    dataRequest->requestSecondaryVertertices(useMC);
  }
  if (src[GID::TPC]) {
    dataRequest->requestClusters(GIndex::getSourcesMask("TPC"), false); // TOF clusters are requested with TOF tracks
  }
  if (src[GID::EMC]) {
    dataRequest->requestEMCALCells(useMC);
  }

  outputs.emplace_back(OutputLabel{"O2bc"}, "AOD", "BC", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2cascade"}, "AOD", "CASCADE", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2collision"}, "AOD", "COLLISION", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2fdd"}, "AOD", "FDD", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2ft0"}, "AOD", "FT0", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2fv0a"}, "AOD", "FV0A", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2fv0c"}, "AOD", "FV0C", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2fwdtrack"}, "AOD", "FWDTRACK", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2fwdtrackcov"}, "AOD", "FWDTRACKCOV", 0, Lifetime::Timeframe);
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
  outputs.emplace_back(OutputLabel{"O2v0"}, "AOD", "V0", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2zdc"}, "AOD", "ZDC", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2caloCell"}, "AOD", "CALO", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2caloCellTRGR"}, "AOD", "CALOTRIGGER", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputSpec{"TFN", "TFNumber"});

  return DataProcessorSpec{
    "aod-producer-workflow",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<AODProducerWorkflowDPL>(src, dataRequest, enableSV, resFile, useMC)},
    Options{
      ConfigParamSpec{"run-number", VariantType::Int64, -1L, {"The run-number. If left default we try to get it from DPL header."}},
      ConfigParamSpec{"aod-timeframe-id", VariantType::Int64, -1L, {"Set timeframe number"}},
      ConfigParamSpec{"fill-calo-cells", VariantType::Int, 1, {"Fill calo cells into cell table"}},
      ConfigParamSpec{"enable-truncation", VariantType::Int, 1, {"Truncation parameter: 1 -- on, != 1 -- off"}},
      ConfigParamSpec{"lpmp-prod-tag", VariantType::String, "", {"LPMProductionTag"}},
      ConfigParamSpec{"anchor-pass", VariantType::String, "", {"AnchorPassName"}},
      ConfigParamSpec{"anchor-prod", VariantType::String, "", {"AnchorProduction"}},
      ConfigParamSpec{"reco-pass", VariantType::String, "", {"RecoPassName"}},
      ConfigParamSpec{"reco-mctracks-only", VariantType::Int, 0, {"Store only reconstructed MC tracks and their mothers/daughters. 0 -- off, != 0 -- on"}}}};
}

} // namespace o2::aodproducer
