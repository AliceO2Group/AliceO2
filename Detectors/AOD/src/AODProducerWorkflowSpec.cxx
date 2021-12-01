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
#include "DataFormatsCTP/Digits.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsZDC/ZDCEnergy.h"
#include "DataFormatsZDC/ZDCTDCData.h"
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
#include "ZDCBase/Constants.h"
#include "GPUTPCGMMergedTrackHit.h"
#include "O2Version.h"
#include "TMath.h"
#include "MathUtils/Utils.h"
#include "Math/SMatrix.h"
#include "TMatrixD.h"
#include "TString.h"
#include "TObjString.h"
#include <map>
#include <unordered_map>
#include <string>
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

bool AODProducerWorkflowDPL::tablesToFill[AODProducerWorkflowDPL::numTables] = {false};
void AODProducerWorkflowDPL::parseTablesList(std::string tablesList)
{
  std::string ss(tablesList);
  LOGF(info, "Input tables list = %s", tablesList);
  std::string sname{};

  if (ss.find(tablesFillAll) != std::string::npos) {
    for (int i = 0; i < numTables; i++) {
      tablesToFill[i] = true;
    }
    return;
  }
  if (ss.find(tablesFillNone) != std::string::npos) {
    for (int i = 0; i < numTables; i++) {
      tablesToFill[i] = false;
    }
    return;
  }
  std::replace(ss.begin(), ss.end(), ' ', ',');
  std::stringstream sss(ss);
  while (getline(sss, sname, ',')) {
    for (int i = 0; i < numTables; i++) {
      if (sname == tablesNames[i]) {
        tablesToFill[i] = true;
        sname = "";
        break;
      }
    }
    if (!sname.empty()) {
      throw std::runtime_error(fmt::format("Wrong entry {:s} in tables list {:s}", sname, tablesList));
    }
  }
}

void AODProducerWorkflowDPL::checkTablesDeps(bool useMC)
{
  // explicitly mark MC tables as unused if data is an input
  if (!useMC) {
    tablesToFill[O2mcparticle] = false;
    tablesToFill[O2mcfwdtracklabel] = false;
    tablesToFill[O2mcmfttracklabel] = false;
    tablesToFill[O2mccollisionlabel] = false;
    tablesToFill[O2mccollision] = false;
    tablesToFill[O2mctracklabel] = false;
  }

  // checking dependencies between tables

  // filling all barrel tracks tables if found any -- for consistency
  bool needTracks = tablesToFill[O2track] || tablesToFill[O2trackcov] || tablesToFill[O2trackextra];
  if (needTracks) {
    tablesToFill[O2track] = true;
    tablesToFill[O2trackcov] = true;
    tablesToFill[O2trackextra] = true;
  }

  // filling all fwd tracks tables if found any -- for consistency
  bool needFwdTracks = tablesToFill[O2fwdtrack] || tablesToFill[O2fwdtrackcov];
  if (needFwdTracks) {
    tablesToFill[O2fwdtrack] = true;
    tablesToFill[O2fwdtrackcov] = true;
  }

  // tracks are filled per collision, so O2collision have to be processed
  bool needAnyTracks = needTracks || needFwdTracks || tablesToFill[O2mfttrack];
  if (needAnyTracks) {
    tablesToFill[O2collision] = true;
  }

  // secondary vertices and cascades depend on barrel tracks
  bool needSVtxOrCasc = tablesToFill[O2v0] || tablesToFill[O2cascade];
  if (needSVtxOrCasc) {
    tablesToFill[O2track] = true;
    tablesToFill[O2trackcov] = true;
    tablesToFill[O2trackextra] = true;
  }

  // filling all calo tables if found any -- for consistency
  bool needCalo = tablesToFill[O2caloCell] || tablesToFill[O2caloCellTRGR];
  if (needCalo) {
    tablesToFill[O2caloCell] = true;
    tablesToFill[O2caloCellTRGR] = true;
  }

  bool needAnyTrackLabels = tablesToFill[O2mcfwdtracklabel] || tablesToFill[O2mcmfttracklabel] || tablesToFill[O2mctracklabel];
  if (needAnyTrackLabels) {
    // O2collision is needed for accessing reco tracks and filtering
    // O2mccollision is needed for correct mcCollisionIDs
    tablesToFill[O2collision] = true;
    tablesToFill[O2mccollision] = true;
    if (tablesToFill[O2mctracklabel]) {
      tablesToFill[O2track] = true;
      tablesToFill[O2trackcov] = true;
      tablesToFill[O2trackextra] = true;
    }
    if (tablesToFill[O2mcmfttracklabel]) {
      tablesToFill[O2mfttrack] = true;
    }
    if (tablesToFill[O2mcfwdtracklabel]) {
      tablesToFill[O2fwdtrack] = true;
      tablesToFill[O2fwdtrackcov] = true;
    }
  }

  // O2bc is always needed (?)
  tablesToFill[O2bc] = true;
}

void AODProducerWorkflowDPL::collectBCs(o2::globaltracking::RecoContainer& data,
                                        const std::vector<o2::InteractionTimeRecord>& mcRecords,
                                        std::map<uint64_t, int>& bcsMap)
{
  const auto& primVertices = data.getPrimaryVertices();
  const auto& fddRecPoints = data.getFDDRecPoints();
  const auto& ft0RecPoints = data.getFT0RecPoints();
  const auto& fv0RecPoints = data.getFV0RecPoints();
  const auto& caloEMCCellsTRGR = data.getEMCALTriggers();
  const auto& ctpDigits = data.getCTPDigits();
  const auto& zdcBCRecData = data.getZDCBCRecData();

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

  for (auto& zdcRecData : zdcBCRecData) {
    uint64_t globalBC = zdcRecData.ir.toLong();
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

  for (auto& ctpDigit : ctpDigits) {
    uint64_t globalBC = ctpDigit.intRecord.toLong();
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
            const float intLen = tofInt.getL();
            extraInfoHolder.length = intLen;
            if (interactionTime > 0) {
              extraInfoHolder.tofSignal = static_cast<float>(tofMatch.getSignal() - interactionTime * 1E3);
            }
            const float mass = o2::constants::physics::MassPionCharged; // default pid = pion
            if (tofInt.getTOF(o2::track::PID::Pion) > 0.f) {
              const float expBeta = (intLen / (tofInt.getTOF(o2::track::PID::Pion) * cSpeed));
              extraInfoHolder.tofExpMom = mass * expBeta / std::sqrt(1.f - expBeta * expBeta);
            }
            // correct the time of the track
            const double massZ = o2::track::PID::getMass2Z(trackPar.getPID());
            const double energy = sqrt((massZ * massZ) + (extraInfoHolder.tofExpMom * extraInfoHolder.tofExpMom));
            const double exp = extraInfoHolder.length * energy / (cSpeed * extraInfoHolder.tofExpMom);
            extraInfoHolder.trackTime = static_cast<float>((tofMatch.getSignal() - exp) * 1e-3 - interactionTime); // tof time in \mus, FIXME: account for time of flight to R TOF
            extraInfoHolder.trackTimeRes = 200e-3;                                                                 // FIXME: calculate actual resolution (if possible?)
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
        // we store all primary particles == particles put by generator
        if (mcParticles[particle].isPrimary()) {
          mToStore[Triplet_t(source, event, particle)] = 1;
          continue;
        }
        if (mToStore.find(Triplet_t(source, event, particle)) == mToStore.end()) {
          continue;
        }
        int mother0 = mcParticles[particle].getMotherTrackId();
        // we store mothers and daughters of particles that are reconstructed
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
              LOG(debug) << "ITS-TPC MCTruth: labelIDs do not match at " << trackIndex.getIndex() << ", src = " << src;
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
    if (tpcClusShMap[absoluteIndex] & GPUCA_NAMESPACE::gpu::GPUTPCGMMergedTrackHit::flagShared) {
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
      LOG(fatal) << "Error: could not find a corresponding BC ID for a EMCal point; globalBC = " << globalBC;
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
    LOG(info) << "TFNumber will be obtained from CCDB";
  }
  if (mRunNumber == -1L) {
    LOG(info) << "The Run number will be obtained from DPL headers";
  }

  // create EventHandler used for calo cells
  mCaloEventHandler = new o2::emcal::EventHandler<o2::emcal::Cell>();

  // set no truncation if selected by user
  if (mTruncate != 1) {
    LOG(info) << "Truncation is not used!";
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

  // initialize zdc helper maps
  for (auto& ChannelName : o2::zdc::ChannelNames) {
    mZDCEnergyMap[(string)ChannelName] = 0;
    mZDCTDCMap[(string)ChannelName] = 999;
  }

  // writing metadata if it's not yet in AOD file
  // note: `--aod-writer-resmode "UPDATE"` has to be used,
  //       so that metadata is not overwritten
  mResFile += ".root";
  auto* fResFile = TFile::Open(mResFile, "UPDATE");
  if (!fResFile) {
    LOGF(fatal, "Could not open file %s", mResFile);
  }
  if (fResFile->FindObjectAny("metaData")) {
    LOGF(warning, "Metadata: target file %s already has metadata, preserving it", mResFile);
  } else {
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
    fResFile->WriteObject(&mMetaData, "metaData", "Overwrite");
  }
  fResFile->Close();

  mTimer.Reset();
}

void AODProducerWorkflowDPL::run(ProcessingContext& pc)
{
  mTimer.Start(false);

  bool foundFatal = false;

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

  auto zdcEnergies = recoData.getZDCEnergy();
  auto zdcBCRecData = recoData.getZDCBCRecData();
  auto zdcTDCData = recoData.getZDCTDCData();

  // get calo information
  auto caloEMCCells = recoData.getEMCALCells();
  auto caloEMCCellsTRGR = recoData.getEMCALTriggers();

  auto ctpDigits = recoData.getCTPDigits();

  LOG(debug) << "FOUND " << primVertices.size() << " primary vertices";
  LOG(debug) << "FOUND " << ft0RecPoints.size() << " FT0 rec. points";
  LOG(debug) << "FOUND " << fv0RecPoints.size() << " FV0 rec. points";
  LOG(debug) << "FOUND " << fddRecPoints.size() << " FDD rec. points";
  LOG(debug) << "FOUND " << caloEMCCells.size() << " EMC cells";
  LOG(debug) << "FOUND " << caloEMCCellsTRGR.size() << " EMC Trigger Records";

  std::vector<TableBuilder> tables;
  for (int i = 0; i < numTables; i++) {
    TableBuilder table;
    tables.emplace_back(table);
  }

  auto bcCursor = tables[O2bc].cursor<o2::aod::BCs>();
  auto cascadesCursor = tables[O2cascade].cursor<o2::aod::StoredCascades>();
  auto collisionsCursor = tables[O2collision].cursor<o2::aod::Collisions>();
  auto fddCursor = tables[O2fdd].cursor<o2::aod::FDDs>();
  auto ft0Cursor = tables[O2ft0].cursor<o2::aod::FT0s>();
  auto fv0aCursor = tables[O2fv0a].cursor<o2::aod::FV0As>();
  auto fv0cCursor = tables[O2fv0c].cursor<o2::aod::FV0Cs>();
  auto fwdTracksCursor = tables[O2fwdtrack].cursor<o2::aodproducer::FwdTracksTable>();
  auto fwdTracksCovCursor = tables[O2fwdtrackcov].cursor<o2::aodproducer::FwdTracksCovTable>();
  auto mcColLabelsCursor = tables[O2mccollisionlabel].cursor<o2::aod::McCollisionLabels>();
  auto mcCollisionsCursor = tables[O2mccollision].cursor<o2::aod::McCollisions>();
  auto mcMFTTrackLabelCursor = tables[O2mcmfttracklabel].cursor<o2::aod::McMFTTrackLabels>();
  auto mcFwdTrackLabelCursor = tables[O2mcfwdtracklabel].cursor<o2::aod::McFwdTrackLabels>();
  auto mcParticlesCursor = tables[O2mcparticle].cursor<o2::aodproducer::MCParticlesTable>();
  auto mcTrackLabelCursor = tables[O2mctracklabel].cursor<o2::aod::McTrackLabels>();
  auto mftTracksCursor = tables[O2mfttrack].cursor<o2::aodproducer::MFTTracksTable>();
  auto tracksCovCursor = tables[O2trackcov].cursor<o2::aodproducer::TracksCovTable>();
  auto tracksCursor = tables[O2track].cursor<o2::aodproducer::TracksTable>();
  auto tracksExtraCursor = tables[O2trackextra].cursor<o2::aodproducer::TracksExtraTable>();
  auto v0sCursor = tables[O2v0].cursor<o2::aod::StoredV0s>();
  auto zdcCursor = tables[O2zdc].cursor<o2::aod::Zdcs>();
  auto caloCellsCursor = tables[O2caloCell].cursor<o2::aod::Calos>();
  auto caloCellsTRGTableCursor = tables[O2caloCellTRGR].cursor<o2::aod::CaloTriggers>();

  std::unique_ptr<o2::steer::MCKinematicsReader> mcReader;
  if (mUseMC) {
    mcReader = std::make_unique<o2::steer::MCKinematicsReader>("collisioncontext.root");
    LOG(debug) << "FOUND " << mcReader->getDigitizationContext()->getEventRecords().size()
               << " records" << mcReader->getDigitizationContext()->getEventParts().size() << " parts";
  }

  std::map<uint64_t, int> bcsMap;
  collectBCs(recoData, mUseMC ? mcReader->getDigitizationContext()->getEventRecords() : std::vector<o2::InteractionTimeRecord>{}, bcsMap);
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

  if (mInputSources[GID::FV0] && tablesToFill[O2fv0a]) {
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
        LOG(fatal) << "Error: could not find a corresponding BC ID for a FV0 rec. point; BC = " << bc;
      }
      fv0aCursor(0,
                 bcID,
                 aAmplitudesA,
                 truncateFloatFraction(fv0RecPoint.getCollisionGlobalMeanTime() * 1E-3, mV0Time), // ps to ns
                 fv0RecPoint.getTrigger().triggerSignals);
    }
  }

  if (mInputSources[GID::FV0] && tablesToFill[O2fv0c]) {
    float dummyFV0AmplC[32] = {0.};
    fv0cCursor(0,
               dummyBC,
               dummyFV0AmplC,
               dummyTime);
  }

  if (mInputSources[GID::ZDC] && tablesToFill[O2zdc]) {
    for (auto zdcRecData : zdcBCRecData) {
      uint64_t bc = zdcRecData.ir.toLong();
      auto item = bcsMap.find(bc);
      int bcID = -1;
      if (item != bcsMap.end()) {
        bcID = item->second;
      } else {
        LOG(fatal) << "Error: could not find a corresponding BC ID for a ZDC rec. point; BC = " << bc;
      }
      float energyZEM1 = 0;
      float energyZEM2 = 0;
      float energyCommonZNA = 0;
      float energyCommonZNC = 0;
      float energyCommonZPA = 0;
      float energyCommonZPC = 0;
      float energySectorZNA[4] = {0.};
      float energySectorZNC[4] = {0.};
      float energySectorZPA[4] = {0.};
      float energySectorZPC[4] = {0.};
      int fe, ne, ft, nt, fi, ni;
      zdcRecData.getRef(fe, ne, ft, nt, fi, ni);
      for (int ie = 0; ie < ne; ie++) {
        auto& zdcEnergyData = zdcEnergies[fe + ie];
        float energy = zdcEnergyData.energy();
        string chName = o2::zdc::channelName(zdcEnergyData.ch());
        mZDCEnergyMap.at(chName) = energy;
      }
      for (int it = 0; it < nt; it++) {
        auto& tdc = zdcTDCData[ft + it];
        float tdcValue = tdc.value();
        int channelID = o2::zdc::TDCSignal[tdc.ch()];
        auto channelName = o2::zdc::ChannelNames[channelID];
        mZDCTDCMap.at((string)channelName) = tdcValue;
      }
      energySectorZNA[0] = mZDCEnergyMap.at("ZNA1");
      energySectorZNA[1] = mZDCEnergyMap.at("ZNA2");
      energySectorZNA[2] = mZDCEnergyMap.at("ZNA3");
      energySectorZNA[3] = mZDCEnergyMap.at("ZNA4");
      energySectorZNC[0] = mZDCEnergyMap.at("ZNC1");
      energySectorZNC[1] = mZDCEnergyMap.at("ZNC2");
      energySectorZNC[2] = mZDCEnergyMap.at("ZNC3");
      energySectorZNC[3] = mZDCEnergyMap.at("ZNC4");
      energySectorZPA[0] = mZDCEnergyMap.at("ZPA1");
      energySectorZPA[1] = mZDCEnergyMap.at("ZPA2");
      energySectorZPA[2] = mZDCEnergyMap.at("ZPA3");
      energySectorZPA[3] = mZDCEnergyMap.at("ZPA4");
      energySectorZPC[0] = mZDCEnergyMap.at("ZPC1");
      energySectorZPC[1] = mZDCEnergyMap.at("ZPC2");
      energySectorZPC[2] = mZDCEnergyMap.at("ZPC3");
      energySectorZPC[3] = mZDCEnergyMap.at("ZPC4");
      zdcCursor(0,
                bcID,
                mZDCEnergyMap.at("ZEM1"),
                mZDCEnergyMap.at("ZEM2"),
                mZDCEnergyMap.at("ZNAC"),
                mZDCEnergyMap.at("ZNCC"),
                mZDCEnergyMap.at("ZPAC"),
                mZDCEnergyMap.at("ZPCC"),
                energySectorZNA,
                energySectorZNC,
                energySectorZPA,
                energySectorZPC,
                mZDCTDCMap.at("ZEM1"),
                mZDCTDCMap.at("ZEM2"),
                mZDCTDCMap.at("ZNAC"),
                mZDCTDCMap.at("ZNCC"),
                mZDCTDCMap.at("ZPAC"),
                mZDCTDCMap.at("ZPCC"));
    }
  }

  // keep track event/source id for each mc-collision
  // using map and not unordered_map to ensure
  // correct ordering when iterating over container elements
  std::map<std::pair<int, int>, int> mcColToEvSrc;

  if (mUseMC && tablesToFill[O2mccollision]) {
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
        LOG(fatal) << "Error: could not find a corresponding BC ID for MC collision; BC = " << globalBC << ", mc collision = " << iCol;
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

  if (mInputSources[GID::FDD] && tablesToFill[O2fdd]) {
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
        LOG(fatal) << "Error: could not find a corresponding BC ID for a FDD rec. point; BC = " << bc;
      }
      fddCursor(0,
                bcID,
                aFDDAmplitudesA,
                aFDDAmplitudesC,
                truncateFloatFraction(fddRecPoint.getCollisionTimeA() * 1E-3, mFDDTime), // ps to ns
                truncateFloatFraction(fddRecPoint.getCollisionTimeC() * 1E-3, mFDDTime), // ps to ns
                fddRecPoint.getTrigger().triggersignals);
    }
  }

  if (mInputSources[GID::FT0] && tablesToFill[O2ft0]) {
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
        LOG(fatal) << "Error: could not find a corresponding BC ID for a FT0 rec. point; BC = " << bc;
      }
      ft0Cursor(0,
                bcID,
                aAmplitudesA,
                aAmplitudesC,
                truncateFloatFraction(ft0RecPoint.getCollisionTimeA() * 1E-3, mT0Time), // ps to ns
                truncateFloatFraction(ft0RecPoint.getCollisionTimeC() * 1E-3, mT0Time), // ps to ns
                ft0RecPoint.getTrigger().triggersignals);
    }
  }

  if (mUseMC && tablesToFill[O2mccollisionlabel]) {
    if (!tablesToFill[O2mccollision]) {
      LOG(warn) << "O2mccollisionlabel is dependent on O2mccollision: add it to the tables list";
      foundFatal = true;
    }
    if (!tablesToFill[O2collision]) {
      LOG(warn) << "O2mccollisionlabel is dependent on O2collision: add it to the tables list";
      foundFatal = true;
    }
    if (foundFatal) {
      LOG(fatal) << "Found fatal error! See warnings above";
    }
    // filling MC collision labels
    for (auto& label : primVerLabels) {
      auto it = mcColToEvSrc.find(std::pair<int, int>(label.getEventID(), label.getSourceID()));
      int32_t mcCollisionID = it != mcColToEvSrc.end() ? it->second : -1;
      uint16_t mcMask = 0; // todo: set mask using normalized weights?
      mcColLabelsCursor(0, mcCollisionID, mcMask);
    }
  }

  bool needTracks = tablesToFill[O2track] || tablesToFill[O2trackcov] || tablesToFill[O2trackextra];
  bool needMFTTracks = tablesToFill[O2mfttrack];
  bool needFwdTracks = tablesToFill[O2fwdtrack] || tablesToFill[O2fwdtrackcov];
  bool needAnyTracks = needTracks || needMFTTracks || needFwdTracks;
  if (needAnyTracks) {
    if (!tablesToFill[O2collision]) {
      LOG(warn) << "Tracks tables are dependent on O2collision: add it to the tables list";
      foundFatal = true;
    }
    if (foundFatal) {
      LOG(fatal) << "Found fatal error! See warnings above";
    }
    // filling unassigned tracks first
    // so that all unassigned tracks are stored in the beginning of the table together
    auto& trackRef = primVer2TRefs.back(); // references to unassigned tracks are at the end
    // fixme: interaction time is undefined for unassigned tracks (?)
    fillTrackTablesPerCollision(-1, -1, trackRef, primVerGIs, recoData, tracksCursor, tracksCovCursor, tracksExtraCursor, mftTracksCursor, fwdTracksCursor, fwdTracksCovCursor, dataformats::PrimaryVertex{});
  }

  if (tablesToFill[O2collision]) {
    // filling collisions and tracks into tables
    int collisionID = 0;
    for (auto& vertex : primVertices) {
      auto& cov = vertex.getCov();
      auto& timeStamp = vertex.getTimeStamp();                       // this is a relative time
      const double interactionTime = timeStamp.getTimeStamp() * 1E3; // mus to ns
      uint64_t globalBC = relativeTime_to_GlobalBC(interactionTime);
      uint64_t localBC = relativeTime_to_LocalBC(interactionTime);
      LOG(debug) << "global BC " << globalBC << " local BC " << localBC << " relative interaction time " << interactionTime;
      // collision timestamp in ns wrt the beginning of collision BC
      const float relInteractionTime = static_cast<float>(localBC * o2::constants::lhc::LHCBunchSpacingNS - interactionTime);
      auto item = bcsMap.find(globalBC);
      int bcID = -1;
      if (item != bcsMap.end()) {
        bcID = item->second;
      } else {
        LOG(fatal) << "Error: could not find a corresponding BC ID for a collision; BC = " << globalBC << ", collisionID = " << collisionID;
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
      if (needAnyTracks) {
        fillTrackTablesPerCollision(collisionID, interactionTime, trackRef, primVerGIs, recoData, tracksCursor, tracksCovCursor, tracksExtraCursor, mftTracksCursor, fwdTracksCursor, fwdTracksCovCursor, vertex);
      }
      collisionID++;
    }
  }

  // filling v0s table
  if (tablesToFill[O2v0]) {
    if (!needTracks) {
      LOG(warn) << "O2v0 is dependent on barrel tracks: add them to the tables list";
      foundFatal = true;
    }
    if (foundFatal) {
      LOG(fatal) << "Found fatal error! See warnings above";
    }
    for (auto& svertex : secVertices) {
      auto trPosID = svertex.getProngID(0);
      auto trNegID = svertex.getProngID(1);
      int posTableIdx = -1;
      int negTableIdx = -1;
      auto item = mGIDToTableID.find(trPosID);
      if (item != mGIDToTableID.end()) {
        posTableIdx = item->second;
      } else {
        LOG(warn) << "Could not find a positive track index for prong ID " << trPosID;
      }
      item = mGIDToTableID.find(trNegID);
      if (item != mGIDToTableID.end()) {
        negTableIdx = item->second;
      } else {
        LOG(warn) << "Could not find a negative track index for prong ID " << trNegID;
      }
      if (posTableIdx != -1 and negTableIdx != -1) {
        v0sCursor(0, posTableIdx, negTableIdx);
      }
    }
  }

  // filling cascades table
  if (tablesToFill[O2cascade]) {
    if (!needTracks) {
      LOG(warn) << "O2v0 is dependent on barrel tracks: add them to the tables list";
      foundFatal = true;
    }
    if (foundFatal) {
      LOG(fatal) << "Found fatal error! See warnings above";
    }
    for (auto& cascade : cascades) {
      auto bachelorID = cascade.getBachelorID();
      int bachTableIdx = -1;
      auto item = mGIDToTableID.find(bachelorID);
      if (item != mGIDToTableID.end()) {
        bachTableIdx = item->second;
      } else {
        LOG(warn) << "Could not find a bachelor track index";
      }
      cascadesCursor(0, cascade.getV0ID(), bachTableIdx);
    }
  }

  mTableTrID = 0;
  mGIDToTableID.clear();

  // helper map for fast search of a corresponding class mask for a bc
  std::unordered_map<uint64_t, uint64_t> bcToClassMask;
  if (mInputSources[GID::CTP]) {
    for (auto& ctpDigit : ctpDigits) {
      uint64_t bc = ctpDigit.intRecord.toLong();
      uint64_t classMask = ctpDigit.CTPClassMask.to_ulong();
      bcToClassMask[bc] = classMask;
    }
  }

  // filling BC table
  uint64_t triggerMask = 0;
  for (auto& item : bcsMap) {
    uint64_t bc = item.first;
    if (mInputSources[GID::CTP]) {
      auto bcClassPair = bcToClassMask.find(bc);
      if (bcClassPair != bcToClassMask.end()) {
        triggerMask = bcClassPair->second;
      } else {
        triggerMask = 0;
      }
    }
    bcCursor(0,
             runNumber,
             bc,
             triggerMask);
  }

  bcToClassMask.clear();

  if (mInputSources[GIndex::EMC] && tablesToFill[O2caloCell] && tablesToFill[O2caloCellTRGR]) {
    // fill EMC cells to tables
    // TODO handle MC info
    fillCaloTable(caloEMCCells, caloEMCCellsTRGR, caloCellsCursor, caloCellsTRGTableCursor, bcsMap);
  }

  bcsMap.clear();

  if (mUseMC) {
    // filling mc particles table
    if (tablesToFill[O2mcparticle]) {
      // reconstructed tracks accessed via collisions -- needed for filtering
      if (!tablesToFill[O2collision]) {
        LOG(warn) << "O2mcparticle is dependent on O2collision: add it to the tables list";
        foundFatal = true;
      }
      // correct mc collision ids can be assigned only if O2mccollision is processed
      if (!tablesToFill[O2mccollision]) {
        LOG(warn) << "O2mcparticle is dependent on O2mccollision: add it to the tables list";
        foundFatal = true;
      }
      if (foundFatal) {
        LOG(fatal) << "Found fatal error! See warnings above";
      }
      fillMCParticlesTable(*mcReader,
                           mcParticlesCursor,
                           primVer2TRefs,
                           primVerGIs,
                           recoData,
                           mcColToEvSrc);
    }

    mcColToEvSrc.clear();

    // ------------------------------------------------------
    // filling track labels

    // need to go through labels in the same order as for tracks
    bool needAnyTrackLabels = tablesToFill[O2mcfwdtracklabel] || tablesToFill[O2mcmfttracklabel] || tablesToFill[O2mctracklabel];
    if (needAnyTrackLabels) {
      fillMCTrackLabelsTable(mcTrackLabelCursor, mcMFTTrackLabelCursor, mcFwdTrackLabelCursor, primVer2TRefs.back(), primVerGIs, recoData);
      for (int iref = 0; iref < primVer2TRefs.size() - 1; iref++) {
        auto& trackRef = primVer2TRefs[iref];
        fillMCTrackLabelsTable(mcTrackLabelCursor, mcMFTTrackLabelCursor, mcFwdTrackLabelCursor, trackRef, primVerGIs, recoData);
      }
    }
  }
  mToStore.clear();

  // pushing requested tables into outputs
  for (int it = 0; it < numTables; it++) {
    if (tablesToFill[it]) {
      auto& table = pc.outputs().make<TableBuilder>(Output{"AOD", tablesDesc[it]});
      std::swap(table, tables[it]);
    }
  }
  pc.outputs().snapshot(Output{"TFN", "TFNumber", 0, Lifetime::Timeframe}, tfNumber);

  tables.clear();

  mTimer.Stop();
}

void AODProducerWorkflowDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "aod producer dpl total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getAODProducerWorkflowSpec(GID::mask_t src, std::string tablesList, bool enableSV, bool useMC, std::string resFile)
{
  using producer = o2::aodproducer::AODProducerWorkflowDPL;
  // parse list of tables and mark tables to be stored
  producer::parseTablesList(tablesList);

  // check dependencies between tables
  producer::checkTablesDeps(useMC);

  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  dataRequest->requestTracks(src, useMC);
  if (producer::tablesToFill[producer::O2collision]) {
    dataRequest->requestPrimaryVertertices(useMC);
  }
  if (src[GID::CTP]) {
    LOGF(info, "Requesting CTP digits");
    dataRequest->requestCTPDigits(useMC);
  }
  if (enableSV) {
    dataRequest->requestSecondaryVertertices(useMC);
  }
  if (src[GID::TPC]) {
    dataRequest->requestClusters(GIndex::getSourcesMask("TPC"), false); // TOF clusters are requested with TOF tracks
  }
  if (src[GID::EMC]) {
    dataRequest->requestEMCALCells(useMC);
  }

  LOGF(info, "Tables to be stored:");
  for (int it = 0; it < producer::numTables; it++) {
    if (producer::tablesToFill[it]) {
      LOGF(info, "%s", producer::tablesNames[it]);
      outputs.emplace_back(OutputLabel{(string)producer::tablesNames[it]}, "AOD", producer::tablesDesc[it], Lifetime::Timeframe);
    }
  }

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
