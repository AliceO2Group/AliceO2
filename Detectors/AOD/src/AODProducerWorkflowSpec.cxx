// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AODProducerWorkflowSpec.cxx

#include "AODProducerWorkflow/AODProducerWorkflowSpec.h"
#include "DataFormatsFT0/RecPoints.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include <CCDB/BasicCCDBManager.h>
#include "CommonDataFormat/InteractionRecord.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisHelpers.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataTypes.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Logger.h"
#include "Framework/TableBuilder.h"
#include "Framework/TableTreeHelpers.h"
#include "GlobalTracking/MatchTOF.h"
#include "GlobalTrackingWorkflow/PrimaryVertexingSpec.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TMath.h"
#include "MathUtils/Utils.h"
#include <map>
#include <vector>

using namespace o2::framework;
using namespace o2::math_utils::detail;

namespace o2::aodproducer
{

void AODProducerWorkflowDPL::findMinMaxBc(gsl::span<const o2::ft0::RecPoints>& ft0RecPoints, gsl::span<const o2::vertexing::PVertex>& primVertices, const std::vector<o2::InteractionTimeRecord>& mcRecords)
{
  for (auto& ft0RecPoint : ft0RecPoints) {
    uint64_t bc = ft0RecPoint.getInteractionRecord().orbit * o2::constants::lhc::LHCMaxBunches + ft0RecPoint.getInteractionRecord().bc;
    if (minGlBC > bc) {
      minGlBC = bc;
    }
    if (maxGlBC < bc) {
      maxGlBC = bc;
    }
  }

  for (auto& vertex : primVertices) {
    const InteractionRecord& colIRMin = vertex.getIRMin();
    const InteractionRecord& colIRMax = vertex.getIRMax();
    uint64_t colBCMin = colIRMin.orbit * o2::constants::lhc::LHCMaxBunches + colIRMin.bc;
    uint64_t colBCMax = colIRMax.orbit * o2::constants::lhc::LHCMaxBunches + colIRMax.bc;
    if (minGlBC > colBCMin) {
      minGlBC = colBCMin;
    }
    if (maxGlBC < colBCMax) {
      maxGlBC = colBCMax;
    }
  }

  for (auto& rec : mcRecords) {
    uint64_t bc = rec.bc + rec.orbit * o2::constants::lhc::LHCMaxBunches;
    if (minGlBC > bc) {
      minGlBC = bc;
    }
    if (maxGlBC < bc) {
      maxGlBC = bc;
    }
  }
}

uint64_t AODProducerWorkflowDPL::getTFNumber(uint64_t firstVtxGlBC, int runNumber)
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

  // firstRec --> calculated using `minimal` global BC in the simulation (see AODProducerWorkflowDPL::findMinMaxBc)
  // firstVtxGlBC --> calculated using global BC corresponding to the first prim. vertex

  uint32_t initialOrbit = mapStartOrbit->at(runNumber);
  uint16_t firstRecBC = minGlBC % o2::constants::lhc::LHCMaxBunches;
  uint32_t firstRecOrbit = minGlBC / o2::constants::lhc::LHCMaxBunches;
  uint16_t firstVtxBC = firstVtxGlBC % o2::constants::lhc::LHCMaxBunches;
  uint32_t firstVtxOrbit = firstVtxGlBC / o2::constants::lhc::LHCMaxBunches;
  const o2::InteractionRecord firstRec(firstRecBC, firstRecOrbit);
  const o2::InteractionRecord firstVtx(firstVtxBC, firstVtxOrbit);
  ts += (firstVtx - firstRec).bc2ns() / 1000000;

  return ts;
};

template <typename TTracks, typename TTracksCursor, typename TTracksCovCursor, typename TTracksExtraCursor>
void AODProducerWorkflowDPL::fillTracksTable(const TTracks& tracks, std::vector<int>& vCollRefs, const TTracksCursor& tracksCursor,
                                             const TTracksCovCursor& tracksCovCursor, const TTracksExtraCursor& tracksExtraCursor, int trackType)
{
  for (int i = 0; i < tracks.size(); i++) {
    auto& track = tracks[i];
    int collisionID = vCollRefs[i];

    float tpcInnerParam = 0.f;
    uint32_t flags = 0;
    uint8_t itsClusterMap = 0;
    uint8_t tpcNClsFindable = 0;
    int8_t tpcNClsFindableMinusFound = 0;
    int8_t tpcNClsFindableMinusCrossedRows = 0;
    uint8_t tpcNClsShared = 0;
    uint8_t trdPattern = 0;
    float itsChi2NCl = -999.f;
    float tpcChi2NCl = -999.f;
    float trdChi2 = -999.f;
    float tofChi2 = -999.f;
    float tpcSignal = -999.f;
    float trdSignal = -999.f;
    float tofSignal = -999.f;
    float length = -999.f;
    float tofExpMom = -999.f;
    float trackEtaEMCAL = -999.f;
    float trackPhiEMCAL = -999.f;

    // filling available columns for different track types
    std::variant<o2::its::TrackITS, o2::tpc::TrackTPC, o2::dataformats::TrackTPCITS> tmp = track;
    std::visit(
      overloaded{
        [&](o2::its::TrackITS itsTrack) {
          itsClusterMap = itsTrack.getPattern();
        },
        [&](o2::tpc::TrackTPC tpcTrack) {
          tpcChi2NCl = tpcTrack.getNClusters() ? tpcTrack.getChi2() / tpcTrack.getNClusters() : 0;
          tpcSignal = tpcTrack.getdEdx().dEdxTotTPC;
          tpcNClsFindable = tpcTrack.getNClusters();
        },
        [&](o2::dataformats::TrackTPCITS itsTpcTrack) {
          LOG(DEBUG) << "TrackTPCITS: check";
        }},
      tmp);

    // TODO:
    // fill trackextra table
    tracksCursor(0,
                 collisionID,
                 trackType,
                 truncateFloatFraction(track.getX(), mTrackX),
                 truncateFloatFraction(track.getAlpha(), mTrackAlpha),
                 track.getY(),
                 track.getZ(),
                 truncateFloatFraction(track.getSnp(), mTrackSnp),
                 truncateFloatFraction(track.getTgl(), mTrackTgl),
                 truncateFloatFraction(track.getQ2Pt(), mTrack1Pt));
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
    tracksExtraCursor(0,
                      truncateFloatFraction(tpcInnerParam, mTrack1Pt),
                      flags,
                      itsClusterMap,
                      tpcNClsFindable,
                      tpcNClsFindableMinusFound,
                      tpcNClsFindableMinusCrossedRows,
                      tpcNClsShared,
                      trdPattern,
                      truncateFloatFraction(itsChi2NCl, mTrackCovOffDiag),
                      truncateFloatFraction(tpcChi2NCl, mTrackCovOffDiag),
                      truncateFloatFraction(trdChi2, mTrackCovOffDiag),
                      truncateFloatFraction(tofChi2, mTrackCovOffDiag),
                      truncateFloatFraction(tpcSignal, mTrackSignal),
                      truncateFloatFraction(trdSignal, mTrackSignal),
                      truncateFloatFraction(tofSignal, mTrackSignal),
                      truncateFloatFraction(length, mTrackSignal),
                      truncateFloatFraction(tofExpMom, mTrack1Pt),
                      truncateFloatFraction(trackEtaEMCAL, mTrackPosEMCAL),
                      truncateFloatFraction(trackPhiEMCAL, mTrackPosEMCAL));
  }
}

template <typename TMCTruthITS, typename TMCTruthTPC>
void AODProducerWorkflowDPL::findRelatives(const o2::steer::MCKinematicsReader& mcReader, const TMCTruthITS& mcTruthITS, const TMCTruthTPC& mcTruthTPC,
                                           std::vector<std::vector<std::vector<int>>>& toStore)
{
  if (mFillTracksITS) {
    for (auto& mcTruth : mcTruthITS) {
      int source = mcTruth.getSourceID();
      int event = mcTruth.getEventID();
      int track = mcTruth.getTrackID();
      toStore[source][event][track] = 1;
    }
  }
  if (mFillTracksTPC) {
    for (auto& mcTruth : mcTruthTPC) {
      int source = mcTruth.getSourceID();
      int event = mcTruth.getEventID();
      int track = mcTruth.getTrackID();
      toStore[source][event][track] = 1;
    }
  }
  for (int source = 0; source < mcReader.getNSources(); source++) {
    for (int event = 0; event < mcReader.getNEvents(source); event++) {
      std::vector<MCTrack> const& mcParticles = mcReader.getTracks(source, event);
      for (int track = mcParticles.size() - 1; track <= 0; track--) {
        int mother0 = mcParticles[track].getMotherTrackId();
        int mother1 = mcParticles[track].getSecondMotherTrackId();
        if (mother0 == -1 || mother1 == -1) {
          toStore[source][event][track] = 1;
        }
        if (toStore[source][event][track] == 0) {
          continue;
        }
        if (mother0 != -1) {
          toStore[source][event][mother0] = 1;
        }
        if (mother1 != -1) {
          toStore[source][event][mother1] = 1;
        }
        int daughter0 = mcParticles[track].getFirstDaughterTrackId();
        int daughterL = mcParticles[track].getLastDaughterTrackId();
        if (daughter0 != -1) {
          toStore[source][event][daughter0] = 1;
        }
        if (daughterL != -1) {
          toStore[source][event][daughterL] = 1;
        }
      }
    }
  }
}

template <typename MCParticlesCursorType>
void AODProducerWorkflowDPL::fillMCParticlesTable(const o2::steer::MCKinematicsReader& mcReader, const MCParticlesCursorType& mcParticlesCursor,
                                                  std::vector<std::vector<std::vector<int>>>& toStore)
{
  int tableIndex = 0;
  for (int source = 0; source < toStore.size(); source++) {
    for (int event = 0; event < toStore[source].size(); event++) {
      for (int track = 0; track < toStore[source][event].size(); track++) {
        if (!toStore[source][event][track] && mRecoOnly) {
          continue;
        }
        toStore[source][event][track] = tableIndex;
        tableIndex++;
      }
    }
  }
  for (int source = 0; source < mcReader.getNSources(); source++) {
    for (int event = 0; event < mcReader.getNEvents(source); event++) {
      std::vector<MCTrack> const& mcParticles = mcReader.getTracks(source, event);
      for (int track = 0; track < mcParticles.size(); track++) {
        if (!toStore[source][event][track] && mRecoOnly) {
          continue;
        }
        int statusCode = 0;
        uint8_t flags = 0;
        float weight = 0.f;
        int mother0 = mcParticles[track].getMotherTrackId() != -1 ? toStore[source][event][mcParticles[track].getMotherTrackId()] : -1;
        int mother1 = mcParticles[track].getSecondMotherTrackId() != -1 ? toStore[source][event][mcParticles[track].getSecondMotherTrackId()] : -1;
        int daughter0 = mcParticles[track].getFirstDaughterTrackId() != -1 ? toStore[source][event][mcParticles[track].getFirstDaughterTrackId()] : -1;
        int daughterL = mcParticles[track].getLastDaughterTrackId() != -1 ? toStore[source][event][mcParticles[track].getLastDaughterTrackId()] : -1;
        mcParticlesCursor(0,
                          event,
                          mcParticles[track].GetPdgCode(),
                          statusCode,
                          flags,
                          mother0,
                          mother1,
                          daughter0,
                          daughterL,
                          truncateFloatFraction(weight, mMcParticleW),
                          truncateFloatFraction((float)mcParticles[track].Px(), mMcParticleMom),
                          truncateFloatFraction((float)mcParticles[track].Py(), mMcParticleMom),
                          truncateFloatFraction((float)mcParticles[track].Pz(), mMcParticleMom),
                          truncateFloatFraction((float)mcParticles[track].GetEnergy(), mMcParticleMom),
                          truncateFloatFraction((float)mcParticles[track].Vx(), mMcParticlePos),
                          truncateFloatFraction((float)mcParticles[track].Vy(), mMcParticlePos),
                          truncateFloatFraction((float)mcParticles[track].Vz(), mMcParticlePos),
                          truncateFloatFraction((float)mcParticles[track].T(), mMcParticlePos));
      }
    }
  }
}

void AODProducerWorkflowDPL::init(InitContext& ic)
{
  mTimer.Stop();

  mFillTracksITS = ic.options().get<int>("fill-tracks-its");
  mFillTracksTPC = ic.options().get<int>("fill-tracks-tpc");
  mFillTracksITSTPC = ic.options().get<int>("fill-tracks-its-tpc");
  mTFNumber = ic.options().get<int>("aod-timeframe-id");
  mIgnoreWriter = ic.options().get<int>("ignore-aod-writer");
  mRecoOnly = ic.options().get<int>("reco-mctracks-only");
  mTruncate = ic.options().get<int>("enable-truncation");

  if (mTFNumber == -1) {
    LOG(INFO) << "TFNumber will be obtained from CCDB";
  }

  LOG(INFO) << "Track filling flags are set to: "
            << "\n ITS = " << mFillTracksITS << "\n TPC = " << mFillTracksTPC << "\n ITSTPC = " << mFillTracksITSTPC;

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

  mTimer.Reset();
}

void AODProducerWorkflowDPL::run(ProcessingContext& pc)
{
  mTimer.Start(false);

  auto ft0ChData = pc.inputs().get<gsl::span<o2::ft0::ChannelDataFloat>>("ft0ChData");
  auto ft0RecPoints = pc.inputs().get<gsl::span<o2::ft0::RecPoints>>("ft0RecPoints");
  auto primVer2TRefs = pc.inputs().get<gsl::span<o2::vertexing::V2TRef>>("primVer2TRefs");
  auto primVerGIs = pc.inputs().get<gsl::span<o2::vertexing::GIndex>>("primVerGIs");
  auto primVertices = pc.inputs().get<gsl::span<o2::vertexing::PVertex>>("primVertices");
  auto tracksITS = pc.inputs().get<gsl::span<o2::its::TrackITS>>("trackITS");
  auto tracksITSTPC = pc.inputs().get<gsl::span<o2::dataformats::TrackTPCITS>>("tracksITSTPC");
  auto tracksTPC = pc.inputs().get<gsl::span<o2::tpc::TrackTPC>>("trackTPC");
  auto tracksTPCMCTruth = pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackTPCMCTruth");
  auto tracksITSMCTruth = pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackITSMCTruth");

  LOG(DEBUG) << "FOUND " << tracksTPC.size() << " TPC tracks";
  LOG(DEBUG) << "FOUND " << tracksITS.size() << " ITS tracks";
  LOG(DEBUG) << "FOUND " << tracksITSTPC.size() << " ITSTPC tracks";

  TableBuilder bcBuilder;
  TableBuilder collisionsBuilder;
  TableBuilder ft0Builder;
  TableBuilder mcCollisionsBuilder;
  TableBuilder tracksBuilder;
  TableBuilder tracksCovBuilder;
  TableBuilder tracksExtraBuilder;
  TableBuilder mcParticlesBuilder;
  TableBuilder mcTrackLabelBuilder;
  TableBuilder fv0aBuilder;
  TableBuilder fddBuilder;
  TableBuilder fv0cBuilder;
  TableBuilder zdcBuilder;
  uint64_t timeFrameNumberBuilder;

  if (mIgnoreWriter == 0) {
    bcBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "BC"});
    collisionsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "COLLISION"});
    ft0Builder = pc.outputs().make<TableBuilder>(Output{"AOD", "FT0"});
    mcCollisionsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCCOLLISION"});
    tracksBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "TRACK"});
    tracksCovBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "TRACKCOV"});
    tracksExtraBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "TRACKEXTRA"});
    mcParticlesBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCPARTICLE"});
    mcTrackLabelBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCTRACKLABEL"});
    fv0aBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FV0A"});
    fddBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FDD"});
    fv0cBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FV0C"});
    zdcBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "ZDC"});
    timeFrameNumberBuilder = pc.outputs().make<uint64_t>(Output{"TFN", "TFNumber"});
  }

  auto bcCursor = bcBuilder.cursor<o2::aod::BCs>();
  auto collisionsCursor = collisionsBuilder.cursor<o2::aod::Collisions>();
  auto ft0Cursor = ft0Builder.cursor<o2::aod::FT0s>();
  auto mcCollisionsCursor = mcCollisionsBuilder.cursor<o2::aod::McCollisions>();
  auto tracksCursor = tracksBuilder.cursor<o2::aodproducer::TracksTable>();
  auto tracksCovCursor = tracksCovBuilder.cursor<o2::aodproducer::TracksCovTable>();
  auto tracksExtraCursor = tracksExtraBuilder.cursor<o2::aodproducer::TracksExtraTable>();
  auto mcParticlesCursor = mcParticlesBuilder.cursor<o2::aodproducer::MCParticlesTable>();
  auto mcTrackLabelCursor = mcTrackLabelBuilder.cursor<o2::aod::McTrackLabels>();
  auto fv0aCursor = fv0aBuilder.cursor<o2::aod::FV0As>();
  auto fv0cCursor = fv0cBuilder.cursor<o2::aod::FV0Cs>();
  auto fddCursor = fddBuilder.cursor<o2::aod::FDDs>();
  auto zdcCursor = zdcBuilder.cursor<o2::aod::Zdcs>();

  o2::steer::MCKinematicsReader mcReader("collisioncontext.root");
  const auto mcContext = mcReader.getDigitizationContext();
  const auto& mcRecords = mcContext->getEventRecords();
  const auto& mcParts = mcContext->getEventParts();

  LOG(DEBUG) << "FOUND " << mcRecords.size() << " records";
  LOG(DEBUG) << "FOUND " << mcParts.size() << " parts";

  findMinMaxBc(ft0RecPoints, primVertices, mcRecords);

  // TODO: get real run number and triggerMask
  int runNumber = 244918;
  uint64_t triggerMask = 1;

  for (uint64_t i = 0; i <= maxGlBC - minGlBC; i++) {
    bcCursor(0,
             runNumber,
             minGlBC + i,
             triggerMask);
  }

  auto tableBC = mIgnoreWriter != 0 ? bcBuilder.finalize() : nullptr;

  uint64_t globalBC;
  uint64_t BCid;

  // TODO: add real FV0A, FV0C, FDD, ZDC tables instead of dummies
  uint64_t dummyBC = 0;
  float dummyTime = 0.f;
  float dummyFV0AmplA[48] = {0.};
  uint8_t dummyTriggerMask = 0;
  fv0aCursor(0,
             dummyBC,
             dummyFV0AmplA,
             dummyTime,
             dummyTriggerMask);

  auto tableFV0A = mIgnoreWriter != 0 ? fv0aBuilder.finalize() : nullptr;

  float dummyFV0AmplC[32] = {0.};
  fv0cCursor(0,
             dummyBC,
             dummyFV0AmplC,
             dummyTime);

  auto tableFV0C = mIgnoreWriter != 0 ? fv0cBuilder.finalize() : nullptr;

  float dummyFDDAmplA[4] = {0.};
  float dummyFDDAmplC[4] = {0.};
  fddCursor(0,
            dummyBC,
            dummyFDDAmplA,
            dummyFDDAmplC,
            dummyTime,
            dummyTime,
            dummyTriggerMask);

  auto tableFDD = mIgnoreWriter != 0 ? fddBuilder.finalize() : nullptr;

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
  auto tableZDC = mIgnoreWriter != 0 ? zdcBuilder.finalize() : nullptr;

  // TODO: figure out collision weight
  float mcColWeight = 1.;
  // filling mcCollision table
  int index = 0;
  for (auto& rec : mcRecords) {
    auto time = rec.getTimeNS();
    globalBC = rec.bc + rec.orbit * o2::constants::lhc::LHCMaxBunches;
    BCid = globalBC - minGlBC;
    if (BCid < 0) {
      BCid = 0;
    } else if (BCid > maxGlBC) {
      BCid = maxGlBC;
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
                         BCid,
                         generatorID,
                         truncateFloatFraction(header.GetX(), mCollisionPosition),
                         truncateFloatFraction(header.GetY(), mCollisionPosition),
                         truncateFloatFraction(header.GetZ(), mCollisionPosition),
                         truncateFloatFraction(time, mCollisionPosition),
                         truncateFloatFraction(mcColWeight, mCollisionPosition),
                         header.GetB());
    }
    index++;
  }

  auto tableMCCollisions = mIgnoreWriter != 0 ? mcCollisionsBuilder.finalize() : nullptr;

  // filling mc particles table
  std::vector<std::vector<std::vector<int>>> toStore;
  for (int source = 0; source < mcReader.getNSources(); source++) {
    std::vector<std::vector<int>> vEvents;
    toStore.push_back(vEvents);
    for (int event = 0; event < mcReader.getNEvents(source); event++) {
      std::vector<int> vTracks;
      toStore[source].push_back(vTracks);
      for (int track = 0; track < mcReader.getTracks(source, event).size(); track++) {
        toStore[source][event].push_back(0);
      }
    }
  }
  if (mRecoOnly) {
    findRelatives(mcReader, tracksITSMCTruth, tracksTPCMCTruth, toStore);
  }
  fillMCParticlesTable(mcReader, mcParticlesCursor, toStore);
  auto tableMCParticles = mIgnoreWriter != 0 ? mcParticlesBuilder.finalize() : nullptr;

  // vector of FT0 amplitudes
  std::vector<float> vAmplitudes(208, 0.);
  // filling FT0 table
  for (auto& ft0RecPoint : ft0RecPoints) {
    const auto channelData = ft0RecPoint.getBunchChannelData(ft0ChData);
    // TODO: switch to calibrated amplitude
    for (auto& channel : channelData) {
      vAmplitudes[channel.ChId] = channel.QTCAmpl; // amplitude, mV
    }
    float aAmplitudesA[96];
    float aAmplitudesC[112];
    for (int i = 0; i < 96; i++) {
      aAmplitudesA[i] = truncateFloatFraction(vAmplitudes[i], mT0Amplitude);
    }
    for (int i = 0; i < 112; i++) {
      aAmplitudesC[i] = truncateFloatFraction(vAmplitudes[i + 96], mT0Amplitude);
    }
    globalBC = ft0RecPoint.getInteractionRecord().orbit * o2::constants::lhc::LHCMaxBunches + ft0RecPoint.getInteractionRecord().bc;
    BCid = globalBC - minGlBC;
    if (BCid < 0) {
      BCid = 0;
    } else if (BCid > maxGlBC) {
      BCid = maxGlBC;
    }
    ft0Cursor(0,
              BCid,
              aAmplitudesA,
              aAmplitudesC,
              truncateFloatFraction(ft0RecPoint.getCollisionTimeA() / 1E3, mT0Time), // ps to ns
              truncateFloatFraction(ft0RecPoint.getCollisionTimeC() / 1E3, mT0Time), // ps to ns
              ft0RecPoint.getTrigger().triggersignals);
  }

  auto tableFT0 = mIgnoreWriter != 0 ? ft0Builder.finalize() : nullptr;

  // initializing vectors for trackID --> collisionID connection
  std::vector<int> vCollRefsITS(tracksITS.size(), -1);
  std::vector<int> vCollRefsTPC(tracksTPC.size(), -1);
  std::vector<int> vCollRefsTPCITS(tracksITSTPC.size(), -1);

  // TODO: determine the beginning of a TF in case when there are no reconstructed vertices
  uint64_t firstVtxGlBC = minGlBC;
  uint64_t startBCofTF = 0;
  if (primVertices.empty()) {
    auto startIRofTF = o2::raw::HBFUtils::Instance().getFirstIRofTF(primVertices[0].getIRMin());
    startBCofTF = startIRofTF.orbit * o2::constants::lhc::LHCMaxBunches + startIRofTF.bc;
    firstVtxGlBC = std::round(startBCofTF + primVertices[0].getTimeStamp().getTimeStamp() / o2::constants::lhc::LHCBunchSpacingMS);
  }

  // filling collisions table
  int collisionID = 0;
  for (auto& vertex : primVertices) {
    auto& cov = vertex.getCov();
    auto& timeStamp = vertex.getTimeStamp();
    Double_t tsTimeStamp = timeStamp.getTimeStamp() * 1E3; // mus to ns
    globalBC = std::round(startBCofTF + tsTimeStamp / o2::constants::lhc::LHCBunchSpacingNS);
    LOG(DEBUG) << globalBC << " " << tsTimeStamp;
    // collision timestamp in ns wrt the beginning of collision BC
    tsTimeStamp = globalBC * o2::constants::lhc::LHCBunchSpacingNS - tsTimeStamp;
    BCid = globalBC - minGlBC;
    if (BCid < 0) {
      BCid = 0;
    } else if (BCid > maxGlBC) {
      BCid = maxGlBC;
    }
    // TODO: get real collision time mask
    int collisionTimeMask = 0;
    collisionsCursor(0,
                     BCid,
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
                     truncateFloatFraction(tsTimeStamp, mCollisionPosition),
                     truncateFloatFraction(timeStamp.getTimeStampError() * 1E3, mCollisionPositionCov),
                     collisionTimeMask);

    auto trackRef = primVer2TRefs[collisionID];
    int start = trackRef.getFirstEntryOfSource(0);
    int ntracks = trackRef.getEntriesOfSource(0);
    // FIXME: `track<-->vertex` ambiguity is not accounted for in this code
    for (int ti = 0; ti < ntracks; ti++) {
      auto trackIndex = primVerGIs[start + ti];
      const auto source = trackIndex.getSource();
      // setting collisionID for tracks attached to vertices
      if (source == o2::vertexing::GIndex::Source::TPC) {
        vCollRefsTPC[trackIndex.getIndex()] = collisionID;
      } else if (source == o2::vertexing::GIndex::Source::ITS) {
        vCollRefsITS[trackIndex.getIndex()] = collisionID;
      } else if (source == o2::vertexing::GIndex::Source::ITSTPC) {
        vCollRefsTPCITS[trackIndex.getIndex()] = collisionID;
      } else {
        LOG(WARNING) << "Unsupported track type!";
      }
    }
    collisionID++;
  }

  auto tableCollisions = mIgnoreWriter != 0 ? collisionsBuilder.finalize() : nullptr;

  // filling tracks tables and track label table

  // labelMask (temporary) usage:
  //   bit 13 -- ITS and TPC labels are not equal
  //   bit 14 -- isNoise() == true
  //   bit 15 -- isFake() == true
  // labelID = std::numeric_limits<uint32_t>::max() -- label is not set

  uint32_t labelID;
  uint32_t labelITS;
  uint32_t labelTPC;
  uint16_t labelMask;

  if (mFillTracksITS) {
    fillTracksTable(tracksITS, vCollRefsITS, tracksCursor, tracksCovCursor, tracksExtraCursor, o2::vertexing::GIndex::Source::ITS);
    for (auto& mcTruthITS : tracksITSMCTruth) {
      labelID = std::numeric_limits<uint32_t>::max();
      // TODO: fill label mask
      labelMask = 0;
      if (mcTruthITS.isValid()) {
        labelID = toStore[mcTruthITS.getSourceID()][mcTruthITS.getEventID()][mcTruthITS.getTrackID()];
      }
      if (mcTruthITS.isFake()) {
        labelMask |= (0x1 << 15);
      }
      if (mcTruthITS.isNoise()) {
        labelMask |= (0x1 << 14);
      }
      mcTrackLabelCursor(0,
                         labelID,
                         labelMask);
    }
  }

  if (mFillTracksTPC) {
    fillTracksTable(tracksTPC, vCollRefsTPC, tracksCursor, tracksCovCursor, tracksExtraCursor, o2::vertexing::GIndex::Source::TPC);
    for (auto& mcTruthTPC : tracksTPCMCTruth) {
      labelID = std::numeric_limits<uint32_t>::max();
      // TODO: fill label mask
      labelMask = 0;
      if (mcTruthTPC.isValid()) {
        labelID = toStore[mcTruthTPC.getSourceID()][mcTruthTPC.getEventID()][mcTruthTPC.getTrackID()];
      }
      if (mcTruthTPC.isFake()) {
        labelMask |= (0x1 << 15);
      }
      if (mcTruthTPC.isNoise()) {
        labelMask |= (0x1 << 14);
      }
      mcTrackLabelCursor(0,
                         labelID,
                         labelMask);
    }
  }

  if (mFillTracksITSTPC) {
    fillTracksTable(tracksITSTPC, vCollRefsTPCITS, tracksCursor, tracksCovCursor, tracksExtraCursor, o2::vertexing::GIndex::Source::ITSTPC);
    for (int i = 0; i < tracksITSTPC.size(); i++) {
      const auto& trc = tracksITSTPC[i];
      auto mcTruthITS = tracksITSMCTruth[trc.getRefITS()];
      auto mcTruthTPC = tracksTPCMCTruth[trc.getRefTPC()];
      labelID = std::numeric_limits<uint32_t>::max();
      labelITS = std::numeric_limits<uint32_t>::max();
      labelTPC = std::numeric_limits<uint32_t>::max();
      // TODO: fill label mask
      // currently using label mask to indicate labelITS != labelTPC
      labelMask = 0;
      if (mcTruthITS.isValid() && mcTruthTPC.isValid()) {
        labelITS = toStore[mcTruthITS.getSourceID()][mcTruthITS.getEventID()][mcTruthITS.getTrackID()];
        labelTPC = toStore[mcTruthTPC.getSourceID()][mcTruthTPC.getEventID()][mcTruthTPC.getTrackID()];
        labelID = labelITS;
      }
      if (mcTruthITS.isFake() || mcTruthTPC.isFake()) {
        labelMask |= (0x1 << 15);
      }
      if (mcTruthITS.isNoise() || mcTruthTPC.isNoise()) {
        labelMask |= (0x1 << 14);
      }
      if (labelITS != labelTPC) {
        LOG(DEBUG) << "ITS-TPC MCTruth: labelIDs do not match at " << i;
        labelMask |= (0x1 << 13);
      }
      mcTrackLabelCursor(0,
                         labelID,
                         labelMask);
    }
  }

  auto tableTracks = mIgnoreWriter != 0 ? tracksBuilder.finalize() : nullptr;
  auto tableTracksCov = mIgnoreWriter != 0 ? tracksCovBuilder.finalize() : nullptr;
  auto tableTracksExtra = mIgnoreWriter != 0 ? tracksExtraBuilder.finalize() : nullptr;
  auto tableMCTrackLabels = mIgnoreWriter != 0 ? mcTrackLabelBuilder.finalize() : nullptr;

  uint64_t tfNumber;
  if (mTFNumber == -1) {
    tfNumber = getTFNumber(firstVtxGlBC, runNumber);
  } else {
    tfNumber = mTFNumber;
  }

  timeFrameNumberBuilder = tfNumber;

  std::string treeName;
  std::string dirName = "DF_" + std::to_string(tfNumber);

  if (mIgnoreWriter != 0) {
    TFile outfile("AOD.root", "UPDATE");
    outfile.mkdir(dirName.c_str());
    {
      treeName = dirName + "/" + "O2bc";
      TableToTree t2t(tableBC, &outfile, treeName.c_str());
      t2t.addAllBranches();
      t2t.process();
    }
    {
      treeName = dirName + "/" + "O2collision";
      TableToTree t2t(tableCollisions, &outfile, treeName.c_str());
      t2t.addAllBranches();
      t2t.process();
    }
    {
      treeName = dirName + "/" + "O2ft0";
      TableToTree t2t(tableFT0, &outfile, treeName.c_str());
      t2t.addAllBranches();
      t2t.process();
    }
    {
      treeName = dirName + "/" + "O2mccollision";
      TableToTree t2t(tableMCCollisions, &outfile, treeName.c_str());
      t2t.addAllBranches();
      t2t.process();
    }
    {
      treeName = dirName + "/" + "O2track";
      TableToTree t2t(tableTracks, &outfile, treeName.c_str());
      t2t.addAllBranches();
      t2t.process();
    }
    {
      treeName = dirName + "/" + "O2trackcov";
      TableToTree t2t(tableTracks, &outfile, treeName.c_str());
      t2t.addAllBranches();
      t2t.process();
    }
    {
      treeName = dirName + "/" + "O2trackextra";
      TableToTree t2t(tableTracks, &outfile, treeName.c_str());
      t2t.addAllBranches();
      t2t.process();
    }
    {
      treeName = dirName + "/" + "O2mcparticle";
      TableToTree t2t(tableMCParticles, &outfile, treeName.c_str());
      t2t.addAllBranches();
      t2t.process();
    }
    {
      treeName = dirName + "/" + "O2mctracklabel";
      TableToTree t2t(tableMCTrackLabels, &outfile, treeName.c_str());
      t2t.addAllBranches();
      t2t.process();
    }
    {
      treeName = dirName + "/" + "O2fv0a";
      TableToTree t2t(tableFV0A, &outfile, treeName.c_str());
      t2t.addAllBranches();
      t2t.process();
    }
    {
      treeName = dirName + "/" + "O2fdd";
      TableToTree t2t(tableFDD, &outfile, treeName.c_str());
      t2t.addAllBranches();
      t2t.process();
    }
    {
      treeName = dirName + "/" + "O2fv0c";
      TableToTree t2t(tableFV0C, &outfile, treeName.c_str());
      t2t.addAllBranches();
      t2t.process();
    }
    {
      treeName = dirName + "/" + "O2zdc";
      TableToTree t2t(tableZDC, &outfile, treeName.c_str());
      t2t.addAllBranches();
      t2t.process();
    }
  }

  mTimer.Stop();
}

void AODProducerWorkflowDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "aod producer dpl total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getAODProducerWorkflowSpec()
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;

  inputs.emplace_back("ft0ChData", "FT0", "RECCHDATA", 0, Lifetime::Timeframe);
  inputs.emplace_back("ft0RecPoints", "FT0", "RECPOINTS", 0, Lifetime::Timeframe);
  inputs.emplace_back("primVer2TRefs", "GLO", "PVTX_TRMTCREFS", 0, Lifetime::Timeframe);
  inputs.emplace_back("primVerGIs", "GLO", "PVTX_TRMTC", 0, Lifetime::Timeframe);
  inputs.emplace_back("primVertices", "GLO", "PVTX", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackITS", "ITS", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("tracksITSTPC", "GLO", "TPCITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackTPC", "TPC", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackTPCMCTruth", "TPC", "TRACKSMCLBL", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackITSMCTruth", "ITS", "TRACKSMCTR", 0, Lifetime::Timeframe);

  outputs.emplace_back(OutputLabel{"O2bc"}, "AOD", "BC", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2collision"}, "AOD", "COLLISION", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2ft0"}, "AOD", "FT0", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mccollision"}, "AOD", "MCCOLLISION", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2track"}, "AOD", "TRACK", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2trackcov"}, "AOD", "TRACKCOV", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2trackextra"}, "AOD", "TRACKEXTRA", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mcparticle"}, "AOD", "MCPARTICLE", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mctracklabel"}, "AOD", "MCTRACKLABEL", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputSpec{"TFN", "TFNumber"});
  outputs.emplace_back(OutputLabel{"O2fv0a"}, "AOD", "FV0A", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2fv0c"}, "AOD", "FV0C", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2fdd"}, "AOD", "FDD", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2zdc"}, "AOD", "ZDC", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "aod-producer-workflow",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<AODProducerWorkflowDPL>()},
    Options{
      ConfigParamSpec{"fill-tracks-its", VariantType::Int, 1, {"Fill ITS tracks into tracks table"}},
      ConfigParamSpec{"fill-tracks-tpc", VariantType::Int, 0, {"Fill TPC tracks into tracks table"}},
      ConfigParamSpec{"fill-tracks-its-tpc", VariantType::Int, 1, {"Fill ITS-TPC tracks into tracks table"}},
      ConfigParamSpec{"aod-timeframe-id", VariantType::Int, -1, {"Set timeframe number"}},
      ConfigParamSpec{"enable-truncation", VariantType::Int, 1, {"Truncation parameter: 1 -- on, != 1 -- off"}},
      ConfigParamSpec{"ignore-aod-writer", VariantType::Int, 0, {"Ignore DPL AOD writer and write tables directly into a file. 0 -- off, != 0 -- on"}},
      ConfigParamSpec{"reco-mctracks-only", VariantType::Int, 0, {"Store only reconstructed MC tracks and their mothers/daughters. 0 -- off, != 0 -- on"}}}};
}

} // namespace o2::aodproducer
