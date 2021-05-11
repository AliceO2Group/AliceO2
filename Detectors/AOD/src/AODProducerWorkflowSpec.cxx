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
#include "CCDB/BasicCCDBManager.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataTypes.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Logger.h"
#include "Framework/TableBuilder.h"
#include "Framework/TableTreeHelpers.h"
#include "GlobalTracking/MatchTOF.h"
#include "GlobalTrackingWorkflow/PrimaryVertexingSpec.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "TMath.h"
#include "MathUtils/Utils.h"
#include <map>
#include <vector>

using namespace o2::framework;
using namespace o2::math_utils::detail;
using PVertex = o2::dataformats::PrimaryVertex;
using V2TRef = o2::dataformats::VtxTrackRef;
using GIndex = o2::dataformats::VtxTrackIndex;

namespace o2::aodproducer
{

void AODProducerWorkflowDPL::findMinMaxBc(gsl::span<const o2::ft0::RecPoints>& ft0RecPoints, gsl::span<const PVertex>& primVertices, const std::vector<o2::InteractionTimeRecord>& mcRecords)
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

template <typename MCParticlesCursorType>
void AODProducerWorkflowDPL::fillMCParticlesTable(o2::steer::MCKinematicsReader& mcReader, const MCParticlesCursorType& mcParticlesCursor,
                                                  gsl::span<const o2::MCCompLabel>& mcTruthITS, gsl::span<const o2::MCCompLabel>& mcTruthTPC,
                                                  TripletsMap_t& toStore)
{
  // mark reconstructed MC particles to store them into the table
  for (auto& mcTruth : mcTruthITS) {
    if (!mcTruth.isValid()) {
      continue;
    }
    int source = mcTruth.getSourceID();
    int event = mcTruth.getEventID();
    int particle = mcTruth.getTrackID();
    toStore[Triplet_t(source, event, particle)] = 1;
  }
  for (auto& mcTruth : mcTruthTPC) {
    if (!mcTruth.isValid()) {
      continue;
    }
    int source = mcTruth.getSourceID();
    int event = mcTruth.getEventID();
    int particle = mcTruth.getTrackID();
    toStore[Triplet_t(source, event, particle)] = 1;
  }
  int tableIndex = 1;
  for (int source = 0; source < mcReader.getNSources(); source++) {
    for (int event = 0; event < mcReader.getNEvents(source); event++) {
      std::vector<MCTrack> const& mcParticles = mcReader.getTracks(source, event);
      // mark tracks to be stored per event
      // loop over stack of MC particles from end to beginning: daughters are stored after mothers
      if (mRecoOnly) {
        for (int particle = mcParticles.size() - 1; particle >= 0; particle--) {
          int mother0 = mcParticles[particle].getMotherTrackId();
          if (mother0 == -1) {
            toStore[Triplet_t(source, event, particle)] = 1;
          }
          if (toStore.find(Triplet_t(source, event, particle)) == toStore.end()) {
            continue;
          }
          if (mother0 != -1) {
            toStore[Triplet_t(source, event, mother0)] = 1;
          }
          int mother1 = mcParticles[particle].getSecondMotherTrackId();
          if (mother1 != -1) {
            toStore[Triplet_t(source, particle, mother1)] = 1;
          }
          int daughter0 = mcParticles[particle].getFirstDaughterTrackId();
          if (daughter0 != -1) {
            toStore[Triplet_t(source, event, daughter0)] = 1;
          }
          int daughterL = mcParticles[particle].getLastDaughterTrackId();
          if (daughterL != -1) {
            toStore[Triplet_t(source, event, daughterL)] = 1;
          }
        }
        // enumerate reconstructed mc particles and their relatives to get mother/daughter relations
        for (int particle = 0; particle < mcParticles.size(); particle++) {
          auto mapItem = toStore.find(Triplet_t(source, event, particle));
          if (mapItem != toStore.end()) {
            mapItem->second = tableIndex;
            tableIndex++;
          }
        }
      }
      // if all mc particles are stored, all mc particles will be enumerated
      if (!mRecoOnly) {
        for (int particle = 0; particle < mcParticles.size(); particle++) {
          toStore[Triplet_t(source, event, particle)] = tableIndex;
          tableIndex++;
        }
      }
      // fill survived mc tracks into the table
      for (int particle = 0; particle < mcParticles.size(); particle++) {
        if (toStore.find(Triplet_t(source, event, particle)) == toStore.end()) {
          continue;
        }
        int statusCode = 0;
        uint8_t flags = 0;
        float weight = 0.f;
        int mcMother0 = mcParticles[particle].getMotherTrackId();
        auto item = toStore.find(Triplet_t(source, event, mcMother0));
        int mother0 = -1;
        if (item != toStore.end()) {
          mother0 = item->second;
        }
        int mcMother1 = mcParticles[particle].getSecondMotherTrackId();
        int mother1 = -1;
        item = toStore.find(Triplet_t(source, event, mcMother1));
        if (item != toStore.end()) {
          mother1 = item->second;
        }
        int mcDaughter0 = mcParticles[particle].getFirstDaughterTrackId();
        int daughter0 = -1;
        item = toStore.find(Triplet_t(source, event, mcDaughter0));
        if (item != toStore.end()) {
          daughter0 = item->second;
        }
        int mcDaughterL = mcParticles[particle].getLastDaughterTrackId();
        int daughterL = -1;
        item = toStore.find(Triplet_t(source, event, mcDaughterL));
        if (item != toStore.end()) {
          daughterL = item->second;
        }
        mcParticlesCursor(0,
                          event,
                          mcParticles[particle].GetPdgCode(),
                          statusCode,
                          flags,
                          mother0,
                          mother1,
                          daughter0,
                          daughterL,
                          truncateFloatFraction(weight, mMcParticleW),
                          truncateFloatFraction((float)mcParticles[particle].Px(), mMcParticleMom),
                          truncateFloatFraction((float)mcParticles[particle].Py(), mMcParticleMom),
                          truncateFloatFraction((float)mcParticles[particle].Pz(), mMcParticleMom),
                          truncateFloatFraction((float)mcParticles[particle].GetEnergy(), mMcParticleMom),
                          truncateFloatFraction((float)mcParticles[particle].Vx(), mMcParticlePos),
                          truncateFloatFraction((float)mcParticles[particle].Vy(), mMcParticlePos),
                          truncateFloatFraction((float)mcParticles[particle].Vz(), mMcParticlePos),
                          truncateFloatFraction((float)mcParticles[particle].T(), mMcParticlePos));
      }
      mcReader.releaseTracksForSourceAndEvent(source, event);
    }
  }
}

void AODProducerWorkflowDPL::writeTableToFile(TFile* outfile, std::shared_ptr<arrow::Table>& table, const std::string& tableName, uint64_t tfNumber)
{
  std::string treeName;
  std::string dirName = "DF_" + std::to_string(tfNumber);
  treeName = dirName + "/" + tableName;
  TableToTree t2t(table, outfile, treeName.c_str());
  t2t.addAllBranches();
  t2t.process();
}

void AODProducerWorkflowDPL::init(InitContext& ic)
{
  mTimer.Stop();

  mFillTracksITS = ic.options().get<int>("fill-tracks-its");
  mFillTracksTPC = ic.options().get<int>("fill-tracks-tpc");
  mFillTracksITSTPC = ic.options().get<int>("fill-tracks-its-tpc");
  mTFNumber = ic.options().get<int64_t>("aod-timeframe-id");
  mRecoOnly = ic.options().get<int>("reco-mctracks-only");
  mTruncate = ic.options().get<int>("enable-truncation");

  if (mTFNumber == -1L) {
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
  auto primVer2TRefs = pc.inputs().get<gsl::span<V2TRef>>("primVer2TRefs");
  auto primVerGIs = pc.inputs().get<gsl::span<GIndex>>("primVerGIs");
  auto primVertices = pc.inputs().get<gsl::span<PVertex>>("primVertices");
  auto primVerLabels = pc.inputs().get<gsl::span<o2::MCEventLabel>>("primVerLabels");
  auto tracksITS = pc.inputs().get<gsl::span<o2::its::TrackITS>>("trackITS");
  auto tracksITSTPC = pc.inputs().get<gsl::span<o2::dataformats::TrackTPCITS>>("tracksITSTPC");
  auto tracksTPC = pc.inputs().get<gsl::span<o2::tpc::TrackTPC>>("trackTPC");
  auto tracksTPCMCTruth = pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackTPCMCTruth");
  auto tracksITSMCTruth = pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackITSMCTruth");

  LOG(DEBUG) << "FOUND " << tracksTPC.size() << " TPC tracks";
  LOG(DEBUG) << "FOUND " << tracksITS.size() << " ITS tracks";
  LOG(DEBUG) << "FOUND " << tracksITSTPC.size() << " ITSTPC tracks";

  TableBuilder bcBuilderS;
  TableBuilder collisionsBuilderS;
  TableBuilder mcColLabelsBuilderS;
  TableBuilder ft0BuilderS;
  TableBuilder mcCollisionsBuilderS;
  TableBuilder tracksBuilderS;
  TableBuilder tracksCovBuilderS;
  TableBuilder tracksExtraBuilderS;
  TableBuilder mcParticlesBuilderS;
  TableBuilder mcTrackLabelBuilderS;
  TableBuilder fv0aBuilderS;
  TableBuilder fddBuilderS;
  TableBuilder fv0cBuilderS;
  TableBuilder zdcBuilderS;

  auto& bcBuilder = mIgnoreWriter == 0 ? pc.outputs().make<TableBuilder>(Output{"AOD", "BC"}) : bcBuilderS;
  auto& collisionsBuilder = mIgnoreWriter == 0 ? pc.outputs().make<TableBuilder>(Output{"AOD", "COLLISION"}) : collisionsBuilderS;
  auto& mcColLabelsBuilder = mIgnoreWriter == 0 ? pc.outputs().make<TableBuilder>(Output{"AOD", "MCCOLLISLABEL"}) : mcColLabelsBuilderS;
  auto& ft0Builder = mIgnoreWriter == 0 ? pc.outputs().make<TableBuilder>(Output{"AOD", "FT0"}) : ft0BuilderS;
  auto& mcCollisionsBuilder = mIgnoreWriter == 0 ? pc.outputs().make<TableBuilder>(Output{"AOD", "MCCOLLISION"}) : mcCollisionsBuilderS;
  auto& tracksBuilder = mIgnoreWriter == 0 ? pc.outputs().make<TableBuilder>(Output{"AOD", "TRACK"}) : tracksBuilderS;
  auto& tracksCovBuilder = mIgnoreWriter == 0 ? pc.outputs().make<TableBuilder>(Output{"AOD", "TRACKCOV"}) : tracksCovBuilderS;
  auto& tracksExtraBuilder = mIgnoreWriter == 0 ? pc.outputs().make<TableBuilder>(Output{"AOD", "TRACKEXTRA"}) : tracksExtraBuilderS;
  auto& mcParticlesBuilder = mIgnoreWriter == 0 ? pc.outputs().make<TableBuilder>(Output{"AOD", "MCPARTICLE"}) : mcParticlesBuilderS;
  auto& mcTrackLabelBuilder = mIgnoreWriter == 0 ? pc.outputs().make<TableBuilder>(Output{"AOD", "MCTRACKLABEL"}) : mcTrackLabelBuilderS;
  auto& fv0aBuilder = mIgnoreWriter == 0 ? pc.outputs().make<TableBuilder>(Output{"AOD", "FV0A"}) : fv0aBuilderS;
  auto& fddBuilder = mIgnoreWriter == 0 ? pc.outputs().make<TableBuilder>(Output{"AOD", "FDD"}) : fddBuilderS;
  auto& fv0cBuilder = mIgnoreWriter == 0 ? pc.outputs().make<TableBuilder>(Output{"AOD", "FV0C"}) : fv0cBuilderS;
  auto& zdcBuilder = mIgnoreWriter == 0 ? pc.outputs().make<TableBuilder>(Output{"AOD", "ZDC"}) : zdcBuilderS;

  auto bcCursor = bcBuilder.cursor<o2::aod::BCs>();
  auto collisionsCursor = collisionsBuilder.cursor<o2::aod::Collisions>();
  auto mcColLabelsCursor = mcColLabelsBuilder.cursor<o2::aod::McCollisionLabels>();
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

  uint64_t globalBC;
  uint64_t BCid;
  std::vector<uint64_t> BCIDs;

  findMinMaxBc(ft0RecPoints, primVertices, mcRecords);

  const auto* dh = o2::header::get<o2::header::DataHeader*>(pc.inputs().getByPos(0).header);
  o2::InteractionRecord startIR = {0, dh->firstTForbit};

  uint64_t firstVtxGlBC = minGlBC;
  uint64_t startBCofTF = startIR.toLong();
  if (!primVertices.empty()) {
    firstVtxGlBC = std::round(startBCofTF + primVertices[0].getTimeStamp().getTimeStamp() / o2::constants::lhc::LHCBunchSpacingMUS);
  }

  uint64_t tfNumber;
  int runNumber = 244918; // TODO: get real run number
  if (mTFNumber == -1L) {
    tfNumber = getTFNumber(firstVtxGlBC, runNumber);
  } else {
    tfNumber = mTFNumber;
  }

  TFile* outfile;
  if (mIgnoreWriter) {
    std::string dirName = "DF_" + std::to_string(tfNumber);
    outfile = TFile::Open("AOD.root", "UPDATE");
    outfile->mkdir(dirName.c_str());
  }

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

  if (mIgnoreWriter) {
    std::shared_ptr<arrow::Table> tableFV0A = fv0aBuilder.finalize();
    std::string tableName("O2fv0a");
    writeTableToFile(outfile, tableFV0A, tableName, tfNumber);
  }

  float dummyFV0AmplC[32] = {0.};
  fv0cCursor(0,
             dummyBC,
             dummyFV0AmplC,
             dummyTime);

  if (mIgnoreWriter) {
    std::shared_ptr<arrow::Table> tableFV0C = fv0cBuilder.finalize();
    std::string tableName("O2fv0c");
    writeTableToFile(outfile, tableFV0C, tableName, tfNumber);
  }

  float dummyFDDAmplA[4] = {0.};
  float dummyFDDAmplC[4] = {0.};
  fddCursor(0,
            dummyBC,
            dummyFDDAmplA,
            dummyFDDAmplC,
            dummyTime,
            dummyTime,
            dummyTriggerMask);

  if (mIgnoreWriter) {
    std::shared_ptr<arrow::Table> tableFDD = fddBuilder.finalize();
    std::string tableName("O2fdd");
    writeTableToFile(outfile, tableFDD, tableName, tfNumber);
  }

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

  if (mIgnoreWriter) {
    std::shared_ptr<arrow::Table> tableZDC = zdcBuilder.finalize();
    std::string tableName("O2zdc");
    writeTableToFile(outfile, tableZDC, tableName, tfNumber);
  }

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
    } else if (BCid > maxGlBC - minGlBC) {
      BCid = maxGlBC - minGlBC;
    }
    BCIDs.push_back(BCid);
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

  if (mIgnoreWriter) {
    std::shared_ptr<arrow::Table> tableMCCollisions = mcCollisionsBuilder.finalize();
    std::string tableName("O2mccollision");
    writeTableToFile(outfile, tableMCCollisions, tableName, tfNumber);
  }

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
    } else if (BCid > maxGlBC - minGlBC) {
      BCid = maxGlBC - minGlBC;
    }
    BCIDs.push_back(BCid);
    ft0Cursor(0,
              BCid,
              aAmplitudesA,
              aAmplitudesC,
              truncateFloatFraction(ft0RecPoint.getCollisionTimeA() / 1E3, mT0Time), // ps to ns
              truncateFloatFraction(ft0RecPoint.getCollisionTimeC() / 1E3, mT0Time), // ps to ns
              ft0RecPoint.getTrigger().triggersignals);
  }

  if (mIgnoreWriter) {
    std::shared_ptr<arrow::Table> tableFT0 = ft0Builder.finalize();
    std::string tableName("O2ft0");
    writeTableToFile(outfile, tableFT0, tableName, tfNumber);
  }

  // initializing vectors for trackID --> collisionID connection
  std::vector<int> vCollRefsITS(tracksITS.size(), -1);
  std::vector<int> vCollRefsTPC(tracksTPC.size(), -1);
  std::vector<int> vCollRefsTPCITS(tracksITSTPC.size(), -1);

  // filling MC collision labels
  for (auto& label : primVerLabels) {
    int32_t mcCollisionID = label.getEventID();
    uint16_t mcMask = 0; // todo: set mask using normalized weights?
    mcColLabelsCursor(0, mcCollisionID, mcMask);
  }

  if (mIgnoreWriter) {
    std::shared_ptr<arrow::Table> tableMCColLabels = mcColLabelsBuilder.finalize();
    std::string tableName("O2mccollisionlabel");
    writeTableToFile(outfile, tableMCColLabels, tableName, tfNumber);
  }

  // filling collisions table
  int collisionID = 0;
  for (auto& vertex : primVertices) {
    auto& cov = vertex.getCov();
    auto& timeStamp = vertex.getTimeStamp();
    Double_t tsTimeStamp = timeStamp.getTimeStamp() * 1E3; // mus to ns
    globalBC = std::round(tsTimeStamp / o2::constants::lhc::LHCBunchSpacingNS);
    LOG(DEBUG) << globalBC << " " << tsTimeStamp;
    // collision timestamp in ns wrt the beginning of collision BC
    tsTimeStamp = globalBC * o2::constants::lhc::LHCBunchSpacingNS - tsTimeStamp;
    BCid = globalBC - minGlBC;
    if (BCid < 0) {
      BCid = 0;
    } else if (BCid > maxGlBC - minGlBC) {
      BCid = maxGlBC - minGlBC;
    }
    BCIDs.push_back(BCid);
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
      if (source == GIndex::Source::TPC) {
        vCollRefsTPC[trackIndex.getIndex()] = collisionID;
      } else if (source == GIndex::Source::ITS) {
        vCollRefsITS[trackIndex.getIndex()] = collisionID;
      } else if (source == GIndex::Source::ITSTPC) {
        vCollRefsTPCITS[trackIndex.getIndex()] = collisionID;
      } else {
        LOG(WARNING) << "Unsupported track type!";
      }
    }
    collisionID++;
  }

  if (mIgnoreWriter) {
    std::shared_ptr<arrow::Table> tableCollisions = collisionsBuilder.finalize();
    std::string tableName("O2collision");
    writeTableToFile(outfile, tableCollisions, tableName, tfNumber);
  }

  // filling BC table
  // TODO: get real triggerMask
  uint64_t triggerMask = 1;
  std::sort(BCIDs.begin(), BCIDs.end());
  uint64_t prevBCid = BCIDs.back();
  for (auto& id : BCIDs) {
    if (id == prevBCid) {
      continue;
    }
    bcCursor(0,
             runNumber,
             startBCofTF + minGlBC + id,
             triggerMask);
    prevBCid = id;
  }

  BCIDs.clear();

  if (mIgnoreWriter) {
    std::shared_ptr<arrow::Table> tableBC = bcBuilder.finalize();
    std::string tableName("O2bc");
    writeTableToFile(outfile, tableBC, tableName, tfNumber);
  }

  // filling mc particles table
  TripletsMap_t toStore;
  fillMCParticlesTable(mcReader, mcParticlesCursor, tracksITSMCTruth, tracksTPCMCTruth, toStore);
  if (mIgnoreWriter) {
    std::shared_ptr<arrow::Table> tableMCParticles = mcParticlesBuilder.finalize();
    std::string tableName("O2mcparticle");
    writeTableToFile(outfile, tableMCParticles, tableName, tfNumber);
  }

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
    fillTracksTable(tracksITS, vCollRefsITS, tracksCursor, tracksCovCursor, tracksExtraCursor, GIndex::Source::ITS);
    for (auto& mcTruthITS : tracksITSMCTruth) {
      labelID = std::numeric_limits<uint32_t>::max();
      // TODO: fill label mask
      labelMask = 0;
      if (mcTruthITS.isValid()) {
        labelID = toStore.at(Triplet_t(mcTruthITS.getSourceID(), mcTruthITS.getEventID(), mcTruthITS.getTrackID()));
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
    fillTracksTable(tracksTPC, vCollRefsTPC, tracksCursor, tracksCovCursor, tracksExtraCursor, GIndex::Source::TPC);
    for (auto& mcTruthTPC : tracksTPCMCTruth) {
      labelID = std::numeric_limits<uint32_t>::max();
      // TODO: fill label mask
      labelMask = 0;
      if (mcTruthTPC.isValid()) {
        labelID = toStore.at(Triplet_t(mcTruthTPC.getSourceID(), mcTruthTPC.getEventID(), mcTruthTPC.getTrackID()));
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
    fillTracksTable(tracksITSTPC, vCollRefsTPCITS, tracksCursor, tracksCovCursor, tracksExtraCursor, GIndex::Source::ITSTPC);
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
        labelITS = toStore.at(Triplet_t(mcTruthITS.getSourceID(), mcTruthITS.getEventID(), mcTruthITS.getTrackID()));
        labelTPC = toStore.at(Triplet_t(mcTruthTPC.getSourceID(), mcTruthTPC.getEventID(), mcTruthTPC.getTrackID()));
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

  toStore.clear();

  if (!mIgnoreWriter) {
    pc.outputs().snapshot(Output{"TFN", "TFNumber", 0, Lifetime::Timeframe}, tfNumber);
  }

  if (mIgnoreWriter) {
    std::shared_ptr<arrow::Table> tableTracks = tracksBuilder.finalize();
    std::string tableName("O2track");
    writeTableToFile(outfile, tableTracks, tableName, tfNumber);
    std::shared_ptr<arrow::Table> tableTracksCov = tracksCovBuilder.finalize();
    tableName = "O2trackcov";
    writeTableToFile(outfile, tableTracksCov, tableName, tfNumber);
    std::shared_ptr<arrow::Table> tableTracksExtra = tracksExtraBuilder.finalize();
    tableName = "O2trackextra";
    writeTableToFile(outfile, tableTracksExtra, tableName, tfNumber);
    std::shared_ptr<arrow::Table> tableMCTrackLabels = mcTrackLabelBuilder.finalize();
    tableName = "O2mctracklabel";
    writeTableToFile(outfile, tableMCTrackLabels, tableName, tfNumber);
    outfile->Close();
    delete outfile;
  }

  mTimer.Stop();
}

void AODProducerWorkflowDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "aod producer dpl total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getAODProducerWorkflowSpec(int mIgnoreWriter)
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;

  inputs.emplace_back("ft0ChData", "FT0", "RECCHDATA", 0, Lifetime::Timeframe);
  inputs.emplace_back("ft0RecPoints", "FT0", "RECPOINTS", 0, Lifetime::Timeframe);
  inputs.emplace_back("primVer2TRefs", "GLO", "PVTX_TRMTCREFS", 0, Lifetime::Timeframe);
  inputs.emplace_back("primVerGIs", "GLO", "PVTX_TRMTC", 0, Lifetime::Timeframe);
  inputs.emplace_back("primVertices", "GLO", "PVTX", 0, Lifetime::Timeframe);
  inputs.emplace_back("primVerLabels", "GLO", "PVTX_MCTR", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackITS", "ITS", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("tracksITSTPC", "GLO", "TPCITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackTPC", "TPC", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackTPCMCTruth", "TPC", "TRACKSMCLBL", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackITSMCTruth", "ITS", "TRACKSMCTR", 0, Lifetime::Timeframe);

  if (!mIgnoreWriter) {
    outputs.emplace_back(OutputLabel{"O2bc"}, "AOD", "BC", 0, Lifetime::Timeframe);
    outputs.emplace_back(OutputLabel{"O2collision"}, "AOD", "COLLISION", 0, Lifetime::Timeframe);
    outputs.emplace_back(OutputLabel{"O2ft0"}, "AOD", "FT0", 0, Lifetime::Timeframe);
    outputs.emplace_back(OutputLabel{"O2mccollision"}, "AOD", "MCCOLLISION", 0, Lifetime::Timeframe);
    outputs.emplace_back(OutputLabel{"O2mccollisionlabel"}, "AOD", "MCCOLLISLABEL", 0, Lifetime::Timeframe);
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
  }

  return DataProcessorSpec{
    "aod-producer-workflow",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<AODProducerWorkflowDPL>(mIgnoreWriter)},
    Options{
      ConfigParamSpec{"fill-tracks-its", VariantType::Int, 1, {"Fill ITS tracks into tracks table"}},
      ConfigParamSpec{"fill-tracks-tpc", VariantType::Int, 0, {"Fill TPC tracks into tracks table"}},
      ConfigParamSpec{"fill-tracks-its-tpc", VariantType::Int, 1, {"Fill ITS-TPC tracks into tracks table"}},
      ConfigParamSpec{"aod-timeframe-id", VariantType::Int64, -1L, {"Set timeframe number"}},
      ConfigParamSpec{"enable-truncation", VariantType::Int, 1, {"Truncation parameter: 1 -- on, != 1 -- off"}},
      ConfigParamSpec{"reco-mctracks-only", VariantType::Int, 0, {"Store only reconstructed MC tracks and their mothers/daughters. 0 -- off, != 0 -- on"}}}};
}

} // namespace o2::aodproducer
