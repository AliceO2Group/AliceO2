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
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "CCDB/BasicCCDBManager.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataTypes.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Logger.h"
#include "Framework/TableBuilder.h"
#include "Framework/TableTreeHelpers.h"
#include "GlobalTracking/MatchTOF.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/V0.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "FT0Base/Geometry.h"
#include "TMath.h"
#include "MathUtils/Utils.h"
#include <map>
#include <unordered_map>
#include <vector>

using namespace o2::framework;
using namespace o2::math_utils::detail;
using PVertex = o2::dataformats::PrimaryVertex;
using V2TRef = o2::dataformats::VtxTrackRef;
using GIndex = o2::dataformats::VtxTrackIndex;
using DataRequest = o2::globaltracking::DataRequest;
using GID = o2::dataformats::GlobalTrackID;

namespace o2::aodproducer
{

// using global variables to pass parameters to extra table filler
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

void AODProducerWorkflowDPL::collectBCs(gsl::span<const o2::ft0::RecPoints>& ft0RecPoints,
                                        gsl::span<const o2::dataformats::PrimaryVertex>& primVertices,
                                        const std::vector<o2::InteractionTimeRecord>& mcRecords,
                                        std::map<uint64_t, int>& bcsMap)
{
  // collecting non-empty BCs and enumerating them
  for (auto& rec : mcRecords) {
    uint64_t globalBC = rec.toLong();
    bcsMap[globalBC] = 1;
  }

  for (auto& ft0RecPoint : ft0RecPoints) {
    uint64_t globalBC = ft0RecPoint.getInteractionRecord().toLong();
    bcsMap[globalBC] = 1;
  }

  for (auto& vertex : primVertices) {
    auto& timeStamp = vertex.getTimeStamp();
    double tsTimeStamp = timeStamp.getTimeStamp() * 1E3; // mus to ns
    uint64_t globalBC = std::round(tsTimeStamp / o2::constants::lhc::LHCBunchSpacingNS);
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
  uint32_t firstRecOrbit = initialOrbit + tfStartIR.orbit;
  const o2::InteractionRecord firstRec(firstRecBC, firstRecOrbit);
  ts += firstRec.bc2ns() / 1000000;

  return ts;
};

template <typename TracksCursorType, typename TracksCovCursorType>
void AODProducerWorkflowDPL::addToTracksTable(TracksCursorType& tracksCursor, TracksCovCursorType& tracksCovCursor,
                                              const o2::track::TrackParCov& track, int collisionID, int src)
{
  // tracks
  tracksCursor(0,
               collisionID,
               src,
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
void AODProducerWorkflowDPL::addToTracksExtraTable(TracksExtraCursorType& tracksExtraCursor)
{
  // extra
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

template <typename MCParticlesCursorType>
void AODProducerWorkflowDPL::fillMCParticlesTable(o2::steer::MCKinematicsReader& mcReader, const MCParticlesCursorType& mcParticlesCursor,
                                                  gsl::span<const o2::MCCompLabel>& mcTruthITS, std::vector<bool>& isStoredITS,
                                                  gsl::span<const o2::MCCompLabel>& mcTruthMFT, std::vector<bool>& isStoredMFT,
                                                  gsl::span<const o2::MCCompLabel>& mcTruthTPC, std::vector<bool>& isStoredTPC,
                                                  TripletsMap_t& toStore)
{
  // mark reconstructed MC particles to store them into the table
  for (int i = 0; i < mcTruthITS.size(); i++) {
    auto& mcTruth = mcTruthITS[i];
    if (!mcTruth.isValid() || !isStoredITS[i]) {
      continue;
    }
    int source = mcTruth.getSourceID();
    int event = mcTruth.getEventID();
    int particle = mcTruth.getTrackID();
    toStore[Triplet_t(source, event, particle)] = 1;
  }
  for (int i = 0; i < mcTruthMFT.size(); i++) {
    auto& mcTruth = mcTruthMFT[i];
    if (!mcTruth.isValid() || !isStoredMFT[i]) {
      continue;
    }
    int source = mcTruth.getSourceID();
    int event = mcTruth.getEventID();
    int particle = mcTruth.getTrackID();
    toStore[Triplet_t(source, event, particle)] = 1;
  }
  for (int i = 0; i < mcTruthTPC.size(); i++) {
    auto& mcTruth = mcTruthTPC[i];
    if (!mcTruth.isValid() || !isStoredTPC[i]) {
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

void AODProducerWorkflowDPL::init(InitContext& ic)
{
  mTimer.Stop();

  mFillTracksITS = ic.options().get<int>("fill-tracks-its");
  mFillTracksMFT = ic.options().get<int>("fill-tracks-mft");
  mFillTracksTPC = ic.options().get<int>("fill-tracks-tpc");
  mFillTracksITSTPC = ic.options().get<int>("fill-tracks-its-tpc");
  mTFNumber = ic.options().get<int64_t>("aod-timeframe-id");
  mRecoOnly = ic.options().get<int>("reco-mctracks-only");
  mTruncate = ic.options().get<int>("enable-truncation");

  if (mTFNumber == -1L) {
    LOG(INFO) << "TFNumber will be obtained from CCDB";
  }

  LOG(INFO) << "Track filling flags are set to: "
            << "\n ITS = " << mFillTracksITS << "\n MFT = " << mFillTracksMFT << "\n TPC = " << mFillTracksTPC << "\n ITSTPC = " << mFillTracksITSTPC;

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

  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest);

  auto primVertices = recoData.getPrimaryVertices();
  auto primVer2TRefs = recoData.getPrimaryVertexMatchedTrackRefs();
  auto primVerGIs = recoData.getPrimaryVertexMatchedTracks();
  auto primVerLabels = recoData.getPrimaryVertexMCLabels();

  // temporary placeholder
  if (mFillSVertices) {
    auto secVertices = recoData.getV0s();
    auto p2secRefs = recoData.getPV2V0Refs();
  }

  auto ft0ChData = recoData.getFT0ChannelsData();
  auto ft0RecPoints = recoData.getFT0RecPoints();

  auto tracksITS = recoData.getITSTracks();
  auto tracksMFT = recoData.getMFTTracks();
  auto tracksTPC = recoData.getTPCTracks();
  auto tracksITSTPC = recoData.getTPCITSTracks();

  auto tracksTPCMCTruth = recoData.getTPCTracksMCLabels();
  auto tracksITSMCTruth = recoData.getITSTracksMCLabels();
  auto tracksMFTMCTruth = recoData.getMFTTracksMCLabels();

  // using vectors to mark referenced tracks
  // todo: should not use these (?), to be removed, when all track types are processed
  std::vector<bool> isStoredTPC(tracksTPC.size(), false);
  std::vector<bool> isStoredITS(tracksITS.size(), false);
  std::vector<bool> isStoredMFT(tracksMFT.size(), false);

  LOG(DEBUG) << "FOUND " << primVertices.size() << " primary vertices";
  LOG(DEBUG) << "FOUND " << tracksTPC.size() << " TPC tracks";
  LOG(DEBUG) << "FOUND " << tracksTPCMCTruth.size() << " TPC labels";
  LOG(DEBUG) << "FOUND " << tracksMFT.size() << " MFT tracks";
  LOG(DEBUG) << "FOUND " << tracksMFTMCTruth.size() << " MFT labels";
  LOG(DEBUG) << "FOUND " << tracksITS.size() << " ITS tracks";
  LOG(DEBUG) << "FOUND " << tracksITSMCTruth.size() << " ITS labels";
  LOG(DEBUG) << "FOUND " << tracksITSTPC.size() << " ITSTPC tracks";
  LOG(DEBUG) << "FOUND " << ft0RecPoints.size() << " FT0 rec. points";

  auto& bcBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "BC"});
  auto& collisionsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "COLLISION"});
  auto& mcColLabelsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCCOLLISIONLABEL"});
  auto& ft0Builder = pc.outputs().make<TableBuilder>(Output{"AOD", "FT0"});
  auto& mcCollisionsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCCOLLISION"});
  auto& tracksBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "TRACK"});
  auto& tracksCovBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "TRACKCOV"});
  auto& tracksExtraBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "TRACKEXTRA"});
  auto& mftTracksBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MFTTRACK"});
  auto& mcParticlesBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCPARTICLE"});
  auto& mcTrackLabelBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCTRACKLABEL"});
  auto& fv0aBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FV0A"});
  auto& fddBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FDD"});
  auto& fv0cBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FV0C"});
  auto& zdcBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "ZDC"});

  auto bcCursor = bcBuilder.cursor<o2::aod::BCs>();
  auto collisionsCursor = collisionsBuilder.cursor<o2::aod::Collisions>();
  auto mcColLabelsCursor = mcColLabelsBuilder.cursor<o2::aod::McCollisionLabels>();
  auto ft0Cursor = ft0Builder.cursor<o2::aod::FT0s>();
  auto mcCollisionsCursor = mcCollisionsBuilder.cursor<o2::aod::McCollisions>();
  auto tracksCursor = tracksBuilder.cursor<o2::aodproducer::TracksTable>();
  auto tracksCovCursor = tracksCovBuilder.cursor<o2::aodproducer::TracksCovTable>();
  auto tracksExtraCursor = tracksExtraBuilder.cursor<o2::aodproducer::TracksExtraTable>();
  auto mftTracksCursor = mftTracksBuilder.cursor<o2::aodproducer::MFTTracksTable>();
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

  std::map<uint64_t, int> bcsMap;
  collectBCs(ft0RecPoints, primVertices, mcRecords, bcsMap);

  const auto* dh = o2::header::get<o2::header::DataHeader*>(pc.inputs().getByPos(0).header);
  o2::InteractionRecord startIR = {0, dh->firstTForbit};

  uint64_t tfNumber;
  // default dummy run number
  int runNumber = 244918; // TODO: get real run number
  if (mTFNumber == -1L) {
    tfNumber = getTFNumber(startIR, runNumber);
  } else {
    tfNumber = mTFNumber;
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

  float dummyFV0AmplC[32] = {0.};
  fv0cCursor(0,
             dummyBC,
             dummyFV0AmplC,
             dummyTime);

  float dummyFDDAmplA[4] = {0.};
  float dummyFDDAmplC[4] = {0.};
  fddCursor(0,
            dummyBC,
            dummyFDDAmplA,
            dummyFDDAmplC,
            dummyTime,
            dummyTime,
            dummyTriggerMask);

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
    }
    index++;
  }

  // vector of FT0 amplitudes
  int nFT0Channels = o2::ft0::Geometry::Nchannels;
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
    float aAmplitudesC[133];
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
              truncateFloatFraction(ft0RecPoint.getCollisionTimeA() / 1E3, mT0Time), // ps to ns
              truncateFloatFraction(ft0RecPoint.getCollisionTimeC() / 1E3, mT0Time), // ps to ns
              ft0RecPoint.getTrigger().triggersignals);
  }

  // filling MC collision labels
  for (auto& label : primVerLabels) {
    int32_t mcCollisionID = label.getEventID();
    uint16_t mcMask = 0; // todo: set mask using normalized weights?
    mcColLabelsCursor(0, mcCollisionID, mcMask);
  }

  // filling unassigned tracks first
  // so that all unassigned tracks are stored in the beginning of the table together
  auto& trackRefU = primVer2TRefs.back(); // references to unassigned tracks are at the end
  for (int src = GIndex::NSources; src--;) {
    int start = trackRefU.getFirstEntryOfSource(src);
    int end = start + trackRefU.getEntriesOfSource(src);
    LOG(DEBUG) << "Unassigned tracks: src = " << src << ", start = " << start << ", end = " << end;
    for (int ti = start; ti < end; ti++) {
      tpcInnerParam = 0.f;
      flags = 0;
      itsClusterMap = 0;
      tpcNClsFindable = 0;
      tpcNClsFindableMinusFound = 0;
      tpcNClsFindableMinusCrossedRows = 0;
      tpcNClsShared = 0;
      trdPattern = 0;
      itsChi2NCl = -999.f;
      tpcChi2NCl = -999.f;
      trdChi2 = -999.f;
      tofChi2 = -999.f;
      tpcSignal = -999.f;
      trdSignal = -999.f;
      tofSignal = -999.f;
      length = -999.f;
      tofExpMom = -999.f;
      trackEtaEMCAL = -999.f;
      trackPhiEMCAL = -999.f;
      auto& trackIndex = primVerGIs[ti];
      if (src == GIndex::Source::ITS && mFillTracksITS) {
        const auto& track = tracksITS[trackIndex.getIndex()];
        isStoredITS[trackIndex.getIndex()] = true;
        // extra info
        itsClusterMap = track.getPattern();
        // track
        addToTracksTable(tracksCursor, tracksCovCursor, track, -1, src);
        addToTracksExtraTable(tracksExtraCursor);
      }
      if (src == GIndex::Source::TPC && mFillTracksTPC) {
        const auto& track = tracksTPC[trackIndex.getIndex()];
        isStoredTPC[trackIndex.getIndex()] = true;
        // extra info
        tpcChi2NCl = track.getNClusters() ? track.getChi2() / track.getNClusters() : 0;
        tpcSignal = track.getdEdx().dEdxTotTPC;
        tpcNClsFindable = track.getNClusters();
        // track
        addToTracksTable(tracksCursor, tracksCovCursor, track, -1, src);
        addToTracksExtraTable(tracksExtraCursor);
      }
      if (src == GIndex::Source::ITSTPC && mFillTracksITSTPC) {
        const auto& track = tracksITSTPC[trackIndex.getIndex()];
        auto contributorsGID = recoData.getSingleDetectorRefs(trackIndex);
        // extra info from sub-tracks
        if (contributorsGID[GIndex::Source::ITS].isIndexSet()) {
          isStoredITS[track.getRefITS()] = true;
          const auto& itsOrig = recoData.getITSTrack(contributorsGID[GIndex::ITS]);
          itsClusterMap = itsOrig.getPattern();
        }
        if (contributorsGID[GIndex::Source::TPC].isIndexSet()) {
          isStoredTPC[track.getRefTPC()] = true;
          const auto& tpcOrig = recoData.getTPCTrack(contributorsGID[GIndex::TPC]);
          tpcChi2NCl = tpcOrig.getNClusters() ? tpcOrig.getChi2() / tpcOrig.getNClusters() : 0;
          tpcSignal = tpcOrig.getdEdx().dEdxTotTPC;
          tpcNClsFindable = tpcOrig.getNClusters();
        }
        addToTracksTable(tracksCursor, tracksCovCursor, track, -1, src);
        addToTracksExtraTable(tracksExtraCursor);
      }
      if (src == GIndex::Source::ITSTPCTOF && mFillTracksITSTPC) {
        auto contributorsGID = recoData.getSingleDetectorRefs(trackIndex);
        const auto& track = recoData.getITSTPCTOFTrack(contributorsGID[GIndex::Source::ITSTPCTOF]);
        const auto& tofMatch = recoData.getTOFMatch(contributorsGID[GIndex::Source::ITSTPCTOF]);
        tofChi2 = tofMatch.getChi2();
        const auto& tofInt = tofMatch.getLTIntegralOut();
        tofSignal = tofInt.getTOF(0); // fixme: what id should be used here?
        length = tofInt.getL();
        // extra info from sub-tracks
        if (contributorsGID[GIndex::Source::ITS].isIndexSet()) {
          isStoredITS[track.getRefITS()] = true;
          const auto& itsOrig = recoData.getITSTrack(contributorsGID[GIndex::ITS]);
          itsClusterMap = itsOrig.getPattern();
        }
        if (contributorsGID[GIndex::Source::TPC].isIndexSet()) {
          isStoredTPC[track.getRefTPC()] = true;
          const auto& tpcOrig = recoData.getTPCTrack(contributorsGID[GIndex::TPC]);
          tpcChi2NCl = tpcOrig.getNClusters() ? tpcOrig.getChi2() / tpcOrig.getNClusters() : 0;
          tpcSignal = tpcOrig.getdEdx().dEdxTotTPC;
          tpcNClsFindable = tpcOrig.getNClusters();
        }
        addToTracksTable(tracksCursor, tracksCovCursor, track, -1, src);
        addToTracksExtraTable(tracksExtraCursor);
      }
      if (src == GIndex::Source::MFT && mFillTracksMFT) {
        const auto& track = tracksMFT[trackIndex.getIndex()];
        isStoredMFT[trackIndex.getIndex()] = true;
        addToMFTTracksTable(mftTracksCursor, track, -1);
      }
    }
  }

  // filling collisions table
  int collisionID = 0;
  for (auto& vertex : primVertices) {
    auto& cov = vertex.getCov();
    auto& timeStamp = vertex.getTimeStamp();
    double tsTimeStamp = timeStamp.getTimeStamp() * 1E3; // mus to ns
    uint64_t globalBC = std::round(tsTimeStamp / o2::constants::lhc::LHCBunchSpacingNS);
    LOG(DEBUG) << globalBC << " " << tsTimeStamp;
    // collision timestamp in ns wrt the beginning of collision BC
    tsTimeStamp = globalBC * o2::constants::lhc::LHCBunchSpacingNS - tsTimeStamp;
    auto item = bcsMap.find(globalBC);
    int bcID = -1;
    if (item != bcsMap.end()) {
      bcID = item->second;
    } else {
      LOG(FATAL) << "Error: could not find a corresponding BC ID for a collision; BC = " << globalBC << ", collisionID = " << collisionID;
    }
    // TODO: get real collision time mask
    int collisionTimeMask = 0;
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
                     truncateFloatFraction(tsTimeStamp, mCollisionPosition),
                     truncateFloatFraction(timeStamp.getTimeStampError() * 1E3, mCollisionPositionCov),
                     collisionTimeMask);
    auto& trackRef = primVer2TRefs[collisionID];
    for (int src = GIndex::NSources; src--;) {
      int start = trackRef.getFirstEntryOfSource(src);
      int end = start + trackRef.getEntriesOfSource(src);
      LOG(DEBUG) << " ====> Collision " << collisionID << " ; src = " << src << " : ntracks = " << end - start;
      LOG(DEBUG) << "start = " << start << ", end = " << end;
      for (int ti = start; ti < end; ti++) {
        tpcInnerParam = 0.f;
        flags = 0;
        itsClusterMap = 0;
        tpcNClsFindable = 0;
        tpcNClsFindableMinusFound = 0;
        tpcNClsFindableMinusCrossedRows = 0;
        tpcNClsShared = 0;
        trdPattern = 0;
        itsChi2NCl = -999.f;
        tpcChi2NCl = -999.f;
        trdChi2 = -999.f;
        tofChi2 = -999.f;
        tpcSignal = -999.f;
        trdSignal = -999.f;
        tofSignal = -999.f;
        length = -999.f;
        tofExpMom = -999.f;
        trackEtaEMCAL = -999.f;
        trackPhiEMCAL = -999.f;
        auto& trackIndex = primVerGIs[ti];
        if (src == GIndex::Source::ITS && mFillTracksITS) {
          const auto& track = tracksITS[trackIndex.getIndex()];
          isStoredITS[trackIndex.getIndex()] = true;
          // extra info
          itsClusterMap = track.getPattern();
          // track
          addToTracksTable(tracksCursor, tracksCovCursor, track, collisionID, src);
          addToTracksExtraTable(tracksExtraCursor);
        }
        if (src == GIndex::Source::TPC && mFillTracksTPC) {
          const auto& track = tracksTPC[trackIndex.getIndex()];
          isStoredTPC[trackIndex.getIndex()] = true;
          // extra info
          tpcChi2NCl = track.getNClusters() ? track.getChi2() / track.getNClusters() : 0;
          tpcSignal = track.getdEdx().dEdxTotTPC;
          tpcNClsFindable = track.getNClusters();
          // track
          addToTracksTable(tracksCursor, tracksCovCursor, track, collisionID, src);
          addToTracksExtraTable(tracksExtraCursor);
        }
        if (src == GIndex::Source::ITSTPC && mFillTracksITSTPC) {
          const auto& track = tracksITSTPC[trackIndex.getIndex()];
          auto contributorsGID = recoData.getSingleDetectorRefs(trackIndex);
          // extra info from sub-tracks
          if (contributorsGID[GIndex::Source::ITS].isIndexSet()) {
            isStoredITS[track.getRefITS()] = true;
            const auto& itsOrig = recoData.getITSTrack(contributorsGID[GIndex::ITS]);
            itsClusterMap = itsOrig.getPattern();
          }
          if (contributorsGID[GIndex::Source::TPC].isIndexSet()) {
            isStoredTPC[track.getRefTPC()] = true;
            const auto& tpcOrig = recoData.getTPCTrack(contributorsGID[GIndex::TPC]);
            tpcChi2NCl = tpcOrig.getNClusters() ? tpcOrig.getChi2() / tpcOrig.getNClusters() : 0;
            tpcSignal = tpcOrig.getdEdx().dEdxTotTPC;
            tpcNClsFindable = tpcOrig.getNClusters();
          }
          addToTracksTable(tracksCursor, tracksCovCursor, track, collisionID, src);
          addToTracksExtraTable(tracksExtraCursor);
        }
        if (src == GIndex::Source::ITSTPCTOF && mFillTracksITSTPC) {
          auto contributorsGID = recoData.getSingleDetectorRefs(trackIndex);
          const auto& track = recoData.getITSTPCTOFTrack(contributorsGID[GIndex::Source::ITSTPCTOF]);
          const auto& tofMatch = recoData.getTOFMatch(contributorsGID[GIndex::Source::ITSTPCTOF]);
          tofChi2 = tofMatch.getChi2();
          const auto& tofInt = tofMatch.getLTIntegralOut();
          tofSignal = tofInt.getTOF(0); // fixme: what id should be used here?
          length = tofInt.getL();
          // extra info from sub-tracks
          if (contributorsGID[GIndex::Source::ITS].isIndexSet()) {
            isStoredITS[track.getRefITS()] = true;
            const auto& itsOrig = recoData.getITSTrack(contributorsGID[GIndex::ITS]);
            itsClusterMap = itsOrig.getPattern();
          }
          if (contributorsGID[GIndex::Source::TPC].isIndexSet()) {
            isStoredTPC[track.getRefTPC()] = true;
            const auto& tpcOrig = recoData.getTPCTrack(contributorsGID[GIndex::TPC]);
            tpcChi2NCl = tpcOrig.getNClusters() ? tpcOrig.getChi2() / tpcOrig.getNClusters() : 0;
            tpcSignal = tpcOrig.getdEdx().dEdxTotTPC;
            tpcNClsFindable = tpcOrig.getNClusters();
          }
          addToTracksTable(tracksCursor, tracksCovCursor, track, collisionID, src);
          addToTracksExtraTable(tracksExtraCursor);
        }
        if (src == GIndex::Source::MFT && mFillTracksMFT) {
          const auto& track = tracksMFT[trackIndex.getIndex()];
          isStoredMFT[trackIndex.getIndex()] = true;
          addToMFTTracksTable(mftTracksCursor, track, collisionID);
        }
      }
    }
    collisionID++;
  }

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
  TripletsMap_t toStore;
  fillMCParticlesTable(mcReader, mcParticlesCursor,
                       tracksITSMCTruth, isStoredITS,
                       tracksMFTMCTruth, isStoredMFT,
                       tracksTPCMCTruth, isStoredTPC,
                       toStore);

  isStoredITS.clear();
  isStoredMFT.clear();
  isStoredTPC.clear();

  // ------------------------------------------------------
  // filling track labels

  // labelMask (temporary) usage:
  //   bit 13 -- ITS and TPC labels are not equal
  //   bit 14 -- isNoise() == true
  //   bit 15 -- isFake() == true
  // labelID = std::numeric_limits<uint32_t>::max() -- label is not set

  uint32_t labelID;
  uint32_t labelITS;
  uint32_t labelTPC;
  uint16_t labelMask;

  // need to go through labels in the same order as for tracks
  for (auto& trackRef : primVer2TRefs) {
    for (int src = GIndex::NSources; src--;) {
      int start = trackRef.getFirstEntryOfSource(src);
      int end = start + trackRef.getEntriesOfSource(src);
      for (int ti = start; ti < end; ti++) {
        auto& trackIndex = primVerGIs[ti];
        labelID = std::numeric_limits<uint32_t>::max();
        labelITS = labelID;
        labelTPC = labelID;
        labelMask = 0;
        // its labels
        if (src == GIndex::Source::ITS && mFillTracksITS) {
          auto& mcTruthITS = tracksITSMCTruth[trackIndex.getIndex()];
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
        // tpc labels
        if (src == GIndex::Source::TPC && mFillTracksTPC) {
          auto& mcTruthTPC = tracksTPCMCTruth[trackIndex.getIndex()];
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
        // its-tpc labels and its-tpc-tof labels
        // todo:
        //  probably need to store both its and tpc labels
        //  for now filling only TPC label
        if ((src == GIndex::Source::ITSTPC || src == GIndex::Source::ITSTPCTOF) && mFillTracksITSTPC) {
          auto contributorsGID = recoData.getSingleDetectorRefs(trackIndex);
          auto& mcTruthITS = tracksITSMCTruth[contributorsGID[GIndex::Source::ITS].getIndex()];
          auto& mcTruthTPC = tracksTPCMCTruth[contributorsGID[GIndex::Source::TPC].getIndex()];
          // its-contributor label
          if (contributorsGID[GIndex::Source::ITS].isIndexSet()) {
            if (mcTruthITS.isValid()) {
              labelITS = toStore.at(Triplet_t(mcTruthITS.getSourceID(), mcTruthITS.getEventID(), mcTruthITS.getTrackID()));
            }
          }
          if (contributorsGID[GIndex::Source::TPC].isIndexSet()) {
            if (mcTruthTPC.isValid()) {
              labelTPC = toStore.at(Triplet_t(mcTruthTPC.getSourceID(), mcTruthTPC.getEventID(), mcTruthTPC.getTrackID()));
            }
          }
          labelID = labelTPC;
          if (mcTruthITS.isFake() || mcTruthTPC.isFake()) {
            labelMask |= (0x1 << 15);
          }
          if (mcTruthITS.isNoise() || mcTruthTPC.isNoise()) {
            labelMask |= (0x1 << 14);
          }
          if (labelITS != labelTPC) {
            LOG(DEBUG) << "ITS-TPC MCTruth: labelIDs do not match at " << trackIndex.getIndex();
            labelMask |= (0x1 << 13);
          }
          mcTrackLabelCursor(0,
                             labelID,
                             labelMask);
        }
        // mft labels
        // todo: move to a separate table
        if (src == GIndex::Source::MFT && mFillTracksMFT) {
          auto& mcTruthMFT = tracksMFTMCTruth[trackIndex.getIndex()];
          if (mcTruthMFT.isValid()) {
            labelID = toStore.at(Triplet_t(mcTruthMFT.getSourceID(), mcTruthMFT.getEventID(), mcTruthMFT.getTrackID()));
          }
          if (mcTruthMFT.isFake()) {
            labelMask |= (0x1 << 15);
          }
          if (mcTruthMFT.isNoise()) {
            labelMask |= (0x1 << 14);
          }
          mcTrackLabelCursor(0,
                             labelID,
                             labelMask);
        }
      }
    }
  }

  toStore.clear();

  pc.outputs().snapshot(Output{"TFN", "TFNumber", 0, Lifetime::Timeframe}, tfNumber);

  mTimer.Stop();
}

void AODProducerWorkflowDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "aod producer dpl total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getAODProducerWorkflowSpec(GID::mask_t src, bool useMC, bool fillSVertices)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  dataRequest->requestTracks(src, useMC);
  dataRequest->requestPrimaryVertertices(useMC);
  if (fillSVertices) {
    dataRequest->requestSecondaryVertertices(useMC);
  }
  dataRequest->requestFT0RecPoints(false);

  outputs.emplace_back(OutputLabel{"O2bc"}, "AOD", "BC", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2collision"}, "AOD", "COLLISION", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2ft0"}, "AOD", "FT0", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mccollision"}, "AOD", "MCCOLLISION", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mccollisionlabel"}, "AOD", "MCCOLLISIONLABEL", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2track"}, "AOD", "TRACK", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2trackcov"}, "AOD", "TRACKCOV", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2trackextra"}, "AOD", "TRACKEXTRA", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mfttrack"}, "AOD", "MFTTRACK", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mcparticle"}, "AOD", "MCPARTICLE", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mctracklabel"}, "AOD", "MCTRACKLABEL", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputSpec{"TFN", "TFNumber"});
  outputs.emplace_back(OutputLabel{"O2fv0a"}, "AOD", "FV0A", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2fv0c"}, "AOD", "FV0C", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2fdd"}, "AOD", "FDD", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2zdc"}, "AOD", "ZDC", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "aod-producer-workflow",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<AODProducerWorkflowDPL>(dataRequest, fillSVertices)},
    Options{
      ConfigParamSpec{"fill-tracks-its", VariantType::Int, 1, {"Fill ITS tracks into tracks table"}},
      ConfigParamSpec{"fill-tracks-mft", VariantType::Int, 1, {"Fill MFT tracks into mfttracks table"}},
      ConfigParamSpec{"fill-tracks-tpc", VariantType::Int, 0, {"Fill TPC tracks into tracks table"}},
      ConfigParamSpec{"fill-tracks-its-tpc", VariantType::Int, 1, {"Fill ITS-TPC tracks into tracks table"}},
      ConfigParamSpec{"aod-timeframe-id", VariantType::Int64, -1L, {"Set timeframe number"}},
      ConfigParamSpec{"enable-truncation", VariantType::Int, 1, {"Truncation parameter: 1 -- on, != 1 -- off"}},
      ConfigParamSpec{"reco-mctracks-only", VariantType::Int, 0, {"Store only reconstructed MC tracks and their mothers/daughters. 0 -- off, != 0 -- on"}}}};
}

} // namespace o2::aodproducer
