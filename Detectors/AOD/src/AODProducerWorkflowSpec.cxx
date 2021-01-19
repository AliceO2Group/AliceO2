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
#include "DetectorsCommonDataFormats/NameConf.h"
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
#include "Headers/DataHeader.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "Steer/MCKinematicsReader.h"
#include "TMath.h"
#include <map>
#include <vector>

using namespace o2::framework;

namespace o2::aodproducer
{

float AODProducerWorkflowDPL::TruncateFloatFraction(float x, uint32_t mask)
{
  union {
    float y;
    uint32_t iy;
  } myu;
  myu.y = x;
  myu.iy &= mask;
  return myu.y;
}

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

int64_t AODProducerWorkflowDPL::getTFNumber(uint64_t firstVtxGlBC, int runNumber)
{
  // FIXME:
  // check if this code is correct

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
  // firstVtxGlBC --> calculated using global BC correspinding to the first prim. vertex

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

template <typename TracksType, typename TracksCursorType>
void AODProducerWorkflowDPL::fillTracksTable(const TracksType& tracks, std::vector<int>& vCollRefs, const TracksCursorType& tracksCursor, int trackType)
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
    // FIXME:
    // is there a more nice/simple way to do this?
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
                 TruncateFloatFraction(track.getX(), mTrackX),
                 TruncateFloatFraction(track.getAlpha(), mTrackAlpha),
                 track.getY(),
                 track.getZ(),
                 TruncateFloatFraction(track.getSnp(), mTrackSnp),
                 TruncateFloatFraction(track.getTgl(), mTrackTgl),
                 TruncateFloatFraction(track.getQ2Pt(), mTrack1Pt),
                 TruncateFloatFraction(TMath::Sqrt(track.getSigmaY2()), mTrackCovDiag),
                 TruncateFloatFraction(TMath::Sqrt(track.getSigmaZ2()), mTrackCovDiag),
                 TruncateFloatFraction(TMath::Sqrt(track.getSigmaSnp2()), mTrackCovDiag),
                 TruncateFloatFraction(TMath::Sqrt(track.getSigmaTgl2()), mTrackCovDiag),
                 TruncateFloatFraction(TMath::Sqrt(track.getSigma1Pt2()), mTrackCovDiag),
                 (Char_t)(128. * track.getSigmaZY() / track.getSigmaZ2() / track.getSigmaY2()),
                 (Char_t)(128. * track.getSigmaSnpY() / track.getSigmaSnp2() / track.getSigmaY2()),
                 (Char_t)(128. * track.getSigmaSnpZ() / track.getSigmaSnp2() / track.getSigmaZ2()),
                 (Char_t)(128. * track.getSigmaTglY() / track.getSigmaTgl2() / track.getSigmaY2()),
                 (Char_t)(128. * track.getSigmaTglZ() / track.getSigmaTgl2() / track.getSigmaZ2()),
                 (Char_t)(128. * track.getSigmaTglSnp() / track.getSigmaTgl2() / track.getSigmaSnp2()),
                 (Char_t)(128. * track.getSigma1PtY() / track.getSigma1Pt2() / track.getSigmaY2()),
                 (Char_t)(128. * track.getSigma1PtZ() / track.getSigma1Pt2() / track.getSigmaZ2()),
                 (Char_t)(128. * track.getSigma1PtSnp() / track.getSigma1Pt2() / track.getSigmaSnp2()),
                 (Char_t)(128. * track.getSigma1PtTgl() / track.getSigma1Pt2() / track.getSigmaTgl2()),
                 TruncateFloatFraction(tpcInnerParam, mTrack1Pt),
                 flags,
                 itsClusterMap,
                 tpcNClsFindable,
                 tpcNClsFindableMinusFound,
                 tpcNClsFindableMinusCrossedRows,
                 tpcNClsShared,
                 trdPattern,
                 TruncateFloatFraction(itsChi2NCl, mTrackCovOffDiag),
                 TruncateFloatFraction(tpcChi2NCl, mTrackCovOffDiag),
                 TruncateFloatFraction(trdChi2, mTrackCovOffDiag),
                 TruncateFloatFraction(tofChi2, mTrackCovOffDiag),
                 TruncateFloatFraction(tpcSignal, mTrackSignal),
                 TruncateFloatFraction(trdSignal, mTrackSignal),
                 TruncateFloatFraction(tofSignal, mTrackSignal),
                 TruncateFloatFraction(length, mTrackSignal),
                 TruncateFloatFraction(tofExpMom, mTrack1Pt),
                 TruncateFloatFraction(trackEtaEMCAL, mTrackPosEMCAL),
                 TruncateFloatFraction(trackPhiEMCAL, mTrackPosEMCAL));
  }
}

void AODProducerWorkflowDPL::init(InitContext& ic)
{
  mTimer.Stop();

  mFillTracksITS = ic.options().get<int>("fill-tracks-its");
  mFillTracksTPC = ic.options().get<int>("fill-tracks-tpc");
  mFillTracksITSTPC = ic.options().get<int>("fill-tracks-its-tpc");
  mTFNumber = ic.options().get<int>("aod-timeframe-id");
  if (mTFNumber == -1) {
    LOG(INFO) << "TFNumber will be obtained from CCDB";
  }
  LOG(INFO) << "Track filling flags are set to: "
            << "\n ITS = " << mFillTracksITS << "\n TPC = " << mFillTracksTPC << "\n ITSTPC = " << mFillTracksITSTPC;

  mTruncate = ic.options().get<int>("enable-truncation");
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
  auto tracksITSTPC_ITSMC = pc.inputs().get<gsl::span<o2::MCCompLabel>>("tracksITSTPC_ITSMC");
  auto tracksITSTPC_TPCMC = pc.inputs().get<gsl::span<o2::MCCompLabel>>("tracksITSTPC_TPCMC");

  LOG(DEBUG) << "FOUND " << tracksTPC.size() << " TPC tracks";
  LOG(DEBUG) << "FOUND " << tracksITS.size() << " ITS tracks";
  LOG(DEBUG) << "FOUND " << tracksITSTPC.size() << " ITCTPC tracks";

  auto& bcBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "BC"});
  auto& collisionsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "COLLISION"});
  auto& ft0Builder = pc.outputs().make<TableBuilder>(Output{"AOD", "FT0"});
  auto& mcCollisionsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCCOLLISION"});
  auto& tracksBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "TRACK"});
  auto& mcParticlesBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCPARTICLE"});
  auto& mcTrackLabelBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCTRACKLABEL"});
  auto& timeFrameNumberBuilder = pc.outputs().make<uint64_t>(Output{"TFN", "TFNumber"});

  auto& fv0aBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FV0A"});
  auto& fddBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FDD"});
  auto& fv0cBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FV0C"});
  auto& zdcBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "ZDC"});

  auto bcCursor = bcBuilder.cursor<o2::aod::BCs>();
  auto collisionsCursor = collisionsBuilder.cursor<o2::aod::Collisions>();
  auto ft0Cursor = ft0Builder.cursor<o2::aod::FT0s>();
  auto mcCollisionsCursor = mcCollisionsBuilder.cursor<o2::aod::McCollisions>();
  auto tracksCursor = tracksBuilder.cursor<o2::aodproducer::TracksTable>();
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

  std::map<uint64_t, uint64_t> mGlobBC2BCID;

  // TODO:
  // get real run number and triggerMask
  int runNumber = 244918;
  uint64_t triggerMask = 1;

  // filling BC table and map<globalBC, BCId>
  for (uint64_t i = 0; i <= maxGlBC - minGlBC; i++) {
    bcCursor(0,
             runNumber,
             minGlBC + i,
             triggerMask);
    mGlobBC2BCID[minGlBC + i] = i;
  }

  // TODO:
  // add real FV0A, FV0C, FDD, ZDC tables instead of dummies
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

  // TODO:
  // figure out collision weight
  float mcColWeight = 1.;

  // filling mcCollison table
  int index = 0;
  for (auto& rec : mcRecords) {
    auto time = rec.getTimeNS();
    uint64_t globalBC = rec.bc + rec.orbit * o2::constants::lhc::LHCMaxBunches;
    auto& colParts = mcParts[index];
    for (int i = 0; i < colParts.size(); i++) {
      auto eventID = colParts[i].entryID;
      auto sourceID = colParts[i].sourceID;
      // FIXME:
      // use generators' names for generatorIDs (?)
      short generatorID = sourceID;
      auto& header = mcReader.getMCEventHeader(sourceID, eventID);
      mcCollisionsCursor(0,
                         mGlobBC2BCID.at(globalBC),
                         generatorID,
                         TruncateFloatFraction(header.GetX(), mCollisionPosition),
                         TruncateFloatFraction(header.GetY(), mCollisionPosition),
                         TruncateFloatFraction(header.GetZ(), mCollisionPosition),
                         TruncateFloatFraction(time, mCollisionPosition),
                         TruncateFloatFraction(mcColWeight, mCollisionPosition),
                         header.GetB());
    }
    index++;
  }

  // tracks --> mc particles
  // std::map<<sourceID, eventID, trackID>, McParticles::Index>
  std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint32_t> mIDsToIndex;

  // filling mcparticle table
  uint32_t mcParticlesIndex = 0;
  for (int sourceID = 0; sourceID < mcReader.getNSources(); sourceID++) {
    for (int mcEventID = 0; mcEventID < mcReader.getNEvents(sourceID); mcEventID++) {
      std::vector<MCTrack> const& mcParticles = mcReader.getTracks(sourceID, mcEventID);

      // TODO
      //  *fill dummy columns
      //  *mother/daughter IDs need to be recalculated before storing into table
      int statusCode = 0;
      uint8_t flags = 0;
      float weight = 0.f;
      int mother0 = 0;
      int mother1 = 0;
      int daughter0 = 0;
      int daughter1 = 0;

      int mcTrackID = 0;
      for (auto& mcParticle : mcParticles) {
        mcParticlesCursor(0,
                          mcEventID,
                          mcParticle.GetPdgCode(),
                          statusCode,
                          flags,
                          mother0,
                          mother1,
                          daughter0,
                          daughter1,
                          TruncateFloatFraction(weight, mMcParticleW),
                          TruncateFloatFraction((float)mcParticle.Px(), mMcParticleMom),
                          TruncateFloatFraction((float)mcParticle.Py(), mMcParticleMom),
                          TruncateFloatFraction((float)mcParticle.Pz(), mMcParticleMom),
                          TruncateFloatFraction((float)mcParticle.GetEnergy(), mMcParticleMom),
                          TruncateFloatFraction((float)mcParticle.Vx(), mMcParticlePos),
                          TruncateFloatFraction((float)mcParticle.Vy(), mMcParticlePos),
                          TruncateFloatFraction((float)mcParticle.Vz(), mMcParticlePos),
                          TruncateFloatFraction((float)mcParticle.T(), mMcParticlePos));
        mIDsToIndex[std::make_tuple(sourceID, mcEventID, mcTrackID)] = mcParticlesIndex;
        mcTrackID++;
        mcParticlesIndex++;
      }
    }
  }

  // vector of FT0 amplitudes
  std::vector<float> vAmplitudes(208, 0.);
  // filling FT0 table
  for (auto& ft0RecPoint : ft0RecPoints) {
    const auto channelData = ft0RecPoint.getBunchChannelData(ft0ChData);
    // TODO:
    // switch to calibrated amplitude
    for (auto& channel : channelData) {
      vAmplitudes[channel.ChId] = channel.QTCAmpl; // amplitude, mV
    }
    float aAmplitudesA[96];
    float aAmplitudesC[112];
    for (int i = 0; i < 96; i++) {
      aAmplitudesA[i] = TruncateFloatFraction(vAmplitudes[i], mT0Amplitude);
    }
    for (int i = 0; i < 112; i++) {
      aAmplitudesC[i] = TruncateFloatFraction(vAmplitudes[i + 96], mT0Amplitude);
    }
    uint64_t globalBC = ft0RecPoint.getInteractionRecord().orbit * o2::constants::lhc::LHCMaxBunches + ft0RecPoint.getInteractionRecord().bc;
    ft0Cursor(0,
              mGlobBC2BCID.at(globalBC),
              aAmplitudesA,
              aAmplitudesC,
              TruncateFloatFraction(ft0RecPoint.getCollisionTimeA() / 1E3, mT0Time), // ps to ns
              TruncateFloatFraction(ft0RecPoint.getCollisionTimeC() / 1E3, mT0Time), // ps to ns
              ft0RecPoint.getTrigger().triggersignals);
  }

  // initializing vectors for trackID --> collisionID connection
  std::vector<int> vCollRefsITS(tracksITS.size(), -1);
  std::vector<int> vCollRefsTPC(tracksTPC.size(), -1);
  std::vector<int> vCollRefsTPCITS(tracksITSTPC.size(), -1);

  // global bc of the 1st vertex for TF number
  uint64_t firstVtxGlBC;

  // filling collisions table
  int collisionID = 0;
  for (auto& vertex : primVertices) {
    auto& cov = vertex.getCov();
    auto& timeStamp = vertex.getTimeStamp();
    Double_t tsTimeStamp = timeStamp.getTimeStamp() * 1E3; // mus to ns
    // FIXME:
    // should use IRMin and IRMax for globalBC calculation
    uint64_t globalBC = std::round(tsTimeStamp / o2::constants::lhc::LHCBunchSpacingNS);

    LOG(DEBUG) << globalBC << " " << tsTimeStamp;

    if (collisionID == 0) {
      firstVtxGlBC = globalBC;
    }

    // collision timestamp in ns wrt the beginning of collision BC
    tsTimeStamp = globalBC * o2::constants::lhc::LHCBunchSpacingNS - tsTimeStamp;
    int BCid = mGlobBC2BCID.at(globalBC);
    // TODO:
    // get real collision time mask
    int collisionTimeMask = 0;
    collisionsCursor(0,
                     BCid,
                     TruncateFloatFraction(vertex.getX(), mCollisionPosition),
                     TruncateFloatFraction(vertex.getY(), mCollisionPosition),
                     TruncateFloatFraction(vertex.getZ(), mCollisionPosition),
                     TruncateFloatFraction(cov[0], mCollisionPositionCov),
                     TruncateFloatFraction(cov[1], mCollisionPositionCov),
                     TruncateFloatFraction(cov[2], mCollisionPositionCov),
                     TruncateFloatFraction(cov[3], mCollisionPositionCov),
                     TruncateFloatFraction(cov[4], mCollisionPositionCov),
                     TruncateFloatFraction(cov[5], mCollisionPositionCov),
                     TruncateFloatFraction(vertex.getChi2(), mCollisionPositionCov),
                     vertex.getNContributors(),
                     TruncateFloatFraction(tsTimeStamp, mCollisionPosition),
                     TruncateFloatFraction(timeStamp.getTimeStampError() * 1E3, mCollisionPositionCov),
                     collisionTimeMask);

    auto trackRef = primVer2TRefs[collisionID];
    int start = trackRef.getFirstEntryOfSource(0);
    int ntracks = trackRef.getEntriesOfSource(0);

    // FIXME:
    // `track<-->vertex` ambiguity is not accounted for in this code
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
    fillTracksTable(tracksITS, vCollRefsITS, tracksCursor, o2::vertexing::GIndex::Source::ITS); // fTrackType = 1
    for (auto& mcTruthITS : tracksITSMCTruth) {
      labelID = std::numeric_limits<uint32_t>::max();
      // TODO:
      // fill label mask
      labelMask = 0;
      if (mcTruthITS.isValid()) {
        labelID = mIDsToIndex.at(std::make_tuple(mcTruthITS.getSourceID(), mcTruthITS.getEventID(), mcTruthITS.getTrackID()));
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
    fillTracksTable(tracksTPC, vCollRefsTPC, tracksCursor, o2::vertexing::GIndex::Source::TPC); // fTrackType = 2
    for (auto& mcTruthTPC : tracksTPCMCTruth) {
      labelID = std::numeric_limits<uint32_t>::max();
      // TODO:
      // fill label mask
      labelMask = 0;
      if (mcTruthTPC.isValid()) {
        labelID = mIDsToIndex.at(std::make_tuple(mcTruthTPC.getSourceID(), mcTruthTPC.getEventID(), mcTruthTPC.getTrackID()));
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
    fillTracksTable(tracksITSTPC, vCollRefsTPCITS, tracksCursor, o2::vertexing::GIndex::Source::ITSTPC); // fTrackType = 0
    for (int i = 0; i < tracksITSTPC.size(); i++) {
      auto& mcTruthITS = tracksITSTPC_ITSMC[i];
      auto& mcTruthTPC = tracksITSTPC_TPCMC[i];
      labelID = std::numeric_limits<uint32_t>::max();
      labelITS = std::numeric_limits<uint32_t>::max();
      labelTPC = std::numeric_limits<uint32_t>::max();
      // TODO:
      // fill label mask
      // currently using label mask to indicate labelITS != labelTPC
      labelMask = 0;
      if (mcTruthITS.isValid() && mcTruthTPC.isValid()) {
        labelITS = mIDsToIndex.at(std::make_tuple(mcTruthITS.getSourceID(), mcTruthITS.getEventID(), mcTruthITS.getTrackID()));
        labelTPC = mIDsToIndex.at(std::make_tuple(mcTruthTPC.getSourceID(), mcTruthTPC.getEventID(), mcTruthTPC.getTrackID()));
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

  if (mTFNumber == -1) {
    timeFrameNumberBuilder = getTFNumber(firstVtxGlBC, runNumber);
  } else {
    timeFrameNumberBuilder = mTFNumber;
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
  inputs.emplace_back("tracksITSTPC_ITSMC", "GLO", "TPCITS_ITSMC", 0, Lifetime::Timeframe);
  inputs.emplace_back("tracksITSTPC_TPCMC", "GLO", "TPCITS_TPCMC", 0, Lifetime::Timeframe);

  outputs.emplace_back(OutputLabel{"O2bc"}, "AOD", "BC", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2collision"}, "AOD", "COLLISION", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2ft0"}, "AOD", "FT0", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mccollision"}, "AOD", "MCCOLLISION", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2track"}, "AOD", "TRACK", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mcparticle"}, "AOD", "MCPARTICLE", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mctracklabel"}, "AOD", "MCTRACKLABEL", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputSpec{"TFN", "TFNumber"});

  // TODO:
  // add  FV0A, FV0C, FDD tables
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
      ConfigParamSpec{"fill-tracks-tpc", VariantType::Int, 1, {"Fill TPC tracks into tracks table"}},
      ConfigParamSpec{"fill-tracks-its-tpc", VariantType::Int, 1, {"Fill ITS-TPC tracks into tracks table"}},
      ConfigParamSpec{"aod-timeframe-id", VariantType::Int, -1, {"Set timeframe number"}},
      ConfigParamSpec{"enable-truncation", VariantType::Int, 1, {"Truncation parameter: 1 -- on (default), != 1 -- off"}}}};
}

} // namespace o2::aodproducer
