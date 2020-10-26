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
#include "Framework/AnalysisDataModel.h"
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

void AODProducerWorkflowDPL::findMinMaxBc(gsl::span<const o2::ft0::RecPoints>& ft0RecPoints, gsl::span<const o2::dataformats::TrackTPCITS>& tracksITSTPC)
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

  for (auto& trackITSTPC : tracksITSTPC) {
    Double_t timeStamp = trackITSTPC.getTimeMUS().getTimeStamp() * 1.E3; // ms to ns
    uint64_t bc = (int)(timeStamp / o2::constants::lhc::LHCBunchSpacingNS);
    if (minGlBC > bc) {
      minGlBC = bc;
    }
    if (maxGlBC < bc) {
      maxGlBC = bc;
    }
  }
}

template <typename TracksType, typename TracksCursorType>
void AODProducerWorkflowDPL::fillTracksTable(const TracksType& tracks, std::vector<int>& vCollRefs, const TracksCursorType& tracksCursor, int trackType)
{
  for (int i = 0; i < tracks.size(); i++) {
    auto& track = tracks[i];
    int collisionID = vCollRefs[i];
    tracksCursor(0,
                 collisionID,
                 trackType,
                 track.getX(),
                 track.getAlpha(),
                 track.getY(),
                 track.getZ(),
                 track.getSnp(),
                 track.getTgl(),
                 track.getQ2Pt(),
                 TMath::Sqrt(track.getSigmaY2()),
                 TMath::Sqrt(track.getSigmaZ2()),
                 TMath::Sqrt(track.getSigmaSnp2()),
                 TMath::Sqrt(track.getSigmaTgl2()),
                 TMath::Sqrt(track.getSigma1Pt2()),
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
}

void AODProducerWorkflowDPL::init(InitContext& ic)
{
  mTimer.Stop();

  mFillTracksITS = ic.options().get<int>("fill-tracks-its");
  mFillTracksTPC = ic.options().get<int>("fill-tracks-tpc");
  mFillTracksITSTPC = ic.options().get<int>("fill-tracks-its-tpc");
  LOG(INFO) << "track filling flags set to: "
            << "\n ITS = " << mFillTracksITS << "\n TPC = " << mFillTracksTPC << "\n ITSTPC = " << mFillTracksITSTPC;

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

  LOG(INFO) << "FOUND " << tracksTPC.size() << " TPC tracks";
  LOG(INFO) << "FOUND " << tracksITS.size() << " ITS tracks";
  LOG(INFO) << "FOUND " << tracksITSTPC.size() << " ITCTPC tracks";

  auto& bcBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "BC"});
  auto& collisionsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "COLLISION"});
  auto& ft0Builder = pc.outputs().make<TableBuilder>(Output{"AOD", "FT0"});
  auto& mcCollisionsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCCOLLISION"});
  auto& tracksBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "TRACK"});

  // TODO:
  // add  FV0A, FV0C, FDD tables
  // auto& fv0aBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FV0A"});
  // auto& fv0cBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FV0C"});
  // auto& fddBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FDD"});

  auto bcCursor = bcBuilder.cursor<o2::aod::BCs>();
  auto collisionsCursor = collisionsBuilder.cursor<o2::aod::Collisions>();
  auto ft0Cursor = ft0Builder.cursor<o2::aod::FT0s>();
  auto mcCollisionsCursor = mcCollisionsBuilder.cursor<o2::aod::McCollisions>();
  auto tracksCursor = tracksBuilder.cursor<o2::aodproducer::TracksTable>();

  // TODO:
  // add  FV0A, FV0C, FDD tables
  // auto fv0aCursor = fv0aBuilder.cursor<o2::aod::FV0As>();
  // auto fv0cCursor = fv0cBuilder.cursor<o2::aod::FV0Cs>();
  // auto fddCursor = fddBuilder.cursor<o2::aod::FDDs>();

  // float dummyfv0AmplA[48] = {0.};
  // float dummyfv0AmplC[32] = {0.};
  // float dummyfddAmplA[4] = {0.};
  // float dummyfddAmplC[4] = {0.};
  // fv0aCursor(0, (uint64_t)0, dummyfv0AmplA, 0.f, (uint8_t)0);
  // fv0cCursor(0, (uint64_t)0, dummyfv0AmplC, 0.f);
  // fddCursor(0, (uint64_t)0, dummyfddAmplA, dummyfddAmplC, 0.f, 0.f, (uint8_t)0);

  o2::steer::MCKinematicsReader mcReader("collisioncontext.root");
  const auto context = mcReader.getDigitizationContext();
  const auto& records = context->getEventRecords();
  const auto& parts = context->getEventParts();

  LOG(INFO) << "FOUND " << records.size() << " records";
  LOG(INFO) << "FOUND " << parts.size() << " parts";

  // TODO:
  // change index to BCId (?)
  // figure out generatorID and collision weight
  int index = 0;
  int generatorID = 0;
  float mcColWeight = 1.;

  // filling mcCollison table
  for (auto& rec : records) {
    auto time = rec.getTimeNS();
    auto& colparts = parts[index];
    auto eventID = colparts[0].entryID;
    auto sourceID = colparts[0].sourceID;
    auto& header = mcReader.getMCEventHeader(sourceID, eventID);
    mcCollisionsCursor(0,
                       index,
                       generatorID,
                       header.GetX(),
                       header.GetY(),
                       header.GetZ(),
                       time,
                       mcColWeight,
                       header.GetB());
    index++;
  }

  findMinMaxBc(ft0RecPoints, tracksITSTPC);

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
    std::copy(vAmplitudes.begin(), vAmplitudes.begin() + 95, aAmplitudesA);
    std::copy(vAmplitudes.begin() + 96, vAmplitudes.end(), aAmplitudesC);
    uint64_t globalBC = ft0RecPoint.getInteractionRecord().orbit * o2::constants::lhc::LHCMaxBunches + ft0RecPoint.getInteractionRecord().bc;
    ft0Cursor(0,
              mGlobBC2BCID[globalBC],
              aAmplitudesA,
              aAmplitudesC,
              ft0RecPoint.getCollisionTimeA() / 1E3, // ps to ns
              ft0RecPoint.getCollisionTimeC() / 1E3, // ps to ns
              ft0RecPoint.getTrigger().triggersignals);
  }

  // initializing vectors for trackID --> collisionID connection
  std::vector<int> vCollRefsITS(tracksITS.size(), -1);
  std::vector<int> vCollRefsTPC(tracksTPC.size(), -1);
  std::vector<int> vCollRefsTPCITS(tracksITSTPC.size(), -1);

  // filling collisions table
  int collisionID = 0;
  for (auto& vertex : primVertices) {
    auto& cov = vertex.getCov();
    auto& timeStamp = vertex.getTimeStamp();
    Double_t tsTimeStamp = timeStamp.getTimeStamp() * 1E3; // ms to ns
    uint64_t globalBC = tsTimeStamp / o2::constants::lhc::LHCBunchSpacingNS;
    int BCid = mGlobBC2BCID[globalBC];
    // TODO:
    // get real collision time mask
    int collisionTimeMask = 1;
    collisionsCursor(0,
                     BCid,
                     vertex.getX(),
                     vertex.getY(),
                     vertex.getZ(),
                     cov[0],
                     cov[1],
                     cov[2],
                     cov[3],
                     cov[4],
                     cov[5],
                     vertex.getChi2(),
                     vertex.getNContributors(),
                     timeStamp.getTimeStamp(),
                     timeStamp.getTimeStampError(),
                     collisionTimeMask);
    auto trackRef = primVer2TRefs[collisionID];
    int start = trackRef.getFirstEntryOfSource(0);
    int ntracks = trackRef.getEntriesOfSource(0);
    for (int ti = 0; ti < ntracks; ti++) {
      auto trackIndex = primVerGIs[start + ti];
      const auto source = trackIndex.getSource();
      // setting collisionID for tracks attached to vertices
      if (source == o2::vertexing::GIndex::Source::TPC) {
        vCollRefsTPC[trackIndex.getIndex()] = collisionID;
      } else if (source == o2::vertexing::GIndex::Source::ITS) {
        vCollRefsITS[trackIndex.getIndex()] = collisionID;
      } else if (source == o2::vertexing::GIndex::Source::TPCITS) {
        vCollRefsTPCITS[trackIndex.getIndex()] = collisionID;
      } else {
        LOG(WARNING) << "Unsupported track type!";
      }
    }
    collisionID++;
  }

  // filling tracks tables
  if (mFillTracksITS) {
    fillTracksTable(tracksITS, vCollRefsITS, tracksCursor, o2::vertexing::GIndex::Source::ITS); // fTrackType = 1
  }
  if (mFillTracksTPC) {
    fillTracksTable(tracksTPC, vCollRefsTPC, tracksCursor, o2::vertexing::GIndex::Source::TPC); // fTrackType = 2
  }
  if (mFillTracksITSTPC) {
    fillTracksTable(tracksITSTPC, vCollRefsTPCITS, tracksCursor, o2::vertexing::GIndex::Source::TPCITS); // fTrackType = 0
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
  inputs.emplace_back("primVer2TRefs", "GLO", "PVTX_CONTIDREFS", 0, Lifetime::Timeframe);
  inputs.emplace_back("primVerGIs", "GLO", "PVTX_CONTID", 0, Lifetime::Timeframe);
  inputs.emplace_back("primVertices", "GLO", "PVTX", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackITS", "ITS", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("tracksITSTPC", "GLO", "TPCITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackTPC", "TPC", "TRACKS", 0, Lifetime::Timeframe);

  outputs.emplace_back(OutputLabel{"O2bc"}, "AOD", "BC", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2collision"}, "AOD", "COLLISION", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2ft0"}, "AOD", "FT0", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mccollision"}, "AOD", "MCCOLLISION", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2track"}, "AOD", "TRACK", 0, Lifetime::Timeframe);

  // TODO:
  // add  FV0A, FV0C, FDD tables
  // outputs.emplace_back(OutputLabel{"O2fv0a"}, "AOD", "FV0A", 0, Lifetime::Timeframe);
  // outputs.emplace_back(OutputLabel{"O2fv0c"}, "AOD", "FV0C", 0, Lifetime::Timeframe);
  // outputs.emplace_back(OutputLabel{"O2fdd"}, "AOD", "FDD", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "aod-producer-workflow",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<AODProducerWorkflowDPL>()},
    Options{
      ConfigParamSpec{"fill-tracks-its", VariantType::Int, 1, {"Fill ITS tracks into tracks table"}},
      ConfigParamSpec{"fill-tracks-tpc", VariantType::Int, 1, {"Fill TPC tracks into tracks table"}},
      ConfigParamSpec{"fill-tracks-its-tpc", VariantType::Int, 1, {"Fill ITS-TPC tracks into tracks table"}}}};
}

} // namespace o2::aodproducer
