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

/// @file   CATrackerSpec.cxx

#include "ITSCAWorkflow/CATrackerSpec.h"

#include <stdexcept>
#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <numeric>
#include <ranges>
#include <utility>
#include <vector>

#include <gsl/span>

#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/DPLAlpideParam.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DetectorsBase/GeometryManager.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Logger.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTTracking/Tracker.h"
#include "ITSMFTTracking/GenericTrackOutputAdapter.h"
#include "ITSMFTTracking/IOUtils.h"
#include "ITSMFTTracking/SurfaceTiming.h"
#include "ITSMFTTracking/ITSMFTDetectorDefinitions.h"
#include "ITSMFTTracking/TrackingConfigParam.h"
#include "ITSMFTTracking/BoundedAllocator.h"
#include "CommonConstants/LHCConstants.h"
#include "DetectorsBase/Propagator.h"
#include <oneapi/tbb/task_arena.h>
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "SimulationDataFormat/O2DatabasePDG.h"
#include "Steer/MCKinematicsReader.h"
#include "ITSCAWorkflow/TruthSeeding.h"

using namespace o2::framework;

namespace o2::its::ca
{

namespace
{
using namespace o2::itsmft::tracking;

template <int NLayers>
constexpr std::array<LayerId, NLayers> detectorLocalToLayoutLayers()
{
  std::array<LayerId, NLayers> order{};
  for (int i = 0; i < NLayers; ++i) {
    order[i] = LayerId{static_cast<uint16_t>(i)};
  }
  return order;
}

inline constexpr auto kLayerToLayout = detectorLocalToLayoutLayers<ITSNLayers>();

bool completePublication(PublicationAdapter& publication,
                         const TimeFrame& frame,
                         const Tracker& tracker,
                         const TrackingResult& result)
{
  const auto configurations = tracker.getIterationConfigurations();
  std::size_t firstTrack = 0;
  for (std::size_t iteration = 0; iteration < configurations.size(); ++iteration) {
    if (iteration >= result.acceptedTrackCounts.size() ||
        result.acceptedTrackCounts[iteration] > frame.getGenericTracks().size() - firstTrack) {
      return false;
    }
    std::vector<uint32_t> trackIndices(result.acceptedTrackCounts[iteration]);
    std::iota(trackIndices.begin(), trackIndices.end(), static_cast<uint32_t>(firstTrack));
    if (!publication.completeAccepted(trackIndices, configurations[iteration].parameters, frame, iteration + 1 == configurations.size())) {
      return false;
    }
    firstTrack += result.acceptedTrackCounts[iteration];
  }
  return firstTrack == frame.getGenericTracks().size();
}

} // namespace

CATrackerDPL::CATrackerDPL(std::shared_ptr<o2::base::GRPGeomRequest> gr, WorkflowOptions options)
  : mGGCCDBRequest(std::move(gr)), mUseMC(options.useMC), mOptions(std::move(options))
{
  mClusterDecoder = std::make_unique<o2::itsmft::tracking::ITSGeometryClusterDecoder>();
  mPublication.adoptITSSharedClusterCompatibility(&mCompatibility);
}

void CATrackerDPL::addTruthSeedingVertices(const o2::InteractionRecord& origin, gsl::span<const o2::itsmft::ROFRecord> rofs)
{
  if (rofs.empty()) {
    return;
  }
  LOGP(info, "ITS CA using truth seeds as vertices");
  const auto& clock = mSession.frame.getROFViews().overlap.getLayer(0);
  const o2::itsmft::tracking::ROFTimingConfig timing{clock.mROFLength, clock.mROFDelay, clock.mROFBias, clock.mROFAddTimeErr};
  const auto first = o2::itsmft::tracking::computeROFIntervalBC(rofs.front().getBCData(), origin, timing, 0);
  const auto last = o2::itsmft::tracking::computeROFIntervalBC(rofs.back().getBCData(), origin, timing, rofs.size() - 1);
  const auto firstWindow = o2::itsmft::tracking::widen(first.interval, timing.rofAddTimeErr);
  const auto lastWindow = o2::itsmft::tracking::widen(last.interval, timing.rofAddTimeErr);
  if (!first.ok() || !last.ok() || !firstWindow.ok() || !lastWindow.ok()) {
    throw std::runtime_error("ITS CA truth seeding received invalid ROF timing");
  }
  const o2::itsmft::tracking::ROFIntervalBC window{std::max(int64_t{0}, firstWindow.interval.begin), lastWindow.interval.end, 0, 0};
  const std::unique_ptr<o2::steer::DigitizationContext> dc{o2::steer::DigitizationContext::loadFromFile(mOptions.truthContext.c_str())};
  if (!dc) {
    throw std::runtime_error("ITS CA truth seeding could not load " + mOptions.truthContext);
  }
  const auto& irs = dc->getEventRecords();
  o2::steer::MCKinematicsReader mcReader(dc.get());
  constexpr int iSrc = 0;
  const auto eveId2colId = dc->getCollisionIndicesForSource(iSrc);
  std::vector<std::pair<o2::its::TimeEstBC, int>> selected;
  for (int iEve = 0; iEve < mcReader.getNEvents(iSrc); ++iEve) {
    const auto collision = eveId2colId.find(iEve);
    if (collision == eveId2colId.end()) {
      continue;
    }
    const auto timestamp = truthSeedingTime(irs.at(collision->second), origin, window, clock.mROFLength / 2);
    if (timestamp) {
      selected.emplace_back(*timestamp, iEve);
    }
  }
  // The ROF vertex lookup performs a binary search by lower timestamp.
  std::sort(selected.begin(), selected.end(), [](const auto& a, const auto& b) {
    return std::pair{a.first.lower(), a.second} < std::pair{b.first.lower(), b.second};
  });
  for (const auto& [timestamp, iEve] : selected) {
    const auto& event = mcReader.getMCEventHeader(iSrc, iEve);
    o2::itsmft::tracking::Vertex vertex;
    vertex.getTimeStamp() = timestamp;
    vertex.setNContributors(std::max(1L, std::ranges::count_if(mcReader.getTracks(iSrc, iEve), [](const auto& track) {
                                       if (!track.isPrimary() || track.GetPt() < 0.05 || std::abs(track.GetEta()) > 1.1) {
                                         return false;
                                       }
                                       const auto* particle = o2::O2DatabasePDG::Instance()->GetParticle(track.GetPdgCode());
                                       return particle && particle->Charge() != 0;
                                     })));
    vertex.setXYZ(static_cast<float>(event.GetX()), static_cast<float>(event.GetY()), static_cast<float>(event.GetZ()));
    vertex.setChi2(1.f);
    constexpr float covariance = 25.e-4f;
    vertex.setSigmaX(covariance);
    vertex.setSigmaY(covariance);
    vertex.setSigmaZ(covariance);
    mSession.frame.addPrimaryVertex(vertex);
    const o2::MCCompLabel label{o2::MCCompLabel::maxTrackID(), iEve, iSrc, false};
    mSession.frame.addPrimaryVertexLabel(o2::itsmft::tracking::VertexLabel{label, 1.f});
    mcReader.releaseTracksForSourceAndEvent(iSrc, iEve);
  }
  LOGP(info, "ITS CA imposed {} pv collisions from MC truth", mSession.frame.getPrimaryVertices().size());
}

void CATrackerDPL::configureROFViews(gsl::span<const o2::itsmft::ROFRecord> rofs)
{
  const auto& detector = mTracker->getDetectorConfiguration();
  const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
  const int nOrbitsPerTF = o2::base::GRPGeomHelper::getNHBFPerTF();
  const auto timings = mSession.layerTimings(alpParams, nOrbitsPerTF, detector.addTimeError);
  mSession.configureTiming(timings, [](int) { return true; });
  (void)rofs;
}

void CATrackerDPL::initialiseTracking()
{
  const auto mode = mOptions.mode;
  auto plan = o2::itsmft::TrackingMode::getTrackingPlan(o2::detectors::DetID::ITS, mode);
  for (auto& pass : plan.iterations) {
    pass.UseDiamond = mOptions.vertexSource == VertexSource::Diamond;
  }
  LOGP(info, "ITS CA tracker initialized in {} mode with {} iteration(s)",
       o2::itsmft::TrackingMode::toString(mode), plan.iterations.size());
  if (plan.iterations.empty()) {
    return;
  }

  mTrackerTraits = std::make_unique<o2::itsmft::tracking::TrackerTraits>();
  std::shared_ptr<tbb::task_arena> taskArena;
  const auto& commonParams = o2::itsmft::ITSCommonCATrackerParam::Instance();
  mTrackerTraits->setNThreads(mOptions.nThreads, taskArena);

  const auto maxMemory = plan.execution.MaxMemory;
  o2::itsmft::tracking::TrackerInitialization configuration{
    .catalog = {o2::itsmft::tracking::kITSStaticSurfaceCatalog.data(),
                static_cast<uint32_t>(o2::itsmft::tracking::kITSStaticSurfaceCatalog.size())},
    .layout = o2::itsmft::tracking::makeDetectorLayout(o2::itsmft::tracking::LayerMask{commonParams.holeLayerMask}),
    .plan = std::move(plan),
    .memoryPool = std::make_shared<o2::itsmft::tracking::BoundedMemoryResource>(maxMemory)};

  mTracker = std::make_unique<o2::itsmft::tracking::Tracker>();
  const auto result = mTracker->initialize(mSession.frame, configuration);
  if (!result.ok()) {
    LOGP(fatal, "ITS CA tracker failed to initialize static configuration (error={} iteration={} layout={})",
         static_cast<int>(result.error), result.failedIteration, static_cast<int>(result.layoutError));
  }
}

o2::itsmft::tracking::TrackingOutcome CATrackerDPL::processTimeFrame(
  gsl::span<const o2::itsmft::ROFRecord> rofs,
  gsl::span<const o2::itsmft::CompClusterExt> clusters,
  gsl::span<const unsigned char> patterns,
  const o2::dataformats::MCTruthContainer<MCCompLabel>* labels)
{
  if (!isActive()) {
    LOGP(info, "ITS CA tracking mode is off, skipping TimeFrame processing");
    return o2::itsmft::tracking::TrackingOutcome::Success;
  }
  mSession.frame.setBz(o2::base::Propagator::Instance()->getNominalBz());
  o2::itsmft::tracking::ClusterSourceInput source;
  source.id = o2::itsmft::tracking::ClusterSourceId{0};
  source.detector = o2::detectors::DetID::ITS;
  source.clusters = clusters;
  source.patterns = patterns;
  source.rofs = rofs;
  source.dictionary = mDictionary;
  source.labels = labels;
  source.layerToSurface = kLayerToLayout;
  source.decoder = mClusterDecoder.get();
  return mSession.process(*mTracker, *mTrackerTraits, source, [&](const o2::InteractionRecord& origin) {
      if (mOptions.vertexSource == VertexSource::Truth) {
        addTruthSeedingVertices(origin, rofs);
        mSession.vertices.update(mSession.frame.getPrimaryVertices().data(), mSession.frame.getPrimaryVertices().size());
      } }, [&](const o2::itsmft::tracking::TrackingResult& result) {
      if (!completePublication(mPublication, mSession.frame, *mTracker, result)) {
        throw std::runtime_error{"failed to seal ITS tracking compatibility"};
      } });
}

void CATrackerDPL::init(InitContext&)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
}

void CATrackerDPL::run(ProcessingContext& pc)
{
  auto publicationCleanup = mPublication.cleanupOnExit();
  updateTimeDependentParams(pc);

  auto rofsinput = pc.inputs().get<const std::vector<o2::itsmft::ROFRecord>>("ROframes");

  if (decideCATrackerPublicationAction(isActive(), o2::itsmft::tracking::TrackingOutcome::Success) == CATrackerPublicationAction::PublishInactiveEmpty) {
    pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{"ITS", "ITSTrackROF", 0},
                                                          rofsinput.begin(), rofsinput.end());
    pc.outputs().make<std::vector<o2::its::TrackITS>>(Output{"ITS", "TRACKS", 0});
    pc.outputs().make<std::vector<int>>(Output{"ITS", "TRACKCLSID", 0});
    return;
  }

  auto compClusters = pc.inputs().get<const std::vector<o2::itsmft::CompClusterExt>>("compClusters");
  gsl::span<const unsigned char> patterns = pc.inputs().get<gsl::span<unsigned char>>("patterns");

  const dataformats::MCTruthContainer<MCCompLabel>* labels = nullptr;
  if (mUseMC && pc.inputs().getPos("labels") >= 0) {
    labels = pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("labels").release();
  }

  LOGP(info, "ITS CA input pulled {} compressed clusters in {} RO frames ({} pattern bytes)",
       compClusters.size(), rofsinput.size(), patterns.size());

  auto cleanup = mSession.cleanupOnExit();
  configureROFViews(gsl::span<const o2::itsmft::ROFRecord>(rofsinput.data(), rofsinput.size()));
  const auto trackingResult = processTimeFrame(gsl::span<const o2::itsmft::ROFRecord>(rofsinput.data(), rofsinput.size()),
                                               gsl::span<const o2::itsmft::CompClusterExt>(compClusters.data(), compClusters.size()),
                                               patterns, labels);

  if (decideCATrackerPublicationAction(isActive(), trackingResult) == CATrackerPublicationAction::SkipDroppedTimeFrame) {
    LOGP(error, "ITS CA tracking dropped this TimeFrame ({} ROFs, {} clusters); publishing nothing and continuing with the next TimeFrame",
         rofsinput.size(), compClusters.size());
    cleanup.frameAlreadyReset();
    return;
  }

  {
    mSession.publicationClock.emplace(mSession.overlap.getView().getClockLayer());
    const o2::itsmft::tracking::GenericTrackPublicationContext context{
      o2::detectors::DetID::ITS, o2::itsmft::tracking::ClusterSourceId{0},
      gsl::span<const o2::itsmft::ROFRecord>{rofsinput.data(), rofsinput.size()}, *mSession.publicationClock,
      kLayerToLayout,
      &mSession.externalIndices, &mSession.clusterSizes};
    o2::itsmft::tracking::GenericTrackOutputAdapterError error = o2::itsmft::tracking::GenericTrackOutputAdapterError::None;
    const auto staged = o2::itsmft::tracking::stageITSGenericTrackOutput(mSession.frame, context, mCompatibility, mUseMC, error);
    if (!staged) {
      throw std::runtime_error{"ITS GenericTrack output staging failed"};
    }

    o2::itsmft::tracking::copyTrackingOutputColumns(pc.outputs(), Output{"ITS", "ITSTrackROF", 0},
                                                    Output{"ITS", "TRACKS", 0}, Output{"ITS", "TRACKCLSID", 0}, *staged);
    LOGP(info, "ITS CA pushed {} tracks in {} ROFs", staged->tracks.size(), staged->trackROFs.size());
    if (mUseMC) {
      pc.outputs().snapshot(Output{"ITS", "TRACKSMCTR", 0}, staged->labels);
      LOGP(info, "ITS CA pushed {} track MC labels", staged->labels.size());
    }
  }
}

void CATrackerDPL::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  pc.inputs().get<o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>*>("itsalppar");
  if (!mTrackingInitialised) {
    mTrackingInitialised = true;
    initialiseTracking();
  }
  static bool initOnceDone = false;
  if (!initOnceDone) {
    initOnceDone = true;
    if (pc.inputs().getPos("itsTGeo") >= 0) {
      pc.inputs().get<o2::its::GeometryTGeo*>("itsTGeo");
    }
    pc.inputs().get<o2::itsmft::TopologyDictionary*>("itscldict");
    o2::its::GeometryTGeo::Instance()->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L,
                                                                                o2::math_utils::TransformType::T2GRot,
                                                                                o2::math_utils::TransformType::T2G));
  }
}

void CATrackerDPL::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    LOG(info) << "ITS CA input cluster dictionary updated";
    mDictionary = static_cast<const o2::itsmft::TopologyDictionary*>(obj);
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "ALPIDEPARAM", 0)) {
    LOG(info) << "ITS CA input Alpide param updated";
    o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance().printKeyValues();
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "GEOMTGEO", 0)) {
    LOG(info) << "ITS CA input GeometryTGeo loaded from CCDB";
    o2::its::GeometryTGeo::adopt(static_cast<o2::its::GeometryTGeo*>(obj));
    o2::its::GeometryTGeo::Instance()->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L,
                                                                                o2::math_utils::TransformType::T2GRot,
                                                                                o2::math_utils::TransformType::T2G));
    // The catalog has static process lifetime; geometry adoption remains
    // necessary for raw cluster decoding.
    return;
  }
}

DataProcessorSpec getCATrackerSpec(const WorkflowOptions& options)
{
  const bool useMC = options.useMC;
  const bool useGeom = options.useFullGeometry;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("compClusters", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("patterns", "ITS", "PATTERNS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);
  inputs.emplace_back("itscldict", "ITS", "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec("ITS/Calib/ClusterDictionary"));
  inputs.emplace_back("itsalppar", "ITS", "ALPIDEPARAM", 0, Lifetime::Condition, ccdbParamSpec("ITS/Config/AlpideParam"));

  if (useMC) {
    inputs.emplace_back("labels", "ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
  }

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,
                                                              true,
                                                              false,
                                                              true,
                                                              true,
                                                              useGeom ? o2::base::GRPGeomRequest::Aligned : o2::base::GRPGeomRequest::None,
                                                              inputs,
                                                              true);
  if (!useGeom) {
    ggRequest->addInput({"itsTGeo", "ITS", "GEOMTGEO", 0, Lifetime::Condition, framework::ccdbParamSpec("ITS/Config/Geometry")}, inputs);
  }

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ITS", "TRACKS", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "TRACKCLSID", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "ITSTrackROF", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("ITS", "TRACKSMCTR", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "its-ca-tracker",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<CATrackerDPL>(ggRequest, options)},
    Options{}};
}

} // namespace o2::its::ca
