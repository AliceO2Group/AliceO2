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

#include "MFTWorkflow/CATrackerSpec.h"

#include <array>
#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include <stdexcept>

#include <gsl/span>

#include "CommonDataFormat/IRFrame.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/DPLAlpideParam.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DetectorsBase/GeometryManager.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Logger.h"
#include "ITSMFTTracking/Tracker.h"
#include "ITSMFTTracking/GenericTrackOutputAdapter.h"
#include "ITSMFTTracking/IOUtils.h"
#include "ITSMFTTracking/SurfaceTiming.h"
#include "ITSMFTTracking/ITSMFTDetectorDefinitions.h"
#include "ITSMFTTracking/TrackingConfigParam.h"
#include "DetectorsBase/Propagator.h"
#include <oneapi/tbb/task_arena.h>
#include "CommonConstants/LHCConstants.h"
#include "MFTBase/GeometryTGeo.h"
#include "MFTTracking/Constants.h"
#include "MFTTracking/MFTTrackingParam.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2::mft
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

inline constexpr auto kLayerToLayout = detectorLocalToLayoutLayers<MFTNLayers>();

bool rofOverlapsIRFrames(const o2::itsmft::ROFRecord& rof, int rofLengthInBC,
                         gsl::span<const o2::dataformats::IRFrame> irFrames)
{
  o2::InteractionRecord start{rof.getBCData()};
  const o2::InteractionRecord end = start + rofLengthInBC - 1;
  const o2::dataformats::IRFrame reference{start, end};
  for (const auto& ir : irFrames) {
    if (ir.info > 0 && reference.getOverlap(ir).isValid()) {
      return true;
    }
  }
  return false;
}

} // namespace

CATrackerDPL::CATrackerDPL(std::shared_ptr<o2::base::GRPGeomRequest> gr, ca::TrackerOptions options)
  : mGGCCDBRequest(std::move(gr)), mUseMC(options.useMC), mOptions(options)
{
  mClusterDecoder = std::make_unique<o2::itsmft::tracking::MFTGeometryClusterDecoder>();
}

void CATrackerDPL::configureROFViews(gsl::span<const o2::itsmft::ROFRecord> rofs,
                                     gsl::span<const o2::dataformats::IRFrame> irFrames)
{
  const auto& detector = mTracker->getDetectorConfiguration();
  const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::MFT>::Instance();
  const bool continuous = o2::base::GRPGeomHelper::instance().getGRPECS()->isDetContinuousReadOut(o2::detectors::DetID::MFT);
  mMFTROFrameLengthInBC = continuous ? alpParams.roFrameLengthInBC : std::max(1, static_cast<int>(alpParams.roFrameLengthTrig / (o2::constants::lhc::LHCBunchSpacingNS * 1e3)));
  const int nOrbitsPerTF = o2::base::GRPGeomHelper::getNHBFPerTF();
  const auto timings = mSession.layerTimings(alpParams, nOrbitsPerTF, detector.addTimeError);
  const auto& trackingParam = o2::mft::MFTTrackingParam::Instance();
  const bool useIrFilter = mOptions.filterIRFrames && !irFrames.empty();
  mSession.configureTiming(timings, [&](int rof) {
    return rof >= static_cast<int>(rofs.size()) ||
           ((!useIrFilter || rofOverlapsIRFrames(rofs[rof], mMFTROFrameLengthInBC, irFrames)) &&
            (!trackingParam.isMultCutRequested() || trackingParam.isPassingMultCut(rofs[rof].getNEntries())));
  });
}

void CATrackerDPL::initialiseTracking()
{
  const auto mode = mOptions.mode;
  const auto& trackerParams = o2::itsmft::tracking::TrackerParamRef<o2::detectors::DetID::MFT>::get();
  auto plan = o2::itsmft::TrackingMode::getTrackingPlan(o2::detectors::DetID::MFT, mode);
  LOGP(info, "MFT CA tracker initialized in {} mode with {} iteration(s)",
       o2::itsmft::TrackingMode::toString(mode), plan.iterations.size());
  if (plan.iterations.empty()) {
    return;
  }

  mTrackerTraits = std::make_unique<o2::itsmft::tracking::TrackerTraits>();
  std::shared_ptr<tbb::task_arena> taskArena;
  mTrackerTraits->setNThreads(mOptions.nThreads, taskArena);

  const auto maxMemory = plan.execution.MaxMemory;
  o2::itsmft::tracking::TrackerInitialization configuration{
    .catalog = {o2::itsmft::tracking::kMFTStaticSurfaceCatalog.data(),
                static_cast<uint32_t>(o2::itsmft::tracking::kMFTStaticSurfaceCatalog.size())},
    .layout = o2::itsmft::tracking::makeDetectorLayout(o2::itsmft::tracking::LayerMask{trackerParams.holeLayerMask}),
    .plan = std::move(plan),
    .memoryPool = std::make_shared<o2::itsmft::tracking::BoundedMemoryResource>(maxMemory)};

  mTracker = std::make_unique<o2::itsmft::tracking::Tracker>();
  const auto result = mTracker->initialize(mSession.frame, configuration);
  if (!result.ok()) {
    LOGP(fatal, "MFT CA tracker failed to initialize static configuration (error={} iteration={} layout={})",
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
    LOGP(info, "MFT CA tracking mode is off, skipping TimeFrame processing");
    return o2::itsmft::tracking::TrackingOutcome::Success;
  }
  mSession.frame.setBz(o2::base::Propagator::Instance()->getNominalBz());
  o2::itsmft::tracking::ClusterSourceInput source;
  source.id = o2::itsmft::tracking::ClusterSourceId{0};
  source.detector = o2::detectors::DetID::MFT;
  source.clusters = clusters;
  source.patterns = patterns;
  source.rofs = rofs;
  source.dictionary = mDictionary;
  source.labels = labels;
  source.layerToSurface = kLayerToLayout;
  source.decoder = mClusterDecoder.get();
  return mSession.process(*mTracker, *mTrackerTraits, source, [](const o2::InteractionRecord&) {}, [](const o2::itsmft::tracking::TrackingResult&) {});
}

void CATrackerDPL::init(InitContext&)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
}

void CATrackerDPL::run(ProcessingContext& pc)
{
  updateTimeDependentParams(pc);

  auto rofsinput = pc.inputs().get<const std::vector<o2::itsmft::ROFRecord>>("ROframes");

  if (decideCATrackerPublicationAction(isActive(), o2::itsmft::tracking::TrackingOutcome::Success) == CATrackerPublicationAction::PublishInactiveEmpty) {
    // Existing production behavior, preserved exactly: publish the input
    // ROFs verbatim (their firstEntry/nEntries are not rewritten here) plus
    // empty track/cluster-index/seed-pattern outputs, when the tracker is
    // not configured to run.
    pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{"MFT", "MFTTrackROF", 0},
                                                          rofsinput.begin(), rofsinput.end());
    pc.outputs().make<std::vector<o2::mft::TrackMFT>>(Output{"MFT", "TRACKS", 0});
    pc.outputs().make<std::vector<int>>(Output{"MFT", "TRACKCLSID", 0});
    pc.outputs().make<std::vector<uint16_t>>(Output{"MFT", "TRACKSEEDPAT", 0});
    return;
  }

  auto compClusters = pc.inputs().get<const std::vector<o2::itsmft::CompClusterExt>>("compClusters");
  gsl::span<const unsigned char> patterns = pc.inputs().get<gsl::span<unsigned char>>("patterns");

  const dataformats::MCTruthContainer<MCCompLabel>* labels = nullptr;
  if (mUseMC && pc.inputs().getPos("labels") >= 0) {
    labels = pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("labels").release();
  }

  gsl::span<const o2::dataformats::IRFrame> irFrames;
  if (pc.inputs().getPos("IRFramesITS") >= 0) {
    irFrames = pc.inputs().get<gsl::span<o2::dataformats::IRFrame>>("IRFramesITS");
  }

  LOGP(info, "MFT CA input pulled {} compressed clusters in {} RO frames ({} pattern bytes)",
       compClusters.size(), rofsinput.size(), patterns.size());

  auto cleanup = mSession.cleanupOnExit();
  configureROFViews(gsl::span<const o2::itsmft::ROFRecord>(rofsinput.data(), rofsinput.size()), irFrames);
  const auto trackingResult = processTimeFrame(gsl::span<const o2::itsmft::ROFRecord>(rofsinput.data(), rofsinput.size()),
                                               gsl::span<const o2::itsmft::CompClusterExt>(compClusters.data(), compClusters.size()),
                                               patterns, labels);

  if (decideCATrackerPublicationAction(isActive(), trackingResult) == CATrackerPublicationAction::SkipDroppedTimeFrame) {
    LOGP(error, "MFT CA tracking dropped this TimeFrame ({} ROFs, {} clusters); publishing nothing and continuing with the next TimeFrame",
         rofsinput.size(), compClusters.size());
    cleanup.frameAlreadyReset();
    return;
  }

  {
    mSession.publicationClock.emplace(mSession.overlap.getView().getClockLayer());
    const o2::itsmft::tracking::GenericTrackPublicationContext context{
      o2::detectors::DetID::MFT, o2::itsmft::tracking::ClusterSourceId{0},
      gsl::span<const o2::itsmft::ROFRecord>{rofsinput.data(), rofsinput.size()}, *mSession.publicationClock,
      kLayerToLayout,
      &mSession.externalIndices, &mSession.clusterSizes};
    o2::itsmft::tracking::GenericTrackOutputAdapterError error = o2::itsmft::tracking::GenericTrackOutputAdapterError::None;
    const auto staged = o2::itsmft::tracking::stageMFTGenericTrackOutput(mSession.frame, context, mUseMC, error);
    if (!staged) {
      throw std::runtime_error{"MFT GenericTrack output staging failed"};
    }

    o2::itsmft::tracking::copyTrackingOutputColumns(pc.outputs(), Output{"MFT", "MFTTrackROF", 0},
                                                    Output{"MFT", "TRACKS", 0}, Output{"MFT", "TRACKCLSID", 0}, *staged);
    auto& allSeedPatterns = pc.outputs().make<std::vector<uint16_t>>(Output{"MFT", "TRACKSEEDPAT", 0});
    allSeedPatterns.assign(staged->seedPatterns.begin(), staged->seedPatterns.end());
    LOGP(info, "MFT CA pushed {} tracks in {} ROFs", staged->tracks.size(), staged->trackROFs.size());
    if (mUseMC) {
      pc.outputs().snapshot(Output{"MFT", "TRACKSMCTR", 0}, staged->labels);
      LOGP(info, "MFT CA pushed {} track MC labels", staged->labels.size());
    }
  }
}

void CATrackerDPL::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  if (!mTrackingInitialised) {
    mTrackingInitialised = true;
    initialiseTracking();
  }
  static bool initOnceDone = false;
  if (!initOnceDone) {
    initOnceDone = true;
    if (pc.inputs().getPos("mftTGeo") >= 0) {
      pc.inputs().get<o2::mft::GeometryTGeo*>("mftTGeo");
    }
    pc.inputs().get<o2::itsmft::TopologyDictionary*>("cldict");
    o2::mft::GeometryTGeo::Instance()->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L,
                                                                                o2::math_utils::TransformType::T2GRot,
                                                                                o2::math_utils::TransformType::T2G,
                                                                                o2::math_utils::TransformType::L2G));
  }
}

void CATrackerDPL::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("MFT", "CLUSDICT", 0)) {
    LOG(info) << "MFT CA input cluster dictionary updated";
    mDictionary = static_cast<const o2::itsmft::TopologyDictionary*>(obj);
    return;
  }
  if (matcher == ConcreteDataMatcher("MFT", "GEOMTGEO", 0)) {
    LOG(info) << "MFT CA input GeometryTGeo loaded from CCDB";
    o2::mft::GeometryTGeo::adopt(static_cast<o2::mft::GeometryTGeo*>(obj));
    o2::mft::GeometryTGeo::Instance()->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L,
                                                                                o2::math_utils::TransformType::T2GRot,
                                                                                o2::math_utils::TransformType::T2G,
                                                                                o2::math_utils::TransformType::L2G));
    // The catalog has static process lifetime; geometry adoption remains
    // necessary for raw cluster decoding.
    return;
  }
}

DataProcessorSpec getCATrackerSpec(const ca::TrackerOptions& options)
{
  const bool useMC = options.useMC;
  const bool useGeom = options.geometry == ca::GeometrySource::Full;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("compClusters", "MFT", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("patterns", "MFT", "PATTERNS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "MFT", "CLUSTERSROF", 0, Lifetime::Timeframe);
  inputs.emplace_back("cldict", "MFT", "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec("MFT/Calib/ClusterDictionary"));

  if (useMC) {
    inputs.emplace_back("labels", "MFT", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
  }

  if (options.irFrames != ca::IRFrameSource::None) {
    inputs.emplace_back("IRFramesITS", "ITS", "IRFRAMES", 0, Lifetime::Timeframe);
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
    ggRequest->addInput({"mftTGeo", "MFT", "GEOMTGEO", 0, Lifetime::Condition, framework::ccdbParamSpec("MFT/Config/Geometry")}, inputs);
  }

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("MFT", "TRACKS", 0, Lifetime::Timeframe);
  outputs.emplace_back("MFT", "MFTTrackROF", 0, Lifetime::Timeframe);
  outputs.emplace_back("MFT", "TRACKCLSID", 0, Lifetime::Timeframe);
  outputs.emplace_back("MFT", "TRACKSEEDPAT", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("MFT", "TRACKSMCTR", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "mft-ca-tracker",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<CATrackerDPL>(ggRequest, options)},
    Options{}};
}

} // namespace o2::mft
