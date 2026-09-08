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
///
/// \file Tracker.cxx
/// \brief
///

#include "ITSMFTTracking/Tracker.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <queue>
#include <ranges>
#include <stdexcept>
#include <utility>

#include "Framework/Logger.h"
#include "GPUCommonMath.h"
#include "ITSMFTTracking/BoundedAllocator.h"
#include "ITSMFTTracking/IndexTableConfiguration.h"
#include "ITSMFTTracking/MaterialPhysics.h"
#include "ITSMFTTracking/detail/TrackerTraversalPreparation.h"

namespace o2::itsmft::tracking
{

namespace
{
constexpr std::size_t kindIndex(SurfaceKind kind) noexcept
{
  return kind == SurfaceKind::Cylinder ? 0u : 1u;
}

TrackingKernelParameters bindTrackingKernelParameters(const IterationParameters& params) noexcept
{
  TrackingKernelParameters out;
  out.trackletMinPt = params.TrackletMinPt;
  out.nSigmaCut = params.NSigmaCut;
  out.maxChi2ClusterAttachment = params.MaxChi2ClusterAttachment;
  out.maxChi2NDF = params.MaxChi2NDF;
  out.pvResolution = params.PVres;
  return out;
}

} // namespace

namespace
{
void validateSparsePlan(const IterationConfiguration& configuration, int iteration, const TraversalTopologyView& layout)
{
  const auto fail = [iteration]() { throw TraversalException{iteration, TraversalFailureReason::SparseTopologyMismatch}; };
  const auto& topology = layout;
  if (layout.catalog.surfaces == nullptr || layout.catalog.nSurfaces == 0 ||
      (topology.nEdges != 0 && (topology.edges == nullptr || topology.pathsByFirstEdgeOffsets == nullptr)) ||
      (topology.nPaths != 0 && (topology.paths == nullptr || topology.pathsByFirstEdge == nullptr))) {
    fail();
  }

  const auto edges = configuration.edgeIds();
  const auto cells = configuration.cellIds();
  if (edges.empty() || edges.size() > topology.nEdges || cells.size() > topology.nPaths) {
    fail();
  }
  for (const auto id : edges) {
    if (!id.isValid() || id.value() >= topology.nEdges || !configuration.getEdgeSlot(id)) {
      fail();
    }
    const auto& edge = topology.getEdge(id);
    if (!configuration.hasLayer(edge.from) || !configuration.hasLayer(edge.to)) {
      fail();
    }
  }
  for (const auto id : cells) {
    if (!id.isValid() || id.value() >= topology.nPaths || !configuration.getCellSlot(id)) {
      fail();
    }
    const auto& path = topology.getPath(id);
    const auto& firstEdge = topology.getEdge(path.first);
    const auto& secondEdge = topology.getEdge(path.second);
    if (!configuration.getEdgeSlot(path.first) || !configuration.getEdgeSlot(path.second) ||
        !configuration.hasLayer(firstEdge.from) || !configuration.hasLayer(firstEdge.to) ||
        !configuration.hasLayer(secondEdge.to) ||
        !configuration.topology.activeLayers.has(firstEdge.from.value()) ||
        !configuration.topology.activeLayers.has(firstEdge.to.value()) ||
        !configuration.topology.activeLayers.has(secondEdge.to.value())) {
      fail();
    }
  }
  for (const auto id : configuration.topology.scheduledPaths) {
    if (!configuration.getCellSlot(id)) {
      fail();
    }
  }
  for (const auto id : configuration.topology.roadStartPaths) {
    if (!configuration.getCellSlot(id)) {
      fail();
    }
  }
}

DetectorConfiguration prepareDetectorConfiguration(const DetectorLayout& layout, const DetectorParameters& parameters)
{
  DetectorConfiguration configuration;
  const auto catalog = layout.getSurfaceCatalog();
  const auto surfaceCount = layout.size();
  if (surfaceCount == 0 || surfaceCount > MaxLayoutSurfaces ||
      parameters.LayerRadii.size() < surfaceCount ||
      parameters.AddTimeError.size() < surfaceCount ||
      parameters.SystError2Col.size() < surfaceCount ||
      parameters.SystError2Row.size() < surfaceCount ||
      parameters.LayerResolution.size() < surfaceCount) {
    throw TraversalException{-1, TraversalFailureReason::InvalidSurfaceParameters};
  }
  configuration.layerRadii.assign(parameters.LayerRadii.begin(), parameters.LayerRadii.begin() + surfaceCount);
  configuration.addTimeError.assign(parameters.AddTimeError.begin(), parameters.AddTimeError.begin() + surfaceCount);
  configuration.layerResolution.assign(parameters.LayerResolution.begin(), parameters.LayerResolution.begin() + surfaceCount);
  configuration.systError2Row.assign(parameters.SystError2Row.begin(), parameters.SystError2Row.begin() + surfaceCount);
  configuration.systError2Col.assign(parameters.SystError2Col.begin(), parameters.SystError2Col.begin() + surfaceCount);
  configuration.positionResolutions.resize(surfaceCount);
  std::array<SurfaceChartRange, MaxLayoutSurfaces> chartRanges{};
  for (std::size_t position = 0; position < surfaceCount; ++position) {
    const auto surface = LayerId{static_cast<uint16_t>(position)};
    const auto& descriptor = catalog.getSurface(surface);
    chartRanges[position] = descriptor.chartRange;
    configuration.positionResolutions[position] = o2::gpu::CAMath::Sqrt(
      0.5f * (parameters.SystError2Col[position] + parameters.SystError2Row[position]) +
      parameters.LayerResolution[position] * parameters.LayerResolution[position]);
  }
  if (!configuration.indexTableConfigs.reset(catalog)) {
    throw TraversalException{-1, TraversalFailureReason::InvalidIndexTableConfiguration};
  }
  const gsl::span<const SurfaceChartRange> chartRangeView{chartRanges.data(), surfaceCount};
  for (const auto kind : {SurfaceKind::Cylinder, SurfaceKind::Disk}) {
    if (configuration.indexTableConfigs.hasKind(kind) &&
        bindIndexTableConfiguration(configuration.indexTableConfigs.forKind(kind), parameters,
                                    static_cast<int>(surfaceCount), kind, chartRangeView) != IndexTableConfigError::None) {
      throw TraversalException{-1, TraversalFailureReason::InvalidIndexTableConfiguration};
    }
  }
  return configuration;
}

void prepareIterationConfiguration(const DetectorLayout& layout, const DetectorConfiguration& detector,
                                   IterationConfiguration& configuration, int iteration)
{
  const auto topology = configuration.getTopologyView(layout.getSurfaceCatalog());
  const auto& parameters = configuration.parameters;
  const auto layerCount = configuration.topology.nLayers;
  if (layerCount == 0 || layerCount > MaxLayoutSurfaces ||
      parameters.NLayers != static_cast<int>(layerCount)) {
    throw TraversalException{iteration, TraversalFailureReason::LegacyMaterialMismatch};
  }

  for (uint16_t position = 0; position < layerCount; ++position) {
    const auto surface = LayerId{position};
    const auto& descriptor = topology.getSurface(surface);
    if (materialCorrectionModeSupport(descriptor.kind, parameters.CorrType) == MaterialCorrectionModeSupport::Unsupported) {
      throw TraversalException{iteration, TraversalFailureReason::UnsupportedMaterialCorrectionMode};
    }
  }

  if (!bindAttachHitConfig(topology.catalog, parameters).isValid(static_cast<int>(layerCount)) ||
      detector.layerRadii.size() < layerCount ||
      detector.positionResolutions.size() < layerCount ||
      detector.indexTableConfigs.size() < layerCount) {
    throw TraversalException{iteration, TraversalFailureReason::InvalidSurfaceParameters};
  }
  configuration.kernelParameters = bindTrackingKernelParameters(parameters);
  if (!configuration.kernelParameters.isValid()) {
    throw TraversalException{iteration, TraversalFailureReason::InvalidSurfaceParameters};
  }
  validateSparsePlan(configuration, iteration, topology);
}

void prepareTraversalEdgeTolerances(
  IterationContext& context,
  int iteration)
{
  auto& scratch = context.scratch;
  const auto& graph = context.topology;
  const auto& trkParam = context.configuration.parameters;
  const auto& topology = graph;

  const int layerCount = context.configuration.topology.nLayers;
  std::array<float, MaxLayoutSurfaces> msAngles{};
  for (int iLayer{0}; iLayer < layerCount; ++iLayer) {
    const auto surface = LayerId{static_cast<uint16_t>(iLayer)};
    if (topology.getSurface(surface).kind == SurfaceKind::Cylinder) {
      msAngles[iLayer] = cylinderLayerMultipleScatteringAngle(
        CylinderLayerScatteringInputs{topology.getSurface(surface).material.xOverX0}, trkParam.TrackletMinPt);
    } else {
      msAngles[iLayer] = diskLayerMultipleScatteringAngle(
        DiskLayerScatteringInputs{topology.getSurface(surface).material.xOverX0,
                                  context.detectorConfiguration.layerRadii[iLayer],
                                  topology.getSurface(surface).referenceCoordinate},
        trkParam.TrackletMinPt);
    }
  }

  auto& edgeMSAngles = scratch.getEdgeMSAngles();
  auto& edgePhiCuts = scratch.getEdgePhiCuts();
  const float oneOverR{0.001f * 0.3f * std::abs(context.bz) / trkParam.TrackletMinPt};
  for (const auto edgeId : context.configuration.edgeIds()) {
    const auto edgeSlot = context.configuration.getEdgeSlot(edgeId);
    if (!edgeSlot) {
      throw TraversalException{iteration, TraversalFailureReason::TraversalBindingMismatch};
    }
    const auto& edge = topology.getEdge(edgeId);
    if (!context.configuration.hasLayer(edge.from) || !context.configuration.hasLayer(edge.to)) {
      throw TraversalException{iteration, TraversalFailureReason::TraversalBindingMismatch};
    }
    const int fromLayer = edge.from.value();
    const int toLayer = edge.to.value();
    const float r1 = std::min(context.detectorConfiguration.layerRadii[fromLayer], context.detectorConfiguration.layerRadii[toLayer]);
    const float r2 = std::max(context.detectorConfiguration.layerRadii[fromLayer], context.detectorConfiguration.layerRadii[toLayer]);
    const float edgeOneOverR = clampEdgeCurvature(oneOverR, r2);
    const float res1 = o2::gpu::CAMath::Hypot(trkParam.PVres, context.detectorConfiguration.positionResolutions[fromLayer]);
    const float res2 = o2::gpu::CAMath::Hypot(trkParam.PVres, context.detectorConfiguration.positionResolutions[toLayer]);
    const auto prep = ::o2::itsmft::tracking::prepareEdgeScatteringAndBending(
      gsl::span<const float>(msAngles.data(), static_cast<std::size_t>(layerCount)), fromLayer, toLayer, r1, r2, edgeOneOverR, res1, res2);
    edgeMSAngles[*edgeSlot] = prep.msAngle;
    edgePhiCuts[*edgeSlot] = prep.phiCut;
  }
}
} // namespace

void Tracker::initializeIteration(IterationContext& context) const
{
  const int iteration = context.iteration;
  if (iteration < 0 || static_cast<size_t>(iteration) >= mIterations.size()) {
    throw TraversalException{iteration, TraversalFailureReason::IterationOutOfRange};
  }
  const auto& configuration = context.configuration;
  const auto& parameters = configuration.parameters;
  auto& frame = context.frame;
  auto& scratch = context.scratch;
  const auto layerCount = configuration.topology.nLayers;

  if (parameters.PassFlags[IterationStep::FirstPass]) {
    frame.prepareIndexTables(context.detectorConfiguration.indexTableConfigs);
  } else {
    for (std::size_t position = 0; position < layerCount; ++position) {
      if (!indexTableConfigurationsMatch(context.detectorConfiguration.indexTableConfigs[position],
                                         frame.getIndexTableUtils(static_cast<int>(position)),
                                         static_cast<int>(layerCount))) {
        throw TraversalException{iteration, TraversalFailureReason::IndexTableConfigurationMismatch};
      }
    }
  }
  if (parameters.PassFlags[IterationStep::RebuildClusterLUT]) {
    frame.prepareClusters(static_cast<int>(layerCount));
  }

  const auto edgeIds = context.configuration.edgeIds();
  const auto cellIds = context.configuration.cellIds();
  std::array<std::size_t, MaxLayoutEdges> trackletLookupSizes;
  for (const auto edgeId : edgeIds) {
    const auto from = context.topology.getEdge(edgeId).from;
    if (!configuration.hasLayer(from) || from.value() >= context.layerGlobalMeasurements.size()) {
      throw TraversalException{iteration, TraversalFailureReason::TraversalBindingMismatch};
    }
    trackletLookupSizes[edgeId.value()] = context.layerGlobalMeasurements[from.value()].size();
  }
  scratch.beginIteration(edgeIds.size(), cellIds.size(), {trackletLookupSizes.data(), edgeIds.size()});

  // Sorted clusters are a locator cache. Validate every enabled ROF that can
  // participate in a configured edge, including LUT-reuse paths.
  // Keep spans local until validation and kind setup complete.
  std::array<bool, MaxLayoutSurfaces> candidateReachableLayers{};
  for (const auto edgeId : edgeIds) {
    const auto& edge = context.topology.getEdge(edgeId);
    if (!configuration.hasLayer(edge.from) || !configuration.hasLayer(edge.to)) {
      throw TraversalException{iteration, TraversalFailureReason::SparseTopologyMismatch};
    }
    candidateReachableLayers[edge.from.value()] = true;
    candidateReachableLayers[edge.to.value()] = true;
  }
  for (std::size_t layer = 0; layer < layerCount; ++layer) {
    if (!candidateReachableLayers[layer]) {
      continue;
    }
    const auto measurements = context.layerGlobalMeasurements[layer];
    const auto rofBoundaries = frame.getROFrameClusters(static_cast<int>(layer));
    const auto rofMask = frame.getROFViews(static_cast<int>(layer)).mask;
    // Orchestration-only users may omit the mask; without it no ROF is reachable.
    if (rofMask.mFlatMask == nullptr || rofMask.mLayerROFOffsets == nullptr) {
      continue;
    }
    for (int rof = 0; rof < frame.getNrof(static_cast<int>(layer)); ++rof) {
      const auto sorted = frame.getClustersOnLayer(rof, static_cast<int>(layer));
      if (sorted.empty()) {
        continue;
      }
      if (!frame.isROFEnabled(static_cast<int>(layer), rof)) {
        continue;
      }
      const int first = rofBoundaries[rof];
      const int last = rofBoundaries[rof + 1];
      if (first < 0 || last < first || last > static_cast<int>(measurements.size()) ||
          sorted.size() != static_cast<size_t>(last - first)) {
        throw TraversalException{iteration, TraversalFailureReason::NormalizedMeasurementMismatch};
      }
      std::vector<uint32_t> seen;
      seen.reserve(sorted.size());
      for (const auto& measurement : sorted) {
        if (!measurement.hasValidClusterId() ||
            frame.getSurfaceMeasurement(LayerId{static_cast<uint16_t>(layer)}, measurement.clusterId) == nullptr) {
          throw TraversalException{iteration, TraversalFailureReason::NormalizedMeasurementMismatch};
        }
        seen.push_back(measurement.clusterId);
      }
      std::sort(seen.begin(), seen.end());
      if (std::adjacent_find(seen.begin(), seen.end()) != seen.end()) {
        throw TraversalException{iteration, TraversalFailureReason::NormalizedMeasurementMismatch};
      }
    }
  }

  prepareTraversalEdgeTolerances(context, iteration);
}

gsl::span<const gsl::span<const GlobalMeasurement>> Tracker::prepareTimeFrame(
  TimeFrame& frame, std::array<gsl::span<const GlobalMeasurement>, MaxLayoutSurfaces>& measurements) const
{
  const auto layerCount = mIterations.front().topology.nLayers;
  for (uint16_t position = 0; position < layerCount; ++position) {
    const auto surface = LayerId{position};
    const auto globals = frame.getGlobalMeasurements(surface);
    if (globals.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw TraversalException{-1, TraversalFailureReason::NormalizedMeasurementMismatch};
    }
    for (const auto& global : globals) {
      if (!global.hasValidClusterId() || global.clusterId > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
          frame.getSurfaceMeasurement(surface, global.clusterId) == nullptr) {
        throw TraversalException{-1, TraversalFailureReason::NormalizedMeasurementMismatch};
      }
    }
    const auto rofBoundaries = frame.getROFrameClusters(static_cast<int>(position));
    if (rofBoundaries.empty() || rofBoundaries.front() != 0 ||
        rofBoundaries.back() != static_cast<int>(globals.size())) {
      throw TraversalException{-1, TraversalFailureReason::NormalizedMeasurementMismatch};
    }
    for (std::size_t rof = 0; rof + 1 < rofBoundaries.size(); ++rof) {
      const int first = rofBoundaries[rof];
      const int last = rofBoundaries[rof + 1];
      if (first < 0 || last < first || last > static_cast<int>(globals.size())) {
        throw TraversalException{-1, TraversalFailureReason::NormalizedMeasurementMismatch};
      }
    }
    measurements[position] = globals;
  }
  return {measurements.data(), layerCount};
}

TrackerInitializationResult Tracker::initialize(TimeFrame& frame, const TrackerInitialization& configuration)
{
  TrackerInitializationResult result;
  if (frame.isConfigured()) {
    result.error = TrackerInitializationError::FrameAlreadyConfigured;
    return result;
  }
  if (configuration.plan.iterations.empty()) {
    result.error = TrackerInitializationError::EmptyConfiguration;
    return result;
  }
  if (configuration.catalog.surfaces == nullptr || configuration.catalog.nSurfaces == 0) {
    result.error = TrackerInitializationError::MissingCatalog;
    return result;
  }
  if (!configuration.memoryPool) {
    result.error = TrackerInitializationError::MissingMemoryPool;
    return result;
  }

  DetectorLayout layout{gsl::span<const SurfaceDescriptor>{configuration.catalog.surfaces,
                                                           configuration.catalog.nSurfaces},
                        configuration.layout};
  if (!layout.valid()) {
    result.error = TrackerInitializationError::LayoutInvalid;
    result.layoutError = layout.getError();
    return result;
  }
  DetectorConfiguration detectorConfiguration;
  try {
    detectorConfiguration = prepareDetectorConfiguration(layout, configuration.plan.detector);
  } catch (const TraversalException&) {
    result.error = TrackerInitializationError::TraversalPlanBuildFailed;
    return result;
  }

  std::vector<IterationConfiguration> iterations;
  std::size_t maxEdges = 0;
  std::size_t maxCells = 0;
  iterations.reserve(configuration.plan.iterations.size());

  for (std::size_t iteration = 0; iteration < configuration.plan.iterations.size(); ++iteration) {
    const auto& input = configuration.plan.iterations[iteration];
    if (input.NLayers != 0 && input.NLayers != layout.size()) {
      result.error = TrackerInitializationError::CapacityMismatch;
      result.failedIteration = iteration;
      return result;
    }
    const auto topology = deriveTraversalTopology(layout, input);
    if (!topology.ok()) {
      result.error = TrackerInitializationError::TraversalPlanBuildFailed;
      result.failedIteration = iteration;
      return result;
    }
    IterationConfiguration iterationConfiguration;
    iterationConfiguration.parameters = input;
    iterationConfiguration.parameters.NLayers = static_cast<int>(layout.size());
    iterationConfiguration.topology = *topology.topology;
    try {
      prepareIterationConfiguration(layout, detectorConfiguration, iterationConfiguration, static_cast<int>(iteration));
    } catch (const TraversalException&) {
      result.error = TrackerInitializationError::TraversalPlanBuildFailed;
      result.failedIteration = iteration;
      return result;
    }
    maxEdges = std::max(maxEdges, iterationConfiguration.topology.edges.size());
    maxCells = std::max(maxCells, iterationConfiguration.topology.paths.size());
    iterations.push_back(std::move(iterationConfiguration));
  }

  if (!frame.configure(std::move(layout), maxEdges, maxCells, configuration.memoryPool)) {
    result.error = TrackerInitializationError::CapacityMismatch;
    return result;
  }
  mExecutionPolicy = configuration.plan.execution;
  mDetectorConfiguration = std::move(detectorConfiguration);
  mIterations = std::move(iterations);
  mFrame = &frame;
  return result;
}

bool Tracker::isConfiguredFor(const TimeFrame& frame) const noexcept
{
  return mFrame == &frame && !mIterations.empty() && frame.isConfigured();
}

void Tracker::computeTracksMClabels(TimeFrame& frame) const
{
  bounded_vector<MCCompLabel> trackLabels(frame.getMemoryPool().get());
  if (!frame.hasMCinformation()) {
    frame.getTrackLabels().swap(trackLabels);
    return;
  }

  const auto& tracks = frame.getGenericTracks();
  const auto& references = frame.getTrackClusterIndices();
  trackLabels.reserve(tracks.size());

  struct Candidate {
    MCCompLabel representative;
    std::size_t count{0};
    std::size_t lastSeenCluster{0};
  };

  for (const auto& track : tracks) {
    if (!isValidTrackRange(track, static_cast<uint32_t>(references.size()))) {
      throw std::logic_error{"Tracker::computeTracksMClabels(): invalid track cluster-reference range"};
    }

    std::vector<Candidate> candidates;
    std::size_t attachedClusters = 0;
    for (uint32_t index = track.firstClusterRef; index < track.clusterRefEnd; ++index) {
      const auto& reference = references[index];
      if (!reference.isValid() || frame.getSurfaceMeasurement(reference.layer, reference.clusterId) == nullptr) {
        throw std::logic_error{"Tracker::computeTracksMClabels(): unresolved track cluster reference"};
      }

      ++attachedClusters;
      for (const auto& label : frame.getLabels(reference.layer, reference.clusterId)) {
        const auto candidate = std::find_if(candidates.begin(), candidates.end(), [&label](const auto& current) {
          return label == current.representative;
        });
        if (candidate == candidates.end()) {
          candidates.push_back({label, 1, attachedClusters});
        } else if (candidate->lastSeenCluster != attachedClusters) {
          ++candidate->count;
          candidate->lastSeenCluster = attachedClusters;
        }
      }
    }

    MCCompLabel winner;
    if (candidates.empty()) {
      winner.setFakeFlag();
    } else {
      const auto best = std::max_element(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        return left.count < right.count;
      });
      winner = best->representative;
      // A single attached cluster without the winning identity makes the
      // reconstructed track fake.
      if (best->count != attachedClusters) {
        winner.setFakeFlag();
      }
    }
    trackLabels.push_back(winner);
  }

  frame.getTrackLabels().swap(trackLabels);
}

void Tracker::configureBeamPosition(TimeFrame& frame) const
{
  const auto& params = mIterations.front().parameters;
  if (!params.UseDiamond) {
    return;
  }
  const float systErrY2 = mDetectorConfiguration.systError2Row.empty() ? 0.f : mDetectorConfiguration.systError2Row[0];
  const float layerRes = mDetectorConfiguration.layerResolution.empty() ? 0.f : mDetectorConfiguration.layerResolution[0];
  frame.setBeamPosition(params.Diamond[0], params.Diamond[1], params.DiamondCov[3], layerRes, systErrY2);
}

TrackingResult Tracker::run(TimeFrame& frame, TrackerTraits& traits)
{
  if (!isConfiguredFor(frame)) {
    throw TraversalException{-1, TraversalFailureReason::MissingLayout};
  }
  float total{0.f};
  std::vector<std::size_t> acceptedTrackCounts;
  auto& estimator = frame.getCapacityEstimator();
  bool estimatorTransactionStarted{false};
  const auto rollbackEstimator = [&] {
    if (estimatorTransactionStarted) {
      estimator.rollbackTransaction();
      estimatorTransactionStarted = false;
    }
  };
  try {
    estimator.beginTransaction();
    estimatorTransactionStarted = true;
    configureBeamPosition(frame);
    auto& scratch = frame.getScratch();
    acceptedTrackCounts.reserve(mIterations.size());
    std::array<gsl::span<const GlobalMeasurement>, MaxLayoutSurfaces> measurementSpans;
    const auto layerGlobalMeasurements = prepareTimeFrame(frame, measurementSpans);
    const auto& memoryPool = frame.getMemoryPool();
    // Apply a tighter event-local limit when configured; this also lets
    // workflows and tests inject a resource failure after loading.
    if (mExecutionPolicy.MaxMemory != std::numeric_limits<size_t>::max() &&
        memoryPool->getMaxMemory() > mExecutionPolicy.MaxMemory) {
      memoryPool->setMaxMemory(mExecutionPolicy.MaxMemory);
    }
    for (int iteration = 0; iteration < static_cast<int>(mIterations.size()); ++iteration) {
      const auto& configuration = mIterations[iteration];
      const auto& trkParam = configuration.parameters;
      if (trkParam.PassFlags[IterationStep::UseUPCMask]) {
        frame.useUPCMask();
      }

      const auto acceptedTrackBegin = frame.getGenericTracks().size();
      IterationContext context{iteration, frame, scratch,
                               configuration.getTopologyView(frame.getLayout().getSurfaceCatalog()),
                               configuration, mDetectorConfiguration, layerGlobalMeasurements,
                               frame.getBz()};
      initializeIteration(context);
      traits.runTraversal(context);
      acceptedTrackCounts.push_back(frame.getGenericTracks().size() - acceptedTrackBegin);
    }
    computeTracksMClabels(frame);
    if (std::getenv("O2_ITSMFT_PRINT_SLAB_STATS") != nullptr) {
      estimator.print();
    }
    estimator.commitTransaction();
    estimatorTransactionStarted = false;
  } catch (const TraversalException& err) {
    // Structural/configuration failures are not per-TF data failures, so
    // DropTFUponFailure does not apply. Reset before propagating.
    LOGP(error, "CA tracker hit a structural traversal failure: {}", err.what());
    rollbackEstimator();
    frame.resetTimeFrame();
    throw;
  } catch (const BoundedMemoryResource::MemoryLimitExceeded& err) {
    // Recoverable per-TF resource failure: the bounded pool budget was
    // exceeded for this TimeFrame.
    LOGP(error, "CA tracker exceeded memory limit: {}", err.what());
    rollbackEstimator();
    frame.resetTimeFrame();
    if (mExecutionPolicy.DropTFUponFailure) {
      return TrackingResult{TrackingOutcome::RecoverableDropped, 0.f};
    }
    throw;
  } catch (const std::bad_alloc& err) {
    // Some CA scratch containers use the plain heap instead of the bounded
    // pool, so memory pressure can surface as bad_alloc. Handle it likewise.
    LOGP(error, "CA tracker allocation failed: {}", err.what());
    rollbackEstimator();
    frame.resetTimeFrame();
    if (mExecutionPolicy.DropTFUponFailure) {
      return TrackingResult{TrackingOutcome::RecoverableDropped, 0.f};
    }
    throw;
  } catch (const std::exception& err) {
    // Unclassified exceptions are treated as structural and always propagate;
    // recoverability is not inferred from std::exception alone.
    LOGP(error, "CA tracker failed with an unclassified exception; treating as structural: {}", err.what());
    rollbackEstimator();
    frame.resetTimeFrame();
    throw;
  }

  return TrackingResult{TrackingOutcome::Success, total, std::move(acceptedTrackCounts)};
}

} // namespace o2::itsmft::tracking
