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

// Tracker failure contract: Tracker::run()
// exception classification, wipe-on-every-failure, and the exact drop
// sentinel.
//
// Contract under test (see Tracker.h/Tracker.cxx):
//  - std::invalid_argument (structural/configuration failure): TimeFrame is
//    wiped, then the exception always rethrows, regardless of
//    DropTFUponFailure.
//  - BoundedMemoryResource::MemoryLimitExceeded
//    (recoverable, per-TF resource failures): TimeFrame is wiped;
//    DropTFUponFailure=true returns TrackingOutcome::RecoverableDropped
//    sentinel, DropTFUponFailure=false rethrows.
//  - Valid empty input (a real layout/topology with zero loaded clusters)
//    completes without throwing and returns a non-negative, non-sentinel
//    result.
//  - A tracker instance that dropped one TimeFrame can immediately process a
//    following one successfully.
//
// Every fixture below establishes a real layout/plan and selected workspace
// and then loads a normalized source -- even the structural-failure cases,
// and even when that source carries zero clusters/ROFs -- before running
// tracking. This is load-bearing, not incidental: TimeFrame::initialise()
// unconditionally calls getNrof(layer) = mROFramesClusters[layer].size()-1
// on every layer, and a never-loaded (default-constructed, size-0)
// mROFramesClusters underflows that subtraction, corrupting memory deep
// inside prepareClusters() rather than throwing a clean exception.
// loadNormalizedSource() sizes mROFramesClusters[layer] to rofs.size()+1 for
// every layer regardless of whether clusters/rofs are empty, which is what
// makes that call, and every "iterate 0..getNrof()" loop reached afterward,
// safe. The structural-failure cases below produce their std::invalid_argument
// through an invalid TrackingParameters/index-table configuration, not
// through a missing/stale plan: Gate 4 B2 Slice 2 removed the plan-currency
// concept entirely (initialiseTimeFrame() now takes the plan as an explicit
// layout/topology view parameter, so "no plan" is no longer a state a
// caller can even construct) -- see the removed
// StructuralFailureViaStaleLayoutAlwaysRethrowsAndWipes test's replacement
// note below for what covers the "always rethrows and wipes" contract now.
//
// The recoverable-failure fixtures tighten the already-used frame allocator
// to its current usage. The next tracking allocation then exercises the
// normal bounded-resource failure/reset contract without changing config.

#define BOOST_TEST_MODULE ITSMFT Tracker failure contract
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include "TrackingParameterTestSupport.h"
#include <boost/test/unit_test.hpp>

#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include <gsl/gsl>
#include <oneapi/tbb/task_arena.h>

#include <TGeoGlobalMagField.h>
#include "Field/MagneticField.h"

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "ITSMFTTracking/Tracker.h"
#include "ITSMFTTracking/Configuration.h"
#include "ITSMFTTracking/IOUtils.h"
#include "ITSMFTTracking/ITSMFTDetectorDefinitions.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"
#include "ITSMFTTracking/detail/TimeFrameScratch.h"
#include "ITSMFTTracking/ClusterDecoding.h"
#include "ITSMFTTracking/IOUtils.h"
#include "ITSMFTTracking/TimeFrame.h"
#include "ITSMFTTracking/TrackerTraits.h"
#include "ITSMFTTracking/TrackingConfigParam.h"
#include "ITSMFTTracking/Constants.h"
#include "ITSMFTTracking/ROFLookupTables.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::itsmft;
using namespace o2::itsmft::tracking;

namespace
{

// Deterministic, geometry-free stand-in for GeometryClusterDecoder<DetId>,
// identical construction to testTimeFrameLifecycle.cxx /
// testTimeFrameNormalizedSource.cxx / testMultiSourceLoading.cxx.
class LegacyLikeDecoder final : public ClusterDecoder
{
 public:
  explicit LegacyLikeDecoder(o2::detectors::DetID::ID detector) : mDetector(detector) {}

  o2::itsmft::tracking::ClusterDecodeResult decode(
    const CompClusterExt& cluster,
    BoundedPatternCursor& patterns,
    const TopologyDictionary* dict,
    uint32_t,
    bool applySysErrors) const override
  {
    const auto clusterData = o2::itsmft::ioutils::extractClusterDataBounded(cluster, patterns, dict);
    if (!clusterData.ok()) {
      o2::itsmft::tracking::ClusterDecodeResult result;
      result.error = clusterData.error;
      return result;
    }

    o2::itsmft::tracking::ClusterDecodeResult result;
    const int sensorID = cluster.getSensorID();
    auto& decoded = result.decoded;
    decoded.global = {static_cast<float>(sensorID) * 10.f, static_cast<float>(cluster.getRow()), static_cast<float>(cluster.getCol())};
    decoded.cylinderFrame = {static_cast<float>(sensorID) + 100.f, static_cast<float>(cluster.getRow()) + 1.f, static_cast<float>(cluster.getCol()) + 2.f, 0.01f * sensorID};
    decoded.rowColumnCovariance = {clusterData.sig2Row, 0.f, clusterData.sig2Col};
    decoded.shape = clusterData.shape;
    decoded.layer = sensorID;
    return result;
  }

 private:
  o2::detectors::DetID::ID mDetector;
};

const TopologyDictionary& dict()
{
  static const TopologyDictionary d;
  return d;
}

// TrackerTraits::findRoads() unconditionally touches the global
// o2::base::Propagator singleton on first use, which in turn requires
// TGeoGlobalMagField to already hold a real o2::field::MagneticField
// object -- with none set (the state of every other test in this suite,
// none of which calls Tracker::run() end to end), Propagator falls
// back to a legacy FairRunAna singleton that also does not exist in this
// process and segfaults dereferencing it. Only the tests that expect a
// genuinely successful Tracker::run() (valid empty input,
// continued processing after a drop) reach findRoads(); the
// structural/recoverable-failure tests throw/return before ever getting
// there and do not need this. A trivial default-constructed
// MagneticField (no field map file, zero solenoid current) is sufficient
// -- these tests never fit or propagate an actual trajectory since there
// are no clusters. TGeoGlobalMagField::Instance()->Lock() only allows one
// SetField() call per process, so this must run at most once.
void ensureTrivialMagneticFieldIsSet()
{
  static const bool done = [] {
    TGeoGlobalMagField::Instance()->SetField(new o2::field::MagneticField());
    TGeoGlobalMagField::Instance()->Lock();
    return true;
  }();
  (void)done;
}

constexpr std::array<unsigned char, 3> onePixelPattern{1, 1, 0x80};
constexpr std::array<unsigned char, 3> threePixelPattern{1, 3, 0xE0};

std::vector<unsigned char> concatPatterns(std::initializer_list<gsl::span<const unsigned char>> parts)
{
  std::vector<unsigned char> bytes;
  for (const auto& p : parts) {
    bytes.insert(bytes.end(), p.begin(), p.end());
  }
  return bytes;
}

std::vector<SurfaceDescriptor> makeITSTestCatalog()
{
  std::vector<SurfaceDescriptor> surfaces;
  surfaces.reserve(ITSNLayers);
  for (uint16_t i = 0; i < ITSNLayers; ++i) {
    surfaces.push_back(SurfaceDescriptor{i, static_cast<uint8_t>(o2::detectors::DetID::ITS), SurfaceKind::Cylinder});
    surfaces.back().chartRange = {-20.f, 20.f};
    // Matches o2::itsmft::resetDetectorDefaults(..., DetID::ITS)'s LayerxX0
    // default, so TrackerTraits::initialiseTimeFrame()'s LegacyMaterialMismatch
    // compatibility check passes for these unperturbed fixtures.
    const float xOverX0 = kNominalITSLayerX0[i];
    surfaces.back().material.xOverX0 = xOverX0;
    surfaces.back().material.arealDensityGPerCm2 = xOverX0 * o2::its::constants::Radl * o2::its::constants::Rho;
  }
  return surfaces;
}

std::vector<LayerId> identitySurfaces(uint16_t nLayers)
{
  std::vector<LayerId> mapping;
  mapping.reserve(nLayers);
  for (uint16_t i = 0; i < nLayers; ++i) {
    mapping.push_back(LayerId{i});
  }
  return mapping;
}

struct Fixture {
  std::vector<CompClusterExt> clusters;
  std::vector<unsigned char> patterns;
  std::vector<ROFRecord> rofs;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> labels;
};

// 4 clusters on layers {0,1,0,2}, partitioned into 3 ROFs. Same shape as
// testTimeFrameLifecycle.cxx's fixture -- only needed to give the
// recoverable-failure fixture genuine per-event content to wipe.
Fixture makeFixture()
{
  Fixture f;
  f.clusters = {
    CompClusterExt{10, 20, CompCluster::InvalidPatternID, 0},
    CompClusterExt{11, 21, CompCluster::InvalidPatternID, 1},
    CompClusterExt{12, 22, CompCluster::InvalidPatternID, 0},
    CompClusterExt{13, 23, CompCluster::InvalidPatternID, 2},
  };
  f.patterns = concatPatterns({onePixelPattern, threePixelPattern, onePixelPattern, threePixelPattern});
  f.rofs = {
    ROFRecord{{100, 5}, 0, 0, 2},
    ROFRecord{{140, 5}, 1, 2, 1},
    ROFRecord{{1000, 6}, 2, 3, 1}};
  for (uint32_t i = 0; i < f.clusters.size(); ++i) {
    f.labels.addElement(i, o2::MCCompLabel{static_cast<int>(i) + 1, 0, 0});
  }
  return f;
}

std::vector<TrackingParameters> makeOneIterationITSParams(bool dropTFUponFailure, size_t maxMemory = std::numeric_limits<size_t>::max())
{
  std::vector<TrackingParameters> params(1);
  resetDetectorDefaults(params[0], o2::detectors::DetID::ITS);
  params[0].DropTFUponFailure = dropTFUponFailure;
  params[0].MaxMemory = maxMemory;
  return params;
}

// A valid FirstPass iteration 0 followed by a non-FirstPass (RebuildClusterLUT
// only, matching the legacy ITS async-iteration-3 shape) iteration 1, both ITS
// defaults -- callers mutate params[1]'s index-table fields to construct a
// deliberate mismatch against the configuration iteration 0 will commit.
std::vector<TrackingParameters> makeTwoIterationITSParams(bool dropTFUponFailure)
{
  std::vector<TrackingParameters> params(2);
  resetDetectorDefaults(params[0], o2::detectors::DetID::ITS);
  resetDetectorDefaults(params[1], o2::detectors::DetID::ITS);
  params[1].PassFlags = IterationSteps{IterationStep::RebuildClusterLUT};
  for (auto& p : params) {
    p.DropTFUponFailure = dropTFUponFailure;
  }
  return params;
}

// Bundles a TimeFrame, real backend, Tracker, and bounded memory pool -- the
// minimal wiring Tracker::run() needs for the ITS configuration tests below.
struct Rig {
  explicit Rig(bool dropTFUponFailure, size_t maxMemory = std::numeric_limits<size_t>::max())
    : pool(std::make_shared<BoundedMemoryResource>()),
      params(makeOneIterationITSParams(dropTFUponFailure, maxMemory)),
      tracker()
  {
    traits.setNThreads(1, arena);
    frame.setBz(0.5f);
  }

  // Stage a GenericTrack and its reference to exercise resetTimeFrame().
  void stageStaleState()
  {
    frame.getTrackClusterIndices().push_back(TrackClusterReference{LayerId{0}, 0, 0});
    GenericTrack track{};
    track.clusterRefEnd = static_cast<uint32_t>(frame.getTrackClusterIndices().size());
    frame.getGenericTracks().push_back(track);
    BOOST_REQUIRE(!frame.getGenericTracks().empty());
    BOOST_REQUIRE(!frame.getTrackClusterIndices().empty());
  }

  std::shared_ptr<BoundedMemoryResource> pool;
  std::vector<TrackingParameters> params;
  TimeFrame frame;
  TrackerTraits traits;
  Tracker tracker;
  // Scratch carries non-owning runtime ROF views. Keep these adapter-edge
  // builders alive across load, initialise, and failure/replacement calls.
  std::optional<o2::its::ROFOverlapTable<ITSNLayers>> rofTable;
  std::optional<o2::its::ROFVertexLookupTable<ITSNLayers>> vertexTable;
  std::optional<o2::its::ROFMaskTable<ITSNLayers>> mask;
  std::shared_ptr<tbb::task_arena> arena;
  std::vector<SurfaceDescriptor> catalog;

  // Builds and atomically installs the complete static configuration.
  void establishValidLayout()
  {
    catalog = makeITSTestCatalog();
    const SurfaceCatalogView catalogView{catalog.data(), static_cast<uint32_t>(catalog.size())};
    TrackerInitialization configuration;
    configuration.catalog = catalogView;
    configuration.memoryPool = pool;
    const auto orderedSurfaces = identitySurfaces(ITSNLayers);
    configuration.layout = makeDetectorLayout();
    configuration.plan = o2::itsmft::tracking::test::makeTrackingPlan(params);
    const auto result = tracker.initialize(frame, configuration);
    BOOST_REQUIRE(result.ok());
    BOOST_REQUIRE_EQUAL(frame.getLayout().size(), orderedSurfaces.size());
  }

  // Loads clusters (or, with an empty Fixture, zero clusters -- still a
  // valid load that sizes every per-layer ROF boundary table to a real,
  // if trivial, state) through the same normalized-loading path production
  // code uses. This sizing is load-bearing: TimeFrame::initialise() calls
  // getNrof(layer) = mROFramesClusters[layer].size() - 1 unconditionally,
  // and a never-loaded (default-constructed, size-0) mROFramesClusters
  // underflows that subtraction, crashing deep inside prepareClusters()
  // before any failure-contract check ever runs. loadNormalizedSource()
  // sizes mROFramesClusters[layer] to rofs.size()+1 for every layer even
  // when rofs/clusters are empty, so calling it with an empty Fixture is
  // the only proven-safe way to reach a genuinely valid, still-empty
  // TimeFrame state.
  void loadSource(const Fixture& f)
  {
    LegacyLikeDecoder decoder{o2::detectors::DetID::ITS};
    const o2::InteractionRecord origin{50, 5};
    const ROFTimingConfig timing{40, 0, 0, 0};
    const auto& layout = frame.getLayout();
    const auto layerMapping = identitySurfaces(ITSNLayers);
    const auto result = loadTimeFrameSource(frame, decoder, origin, timing, f.clusters, f.patterns, f.rofs, &dict(),
                                            f.labels.getIndexedSize() > 0 ? &f.labels : nullptr, o2::detectors::DetID::ITS,
                                            gsl::span<const LayerId>{layerMapping}, layout.getSurfaceCatalog());
    BOOST_REQUIRE(result.ok());

    // TrackerTraits::computeLayerTracklets() reads per-layer ROF counts
    // from mROFOverlapTableView (o2::its::LayerTiming), a separate table
    // from mROFramesClusters/getNrof() -- it is never populated by
    // loadNormalizedSource() and defaults to an unconfigured/garbage view.
    // A traversal that reaches computeLayerTracklets() without this being
    // set derives its ROF loop bound from that garbage view and walks out
    // of bounds. Mirrors the workflow timing-table construction's
    // shape, but with every layer given the same trivial timing matching
    // this fixture's single combined ROF stream (real production input has
    // per-detector-param ROF length/delay/bias; none of that is exercised
    // by the failure-contract cases here, only the ROF *count* is load
    // -bearing).
    o2::its::LayerTiming timing2{};
    timing2.mNROFsTF = static_cast<unsigned int>(f.rofs.size());
    timing2.mROFLength = 40;
    rofTable.emplace();
    for (int iLayer = 0; iLayer < ITSNLayers; ++iLayer) {
      rofTable->defineLayer(iLayer, timing2);
    }
    rofTable->init();
    vertexTable.emplace();
    for (int iLayer = 0; iLayer < ITSNLayers; ++iLayer) {
      vertexTable->defineLayer(iLayer, timing2);
    }
    vertexTable->init();

    mask.emplace(*rofTable);
    mask->resetMask();
    for (int iLayer = 0; iLayer < ITSNLayers; ++iLayer) {
      mask->setROFsEnabled(iLayer, 0, timing2.mNROFsTF, 1);
    }
    frame.setROFViews(RuntimeROFViews{rofTable->getView(), vertexTable->getView(), mask->getView(), {}});
  }

  // Set the event-local budget at the current usage; the next allocation is
  // the controlled recoverable failure.
  void forceMemoryLimitAtCurrentUsage()
  {
    const auto used = pool->getUsedMemory();
    pool->setMaxMemory(used);
  }

  void restoreUnboundedMemory()
  {
    pool->setMaxMemory(std::numeric_limits<size_t>::max());
  }
};

Fixture emptyFixture()
{
  return Fixture{};
}

} // namespace

// --- Recoverable failure: DropTFUponFailure decides, always wipes --------

BOOST_AUTO_TEST_CASE(StructuralFailureAlwaysRethrowsAndResetsTimeFrame)
{
  ensureTrivialMagneticFieldIsSet();
  for (const bool dropFlag : {false, true}) {
    Rig rig{dropFlag};
    rig.establishValidLayout();
    rig.loadSource(makeFixture());
    rig.stageStaleState();
    auto measurements = rig.frame.getGlobalMeasurements(LayerId{0});
    BOOST_REQUIRE(!measurements.empty());
    measurements.front().clusterId = std::numeric_limits<uint32_t>::max();

    BOOST_CHECK_THROW(rig.tracker.run(rig.frame, rig.traits), std::invalid_argument);
    BOOST_CHECK_EQUAL(rig.frame.getTotalMeasurements(), 0u);
    BOOST_CHECK(rig.frame.getGenericTracks().empty());
    BOOST_CHECK(rig.frame.getTrackClusterIndices().empty());
  }
}

BOOST_AUTO_TEST_CASE(RecoverableFailureDroppedReturnsExactSentinelAndWipes)
{
  Rig rig{/*dropTFUponFailure=*/true};
  rig.establishValidLayout();
  rig.loadSource(makeFixture());
  BOOST_REQUIRE(rig.frame.getTotalMeasurements() > 0u);

  rig.forceMemoryLimitAtCurrentUsage();

  const auto result = rig.tracker.run(rig.frame, rig.traits);

  BOOST_CHECK(result.outcome == TrackingOutcome::RecoverableDropped);
  BOOST_CHECK_EQUAL(rig.frame.getTotalMeasurements(), 0u);
  BOOST_CHECK(rig.frame.getGenericTracks().empty());
}

BOOST_AUTO_TEST_CASE(RecoverableFailureNotDroppedRethrowsButStillWipesFirst)
{
  Rig rig{/*dropTFUponFailure=*/false};
  rig.establishValidLayout();
  rig.loadSource(makeFixture());
  BOOST_REQUIRE(rig.frame.getTotalMeasurements() > 0u);

  rig.forceMemoryLimitAtCurrentUsage();

  BOOST_CHECK_THROW(rig.tracker.run(rig.frame, rig.traits), BoundedMemoryResource::MemoryLimitExceeded);

  // Wipe must have already happened before the exception propagated -- not
  // "the process is going down anyway".
  BOOST_CHECK_EQUAL(rig.frame.getTotalMeasurements(), 0u);
  BOOST_CHECK(rig.frame.getGenericTracks().empty());
}

// --- std::bad_alloc: recoverable, same drop-or-rethrow policy ------------
//
// A real ten-disk MFT event exercises Tracker::run() while a test-owned
// upstream resource injects the plain-heap failure category.

// --- Index-table configuration failures: structural, always rethrow -------
//
BOOST_AUTO_TEST_CASE(InvalidIndexTableConfigurationIsRejectedBeforeTimeFrameConfiguration)
{
  for (const bool dropFlag : {false, true}) {
    Rig rig{dropFlag};
    rig.params[0].RowBins = 0; // structurally invalid
    rig.catalog = makeITSTestCatalog();
    const auto orderedSurfaces = identitySurfaces(ITSNLayers);
    TrackerInitialization configuration;
    configuration.catalog = {rig.catalog.data(), static_cast<uint32_t>(rig.catalog.size())};
    configuration.memoryPool = rig.pool;
    configuration.layout = makeDetectorLayout();
    configuration.plan = o2::itsmft::tracking::test::makeTrackingPlan(rig.params);
    const auto result = rig.tracker.initialize(rig.frame, configuration);
    BOOST_CHECK(!result.ok());
    BOOST_CHECK(!rig.frame.isConfigured());
  }
}

BOOST_AUTO_TEST_CASE(IterationSpecificInvalidKernelIsRejectedBeforeCommit)
{
  for (const bool dropFlag : {false, true}) {
    Rig rig{dropFlag};
    rig.params = makeTwoIterationITSParams(dropFlag);
    rig.params[1].TrackletMinPt = -1.f;
    rig.catalog = makeITSTestCatalog();
    TrackerInitialization configuration;
    configuration.catalog = {rig.catalog.data(), static_cast<uint32_t>(rig.catalog.size())};
    configuration.memoryPool = rig.pool;
    configuration.layout = makeDetectorLayout();
    configuration.plan = o2::itsmft::tracking::test::makeTrackingPlan(rig.params);
    const auto result = rig.tracker.initialize(rig.frame, configuration);
    BOOST_CHECK(!result.ok());
    BOOST_CHECK_EQUAL(result.failedIteration, 1u);
    BOOST_CHECK(!rig.frame.isConfigured());
    BOOST_CHECK(rig.frame.getGenericTracks().empty());
  }
}

// --- Valid empty input -----------------------------------------------------

BOOST_AUTO_TEST_CASE(ValidEmptyInputCompletesWithoutErrorAndProducesNoTracks)
{
  ensureTrivialMagneticFieldIsSet();
  Rig rig{/*dropTFUponFailure=*/false};
  rig.establishValidLayout();
  rig.loadSource(emptyFixture());
  BOOST_REQUIRE_EQUAL(rig.frame.getTotalMeasurements(), 0u);

  TrackingResult result{TrackingOutcome::Structural, std::numeric_limits<float>::quiet_NaN()};
  BOOST_CHECK_NO_THROW(result = rig.tracker.run(rig.frame, rig.traits));

  BOOST_CHECK(result.outcome == TrackingOutcome::Success);
  BOOST_CHECK(result.elapsedMs >= 0.f);
  BOOST_CHECK_EQUAL(rig.frame.getGenericTracks().size(), 0u);
}

BOOST_AUTO_TEST_CASE(TrackingOutcomeValuesAreDistinct)
{
  BOOST_CHECK(TrackingOutcome::Success != TrackingOutcome::RecoverableDropped);
  BOOST_CHECK(TrackingOutcome::Success != TrackingOutcome::Structural);
  BOOST_CHECK(TrackingOutcome::RecoverableDropped != TrackingOutcome::Structural);

  constexpr TrackingResult defaulted{};
  BOOST_CHECK(defaulted.outcome == TrackingOutcome::Success);
  BOOST_CHECK_EQUAL(defaulted.elapsedMs, 0.f);
}

// --- No stale TimeFrame/GenericTrack state survives -------------------------
//
// A recoverable-dropped return must clear GenericTrack storage along with
// the normalized measurements.

BOOST_AUTO_TEST_CASE(RecoverableDroppedLeavesNoStaleGenericTrackState)
{
  Rig rig{/*dropTFUponFailure=*/true};
  rig.establishValidLayout();
  rig.loadSource(makeFixture());
  rig.stageStaleState();

  rig.forceMemoryLimitAtCurrentUsage();
  const auto result = rig.tracker.run(rig.frame, rig.traits);

  BOOST_CHECK(result.outcome == TrackingOutcome::RecoverableDropped);
  BOOST_CHECK(rig.frame.getGenericTracks().empty());
  BOOST_CHECK(rig.frame.getTrackClusterIndices().empty());
}

// --- Continued processing after a drop ------------------------------------

BOOST_AUTO_TEST_CASE(TrackerRemainsUsableAfterADroppedTimeFrame)
{
  ensureTrivialMagneticFieldIsSet();
  Rig rig{/*dropTFUponFailure=*/true};
  rig.establishValidLayout();
  rig.loadSource(makeFixture());

  rig.forceMemoryLimitAtCurrentUsage();
  const auto dropped = rig.tracker.run(rig.frame, rig.traits);
  BOOST_REQUIRE(dropped.outcome == TrackingOutcome::RecoverableDropped);

  // Restore headroom and process a fresh (here, empty) TimeFrame on the
  // SAME Tracker/TrackerTraits instance -- proving the tracker/device stays
  // usable after a drop, matching the DPL device staying alive.
  rig.restoreUnboundedMemory();
  rig.loadSource(emptyFixture());

  TrackingResult result{TrackingOutcome::Structural, std::numeric_limits<float>::quiet_NaN()};
  BOOST_CHECK_NO_THROW(result = rig.tracker.run(rig.frame, rig.traits));
  BOOST_CHECK(result.outcome == TrackingOutcome::Success);
  BOOST_CHECK(result.elapsedMs >= 0.f);
}
