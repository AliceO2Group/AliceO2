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
//  - TraversalException (structural/configuration failure): TimeFrame is
//    wiped, then the exception always rethrows, regardless of
//    DropTFUponFailure.
//  - BoundedMemoryResource::MemoryLimitExceeded and std::bad_alloc
//    (recoverable, per-TF resource failures): TimeFrame is wiped;
//    DropTFUponFailure=true returns TrackingOutcome::RecoverableDropped
//    sentinel, DropTFUponFailure=false rethrows.
//  - Any other std::exception (e.g. std::runtime_error): treated as
//    unclassified/structural, wiped, always rethrows regardless of the
//    flag -- it must never be silently converted into a dropped-TF result.
//  - Valid empty input (a real layout/topology with zero loaded clusters)
//    completes without throwing and returns a non-negative, non-sentinel
//    result.
//  - A tracker instance that dropped one TimeFrame can immediately process a
//    following one successfully.
//
// The std::bad_alloc and unclassified-std::exception cases use a real MFT
// road and a test-owned upstream memory resource that throws during normal
// traversal allocation. This keeps failure injection outside the production
// Tracker and refit APIs.
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
// safe. The structural-failure cases below produce their TraversalException
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
#include <functional>
#include <memory>
#include <memory_resource>
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
#include "ITSMFTTracking/detail/ITSSharedClusterCompatibility.h"
#include "ITSMFTTracking/IOUtils.h"
#include "ITSMFTTracking/ITSMFTDetectorDefinitions.h"
#include "ITSMFTTracking/detail/MFTFwdTrackHelpers.h"
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

enum class AllocationFailure { None,
                               BadAlloc,
                               UnclassifiedRuntimeError };

class ControlledMemoryResource final : public std::pmr::memory_resource
{
 public:
  using FailurePredicate = std::function<bool()>;

  void arm(AllocationFailure failure, FailurePredicate predicate = {})
  {
    mFailureCount = 0;
    mPredicate = std::move(predicate);
    mFailure = failure;
  }

  void disarm()
  {
    mFailure = AllocationFailure::None;
    mPredicate = {};
  }

  int failureCount() const noexcept { return mFailureCount; }

 private:
  void* do_allocate(std::size_t bytes, std::size_t alignment) final
  {
    if (mFailure != AllocationFailure::None && (!mPredicate || mPredicate())) {
      ++mFailureCount;
      if (mFailure == AllocationFailure::BadAlloc) {
        throw std::bad_alloc{};
      }
      throw std::runtime_error{"controlled upstream allocation failure"};
    }
    return mUpstream->allocate(bytes, alignment);
  }

  void do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) final
  {
    mUpstream->deallocate(pointer, bytes, alignment);
  }

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept final
  {
    return this == &other;
  }

  AllocationFailure mFailure{AllocationFailure::None};
  FailurePredicate mPredicate;
  int mFailureCount{0};
  std::pmr::memory_resource* mUpstream{BoundedMemoryResource::cachingUpstream()};
};

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

  // Stages one pending sidecar entry and one GenericTrack/TrackClusterReference
  // pair directly on `frame` -- deliberately not through a real CA seed (out
  // of scope here): only frame.resetTimeFrame()'s unconditional clear of these two
  // containers and the workflow-edge sidecar reset are under test.
  void stageStaleState()
  {
    ITSSharedClusterCompatibilityTransaction txn{sidecar};
    BOOST_REQUIRE(txn.validate(0));
    txn.reserve();
    txn.append(0);
    BOOST_REQUIRE_EQUAL(sidecar.pendingSize(), 1u);

    frame.getTrackClusterIndices().push_back(TrackClusterReference{LayerId{0}, 0, 0});
    GenericTrack track{};
    track.clusterRefEnd = static_cast<uint32_t>(frame.getTrackClusterIndices().size());
    frame.getGenericTracks().push_back(track);
    BOOST_REQUIRE(!frame.getGenericTracks().empty());
    BOOST_REQUIRE(!frame.getTrackClusterIndices().empty());
  }

  void resetPublication() noexcept { sidecar.clear(); }

  std::shared_ptr<BoundedMemoryResource> pool;
  std::vector<TrackingParameters> params;
  TimeFrame frame;
  TrackerTraits traits;
  Tracker tracker;
  ITSSharedClusterCompatibility sidecar;
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

class MftRoadDecoder final : public ClusterDecoder
{
 public:
  explicit MftRoadDecoder(std::vector<DecodedCluster> clusters) : mClusters{std::move(clusters)} {}

  ClusterDecodeResult decode(const CompClusterExt& cluster, BoundedPatternCursor& patterns,
                             const TopologyDictionary* dictionary, uint32_t externalIndex, bool) const final
  {
    const auto clusterData = ioutils::extractClusterDataBounded(cluster, patterns, dictionary);
    if (!clusterData.ok()) {
      ClusterDecodeResult result;
      result.error = clusterData.error;
      return result;
    }
    ClusterDecodeResult result;
    if (externalIndex >= mClusters.size()) {
      return result;
    }
    auto decoded = mClusters[externalIndex];
    decoded.shape = clusterData.shape;
    result.decoded = decoded;
    return result;
  }

 private:
  std::vector<DecodedCluster> mClusters;
};

std::vector<DecodedCluster> makeMftRoad(const TrackingParameters& parameters, float bz)
{
  std::vector<DecodedCluster> result;
  result.reserve(MFTNLayers);
  float x = 3.f;
  float y = 1.5f;
  float z = detail::mftLayerZ(0);
  for (int layer = 0; layer < MFTNLayers; ++layer) {
    DecodedCluster cluster{};
    cluster.global = {x, y, z};
    cluster.rowColumnCovariance = {1.e-2f, 0.f, 1.e-2f};
    cluster.layer = layer;
    result.push_back(cluster);
    if (layer + 1 == MFTNLayers) {
      break;
    }
    const float nextZ = detail::mftLayerZ(layer + 1);
    float nextX = 0.f;
    float nextY = 0.f;
    detail::mftTrackletProject(x, y, z, parameters.Diamond[0], parameters.Diamond[1], parameters.Diamond[2],
                               layer, layer + 1, bz, parameters.TrackletMinPt, nextX, nextY);
    x = nextX;
    y = nextY;
    z = nextZ;
  }
  return result;
}

std::vector<SurfaceDescriptor> makeMftCatalog()
{
  std::vector<SurfaceDescriptor> catalog;
  catalog.reserve(MFTNLayers);
  for (uint16_t layer = 0; layer < MFTNLayers; ++layer) {
    SurfaceDescriptor surface{layer, static_cast<uint8_t>(o2::detectors::DetID::MFT), SurfaceKind::Disk};
    surface.chartRange = {kMFTLookupRMin[layer], kMFTLookupRMax[layer]};
    surface.referenceCoordinate = detail::mftLayerZ(layer);
    const float xOverX0 = kNominalMFTLayerX0[layer];
    surface.material.xOverX0 = xOverX0;
    surface.material.arealDensityGPerCm2 = xOverX0 * o2::its::constants::Radl * o2::its::constants::Rho;
    catalog.push_back(surface);
  }
  return catalog;
}

// This fixture forms the smallest established full MFT chain: one hit on
// every disk surface. Its test-owned upstream resource can inject failures
// at selected normal traversal allocations without altering production APIs.
struct MftFailureRig {
  explicit MftFailureRig(bool dropTFUponFailure)
    : pool(std::make_shared<BoundedMemoryResource>(std::numeric_limits<size_t>::max(), &controlledMemory))
  {
    resetDetectorDefaults(parameters, o2::detectors::DetID::MFT);
    parameters.UseDiamond = true;
    parameters.CreateArtefactLabels = false;
    parameters.DropTFUponFailure = dropTFUponFailure;
    frame.setBz(.5f);
    traits.setNThreads(1, arena);
  }

  void configure(std::size_t iterations = 1)
  {
    catalog = makeMftCatalog();
    TrackerInitialization configuration;
    configuration.catalog = {catalog.data(), static_cast<uint32_t>(catalog.size())};
    configuration.memoryPool = pool;
    const auto surfaces = identitySurfaces(MFTNLayers);
    configuration.layout = makeDetectorLayout();
    configuration.plan = o2::itsmft::tracking::test::makeTrackingPlan(parameters);
    configuration.plan.iterations.assign(iterations, parameters);
    BOOST_REQUIRE(tracker.initialize(frame, configuration).ok());
    const auto key = CapacityEstimator::makeKey(SlabSite::Roads, 7,
                                                CapacityEstimator::makeVariant(5, 3), CellPathId{7});
    frame.getCapacityEstimator().update(key, 1000., 12000, 12000, false, false);
    BOOST_REQUIRE_GT(frame.getCapacityEstimator().capacity(key, 1000.), 1024u);
  }

  void loadRoad()
  {
    const auto decoded = makeMftRoad(parameters, frame.getBz());
    std::vector<CompClusterExt> compact;
    std::vector<unsigned char> patterns;
    compact.reserve(decoded.size());
    patterns.reserve(decoded.size() * onePixelPattern.size());
    for (const auto& cluster : decoded) {
      compact.emplace_back(0, 0, CompCluster::InvalidPatternID, cluster.layer);
      patterns.insert(patterns.end(), onePixelPattern.begin(), onePixelPattern.end());
    }
    const std::vector<ROFRecord> rofs{ROFRecord{{100, 5}, 0, 0, static_cast<int>(compact.size())}};
    MftRoadDecoder decoder{decoded};
    const auto& layout = frame.getLayout();
    const auto layerMapping = identitySurfaces(MFTNLayers);
    BOOST_REQUIRE(loadTimeFrameSource(frame, decoder, o2::InteractionRecord{50, 5}, ROFTimingConfig{40, 0, 0, 0},
                                      compact, patterns, rofs, &dict(), nullptr, o2::detectors::DetID::MFT,
                                      gsl::span<const LayerId>{layerMapping}, layout.getSurfaceCatalog())
                    .ok());
    o2::its::LayerTiming timing{};
    timing.mNROFsTF = 1;
    timing.mROFLength = 40;
    rofTable.emplace();
    vertexTable.emplace();
    for (int layer = 0; layer < MFTNLayers; ++layer) {
      rofTable->defineLayer(layer, timing);
      vertexTable->defineLayer(layer, timing);
    }
    rofTable->init();
    vertexTable->init();
    mask.emplace(*rofTable);
    mask->resetMask();
    for (int layer = 0; layer < MFTNLayers; ++layer) {
      mask->setROFsEnabled(layer, 0, 1, 1);
    }
    frame.setROFViews(RuntimeROFViews{rofTable->getView(), vertexTable->getView(), mask->getView(), {}});
  }

  void assertReset() const
  {
    BOOST_CHECK_EQUAL(frame.getTotalMeasurements(), 0u);
    BOOST_CHECK(frame.getGenericTracks().empty());
    BOOST_CHECK(frame.getTrackClusterIndices().empty());
    const auto key = CapacityEstimator::makeKey(SlabSite::Roads, 7,
                                                CapacityEstimator::makeVariant(5, 3), CellPathId{7});
    BOOST_CHECK_GT(frame.getCapacityEstimator().capacity(key, 1000.), 1024u);
  }

  void stageStaleState()
  {
    ITSSharedClusterCompatibilityTransaction txn{sidecar};
    BOOST_REQUIRE(txn.validate(0));
    txn.reserve();
    txn.append(0);
    frame.getTrackClusterIndices().push_back(TrackClusterReference{LayerId{0}, 0, 0});
    GenericTrack track{};
    track.clusterRefEnd = static_cast<uint32_t>(frame.getTrackClusterIndices().size());
    frame.getGenericTracks().push_back(track);
  }

  void resetPublication() noexcept { sidecar.clear(); }

  void armFailure(AllocationFailure failure, ControlledMemoryResource::FailurePredicate predicate = {})
  {
    controlledMemory.arm(failure, std::move(predicate));
  }

  void disarmFailure() { controlledMemory.disarm(); }

  int failureCount() const noexcept { return controlledMemory.failureCount(); }

  ControlledMemoryResource controlledMemory;
  std::shared_ptr<BoundedMemoryResource> pool;
  TrackingParameters parameters{};
  TimeFrame frame;
  TrackerTraits traits;
  Tracker tracker;
  ITSSharedClusterCompatibility sidecar;
  std::shared_ptr<tbb::task_arena> arena;
  std::vector<SurfaceDescriptor> catalog;
  std::optional<o2::its::ROFOverlapTable<MFTNLayers>> rofTable;
  std::optional<o2::its::ROFVertexLookupTable<MFTNLayers>> vertexTable;
  std::optional<o2::its::ROFMaskTable<MFTNLayers>> mask;
};

Fixture emptyFixture()
{
  return Fixture{};
}

} // namespace

// --- Structural failure: always rethrows, always wipes -------------------
//
// Gate 4 B2 Slice 2 removed this section's original mechanism
// (StructuralFailureViaStaleLayoutAlwaysRethrowsAndWipes: establish a valid
// layout, then TimeFrame::invalidateTraversalState() right before running
// tracking to deterministically produce TraversalException{StaleLayout}).
// Neither invalidateTraversalState() nor TraversalFailureReason::StaleLayout
// is reachable any more: initialiseTimeFrame() now takes the plan as an
// explicit topology parameter with no TimeFrame-owned
// currency concept to invalidate. The "TraversalException (structural/
// configuration failure): TimeFrame is wiped, then the exception always
// rethrows, regardless of DropTFUponFailure" contract this test protected is
// still covered below, through a different structural-failure reason
// (InvalidIndexTableConfigurationAlwaysRethrowsAndWipesRegardlessOfFlag /
// IndexTableConfigurationMismatchAlwaysRethrowsAndWipesRegardlessOfFlag): the
// contract under test is about TraversalException as a *category*, not about
// any one specific TraversalFailureReason value.

// --- Recoverable failure: DropTFUponFailure decides, always wipes --------

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

BOOST_AUTO_TEST_CASE(BadAllocDroppedReturnsExactSentinelAndWipes)
{
  ensureTrivialMagneticFieldIsSet();
  MftFailureRig rig{/*dropTFUponFailure=*/true};
  rig.configure();
  rig.loadRoad();
  BOOST_REQUIRE(rig.frame.getTotalMeasurements() > 0u);

  rig.armFailure(AllocationFailure::BadAlloc);
  const auto result = rig.tracker.run(rig.frame, rig.traits);
  rig.disarmFailure();

  BOOST_CHECK(result.outcome == TrackingOutcome::RecoverableDropped);
  BOOST_CHECK_GT(rig.failureCount(), 0);
  rig.assertReset();
}

BOOST_AUTO_TEST_CASE(BadAllocNotDroppedRethrowsButStillWipesFirst)
{
  ensureTrivialMagneticFieldIsSet();
  MftFailureRig rig{/*dropTFUponFailure=*/false};
  rig.configure();
  rig.loadRoad();
  BOOST_REQUIRE(rig.frame.getTotalMeasurements() > 0u);

  rig.armFailure(AllocationFailure::BadAlloc);
  BOOST_CHECK_THROW(rig.tracker.run(rig.frame, rig.traits), std::bad_alloc);
  rig.disarmFailure();

  BOOST_CHECK_GT(rig.failureCount(), 0);
  rig.assertReset();
}

BOOST_AUTO_TEST_CASE(EstimatorLearningRollsBackAfterFailureAndNextEventCommits)
{
  ensureTrivialMagneticFieldIsSet();
  MftFailureRig rig{/*dropTFUponFailure=*/true};
  rig.configure();
  auto& estimator = rig.frame.getCapacityEstimator();
  const auto key = CapacityEstimator::makeKey(SlabSite::Tracklets, 0, 0, EdgeId{0});
  constexpr double scale = 1.;
  estimator.update(key, scale, 17, 15, 13, 2, true, false);
  const auto beforeStats = estimator.statistics(key);
  const auto beforeCapacity = estimator.capacity(key, scale);
  const auto beforePeak = estimator.peakCapacity(key);
  const auto beforeExpected = estimator.expected(key, scale);

  rig.loadRoad();
  rig.armFailure(AllocationFailure::BadAlloc, [&] {
    return estimator.statistics(key).samples > beforeStats.samples;
  });
  const auto dropped = rig.tracker.run(rig.frame, rig.traits);
  rig.disarmFailure();
  BOOST_REQUIRE(dropped.outcome == TrackingOutcome::RecoverableDropped);
  BOOST_REQUIRE_GT(rig.failureCount(), 0);
  rig.assertReset();

  const auto rolledBack = estimator.statistics(key);
  BOOST_TEST(rolledBack.requested == beforeStats.requested);
  BOOST_TEST(rolledBack.granted == beforeStats.granted);
  BOOST_TEST(rolledBack.emitted == beforeStats.emitted);
  BOOST_TEST(rolledBack.spilled == beforeStats.spilled);
  BOOST_TEST(rolledBack.samples == beforeStats.samples);
  BOOST_TEST(rolledBack.overflowEvents == beforeStats.overflowEvents);
  BOOST_TEST(estimator.capacity(key, scale) == beforeCapacity);
  BOOST_TEST(estimator.peakCapacity(key) == beforePeak);
  BOOST_TEST(estimator.expected(key, scale) == beforeExpected);

  // A successful event on the same Tracker/TimeFrame must be able to start a
  // new transaction and retain the update made at this same production site.
  rig.loadRoad();
  TrackingResult succeeded{TrackingOutcome::Structural, std::numeric_limits<float>::quiet_NaN()};
  BOOST_CHECK_NO_THROW(succeeded = rig.tracker.run(rig.frame, rig.traits));
  BOOST_REQUIRE(succeeded.outcome == TrackingOutcome::Success);
  const auto committed = estimator.statistics(key);
  BOOST_TEST(committed.samples > beforeStats.samples);
  BOOST_TEST(committed.requested > beforeStats.requested);
}

// --- Unclassified std::exception: always structural, never a sentinel ----
//
// A plain std::runtime_error (or any std::exception that is neither
// TraversalException, BoundedMemoryResource::MemoryLimitExceeded, nor
// std::bad_alloc) must always rethrow and never be silently converted into
// a dropped-TF result, regardless of DropTFUponFailure.

BOOST_AUTO_TEST_CASE(UnclassifiedExceptionAlwaysRethrowsAndWipesRegardlessOfFlag)
{
  for (const bool dropFlag : {false, true}) {
    ensureTrivialMagneticFieldIsSet();
    MftFailureRig rig{dropFlag};
    rig.configure();
    rig.loadRoad();
    BOOST_REQUIRE(rig.frame.getTotalMeasurements() > 0u);

    rig.armFailure(AllocationFailure::UnclassifiedRuntimeError);
    BOOST_CHECK_THROW(rig.tracker.run(rig.frame, rig.traits), std::runtime_error);
    rig.disarmFailure();

    BOOST_CHECK_GT(rig.failureCount(), 0);
    rig.assertReset();
  }
}

BOOST_AUTO_TEST_CASE(LaterIterationFailureWipesEveryIterationWorkspace)
{
  ensureTrivialMagneticFieldIsSet();
  MftFailureRig rig{/*dropTFUponFailure=*/true};
  rig.configure(/*iterations=*/2);
  rig.loadRoad();

  const auto secondIterationKey = CapacityEstimator::makeKey(SlabSite::Tracklets, 1, 0, EdgeId{0});
  const auto secondIterationSamples = rig.frame.getCapacityEstimator().statistics(secondIterationKey).samples;
  rig.armFailure(AllocationFailure::UnclassifiedRuntimeError, [&] {
    return rig.frame.getCapacityEstimator().statistics(secondIterationKey).samples > secondIterationSamples;
  });
  BOOST_CHECK_THROW(rig.tracker.run(rig.frame, rig.traits), std::runtime_error);
  rig.disarmFailure();

  // Failure is armed only after the second iteration's first tracklet update,
  // so the first iteration completed before the injected exception. No
  // iteration workspace may remain selectable by a later adapter pass.
  BOOST_CHECK_GT(rig.failureCount(), 0);
  rig.assertReset();
}

// --- Index-table configuration failures: structural, always rethrow -------
//
// Both new TraversalFailureReason values (InvalidIndexTableConfiguration,
// IndexTableConfigurationMismatch; TrackerTraits.cxx::initialiseTimeFrame())
// are TraversalException, the same structural-failure category the removed
// StaleLayout test above used to cover -- so they must follow the identical
// always-rethrow-and-wipe contract, regardless of DropTFUponFailure.

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

// --- Direct outcome classification ----------------------------------------
//
// TrackingOutcome::Structural is part of this type's vocabulary for a future
// caller that catches Tracker::run()'s propagated exception itself -- run()
// never constructs it via a
// normal return, since every structural/unclassified/non-dropped-recoverable
// failure keeps the exact "retain exceptions" contract already proven above
// (UnclassifiedExceptionAlwaysRethrowsAndWipesRegardlessOfFlag,
// InvalidIndexTableConfigurationAlwaysRethrowsAndWipesRegardlessOfFlag,
// BadAllocNotDroppedRethrowsButStillWipesFirst,
// RecoverableFailureNotDroppedRethrowsButStillWipesFirst): those tests *are*
// this outcome's structural-failure classification evidence, expressed the
// only way it is currently observable (a thrown exception, never a returned
// value). This test only proves the three values the type actually defines
// are distinct and that TrackingResult's fields carry what each documented
// path above already relies on.
BOOST_AUTO_TEST_CASE(TrackingOutcomeValuesAreDistinct)
{
  BOOST_CHECK(TrackingOutcome::Success != TrackingOutcome::RecoverableDropped);
  BOOST_CHECK(TrackingOutcome::Success != TrackingOutcome::Structural);
  BOOST_CHECK(TrackingOutcome::RecoverableDropped != TrackingOutcome::Structural);

  constexpr TrackingResult defaulted{};
  BOOST_CHECK(defaulted.outcome == TrackingOutcome::Success);
  BOOST_CHECK_EQUAL(defaulted.elapsedMs, 0.f);
}

// --- No stale TimeFrame/GenericTrack/sidecar state survives -----------------
//
// Both non-success return paths from Tracker::run() (structural-rethrow
// and recoverable-dropped) must leave the shared TimeFrame's GenericTrack
// storage and the tracker's adopted compatibility sidecar exactly as empty
// as a freshly wiped/cleared TimeFrame would -- not merely the normalized
// frame and legacy tracks storage the tests above already check.

BOOST_AUTO_TEST_CASE(RecoverableDroppedLeavesNoStaleGenericTrackOrSidecarState)
{
  Rig rig{/*dropTFUponFailure=*/true};
  rig.establishValidLayout();
  rig.loadSource(makeFixture());
  rig.stageStaleState();

  rig.forceMemoryLimitAtCurrentUsage();
  const auto result = rig.tracker.run(rig.frame, rig.traits);
  rig.resetPublication();

  BOOST_CHECK(result.outcome == TrackingOutcome::RecoverableDropped);
  BOOST_CHECK(rig.frame.getGenericTracks().empty());
  BOOST_CHECK(rig.frame.getTrackClusterIndices().empty());
  BOOST_CHECK_EQUAL(rig.sidecar.pendingSize(), 0u);
}

BOOST_AUTO_TEST_CASE(StructuralFailureLeavesNoStaleGenericTrackOrSidecarState)
{
  for (const bool dropFlag : {false, true}) {
    ensureTrivialMagneticFieldIsSet();
    MftFailureRig rig{dropFlag};
    rig.configure();
    rig.loadRoad();
    rig.stageStaleState();

    rig.armFailure(AllocationFailure::UnclassifiedRuntimeError);
    BOOST_CHECK_THROW(rig.tracker.run(rig.frame, rig.traits), std::runtime_error);
    rig.disarmFailure();
    rig.resetPublication();

    BOOST_CHECK_GT(rig.failureCount(), 0);
    rig.assertReset();
    BOOST_CHECK_EQUAL(rig.sidecar.pendingSize(), 0u);
  }
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
