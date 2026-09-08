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

// Orchestration coverage for TrackerTraits<NLayers>::computeLayerCells after
// its detector-family branch was replaced by a one-shot outer dispatch to
// cell-seed leaves (Architecture.md Sec 10/10.1). cell-seed leaves's
// numerical parity with the legacy inline formulas is already proven by
// testTrackletFinding.cxx; this file does not re-derive or
// duplicate that formula. It proves instead that the real public
// computeLayerCells() entry point:
//  - resolves the three clusters for a candidate in strict
//    {inner, middle, outer} order and stores the corresponding linearized
//    triplet factor without prematurely constructing a track state;
//  - exercises cylinder and disk cells through the same public orchestration
//    entry point, with coordinate differences confined to cell-seed leaves;
//  - leaves cellIndex indexing, the LUT, MC-label construction, and
//    one-pass/two-pass ordering untouched;
//  - fails closed (TraversalException::InvalidTraversalSchedule) through the
//    existing public API alone, with no test-only seam into private
//    traversal-cache state.

#define BOOST_TEST_MODULE ITSMFT ComputeLayerCells orchestration
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <oneapi/tbb/task_arena.h>

#include "CommonConstants/MathConstants.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "ITSMFTTracking/Configuration.h"
#include "ITSMFTTracking/ClusterDecoding.h"
#include "ITSMFTTracking/IOUtils.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"
#include "ITSMFTTracking/detail/TimeFrameScratch.h"
#include "ITSMFTTracking/TimeFrame.h"
#include "ITSMFTTracking/Tracker.h"
#include "ITSMFTTracking/TrackerTraits.h"
#include "ITSMFTTracking/TrackingConfigParam.h"
#include "ITSMFTTracking/TripletFitting.h"
#include "ITSMFTTracking/Constants.h"
#include "MFTTracking/Constants.h"

#include "TraversalTestSupport.h"

#include "TrackingParameterTestSupport.h"

using o2::itsmft::tracking::test::ReferenceTrackingParameters;
using namespace o2::itsmft;
using namespace o2::itsmft::tracking;

namespace
{

constexpr float Bz = 0.5f;

// Preflight-only fixtures (Rig::establishLayout()) load zero real clusters --
// this decoder's decode() is never actually invoked there. It exists only to
// satisfy loadNormalizedSource()'s interface, mirroring
// testTrackerFailureContract.cxx's LegacyLikeDecoder.
class NeverDecodedDecoder final : public ClusterDecoder
{
 public:
  explicit NeverDecodedDecoder(o2::detectors::DetID::ID detector) : mDetector(detector) {}

  o2::itsmft::tracking::ClusterDecodeResult decode(
    const CompClusterExt&, BoundedPatternCursor&, const TopologyDictionary*,
    uint32_t, bool) const override
  {
    return {};
  }

 private:
  o2::detectors::DetID::ID mDetector;
};

// Stage-B normalized-CA-measurements slice: computeLayerCells() now reads
// the TimeFrame's source-indexed SurfaceMeasurements. Candidate fixtures
// therefore load their three clusters through the real loadNormalizedSource()
// path -- backfilling both the normalized frame and every legacy
// compatibility structure (unsorted clusters, TrackingFrameInfo, external
// indices, ROF boundaries) together, in lockstep -- rather than poking legacy
// structures directly. This decoder returns exactly the caller-supplied
// SurfaceMeasurement for a given detector-local layer (encoded as the
// synthetic CompClusterExt's chipID/sensorID) as decoded geometry facts.
class FixedMeasurementDecoder final : public ClusterDecoder
{
 public:
  struct MeasurementPair {
    DecodedCluster decoded{};
  };

  FixedMeasurementDecoder(o2::detectors::DetID::ID detector, SurfaceKind kind) : mDetector(detector), mKind(kind) {}

  void setMeasurement(int layer, const MeasurementPair& measurement) { mByLayer[layer] = measurement; }

  o2::itsmft::tracking::ClusterDecodeResult decode(
    const CompClusterExt& cluster,
    BoundedPatternCursor&,
    const TopologyDictionary*,
    uint32_t,
    bool) const override
  {
    o2::itsmft::tracking::ClusterDecodeResult result;
    const int layer = cluster.getSensorID();
    const auto it = mByLayer.find(layer);
    BOOST_REQUIRE(it != mByLayer.end());
    result.decoded = it->second.decoded;
    result.decoded.layer = layer;
    return result;
  }

 private:
  o2::detectors::DetID::ID mDetector;
  SurfaceKind mKind;
  std::map<int, MeasurementPair> mByLayer;
};

const TopologyDictionary& dict()
{
  static const TopologyDictionary d;
  return d;
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

// The test-only reference material values become the authoritative catalog
// material before running computeLayerCells(). Production has no shadow vector.
std::vector<SurfaceDescriptor> makeCatalog(uint16_t nLayers, o2::detectors::DetID::ID det,
                                           gsl::span<const SurfaceKind> kinds, gsl::span<const float> layerxX0)
{
  std::vector<SurfaceDescriptor> surfaces;
  surfaces.reserve(nLayers);
  for (uint16_t i = 0; i < nLayers; ++i) {
    const auto kind = kinds[i];
    surfaces.push_back(SurfaceDescriptor{i, static_cast<uint8_t>(det), kind});
    surfaces.back().chartRange = kind == SurfaceKind::Disk ? SurfaceChartRange{0.1f, 20.f} : SurfaceChartRange{-20.f, 20.f};
    surfaces.back().referenceCoordinate = kind == SurfaceKind::Cylinder
                                            ? 3.f + static_cast<float>(i)
                                            : -0.4f - 0.2f * static_cast<float>(i);
    const float xOverX0 = layerxX0[i];
    surfaces.back().material.xOverX0 = xOverX0;
    surfaces.back().material.arealDensityGPerCm2 = xOverX0 * o2::its::constants::Radl * o2::its::constants::Rho;
  }
  return surfaces;
}

// Same construction as testTrackletFinding.cxx's helpers -- plain
// input-struct builders, not a reimplementation of any fit formula.
GlobalMeasurement makeGlobalCluster(float x, float y, float z, int id = 0)
{
  GlobalMeasurement measurement{};
  measurement.position = {x, y, z};
  measurement.radius = std::hypot(x, y);
  measurement.phi = std::atan2(y, x);
  measurement.clusterId = static_cast<uint32_t>(id);
  return measurement;
}

struct TestLocalMeasurement {
  float xTrackingFrame{0.f};
  float alphaTrackingFrame{0.f};
  std::array<float, 2> positionTrackingFrame{};
  std::array<float, 3> covarianceTrackingFrame{};
};

TestLocalMeasurement makeBarrelHit(float xTF, float alpha, float y, float z, float sigma2Y = 1.e-4f, float sigma2Z = 1.e-4f)
{
  return {xTF, alpha, {y, z}, {sigma2Y, 0.f, sigma2Z}};
}

TestLocalMeasurement makeDiskHit(float z, float x, float y, float sigma2X = 1.e-2f, float sigma2Y = 1.e-2f)
{
  return {z, 0.f, {x, y}, {sigma2X, 0.f, sigma2Y}};
}

// Test-local field-mapping helpers (not a production API), matching the same
// Cylinder/Disk field mapping used by the production migration and by
// testCellFinding.cxx: the single SurfaceMeasurement now
// standing in for the retired {Cluster, TrackingFrameInfo} pair at each
// candidate position.
FixedMeasurementDecoder::MeasurementPair barrelMeasurementFor(const GlobalMeasurement& cluster, const TestLocalMeasurement& hit)
{
  FixedMeasurementDecoder::MeasurementPair measurement{};
  measurement.decoded.global = {cluster.x, cluster.y, cluster.z};
  measurement.decoded.cylinderFrame = {hit.xTrackingFrame, hit.positionTrackingFrame[0],
                                       hit.positionTrackingFrame[1], hit.alphaTrackingFrame};
  measurement.decoded.rowColumnCovariance = {hit.covarianceTrackingFrame[0],
                                             hit.covarianceTrackingFrame[1],
                                             hit.covarianceTrackingFrame[2]};
  return measurement;
}

FixedMeasurementDecoder::MeasurementPair diskMeasurementFor(const GlobalMeasurement& cluster, const TestLocalMeasurement& hit)
{
  FixedMeasurementDecoder::MeasurementPair measurement{};
  measurement.decoded.global = {cluster.x, cluster.y, cluster.z};
  measurement.decoded.rowColumnCovariance = {hit.covarianceTrackingFrame[0], 0.f,
                                             hit.covarianceTrackingFrame[2]};
  return measurement;
}

void checkTripletFitFactorEqual(const TripletFitFactor& lhs, const TripletFitFactor& rhs)
{
  BOOST_CHECK_EQUAL(lhs.psi.theta, rhs.psi.theta);
  BOOST_CHECK_EQUAL(lhs.psi.phi, rhs.psi.phi);
  BOOST_CHECK_EQUAL(lhs.rho.theta, rhs.rho.theta);
  BOOST_CHECK_EQUAL(lhs.rho.phi, rhs.rho.phi);
  for (int hit = 0; hit < 3; ++hit) {
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      BOOST_CHECK_EQUAL(lhs.h[hit].theta[coordinate], rhs.h[hit].theta[coordinate]);
      BOOST_CHECK_EQUAL(lhs.h[hit].phi[coordinate], rhs.h[hit].phi[coordinate]);
    }
  }
}

void checkTrackSeedContents(const TrackSeed& trackSeed, const CellSeed& cell,
                            SurfaceKind expectedKind)
{
  BOOST_CHECK_EQUAL(trackSeed.getHitLayerMask().value(), cell.getHitLayerMask().value());
  for (int slot = 0; slot < 3; ++slot) {
    const auto reference = cell.getClusterReference(slot);
    BOOST_CHECK_EQUAL(trackSeed.getCluster(reference.surfacePosition), reference.clusterIndex);
  }
  BOOST_CHECK_EQUAL(trackSeed.getLevel(), cell.getLevel());
  BOOST_CHECK_EQUAL(trackSeed.getFirstTrackletIndex(), cell.getFirstTrackletIndex());
  BOOST_CHECK_EQUAL(trackSeed.getSecondTrackletIndex(), cell.getSecondTrackletIndex());
  BOOST_CHECK_EQUAL(trackSeed.getTimeStamp().getTimeStamp(), cell.getTimeStamp().getTimeStamp());
  BOOST_CHECK_EQUAL(trackSeed.getTimeStamp().getTimeStampError(), cell.getTimeStamp().getTimeStampError());
  BOOST_CHECK(trackSeed.state().kind == expectedKind);
  BOOST_CHECK(std::isfinite(trackSeed.getChi2()));
  for (const float parameter : trackSeed.state().parameters) {
    BOOST_CHECK(std::isfinite(parameter));
  }
  for (const float covariance : trackSeed.state().covariance) {
    BOOST_CHECK(std::isfinite(covariance));
  }
}

void checkTrackSeedsEqual(const TrackSeed& lhs, const TrackSeed& rhs)
{
  BOOST_CHECK_EQUAL(lhs.getHitLayerMask().value(), rhs.getHitLayerMask().value());
  BOOST_CHECK_EQUAL(lhs.getChi2(), rhs.getChi2());
  BOOST_CHECK_EQUAL(lhs.getLevel(), rhs.getLevel());
  BOOST_CHECK_EQUAL(lhs.getFirstTrackletIndex(), rhs.getFirstTrackletIndex());
  BOOST_CHECK_EQUAL(lhs.getSecondTrackletIndex(), rhs.getSecondTrackletIndex());
  BOOST_CHECK_EQUAL(lhs.getTimeStamp().getTimeStamp(), rhs.getTimeStamp().getTimeStamp());
  BOOST_CHECK_EQUAL(lhs.getTimeStamp().getTimeStampError(), rhs.getTimeStamp().getTimeStampError());
  for (int position = 0; position < TrackSeed::MaxSurfaces; ++position) {
    BOOST_CHECK_EQUAL(lhs.getCluster(position), rhs.getCluster(position));
  }
  for (int parameter = 0; parameter < 5; ++parameter) {
    BOOST_CHECK_EQUAL(lhs.state().parameters[parameter], rhs.state().parameters[parameter]);
  }
  for (int covariance = 0; covariance < 15; ++covariance) {
    BOOST_CHECK_EQUAL(lhs.state().covariance[covariance], rhs.state().covariance[covariance]);
  }
  BOOST_CHECK_EQUAL(lhs.state().referenceCoordinate, rhs.state().referenceCoordinate);
  BOOST_CHECK_EQUAL(lhs.state().alpha, rhs.state().alpha);
  BOOST_CHECK(lhs.state().kind == rhs.state().kind);
  BOOST_CHECK_EQUAL(lhs.state().flags, rhs.state().flags);
  BOOST_CHECK_EQUAL(lhs.state().absCharge, rhs.state().absCharge);
  BOOST_CHECK(lhs.state().pid == rhs.state().pid);
}

void checkTrackSeedMaterialization(TrackerTraits& traits, IterationContext& view,
                                   int cellPathId, const CellSeed& cell,
                                   SurfaceKind expectedKind)
{
  TrackSeed trackSeed{};
  OperationFailureReason reason{};
  BOOST_REQUIRE(TrackerTestAccess::buildTrackSeed(
    traits, view, cellPathId, cell, trackSeed, reason));
  checkTrackSeedContents(trackSeed, cell, expectedKind);
}

// Minimal wiring TrackerTraits<NLayers>::computeLayerCells() needs: a real
// layout/topology (so initialiseTimeFrame() genuinely binds
// the edge/cell schedule and tracking parameters
// -- computeLayerCells()'s own private caches, never poked directly), and a
// validly-sized-but-empty normalized load (proven pattern from
// testTrackerFailureContract.cxx: TimeFrame::initialise() unconditionally
// reads mROFramesClusters sizes, which only loadNormalizedSource() sets up
// safely, even for zero clusters).
struct RigFrameStorage {
  RigFrameStorage() : pool(std::make_shared<BoundedMemoryResource>()) { frame.setMemoryPool(pool); }

  std::shared_ptr<BoundedMemoryResource> pool;
  TimeFrame frame;
};

template <int NLayers>
struct Rig : RigFrameStorage {

  Rig(o2::detectors::DetID::ID det, SurfaceKind kind, int nThreads = 1)
    : params(1),
      mDet(det),
      mKinds(NLayers, kind)
  {
    resetDetectorDefaults(params[0], det);
    // This file bypasses computeLayerTracklets()'s phi/z/index-table cuts
    // entirely (candidates are injected directly, see
    // injectCandidateTracklets() below): clearing RebuildClusterLUT keeps
    // TimeFrame::initialise() from also exercising prepareClusters()'s
    // index-table row/col binning and ROF-mask lookup on the synthetic
    // candidate positions/ROF this file uses -- that out-of-scope subsystem
    // is not configured for this file's candidates (in particular, no
    // multiplicity/UPC ROF mask is ever loaded, so its default view is
    // never a valid one to index once real clusters are present, unlike
    // when this file loaded zero real clusters).
    params[0].PassFlags.reset(IterationStep::RebuildClusterLUT);
    traits.setNThreads(nThreads, arena);
    frame.setBz(Bz);
  }

  // Establishes the catalog/layout and loads a (zero-cluster) normalized
  // source. Deliberately not run by the constructor: it builds the catalog's
  // nominal material from the *current* params[0].LayerxX0, so callers must
  // finish any test reference material override before establishing the layout.
  void establishLayout()
  {
    catalog = makeCatalog(static_cast<uint16_t>(NLayers), mDet, gsl::span<const SurfaceKind>{mKinds}, gsl::span<const float>(params[0].LayerxX0));
    const auto orderedSurfaces = identitySurfaces(static_cast<uint16_t>(NLayers));
    const SurfaceCatalogView catalogView{catalog.data(), static_cast<uint32_t>(catalog.size())};
    TrackerInitialization configuration;
    configuration.catalog = catalogView;
    configuration.memoryPool = pool;
    configuration.layout = makeDetectorLayout(holeLayers);
    configuration.plan = o2::itsmft::tracking::test::makeTrackingPlan(params[0]);
    BOOST_REQUIRE(tracker.initialize(frame, configuration).ok());
    tf = &frame.getScratch();
    const auto& layout = frame.getLayout();

    NeverDecodedDecoder decoder{mDet};
    const o2::InteractionRecord origin{50, 5};
    const ROFTimingConfig timing{40, 0, 0, 0};
    const std::vector<CompClusterExt> noClusters;
    const std::vector<unsigned char> noPatterns;
    const std::vector<ROFRecord> noRofs;
    const auto loadResult = loadTimeFrameSource(frame, decoder, origin, timing, noClusters, noPatterns, noRofs, &dict(), nullptr, mDet,
                                                gsl::span<const LayerId>{orderedSurfaces}, layout.getSurfaceCatalog());
    BOOST_REQUIRE(loadResult.ok());
  }

  o2::detectors::DetID::ID detector() const noexcept { return mDet; }
  SurfaceKind kind() const noexcept { return mKinds.front(); }
  SurfaceKind kind(int layer) const noexcept { return mKinds[layer]; }
  void setSurfaceKind(int layer, SurfaceKind kind) { mKinds[layer] = kind; }

  std::vector<ReferenceTrackingParameters> params;
  LayerMask holeLayers{};
  // Gate 4 B3.1: `frame` declared before `tf` so it is constructed first and
  // destroyed last (see TimeFrameScratch's own lifetime-contract doc).
  TimeFrameScratch* tf{nullptr};
  Tracker tracker;
  std::array<gsl::span<const GlobalMeasurement>, MaxLayoutSurfaces> measurementSpans;
  TrackerTraits traits;
  std::shared_ptr<tbb::task_arena> arena;
  // The catalog must outlive the immutable layout and all event-local views.
  std::vector<SurfaceDescriptor> catalog;

 private:
  o2::detectors::DetID::ID mDet;
  std::vector<SurfaceKind> mKinds;
};

template <int NLayers>
IterationContext prepare(Rig<NLayers>& rig)
{
  return TrackerTestAccess::prepare(rig.tracker, rig.frame, 0, rig.measurementSpans);
}

template <int NLayers>
TraversalTopologyView topologyView(const Rig<NLayers>& rig)
{
  return rig.tracker.getIterationConfigurations()[0].getTopologyView(rig.frame.getLayout().getSurfaceCatalog());
}

// Loads exactly the three supplied {cluster, hit} candidates at legacy
// layers {0, 1, 2} (every test in this file locates its candidate cell via
// findCellIndex(topology, 0, 1, 2), so the layer mapping is always this
// identity triple) through the real loadNormalizedSource() path, via
// FixedMeasurementDecoder -- so the normalized frame and every legacy
// compatibility structure are populated together, in lockstep, exactly as
// TrackerTraits::initialiseTimeFrame()'s one-time normalized-measurement
// binding requires. Must be called after Rig::establishLayout() (which needs
// the catalog/topology first) and before TrackerTraits::initialiseTimeFrame()
// (which validates the normalized frame against the legacy structures this
// call also populates).
template <int NLayers>
void loadCandidateClusters(Rig<NLayers>& rig,
                           const std::array<GlobalMeasurement, 3>& clusters,
                           const std::array<TestLocalMeasurement, 3>& hits)
{
  FixedMeasurementDecoder decoder{rig.detector(), rig.kind()};
  std::vector<CompClusterExt> compClusters;
  compClusters.reserve(3);
  for (int layer = 0; layer < 3; ++layer) {
    compClusters.emplace_back(0, 0, CompCluster::InvalidPatternID, static_cast<uint16_t>(layer));
    const auto measurement = rig.kind(layer) == SurfaceKind::Disk
                               ? diskMeasurementFor(clusters[layer], hits[layer])
                               : barrelMeasurementFor(clusters[layer], hits[layer]);
    decoder.setMeasurement(layer, measurement);
  }
  const std::vector<unsigned char> noPatterns;
  const std::vector<ROFRecord> rofs{ROFRecord{{0, 0}, 0, 0, 3}};
  const o2::InteractionRecord origin{50, 5};
  const ROFTimingConfig timing{40, 0, 0, 0};
  const auto layerMapping = identitySurfaces(static_cast<uint16_t>(NLayers));
  const auto result = loadTimeFrameSource(rig.frame, decoder, origin, timing, compClusters, noPatterns, rofs, &dict(), nullptr, rig.detector(),
                                          gsl::span<const LayerId>{layerMapping}, rig.frame.getLayout().getSurfaceCatalog());
  BOOST_REQUIRE(result.ok());
}

// Finds the cellIndex whose two edges span exactly
// inner->middle->outer, without assuming any particular enumeration order
// out of the sparse topology's builder enumeration.
template <typename TopologyView>
int findCellIndex(const TopologyView& topology, int inner, int middle, int outer)
{
  for (int i = 0; i < topology.nPaths; ++i) {
    const auto& cell = topology.getPath(CellPathId{static_cast<uint16_t>(i)});
    const auto& first = topology.getEdge(cell.first);
    const auto& second = topology.getEdge(cell.second);
    if (first.from.value() == inner && first.to.value() == middle && second.from.value() == middle && second.to.value() == outer) {
      return i;
    }
  }
  return -1;
}

Tracklet candidateTracklet(const GlobalMeasurement& first, const GlobalMeasurement& second,
                           const o2::its::TimeEstBC& timestamp)
{
  const float deltaR = first.radius - second.radius;
  const float deltaZ = first.z - second.z;
  const float tanLambda = deltaR * deltaR > o2::constants::math::Almost0
                            ? deltaZ / deltaR
                            : std::copysign(o2::constants::math::VeryBig, deltaZ);
  const float phi = std::atan2(first.y - second.y, first.x - second.x);
  return {0, 0, tanLambda, phi, timestamp};
}

// Bypasses the real (untouched, out-of-scope-for-this-change)
// computeLayerTracklets() phi/z/index-table cuts entirely. The real loader
// has already installed the authoritative compact globals and fitting
// measurements; this helper only injects one tracklet per edge of cellIndex,
// wired so
// computeLayerCellsForKind<Tag>'s tracklet-pairing loop finds exactly one
// candidate pair.
template <int NLayers>
void injectCandidateTracklets(Rig<NLayers>& rig, int cellIndex, const std::array<GlobalMeasurement, 3>& clusters)
{
  const auto topology = topologyView(rig);
  const auto& cell = topology.getPath(CellPathId{static_cast<uint16_t>(cellIndex)});
  const auto& first = topology.getEdge(cell.first);
  const auto& second = topology.getEdge(cell.second);
  const int layers[3] = {first.from.value(), first.to.value(), second.to.value()};

  for (int i = 0; i < 3; ++i) {
    BOOST_REQUIRE_EQUAL(rig.frame.getClusters()[layers[i]].size(), 1u);
    BOOST_CHECK_EQUAL(rig.frame.getClusters()[layers[i]][0].x, clusters[i].x);
    BOOST_CHECK_EQUAL(rig.frame.getClusters()[layers[i]][0].y, clusters[i].y);
    BOOST_CHECK_EQUAL(rig.frame.getClusters()[layers[i]][0].z, clusters[i].z);
  }

  const o2::its::TimeEstBC ts{static_cast<uint32_t>(0), static_cast<uint16_t>(1)};
  rig.tf->getTracklets()[cell.first.value()].push_back(candidateTracklet(clusters[0], clusters[1], ts));
  rig.tf->getTracklets()[cell.second.value()].push_back(candidateTracklet(clusters[1], clusters[2], ts));

  auto& secondLUT = rig.tf->getTrackletsLookupTable()[cell.second.value()];
  secondLUT.resize(2);
  secondLUT[0] = 0;
  secondLUT[1] = 1;
}

// Gate 4 Slice 0b additions below: multi-cell parity coverage for the
// migrated computeLayerCells()/computeLayerCellsForKind(), extending this
// file's existing single-cell (always layers {0,1,2}) machinery to an
// arbitrary ordered set of N>=3 layers so several simultaneously-populated
// cells (sharing edges between adjacent triples) can be checked in one
// run.

// Same technique as loadCandidateClusters() (real loadNormalizedSource()
// path via FixedMeasurementDecoder), generalized to N candidate layers
// instead of the fixed {0,1,2} triple.
template <int NLayers, size_t N>
void loadCandidateClustersAtLayers(Rig<NLayers>& rig,
                                   const std::array<int, N>& layers,
                                   const std::array<GlobalMeasurement, N>& clusters,
                                   const std::array<TestLocalMeasurement, N>& hits)
{
  FixedMeasurementDecoder decoder{rig.detector(), rig.kind()};
  std::vector<CompClusterExt> compClusters;
  compClusters.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    compClusters.emplace_back(0, 0, CompCluster::InvalidPatternID, static_cast<uint16_t>(layers[i]));
    const auto measurement = rig.kind(layers[i]) == SurfaceKind::Disk
                               ? diskMeasurementFor(clusters[i], hits[i])
                               : barrelMeasurementFor(clusters[i], hits[i]);
    decoder.setMeasurement(layers[i], measurement);
  }
  const std::vector<unsigned char> noPatterns;
  const std::vector<ROFRecord> rofs{ROFRecord{{0, 0}, 0, 0, static_cast<int>(N)}};
  const o2::InteractionRecord origin{50, 5};
  const ROFTimingConfig timing{40, 0, 0, 0};
  const auto layerMapping = identitySurfaces(static_cast<uint16_t>(NLayers));
  const auto result = loadTimeFrameSource(rig.frame, decoder, origin, timing, compClusters, noPatterns, rofs, &dict(), nullptr, rig.detector(),
                                          gsl::span<const LayerId>{layerMapping}, rig.frame.getLayout().getSurfaceCatalog());
  BOOST_REQUIRE(result.ok());
}

// Finds the edgeId spanning exactly from->to, mirroring
// findCellIndex()'s linear-search style over the legacy view.
template <typename TopologyView>
int findEdgeId(const TopologyView& topology, int from, int to)
{
  for (int i = 0; i < topology.nEdges; ++i) {
    const auto& t = topology.getEdge(EdgeId{static_cast<uint16_t>(i)});
    if (t.from.value() == from && t.to.value() == to) {
      return i;
    }
  }
  return -1;
}

// Generalizes injectCandidateTracklets() to an ordered chain of N>=3 layers:
// writes each physical layer's single cluster exactly once, then touches
// each of the N-1 adjacent-pair edges exactly once (one synthetic
// tracklet + one LUT {0,1}), regardless of how many downstream cells in the
// chain share that edge. Naively calling the single-cell
// injectCandidateTracklets() once per overlapping cell would instead
// double-write any shared edge (extra duplicate tracklet, and a LUT
// left however the last call set it) and silently clobber a shared physical
// layer's cluster across calls -- this helper touches every physical layer
// and every edge exactly once, by construction.
template <int NLayers, size_t N>
void injectChainCandidateTracklets(Rig<NLayers>& rig, const std::array<int, N>& layers, const std::array<GlobalMeasurement, N>& clusters)
{
  static_assert(N >= 3, "a chain needs at least 3 layers to form one cell");
  const auto topology = topologyView(rig);
  for (size_t i = 0; i < N; ++i) {
    BOOST_REQUIRE_EQUAL(rig.frame.getClusters()[layers[i]].size(), 1u);
    BOOST_CHECK_EQUAL(rig.frame.getClusters()[layers[i]][0].x, clusters[i].x);
    BOOST_CHECK_EQUAL(rig.frame.getClusters()[layers[i]][0].y, clusters[i].y);
    BOOST_CHECK_EQUAL(rig.frame.getClusters()[layers[i]][0].z, clusters[i].z);
  }

  const o2::its::TimeEstBC ts{static_cast<uint32_t>(0), static_cast<uint16_t>(1)};
  for (size_t i = 0; i + 1 < N; ++i) {
    const int edgeId = findEdgeId(topology, layers[i], layers[i + 1]);
    BOOST_REQUIRE_GE(edgeId, 0);
    rig.tf->getTracklets()[edgeId].push_back(candidateTracklet(clusters[i], clusters[i + 1], ts));
    auto& lut = rig.tf->getTrackletsLookupTable()[edgeId];
    lut.resize(2);
    lut[0] = 0;
    lut[1] = 1;
  }
}

std::array<TestLocalMeasurement, 3> makeLocalMeasurements(
  const std::array<SurfaceKind, 3>& kinds,
  const std::array<GlobalMeasurement, 3>& clusters)
{
  std::array<TestLocalMeasurement, 3> hits{};
  for (int layer = 0; layer < 3; ++layer) {
    const auto& position = clusters[layer].position;
    hits[layer] = kinds[layer] == SurfaceKind::Disk
                    ? makeDiskHit(position.z, position.x, position.y)
                    : makeBarrelHit(position.x, 0.f, position.y, position.z);
  }
  return hits;
}

template <int NLayers>
void checkDirectTrackSeedConstruction(const std::array<SurfaceKind, 3>& kinds,
                                      const std::array<GlobalMeasurement, 3>& clusters)
{
  Rig<NLayers> rig{o2::detectors::DetID::ITS, kinds[0]};
  rig.params[0].MaxChi2ClusterAttachment = 1.e6f;
  for (int layer = 0; layer < 3; ++layer) {
    rig.setSurfaceKind(layer, kinds[layer]);
    rig.params[0].LayerxX0[layer] = 0.f;
  }
  rig.establishLayout();
  loadCandidateClusters(rig, clusters, makeLocalMeasurements(kinds, clusters));

  auto view = prepare(rig);
  const int cellPathId = findCellIndex(topologyView(rig), 0, 1, 2);
  BOOST_REQUIRE_GE(cellPathId, 0);
  CellSeed cell{0, 0, 0, 0, 17, 23, o2::its::TimeEstBC{111, 9}};
  cell.setLevel(6);

  TrackSeed first{};
  TrackSeed second{};
  OperationFailureReason firstReason{};
  OperationFailureReason secondReason{};
  BOOST_REQUIRE(TrackerTestAccess::buildTrackSeed(
    rig.traits, view, cellPathId, cell, first, firstReason));
  BOOST_REQUIRE(TrackerTestAccess::buildTrackSeed(
    rig.traits, view, cellPathId, cell, second, secondReason));

  checkTrackSeedContents(first, cell, kinds[0]);
  checkTrackSeedsEqual(first, second);
}

constexpr std::array<SurfaceKind, 3> CylinderCylinderCylinder{
  SurfaceKind::Cylinder, SurfaceKind::Cylinder, SurfaceKind::Cylinder};
constexpr std::array<SurfaceKind, 3> DiskDiskDisk{
  SurfaceKind::Disk, SurfaceKind::Disk, SurfaceKind::Disk};
constexpr std::array<SurfaceKind, 3> CylinderDiskCylinder{
  SurfaceKind::Cylinder, SurfaceKind::Disk, SurfaceKind::Cylinder};
constexpr std::array<SurfaceKind, 3> DiskCylinderDisk{
  SurfaceKind::Disk, SurfaceKind::Cylinder, SurfaceKind::Disk};

const std::array<GlobalMeasurement, 3> NominalTrackSeedClusters{
  makeGlobalCluster(3.0f, 0.100f, 0.90f, 0),
  makeGlobalCluster(4.0f, 0.150f, 1.05f, 0),
  makeGlobalCluster(5.0f, 0.201f, 1.25f, 0)};

} // namespace

BOOST_AUTO_TEST_CASE(BuildTrackSeedCylinderCylinderCylinderIsDeterministic)
{
  checkDirectTrackSeedConstruction<ITSNLayers>(CylinderCylinderCylinder, NominalTrackSeedClusters);
}

BOOST_AUTO_TEST_CASE(BuildTrackSeedDiskDiskDiskIsDeterministic)
{
  checkDirectTrackSeedConstruction<ITSNLayers>(DiskDiskDisk, NominalTrackSeedClusters);
}

BOOST_AUTO_TEST_CASE(BuildTrackSeedCylinderDiskCylinderConvertsBackToCylinder)
{
  checkDirectTrackSeedConstruction<ITSNLayers>(CylinderDiskCylinder, NominalTrackSeedClusters);
}

BOOST_AUTO_TEST_CASE(BuildTrackSeedDiskCylinderDiskConvertsBackToDisk)
{
  checkDirectTrackSeedConstruction<ITSNLayers>(DiskCylinderDisk, NominalTrackSeedClusters);
}

BOOST_AUTO_TEST_CASE(BuildTrackSeedDegenerateMixedTripletPreservesDestination)
{
  Rig<ITSNLayers> rig{o2::detectors::DetID::ITS, SurfaceKind::Cylinder};
  rig.params[0].MaxChi2ClusterAttachment = 1.e6f;
  for (int layer = 0; layer < 3; ++layer) {
    rig.setSurfaceKind(layer, CylinderDiskCylinder[layer]);
    rig.params[0].LayerxX0[layer] = 0.f;
  }
  rig.establishLayout();

  const std::array<GlobalMeasurement, 3> degenerateClusters{
    makeGlobalCluster(3.f, 0.1f, 0.9f, 0),
    makeGlobalCluster(3.f, 0.1f, 1.0f, 0),
    makeGlobalCluster(3.f, 0.1f, 1.1f, 0)};
  loadCandidateClusters(rig, degenerateClusters,
                        makeLocalMeasurements(CylinderDiskCylinder, degenerateClusters));
  auto view = prepare(rig);
  const int cellPathId = findCellIndex(topologyView(rig), 0, 1, 2);
  BOOST_REQUIRE_GE(cellPathId, 0);
  CellSeed cell{0, 0, 0, 0, 17, 23, o2::its::TimeEstBC{111, 9}};
  cell.setLevel(6);

  SurfaceTrackState sentinelState{};
  sentinelState.kind = SurfaceKind::Disk;
  sentinelState.referenceCoordinate = -42.f;
  sentinelState.parameters[0] = 13.f;
  TrackSeed destination{cell, sentinelState, 71.f};
  const TrackSeed before = destination;
  OperationFailureReason reason{};
  BOOST_CHECK(!TrackerTestAccess::buildTrackSeed(
    rig.traits, view, cellPathId, cell, destination, reason));
  BOOST_CHECK(reason == OperationFailureReason::SurfaceKindConversionFailure);
  checkTrackSeedsEqual(destination, before);
}

// --- Barrel: real orchestration matches the cell-seed leaves oracle -----

BOOST_AUTO_TEST_CASE(CylinderComputeLayerCellsMatchesBuildCellSeedOracle)
{
  Rig<ITSNLayers> rig{o2::detectors::DetID::ITS, SurfaceKind::Cylinder};
  rig.params[0].MaxChi2ClusterAttachment = 1.e6f;
  rig.params[0].LayerxX0[0] = 0.005f; // inner
  rig.params[0].LayerxX0[1] = 0.005f; // middle
  rig.params[0].LayerxX0[2] = 0.f;    // outer: contractually unused by Cylinder
  rig.establishLayout();

  const std::array<GlobalMeasurement, 3> clusters{makeGlobalCluster(3.0f, 0.100f, 0.9f, 0),
                                                  makeGlobalCluster(4.0f, 0.150f, 1.05f, 0),
                                                  makeGlobalCluster(5.0f, 0.201f, 1.20f, 0)};
  loadCandidateClusters(rig, clusters,
                        {makeBarrelHit(3.f, 0.f, 0.100f, 0.9f),
                         makeBarrelHit(4.f, 0.f, 0.150f, 1.05f),
                         makeBarrelHit(5.f, 0.f, 0.201f, 1.20f)});

  auto view = TrackerTestAccess::prepare(rig.tracker, rig.frame, 0, rig.measurementSpans);

  const auto topology = topologyView(rig);
  const int cellIndex = findCellIndex(topology, 0, 1, 2);
  BOOST_REQUIRE_GE(cellIndex, 0);

  injectCandidateTracklets(rig, cellIndex, clusters);

  // Any other cellIndex keeps its empty-edge early-continue
  // path: cleared once up front, never touched again.
  int othercellIndex = -1;
  for (int i = 0; i < topology.nPaths; ++i) {
    if (i != cellIndex) {
      othercellIndex = i;
      break;
    }
  }
  BOOST_REQUIRE_GE(othercellIndex, 0);

  TrackerTestAccess::computeCells(rig.traits, view);

  BOOST_CHECK(rig.tf->getCells()[othercellIndex].empty());
  BOOST_CHECK(rig.tf->getCellsLookupTable()[othercellIndex].empty());
  BOOST_CHECK(rig.tf->getCellsLabel(othercellIndex).empty());

  BOOST_REQUIRE_EQUAL(rig.tf->getCells()[cellIndex].size(), 1u);
  const auto& producedCell = rig.tf->getCells()[cellIndex][0];
  BOOST_CHECK(producedCell.tripletFactor().isValid());
  for (int slot = 0; slot < 3; ++slot) {
    const auto reference = producedCell.getClusterReference(slot);
    BOOST_CHECK_EQUAL(reference.surfacePosition, slot);
    BOOST_CHECK_EQUAL(reference.clusterIndex, producedCell.getClusters()[slot]);
  }

  BOOST_REQUIRE_EQUAL(rig.tf->getCellsLookupTable()[cellIndex].size(), 2u);
  BOOST_CHECK_EQUAL(rig.tf->getCellsLookupTable()[cellIndex][0], 0);
  BOOST_CHECK_EQUAL(rig.tf->getCellsLookupTable()[cellIndex][1], 1);

  // hasMCinformation() is false (no labels were loaded), so label
  // construction is skipped, exactly as before this change.
  BOOST_CHECK(rig.tf->getCellsLabel(cellIndex).empty());

  // Oracle: independently reconstruct the geometry-only factor from the
  // ordered global measurements. Track-state construction belongs to
  // TrackerTraits::buildTrackSeed(), after the CA has selected a cell.
  const auto layerGlobalMeasurements = gsl::span<const gsl::span<const GlobalMeasurement>>{view.layerGlobalMeasurements};
  const auto& oracleGlobalInner = layerGlobalMeasurements[0][producedCell.getFirstClusterIndex()];
  const auto& oracleGlobalMiddle = layerGlobalMeasurements[1][producedCell.getSecondClusterIndex()];
  const auto& oracleGlobalOuter = layerGlobalMeasurements[2][producedCell.getThirdClusterIndex()];
  const std::array<GlobalMeasurement, 3> measurements{
    oracleGlobalInner, oracleGlobalMiddle, oracleGlobalOuter};
  TripletFitFactor oracleFactor{};
  BOOST_REQUIRE(makeTripletFitFactor(measurements, oracleFactor));
  checkTripletFitFactorEqual(producedCell.tripletFactor(), oracleFactor);
  checkTrackSeedMaterialization(rig.traits, view, cellIndex, producedCell,
                                SurfaceKind::Cylinder);
}

BOOST_AUTO_TEST_CASE(CylinderCellCombinationUsesTrackletMinPtScattering)
{
  auto acceptedCells = [](float trackletMinPt) {
    Rig<ITSNLayers> rig{o2::detectors::DetID::ITS, SurfaceKind::Cylinder};
    rig.params[0].TrackletMinPt = trackletMinPt;
    rig.params[0].MaxChi2ClusterAttachment = 1.e6f;
    rig.params[0].LayerxX0[0] = 0.005f;
    rig.params[0].LayerxX0[1] = 0.01f;
    rig.params[0].LayerxX0[2] = 0.f;
    rig.establishLayout();

    const std::array<GlobalMeasurement, 3> clusters{
      makeGlobalCluster(3.f, 0.100f, 0.9f),
      makeGlobalCluster(4.f, 0.150f, 1.05f),
      makeGlobalCluster(5.f, 0.201f, 1.22f)};
    loadCandidateClusters(rig, clusters,
                          {makeBarrelHit(3.f, 0.f, 0.100f, 0.9f, 1.e-6f, 1.e-6f),
                           makeBarrelHit(4.f, 0.f, 0.150f, 1.05f, 1.e-6f, 1.e-6f),
                           makeBarrelHit(5.f, 0.f, 0.201f, 1.22f, 1.e-6f, 1.e-6f)});

    auto view = prepare(rig);
    const auto topology = topologyView(rig);
    const int cellIndex = findCellIndex(topology, 0, 1, 2);
    BOOST_REQUIRE_GE(cellIndex, 0);
    injectCandidateTracklets(rig, cellIndex, clusters);
    TrackerTestAccess::computeCells(rig.traits, view);
    return rig.tf->getCells()[cellIndex].size();
  };

  BOOST_CHECK_EQUAL(acceptedCells(0.3f), 1u);
  BOOST_CHECK_EQUAL(acceptedCells(1.f), 0u);
}

BOOST_AUTO_TEST_CASE(ForwardCellProjectsScatteringIntoAzimuth)
{
  Rig<MFTNLayers> rig{o2::detectors::DetID::MFT, SurfaceKind::Disk};
  rig.params[0].TrackletMinPt = 0.3f;
  rig.params[0].MaxChi2ClusterAttachment = 1.e6f;
  rig.params[0].LayerxX0[0] = 0.015f;
  rig.params[0].LayerxX0[1] = 0.017f;
  rig.params[0].LayerxX0[2] = 0.02f;
  rig.establishLayout();

  const std::array<GlobalMeasurement, 3> clusters{
    makeGlobalCluster(1.f, 0.f, -0.4f),
    makeGlobalCluster(1.01f, 0.f, -0.6f),
    makeGlobalCluster(1.01995f, 0.000998f, -0.9f)};
  loadCandidateClusters(rig, clusters,
                        {makeDiskHit(-0.4f, 1.f, 0.f),
                         makeDiskHit(-0.6f, 1.01f, 0.f),
                         makeDiskHit(-0.9f, 1.01995f, 0.000998f)});

  auto view = prepare(rig);
  const auto topology = topologyView(rig);
  const int cellIndex = findCellIndex(topology, 0, 1, 2);
  BOOST_REQUIRE_GE(cellIndex, 0);
  injectCandidateTracklets(rig, cellIndex, clusters);
  TrackerTestAccess::computeCells(rig.traits, view);

  BOOST_CHECK_EQUAL(rig.tf->getCells()[cellIndex].size(), 1u);
}

// --- Disk: real orchestration matches the generic cell-seed oracle -------

BOOST_AUTO_TEST_CASE(DiskComputeLayerCellsMatchesBuildCellSeedOracle)
{
  Rig<MFTNLayers> rig{o2::detectors::DetID::MFT, SurfaceKind::Disk};
  rig.params[0].MaxChi2ClusterAttachment = 1.e6f;
  rig.params[0].TrackletMinPt = 0.3f;
  rig.params[0].LayerxX0[0] = 0.015f; // inner
  rig.params[0].LayerxX0[1] = 0.017f; // middle
  rig.params[0].LayerxX0[2] = 0.02f;  // outer
  rig.establishLayout();

  const std::array<GlobalMeasurement, 3> clusters{makeGlobalCluster(1.0f, 0.5f, -0.4f, 0),
                                                  makeGlobalCluster(1.3f, 0.62f, -0.6f, 0),
                                                  makeGlobalCluster(1.7f, 0.78f, -0.9f, 0)};
  loadCandidateClusters(rig, clusters,
                        {makeDiskHit(-0.4f, 1.0f, 0.5f),
                         makeDiskHit(-0.6f, 1.3f, 0.62f),
                         makeDiskHit(-0.9f, 1.7f, 0.78f)});

  auto view = prepare(rig);

  const auto topology = topologyView(rig);
  const int cellIndex = findCellIndex(topology, 0, 1, 2);
  BOOST_REQUIRE_GE(cellIndex, 0);

  injectCandidateTracklets(rig, cellIndex, clusters);

  TrackerTestAccess::computeCells(rig.traits, view);

  BOOST_REQUIRE_EQUAL(rig.tf->getCells()[cellIndex].size(), 1u);
  const auto& producedCell = rig.tf->getCells()[cellIndex][0];
  BOOST_CHECK(producedCell.tripletFactor().isValid());
  for (int slot = 0; slot < 3; ++slot) {
    const auto reference = producedCell.getClusterReference(slot);
    BOOST_CHECK_EQUAL(reference.surfacePosition, slot);
    BOOST_CHECK_EQUAL(reference.clusterIndex, producedCell.getClusters()[slot]);
  }

  const auto layerGlobalMeasurements = gsl::span<const gsl::span<const GlobalMeasurement>>{view.layerGlobalMeasurements};
  const auto& oracleGlobalInner = layerGlobalMeasurements[0][producedCell.getFirstClusterIndex()];
  const auto& oracleGlobalMiddle = layerGlobalMeasurements[1][producedCell.getSecondClusterIndex()];
  const auto& oracleGlobalOuter = layerGlobalMeasurements[2][producedCell.getThirdClusterIndex()];
  const std::array<GlobalMeasurement, 3> measurements{
    oracleGlobalInner, oracleGlobalMiddle, oracleGlobalOuter};
  TripletFitFactor oracleFactor{};
  BOOST_REQUIRE(makeTripletFitFactor(measurements, oracleFactor));
  checkTripletFitFactorEqual(producedCell.tripletFactor(), oracleFactor);
  checkTrackSeedMaterialization(rig.traits, view, cellIndex, producedCell,
                                SurfaceKind::Disk);
}

// --- One-pass vs two-pass: identical result regardless of thread count ----

BOOST_AUTO_TEST_CASE(CylinderComputeLayerCellsOnePassAndTwoPassAgree)
{
  struct Result {
    int cellIndex{-1};
    std::vector<int> lut;
    TripletFitFactor factor{};
    int cl0{-1}, cl1{-1}, cl2{-1};
  };

  auto run = [](int nThreads) {
    Rig<ITSNLayers> rig{o2::detectors::DetID::ITS, SurfaceKind::Cylinder, nThreads};
    rig.params[0].MaxChi2ClusterAttachment = 1.e6f;
    rig.params[0].LayerxX0[0] = 0.005f;
    rig.params[0].LayerxX0[1] = 0.005f;
    rig.establishLayout();

    const std::array<GlobalMeasurement, 3> clusters{makeGlobalCluster(3.0f, 0.100f, 0.9f, 0),
                                                    makeGlobalCluster(4.0f, 0.150f, 1.05f, 0),
                                                    makeGlobalCluster(5.0f, 0.201f, 1.20f, 0)};
    loadCandidateClusters(rig, clusters,
                          {makeBarrelHit(3.f, 0.f, 0.100f, 0.9f),
                           makeBarrelHit(4.f, 0.f, 0.150f, 1.05f),
                           makeBarrelHit(5.f, 0.f, 0.201f, 1.20f)});

    auto view = prepare(rig);

    const auto topology = topologyView(rig);
    const int cellIndex = findCellIndex(topology, 0, 1, 2);
    BOOST_REQUIRE_GE(cellIndex, 0);

    injectCandidateTracklets(rig, cellIndex, clusters);

    TrackerTestAccess::computeCells(rig.traits, view);

    Result r;
    r.cellIndex = cellIndex;
    const auto& lut = rig.tf->getCellsLookupTable()[cellIndex];
    r.lut.assign(lut.begin(), lut.end());
    BOOST_REQUIRE_EQUAL(rig.tf->getCells()[cellIndex].size(), 1u);
    const auto& cell = rig.tf->getCells()[cellIndex][0];
    r.factor = cell.tripletFactor();
    r.cl0 = cell.getFirstClusterIndex();
    r.cl1 = cell.getSecondClusterIndex();
    r.cl2 = cell.getThirdClusterIndex();
    return r;
  };

  const auto onePass = run(1);
  const auto twoPass = run(4);

  BOOST_CHECK_EQUAL(onePass.cellIndex, twoPass.cellIndex);
  BOOST_CHECK_EQUAL_COLLECTIONS(onePass.lut.begin(), onePass.lut.end(), twoPass.lut.begin(), twoPass.lut.end());
  checkTripletFitFactorEqual(onePass.factor, twoPass.factor);
  BOOST_CHECK_EQUAL(onePass.cl0, twoPass.cl0);
  BOOST_CHECK_EQUAL(onePass.cl1, twoPass.cl1);
  BOOST_CHECK_EQUAL(onePass.cl2, twoPass.cl2);
}

BOOST_AUTO_TEST_CASE(DiskCellRejectsKinkBeyondNominalScatteringTolerance)
{
  Rig<MFTNLayers> rig{o2::detectors::DetID::MFT, SurfaceKind::Disk};
  rig.params[0].TrackletMinPt = 0.3f;
  rig.establishLayout();

  // This kinked triplet used to be the threading/repeated-call fixture.
  // Its dip-angle change exceeds the tolerance with nominal MFT material.
  const std::array<GlobalMeasurement, 3> clusters{makeGlobalCluster(1.0f, 0.5f, -0.4f, 0),
                                                  makeGlobalCluster(1.3f, 0.62f, -0.6f, 0),
                                                  makeGlobalCluster(1.7f, 0.78f, -0.9f, 0)};
  loadCandidateClusters(rig, clusters,
                        {makeDiskHit(-0.4f, 1.0f, 0.5f),
                         makeDiskHit(-0.6f, 1.3f, 0.62f),
                         makeDiskHit(-0.9f, 1.7f, 0.78f)});
  auto view = prepare(rig);
  const auto topology = topologyView(rig);
  const int cellIndex = findCellIndex(topology, 0, 1, 2);
  BOOST_REQUIRE_GE(cellIndex, 0);
  injectCandidateTracklets(rig, cellIndex, clusters);

  const auto& path = topology.getPath(CellPathId{static_cast<uint16_t>(cellIndex)});
  const auto& first = rig.tf->getTracklets()[path.first.value()][0];
  const auto& second = rig.tf->getTracklets()[path.second.value()][0];
  const float deltaLambda = std::abs(std::atan(first.tanLambda) - std::atan(second.tanLambda));
  const float angularTolerance = view.configuration.kernelParameters.nSigmaCut * rig.tf->getEdgeMSAngle(path.second.value());
  BOOST_REQUIRE_GT(deltaLambda, angularTolerance);
  // Exclude a failed triplet fit as the reason for rejecting this candidate.
  const std::array<GlobalMeasurement, 3> measurements{view.layerGlobalMeasurements[0][0],
                                                      view.layerGlobalMeasurements[1][0],
                                                      view.layerGlobalMeasurements[2][0]};
  TripletFitFactor factor{};
  BOOST_REQUIRE(makeTripletFitFactor(measurements, factor));
  BOOST_REQUIRE(factor.isValid());

  TrackerTestAccess::computeCells(rig.traits, view);
  BOOST_CHECK(rig.tf->getCells()[cellIndex].empty());
}

BOOST_AUTO_TEST_CASE(DiskComputeLayerCellsOnePassAndTwoPassAgree)
{
  struct Result {
    int cellIndex{-1};
    std::vector<int> lut;
    TripletFitFactor factor{};
    int cl0{-1}, cl1{-1}, cl2{-1};
  };

  auto run = [](int nThreads) {
    Rig<MFTNLayers> rig{o2::detectors::DetID::MFT, SurfaceKind::Disk, nThreads};
    rig.params[0].MaxChi2ClusterAttachment = 1.e6f;
    rig.params[0].TrackletMinPt = 0.3f;
    rig.establishLayout();

    // Straight triplet on the synthetic disk planes, comfortably inside the
    // nominal-material angular cut: this test checks threading, not rejection.
    const std::array<GlobalMeasurement, 3> clusters{makeGlobalCluster(1.0f, 0.5f, -0.4f, 0),
                                                    makeGlobalCluster(1.3f, 0.62f, -0.6f, 0),
                                                    makeGlobalCluster(1.6f, 0.74f, -0.8f, 0)};
    loadCandidateClusters(rig, clusters,
                          {makeDiskHit(-0.4f, 1.0f, 0.5f),
                           makeDiskHit(-0.6f, 1.3f, 0.62f),
                           makeDiskHit(-0.8f, 1.6f, 0.74f)});

    auto view = prepare(rig);

    const auto topology = topologyView(rig);
    const int cellIndex = findCellIndex(topology, 0, 1, 2);
    BOOST_REQUIRE_GE(cellIndex, 0);

    injectCandidateTracklets(rig, cellIndex, clusters);

    TrackerTestAccess::computeCells(rig.traits, view);

    Result r;
    r.cellIndex = cellIndex;
    const auto& lut = rig.tf->getCellsLookupTable()[cellIndex];
    r.lut.assign(lut.begin(), lut.end());
    BOOST_REQUIRE_EQUAL(rig.tf->getCells()[cellIndex].size(), 1u);
    const auto& cell = rig.tf->getCells()[cellIndex][0];
    r.factor = cell.tripletFactor();
    r.cl0 = cell.getFirstClusterIndex();
    r.cl1 = cell.getSecondClusterIndex();
    r.cl2 = cell.getThirdClusterIndex();
    return r;
  };

  const auto onePass = run(1);
  const auto twoPass = run(4);

  BOOST_CHECK_EQUAL(onePass.cellIndex, twoPass.cellIndex);
  BOOST_CHECK_EQUAL_COLLECTIONS(onePass.lut.begin(), onePass.lut.end(), twoPass.lut.begin(), twoPass.lut.end());
  checkTripletFitFactorEqual(onePass.factor, twoPass.factor);
  BOOST_CHECK_EQUAL(onePass.cl0, twoPass.cl0);
  BOOST_CHECK_EQUAL(onePass.cl1, twoPass.cl1);
  BOOST_CHECK_EQUAL(onePass.cl2, twoPass.cl2);
}

BOOST_AUTO_TEST_CASE(RepeatedComputeLayerCellsCallsDoNotRebindOrIncreaseCounts)
{
  Rig<MFTNLayers> rig{o2::detectors::DetID::MFT, SurfaceKind::Disk};
  rig.params[0].MaxChi2ClusterAttachment = 1.e6f;
  rig.params[0].TrackletMinPt = 0.3f;
  rig.establishLayout();

  // Use the same accepted straight triplet as the threading test above.
  const std::array<GlobalMeasurement, 3> clusters{makeGlobalCluster(1.0f, 0.5f, -0.4f, 0),
                                                  makeGlobalCluster(1.3f, 0.62f, -0.6f, 0),
                                                  makeGlobalCluster(1.6f, 0.74f, -0.8f, 0)};
  loadCandidateClusters(rig, clusters,
                        {makeDiskHit(-0.4f, 1.0f, 0.5f),
                         makeDiskHit(-0.6f, 1.3f, 0.62f),
                         makeDiskHit(-0.8f, 1.6f, 0.74f)});

  auto view = prepare(rig);

  const auto topology = topologyView(rig);
  const int cellIndex = findCellIndex(topology, 0, 1, 2);
  BOOST_REQUIRE_GE(cellIndex, 0);

  injectCandidateTracklets(rig, cellIndex, clusters);

  TrackerTestAccess::computeCells(rig.traits, view);
  BOOST_REQUIRE_EQUAL(rig.tf->getCells()[cellIndex].size(), 1u);
  const auto firstFactor = rig.tf->getCells()[cellIndex][0].tripletFactor();

  TrackerTestAccess::computeCells(rig.traits, view);
  TrackerTestAccess::computeCells(rig.traits, view);

  // Re-inject tracklets and recompute (the underlying candidate clusters/
  // measurements loaded above are untouched -- reloading them here would
  // invalidate the frame-owned source measurement lookup without a fresh
  // initialiseTimeFrame() call to re-resolve it, which is not what this test
  // checks): a fresh call after the tracklets were consumed must still
  // reproduce the identical triplet factor through the same cache.
  injectCandidateTracklets(rig, cellIndex, clusters);
  TrackerTestAccess::computeCells(rig.traits, view);

  BOOST_REQUIRE_EQUAL(rig.tf->getCells()[cellIndex].size(), 1u);
  checkTripletFitFactorEqual(rig.tf->getCells()[cellIndex][0].tripletFactor(), firstFactor);
}

// Material-correction preflight has its own focused test target; this file
// covers only direct cell-stage orchestration.

BOOST_AUTO_TEST_CASE(CylinderComputeLayerCellsMultiCellChainProducesCorrectCellsAndOrder)
{
  // 5-layer chain at global X = 3..7 (small Y, alpha=0.f, matching the
  // single-cell oracle tests' convention above): proves edge-level/
  // cell-level parity across three simultaneously-populated cells (0,1,2),
  // (1,2,3), (2,3,4) -- each resolved through the migrated
  // computeLayerCellsForKind() via a fresh mSurfaceToLegacyLayer lookup
  // per derived path -- not just the single path the tests above check,
  // while every non-participating cellIndex stays empty.
  Rig<ITSNLayers> rig{o2::detectors::DetID::ITS, SurfaceKind::Cylinder};
  rig.params[0].MaxChi2ClusterAttachment = 1.e6f;
  for (int layer = 0; layer < ITSNLayers; ++layer) {
    rig.params[0].LayerxX0[layer] = 0.005f;
  }
  rig.establishLayout();

  // Y values lie exactly on one real circle (center (0,5000), radius 5000,
  // through the origin) rather than an ad hoc linear Y(X): a single
  // physically consistent curvature across all 5 points avoids the
  // rotation-boundary edge cases (BarrelSurfaceStateOperations.cxx's
  // csp*ca+snp*sa<0 checks) an inconsistent, near-degenerate linear Y(X)
  // can trip for some sub-triples but not others.
  constexpr std::array<int, 5> layers{0, 1, 2, 3, 4};
  constexpr std::array<float, 5> xs{3.f, 4.f, 5.f, 6.f, 7.f};
  constexpr std::array<float, 5> ys{0.0009f, 0.0016f, 0.0025f, 0.0036f, 0.0049f};
  constexpr std::array<float, 5> zs{0.90f, 1.05f, 1.20f, 1.35f, 1.50f};
  std::array<GlobalMeasurement, 5> clusters;
  std::array<TestLocalMeasurement, 5> hits;
  for (size_t i = 0; i < 5; ++i) {
    clusters[i] = makeGlobalCluster(xs[i], ys[i], zs[i], 0);
    hits[i] = makeBarrelHit(xs[i], 0.f, ys[i], zs[i]);
  }
  loadCandidateClustersAtLayers(rig, layers, clusters, hits);

  auto view = prepare(rig);

  const auto topology = topologyView(rig);
  injectChainCandidateTracklets(rig, layers, clusters);

  TrackerTestAccess::computeCells(rig.traits, view);

  const std::array<std::array<int, 3>, 3> triples{{{0, 1, 2}, {1, 2, 3}, {2, 3, 4}}};
  std::array<int, 3> topologyIds{};
  std::vector<bool> participating(topology.nPaths, false);
  for (size_t i = 0; i < triples.size(); ++i) {
    const auto& triple = triples[i];
    const int cellIndex = findCellIndex(topology, triple[0], triple[1], triple[2]);
    BOOST_REQUIRE_GE(cellIndex, 0);
    topologyIds[i] = cellIndex;
    participating[cellIndex] = true;

    BOOST_REQUIRE_EQUAL(rig.tf->getCells()[cellIndex].size(), 1u);
    const auto& producedCell = rig.tf->getCells()[cellIndex][0];
    BOOST_CHECK_EQUAL(producedCell.getFirstClusterIndex(), 0);
    BOOST_CHECK_EQUAL(producedCell.getSecondClusterIndex(), 0);
    BOOST_CHECK_EQUAL(producedCell.getThirdClusterIndex(), 0);
    BOOST_CHECK_EQUAL(producedCell.getHitLayerMask().value(), LayerMask(triple[0], triple[1], triple[2]).value());

    BOOST_REQUIRE_EQUAL(rig.tf->getCellsLookupTable()[cellIndex].size(), 2u);
    BOOST_CHECK_EQUAL(rig.tf->getCellsLookupTable()[cellIndex][0], 0);
    BOOST_CHECK_EQUAL(rig.tf->getCellsLookupTable()[cellIndex][1], 1);
  }

  for (int i = 0; i < topology.nPaths; ++i) {
    if (!participating[i]) {
      BOOST_CHECK(rig.tf->getCells()[i].empty());
    }
  }

  TrackerTestAccess::findNeighbours(rig.traits, view);
  for (size_t i = 0; i < topologyIds.size(); ++i) {
    BOOST_CHECK_EQUAL(rig.tf->getCells()[topologyIds[i]][0].getLevel(), static_cast<int>(i + 1));
    if (i == 0) {
      BOOST_CHECK(rig.tf->getCellsNeighbours()[topologyIds[i]].empty());
      continue;
    }
    BOOST_REQUIRE_EQUAL(rig.tf->getCellsNeighbours()[topologyIds[i]].size(), 1u);
    BOOST_CHECK_EQUAL(rig.tf->getCellsNeighbours()[topologyIds[i]][0], 0);
    BOOST_CHECK_EQUAL(rig.tf->getCellsNeighboursTopology()[topologyIds[i]][0], topologyIds[i - 1]);
  }
}

BOOST_AUTO_TEST_CASE(DiskComputeLayerCellsMultiCellChainProducesCorrectCellsAndOrder)
{
  // Same multi-cell parity property for the Disk/forward family:
  // cell-seed leaves genuinely branches per family (Cylinder
  // reads [1] then [0]; Disk reads [2],[1],[0] -- see the comment on
  // that call in computeLayerCellsForKind()), so multi-edge
  // cell-chaining for Disk is real, otherwise-unproven coverage.
  Rig<MFTNLayers> rig{o2::detectors::DetID::MFT, SurfaceKind::Disk};
  rig.params[0].MaxChi2ClusterAttachment = 1.e6f;
  rig.params[0].TrackletMinPt = 0.3f;
  rig.establishLayout();

  constexpr std::array<int, 5> layers{0, 1, 2, 3, 4};
  constexpr std::array<float, 5> xs{1.0f, 1.3f, 1.6f, 1.9f, 2.2f};
  constexpr std::array<float, 5> ys{0.50f, 0.62f, 0.74f, 0.86f, 0.98f};
  constexpr std::array<float, 5> zs{-0.40f, -0.60f, -0.80f, -1.00f, -1.20f};
  std::array<GlobalMeasurement, 5> clusters;
  std::array<TestLocalMeasurement, 5> hits;
  for (size_t i = 0; i < 5; ++i) {
    clusters[i] = makeGlobalCluster(xs[i], ys[i], zs[i], 0);
    hits[i] = makeDiskHit(zs[i], xs[i], ys[i]);
  }
  loadCandidateClustersAtLayers(rig, layers, clusters, hits);

  auto view = prepare(rig);

  const auto topology = topologyView(rig);
  injectChainCandidateTracklets(rig, layers, clusters);

  TrackerTestAccess::computeCells(rig.traits, view);

  const std::array<std::array<int, 3>, 3> triples{{{0, 1, 2}, {1, 2, 3}, {2, 3, 4}}};
  std::array<int, 3> topologyIds{};
  std::vector<bool> participating(topology.nPaths, false);
  for (size_t i = 0; i < triples.size(); ++i) {
    const auto& triple = triples[i];
    const int cellIndex = findCellIndex(topology, triple[0], triple[1], triple[2]);
    BOOST_REQUIRE_GE(cellIndex, 0);
    topologyIds[i] = cellIndex;
    participating[cellIndex] = true;

    BOOST_REQUIRE_EQUAL(rig.tf->getCells()[cellIndex].size(), 1u);
    const auto& producedCell = rig.tf->getCells()[cellIndex][0];
    BOOST_CHECK_EQUAL(producedCell.getHitLayerMask().value(), LayerMask(triple[0], triple[1], triple[2]).value());

    BOOST_REQUIRE_EQUAL(rig.tf->getCellsLookupTable()[cellIndex].size(), 2u);
    BOOST_CHECK_EQUAL(rig.tf->getCellsLookupTable()[cellIndex][0], 0);
    BOOST_CHECK_EQUAL(rig.tf->getCellsLookupTable()[cellIndex][1], 1);
  }

  for (int i = 0; i < topology.nPaths; ++i) {
    if (!participating[i]) {
      BOOST_CHECK(rig.tf->getCells()[i].empty());
    }
  }

  TrackerTestAccess::findNeighbours(rig.traits, view);
  for (size_t i = 0; i < topologyIds.size(); ++i) {
    BOOST_CHECK_EQUAL(rig.tf->getCells()[topologyIds[i]][0].getLevel(), static_cast<int>(i + 1));
    if (i == 0) {
      BOOST_CHECK(rig.tf->getCellsNeighbours()[topologyIds[i]].empty());
      continue;
    }
    BOOST_REQUIRE_EQUAL(rig.tf->getCellsNeighbours()[topologyIds[i]].size(), 1u);
    BOOST_CHECK_EQUAL(rig.tf->getCellsNeighbours()[topologyIds[i]][0], 0);
    BOOST_CHECK_EQUAL(rig.tf->getCellsNeighboursTopology()[topologyIds[i]][0], topologyIds[i - 1]);
  }
}

BOOST_AUTO_TEST_CASE(CylinderComputeLayerCellsHoleCellReconstructsCorrectLayerMask)
{
  // MaxHoles=1 with layer 1 an allowed hole introduces a (0,2)-skip-1
  // edge; combined with the adjacent (2,3) edge this forms cell
  // (0,2,3) -- a direct, non-adjacent exercise of resolveCellHitLayers()
  // (mSurfaceToLegacyLayer) resolving a cell's endpoints correctly, and of
  // hole/skipped-surface behaviour staying identical to the pre-migration
  // code (which read the same fromLayer/toLayer straight off the legacy
  // view). No cluster is placed on layer 1 at all.
  Rig<ITSNLayers> rig{o2::detectors::DetID::ITS, SurfaceKind::Cylinder};
  rig.params[0].MaxChi2ClusterAttachment = 1.e6f;
  rig.params[0].MaxHoles = 1;
  rig.holeLayers = LayerMask{static_cast<uint16_t>(1u << 1)};
  rig.establishLayout();

  constexpr std::array<int, 3> layers{0, 2, 3};
  const std::array<GlobalMeasurement, 3> clusters{
    makeGlobalCluster(3.f, 0.10f, 0.90f, 0),
    makeGlobalCluster(5.f, 0.20f, 1.20f, 0),
    makeGlobalCluster(6.f, 0.25f, 1.35f, 0)};
  const std::array<TestLocalMeasurement, 3> hits{
    makeBarrelHit(3.f, 0.f, 0.10f, 0.90f),
    makeBarrelHit(5.f, 0.f, 0.20f, 1.20f),
    makeBarrelHit(6.f, 0.f, 0.25f, 1.35f)};
  loadCandidateClustersAtLayers(rig, layers, clusters, hits);

  auto view = prepare(rig);

  const auto topology = topologyView(rig);
  injectChainCandidateTracklets(rig, layers, clusters);

  TrackerTestAccess::computeCells(rig.traits, view);

  const int cellIndex = findCellIndex(topology, 0, 2, 3);
  BOOST_REQUIRE_GE(cellIndex, 0);
  BOOST_REQUIRE_EQUAL(rig.tf->getCells()[cellIndex].size(), 1u);
  const auto& producedCell = rig.tf->getCells()[cellIndex][0];
  BOOST_CHECK_EQUAL(producedCell.getHitLayerMask().value(), LayerMask(0, 2, 3).value());

  BOOST_REQUIRE_EQUAL(rig.tf->getCellsLookupTable()[cellIndex].size(), 2u);
  BOOST_CHECK_EQUAL(rig.tf->getCellsLookupTable()[cellIndex][0], 0);
  BOOST_CHECK_EQUAL(rig.tf->getCellsLookupTable()[cellIndex][1], 1);

  for (int i = 0; i < topology.nPaths; ++i) {
    if (i != cellIndex) {
      BOOST_CHECK(rig.tf->getCells()[i].empty());
    }
  }
}
