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

#define BOOST_TEST_MODULE ITSMFT ComputeLayerTracklets orchestration
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <stdexcept>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <oneapi/tbb/task_arena.h>

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "ITSMFTTracking/Configuration.h"
#include "ITSMFTTracking/ITSMFTDetectorDefinitions.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"
#include "ITSMFTTracking/detail/TimeFrameScratch.h"
#include "ITSMFTTracking/detail/TrackerTraversalPreparation.h"
#include "ITSMFTTracking/TimeFrame.h"
#include "ITSMFTTracking/TrackerTraits.h"
#include "TraversalTestSupport.h"
#include "ITSMFTTracking/TrackingConfigParam.h"
#include "ITSMFTTracking/Constants.h"
#include "ITSMFTTracking/MathUtils.h"
#include "ITSMFTTracking/ROFLookupTables.h"
#include "MFTTracking/Constants.h"
#include "CommonConstants/MathConstants.h"

#include "TrackingParameterTestSupport.h"

using o2::itsmft::tracking::test::ReferenceTrackingParameters;
using namespace o2::itsmft;
using namespace o2::itsmft::tracking;

namespace
{

constexpr float Bz = 0.5f;
constexpr std::array<unsigned char, 3> OnePixelPattern{1, 1, 0x80};

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

std::vector<SurfaceDescriptor> makeCatalog(uint16_t nLayers, o2::detectors::DetID::ID detector, SurfaceKind kind)
{
  std::vector<SurfaceDescriptor> surfaces;
  surfaces.reserve(nLayers);
  for (uint16_t i = 0; i < nLayers; ++i) {
    surfaces.push_back(SurfaceDescriptor{i, static_cast<uint8_t>(detector), kind});
    surfaces.back().chartRange = kind == SurfaceKind::Disk ? SurfaceChartRange{0.1f, 20.f} : SurfaceChartRange{-20.f, 20.f};
    surfaces.back().referenceCoordinate = kind == SurfaceKind::Disk
                                            ? o2::mft::constants::mft::LayerZCoordinate()[i % MFTNLayers]
                                            : 3.f + static_cast<float>(i);
    // Matches o2::itsmft::resetDetectorDefaults()'s per-detector LayerxX0
    // default, so TrackerTraits::initialiseTimeFrame()'s LegacyMaterialMismatch
    // compatibility check passes for these unperturbed fixtures.
    const float xOverX0 = detector == o2::detectors::DetID::MFT ? kNominalMFTLayerX0[i % MFTNLayers] : kNominalITSLayerX0[i % ITSNLayers];
    surfaces.back().material.xOverX0 = xOverX0;
    surfaces.back().material.arealDensityGPerCm2 = xOverX0 * o2::its::constants::Radl * o2::its::constants::Rho;
  }
  return surfaces;
}

class PrescribedDecoder
{
 public:
  PrescribedDecoder(o2::detectors::DetID::ID detector, SurfaceKind kind, std::vector<DecodedCluster> clusters)
    : mDetector{detector}, mKind{kind}, mClusters{std::move(clusters)}
  {
  }

  o2::itsmft::tracking::DecodedCluster decode(
    const CompClusterExt& cluster,
    gsl::span<const unsigned char>::iterator& patterns,
    const TopologyDictionary* dictionary,
    uint32_t externalIndex,
    bool) const
  {
    const auto clusterData = o2::itsmft::ioutils::extractClusterData(cluster, patterns, dictionary);
    o2::itsmft::tracking::DecodedCluster result;
    if (externalIndex >= mClusters.size()) {
      return result;
    }
    auto decoded = mClusters[externalIndex];
    decoded.nPixels = clusterData.nPixels;
    result = decoded;
    return result;
  }

 private:
  o2::detectors::DetID::ID mDetector;
  SurfaceKind mKind;
  std::vector<DecodedCluster> mClusters;
};

struct TrackletSnapshot {
  int edgeId{-1};
  std::vector<Tracklet> tracklets;
  std::vector<int> lookup;
  o2::its::TimeEstBC expectedTimestamp;
  bool nonparticipatingEdgesEmpty{false};
  // Gate 4 Slice 0a additions: full per-(legacy-edgeId) tracklet/LUT
  // content and (fromLayer,toLayer) identity, for multi-edge
  // candidate-set/order/LUT parity checks that go beyond the single
  // `edgeId` above. Indices across these three vectors correspond
  // 1:1, in ascending legacy edgeId order.
  std::vector<int> allEdgeFromLayer;
  std::vector<int> allEdgeToLayer;
  std::vector<std::vector<Tracklet>> allTracklets;
  std::vector<std::vector<int>> allLookups;
};

template <int NLayers>
TrackletSnapshot runFixture(o2::detectors::DetID::ID detector,
                            SurfaceKind kind,
                            SurfaceKind tag,
                            std::vector<DecodedCluster> decoded,
                            int nThreads,
                            std::function<void(ReferenceTrackingParameters&)> customizeParams = {},
                            LayerMask holeLayers = {})
{
  auto pool = std::make_shared<BoundedMemoryResource>();
  TimeFrame frame;
  Tracker tracker;
  TrackerTraits traits;
  std::shared_ptr<tbb::task_arena> arena;
  std::vector<ReferenceTrackingParameters> params(1);
  resetDetectorDefaults(params[0], detector);
  params[0].UseDiamond = true;
  params[0].CreateArtefactLabels = false;
  params[0].PassFlags.reset();
  params[0].PassFlags.set(IterationStep::FirstPass, IterationStep::RebuildClusterLUT);
  if (customizeParams) {
    customizeParams(params[0]);
  }

  traits.setNThreads(nThreads, arena);
  frame.setBz(Bz);

  const auto orderedSurfaces = identitySurfaces(static_cast<uint16_t>(NLayers));
  const auto catalog = makeCatalog(static_cast<uint16_t>(NLayers), detector, kind);
  const SurfaceCatalogView catalogView{catalog.data(), static_cast<uint32_t>(catalog.size())};
  TrackerInitialization configuration;
  configuration.catalog = catalogView;
  configuration.memoryPool = pool;
  configuration.layout = makeDetectorLayout(holeLayers);
  configuration.plan = o2::itsmft::tracking::test::makeTrackingPlan(params[0]);
  BOOST_REQUIRE(tracker.initialize(frame, configuration).ok());
  auto& tf = frame.getScratch();
  const auto& layout = frame.getLayout();

  std::vector<CompClusterExt> compactClusters;
  std::vector<unsigned char> patterns;
  compactClusters.reserve(decoded.size());
  patterns.reserve(decoded.size() * OnePixelPattern.size());
  for (const auto& cluster : decoded) {
    compactClusters.emplace_back(0, 0, CompCluster::InvalidPatternID, cluster.layer);
    patterns.insert(patterns.end(), OnePixelPattern.begin(), OnePixelPattern.end());
  }
  const std::vector<ROFRecord> rofs{ROFRecord{{100, 5}, 0, 0, static_cast<int>(compactClusters.size())}};
  PrescribedDecoder decoder{detector, kind, std::move(decoded)};
  BOOST_REQUIRE_NO_THROW(test::loadTimeFrameSource(frame, decoder, o2::InteractionRecord{50, 5}, ROFTimingConfig{40, 0, 0, 0},
                                                   compactClusters, patterns, rofs, &dict(), nullptr, detector,
                                                   gsl::span<const LayerId>{orderedSurfaces}, layout.getSurfaceCatalog()));

  o2::its::LayerTiming layerTiming{};
  layerTiming.mNROFsTF = 1;
  layerTiming.mROFLength = 40;
  o2::its::ROFOverlapTable<NLayers> rofTable;
  for (int layer = 0; layer < NLayers; ++layer) {
    rofTable.defineLayer(layer, layerTiming);
  }
  rofTable.init();
  // Real production workflow timing construction
  // always builds and sets this alongside the ROFOverlapTable above, from
  // the same per-layer LayerTiming, regardless of UseDiamond -- the diamond
  // vertex derived per-ROF for tracklet finding (TrackerTraits.cxx) is
  // checked through the genuine isVertexCompatible() on this table, not a
  // useDiamond-skipped shortcut, so this fixture needs it populated too.
  o2::its::ROFVertexLookupTable<NLayers> vtxTable;
  for (int layer = 0; layer < NLayers; ++layer) {
    vtxTable.defineLayer(layer, layerTiming);
  }
  vtxTable.init();
  o2::its::ROFMaskTable<NLayers> mask{rofTable};
  mask.resetMask();
  for (int layer = 0; layer < NLayers; ++layer) {
    mask.setROFsEnabled(layer, 0, 1, 1);
  }
  frame.setROFViews(RuntimeROFViews{rofTable.getView(), vtxTable.getView(), mask.getView(), {}});

  std::array<gsl::span<const GlobalMeasurement>, MaxLayoutSurfaces> measurementSpans;
  auto view = TrackerTestAccess::prepare(tracker, frame, 0, measurementSpans);
  BOOST_CHECK(view.layerGlobalMeasurements.data() == measurementSpans.data());
  const auto layoutView = view.topology;

  // Prepared edge arrays must be complete and finite.
  {
    const auto preparedTopology = layoutView;
    const auto& msAngles = tf.getEdgeMSAngles();
    const auto& phiCuts = tf.getEdgePhiCuts();
    BOOST_REQUIRE_EQUAL(msAngles.size(), static_cast<size_t>(preparedTopology.nEdges));
    BOOST_REQUIRE_EQUAL(phiCuts.size(), static_cast<size_t>(preparedTopology.nEdges));
    for (int id = 0; id < preparedTopology.nEdges; ++id) {
      BOOST_CHECK(std::isfinite(msAngles[id]));
      BOOST_CHECK(std::isfinite(phiCuts[id]));
    }
  }

  const auto topology = layoutView;
  int edgeId = -1;
  for (int id = 0; id < topology.nEdges; ++id) {
    const auto& edge = topology.getEdge(EdgeId{static_cast<uint16_t>(id)});
    if (edge.from.value() == 0 && edge.to.value() == 1) {
      edgeId = id;
      break;
    }
  }
  BOOST_REQUIRE_GE(edgeId, 0);

  TrackerTestAccess::computeTracklets(traits, view, 0);

  TrackletSnapshot result;
  result.edgeId = edgeId;
  result.expectedTimestamp = frame.getROFOverlapView().getTimeStamp(0, 0, 1, 0);
  const auto& tracklets = tf.getTracklets()[edgeId];
  result.tracklets.assign(tracklets.begin(), tracklets.end());
  const auto& lookup = tf.getTrackletsLookupTable()[edgeId];
  result.lookup.assign(lookup.begin(), lookup.end());
  result.nonparticipatingEdgesEmpty = true;
  for (int id = 0; id < topology.nEdges; ++id) {
    if (id != edgeId && !tf.getTracklets()[id].empty()) {
      result.nonparticipatingEdgesEmpty = false;
      break;
    }
  }

  // Gate 4 Slice 0a: full per-edge snapshot, ascending legacy
  // edgeId order, for multi-edge candidate-set/order/LUT parity
  // checks (see e.g. ItsIdentityLayoutTrackletsSpanMultipleAdjacentEdgesInOrder).
  for (int id = 0; id < topology.nEdges; ++id) {
    const auto& edge = topology.getEdge(EdgeId{static_cast<uint16_t>(id)});
    result.allEdgeFromLayer.push_back(edge.from.value());
    result.allEdgeToLayer.push_back(edge.to.value());
    const auto& idTracklets = tf.getTracklets()[id];
    result.allTracklets.emplace_back(idTracklets.begin(), idTracklets.end());
    const auto& idLookup = tf.getTrackletsLookupTable()[id];
    result.allLookups.emplace_back(idLookup.begin(), idLookup.end());
  }
  return result;
}

void checkSame(const TrackletSnapshot& serial, const TrackletSnapshot& parallel)
{
  BOOST_CHECK_EQUAL(serial.edgeId, parallel.edgeId);
  BOOST_REQUIRE_EQUAL(serial.tracklets.size(), parallel.tracklets.size());
  BOOST_CHECK_EQUAL_COLLECTIONS(serial.lookup.begin(), serial.lookup.end(), parallel.lookup.begin(), parallel.lookup.end());
  for (size_t i = 0; i < serial.tracklets.size(); ++i) {
    BOOST_CHECK(serial.tracklets[i] == parallel.tracklets[i]);
    BOOST_CHECK_EQUAL(serial.tracklets[i].tanLambda, parallel.tracklets[i].tanLambda);
    BOOST_CHECK_EQUAL(serial.tracklets[i].phi, parallel.tracklets[i].phi);
    BOOST_CHECK_EQUAL(serial.tracklets[i].getTimeStamp().getTimeStamp(), parallel.tracklets[i].getTimeStamp().getTimeStamp());
    BOOST_CHECK_EQUAL(serial.tracklets[i].getTimeStamp().getTimeStampError(), parallel.tracklets[i].getTimeStamp().getTimeStampError());
  }
}

void checkExactTracklet(const TrackletSnapshot& snapshot, float expectedTanLambda, float expectedPhi)
{
  BOOST_REQUIRE_EQUAL(snapshot.tracklets.size(), 1u);
  const auto& tracklet = snapshot.tracklets.front();
  BOOST_CHECK_EQUAL(tracklet.firstClusterIndex, 0);
  BOOST_CHECK_EQUAL(tracklet.secondClusterIndex, 0);
  BOOST_CHECK_EQUAL(tracklet.tanLambda, expectedTanLambda);
  BOOST_CHECK_EQUAL(tracklet.phi, expectedPhi);
  BOOST_CHECK_EQUAL(tracklet.getTimeStamp().getTimeStamp(), snapshot.expectedTimestamp.getTimeStamp());
  BOOST_CHECK_EQUAL(tracklet.getTimeStamp().getTimeStampError(), snapshot.expectedTimestamp.getTimeStampError());
  const std::vector<int> expectedLookup{0, 1};
  BOOST_CHECK_EQUAL_COLLECTIONS(snapshot.lookup.begin(), snapshot.lookup.end(), expectedLookup.begin(), expectedLookup.end());
  BOOST_CHECK(snapshot.nonparticipatingEdgesEmpty);
}

DecodedCluster cylinderCluster(float radius, float z, int layer)
{
  DecodedCluster cluster{};
  cluster.global = {radius, 0.f, z};
  cluster.cylinderFrame = {radius, 0.f, z, 0.f};
  cluster.rowColumnCovariance = {1.e-4f, 0.f, 1.e-4f};
  cluster.layer = layer;
  return cluster;
}

DecodedCluster diskCluster(float x, float y, float z, int layer)
{
  DecodedCluster cluster{};
  cluster.global = {x, y, z};
  cluster.rowColumnCovariance = {1.e-2f, 0.f, 1.e-2f};
  cluster.layer = layer;
  return cluster;
}

} // namespace

BOOST_AUTO_TEST_CASE(CylinderOnePassAndTwoPassProduceIdenticalTracklets)
{
  const std::vector<DecodedCluster> clusters{
    cylinderCluster(3.f, 0.3f, 0),
    cylinderCluster(4.f, 0.4f, 1)};
  const auto serial = runFixture<ITSNLayers>(o2::detectors::DetID::ITS, SurfaceKind::Cylinder,
                                             SurfaceKind::Cylinder, clusters, 1);
  const auto parallel = runFixture<ITSNLayers>(o2::detectors::DetID::ITS, SurfaceKind::Cylinder,
                                               SurfaceKind::Cylinder, clusters, 4);
  checkExactTracklet(serial, (0.3f - 0.4f) / (3.f - 4.f), o2::gpu::CAMath::ATan2(0.f, -1.f));
  checkExactTracklet(parallel, (0.3f - 0.4f) / (3.f - 4.f), o2::gpu::CAMath::ATan2(0.f, -1.f));
  checkSame(serial, parallel);
}

BOOST_AUTO_TEST_CASE(CylinderDisplacedChordPreservesBothLongitudinalSigns)
{
  // A line parallel to x, displaced by y=1: its transverse length is exactly
  // one, while the difference of beam-axis radii is smaller than one.
  for (const float sign : {-1.f, 1.f}) {
    std::vector<DecodedCluster> clusters;
    for (int layer = 0; layer < 2; ++layer) {
      const float x = 3.f + layer;
      const float z = sign * 0.25f * (layer + 1);
      auto cluster = cylinderCluster(x, z, layer);
      cluster.global.y = 1.f;
      cluster.cylinderFrame.u = 1.f;
      clusters.push_back(cluster);
    }
    const auto widenSearch = [](ReferenceTrackingParameters& p) {
      p.NSigmaCut = 100.f;
      p.PVres = 10.f; // Widen the independent azimuthal search gate too.
    };
    const auto serial = runFixture<ITSNLayers>(o2::detectors::DetID::ITS, SurfaceKind::Cylinder,
                                               SurfaceKind::Cylinder, clusters, 1, widenSearch);
    const auto parallel = runFixture<ITSNLayers>(o2::detectors::DetID::ITS, SurfaceKind::Cylinder,
                                                 SurfaceKind::Cylinder, clusters, 4, widenSearch);
    const float expectedPhi = o2::gpu::CAMath::ATan2(0.f, -1.f);
    checkExactTracklet(serial, sign * 0.25f, expectedPhi);
    checkExactTracklet(parallel, sign * 0.25f, expectedPhi);
    checkSame(serial, parallel);
  }
}

BOOST_AUTO_TEST_CASE(DiskEqualRadiusDistinctHitsHaveFiniteSignedSlope)
{
  const float fromZ = kMFTStaticSurfaceCatalog[0].referenceCoordinate;
  const float toZ = kMFTStaticSurfaceCatalog[1].referenceCoordinate;
  // Same radius, different positions, with a transverse chord of exactly one.
  const std::vector<DecodedCluster> clusters{
    diskCluster(1.f, 0.5f, fromZ, 0),
    diskCluster(1.f, -0.5f, toZ, 1)};
  const auto widenSearch = [](ReferenceTrackingParameters& p) {
    p.NSigmaCut = 100.f;
    p.PVres = 10.f;
  };
  const auto serial = runFixture<MFTNLayers>(o2::detectors::DetID::MFT, SurfaceKind::Disk,
                                             SurfaceKind::Disk, clusters, 1, widenSearch);
  const auto parallel = runFixture<MFTNLayers>(o2::detectors::DetID::MFT, SurfaceKind::Disk,
                                               SurfaceKind::Disk, clusters, 4, widenSearch);
  const float expectedPhi = o2::gpu::CAMath::ATan2(1.f, 0.f);
  checkExactTracklet(serial, toZ - fromZ, expectedPhi);
  checkExactTracklet(parallel, toZ - fromZ, expectedPhi);
  checkSame(serial, parallel);
}

BOOST_AUTO_TEST_CASE(DiskZeroTransverseChordRejectsTracklet)
{
  const float fromZ = kMFTStaticSurfaceCatalog[0].referenceCoordinate;
  const float toZ = kMFTStaticSurfaceCatalog[1].referenceCoordinate;
  const std::vector<DecodedCluster> clusters{
    diskCluster(1.f, 0.5f, fromZ, 0),
    diskCluster(1.f, 0.5f, toZ, 1)};
  const auto widenSearch = [](ReferenceTrackingParameters& params) { params.NSigmaCut = 1.e6f; };
  const auto serial = runFixture<MFTNLayers>(o2::detectors::DetID::MFT, SurfaceKind::Disk,
                                             SurfaceKind::Disk, clusters, 1, widenSearch);
  const auto parallel = runFixture<MFTNLayers>(o2::detectors::DetID::MFT, SurfaceKind::Disk,
                                               SurfaceKind::Disk, clusters, 4, widenSearch);
  BOOST_CHECK(serial.tracklets.empty());
  BOOST_CHECK(parallel.tracklets.empty());
  checkSame(serial, parallel);
}

BOOST_AUTO_TEST_CASE(PerTimeFrameValidationFailureLeavesEdgeArraysZeroFilledNotPartial)
{
  // Edge arrays are cleared before validating normalized measurements.
  // Duplicate cluster IDs below must fail before any edge values are computed,
  // leaving correctly sized, zero-filled arrays rather than partial results.
  auto pool = std::make_shared<BoundedMemoryResource>();
  TimeFrame frame;
  Tracker tracker;
  TrackerTraits traits;
  std::shared_ptr<tbb::task_arena> arena;
  std::vector<ReferenceTrackingParameters> params(1);
  resetDetectorDefaults(params[0], o2::detectors::DetID::ITS);
  params[0].PassFlags.reset();
  params[0].PassFlags.set(IterationStep::FirstPass, IterationStep::RebuildClusterLUT);

  traits.setNThreads(1, arena);
  frame.setBz(Bz);

  const auto orderedSurfaces = identitySurfaces(static_cast<uint16_t>(ITSNLayers));
  const auto catalog = makeCatalog(static_cast<uint16_t>(ITSNLayers), o2::detectors::DetID::ITS, SurfaceKind::Cylinder);
  const SurfaceCatalogView catalogView{catalog.data(), static_cast<uint32_t>(catalog.size())};
  TrackerInitialization configuration;
  configuration.catalog = catalogView;
  configuration.memoryPool = pool;
  configuration.layout = makeDetectorLayout();
  configuration.plan = o2::itsmft::tracking::test::makeTrackingPlan(params[0]);
  BOOST_REQUIRE(tracker.initialize(frame, configuration).ok());
  auto& tf = frame.getScratch();
  const auto& layout = frame.getLayout();
  const auto topologyBuild = deriveTraversalTopology(layout, params[0]);
  BOOST_REQUIRE(topologyBuild.ok());
  const auto layoutView = topologyBuild.topology->getView(layout.getSurfaceCatalog());

  // Same minimal cluster/ROF/mask setup as runFixture(): TimeFrame::initialise()
  // (called unconditionally, before any of this test's induced failure) needs
  // it to size mIndexTables/mClusters correctly, regardless of what this test
  // is actually probing.
  const std::vector<DecodedCluster> decoded{cylinderCluster(3.f, 0.3f, 0), cylinderCluster(3.1f, 0.31f, 0),
                                            cylinderCluster(4.f, 0.4f, 1)};
  std::vector<CompClusterExt> compactClusters;
  std::vector<unsigned char> patterns;
  compactClusters.reserve(decoded.size());
  patterns.reserve(decoded.size() * OnePixelPattern.size());
  for (const auto& cluster : decoded) {
    compactClusters.emplace_back(0, 0, CompCluster::InvalidPatternID, cluster.layer);
    patterns.insert(patterns.end(), OnePixelPattern.begin(), OnePixelPattern.end());
  }
  const std::vector<ROFRecord> rofs{ROFRecord{{100, 5}, 0, 0, static_cast<int>(compactClusters.size())}};
  PrescribedDecoder decoder{o2::detectors::DetID::ITS, SurfaceKind::Cylinder, decoded};
  BOOST_REQUIRE_NO_THROW(test::loadTimeFrameSource(frame, decoder, o2::InteractionRecord{50, 5}, ROFTimingConfig{40, 0, 0, 0},
                                                   compactClusters, patterns, rofs, &dict(), nullptr, o2::detectors::DetID::ITS,
                                                   gsl::span<const LayerId>{orderedSurfaces}, layout.getSurfaceCatalog()));
  auto layer0 = frame.getGlobalMeasurements(LayerId{0});
  BOOST_REQUIRE_EQUAL(layer0.size(), 2u);
  layer0[1].clusterId = layer0[0].clusterId;

  o2::its::LayerTiming layerTiming{};
  layerTiming.mNROFsTF = 1;
  layerTiming.mROFLength = 40;
  o2::its::ROFOverlapTable<ITSNLayers> rofTable;
  for (int layer = 0; layer < ITSNLayers; ++layer) {
    rofTable.defineLayer(layer, layerTiming);
  }
  rofTable.init();
  o2::its::ROFVertexLookupTable<ITSNLayers> vtxTable;
  for (int layer = 0; layer < ITSNLayers; ++layer) {
    vtxTable.defineLayer(layer, layerTiming);
  }
  vtxTable.init();
  o2::its::ROFMaskTable<ITSNLayers> mask{rofTable};
  mask.resetMask();
  for (int layer = 0; layer < ITSNLayers; ++layer) {
    mask.setROFsEnabled(layer, 0, 1, 1);
  }
  frame.setROFViews(RuntimeROFViews{rofTable.getView(), vtxTable.getView(), mask.getView(), {}});

  std::array<gsl::span<const GlobalMeasurement>, MaxLayoutSurfaces> measurementSpans;
  BOOST_CHECK_THROW(TrackerTestAccess::prepare(tracker, frame, 0, measurementSpans), std::invalid_argument);

  const auto topology = layoutView;
  const auto& msAngles = tf.getEdgeMSAngles();
  const auto& phiCuts = tf.getEdgePhiCuts();
  BOOST_REQUIRE_EQUAL(msAngles.size(), static_cast<size_t>(topology.nEdges));
  BOOST_REQUIRE_EQUAL(phiCuts.size(), static_cast<size_t>(topology.nEdges));
  for (int id = 0; id < topology.nEdges; ++id) {
    BOOST_CHECK_EQUAL(msAngles[id], 0.f);
    BOOST_CHECK_EQUAL(phiCuts[id], 0.f);
  }
}

// ---------------------------------------------------------------------------
// Gate 4 Slice 0a (sparse-topology tracklet migration) additions below.
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(ItsIdentityLayoutTrackletsSpanMultipleAdjacentEdgesInOrder)
{
  // Collinear track across 4 barrel layers (z = 0.1 * r for every cluster).
  // Under ITS's default MaxHoles=0 only strictly-adjacent edges exist
  // at all, so this directly proves edge-level tracklet/LUT/order
  // parity across three distinct edges simultaneously -- each
  // resolved through the migrated computeLayerTrackletsForKind() via a
  // fresh mSurfaceToLegacyLayer lookup -- not just the single edge the
  // tests above check, while every non-participating edge (touching
  // layers 4/5/6) stays empty.
  const std::vector<DecodedCluster> clusters{
    cylinderCluster(3.f, 0.3f, 0),
    cylinderCluster(4.f, 0.4f, 1),
    cylinderCluster(5.f, 0.5f, 2),
    cylinderCluster(6.f, 0.6f, 3)};
  const auto snapshot = runFixture<ITSNLayers>(o2::detectors::DetID::ITS, SurfaceKind::Cylinder,
                                               SurfaceKind::Cylinder, clusters, 1);
  // Each edge's expected tanLambda is computed from its own specific
  // (radius, z) pair rather than one shared constant: although every pair
  // shares the same nominal slope (z = 0.1 * r), float subtraction/division
  // of different operand pairs does not generally round to the identical
  // bit pattern even when the mathematical result is the same value.
  constexpr std::array<float, 4> radii{3.f, 4.f, 5.f, 6.f};
  constexpr std::array<float, 4> zs{0.3f, 0.4f, 0.5f, 0.6f};
  const float expectedPhi = o2::gpu::CAMath::ATan2(0.f, -1.f);
  const std::vector<int> expectedLookup{0, 1};

  BOOST_REQUIRE_EQUAL(snapshot.allEdgeFromLayer.size(), snapshot.allTracklets.size());
  BOOST_REQUIRE_EQUAL(snapshot.allEdgeFromLayer.size(), snapshot.allLookups.size());
  bool sawEdge01 = false, sawEdge12 = false, sawEdge23 = false;
  for (size_t id = 0; id < snapshot.allEdgeFromLayer.size(); ++id) {
    const int from = snapshot.allEdgeFromLayer[id];
    const int to = snapshot.allEdgeToLayer[id];
    const bool participates = (from == 0 && to == 1) || (from == 1 && to == 2) || (from == 2 && to == 3);
    if (participates) {
      BOOST_REQUIRE_EQUAL(snapshot.allTracklets[id].size(), 1u);
      const auto& tracklet = snapshot.allTracklets[id].front();
      BOOST_CHECK_EQUAL(tracklet.firstClusterIndex, 0);
      BOOST_CHECK_EQUAL(tracklet.secondClusterIndex, 0);
      const float expectedTanLambda = (zs[from] - zs[to]) / (radii[from] - radii[to]);
      BOOST_CHECK_EQUAL(tracklet.tanLambda, expectedTanLambda);
      BOOST_CHECK_EQUAL(tracklet.phi, expectedPhi);
      BOOST_CHECK_EQUAL_COLLECTIONS(snapshot.allLookups[id].begin(), snapshot.allLookups[id].end(), expectedLookup.begin(), expectedLookup.end());
      sawEdge01 |= (from == 0 && to == 1);
      sawEdge12 |= (from == 1 && to == 2);
      sawEdge23 |= (from == 2 && to == 3);
    } else {
      BOOST_CHECK(snapshot.allTracklets[id].empty());
    }
  }
  BOOST_CHECK(sawEdge01);
  BOOST_CHECK(sawEdge12);
  BOOST_CHECK(sawEdge23);
}

BOOST_AUTO_TEST_CASE(ItsHoleEdgeTrackletResolvesCorrectLegacyLayerEndpoints)
{
  // MaxHoles=1 with layer 1 an allowed hole introduces a (0,2)-skip-1
  // edge whose sparse Edge endpoints are LayerId{0}/
  // LayerId{2} -- a direct, non-adjacent exercise of mSurfaceToLegacyLayer
  // resolving a edge's endpoints correctly, and of hole/skipped-surface
  // behaviour staying identical to the pre-migration code (which read the
  // same fromLayer/toLayer straight off the legacy view). No cluster is
  // placed on layer 1 at all, so only the hole edge can produce a
  // tracklet.
  const std::vector<DecodedCluster> clusters{
    cylinderCluster(3.f, 0.3f, 0),
    cylinderCluster(5.f, 0.5f, 2)};
  const auto snapshot = runFixture<ITSNLayers>(
    o2::detectors::DetID::ITS, SurfaceKind::Cylinder, SurfaceKind::Cylinder, clusters, 1,
    [](ReferenceTrackingParameters& p) {
      p.MaxHoles = 1;
    },
    LayerMask{static_cast<uint16_t>(1u << 1)});

  const float expectedTanLambda = (0.3f - 0.5f) / (3.f - 5.f);
  const float expectedPhi = o2::gpu::CAMath::ATan2(0.f, -1.f);
  bool sawHoleEdge = false;
  BOOST_REQUIRE_EQUAL(snapshot.allEdgeFromLayer.size(), snapshot.allTracklets.size());
  for (size_t id = 0; id < snapshot.allEdgeFromLayer.size(); ++id) {
    const int from = snapshot.allEdgeFromLayer[id];
    const int to = snapshot.allEdgeToLayer[id];
    if (from == 0 && to == 2) {
      sawHoleEdge = true;
      BOOST_REQUIRE_EQUAL(snapshot.allTracklets[id].size(), 1u);
      const auto& tracklet = snapshot.allTracklets[id].front();
      BOOST_CHECK_EQUAL(tracklet.tanLambda, expectedTanLambda);
      BOOST_CHECK_EQUAL(tracklet.phi, expectedPhi);
      const std::vector<int> expectedLookup{0, 1};
      BOOST_CHECK_EQUAL_COLLECTIONS(snapshot.allLookups[id].begin(), snapshot.allLookups[id].end(), expectedLookup.begin(), expectedLookup.end());
    } else {
      BOOST_CHECK(snapshot.allTracklets[id].empty());
    }
  }
  BOOST_CHECK(sawHoleEdge);
}

BOOST_AUTO_TEST_CASE(DenseLayerIdentityIsDerivedFromDescriptorPosition)
{
  const auto surfaces = makeCatalog(static_cast<uint16_t>(ITSNLayers), o2::detectors::DetID::ITS, SurfaceKind::Cylinder);
  const auto layout = DetectorLayout{surfaces};
  BOOST_REQUIRE(layout.valid());
  BOOST_REQUIRE_EQUAL(layout.size(), static_cast<std::size_t>(ITSNLayers));
  for (uint16_t position = 0; position < ITSNLayers; ++position) {
    BOOST_CHECK(&layout[LayerId{position}] == &layout.getLayers()[position]);
  }
}

BOOST_AUTO_TEST_CASE(CombinedCylinderAndDiskLayoutBindsAsOneDisconnectedPlan)
{
  const auto nCylinders = static_cast<uint16_t>(ITSNLayers);
  const auto nDisks = static_cast<uint16_t>(MFTNLayers);
  auto surfaces = makeCatalog(nCylinders, o2::detectors::DetID::ITS, SurfaceKind::Cylinder);
  auto disks = makeCatalog(nDisks, o2::detectors::DetID::MFT, SurfaceKind::Disk);
  surfaces.insert(surfaces.end(), disks.begin(), disks.end());
  DetectorLayoutDefinition definition;
  definition.componentOffsets = {0, nCylinders};
  const auto layout = DetectorLayout{surfaces, std::move(definition)};
  ReferenceTrackingParameters parameters;
  parameters.NLayers = static_cast<int>(layout.size());
  const auto result = deriveTraversalTopology(layout, parameters);
  BOOST_REQUIRE(result.ok());
  BOOST_CHECK_EQUAL(result.topology->edges.size(), static_cast<std::size_t>(nCylinders + nDisks - 2));
}
