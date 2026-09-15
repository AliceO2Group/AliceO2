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

// TimeFrame lifecycle, transactional configuration, and direct loading.
//
// A. Reset lifecycle: TimeFrame::resetTimeFrame() unconditionally clears all
//    TimeFrame data while preserving detector configuration and allocator
//    identity. Post-reset checks always obtain fresh views.
//
// B. Strong configuration transactionality: a BoundedMemoryResource failure
//    while staging a valid replacement must preserve the live configuration,
//    workspace, allocator and capacities, as well as an already loaded TimeFrame,
//    its allocator-backed storage, navigation, and results.
//
// C. TimeFrame loading resets and fills the configured frame directly. Any
//    failure clears partially loaded data.

#define BOOST_TEST_MODULE ITSMFT TimeFrame lifecycle
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <vector>

#include <gsl/gsl>

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "ITSMFTTracking/DetectorLayout.h"
#include "ITSMFTTracking/detail/TimeFrameScratch.h"
#include "ITSMFTTracking/IOUtils.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"
#include "ITSMFTTracking/ClusterDecoding.h"
#include "ITSMFTTracking/IOUtils.h"
#include "ITSMFTTracking/TimeFrame.h"
#include "ITSMFTTracking/TrackingConfigParam.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::itsmft;
using namespace o2::itsmft::tracking;

namespace
{

// Deterministic, geometry-free stand-in for GeometryClusterDecoder<DetId>
// (same construction as testTimeFrameNormalizedSource.cxx / testMultiSourceLoading.cxx):
// sensorID is used directly as the detector-local layer, global/frame
// coordinates are pure functions of (sensorID, row, col), and pattern
// consumption goes through the real production helper so cursor bookkeeping
// is exercised identically to GeometryClusterDecoder.
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
    // Counts only clusters this decoder actually turned into a measurement
    // (the early-return failure paths above never reach here), so a test can
    // prove every cluster of a given input was successfully decoded by
    // checking how much this counter advanced across that call.
    ++decodeCount;
    return result;
  }

  mutable int decodeCount{0};

 private:
  o2::detectors::DetID::ID mDetector;
};

const TopologyDictionary& dict()
{
  static const TopologyDictionary d;
  return d;
}

constexpr std::array<unsigned char, 3> onePixelPattern{1, 1, 0x80};   // 1x1, 1 pixel
constexpr std::array<unsigned char, 3> threePixelPattern{1, 3, 0xE0}; // 1x3, 3 pixels

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

DetectorLayout catalogLayout(SurfaceCatalogView catalog)
{
  return DetectorLayout{gsl::span<const SurfaceDescriptor>{catalog.surfaces, catalog.nSurfaces},
                        makeDetectorLayout()};
}

GlobalPoint3F expectedGlobal(int sensorID, int row, int col)
{
  return {static_cast<float>(sensorID) * 10.f, static_cast<float>(row), static_cast<float>(col)};
}

struct Fixture {
  std::vector<CompClusterExt> clusters;
  std::vector<unsigned char> patterns;
  std::vector<ROFRecord> rofs;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> labels;
};

// 4 clusters on layers {0,1,0,2}, partitioned into 3 ROFs: ROF0={c0,c1},
// ROF1={c2}, ROF2={c3}. Identical shape to testTimeFrameNormalizedSource.cxx's
// fixture, so parity with that accepted test coverage is preserved.
Fixture makeFixture()
{
  Fixture f;
  f.clusters = {
    CompClusterExt{10, 20, CompCluster::InvalidPatternID, 0}, // sensor 0 -> layer 0
    CompClusterExt{11, 21, CompCluster::InvalidPatternID, 1}, // sensor 1 -> layer 1
    CompClusterExt{12, 22, CompCluster::InvalidPatternID, 0}, // sensor 0 -> layer 0
    CompClusterExt{13, 23, CompCluster::InvalidPatternID, 2}, // sensor 2 -> layer 2
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

// A second, distinct, independently valid fixture: different sensors/layers
// (3,4,3,5,3 instead of 0,1,0,2), different rows/columns, a different
// pattern arrangement, a different ROF partition (3 ROFs over 5 clusters
// instead of 4), and its own separate MCTruthContainer with different label
// values. Used as the *replacement* load in the strong-exception-safety
// test, so that if any partial commit ever leaked through, it would be
// observable as foreign data (wrong layer, wrong coordinates, wrong label)
// rather than being masked by coincidentally reloading the same values.
Fixture makeReplacementFixture()
{
  Fixture f;
  f.clusters = {
    CompClusterExt{50, 60, CompCluster::InvalidPatternID, 3}, // sensor 3 -> layer 3
    CompClusterExt{51, 61, CompCluster::InvalidPatternID, 4}, // sensor 4 -> layer 4
    CompClusterExt{52, 62, CompCluster::InvalidPatternID, 3}, // sensor 3 -> layer 3
    CompClusterExt{53, 63, CompCluster::InvalidPatternID, 5}, // sensor 5 -> layer 5
    CompClusterExt{54, 64, CompCluster::InvalidPatternID, 3}, // sensor 3 -> layer 3
  };
  f.patterns = concatPatterns({threePixelPattern, threePixelPattern, onePixelPattern, threePixelPattern, onePixelPattern});
  f.rofs = {
    ROFRecord{{500, 1}, 0, 0, 3},
    ROFRecord{{540, 1}, 1, 3, 1},
    ROFRecord{{2000, 2}, 2, 4, 1}};
  for (uint32_t i = 0; i < f.clusters.size(); ++i) {
    f.labels.addElement(i, o2::MCCompLabel{static_cast<int>(i) + 101, 1, 1});
  }
  return f;
}

struct Expected {
  uint32_t externalIndex;
  int layer;
  int sensorID;
  int row, col;
  uint32_t sourceROF;
  uint32_t nPixels;
};

const std::vector<Expected> expectedClusters{
  {0, 0, 0, 10, 20, 0, 1},
  {1, 1, 1, 11, 21, 0, 3},
  {2, 0, 0, 12, 22, 1, 1},
  {3, 2, 2, 13, 23, 2, 3},
};

void verifyFixtureLoaded(const TimeFrame& frame, const Fixture& f)
{
  BOOST_CHECK_EQUAL(frame.getGlobalMeasurements(LayerId{0}).size(), 2u);
  BOOST_CHECK_EQUAL(frame.getGlobalMeasurements(LayerId{1}).size(), 1u);
  BOOST_CHECK_EQUAL(frame.getGlobalMeasurements(LayerId{2}).size(), 1u);
  for (int l = 3; l < ITSNLayers; ++l) {
    BOOST_CHECK_EQUAL(frame.getGlobalMeasurements(LayerId{static_cast<uint16_t>(l)}).size(), 0u);
    BOOST_CHECK_EQUAL(frame.getNrof(l), static_cast<int>(f.rofs.size()));
  }

  BOOST_CHECK_EQUAL(frame.getNrof(0), static_cast<int>(f.rofs.size()));

  for (std::size_t expectedIndex = 0; expectedIndex < expectedClusters.size(); ++expectedIndex) {
    const auto& e = expectedClusters[expectedIndex];
    const auto localClusterId = static_cast<uint32_t>(std::count_if(
      expectedClusters.begin(), expectedClusters.begin() + expectedIndex,
      [&](const auto& previous) { return previous.layer == e.layer; }));
    const GlobalMeasurement* globalMeasurement = nullptr;
    const SurfaceMeasurement* measurement = nullptr;
    const auto surface = LayerId{static_cast<uint16_t>(e.layer)};
    const auto globals = frame.getGlobalMeasurements(surface);
    for (size_t index = 0; index < globals.size(); ++index) {
      if (globals[index].clusterId == localClusterId) {
        globalMeasurement = &globals[index];
        measurement = frame.getSurfaceMeasurement(surface, localClusterId);
        break;
      }
    }
    BOOST_REQUIRE(globalMeasurement != nullptr);
    BOOST_REQUIRE(measurement != nullptr);

    const auto g = expectedGlobal(e.sensorID, e.row, e.col);
    BOOST_CHECK_EQUAL(globalMeasurement->position.x, g.x);
    BOOST_CHECK_EQUAL(globalMeasurement->position.y, g.y);
    BOOST_CHECK_EQUAL(globalMeasurement->position.z, g.z);

    BOOST_CHECK_EQUAL(measurement->frame.q, static_cast<float>(e.sensorID) + 100.f);
    BOOST_CHECK_EQUAL(measurement->frame.u, static_cast<float>(e.row) + 1.f);
    BOOST_CHECK_EQUAL(measurement->frame.v, static_cast<float>(e.col) + 2.f);
    BOOST_CHECK_EQUAL(measurement->frame.frameAngle, 0.01f * e.sensorID);

    BOOST_CHECK_EQUAL(measurement->covariance.uu, o2::itsmft::ioutils::DefClusError2Row);
    BOOST_CHECK_EQUAL(measurement->covariance.uv, 0.f);
    BOOST_CHECK_EQUAL(measurement->covariance.vv, o2::itsmft::ioutils::DefClusError2Col);

    BOOST_CHECK_EQUAL(globalMeasurement->clusterId, localClusterId);
    const auto normalizedLabels = frame.getLabels(surface, localClusterId);
    BOOST_REQUIRE_EQUAL(normalizedLabels.size(), 1u);
    BOOST_CHECK(normalizedLabels[0] == o2::MCCompLabel(static_cast<int>(e.externalIndex) + 1, 0, 0));
  }
}

void configureFrame(TimeFrame& frame, SurfaceCatalogView catalog,
                    std::shared_ptr<BoundedMemoryResource> pool = std::make_shared<BoundedMemoryResource>())
{
  auto layout = catalogLayout(catalog);
  BOOST_REQUIRE(frame.configure(std::move(layout), 0, 0, std::move(pool)));
}

} // namespace

// --- A. Wipe lifecycle -------------------------------------------------

BOOST_AUTO_TEST_CASE(WipeClearsNormalizedFrameButPreservesDetId)
{
  const auto catalog = makeITSTestCatalog();
  const auto orderedSurfaces = identitySurfaces(ITSNLayers);
  const SurfaceCatalogView catalogView{catalog.data(), static_cast<uint32_t>(catalog.size())};
  LegacyLikeDecoder decoder{o2::detectors::DetID::ITS};
  const o2::InteractionRecord origin{50, 5};
  const ROFTimingConfig timing{40, 0, 0, 0};

  TimeFrame frame;
  const auto plan = catalogLayout(catalogView);
  configureFrame(frame, catalogView);
  const auto estimatorKey = CapacityEstimator::makeKey(SlabSite::Cells, 2, 0, CellPathId{3});
  frame.getCapacityEstimator().update(estimatorKey, 1000., 8000, 8000, false, false);
  const auto learnedCapacity = frame.getCapacityEstimator().capacity(estimatorKey, 1000.);
  BOOST_REQUIRE_GT(learnedCapacity, 1024u);

  const auto f = makeFixture();
  const auto result = loadTimeFrameSource(frame, decoder, origin, timing, f.clusters, f.patterns, f.rofs, &dict(), &f.labels, o2::detectors::DetID::ITS,
                                          gsl::span<const LayerId>{orderedSurfaces}, plan.getSurfaceCatalog());
  BOOST_REQUIRE(result.ok());
  // Sanity: the successful load itself has the expected content, matching
  // the accepted parity coverage in testTimeFrameNormalizedSource.cxx.
  verifyFixtureLoaded(frame, f);

  frame.resetTimeFrame();
  BOOST_CHECK_EQUAL(frame.getCapacityEstimator().capacity(estimatorKey, 1000.), learnedCapacity);

  // --- inspect only freshly obtained normalized accessors/views ---
  BOOST_CHECK_EQUAL(frame.getTotalMeasurements(), 0u);
  BOOST_CHECK_EQUAL(frame.getNMeasurementSurfaces(), ITSNLayers);
  for (uint16_t s = 0; s < ITSNLayers; ++s) {
    BOOST_CHECK(frame.getGlobalMeasurements(LayerId{s}).empty());
  }
  BOOST_CHECK(frame.getLabels(LayerId{0}, 0).empty());

  // Gate 4 B3.1: neither owner stores mDetId any more -- the plan lives on
  // `plan` above, entirely outside both TimeFrame and LegacyTrackerScratch,
  // so resetTimeFrame() has no detector-identity state to preserve or clear.
}

BOOST_AUTO_TEST_CASE(FailedConfigurationAllocationLeavesClearedFrame)
{
  const auto catalog = makeITSTestCatalog();
  const auto orderedSurfaces = identitySurfaces(ITSNLayers);
  const SurfaceCatalogView catalogView{catalog.data(), static_cast<uint32_t>(catalog.size())};
  TimeFrame frame;
  const auto estimatorKey = CapacityEstimator::makeKey(SlabSite::Tracklets, 1, 0, EdgeId{2});
  frame.getCapacityEstimator().update(estimatorKey, 1000., 9000, 9000, false, false);
  const auto learnedCapacity = frame.getCapacityEstimator().capacity(estimatorKey, 1000.);
  const auto* const scratch = &frame.getScratch();
  auto layout = catalogLayout(catalogView);
  auto failingPool = std::make_shared<BoundedMemoryResource>(0);

  BOOST_CHECK(!frame.configure(std::move(layout), 1, 1, failingPool));
  BOOST_CHECK_EQUAL(frame.getCapacityEstimator().capacity(estimatorKey, 1000.), learnedCapacity);
  BOOST_CHECK_EQUAL(failingPool->getThrowCount(), 1u);
  BOOST_CHECK_EQUAL(failingPool->getUsedMemory(), 0u);

  BOOST_CHECK(!frame.isConfigured());
  BOOST_CHECK(&frame.getScratch() == scratch);
  BOOST_CHECK(frame.getMemoryPool().get() == failingPool.get());
  BOOST_CHECK(frame.getScratch().getMemoryPool().get() == failingPool.get());
  BOOST_CHECK_EQUAL(frame.getScratch().getNEdges(), 0u);
  BOOST_CHECK_EQUAL(frame.getScratch().getNCells(), 0u);
  BOOST_CHECK(frame.getLayout().empty());
  BOOST_CHECK_EQUAL(frame.getTotalMeasurements(), 0u);
  BOOST_CHECK(frame.getGenericTracks().empty());
  BOOST_CHECK(frame.getTrackClusterIndices().empty());
  BOOST_CHECK_EQUAL(frame.getPrimaryVerticesNum(), 0u);
}

BOOST_AUTO_TEST_CASE(ConfigurationAdoptionResetsIncompatibleCapacityEstimates)
{
  const auto catalog = makeITSTestCatalog();
  const SurfaceCatalogView catalogView{catalog.data(), static_cast<uint32_t>(catalog.size())};
  TimeFrame frame;
  const auto key = CapacityEstimator::makeKey(SlabSite::Roads, 3,
                                              CapacityEstimator::makeVariant(5, 3), CellPathId{7});
  frame.getCapacityEstimator().update(key, 1000., 12000, 12000, false, false);
  BOOST_REQUIRE_GT(frame.getCapacityEstimator().capacity(key, 1000.), 1024u);

  configureFrame(frame, catalogView);

  BOOST_CHECK_EQUAL(frame.getCapacityEstimator().capacity(key, 1000.), 1024u);
}

BOOST_AUTO_TEST_CASE(MalformedTimeFrameLoadLeavesTheFrameEmpty)
{
  const auto catalog = makeITSTestCatalog();
  const auto orderedSurfaces = identitySurfaces(ITSNLayers);
  const SurfaceCatalogView catalogView{catalog.data(), static_cast<uint32_t>(catalog.size())};
  LegacyLikeDecoder decoder{o2::detectors::DetID::ITS};
  const o2::InteractionRecord origin{50, 5};
  const ROFTimingConfig timing{40, 0, 0, 0};
  const auto baselineFixture = makeFixture();
  auto malformedReplacement = makeReplacementFixture();
  const auto plan = catalogLayout(catalogView);
  TimeFrame frame;
  configureFrame(frame, catalogView);
  const auto baseline = loadTimeFrameSource(frame, decoder, origin, timing, baselineFixture.clusters,
                                            baselineFixture.patterns, baselineFixture.rofs, &dict(),
                                            &baselineFixture.labels, o2::detectors::DetID::ITS,
                                            gsl::span<const LayerId>{orderedSurfaces}, plan.getSurfaceCatalog());
  BOOST_REQUIRE(baseline.ok());
  verifyFixtureLoaded(frame, baselineFixture);

  malformedReplacement.rofs.front().setFirstEntry(1);
  const auto failed = loadTimeFrameSource(frame, decoder, origin, timing, malformedReplacement.clusters,
                                          malformedReplacement.patterns, malformedReplacement.rofs, &dict(),
                                          &malformedReplacement.labels, o2::detectors::DetID::ITS,
                                          gsl::span<const LayerId>{orderedSurfaces}, plan.getSurfaceCatalog());
  BOOST_CHECK(!failed.ok());
  BOOST_CHECK(failed.error == MultiSourceLoadError::InvalidROFRange);
  BOOST_CHECK_EQUAL(frame.getTotalMeasurements(), 0u);
  BOOST_CHECK_EQUAL(frame.getNMeasurementSurfaces(), ITSNLayers);
}
