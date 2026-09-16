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

#define BOOST_TEST_MODULE ITSMFT MultiSourceLoading
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <limits>
#include <memory>
#include <vector>

#include <gsl/gsl>

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "ITSMFTTracking/DetectorLayout.h"
#include "TrackingParameterTestSupport.h"
#include "ITSMFTTracking/TimeFrame.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::itsmft;
using namespace o2::itsmft::tracking;

namespace
{

// Host-only test decoder (no geometry singletons): maps a chip ID to a
// detector-local layer via an explicit table, and reuses the same pattern
// consumption path (extractClusterData) that the production decoder uses,
// so pattern-cursor bookkeeping is exercised identically.
enum class Corruption {
  None,
  NegativeLayer,
  LayerOutOfRange
};

class FakeClusterDecoder
{
 public:
  FakeClusterDecoder(o2::detectors::DetID::ID detector, std::vector<int> sensorToLayer, bool disk, Corruption corruption = Corruption::None)
    : mDetector(detector), mSensorToLayer(std::move(sensorToLayer)), mDisk(disk), mCorruption(corruption)
  {
  }

  o2::itsmft::tracking::DecodedCluster decode(
    const CompClusterExt& cluster,
    gsl::span<const unsigned char>::iterator& patterns,
    const TopologyDictionary* dict,
    uint32_t,
    bool) const
  {
    if (mCorruption == Corruption::NegativeLayer) {
      o2::itsmft::tracking::DecodedCluster result;
      result.layer = -1;
      return result;
    }
    if (mCorruption == Corruption::LayerOutOfRange) {
      o2::itsmft::tracking::DecodedCluster result;
      result.layer = std::numeric_limits<int>::max();
      return result;
    }

    const auto clusterData = o2::itsmft::ioutils::extractClusterData(cluster, patterns, dict);
    o2::itsmft::tracking::DecodedCluster result;
    const auto sensorID = cluster.getSensorID();
    const int layer = (sensorID >= 0 && static_cast<size_t>(sensorID) < mSensorToLayer.size()) ? mSensorToLayer[sensorID] : -1;
    auto& decoded = result;
    decoded.global = {static_cast<float>(sensorID), static_cast<float>(cluster.getRow()), static_cast<float>(cluster.getCol())};
    decoded.cylinderFrame = {10.f + sensorID, 1.f, 2.f, 0.1f};
    decoded.rowColumnCovariance = {clusterData.sig2Row, 0.f, clusterData.sig2Col};
    decoded.nPixels = clusterData.nPixels;
    decoded.layer = layer;
    return result;
  }

 private:
  o2::detectors::DetID::ID mDetector;
  std::vector<int> mSensorToLayer;
  bool mDisk;
  Corruption mCorruption;
};

// Geometry-free decoder used only to exercise the normalized loader's
// dictionary/common/group/explicit pattern contract. Pattern ID 0 represents
// a common dictionary entry (no explicit bytes), pattern ID 1 represents a
// grouped dictionary entry (explicit bytes required), and InvalidPatternID
// represents an ordinary explicit pattern.
class PatternContractDecoder
{
 public:
  o2::itsmft::tracking::DecodedCluster decode(
    const CompClusterExt& cluster,
    gsl::span<const unsigned char>::iterator& patterns,
    const TopologyDictionary* dictionary,
    uint32_t,
    bool) const
  {
    o2::itsmft::tracking::DecodedCluster result;
    if (dictionary == nullptr) {
      throw std::runtime_error("Cluster dictionary is not available");
    }
    uint32_t nPixels = 1;
    if (cluster.getPatternID() != 0) {
      const o2::itsmft::ClusterPattern pattern{patterns};
      nPixels = static_cast<uint32_t>(pattern.getNPixels());
    }

    auto& decoded = result;
    decoded.global = {1.f, 2.f, 3.f};
    decoded.cylinderFrame = {4.f, 5.f, 6.f, 0.f};
    decoded.rowColumnCovariance = {0.1f, 0.f, 0.2f};
    decoded.nPixels = nPixels;
    decoded.layer = 0;
    return result;
  }
};

struct BuiltLayout {
  DetectorLayout layout;
  std::vector<SurfaceDescriptor> surfaces;

  bool valid() const noexcept { return layout.valid(); }
  SurfaceCatalogView getCatalog() const noexcept
  {
    return layout.getSurfaceCatalog();
  }
};

// 4-surface disconnected ITS(cylinder)+MFT(disk) layout: surfaces {0,1} are
// ITS layers 0/1, surfaces {2,3} are MFT layers 0/1. No edges are
// needed to exercise loading.
BuiltLayout makeCombinedLayout()
{
  std::vector<SurfaceDescriptor> surfaces;
  surfaces.push_back(SurfaceDescriptor{0, static_cast<uint8_t>(o2::detectors::DetID::ITS), SurfaceKind::Cylinder});
  surfaces.push_back(SurfaceDescriptor{1, static_cast<uint8_t>(o2::detectors::DetID::ITS), SurfaceKind::Cylinder});
  surfaces.push_back(SurfaceDescriptor{0, static_cast<uint8_t>(o2::detectors::DetID::MFT), SurfaceKind::Disk});
  surfaces.push_back(SurfaceDescriptor{1, static_cast<uint8_t>(o2::detectors::DetID::MFT), SurfaceKind::Disk});
  DetectorLayoutDefinition definition;
  definition.componentOffsets = {0, 2};
  return BuiltLayout{DetectorLayout{surfaces, std::move(definition)}, std::move(surfaces)};
}

void configureFrame(TimeFrame& frame, const BuiltLayout& built)
{
  DetectorLayoutDefinition definition;
  const auto& layout = built.layout;
  definition.componentOffsets.assign(layout.getComponentOffsets().begin(), layout.getComponentOffsets().end());
  definition.holeLayers = layout.getHoleLayers();
  const auto catalog = layout.getSurfaceCatalog();
  BOOST_REQUIRE(frame.configure(DetectorLayout{gsl::span<const SurfaceDescriptor>{catalog.surfaces, catalog.nSurfaces},
                                               std::move(definition)},
                                0, 0, std::make_shared<BoundedMemoryResource>()));
}

// One explicit (non-grouped) 1-pixel pattern: rowSpan=1, colSpan=1, one
// bitmap byte. Three bytes are consumed per cluster.
constexpr std::array<unsigned char, 3> onePixelPattern{1, 1, 0x80};

std::vector<unsigned char> makePatternBytes(size_t nClusters)
{
  std::vector<unsigned char> bytes;
  bytes.reserve(nClusters * onePixelPattern.size());
  for (size_t i = 0; i < nClusters; ++i) {
    bytes.insert(bytes.end(), onePixelPattern.begin(), onePixelPattern.end());
  }
  return bytes;
}

const TopologyDictionary& dict()
{
  static const TopologyDictionary d;
  return d;
}

const std::array<LayerId, 2> itsLayerToSurface{LayerId{0}, LayerId{1}};
const std::array<LayerId, 2> mftLayerToSurface{LayerId{2}, LayerId{3}};
const std::array<LayerId, 1> firstITSSurface{LayerId{0}};
const std::array<LayerId, 1> secondITSSurface{LayerId{1}};
const std::array<LayerId, 1> firstMFTSurface{LayerId{2}};

} // namespace

BOOST_AUTO_TEST_CASE(SingleITSSourceLoadsIntoExpectedSurfaces)
{
  const auto layout = makeCombinedLayout();
  BOOST_REQUIRE(layout.valid());

  const std::vector<CompClusterExt> clusters{
    {10, 20, CompCluster::InvalidPatternID, 0}, // sensor 0 -> layer 0
    {11, 21, CompCluster::InvalidPatternID, 1}, // sensor 1 -> layer 1
  };
  const auto patterns = makePatternBytes(clusters.size());
  const std::vector<ROFRecord> rofs{ROFRecord{{0, 0}, 0, 0, 2}};

  FakeClusterDecoder decoder{o2::detectors::DetID::ITS, {0, 1}, false};
  test::TestClusterSourceInput src;
  src.id = ClusterSourceId{0};
  src.detector = o2::detectors::DetID::ITS;
  src.clusters = clusters;
  src.patterns = patterns;
  src.rofs = rofs;
  src.dictionary = &dict();
  src.layerToSurface = itsLayerToSurface;
  src.timing = ROFTimingConfig{40, 0, 0, 0};
  src.setDecoder(decoder);

  TimeFrame frame;
  configureFrame(frame, layout);
  std::vector<std::vector<uint32_t>> externalIndicesBySurface;
  BOOST_REQUIRE_NO_THROW(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(&src, 1), {0, 0},
                                           &externalIndicesBySurface));

  BOOST_CHECK_EQUAL(frame.getGlobalMeasurements(LayerId{0}).size(), 1u);
  BOOST_CHECK_EQUAL(frame.getGlobalMeasurements(LayerId{1}).size(), 1u);
  BOOST_CHECK_EQUAL(frame.getGlobalMeasurements(LayerId{2}).size(), 0u);
  BOOST_CHECK_EQUAL(externalIndicesBySurface[0][0], 0u);
}

BOOST_AUTO_TEST_CASE(InvalidTimingConfigurationIsReportedWithBuildErrorDetail)
{
  // Invalid timing must throw with source and ROF context before decoding.
  const auto layout = makeCombinedLayout();
  BOOST_REQUIRE(layout.valid());

  const std::vector<CompClusterExt> clusters{{1, 1, CompCluster::InvalidPatternID, 0}};
  const auto patterns = makePatternBytes(clusters.size());
  const std::vector<ROFRecord> rofs{ROFRecord{{0, 0}, 0, 0, 1}};

  FakeClusterDecoder decoder{o2::detectors::DetID::ITS, {0}, false};
  test::TestClusterSourceInput src;
  src.id = ClusterSourceId{0};
  src.detector = o2::detectors::DetID::ITS;
  src.clusters = clusters;
  src.patterns = patterns;
  src.rofs = rofs;
  src.dictionary = &dict();
  src.layerToSurface = itsLayerToSurface;
  src.timing = ROFTimingConfig{0, 0, 0, 0}; // rofLength <= 0
  src.setDecoder(decoder);

  TimeFrame frame;
  configureFrame(frame, layout);
  BOOST_CHECK_EXCEPTION(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(&src, 1), {0, 0}), std::runtime_error, [](const std::runtime_error& error) { return std::string(error.what()).find("Invalid ROF timing: source=0 rof=0") != std::string::npos; });
  BOOST_CHECK_EQUAL(frame.getTotalMeasurements(), 0u);
}

BOOST_AUTO_TEST_CASE(SingleMFTSourceLoadsIntoExpectedSurfaces)
{
  const auto layout = makeCombinedLayout();
  BOOST_REQUIRE(layout.valid());

  const std::vector<CompClusterExt> clusters{
    {5, 6, CompCluster::InvalidPatternID, 0},
    {7, 8, CompCluster::InvalidPatternID, 1},
  };
  const auto patterns = makePatternBytes(clusters.size());
  const std::vector<ROFRecord> rofs{ROFRecord{{0, 0}, 0, 0, 2}};

  FakeClusterDecoder decoder{o2::detectors::DetID::MFT, {0, 1}, true};
  test::TestClusterSourceInput src;
  src.id = ClusterSourceId{0};
  src.detector = o2::detectors::DetID::MFT;
  src.clusters = clusters;
  src.patterns = patterns;
  src.rofs = rofs;
  src.dictionary = &dict();
  src.layerToSurface = mftLayerToSurface;
  src.timing = ROFTimingConfig{40, 0, 0, 0};
  src.setDecoder(decoder);

  TimeFrame frame;
  configureFrame(frame, layout);
  std::vector<std::vector<uint32_t>> externalIndicesBySurface;
  BOOST_REQUIRE_NO_THROW(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(&src, 1), {0, 0},
                                           &externalIndicesBySurface));

  BOOST_CHECK_EQUAL(frame.getGlobalMeasurements(LayerId{2}).size(), 1u);
  BOOST_CHECK_EQUAL(frame.getGlobalMeasurements(LayerId{3}).size(), 1u);
  BOOST_CHECK_EQUAL(externalIndicesBySurface[2][0], 0u);
}

BOOST_AUTO_TEST_CASE(CombinedITSAndMFTSourcesLoadTogether)
{
  const auto layout = makeCombinedLayout();
  BOOST_REQUIRE(layout.valid());

  const std::vector<CompClusterExt> itsClusters{{1, 1, CompCluster::InvalidPatternID, 0},
                                                {2, 2, CompCluster::InvalidPatternID, 1},
                                                {3, 3, CompCluster::InvalidPatternID, 0}};
  const auto itsPatterns = makePatternBytes(itsClusters.size());
  const std::vector<ROFRecord> itsRofs{ROFRecord{{0, 0}, 0, 0, 2},
                                       ROFRecord{{40, 0}, 1, 2, 0},
                                       ROFRecord{{80, 0}, 2, 2, 1}};
  FakeClusterDecoder itsDecoder{o2::detectors::DetID::ITS, {0, 1}, false};

  const std::vector<CompClusterExt> mftClusters{{2, 2, CompCluster::InvalidPatternID, 0}};
  const auto mftPatterns = makePatternBytes(mftClusters.size());
  const std::vector<ROFRecord> mftRofs{ROFRecord{{0, 0}, 0, 0, 1}};
  FakeClusterDecoder mftDecoder{o2::detectors::DetID::MFT, {1}, true}; // sensor 0 -> layer 1 -> surface 2

  const std::array<LayerId, 2> itsMapping{LayerId{1}, LayerId{0}};
  const std::array<LayerId, 2> mftMapping{LayerId{3}, LayerId{2}};
  std::array<test::TestClusterSourceInput, 2> sources{};
  sources[0].id = ClusterSourceId{0};
  sources[0].detector = o2::detectors::DetID::ITS;
  sources[0].clusters = itsClusters;
  sources[0].patterns = itsPatterns;
  sources[0].rofs = itsRofs;
  sources[0].dictionary = &dict();
  sources[0].layerToSurface = itsMapping;
  sources[0].timing = ROFTimingConfig{40, 0, 0, 0};
  sources[0].setDecoder(itsDecoder);

  sources[1].id = ClusterSourceId{1};
  sources[1].detector = o2::detectors::DetID::MFT;
  sources[1].clusters = mftClusters;
  sources[1].patterns = mftPatterns;
  sources[1].rofs = mftRofs;
  sources[1].dictionary = &dict();
  sources[1].layerToSurface = mftMapping;
  sources[1].timing = ROFTimingConfig{50, 0, 0, 0};
  sources[1].setDecoder(mftDecoder);

  TimeFrame frame;
  configureFrame(frame, layout);
  BOOST_REQUIRE_NO_THROW(test::loadTimeFrameSources(frame, sources, layout.getCatalog(), {0, 0}));

  BOOST_CHECK_EQUAL(frame.getGlobalMeasurements(LayerId{0}).size(), 1u);
  BOOST_CHECK_EQUAL(frame.getGlobalMeasurements(LayerId{1}).size(), 2u);
  BOOST_CHECK_EQUAL(frame.getGlobalMeasurements(LayerId{2}).size(), 1u);
  const std::array<std::vector<int>, 4> expectedBoundaries{{{0, 1, 1, 1}, {0, 1, 1, 2}, {0, 1}, {0, 0}}};
  for (int surface = 0; surface < 4; ++surface) {
    const auto boundaries = frame.getROFrameClusters(surface);
    BOOST_CHECK_EQUAL_COLLECTIONS(boundaries.begin(), boundaries.end(), expectedBoundaries[surface].begin(), expectedBoundaries[surface].end());
    BOOST_CHECK_EQUAL(frame.getROFLocalLayer(surface), 1 - surface % 2);
  }

  // A missing mapped layer is rejected before any source is decoded.
  sources[1].layerToSurface = firstMFTSurface;
  BOOST_CHECK_THROW(test::loadTimeFrameSources(frame, sources, layout.getCatalog(), {0, 0}), std::runtime_error);
  BOOST_CHECK_EQUAL(frame.getTotalMeasurements(), 0u);
}

BOOST_AUTO_TEST_CASE(TwoSourcesCannotOwnTheSameSurface)
{
  const auto layout = makeCombinedLayout();
  BOOST_REQUIRE(layout.valid());

  const std::vector<CompClusterExt> clustersA{{1, 1, CompCluster::InvalidPatternID, 0}};
  const std::vector<CompClusterExt> clustersB{{2, 2, CompCluster::InvalidPatternID, 0}};
  const auto patternsA = makePatternBytes(clustersA.size());
  const auto patternsB = makePatternBytes(clustersB.size());
  const std::vector<ROFRecord> rofsA{ROFRecord{{0, 0}, 0, 0, 1}};
  const std::vector<ROFRecord> rofsB{ROFRecord{{0, 0}, 0, 0, 1}};

  FakeClusterDecoder decoderA{o2::detectors::DetID::ITS, {0}, false};
  FakeClusterDecoder decoderB{o2::detectors::DetID::ITS, {0}, false};

  std::array<test::TestClusterSourceInput, 2> sources{};
  sources[0].id = ClusterSourceId{0};
  sources[0].detector = o2::detectors::DetID::ITS;
  sources[0].clusters = clustersA;
  sources[0].patterns = patternsA;
  sources[0].rofs = rofsA;
  sources[0].dictionary = &dict();
  sources[0].layerToSurface = itsLayerToSurface;
  sources[0].timing = ROFTimingConfig{40, 0, 0, 0};
  sources[0].setDecoder(decoderA);

  sources[1].id = ClusterSourceId{1};
  sources[1].detector = o2::detectors::DetID::ITS;
  sources[1].clusters = clustersB;
  sources[1].patterns = patternsB;
  sources[1].rofs = rofsB;
  sources[1].dictionary = &dict();
  sources[1].layerToSurface = itsLayerToSurface;
  sources[1].timing = ROFTimingConfig{40, 0, 0, 0};
  sources[1].setDecoder(decoderB);

  TimeFrame frame;
  configureFrame(frame, layout);
  BOOST_CHECK_EXCEPTION(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(sources), {0, 0}), std::runtime_error, [](const std::runtime_error& error) { return std::string(error.what()).find("Invalid source-to-surface layer mapping") != std::string::npos; });
  BOOST_CHECK_EQUAL(frame.getTotalMeasurements(), 0u);
}

BOOST_AUTO_TEST_CASE(IdenticalExternalIndicesInDifferentSourcesDoNotCollide)
{
  const auto layout = makeCombinedLayout();
  BOOST_REQUIRE(layout.valid());

  const std::vector<CompClusterExt> clustersA{{1, 1, CompCluster::InvalidPatternID, 0}}; // external index 0
  const std::vector<CompClusterExt> clustersB{{2, 2, CompCluster::InvalidPatternID, 0}}; // external index 0 too
  const auto patternsA = makePatternBytes(clustersA.size());
  const auto patternsB = makePatternBytes(clustersB.size());
  const std::vector<ROFRecord> rofsA{ROFRecord{{0, 0}, 0, 0, 1}};
  const std::vector<ROFRecord> rofsB{ROFRecord{{0, 0}, 0, 0, 1}};

  o2::dataformats::MCTruthContainer<o2::MCCompLabel> labelsA;
  labelsA.addElement(0, o2::MCCompLabel{1, 0, 0});
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> labelsB;
  labelsB.addElement(0, o2::MCCompLabel{2, 0, 0});

  FakeClusterDecoder decoderA{o2::detectors::DetID::ITS, {0}, false};
  FakeClusterDecoder decoderB{o2::detectors::DetID::ITS, {0}, false};

  std::array<test::TestClusterSourceInput, 2> sources{};
  sources[0].id = ClusterSourceId{0};
  sources[0].detector = o2::detectors::DetID::ITS;
  sources[0].clusters = clustersA;
  sources[0].patterns = patternsA;
  sources[0].rofs = rofsA;
  sources[0].dictionary = &dict();
  sources[0].labels = &labelsA;
  sources[0].layerToSurface = firstITSSurface;
  sources[0].timing = ROFTimingConfig{40, 0, 0, 0};
  sources[0].setDecoder(decoderA);

  sources[1].id = ClusterSourceId{1};
  sources[1].detector = o2::detectors::DetID::ITS;
  sources[1].clusters = clustersB;
  sources[1].patterns = patternsB;
  sources[1].rofs = rofsB;
  sources[1].dictionary = &dict();
  sources[1].labels = &labelsB;
  sources[1].layerToSurface = secondITSSurface;
  sources[1].timing = ROFTimingConfig{40, 0, 0, 0};
  sources[1].setDecoder(decoderB);

  TimeFrame frame;
  configureFrame(frame, layout);
  BOOST_REQUIRE_NO_THROW(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(sources), {0, 0}));

  const auto onSurfaceZero = frame.getGlobalMeasurements(LayerId{0});
  BOOST_REQUIRE_EQUAL(onSurfaceZero.size(), 1u);
  BOOST_REQUIRE_EQUAL(frame.getGlobalMeasurements(LayerId{1}).size(), 1u);
  BOOST_CHECK_EQUAL(onSurfaceZero[0].clusterId, 0u);
  BOOST_CHECK_EQUAL(frame.getGlobalMeasurements(LayerId{1})[0].clusterId, 0u);

  const auto labelSpanA = frame.getLabels(LayerId{0}, 0);
  const auto labelSpanB = frame.getLabels(LayerId{1}, 0);
  BOOST_REQUIRE_EQUAL(labelSpanA.size(), 1u);
  BOOST_REQUIRE_EQUAL(labelSpanB.size(), 1u);
  BOOST_CHECK(labelSpanA[0] != labelSpanB[0]);
}

BOOST_AUTO_TEST_CASE(OriginalClusterIdResolvesLabelsAndCompactGlobal)
{
  const auto layout = makeCombinedLayout();
  BOOST_REQUIRE(layout.valid());

  const std::vector<CompClusterExt> clusters{{1, 1, CompCluster::InvalidPatternID, 0}};
  const auto patterns = makePatternBytes(clusters.size());
  const std::vector<ROFRecord> rofs{ROFRecord{{0, 0}, 0, 0, 1}};

  o2::dataformats::MCTruthContainer<o2::MCCompLabel> labels;
  labels.addElement(0, o2::MCCompLabel{1, 0, 0});

  FakeClusterDecoder decoder{o2::detectors::DetID::ITS, {0}, false};
  test::TestClusterSourceInput src;
  src.id = ClusterSourceId{0};
  src.detector = o2::detectors::DetID::ITS;
  src.clusters = clusters;
  src.patterns = patterns;
  src.rofs = rofs;
  src.dictionary = &dict();
  src.labels = &labels;
  src.layerToSurface = itsLayerToSurface;
  src.timing = ROFTimingConfig{40, 0, 0, 0};
  src.setDecoder(decoder);

  TimeFrame frame;
  configureFrame(frame, layout);
  BOOST_REQUIRE_NO_THROW(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(&src, 1), {0, 0}));

  constexpr uint32_t clusterId = 0;
  const auto labelPlain = frame.getLabels(LayerId{0}, clusterId);
  BOOST_REQUIRE_EQUAL(labelPlain.size(), 1u);

  // The sorted global value carries only the stable source-local ID.
  const auto measurement = frame.getGlobalMeasurements(LayerId{0})[0];
  BOOST_CHECK_EQUAL(measurement.clusterId, clusterId);
}

BOOST_AUTO_TEST_CASE(IndependentROFCountsAcrossSourcesAreAllowed)
{
  const auto layout = makeCombinedLayout();
  BOOST_REQUIRE(layout.valid());

  // Source A: 3 ROFs of 1 cluster each. Source B: 1 ROF of 1 cluster.
  const std::vector<CompClusterExt> clustersA{
    {1, 1, CompCluster::InvalidPatternID, 0},
    {2, 2, CompCluster::InvalidPatternID, 0},
    {3, 3, CompCluster::InvalidPatternID, 0}};
  const auto patternsA = makePatternBytes(clustersA.size());
  const std::vector<ROFRecord> rofsA{
    ROFRecord{{0, 0}, 0, 0, 1},
    ROFRecord{{40, 0}, 1, 1, 1},
    ROFRecord{{80, 0}, 2, 2, 1}};

  const std::vector<CompClusterExt> clustersB{{4, 4, CompCluster::InvalidPatternID, 0}};
  const auto patternsB = makePatternBytes(clustersB.size());
  const std::vector<ROFRecord> rofsB{ROFRecord{{0, 0}, 0, 0, 1}};

  FakeClusterDecoder decoderA{o2::detectors::DetID::ITS, {0}, false};
  FakeClusterDecoder decoderB{o2::detectors::DetID::ITS, {0}, false};

  std::array<test::TestClusterSourceInput, 2> sources{};
  sources[0].id = ClusterSourceId{0};
  sources[0].detector = o2::detectors::DetID::ITS;
  sources[0].clusters = clustersA;
  sources[0].patterns = patternsA;
  sources[0].rofs = rofsA;
  sources[0].dictionary = &dict();
  sources[0].layerToSurface = firstITSSurface;
  sources[0].timing = ROFTimingConfig{40, 0, 0, 0};
  sources[0].setDecoder(decoderA);

  sources[1].id = ClusterSourceId{1};
  sources[1].detector = o2::detectors::DetID::ITS;
  sources[1].clusters = clustersB;
  sources[1].patterns = patternsB;
  sources[1].rofs = rofsB;
  sources[1].dictionary = &dict();
  sources[1].layerToSurface = secondITSSurface;
  sources[1].timing = ROFTimingConfig{100, 0, 0, 0};
  sources[1].setDecoder(decoderB);

  TimeFrame frame;
  configureFrame(frame, layout);
  BOOST_REQUIRE_NO_THROW(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(sources), {0, 0}));

  BOOST_CHECK_EQUAL(frame.getTotalMeasurements(), 4u);
  BOOST_CHECK_EQUAL(frame.getNrof(0), 3);
  BOOST_CHECK_EQUAL(frame.getNrof(1), 1);
}

BOOST_AUTO_TEST_CASE(OverlappingAndNonOverlappingSourceTimingIntervals)
{
  const auto layout = makeCombinedLayout();
  BOOST_REQUIRE(layout.valid());

  const std::vector<CompClusterExt> clustersA{{1, 1, CompCluster::InvalidPatternID, 0}};
  const std::vector<CompClusterExt> clustersB{{2, 2, CompCluster::InvalidPatternID, 0}};
  const auto patternsA = makePatternBytes(clustersA.size());
  const auto patternsB = makePatternBytes(clustersB.size());
  // Source A ROF at BC 0..40 (TF-relative); source B ROF at real BC 30 -> its
  // own interval overlaps A's despite a different, unrelated ROF ordinal.
  const std::vector<ROFRecord> rofsA{ROFRecord{{0, 0}, 0, 0, 1}};
  const std::vector<ROFRecord> rofsB{ROFRecord{{30, 0}, 0, 0, 1}};
  // Source C ROF at real BC 1000: far away, must not overlap A.
  const std::vector<CompClusterExt> clustersC{{3, 3, CompCluster::InvalidPatternID, 0}};
  const auto patternsC = makePatternBytes(clustersC.size());
  const std::vector<ROFRecord> rofsC{ROFRecord{{1000, 0}, 0, 0, 1}};

  FakeClusterDecoder decoderA{o2::detectors::DetID::ITS, {0}, false};
  FakeClusterDecoder decoderB{o2::detectors::DetID::ITS, {0}, false};
  FakeClusterDecoder decoderC{o2::detectors::DetID::MFT, {0}, true};

  std::array<test::TestClusterSourceInput, 3> sources{};
  sources[0].id = ClusterSourceId{0};
  sources[0].detector = o2::detectors::DetID::ITS;
  sources[0].clusters = clustersA;
  sources[0].patterns = patternsA;
  sources[0].rofs = rofsA;
  sources[0].dictionary = &dict();
  sources[0].layerToSurface = firstITSSurface;
  sources[0].timing = ROFTimingConfig{40, 0, 0, 0};
  sources[0].setDecoder(decoderA);

  sources[1].id = ClusterSourceId{1};
  sources[1].detector = o2::detectors::DetID::ITS;
  sources[1].clusters = clustersB;
  sources[1].patterns = patternsB;
  sources[1].rofs = rofsB;
  sources[1].dictionary = &dict();
  sources[1].layerToSurface = secondITSSurface;
  sources[1].timing = ROFTimingConfig{40, 0, 0, 0};
  sources[1].setDecoder(decoderB);

  sources[2].id = ClusterSourceId{2};
  sources[2].detector = o2::detectors::DetID::MFT;
  sources[2].clusters = clustersC;
  sources[2].patterns = patternsC;
  sources[2].rofs = rofsC;
  sources[2].dictionary = &dict();
  sources[2].layerToSurface = firstMFTSurface;
  sources[2].timing = ROFTimingConfig{40, 0, 0, 0};
  sources[2].setDecoder(decoderC);

  TimeFrame frame;
  configureFrame(frame, layout);
  BOOST_REQUIRE_NO_THROW(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(sources), {0, 0}));

  BOOST_CHECK_EQUAL(frame.getTotalMeasurements(), 3u);
}

BOOST_AUTO_TEST_CASE(TriggeredAndContinuousReadoutAreBothSupportedTogether)
{
  // Continuous source: ROFs sit at a fixed cadence equal to the readout
  // length, so consecutive interval begins are regularly spaced by
  // rofLength (mirrors a periodic strobe). Triggered source: ROFs sit at
  // sparse, irregular real interaction records (individual triggers) with a
  // short trigger-specific window, so consecutive interval begins follow the
  // trigger BCs exactly rather than any ordinal*rofLength formula. Both must
  // load into the same frame and their intervals must remain independently
  // and correctly comparable via intersection.
  const auto layout = makeCombinedLayout();
  BOOST_REQUIRE(layout.valid());

  const std::vector<CompClusterExt> continuousClusters{
    {1, 1, CompCluster::InvalidPatternID, 0},
    {2, 2, CompCluster::InvalidPatternID, 0},
    {3, 3, CompCluster::InvalidPatternID, 0}};
  const auto continuousPatterns = makePatternBytes(continuousClusters.size());
  const std::vector<ROFRecord> continuousRofs{
    ROFRecord{{0, 0}, 0, 0, 1},
    ROFRecord{{40, 0}, 1, 1, 1},
    ROFRecord{{80, 0}, 2, 2, 1}};
  constexpr TFBC continuousRofLength = 40;

  const std::vector<CompClusterExt> triggeredClusters{
    {4, 4, CompCluster::InvalidPatternID, 0},
    {5, 5, CompCluster::InvalidPatternID, 0},
    {6, 6, CompCluster::InvalidPatternID, 0}};
  const auto triggeredPatterns = makePatternBytes(triggeredClusters.size());
  // Sparse, irregular trigger BCs; a short single-BC-scale trigger window.
  const std::vector<ROFRecord> triggeredRofs{
    ROFRecord{{5, 0}, 0, 0, 1},
    ROFRecord{{137, 0}, 1, 1, 1},
    ROFRecord{{812, 0}, 2, 2, 1}};
  constexpr TFBC triggeredRofLength = 4;

  FakeClusterDecoder continuousDecoder{o2::detectors::DetID::ITS, {0}, false};
  FakeClusterDecoder triggeredDecoder{o2::detectors::DetID::ITS, {0}, false};

  std::array<test::TestClusterSourceInput, 2> sources{};
  sources[0].id = ClusterSourceId{0};
  sources[0].detector = o2::detectors::DetID::ITS;
  sources[0].clusters = continuousClusters;
  sources[0].patterns = continuousPatterns;
  sources[0].rofs = continuousRofs;
  sources[0].dictionary = &dict();
  sources[0].layerToSurface = firstITSSurface;
  sources[0].timing = ROFTimingConfig{continuousRofLength, 0, 0, 0};
  sources[0].setDecoder(continuousDecoder);

  sources[1].id = ClusterSourceId{1};
  sources[1].detector = o2::detectors::DetID::ITS;
  sources[1].clusters = triggeredClusters;
  sources[1].patterns = triggeredPatterns;
  sources[1].rofs = triggeredRofs;
  sources[1].dictionary = &dict();
  sources[1].layerToSurface = secondITSSurface;
  sources[1].timing = ROFTimingConfig{triggeredRofLength, 0, 0, 0};
  sources[1].setDecoder(triggeredDecoder);

  TimeFrame frame;
  configureFrame(frame, layout);
  BOOST_REQUIRE_NO_THROW(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(sources), {0, 0}));

  BOOST_CHECK_EQUAL(frame.getTotalMeasurements(), 6u);
}

BOOST_AUTO_TEST_CASE(SourceSpecificPatternCursorsAreIndependent)
{
  const auto layout = makeCombinedLayout();
  BOOST_REQUIRE(layout.valid());

  const std::vector<CompClusterExt> clustersA{
    {1, 1, CompCluster::InvalidPatternID, 0},
    {2, 2, CompCluster::InvalidPatternID, 0}};
  const std::vector<CompClusterExt> clustersB{
    {3, 3, CompCluster::InvalidPatternID, 0},
    {4, 4, CompCluster::InvalidPatternID, 0}};
  const auto patternsA = makePatternBytes(clustersA.size());
  const auto patternsB = makePatternBytes(clustersB.size());
  const std::vector<ROFRecord> rofsA{ROFRecord{{0, 0}, 0, 0, 2}};
  const std::vector<ROFRecord> rofsB{ROFRecord{{0, 0}, 0, 0, 2}};

  FakeClusterDecoder decoderA{o2::detectors::DetID::ITS, {0}, false};
  FakeClusterDecoder decoderB{o2::detectors::DetID::ITS, {0}, false};

  std::array<test::TestClusterSourceInput, 2> sources{};
  sources[0].id = ClusterSourceId{0};
  sources[0].detector = o2::detectors::DetID::ITS;
  sources[0].clusters = clustersA;
  sources[0].patterns = patternsA;
  sources[0].rofs = rofsA;
  sources[0].dictionary = &dict();
  sources[0].layerToSurface = firstITSSurface;
  sources[0].timing = ROFTimingConfig{40, 0, 0, 0};
  sources[0].setDecoder(decoderA);

  sources[1].id = ClusterSourceId{1};
  sources[1].detector = o2::detectors::DetID::ITS;
  sources[1].clusters = clustersB;
  sources[1].patterns = patternsB;
  sources[1].rofs = rofsB;
  sources[1].dictionary = &dict();
  sources[1].layerToSurface = secondITSSurface;
  sources[1].timing = ROFTimingConfig{40, 0, 0, 0};
  sources[1].setDecoder(decoderB);

  TimeFrame frame;
  configureFrame(frame, layout);
  std::vector<std::vector<uint32_t>> clusterSizesBySurface;
  BOOST_REQUIRE_NO_THROW(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(sources), {0, 0},
                                           nullptr, &clusterSizesBySurface));

  // Every cluster consumed exactly one 1-pixel pattern regardless of source.
  for (const auto layer : {LayerId{0}, LayerId{1}}) {
    for (const auto& m : frame.getGlobalMeasurements(layer)) {
      BOOST_CHECK_EQUAL(clusterSizesBySurface[layer.value()][m.clusterId], 1u);
    }
  }
}

BOOST_AUTO_TEST_CASE(CommonDictionaryPatternDoesNotConsumeExplicitBytes)
{
  const auto layout = makeCombinedLayout();
  const std::vector<CompClusterExt> clusters{
    {1, 1, 0, 0},                              // common dictionary pattern
    {2, 2, CompCluster::InvalidPatternID, 0}}; // explicit pattern
  const std::vector<unsigned char> patterns{onePixelPattern.begin(), onePixelPattern.end()};
  const std::vector<ROFRecord> rofs{ROFRecord{{0, 0}, 0, 0, 2}};
  PatternContractDecoder decoder;

  test::TestClusterSourceInput src;
  src.id = ClusterSourceId{0};
  src.detector = o2::detectors::DetID::ITS;
  src.clusters = clusters;
  src.patterns = patterns;
  src.rofs = rofs;
  src.dictionary = &dict();
  src.layerToSurface = itsLayerToSurface;
  src.timing = ROFTimingConfig{40, 0, 0, 0};
  src.setDecoder(decoder);

  TimeFrame frame;
  configureFrame(frame, layout);
  std::vector<std::vector<uint32_t>> clusterSizesBySurface;
  BOOST_REQUIRE_NO_THROW(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(&src, 1), {0, 0},
                                           nullptr, &clusterSizesBySurface));
  BOOST_REQUIRE_EQUAL(frame.getGlobalMeasurements(LayerId{0}).size(), 2u);
  BOOST_CHECK_EQUAL(clusterSizesBySurface[0][frame.getGlobalMeasurements(LayerId{0})[0].clusterId], 1u);
  BOOST_CHECK_EQUAL(clusterSizesBySurface[0][frame.getGlobalMeasurements(LayerId{0})[1].clusterId], 1u);
}

BOOST_AUTO_TEST_CASE(ExactPatternConsumptionSucceedsAndTrailingBytesAreRejected)
{
  const auto layout = makeCombinedLayout();
  PatternContractDecoder decoder;
  const std::vector<CompClusterExt> clusters{{1, 1, CompCluster::InvalidPatternID, 0}};
  const std::vector<ROFRecord> rofs{ROFRecord{{0, 0}, 0, 0, 1}};

  auto makeSource = [&](gsl::span<const unsigned char> patterns) {
    test::TestClusterSourceInput src;
    src.id = ClusterSourceId{0};
    src.detector = o2::detectors::DetID::ITS;
    src.clusters = clusters;
    src.patterns = patterns;
    src.rofs = rofs;
    src.dictionary = &dict();
    src.layerToSurface = itsLayerToSurface;
    src.timing = ROFTimingConfig{40, 0, 0, 0};
    src.setDecoder(decoder);
    return src;
  };

  const std::vector<unsigned char> exact{onePixelPattern.begin(), onePixelPattern.end()};
  auto exactSource = makeSource(exact);
  TimeFrame frame;
  configureFrame(frame, layout);
  BOOST_REQUIRE_NO_THROW(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(&exactSource, 1), {0, 0}));

  const std::vector<unsigned char> trailing{1, 1, 0x80, 0xff};
  auto trailingSource = makeSource(trailing);
  BOOST_CHECK_EXCEPTION(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(&trailingSource, 1), {0, 0}), std::runtime_error, [](const std::runtime_error& error) { return std::string(error.what()).find("Trailing cluster pattern data") != std::string::npos; });
  frame.resetTimeFrame();
  BOOST_CHECK_EQUAL(frame.getTotalMeasurements(), 0u);

  auto missingDictionarySource = makeSource(exact);
  missingDictionarySource.dictionary = nullptr;
  BOOST_CHECK_EXCEPTION(test::loadSources(
                          frame, layout.getCatalog(),
                          gsl::span<const test::TestClusterSourceInput>(&missingDictionarySource, 1), {0, 0}),
                        std::runtime_error, [](const std::runtime_error& error) { return std::string(error.what()).find("Cluster dictionary is not available") != std::string::npos; });
  BOOST_CHECK_EQUAL(frame.getTotalMeasurements(), 0u);
}

BOOST_AUTO_TEST_CASE(MissingDictionaryThrowsBeforeProductionGeometryDecode)
{
  const auto layout = makeCombinedLayout();
  TimeFrame frame;
  configureFrame(frame, layout);
  const std::array<CompClusterExt, 1> clusters{{{1, 1, CompCluster::InvalidPatternID, 0}}};
  ClusterSourceInput source;
  source.id = ClusterSourceId{0};
  source.clusters = clusters;
  source.patterns = onePixelPattern;
  source.layerToSurface = itsLayerToSurface;
  BOOST_CHECK_EXCEPTION(loadTimeFrameSources(frame, gsl::span<const ClusterSourceInput>{&source, 1}, layout.getCatalog(), {0, 0}),
                        std::runtime_error, [](const std::runtime_error& error) {
                          return std::string(error.what()).find("Cluster dictionary is not available source=0") != std::string::npos;
                        });
}

BOOST_AUTO_TEST_CASE(AbsentLabelsAreLegal)
{
  const auto layout = makeCombinedLayout();
  BOOST_REQUIRE(layout.valid());

  const std::vector<CompClusterExt> clusters{{1, 1, CompCluster::InvalidPatternID, 0}};
  const auto patterns = makePatternBytes(clusters.size());
  const std::vector<ROFRecord> rofs{ROFRecord{{0, 0}, 0, 0, 1}};

  FakeClusterDecoder decoder{o2::detectors::DetID::ITS, {0}, false};
  test::TestClusterSourceInput src;
  src.id = ClusterSourceId{0};
  src.detector = o2::detectors::DetID::ITS;
  src.clusters = clusters;
  src.patterns = patterns;
  src.rofs = rofs;
  src.dictionary = &dict();
  src.labels = nullptr; // no MC labels for this source
  src.layerToSurface = itsLayerToSurface;
  src.timing = ROFTimingConfig{40, 0, 0, 0};
  src.setDecoder(decoder);

  TimeFrame frame;
  configureFrame(frame, layout);
  BOOST_REQUIRE_NO_THROW(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(&src, 1), {0, 0}));

  BOOST_CHECK(frame.getLabels(LayerId{0}, 0).empty());
  BOOST_CHECK(frame.getLabels(LayerId{}, 0).empty());
}

BOOST_AUTO_TEST_CASE(NonDenseAndDuplicateAndInvalidSourceIdsAreRejected)
{
  const auto layout = makeCombinedLayout();
  BOOST_REQUIRE(layout.valid());
  const std::vector<CompClusterExt> clusters{{1, 1, CompCluster::InvalidPatternID, 0}};
  const auto patterns = makePatternBytes(clusters.size());
  const std::vector<ROFRecord> rofs{ROFRecord{{0, 0}, 0, 0, 1}};
  FakeClusterDecoder decoderA{o2::detectors::DetID::ITS, {0}, false};
  FakeClusterDecoder decoderB{o2::detectors::DetID::ITS, {0}, false};

  auto makeSource = [&](ClusterSourceId id, FakeClusterDecoder& decoder) {
    test::TestClusterSourceInput src;
    src.id = id;
    src.detector = o2::detectors::DetID::ITS;
    src.clusters = clusters;
    src.patterns = patterns;
    src.rofs = rofs;
    src.dictionary = &dict();
    src.layerToSurface = itsLayerToSurface;
    src.timing = ROFTimingConfig{40, 0, 0, 0};
    src.setDecoder(decoder);
    return src;
  };

  {
    // Non-dense: ids {0, 2} for two sources.
    std::array<test::TestClusterSourceInput, 2> sources{makeSource(ClusterSourceId{0}, decoderA), makeSource(ClusterSourceId{2}, decoderB)};
    TimeFrame frame;
    configureFrame(frame, layout);
    BOOST_CHECK_EXCEPTION(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(sources), {0, 0}), std::runtime_error, [](const std::runtime_error& error) { return std::string(error.what()).find("Source IDs must be dense") != std::string::npos; });
  }
  {
    // Duplicate ids {0, 0}.
    std::array<test::TestClusterSourceInput, 2> sources{makeSource(ClusterSourceId{0}, decoderA), makeSource(ClusterSourceId{0}, decoderB)};
    TimeFrame frame;
    configureFrame(frame, layout);
    BOOST_CHECK_EXCEPTION(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(sources), {0, 0}), std::runtime_error, [](const std::runtime_error& error) { return std::string(error.what()).find("Duplicate source ID") != std::string::npos; });
  }
  {
    // Explicitly invalid id.
    std::array<test::TestClusterSourceInput, 1> sources{makeSource(ClusterSourceId::invalid(), decoderA)};
    TimeFrame frame;
    configureFrame(frame, layout);
    BOOST_CHECK_EXCEPTION(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(sources), {0, 0}), std::runtime_error, [](const std::runtime_error& error) { return std::string(error.what()).find("Source IDs must be dense") != std::string::npos; });
  }
}

BOOST_AUTO_TEST_CASE(InvalidROFClusterRangesAreRejected)
{
  const auto layout = makeCombinedLayout();
  BOOST_REQUIRE(layout.valid());
  const std::vector<CompClusterExt> clusters{
    {1, 1, CompCluster::InvalidPatternID, 0},
    {2, 2, CompCluster::InvalidPatternID, 0}};
  const auto patterns = makePatternBytes(clusters.size());
  FakeClusterDecoder decoder{o2::detectors::DetID::ITS, {0}, false};

  auto makeSrc = [&](const std::vector<ROFRecord>& rofs) {
    test::TestClusterSourceInput src;
    src.id = ClusterSourceId{0};
    src.detector = o2::detectors::DetID::ITS;
    src.clusters = clusters;
    src.patterns = patterns;
    src.rofs = rofs;
    src.dictionary = &dict();
    src.layerToSurface = itsLayerToSurface;
    src.timing = ROFTimingConfig{40, 0, 0, 0};
    src.setDecoder(decoder);
    return src;
  };

  {
    // Out of bounds: firstEntry+nEntries exceeds the cluster span.
    const std::vector<ROFRecord> rofs{ROFRecord{{0, 0}, 0, 0, 5}};
    auto src = makeSrc(rofs);
    TimeFrame frame;
    configureFrame(frame, layout);
    BOOST_CHECK_EXCEPTION(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(&src, 1), {0, 0}), std::runtime_error, [](const std::runtime_error& error) { return std::string(error.what()).find("Invalid ROF cluster range") != std::string::npos; });
  }
  {
    // Overlapping ranges.
    const std::vector<ROFRecord> rofs{ROFRecord{{0, 0}, 0, 0, 2}, ROFRecord{{40, 0}, 1, 1, 1}};
    auto src = makeSrc(rofs);
    TimeFrame frame;
    configureFrame(frame, layout);
    BOOST_CHECK_EXCEPTION(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(&src, 1), {0, 0}), std::runtime_error, [](const std::runtime_error& error) { return std::string(error.what()).find("Invalid ROF cluster range") != std::string::npos; });
  }
  {
    // Leading gap: first ROF does not begin at cluster index 0.
    const std::vector<ROFRecord> rofs{ROFRecord{{0, 0}, 0, 1, 1}};
    auto src = makeSrc(rofs);
    TimeFrame frame;
    configureFrame(frame, layout);
    BOOST_CHECK_EXCEPTION(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(&src, 1), {0, 0}), std::runtime_error, [](const std::runtime_error& error) { return std::string(error.what()).find("Invalid ROF cluster range") != std::string::npos; });
  }
  {
    // Internal gap: rof0 covers [0,1), rof1 covers [2,2) i.e. starts at 2
    // while only cluster index 1 is unreferenced in between (2 clusters
    // total, so this leaves cluster 1 outside any ROF).
    const std::vector<ROFRecord> rofs{ROFRecord{{0, 0}, 0, 0, 1}, ROFRecord{{40, 0}, 1, 2, 0}};
    auto src = makeSrc(rofs);
    TimeFrame frame;
    configureFrame(frame, layout);
    BOOST_CHECK_EXCEPTION(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(&src, 1), {0, 0}), std::runtime_error, [](const std::runtime_error& error) { return std::string(error.what()).find("Invalid ROF cluster range") != std::string::npos; });
  }
  {
    // Trailing cluster: the ROFs cover only the first cluster, leaving the
    // second cluster unreferenced by any ROF.
    const std::vector<ROFRecord> rofs{ROFRecord{{0, 0}, 0, 0, 1}};
    auto src = makeSrc(rofs);
    TimeFrame frame;
    configureFrame(frame, layout);
    BOOST_CHECK_EXCEPTION(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(&src, 1), {0, 0}), std::runtime_error, [](const std::runtime_error& error) { return std::string(error.what()).find("Invalid ROF cluster range") != std::string::npos; });
  }
  {
    // Clusters without ROFs: zero ROFs is only valid when clusters is also
    // empty, but this source has two clusters.
    const std::vector<ROFRecord> rofs{};
    auto src = makeSrc(rofs);
    TimeFrame frame;
    configureFrame(frame, layout);
    BOOST_CHECK_EXCEPTION(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(&src, 1), {0, 0}), std::runtime_error, [](const std::runtime_error& error) { return std::string(error.what()).find("Invalid ROF cluster range") != std::string::npos; });
  }
}

BOOST_AUTO_TEST_CASE(ZeroROFsIsValidWithZeroClusters)
{
  const auto layout = makeCombinedLayout();
  BOOST_REQUIRE(layout.valid());
  const std::vector<CompClusterExt> clusters{};
  const std::vector<unsigned char> patterns{};
  const std::vector<ROFRecord> rofs{};
  FakeClusterDecoder decoder{o2::detectors::DetID::ITS, {0}, false};

  test::TestClusterSourceInput src;
  src.id = ClusterSourceId{0};
  src.detector = o2::detectors::DetID::ITS;
  src.clusters = clusters;
  src.patterns = patterns;
  src.rofs = rofs;
  src.dictionary = &dict();
  src.layerToSurface = itsLayerToSurface;
  src.timing = ROFTimingConfig{40, 0, 0, 0};
  src.setDecoder(decoder);

  TimeFrame frame;
  configureFrame(frame, layout);
  BOOST_REQUIRE_NO_THROW(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(&src, 1), {0, 0}));
  BOOST_CHECK_EQUAL(frame.getTotalMeasurements(), 0u);
}

BOOST_AUTO_TEST_CASE(InvalidLayerToSurfaceMappingIsRejected)
{
  const auto layout = makeCombinedLayout();
  BOOST_REQUIRE(layout.valid());
  const std::vector<CompClusterExt> clusters{{1, 1, CompCluster::InvalidPatternID, 1}}; // sensor 1 -> layer 1
  const auto patterns = makePatternBytes(clusters.size());
  const std::vector<ROFRecord> rofs{ROFRecord{{0, 0}, 0, 0, 1}};
  FakeClusterDecoder decoder{o2::detectors::DetID::ITS, {-1, 1}, false};

  test::TestClusterSourceInput src;
  src.id = ClusterSourceId{0};
  src.detector = o2::detectors::DetID::ITS;
  src.clusters = clusters;
  src.patterns = patterns;
  src.rofs = rofs;
  src.dictionary = &dict();
  src.layerToSurface = gsl::span<const LayerId>(itsLayerToSurface.data(), 1); // too short: only covers layer 0
  src.timing = ROFTimingConfig{40, 0, 0, 0};
  src.setDecoder(decoder);

  TimeFrame frame;
  configureFrame(frame, layout);
  BOOST_CHECK_EXCEPTION(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(&src, 1), {0, 0}), std::runtime_error, [](const std::runtime_error& error) { return std::string(error.what()).find("Invalid source-to-surface layer mapping") != std::string::npos; });
}

BOOST_AUTO_TEST_CASE(DetectorSurfaceMismatchIsRejected)
{
  const auto layout = makeCombinedLayout();
  BOOST_REQUIRE(layout.valid());
  const std::vector<CompClusterExt> clusters{{1, 1, CompCluster::InvalidPatternID, 0}};
  const auto patterns = makePatternBytes(clusters.size());
  const std::vector<ROFRecord> rofs{ROFRecord{{0, 0}, 0, 0, 1}};
  FakeClusterDecoder decoder{o2::detectors::DetID::ITS, {0}, false};

  test::TestClusterSourceInput src;
  src.id = ClusterSourceId{0};
  src.detector = o2::detectors::DetID::ITS;
  src.clusters = clusters;
  src.patterns = patterns;
  src.rofs = rofs;
  src.dictionary = &dict();
  // Deliberately mapped to an MFT surface: ITS source, MFT surface.
  const std::array<LayerId, 1> wrongMapping{LayerId{2}};
  src.layerToSurface = wrongMapping;
  src.timing = ROFTimingConfig{40, 0, 0, 0};
  src.setDecoder(decoder);

  TimeFrame frame;
  configureFrame(frame, layout);
  BOOST_CHECK_EXCEPTION(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(&src, 1), {0, 0}), std::runtime_error, [](const std::runtime_error& error) { return std::string(error.what()).find("Source detector does not match its surface") != std::string::npos; });
}

BOOST_AUTO_TEST_CASE(UnsafeDecodedLayerIsRejected)
{
  // The loader validates the decoded detector-local layer before using it to
  // index the authoritative layer-to-surface mapping.
  const auto layout = makeCombinedLayout();
  BOOST_REQUIRE(layout.valid());
  const std::vector<CompClusterExt> clusters{{1, 1, CompCluster::InvalidPatternID, 0}};
  const auto patterns = makePatternBytes(clusters.size());
  const std::vector<ROFRecord> rofs{ROFRecord{{0, 0}, 0, 0, 1}};

  const std::array<Corruption, 2> corruptions{
    Corruption::NegativeLayer, Corruption::LayerOutOfRange};
  for (const auto corruption : corruptions) {
    FakeClusterDecoder decoder{o2::detectors::DetID::ITS, {0}, false, corruption};
    test::TestClusterSourceInput src;
    src.id = ClusterSourceId{0};
    src.detector = o2::detectors::DetID::ITS;
    src.clusters = clusters;
    src.patterns = patterns;
    src.rofs = rofs;
    src.dictionary = &dict();
    src.layerToSurface = itsLayerToSurface;
    src.timing = ROFTimingConfig{40, 0, 0, 0};
    src.setDecoder(decoder);

    TimeFrame frame;
    configureFrame(frame, layout);
    BOOST_CHECK_EXCEPTION(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(&src, 1), {0, 0}), std::runtime_error, [](const std::runtime_error& error) { return std::string(error.what()).find("Invalid source-to-surface layer mapping") != std::string::npos; });
    BOOST_CHECK_EQUAL(frame.getTotalMeasurements(), 0u);
  }
}

BOOST_AUTO_TEST_CASE(LoadingClearsPreviousDataBeforeValidation)
{
  const auto layout = makeCombinedLayout();
  BOOST_REQUIRE(layout.valid());

  const std::vector<CompClusterExt> clusters{{1, 1, CompCluster::InvalidPatternID, 0}};
  const auto patterns = makePatternBytes(clusters.size());
  const std::vector<ROFRecord> rofs{ROFRecord{{0, 0}, 0, 0, 1}};
  FakeClusterDecoder decoder{o2::detectors::DetID::ITS, {0}, false};

  test::TestClusterSourceInput goodSrc;
  goodSrc.id = ClusterSourceId{0};
  goodSrc.detector = o2::detectors::DetID::ITS;
  goodSrc.clusters = clusters;
  goodSrc.patterns = patterns;
  goodSrc.rofs = rofs;
  goodSrc.dictionary = &dict();
  goodSrc.layerToSurface = itsLayerToSurface;
  goodSrc.timing = ROFTimingConfig{40, 0, 0, 0};
  goodSrc.setDecoder(decoder);

  TimeFrame frame;
  configureFrame(frame, layout);
  BOOST_REQUIRE_NO_THROW(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(&goodSrc, 1), {0, 0}));
  BOOST_REQUIRE_EQUAL(frame.getTotalMeasurements(), 1u);

  // Now attempt an invalid load (duplicate ids) on the SAME frame.
  std::array<test::TestClusterSourceInput, 2> badSources{goodSrc, goodSrc}; // both id==0
  BOOST_CHECK_THROW(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(badSources), {0, 0}), std::runtime_error);

  BOOST_CHECK_EQUAL(frame.getTotalMeasurements(), 0u);
}

BOOST_AUTO_TEST_CASE(CallerResetsAfterPartialLoadFailure)
{
  // The second source returns an invalid layer after the first source has
  // already populated the frame. A direct caller owns cleanup after the throw.
  const auto layout = makeCombinedLayout();
  BOOST_REQUIRE(layout.valid());

  const std::vector<CompClusterExt> clusters{{1, 1, CompCluster::InvalidPatternID, 0}};
  const auto patterns = makePatternBytes(clusters.size());
  const std::vector<ROFRecord> rofs{ROFRecord{{0, 0}, 0, 0, 1}};
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> labels;
  labels.addElement(0, o2::MCCompLabel{1, 0, 0});
  FakeClusterDecoder decoder{o2::detectors::DetID::ITS, {0}, false};

  test::TestClusterSourceInput goodSrc;
  goodSrc.id = ClusterSourceId{0};
  goodSrc.detector = o2::detectors::DetID::ITS;
  goodSrc.clusters = clusters;
  goodSrc.patterns = patterns;
  goodSrc.rofs = rofs;
  goodSrc.dictionary = &dict();
  goodSrc.labels = &labels;
  goodSrc.layerToSurface = itsLayerToSurface;
  goodSrc.timing = ROFTimingConfig{40, 0, 0, 0};
  goodSrc.setDecoder(decoder);

  TimeFrame frame;
  configureFrame(frame, layout);
  BOOST_REQUIRE_NO_THROW(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(&goodSrc, 1), {0, 0}));

  BOOST_REQUIRE_EQUAL(frame.getGlobalMeasurements(LayerId{0}).size(), 1u);
  BOOST_REQUIRE_EQUAL(frame.getLabels(LayerId{0}, 0).size(), 1u);

  FakeClusterDecoder decoderA{o2::detectors::DetID::ITS, {0}, false};
  FakeClusterDecoder decoderB{o2::detectors::DetID::MFT, {0}, true, Corruption::NegativeLayer};

  test::TestClusterSourceInput srcA = goodSrc;
  srcA.setDecoder(decoderA);

  test::TestClusterSourceInput srcB;
  srcB.id = ClusterSourceId{1};
  srcB.detector = o2::detectors::DetID::MFT;
  srcB.clusters = clusters;
  srcB.patterns = patterns;
  srcB.rofs = rofs;
  srcB.dictionary = &dict();
  const std::array<LayerId, 1> mapping{LayerId{2}};
  srcB.layerToSurface = mapping;
  srcB.timing = ROFTimingConfig{40, 0, 0, 0};
  srcB.setDecoder(decoderB);

  std::array<test::TestClusterSourceInput, 2> sources{srcA, srcB};
  BOOST_CHECK_EXCEPTION(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>(sources), {0, 0}), std::runtime_error, [](const std::runtime_error& error) { return std::string(error.what()).find("Invalid source-to-surface layer mapping") != std::string::npos; });

  frame.resetTimeFrame();
  BOOST_CHECK(frame.getGlobalMeasurements(LayerId{0}).empty());
  BOOST_CHECK(frame.getLabels(LayerId{0}, 0).empty());
  BOOST_CHECK(frame.getLabels(LayerId{2}, 0).empty());
}

BOOST_AUTO_TEST_CASE(EmptyFrameAccessorsAvoidNullPointerArithmetic)
{
  TimeFrame frame;

  BOOST_CHECK(frame.getSurfaceMeasurement(LayerId{0}, 0) == nullptr);
  BOOST_CHECK(frame.getLabels(LayerId{0}, 0).empty());

  // Loading zero sources into a layout with surfaces is legal and must
  // leave every per-surface bucket empty.
  const auto layout = makeCombinedLayout();
  BOOST_REQUIRE(layout.valid());
  configureFrame(frame, layout);
  BOOST_REQUIRE_NO_THROW(test::loadSources(frame, layout.getCatalog(), gsl::span<const test::TestClusterSourceInput>{}, {0, 0}));
  BOOST_CHECK_EQUAL(frame.getTotalMeasurements(), 0u);

  BOOST_CHECK(frame.getSurfaceMeasurement(LayerId{0}, 0) == nullptr);
  BOOST_CHECK(frame.getGlobalMeasurements(LayerId{0}).empty());
}

BOOST_AUTO_TEST_CASE(UnconfiguredFrameRejectsEvenAnEmptyLoad)
{
  // A layout with no surfaces at all, combined with zero sources, is the
  // most degenerate legal input: nothing to validate, nothing to decode,
  // nothing to commit.
  const SurfaceCatalogView emptyCatalog{};

  TimeFrame frame;
  BOOST_CHECK_EXCEPTION(test::loadSources(frame, emptyCatalog, gsl::span<const test::TestClusterSourceInput>{}, {0, 0}), std::runtime_error, [](const std::runtime_error& error) { return std::string(error.what()).find("TimeFrame is not configured") != std::string::npos; });
  BOOST_CHECK_EQUAL(frame.getTotalMeasurements(), 0u);
  BOOST_CHECK_EQUAL(frame.getNMeasurementSurfaces(), 0u);
}

BOOST_AUTO_TEST_CASE(MaxVerticesIncludesEachDistinctSourceLookup)
{
  TimeFrame frame;
  BOOST_CHECK_EQUAL(frame.getMaxVerticesPerROF(), 0);
  configureFrame(frame, makeCombinedLayout());
  BOOST_CHECK_EQUAL(frame.getMaxVerticesPerROF(), 0);

  // Two sources share storage but have different row indices. Each lookup
  // is also shared by both surfaces belonging to that source.
  const std::array<RuntimeROFTableEntry, 4> entries{{{0, 2}, {0, 3}, {0, 5}, {0, 7}}};
  const std::array<RuntimeROFTableEntry, 2> itsIndices{{{0, 1}, {1, 1}}};
  const std::array<RuntimeROFTableEntry, 2> mftIndices{{{2, 1}, {3, 1}}};
  RuntimeROFViews itsViews, mftViews;
  itsViews.vertexLookup = {entries.data(), itsIndices.data(), nullptr, 2};
  mftViews.vertexLookup = {entries.data(), mftIndices.data(), nullptr, 2};
  frame.setROFViews(itsViews);
  BOOST_CHECK_EQUAL(frame.getMaxVerticesPerROF(), 3);
  const std::array<int, 2> boundaries{0, 0};
  frame.setROFNavigation(2, boundaries, mftViews, 0);
  frame.setROFNavigation(3, boundaries, mftViews, 1);
  BOOST_CHECK_EQUAL(frame.getMaxVerticesPerROF(), 7);

  // Lookup views can cover different numbers of rows in the same table.
  mftViews.vertexLookup.mLayerCount = 1;
  frame.setROFNavigation(2, boundaries, mftViews, 0);
  BOOST_CHECK_EQUAL(frame.getMaxVerticesPerROF(), 7);
}
