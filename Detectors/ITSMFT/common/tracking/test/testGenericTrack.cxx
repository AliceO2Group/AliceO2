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

// Gate 4 GenericTrack foundation. Covers:
//  - GenericTrack/TrackClusterReference/GenericTrackTimestamp layout and
//    device-compatibility traits;
//  - isValidTrackRange()'s exact validity condition (empty/default, single-,
//    multi- and hole-containing ranges, out-of-range and reversed ranges);
//  - sorted global storage and source-indexed fitting-measurement lookup;
//  - cross-surface and cross-source TrackClusterReference resolution;
//  - that a completed track's hitLayers is the union of the LayerId of
//    every measurement its range references, and that each resolved
//    measurement's own surface matches the reference it was resolved from;
//  - that TimeFrame loading clears GenericTrack/track-label/track-reference
//    storage on both success and failure;
//  - that TimeFrame::resetTimeFrame() invalidates those result sidecars
//    together;
//  - that GenericTrack itself has no detector/public-output dependency.
//
// This slice does not populate GenericTrack from CA seeds: every track/range
// below is constructed directly by the test.

#define BOOST_TEST_MODULE ITSMFT GenericTrack
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <gsl/gsl>

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "ITSMFTTracking/GenericTrack.h"
#include "ITSMFTTracking/DetectorLayout.h"
#include "ITSMFTTracking/IOUtils.h"
#include "ITSMFTTracking/ClusterDecoding.h"
#include "ITSMFTTracking/detail/TimeFrameScratch.h"
#include "ITSMFTTracking/detail/ITSSharedClusterCompatibility.h"
#include "ITSMFTTracking/GenericTrackOutputAdapter.h"
#include "ITSMFTTracking/TimeFrame.h"
#include "ITSMFTTracking/TrackingConfigParam.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::itsmft;
using namespace o2::itsmft::tracking;

// ---------------------------------------------------------------------
// GenericTrack has no detector/public-output dependency.
//
// This is a structural claim about ITSMFTTracking/GenericTrack.h itself, not
// something a runtime assertion can observe: GenericTrack.h's own include
// list (GPUCommonDef.h, and the ITSMFTTracking/Surface{Id,KinematicState,
// Mask,Timing}.h common primitives) contains no DetID.h,
// TrackITS.h/TrackITSExt.h, typed MFT output header, GeometryTGeo.h, or workflow
// header, and GenericTrack/TrackClusterReference declare no
// DetID/NLayers/publication-type field -- every field is either a plain
// scalar, or one of the shared LayerId/SurfaceTrackState/LayerMask/
// a dense source cluster ID, or a GenericTrackTimestamp device POD. This test case
// exercises GenericTrack using exactly that narrow surface, so that if a
// future edit to GenericTrack.h ever added such a dependency, the type
// itself (constructible, copyable, comparable-by-field here) would still
// need no wider include to keep working -- the absence is enforced by
// review of GenericTrack.h's own include list, restated here as the
// authoritative claim this test documents.
// ---------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GenericTrackHasNoDetectorOrPublicationOutputDependency)
{
  GenericTrack track{};
  track.innerState.kind = SurfaceKind::Cylinder;
  track.outerState.kind = SurfaceKind::Cylinder;
  track.chi2 = 1.5f;
  track.timestamp = GenericTrackTimestamp{100, 140};
  track.hitLayers.set(0);
  track.firstClusterRef = 0;
  track.clusterRefEnd = 1;
  BOOST_CHECK(track.hitLayers.has(0));
  BOOST_CHECK_EQUAL(trackClusterRefCount(track), 1u);

  const TrackClusterReference reference{LayerId{0}, 0, 17};
  BOOST_CHECK(reference.layer == LayerId{0});
  BOOST_CHECK_EQUAL(reference.clusterId, 17u);
}

BOOST_AUTO_TEST_CASE(GenericTrackLayoutAndDeviceCompatibilityTraits)
{
  static_assert(std::is_standard_layout_v<GenericTrack>);
  static_assert(std::is_trivially_copyable_v<GenericTrack>);
  static_assert(sizeof(GenericTrack) == 224);
  static_assert(alignof(GenericTrack) == alignof(GenericTrackTimestamp));
  static_assert(std::is_standard_layout_v<TrackClusterReference>);
  static_assert(std::is_trivially_copyable_v<TrackClusterReference>);
  static_assert(std::is_standard_layout_v<GenericTrackTimestamp>);
  static_assert(std::is_trivially_copyable_v<GenericTrackTimestamp>);

  static_assert(std::is_same_v<decltype(GenericTrack::hitLayers), LayerMask>);
  static_assert(std::is_same_v<decltype(GenericTrack::innerState), SurfaceTrackState>);
  static_assert(std::is_same_v<decltype(GenericTrack::outerState), SurfaceTrackState>);
  static_assert(std::is_same_v<decltype(GenericTrack::timestamp), GenericTrackTimestamp>);
  static_assert(std::is_same_v<decltype(GenericTrack::firstClusterRef), uint32_t>);
  static_assert(std::is_same_v<decltype(GenericTrack::clusterRefEnd), uint32_t>);
  static_assert(std::is_same_v<decltype(TrackClusterReference::layer), LayerId>);
  static_assert(std::is_same_v<decltype(TrackClusterReference::clusterId), uint32_t>);

  // Default-constructed: zeroed range, empty mask, no NLayers/detector
  // dependency of any kind. Not constructed as `constexpr` here:
  // o2::track::PID's constructor (SurfaceTrackState::pid's default
  // member initializer) is not itself constexpr, so a GenericTrack instance
  // cannot be a core-constant-expression -- a property of PID, unrelated to
  // GenericTrack's trivial-copyability asserted above.
  const GenericTrack defaultTrack{};
  BOOST_CHECK_EQUAL(defaultTrack.firstClusterRef, 0u);
  BOOST_CHECK_EQUAL(defaultTrack.clusterRefEnd, 0u);
  BOOST_CHECK(defaultTrack.hitLayers.empty());
  BOOST_CHECK_EQUAL(defaultTrack.chi2, 0.f);
  BOOST_CHECK(!defaultTrack.timestamp.isValid()); // default {0,0}: begin < end is false
}

// --- isValidTrackRange() -------------------------------------------------

BOOST_AUTO_TEST_CASE(EmptyDefaultRangeIsValidForAnyContainerSize)
{
  const GenericTrack track{};
  BOOST_CHECK(isValidTrackRange(track, 0));
  BOOST_CHECK(isValidTrackRange(track, 5));
  BOOST_CHECK_EQUAL(trackClusterRefCount(track), 0u);
}

BOOST_AUTO_TEST_CASE(ValidSingleMultiAndHoleContainingRanges)
{
  // Single-hit range: [0,1) into a 1-element array.
  GenericTrack single{};
  single.firstClusterRef = 0;
  single.clusterRefEnd = 1;
  BOOST_CHECK(isValidTrackRange(single, 1));
  BOOST_CHECK_EQUAL(trackClusterRefCount(single), 1u);

  // Multi-hit range: [1,4) into a 5-element array (some entries before/after
  // the range belong to other tracks sharing the same flat array).
  GenericTrack multi{};
  multi.firstClusterRef = 1;
  multi.clusterRefEnd = 4;
  BOOST_CHECK(isValidTrackRange(multi, 5));
  BOOST_CHECK_EQUAL(trackClusterRefCount(multi), 3u);

  // Hole-containing: the range itself is a dense [first,end) span of
  // *present* references (holes are never stored as sentinel entries); a
  // hole instead shows up as a gap in hitLayers' LayerId numbering. A
  // 2-hit track on surfaces {0,2} (skipping surface 1) is a valid,
  // completed, hole-containing track: its range is still contiguous and
  // valid, only its mask has a gap.
  GenericTrack withHole{};
  withHole.firstClusterRef = 0;
  withHole.clusterRefEnd = 2;
  withHole.hitLayers.set(0);
  withHole.hitLayers.set(2);
  BOOST_CHECK(isValidTrackRange(withHole, 2));
  BOOST_CHECK_EQUAL(withHole.hitLayers.count(), 2);
  BOOST_CHECK(!withHole.hitLayers.has(1)); // the hole
}

BOOST_AUTO_TEST_CASE(OutOfRangeAndReversedRangesAreRejected)
{
  GenericTrack pastEnd{};
  pastEnd.firstClusterRef = 0;
  pastEnd.clusterRefEnd = 6;
  BOOST_CHECK(!isValidTrackRange(pastEnd, 5)); // clusterRefEnd > size

  GenericTrack exactlyAtSize{};
  exactlyAtSize.firstClusterRef = 0;
  exactlyAtSize.clusterRefEnd = 5;
  BOOST_CHECK(isValidTrackRange(exactlyAtSize, 5)); // clusterRefEnd == size is valid (half-open)

  GenericTrack reversed{};
  reversed.firstClusterRef = 3;
  reversed.clusterRefEnd = 1;
  BOOST_CHECK(!isValidTrackRange(reversed, 5)); // firstClusterRef > clusterRefEnd
}

// --- Per-surface measurement storage / TrackClusterReference resolution --

namespace
{

// Minimal, geometry-free decoder (same construction as
// testMultiSourceLoading.cxx/testTimeFrameLifecycle.cxx): sensorID is used
// directly as the detector-local layer.
class FakeClusterDecoder final : public ClusterDecoder
{
 public:
  FakeClusterDecoder(o2::detectors::DetID::ID detector, bool disk) : mDetector(detector), mDisk(disk) {}

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
    decoded.global = {static_cast<float>(sensorID), static_cast<float>(cluster.getRow()), static_cast<float>(cluster.getCol())};
    decoded.cylinderFrame = {10.f + sensorID, 1.f, 2.f, 0.1f};
    decoded.rowColumnCovariance = {clusterData.sig2Row, 0.f, clusterData.sig2Col};
    decoded.shape = clusterData.shape;
    decoded.layer = sensorID;
    return result;
  }

 private:
  o2::detectors::DetID::ID mDetector;
  bool mDisk;
};

struct BuiltLayout {
  DetectorLayout layout;
  std::vector<SurfaceDescriptor> surfaces;

  SurfaceCatalogView getCatalog() const noexcept
  {
    return layout.getSurfaceCatalog();
  }
};

// 4-surface disconnected ITS(cylinder){0,1,2}+MFT(disk){3} layout, matching
// this file's fixtures below.
BuiltLayout makeCombinedLayout()
{
  std::vector<SurfaceDescriptor> surfaces;
  surfaces.push_back(SurfaceDescriptor{0, static_cast<uint8_t>(o2::detectors::DetID::ITS), SurfaceKind::Cylinder});
  surfaces.push_back(SurfaceDescriptor{1, static_cast<uint8_t>(o2::detectors::DetID::ITS), SurfaceKind::Cylinder});
  surfaces.push_back(SurfaceDescriptor{2, static_cast<uint8_t>(o2::detectors::DetID::ITS), SurfaceKind::Cylinder});
  surfaces.push_back(SurfaceDescriptor{0, static_cast<uint8_t>(o2::detectors::DetID::MFT), SurfaceKind::Disk});
  DetectorLayoutDefinition definition;
  definition.componentOffsets = {0, 3};
  return BuiltLayout{DetectorLayout{surfaces, std::move(definition)}, std::move(surfaces)};
}

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

// Builds a combined ITS(surfaces {0,1,2}, source 0)+MFT(surface {3}, source
// 1) TimeFrame with exactly one measurement on each of surfaces
// {0,1,3} (surface 2 is left empty, a deliberate hole in the catalog's own
// numbering -- not exercised by any track in these tests, only present to
// prove per-surface storage does not require every surface to be non-empty).
void loadThreeMeasurementFrame(TimeFrame& frame, const BuiltLayout& layout,
                               std::vector<std::vector<uint32_t>>* externalIndicesBySurface = nullptr,
                               std::vector<std::vector<uint32_t>>* clusterSizesBySurface = nullptr)
{
  if (!frame.isConfigured()) {
    DetectorLayoutDefinition definition;
    definition.componentOffsets.assign(layout.layout.getComponentOffsets().begin(), layout.layout.getComponentOffsets().end());
    definition.holeLayers = layout.layout.getHoleLayers();
    const auto catalog = layout.getCatalog();
    BOOST_REQUIRE(frame.configure(DetectorLayout{gsl::span<const SurfaceDescriptor>{catalog.surfaces, catalog.nSurfaces},
                                                 std::move(definition)},
                                  0, 0, std::make_shared<BoundedMemoryResource>()));
  }
  const std::vector<CompClusterExt> itsClusters{
    {10, 20, CompCluster::InvalidPatternID, 0},
    {11, 21, CompCluster::InvalidPatternID, 1},
  };
  const auto itsPatterns = makePatternBytes(itsClusters.size());
  const std::vector<ROFRecord> itsRofs{ROFRecord{{0, 0}, 0, 0, 2}};
  const std::array<LayerId, 2> itsLayerToSurface{LayerId{0}, LayerId{1}};
  static const FakeClusterDecoder itsDecoder{o2::detectors::DetID::ITS, false};

  const std::vector<CompClusterExt> mftClusters{{5, 6, CompCluster::InvalidPatternID, 0}};
  const auto mftPatterns = makePatternBytes(mftClusters.size());
  const std::vector<ROFRecord> mftRofs{ROFRecord{{0, 0}, 0, 0, 1}};
  const std::array<LayerId, 1> mftLayerToSurface{LayerId{3}};
  static const FakeClusterDecoder mftDecoder{o2::detectors::DetID::MFT, true};

  std::array<ClusterSourceInput, 2> sources{};
  sources[0].id = ClusterSourceId{0};
  sources[0].detector = o2::detectors::DetID::ITS;
  sources[0].clusters = itsClusters;
  sources[0].patterns = itsPatterns;
  sources[0].rofs = itsRofs;
  sources[0].dictionary = &dict();
  sources[0].layerToSurface = itsLayerToSurface;
  sources[0].timing = ROFTimingConfig{40, 0, 0, 0};
  sources[0].decoder = &itsDecoder;

  sources[1].id = ClusterSourceId{1};
  sources[1].detector = o2::detectors::DetID::MFT;
  sources[1].clusters = mftClusters;
  sources[1].patterns = mftPatterns;
  sources[1].rofs = mftRofs;
  sources[1].dictionary = &dict();
  sources[1].layerToSurface = mftLayerToSurface;
  sources[1].timing = ROFTimingConfig{50, 0, 0, 0};
  sources[1].decoder = &mftDecoder;

  BOOST_REQUIRE(loadSources(frame, layout.getCatalog(), gsl::span<const ClusterSourceInput>(sources), {0, 0},
                            externalIndicesBySurface, clusterSizesBySurface)
                  .ok());
}

} // namespace

BOOST_AUTO_TEST_CASE(SurfaceMeasurementStorageUsesStablePreSortIndices)
{
  const auto layout = makeCombinedLayout();
  TimeFrame frame;
  loadThreeMeasurementFrame(frame, layout);

  BOOST_REQUIRE_EQUAL(frame.getGlobalMeasurements(LayerId{0}).size(), 1u);
  BOOST_REQUIRE_EQUAL(frame.getGlobalMeasurements(LayerId{1}).size(), 1u);
  BOOST_REQUIRE_EQUAL(frame.getGlobalMeasurements(LayerId{2}).size(), 0u);
  BOOST_REQUIRE_EQUAL(frame.getGlobalMeasurements(LayerId{3}).size(), 1u);

  // The compact global on each layer carries its stable position in that
  // layer's pre-sort measurement arrays.
  const auto& onZero = frame.getGlobalMeasurements(LayerId{0})[0];
  const auto& onOne = frame.getGlobalMeasurements(LayerId{1})[0];
  const auto& onThree = frame.getGlobalMeasurements(LayerId{3})[0];
  BOOST_CHECK_EQUAL(onZero.clusterId, 0u);
  BOOST_CHECK_EQUAL(onOne.clusterId, 0u);
  BOOST_CHECK_EQUAL(onThree.clusterId, 0u);
  BOOST_CHECK(frame.getSurfaceMeasurement(LayerId{0}, onZero.clusterId) != nullptr);
  BOOST_CHECK(frame.getSurfaceMeasurement(LayerId{1}, onOne.clusterId) != nullptr);
  BOOST_CHECK(frame.getSurfaceMeasurement(LayerId{3}, onThree.clusterId) != nullptr);

  // An ID beyond the TimeFrame-owned surface's dense range is unresolved.
  BOOST_CHECK(frame.getSurfaceMeasurement(LayerId{0}, 99) == nullptr);
  // Surface 2 has zero measurements: even index 0 is out of range.
  BOOST_CHECK(frame.getGlobalMeasurements(LayerId{2}).empty());
  // Invalid surface id (out of range for a 4-surface catalog).
  BOOST_CHECK(frame.getSurfaceMeasurement(LayerId{4}, 0) == nullptr);
}

BOOST_AUTO_TEST_CASE(CrossSurfaceAndCrossSourceTrackClusterReferenceResolution)
{
  const auto layout = makeCombinedLayout();
  TimeFrame frame;
  loadThreeMeasurementFrame(frame, layout);

  // A single common track crossing the ITS/MFT source boundary, traversal
  // order inner to outer: surface 0 (ITS, source 0), surface 1 (ITS, source
  // 0), surface 3 (MFT, source 1) -- skipping surface 2 as a hole. Each
  // reference pairs the surface with that surface's own (surface-local)
  // measurement index, never a raw external cluster index or a global
  // position.
  const std::vector<TrackClusterReference> trackClusterIndices{
    {LayerId{0}, 0, 0},
    {LayerId{1}, 0, 0},
    {LayerId{3}, 0, 0},
  };

  GenericTrack track{};
  track.firstClusterRef = 0;
  track.clusterRefEnd = static_cast<uint32_t>(trackClusterIndices.size());
  track.hitLayers.set(0);
  track.hitLayers.set(1);
  track.hitLayers.set(3);
  BOOST_REQUIRE(isValidTrackRange(track, static_cast<uint32_t>(trackClusterIndices.size())));

  bool foundITSZero = false, foundITSOne = false, foundMFT = false;
  for (uint32_t i = track.firstClusterRef; i < track.clusterRefEnd; ++i) {
    const auto& reference = trackClusterIndices[i];
    const auto* measurement = frame.getSurfaceMeasurement(reference.layer, reference.clusterId);
    BOOST_REQUIRE(measurement != nullptr);
    if (reference.layer == LayerId{0}) {
      foundITSZero = true;
    } else if (reference.layer == LayerId{1}) {
      foundITSOne = true;
    } else if (reference.layer == LayerId{3}) {
      foundMFT = true;
    }
  }
  BOOST_CHECK(foundITSZero);
  BOOST_CHECK(foundITSOne);
  BOOST_CHECK(foundMFT);
}

BOOST_AUTO_TEST_CASE(HitSurfacesEqualsUnionAndEachMeasurementSurfaceMatchesItsReference)
{
  const auto layout = makeCombinedLayout();
  TimeFrame frame;
  loadThreeMeasurementFrame(frame, layout);

  const std::vector<TrackClusterReference> trackClusterIndices{
    {LayerId{0}, 0, 0},
    {LayerId{1}, 0, 0},
    {LayerId{3}, 0, 0},
  };

  GenericTrack track{};
  track.firstClusterRef = 0;
  track.clusterRefEnd = static_cast<uint32_t>(trackClusterIndices.size());
  track.hitLayers.set(0);
  track.hitLayers.set(1);
  track.hitLayers.set(3);

  LayerMask observed{};
  BOOST_REQUIRE(isValidTrackRange(track, static_cast<uint32_t>(trackClusterIndices.size())));
  for (uint32_t i = track.firstClusterRef; i < track.clusterRefEnd; ++i) {
    const auto& reference = trackClusterIndices[i];
    const auto* measurement = frame.getSurfaceMeasurement(reference.layer, reference.clusterId);
    BOOST_REQUIRE(measurement != nullptr);
    observed.set(reference.layer.value());
  }
  BOOST_CHECK(observed == track.hitLayers);

  // A hole-containing sub-track referencing only surfaces 0 and 3 (skipping
  // 1): still a valid, completed track, mask still matches exactly the
  // (smaller) referenced set.
  const std::vector<TrackClusterReference> holeIndices{
    {LayerId{0}, 0, 0},
    {LayerId{3}, 0, 0},
  };
  GenericTrack holeTrack{};
  holeTrack.firstClusterRef = 0;
  holeTrack.clusterRefEnd = 2;
  holeTrack.hitLayers.set(0);
  holeTrack.hitLayers.set(3);

  LayerMask observedHole{};
  BOOST_REQUIRE(isValidTrackRange(holeTrack, static_cast<uint32_t>(holeIndices.size())));
  for (uint32_t i = holeTrack.firstClusterRef; i < holeTrack.clusterRefEnd; ++i) {
    const auto& reference = holeIndices[i];
    const auto* measurement = frame.getSurfaceMeasurement(reference.layer, reference.clusterId);
    BOOST_REQUIRE(measurement != nullptr);
    observedHole.set(reference.layer.value());
  }
  BOOST_CHECK(observedHole == holeTrack.hitLayers);
  BOOST_CHECK(!observedHole.has(1)); // the hole
}

// --- TimeFrame reload/wipe lifecycle --------------------------------------

namespace
{

// Deterministic, geometry-free stand-in for GeometryClusterDecoder<DetId>
// (same construction as testTimeFrameLifecycle.cxx): sensorID is used
// directly as the detector-local layer.
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

struct TimeFrameFixture {
  TimeFrame tf;
  std::vector<std::vector<uint32_t>> externalIndicesBySurface;
  std::vector<std::vector<uint32_t>> clusterSizesBySurface;
  std::vector<LayerId> layerMapping{identitySurfaces(ITSNLayers)};
  // Keep the catalog with the layout fixture so initialization inputs have one
  // explicit owner.
  std::vector<SurfaceDescriptor> catalog{makeITSTestCatalog()};
  LegacyLikeDecoder decoder{o2::detectors::DetID::ITS};
  o2::InteractionRecord origin{50, 5};
  ROFTimingConfig timing{40, 0, 0, 0};

  TimeFrameFixture()
  {
    DetectorLayout layout{gsl::span<const SurfaceDescriptor>{catalog}, makeDetectorLayout()};
    BOOST_REQUIRE(tf.configure(std::move(layout), 0, 0,
                               std::make_shared<BoundedMemoryResource>()));
  }

  // One cluster on layer 0, one ROF: the minimal input that succeeds.
  LoadSourcesResult load()
  {
    const std::vector<CompClusterExt> clusters{{0, 1, CompCluster::InvalidPatternID, 0}};
    const auto patterns = makePatternBytes(clusters.size());
    const std::vector<ROFRecord> rofs{ROFRecord{{100, 5}, 0, 0, 1}};
    return loadTimeFrameSource(tf, decoder, origin, timing, clusters, patterns, rofs, &dict(), nullptr, o2::detectors::DetID::ITS,
                               gsl::span<const LayerId>{layerMapping}, tf.getLayout().getSurfaceCatalog(), true,
                               &externalIndicesBySurface, &clusterSizesBySurface);
  }
};

o2::its::LayerTiming makeFixtureClockTiming()
{
  // TimeFrameFixture::load() deliberately passes a temporary ROF vector to
  // the frame loader. Its overlap-table view is therefore not a retained
  // publication input. Build the same immutable clock timing explicitly for
  // output-boundary tests rather than dereferencing that non-owning view.
  o2::its::LayerTiming timing{};
  timing.mNROFsTF = 1;
  timing.mROFLength = 40;
  timing.mROFDelay = 100;
  return timing;
}

struct TestGenericTrack {
  GenericTrack track;
  std::vector<TrackClusterReference> references;
};

TestGenericTrack makeTestGenericTrack()
{
  TestGenericTrack record;
  record.track.innerState.kind = SurfaceKind::Cylinder;
  record.track.outerState.kind = SurfaceKind::Cylinder;
  record.track.timestamp = {100, 140};
  record.track.hitLayers.set(0);
  record.references.push_back({LayerId{0}, 0, 0});
  return record;
}

uint32_t storeTestGenericTrack(TimeFrame& frame, TestGenericTrack record)
{
  const auto index = static_cast<uint32_t>(frame.getGenericTracks().size());
  record.track.firstClusterRef = static_cast<uint32_t>(frame.getTrackClusterIndices().size());
  frame.getTrackClusterIndices().insert(frame.getTrackClusterIndices().end(), record.references.begin(), record.references.end());
  record.track.clusterRefEnd = static_cast<uint32_t>(frame.getTrackClusterIndices().size());
  frame.getGenericTracks().push_back(record.track);
  return index;
}

// Populates the common result sidecars with arbitrary, self-consistent
// content so a subsequent clear can be observed.
void populateCommonResults(TimeFrame& tf)
{
  tf.getTrackClusterIndices().push_back(TrackClusterReference{LayerId{0}, 0, 0});
  tf.getTrackClusterIndices().push_back(TrackClusterReference{LayerId{1}, 0, 1});
  GenericTrack track{};
  track.firstClusterRef = 0;
  track.clusterRefEnd = 2;
  track.hitLayers.set(0);
  track.hitLayers.set(1);
  tf.getGenericTracks().push_back(track);
  tf.getTrackLabels().push_back(o2::MCCompLabel{1, 0, 0, false});
}

} // namespace

BOOST_AUTO_TEST_CASE(SuccessfulReloadClearsCommonTrackResults)
{
  TimeFrameFixture fixture;
  BOOST_REQUIRE(fixture.load().ok());

  populateCommonResults(fixture.tf);
  BOOST_REQUIRE_EQUAL(fixture.tf.getGenericTracks().size(), 1u);
  BOOST_REQUIRE_EQUAL(fixture.tf.getTrackLabels().size(), 1u);
  BOOST_REQUIRE_EQUAL(fixture.tf.getTrackClusterIndices().size(), 2u);

  // A second, independently successful load on the same TimeFrame: the
  // normalized frame is replaced, and the common track result sidecars built
  // against the previous frame must be cleared in the same successful commit.
  BOOST_REQUIRE(fixture.load().ok());
  BOOST_CHECK(fixture.tf.getGenericTracks().empty());
  BOOST_CHECK(fixture.tf.getTrackLabels().empty());
  BOOST_CHECK(fixture.tf.getTrackClusterIndices().empty());
}

BOOST_AUTO_TEST_CASE(ITSSharedClusterCompatibilityUsesExplicitPreSortAssociations)
{
  struct MarkedTrack {
    bool shared = false;
    bool hasSharedClusters() const { return shared; }
  };

  TimeFrameFixture fixture;
  BOOST_REQUIRE(fixture.load().ok());
  const auto record = makeTestGenericTrack();
  ITSSharedClusterCompatibility sidecar;

  // Deliberately use a non-identity conceptual fclusSort permutation. The
  // status is read later from the original accepted slots, not this order.
  std::array<MarkedTrack, 3> accepted{{{false}, {true}, {false}}};
  const std::array<int, 3> fclusSort{{2, 0, 1}};
  BOOST_CHECK_NE(fclusSort[0], 0);
  for (size_t i = 0; i < accepted.size(); ++i) {
    ITSSharedClusterCompatibilityTransaction tx{sidecar};
    const auto index = storeTestGenericTrack(fixture.tf, record);
    BOOST_REQUIRE(tx.validate(index));
    tx.reserve();
    tx.append(index);
    BOOST_CHECK_EQUAL(index, i);
  }
  BOOST_CHECK_EQUAL(sidecar.pendingSize(), accepted.size());
  BOOST_CHECK(sidecar.sealFromMarkedTracks(accepted));
  BOOST_CHECK(sidecar.isSealed());
  BOOST_REQUIRE_EQUAL(sidecar.entries().size(), accepted.size());
  BOOST_CHECK_EQUAL(sidecar.entries()[0].genericTrackIndex, 0u);
  BOOST_CHECK(!sidecar.entries()[0].hasSharedClusters);
  BOOST_CHECK_EQUAL(sidecar.entries()[1].genericTrackIndex, 1u);
  BOOST_CHECK(sidecar.entries()[1].hasSharedClusters);

  // A later legacy output sort cannot change the already global-index-keyed
  // sealed result.
  std::reverse(accepted.begin(), accepted.end());
  BOOST_CHECK_EQUAL(sidecar.entries()[1].genericTrackIndex, 1u);
  BOOST_CHECK(sidecar.entries()[1].hasSharedClusters);

  // Scratch-only reset has no authority over TimeFrame-owned GenericTracks
  // or the bridge-owned compatibility result they index.
  fixture.tf.getScratch().reset();
  BOOST_CHECK_EQUAL(fixture.tf.getGenericTracks().size(), 3u);
  BOOST_CHECK_EQUAL(sidecar.entries().size(), 3u);

  ITSSharedClusterCompatibility malformed;
  const auto malformedIndex = storeTestGenericTrack(fixture.tf, record);
  ITSSharedClusterCompatibilityTransaction tx{malformed};
  BOOST_REQUIRE(tx.validate(malformedIndex));
  tx.reserve();
  tx.append(malformedIndex);
  BOOST_CHECK(!malformed.sealFromMarkedTracks(accepted)); // pending/track cardinality mismatch
  BOOST_CHECK(!malformed.isSealed());
  BOOST_CHECK(malformed.entries().empty());

  TimeFrameFixture rollbackFixture;
  BOOST_REQUIRE(rollbackFixture.load().ok());
  ITSSharedClusterCompatibility rollback;
  const auto rollbackIndex = storeTestGenericTrack(rollbackFixture.tf, record);
  ITSSharedClusterCompatibilityTransaction rollbackTx{rollback};
  BOOST_REQUIRE(rollbackTx.validate(rollbackIndex));
  rollbackTx.reserve();
  rollbackTx.append(rollbackIndex);
  BOOST_CHECK_EQUAL(rollback.pendingSize(), 1u);

  ITSSharedClusterCompatibility sealingFailure;
  ITSSharedClusterCompatibilityTransaction sealingTx{sealingFailure};
  const auto sealingIndex = storeTestGenericTrack(rollbackFixture.tf, record);
  BOOST_REQUIRE(sealingTx.validate(sealingIndex));
  sealingTx.reserve();
  sealingTx.append(sealingIndex);
  std::array<MarkedTrack, 1> oneTrack{{{true}}};
  BOOST_CHECK(sealingFailure.sealFromMarkedTracks(oneTrack));
  BOOST_CHECK(sealingFailure.isSealed());
  BOOST_REQUIRE_EQUAL(sealingFailure.entries().size(), 1u);

  sidecar.clear();
  fixture.tf.resetTimeFrame();
  BOOST_CHECK(fixture.tf.getGenericTracks().empty());
  BOOST_CHECK_EQUAL(sidecar.pendingSize(), 0u);
  BOOST_CHECK(sidecar.entries().empty());
}

BOOST_AUTO_TEST_CASE(FailedLoadClearsCommonTrackResults)
{
  TimeFrameFixture fixture;
  BOOST_REQUIRE(fixture.load().ok());

  populateCommonResults(fixture.tf);
  BOOST_REQUIRE_EQUAL(fixture.tf.getGenericTracks().size(), 1u);
  BOOST_REQUIRE_EQUAL(fixture.tf.getTrackLabels().size(), 1u);
  BOOST_REQUIRE_EQUAL(fixture.tf.getTrackClusterIndices().size(), 2u);
  BOOST_REQUIRE(fixture.tf.getTotalMeasurements() > 0u);

  // Deliberately fail: the frame loader preflight rejects an
  // unsupported detector before touching anything.
  const std::vector<CompClusterExt> clusters{{0, 1, CompCluster::InvalidPatternID, 0}};
  const auto patterns = makePatternBytes(clusters.size());
  const std::vector<ROFRecord> rofs{ROFRecord{{200, 5}, 0, 0, 1}};
  const auto& orderedSurfaces = fixture.layerMapping;
  const auto failed = loadTimeFrameSource(fixture.tf, fixture.decoder, fixture.origin, fixture.timing, clusters, patterns, rofs,
                                          &dict(), nullptr, o2::detectors::DetID::TPC,
                                          gsl::span<const LayerId>{orderedSurfaces}, fixture.tf.getLayout().getSurfaceCatalog());
  BOOST_REQUIRE(!failed.ok());
  BOOST_CHECK(failed.error == MultiSourceLoadError::UnsupportedDetector);

  BOOST_CHECK_EQUAL(fixture.tf.getTotalMeasurements(), 0u);
  BOOST_CHECK(fixture.tf.getGenericTracks().empty());
  BOOST_CHECK(fixture.tf.getTrackLabels().empty());
  BOOST_CHECK(fixture.tf.getTrackClusterIndices().empty());
}

BOOST_AUTO_TEST_CASE(TimeFrameWipeInvalidatesCommonTrackResultsTogether)
{
  TimeFrame tf;
  populateCommonResults(tf);

  BOOST_REQUIRE_EQUAL(tf.getGenericTracks().size(), 1u);
  BOOST_REQUIRE_EQUAL(tf.getTrackLabels().size(), 1u);
  BOOST_REQUIRE_EQUAL(tf.getTrackClusterIndices().size(), 2u);
  BOOST_REQUIRE(isValidTrackRange(tf.getGenericTracks()[0], static_cast<uint32_t>(tf.getTrackClusterIndices().size())));

  tf.resetTimeFrame();

  BOOST_CHECK(tf.getGenericTracks().empty());
  BOOST_CHECK(tf.getTrackLabels().empty());
  BOOST_CHECK(tf.getTrackClusterIndices().empty());

  // Reload after wipe: both containers accept new content independently of
  // whatever they held before, confirming they are ordinary per-event state
  // rather than something resetTimeFrame() leaves in a half-cleared condition.
  tf.getTrackClusterIndices().push_back(TrackClusterReference{LayerId{2}, 0, 0});
  GenericTrack reloaded{};
  reloaded.firstClusterRef = 0;
  reloaded.clusterRefEnd = 1;
  tf.getGenericTracks().push_back(reloaded);
  BOOST_CHECK_EQUAL(tf.getGenericTracks().size(), 1u);
  BOOST_CHECK_EQUAL(tf.getTrackClusterIndices().size(), 1u);
}

BOOST_AUTO_TEST_CASE(GenericTrackOutputAdapterTimestampIsSymmetricAndClamped)
{
  GenericTrackOutputAdapterError error = GenericTrackOutputAdapterError::None;
  o2::its::LayerTiming clock{};
  clock.mROFLength = 14;
  const ClockTimingPublicationView view{clock};
  const auto timestamp = makeOutputTimestamp({100, 120}, view, error);
  BOOST_REQUIRE(timestamp);
  BOOST_CHECK_EQUAL(timestamp->getTimeStamp(), 110.f);
  BOOST_CHECK_EQUAL(timestamp->getTimeStampError(), 7.f);
  BOOST_CHECK(!makeOutputTimestamp({20, 20}, view, error));
  BOOST_CHECK(error == GenericTrackOutputAdapterError::InvalidTimestamp);
}

BOOST_AUTO_TEST_CASE(GenericTrackOutputAdapterUsesLegacyPublicationOrder)
{
  TimeFrameFixture fixture;
  BOOST_REQUIRE(fixture.load().ok());

  auto later = makeTestGenericTrack();
  later.track.timestamp = {200, 240};
  later.track.chi2 = 1.f;
  auto earlier = makeTestGenericTrack();
  earlier.track.timestamp = {100, 140};
  earlier.track.chi2 = 2.f;
  BOOST_CHECK_EQUAL(storeTestGenericTrack(fixture.tf, later), 0u);
  BOOST_CHECK_EQUAL(storeTestGenericTrack(fixture.tf, earlier), 1u);

  o2::its::LayerTiming clock{};
  clock.mROFLength = 40;
  GenericTrackOutputAdapterError error = GenericTrackOutputAdapterError::None;
  const GenericTrackOutputAdapterSelection selection{{0u, 1u}};
  const auto ordered = makeLegacyOutputOrder(fixture.tf, selection, ClockTimingPublicationView{clock}, error);
  BOOST_REQUIRE(ordered);
  BOOST_REQUIRE_EQUAL(ordered->size(), 2u);
  BOOST_CHECK_EQUAL((*ordered)[0].globalIndex, 1u);
  BOOST_CHECK_EQUAL((*ordered)[1].globalIndex, 0u);
}

BOOST_AUTO_TEST_CASE(ClockTimingPublicationViewDelegatesLegacyClockSemantics)
{
  for (const uint32_t length : {9u, 10u}) {
    o2::its::LayerTiming legacy{};
    legacy.mNROFsTF = 4;
    legacy.mROFLength = length;
    legacy.mROFDelay = 3;
    legacy.mROFBias = 2;
    const ClockTimingPublicationView view{legacy};
    const std::array<GenericTrackTimestamp, 4> timestamps{{{5, 6}, {5, 5 + length}, {5 + length, 5 + 2 * length}, {5 + 3 * length, 5 + 4 * length}}};
    for (const auto timestamp : timestamps) {
      const auto asymmetric = view.makeTimeEstBC(timestamp);
      BOOST_REQUIRE(asymmetric);
      auto expected = asymmetric->makeSymmetrical();
      if (expected.getTimeStampError() > legacy.mROFLength * .5f)
        expected.setTimeStampError(legacy.mROFLength * .5f);
      const auto actual = view.makeOutputTimestamp(timestamp);
      BOOST_REQUIRE(actual);
      BOOST_CHECK_EQUAL(actual->getTimeStamp(), expected.getTimeStamp());
      BOOST_CHECK_EQUAL(actual->getTimeStampError(), expected.getTimeStampError());
      BOOST_CHECK_EQUAL(view.getROF(*actual), legacy.getROF(expected));
    }
  }
  o2::its::LayerTiming clock{};
  const ClockTimingPublicationView view{clock};
  BOOST_CHECK(!view.makeTimeEstBC({0, 0}));
  BOOST_CHECK(!view.makeTimeEstBC({-1, 1}));
  BOOST_CHECK(!view.makeTimeEstBC({0, static_cast<TFBC>(std::numeric_limits<uint32_t>::max()) + 1}));
  BOOST_CHECK(!view.makeTimeEstBC({0, static_cast<TFBC>(std::numeric_limits<uint16_t>::max()) + 1}));
}

BOOST_AUTO_TEST_CASE(ITSGenericPublicationPreservesClusterLayoutAndReordersLabelsWithoutMutatingSources)
{
  TimeFrameFixture fixture;
  fixture.externalIndicesBySurface = {{500000}, {0}, {7}, {42}, {0}, {123456}, {0}};
  fixture.clusterSizesBySurface = {{3}, {0}, {11}, {9}, {0}, {15}, {0}};
  auto later = makeTestGenericTrack();
  later.track.timestamp = {120, 130};
  later.track.hitLayers.set(2);
  later.track.hitLayers.set(5);
  later.references = {{LayerId{0}, 0, 0}, {LayerId{2}, 0, 0}, {LayerId{5}, 0, 0}};
  auto earlier = makeTestGenericTrack();
  earlier.track.timestamp = {100, 110};
  earlier.track.hitLayers = {};
  earlier.track.hitLayers.set(3);
  earlier.references = {{LayerId{3}, 0, 0}};
  ITSSharedClusterCompatibility shared;
  for (auto record : {later, earlier}) {
    const auto index = storeTestGenericTrack(fixture.tf, record);
    ITSSharedClusterCompatibilityTransaction transaction{shared};
    BOOST_REQUIRE(transaction.validate(index));
    transaction.reserve();
    transaction.append(index);
  }
  struct MarkedTrack {
    bool shared;
    bool hasSharedClusters() const { return shared; }
  };
  const std::array<MarkedTrack, 2> marked{{{true}, {false}}};
  BOOST_REQUIRE(shared.sealFromMarkedTracks(marked));
  const std::array<o2::MCCompLabel, 2> labels{{{7, 3, 1, true}, {8, 3, 1, false}}};
  fixture.tf.getTrackLabels().assign(labels.begin(), labels.end());
  const auto snapshotBytes = [](const auto& values) {
    const auto* first = reinterpret_cast<const unsigned char*>(values.data());
    return std::vector<unsigned char>(first, first + values.size() * sizeof(values[0]));
  };
  const auto tracksBefore = snapshotBytes(fixture.tf.getGenericTracks());
  const auto referencesBefore = snapshotBytes(fixture.tf.getTrackClusterIndices());
  const auto indicesBefore = fixture.externalIndicesBySurface;
  const auto sizesBefore = fixture.clusterSizesBySurface;
  const std::vector<ROFRecord> rofs{ROFRecord{{100, 5}, 0, 7, 3}};
  const GenericTrackPublicationContext context{o2::detectors::DetID::ITS, ClusterSourceId{0}, rofs,
                                               ClockTimingPublicationView{makeFixtureClockTiming()}, fixture.layerMapping,
                                               &fixture.externalIndicesBySurface, &fixture.clusterSizesBySurface};
  GenericTrackOutputAdapterError error{};
  // Every publication owns a fresh flattened range; repeated staging must not
  // change the frame or append onto the preceding publication's indices.
  for (int publication = 0; publication < 2; ++publication) {
    auto output = stageITSGenericTrackOutput(fixture.tf, context, shared, true, error);
    BOOST_REQUIRE(output);
    BOOST_REQUIRE_EQUAL(output->tracks.size(), 2u);
    const std::vector<int> expected{42, 123456, 7, 500000};
    BOOST_CHECK_EQUAL_COLLECTIONS(output->clusterIndices.begin(), output->clusterIndices.end(), expected.begin(), expected.end());
    BOOST_CHECK_EQUAL(output->tracks[0].getFirstClusterEntry(), 0);
    BOOST_CHECK_EQUAL(output->tracks[0].getNumberOfClusters(), 1);
    BOOST_CHECK_EQUAL(output->tracks[1].getFirstClusterEntry(), 1);
    BOOST_CHECK_EQUAL(output->tracks[1].getNumberOfClusters(), 3);
    BOOST_CHECK_EQUAL(output->tracks[0].getClusterSize(3), 9);
    for (int layer = 0; layer < ITSNLayers; ++layer) {
      const int expectedSize = layer == 0 ? 3 : layer == 2 ? 11
                                              : layer == 5 ? 15
                                                           : 0;
      BOOST_CHECK_EQUAL(output->tracks[1].getClusterSize(layer), expectedSize);
    }
    BOOST_CHECK(!output->tracks[0].hasSharedClusters());
    BOOST_CHECK(output->tracks[1].hasSharedClusters());
    BOOST_REQUIRE_EQUAL(output->labels.size(), 2u);
    BOOST_CHECK_EQUAL(output->labels[0].getRawValue(), labels[1].getRawValue());
    BOOST_CHECK_EQUAL(output->labels[1].getRawValue(), labels[0].getRawValue());
    BOOST_CHECK_EQUAL(output->trackROFs[0].getNEntries(), 2);
    BOOST_CHECK_EQUAL(output->trackROFs[0].getFlags(), rofs[0].getFlags());
    BOOST_CHECK(snapshotBytes(fixture.tf.getGenericTracks()) == tracksBefore);
    BOOST_CHECK(snapshotBytes(fixture.tf.getTrackClusterIndices()) == referencesBefore);
    BOOST_CHECK(fixture.externalIndicesBySurface == indicesBefore);
    BOOST_CHECK(fixture.clusterSizesBySurface == sizesBefore);
    BOOST_CHECK_EQUAL(fixture.tf.getTrackLabels()[0].getRawValue(), labels[0].getRawValue());
    BOOST_CHECK_EQUAL(fixture.tf.getTrackLabels()[1].getRawValue(), labels[1].getRawValue());
    BOOST_CHECK_EQUAL(rofs[0].getFirstEntry(), 7);
    BOOST_CHECK_EQUAL(rofs[0].getNEntries(), 3);
  }
}

BOOST_AUTO_TEST_CASE(ITSGenericPublicationOmitsTracksWithoutClusterReferences)
{
  TimeFrameFixture fixture;
  auto record = makeTestGenericTrack();
  record.references.clear();
  record.track.hitLayers = {};
  storeTestGenericTrack(fixture.tf, record);
  const std::vector<ROFRecord> rofs{ROFRecord{{100, 5}, 0, 7, 3}};
  const GenericTrackPublicationContext context{o2::detectors::DetID::ITS, ClusterSourceId{0}, rofs,
                                               ClockTimingPublicationView{makeFixtureClockTiming()}, fixture.layerMapping};
  ITSSharedClusterCompatibility unsealed;
  GenericTrackOutputAdapterError error{};
  const auto output = stageITSGenericTrackOutput(fixture.tf, context, unsealed, false, error);
  BOOST_REQUIRE(output);
  BOOST_CHECK(output->tracks.empty());
  BOOST_CHECK(output->clusterIndices.empty());
  BOOST_CHECK(output->labels.empty());
  BOOST_REQUIRE_EQUAL(output->trackROFs.size(), 1u);
  BOOST_CHECK_EQUAL(output->trackROFs[0].getFirstEntry(), 0);
  BOOST_CHECK_EQUAL(output->trackROFs[0].getNEntries(), 0);
  BOOST_CHECK_EQUAL(fixture.tf.getGenericTracks().size(), 1u);
}

BOOST_AUTO_TEST_CASE(GenericTrackOutputAdapterStagesITSAndFailsClosed)
{
  TimeFrameFixture fixture;
  BOOST_REQUIRE(fixture.load().ok());
  auto record = makeTestGenericTrack();
  record.track.chi2 = 3.f;
  ITSSharedClusterCompatibility shared;
  const auto genericTrackIndex = storeTestGenericTrack(fixture.tf, record);
  const o2::MCCompLabel storedLabel{7, 3, 1, true};
  fixture.tf.getTrackLabels().push_back(storedLabel);
  ITSSharedClusterCompatibilityTransaction transaction{shared};
  BOOST_REQUIRE(transaction.validate(genericTrackIndex));
  transaction.reserve();
  transaction.append(genericTrackIndex);
  struct MarkedTrack {
    bool shared{};
    bool hasSharedClusters() const { return shared; }
  };
  const std::array<MarkedTrack, 1> marked{{{true}}};
  BOOST_REQUIRE(shared.sealFromMarkedTracks(marked));
  const auto& measurement = fixture.tf.getGlobalMeasurements(LayerId{0})[0];
  BOOST_REQUIRE_EQUAL(measurement.clusterId, 0u);
  BOOST_REQUIRE_EQUAL(fixture.externalIndicesBySurface[0].size(), 1u);
  BOOST_REQUIRE_EQUAL(fixture.clusterSizesBySurface[0].size(), 1u);
  fixture.externalIndicesBySurface[0][measurement.clusterId] = 42u;
  fixture.clusterSizesBySurface[0][measurement.clusterId] = 13u;
  const auto source = ClusterSourceId{0};
  const std::vector<ROFRecord> rofs{ROFRecord{{100, 5}, 0, 7, 3}};
  GenericTrackOutputAdapterError error = GenericTrackOutputAdapterError::None;
  const auto clock = makeFixtureClockTiming();
  const GenericTrackOutputTimingContext timing{rofs, ClockTimingPublicationView{clock}};
  auto output = stageITSGenericTrackOutput(fixture.tf,
                                           gsl::span<const LayerId>{fixture.layerMapping}, timing, shared,
                                           true, error, &fixture.externalIndicesBySurface, &fixture.clusterSizesBySurface);
  BOOST_REQUIRE(output);
  BOOST_CHECK_EQUAL(output->tracks.size(), 1u);
  BOOST_CHECK_EQUAL(output->clusterIndices.size(), 1u);
  BOOST_CHECK_EQUAL(output->clusterIndices[0], 42);
  BOOST_CHECK_EQUAL(output->tracks[0].getClusterSize(0), 13);
  BOOST_CHECK(output->tracks[0].hasSharedClusters());
  BOOST_CHECK_EQUAL(output->tracks[0].getChi2(), 3.f);
  BOOST_CHECK_EQUAL(output->trackROFs[0].getFirstEntry(), 0);
  BOOST_CHECK_EQUAL(output->trackROFs[0].getNEntries(), 1);
  BOOST_CHECK_EQUAL(output->trackROFs[0].getFlags(), rofs[0].getFlags());
  BOOST_REQUIRE_EQUAL(output->labels.size(), 1u);
  BOOST_CHECK_EQUAL(output->labels[0].getRawValue(), storedLabel.getRawValue());

  // This is the workflow-facing binding: the workflow combines its own ROF
  // span with the immutable export returned by the tracking interface.  The
  // adapter accepts no scratch state and keeps the source/layout binding
  // explicit at this boundary.
  const GenericTrackPublicationContext publicationContext{
    o2::detectors::DetID::ITS, source, rofs, ClockTimingPublicationView{clock},
    gsl::span<const LayerId>{fixture.layerMapping},
    &fixture.externalIndicesBySurface, &fixture.clusterSizesBySurface};
  const auto contextOutput = stageITSGenericTrackOutput(fixture.tf, publicationContext, shared, true, error);
  BOOST_REQUIRE(contextOutput);
  BOOST_CHECK_EQUAL(contextOutput->tracks.size(), output->tracks.size());
  BOOST_REQUIRE_EQUAL(contextOutput->clusterIndices.size(), output->clusterIndices.size());
  for (size_t i = 0; i < output->clusterIndices.size(); ++i) {
    BOOST_CHECK_EQUAL(contextOutput->clusterIndices[i], output->clusterIndices[i]);
  }
  BOOST_CHECK_EQUAL(contextOutput->trackROFs.size(), output->trackROFs.size());

  fixture.tf.getTrackLabels().clear();
  BOOST_CHECK(!stageITSGenericTrackOutput(fixture.tf, publicationContext, shared, true, error));
  BOOST_CHECK(error == GenericTrackOutputAdapterError::MissingMCLabels);
  fixture.tf.getTrackLabels().push_back(storedLabel);

  auto wrongDetectorContext = publicationContext;
  wrongDetectorContext.detector = o2::detectors::DetID::MFT;
  BOOST_CHECK(!stageITSGenericTrackOutput(fixture.tf, wrongDetectorContext, shared, false, error));
  BOOST_CHECK(error == GenericTrackOutputAdapterError::MixedDetector);

  auto missingClusterSizesContext = publicationContext;
  missingClusterSizesContext.clusterSizesBySurface = nullptr;
  BOOST_CHECK(!stageITSGenericTrackOutput(fixture.tf, missingClusterSizesContext, shared, false, error));
  BOOST_CHECK(error == GenericTrackOutputAdapterError::UnresolvedReference);

  // Legacy publication retains a track even when its selected output
  // timestamp falls outside the workflow ROF span; it simply does not
  // increment a TrackROF entry. The adapter must preserve that behavior.
  fixture.tf.getGenericTracks()[0].timestamp = {1000, 1001};
  const auto outOfRangeOutput = stageITSGenericTrackOutput(fixture.tf, publicationContext, shared, false, error);
  BOOST_REQUIRE(outOfRangeOutput);
  BOOST_REQUIRE_EQUAL(outOfRangeOutput->tracks.size(), 1u);
  BOOST_CHECK_EQUAL(outOfRangeOutput->trackROFs[0].getFirstEntry(), 0);
  BOOST_CHECK_EQUAL(outOfRangeOutput->trackROFs[0].getNEntries(), 0);
  fixture.tf.getGenericTracks()[0].timestamp = record.track.timestamp;

  const auto oldTracks = fixture.tf.getGenericTracks().size();
  const auto oldReferences = fixture.tf.getTrackClusterIndices().size();
  // The original workflow ROF span can have more (or fewer) entries than
  // the LayerTiming clock. The GenericTrack adapter preserves that span
  // verbatim and groups only in-range clock slots.
  const std::vector<ROFRecord> mismatchedROFs{ROFRecord{{100, 5}, 0, 1, 2}, ROFRecord{{100, 6}, 1, 2, 3}};
  const GenericTrackOutputTimingContext mismatchedROF{mismatchedROFs, ClockTimingPublicationView{clock}};
  const auto mismatchedOutput = stageITSGenericTrackOutput(fixture.tf,
                                                           gsl::span<const LayerId>{fixture.layerMapping}, mismatchedROF, shared, false, error,
                                                           &fixture.externalIndicesBySurface, &fixture.clusterSizesBySurface);
  BOOST_REQUIRE(mismatchedOutput);
  BOOST_REQUIRE_EQUAL(mismatchedOutput->trackROFs.size(), mismatchedROFs.size());
  BOOST_CHECK_EQUAL(mismatchedOutput->trackROFs[0].getNEntries(), 1);
  BOOST_CHECK_EQUAL(mismatchedOutput->trackROFs[1].getNEntries(), 0);
  BOOST_CHECK_EQUAL(fixture.tf.getGenericTracks().size(), oldTracks);
  BOOST_CHECK_EQUAL(fixture.tf.getTrackClusterIndices().size(), oldReferences);
}

BOOST_AUTO_TEST_CASE(GenericTrackOutputAdapterStagesMFTCompatibilityWithoutSeedPt)
{
  const auto layout = makeCombinedLayout();
  TimeFrame frame;
  std::vector<std::vector<uint32_t>> externalIndicesBySurface;
  std::vector<std::vector<uint32_t>> clusterSizesBySurface;
  loadThreeMeasurementFrame(frame, layout, &externalIndicesBySurface, &clusterSizesBySurface);
  TestGenericTrack record;
  record.track.innerState.kind = SurfaceKind::Disk;
  record.track.outerState.kind = SurfaceKind::Disk;
  record.track.innerState.referenceCoordinate = -77.f;
  record.track.outerState.referenceCoordinate = -12.f;
  for (uint8_t i = 0; i < 5; ++i) {
    record.track.innerState.parameters[i] = 0.5f + i;
    record.track.outerState.parameters[i] = 3.5f + i;
  }
  for (uint8_t i = 0; i < 15; ++i) {
    record.track.innerState.covariance[i] = 0.01f * (i + 1);
    record.track.outerState.covariance[i] = 0.02f * (i + 1);
  }
  record.track.chi2 = 8.f;
  record.track.timestamp = {100, 124};
  record.track.hitLayers.set(3);
  record.references.push_back({LayerId{3}, 0, 0});
  storeTestGenericTrack(frame, record);
  const o2::MCCompLabel storedLabel{11, 4, 2, false};
  frame.getTrackLabels().push_back(storedLabel);
  const auto& measurement = frame.getGlobalMeasurements(LayerId{3})[0];
  const auto source = ClusterSourceId{1};
  const std::vector<ROFRecord> rofs{ROFRecord{{7, 9}, 2, 4, 5}};
  GenericTrackOutputAdapterError error = GenericTrackOutputAdapterError::None;
  o2::its::LayerTiming clock{};
  clock.mNROFsTF = 1;
  clock.mROFLength = 18;
  clock.mROFDelay = 100;
  const GenericTrackOutputTimingContext timing{rofs, ClockTimingPublicationView{clock}};
  const std::array<LayerId, 1> surfaces{LayerId{3}};
  const auto output = stageMFTGenericTrackOutput(frame, surfaces, timing, true, error,
                                                 &externalIndicesBySurface, &clusterSizesBySurface);
  BOOST_REQUIRE(output);
  BOOST_REQUIRE_EQUAL(output->tracks.size(), 1u);
  BOOST_CHECK_EQUAL(output->tracks[0].getZ(), -77.);
  BOOST_CHECK_EQUAL(output->tracks[0].getOutParam().getZ(), -12.);
  BOOST_CHECK_EQUAL(output->tracks[0].getCovariances()(4, 3), record.track.innerState.covariance[packedCovarianceIndex(4, 3)]);
  BOOST_CHECK_EQUAL(output->tracks[0].getOutParam().getCovariances()(4, 3), record.track.outerState.covariance[packedCovarianceIndex(4, 3)]);
  BOOST_CHECK_EQUAL(output->tracks[0].getTrackChi2(), 8.);
  BOOST_CHECK_EQUAL(output->tracks[0].getInvQPtSeed(), 0.);
  BOOST_CHECK_EQUAL(output->tracks[0].getChi2QPtSeed(), 0.);
  BOOST_REQUIRE_EQUAL(output->seedPatterns.size(), 1u);
  BOOST_CHECK_EQUAL(output->seedPatterns[0], 0x1u);
  BOOST_CHECK_EQUAL(output->clusterIndices[0], static_cast<int>(measurement.clusterId));
  BOOST_CHECK_EQUAL(output->trackROFs[0].getFirstEntry(), 0);
  BOOST_CHECK_EQUAL(output->trackROFs[0].getNEntries(), 1);
  BOOST_CHECK_EQUAL(output->trackROFs[0].getFlags(), rofs[0].getFlags());
  BOOST_REQUIRE_EQUAL(output->labels.size(), 1u);
  BOOST_CHECK_EQUAL(output->labels[0].getRawValue(), storedLabel.getRawValue());

  const GenericTrackPublicationContext publicationContext{
    o2::detectors::DetID::MFT, source, rofs, ClockTimingPublicationView{clock}, surfaces,
    &externalIndicesBySurface, &clusterSizesBySurface};
  const auto contextOutput = stageMFTGenericTrackOutput(frame, publicationContext, true, error);
  BOOST_REQUIRE(contextOutput);
  BOOST_CHECK_EQUAL(contextOutput->tracks.size(), output->tracks.size());
  BOOST_CHECK_EQUAL_COLLECTIONS(contextOutput->seedPatterns.begin(), contextOutput->seedPatterns.end(),
                                output->seedPatterns.begin(), output->seedPatterns.end());

  auto wrongDetectorContext = publicationContext;
  wrongDetectorContext.detector = o2::detectors::DetID::ITS;
  BOOST_CHECK(!stageMFTGenericTrackOutput(frame, wrongDetectorContext, false, error));
  BOOST_CHECK(error == GenericTrackOutputAdapterError::MixedDetector);
  BOOST_CHECK_EQUAL(frame.getGenericTracks().size(), 1u);
  BOOST_CHECK_EQUAL(frame.getTrackClusterIndices().size(), 1u);
}

BOOST_AUTO_TEST_CASE(GenericTrackOutputAdapterRejectsMalformedInputsWithoutMutatingOwners)
{
  TimeFrameFixture fixture;
  BOOST_REQUIRE(fixture.load().ok());
  const auto record = makeTestGenericTrack();
  storeTestGenericTrack(fixture.tf, record);
  const std::vector<ROFRecord> rofs{ROFRecord{{1, 2}, 0, 0, 1}};
  const auto clock = makeFixtureClockTiming();
  const GenericTrackOutputTimingContext timing{rofs, ClockTimingPublicationView{clock}};
  const auto surfaces = gsl::span<const LayerId>{fixture.layerMapping};
  const auto tracks = fixture.tf.getGenericTracks().size();
  const auto refs = fixture.tf.getTrackClusterIndices().size();
  const auto measurements = fixture.tf.getTotalMeasurements();
  GenericTrackOutputAdapterError error = GenericTrackOutputAdapterError::None;
  ITSSharedClusterCompatibility unsealed;
  BOOST_CHECK(!stageITSGenericTrackOutput(fixture.tf, surfaces, timing, unsealed, false, error));
  BOOST_CHECK(error == GenericTrackOutputAdapterError::MissingCompatibility);
  const std::array<LayerId, 1> foreignSurfaces{LayerId{3}};
  const auto foreignSelection = stageITSGenericTrackOutput(fixture.tf, foreignSurfaces, timing, unsealed, false, error);
  BOOST_REQUIRE(foreignSelection);
  BOOST_CHECK(foreignSelection->tracks.empty());
  BOOST_CHECK_EQUAL(fixture.tf.getGenericTracks().size(), tracks);
  BOOST_CHECK_EQUAL(fixture.tf.getTrackClusterIndices().size(), refs);
  BOOST_CHECK_EQUAL(fixture.tf.getTotalMeasurements(), measurements);

  fixture.tf.getGenericTracks()[0].clusterRefEnd = refs + 1;
  BOOST_CHECK(!selectGenericTracksForSurfaces(fixture.tf, surfaces, error));
  BOOST_CHECK(error == GenericTrackOutputAdapterError::InvalidTrackRange);
  fixture.tf.getGenericTracks()[0].clusterRefEnd = refs;
  fixture.tf.getTrackClusterIndices()[0].layer = LayerId::invalid();
  BOOST_CHECK(!selectGenericTracksForSurfaces(fixture.tf, surfaces, error));
  BOOST_CHECK(error == GenericTrackOutputAdapterError::UnresolvedReference);
  fixture.tf.getTrackClusterIndices()[0].layer = LayerId{0};

  ITSSharedClusterCompatibility sealed;
  ITSSharedClusterCompatibilityTransaction tx{sealed};
  const auto secondTrackIndex = storeTestGenericTrack(fixture.tf, record);
  BOOST_REQUIRE(tx.validate(secondTrackIndex));
  tx.reserve();
  tx.append(secondTrackIndex);
  struct Marked {
    bool hasSharedClusters() const { return false; }
  };
  const std::array<Marked, 0> none{};
  BOOST_CHECK(!sealed.sealFromMarkedTracks(none)); // pending cardinality mismatch fails closed
  BOOST_CHECK(!sealed.isSealed());
}
