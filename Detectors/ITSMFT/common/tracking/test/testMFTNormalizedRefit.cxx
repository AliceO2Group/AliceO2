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

// Focused normalized-measurement authority and covariance coverage for the
// descriptor-driven seed refit path.

#define BOOST_TEST_MODULE ITSMFT MFTNormalizedRefit
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <boost/test/unit_test.hpp>

#include "ITSMFTTracking/RefitDriver.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"
#include "ITSMFTTracking/TimeFrame.h"
#include "ITSMFTTracking/Constants.h"
#include "MFTTracking/Constants.h"

using namespace o2::itsmft::tracking;

namespace
{

constexpr int NLayers = o2::mft::constants::mft::LayersNumber;
// Field-off exercises the native linear propagation model deterministically.
constexpr float Bz = 0.f;
constexpr float DefaultSigma2 = 2.5e-7f; // (~0.5 micron)^2, MFT-scale resolution

SurfaceTrackState makeDiskRefitStateFixture(
  const SurfaceMeasurement& inner, const SurfaceMeasurement& outer,
  float trackletMinPt)
{
  const float dx = outer.frame.u - inner.frame.u;
  const float dy = outer.frame.v - inner.frame.v;
  const float transverseLength = std::hypot(dx, dy);
  const float qOverPt = trackletMinPt > 0.f ? 1.f / trackletMinPt : 0.f;

  SurfaceTrackState state{};
  state.referenceCoordinate = outer.frame.q;
  state.parameters[0] = outer.frame.u;
  state.parameters[1] = outer.frame.v;
  state.parameters[2] = std::atan2(dy, dx);
  state.parameters[3] = (outer.frame.q - inner.frame.q) / transverseLength;
  state.parameters[4] = qOverPt;
  state.covariance[packedCovarianceIndex(0, 0)] = outer.covariance.uu;
  state.covariance[packedCovarianceIndex(1, 0)] = outer.covariance.uv;
  state.covariance[packedCovarianceIndex(1, 1)] = outer.covariance.vv;
  state.covariance[packedCovarianceIndex(2, 2)] = 1.f;
  state.covariance[packedCovarianceIndex(3, 3)] = 1.f;
  const float qOverPtSigma = std::clamp(std::abs(qOverPt), 1.f, 10.f);
  state.covariance[packedCovarianceIndex(4, 4)] = qOverPtSigma * qOverPtSigma;
  state.kind = SurfaceKind::Disk;
  state.absCharge = 1;
  state.pid = o2::track::PID::Pion;
  return state;
}

// A straight track through every MFT disk.
struct StraightTrackGeometry {
  std::array<float, NLayers> x{};
  std::array<float, NLayers> y{};
  std::array<float, NLayers> z{};
  float xSlope{};

  explicit StraightTrackGeometry(float slope) : xSlope(slope)
  {
    const auto zLayer = o2::mft::constants::mft::LayerZCoordinate();
    const float z0 = zLayer[0];
    for (int layer = 0; layer < NLayers; ++layer) {
      z[layer] = zLayer[layer];
      x[layer] = 1.f + xSlope * (z[layer] - z0);
      y[layer] = 0.5f - 0.006f * (z[layer] - z0);
    }
  }
};

// Owns one normalized refit fixture.
struct RefitFixture {
  std::array<std::vector<SurfaceMeasurement>, NLayers> storage;
  std::array<std::vector<GlobalMeasurement>, NLayers> globalStorage;
  std::vector<gsl::span<const GlobalMeasurement>> layerGlobals = std::vector<gsl::span<const GlobalMeasurement>>(NLayers);
  std::vector<SurfaceDescriptor> catalogSurfaces;
  SurfaceCatalogView catalog{};
  TimeFrame frame;
  TrackSeed seed;
  o2::itsmft::TrackingParameters params;
  int nHitLayers{0};

  explicit RefitFixture(const StraightTrackGeometry& geometry, int hits = NLayers)
    : nHitLayers(hits)
  {
    params.MinTrackLength = 5;
    params.MinPt.assign(NLayers + 1, 0.f);
    params.MaxChi2NDF = 30.f;

    catalogSurfaces.resize(NLayers);
    for (int layer = 0; layer < NLayers; ++layer) {
      catalogSurfaces[layer].detectorSurfaceIndex = static_cast<uint16_t>(layer);
      catalogSurfaces[layer].kind = SurfaceKind::Disk;
      catalogSurfaces[layer].material = NominalSurfaceMaterial{0.f, 0.f};
    }
    catalog = SurfaceCatalogView{catalogSurfaces.data(), static_cast<uint32_t>(catalogSurfaces.size())};
    BOOST_REQUIRE(frame.configure(DetectorLayout{catalogSurfaces, makeDetectorLayout()}, 0, 0,
                                  std::make_shared<BoundedMemoryResource>()));

    uint16_t mask = 0;
    for (int layer = 0; layer < hits; ++layer) {
      setMeasurement(layer, geometry.x[layer], geometry.y[layer], geometry.z[layer],
                     DefaultSigma2, DefaultSigma2);
      seed.getClusters()[layer] = 0;
      mask |= static_cast<uint16_t>(uint16_t(1) << layer);
    }
    seed.setHitLayerMask(LayerMask{mask});

    // The native driver starts from the CA seed state.
    const int innerLayer = 0;
    const int outerLayer = hits - 1;
    seed.state() = makeDiskRefitStateFixture(
      storage[innerLayer][0], storage[outerLayer][0], params.TrackletMinPt);
  }

  void setMeasurement(int layer, float x, float y, float z, float uu, float vv, float uv = 0.f)
  {
    SurfaceMeasurement m{};
    // Disk measurements propagate to frame.q, their global z coordinate.
    m.frame = {z, x, y, 0.f};
    m.covariance.uu = uu;
    m.covariance.vv = vv;
    m.covariance.uv = uv;
    GlobalMeasurement global{};
    global.position = {x, y, z};
    global.radius = std::hypot(x, y);
    global.covariance = {uu, uv, 0.f, vv, 0.f, 0.f};
    global.clusterId = 0u;
    storage[layer].assign(1, m);
    globalStorage[layer].assign(1, global);
    layerGlobals[layer] = globalStorage[layer];
  }

  void syncFrame()
  {
    frame.resetTimeFrame();
    for (int layer = 0; layer < NLayers; ++layer) {
      for (std::size_t cluster = 0; cluster < globalStorage[layer].size(); ++cluster) {
        frame.addMeasurement(LayerId{static_cast<uint16_t>(layer)}, globalStorage[layer][cluster],
                             storage[layer][cluster]);
      }
    }
  }
};

bool refit(RefitFixture& fixture, TrackingCandidate& candidate)
{
  fixture.syncFrame();
  SurfaceTrackState innerState{};
  SurfaceTrackState outerState{};
  float chi2 = 0.f;
  OperationFailureReason reason{};
  if (!fitTrackSeedLegs(fixture.seed, fixture.frame, fixture.layerGlobals, fixture.catalog, Bz,
                        fixture.params.ShiftRefToCluster, fixture.params.MaxChi2ClusterAttachment,
                        fixture.params.MaxChi2NDF, fixture.params.RepeatRefitOut,
                        gsl::span<const float>(fixture.params.MinPt),
                        innerState, outerState, chi2, reason)) {
    return false;
  }
  candidate.seed = fixture.seed;
  candidate.track.innerState = innerState;
  candidate.track.outerState = outerState;
  candidate.track.chi2 = chi2;
  return true;
}

void checkTrackUnchanged(const TrackingCandidate& before, const TrackingCandidate& after)
{
  BOOST_CHECK_EQUAL(before.seed.getHitLayerMask().value(), after.seed.getHitLayerMask().value());
  for (int position = 0; position < TrackSeed::MaxSurfaces; ++position) {
    BOOST_CHECK_EQUAL(before.seed.getCluster(position), after.seed.getCluster(position));
  }
  for (int i = 0; i < 5; ++i) {
    BOOST_CHECK_EQUAL(before.track.innerState.parameters[i], after.track.innerState.parameters[i]);
    BOOST_CHECK_EQUAL(before.track.outerState.parameters[i], after.track.outerState.parameters[i]);
  }
  for (int i = 0; i < 15; ++i) {
    BOOST_CHECK_EQUAL(before.track.innerState.covariance[i], after.track.innerState.covariance[i]);
    BOOST_CHECK_EQUAL(before.track.outerState.covariance[i], after.track.outerState.covariance[i]);
  }
  BOOST_CHECK_EQUAL(before.track.innerState.referenceCoordinate, after.track.innerState.referenceCoordinate);
  BOOST_CHECK_EQUAL(before.track.outerState.referenceCoordinate, after.track.outerState.referenceCoordinate);
  BOOST_CHECK_EQUAL(static_cast<int>(before.track.innerState.kind), static_cast<int>(after.track.innerState.kind));
  BOOST_CHECK_EQUAL(static_cast<int>(before.track.outerState.kind), static_cast<int>(after.track.outerState.kind));
  BOOST_CHECK_EQUAL(before.track.chi2, after.track.chi2);
  BOOST_CHECK_EQUAL(before.phi, after.phi);
  BOOST_CHECK_EQUAL(before.eta, after.eta);
  BOOST_CHECK_EQUAL(before.charge, after.charge);
}

} // namespace

// --- Normalized data drives the output --------------------------------------

BOOST_AUTO_TEST_CASE(NormalizedGlobalCoordinateChangeAltersOutput)
{
  const StraightTrackGeometry geometry(0.3f);

  RefitFixture reference(geometry);
  TrackingCandidate referenceTrack;
  BOOST_REQUIRE(refit(reference, referenceTrack));

  // Perturb only the normalized global.x of one interior layer -- legacy
  // backfill is absent (never populated) in both fixtures, so this isolates
  // the normalized measurement as the sole cause of the changed outcome. The
  // shift is far larger than DefaultSigma2's resolution, so the previously
  // ~0 chi2/ndf now certainly exceeds MaxChi2NDF.
  RefitFixture perturbed(geometry);
  auto perturbedMeasurement = perturbed.storage[5].front();
  perturbedMeasurement.frame.u += 0.05f;
  perturbed.storage[5].assign(1, perturbedMeasurement);

  TrackingCandidate perturbedTrack;
  const bool perturbedOk = refit(perturbed, perturbedTrack);
  BOOST_CHECK(!perturbedOk);
}

BOOST_AUTO_TEST_CASE(NormalizedCovarianceChangeAltersOutput)
{
  const StraightTrackGeometry geometry(0.3f);

  RefitFixture reference(geometry);
  TrackingCandidate referenceTrack;
  BOOST_REQUIRE(refit(reference, referenceTrack));

  // Scale up every layer's diagonal covariance uniformly (legacy backfill
  // again absent in both fixtures): with exact-colinear points the fitted
  // position/chi2 are unaffected, but the posterior parameter covariance the
  // Kalman filter propagates is not -- a strictly larger measurement variance
  // must not shrink the output covariance. This is a generic Kalman-filter
  // property, unaffected by which per-hit update formula produces it.
  RefitFixture loose(geometry);
  for (int layer = 0; layer < NLayers; ++layer) {
    auto m = loose.storage[layer].front();
    m.covariance.uu *= 400.f;
    m.covariance.vv *= 400.f;
    loose.storage[layer].assign(1, m);
  }
  TrackingCandidate looseTrack;
  BOOST_REQUIRE(refit(loose, looseTrack));

  BOOST_CHECK_GT(looseTrack.track.outerState.covariance[packedCovarianceIndex(0, 0)],
                 referenceTrack.track.outerState.covariance[packedCovarianceIndex(0, 0)]);
  BOOST_CHECK_GT(looseTrack.track.outerState.covariance[packedCovarianceIndex(1, 1)],
                 referenceTrack.track.outerState.covariance[packedCovarianceIndex(1, 1)]);
}

// --- C. Invalid normalized input fails cleanly, destination untouched -------

BOOST_AUTO_TEST_CASE(NonFiniteSurfaceCoordinateFailsCleanly)
{
  const StraightTrackGeometry geometry(0.3f);
  RefitFixture fx(geometry);
  auto m = fx.storage[3].front();
  m.frame.u = std::numeric_limits<float>::quiet_NaN();
  fx.storage[3].assign(1, m);

  TrackingCandidate before;
  TrackingCandidate track = before;
  BOOST_CHECK(!refit(fx, track));
  checkTrackUnchanged(before, track);
}

BOOST_AUTO_TEST_CASE(InfiniteSurfaceCoordinateFailsCleanly)
{
  const StraightTrackGeometry geometry(0.3f);
  RefitFixture fx(geometry);
  auto m = fx.storage[3].front();
  m.frame.q = std::numeric_limits<float>::infinity();
  fx.storage[3].assign(1, m);

  TrackingCandidate before;
  TrackingCandidate track = before;
  BOOST_CHECK(!refit(fx, track));
  checkTrackUnchanged(before, track);
}

BOOST_AUTO_TEST_CASE(NonFiniteCovarianceFailsCleanly)
{
  const StraightTrackGeometry geometry(0.3f);
  RefitFixture fx(geometry);
  auto m = fx.storage[3].front();
  m.covariance.uu = std::numeric_limits<float>::quiet_NaN();
  fx.storage[3].assign(1, m);

  TrackingCandidate before;
  TrackingCandidate track = before;
  BOOST_CHECK(!refit(fx, track));
  checkTrackUnchanged(before, track);
}

BOOST_AUTO_TEST_CASE(NegativeCovarianceFailsCleanly)
{
  const StraightTrackGeometry geometry(0.3f);
  RefitFixture fx(geometry);
  auto m = fx.storage[3].front();
  m.covariance.vv = -1.f;
  fx.storage[3].assign(1, m);

  TrackingCandidate before;
  TrackingCandidate track = before;
  BOOST_CHECK(!refit(fx, track));
  checkTrackUnchanged(before, track);
}

BOOST_AUTO_TEST_CASE(OutOfRangeClusterIndexFailsCleanly)
{
  const StraightTrackGeometry geometry(0.3f);
  RefitFixture fx(geometry);
  fx.seed.getClusters()[3] = 99; // storage[3] only ever has one element (index 0)

  TrackingCandidate before;
  TrackingCandidate track = before;
  BOOST_CHECK(!refit(fx, track));
  checkTrackUnchanged(before, track);
}

BOOST_AUTO_TEST_CASE(InvalidClusterRefFailsCleanly)
{
  const StraightTrackGeometry geometry(0.3f);
  RefitFixture fx(geometry);
  auto m = fx.globalStorage[3].front();
  m.clusterId = std::numeric_limits<uint32_t>::max();
  fx.globalStorage[3].assign(1, m);
  fx.layerGlobals[3] = fx.globalStorage[3];

  TrackingCandidate before;
  TrackingCandidate track = before;
  BOOST_CHECK(!refit(fx, track));
  checkTrackUnchanged(before, track);
}

BOOST_AUTO_TEST_CASE(RefitRejectsInvalidSurfaceCountsWithoutChangingOutput)
{
  RefitFixture fixture(StraightTrackGeometry{0.3f});
  fixture.syncFrame();
  for (const std::size_t count : {std::size_t{0}, std::size_t{MaxLayoutSurfaces + 1}}) {
    std::vector<gsl::span<const GlobalMeasurement>> layers(count);
    TrackingCandidate before;
    before.track.innerState = fixture.seed.state();
    before.track.outerState = fixture.seed.state();
    before.track.chi2 = 123.f;
    auto after = before;
    OperationFailureReason reason{};
    BOOST_CHECK(!fitTrackSeedLegs(fixture.seed, fixture.frame, layers, fixture.catalog, Bz,
                                  fixture.params.ShiftRefToCluster, fixture.params.MaxChi2ClusterAttachment,
                                  fixture.params.MaxChi2NDF, true, fixture.params.MinPt,
                                  after.track.innerState, after.track.outerState, after.track.chi2, reason));
    BOOST_CHECK(reason == OperationFailureReason::InvalidSurfaceCatalogAssociation);
    checkTrackUnchanged(before, after);
  }
}

BOOST_AUTO_TEST_CASE(RefitBufferHandlesMaximumLayoutAndRepeatedLegs)
{
  RefitFixture fixture(StraightTrackGeometry{0.3f});
  for (const bool repeat : {false, true}) {
    fixture.params.RepeatRefitOut = repeat;
    fixture.layerGlobals.resize(NLayers);
    TrackingCandidate compact;
    BOOST_REQUIRE(refit(fixture, compact));
    // Additional absent surfaces must neither overflow the bounded buffer
    // nor retain a measurement from the preceding refit leg.
    fixture.layerGlobals.resize(MaxLayoutSurfaces);
    TrackingCandidate maximum;
    BOOST_REQUIRE(refit(fixture, maximum));
    checkTrackUnchanged(compact, maximum);
  }
}

// --- D. Preservation ---------------------------------------------------------

BOOST_AUTO_TEST_CASE(PreservesSeedMembershipForGenericRefit)
{
  const StraightTrackGeometry geometry(0.3f);
  // Holes at layers 2 and 7: MinTrackLength(5) <= 8 remaining hits.
  RefitFixture fx(geometry);
  fx.seed.getClusters()[2] = o2::its::constants::UnusedIndex;
  fx.seed.getClusters()[7] = o2::its::constants::UnusedIndex;
  LayerMask mask = fx.seed.getHitLayerMask();
  mask.reset(2);
  mask.reset(7);
  fx.seed.setHitLayerMask(mask);

  TrackingCandidate track;
  BOOST_REQUIRE(refit(fx, track));

  BOOST_CHECK_EQUAL(track.getNumberOfClusters(), NLayers - 2);
  for (int layer = 0; layer < NLayers; ++layer) {
    if (layer == 2 || layer == 7) {
      BOOST_CHECK_EQUAL(track.getClusterIndex(layer), o2::its::constants::UnusedIndex);
      BOOST_CHECK(!track.seed.hasCluster(layer));
    } else {
      BOOST_CHECK_EQUAL(track.getClusterIndex(layer), 0);
      BOOST_CHECK(track.seed.hasCluster(layer));
    }
  }
}

// The native update uses the full uu/uv/vv measurement covariance.
BOOST_AUTO_TEST_CASE(OffDiagonalCovarianceIsUsedByNativeUpdate)
{
  const StraightTrackGeometry geometry(0.3f);

  RefitFixture reference(geometry);
  TrackingCandidate referenceTrack;
  BOOST_REQUIRE(refit(reference, referenceTrack));

  RefitFixture withUv(geometry);
  // A generous per-hit/per-track chi2 gate: this test's goal is only to
  // prove a physically valid off-diagonal correlation changes the native
  // update's output, not to probe chi2-gate behavior -- a nonzero
  // correlation legitimately raises the predicted chi2 against a reference
  // fit tuned for the uncorrelated (uv == 0) case.
  withUv.params.MaxChi2ClusterAttachment = 1.e4f;
  withUv.params.MaxChi2NDF = 1.e4f;
  for (int layer = 0; layer < NLayers; ++layer) {
    auto m = withUv.storage[layer].front();
    // A modest, physically valid correlation (|coefficient| << 1) suffices
    // to prove the point.
    m.covariance.uv = 0.05f * std::sqrt(m.covariance.uu * m.covariance.vv);
    withUv.storage[layer].assign(1, m);
  }
  TrackingCandidate withUvTrack;
  BOOST_REQUIRE(refit(withUv, withUvTrack));

  BOOST_CHECK_NE(withUvTrack.track.outerState.covariance[packedCovarianceIndex(0, 0)],
                 referenceTrack.track.outerState.covariance[packedCovarianceIndex(0, 0)]);
}

// --- Regression: stable pre-sort seed-cluster identity ---------------------

BOOST_AUTO_TEST_CASE(GenericRefitUsesStablePreSortClusterIdentity)
{
  // Every hit layer has sorted seed index zero pointing at pre-sort cluster
  // ID one. The generic refit must use that stable ID to retrieve the matching
  // SurfaceMeasurement, rather than treating the sorted position as the ID.
  const StraightTrackGeometry geometry(0.3f);
  std::array<std::vector<SurfaceMeasurement>, NLayers> storage;
  std::array<std::vector<GlobalMeasurement>, NLayers> globalStorage;
  std::vector<gsl::span<const GlobalMeasurement>> layerGlobals = std::vector<gsl::span<const GlobalMeasurement>>(NLayers);
  std::vector<SurfaceDescriptor> catalogSurfaces(NLayers);
  for (int layer = 0; layer < NLayers; ++layer) {
    catalogSurfaces[layer].kind = SurfaceKind::Disk;
    catalogSurfaces[layer].material = NominalSurfaceMaterial{0.f, 0.f};
  }
  SurfaceCatalogView catalog{catalogSurfaces.data(), static_cast<uint32_t>(catalogSurfaces.size())};
  TrackSeed seed;
  o2::itsmft::TrackingParameters params;
  params.MinTrackLength = 5;
  params.MinPt.assign(NLayers + 1, 0.f);
  params.MaxChi2NDF = 30.f;

  uint16_t mask = 0;
  for (int layer = 0; layer < NLayers; ++layer) {
    SurfaceMeasurement m{};
    m.frame = {geometry.z[layer], geometry.x[layer], geometry.y[layer], 0.f};
    m.covariance.uu = DefaultSigma2;
    m.covariance.vv = DefaultSigma2;
    auto distractor = m;
    distractor.covariance.uu = std::numeric_limits<float>::quiet_NaN();
    GlobalMeasurement global{};
    global.position = {geometry.x[layer], geometry.y[layer], geometry.z[layer]};
    global.radius = std::hypot(geometry.x[layer], geometry.y[layer]);
    global.covariance = {DefaultSigma2, 0.f, 0.f, DefaultSigma2, 0.f, 0.f};
    global.clusterId = 1u;
    auto distractorGlobal = global;
    distractorGlobal.position.x += 100.f;
    distractorGlobal.radius = std::hypot(distractorGlobal.position.x, distractorGlobal.position.y);
    distractorGlobal.clusterId = 0u;
    storage[layer] = {distractor, m};
    globalStorage[layer] = {global, distractorGlobal};
    layerGlobals[layer] = globalStorage[layer];

    seed.getClusters()[layer] = 0;
    mask |= static_cast<uint16_t>(uint16_t(1) << layer);
  }
  seed.setHitLayerMask(LayerMask{mask});

  seed.state() = makeDiskRefitStateFixture(
    storage[0][1], storage[NLayers - 1][1], params.TrackletMinPt);

  TrackingCandidate track;
  std::vector<std::vector<GlobalMeasurement>> globals(NLayers);
  std::vector<std::vector<SurfaceMeasurement>> measurements(NLayers);
  for (int layer = 0; layer < NLayers; ++layer) {
    globals[layer] = globalStorage[layer];
    measurements[layer] = storage[layer];
  }
  TimeFrame frame;
  BOOST_REQUIRE(frame.configure(DetectorLayout{catalogSurfaces, makeDetectorLayout()}, 0, 0,
                                std::make_shared<BoundedMemoryResource>()));
  for (int layer = 0; layer < NLayers; ++layer) {
    for (std::size_t cluster = 0; cluster < globals[layer].size(); ++cluster) {
      frame.addMeasurement(LayerId{static_cast<uint16_t>(layer)}, globals[layer][cluster],
                           measurements[layer][cluster]);
    }
  }
  SurfaceTrackState innerState{};
  SurfaceTrackState outerState{};
  float chi2 = 0.f;
  OperationFailureReason reason{};
  BOOST_REQUIRE(fitTrackSeedLegs(seed, frame, layerGlobals, catalog, Bz,
                                 params.ShiftRefToCluster, params.MaxChi2ClusterAttachment, params.MaxChi2NDF,
                                 params.RepeatRefitOut, gsl::span<const float>(params.MinPt),
                                 innerState, outerState, chi2, reason));
  track.seed = seed;
  track.track.innerState = innerState;
  track.track.outerState = outerState;
  track.track.chi2 = chi2;

  for (int layer = 0; layer < NLayers; ++layer) {
    BOOST_CHECK(track.seed.hasCluster(layer));
    BOOST_CHECK_EQUAL(track.getClusterIndex(layer), 0);
    BOOST_CHECK_EQUAL(layerGlobals[layer][0].clusterId, 1u);
  }
}
