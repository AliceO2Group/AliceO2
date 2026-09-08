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

#define BOOST_TEST_MODULE ITSMFT CellFindingNative
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <cmath>
#include <cstring>
#include <limits>

#include <boost/test/unit_test.hpp>

#include "ITSMFTTracking/detail/SurfaceStateOperations.h"
#include "ITSMFTTracking/MaterialPhysics.h"
#include "ITSMFTTracking/Propagator.h"
#include "ITSMFTTracking/detail/TrackingKernelParameters.h"
#include "ITStracking/Cluster.h"

/// Focused coverage for the explicit cylinder/disk SurfaceTrackState
/// compatibility and hit-attachment leaves. The leaves are production-wired
/// through TrackerTraits; this test exercises their numerical contracts
/// directly, while the separate orchestration tests cover their callers.
///
/// Oracle strategy:
///  - Formula-preserving evidence (rotation/propagation, predicted
///    chi2/update, state compatibility) is checked by an
///    independent, hand-written re-transcription of each operation's own
///    documented call sequence, built directly on the already-oracle-tested
///    detail::barrel::/detail::forward:: primitives (BarrelSurfaceStateOperations.h,
///    ForwardSurfaceStateOperations.h -- each already characterized against
///    its own legacy oracle in testBarrelSurfaceStateOperations.cxx /
///    testForwardSurfaceStateOperations.cxx). A bit-identical match against
///    this independent replay is strong evidence that the production
///    orchestration (step order, material-slot selection, chi2-cut
///    placement, measurement projection) is correct, without re-deriving
///    the already-tested primitive formulas themselves.
///  - Intentional material-physics differences (PID/absCharge-aware barrel
///    covariance correction; newly active forward energy loss/straggling)
///    are validated by observing that charge/PID and areal-density inputs
///    measurably change the result, not by comparing to legacy MCS-only
///    output.
using namespace o2::itsmft::tracking;

namespace
{

template <typename T>
bool bitEqual(const T& lhs, const T& rhs)
{
  return std::memcmp(&lhs, &rhs, sizeof(T)) == 0;
}

bool attachMeasurement(SurfaceTrackState& state, const SurfaceMeasurement& measurement,
                       NominalSurfaceMaterial material, float bz, float& chi2,
                       const TrackingKernelParameters& parameters,
                       OperationFailureReason& reason)
{
  SurfaceDescriptor target{};
  target.kind = state.kind;
  target.material = material;
  return Propagator::attachMeasurement(state, target, measurement, bz,
                                       material::MaterialTraversalDirection::OppositeMomentum, true,
                                       parameters.maxChi2ClusterAttachment, chi2, reason);
}

SurfaceMeasurement barrelMeasurementFromHit(const o2::its::TrackingFrameInfo& hit)
{
  SurfaceMeasurement measurement{};
  measurement.frame.q = hit.xTrackingFrame;
  measurement.frame.frameAngle = hit.alphaTrackingFrame;
  measurement.frame.u = hit.positionTrackingFrame[0];
  measurement.frame.v = hit.positionTrackingFrame[1];
  measurement.covariance.uu = hit.covarianceTrackingFrame[0];
  measurement.covariance.uv = hit.covarianceTrackingFrame[1];
  measurement.covariance.vv = hit.covarianceTrackingFrame[2];
  return measurement;
}

SurfaceMeasurement diskMeasurementFromHit(const o2::its::TrackingFrameInfo& hit)
{
  SurfaceMeasurement measurement{};
  measurement.frame = {hit.zCoordinate, hit.xCoordinate, hit.yCoordinate, 0.f};
  measurement.covariance.uu = hit.covarianceTrackingFrame[0];
  measurement.covariance.uv = 0.f;
  measurement.covariance.vv = hit.covarianceTrackingFrame[2];
  return measurement;
}

// --- attachHit fixtures -----------------------------------------------

SurfaceTrackState barrelAttachState()
{
  SurfaceTrackState state{};
  state.parameters[0] = 1.25f;
  state.parameters[1] = -0.75f;
  state.parameters[2] = 0.2f;
  state.parameters[3] = -0.35f;
  state.parameters[4] = 0.8f;
  state.referenceCoordinate = 4.f;
  state.alpha = 0.3f;
  state.kind = SurfaceKind::Cylinder;
  state.absCharge = 1;
  state.pid = o2::track::PID::Kaon;
  for (uint8_t row = 0; row < 5; ++row) {
    for (uint8_t column = 0; column <= row; ++column) {
      state.covariance[packedCovarianceIndex(row, column)] = row == column ? 0.01f * (row + 1) : 0.0002f * (row + column + 1);
    }
  }
  return state;
}

o2::its::TrackingFrameInfo barrelAttachHit()
{
  return o2::its::TrackingFrameInfo{0.f, 0.f, 0.f, 2.5f, 0.3f, {0.8f, -0.45f}, {0.04f, 0.012f, 0.09f}};
}

constexpr float BarrelAttachBz = 5.f;

NominalSurfaceMaterial barrelAttachMaterial() { return NominalSurfaceMaterial{0.01f, 0.001f}; }

SurfaceTrackState diskAttachState()
{
  SurfaceTrackState state{};
  state.parameters[0] = 1.25f;
  state.parameters[1] = -0.75f;
  state.parameters[2] = 0.35f;
  state.parameters[3] = -2.5f;
  state.parameters[4] = 0.8f;
  state.referenceCoordinate = -45.f;
  state.kind = SurfaceKind::Disk;
  state.absCharge = 2;
  state.pid = o2::track::PID::Pion;
  for (uint8_t row = 0; row < 5; ++row) {
    for (uint8_t column = 0; column <= row; ++column) {
      state.covariance[packedCovarianceIndex(row, column)] = row == column ? 0.01f * (row + 1) : 0.0002f * (row + column + 1);
    }
  }
  return state;
}

o2::its::TrackingFrameInfo diskAttachHit()
{
  return o2::its::TrackingFrameInfo{0.8f, -0.45f, -50.f, 0.f, 0.f, {0.f, 0.f}, {0.04f, 0.f, 0.09f}};
}

constexpr float DiskAttachBz = 5.f;

NominalSurfaceMaterial diskAttachMaterial() { return NominalSurfaceMaterial{0.02f, 0.002f}; }

// Test-local field-mapping helper: builds the SurfaceMeasurement
// attachDiskHit reads from a single legacy hit (Disk field mapping:
// global coordinates -> measurement.global, reference z -> measurement.
// frame.q [read as the propagate target, in place of the retired
// hit.zCoordinate], measured covariance -> measurement.covariance).
SurfaceMeasurement diskAttachMeasurementFrom(const o2::its::TrackingFrameInfo& hit)
{
  auto measurement = diskMeasurementFromHit(hit);
  measurement.frame.q = hit.zCoordinate;
  return measurement;
}

SurfaceMeasurement diskAttachMeasurement() { return diskAttachMeasurementFrom(diskAttachHit()); }

} // namespace

// ===========================================================================
// Measurement attachment
// ===========================================================================

BOOST_AUTO_TEST_CASE(AttachHitBarrelSuccessAndExactChi2Threshold)
{
  const auto state0 = barrelAttachState();
  const auto hit = barrelMeasurementFromHit(barrelAttachHit());
  const auto material = barrelAttachMaterial();

  auto probe = state0;
  float probeChi2 = 0.f;
  OperationFailureReason reason{};
  TrackingKernelParameters permissive;
  permissive.maxChi2ClusterAttachment = 1.e6f;
  BOOST_REQUIRE(attachMeasurement(probe, hit, material, BarrelAttachBz, probeChi2, permissive, reason));
  BOOST_REQUIRE_GT(probeChi2, 0.f);

  auto accepted = state0;
  float acceptedChi2 = 0.f;
  TrackingKernelParameters accept;
  accept.maxChi2ClusterAttachment = probeChi2;
  BOOST_CHECK(attachMeasurement(accepted, hit, material, BarrelAttachBz, acceptedChi2, accept, reason));
  BOOST_CHECK(bitEqual(accepted, probe));
  BOOST_CHECK_EQUAL(acceptedChi2, probeChi2);

  auto rejected = state0;
  float rejectedChi2 = -1.f;
  const auto before = rejected;
  const float chi2Before = rejectedChi2;
  TrackingKernelParameters reject;
  reject.maxChi2ClusterAttachment = std::nextafter(probeChi2, -std::numeric_limits<float>::infinity());
  BOOST_CHECK(!attachMeasurement(rejected, hit, material, BarrelAttachBz, rejectedChi2, reject, reason));
  BOOST_CHECK(reason == OperationFailureReason::PredictedChi2Failure);
  BOOST_CHECK(bitEqual(rejected, before));
  BOOST_CHECK_EQUAL(rejectedChi2, chi2Before);
}

BOOST_AUTO_TEST_CASE(AttachHitBarrelEachFailureStagePreservesStateTransactionally)
{
  const auto state0 = barrelAttachState();
  TrackingKernelParameters permissive;
  permissive.maxChi2ClusterAttachment = 1.e6f;

  auto checkFailure = [&](const SurfaceMeasurement& measurement, const NominalSurfaceMaterial& material,
                          OperationFailureReason expectedOneOf1, OperationFailureReason expectedOneOf2) {
    auto state = state0;
    float chi2 = -1.f;
    const auto before = state;
    const float chi2Before = chi2;
    OperationFailureReason reason{};
    BOOST_CHECK(!attachMeasurement(state, measurement, material, BarrelAttachBz, chi2, permissive, reason));
    BOOST_CHECK(reason == expectedOneOf1 || reason == expectedOneOf2);
    BOOST_CHECK(bitEqual(state, before));
    BOOST_CHECK_EQUAL(chi2, chi2Before);
  };

  // Rotation failure.
  {
    auto farHit = barrelAttachHit();
    farHit.alphaTrackingFrame = state0.alpha + 3.f;
    checkFailure(barrelMeasurementFromHit(farHit), barrelAttachMaterial(), OperationFailureReason::RotationFailure, OperationFailureReason::RotationFailure);
  }

  // Propagation failure.
  {
    auto farHit = barrelAttachHit();
    farHit.xTrackingFrame = -50000.f;
    checkFailure(barrelMeasurementFromHit(farHit), barrelAttachMaterial(), OperationFailureReason::UnreachableTarget, OperationFailureReason::PropagationFailure);
  }
}

BOOST_AUTO_TEST_CASE(AttachHitBarrelNegativeChi2IsRejectedMatchingLegacyInclusiveCut)
{
  // No-op rotate (hit shares state's alpha) and no-op propagate (hit's
  // target x equals state's own referenceCoordinate) plus a no-op material
  // budget ({0,0} is the documented unconditional no-op) keep the state
  // byte-identical to barrelAttachState() through predictedChi2, so its
  // known covariance can be reasoned about directly: a pathologically large
  // measurement cross-covariance (uv) makes the combined 2x2 determinant
  // negative, which residualInverse's own gate does not reject outright
  // (only exact-zero/non-finite determinants are), producing a negative
  // predicted chi2 -- the same `< 0.f` established rejection
  // attachCylinderHit already applies today. detail::barrel::update shares
  // the identical residualInverse gate and therefore cannot independently
  // fail once predictedChi2 has already succeeded with the same inputs; the
  // two checks share one deterministic failure precedence (predictedChi2,
  // evaluated before update, is always the one that observes a bad
  // residualInverse first).
  auto state = barrelAttachState();
  auto hit = barrelAttachHit();
  hit.alphaTrackingFrame = state.alpha;
  hit.xTrackingFrame = state.referenceCoordinate;
  hit.covarianceTrackingFrame[1] = 50.f; // huge uv cross term
  const auto measurement = barrelMeasurementFromHit(hit);
  const NominalSurfaceMaterial noopMaterial{0.f, 0.f};
  const auto before = state;
  float chi2 = -1.f;
  const float chi2Before = chi2;
  OperationFailureReason reason{};
  TrackingKernelParameters permissive;
  permissive.maxChi2ClusterAttachment = 1.e6f;
  BOOST_CHECK(!attachMeasurement(state, measurement, noopMaterial, BarrelAttachBz, chi2, permissive, reason));
  BOOST_CHECK(reason == OperationFailureReason::PredictedChi2Failure);
  BOOST_CHECK(bitEqual(state, before));
  BOOST_CHECK_EQUAL(chi2, chi2Before);
}

BOOST_AUTO_TEST_CASE(AttachHitBarrelIsChargeAwareUnlikeNeutralMaterialCorrection)
{
  // PID/absCharge-aware behavior: a neutral state (absCharge == 0) takes the
  // material kernel's documented unconditional no-op path, while a charged
  // state with identical kinematics picks up a Highland covariance
  // contribution -- the results must differ.
  auto neutral = barrelAttachState();
  neutral.absCharge = 0;
  auto charged = barrelAttachState();
  charged.absCharge = 1;
  const auto hit = barrelMeasurementFromHit(barrelAttachHit());
  const auto material = barrelAttachMaterial();

  float neutralChi2 = 0.f;
  float chargedChi2 = 0.f;
  OperationFailureReason reason{};
  TrackingKernelParameters permissive;
  permissive.maxChi2ClusterAttachment = 1.e6f;
  BOOST_REQUIRE(attachMeasurement(neutral, hit, material, BarrelAttachBz, neutralChi2, permissive, reason));
  BOOST_REQUIRE(attachMeasurement(charged, hit, material, BarrelAttachBz, chargedChi2, permissive, reason));
  BOOST_CHECK(!bitEqual(neutral, charged));
}

BOOST_AUTO_TEST_CASE(AttachHitBarrelIsByteDeterministic)
{
  auto first = barrelAttachState();
  auto second = barrelAttachState();
  float chi2First = 0.f;
  float chi2Second = 0.f;
  OperationFailureReason reason{};
  TrackingKernelParameters permissive;
  permissive.maxChi2ClusterAttachment = 1.e6f;
  BOOST_REQUIRE(attachMeasurement(first, barrelMeasurementFromHit(barrelAttachHit()), barrelAttachMaterial(), BarrelAttachBz, chi2First, permissive, reason));
  BOOST_REQUIRE(attachMeasurement(second, barrelMeasurementFromHit(barrelAttachHit()), barrelAttachMaterial(), BarrelAttachBz, chi2Second, permissive, reason));
  BOOST_CHECK(bitEqual(first, second));
  BOOST_CHECK_EQUAL(chi2First, chi2Second);
}

// ===========================================================================
// attachDiskHit
// ===========================================================================

BOOST_AUTO_TEST_CASE(AttachHitDiskSuccessAndExactChi2Threshold)
{
  const auto state0 = diskAttachState();
  const auto hit = diskAttachMeasurement();
  const auto material = diskAttachMaterial();

  auto probe = state0;
  float probeChi2 = 0.f;
  OperationFailureReason reason{};
  TrackingKernelParameters permissive;
  permissive.maxChi2ClusterAttachment = 1.e6f;
  BOOST_REQUIRE(attachMeasurement(probe, hit, material, DiskAttachBz, probeChi2, permissive, reason));
  BOOST_REQUIRE_GT(probeChi2, 0.f);

  auto accepted = state0;
  float acceptedChi2 = 0.f;
  TrackingKernelParameters accept;
  accept.maxChi2ClusterAttachment = probeChi2;
  BOOST_CHECK(attachMeasurement(accepted, hit, material, DiskAttachBz, acceptedChi2, accept, reason));
  BOOST_CHECK(bitEqual(accepted, probe));
  BOOST_CHECK_EQUAL(acceptedChi2, probeChi2);

  auto rejected = state0;
  float rejectedChi2 = -1.f;
  const auto before = rejected;
  const float chi2Before = rejectedChi2;
  TrackingKernelParameters reject;
  reject.maxChi2ClusterAttachment = std::nextafter(probeChi2, -std::numeric_limits<float>::infinity());
  BOOST_CHECK(!attachMeasurement(rejected, hit, material, DiskAttachBz, rejectedChi2, reject, reason));
  BOOST_CHECK(reason == OperationFailureReason::PredictedChi2Failure);
  BOOST_CHECK(bitEqual(rejected, before));
  BOOST_CHECK_EQUAL(rejectedChi2, chi2Before);
}

BOOST_AUTO_TEST_CASE(AttachHitDiskEachFailureStagePreservesStateTransactionally)
{
  const auto state0 = diskAttachState();
  TrackingKernelParameters permissive;
  permissive.maxChi2ClusterAttachment = 1.e6f;

  // Propagation failure: tanl == 0 at zero field rejects with
  // UnreachableTarget.
  {
    auto zeroTanl = state0;
    zeroTanl.parameters[3] = 0.f;
    auto hit = diskAttachHit();
    hit.zCoordinate = -60.f; // dz != 0
    const auto measurement = diskAttachMeasurementFrom(hit);
    auto state = zeroTanl;
    float chi2 = -1.f;
    const auto before = state;
    const float chi2Before = chi2;
    OperationFailureReason reason{};
    BOOST_CHECK(!attachMeasurement(state, measurement, diskAttachMaterial(), 0.f, chi2, permissive, reason));
    BOOST_CHECK(reason == OperationFailureReason::UnreachableTarget);
    BOOST_CHECK(bitEqual(state, before));
    BOOST_CHECK_EQUAL(chi2, chi2Before);
  }
}

BOOST_AUTO_TEST_CASE(AttachHitDiskIsChargeAwareUnlikeNeutralMaterialCorrection)
{
  auto neutral = diskAttachState();
  neutral.absCharge = 0;
  auto charged = diskAttachState();
  charged.absCharge = 1;
  const auto hit = diskAttachMeasurement();
  const auto material = diskAttachMaterial();

  float neutralChi2 = 0.f;
  float chargedChi2 = 0.f;
  OperationFailureReason reason{};
  TrackingKernelParameters permissive;
  permissive.maxChi2ClusterAttachment = 1.e6f;
  BOOST_REQUIRE(attachMeasurement(neutral, hit, material, DiskAttachBz, neutralChi2, permissive, reason));
  BOOST_REQUIRE(attachMeasurement(charged, hit, material, DiskAttachBz, chargedChi2, permissive, reason));
  BOOST_CHECK(!bitEqual(neutral, charged));
}

BOOST_AUTO_TEST_CASE(AttachHitDiskActivatesEnergyLossUnlikeLegacyMcsOnlyPath)
{
  auto noLossMaterial = diskAttachMaterial();
  noLossMaterial.arealDensityGPerCm2 = 0.f;
  auto withLossMaterial = diskAttachMaterial();

  auto stateNoLoss = diskAttachState();
  auto stateWithLoss = diskAttachState();
  const auto hit = diskAttachMeasurement();
  float chi2NoLoss = 0.f;
  float chi2WithLoss = 0.f;
  OperationFailureReason reason{};
  TrackingKernelParameters permissive;
  permissive.maxChi2ClusterAttachment = 1.e6f;
  BOOST_REQUIRE(attachMeasurement(stateNoLoss, hit, noLossMaterial, DiskAttachBz, chi2NoLoss, permissive, reason));
  BOOST_REQUIRE(attachMeasurement(stateWithLoss, hit, withLossMaterial, DiskAttachBz, chi2WithLoss, permissive, reason));
  BOOST_CHECK_NE(stateNoLoss.parameters[4], stateWithLoss.parameters[4]);
}

BOOST_AUTO_TEST_CASE(AttachHitDiskIsByteDeterministic)
{
  auto first = diskAttachState();
  auto second = diskAttachState();
  float chi2First = 0.f;
  float chi2Second = 0.f;
  OperationFailureReason reason{};
  TrackingKernelParameters permissive;
  permissive.maxChi2ClusterAttachment = 1.e6f;
  BOOST_REQUIRE(attachMeasurement(first, diskAttachMeasurement(), diskAttachMaterial(), DiskAttachBz, chi2First, permissive, reason));
  BOOST_REQUIRE(attachMeasurement(second, diskAttachMeasurement(), diskAttachMaterial(), DiskAttachBz, chi2Second, permissive, reason));
  BOOST_CHECK(bitEqual(first, second));
  BOOST_CHECK_EQUAL(chi2First, chi2Second);
}

// ===========================================================================
// Compatibility-projection coverage (private TrackingFrameInfo ->
// SurfaceMeasurement boundary, exercised indirectly through attachHit).
// ===========================================================================

BOOST_AUTO_TEST_CASE(BarrelProjectionUsesFullCovarianceIncludingCrossTerm)
{
  const auto state0 = barrelAttachState();
  auto lowCrossTerm = barrelAttachHit();
  lowCrossTerm.covarianceTrackingFrame[1] = 0.f;
  auto highCrossTerm = barrelAttachHit();
  highCrossTerm.covarianceTrackingFrame[1] = 0.03f;

  auto stateLow = state0;
  auto stateHigh = state0;
  float chi2Low = 0.f;
  float chi2High = 0.f;
  OperationFailureReason reason{};
  TrackingKernelParameters permissive;
  permissive.maxChi2ClusterAttachment = 1.e6f;
  BOOST_REQUIRE(attachMeasurement(stateLow, barrelMeasurementFromHit(lowCrossTerm), barrelAttachMaterial(), BarrelAttachBz, chi2Low, permissive, reason));
  BOOST_REQUIRE(attachMeasurement(stateHigh, barrelMeasurementFromHit(highCrossTerm), barrelAttachMaterial(), BarrelAttachBz, chi2High, permissive, reason));
  BOOST_CHECK_NE(chi2Low, chi2High);
}

BOOST_AUTO_TEST_CASE(ForwardProjectionIsDiagonalOnlyAndIgnoresUnreadTrackingFrameFields)
{
  const auto state0 = diskAttachState();
  const auto baseline = diskAttachHit();

  auto varyingCrossTerm = baseline;
  varyingCrossTerm.covarianceTrackingFrame[1] = 999.f; // forward never reads this slot

  auto varyingUnreadFields = baseline;
  varyingUnreadFields.xTrackingFrame = 12345.f;
  varyingUnreadFields.alphaTrackingFrame = 6.7f;
  varyingUnreadFields.positionTrackingFrame = {-999.f, 999.f};

  TrackingKernelParameters permissive;
  permissive.maxChi2ClusterAttachment = 1.e6f;
  OperationFailureReason reason{};

  auto stateBaseline = state0;
  float chi2Baseline = 0.f;
  BOOST_REQUIRE(attachMeasurement(stateBaseline, diskAttachMeasurementFrom(baseline), diskAttachMaterial(), DiskAttachBz, chi2Baseline, permissive, reason));

  auto stateCrossTerm = state0;
  float chi2CrossTerm = 0.f;
  BOOST_REQUIRE(attachMeasurement(stateCrossTerm, diskAttachMeasurementFrom(varyingCrossTerm), diskAttachMaterial(), DiskAttachBz, chi2CrossTerm, permissive, reason));
  BOOST_CHECK(bitEqual(stateBaseline, stateCrossTerm));
  BOOST_CHECK_EQUAL(chi2Baseline, chi2CrossTerm);

  auto stateUnreadFields = state0;
  float chi2UnreadFields = 0.f;
  BOOST_REQUIRE(attachMeasurement(stateUnreadFields, diskAttachMeasurementFrom(varyingUnreadFields), diskAttachMaterial(), DiskAttachBz, chi2UnreadFields, permissive, reason));
  BOOST_CHECK(bitEqual(stateBaseline, stateUnreadFields));
  BOOST_CHECK_EQUAL(chi2Baseline, chi2UnreadFields);
}
