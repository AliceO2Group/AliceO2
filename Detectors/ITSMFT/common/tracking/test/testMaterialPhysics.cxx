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

#define BOOST_TEST_MODULE ITSMFTMaterialPhysics
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "CommonConstants/MathConstants.h"
#include "ITSMFTTracking/MaterialPhysics.h"
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/TrackParametrization.h"
#include "ReconstructionDataFormats/TrackUtils.h"

namespace
{
using namespace o2::itsmft::tracking::material;
using o2::track::PID;

constexpr float AbsTol = 1.e-5f;
constexpr float RelTol = 5.e-4f;

bool closeTo(float a, float b, float absTol = AbsTol, float relTol = RelTol)
{
  const float diff = std::fabs(a - b);
  return diff <= absTol || diff <= relTol * std::fabs(b);
}

// Reference copies of the production-private Highland/straggling constants,
// used only to build the double-precision oracle below. Retained here as
// characterization/reference evidence; not production arithmetic.
constexpr double kHighlandConst2 = 0.0136 * 0.0136;
constexpr double kStragglingConst = 0.0007;
constexpr float kMinMomentumGeV = 0.01f;

// Higher-precision (double) replica of the accepted capped-substep
// algorithm. This independently re-derives, at double precision, the exact
// sequence of operations the float production kernel performs, and serves
// only as test-side characterization/reference evidence -- it is never
// linked into or used by production code.
struct Oracle {
  double momentumAfterGeV{};
  double signedEnergyChangeGeV{};
  double highlandTheta2Rad2{};
  double relativeInverseMomentumVariance{};
  uint8_t substeps{0};
  bool requestedAboveCap{false};
  bool stopped{false};
  bool nonFinite{false};
};

Oracle referenceCharged(double p0, double mass, double absCharge, double xOverX0, double arealDensity,
                        bool alongMomentum)
{
  Oracle oracle{};
  const double q2 = absCharge * absCharge;
  const double e0 = std::sqrt(p0 * p0 + mass * mass);
  const double beta2 = (p0 * p0) / (e0 * e0);

  double e = e0;
  double p = p0;

  if (arealDensity > 0.) {
    const double ekin = e0 - mass;
    const double bg0 = p0 / mass;
    const double dedx0 = o2::track::BetheBlochSolidOpt<double>(bg0) * q2;
    const double fullStepLoss = dedx0 * arealDensity;
    const double ratio = std::fabs(fullStepLoss) / ekin * o2::track::ELoss2EKinThreshInv;
    if (!std::isfinite(ratio) || ratio >= static_cast<double>(o2::track::MaxELossIter)) {
      oracle.substeps = static_cast<uint8_t>(o2::track::MaxELossIter);
      oracle.requestedAboveCap = true;
    } else {
      oracle.substeps = static_cast<uint8_t>(1 + static_cast<int>(ratio));
    }
    const double arealDensityStep = arealDensity / static_cast<double>(oracle.substeps);
    for (uint8_t i = 0; i < oracle.substeps; ++i) {
      const double bg = p / mass;
      const double dedx = o2::track::BetheBlochSolidOpt<double>(bg) * q2;
      const double dE = dedx * arealDensityStep;
      e = alongMomentum ? (e - dE) : (e + dE);
      if (!std::isfinite(e)) {
        oracle.nonFinite = true;
        break;
      }
      if (e <= mass) {
        oracle.stopped = true;
        break;
      }
      p = std::sqrt(e * e - mass * mass);
      if (!std::isfinite(p)) {
        oracle.nonFinite = true;
        break;
      }
    }
  }

  oracle.momentumAfterGeV = p;
  oracle.signedEnergyChangeGeV = e - e0;
  oracle.highlandTheta2Rad2 = (xOverX0 > 0.) ? (kHighlandConst2 / (beta2 * p0 * p0) * xOverX0 * q2) : 0.;
  oracle.relativeInverseMomentumVariance = (oracle.signedEnergyChangeGeV != 0.)
                                             ? (kStragglingConst * kStragglingConst * std::fabs(oracle.signedEnergyChangeGeV) * e0 * e0 / (p0 * p0 * p0 * p0))
                                             : 0.;
  return oracle;
}

void expectDeterministicFailure(const MaterialOperationResult& result, MaterialFailureReason reason, float momentumGeV)
{
  BOOST_CHECK(!result.ok());
  BOOST_CHECK(result.failure == reason);
  if (std::isnan(momentumGeV)) {
    BOOST_CHECK(std::isnan(result.momentumBeforeGeV));
  } else {
    BOOST_CHECK_EQUAL(result.momentumBeforeGeV, momentumGeV);
  }
  BOOST_CHECK_EQUAL(result.momentumAfterGeV, 0.f);
  BOOST_CHECK_EQUAL(result.signedEnergyChangeGeV, 0.f);
  BOOST_CHECK_EQUAL(result.highlandTheta2Rad2, 0.f);
  BOOST_CHECK_EQUAL(result.relativeInverseMomentumVariance, 0.f);
  BOOST_CHECK_EQUAL(result.energyLossSubsteps, 0);
  BOOST_CHECK(result.flags == MaterialOperationFlags::None);
  BOOST_CHECK_EQUAL(result.reserved, 0);
}
} // namespace

BOOST_AUTO_TEST_CASE(RepresentationLayout)
{
  static_assert(std::is_standard_layout_v<IntegratedMaterialBudget>);
  static_assert(std::is_trivially_copyable_v<IntegratedMaterialBudget>);
  static_assert(sizeof(IntegratedMaterialBudget) == 8);
  static_assert(alignof(IntegratedMaterialBudget) == 4);

  static_assert(std::is_standard_layout_v<MaterialOperationResult>);
  static_assert(std::is_trivially_copyable_v<MaterialOperationResult>);
  static_assert(sizeof(MaterialOperationResult) == 24);
  static_assert(alignof(MaterialOperationResult) == 4);

  static_assert(sizeof(MaterialTraversalDirection) == 1);
  static_assert(sizeof(MaterialFailureReason) == 1);
  static_assert(sizeof(MaterialOperationFlags) == 1);

  // Lock the exact numeric values already reported as part of the reviewed
  // API, even though the enums are not yet a durable serialized/device ABI.
  static_assert(static_cast<uint8_t>(MaterialTraversalDirection::AlongMomentum) == 0);
  static_assert(static_cast<uint8_t>(MaterialTraversalDirection::OppositeMomentum) == 1);

  static_assert(static_cast<uint8_t>(MaterialFailureReason::None) == 0);
  static_assert(static_cast<uint8_t>(MaterialFailureReason::SourceSurfaceKindMismatch) == 1);
  static_assert(static_cast<uint8_t>(MaterialFailureReason::NonFiniteState) == 2);
  static_assert(static_cast<uint8_t>(MaterialFailureReason::InvalidStateKinematics) == 3);
  static_assert(static_cast<uint8_t>(MaterialFailureReason::InvalidPID) == 4);
  static_assert(static_cast<uint8_t>(MaterialFailureReason::ChargedMasslessPID) == 5);
  static_assert(static_cast<uint8_t>(MaterialFailureReason::InvalidDirection) == 6);
  static_assert(static_cast<uint8_t>(MaterialFailureReason::InvalidMaterial) == 7);
  static_assert(static_cast<uint8_t>(MaterialFailureReason::StoppedInMaterial) == 8);
  static_assert(static_cast<uint8_t>(MaterialFailureReason::MomentumBelowMinimum) == 9);
  static_assert(static_cast<uint8_t>(MaterialFailureReason::ExcessiveScattering) == 10);
  static_assert(static_cast<uint8_t>(MaterialFailureReason::InvalidCovariance) == 11);
  static_assert(static_cast<uint8_t>(MaterialFailureReason::NonFiniteResult) == 12);

  static_assert(static_cast<uint8_t>(MaterialOperationFlags::None) == 0);
  static_assert(static_cast<uint8_t>(MaterialOperationFlags::SubstepCountClamped) == 1);

  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(EveryValidPidIdNeutralSucceeds)
{
  IntegratedMaterialBudget material{0.01f, 0.1f};
  for (uint8_t id = 0; id < PID::NIDsTot; ++id) {
    PID pid(static_cast<PID::ID>(id));
    auto result = calculateMaterialPhysics(1.f, pid, 0, MaterialTraversalDirection::AlongMomentum, material);
    BOOST_CHECK_MESSAGE(result.ok(), "PID id " << static_cast<int>(id) << " failed with reason " << static_cast<int>(result.failure));
    BOOST_CHECK_EQUAL(result.momentumAfterGeV, 1.f);
    BOOST_CHECK_EQUAL(result.energyLossSubsteps, 0);
  }
}

BOOST_AUTO_TEST_CASE(EveryValidMassivePidIdChargedSucceeds)
{
  IntegratedMaterialBudget material{0.01f, 0.05f};
  for (uint8_t id = 0; id < PID::NIDsTot; ++id) {
    PID pid(static_cast<PID::ID>(id));
    if (pid.getMass() == 0.f) {
      continue; // massless PIDs are covered by ChargedMasslessRejection below
    }
    auto result = calculateMaterialPhysics(2.f, pid, 1, MaterialTraversalDirection::AlongMomentum, material);
    BOOST_CHECK_MESSAGE(result.ok(), "PID id " << static_cast<int>(id) << " failed with reason " << static_cast<int>(result.failure));
  }
}

BOOST_AUTO_TEST_CASE(InvalidPidIdsRejectedBeforeMassLookup)
{
  IntegratedMaterialBudget material{0.f, 0.f};
  for (uint8_t id : {static_cast<uint8_t>(PID::NIDsTot), static_cast<uint8_t>(255)}) {
    PID pid(static_cast<PID::ID>(id));
    auto neutral = calculateMaterialPhysics(1.f, pid, 0, MaterialTraversalDirection::AlongMomentum, material);
    expectDeterministicFailure(neutral, MaterialFailureReason::InvalidPID, 1.f);
    auto charged = calculateMaterialPhysics(1.f, pid, 1, MaterialTraversalDirection::AlongMomentum, material);
    expectDeterministicFailure(charged, MaterialFailureReason::InvalidPID, 1.f);
  }
}

BOOST_AUTO_TEST_CASE(PidAndChargeAreIndependent)
{
  // PID::Electron has a nominal charge of 1 in the PID table, but absCharge
  // is supplied independently and must be the only source of q^2 scaling.
  IntegratedMaterialBudget material{0.05f, 0.f};
  auto q1 = calculateMaterialPhysics(2.f, PID::Electron, 1, MaterialTraversalDirection::AlongMomentum, material);
  auto q2result = calculateMaterialPhysics(2.f, PID::Electron, 2, MaterialTraversalDirection::AlongMomentum, material);
  BOOST_REQUIRE(q1.ok());
  BOOST_REQUIRE(q2result.ok());
  // Highland variance scales with absCharge^2, independent of PID::getCharge().
  BOOST_CHECK(closeTo(q2result.highlandTheta2Rad2, 4.f * q1.highlandTheta2Rad2));
}

BOOST_AUTO_TEST_CASE(NeutralMassiveAndMasslessAccepted)
{
  IntegratedMaterialBudget material{0.2f, 5.f};
  for (PID pid : {PID(PID::K0), PID(PID::Photon)}) {
    auto result = calculateMaterialPhysics(3.f, pid, 0, MaterialTraversalDirection::OppositeMomentum, material);
    BOOST_REQUIRE(result.ok());
    BOOST_CHECK_EQUAL(result.momentumAfterGeV, 3.f);
    BOOST_CHECK_EQUAL(result.signedEnergyChangeGeV, 0.f);
    BOOST_CHECK_EQUAL(result.highlandTheta2Rad2, 0.f);
    BOOST_CHECK_EQUAL(result.relativeInverseMomentumVariance, 0.f);
    BOOST_CHECK_EQUAL(result.energyLossSubsteps, 0);
  }
}

BOOST_AUTO_TEST_CASE(ChargedMasslessRejected)
{
  IntegratedMaterialBudget material{0.f, 0.f};
  for (uint8_t absCharge : {1, 2, 3, 255}) {
    auto result = calculateMaterialPhysics(1.f, PID::Photon, absCharge, MaterialTraversalDirection::AlongMomentum, material);
    expectDeterministicFailure(result, MaterialFailureReason::ChargedMasslessPID, 1.f);
  }
}

BOOST_AUTO_TEST_CASE(AbsChargeVariantsScaleHighlandQuadratically)
{
  IntegratedMaterialBudget material{0.03f, 0.f}; // MCS-only: isolates the charge scaling.
  auto base = calculateMaterialPhysics(1.5f, PID::Pion, 1, MaterialTraversalDirection::AlongMomentum, material);
  BOOST_REQUIRE(base.ok());
  for (uint8_t absCharge : {2, 3, 200}) {
    auto result = calculateMaterialPhysics(1.5f, PID::Pion, absCharge, MaterialTraversalDirection::AlongMomentum, material);
    BOOST_REQUIRE(result.ok());
    const float expectedRatio = static_cast<float>(absCharge) * static_cast<float>(absCharge);
    BOOST_CHECK(closeTo(result.highlandTheta2Rad2, expectedRatio * base.highlandTheta2Rad2));
  }
}

BOOST_AUTO_TEST_CASE(DirectionInvalidCastRejected)
{
  IntegratedMaterialBudget material{0.f, 0.f};
  for (uint8_t raw : {2, 255}) {
    auto direction = static_cast<MaterialTraversalDirection>(raw);
    auto result = calculateMaterialPhysics(1.f, PID::Pion, 1, direction, material);
    expectDeterministicFailure(result, MaterialFailureReason::InvalidDirection, 1.f);
  }
}

BOOST_AUTO_TEST_CASE(ValidationPrecedenceWithCombinedInvalidInputs)
{
  const auto badDirection = static_cast<MaterialTraversalDirection>(255);
  const IntegratedMaterialBudget badMaterial{0.1f, -1.f};
  const IntegratedMaterialBudget goodMaterial{0.f, 0.f};
  const float badMomentum = -1.f;
  const float goodMomentum = 1.f;
  const PID badPid(static_cast<PID::ID>(255));
  const PID goodMasslessPid = PID::Photon;

  // 1. invalid direction wins over invalid material/momentum/PID.
  auto r1 = calculateMaterialPhysics(badMomentum, badPid, 1, badDirection, badMaterial);
  BOOST_CHECK(r1.failure == MaterialFailureReason::InvalidDirection);

  // 2. invalid material wins over invalid momentum/PID.
  auto r2 = calculateMaterialPhysics(badMomentum, badPid, 1, MaterialTraversalDirection::AlongMomentum, badMaterial);
  BOOST_CHECK(r2.failure == MaterialFailureReason::InvalidMaterial);

  // 3. invalid momentum wins over invalid PID.
  auto r3 = calculateMaterialPhysics(badMomentum, badPid, 1, MaterialTraversalDirection::AlongMomentum, goodMaterial);
  BOOST_CHECK(r3.failure == MaterialFailureReason::MomentumBelowMinimum);

  // 4. invalid PID wins over charged-massless inspection: an unresolvable
  // id must surface InvalidPID, never attempting the mass lookup that
  // ChargedMasslessPID depends on.
  auto r4 = calculateMaterialPhysics(goodMomentum, badPid, 1, MaterialTraversalDirection::AlongMomentum, goodMaterial);
  BOOST_CHECK(r4.failure == MaterialFailureReason::InvalidPID);

  // Control: same absCharge/direction/material/momentum, but a valid
  // massless PID -- confirms r4 is really about id validity winning over
  // the charged-massless inspection, not some unrelated mismatch.
  auto r4Control = calculateMaterialPhysics(goodMomentum, goodMasslessPid, 1, MaterialTraversalDirection::AlongMomentum, goodMaterial);
  BOOST_CHECK(r4Control.failure == MaterialFailureReason::ChargedMasslessPID);
}

BOOST_AUTO_TEST_CASE(MaterialFieldsMustBeNonNegative)
{
  const std::vector<IntegratedMaterialBudget> invalidMaterials = {
    {-1.f, 0.1f}, {0.1f, -1.f}, {-1.f, -1.f}};
  for (auto material : invalidMaterials) {
    auto result = calculateMaterialPhysics(1.f, PID::Pion, 1, MaterialTraversalDirection::AlongMomentum, material);
    expectDeterministicFailure(result, MaterialFailureReason::InvalidMaterial, 1.f);
  }
}

BOOST_AUTO_TEST_CASE(MomentumMustBePositive)
{
  IntegratedMaterialBudget material{0.f, 0.f};
  for (float momentum : {0.f, -1.f}) {
    auto result = calculateMaterialPhysics(momentum, PID::Pion, 1, MaterialTraversalDirection::AlongMomentum, material);
    expectDeterministicFailure(result, MaterialFailureReason::MomentumBelowMinimum, momentum);
  }
}

BOOST_AUTO_TEST_CASE(ZeroMaterialIsAPassThrough)
{
  IntegratedMaterialBudget material{0.f, 0.f};
  auto result = calculateMaterialPhysics(1.f, PID::Pion, 1, MaterialTraversalDirection::AlongMomentum, material);
  BOOST_REQUIRE(result.ok());
  BOOST_CHECK_EQUAL(result.momentumAfterGeV, 1.f);
  BOOST_CHECK_EQUAL(result.signedEnergyChangeGeV, 0.f);
  BOOST_CHECK_EQUAL(result.highlandTheta2Rad2, 0.f);
  BOOST_CHECK_EQUAL(result.relativeInverseMomentumVariance, 0.f);
  BOOST_CHECK_EQUAL(result.energyLossSubsteps, 0);
  BOOST_CHECK(result.flags == MaterialOperationFlags::None);
}

BOOST_AUTO_TEST_CASE(McsOnlyMaterialMatchesAnalyticHighland)
{
  const float p0 = 2.f;
  const float mass = PID(PID::Pion).getMass();
  IntegratedMaterialBudget material{0.05f, 0.f};
  auto result = calculateMaterialPhysics(p0, PID::Pion, 1, MaterialTraversalDirection::AlongMomentum, material);
  BOOST_REQUIRE(result.ok());
  BOOST_CHECK_EQUAL(result.momentumAfterGeV, p0);
  BOOST_CHECK_EQUAL(result.signedEnergyChangeGeV, 0.f);
  BOOST_CHECK_EQUAL(result.energyLossSubsteps, 0);
  BOOST_CHECK_EQUAL(result.relativeInverseMomentumVariance, 0.f);

  const double e0 = std::sqrt(static_cast<double>(p0) * p0 + static_cast<double>(mass) * mass);
  const double beta2 = (static_cast<double>(p0) * p0) / (e0 * e0);
  const double expectedTheta2 = kHighlandConst2 / (beta2 * p0 * p0) * material.xOverX0;
  BOOST_CHECK(closeTo(result.highlandTheta2Rad2, static_cast<float>(expectedTheta2)));
}

BOOST_AUTO_TEST_CASE(EnergyLossOnlyMaterialProducesNoScattering)
{
  IntegratedMaterialBudget material{0.f, 0.02f};
  auto result = calculateMaterialPhysics(2.f, PID::Pion, 1, MaterialTraversalDirection::AlongMomentum, material);
  BOOST_REQUIRE(result.ok());
  BOOST_CHECK_EQUAL(result.highlandTheta2Rad2, 0.f);
  BOOST_CHECK_LT(result.momentumAfterGeV, 2.f);
  BOOST_CHECK_LT(result.signedEnergyChangeGeV, 0.f);
  BOOST_CHECK_GT(result.energyLossSubsteps, 0);
  BOOST_CHECK_GT(result.relativeInverseMomentumVariance, 0.f);
}

BOOST_AUTO_TEST_CASE(CombinedMaterialMatchesOracle)
{
  const float p0 = 1.2f;
  const PID pid = PID::Kaon;
  const uint8_t absCharge = 1;
  IntegratedMaterialBudget material{0.04f, 0.03f};
  auto result = calculateMaterialPhysics(p0, pid, absCharge, MaterialTraversalDirection::AlongMomentum, material);
  BOOST_REQUIRE(result.ok());

  auto oracle = referenceCharged(p0, pid.getMass(), absCharge, material.xOverX0, material.arealDensityGPerCm2, true);
  BOOST_CHECK(!oracle.stopped && !oracle.nonFinite);
  BOOST_CHECK_EQUAL(result.energyLossSubsteps, oracle.substeps);
  BOOST_CHECK(closeTo(result.momentumAfterGeV, static_cast<float>(oracle.momentumAfterGeV)));
  BOOST_CHECK(closeTo(result.signedEnergyChangeGeV, static_cast<float>(oracle.signedEnergyChangeGeV)));
  BOOST_CHECK(closeTo(result.highlandTheta2Rad2, static_cast<float>(oracle.highlandTheta2Rad2)));
  BOOST_CHECK(closeTo(result.relativeInverseMomentumVariance, static_cast<float>(oracle.relativeInverseMomentumVariance)));
}

BOOST_AUTO_TEST_CASE(LossAndGainHaveOppositeSignedEnergyChange)
{
  const float p0 = 1.5f;
  IntegratedMaterialBudget material{0.f, 0.005f}; // small enough to stay single-substep
  auto loss = calculateMaterialPhysics(p0, PID::Proton, 1, MaterialTraversalDirection::AlongMomentum, material);
  auto gain = calculateMaterialPhysics(p0, PID::Proton, 1, MaterialTraversalDirection::OppositeMomentum, material);
  BOOST_REQUIRE(loss.ok());
  BOOST_REQUIRE(gain.ok());
  BOOST_CHECK_EQUAL(loss.energyLossSubsteps, 1);
  BOOST_CHECK_EQUAL(gain.energyLossSubsteps, 1);
  BOOST_CHECK_LT(loss.signedEnergyChangeGeV, 0.f);
  BOOST_CHECK_GT(gain.signedEnergyChangeGeV, 0.f);
  BOOST_CHECK(closeTo(loss.signedEnergyChangeGeV, -gain.signedEnergyChangeGeV, AbsTol, 1.e-2f));
  BOOST_CHECK_LT(loss.momentumAfterGeV, p0);
  BOOST_CHECK_GT(gain.momentumAfterGeV, p0);
}

BOOST_AUTO_TEST_CASE(SubstepCountsAcrossRange)
{
  const float p0 = 1.f;
  const PID pid = PID::Proton;
  const double mass = pid.getMass();
  const double e0 = std::sqrt(static_cast<double>(p0) * p0 + mass * mass);
  const double ekin = e0 - mass;
  const double bg0 = p0 / mass;
  const double dedx0 = o2::track::BetheBlochSolidOpt<double>(bg0);

  auto arealDensityForRatio = [&](double ratio) {
    return ratio * ekin / (o2::track::ELoss2EKinThreshInv * dedx0);
  };

  // na = 1 + floor(ratio); choose ratio well inside each unit interval.
  const struct {
    double ratio;
    uint8_t expectedSubsteps;
    bool expectedClamped;
  } cases[] = {
    {0.3, 1, false},
    {5.5, 6, false},
    {48.9, 49, false},
    {49.5, 50, false}, // na = 50, must NOT be reported as clamped
    {60.0, 50, true},
    {1.e6, 50, true},
  };

  // OppositeMomentum (energy gain) is used deliberately: it isolates the
  // substep-count bookkeeping from the (physically legitimate) risk that a
  // large requested ratio also represents more energy loss than the
  // particle's kinetic energy can absorb, which is covered separately by
  // the StoppingIsDetected test.
  for (const auto& c : cases) {
    IntegratedMaterialBudget material{0.f, static_cast<float>(arealDensityForRatio(c.ratio))};
    auto result = calculateMaterialPhysics(p0, pid, 1, MaterialTraversalDirection::OppositeMomentum, material);
    BOOST_REQUIRE_MESSAGE(result.ok(), "unexpected failure " << static_cast<int>(result.failure) << " for ratio " << c.ratio);
    BOOST_CHECK_EQUAL(result.energyLossSubsteps, c.expectedSubsteps);
    if (c.expectedClamped) {
      BOOST_CHECK(result.flags == MaterialOperationFlags::SubstepCountClamped);
    } else {
      BOOST_CHECK(result.flags == MaterialOperationFlags::None);
    }

    auto oracle = referenceCharged(p0, mass, 1., 0., material.arealDensityGPerCm2, false);
    BOOST_REQUIRE(!oracle.stopped && !oracle.nonFinite);
    BOOST_CHECK(closeTo(result.momentumAfterGeV, static_cast<float>(oracle.momentumAfterGeV)));
  }
}

BOOST_AUTO_TEST_CASE(ClampedSubstepsStillProcessCompleteArealDensity)
{
  // Use OppositeMomentum (energy gain) so a very large ratio clamps the
  // substep count without stopping the particle, letting us verify the
  // full arealDensityGPerCm2 was processed across exactly 50 substeps.
  const float p0 = 1.f;
  const PID pid = PID::Proton;
  IntegratedMaterialBudget material{0.f, 500.f};
  auto result = calculateMaterialPhysics(p0, pid, 1, MaterialTraversalDirection::OppositeMomentum, material);
  BOOST_REQUIRE(result.ok());
  BOOST_CHECK_EQUAL(result.energyLossSubsteps, o2::track::MaxELossIter);
  BOOST_CHECK(result.flags == MaterialOperationFlags::SubstepCountClamped);

  auto oracle = referenceCharged(p0, pid.getMass(), 1., 0., material.arealDensityGPerCm2, false);
  BOOST_REQUIRE(!oracle.stopped && !oracle.nonFinite);
  BOOST_CHECK_EQUAL(oracle.substeps, o2::track::MaxELossIter);
  BOOST_CHECK(closeTo(result.signedEnergyChangeGeV, static_cast<float>(oracle.signedEnergyChangeGeV), AbsTol, 2.e-3f));
}

BOOST_AUTO_TEST_CASE(BetheBlochIsRecomputedPerSubstep)
{
  // A naive fixed-dedx-at-entry integration must differ measurably from the
  // recompute-per-substep result once the momentum changes appreciably
  // across the traversal.
  const float p0 = 0.3f;
  const PID pid = PID::Proton;
  const double mass = pid.getMass();
  IntegratedMaterialBudget material{0.f, 1.f};
  auto result = calculateMaterialPhysics(p0, pid, 1, MaterialTraversalDirection::AlongMomentum, material);
  BOOST_REQUIRE(result.ok());
  BOOST_REQUIRE_GT(result.energyLossSubsteps, 1);

  const double e0 = std::sqrt(static_cast<double>(p0) * p0 + mass * mass);
  const double bg0 = p0 / mass;
  const double dedx0 = o2::track::BetheBlochSolidOpt<double>(bg0);
  const double naiveEnergyAfter = e0 - dedx0 * material.arealDensityGPerCm2;

  auto oracle = referenceCharged(p0, mass, 1., 0., material.arealDensityGPerCm2, true);
  BOOST_REQUIRE(!oracle.stopped && !oracle.nonFinite);
  const double recomputedEnergyAfter = e0 + oracle.signedEnergyChangeGeV;

  BOOST_CHECK(closeTo(result.signedEnergyChangeGeV, static_cast<float>(oracle.signedEnergyChangeGeV)));
  BOOST_CHECK_GT(std::fabs(recomputedEnergyAfter - naiveEnergyAfter), 1.e-4);
}

BOOST_AUTO_TEST_CASE(StoppingIsDetected)
{
  IntegratedMaterialBudget material{0.f, 50.f}; // grossly exceeds a 0.5 GeV/c proton's kinetic energy
  auto result = calculateMaterialPhysics(0.5f, PID::Proton, 1, MaterialTraversalDirection::AlongMomentum, material);
  expectDeterministicFailure(result, MaterialFailureReason::StoppedInMaterial, 0.5f);
}

BOOST_AUTO_TEST_CASE(FinalMomentumBoundary)
{
  IntegratedMaterialBudget material{0.f, 0.f}; // zero material: momentumAfter == momentumBefore exactly
  auto atThreshold = calculateMaterialPhysics(kMinMomentumGeV, PID::Pion, 1, MaterialTraversalDirection::AlongMomentum, material);
  BOOST_REQUIRE(atThreshold.ok());
  BOOST_CHECK_EQUAL(atThreshold.momentumAfterGeV, kMinMomentumGeV);

  auto belowThreshold = calculateMaterialPhysics(std::nextafter(kMinMomentumGeV, 0.f), PID::Pion, 1,
                                                 MaterialTraversalDirection::AlongMomentum, material);
  expectDeterministicFailure(belowThreshold, MaterialFailureReason::MomentumBelowMinimum, std::nextafter(kMinMomentumGeV, 0.f));
}

BOOST_AUTO_TEST_CASE(ExcessiveScatteringIsRejected)
{
  IntegratedMaterialBudget material{500.f, 0.f}; // absurdly thick, drives theta^2 past pi^2
  auto result = calculateMaterialPhysics(0.1f, PID::Pion, 1, MaterialTraversalDirection::AlongMomentum, material);
  expectDeterministicFailure(result, MaterialFailureReason::ExcessiveScattering, 0.1f);
}

BOOST_AUTO_TEST_CASE(HugeFiniteArealDensityDeterministicallyStops)
{
  // 1e30 g/cm^2 is many orders of magnitude beyond what a 1 GeV/c proton's
  // kinetic energy can absorb: even after the substep count clamps to 50
  // (since the requested count vastly exceeds it), the very first substep's
  // energy loss drives the particle's energy far below its rest mass. This
  // must terminate deterministically without any float-to-int UB in the
  // substep-count calculation.
  const float p0 = 1.f;
  IntegratedMaterialBudget material{0.f, 1.e30f};
  auto result = calculateMaterialPhysics(p0, PID::Proton, 1, MaterialTraversalDirection::AlongMomentum, material);
  expectDeterministicFailure(result, MaterialFailureReason::StoppedInMaterial, p0);

  auto repeat = calculateMaterialPhysics(p0, PID::Proton, 1, MaterialTraversalDirection::AlongMomentum, material);
  BOOST_CHECK_EQUAL(std::memcmp(&result, &repeat, sizeof(MaterialOperationResult)), 0);
}

BOOST_AUTO_TEST_CASE(DirectBetheBlochReferenceValue)
{
  const float p0 = 1.f;
  const PID pid = PID::Proton;
  const double mass = pid.getMass();
  IntegratedMaterialBudget material{0.f, 0.001f}; // small enough to guarantee a single substep
  auto result = calculateMaterialPhysics(p0, pid, 1, MaterialTraversalDirection::AlongMomentum, material);
  BOOST_REQUIRE(result.ok());
  BOOST_REQUIRE_EQUAL(result.energyLossSubsteps, 1);

  const double e0 = std::sqrt(static_cast<double>(p0) * p0 + mass * mass);
  const double bg0 = p0 / mass;
  const double dedx = o2::track::BetheBlochSolidOpt<double>(bg0);
  const double expectedEnergyAfter = e0 - dedx * material.arealDensityGPerCm2;
  const double expectedSignedChange = expectedEnergyAfter - e0;
  BOOST_CHECK(closeTo(result.signedEnergyChangeGeV, static_cast<float>(expectedSignedChange)));
}

BOOST_AUTO_TEST_CASE(ChargeSquaredScalesSingleSubstepEnergyLoss)
{
  // Material thin enough that absCharge up to 3 (q^2 up to 9) still resolves
  // to a single substep for every case below. PID::Electron's nominal
  // PID::getCharge() is fixed at 1 regardless of absCharge, so any observed
  // scaling with absCharge (not with PID::getCharge()) demonstrates that
  // getCharge() is never consulted.
  const float p0 = 1.f;
  const PID pid = PID::Electron;
  const double mass = pid.getMass();
  const IntegratedMaterialBudget material{0.f, 0.0001f};

  const double e0 = std::sqrt(static_cast<double>(p0) * p0 + mass * mass);
  const double bg0 = p0 / mass;
  const double dedxUnit = o2::track::BetheBlochSolidOpt<double>(bg0); // reference dE/dx at q^2 = 1

  float baseSignedChange = 0.f;
  float baseVariance = 0.f;
  for (uint8_t absCharge : {1, 2, 3}) {
    auto result = calculateMaterialPhysics(p0, pid, absCharge, MaterialTraversalDirection::AlongMomentum, material);
    BOOST_REQUIRE(result.ok());
    BOOST_REQUIRE_EQUAL(result.energyLossSubsteps, 1);

    const double q2 = static_cast<double>(absCharge) * absCharge;
    const double expectedDE = dedxUnit * q2 * material.arealDensityGPerCm2;
    const double expectedEnergyAfter = e0 - expectedDE;
    const double expectedSignedChange = expectedEnergyAfter - e0;
    const double expectedMomentumAfter = std::sqrt(expectedEnergyAfter * expectedEnergyAfter - mass * mass);
    const double expectedVariance = kStragglingConst * kStragglingConst * std::fabs(expectedSignedChange) * e0 * e0 /
                                    (static_cast<double>(p0) * p0 * p0 * p0);

    BOOST_CHECK(closeTo(result.signedEnergyChangeGeV, static_cast<float>(expectedSignedChange)));
    BOOST_CHECK(closeTo(result.momentumAfterGeV, static_cast<float>(expectedMomentumAfter)));
    BOOST_CHECK(closeTo(result.relativeInverseMomentumVariance, static_cast<float>(expectedVariance)));

    if (absCharge == 1) {
      baseSignedChange = result.signedEnergyChangeGeV;
      baseVariance = result.relativeInverseMomentumVariance;
    } else {
      const float q2f = static_cast<float>(absCharge) * static_cast<float>(absCharge);
      BOOST_CHECK(closeTo(result.signedEnergyChangeGeV, q2f * baseSignedChange));
      BOOST_CHECK(closeTo(result.relativeInverseMomentumVariance, q2f * baseVariance));
    }
  }
}

BOOST_AUTO_TEST_CASE(RepeatedCallsAreByteIdentical)
{
  IntegratedMaterialBudget material{0.03f, 0.02f};
  auto a = calculateMaterialPhysics(1.3f, PID::Kaon, 1, MaterialTraversalDirection::AlongMomentum, material);
  auto b = calculateMaterialPhysics(1.3f, PID::Kaon, 1, MaterialTraversalDirection::AlongMomentum, material);
  BOOST_CHECK_EQUAL(std::memcmp(&a, &b, sizeof(MaterialOperationResult)), 0);
}

BOOST_AUTO_TEST_CASE(ReservedIsAlwaysZero)
{
  IntegratedMaterialBudget material{0.02f, 0.01f};
  auto success = calculateMaterialPhysics(1.f, PID::Pion, 1, MaterialTraversalDirection::AlongMomentum, material);
  BOOST_CHECK_EQUAL(success.reserved, 0);
  auto failure = calculateMaterialPhysics(-1.f, PID::Pion, 1, MaterialTraversalDirection::AlongMomentum, material);
  BOOST_CHECK_EQUAL(failure.reserved, 0);
}
