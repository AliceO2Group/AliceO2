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
#include <bit>
#include <limits>
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

float energyChange(float before, float after, PID pid)
{
  const double mass = pid.getMass();
  return std::sqrt(static_cast<double>(after) * after + mass * mass) -
         std::sqrt(static_cast<double>(before) * before + mass * mass);
}

} // namespace

BOOST_AUTO_TEST_CASE(EveryValidPidIdNeutralSucceeds)
{
  IntegratedMaterialBudget material{0.01f, 0.1f};
  for (uint8_t id = 0; id < PID::NIDsTot; ++id) {
    PID pid(static_cast<PID::ID>(id));

    float resultMomentum = 0.f;
    float resultTheta2 = 0.f;
    float resultVariance = 0.f;
    const bool result = calculateMaterialPhysics(1.f, pid, 0, MaterialTraversalDirection::AlongMomentum, material, resultMomentum, resultTheta2, resultVariance);
    BOOST_CHECK_MESSAGE(result, "PID id " << static_cast<int>(id) << " failed");
    BOOST_CHECK_EQUAL(resultMomentum, 1.f);
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

    float resultMomentum = 0.f;
    float resultTheta2 = 0.f;
    float resultVariance = 0.f;
    const bool result = calculateMaterialPhysics(2.f, pid, 1, MaterialTraversalDirection::AlongMomentum, material, resultMomentum, resultTheta2, resultVariance);
    BOOST_CHECK_MESSAGE(result, "PID id " << static_cast<int>(id) << " failed");
  }
}

BOOST_AUTO_TEST_CASE(InvalidPidIdsRejectedBeforeMassLookup)
{
  IntegratedMaterialBudget material{0.f, 0.f};
  for (uint8_t id : {static_cast<uint8_t>(PID::NIDsTot), static_cast<uint8_t>(255)}) {
    PID pid(static_cast<PID::ID>(id));

    float neutralMomentum = 0.f;
    float neutralTheta2 = 0.f;
    float neutralVariance = 0.f;
    const bool neutral = calculateMaterialPhysics(1.f, pid, 0, MaterialTraversalDirection::AlongMomentum, material, neutralMomentum, neutralTheta2, neutralVariance);
    BOOST_CHECK(!neutral);
    BOOST_CHECK_EQUAL(neutralMomentum, 0.f);
    BOOST_CHECK_EQUAL(neutralTheta2, 0.f);
    BOOST_CHECK_EQUAL(neutralVariance, 0.f);

    float chargedMomentum = 0.f;
    float chargedTheta2 = 0.f;
    float chargedVariance = 0.f;
    const bool charged = calculateMaterialPhysics(1.f, pid, 1, MaterialTraversalDirection::AlongMomentum, material, chargedMomentum, chargedTheta2, chargedVariance);
    BOOST_CHECK(!charged);
    BOOST_CHECK_EQUAL(chargedMomentum, 0.f);
    BOOST_CHECK_EQUAL(chargedTheta2, 0.f);
    BOOST_CHECK_EQUAL(chargedVariance, 0.f);
  }
}

BOOST_AUTO_TEST_CASE(PidAndChargeAreIndependent)
{
  // PID::Electron has a nominal charge of 1 in the PID table, but absCharge
  // is supplied independently and must be the only source of q^2 scaling.
  IntegratedMaterialBudget material{0.05f, 0.f};

  float q1Momentum = 0.f;
  float q1Theta2 = 0.f;
  float q1Variance = 0.f;
  const bool q1 = calculateMaterialPhysics(2.f, PID::Electron, 1, MaterialTraversalDirection::AlongMomentum, material, q1Momentum, q1Theta2, q1Variance);

  float q2resultMomentum = 0.f;
  float q2resultTheta2 = 0.f;
  float q2resultVariance = 0.f;
  const bool q2result = calculateMaterialPhysics(2.f, PID::Electron, 2, MaterialTraversalDirection::AlongMomentum, material, q2resultMomentum, q2resultTheta2, q2resultVariance);
  BOOST_REQUIRE(q1);
  BOOST_REQUIRE(q2result);
  // Highland variance scales with absCharge^2, independent of PID::getCharge().
  BOOST_CHECK(closeTo(q2resultTheta2, 4.f * q1Theta2));
}

BOOST_AUTO_TEST_CASE(NeutralMassiveAndMasslessAccepted)
{
  IntegratedMaterialBudget material{0.2f, 5.f};
  for (PID pid : {PID(PID::K0), PID(PID::Photon)}) {

    float resultMomentum = 0.f;
    float resultTheta2 = 0.f;
    float resultVariance = 0.f;
    const bool result = calculateMaterialPhysics(3.f, pid, 0, MaterialTraversalDirection::OppositeMomentum, material, resultMomentum, resultTheta2, resultVariance);
    BOOST_REQUIRE(result);
    BOOST_CHECK_EQUAL(resultMomentum, 3.f);
    BOOST_CHECK_EQUAL(energyChange(3.f, resultMomentum, pid), 0.f);
    BOOST_CHECK_EQUAL(resultTheta2, 0.f);
    BOOST_CHECK_EQUAL(resultVariance, 0.f);
  }
}

BOOST_AUTO_TEST_CASE(ChargedMasslessRejected)
{
  IntegratedMaterialBudget material{0.f, 0.f};
  for (uint8_t absCharge : {1, 2, 3, 255}) {

    float resultMomentum = 0.f;
    float resultTheta2 = 0.f;
    float resultVariance = 0.f;
    const bool result = calculateMaterialPhysics(1.f, PID::Photon, absCharge, MaterialTraversalDirection::AlongMomentum, material, resultMomentum, resultTheta2, resultVariance);
    BOOST_CHECK(!result);
    BOOST_CHECK_EQUAL(resultMomentum, 0.f);
    BOOST_CHECK_EQUAL(resultTheta2, 0.f);
    BOOST_CHECK_EQUAL(resultVariance, 0.f);
  }
}

BOOST_AUTO_TEST_CASE(AbsChargeVariantsScaleHighlandQuadratically)
{
  IntegratedMaterialBudget material{0.03f, 0.f}; // MCS-only: isolates the charge scaling.

  float baseMomentum = 0.f;
  float baseTheta2 = 0.f;
  float baseVariance = 0.f;
  const bool base = calculateMaterialPhysics(1.5f, PID::Pion, 1, MaterialTraversalDirection::AlongMomentum, material, baseMomentum, baseTheta2, baseVariance);
  BOOST_REQUIRE(base);
  for (uint8_t absCharge : {2, 3, 200}) {

    float resultMomentum = 0.f;
    float resultTheta2 = 0.f;
    float resultVariance = 0.f;
    const bool result = calculateMaterialPhysics(1.5f, PID::Pion, absCharge, MaterialTraversalDirection::AlongMomentum, material, resultMomentum, resultTheta2, resultVariance);
    BOOST_REQUIRE(result);
    const float expectedRatio = static_cast<float>(absCharge) * static_cast<float>(absCharge);
    BOOST_CHECK(closeTo(resultTheta2, expectedRatio * baseTheta2));
  }
}

BOOST_AUTO_TEST_CASE(DirectionInvalidCastRejected)
{
  IntegratedMaterialBudget material{0.f, 0.f};
  for (uint8_t raw : {2, 255}) {
    auto direction = static_cast<MaterialTraversalDirection>(raw);

    float resultMomentum = 0.f;
    float resultTheta2 = 0.f;
    float resultVariance = 0.f;
    const bool result = calculateMaterialPhysics(1.f, PID::Pion, 1, direction, material, resultMomentum, resultTheta2, resultVariance);
    BOOST_CHECK(!result);
    BOOST_CHECK_EQUAL(resultMomentum, 0.f);
    BOOST_CHECK_EQUAL(resultTheta2, 0.f);
    BOOST_CHECK_EQUAL(resultVariance, 0.f);
  }
}

BOOST_AUTO_TEST_CASE(MaterialFieldsMustBeNonNegative)
{
  const std::vector<IntegratedMaterialBudget> invalidMaterials = {
    {-1.f, 0.1f}, {0.1f, -1.f}, {-1.f, -1.f}};
  for (auto material : invalidMaterials) {

    float resultMomentum = 0.f;
    float resultTheta2 = 0.f;
    float resultVariance = 0.f;
    const bool result = calculateMaterialPhysics(1.f, PID::Pion, 1, MaterialTraversalDirection::AlongMomentum, material, resultMomentum, resultTheta2, resultVariance);
    BOOST_CHECK(!result);
    BOOST_CHECK_EQUAL(resultMomentum, 0.f);
    BOOST_CHECK_EQUAL(resultTheta2, 0.f);
    BOOST_CHECK_EQUAL(resultVariance, 0.f);
  }
}

BOOST_AUTO_TEST_CASE(MomentumMustBePositive)
{
  IntegratedMaterialBudget material{0.f, 0.f};
  for (float momentum : {0.f, -1.f}) {

    float resultMomentum = 0.f;
    float resultTheta2 = 0.f;
    float resultVariance = 0.f;
    const bool result = calculateMaterialPhysics(momentum, PID::Pion, 1, MaterialTraversalDirection::AlongMomentum, material, resultMomentum, resultTheta2, resultVariance);
    BOOST_CHECK(!result);
    BOOST_CHECK_EQUAL(resultMomentum, 0.f);
    BOOST_CHECK_EQUAL(resultTheta2, 0.f);
    BOOST_CHECK_EQUAL(resultVariance, 0.f);
  }
}

BOOST_AUTO_TEST_CASE(ZeroMaterialIsAPassThrough)
{
  IntegratedMaterialBudget material{0.f, 0.f};

  float resultMomentum = 0.f;
  float resultTheta2 = 0.f;
  float resultVariance = 0.f;
  const bool result = calculateMaterialPhysics(1.f, PID::Pion, 1, MaterialTraversalDirection::AlongMomentum, material, resultMomentum, resultTheta2, resultVariance);
  BOOST_REQUIRE(result);
  BOOST_CHECK_EQUAL(resultMomentum, 1.f);
  BOOST_CHECK_EQUAL(energyChange(1.f, resultMomentum, PID::Pion), 0.f);
  BOOST_CHECK_EQUAL(resultTheta2, 0.f);
  BOOST_CHECK_EQUAL(resultVariance, 0.f);
}

BOOST_AUTO_TEST_CASE(McsOnlyMaterialMatchesAnalyticHighland)
{
  const float p0 = 2.f;
  const float mass = PID(PID::Pion).getMass();
  IntegratedMaterialBudget material{0.05f, 0.f};

  float resultMomentum = 0.f;
  float resultTheta2 = 0.f;
  float resultVariance = 0.f;
  const bool result = calculateMaterialPhysics(p0, PID::Pion, 1, MaterialTraversalDirection::AlongMomentum, material, resultMomentum, resultTheta2, resultVariance);
  BOOST_REQUIRE(result);
  BOOST_CHECK_EQUAL(resultMomentum, p0);
  BOOST_CHECK_EQUAL(energyChange(p0, resultMomentum, PID::Pion), 0.f);

  BOOST_CHECK_EQUAL(resultVariance, 0.f);

  const double e0 = std::sqrt(static_cast<double>(p0) * p0 + static_cast<double>(mass) * mass);
  const double beta2 = (static_cast<double>(p0) * p0) / (e0 * e0);
  const double expectedTheta2 = kHighlandConst2 / (beta2 * p0 * p0) * material.xOverX0;
  BOOST_CHECK(closeTo(resultTheta2, static_cast<float>(expectedTheta2)));
}

BOOST_AUTO_TEST_CASE(EnergyLossOnlyMaterialProducesNoScattering)
{
  IntegratedMaterialBudget material{0.f, 0.02f};

  float resultMomentum = 0.f;
  float resultTheta2 = 0.f;
  float resultVariance = 0.f;
  const bool result = calculateMaterialPhysics(2.f, PID::Pion, 1, MaterialTraversalDirection::AlongMomentum, material, resultMomentum, resultTheta2, resultVariance);
  BOOST_REQUIRE(result);
  BOOST_CHECK_EQUAL(resultTheta2, 0.f);
  BOOST_CHECK_LT(resultMomentum, 2.f);
  BOOST_CHECK_LT(energyChange(2.f, resultMomentum, PID::Pion), 0.f);

  BOOST_CHECK_GT(resultVariance, 0.f);
}

BOOST_AUTO_TEST_CASE(CombinedMaterialMatchesOracle)
{
  const float p0 = 1.2f;
  const PID pid = PID::Kaon;
  const uint8_t absCharge = 1;
  IntegratedMaterialBudget material{0.04f, 0.03f};

  float resultMomentum = 0.f;
  float resultTheta2 = 0.f;
  float resultVariance = 0.f;
  const bool result = calculateMaterialPhysics(p0, pid, absCharge, MaterialTraversalDirection::AlongMomentum, material, resultMomentum, resultTheta2, resultVariance);
  BOOST_REQUIRE(result);

  auto oracle = referenceCharged(p0, pid.getMass(), absCharge, material.xOverX0, material.arealDensityGPerCm2, true);
  BOOST_CHECK(!oracle.stopped && !oracle.nonFinite);

  BOOST_CHECK(closeTo(resultMomentum, static_cast<float>(oracle.momentumAfterGeV)));
  BOOST_CHECK(closeTo(energyChange(p0, resultMomentum, pid), static_cast<float>(oracle.signedEnergyChangeGeV)));
  BOOST_CHECK(closeTo(resultTheta2, static_cast<float>(oracle.highlandTheta2Rad2)));
  BOOST_CHECK(closeTo(resultVariance, static_cast<float>(oracle.relativeInverseMomentumVariance)));
}

BOOST_AUTO_TEST_CASE(LossAndGainHaveOppositeSignedEnergyChange)
{
  const float p0 = 1.5f;
  IntegratedMaterialBudget material{0.f, 0.005f}; // small enough to stay single-substep

  float lossMomentum = 0.f;
  float lossTheta2 = 0.f;
  float lossVariance = 0.f;
  const bool loss = calculateMaterialPhysics(p0, PID::Proton, 1, MaterialTraversalDirection::AlongMomentum, material, lossMomentum, lossTheta2, lossVariance);

  float gainMomentum = 0.f;
  float gainTheta2 = 0.f;
  float gainVariance = 0.f;
  const bool gain = calculateMaterialPhysics(p0, PID::Proton, 1, MaterialTraversalDirection::OppositeMomentum, material, gainMomentum, gainTheta2, gainVariance);
  BOOST_REQUIRE(loss);
  BOOST_REQUIRE(gain);

  BOOST_CHECK_LT(energyChange(p0, lossMomentum, PID::Proton), 0.f);
  BOOST_CHECK_GT(energyChange(p0, gainMomentum, PID::Proton), 0.f);
  BOOST_CHECK(closeTo(energyChange(p0, lossMomentum, PID::Proton), -energyChange(p0, gainMomentum, PID::Proton), AbsTol, 1.e-2f));
  BOOST_CHECK_LT(lossMomentum, p0);
  BOOST_CHECK_GT(gainMomentum, p0);
}

BOOST_AUTO_TEST_CASE(MaterialAcrossSubstepRangeMatchesOracle)
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

  // OppositeMomentum (energy gain) is used deliberately: it isolates the
  // substep-count bookkeeping from the (physically legitimate) risk that a
  // large requested ratio also represents more energy loss than the
  // particle's kinetic energy can absorb, which is covered separately by
  // the StoppingIsDetected test.
  for (const double ratio : {0.3, 5.5, 48.9, 49.5, 60.0, 1.e6}) {
    IntegratedMaterialBudget material{0.f, static_cast<float>(arealDensityForRatio(ratio))};

    float resultMomentum = 0.f;
    float resultTheta2 = 0.f;
    float resultVariance = 0.f;
    const bool result = calculateMaterialPhysics(p0, pid, 1, MaterialTraversalDirection::OppositeMomentum, material, resultMomentum, resultTheta2, resultVariance);
    BOOST_REQUIRE_MESSAGE(result, "unexpected failure for ratio " << ratio);

    auto oracle = referenceCharged(p0, mass, 1., 0., material.arealDensityGPerCm2, false);
    BOOST_REQUIRE(!oracle.stopped && !oracle.nonFinite);
    BOOST_CHECK(closeTo(resultMomentum, static_cast<float>(oracle.momentumAfterGeV)));
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

  float resultMomentum = 0.f;
  float resultTheta2 = 0.f;
  float resultVariance = 0.f;
  const bool result = calculateMaterialPhysics(p0, pid, 1, MaterialTraversalDirection::OppositeMomentum, material, resultMomentum, resultTheta2, resultVariance);
  BOOST_REQUIRE(result);

  auto oracle = referenceCharged(p0, pid.getMass(), 1., 0., material.arealDensityGPerCm2, false);
  BOOST_REQUIRE(!oracle.stopped && !oracle.nonFinite);
  BOOST_CHECK_EQUAL(oracle.substeps, o2::track::MaxELossIter);
  BOOST_CHECK(closeTo(energyChange(p0, resultMomentum, pid), static_cast<float>(oracle.signedEnergyChangeGeV), AbsTol, 2.e-3f));
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

  float resultMomentum = 0.f;
  float resultTheta2 = 0.f;
  float resultVariance = 0.f;
  const bool result = calculateMaterialPhysics(p0, pid, 1, MaterialTraversalDirection::AlongMomentum, material, resultMomentum, resultTheta2, resultVariance);
  BOOST_REQUIRE(result);

  const double e0 = std::sqrt(static_cast<double>(p0) * p0 + mass * mass);
  const double bg0 = p0 / mass;
  const double dedx0 = o2::track::BetheBlochSolidOpt<double>(bg0);
  const double naiveEnergyAfter = e0 - dedx0 * material.arealDensityGPerCm2;

  auto oracle = referenceCharged(p0, mass, 1., 0., material.arealDensityGPerCm2, true);
  BOOST_REQUIRE(!oracle.stopped && !oracle.nonFinite);
  const double recomputedEnergyAfter = e0 + oracle.signedEnergyChangeGeV;

  BOOST_CHECK(closeTo(energyChange(p0, resultMomentum, pid), static_cast<float>(oracle.signedEnergyChangeGeV)));
  BOOST_CHECK_GT(std::fabs(recomputedEnergyAfter - naiveEnergyAfter), 1.e-4);
}

BOOST_AUTO_TEST_CASE(StoppingIsDetected)
{
  IntegratedMaterialBudget material{0.f, 50.f}; // grossly exceeds a 0.5 GeV/c proton's kinetic energy

  float resultMomentum = 0.f;
  float resultTheta2 = 0.f;
  float resultVariance = 0.f;
  const bool result = calculateMaterialPhysics(0.5f, PID::Proton, 1, MaterialTraversalDirection::AlongMomentum, material, resultMomentum, resultTheta2, resultVariance);
  BOOST_CHECK(!result);
  BOOST_CHECK_EQUAL(resultMomentum, 0.f);
  BOOST_CHECK_EQUAL(resultTheta2, 0.f);
  BOOST_CHECK_EQUAL(resultVariance, 0.f);
}

BOOST_AUTO_TEST_CASE(FinalMomentumBoundary)
{
  IntegratedMaterialBudget material{0.f, 0.f}; // zero material: momentumAfter == momentumBefore exactly

  float atThresholdMomentum = 0.f;
  float atThresholdTheta2 = 0.f;
  float atThresholdVariance = 0.f;
  const bool atThreshold = calculateMaterialPhysics(kMinMomentumGeV, PID::Pion, 1, MaterialTraversalDirection::AlongMomentum, material, atThresholdMomentum, atThresholdTheta2, atThresholdVariance);
  BOOST_REQUIRE(atThreshold);
  BOOST_CHECK_EQUAL(atThresholdMomentum, kMinMomentumGeV);

  float belowThresholdMomentum = 0.f;
  float belowThresholdTheta2 = 0.f;
  float belowThresholdVariance = 0.f;
  const bool belowThreshold = calculateMaterialPhysics(std::nextafter(kMinMomentumGeV, 0.f), PID::Pion, 1,
                                                       MaterialTraversalDirection::AlongMomentum, material, belowThresholdMomentum, belowThresholdTheta2, belowThresholdVariance);
  BOOST_CHECK(!belowThreshold);
  BOOST_CHECK_EQUAL(belowThresholdMomentum, 0.f);
  BOOST_CHECK_EQUAL(belowThresholdTheta2, 0.f);
  BOOST_CHECK_EQUAL(belowThresholdVariance, 0.f);
}

BOOST_AUTO_TEST_CASE(ExcessiveScatteringIsRejected)
{
  IntegratedMaterialBudget material{500.f, 0.f}; // absurdly thick, drives theta^2 past pi^2

  float resultMomentum = 0.f;
  float resultTheta2 = 0.f;
  float resultVariance = 0.f;
  const bool result = calculateMaterialPhysics(0.1f, PID::Pion, 1, MaterialTraversalDirection::AlongMomentum, material, resultMomentum, resultTheta2, resultVariance);
  BOOST_CHECK(!result);
  BOOST_CHECK_EQUAL(resultMomentum, 0.f);
  BOOST_CHECK_EQUAL(resultTheta2, 0.f);
  BOOST_CHECK_EQUAL(resultVariance, 0.f);
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

  float resultMomentum = 0.f;
  float resultTheta2 = 0.f;
  float resultVariance = 0.f;
  const bool result = calculateMaterialPhysics(p0, PID::Proton, 1, MaterialTraversalDirection::AlongMomentum, material, resultMomentum, resultTheta2, resultVariance);
  BOOST_CHECK(!result);
  BOOST_CHECK_EQUAL(resultMomentum, 0.f);
  BOOST_CHECK_EQUAL(resultTheta2, 0.f);
  BOOST_CHECK_EQUAL(resultVariance, 0.f);

  float repeatMomentum = 0.f;
  float repeatTheta2 = 0.f;
  float repeatVariance = 0.f;
  const bool repeat = calculateMaterialPhysics(p0, PID::Proton, 1, MaterialTraversalDirection::AlongMomentum, material, repeatMomentum, repeatTheta2, repeatVariance);
  BOOST_CHECK_EQUAL(result, repeat);
  BOOST_CHECK_EQUAL(std::bit_cast<uint32_t>(resultMomentum), std::bit_cast<uint32_t>(repeatMomentum));
  BOOST_CHECK_EQUAL(std::bit_cast<uint32_t>(resultTheta2), std::bit_cast<uint32_t>(repeatTheta2));
  BOOST_CHECK_EQUAL(std::bit_cast<uint32_t>(resultVariance), std::bit_cast<uint32_t>(repeatVariance));
}

BOOST_AUTO_TEST_CASE(DirectBetheBlochReferenceValue)
{
  const float p0 = 1.f;
  const PID pid = PID::Proton;
  const double mass = pid.getMass();
  IntegratedMaterialBudget material{0.f, 0.001f}; // small enough to guarantee a single substep

  float resultMomentum = 0.f;
  float resultTheta2 = 0.f;
  float resultVariance = 0.f;
  const bool result = calculateMaterialPhysics(p0, pid, 1, MaterialTraversalDirection::AlongMomentum, material, resultMomentum, resultTheta2, resultVariance);
  BOOST_REQUIRE(result);

  const double e0 = std::sqrt(static_cast<double>(p0) * p0 + mass * mass);
  const double bg0 = p0 / mass;
  const double dedx = o2::track::BetheBlochSolidOpt<double>(bg0);
  const double expectedEnergyAfter = e0 - dedx * material.arealDensityGPerCm2;
  const double expectedSignedChange = expectedEnergyAfter - e0;
  BOOST_CHECK(closeTo(energyChange(p0, resultMomentum, pid), static_cast<float>(expectedSignedChange)));
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

    float resultMomentum = 0.f;
    float resultTheta2 = 0.f;
    float resultVariance = 0.f;
    const bool result = calculateMaterialPhysics(p0, pid, absCharge, MaterialTraversalDirection::AlongMomentum, material, resultMomentum, resultTheta2, resultVariance);
    BOOST_REQUIRE(result);

    const double q2 = static_cast<double>(absCharge) * absCharge;
    const double expectedDE = dedxUnit * q2 * material.arealDensityGPerCm2;
    const double expectedEnergyAfter = e0 - expectedDE;
    const double expectedSignedChange = expectedEnergyAfter - e0;
    const double expectedMomentumAfter = std::sqrt(expectedEnergyAfter * expectedEnergyAfter - mass * mass);
    const double expectedVariance = kStragglingConst * kStragglingConst * std::fabs(expectedSignedChange) * e0 * e0 /
                                    (static_cast<double>(p0) * p0 * p0 * p0);

    BOOST_CHECK(closeTo(energyChange(p0, resultMomentum, pid), static_cast<float>(expectedSignedChange)));
    BOOST_CHECK(closeTo(resultMomentum, static_cast<float>(expectedMomentumAfter)));
    BOOST_CHECK(closeTo(resultVariance, static_cast<float>(expectedVariance)));

    if (absCharge == 1) {
      baseSignedChange = energyChange(p0, resultMomentum, pid);
      baseVariance = resultVariance;
    } else {
      const float q2f = static_cast<float>(absCharge) * static_cast<float>(absCharge);
      BOOST_CHECK(closeTo(energyChange(p0, resultMomentum, pid), q2f * baseSignedChange));
      BOOST_CHECK(closeTo(resultVariance, q2f * baseVariance));
    }
  }
}

BOOST_AUTO_TEST_CASE(RepeatedCallsHaveIdenticalPhysicsOutputs)
{
  IntegratedMaterialBudget material{0.03f, 0.02f};

  float aMomentum = 0.f;
  float aTheta2 = 0.f;
  float aVariance = 0.f;
  const bool a = calculateMaterialPhysics(1.3f, PID::Kaon, 1, MaterialTraversalDirection::AlongMomentum, material, aMomentum, aTheta2, aVariance);

  float bMomentum = 0.f;
  float bTheta2 = 0.f;
  float bVariance = 0.f;
  const bool b = calculateMaterialPhysics(1.3f, PID::Kaon, 1, MaterialTraversalDirection::AlongMomentum, material, bMomentum, bTheta2, bVariance);
  BOOST_CHECK_EQUAL(a, b);
  BOOST_CHECK_EQUAL(std::bit_cast<uint32_t>(aMomentum), std::bit_cast<uint32_t>(bMomentum));
  BOOST_CHECK_EQUAL(std::bit_cast<uint32_t>(aTheta2), std::bit_cast<uint32_t>(bTheta2));
  BOOST_CHECK_EQUAL(std::bit_cast<uint32_t>(aVariance), std::bit_cast<uint32_t>(bVariance));
}
