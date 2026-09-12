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

#include "ITSMFTTracking/MaterialPhysics.h"

#include <cmath>

// Reuse the public energy-loss constants and Bethe-Bloch helper. These
// headers are implementation details of this translation unit.
#include "ReconstructionDataFormats/TrackParametrization.h"
#include "ReconstructionDataFormats/TrackUtils.h"

namespace o2::itsmft::tracking::material
{

namespace
{
constexpr float kHighlandConst2 = 0.0136f * 0.0136f;
constexpr float kStragglingConst = 0.0007f;
constexpr float kMinMomentumGeV = 0.01f;

MaterialOperationResult makeFailureResult(float momentumGeV, MaterialFailureReason reason) noexcept
{
  MaterialOperationResult result{};
  result.momentumBeforeGeV = momentumGeV;
  result.momentumAfterGeV = 0.f;
  result.signedEnergyChangeGeV = 0.f;
  result.highlandTheta2Rad2 = 0.f;
  result.relativeInverseMomentumVariance = 0.f;
  result.energyLossSubsteps = 0;
  result.flags = MaterialOperationFlags::None;
  result.failure = reason;
  result.reserved = 0;
  return result;
}

MaterialOperationResult makeSuccessResult(float momentumBeforeGeV, float momentumAfterGeV,
                                          float signedEnergyChangeGeV, float highlandTheta2Rad2,
                                          float relativeInverseMomentumVariance,
                                          uint8_t energyLossSubsteps,
                                          MaterialOperationFlags flags) noexcept
{
  MaterialOperationResult result{};
  result.momentumBeforeGeV = momentumBeforeGeV;
  result.momentumAfterGeV = momentumAfterGeV;
  result.signedEnergyChangeGeV = signedEnergyChangeGeV;
  result.highlandTheta2Rad2 = highlandTheta2Rad2;
  result.relativeInverseMomentumVariance = relativeInverseMomentumVariance;
  result.energyLossSubsteps = energyLossSubsteps;
  result.flags = flags;
  result.failure = MaterialFailureReason::None;
  result.reserved = 0;
  return result;
}

// Compute the capped substep count without an out-of-range float-to-int
// conversion. Also report whether the requested count exceeded the cap.
void classifySubsteps(float fullStepEnergyLossGeV, float kineticEnergyGeV, uint8_t& substeps, bool& clamped) noexcept
{
  const float ratio = std::fabs(fullStepEnergyLossGeV) / kineticEnergyGeV * o2::track::ELoss2EKinThreshInv;
  if (ratio >= static_cast<float>(o2::track::MaxELossIter)) {
    substeps = static_cast<uint8_t>(o2::track::MaxELossIter);
    clamped = true;
    return;
  }
  // Keep the conversion in range even when ratio is unordered. Subsequent
  // arithmetic remains responsible for propagating invalid inputs.
  const float boundedRatio = ratio < static_cast<float>(o2::track::MaxELossIter) ? ratio : 0.f;
  const int requested = 1 + static_cast<int>(boundedRatio);
  substeps = static_cast<uint8_t>(requested);
  clamped = false;
}
} // namespace

MaterialOperationResult calculateMaterialPhysics(
  float momentumGeV,
  o2::track::PID pid,
  uint8_t absCharge,
  MaterialTraversalDirection direction,
  IntegratedMaterialBudget material) noexcept
{
  if (direction != MaterialTraversalDirection::AlongMomentum && direction != MaterialTraversalDirection::OppositeMomentum) {
    return makeFailureResult(momentumGeV, MaterialFailureReason::InvalidDirection);
  }
  if (material.xOverX0 < 0.f || material.arealDensityGPerCm2 < 0.f) {
    return makeFailureResult(momentumGeV, MaterialFailureReason::InvalidMaterial);
  }
  if (momentumGeV <= 0.f) {
    return makeFailureResult(momentumGeV, MaterialFailureReason::MomentumBelowMinimum);
  }
  if (pid.getID() >= o2::track::PID::NIDsTot) {
    return makeFailureResult(momentumGeV, MaterialFailureReason::InvalidPID);
  }
  const float mass = pid.getMass();
  if (absCharge != 0 && mass == 0.f) {
    return makeFailureResult(momentumGeV, MaterialFailureReason::ChargedMasslessPID);
  }

  if (absCharge == 0) {
    return makeSuccessResult(momentumGeV, momentumGeV, 0.f, 0.f, 0.f, 0, MaterialOperationFlags::None);
  }

  const float q2 = static_cast<float>(absCharge) * static_cast<float>(absCharge);
  const float p0 = momentumGeV;
  const float p0Squared = p0 * p0;
  const float e0 = std::sqrt(p0Squared + mass * mass);
  const float beta2 = p0Squared / (e0 * e0);
  if (beta2 <= 0.f) {
    return makeFailureResult(momentumGeV, MaterialFailureReason::NonFiniteResult);
  }

  float e = e0;
  float p = p0;
  uint8_t substeps = 0;
  MaterialOperationFlags flags = MaterialOperationFlags::None;

  if (material.arealDensityGPerCm2 > 0.f) {
    const float ekin = e0 - mass;
    const float bg0 = p0 / mass;
    const float dedx0 = o2::track::BetheBlochSolidOpt<float>(bg0) * q2;
    const float fullStepEnergyLoss = dedx0 * material.arealDensityGPerCm2;

    bool clamped = false;
    classifySubsteps(fullStepEnergyLoss, ekin, substeps, clamped);
    if (clamped) {
      flags = MaterialOperationFlags::SubstepCountClamped;
    }

    const float arealDensityStep = material.arealDensityGPerCm2 / static_cast<float>(substeps);
    MaterialFailureReason loopFailure = MaterialFailureReason::None;
    for (uint8_t i = 0; i < substeps; ++i) {
      const float bg = p / mass;
      const float dedx = o2::track::BetheBlochSolidOpt<float>(bg) * q2;
      const float dE = dedx * arealDensityStep;
      e = (direction == MaterialTraversalDirection::AlongMomentum) ? (e - dE) : (e + dE);
      if (e <= mass) {
        loopFailure = MaterialFailureReason::StoppedInMaterial;
        break;
      }
      p = std::sqrt(e * e - mass * mass);
    }
    if (loopFailure != MaterialFailureReason::None) {
      return makeFailureResult(momentumGeV, loopFailure);
    }
  }

  if (p < kMinMomentumGeV) {
    return makeFailureResult(momentumGeV, MaterialFailureReason::MomentumBelowMinimum);
  }
  const float signedEnergyChangeGeV = e - e0;

  float highlandTheta2Rad2 = 0.f;
  if (material.xOverX0 > 0.f) {
    highlandTheta2Rad2 = kHighlandConst2 / (beta2 * p0 * p0) * material.xOverX0 * q2;
    if (highlandTheta2Rad2 > o2::constants::math::PI * o2::constants::math::PI) {
      return makeFailureResult(momentumGeV, MaterialFailureReason::ExcessiveScattering);
    }
  }

  float relativeInverseMomentumVariance = 0.f;
  if (signedEnergyChangeGeV != 0.f) {
    relativeInverseMomentumVariance = kStragglingConst * kStragglingConst * std::fabs(signedEnergyChangeGeV) * e0 * e0 / (p0 * p0 * p0 * p0);
  }

  return makeSuccessResult(momentumGeV, p, signedEnergyChangeGeV, highlandTheta2Rad2,
                           relativeInverseMomentumVariance, substeps, flags);
}

} // namespace o2::itsmft::tracking::material
