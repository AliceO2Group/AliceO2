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

// Defines both detail::barrel::correctForMaterial(state, material, direction) and
// detail::forward::correctForMaterial(state, material, direction): the PID/absCharge-
// aware composite cylinder/disk operations built on the detector-neutral
// scalar kernel in MaterialPhysics.h. Both overloads share the complete
// preflight-validation/momentum-derivation/scratch-and-commit orchestration
// below; only the coordinate-specific kinematics check and covariance
// projection formula differ between them.
//
// This translation unit is host-only, does not construct or delegate through
// TrackParCovF or TrackParCovFwd, and includes TrackParametrization.h solely
// to reuse its public kCY2max/kCZ2max/kCSnp2max/kCTgl2max/kC1Pt2max constants
// for the retained barrel covariance-range handling (no narrower public
// header declares them; the same reuse pattern is already used by
// MaterialPhysics.cxx for its own constants).

#include "ITSMFTTracking/detail/SurfaceStateOperations.h"
#include "ITSMFTTracking/MaterialPhysics.h"

#include <cmath>
#include <cstdint>

#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/TrackParametrization.h"

namespace o2::itsmft::tracking
{
namespace
{

bool covarianceDiagonalsNonNegative(const SurfaceTrackState& state) noexcept
{
  for (uint8_t i = 0; i < 5; ++i) {
    if (state.covariance[packedCovarianceIndex(i, i)] < 0.f) {
      return false;
    }
  }
  return true;
}

// Physical-momentum derivation shared by both coordinate conventions. u = slot 4, t = slot 3.
bool derivePhysicalMomentum(const SurfaceTrackState& state, float& momentumGeV) noexcept
{
  const float t = state.parameters[3];
  const float u = state.parameters[4];
  const float absU = std::abs(u);
  const float pT = (state.absCharge == 0) ? (1.f / absU) : (static_cast<float>(state.absCharge) / absU);
  if (pT <= 0.f) {
    return false;
  }
  const float p = pT * std::sqrt(1.f + t * t);
  if (p <= 0.f) {
    return false;
  }
  momentumGeV = p;
  return true;
}

material::MaterialOperationResult makePreflightFailure(material::MaterialFailureReason reason) noexcept
{
  material::MaterialOperationResult result{};
  result.momentumBeforeGeV = 0.f;
  result.momentumAfterGeV = 0.f;
  result.signedEnergyChangeGeV = 0.f;
  result.highlandTheta2Rad2 = 0.f;
  result.relativeInverseMomentumVariance = 0.f;
  result.energyLossSubsteps = 0;
  result.flags = material::MaterialOperationFlags::None;
  result.failure = reason;
  result.reserved = 0;
  return result;
}

material::MaterialOperationResult makeProjectionFailure(const material::MaterialOperationResult& scalarResult,
                                                        material::MaterialFailureReason reason) noexcept
{
  material::MaterialOperationResult result{};
  result.momentumBeforeGeV = scalarResult.momentumBeforeGeV;
  result.momentumAfterGeV = 0.f;
  result.signedEnergyChangeGeV = 0.f;
  result.highlandTheta2Rad2 = 0.f;
  result.relativeInverseMomentumVariance = 0.f;
  result.energyLossSubsteps = 0;
  result.flags = material::MaterialOperationFlags::None;
  result.failure = reason;
  result.reserved = 0;
  return result;
}

// Barrel covariance-range upper bound, in (Y, Z, Snp, Tgl, Q2Pt) slot order:
// the retained TrackParametrizationWithError<float>::checkCovariance()
// range-clamp values, and the same five constants
// PropagatorBarrelOperations.cxx's post-propagate/rotate/update
// sanitization (ADR 0008) enforces.
constexpr float kBarrelMaxDiagonal[5] = {o2::track::kCY2max, o2::track::kCZ2max, o2::track::kCSnp2max,
                                         o2::track::kCTgl2max, o2::track::kC1Pt2max};

// Thin wrapper over the shared, detector-neutral sanitizeCovariance()
// (SurfaceTrackState.h): abs()'s each diagonal and, if it still exceeds
// the retained maximum, clamps it and rescales every off-diagonal entry
// involving that parameter by sqrt(max/diagonal). No legacy track object is
// constructed; this operates directly on the packed float covariance array.
// Formerly a private reimplementation of this exact behavior; now delegates to
// the one shared implementation also used by the barrel state operations'
// own post-propagate/rotate/update sanitization, with no behavioral change.
void limitBarrelCovariance(SurfaceTrackState& scratch) noexcept
{
  sanitizeCovariance(scratch, kBarrelMaxDiagonal);
}

// Shared preflight validation, steps 1-6 of the required order. Step 3's
// kind-specific extra check (barrel |Snp|<1 / forward alpha==0) is
// supplied by the caller; the shared slot-4-nonzero part of step 3 is applied
// here for both kinds.
template <typename FamilyKinematicsCheck>
bool preflightValidate(const SurfaceTrackState& state, SurfaceKind expectedFamily, FamilyKinematicsCheck&& familyCheck,
                       material::MaterialFailureReason& failure) noexcept
{
  if (state.kind != expectedFamily) {
    failure = material::MaterialFailureReason::SourceSurfaceKindMismatch;
    return false;
  }
  const float u = state.parameters[4];
  if (!familyCheck(state) || u == 0.f) {
    failure = material::MaterialFailureReason::InvalidStateKinematics;
    return false;
  }
  if (state.pid.getID() >= o2::track::PID::NIDsTot) {
    failure = material::MaterialFailureReason::InvalidPID;
    return false;
  }
  if (state.absCharge != 0 && state.pid.getMass() == 0.f) {
    failure = material::MaterialFailureReason::ChargedMasslessPID;
    return false;
  }
  if (!covarianceDiagonalsNonNegative(state)) {
    failure = material::MaterialFailureReason::InvalidCovariance;
    return false;
  }
  return true;
}

// Complete incidence-aware transactional operation shared by cylinder and disk
// states (Slice 2 "Transactional result contract"): validate the state and its
// incidence reference, derive physical momentum, scale the nominal material by
// the incidence path length, invoke the scalar kernel, project covariance on
// scratch only, validate the projected scratch, and commit exactly once.
// projectCovariance may additionally apply cylinder-specific covariance range
// handling (barrel only); it must not touch state.parameters[4], which this
// function updates uniformly for both kinds after projection.
//
// Unconditional no-op contract: once the scalar kernel succeeds, absCharge
// == 0 or an exactly-{0,0} materialBudget returns the scalar result
// immediately, before projectCovariance (and any barrel covariance-range
// limiting it applies) or the slot-4 update ever run. This holds even when
// the source state's barrel covariance diagonals already exceed the
// retained checkCovariance limits: those diagonals must not be silently
// clamped by an operation that has no material to apply.
template <typename FamilyKinematicsCheck, typename ScaleMaterial, typename ProjectCovariance>
material::MaterialOperationResult correctForMaterialImpl(SurfaceTrackState& state, SurfaceTrackParameters& incidenceReference,
                                                         SurfaceKind expectedFamily,
                                                         material::IntegratedMaterialBudget materialBudget,
                                                         material::MaterialTraversalDirection direction,
                                                         FamilyKinematicsCheck&& familyCheck,
                                                         ScaleMaterial&& scaleMaterial,
                                                         ProjectCovariance&& projectCovariance) noexcept
{
  material::MaterialFailureReason failure{};
  if (incidenceReference.kind != expectedFamily) {
    return makePreflightFailure(material::MaterialFailureReason::SourceSurfaceKindMismatch);
  }
  if (!familyCheck(incidenceReference) || incidenceReference.parameters[4] == 0.f) {
    return makePreflightFailure(material::MaterialFailureReason::InvalidStateKinematics);
  }
  if (!preflightValidate(state, expectedFamily, familyCheck, failure)) {
    return makePreflightFailure(failure);
  }

  float momentumBeforeGeV = 0.f;
  if (!derivePhysicalMomentum(state, momentumBeforeGeV)) {
    return makePreflightFailure(material::MaterialFailureReason::InvalidStateKinematics);
  }

  SurfaceTrackState scratchState = state;
  SurfaceTrackParameters scratchReference = incidenceReference;
  scaleMaterial(materialBudget, scratchReference);
  const auto scalarResult = material::calculateMaterialPhysics(momentumBeforeGeV, scratchState.pid, scratchState.absCharge, direction, materialBudget);
  if (!scalarResult.ok()) {
    return scalarResult;
  }

  const bool isNoopMaterial = (materialBudget.xOverX0 == 0.f && materialBudget.arealDensityGPerCm2 == 0.f);
  if (scratchState.absCharge == 0 || isNoopMaterial) {
    return scalarResult;
  }

  const float tBefore = scratchState.parameters[3];
  const float kBefore = scratchState.parameters[4];
  projectCovariance(scratchState, scalarResult, tBefore, kBefore);

  // The equality branch preserves the exact no-op invariant for the
  // MCS-only-with-unchanged-momentum case (xOverX0 > 0, arealDensity == 0):
  // x == y implies kAfter == kBefore bit-for-bit with no division rounding.
  // The nonzero-change branch keeps the accepted/legacy left-to-right
  // arithmetic (multiply, then divide) rather than dividing the momenta
  // first, which would prematurely underflow for extreme momentum ratios
  // and would not reproduce the retained nonzero-material rounding.
  const float kAfter = (scalarResult.momentumBeforeGeV == scalarResult.momentumAfterGeV)
                         ? kBefore
                         : (kBefore * scalarResult.momentumBeforeGeV) / scalarResult.momentumAfterGeV;
  scratchState.parameters[4] = kAfter;

  // Complete post-projection domain validation: the projected state must
  // still satisfy every kind/kinematics precondition the source state was
  // required to satisfy, and physical momentum must still be re-derivable.
  if (scratchState.parameters[4] == 0.f || !familyCheck(scratchState)) {
    return makeProjectionFailure(scalarResult, material::MaterialFailureReason::InvalidStateKinematics);
  }
  float momentumAfterDerived = 0.f;
  if (!derivePhysicalMomentum(scratchState, momentumAfterDerived)) {
    return makeProjectionFailure(scalarResult, material::MaterialFailureReason::InvalidStateKinematics);
  }
  if (!covarianceDiagonalsNonNegative(scratchState)) {
    return makeProjectionFailure(scalarResult, material::MaterialFailureReason::InvalidCovariance);
  }

  // Energy loss changes q/pT in the covariance-bearing state and its
  // incidence reference by the same pBefore/pAfter factor. The equality
  // branch keeps MCS-only corrections bit-exact.
  const float referenceKBefore = scratchReference.parameters[4];
  scratchReference.parameters[4] = (scalarResult.momentumBeforeGeV == scalarResult.momentumAfterGeV)
                                     ? referenceKBefore
                                     : (referenceKBefore * scalarResult.momentumBeforeGeV) / scalarResult.momentumAfterGeV;
  if (scratchReference.parameters[4] == 0.f || !std::isfinite(scratchReference.parameters[4])) {
    return makeProjectionFailure(scalarResult, material::MaterialFailureReason::InvalidStateKinematics);
  }

  state = scratchState;
  incidenceReference = scratchReference;
  return scalarResult;
}

} // namespace
} // namespace o2::itsmft::tracking

namespace o2::itsmft::tracking::detail::barrel
{
material::MaterialOperationResult correctForMaterial(SurfaceTrackState& state, SurfaceTrackParameters& linRef,
                                                     material::IntegratedMaterialBudget materialBudget,
                                                     material::MaterialTraversalDirection direction) noexcept
{
  auto familyCheck = [](const auto& s) noexcept {
    return std::abs(s.parameters[2]) < 1.f;
  };
  // ITS layer budgets describe a normal crossing of the cylindrical layer.
  // Match TrackParametrizationWithError::correctForMaterial(..., true), which
  // the legacy ITS tracker uses at every layer: lengthen both material
  // quantities by the path of the incident track before evaluating energy
  // loss and multiple scattering.
  auto scaleMaterial = [](material::IntegratedMaterialBudget& material, const SurfaceTrackParameters& incidence) noexcept {
    const float snp = incidence.parameters[2];
    const float tgl = incidence.parameters[3];
    const float cosPhi2 = (1.f - snp) * (1.f + snp);
    const float inverseCosLambda2 = 1.f + tgl * tgl;
    const float incidenceScale = std::sqrt(inverseCosLambda2 / cosPhi2);
    material.xOverX0 *= incidenceScale;
    material.arealDensityGPerCm2 *= incidenceScale;
  };
  // Barrel parameters are (Y, Z, Snp, Tgl, Q2Pt). The accepted Jacobian
  // requires q/pT unconditionally in slots 13/14, fixing the retained
  // TrackParametrizationWithError<float>::correctForMaterial() unit-charge
  // conditional that omits q/pT there (see the module doc comment).
  auto projectCovariance = [](SurfaceTrackState& scratch, const material::MaterialOperationResult& scalarResult,
                              float t, float k) noexcept {
    const float A = 1.f + t * t;
    const float snp = scratch.parameters[2];
    const float c2 = 1.f - snp * snp;
    const float h = scalarResult.highlandTheta2Rad2;
    const float R = scalarResult.relativeInverseMomentumVariance;
    scratch.covariance[packedCovarianceIndex(2, 2)] += h * A * c2;
    scratch.covariance[packedCovarianceIndex(3, 3)] += h * A * A;
    scratch.covariance[packedCovarianceIndex(4, 3)] += h * A * t * k;
    scratch.covariance[packedCovarianceIndex(4, 4)] += h * (t * k) * (t * k) + k * k * R;
    limitBarrelCovariance(scratch);
  };
  return correctForMaterialImpl(state, linRef, SurfaceKind::Cylinder, materialBudget, direction,
                                familyCheck, scaleMaterial, projectCovariance);
}

material::MaterialOperationResult correctForMaterial(SurfaceTrackState& state, material::IntegratedMaterialBudget materialBudget,
                                                     material::MaterialTraversalDirection direction) noexcept
{
  SurfaceTrackParameters incidenceReference{state};
  return correctForMaterial(state, incidenceReference, materialBudget, direction);
}

} // namespace o2::itsmft::tracking::detail::barrel

namespace o2::itsmft::tracking::detail::forward
{
material::MaterialOperationResult correctForMaterial(SurfaceTrackState& state, SurfaceTrackParameters& linRef,
                                                     material::IntegratedMaterialBudget materialBudget,
                                                     material::MaterialTraversalDirection direction) noexcept
{
  auto familyCheck = [](const auto& s) noexcept {
    return s.alpha == 0.f && s.parameters[3] != 0.f;
  };
  // MFT layer budgets describe a normal crossing of a disk. Match
  // TrackParCovFwd::addMCSEffect(), which lengthens x/X0 by csc(lambda), and
  // apply the same path-length scaling to the areal density used for energy
  // loss. For a linearized propagation the incidence comes from the
  // reference trajectory, exactly as for the barrel operation above.
  auto scaleMaterial = [](material::IntegratedMaterialBudget& material, const SurfaceTrackParameters& incidence) noexcept {
    const float tgl = incidence.parameters[3];
    const float incidenceScale = std::sqrt(1.f + tgl * tgl) / std::abs(tgl);
    material.xOverX0 *= incidenceScale;
    material.arealDensityGPerCm2 *= incidenceScale;
  };
  // Forward parameters are (X, Y, Phi, Tanl, Q2Pt); unlike barrel there is no
  // cos(phi)-like factor on the angular diagonal term, and forward does not
  // inherit barrel-only covariance range limiting. Slot 13 (the Q2Pt/Tanl
  // cross term) and the k^2*R straggling contribution to slot 14 are new
  // physics: the legacy TrackParCovFwd::addMCSEffect() never populates
  // slot 13 and has no charge/PID/energy-loss awareness at all.
  auto projectCovariance = [](SurfaceTrackState& scratch, const material::MaterialOperationResult& scalarResult,
                              float t, float k) noexcept {
    const float A = 1.f + t * t;
    const float h = scalarResult.highlandTheta2Rad2;
    const float R = scalarResult.relativeInverseMomentumVariance;
    scratch.covariance[packedCovarianceIndex(2, 2)] += h * A;
    scratch.covariance[packedCovarianceIndex(3, 3)] += h * A * A;
    scratch.covariance[packedCovarianceIndex(4, 3)] += h * A * t * k;
    scratch.covariance[packedCovarianceIndex(4, 4)] += h * (t * k) * (t * k) + k * k * R;
  };
  return correctForMaterialImpl(state, linRef, SurfaceKind::Disk, materialBudget, direction,
                                familyCheck, scaleMaterial, projectCovariance);
}

material::MaterialOperationResult correctForMaterial(SurfaceTrackState& state, material::IntegratedMaterialBudget materialBudget,
                                                     material::MaterialTraversalDirection direction) noexcept
{
  SurfaceTrackParameters incidenceReference{state};
  return correctForMaterial(state, incidenceReference, materialBudget, direction);
}

} // namespace o2::itsmft::tracking::detail::forward
