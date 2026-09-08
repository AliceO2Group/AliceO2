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

#ifndef ALICEO2_ITSMFT_TRACKING_MATERIALPHYSICS_H_
#define ALICEO2_ITSMFT_TRACKING_MATERIALPHYSICS_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ReconstructionDataFormats/PID.h"

// This header and its implementation are host-only; GPU compilation is not
// supported.

namespace o2::itsmft::tracking::material
{

// Material traversal direction relative to the particle momentum, independent
// of any propagation or covariance sign convention in the caller.
enum class MaterialTraversalDirection : uint8_t {
  AlongMomentum = 0,
  OppositeMomentum = 1
};

// Unsigned, path-integrated material budget. Both fields are non-negative;
// direction is supplied separately.
struct IntegratedMaterialBudget {
  float xOverX0;             ///< thickness in units of radiation length
  float arealDensityGPerCm2; ///< crossed length*density, g/cm^2
};

// Reasons a scalar material-physics operation can fail. SourceSurfaceKindMismatch,
// NonFiniteState, InvalidStateKinematics, and InvalidCovariance are reserved
// for future full-state operations and are never emitted by
// calculateMaterialPhysics().
enum class MaterialFailureReason : uint8_t {
  None = 0,
  SourceSurfaceKindMismatch = 1,
  NonFiniteState = 2,
  InvalidStateKinematics = 3,
  InvalidPID = 4,
  ChargedMasslessPID = 5,
  InvalidDirection = 6,
  InvalidMaterial = 7,
  StoppedInMaterial = 8,
  MomentumBelowMinimum = 9,
  ExcessiveScattering = 10,
  InvalidCovariance = 11,
  NonFiniteResult = 12
};

enum class MaterialOperationFlags : uint8_t {
  None = 0,
  SubstepCountClamped = 1
};

// Result of one scalar material-physics evaluation. On failure,
// momentumBeforeGeV echoes the input, failure gives the reason, and all other
// fields are deterministic zero/None values with no physical meaning.
// reserved is always zero.
struct MaterialOperationResult {
  float momentumBeforeGeV;
  float momentumAfterGeV;
  float signedEnergyChangeGeV;
  float highlandTheta2Rad2;
  float relativeInverseMomentumVariance;
  uint8_t energyLossSubsteps;
  MaterialOperationFlags flags;
  MaterialFailureReason failure;
  uint8_t reserved;

  bool ok() const noexcept { return failure == MaterialFailureReason::None; }
};

// Lock the current in-memory layout; this is not a serialized or device ABI.
#define O2_ITSMFT_MATERIAL_ASSERT_LAYOUT(Type, Size, Alignment) \
  static_assert(std::is_standard_layout_v<Type>);               \
  static_assert(std::is_trivially_copyable_v<Type>);            \
  static_assert(sizeof(Type) == Size);                          \
  static_assert(alignof(Type) == Alignment)

O2_ITSMFT_MATERIAL_ASSERT_LAYOUT(IntegratedMaterialBudget, 8, 4);
O2_ITSMFT_MATERIAL_ASSERT_LAYOUT(MaterialOperationResult, 24, 4);

#undef O2_ITSMFT_MATERIAL_ASSERT_LAYOUT

static_assert(offsetof(MaterialOperationResult, momentumBeforeGeV) == 0);
static_assert(offsetof(MaterialOperationResult, momentumAfterGeV) == 4);
static_assert(offsetof(MaterialOperationResult, signedEnergyChangeGeV) == 8);
static_assert(offsetof(MaterialOperationResult, highlandTheta2Rad2) == 12);
static_assert(offsetof(MaterialOperationResult, relativeInverseMomentumVariance) == 16);
static_assert(offsetof(MaterialOperationResult, energyLossSubsteps) == 20);
static_assert(offsetof(MaterialOperationResult, flags) == 21);
static_assert(offsetof(MaterialOperationResult, failure) == 22);
static_assert(offsetof(MaterialOperationResult, reserved) == 23);

// Detector-neutral, PID/absCharge-aware scalar material-physics kernel.
// pid supplies the mass; absCharge supplies |q| for energy-loss and
// scattering scale factors. It need not equal PID::getCharge(). For
// absCharge == 0, validation still runs, then the operation succeeds with
// unchanged momentum and zero material effects.
//
// Validation precedence (first failure wins): invalid direction, negative
// material, non-positive momentum, invalid PID, then a charged massless PID.
// The PID range is checked before accessing its mass.
//
// For charged massive states, non-positive beta^2 is rejected before either
// material-effect calculation. Failure gives NonFiniteResult.
//
// For charged massive states, momentumGeV is the caller-selected physical
// momentum; no covariance projection is performed. Energy loss uses the same
// capped-substep Bethe-Bloch algorithm as
// o2::track::TrackParametrizationWithError::correctForMaterial(). The
// requested substep count is
//   1 + floor(|dE_full| / eKin * o2::track::ELoss2EKinThreshInv)
// with a range-bounded float-to-int conversion, capped at
// o2::track::MaxELossIter (50). flags marks SubstepCountClamped when the
// request exceeds 50. All arealDensityGPerCm2 is processed; only the
// granularity changes. Bethe-Bloch is recomputed from the current momentum
// at each substep.
// MaterialTraversalDirection::AlongMomentum subtracts energy per substep;
// OppositeMomentum adds it; signedEnergyChangeGeV is always
// Eafter - Ebefore. A particle whose energy would fall to or below its rest
// mass fails with StoppedInMaterial; a particle that completes with
// momentum below 0.01 GeV/c fails with MomentumBelowMinimum.
//
// highlandTheta2Rad2 and relativeInverseMomentumVariance use the simplified
// O2 Highland variance (no logarithmic correction) and pre-material momentum,
// energy, and beta. Both scale with absCharge^2. highlandTheta2Rad2 > pi^2
// fails with ExcessiveScattering.
//
// This kernel does not construct track states, detector geometry, or
// ITS/MFT/topology/propagation objects.
MaterialOperationResult calculateMaterialPhysics(
  float momentumGeV,
  o2::track::PID pid,
  uint8_t absCharge,
  MaterialTraversalDirection direction,
  IntegratedMaterialBudget material) noexcept;

} // namespace o2::itsmft::tracking::material

#endif // ALICEO2_ITSMFT_TRACKING_MATERIALPHYSICS_H_
