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

#include <cstdint>

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

// Scalar material-physics kernel for charged particles.
// pid supplies the mass; absCharge supplies |q| for energy-loss and
// scattering scale factors. absCharge must be nonzero and need not equal
// PID::getCharge().
//
// Validation precedence (first failure wins): invalid direction, negative
// material, non-positive momentum, invalid PID, then a charged massless PID.
// The PID range is checked before accessing its mass.
//
// For charged massive states, non-positive beta^2 is rejected before either
// material-effect calculation.
//
// For charged massive states, momentumGeV is the caller-selected physical
// momentum; no covariance projection is performed. Energy loss uses the same
// capped-substep Bethe-Bloch algorithm as
// o2::track::TrackParametrizationWithError::correctForMaterial(). The
// requested substep count is
//   1 + floor(|dE_full| / eKin * o2::track::ELoss2EKinThreshInv)
// with a range-bounded float-to-int conversion, capped at
// o2::track::MaxELossIter (50). All arealDensityGPerCm2 is processed; only the
// granularity changes. Bethe-Bloch is recomputed from the current momentum
// at each substep.
// MaterialTraversalDirection::AlongMomentum subtracts energy per substep;
// OppositeMomentum adds it. Reject particles stopped in material or ending
// with momentum below 0.01 GeV/c.
//
// highlandTheta2Rad2 and relativeInverseMomentumVariance use the simplified
// O2 Highland variance (no logarithmic correction) and pre-material momentum,
// energy, and beta. Both scale with absCharge^2. highlandTheta2Rad2 > pi^2
// is rejected.
//
// This kernel does not construct track states, detector geometry, or
// ITS/MFT/topology/propagation objects.
// Output values are committed only on success.
bool calculateMaterialPhysics(
  float momentumGeV,
  o2::track::PID pid,
  uint8_t absCharge,
  MaterialTraversalDirection direction,
  IntegratedMaterialBudget material,
  float& momentumAfterGeV,
  float& highlandTheta2Rad2,
  float& relativeInverseMomentumVariance) noexcept;

} // namespace o2::itsmft::tracking::material

#endif // ALICEO2_ITSMFT_TRACKING_MATERIALPHYSICS_H_
