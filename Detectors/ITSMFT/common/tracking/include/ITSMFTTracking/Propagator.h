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

#ifndef ALICEO2_ITSMFT_TRACKING_PROPAGATOR_H_
#define ALICEO2_ITSMFT_TRACKING_PROPAGATOR_H_

#include "GPUCommonDef.h"

#ifndef GPUCA_GPUCODE

#include "ITSMFTTracking/MaterialPhysics.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"
#include "ITSMFTTracking/SurfaceTrackState.h"
#include "ITSMFTTracking/SurfaceMeasurement.h"

// Descriptor-driven propagation using the material and kind resolved from
// SurfaceDescriptor and SurfaceCatalogView.
namespace o2::itsmft::tracking
{

class Propagator
{
 public:
  // Convert to the target descriptor's convention, then propagate, apply its
  // material, gate the residual and update using the nonlinear seed fit.
  // State and chi2 are committed only after complete success.
  static bool attachMeasurement(SurfaceTrackState& state, const SurfaceDescriptor& targetSurface,
                                const SurfaceMeasurement& measurement, float bz,
                                material::MaterialTraversalDirection direction,
                                bool chi2GateEnabled, float maxChi2, float& chi2) noexcept;

  // Propagate in the state’s current surface convention to its target
  // reference coordinate. Disk transport uses helix propagation for
  // |bz| > 0.01f and linear transport otherwise. Both objects are unchanged
  // on failure when a linearization reference is supplied.
  static bool propagateToReference(SurfaceTrackState& state, float targetReferenceCoordinate, float bz) noexcept;
  static bool propagateToReference(SurfaceTrackState& state, SurfaceTrackParameters& linRef,
                                   float targetReferenceCoordinate, float bz) noexcept;

  // Re-express the state on the fixed target plane through its nominal point:
  // fixed z for Disk, fixed local x and radial alpha for Cylinder. Transport
  // the covariance with the surface-intersection Jacobian, including the
  // direction variation in bz. A matching kind is a no-op.
  //
  // Preserves absCharge, PID, and all fields outside the parameter convention.
  // Rejects tangent/unsupported directions and non-finite conversions without
  // changing the state. Cylinder targets require an outward radial direction.
  static bool convertKind(SurfaceTrackState& state, SurfaceKind targetKind, float bz) noexcept;

  // Propagate to a measurement, converting the state to the target surface
  // kind when needed, then applying material, the chi2 gate, and the update.
  // State, reference, and chi2 are committed only after complete success.
  //
  // The incoming chi2 must be finite and non-negative. maxChi2 is validated
  // the same way when the gate is enabled.
  static bool propagateToMeasurement(SurfaceTrackState& state, SurfaceTrackParameters& linRef,
                                     const SurfaceDescriptor& targetSurface, const SurfaceMeasurement& targetMeasurement,
                                     float bz, material::MaterialTraversalDirection direction,
                                     bool chi2GateEnabled, float maxChi2, float& chi2,
                                     bool shiftReferenceToMeasurement) noexcept;

 private:
  // Called only after propagation validates matching Cylinder/Disk kinds for
  // the state and incidence reference. Select material formulas from state.kind.
  static bool correctForMaterial(SurfaceTrackState& state, SurfaceTrackParameters& incidenceReference,
                                 material::IntegratedMaterialBudget materialBudget,
                                 material::MaterialTraversalDirection direction) noexcept;
};

} // namespace o2::itsmft::tracking

#endif // GPUCA_GPUCODE

#endif /* ALICEO2_ITSMFT_TRACKING_PROPAGATOR_H_ */
