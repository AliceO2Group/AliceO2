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

#ifndef ALICEO2_ITSMFT_TRACKING_DETAIL_SURFACESTATEOPERATIONS_H_
#define ALICEO2_ITSMFT_TRACKING_DETAIL_SURFACESTATEOPERATIONS_H_

#include "ITSMFTTracking/MaterialPhysics.h"
#include "ITSMFTTracking/SurfaceTrackState.h"
#include "ITSMFTTracking/SurfaceMeasurement.h"
#include "ITSMFTTracking/SurfaceStateOperationResult.h"

// Coordinate-family leaves used only by Propagator and their numerical
// tests. Production callers use Propagator's
// descriptor/state-driven API rather than selecting a family themselves.
namespace o2::itsmft::tracking::detail
{
namespace barrel
{
bool rotate(SurfaceTrackState& state, float targetAlpha, OperationFailureReason& reason) noexcept;
bool propagate(SurfaceTrackState& state, float targetX, float bz, OperationFailureReason& reason) noexcept;
bool predictedChi2(const SurfaceTrackState& state, const SurfaceMeasurement& measurement, float& chi2,
                   OperationFailureReason& reason) noexcept;
bool update(SurfaceTrackState& state, const SurfaceMeasurement& measurement, float& chi2,
            OperationFailureReason& reason) noexcept;
material::MaterialOperationResult correctForMaterial(SurfaceTrackState& state, material::IntegratedMaterialBudget materialBudget,
                                                     material::MaterialTraversalDirection direction) noexcept;
material::MaterialOperationResult correctForMaterial(SurfaceTrackState& state, SurfaceTrackParameters& linRef,
                                                     material::IntegratedMaterialBudget materialBudget,
                                                     material::MaterialTraversalDirection direction) noexcept;
bool stateChi2(const SurfaceTrackState& reference, const SurfaceTrackState& candidate, float& chi2,
               OperationFailureReason& reason) noexcept;

#ifndef GPUCA_GPUCODE
bool rotate(SurfaceTrackState& state, SurfaceTrackParameters& linRef, float targetAlpha, float bz,
            OperationFailureReason& reason) noexcept;
bool propagate(SurfaceTrackState& state, SurfaceTrackParameters& linRef, float targetX, float bz,
               OperationFailureReason& reason) noexcept;
bool shiftReferenceToMeasurement(SurfaceTrackParameters& linRef, const SurfaceMeasurement& measurement,
                                 OperationFailureReason& reason) noexcept;
#endif
} // namespace barrel

namespace forward
{
bool propagate(SurfaceTrackState& state, float targetZ, float bz, OperationFailureReason& reason) noexcept;
bool propagate(SurfaceTrackState& state, SurfaceTrackParameters& linRef,
               float targetZ, float bz, OperationFailureReason& reason) noexcept;
bool predictedChi2(const SurfaceTrackState& state, const SurfaceMeasurement& measurement, float& chi2,
                   OperationFailureReason& reason) noexcept;
bool update(SurfaceTrackState& state, const SurfaceMeasurement& measurement, float& chi2,
            OperationFailureReason& reason) noexcept;
constexpr float highlandTheta2(float inverseMomentum, float xOverX0) noexcept
{
  const float theta = 0.0136f * inverseMomentum;
  return theta * theta * xOverX0;
}
bool correctForMaterial(SurfaceTrackState& state, float xOverX0, OperationFailureReason& reason) noexcept;
material::MaterialOperationResult correctForMaterial(SurfaceTrackState& state, material::IntegratedMaterialBudget materialBudget,
                                                     material::MaterialTraversalDirection direction) noexcept;
material::MaterialOperationResult correctForMaterial(SurfaceTrackState& state, SurfaceTrackParameters& linRef,
                                                     material::IntegratedMaterialBudget materialBudget,
                                                     material::MaterialTraversalDirection direction) noexcept;
bool stateChi2(const SurfaceTrackState& reference, const SurfaceTrackState& candidate, float& chi2,
               OperationFailureReason& reason) noexcept;

#ifndef GPUCA_GPUCODE
bool shiftReferenceToMeasurement(SurfaceTrackParameters& linRef, const SurfaceMeasurement& measurement,
                                 OperationFailureReason& reason) noexcept;
#endif
} // namespace forward
} // namespace o2::itsmft::tracking::detail

#endif
