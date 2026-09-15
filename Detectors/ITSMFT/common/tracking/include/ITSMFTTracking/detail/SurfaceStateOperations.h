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

#include "ITSMFTTracking/SurfaceTrackState.h"
#include "ITSMFTTracking/SurfaceMeasurement.h"

// Coordinate-family leaves used only by Propagator and their numerical
// tests. Production callers use Propagator's
// descriptor/state-driven API rather than selecting a family themselves.
namespace o2::itsmft::tracking::detail
{
namespace barrel
{
bool rotate(SurfaceTrackState& state, float targetAlpha) noexcept;
bool propagate(SurfaceTrackState& state, float targetX, float bz) noexcept;
bool predictedChi2(const SurfaceTrackState& state, const SurfaceMeasurement& measurement, float& chi2) noexcept;
bool update(SurfaceTrackState& state, const SurfaceMeasurement& measurement, float& chi2) noexcept;
bool stateChi2(const SurfaceTrackState& reference, const SurfaceTrackState& candidate, float& chi2) noexcept;

#ifndef GPUCA_GPUCODE
bool rotate(SurfaceTrackState& state, SurfaceTrackParameters& linRef, float targetAlpha, float bz) noexcept;
bool propagate(SurfaceTrackState& state, SurfaceTrackParameters& linRef, float targetX, float bz) noexcept;
bool shiftReferenceToMeasurement(SurfaceTrackParameters& linRef, const SurfaceMeasurement& measurement) noexcept;
#endif
} // namespace barrel

namespace forward
{
bool propagate(SurfaceTrackState& state, float targetZ, float bz) noexcept;
bool propagate(SurfaceTrackState& state, SurfaceTrackParameters& linRef,
               float targetZ, float bz) noexcept;
bool predictedChi2(const SurfaceTrackState& state, const SurfaceMeasurement& measurement, float& chi2) noexcept;
bool update(SurfaceTrackState& state, const SurfaceMeasurement& measurement, float& chi2) noexcept;
bool stateChi2(const SurfaceTrackState& reference, const SurfaceTrackState& candidate, float& chi2) noexcept;

#ifndef GPUCA_GPUCODE
bool shiftReferenceToMeasurement(SurfaceTrackParameters& linRef, const SurfaceMeasurement& measurement) noexcept;
#endif
} // namespace forward
} // namespace o2::itsmft::tracking::detail

#endif
