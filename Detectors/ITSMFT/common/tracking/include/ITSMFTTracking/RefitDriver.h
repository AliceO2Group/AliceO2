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

#ifndef ALICEO2_ITSMFT_TRACKING_REFITDRIVER_H_
#define ALICEO2_ITSMFT_TRACKING_REFITDRIVER_H_

#include "GPUCommonDef.h"

#ifndef GPUCA_GPUCODE

#include <array>
#include <cmath>

#include <gsl/span>

#include "CommonConstants/MathConstants.h"
#include "ITSMFTTracking/Cell.h"
#include "ITSMFTTracking/GlobalMeasurement.h"
#include "ITSMFTTracking/TimeFrame.h"
#include "ITSMFTTracking/Propagator.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"
#include "ITSMFTTracking/SurfaceStateOperationResult.h"
#include "ReconstructionDataFormats/TrackParametrization.h"

// Descriptor-driven refit built on Propagator operations.
namespace o2::itsmft::tracking
{

namespace detail
{

struct RefitMeasurementSlot {
  SurfaceMeasurement measurement{};
  LayerId surface{};
  bool present{false};
};

/// Builds an ordered refit leg; holes remain explicit.
inline gsl::span<const RefitMeasurementSlot> assembleRefitLegSlots(
  const TrackSeed& seed,
  const TimeFrame& frame,
  gsl::span<const gsl::span<const GlobalMeasurement>> layerGlobals,
  int start, int end, int step,
  gsl::span<RefitMeasurementSlot> out,
  bool& valid) noexcept
{
  valid = layerGlobals.size() <= MaxLayoutSurfaces;
  int position = 0;
  for (int surfacePosition = start; surfacePosition != end && position < static_cast<int>(out.size()); surfacePosition += step) {
    const int clsIdx = seed.getCluster(surfacePosition);
    if (clsIdx == o2::its::constants::UnusedIndex) {
      out[position++] = {};
      continue;
    }
    if (!valid || clsIdx < 0 || static_cast<std::size_t>(clsIdx) >= layerGlobals[surfacePosition].size()) {
      valid = false;
      return {};
    }
    const auto& global = layerGlobals[surfacePosition][clsIdx];
    const auto surface = LayerId{static_cast<uint16_t>(surfacePosition)};
    const auto* measurement = frame.getSurfaceMeasurement(surface, global.clusterId);
    if (measurement == nullptr) {
      valid = false;
      return {};
    }
    out[position++] = RefitMeasurementSlot{*measurement, surface, true};
  }
  return gsl::span<const RefitMeasurementSlot>(out.data(), position);
}

// Holes are skipped; present slots must resolve to a descriptor. Commit state,
// reference, chi2 and count only after the full leg succeeds.
inline bool driveRefitLeg(SurfaceTrackState& state, SurfaceTrackParameters& linRef,
                          float& chi2, uint32_t& acceptedHitCount,
                          gsl::span<const RefitMeasurementSlot> orderedSlots, SurfaceCatalogView surfaceCatalog,
                          float bz, material::MaterialTraversalDirection direction,
                          bool shiftReferenceToMeasurement, float maxChi2, OperationFailureReason& reason) noexcept
{
  if (chi2 < 0.f) {
    reason = OperationFailureReason::PredictedChi2Failure;
    return false;
  }

  SurfaceTrackState scratchState = state;
  SurfaceTrackParameters scratchLinRef = linRef;
  float scratchChi2 = chi2;
  uint32_t scratchAcceptedHitCount = 0;
  constexpr uint32_t kChi2GateMinAcceptedHits = 3;
  for (const auto& slot : orderedSlots) {
    if (!slot.present) {
      continue;
    }
    if (!slot.surface.isValid() || !(surfaceCatalog.nSurfaces == 0 || surfaceCatalog.surfaces != nullptr) ||
        !(slot.surface.value() < surfaceCatalog.nSurfaces)) {
      reason = OperationFailureReason::InvalidSurfaceCatalogAssociation;
      return false;
    }
    const SurfaceDescriptor& descriptor = surfaceCatalog.getSurface(slot.surface);
    if (!Propagator::propagateToMeasurement(scratchState, scratchLinRef, descriptor, slot.measurement, bz, direction,
                                            scratchAcceptedHitCount >= kChi2GateMinAcceptedHits, maxChi2, scratchChi2,
                                            shiftReferenceToMeasurement, reason)) {
      return false;
    }
    ++scratchAcceptedHitCount;
  }
  state = scratchState;
  linRef = scratchLinRef;
  chi2 = scratchChi2;
  acceptedHitCount = scratchAcceptedHitCount;
  return true;
}

} // namespace detail

// Reset a refit leg to a loose diagonal covariance.
GPUhdi() void resetCovarianceForRefit(SurfaceTrackState& state) noexcept
{
  for (auto& element : state.covariance) {
    element = 0.f;
  }
  if (state.kind == SurfaceKind::Cylinder) {
    state.covariance[packedCovarianceIndex(0, 0)] = o2::track::kCY2max;
    state.covariance[packedCovarianceIndex(1, 1)] = o2::track::kCZ2max;
    state.covariance[packedCovarianceIndex(2, 2)] = o2::track::kCSnp2max;
    state.covariance[packedCovarianceIndex(3, 3)] = o2::track::kCTgl2max;
    const float q2pt = state.parameters[4];
    state.covariance[packedCovarianceIndex(4, 4)] = q2pt * q2pt * o2::track::kC1Pt2max;
  } else {
    constexpr float kCPhi2maxForward = o2::constants::math::PI * o2::constants::math::PI;
    state.covariance[packedCovarianceIndex(0, 0)] = o2::track::kCY2max;
    state.covariance[packedCovarianceIndex(1, 1)] = o2::track::kCY2max;
    state.covariance[packedCovarianceIndex(2, 2)] = kCPhi2maxForward;
    state.covariance[packedCovarianceIndex(3, 3)] = o2::track::kCTgl2max;
    const float invQPt = state.parameters[4];
    state.covariance[packedCovarianceIndex(4, 4)] = invQPt * invQPt * o2::track::kC1Pt2max;
  }
}

// parameters[4] is signed q/pT for both coordinate conventions.
GPUhdi() float ptFromQOverPt(float q2pt, uint8_t absCharge) noexcept
{
  float ptInv = std::abs(q2pt);
  if (ptInv < o2::track::MinPTInv) {
    ptInv = o2::track::MinPTInv;
  }
  if (absCharge > 1) {
    ptInv /= static_cast<float>(absCharge);
  }
  return 1.f / ptInv;
}

// Refit inward, outward, then optionally inward again; commit on success.
inline bool fitTrackSeedLegs(
  const TrackSeed& seed,
  const TimeFrame& frame,
  gsl::span<const gsl::span<const GlobalMeasurement>> layerGlobals,
  SurfaceCatalogView surfaceCatalog,
  float bz,
  bool shiftReferenceToMeasurement,
  float maxChi2ClusterAttachment,
  float maxChi2NDF,
  bool repeatRefitOut,
  gsl::span<const float> minPt,
  SurfaceTrackState& outParamIn,
  SurfaceTrackState& outParamOut,
  float& outChi2,
  OperationFailureReason& reason) noexcept
{
  if (layerGlobals.empty() || layerGlobals.size() > MaxLayoutSurfaces) {
    reason = OperationFailureReason::InvalidSurfaceCatalogAssociation;
    return false;
  }
  // Legs run sequentially; reuse bounded storage without allocating inside
  // this noexcept refit. Only the active portion is exposed to the assembler.
  std::array<detail::RefitMeasurementSlot, MaxLayoutSurfaces> slotsBuffer{};
  const gsl::span<detail::RefitMeasurementSlot> activeSlots{slotsBuffer.data(), layerGlobals.size()};
  auto legAcceptable = [](const SurfaceTrackState& state, float chi2, uint32_t acceptedHitCount,
                          float maxQoverPt, float maxChi2NDFValue) noexcept -> bool {
    if (!(std::abs(state.parameters[4]) < maxQoverPt)) {
      return false;
    }
    return chi2 < maxChi2NDFValue * static_cast<float>(static_cast<int>(acceptedHitCount) * 2 - 5);
  };

  // Leg A: inward.
  SurfaceTrackState stateA = seed.state();
  SurfaceTrackParameters linRefA{stateA};
  resetCovarianceForRefit(stateA);
  float chi2A = 0.f;
  uint32_t acceptedA = 0;
  const int activeSurfaceCount = static_cast<int>(layerGlobals.size());
  bool validSlots = false;
  const auto slotsA = detail::assembleRefitLegSlots(seed, frame, layerGlobals, 0, activeSurfaceCount, 1, activeSlots, validSlots);
  if (!validSlots) {
    reason = OperationFailureReason::InvalidSurfaceCatalogAssociation;
    return false;
  }
  if (!detail::driveRefitLeg(stateA, linRefA, chi2A, acceptedA, slotsA, surfaceCatalog, bz,
                             material::MaterialTraversalDirection::AlongMomentum, shiftReferenceToMeasurement,
                             maxChi2ClusterAttachment, reason)) {
    return false;
  }
  if (!legAcceptable(stateA, chi2A, acceptedA, o2::constants::math::VeryBig, maxChi2NDF)) {
    reason = OperationFailureReason::LegAcceptanceFailure;
    return false;
  }

  // Leg B: outward; this is the reported inner result.
  SurfaceTrackState stateB = stateA;
  SurfaceTrackParameters linRefB{stateB};
  resetCovarianceForRefit(stateB);
  float chi2B = 0.f;
  uint32_t acceptedB = 0;
  const auto slotsB = detail::assembleRefitLegSlots(seed, frame, layerGlobals, activeSurfaceCount - 1, -1, -1, activeSlots, validSlots);
  if (!validSlots) {
    reason = OperationFailureReason::InvalidSurfaceCatalogAssociation;
    return false;
  }
  if (!detail::driveRefitLeg(stateB, linRefB, chi2B, acceptedB, slotsB, surfaceCatalog, bz,
                             material::MaterialTraversalDirection::OppositeMomentum, shiftReferenceToMeasurement,
                             maxChi2ClusterAttachment, reason)) {
    return false;
  }
  if (!legAcceptable(stateB, chi2B, acceptedB, 50.f, maxChi2NDF)) {
    reason = OperationFailureReason::LegAcceptanceFailure;
    return false;
  }

  // MinPt uses the seed's attached-cluster count.
  const int nClAttached = seed.getHitLayerMask().count();
  const int minPtSlot = activeSurfaceCount - nClAttached;
  if (minPtSlot >= 0 && minPtSlot < static_cast<int>(minPt.size())) {
    const float minPtThreshold = minPt[minPtSlot];
    if (minPtThreshold > 0.f && ptFromQOverPt(stateB.parameters[4], stateB.absCharge) < minPtThreshold) {
      reason = OperationFailureReason::MinPtFailure;
      return false;
    }
  }

  // Optional leg C: inward again.
  SurfaceTrackState stateOut = stateA;
  if (repeatRefitOut) {
    SurfaceTrackState stateC = stateB;
    SurfaceTrackParameters linRefC{stateC};
    resetCovarianceForRefit(stateC);
    float chi2C = 0.f;
    uint32_t acceptedC = 0;
    const auto slotsC = detail::assembleRefitLegSlots(seed, frame, layerGlobals, 0, activeSurfaceCount, 1, activeSlots, validSlots);
    if (!validSlots) {
      reason = OperationFailureReason::InvalidSurfaceCatalogAssociation;
      return false;
    }
    if (!detail::driveRefitLeg(stateC, linRefC, chi2C, acceptedC, slotsC, surfaceCatalog, bz,
                               material::MaterialTraversalDirection::AlongMomentum, shiftReferenceToMeasurement,
                               maxChi2ClusterAttachment, reason)) {
      return false;
    }
    if (!legAcceptable(stateC, chi2C, acceptedC, o2::constants::math::VeryBig, maxChi2NDF)) {
      reason = OperationFailureReason::LegAcceptanceFailure;
      return false;
    }
    stateOut = stateC;
  }

  outParamIn = stateB;
  outParamOut = stateOut;
  outChi2 = chi2B;
  return true;
}

} // namespace o2::itsmft::tracking

#endif // GPUCA_GPUCODE

#endif /* ALICEO2_ITSMFT_TRACKING_REFITDRIVER_H_ */
