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

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <gsl/span>

#include "CommonConstants/MathConstants.h"
#include "ITSMFTTracking/TrackSeed.h"
#include "ITSMFTTracking/GlobalMeasurement.h"
#include "ITSMFTTracking/TimeFrame.h"
#include "ITSMFTTracking/Propagator.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"
#include "ReconstructionDataFormats/TrackParametrization.h"

// Descriptor-driven refit built on Propagator operations.
namespace o2::itsmft::tracking
{

namespace detail
{

constexpr float MinCircleFitBz = 0.01f; // kG

struct CircleFitPoint {
  double x, y;
  double xx, xy, yy;
};

// Fit y = a + b*x + c*(x*x + y*y) after translating, rotating and scaling
// the attached hits. Iteratively project their xy covariance onto the circle
// normal. Double precision is confined to this weak-bending seed estimate.
inline double estimateCircleQOverPt(gsl::span<const CircleFitPoint> points, double bz) noexcept
{
  constexpr double invalid = std::numeric_limits<double>::quiet_NaN();
  if (points.size() < 3 || !std::isfinite(bz) || std::abs(bz) < MinCircleFitBz) {
    return invalid;
  }
  const double x0 = points.front().x, y0 = points.front().y;
  const double dx = points.back().x - x0, dy = points.back().y - y0;
  const double length = std::hypot(dx, dy);
  if (!(length > 0.) || !std::isfinite(length)) {
    return invalid;
  }
  const double cs = dx / length, sn = dy / length;
  std::array<double, 3> fit{};
  for (int iteration = 0; iteration < 4; ++iteration) {
    double matrix[3][4]{};
    for (const auto& point : points) {
      const double x = ((point.x - x0) * cs + (point.y - y0) * sn) / length;
      const double y = (-(point.x - x0) * sn + (point.y - y0) * cs) / length;
      const double nx = -fit[1] - 2 * fit[2] * x, ny = 1 - 2 * fit[2] * y;
      const double gx = cs * nx - sn * ny, gy = sn * nx + cs * ny;
      const double variance = (gx * gx * point.xx + 2 * gx * gy * point.xy + gy * gy * point.yy) / (length * length);
      if (!(variance > 0.) || !std::isfinite(variance)) {
        return invalid;
      }
      const double weight = 1 / variance, basis[3] = {1, x, x * x + y * y};
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          matrix[i][j] += weight * basis[i] * basis[j];
        }
        matrix[i][3] += weight * basis[i] * y;
      }
    }
    // Solve the three normal equations with partial pivoting.
    for (int i = 0; i < 3; ++i) {
      int pivot = i;
      for (int j = i + 1; j < 3; ++j) {
        if (std::abs(matrix[j][i]) > std::abs(matrix[pivot][i])) {
          pivot = j;
        }
      }
      for (int k = i; k < 4; ++k) {
        std::swap(matrix[i][k], matrix[pivot][k]);
      }
      const double diagonal = matrix[i][i];
      if (std::abs(diagonal) < 1.e-15) {
        return invalid;
      }
      for (int k = i; k < 4; ++k) {
        matrix[i][k] /= diagonal;
      }
      for (int j = 0; j < 3; ++j) {
        if (j != i) {
          const double factor = matrix[j][i];
          for (int k = i; k < 4; ++k) {
            matrix[j][k] -= factor * matrix[i][k];
          }
        }
      }
    }
    for (int i = 0; i < 3; ++i) {
      fit[i] = matrix[i][3];
    }
  }
  const double discriminant = 1 + fit[1] * fit[1] - 4 * fit[0] * fit[2];
  return discriminant > 0 ? 2 * fit[2] / (length * std::sqrt(discriminant) * bz * o2::constants::math::B2C) : invalid;
}

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
                          bool shiftReferenceToMeasurement, float maxChi2) noexcept
{
  if (chi2 < 0.f) {
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
      return false;
    }
    const SurfaceDescriptor& descriptor = surfaceCatalog.getSurface(slot.surface);
    if (!Propagator::propagateToMeasurement(scratchState, scratchLinRef, descriptor, slot.measurement, bz, direction,
                                            scratchAcceptedHitCount >= kChi2GateMinAcceptedHits, maxChi2, scratchChi2,
                                            shiftReferenceToMeasurement)) {
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

// Common first-pass prior for the two position coordinates, direction and q/pT.
GPUhdi() void resetCovarianceForRefit(SurfaceTrackState& state) noexcept
{
  for (auto& element : state.covariance) {
    element = 0.f;
  }
  for (int i = 0; i < 4; ++i) {
    state.covariance[packedCovarianceIndex(i, i)] = 1.f;
  }
  // This is the variance, not the standard deviation.
  state.covariance[packedCovarianceIndex(4, 4)] = std::clamp(std::abs(state.parameters[4]), 1.f, 10.f);
}

// Start a subsequent leg with five times the previous parameter uncertainties.
GPUhdi() void inflateDiagonalCovarianceForRefit(SurfaceTrackState& state) noexcept
{
  constexpr float varianceInflation = 25.f;
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < i; ++j) {
      state.covariance[packedCovarianceIndex(i, j)] = 0.f;
    }
    state.covariance[packedCovarianceIndex(i, i)] *= varianceInflation;
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
  float& outChi2) noexcept
{
  if (layerGlobals.empty() || layerGlobals.size() > MaxLayoutSurfaces) {
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
  if (!std::isfinite(bz)) {
    return false;
  }
  // There is no curvature constraint with the field off; keep the CA seed.
  if (std::abs(bz) >= detail::MinCircleFitBz) {
    std::array<detail::CircleFitPoint, MaxLayoutSurfaces> points{};
    std::size_t nPoints = 0;
    for (int layer = 0; layer < static_cast<int>(layerGlobals.size()); ++layer) {
      const int cluster = seed.getCluster(layer);
      if (cluster == o2::its::constants::UnusedIndex) {
        continue;
      }
      if (cluster < 0 || static_cast<std::size_t>(cluster) >= layerGlobals[layer].size()) {
        return false;
      }
      const auto& global = layerGlobals[layer][cluster];
      points[nPoints++] = {global.x, global.y, global.covariance.xx, global.covariance.xy, global.covariance.yy};
    }
    const float qOverPt = detail::estimateCircleQOverPt({points.data(), nPoints}, bz);
    if (!std::isfinite(qOverPt)) {
      return false;
    }
    stateA.parameters[4] = qOverPt;
  }
  SurfaceTrackParameters linRefA{stateA};
  resetCovarianceForRefit(stateA);
  float chi2A = 0.f;
  uint32_t acceptedA = 0;
  const int activeSurfaceCount = static_cast<int>(layerGlobals.size());
  bool validSlots = false;
  const auto slotsA = detail::assembleRefitLegSlots(seed, frame, layerGlobals, 0, activeSurfaceCount, 1, activeSlots, validSlots);
  if (!validSlots) {
    return false;
  }
  if (!detail::driveRefitLeg(stateA, linRefA, chi2A, acceptedA, slotsA, surfaceCatalog, bz,
                             material::MaterialTraversalDirection::AlongMomentum, shiftReferenceToMeasurement,
                             maxChi2ClusterAttachment)) {
    return false;
  }
  if (!legAcceptable(stateA, chi2A, acceptedA, o2::constants::math::VeryBig, maxChi2NDF)) {
    return false;
  }

  // Leg B: outward; this is the reported inner result.
  SurfaceTrackState stateB = stateA;
  SurfaceTrackParameters linRefB{stateB};
  inflateDiagonalCovarianceForRefit(stateB);
  float chi2B = 0.f;
  uint32_t acceptedB = 0;
  const auto slotsB = detail::assembleRefitLegSlots(seed, frame, layerGlobals, activeSurfaceCount - 1, -1, -1, activeSlots, validSlots);
  if (!validSlots) {
    return false;
  }
  if (!detail::driveRefitLeg(stateB, linRefB, chi2B, acceptedB, slotsB, surfaceCatalog, bz,
                             material::MaterialTraversalDirection::OppositeMomentum, shiftReferenceToMeasurement,
                             maxChi2ClusterAttachment)) {
    return false;
  }
  if (!legAcceptable(stateB, chi2B, acceptedB, 50.f, maxChi2NDF)) {
    return false;
  }

  // MinPt uses the seed's attached-cluster count.
  const int nClAttached = seed.getHitLayerMask().count();
  const int minPtSlot = activeSurfaceCount - nClAttached;
  if (minPtSlot >= 0 && minPtSlot < static_cast<int>(minPt.size())) {
    const float minPtThreshold = minPt[minPtSlot];
    if (minPtThreshold > 0.f && ptFromQOverPt(stateB.parameters[4], stateB.absCharge) < minPtThreshold) {
      return false;
    }
  }

  // Optional leg C: inward again.
  SurfaceTrackState stateOut = stateA;
  if (repeatRefitOut) {
    SurfaceTrackState stateC = stateB;
    SurfaceTrackParameters linRefC{stateC};
    inflateDiagonalCovarianceForRefit(stateC);
    float chi2C = 0.f;
    uint32_t acceptedC = 0;
    const auto slotsC = detail::assembleRefitLegSlots(seed, frame, layerGlobals, 0, activeSurfaceCount, 1, activeSlots, validSlots);
    if (!validSlots) {
      return false;
    }
    if (!detail::driveRefitLeg(stateC, linRefC, chi2C, acceptedC, slotsC, surfaceCatalog, bz,
                               material::MaterialTraversalDirection::AlongMomentum, shiftReferenceToMeasurement,
                               maxChi2ClusterAttachment)) {
      return false;
    }
    if (!legAcceptable(stateC, chi2C, acceptedC, o2::constants::math::VeryBig, maxChi2NDF)) {
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
