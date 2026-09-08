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

#ifndef ALICEO2_ITSMFT_TRACKING_SURFACETRACKSTATE_H_
#define ALICEO2_ITSMFT_TRACKING_SURFACETRACKSTATE_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "GPUCommonDef.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"
#include "ReconstructionDataFormats/PID.h"

namespace o2::itsmft::tracking
{

// The interpretation of parameters and covariance is selected by kind:
// Barrel:  (Y, Z, Snp, Tgl, Q2Pt), referenceCoordinate is local X, alpha is frame angle.
// Forward: (X, Y, Phi, Tgl, Q2Pt), referenceCoordinate is global Z, alpha is unused (zero).

// Fitted surface state. The field order keeps the device-facing representation
// compact while the parameter-only linearization state remains independent.
struct SurfaceTrackState {
  float parameters[5]{};
  float covariance[15]{};
  float referenceCoordinate{0.f};
  float alpha{0.f};
  SurfaceKind kind{SurfaceKind::Undefined};
  uint8_t flags{0};
  uint8_t absCharge{0};
  o2::track::PID pid{o2::track::PID::Pion};

  GPUhdi() constexpr bool hasRecognizedKind() const noexcept { return isRecognizedSurfaceKind(kind); }
};

static_assert(std::is_standard_layout_v<SurfaceTrackState>);
static_assert(std::is_trivially_copyable_v<SurfaceTrackState>);
static_assert(sizeof(SurfaceTrackState) == 92);
static_assert(alignof(SurfaceTrackState) == 4);
static_assert(offsetof(SurfaceTrackState, parameters) == 0);
static_assert(offsetof(SurfaceTrackState, covariance) == 20);
static_assert(offsetof(SurfaceTrackState, referenceCoordinate) == 80);
static_assert(offsetof(SurfaceTrackState, alpha) == 84);
static_assert(offsetof(SurfaceTrackState, kind) == 88);
static_assert(offsetof(SurfaceTrackState, flags) == 89);
static_assert(offsetof(SurfaceTrackState, absCharge) == 90);
static_assert(offsetof(SurfaceTrackState, pid) == 91);

// Covariance-free surface parameters used as the propagation linearization
// point paired with one SurfaceTrackState.
struct SurfaceTrackParameters {
  float parameters[5]{};
  float referenceCoordinate{0.f};
  float alpha{0.f};
  SurfaceKind kind{SurfaceKind::Undefined};

  GPUhdi() constexpr SurfaceTrackParameters() noexcept = default;
  GPUhdi() constexpr explicit SurfaceTrackParameters(const SurfaceTrackState& state) noexcept
    : referenceCoordinate{state.referenceCoordinate}, alpha{state.alpha}, kind{state.kind}
  {
    for (uint8_t i = 0; i < 5; ++i) {
      parameters[i] = state.parameters[i];
    }
  }

  GPUhdi() constexpr bool hasRecognizedKind() const noexcept { return isRecognizedSurfaceKind(kind); }
};

static_assert(std::is_standard_layout_v<SurfaceTrackParameters>);
static_assert(std::is_trivially_copyable_v<SurfaceTrackParameters>);
static_assert(sizeof(SurfaceTrackParameters) == 32);
static_assert(alignof(SurfaceTrackParameters) == 4);
static_assert(offsetof(SurfaceTrackParameters, parameters) == 0);
static_assert(offsetof(SurfaceTrackParameters, referenceCoordinate) == 20);
static_assert(offsetof(SurfaceTrackParameters, alpha) == 24);
static_assert(offsetof(SurfaceTrackParameters, kind) == 28);

GPUhdi() constexpr uint8_t packedCovarianceIndex(uint8_t row, uint8_t column) noexcept
{
  return row >= column ? row * (row + 1) / 2 + column : column * (column + 1) / 2 + row;
}

// Sanitize a packed covariance after a successful mutation. Diagonal values
// are made non-negative and capped, with corresponding row/column rescaling;
// off-diagonals are then limited to their pairwise Cauchy-Schwarz bounds.
GPUhdi() void sanitizeCovariance(SurfaceTrackState& state, const float (&maxDiagonal)[5]) noexcept
{
  auto& c = state.covariance;
  for (uint8_t i = 0; i < 5; ++i) {
    const uint8_t diagIndex = packedCovarianceIndex(i, i);
    c[diagIndex] = c[diagIndex] < 0.f ? -c[diagIndex] : c[diagIndex];
    if (c[diagIndex] > maxDiagonal[i]) {
      const float scale = std::sqrt(maxDiagonal[i] / c[diagIndex]);
      c[diagIndex] = maxDiagonal[i];
      for (uint8_t j = 0; j < 5; ++j) {
        if (j != i) {
          c[packedCovarianceIndex(i, j)] *= scale;
        }
      }
    }
  }
  for (uint8_t i = 0; i < 5; ++i) {
    for (uint8_t j = 0; j < i; ++j) {
      const float bound = std::sqrt(c[packedCovarianceIndex(i, i)] * c[packedCovarianceIndex(j, j)]);
      const uint8_t offIndex = packedCovarianceIndex(i, j);
      if (c[offIndex] > bound) {
        c[offIndex] = bound;
      } else if (c[offIndex] < -bound) {
        c[offIndex] = -bound;
      }
    }
  }
}

} // namespace o2::itsmft::tracking

#endif // ALICEO2_ITSMFT_TRACKING_SURFACETRACKSTATE_H_
