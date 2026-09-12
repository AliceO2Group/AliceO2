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

#include "ITSMFTTracking/detail/SurfaceStateOperations.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "CommonConstants/MathConstants.h"
#include "GPUROOTSMatrixFwd.h"
#include <Math/SMatrix.h>

// Provides covariance/curvature constants for device-visible operations;
// no track object is constructed here.
#include "ReconstructionDataFormats/TrackParametrization.h"

namespace o2::itsmft::tracking::detail::barrel
{
namespace
{

using DenseMatrix5 = float[5][5];

// Packed symmetric 5x5 covariance for stateChi2. MatRepSym::offset() matches
// packedCovarianceIndex exactly, so the combined covariance is built directly
// in packed storage.
using CombinedCovariance = o2::math_utils::SMatrix<float, 5, 5, o2::math_utils::MatRepSym<float, 5>>;
static_assert(o2::math_utils::MatRepSym<float, 5>::kSize == 15, "packed symmetric 5x5 representation must hold exactly 15 floats");
static_assert(sizeof(CombinedCovariance) == 15 * sizeof(float), "combined covariance must occupy exactly 15 floats");

// sanitizeCovariance() upper bounds in (Y, Z, Snp, Tgl, Q2Pt) order. These
// match the barrel limits used by FamilyMaterialOperations.
constexpr float kBarrelMaxDiagonal[5] = {o2::track::kCY2max, o2::track::kCZ2max, o2::track::kCSnp2max,
                                         o2::track::kCTgl2max, o2::track::kC1Pt2max};

bool validateSource(const SurfaceTrackState& state, OperationFailureReason& reason) noexcept
{
  if (state.kind != SurfaceKind::Cylinder) {
    reason = OperationFailureReason::SourceSurfaceKindMismatch;
    return false;
  }
  return true;
}

void unpackCovariance(const SurfaceTrackState& state, DenseMatrix5& covariance) noexcept
{
  for (uint8_t row = 0; row < 5; ++row) {
    for (uint8_t column = 0; column < 5; ++column) {
      covariance[row][column] = state.covariance[packedCovarianceIndex(row, column)];
    }
  }
}

void packCovariance(const DenseMatrix5& covariance, SurfaceTrackState& state) noexcept
{
  for (uint8_t row = 0; row < 5; ++row) {
    for (uint8_t column = 0; column <= row; ++column) {
      state.covariance[packedCovarianceIndex(row, column)] = covariance[row][column];
    }
  }
}

void identity(DenseMatrix5& matrix) noexcept
{
  for (uint8_t i = 0; i < 5; ++i) {
    matrix[i][i] = 1.f;
  }
}

void transportCovariance(SurfaceTrackState& state, const DenseMatrix5& jacobian) noexcept
{
  DenseMatrix5 covariance{};
  DenseMatrix5 product{};
  DenseMatrix5 transported{};
  unpackCovariance(state, covariance);
  for (uint8_t row = 0; row < 5; ++row) {
    for (uint8_t column = 0; column < 5; ++column) {
      for (uint8_t inner = 0; inner < 5; ++inner) {
        product[row][column] += jacobian[row][inner] * covariance[inner][column];
      }
    }
  }
  for (uint8_t row = 0; row < 5; ++row) {
    for (uint8_t column = 0; column < 5; ++column) {
      for (uint8_t inner = 0; inner < 5; ++inner) {
        transported[row][column] += product[row][inner] * jacobian[column][inner];
      }
    }
  }
  packCovariance(transported, state);
}

// Shared commit point for non-linRef rotate() and propagate(). It validates
// and sanitizes the covariance (ADR 0008) on every exit, including dx == 0.
bool commit(SurfaceTrackState& destination, SurfaceTrackState& scratch) noexcept
{
  sanitizeCovariance(scratch, kBarrelMaxDiagonal);
  destination = scratch;
  return true;
}

bool residualInverse(const SurfaceTrackState& state, const SurfaceMeasurement& measurement,
                     float& inverse00, float& inverse01, float& inverse11,
                     OperationFailureReason& reason) noexcept
{
  const float s00 = state.covariance[packedCovarianceIndex(0, 0)] + measurement.covariance.uu;
  const float s01 = state.covariance[packedCovarianceIndex(1, 0)] + measurement.covariance.uv;
  const float s11 = state.covariance[packedCovarianceIndex(1, 1)] + measurement.covariance.vv;
  const float determinant = s00 * s11 - s01 * s01;
  if (determinant == 0.f) {
    reason = OperationFailureReason::InvalidCovariance;
    return false;
  }
  const float inverseDeterminant = 1.f / determinant;
  inverse00 = s11 * inverseDeterminant;
  inverse01 = -s01 * inverseDeterminant;
  inverse11 = s00 * inverseDeterminant;
  return true;
}

} // namespace

bool rotate(SurfaceTrackState& state, float targetAlpha, OperationFailureReason& reason) noexcept
{
  if (!validateSource(state, reason)) {
    return false;
  }
  SurfaceTrackState scratch = state;
  const float canonicalTargetAlpha = std::remainder(targetAlpha, 2.f * o2::constants::math::PI);
  const float delta = std::remainder(canonicalTargetAlpha - scratch.alpha, 2.f * o2::constants::math::PI);
  const float sine = std::sin(delta);
  const float cosine = std::cos(delta);
  const float snp = scratch.parameters[2];
  if (std::abs(snp) >= 1.f) {
    reason = OperationFailureReason::RotationFailure;
    return false;
  }
  const float csp = std::sqrt((1.f - snp) * (1.f + snp));
  const float rotatedCosine = csp * cosine + snp * sine;
  const float rotatedSnp = snp * cosine - csp * sine;
  if (rotatedCosine < 0.f || std::abs(rotatedSnp) >= 1.f || csp == 0.f) {
    reason = OperationFailureReason::RotationFailure;
    return false;
  }
  const float x = scratch.referenceCoordinate;
  const float y = scratch.parameters[0];
  scratch.referenceCoordinate = x * cosine + y * sine;
  scratch.parameters[0] = -x * sine + y * cosine;
  scratch.parameters[2] = rotatedSnp;
  scratch.alpha = canonicalTargetAlpha;
  const float ratio = cosine + snp / csp * sine;
  scratch.covariance[packedCovarianceIndex(0, 0)] *= cosine * cosine;
  scratch.covariance[packedCovarianceIndex(1, 0)] *= cosine;
  scratch.covariance[packedCovarianceIndex(2, 0)] *= cosine * ratio;
  scratch.covariance[packedCovarianceIndex(2, 1)] *= ratio;
  scratch.covariance[packedCovarianceIndex(2, 2)] *= ratio * ratio;
  scratch.covariance[packedCovarianceIndex(3, 0)] *= cosine;
  scratch.covariance[packedCovarianceIndex(3, 2)] *= ratio;
  scratch.covariance[packedCovarianceIndex(4, 0)] *= cosine;
  scratch.covariance[packedCovarianceIndex(4, 2)] *= ratio;
  return commit(state, scratch);
}

bool propagate(SurfaceTrackState& state, float targetX, float bz, OperationFailureReason& reason) noexcept
{
  if (!validateSource(state, reason)) {
    return false;
  }
  SurfaceTrackState scratch = state;
  const float dx = targetX - scratch.referenceCoordinate;
  if (dx == 0.f) {
    scratch.referenceCoordinate = targetX;
    return commit(state, scratch);
  }
  const float snp = scratch.parameters[2];
  const float curvature = scratch.absCharge == 0 ? 0.f : scratch.parameters[4] * bz * o2::constants::math::B2C;
  const float propagatedSnp = snp + curvature * dx;
  if (std::abs(snp) >= 1.f || std::abs(propagatedSnp) >= 1.f) {
    reason = OperationFailureReason::UnreachableTarget;
    return false;
  }
  const float csp = std::sqrt((1.f - snp) * (1.f + snp));
  const float propagatedCsp = std::sqrt((1.f - propagatedSnp) * (1.f + propagatedSnp));
  if (csp == 0.f || propagatedCsp == 0.f) {
    reason = OperationFailureReason::UnreachableTarget;
    return false;
  }
  const float reciprocalCosines = 1.f / (csp + propagatedCsp);
  const float dyOverDx = (snp + propagatedSnp) * reciprocalCosines;
  const float x2r = curvature * dx;
  const bool arcZ = std::abs(x2r) > 0.05f;
  float dz = 0.f;
  if (arcZ) {
    const float argument = csp * propagatedSnp - propagatedCsp * snp;
    if (std::abs(argument) > 1.f || curvature == 0.f) {
      reason = OperationFailureReason::PropagationFailure;
      return false;
    }
    float angle = std::asin(argument);
    if (snp * snp + propagatedSnp * propagatedSnp > 1.f && snp * propagatedSnp < 0.f) {
      angle = propagatedSnp > 0.f ? o2::constants::math::PI - angle : -o2::constants::math::PI - angle;
    }
    dz = scratch.parameters[3] / curvature * angle;
  } else {
    dz = dx * (propagatedCsp + propagatedSnp * dyOverDx) * scratch.parameters[3];
  }
  scratch.referenceCoordinate = targetX;
  scratch.parameters[0] += dx * dyOverDx;
  scratch.parameters[1] += dz;
  scratch.parameters[2] = propagatedSnp;

  const float propagatedCspInverse = 1.f / propagatedCsp;
  const float dxOverCosines = dx * reciprocalCosines;
  const float hh = dxOverCosines * propagatedCspInverse * (1.f + csp * propagatedCsp + snp * propagatedSnp);
  const float jj = dx * (dyOverDx - propagatedSnp * propagatedCspInverse);
  DenseMatrix5 jacobian{};
  identity(jacobian);
  jacobian[0][2] = hh / csp;
  jacobian[0][4] = hh * dxOverCosines * bz * o2::constants::math::B2C;
  jacobian[1][2] = scratch.parameters[3] * (jacobian[0][2] * propagatedSnp + jj);
  jacobian[1][3] = dx * (propagatedCsp + propagatedSnp * dyOverDx);
  jacobian[1][4] = scratch.parameters[3] * (jacobian[0][4] * propagatedSnp + jj * dx * bz * o2::constants::math::B2C);
  jacobian[2][4] = dx * bz * o2::constants::math::B2C;
  transportCovariance(scratch, jacobian);
  return commit(state, scratch);
}

bool predictedChi2(const SurfaceTrackState& state, const SurfaceMeasurement& measurement, float& chi2,
                   OperationFailureReason& reason) noexcept
{
  if (!validateSource(state, reason)) {
    return false;
  }
  float inverse00 = 0.f;
  float inverse01 = 0.f;
  float inverse11 = 0.f;
  if (!residualInverse(state, measurement, inverse00, inverse01, inverse11, reason)) {
    return false;
  }
  const float residualY = measurement.frame.u - state.parameters[0];
  const float residualZ = measurement.frame.v - state.parameters[1];
  const float scratchChi2 = residualY * (inverse00 * residualY + inverse01 * residualZ) +
                            residualZ * (inverse01 * residualY + inverse11 * residualZ);
  chi2 = scratchChi2;
  return true;
}

bool update(SurfaceTrackState& state, const SurfaceMeasurement& measurement, float& chi2,
            OperationFailureReason& reason) noexcept
{
  if (!validateSource(state, reason)) {
    return false;
  }
  float inverse00 = 0.f;
  float inverse01 = 0.f;
  float inverse11 = 0.f;
  if (!residualInverse(state, measurement, inverse00, inverse01, inverse11, reason)) {
    return false;
  }
  DenseMatrix5 covariance{};
  DenseMatrix5 josephTransform{};
  DenseMatrix5 transformedCovariance{};
  DenseMatrix5 updatedCovariance{};
  float gain[5][2]{};
  unpackCovariance(state, covariance);
  const float residual[2] = {measurement.frame.u - state.parameters[0], measurement.frame.v - state.parameters[1]};
  SurfaceTrackState scratch = state;
  for (uint8_t row = 0; row < 5; ++row) {
    gain[row][0] = covariance[row][0] * inverse00 + covariance[row][1] * inverse01;
    gain[row][1] = covariance[row][0] * inverse01 + covariance[row][1] * inverse11;
    scratch.parameters[row] += gain[row][0] * residual[0] + gain[row][1] * residual[1];
  }

  // Joseph covariance update: (I - K H) P (I - K H)^T + K R K^T.
  // The surface measurement matrix H selects state parameters 0 and 1.
  identity(josephTransform);
  for (uint8_t row = 0; row < 5; ++row) {
    josephTransform[row][0] -= gain[row][0];
    josephTransform[row][1] -= gain[row][1];
  }
  for (uint8_t row = 0; row < 5; ++row) {
    for (uint8_t column = 0; column < 5; ++column) {
      for (uint8_t inner = 0; inner < 5; ++inner) {
        transformedCovariance[row][column] += josephTransform[row][inner] * covariance[inner][column];
      }
    }
  }
  for (uint8_t row = 0; row < 5; ++row) {
    for (uint8_t column = 0; column < 5; ++column) {
      for (uint8_t inner = 0; inner < 5; ++inner) {
        updatedCovariance[row][column] += transformedCovariance[row][inner] * josephTransform[column][inner];
      }
      updatedCovariance[row][column] +=
        gain[row][0] * (measurement.covariance.uu * gain[column][0] + measurement.covariance.uv * gain[column][1]) +
        gain[row][1] * (measurement.covariance.uv * gain[column][0] + measurement.covariance.vv * gain[column][1]);
    }
  }
  for (uint8_t row = 0; row < 5; ++row) {
    for (uint8_t column = 0; column < row; ++column) {
      const float symmetric = 0.5f * (updatedCovariance[row][column] + updatedCovariance[column][row]);
      updatedCovariance[row][column] = symmetric;
      updatedCovariance[column][row] = symmetric;
    }
  }
  packCovariance(updatedCovariance, scratch);
  const float scratchChi2 = residual[0] * (inverse00 * residual[0] + inverse01 * residual[1]) +
                            residual[1] * (inverse01 * residual[0] + inverse11 * residual[1]);
  // Preserve the established covariance bounds after the Joseph update.
  sanitizeCovariance(scratch, kBarrelMaxDiagonal);
  state = scratch;
  chi2 = scratchChi2;
  return true;
}

bool stateChi2(const SurfaceTrackState& reference, const SurfaceTrackState& candidate, float& chi2,
               OperationFailureReason& reason) noexcept
{
  if (reference.kind != SurfaceKind::Cylinder || candidate.kind != SurfaceKind::Cylinder) {
    reason = OperationFailureReason::SourceSurfaceKindMismatch;
    return false;
  }
  if (std::abs(reference.alpha - candidate.alpha) > o2::constants::math::Epsilon) {
    reason = OperationFailureReason::AlphaMismatch;
    return false;
  }
  if (std::abs(reference.referenceCoordinate - candidate.referenceCoordinate) > o2::constants::math::Epsilon) {
    reason = OperationFailureReason::ReferenceCoordinateMismatch;
    return false;
  }

  CombinedCovariance combined;
  float* packed = combined.Array();
  for (uint8_t i = 0; i < 15; ++i) {
    packed[i] = reference.covariance[i] + candidate.covariance[i];
  }
  if (!combined.Invert()) {
    reason = OperationFailureReason::InvalidCovariance;
    return false;
  }

  float diff[5];
  for (uint8_t i = 0; i < 5; ++i) {
    diff[i] = reference.parameters[i] - candidate.parameters[i];
  }
  float chi2diag = 0.f;
  float chi2ndiag = 0.f;
  for (uint8_t i = 0; i < 5; ++i) {
    chi2diag += diff[i] * diff[i] * packed[packedCovarianceIndex(i, i)];
    for (uint8_t j = 0; j < i; ++j) {
      chi2ndiag += diff[i] * diff[j] * packed[packedCovarianceIndex(i, j)];
    }
  }
  const float scratchChi2 = chi2diag + 2.f * chi2ndiag;
  chi2 = scratchChi2;
  return true;
}

#ifndef GPUCA_GPUCODE

namespace
{

// Covariance-free propagation of SurfaceTrackParameters using the
// TrackParametrization::propagateParamTo formula. stateAbsCharge supplies the
// charge absent from SurfaceTrackParameters; it matches the paired
// state's particle hypothesis.
bool propagateReferenceParams(SurfaceTrackParameters& ref, uint8_t stateAbsCharge, float targetX, float bz,
                              OperationFailureReason& reason) noexcept
{
  const float dx = targetX - ref.referenceCoordinate;
  if (dx == 0.f) {
    ref.referenceCoordinate = targetX;
    return true;
  }
  const float snp = ref.parameters[2];
  const float curvature = stateAbsCharge == 0 ? 0.f : ref.parameters[4] * bz * o2::constants::math::B2C;
  const float propagatedSnp = snp + curvature * dx;
  if (std::abs(snp) >= 1.f || std::abs(propagatedSnp) >= 1.f) {
    reason = OperationFailureReason::UnreachableTarget;
    return false;
  }
  const float csp = std::sqrt((1.f - snp) * (1.f + snp));
  const float propagatedCsp = std::sqrt((1.f - propagatedSnp) * (1.f + propagatedSnp));
  if (csp == 0.f || propagatedCsp == 0.f) {
    reason = OperationFailureReason::UnreachableTarget;
    return false;
  }
  const float reciprocalCosines = 1.f / (csp + propagatedCsp);
  const float dyOverDx = (snp + propagatedSnp) * reciprocalCosines;
  const float x2r = curvature * dx;
  const bool arcZ = std::abs(x2r) > 0.05f;
  float dz = 0.f;
  if (arcZ) {
    const float argument = csp * propagatedSnp - propagatedCsp * snp;
    if (std::abs(argument) > 1.f || curvature == 0.f) {
      reason = OperationFailureReason::PropagationFailure;
      return false;
    }
    float angle = std::asin(argument);
    if (snp * snp + propagatedSnp * propagatedSnp > 1.f && snp * propagatedSnp < 0.f) {
      angle = propagatedSnp > 0.f ? o2::constants::math::PI - angle : -o2::constants::math::PI - angle;
    }
    dz = ref.parameters[3] / curvature * angle;
  } else {
    dz = dx * (propagatedCsp + propagatedSnp * dyOverDx) * ref.parameters[3];
  }
  ref.referenceCoordinate = targetX;
  ref.parameters[0] += dx * dyOverDx;
  ref.parameters[1] += dz;
  ref.parameters[2] = propagatedSnp;
  return true;
}

} // namespace

bool rotate(SurfaceTrackState& state, SurfaceTrackParameters& linRef, float targetAlpha, float bz,
            OperationFailureReason& reason) noexcept
{
  if (!validateSource(state, reason)) {
    return false;
  }
  if (linRef.kind != SurfaceKind::Cylinder) {
    reason = OperationFailureReason::SourceSurfaceKindMismatch;
    return false;
  }
  // Pairing requires exact referenceCoordinate/alpha equality. Parameters may
  // differ because linRef is a linearization reference.
  if (state.referenceCoordinate != linRef.referenceCoordinate) {
    reason = OperationFailureReason::ReferenceCoordinateMismatch;
    return false;
  }
  if (state.alpha != linRef.alpha) {
    reason = OperationFailureReason::AlphaMismatch;
    return false;
  }
  const float stateSnp = state.parameters[2];
  if (std::abs(stateSnp) >= 1.f) {
    reason = OperationFailureReason::RotationFailure;
    return false;
  }

  SurfaceTrackState scratchState = state;
  SurfaceTrackParameters scratchRef = linRef;

  const float canonicalAlpha = std::remainder(targetAlpha, 2.f * o2::constants::math::PI);

  // Rotate the reference using its own pre-rotation snp.
  const float refSnpBefore = scratchRef.parameters[2];
  if (std::abs(refSnpBefore) >= 1.f) {
    reason = OperationFailureReason::RotationFailure;
    return false;
  }
  const float delta = std::remainder(canonicalAlpha - scratchRef.alpha, 2.f * o2::constants::math::PI);
  const float sa = std::sin(delta);
  const float ca = std::cos(delta);
  const float refCsp0 = std::sqrt((1.f - refSnpBefore) * (1.f + refSnpBefore));
  if (refCsp0 * ca + refSnpBefore * sa < 0.f) {
    reason = OperationFailureReason::RotationFailure;
    return false;
  }
  const float refSnpRotated = refSnpBefore * ca - refCsp0 * sa;
  if (std::abs(refSnpRotated) >= 1.f) {
    reason = OperationFailureReason::RotationFailure;
    return false;
  }
  const float refXOld = scratchRef.referenceCoordinate;
  const float refYOld = scratchRef.parameters[0];
  scratchRef.alpha = canonicalAlpha;
  scratchRef.referenceCoordinate = refXOld * ca + refYOld * sa;
  scratchRef.parameters[0] = -refXOld * sa + refYOld * ca;
  scratchRef.parameters[2] = refSnpRotated;

  // Rotate the state's pre-rotation X,Y by the reference delta.
  const float trackX = scratchState.referenceCoordinate * ca + scratchState.parameters[0] * sa;

  if (!propagateReferenceParams(scratchRef, state.absCharge, trackX, bz, reason)) {
    reason = OperationFailureReason::RotationFailure;
    return false;
  }

  // Rotate the state using its own snp and post-rotation validity.
  const float csp = std::sqrt((1.f - stateSnp) * (1.f + stateSnp));
  if (csp * ca + stateSnp * sa < 0.f) {
    reason = OperationFailureReason::RotationFailure;
    return false;
  }
  const float updatedSnp = stateSnp * ca - csp * sa;
  if (std::abs(updatedSnp) >= 1.f) {
    reason = OperationFailureReason::RotationFailure;
    return false;
  }
  const float stateXOld = scratchState.referenceCoordinate;
  const float stateYOld = scratchState.parameters[0];
  scratchState.parameters[0] = -stateXOld * sa + stateYOld * ca;
  scratchState.referenceCoordinate = trackX;
  scratchState.parameters[2] = updatedSnp;
  scratchState.alpha = canonicalAlpha;

  // Evaluate the covariance Jacobian at the reference, not the state's snp.
  // Compute cspRef1 algebraically to match the legacy formula.
  const float cspRef1 = ca * refCsp0 + sa * refSnpBefore;
  if (cspRef1 == 0.f) {
    reason = OperationFailureReason::RotationFailure;
    return false;
  }
  const float rr = cspRef1 / refCsp0;

  // Compute the extra lower-triangle row before the plane-rotation multiplies,
  // matching the legacy evaluation order.
  const float cXSigY = scratchState.covariance[packedCovarianceIndex(0, 0)] * ca * sa;
  const float cXSigZ = scratchState.covariance[packedCovarianceIndex(1, 0)] * sa;
  const float cXSigSnp = scratchState.covariance[packedCovarianceIndex(2, 0)] * rr * sa;
  const float cXSigTgl = scratchState.covariance[packedCovarianceIndex(3, 0)] * sa;
  const float cXSigQ2Pt = scratchState.covariance[packedCovarianceIndex(4, 0)] * sa;
  const float cSigX2 = scratchState.covariance[packedCovarianceIndex(0, 0)] * sa * sa;

  scratchState.covariance[packedCovarianceIndex(0, 0)] *= ca * ca;
  scratchState.covariance[packedCovarianceIndex(1, 0)] *= ca;
  scratchState.covariance[packedCovarianceIndex(2, 0)] *= ca * rr;
  scratchState.covariance[packedCovarianceIndex(2, 1)] *= rr;
  scratchState.covariance[packedCovarianceIndex(2, 2)] *= rr * rr;
  scratchState.covariance[packedCovarianceIndex(3, 0)] *= ca;
  scratchState.covariance[packedCovarianceIndex(3, 2)] *= rr;
  scratchState.covariance[packedCovarianceIndex(4, 0)] *= ca;
  scratchState.covariance[packedCovarianceIndex(4, 2)] *= rr;

  const float cspRef1Inv = 1.f / cspRef1;
  const float j3 = -refSnpRotated * cspRef1Inv;
  const float j4 = -scratchRef.parameters[3] * cspRef1Inv;
  const float j5 = state.absCharge != 0 ? scratchRef.parameters[4] * bz * o2::constants::math::B2C : 0.f;

  const float hXSigY = cXSigY + cSigX2 * j3;
  const float hXSigZ = cXSigZ + cSigX2 * j4;
  const float hXSigSnp = cXSigSnp + cSigX2 * j5;

  scratchState.covariance[packedCovarianceIndex(0, 0)] += j3 * (cXSigY + hXSigY);
  scratchState.covariance[packedCovarianceIndex(1, 1)] += j4 * (cXSigZ + hXSigZ);
  scratchState.covariance[packedCovarianceIndex(2, 0)] += cXSigSnp * j3 + hXSigY * j5;
  scratchState.covariance[packedCovarianceIndex(2, 2)] += j5 * (cXSigSnp + hXSigSnp);
  scratchState.covariance[packedCovarianceIndex(3, 1)] += cXSigTgl * j4;
  scratchState.covariance[packedCovarianceIndex(4, 0)] += cXSigQ2Pt * j3;
  scratchState.covariance[packedCovarianceIndex(4, 2)] += cXSigQ2Pt * j5;

  scratchState.covariance[packedCovarianceIndex(1, 0)] += cXSigZ * j3 + hXSigY * j4;
  scratchState.covariance[packedCovarianceIndex(2, 1)] += cXSigSnp * j4 + hXSigZ * j5;
  scratchState.covariance[packedCovarianceIndex(3, 0)] += cXSigTgl * j3;
  scratchState.covariance[packedCovarianceIndex(3, 2)] += cXSigTgl * j5;
  scratchState.covariance[packedCovarianceIndex(4, 1)] += cXSigQ2Pt * j4;

  sanitizeCovariance(scratchState, kBarrelMaxDiagonal);
  state = scratchState;
  linRef = scratchRef;
  return true;
}

bool propagate(SurfaceTrackState& state, SurfaceTrackParameters& linRef, float targetX, float bz,
               OperationFailureReason& reason) noexcept
{
  if (!validateSource(state, reason)) {
    return false;
  }
  if (linRef.kind != SurfaceKind::Cylinder) {
    reason = OperationFailureReason::SourceSurfaceKindMismatch;
    return false;
  }
  // Pairing requires exact referenceCoordinate/alpha equality; parameters may
  // differ.
  if (state.referenceCoordinate != linRef.referenceCoordinate) {
    reason = OperationFailureReason::ReferenceCoordinateMismatch;
    return false;
  }
  if (state.alpha != linRef.alpha) {
    reason = OperationFailureReason::AlphaMismatch;
    return false;
  }

  const float effectiveBz = state.absCharge == 0 ? 0.f : bz;
  const float dx = targetX - state.referenceCoordinate;
  if (std::abs(dx) < o2::constants::math::Almost0) {
    SurfaceTrackState scratchState = state;
    SurfaceTrackParameters scratchRef = linRef;
    scratchState.referenceCoordinate = targetX;
    scratchRef.referenceCoordinate = targetX;
    state = scratchState;
    linRef = scratchRef;
    return true;
  }

  SurfaceTrackParameters scratchRef = linRef;
  const float snpRef0 = scratchRef.parameters[2];
  const float cspRef0 = std::sqrt((1.f - snpRef0) * (1.f + snpRef0));
  const float tglRef0 = scratchRef.parameters[3];

  if (!propagateReferenceParams(scratchRef, state.absCharge, targetX, effectiveBz, reason)) {
    return false;
  }
  const float snpRef1 = scratchRef.parameters[2];
  const float cspRef1 = std::sqrt((1.f - snpRef1) * (1.f + snpRef1));
  if (cspRef0 == 0.f || cspRef1 == 0.f) {
    reason = OperationFailureReason::PropagationFailure;
    return false;
  }

  const float kb = effectiveBz * o2::constants::math::B2C;
  const float cspRef0Inv = 1.f / cspRef0;
  const float cspRef1Inv = 1.f / cspRef1;
  const float cc = cspRef0 + cspRef1;
  const float ccInv = 1.f / cc;
  const float dy2dx = (snpRef0 + snpRef1) * ccInv;
  const float dxccInv = dx * ccInv;
  const float hh = dxccInv * cspRef1Inv * (1.f + cspRef0 * cspRef1 + snpRef0 * snpRef1);
  const float jj = dx * (dy2dx - snpRef1 * cspRef1Inv);

  const float f02 = hh * cspRef0Inv;
  const float f04 = hh * dxccInv * kb;
  const float f24 = dx * kb;
  const float f12 = tglRef0 * (f02 * snpRef1 + jj);
  const float f13 = dx * (cspRef1 + snpRef1 * dy2dx);
  const float f14 = tglRef0 * (f04 * snpRef1 + jj * f24);

  float diff[5];
  for (uint8_t i = 0; i < 5; ++i) {
    diff[i] = state.parameters[i] - linRef.parameters[i];
  }
  const float snpUpd = snpRef1 + diff[2] + f24 * diff[4];
  if (std::abs(snpUpd) >= 1.f) {
    reason = OperationFailureReason::PropagationFailure;
    return false;
  }

  SurfaceTrackState scratchState = state;
  scratchState.referenceCoordinate = targetX;
  scratchState.parameters[0] = scratchRef.parameters[0] + diff[0] + f02 * diff[2] + f04 * diff[4];
  scratchState.parameters[1] = scratchRef.parameters[1] + diff[1] + f13 * diff[3] + f14 * diff[4];
  scratchState.parameters[2] = snpUpd;
  scratchState.parameters[3] = scratchRef.parameters[3] + diff[3];
  scratchState.parameters[4] = scratchRef.parameters[4] + diff[4];

  const float c00 = state.covariance[packedCovarianceIndex(0, 0)];
  const float c10 = state.covariance[packedCovarianceIndex(1, 0)];
  const float c11 = state.covariance[packedCovarianceIndex(1, 1)];
  const float c20 = state.covariance[packedCovarianceIndex(2, 0)];
  const float c21 = state.covariance[packedCovarianceIndex(2, 1)];
  const float c22 = state.covariance[packedCovarianceIndex(2, 2)];
  const float c30 = state.covariance[packedCovarianceIndex(3, 0)];
  const float c31 = state.covariance[packedCovarianceIndex(3, 1)];
  const float c32 = state.covariance[packedCovarianceIndex(3, 2)];
  const float c33 = state.covariance[packedCovarianceIndex(3, 3)];
  const float c40 = state.covariance[packedCovarianceIndex(4, 0)];
  const float c41 = state.covariance[packedCovarianceIndex(4, 1)];
  const float c42 = state.covariance[packedCovarianceIndex(4, 2)];
  const float c43 = state.covariance[packedCovarianceIndex(4, 3)];
  const float c44 = state.covariance[packedCovarianceIndex(4, 4)];

  const float b00 = f02 * c20 + f04 * c40;
  const float b01 = f12 * c20 + f14 * c40 + f13 * c30;
  const float b02 = f24 * c40;
  const float b10 = f02 * c21 + f04 * c41;
  const float b11 = f12 * c21 + f14 * c41 + f13 * c31;
  const float b12 = f24 * c41;
  const float b20 = f02 * c22 + f04 * c42;
  const float b21 = f12 * c22 + f14 * c42 + f13 * c32;
  const float b22 = f24 * c42;
  const float b40 = f02 * c42 + f04 * c44;
  const float b41 = f12 * c42 + f14 * c44 + f13 * c43;
  const float b42 = f24 * c44;
  const float b30 = f02 * c32 + f04 * c43;
  const float b31 = f12 * c32 + f14 * c43 + f13 * c33;
  const float b32 = f24 * c43;

  const float a00 = f02 * b20 + f04 * b40;
  const float a01 = f02 * b21 + f04 * b41;
  const float a02 = f02 * b22 + f04 * b42;
  const float a11 = f12 * b21 + f14 * b41 + f13 * b31;
  const float a12 = f12 * b22 + f14 * b42 + f13 * b32;
  const float a22 = f24 * b42;

  scratchState.covariance[packedCovarianceIndex(0, 0)] = c00 + b00 + b00 + a00;
  scratchState.covariance[packedCovarianceIndex(1, 0)] = c10 + b10 + b01 + a01;
  scratchState.covariance[packedCovarianceIndex(2, 0)] = c20 + b20 + b02 + a02;
  scratchState.covariance[packedCovarianceIndex(3, 0)] = c30 + b30;
  scratchState.covariance[packedCovarianceIndex(4, 0)] = c40 + b40;
  scratchState.covariance[packedCovarianceIndex(1, 1)] = c11 + b11 + b11 + a11;
  scratchState.covariance[packedCovarianceIndex(2, 1)] = c21 + b21 + b12 + a12;
  scratchState.covariance[packedCovarianceIndex(3, 1)] = c31 + b31;
  scratchState.covariance[packedCovarianceIndex(4, 1)] = c41 + b41;
  scratchState.covariance[packedCovarianceIndex(2, 2)] = c22 + b22 + b22 + a22;
  scratchState.covariance[packedCovarianceIndex(3, 2)] = c32 + b32;
  scratchState.covariance[packedCovarianceIndex(4, 2)] = c42 + b42;
  scratchState.covariance[packedCovarianceIndex(3, 3)] = c33;
  scratchState.covariance[packedCovarianceIndex(4, 3)] = c43;
  scratchState.covariance[packedCovarianceIndex(4, 4)] = c44;

  // A large Jacobian step can invalidate covariance through an off-diagonal
  // term even when all diagonals look valid. Sanitize before committing.
  sanitizeCovariance(scratchState, kBarrelMaxDiagonal);
  state = scratchState;
  linRef = scratchRef;
  return true;
}

bool shiftReferenceToMeasurement(SurfaceTrackParameters& linRef, const SurfaceMeasurement& measurement,
                                 OperationFailureReason& reason) noexcept
{
  if (linRef.kind != SurfaceKind::Cylinder) {
    reason = OperationFailureReason::SourceSurfaceKindMismatch;
    return false;
  }
  SurfaceTrackParameters scratch = linRef;
  scratch.parameters[0] = measurement.frame.u;
  scratch.parameters[1] = measurement.frame.v;
  linRef = scratch;
  return true;
}

#endif // GPUCA_GPUCODE

} // namespace o2::itsmft::tracking::detail::barrel
