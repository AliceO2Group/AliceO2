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
#include <limits>

#include "CommonConstants/MathConstants.h"
#include "GPUROOTSMatrixFwd.h"
#include <Math/SMatrix.h>

namespace o2::itsmft::tracking::detail::forward
{
namespace
{

using DenseMatrix5 = float[5][5];

// Packed symmetric 5x5 covariance for stateChi2. MatRepSym::offset matches
// packedCovarianceIndex (row*(row+1)/2+column), enabling direct construction.
using CombinedCovariance = o2::math_utils::SMatrix<float, 5, 5, o2::math_utils::MatRepSym<float, 5>>;
static_assert(o2::math_utils::MatRepSym<float, 5>::kSize == 15, "packed symmetric 5x5 representation must hold exactly 15 floats");
static_assert(sizeof(CombinedCovariance) == 15 * sizeof(float), "combined covariance must occupy exactly 15 floats");

// Forward diagonals have no finite ceiling; non-negativity and correlations
// are still checked.
constexpr float kForwardNoRangeLimit = std::numeric_limits<float>::max();
constexpr float kForwardMaxDiagonal[5] = {kForwardNoRangeLimit, kForwardNoRangeLimit, kForwardNoRangeLimit,
                                          kForwardNoRangeLimit, kForwardNoRangeLimit};

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

void identity(DenseMatrix5& matrix) noexcept
{
  for (uint8_t i = 0; i < 5; ++i) {
    matrix[i][i] = 1.f;
  }
}

bool validateSource(const SurfaceTrackState& state, OperationFailureReason& reason) noexcept
{
  if (state.kind != SurfaceKind::Disk) {
    reason = OperationFailureReason::SourceSurfaceKindMismatch;
    return false;
  }
  return true;
}

// Sanitize covariance once, at the propagation commit point.
bool commitPropagation(SurfaceTrackState& destination, SurfaceTrackState& scratch) noexcept
{
  sanitizeCovariance(scratch, kForwardMaxDiagonal);
  destination = scratch;
  return true;
}

bool propagateLinear(SurfaceTrackState& state, float targetZ, OperationFailureReason& reason) noexcept
{
  const float dz = targetZ - state.referenceCoordinate;
  const float tanl = state.parameters[3];
  if (tanl == 0.f && dz != 0.f) {
    reason = OperationFailureReason::UnreachableTarget;
    return false;
  }
  if (dz == 0.f) {
    return true;
  }
  const float inverseTanl = 1.f / tanl;
  const float n = dz * inverseTanl;
  const float m = n * inverseTanl;
  const float sinPhi = std::sin(state.parameters[2]);
  const float cosPhi = std::cos(state.parameters[2]);
  state.parameters[0] += n * cosPhi;
  state.parameters[1] += n * sinPhi;
  state.referenceCoordinate = targetZ;

  DenseMatrix5 jacobian{};
  identity(jacobian);
  jacobian[0][2] = -n * sinPhi;
  jacobian[0][3] = -m * cosPhi;
  jacobian[1][2] = n * cosPhi;
  jacobian[1][3] = -m * sinPhi;
  transportCovariance(state, jacobian);
  return true;
}

bool propagateHelixParameters(SurfaceTrackState& state, float targetZ, float bz,
                              OperationFailureReason& reason) noexcept
{
  const float dz = targetZ - state.referenceCoordinate;
  if (dz == 0.f) {
    return true;
  }
  const float tanl = state.parameters[3];
  const float inverseQPt = state.parameters[4];
  if (tanl == 0.f) {
    reason = OperationFailureReason::UnreachableTarget;
    return false;
  }
  if (bz == 0.f || inverseQPt == 0.f) {
    reason = OperationFailureReason::PropagationFailure;
    return false;
  }
  const float inverseTanl = 1.f / tanl;
  const float qPt = 1.f / inverseQPt;
  const float phi = state.parameters[2];
  const float sinPhi = std::sin(phi);
  const float cosPhi = std::cos(phi);
  const float k = std::abs(o2::constants::math::B2C * bz);
  const float inverseK = 1.f / k;
  const float theta = -inverseQPt * dz * k * inverseTanl;
  const float sinTheta = std::sin(theta);
  const float cosTheta = std::cos(theta);
  const float fieldSign = std::copysign(1.f, bz);
  const float y = sinPhi * qPt * inverseK;
  const float x = cosPhi * qPt * inverseK;
  state.parameters[0] += fieldSign * (y - y * cosTheta) - x * sinTheta;
  state.parameters[1] += fieldSign * (-x + x * cosTheta) - y * sinTheta;
  state.parameters[2] += fieldSign * theta;
  state.referenceCoordinate = targetZ;
  return true;
}

bool propagateHelix(SurfaceTrackState& state, float targetZ, float bz,
                    OperationFailureReason& reason) noexcept
{
  const float originalZ = state.referenceCoordinate;
  const float dz = targetZ - originalZ;
  if (dz == 0.f) {
    return true;
  }
  const float phi = state.parameters[2];
  const float tanl = state.parameters[3];
  const float inverseQPt = state.parameters[4];
  if (!propagateHelixParameters(state, targetZ, bz, reason)) {
    return false;
  }
  const float inverseTanl = 1.f / tanl;
  const float qPt = 1.f / inverseQPt;
  const float sinPhi = std::sin(phi);
  const float cosPhi = std::cos(phi);
  const float k = std::abs(o2::constants::math::B2C * bz);
  const float inverseK = 1.f / k;
  const float theta = -inverseQPt * dz * k * inverseTanl;
  const float sinTheta = std::sin(theta);
  const float cosTheta = std::cos(theta);
  const float fieldSign = std::copysign(1.f, bz);
  const float n = dz * inverseTanl;
  const float m = n * inverseTanl;
  const float o = sinTheta * cosPhi;
  const float p = sinPhi * cosTheta;
  const float r = sinPhi * sinTheta;
  const float s = cosPhi * cosTheta;
  const float y = sinPhi * qPt * inverseK;
  const float x = cosPhi * qPt * inverseK;
  const float t = qPt * cosTheta;
  const float u = qPt * sinTheta;
  const float v = qPt;
  const float nn = dz * inverseTanl * qPt;

  DenseMatrix5 jacobian{};
  identity(jacobian);
  jacobian[0][2] = fieldSign * x - fieldSign * x * cosTheta + y * sinTheta;
  jacobian[0][3] = fieldSign * r * m - s * m;
  jacobian[0][4] = -fieldSign * nn * r + fieldSign * t * y - fieldSign * v * y + nn * s + u * x;
  jacobian[1][2] = fieldSign * y - fieldSign * y * cosTheta - x * sinTheta;
  jacobian[1][3] = -fieldSign * o * m - p * m;
  jacobian[1][4] = fieldSign * nn * o - fieldSign * t * x + fieldSign * v * x + nn * p + u * y;
  jacobian[2][3] = -fieldSign * theta * inverseTanl;
  jacobian[2][4] = -fieldSign * k * n;
  transportCovariance(state, jacobian);
  return true;
}

bool propagateAccepted(SurfaceTrackState& destination, float targetZ, float bz,
                       OperationFailureReason& reason) noexcept
{
  if (!validateSource(destination, reason)) {
    return false;
  }
  SurfaceTrackState scratch = destination;
  const bool success = std::abs(bz) > 0.01f ? propagateHelix(scratch, targetZ, bz, reason)
                                            : propagateLinear(scratch, targetZ, reason);
  return success && commitPropagation(destination, scratch);
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
  const float residualX = measurement.frame.u - state.parameters[0];
  const float residualY = measurement.frame.v - state.parameters[1];
  const float scratchChi2 = residualX * (inverse00 * residualX + inverse01 * residualY) +
                            residualY * (inverse01 * residualX + inverse11 * residualY);
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
  sanitizeCovariance(scratch, kForwardMaxDiagonal);
  state = scratch;
  chi2 = scratchChi2;
  return true;
}

bool correctForMaterial(SurfaceTrackState& state, float xOverX0, OperationFailureReason& reason) noexcept
{
  if (!validateSource(state, reason)) {
    return false;
  }
  if (xOverX0 == 0.f) {
    return true;
  }
  const float tanl = state.parameters[3];
  if (tanl == 0.f) {
    reason = OperationFailureReason::MaterialFailure;
    return false;
  }
  const float inverseQPt = state.parameters[4];
  const float onePlusTanl2 = 1.f + tanl * tanl;
  const float inverseMomentum = std::abs(inverseQPt) / std::sqrt(onePlusTanl2);
  const float pathLengthOverX0 = xOverX0 * std::abs(std::sqrt(onePlusTanl2) / tanl);
  const float theta2 = highlandTheta2(inverseMomentum, pathLengthOverX0);
  SurfaceTrackState scratch = state;
  scratch.covariance[packedCovarianceIndex(2, 2)] += theta2 * onePlusTanl2;
  scratch.covariance[packedCovarianceIndex(3, 3)] += theta2 * onePlusTanl2 * onePlusTanl2;
  scratch.covariance[packedCovarianceIndex(4, 4)] += theta2 * tanl * tanl * inverseQPt * inverseQPt;
  state = scratch;
  return true;
}

bool stateChi2(const SurfaceTrackState& reference, const SurfaceTrackState& candidate, float& chi2,
               OperationFailureReason& reason) noexcept
{
  if (reference.kind != SurfaceKind::Disk || candidate.kind != SurfaceKind::Disk) {
    reason = OperationFailureReason::SourceSurfaceKindMismatch;
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

  // Use direct (unwrapped) differences of (X, Y, Phi, Tanl, InvQPt).
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

// Reference-only position update with the Jacobian at the original parameters.
bool referencePropagateLinear(SurfaceTrackParameters& ref, float targetZ, DenseMatrix5& jacobian,
                              OperationFailureReason& reason) noexcept
{
  identity(jacobian);
  const float dz = targetZ - ref.referenceCoordinate;
  const float tanl = ref.parameters[3];
  if (tanl == 0.f && dz != 0.f) {
    reason = OperationFailureReason::UnreachableTarget;
    return false;
  }
  if (dz == 0.f) {
    return true;
  }
  const float inverseTanl = 1.f / tanl;
  const float n = dz * inverseTanl;
  const float m = n * inverseTanl;
  const float sinPhi = std::sin(ref.parameters[2]);
  const float cosPhi = std::cos(ref.parameters[2]);
  ref.parameters[0] += n * cosPhi;
  ref.parameters[1] += n * sinPhi;
  ref.referenceCoordinate = targetZ;

  jacobian[0][2] = -n * sinPhi;
  jacobian[0][3] = -m * cosPhi;
  jacobian[1][2] = n * cosPhi;
  jacobian[1][3] = -m * sinPhi;
  return true;
}

// Position-only helix step, matching propagateHelixParameters.
bool referencePropagateHelixParameters(SurfaceTrackParameters& ref, float targetZ, float bz,
                                       OperationFailureReason& reason) noexcept
{
  const float dz = targetZ - ref.referenceCoordinate;
  if (dz == 0.f) {
    return true;
  }
  const float tanl = ref.parameters[3];
  const float inverseQPt = ref.parameters[4];
  if (tanl == 0.f) {
    reason = OperationFailureReason::UnreachableTarget;
    return false;
  }
  if (bz == 0.f || inverseQPt == 0.f) {
    reason = OperationFailureReason::PropagationFailure;
    return false;
  }
  const float inverseTanl = 1.f / tanl;
  const float qPt = 1.f / inverseQPt;
  const float phi = ref.parameters[2];
  const float sinPhi = std::sin(phi);
  const float cosPhi = std::cos(phi);
  const float k = std::abs(o2::constants::math::B2C * bz);
  const float inverseK = 1.f / k;
  const float theta = -inverseQPt * dz * k * inverseTanl;
  const float sinTheta = std::sin(theta);
  const float cosTheta = std::cos(theta);
  const float fieldSign = std::copysign(1.f, bz);
  const float y = sinPhi * qPt * inverseK;
  const float x = cosPhi * qPt * inverseK;
  ref.parameters[0] += fieldSign * (y - y * cosTheta) - x * sinTheta;
  ref.parameters[1] += fieldSign * (-x + x * cosTheta) - y * sinTheta;
  ref.parameters[2] += fieldSign * theta;
  ref.referenceCoordinate = targetZ;
  return true;
}

bool referencePropagateHelix(SurfaceTrackParameters& ref, float targetZ, float bz, DenseMatrix5& jacobian,
                             OperationFailureReason& reason) noexcept
{
  identity(jacobian);
  const float originalZ = ref.referenceCoordinate;
  const float dz = targetZ - originalZ;
  if (dz == 0.f) {
    return true;
  }
  const float phi = ref.parameters[2];
  const float tanl = ref.parameters[3];
  const float inverseQPt = ref.parameters[4];
  if (!referencePropagateHelixParameters(ref, targetZ, bz, reason)) {
    return false;
  }
  const float inverseTanl = 1.f / tanl;
  const float qPt = 1.f / inverseQPt;
  const float sinPhi = std::sin(phi);
  const float cosPhi = std::cos(phi);
  const float k = std::abs(o2::constants::math::B2C * bz);
  const float inverseK = 1.f / k;
  const float theta = -inverseQPt * dz * k * inverseTanl;
  const float sinTheta = std::sin(theta);
  const float cosTheta = std::cos(theta);
  const float fieldSign = std::copysign(1.f, bz);
  const float n = dz * inverseTanl;
  const float m = n * inverseTanl;
  const float o = sinTheta * cosPhi;
  const float p = sinPhi * cosTheta;
  const float r = sinPhi * sinTheta;
  const float s = cosPhi * cosTheta;
  const float y = sinPhi * qPt * inverseK;
  const float x = cosPhi * qPt * inverseK;
  const float t = qPt * cosTheta;
  const float u = qPt * sinTheta;
  const float v = qPt;
  const float nn = dz * inverseTanl * qPt;

  jacobian[0][2] = fieldSign * x - fieldSign * x * cosTheta + y * sinTheta;
  jacobian[0][3] = fieldSign * r * m - s * m;
  jacobian[0][4] = -fieldSign * nn * r + fieldSign * t * y - fieldSign * v * y + nn * s + u * x;
  jacobian[1][2] = fieldSign * y - fieldSign * y * cosTheta - x * sinTheta;
  jacobian[1][3] = -fieldSign * o * m - p * m;
  jacobian[1][4] = fieldSign * nn * o - fieldSign * t * x + fieldSign * v * x + nn * p + u * y;
  jacobian[2][3] = -fieldSign * theta * inverseTanl;
  jacobian[2][4] = -fieldSign * k * n;
  return true;
}

bool propagateAccepted(SurfaceTrackState& state, SurfaceTrackParameters& linRef, float targetZ, float bz,
                       OperationFailureReason& reason) noexcept
{
  if (!validateSource(state, reason)) {
    return false;
  }
  if (linRef.kind != SurfaceKind::Disk) {
    reason = OperationFailureReason::SourceSurfaceKindMismatch;
    return false;
  }
  // The fitted state and linearization reference must share the exact anchor;
  // their parameters may differ. Forward alpha is always 0/unused.
  if (state.referenceCoordinate != linRef.referenceCoordinate) {
    reason = OperationFailureReason::ReferenceCoordinateMismatch;
    return false;
  }

  SurfaceTrackParameters scratchRef = linRef;
  DenseMatrix5 jacobian{};
  const bool ok = std::abs(bz) > 0.01f ? referencePropagateHelix(scratchRef, targetZ, bz, jacobian, reason)
                                       : referencePropagateLinear(scratchRef, targetZ, jacobian, reason);
  if (!ok) {
    return false;
  }

  float diff[5];
  for (uint8_t i = 0; i < 5; ++i) {
    diff[i] = state.parameters[i] - linRef.parameters[i];
  }

  SurfaceTrackState scratchState = state;
  scratchState.referenceCoordinate = targetZ;
  for (uint8_t row = 0; row < 5; ++row) {
    float value = scratchRef.parameters[row];
    for (uint8_t column = 0; column < 5; ++column) {
      value += jacobian[row][column] * diff[column];
    }
    scratchState.parameters[row] = value;
  }
  transportCovariance(scratchState, jacobian);

  // ADR 0008: a large Jacobian step can break positive semidefiniteness via
  // an off-diagonal term even when diagonals look valid. Sanitize before the
  // next operation receives the covariance.
  sanitizeCovariance(scratchState, kForwardMaxDiagonal);
  state = scratchState;
  linRef = scratchRef;
  return true;
}

} // namespace

bool shiftReferenceToMeasurement(SurfaceTrackParameters& linRef, const SurfaceMeasurement& measurement,
                                 OperationFailureReason& reason) noexcept
{
  if (linRef.kind != SurfaceKind::Disk) {
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

bool propagate(SurfaceTrackState& state, float targetZ, float bz,
               OperationFailureReason& reason) noexcept
{
  return propagateAccepted(state, targetZ, bz, reason);
}

bool propagate(SurfaceTrackState& state, SurfaceTrackParameters& linRef,
               float targetZ, float bz, OperationFailureReason& reason) noexcept
{
  return propagateAccepted(state, linRef, targetZ, bz, reason);
}

} // namespace o2::itsmft::tracking::detail::forward
