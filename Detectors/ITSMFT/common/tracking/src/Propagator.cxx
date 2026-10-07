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

#include "ITSMFTTracking/Propagator.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include "CommonConstants/MathConstants.h"
#include "ITSMFTTracking/MaterialPhysics.h"
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/TrackParametrization.h"

namespace o2::itsmft::tracking
{

namespace
{

// Remove tiny negative diagonal values caused by floating-point cancellation
// during covariance transport. Larger negative values remain errors.
void clampNegligibleCovarianceNoise(SurfaceTrackState& state) noexcept
{
  constexpr float kNoiseFloor = 1.e-3f;
  for (uint8_t i = 0; i < 5; ++i) {
    const uint8_t index = packedCovarianceIndex(i, i);
    if (state.covariance[index] < 0.f && state.covariance[index] > -kNoiseFloor) {
      state.covariance[index] = 0.f;
    }
  }
}

// Apply outCov = J * inCov * J^T to a packed-symmetric 5x5 covariance.
void congruenceTransform(const float (&inCov)[15], const float (&jacobian)[5][5], float (&outCov)[15]) noexcept
{
  float full[5][5];
  for (uint8_t row = 0; row < 5; ++row) {
    for (uint8_t col = 0; col < 5; ++col) {
      full[row][col] = inCov[packedCovarianceIndex(row, col)];
    }
  }
  float tmp[5][5];
  for (uint8_t row = 0; row < 5; ++row) {
    for (uint8_t col = 0; col < 5; ++col) {
      float sum = 0.f;
      for (uint8_t k = 0; k < 5; ++k) {
        sum += jacobian[row][k] * full[k][col];
      }
      tmp[row][col] = sum;
    }
  }
  for (uint8_t row = 0; row < 5; ++row) {
    for (uint8_t col = 0; col <= row; ++col) {
      float sum = 0.f;
      for (uint8_t k = 0; k < 5; ++k) {
        sum += tmp[row][k] * jacobian[col][k];
      }
      outCov[packedCovarianceIndex(row, col)] = sum;
    }
  }
}

// Convert Barrel (bY, bZ, Snp, Tgl, Q2Pt) to Forward
// (X, Y, Phi, Tanl, InvQPt) on the fixed-z plane through the nominal point.
bool barrelToForward(SurfaceTrackState& state, float bz) noexcept
{
  const float snp = state.parameters[2];
  const float tanl = state.parameters[3];
  if (!(std::abs(snp) < 1.f) || tanl == 0.f) {
    return false;
  }
  const float csA = std::cos(state.alpha);
  const float snA = std::sin(state.alpha);
  const float csp = std::sqrt((1.f - snp) * (1.f + snp));
  const float bX = state.referenceCoordinate;
  const float bY = state.parameters[0];

  const float xGlo = bX * csA - bY * snA;
  const float yGlo = bX * snA + bY * csA;
  const float zGlo = state.parameters[1];
  float phi = std::remainder(state.alpha + std::asin(snp), o2::constants::math::TwoPI);
  // Match the library's (-pi, pi] angle convention.
  if (phi <= -o2::constants::math::PI) {
    phi += o2::constants::math::TwoPI;
  }

  // A displaced source z reaches the fixed target plane after transverse
  // path -deltaZ/tanl. Include both position and direction along that path.
  const float curvature = state.parameters[4] * bz * o2::constants::math::B2C;
  const float jacobian[5][5] = {
    {-snA, -(csA * csp - snA * snp) / tanl, 0.f, 0.f, 0.f},
    {csA, -(snA * csp + csA * snp) / tanl, 0.f, 0.f, 0.f},
    {0.f, -curvature / tanl, 1.f / csp, 0.f, 0.f},
    {0.f, 0.f, 0.f, 1.f, 0.f},
    {0.f, 0.f, 0.f, 0.f, 1.f}};
  float newCov[15];
  congruenceTransform(state.covariance, jacobian, newCov);

  const float newParameters[5] = {xGlo, yGlo, phi, state.parameters[3], state.parameters[4]};
  for (uint8_t i = 0; i < 5; ++i) {
    state.parameters[i] = newParameters[i];
  }
  for (uint8_t i = 0; i < 15; ++i) {
    state.covariance[i] = newCov[i];
  }
  state.referenceCoordinate = zGlo;
  state.alpha = 0.f;
  state.kind = SurfaceKind::Disk;
  return true;
}

// Convert Forward (X, Y, Phi, Tanl, InvQPt) to Barrel
// (bY, bZ, Snp, Tgl, Q2Pt) on the fixed local-x plane through the nominal
// point. Both target alpha and local x are held fixed in the Jacobian.
bool forwardToBarrel(SurfaceTrackState& state, float bz) noexcept
{
  const float x = state.parameters[0];
  const float y = state.parameters[1];
  const float r = std::sqrt(x * x + y * y);
  if (!(r > 1.e-6f)) {
    return false;
  }
  const float alpha = std::atan2(y, x);
  const float csA = std::cos(alpha);
  const float snA = std::sin(alpha);
  const float phi = state.parameters[2];
  const float csp = std::cos(phi - alpha);
  const float snp = std::sin(phi - alpha);
  // The barrel convention encodes only the positive-cosine branch at alpha.
  // Reject inward/tangent directions rather than silently reversing them.
  if (!(csp > 0.f && std::abs(snp) < 1.f)) {
    return false;
  }

  const float bX = x * csA + y * snA;
  const float bY = -x * snA + y * csA;
  const float bZ = state.referenceCoordinate;

  // A displacement along the plane normal shifts the intersection by
  // transverse path -deltaX/csp, inducing local-y, z and direction errors.
  const float curvature = state.parameters[4] * bz * o2::constants::math::B2C;
  const float tanlOverCsp = state.parameters[3] / csp;
  const float jacobian[5][5] = {
    {-snA - snp * csA / csp, csA - snp * snA / csp, 0.f, 0.f, 0.f},
    {-tanlOverCsp * csA, -tanlOverCsp * snA, 0.f, 0.f, 0.f},
    {-curvature * csA, -curvature * snA, csp, 0.f, 0.f},
    {0.f, 0.f, 0.f, 1.f, 0.f},
    {0.f, 0.f, 0.f, 0.f, 1.f}};
  float newCov[15];
  congruenceTransform(state.covariance, jacobian, newCov);

  const float newParameters[5] = {bY, bZ, snp, state.parameters[3], state.parameters[4]};
  for (uint8_t i = 0; i < 5; ++i) {
    state.parameters[i] = newParameters[i];
  }
  for (uint8_t i = 0; i < 15; ++i) {
    state.covariance[i] = newCov[i];
  }
  state.referenceCoordinate = bX;
  state.alpha = alpha;
  state.kind = SurfaceKind::Cylinder;
  return true;
}

// Both attachment algorithms work on a candidate and commit only after every
// fallible operation succeeds. Linearized attachment also keeps a local reference.
struct AttachmentTransaction {
  SurfaceTrackState state;
  float chi2;

  void commit(SurfaceTrackState& destination, float& destinationChi2) const noexcept
  {
    destination = state;
    destinationChi2 = chi2;
  }
};

bool acceptsAttachmentChi2(float predictedChi2, bool gateEnabled, float maxChi2) noexcept
{
  if (predictedChi2 < 0.f || (gateEnabled && predictedChi2 > maxChi2)) {
    return false;
  }
  return true;
}

bool covarianceDiagonalsNonNegative(const SurfaceTrackState& state) noexcept
{
  for (uint8_t i = 0; i < 5; ++i) {
    if (state.covariance[packedCovarianceIndex(i, i)] < 0.f) {
      return false;
    }
  }
  return true;
}

// Barrel covariance-range upper bound, in (Y, Z, Snp, Tgl, Q2Pt) slot order:
// the retained TrackParametrizationWithError<float>::checkCovariance()
// range-clamp values shared by material correction and barrel
// propagation, rotation, and update sanitization.
constexpr float kBarrelMaxDiagonal[5] = {o2::track::kCY2max, o2::track::kCZ2max, o2::track::kCSnp2max,
                                         o2::track::kCTgl2max, o2::track::kC1Pt2max};

using DenseMatrix5 = float[5][5];

bool validateBarrelSource(const SurfaceTrackState& state) noexcept
{
  if (state.kind != SurfaceKind::Cylinder) {
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
// and sanitizes the covariance on every exit, including dx == 0.
bool commitBarrelPropagation(SurfaceTrackState& destination, SurfaceTrackState& scratch) noexcept
{
  sanitizeCovariance(scratch, kBarrelMaxDiagonal);
  destination = scratch;
  return true;
}

bool residualInverse(const SurfaceTrackState& state, const SurfaceMeasurement& measurement,
                     float& inverse00, float& inverse01, float& inverse11) noexcept
{
  if (!(measurement.covariance.uu >= 0.f) || !(measurement.covariance.vv >= 0.f)) {
    return false;
  }
  const float s00 = state.covariance[packedCovarianceIndex(0, 0)] + measurement.covariance.uu;
  const float s01 = state.covariance[packedCovarianceIndex(1, 0)] + measurement.covariance.uv;
  const float s11 = state.covariance[packedCovarianceIndex(1, 1)] + measurement.covariance.vv;
  const float determinant = s00 * s11 - s01 * s01;
  if (determinant == 0.f) {
    return false;
  }
  const float inverseDeterminant = 1.f / determinant;
  inverse00 = s11 * inverseDeterminant;
  inverse01 = -s01 * inverseDeterminant;
  inverse11 = s00 * inverseDeterminant;
  return true;
}

// Covariance-free propagation of SurfaceTrackParameters using the
// TrackParametrization::propagateParamTo formula for charged particles.
bool propagateReferenceParams(SurfaceTrackParameters& ref, float targetX, float bz) noexcept
{
  const float dx = targetX - ref.referenceCoordinate;
  if (dx == 0.f) {
    ref.referenceCoordinate = targetX;
    return true;
  }
  const float snp = ref.parameters[2];
  const float curvature = ref.parameters[4] * bz * o2::constants::math::B2C;
  const float propagatedSnp = snp + curvature * dx;
  if (std::abs(snp) >= 1.f || std::abs(propagatedSnp) >= 1.f) {
    return false;
  }
  const float csp = std::sqrt((1.f - snp) * (1.f + snp));
  const float propagatedCsp = std::sqrt((1.f - propagatedSnp) * (1.f + propagatedSnp));
  if (csp == 0.f || propagatedCsp == 0.f) {
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

// Forward diagonals have no finite ceiling; non-negativity and correlations
// are still checked.
constexpr float kForwardNoRangeLimit = std::numeric_limits<float>::max();
constexpr float kForwardMaxDiagonal[5] = {kForwardNoRangeLimit, kForwardNoRangeLimit, kForwardNoRangeLimit,
                                          kForwardNoRangeLimit, kForwardNoRangeLimit};

bool validateForwardSource(const SurfaceTrackState& state) noexcept
{
  if (state.kind != SurfaceKind::Disk) {
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

bool propagateLinear(SurfaceTrackState& state, float targetZ) noexcept
{
  const float dz = targetZ - state.referenceCoordinate;
  const float tanl = state.parameters[3];
  if (tanl == 0.f && dz != 0.f) {
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

// Share the same helix and Jacobian between direct and reference propagation.
// The midpoint-angle form avoids subtracting O(1/curvature) coordinates;
// its sinc derivative also remains well conditioned for almost straight tracks.
template <typename State>
bool propagateHelixWithJacobian(State& state, float targetZ, float bz, DenseMatrix5& jacobian) noexcept
{
  identity(jacobian);
  const float dz = targetZ - state.referenceCoordinate;
  if (dz == 0.f) {
    return true;
  }
  const float tanl = state.parameters[3];
  const float inverseQPt = state.parameters[4];
  if (tanl == 0.f || bz == 0.f || inverseQPt == 0.f) {
    return false;
  }
  const float n = dz / tanl;
  const float curvatureScale = -std::abs(o2::constants::math::B2C) * bz;
  const float halfAnglePerQPt = 0.5f * curvatureScale * n;
  const float halfAngle = inverseQPt * halfAnglePerQPt;
  float sinc, sincDerivative;
  if (std::abs(halfAngle) < 0.25f) {
    // sin(h)/h and its derivative, including their limits at h = 0.
    // Keep the cancellation-prone derivative quotient away from small h.
    // At |h| <= 0.25 the omitted terms are below float precision.
    const float h2 = halfAngle * halfAngle;
    sinc = std::fma(h2, std::fma(h2, std::fma(h2, -1.f / 5040.f, 1.f / 120.f), -1.f / 6.f), 1.f);
    sincDerivative = halfAngle * std::fma(h2, std::fma(h2, -1.f / 840.f, 1.f / 30.f), -1.f / 3.f);
  } else {
    sinc = std::sin(halfAngle) / halfAngle;
    sincDerivative = (std::cos(halfAngle) - sinc) / halfAngle;
  }
  const float phi = state.parameters[2];
  const float sinMid = std::sin(phi + halfAngle);
  const float cosMid = std::cos(phi + halfAngle);
  const float endPhi = phi + 2.f * halfAngle;
  const float dx = n * sinc * cosMid;
  const float dy = n * sinc * sinMid;

  jacobian[0][2] = -dy;
  jacobian[1][2] = dx;
  jacobian[0][3] = -n / tanl * std::cos(endPhi);
  jacobian[1][3] = -n / tanl * std::sin(endPhi);
  jacobian[0][4] = n * halfAnglePerQPt * std::fma(sincDerivative, cosMid, -sinc * sinMid);
  jacobian[1][4] = n * halfAnglePerQPt * std::fma(sincDerivative, sinMid, sinc * cosMid);
  jacobian[2][3] = -2.f * halfAngle / tanl;
  jacobian[2][4] = 2.f * halfAnglePerQPt;

  state.parameters[0] = std::fma(n * sinc, cosMid, state.parameters[0]);
  state.parameters[1] = std::fma(n * sinc, sinMid, state.parameters[1]);
  state.parameters[2] = endPhi;
  state.referenceCoordinate = targetZ;
  return true;
}

bool propagateHelix(SurfaceTrackState& state, float targetZ, float bz) noexcept
{
  if (targetZ == state.referenceCoordinate) {
    return true;
  }
  DenseMatrix5 jacobian{};
  if (!propagateHelixWithJacobian(state, targetZ, bz, jacobian)) {
    return false;
  }
  transportCovariance(state, jacobian);
  return true;
}

bool propagateAccepted(SurfaceTrackState& destination, float targetZ, float bz) noexcept
{
  if (!validateForwardSource(destination)) {
    return false;
  }
  SurfaceTrackState scratch = destination;
  const bool success = std::abs(bz) > 0.01f ? propagateHelix(scratch, targetZ, bz)
                                            : propagateLinear(scratch, targetZ);
  return success && commitPropagation(destination, scratch);
}

// Reference-only position update with the Jacobian at the original parameters.
bool referencePropagateLinear(SurfaceTrackParameters& ref, float targetZ, DenseMatrix5& jacobian) noexcept
{
  identity(jacobian);
  const float dz = targetZ - ref.referenceCoordinate;
  const float tanl = ref.parameters[3];
  if (tanl == 0.f && dz != 0.f) {
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

bool referencePropagateHelix(SurfaceTrackParameters& ref, float targetZ, float bz, DenseMatrix5& jacobian) noexcept
{
  return propagateHelixWithJacobian(ref, targetZ, bz, jacobian);
}

bool propagateAccepted(SurfaceTrackState& state, SurfaceTrackParameters& linRef, float targetZ, float bz) noexcept
{
  if (!validateForwardSource(state)) {
    return false;
  }
  if (linRef.kind != SurfaceKind::Disk) {
    return false;
  }
  // The fitted state and linearization reference must share the exact anchor;
  // their parameters may differ. Forward alpha is always 0/unused.
  if (state.referenceCoordinate != linRef.referenceCoordinate) {
    return false;
  }

  SurfaceTrackParameters scratchRef = linRef;
  DenseMatrix5 jacobian{};
  const bool ok = std::abs(bz) > 0.01f ? referencePropagateHelix(scratchRef, targetZ, bz, jacobian)
                                       : referencePropagateLinear(scratchRef, targetZ, jacobian);
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

  // a large Jacobian step can break positive semidefiniteness via
  // an off-diagonal term even when diagonals look valid. Sanitize before the
  // next operation receives the covariance.
  sanitizeCovariance(scratchState, kForwardMaxDiagonal);
  state = scratchState;
  linRef = scratchRef;
  return true;
}

} // namespace

// Work on copies so that any rejection leaves both the fitted state and its
// incidence reference unchanged.
bool Propagator::correctForMaterial(SurfaceTrackState& state, SurfaceTrackParameters& incidenceReference,
                                    material::IntegratedMaterialBudget materialBudget,
                                    material::MaterialTraversalDirection direction) noexcept
{
  if (state.parameters[4] == 0.f || incidenceReference.parameters[4] == 0.f) {
    return false;
  }
  if (state.kind == SurfaceKind::Cylinder) {
    if (!(std::abs(state.parameters[2]) < 1.f) || !(std::abs(incidenceReference.parameters[2]) < 1.f)) {
      return false;
    }
  } else if (state.parameters[3] == 0.f || incidenceReference.parameters[3] == 0.f) {
    return false;
  }
  if (state.pid.getID() >= o2::track::PID::NIDsTot) {
    return false;
  }
  if (state.pid.getMass() == 0.f) {
    return false;
  }
  if (!covarianceDiagonalsNonNegative(state)) {
    return false;
  }

  float momentumBeforeGeV = state.getP();
  SurfaceTrackState scratchState = state;
  SurfaceTrackParameters scratchReference = incidenceReference;
  // Layer budgets describe normal incidence. Use the reference trajectory to
  // scale both radiation length and areal density to the crossed path length.
  const float tgl = scratchReference.parameters[3];
  float incidenceScale;
  if (state.kind == SurfaceKind::Cylinder) {
    const float snp = scratchReference.parameters[2];
    const float cosPhi2 = (1.f - snp) * (1.f + snp);
    const float inverseCosLambda2 = 1.f + tgl * tgl;
    incidenceScale = std::sqrt(inverseCosLambda2 / cosPhi2);
  } else {
    incidenceScale = std::sqrt(1.f + tgl * tgl) / std::abs(tgl);
  }
  materialBudget.xOverX0 *= incidenceScale;
  materialBudget.arealDensityGPerCm2 *= incidenceScale;
  float momentumAfterGeV = 0.f;
  float highlandTheta2Rad2 = 0.f;
  float relativeInverseMomentumVariance = 0.f;
  if (!material::calculateMaterialPhysics(momentumBeforeGeV, scratchState.pid, scratchState.absCharge, direction, materialBudget,
                                          momentumAfterGeV, highlandTheta2Rad2, relativeInverseMomentumVariance)) {
    return false;
  }

  // No material must also bypass covariance limiting.
  const bool isNoopMaterial = (materialBudget.xOverX0 == 0.f && materialBudget.arealDensityGPerCm2 == 0.f);
  if (isNoopMaterial) {
    return true;
  }

  const float tBefore = scratchState.parameters[3];
  const float kBefore = scratchState.parameters[4];
  const float A = 1.f + tBefore * tBefore;
  const float h = highlandTheta2Rad2;
  const float R = relativeInverseMomentumVariance;
  if (state.kind == SurfaceKind::Cylinder) {
    // Barrel slot 2 is sin(phi); disk slot 2 is phi itself.
    const float snp = scratchState.parameters[2];
    const float c2 = 1.f - snp * snp;
    scratchState.covariance[packedCovarianceIndex(2, 2)] += h * A * c2;
  } else {
    scratchState.covariance[packedCovarianceIndex(2, 2)] += h * A;
  }
  scratchState.covariance[packedCovarianceIndex(3, 3)] += h * A * A;
  scratchState.covariance[packedCovarianceIndex(4, 3)] += h * A * tBefore * kBefore;
  scratchState.covariance[packedCovarianceIndex(4, 4)] += h * (tBefore * kBefore) * (tBefore * kBefore) + kBefore * kBefore * R;
  if (state.kind == SurfaceKind::Cylinder) {
    sanitizeCovariance(scratchState, kBarrelMaxDiagonal);
  }

  // The equality branch preserves the exact no-op invariant for the
  // MCS-only-with-unchanged-momentum case (xOverX0 > 0, arealDensity == 0):
  // x == y implies kAfter == kBefore bit-for-bit with no division rounding.
  // The nonzero-change branch keeps the accepted/legacy left-to-right
  // arithmetic (multiply, then divide) rather than dividing the momenta
  // first, which would prematurely underflow for extreme momentum ratios
  // and would not reproduce the retained nonzero-material rounding.
  const float kAfter = (momentumBeforeGeV == momentumAfterGeV)
                         ? kBefore
                         : (kBefore * momentumBeforeGeV) / momentumAfterGeV;
  scratchState.parameters[4] = kAfter;

  // Only covariance and inverse transverse momentum changed; the coordinate
  // preconditions checked above still hold.
  if (scratchState.parameters[4] == 0.f) {
    return false;
  }
  float momentumAfterDerived = scratchState.getP();
  if (!covarianceDiagonalsNonNegative(scratchState)) {
    return false;
  }

  // Energy loss changes q/pT in the covariance-bearing state and its
  // incidence reference by the same pBefore/pAfter factor. The equality
  // branch keeps MCS-only corrections bit-exact.
  const float referenceKBefore = scratchReference.parameters[4];
  scratchReference.parameters[4] = (momentumBeforeGeV == momentumAfterGeV)
                                     ? referenceKBefore
                                     : (referenceKBefore * momentumBeforeGeV) / momentumAfterGeV;
  if (scratchReference.parameters[4] == 0.f || !std::isfinite(scratchReference.parameters[4])) {
    return false;
  }

  state = scratchState;
  incidenceReference = scratchReference;
  return true;
}

bool Propagator::attachMeasurement(SurfaceTrackState& state, const SurfaceDescriptor& targetSurface,
                                   const SurfaceMeasurement& measurement, float bz,
                                   material::MaterialTraversalDirection direction,
                                   bool chi2GateEnabled, float maxChi2, float& chi2) noexcept
{
  if (!acceptsAttachmentChi2(0.f, chi2GateEnabled, maxChi2)) {
    return false;
  }

  AttachmentTransaction transaction{state, chi2};
  auto& scratch = transaction.state;
  if (!convertKind(scratch, targetSurface.kind, bz)) {
    return false;
  }
  const auto materialBudget = targetSurface.material;
  float predictedChi2 = 0.f;
  float updateChi2 = 0.f;
  const material::IntegratedMaterialBudget integratedMaterial{materialBudget.xOverX0, materialBudget.arealDensityGPerCm2};
  if (scratch.kind == SurfaceKind::Cylinder) {
    if (!rotateBarrel(scratch, measurement.frame.frameAngle) ||
        !propagateBarrel(scratch, measurement.frame.q, bz)) {
      return false;
    }
    SurfaceTrackParameters incidenceReference{scratch};
    const auto materialResult = correctForMaterial(scratch, incidenceReference, integratedMaterial, direction);
    if (!materialResult) {
      return false;
    }
    if (!predictedChi2Barrel(scratch, measurement, predictedChi2)) {
      return false;
    }
    if (!acceptsAttachmentChi2(predictedChi2, chi2GateEnabled, maxChi2)) {
      return false;
    }
    if (!updateBarrel(scratch, measurement, updateChi2)) {
      return false;
    }
  } else if (scratch.kind == SurfaceKind::Disk) {
    if (!propagateToReference(scratch, measurement.frame.q, bz)) {
      return false;
    }
    SurfaceTrackParameters incidenceReference{scratch};
    const auto materialResult = correctForMaterial(scratch, incidenceReference, integratedMaterial, direction);
    if (!materialResult) {
      return false;
    }
    if (!predictedChi2Forward(scratch, measurement, predictedChi2)) {
      return false;
    }
    if (!acceptsAttachmentChi2(predictedChi2, chi2GateEnabled, maxChi2)) {
      return false;
    }
    if (!updateForward(scratch, measurement, updateChi2)) {
      return false;
    }
  } else {
    return false;
  }
  transaction.chi2 += updateChi2;
  transaction.commit(state, chi2);
  return true;
}

bool Propagator::propagateToReference(SurfaceTrackState& state, float targetReferenceCoordinate, float bz) noexcept
{
  if (state.kind == SurfaceKind::Cylinder) {
    return propagateBarrel(state, targetReferenceCoordinate, bz);
  }
  if (state.kind == SurfaceKind::Disk) {
    return propagateForward(state, targetReferenceCoordinate, bz);
  }

  return false;
}

bool Propagator::propagateToReference(SurfaceTrackState& state, SurfaceTrackParameters& linRef,
                                      float targetReferenceCoordinate, float bz) noexcept
{
  if (state.kind != linRef.kind) {
    return false;
  }
  if (state.kind == SurfaceKind::Cylinder) {
    return propagateBarrel(state, linRef, targetReferenceCoordinate, bz);
  }
  if (state.kind == SurfaceKind::Disk) {
    return propagateForward(state, linRef, targetReferenceCoordinate, bz);
  }

  return false;
}

bool Propagator::convertKind(SurfaceTrackState& state, SurfaceKind targetKind, float bz) noexcept
{
  if (targetKind != SurfaceKind::Cylinder && targetKind != SurfaceKind::Disk) {
    return false;
  }
  if (state.kind != SurfaceKind::Cylinder && state.kind != SurfaceKind::Disk) {
    return false;
  }
  if (state.kind == targetKind) {
    return true;
  }
  auto finiteState = [](const SurfaceTrackState& value) {
    if (!std::isfinite(value.referenceCoordinate) || !std::isfinite(value.alpha)) {
      return false;
    }
    for (float parameter : value.parameters) {
      if (!std::isfinite(parameter)) {
        return false;
      }
    }
    for (float covariance : value.covariance) {
      if (!std::isfinite(covariance)) {
        return false;
      }
    }
    return true;
  };
  if (!std::isfinite(bz) || !finiteState(state)) {
    return false;
  }
  SurfaceTrackState scratch = state;
  const bool converted = targetKind == SurfaceKind::Disk ? barrelToForward(scratch, bz)
                                                         : forwardToBarrel(scratch, bz);
  if (!converted || !finiteState(scratch)) {
    return false;
  }
  state = scratch;
  return true;
}

bool Propagator::propagateToMeasurement(SurfaceTrackState& state, SurfaceTrackParameters& linRef,
                                        const SurfaceDescriptor& targetSurface, const SurfaceMeasurement& targetMeasurement,
                                        float bz, material::MaterialTraversalDirection direction,
                                        bool chi2GateEnabled, float maxChi2, float& chi2,
                                        bool shiftReferenceToMeasurement) noexcept
{
  if (chi2 < 0.f) {
    return false;
  }
  if (!acceptsAttachmentChi2(0.f, chi2GateEnabled, maxChi2)) {
    return false;
  }

  const SurfaceKind targetKind = targetSurface.kind;
  if (targetKind == SurfaceKind::Undefined) {
    return false;
  }

  AttachmentTransaction transaction{state, chi2};
  auto& scratchState = transaction.state;
  SurfaceTrackParameters scratchRef = linRef;

  if (scratchState.kind != targetKind) {
    if (!convertKind(scratchState, targetKind, bz)) {
      return false;
    }
    // Changing parameter conventions is also a relinearization boundary.
    // The conversion Jacobian is evaluated at scratchState, so begin the
    // target-kind propagation from that same point.
    scratchRef = SurfaceTrackParameters{scratchState};
  }

  const material::IntegratedMaterialBudget materialBudget{targetSurface.material.xOverX0, targetSurface.material.arealDensityGPerCm2};
  auto& scratchChi2 = transaction.chi2;
  float predChi2 = 0.f;
  float updateChi2 = 0.f;

  if (targetKind == SurfaceKind::Cylinder) {
    if (!rotateBarrel(scratchState, scratchRef, targetMeasurement.frame.frameAngle, bz)) {
      return false;
    }
    if (!propagateBarrel(scratchState, scratchRef, targetMeasurement.frame.q, bz)) {
      return false;
    }
    clampNegligibleCovarianceNoise(scratchState);
    const auto materialResult = correctForMaterial(scratchState, scratchRef, materialBudget, direction);
    if (!materialResult) {
      return false;
    }
    if (!predictedChi2Barrel(scratchState, targetMeasurement, predChi2)) {
      return false;
    }
  } else {
    if (!Propagator::propagateToReference(scratchState, scratchRef, targetMeasurement.frame.q, bz)) {
      return false;
    }
    clampNegligibleCovarianceNoise(scratchState);
    const auto materialResult = correctForMaterial(scratchState, scratchRef, materialBudget, direction);
    if (!materialResult) {
      return false;
    }
    if (!predictedChi2Forward(scratchState, targetMeasurement, predChi2)) {
      return false;
    }
  }

  if (!acceptsAttachmentChi2(predChi2, chi2GateEnabled, maxChi2)) {
    return false;
  }

  if (targetKind == SurfaceKind::Cylinder) {
    if (!updateBarrel(scratchState, targetMeasurement, updateChi2)) {
      return false;
    }
  } else {
    if (!updateForward(scratchState, targetMeasurement, updateChi2)) {
      return false;
    }
  }
  scratchChi2 += updateChi2;
  if (scratchChi2 < 0.f) {
    return false;
  }

  if (shiftReferenceToMeasurement) {
    if (targetKind == SurfaceKind::Cylinder) {
      if (!shiftReferenceToMeasurementBarrel(scratchRef, targetMeasurement)) {
        return false;
      }
    } else {
      if (!shiftReferenceToMeasurementForward(scratchRef, targetMeasurement)) {
        return false;
      }
    }
  }

  transaction.commit(state, chi2);
  linRef = scratchRef;
  return true;
}

bool Propagator::rotateBarrel(SurfaceTrackState& state, float targetAlpha) noexcept
{
  if (!validateBarrelSource(state)) {
    return false;
  }
  SurfaceTrackState scratch = state;
  const float canonicalTargetAlpha = std::remainder(targetAlpha, 2.f * o2::constants::math::PI);
  const float delta = std::remainder(canonicalTargetAlpha - scratch.alpha, 2.f * o2::constants::math::PI);
  const float sine = std::sin(delta);
  const float cosine = std::cos(delta);
  const float snp = scratch.parameters[2];
  if (std::abs(snp) >= 1.f) {
    return false;
  }
  const float csp = std::sqrt((1.f - snp) * (1.f + snp));
  const float rotatedCosine = csp * cosine + snp * sine;
  const float rotatedSnp = snp * cosine - csp * sine;
  if (rotatedCosine < 0.f || std::abs(rotatedSnp) >= 1.f || csp == 0.f) {
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
  return commitBarrelPropagation(state, scratch);
}

bool Propagator::propagateBarrel(SurfaceTrackState& state, float targetX, float bz) noexcept
{
  if (!validateBarrelSource(state)) {
    return false;
  }
  SurfaceTrackState scratch = state;
  const float dx = targetX - scratch.referenceCoordinate;
  if (dx == 0.f) {
    scratch.referenceCoordinate = targetX;
    return commitBarrelPropagation(state, scratch);
  }
  const float snp = scratch.parameters[2];
  const float curvature = scratch.parameters[4] * bz * o2::constants::math::B2C;
  const float propagatedSnp = snp + curvature * dx;
  if (std::abs(snp) >= 1.f || std::abs(propagatedSnp) >= 1.f) {
    return false;
  }
  const float csp = std::sqrt((1.f - snp) * (1.f + snp));
  const float propagatedCsp = std::sqrt((1.f - propagatedSnp) * (1.f + propagatedSnp));
  if (csp == 0.f || propagatedCsp == 0.f) {
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
  return commitBarrelPropagation(state, scratch);
}

bool Propagator::predictedChi2Barrel(const SurfaceTrackState& state, const SurfaceMeasurement& measurement, float& chi2) noexcept
{
  if (!validateBarrelSource(state)) {
    return false;
  }
  float inverse00 = 0.f;
  float inverse01 = 0.f;
  float inverse11 = 0.f;
  if (!residualInverse(state, measurement, inverse00, inverse01, inverse11)) {
    return false;
  }
  const float residualY = measurement.frame.u - state.parameters[0];
  const float residualZ = measurement.frame.v - state.parameters[1];
  const float scratchChi2 = residualY * (inverse00 * residualY + inverse01 * residualZ) +
                            residualZ * (inverse01 * residualY + inverse11 * residualZ);
  chi2 = scratchChi2;
  return true;
}

bool Propagator::updateBarrel(SurfaceTrackState& state, const SurfaceMeasurement& measurement, float& chi2) noexcept
{
  if (!validateBarrelSource(state)) {
    return false;
  }
  float inverse00 = 0.f;
  float inverse01 = 0.f;
  float inverse11 = 0.f;
  if (!residualInverse(state, measurement, inverse00, inverse01, inverse11)) {
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

bool Propagator::rotateBarrel(SurfaceTrackState& state, SurfaceTrackParameters& linRef, float targetAlpha, float bz) noexcept
{
  if (!validateBarrelSource(state)) {
    return false;
  }
  if (linRef.kind != SurfaceKind::Cylinder) {
    return false;
  }
  // Pairing requires exact referenceCoordinate/alpha equality. Parameters may
  // differ because linRef is a linearization reference.
  if (state.referenceCoordinate != linRef.referenceCoordinate) {
    return false;
  }
  if (state.alpha != linRef.alpha) {
    return false;
  }
  const float stateSnp = state.parameters[2];
  if (std::abs(stateSnp) >= 1.f) {
    return false;
  }

  SurfaceTrackState scratchState = state;
  SurfaceTrackParameters scratchRef = linRef;

  const float canonicalAlpha = std::remainder(targetAlpha, 2.f * o2::constants::math::PI);

  // Rotate the reference using its own pre-rotation snp.
  const float refSnpBefore = scratchRef.parameters[2];
  if (std::abs(refSnpBefore) >= 1.f) {
    return false;
  }
  const float delta = std::remainder(canonicalAlpha - scratchRef.alpha, 2.f * o2::constants::math::PI);
  const float sa = std::sin(delta);
  const float ca = std::cos(delta);
  const float refCsp0 = std::sqrt((1.f - refSnpBefore) * (1.f + refSnpBefore));
  if (refCsp0 * ca + refSnpBefore * sa < 0.f) {
    return false;
  }
  const float refSnpRotated = refSnpBefore * ca - refCsp0 * sa;
  if (std::abs(refSnpRotated) >= 1.f) {
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

  if (!propagateReferenceParams(scratchRef, trackX, bz)) {
    return false;
  }

  // Rotate the state using its own snp and post-rotation validity.
  const float csp = std::sqrt((1.f - stateSnp) * (1.f + stateSnp));
  if (csp * ca + stateSnp * sa < 0.f) {
    return false;
  }
  const float updatedSnp = stateSnp * ca - csp * sa;
  if (std::abs(updatedSnp) >= 1.f) {
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
  const float j5 = scratchRef.parameters[4] * bz * o2::constants::math::B2C;

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

bool Propagator::propagateBarrel(SurfaceTrackState& state, SurfaceTrackParameters& linRef, float targetX, float bz) noexcept
{
  if (!validateBarrelSource(state)) {
    return false;
  }
  if (linRef.kind != SurfaceKind::Cylinder) {
    return false;
  }
  // Pairing requires exact referenceCoordinate/alpha equality; parameters may
  // differ.
  if (state.referenceCoordinate != linRef.referenceCoordinate) {
    return false;
  }
  if (state.alpha != linRef.alpha) {
    return false;
  }

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

  if (!propagateReferenceParams(scratchRef, targetX, bz)) {
    return false;
  }
  const float snpRef1 = scratchRef.parameters[2];
  const float cspRef1 = std::sqrt((1.f - snpRef1) * (1.f + snpRef1));
  if (cspRef0 == 0.f || cspRef1 == 0.f) {
    return false;
  }

  const float kb = bz * o2::constants::math::B2C;
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

bool Propagator::shiftReferenceToMeasurementBarrel(SurfaceTrackParameters& linRef, const SurfaceMeasurement& measurement) noexcept
{
  if (linRef.kind != SurfaceKind::Cylinder) {
    return false;
  }
  SurfaceTrackParameters scratch = linRef;
  scratch.parameters[0] = measurement.frame.u;
  scratch.parameters[1] = measurement.frame.v;
  linRef = scratch;
  return true;
}

bool Propagator::predictedChi2Forward(const SurfaceTrackState& state, const SurfaceMeasurement& measurement, float& chi2) noexcept
{
  if (!validateForwardSource(state)) {
    return false;
  }
  float inverse00 = 0.f;
  float inverse01 = 0.f;
  float inverse11 = 0.f;
  if (!residualInverse(state, measurement, inverse00, inverse01, inverse11)) {
    return false;
  }
  const float residualX = measurement.frame.u - state.parameters[0];
  const float residualY = measurement.frame.v - state.parameters[1];
  const float scratchChi2 = residualX * (inverse00 * residualX + inverse01 * residualY) +
                            residualY * (inverse01 * residualX + inverse11 * residualY);
  chi2 = scratchChi2;
  return true;
}

bool Propagator::updateForward(SurfaceTrackState& state, const SurfaceMeasurement& measurement, float& chi2) noexcept
{
  if (!validateForwardSource(state)) {
    return false;
  }
  float inverse00 = 0.f;
  float inverse01 = 0.f;
  float inverse11 = 0.f;
  if (!residualInverse(state, measurement, inverse00, inverse01, inverse11)) {
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

bool Propagator::shiftReferenceToMeasurementForward(SurfaceTrackParameters& linRef, const SurfaceMeasurement& measurement) noexcept
{
  if (linRef.kind != SurfaceKind::Disk) {
    return false;
  }
  SurfaceTrackParameters scratch = linRef;
  scratch.parameters[0] = measurement.frame.u;
  scratch.parameters[1] = measurement.frame.v;
  linRef = scratch;
  return true;
}

bool Propagator::propagateForward(SurfaceTrackState& state, float targetZ, float bz) noexcept
{
  return propagateAccepted(state, targetZ, bz);
}

bool Propagator::propagateForward(SurfaceTrackState& state, SurfaceTrackParameters& linRef,
                                  float targetZ, float bz) noexcept
{
  return propagateAccepted(state, linRef, targetZ, bz);
}

} // namespace o2::itsmft::tracking
