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

// M5d covariance-validity correction (doc/decisions/0008-native-refit-activation.md,
// covariance-fault-localization investigation): focused, deterministic
// regression coverage for sanitizeCovariance() (SurfaceTrackState.h) and
// its eight call sites (barrel rotate/propagate x2 overloads/update, forward
// propagation x2 overloads/update). Several fixtures below reproduce a
// real captured production failure verbatim (exact state/covariance/
// measurement values from a checksummed replay of the
// pp-20ev-run303000-seed20260716-daily20260717 fixture, candidate keys
// "13,6,6,5,4,9,5" (ITS) and "68,71,73,67,72,73,62,76,80,-1" (MFT)) rather
// than a synthetic approximation, per the covariance-fault-localization
// investigation's minimal-reproducer design.

#define BOOST_TEST_MODULE ITSMFTCovarianceSanitization
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "ITSMFTTracking/Propagator.h"

#include "ITSMFTTracking/detail/SurfaceStateOperations.h"
#include "ITSMFTTracking/SurfaceTrackState.h"
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/TrackParametrization.h"

namespace
{
using namespace o2::itsmft::tracking;

bool allDiagonalsNonNegative(const SurfaceTrackState& state)
{
  for (uint8_t i = 0; i < 5; ++i) {
    if (state.covariance[packedCovarianceIndex(i, i)] < 0.f) {
      return false;
    }
  }
  return true;
}

// Returns the magnitude of the worst pairwise-correlation violation found
// (0 if none), i.e. max(0, |c_ij|/sqrt(c_ii*c_jj) - 1) over every off-diagonal
// pair. Callers with a non-negative diagonal already established can compare
// this against a small float tolerance.
float maxCorrelationViolation(const SurfaceTrackState& state)
{
  float worst = 0.f;
  for (uint8_t i = 0; i < 5; ++i) {
    for (uint8_t j = 0; j < i; ++j) {
      const float dii = state.covariance[packedCovarianceIndex(i, i)];
      const float djj = state.covariance[packedCovarianceIndex(j, j)];
      if (dii <= 0.f || djj <= 0.f) {
        continue;
      }
      const float rho = state.covariance[packedCovarianceIndex(i, j)] / std::sqrt(dii * djj);
      worst = std::max(worst, std::abs(rho) - 1.f);
    }
  }
  return worst;
}

// The DECLARED invariant sanitizeCovariance() (SurfaceTrackState.h)
// establishes -- non-negative diagonals and no individual pairwise
// correlation exceeding unity -- and nothing more. This is deliberately NOT
// a full positive-semi-definite check (that would additionally require,
// e.g., every leading principal minor non-negative / every eigenvalue
// non-negative): the doc comment on sanitizeCovariance() proves with a real
// captured counter-example that pairwise-valid does not imply full PSD, and
// this codebase does not claim otherwise. A test asserting full PSD here
// would be testing an invariant the production code does not establish.
bool covarianceSatisfiesDeclaredInvariant(const SurfaceTrackState& state, float tolerance = 1.e-3f)
{
  return allDiagonalsNonNegative(state) && maxCorrelationViolation(state) <= tolerance;
}

bool closeTo(float a, float b, float absTol = 5.e-4f, float relTol = 2.e-3f)
{
  const float diff = std::fabs(a - b);
  return diff <= absTol || diff <= relTol * std::fabs(b);
}

template <typename T>
bool bitEqual(const T& lhs, const T& rhs)
{
  return std::memcmp(&lhs, &rhs, sizeof(T)) == 0;
}
} // namespace

// --- 1. sanitizeCovariance() itself: the core rule, in isolation. ----------

BOOST_AUTO_TEST_CASE(SanitizeCovarianceAbsNegativeDiagonal)
{
  SurfaceTrackState state{};
  state.covariance[packedCovarianceIndex(0, 0)] = -0.25f;
  state.covariance[packedCovarianceIndex(1, 1)] = 0.5f;
  state.covariance[packedCovarianceIndex(2, 2)] = 0.5f;
  state.covariance[packedCovarianceIndex(3, 3)] = 0.5f;
  state.covariance[packedCovarianceIndex(4, 4)] = 0.5f;
  const float maxDiagonal[5] = {1.f, 1.f, 1.f, 1.f, 1.f};
  sanitizeCovariance(state, maxDiagonal);
  BOOST_CHECK_CLOSE(state.covariance[packedCovarianceIndex(0, 0)], 0.25f, 1e-4f);
  BOOST_CHECK(allDiagonalsNonNegative(state));
}

BOOST_AUTO_TEST_CASE(SanitizeCovarianceClampsOverRangeAndRescalesOffDiagonal)
{
  // Pass 2 (pairwise correlation clamp) runs after pass 1 and would itself
  // touch an off-diagonal whose implied correlation, computed from the
  // POST-pass-1 (already range-clamped) diagonals, still exceeds 1 -- so
  // this fixture is deliberately chosen so pass 1's own rescale already
  // brings every off-diagonal within pass 2's bound too, isolating pass 1
  // in observable behavior (SanitizeCovarianceClampsOverRangeToCauchySchwarzBound
  // below exercises pass 2 specifically, including its interaction with an
  // already-pass-1-clamped diagonal).
  SurfaceTrackState state{};
  state.covariance[packedCovarianceIndex(0, 0)] = 4.f; // 4x the max below.
  state.covariance[packedCovarianceIndex(1, 0)] = 1.f; // Shares row/column 0.
  state.covariance[packedCovarianceIndex(2, 0)] = 0.4f;
  state.covariance[packedCovarianceIndex(1, 1)] = 0.3f;
  state.covariance[packedCovarianceIndex(2, 2)] = 0.3f;
  state.covariance[packedCovarianceIndex(3, 3)] = 0.3f;
  state.covariance[packedCovarianceIndex(4, 4)] = 0.3f;
  const float maxDiagonal[5] = {1.f, 1.f, 1.f, 1.f, 1.f};
  sanitizeCovariance(state, maxDiagonal);
  // scale = sqrt(max/old) = sqrt(1/4) = 0.5.
  BOOST_CHECK_CLOSE(state.covariance[packedCovarianceIndex(0, 0)], 1.f, 1e-4f);
  BOOST_CHECK_CLOSE(state.covariance[packedCovarianceIndex(1, 0)], 0.5f, 1e-4f);
  BOOST_CHECK_CLOSE(state.covariance[packedCovarianceIndex(2, 0)], 0.2f, 1e-4f);
  // Untouched entries not sharing the clamped row/column.
  BOOST_CHECK_CLOSE(state.covariance[packedCovarianceIndex(1, 1)], 0.3f, 1e-4f);
  // Confirms pass 2 really was a no-op for this fixture, not merely unlucky
  // arithmetic: every pairwise correlation is within bound.
  BOOST_CHECK_LE(maxCorrelationViolation(state), 1.e-4f);
}

BOOST_AUTO_TEST_CASE(SanitizeCovarianceClampsOverRangeToCauchySchwarzBound)
{
  // Pass 2 in isolation (diagonals already within maxDiagonal, so pass 1 is
  // a no-op here): an off-diagonal whose magnitude implies |correlation|>1
  // is clamped to exactly sqrt(c_ii*c_jj), sign preserved; a pair already
  // within bound is untouched.
  SurfaceTrackState state{};
  state.covariance[packedCovarianceIndex(0, 0)] = 4.f;
  state.covariance[packedCovarianceIndex(1, 1)] = 9.f;
  state.covariance[packedCovarianceIndex(1, 0)] = -100.f; // |rho| = 100/sqrt(4*9) = 16.67, deliberately over 1.
  state.covariance[packedCovarianceIndex(2, 2)] = 4.f;
  state.covariance[packedCovarianceIndex(2, 0)] = 3.f; // |rho| = 3/sqrt(4*4) = 0.75, already within bound.
  state.covariance[packedCovarianceIndex(3, 3)] = 1.f;
  state.covariance[packedCovarianceIndex(4, 4)] = 1.f;
  const float maxDiagonal[5] = {1.e30f, 1.e30f, 1.e30f, 1.e30f, 1.e30f}; // Effectively unreachable: isolates pass 2.
  sanitizeCovariance(state, maxDiagonal);
  BOOST_CHECK_CLOSE(state.covariance[packedCovarianceIndex(1, 0)], -6.f, 1e-4f); // -sqrt(4*9) = -6.
  BOOST_CHECK_CLOSE(state.covariance[packedCovarianceIndex(2, 0)], 3.f, 1e-4f);  // Untouched: already within bound.
  BOOST_CHECK_LE(maxCorrelationViolation(state), 1.e-4f);
}

BOOST_AUTO_TEST_CASE(SanitizeCovariancePreservesSymmetryByConstruction)
{
  // Packed lower-triangular storage: only one entry exists per (row,column)
  // pair, so "symmetry" is a representation invariant, not a check --
  // packedCovarianceIndex(i,j) == packedCovarianceIndex(j,i) is exercised
  // directly by every read/write sanitizeCovariance performs. Diagonals are
  // set generously large (relative to the off-diagonal under test) so pass
  // 2's correlation clamp is a no-op here and does not confound the
  // symmetry check with a legitimate clamp.
  SurfaceTrackState state{};
  state.covariance[packedCovarianceIndex(1, 1)] = 100.f;
  state.covariance[packedCovarianceIndex(3, 3)] = 100.f;
  state.covariance[packedCovarianceIndex(3, 1)] = 5.f;
  const float maxDiagonal[5] = {1.e30f, 1.e30f, 1.e30f, 1.e30f, 1.e30f};
  sanitizeCovariance(state, maxDiagonal);
  BOOST_CHECK_EQUAL(packedCovarianceIndex(1, 3), packedCovarianceIndex(3, 1));
  BOOST_CHECK_CLOSE(state.covariance[packedCovarianceIndex(1, 3)], 5.f, 1e-4f);
}

// --- 2. ITS legB reproducer: detail::barrel::update() on the exact captured real --
// prior state/covariance and measurement (candidate "13,6,6,5,4,9,5", hit 5)
// that produced OperationFailureReason::MaterialFailure /
// MaterialFailureReason::InvalidCovariance before this correction (posterior
// Q2Pt-Q2Pt diagonal = -0.032802999, real production value, captured
// verbatim from the checksummed 20-event replay).

BOOST_AUTO_TEST_CASE(ITSLegBReproducerNowSanitizesToValidCovariance)
{
  SurfaceTrackState state{};
  state.kind = SurfaceKind::Cylinder;
  state.referenceCoordinate = 3.76323366f;
  state.alpha = -0.12901926f;
  state.parameters[0] = 0.642236829f;
  state.parameters[1] = -6.11814785f;
  state.parameters[2] = 0.167980343f;
  state.parameters[3] = -1.58871007f;
  state.parameters[4] = 1.2842629f;
  state.absCharge = 1;
  state.pid = o2::track::PID::Pion;
  const float cov[15] = {
    0.0615117364f, -0.0162716303f, 0.00781002454f, -0.00648284703f, 0.00164899346f, 0.000680086901f,
    0.000152464694f, -0.000262976653f, -1.178005e-05f, 1.45164713e-05f,
    -0.22814776f, 0.0546577908f, 0.0237723477f, -0.000194984852f, 0.822642863f};
  for (int i = 0; i < 15; ++i) {
    state.covariance[i] = cov[i];
  }

  SurfaceMeasurement meas{};
  meas.frame.u = 0.633100867f;
  meas.frame.v = -6.10807085f;
  meas.covariance.uu = 1.18710993e-07f;
  meas.covariance.uv = 0.f;
  meas.covariance.vv = 3.60069805e-07f;

  float chi2 = 0.f;
  OperationFailureReason reason{};
  const bool ok = detail::barrel::update(state, meas, chi2, reason);

  BOOST_REQUIRE(ok);
  BOOST_CHECK(allDiagonalsNonNegative(state));
  BOOST_CHECK_CLOSE(state.covariance[packedCovarianceIndex(4, 4)], 0.0328048468f, 5.f); // sign-flipped, matches production magnitude within float tolerance.
}

// --- 3. MFT reproducer: detail::forward::update() on the exact captured real ------
// prior state/covariance and measurement (candidate
// "68,71,73,67,72,73,62,76,80,-1", legB, hit 3) that produced a
// Q2Pt-Q2Pt diagonal of -52.064167 (real production value) before this
// correction.

BOOST_AUTO_TEST_CASE(MFTReproducerNowSanitizesToValidCovariance)
{
  SurfaceTrackState state{};
  state.kind = SurfaceKind::Disk;
  state.referenceCoordinate = -67.6889038f;
  state.alpha = 0.f;
  state.parameters[0] = -3.40663648f;
  state.parameters[1] = -3.04799104f;
  state.parameters[2] = -2.40926218f;
  state.parameters[3] = -15.2632132f;
  state.parameters[4] = -0.0805783421f;
  state.absCharge = 1;
  state.pid = o2::track::PID::Pion;
  const float cov[15] = {
    5.45968469e-05f, 7.92145147e-05f, 2.34508334e-05f, 0.000361069426f, 9.60996113e-05f, 0.00121101411f,
    0.00140110496f, 0.0018511567f, 0.00489055132f, 0.0514357649f,
    0.117691882f, 0.0217776336f, 0.452787817f, 1.21933973f, 168.588654f};
  for (int i = 0; i < 15; ++i) {
    state.covariance[i] = cov[i];
  }

  SurfaceMeasurement meas{};
  meas.frame.u = -3.4059999f;
  meas.frame.v = -3.04678011f;
  meas.covariance.uu = 4.4239976e-05f;
  meas.covariance.uv = 0.f;
  meas.covariance.vv = 0.000105412393f;

  float chi2 = 0.f;
  OperationFailureReason reason{};
  const bool ok = detail::forward::update(state, meas, chi2, reason);

  BOOST_REQUIRE(ok);
  BOOST_CHECK(allDiagonalsNonNegative(state));
}

// --- 4. Large-step propagation invariant: detail::barrel::propagate(state, linRef, --
// ...) on the exact captured real inputs that fed the ITS legB reproducer
// above (the immediately preceding hit) must itself leave the covariance
// invariant satisfied before the next update() ever runs. The raw off-
// diagonal transport for this large (~-15.5cm) step makes THREE pairwise
// correlations simultaneously exceed 1 in magnitude -- (Y,Snp), (Y,Q2Pt),
// (Snp,Q2Pt) -- confirmed against the real captured (pre-correction)
// production values: c(Y,Y)=0.0615117364, c(Y,Q2Pt)=-0.22814776,
// c(Q2Pt,Q2Pt)=0.822642863 give rho(Y,Q2Pt) = -0.22814776 /
// sqrt(0.0615117364*0.822642863) = -1.0142..., i.e. |rho|>1 while every
// diagonal individually stays positive and unremarkable -- exactly the
// precondition the covariance-fault-localization investigation traced.
// sanitizeCovariance()'s pass 2 must repair all three before this function
// returns, and the immediately following measurement update (same real
// captured measurement) must then observe the DECLARED invariant on its
// own committed output too -- not merely "not obviously wrong": pass 2
// alone measurably shrinks (from -0.0328 to a much smaller magnitude) but
// does not eliminate the negative diagonal the update's own naive Kalman
// subtraction still produces from an otherwise-repaired input (see
// sanitizeCovariance()'s own doc comment for the full empirical accounting
// of this), so pass 1 (diagonal abs) remains load-bearing for the
// observable, committed result even with pass 2 active.
BOOST_AUTO_TEST_CASE(LargeStepPropagationRepairsCorrelationBeforeUpdate)
{
  SurfaceTrackState state{};
  state.kind = SurfaceKind::Cylinder;
  state.referenceCoordinate = 19.2192478f;
  state.alpha = -0.12901926f;
  state.parameters[0] = 3.03678966f;
  state.parameters[1] = -30.9622726f;
  state.parameters[2] = 0.138186395f;
  state.parameters[3] = -1.58871007f;
  state.parameters[4] = 1.2842629f;
  state.absCharge = 1;
  state.pid = o2::track::PID::Pion;
  const float cov[15] = {
    1.98605832e-07f, -7.50241043e-08f, 2.4906555e-07f, -9.13163856e-09f, -3.06560999e-08f, 1.98362322e-05f,
    3.80933152e-09f, -3.28565477e-08f, -7.25654581e-06f, 1.45164713e-05f,
    -1.76052566e-07f, -8.04973183e-07f, 0.00468764221f, -0.000194984852f, 0.822642863f};
  for (int i = 0; i < 15; ++i) {
    state.covariance[i] = cov[i];
  }

  SurfaceTrackParameters linRef{};
  linRef.kind = SurfaceKind::Cylinder;
  linRef.referenceCoordinate = 19.2192478f;
  linRef.alpha = -0.12901926f;
  linRef.parameters[0] = 3.03678894f;
  linRef.parameters[1] = -30.962265f;
  linRef.parameters[2] = 0.137899101f;
  linRef.parameters[3] = -1.58717895f;
  linRef.parameters[4] = 1.21498108f;

  const float targetX = 3.76323366f;
  const float bz = 5.00675011f;
  OperationFailureReason reason{};
  const bool ok = detail::barrel::propagate(state, linRef, targetX, bz, reason);

  BOOST_REQUIRE(ok);
  BOOST_CHECK(covarianceSatisfiesDeclaredInvariant(state));
  // Diagonals themselves are untouched by pass 2 (only off-diagonals move):
  // still match the real captured production values exactly.
  BOOST_CHECK(closeTo(state.covariance[packedCovarianceIndex(0, 0)], 0.0615117364f));
  BOOST_CHECK(closeTo(state.covariance[packedCovarianceIndex(4, 4)], 0.822642863f));
  // The (Y,Q2Pt) pair is now repaired to exactly touch (not exceed) the
  // Cauchy-Schwarz bound, rather than the real pre-correction production
  // value of -0.22814776 (|rho|=1.0142).
  const float expectedC40 = -std::sqrt(state.covariance[packedCovarianceIndex(0, 0)] * state.covariance[packedCovarianceIndex(4, 4)]);
  BOOST_CHECK(closeTo(state.covariance[packedCovarianceIndex(4, 0)], expectedC40));
  BOOST_CHECK_LE(maxCorrelationViolation(state), 1.e-3f);

  // The following update (same real captured measurement) must observe the
  // declared invariant on its own committed output.
  SurfaceMeasurement meas{};
  meas.frame.u = 0.633100867f;
  meas.frame.v = -6.10807085f;
  meas.covariance.uu = 1.18710993e-07f;
  meas.covariance.uv = 0.f;
  meas.covariance.vv = 3.60069805e-07f;
  float chi2 = 0.f;
  OperationFailureReason updateReason{};
  BOOST_REQUIRE(detail::barrel::update(state, meas, chi2, updateReason));
  BOOST_CHECK(covarianceSatisfiesDeclaredInvariant(state));
}

// --- 5. Every rotate/propagate/update independently sanitizes, both -------
// families. Each case below uses a deliberate zero-step (rotate: delta==0;
// propagate: dx/dz==0) or an otherwise-trivial transport so the operation's
// own transform is a documented no-op/identity on the covariance, isolating
// the sanitization call itself as the only thing that can explain a clamped
// result -- rather than depending on a from-scratch derivation of each
// operation's own Jacobian to predict a non-trivial expected output.

SurfaceTrackState makeOverRangeBarrelState()
{
  SurfaceTrackState state{};
  state.kind = SurfaceKind::Cylinder;
  state.referenceCoordinate = 4.f;
  state.alpha = 0.3f;
  state.parameters[0] = 1.25f;
  state.parameters[1] = -0.75f;
  state.parameters[2] = 0.2f;
  state.parameters[3] = -0.35f;
  state.parameters[4] = 0.05f; // Small |Q2Pt| so Q2Pt-Q2Pt max isn't reached trivially by other tests.
  state.absCharge = 1;
  state.pid = o2::track::PID::Pion;
  state.covariance[packedCovarianceIndex(0, 0)] = 50.f * o2::track::kCY2max; // Deliberately over range.
  state.covariance[packedCovarianceIndex(1, 1)] = 0.01f;
  state.covariance[packedCovarianceIndex(2, 2)] = 0.01f;
  state.covariance[packedCovarianceIndex(3, 3)] = 0.01f;
  state.covariance[packedCovarianceIndex(4, 4)] = 0.01f;
  return state;
}

BOOST_AUTO_TEST_CASE(BarrelRotateSanitizesOnZeroDeltaTrivialStep)
{
  SurfaceTrackState state = makeOverRangeBarrelState();
  OperationFailureReason reason{};
  const bool ok = detail::barrel::rotate(state, state.alpha, reason); // delta == 0: ratio == 1, transform is identity.
  BOOST_REQUIRE(ok);
  BOOST_CHECK_CLOSE(state.covariance[packedCovarianceIndex(0, 0)], o2::track::kCY2max, 1e-3f);
}

BOOST_AUTO_TEST_CASE(BarrelPropagateSanitizesOnZeroDxTrivialStep)
{
  SurfaceTrackState state = makeOverRangeBarrelState();
  OperationFailureReason reason{};
  const bool ok = detail::barrel::propagate(state, state.referenceCoordinate, 0.5f, reason); // dx == 0: early-return path.
  BOOST_REQUIRE(ok);
  BOOST_CHECK_CLOSE(state.covariance[packedCovarianceIndex(0, 0)], o2::track::kCY2max, 1e-3f);
}

BOOST_AUTO_TEST_CASE(BarrelUpdateSanitizesReproducer)
{
  // Same fixture and assertion as ITSLegBReproducerNowSanitizesToValidCovariance
  // above; kept as a separate, minimally-named case so "update sanitizes" is
  // independently visible in the test list without relying on the reproducer
  // test's name to convey it.
  SurfaceTrackState state{};
  state.kind = SurfaceKind::Cylinder;
  state.referenceCoordinate = 3.76323366f;
  state.alpha = -0.12901926f;
  state.parameters[0] = 0.642236829f;
  state.parameters[1] = -6.11814785f;
  state.parameters[2] = 0.167980343f;
  state.parameters[3] = -1.58871007f;
  state.parameters[4] = 1.2842629f;
  state.absCharge = 1;
  state.pid = o2::track::PID::Pion;
  const float cov[15] = {
    0.0615117364f, -0.0162716303f, 0.00781002454f, -0.00648284703f, 0.00164899346f, 0.000680086901f,
    0.000152464694f, -0.000262976653f, -1.178005e-05f, 1.45164713e-05f,
    -0.22814776f, 0.0546577908f, 0.0237723477f, -0.000194984852f, 0.822642863f};
  for (int i = 0; i < 15; ++i) {
    state.covariance[i] = cov[i];
  }
  SurfaceMeasurement meas{};
  meas.frame.u = 0.633100867f;
  meas.frame.v = -6.10807085f;
  meas.covariance.uu = 1.18710993e-07f;
  meas.covariance.vv = 3.60069805e-07f;
  float chi2 = 0.f;
  OperationFailureReason reason{};
  BOOST_REQUIRE(detail::barrel::update(state, meas, chi2, reason));
  BOOST_CHECK(allDiagonalsNonNegative(state));
}

BOOST_AUTO_TEST_CASE(BarrelLinRefRotateSanitizesOnZeroDeltaTrivialStep)
{
  SurfaceTrackState state = makeOverRangeBarrelState();
  SurfaceTrackParameters linRef{};
  linRef.kind = SurfaceKind::Cylinder;
  linRef.referenceCoordinate = state.referenceCoordinate;
  linRef.alpha = state.alpha;
  for (int i = 0; i < 5; ++i) {
    linRef.parameters[i] = state.parameters[i];
  }
  OperationFailureReason reason{};
  const bool ok = detail::barrel::rotate(state, linRef, state.alpha, 0.5f, reason);
  BOOST_REQUIRE(ok);
  BOOST_CHECK_CLOSE(state.covariance[packedCovarianceIndex(0, 0)], o2::track::kCY2max, 1e-3f);
}

BOOST_AUTO_TEST_CASE(BarrelLinRefPropagateSanitizesLargeStep)
{
  // Same fixture and assertion as LargeStepPropagationPreservesInvariantBeforeUpdate
  // above; kept as a separate, minimally-named case for the same reason as
  // BarrelUpdateSanitizesReproducer.
  SurfaceTrackState state{};
  state.kind = SurfaceKind::Cylinder;
  state.referenceCoordinate = 19.2192478f;
  state.alpha = -0.12901926f;
  state.parameters[0] = 3.03678966f;
  state.parameters[1] = -30.9622726f;
  state.parameters[2] = 0.138186395f;
  state.parameters[3] = -1.58871007f;
  state.parameters[4] = 1.2842629f;
  state.absCharge = 1;
  state.pid = o2::track::PID::Pion;
  const float cov[15] = {
    1.98605832e-07f, -7.50241043e-08f, 2.4906555e-07f, -9.13163856e-09f, -3.06560999e-08f, 1.98362322e-05f,
    3.80933152e-09f, -3.28565477e-08f, -7.25654581e-06f, 1.45164713e-05f,
    -1.76052566e-07f, -8.04973183e-07f, 0.00468764221f, -0.000194984852f, 0.822642863f};
  for (int i = 0; i < 15; ++i) {
    state.covariance[i] = cov[i];
  }
  SurfaceTrackParameters linRef{};
  linRef.kind = SurfaceKind::Cylinder;
  linRef.referenceCoordinate = 19.2192478f;
  linRef.alpha = -0.12901926f;
  linRef.parameters[0] = 3.03678894f;
  linRef.parameters[1] = -30.962265f;
  linRef.parameters[2] = 0.137899101f;
  linRef.parameters[3] = -1.58717895f;
  linRef.parameters[4] = 1.21498108f;
  OperationFailureReason reason{};
  BOOST_REQUIRE(detail::barrel::propagate(state, linRef, 3.76323366f, 5.00675011f, reason));
  BOOST_CHECK(allDiagonalsNonNegative(state));
}

// Forward has no established diagonal-range validity bound (see
// kForwardMaxDiagonal's own doc comment, ForwardSurfaceStateOperations.cxx:
// legacy MFT's fitting engine has no covariance-sanitization mechanism at
// all, so forward's range-clamp sub-pass is deliberately disabled pending a
// separate design decision), so an over-range diagonal is no longer a valid
// forward wiring probe. A deliberately over-correlated off-diagonal pair is:
// the pairwise correlation bound is mathematically universal (Cauchy-
// Schwarz), not a detector-specific bound, and is fully active for forward.
SurfaceTrackState makeOverCorrelatedForwardState()
{
  SurfaceTrackState state{};
  state.kind = SurfaceKind::Disk;
  state.referenceCoordinate = -40.f;
  state.alpha = 0.f;
  state.parameters[0] = 1.f;
  state.parameters[1] = -1.f;
  state.parameters[2] = 0.1f;
  state.parameters[3] = -2.f;
  state.parameters[4] = 0.05f;
  state.absCharge = 1;
  state.pid = o2::track::PID::Pion;
  state.covariance[packedCovarianceIndex(0, 0)] = 4.f;
  state.covariance[packedCovarianceIndex(1, 0)] = 100.f; // |rho(X,Y)| = 100/sqrt(4*1) = 50, deliberately over 1.
  state.covariance[packedCovarianceIndex(1, 1)] = 1.f;
  state.covariance[packedCovarianceIndex(2, 2)] = 0.01f;
  state.covariance[packedCovarianceIndex(3, 3)] = 0.01f;
  state.covariance[packedCovarianceIndex(4, 4)] = 0.01f;
  return state;
}

BOOST_AUTO_TEST_CASE(ForwardPropagateSanitizesOnZeroDzTrivialStep)
{
  SurfaceTrackState state = makeOverCorrelatedForwardState();
  OperationFailureReason reason{};
  const bool ok = Propagator::propagateToReference(state, state.referenceCoordinate, 0.5f, reason);
  BOOST_REQUIRE(ok);
  BOOST_CHECK(covarianceSatisfiesDeclaredInvariant(state));
  BOOST_CHECK_CLOSE(state.covariance[packedCovarianceIndex(1, 0)], 2.f, 1e-3f); // sqrt(4*1) = 2, sign-preserved.
}

BOOST_AUTO_TEST_CASE(ForwardLinRefPropagateSanitizesOnZeroDzTrivialStep)
{
  SurfaceTrackState state = makeOverCorrelatedForwardState();
  SurfaceTrackParameters linRef{};
  linRef.kind = SurfaceKind::Disk;
  linRef.referenceCoordinate = state.referenceCoordinate;
  for (int i = 0; i < 5; ++i) {
    linRef.parameters[i] = state.parameters[i];
  }
  OperationFailureReason reason{};
  const bool ok = Propagator::propagateToReference(state, linRef, state.referenceCoordinate, 0.5f, reason);
  BOOST_REQUIRE(ok);
  BOOST_CHECK(covarianceSatisfiesDeclaredInvariant(state));
  BOOST_CHECK_CLOSE(state.covariance[packedCovarianceIndex(1, 0)], 2.f, 1e-3f); // sqrt(4*1) = 2, sign-preserved.
}

BOOST_AUTO_TEST_CASE(ForwardUpdateSanitizesReproducer)
{
  SurfaceTrackState state{};
  state.kind = SurfaceKind::Disk;
  state.referenceCoordinate = -67.6889038f;
  state.alpha = 0.f;
  state.parameters[0] = -3.40663648f;
  state.parameters[1] = -3.04799104f;
  state.parameters[2] = -2.40926218f;
  state.parameters[3] = -15.2632132f;
  state.parameters[4] = -0.0805783421f;
  state.absCharge = 1;
  state.pid = o2::track::PID::Pion;
  const float cov[15] = {
    5.45968469e-05f, 7.92145147e-05f, 2.34508334e-05f, 0.000361069426f, 9.60996113e-05f, 0.00121101411f,
    0.00140110496f, 0.0018511567f, 0.00489055132f, 0.0514357649f,
    0.117691882f, 0.0217776336f, 0.452787817f, 1.21933973f, 168.588654f};
  for (int i = 0; i < 15; ++i) {
    state.covariance[i] = cov[i];
  }
  SurfaceMeasurement meas{};
  meas.frame.u = -3.4059999f;
  meas.frame.v = -3.04678011f;
  meas.covariance.uu = 4.4239976e-05f;
  meas.covariance.vv = 0.000105412393f;
  float chi2 = 0.f;
  OperationFailureReason reason{};
  BOOST_REQUIRE(detail::forward::update(state, meas, chi2, reason));
  BOOST_CHECK(allDiagonalsNonNegative(state));
}

// --- 6. preflightValidate remains strict: a deliberately malformed --------
// *externally supplied* state (never touched by propagate/rotate/update) is
// still rejected by correctForMaterial's preflight, proving the fix does not
// weaken or bypass that check -- it only ensures the Propagator's own
// internal callers never hand it an invalid state in normal operation.

BOOST_AUTO_TEST_CASE(MalformedExternalBarrelCovarianceStillRejectedByPreflight)
{
  SurfaceTrackState state{};
  state.kind = SurfaceKind::Cylinder;
  state.referenceCoordinate = 4.f;
  state.alpha = 0.3f;
  state.parameters[0] = 1.25f;
  state.parameters[1] = -0.75f;
  state.parameters[2] = 0.2f;
  state.parameters[3] = -0.35f;
  state.parameters[4] = 0.8f;
  state.absCharge = 1;
  state.pid = o2::track::PID::Pion;
  for (uint8_t i = 0; i < 5; ++i) {
    state.covariance[packedCovarianceIndex(i, i)] = 0.01f;
  }
  state.covariance[packedCovarianceIndex(4, 4)] = -0.01f; // Deliberately invalid, constructed directly.

  const material::IntegratedMaterialBudget budget{0.01f, 0.05f};
  const auto result = detail::barrel::correctForMaterial(state, budget, material::MaterialTraversalDirection::AlongMomentum);
  BOOST_CHECK(!result.ok());
  BOOST_CHECK(result.failure == material::MaterialFailureReason::InvalidCovariance);
}

BOOST_AUTO_TEST_CASE(MalformedExternalForwardCovarianceStillRejectedByPreflight)
{
  SurfaceTrackState state{};
  state.kind = SurfaceKind::Disk;
  state.referenceCoordinate = -40.f;
  state.alpha = 0.f;
  state.parameters[0] = 1.f;
  state.parameters[1] = -1.f;
  state.parameters[2] = 0.1f;
  state.parameters[3] = -2.f;
  state.parameters[4] = 0.05f;
  state.absCharge = 1;
  state.pid = o2::track::PID::Pion;
  for (uint8_t i = 0; i < 5; ++i) {
    state.covariance[packedCovarianceIndex(i, i)] = 0.01f;
  }
  state.covariance[packedCovarianceIndex(2, 2)] = -0.01f; // Deliberately invalid, constructed directly.

  const material::IntegratedMaterialBudget budget{0.01f, 0.05f};
  const auto result = detail::forward::correctForMaterial(state, budget, material::MaterialTraversalDirection::AlongMomentum);
  BOOST_CHECK(!result.ok());
  BOOST_CHECK(result.failure == material::MaterialFailureReason::InvalidCovariance);
}

// --- 7. Operation failure remains transactional: a failing rotate/propagate
// call must leave the input state byte-for-byte unchanged -- the
// new sanitization call must never run (and never partially mutate state)
// on a failure path.

BOOST_AUTO_TEST_CASE(FailingBarrelRotateLeavesStateUnchanged)
{
  SurfaceTrackState state{};
  state.kind = SurfaceKind::Cylinder;
  state.referenceCoordinate = 4.f;
  state.alpha = 0.3f;
  state.parameters[0] = 1.25f;
  state.parameters[1] = -0.75f;
  state.parameters[2] = 1.5f; // |Snp| >= 1: rotate must reject before touching anything.
  state.parameters[3] = -0.35f;
  state.parameters[4] = 0.8f;
  state.absCharge = 1;
  state.pid = o2::track::PID::Pion;
  for (uint8_t i = 0; i < 5; ++i) {
    state.covariance[packedCovarianceIndex(i, i)] = 0.01f;
  }
  const SurfaceTrackState original = state;

  OperationFailureReason reason{};
  const bool ok = detail::barrel::rotate(state, state.alpha + 3.0f, reason); // Large rotation: local direction inversion.

  BOOST_CHECK(!ok);
  BOOST_CHECK(bitEqual(state, original));
}
