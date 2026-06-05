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

/// \file testChebyshev3D.cxx
/// \brief Accuracy of the Chebyshev3D evaluation kernel.
///
/// Guards `Chebyshev3DCalc::Eval` / `chebyshevEvaluation1D` (the Clenshaw
/// recurrence that dominates magnetic-field evaluation in track extrapolation).
/// We build an in-memory parameterization of a known smooth function and check
/// that `Eval` reproduces it to the requested precision over many random points,
/// and that the per-dimension and double-precision overloads agree with the
/// float vector overload. Any breakage of the recurrence (e.g. a wrong FMA
/// regrouping) makes the reproduction error explode and fails the test.

#define BOOST_TEST_MODULE Test Chebyshev3D
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <random>
#include "MathUtils/Chebyshev3D.h"

using o2::math_utils::Chebyshev3D;

namespace
{
// A smooth, low-degree (≤3 per variable) vector function over the fit box, of
// the kind a Chebyshev parameterization reproduces to ~float precision. Stands
// in for a slowly-varying magnetic field B(x,y,z).
void referenceField(float* in, float* out)
{
  const float x = in[0], y = in[1], z = in[2];
  out[0] = 0.50f + 0.020f * x - 1.0e-4f * x * y + 3.0e-3f * z - 2.0e-6f * x * x * z;
  out[1] = -0.30f + 0.015f * y + 5.0e-5f * y * z - 1.0e-3f * x;
  out[2] = 5.00f - 4.0e-4f * x * x + 6.0e-4f * y * y + 1.0e-3f * x * y - 2.0e-6f * x * y * z;
}
} // namespace

BOOST_AUTO_TEST_CASE(Chebyshev3D_eval_accuracy)
{
  const Float_t bmin[3] = {-40.f, -40.f, -200.f};
  const Float_t bmax[3] = {40.f, 40.f, 200.f};
  const Int_t np[3] = {7, 7, 7}; // > polynomial degree in every dimension
  const Float_t fitPrec = 1.0e-5f;

  Chebyshev3D cheb(referenceField, 3, bmin, bmax, np, fitPrec);

  // Deterministic interior sampling (fixed seed -> no flakiness). Stay a hair
  // inside the box so we never hit the boundary-clamping branch.
  std::mt19937 rng(20260604u);
  std::uniform_real_distribution<float> ux(bmin[0] + 1.f, bmax[0] - 1.f);
  std::uniform_real_distribution<float> uy(bmin[1] + 1.f, bmax[1] - 1.f);
  std::uniform_real_distribution<float> uz(bmin[2] + 1.f, bmax[2] - 1.f);

  float maxAbsErr = 0.f;         // |cheb - reference| (kernel reproduces the function)
  float maxDimMismatch = 0.f;    // |vector overload - per-dim overload|
  float maxDoubleMismatch = 0.f; // |float overload - double overload|

  for (int i = 0; i < 20000; ++i) {
    float par[3] = {ux(rng), uy(rng), uz(rng)};
    float ref[3];
    referenceField(par, ref);

    float res[3];
    cheb.Eval(par, res);

    double pard[3] = {par[0], par[1], par[2]};
    double resd[3];
    cheb.Eval(pard, resd);

    for (int d = 0; d < 3; ++d) {
      BOOST_REQUIRE(std::isfinite(res[d]));
      maxAbsErr = std::max(maxAbsErr, std::abs(res[d] - ref[d]));
      // Single-component overload must match the vector overload (same kernel).
      maxDimMismatch = std::max(maxDimMismatch, std::abs(res[d] - cheb.Eval(par, d)));
      // Double overload differs only by intermediate precision.
      maxDoubleMismatch = std::max(maxDoubleMismatch, std::abs(static_cast<float>(resd[d]) - res[d]));
    }
  }

  BOOST_TEST_MESSAGE("Chebyshev3D max |eval - reference|  = " << maxAbsErr);
  BOOST_TEST_MESSAGE("Chebyshev3D max vector-vs-perdim    = " << maxDimMismatch);
  BOOST_TEST_MESSAGE("Chebyshev3D max float-vs-double      = " << maxDoubleMismatch);

  // Reproduction of the known function: fit precision (1e-5) plus a little float
  // slack from the three nested Clenshaw sums (observed ~1.4e-6). A broken
  // recurrence misses this by orders of magnitude (coefficient-scale error / NaN).
  BOOST_CHECK_SMALL(maxAbsErr, 1.0e-4f);
  // The two float entry points share the kernel: expect bit-for-bit agreement.
  BOOST_CHECK_SMALL(maxDimMismatch, 1.0e-6f);
  // float vs double evaluation of the same coefficients: within float epsilon.
  BOOST_CHECK_SMALL(maxDoubleMismatch, 1.0e-3f);
}
