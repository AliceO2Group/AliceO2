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

#define BOOST_TEST_MODULE ITSMFTTrackingTripletFitting
#include <boost/test/unit_test.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

#include "ITSMFTTracking/TripletFitting.h"

using namespace o2::itsmft::tracking;

namespace
{

constexpr double Radius = 50.;
constexpr double TanLambda = 0.4;

GlobalCovariance3F makeCovariance()
{
  // Positive definite, non-axis-aligned covariance in cm^2.
  return {4.e-6f, 0.8e-6f, -0.4e-6f, 3.e-6f, 0.3e-6f, 5.e-6f};
}

GlobalMeasurement makeMeasurement(float x, float y, float z,
                                  GlobalCovariance3F covariance = makeCovariance())
{
  GlobalMeasurement measurement{};
  measurement.position = {x, y, z};
  measurement.covariance = covariance;
  return measurement;
}

std::array<GlobalMeasurement, 3> makeHelixMeasurements()
{
  const std::array<double, 3> angles{0.1, 0.16, 0.25};
  std::array<GlobalMeasurement, 3> measurements{};
  for (std::size_t i = 0; i < measurements.size(); ++i) {
    measurements[i].position = {static_cast<float>(3. + Radius * std::cos(angles[i])),
                                static_cast<float>(-2. + Radius * std::sin(angles[i])),
                                static_cast<float>(1.5 + Radius * angles[i] * TanLambda)};
    measurements[i].covariance = makeCovariance();
  }
  return measurements;
}

std::array<GlobalMeasurement, 4> makeAdjacentHelixMeasurements()
{
  const std::array<double, 4> angles{0.1, 0.16, 0.25, 0.33};
  std::array<GlobalMeasurement, 4> measurements{};
  for (std::size_t i = 0; i < measurements.size(); ++i) {
    measurements[i].position = {static_cast<float>(3. + Radius * std::cos(angles[i])),
                                static_cast<float>(-2. + Radius * std::sin(angles[i])),
                                static_cast<float>(1.5 + Radius * angles[i] * TanLambda)};
    measurements[i].covariance = makeCovariance();
  }
  return measurements;
}

std::array<TripletFitFactor, 2> fitAdjacentFactors(
  const std::array<GlobalMeasurement, 4>& measurements)
{
  const std::array<GlobalMeasurement, 3> first{
    measurements[0], measurements[1], measurements[2]};
  const std::array<GlobalMeasurement, 3> second{
    measurements[1], measurements[2], measurements[3]};
  std::array<TripletFitFactor, 2> factors{};
  BOOST_REQUIRE(makeTripletFitFactor(first, factors[0]));
  BOOST_REQUIRE(makeTripletFitFactor(second, factors[1]));
  return factors;
}

GlobalCovariance3F rotateCovarianceAroundZ(const GlobalCovariance3F& covariance,
                                           double angle)
{
  const double cosine = std::cos(angle);
  const double sine = std::sin(angle);
  return {static_cast<float>(cosine * cosine * covariance.xx - 2. * sine * cosine * covariance.xy + sine * sine * covariance.yy),
          static_cast<float>(sine * cosine * covariance.xx + (cosine * cosine - sine * sine) * covariance.xy -
                             sine * cosine * covariance.yy),
          static_cast<float>(cosine * covariance.xz - sine * covariance.yz),
          static_cast<float>(sine * sine * covariance.xx + 2. * sine * cosine * covariance.xy + cosine * cosine * covariance.yy),
          static_cast<float>(sine * covariance.xz + cosine * covariance.yz),
          covariance.zz};
}

void checkClose(double actual, double expected, double relativeTolerance, double absoluteTolerance = 0.)
{
  BOOST_CHECK_SMALL(actual - expected,
                    std::max(absoluteTolerance, relativeTolerance * std::max(std::abs(actual), std::abs(expected))));
}

double factorCovariance(const TripletFitFactor& factor,
                        const std::array<GlobalMeasurement, 3>& measurements,
                        bool leftTheta, bool rightTheta)
{
  double covariance = 0.;
  for (std::size_t hit = 0; hit < measurements.size(); ++hit) {
    const auto& left = leftTheta ? factor.h[hit].theta : factor.h[hit].phi;
    const auto& right = rightTheta ? factor.h[hit].theta : factor.h[hit].phi;
    const auto& v = measurements[hit].covariance;
    covariance += left[0] * (v.xx * right[0] + v.xy * right[1] + v.xz * right[2]) +
                  left[1] * (v.xy * right[0] + v.yy * right[1] + v.yz * right[2]) +
                  left[2] * (v.xz * right[0] + v.yz * right[1] + v.zz * right[2]);
  }
  return covariance;
}

} // namespace

BOOST_AUTO_TEST_CASE(ExactHelixProducesAConsistentFactor)
{
  const auto measurements = makeHelixMeasurements();
  TripletFitFactor factor{};
  BOOST_REQUIRE(makeTripletFitFactor(measurements, factor));
  BOOST_REQUIRE(factor.isValid());
  const double referenceCurvature = -static_cast<double>(factor.psi.phi) / factor.rho.phi;
  const double expectedCurvature = (1. / Radius) / std::sqrt(1. + TanLambda * TanLambda);
  checkClose(referenceCurvature, expectedCurvature, 4.e-4);
  // Native float hit coordinates leave this residual after the otherwise
  // double-precision geometry calculation.
  BOOST_CHECK_SMALL(static_cast<double>(factor.psi.theta) +
                      static_cast<double>(factor.rho.theta) * referenceCurvature,
                    2.e-7);
  BOOST_CHECK_GT(factorCovariance(factor, measurements, true, true), 0.);
  BOOST_CHECK_NE(factorCovariance(factor, measurements, true, false), 0.);
}

BOOST_AUTO_TEST_CASE(AdjacentFactorsImplementEquation19ClosedForm)
{
  const GlobalCovariance3F exact{};
  const std::array<GlobalMeasurement, 4> measurements{{
    makeMeasurement(0.f, 0.f, 0.f, exact),
    makeMeasurement(1.f, 0.f, 0.f, exact),
    makeMeasurement(2.f, 0.f, 0.f, exact),
    makeMeasurement(3.f, 0.f, 0.f, exact),
  }};
  std::array<TripletFitFactor, 2> factors{};
  factors[0].psi = {1.f, 2.f};
  factors[0].rho = {1.f, 1.f};
  factors[1].psi = {3.f, 4.f};
  factors[1].rho = {1.f, 1.f};

  AdjacentTripletFitResult result{};
  BOOST_REQUIRE(fitAdjacentTripletFactors(factors[0], factors[1], measurements, {4.f, 9.f}, result));
  const double rhoKpsi = 1. / 4. + 2. / 4. + 3. / 9. + 4. / 9.;
  const double rhoKrho = 1. / 4. + 1. / 4. + 1. / 9. + 1. / 9.;
  const double psiKpsi = 1. / 4. + 4. / 4. + 9. / 9. + 16. / 9.;
  checkClose(result.curvature, -rhoKpsi / rhoKrho, 2.e-6);
  checkClose(result.curvatureVariance, 1. / rhoKrho, 2.e-6);
  checkClose(result.chi2, psiKpsi - rhoKpsi * rhoKpsi / rhoKrho, 2.e-6);
}

BOOST_AUTO_TEST_CASE(AdjacentFactorsRetainSharedHitCrossCovariance)
{
  const GlobalCovariance3F exact{};
  std::array<GlobalMeasurement, 4> measurements{{
    makeMeasurement(0.f, 0.f, 0.f, exact),
    makeMeasurement(1.f, 0.f, 0.f, {2.f, 0.5f, 0.f, 3.f, 0.f, 0.f}),
    makeMeasurement(2.f, 0.f, 0.f, exact),
    makeMeasurement(3.f, 0.f, 0.f, exact),
  }};
  std::array<TripletFitFactor, 2> factors{};
  factors[0].rho.phi = 1.f;
  factors[1].rho.phi = 1.f;
  factors[0].h[1].theta = {1.f, 2.f, 0.f};
  factors[0].h[1].phi = {-1.f, 1.f, 0.f};
  factors[1].h[0].theta = {3.f, -2.f, 0.f};
  factors[1].h[0].phi = {2.f, 4.f, 0.f};

  AdjacentTripletFitResult correlated{};
  BOOST_REQUIRE(fitAdjacentTripletFactors(factors[0], factors[1], measurements,
                                          {100.f, 100.f}, correlated));

  // Move the second factor's identical covariance contribution from shared
  // hit 1 to private hit 3. Diagonal blocks stay equal; only H V H^T's
  // cross-triplet block disappears.
  auto independentFactors = factors;
  auto independentMeasurements = measurements;
  independentFactors[1].h[2] = independentFactors[1].h[0];
  independentFactors[1].h[0] = {};
  independentMeasurements[3].covariance = measurements[1].covariance;
  AdjacentTripletFitResult independent{};
  BOOST_REQUIRE(fitAdjacentTripletFactors(independentFactors[0], independentFactors[1],
                                          independentMeasurements, {100.f, 100.f}, independent));
  BOOST_CHECK_NE(correlated.curvatureVariance, independent.curvatureVariance);
}

BOOST_AUTO_TEST_CASE(AdjacentFactorsApplySpaceAngleMSGeometry)
{
  const GlobalCovariance3F exact{};
  const std::array<GlobalMeasurement, 4> measurements{{
    makeMeasurement(0.f, 0.f, 0.f, exact),
    makeMeasurement(1.f, 0.f, 1.f, exact),
    makeMeasurement(2.f, 0.f, 2.f, exact),
    makeMeasurement(3.f, 0.f, 3.f, exact),
  }};
  std::array<TripletFitFactor, 2> factors{};
  factors[0].rho = {1.f, 1.f};
  factors[1].rho = {1.f, 1.f};
  AdjacentTripletFitResult result{};
  BOOST_REQUIRE(fitAdjacentTripletFactors(factors[0], factors[1], measurements, {4.f, 9.f}, result));
  const double expectedPrecision = 1. / 4. + 1. / 8. + 1. / 9. + 1. / 18.;
  checkClose(result.curvatureVariance, 1. / expectedPrecision, 2.e-6);
}

BOOST_AUTO_TEST_CASE(AdjacentExactHelixHasCommonCurvatureAndZeroQuality)
{
  const auto measurements = makeAdjacentHelixMeasurements();
  const std::array<float, 2> angularVariance{1.e-8f, 2.e-8f};
  const auto factors = fitAdjacentFactors(measurements);
  AdjacentTripletFitResult result{};
  BOOST_REQUIRE(fitAdjacentTripletFactors(factors[0], factors[1], measurements, angularVariance, result));
  const double expectedCurvature = (1. / Radius) / std::sqrt(1. + TanLambda * TanLambda);
  checkClose(result.curvature, expectedCurvature, 4.e-4);
  // Native float measurements and persisted float factors leave only this
  // numerical residue in an otherwise exactly common-curvature helix.
  BOOST_CHECK_SMALL(result.chi2, 2.e-6f);
  BOOST_CHECK_GT(result.curvatureVariance, 0.);
}

BOOST_AUTO_TEST_CASE(AdjacentFactorFitIsRotationInvariant)
{
  const auto original = makeAdjacentHelixMeasurements();
  auto rotated = original;
  const double angle = 0.73;
  const double cosine = std::cos(angle);
  const double sine = std::sin(angle);
  for (auto& measurement : rotated) {
    const double x = measurement.x;
    const double y = measurement.y;
    measurement.x = static_cast<float>(cosine * x - sine * y);
    measurement.y = static_cast<float>(sine * x + cosine * y);
    measurement.covariance = rotateCovarianceAroundZ(measurement.covariance, angle);
  }
  const std::array<float, 2> angularVariance{2.e-8f, 3.e-8f};
  const auto originalFactors = fitAdjacentFactors(original);
  const auto rotatedFactors = fitAdjacentFactors(rotated);
  AdjacentTripletFitResult first{};
  AdjacentTripletFitResult second{};
  BOOST_REQUIRE(fitAdjacentTripletFactors(originalFactors[0], originalFactors[1], original, angularVariance, first));
  BOOST_REQUIRE(fitAdjacentTripletFactors(rotatedFactors[0], rotatedFactors[1], rotated, angularVariance, second));
  // Rotating and storing the coordinates and covariance back into floats
  // limits the invariance of the derived Jacobian and covariance.
  checkClose(second.curvature, first.curvature, 2.e-6);
  checkClose(second.curvatureVariance, first.curvatureVariance, 5.e-2);
  checkClose(second.chi2, first.chi2, 1.e-5, 5.e-7);
}

BOOST_AUTO_TEST_CASE(StraightTripletUsesTheRemovableZeroBendingLimit)
{
  const GlobalCovariance3F covariance{1.e-6f, 0.f, 0.f, 1.e-6f, 0.f, 1.e-6f};
  const std::array<GlobalMeasurement, 3> measurements{{
    makeMeasurement(1.f, 2.f, 3.f, covariance),
    makeMeasurement(2.f, 2.f, 3.5f, covariance),
    makeMeasurement(4.f, 2.f, 4.5f, covariance),
  }};
  TripletFitFactor factor{};
  BOOST_REQUIRE(makeTripletFitFactor(measurements, factor));
  BOOST_REQUIRE(factor.isValid());
  BOOST_CHECK_SMALL(-static_cast<double>(factor.psi.phi) / factor.rho.phi, 1.e-15);
}

BOOST_AUTO_TEST_CASE(FactorConstructionGeometryFailureIsTransactional)
{
  TripletFitFactor sentinel{};
  sentinel.psi = {1.f, 2.f};
  sentinel.rho = {3.f, 4.f};
  auto measurements = makeHelixMeasurements();
  TripletFitFactor result = sentinel;

  measurements[1].position = measurements[0].position;
  BOOST_CHECK(!makeTripletFitFactor(measurements, result));
  BOOST_CHECK_EQUAL(std::memcmp(&result, &sentinel, sizeof(result)), 0);
}

BOOST_AUTO_TEST_CASE(CharacterizeFactorConstructionHostCost)
{
  const auto measurements = makeHelixMeasurements();
  constexpr int Repetitions = 20000;
  double checksum = 0.;
  const auto start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < Repetitions; ++iteration) {
    TripletFitFactor factor{};
    BOOST_REQUIRE(makeTripletFitFactor(measurements, factor));
    checksum += factor.psi.theta + factor.psi.phi + factor.rho.theta + factor.rho.phi;
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;
  const double nanosecondsPerFit =
    std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() /
    static_cast<double>(Repetitions);
  BOOST_TEST_MESSAGE("triplet-factor construction host cost: " << nanosecondsPerFit << " ns/factor; checksum=" << checksum);
  BOOST_CHECK_NE(checksum, 0.);
}

BOOST_AUTO_TEST_CASE(CharacterizeAdjacentFactorHostCost)
{
  const auto measurements = makeAdjacentHelixMeasurements();
  const std::array<float, 2> angularVariance{2.e-7f, 3.e-7f};
  const auto factors = fitAdjacentFactors(measurements);
  constexpr int Repetitions = 20000;
  double checksum = 0.;
  const auto start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < Repetitions; ++iteration) {
    AdjacentTripletFitResult result{};
    BOOST_REQUIRE(fitAdjacentTripletFactors(factors[0], factors[1], measurements, angularVariance, result));
    checksum += result.curvature + result.chi2;
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;
  const double nanosecondsPerFit =
    std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() /
    static_cast<double>(Repetitions);
  BOOST_TEST_MESSAGE("adjacent triplet-factor fit host cost: " << nanosecondsPerFit << " ns/fit; checksum=" << checksum);
  BOOST_CHECK_GT(checksum, 0.);
}
