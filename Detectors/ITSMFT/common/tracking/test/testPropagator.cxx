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

#define BOOST_TEST_MODULE ITSMFTPropagator
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>

#include "CommonConstants/MathConstants.h"
#include "ITSMFTTracking/ITSMFTDetectorDefinitions.h"
#include "ITSMFTTracking/RefitDriver.h"
#include "ITSMFTTracking/detail/SurfaceStateOperations.h"
#include "ITSMFTTracking/Propagator.h"

#if __has_include("ITSMFTTracking/BarrelSurfaceStateOperations.h") || __has_include("ITSMFTTracking/ForwardSurfaceStateOperations.h")
#error "coordinate-family state operations must remain private to Propagator"
#endif

using namespace o2::itsmft::tracking;

namespace
{

template <typename T>
bool bitEqual(const T& lhs, const T& rhs)
{
  return std::memcmp(&lhs, &rhs, sizeof(T)) == 0;
}

// --- Barrel fixtures (same convention as testRefitHit.cxx's barrelState()) --

SurfaceTrackState barrelState(uint8_t absCharge = 1, o2::track::PID pid = o2::track::PID::Pion)
{
  SurfaceTrackState state{};
  state.parameters[0] = 1.25f;
  state.parameters[1] = -0.75f;
  state.parameters[2] = 0.2f;
  state.parameters[3] = -0.35f;
  state.parameters[4] = 0.8f;
  state.referenceCoordinate = 4.f;
  state.alpha = 0.3f;
  state.kind = SurfaceKind::Cylinder;
  state.absCharge = absCharge;
  state.pid = pid;
  for (uint8_t row = 0; row < 5; ++row) {
    for (uint8_t column = 0; column <= row; ++column) {
      state.covariance[packedCovarianceIndex(row, column)] = row == column ? 0.01f * (row + 1) : 0.0002f * (row + column + 1);
    }
  }
  return state;
}

SurfaceTrackParameters barrelLinRef(const SurfaceTrackState& state)
{
  return SurfaceTrackParameters{state};
}

SurfaceMeasurement barrelMeasurement()
{
  SurfaceMeasurement measurement{};
  measurement.frame.q = 2.5f;
  measurement.frame.frameAngle = 0.3f; // same alpha as barrelState(): no rotation needed
  measurement.frame.u = 0.8f;
  measurement.frame.v = -0.45f;
  measurement.covariance = {0.04f, 0.012f, 0.09f};
  return measurement;
}

constexpr float BarrelBz = 5.f;

SurfaceDescriptor cylinderDescriptor(NominalSurfaceMaterial material)
{
  SurfaceDescriptor descriptor{};
  descriptor.kind = SurfaceKind::Cylinder;
  descriptor.referenceCoordinate = 2.5f;
  descriptor.material = material;
  return descriptor;
}

// --- Disk fixtures (same convention as testRefitHit.cxx's diskState()) -----

SurfaceTrackState diskState(uint8_t absCharge = 1, o2::track::PID pid = o2::track::PID::Pion)
{
  SurfaceTrackState state{};
  state.parameters[0] = 1.25f;
  state.parameters[1] = -0.75f;
  state.parameters[2] = 0.35f;
  state.parameters[3] = -2.5f;
  state.parameters[4] = 0.8f;
  state.referenceCoordinate = -45.f;
  state.kind = SurfaceKind::Disk;
  state.absCharge = absCharge;
  state.pid = pid;
  for (uint8_t row = 0; row < 5; ++row) {
    for (uint8_t column = 0; column <= row; ++column) {
      state.covariance[packedCovarianceIndex(row, column)] = row == column ? 0.01f * (row + 1) : 0.0002f * (row + column + 1);
    }
  }
  return state;
}

SurfaceTrackParameters diskLinRef(const SurfaceTrackState& state)
{
  return SurfaceTrackParameters{state};
}

SurfaceMeasurement diskMeasurement()
{
  SurfaceMeasurement measurement{};
  measurement.frame = {-50.f, 0.8f, -0.45f, 0.f};
  measurement.frame.q = -50.f;
  measurement.frame.u = 0.8f;
  measurement.frame.v = -0.45f;
  measurement.covariance = {0.04f, 0.f, 0.09f};
  return measurement;
}

constexpr float DiskBz = 5.f;

SurfaceDescriptor diskDescriptor(NominalSurfaceMaterial material)
{
  SurfaceDescriptor descriptor{};
  descriptor.kind = SurfaceKind::Disk;
  descriptor.referenceCoordinate = -50.f;
  descriptor.material = material;
  return descriptor;
}

// Independent double-precision helix intersections for numerical derivatives.
// The target reference plane is fixed for every perturbed source state.
std::array<double, 5> intersectConversionPlane(const SurfaceTrackState& source,
                                               const std::array<double, 5>& p,
                                               const SurfaceTrackState& target, double bz)
{
  double x = p[0], y = p[1], z = source.referenceCoordinate, phi = p[2];
  if (source.kind == SurfaceKind::Cylinder) {
    x = source.referenceCoordinate * std::cos(double(source.alpha)) - p[0] * std::sin(double(source.alpha));
    y = source.referenceCoordinate * std::sin(double(source.alpha)) + p[0] * std::cos(double(source.alpha));
    z = p[1];
    phi = source.alpha + std::asin(p[2]);
  }
  const double curvature = source.absCharge == 0 ? 0. : p[4] * bz * o2::constants::math::B2C;
  auto pointAt = [&](double path) {
    const double halfAngle = curvature * path / 2.;
    const double sinc = halfAngle == 0. ? 1. : std::sin(halfAngle) / halfAngle;
    return std::array<double, 3>{x + path * sinc * std::cos(phi + halfAngle),
                                 y + path * sinc * std::sin(phi + halfAngle), z + path * p[3]};
  };
  double path = 0.;
  if (target.kind == SurfaceKind::Disk) {
    path = (target.referenceCoordinate - z) / p[3];
    const auto position = pointAt(path);
    return {position[0], position[1], phi + curvature * path, p[3], p[4]};
  }
  const double csA = std::cos(double(target.alpha)), snA = std::sin(double(target.alpha));
  // Newton iteration finds the local intersection continuously connected to
  // the nominal point; it does not reuse the production Jacobian.
  for (int iteration = 0; iteration < 6; ++iteration) {
    const auto position = pointAt(path);
    path -= (position[0] * csA + position[1] * snA - target.referenceCoordinate) /
            std::cos(phi + curvature * path - target.alpha);
  }
  const auto position = pointAt(path);
  return {-position[0] * snA + position[1] * csA, position[2],
          std::sin(phi + curvature * path - target.alpha), p[3], p[4]};
}

void checkConversionCovariance(const SurfaceTrackState& source, float bz)
{
  auto target = source;
  OperationFailureReason reason{};
  const auto targetKind = source.kind == SurfaceKind::Cylinder ? SurfaceKind::Disk : SurfaceKind::Cylinder;
  BOOST_REQUIRE(Propagator::convertKind(target, targetKind, bz, reason));
  double jacobian[5][5]{};
  std::array<double, 5> nominal{};
  std::copy(std::begin(source.parameters), std::end(source.parameters), nominal.begin());
  constexpr double step = 1.e-5;
  for (int column = 0; column < 5; ++column) {
    auto plus = nominal, minus = nominal;
    plus[column] += step;
    minus[column] -= step;
    const auto high = intersectConversionPlane(source, plus, target, bz);
    const auto low = intersectConversionPlane(source, minus, target, bz);
    for (int row = 0; row < 5; ++row) {
      jacobian[row][column] = (high[row] - low[row]) / (2. * step);
    }
  }
  for (int row = 0; row < 5; ++row) {
    for (int column = 0; column <= row; ++column) {
      double expected = 0.;
      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
          expected += jacobian[row][i] * source.covariance[packedCovarianceIndex(i, j)] * jacobian[column][j];
        }
      }
      const float actual = target.covariance[packedCovarianceIndex(row, column)];
      BOOST_CHECK_SMALL(double(actual) - expected, 1.e-7 + 2.e-5 * std::abs(expected));
    }
  }
}

} // namespace

// --- 1/2: same-family propagate-to-measurement succeeds ---------------------

BOOST_AUTO_TEST_CASE(CylinderToCylinderPropagateAndUpdateSucceeds)
{
  auto state = barrelState();
  auto linRef = barrelLinRef(state);
  const auto measurement = barrelMeasurement();
  const auto descriptor = cylinderDescriptor(NominalSurfaceMaterial{0.f, 0.f});
  float chi2 = 0.f;
  OperationFailureReason reason{};

  BOOST_REQUIRE(Propagator::propagateToMeasurement(state, linRef, descriptor, measurement, BarrelBz,
                                                   material::MaterialTraversalDirection::AlongMomentum,
                                                   false, 0.f, chi2, false, reason));
  BOOST_CHECK_EQUAL(static_cast<int>(state.kind), static_cast<int>(SurfaceKind::Cylinder));
  BOOST_CHECK_EQUAL(state.referenceCoordinate, measurement.frame.q);
  BOOST_CHECK(std::isfinite(chi2));
  BOOST_CHECK_GE(chi2, 0.f);
}

BOOST_AUTO_TEST_CASE(DiskToDiskPropagateAndUpdateSucceeds)
{
  auto state = diskState();
  auto linRef = diskLinRef(state);
  const auto measurement = diskMeasurement();
  const auto descriptor = diskDescriptor(NominalSurfaceMaterial{0.f, 0.f});
  float chi2 = 0.f;
  OperationFailureReason reason{};

  BOOST_REQUIRE(Propagator::propagateToMeasurement(state, linRef, descriptor, measurement, DiskBz,
                                                   material::MaterialTraversalDirection::AlongMomentum,
                                                   false, 0.f, chi2, false, reason));
  BOOST_CHECK_EQUAL(static_cast<int>(state.kind), static_cast<int>(SurfaceKind::Disk));
  BOOST_CHECK_EQUAL(state.referenceCoordinate, measurement.frame.q);
  BOOST_CHECK(std::isfinite(chi2));
  BOOST_CHECK_GE(chi2, 0.f);
}

BOOST_AUTO_TEST_CASE(AcceptedForwardPropagationSelectsFieldAndLowFieldPaths)
{
  auto fieldOn = diskState();
  auto lowPositive = diskState();
  auto lowNegative = diskState();
  OperationFailureReason reason{};

  BOOST_REQUIRE(Propagator::propagateToReference(fieldOn, -50.f, 5.f, reason));
  BOOST_REQUIRE(Propagator::propagateToReference(lowPositive, -50.f, 0.01f, reason));
  BOOST_REQUIRE(Propagator::propagateToReference(lowNegative, -50.f, -0.01f, reason));
  BOOST_CHECK(bitEqual(lowPositive, lowNegative));
  BOOST_CHECK(!bitEqual(fieldOn, lowPositive));
}

BOOST_AUTO_TEST_CASE(PropagatorSelectsCompatibilityFromStateKind)
{
  auto cylinderReference = barrelState();
  auto cylinderCandidate = cylinderReference;
  auto diskReference = diskState();
  auto diskCandidate = diskReference;
  float chi2 = -1.f;
  OperationFailureReason reason{};

  BOOST_REQUIRE(Propagator::stateChi2(cylinderReference, cylinderCandidate, chi2, reason));
  BOOST_CHECK_EQUAL(chi2, 0.f);
  BOOST_REQUIRE(Propagator::stateChi2(diskReference, diskCandidate, chi2, reason));
  BOOST_CHECK_EQUAL(chi2, 0.f);
  BOOST_CHECK(!Propagator::stateChi2(cylinderReference, diskCandidate, chi2, reason));
  BOOST_CHECK(reason == OperationFailureReason::SourceSurfaceKindMismatch);
}

// --- 3: compatible family never converts -- exact agreement with a direct
// detail::barrel::rotate/propagate/correctForMaterial/predictedChi2/update replay ---

BOOST_AUTO_TEST_CASE(CompatibleFamilyMatchesDirectBarrelPrimitiveReplay)
{
  auto viaPropagator = barrelState();
  auto viaPropagatorRef = barrelLinRef(viaPropagator);
  auto viaDirect = viaPropagator;
  auto viaDirectRef = viaPropagatorRef;
  const auto measurement = barrelMeasurement();
  const auto material = NominalSurfaceMaterial{0.01f, 0.001f};
  const auto descriptor = cylinderDescriptor(material);
  float chi2Propagator = 0.f;
  float chi2Direct = 0.f;
  OperationFailureReason reason{};

  BOOST_REQUIRE(Propagator::propagateToMeasurement(viaPropagator, viaPropagatorRef, descriptor, measurement, BarrelBz,
                                                   material::MaterialTraversalDirection::OppositeMomentum,
                                                   false, 0.f, chi2Propagator, true, reason));

  BOOST_REQUIRE(detail::barrel::rotate(viaDirect, viaDirectRef, measurement.frame.frameAngle, BarrelBz, reason));
  BOOST_REQUIRE(detail::barrel::propagate(viaDirect, viaDirectRef, measurement.frame.q, BarrelBz, reason));
  const auto materialResult = detail::barrel::correctForMaterial(
    viaDirect, viaDirectRef, material::IntegratedMaterialBudget{material.xOverX0, material.arealDensityGPerCm2},
    material::MaterialTraversalDirection::OppositeMomentum);
  BOOST_REQUIRE(materialResult.ok());
  float predChi2 = 0.f;
  BOOST_REQUIRE(detail::barrel::predictedChi2(viaDirect, measurement, predChi2, reason));
  float updateChi2 = 0.f;
  BOOST_REQUIRE(detail::barrel::update(viaDirect, measurement, updateChi2, reason));
  chi2Direct = updateChi2;
  BOOST_REQUIRE(detail::barrel::shiftReferenceToMeasurement(viaDirectRef, measurement, reason));

  BOOST_CHECK(bitEqual(viaPropagator, viaDirect));
  BOOST_CHECK(bitEqual(viaPropagatorRef, viaDirectRef));
  BOOST_CHECK_EQUAL(chi2Propagator, chi2Direct);
}

BOOST_AUTO_TEST_CASE(BarrelMaterialUsesLegacyIncidencePathLength)
{
  auto state = barrelState();
  state.parameters[2] = 0.6f;
  state.parameters[3] = 1.2f;
  const auto original = state;
  const material::IntegratedMaterialBudget nominalMaterial{0.01f, 0.001f};

  const float snp = original.parameters[2];
  const float tgl = original.parameters[3];
  const float incidenceScale = std::sqrt((1.f + tgl * tgl) / ((1.f - snp) * (1.f + snp)));
  const material::IntegratedMaterialBudget legacyMaterial{
    nominalMaterial.xOverX0 * incidenceScale,
    nominalMaterial.arealDensityGPerCm2 * incidenceScale};
  const float transverseMomentum = static_cast<float>(original.absCharge) / std::abs(original.parameters[4]);
  const float momentum = transverseMomentum * std::sqrt(1.f + tgl * tgl);

  const auto expected = material::calculateMaterialPhysics(momentum, original.pid, original.absCharge,
                                                           material::MaterialTraversalDirection::AlongMomentum,
                                                           legacyMaterial);
  const auto uncorrected = material::calculateMaterialPhysics(momentum, original.pid, original.absCharge,
                                                              material::MaterialTraversalDirection::AlongMomentum,
                                                              nominalMaterial);
  const auto result = detail::barrel::correctForMaterial(state, nominalMaterial,
                                                         material::MaterialTraversalDirection::AlongMomentum);

  BOOST_REQUIRE(expected.ok());
  BOOST_REQUIRE(uncorrected.ok());
  BOOST_REQUIRE(result.ok());
  BOOST_CHECK_EQUAL(result.momentumBeforeGeV, expected.momentumBeforeGeV);
  BOOST_CHECK_EQUAL(result.momentumAfterGeV, expected.momentumAfterGeV);
  BOOST_CHECK_EQUAL(result.signedEnergyChangeGeV, expected.signedEnergyChangeGeV);
  BOOST_CHECK_EQUAL(result.highlandTheta2Rad2, expected.highlandTheta2Rad2);
  BOOST_CHECK_EQUAL(result.relativeInverseMomentumVariance, expected.relativeInverseMomentumVariance);
  BOOST_CHECK_EQUAL(result.energyLossSubsteps, expected.energyLossSubsteps);
  BOOST_CHECK_GT(result.highlandTheta2Rad2, uncorrected.highlandTheta2Rad2);
  BOOST_CHECK_LT(result.momentumAfterGeV, uncorrected.momentumAfterGeV);
}

BOOST_AUTO_TEST_CASE(LinearizedBarrelMaterialUsesLegacyReferenceIncidence)
{
  auto state = barrelState();
  state.parameters[2] = 0.1f;
  state.parameters[3] = 0.2f;
  auto linRef = barrelLinRef(state);
  linRef.parameters[2] = 0.6f;
  linRef.parameters[3] = 1.2f;
  const float stateQ2PtBefore = state.parameters[4];
  const float referenceQ2PtBefore = linRef.parameters[4];
  const material::IntegratedMaterialBudget nominalMaterial{0.01f, 0.001f};

  const float snp = linRef.parameters[2];
  const float tgl = linRef.parameters[3];
  const float incidenceScale = std::sqrt((1.f + tgl * tgl) / ((1.f - snp) * (1.f + snp)));
  const material::IntegratedMaterialBudget legacyMaterial{
    nominalMaterial.xOverX0 * incidenceScale,
    nominalMaterial.arealDensityGPerCm2 * incidenceScale};
  const float stateTgl = state.parameters[3];
  const float transverseMomentum = static_cast<float>(state.absCharge) / std::abs(state.parameters[4]);
  const float momentum = transverseMomentum * std::sqrt(1.f + stateTgl * stateTgl);

  const auto expected = material::calculateMaterialPhysics(momentum, state.pid, state.absCharge,
                                                           material::MaterialTraversalDirection::AlongMomentum,
                                                           legacyMaterial);
  const auto result = detail::barrel::correctForMaterial(state, linRef, nominalMaterial,
                                                         material::MaterialTraversalDirection::AlongMomentum);

  BOOST_REQUIRE(expected.ok());
  BOOST_REQUIRE(result.ok());
  BOOST_CHECK_EQUAL(result.momentumAfterGeV, expected.momentumAfterGeV);
  BOOST_CHECK_EQUAL(result.highlandTheta2Rad2, expected.highlandTheta2Rad2);
  const float expectedStateQ2Pt = (stateQ2PtBefore * result.momentumBeforeGeV) / result.momentumAfterGeV;
  const float expectedReferenceQ2Pt = (referenceQ2PtBefore * result.momentumBeforeGeV) / result.momentumAfterGeV;
  BOOST_CHECK_EQUAL(state.parameters[4], expectedStateQ2Pt);
  BOOST_CHECK_EQUAL(linRef.parameters[4], expectedReferenceQ2Pt);
}

BOOST_AUTO_TEST_CASE(LinearizedBarrelMaterialKeepsReferenceQ2PtForMCSOnly)
{
  auto state = barrelState();
  auto linRef = barrelLinRef(state);
  const auto referenceBefore = linRef;

  const auto result = detail::barrel::correctForMaterial(
    state, linRef, material::IntegratedMaterialBudget{0.01f, 0.f},
    material::MaterialTraversalDirection::AlongMomentum);

  BOOST_REQUIRE(result.ok());
  BOOST_CHECK_EQUAL(result.momentumBeforeGeV, result.momentumAfterGeV);
  BOOST_CHECK(bitEqual(linRef, referenceBefore));
}

BOOST_AUTO_TEST_CASE(FailingLinearizedBarrelMaterialLeavesStateAndReferenceUnchanged)
{
  auto state = barrelState();
  auto linRef = barrelLinRef(state);
  const auto stateBefore = state;
  const auto referenceBefore = linRef;

  const auto result = detail::barrel::correctForMaterial(
    state, linRef, material::IntegratedMaterialBudget{1.e8f, 0.f},
    material::MaterialTraversalDirection::AlongMomentum);

  BOOST_CHECK(!result.ok());
  BOOST_CHECK(result.failure == material::MaterialFailureReason::ExcessiveScattering);
  BOOST_CHECK(bitEqual(state, stateBefore));
  BOOST_CHECK(bitEqual(linRef, referenceBefore));
}

BOOST_AUTO_TEST_CASE(CompatibleFamilyMatchesDirectForwardPrimitiveReplay)
{
  auto viaPropagator = diskState();
  auto viaPropagatorRef = diskLinRef(viaPropagator);
  auto viaDirect = viaPropagator;
  auto viaDirectRef = viaPropagatorRef;
  const auto measurement = diskMeasurement();
  const auto material = NominalSurfaceMaterial{0.01f, 0.001f};
  const auto descriptor = diskDescriptor(material);
  float chi2Propagator = 0.f;
  float chi2Direct = 0.f;
  OperationFailureReason reason{};

  BOOST_REQUIRE(Propagator::propagateToMeasurement(viaPropagator, viaPropagatorRef, descriptor, measurement, DiskBz,
                                                   material::MaterialTraversalDirection::OppositeMomentum,
                                                   false, 0.f, chi2Propagator, true, reason));

  BOOST_REQUIRE(detail::forward::propagate(viaDirect, viaDirectRef, measurement.frame.q, DiskBz, reason));
  const auto materialResult = detail::forward::correctForMaterial(
    viaDirect, viaDirectRef, material::IntegratedMaterialBudget{material.xOverX0, material.arealDensityGPerCm2},
    material::MaterialTraversalDirection::OppositeMomentum);
  BOOST_REQUIRE(materialResult.ok());
  float predChi2 = 0.f;
  BOOST_REQUIRE(detail::forward::predictedChi2(viaDirect, measurement, predChi2, reason));
  float updateChi2 = 0.f;
  BOOST_REQUIRE(detail::forward::update(viaDirect, measurement, updateChi2, reason));
  chi2Direct = updateChi2;
  BOOST_REQUIRE(detail::forward::shiftReferenceToMeasurement(viaDirectRef, measurement, reason));

  BOOST_CHECK(bitEqual(viaPropagator, viaDirect));
  BOOST_CHECK(bitEqual(viaPropagatorRef, viaDirectRef));
  BOOST_CHECK_EQUAL(chi2Propagator, chi2Direct);
}

BOOST_AUTO_TEST_CASE(ForwardMaterialUsesLegacyIncidencePathLength)
{
  auto state = diskState();
  state.parameters[3] = -0.5f;
  const auto original = state;
  const material::IntegratedMaterialBudget nominalMaterial{0.01f, 0.001f};

  const float tgl = original.parameters[3];
  const float incidenceScale = std::sqrt(1.f + tgl * tgl) / std::abs(tgl);
  const material::IntegratedMaterialBudget legacyMaterial{
    nominalMaterial.xOverX0 * incidenceScale,
    nominalMaterial.arealDensityGPerCm2 * incidenceScale};
  const float transverseMomentum = static_cast<float>(original.absCharge) / std::abs(original.parameters[4]);
  const float momentum = transverseMomentum * std::sqrt(1.f + tgl * tgl);

  const auto expected = material::calculateMaterialPhysics(momentum, original.pid, original.absCharge,
                                                           material::MaterialTraversalDirection::AlongMomentum,
                                                           legacyMaterial);
  const auto uncorrected = material::calculateMaterialPhysics(momentum, original.pid, original.absCharge,
                                                              material::MaterialTraversalDirection::AlongMomentum,
                                                              nominalMaterial);
  const auto result = detail::forward::correctForMaterial(state, nominalMaterial,
                                                          material::MaterialTraversalDirection::AlongMomentum);

  BOOST_REQUIRE(expected.ok());
  BOOST_REQUIRE(uncorrected.ok());
  BOOST_REQUIRE(result.ok());
  BOOST_CHECK_EQUAL(result.momentumBeforeGeV, expected.momentumBeforeGeV);
  BOOST_CHECK_EQUAL(result.momentumAfterGeV, expected.momentumAfterGeV);
  BOOST_CHECK_EQUAL(result.signedEnergyChangeGeV, expected.signedEnergyChangeGeV);
  BOOST_CHECK_EQUAL(result.highlandTheta2Rad2, expected.highlandTheta2Rad2);
  BOOST_CHECK_EQUAL(result.relativeInverseMomentumVariance, expected.relativeInverseMomentumVariance);
  BOOST_CHECK_EQUAL(result.energyLossSubsteps, expected.energyLossSubsteps);
  BOOST_CHECK_GT(result.highlandTheta2Rad2, uncorrected.highlandTheta2Rad2);
  BOOST_CHECK_LT(result.momentumAfterGeV, uncorrected.momentumAfterGeV);
}

BOOST_AUTO_TEST_CASE(LinearizedForwardMaterialUsesReferenceIncidence)
{
  auto state = diskState();
  auto linRef = diskLinRef(state);
  linRef.parameters[3] = -0.5f;
  const float stateQ2PtBefore = state.parameters[4];
  const float referenceQ2PtBefore = linRef.parameters[4];
  const material::IntegratedMaterialBudget nominalMaterial{0.01f, 0.001f};

  const float referenceTgl = linRef.parameters[3];
  const float incidenceScale = std::sqrt(1.f + referenceTgl * referenceTgl) / std::abs(referenceTgl);
  const material::IntegratedMaterialBudget scaledMaterial{
    nominalMaterial.xOverX0 * incidenceScale,
    nominalMaterial.arealDensityGPerCm2 * incidenceScale};
  const float stateTgl = state.parameters[3];
  const float transverseMomentum = static_cast<float>(state.absCharge) / std::abs(state.parameters[4]);
  const float momentum = transverseMomentum * std::sqrt(1.f + stateTgl * stateTgl);

  const auto expected = material::calculateMaterialPhysics(momentum, state.pid, state.absCharge,
                                                           material::MaterialTraversalDirection::AlongMomentum,
                                                           scaledMaterial);
  const auto result = detail::forward::correctForMaterial(state, linRef, nominalMaterial,
                                                          material::MaterialTraversalDirection::AlongMomentum);

  BOOST_REQUIRE(expected.ok());
  BOOST_REQUIRE(result.ok());
  BOOST_CHECK_EQUAL(result.momentumAfterGeV, expected.momentumAfterGeV);
  BOOST_CHECK_EQUAL(result.highlandTheta2Rad2, expected.highlandTheta2Rad2);
  const float expectedStateQ2Pt = (stateQ2PtBefore * result.momentumBeforeGeV) / result.momentumAfterGeV;
  const float expectedReferenceQ2Pt = (referenceQ2PtBefore * result.momentumBeforeGeV) / result.momentumAfterGeV;
  BOOST_CHECK_EQUAL(state.parameters[4], expectedStateQ2Pt);
  BOOST_CHECK_EQUAL(linRef.parameters[4], expectedReferenceQ2Pt);
}

BOOST_AUTO_TEST_CASE(LinearizedForwardMaterialKeepsReferenceQ2PtForMCSOnly)
{
  auto state = diskState();
  auto linRef = diskLinRef(state);
  const auto referenceBefore = linRef;

  const auto result = detail::forward::correctForMaterial(
    state, linRef, material::IntegratedMaterialBudget{0.01f, 0.f},
    material::MaterialTraversalDirection::AlongMomentum);

  BOOST_REQUIRE(result.ok());
  BOOST_CHECK_EQUAL(result.momentumBeforeGeV, result.momentumAfterGeV);
  BOOST_CHECK(bitEqual(linRef, referenceBefore));
}

BOOST_AUTO_TEST_CASE(FailingLinearizedForwardMaterialLeavesStateAndReferenceUnchanged)
{
  auto state = diskState();
  auto linRef = diskLinRef(state);
  const auto stateBefore = state;
  const auto referenceBefore = linRef;

  const auto result = detail::forward::correctForMaterial(
    state, linRef, material::IntegratedMaterialBudget{1.e8f, 0.f},
    material::MaterialTraversalDirection::AlongMomentum);

  BOOST_CHECK(!result.ok());
  BOOST_CHECK(result.failure == material::MaterialFailureReason::ExcessiveScattering);
  BOOST_CHECK(bitEqual(state, stateBefore));
  BOOST_CHECK(bitEqual(linRef, referenceBefore));
}

// --- 4: incompatible family converts, then propagates -----------------------

BOOST_AUTO_TEST_CASE(BarrelStateConvertsToForwardThenPropagatesToDiskMeasurement)
{
  auto state = barrelState();
  auto linRef = barrelLinRef(state);
  const auto poisonState = state;

  // A disk far enough along z that the converted (Forward) state can reach it.
  SurfaceMeasurement measurement{};
  measurement.frame.q = -10.f;
  measurement.frame.u = 5.f;
  measurement.frame.v = -5.f;
  measurement.covariance = {10.f, 0.f, 10.f}; // loose: the point is not expected to land exactly here
  const auto descriptor = diskDescriptor(NominalSurfaceMaterial{0.f, 0.f});
  float chi2 = 0.f;
  OperationFailureReason reason{};

  const bool ok = Propagator::propagateToMeasurement(state, linRef, descriptor, measurement, BarrelBz,
                                                     material::MaterialTraversalDirection::AlongMomentum,
                                                     false, 0.f, chi2, false, reason);
  BOOST_REQUIRE(ok);
  BOOST_CHECK_EQUAL(static_cast<int>(state.kind), static_cast<int>(SurfaceKind::Disk));
  BOOST_CHECK_EQUAL(state.referenceCoordinate, measurement.frame.q);
  BOOST_CHECK_EQUAL(state.absCharge, poisonState.absCharge);
  BOOST_CHECK(state.pid == poisonState.pid);
  for (float value : state.parameters) {
    BOOST_CHECK(std::isfinite(value));
  }
  for (float value : state.covariance) {
    BOOST_CHECK(std::isfinite(value));
  }
}

BOOST_AUTO_TEST_CASE(KindConversionRelinearizesAtConvertedState)
{
  auto nominalState = barrelState();
  auto nominalRef = barrelLinRef(nominalState);
  auto perturbedState = nominalState;
  auto perturbedRef = nominalRef;
  perturbedRef.parameters[0] += 0.1f;
  perturbedRef.parameters[1] -= 0.2f;
  perturbedRef.parameters[2] += 0.01f;
  perturbedRef.parameters[3] -= 0.02f;
  perturbedRef.parameters[4] += 0.001f;

  SurfaceMeasurement measurement{};
  measurement.frame.q = -10.f;
  measurement.frame.u = 5.f;
  measurement.frame.v = -5.f;
  measurement.covariance = {10.f, 0.f, 10.f};
  const auto descriptor = diskDescriptor(NominalSurfaceMaterial{0.f, 0.f});
  float nominalChi2 = 0.f;
  float perturbedChi2 = 0.f;
  OperationFailureReason reason{};

  BOOST_REQUIRE(Propagator::propagateToMeasurement(nominalState, nominalRef, descriptor, measurement, BarrelBz,
                                                   material::MaterialTraversalDirection::AlongMomentum,
                                                   false, 0.f, nominalChi2, false, reason));
  BOOST_REQUIRE(Propagator::propagateToMeasurement(perturbedState, perturbedRef, descriptor, measurement, BarrelBz,
                                                   material::MaterialTraversalDirection::AlongMomentum,
                                                   false, 0.f, perturbedChi2, false, reason));

  BOOST_CHECK(bitEqual(perturbedState, nominalState));
  BOOST_CHECK(bitEqual(perturbedRef, nominalRef));
  BOOST_CHECK_EQUAL(perturbedChi2, nominalChi2);
}

BOOST_AUTO_TEST_CASE(ReverseKindConversionRelinearizesAtConvertedState)
{
  auto nominalState = diskState();
  auto nominalRef = diskLinRef(nominalState);
  auto perturbedState = nominalState;
  auto perturbedRef = nominalRef;
  perturbedRef.parameters[0] += 0.1f;
  perturbedRef.parameters[1] -= 0.2f;
  perturbedRef.parameters[2] += 0.01f;
  perturbedRef.parameters[3] -= 0.02f;
  perturbedRef.parameters[4] += 0.001f;

  const auto measurement = barrelMeasurement();
  const auto descriptor = cylinderDescriptor(NominalSurfaceMaterial{0.f, 0.f});
  float nominalChi2 = 0.f;
  float perturbedChi2 = 0.f;
  OperationFailureReason reason{};

  BOOST_REQUIRE(Propagator::propagateToMeasurement(nominalState, nominalRef, descriptor, measurement, DiskBz,
                                                   material::MaterialTraversalDirection::AlongMomentum,
                                                   false, 0.f, nominalChi2, false, reason));
  BOOST_REQUIRE(Propagator::propagateToMeasurement(perturbedState, perturbedRef, descriptor, measurement, DiskBz,
                                                   material::MaterialTraversalDirection::AlongMomentum,
                                                   false, 0.f, perturbedChi2, false, reason));

  BOOST_CHECK(bitEqual(perturbedState, nominalState));
  BOOST_CHECK(bitEqual(perturbedRef, nominalRef));
  BOOST_CHECK_EQUAL(perturbedChi2, nominalChi2);
}

BOOST_AUTO_TEST_CASE(ConversionCovarianceMatchesFixedPlaneHelixDifferences)
{
  for (const float bz : {-5.f, 0.f, 5.f}) {
    for (const float sign : {-1.f, 1.f}) {
      auto barrel = barrelState();
      barrel.parameters[3] *= sign;
      barrel.parameters[4] *= sign;
      checkConversionCovariance(barrel, bz);
      auto disk = diskState();
      disk.parameters[3] *= sign;
      disk.parameters[4] *= sign;
      checkConversionCovariance(disk, bz);
    }
  }
}

BOOST_AUTO_TEST_CASE(BarrelZUncertaintySurvivesConversionAndRoundTrip)
{
  auto state = barrelState();
  state.alpha = 0.f;
  state.referenceCoordinate = 10.f;
  state.parameters[0] = 0.f;
  state.parameters[2] = 0.f;
  state.parameters[3] = 2.f;
  std::fill(std::begin(state.covariance), std::end(state.covariance), 0.f);
  state.covariance[packedCovarianceIndex(1, 1)] = 1.f;
  const auto before = state;
  OperationFailureReason reason{};
  BOOST_REQUIRE(Propagator::convertKind(state, SurfaceKind::Disk, 5.f, reason));
  BOOST_CHECK_CLOSE(state.covariance[packedCovarianceIndex(0, 0)], 0.25f, 1.e-4f);
  const float curvature = before.parameters[4] * 5.f * o2::constants::math::B2C;
  BOOST_CHECK_CLOSE(state.covariance[packedCovarianceIndex(2, 0)], curvature / 4.f, 1.e-4f);
  BOOST_REQUIRE(Propagator::convertKind(state, SurfaceKind::Cylinder, 5.f, reason));
  for (int i = 0; i < 15; ++i) {
    BOOST_CHECK_SMALL(state.covariance[i] - before.covariance[i], 1.e-6f);
  }
}

BOOST_AUTO_TEST_CASE(ConversionRejectsSingularAndNonFiniteInputsTransactionally)
{
  for (const float tanl : {0.f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
    auto state = barrelState();
    state.parameters[3] = tanl;
    const auto before = state;
    OperationFailureReason reason{};
    BOOST_CHECK(!Propagator::convertKind(state, SurfaceKind::Disk, 5.f, reason));
    BOOST_CHECK(reason == OperationFailureReason::SurfaceKindConversionFailure);
    BOOST_CHECK(bitEqual(state, before));
  }
}

BOOST_AUTO_TEST_CASE(NonlinearAttachmentUsesTargetKindAndRollsBackAfterConversion)
{
  for (const bool startOnDisk : {false, true}) {
    const auto source = startOnDisk ? diskState() : barrelState();
    const auto target = startOnDisk ? cylinderDescriptor({0.f, 0.f}) : diskDescriptor({0.f, 0.f});
    auto converted = source;
    OperationFailureReason reason{};
    BOOST_REQUIRE(Propagator::convertKind(converted, target.kind, 0.f, reason));
    SurfaceMeasurement measurement{};
    measurement.frame = {converted.referenceCoordinate, converted.parameters[0], converted.parameters[1], converted.alpha};
    measurement.covariance = {0.04f, 0.f, 0.04f};
    auto state = source;
    float chi2 = 0.f;
    BOOST_REQUIRE(Propagator::attachMeasurement(state, target, measurement, 0.f,
                                                material::MaterialTraversalDirection::OppositeMomentum,
                                                true, 100.f, chi2, reason));
    BOOST_CHECK(state.kind == target.kind);
    for (int i = 0; i < 5; ++i) {
      BOOST_CHECK_SMALL(state.parameters[i] - converted.parameters[i], 1.e-5f);
    }
    BOOST_CHECK_SMALL(chi2, 1.e-5f);

    // Conversion may succeed while the measurement gate fails; neither the
    // converted representation nor a partial chi2 may escape to the caller.
    measurement.frame.u += 10.f;
    state = source;
    chi2 = 3.f;
    BOOST_CHECK(!Propagator::attachMeasurement(state, target, measurement, 0.f,
                                               material::MaterialTraversalDirection::OppositeMomentum,
                                               true, 1.e-6f, chi2, reason));
    BOOST_CHECK(reason == OperationFailureReason::PredictedChi2Failure);
    BOOST_CHECK(bitEqual(state, source));
    BOOST_CHECK_EQUAL(chi2, 3.f);
  }
}

BOOST_AUTO_TEST_CASE(ConvertFamilyPreservesChargeAndPID)
{
  auto state = barrelState(2, o2::track::PID::Kaon);
  OperationFailureReason reason{};
  BOOST_REQUIRE(Propagator::convertKind(state, SurfaceKind::Disk, BarrelBz, reason));
  BOOST_CHECK_EQUAL(static_cast<int>(state.kind), static_cast<int>(SurfaceKind::Disk));
  BOOST_CHECK_EQUAL(state.absCharge, uint8_t{2});
  BOOST_CHECK(state.pid == o2::track::PID::Kaon);
}

BOOST_AUTO_TEST_CASE(ConvertFamilySameFamilyIsNoOpSuccess)
{
  auto state = barrelState();
  const auto before = state;
  OperationFailureReason reason{};
  BOOST_REQUIRE(Propagator::convertKind(state, SurfaceKind::Cylinder, DiskBz, reason));
  BOOST_CHECK(bitEqual(state, before));
}

// --- 5: degenerate conversion fails, transactionally ------------------------

BOOST_AUTO_TEST_CASE(ForwardToBarrelConversionFailsAtOriginTransactionally)
{
  auto state = diskState();
  state.parameters[0] = 0.f; // X
  state.parameters[1] = 0.f; // Y: R == 0, alpha undefined
  const auto poison = state;
  OperationFailureReason reason{};

  BOOST_CHECK(!Propagator::convertKind(state, SurfaceKind::Cylinder, DiskBz, reason));
  BOOST_CHECK_EQUAL(static_cast<int>(reason), static_cast<int>(OperationFailureReason::SurfaceKindConversionFailure));
  BOOST_CHECK(bitEqual(state, poison));
}

BOOST_AUTO_TEST_CASE(ForwardToBarrelRejectsUnrepresentableDirectionsTransactionally)
{
  for (const float phi : {o2::constants::math::PI, -2.f, 2.f, o2::constants::math::PIHalf}) {
    auto state = diskState();
    state.parameters[0] = 10.f;
    state.parameters[1] = 0.f;
    state.parameters[2] = phi;
    const auto before = state;
    OperationFailureReason reason{};
    BOOST_CHECK(!Propagator::convertKind(state, SurfaceKind::Cylinder, DiskBz, reason));
    BOOST_CHECK(reason == OperationFailureReason::SurfaceKindConversionFailure);
    BOOST_CHECK(bitEqual(state, before));
  }
}

// --- Zero-material and nonzero-material (MatLUT/nominal-material) paths -----

BOOST_AUTO_TEST_CASE(ZeroMaterialPathSucceeds)
{
  auto state = barrelState();
  auto linRef = barrelLinRef(state);
  const auto measurement = barrelMeasurement();
  const auto descriptor = cylinderDescriptor(NominalSurfaceMaterial{0.f, 0.f});
  float chi2 = 0.f;
  OperationFailureReason reason{};
  BOOST_REQUIRE(Propagator::propagateToMeasurement(state, linRef, descriptor, measurement, BarrelBz,
                                                   material::MaterialTraversalDirection::AlongMomentum,
                                                   false, 0.f, chi2, false, reason));
}

BOOST_AUTO_TEST_CASE(NonzeroNominalMaterialChangesResultRelativeToZeroMaterial)
{
  auto zeroState = barrelState();
  auto zeroRef = barrelLinRef(zeroState);
  auto materialState = barrelState();
  auto materialRef = barrelLinRef(materialState);
  const auto measurement = barrelMeasurement();
  const auto zeroDescriptor = cylinderDescriptor(NominalSurfaceMaterial{0.f, 0.f});
  const auto materialDescriptor = cylinderDescriptor(NominalSurfaceMaterial{0.05f, 0.01f});
  float zeroChi2 = 0.f;
  float materialChi2 = 0.f;
  OperationFailureReason reason{};

  BOOST_REQUIRE(Propagator::propagateToMeasurement(zeroState, zeroRef, zeroDescriptor, measurement, BarrelBz,
                                                   material::MaterialTraversalDirection::OppositeMomentum,
                                                   false, 0.f, zeroChi2, false, reason));
  BOOST_REQUIRE(Propagator::propagateToMeasurement(materialState, materialRef, materialDescriptor, measurement, BarrelBz,
                                                   material::MaterialTraversalDirection::OppositeMomentum,
                                                   false, 0.f, materialChi2, false, reason));

  // The material budget is read from the target SurfaceDescriptor (the
  // "MatLUT" mechanism, task requirement 6) -- not equal, not a parallel
  // model producing a byte-identical result either.
  BOOST_CHECK(!bitEqual(zeroState, materialState));
}

// --- Holes are skipped by the native refit driver ----------------------------

BOOST_AUTO_TEST_CASE(RefitDriverSkipsHoleSlots)
{
  auto state = barrelState();
  auto linRef = barrelLinRef(state);
  const auto measurement = barrelMeasurement();

  std::array<SurfaceDescriptor, 1> surfaces{cylinderDescriptor(NominalSurfaceMaterial{0.f, 0.f})};
  SurfaceCatalogView catalog{surfaces.data(), static_cast<uint32_t>(surfaces.size())};

  const detail::RefitMeasurementSlot present{measurement, LayerId{0}, true};
  const detail::RefitMeasurementSlot hole{};

  std::array<detail::RefitMeasurementSlot, 3> slots{hole, present, hole};
  float chi2 = 0.f;
  uint32_t acceptedHitCount = 999;
  OperationFailureReason reason{};

  BOOST_REQUIRE(detail::driveRefitLeg(state, linRef, chi2, acceptedHitCount, slots, catalog, BarrelBz,
                                      material::MaterialTraversalDirection::AlongMomentum, false, 100.f, reason));
  BOOST_CHECK_EQUAL(acceptedHitCount, 1u);
}

BOOST_AUTO_TEST_CASE(FullMFTRefitLegUsesOneDetectorMaterialBudget)
{
  const SurfaceCatalogView catalog{kMFTStaticSurfaceCatalog.data(), MFTNLayers};
  for (const auto direction : {material::MaterialTraversalDirection::AlongMomentum,
                               material::MaterialTraversalDirection::OppositeMomentum}) {
    const bool alongMomentum = direction == material::MaterialTraversalDirection::AlongMomentum;
    auto state = diskState();
    state.referenceCoordinate = kMFTStaticSurfaceCatalog[alongMomentum ? 0 : MFTNLayers - 1].referenceCoordinate;
    // Field-off and exact measurements isolate the accumulated energy loss.
    for (uint8_t row = 0; row < 5; ++row) {
      for (uint8_t column = 0; column < row; ++column) {
        state.covariance[packedCovarianceIndex(row, column)] = 0.f;
      }
    }
    auto linRef = diskLinRef(state);
    const float tanl = state.parameters[3];
    const float momentumScale = std::sqrt(1.f + tanl * tanl);
    float expectedMomentum = momentumScale / std::abs(state.parameters[4]);
    const float initialMomentum = expectedMomentum;
    const float pathX0 = kMFTNominalRadLength / MFTNLayers * momentumScale / std::abs(tanl);
    const material::IntegratedMaterialBudget expectedMaterial{
      pathX0, pathX0 * o2::its::constants::Radl * o2::its::constants::Rho};
    std::array<detail::RefitMeasurementSlot, MFTNLayers> slots{};
    for (int hit = 0; hit < MFTNLayers; ++hit) {
      const auto layer = static_cast<uint16_t>(alongMomentum ? hit : MFTNLayers - 1 - hit);
      auto& slot = slots[hit];
      slot.surface = LayerId{layer};
      slot.present = true;
      const float z = kMFTStaticSurfaceCatalog[layer].referenceCoordinate;
      const float transverseDistance = (z - state.referenceCoordinate) / tanl;
      slot.measurement.frame = {z,
                                state.parameters[0] + transverseDistance * std::cos(state.parameters[2]),
                                state.parameters[1] + transverseDistance * std::sin(state.parameters[2]), 0.f};
      slot.measurement.covariance = {0.04f, 0.f, 0.04f};
      const auto result = material::calculateMaterialPhysics(expectedMomentum, state.pid, state.absCharge,
                                                             direction, expectedMaterial);
      BOOST_REQUIRE(result.ok());
      expectedMomentum = result.momentumAfterGeV;
    }
    float chi2 = 0.f;
    uint32_t acceptedHitCount = 0;
    OperationFailureReason reason{};
    BOOST_REQUIRE(detail::driveRefitLeg(state, linRef, chi2, acceptedHitCount, slots, catalog, 0.f,
                                        direction, false, 100.f, reason));
    BOOST_CHECK_EQUAL(acceptedHitCount, MFTNLayers);
    BOOST_CHECK_CLOSE(momentumScale / std::abs(state.parameters[4]), expectedMomentum, 1.e-4f);
    BOOST_CHECK(alongMomentum ? expectedMomentum < initialMomentum : expectedMomentum > initialMomentum);
  }
}

// --- 10/11: chi2-gate failure and atomicity ----------------------------------

BOOST_AUTO_TEST_CASE(Chi2GateRejectsOversizedPredictedChi2Transactionally)
{
  auto state = barrelState();
  auto linRef = barrelLinRef(state);
  const auto poisonState = state;
  const auto poisonRef = linRef;
  auto measurement = barrelMeasurement();
  measurement.frame.u += 5.f; // far outlier vs the state's predicted local Y
  const auto descriptor = cylinderDescriptor(NominalSurfaceMaterial{0.f, 0.f});
  float chi2 = 0.f;
  const float poisonChi2 = chi2;
  OperationFailureReason reason{};

  const bool ok = Propagator::propagateToMeasurement(state, linRef, descriptor, measurement, BarrelBz,
                                                     material::MaterialTraversalDirection::AlongMomentum,
                                                     true, 1.e-6f, chi2, false, reason);
  BOOST_CHECK(!ok);
  BOOST_CHECK_EQUAL(static_cast<int>(reason), static_cast<int>(OperationFailureReason::PredictedChi2Failure));
  BOOST_CHECK(bitEqual(state, poisonState));
  BOOST_CHECK(bitEqual(linRef, poisonRef));
  BOOST_CHECK_EQUAL(chi2, poisonChi2);
}

BOOST_AUTO_TEST_CASE(UnrecognizedTargetSurfaceKindFails)
{
  auto state = barrelState();
  auto linRef = barrelLinRef(state);
  const auto poisonState = state;
  const auto measurement = barrelMeasurement();
  SurfaceDescriptor descriptor = cylinderDescriptor(NominalSurfaceMaterial{0.f, 0.f});
  // SurfaceKind currently only has Cylinder/Disk (both recognized); this
  // proves the routing guard itself, not a reachable production input.
  descriptor.kind = static_cast<SurfaceKind>(0xFFu);
  float chi2 = 0.f;
  OperationFailureReason reason{};

  const bool ok = Propagator::propagateToMeasurement(state, linRef, descriptor, measurement, BarrelBz,
                                                     material::MaterialTraversalDirection::AlongMomentum,
                                                     false, 0.f, chi2, false, reason);
  BOOST_CHECK(!ok);
  BOOST_CHECK_EQUAL(static_cast<int>(reason), static_cast<int>(OperationFailureReason::SurfaceKindConversionFailure));
  BOOST_CHECK(bitEqual(state, poisonState));
}
