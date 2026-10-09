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

#include "ITSMFTTracking/TripletFitting.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace o2::itsmft::tracking
{
namespace
{

constexpr std::size_t NMeasurementCoordinates = 9;
constexpr std::size_t NCoordinates = NMeasurementCoordinates;
constexpr std::size_t NAdjacentKinks = 4;

using KinkVector = std::array<float, NAdjacentKinks>;
using KinkCovariance = std::array<std::array<float, NAdjacentKinks>, NAdjacentKinks>;

struct DualNumber {
  float value{0.};
  std::array<float, NCoordinates> derivative{};

  static DualNumber variable(float val, std::size_t index) noexcept
  {
    DualNumber result{val};
    result.derivative[index] = 1.;
    return result;
  }
};

DualNumber operator+(const DualNumber& lhs, const DualNumber& rhs) noexcept
{
  DualNumber result{lhs.value + rhs.value};
  for (std::size_t i = 0; i < NCoordinates; ++i) {
    result.derivative[i] = lhs.derivative[i] + rhs.derivative[i];
  }
  return result;
}

DualNumber operator-(const DualNumber& lhs, const DualNumber& rhs) noexcept
{
  DualNumber result{lhs.value - rhs.value};
  for (std::size_t i = 0; i < NCoordinates; ++i) {
    result.derivative[i] = lhs.derivative[i] - rhs.derivative[i];
  }
  return result;
}

DualNumber operator-(const DualNumber& value) noexcept
{
  DualNumber result{-value.value};
  for (std::size_t i = 0; i < NCoordinates; ++i) {
    result.derivative[i] = -value.derivative[i];
  }
  return result;
}

DualNumber operator*(const DualNumber& lhs, const DualNumber& rhs) noexcept
{
  DualNumber result{lhs.value * rhs.value};
  for (std::size_t i = 0; i < NCoordinates; ++i) {
    result.derivative[i] = lhs.derivative[i] * rhs.value + lhs.value * rhs.derivative[i];
  }
  return result;
}

DualNumber operator/(const DualNumber& lhs, const DualNumber& rhs) noexcept
{
  const float inverse = 1. / rhs.value;
  DualNumber result{lhs.value * inverse};
  for (std::size_t i = 0; i < NCoordinates; ++i) {
    result.derivative[i] = (lhs.derivative[i] - result.value * rhs.derivative[i]) * inverse;
  }
  return result;
}

DualNumber squareRoot(const DualNumber& argument) noexcept
{
  const float root = std::sqrt(argument.value);
  DualNumber result{root};
  const float scale = 0.5 / root;
  for (std::size_t i = 0; i < NCoordinates; ++i) {
    result.derivative[i] = scale * argument.derivative[i];
  }
  return result;
}

DualNumber arcSine(const DualNumber& argument) noexcept
{
  DualNumber result{std::asin(argument.value)};
  const float scale = 1. / std::sqrt(1. - argument.value * argument.value);
  for (std::size_t i = 0; i < NCoordinates; ++i) {
    result.derivative[i] = scale * argument.derivative[i];
  }
  return result;
}

DualNumber arcTangent2(const DualNumber& y, const DualNumber& x) noexcept
{
  DualNumber result{std::atan2(y.value, x.value)};
  const float denominator = x.value * x.value + y.value * y.value;
  for (std::size_t i = 0; i < NCoordinates; ++i) {
    result.derivative[i] = (x.value * y.derivative[i] - y.value * x.derivative[i]) / denominator;
  }
  return result;
}

struct SegmentGeometry {
  DualNumber bendingAngle;
  DualNumber transverseArcLength;
  DualNumber cotangentTheta;
  DualNumber sineTheta;
  DualNumber cosineTheta;
  DualNumber index;
};

bool makeSegmentGeometry(const DualNumber& transverseCurvature, const DualNumber& chordLength,
                         const DualNumber& deltaZ, SegmentGeometry& result) noexcept
{
  const DualNumber halfSine = DualNumber{0.5} * transverseCurvature * chordLength;
  if (std::abs(halfSine.value) >= 1.) {
    return false;
  }

  const DualNumber halfSine2 = halfSine * halfSine;
  const DualNumber halfSine4 = halfSine2 * halfSine2;
  DualNumber asinOverArgument;
  DualNumber angleCotangent;
  const DualNumber halfAngle = arcSine(halfSine);
  if (std::abs(halfSine.value) < 1.e-4) {
    asinOverArgument = DualNumber{1.} + halfSine2 * DualNumber{1. / 6.} + halfSine4 * DualNumber{3. / 40.};
    angleCotangent = DualNumber{1.} - halfSine2 * DualNumber{1. / 3.} - halfSine4 * DualNumber{2. / 15.};
  } else {
    asinOverArgument = halfAngle / halfSine;
    angleCotangent = halfAngle * squareRoot(DualNumber{1.} - halfSine2) / halfSine;
  }

  const DualNumber bendingAngle = DualNumber{2.} * halfAngle;
  const DualNumber transverseArcLength = chordLength * asinOverArgument;
  const DualNumber cotangentTheta = deltaZ / transverseArcLength;
  const DualNumber sineTheta = DualNumber{1.} / squareRoot(DualNumber{1.} + cotangentTheta * cotangentTheta);
  const DualNumber cosineTheta = cotangentTheta * sineTheta;
  const DualNumber index = DualNumber{1.} /
                           (angleCotangent * sineTheta * sineTheta + cosineTheta * cosineTheta);
  if (transverseArcLength.value <= 0. || sineTheta.value <= 0. || index.value <= 0.) {
    return false;
  }
  result = {bendingAngle, transverseArcLength, cotangentTheta, sineTheta, cosineTheta, index};
  return true;
}

struct TripletGeometry {
  DualNumber phiTilde;
  DualNumber thetaTilde;
  DualNumber rhoPhi;
  DualNumber rhoTheta;
};

bool makeTripletGeometry(const std::array<GlobalMeasurement, 3>& measurements,
                         TripletGeometry& result) noexcept
{
  std::array<std::array<DualNumber, 3>, 3> point{};
  for (std::size_t hit = 0; hit < measurements.size(); ++hit) {
    const std::array<float, 3> position{
      measurements[hit].x, measurements[hit].y, measurements[hit].z};
    for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
      const std::size_t index = 3 * hit + coordinate;
      point[hit][coordinate] = DualNumber::variable(position[coordinate], index);
    }
  }

  const DualNumber dx01 = point[1][0] - point[0][0];
  const DualNumber dy01 = point[1][1] - point[0][1];
  const DualNumber dz01 = point[1][2] - point[0][2];
  const DualNumber dx12 = point[2][0] - point[1][0];
  const DualNumber dy12 = point[2][1] - point[1][1];
  const DualNumber dz12 = point[2][2] - point[1][2];
  const DualNumber dx02 = point[2][0] - point[0][0];
  const DualNumber dy02 = point[2][1] - point[0][1];
  const DualNumber length01 = squareRoot(dx01 * dx01 + dy01 * dy01);
  const DualNumber length12 = squareRoot(dx12 * dx12 + dy12 * dy12);
  const DualNumber length02 = squareRoot(dx02 * dx02 + dy02 * dy02);
  if (length01.value <= 0. ||
      length12.value <= 0. || length02.value <= 0.) {
    return false;
  }

  const DualNumber cross = dx01 * dy12 - dy01 * dx12;
  const DualNumber transverseCurvature = DualNumber{2.} * cross / (length01 * length12 * length02);
  SegmentGeometry firstSegment;
  SegmentGeometry secondSegment;
  if (!makeSegmentGeometry(transverseCurvature, length01, dz01, firstSegment) ||
      !makeSegmentGeometry(transverseCurvature, length12, dz12, secondSegment)) {
    return false;
  }

  const DualNumber theta01 = arcTangent2(firstSegment.transverseArcLength, dz01);
  const DualNumber theta12 = arcTangent2(secondSegment.transverseArcLength, dz12);
  const DualNumber phiTilde = DualNumber{0.5} *
                              (firstSegment.bendingAngle * firstSegment.index +
                               secondSegment.bendingAngle * secondSegment.index);
  const DualNumber thetaTilde = theta12 - theta01 +
                                (DualNumber{1.} - secondSegment.index) * secondSegment.cotangentTheta -
                                (DualNumber{1.} - firstSegment.index) * firstSegment.cotangentTheta;
  const DualNumber rhoPhi = DualNumber{-0.5} *
                            (firstSegment.transverseArcLength * firstSegment.index / firstSegment.sineTheta +
                             secondSegment.transverseArcLength * secondSegment.index / secondSegment.sineTheta);

  DualNumber rhoTheta;
  const float maximumHalfSine = 0.5 * std::abs(transverseCurvature.value) *
                                std::max(length01.value, length12.value);
  if (maximumHalfSine < 1.e-4) {
    rhoTheta = transverseCurvature *
               (length12 * length12 * secondSegment.cosineTheta -
                length01 * length01 * firstSegment.cosineTheta) /
               DualNumber{12.};
  } else {
    rhoTheta = ((DualNumber{1.} - firstSegment.index) * firstSegment.cotangentTheta / firstSegment.sineTheta -
                (DualNumber{1.} - secondSegment.index) * secondSegment.cotangentTheta / secondSegment.sineTheta) /
               transverseCurvature;
  }

  if (rhoPhi.value == 0.) {
    return false;
  }
  result = {phiTilde, thetaTilde, rhoPhi, rhoTheta};
  return true;
}

float covarianceContraction(const std::array<float, 3>& left,
                            const GlobalCovariance3F& covariance,
                            const std::array<float, 3>& right) noexcept
{
  return left[0] * (covariance.xx * right[0] + covariance.xy * right[1] + covariance.xz * right[2]) +
         left[1] * (covariance.xy * right[0] + covariance.yy * right[1] + covariance.yz * right[2]) +
         left[2] * (covariance.xz * right[0] + covariance.yz * right[1] + covariance.zz * right[2]);
}

bool choleskyDecompose(const KinkCovariance& covariance,
                       KinkCovariance& lower) noexcept
{
  for (std::size_t row = 0; row < NAdjacentKinks; ++row) {
    for (std::size_t column = 0; column <= row; ++column) {
      float value = covariance[row][column];
      for (std::size_t k = 0; k < column; ++k) {
        value -= lower[row][k] * lower[column][k];
      }
      if (row == column) {
        if (value <= 0.) {
          return false;
        }
        lower[row][column] = std::sqrt(value);
      } else {
        lower[row][column] = value / lower[column][column];
      }
    }
  }
  return true;
}

bool choleskySolve(const KinkCovariance& lower, const KinkVector& right,
                   KinkVector& solution) noexcept
{
  KinkVector intermediate{};
  for (std::size_t row = 0; row < NAdjacentKinks; ++row) {
    float value = right[row];
    for (std::size_t column = 0; column < row; ++column) {
      value -= lower[row][column] * intermediate[column];
    }
    intermediate[row] = value / lower[row][row];
  }
  for (int row = static_cast<int>(NAdjacentKinks) - 1; row >= 0; --row) {
    float value = intermediate[row];
    for (std::size_t column = static_cast<std::size_t>(row) + 1;
         column < NAdjacentKinks; ++column) {
      value -= lower[column][row] * solution[column];
    }
    solution[row] = value / lower[row][row];
  }
  return true;
}

float dotProduct(const KinkVector& left, const KinkVector& right) noexcept
{
  float result = 0.;
  for (std::size_t i = 0; i < NAdjacentKinks; ++i) {
    result += left[i] * right[i];
  }
  return result;
}

bool referenceSinTheta(const GlobalMeasurement& first,
                       const GlobalMeasurement& third,
                       float& sineTheta) noexcept
{
  const float dx = third.x - first.x;
  const float dy = third.y - first.y;
  const float dz = third.z - first.z;
  const float transverse = std::hypot(dx, dy);
  const float length = std::hypot(transverse, dz);
  sineTheta = transverse / length;
  return sineTheta > 0. && sineTheta <= 1.;
}

} // namespace

bool makeTripletFitFactor(
  const std::array<GlobalMeasurement, 3>& measurements,
  TripletFitFactor& result) noexcept
{
  TripletGeometry geometry;
  if (!makeTripletGeometry(measurements, geometry)) {
    return false;
  }
  const float kappaReference = -geometry.phiTilde.value / geometry.rhoPhi.value;
  TripletFitFactor scratch{
    {geometry.thetaTilde.value, geometry.phiTilde.value},
    {geometry.rhoTheta.value, geometry.rhoPhi.value},
    {}};

  for (std::size_t hit = 0; hit < measurements.size(); ++hit) {
    for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
      const std::size_t index = 3 * hit + coordinate;
      const float gradientTheta = geometry.thetaTilde.derivative[index] +
                                  kappaReference * geometry.rhoTheta.derivative[index];
      const float gradientPhi = geometry.phiTilde.derivative[index] +
                                kappaReference * geometry.rhoPhi.derivative[index];
      scratch.h[hit].theta[coordinate] = gradientTheta;
      scratch.h[hit].phi[coordinate] = gradientPhi;
    }
  }
  if (!scratch.isValid()) {
    return false;
  }
  result = scratch;
  return true;
}

bool fitAdjacentTripletFactors(
  const TripletFitFactor& firstFactor,
  const TripletFitFactor& secondFactor,
  const std::array<GlobalMeasurement, 4>& measurements,
  const std::array<float, 2>& angularVariance,
  AdjacentTripletFitResult& result) noexcept
{
  std::array<float, 2> sineTheta{};
  if (!referenceSinTheta(measurements[0], measurements[2], sineTheta[0]) ||
      !referenceSinTheta(measurements[1], measurements[3], sineTheta[1])) {
    return false;
  }

  const KinkVector psi{
    firstFactor.psi.theta, firstFactor.psi.phi,
    secondFactor.psi.theta, secondFactor.psi.phi};
  const KinkVector rho{
    firstFactor.rho.theta, firstFactor.rho.phi,
    secondFactor.rho.theta, secondFactor.rho.phi};
  KinkCovariance covariance{};
  covariance[0][0] = angularVariance[0];
  covariance[1][1] = angularVariance[0] / (sineTheta[0] * sineTheta[0]);
  covariance[2][2] = angularVariance[1];
  covariance[3][3] = angularVariance[1] / (sineTheta[1] * sineTheta[1]);

  // Build H for four unique hits. The factors use slots (0,1,2) and (1,2,3),
  // so shared hits contribute to the cross-triplet covariance.
  std::array<std::array<std::array<float, 3>, NAdjacentKinks>, 4> gradients{};
  for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
    for (std::size_t hit = 0; hit < 3; ++hit) {
      gradients[hit][0][coordinate] = firstFactor.h[hit].theta[coordinate];
      gradients[hit][1][coordinate] = firstFactor.h[hit].phi[coordinate];
      gradients[hit + 1][2][coordinate] = secondFactor.h[hit].theta[coordinate];
      gradients[hit + 1][3][coordinate] = secondFactor.h[hit].phi[coordinate];
    }
  }
  for (std::size_t hit = 0; hit < measurements.size(); ++hit) {
    for (std::size_t row = 0; row < NAdjacentKinks; ++row) {
      for (std::size_t column = 0; column <= row; ++column) {
        const float contribution = covarianceContraction(
          gradients[hit][row], measurements[hit].covariance, gradients[hit][column]);
        covariance[row][column] += contribution;
        if (row != column) {
          covariance[column][row] += contribution;
        }
      }
    }
  }

  KinkCovariance lower{};
  KinkVector precisionPsi{};
  KinkVector precisionRho{};
  if (!choleskyDecompose(covariance, lower) ||
      !choleskySolve(lower, psi, precisionPsi) ||
      !choleskySolve(lower, rho, precisionRho)) {
    return false;
  }
  const float rhoPrecisionPsi = dotProduct(rho, precisionPsi);
  const float rhoPrecisionRho = dotProduct(rho, precisionRho);
  const float psiPrecisionPsi = dotProduct(psi, precisionPsi);
  if (rhoPrecisionRho <= 0.) {
    return false;
  }

  const float curvature = -rhoPrecisionPsi / rhoPrecisionRho;
  const float curvatureVariance = 1. / rhoPrecisionRho;
  const float removedCurvatureTerm = rhoPrecisionPsi * rhoPrecisionPsi / rhoPrecisionRho;
  float chi2 = psiPrecisionPsi - removedCurvatureTerm;
  const float chi2Tolerance = 128. * std::numeric_limits<float>::epsilon() *
                              std::max(std::abs(psiPrecisionPsi), std::abs(removedCurvatureTerm));
  if (chi2 < 0. && chi2 >= -chi2Tolerance) {
    chi2 = 0.;
  }
  if (curvatureVariance <= 0. || chi2 < 0.) {
    return false;
  }

  result = {curvature, curvatureVariance, chi2};
  return true;
}

} // namespace o2::itsmft::tracking
