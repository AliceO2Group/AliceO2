// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ITS3Align/MisalignmentUtils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>
#include <array>

#include <TMatrixD.h>
#include <nlohmann/json.hpp>

#include "Framework/Logger.h"
#include "CommonConstants/MathConstants.h"
#include "ITS3Base/SpecsV2.h"

namespace o2::its3::align
{

bool MisalignmentModel::empty() const noexcept
{
  return std::all_of(sensors.begin(), sensors.end(), [](const auto& sensor) { return sensor.empty(); });
}

MisalignmentModel loadMisalignmentModel(const std::string& jsonPath)
{
  MisalignmentModel model;
  if (jsonPath.empty()) {
    return model;
  }

  std::ifstream f(jsonPath);
  if (!f.is_open()) {
    LOGP(fatal, "Cannot open misalignment JSON file: {}", jsonPath);
  }

  using json = nlohmann::json;
  const auto data = json::parse(f);
  for (const auto& item : data) {
    const int id = item["id"].get<int>();
    if (id < 0 || id >= static_cast<int>(MisalignmentModel::NSensors)) {
      LOGP(fatal, "Misalignment sensor id {} out of range [0, {}) in {}", id, MisalignmentModel::NSensors, jsonPath);
    }

    auto& sensor = model[id];
    if (item.contains("matrix")) {
      auto v = item["matrix"].get<std::vector<std::vector<double>>>();
      if (v.empty()) {
        LOGP(fatal, "Legendre matrix for sensor {} is empty in {}", id, jsonPath);
      }
      TMatrixD m(v.size(), v.back().size());
      for (std::size_t r{0}; r < v.size(); ++r) {
        for (std::size_t c{0}; c < v[r].size(); ++c) {
          m(r, c) = v[r][c];
        }
      }
      sensor.legendre = o2::math_utils::Legendre2DPolynominal(m);
      sensor.hasLegendre = true;
    }
    if (item.contains("inextensional")) {
      const auto& inex = item["inextensional"];
      sensor.hasInextensional = true;
      if (inex.contains("modes")) {
        for (const auto& [key, val] : inex["modes"].items()) {
          sensor.inextensional.modes[std::stoi(key)] = val.get<std::array<double, 4>>();
        }
      }
      if (inex.contains("alpha")) {
        sensor.inextensional.alpha = inex["alpha"].get<double>();
      }
      if (inex.contains("beta")) {
        sensor.inextensional.beta = inex["beta"].get<double>();
      }
    }
  }

  return model;
}

MisalignmentShift evaluateLegendreShift(const SensorMisalignment& sensor, const MisalignmentFrame& frame, const TrackSlopes& slopes)
{
  MisalignmentShift shift;
  if (!sensor.hasLegendre) {
    return shift;
  }

  const double gloX = frame.x * std::cos(frame.alpha);
  const double gloY = frame.x * std::sin(frame.alpha);
  const double gloZ = frame.z;
  auto [u, v] = computeUV(gloX, gloY, gloZ, frame.sensorID, constants::radii[frame.layerID]);
  const double h = sensor.legendre(u, v);

  // this is the shift due to back-projection of the track on the ideal surface
  shift.dy = slopes.dydx * h;
  shift.dz = slopes.dzdx * h;

  if (std::abs(u) > o2::constants::math::Almost0) {
    // account for additional tangential movement due to radial shift
    // we have to approximate the difference in arc-length from the reference pnt on the deformed surface
    // this is done by integrating the height function via Gauss-Legendre quadrature (from Numerical recipes 4.6 [1])
    constexpr std::array<double, 8> x = {-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498, 0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363};
    constexpr std::array<double, 8> w = {0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620, 0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763};
    const double mid = 0.5 * u;
    const double half = 0.5 * u;
    double integral = 0.;
    for (int i = 0; i < 8; ++i) {
      const double up = mid + (half * x[i]);
      integral += w[i] * sensor.legendre(up, v);
    }
    integral *= half;
    shift.dy += 0.5 * getSensorPhiWidth(frame.sensorID, constants::radii[frame.layerID]) * integral;
  }

  const double newGloY = gloY + (shift.dy * std::cos(frame.alpha));
  const double newGloX = gloX - (shift.dy * std::sin(frame.alpha));
  const double newGloZ = gloZ + shift.dz;
  auto [uNew, vNew] = computeUV(newGloX, newGloY, newGloZ, frame.sensorID, constants::radii[frame.layerID]);
  shift.accepted = std::abs(uNew) <= 1. && std::abs(vNew) <= 1.;
  return shift;
}

MisalignmentShift evaluateInextensionalShift(const SensorMisalignment& sensor, const MisalignmentFrame& frame, const TrackSlopes& slopes)
{
  MisalignmentShift shift;
  if (!sensor.hasInextensional) {
    return shift;
  }

  const double r = constants::radii[frame.layerID];
  const double phi = std::atan2(r * std::sin(frame.alpha), r * std::cos(frame.alpha));
  const double z = frame.z;
  const auto& inex = sensor.inextensional;

  double uz = 0., uphi = 0., ur = 0.;
  for (const auto& [n, coeffs] : inex.modes) {
    const double a_n = coeffs[0], b_n = coeffs[1], c_n = coeffs[2], d_n = coeffs[3];
    const double sn = std::sin(n * phi);
    const double cn = std::cos(n * phi);
    const int n2 = n * n;

    const double fn = (a_n * cn) + (b_n * sn);
    const double fpn = (-n * a_n * sn) + (n * b_n * cn);
    const double fppn = (-n2 * a_n * cn) - (n2 * b_n * sn);
    const double gn = (c_n * cn) + (d_n * sn);
    const double gpn = (-n * c_n * sn) + (n * d_n * cn);

    uz += fn;
    uphi += -(z / r) * fpn + gn;
    ur += (z / r) * fppn - gpn;
  }

  uz += inex.alpha * phi;
  uphi += -(z / r) * inex.alpha + inex.beta * phi;
  ur += -inex.beta;

  shift.dy = -uphi + (slopes.dydx * ur);
  shift.dz = -uz + (slopes.dzdx * ur);
  return shift;
}

} // namespace o2::its3::align
