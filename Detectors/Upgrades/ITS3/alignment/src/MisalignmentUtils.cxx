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
#include <string>
#include <vector>
#include <array>

#include <TMatrixD.h>
#include <nlohmann/json.hpp>

#include "Framework/Logger.h"
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
      // {"f": {"1": ..., "2": ...}, "g": {...}, "h": {"2_1": ..., "4_2": ...}}
      if (inex.contains("f")) {
        for (const auto& [key, val] : inex["f"].items()) {
          sensor.inextensional.f[std::stoi(key)] = val.get<double>();
        }
      }
      if (inex.contains("g")) {
        for (const auto& [key, val] : inex["g"].items()) {
          sensor.inextensional.g[std::stoi(key)] = val.get<double>();
        }
      }
      if (inex.contains("h")) {
        for (const auto& [key, val] : inex["h"].items()) {
          const auto sep = key.find('_');
          if (sep == std::string::npos) {
            LOGP(fatal, "Inextensional h key '{}' for sensor {} must be of the form '<k>_<l>' in {}", key, id, jsonPath);
          }
          const int k = std::stoi(key.substr(0, sep));
          const int l = std::stoi(key.substr(sep + 1));
          if (l < 1) {
            LOGP(fatal, "Inextensional h key '{}' for sensor {}: l must be >= 1 (l = 0 is spanned by g) in {}", key, id, jsonPath);
          }
          sensor.inextensional.h[{k, l}] = val.get<double>();
        }
      }
      // An "inextensional" block that yields no coefficients would silently
      // produce a zero displacement field, i.e. a misalignment study that
      // looks identical to the ideal one. Most likely cause: a file still in
      // the old Fourier schema ("modes"/"alpha"/"beta").
      const auto& parsed = sensor.inextensional;
      if (parsed.f.empty() && parsed.g.empty() && parsed.h.empty()) {
        LOGP(fatal,
             "Sensor {}: 'inextensional' block in {} contains none of the expected "
             "keys 'f', 'g', 'h' - no deformation would be applied. Keys present: {}",
             id, jsonPath, [&inex] { std::string s; for (const auto& [k, v] : inex.items()) { s += (s.empty() ? "" : ", ") + k; } return s; }());
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
  const double gloX = frame.x * std::cos(frame.alpha);
  const double gloY = frame.x * std::sin(frame.alpha);
  const auto [u, v] = computeUV(gloX, gloY, frame.z, frame.sensorID, r);
  const double cPhi = phiScale(r);
  const double zOverR = frame.z / r;
  const auto& inex = sensor.inextensional;

  int maxK = 0;
  for (const auto& [k, val] : inex.f) {
    maxK = std::max(maxK, k);
  }
  for (const auto& [k, val] : inex.g) {
    maxK = std::max(maxK, k);
  }
  int maxKh = 0, maxL = 0;
  for (const auto& [kl, val] : inex.h) {
    maxKh = std::max(maxKh, kl.first);
    maxL = std::max(maxL, kl.second);
  }

  const auto pu = legendrePols(std::max(maxK, maxKh), u);
  const auto pu1 = legendrePolsD1(maxK, u);
  const auto pu2 = legendrePolsD2(maxK, u);

  // u_z = f, u_phi = -(z/r) f' + g, u_r = (z/r) f'' - g'
  double uz = 0., uphi = 0., ur = 0.;
  for (const auto& [k, fk] : inex.f) {
    uz += fk * pu[k];
    uphi += -zOverR * cPhi * fk * pu1[k];
    ur += zOverR * cPhi * cPhi * fk * pu2[k];
  }
  for (const auto& [k, gk] : inex.g) {
    uphi += gk * pu[k];
    ur += -cPhi * gk * pu1[k];
  }
  if (!inex.h.empty()) {
    const auto pv = legendrePols(maxL, v);
    for (const auto& [kl, hkl] : inex.h) {
      ur += hkl * pu[kl.first] * pv[kl.second];
    }
  }

  shift.dy = -uphi + (slopes.dydx * ur);
  shift.dz = -uz + (slopes.dzdx * ur);
  return shift;
}

} // namespace o2::its3::align
