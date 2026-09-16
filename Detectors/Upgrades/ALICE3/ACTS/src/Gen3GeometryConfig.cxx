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

///
/// \file Gen3GeometryConfig.cxx
/// \author Paolo Butti
///
/// Ported from actsO2 (ActsAlgorithms/Geometry/src/Gen3GeometryConfig.cpp).
///
/// JSON loader for Gen3GeometryConfig. The JSON is the single source of truth:
/// a missing file, a parse error, or a missing required key is a hard error.
///

#include "ALICE3ACTS/Gen3GeometryConfig.h"

#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace o2::alice3::gen3cfg
{

namespace
{

using nlohmann::json;

/// Return j[key] or throw a clear error naming the missing key.
const json& require(const json& j, const char* key) {
  const auto it = j.find(key);
  if (it == j.end()) {
    throw std::runtime_error(
        std::string("Gen3 config: missing required key '") + key + "'");
  }
  return *it;
}

template <typename T>
T get(const json& j, const char* key) {
  try {
    return require(j, key).get<T>();
  } catch (const json::exception& e) {
    throw std::runtime_error(std::string("Gen3 config: key '") + key +
                             "' has the wrong type (" + e.what() + ")");
  }
}

PassiveCylinderCfg parsePassiveCylinder(const json& j) {
  return {get<std::string>(j, "name"), get<double>(j, "r"),
          get<double>(j, "halfZ"), get<double>(j, "zCentre")};
}

ForwardCylinderCfg parseForwardCylinder(const json& j) {
  return {get<std::string>(j, "name"), get<double>(j, "r"),
          get<double>(j, "zMin"), get<double>(j, "zMax")};
}

PassiveDiscCfg parsePassiveDisc(const json& j) {
  return {get<std::string>(j, "name"), get<double>(j, "z"),
          get<double>(j, "rMin"), get<double>(j, "rMax")};
}

} // namespace

Gen3GeometryConfig loadGen3GeometryConfig(const std::string& jsonPath) {
  std::ifstream in(jsonPath);
  if (!in.is_open()) {
    throw std::runtime_error("Gen3 config: cannot open JSON file '" + jsonPath +
                             "'");
  }

  json j;
  try {
    in >> j;
  } catch (const std::exception& e) {
    throw std::runtime_error("Gen3 config: failed to parse '" + jsonPath +
                             "': " + e.what());
  }

  Gen3GeometryConfig c;

  c.sensitiveMatches = get<std::vector<std::string>>(j, "sensitiveMatches");
  c.endOfStaveMatches = get<std::vector<std::string>>(j, "endOfStaveMatches");
  c.endOfStaveRTol = get<double>(j, "endOfStaveRTol");

  c.axesThinZ = get<std::string>(j, "axesThinZ");
  c.axesThinY = get<std::string>(j, "axesThinY");

  c.rInnerCoreMax = get<double>(j, "rInnerCoreMax");
  c.rMainMax = get<double>(j, "rMainMax");
  c.zCentralMax = get<double>(j, "zCentralMax");
  c.zMainMax = get<double>(j, "zMainMax");

  c.tolVertexDetector = get<double>(j, "tolVertexDetector");
  c.tolTrkBarrel = get<double>(j, "tolTrkBarrel");
  c.tolItof = get<double>(j, "tolItof");
  c.tolOtof = get<double>(j, "tolOtof");
  c.tolFt3Disc = get<double>(j, "tolFt3Disc");

  c.fwdDiscRMax = get<double>(j, "fwdDiscRMax");

  c.matBinsPhi = get<std::size_t>(j, "matBinsPhi");
  c.matBinsZ = get<std::size_t>(j, "matBinsZ");
  c.matBinsR = get<std::size_t>(j, "matBinsR");

  for (const auto& e : require(j, "passiveCylinders")) {
    c.passiveCylinders.push_back(parsePassiveCylinder(e));
  }
  for (const auto& e : require(j, "forwardCylinders")) {
    c.forwardCylinders.push_back(parseForwardCylinder(e));
  }
  for (const auto& e : require(j, "passiveDiscs")) {
    c.passiveDiscs.push_back(parsePassiveDisc(e));
  }

  const json& v = require(j, "volumeIds");
  c.volFwdNeg = get<std::uint64_t>(v, "fwdNeg");
  c.volFt3InnerNeg = get<std::uint64_t>(v, "ft3InnerNeg");
  c.volInnerBarrel = get<std::uint64_t>(v, "innerBarrel");
  c.volOuterTrackerBarrel = get<std::uint64_t>(v, "outerTrackerBarrel");
  c.volFt3InnerPos = get<std::uint64_t>(v, "ft3InnerPos");
  c.volFwdPos = get<std::uint64_t>(v, "fwdPos");
  c.volOtof = get<std::uint64_t>(v, "otof");

  return c;
}

} // namespace o2::alice3::gen3cfg
