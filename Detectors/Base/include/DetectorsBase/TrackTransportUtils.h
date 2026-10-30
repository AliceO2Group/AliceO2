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

/// Helpers for the versioned simnet.birth.v1 raw-input transport contract.
#ifndef O2_TRACK_TRANSPORT_UTILS_H
#define O2_TRACK_TRANSPORT_UTILS_H

#include "SimulationDataFormat/O2DatabasePDG.h"
#include <TDatabasePDG.h>
#include <TParticle.h>
#include <TParticlePDG.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace o2::data::detail
{
// Stable, append-only ABI. Models Gather their selected columns in the ONNX
// graph; unused NaN/Inf values must not inhibit an otherwise valid prediction.
constexpr std::array<std::string_view, 34> TrackTransportFeatureNames{
  "charge_sign",
  "mass",
  "ekin",
  "pt",
  "eta",
  "phi",
  "theta",
  "vx",
  "vy",
  "vz",
  "r_xy",
  "t_ns",
  "pdg_class",
  "lifetime_ns",
  "bending_radius_cm",
  "z_at_r40",
  "z_at_r250",
  "medium_code",
  "has_gen_mother",
  "pdg",
  "abs_pdg",
  "energy",
  "px",
  "py",
  "pz",
  "p",
  "rapidity",
  "dx_from_event",
  "dy_from_event",
  "dz_from_event",
  "r_from_event_xy",
  "r3_from_event",
  "mother_pdg",
  "mother_abs_pdg"};
constexpr size_t TrackTransportFeatureCount = TrackTransportFeatureNames.size();

inline bool validTrackTransportFeatures(const std::vector<float>& features)
{
  // Numerical/domain checks belong in the graph AFTER selecting model inputs.
  return features.size() == TrackTransportFeatureCount;
}

inline float trackTransportMediumCode(std::string_view medium)
{
  return medium == "PIPE_VACUUM" ? 1.f : medium == "TPC_DriftGas2" ? 2.f
                                       : medium == "TPC_Air"       ? 3.f
                                                                   : 0.f;
}

inline double trackTransportZAtRadius(const TParticle& p, double radius)
{
  const double pt = std::hypot(p.Px(), p.Py());
  if (pt <= 0.) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double r0 = std::hypot(p.Vx(), p.Vy());
  if (r0 >= radius) {
    return p.Vz();
  }
  const double b = p.Vx() * p.Px() + p.Vy() * p.Py();
  const double c = r0 * r0 - radius * radius;
  return p.Vz() + (-b + std::sqrt(std::max(0., b * b - pt * pt * c))) / (pt * pt) * p.Pz();
}

// Birth-time quantities only, in TrackTransportFeatureNames order. Missing
// context is NaN; a graph that does not select that context can still classify.
// motherPdg is the FIRST mother's PDG, including transport mothers.
inline std::vector<float> makeTrackTransportFeatures(
  const TParticle& particle, double motherPdg, float mediumCode,
  double eventX = std::numeric_limits<double>::quiet_NaN(),
  double eventY = std::numeric_limits<double>::quiet_NaN(),
  double eventZ = std::numeric_limits<double>::quiet_NaN())
{
  const double px = particle.Px(), py = particle.Py(), pz = particle.Pz();
  const double momentum = std::sqrt(px * px + py * py + pz * pz);
  const double pt = std::hypot(px, py);
  const int code = particle.GetPdgCode();
  const auto absCode = std::abs(static_cast<int64_t>(code));
  bool massKnown = false;
  double mass = o2::O2DatabasePDG::Mass(code, massKnown);
  const auto* pdg = TDatabasePDG::Instance()->GetParticle(code);
  if (!massKnown) {
    mass = pdg ? pdg->Mass() : 0.;
  }
  double charge = !pdg || pdg->Charge() == 0. ? 0. : std::copysign(1., pdg->Charge());
  if (absCode >= 1000000000) {
    charge = (absCode / 10000) % 1000 == 0 ? 0. : (code > 0 ? 1. : -1.);
  }
  int pdgClass = 9;
  if (absCode == 22) {
    pdgClass = 0;
  } else if (absCode == 11) {
    pdgClass = 1;
  } else if (absCode == 13) {
    pdgClass = 2;
  } else if (absCode == 12 || absCode == 14 || absCode == 16) {
    pdgClass = 3;
  } else if (absCode >= 1000000000) {
    pdgClass = 8;
  } else if (absCode >= 1000 && absCode < 10000) {
    pdgClass = charge != 0. ? 6 : 7;
  } else if (absCode >= 100 && absCode < 1000) {
    pdgClass = charge != 0. ? 4 : 5;
  }
  const double missing = std::numeric_limits<double>::quiet_NaN();
  const double energy = std::sqrt(std::max(0., mass * mass + momentum * momentum));
  const double eta = momentum > std::abs(pz) ? 0.5 * std::log((momentum + pz) / (momentum - pz)) : missing;
  const double theta = momentum > 0. ? std::acos(pz / momentum) : missing;
  const double rapidity = energy > std::abs(pz) ? 0.5 * std::log((energy + pz) / (energy - pz)) : missing;
  const double dx = particle.Vx() - eventX, dy = particle.Vy() - eventY, dz = particle.Vz() - eventZ;
  const double hasMother = std::isfinite(motherPdg) ? (motherPdg != 0. ? 1. : 0.) : missing;
  return {static_cast<float>(charge), static_cast<float>(mass), static_cast<float>(energy - mass),
          static_cast<float>(pt), static_cast<float>(eta), static_cast<float>(std::atan2(py, px)),
          static_cast<float>(theta), static_cast<float>(particle.Vx()), static_cast<float>(particle.Vy()),
          static_cast<float>(particle.Vz()), static_cast<float>(std::hypot(particle.Vx(), particle.Vy())),
          static_cast<float>(particle.T() * 1.e9), static_cast<float>(pdgClass),
          static_cast<float>(pdg ? pdg->Lifetime() * 1.e9 : 0.),
          static_cast<float>(charge != 0. ? pt / (0.3 * 0.5) * 100. : 0.),
          static_cast<float>(trackTransportZAtRadius(particle, 40.)),
          static_cast<float>(trackTransportZAtRadius(particle, 250.)), mediumCode, static_cast<float>(hasMother),
          static_cast<float>(code), static_cast<float>(absCode), static_cast<float>(energy),
          static_cast<float>(px), static_cast<float>(py), static_cast<float>(pz), static_cast<float>(momentum),
          static_cast<float>(rapidity), static_cast<float>(dx), static_cast<float>(dy), static_cast<float>(dz),
          static_cast<float>(std::hypot(dx, dy)), static_cast<float>(std::sqrt(dx * dx + dy * dy + dz * dz)),
          static_cast<float>(motherPdg), static_cast<float>(std::abs(motherPdg))};
}

inline bool validTrackTransportOutput(const std::vector<std::vector<int64_t>>& shapes, int index)
{
  if (shapes.size() != 1 || shapes[0].empty() || shapes[0].size() > 2 ||
      (shapes[0][0] != 1 && shapes[0][0] != -1) || index < 0) {
    return false;
  }
  // The existing NN exports squeeze(-1), so [batch] is a single score.
  return shapes[0].size() == 1 ? index == 0 : index < shapes[0][1];
}

inline bool transportFromOnnxScore(float score, double threshold, bool applySigmoid, bool invert = false)
{
  if (!std::isfinite(threshold) || threshold < 0. || threshold > std::nextafter(1., std::numeric_limits<double>::infinity())) {
    throw std::runtime_error("ONNX pruning requires an explicit probability threshold (or nextafter(1,+inf) to keep all)");
  }
  // Invalid predictions always keep the track, including with inversion enabled.
  if (!std::isfinite(score) || (!applySigmoid && (score < 0.f || score > 1.f))) {
    return true;
  }
  if (applySigmoid) {
    score = score >= 0.f ? 1.f / (1.f + std::exp(-score)) : std::exp(score) / (1.f + std::exp(score));
  }
  const bool transport = static_cast<double>(score) < threshold; // class 1 is a hit-free subtree
  return invert ? !transport : transport;
}
} // namespace o2::data::detail
#endif
