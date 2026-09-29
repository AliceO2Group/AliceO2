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

/// \file Stack.h
/// \brief Definition of the Stack class
/// \author M. Al-Turany - June 2014

#ifndef O2_TRACK_TRANSPORT_UTILS_H
#define O2_TRACK_TRANSPORT_UTILS_H

#include "SimulationDataFormat/O2DatabasePDG.h"
#include <TParticle.h>
#include <TParticlePDG.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace o2::data::detail
{
// Same feature order, units, PDG masses and missing values as
// extract_o2_kine_training.C. Normalisation/imputation belongs in the graph.
inline std::vector<float> makeTrackTransportFeatures(const TParticle& particle, double eventX, double eventY, double eventZ)
{
  const double px = particle.Px();
  const double py = particle.Py();
  const double pz = particle.Pz();
  const double momentum = std::sqrt(px * px + py * py + pz * pz);
  const double pt = std::hypot(px, py);
  bool massKnown = false;
  double mass = o2::O2DatabasePDG::Mass(particle.GetPdgCode(), massKnown);
  const auto* pdgInfo = particle.GetPDG();
  if (!massKnown) mass = pdgInfo ? pdgInfo->Mass() : 0.;
  const double missing = std::numeric_limits<double>::quiet_NaN();
  const double energy = std::sqrt(std::max(0., mass * mass + momentum * momentum));
  const double eta = momentum > std::abs(pz) ? 0.5 * std::log((momentum + pz) / (momentum - pz)) : missing;
  const double theta = momentum > 0. ? std::acos(pz / momentum) : missing;
  const double rapidity = energy > std::abs(pz) ? 0.5 * std::log((energy + pz) / (energy - pz)) : missing;
  double chargeSign = pdgInfo == nullptr || pdgInfo->Charge() == 0. ? 0. : std::copysign(1., pdgInfo->Charge());
  const int code = particle.GetPdgCode();
  if (std::abs(code) >= 1000000000) {
    chargeSign = (std::abs(code) / 10000) % 1000 == 0 ? 0. : (code > 0 ? 1. : -1.);
  }
  const double dx = particle.Vx() - eventX;
  const double dy = particle.Vy() - eventY;
  const double dz = particle.Vz() - eventZ;
  const double pdg = particle.GetPdgCode();

  return {static_cast<float>(pdg), static_cast<float>(std::abs(pdg)), static_cast<float>(chargeSign),
          static_cast<float>(mass), static_cast<float>(energy), static_cast<float>(energy - mass),
          static_cast<float>(px), static_cast<float>(py), static_cast<float>(pz), static_cast<float>(momentum),
          static_cast<float>(pt), static_cast<float>(eta), static_cast<float>(std::atan2(py, px)), static_cast<float>(theta),
          static_cast<float>(rapidity), static_cast<float>(particle.Vx()), static_cast<float>(particle.Vy()),
          static_cast<float>(particle.Vz()), static_cast<float>(particle.T() * 1.e9), static_cast<float>(dx),
          static_cast<float>(dy), static_cast<float>(dz), static_cast<float>(std::hypot(particle.Vx(), particle.Vy())),
          static_cast<float>(std::hypot(dx, dy)), static_cast<float>(std::sqrt(dx * dx + dy * dy + dz * dz))};
}


inline bool transportFromOnnxScore(float score, float threshold, bool applySigmoid)
{
  if (!std::isfinite(threshold) || threshold < 0.f || threshold > 1.f) {
    throw std::runtime_error("ONNX pruning threshold must be finite and in [0,1]");
  }
  if (!std::isfinite(score)) {
    throw std::runtime_error("ONNX pruning returned a non-finite score; refusing to reject a track");
  }
  if (applySigmoid) {
    score = score >= 0.f ? 1.f / (1.f + std::exp(-score)) : std::exp(score) / (1.f + std::exp(score));
  }
  if (score < 0.f || score > 1.f) {
    throw std::runtime_error("ONNX pruning probability is outside [0,1]");
  }
  // Class 1: this track and all descendants produce zero detector hits.
  return score < threshold;
}
} // namespace o2::data::detail
#endif
