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

#ifndef MC_H
#define MC_H

#include "Framework/Logger.h"
#include "Framework/AnalysisDataModel.h"

#include "TPDGCode.h"

namespace MC
{
bool isStable(int pdg)
{
  // Decide whether particle (pdg) is stable

  // All ions/nucleons are considered as stable
  // Nuclear code is 10LZZZAAAI
  if (pdg > 1000000000) {
    return true;
  }

  constexpr int kNstable = 18;
  int pdgStable[kNstable] = {
    kGamma,      // Photon
    kElectron,   // Electron
    kMuonPlus,   // Muon
    kPiPlus,     // Pion
    kKPlus,      // Kaon
    kK0Short,    // K0s
    kK0Long,     // K0l
    kProton,     // Proton
    kNeutron,    // Neutron
    kLambda0,    // Lambda_0
    kSigmaMinus, // Sigma Minus
    kSigmaPlus,  // Sigma Plus
    3312,        // Xsi Minus
    3322,        // Xsi
    3334,        // Omega
    kNuE,        // Electron Neutrino
    kNuMu,       // Muon Neutrino
    kNuTau       // Tau Neutrino
  };

  for (int i = 0; i < kNstable; i++) {
    if (pdg == std::abs(pdgStable[i])) {
      return true;
    }
  }

  return false;
}

// Ported from AliRoot AliStack::IsPhysicalPrimary
template <typename Particle>
bool isPhysicalPrimary(Particle const& particle)
{
  // Test if a particle is a physical primary according to the following definition:
  // Particles produced in the collision including products of strong and
  // electromagnetic decay and excluding feed-down from weak decays of strange
  // particles.

  LOGF(debug, "isPhysicalPrimary for %d", particle.index());

  const int ist = particle.statusCode();
  const int pdg = std::abs(particle.pdgCode());

  // Initial state particle
  // Solution for K0L decayed by Pythia6
  // ->
  if ((ist > 1) && (pdg != 130) && particle.producedByGenerator()) {
    LOGF(debug, "isPhysicalPrimary F1");
    return false;
  }
  if ((ist > 1) && !particle.producedByGenerator()) {
    LOGF(debug, "isPhysicalPrimary F2");
    return false;
  }
  // <-

  if (!isStable(pdg)) {
    LOGF(debug, "isPhysicalPrimary F3");
    return false;
  }

  if (particle.producedByGenerator()) {
    // Solution for K0L decayed by Pythia6
    // ->
    if (particle.has_mother0()) {
      auto mother = particle.template mother0_as<typename Particle::parent_t>();
      if (std::abs(mother.pdgCode()) == 130) {
        LOGF(debug, "isPhysicalPrimary F4");
        return false;
      }
    }
    // <-
    // check for direct photon in parton shower
    // ->
    if (pdg == 22 && particle.has_daughter0()) {
      LOGF(debug, "D %d", particle.daughter0Id());
      auto daughter = particle.template daughter0_as<typename Particle::parent_t>();
      if (daughter.pdgCode() == 22) {
        LOGF(debug, "isPhysicalPrimary F5");
        return false;
      }
    }
    // <-
    LOGF(debug, "isPhysicalPrimary T1");
    return true;
  }

  // Particle produced during transport

  LOGF(debug, "M0 %d %d", particle.producedByGenerator(), particle.mother0Id());
  auto mother = particle.template mother0_as<typename Particle::parent_t>();
  int mpdg = std::abs(mother.pdgCode());

  // Check for Sigma0
  if ((mpdg == 3212) && mother.producedByGenerator()) {
    LOGF(debug, "isPhysicalPrimary T2");
    return true;
  }

  // Check if it comes from a pi0 decay
  if ((mpdg == kPi0) && mother.producedByGenerator()) {
    LOGF(debug, "isPhysicalPrimary T3");
    return true;
  }

  // Check if this is a heavy flavor decay product
  int mfl = int(mpdg / std::pow(10, int(std::log10(mpdg))));

  // Light hadron
  if (mfl < 4) {
    LOGF(debug, "isPhysicalPrimary F6");
    return false;
  }

  // Heavy flavor hadron produced by generator
  if (mother.producedByGenerator()) {
    LOGF(debug, "isPhysicalPrimary T4");
    return true;
  }

  // To be sure that heavy flavor has not been produced in a secondary interaction
  // Loop back to the generated mother
  LOGF(debug, "M0 %d %d", mother.producedByGenerator(), mother.mother0Id());
  while (mother.has_mother0() && !mother.producedByGenerator()) {
    mother = mother.template mother0_as<typename Particle::parent_t>();
    LOGF(debug, "M+ %d %d", mother.producedByGenerator(), mother.mother0Id());
    mpdg = std::abs(mother.pdgCode());
    mfl = int(mpdg / std::pow(10, int(std::log10(mpdg))));
  }

  if (mfl < 4) {
    LOGF(debug, "isPhysicalPrimary F7");
    return false;
  } else {
    LOGF(debug, "isPhysicalPrimary T5");
    return true;
  }
}
}; // namespace MC

#endif
