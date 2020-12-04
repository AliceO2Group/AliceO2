// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef MC_H
#define MC_H

#include "Framework/Logger.h"

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
template <typename TMCParticle, typename TMCParticles>
bool isPhysicalPrimary(TMCParticles& mcParticles, TMCParticle const& particle)
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
    return false;
  }
  if ((ist > 1) && !particle.producedByGenerator()) {
    return false;
  }
  // <-

  if (!isStable(pdg)) {
    return false;
  }

  if (particle.producedByGenerator()) {
    // Solution for K0L decayed by Pythia6
    // ->
    if (particle.mother0() != -1) {
      auto mother = mcParticles.iteratorAt(particle.mother0());
      if (std::abs(mother.pdgCode()) == 130) {
        return false;
      }
    }
    // <-
    // check for direct photon in parton shower
    // ->
    if (pdg == 22 && particle.daughter0() != -1) {
      LOGF(debug, "D %d", particle.daughter0());
      auto daughter = mcParticles.iteratorAt(particle.daughter0());
      if (daughter.pdgCode() == 22) {
        return false;
      }
    }
    // <-
    return true;
  }

  // Particle produced during transport

  LOGF(debug, "M0 %d %d", particle.producedByGenerator(), particle.mother0());
  auto mother = mcParticles.iteratorAt(particle.mother0());
  int mpdg = std::abs(mother.pdgCode());

  // Check for Sigma0
  if ((mpdg == 3212) && mother.producedByGenerator()) {
    return true;
  }

  // Check if it comes from a pi0 decay
  if ((mpdg == kPi0) && mother.producedByGenerator()) {
    return true;
  }

  // Check if this is a heavy flavor decay product
  int mfl = int(mpdg / std::pow(10, int(std::log10(mpdg))));

  // Light hadron
  if (mfl < 4) {
    return false;
  }

  // Heavy flavor hadron produced by generator
  if (mother.producedByGenerator()) {
    return true;
  }

  // To be sure that heavy flavor has not been produced in a secondary interaction
  // Loop back to the generated mother
  LOGF(debug, "M0 %d %d", mother.producedByGenerator(), mother.mother0());
  while (mother.mother0() != -1 && !mother.producedByGenerator()) {
    mother = mcParticles.iteratorAt(mother.mother0());
    LOGF(debug, "M+ %d %d", mother.producedByGenerator(), mother.mother0());
    mpdg = std::abs(mother.pdgCode());
    mfl = int(mpdg / std::pow(10, int(std::log10(mpdg))));
  }

  if (mfl < 4) {
    return false;
  } else {
    return true;
  }
}

}; // namespace MC

#endif
