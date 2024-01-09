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

/// \file MCUtils.h
/// \brief Utility functions for MC particles
/// \author S. Wenzel - November 2021

#ifndef O2_MCUTILS_H
#define O2_MCUTILS_H

#include <string>
#include <SimulationDataFormat/MCTrack.h>
#include <SimulationDataFormat/ParticleStatus.h>
#include "TPDGCode.h"
#include "TParticle.h"

namespace o2
{
namespace mcutils
{

/// A couple of functions to query on MC tracks ( that needs navigation within the global container
/// of available tracks. It is a class so as to make it available for interactive ROOT more easily.
class MCTrackNavigator
{
 public:
  /// Function to determine if a MC track/particle p is a primary according to physics criteria.
  /// Needs the particle as input, as well as the whole navigable container of particles
  /// (of which p needs to be a part itself). The container can be fetched via MCKinematicsReader.
  static bool isPhysicalPrimary(o2::MCTrack const& p, std::vector<o2::MCTrack> const& pcontainer);

  /// return true of particle is to be kept for physics analysis in any case
  /// (follows logic used in particle stack class
  static bool isKeepPhysics(o2::MCTrack const& p, std::vector<o2::MCTrack> const& pcontainer);
  static bool isFromPrimaryDecayChain(o2::MCTrack const& p, std::vector<o2::MCTrack> const& pcontainer);

  // some convenience functions for navigation

  /// Given an MCTrack p; Return the first primary mother particle in the upward parent chain (follow
  /// only first mothers). The first primary mother may have further parent (put by the generator).
  /// Return p itself if p is a primary.
  static o2::MCTrack const& getFirstPrimary(o2::MCTrack const& p, std::vector<o2::MCTrack> const& pcontainer);

  /// Given an MCTrack p; Return it's direct mother or nullptr. (we follow only first mother)
  static o2::MCTrack const* getMother(o2::MCTrack const& p, std::vector<o2::MCTrack> const& pcontainer);

  /// Given an MCTrack p; Return it's direct daughter or nullptr. (we follow only first daughter)
  static o2::MCTrack const* getDaughter(o2::MCTrack const& p, std::vector<o2::MCTrack> const& pcontainer);

  /// Given an MCTrack p; Return it's first direct daughter or nullptr.
  static o2::MCTrack const* getDaughter0(o2::MCTrack const& p, std::vector<o2::MCTrack> const& pcontainer);
  /// Given an MCTrack p; Return it's second direct daughter or nullptr.
  static o2::MCTrack const* getDaughter1(o2::MCTrack const& p, std::vector<o2::MCTrack> const& pcontainer);

  /// Given an MCTrack p; Fill the complete parent chain (ancestorchain) up to the most fundamental particle (follow only
  /// first mothers).
  // static void getParentChain(o2::MCTrack const& p, std::vector<o2::MCTrack> const& pcontainer, std::vector<o2::MCTrack> &ancestorchain);

  /// query if a track is a direct **or** indirect daughter of a parentID
  /// if trackid is same as parentid it returns true
  /// bool isTrackDaughterOf(int /*trackid*/, int /*parentid*/) const;
  /// we can think about offering some visitor like patterns executing a
  /// user hook on nodes

  ClassDefNV(MCTrackNavigator, 1);
};

class MCGenHelper
{
 public:
  // Helper function for users which takes over encoding of status code as well as (re-)setting correct transport bit.
  // Has to be in a class as a static methid. Just in a namespace it doesn't work to use this function in ROOT macros.
  static void encodeParticleStatusAndTracking(TParticle& particle, bool wanttracking = true);
  static void encodeParticleStatusAndTracking(TParticle& particle, int hepmcStatus, int genStatus, bool wanttracking = true);
  ClassDefNV(MCGenHelper, 1)
};

/// Determine if a particle (identified by pdg) is stable
inline bool isStable(int pdg)
{
  //
  // Decide whether particle (pdg) is stable
  //

  // All ions/nucleons are considered as stable
  // Nuclear code is 10LZZZAAAI
  if (pdg > 1000000000) {
    return true;
  }

  const int kNstable = 18;
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

  // this is linear search ---> a hash map or binary search should be more appropriate??
  for (int i = 0; i < kNstable; ++i) {
    if (pdg == std::abs(pdgStable[i])) {
      return true;
    }
  }
  return false;
}

} // namespace mcutils
} // namespace o2

#endif // O2_MCUTILS_H
