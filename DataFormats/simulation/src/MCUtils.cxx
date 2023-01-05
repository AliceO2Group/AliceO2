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

//
// Created by Sandro Wenzel on 03.11.21.
//

#include <SimulationDataFormat/MCUtils.h>
#include <SimulationDataFormat/MCGenStatus.h>

namespace o2::mcutils
{

o2::MCTrack const* MCTrackNavigator::getMother(const o2::MCTrack& p, const std::vector<o2::MCTrack>& pcontainer)
{
  const auto mid = p.getMotherTrackId();
  if (mid < 0 or mid > pcontainer.size()) {
    return nullptr;
  }
  return &(pcontainer[mid]);
}

o2::MCTrack const* MCTrackNavigator::getDaughter(const o2::MCTrack& p, const std::vector<o2::MCTrack>& pcontainer)
{
  const auto did = p.getFirstDaughterTrackId();
  if (did < 0 or did > pcontainer.size()) {
    return nullptr;
  }
  return &(pcontainer[did]);
}

o2::MCTrack const* MCTrackNavigator::getDaughter0(const o2::MCTrack& p, const std::vector<o2::MCTrack>& pcontainer)
{
  const auto did = p.getFirstDaughterTrackId();
  if (did < 0 or did > pcontainer.size()) {
    return nullptr;
  }
  return &(pcontainer[did]);
}

o2::MCTrack const* MCTrackNavigator::getDaughter1(const o2::MCTrack& p, const std::vector<o2::MCTrack>& pcontainer)
{
  const auto did = p.getLastDaughterTrackId();
  if (did < 0 or did > pcontainer.size()) {
    return nullptr;
  }
  return &(pcontainer[did]);
}

o2::MCTrack const& MCTrackNavigator::getFirstPrimary(const o2::MCTrack& p, const std::vector<o2::MCTrack>& pcontainer)
{
  if (p.isPrimary()) {
    return p;
  }
  o2::MCTrack const* ptr = &p;
  while (true) {
    ptr = getMother(*ptr, pcontainer);
    if (ptr->isPrimary()) {
      return *ptr;
    }
  }
}

// taken from AliRoot and adapted to use of o2::MCTrack class
bool MCTrackNavigator::isPhysicalPrimary(o2::MCTrack const& p, std::vector<o2::MCTrack> const& pcontainer)
{
  // Test if a particle is a physical primary according to the following definition:
  // Particles produced in the collision including products of strong and
  // electromagnetic decay and excluding feed-down from weak decays of strange
  // particles.
  //

  const int hepmcStatusCode = o2::mcgenstatus::getHepMCStatusCode(p.getStatusCode()); // the HepMC status code
  const int pdg = std::abs(p.GetPdgCode());
  //
  // Initial state particle
  // Solution for K0L decayed by Pythia6
  // ->
  // ist > 1 --> essentially means unstable
  if ((hepmcStatusCode > 1) && (pdg != 130) && p.isPrimary()) {
    return false;
  }
  if ((hepmcStatusCode > 1) && p.isSecondary()) {
    return false;
  }
  // <-

  if (!isStable(pdg)) {
    return false;
  }

  if (p.isPrimary()) {
    //
    // Particle produced by generator
    // Solution for K0L decayed by Pythia6
    // ->
    const auto ipm = getMother(p, pcontainer);
    if (ipm != nullptr) {
      if (std::abs(ipm->GetPdgCode()) == 130) {
        return false;
      }
    }
    // <-
    // check for direct photon in parton shower
    // ->
    if (pdg == 22) {
      const auto ipd = getDaughter(p, pcontainer);
      if (ipd && ipd->GetPdgCode() == 22) {
        return false;
      }
    }
    // <-
    return true;
  } else {
    //
    // Particle produced during transport
    //

    auto pm = getMother(p, pcontainer);
    int mpdg = std::abs(pm->GetPdgCode());
    // Check for Sigma0
    if ((mpdg == 3212) && pm->isPrimary()) {
      return true;
    }

    //
    // Check if it comes from a pi0 decay
    //
    if ((mpdg == kPi0) && pm->isPrimary()) {
      return true;
    }

    // Check if this is a heavy flavor decay product
    int mfl = int(mpdg / std::pow(10, int(std::log10(mpdg))));
    //
    // Light hadron
    if (mfl < 4) {
      return false;
    }

    //
    // Heavy flavor hadron produced by generator
    if (pm->isPrimary()) {
      return true;
    }

    // To be sure that heavy flavor has not been produced in a secondary interaction
    // Loop back to the generated mother
    while (!pm->isPrimary()) {
      pm = getMother(*pm, pcontainer);
    }
    mpdg = std::abs(pm->GetPdgCode());
    mfl = int(mpdg / std::pow(10, int(std::log10(mpdg))));

    if (mfl < 4) {
      return false;
    } else {
      return true;
    }
  } // end else branch produced by generator
}

bool MCTrackNavigator::isFromPrimaryDecayChain(o2::MCTrack const& p, std::vector<o2::MCTrack> const& pcontainer)
{
  /** check if the particle is from the
      decay chain of a primary particle **/

  /** check if from decay **/
  if (p.getProcess() != kPDecay) {
    return false;
  }
  /** check if mother is primary **/
  auto mother = getMother(p, pcontainer);
  if (!mother || mother->isPrimary()) {
    return true;
  }
  /** else check if mother is from primary decay **/
  return isFromPrimaryDecayChain(*mother, pcontainer);
}

bool MCTrackNavigator::isKeepPhysics(o2::MCTrack const& p, std::vector<o2::MCTrack> const& pcontainer)
{
  auto isFromPrimaryPairProduction = [&pcontainer](const MCTrack& part) {
    /** check if the particle is from
        pair production from a particle
        belonging to the primary decay chain **/

    /** check if from pair production **/
    if (part.getProcess() != kPPair) {
      return false;
    }
    auto mother = getMother(part, pcontainer);
    if (!mother || mother->isPrimary()) {
      return true;
    }
    /** else check if mother is from primary decay **/
    return isFromPrimaryDecayChain(*mother, pcontainer);
  };
  //
  return p.isPrimary() || isFromPrimaryPairProduction(p) || isFromPrimaryDecayChain(p, pcontainer);
}

} // namespace o2::mcutils
