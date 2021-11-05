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

namespace o2::mcutils
{

// taken from AliRoot and adapted to use of o2::MCTrack class
bool MCTrackNavigator::isPhysicalPrimary(o2::MCTrack const& p, std::vector<o2::MCTrack> const& pcontainer)
{
  // Test if a particle is a physical primary according to the following definition:
  // Particles produced in the collision including products of strong and
  // electromagnetic decay and excluding feed-down from weak decays of strange
  // particles.
  //

  const int ist = p.getStatusCode(); // the generator status code
  const int pdg = std::abs(p.GetPdgCode());
  //
  // Initial state particle
  // Solution for K0L decayed by Pythia6
  // ->
  // ist > 1 --> essentially means unstable
  if ((ist > 1) && (pdg != 130) && p.isPrimary()) {
    return false;
  }
  if ((ist > 1) && p.isSecondary()) {
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
    const int ipm = p.getMotherTrackId();
    if (ipm > -1) {
      const auto& ppm = pcontainer[ipm];
      if (std::abs(ppm.GetPdgCode()) == 130) {
        return false;
      }
    }
    // <-
    // check for direct photon in parton shower
    // ->
    const int ipd = p.getFirstDaughterTrackId();
    if (pdg == 22 && ipd > -1) {
      const auto& ppd = pcontainer[ipd];
      if (ppd.GetPdgCode() == 22) {
        return false;
      }
    }
    // <-
    return true;
  } else {
    //
    // Particle produced during transport
    //

    int imo = p.getMotherTrackId();
    auto pm = &(pcontainer[imo]);
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
      imo = pm->getMotherTrackId();
      pm = &(pcontainer[imo]);
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

} // namespace o2::mcutils
