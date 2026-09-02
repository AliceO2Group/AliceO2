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

/// \file MonopoleParticles.h
/// \brief Identification of the BSM magnetic monopoles across the simulation
/// @author M. Giacalone - September 2026
///
/// The monopole PDG codes are defined in O2MCApplication::AddParticles() and in
/// O2DatabasePDG. They are collected here for better syncing inside the simulation code
///
/// To do: add dyons and other monopoles configurations when needed

#ifndef ALICEO2_SIMULATION_MONOPOLEPARTICLES_H_
#define ALICEO2_SIMULATION_MONOPOLEPARTICLES_H_

namespace o2::sim
{

/// monopole carrying equal electric and magnetic charge
constexpr int MonopolePdgSymm = 4110000;
/// monopole carrying opposite electric and magnetic charge
constexpr int MonopolePdgAsymm = 4120000;

/// true for the monopole and antimonopole of both species
constexpr bool isMonopole(int pdg) noexcept
{
  const int abspdg = pdg < 0 ? -pdg : pdg;
  return abspdg == MonopolePdgSymm || abspdg == MonopolePdgAsymm;
}

} // namespace o2::sim

#endif
