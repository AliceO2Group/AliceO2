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

/// \file PhysicsConstants.h
/// \brief Header to collect physics constants
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_PHYSICSCONSTANTS_H_
#define ALICEO2_PHYSICSCONSTANTS_H_

#include "CommonConstants/PDG.h"

namespace o2::constants::physics
{
// particles masses
constexpr double MassPhoton = MassGamma;
constexpr double MassMuon = MassMuonMinus;
constexpr double MassPionCharged = MassPiPlus;
constexpr double MassPionNeutral = MassPi0;
constexpr double MassKaonCharged = MassKPlus;
constexpr double MassKaonNeutral = MassK0;
constexpr double MassLambda = MassLambda0;
constexpr double MassHyperhydrog4 = MassHyperHydrogen4;
constexpr double MassHyperhelium4 = MassHyperHelium4;

constexpr float LightSpeedCm2S = 299792458.e2;           // C in cm/s
constexpr float LightSpeedCm2NS = LightSpeedCm2S * 1e-9; // C in cm/ns
} // namespace o2::constants::physics

#endif
