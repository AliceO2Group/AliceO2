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

namespace o2
{
namespace constants
{
namespace physics
{
// particles masses
constexpr float MassPhoton = 0.0;
constexpr float MassElectron = 0.000511;
constexpr float MassMuon = 0.105658;
constexpr float MassPionCharged = 0.139570;
constexpr float MassPionNeutral = 0.134976;
constexpr float MassKaonCharged = 0.493677;
constexpr float MassKaonNeutral = 0.497648;
constexpr float MassProton = 0.938272;
constexpr float MassLambda = 1.115683;
constexpr float MassDeuteron = 1.8756129;
constexpr float MassTriton = 2.8089211;
constexpr float MassHelium3 = 2.8083916;
constexpr float MassAlpha = 3.7273794;
constexpr float MassHyperTriton = 2.992;
constexpr float MassHyperhydrog4 = 3.931;
constexpr float MassXiMinus = 1.32171;
constexpr float MassOmegaMinus = 1.67245;

constexpr float LightSpeedCm2S = 299792458.e2;           // C in cm/s
constexpr float LightSpeedCm2NS = LightSpeedCm2S * 1e-9; // C in cm/ns
} // namespace physics
} // namespace constants
} // namespace o2

#endif
