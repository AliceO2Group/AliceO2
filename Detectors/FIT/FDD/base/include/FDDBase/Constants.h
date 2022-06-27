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

#ifndef ALICEO2_FDD_CONSTANTS_H
#define ALICEO2_FDD_CONSTANTS_H

#include "CommonConstants/PhysicsConstants.h"
#include <cstdint>
#include <string_view>
#include <string>
#include <type_traits>

namespace o2
{
namespace fdd
{
constexpr short Nchannels = 16;
constexpr short Nmodules = 2;
constexpr short NChPerMod = 8;
constexpr short Ntriggers = 5;
constexpr float IntTimeRes = 0.4;
constexpr float PhotoCathodeEfficiency = 0.18;
constexpr float ChargePerADC = 0.6e-12;
constexpr float invTimePerTDC = 1. / 0.01302; // time conversion from ns to TDC channels
constexpr float timePerTDC = 0.01302;         // time conversion from TDC channels to ns
constexpr float PMTransitTime = 6.0;          // PM response time (corresponds to 1.9 ns rise time)
constexpr float PMTransparency = 0.25;        // Transparency of the first dynode of the PM
constexpr float PMNbOfSecElec = 6.0;          // Number of secondary electrons emitted from first dynode (per ph.e.)

constexpr int NTimeBinsPerBC = 256; // number of samples per BC

// Detector TOF correction in ns
constexpr float FDAdist = 1696.67;
constexpr float FDCdist = 1954.4;
constexpr float LayerWidth = 1.27;

constexpr float getTOFCorrection(int det)
{
  constexpr float TOFCorr[4] = {
    (FDCdist + LayerWidth) / o2::constants::physics::LightSpeedCm2NS,
    (FDCdist - LayerWidth) / o2::constants::physics::LightSpeedCm2NS,
    (FDAdist - LayerWidth) / o2::constants::physics::LightSpeedCm2NS,
    (FDAdist + LayerWidth) / o2::constants::physics::LightSpeedCm2NS};
  return TOFCorr[det];
}

} // namespace fdd
} // namespace o2

#endif
