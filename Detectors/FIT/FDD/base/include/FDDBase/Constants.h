// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  constexpr UShort_t mNchannels = 16;
  constexpr UShort_t mNtriggers = 5;
  constexpr Float_t mIntTimeRes = 0.4;
  constexpr Float_t mPhotoCathodeEfficiency = 0.18;
  constexpr Float_t mChargePerADC = 0.6e-12;
  constexpr Float_t mPMTransitTime = 6.0;   // PM response time (corresponds to 1.9 ns rise time)
  constexpr Float_t mPMTransparency = 0.25; // Transparency of the first dynode of the PM
  constexpr Float_t mPMNbOfSecElec = 6.0;   // Number of secondary electrons emitted from first dynode (per ph.e.)
  
  constexpr Int_t mNTimeBinsPerBC = 256;    // number of samples per BC


//Detector TOF correction in ns
 constexpr Float_t mFDAdist = 1696.67;
 constexpr Float_t mFDCdist = 1954.4;
 constexpr Float_t mLayerWidth = 1.27;

constexpr float getTOFCorrection(int det)
{
  constexpr float TOFCorr[4] = {
    (mFDCdist+mLayerWidth) / o2::constants::physics::LightSpeedCm2NS,
    (mFDCdist-mLayerWidth) / o2::constants::physics::LightSpeedCm2NS,
    (mFDAdist-mLayerWidth) / o2::constants::physics::LightSpeedCm2NS,
    (mFDAdist+mLayerWidth) / o2::constants::physics::LightSpeedCm2NS};
  return TOFCorr[det];
}

} // namespace fdd
} // namespace o2

#endif
