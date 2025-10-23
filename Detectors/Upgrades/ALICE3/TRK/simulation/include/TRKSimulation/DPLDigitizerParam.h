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

#ifndef ALICEO2_TRKDPLDIGITIZERPARAM_H_
#define ALICEO2_TRKDPLDIGITIZERPARAM_H_

#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include <string_view>

namespace o2
{
namespace trk
{
template <int N>
struct DPLDigitizerParam : public o2::conf::ConfigurableParamHelper<DPLDigitizerParam<N>> {
  static_assert(N == o2::detectors::DetID::TRK || N == o2::detectors::DetID::FT3, "only DetID::TRK or DetID::FT3 are allowed");

  static constexpr std::string_view getParamName()
  {
    return N == o2::detectors::DetID::TRK ? ParamName[0] : ParamName[1];
  }

  bool continuous = true;                   ///< flag for continuous simulation
  float noisePerPixel = DEFNoisePerPixel(); ///< ALPIDE Noise per channel
  float strobeFlatTop = 7500.;              ///< strobe shape flat top
  float strobeMaxRiseTime = 1100.;          ///< strobe max rise time
  float strobeQRiseTime0 = 450.;            ///< q @ which strobe rise time is 0

  double timeOffset = 0.;                 ///< time offset (in seconds!) to calculate ROFrame from hit time
  int chargeThreshold = 1;                ///< charge threshold in Nelectrons
  int minChargeToAccount = 1;             ///< minimum charge contribution to account
  int nSimSteps = 25;                     ///< number of steps in response simulation
  float energyToNElectrons = 1. / 3.6e-9; // conversion of eloss to Nelectrons

  float Vbb = 0.0;   ///< back bias absolute value for MFT (in Volt)
  float IBVbb = 0.0; ///< back bias absolute value for ITS Inner Barrel (in Volt)
  float OBVbb = 0.0; ///< back bias absolute value for ITS Outter Barrel (in Volt)

  std::string noiseFilePath{}; ///< optional noise masks file path. FIXME to be removed once switch to CCDBFetcher

  // boilerplate stuff + make principal key
  O2ParamDef(DPLDigitizerParam, getParamName().data());

 private:
  static constexpr float DEFNoisePerPixel()
  {
    return N == o2::detectors::DetID::TRK ? 1e-8 : 1e-8; // ITS/MFT values here!!
  }

  static constexpr std::string_view ParamName[2] = {"TRKDigitizerParam", "FT3DigitizerParam"};
};

template <int N>
DPLDigitizerParam<N> DPLDigitizerParam<N>::sInstance;

} // namespace trk
} // namespace o2

#endif
