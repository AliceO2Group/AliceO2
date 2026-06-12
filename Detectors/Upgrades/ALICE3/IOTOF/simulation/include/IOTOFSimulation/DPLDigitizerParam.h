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

#ifndef ALICEO2_TF3DPLDIGITIZERPARAM_H_
#define ALICEO2_TF3DPLDIGITIZERPARAM_H_

#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include <string_view>

namespace o2
{
namespace iotof
{
struct DPLDigitizerParam : public o2::conf::ConfigurableParamHelper<DPLDigitizerParam> {

  bool continuous = true;                   ///< flag for continuous simulation
  float noisePerPixel = DEFNoisePerPixel(); ///< ALPIDE Noise per channel

  double timeOffset = 0.;                 ///< time offset (in seconds!) to calculate ROFrame from hit time
  int chargeThreshold = 75;               ///< charge threshold in Nelectrons
  int minChargeToAccount = 7;             ///< minimum charge contribution to account
  int nSimSteps = 475;                    ///< number of steps in response simulation
  float energyToNElectrons = 1. / 3.6e-9; // conversion of eloss to Nelectrons

  std::string noiseFilePath{}; ///< optional noise masks file path. FIXME to be removed once switch to CCDBFetcher

  // boilerplate stuff + make principal key
  O2ParamDef(DPLDigitizerParam, "TF3DigitizerParam");

 private:
  static constexpr float DEFNoisePerPixel()
  {
    return 1e-8; // ITS/MFT values here!!
  }
};

} // namespace iotof
} // namespace o2

#endif
