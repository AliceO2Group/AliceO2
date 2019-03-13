// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_ITSMFTDPLDIGITIZERPARAM_H_
#define ALICEO2_ITSMFTDPLDIGITIZERPARAM_H_

#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"
#include <string_view>

namespace o2
{
namespace ITSMFT
{
template <int N>
struct DPLDigitizerParam : public o2::conf::ConfigurableParamHelper<DPLDigitizerParam<N>> {
  static_assert(N == 0 || N == 1, "only 0(ITS) or 1(MFT) are allowed");

  static constexpr std::string_view ParamName[2] = { "ITSDigitizerParam", "MFTDigitizerParam" };

  bool continuous = true;          ///< flag for continuous simulation
  float noisePerPixel = 1.e-7;     ///< ALPIDE Noise per chip
  float roFrameLength = 6000.;     ///< length of RO frame in ns
  float strobeDelay = 6000.;       ///< strobe start (in ns) wrt ROF start
  float strobeLength = 100.;       ///< length of the strobe in ns (sig. over threshold checked in this window only)
  float strobeFlatTop = 7500.;     ///< strobe shape flat top
  float strobeMaxRiseTime = 1100.; ///< strobe max rise time
  float strobeQRiseTime0 = 450.;   ///< q @ which strobe rise time is 0

  double timeOffset = 0.;                 ///< time offset (in seconds!) to calculate ROFrame from hit time
  int chargeThreshold = 150;              ///< charge threshold in Nelectrons
  int minChargeToAccount = 15;            ///< minimum charge contribution to account
  int nSimSteps = 7;                      ///< number of steps in response simulation
  float energyToNElectrons = 1. / 3.6e-9; // conversion of eloss to Nelectrons

  // boilerplate stuff + make principal key
  O2ParamDef(DPLDigitizerParam, ParamName[N].data());
};

template <int N>
DPLDigitizerParam<N> DPLDigitizerParam<N>::sInstance;

} // namespace ITSMFT
} // namespace o2

#endif
