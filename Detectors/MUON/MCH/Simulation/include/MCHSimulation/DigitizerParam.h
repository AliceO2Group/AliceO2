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

#ifndef O2_MCH_SIMULATION_DIGITIZER_PARAM_H_
#define O2_MCH_SIMULATION_DIGITIZER_PARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::mch
{

struct DigitizerParam : public o2::conf::ConfigurableParamHelper<DigitizerParam> {

  bool continuous = true; ///< whether we assume continuous mode or not

  int seed = 0; ///< seed for random number generators used for time, noise and threshold (0 means no seed given)

  float timeSigma = 2.f; ///< time dispersion added to digit times (in bc unit)

  float noiseSigma = 0.5f; ///< dispersion of noise added to physical signal per ADC sample (in ADC counts)

  float noiseOnlyProba = 1.e-7f; ///< probability of noise-only signal (per pad per ROF=4BC)
  float noiseOnlyMean = 23.f;    ///< mean value of noise-only signal (in ADC counts)
  float noiseOnlySigma = 3.f;    ///< dispersion of noise-only signal (in ADC counts)

  bool onlyNoise = false; ///< for debug only: disable treatment of physical signals (i.e. keep only noise)

  float minChargeMean = 22.2f; ///< mean value of lower charge threshold for a signal to be digitized (in ADC counts)
  float minChargeSigma = 2.8f; ///< dispersion of lower charge threshold for a signal to be digitized (in ADC counts)

  bool handlePileup = true; ///< merge digits in overlapping readout windows (defined by the number of samples + 2)

  O2ParamDef(DigitizerParam, "MCHDigitizer")
};

} // namespace o2::mch

#endif
