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

  bool continuous = true;   // whether we assume continuous mode or not
  float noiseProba = 1E-6;  // noise proba (per pad per ROF=100ns)
  float noiseMean = 42;     // mean noise (in ADC counts). default value assumes 3 ADC count per sample, 14 samples for a signal.
  float noiseSigma = 11.22; // noise sigma (in ADC counts) = sqrt(14)*sigma_for_one_sample
  float timeSpread = 150;   // time spread added to digit times (in nanoseconds)
  int seed = 0;             // seed for random number generator(s) used for time and noise (0 means no seed given)
  bool onlyNoise = false;   // for debug only : disable treatment of signal (i.e. keep only noise)

  O2ParamDef(DigitizerParam, "MCHDigitizerParam")
};

} // namespace o2::mch

#endif
