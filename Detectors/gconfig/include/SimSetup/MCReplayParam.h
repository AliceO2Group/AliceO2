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

/// \author A+Morsch

#ifndef ALICEO2_EVENTGEN_MCREPLAYPARAM_H_
#define ALICEO2_EVENTGEN_MCREPLAYPARAM_H_

#include <string>
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
/**
 ** A parameter class/struct holding values for
 ** MCReplay Transport Code
 **/
struct MCReplayParam : public o2::conf::ConfigurableParamHelper<MCReplayParam> {
  std::string stepTreename = "StepLoggerTree";          // name of the TTree containing the actual steps
  std::string stepFilename = "MCStepLoggerOutput.root"; // filename where to find the stepTreename
  float energyCut = -1.;                                // minimum energy required for a step to continue tracking
  std::string cutFile = "";
  O2ParamDef(MCReplayParam, "MCReplayParam");
};
} // end namespace o2

#endif // ALICEO2_EVENTGEN_MCREPLAYPARAM_H_
