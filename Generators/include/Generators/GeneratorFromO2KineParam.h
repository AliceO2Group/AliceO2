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

/// \author R+Preghenella - January 2021

#ifndef ALICEO2_EVENTGEN_GENERATORFROMO2KINEPARAM_H_
#define ALICEO2_EVENTGEN_GENERATORFROMO2KINEPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace eventgen
{

/**
 ** a parameter class/struct to keep the settings of
 ** the FromO2Kine event generator and
 ** allow the user to modify them 
 **/

struct GeneratorFromO2KineParam : public o2::conf::ConfigurableParamHelper<GeneratorFromO2KineParam> {
  bool skipNonTrackable = true;
  bool continueMode = false;
  bool roundRobin = false;   // read events with period boundary conditions
  std::string fileName = ""; // filename to read from - takes precedence over SimConfig if given
  O2ParamDef(GeneratorFromO2KineParam, "GeneratorFromO2Kine");
};

} // end namespace eventgen
} // end namespace o2

#endif // ALICEO2_EVENTGEN_GENERATORFROMO2KINEPARAM_H_
