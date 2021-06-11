// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  O2ParamDef(GeneratorFromO2KineParam, "GeneratorFromO2Kine");
};

} // end namespace eventgen
} // end namespace o2

#endif // ALICEO2_EVENTGEN_GENERATORFROMO2KINEPARAM_H_
