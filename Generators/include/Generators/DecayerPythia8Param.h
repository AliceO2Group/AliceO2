// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - June 2020

#ifndef ALICEO2_EVENTGEN_DECAYERPYTHIA8PARAM_H_
#define ALICEO2_EVENTGEN_DECAYERPYTHIA8PARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include <string>

namespace o2
{
namespace eventgen
{

/**
 ** a parameter class/struct to keep the settings of
 ** the Pythia8 decayer and
 ** allow the user to modify them 
 **/

struct DecayerPythia8Param : public o2::conf::ConfigurableParamHelper<DecayerPythia8Param> {
  std::string config[8] = {"${O2_ROOT}/share/Generators/pythia8/decays/base.cfg", "", "", "", "", "", "", ""};
  bool verbose = false;
  bool showChanged = false;
  O2ParamDef(DecayerPythia8Param, "DecayerPythia8");
};

} // end namespace eventgen
} // end namespace o2

#endif // ALICEO2_EVENTGEN_DECAYERPYTHIA8PARAM_H_
