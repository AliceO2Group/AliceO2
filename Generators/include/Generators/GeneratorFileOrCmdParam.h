// Copyright 2023-2099 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @author Christian Holm Christensen <cholm@nbi.dk>

#ifndef ALICEO2_EVENTGEN_GENERATORFILEORCMDPARAM_H_
#define ALICEO2_EVENTGEN_GENERATORFILEORCMDPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include <string>

namespace o2
{
namespace eventgen
{

/**
 ** a parameter class/struct to keep the settings of
 ** the Fileorcmd event generator and
 ** allow the user to modify them
 **/
struct GeneratorFileOrCmdParam : public o2::conf::ConfigurableParamHelper<GeneratorFileOrCmdParam> {
  std::string fileNames = "";
  std::string cmd = ""; // Program command line to spawn
  std::string outputSwitch = ">";
  std::string seedSwitch = "-s";
  std::string bMaxSwitch = "-b";
  std::string nEventsSwitch = "-n";
  std::string backgroundSwitch = "&";
  O2ParamDef(GeneratorFileOrCmdParam, "GeneratorFileOrCmd");
};

} // end namespace eventgen
} // end namespace o2

#endif // ALICEO2_EVENTGEN_GENERATORFILEORCMDPARAM_H_
