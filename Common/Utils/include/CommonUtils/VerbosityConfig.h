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

/// \author ruben.shahoyan@cern.ch
/// \brief params to steer common verbosity behaviour

#ifndef COMMON_VERBOSITY_CONFIG_H_
#define COMMON_VERBOSITY_CONFIG_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "Framework/Logger.h"

namespace o2
{
namespace conf
{
struct VerbosityConfig : public o2::conf::ConfigurableParamHelper<VerbosityConfig> {
  size_t maxWarnDeadBeef = 5; // max amount of consecutive DeadBeef TF messages to report
  size_t maxWarnRawParser = 5; // max amount of consecutive messages on RawParser creation failure
  fair::Severity rawParserSeverity = fair::Severity::alarm;

  O2ParamDef(VerbosityConfig, "VerbosityConfig");
};
} // namespace conf
} // namespace o2

#endif
