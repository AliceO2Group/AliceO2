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
#ifndef FRAMEWORK_CONFIG_CONTEXT_H
#define FRAMEWORK_CONFIG_CONTEXT_H

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ServiceRegistry.h"

namespace o2
{
namespace framework
{

/// This is the context class for information which are available at
/// (re)configuration of the topology. It's automatically filled by the data
/// processing layer and passed to the user `defineDataProcessing` function.
class ConfigContext
{
 public:
  ConfigContext(ConfigParamRegistry& options, int argc, char** argv) : mOptions{options}, mArgc{argc}, mArgv{argv} {}

  ConfigParamRegistry& options() const { return mOptions; }

  bool helpOnCommandLine() const;

  int argc() const { return mArgc; }
  char* const* const argv() const { return mArgv; }

 private:
  ConfigParamRegistry& mOptions;
  // additionaly keep information about the original command line
  int mArgc = 0;
  char** mArgv = nullptr;
};

} // namespace framework
} // namespace o2

#endif
