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
#ifndef O2_FRAMEWORK_CONFIG_CONTEXT_H_
#define O2_FRAMEWORK_CONFIG_CONTEXT_H_

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ServiceRegistryRef.h"

namespace o2::framework
{

/// This is the context class for information which are available at
/// (re)configuration of the topology. It's automatically filled by the data
/// processing layer and passed to the user `defineDataProcessing` function.
class ConfigContext
{
 public:
  ConfigContext(ConfigParamRegistry& options, ServiceRegistryRef services, int argc, char** argv);

  [[nodiscard]] ConfigParamRegistry& options() const { return mOptions; }
  [[nodiscard]] ServiceRegistryRef services() const { return mServices; }

  [[nodiscard]] bool helpOnCommandLine() const;
  /// Whether this process may run @a processor. A device only runs its own;
  /// it gets the specs of the others from the driver, so they need not be built.
  [[nodiscard]] bool mayRun(std::string_view processor) const;

  [[nodiscard]] int argc() const { return mArgc; }
  [[nodiscard]] char* const* argv() const { return mArgv; }

 private:
  ConfigParamRegistry& mOptions;

  ServiceRegistryRef mServices;
  // additionaly keep information about the original command line
  int mArgc = 0;
  char** mArgv = nullptr;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_CONFIG_CONTEXT_H_
