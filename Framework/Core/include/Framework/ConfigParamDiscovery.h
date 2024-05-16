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

#ifndef O2_FRAMEWORK_CONFIGPARAMDISCOVERY_H_
#define O2_FRAMEWORK_CONFIGPARAMDISCOVERY_H_

#include "Framework/ConfigParamRegistry.h"

namespace o2::framework
{
// A plugin which can be used to inject extra configuration
// options.
struct ConfigParamDiscovery {
  static std::vector<ConfigParamSpec> discover(ConfigParamRegistry&, int, char**);
};

struct ConfigDiscovery {
  std::function<void()> init;
  /// A function which uses the arguments available so far to discover more
  /// @return the extra ConfigParamSpecs it derives.
  std::function<std::vector<ConfigParamSpec>(ConfigParamRegistry&, int, char**)> discover;
};

struct ConfigDiscoveryPlugin {
  virtual ConfigDiscovery* create() = 0;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_CONFIGPARAMDISCOVERY_H_
