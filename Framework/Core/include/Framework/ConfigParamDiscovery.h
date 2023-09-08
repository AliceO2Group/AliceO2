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
struct ConfigParamDiscovery {
  /// Given a ConfigParamRegistry, discovers all the extrac ConfigParam
  /// which require an additional lookup starting from the ones already
  /// present. E.g. you could use this to populate the registry with
  /// ConfigParams defined in a file.
  static void discover(ConfigParamRegistry& registry);
};

/// A plugin which is able to discover more option then the ones
/// provided on the command line.
struct ConfigDiscoveryPlugin {
  virtual void discover(ConfigParamRegistry& registry) = 0;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_CONFIGPARAMDISCOVERY_H_
