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

#include "Framework/ConfigParamDiscovery.h"
#include "Framework/PluginManager.h"
#include "Framework/Signpost.h"
#include "Framework/Capability.h"
#include <TFile.h>
#include <TMap.h>
#include <TObjString.h>

O2_DECLARE_DYNAMIC_LOG(config_parameter_discovery);

namespace o2::framework
{

std::vector<ConfigParamSpec> ConfigParamDiscovery::discover(ConfigParamRegistry& registry, int argc, char** argv)
{
  std::vector<char const*> capabilitiesSpecs = {
    "O2Framework:DiscoverMetadataInAODCapability",
    "O2Framework:DiscoverMetadataInCommandLineCapability",
  };

  // Load all the requested plugins and discover what we can do.
  O2_SIGNPOST_ID_GENERATE(sid, config_parameter_discovery);
  std::vector<LoadablePlugin> plugins;
  for (auto spec : capabilitiesSpecs) {
    O2_SIGNPOST_EVENT_EMIT(config_parameter_discovery, sid, "capability", "Parsing Capability plugin: %{public}s", spec);
    auto morePlugins = PluginManager::parsePluginSpecString(spec);
    for (auto& extra : morePlugins) {
      plugins.push_back(extra);
    }
  }

  std::vector<Capability> availableCapabilities;
  std::vector<char const*> configDiscoverySpec = {};
  PluginManager::loadFromPlugin<Capability, CapabilityPlugin>(plugins, availableCapabilities);
  for (auto& capability : availableCapabilities) {
    O2_SIGNPOST_EVENT_EMIT(config_parameter_discovery, sid, "capability", "- Using Capability of : %{public}s",
                           capability.name.c_str());

    if (capability.checkIfNeeded(registry, argc, argv)) {
      O2_SIGNPOST_EVENT_EMIT(config_parameter_discovery, sid, "capability", "- Required: %{}s",
                             capability.requiredPlugin);
      configDiscoverySpec.push_back(capability.requiredPlugin);
    } else {
      O2_SIGNPOST_EVENT_EMIT(config_parameter_discovery, sid, "capability", "- No requirements added");
    }
  }

  std::vector<LoadablePlugin> configDiscoveryPlugins;
  for (auto spec : configDiscoverySpec) {
    O2_SIGNPOST_EVENT_EMIT(config_parameter_discovery, sid, "discovery", "Parsing ConfigDiscovery plugin: %{public}s", spec);
    auto morePlugins = PluginManager::parsePluginSpecString(spec);
    for (auto& extra : morePlugins) {
      configDiscoveryPlugins.push_back(extra);
    }
  }

  std::vector<ConfigDiscovery> availableConfigDiscovery;
  PluginManager::loadFromPlugin<ConfigDiscovery, ConfigDiscoveryPlugin>(configDiscoveryPlugins, availableConfigDiscovery);
  std::vector<ConfigParamSpec> result;
  for (auto& configDiscovery : availableConfigDiscovery) {
    O2_SIGNPOST_EVENT_EMIT(config_parameter_discovery, sid, "discovery", "Parsing extra configuration as needed");
    auto extras = configDiscovery.discover(registry, argc, argv);
    for (auto& extra : extras) {
      result.push_back(extra);
    }
  }
  return result;
}

} // namespace o2::framework
