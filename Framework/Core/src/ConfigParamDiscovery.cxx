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
#include <fmt/format.h>

#include <algorithm>
#include <charconv>
#include <ranges>
#include <stdexcept>

O2_DECLARE_DYNAMIC_LOG(config_parameter_discovery);

namespace o2::framework
{

std::vector<ConfigParamSpec> ConfigParamDiscovery::discover(ConfigParamRegistry& registry, int argc, char** argv)
{
  std::vector<char const*> capabilitiesSpecs = {
    "O2Framework:DiscoverAODOptionsInCommandLineCapability",
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
    registry.loadExtra(extras);
  }
  applyOriginMetadata(registry);
  return result;
}

LabeledArray<int> ConfigParamDiscovery::parseOriginLevelMapping(std::string_view mapping)
{
  std::vector<std::string> origins;
  std::vector<int> levels;
  for (auto pairRange : mapping | std::views::split(',')) {
    std::string_view pair{pairRange.begin(), pairRange.end()};
    auto colonPos = pair.find(':');
    if (colonPos == std::string_view::npos) {
      throw std::runtime_error(fmt::format("Badly formatted aod-origin-level-mapping entry: \"{}\"", pair));
    }
    std::string origin{pair.substr(0, colonPos)};
    if (origin.empty() || origin.size() > 4) {
      throw std::runtime_error(fmt::format("Origin in aod-origin-level-mapping entry must have 1 to 4 characters: \"{}\"", pair));
    }
    if (std::ranges::find(origins, origin) != origins.end()) {
      throw std::runtime_error(fmt::format("Duplicate origin in aod-origin-level-mapping: \"{}\"", origin));
    }
    auto levelStr = pair.substr(colonPos + 1);
    int level{};
    auto [ptr, ec] = std::from_chars(levelStr.data(), levelStr.data() + levelStr.size(), level);
    if (levelStr.empty() || ec != std::errc{} || ptr != levelStr.data() + levelStr.size() || level < 0) {
      throw std::runtime_error(fmt::format("Unable to parse level in aod-origin-level-mapping entry: \"{}\"", pair));
    }
    origins.push_back(std::move(origin));
    levels.push_back(level);
  }
  if (origins.empty()) {
    throw std::runtime_error("Empty aod-origin-level-mapping");
  }
  return LabeledArray<int>{levels.data(), static_cast<uint32_t>(levels.size()), 1, origins, {"level"}};
}

void ConfigParamDiscovery::applyOriginMetadata(ConfigParamRegistry& registry)
{
  if (!registry.hasOption("aod-metadata-origins")) {
    return;
  }
  std::string origin = "AOD";
  if (registry.hasOption("aod-origin-replace")) {
    auto replace = registry.get<std::string>("aod-origin-replace");
    if (!replace.empty()) {
      origin = replace;
    }
  }
  auto metadata = registry.get<LabeledArray<std::string>>("aod-metadata-origins");
  auto* row = metadata.getRow(origin.c_str());
  if (row == nullptr) {
    return;
  }
  O2_SIGNPOST_ID_GENERATE(sid, config_parameter_discovery);
  auto const& keys = metadata.getLabelsCols();
  auto disable = metadata.colmap.find("disable");
  if (disable != metadata.colmap.end() && row[disable->second] == "1") {
    O2_SIGNPOST_EVENT_EMIT(config_parameter_discovery, sid, "origin metadata", "No metadata for origin %{public}s", origin.c_str());
    for (auto const& key : keys) {
      registry.erase(("aod-metadata-" + key).c_str());
    }
    registry.erase("aod-metadata-tables");
    registry.override("aod-metadata-disable", std::string{"1"});
    return;
  }
  O2_SIGNPOST_EVENT_EMIT(config_parameter_discovery, sid, "origin metadata", "Using metadata for origin %{public}s", origin.c_str());
  for (auto j = 0U; j < metadata.cols(); ++j) {
    if (keys[j] == "disable") {
      continue;
    }
    auto key = "aod-metadata-" + keys[j];
    if (row[j].empty()) {
      registry.erase(key.c_str());
    } else {
      registry.override(key.c_str(), row[j]);
    }
  }
  registry.erase("aod-metadata-disable");
  auto tablesKey = "aod-metadata-origin-" + origin + "-tables";
  if (registry.hasOption(tablesKey.c_str())) {
    registry.override("aod-metadata-tables", registry.get<std::vector<std::string>>(tablesKey.c_str()));
  } else {
    registry.erase("aod-metadata-tables");
  }
}

} // namespace o2::framework
