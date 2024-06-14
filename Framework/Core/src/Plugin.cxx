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
#include "Framework/Plugins.h"
#include "Framework/ConfigParamDiscovery.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "Framework/Capability.h"
#include "Framework/Signpost.h"
#include "Framework/VariantJSONHelpers.h"
#include <string_view>

O2_DECLARE_DYNAMIC_LOG(capabilities);
namespace o2::framework
{
auto lookForAodFile = [](ConfigParamRegistry& registry, int argc, char** argv) -> bool {
  O2_SIGNPOST_ID_GENERATE(sid, capabilities);
  if (registry.hasOption("aod-file") && registry.isSet("aod-file")) {
    for (size_t i = 0; i < argc; i++) {
      std::string_view arg = argv[i];
      if (arg.starts_with("--aod-metadata-")) {
        return false;
      }
    }
    O2_SIGNPOST_EVENT_EMIT(capabilities, sid, "DiscoverMetadataInAODCapability", "Metadata not found in arguments. Checking in AOD file.");
    return true;
  }
  return false;
};

auto lookForCommandLineOptions = [](ConfigParamRegistry& registry, int argc, char** argv) -> bool {
  O2_SIGNPOST_ID_GENERATE(sid, capabilities);
  for (size_t i = 0; i < argc; i++) {
    std::string_view arg = argv[i];
    if (arg.starts_with("--aod-metadata-")) {
      O2_SIGNPOST_EVENT_EMIT(capabilities, sid, "DiscoverMetadataInCommandLineCapability", "Metadata found in arguments. Populating from them.");
      return true;
    }
  }
  return false;
};

struct DiscoverMetadataInAODCapability : o2::framework::CapabilityPlugin {
  Capability* create() override
  {
    return new Capability{
      .name = "DiscoverMetadataInAODCapability",
      .checkIfNeeded = lookForAodFile,
      .requiredPlugin = "O2FrameworkAnalysisSupport:DiscoverMetadataInAOD"};
  }
};

// Trigger discovery of metadata from command line, if needed.
struct DiscoverMetadataInCommandLineCapability : o2::framework::CapabilityPlugin {
  Capability* create() override
  {
    return new Capability{
      .name = "DiscoverMetadataInCommandLineCapability",
      .checkIfNeeded = lookForCommandLineOptions,
      .requiredPlugin = "O2Framework:DiscoverMetadataInCommandLine"};
  }
};

struct DiscoverMetadataInCommandLine : o2::framework::ConfigDiscoveryPlugin {
  ConfigDiscovery* create() override
  {
    return new ConfigDiscovery{
      .init = []() {},
      .discover = [](ConfigParamRegistry& registry, int argc, char** argv) -> std::vector<ConfigParamSpec> {
        O2_SIGNPOST_ID_GENERATE(sid, capabilities);
        O2_SIGNPOST_EVENT_EMIT(capabilities, sid, "DiscoverMetadataInCommandLine",
                               "Discovering metadata for analysis from well known environment variables.");
        std::vector<ConfigParamSpec> results;
        for (size_t i = 0; i < argc; i++) {
          std::string_view arg = argv[i];
          if (!arg.starts_with("--aod-metadata")) {
            continue;
          }
          std::string key = arg.data() + 2;
          std::string value = argv[i + 1];
          O2_SIGNPOST_EVENT_EMIT(capabilities, sid, "DiscoverMetadataInCommandLine",
                                 "Found %{public}s with value %{public}s.", key.c_str(), value.c_str());
          if (key == "aod-metadata-tables") {
            std::stringstream is(value);
            auto arrayValue = VariantJSONHelpers::read<VariantType::ArrayString>(is);
            results.push_back(ConfigParamSpec{key, VariantType::ArrayString, arrayValue, {"Metadata in command line"}});
          } else {
            results.push_back(ConfigParamSpec{key, VariantType::String, value, {"Metadata in command line"}});
          }
        }
        return results;
      }};
  }
};
DEFINE_DPL_PLUGINS_BEGIN
DEFINE_DPL_PLUGIN_INSTANCE(DiscoverMetadataInAODCapability, Capability);
DEFINE_DPL_PLUGIN_INSTANCE(DiscoverMetadataInCommandLineCapability, Capability);
DEFINE_DPL_PLUGIN_INSTANCE(DiscoverMetadataInCommandLine, ConfigDiscovery);
DEFINE_DPL_PLUGINS_END
} // namespace o2::framework
