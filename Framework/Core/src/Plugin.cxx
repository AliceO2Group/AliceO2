// Copyright 2019-2024 CERN and copyright holders of ALICE O2.
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
#include "Framework/RootArrowFilesystem.h"
#include "Framework/Logger.h"
#include "Framework/Capability.h"
#include "Framework/Signpost.h"
#include "Framework/VariantJSONHelpers.h"
#include "Framework/PluginManager.h"
#include <TBufferFile.h>
#include <TDirectory.h>
#include <TDirectoryFile.h>
#include <TClass.h>
#include <arrow/dataset/file_base.h>
#include <arrow/filesystem/filesystem.h>
#include <cstddef>
#include <memory>
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

auto lookForCommandLineAODOptions = [](ConfigParamRegistry& registry, int argc, char** argv) -> bool {
  O2_SIGNPOST_ID_GENERATE(sid, capabilities);
  // If one of the options for aod-writer is specified, we should allow configuring compression.
  for (size_t i = 0; i < argc; i++) {
    std::string_view arg = argv[i];
    if (arg.starts_with("--aod-writer-")) {
      O2_SIGNPOST_EVENT_EMIT(capabilities, sid, "DiscoverAODOptionsInCommandLineCapability", "AOD options found in arguments. Populating from them.");
      return true;
    }
    if (arg.starts_with("--aod-parent-")) {
      O2_SIGNPOST_EVENT_EMIT(capabilities, sid, "DiscoverAODOptionsInCommandLineCapability", "AOD options found in arguments. Populating from them.");
      return true;
    }
    if (arg.starts_with("--aod-origin-")) {
      O2_SIGNPOST_EVENT_EMIT(capabilities, sid, "DiscoverAODOptionsInCommandLineCapability", "AOD options found in arguments. Populating from them.");
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

struct DiscoverAODOptionsInCommandLineCapability : o2::framework::CapabilityPlugin {
  Capability* create() override
  {
    return new Capability{
      .name = "DiscoverAODOptionsInCommandLineCapability",
      .checkIfNeeded = lookForCommandLineAODOptions,
      .requiredPlugin = "O2Framework:DiscoverAODOptionsInCommandLine"};
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

struct DiscoverAODOptionsInCommandLine : o2::framework::ConfigDiscoveryPlugin {
  ConfigDiscovery* create() override
  {
    return new ConfigDiscovery{
      .init = []() {},
      .discover = [](ConfigParamRegistry& registry, int argc, char** argv) -> std::vector<ConfigParamSpec> {
        O2_SIGNPOST_ID_GENERATE(sid, capabilities);
        O2_SIGNPOST_EVENT_EMIT(capabilities, sid, "DiscoverAODOptionsInCommandLine",
                               "Discovering AOD handling related options in commandline arguments.");
        std::vector<ConfigParamSpec> results;
        bool injectOption = true;
        for (size_t i = 0; i < argc; i++) {
          std::string_view arg = argv[i];
          if (!arg.starts_with("--aod-writer-") && !arg.starts_with("--aod-parent-") && !arg.starts_with("--aod-origin-")) {
            continue;
          }
          std::string key = arg.data() + 2;
          std::string value = argv[i + 1];
          O2_SIGNPOST_EVENT_EMIT(capabilities, sid, "DiscoverAODOptionsInCommandLine",
                                 "Found %{public}s with value %{public}s.", key.c_str(), value.c_str());
          if (key == "aod-writer-compression") {
            int numericValue = std::stoi(value);
            results.push_back(ConfigParamSpec{"aod-writer-compression", VariantType::Int, numericValue, {"AOD Compression options"}});
            injectOption = false;
          }
          if (key == "aod-parent-base-path-replacement") {
            results.push_back(ConfigParamSpec{"aod-parent-base-path-replacement", VariantType::String, value, {R"(Replace base path of parent files. Syntax: FROM;TO. E.g. "alien:///path/in/alien;/local/path". Enclose in "" on the command line.)"}});
          }
          if (key == "aod-parent-access-level") {
            results.push_back(ConfigParamSpec{"aod-parent-access-level", VariantType::String, value, {"Allow parent file access up to specified level. Default: no (0)"}});
          }
          if (key == "aod-origin-level-mapping") {
            results.push_back(ConfigParamSpec{"aod-origin-level-mapping", VariantType::String, value, {"Map origin to parent level for AOD reading. Syntax: ORIGIN:LEVEL[,ORIGIN2:LEVEL2,...]. E.g. \"DYN:1\"."}});
          }
        }
        if (injectOption) {
          results.push_back(ConfigParamSpec{"aod-writer-compression", VariantType::Int, 505, {"AOD Compression options"}});
        }
        return results;
      }};
  }
};

struct ImplementationContext {
  std::vector<RootArrowFactory> implementations;
};

std::function<void*(std::shared_ptr<arrow::fs::FileSystem>, std::string const&)> getHandleByClass(char const* classname)
{
  return [c = TClass::GetClass(classname)](std::shared_ptr<arrow::fs::FileSystem> fs, std::string const& path) -> void* {
    if (auto tfileFS = std::dynamic_pointer_cast<TFileFileSystem>(fs)) {
      return tfileFS->GetFile()->GetObjectChecked(path.c_str(), c);
    } else if (auto tbufferFS = std::dynamic_pointer_cast<TBufferFileFS>(fs)) {
      tbufferFS->GetBuffer()->Reset();
      return tbufferFS->GetBuffer()->ReadObjectAny(c);
    }
    return nullptr;
  };
}

std::function<bool(char const*)> matchClassByName(std::string_view classname)
{
  return [c = classname](char const* attempt) -> bool {
    return c == attempt;
  };
}

void lazyLoadFactory(std::vector<RootArrowFactory>& implementations, char const* specs)
{
  // Lazy loading of the plugin so that we do not bring in RNTuple / TTree if not needed
  if (implementations.empty()) {
    std::vector<LoadablePlugin> plugins;
    auto morePlugins = PluginManager::parsePluginSpecString(specs);
    for (auto& extra : morePlugins) {
      plugins.push_back(extra);
    }
    PluginManager::loadFromPlugin<RootArrowFactory, RootArrowFactoryPlugin>(plugins, implementations);
    if (implementations.empty()) {
      return;
    }
  }
}

struct RNTupleObjectReadingCapability : o2::framework::RootObjectReadingCapabilityPlugin {
  RootObjectReadingCapability* create() override
  {
    auto context = new ImplementationContext;

    return new RootObjectReadingCapability
    {
      .name = "rntuple",
      .lfn2objectPath = [](std::string s) -> std::string {
         std::replace(s.begin()+1, s.end(), '/', '-');
#if __has_include(<ROOT/RFieldBase.hxx>)
         if (s.starts_with("/")) {
          return std::string(s.begin() + 1, s.end());
        } else {
          return s;
        } },
#else
         if (s.starts_with("/")) {
          return s;
        } else {
          return "/" + s;
        } },
#endif
#if __has_include(<ROOT/RFieldBase.hxx>)
      .getHandle = getHandleByClass("ROOT::RNTuple"),
      .checkSupport = matchClassByName("ROOT::RNTuple"),
#else
      .getHandle = getHandleByClass("ROOT::Experimental::RNTuple"),
      .checkSupport = matchClassByName("ROOT::Experimental::RNTuple"),
#endif
      .factory = [context]() -> RootArrowFactory& {
        lazyLoadFactory(context->implementations, "O2FrameworkAnalysisRNTupleSupport:RNTupleObjectReadingImplementation");
        return context->implementations.back();
      }
    };
  }
};

struct TTreeObjectReadingCapability : o2::framework::RootObjectReadingCapabilityPlugin {
  RootObjectReadingCapability* create() override
  {
    auto context = new ImplementationContext;

    return new RootObjectReadingCapability{
      .name = "ttree",
      .lfn2objectPath = [](std::string s) { return s; },
      .getHandle = getHandleByClass("TTree"),
      .checkSupport = matchClassByName("TTree"),
      .factory = [context]() -> RootArrowFactory& {
        lazyLoadFactory(context->implementations, "O2FrameworkAnalysisTTreeSupport:TTreeObjectReadingImplementation");
        return context->implementations.back();
      }};
  }
};

DEFINE_DPL_PLUGINS_BEGIN
DEFINE_DPL_PLUGIN_INSTANCE(DiscoverMetadataInAODCapability, Capability);
DEFINE_DPL_PLUGIN_INSTANCE(DiscoverMetadataInCommandLineCapability, Capability);
DEFINE_DPL_PLUGIN_INSTANCE(DiscoverAODOptionsInCommandLineCapability, Capability);
DEFINE_DPL_PLUGIN_INSTANCE(DiscoverMetadataInCommandLine, ConfigDiscovery);
DEFINE_DPL_PLUGIN_INSTANCE(DiscoverAODOptionsInCommandLine, ConfigDiscovery);
DEFINE_DPL_PLUGIN_INSTANCE(RNTupleObjectReadingCapability, RootObjectReadingCapability);
DEFINE_DPL_PLUGIN_INSTANCE(TTreeObjectReadingCapability, RootObjectReadingCapability);
DEFINE_DPL_PLUGINS_END
} // namespace o2::framework
