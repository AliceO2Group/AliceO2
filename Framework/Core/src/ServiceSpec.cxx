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

#include "Framework/ServiceSpec.h"
#include "Framework/RuntimeError.h"
#include "Framework/Logger.h"
#include "Framework/Plugins.h"
#include <sstream>
#include <unordered_set>
#include <uv.h>

namespace o2::framework
{

auto findOverrideByName = [](std::string_view name) {
  return [name](const OverrideServiceSpec& spec) { return spec.name == name; };
};

OverrideServiceSpecs ServiceSpecHelpers::parseOverrides(char const* overrideString)
{
  if (overrideString == nullptr || *overrideString == '\0') {
    return {};
  }
  // Helper to split a string into a vector of string_views
  auto split = [](std::string_view str, char delim) {
    std::vector<std::string_view> result;
    std::string_view::size_type start = 0;
    auto end = str.find(delim);
    while (end != std::string_view::npos) {
      result.push_back(str.substr(start, end - start));
      start = end + 1;
      end = str.find(delim, start);
    }
    result.push_back(str.substr(start, end));
    return result;
  };

  // if it is set, it will be a comma separated list of service names and their
  // enabled/disabled status.
  // if the service is enabled, it will be added to the list of services to be forced to be enabled.
  // if the service is disabled, it will be added to the list of services to be forced to be disabled.

  // Keep it simple. This is only a few elements, so we can use a vector.
  std::vector<OverrideServiceSpec> overrides;
  auto msg =
    "DPL(_DRIVER)_OVERRIDE_SERVICES must be a comma separated list of service names and their enabled/disabled status. "
    "The format is <service name>:<enable/disable/1/0>"
    "Error while parsing: " +
    std::string(overrideString);
  auto overrideList = split(overrideString, ',');
  for (auto& overrideStr : overrideList) {
    auto overrideParts = split(overrideStr, ':');
    if (overrideParts.size() != 2) {
      throw std::runtime_error(msg.data());
    }
    bool active;
    std::string serviceName{overrideParts[0]};
    if (overrideParts[1] == "1") {
      active = true;
    } else if (overrideParts[1] == "0") {
      active = false;
    } else if (overrideParts[1] == "enable") {
      active = true;
    } else if (overrideParts[1] == "disable") {
      active = false;
    } else {
      throw std::runtime_error(msg.data());
    }
    if (std::find_if(overrides.begin(), overrides.end(), findOverrideByName(serviceName)) != overrides.end()) {
      throw std::runtime_error("Duplicate service name in DPL(_DRIVER)_OVERRIDE_SERVICES: " + serviceName);
    }
    overrides.push_back(OverrideServiceSpec{serviceName, active});
  }
  // Print the configuration
  LOG(detail) << "The following services will be overridden by the DPL_OVERRIDE_SERVICES environment variable:";
  for (auto& override : overrides) {
    LOG(detail) << override.name << ": " << override.active;
  }
  return overrides;
};

ServiceSpecs ServiceSpecHelpers::filterDisabled(ServiceSpecs originals, OverrideServiceSpecs const& overrides)
{

  std::vector<ServiceSpec> result;
  for (auto& original : originals) {
    auto override = std::find_if(overrides.begin(), overrides.end(), findOverrideByName(original.name));
    if (override != overrides.end()) {
      LOGP(detail, "Overriding service {} to {}", original.name, override->active ? "enabled" : "disabled");
      original.active = override->active;
    }
    if (original.active) {
      result.push_back(original);
    }
  }
  return result;
}

std::vector<LoadableService> ServiceHelpers::parseServiceSpecString(char const* str)
{
  std::vector<LoadableService> loadableServices;
  enum struct ParserState : int {
    IN_LIBRARY,
    IN_NAME,
    IN_END,
    IN_ERROR,
  };
  const char* cur = str;
  const char* next = cur;
  ParserState state = ParserState::IN_LIBRARY;
  std::string_view library;
  std::string_view name;
  while (cur && *cur != '\0') {
    ParserState previousState = state;
    state = ParserState::IN_ERROR;
    switch (previousState) {
      case ParserState::IN_LIBRARY:
        next = strchr(cur, ':');
        if (next != nullptr) {
          state = ParserState::IN_NAME;
          library = std::string_view(cur, next - cur);
        }
        break;
      case ParserState::IN_NAME:
        next = strchr(cur, ',');
        if (next == nullptr) {
          state = ParserState::IN_END;
          name = std::string_view(cur, strlen(cur));
        } else {
          name = std::string_view(cur, next - cur);
          state = ParserState::IN_LIBRARY;
        }
        loadableServices.push_back({std::string(name), std::string(library)});
        break;
      case ParserState::IN_END:
        break;
      case ParserState::IN_ERROR:
        LOG(error) << "Error while parsing DPL_LOAD_SERVICES";
        break;
    }
    if (!next) {
      break;
    }
    cur = next + 1;
  };
  return loadableServices;
}

void ServiceHelpers::loadFromPlugin(std::vector<LoadableService> const& loadableServices, std::vector<ServiceSpec>& specs)
{
  struct LoadedDSO {
    std::string library;
    uv_lib_t handle;
  };

  struct LoadedPlugin {
    std::string name;
    ServicePlugin* factory;
  };
  std::vector<LoadedDSO> loadedDSOs;
  std::vector<LoadedPlugin> loadedPlugins;
  for (auto& loadableService : loadableServices) {
    auto loadedDSO = std::find_if(loadedDSOs.begin(), loadedDSOs.end(), [&loadableService](auto& dso) {
      return dso.library == loadableService.library;
    });

    if (loadedDSO == loadedDSOs.end()) {
      uv_lib_t handle;
#ifdef __APPLE__
      auto libraryName = fmt::format("lib{}.dylib", loadableService.library);
#else
      auto libraryName = fmt::format("lib{}.so", loadableService.library);
#endif
      auto ret = uv_dlopen(libraryName.c_str(), &handle);
      if (ret != 0) {
        LOGP(error, "Could not load library {}", loadableService.library);
        LOG(error) << uv_dlerror(&handle);
        continue;
      }
      loadedDSOs.push_back({loadableService.library, handle});
      loadedDSO = loadedDSOs.end() - 1;
    }
    int result = 0;

    auto loadedPlugin = std::find_if(loadedPlugins.begin(), loadedPlugins.end(), [&loadableService](auto& plugin) {
      return plugin.name == loadableService.name;
    });

    if (loadedPlugin == loadedPlugins.end()) {
      DPLPluginHandle* (*dpl_plugin_callback)(DPLPluginHandle*);
      result = uv_dlsym(&loadedDSO->handle, "dpl_plugin_callback", (void**)&dpl_plugin_callback);
      if (result == -1) {
        LOG(error) << uv_dlerror(&loadedDSO->handle);
        continue;
      }

      DPLPluginHandle* pluginInstance = dpl_plugin_callback(nullptr);
      ServicePlugin* factory = PluginManager::getByName<ServicePlugin>(pluginInstance, loadableService.name.c_str());
      if (factory == nullptr) {
        LOGP(error, "Could not find service {} in library {}", loadableService.name, loadableService.library);
        continue;
      }

      loadedPlugins.push_back({loadableService.name, factory});
      loadedPlugin = loadedPlugins.begin() + loadedPlugins.size() - 1;
    }
    assert(loadedPlugin != loadedPlugins.end());
    assert(loadedPlugin->factory != nullptr);

    ServiceSpec* spec = loadedPlugin->factory->create();
    if (!spec) {
      LOG(error) << "Plugin " << loadableService.name << " could not be created";
      continue;
    }
    LOGP(debug, "Loading service {} from {}", loadableService.name, loadableService.library);
    specs.push_back(*spec);
  }
}
} // namespace o2::framework
