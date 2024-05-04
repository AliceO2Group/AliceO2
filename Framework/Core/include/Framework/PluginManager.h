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
#ifndef O2_FRAMEWORK_PLUGIN_MANAGER_H_
#define O2_FRAMEWORK_PLUGIN_MANAGER_H_

#include "Framework/Plugins.h"
#include <cstring>
#include <uv.h>
#include <functional>

// Struct to hold live plugin information which the plugin itself cannot
// know and that is owned by the framework.
struct PluginInfo {
  uv_lib_t* dso = nullptr;
  std::string name;
  DPLPluginHandle* instance = nullptr;
};

// Struct to hold information about the location of a plugin
struct LoadablePlugin {
  std::string name;
  std::string library;
};

namespace o2::framework
{
struct PluginManager {
  using WrapperProcessCallback = std::function<void(AlgorithmSpec::ProcessCallback&, ProcessingContext&)>;

  template <typename T>
  static T* getByName(DPLPluginHandle* handle, char const* name)
  {
    while (handle != nullptr) {
      if (strncmp(handle->name, name, strlen(name)) == 0) {
        return reinterpret_cast<T*>(handle->instance);
      }
      handle = handle->previous;
    }
    return nullptr;
  }
  /// Load a DSO called @a dso and insert its handle in @a infos
  /// On successfull completion @a onSuccess is called passing
  /// the DPLPluginHandle provided by the library.
  static void load(std::vector<PluginInfo>& infos, const char* dso, std::function<void(DPLPluginHandle*)>& onSuccess);
  /// Load an called @plugin from a library called @a library and
  /// return the associtated AlgorithmSpec.
  static auto loadAlgorithmFromPlugin(std::string library, std::string plugin) -> AlgorithmSpec;
  /// Wrap an algorithm with some lambda @wrapper which will be called
  /// with the original callback and the ProcessingContext.
  static auto wrapAlgorithm(AlgorithmSpec const& spec, WrapperProcessCallback&& wrapper) -> AlgorithmSpec;

  /// Parse a comma separated list of <library>:<plugin-name> plugin declarations.
  static std::vector<LoadablePlugin> parsePluginSpecString(char const* str);

  template <typename CONCRETE, typename PLUGIN>
  static void loadFromPlugin(std::vector<LoadablePlugin> const& loadablePlugins, std::vector<CONCRETE>& specs)
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
    for (auto& loadablePlugin : loadablePlugins) {
      auto loadedDSO = std::find_if(loadedDSOs.begin(), loadedDSOs.end(), [&loadablePlugin](auto& dso) {
        return dso.library == loadablePlugin.library;
      });

      if (loadedDSO == loadedDSOs.end()) {
        uv_lib_t handle;
#ifdef __APPLE__
        auto libraryName = fmt::format("lib{}.dylib", loadablePlugin.library);
#else
        auto libraryName = fmt::format("lib{}.so", loadablePlugin.library);
#endif
        auto ret = uv_dlopen(libraryName.c_str(), &handle);
        if (ret != 0) {
          LOGP(error, "Could not load library {}", loadablePlugin.library);
          LOG(error) << uv_dlerror(&handle);
          continue;
        }
        loadedDSOs.push_back({loadablePlugin.library, handle});
        loadedDSO = loadedDSOs.end() - 1;
      }
      int result = 0;

      auto loadedPlugin = std::find_if(loadedPlugins.begin(), loadedPlugins.end(), [&loadablePlugin](auto& plugin) {
        return plugin.name == loadablePlugin.name;
      });

      if (loadedPlugin == loadedPlugins.end()) {
        DPLPluginHandle* (*dpl_plugin_callback)(DPLPluginHandle*);
        result = uv_dlsym(&loadedDSO->handle, "dpl_plugin_callback", (void**)&dpl_plugin_callback);
        if (result == -1) {
          LOG(error) << uv_dlerror(&loadedDSO->handle);
          continue;
        }

        DPLPluginHandle* pluginInstance = dpl_plugin_callback(nullptr);
        PLUGIN* factory = PluginManager::getByName<PLUGIN>(pluginInstance, loadablePlugin.name.c_str());
        if (factory == nullptr) {
          LOGP(error, "Could not find service {} in library {}", loadablePlugin.name, loadablePlugin.library);
          continue;
        }

        loadedPlugins.push_back({loadablePlugin.name, factory});
        loadedPlugin = loadedPlugins.begin() + loadedPlugins.size() - 1;
      }
      assert(loadedPlugin != loadedPlugins.end());
      assert(loadedPlugin->factory != nullptr);

      CONCRETE* spec = loadedPlugin->factory->create();
      if (!spec) {
        LOG(error) << "Plugin " << loadablePlugin.name << " could not be created";
        continue;
      }
      LOGP(debug, "Loading service {} from {}", loadablePlugin.name, loadablePlugin.library);
      specs.push_back(*spec);
    }
  }
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_PLUGIN_MANAGER_H_
