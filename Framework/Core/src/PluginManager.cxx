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
#include "Framework/PluginManager.h"
#include "Framework/Logger.h"
#include <uv.h>
#include <functional>
#include <vector>

namespace o2::framework
{
std::vector<LoadablePlugin> PluginManager::parsePluginSpecString(char const* str)
{
  std::vector<LoadablePlugin> loadablePlugins;
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
        loadablePlugins.push_back({std::string(name), std::string(library)});
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
  return loadablePlugins;
}

void PluginManager::load(std::vector<PluginInfo>& libs, const char* dso, std::function<void(DPLPluginHandle*)>& onSuccess)
{
  auto plugin = std::find_if(libs.begin(), libs.end(), [dso](PluginInfo& info) { return info.name == dso; });
  if (plugin != libs.end()) {
    return onSuccess(plugin->instance);
  }
  auto* supportLib = (uv_lib_t*)malloc(sizeof(uv_lib_t));
  int result = 0;
#ifdef __APPLE__
  char const* extension = "dylib";
#else
  char const* extension = "so";
#endif
  std::string filename = fmt::format("lib{}.{}", dso, extension);
  result = uv_dlopen(filename.c_str(), supportLib);
  if (result == -1) {
    LOG(fatal) << uv_dlerror(supportLib);
    return;
  }
  DPLPluginHandle* (*dpl_plugin_callback)(DPLPluginHandle*);

  result = uv_dlsym(supportLib, "dpl_plugin_callback", (void**)&dpl_plugin_callback);
  if (result == -1) {
    LOG(fatal) << uv_dlerror(supportLib);
    return;
  }
  if (dpl_plugin_callback == nullptr) {
    LOGP(fatal, "Could not find the {} plugin.", dso);
    return;
  }
  DPLPluginHandle* pluginInstance = dpl_plugin_callback(nullptr);
  libs.push_back({supportLib, dso});
  onSuccess(pluginInstance);
}

auto PluginManager::loadAlgorithmFromPlugin(std::string library, std::string plugin) -> AlgorithmSpec
{
  std::shared_ptr<AlgorithmSpec> algorithm{nullptr};
  return AlgorithmSpec{[algorithm, library, plugin](InitContext& ic) mutable -> AlgorithmSpec::ProcessCallback {
    if (algorithm.get()) {
      return algorithm->onInit(ic);
    }

    uv_lib_t supportLib;
    std::string libName = "lib" + library;
#ifdef __APPLE__
    libName += ".dylib";
#else
    libName += ".so";
#endif
    int result = uv_dlopen(libName.c_str(), &supportLib);
    if (result == -1) {
      LOG(fatal) << uv_dlerror(&supportLib);
    }
    DPLPluginHandle* (*dpl_plugin_callback)(DPLPluginHandle*);

    result = uv_dlsym(&supportLib, "dpl_plugin_callback", (void**)&dpl_plugin_callback);
    if (result == -1) {
      LOG(fatal) << uv_dlerror(&supportLib);
    }
    if (dpl_plugin_callback == nullptr) {
      LOGP(fatal, "Could not find the {} plugin in {}.", plugin, libName);
    }
    DPLPluginHandle* pluginInstance = dpl_plugin_callback(nullptr);
    auto* creator = PluginManager::getByName<AlgorithmPlugin>(pluginInstance, plugin.c_str());
    if (!creator) {
      LOGP(fatal, "Could not find the {} plugin in {}.", plugin, libName);
    }
    algorithm = std::make_shared<AlgorithmSpec>(creator->create());
    return algorithm->onInit(ic);
  }};
};

auto PluginManager::wrapAlgorithm(AlgorithmSpec const& spec, std::function<void(AlgorithmSpec::ProcessCallback&, ProcessingContext&)>&& wrapper) -> AlgorithmSpec
{
  return AlgorithmSpec{[spec, wrapper = std::move(wrapper)](InitContext& ic) mutable -> AlgorithmSpec::ProcessCallback {
    // We either use provided onInit to create the ProcessCallback, or we
    // call directly the one provided
    auto old = spec.onInit ? spec.onInit(ic) : spec.onProcess;
    return [old, wrapper = std::move(wrapper)](ProcessingContext& pc) mutable {
      // Make sure the wrapper calls the old callback at some point.
      wrapper(old, pc);
    };
  }};
}
} // namespace o2::framework
