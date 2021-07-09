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
#include "Framework/Logger.h"
#include <uv.h>
#include <functional>
#include <vector>

namespace o2::framework
{

void PluginManager::load(std::vector<PluginInfo>& libs, const char* dso, std::function<void(DPLPluginHandle*)>& onSuccess)
{
  auto plugin = std::find_if(libs.begin(), libs.end(), [dso](PluginInfo& info) { return info.name == dso; });
  if (plugin != libs.end()) {
    return onSuccess(plugin->instance);
  }
  uv_lib_t* supportLib = (uv_lib_t*)malloc(sizeof(uv_lib_t));
  int result = 0;
#ifdef __APPLE__
  char const* extension = "dylib";
#else
  char const* extension = "so";
#endif
  std::string filename = fmt::format("lib{}.{}", dso, extension);
  result = uv_dlopen(filename.c_str(), supportLib);
  if (result == -1) {
    LOG(FATAL) << uv_dlerror(supportLib);
    return;
  }
  void* callback = nullptr;
  DPLPluginHandle* (*dpl_plugin_callback)(DPLPluginHandle*);

  result = uv_dlsym(supportLib, "dpl_plugin_callback", (void**)&dpl_plugin_callback);
  if (result == -1) {
    LOG(FATAL) << uv_dlerror(supportLib);
    return;
  }
  if (dpl_plugin_callback == nullptr) {
    LOGP(FATAL, "Could not find the {} plugin.", dso);
    return;
  }
  DPLPluginHandle* pluginInstance = dpl_plugin_callback(nullptr);
  libs.push_back({supportLib, dso});
  onSuccess(pluginInstance);
}
} // namespace o2::framework
