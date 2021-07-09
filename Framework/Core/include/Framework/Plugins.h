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
#ifndef O2_FRAMEWORK_PLUGINS_H_
#define O2_FRAMEWORK_PLUGINS_H_

#include "Framework/AlgorithmSpec.h"
#include <cstring>
#include <string>
#include <functional>
#include <uv.h>

namespace o2::framework
{

enum struct DplPluginKind : int {
  // A plugin which can customise the workflow. Needs to return
  // an object of kind o2::framework::WorkflowCustomizationService
  CustomAlgorithm,
  // A plugin which implements a ImGUI GUI. Needs to return an
  // object of the kind o2::framework::DebugGUIImpl
  DebugGUIImpl,
  // A plugin which implements a custom Services. Needs to return
  // an object of the kind o2::framework::ServiceSpec
  CustomService
};

} // namespace o2::framework

/// An handle for a generic DPL plugin.
/// The handle is returned by the dpl_plugin_callback()
struct DPLPluginHandle {
  void* instance;
  char const* name;
  enum o2::framework::DplPluginKind kind;
  DPLPluginHandle* previous;
};

// Struct to hold live plugin information which the plugin itself cannot
// know and that is owned by the framework.
struct PluginInfo {
  uv_lib_t* dso = nullptr;
  std::string name;
  DPLPluginHandle* instance = nullptr;
};

#define DEFINE_DPL_PLUGIN(NAME, KIND)                                                                    \
  extern "C" {                                                                                           \
  DPLPluginHandle* dpl_plugin_callback(DPLPluginHandle* previous)                                        \
  {                                                                                                      \
    return new DPLPluginHandle{new NAME{}, strdup(#NAME), o2::framework::DplPluginKind::KIND, previous}; \
  }                                                                                                      \
  }

#define DEFINE_DPL_PLUGINS_BEGIN                                  \
  extern "C" {                                                    \
  DPLPluginHandle* dpl_plugin_callback(DPLPluginHandle* previous) \
  {

#define DEFINE_DPL_PLUGIN_INSTANCE(NAME, KIND) \
  previous = new DPLPluginHandle{new NAME{}, strdup(#NAME), o2::framework::DplPluginKind::KIND, previous};

#define DEFINE_DPL_PLUGINS_END \
  return previous;             \
  }                            \
  }

namespace o2::framework
{
struct PluginManager {
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
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_PLUGINS_H_
