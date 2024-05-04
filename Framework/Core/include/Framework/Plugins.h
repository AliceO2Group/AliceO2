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
#include <string>

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
  CustomService,
  // A plugin which was not initialised properly.
  Unknown
};

/// A service which can be loaded from a shared library.
/// Description is the actual string "LibraryName:ServiceName"
/// which can be used to load it.
template <typename T>
struct LoadableServicePlugin {
  // How to load the given service.
  std::string loadSpec;

  void setInstance(T* instance_)
  {
    ptr = instance_;
  }

  T& operator*() const
  {
    return ptr;
  }

  T* operator->() const
  {
    return ptr;
  }

  T* get() const
  {
    return ptr;
  }

  void reset()
  {
    delete ptr;
    ptr = nullptr;
  }

  T* ptr = nullptr;
};

} // namespace o2::framework

/// An handle for a generic DPL plugin.
/// The handle is returned by the dpl_plugin_callback()
struct DPLPluginHandle {
  void* instance = nullptr;
  char const* name = nullptr;
  enum o2::framework::DplPluginKind kind = o2::framework::DplPluginKind::Unknown;
  DPLPluginHandle* previous = nullptr;
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

#endif // O2_FRAMEWORK_PLUGINS_H_
