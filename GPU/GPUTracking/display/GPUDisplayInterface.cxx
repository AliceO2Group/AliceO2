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

/// \file GPUDisplayInterface.cxx
/// \author David Rohr

#include "GPUDisplayInterface.h"
#include "GPULogging.h"
#include "utils/qlibload.h"

#include <dlfcn.h>
#include <mutex>
#include <tuple>
#include <stdexcept>

using namespace GPUCA_NAMESPACE::gpu;

static constexpr const char* libName = "lib" LIBRARY_PREFIX "GPUTrackingDisplay" LIBRARY_EXTENSION;
static constexpr const char* funcName = "GPUTrackingDisplayLoader";

static void* loadUnloadLib(bool load)
{
  static size_t count = 0;
  static LIBRARY_TYPE lib = nullptr;
  static void* func = nullptr;

  static std::mutex mut;
  std::lock_guard<std::mutex> lock(mut);

  if (load) {
    if (count == 0) {
      lib = LIBRARY_LOAD(libName);
      if (lib == nullptr) {
        GPUError("Cannot load display library %s", libName);
        return nullptr;
      }
      func = LIBRARY_FUNCTION(lib, funcName);
      if (func == nullptr) {
        GPUError("Error getting factory function from display library %s", libName);
        LIBRARY_CLOSE(lib);
        lib = nullptr;
        return nullptr;
      }
    }
    count++;
    return func;
  } else {
    if (count == 0) {
      throw std::runtime_error("reference count mismatch");
    }
    if (--count == 0) {
      LIBRARY_CLOSE(lib);
      lib = nullptr;
      func = nullptr;
    }
  }
  return nullptr;
}

GPUDisplayInterface* GPUDisplayInterface::getDisplay(GPUDisplayFrontendInterface* frontend, GPUChainTracking* chain, GPUQA* qa, const GPUParam* param, const GPUCalibObjectsConst* calib, const GPUSettingsDisplay* config)
{
  std::tuple args = {frontend, chain, qa, param, calib, config};
  auto func = (GPUDisplayInterface * (*)(const char*, void*)) loadUnloadLib(true);
  return func ? func("display", &args) : nullptr;
}

GPUDisplayFrontendInterface* GPUDisplayFrontendInterface::getFrontend(const char* type)
{
  std::tuple args = {type};
  auto func = (GPUDisplayFrontendInterface * (*)(const char*, void*)) loadUnloadLib(true);
  return func ? func("frontend", &args) : nullptr;
}

GPUDisplayInterface::GPUDisplayInterface() = default;
GPUDisplayInterface::~GPUDisplayInterface()
{
  loadUnloadLib(false);
}
GPUDisplayFrontendInterface::GPUDisplayFrontendInterface() = default;
GPUDisplayFrontendInterface::~GPUDisplayFrontendInterface()
{
  loadUnloadLib(false);
}
