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

/// \file GPUDisplayLoader.cxx
/// \author David Rohr

#include "GPUDisplay.h"
#include "frontend/GPUDisplayFrontend.h"
#include "GPUDisplayInterface.h"

#include <tuple>
#include <stdexcept>

using namespace GPUCA_NAMESPACE::gpu;

extern "C" void* GPUTrackingDisplayLoader(const char*, void*);

template <class T, typename... Args>
static inline T* createHelper(Args... args)
{
  return new T(args...);
}

void* GPUTrackingDisplayLoader(const char* type, void* args)
{
  if (strcmp(type, "display") == 0) {
    auto x = (std::tuple<GPUDisplayFrontend*, GPUChainTracking*, GPUQA*, const GPUParam*, const GPUCalibObjectsConst*, const GPUSettingsDisplay*>*)args;
    return std::apply([](auto&&... y) { return createHelper<GPUDisplay>(y...); }, *x);
  } else if (strcmp(type, "frontend") == 0) {
    auto x = (std::tuple<const char*>*)args;
    return std::apply([](auto&&... y) { return GPUDisplayFrontend::getFrontend(y...); }, *x);
  } else {
    throw std::runtime_error("Invalid display obejct type specified");
  }
  return nullptr;
}
