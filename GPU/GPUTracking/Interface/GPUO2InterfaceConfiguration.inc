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

/// \file GPUO2InterfaceConfiguration.inc
/// \author David Rohr

// Note, this must potentionally be included first, before other GPU headers!
// Do not include in .h files!

#ifndef GPUO2INTERFACECONFIGURATIONINC_H
#define GPUO2INTERFACECONFIGURATIONINC_H

#ifdef GPUCA_STANDALONE
#define GPU_GET_CONFIG(configName) static_assert(false, "GPU_GET_CONFIG not available in standalone benchmark")
#else
#include "GPUO2ExternalUser.h"
#include "GPUCommonRtypes.h"
#include "GPUDefMacros.h"
#include "GPUO2InterfaceConfiguration.h"
#include "GPUO2ConfigurableParam.h"
#include <type_traits>

#define GPU_GET_CONFIG(configName) []() -> decltype(auto) {                            \
  static_assert(!std::is_same_v<o2::gpu::configName, o2::gpu::GPUSettingsProcessing>); \
  return o2::gpu::GPUCA_M_CAT(GPUConfigurableParam, configName)::Instance();           \
}()
#endif

#endif
