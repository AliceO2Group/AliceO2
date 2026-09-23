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

/// \file GPUDefParametersWrapper.h
/// \author David Rohr

// Wrapper file to load all compile-time parameters (architecture / rtc - dependent ones, and constant ones)
// Compile-time constants affecting the tracking algorithms / results are located in GPUDefConstantsAndSettings.h

#ifndef GPUDEFPARAMETERSWRAPPER_H
#define GPUDEFPARAMETERSWRAPPER_H
// clang-format off

#include "GPUCommonDef.h"
#include "GPUDefMacros.h"

#include "GPUDefParametersDefaults.h"
#include "GPUDefParametersConstants.h"

namespace o2::gpu
{
#if defined(GPUCA_GPUCODE) && !defined(GPUCA_GPUCODE_NO_LAUNCH_BOUNDS)
  template <typename... Args> GPUhdi() static constexpr uint32_t GPUCA_GET_THREAD_COUNT(uint32_t val, Args...) { return val; }
  template <typename... Args> GPUhdi() static constexpr uint32_t GPUCA_GET_WARP_COUNT(uint32_t val, Args...) { return val / GPUCA_WARP_SIZE; }
#else
  static constexpr uint32_t GPUCA_WARP_SIZE = 1; // On the host, a thread is a block is a warp, and we run 1 "device thread" per block.
  #define GPUCA_GET_THREAD_COUNT(...) 1          // This must be a define not a constexpr function
  #define GPUCA_GET_WARP_COUNT(...) 1            // since launch bound constants are not defined in host-code, and must evaluate to 1!
#endif

#define GPUCA_PAR_MERGER_INTERPOLATION_ERROR_TYPE_A GPUCA_DETERMINISTIC_CODE(float, GPUCA_PAR_MERGER_INTERPOLATION_ERROR_TYPE)
#define GPUCA_PAR_DEDX_STORAGE_TYPE_A GPUCA_DETERMINISTIC_CODE(float, GPUCA_PAR_DEDX_STORAGE_TYPE)

// #define GPUCA_TRACKLET_CONSTRUCTOR_DO_PROFILE                       // Output Profiling Data for Tracklet Constructor Tracklet Scheduling

// #define GPUCA_KERNEL_DEBUGGER_OUTPUT

} // namespace o2::gpu

// clang-format on
#endif // GPUDEFPARAMETERSWRAPPER_H
