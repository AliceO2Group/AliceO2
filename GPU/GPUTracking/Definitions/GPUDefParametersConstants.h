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

/// \file GPUDefParametersConstants.h
/// \author David Rohr

// This file contains compile-time constants, independent from the backend

#ifndef GPUDEFPARAMETERSCONSTANTS_H
#define GPUDEFPARAMETERSCONSTANTS_H

#include "GPUCommonDef.h"
#include "DataFormatsTPC/Constants.h"
#ifndef GPUCA_GPUCODE_DEVICE
#include <cstddef>
#endif

// clang-format off

#if defined(__CUDACC__) || defined(__HIPCC__)
  #define GPUCA_SPECIALIZE_THRUST_SORTS // Not compiled with RTC, so must be compile-time constant
#endif

namespace o2::gpu::constants
{
static constexpr size_t GPU_MAX_THREADS = 1024;
static constexpr size_t GPU_MAX_STREAMS = 36;

static constexpr size_t GPU_ROWALIGNMENT =  16;      // Align of Row Hits and Grid
static constexpr size_t GPU_BUFFER_ALIGNMENT =  64;  // Alignment of buffers obtained from SetPointers
static constexpr size_t GPU_MEMALIGN =  (64 * 1024); // Alignment of allocated memory blocks

//; Default maximum numbers
static constexpr size_t GPU_MEM_MAX_TPC_CLUSTERS =         1024 * 1024 * 1024ull; // Maximum number of TPC clusters
static constexpr size_t GPU_MEM_MAX_TRD_TRACKLETS =                128 * 1024ull; // Maximum number of TRD tracklets
static constexpr size_t GPU_DEFAULT_MEMORY_SIZE =      6 * 1024 * 1024 * 1024ull; // Size of memory allocated on Device
static constexpr size_t GPU_DEFAULT_HOST_MEMORY_SIZE = 1 * 1024 * 1024 * 1024ull; // Size of memory allocated on Host
static constexpr size_t GPU_STACK_SIZE =                             8 * 1024ull; // Stack size per GPU thread
static constexpr size_t GPU_HEAP_SIZE =                      16 * 1025 * 1024ull; // Stack size per GPU thread
} // namespace o2::gpu::constants

// clang-format on
#endif // GPUDEFPARAMETERSCONSTANTS_H
