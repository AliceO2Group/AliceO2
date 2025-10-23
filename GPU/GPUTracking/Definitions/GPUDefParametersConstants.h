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
// clang-format off

#if defined(__CUDACC__) || defined(__HIPCC__)
  #define GPUCA_SPECIALIZE_THRUST_SORTS // Not compiled with RTC, so must be compile-time constant
#endif

#define GPUCA_MAX_THREADS 1024
#define GPUCA_MAX_STREAMS 36

#define GPUCA_ROWALIGNMENT 16                                          // Align of Row Hits and Grid
#define GPUCA_BUFFER_ALIGNMENT 64                                      // Alignment of buffers obtained from SetPointers
#define GPUCA_MEMALIGN (64 * 1024)                                     // Alignment of allocated memory blocks

// Default maximum numbers
#define GPUCA_MAX_CLUSTERS           ((size_t)     1024 * 1024 * 1024) // Maximum number of TPC clusters
#define GPUCA_MAX_TRD_TRACKLETS      ((size_t)             128 * 1024) // Maximum number of TRD tracklets
#define GPUCA_MEMORY_SIZE            ((size_t) 6 * 1024 * 1024 * 1024) // Size of memory allocated on Device
#define GPUCA_HOST_MEMORY_SIZE       ((size_t) 1 * 1024 * 1024 * 1024) // Size of memory allocated on Host
#define GPUCA_GPU_STACK_SIZE         ((size_t)               8 * 1024) // Stack size per GPU thread
#define GPUCA_GPU_HEAP_SIZE          ((size_t)       16 * 1025 * 1024) // Stack size per GPU thread

// clang-format on
#endif // GPUDEFPARAMETERSCONSTANTS_H
