// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDefGPUParameters.h
/// \author David Rohr

// This files contains compile-time constants affecting the GPU performance.
// Many of these constants are GPU-architecture specific.
// This file also contains all constants describing memory limitations, essentially limiting the total number of tracks, etc.
// Compile-time constants affecting the tracking algorithms / results are located in GPUDefConstantsAndSettings.h

#ifndef GPUDEFGPUPARAMETERS_H
#define GPUDEFGPUPARAMETERS_H
// clang-format off

#ifndef GPUDEF_H
#error Please include GPUDef.h
#endif

// GPU Run Configuration
#if defined(GPUCA_GPUTYPE_VEGA)
  #define GPUCA_WARP_SIZE 64
  #define GPUCA_MINBLOCK_COUNT_CONSTRUCTOR 1
  #define GPUCA_MINBLOCK_COUNT_SELECTOR 8
  #define GPUCA_MINBLOCK_COUNT_HITSSORTER 2
  #define GPUCA_MINBLOCK_COUNT_FINDER 1
  #define GPUCA_THREAD_COUNT 256
  #define GPUCA_THREAD_COUNT_HITSSORTER 1024
  #define GPUCA_THREAD_COUNT_HITSFINDER 1024
  #define GPUCA_THREAD_COUNT_CONSTRUCTOR 512
  #define GPUCA_THREAD_COUNT_SELECTOR 256
  #define GPUCA_THREAD_COUNT_FINDER 1024
  #define GPUCA_THREAD_COUNT_CLEANER 896
  #define GPUCA_THREAD_COUNT_CFDECODE 64
  #define GPUCA_THREAD_COUNT_FIT 64
  #define GPUCA_THREAD_COUNT_PEAK_FINDER 384 // To be used in peak finder, once clusterer supports scratch pads of different size
  #define GPUCA_THREAD_COUNT_COMPRESSION1 256
  #define GPUCA_THREAD_COUNT_COMPRESSION2 512
  #define GPUCA_THREAD_COUNT_CLUSTERER 512
  #define GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP 5
  #define GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE 6
  #define GPUCA_CONSTRUCTOR_IN_PIPELINE 0
  #define GPUCA_SELECTOR_IN_PIPELINE 0
  #define GPUCA_NO_ATOMIC_PRECHECK 1
#elif defined(GPUCA_GPUTYPE_TURING)
  #define GPUCA_WARP_SIZE 32
  #define GPUCA_MINBLOCK_COUNT_CONSTRUCTOR 1
  #define GPUCA_MINBLOCK_COUNT_SELECTOR 2
  #define GPUCA_MINBLOCK_COUNT_HITSSORTER 1
  #define GPUCA_MINBLOCK_COUNT_FINDER 1
  #define GPUCA_MINBLOCK_COUNT_DECODE 4
  #define GPUCA_MINBLOCK_COUNT_FIT 1
  #define GPUCA_THREAD_COUNT 512
  #define GPUCA_THREAD_COUNT_HITSSORTER 512
  #define GPUCA_THREAD_COUNT_HITSFINDER 512
  #define GPUCA_THREAD_COUNT_CONSTRUCTOR 384
  #define GPUCA_THREAD_COUNT_SELECTOR 512
  #define GPUCA_THREAD_COUNT_FINDER 640
  #define GPUCA_THREAD_COUNT_CLEANER 512
  #define GPUCA_THREAD_COUNT_CFDECODE 96
  #define GPUCA_THREAD_COUNT_FIT 256
  #define GPUCA_THREAD_COUNT_COMPRESSION1 128
  #define GPUCA_THREAD_COUNT_COMPRESSION2 448
  #define GPUCA_THREAD_COUNT_CLUSTERER 512
  #define GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP 4
  #define GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE 20
  #define GPUCA_CONSTRUCTOR_IN_PIPELINE 1
  #define GPUCA_SELECTOR_IN_PIPELINE 0
  #define GPUCA_TRACKLET_SELECTOR_SLICE_COUNT 1
  #define GPUCA_NO_ATOMIC_PRECHECK 1
  // #define GPUCA_USE_TEXTURES
#elif defined(GPUCA_GPUTYPE_OPENCL)
#elif defined(GPUCA_GPUCODE)
  #error GPU TYPE NOT SET
#endif

#ifdef GPUCA_GPUCODE
  // Default settings, if not already set for selected GPU type
  #ifndef GPUCA_MINBLOCK_COUNT_CONSTRUCTOR
  #define GPUCA_MINBLOCK_COUNT_CONSTRUCTOR 1
  #endif
  #ifndef GPUCA_MINBLOCK_COUNT_SELECTOR
  #define GPUCA_MINBLOCK_COUNT_SELECTOR 1
  #endif
  #ifndef GPUCA_MINBLOCK_COUNT_FINDER
  #define GPUCA_MINBLOCK_COUNT_FINDER 1
  #endif
  #ifndef GPUCA_MINBLOCK_COUNT_DECODE
  #define GPUCA_MINBLOCK_COUNT_DECODE 4
  #endif
  #ifndef GPUCA_MINBLOCK_COUNT_HITSSORTER
  #define GPUCA_MINBLOCK_COUNT_HITSSORTER 1
  #endif
  #ifndef GPUCA_MINBLOCK_COUNT_FIT
  #define GPUCA_MINBLOCK_COUNT_FIT 1
  #endif
  #ifndef GPUCA_THREAD_COUNT
  #define GPUCA_THREAD_COUNT 256
  #endif
  #ifndef GPUCA_THREAD_COUNT_CONSTRUCTOR
  #define GPUCA_THREAD_COUNT_CONSTRUCTOR 256
  #endif
  #ifndef GPUCA_THREAD_COUNT_SELECTOR
  #define GPUCA_THREAD_COUNT_SELECTOR 256
  #endif
  #ifndef GPUCA_THREAD_COUNT_FINDER
  #define GPUCA_THREAD_COUNT_FINDER 256
  #endif
  #ifndef GPUCA_THREAD_COUNT_CLEANER
  #define GPUCA_THREAD_COUNT_CLEANER 256
  #endif
  #ifndef GPUCA_THREAD_COUNT_TRD
  #define GPUCA_THREAD_COUNT_TRD 512
  #endif
  #ifndef GPUCA_THREAD_COUNT_CONVERTER
  #define GPUCA_THREAD_COUNT_CONVERTER 256
  #endif
  #ifndef GPUCA_THREAD_COUNT_COMPRESSION1
  #define GPUCA_THREAD_COUNT_COMPRESSION1 256
  #endif
  #ifndef GPUCA_THREAD_COUNT_COMPRESSION2
  #define GPUCA_THREAD_COUNT_COMPRESSION2 256
  #endif
  #ifndef GPUCA_THREAD_COUNT_CFDECODE
  #define GPUCA_THREAD_COUNT_CFDECODE 128
  #endif
  #ifndef GPUCA_THREAD_COUNT_CLUSTERER
  #define GPUCA_THREAD_COUNT_CLUSTERER 128
  #endif
  #ifndef GPUCA_THREAD_COUNT_FIT
  #define GPUCA_THREAD_COUNT_FIT 256
  #endif
  #ifndef GPUCA_THREAD_COUNT_ITS
  #define GPUCA_THREAD_COUNT_ITS 256
  #endif
  #ifndef GPUCA_THREAD_COUNT_HITSFINDER
  #define GPUCA_THREAD_COUNT_HITSFINDER 256
  #endif
  #ifndef GPUCA_THREAD_COUNT_HITSSORTER
  #define GPUCA_THREAD_COUNT_HITSSORTER 256
  #endif
#else
  // The following defaults are needed to compile the host code
  #define GPUCA_THREAD_COUNT_CLUSTERER 1
  #define GPUCA_THREAD_COUNT_COMPRESSION2 1
#endif
//#ifndef GPUCA_THREAD_COUNT_SCAN
#define GPUCA_THREAD_COUNT_SCAN 512 // TODO: WARNING!!! Must not be GPUTYPE-dependent right now! // TODO: Fix!
//#endif

#ifndef GPUCA_NEIGHBORSFINDER_REGS
#define GPUCA_NEIGHBORSFINDER_REGS NONE, 0
#endif
#ifdef GPUCA_GPUCODE
  #ifndef GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP
  #define GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP 6
  #endif
  #ifndef GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE
  #define GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE 12
  #endif
  #ifndef GPUCA_CONSTRUCTOR_IN_PIPELINE
  #define GPUCA_CONSTRUCTOR_IN_PIPELINE 1
  #endif
  #ifndef GPUCA_SELECTOR_IN_PIPELINE
  #define GPUCA_SELECTOR_IN_PIPELINE 0
  #endif
  #ifndef GPUCA_TRACKLET_SELECTOR_SLICE_COUNT
  #define GPUCA_TRACKLET_SELECTOR_SLICE_COUNT 8                          // Currently must be smaller than avaiable MultiProcessors on GPU or will result in wrong results
  #endif
#else
  #define GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP 0
  #define GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE 0
  #define GPUCA_CONSTRUCTOR_IN_PIPELINE 1
  #define GPUCA_SELECTOR_IN_PIPELINE 1
  #define GPUCA_TRACKLET_SELECTOR_SLICE_COUNT 1
  #define GPUCA_THREAD_COUNT_FINDER 1
#endif

#ifndef GPUCA_WARP_SIZE
#ifdef GPUCA_GPUCODE
#define GPUCA_WARP_SIZE 32
#else
#define GPUCA_WARP_SIZE 1
#endif
#endif

#define GPUCA_MAX_THREADS 1024
#define GPUCA_MAX_STREAMS 32

#define GPUCA_SORT_STARTHITS_GPU                                       // Sort the start hits when running on GPU
#define GPUCA_ROWALIGNMENT 16                                          // Align of Row Hits and Grid
#define GPUCA_BUFFER_ALIGNMENT 64                                      // Alignment of buffers obtained from SetPointers
#define GPUCA_MEMALIGN (64 * 1024)                                     // Alignment of allocated memory blocks

// #define GPUCA_TRACKLET_CONSTRUCTOR_DO_PROFILE                       // Output Profiling Data for Tracklet Constructor Tracklet Scheduling

// Default maximum numbers
#define GPUCA_MAX_CLUSTERS           ((size_t)     1024 * 1024 * 1024) // Maximum number of TPC clusters
#define GPUCA_MAX_TRD_TRACKLETS      ((size_t)             128 * 1024) // Maximum number of TRD tracklets
#define GPUCA_MAX_ITS_FIT_TRACKS     ((size_t)              96 * 1024) // Max number of tracks for ITS track fit
#define GPUCA_TRACKER_CONSTANT_MEM   ((size_t)              63 * 1024) // Amount of Constant Memory to reserve
#define GPUCA_MEMORY_SIZE            ((size_t) 6 * 1024 * 1024 * 1024) // Size of memory allocated on Device
#define GPUCA_HOST_MEMORY_SIZE       ((size_t) 6 * 1024 * 1024 * 1024) // Size of memory allocated on Host
#define GPUCA_GPU_STACK_SIZE         ((size_t)               8 * 1024) // Stack size per GPU thread

#define GPUCA_MAX_SLICE_NTRACK (2 << 24)                               // Maximum number of tracks per slice (limited by track id format)

// #define GPUCA_KERNEL_DEBUGGER_OUTPUT

// Some assertions to make sure out parameters are not invalid
#ifdef GPUCA_NOCOMPAT
  static_assert(GPUCA_MAXN >= GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP, "Invalid GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP");
  static_assert(GPUCA_ROW_COUNT >= GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE, "Invalid GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE");
  static_assert(GPUCA_THREAD_COUNT_COMPRESSION2 * 2 <= GPUCA_TPC_COMP_CHUNK_SIZE, "Invalid GPUCA_TPC_COMP_CHUNK_SIZE");
#endif

// Derived parameters
#ifdef GPUCA_USE_TEXTURES
  #define GPUCA_TEXTURE_FETCH_CONSTRUCTOR                              // Fetch data through texture cache
  #define GPUCA_TEXTURE_FETCH_NEIGHBORS                                // Fetch also in Neighbours Finder
#endif
#if defined(GPUCA_SORT_STARTHITS_GPU) && defined(GPUCA_GPUCODE)
  #define GPUCA_SORT_STARTHITS
#endif

// Error Codes for GPU Tracker
#define GPUCA_ERROR_NONE 0
#define GPUCA_ERROR_ROWSTARTHIT_OVERFLOW 1
#define GPUCA_ERROR_STARTHIT_OVERFLOW 2
#define GPUCA_ERROR_TRACKLET_OVERFLOW 3
#define GPUCA_ERROR_TRACKLET_HIT_OVERFLOW 4
#define GPUCA_ERROR_TRACK_OVERFLOW 5
#define GPUCA_ERROR_TRACK_HIT_OVERFLOW 6
#define GPUCA_ERROR_GLOBAL_TRACKING_TRACK_OVERFLOW 7
#define GPUCA_ERROR_GLOBAL_TRACKING_TRACK_HIT_OVERFLOW 8
#define GPUCA_ERROR_STRINGS {"GPUCA_ERROR_NONE", "GPUCA_ERROR_ROWSTARTHIT_OVERFLOW", "GPUCA_ERROR_STARTHIT_OVERFLOW", "GPUCA_ERROR_TRACKLET_OVERFLOW", "GPUCA_ERROR_TRACKLET_HIT_OVERFLOW", "GPUCA_ERROR_TRACK_OVERFLOW", "GPUCA_ERROR_TRACK_HIT_OVERFLOW", "GPUCA_ERROR_GLOBAL_TRACKING_TRACK_OVERFLOW", "GPUCA_ERROR_GLOBAL_TRACKING_TRACK_HIT_OVERFLOW"}

// clang-format on
#endif
