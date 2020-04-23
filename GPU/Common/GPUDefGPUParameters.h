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

#include "GPUDefMacros.h"

// GPU Run Configuration
#if defined(GPUCA_GPUTYPE_VEGA)
  #define GPUCA_WARP_SIZE 64
  #define GPUCA_THREAD_COUNT 256
  #define GPUCA_LB_GPUTPCStartHitsSorter 1024, 2
  #define GPUCA_LB_GPUTPCStartHitsFinder 1024
  #define GPUCA_LB_GPUTPCTrackletConstructor 512, 1
  #define GPUCA_LB_GPUTPCTrackletSelector 256, 8
  #define GPUCA_LB_GPUTPCNeighboursFinder 1024, 1
  #define GPUCA_LB_GPUTPCNeighboursCleaner 896
  #define GPUCA_LB_GPUTPCCFDecodeZS 64, 4
  #define GPUCA_LB_GPUTPCGMMergerTrackFit 64, 1
  #define GPUCA_LB_GPUTPCGMMergerFollowLoopers 256, 1
  #define GPUCA_LB_GPUTPCCompressionKernels_step0attached 256
  #define GPUCA_LB_GPUTPCCompressionKernels_step1unattached 512
  #define GPUCA_LB_CLUSTER_FINDER 512
  #define GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP 5
  #define GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE 6
  #define GPUCA_CONSTRUCTOR_IN_PIPELINE 0
  #define GPUCA_SELECTOR_IN_PIPELINE 0
  #define GPUCA_NO_ATOMIC_PRECHECK 1
#elif defined(GPUCA_GPUTYPE_TURING)
  #define GPUCA_WARP_SIZE 32
  #define GPUCA_THREAD_COUNT 512
  #define GPUCA_LB_GPUTPCStartHitsSorter 512, 1
  #define GPUCA_LB_GPUTPCStartHitsFinder 512
  #define GPUCA_LB_GPUTPCTrackletConstructor 384, 1
  #define GPUCA_LB_GPUTPCTrackletSelector 512, 2
  #define GPUCA_LB_GPUTPCNeighboursFinder 640, 1
  #define GPUCA_LB_GPUTPCNeighboursCleaner 512
  #define GPUCA_LB_GPUTPCCFDecodeZS 96, 4
  #define GPUCA_LB_GPUTPCGMMergerTrackFit 256, 1
  #define GPUCA_LB_GPUTPCGMMergerFollowLoopers 256, 1
  #define GPUCA_LB_GPUTPCCompressionKernels_step0attached 128
  #define GPUCA_LB_GPUTPCCompressionKernels_step1unattached 448
  #define GPUCA_LB_CLUSTER_FINDER 512
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
  #ifndef GPUCA_THREAD_COUNT
  #define GPUCA_THREAD_COUNT 256
  #endif
  #ifndef GPUCA_LB_GPUTPCTrackletConstructor
  #define GPUCA_LB_GPUTPCTrackletConstructor 256
  #endif
  #ifndef GPUCA_LB_GPUTPCTrackletSelector
  #define GPUCA_LB_GPUTPCTrackletSelector 256
  #endif
  #ifndef GPUCA_LB_GPUTPCNeighboursFinder
  #define GPUCA_LB_GPUTPCNeighboursFinder 256, 1
  #endif
  #ifndef GPUCA_LB_GPUTPCNeighboursCleaner
  #define GPUCA_LB_GPUTPCNeighboursCleaner 256
  #endif
  #ifndef GPUCA_LB_GPUTRDTrackerKernels
  #define GPUCA_LB_GPUTRDTrackerKernels 512
  #endif
  #ifndef GPUCA_LB_GPUTPCConvertKernel
  #define GPUCA_LB_GPUTPCConvertKernel 256
  #endif
  #ifndef GPUCA_LB_GPUTPCCompressionKernels_step0attached
  #define GPUCA_LB_GPUTPCCompressionKernels_step0attached 256
  #endif
  #ifndef GPUCA_LB_GPUTPCCompressionKernels_step1unattached
  #define GPUCA_LB_GPUTPCCompressionKernels_step1unattached 256
  #endif
  #ifndef GPUCA_LB_GPUTPCCFDecodeZS
  #define GPUCA_LB_GPUTPCCFDecodeZS 128, 4
  #endif
  #ifndef GPUCA_LB_CLUSTER_FINDER
  #define GPUCA_LB_CLUSTER_FINDER 128
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerTrackFit
  #define GPUCA_LB_GPUTPCGMMergerTrackFit 256, 1
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerFollowLoopers
  #define GPUCA_LB_GPUTPCGMMergerFollowLoopers 256, 1
  #endif
  #ifndef GPUCA_LB_GPUITSFitterKernel
  #define GPUCA_LB_GPUITSFitterKernel 256
  #endif
  #ifndef GPUCA_LB_GPUTPCStartHitsFinder
  #define GPUCA_LB_GPUTPCStartHitsFinder 256
  #endif
  #ifndef GPUCA_LB_GPUTPCStartHitsSorter
  #define GPUCA_LB_GPUTPCStartHitsSorter 256
  #endif
  #define GPUCA_GET_THREAD_COUNT(...) GPUCA_M_FIRST(__VA_ARGS__)
#else
  // The following defaults are needed to compile the host code
  #define GPUCA_GET_THREAD_COUNT(...) 1
#endif

#define GPUCA_THREAD_COUNT_SCAN 512 // TODO: WARNING!!! Must not be GPUTYPE-dependent right now! // TODO: Fix!

#define GPUCA_LB_GPUTPCCFChargeMapFiller GPUCA_LB_CLUSTER_FINDER
#define GPUCA_LB_GPUTPCCFChargeMapFiller GPUCA_LB_CLUSTER_FINDER
#define GPUCA_LB_GPUTPCCFChargeMapFiller GPUCA_LB_CLUSTER_FINDER
#define GPUCA_LB_GPUTPCCFPeakFinder GPUCA_LB_CLUSTER_FINDER
#define GPUCA_LB_GPUTPCCFNoiseSuppression GPUCA_LB_CLUSTER_FINDER
#define GPUCA_LB_GPUTPCCFNoiseSuppression GPUCA_LB_CLUSTER_FINDER
#define GPUCA_LB_GPUTPCCFDeconvolution GPUCA_LB_CLUSTER_FINDER
#define GPUCA_LB_GPUTPCCFClusterizer GPUCA_LB_CLUSTER_FINDER
#define GPUCA_LB_GPUTPCCFMCLabelFlattener GPUCA_LB_CLUSTER_FINDER
#define GPUCA_LB_GPUTPCCFMCLabelFlattener GPUCA_LB_CLUSTER_FINDER
#define GPUCA_LB_GPUTPCCFStreamCompaction_nativeScanUpStart GPUCA_THREAD_COUNT_SCAN
#define GPUCA_LB_GPUTPCCFStreamCompaction_nativeScanUp GPUCA_THREAD_COUNT_SCAN
#define GPUCA_LB_GPUTPCCFStreamCompaction_nativeScanTop GPUCA_THREAD_COUNT_SCAN
#define GPUCA_LB_GPUTPCCFStreamCompaction_nativeScanDown GPUCA_THREAD_COUNT_SCAN
#define GPUCA_LB_GPUTPCCFStreamCompaction_compact GPUCA_THREAD_COUNT_SCAN

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
  #ifdef GPUCA_GPUCODE
    static_assert(GPUCA_M_FIRST(GPUCA_LB_GPUTPCCompressionKernels_step1unattached) * 2 <= GPUCA_TPC_COMP_CHUNK_SIZE, "Invalid GPUCA_TPC_COMP_CHUNK_SIZE");
  #endif
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
#define GPUCA_ERROR_LOOPER_OVERFLOW 9
#define GPUCA_ERROR_STRINGS {"GPUCA_ERROR_NONE", "GPUCA_ERROR_ROWSTARTHIT_OVERFLOW", "GPUCA_ERROR_STARTHIT_OVERFLOW", "GPUCA_ERROR_TRACKLET_OVERFLOW", "GPUCA_ERROR_TRACKLET_HIT_OVERFLOW", "GPUCA_ERROR_TRACK_OVERFLOW", "GPUCA_ERROR_TRACK_HIT_OVERFLOW", "GPUCA_ERROR_GLOBAL_TRACKING_TRACK_OVERFLOW", "GPUCA_ERROR_GLOBAL_TRACKING_TRACK_HIT_OVERFLOW", "GPUCA_ERROR_LOOPER_OVERFLOW"}

// clang-format on
#endif
