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

#ifndef GPUDEFGPUPARAMETERS_H
#define GPUDEFGPUPARAMETERS_H
// clang-format off

#ifndef GPUDEF_H
#error Please include GPUDef.h
#endif

//GPU Run Configuration
#ifdef GPUCA_GPUTYPE_RADEON
  #define GPUCA_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER 2
  #define GPUCA_BLOCK_COUNT_SELECTOR_MULTIPLIER 3
  #define GPUCA_THREAD_COUNT 256
  #define GPUCA_THREAD_COUNT_CONSTRUCTOR 256
  #define GPUCA_THREAD_COUNT_SELECTOR 256
  #define GPUCA_THREAD_COUNT_FINDER 256
#elif defined(GPUCA_GPUTYPE_HIP)
  #define GPUCA_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER 2
  #define GPUCA_BLOCK_COUNT_SELECTOR_MULTIPLIER 4
  #define GPUCA_THREAD_COUNT 128
  #define GPUCA_THREAD_COUNT_CONSTRUCTOR 128
  #define GPUCA_THREAD_COUNT_SELECTOR 128
  #define GPUCA_THREAD_COUNT_FINDER 128
#elif defined(GPUCA_GPUTYPE_PASCAL)
  #define GPUCA_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER 2
  #define GPUCA_BLOCK_COUNT_SELECTOR_MULTIPLIER 4
  #define GPUCA_THREAD_COUNT 256
  #define GPUCA_THREAD_COUNT_CONSTRUCTOR 1024
  #define GPUCA_THREAD_COUNT_SELECTOR 512
  #define GPUCA_THREAD_COUNT_FINDER 512
  //#define GPUCA_USE_TEXTURES
#elif defined(GPUCA_GPUTYPE_KEPLER)
  #define GPUCA_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER 4
  #define GPUCA_BLOCK_COUNT_SELECTOR_MULTIPLIER 3
  #define GPUCA_THREAD_COUNT 256
  #define GPUCA_THREAD_COUNT_CONSTRUCTOR 512
  #define GPUCA_THREAD_COUNT_SELECTOR 256
  #define GPUCA_THREAD_COUNT_FINDER 256
#elif defined(GPUCA_GPUTYPE_FERMI) || defined(__OPENCL__)
  #define GPUCA_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER 2
  #define GPUCA_BLOCK_COUNT_SELECTOR_MULTIPLIER 3
  #define GPUCA_THREAD_COUNT 256
  #define GPUCA_THREAD_COUNT_CONSTRUCTOR 256
  #define GPUCA_THREAD_COUNT_SELECTOR 256
  #define GPUCA_THREAD_COUNT_FINDER 256
#elif defined(GPUCA_GPUCODE)
  #error GPU TYPE NOT SET
#endif
#define GPUCA_THREAD_COUNT_TRD 512

#define GPUCA_MAX_STREAMS 32

#define GPUCA_ROWALIGNMENT uint4                                       //Align Row Hits and Grid

#ifdef GPUCA_USE_TEXTURES
  #define GPUCA_TEXTURE_FETCH_CONSTRUCTOR                              //Fetch data through texture cache
  #define GPUCA_TEXTURE_FETCH_NEIGHBORS                                //Fetch also in Neighbours Finder
#endif

//#define GPUCA_TRACKLET_CONSTRUCTOR_DO_PROFILE                        //Output Profiling Data for Tracklet Constructor Tracklet Scheduling

#define GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE 12
#define GPUCA_TRACKLET_SELECTOR_SLICE_COUNT 8                          //Currently must be smaller than avaiable MultiProcessors on GPU or will result in wrong results

#define GPUCA_MAX_CLUSTERS (1024 * 1024 * 1024)                        //Maximum number of TPC clusters
#define GPUCA_MAX_TRACKLETS 65536                                      //Max Number of Tracklets that can be processed by GPU Tracker, Should be divisible by 16 at least
#define GPUCA_MAX_TRACKS 32768                                         //Max number of Tracks that can be processd by GPU Tracker per sector, must be below 2^24 for track ID format!!!
#define GPUCA_MAX_ROWSTARTHITS 32768                                   //Maximum number of start hits per row

#define GPUCA_TRACKER_CONSTANT_MEM 65000                               //Amount of Constant Memory to reserve

#define GPUCA_MEMALIGN               ((size_t)           64 * 1024)    //Alignment of memory blocks, all constants above must be multiple of this!!!
#define GPUCA_MEMALIGN_SMALL         ((size_t)           64 * 1024)    //Alignment of small blocks, GPUCA_MEMALIGN must be multiple of this!!!
#define GPUCA_MEMORY_SIZE            ((size_t)  4096 * 1024 * 1024)    //Size of memory allocated on Device
#define GPUCA_HOST_MEMORY_SIZE       ((size_t)  4096 * 1024 * 1024)    //Size of memory allocated on Host

#define GPUCA_GPU_STACK_SIZE 8192                                      //Stack size per GPU thread

//Make sure options do not interfere

#ifndef GPUCA_GPUCODE
  //No texture fetch for CPU Tracker
  #ifdef GPUCA_TEXTURE_FETCH_CONSTRUCTOR
    #undef GPUCA_TEXTURE_FETCH_CONSTRUCTOR
  #endif
  #ifdef GPUCA_TEXTURE_FETCH_NEIGHBORS
    #undef GPUCA_TEXTURE_FETCH_NEIGHBORS
  #endif

  //Do not cache Row Hits during Tracklet selection in Registers for CPU Tracker
  #undef GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE
  #define GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE 0

  #define GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP 0
#else
  #define GPUCA_SORT_STARTHITS // Sort start hits for GPU tracker
  #define GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP 6
#endif

#ifdef GPUCA_NOCOMPAT
  static_assert(GPUCA_MAXN > GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP);
#endif

//Error Codes for GPU Tracker
#define GPUCA_ERROR_NONE 0
#define GPUCA_ERROR_ROWSTARTHIT_OVERFLOW 1
#define GPUCA_ERROR_TRACKLET_OVERFLOW 2
#define GPUCA_ERROR_TRACK_OVERFLOW 3
#define GPUCA_ERROR_TRACK_HIT_OVERFLOW 4
#define GPUCA_ERROR_GLOBAL_TRACKING_TRACK_OVERFLOW 5
#define GPUCA_ERROR_GLOBAL_TRACKING_TRACK_HIT_OVERFLOW 6
#define GPUCA_ERROR_STRINGS {"GPUCA_ERROR_NONE", "GPUCA_ERROR_ROWSTARTHIT_OVERFLOW", "GPUCA_ERROR_TRACKLET_OVERFLOW", "GPUCA_ERROR_TRACK_OVERFLOW", "GPUCA_ERROR_TRACK_HIT_OVERFLOW", "GPUCA_ERROR_GLOBAL_TRACKING_TRACK_OVERFLOW", "GPUCA_ERROR_GLOBAL_TRACKING_TRACK_HIT_OVERFLOW"}

// clang-format on
#endif
