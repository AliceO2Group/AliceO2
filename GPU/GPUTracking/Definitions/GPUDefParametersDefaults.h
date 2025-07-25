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

/// \file GPUDefParametersDefaults.h
/// \author David Rohr

// This file contains compile-time constants affecting the GPU performance.

#if !defined(GPUDEFPARAMETERSDEFAULTS_H)
#define GPUDEFPARAMETERSDEFAULTS_H
// clang-format off

// Launch bound definition, 3 optional parameters: maxThreads per block, minBlocks per multiprocessor, force number of blocks (not passed to compiler as launch bounds)

// GPU Run Configuration
#if defined(GPUCA_GPUCODE) && !defined(GPUCA_GPUCODE_GENRTC) && !defined(GPUCA_GPUCODE_NO_LAUNCH_BOUNDS) // Avoid including for RTC generation besides normal include protection.
  // GPU-architecture-dependent default settings
  #if defined(GPUCA_GPUTYPE_MI2xx)
    #define GPUCA_WARP_SIZE 64
    #define GPUCA_THREAD_COUNT_DEFAULT 256
    #define GPUCA_LB_GPUTPCCreateTrackingData 256
    #define GPUCA_LB_GPUTPCStartHitsSorter 512, 1
    #define GPUCA_LB_GPUTPCStartHitsFinder 1024
    #define GPUCA_LB_GPUTPCTrackletConstructor 512, 2
    #define GPUCA_LB_GPUTPCTrackletSelector 192, 3
    #define GPUCA_LB_GPUTPCNeighboursFinder 1024, 1
    #define GPUCA_LB_GPUTPCNeighboursCleaner 896
    #define GPUCA_LB_GPUTPCExtrapolationTracking 256
    #define GPUCA_LB_GPUTPCCFDecodeZS 64, 4
    #define GPUCA_LB_GPUTPCCFDecodeZSLink GPUCA_WARP_SIZE
    #define GPUCA_LB_GPUTPCCFDecodeZSDenseLink GPUCA_WARP_SIZE
    #define GPUCA_LB_GPUTPCCFGather 1024, 1
    #define GPUCA_LB_GPUTPCGMMergerTrackFit 128, 1
    #define GPUCA_LB_GPUTPCGMMergerFollowLoopers 64, 12
    #define GPUCA_LB_GPUTPCGMMergerSectorRefit 256
    #define GPUCA_LB_GPUTPCGMMergerUnpackResetIds 256
    #define GPUCA_LB_GPUTPCGMMergerUnpackGlobal 256
    #define GPUCA_LB_GPUTPCGMMergerResolve_step0 512
    #define GPUCA_LB_GPUTPCGMMergerResolve_step1 512
    #define GPUCA_LB_GPUTPCGMMergerResolve_step2 512
    #define GPUCA_LB_GPUTPCGMMergerResolve_step3 512
    #define GPUCA_LB_GPUTPCGMMergerResolve_step4 512
    #define GPUCA_LB_GPUTPCGMMergerClearLinks 256
    #define GPUCA_LB_GPUTPCGMMergerMergeWithinPrepare 256
    #define GPUCA_LB_GPUTPCGMMergerMergeSectorsPrepare 256
    #define GPUCA_LB_GPUTPCGMMergerMergeBorders_step0 512
    #define GPUCA_LB_GPUTPCGMMergerMergeBorders_step2 512
    #define GPUCA_LB_GPUTPCGMMergerMergeCE 512
    #define GPUCA_LB_GPUTPCGMMergerLinkExtrapolatedTracks 256
    #define GPUCA_LB_GPUTPCGMMergerCollect 512
    #define GPUCA_LB_GPUTPCGMMergerSortTracksPrepare 256
    #define GPUCA_LB_GPUTPCGMMergerPrepareClusters_step0 256
    #define GPUCA_LB_GPUTPCGMMergerPrepareClusters_step1 256
    #define GPUCA_LB_GPUTPCGMMergerPrepareClusters_step2 256
    #define GPUCA_LB_GPUTPCGMMergerFinalize_0 256
    #define GPUCA_LB_GPUTPCGMMergerFinalize_1 256
    #define GPUCA_LB_GPUTPCGMMergerFinalize_2 256
    #define GPUCA_LB_GPUTPCCompressionKernels_step0attached 64, 2
    #define GPUCA_LB_GPUTPCCompressionKernels_step1unattached 512, 2
    #define GPUCA_LB_GPUTPCDecompressionKernels_step0attached 128, 2
    #define GPUCA_LB_GPUTPCDecompressionKernels_step1unattached 64, 2
    #define GPUCA_LB_GPUTPCCFCheckPadBaseline 64
    #define GPUCA_LB_GPUTPCCFChargeMapFiller_fillIndexMap 512
    #define GPUCA_LB_GPUTPCCFChargeMapFiller_fillFromDigits 512
    #define GPUCA_LB_GPUTPCCFChargeMapFiller_findFragmentStart 512
    #define GPUCA_LB_GPUTPCCFPeakFinder 512
    #define GPUCA_LB_GPUTPCCFNoiseSuppression 512
    #define GPUCA_LB_GPUTPCCFDeconvolution 512
    #define GPUCA_LB_GPUTPCCFClusterizer 448
    #define GPUCA_LB_COMPRESSION_GATHER 1024
    #define GPUCA_PAR_NEIGHBOURS_FINDER_MAX_NNEIGHUP 5
    #define GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE 20
    #define GPUCA_PAR_ALTERNATE_BORDER_SORT 1
    #define GPUCA_PAR_SORT_BEFORE_FIT 1
    #define GPUCA_PAR_NO_ATOMIC_PRECHECK 1
    #define GPUCA_PAR_DEDX_STORAGE_TYPE uint16_t
    #define GPUCA_PAR_MERGER_INTERPOLATION_ERROR_TYPE half
    #define GPUCA_PAR_COMP_GATHER_KERNEL 4
    #define GPUCA_PAR_COMP_GATHER_MODE 3
  #elif defined(GPUCA_GPUTYPE_VEGA)
    #define GPUCA_WARP_SIZE 64
    #define GPUCA_THREAD_COUNT_DEFAULT 256
    #define GPUCA_LB_GPUTPCCreateTrackingData 128
    #define GPUCA_LB_GPUTPCStartHitsSorter 1024, 2
    #define GPUCA_LB_GPUTPCStartHitsFinder 1024
    #define GPUCA_LB_GPUTPCTrackletConstructor 256, 2
    #define GPUCA_LB_GPUTPCTrackletSelector 256, 8
    #define GPUCA_LB_GPUTPCNeighboursFinder 1024, 1
    #define GPUCA_LB_GPUTPCNeighboursCleaner 896
    #define GPUCA_LB_GPUTPCExtrapolationTracking 256
    #define GPUCA_LB_GPUTPCCFDecodeZS 64, 4
    #define GPUCA_LB_GPUTPCCFDecodeZSLink GPUCA_WARP_SIZE
    #define GPUCA_LB_GPUTPCCFDecodeZSDenseLink GPUCA_WARP_SIZE
    #define GPUCA_LB_GPUTPCCFGather 1024, 1
    #define GPUCA_LB_GPUTPCGMMergerTrackFit 64, 1
    #define GPUCA_LB_GPUTPCGMMergerFollowLoopers 256, 4, 200
    #define GPUCA_LB_GPUTPCGMMergerSectorRefit 256
    #define GPUCA_LB_GPUTPCGMMergerUnpackResetIds 256
    #define GPUCA_LB_GPUTPCGMMergerUnpackGlobal 256
    #define GPUCA_LB_GPUTPCGMMergerResolve_step0 256
    #define GPUCA_LB_GPUTPCGMMergerResolve_step1 256
    #define GPUCA_LB_GPUTPCGMMergerResolve_step2 256
    #define GPUCA_LB_GPUTPCGMMergerResolve_step3 256
    #define GPUCA_LB_GPUTPCGMMergerResolve_step4 256
    #define GPUCA_LB_GPUTPCGMMergerClearLinks 256
    #define GPUCA_LB_GPUTPCGMMergerMergeWithinPrepare 256
    #define GPUCA_LB_GPUTPCGMMergerMergeSectorsPrepare 256
    #define GPUCA_LB_GPUTPCGMMergerMergeBorders_step0 256
    #define GPUCA_LB_GPUTPCGMMergerMergeBorders_step2 256
    #define GPUCA_LB_GPUTPCGMMergerMergeCE 256
    #define GPUCA_LB_GPUTPCGMMergerLinkExtrapolatedTracks 256
    #define GPUCA_LB_GPUTPCGMMergerCollect 512
    #define GPUCA_LB_GPUTPCGMMergerSortTracksPrepare 256
    #define GPUCA_LB_GPUTPCGMMergerPrepareClusters_step0 256
    #define GPUCA_LB_GPUTPCGMMergerPrepareClusters_step1 256
    #define GPUCA_LB_GPUTPCGMMergerPrepareClusters_step2 256
    #define GPUCA_LB_GPUTPCGMMergerFinalize_0 256
    #define GPUCA_LB_GPUTPCGMMergerFinalize_1 256
    #define GPUCA_LB_GPUTPCGMMergerFinalize_2 256
    #define GPUCA_LB_GPUTPCCompressionKernels_step0attached 192, 2
    #define GPUCA_LB_GPUTPCCompressionKernels_step1unattached 512, 2
    #define GPUCA_LB_GPUTPCDecompressionKernels_step0attached 128, 2
    #define GPUCA_LB_GPUTPCDecompressionKernels_step1unattached 64, 2
    #define GPUCA_LB_GPUTPCCFCheckPadBaseline 64
    #define GPUCA_LB_GPUTPCCFChargeMapFiller_fillIndexMap 512
    #define GPUCA_LB_GPUTPCCFChargeMapFiller_fillFromDigits 512
    #define GPUCA_LB_GPUTPCCFChargeMapFiller_findFragmentStart 512
    #define GPUCA_LB_GPUTPCCFPeakFinder 512
    #define GPUCA_LB_GPUTPCCFNoiseSuppression 512
    #define GPUCA_LB_GPUTPCCFDeconvolution 512
    #define GPUCA_LB_GPUTPCCFClusterizer 512
    #define GPUCA_LB_COMPRESSION_GATHER 1024
    #define GPUCA_PAR_NEIGHBOURS_FINDER_MAX_NNEIGHUP 5
    #define GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE 20
    #define GPUCA_PAR_ALTERNATE_BORDER_SORT 1
    #define GPUCA_PAR_SORT_BEFORE_FIT 1
    #define GPUCA_PAR_NO_ATOMIC_PRECHECK 1
    #define GPUCA_PAR_DEDX_STORAGE_TYPE uint16_t
    #define GPUCA_PAR_MERGER_INTERPOLATION_ERROR_TYPE half
    #define GPUCA_PAR_COMP_GATHER_KERNEL 4
    #define GPUCA_PAR_COMP_GATHER_MODE 3
  #elif defined(GPUCA_GPUTYPE_AMPERE)
    #define GPUCA_WARP_SIZE 32
    #define GPUCA_THREAD_COUNT_DEFAULT 512
    #define GPUCA_LB_GPUTPCCreateTrackingData 384
    #define GPUCA_LB_GPUTPCStartHitsSorter 512, 1
    #define GPUCA_LB_GPUTPCStartHitsFinder 512
    #define GPUCA_LB_GPUTPCTrackletConstructor 256, 2 // best single-kernel: 128, 4
    #define GPUCA_LB_GPUTPCTrackletSelector 192, 3    // best single-kernel: 128, 4
    #define GPUCA_LB_GPUTPCNeighboursFinder 640, 1    // best single-kernel: 768, 1
    #define GPUCA_LB_GPUTPCNeighboursCleaner 512
    #define GPUCA_LB_GPUTPCExtrapolationTracking 128, 4
    #define GPUCA_LB_GPUTPCCFDecodeZS 64, 10
    #define GPUCA_LB_GPUTPCCFDecodeZSLink GPUCA_WARP_SIZE
    #define GPUCA_LB_GPUTPCCFDecodeZSDenseLink GPUCA_WARP_SIZE
    #define GPUCA_LB_GPUTPCCFGather 1024, 1
    #define GPUCA_LB_GPUTPCGMMergerTrackFit 64, 4
    #define GPUCA_LB_GPUTPCGMMergerFollowLoopers 64, 12
    #define GPUCA_LB_GPUTPCGMMergerSectorRefit 32, 6
    #define GPUCA_LB_GPUTPCGMMergerUnpackResetIds 256
    #define GPUCA_LB_GPUTPCGMMergerUnpackGlobal 256
    #define GPUCA_LB_GPUTPCGMMergerResolve_step0 256
    #define GPUCA_LB_GPUTPCGMMergerResolve_step1 256
    #define GPUCA_LB_GPUTPCGMMergerResolve_step2 256
    #define GPUCA_LB_GPUTPCGMMergerResolve_step3 256
    #define GPUCA_LB_GPUTPCGMMergerResolve_step4 256, 4
    #define GPUCA_LB_GPUTPCGMMergerClearLinks 256
    #define GPUCA_LB_GPUTPCGMMergerMergeWithinPrepare 256
    #define GPUCA_LB_GPUTPCGMMergerMergeSectorsPrepare 256, 2
    #define GPUCA_LB_GPUTPCGMMergerMergeBorders_step0 192
    #define GPUCA_LB_GPUTPCGMMergerMergeBorders_step2 64, 2
    #define GPUCA_LB_GPUTPCGMMergerMergeCE 256
    #define GPUCA_LB_GPUTPCGMMergerLinkExtrapolatedTracks 256
    #define GPUCA_LB_GPUTPCGMMergerCollect 256, 2
    #define GPUCA_LB_GPUTPCGMMergerSortTracksPrepare 256
    #define GPUCA_LB_GPUTPCGMMergerPrepareClusters_step0 256
    #define GPUCA_LB_GPUTPCGMMergerPrepareClusters_step1 256
    #define GPUCA_LB_GPUTPCGMMergerPrepareClusters_step2 256
    #define GPUCA_LB_GPUTPCGMMergerFinalize_0 256
    #define GPUCA_LB_GPUTPCGMMergerFinalize_1 256
    #define GPUCA_LB_GPUTPCGMMergerFinalize_2 256
    #define GPUCA_LB_GPUTPCCompressionKernels_step0attached 64, 2
    #define GPUCA_LB_GPUTPCCompressionKernels_step1unattached 512, 3
    #define GPUCA_LB_GPUTPCDecompressionKernels_step0attached 32, 1
    #define GPUCA_LB_GPUTPCDecompressionKernels_step1unattached 32, 1
    #define GPUCA_LB_GPUTPCCFCheckPadBaseline 64,8
    #define GPUCA_LB_GPUTPCCFChargeMapFiller_fillIndexMap 448
    #define GPUCA_LB_GPUTPCCFChargeMapFiller_fillFromDigits 448
    #define GPUCA_LB_GPUTPCCFChargeMapFiller_findFragmentStart 448
    #define GPUCA_LB_GPUTPCCFPeakFinder 128
    #define GPUCA_LB_GPUTPCCFNoiseSuppression 448
    #define GPUCA_LB_GPUTPCCFDeconvolution 384
    #define GPUCA_LB_GPUTPCCFClusterizer 448
    #define GPUCA_LB_COMPRESSION_GATHER 1024
    #define GPUCA_PAR_NEIGHBOURS_FINDER_MAX_NNEIGHUP 4
    #define GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE 20
    #define GPUCA_PAR_ALTERNATE_BORDER_SORT 1
    #define GPUCA_PAR_SORT_BEFORE_FIT 1
    #define GPUCA_PAR_NO_ATOMIC_PRECHECK 1
    #define GPUCA_PAR_DEDX_STORAGE_TYPE uint16_t
    #define GPUCA_PAR_MERGER_INTERPOLATION_ERROR_TYPE half
    #define GPUCA_PAR_COMP_GATHER_KERNEL 4
    #define GPUCA_PAR_COMP_GATHER_MODE 3
  #elif defined(GPUCA_GPUTYPE_TURING)
    #define GPUCA_WARP_SIZE 32
    #define GPUCA_THREAD_COUNT_DEFAULT 512
    #define GPUCA_LB_GPUTPCCreateTrackingData 256
    #define GPUCA_LB_GPUTPCStartHitsSorter 512, 1
    #define GPUCA_LB_GPUTPCStartHitsFinder 512
    #define GPUCA_LB_GPUTPCTrackletConstructor 256, 2
    #define GPUCA_LB_GPUTPCTrackletSelector 192, 3
    #define GPUCA_LB_GPUTPCNeighboursFinder 640, 1
    #define GPUCA_LB_GPUTPCNeighboursCleaner 512
    #define GPUCA_LB_GPUTPCExtrapolationTracking 192, 2
    #define GPUCA_LB_GPUTPCCFDecodeZS 64, 8
    #define GPUCA_LB_GPUTPCCFDecodeZSLink GPUCA_WARP_SIZE
    #define GPUCA_LB_GPUTPCCFDecodeZSDenseLink GPUCA_WARP_SIZE
    #define GPUCA_LB_GPUTPCCFGather 1024, 1
    #define GPUCA_LB_GPUTPCGMMergerTrackFit 32, 8
    #define GPUCA_LB_GPUTPCGMMergerFollowLoopers 128, 4
    #define GPUCA_LB_GPUTPCGMMergerSectorRefit 64, 5
    #define GPUCA_LB_GPUTPCGMMergerUnpackResetIds 256
    #define GPUCA_LB_GPUTPCGMMergerUnpackGlobal 256
    #define GPUCA_LB_GPUTPCGMMergerResolve_step0 256
    #define GPUCA_LB_GPUTPCGMMergerResolve_step1 256
    #define GPUCA_LB_GPUTPCGMMergerResolve_step2 256
    #define GPUCA_LB_GPUTPCGMMergerResolve_step3 256
    #define GPUCA_LB_GPUTPCGMMergerResolve_step4 256, 4
    #define GPUCA_LB_GPUTPCGMMergerClearLinks 256
    #define GPUCA_LB_GPUTPCGMMergerMergeWithinPrepare 256
    #define GPUCA_LB_GPUTPCGMMergerMergeSectorsPrepare 256, 2
    #define GPUCA_LB_GPUTPCGMMergerMergeBorders_step0 192
    #define GPUCA_LB_GPUTPCGMMergerMergeBorders_step2 256
    #define GPUCA_LB_GPUTPCGMMergerMergeCE 256
    #define GPUCA_LB_GPUTPCGMMergerLinkExtrapolatedTracks 256
    #define GPUCA_LB_GPUTPCGMMergerCollect 128, 2
    #define GPUCA_LB_GPUTPCGMMergerSortTracksPrepare 256
    #define GPUCA_LB_GPUTPCGMMergerPrepareClusters_step0 256
    #define GPUCA_LB_GPUTPCGMMergerPrepareClusters_step1 256
    #define GPUCA_LB_GPUTPCGMMergerPrepareClusters_step2 256
    #define GPUCA_LB_GPUTPCGMMergerFinalize_0 256
    #define GPUCA_LB_GPUTPCGMMergerFinalize_1 256
    #define GPUCA_LB_GPUTPCGMMergerFinalize_2 256
    #define GPUCA_LB_GPUTPCCompressionKernels_step0attached 128
    #define GPUCA_LB_GPUTPCCompressionKernels_step1unattached 512, 2
    #define GPUCA_LB_GPUTPCDecompressionKernels_step0attached 32, 1
    #define GPUCA_LB_GPUTPCDecompressionKernels_step1unattached 32, 1
    #define GPUCA_LB_COMPRESSION_GATHER 1024
    #define GPUCA_PAR_NEIGHBOURS_FINDER_MAX_NNEIGHUP 4
    #define GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE 20
    #define GPUCA_PAR_ALTERNATE_BORDER_SORT 1
    #define GPUCA_PAR_SORT_BEFORE_FIT 1
    #define GPUCA_PAR_NO_ATOMIC_PRECHECK 1
    #define GPUCA_PAR_COMP_GATHER_KERNEL 4
    #define GPUCA_PAR_COMP_GATHER_MODE 3
    #define GPUCA_PAR_DEDX_STORAGE_TYPE uint16_t
    #define GPUCA_PAR_MERGER_INTERPOLATION_ERROR_TYPE half
  #elif defined(GPUCA_GPUTYPE_OPENCL)
  #else
    #error GPU TYPE NOT SET
  #endif

  // Default settings for GPU, if not already set for selected GPU type
  #ifndef GPUCA_WARP_SIZE
    #define GPUCA_WARP_SIZE 32
  #endif
  #ifndef GPUCA_THREAD_COUNT_DEFAULT
    #define GPUCA_THREAD_COUNT_DEFAULT 256
  #endif
  #ifndef GPUCA_LB_GPUTPCCreateTrackingData
    #define GPUCA_LB_GPUTPCCreateTrackingData 256
  #endif
  #ifndef GPUCA_LB_GPUTPCTrackletConstructor
    #define GPUCA_LB_GPUTPCTrackletConstructor 256
  #endif
  #ifndef GPUCA_LB_GPUTPCTrackletSelector
    #define GPUCA_LB_GPUTPCTrackletSelector 256
  #endif
  #ifndef GPUCA_LB_GPUTPCNeighboursFinder
    #define GPUCA_LB_GPUTPCNeighboursFinder 256
  #endif
  #ifndef GPUCA_LB_GPUTPCNeighboursCleaner
    #define GPUCA_LB_GPUTPCNeighboursCleaner 256
  #endif
  #ifndef GPUCA_LB_GPUTPCExtrapolationTracking
    #define GPUCA_LB_GPUTPCExtrapolationTracking 256
  #endif
  #ifndef GPUCA_LB_GPUTRDTrackerKernels_gpuVersion
    #define GPUCA_LB_GPUTRDTrackerKernels_gpuVersion 512
  #endif
  #ifndef GPUCA_LB_GPUTPCCreateOccupancyMap_fill
    #define GPUCA_LB_GPUTPCCreateOccupancyMap_fill 256
  #endif
  #ifndef GPUCA_LB_GPUTPCCreateOccupancyMap_fold
    #define GPUCA_LB_GPUTPCCreateOccupancyMap_fold 256
  #endif
  #ifndef GPUCA_LB_GPUTRDTrackerKernels_o2Version
    #define GPUCA_LB_GPUTRDTrackerKernels_o2Version 512
  #endif
  #ifndef GPUCA_LB_GPUTPCCompressionKernels_step0attached
    #define GPUCA_LB_GPUTPCCompressionKernels_step0attached 256
  #endif
  #ifndef GPUCA_LB_GPUTPCCompressionKernels_step1unattached
    #define GPUCA_LB_GPUTPCCompressionKernels_step1unattached 256
  #endif
  #ifndef GPUCA_LB_GPUTPCDecompressionKernels_step0attached
    #define GPUCA_LB_GPUTPCDecompressionKernels_step0attached 256
  #endif
  #ifndef GPUCA_LB_GPUTPCDecompressionKernels_step1unattached
    #define GPUCA_LB_GPUTPCDecompressionKernels_step1unattached 256
  #endif
  #ifndef GPUCA_LB_GPUTPCDecompressionUtilKernels_sortPerSectorRow
    #define GPUCA_LB_GPUTPCDecompressionUtilKernels_sortPerSectorRow 256
  #endif
  #ifndef GPUCA_LB_GPUTPCDecompressionUtilKernels_countFilteredClusters
    #define GPUCA_LB_GPUTPCDecompressionUtilKernels_countFilteredClusters 256
  #endif
  #ifndef GPUCA_LB_GPUTPCDecompressionUtilKernels_storeFilteredClusters
    #define GPUCA_LB_GPUTPCDecompressionUtilKernels_storeFilteredClusters 256
  #endif
  #ifndef GPUCA_LB_GPUTPCCFDecodeZS
    #define GPUCA_LB_GPUTPCCFDecodeZS 128, 4
  #endif
  #ifndef GPUCA_LB_GPUTPCCFDecodeZSLink
    #define GPUCA_LB_GPUTPCCFDecodeZSLink GPUCA_WARP_SIZE
  #endif
  #ifndef GPUCA_LB_GPUTPCCFDecodeZSDenseLink
    #define GPUCA_LB_GPUTPCCFDecodeZSDenseLink GPUCA_WARP_SIZE
  #endif
  #ifndef GPUCA_LB_GPUTPCCFGather
    #define GPUCA_LB_GPUTPCCFGather 1024, 1
  #endif
  #ifndef GPUCA_LB_COMPRESSION_GATHER
    #define GPUCA_LB_COMPRESSION_GATHER 1024
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerTrackFit
    #define GPUCA_LB_GPUTPCGMMergerTrackFit 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerFollowLoopers
    #define GPUCA_LB_GPUTPCGMMergerFollowLoopers 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerSectorRefit
    #define GPUCA_LB_GPUTPCGMMergerSectorRefit 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerUnpackResetIds
    #define GPUCA_LB_GPUTPCGMMergerUnpackResetIds 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerUnpackGlobal
    #define GPUCA_LB_GPUTPCGMMergerUnpackGlobal 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerResolve_step0
    #define GPUCA_LB_GPUTPCGMMergerResolve_step0 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerResolve_step1
    #define GPUCA_LB_GPUTPCGMMergerResolve_step1 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerResolve_step2
    #define GPUCA_LB_GPUTPCGMMergerResolve_step2 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerResolve_step3
    #define GPUCA_LB_GPUTPCGMMergerResolve_step3 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerResolve_step4
    #define GPUCA_LB_GPUTPCGMMergerResolve_step4 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerClearLinks
    #define GPUCA_LB_GPUTPCGMMergerClearLinks 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerMergeWithinPrepare
    #define GPUCA_LB_GPUTPCGMMergerMergeWithinPrepare 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerMergeSectorsPrepare
    #define GPUCA_LB_GPUTPCGMMergerMergeSectorsPrepare 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerMergeBorders_step0
    #define GPUCA_LB_GPUTPCGMMergerMergeBorders_step0 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerMergeBorders_step2
    #define GPUCA_LB_GPUTPCGMMergerMergeBorders_step2 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerMergeCE
    #define GPUCA_LB_GPUTPCGMMergerMergeCE 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerLinkExtrapolatedTracks
    #define GPUCA_LB_GPUTPCGMMergerLinkExtrapolatedTracks 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerCollect
    #define GPUCA_LB_GPUTPCGMMergerCollect 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerSortTracksPrepare
    #define GPUCA_LB_GPUTPCGMMergerSortTracksPrepare 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerPrepareClusters_step0
    #define GPUCA_LB_GPUTPCGMMergerPrepareClusters_step0 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerPrepareClusters_step1
    #define GPUCA_LB_GPUTPCGMMergerPrepareClusters_step1 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerPrepareClusters_step2
    #define GPUCA_LB_GPUTPCGMMergerPrepareClusters_step2 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerFinalize_step0
    #define GPUCA_LB_GPUTPCGMMergerFinalize_step0 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerFinalize_step1
    #define GPUCA_LB_GPUTPCGMMergerFinalize_step1 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerFinalize_step2
    #define GPUCA_LB_GPUTPCGMMergerFinalize_step2 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerMergeLoopers_step0
    #define GPUCA_LB_GPUTPCGMMergerMergeLoopers_step0 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerMergeLoopers_step1
    #define GPUCA_LB_GPUTPCGMMergerMergeLoopers_step1 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMMergerMergeLoopers_step2
    #define GPUCA_LB_GPUTPCGMMergerMergeLoopers_step2 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMO2Output_prepare
    #define GPUCA_LB_GPUTPCGMO2Output_prepare 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMO2Output_output
    #define GPUCA_LB_GPUTPCGMO2Output_output 256
  #endif
  #ifndef GPUCA_LB_GPUTPCStartHitsFinder
    #define GPUCA_LB_GPUTPCStartHitsFinder 256
  #endif
  #ifndef GPUCA_LB_GPUTPCStartHitsSorter
    #define GPUCA_LB_GPUTPCStartHitsSorter 256
  #endif
  #ifndef GPUCA_LB_GPUTPCCFCheckPadBaseline
    #define GPUCA_LB_GPUTPCCFCheckPadBaseline 64
  #endif
  #ifndef GPUCA_LB_GPUTPCCFChargeMapFiller_fillIndexMap
    #define GPUCA_LB_GPUTPCCFChargeMapFiller_fillIndexMap 512
  #endif
  #ifndef GPUCA_LB_GPUTPCCFChargeMapFiller_fillFromDigits
    #define GPUCA_LB_GPUTPCCFChargeMapFiller_fillFromDigits 512
  #endif
  #ifndef GPUCA_LB_GPUTPCCFChargeMapFiller_findFragmentStart
    #define GPUCA_LB_GPUTPCCFChargeMapFiller_findFragmentStart 512
  #endif
  #ifndef GPUCA_LB_GPUTPCCFPeakFinder
    #define GPUCA_LB_GPUTPCCFPeakFinder 512
  #endif
  #ifndef GPUCA_LB_GPUTPCCFNoiseSuppression
    #define GPUCA_LB_GPUTPCCFNoiseSuppression 512
  #endif
  #ifndef GPUCA_LB_GPUTPCCFDeconvolution
    #define GPUCA_LB_GPUTPCCFDeconvolution 512
  #endif
  #ifndef GPUCA_LB_GPUTPCCFClusterizer
    #define GPUCA_LB_GPUTPCCFClusterizer 512
  #endif
  #ifndef GPUCA_LB_GPUTPCNNClusterizerKernels
    #define GPUCA_LB_GPUTPCNNClusterizerKernels 512
  #endif
  #ifndef GPUCA_LB_GPUTrackingRefitKernel_mode0asGPU
    #define GPUCA_LB_GPUTrackingRefitKernel_mode0asGPU 256
  #endif
  #ifndef GPUCA_LB_GPUTrackingRefitKernel_mode1asTrackParCov
    #define GPUCA_LB_GPUTrackingRefitKernel_mode1asTrackParCov 256
  #endif
  #ifndef GPUCA_LB_GPUMemClean16
    #define GPUCA_LB_GPUMemClean16 GPUCA_THREAD_COUNT_DEFAULT, 1
  #endif
  #ifndef GPUCA_LB_GPUitoa
    #define GPUCA_LB_GPUitoa GPUCA_THREAD_COUNT_DEFAULT, 1
  #endif
  // These kernel launch-bounds are derrived from one of the constants set above
  #define GPUCA_LB_GPUTPCCFNoiseSuppression_noiseSuppression GPUCA_LB_GPUTPCCFNoiseSuppression
  #define GPUCA_LB_GPUTPCCFNoiseSuppression_updatePeaks GPUCA_LB_GPUTPCCFNoiseSuppression

  #define GPUCA_LB_GPUTPCNNClusterizerKernels_runCfClusterizer GPUCA_LB_GPUTPCNNClusterizerKernels
  #define GPUCA_LB_GPUTPCNNClusterizerKernels_fillInputNNCPU GPUCA_LB_GPUTPCNNClusterizerKernels
  #define GPUCA_LB_GPUTPCNNClusterizerKernels_fillInputNNGPU GPUCA_LB_GPUTPCNNClusterizerKernels
  #define GPUCA_LB_GPUTPCNNClusterizerKernels_determineClass1Labels GPUCA_LB_GPUTPCNNClusterizerKernels
  #define GPUCA_LB_GPUTPCNNClusterizerKernels_determineClass2Labels GPUCA_LB_GPUTPCNNClusterizerKernels
  #define GPUCA_LB_GPUTPCNNClusterizerKernels_publishClass1Regression GPUCA_LB_GPUTPCNNClusterizerKernels
  #define GPUCA_LB_GPUTPCNNClusterizerKernels_publishClass2Regression GPUCA_LB_GPUTPCNNClusterizerKernels
  #define GPUCA_LB_GPUTPCNNClusterizerKernels_publishDeconvolutionFlags GPUCA_LB_GPUTPCNNClusterizerKernels

  #define GPUCA_LB_GPUTPCCFStreamCompaction_scanStart GPUCA_PAR_CF_SCAN_WORKGROUP_SIZE
  #define GPUCA_LB_GPUTPCCFStreamCompaction_scanUp GPUCA_PAR_CF_SCAN_WORKGROUP_SIZE
  #define GPUCA_LB_GPUTPCCFStreamCompaction_scanTop GPUCA_PAR_CF_SCAN_WORKGROUP_SIZE
  #define GPUCA_LB_GPUTPCCFStreamCompaction_scanDown GPUCA_PAR_CF_SCAN_WORKGROUP_SIZE
  #define GPUCA_LB_GPUTPCCFStreamCompaction_compactDigits GPUCA_PAR_CF_SCAN_WORKGROUP_SIZE
  #define GPUCA_LB_GPUTPCCompressionGatherKernels_unbuffered GPUCA_LB_COMPRESSION_GATHER
  #define GPUCA_LB_GPUTPCCompressionGatherKernels_buffered32 GPUCA_LB_COMPRESSION_GATHER
  #define GPUCA_LB_GPUTPCCompressionGatherKernels_buffered64 GPUCA_LB_COMPRESSION_GATHER
  #define GPUCA_LB_GPUTPCCompressionGatherKernels_buffered128 GPUCA_LB_COMPRESSION_GATHER
  #define GPUCA_LB_GPUTPCCompressionGatherKernels_multiBlock GPUCA_LB_COMPRESSION_GATHER

  // Defaults for non-LB parameters
  #ifndef GPUCA_PAR_SORT_STARTHITS
    #define GPUCA_PAR_SORT_STARTHITS 1
  #endif
  #ifndef GPUCA_PAR_NEIGHBOURS_FINDER_MAX_NNEIGHUP
    #define GPUCA_PAR_NEIGHBOURS_FINDER_MAX_NNEIGHUP 6
  #endif
  #ifndef GPUCA_PAR_NEIGHBOURS_FINDER_UNROLL_GLOBAL
    #define GPUCA_PAR_NEIGHBOURS_FINDER_UNROLL_GLOBAL 4
  #endif
  #ifndef GPUCA_PAR_NEIGHBOURS_FINDER_UNROLL_SHARED
    #define GPUCA_PAR_NEIGHBOURS_FINDER_UNROLL_SHARED 1
  #endif
  #ifndef GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE
    #define GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE 12
  #endif
  #ifndef GPUCA_PAR_ALTERNATE_BORDER_SORT
    #define GPUCA_PAR_ALTERNATE_BORDER_SORT 0
  #endif
  #ifndef GPUCA_PAR_SORT_BEFORE_FIT
    #define GPUCA_PAR_SORT_BEFORE_FIT 0
  #endif
  #ifndef GPUCA_PAR_COMP_GATHER_KERNEL
    #define GPUCA_PAR_COMP_GATHER_KERNEL 0
  #endif
  #ifndef GPUCA_PAR_COMP_GATHER_MODE
    #define GPUCA_PAR_COMP_GATHER_MODE 2
  #endif
  #ifndef GPUCA_PAR_CF_SCAN_WORKGROUP_SIZE
    #define GPUCA_PAR_CF_SCAN_WORKGROUP_SIZE 512
  #endif
#endif // defined(GPUCA_GPUCODE) && !defined(GPUCA_GPUCODE_GENRTC) && !defined(GPUCA_GPUCODE_NO_LAUNCH_BOUNDS)

#ifndef GPUCA_GPUCODE_GENRTC
  // Defaults (also for CPU) for non-LB parameters
  #ifndef GPUCA_PAR_SORT_STARTHITS
    #define GPUCA_PAR_SORT_STARTHITS 0
  #endif
  #ifndef GPUCA_PAR_NEIGHBOURS_FINDER_MAX_NNEIGHUP
    #define GPUCA_PAR_NEIGHBOURS_FINDER_MAX_NNEIGHUP 0
  #endif
  #ifndef GPUCA_PAR_NEIGHBOURS_FINDER_UNROLL_GLOBAL
    #define GPUCA_PAR_NEIGHBOURS_FINDER_UNROLL_GLOBAL 0
  #endif
  #ifndef GPUCA_PAR_NEIGHBOURS_FINDER_UNROLL_SHARED
    #define GPUCA_PAR_NEIGHBOURS_FINDER_UNROLL_SHARED 0
  #endif
  #ifndef GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE
    #define GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE 0
  #endif
  #ifndef GPUCA_PAR_ALTERNATE_BORDER_SORT
    #define GPUCA_PAR_ALTERNATE_BORDER_SORT 0
  #endif
  #ifndef GPUCA_PAR_SORT_BEFORE_FIT
    #define GPUCA_PAR_SORT_BEFORE_FIT 0
  #endif
  #ifndef GPUCA_PAR_COMP_GATHER_KERNEL
    #define GPUCA_PAR_COMP_GATHER_KERNEL 0
  #endif
  #ifndef GPUCA_PAR_COMP_GATHER_MODE
    #define GPUCA_PAR_COMP_GATHER_MODE 0
  #endif
  #ifndef GPUCA_PAR_NO_ATOMIC_PRECHECK
    #define GPUCA_PAR_NO_ATOMIC_PRECHECK 0
  #endif
  #ifndef GPUCA_PAR_CF_SCAN_WORKGROUP_SIZE
    #define GPUCA_PAR_CF_SCAN_WORKGROUP_SIZE 0
  #endif
  #ifndef GPUCA_PAR_DEDX_STORAGE_TYPE
    #define GPUCA_PAR_DEDX_STORAGE_TYPE float
  #endif
  #ifndef GPUCA_PAR_MERGER_INTERPOLATION_ERROR_TYPE
    #define GPUCA_PAR_MERGER_INTERPOLATION_ERROR_TYPE float
  #endif
#endif // GPUCA_GPUCODE_GENRTC

// clang-format on
#endif // GPUDEFPARAMETERSDEFAULTS_H
