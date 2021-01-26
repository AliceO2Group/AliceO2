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
#ifdef GPUCA_GPUCODE
#if defined(GPUCA_GPUTYPE_VEGA)
  #define GPUCA_WARP_SIZE 64
  #define GPUCA_THREAD_COUNT 256
  #define GPUCA_LB_GPUTPCCreateSliceData 128
  #define GPUCA_LB_GPUTPCStartHitsSorter 1024, 2
  #define GPUCA_LB_GPUTPCStartHitsFinder 1024
  #define GPUCA_LB_GPUTPCTrackletConstructor 256, 2
  #define GPUCA_LB_GPUTPCTrackletSelector 256, 8
  #define GPUCA_LB_GPUTPCNeighboursFinder 1024, 1
  #define GPUCA_LB_GPUTPCNeighboursCleaner 896
  #define GPUCA_LB_GPUTPCGlobalTracking 256
  #define GPUCA_LB_GPUTPCCFDecodeZS 64, 4
  #define GPUCA_LB_GPUTPCCFGather 1024, 1
  #define GPUCA_LB_GPUTPCGMMergerTrackFit 64, 1
  #define GPUCA_LB_GPUTPCGMMergerFollowLoopers 256, 4, 200
  #define GPUCA_LB_GPUTPCGMMergerSliceRefit 256
  #define GPUCA_LB_GPUTPCGMMergerUnpackResetIds 256
  #define GPUCA_LB_GPUTPCGMMergerUnpackGlobal 256
  #define GPUCA_LB_GPUTPCGMMergerResolve_step0 256
  #define GPUCA_LB_GPUTPCGMMergerResolve_step1 256
  #define GPUCA_LB_GPUTPCGMMergerResolve_step2 256
  #define GPUCA_LB_GPUTPCGMMergerResolve_step3 256
  #define GPUCA_LB_GPUTPCGMMergerResolve_step4 256
  #define GPUCA_LB_GPUTPCGMMergerClearLinks 256
  #define GPUCA_LB_GPUTPCGMMergerMergeWithinPrepare 256
  #define GPUCA_LB_GPUTPCGMMergerMergeSlicesPrepare 256
  #define GPUCA_LB_GPUTPCGMMergerMergeBorders_step0 256
  #define GPUCA_LB_GPUTPCGMMergerMergeBorders_step2 256
  #define GPUCA_LB_GPUTPCGMMergerMergeCE 256
  #define GPUCA_LB_GPUTPCGMMergerLinkGlobalTracks 256
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
  #define GPUCA_LB_GPUTPCCFCheckPadBaseline 64
  #define GPUCA_LB_GPUTPCCFChargeMapFiller_fillIndexMap 512
  #define GPUCA_LB_GPUTPCCFChargeMapFiller_fillFromDigits 512
  #define GPUCA_LB_GPUTPCCFChargeMapFiller_findFragmentStart 512
  #define GPUCA_LB_GPUTPCCFPeakFinder 512
  #define GPUCA_LB_GPUTPCCFNoiseSuppression 512
  #define GPUCA_LB_GPUTPCCFDeconvolution 512
  #define GPUCA_LB_GPUTPCCFClusterizer 512
  #define GPUCA_LB_COMPRESSION_GATHER 1024
  #define GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP 5
  #define GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE 20
  #define GPUCA_CONSTRUCTOR_IN_PIPELINE 1
  #define GPUCA_SELECTOR_IN_PIPELINE 1
  #define GPUCA_ALTERNATE_BORDER_SORT 1
  #define GPUCA_SORT_BEFORE_FIT 1
  #define GPUCA_MERGER_SPLIT_LOOP_INTERPOLATION 1
  #define GPUCA_TRACKLET_SELECTOR_SLICE_COUNT 1
  #define GPUCA_NO_ATOMIC_PRECHECK 1
  #define GPUCA_DEDX_STORAGE_TYPE unsigned short
  #define GPUCA_MERGER_INTERPOLATION_ERROR_TYPE half
  #define GPUCA_COMP_GATHER_KERNEL 4
  #define GPUCA_COMP_GATHER_MODE 3
#elif defined(GPUCA_GPUTYPE_AMPERE)
  #define GPUCA_WARP_SIZE 32
  #define GPUCA_THREAD_COUNT 512
  #define GPUCA_LB_GPUTPCCreateSliceData 384
  #define GPUCA_LB_GPUTPCStartHitsSorter 512, 1
  #define GPUCA_LB_GPUTPCStartHitsFinder 512
  #define GPUCA_LB_GPUTPCTrackletConstructor 256, 2 // best single-kernel: 128, 4
  #define GPUCA_LB_GPUTPCTrackletSelector 192, 3    // best single-kernel: 128, 4
  #define GPUCA_LB_GPUTPCNeighboursFinder 640, 1    // best single-kernel: 768, 1
  #define GPUCA_LB_GPUTPCNeighboursCleaner 512
  #define GPUCA_LB_GPUTPCGlobalTracking 128, 4
  #define GPUCA_LB_GPUTPCCFDecodeZS 64, 10
  #define GPUCA_LB_GPUTPCCFGather 1024, 1
  #define GPUCA_LB_GPUTPCGMMergerTrackFit 64, 4
  #define GPUCA_LB_GPUTPCGMMergerFollowLoopers 64, 12
  #define GPUCA_LB_GPUTPCGMMergerSliceRefit 32, 6
  #define GPUCA_LB_GPUTPCGMMergerUnpackResetIds 256
  #define GPUCA_LB_GPUTPCGMMergerUnpackGlobal 256
  #define GPUCA_LB_GPUTPCGMMergerResolve_step0 256
  #define GPUCA_LB_GPUTPCGMMergerResolve_step1 256
  #define GPUCA_LB_GPUTPCGMMergerResolve_step2 256
  #define GPUCA_LB_GPUTPCGMMergerResolve_step3 256
  #define GPUCA_LB_GPUTPCGMMergerResolve_step4 256, 4
  #define GPUCA_LB_GPUTPCGMMergerClearLinks 256
  #define GPUCA_LB_GPUTPCGMMergerMergeWithinPrepare 256
  #define GPUCA_LB_GPUTPCGMMergerMergeSlicesPrepare 256, 2
  #define GPUCA_LB_GPUTPCGMMergerMergeBorders_step0 192
  #define GPUCA_LB_GPUTPCGMMergerMergeBorders_step2 64, 2
  #define GPUCA_LB_GPUTPCGMMergerMergeCE 256
  #define GPUCA_LB_GPUTPCGMMergerLinkGlobalTracks 256
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
  #define GPUCA_LB_GPUTPCCFCheckPadBaseline 64,8
  #define GPUCA_LB_GPUTPCCFChargeMapFiller_fillIndexMap 448
  #define GPUCA_LB_GPUTPCCFChargeMapFiller_fillFromDigits 448
  #define GPUCA_LB_GPUTPCCFChargeMapFiller_findFragmentStart 448
  #define GPUCA_LB_GPUTPCCFPeakFinder 128
  #define GPUCA_LB_GPUTPCCFNoiseSuppression 448
  #define GPUCA_LB_GPUTPCCFDeconvolution 384
  #define GPUCA_LB_GPUTPCCFClusterizer 448
  #define GPUCA_LB_COMPRESSION_GATHER 1024
  #define GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP 4
  #define GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE 20
  #define GPUCA_CONSTRUCTOR_IN_PIPELINE 1
  #define GPUCA_SELECTOR_IN_PIPELINE 1
  #define GPUCA_ALTERNATE_BORDER_SORT 1
  #define GPUCA_SORT_BEFORE_FIT 1
  #define GPUCA_MERGER_SPLIT_LOOP_INTERPOLATION 1
  #define GPUCA_TRACKLET_SELECTOR_SLICE_COUNT 1
  #define GPUCA_NO_ATOMIC_PRECHECK 1
  #define GPUCA_DEDX_STORAGE_TYPE unsigned short
  #define GPUCA_MERGER_INTERPOLATION_ERROR_TYPE half
  #define GPUCA_COMP_GATHER_KERNEL 4
  #define GPUCA_COMP_GATHER_MODE 3
#elif defined(GPUCA_GPUTYPE_TURING)
  #define GPUCA_WARP_SIZE 32
  #define GPUCA_THREAD_COUNT 512
  #define GPUCA_LB_GPUTPCCreateSliceData 256
  #define GPUCA_LB_GPUTPCStartHitsSorter 512, 1
  #define GPUCA_LB_GPUTPCStartHitsFinder 512
  #define GPUCA_LB_GPUTPCTrackletConstructor 256, 2
  #define GPUCA_LB_GPUTPCTrackletSelector 192, 3
  #define GPUCA_LB_GPUTPCNeighboursFinder 640, 1
  #define GPUCA_LB_GPUTPCNeighboursCleaner 512
  #define GPUCA_LB_GPUTPCGlobalTracking 192, 2
  #define GPUCA_LB_GPUTPCCFDecodeZS 64, 8
  #define GPUCA_LB_GPUTPCCFGather 1024, 1
  #define GPUCA_LB_GPUTPCGMMergerTrackFit 32, 8
  #define GPUCA_LB_GPUTPCGMMergerFollowLoopers 128, 4
  #define GPUCA_LB_GPUTPCGMMergerSliceRefit 64, 5
  #define GPUCA_LB_GPUTPCGMMergerUnpackResetIds 256
  #define GPUCA_LB_GPUTPCGMMergerUnpackGlobal 256
  #define GPUCA_LB_GPUTPCGMMergerResolve_step0 256
  #define GPUCA_LB_GPUTPCGMMergerResolve_step1 256
  #define GPUCA_LB_GPUTPCGMMergerResolve_step2 256
  #define GPUCA_LB_GPUTPCGMMergerResolve_step3 256
  #define GPUCA_LB_GPUTPCGMMergerResolve_step4 256, 4
  #define GPUCA_LB_GPUTPCGMMergerClearLinks 256
  #define GPUCA_LB_GPUTPCGMMergerMergeWithinPrepare 256
  #define GPUCA_LB_GPUTPCGMMergerMergeSlicesPrepare 256, 2
  #define GPUCA_LB_GPUTPCGMMergerMergeBorders_step0 192
  #define GPUCA_LB_GPUTPCGMMergerMergeBorders_step2 256
  #define GPUCA_LB_GPUTPCGMMergerMergeCE 256
  #define GPUCA_LB_GPUTPCGMMergerLinkGlobalTracks 256
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
  #define GPUCA_LB_COMPRESSION_GATHER 1024
  #define GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP 4
  #define GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE 20
  #define GPUCA_CONSTRUCTOR_IN_PIPELINE 1
  #define GPUCA_SELECTOR_IN_PIPELINE 1
  #define GPUCA_ALTERNATE_BORDER_SORT 1
  #define GPUCA_SORT_BEFORE_FIT 1
  #define GPUCA_MERGER_SPLIT_LOOP_INTERPOLATION 1
  #define GPUCA_TRACKLET_SELECTOR_SLICE_COUNT 1
  #define GPUCA_NO_ATOMIC_PRECHECK 1
  #define GPUCA_COMP_GATHER_KERNEL 0
  #define GPUCA_DEDX_STORAGE_TYPE unsigned short
  #define GPUCA_MERGER_INTERPOLATION_ERROR_TYPE half
  // #define GPUCA_USE_TEXTURES
#elif defined(GPUCA_GPUTYPE_OPENCL)
#else
  #error GPU TYPE NOT SET
#endif
#endif // GPUCA_GPUCODE

#ifdef GPUCA_GPUCODE
  // Default settings, if not already set for selected GPU type
  #ifndef GPUCA_THREAD_COUNT
    #define GPUCA_THREAD_COUNT 256
  #endif
  #ifndef GPUCA_LB_GPUTPCCreateSliceData
    #define GPUCA_LB_GPUTPCCreateSliceData 256
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
  #ifndef GPUCA_LB_GPUTPCGlobalTracking
    #define GPUCA_LB_GPUTPCGlobalTracking 256
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
  #ifndef GPUCA_LB_GPUTPCGMMergerSliceRefit
    #define GPUCA_LB_GPUTPCGMMergerSliceRefit 256
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
  #ifndef GPUCA_LB_GPUTPCGMMergerMergeSlicesPrepare
    #define GPUCA_LB_GPUTPCGMMergerMergeSlicesPrepare 256
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
  #ifndef GPUCA_LB_GPUTPCGMMergerLinkGlobalTracks
    #define GPUCA_LB_GPUTPCGMMergerLinkGlobalTracks 256
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
  #ifndef GPUCA_LB_GPUTPCGMMergerMergeLoopers
    #define GPUCA_LB_GPUTPCGMMergerMergeLoopers 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMO2Output_prepare
    #define GPUCA_LB_GPUTPCGMO2Output_prepare 256
  #endif
  #ifndef GPUCA_LB_GPUTPCGMO2Output_output
    #define GPUCA_LB_GPUTPCGMO2Output_output 256
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
  #ifndef GPUCA_LB_GPUTrackingRefitKernel_mode0asGPU
    #define GPUCA_LB_GPUTrackingRefitKernel_mode0asGPU 256
  #endif
  #ifndef GPUCA_LB_GPUTrackingRefitKernel_mode1asTrackParCov
    #define GPUCA_LB_GPUTrackingRefitKernel_mode1asTrackParCov 256
  #endif
  #define GPUCA_GET_THREAD_COUNT(...) GPUCA_M_FIRST(__VA_ARGS__)
#else
  // The following defaults are needed to compile the host code
  #define GPUCA_GET_THREAD_COUNT(...) 1
#endif

#define GPUCA_GET_WARP_COUNT(...) (GPUCA_GET_THREAD_COUNT(__VA_ARGS__) / GPUCA_WARP_SIZE)

#define GPUCA_THREAD_COUNT_SCAN 512 // TODO: WARNING!!! Must not be GPUTYPE-dependent right now! // TODO: Fix!

#define GPUCA_LB_GPUTPCCFNoiseSuppression_noiseSuppression GPUCA_LB_GPUTPCCFNoiseSuppression
#define GPUCA_LB_GPUTPCCFNoiseSuppression_updatePeaks GPUCA_LB_GPUTPCCFNoiseSuppression
#define GPUCA_LB_GPUTPCCFStreamCompaction_scanStart GPUCA_THREAD_COUNT_SCAN
#define GPUCA_LB_GPUTPCCFStreamCompaction_scanUp GPUCA_THREAD_COUNT_SCAN
#define GPUCA_LB_GPUTPCCFStreamCompaction_scanTop GPUCA_THREAD_COUNT_SCAN
#define GPUCA_LB_GPUTPCCFStreamCompaction_scanDown GPUCA_THREAD_COUNT_SCAN
#define GPUCA_LB_GPUTPCCFStreamCompaction_compactDigits GPUCA_THREAD_COUNT_SCAN
#define GPUCA_LB_GPUTPCTrackletConstructor_singleSlice GPUCA_LB_GPUTPCTrackletConstructor
#define GPUCA_LB_GPUTPCTrackletConstructor_allSlices GPUCA_LB_GPUTPCTrackletConstructor
#define GPUCA_LB_GPUTPCCompressionGatherKernels_unbuffered GPUCA_LB_COMPRESSION_GATHER
#define GPUCA_LB_GPUTPCCompressionGatherKernels_buffered32 GPUCA_LB_COMPRESSION_GATHER
#define GPUCA_LB_GPUTPCCompressionGatherKernels_buffered64 GPUCA_LB_COMPRESSION_GATHER
#define GPUCA_LB_GPUTPCCompressionGatherKernels_buffered128 GPUCA_LB_COMPRESSION_GATHER
#define GPUCA_LB_GPUTPCCompressionGatherKernels_multiBlock GPUCA_LB_COMPRESSION_GATHER

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
  #ifndef GPUCA_ALTERNATE_BORDER_SORT
  #define GPUCA_ALTERNATE_BORDER_SORT 0
  #endif
  #ifndef GPUCA_SORT_BEFORE_FIT
  #define GPUCA_SORT_BEFORE_FIT 0
  #endif
  #ifndef GPUCA_MERGER_SPLIT_LOOP_INTERPOLATION
  #define GPUCA_MERGER_SPLIT_LOOP_INTERPOLATION 0
  #endif
  #ifndef GPUCA_TRACKLET_SELECTOR_SLICE_COUNT
  #define GPUCA_TRACKLET_SELECTOR_SLICE_COUNT 8                          // Currently must be smaller than avaiable MultiProcessors on GPU or will result in wrong results
  #endif
  #ifndef GPUCA_COMP_GATHER_KERNEL
  #define GPUCA_COMP_GATHER_KERNEL 0
  #endif
  #ifndef GPUCA_COMP_GATHER_MODE
  #define GPUCA_COMP_GATHER_MODE 2
  #endif
#else
  #define GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP 0
  #define GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE 0
  #define GPUCA_CONSTRUCTOR_IN_PIPELINE 1
  #define GPUCA_SELECTOR_IN_PIPELINE 1
  #define GPUCA_ALTERNATE_BORDER_SORT 0
  #define GPUCA_SORT_BEFORE_FIT 0
  #define GPUCA_MERGER_SPLIT_LOOP_INTERPOLATION 0
  #define GPUCA_TRACKLET_SELECTOR_SLICE_COUNT 1
  #define GPUCA_THREAD_COUNT_FINDER 1
  #define GPUCA_COMP_GATHER_KERNEL 0
  #define GPUCA_COMP_GATHER_MODE 0
#endif
#ifndef GPUCA_DEDX_STORAGE_TYPE
#define GPUCA_DEDX_STORAGE_TYPE float
#endif
#ifndef GPUCA_MERGER_INTERPOLATION_ERROR_TYPE
#define GPUCA_MERGER_INTERPOLATION_ERROR_TYPE float
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
#define GPUCA_HOST_MEMORY_SIZE       ((size_t) 1 * 1024 * 1024 * 1024) // Size of memory allocated on Host
#define GPUCA_GPU_STACK_SIZE         ((size_t)               8 * 1024) // Stack size per GPU thread
#define GPUCA_GPU_HEAP_SIZE          ((size_t)       16 * 1025 * 1024) // Stack size per GPU thread

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

#if defined(__cplusplus) && __cplusplus >= 201703L
#define GPUCA_NEW_ALIGNMENT (std::align_val_t{GPUCA_BUFFER_ALIGNMENT})
#define GPUCA_OPERATOR_NEW_ALIGNMENT ,GPUCA_NEW_ALIGNMENT
#else
#define GPUCA_NEW_ALIGNMENT
#define GPUCA_OPERATOR_NEW_ALIGNMENT
#endif

// clang-format on
#endif
