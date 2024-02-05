# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

# file kernels.cmake
# author David Rohr

o2_gpu_add_kernel("GPUTPCNeighboursFinder"                       LB_OCL1 single)
o2_gpu_add_kernel("GPUTPCNeighboursCleaner"                      LB_OCL1 single)
o2_gpu_add_kernel("GPUTPCStartHitsFinder"                        LB_OCL1 single)
o2_gpu_add_kernel("GPUTPCStartHitsSorter"                        LB_OCL1 single)
o2_gpu_add_kernel("GPUTPCTrackletConstructor, singleSlice"       LB_OCL1 single)
o2_gpu_add_kernel("GPUTPCTrackletConstructor, allSlices"         LB_OCL1 single)
o2_gpu_add_kernel("GPUTPCTrackletSelector"                       LB_OCL1 both)
o2_gpu_add_kernel("GPUMemClean16"                                NO_OCL1 "simple, REG, (GPUCA_THREAD_COUNT, 1)" void* ptr "unsigned long" size)
o2_gpu_add_kernel("GPUTPCGlobalTrackingCopyNumbers"              NO_OCL1 single int n)
o2_gpu_add_kernel("GPUTPCCreateSliceData"                        LB      single)
o2_gpu_add_kernel("GPUTPCGlobalTracking"                         LB      single)
o2_gpu_add_kernel("GPUTPCCreateOccupancyMap, fill"               LB      simple GPUTPCClusterOccupancyMapBin* map)
o2_gpu_add_kernel("GPUTPCCreateOccupancyMap, fold"               LB      simple GPUTPCClusterOccupancyMapBin* map)
o2_gpu_add_kernel("GPUTPCGMMergerTrackFit"                       LB      simple int mode)
o2_gpu_add_kernel("GPUTPCGMMergerFollowLoopers"                  LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerUnpackResetIds"                 LB      simple int iSlice)
o2_gpu_add_kernel("GPUTPCGMMergerSliceRefit"                     LB      simple int iSlice)
o2_gpu_add_kernel("GPUTPCGMMergerUnpackGlobal"                   LB      simple int iSlice)
o2_gpu_add_kernel("GPUTPCGMMergerUnpackSaveNumber"               NO      simple int id)
o2_gpu_add_kernel("GPUTPCGMMergerResolve, step0"                 LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerResolve, step1"                 LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerResolve, step2"                 LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerResolve, step3"                 LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerResolve, step4"                 LB      simple char useOrigTrackParam char mergeAll)
o2_gpu_add_kernel("GPUTPCGMMergerClearLinks"                     LB      simple char nOutput)
o2_gpu_add_kernel("GPUTPCGMMergerMergeWithinPrepare"             LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerMergeSlicesPrepare"             LB      simple int border0 int border1 char useOrigTrackParam)
o2_gpu_add_kernel("GPUTPCGMMergerMergeBorders, step0"            LB      simple int iSlice char withinSlice char mergeMode)
o2_gpu_add_kernel("GPUTPCGMMergerMergeBorders, step1"            NO      simple int iSlice char withinSlice char mergeMode)
o2_gpu_add_kernel("GPUTPCGMMergerMergeBorders, step2"            LB      simple int iSlice char withinSlice char mergeMode)
o2_gpu_add_kernel("GPUTPCGMMergerMergeBorders, variant"          NO      simple gputpcgmmergertypes::GPUTPCGMBorderRange* range int N int cmpMax)
o2_gpu_add_kernel("GPUTPCGMMergerMergeCE"                        LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerLinkGlobalTracks"               LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerCollect"                        LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerSortTracks"                     NO      simple)
o2_gpu_add_kernel("GPUTPCGMMergerSortTracksQPt"                  NO      simple)
o2_gpu_add_kernel("GPUTPCGMMergerSortTracksPrepare"              LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerPrepareClusters, step0"         LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerPrepareClusters, step1"         LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerPrepareClusters, step2"         LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerFinalize, step0"                LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerFinalize, step1"                LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerFinalize, step2"                LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerMergeLoopers, step0"            LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerMergeLoopers, step1"            LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerMergeLoopers, step2"            LB      simple)

if(NOT ALIGPU_BUILD_TYPE STREQUAL "ALIROOT" OR CONFIG_O2_EXTENSIONS)
o2_gpu_add_kernel("GPUTPCGMO2Output, prepare"                    LB      simple)
o2_gpu_add_kernel("GPUTPCGMO2Output, sort"                       NO      simple)
o2_gpu_add_kernel("GPUTPCGMO2Output, output"                     LB      simple)
o2_gpu_add_kernel("GPUTPCGMO2Output, mc"                         NO      simple)
o2_gpu_add_kernel("GPUTRDTrackerKernels, gpuVersion"             LB      simple GPUTRDTrackerGPU* externalInstance)
o2_gpu_add_kernel("GPUTRDTrackerKernels, o2Version"              LB      simple GPUTRDTracker* externalInstance)
o2_gpu_add_kernel("GPUITSFitterKernel"                           LB      simple)
o2_gpu_add_kernel("GPUTPCConvertKernel"                          LB      simple)
o2_gpu_add_kernel("GPUTPCCompressionKernels, step0attached"      LB      simple)
o2_gpu_add_kernel("GPUTPCCompressionKernels, step1unattached"    LB      simple)
o2_gpu_add_kernel("GPUTPCCompressionGatherKernels, unbuffered"   LB      simple)
o2_gpu_add_kernel("GPUTPCCompressionGatherKernels, buffered32"   LB      simple)
o2_gpu_add_kernel("GPUTPCCompressionGatherKernels, buffered64"   LB      simple)
o2_gpu_add_kernel("GPUTPCCompressionGatherKernels, buffered128"  LB      simple)
o2_gpu_add_kernel("GPUTPCCompressionGatherKernels, multiBlock"   LB      simple)
o2_gpu_add_kernel("GPUTPCCFCheckPadBaseline"                     LB      single)
o2_gpu_add_kernel("GPUTPCCFChargeMapFiller, fillIndexMap"        LB      single)
o2_gpu_add_kernel("GPUTPCCFChargeMapFiller, fillFromDigits"      LB      single)
o2_gpu_add_kernel("GPUTPCCFChargeMapFiller, findFragmentStart"   LB      single char setPositions)
o2_gpu_add_kernel("GPUTPCCFPeakFinder"                           LB      single)
o2_gpu_add_kernel("GPUTPCCFNoiseSuppression, noiseSuppression"   LB      single)
o2_gpu_add_kernel("GPUTPCCFNoiseSuppression, updatePeaks"        LB      single)
o2_gpu_add_kernel("GPUTPCCFDeconvolution"                        LB      single)
o2_gpu_add_kernel("GPUTPCCFClusterizer"                          LB      single char onlyMC)
o2_gpu_add_kernel("GPUTPCCFMCLabelFlattener, setRowOffsets"      NO      single)
o2_gpu_add_kernel("GPUTPCCFMCLabelFlattener, flatten"            NO      single GPUTPCLinearLabels* out)
o2_gpu_add_kernel("GPUTPCCFStreamCompaction, scanStart"          LB      single int iBuf int stage)
o2_gpu_add_kernel("GPUTPCCFStreamCompaction, scanUp"             LB      single int iBuf int nElems)
o2_gpu_add_kernel("GPUTPCCFStreamCompaction, scanTop"            LB      single int iBuf int nElems)
o2_gpu_add_kernel("GPUTPCCFStreamCompaction, scanDown"           LB      single int iBuf "unsigned int" offset int nElems)
o2_gpu_add_kernel("GPUTPCCFStreamCompaction, compactDigits"      LB      single int iBuf int stage ChargePos* in ChargePos* out)
o2_gpu_add_kernel("GPUTPCCFDecodeZS"                             LB      single int firstHBF)
o2_gpu_add_kernel("GPUTPCCFDecodeZSLink"                         LB      single int firstHBF)
o2_gpu_add_kernel("GPUTPCCFDecodeZSDenseLink"                    LB      single int firstHBF)
o2_gpu_add_kernel("GPUTPCCFGather"                               LB      single o2::tpc::ClusterNative* dest)
o2_gpu_add_kernel("GPUTrackingRefitKernel, mode0asGPU"           LB      simple)
o2_gpu_add_kernel("GPUTrackingRefitKernel, mode1asTrackParCov"   LB      simple)
endif()
