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

o2_gpu_kernel_file_list(ERRORS GPUErrors.cxx)
o2_gpu_kernel_file_list(TPCTRACKER ERRORS GPUTPCTrackParam.cxx GPUTPCTrack.cxx GPUTPCGrid.cxx GPUTPCRow.cxx GPUTPCTracker.cxx)
o2_gpu_kernel_file_list(TPCTRACKLETCONS GPUTPCTrackletConstructor.cxx)
o2_gpu_kernel_file_list(TPCSLICEDATA TPCTRACKER GPUTPCSliceData.cxx)
o2_gpu_kernel_file_list(TPCOCCUPANCY GPUTPCClusterOccupancyMap.cxx)
if(ALIGPU_BUILD_TYPE STREQUAL "O2" OR CONFIG_O2_EXTENSIONS)
o2_gpu_kernel_file_list(TPCDEDX GPUdEdx.cxx)
o2_gpu_kernel_file_list(MATLUT MatLayerCylSet.cxx MatLayerCyl.cxx Ray.cxx)
o2_gpu_kernel_file_list(TPCMERGER ERRORS GPUTPCGMMerger.cxx GPUTPCGMSliceTrack.cxx GPUTPCGMTrackParam.cxx GPUTPCGMPhysicalTrackModel.cxx GPUTPCGMPropagator.cxx)
o2_gpu_kernel_file_list(O2PROPAGATOR TrackParametrization.cxx TrackParametrizationWithError.cxx Propagator.cxx TrackLTIntegral.cxx)
o2_gpu_kernel_file_list(TPCCOMPRESSION GPUTPCCompressionTrackModel.cxx)
o2_gpu_kernel_file_list(TPCDECOMPRESSION GPUTPCCompressionTrackModel.cxx ERRORS)
o2_gpu_kernel_file_list(TPCCLUSTERFINDER ERRORS ClusterAccumulator.cxx)
o2_gpu_kernel_file_list(TRDTRACKER GPUTRDTrack.cxx GPUTRDTracker.cxx GPUTRDTrackletWord.cxx GeometryBase.cxx)
o2_gpu_kernel_file_list(GLOBALREFIT TPCMERGER O2PROPAGATOR MATLUT GPUTrackingRefit.cxx)
else()
o2_gpu_kernel_file_list(TPCDEDX)
o2_gpu_kernel_file_list(MATLUT)
o2_gpu_kernel_file_list(TPCMERGER)
endif()

o2_gpu_add_kernel("GPUTPCNeighboursFinder"                            "= TPCTRACKER"                                          LB_OCL1 single)
o2_gpu_add_kernel("GPUTPCNeighboursCleaner"                           "= TPCTRACKER"                                          LB_OCL1 single)
o2_gpu_add_kernel("GPUTPCStartHitsFinder"                             "= TPCTRACKER"                                          LB_OCL1 single)
o2_gpu_add_kernel("GPUTPCStartHitsSorter"                             "= TPCTRACKER"                                          LB_OCL1 single)
o2_gpu_add_kernel("GPUTPCTrackletConstructor, singleSlice"            "= TPCTRACKER"                                          LB_OCL1 single)
o2_gpu_add_kernel("GPUTPCTrackletConstructor, allSlices"              "= TPCTRACKER"                                          LB_OCL1 single)
o2_gpu_add_kernel("GPUTPCTrackletSelector"                            "= TPCTRACKER"                                          LB_OCL1 both)
o2_gpu_add_kernel("GPUMemClean16"                                     "GPUGeneralKernels"                                     NO_OCL1 "simple, REG, (GPUCA_THREAD_COUNT, 1)" void* ptr "unsigned long" size)
o2_gpu_add_kernel("GPUitoa"                                           "GPUGeneralKernels"                                     NO_OCL1 "simple, REG, (GPUCA_THREAD_COUNT, 1)" int* ptr "unsigned long" size)
o2_gpu_add_kernel("GPUTPCGlobalTrackingCopyNumbers"                   "GPUTPCGlobalTracking TPCTRACKER"                       NO_OCL1 single int n)
o2_gpu_add_kernel("GPUTPCGlobalTracking"                              "= TPCTRACKER TPCTRACKLETCONS"                          LB      single)
o2_gpu_add_kernel("GPUTPCCreateSliceData"                             "= TPCTRACKER TPCSLICEDATA"                             LB      single)
o2_gpu_add_kernel("GPUTPCSectorDebugSortKernels, hitData"             "= TPCTRACKER"                                          NO      single)
o2_gpu_add_kernel("GPUTPCSectorDebugSortKernels, startHits"           "= TPCTRACKER"                                          NO      single)
o2_gpu_add_kernel("GPUTPCSectorDebugSortKernels, sliceTracks"         "= TPCTRACKER"                                          NO      single)
o2_gpu_add_kernel("GPUTPCGlobalDebugSortKernels, clearIds"            "= TPCMERGER"                                           NO      single char parameter)
o2_gpu_add_kernel("GPUTPCGlobalDebugSortKernels, sectorTracks"        "= TPCMERGER"                                           NO      single char parameter)
o2_gpu_add_kernel("GPUTPCGlobalDebugSortKernels, globalTracks1"       "= TPCMERGER"                                           NO      single char parameter)
o2_gpu_add_kernel("GPUTPCGlobalDebugSortKernels, globalTracks2"       "= TPCMERGER"                                           NO      single char parameter)
o2_gpu_add_kernel("GPUTPCGlobalDebugSortKernels, borderTracks"        "= TPCMERGER"                                           NO      single char parameter)
o2_gpu_add_kernel("GPUTPCCreateOccupancyMap, fill"                    "= TPCOCCUPANCY"                                        LB      simple GPUTPCClusterOccupancyMapBin* map)
o2_gpu_add_kernel("GPUTPCCreateOccupancyMap, fold"                    "= TPCOCCUPANCY"                                        LB      simple GPUTPCClusterOccupancyMapBin* map "unsigned int*" output)
o2_gpu_add_kernel("GPUTPCGMMergerTrackFit"                            "GPUTPCGMMergerGPU TPCMERGER TPCTRACKER MATLUT TPCDEDX" LB      simple int mode)
o2_gpu_add_kernel("GPUTPCGMMergerFollowLoopers"                       "GPUTPCGMMergerGPU TPCMERGER TPCTRACKER MATLUT"         LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerUnpackResetIds"                      "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple int iSlice)
o2_gpu_add_kernel("GPUTPCGMMergerSliceRefit"                          "GPUTPCGMMergerGPU TPCMERGER MATLUT"                    LB      simple int iSlice)
o2_gpu_add_kernel("GPUTPCGMMergerUnpackGlobal"                        "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple int iSlice)
o2_gpu_add_kernel("GPUTPCGMMergerUnpackSaveNumber"                    "GPUTPCGMMergerGPU TPCMERGER"                           NO      simple int id)
o2_gpu_add_kernel("GPUTPCGMMergerResolve, step0"                      "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerResolve, step1"                      "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerResolve, step2"                      "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerResolve, step3"                      "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerResolve, step4"                      "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple char useOrigTrackParam char mergeAll)
o2_gpu_add_kernel("GPUTPCGMMergerClearLinks"                          "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple char output)
o2_gpu_add_kernel("GPUTPCGMMergerMergeWithinPrepare"                  "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerMergeSlicesPrepare"                  "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple int border0 int border1 char useOrigTrackParam)
o2_gpu_add_kernel("GPUTPCGMMergerMergeBorders, step0"                 "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple int iSlice char withinSlice char mergeMode)
o2_gpu_add_kernel("GPUTPCGMMergerMergeBorders, step1"                 "GPUTPCGMMergerGPU TPCMERGER"                           NO      simple int iSlice char withinSlice char mergeMode)
o2_gpu_add_kernel("GPUTPCGMMergerMergeBorders, step2"                 "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple int iSlice char withinSlice char mergeMode)
o2_gpu_add_kernel("GPUTPCGMMergerMergeBorders, variant"               "GPUTPCGMMergerGPU TPCMERGER"                           NO      simple gputpcgmmergertypes::GPUTPCGMBorderRange* range int N int cmpMax)
o2_gpu_add_kernel("GPUTPCGMMergerMergeCE"                             "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerLinkGlobalTracks"                    "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerCollect"                             "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerSortTracks"                          "GPUTPCGMMergerGPU TPCMERGER"                           NO      simple)
o2_gpu_add_kernel("GPUTPCGMMergerSortTracksQPt"                       "GPUTPCGMMergerGPU TPCMERGER"                           NO      simple)
o2_gpu_add_kernel("GPUTPCGMMergerSortTracksPrepare"                   "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerPrepareClusters, step0"              "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerPrepareClusters, step1"              "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerPrepareClusters, step2"              "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerFinalize, step0"                     "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerFinalize, step1"                     "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerFinalize, step2"                     "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerMergeLoopers, step0"                 "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerMergeLoopers, step1"                 "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple)
o2_gpu_add_kernel("GPUTPCGMMergerMergeLoopers, step2"                 "GPUTPCGMMergerGPU TPCMERGER"                           LB      simple)

if(ALIGPU_BUILD_TYPE STREQUAL "O2" OR CONFIG_O2_EXTENSIONS)
o2_gpu_add_kernel("GPUTPCGMO2Output, prepare"                         "= TPCMERGER"                                           LB      simple)
o2_gpu_add_kernel("GPUTPCGMO2Output, sort"                            "= TPCMERGER"                                           NO      simple)
o2_gpu_add_kernel("GPUTPCGMO2Output, output"                          "= TPCMERGER"                                           LB      simple)
o2_gpu_add_kernel("GPUTPCGMO2Output, mc"                              "= TPCMERGER"                                           NO      simple)
o2_gpu_add_kernel("GPUTRDTrackerKernels, gpuVersion"                  "= TRDTRACKER MATLUT TPCMERGER"                         LB      simple GPUTRDTrackerGPU* externalInstance)
o2_gpu_add_kernel("GPUTRDTrackerKernels, o2Version"                   "= TRDTRACKER MATLUT O2PROPAGATOR"                      LB      simple GPUTRDTracker* externalInstance)
o2_gpu_add_kernel("GPUITSFitterKernels"                               "= TPCMERGER MATLUT"                                    LB      simple)
o2_gpu_add_kernel("GPUTPCConvertKernel"                               "="                                                     LB      simple)
o2_gpu_add_kernel("GPUTPCCompressionKernels, step0attached"           "= TPCCOMPRESSION"                                      LB      simple)
o2_gpu_add_kernel("GPUTPCCompressionKernels, step1unattached"         "= ERRORS"                                              LB      simple)
o2_gpu_add_kernel("GPUTPCCompressionGatherKernels, unbuffered"        "GPUTPCCompressionKernels"                              LB      simple)
o2_gpu_add_kernel("GPUTPCCompressionGatherKernels, buffered32"        "GPUTPCCompressionKernels"                              LB      simple)
o2_gpu_add_kernel("GPUTPCCompressionGatherKernels, buffered64"        "GPUTPCCompressionKernels"                              LB      simple)
o2_gpu_add_kernel("GPUTPCCompressionGatherKernels, buffered128"       "GPUTPCCompressionKernels"                              LB      simple)
o2_gpu_add_kernel("GPUTPCCompressionGatherKernels, multiBlock"        "GPUTPCCompressionKernels"                              LB      simple)
o2_gpu_add_kernel("GPUTPCDecompressionKernels, step0attached"         "= TPCDECOMPRESSION"                                    LB      simple)
o2_gpu_add_kernel("GPUTPCDecompressionKernels, step1unattached"       "= TPCDECOMPRESSION"                                    LB      simple)
o2_gpu_add_kernel("GPUTPCDecompressionUtilKernels, sortPerSectorRow"  "GPUTPCDecompressionKernels"                            LB      simple)
o2_gpu_add_kernel("GPUTPCCFCheckPadBaseline"                          "= TPCCLUSTERFINDER"                                    LB      single)
o2_gpu_add_kernel("GPUTPCCFChargeMapFiller, fillIndexMap"             "= TPCCLUSTERFINDER"                                    LB      single)
o2_gpu_add_kernel("GPUTPCCFChargeMapFiller, fillFromDigits"           "= TPCCLUSTERFINDER"                                    LB      single)
o2_gpu_add_kernel("GPUTPCCFChargeMapFiller, findFragmentStart"        "= TPCCLUSTERFINDER"                                    LB      single char setPositions)
o2_gpu_add_kernel("GPUTPCCFPeakFinder"                                "= TPCCLUSTERFINDER"                                    LB      single)
o2_gpu_add_kernel("GPUTPCCFNoiseSuppression, noiseSuppression"        "= TPCCLUSTERFINDER"                                    LB      single)
o2_gpu_add_kernel("GPUTPCCFNoiseSuppression, updatePeaks"             "= TPCCLUSTERFINDER"                                    LB      single)
o2_gpu_add_kernel("GPUTPCCFDeconvolution"                             "= TPCCLUSTERFINDER"                                    LB      single)
o2_gpu_add_kernel("GPUTPCCFClusterizer"                               "= TPCCLUSTERFINDER"                                    LB      single char onlyMC)
o2_gpu_add_kernel("GPUTPCCFMCLabelFlattener, setRowOffsets"           "= TPCCLUSTERFINDER"                                    NO      single)
o2_gpu_add_kernel("GPUTPCCFMCLabelFlattener, flatten"                 "= TPCCLUSTERFINDER"                                    NO      single GPUTPCLinearLabels* out)
o2_gpu_add_kernel("GPUTPCCFStreamCompaction, scanStart"               "= TPCCLUSTERFINDER"                                    LB      single int iBuf int stage)
o2_gpu_add_kernel("GPUTPCCFStreamCompaction, scanUp"                  "= TPCCLUSTERFINDER"                                    LB      single int iBuf int nElems)
o2_gpu_add_kernel("GPUTPCCFStreamCompaction, scanTop"                 "= TPCCLUSTERFINDER"                                    LB      single int iBuf int nElems)
o2_gpu_add_kernel("GPUTPCCFStreamCompaction, scanDown"                "= TPCCLUSTERFINDER"                                    LB      single int iBuf "unsigned int" offset int nElems)
o2_gpu_add_kernel("GPUTPCCFStreamCompaction, compactDigits"           "= TPCCLUSTERFINDER"                                    LB      single int iBuf int stage ChargePos* in ChargePos* out)
o2_gpu_add_kernel("GPUTPCCFDecodeZS"                                  "= TPCCLUSTERFINDER"                                    LB      single int firstHBF)
o2_gpu_add_kernel("GPUTPCCFDecodeZSLink"                              "GPUTPCCFDecodeZS"                                      LB      single int firstHBF)
o2_gpu_add_kernel("GPUTPCCFDecodeZSDenseLink"                         "GPUTPCCFDecodeZS"                                      LB      single int firstHBF)
o2_gpu_add_kernel("GPUTPCCFGather"                                    "="                                                     LB      single o2::tpc::ClusterNative* dest)
o2_gpu_add_kernel("GPUTrackingRefitKernel, mode0asGPU"                "= GLOBALREFIT "                                        LB      simple)
o2_gpu_add_kernel("GPUTrackingRefitKernel, mode1asTrackParCov"        "= GLOBALREFIT "                                        LB      simple)
endif()
