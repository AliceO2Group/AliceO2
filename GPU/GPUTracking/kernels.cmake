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
o2_gpu_kernel_file_list(TPCTRACKER ERRORS GPUTPCTrackParam.cxx GPUTPCTrack.cxx GPUTPCGrid.cxx GPUTPCTracker.cxx)
o2_gpu_kernel_file_list(TPCTRACKLETCONS GPUTPCTrackletConstructor.cxx)
o2_gpu_kernel_file_list(TPCSECTORDATA TPCTRACKER GPUTPCTrackingData.cxx)
o2_gpu_kernel_file_list(TPCOCCUPANCY GPUTPCClusterOccupancyMap.cxx)
o2_gpu_kernel_file_list(TPCDEDX GPUdEdx.cxx)
o2_gpu_kernel_file_list(MATLUT MatLayerCylSet.cxx MatLayerCyl.cxx Ray.cxx)
o2_gpu_kernel_file_list(TPCMERGER ERRORS GPUTPCGMSectorTrack.cxx GPUTPCGMMerger.cxx GPUTPCGMTrackParam.cxx GPUTPCGMPhysicalTrackModel.cxx GPUTPCGMPropagator.cxx)
o2_gpu_kernel_file_list(O2PROPAGATOR TrackParametrization.cxx TrackParametrizationWithError.cxx Propagator.cxx TrackLTIntegral.cxx)
o2_gpu_kernel_file_list(TPCCOMPRESSION GPUTPCCompressionTrackModel.cxx)
o2_gpu_kernel_file_list(TPCDECOMPRESSION GPUTPCCompressionTrackModel.cxx ERRORS)
o2_gpu_kernel_file_list(TPCCLUSTERFINDER ERRORS ClusterAccumulator.cxx)
o2_gpu_kernel_file_list(TRDTRACKER GPUTRDTrack.cxx GPUTRDTracker.cxx GPUTRDTrackletWord.cxx GeometryBase.cxx)
o2_gpu_kernel_file_list(GLOBALREFIT TPCMERGER O2PROPAGATOR MATLUT GPUTrackingRefit.cxx)
if(onnxruntime_FOUND)
o2_gpu_kernel_file_list(TPCNNCLUSTERFINDER ERRORS ClusterAccumulator.cxx GPUTPCNNClusterizerKernels.cxx)
endif()

o2_gpu_add_kernel("GPUTPCNeighboursFinder"                                "= TPCTRACKER"                                          LB)
o2_gpu_add_kernel("GPUTPCNeighboursCleaner"                               "= TPCTRACKER"                                          LB)
o2_gpu_add_kernel("GPUTPCStartHitsFinder"                                 "= TPCTRACKER"                                          LB)
o2_gpu_add_kernel("GPUTPCStartHitsSorter"                                 "= TPCTRACKER"                                          LB)
o2_gpu_add_kernel("GPUTPCTrackletConstructor"                             "= TPCTRACKER"                                          LB)
o2_gpu_add_kernel("GPUTPCTrackletSelector"                                "= TPCTRACKER"                                          LB)
o2_gpu_add_kernel("GPUMemClean16"                                         "GPUGeneralKernels"                                     NO void* ptr uint64_t size)
o2_gpu_add_kernel("GPUitoa"                                               "GPUGeneralKernels"                                     NO int32_t* ptr uint64_t size)
o2_gpu_add_kernel("GPUTPCExtrapolationTrackingCopyNumbers"                "GPUTPCExtrapolationTracking TPCTRACKER"                NO int32_t n)
o2_gpu_add_kernel("GPUTPCExtrapolationTracking"                           "= TPCTRACKER TPCTRACKLETCONS"                          LB)
o2_gpu_add_kernel("GPUTPCCreateTrackingData"                              "= TPCTRACKER TPCSECTORDATA"                            LB)
o2_gpu_add_kernel("GPUTPCSectorDebugSortKernels, hitData"                 "= TPCTRACKER")
o2_gpu_add_kernel("GPUTPCSectorDebugSortKernels, startHits"               "= TPCTRACKER")
o2_gpu_add_kernel("GPUTPCSectorDebugSortKernels, sectorTracks"            "= TPCTRACKER")
o2_gpu_add_kernel("GPUTPCGlobalDebugSortKernels, clearIds"                "= TPCMERGER"                                           NO int8_t parameter)
o2_gpu_add_kernel("GPUTPCGlobalDebugSortKernels, sectorTracks"            "= TPCMERGER"                                           NO int8_t parameter)
o2_gpu_add_kernel("GPUTPCGlobalDebugSortKernels, mergedTracks1"           "= TPCMERGER"                                           NO int8_t parameter)
o2_gpu_add_kernel("GPUTPCGlobalDebugSortKernels, mergedTracks2"           "= TPCMERGER"                                           NO int8_t parameter)
o2_gpu_add_kernel("GPUTPCGlobalDebugSortKernels, borderTracks"            "= TPCMERGER"                                           NO int8_t parameter)
o2_gpu_add_kernel("GPUTPCCreateOccupancyMap, fill"                        "= TPCOCCUPANCY"                                        LB GPUTPCClusterOccupancyMapBin* map)
o2_gpu_add_kernel("GPUTPCCreateOccupancyMap, fold"                        "= TPCOCCUPANCY"                                        LB GPUTPCClusterOccupancyMapBin* map uint32_t* output)
o2_gpu_add_kernel("GPUTPCGMMergerTrackFit"                                "GPUTPCGMMergerGPU TPCMERGER TPCTRACKER MATLUT TPCDEDX" LB int32_t mode)
o2_gpu_add_kernel("GPUTPCGMMergerFollowLoopers"                           "GPUTPCGMMergerGPU TPCMERGER TPCTRACKER MATLUT"         LB)
o2_gpu_add_kernel("GPUTPCGMMergerUnpackResetIds"                          "GPUTPCGMMergerGPU TPCMERGER"                           LB int32_t iSector)
o2_gpu_add_kernel("GPUTPCGMMergerSectorRefit"                             "GPUTPCGMMergerGPU TPCMERGER MATLUT"                    LB int32_t iSector)
o2_gpu_add_kernel("GPUTPCGMMergerUnpackGlobal"                            "GPUTPCGMMergerGPU TPCMERGER"                           LB int32_t iSector)
o2_gpu_add_kernel("GPUTPCGMMergerUnpackSaveNumber"                        "GPUTPCGMMergerGPU TPCMERGER"                           NO int32_t id)
o2_gpu_add_kernel("GPUTPCGMMergerResolve, step0"                          "GPUTPCGMMergerGPU TPCMERGER"                           LB)
o2_gpu_add_kernel("GPUTPCGMMergerResolve, step1"                          "GPUTPCGMMergerGPU TPCMERGER"                           LB)
o2_gpu_add_kernel("GPUTPCGMMergerResolve, step2"                          "GPUTPCGMMergerGPU TPCMERGER"                           LB)
o2_gpu_add_kernel("GPUTPCGMMergerResolve, step3"                          "GPUTPCGMMergerGPU TPCMERGER"                           LB)
o2_gpu_add_kernel("GPUTPCGMMergerResolve, step4"                          "GPUTPCGMMergerGPU TPCMERGER"                           LB int8_t useOrigTrackParam int8_t mergeAll)
o2_gpu_add_kernel("GPUTPCGMMergerClearLinks"                              "GPUTPCGMMergerGPU TPCMERGER"                           LB int8_t output)
o2_gpu_add_kernel("GPUTPCGMMergerMergeWithinPrepare"                      "GPUTPCGMMergerGPU TPCMERGER"                           LB)
o2_gpu_add_kernel("GPUTPCGMMergerMergeSectorsPrepare"                     "GPUTPCGMMergerGPU TPCMERGER"                           LB int32_t border0 int32_t border1 int8_t useOrigTrackParam)
o2_gpu_add_kernel("GPUTPCGMMergerMergeBorders, step0"                     "GPUTPCGMMergerGPU TPCMERGER"                           LB int32_t iSector uint8_t mergeMode)
o2_gpu_add_kernel("GPUTPCGMMergerMergeBorders, step1"                     "GPUTPCGMMergerGPU TPCMERGER"                           NO int32_t iSector uint8_t mergeMode)
o2_gpu_add_kernel("GPUTPCGMMergerMergeBorders, step2"                     "GPUTPCGMMergerGPU TPCMERGER"                           LB int32_t iSector uint8_t mergeMode)
o2_gpu_add_kernel("GPUTPCGMMergerMergeBorders, variant"                   "GPUTPCGMMergerGPU TPCMERGER"                           NO gputpcgmmergertypes::GPUTPCGMBorderRange* range int32_t N int32_t cmpMax)
o2_gpu_add_kernel("GPUTPCGMMergerMergeCE"                                 "GPUTPCGMMergerGPU TPCMERGER"                           LB)
o2_gpu_add_kernel("GPUTPCGMMergerLinkExtrapolatedTracks"                  "GPUTPCGMMergerGPU TPCMERGER"                           LB)
o2_gpu_add_kernel("GPUTPCGMMergerCollect"                                 "GPUTPCGMMergerGPU TPCMERGER"                           LB)
o2_gpu_add_kernel("GPUTPCGMMergerSortTracks"                              "GPUTPCGMMergerGPU TPCMERGER")
o2_gpu_add_kernel("GPUTPCGMMergerSortTracksQPt"                           "GPUTPCGMMergerGPU TPCMERGER")
o2_gpu_add_kernel("GPUTPCGMMergerSortTracksPrepare"                       "GPUTPCGMMergerGPU TPCMERGER"                           LB)
o2_gpu_add_kernel("GPUTPCGMMergerPrepareForFit, step0"                    "GPUTPCGMMergerGPU TPCMERGER"                           LB)
o2_gpu_add_kernel("GPUTPCGMMergerPrepareForFit, step1"                    "GPUTPCGMMergerGPU TPCMERGER"                           LB)
o2_gpu_add_kernel("GPUTPCGMMergerPrepareForFit, step2"                    "GPUTPCGMMergerGPU TPCMERGER"                           LB)
o2_gpu_add_kernel("GPUTPCGMMergerFinalize, step0"                         "GPUTPCGMMergerGPU TPCMERGER"                           LB)
o2_gpu_add_kernel("GPUTPCGMMergerFinalize, step1"                         "GPUTPCGMMergerGPU TPCMERGER"                           LB)
o2_gpu_add_kernel("GPUTPCGMMergerFinalize, step2"                         "GPUTPCGMMergerGPU TPCMERGER"                           LB)
o2_gpu_add_kernel("GPUTPCGMMergerMergeLoopers, step0"                     "GPUTPCGMMergerGPU TPCMERGER"                           LB)
o2_gpu_add_kernel("GPUTPCGMMergerMergeLoopers, step1"                     "GPUTPCGMMergerGPU TPCMERGER"                           LB)
o2_gpu_add_kernel("GPUTPCGMMergerMergeLoopers, step2"                     "GPUTPCGMMergerGPU TPCMERGER"                           LB)
o2_gpu_add_kernel("GPUTPCGMO2Output, prepare"                             "= TPCMERGER"                                           LB)
o2_gpu_add_kernel("GPUTPCGMO2Output, sort"                                "= TPCMERGER")
o2_gpu_add_kernel("GPUTPCGMO2Output, output"                              "= TPCMERGER"                                           LB)
o2_gpu_add_kernel("GPUTPCGMO2Output, mc"                                  "= TPCMERGER")
o2_gpu_add_kernel("GPUTRDTrackerKernels, gpuVersion"                      "= TRDTRACKER MATLUT TPCMERGER"                         LB GPUTRDTrackerGPU* externalInstance)
o2_gpu_add_kernel("GPUTRDTrackerKernels, o2Version"                       "= TRDTRACKER MATLUT O2PROPAGATOR"                      LB GPUTRDTracker* externalInstance)
o2_gpu_add_kernel("GPUTPCCompressionKernels, step0attached"               "= TPCCOMPRESSION"                                      LB)
o2_gpu_add_kernel("GPUTPCCompressionKernels, step1unattached"             "= ERRORS"                                              LB)
o2_gpu_add_kernel("GPUTPCCompressionGatherKernels, unbuffered"            "GPUTPCCompressionKernels"                              LB)
o2_gpu_add_kernel("GPUTPCCompressionGatherKernels, buffered32"            "GPUTPCCompressionKernels"                              LB)
o2_gpu_add_kernel("GPUTPCCompressionGatherKernels, buffered64"            "GPUTPCCompressionKernels"                              LB)
o2_gpu_add_kernel("GPUTPCCompressionGatherKernels, buffered128"           "GPUTPCCompressionKernels"                              LB)
o2_gpu_add_kernel("GPUTPCCompressionGatherKernels, multiBlock"            "GPUTPCCompressionKernels"                              LB)
o2_gpu_add_kernel("GPUTPCDecompressionKernels, step0attached"             "= TPCDECOMPRESSION"                                    LB int32_t trackStart int32_t trackEnd)
o2_gpu_add_kernel("GPUTPCDecompressionKernels, step1unattached"           "= TPCDECOMPRESSION"                                    LB int32_t sectorStart int32_t nSectors)
o2_gpu_add_kernel("GPUTPCDecompressionUtilKernels, sortPerSectorRow"      "GPUTPCDecompressionKernels"                            LB)
o2_gpu_add_kernel("GPUTPCDecompressionUtilKernels, countFilteredClusters" "GPUTPCDecompressionKernels"                            LB)
o2_gpu_add_kernel("GPUTPCDecompressionUtilKernels, storeFilteredClusters" "GPUTPCDecompressionKernels"                            LB)
o2_gpu_add_kernel("GPUTPCCFCheckPadBaseline"                              "= TPCCLUSTERFINDER"                                    LB)
o2_gpu_add_kernel("GPUTPCCFHIPTailConnector"                              "GPUTPCCFCheckPadBaseline TPCCLUSTERFINDER"             LB)
o2_gpu_add_kernel("GPUTPCCFHIPClusterizer"                                "GPUTPCCFCheckPadBaseline TPCCLUSTERFINDER"             LB)
o2_gpu_add_kernel("GPUTPCCFChargeMapFiller, fillIndexMap"                 "= TPCCLUSTERFINDER"                                    LB)
o2_gpu_add_kernel("GPUTPCCFChargeMapFiller, fillFromDigits"               "= TPCCLUSTERFINDER"                                    LB)
o2_gpu_add_kernel("GPUTPCCFChargeMapFiller, findFragmentStart"            "= TPCCLUSTERFINDER"                                    LB int8_t setPositions)
o2_gpu_add_kernel("GPUTPCCFPeakFinder"                                    "= TPCCLUSTERFINDER"                                    LB)
o2_gpu_add_kernel("GPUTPCCFNoiseSuppression, noiseSuppression"            "= TPCCLUSTERFINDER"                                    LB)
o2_gpu_add_kernel("GPUTPCCFNoiseSuppression, updatePeaks"                 "= TPCCLUSTERFINDER"                                    LB)
o2_gpu_add_kernel("GPUTPCCFDeconvolution"                                 "= TPCCLUSTERFINDER"                                    LB uint8_t overwriteCharge)
o2_gpu_add_kernel("GPUTPCCFClusterizer"                                   "= TPCCLUSTERFINDER"                                    LB int8_t onlyMC)
o2_gpu_add_kernel("GPUTPCCFMCLabelFlattener, setRowOffsets"               "= TPCCLUSTERFINDER")
o2_gpu_add_kernel("GPUTPCCFMCLabelFlattener, flatten"                     "= TPCCLUSTERFINDER"                                    NO GPUTPCLinearLabels* out)
o2_gpu_add_kernel("GPUTPCCFStreamCompaction, scanStart"                   "= TPCCLUSTERFINDER"                                    LB int32_t iBuf int32_t stage)
o2_gpu_add_kernel("GPUTPCCFStreamCompaction, scanUp"                      "= TPCCLUSTERFINDER"                                    LB int32_t iBuf int32_t nElems)
o2_gpu_add_kernel("GPUTPCCFStreamCompaction, scanTop"                     "= TPCCLUSTERFINDER"                                    LB int32_t iBuf int32_t nElems)
o2_gpu_add_kernel("GPUTPCCFStreamCompaction, scanDown"                    "= TPCCLUSTERFINDER"                                    LB int32_t iBuf uint32_t offset int32_t nElems)
o2_gpu_add_kernel("GPUTPCCFStreamCompaction, compactDigits"               "= TPCCLUSTERFINDER"                                    LB int32_t iBuf int32_t stage CfChargePos* in CfChargePos* out)
o2_gpu_add_kernel("GPUTPCCFDecodeZS"                                      "= TPCCLUSTERFINDER"                                    LB int32_t firstHBF int32_t tpcTimeBinCut)
o2_gpu_add_kernel("GPUTPCCFDecodeZSLink"                                  "GPUTPCCFDecodeZS"                                      LB int32_t firstHBF int32_t tpcTimeBinCut)
o2_gpu_add_kernel("GPUTPCCFDecodeZSDenseLink"                             "GPUTPCCFDecodeZS ERRORS"                               LB int32_t firstHBF int32_t tpcTimeBinCut)
o2_gpu_add_kernel("GPUTPCCFGather"                                        "="                                                     LB o2::tpc::ClusterNative* dest)
o2_gpu_add_kernel("GPUTrackingRefitKernel, mode0asGPU"                    "= GLOBALREFIT "                                        LB)
o2_gpu_add_kernel("GPUTrackingRefitKernel, mode1asTrackParCov"            "= GLOBALREFIT "                                        LB)
if(onnxruntime_FOUND)
o2_gpu_add_kernel("GPUTPCNNClusterizerKernels, runCfClusterizer"          "= TPCNNCLUSTERFINDER"                                  LB uint8_t sector int8_t dtype int8_t withMC uint32_t batchStart)
o2_gpu_add_kernel("GPUTPCNNClusterizerKernels, fillInputNNCPU"            "= TPCNNCLUSTERFINDER"                                  LB uint8_t sector int8_t dtype int8_t withMC uint32_t batchStart)
o2_gpu_add_kernel("GPUTPCNNClusterizerKernels, fillInputNNGPU"            "= TPCNNCLUSTERFINDER"                                  LB uint8_t sector int8_t dtype int8_t withMC uint32_t batchStart)
o2_gpu_add_kernel("GPUTPCNNClusterizerKernels, determineClass1Labels"     "= TPCNNCLUSTERFINDER"                                  LB uint8_t sector int8_t dtype int8_t withMC uint32_t batchStart)
o2_gpu_add_kernel("GPUTPCNNClusterizerKernels, determineClass2Labels"     "= TPCNNCLUSTERFINDER"                                  LB uint8_t sector int8_t dtype int8_t withMC uint32_t batchStart)
o2_gpu_add_kernel("GPUTPCNNClusterizerKernels, publishClass1Regression"   "= TPCNNCLUSTERFINDER"                                  LB uint8_t sector int8_t dtype int8_t withMC uint32_t batchStart)
o2_gpu_add_kernel("GPUTPCNNClusterizerKernels, publishClass2Regression"   "= TPCNNCLUSTERFINDER"                                  LB uint8_t sector int8_t dtype int8_t withMC uint32_t batchStart)
o2_gpu_add_kernel("GPUTPCNNClusterizerKernels, publishDeconvolutionFlags" "= TPCNNCLUSTERFINDER"                                  LB uint8_t sector int8_t dtype int8_t withMC uint32_t batchStart)
endif()

o2_gpu_kernel_add_parameter(NEIGHBOURS_FINDER_MAX_NNEIGHUP  # Number of neighhbours finder hits to cache in shared memory
                            NEIGHBOURS_FINDER_UNROLL_GLOBAL # Unroll factor for neighbours finder iterating hits in local memory
                            NEIGHBOURS_FINDER_UNROLL_SHARED # Fully unroll iteration over neighbours finder hits in shared memory [0/1]
                            TRACKLET_SELECTOR_HITS_REG_SIZE # Number of hits to cache in shared memory in tracklet selector
                            ALTERNATE_BORDER_SORT           # Use alternative border sort approach [0/1]
                            SORT_BEFORE_FIT                 # Sort tracks after length to reduce warp serialization [0/1]
                            NO_ATOMIC_PRECHECK              # Skip atomic precheck to reduce posterior synchronization [0/1]
                            COMP_GATHER_KERNEL              # Default kernel to use for Compression Gather Operation [0 - 4]
                            COMP_GATHER_MODE                # TPC Compression Gather Mode [0 - 3]
                            SORT_STARTHITS                  # Sort start hits to improve cache locality during tracklet construction [0/1]
                            CF_SCAN_WORKGROUP_SIZE          # Work group size to use in clusterizer scan operation
			    AMD_EUS_PER_CU)	  	    # Number of SIMD units per Compute Unit (only for AMD GPUs)

o2_gpu_kernel_add_string_parameter(DEDX_STORAGE_TYPE                # Data type to use for intermediate storage of dEdx truncated mean inputs
                                   MERGER_INTERPOLATION_ERROR_TYPE) # Data type for storing intermediate track residuals for interpolation

o2_gpu_kernel_requires_1_warp("GPUTPCCFDecodeZSLink")
o2_gpu_kernel_requires_1_warp("GPUTPCCFDecodeZSDenseLink")
