// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUSettingsList.h
/// \author David Rohr

// This file contains macros to generate all settings for the GPU Reconstruction.
// Macros are used in the following places:
// Create ConfigurableParam object for workflow.
// Configure standalone benchmark.
// Create plain-C struct for GPU code.
// Create static constexpr with default values for GPU run time compilation

#include "GPUDefConstantsAndSettings.h"
#include "GPUSettings.h"

// clang-format off

#ifdef QCONFIG_INSTANCE
using namespace GPUCA_NAMESPACE::gpu;
#endif
#ifdef BeginNamespace // File should not be included without defining the macros, but rootcling will do for dictionary generation
BeginNamespace(GPUCA_NAMESPACE)
BeginNamespace(gpu)

// Settings concerning the reconstruction
// There must be no bool in here, use char, as sizeof(bool) is compiler dependent and fails on GPUs!!!!!!
BeginSubConfig(GPUSettingsRec, rec, configStandalone, "REC", 0, "Reconstruction settings")
AddOption(tpcRejectQPt, float, 1.f / 0.05f, "", 0, "QPt threshold to reject clusters of TPC tracks (Inverse Pt!!!)")
AddOption(HitPickUpFactor, float, 2., "", 0, "multiplier for the chi2 window for hit pick up procedure")
AddOption(NeighboursSearchArea, float, 3., "", 0, "area in cm for the search of neighbours")
AddOption(ClusterError2CorrectionY, float, 1., "", 0, "correction for the squared cluster error during tracking")
AddOption(ClusterError2CorrectionZ, float, 1., "", 0, "correction for the squared cluster error during tracking")
AddOption(MinNTrackClusters, int, -1, "", 0, "required min number of clusters on the track")
AddOption(MaxTrackQPt, float, 1.f / GPUCA_MIN_TRACK_PT_DEFAULT, "", 0, "required max Q/Pt (==min Pt) of tracks")
AddOption(SearchWindowDZDR, float, 2.5, "", 0, "Use DZDR window for seeding instead of vertex window")
AddOption(TrackReferenceX, float, 1000.f, "", 0, "Transport all tracks to this X after tracking (disabled if > 500, auto = 1000)")
AddOption(tpcZSthreshold, float, 2.0f, "", 0, "Zero-Suppression threshold")
AddOption(tpcCFqmaxCutoff, unsigned char, 3, "", 0, "Cluster Finder rejects cluster with qmax below this threshold")
AddOption(tpcCFqtotCutoff, unsigned char, 0, "", 0, "Cluster Finder rejects cluster with qtot below this threshold")
AddOption(tpcCFinnerThreshold, unsigned char, 0, "", 0, "Cluster Finder extends cluster if inner charge above this threshold")
AddOption(tpcCFminSplitNum, unsigned char, 1, "", 0, "Minimum number of split charges in a cluster for the cluster to be marked as split")
AddOption(tpcCFnoiseSuppressionEpsilon, unsigned char, 10, "", 0, "Cluster Finder: Difference between peak and charge for the charge to count as a minima during noise suppression")
AddOption(NWays, char, 3, "", 0, "Do N fit passes in final fit of merger")
AddOption(NWaysOuter, char, 0, "", 0, "Store outer param")
AddOption(RejectMode, char, 5, "", 0, "0: no limit on rejection or missed hits, >0: break after n rejected hits, <0: reject at max -n hits")
AddOption(NonConsecutiveIDs, char, false, "", 0, "Non-consecutive cluster IDs as in HLT, disables features that need access to slice data in TPC merger")
AddOption(dEdxTruncLow, unsigned char, 2, "", 0, "Low truncation threshold, fraction of 128")
AddOption(dEdxTruncHigh, unsigned char, 77, "", 0, "High truncation threshold, fraction of 128")
AddOption(GlobalTracking, char, 1, "", 0, "Enable Global Tracking (prolong tracks to adjacent sectors to find short segments)")
AddOption(DisableRefitAttachment, unsigned char, 0, "", 0, "Bitmask to disable certain attachment steps during refit (1: attachment, 2: propagation, 4: loop following, 8: mirroring)")
AddOption(tpcRejectionMode, unsigned char, GPUCA_NAMESPACE::gpu::GPUSettings::RejectionStrategyA, "", 0, "Enable rejection of TPC clusters for compression (0 = no, 1 = strategy A, 2 = strategy B)")
AddOption(tpcCompressionModes, unsigned char, GPUCA_NAMESPACE::gpu::GPUSettings::CompressionFull, "", 0, "TPC Compression mode bits (1=truncate charge/width LSB, 2=differences, 4=track-model)")
AddOption(tpcCompressionSortOrder, unsigned char, GPUCA_NAMESPACE::gpu::GPUSettings::SortTime, "", 0, "Sort order of TPC compression (0 = time, 1 = pad, 2 = Z-time-pad, 3 = Z-pad-time, 4 = no sorting (use incoming order))")
AddOption(tpcSigBitsCharge, unsigned char, 4, "", 0, "Number of significant bits for TPC cluster charge in compression mode 1")
AddOption(tpcSigBitsWidth, unsigned char, 3, "", 0, "Number of significant bits for TPC cluster width in compression mode 1")
AddOption(ForceEarlyTPCTransform, char, -1, "", 0, "Force early TPC transformation also for continuous data (-1 = auto)")
AddOption(fwdTPCDigitsAsClusters, unsigned char, 0, "", 0, "Forward TPC digits as clusters (if they pass the ZS threshold)")
AddOption(bz0Pt, unsigned char, 60, "", 0, "Nominal Pt to set when bz = 0 (in 10 MeV)")
AddOption(dropLoopers, unsigned char, 0, "", 0, "Drop looping tracks starting from second loop")
AddOption(mergerCovSource, unsigned char, 2, "", 0, "Method to obtain covariance in track merger: 0 = simple filterErrors method, 1 = use cov from track following, 2 = refit")
AddOption(mergerInterpolateErrors, unsigned char, 1, "", 0, "Use interpolation instead of extrapolation for chi2 based cluster rejection")
AddOption(fitInProjections, char, -1, "", 0, "Fit in projection, -1 to enable for all but passes but the first one")
AddOption(fitPropagateBzOnly, char, -1, "", 0, "Propagate using Bz only for n passes")
AddOption(retryRefit, char, 1, "", 0, "Retry refit when fit fails")
AddOption(loopInterpolationInExtraPass, char, -1, "", 0, "Perform loop interpolation in an extra pass")
AddOption(mergerReadFromTrackerDirectly, char, 1, "", 0, "Forward data directly from tracker to merger on GPU")
AddOption(useMatLUT, char, 0, "", 0, "Use material lookup table for TPC refit")
AddCustomCPP(void SetMinTrackPt(float v) { MaxTrackQPt = v > 0.001 ? (1. / v) : (1. / 0.001); })
AddHelp("help", 'h')
EndConfig()

// Settings steering the processing once the device was selected
BeginSubConfig(GPUSettingsProcessing, proc, configStandalone, "PROC", 0, "Processing settings")
AddOption(platformNum, int, -1, "", 0, "Platform to use, in case the backend provides multiple platforms (-1 = auto-select)")
AddOption(gpuDeviceOnly, bool, false, "", 0, "Use only GPU as device (i.e. no CPU for OpenCL)")
AddOption(debugMask, int, -1, "", 0, "Mask for debug output dumps to file")
AddOption(comparableDebutOutput, bool, true, "", 0, "Make CPU and GPU debug output comparable (sort / skip concurrent parts)")
AddOption(resetTimers, int, 1, "", 0, "Reset timers every event")
AddOption(stuckProtection, int, 0, "", 0, "Timeout in us, When AMD GPU is stuck, just continue processing and skip tracking, do not crash or stall the chain")
AddOption(keepAllMemory, bool, false, "", 0, "Allocate all memory on both device and host, and do not reuse")
AddOption(keepDisplayMemory, bool, false, "", 0, "Like keepAllMemory, but only for memory required for event display")
AddOption(trackletSelectorSlices, char, -1, "", 0, "Number of slices to processes in parallel at max")
AddOption(nTPCClustererLanes, int, 3, "", 0, "Number of TPC clusterers that can run in parallel")
AddOption(disableMemoryReuse, bool, false, "", 0, "Disable memory reusage (for debugging only)")
AddOption(fullMergerOnGPU, bool, true, "", 0, "Perform full TPC track merging on GPU instead of only refit")
AddOption(delayedOutput, bool, true, "", 0, "Delay output to be parallel to track fit")
AddOption(tpccfGatherKernel, bool, true, "", 0, "Use a kernel instead of the DMA engine to gather the clusters")
AddOption(doublePipelineClusterizer, bool, true, "", 0, "Include the input data of the clusterizer in the double-pipeline")
AddOption(deviceNum, int, -1, "gpuDevice", 0, "Set GPU device to use (-1: automatic, -2: for round-robin usage in timeslice-pipeline)")
AddOption(globalInitMutex, bool, false, "", 0, "Use global mutex to synchronize initialization of multiple GPU instances")
AddOption(ompThreads, int, -1, "omp", 't', "Number of OMP threads to run (-1: all)", min(-1), message("Using %d OMP threads"))
AddOption(nDeviceHelperThreads, int, 1, "", 0, "Number of CPU helper threads for CPU processing")
AddOption(nStreams, int, 8, "", 0, "Number of GPU streams / command queues")
AddOption(trackletConstructorInPipeline, int, -1, "", 0, "Run tracklet constructor in the pipeline")
AddOption(trackletSelectorInPipeline, int, -1, "", 0, "Run tracklet selector in the pipeline")
AddOption(mergerSortTracks, int, -1, "", 0, "Sort track indizes for GPU track fit")
AddOption(tpcCompressionGatherMode, int, -1, "", 0, "TPC Compressed Clusters Gather Mode")
AddOption(tpcCompressionGatherModeKernel, int, -1, "", 0, "TPC Compressed Clusters Gather Mode Kernel")
AddOption(runMC, bool, false, "", 0, "Process MC labels")
AddOption(ompKernels, bool, true, "", 0, "Parallelize with OMP inside kernels instead of over slices")
AddOption(doublePipeline, bool, false, "", 0, "Double pipeline mode")
AddOption(prefetchTPCpageScan, int, 0, "", 0, "Prefetch Data for TPC page scan in CPU cache")
AddOption(debugLevel, int, -1, "debug", 'd', "Set debug level (-1 = silend)")
AddOption(allocDebugLevel, int, 0, "allocDebug", 0, "Some debug output for memory allocations (without messing with normal debug level)")
AddOption(runQA, bool, false, "qa", 'q', "Enable tracking QA", message("Running QA: %s"))
AddOption(runCompressionStatistics, bool, false, "compressionStat", 0, "Run statistics and verification for cluster compression")
AddOption(forceMemoryPoolSize, unsigned long, 1, "memSize", 0, "Force size of allocated GPU / page locked host memory", min(0ul))
AddOption(forceHostMemoryPoolSize, unsigned long, 0, "hostMemSize", 0, "Force size of allocated host page locked host memory (overriding memSize)", min(0ul))
AddOption(memoryAllocationStrategy, int, 0, "", 0, "Memory Allocation Stragegy (0 = auto, 1 = individual allocations, 2 = single global allocation)")
AddOption(deviceTimers, bool, true, "", 0, "Use device timers instead of host-based time measurement")
AddOption(registerStandaloneInputMemory, bool, false, "registerInputMemory", 0, "Automatically register input memory buffers for the GPU")
AddOption(memoryScalingFactor, float, 1.f, "", 0, "Factor to apply to all memory scalers")
AddOption(alternateBorderSort, int, -1, "", 0, "Alternative implementation for sorting of border tracks")
AddOption(enableRTC, bool, false, "", 0, "Use RTC to optimize GPU code")
AddOption(showOutputStat, bool, false, "", 0, "Print some track output statistics")
AddVariable(eventDisplay, GPUCA_NAMESPACE::gpu::GPUDisplayBackend*, nullptr)
AddHelp("help", 'h')
EndConfig()

#ifndef GPUCA_GPUCODE_DEVICE
// Settings concerning the event display
BeginSubConfig(GPUSettingsDisplay, GL, configStandalone, "GL", 'g', "OpenGL display settings")
AddOption(clustersOnly, bool, false, "", 0, "Visualize clusters only")
AddHelp("help", 'h')
EndConfig()

// Settings concerning the standalone QA
BeginSubConfig(GPUSettingsQA, QA, configStandalone, "QA", 'q', "QA settings")
AddOptionVec(compareInputs, const char*, "QAinput", 0, "Read histogram from these input files and include them in the output plots")
AddOptionVec(compareInputNames, const char*, "QAinputName", 0, "Legend entries for data from comparison histogram files")
AddOption(name, std::string, "", "", 0, "Legend entry for new data from current processing")
AddOption(output, std::string, "", "", 0, "Store histograms in output root file", def(std::string("histograms.root")))
AddOption(inputHistogramsOnly, bool, false, "", 0, "Do not run tracking, but just create PDFs from input root files")
AddOption(strict, bool, true, "", 0, "Strict QA mode: Only consider resolution of tracks where the fit ended within 5 cm of the reference, and remove outliers.")
AddOption(qpt, float, 10.f, "", 0, "Set cut for Q/Pt", def(2.f))
AddOption(recThreshold, float, 0.9f, "", 0, "Compute the efficiency including impure tracks with fake contamination")
AddOption(csvDump, bool, false, "", 0, "Dump all clusters and Pt information into csv file")
AddOption(maxResX, float, 1e6f, "", 0, "Maxmimum X (~radius) for reconstructed track position to take into accound for resolution QA in cm")
AddOption(resPrimaries, int, 0, "", 0, "0: Resolution for all tracks, 1: only for primary tracks, 2: only for non-primaries", def(1))
AddOption(nativeFitResolutions, bool, false, "", 0, "Create resolution histograms in the native fit units (sin(phi), tan(lambda), Q/Pt)")
AddOption(filterCharge, int, 0, "", 0, "Filter for positive (+1) or negative (-1) charge")
AddOption(filterPID, int, -1, "", 0, "Filter for Particle Type (0 Electron, 1 Muon, 2 Pion, 3 Kaon, 4 Proton)")
AddOption(writeMCLabels, bool, false, "", 0, "Store mc labels to file for later matching")
AddOptionVec(matchMCLabels, const char*, "", 0, "Read labels from files and match them, only process tracks where labels differ")
AddOption(matchDisplayMinPt, float, 0, "", 0, "Minimum Pt of a matched track to be displayed")
AddOption(writeRootFiles, bool, false, "", 0, "Create ROOT canvas files")
AddShortcut("compare", 0, "--QAinput", "Compare QA histograms", "--qa", "--QAinputHistogramsOnly")
AddHelp("help", 'h')
EndConfig()

#ifdef GPUCA_STANDALONE
// Settings concerning the standlone timeframe from run 2 events assembly tool
BeginSubConfig(GPUSettingsTF, TF, configStandalone, "TF", 't', "Timeframe settings")
AddOption(nMerge, int, 0, "", 0, "Merge n events in a timeframe", min(0))
AddOption(averageDistance, float, 50., "", 0, "Average distance in cm of events merged into timeframe", min(0.f))
AddOption(randomizeDistance, bool, true, "", 0, "Randomize distance around average distance of merged events")
AddOption(shiftFirstEvent, bool, true, "", 0, "Also shift the first event in z when merging events to a timeframe")
AddOption(bunchSim, int, 0, "", 0, "Simulate correct bunch interactions instead of placing only the average number of events. A value [n] > 1 sets TFlen for [n] collisions in average. (Incompatible to TFmerge)")
AddOption(bunchCount, int, 12, "", 0, "Number of bunches per trainsort")
AddOption(bunchSpacing, int, 50, "", 0, "Spacing between benches in ns")
AddOption(bunchTrainCount, int, 48, "", 0, "Number of bunch trains")
AddOption(abortGapTime, int, (3000), "", 0, "Length of abort gap in ns")
AddOption(interactionRate, int, 50000, "", 0, "Instantaneous interaction rate")
AddOption(timeFrameLen, long, (1000000000 / 44), "", 'l', "Timeframe len in ns")
AddOption(noBorder, bool, false, "", 0, "Do not simulate border effects (partial events)")
AddOption(noEventRepeat, int, 0, "", 0, "0: Place random events, 1: Place events in timeframe one after another, 2: Place random events but do not repat", def(1))
AddOption(nTotalEventsInTF, int, 0, "", 0, "Total number of collisions to be placed in the interior of all time frames (excluding borders)")
AddOption(eventStride, int, 0, "", 0, "Do not select random event, but walk over array of events in stride steps")
AddOption(overlayRaw, bool, false, "", 0, "Overlay raw TPC data instead of spatial clusters")
AddHelp("help", 'h')
EndConfig()

// Settings concerning standalone toy event generator
BeginSubConfig(GPUSettingsEG, EG, configStandalone, "EG", 0, "Event generator settings")
AddOption(numberOfTracks, int, 1, "", 0, "Number of tracks per generated event")
AddHelp("help", 'h')
EndConfig()

// Settings for the standalone benchmark
BeginConfig(GPUSettingsStandalone, configStandalone)
#if defined(CUDA_ENABLED) || defined(OPENCL1_ENABLED) || defined(OPENCL2_ENABLED) || defined(HIP_ENABLED)
AddOption(runGPU, bool, true, "", 'g', "Use GPU for processing", message("GPU processing: %s"))
#else
AddOption(runGPU, bool, false, "", 'g', "Use GPU for processing", message("GPU processing: %s"))
#endif
AddOptionSet(runGPU, bool, false, "", 'c', "Use CPU for processing", message("CPU enabled"))
#if defined(CUDA_ENABLED)
AddOption(gpuType, const char*, "CUDA", "", 0, "GPU type (CUDA / HIP / OCL / OCL2)")
#elif defined(OPENCL2_ENABLED)
AddOption(gpuType, const char*, "OCL2", "", 0, "GPU type (CUDA / HIP / OCL / OCL2)")
#elif defined(OPENCL1_ENABLED)
AddOption(gpuType, const char*, "OCL", "", 0, "GPU type (CUDA / HIP / OCL / OCL2)")
#elif defined(HIP_ENABLED)
AddOption(gpuType, const char*, "HIP", "", 0, "GPU type (CUDA / HIP / OCL / OCL2)")
#else
AddOption(gpuType, const char*, "", "", 0, "GPU type (CUDA / HIP / OCL / OCL2)")
#endif
AddOption(runGPUforce, bool, true, "", 0, "Force usage of the specified GPU device type, no CPU fallback")
AddOption(noprompt, bool, true, "", 0, "Do prompt for keypress before exiting")
AddOption(continueOnError, bool, false, "", 0, "Continue processing after an error")
AddOption(seed, int, -1, "", 0, "Set srand seed (-1: random)")
AddOption(StartEvent, int, 0, "", 's', "First event to process", min(0))
AddOption(NEvents, int, -1, "", 'n', "Number of events to process (-1; all)", min(0))
AddOption(runs, int, 1, "runs", 'r', "Number of iterations to perform (repeat each event)", min(0))
AddOption(runs2, int, 1, "runsExternal", 0, "Number of iterations to perform (repeat full processing)", min(1))
AddOption(runsInit, int, 1, "", 0, "Number of initial iterations excluded from average", min(0))
AddOption(EventsDir, const char*, "pp", "events", 'e', "Directory with events to process", message("Reading events from Directory events/%s"))
AddOption(eventDisplay, int, 0, "display", 'd', "Show standalone event display", def(1)) //1: default display (Windows / X11), 2: glut, 3: glfw
AddOption(eventGenerator, bool, false, "", 0, "Run event generator")
AddOption(cont, bool, false, "", 0, "Process continuous timeframe data")
AddOption(outputcontrolmem, unsigned long, 0, "outputMemory", 0, "Use predefined output buffer of this size", min(0ul), message("Using %lld bytes as output memory"))
AddOption(inputcontrolmem, unsigned long, 0, "inputMemory", 0, "Use predefined input buffer of this size", min(0ul), message("Using %lld bytes as input memory"))
AddOption(cpuAffinity, int, -1, "", 0, "Pin CPU affinity to this CPU core", min(-1))
AddOption(fifoScheduler, bool, false, "", 0, "Use FIFO realtime scheduler", message("Setting FIFO scheduler: %s"))
AddOption(fpe, bool, true, "", 0, "Trap on floating point exceptions")
AddOption(flushDenormals, bool, true, "", 0, "Enable FTZ and DAZ (Flush all denormals to zero)")
AddOption(solenoidBz, float, -1e6f, "", 0, "Field strength of solenoid Bz in kGaus")
AddOption(constBz, bool, false, "", 0, "Force constand Bz")
AddOption(overrideMaxTimebin, bool, false, "", 0, "Override max time bin setting for continuous data with max time bin in time frame")
AddOption(encodeZS, int, -1, "", 0, "Zero-Suppress TPC data", def(1))
AddOption(zsFilter, int, -1, "", 0, "Apply Zero-Suppression when loading digits and remove those below threshold", def(1))
AddOption(zs12bit, bool, true, "", 0, "Perform 12 bit zero-suppression encoding / filter")
AddOption(dumpEvents, bool, false, "", 0, "Dump events (after transformation such as encodeZS")
AddOption(stripDumpedEvents, bool, false, "", 0, "Remove redundant inputs (e.g. digits and ZS) before dumping")
AddOption(printSettings, bool, false, "", 0, "Print all settings")
AddOption(memoryStat, bool, false, "", 0, "Print memory statistics")
AddOption(testSyncAsync, bool, false, "syncAsync", 0, "Test first synchronous and then asynchronous processing")
AddOption(testSync, bool, false, "sync", 0, "Test settings for synchronous phase")
AddOption(timeFrameTime, bool, false, "tfTime", 0, "Print some debug information about time frame processing time")
AddOption(controlProfiler, bool, false, "", 0, "Issues GPU profiler stop and start commands to profile only the relevant processing part")
AddOption(preloadEvents, bool, false, "", 0, "Preload events into host memory before start processing")
AddOption(recoSteps, int, -1, "", 0, "Bitmask for RecoSteps")
AddOption(recoStepsGPU, int, -1, "", 0, "Bitmask for RecoSteps")
AddOption(runMerger, int, 1, "", 0, "Run track merging / refit", min(0), max(1))
AddOption(runTRD, int, -1, "", 0, "Enable TRD processing")
AddOption(rundEdx, int, -1, "", 0, "Enable dEdx processing")
AddOption(runCompression, int, 1, "", 0, "Enable TPC Compression")
AddOption(runTransformation, int, 1, "", 0, "Enable TPC Transformation")
AddOption(runRefit, bool, false, "", 0, "Enable final track refit")
AddHelp("help", 'h')
AddHelpAll("helpall", 'H')
AddSubConfig(GPUSettingsRec, rec)
AddSubConfig(GPUSettingsProcessing, proc)
AddSubConfig(GPUSettingsTF, TF)
AddSubConfig(GPUSettingsQA, QA)
AddSubConfig(GPUSettingsDisplay, GL)
AddSubConfig(GPUSettingsEG, EG)
EndConfig()
#elif defined(GPUCA_O2_LIB) || defined(GPUCA_O2_INTERFACE) // GPUCA_STANDALONE
BeginSubConfig(GPUSettingsO2, global, configStandalone, "O2", 0, "O2 workflow settings")
AddOption(solenoidBz, float, -1000.f, "", 0, "solenoid field strength")
AddOption(constBz, bool, false, "", 0, "force constant Bz for tests")
AddOption(continuousMaxTimeBin, int, 0, "", 0, "maximum time bin of continuous data, 0 for triggered events, -1 for default of 23ms")
AddOption(deviceType, std::string, "CPU", "", 0, "Device type, CPU | CUDA | HIP | OCL1 | OCL2")
AddOption(forceDeviceType, bool, true, "", 0, "force device type, otherwise allows fall-back to CPU")
AddOption(dump, int, 0, "", 0, "Dump events for standalone benchmark: 1 = dump events, 2 = dump events and skip processing in workflow")
AddOption(runDisplay, bool, false, "", 0, "Run event visualization after processing")
AddOption(dEdxFile, std::string, "", "", 0, "File name of dEdx Splines file")
AddOption(transformationFile, std::string, "", "", 0, "File name of TPC fast transformation map")
AddOption(matLUTFile, std::string, "", "", 0, "File name of material LUT file")
AddOption(gainCalibFile, std::string, "", "", 0, "File name of TPC pad gain calibration")
AddOption(allocateOutputOnTheFly, bool, true, "", 0, "Allocate shm output buffers on the fly, instead of using preallocated buffer with upper bound size")
AddOption(outputBufferSize, unsigned long, 200000000ul, "", 0, "Size of the output buffers to be allocated")
AddOption(synchronousProcessing, bool, false, "", 0, "Apply performance shortcuts for synchronous processing, disable unneeded steps")
AddOption(mutexMemReg, bool, false, "", 0, "Global mutex to serialize GPU memory registration")
AddOption(display, bool, false, "", 0, "Enable standalone gpu tracking visualizaion")
AddOption(dropSecondaryLegs, bool, true, "", 0, "Do not store secondary legs of looping track in TrackTPC")
EndConfig()
#endif // GPUCA_O2_LIB
#endif // !GPUCA_GPUCODE_DEVICE

EndNamespace() // gpu
EndNamespace() // GPUCA_NAMESPACE
#endif
  // clang-format on
