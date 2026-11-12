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

/// \file GPUSettingsListStandalone.h
/// \author David Rohr

// clang-format off
BeginNamespace(o2::gpu)

#ifndef GPUCA_GPUCODE_DEVICE
// Settings concerning the standlone timeframe from run 2 events assembly tool
BeginSubConfig(GPUSettingsTFSim, TF, configStandalone, "TF", 't', "Timeframe settings")
AddOption(nMerge, int32_t, 0, "", 0, "Merge n events in a timeframe", min(0))
AddOption(averageDistance, float, 50.f, "", 0, "Average distance in cm of events merged into timeframe", min(0.f))
AddOption(randomizeDistance, bool, true, "", 0, "Randomize distance around average distance of merged events")
AddOption(shiftFirstEvent, bool, true, "", 0, "Also shift the first event in z when merging events to a timeframe")
AddOption(bunchSim, int32_t, 0, "", 0, "Simulate correct bunch interactions instead of placing only the average number of events. A value [n] > 1 sets TFlen for [n] collisions in average. (Incompatible to TFmerge)")
AddOption(bunchCount, int32_t, 12, "", 0, "Number of bunches per trainsort")
AddOption(bunchSpacing, int32_t, 50, "", 0, "Spacing between benches in ns")
AddOption(bunchTrainCount, int32_t, 48, "", 0, "Number of bunch trains")
AddOption(abortGapTime, int32_t, (3000), "", 0, "Length of abort gap in ns")
AddOption(interactionRate, int32_t, 50000, "", 0, "Instantaneous interaction rate")
AddOption(timeFrameLen, int64_t, (1000000000 / 44), "", 'l', "Timeframe len in ns")
AddOption(noBorder, bool, false, "", 0, "Do not simulate border effects (partial events)")
AddOption(noEventRepeat, int32_t, 0, "", 0, "0: Place random events, 1: Place events in timeframe one after another, 2: Place random events but do not repat", def(1))
AddOption(nTotalEventsInTF, int32_t, 0, "", 0, "Total number of collisions to be placed in the interior of all time frames (excluding borders)")
AddOption(eventStride, int32_t, 0, "", 0, "Do not select random event, but walk over array of events in stride steps")
AddOption(overlayRaw, bool, false, "", 0, "Overlay raw TPC data instead of spatial clusters")
AddHelp("help", 'h')
EndConfig()

// Settings concerning standalone toy event generator
BeginSubConfig(GPUSettingsEG, EG, configStandalone, "EG", 0, "Event generator settings")
AddOption(numberOfTracks, int32_t, 1, "", 0, "Number of tracks per generated event")
AddHelp("help", 'h')
EndConfig()

// Settings for the standalone benchmark
BeginConfig(GPUSettingsStandalone, configStandalone)
AddOption(runGPU, uint8_t, 1, "", 'g', "Use GPU for processing", message("GPU processing enabled"), set(2))
AddOptionSet(runGPU, uint8_t, 0, "", 'c', "Use CPU for processing", message("CPU enabled"))
AddOption(gpuType, std::string, "AUTO", "", 0, "GPU type (CUDA / HIP / OCL / METAL / CPUDEVICE) or CPU or AUTO")
AddOption(runGPUforce, bool, true, "", 0, "Force usage of the specified GPU device type, no CPU fallback")
AddOption(promptOnExit, bool, false, "", 0, "Prompt for keypress before exiting")
AddOption(continueOnError, bool, false, "", 0, "Continue processing after an error")
AddOption(seed, int32_t, -1, "", 0, "Set srand seed (-1: random)")
AddOption(StartEvent, int32_t, 0, "", 's', "First event to process", min(0))
AddOption(nEvents, int32_t, -1, "", 'n', "Number of events to process (-1; all)", min(0))
AddOption(runs, int32_t, 1, "runs", 'r', "Number of iterations to perform (repeat each event)", min(0))
AddOption(runs2, int32_t, 1, "runsExternal", 0, "Number of iterations to perform (repeat full processing)", min(1))
AddOption(runsInit, int32_t, 1, "", 0, "Number of initial iterations excluded from average", min(0))
AddOption(eventsDir, const char*, "pp", "events", 'e', "Directory with events to process", message("Reading events from Directory %s"))
AddOption(absoluteEventsDir, bool, false, "", 0, "Events directory is absolute, and not inside './events/'")
AddOption(noEvents, bool, false, "", 0, "Run without data (e.g. for field visualization)")
AddOption(eventDisplay, int32_t, 0, "display", 'd', "Show standalone event display", def(1))
AddOption(eventGenerator, bool, false, "", 0, "Run event generator")
AddOption(cont, bool, false, "", 0, "Process continuous timeframe data, even if input is triggered")
AddOption(setMaxTimeBin, int32_t, -2, "", 0, "maximum time bin of continuous data, 0 for triggered events, -1 for automatic continuous mode, -2 for automatic continuous / triggered")
AddOption(outputcontrolmem, uint64_t, 0, "outputMemory", 0, "Use predefined output buffer of this size", min(uint64_t(0)), message("Using %s bytes as output memory"))
AddOption(inputcontrolmem, uint64_t, 0, "inputMemory", 0, "Use predefined input buffer of this size", min(uint64_t(0)), message("Using %s bytes as input memory"))
AddOption(cpuAffinity, int32_t, -1, "", 0, "Pin CPU affinity to this CPU core", min(-1))
AddOption(fifoScheduler, bool, false, "", 0, "Use FIFO realtime scheduler", message("Setting FIFO scheduler: %s"))
AddOption(fpe, int8_t, -1, "", 0, "Trap on floating point exceptions (-1 = if no ffast-math)")
AddOption(flushDenormals, int8_t, -1, "", 0, "Enable FTZ and DAZ (Flush all denormals to zero), -1 = enable automatically if not prevented by deterministic mode")
AddOption(solenoidBzNominalGPU, float, -1e6f, "", 0, "Field strength of solenoid Bz in kGaus")
AddOption(constBz, bool, false, "", 0, "Force constand Bz")
AddOption(overrideMaxTimebin, bool, false, "", 0, "Override max time bin setting for continuous data with max time bin in time frame")
AddOption(encodeZS, int32_t, -1, "", 0, "Zero-Suppress TPC data", def(1))
AddOption(zsFilter, int32_t, -1, "", 0, "Apply Zero-Suppression when loading digits and remove those below threshold", def(1))
AddOption(zsVersion, int32_t, 2, "", 0, "ZS Version: 1 = 10-bit ADC row based, 2 = 12-bit ADC row based, 3 = improved link based, 4 = dense link based")
AddOption(dumpEvents, bool, false, "", 0, "Dump events (after transformation such as encodeZS")
AddOption(stripDumpedEvents, bool, false, "", 0, "Remove redundant inputs (e.g. digits and ZS) before dumping")
AddOption(printSettings, int32_t, 0, "", 0, "Print all settings", def(1))
AddOption(testSyncAsync, bool, false, "syncAsync", 0, "Test first synchronous and then asynchronous processing")
AddOption(testSync, bool, false, "sync", 0, "Test settings for synchronous phase")
AddOption(testSyncAsyncQcInSync, bool, false, "syncAsyncSyncQC", 0, "Run QC in sync phase of testSyncAsync")
AddOption(timeFrameTime, bool, false, "tfTime", 0, "Print some debug information about time frame processing time")
AddOption(controlProfiler, bool, false, "", 0, "Issues GPU profiler stop and start commands to profile only the relevant processing part")
AddOption(preloadEvents, bool, false, "", 0, "Preload events into host memory before start processing")
AddOption(recoSteps, int32_t, -1, "", 0, "Bitmask for RecoSteps. Run --listRecoSteps to get a list")
AddOption(recoStepsGPU, int32_t, -1, "", 0, "Bitmask for RecoSteps to run on GPU")
AddOption(listRecoSteps, bool, false, "", 0, "Print the list of RecoSteps with their bitmask values")
AddOption(runMerger, int32_t, 1, "", 0, "Run track merging / refit", min(0), max(1))
AddOption(runTRD, int32_t, -1, "", 0, "Enable TRD processing (-1 = decide automatically)")
AddOption(rundEdx, int32_t, -1, "", 0, "Enable dEdx processing (-1 = decide automatically)")
AddOption(runCompression, int32_t, 1, "", 0, "Enable TPC Compression")
AddOption(runTransformation, int32_t, 1, "", 0, "Enable TPC Transformation")
AddOption(runRefit, bool, false, "", 0, "Enable final track refit")
AddOption(setO2Settings, bool, false, "", 0, "Set O2 defaults for output of shared cluster map, referenceX")
AddOption(recreateTrivialCalibObjects, int64_t, 0, "", 0, "Recreate trivial calibration objects (TPCTransform, dEdxCorrection, ...) from scratch, O2 build: CCDB objects (material LUT) for the current time, or for this timestamp (ms) if != 1", def(1))
AddHelp("help", 'h')
AddHelpAll("helpall", 'H')
AddSubConfig(GPUSettingsRec, rec)
AddSubConfig(GPUSettingsProcessing, proc)
AddSubConfig(GPUSettingsQA, QA)
AddSubConfig(GPUSettingsDisplay, display)
AddSubConfig(GPUSettingsTFSim, TF)
AddSubConfig(GPUSettingsEG, EG)
EndConfig()
#endif

EndNamespace() // o2::gpu
