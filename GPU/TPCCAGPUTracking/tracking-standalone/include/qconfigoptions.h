typedef std::tuple<const char*, int> tupleGpuOpt;

BeginSubConfig(structConfigGL, configGL, configStandalone, "GL", 'g', "OpenGL display settings")
AddOption(clustersOnly, bool, false, "clustersOnly", 0, "Visualize clusters only")
AddHelp("help", 'h')
EndConfig()

BeginSubConfig(structConfigTF, configTF, configStandalone, "TF", 't', "Timeframe settings")
AddOption(nMerge, int, 0, "merge", 0, "Merge n events in a timeframe", min(0))
AddOption(averageDistance, float, 50., "mergeDist", 0, "Average distance of events merged into timeframe", min(0.f))
AddOption(randomizeDistance, bool, true, "mergeRand", 0, "Randomize distance around average distance of merged events")
AddOption(shiftFirstEvent, bool, true, "mergeFirst", 0, "Also shift the first event in z when merging events to a timeframe")
AddOption(bunchSim, bool, false, "simBunches", 0, "Simulate correct bunch interactions instead of placing only the average number of events, ignores the merge parameters")
AddOption(bunchCount, int, 12, "bunchCount", 0, "Number of bunches per train")
AddOption(bunchSpacing, int, 50, "bunchSpacing", 0, "Spacing between benches in ns")
AddOption(bunchTrainCount, int, 48, "bunchTrainCount", 0, "Number of bunch trains")
AddOption(abortGapTime, int, (3000), "abortGap", 0, "Length of abort gap in ns")
AddOption(interactionRate, int, 50000, "rate", 0, "Instantaneous interaction rate")
AddOption(timeFrameLen, long long int, (1000000000 / 44), "len", 'l', "Timeframe len in ns")
AddOption(noEventRepeat, int, 0, "noEventRepeat", 0, "0: Place random events, 1: Place events in timeframe one after another, 2: Place random events but do not repat", def(1))
AddOption(nTotalInTFEvents, int, 0, "nTotal", 0, "Total number of collisions to be placed in the interior of all time frames (excluding borders)")
AddOption(eventStride, int, 0, "eventStride", 0, "Do not select random event, but walk over array of events in stride steps")
AddOption(dumpO2, bool, false, "dumpO2", 0, "Dump time frame for O2 in ClusterHardware format")
AddHelp("help", 'h')
EndConfig()

BeginSubConfig(structConfigQA, configQA, configStandalone, "QA", 'q', "QA settings")
AddOptionVec(compareInputs, const char*, "input", 0, "Read histogram from these input files and include them in the output plots")
AddOptionVec(compareInputNames, const char*, "inputName", 0, "Legend entries for data from comparison histogram files")
AddOption(name, const char*, NULL, "name", 0, "Legend entry for new data from current processing")
AddOption(output, const char*, NULL, "histOut", 0, "Store histograms in output root file", def("histograms.root"))
AddOption(inputHistogramsOnly, bool, false, "only", 0, "Do not run tracking, but just create PDFs from input root files")
AddOption(strict, bool, true, "strict", 0, "Strict QA mode: Only consider resolution of tracks where the fit ended within 5 cm of the reference, and remove outliers.")
AddOption(qpt, float, 10.f, "qpt", 0, "Set cut for Q/Pt", def(2.f))
AddOption(recThreshold, float, 0.9f, "recThreshold", 0, "Compute the efficiency including impure tracks with fake contamination")
AddOption(csvDump, bool, false, "csvDump", 0, "Dump all clusters and Pt information into csv file")
AddOption(maxResX, float, 1e6f, "maxResX", 0, "Maxmimum X (~radius) for reconstructed track position to take into accound for resolution QA in cm")
AddOption(resPrimaries, int, 0, "resPrimaries", 0, "0: Resolution for all tracks, 1: only for primary tracks, 2: only for non-primaries", def(1))
AddOption(nativeFitResolutions, bool, false, "nativeFitResolutions", 0, "Create resolution histograms in the native fit units (sin(phi), tan(lambda), Q/Pt)")
AddOption(filterCharge, int, 0, "filterCharge", 0, "Filter for positive (+1) or negative (-1) charge")
AddOption(filterPID, int, -1, "filterPID", 0, "Filter for Particle Type (0 Electron, 1 Muon, 2 Pion, 3 Kaon, 4 Proton)")
AddOption(writeMCLabels, bool, false, "writeLabels", 0, "Store mc labels to file for later matching")
AddOptionVec(matchMCLabels, const char*, "matchLabels", 0, "Read labels from files and match them, only process tracks where labels differ")
AddOption(matchDisplayMinPt, float, 0, "matchDisplayMinPt", 0, "Minimum Pt of a matched track to be displayed")
AddShortcut("compare", 0, "--QAinput", "Compare QA histograms", "--qa", "--QAonly")
AddHelp("help", 'h')
EndConfig()

BeginSubConfig(structConfigEG, configEG, configStandalone, "EG", 0, "Event generator settings")
AddOption(numberOfTracks, int, 1, "numberOfTracks", 0, "Number of tracks per generated event")
AddHelp("help", 'h')
EndConfig()

BeginConfig(structConfigStandalone, configStandalone)
AddOption(runGPU, bool, true, "gpu", 'g', "Use GPU for processing", message("GPU processing: %s"))
AddOptionSet(runGPU, bool, false, "cpu", 'c', "Use CPU for processing", message("CPU enabled"))
AddOption(noprompt, bool, true, "prompt", 0, "Do prompt for keypress before exiting", def(false))
AddOption(continueOnError, bool, false, "continue", 0, "Continue processing after an error")
AddOption(writeoutput, bool, false, "write", 0, "Write tracks found to text output file")
AddOption(writebinary, bool, false, "writeBinary", 0, "Write tracks found to binary output file")
AddOption(DebugLevel, int, 0, "debug", 'd', "Set debug level")
AddOption(seed, int, -1, "seed", 0, "Set srand seed (-1: random)")
AddOption(cleardebugout, bool, false, "clearDebugFile", 0, "Clear debug output file when processing next event")
AddOption(sliceCount, int, -1, "sliceCount", 0, "Number of slices to process (-1: all)", min(-1), max(36))
AddOption(forceSlice, int, -1, "slice", 0, "Process only this slice (-1: disable)", min(-1), max(36))
AddOption(cudaDevice, int, -1, "gpuDevice", 0, "Set GPU device to use (-1: automatic)")
AddOption(StartEvent, int, 0, "s", 's', "First event to process", min(0))
AddOption(NEvents, int, -1, "n", 'n', "Number of events to process (-1; all)", min(1))
AddOption(merger, int, 1, "runMerger", 0, "Run track merging / refit", min(0), max(1))
AddOption(runs, int, 1, "runs", 'r', "Number of iterations to perform (repeat each event)", min(1))
AddOption(runs2, int, 1, "runsExternal", 0, "Number of iterations to perform (repeat full processing)", min(1))
AddOption(runsInit, int, 0, "runsInit", 0, "Number of initial iterations excluded from average", min(0))
AddOption(EventsDir, const char*, "pp", "events", 'e', "Directory with events to process", message("Reading events from Directory events/%s"))
AddOption(OMPThreads, int, -1, "omp", 't', "Number of OMP threads to run (-1: all)", min(-1), message("Using %d OMP threads"))
AddOption(eventDisplay, bool, false, "display", 'd', "Show standalone event display", message("Event display: %s"))
AddOption(qa, bool, false, "qa", 'q', "Enable tracking QA", message("Running QA: %s"))
AddOption(eventGenerator, bool, false, "eventGenerator", 0, "Run event generator")
AddOption(resetids, bool, false, "enumerateClusterIDs", 0, "Enumerate cluster IDs when loading clusters overwriting predefined IDs")
AddOption(nways, int, 3, "NWays", 0, "Use n-way track-fit", min(1))
AddOptionSet(nways, int, 3, "3Way", 0, "Use 3-way track-fit")
AddOptionSet(nways, int, 1, "1Way", 0, "Use 3-way track-fit")
AddOption(nwaysouter, bool, false, "OuterParam", 0, "Create OuterParam")
AddOption(dzdr, float, 2.5f, "DzDr", 0, "Use dZ/dR search window instead of vertex window")
AddOption(cont, bool, false, "continuous", 0, "Process continuous timeframe data")
AddOption(outputcontrolmem, unsigned long long int, 0, "outputMemory", 0, "Use predefined output buffer of this size", min(0ull), message("Using %lld bytes as output memory"))
AddOption(affinity, int, -1, "cpuAffinity", 0, "Pin CPU affinity to this CPU core", min(-1), message("Setting affinity to restrict on CPU %d"))
AddOption(fifo, bool, false, "fifoScheduler", 0, "Use FIFO realtime scheduler", message("Setting FIFO scheduler: %s"))
AddOption(fpe, bool, true, "fpe", 0, "Trap on floating point exceptions")
AddOption(solenoidBz, float, -1e6f, "solenoidBz", 0, "Field strength of solenoid Bz in kGaus")
AddOption(constBz, bool, false, "constBz", 0, "Force constand Bz")
AddOption(referenceX, float, 500.f, "referenceX", 0, "Reference X position to transport track to after fit")
AddOptionVec(gpuOptions, tupleGpuOpt, "gpuOpt", 0, "Options for GPU tracker")
AddOption(printSettings, bool, false, "printSettings", 0, "Print all settings")
AddHelp("help", 'h')
AddHelpAll("helpall", 'H')
AddSubConfig(structConfigTF, configTF)
AddSubConfig(structConfigQA, configQA)
AddSubConfig(structConfigEG, configEG)
AddSubConfig(structConfigGL, configGL)
EndConfig()
