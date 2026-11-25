This is a quick documentation on the TPC Tracking QA for Resolution, Efficiency, and Cluster Attachment based on MC data.

The TPC QA can produce 3 different output types:
* *mergeble* histograms: A collection of ROOT histograms that can be merged from different inputs. The need to be postprocessed to get meaningful output.
* *postprocessed* histograms: Histograms showing the efficiencies, resolutions etc. These histograms can no longer be merged from multiple inputs.
* *layouts*: TCanvases with multiple postprocessed histograms arranged in reasonable layouts.

The TPC Tracking QA consists of multiple QA subtasks ((de)activated via a bitmask):
* *Efficiency* / *Clone Rate* / *Fake Rate* (1)
* *Resolution* (2)
* *Pulls* (4)
* *Cluster Attachment Statistics* (8)
* *nClusters and pt distribution* (16)
* *Cluster rejection counts* (32) (both as aggregate text report and as histogram)

The TPC QA can run in 3 different ways:
* *Standalone* inside the tracking (o2-tpc-reco-workflow), it will write its output in pdf format to the plots folder in the current directory: supports *all subtasks* and will always produce the *layouts* output.
* As *external source* to QC: the QA is running inside the o2-tpc-reco-workflow and ships the histograms to QC via DPL: supports *all subtasks*, can produce the *postprocessed* or the *layouts* output.
* As *independent* QC tasks, operating on DPL input, that can e.g. be read from ROOT files: supports *subtasks* *1*, *2*, and *4*, currently is hardcoded to the *mergeable* output, but could be made configurable.
_(Note: the reason that the independent QC supports fewer subtasks is that the other tasks require internal tracking data structures that are no available a posteriori.)_

Remark on the *Cluster Rejection count histograms* for the *online QC*:
* These are mainly meant for monitoring the TPC compression during data taking without MC information, while most other subtasks rely on MC information.
* These are always in the mergeable format, the postprocessing will just forward them.
* By default they are disabled and only aggregate text output shows the rejection ratios, they must be enabled explicitly as explained below.

Running the TPC QA standalone:
* It must be enabled via the configKeyValue GPU_proc.runQA in the o2-tpc-reco-workflow. The QA will run as part of the normal TPC tracking and will have access to all data structures of the tracking. Otherwise, the normal settings for the o2-tpc-reco-workflow apply.
* Example to run on digits:
```o2-tpc-reco-workflow -b --infile tpcdigits.root --configKeyValues "GPU_proc.runQA=1;" --output-type clusters,tracks```
* Example to run on clusters:
```o2-tpc-reco-workflow --input-type clusters --infile tpc-native-clusters.root --output-type tracks --configKeyValues "GPU_proc.runQA=1"```

Running the TPC QA in the standalone benchmark:
* `./ca -e [dataset] --qa` runs the QA on a dataset with MC labels, prints the efficiency / clone / fake rates (`QA Stats` lines) and the cluster attachment statistics, and writes the plots to the `plots` folder. Use a build without `GPUCA_BUILD_DEBUG`, the QA is slow.
* `--QAoutput file.root` stores the histograms, `--QAinput file.root` overlays them in the plots of a new run.
* QA options are set with `--QA[option]` (`GPU_QA.[option]` in O2), see [GPUSettings.md](https://github.com/AliceO2Group/AliceO2/blob/dev/GPU/documentation/GPUSettings.md).

Running the TPC QA inside the o2-tpc-reco-workflow as external source for qc:
* As in the standalone mode, the TPC Tracking QA will run as part of the o2-tpc-reco-workflow with full access to the tracking data structures.
* The output is shipped to QC as external qc-input in the form of ROOT histograms, which disables some subtasks that do not produce ROOT histograms (such as the cluster counts (32)).
* A merged workflow of the o2-tpc-reco-workflow and qc must be configures, such as:
```o2-tpc-reco-workflow --input-type clusters --infile tpc-native-clusters.root --output-type tracks,qa | o2-qc --config json:/${QUALITYCONTROL_ROOT}/etc/tpcQCTrackingFromExternal_direct.json```
(Note that by default, the output will be uploaded and visible at https://qcg-test.cern.ch/)
* By default, this will create postprocessed histograms, and it can be switched via configKeyValues to layouts output via:
```o2-tpc-reco-workflow --input-type clusters --infile tpc-native-clusters.root --output-type tracks,qa --configKeyValues "GPU_QA.shipToQCAsCanvas=true" | o2-qc --config json:/${QUALITYCONTROL_ROOT}/etc/tpcQCTrackingFromExternal_direct.json```
* This mode can be combined with the standalone QA mode.
* Running the *online cluster rejection histogram QA*:
** This mode works also without MC information, it must be enabled explicitly. Irrespective of the output mode, it will always furnish the same mergeable histograms, since there is no postprocessing. An example to run them on ROOT files:
```o2-tpc-reco-workflow --input-type clusters --infile tpc-native-clusters.root --output-type tracks,qa --disable-mc --configKeyValues "GPU_QA.clusterRejectionHistograms=1" | o2-qc --config json:/${QUALITYCONTROL_ROOT}/etc/tpcQCTrackingFromExternal_direct.json```
** In order to set the x-axis scale (number of clusters), use the configKeyValue GPU_QA.histMaxNClusters.

Running the QA as independent QC task:
* In this mode, the QA runs independently from the o2-tpc-reco-workflow, getting clusters, tracks, and MC labels via DPL.
* The example below uses the track reader and the reco workflow to fetch the input and ship them via DPL, but the inputs can of course also come from other sources.
* This mode is mostly foreseen for the mergeble output. In this way, many instance can run in parallel and the output can be merged before being postprocessed.
* Currently, the mergeble output is hardcoded, but the task could easily be extended for other outputs (see below).
* To run the tasks on ROOT file input, you can use the following example:
```o2-tpc-track-reader | o2-tpc-reco-workflow --input-type clusters --infile tpc-native-clusters.root --output-type disable-writer | o2-qc --config json:/${QUALITYCONTROL_ROOT}/etc/tpcQCTracking_direct.json```

Postprocessing the mergeble output:
* The o2::tpc::qc::Tracking class can be used for the postprocessing in a standalone way.
* Initialize the class with the settings postprocessOnly = true and outputMode = outputPostprocessed or outputLayout.
* Call the postprocess(...) function, passing in std::vectors of the 3 types of ROOT histograms (TH1F, TH2F, TH1D) which are used by the QA. Note that the order of the histograms must be the same as obtained in the mergeble output.
* Depending on the outputMode setting, the class will fill the out object witl either the postprocessed histograms or the canvas layouts.

The following classes in O2 / QC belong to the TPC tracking QA:
* o2::gpu::GPUQA (O2/GPU/GPUTracking/qa/GPUQA.cxx): The main QA class, which can produce the standalone output, or can run with external input driven from the o2::gpu::GPUO2InterfaceQA.
* o2::gpu::GPUO2InterfaceQA (O2/GPU/GPUTracking/Interface/GPUO2InterfaceQA.cxx): Internal interface class, uses o2::gpu::GPUQA.
* o2::tpc::qc::Tracking (O2/Detectors/TPC/qc/src/Tracking.cxx): The main QC class for running the TPC QA independent from the o2-tpc-reco-workflow, uses o2::gpu::GPUO2InterfaceQA.
** Can produce all output types from tracks / clusters as input.
** Can produce postprocessed / layout output from mergeble input.
** Is limited to subtasks that do not require access to the internal tracking data structures.
* o2::quality_control_modules::tpc::Tracking (QC/Modules/TPC/src/Tracking.cxx): Mostly a QC wrapper for o2::tpc::qc::Tracking, with the necessary framework code to receive clusters / tracks / MC labels via DPL.

Several additional settings can be configured via configKeyValues as listed in https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/TPC/qc/include/TPCQC/Tracking.h#L44:
* "GPU_QA.strict=[bool]"               Strict QA mode: Only consider resolution of tracks where the fit ended within 5 cm of the reference, and remove outliers. (Default: true)
* "GPU_QA.qpt=[float]"                 Set cut for Q/Pt. (Default: 10.0)
* "GPU_QA.recThreshold=[float]"        Compute the efficiency including impure tracks with fake contamination. (Default 0.9)
* "GPU_QA.maxResX=[float]"             Maxmimum X (~radius) for reconstructed track position to take into accound for resolution QA in cm (Default: no limit)
* "GPU_QA.nativeFitResolutions=[bool]" Create resolution histograms in the native fit units (sin(phi), tan(lambda), Q/Pt) (Default: false)
* "GPU_QA.filterCharge=[int]"          Filter for positive (+1) or negative (-1) charge (Default: no filter)
* "GPU_QA.filterPID=[int]"             Filter for Particle Type (0 Electron, 1 Muon, 2 Pion, 3 Kaon, 4 Proton) (Default: no filter)
