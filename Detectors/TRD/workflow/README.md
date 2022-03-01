<!-- doxy
\page refDetectorsTRDworkflow TRD workflow
/doxy -->

# DPL workflows for the TRD

## TRD reconstruction workflow
The TRD reconstruction workflow requires TRD tracklets, TRD trigger records and as additional input either TPC tracks, ITS-TPC matches, or both.
Take a look at the Input data section to see how to generate the input from a simulation.

The reconstruction workflow consists of the following DPL processors:

* `TRDTrackletReader`
* `tpc-track-reader`
* `itstpc-track-reader`
* `trd-globaltracking[TPC_ITS-TPC]` using [o2::gpu::GPUTRDTracker_t<TrackType,PropagatorType>](../../../../tree/dev/GPU/GPUTracking/TRDTracking/GPUTRDTracker.h)
* `trd-trackbased-calib` using [o2::trd::TrackBasedCalib](../../../../tree/dev/Detectors/TRD/calibration/include/TRDCalibration/TrackBasedCalib.h)
* `trd-track-writer-tpc`
* `trd-track-writer-tpcits`
* `TRDCalibWriter`

The different track reader and writer are only added to the workflow if the respective tracking source is configured. The workflow is started with the command:
`o2-trd-global-tracking`. Available options are:
* `--disable-mc` disable MC labels
* `--disable-root-input` the input is provided by another DPL device
* `--disable-root-output` the output is not written to file
* `--track-sources ITS-TPC,TPC` a comma-seperated list of sources to use for the tracking. Default is `ALL` which results in the same workflow as `ITS-TPC,TPC`
* `--strict-matching` if enabled, TPC-TRD tracks which have another hypothesis close to the best one are flagged as ambiguous and not written out. The minimum chi2 gap between best and second best hypothesis can be configured with `GPU_rec_trd.chi2SeparationCut`
* `--filter-trigrec` if enabled, the trigger records for which no ITS information is available are ignored. Per default on in the synchronous reconstruction, since there is no matching to TPC-only tracks in that case.
* `--configKeyValues "GPU_rec_trd.nSigmaTerrITSTPC=4;"` only one example for the different options for the TRD tracker. All are listed in [GPUSettingsList.h](../../../../dev/GPU/GPUTracking/Definitions/GPUSettingsList.h). Search for GPUSettingsRecTRD


## Input data
Since the TRD reconstruction requires either ITS-TPC or TPC tracks to run there are some additional steps to be taken next to the simulation and digitization of an example data sample:
These commands will generate a small pp data sample with 10 events:
* `o2-sim -n 10 -g pythia8pp --skipModules ZDC`
* `o2-sim-digitizer-workflow -b` this will create both TRD digits and TRD tracklets
* `o2-tpc-reco-workflow -b --input-type digits --output-type clusters,tracks --configKeyValues "GPU_proc.ompThreads=4;" --shm-segment-size 10000000000 --run`
* `o2-its-reco-workflow -b --trackerCA --tracking-mode async --shm-segment-size 10000000000 --run`
* `o2-tpcits-match-workflow -b --tpc-track-reader tpctracks.root --tpc-native-cluster-reader "--infile tpc-native-clusters.root" --shm-segment-size 10000000000 --run`
* `o2-trd-tracklet-transformer -b --filter-trigrec` if the trigger filter is active here, it must also be active for the TRD tracking. Otherwise an error will be thrown
* `o2-trd-global-tracking -b --filter-trigrec`

Of course one can also concatenate the workflows. For example:`o2-trd-tracklet-transformer -b --disable-root-output | o2-trd-global-tracking -b`. This will create the calibrated tracklets on-the-fly and not create the `trdcalibratedtracklets.root` file.

