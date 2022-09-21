<!-- doxy
\page refprodtestsfull-system-test Full system test configuration and scripts
/doxy -->

## Full system test overview and quickstart guide

The full system test consists of 2 parts (detailed below):
* The time frame generation part, which runs the simulation, digitization, raw conversion, and other misc tasks.
* A DPL workflow, which can run the synchronous and the asynchronous reconstruction.

The relevant scripts are `/prodtests/full_system_test.sh` and all scripts in `/prodtests/full-system-test`.
Note that by default the `full_system_test.sh` script will do both, run the generation and then the sysc and the async workflow.
This is only a quickstart guide, for more information see https://alice.its.cern.ch/jira/browse/O2-1492.

In order to run the full system test, you need to run in the O2sim environment (`alienv enter O2sim/latest`):
```
NEvents=[N] NEventsQED=[NQ] SHMSIZE=[S] TPCTRACKERSCRATCHMEMORY=[ST] $O2_ROOT/prodtests/full_system_test.sh
```

Parameters are set via environment variables. The 2 most important and mandatory parameters are `NEvents` and `NEventsQED`, which define how many collisions are in the timeframe, and how many collisions are simulated for the QED background.
For simulating larger datasets, some memory sizes must be increased over the default, most importantly `SHMSIZE` and `TPCTRACKERSCRATCHMEMORY`.
More options are available and documented in the script itself.

For a quick run of a timeframe with 5 events (depending on yous system it should take 3 to 0 minutes), run
```
NEvents=5 NEventsQED=100 $O2_ROOT/prodtests/full_system_test.sh
```

For a simulation of a full 128 orbit time frame, run
```
NEvents=650 NEventsQED=30000 SHMSIZE=128000000000 TPCTRACKERSCRATCHMEMORY=30000000000 $O2_ROOT/prodtests/full_system_test.sh
```

## Full system test time frame generation part

The generation part (in `prodtests/full_system_test.sh` runs the following steps:
* Simulate the QED collisions
* Simulate the collisions
* Digitize all data, by default split between in 2 phases for TRD and non-TRD to reduce the memory footprint (configured via `$SPLITTRDDIGI`).
* Run TRD trap simulation
* Optionally create optimized dictionaries for ITS and MFT
* Run digits 2 raw conversion.
* By default, afterwards it will also run the sync and async DPL workflows (configured via `$DISABLE_PROCESSING`).

The `prodtests/full_system_test.sh` uses `Utilities/Tools/jobutils.sh` for running the jobs, which creates a log file for each step, and which will automatically skip steps that have already succeeded if the test is rerun in the current folder. I.e. if you break the FST or it failed at some point, you can rerun the same command line and it will continue after the last successful step. See `Utilities/Tools/jobutils.sh` for details.

Note that by default, the generation produces raw files, which can be consumed by the `raw-file-reader-workflow` and by `o2-readout-exe`.
The files can be converted into timeframes files readable by the StfBuilder as described in https://alice.its.cern.ch/jira/browse/O2-1492.

## Full system test DPL-workflow configuration and scripts

The full system test workflow scripts consist of 3 shell scripts:
* `dpl-workflow.sh` : The main script that runs the dpl-workflow for the reconstruction.
   It can read the input either internally, or receive it externally by one of the others.
* `raw-reader.sh` : Runs the `o2-raw-file-reader` to read the raw files as external input to `dpl-workflow.sh`.
* `datadistribution.sh` : Run the `StfBuilder` to read time frame files as external input to `dpl-workflow.sh`.

One can either run the `dpl-workflow.sh` standalone (with `EXTINPUT=0`) or in parallel with one of the other scripts in separate shells (with `EXTINPUT=1`)

There are the additional benchmark script:
* `start_tmux.sh` : This starts the full test in the configuration for the EPN with 2 NUMA domains, 512 GB RAM, 8 GPUs.
   It will run tmux with 3 sessions, running twice the `dpl-workflow.sh` and once one of the external input scripts (selected via `dd` and `rr` command line option).
   * Please note that `start_tmux.sh` overrides several of the environment options (see below) with the defaults for the EPN.
     The only relevant options for `start_tmux.sh` should be `TFDELAY` and `GPUMEMSIZE`.
   * Note also that while `dpl-workflow.sh` is a generic flexible script that can be used for actual operation, `start_tmux.sh` is a benchmark script to demonstrate how the full workflow is supposed to run on the EPN.
     It is meant for standalone tests and not to really start the actual processing on the EPN.
* `shm-tool.sh` : This starts the SHM management tool in the configuration for start_tmux.sh. This can be used to mimic the fast memory allocation / gpu registration on the EPNs. (See te JIRA ticket for details)

In any case, the shared `setenv.sh` script sets default configuration options.

The `dpl-workflow.sh` can run both the synchronous and the asynchronous workflow, selected via the `SYNCMODE` option (see below), but note the following constraints.
* By default, it will run the full chain (EPN + FLP parts) such that it can operate as a full standalone benchmark processing simulated raw data.
* In order to run only the EPN part (skipping the steps that will run on the FLP), an `EPNONLY` option will be added later.

All settings are configured via environment variables.
The default settings (if no env variable is exported) are defined in `setenv.sh` which is sourced by all other scripts.
(Please note that `start_tmux.sh` overrides a couple of options with EPN defaults).
The environment variables are documented here: https://github.com/AliceO2Group/O2DPG/blob/master/DATA/common/README.md

## Files produced / required by the full system test

The local files like geometry, material LUT etc. are not anymore needed for reconstruction and calibration workflows, but some QC workflows still need
* geometry file o2sim_geometry-aligned.root
* old-style (dummy) monolitic GRP file o2sim_grp.root

The full system test produces the following output:
* QED simulation output in the folder `qed`.
* Normal simulation output (as `o2-sim`).
* Digits from `o2-sim-digitizer-workflow` and TRD tracklets.
* Depending on the options mentioned above, `ctf_dictionary.root` and `ITSdictionary.bin` / `MFTdictionary.bin`.
* RAW files in `raw/[DETECTOR_NAME]`

By default the `dpl-workflow.sh` adds sub-workflows with the option `--disable-root-output`, preventing initialization of writer devices (therefore the intermediate reconstruction files are not produced).
This behaviour can be overridden by passing `DISABLE_ROOT_OUTPUT=0` which will trigger storing all intermediate reconstruction files as well as the creation of the `o2_tfidinfo.root` file containing timing information of every processed TF (produced by `o2-tfidinfo-writer-workflow`).
In case intermediate files of only a few workflows are needed, instead of enabling all writers via `DISABLE_ROOT_OUTPUT=0` one can enable those of particular devices only by passing `ENABLE_ROOT_OUTPUT_<workflow_name_with_underscores>=`.
I.e. to enable storing the output of the `o2-its-reco-workflow` it is enough to prepend the script by `ENABLE_ROOT_OUTPUT_o2_its_reco_workflow=`. The `o2-tfidinfo-writer-workflow` will be invoked if there is at least one workflow with root outputs enabled.
