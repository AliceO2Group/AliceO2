<!-- doxy
\page refDetectorsTOFtestWorkflow testWorkflow
/doxy -->

# TOF calibration workflows

## DCS DP processing:

Local example workflow with local CCDB (running on port 8080) :

This will read the list of DPs to be associated to TOF from CCDB (remove
`--use-ccdb-to-configure` if you don't want this, but use hardcoded
aliases. You can specify the path of CCDB also with `--ccdb-path`.
You can also specify to run in verbose mode (`--use-verbose-mode`)

```shell
o2-calibration-tof-dcs-sim-workflow --max-timeframes 3 --delta-fraction 0.5 -b |
o2-calibration-tof-dcs-workflow --use-ccdb-to-configure -b |
o2-calibration-ccdb-populator-workflow --ccdb-path="http://localhost:8080" -b
```
To populate locally a DCS entry for the configuration, run the:

`O2/Detectors/TOF/calibration/macros/makeCCDBEntryForDCS.C`

macro.

Detailed explanation of the command above:

1. https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/TOF/calibration/testWorkflow/tof-dcs-sim-workflow.cxx --> executable to generate the TOF DCS DPs

1. https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/DCS/testWorkflow/src/DCSRandomDataGeneratorSpec.cxx --> data processor executed by the "tof-dcs-sim-workflow.cxx" above (see 1); it is defined with the list of aliases and the name of the detector; it will take care of generating randomly the data points, and send them wither as a Full Buffer Image (FBI, see code, for the full list of data points), or as a "delta" (containing only the values of the DPs that changed)

1. https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/TOF/calibration/testWorkflow/tof-dcs-data-workflow.cxx --> executable to trigger the DCS processing;

1. https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/TOF/calibration/testWorkflow/TOFDCSDataProcessorSpec.h --> data processor executed by the above "tof-dcs-data-workflow.cxx" (see 3); its input is the block of DCS data subscribed by TOF, the output is the CCDB payload and information. In between, "TOFDCSProcessor.cxx" is used (see 5).

1. https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/TOF/calibration/src/TOFDCSProcessor.cxx --> this is the core of the processing, which is detector dependent; it iw what will run at every new "packet" of data, and it will create the object to be sent to the CCDB;

1. https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/Calibration/workflow/ccdb-populator-workflow.cxx --> executable (service in common to everything that populates the CCDB) that fills the CCDB from the payload and info created in 5.

When all is tested locally, the central simulation in DCS can be done, and the procedure can be tested centrally with the "dcs-proxy" (not for development, see https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/DCS/testWorkflow/src/dcs-proxy.cxx).

The definition per detector of the DPs that it has to subscribe to should be stored in CCDB, see e.g. https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/TOF/calibration/macros/makeTOFCCDBEntryForDCS.C.

## LHC phase:

This will process the LHC phase simulated by the Generator workflow

```shell
LHC phase
o2-calibration-data-generator-workflow --lanes 10 --mean-latency 100000 --max-timeframes 500 |
o2-calibration-lhc-clockphase-workflow --tf-per-slot 20 |
o2-calibration-ccdb-populator-workflow --ccdb-path localhost:8080
```

## TOF channel calibration:

To obtain the TOF channel offsets (at end of processing). Input from simulation, but is should work if attached to reco+calib flow

* simulating reading from ccdb, and using it, with "-b", in "test" mode --> to use this, we need an appropriate CCDB object in the CCDB

```shell
o2-calibration-data-generator-workflow --lanes 10 --mean-latency 100000 --max-timeframes 500 --do-TOF-channel-calib --do-TOF-channel-calib-in-test-mode -b |
o2-testworkflows-tof-dummy-ccdb -b |
o2-calibration-tof-channel-calib-workflow --min-entries 50 --do-TOF-channel-calib-in-test-mode --use-ccdb -b |
o2-calibration-ccdb-populator-workflow --ccdb-path localhost:8080 -b
```

* simulating reading from ccdb, but not using it, with "-b", in "test" mode --> to use this, we need an appropriate CCDB object in the CCDB

```
o2-calibration-data-generator-workflow --lanes 10 --mean-latency 100000 --max-timeframes 500 --do-TOF-channel-calib --do-TOF-channel-calib-in-test-mode -b |
o2-testworkflows-tof-dummy-ccdb -b
| o2-calibration-tof-channel-calib-workflow --min-entries 50 --do-TOF-channel-calib-in-test-mode -b
| o2-calibration-ccdb-populator-workflow --ccdb-path localhost:8080 -b
```

* Using the workflow that has both LHCclockPhase and TOFChannelCalib; for now I can enable only one, or the CCDB populator will not work

```shell
o2-calibration-data-generator-workflow --lanes 10 --mean-latency 100000 --max-timeframes 500 --do-TOF-channel-calib --do-TOF-channel-calib-in-test-mode -b |
o2-calibration-tof-dummy-ccdb-for-calib -b |
o2-calibration-tof-calib-workflow --do-channel-offset --min-entries 50 --do-TOF-channel-calib-in-test-mode -b |
o2-calibration-ccdb-populator-workflow --ccdb-path localhost:8080 -b
```
* same as above, enabling CCDB

```shell
o2-calibration-data-generator-workflow --lanes 10 --mean-latency 100000 --max-timeframes 500 --do-TOF-channel-calib --do-TOF-channel-calib-in-test-mode -b |
o2-calibration-tof-dummy-ccdb-for-calib -b |
o2-calibration-tof-calib-workflow --do-channel-offset --use-ccdb --min-entries 50 --do-TOF-channel-calib-in-test-mode -b |
o2-calibration-ccdb-populator-workflow --ccdb-path localhost:8080 -b
```

By default the rate of the TF sampling will match to `mean-latence / lanes`, so that the sampling and generation are balanced and no TF is dropped out at the generation level.
One can provide the option `--pressure <float, D=1.>`, in which case the sampling rate will be increased by this factor, creating a back-pressure on the processing due to its latency.

There is a possibility to generate data in multiple independent workflows, still avoiding TF-s with the same ID in different workflow. Options `--gen-norm <N, D=1> --gen-slot <I,D=0>` will
enforce skipping all TF-s except those with ``(TFID/lanes)%N == I``. I.e. for example, to emulate the processing with 3 different EPN's with 8 lanes each one can run in 3 terminals:

```shell
# Term.1: will produce TFs [0:7], [24:31], [48:55] ...
o2-calibration-data-generator-workflow --lanes 8 --max-timeframes 5000 --gen-norm 3 --gen-slot 0

# Term.2: will produce TFs [8:15], [32,39], [56:63] ...
o2-calibration-data-generator-workflow --lanes 8 --max-timeframes 5000 --gen-norm 3 --gen-slot 1

# Term.3: will produce TFs [16:23], [40:47], [64:71] ...
o2-calibration-data-generator-workflow --lanes 8 --max-timeframes 5000 --gen-norm 3 --gen-slot 2
```

* To run the calibration with cosmics:

```shell
o2-tof-cluscal-reader-workflow -b   | o2-calibration-tof-calib-workflow -b --cosmics --do-channel-offset --min-entries 50 --update-interval 10000
```

## TimeSlewing:

For Time Slewing. Will save the Time Slewing information in files when a certain condition is reached. A post-processing
should then take care of extracting the CCDB

* test mode:

``` shell
o2-calibration-data-generator-workflow --lanes 10 --mean-latency 100000 --max-timeframes 10 -b |
o2-calibration-tof-collect-calib-workflow --tf-sending-policy --running-in-test-mode -b

* non-test but simplified (using option "is-max-number-hits-to-fill-tree-absolute"):

```shell
o2-calibration-data-generator-workflow --lanes 10 --mean-latency 100000 --max-timeframes 10 -b |
o2-calibration-tof-collect-calib-workflow --max-number-hits-to-fill-tree 300 --is-max-number-hits-to-fill-tree-absolute -b
```
