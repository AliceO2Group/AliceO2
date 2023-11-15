<!-- doxy
\page refDetectorsTRDworkflow TRD workflow
/doxy -->

# DPL workflows for the TRD

## TRD synchronous reconstruction workflow

An overview of the current synchronous reconstruction workflow is shown in the figure below:

![TRDReco](https://user-images.githubusercontent.com/26281793/259394523-cf115d1b-954e-4d8f-b2cd-ff0eef6a0c99.png)

We have the following binaries which can be used to assemble a global workflow:

* `o2-trd-datareader`: extracts digits, tracklets, trigger records and some statistics from the raw data (not needed in asynchronous reconstruction, when starting from CTFs)
* `o2-trd-tracklet-transformer`: creates calibrated tracklets (space points) from the `Tracklet64` data type it receives from the `o2-trd-datareader`. Optionally this is done only for triggers where ITS ROF where reconstructed to save computing time during synchronous running.
* `o2-trd-global-tracking`: uses tracklets, ITS-TPC matched tracks or TPC-only tracks as input to create global tracks with TRD tracklets attached. Can in addition run track-based calibrations, e.g. for vDrift, gain or t0 monitoring.
* `o2-calibration-trd-workflow`: used for any of the TRD calibrations running on the aggregator node taking input from the track-based calibrations or also for noise runs taking the input directly from the `o2-trd-datareader`.

Run any of the workflows with `--help` or `--help full` to see the available options and their explanations.

The available `--configKeyValues` for the global tracking workflow are mostly listed in [GPUSettingsList.h](../../../../dev/GPU/GPUTracking/Definitions/GPUSettingsList.h). Search for GPUSettingsRecTRD to find options as for example the minimum number of TRD tracklets required for a TRD track to not be discarded.



## Input data

In order to run any of the workflows locally you have to have some input data. This can be retrieved either from real runs as described in the [PDP documentation](https://alice-pdp-operations.docs.cern.ch/ctf-reprocessing/) or you can run your own simulation. The easiest will be to use the [sim_challenge.sh](../../../../dev/prodtests/sim_challenge.sh) script. Run it as

    $O2_ROOT/prodtests/sim_challenge.sh -h

to see the available options. Without any arguments you will simulate a pp TF with 10 events. You will **need an [alien token](https://alice-doc.github.io/alice-analysis-tutorial/start/cert.html#test-your-certificate)** for the simulation to work.


## QC

In order to attach the QC manually to a workflow for local tests one can do for example

    o2-trd-digit-reader-workflow -b --disable-mc | o2-qc --config json://TRDdigits.json -b

The `o2-qc` is steering the QC, in the json file `TRDdigits.json` we can specify which QC tasks should run, if there is any data sampling to be done or where to store the output. Also parameters can be passed to the QC tasks and much more. A very simple example for running only digits QC would be

```json
{
  "qc": {
    "config": {
      "database": {
        "implementation": "CCDB",
        "host": "ccdb-test.cern.ch:8080",
        "username": "not_applicable",
        "password": "not_applicable",
        "name": "not_applicable"
      },
      "Activity": {
        "number": "42",
        "type": "2"
      },
      "monitoring": {
        "url": "infologger:///debug?qc"
      },
      "consul": {
        "url": ""
      },
      "conditionDB": {
        "url": "http://alice-ccdb.cern.ch"
      },
      "infologger": {
        "filterDiscardDebug": "false",
        "filterDiscardLevel": "20"
      }
    },
    "tasks" : {
      "DigitTask" : {
        "active": "true",
        "className": "o2::quality_control_modules::trd::DigitsTask",
        "moduleName": "QcTRD",
        "detectorName": "TRD",
        "cycleDurationSeconds": "60",
        "maxNumberCycles": "-1",
        "dataSource": {
          "type": "direct",
          "query": "digits:TRD/DIGITS;triggers:TRD/TRKTRGRD"
        },
        "saveObjectsToFile":"QC_TRD_digits.root"
      }
    },
    "dataSamplingPolicies": [
    ]
  }
}
```
This would run the `DigitsTask` from QC without any data sampling (so taking directly all the digits provided by the reader) and store the QC histograms in `QC_TRD_digits.root`.

One can also check the results in the test QCG at <https://qcg-test.cern.ch/?page=objectTree>. If you go to `TRD/MO/DigitTask/digitsperevent` you will see the same histogram which you have in your local file.
