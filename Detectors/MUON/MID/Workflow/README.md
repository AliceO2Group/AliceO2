<!-- doxy
\page refMUONMIDWorkflow MID Workflow
/doxy -->

# MID reconstruction workflow
The MID reconstruction can start either from simulated digits or from raw data.
The reconstruction algorithm is the same in the two cases, but the workflow is slightly different.
Indeed, in the case of MC digits, the MC labels are propagated as well, thus allowing to relate the reconstructed tracks with the corresponding generated particles.
The procedure to run the reconstruction, either from the digits or from raw data, is detailed in the following.

## Preface: getting the digits
If you do not have the digits, you can obtain a sample with:
```bash
o2-sim -g fwmugen -m MID -n 100
o2-sim-digitizer-workflow
```
## Reconstruction from MC digits
To run the MC reconstruction workflow, run:
```bash
o2-mid-reco-workflow-mc
```

## Reconstruction from raw data
The reconstruction from raw data can also be tested using as input raw data obtained from the MC digits.
### From MC digits to Raw data
To convert the MC digits into raw data format, run:
```bash
o2-mid-digits-to-raw-workflow
```
The output will be a binary file named by default *raw_mid.dat*.
Notice that the executable also generate a configuration file that is needed to read the file with the raw reader workflow (see [here](../../../Raw/README.md) for further details)

### Reconstruction from raw data
To reconstruct the raw data (either from converted MC digits or real data), run:
```bash
o2-raw-file-reader-workflow --conf mid_raw.cfg | o2-mid-reco-workflow
```

## Timing
In each device belonging to the reconstruction workflow, the execution time is measured using the `chrono` c++ library.
At the end of the execution, when the *stop* command is launched, the execution time is written to the `LOG(INFO)`.
An example output is the following:
```
Processing time / 90 ROFs: full: 3.55542 us  tracking: 2.02182 us
```
Two timing values are provided: one is for the full execution of the device (including retrieval and sending of the DPL messages) and one which concerns only the execution of the algorithm (the tracking algorithm in the above example)
The timing refers to the time needed to process one read-out-frame, i.e. one event.
