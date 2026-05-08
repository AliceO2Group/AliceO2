# Calibrations

## Events per BC Calibration
### Description
Generates histograms of **Events per Bunch Crossing (BC)**. Events can be filtered by applying amplitude thresholds to the **A-side** and **C-side**.

### Command-Line Options
| Option | Default | Description |
| :--- | :--- | :--- |
| `--slot-len-sec` | `3600` | Duration of each slot in seconds. |
| `--slot-len-tf` | `0` | Slot length in Time Frames (TFs). |
| `--one-object-per-run` | — | If set, the workflow creates only one calibration object per run. |
| `--min-entries-number` | `0` | Minimum number of entries required for a slot to be valid. |
| `--min-ampl-side-a` | `-2147483648` | Amplitude threshold for Side A events. |
| `--min-ampl-side-c` | `-2147483648` | Amplitude threshold for Side C events. |

---

## How to Run

### Simulation Data
First, it is important to digitize data with a non-zero run number, orbit, and timestamp. To set these parameters, one can use the `--configKeyValues` option, as shown in the example below.
```
o2-sim-digitizer-workflow \
--onlyDet FT0 \
--configKeyValues="HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=128;HBFUtils.orbitFirstSampled=256;HBFUtils.runNumber=560560;HBFUtils.startTime=1768464099000"
```

To process simulation data, digits must first be converted to RAW format. The `o2-ft0-digi2raw` tool performs this conversion and generates the required configuration file.

Once converted, you can run the calibration either as a single integrated workflow or by spawning as the sender and receiver components separately.

#### Single Workflow Example
Execute the following command within the simulation directory:
```
o2-raw-file-reader-workflow --input-conf FT0raw.cfg --loop -1 \
| o2-ft0-flp-dpl-workflow --condition-backend=http://localhost:8080 \
| o2-calibration-ft0-events-per-bc-processor --FT0EventsPerBcProcessor "--slot-len-sec=10" \
| o2-calibration-ccdb-populator-workflow --ccdb-path=http://localhost:8080
```

Sender example (in simulation directory):
```
o2-raw-file-reader-workflow --input-conf FT0raw.cfg --loop -1 \
| o2-ft0-flp-dpl-workflow --condition-backend=http://localhost:8080 \
| o2-dpl-output-proxy --channel-config "name=downstream,method=connect,address=tcp://localhost:30453,type=push,transport=zeromq" --dataspec "downstream:FT0/DIGITSBC"
```

Receiver example:
```
o2-dpl-raw-proxy --channel-config "name=readout-proxy,type=pull,method=bind,address=tcp://localhost:30453,rateLogging=1,transport=zeromq" --dataspec "A:FT0/DIGITSBC/0" \
| o2-calibration-ft0-events-per-bc-processor --FT0EventsPerBcProcessor "--slot-len-sec=10 --min-ampl-side-a=0" \
| o2-calibration-ccdb-populator-workflow --ccdb-path=http://localhost:8080/
```

### CTF Data
Example:
```
o2-ctf-reader-workflow --ctf-input ctf.root --onlyDet FT0 \
| o2-calibration-ft0-events-per-bc-processor --FT0EventsPerBcProcessor "--slot-len-sec=10" \
| o2-calibration-ccdb-populator-workflow --ccdb-path=http://localhost:8080/
```