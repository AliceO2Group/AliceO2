<!-- doxy
\page refTPCworkflow TPC workflow
/doxy -->

# DPL workflows for the TPC

## TPC reconstruction workflow
The TPC reconstruction workflow starts from the TPC digits, the *clusterer* reconstructs clusters in the
[ClusterHardware](../../../DataFormats/Detectors/TPC/include/DataFormatsTPC/ClusterHardware.h) format.
The clusters are directly written in the  *RAW page* format. The raw data are passed onto the *decoder*
providing the [TPC native cluster](../../../DataFormats/Detectors/TPC/include/DataFormatsTPC/ClusterNative.h)
format to the *tracker*.

Note: The format of the raw pages is preliminary and does not reflect what is currently implemented in the CRU.

The workflow consists of the following DPL processors:

* `tpc-digit-reader` -> using tool [o2::framework::RootTreeReader](../../../Framework/Utils/include/Utils/RootTreeReader.h)
* `tpc-clusterer` -> interfaces [o2::tpc::HwClusterer](../reconstruction/include/TPCReconstruction/HwClusterer.h)
* `tpc-cluster-decoder` -> interfaces [o2::tpc::HardwareClusterDecoder](../reconstruction/include/TPCReconstruction/HardwareClusterDecoder.h)
* `gpu-reconstruction` -> interfaces [o2::tpc::GPUCATracking](../reconstruction/include/TPCReconstruction/GPUCATracking.h)
* `tpc-track-writer` -> implements simple writing to ROOT file

Depending on the input and output types the default workflow is extended by the following readers and writers:
* `tpc-raw-cluster-writer` writes the binary raw format data to binary branches in a ROOT file
* `tpc-raw-cluster-reader` reads data from binary branches of a ROOT file
* `tpc-cluster-writer` writes the binary native cluster data to binary branches in a ROOT file
* `tpc-cluster-reader` reads data from binary branches of a ROOT file

MC labels are passed through the workflow along with the data objects and also written together with the
output at the configured stages (see output types).

### Input data
The input can be created by running the simulation (`o2sim`) and the digitizer workflow (`digitizer-workflow`).
The digitizer workflow produces the file `tpcdigits.root` by default, data is stored in separated branches for
all sectors.

The workflow can be run starting from digits, raw clusters, or (native) clusters, or directly attached to the
`o2-sim-digitizer-workflow`, see comment on inputs types below.

### Quickstart running the reconstruction workflow
The workflow is implemented in the `o2-tpc-reco-workflow` executable.

Display all options
```
o2-tpc-reco-workflow --help
```

Important options for the `tpc-digit-reader` as initial publisher
```
--infile arg                          Name of the input file
--treename arg (=o2sim)               Name of the input tree
--digitbranch arg (=TPCDigit)         Digit branch
--mcbranch arg (=TPCDigitMCTruth)     MC label branch
--nevents                             Number of events
```

The `tpc-raw-cluster-reader` uses the same options except the branch name configuration
--databranch arg (==TPCClusterHw)       RAW Cluster branch
--mcbranch arg (==TPCClusterHwMCTruth)  MC label branch

Options for the `tpc-track-writer` process
```
--outfile arg (=tpctracks.root)               Name of the output file
--treename arg (=tpcrec)                      Name of tree
--nevents arg (=-1)                           Number of events to execute
--terminate arg (=process)                    Terminate the 'process' or 'workflow'
--track-branch-name arg (=TPCTracks)          configurable branch name for the tracks
--trackmc-branch-name arg (=TPCTracksMCTruth) configurable branch name for the MC labels
```

Examples:
```
o2-tpc-reco-workflow --infile tpcdigits.root --tpc-sectors 0-17 --tracker-options "cont refX=83 bz=-5.0068597793"
```

```
o2-tpc-reco-workflow --infile tpcdigits.root --tpc-sectors 0-17 --disable-mc 1 --tracker-options "cont refX=83 bz=-5.0068597793"
```

### Global workflow options:
```
--input-type arg (=digits)            digitizer, digits, raw, clusters
--output-type arg (=tracks)           digits, raw, clusters, tracks
--disable-mc arg (=0)                 disable sending of MC information
--tpc-lanes arg (=1)                  number of parallel lanes up to the tracker
--tpc-sectors arg (=0-35)             TPC sector range, e.g. 5-7,8,9
```

#### Input Type
Input type `digitizer` will create the clusterers with dangling input, this is used
to connect the reconstruction workflow directly to the digitizer workflow.

All other input types will create a publisher process reading data from branches of
a ROOT file. File and branch names are configurable. The MC labels are always read
from a parallel branch, the sequence of data and MC objects is assumed to be identical.

#### Output Type
The output type selects up to which final product the workflow is executed. Multiple outputs
are supported in order to write data at intermediate steps, e.g.
```
--output-type raw,tracks
```

MC label data are stored in corresponding branches per sector. The sequence of MC objects must match
the sequence of data objects.

By default, all data is written to ROOT files, even the data in binary format like the raw data and cluster
data. This allows to record multiple sets (i.e. timeframes/events) in one file alongside with the MC labels.

#### Parallel processing
Parallel processing is controlled by the option `--tpc-lanes n`. The digit reader will fan out to n processing
lanes, each with clusterer, and decoder. The tracker will fan in from multiple parallel lanes.
For each sector, a dedicated DPL data channel is created. The channels are distributed among the lanes.
The default configuration processes sector data belonging together in the same time slice, but in earlier
implementations the sector data was distributed among multiple time slices (thus abusing the DPL time
slice concept). The tracker spec implements optional buffering of the input data, if the set is not complete
within one invocation of the processing function.

#### TPC sector selection
By default, all TPC sectors are processed by the workflow, option `--tpc-sectors` reduces this to a subset.

### Processor options

#### TPC CA tracker
The [tracker spec](src/CATrackerSpec.cxx) interfaces the [o2::tpc::GPUCATracking](../reconstruction/include/TPCReconstruction/GPUCATracking.h)
worker class which can be initialized using an option string. The processor spec defines the option `--tracker-option`. Currently, the tracker
should be run with options:
```
--tracker-option "refX=83 cont"
```

The most important tracker options are:
```
cont    activation of continuous mode
refX=   reference x coordinate, tracker tries to propagate all track to the reference
bz=     magnetic field
```

### Current limitations/TODO
* the propagation of MC labels goes together with multiple rearrangements and thus copy
* raw pages are using RawDataHeader version 2 with 4 64bit words, need to be converted to version 4
  also the CRU will pad all words, RawDataHeader and data words to 128 bits, this is not yet reflected
  in the produced raw data

* implement configuration from the GRP

## Open questions

## Calibration workflows

### Pedestal calibration
#### Options
```bash
--direct-file-dump     write final calibration to local root file
--max-events           maximum number of events to process
--no-write-ccdb        don't send the calibration data via DPL (required in case the calibration write is not attached)
--use-old-subspec      use old subspec definition (CruId << 16) | ((LinkId + 1) << (CruEndPoint == 1 ? 8 : 0))
--lanes arg (=1)       number of parallel processes
--sectors arg (=0-35)  list of TPC sectors, comma separated ranges, e.g. 0-3,7,9-15
```

#### Running with data distribution
In one shell start the data distribution playback, e.g.
```bash
StfBuilder --id builder --data-source-enable --data-source-dir=2020-11-11T14_18_25Z --data-source-rate=100 --dpl-channel-name=dpl-chan  --channel-config "name=dpl-chan,type=pair,method=connect,address=ipc:///tmp/stf-builder-dpl-pipe-0,transport=zeromq,rateLogging=1"
```

In another shell start the pedestal calibration
```bash
o2-dpl-raw-proxy -b --dataspec "A:TPC/RAWDATA" --channel-config "name=readout-proxy,type=pair,method=bind,address=ipc:///tmp/stf-builder-dpl-pipe-0,transport=zeromq,rateLogging=1" \
| o2-tpc-calib-pedestal  --max-events 5 --direct-file-dump --shm-segment-size $((8<<30)) --no-calib-output --use-old-subspec
```

#### Running with raw file playback
Create a raw-reader.cfg e.g.
```bash
i=0; echo -e "[defaults]\ndataOrigin = TPC\ndataDescription = RAWDATA\n" > raw-reader.cfg; echo; for file in *.raw; do echo "[input-$i]"; echo "dataOrigin = TPC"; echo "dataDescription = RAWDATA"; echo "filePath=$file"; echo; i=$((i+1)); done >> raw-reader.cfg
```

Then run
```bash
o2-raw-file-reader-workflow  --input-conf raw-reader.cfg --nocheck-hbf-per-tf --nocheck-hbf-jump --shm-segment-size $((8<<30)) \
|  o2-tpc-calib-pedestal  --max-events 5 --direct-file-dump --shm-segment-size $((8<<30)) --no-calib-output
```

#### Send data to CCDB
Remove the `--no-write-ccdb` option and add
```bash
| o2-calibration-ccdb-populator-workflow
```

### Laser track calibration
#### Laser track filter
`o2-tpc-laser-track-filter` filters `TPC/TRACKS` looking for laser track candidates. The output is provided as `TPC/LASERTRACKS`.
With the option `--enable-writer`, the filtere tracks can be writte to file (`tpc-laser-tracks.root`).

#### linear workflow reading from a local track file, direclty running the calibration component
By default `o2-tpc-calib-laser-tracks` assumes non-filtered `TPC/TRACKS` as input.
Using the option `--use-filtered-tracks` the input `TPC/LASERTRACKS` will be used.

**running without laser track filter**
```bash
o2-tpc-file-reader --input-type tracks --disable-mc --tpc-track-reader '--infile tpctracks.root' \
  | o2-tpc-calib-laser-tracks --write-debug
```

**running with laser track filter**
```bash
o2-tpc-file-reader --input-type tracks --disable-mc --tpc-track-reader '--infile tpctracks.root' \
  | o2-tpc-laser-track-filter \
  |  o2-tpc-calib-laser-tracks --use-filtered-tracks --write-debug --run
```

#### linear workflow reading from a local track file, running with time slot calibration
The time slot calibration assumes as input prefiltered laser track candates (`o2-tpc-laser-track-filter`).
They are published as `TPC/LASERTRACKS`.

##### Options
  --tf-per-slot arg (=5000)             number of TFs per calibration time slot
  --max-delay arg (=3)                  number of slots in past to consider
  --min-entries arg (=100)              minimum number of TFs with at least 50 tracks on each sideto finalize a slot
                                        so 100 means 5000 matched laser tracks on each side.
  --write-debug                         write a debug output tree.

```bash
o2-tpc-file-reader --input-type tracks --disable-mc --tpc-track-reader '--infile tpctracks.root' \
 | o2-tpc-laser-track-filter \
 | o2-tpc-laser-tracks-calibrator --write-debug --min-entries 100 --tf-per-slot 5000 --run
```

#### simple distributed workflow with output and input proxy, running with time slot calibration
**Sending side zeromq**
```bash
o2-tpc-file-reader --input-type tracks --disable-mc --tpc-track-reader '--infile tpctracks.root' \
 | o2-tpc-laser-track-filter \
 | o2-dpl-output-proxy --channel-config "name=downstream,method=connect,address=tcp://localhost:30453,type=push,transport=zeromq" --dataspec lasertracks:TPC/LASERTRACKS -b --run
```

**Receeving side zeromq**
```bash
o2-dpl-raw-proxy --dataspec lasertracks:TPC/LASERTRACKS/0 --channel-config "name=readout-proxy,type=pull,method=bind,address=tcp://localhost:30453,rateLogging=1,transport=zeromq" \
  | o2-tpc-laser-tracks-calibrator --write-debug --min-entries 100 --tf-per-slot 5000 --run
```

**Sending side shmem**
```bash
ARGS_ALL="--session tpc-laser-tracks -b"
o2-tpc-file-reader $ARGS_ALL --input-type tracks --disable-mc --tpc-track-reader '--infile tpctracks.root' \
 | o2-tpc-laser-track-filter $ARGS_ALL \
 | o2-dpl-output-proxy $ARGS_ALL --channel-config "name=downstream,type=push,method=bind,address=ipc://@tpc-laser-tracks-0,transport=shmem,rateLogging=1" --dataspec lasertracks:TPC/LASERTRACKS \
 | o2-dpl-run $ARGS_ALL --run

```

**Receeving side shmem**
```bash
ARGS_ALL="--session tpc-laser-tracks -b"
o2-dpl-raw-proxy $ARGS_ALL --dataspec lasertracks:TPC/LASERTRACKS/0 --channel-config "name=readout-proxy,type=pull,method=connect,address=ipc://@tpc-laser-tracks-0,transport=shmem,rateLogging=1" \
  | o2-tpc-laser-tracks-calibrator $ARGS_ALL --write-debug --min-entries 100 --tf-per-slot 5000 \
  | o2-dpl-run $ARGS_ALL --run
```

## Running the recontruction on GBT raw data
This requires to do zero suppression in the first stage. For this the `DigiDump` class is used, wrapped in an o2 workflow.

Use either the [DD](#running-with-data-distribution) part or [raw file playback](#running-with-raw-file-playback) from above and add as processor
```bash
| o2-tpc-raw-to-digits-workflow --pedestal-file pedestals.root --configKeyValues "TPCDigitDump.ADCMin=3;TPCDigitDump.NoiseThreshold=3" \
| o2-tpc-reco-workflow --input-type digitizer --output-type tracks --disable-mc
```

To directly dump the digits to file for inspection use for the reco workflow
```bash
| o2-tpc-reco-workflow --input-type digitizer --output-type digits --disable-mc
```

## TPC Common Mode Value (CMV) Workflows

The CMV workflows parse raw TPC data, buffer Common Mode Values per CRU on FLPs, then merge and aggregate them on a calibration node before serializing the CMVContainer in a TTree. The resulting object can be uploaded to the CCDB or written to the disk.

### Workflow components

| Executable | Output | Description |
|---|---|---|
| `o2-tpc-cmv-to-vector` | `TPC/CMVVECTOR` | Parses raw TPC data and creates vectors of CMVs per CRU |
| `o2-tpc-cmv-flp` | `TPC/CMVGROUP` | Buffers N TFs per CRU on the FLP and groups them for forwarding |
| `o2-tpc-cmv-distribute` | `TPC/CMVAGG*` | Routes grouped CMV batches from the calibration node to the aggregate workflow while preserving buffered TF and lane handling |
| `o2-tpc-cmv-aggregate` | TTree / CCDB payload | Collects all CRUs for each aggregate lane, preprocesses and compresses CMVs per buffered TF slice, then writes the CMVContainer TTree to disk (`--output-dir`) and/or forwards it as a CCDB object (`--enable-CCDB-output`) |

#### `o2-tpc-cmv-to-vector`

| Option | Default | Description |
|---|---|---|
| `--input-spec` | `A:TPC/RAWDATA` | DPL input spec for raw TPC data |
| `--crus` | `0-359` | CRU range to process, comma-separated ranges |
| `--write-debug` | false | Write a debug output tree every TF |
| `--write-debug-on-error` | false | Write a debug output tree only when decoding errors occur |
| `--debug-file-name` | `/tmp/cmv_vector_debug.{run}.root` | Name of the debug output ROOT file |
| `--write-raw-data-on-error` | false | Dump raw data to file when decoding errors occur |
| `--raw-file-name` | `/tmp/cmv_debug.{run}.{raw_type}` | Name of the raw debug output file |
| `--raw-data-type` | 0 | Raw data format to dump on error: 0 = full TPC with DPL header, 1 = full TPC with DPL header (skip empty), 2 = full TPC no DPL header, 3 = full TPC no DPL header (skip empty), 4 = IDC raw only, 5 = CMV raw only |
| `--check-incomplete-hbf` | false | Check and report incomplete HBFs in the raw parser |

#### `o2-tpc-cmv-flp`

| Option | Default | Description |
|---|---|---|
| `--crus` | `0-359` | CRU range handled by this FLP |
| `--lanes` | hw_concurrency/2 | Parallel processing lanes (CRUs split per lane) |
| `--time-lanes` | 1 | Parallel lanes for time-frame splitting |
| `--n-TFs-buffer` | 1 | Number of TFs to buffer before forwarding |
| `--dump-cmvs-flp` | false | Dump raw CMV vectors per CRU to a ROOT file each TF (for debugging) |

#### `o2-tpc-cmv-distribute`

| Option | Default | Description |
|---|---|---|
| `--crus` | `0-359` | CRU range expected from upstream |
| `--timeframes` | 2000 | Number of TFs aggregated per calibration interval |
| `--firstTF` | -1 | First time frame index; -1 = auto-detect from first incoming TF; values < -1 set an offset of `\|firstTF\|+1` TFs before the first interval begins |
| `--lanes` | 1 | Number of parallel lanes (CRUs are split evenly across lanes) |
| `--output-lanes` | 1 | Number of aggregate pipelines downstream; these lanes rotate whole CMV aggregation intervals, not CRU subsets |
| `--n-TFs-buffer` | 1 | Number of TFs buffered per group in the upstream `o2-tpc-cmv-flp` (must match that workflow's setting) |
| `--send-precise-timestamp` | false | Forward orbit-reset timing information needed by the aggregate workflow for precise CCDB validity timestamps |
| `--drop-data-after-nTFs` | 0 | Drop data for a relative TF slot after this many TFs have passed without receiving all CRUs; 0 uses the default derived from `--check-data-every-n` |
| `--check-data-every-n` | 0 | Check for missing CRU data every N invocations of the run function; -1 disables checking, 0 uses the default (timeframes/2) |
| `--nFactorTFs` | 1000 | Number of TFs to skip before flushing the oldest incomplete aggregation interval |

#### `o2-tpc-cmv-aggregate`

> **Important:** `--n-TFs-buffer` must be set to the same value as in `o2-tpc-cmv-distribute` and `o2-tpc-cmv-flp`. Mismatched values will silently corrupt the relTF mapping and TTree entry count.

| Option | Default | Description |
|---|---|---|
| `--crus` | `0-359` | Full CRU range expected for each aggregate interval |
| `--timeframes` | 2000 | Number of TFs aggregated per calibration interval |
| `--input-lanes` | 1 | Number of aggregate pipelines; must match `o2-tpc-cmv-distribute --output-lanes` |
| `--n-TFs-buffer` | 1 | Number of real TFs packed into one CMV batch from upstream; **must match** `o2-tpc-cmv-distribute --n-TFs-buffer` |
| `--enable-CCDB-output` | false | Forward the CMVContainer TTree as a CCDB object to `o2-calibration-ccdb-populator-workflow` |
| `--use-precise-timestamp` | false | Use orbit-reset timing forwarded by the distribute lane (requires `o2-tpc-cmv-distribute --send-precise-timestamp`) for precise CCDB validity start timestamps |
| `--output-dir` | `none` | Output directory for writing the CMVContainer ROOT file; must exist |
| `--nthreads` | 1 | Number of threads used for CMV preprocessing and compression; each thread processes a contiguous slice of buffered TFs |
| `--use-sparse` | false | Sparse encoding: skip zero time bins (raw uint16 values; combine with `--use-compression-varint` or `--use-compression-huffman` for compressed sparse output) |
| `--use-compression-varint` | false | Delta + zigzag + varint compression over all values; combined with `--use-sparse`: varint-encoded exact values at non-zero positions |
| `--use-compression-huffman` | false | Huffman encoding over all values; combined with `--use-sparse`: Huffman-encoded exact values at non-zero positions |
| `--cmv-zero-threshold` | 0 | Zero out CMV values whose magnitude is below this threshold (ADC) after optional rounding and before compression; 0 disables |
| `--cmv-round-integers-threshold` | 0 | Round values to nearest integer ADC for \|v\| ≤ N ADC before compression; 0 disables |
| `--cmv-dynamic-precision-mean` | 1.0 | Gaussian centre in \|CMV\| (ADC) where the strongest fractional-bit trimming is applied |
| `--cmv-dynamic-precision-sigma` | 0 | Gaussian width (ADC) for smooth CMV fractional-bit trimming; 0 disables |

### Example 1 — Simple usage for testing

```bash
#!/bin/bash

hash="test"
MAX_TFS=1
CRUS="0-359"

ARGS_ALL="-b --session ${USER}.${hash} --shm-segment-size $((8<<30))"

o2-raw-tf-reader-workflow $ARGS_ALL \
  --input-data tf.subset.list \
  --max-tf ${MAX_TFS} |
o2-tfidinfo-writer-workflow $ARGS_ALL \
  --early-forward-policy noraw \
  --fairmq-rate-logging 0 \
  --timeframes-rate-limit ${MAX_TFS} \
  --timeframes-rate-limit-ipcid 583693664 |
o2-tpc-cmv-to-vector $ARGS_ALL \
  --input-spec "A:TPC/RAWDATA" \
  --write-debug-on-error \
  --crus ${CRUS} |
o2-tpc-cmv-flp $ARGS_ALL \
  --crus ${CRUS} |
o2-tpc-cmv-distribute $ARGS_ALL \
  --crus ${CRUS} \
  --output-lanes 1 \
  --send-precise-timestamp \
|
o2-tpc-cmv-aggregate $ARGS_ALL \
  --crus ${CRUS} \
  --output-dir ./ \
  --enable-CCDB-output \
  --cmv-zero-threshold 1.0 \
  --cmv-dynamic-precision-mean 1.0 \
  --cmv-dynamic-precision-sigma 8.0 \
  --use-sparse \
  --use-compression-huffman |
o2-calibration-ccdb-populator-workflow $ARGS_ALL \
  --ccdb-path ccdb-test.cern.ch:8080
```

### Example 2 — Bash scripts for more realistic testing

In a real online setup, multiple FLPs each process their own CRU subset and forward compressed CMV groups to a central aggregator node via ZeroMQ.

**FLP side (`Send.sh`)** — run one instance per FLP (pass `N_FLPs` as first argument):

```bash
#!/bin/bash

# Number of FLPs (passed as first argument, default 1)
N_FLPs=${1:-1}

hash="test"
MAX_TFS=1

minCRU=0
maxCRU=360

ARGS_ALL="-b --shm-segment-size $((8<<30))"

for ((i = 0; i < ${N_FLPs}; i++)); do
  xpos_start=100
  xpos=$((xpos_start + 1000 * $i))

  let diff=${maxCRU}-${minCRU}
  let Start=${minCRU}+$i*${diff}/${N_FLPs}
  let End=$Start+${diff}/${N_FLPs}-1

  crus="$Start-$End"
  echo "FLP $i: crus $crus"

  xterm -hold -geometry 150x41+$xpos+300 -e bash -c "unset PYTHONHOME PYTHONPATH; echo FLP $i;
  o2-raw-tf-reader-workflow $ARGS_ALL \
    --session ${USER}.${hash}.send.$i \
    --input-data tf.subset.list \
    --max-tf ${MAX_TFS} |
  o2-tfidinfo-writer-workflow $ARGS_ALL \
    --session ${USER}.${hash}.send.$i \
    --early-forward-policy noraw \
    --fairmq-rate-logging 0 \
    --timeframes-rate-limit ${MAX_TFS} \
    --timeframes-rate-limit-ipcid $((583693664 + $i)) |
  o2-tpc-cmv-to-vector $ARGS_ALL \
    --session ${USER}.${hash}.send.$i \
    --input-spec 'A:TPC/RAWDATA' \
    --write-debug-on-error \
    --crus ${crus} |
  o2-tpc-cmv-flp $ARGS_ALL \
    --session ${USER}.${hash}.send.$i \
    --crus ${crus} |
  o2-dpl-output-proxy $ARGS_ALL \
    --session ${USER}.${hash}.send.$i \
    --sporadic-inputs \
    --channel-config 'name=downstream,method=connect,address=tcp://localhost:30453,type=push,transport=zeromq' \
    --dataspec 'downstream:TPC/CMVGROUP;downstream:TPC/CMVORBITINFO'; exec bash" &
done
```

Each FLP connects to the aggregator's pull socket on port `30453` and pushes `TPC/CMVGROUP` and `TPC/CMVORBITINFO` messages. The CRU range is automatically split evenly across `N_FLPs`.

**Aggregator side (`Receive.sh`)**:

```bash
#!/bin/bash

hash="test"
CRUS="0-359"

ARGS_ALL="-b --session ${USER}.${hash}.receive --shm-segment-size $((8<<30))"

# ZeroMQ proxy: pull from all FLPs connecting on port 30453
configProxy="name=readout-proxy,type=pull,method=bind,address=tcp://localhost:30453,rateLogging=1,transport=zeromq"

o2-dpl-raw-proxy $ARGS_ALL \
  --channel-config "${configProxy}" \
  --dataspec "A:TPC/CMVGROUP;A:TPC/CMVORBITINFO" |
o2-tpc-cmv-distribute $ARGS_ALL \
  --crus ${CRUS} \
  --output-lanes 1 \
  --send-precise-timestamp \
|
o2-tpc-cmv-aggregate $ARGS_ALL \
  --crus ${CRUS} \
  --output-dir ./ \
  --enable-CCDB-output \
  --cmv-zero-threshold 1.0 \
  --cmv-dynamic-precision-mean 1.0 \
  --cmv-dynamic-precision-sigma 8.0 \
  --use-sparse \
  --use-compression-huffman |
o2-calibration-ccdb-populator-workflow $ARGS_ALL \
  --ccdb-path ccdb-test.cern.ch:8080
```

The aggregator binds the ZeroMQ pull socket and waits for all FLPs to connect. Once `TPC/CMVGROUP` and `TPC/CMVORBITINFO` data arrive, `o2-tpc-cmv-distribute` routes the grouped CMV batches, and `o2-tpc-cmv-aggregate` gathers the full CRU set for each interval, applies preprocessing and compression, writes the object to disk, and uploads it to the CCDB.
