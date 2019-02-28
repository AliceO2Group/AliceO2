# DPL workflows for the TPC

## TPC reconstruction workflow
The TPC reconstruction workflow starts from the TPC digits, the *clusterer* reconstructs clusters in the
[ClusterHardware](../../../DataFormats/Detectors/TPC/include/DataFormatsTPC/ClusterHardware.h) format.
The clusters are written to RAW pages and passed onto the *decoder* providing the decoded (native) cluster
format to the *tracker*.

Note: The format of the raw pages is preliminary and does not reflect what is currently implemented in the CRU.

The workflow consists of the following DPL processors:

* `tpc-digit-reader` -> using tool [o2::framework::RootTreeReader](../../../Framework/Utils/include/Utils/RootTreeReader.h)
* `tpc-clusterer` -> interfaces [o2::TPC::HwClusterer](../reconstruction/include/TPCReconstruction/HwClusterer.h)
* `tpc-cluster-decoder` -> interfaces [o2::TPC::HardwareClusterDecoder](../reconstruction/include/TPCReconstruction/HardwareClusterDecoder.h)
* `tpc-tracker`	-> interfaces [o2::TPC::TPCCATracking](../reconstruction/include/TPCReconstruction/TPCCATracking.h)
* `tpc-track-writer` -> implements simple writing to ROOT file

Depending on the input and output types the default workflow is extended by the following readers and writers:
* `tpc-raw-cluster-reader`
* `tpc-decoded-cluster-reader`
* `tpc-raw-cluster-writer`
* `tpc-decoded-cluster-writer`


MC labels are passed through the workflow along with the data objects and also written together with the
output at the configured stages (see output types).

### Input data
The input can be created by running the simulation (`o2sim`) and the digitizer workflow (`digitizer-workflow`).
The digitizer workflow produces the file `tpcdigits.root` by default, data is stored in separated branches for
all sectors.

The workflow can be started starting from digits, raw clusters, or decoded (native) clusters, or directly attached to the
`digitizer-workflow`, see comment on inputs types below.

### Quickstart running the reconstruction workflow
The workflow is implemented in the `tpc-reco-workflow` executable.

Display all options
```
tpc-reco-workflow --help
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
tpc-reco-workflow --infile tpcdigits.root --tpc-sectors 0-15 --tracker-options "cont refX=83 bz=-5.0068597793"
```

```
tpc-reco-workflow --infile tpcdigits.root --tpc-sectors 0-15 --disable-mc 1 --tracker-options "cont refX=83 bz=-5.0068597793"
```

### Global workflow options:
```
--input-type arg (=digits)            digitizer, digits, raw, decoded-clusters
--output-type arg (=tracks)           digits, raw, decoded-clusters, tracks
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

#### Parallel processing
Parallel processing is controlled by the option `--tpc-lanes n`. The digit reader will fan out to n processing
lanes, each with clusterer, converter and decoder. The tracker will fan in from the parallel lanes.
For each sector, a dedicated DPL data channel is created. The channels are distributed among the lanes.
The default configuration processes sector data belonging together in the same time slice, but in earlier
implementations the sector data was distributed among multiple time slices (thus abusing the DPL time
slice concept). The tracker spec implements optional buffering of the input data, if the set is not complete
within one invocation of the processing function.

#### TPC sector selection
By default, all TPC sectors are processed by the workflow, option `--tpc-sectors` reduces this to a subset.

### Processor options

#### TPC CA tracker
The [tracker spec](src/CATrackerSpec.cxx) interfaces the [o2::TPC::TPCCATracking](../reconstruction/include/TPCReconstruction/TPCCATracking.h)
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
