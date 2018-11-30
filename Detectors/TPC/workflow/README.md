# DPL workflows for the TPC

## TPC reconstruction workflow
The TPC reconstruction workflow starts from the TPC digits, the *clusterer* reconstructs clusters. Those are
*converted* to RAW pages and passed onto the *decoder* providing the decoded (native) cluster format to the
*tracker*.

The workflow consists of the following DPL processors:

* `tpc-digit-reader` -> using tool [o2::framework::RootTreeReader](../../../Framework/Utils/include/Utils/RootTreeReader.h)
* `tpc-clusterer` -> interfaces [o2::TPC::HwClusterer](../reconstruction/include/TPCReconstruction/HwClusterer.h)
* `tpc-cluster-converter`-> implements conversions to raw pages, to be moved to a worker class
* `tpc-cluster-decoder` -> interfaces [o2::TPC::HardwareClusterDecoder](reconstruction/include/TPCReconstruction/HardwareClusterDecoder.h)
* `tpc-tracker`	-> interfaces [o2::TPC::TPCCATracking](reconstruction/include/TPCReconstruction/TPCCATracking.h)
* `tpc-track-writer` -> implements simple writing to ROOT file

MC labels are passed through the workflow along with the data objects and also written together with the
output at the configured stages (see output types).

### Input data
The input can be created by running the simulation (`o2sim`) and the digitizer workflow (`digitizer-workflow`).
The digitizer workflow produces the file `tpcdigits.root` by default, data is stored in separated branches for
all sectors.

The workflow can currently start from the digits or the clusters, or directly attached to the `digitizer-workflow`,
see comment on inputs types.

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
```

The `tpc-cluster-reader` uses the same options except the branch name configuration
--clusterbranch arg (=TPCCluster)           Cluster branch
--clustermcbranch arg (=TPCClusterMCTruth)  MC label branch

Options for the `tpc-track-writer` process
```
--outfile arg (=tpctracks.root)                Name of the input file
--treename arg (=o2sim)                        Name of output tree
--track-branch-name arg (=TPCTracks)           Branch name for TPC tracks
--trackmc-branch-name arg (=TPCTracksMCTruth)  Branch name for TPC track mc labels
```

Examples:
```
tpc-reco-workflow --infile tpcdigits.root --tpc-sectors 0-15
```

```
tpc-reco-workflow --infile tpcdigits.root --tpc-sectors 0-15 --disable-mc 1
```

### Global workflow options:
```
--input-type arg (=digits)            digitizer, digits, clusters, raw
--output-type arg (=tracks)           clusters, raw, decoded-clusters, tracks
--disable-mc arg (=0)                 disable sending of MC information
--tpc-lanes arg (=1)                  number of parallel lanes up to the tracker
--tpc-sectors arg (=0-35)             TPC sector range, e.g. 5-7,8,9
```

#### Input Type
The input and output types `raw` have not yet been implemented.

Input type `digitizer` will create the clusterers with dangling input, this is used
to connect the reconstruction workflow directly to the digitizer workflow.

#### Output Type
The output type selects up to which final product the workflow is executed. Multiple outputs
are supported in order to write data at intermediate steps, e.g.
```
--output-type clusters,tracks
```

#### Parallel processing
Parallel processing is controlled by the option `--tpc-lanes n`. The digit reader will fan out to n processing
lanes, each with clusterer, converter and decoder. The tracker will fan in from the parallel lanes.

#### TPC sector selection
By default, all TPC sectors are processed by the workflow, option `--tpc-sectors` reduces this to a subset.

### Processor options

#### TPC CA tracker
The [tracker spec](src/CATrackerSpec.cxx) interfaces the [o2::TPC::TPCCATracking](reconstruction/include/TPCReconstruction/TPCCATracking.h) worker class which can be initialized using an option string. The processor spec defines the option `--tracker-option`. Currently, the tracker should be run with options
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
* input and output types `raw` are not yet implemented

* the propagation of MC labels goes together with multiple rearrangements and thus copy

* sequential workflow where the TPC sectors are processed individually and the data is buffered in the
  tracker until complete. We are currently (ab)using DPLs time slice concept to process individual
  TPC sectors. This is probably anyhow solved sufficiently when the trigger timing is implemented in
  DPL

* raw pages are using RawDataHeader version 2 with 4 64bit words, will be converted to version 3 soon

* implement configuration from the GRP

## Open questions
* clarify the buffering of input in the tracker
* clarify whether or not to call `finishProcessing` of the clusterer
