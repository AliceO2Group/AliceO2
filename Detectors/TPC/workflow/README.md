# DPL workflows for the TPC

## TPC reconstruction workflow
The TPC reconstruction workflow starts from the TPC digits, the *clusterer* reconstructs clusters. Those are
*converted* to RAW pages and passed onto the *decoder* providing the decoded (native) cluster format to the
*tracker*.

The workflow consists of the following DPL processors:

* `digit-reader` -> using tool [o2::framework::RootTreeReader](../../../Framework/Utils/include/Utils/RootTreeReader.h)
* `clusterer` -> interfaces [o2::TPC::HwClusterer](../reconstruction/include/TPCReconstruction/HwClusterer.h)
* `converter` -> implements conversions to raw pages, to be moved to a worker class
* `decoder` -> interfaces [o2::TPC::HardwareClusterDecoder](reconstruction/include/TPCReconstruction/HardwareClusterDecoder.h)
* `tracker` -> interfaces [o2::TPC::TPCCATracking](reconstruction/include/TPCReconstruction/TPCCATracking.h)
* `writer` -> implements simple writing to ROOT file

Clusters are created and processed together with the MC labels, however track labels are not yet written
to output file. This will come soon.

### Input data
The input can be created by running the simulation (`o2sim`) and the digitizer workflow (`digitizer-workflow`).
The digitizer workflow produces the file `tpcdigits.root` by default, data is stored in separated branches for
all sectors.

### Quickstart running the reconstruction workflow
The workflow is implemented in the `tpc-reco-workflow` executable.

Display all options
```
tpc-reco-workflow --help
```

Important options for the `digit-reader` as initial publisher
```
--infile arg                          Name of the input file
--treename arg (=o2sim)               Name of the input tree
--digitbranch arg (=TPCDigit)         Digit branch
--mcbranch arg (=TPCDigitMCTruth)     MC info branch
--tpc-sectors arg (=0-35)             TPC sector range, e.g. 5-7,8,9
```

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
--input-type arg (=digits)            digits, clusters, raw
--output-type arg (=tracks)           clusters, raw, tracks
--disable-mc arg (=0)                 disable sending of MC information
--tpc-lanes arg (=1)                  number of parallel lanes up to the tracker
```
Support for all other output types than `tracks` is going to be implemented soon, multiple outputs
will be supported in order to keep the data at intermediate steps.

#### Parallel processing
Parallel processing is controlled by the option `--tpc-lanes n`. The digit reader will fan out to n processing
lanes, each with clusterer, converter and decoder. The tracker will fan in from the parallel lanes.

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

### Current limitations
* track labels not yet written to file
* only the full workflow with input type `digits` and output type `tracks` has been tested so far
* the propagation of MC labels goes together with multiple rearrangements and thus copy
* only one timeframe is processed, this matches the current working scheme of the digitizer workflow
* sequential workflow where the TPC sectors are processed individually and the data is buffered in the
  tracker until complete. We are currently (ab)using DPLs time slice concept to process individual
  TPC sectors. This is probably anyhow solved sufficiently when the trigger timing is implemented in
  DPL
* raw pages are using RawDataHeader version 2 with 4 64bit words, will be converted to version 3 soon

## Open questions
* how to implement further workflow control with multiple timeframes
* clarify the buffering of input in the tracker
* clarify whether or not to call `finishProcessing` of the clusterer
* the tracker reports about clusters being dropped, can be an indication that some conversion step in
  the workflow is incorrect.
