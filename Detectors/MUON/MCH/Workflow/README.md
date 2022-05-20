<!-- doxy
\page refDetectorsMUONMCHWorkflow Workflows
/doxy -->

# MCH Workflows

<!-- vim-markdown-toc GFM -->

* [A note for developers](#a-note-for-developers)
* [Raw to digits](#raw-to-digits)
* [Digit filtering](#digit-filtering)
* [Time clustering](#time-clustering)
* [Event finding](#event-finding)
* [Preclustering](#preclustering)
* [Clustering](#clustering)
* [CTF encoding/decoding](#ctf-encodingdecoding)
* [Local to global cluster transformation](#local-to-global-cluster-transformation)
* [Tracking](#tracking)
  * [Original track finder](#original-track-finder)
  * [New track finder](#new-track-finder)
* [Track extrapolation to vertex](#track-extrapolation-to-vertex)
* [Track fitter](#track-fitter)
* [Samplers](#samplers)
  * [Digit sampler](#digit-sampler)
  * [Cluster sampler](#cluster-sampler)
  * [Cluster reader](#cluster-reader)
  * [Track sampler](#track-sampler)
  * [Track reader](#track-reader)
  * [Vertex sampler](#vertex-sampler)
* [Sinks](#sinks)
  * [Precluster sink](#precluster-sink)
  * [Cluster sink](#cluster-sink)
  * [Cluster writer](#cluster-writer)
  * [Track sink](#track-sink)
  * [Track writer](#track-writer)

<!-- vim-markdown-toc -->


## A note for developers

When defining a function that returns a `DataProcessorSpec`, please stick to the following pattern for its parameters :

    DataProcessorSpec getXXX([bool useMC], const char* specName="mch-xxx", other parameters);

* first parameter, if relevant, should be a boolean indicating whether the processor has to deal with Monte Carlo data or not. For a processor that never has to deal with MC, leave that parameter out
* second parameter is the name by which that device will be referenced in all log files, so, in order to be easily recognizable, it *must* start with the prefix `mch-`
* the rest of the parameters (if any) is specific to each device

## Raw to digits

```shell
o2-mch-raw-to-digits-workflow
```

The workflow accepts the following options:

* `--debug`: enable verbose output
* `--dataspec`: selection string for the input data (default: `"TF:MCH/RAWDATA"`)
* `--cru-map`: path to custom CRU mapping file
* `--fec-map`: path to custom FEC mapping file
* `--ds2manu`: convert channel numbering from Run3 to Run1-2 order

Example of a DPL chain to go from a raw data file to a file of preclusters :

```shell
o2-raw-file-sampler-workflow --input-conf file-reader.cfg --loop 0  -b |
o2-mch-raw-to-digits-workflow -b |
o2-mch-digits-to-preclusters-workflow -b |
o2-mch-preclusters-sink-workflow -b
```

where the `file-reader.cfg` looks like this:

```ini
[input-0]
dataOrigin = MCH
dataDescription = RAWDATA
filePath = /home/data/data-de819-ped-raw.raw
```

## Digit filtering

Filter out (i.e. remove) some digits [more...](/Detectors/MUON/MCH/DigitFiltering/README.md)

## Time clustering

Cluster ROFs per time, thus making IR ranges of interest. [more...](/Detectors/MUON/MCH/TimeClustering/README.md)

## Event finding

```shell
o2-mch-event-finder-workflow
```

Inputs :
- list of all MCH digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)) in the current time frame, with the (default) data description `F-DIGITS` (can be changed with `--input-digits-data-description` option)
- list of MCH ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the digits associated to each ROF, with the (default) data description `F-DIGITROFS` (can be changed with `--input-digit-rofs-data-description` option)
- list of MID ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MID/include/DataFormatsMID/ROFRecord.h)) used to trigger the event finding, with the data description `TRACKROFS`

Outputs :
- list of MCH digits associated to an event, with the (default) data description `E-F-DIGITS` (can be changed with `--output-digits-data-description` option)
- list of MCH ROF records pointing to the digits associated to each event, with the (default) data description `E-F-DIGITROFS` (can be changed with `--output-digit-rofs-data-description` option)

Option `--mch-config "file.json"` or `--mch-config "file.ini"` allows to change the triggering parameters from a configuration file. This file can be either in JSON or in INI format, as described below:

* Example of configuration file in JSON format:
```json
{
    "MCHTriggering": {
        "triggerRange[0]": "-3",
        "triggerRange[1]": "3"
    }
}
```
* Example of configuration file in INI format:
```ini
[MCHTriggering]
triggerRange[0]=-3
triggerRange[1]=3
```

Option `--configKeyValues "key1=value1;key2=value2;..."` allows to change the triggering parameters from the command line. The parameters changed from the command line will supersede the ones changed from a configuration file.

* Example of parameters changed from the command line:
```shell
--configKeyValues "MCHTriggering.triggerRange[0]=-3;MCHTriggering.triggerRange[1]=3"
```
## Preclustering

```shell
o2-mch-digits-to-preclusters-workflow
```

Take as input the list of all digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)) in the current time frame, with the data description "DIGITS", and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the digits associated to each interaction.
The ROF records input can have the data description "inputrofs:MCH/DIGITROFS" if the direct output of the raw decoder is used, or "inputrofs:MCH/TIMECLUSTERROFS" if the time clustering output is used (default option). The ROF input description can be set on the command line via the `rof-spec` option.

Send the list of all preclusters ([PreCluster](../Base/include/MCHBase/PreCluster.h)) in the time frame, the list of all associated digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)) and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the preclusters associated to each interaction in three separate messages with the data description "PRECLUSTERS", "PRECLUSTERDIGITS" and "PRECLUSTERROFS", respectively.

Option `--check-no-leftover-digits xxx` allows to drop an error message (`xxx = "error"` (default)) or an exception (`xxx = "fatal"`) in case not all the input digits end up in a precluster, or to disable this check (`xxx = "off"`).

## Clustering

```shell
o2-mch-preclusters-to-clusters-original-workflow
```

Take as input the list of all preclusters ([PreCluster](../Base/include/MCHBase/PreCluster.h)) in the current time frame, the list of all associated digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)) and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the preclusters associated to each interaction, with the data description "PRECLUSTERS", "PRECLUSTERDIGITS" and "PRECLUSTERROFS", respectively. Send the list of all clusters ([Cluster](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h)) in the time frame, the list of all associated digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)) and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the clusters associated to each interaction in three separate messages with the data description "CLUSTERS", "CLUSTERDIGITS" and "CLUSTERROFS", respectively.

Option `--run2-config` allows to configure the clustering to process run2 data.

Option `--mch-config "file.json"` or `--mch-config "file.ini"` allows to change the clustering parameters from a configuration file. This file can be either in JSON or in INI format, as described below:

* Example of configuration file in JSON format:
```json
{
    "MCHClustering": {
        "lowestPadCharge": "4.",
        "defaultClusterResolution": "0.4"
    }
}
```
* Example of configuration file in INI format:
```ini
[MCHClustering]
lowestPadCharge=4.
defaultClusterResolution=0.4
```

Option `--configKeyValues "key1=value1;key2=value2;..."` allows to change the clustering parameters from the command line. The parameters changed from the command line will supersede the ones changed from a configuration file.

* Example of parameters changed from the command line:
```shell
--configKeyValues "MCHClustering.lowestPadCharge=4.;MCHClustering.defaultClusterResolution=0.4"
```

## CTF encoding/decoding

Entropy encoding is done be attaching the `o2-mch-entropy-encoder-workflow` to the output of `DIGITS` and `DIGITROF` data-descriptions, providing `Digit` and `ROFRecord` respectively. Afterwards the encoded data can be stored by the `o2-ctf-writer-workflow`.

```shell
o2-raw-file-reader-workflow --input-conf raw/MCH/MCHraw.cfg | o2-mch-raw-to-digits-workflow  | o2-mch-entropy-encoder-workflow | o2-ctf-writer-workflow --onlyDet MCH
```

The decoding is done automatically by the `o2-ctf-reader-workflow`.

## Local to global cluster transformation

Converts the clusters coordinates from local (2D within detection element plane) to global (3D within Alice reference frame) [more...](/Detectors/MUON/MCH/Geometry/Transformer/README.md)

## Tracking

### Original track finder

```shell
o2-mch-clusters-to-tracks-original-workflow
```

Take as input the list of all clusters ([Cluster](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h)) in the current time frame, with the data description "CLUSTERS", and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the clusters associated to each interaction, with the data description "CLUSTERROFS". Send the list of all MCH tracks ([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)) in the time frame, the list of all associated clusters ([Cluster](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h)) and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the tracks associated to each interaction in three separate messages with the data description "TRACKS", "TRACKCLUSTERS" and "TRACKROFS", respectively.

Options `--l3Current xxx` and `--dipoleCurrent yyy` allow to specify the current in L3 and in the dipole to be used to set the magnetic field.

Option `--debug x` allows to enable the debug level x (0 = no debug, 1 or 2).

Option `--mch-config "file.json"` or `--mch-config "file.ini"` allows to change the tracking parameters from a configuration file. This file can be either in JSON or in INI format, as described below:

* Example of configuration file in JSON format:
```json
{
    "MCHTracking": {
        "chamberResolutionY": "0.1",
        "requestStation[1]": "false",
        "moreCandidates": "true"
    }
}
```
* Example of configuration file in INI format:
```ini
[MCHTracking]
chamberResolutionY=0.1
requestStation[1]=false
moreCandidates=true
```

Option `--configKeyValues "key1=value1;key2=value2;..."` allows to change the tracking parameters from the command line. The parameters changed from the command line will supersede the ones changed from a configuration file.

* Example of parameters changed from the command line:
```shell
--configKeyValues "MCHTracking.chamberResolutionY=0.1;MCHTracking.requestStation[1]=false;MCHTracking.moreCandidates=true"
```

### New track finder

```shell
o2-mch-clusters-to-tracks-workflow
```

Same behavior and options as [Original track finder](#original-track-finder)

## Track extrapolation to vertex

```shell
o2-mch-tracks-to-tracks-at-vertex-workflow
```

Take as input the list of all MCH tracks ([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)) in the current time frame, the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the tracks associated to each interaction and their vertex position (`Point3D<double>`), with the data description "TRACKS", "TRACKROFS" and "VERTICES", respectively. Send the list of all tracks at vertex (`TrackAtVtxStruct` as described below) in the time frame with the data description "TRACKSATVERTEX".

```c++
struct TrackAtVtxStruct {
  TrackParamStruct paramAtVertex{};
  double dca = 0.;
  double rAbs = 0.;
  int mchTrackIdx = 0;
};
```

Options `--l3Current xxx` and `--dipoleCurrent yyy` allow to specify the current in L3 and in the dipole to be used to set the magnetic field.

## Track fitter

```shell
o2-mch-tracks-to-tracks-workflow
```

Take as input the list of all MCH tracks ([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)) in the current time frame, the list of all associated clusters ([Cluster](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h)) and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the tracks associated to each interaction, with the data description "TRACKSIN", "TRACKCLUSTERSIN" and "TRACKROFSIN", respectively. Send the list of all refitted MCH tracks ([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)) in the time frame, the list of all associated clusters ([Cluster](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h)) and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the tracks associated to each interaction in three separate messages with the data description "TRACKS", "TRACKCLUSTERS" and "TRACKROFS", respectively.

Options `--l3Current xxx` and `--dipoleCurrent yyy` allow to specify the current in L3 and in the dipole to be used to set the magnetic field.

## Samplers

### Digit sampler

```shell
o2-mch-digits-sampler-workflow --infile "digits.in"
```

where `digits.in` is a binary file containing for each event:

* number of digits (int)
* list of digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h))

Send the list of all digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)) in the current time frame, with the data description "DIGITS", and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the digits associated to each interaction, with the data description "DIGITROFS".

Option `--useRun2DigitUID` allows to specify that the input digits data member mPadID contains the digit UID in run2 format and need to be converted in the corresponding run3 pad ID.

Option `--print` allows to print the digitUID - padID conversion in the terminal.

Option `--nEvents xxx` allows to limit the number of events to process (all by default).

Option `--event xxx` allows to process only this event.

Option `--nEventsPerTF xxx` allows to set the number of events (i.e. ROF records) to send per time frame (default = 1).

### Cluster sampler

```shell
o2-mch-clusters-sampler-workflow --infile "clusters.in" [--global]
```

where `clusters.in` is a binary file containing for each event:

* number of clusters (int)
* number of associated digits (int)
* list of clusters ([Cluster](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h))
* list of associated digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h))

Send the list of all clusters ([Cluster](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h)) in the current time frame, with the data description "CLUSTERS" (or "GLOBALCLUSTERS" if `--global` option is used), and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the clusters associated to each interaction, with the data description "CLUSTERROFS".

Option `--nEventsPerTF xxx` allows to set the number of events (i.e. ROF records) to send per time frame (default = 1).

### Cluster reader

```
o2-mch-clusters-reader-workflow --infile mchclusters.root [--enable-mc] [--local] [--digits]
```
Send the list of all clusters ([Cluster](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h)) in the current time frame, with the data description "GLOBALCLUSTERS", and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the clusters associated to each interaction, with the data description "CLUSTERROFS".

Option `--local` assumes that clusters are in the local coordinate system and send them with the description "CLUSTERS".

Option `--digits` allows to also send the associated digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)) with the data description "CLUSTERDIGITS".

Option `--enable-mc` allows to also send the cluster MC labels with the data description "CLUSTERLABELS".

### Track sampler

```shell
o2-mch-tracks-sampler-workflow --infile "tracks.in"
```

where `tracks.in` is a binary file with the same format as the one written by the workflow [o2-mch-tracks-sink-workflow](#track-sink)

Send the list of all MCH tracks ([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)) in the current time frame, the list of all associated clusters ([Cluster](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h)) and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the tracks associated to each interaction in three separate messages with the data description "TRACKS", "TRACKCLUSTERS" and "TRACKROFS", respectively.

Option `--forTrackFitter` allows to send the messages with the data description "TRACKSIN", "TRACKCLUSTERSIN" and TRACKROFSIN, respectively, as expected by the workflow [o2-mch-tracks-to-tracks-workflow](#track-fitter).

Option `--nEventsPerTF xxx` allows to set the number of events (i.e. ROF records) to send per time frame (default = 1).

### Track reader

```
o2-mch-tracks-reader-workflow --infile mchtracks.root
```

Does the same work as the [Track sampler](#track-sampler) but starting from a Root file (`mchtracks.root`)  containing `TRACKS`, `TRACKROFS` and `TRACKCLUSTERS` containers written e.g. by the [o2-mch-tracks-writer-workflow](#track-writer).
Note that a very basic utility also exists to get a textual dump of a Root tracks file : `o2-mch-tracks-file-dumper`.

### Vertex sampler

```shell
o2-mch-vertex-sampler-workflow
```

Take as input the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) in the current time frame, with the data description "TRACKROFS". Send the list of all vertex positions (`Point3D<double>`) in the time frame, one per interaction, with the data description "VERTICES".

Option `--infile "vertices.in"` allows to read the position of the vertex from the binary file `vertices.in` containing for each event:

* event number (int)
* x (double)
* y (double)
* z (double)

If no binary file is provided, the vertex is always set to (0,0,0).

## Sinks

### Precluster sink

```shell
o2-mch-preclusters-sink-workflow --outfile "preclusters.out"
```

Take as input the list of all preclusters ([PreCluster](../Base/include/MCHBase/PreCluster.h)) in the current time frame, the list of all associated digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)) and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the preclusters associated to each interaction, with the data description "PRECLUSTERS", "PRECLUSTERDIGITS" and "PRECLUSTERROFS", respectively, and write them event-by-event in the binary file `preclusters.out` with the following format for each event:

* number of preclusters (int)
* number of associated digits (int)
* list of preclusters ([PreCluster](../Base/include/MCHBase/PreCluster.h))
* list of associated digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h))

Option `--txt` allows to write the preclusters in the output file in text format.

Option `--useRun2DigitUID` allows to convert the run3 pad ID stored in the digit data member mPadID into a digit UID in run2 format.

### Cluster sink

```shell
o2-mch-clusters-sink-workflow --outfile "clusters.out" [--txt] [--no-digits] [--global]
```

Take as input the list of all clusters ([Cluster](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h)) in the current time frame, and, optionnally, the list of all associated digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)) and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the clusters associated to each interaction, with the data description "CLUSTERS" (or "GLOBALCLUSTERS" if `--global` option is used), "CLUSTERDIGITS" (unless `--no-digits` option is used) and "CLUSTERROFS", respectively, and write them event-by-event in the binary file `clusters.out` with the following format for each event:

* number of clusters (int)
* number of associated digits (int) (= 0 if `--no-digits` is used)
* list of clusters ([Cluster](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h))
* list of associated digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h))(unless option `--no-digits` is used)

Option `--txt` allows to write the clusters in the output file in text format.

Option `--useRun2DigitUID` allows to convert the run3 pad ID stored in the digit data member mPadID into a digit UID in run2 format.

### Cluster writer

```
o2-mch-clusters-writer-workflow [--enable-mc] [--local] [--no-digits]
```
Take as input the list of all clusters ([Cluster](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h)) in the current time frame, with the data description "GLOBALCLUSTERS", the list of associated digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)), with the data description "CLUSTERDIGITS", and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the clusters associated to each interaction, with the data description "CLUSTERROFS", and write them in the root file "mchclusters.root".

Option `--local` allows to write the list of clusters in the local coordinate system, with the data description "CLUSTERS".

Option `--no-digits` allows to do not write the associated digits.

Option `--enable-mc` allows to also write the cluster MC labels, with the data description "CLUSTERLABELS".

### Track sink

```shell
o2-mch-tracks-sink-workflow --outfile "tracks.out"
```

Take as input the list of all tracks at vertex ([TrackAtVtxStruct](#track-extrapolation-to-vertex)) in the current time frame, the list of all MCH tracks ([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)), the list of all associated clusters ([Cluster](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h)) and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the MCH tracks associated to each interaction, with the data description "TRACKSATVERTEX", "TRACKS", "TRACKCLUSTERS" and "TRACKROFS", respectively, and write them event-by-event in the binary file `tracks.out` with the following format for each event:

* number of tracks at vertex (int)
* number of MCH tracks (int)
* number of associated clusters (int)
* list of tracks at vertex ([TrackAtVtxStruct](#track-extrapolation-to-vertex))
* list of MCH tracks ([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h))
* list of associated clusters ([Cluster](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h))

Option `--tracksAtVertexOnly` allows to take as input and write only the tracks at vertex (number of MCH tracks and number of associated clusters = 0).

Option `--mchTracksOnly` allows to take as input and write only the MCH tracks and associated clusters (number of tracks at vertex = 0).

### Track writer

```shell
o2-mch-tracks-writer-workflow --outfile "mchtracks.root"
```

Does the same kind of work as the [track sink](#track-sink) but the output is in Root format instead of custom binary one. It is implemented using the generic [MakeRootTreeWriterSpec](/DPLUtils/MakeRootTreeWriterSpec.h) and thus offers the same options.
