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
* [Track extrapolation to vertex](#track-extrapolation-to-vertex)
* [Track fitter](#track-fitter)
* [Error merger](#error-merger)
* [Readers/Writers and Samplers/Sinks](#readerswriters-and-samplerssinks)
    * [Error reader](#error-reader)
    * [Vertex sampler](#vertex-sampler)
    * [Error writer](#error-writer)

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

Group the digits in preclusters. [more...](/Detectors/MUON/MCH/PreClustering/README.md)

## Clustering

```shell
o2-mch-preclusters-to-clusters-original-workflow
```

Take as input the list of all preclusters ([PreCluster](../Base/include/MCHBase/PreCluster.h)) in the current time frame, the list of all associated digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)) and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the preclusters associated to each interaction, with the data description "PRECLUSTERS", "PRECLUSTERDIGITS" and "PRECLUSTERROFS", respectively. Send the list of all clusters ([Cluster](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h)) in the time frame, the list of all associated digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)), the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the clusters associated to each interaction and the list of processing errors ([Error](../Base/include/MCHBase/Error.h)) in four separate messages with the data description "CLUSTERS", "CLUSTERDIGITS", "CLUSTERROFS" and "CLUSTERERRORS", respectively.

Option `--run2-config` allows to configure the clustering to process run2 data.

Option `--mch-config "file.json"` or `--mch-config "file.ini"` allows to change the clustering parameters from a configuration file. This file can be either in JSON or in INI format, as described below:

* Example of configuration file in JSON format:
```json
{
    "MCHClustering": {
        "lowestPadCharge": "4.",
        "defaultClusterResolutionX": "0.4",
        "defaultClusterResolutionY": "0.4"
    }
}
```
* Example of configuration file in INI format:
```ini
[MCHClustering]
lowestPadCharge=4.
defaultClusterResolutionX=0.4
defaultClusterResolutionY=0.4
```

Option `--configKeyValues "key1=value1;key2=value2;..."` allows to change the clustering parameters from the command line. The parameters changed from the command line will supersede the ones changed from a configuration file.

* Example of parameters changed from the command line:
```shell
--configKeyValues "MCHClustering.lowestPadCharge=4.;MCHClustering.defaultClusterResolutionX=0.4;MCHClustering.defaultClusterResolutionY=0.4"
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

Combine the clusters to reconstruct the tracks. [more...](/Detectors/MUON/MCH/Tracking/README.md)

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

Refit the tracks to their associated clusters. [more...](/Detectors/MUON/MCH/Tracking/README.md)

## Error merger

```shell
o2-mch-errors-merger-workflow
```

Take as input the list of all MCH preclustering, clustering and tracking errors ([Error](../Base/include/MCHBase/Error.h)) in the current time frame, with the data description "PRECLUSTERERRORS", "CLUSTERERRORS" and "TRACKERRORS", respectively. Send the merged list of all MCH processing errors ([Error](../Base/include/MCHBase/Error.h)) in the time frame, with the data description "PROCERRORS".

Options `--disable-preclustering-errors` allows to skip the preclustering errors.

Options `--disable-clustering-errors` allows to skip the clustering errors.

Options `--disable-tracking-errors` allows to skip the tracking errors.

## Readers/Writers and Samplers/Sinks

Readers (writers) workflows are reading from (to) Root files.
[more...](/Detectors/MUON/MCH/IO/README.md)

Samplers (sinks) workflows are reading from (to) files written in MCH custom
binary format(s). [more...](/Detectors/MUON/MCH/DevIO/README.md)

### Error reader

```shell
o2-mch-errors-reader-workflow --infile mcherrors.root
```

Send the list of all MCH processing errors ([Error](../Base/include/MCHBase/Error.h)) in the current time frame, with the data description "PROCERRORS".

Option `--input-dir` allows to set the name of the directory containing the input file (default = current directory).

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

### Error writer

```shell
o2-mch-errors-writer-workflow
```

Take as input the list of all MCH processing errors ([Error](../Base/include/MCHBase/Error.h)) in the current time frame, with the data description "PROCERRORS", and write it in the root file "mcherrors.root".
