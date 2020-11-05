<!-- doxy
\page refDetectorsMUONMCHWorkflow Workflows
/doxy -->

# MCH Workflows

<!-- vim-markdown-toc GFM -->

* [Raw to digits](#raw-to-digits)
* [Digits to preclusters](#digits-to-preclusters)
* [Tracking](#tracking)
  * [Original track finder](#original-track-finder)
  * [New track finder](#new-track-finder)
* [Track extrapolation to vertex](#track-extrapolation-to-vertex)
* [Track fitter](#track-fitter)
* [Samplers](#samplers)
  * [Cluster sampler](#cluster-sampler)
  * [Track sampler](#track-sampler)
  * [Vertex sampler](#vertex-sampler)
* [Sinks](#sinks)
  * [Preclusters sink](#preclusters-sink)
  * [Track sink](#track-sink)

<!-- vim-markdown-toc -->

## Raw to digits

`o2-mch-raw-to-digits-workflow`

## Digits to preclusters

`o2-mch-digits-to-preclusters-workflow`

Example of a DPL chain to go from a raw data file to a file of preclusters :

```shell
o2-raw-file-reader-workflow --conf file-reader.cfg --loop 0  -b |
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

## Tracking

### Original track finder

`o2-mch-clusters-to-tracks-original-workflow`

Take as input the list of clusters ([ClusterStruct](../Base/include/MCHBase/ClusterBlock.h)) of the current event with the data description "CLUSTERS" and send the list of MCH tracks ([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)) and the list of associated clusters ([ClusterStruct](../Base/include/MCHBase/ClusterBlock.h)) in two separate messages with the data description "TRACKS" and "TRACKCLUSTERS", respectively.

Options `--l3Current xxx` and `--dipoleCurrent yyy` allow to specify the current in L3 and in the dipole to be used to set the magnetic field.

Option `--moreCandidates` allows to enable the search for more track candidates in stations 4 & 5.

Option `--debug x` allows to enable the debug level x (0 = no debug, 1 or 2).

### New track finder

`o2-mch-clusters-to-tracks-workflow`

Same behavior and options as [Original track finder](#original-track-finder)

## Track extrapolation to vertex

`o2-mch-tracks-to-tracks-at-vertex-workflow`

Take as input the vertex position (`Point3D<double>`) and the list of MCH tracks ([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)) of the current event with the data description "VERTEX" and "TRACKS", respectively, and send the list of tracks at vertex (`TrackAtVtxStruct` as described below) with the data description "TRACKSATVERTEX".

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

`o2-mch-tracks-to-tracks-workflow`

Take as input the list of MCH tracks ([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)) and the list of associated clusters ([ClusterStruct](../Base/include/MCHBase/ClusterBlock.h)) of the current event with the data description "TRACKSIN" and "TRACKCLUSTERSIN", respectively, and send the list of refitted MCH tracks ([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)) and the list of associated clusters ([ClusterStruct](../Base/include/MCHBase/ClusterBlock.h)) in two separate messages with the data description "TRACKS" and "TRACKCLUSTERS", respectively.

Options `--l3Current xxx` and `--dipoleCurrent yyy` allow to specify the current in L3 and in the dipole to be used to set the magnetic field.

## Samplers

### Cluster sampler

```shell
o2-mch-clusters-sampler-workflow --infile "clusters.in"
```

where `clusters.in` is a binary file containing for each event:

* number of clusters (int)
* number of associated digits (int)
* list of clusters ([ClusterStruct](../Base/include/MCHBase/ClusterBlock.h))
* list of associated digits ([Digit](../Base/include/MCHBase/Digit.h))

Send the list of clusters ([ClusterStruct](../Base/include/MCHBase/ClusterBlock.h)) of the current event with the data description "CLUSTERS".

### Track sampler

`o2-mch-tracks-sampler-workflow --infile "tracks.in"`

where `tracks.in` is a binary file with the same format as the one written by the workflow [o2-mch-tracks-sink-workflow](#track-sink)

Send the list of MCH tracks ([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)) and the list of associated clusters ([ClusterStruct](../Base/include/MCHBase/ClusterBlock.h)) in two separate messages with the data description "TRACKS" and "TRACKCLUSTERS", respectively.

Option `--forTrackFitter` allows to send the messages with the data description "TRACKSIN" and "TRACKCLUSTERSIN", respectively, as expected by the workflow [o2-mch-tracks-to-tracks-workflow](#track-fitter).

### Vertex sampler

`o2-mch-vertex-sampler-workflow`

Send the vertex position (`Point3D<double>`) of the current event with the data description "VERTEX".

Option `--infile "vertices.in"` allows to read the position of the vertex from the binary file `vertices.in` containing for each event:

* event number (int)
* x (double)
* y (double)
* z (double)

If no binary file is provided, the vertex is always set to (0,0,0).

## Sinks

### Preclusters sink

`o2-mch-preclusters-sink-workflow`

### Track sink

`o2-mch-tracks-sink-workflow --outfile "tracks.out"`

Take as input the list of tracks at vertex ([TrackAtVtxStruct](#track-extrapolation-to-vertex)), the list of MCH tracks ([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)) and the list of associated clusters ([ClusterStruct](../Base/include/MCHBase/ClusterBlock.h)) of the current event with the data description "TRACKSATVERTEX", "TRACKS" and "TRACKCLUSTERS", respectively, and write them in the binary file `tracks.out` with the following format:

* number of tracks at vertex (int)
* number of MCH tracks (int)
* number of associated clusters (int)
* list of tracks at vertex ([TrackAtVtxStruct](#track-extrapolation-to-vertex))
* list of MCH tracks ([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h))
* list of associated clusters ([ClusterStruct](../Base/include/MCHBase/ClusterBlock.h))

Option `--tracksAtVertexOnly` allows to take as input and write only the tracks at vertex (number of MCH tracks and number of associated clusters = 0).

Option `--mchTracksOnly` allows to take as input and write only the MCH tracks and associated clusters (number of tracks at vertex = 0).
