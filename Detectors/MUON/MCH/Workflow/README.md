<!-- doxy
\page refDetectorsMUONMCHWorkflow Workflows
/doxy -->

# MCH Workflows

## Raw data decoding:

`o2-mch-raw-to-digits-workflow`

## Pre-clustering:

`o2-mch-digits-to-preclusters-workflow`

## Pre-clustering sink:

`o2-mch-preclusters-sink-workflow`

## Example of DPL chain:

`o2-raw-file-reader-workflow --conf file-reader.cfg --loop 0  -b | o2-mch-raw-to-digits-workflow -b | o2-mch-digits-to-preclusters-workflow -b | o2-mch-preclusters-sink-workflow -b`

where the `file-reader.cfg` looks like this:

    [input-0]
    dataOrigin = MCH
    dataDescription = RAWDATA     
    filePath = /home/data/data-de819-ped-raw.raw

## Cluster sampler

`o2-mch-clusters-sampler-workflow --infile "clusters.in"`

where `clusters.in` is a binary file containing for each event:
> number of clusters (int)<br>
> number of associated digits (int)<br>
> list of clusters ([ClusterStruct](../Base/include/MCHBase/ClusterBlock.h))<br>
> list of associated digits ([Digit](../Base/include/MCHBase/Digit.h))<br>

Send the list of clusters ([ClusterStruct](../Base/include/MCHBase/ClusterBlock.h)) of the current event with the data
description "CLUSTERS".

## Original track finder

`o2-mch-clusters-to-tracks-original-workflow`

Take as input the list of clusters ([ClusterStruct](../Base/include/MCHBase/ClusterBlock.h)) of the current event with
the data description "CLUSTERS" and send the list of MCH tracks
([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)) and the list of associated
clusters ([ClusterStruct](../Base/include/MCHBase/ClusterBlock.h)) in two separate messages with the data description
"TRACKS" and "TRACKCLUSTERS", respectively.

Options `--l3Current xxx` and `--dipoleCurrent yyy` allow to specify the current in L3 and in the dipole to be used to
set the magnetic field.

Option `--moreCandidates` allows to enable the search for more track candidates in stations 4 & 5.

Option `--debug x` allows to enable the debug level x (0 = no debug, 1 or 2).

## New track finder

`o2-mch-clusters-to-tracks-workflow`

Take as input the list of clusters ([ClusterStruct](../Base/include/MCHBase/ClusterBlock.h)) of the current event with
the data description "CLUSTERS" and send the list of MCH tracks
([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)) and the list of associated
clusters ([ClusterStruct](../Base/include/MCHBase/ClusterBlock.h)) in two separate messages with the data description
"TRACKS" and "TRACKCLUSTERS", respectively.

Options `--l3Current xxx` and `--dipoleCurrent yyy` allow to specify the current in L3 and in the dipole to be used to
set the magnetic field.

Option `--moreCandidates` allows to enable the search for more track candidates in stations 4 & 5.

Option `--debug x` allows to enable the debug level x (0 = no debug, 1 or 2).

## Vertex sampler

`o2-mch-vertex-sampler-workflow`

Send the vertex position (`Point3D<double>`) of the current event with the data description "VERTEX".

Option `--infile "vertices.in"` allows to read the position of the vertex from the binary file `vertices.in` containing
for each event:
> event number (int)<br>
> x (double)<br>
> y (double)<br>
> z (double)<br>

If no binary file is provided, the vertex is always set to (0,0,0).

## Track extrapolation to vertex

`o2-mch-tracks-to-tracks-at-vertex-workflow`

Take as input the vertex position (`Point3D<double>`) and the list of MCH tracks
([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)) of the current event with
the data description "VERTEX" and "TRACKS", respectively, and send the list of tracks at vertex (`TrackAtVtxStruct` as
described below) with the data description "TRACKSATVERTEX".
```c++
  struct TrackAtVtxStruct {
    TrackParamStruct paramAtVertex{};
    double dca = 0.;
    double rAbs = 0.;
    int mchTrackIdx = 0;
  };
```

Options `--l3Current xxx` and `--dipoleCurrent yyy` allow to specify the current in L3 and in the dipole to be used to
set the magnetic field.

## Track sink

`o2-mch-tracks-sink-workflow --outfile "tracks.out"`

Take as input the list of tracks at vertex (`TrackAtVtxStruct` as described above), the list of MCH tracks
([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)) and the list of associated
clusters ([ClusterStruct](../Base/include/MCHBase/ClusterBlock.h)) of the current event with the data description
"TRACKSATVERTEX", "TRACKS" and "TRACKCLUSTERS", respectively, and write them in the binary file `tracks.out` with the
following format:
> number of tracks at vertex (int)<br>
> number of MCH tracks (int)<br>
> number of associated clusters (int)<br>
> list of tracks at vertex (`TrackAtVtxStruct` as described above)<br>
> list of MCH tracks ([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h))<br>
> list of associated clusters ([ClusterStruct](../Base/include/MCHBase/ClusterBlock.h))<br>

Option `--tracksAtVertexOnly` allows to take as input and write only the tracks at vertex (number of MCH tracks and
number of associated clusters = 0).

Option `--mchTracksOnly` allows to take as input and write only the MCH tracks and associated clusters (number of tracks
at vertex = 0).

## Track sampler

`o2-mch-tracks-sampler-workflow --infile "tracks.in"`

where `tracks.in` is a binary file with the same format as the one written by the workflow `o2-mch-tracks-sink-workflow`
described above.

Send the list of MCH tracks ([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h))
and the list of associated clusters ([ClusterStruct](../Base/include/MCHBase/ClusterBlock.h)) in two separate messages
with the data description "TRACKS" and "TRACKCLUSTERS", respectively.

Option `--forTrackFitter` allows to send the messages with the data description "TRACKSIN" and "TRACKCLUSTERSIN",
respectively, as expected by the workflow `o2-mch-tracks-to-tracks-workflow` described below.

## Track fitter

`o2-mch-tracks-to-tracks-workflow`

Take as input the list of MCH tracks
([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)) and the list of associated
clusters ([ClusterStruct](../Base/include/MCHBase/ClusterBlock.h)) of the current event with the data description
"TRACKSIN" and "TRACKCLUSTERSIN", respectively, and send the list of refitted MCH tracks
([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)) and the list of associated
clusters ([ClusterStruct](../Base/include/MCHBase/ClusterBlock.h)) in two separate messages with the data description
"TRACKS" and "TRACKCLUSTERS", respectively.

Options `--l3Current xxx` and `--dipoleCurrent yyy` allow to specify the current in L3 and in the dipole to be used to
set the magnetic field.
