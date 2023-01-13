<!-- doxy
\page refDetectorsMUONMCHTracking Tracking
/doxy -->

# MCH Tracking

* [TrackExtrap.h(cxx)](#trackextraphcxx)
* [TrackFitter.h(cxx)](#trackfitterhcxx)
* [TrackFinderOriginal.h(cxx)](#trackfinderoriginalhcxx)
* [TrackFinder.h(cxx)](#trackfinderhcxx)
* [Workflows](#workflows)
  * [Track fitter](#track-fitter)
  * [Track finder](#track-finder)
* [Examples of workflow](#examples-of-workflow)

## TrackExtrap.h(cxx)

Contains all the tools to:
- extrapolate the track parameters (and covariances) from one z position to another, taking into account or not the
magnetic field.
- add the dispersion from Multiple Coulomb Scattering (MCS) to the covariance matrix when crossing a tracking chamber.
- extrapolate the track parameters (and covariances) through the front absorber, taking into account (or not) the
effects from MCS and energy loss.

## TrackFitter.h(cxx)

Fit a track to the clusters attached to it, using the Kalman Filter.

### Short description of the algorithm

The seeds for the track parameters and covariances are computed using the position and resolution of the 2 clusters in
the last two fired chambers. An estimate of the bending momentum is obtained by assuming that the track is coming from
the vertex (0,0,0) and crosses a constant magnetic field (both with sufficiently large uncertainties to do not bias the
final result).
The initial track parameters and covariances are then propagated to every clusters, up to the most upstream one, and are
updated at each step by "attaching" the cluster with the Kalman filter. The dispersion from MCS is added to the
covariances each time a chamber is crossed.
This algorithm gives the track parameters and covariances at the first cluster (the last attached one). Optionally, the
Smoother algorithm can be run to fit back the track and compute the track parameters and covariances at every clusters.
If the pointer to the parameters at one of the attached clusters is given, the fit is resumed starting from this point.

## TrackFinderOriginal.h(cxx)

This is a reimplementation of the original tracking algorithm from AliRoot.

### Input / Output

It takes as input a list of reconstructed clusters, the clusters' position being given in the global coordinate system.
It returns the list of reconstructed tracks, each of them containing (in the TrackParams) the pointers to the associated
clusters.

### Short description of the algorithm

The original tracking algorithm is described in detail here:
https://edms.cern.ch/ui/file/1054937/1/ALICE-INT-2009-044.pdf

The main difference with the original implementation is in the order the tracking of the initial candidates (made of
clusters in stations 4&5) is performed. In the original implementation, all candidates are propagated to station 3 to
search for compatible clusters, tracks are duplicated to consider all possibilities, then all candidates are propagated
to station 2, and so on. In some events with very high multiplicity, it results in a temporarily huge number of
candidates in memory at the same time (>10000) and a limitation is needed to abort the tracking in such case to do not
break the reconstruction. In the new implementation, each candidate is treated one by one. The current candidate is
propagated to station 3 to search for compatible clusters, duplicated to consider every posibilities, then each of them
is tracked one by one to station 2, then to station 1. This limits drastically the number of candidates in memory at the
same time, thus avoiding to abort the tracking for any event, while reconstructing exactly the same tracks in the end.

## TrackFinder.h(cxx)

This is the implementation of the new tracking algorithm.

### Input / Output

It takes as input a list of reconstructed clusters, the clusters' position being given in the global coordinate system.
It returns the list of reconstructed tracks, each of them containing (in the TrackParams) the pointers to the associated
clusters.

### Short description of the algorithm

#### General ideas to improve the original algorithm:
- Consider the overlaps between DE on the same chamber during the tracking (the original algorithm considers only one
cluster per chamber during the tracking, duplicating the tracks with 2 clusters in overlapping DE to consider both), so
there is no need to complete the tracks to add the missing clusters afterward and to cleanup the duplicates. We assume
there is no overlap between the 2 half-planes of chambers 5 to 10.
- Do not try to recover tracks by removing one of the previously attached clusters in the previous station if no
compatible cluster is found in the current one. Assume that a good track will not be lost if several clusters are
attached in a station and some of them are bad (i.e. not belonging to that track).
- Further reduce the number of track duplication and memory consumption by pursuing the tracking of the current
candidate with the newly found compatible clusters down to station 1, or until the tracking fails, before testing other
clusters, so that we duplicate the original candidate only to store valid complete tracks.
- Speedup the cleanup of connected tracks in the end by reducing the number of imbricated loops.
- Speedup the track extrapolation in the magnetic field by reducing the number of steps to reach to desired z position.

#### Algorithm:
- Build candidates on station 5 starting from pairs of clusters on chambers 9 and 10 and looking for compatible clusters
in overlap DE. Consider every combinations of different clusters and discard candidates whose parameters are outside of
acceptance limits within uncertainties.
- Propagate the candidates to station 4 and look for compatible clusters, taking into account overlaps. At least one
cluster is requested and the track is duplicated to consider every possibilities, eliminating the ones driving the track
parameters outside of acceptance limits within uncertainties.
- Repeat the two steps above starting from station 4 then going to station 5 to look for additional candidates with only
one chamber fired on station 5.
- Propagate each candidate found in the previous steps from chamber 6 to 1, looking for compatible clusters taking into
account the overlaps. This is a recursive procedure. For each cluster or couple of clusters found on one chamber, the
current parameters of the initial candidate are updated and the tracking continue to the next chamber, and so on and so
forth, until chamber 1 or until the tracking fails (at least one cluster per station is requested to continue). Only at
the end, when/if the track reaches the first station, it is duplicated and the clusters found in the process are
attached to the new track. Clusters that drive the track parameters outside of acceptance limits within uncertainties
are discarded.
- Improve the tracks: run the smoother to recompute the local chi2 at each cluster, remove the worst cluster if it does
not pass a stricter chi2 cut, refit the track and repeat the procedure until all clusters pass the cut or one of them
cannot be removed without violating the tracking conditions (by default, the track must contain at least 1 cluster per
station and both chambers fired on station 4 or 5), in which case the track is removed.
- Remove connected tracks in station 3, 4 and 5. If two tracks share at least one cluster in these stations, remove the
one with the smallest number of fired chambers or with the highest chi2/(ndf-1) in case of equality, assuming it is a
fake track.

In all stations, the search for compatible clusters is done in a way to consider every possibilities, i.e. every
combinations of 1 to 4 clusters, while skipping the already tested combinations. This includes subsets of previously
found combinations, so that we always attach the maximum number of clusters to a candidate (e.g. if we can attach 4
clusters in one station because of the overlaps, we do not consider the possibility to attach only 3, 2, or 1 of them
even if the track would still be valid).

A more detailed description of the various parts of the algorithm is given in the code itself.

#### Available options:
- Find more track candidates, with only one chamber fired on station 4 and one on station 5, taking into account the
overlaps between DE and excluding those whose parameters are outside of acceptance limits within uncertainties.
- Do not request certain stations. When building initial candidates on stations 4 and 5, it means to keep the candidates
found on station 5(4) even if no compatible cluster is found on station 4(5) if that station is not requested. However,
if one or more compatible clusters are found we do not consider the possibility to do not attach any cluster on that
station. When tracking the candidates to station 3 to 1, it means to consider the possibility to continue the tracking
without attaching any cluster on the station that is not requested, even if compatible clusters are found on that
station. However the clusters found upstream after having attached cluster(s) on that station are skipped.

## Workflows

### Track fitter

```shell
o2-mch-tracks-to-tracks-workflow
```

Take as input the list of all MCH tracks ([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)) in the current time frame, the list of all associated clusters ([Cluster](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h)) and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the tracks associated to each interaction, with the data description "TRACKSIN", "TRACKCLUSTERSIN" and "TRACKROFSIN", respectively. Send the list of all refitted MCH tracks ([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)) in the time frame, the list of all associated clusters ([Cluster](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h)) and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the tracks associated to each interaction in three separate messages with the data description "TRACKS", "TRACKCLUSTERS" and "TRACKROFS", respectively.

#### Workflow / Spec options:

`--l3Current xxx` and `--dipoleCurrent yyy` allow to specify the current in L3 and in the dipole to be used to set the magnetic field.

### Track finder

```shell
o2-mch-clusters-to-tracks-workflow
```

Take as input the list of all clusters ([Cluster](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h)) in the current time frame, with the data description "CLUSTERS", and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the clusters associated to each interaction, with the data description "CLUSTERROFS". Send the list of all MCH tracks ([TrackMCH](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)) in the time frame, the list of all associated clusters ([Cluster](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h)) and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the tracks associated to each interaction in three separate messages with the data description "TRACKS", "TRACKCLUSTERS" and "TRACKROFS", respectively. Depending on the options, it may also need as input the list of digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)) associated to clusters, with the data description "CLUSTERDIGITS", and send the list of digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)) associated to tracks, with the data description "TRACKDIGITS".

#### Workflow / Spec options:

`--original` allows to use the original track finding algorithm (the new one is used by default).

`--disable-magfield-from-ccdb` allows to disable the loading of the magnetic field from the CCDB. In this case, the magnetic field is loaded from the GRP file if available. Otherwise, it is created using the provided currents in L3 and in the dipole.

`--grp-file` allows to set the name of the GRP file containing the magnetic field.

`--l3Current xxx` and `--dipoleCurrent yyy` allow to specify the current in L3 and in the dipole to be used to set the magnetic field.

`--disable-time-computation` allows to disable the computation of the track time from the associated digits. Instead, the track time is set to cover the ROF duration.

`--digits` allows to send the list of digits associated to the tracks.

`--debug x` allows to enable the debug level x (0 = no debug, 1 or 2).

`--mch-config "file.json"` or `--mch-config "file.ini"` allows to change the tracking parameters from a configuration file. This file can be either in JSON or in INI format, as described below:

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

`--configKeyValues "key1=value1;key2=value2;..."` allows to change the tracking parameters from the command line. The parameters changed from the command line will supersede the ones changed from a configuration file.

* Example of parameters changed from the command line:
```shell
--configKeyValues "MCHTracking.chamberResolutionY=0.1;MCHTracking.requestStation[1]=false;MCHTracking.moreCandidates=true"
```

## Examples of workflow

- The line below allows to read the clusters from the file `clusters.in`, run the new tracking algorithm, read the
associated vertex from the file `vertices.in`, extrapolate the tracks to the vertex and write the result (tracks at
vertex, MCH tracks and associated clusters) in the file `tracks.out`:

```shell
o2-mch-clusters-sampler-workflow --infile "clusters.in" --global | o2-mch-clusters-to-tracks-workflow --disable-magfield-from-ccdb --l3Current 29999.998047 --dipoleCurrent 5999.966797 | o2-mch-vertex-sampler-workflow --infile "vertices.in" | o2-mch-tracks-to-tracks-at-vertex-workflow --l3Current 29999.998047 --dipoleCurrent 5999.966797 | o2-mch-tracks-sink-workflow --outfile "tracks.out"
```

- The line below allows to read the MCH tracks and associated clusters from the file `tracks.in`, refit them and write
them in the file `tracks.out`:

```shell
o2-mch-tracks-sampler-workflow --infile "tracks.in" --forTrackFitter | o2-mch-tracks-to-tracks-workflow --l3Current -30000.025391 --dipoleCurrent -5999.954590 | o2-mch-tracks-sink-workflow --outfile "tracks.out" --mchTracksOnly
```
