<!-- doxy
\page refDetectorsMUONMatching Matching
/doxy -->

# TrackMatcher.h(cxx)

Implementation of the MCH-MID matching algorithm.

## Input / Output

It takes as input the lists or MCH ROFs, MCH tracks, MID ROFs and MID tracks. It returns the list of matched tracks with the data format [TrackMCHMID](../../../DataFormats/Reconstruction/include/ReconstructionDataFormats/TrackMCHMID.h).

## Short description of the algorithm

The matching of each MCH track in each MCH ROF is done in 2 steps:
1) by time: finding all the MID ROFs (resolution = 1 BC) compatible with the MCH ROF (BC range)
2) by position: finding the MID track in those MID ROFs that is the most compatible (best matching chi2) with the MCH track. The chi2 must be lower than a threshold, defined via TrackMatcherParam (see below), to validate the matching.

# TrackMatcherParam.h(cxx)

Definition of the sigma cut used to set the maximum matching chi2 = 4 * (sigma cut)^2. The factor 4 is because the matching is done on 4 track parameters (x, slopex, y, slopey).

The sigma cut is configurable via the command line or an INI or JSON configuration file (cf. [workflow documentation](../Workflow/README.md)).

## Example of workflow

`o2-muon-tracks-matcher-workflow --disable-mc`

This takes as input the root files with MCH and MID tracks and ROFs and gives as ouput a root file with the matched tracks.

See the [workflow documentation](../Workflow/README.md) for more details about the track matcher workflow.
