<!-- doxy
\page refDetectorsMUONMCHEvaluation MCH Evaluation
/doxy -->

# MCH Evaluation

This package offers a few helpers to assess (some) performance of MCH.

The class [ExtendedTrack](include/MCHEvaluation/ExtendedTrack.h) converts a standalone MCH track (found in `mchtracks.root` files) into a track extrapolated to a vertex (that must be provided). It's a convenient object that also packs the clusters associated to the track (contrary to what's found in the `mchtracks.root` where tracks and clusters are in two different containers).

The `o2-mch-compare-tracks-workflow` can be used to produce many histograms (output is a root file and optionally a pdf file) comparing tracks from two different `mchtracks.root` files.

For instance to compare tracks coming from reconstructions using the two different clustering we have :

```shell
$ o2-mch-tracks-reader-workflow --hbfutils-config o2_tfidinfo.root --infile mchtracks.legacy.root | o2-mch-tracks-reader-workflow --hbfutils-config o2_tfidinfo.root --infile mchtracks.gem.root --subspec 1 | o2-mch-compare-tracks-worfklow --pdf-outfile compare.pdf -b
$ root
root[0] o2::mch::eval::drawAll("compare.root")
```
# MCH Cluster Maps

 - Goal : The general purpose is to track "unexpected" detector issues not well reproduced with MC simulations. These problems generate non-negligible bias in Acc*Eff corrections resulting in large tracking systematic uncertainties. During the data reconstruction, the status of the detector is calculated with the CCDB which is used to discard most of the detector issues. This status map is built with information based on pedestals, occupancy etc. (high and low voltage will be included soon in the statusmap.) Nevertheless, some detector issues (e.g. a cable swapping) are not well detected online and consequently not properly reproduced by the CCBD. The main objective of this code is to spot these issues not included in the status map.

 - SRC FILE:
 `clustermap-display.cxx`

- INPUT FILES (see readme_more.md for more info):
`DATA_QC.root`
`MC_QC.root`
`o2sim_geometry-aligned.root`

 - HELP MESSAGE TO KNOW WHICH OPTIONS ARE AVAILABLE:
```shell

o2-mch-clustermap-display --help

```

 - EXECUTION COMMAND:

```shell

o2-mch-clustermap-display --green --normperarea --rootfileleft DATA_QC.root --rootfileright 100mil.root --help

```

  - OUTPUT FILES:

 Non-bending(NB):

 ```shell

CHAMBERS-1-NB.html CHAMBERS-2-NB.html CHAMBERS-3-NB.html CHAMBERS-4-NB.html CHAMBERS-5-NB.html CHAMBERS-6-NB.html CHAMBERS-7-NB.html CHAMBERS-8-NB.html CHAMBERS-9-NB.html CHAMBERS-10-NB.html

```
Bending(B):
 ```shell

CHAMBERS-1-B.html CHAMBERS-2-B.html CHAMBERS-3-B.html  CHAMBERS-4-B.html CHAMBERS-5-B.html CHAMBERS-6-B.html CHAMBERS-7-B.html CHAMBERS-8-B.html CHAMBERS-9-B.html  CHAMBERS-10-B.html

```
