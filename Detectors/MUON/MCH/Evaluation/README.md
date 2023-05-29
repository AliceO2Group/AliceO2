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
