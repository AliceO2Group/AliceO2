<!-- doxy
\page refDetectorsMUONMCHClustering Clustering
/doxy -->

# MCH Original Clustering

This is a reimplementation of the original MLEM-based clustering algorithm from AliRoot.

## Input / Output
It takes as input the list of digits associated to one precluster and fills the internal
structure with the clusters and the list of associated digits. The list of clusters and
associated digits can be retreived with the corresponding getters and cleared with the
reset function. An example of usage is given in the ClusterFinderOriginalSpec.cxx device.

## Short description of the algorithm

The algorithm starts with a simplification of the precluster, sending back some digits to
the preclustering for further use. It then builds an array of pixels, which is basically
the intersect of the pads from both cathodes, that will be used to determine the number
of clusters embedded in the precluster and find there approximate locations to be used as
seeds for the fit of the charge distribution with a sum of Mathieson functions.
- For small preclusters, the fit is performed with one distribution starting at the center
of gravity of the pixels, without further treatment.
- For "reasonably" large preclusters, the MLEM procedure is applied: The charge of the
pixels is recomputed according to the fraction of charge of each pad seen by this pixel,
assuming a cluster is centered at this position, then only the pixels around the maximum
charges are kepts and their size is divided by 2. The procedure is repeated until the size
is small enough then the splitting algorithm is applied: Clusters of pixels above a
minimum charge are formed, then grouped according to their coupling with the pads and
split in sub-groups of a maximum of 3 clusters of pixels. For every sub-group, the pads
with enough coupling with the pixels are selected for the fit and the groups of pixels are
used to determine the seeds.
- For very large preclusters, the pixels and associated pads around each local maximum are
extracted and sent separately to the MLEM algorithm described above.

A more detailed description of the various parts of the algorithm is given in the code itself.

## Example of workflow

The line below allows to read run2 digits from the file digits.in, run the preclustering,
then the clustering and write the clusters with associated digits in the file clusters.out:

`o2-mch-digits-reader-workflow --infile "digits.in" --useRun2DigitUID | o2-mch-digits-to-preclusters-workflow | o2-mch-preclusters-to-clusters-original-workflow | o2-mch-clusters-sink-workflow --outfile "clusters.out" --useRun2DigitUID`

# MCH GEM Clustering (new)

This preliminary development is the new algorithm implementation of the clustering/fitting.
It is called GEM (Gaussian Expected-Maximiztion)

## Input / Output
Exactly the same as the orignal version (see 

