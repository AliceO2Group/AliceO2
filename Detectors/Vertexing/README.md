<!-- doxy
\page refDetectorsVertexing Detectors Vertexing
/doxy -->

# Classes for Vertexing

## DCAFitterN

Was moved to Common/DCAFitte.

## Primary Vertexing

The workflow is `o2-primary-vertexing-workflow`, the vertexing parameters are provided via configurable param `pvertexer...` of `PVertexerParams` class. The vertexing first runs an improvized version of `DBSCan` to group tracks losely converging to `MeanVertex` into `time-Z` clusters, then finds for each such a cluster vertices using as a seed the peaks from histogrammed tracks `time-Z` values.

Lot of numerical values in the params must be fine-tuned when the tracking performance will be close to final, particularly the fake ITS-TPC matches (which are prone to create fake vertices).
By default the vertices will be fitted using the `MeanVertex` as an extra measured point.

Two groups of parameters need particular attention:

1) Debris (split vertices) reduction: after finding the vertices it tries to suppress low-multiplicity vertices in a close proximity (in Z and in time) of high-multiplicity ones. See `PVertexerParams.*Debris` parameters comments and `PVertexer::reduceDebris` method.
2) Tracks re-attachment: after finding the vertices and optionally reducing debris, it finds for each track closest vertex (in time and in Z) and refits these groups of tracks using corresponding vertices as seeds. The reason to apply this procedure is the bad time resolution of ITS standalone tracks (=ROF duration): high multiplicity vertex has high chances to steal such tracks belonging to nearby low-multiplicity ones (since it is found first). The reattachment allows to eliminate this effect of particular order of vertex finding.
It should be applied *only* in case the debris reduction was performed, otherwise the low-multiplicity split vertices will steal tracks from high-multiplicity ones. The tracks are tested for belonging to given vertex only if they are in certain `time-range` from the fitted vertex time. This `time-range` is defined as `PVertexerParams.timeMarginReattach` + half of `DBSCan` time difference cut value
`PVertexerParams.dbscanDeltaT` or half `ITS` strobe length (ROF), whichever larger.

In order to tune the parameters, a special debug output file is written when the code is compiled with `_PV_DEBUG_TREE_` uncommented in `PVertexer.h`. It contains the (i) tree of `time-Z` clusters found by `DBSCan` (`pvtxDBScan`), the seeding histograms for every `time-Z` cluster after every vertexing iteration; (ii) the `pvtxComp` tree containing the pairs of vertices which were considered as close by the `reduceDebris` routine, their mutual `chi2` in `Z` and `time`, as well as the decision to reject the vertex with lower multiplicity (2nd one);
(iii) the `pvtx` tree with final vertices and their belonging tracks.

To see the effect of running with and w/o `re-attachment`, one can compare the outputs of 2 tests, e.g.
````
o2-primary-vertexing-workflow --run --configKeyValues "pvertexer.useMeanVertexConstraint=true;pvertexer.applyDebrisReduction=true;pvertexer.applyReattachment=false"
````
and
````
o2-primary-vertexing-workflow --run --configKeyValues "pvertexer.useMeanVertexConstraint=true;pvertexer.applyDebrisReduction=true;pvertexer.applyReattachment=true"
````

## Secondary Vertexing

The workflow is `o2-secondary-vertexing-workflow`. At the moment the TPC tracks are not involved in secondary vertex search.
The available options are:
```
--vertexing-sources arg (=all)        comma-separated list of sources to use in vertexing
```
```
--disable-cascade-finder              do not run cascade finder
```
Plenty of options can be provided via ` --configKeyValues "svertexer.<key>=<value> `, see SVertexerParams class for details.
Note the parameter `maxPVContributors` which tells how many primary vertex contributors can be used in V0 (in case of 0 the PV contributors are not included into the tracks pool). If `minDCAToPV` is positive, then only tracks having their DCA to `MeanVertex` (not the PV!) above this value will be used.
