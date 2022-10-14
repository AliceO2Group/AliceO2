<!-- doxy
\page refDetectorsVertexing Detectors Vertexing
/doxy -->

# Classes for Vertexing

## DCAFitterN

Templated class to fit the Point of Closest Approach (PCA) of secondary vertex with N prongs. Allows minimization of either absolute or weighted Distances of Closest Approach (DCA) of N tracks to their common PCA.

For every N (prongs) a separate specialization must be instantiated, e.g.
```cpp
using Track = o2::track::TrackParCov;
o2::vertexing::DCAFitterN<2,Track> ft2; // 2-prongs fitter
// or, to set at once some parameters
float bz = 5.;           // field in kGauss
bool useAbsDCA = true;   // use abs. DCA minimizaition instead of default weighted
bool propToDCA = true;   // after fit, create a copy of tracks at the found PCA
o2::vertexing::DCAFitterN<3,Track> ft3(bz, useAbsDCA, propToDCA); // 3-prongs fitter
```
One can also use predefined aliases ``o2::vertexing::DCAFitter2`` and ``o2::vertexing::DCAFitter3``;
The main processing method is
```cpp
o2::vertexing::DCAFitterN<N,Track>::process(const Track& trc1,..., cons Track& trcN);
```

The typical use case is (for e.g. 3-prong fitter):
```cpp
using Vec3D    =  ROOT::Math::SVector<double,3>; // this is a type of the fitted vertex
o2::vertexing::DCAFitter3 ft;
ft.setBz(5.0);
ft.setPropagateToPCA(true); // After finding the vertex, propagate tracks to the DCA. This is default anyway
ft.setMaxR(200);            // do not consider V0 seeds with 2D circles crossing above this R. This is default anyway
ft.setMaxDZIni(4);          // do not consider V0 seeds with tracks Z-distance exceeding this. This is default anyway
ft.setMaxDXYIni(4);         // do not consider V0 seeds with tracks XY-distance exceeding this. This is default anyway
ft.setMinParamChange(1e-3); // stop iterations if max correction is below this value. This is default anyway
ft.setMinRelChi2Change(0.9);// stop iterations if chi2 improves by less that this factor
ft.setMaxChi2(10);          // discard vertices with chi2/Nprongs (or sum{DCAi^2}/Nprongs for abs. distance minimization)

Track tr0,tr1,tr2;                  // decide candidate tracks
int nc = ft.process(tr0,tr1,tr2);   // one can have up to 2 candidates, though the 2nd (if any) will have worse quality
if (nc) {
   Vec3D vtx = ft.getPCACandidate(); // same as ft.getPCACandidate(0);
   LOG(info) << "found vertex " << vtx[0] << ' ' << vtx[1] << ' ' << vtx[2];
   // access the track's X parameters at PCA
   for (int i=0;i<3;i++) {
    LOG(info) << "Track " << i << " at PCA for X = " << ft.getTrackX(i);
   }
   // access directly the tracks propagated to the DCA
   for (int i=0;i<3;i++) {
     const auto& track = ft.getTrack(i);
     track.print();
   }
}
```

By default the propagation is done with bZ provided by the user and w/o material corrections applied.
One can request the propagation with full local field and/or material corrections by setting
```
ft.setUsePropagator(true); // use must take care of initialization of the propagator (loading geometry and magnetic field)
ft.setMatCorrType(o2::base::Propagator::MatCorrType::USEMatCorrLUT); // of USEMatCorrTGeo
```

Note that if material correction is not default USEMatCorrNone, then the propagator will be used even if not requested (hence must be initialized by the user).

To get the most precise results one can request `ft.setRefitWithMatCorr(true)`: in this case when `propagateTracksToVertex()` is called, the tracks will be propagated
to the V0 with requested material corrections, one new V0 minimization will be done and only after that the final propagation to final V0 position will be done.
Since this is CPU consiming, it is reccomended to disable propagation to V0 by default (`ft.setPropagateToPCA(false)`) and call separately `ft.propagateTracksToVertex()`
after preliminary checks on the V0 candidate.

By default the final V0 position is defined as
1) With `useAbsDCA = true`: simple average of tracks position propagated to respective `X_dca` parameters and rotated to the lab. frame.
2) With `useAbsDCA = false`: weighted (by tracks covariances) average of tracks position propagated to respective `X_dca` parameters and rotated to the lab. frame.

Extra method `setWeightedFinalPCA(bool)` is provided for the "mixed" mode: if `setWeightedFinalPCA(true)` is set with `useAbsDCA = true` before the `process` call, the minimization will be done neglecting the track covariances,
but the final V0 position will be calculated using weighted average. One can also recalculate the V0 position by the weighted average method by calling explicitly
`ft.recalculatePCAWithErrors(int icand=0)`, w/o prior call of `setWeightedFinalPCA(true)`: this will update the position returned by the `getPCACandidate(int cand = 0)`.

The covariance matrix of the V0 position is calculated as an inversed sum of tracks inversed covariances at respective `X_dca` points.

See ``O2/Detectors/Base/test/testDCAFitterN.cxx`` for more extended example.
Currently only 2 and 3 prongs permitted, thought this can be changed by modifying ``DCAFitterN::NMax`` constant.

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
