<!-- doxy
\page refDetectorsVertexing Detectors Vertexing
/doxy -->

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
