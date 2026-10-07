# Ideas

Proposals for `Detectors/CADSupport` that are not yet decided. Work that has been decided on and
deferred is in `reference/Roadmap.md`; open defects are in `known-issues.md`.

## Performance

- Give the hot entry points hidden visibility and inline them, to undo the indirect calls the
  library boundary adds. That is the standard remedy for the 4–5 % in `known-issues.md`.
- Time a flat-CSG part through `o2-bench-cadsupport-solid-harness`, so the pruning gain on
  `DistFromInside` has a number of its own.
- Report `O2FlatCSG::GetUnprunedRetryCount()` from a benchmark run, so it is visible how often the
  flat-CSG safety net falls back to an unpruned traversal.

## Reach

- Teach `tgeo2vecgeom` and VGM about the CAD solids. A converted geometry navigates under TGeo only,
  so it cannot use the VecGeom or the native Geant4 navigator.
- Ship the browser viewer for the per-part reports, which lives outside this module today.
- Support free-form surfaces that no exact representation covers, instead of falling back to a mesh.

## Testing

- Split `test/testBVHSurfaceSolid.cxx` along its own section banners; it is larger than the code it
  tests.
- Add a unit test for the axis fallback in `O2OverlapCheck`'s `containmentFlips`, which only the
  overlap census exercises today.
- Give `O2FlatCSG`'s flip-containment test an assertion independent of the sampler's own rule, for
  example that each sampled point lies within tolerance of a halfspace.
- Use the edge-graze fixture for the direction-sensitive `Contains` overload, which a convex box
  cannot exercise.
- Move the `RepBench*` cases out of `test/testBVHSurfaceSolid.cxx` into their own test target; they
  exercise `RepresentationBench.h`, not the solid.
