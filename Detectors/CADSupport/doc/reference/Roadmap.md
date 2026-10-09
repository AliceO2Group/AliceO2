# Roadmap — deferred work

This is the list of work that has been decided on but deferred. Each item states what is missing
and why it waits. Nothing here is scheduled.

Open defects and limitations are in `../known-issues.md`; proposals that are not yet decided are in
`../ideas.md`.

## Performance

- **An approximate `Safety`.** `O2BVHSurfaceSolid::Safety` is always exact. Stopping the BVH descent
  early, with a guaranteed underestimate, would make it cheaper. It waits because a looser safety
  costs extra transport steps, which only a transport-level measurement can price.
- **Safety caching.** A per-thread cache of recent safety answers would combine with the above.
  It waits on that measurement too.
- **Per-candidate trim cost.** Once the BVH has pruned, `Contains` spends most of its time in the
  trim test, winding and `Curve2D::closestPoint` on B-spline polylines. The options are an exact
  Bézier-clipping point-in-trim, or cover boxes subdivided in the trim domain so each candidate
  carries a shorter wire. It waits because the cover boxes already cut the candidate count; the
  per-candidate cost is next.
- **Optional Embree BVH engine.** It waits on Embree entering the software stack. The BVH should
  sit behind an interface thin enough to swap the engine, and Embree must stay optional because
  the SIMD situation differs on aarch64.
- **Ahead-of-time specialised kernels.** A converted geometry is static, so the converter could
  emit per-part code with constants folded and no virtual dispatch. It waits until exact
  shared-edge trims give the inner loops a small closed form. Templates over patch archetypes are
  the cheaper first step.
- **A device (GPU) port of `O2FlatCSG`.** The data are already PODs in four arrays and the BVH
  traversal uses an explicit stack. The remaining work is fixed-size scratch arrays instead of
  `thread_local` vectors, a device BVH traversal, and a float or mixed-precision
  `EvalHalfspace`. It waits on a decision about which kernel mix a device port should optimise.
- **A device port of `O2BVHSurfaceSolid`.** It needs a rewrite of the representation: the
  virtual `BoundedSurface` hierarchy flattened to a tagged POD, pooled trim curves, and a
  fixed-capacity hit buffer. It waits behind the `O2FlatCSG` port.

## Coverage

- **Free-form surfaces.** Genuine B-spline carriers are the largest remaining coverage gap. The
  work is an iterative ray/surface intersector: Bézier clipping, or a BVH of Bézier sub-patches
  with Newton refinement. Its benefit over a fine mesh is small, so it must be weighed against the
  tessellated fallback before it is started.
- **Non-iso trims on recognised quadrics.** A NURBS face recognised as a quadric is refused when a
  boundary edge is a slanted or curved cut in the recognised frame. Fixing this needs a numeric
  re-fit of that edge in (phi, h).
- **Torus recognition** in the surface recogniser. The CSG path's Tier 0 already recognises tori.
- **Exact shared-edge trims.** Both faces of a shared edge should derive their trims from one object
  per `TopoDS_Edge`. That removes the one-sided trim sliver and makes the rim band exact.
  Sidecar v3 already carries the edge identity for the closure verdict.
- **Exact arrangement cells for trims.** This is a research-grade route to parts that neither
  tier represents today.
- **The default decomposition cell budget.** `--max-cells` raises it per run. Before moving the
  default, the decomposition time at higher budgets has to be measured over the deep-boolean parts.
- **Boundary-gap declines.** Splitting at every carrier crossing instead of at trusted concave edges
  would convert them. It is a change to `decompose.py` with its own risk.
- **Direct `TGeoShape` → `O2FlatCSG` emitter.** It would give one flat device representation for a
  whole geometry. Primitives map by template. A `TGeoCompositeShape` maps by pushing complements
  down into DNF, which blows up unless redundant bounding halfspaces of subtracted tools are dropped
  and empty cells are pruned with `HalfspaceRange`. The conjecture that drilled holes collapse to a
  single halfspace is unverified.
- **Pcon, Pgon and Xtru round-trip bench.** Pick specimens from the Run 3 geometry, export them
  with `O2_TGeoToCAD.py`, convert them back, and score the result against the source `TGeoShape`.

## Meshing

- **Separate linear and angular precision.** `--mesh-prec` sets both the linear and the angular
  deflection, and in practice the angular one dominates. It waits on a per-volume precision
  scheme, which the next item needs anyway.
- **Precision by physics relevance.** Linear deflection would be set per volume from the distance
  to the interaction point or from |η|. Two cautions apply. Mesh validity is not monotone in
  precision, so each volume has to be validated on its own. For far-field volumes, the acceptance
  criterion should be the capacity error rather than the chordal deviation.
- **Mesh healing.** A mesh can be invalid, not just inaccurate, and chordal accuracy does not
  detect that.

## Navigation and assemblies

- **Assembly-level transport under `TGeoNavigator`.** `assemblyOracle.py` exists; the navigator
  side and a leak counter do not. It waits for the Geant integration test, which exercises the
  whole geometry.
- **A face-adjacency lookup for CAD-native geometry.** Adjacent CAD parts share faces explicitly,
  which could replace the per-step search among siblings with a lookup. It is unmeasured.
- **Carving with an assembly daughter.** `--carve-mothers` cannot subtract an assembly daughter.
  The fix is to fuse its placed leaves into the cutter. Some mothers also fail to carve when their
  daughters consume them completely.
- **A `TGeoOCCTSolid`.** OCCT itself as a shape, either as the fallback of last resort or as an
  in-process oracle. It waits on checks of OCCT's thread safety and of the memory cost of resident
  B-reps.
- **Overlap repair at the STEP level.** It is mechanically possible with `BRepAlgoAPI_Cut`, but
  which part yields is a modelling decision per assembly. It has low priority because
  `TGeoNavigator` tolerates overlaps.
- **Parity on non-manifold input.** Such parts are reported `NonManifold` and answered by vote.
  The open decision is whether to reject them at `CloseShape`.

## Tools

- **Live event display.** Run o2-sim in service mode with warm workers and tap MCStepLogger or the
  O2HitMerger channel. The batch replay comes first.
