# `O2BVHSurfaceSolid` — the exact-surface solid

`o2::cad::O2BVHSurfaceSolid` is a `TGeoBBox`-derived shape in `libO2CADSupport`. It represents a
CAD solid by its exact boundary: a set of analytic surface patches (plane, cylinder, cone, sphere,
torus), each trimmed to its face. A BVH over boxes that cover the patches accelerates every
navigation query. It is the exact alternative to a tessellated mesh for parts whose faces are all
analytic.

The converter `O2_CADtoTGeo.py` produces it with `--exact-surfaces auto|required`. Each exact
volume gets a sidecar `surfaces_<VOLNAME>_<LID>.bin`, which the generated `geom.C` loads with
`o2::cad::LoadSurfaceSolid` (`CADSupport/O2SurfaceSolidIO.h`).

Tolerances are listed with their values and reasons in [`TolerancePolicy.md`](TolerancePolicy.md).
The harness that validates and times the solid is described in
[`SolidNavigationHarness.md`](SolidNavigationHarness.md).

## 1. Representation

### 1.1 Surfaces

The surface classes are private (`src/BoundedSurface.h`, namespace `o2::cad::surface`). All derive
from the abstract `BoundedSurface`.

| class | carrier | parametric domain (u, v) | trim |
| --- | --- | --- | --- |
| `PlanarBoundedSurface` | plane, axes need not be orthonormal | (u, v) along axisU, axisV, cm | line-segment polygon |
| `CurvedPlanarBoundedSurface` | plane, orthonormal axes | (u, v), cm | line / arc / B-spline wires |
| `CylindricalBoundedSurface` | cylinder | (phi [rad], h [cm]) | rectangle or wire |
| `ConicalBoundedSurface` | cone, linear radius law r(h) | (phi [rad], h [cm]) | rectangle or wire |
| `SphericalBoundedSurface` | sphere | (phi [rad], theta [rad]) | rectangle or wire |
| `TorusBoundedSurface` | torus | (phiRing [rad], phiTube [rad]) | rectangle or wire |

`phi` is measured from `referenceAxisU` projected perpendicular to the axis. `theta` is measured
from the +polar-axis pole. `phiTube` is measured around the tube from the outer equator towards
the +axis pole. A cone may have zero radius at one end (apex cone).

A quadric or torus patch is trimmed either by a scalar parametric rectangle (phi sweep times
height, theta or tube range) or by a general wire in its (u, v) domain. With a wire, the wire is
authoritative for containment and the scalar parameters only fix the frame and a conservative
window. A wire trim may not wrap more than one full turn in any periodic angle.

The `innerWall` flag reverses the outward normal of a quadric or torus: it then points towards the
axis, the centre or the tube spine. It marks a hole wall.

### 1.2 Trim curves and wires

A trim curve (`Curve2D`) is one of three kinds:

- a line segment;
- a circular arc: centre, radius, start angle and signed sweep (a full circle is a sweep of ±2π);
- a clamped B-spline, optionally rational: degree, poles, weights, flat knot vector.

B-splines are evaluated by de Boor. Their enclosed area is integrated by Gauss-Legendre per knot
span, which is exact for non-rational curves. Point-in-wire winding and point-to-curve distance use
one cached flattened polyline per curve. The flattener subdivides until each chord is within
`kBSplineFlatness` of the curve, judged at t = 1/4, 1/2 and 3/4 of the interval, and it never
declares an interval flat while it still contains an interior knot.

A wire (`CurveWire` for curves, `SurfaceWire` for polygons) is one closed loop. A face has one
outer wire and any number of inner wires (holes). Wires are normalised to outer counter-clockwise
and inner clockwise; a re-orientation is logged. A wire is rejected when it is non-finite, open,
of zero area or self-touching. Consecutive curve endpoints must meet within the wire-join band,
measured as a 3D length through the surface's first fundamental form.

### 1.3 Public construction API

```cpp
bool AddPlanarSurface(origin, axisU, axisV, outerWire, innerWires = {});
bool AddCurvedPlanarSurface(origin, axisU, axisV, outerWire, innerWires = {});
bool AddCylindricalSurface(centerPoint, axis, referenceAxisU, radius, heightMin, heightMax,
                           phiStart = 0, phiSweep = 2pi, innerWall = false);
bool AddConicalSurface(centerPoint, axis, referenceAxisU, radiusAtMin, radiusAtMax,
                       heightMin, heightMax, phiStart = 0, phiSweep = 2pi, innerWall = false);
bool AddSphericalSurface(center, polarAxis, referenceAxisU, radius, thetaMin = 0, thetaMax = pi,
                         phiStart = 0, phiSweep = 2pi, innerWall = false);
bool AddToroidalSurface(centerPoint, axis, referenceAxisU, majorRadius, minorRadius,
                        phiStart = 0, phiSweep = 2pi, tubeStart = 0, tubeSweep = 2pi,
                        innerWall = false);
```

Each quadric and the torus has a second overload that appends `outerTrim` and `innerTrims` as
vectors of `PlanarBoundaryCurve`, the public mirror of `Curve2D` (`makeLine`, `makeArc`,
`makeBSpline`). `AddCurvedPlanarSurface` requires orthonormal axes; its outward normal is
axisU × axisV.

`SetSurfaceBoundaryEdges(surfaceIndex, edgeIds, edgeFlags)` attaches the source-edge identity of a
face (section 4.2). `SetModelTolerance(cm)` records the source model's declared tolerance; zero
means "not stated".

### 1.4 `CloseShape(check = true)`

`CloseShape` computes the bounding box, the display mesh, the safety anchors, the BVH and the
closure diagnostics, in that order. With `check` set, closure defects are reported as `Error`
messages that state the consequence for navigation. A solid with no surfaces stays undefined and
reports `NavigationReliability::Undetermined`.

### 1.5 The BVH

The BVH is a `bvh::v2` float BVH over **cover boxes**, with one cover box per leaf. Each surface
supplies its cover boxes through `appendCoverBoxes`. Their union must contain both the trimmed
patch (every ray hit and on-surface point) and every point at which `distanceSqToPatch` can be
realised, so that one BVH serves the ray and the nearest-patch traversals.

- Planes use one box.
- Cylinders and cones split their sweep into chunks of at most `kCoverChunkAngle` (π/4), each
  bounded exactly.
- Spheres and tori cover the full surface of revolution, because their distance kernels project
  onto the whole surface and ignore the trim.

Every box is widened by `kBVHBoxTolerance` and rounded outward to float. Because a surface can own
several leaves, each traversal hands each surface on only once, using an epoch-stamped
`thread_local` marker.

## 2. Queries

All queries fall back to their loop version before `CloseShape` has built the BVH.

### 2.1 `Contains`

1. A point outside the bounding box (plus `kTolerance`) is outside.
2. A point within `kTolerance` of any patch is inside. Candidate patches come from a BVH
   point-in-box traversal.
3. Otherwise the answer is the parity of the crossings along a fixed skew direction
   (1, √2, √3), normalised. Hits within `kIntersectionTolerance` of each other form a cluster.
   A cluster whose hits all enter, or all exit, is one crossing. A cluster that mixes entering and
   exiting hits is a graze and counts as none.

On a `Reliable` solid (section 4) one parity shot is the answer, unless a counted hit carried
`onTrimBoundary`. That flag means the hit lay inside its patch's on-boundary band, where the trim
test resolves the tie as "inside the trim". The solid then re-shoots.

On any other solid, and on a re-shoot, `Contains` takes a majority vote over five golden-spiral
directions and stops once three agree. Shots that rest on a trim tie-break are counted apart and
decide only when the other shots are tied.

### 2.2 `DistFromOutside` and `DistFromInside`

Both call one template, `nearestCrossing<wantEntering>`. Entering and exiting are decided by the
sign of normal · direction. The traversal is `bvh->intersect<false, robust = true>` with a leaf
lambda. As candidates are found, the ray's `tmax` shrinks to the best candidate plus a cluster
margin, rounded up by `kBVHBoxTolerance` and one float ulp. This prunes nodes beyond the best hit
without losing the hits that decide whether that candidate is a crossing or a graze.

If the nearest candidate turns out to be a graze, the query is repeated without pruning. Hits are
accepted from `-kRayTolerance` so that a crossing at the origin is not lost. `stepmax` bounds the
traversal and the result. `DistFromOutside` first rejects a point whose gap to the bounding box
exceeds `stepmax + kBVHBoxTolerance` on any axis.

Both follow ROOT's `iact` contract. For `iact` below 3, and with a `safe` pointer, they first
compute `Safety` into `safe`. They then return `TGeoShape::Big()` without tracing the ray for
`iact` 0, and for `iact` 1 when `stepmax` is below the safety. `iact` 3 computes no safety.

### 2.3 `Safety` and `ComputeNormal`

`Safety` is the exact distance to the nearest patch, found by an ordered BVH descent with a
running best. Nodes are pruned on their box distance, scaled down by (1 − 1e-12) so the bound
stays a lower bound. The running best is seeded from 24 display vertices (the safety anchors),
which lie on patches and so give an upper bound. The result is rounded down by one ulp. The `in`
argument is not used.

Per patch, `distanceSqToPatch` is exact for planes and for untrimmed quadrics. For wire-trimmed
patches, and for sphere or torus points whose projection falls outside the trim, it is a
conservative lower bound. `Safety` is therefore always a valid underestimate.

`ComputeNormal` uses the same traversal to find the nearest patch. It returns that
patch's outward normal, flipped to point along `dir`.

### 2.4 `Capacity`

`Capacity` is the absolute value of the sum of each patch's divergence-theorem contribution,
(1/3)∫X·n dA over the trimmed patch. `GetSurfaceCapacityContributions` returns the terms.

- Polygons, curved planes without B-spline trims, and untrimmed quadrics and tori have closed forms.
- Wire-trimmed quadrics and tori integrate by Green's theorem around the trim wire, with
  20-point Gauss-Legendre per piece and pieces no wider than π/4 in u. `capacityIsExact()` is false
  for them, but the result is accurate to rounding on a closed solid.
- A curved plane with a B-spline trim reports `capacityIsExact()` false.

On an open solid, `Capacity` measures the closure defect as well as the volume.

### 2.5 Visualisation and sampling

Each surface emits its own display triangulation, with `kArcSamples` (24) chords per full turn.
`GetBuffer3D`, `SetPoints` and `SetSegsAndPols` use it. Navigation never depends on it; only the
safety seed reads display vertices, and only as an upper bound.

`GetPointsOnSegments`, used by `TGeoManager::CheckOverlaps`, projects each sample back onto its
exact patch to within `kSurfacePointTolerance` (1e-11 cm). It returns `kFALSE` when fewer points
than display vertices are requested, so that ROOT falls back to `SetPoints`.

### 2.6 Loop twins and diagnostic hooks

`Contains_Loop`, `DistFromOutside_Loop`, `DistFromInside_Loop`, `Safety_Loop` and
`ComputeNormal_Loop` visit every surface without the BVH. They share the per-hit logic with the
accelerated queries, so the two must agree bit for bit. They serve both as the oracle and as the
performance baseline.

Diagnostic hooks:

- `ContainsAlongDirection` is parity along one explicit direction, without the re-shoot policy.
- `DescribeContainsCrossings` returns the crossing list of the BVH and of the loop.
- `CountBVHRayCandidates`, `HasBVH` and `GetBVHRootBounds` inspect the BVH.
- `SetRayTMaxPruning`, `ResetRayCandidateCounter` / `GetRayCandidateCount` and
  `ResetSafetyCandidateCounter` / `GetSafetyCandidateCount` price the pruning.
- `SetSafetyBoundUnsoundForTest` is a negative control for the tests only.

The measurement switches are process-wide and must not be flipped while queries run. Scratch
buffers and traversal stacks are `thread_local`, so queries allocate nothing after warm-up. The
B-spline polylines are built with their wire, so a query only reads the shared shape and is safe
to call from several threads.

## 3. Persistence

The solid persists the sequence of `Add*Surface` calls as `BVHSurfaceRecord`s (with their curves as
`BVHSurfaceCurveRecord`s), the source-edge identities, and the model tolerance. The custom
`Streamer` reads the records, replays them through `Add*Surface` and calls `CloseShape`, so the
closure diagnostics of a read-back solid are recomputed. A solid with no records reads back
undefined and not navigable. The class version is 3.

## 4. Closure and navigation reliability

Parity containment is defined only on a closed, consistently oriented 2-manifold. `CloseShape`
decides which case applies and reports it as `NavigationReliability`:

| state | meaning | consequence |
| --- | --- | --- |
| `Undetermined` | `CloseShape` has not run, or the solid is empty | no answer is trusted |
| `Reliable` | closed and consistently oriented | single-shot parity |
| `ReversedFaces` | a shared boundary is traversed the same way by both faces | distance queries may return the wrong side |
| `OpenSurfaceSet` | a trim loop has no neighbouring face | wrong answers in the shadow of each gap |
| `NonManifold` | a trim loop runs along two or more other faces | parity is not well defined |

The states are ordered by severity; the worst one present is reported. `IsNavigable()` is true
only for `Reliable`. A solid that is not navigable still answers every query.

### 4.1 Rim matching (the default)

Each face emits one 3D polyline per trim loop (a rim). Each chord midpoint of a rim is matched
against the chords of every other face. A chord counts as matched when another face's chord lies
within the rim-match tolerance plus the sampling sagitta of both chords. The rim-match tolerance is
the model tolerance, or `kRimMatchTolerance` when none is stated. The non-manifold test uses the
tolerance alone, without the sagitta.

Reversed duplicate edges inside one face (a seam) cancel before chaining. Rims are chained by
matching endpoints.

`GetRimReports()` returns one record per rim, with its face, loop, chord count, length, unmatched
length and state. `GetMaxRimIsolation()` is the largest distance from any chord to the nearest
chord of another face. It measures how isolated the loneliest chord is, not the width of a seam,
and it does not change with the tolerance. `GetRimChordResolution()` is the sampling floor below
which rim distances mean nothing.

The per-chord counters (`GetBoundaryEdgeCount` and its siblings) remain as diagnostics only.

### 4.2 Edge identity (sidecar version 3)

When every face states its source edges, closure is decided by counting instead of by proximity.
Every edge must be used exactly twice, in opposite senses. Degenerate edges (a cone apex or sphere
pole) are excluded from the count.

`GetMaxSharedEdgeDeviation()` then reports, as a measurement only, the largest symmetric Hausdorff
distance between the two faces' realisations of one shared edge. Each realisation is sampled at 33
points. Only anchored edges can be measured. If any face states no edges, the rim verdict of 4.1
applies in full.

## 5. Surface sidecar format (`surfaces_<VOLNAME>_<LID>.bin`)

The format is written by `write_surfaces_bin` in `O2_CADtoTGeo.py` and read by
`o2::cad::LoadSurfaceSolid`. Integers are little-endian `uint32` (plus one `uint8` flag),
geometry values are little-endian `float64`, lengths are in cm and angles in radians. The converter
writes version 3. The reader accepts versions 1 to 3.

```
header:
  char[4]  magic          = "O2SS"
  uint32   version        = 3
  uint32   nSurfaces
  uint32   reserved       = 0
  float64  modelTolerance      # version >= 2; cm; 0 = not stated
  uint32   nModelEdges         # version 3; size of the solid's edge table; 0 = not stated
per surface (nSurfaces times):
  uint32   surfaceType    1=plane 2=cylinder 3=cone 4=sphere 5=torus
  uint32   flags          bit 0: innerWall
  uint32   nParams
  float64  params[nParams]     per-type layout below
  uint32   nWires
  per wire (nWires times):
    uint32   wireRole     0=outer 1=inner
    uint32   nEdges
    per edge (nEdges times):
      uint32   curveType  0=line 1=arc 2=bspline
      uint32   nCurveParams
      float64  curveParams[nCurveParams]
        line:    u0 v0 u1 v1
        arc:     cu cv radius phiStart phiSweep     (signed sweep; full circle = ±2π)
        bspline: degree nPoles poles[2*nPoles] weights[nPoles] knots[nPoles+degree+1]
                 (clamped flat knot vector; weights all 1 = non-rational)
  uint32   nBoundaryEdges     # version 3; 0 = this face states no identity
  per boundary edge (nBoundaryEdges times):
    uint32   edgeId      index into the solid's edge table
    uint8    edgeFlags   bit 0 reversed    the face runs against the edge's direction
                         bit 1 degenerate  cone apex / sphere pole: a point, no partner
                         bit 2 anchored    entry i is trim curve i of this face
```

The version differences:

- A version-1 file is a version-2 file without `modelTolerance`. The reader substitutes
  1e-6 cm and warns.
- A version-2 file is a version-3 file without `nModelEdges` and without the per-face edge block.

Per-type `params`, in the order of the `Add*Surface` arguments:

| type | n | layout |
| --- | --- | --- |
| plane | 9 | origin xyz, axisU xyz, axisV xyz |
| cylinder | 14 | centerPoint xyz, axis xyz, referenceAxisU xyz, radius, heightMin, heightMax, phiStart, phiSweep |
| cone | 15 | centerPoint xyz, axis xyz, referenceAxisU xyz, radiusAtMin, radiusAtMax, heightMin, heightMax, phiStart, phiSweep |
| sphere | 14 | center xyz, polarAxis xyz, referenceAxisU xyz, radius, thetaMin, thetaMax, phiStart, phiSweep |
| torus | 15 | centerPoint xyz, axis xyz, referenceAxisU xyz, majorRadius, minorRadius, phiStart, phiSweep, tubeStart, tubeSweep |

The reader rejects a record whose `nParams` differs from this table.

Rules for the wire block:

- **Planes** always carry a wire block, with exactly one outer wire. A loop made only of lines loads
  through `AddPlanarSurface`. Any arc or B-spline edge routes the face through
  `AddCurvedPlanarSurface`.
- **Quadrics and tori** carry no wire block when the trim is the scalar rectangle in `params`.
  Otherwise the block holds one outer wire and optional holes in the patch's (u, v) domain
  (section 1.1). The converter writes such a block only when the trim does not fill the (u, v)
  rectangle.
- **Curved pcurves on quadrics** (circles, ellipses, Béziers, B-splines) are written as B-splines
  whose poles have been pushed through the affine (u, v) → (phi, h or theta) map. This is exact,
  because a B-spline is closed under affine maps.
- **Wire closure.** Consecutive edge endpoints must meet within the join band as a 3D length. The
  band is the declared model tolerance, or 1e-6 cm when that is smaller or not stated. The reader
  and the kernel apply the same rule.

Rules for the edge block:

- The boundary-edge list is written wire by wire in the file's wire order, which is
  `BRepTools_WireExplorer` order — the same order as the trim curves. The reader permutes it into
  the kernel's order (outer wire first).
- A face without a wire block still lists its edges, unanchored.
- Closure is decided by identity only if every surface of the solid states its edges.
- The reader refuses an `edgeId` outside the edge table when `nModelEdges` is stated.
- Each boundary-edge entry is packed without padding, 5 bytes (`uint32` + `uint8`): the writer emits
  it with `struct.pack("<IB", edgeId, edgeFlags)`, and the reader's size check assumes 5 bytes per
  entry before reading any of them.

The reader also refuses a count that claims more wires, edges or edge identities than the file
holds. A refused file may leave the solid partly filled, so the caller discards it.

`nParams` and `nCurveParams` make each record self-describing. An incompatible change bumps
`version`.

## 6. Converter side

The exact path runs in `O2_CADtoTGeo.py` under `--exact-surfaces auto|required`. A leaf solid is
exact only if every one of its faces extracts. In `auto` mode it otherwise falls back to the mesh;
in `required` mode the run stops with a per-face report. `--surface-report <path>` writes the
per-face classification without changing the output. `--dump-brep` writes the OCCT BREP of each
exact leaf, in cm, for the OCCT oracle.

- **Stored analytic faces.** Plane, cylinder, cone, sphere and torus faces extract directly. Quadric
  and torus pcurves are converted to lines or to affine-mapped B-splines. `inner_wall` follows
  `TopAbs_REVERSED`.
- **Recognised faces.** A B-spline, Bézier, extrusion or revolution face is tested against plane,
  sphere, cylinder and cone models. It is accepted only at a relative gap below 1e-9
  (`--recognize-surfaces exact`, the default). Its trim is rebuilt by sampling the 3D boundary
  edges in the recognised frame. Every edge must then be iso-parametric: a rim or a generator.
- **Trim curves.** A B-spline trim edge that is exactly a line (collinear poles) or a circle
  (relative residual below 1e-9) is stored as that curve.
- **Planar faces** accept line, circle, ellipse, B-spline and Bézier edges. Ellipses and Béziers are
  converted to B-splines.
- **Model tolerance and edge table.** The model tolerance is the largest BRep tolerance of the
  shape, in cm. Edge ids come from one edge table per solid.

## 7. Known limits

- **Free-form surfaces** (genuine B-spline or NURBS carriers) are not supported. Such parts ship as
  CSG, if the CSG path accepts them, or as a mesh.
- **Recognised quadrics with non-iso trims** are refused: a face whose boundary is a slanted or
  curved cut in the recognised frame falls back. The surface recogniser has no torus model.
- **The trim tie-break is one-sided.** A hit inside a patch's on-boundary band counts as inside the
  trim, so a B-spline-trimmed patch can overhang its true seam by up to about `kBSplineFlatness`.
  `Contains` detects this and re-shoots. `DistFromOutside`, `DistFromInside` and `ComputeNormal`
  use such hits without a check.
- **Rim sampling is fixed at `kArcSamples` per turn.** Rim distances below the chord sagitta,
  r(1 − cos(π/24)), cannot be resolved. The per-chord sagitta band also underestimates the
  disagreement between two independently flattened polylines of one curve.
- **Two faces of one shared edge carry independent trims.** Edge identity makes the closure
  verdict structural, but the geometry of the two trims is still per face.
- **`Safety` can be loose** for wire-trimmed patches and for trimmed spheres and tori.
- **Per-candidate cost** is dominated by the trim test (winding and closest point on the curve
  polylines), not by the root solve. Each candidate patch also costs a virtual call.
