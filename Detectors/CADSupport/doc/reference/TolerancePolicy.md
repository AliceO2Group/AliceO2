# Tolerance policy

This document is the register of the numerical tolerances used by the CAD support code. For each
constant it gives the value and the reason. It also states the rules the constants follow, and the
known limits of the scheme.

## 1. Rules

1. **Compare like with like.** A tolerance is compared only against a quantity of the same
   dimension. Parametric separations on a quadric mix radians and centimetres, so they are first
   converted to a 3D length through the surface's first fundamental form (section 2).
2. **Normalise algebraic guards.** Where a guard asks whether a discriminant, resolvent or
   derivative is zero, the problem is normalised to be dimensionless first, and the threshold is a
   multiple of the machine epsilon. Where an exact structural condition can decide instead, it is
   used and no constant exists.
3. **Prefer the model's own tolerance.** When the source model declares a tolerance (sidecar
   version 2 or later), it replaces the fallback constants, but never goes below the extractor
   floor.
4. **Lower bounds stay lower bounds.** Every guard on a safety or pruning bound errs towards a
   smaller distance. A too-small safety costs a step; a too-large one lets the navigator cross a
   wall.

## 2. The parametric metric

`BoundedSurface::parametricMetric(uv, gUU, gUV, gVV)` gives the first fundamental form at `uv`. A
displacement (du, dv) then spans the length sqrt(gUU·du² + 2·gUV·du·dv + gVV·dv²).

| surface | gUU | gUV | gVV |
| --- | --- | --- | --- |
| plane, curved plane | axisU·axisU | axisU·axisV | axisV·axisV |
| cylinder | r² | 0 | 1 |
| cone | r(h)² | 0 | 1 + k², with k = dr/dh |
| sphere | (R sin θ)² | 0 | R² |
| torus | (R + r cos φ_tube)² | 0 | r² |

The form varies over the domain, so it is evaluated at the point of interest. gUU vanishes at a
sphere pole and at a cone apex; code that divides by it must handle zero. The wire-join checks in
the kernel and in the sidecar reader both go through this metric.

## 3. Kernel constants (`src/BoundedSurface.h`)

| constant | value | reason |
| --- | --- | --- |
| `kTolerance` | 1e-9 cm | Generic length tolerance. It sets the on-surface test and the floor of the on-boundary band for exact curves. |
| `kAreaTolerance` | 1e-18 | Parametric area below which a wire is degenerate. |
| `kRayTolerance` | 1e-9 cm | Minimum positive ray parameter for parity hits. |
| `kIntersectionTolerance` | 1e-7, relative | Two hits are one cluster if \|t1 − t2\| ≤ 1e-7·max(1, \|t1\|, \|t2\|). It is absolute below 1 cm. |
| `kClosureQuantum` | 1e-7 cm | Vertex lattice for the per-chord half-edge counters. These counters are diagnostic only and decide no verdict. |
| `kWireJoinTolerance` | 1e-6 cm | Wire-join band, as a 3D length: the extractor's endpoint precision. `wireJoinToleranceFor(t)` returns max(t, 1e-6) for a declared model tolerance t. |
| `kBSplineFlatness` | 1e-5, parametric | Chord flatness of the B-spline polyline. It is also the on-boundary band floor for B-spline trims, because the polyline is the boundary as far as winding is concerned. |
| `kRimMatchTolerance` | 1e-6 cm | Rim-matching tolerance when the model states none. Same origin as `kWireJoinTolerance`. |
| `kBVHBoxTolerance` | 1e-3 cm | Widening of every BVH cover box before outward float rounding. It must dominate every navigation length tolerance, so that a hit or on-surface point is never pruned. |
| `kQuarticEpsilon` | 32·DBL_EPSILON | Zero test for the normalised quartic solver (section 7). It is a running-error allowance for sums of three or four products of coefficients bounded by 1, not a fitted value. |
| `kArcSamples` | 24 per turn | Chord count for display meshes and rims. It must be divisible by 4 so that quarter-turn-rotated frames sample one shared circle at the same points. |
| `angularTolerance(r)` | kTolerance / max(r, kTolerance) | Angle corresponding to a `kTolerance` arc length at radius r. |
| `kCoverChunkAngle` | π/4 | Widest angular span of one cover box. A chunk's box stays within 1 − cos(π/8) (about 8%) of its arc, and a full sweep costs eight boxes. |
| `kContourQuadratureOrder` | 20 | Gauss-Legendre order of the Green's-theorem capacity integral for wire-trimmed quadrics. |
| `kContourMaxSpanU` | π/4 | Widest u-span of one contour quadrature piece. |
| `kSharedEdgeSamples` | 33 | Samples per edge in the shared-edge deviation measurement. |
| `kMaxSmoothTurn` | 0.52 rad (about 30°) | Rim vertices turning by more than this are corners and are left out of the sampling-noise estimate. A rim sampled at 24 per turn turns by 15° per vertex. |

## 4. Solid constants (`O2BVHSurfaceSolid`)

| constant | value | reason |
| --- | --- | --- |
| `kSurfacePointTolerance` | 1e-11 cm | Distance to its patch within which `GetPointsOnSegments` accepts a projected point. |
| `kDistanceRayTolerance` | −kRayTolerance | Distance queries accept hits from just behind the origin, so that a crossing at the origin is not lost. |
| box-distance guard | ×(1 − 1e-12) | Scales the squared point-to-box distance down, three orders above its rounding error, so it stays a lower bound. |
| anchor seed | d·(1 + 1e-12) + 1e-10 cm | Inflates the upper bound taken from the safety anchors. This stays far below `kBVHBoxTolerance`, so the winning patch is still visited. |
| safety anchors | 24 | Display vertices used to seed the nearest-patch traversal. |
| re-shoot directions | 5, majority 3 | Golden-spiral directions for the containment vote. Three directions were too few; thirteen gained little over five. |
| float ray bound | + FLT_EPSILON·\|t\| | `truncateRoundUp`: a float `tmax` is never shorter than the double bound it stands for. |

## 5. O2Tessellated pruning constants (`Detectors/Base/src/O2Tessellated.cxx`)

`O2Tessellated` stays in `Detectors/Base` (it is also used by `Steer/O2MCApplication`), but its BVH
ray queries follow the same pruning idea as the solid and flat-CSG traversals: lowering the ray's
own `tmax` on a hit prunes the rest of the traversal, and the constants below state how far that may
go without dropping a nearer facet.

| constant | value | reason |
| --- | --- | --- |
| `kFacetBoxPad` | 0.001 cm | Outward pad of every BVH leaf box, so the facet it stands for lies strictly inside it. |
| `kMaxPruneScale` | `kFacetBoxPad · 2²⁴ / 8` | Largest sum of \|origin\| and \|box\| below which lowering the ray bound on a hit cannot drop a nearer facet: float rounding of ray, box and traversal then stays well inside `kFacetBoxPad`. `pruneLimit()` subtracts both from this to get the per-query cutoff. |

## 6. IO, assembly, overlap-check, harness and flat-CSG constants

| constant | where | value | reason |
| --- | --- | --- | --- |
| `kSidecarV1FallbackTolerance` | `O2SurfaceSolidIO.cxx` | 1e-6 cm | Model tolerance assumed for a version-1 sidecar. It is the extractor precision; the reader warns when it uses it. |
| `kBoxTolerance` | `O2BVHAssembly.cxx` | 1e-3 cm | Daughter box widening. Same value and reason as `kBVHBoxTolerance`. |
| `kSafetyBoundShare` | `O2BVHAssembly.cxx` | 1/3 | Share of a node's squared box distance that bounds a daughter's `Safety`. `TGeoBBox::Safety` returns the largest per-axis gap, which is at least the Euclidean distance over √3. |
| box-distance guard | `O2BVHAssembly.cxx` | ×(1 − 1e-12) | Same purpose as the solid's copy (section 4): scales the squared point-to-box distance down, three orders above its rounding error, so it stays a lower bound for `Safety`. |
| `kMaxRootsPerHalfspace` | `O2FlatCSG.cxx` | 4 | A quartic has at most four real roots. |
| `kMaxCubifySplits` | `O2FlatCSG.cxx` | 10 | Per-path ceiling on splits that only equalise aspect ratio. |
| `fSplitDepth` | `O2FlatCSG.h` | 4 | Sub-cell subdivision depth cap, chosen for query cost on the shipped parts. |
| `fMinBoxFraction` | `O2FlatCSG.h` | 0.05 | Minimum box size as a fraction of the part diagonal. |
| `kPadFactor` | `O2FlatCSG.cxx` | 64·DBL_EPSILON | Pads a halfspace range bound by the magnitude accumulated when evaluating it, so the bound survives cancellation. |
| debug bbox probe | `O2FlatCSG.cxx` | 1e-6 · diagonal | Outward offset of the 5×5 face samples that check a cell box contains its cell (debug builds). |
| `kFlipProbe` | `O2FlatCSG.cxx` | 1e-6 cm | Offset either side of a sampled boundary point. `GetPointsOnSegments` keeps the point only if `Contains` differs across it. |
| linear-solve cutoff | `O2FlatCSG.cxx` | \|α\| ≤ 1e-14·(\|β\|+\|γ\|) | A ray whose quadric coefficient α is this small relative to β and γ is solved as hitting a plane instead of a quadratic; the discarded root would lie beyond about 1e6 cm, outside any ALICE geometry. |
| `depthTolerance` | `O2OverlapCheck.h` | 1e-6 cm | A containment shallower than this is a shared boundary, not an overlap. |
| `residualTolerance` | `O2OverlapCheck.h` | 1e-6 cm | A sampled boundary point farther than this from its own solid's boundary is not evidence about anything and is discarded. |
| default boundary band | `O2SolidHarness.cxx` | 1e-3 · bounding-box diagonal | Fallback used when the harness config leaves `boundaryBand` unset, sizing the near-boundary sample band from the part's own extent. |

## 7. The quartic solver

`solveQuarticReal` (ray and torus) first substitutes x = s·y. Here s is the Cauchy root bound
max(|b|, |c|^½, |d|^⅓, |e|^¼) of the monic quartic, rounded up to a power of two. Every
coefficient then lies in [−1, 1], and the branch guards compare against `kQuarticEpsilon`.

Scaling by a power of two is exact in binary floating point, so the normalisation changes no
rounding and no answer; only the guards change. An unrounded Cauchy bound does not have this
property.

Two guards use structural conditions instead of a constant:

- The Newton polishing step is taken if it is finite and no longer than 2, the root bound in
  normalised units.
- `solveDepressedCubic` branches on P ≥ 0 (Cardano) versus P < 0 (trigonometric). No threshold is
  needed, and P = Q = 0 returns 0 through Cardano.

## 8. Bands built from the constants

- **On-boundary band of a trim.** `CurveWire::boundaryBand` is the larger of `kTolerance`
  (converted to parametric units through the metric's largest scale) and the wire's
  `representationTolerance()`. That is `kBSplineFlatness` if any curve is a B-spline, else 0.
  Winding and distance use the same polyline. A point inside the band is classified `Boundary` and
  resolved as inside the trim. The hit is flagged `onTrimBoundary`, and `Contains` re-shoots when
  a counted crossing carries the flag.
- **Wire join.** The 3D gap between consecutive endpoints must not exceed
  `wireJoinToleranceFor(modelTolerance)`. The reader and the kernel apply the same rule.
- **Rim matching.** A chord is matched when another face's chord lies within
  `rimEpsilon + own sagitta + partner sagitta`. `rimEpsilon` is the model tolerance, or
  `kRimMatchTolerance`. The non-manifold test uses `rimEpsilon` alone, because at a corner a third
  face legitimately comes within a chord length.
- **Rim sampling floor.** The sagitta of a rim chord is estimated from the turn angle,
  (chord/2)·tan(turn/4), not from the vertex offset. A box corner would otherwise read as
  sampling noise.

## 9. Converter tolerances (Python)

| constant | where | value | reason |
| --- | --- | --- | --- |
| `_RECOGNIZE_TOL_EXACT` | `O2_CADtoTGeo.py` | 1e-9, relative to the sample box diagonal | A recognised plane, sphere, cylinder or cone must lie on the stored surface at machine precision. |
| `_CANONICAL_CURVE_TOL` | `O2_CADtoTGeo.py` | 1e-9, relative to the curve extent | A B-spline trim edge becomes a line or circle only at machine precision. |
| `_EXTRACT_TOL` | `O2_CADtoTGeo.py` | 1e-7 | Degeneracy floor for extracted sweeps, heights, radii and areas; a record below it is not emitted. |
| `REL_TOL`, `ANG_TOL` | `cadsupport/recognise.py` | 1e-6, relative to the part diagonal | CSG template matching. CAD faces meant to coincide agree to about 1e-7 relative. |
| `REL_TOL` | `cadsupport/tier0.py` | 1e-6 | Same value as the recogniser, so that carriers and faces share one notion of "the same". |
| `TEMPLATE_REL_TOL`, `TEMPLATE_ANG_TOL` | `validation/csgCensus.py` | 1e-6 | Same, for the census. |
| `VOLUME_REL_TOL` | `cadsupport/decompose.py` | 1e-6 | The split pieces must sum to the part's volume. A breach declines the part. |
| `TANGENTIAL_SIN` | `cadsupport/census.py` | 1e-6 | Below this \|n1 × n2\| two face normals across an edge are parallel enough that the dihedral has no reliable sign; `edge_dihedral` classifies the edge `tangential` instead of convex or concave. |
| `NEAR_TANGENTIAL_SIN` | `cadsupport/census.py` | 1e-3 | Below this a `concave`/`mixed` verdict is a blend seam, not a reliable split witness: `decompose.py`'s `first_trusted_concave_edge` refuses one below it rather than cut there. |
| `_BAND_FACTOR` | `cadsupport/accept.py` | 1.0 | CSG acceptance: dV_sym ≤ factor · modelTolerance · area(original). |
| `_CELL_MARGIN` | `cadsupport/recognise.py` | 0.25 of the part diagonal | Padding of a halfspace bounded into a native primitive for tree emission. |
| `_FLAT_BOX_MARGIN` | `cadsupport/recognise.py` | 1e-3 of the part diagonal | Widening of a flat-CSG cell box, three orders above the 1e-6 agreement of the piece. |
| `_FLAT_BOX_PROBE_GRID` | `cadsupport/recognise.py` | 3 | Per-face grid of the outward probe that checks a flat cell box holds its cell. |
| `_FLAT_BOX_PROBE_OFFSETS` | `cadsupport/recognise.py` | 1e-6, 0.25, 1 and 4 box diagonals | Distances outside the box at which that probe samples. |
| `_IDENTITY_EPS` | `cadsupport/primitives.py` | 1e-12 | Frames closer than this are the same; the identity fast path needs an exact rotation. |
| `_CONE_DEGENERATE_EPS` | `cadsupport/primitives.py` | 1e-12, relative | Below this relative difference a cone's two radii are the same radius, and OCCT wants a cylinder rather than a cone. |

## 10. Exporter tolerances (`O2_TGeoToCAD.py`)

| constant | value | reason |
| --- | --- | --- |
| `BOOLEAN_VOLUME_TOL` | 1e-4, relative | Slack on the boolean volume invariant: a composite's exported volume must match the source TGeo volume within this fraction. |
| `_ORTHO_TOL` | 1e-6 | Band within which a hand-written rotation matrix is snapped to the nearest exact rotation and reported; outside it the matrix is refused as rigid and baked instead. |
| `_ISOMETRY_TOL` | 1e-6, relative | Relative volume band a baked isometry must preserve. |
| `EPS` | 1e-12 | Below this a tube's `rmin` is treated as zero: whether the export takes the hollow or the solid path, and whether an inner ring is even built. |

## 11. Validation tolerances (Python)

| constant | where | value | reason |
| --- | --- | --- | --- |
| `CAPACITY_TOLERANCE` | `validation/checkKnownSource.py` | 1e-9, relative | Capacity is compared as a relative deviation, reported as a flag rather than a failure. |
| `PROFILE_TOLERANCE` | `validation/checkKnownSource.py` | 1e-6 | The recogniser's `REL_TOL`, relative to the bounding-box diagonal. |
| `DEFAULT_SKIN_CM` | `validation/checkKnownSource.py` | 1e-9 cm | Not itself compared against anything: it is the default of `--skin`, the band within which a sampled point is too close to either shape's boundary to be scored and is counted instead. |
| `_RAY_EPS` | `validation/occtOracle.py`, `validation/xrayOracle.py`, `validation/assemblyOracle.py` | 1e-9 | Ray-intersector and classifier tolerance passed to OCCT for every oracle ray query. |

## 12. Known limits

- **`kBSplineFlatness` is an absolute parametric value.** On a small part the trim sliver it
  permits is larger relative to the part.
- **`sameIntersection` is absolute below 1 cm.**
- **Rims use a fixed 24 chords per turn**, so rim distances below r(1 − cos(π/24)) cannot be
  resolved. Sampling by a target sagitta in cm would remove this limit.
- **The sagitta band bounds each polyline against its own curve**, not against the other face's.
  It underestimates the polyline-to-polyline disagreement, and tightening `kBSplineFlatness`
  shrinks the band faster than it shrinks the disagreement. Deriving both faces' trims from one
  shared edge object is the fix.
- **`boundaryBand` resolves `Boundary` as inside**, so the overhang is one-sided. Only `Contains`
  checks for it.
