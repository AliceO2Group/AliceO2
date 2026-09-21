# `O2FlatCSG` — the flat-DNF halfspace solid

`o2::cad::O2FlatCSG` is a `TGeoBBox`-derived shape that stores a solid as a union of cells, where
each cell is an intersection of signed implicit halfspaces. The depth is always two. It carries
parts that the CSG decomposition splits into cells but that are too deep to ship as a
`TGeoCompositeShape`. A BVH over sub-cell boxes keeps the cost proportional to the halfspaces that
are undecided where the query lands.

The section numbers are referenced from `test/testFlatCSG.cxx` and `test/runXRayBenchmark.cxx`;
keep them stable.

## 1. Why it exists

A union of many cells shipped as a `TGeoCompositeShape` is a deep binary tree of `TGeoBoolNode`s.
Each query recurses into both children of every node, so its cost grows with the whole tree. The
tree path also has to bound each halfspace into a padded native primitive (`_cell_leaf`, sized by
`_CELL_MARGIN`), which equals the cell only near the part.

`O2FlatCSG` stores the halfspaces themselves. There is no padding and no margin, and the cost
follows the locally undecided halfspaces.

## 2. Scope

The class covers the C++ shape, its sidecar IO, the flat emitter in the converter, and the `_Loop`
twins and their tests. Geant4 is out of scope (section 11).

## 3. Representation

### 3.1 Halfspaces

A halfspace is `FlatCSGHalfspace { int kind; double sign; double c[11]; }`. The material side is
sign · f(x) ≤ 0, with sign = ±1.

A quadric (`kQuadric`) stores f(x) = xᵀAx + 2bᵀx + c as ten doubles
(a00, a01, a02, a11, a12, a22, b0, b1, b2, c). Plane, sphere, cylinder, cone and elliptic cylinder
are all this one block:

| carrier | A | 2b | c |
| --- | --- | --- | --- |
| plane, unit outward n through p | 0 | n | −n·p |
| sphere, centre p, radius r | I | −2p | \|p\|² − r² |
| cylinder, axis (p, d), radius r | I − ddᵀ | −2Ap | pᵀAp − r² |
| cone, axis (p, d), ref. radius r, k = tan α | I − (1+k²)ddᵀ | −2Ap − 2rk·d | pᵀAp + 2rk(p·d) − r² |
| elliptic cylinder, axes x̂, ŷ, semi-axes a, b | x̂x̂ᵀ/a² + ŷŷᵀ/b² | −2Ap | pᵀAp − 1 |

A torus (`kTorus`) stores its canonical form (px, py, pz, dx, dy, dz, R, r) in the first eight
slots: centre, unit axis, major and minor radius. Inside means
sign · (√((ρ − R)² + z²) − r) ≤ 0, with ρ the distance from the axis and z the coordinate along it.
The canonical form gives both the quartic for rays and the exact signed distance for the range
bound. No reference direction is stored, because a phi limit is a plane of the same cell.

A cell is `FlatCSGCell { int first; int count; double volume; }`, a range of the halfspace array.

**The plane row must use a unit normal, b = n/2.** Any positive rescaling describes the same
halfspace, but only a power-of-two rescaling keeps the accelerated distances bit-identical to
their `_Loop` twins. A cell box face often lies on one of the cell's own axis-aligned planes. The
BVH then reaches that parameter as a slab bound, and the interval clipping reaches it as a
quadratic root. Both give the same double only when the scale is a power of two. The test
`the_accelerated_distances_track_their_twins_when_a_plane_is_rescaled` measures the cost of a ×3
rescale.

### 3.2 Fidelity

The shipped solid is the intersection of the halfspaces the faces carry, everywhere. The tree
path, by contrast, is exact only inside its padded window.

### 3.3 Sign convention

The material side combines the carrier's orientation (a plane's normal is already flipped for
`TopAbs_REVERSED`) with the `side` field from `census.halfspace_side`. An inverted halfspace still
yields a solid, so the error would be silent. The self-test of `cadsupport.emit` therefore checks
every flat cell against `_cell_leaf`'s padded-primitive conjunction on a sample of points.

## 4. The sub-cell BVH

### 4.1 Why boxes, not cells

The AABB of a long diagonal, curved or L-shaped cell is mostly empty. The BVH primitives are
therefore sub-boxes of cells, each with its own list of still-active halfspaces.

### 4.2 The build

For each cell, `CloseShape` starts from the cell box given by `SetCellBBox` and splits it
recursively at the median of the longest axis. At each box, every halfspace still active in the
parent is classified with a rigorous range bound of sign · f over the box:

- **Quadric:** with centre m, half-extents h and g = Am + b,
  |Q(x) − Q(m)| ≤ 2Σ|gᵢ|hᵢ + Σ|Aᵢⱼ|hᵢhⱼ.
- **Torus:** the signed distance is 1-Lipschitz, so f ∈ [f(m) − ‖h‖, f(m) + ‖h‖].

Both bounds are padded by `kPadFactor` times the magnitude accumulated when evaluating f(m).

For each box and halfspace:

- if min(sign · f) > 0, the box is outside the cell and is dropped;
- if max(sign · f) ≤ 0, the halfspace holds everywhere in the box and leaves its active list;
- otherwise the halfspace stays active.

A box whose active list is empty lies wholly inside its cell.

Splitting stops, and the box is kept, when any of these holds:

- the active list is empty;
- the depth budget (`fSplitDepth`, 4) is spent;
- the longest side is no larger than `fMinBoxFraction` (0.05) times the part diagonal;
- the box is still far from cubic once the cubify budget (`kMaxCubifySplits`, 10 per
  root-to-leaf path) is spent.

A split is charged to the cubify budget, not the depth budget, while the box is far from cubic
(longest > 2 · max(shortest, minSize)). Halving the longest extent of a box with ratio ≤ 2 keeps
the ratio ≤ 2. A box that starts near-cubic therefore never draws on the cubify budget, and its
tree is the same as under a depth-only rule. Flooring `shortest` at `minSize` stops a flat cell
from spending the cubify budget on an axis that is never split.

All surviving boxes of all cells go into one `bvh::v2::Bvh`. An over-wide range bound loses
pruning, never correctness.

**The cell box is a correctness obligation on the converter.** No box is ever built outside it,
so material outside the declared box is invisible to the accelerated queries but visible to the
twins. The converter supplies the bounding box of the CAD piece the cell came from, widened by
`_FLAT_BOX_MARGIN`. It refuses the part (`_flat_box_holds_cell`) if an outward probe finds the
cell extending past that box. `emit.crosscheck_contains` then compares `Contains` against
`Contains_Loop` on the shipped shape.

`CloseShape` refuses, and builds nothing, when a cell box is unset, inverted or non-finite. Debug
builds also sample a 5×5 grid on each box face, offset outward by 1e-6 of the diagonal, and require
every sample to be outside the cell.

### 4.3 What the boxes buy

- Tight boxes on long, diagonal and curved cells.
- Short active lists: a query evaluates only the few halfspaces undecided in its box.
- A rigorous `Safety` without any point-to-quadric distance formula (section 5.4).

### 4.4 The correctness invariant

An active list describes the cell only inside its own box. Every ray query clips the ray to a
box's slab interval before it runs the interval clipping over that box's list. Gathering active
lists across boxes and clipping once is wrong. The `_Loop` twins exist mainly to catch a violation
of this rule.

## 5. Queries

All accelerated queries fall back to their twin when `IsClosed()` is false.

### 5.1 `Contains`

`Contains` finds the boxes containing the point. A box with an empty active list answers inside at
once. Otherwise the point is inside if every active halfspace of the box satisfies
sign · f(p) ≤ 0. Cells are disjoint, so the first box that says yes decides.

### 5.2 Distances

Within one box, along the ray clipped to the box, the query collects the roots of every active
halfspace. A quadric gives at most two roots of αt² + 2βt + γ, with α = dᵀAd, β = dᵀ(Ao + b) and
γ = Q(o). When α ≈ 0 the equation is solved as linear. A torus gives at most four roots. The roots
are sorted, and the midpoint of each sub-interval is classified. The result is the cell's occupancy
as a set of intervals. No convexity is assumed, which is required because complemented halfspaces
make cells non-convex.

The traversal visits the boxes the ray meets, rejoins each cell's pieces across boxes, and then
applies the tolerance rule: an interval counts only if its exit clears `TGeoShape::Tolerance()`.
This makes the result independent of the order in which boxes are visited, and equal to the twin.
`DistFromOutside` also keeps a running bound on the nearest entry and skips every box the ray
enters beyond it. Each box it keeps is still clipped to [0, step], so the answer does not change.

- `DistFromOutside` is the first entry at t > 0.
- `DistFromInside` is the far end of the interval of the union across cells that contains t = 0.

Both follow ROOT's `iact`/`step` contract: for `iact` below 3 they compute `Safety` first, and
`iact` 0, or `iact` 1 with `step` below the safety, returns without tracing. Scratch buffers and
traversal stacks are `thread_local`.

### 5.3 `Capacity`

`Capacity` is the sum of the per-cell volumes, which the converter takes from OCCT `GProp` on each
source piece. The cells are disjoint, so no inclusion–exclusion is needed. The value is inherited
from OCCT rather than computed from the shipped solid.

### 5.4 `Safety`

- **Outside:** the distance to the nearest box is a lower bound, because every point of the solid
  lies in some box.
- **Inside:** in a box with an empty active list, the distance to that box's faces is a bound. In
  an undecided box the answer is 0.

### 5.5 The rest of the `TGeoShape` contract

`ComputeBBox` is the union of the retained boxes, which is tighter than the union of the cell
boxes. `ComputeNormal` is the gradient of the active halfspace closest to equality: 2·sign·(Ax + b)
for a quadric, or the signed-distance gradient for a torus. It is normalised and oriented along
`dir`. Drawing follows `O2Tessellated`.

`GetPointsOnSegments` fires deterministic rays from the boxes that carry boundary and keeps a
crossing only where `Contains` changes within `kFlipProbe` either side, so a face shared by two
cells never yields a point. The overlap check (`O2OverlapCheck`) applies the same flip test to
every `O2FlatCSG` point it samples, because `Safety` is 0 inside an undecided box and so cannot
show that a point is on the boundary.

## 6. The `_Loop` twins

`Contains_Loop`, `DistFromOutside_Loop` and `DistFromInside_Loop` walk all cells and all
halfspaces, without the BVH and without active lists. They define the answer, and the tests require
bit identity with the accelerated queries. `Safety_Loop` walks all boxes without the BVH. It must
equal `Safety` and must also be a sound bound.

## 7. Persistence

The generated `geom.C` constructs the shape and fills it with `LoadFlatCSG(file, solid)` from
`flatcsg_<VOLNAME>_<LID>.bin`, then calls `CloseShape()`. The BVH and the sub-cell boxes are never
stored; they are rebuilt on load.

The class also has an automatic ROOT streamer. A `#pragma read` rule in `CADSupportLinkDef.h` calls
`CloseShape()` on every object read, and reports an error if the build is refused. A shape read
from a file is therefore closed; only a shape built by hand needs an explicit `CloseShape()`.

### 7.1 Flat-CSG sidecar format (`flatcsg_<VOLNAME>_<LID>.bin`)

The format is read and written by `o2::cad::LoadFlatCSG` / `WriteFlatCSG`. The production writer is
`tools/cadsupport/flat.py`; the two writers must agree byte for byte. Integers are little-endian
`int32`/`uint32`, geometry values are little-endian `float64`, and lengths are in cm.

```
magic          char[8]   "O2FLTCSG"
version        uint32    1
nHalfspaces    uint32
nCells         uint32
halfspaces     nHalfspaces * { int32 kind; float64 sign; float64 c[11] }
cells          nCells     * { int32 first; int32 count; float64 volume;
                              float64 lo[3]; float64 hi[3] }
```

- `kind` is 0 for a quadric and 1 for a torus. `sign` is ±1. The layout of `c` is given in 3.1;
  unused slots are written but ignored.
- `first` and `count` give the cell's halfspace range. The loader rejects first < 0, count ≤ 0 and
  first + count > nHalfspaces. It also rejects an unknown `kind`, a non-finite coefficient and a
  torus with a zero axis.
- `volume` is the OCCT volume of the source piece.
- `lo` and `hi` are the cell box passed to `SetCellBBox`: an outer bound owed by the converter.

Records are packed without padding: 100 bytes per halfspace and 64 bytes per cell. They are read
field by field, because the natural C++ struct pads to 104 bytes. The loader checks the remaining
file length against nHalfspaces·100 + nCells·64 before reading any record, so a truncated or
overlong file is refused. `WriteFlatCSG` refuses a shape that is not closed, because its unset cell
boxes would be written as zeros.

## 8. The converter side

The decomposition (`tools/cadsupport/decompose.py`) splits a part into cells at trusted concave
edges. Its budget covers the whole working set of cells, pending pieces and unresolved pieces:
`PART_MAX_CELLS` = 64, `MAX_SPLITS` = 256 and `TIMEOUT_S` = 60 s by default. The converter can raise
all three with `--max-cells`, `--max-splits` and `--decompose-timeout`.

The flat emitter (`tools/cadsupport/flat.py`) maps each carrier from `_halfspace_carriers` directly
to a quadric or torus block. The flat path has its own budgets, `_PART_MAX_FLAT_CELLS` = 256 and
`_PART_MAX_FLAT_HALFSPACES` = 1024. The tree path keeps `_PART_MAX_LEAVES` = 64.

Ordering rules:

- The flat path runs last, after every whole-part matcher, the single-cell reading and the
  union-of-cells tree have declined. No part that another tier accepts changes representation.
- A one-piece decomposition keeps the whole-part guards: an all-planar body belongs to the prism
  templates and a one-carrier body to the tier-1 templates. The flat path does not overrule them.
- A single cell is admissible in the class and in `primitives.flat_cells`.

## 9. Open measurements

- The crossover between flat and composite emission, in cells and in halfspaces.
- The split parameters (depth, minimum box size, cubify budget) against leaf list length, box
  count, memory and query cost.
- `Safety` quality against the true distance, since a sound but weak bound costs transport steps.

## 10. Acceptance

A flat part ships only if it passes the same tests as any CSG part: the OCCT symmetric difference
within tolerance, the oracle gate, and `checkKnownSource.py` against the source `TGeoShape` when
one exists. The false-accept guard `accept.contains_disagreements` also runs, because
`BRepAlgoAPI_Cut` can report success with no solid in either direction.

## 11. Risks and limits

1. **The sign convention** is the likeliest silent error. It is mitigated by the check against
   `_cell_leaf`.
2. **The clip-inside-the-box rule** is the likeliest acceleration error. It is mitigated by the
   twins.
3. **The range bound is conservative.** Nearly tangent halfspaces stay undecided for many levels,
   which costs boxes but never correctness.
4. **`Capacity` comes from OCCT**, not from the shipped solid.
5. **Geant4 has no direct equivalent.** A pure union of cells with all-interior halfspaces maps to
   `G4MultiUnion`. The general case, with complemented halfspaces, needs a `G4VSolid` subclass
   mirroring this class.
6. **Boundary-gap declines stay declined.** These are parts whose convex pieces are not cells of
   the carrier arrangement. Splitting at every carrier crossing would fix them. This class makes
   the resulting larger cell counts affordable.
