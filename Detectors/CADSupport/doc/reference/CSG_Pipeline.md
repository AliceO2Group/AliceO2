# The CSG pipeline

The CSG pipeline converts a CAD leaf solid into a native ROOT CSG shape, or into `O2FlatCSG`, when
the solid can be described exactly by combining analytic carriers. It is enabled with
`O2_CADtoTGeo.py --csg auto|required`, and the code lives in `tools/cadsupport/`.

## 1. Why CSG

A B-rep describes a solid by its faces and by where each face stops. The "where it stops" needs
trim curves, and the intersection curve of two quadrics cannot be represented exactly in either
face's chart. A CSG description needs only the carriers and a sign for each: a point is inside
when its signs match. The intersection curve is implied by two sign tests and never represented,
so the two faces have nothing to disagree about.

## 2. The cascade

With `--csg auto` the converter tries three representations per leaf solid, in order:

    CSG  ->  exact surfaces (O2BVHSurfaceSolid)  ->  tessellated (O2Tessellated)

With `--csg required`, the run stops with a report if any leaf is not CSG. `geom.C` builds the
first representation that is accepted. The other representations are still written if they were
requested, so that the validation can score every representation of a part side by side.

The converter prints a cascade table and writes `csg_report.json`, which records each part's choice
and the evidence for it. `tools/cadsupport/decline_catalogue.py` joins that report with
`surface_report.json` into one table of the reasons a part declined each tier.

## 3. Recognition

`tools/cadsupport/recognise.py` proposes a description from the part's carriers. Its matchers form
a ladder from specific to general:

1. **Elliptic or toroidal laterals:** a part with an extruded-ellipse face goes to the `TGeoEltu`
   template, and one with a toroidal face to the `TGeoTorus` template.
2. **Box or prism:** a `TGeoBBox` or, for any other all-planar part, a stack of planar sections
   read as `TGeoTrd1`, `TGeoTrd2`, `TGeoArb8`, `TGeoXtru` or `TGeoPgon`.
3. **Sphere:** `TGeoSphere`.
4. **One axis:** a tube, tube segment or cone (`TGeoTube`, `TGeoTubeSeg`, `TGeoCone`), and failing
   that a revolved profile with any number of z sections, as a `TGeoPcon`.
5. **Two axes:** two cylinder clusters on non-parallel axes (a barrel and a lug), as
   `TGeoTube ∪ TGeoTube`. A part with more than two axis clusters does not enter this rung.
6. **Single cell:** one intersection of halfspaces.
7. **Union of cells:** the decomposition of section 4, emitted as a `TGeoCompositeShape` of at most
   `_PART_MAX_LEAVES` (64) leaves.
8. **Flat cells:** the same decomposition, emitted as `O2FlatCSG`
   ([`Design_FlatCSGSolid.md`](Design_FlatCSGSolid.md)).

A decline anywhere in rungs 1 to 5 passes the part to rung 6. Each of rungs 6 to 8 runs only when
the one before it has declined.

All thresholds are relative to the part's bounding-box diagonal (`REL_TOL`, `ANG_TOL` = 1e-6).
Every unhandled structure returns a reason, not a guess. Extents come from
`BRepTools.UVBounds` of the trimmed face, not from the carrier.

**Tier 0** (`tools/cadsupport/tier0.py`) lets a face stored as a B-spline take part as the plane,
cylinder, cone, sphere or torus it exactly is. Recognition then treats it like a natively analytic
face.

## 4. Decomposition

`tools/cadsupport/decompose.py` splits a part into cells:

    start from the part's connected solids;
    while a piece has a trusted concave (or mixed) edge:
        extend the carrier of one of the edge's faces to a full surface;
        split the piece with BRepAlgoAPI_Splitter;
    a piece with no trusted concave edge is one cell.

Connected solids come first, because a part with no concave edges can still be several disjoint
pieces. The split pieces must sum to the part's volume within `VOLUME_REL_TOL` (1e-6), or the part
declines.

The budgets apply to the whole working set:

| budget | default | override |
| --- | --- | --- |
| `PART_MAX_CELLS` | 64 | `--max-cells` |
| `MAX_SPLITS` | 256 | `--max-splits` |
| `TIMEOUT_S` | 60 s | `--decompose-timeout` |

## 5. Acceptance

A proposal is shipped only if it passes the acceptance tests. The recogniser can therefore be
greedy, because the acceptance is exact.

1. **Symmetric difference** (`tools/cadsupport/accept.py`): OCCT's
   volume(candidate − original) + volume(original − candidate) must not exceed
   `_BAND_FACTOR` (1.0) × model tolerance × area(original). The candidate is an OCCT realisation of
   the description.
2. **False-accept guard** (`accept.contains_disagreements`): `BRepAlgoAPI_Cut` can report success
   with no solid in either direction. A sampled containment comparison catches that case.
3. **Oracle gate** (`validation/runOracleGate.py`): scores the ROOT realisation of the same
   description against the OCCT oracle.
4. **Known source** (`validation/checkKnownSource.py`): scores a shape against the `TGeoShape` it
   was exported from, when the model came from TGeo.

`tools/cadsupport/primitives.py` realises one description twice, with `build_occ()` and
`build_root()`. The symmetric difference and the gate therefore test the same description through
two independent builders.

## 6. Outputs

| file | content |
| --- | --- |
| `shape_<VOL>_<LID>.root` | the accepted `TGeoShape` under key `"shape"`, in cm |
| `csg_<VOL>_<LID>.json` | the description and its evidence |
| `flatcsg_<VOL>_<LID>.bin` | an `O2FlatCSG` part (format in `Design_FlatCSGSolid.md` 7.1) |
| `csg_report.json` | the per-part cascade decision |

Writing a `.root` file needs PyROOT. When ROOT cannot be imported, only the JSON is written, and
`python3 -m cadsupport.emit --from-json <dir>` produces the `.root` files afterwards. A part whose
`.root` file does not exist is not dispatched to CSG in `geom.C`.

`validation/csgCensus.py` measures, per solid, which representation could apply: face types,
whether the solid is quadric-only, concave-edge counts, tier-1 template matches and Tier-0
candidates. It uses OCCT's `ShapeAnalysis_CanonicalRecognition` only as a cross-check. Its topology
helpers are in `tools/cadsupport/census.py`. `python3 -m cadsupport.emit --self-test` runs the
package self-tests.

## 7. Limits

- **Free-form surfaces cannot be CSG.** A solid with a genuine B-spline face goes to the surface or
  mesh tier.
- **Tangential carriers** are where splitting is least robust. A failed split declines the part to
  the next tier.
- **Boundary gaps.** A convex splitter piece need not be a cell of the carrier arrangement, and such
  parts decline.
- **Budgets.** Very deep booleans exceed the decomposition budgets unless they are raised.
- **Tolerance.** Acceptance is against OCCT's tolerant model, so "equal" means equal to the model
  tolerance.

## 8. Relation to `O2BVHSurfaceSolid`

The two exact representations complement each other. CSG copes with deep boolean structure but
scales poorly with face count. The surface solid scales with face count through its BVH but has
to represent every seam as a trim curve. The cascade takes the seams the surface solid cannot
represent exactly and leaves it the many-face parts and arbitrary trims.
