# Solid navigation harness

`o2-bench-cadsupport-solid-harness` validates and times the navigation of the CAD-derived shapes
part by part. Every part is held in more than one representation, and all of them are scored
against the same sample set.

- **surface:** `surfaces_<part>.bin`, loaded as `O2BVHSurfaceSolid`.
- **mesh:** `facets_<part>.bin`, loaded as `O2Tessellated`. This is also the default sampling
  reference.
- **shape:** `shape_<part>.root`, any `TGeoShape`, as written by the CSG emitter.

The reusable core is `CADSupport/O2SolidHarness.h` (namespace `o2::cad::harness`). It is typed on
`TGeoShape*`, so the unit tests use the same code against ROOT primitives. The front end is
`test/runSolidHarness.cxx`.

## 1. Build a part database

`validation/makeTestPartDB.py` runs the converter on each model and pairs the resulting sidecars
by `<VOL>_<LID>`. A part enters the database only when both `surfaces_*.bin` and `facets_*.bin`
exist. It writes `<db>/manifest.json`.

```bash
python3 $O2_SRC/Detectors/CADSupport/validation/makeTestPartDB.py \
    --models ExcavatorArm.step as1-oc-214.stp --output <db>
```

| option | default | meaning |
| --- | --- | --- |
| `--models F...` | `ExcavatorArm.step as1-oc-214.stp` | CAD files; relative names resolve against `Detectors/CADSupport/examples/` |
| `--output DIR` | `validation/test_part_db` | database directory |
| `--skip-existing` | off | reuse a model's converted directory and only re-index it |
| `--force` | off | regenerate a model's directory even if it exists |
| `--csg off\|auto\|required` | `auto` | converter CSG mode; `auto` records the per-part choice in `csg_report.json` |
| `--include-name RE` | none | passed to the converter; may be repeated |
| `--mesh-prec P` | converter default (0.1) | meshing precision passed to the converter |

Each manifest entry holds `id`, `model`, `volume`, `lid`, `surfaces`, `facets`, `nTriangles` and
`bbox`. It also holds `shape` when a `shape_*.root` exists, and `shipped`, the representation chosen by
the converter's cascade.

## 2. Run the harness

```bash
o2-bench-cadsupport-solid-harness --db <db> [options]
o2-bench-cadsupport-solid-harness --surfaces <file> --facets <file> [--shape <file>] [options]
```

| option | default | meaning |
| --- | --- | --- |
| `--db DIR` | — | database built by `makeTestPartDB.py` (reads `manifest.json`) |
| `--surfaces F`, `--facets F` | — | ad-hoc mode: one part given by its two sidecars |
| `--shape F` | derived | a `shape_*.root` to score as well; in `--db` mode it is taken from the manifest or derived from the `surfaces_*.bin` name |
| `--parts S` | all | only parts whose id contains the substring `S` |
| `--points N` | 5000 | point samples per part |
| `--rays N` | 5000 | ray samples per part |
| `--seed N` | 1 | sampling seed |
| `--only LIST` | `contains,distout,distin,safety` | kernels to run |
| `--loop-crosscheck` | off | also run the surface solid's `_Loop` twins and require exact agreement |
| `--pruning-ab` | off | re-run the distance kernels with ray `tmax` pruning off and report candidate counts and ns/call both ways |
| `--rims` | off | list every trim loop, not only the unmatched ones |
| `--edge-identity` | off | print the sidecar-v3 edge-identity counts and the maximum shared-edge deviation |
| `--json F` | none | write the full report as JSON |
| `--warmup N` | 1 | untimed passes before timing |
| `--repeat N` | 3 | timed passes |
| `--dump-samples D` | none | write each part's sample set to `D/samples_<part>.json` |
| `--load-samples D` | none | read sample sets from `D` instead of generating them; `--points`, `--rays` and `--seed` are then ignored |
| `--ref-answers D` | none | validate against `D/answers_<part>.json` from the OCCT oracle instead of the mesh |
| `-h`, `--help` | | print usage |

A `shape_*.root` file holds one `TGeoShape` under the key `"shape"`, in cm. It may also hold an
optional `TGeoHMatrix` under `"placement"`, which takes the shape from its own frame into the part's
frame. Points and rays are transformed into the shape's frame before it is queried.

## 3. What is measured

**Sampling** is deterministic, given the seed and the bounding box. It draws:

- bulk points in the inflated box;
- points within a band of the reference surface;
- points the reference calls inside;
- rays from outside points, half of them aimed at random interior points so that
  `DistFromOutside` hit rates stay meaningful;
- rays from inside points.

**Validation** compares each representation with the reference. Every disagreement is sorted into
one of four bins:

- within the reference's own band (for the mesh, the chord sagitta);
- a missed surface (a wall missed or tunnelled through; this is never excused);
- unexplained;
- no verdict (the oracle declined).

The worst offenders are printed with their point and direction, so each one can be reproduced.
`Safety` is checked only against its contract, 0 ≤ safety ≤ true distance, and never compared
between two shapes.

**Timing** runs each kernel over the same sample order for every representation. A checksum of the
results stops the compiler from removing the calls. The output is ns/call and the ratio between
representations. The run also reports primitive counts, `CloseShape` time and BVH candidate counts.

**Reliability** of each surface solid is printed per part as a `navigation:` line, and in the JSON
under `navigation`: the reliability state, whether the part is navigable, and the rim and edge
counts. Unnavigable parts are listed again at the end. An accuracy figure for a part that is not
navigable describes an incomplete solid.

## 4. Rules for reading the output

- **The mesh is a reference, not the truth.** It is inscribed, so on curved parts the exact solid
  exits later along inside rays and enters later from outside. Mismatches within the band are
  expected.
- **Compare against `O2Tessellated`, never `TGeoTessellated`.** The ROOT class does not implement
  navigation and falls back to its bounding box.
- **The `_Loop` cross-check is the correctness guard that does not involve the mesh.** The BVH and
  loop paths minimise over the same hits, so any difference is a traversal bug.
- **Seeds are fixed.** A number that cannot be reproduced exactly is not a measurement.
- **Look at the per-part numbers.** The spread between parts is wide, so a median alone hides it.

## 5. OCCT oracle round trip

The OCCT oracle gives exact answers from the part's BREP, which the converter writes with
`--dump-brep`.

```bash
o2-bench-cadsupport-solid-harness --db <db> --dump-samples /tmp/o
python3 $O2_SRC/Detectors/CADSupport/validation/occtOracle.py \
    --brep <part>.brep --samples /tmp/o/samples_<part>.json --out /tmp/o/answers_<part>.json
o2-bench-cadsupport-solid-harness --db <db> --ref-answers /tmp/o
```

With `--ref-answers`, the tolerance band is the model's declared tolerance. The oracle's own
classification of each ray origin decides which entry point is asked. A disagreement outside the
tolerance is a defect. `validation/runOracleGate.py` automates the conversion, sampling, oracle and
scoring for one model or for the fixture set.

## 6. Profiling

`--only` with a single kernel and one part is the entry point for `perf`:

```bash
perf record -g o2-bench-cadsupport-solid-harness --db <db> --parts <part> --only distout --rays 200000
```
