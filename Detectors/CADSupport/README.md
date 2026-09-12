# CAD support: STEP to TGeo and back

`Detectors/CADSupport` converts CAD geometry exported as STEP into ROOT TGeo geometry for
simulation. It also exports TGeo geometry back to STEP.

The converter writes one ROOT macro, `geom.C`, together with its binary payloads. The macro can be
loaded in ROOT on its own, or injected into `o2-sim` as a passive module or as a sensitive external
detector. Injection is data-driven: a JSON file tells `o2-sim` which macro to load, where to anchor
it and, for detectors, which volumes produce hits. Nothing is recompiled.

The tutorial `doc/tutorial/index.html` walks through the whole route on the shipped `ExcavatorArm.step`
model. This file is the option reference.

## Software setup

The converter needs pythonOCC, which is a separate aliBuild package:

```bash
aliBuild build pythonOCC --defaults o2 --no-system SWIG
alienv enter O2sim/latest,pythonOCC/latest
o2-cad-to-tgeo --help
o2-cad-to-tgeo --self-test
```

The installed wrappers `o2-cad-to-tgeo` and `o2-tgeo-to-cad` run
`$O2_ROOT/share/CADSupport/tools/O2_CADtoTGeo.py` and `O2_TGeoToCAD.py`. The example models are
installed in `$O2_ROOT/share/CADSupport/examples/`. The Geant4 NIST material table is
`$O2_ROOT/share/CADSupport/tools/g4_nist_database/G4_NIST_DB.json`. The legacy names
`O2_CADtoTGeo.py` and `O2_TGeoToCAD.py` are installed alongside them and work the same way.

Outside the ALICE stack, a conda environment with `pythonocc-core` also works. There, run the
script from the source tree:

```bash
conda create -n occ -c conda-forge python=3.10 pythonocc-core -y
conda activate occ
python3 $O2_SRC/Detectors/CADSupport/tools/O2_CADtoTGeo.py --help
```

## Convert a STEP file

```bash
mkdir -p cad_out/excavator
o2-cad-to-tgeo $O2_ROOT/share/CADSupport/examples/ExcavatorArm.step \
    --output-folder cad_out/excavator -o geom.C --step-unit auto \
    --csg auto --exact-surfaces auto --mesh --mesh-prec 0.05
```

Each leaf solid is carried by the first representation that accepts it:

| representation | flag | shape class | payload |
| --- | --- | --- | --- |
| native ROOT CSG, or a flat CSG solid | `--csg auto\|required` | `TGeoBBox`, `TGeoTube`, ..., `TGeoCompositeShape`, `O2FlatCSG` | `shape_*.root`, `flatcsg_*.bin` |
| exact trimmed surfaces | `--exact-surfaces auto\|required` | `O2BVHSurfaceSolid` | `surfaces_*.bin` |
| triangle mesh | `--mesh` | `O2Tessellated` (`--mesh-solid o2`, default) | `facets_*.bin` |

`off` is the default for `--csg` and `--exact-surfaces`. `auto` uses a tier where it is accepted
and falls through elsewhere. `required` stops with a report if any leaf cannot use it. Without
`--mesh`, the fallback tier emits bounding boxes.

`--mesh-prec` sets both the linear and the angular deflection of the OCCT mesher; the default is
0.1. `--mesh-solid tgeo` emits ROOT's `TGeoTessellated`, which does not implement navigation;
use it only for a macro that must load outside O2.

The output folder holds:

- `geom.C`;
- the payloads above;
- `csg_report.json` (with `--csg`);
- `brep_*.brep` (with `--dump-brep`);
- `surface_report.json` (with `--surface-report PATH`).

The macro loads its payloads relative to its own location, so move the folder as a whole.

`geom.C` exports `get_builder_hook_unchecked()`, which `o2-sim` calls, and
`build_and_export(const char* out_root = "geom.root", bool check = true, bool checkOverlaps = false)`
for standalone use:

```bash
(cd cad_out/excavator && root -l -b -q -e '.L geom.C' -e 'build_and_export("geom.root");')             # build and export
(cd cad_out/excavator && root -l -b -q -e '.L geom.C' -e 'build_and_export("geom.root", true, true);')  # also CheckOverlaps
```

Other conversion options:

| option | meaning |
| --- | --- |
| `--step-unit auto\|mm\|cm\|m\|in\|ft` | STEP length unit; `auto` reads the file's declaration |
| `--recognize-surfaces exact\|off` | recover exact planes, spheres, cylinders and cones stored as NURBS (default `exact`) |
| `--surface-report PATH` | per-face classification and exact-conversion eligibility, as JSON |
| `--csg-report PATH` | where to write `csg_report.json` |
| `--max-cells N`, `--max-splits N`, `--decompose-timeout S` | raise the CSG decomposition budgets (defaults 64, 256, 60 s) |
| `--print-tree` | print the assembly tree and exit |
| `--in-field [IFIELD,FIELDM]` | take field tracking parameters from the live field (seed `2,10`) |

## Convert part of a model

`--include-name RE` and `--exclude-name RE` select CAD labels by regular expression. Both may be
repeated, and a matching assembly includes its whole subtree. Matching is case-insensitive unless
`--name-filter-case-sensitive` is given.

`--clip-box XMIN YMIN ZMIN XMAX YMAX ZMAX` keeps only the geometry inside an axis-aligned box. The
box is given in STEP file units, in the assembly's world frame, with each minimum below its
maximum.

- Solids fully outside the box are dropped.
- Solids fully inside are kept.
- Solids that straddle the boundary are intersected with the box.
- Assemblies left with no children are removed.

`--clip-deduplicate intact` (the default) reuses shared definitions for subtrees fully inside the
box. `none` makes one volume per surviving occurrence.

## Materials

A bill-of-materials CSV assigns materials and, where masses and CAD volumes are both available,
effective densities. Material names are matched against the Geant4 NIST table:

```bash
o2-cad-to-tgeo $O2_ROOT/share/CADSupport/examples/ExcavatorArm.step \
    --output-folder cad_out/excavator -o geom.C --csg auto --exact-surfaces auto --mesh \
    --materials-csv $O2_ROOT/share/CADSupport/examples/ExcavatorArm_MATERIALS.csv \
    --bom-mass-unit kg \
    --g4-nist-json $O2_ROOT/share/CADSupport/tools/g4_nist_database/G4_NIST_DB.json
```

Rows are read when the first two columns are `CAD,Mechanical/Part`. Their layout is
`CAD,Mechanical/Part,<PartNumber>,<Revision>,<Name>,<Mass>,<Material>,...`.

An ambiguous or missing match falls back to a simple material and leaves a comment in `geom.C`. The
matching is tuned by `--mat-min-score`, `--mat-ambiguity-delta`, `--mat-w-token`,
`--mat-w-density`, `--mat-max-log-density-diff` and `--mat-compound-penalty`.

Geometry that came out of TGeo with `o2-tgeo-to-cad` should instead use `--media-json`. That
rebuilds the original media verbatim and takes precedence over the BOM.

Without `--in-field`, a CAD medium has all tracking parameters zero, including `ifield`.

## Passive geometry in `o2-sim`

`externalGeometry.json`:

```json
{
  "externalModules": [
    {
      "name": "EXCV",
      "title": "Excavator support structure from CAD",
      "macro": "cad_out/excavator/geom.C",
      "anchor": "barrel",
      "placement": { "translation": [21.01, -13.22, -19.66], "rotation_deg": [0.0, 0.0, 0.0] }
    }
  ]
}
```

`detectorlist.json`:

```json
{ "EXTCAD": ["EXCV"] }
```

```bash
o2-sim -n 1 -g boxgen --detectorList EXTCAD:detectorlist.json --extGeomFile externalGeometry.json
```

A module is added only when its `name` is in the active module list. `anchor` must be an existing
volume; `barrel` sits at (0, −30, 0) in the cave. `placement` is given in cm and degrees in the
anchor's frame. Several modules, each from its own `geom.C`, can be listed together: the loader compiles
each macro into its own namespace, so their identical function names do not collide.

## Sensitive external detectors

Use an `externalDetectors` array. It takes the same fields as a module, plus `detID` and at least
one of `sensitiveVolumes` or `sensitiveMedia`:

```json
{
  "externalDetectors": [
    {
      "name": "EXCV",
      "title": "Excavator as a sensitive detector",
      "macro": "cad_out/excavator/geom.C",
      "anchor": "barrel",
      "detID": "TST",
      "sensitiveVolumes": ["Bucket"],
      "placement": { "translation": [21.01, -13.22, -19.66] }
    }
  ]
}
```

- `sensitiveVolumes` and `sensitiveMedia` match **substrings** of TGeo volume and medium names.
  `"Bucket"` above selects five volumes.
- `detID` is an existing detector identity that no active built-in detector uses. The default is
  `ITS`. It decides the hit file, for example `o2sim_HitsTST.root`. The branch keeps the module
  name, here `EXCVHit`.
- Without `sensitiveMacro`, the built-in action records one entrance/exit hit per charged track in
  `o2::ext::Hit`.
- A custom action is a macro, named by `sensitiveMacro` and `sensitiveFunction`, that returns an
  `o2::ext::ExternalDetector::SensitiveFcn`. It is compiled at run time and can use
  `TVirtualMC::GetMC()`, `currentSensorID()`, `currentTrackID()` and `addHit()`. See
  `Detectors/External/macro/sensitiveActionExample.macro`.

In parallel mode, the hit merger reads the same `--extGeomFile` and persists the external hits.

`run/SimExamples/External_Sensitive_Detectors` defines two detectors, `ACYL` and `BDISK`, from
hand-written macros. It needs no CAD input; run `./run.sh` there.

## TGeo to STEP

```bash
o2-tgeo-to-cad geometry.root out.step [--top VOLUME] [--include-name RE] [--carve-mothers] \
    [--media-json out_media.json] [--report report.json]
```

`o2-tgeo-to-cad --help` lists the remaining options. Converting the resulting STEP back with
`--media-json` closes the round trip.

## Checks and validation tools

`validation/` is not installed. Run its scripts from `$O2_SRC/Detectors/CADSupport/validation/`.

- `root -l -b -q "$O2_SRC/Detectors/CADSupport/test/checkSurfaceSidecars.macro(\"cad_out/excavator\")"`
  loads every `surfaces_*.bin` in a folder and reports closure, orientation and capacity.
- `--surface-report PATH` shows which faces are exact, recognised or unsupported.
- `validation/makeTestPartDB.py` builds a database of parts held both as surfaces and as meshes.
  `o2-bench-cadsupport-solid-harness` validates and times them. See
  `doc/reference/SolidNavigationHarness.md`.
- `validation/runOracleGate.py` is the acceptance gate: it converts models, samples each part and
  scores it against the OpenCascade oracle. `compareGateRuns.py` compares two gate reports.
- The oracles answer from OpenCascade: `occtOracle.py` per solid, `xrayOracle.py` as crossing lists
  for the X-ray benchmark (`runXRayBench.py`), and `assemblyOracle.py` volume by volume along a ray
  through an assembly. `checkKnownSource.py` scores a part against the `TGeoShape` it came from.
- `validation/overlapCensus.py` sorts every pair of placed solids in a STEP assembly into
  disjoint, touching or interpenetrating.
- `validation/roundTripReport.py` reports what the TGeo → STEP → TGeo round trip made of each part;
  `exportSourceShapes.py` exports the source shapes it compares against.
- `validation/renderTGeo.py` raytraces a TGeo geometry through the navigator into a PNG, coloured
  by representation with `--csg-report`.
- `validation/closure/` runs the same events through a TGeo module and through its STEP round trip
  and compares the hits (`run_closure.sh`).
- `validation/demo/` converts ExcavatorArm into exact and tessellated geometry and compares `o2-sim` runs
  over both (`convert_all.sh`, then `run_all.sh`).

Tests and benchmarks built with the module:

| binary | what |
| --- | --- |
| `o2-test-cadsupport-BVHSurfaceSolid` | unit tests of `O2BVHSurfaceSolid` and the sidecar reader |
| `o2-test-cadsupport-BVHAssembly` | unit tests of `O2BVHAssembly` |
| `o2-test-cadsupport-FlatCSG` | unit tests of `O2FlatCSG` |
| `o2-bench-cadsupport-solid-harness` | per-part validation and timing |
| `o2-bench-cadsupport-xray` | X-ray transport benchmark over a part database |
| `o2-bench-cadsupport-overlap` | overlap census of a placed geometry |

## Reference documents

`doc/reference/`:

- `BVHSurfaceSolid.md`: the exact-surface solid and its sidecar format.
- `Design_FlatCSGSolid.md`: the flat CSG solid and its sidecar format.
- `CSG_Pipeline.md`: CSG recognition and acceptance.
- `TolerancePolicy.md`: every tolerance, with its value and reason.
- `SolidNavigationHarness.md`: the validation harness.
- `Roadmap.md`: deferred work.

Beside them in `doc/`:

- `known-issues.md`: open defects and limitations.
- `ideas.md`: proposals that are not yet decided.
