# CAD-to-TGeo geometry import

`O2_CADtoTGeo.py` converts CAD geometries exported as STEP files into ROOT TGeo geometry.
The converter emits a small ROOT macro plus compact binary facet payloads. The generated
macro can be loaded directly in ROOT, or injected into `o2-sim` as an external passive
module or as a sensitive external detector.

The current integration path is data-driven: the CAD geometry is converted once, then a
JSON file tells `o2-sim` which generated macro to load, where to anchor it in the existing
geometry, and, for sensitive detectors, which volumes or media should produce hits.

## Software setup

The preferred setup is the normal ALICE software environment. The `pythonOCC` package pulls
in OpenCascade and the Python bindings needed by the converter:

```bash
alienv enter O2sim/latest,pythonOCC/latest
python PATH_TO_ALICEO2_SOURCES/scripts/geometry/O2_CADtoTGeo.py --help
```

If you are working from a local O2 checkout, `PATH_TO_ALICEO2_SOURCES` is the directory that
contains this `scripts/geometry` folder.

For standalone studies outside the ALICE software stack, a conda environment with
`pythonocc-core` can also be used:

```bash
conda create -n occ python=3.10 -y
conda activate occ
conda install -c conda-forge pythonocc-core -y

python PATH_TO_ALICEO2_SOURCES/scripts/geometry/O2_CADtoTGeo.py --help
```

## Convert a STEP file

For a quick, robust geometry preview, convert leaves to bounding boxes:

```bash
mkdir -p cad_out/mydet
python PATH_TO_ALICEO2_SOURCES/scripts/geometry/O2_CADtoTGeo.py \
  my_detector.step \
  --output-folder cad_out/mydet \
  -o geom.C \
  --step-unit auto
```

For a more detailed faceted representation, enable meshing:

```bash
python PATH_TO_ALICEO2_SOURCES/scripts/geometry/O2_CADtoTGeo.py \
  my_detector.step \
  --output-folder cad_out/mydet \
  -o geom.C \
  --mesh \
  --mesh-prec 0.05 \
  --step-unit auto
```

The output folder contains:

- `geom.C`, a ROOT macro exporting `get_builder_hook_unchecked()`
- `facets_*.bin`, one compact triangle payload per leaf logical volume

The generated macro is the file referenced from the external-geometry JSON examples below.
The macro and its facet binaries should stay together, because the macro loads the facet files
relative to its own location.

## Restrict the converted region with a clip box

Large CAD assemblies often contain far more than the region of interest. The `--clip-box`
option restricts the conversion to an axis-aligned bounding box, so only the geometry inside
(or overlapping) that box is written out:

```bash
python PATH_TO_ALICEO2_SOURCES/scripts/geometry/O2_CADtoTGeo.py \
  my_detector.step \
  --output-folder cad_out/mydet \
  -o geom.C \
  --mesh \
  --clip-box XMIN YMIN ZMIN XMAX YMAX ZMAX
```

Notes on the coordinates:

- The six values are `xmin ymin zmin xmax ymax zmax` and must satisfy `xmin < xmax`,
  `ymin < ymax`, and `zmin < zmax`.
- Coordinates are given in the **STEP file units** (before conversion to cm), and are applied
  in the global/world coordinate system of the assembly.

Each solid is classified against the box before meshing:

- Solids fully outside the box are dropped.
- Solids fully inside the box are kept unchanged.
- Solids straddling the box boundary are cut against it (a boolean intersection), so only the
  part inside the box is meshed.

Assemblies that end up with no surviving children are removed from the output tree.

### Deduplication mode

The `--clip-deduplicate` option controls how subtrees that fall entirely inside the box are
emitted:

- `intact` (default): subtrees fully inside the box reuse their original shared logical
  definitions, keeping the output compact.
- `none`: every surviving occurrence becomes its own volume, which is useful when you need a
  flat, per-instance representation.

```bash
python PATH_TO_ALICEO2_SOURCES/scripts/geometry/O2_CADtoTGeo.py \
  my_detector.step \
  --output-folder cad_out/mydet \
  -o geom.C \
  --mesh \
  --clip-box -50 -50 -20 50 50 20 \
  --clip-deduplicate none
```

Clipping can be combined with the name-based selection options (`--include-name` /
`--exclude-name`) to further narrow down which parts are converted.

## Optional material mapping

The converter can use a BOM CSV to assign materials and, when part masses and CAD volumes are
available, derive effective densities. A Geant4 NIST material JSON dump enables richer material
and tracking-length information:

```bash
python PATH_TO_ALICEO2_SOURCES/scripts/geometry/O2_CADtoTGeo.py \
  my_detector.step \
  --output-folder cad_out/mydet \
  -o geom.C \
  --mesh \
  --materials-csv detector_bom.csv \
  --bom-mass-unit kg \
  --g4-nist-json g4_nist_materials.json
```

The expected BOM rows are mechanical part rows of the form:

```text
CAD,Mechanical/Part,<PartNumber>,<Revision>,<Name>,<Mass>,<Material>,...
```

Material names are matched to the NIST database when possible. If a match is ambiguous or not
available, the generated macro falls back to a simple material and leaves comments in `geom.C`
for follow-up.

## Inject passive CAD geometry into `o2-sim`

Passive external modules are configured under an `externalModules` array. Each entry needs a
module name, the generated macro, and an anchor volume already present in the O2 geometry. An
optional placement can translate and rotate the imported geometry inside the anchor volume:

```json
{
  "externalModules": [
    {
      "name": "IRIS",
      "title": "IRIS support from CAD",
      "macro": "cad_out/iris/geom.C",
      "anchor": "barrel",
      "placement": {
        "translation": [0.0, 0.0, 0.0],
        "rotation_deg": [0.0, 0.0, 15.0]
      }
    }
  ]
}
```

The module is only added when its `name` is present in the active detector/module list. One
way to make such a list is a detector-list JSON file:

```json
{
  "EXTCAD": ["IRIS"]
}
```

Run `o2-sim` with both files:

```bash
o2-sim -n 1 -g boxgen \
  --detectorList EXTCAD:detectorlist.json \
  --extGeomFile externalGeometry.json
```

Multiple passive modules can be listed in the same file. The CAD macro loader JIT-compiles each
macro into a unique namespace, so several `O2_CADtoTGeo.py` outputs can coexist even though they
export the same builder-hook symbol names.

## Inject sensitive external CAD detectors

Sensitive external detectors are configured under an `externalDetectors` array. They use the
same generated geometry macro, but additionally select sensitive volumes or media and bind the
detector to a free O2 `DetID` slot. All such detectors are instances of
`o2::ext::ExternalDetector` and write the generic `o2::ext::Hit` format.

```json
{
  "externalDetectors": [
    {
      "name": "ECYL",
      "title": "External silicon cylinder",
      "macro": "cad_out/ecyl/geom.C",
      "anchor": "barrel",
      "detID": "ITS",
      "sensitiveVolumes": ["ECYL_SENSOR"],
      "placement": { "translation": [0.0, 0.0, 0.0] }
    },
    {
      "name": "EDISK",
      "title": "External endcap disk with custom action",
      "macro": "cad_out/edisk/geom.C",
      "anchor": "barrel",
      "detID": "TST",
      "sensitiveMedia": ["Silicon"],
      "sensitiveMacro": "sensitive_action.macro",
      "sensitiveFunction": "sensitiveAction()"
    }
  ]
}
```

Selection rules:

- `sensitiveVolumes` matches substrings of TGeo volume names.
- `sensitiveMedia` matches substrings of TGeo medium names.
- At least one of the two arrays must be non-empty.

The `detID` determines the hit-file identity, for example `o2sim_HitsITS.root` or
`o2sim_HitsTST.root`. Choose a DetID that is not already occupied by an active built-in detector.
The branch name keeps the external detector name, for example `ECYLHit` or `EDISKHit`.

If no `sensitiveMacro` is provided, the built-in action records a charged-track entrance/exit hit.
With a custom action, the macro is JIT-compiled at runtime and must return an
`o2::ext::ExternalDetector::SensitiveFcn`. The action can query `TVirtualMC::GetMC()` and use
helpers such as `currentSensorID()`, `currentTrackID()`, and `addHit()`.

Run the detectors just like passive modules, with their names in the detector-list JSON:

```json
{
  "EXTCAD": ["ECYL", "EDISK"]
}
```

```bash
o2-sim -j 2 -n 5 -g boxgen \
  --detectorList EXTCAD:detectorlist.json \
  --extGeomFile externalGeometry.json \
  --configKeyValues 'BoxGun.number=50'
```

In parallel mode, the hit merger reads the same `--extGeomFile`, registers the configured active
external detectors, and persists their generic external hits like built-in detector hits.

## Complete runnable example

A self-contained example is available in:

```bash
run/SimExamples/External_Sensitive_Detectors
```

It defines two artificial sensitive detectors entirely from data:

- `ACYL`, a silicon barrel cylinder using the built-in entrance/exit action
- `BDISK`, a silicon endcap disk using a custom JITed sensitive action

The example uses hand-written geometry macros that mimic `O2_CADtoTGeo.py` output, so it does not
require CAD input files. Run it from its directory:

```bash
cd run/SimExamples/External_Sensitive_Detectors
./run.sh
```

The script transports a few box-generator events and prints the hit counts for the produced
external-detector branches.