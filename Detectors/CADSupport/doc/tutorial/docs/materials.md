# Give it materials

So far the geometry has shape but no substance. Without material information every volume is assigned
a dummy medium called `Default`, which is fine while you are checking that things are in the right
place and quite wrong the moment you want physics out of it.

The normal route is the **bill of materials** that the CAD system can export alongside the geometry.
We hand that to the converter as a CSV and it matches each part's material name against a Geant4 NIST
database. The rows it looks for are mechanical part rows in this shape:

`detector_bom.csv`

```csv
Type,...,Part Number,Version,Name,Mass (kg),Material
CAD,Mechanical/Part,Base,AA.01,Base,,Stainless Steel
CAD,Mechanical/Part,BasePin,AA.01,BasePin,,Stainless Steel
```

Adding both files to the conversion is all that is required:

```bash
o2-cad-to-tgeo my.step \
    --output-folder cad_out/mydet -o geom.C \
    --csg auto --exact-surfaces auto --mesh --mesh-prec 0.05 \
    --materials-csv detector_bom.csv \
    --bom-mass-unit kg \
    --g4-nist-json $O2_ROOT/share/CADSupport/tools/g4_nist_database/G4_NIST_DB.json
```

```text
Loaded Geant4 NIST DB with 309 materials from: .../G4_NIST_DB.json
Loaded 13 BOM entries from: detector_bom.csv
```

Matching uses a combined score of name similarity and density plausibility, which handles the fact
that engineers write “Stainless Steel” where Geant4 says `G4_STAINLESS-STEEL`. A confident match
becomes a real `TGeoMixture` carrying its element composition, radiation length and interaction
length. An ambiguous or missing one falls back to a simple material and leaves a comment in `geom.C`
naming the part — so unresolved materials stay visible and greppable rather than silently wrong. The
scoring thresholds are adjustable (`--mat-min-score`, `--mat-ambiguity-delta` and a few others), but
the defaults are usually right, and it is better to fix an ambiguous name in the BOM than to loosen
the matcher.

One nice consequence of feeding in the BOM: where both a part mass and a CAD volume are available,
the converter derives an effective density from them. That is how a perforated bracket or a
partly-filled cable tray ends up with an honest average density instead of the density of solid
metal.

> [!NOTE]
> **If your model came from TGeo in the first place**
>
> Geometry exported out of ALICE with `o2-tgeo-to-cad` and coming back should use `--media-json`
> instead. That rebuilds the original media verbatim, field by field, rather than guessing them from
> names, and takes precedence over the BOM for every part it names. The
> [ITS worked example](its-round-trip.md) does exactly this.
