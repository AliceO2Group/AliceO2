# Convert only part of a model

Real engineering assemblies contain far more than you want to simulate — the mounting frame, the
trolley it sits on, sometimes the building. Converting all of it wastes time and fills your geometry
with volumes no particle will ever reach, so the converter offers two independent ways of cutting a
model down. They combine freely.

## Selecting by name

The first is by name. `--include-name` and `--exclude-name` take regular expressions matched against
the part name stored in the CAD file, case-insensitively, and either may be repeated. Matching an
assembly takes its whole subtree along with it, which is usually what you want:

```bash
--include-name 'Bucket' --exclude-name '^SOLID\b'
```

Add `--name-filter-case-sensitive` if you need the matching to respect case.

## Selecting by region

The second is geometric. `--clip-box` restricts the conversion to an axis-aligned box, given as
`xmin ymin zmin xmax ymax zmax` in the assembly's global frame. Note that these are **STEP file
units**, before the conversion to centimetres — so if your file is in millimetres, so is your clip
box:

```bash
--clip-box -50 -50 -20 50 50 20
```

Every solid is then classified against that box before any meshing happens. Solids fully outside are
dropped; solids fully inside are kept unchanged; and solids straddling the boundary are cut against
it with a boolean intersection, so only the part inside survives. Assemblies left with no surviving
children disappear from the output tree altogether.

By default, subtrees that end up entirely inside the box keep their shared logical definitions, which
keeps the output compact when a part is repeated many times. If you need one distinct volume per
surviving occurrence instead — say because you want to name them individually later — pass
`--clip-deduplicate none`.
