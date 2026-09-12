# Field and cuts

There is one place where the converter cannot give you everything, and it is worth being explicit
about rather than discovering later. A CAD file describes a *part*. It cannot describe how you want
that part simulated — how the magnetic field should be integrated through it, how long a step may be,
which secondaries are worth producing. Those are simulation choices, and no CAD format has anywhere
to record them.

## Magnetic field

For the field there is a clean answer. Pass `--in-field` when the module sits inside the magnet, and
the emitted macro will ask the **live** field for its integration method and maximum field strength
at the moment the geometry is built — which is exactly what a hand-written O2 detector does from its
own `createMaterials()`. Nothing is baked into the file:

`geom.C · emitted`

```cpp
int   cad_ifield = 2;
float cad_fieldm = 10;
cadFieldTrackingParams(cad_ifield, cad_fieldm);   // queries the loaded field
med_Stainless_Steel->SetParam(1, cad_ifield);     // ifield, from the live field
med_Stainless_Steel->SetParam(2, cad_fieldm);     // fieldm, from the live field
```

The `2,10` you see there is only a seed, used if no field happens to be loaded, and `--in-field 1,5.5`
overrides it. To confirm that the query really happened, check `fieldm` rather than `ifield`:
`ifield = 2` is also the seed value and therefore proves nothing, whereas a `fieldm` the seed could
not have produced — ALICE reports 15 — proves the live field answered.

## Step control and physics cuts

> [!WARNING]
> **These silently default to nothing**
>
> Without `--in-field`, a CAD-authored medium is built through ROOT's three-argument `TGeoMedium`
> constructor, which **zeroes every parameter** — including `ifield`, meaning no field tracking at
> all. Step control (`tmaxfd stemax deemax epsil stmin`) stays at the transport default in every
> case, and special physics cuts are never applied, because there is no `simcuts.dat` for a module
> with no detector directory to hold one. None of this is loud: the simulation runs and the numbers
> look plausible. So set `--in-field` deliberately, and treat cuts as a known open item until your
> study grows into a [real detector](real-detector.md), which is where they come back.
