# Check your geometry

Before trusting any physics that came out of a conversion, it is worth spending a few minutes on four
checks. They are ordered cheapest first, and in practice the first two catch most problems.

## 1 · Read the cascade table

The converter already told you what it decided for every part, and wrote the same information to
`csg_report.json`. A part that declined CSG says which test it failed and by how much, which is often
enough to see that a model is nearly-but-not-quite a primitive. A large tessellated count on a model
you expected to be analytic is the signal to look at `--recognize-surfaces` and the surface report
below.

## 2 · Look for overlaps

Run `build_and_export("geom.root", true, true)` to get `CheckOverlaps`; zero illegal overlaps is what
you want to see. A non-zero count is worth taking seriously, but do not assume it is the conversion's
fault: engineering assemblies are drawn for manufacture, not for particle transport, and slightly
interpenetrating parts are common in perfectly good CAD models.

## 3 · Confirm the exact solids really load

Successfully extracting a solid's surfaces does not guarantee the result is a usable, watertight body.
This macro loads every `surfaces_*.bin` in a directory the same way the transport does, and reports
closure, orientation consistency and enclosed volume:

```bash
# $O2_SRC is your AliceO2 source directory
root -l -b -q "$O2_SRC/Detectors/CADSupport/test/checkSurfaceSidecars.macro(\"cad_out/excavator\")"
```

```text
OK    surfaces_Bucket_0_1_1_6.bin      surfaces=   97  closed=1  orient=1  capacity=58.3121
OK    surfaces_Base_0_1_1_3.bin        surfaces=   44  closed=1  orient=1  capacity=241.281
...
SUMMARY cad_out/excavator
  sidecars found            : 13
  loaded                    : 13
  rejected by the reader    : 0
  loaded but not IsClosed() : 0
  orientation inconsistent  : 0
```

`closed=1` means the solid is a watertight manifold, which is precisely what navigation requires. Any
non-zero number on the last three summary lines identifies a part that will not transport correctly.

## 4 · Find out what the geometry really is

A subtlety worth knowing: the surface type stored in a STEP file describes the *exporter*, not the
geometry. CAD kernels routinely write an exact cylinder as a rational B-spline, which is an exact
representation rather than an approximation — but dispatching on the stored type would throw that
exactness away. The converter therefore classifies faces by their actual shape, and its surface report
shows the effect:

```bash
# a per-face classification, written alongside a normal conversion
--surface-report cad_out/mydet/surface_report.json
```

## Going further

`Detectors/CADSupport/validation/` holds the tools the development of this system is validated with:
an acceptance gate that scores converted parts against the OpenCascade oracle, an overlap census, a
round-trip report, and the closure test that the [ITS example](its-round-trip.md) follows. They are
not installed — run them from the source tree. `README.md` lists them all.
