# Limits and pain points

The honest list. These are the things known to catch people today, roughly in order of how often they
do it. None is a reason not to use the system, but all of them are cheaper to read about here than to
rediscover in a result.

| What | Why it happens | What to do |
| --- | --- | --- |
| One `geom.C` per hooked thing | The macro exports a single builder hook, and that hook is what the JSON refers to. | Run the converter once per subsystem, into its own folder. They coexist happily in one JSON. |
| Media, cuts and field default to zero | A CAD file carries a material, never a medium, and the emitter uses a three-argument `TGeoMedium` which zeroes every parameter. | Pass `--in-field`. Accept transport defaults for step control, and treat production cuts as unset until you write a real detector. |
| The anchor volume must already exist | Placement is expressed inside the frame of an existing O2 volume. | Use `barrel` unless you have a reason not to, and remember it sits at cave `(0, -30, 0)`. |
| Free-form surfaces stay tessellated | Genuine B-spline *surfaces* are not supported by the exact tier at all. | Check the surface report. Recognition already recovers quadrics written as NURBS, which is the large majority of them. |
| Illegal overlaps in the CAD model | Engineering assemblies are not drawn as legal transport worlds, and parts routinely interpenetrate. | Read `CheckOverlaps`, then fix in CAD or clip the offending region. |
| Degenerate facets at coarse precision | `O2Tessellated` drops triangles that collapse to a line. | Treat it as a mesh-quality signal: lower `--mesh-prec`, or move the part onto an exact tier. |
| A surprisingly huge output directory | Meshing a metre-scale curved part at a fine chord tolerance. | Convert large models without `--mesh`, and never use the default `--mesh-prec` on something metre-sized. |
| `o2-sim` complains about a missing `externalModules` array | Cosmetic. The message is emitted even when your JSON correctly contains only `externalDetectors`. | Ignore it. |

## One rule that is not a preference

Run `--csg auto` conversions **strictly serially**. Parallel runs race each other and silently lose
shapes, which produces a geometry that looks complete and is not — the worst possible failure mode,
and the hardest to notice afterwards.

---

Deeper material lives in `Detectors/CADSupport`: `README.md` for the complete option reference,
`doc/reference/` for the exact-surface solid, its file format and the CSG pipeline, and
`doc/known-issues.md` for open defects.
