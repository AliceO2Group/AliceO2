# The geom.C file

Everything the converter does ends up in one ROOT macro, and it is the artefact worth caring about.
It exports two functions: `get_builder_hook_unchecked()`, which is what `o2-sim` calls when it loads
your geometry, and `build_and_export()`, which you already used to look at the model on its own.

Alongside it, the output folder holds the binary payloads the macro reads — `facets_*.bin` for meshed
parts, `surfaces_*.bin` for exact ones and `flatcsg_*.bin` for flat CSG solids — plus
`csg_report.json`, which records what each part became and why.

> [!WARNING]
> **The macro and its binaries travel together**
>
> `geom.C` loads those `.bin` files **relative to its own location**. Move or copy the macro without
> the rest of its folder and it will build an empty geometry without complaining. Always move the
> directory.

`build_and_export()` runs `CheckOverlaps` only when asked, because on large models it is slow:

```bash
root -l -b -q -e '.L geom.C' -e 'build_and_export("geom.root", true, true);'
```

```text
Info in <TGeoManager::CloseGeometry>: 14 nodes/ 14 volume UID's in geom
Info in <TGeoNodeMatrix::CheckOverlaps>: Checking overlaps for Assembly and daughters within 0.1
Info in <TGeoNodeMatrix::CheckOverlaps>: Number of illegal overlaps/extrusions : 0
```

Finally, a structural point that shapes how you organise your work: each converted directory holds
exactly one `geom.C`, and each `geom.C` describes one thing you hook into the simulation. If your
study involves three CAD subsystems, you run the converter three times into three folders. They
coexist without trouble, because the loader compiles each macro into its own namespace at run time,
so the identical function names inside them never collide.
