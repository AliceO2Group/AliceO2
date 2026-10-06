# Add passive geometry

With a macro in hand we can put the geometry into ALICE. The mechanism is deliberately data-driven:
two small JSON files, no code and no rebuild. We start with the simpler case — passive material such
as supports, cooling or cabling, which should scatter particles but does not record anything. That
goes into an `externalModules` array:

`externalGeometry.json`

```json
{
  "externalModules": [
    {
      "name":  "EXCV",
      "title": "Excavator support structure from CAD",
      "macro": "cad_out/excavator/geom.C",
      "anchor": "barrel",
      "placement": {
        "translation": [21.01, -13.22, -19.66],
        "rotation_deg": [0.0, 0.0, 0.0]
      }
    }
  ]
}
```

| field | meaning |
| --- | --- |
| `name` | a short tag for the module. It must also appear in the module list below, or the module is silently skipped. |
| `macro` | the path to the `geom.C` you produced. |
| `anchor` | a volume that already exists in the ALICE geometry. `barrel` is the usual choice, and it sits at cave coordinates `(0, -30, 0)`. |
| `placement` | translation and rotation **within the anchor's frame**, in centimetres and degrees. |

The second file is the module list, which is what actually switches the module on. The split exists
so that you can describe several modules in one geometry file and enable them individually:

`detectorlist.json`

```json
{ "EXTCAD": ["EXCV"] }
```

Then run the simulation, pointing at both:

```bash
o2-sim-serial -n 1 -g boxgen \
    --detectorList EXTCAD:detectorlist.json \
    --extGeomFile externalGeometry.json
```

```text
Configured external module 'EXCV' from macro 'cad_out/excavator/geom.C' anchored to volume 'barrel'
Activating EXCV module
Setting special cuts for passive module EXCV
```

Those three lines mean your CAD geometry is in the simulation and particles are being transported
through it. You can list as many modules in the same array as you like.

## Combine it with the built-in detectors

A custom `--detectorList` replaces the official list, it does not extend it. `o2-sim` takes the
module set from the one list you name, and `-m` may only select from that set. To simulate your CAD
module together with, say, the ALICE 3 detectors, put all of them in the same list:

`detectorlist.json`

```json
{ "EXTCAD": ["A3IP", "TRK", "FT3", "TF3", "EOS"] }
```

```bash
o2-sim-serial-run5 -n 1 -g boxgen \
    --detectorList EXTCAD:detectorlist.json \
    --extGeomFile externalGeometry.json
```

Here `EOS` is the `name` of the module in `externalGeometry.json`. Leave `-m` out, so that every
module in the list is active. If you pass `-m A3IP TRK` together with a list that does not contain
them, you get `Modules specified that are not present in detector list`.

- Copy the entries of the official list you want from `$O2_ROOT/share/config/o2simdefaultdetectorlist.json`.
- Use `o2-sim-serial-run5` (or `o2-sim-run5`) for ALICE 3 modules, and `o2-sim-serial` for Run 3 ones.
- Switch a module off by removing it from the list, or with `--skipModules`.
- The cave is always built, so `barrel` stays a valid anchor.
