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
