# External sensitive detectors

A minimal example showing how to add **sensitive** detectors to `o2-sim` without compiling
anything into O2. Two artificial detectors are injected purely from data:

| name    | geometry (runtime macro)      | DetID slot | sensitive action                |
|---------|-------------------------------|------------|---------------------------------|
| `ACYL`  | `geometry_innerCylinder.macro`| `ITS`      | built-in entrance/exit action   |
| `BDISK` | `geometry_outerDisk.macro`    | `TST`      | custom action `sensitive_action.macro` |

Both are instances of the single compiled `o2::ext::ExternalDetector` class and produce the
single generic hit type `o2::ext::Hit`. They differ only in their data: geometry, the DetID
slot they occupy, and (optionally) a sensitive-action macro.

## Run

```bash
./run.sh
```

This transports a few `boxgen` events and prints the per-detector hit counts, e.g.

```
External sensitive detector hits:
  ACYLHit                : ~430 hits over 5 events, mean radius 20.0 cm  [o2sim_HitsITS.root]
  BDISKHit               : ~120 hits over 5 events, mean radius ...  cm  [o2sim_HitsTST.root]
```

## How it works

* **Geometry** — each `"macro"` is a ROOT macro exporting `get_builder_hook_unchecked()`, the
  same symbol `O2_CADtoTGeo.py` emits when converting CAD (STEP) files to TGeo. The macros here
  are hand-written so the example is self-contained (no CAD binaries). The volumes whose names
  match `"sensitiveVolumes"` are registered as sensitive.

* **Sensitive action** — `ACYL` has none, so it uses the built-in action that records an
  entrance/exit hit per track. `BDISK` points `"sensitiveMacro"` at a macro whose
  `sensitiveAction()` returns the per-step callable; it is JIT-compiled at runtime via
  `o2::conf::GetFromMacro` (the same mechanism as the generator / stepping hooks — no
  recompilation, no ACLIC). The callable has the full `TVirtualMC` singleton and a few helpers
  (`currentSensorID()`, `currentTrackID()`, `addHit()`).

* **Persistence** — each detector is tied to a free **DetID** slot (`ITS`, `TST` here, chosen
  because no real detector of that name is active). The DetID is the scarce resource: it fixes
  the hit-file name (`o2sim_Hits<DetID>.root`) and lets the hit merger instantiate a matching
  receiver, so hits are written in parallel multi-worker mode just like for any built-in
  detector. The branch keeps the detector's own name (`ACYLHit`, `BDISKHit`).

## Adding more detectors

Append entries to `externalDetectors.json` (each on a different free DetID slot) and list their
names in `detectorlist.json`. The mechanism is fully data-driven; nothing needs to be rebuilt.

See also `Detectors/External` and `scripts/geometry/O2_CADtoTGeo.py`.
