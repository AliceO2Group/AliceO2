# Grow it into a real detector

> [!WARNING]
> **Not yet exercised end to end**
>
> Everything before this page has been run, with its output pasted from a real terminal. This route
> follows from how `ExternalDetector` and the built-in detectors are written, but no detector has
> yet been built this way. Treat it as a design rather than a recipe, and expect to debug it.

The external-detector route deliberately trades flexibility for speed: you get one generic hit type
and a borrowed `DetID`, and in exchange you get results the same afternoon. Once a study turns into a
real subdetector you will want your own hit class, your own digitisation and a `DetID` of your own —
and none of that requires giving up the CAD import. The generated geometry simply becomes one step
inside an ordinary O2 detector.

Three changes to a normal detector implementation are involved:

1. **Build the geometry from the macro instead of by hand.** Copy `geom.C` into your detector's
   simulation directory and call its builder hook from `ConstructGeometry()`, in place of the
   `new TGeoTube(...)` code you would otherwise write. Keep the `.bin` payloads beside it and install
   them with the detector's data files, since the macro resolves them relative to itself.
2. **Register your own sensitive volumes.** Call `AddSensitiveVolume()` for the volumes the macro
   created, using the names the converter derived from the CAD part names. Print them once from
   `geom.root` and pin them down in code, because a rename in CAD would otherwise quietly unregister a
   sensor.
3. **Write your own hits.** Implement `ProcessHits()` with your own hit class and your own `DetID`,
   exactly as any hand-written detector does. Nothing about the geometry's CAD origin constrains this.

Two things come back the moment you take this step, both of which the external-detector route cannot
offer: `initFieldTrackingParams()` called from your own `createMaterials()`, and
`SetSpecialPhysicsCuts()` reading a real `simcuts.dat` from your detector's data directory. That
closes the gap described under [Field and cuts](field-and-cuts.md).

The payoff is that re-running the converter after a CAD change regenerates only the geometry. Your
detector code stays untouched, which is the whole point of importing rather than transcribing.
