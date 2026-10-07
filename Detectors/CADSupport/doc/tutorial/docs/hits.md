# Make it produce hits

Passive geometry answers questions about material budget. To ask whether your detector is actually
hit, and how often, some of its volumes need to be sensitive. This is the fastest route from a CAD
file to plottable hits, and it still needs no detector class and no rebuild — we simply change the
array name to `externalDetectors` and say which volumes should record:

`externalGeometry.json`

```json
{
  "externalDetectors": [
    {
      "name":  "EXCV",
      "title": "Excavator as a sensitive detector",
      "macro": "cad_out/excavator/geom.C",
      "anchor": "barrel",
      "detID": "TST",
      "sensitiveVolumes": ["Bucket"],
      "placement": { "translation": [21.01, -13.22, -19.66] }
    }
  ]
}
```

## Choosing the sensitive volumes

There are two ways of selecting them, and you may use either or both as long as at least one is
non-empty. `sensitiveVolumes` matches against TGeo volume names, and `sensitiveMedia` matches against
medium names — the latter being a convenient way to make every silicon part in an assembly sensitive
at once, however the parts happen to be named.

> [!WARNING]
> **Both match substrings, not whole names**
>
> This catches people out. On the excavator model, `"sensitiveVolumes": ["Bucket"]` selects **five**
> volumes rather than one — `Bucket`, `BucketLink1`, `BucketLink2`, `BucketCylinderInner` and
> `BucketCylinderOuter`. The startup log prints every volume it registered, so read it and tighten
> the string if that was not what you meant.

## Choosing a DetID

The `detID` field ties your detector to an existing O2 detector identity, which is what determines
where the hits are filed. Pick a slot no active built-in detector is using:

- `TST` is the general-purpose test slot, and the right default for a quick study.
- An upgrade study normally borrows the slot it stands in for — `TRK` for an ALICE 3 tracker, for
  instance — because it is semantically honest and keeps downstream tooling happy.

The hit branch keeps *your* module name rather than the borrowed one, so the configuration above
produces a branch called `EXCVHit`.

## Running it

```bash
o2-sim-serial -n 3 -g boxgen --seed 42 \
    --detectorList EXTCAD:detectorlist.json \
    --extGeomFile externalGeometry.json \
    --configKeyValues 'BoxGun.number=500;BoxGun.pdg=211;BoxGun.eta[0]=-1;BoxGun.eta[1]=1;BoxGun.prange[0]=2.0;BoxGun.prange[1]=5.0'
```

```text
External detector EXCV: 5 sensitive volume(s) selected
External detector EXCV: registered sensitive volume 'Bucket' (MC volID 8, sensor 0)
CREATING BRANCH EXCVHit
External detector EXCV EndOfEvent: 681 sensitive step(s) -> 94 hit(s)
External detector EXCV EndOfEvent: 402 sensitive step(s) -> 59 hit(s)
External detector EXCV EndOfEvent: 927 sensitive step(s) -> 124 hit(s)
```

The hits land in `o2sim.root`, one entry per event:

```bash
root -l -b -q -e 'TFile f("o2sim.root"); TTree *t=(TTree*)f.Get("o2sim");
                  t->Draw("EXCVHit@.size()");'
```

> [!NOTE]
> **Zero hits is usually aim, not breakage**
>
> The most common first result is `0 sensitive step(s)`, and the instinct is to suspect the
> conversion. Check where the particles are going first. The run above produces nothing at all at
> the default multiplicity of 10, simply because the excavator is a 40 cm object sitting 40 cm
> off-axis and is a small target. Raise the multiplicity or aim the gun. To rule out the geometry
> independently, shoot a ray through it in ROOT with `gGeoManager->FindNextBoundaryAndStep()` and
> print the volume names you cross — if they appear, navigation is fine and the problem is aim.

## Custom sensitive actions

With no further configuration, every sensitive volume records a charged-track entrance and exit hit in
the generic `o2::ext::Hit` format: position in and out, momentum, energy loss, PDG code and track
length. That is enough for occupancy, acceptance and material studies, which covers most first
questions.

When you need something else — a different hit definition, a cut applied at scoring time, extra
quantities — you can point at a macro returning an `o2::ext::ExternalDetector::SensitiveFcn`. It is
compiled at run time and can query `TVirtualMC::GetMC()` and call helpers such as `currentSensorID()`,
`currentTrackID()` and `addHit()`:

`externalGeometry.json · fragment`

```json
"sensitiveMedia": ["Silicon"],
"sensitiveMacro": "sensitive_action.macro",
"sensitiveFunction": "sensitiveAction()"
```

> [!NOTE]
> **A worked example that needs no CAD file**
>
> `run/SimExamples/External_Sensitive_Detectors` defines two artificial detectors entirely from data
> — one using the built-in action, one with a custom action compiled at run time — from hand-written
> macros that mimic converter output. Running `./run.sh` in that directory shows both hit branches
> appearing.
