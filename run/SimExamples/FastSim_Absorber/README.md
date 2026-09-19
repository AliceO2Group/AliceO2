# Fast simulation of the front absorber

Replaces the detailed transport through the ALICE front absorber by a model, and
compares the result against a full simulation of the same events.

`run.sh` runs both: a reference `o2-sim` with PIPE and ABSO only, and the same
setup with the fast simulation switched on.

| file | |
|---|---|
| `run.sh` | the two simulations and the comparison |
| `countTracks.macro` | tracks per event of an `o2-sim` output |

## Switching it on

The feature does nothing unless a model is named:

```
--configKeyValues "G4.fastSimModels=toyAbsorber;G4.fastSimRegions=ABSO_AIR_ENVELOPE"
```

`G4.fastSimMinEnergy` (GeV, default 1) is the threshold below which the detailed
transport still runs, because a surrogate below it would be extrapolating and
the transport there is cheap anyway.

## What the model does

`toyAbsorber` is a placeholder. It returns the incident particle continuing in
its direction with the energy attenuated exponentially over the path through the
envelope — a real absorber turns one incident hadron into a shower, so the
numbers this produces are not physics. It is here so that the machinery can be
exercised end to end before a trained model exists.

A model implements one function, `sample()`, which maps the particle that
entered the region to the particles that leave it
(`Detectors/FastSim/include/FastSim/FastSimModel.h`). Everything around it —
measuring the distance to the envelope surface, killing the incident particle,
stacking what comes back, booking the energy difference as a deposit — is shared
and does not have to be reimplemented.

## Why the region is named `ABSO_AIR_ENVELOPE`

Regions are selected by tracking medium, which Geant4-VMC maps to that medium's
material, adding every volume of that material to the region. A volume can
therefore only be addressed on its own if its material is its own, which is why
`AFaM` — the mother volume of the whole absorber — carries a dedicated air
material (`Detectors/Passive/src/Absorber.cxx`). Selecting it means "the
absorber", with all of its daughters inside.

Selection by volume is not available: the VMC special cuts already root every
logical volume in a per-material region, and Geant4 allows a logical volume in
exactly one region.

## Status: the model does not fire on the muon-arm path

Measured on a real run, and it is a limitation of the region mechanism rather
than of the model.

`ABSO_AIR_ENVELOPE` selects a region containing `AFaM` **and nothing else**. The
VMC special cuts create one `G4Region` per material and make every logical volume
a root of its own, and Geant4 stops propagating a region down the tree at any
daughter that is itself a region root — so `AFaMgRing` is in `ABSO_MAGNESIUM$`,
`AFaGraphiteConeO` in `ABSO_CARBON0$`, and the absorber's mother region covers
none of them. On top of that, `AFaM`'s daughters touch its surface, so a track
entering the absorber lands straight in a daughter and never has `AFaM` as its
volume at all.

A 20 GeV muon fired into the muon-arm acceptance with `/tracking/verbose 1`
therefore steps through the absorber identically with and without the fast
simulation enabled:

```
   13     54.2   -0.238     -900  1.99e+04   0.0405      287       902   AFaMgRing Transportation
   14     55.5   -0.244     -920  1.99e+04     5.33       20       922 AFaGraphiteConeO Transportation
   15     56.7   -0.254     -940  1.99e+04     5.33     20.2       942 AFaGraphiteConeO muIoni
```

Only `muIoni`, `eIoni`, `Transportation` and `specialCutForElectron` appear; no
fast-simulation process does.

So a region in O2 can be "every volume of a given material" but not "this volume
and its daughters", and a surrogate for a whole module is not expressible through
this interface as it stands. The two ways forward are to name the absorber's
constituent materials and accept a per-piece envelope, or to change
`TG4RegionsManager` upstream so that the cuts regions do not claim every volume
as a root.

**Do not read the track counts below as a measurement of the model.** They come
from two runs whose random sequences diverge before the absorber is even reached
— visible in the beam pipe — so the difference is not attributable to the fast
simulation.

## Measured

Five pp minimum-bias events, Pythia8 and Geant4, `-m PIPE ABSO`, on one EPN node
against `O2PDPSuite/daily-20260819-0000-1`:

| | full | fast |
|---|---|---|
| tracks per event | 3544 | 3160 |
| transport real time | 29.8 s | 28.0 s |

geant4_vmc does report that the region resolved, which is necessary but, as
above, not sufficient:

```
fast simulation is ENABLED for regions 'ABSO_AIR_ENVELOPE'
fast simulation: registering model toyAbsorber above 1 GeV
Adding fast simulation model toyAbsorber to regions ABSO_AIR_ENVELOPE0$
```

where the last line is the tracking medium having resolved to its material, which
is the step that silently does nothing if the medium name is wrong.

**These numbers are not a performance result and not a measurement of the
model** — see the section above. They are recorded only to show what the example
currently produces.
