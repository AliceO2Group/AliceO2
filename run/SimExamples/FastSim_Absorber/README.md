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

## What is not measured here

CPU. The step and track counts show that the transport is being replaced; a
timing number needs a realistic workload rather than five events of PIPE+ABSO.
