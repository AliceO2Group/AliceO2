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

## Measured

Five pp minimum-bias events, Pythia8 and Geant4, `-m PIPE ABSO`, on one EPN node
against `O2PDPSuite/daily-20260819-0000-1`:

| | full | fast |
|---|---|---|
| tracks per event | 3544 | 3160 |
| transport real time | 29.8 s | 28.0 s |

The model is demonstrably applied — geant4_vmc reports

```
fast simulation is ENABLED for regions 'ABSO_AIR_ENVELOPE'
fast simulation: registering model toyAbsorber above 1 GeV
Adding fast simulation model toyAbsorber to regions ABSO_AIR_ENVELOPE0$
```

where the last line is the tracking medium having resolved to its material, which
is the step that silently does nothing if the medium name is wrong.

**But the saving here is small, and the example should not be read as a
performance claim.** The absorber sits at z = −90 to −501 cm and only the forward
cone of a minimum-bias pp event ever enters it, so most of the transport this
setup does is not in the region at all; and `fastSimMinEnergy=1` leaves
everything below 1 GeV to the detailed transport. An honest CPU number needs a
workload where the absorber is on the critical path — a forward-biased generator,
or the full detector where the muon arm is what the absorber exists to protect.
