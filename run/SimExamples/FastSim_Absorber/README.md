# Fast simulation of the front absorber

Replaces the detailed transport through the ALICE front absorber by a model, and
compares the result against a full simulation of the same events.

`run.sh` runs both: a reference `o2-sim` with PIPE and ABSO only, and the same
setup with the fast simulation switched on. Both use the same seed, so the two
simulations see the same primaries and the comparison is paired.

| file | |
|---|---|
| `run.sh` | the two simulations and the comparison |
| `countTracks.macro` | tracks per event of an `o2-sim` output |

## Switching it on

The feature does nothing unless a model is named:

```
--configKeyValues "G4.fastSimModels=toyAbsorber;G4.fastSimEnvelope=AFaM"
```

`G4.fastSimEnvelope` is the volume the model stands in for — here `AFaM`, the
mother of the whole absorber. `G4.fastSimMinEnergy` (GeV, default 1) is the
threshold below which the detailed transport still runs.

## What the model does

`toyAbsorber` is a placeholder: it returns the incident particle continuing in
its direction with the energy attenuated exponentially over its path through the
envelope. A real absorber turns one incident hadron into a shower, so these are
not physics numbers.

A model implements one function, `sample()`, which maps the particle that
entered the region to the particles that leave it
(`Detectors/FastSim/include/FastSim/FastSimModel.h`). The surrounding work —
measuring the distance to the envelope surface, killing the incident particle,
stacking what comes back, booking the energy difference as a deposit — is shared.

## How the envelope and the regions relate

Geant4 consults a fast simulation model through *regions*, and in O2 a region can
only ever be "every volume of a given material": the VMC special cuts make every
logical volume a root of its own material's region, and Geant4 stops propagating
a region at any such daughter. A region named after `AFaM` would therefore
contain `AFaM` alone, which tracks skip entirely because its daughters touch its
surface.

So the two jobs are separated. The regions are derived by walking the envelope's
subtree and collecting its media, which for `AFaM` finds fourteen:

```
fast simulation: model toyAbsorber covers 14 media found under 'AFaM'
Adding fast simulation model toyAbsorber to regions ABSO_AIR0$ ABSO_AIR_ENVELOPE0$
  ABSO_CONCRETE2$ ABSO_POLYETHYLEN2$ ABSO_CARBON0$ ABSO_CARBON2$ ABSO_MAGNESIUM$
  ABSO_Ni-W-Cu0$ ABSO_Ni-W-Cu2$ ABSO_LEAD0$ ABSO_LEAD2$ ABSO_STAINLESS STEEL0$
  ABSO_STAINLESS STEEL2$
```

What the model measures against is the envelope itself, read from the track's own
touchable. `ModelTrigger` also requires geometric containment in it, which is
what excludes the steel support cradle — it shares its material with the end
plate, so no selection by material could separate them.

## Measured

Five pp minimum-bias events, Pythia8 and Geant4, `-m PIPE ABSO`, same seed, one
EPN node against `O2PDPSuite/daily-20260819-0000-1`:

| | full | fast |
|---|---|---|
| tracks per event | 1547 | 1144 |
| transport real time | 14.9 s | 14.2 s |

A 20 GeV muon fired into the muon-arm acceptance with `/tracking/verbose 1`
shows what the model does to a single track:

```
   10     54.4   -0.137     -900  1.99e+04   0.0749      287       902   AFaMgRing Transportation
   11     54.4   -0.137     -900         0 1.99e+04  4.1e+03     5e+03   AFaMgRing G4FastSimulationManagerProcess
```

One step of 4.1 m across the whole absorber, in place of roughly twenty `muIoni`
steps through graphite, concrete and steel.

## What these numbers are not

**Not a performance result.** A quarter fewer tracks buys only five percent of
wall clock, because the tracks the absorber's shower contributes are cheap
low-energy ones and most of the CPU in this setup is spent elsewhere. And with
minimum-bias pp only the forward cone reaches the absorber at all. A CPU number
worth quoting needs a workload where the absorber is on the critical path — a
forward-biased generator, or the full detector where the muon arm is the point.

**Not a physics validation.** The toy returns one particle where a real absorber
returns a shower, so the track counts above say more about the placeholder than
about the absorber. Comparing the outgoing multiplicity and spectrum against a
full simulation is what a trained model has to pass, and that is the measurement
this example is scaffolding for.

Two further caveats worth knowing when reading any output of this: a fast step
does not call the sensitive detector, so its steps disappear from MCStepLogger
(harmless for a passive envelope, which has no hits); and secondaries the model
creates carry `TMCProcess` `kPNull`, the code geant4_vmc gives to everything it
has no VMC equivalent for, so they cannot be told apart from other tracks by
process alone.
