# Geometry step filtering

Stops transport of a track once it steps outside a hull wrapped around the
detector, and derives that hull from the geometry.

`run.sh` runs the whole chain: a reference simulation with the MCStepLogger, the
derivation of a hull from its geometry, a check that every recorded hit lies
inside that hull, and two MCReplay runs — one without cuts, one with the hull —
whose step and hit counts can be compared.

| file | |
|---|---|
| `makeKeepStepCylinders.macro` | derives the hull and writes a `keepStep()` macro |
| `checkKeepStepCylinders.macro` | verifies that no recorded hit lies outside a hull |
| `countHits.macro` | hits per detector of an `o2-sim` output |

The hull is applied through `SimCutParams.stepFilteringMacro`, in addition to
the built-in z/R cut, so it can only remove steps. A ready-made one is installed
at `$O2_ROOT/share/Detectors/gconfig/KeepStepCylinders.macro`; regenerate it
whenever the geometry changes.

`MCReplayParam.allowStopTrack=true` is what lets a replay act on the
`StopTrack()` calls the hull makes.

Measured on 10 pp minimum-bias events, Pythia8 and Geant4, with the hull
generated from the geometry of the run itself:

| module list | steps removed | hits lost |
|---|---|---|
| default minus ZDC | about 5 % | 0.18 % |
| default | about 2 % | 0.20 % |

Hits exclude FT0, which draws a random number while creating hits and does not
reproduce under replay.

**Read the replay's step accounting with care.** `MCReplayParam.allowStopTrack`
makes a replay honour *every* `StopTrack()` of a replayed step, not only the
ones the hull causes. FT0's photocathode efficiency and HMPID's Fresnel loss
stop tracks on a random draw, and those draws do not repeat on replay, so a
replay with `allowStopTrack=true` and a `keepStep()` that returns `true`
already reports about 8 % of steps skipped with no cut in play; running it with
`--skipModules ZDC FT0 HMP` reports exactly zero. Use the replay only as a
difference against such a no-op macro, and expect that difference to still read
a little high, because a secondary is dropped whenever its parent was skipped
before it was born, which approximates rather than reproduces what Geant4 does.

The number to quote comes from a real Geant4 pair with `SimCutParams.trackSeed`
on, summing the per-event `This event/chunk did N steps`. Without per-track
seeding the two runs diverge into different physics and the comparison is
worthless: on one sample it returned 0.19 % where the truth was 4.4 %.

Do not expect the step reduction to become a CPU reduction. Without ZDC,
measured as user time over sequential runs on an idle machine with the same
generator seed:

| run | Geant4 steps | CPU |
|---|---|---|
| no macro | 9 855 230 | 88.7 s |
| macro returning `true` | 9 855 230 | 91.1 s |
| the hull | 9 380 571 | 90.4 s |

Two things follow. Reaching a cling-compiled `keepStep()` through a
`std::function` costs about 235 ns per step on its own -- the middle row removes
no steps at all -- while the cylinder scan itself is negligible next to it. And
the steps the hull removes are cheap ones: the ~5 % of steps it removes are
worth only about 0.7 % of the runtime, so even at zero hook overhead the gain
here would be small. The macro is compiled once at start-up, not per step.

Evaluating a cylinder set natively rather than through a macro would remove the
overhead; whether a geometry hull is worth it without ZDC is a separate
question.

Background: A. Swain, *Geometric Hyperparameter Optimisation of ALICE Monte
Carlo Transport Simulations*, CERN-STUDENTS-Note-2023-164, and B. Völkel,
*Geometry cuts in MC transport*, WP12/13 meeting, 11.10.2023
(https://indico.cern.ch/event/1334852/contributions/5620122/).
