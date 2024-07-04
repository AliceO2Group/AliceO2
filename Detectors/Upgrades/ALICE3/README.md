<!-- doxy
\page refDetectorsUpgradesALICE3 ALICE3 Full Simulation
/doxy -->

# ALICE 3 full simulation

## Simulation
### Simulation software for Run 5
Current simulation approach for ALICE 3 is a specification of what it is used in the main O2 simulation package.
It shares the same structure and possibly inherits all the features in place already.
For all of the features that are not really Run-5 specific please refer to the official [O2 Simulation Documentation](https://aliceo2group.github.io/simulation/docs/).

### Rationale for the "Upgrades" codebase
Run 5 code for the detectors is stored in the `Detectors/Upgrades/ALICE3` directory and its suboflder strucure mimicks the same we have in `Detectors`.
For specific data formats the plan is to keep the same approach in `DataFormats/Upgrades`.

### Available modules
Run 5 executable is called `o2-sim-run5` and accepts the same syntax as `o2-sim`.
There is also the serial vertsion `o2-sim-serial-run5` that is the serial implementation.
The specific modules for Run 5 are enabled by passing their their IDs to the `-m` parameter of the `o2-sim`.
A list of the available DetIDs is reproted in the table below:

| Detector ID | Detector description             |
|-------------|----------------------------------|
| `A3IP`      | Beam pipe                        |
| `TRK`       | Barrel Tracker                   |
| `TF3`       | Time Of Flight detectors         |
| `FT3`       | Forward endcaps                  |
| `RCH`       | Ring Imaging Cherenkov detectors |
| `ECL`       | Electromagnetic Calorimeter      |
| `MI3`       | Muon Identification              |
| `FCT`       | Forward Conversion Tracker       |
| `A3ABSO`    | Absorber                         |
| `A3MAG`     | Magnet                           |

Names are arbitrarily chosen and are such as to be orthogonal to any Run 3+ other DetID.
The detector IDs of sensitive modules are mapped to the corresponding `o2::detector::DetID` class definition for convenience, so to be consistent with output names of the hit files.

### Use the ALICE 3 magnetic field
By definition the `o2-sim-run5` will use the Run 3 magnetic field configurations.
Field description can be overridden with a custom macro with a path exported in the `ALICE3_MAGFIELD_MACRO` environment variable.
The env var `ALICE3_SIM_FIELD` needs also to be set to `ON`.
Example:

```bash
export ALICE3_SIM_FIELD=ON
export ALICE3_MAGFIELD_MACRO=../ALICE3Field.C
```

An exampling macro for a custom magnetic field is stored in `Detectors/Upgrades/macros/ALICE3Field.C`.

### Run a simple simulation for run 5
The simplest command to be run to test the simulation is working is:

```bash
o2-sim-run5 -n 0
```
This will not produce any events but will spin the machinery and will produce the `o2sim_geometry.root` file with the whole geometry description.
To enable a specific set of modules, e.g. the beampipe and the TOFs one can specify which modules to enable, e.g.:

```bash
o2-sim-run5 -n 10 -m A3IP TF3
```
### Output of the simulation
The simulation will produce a `o2sim_Hits<DetID>.root` file with a tree with the hits related to that detector.
Currently, hits are produced for: `TRK`, `FT3`, and `TF3`.
More detectors will be included.

## Reconstruction
WIP

## Analysis
WIP