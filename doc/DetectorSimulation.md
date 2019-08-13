\page refdocDetectorSimulation Detector Simulation

# Detector simulation documentation

The present document collects information about the ALICE detector simulation executable and digitization procedure used in LHC Run3.

## Overview

Detector simulation, the simulation of detector response from virtual particle events, consists of essentialy 2 parts:
  a) the generation of simple (energy deposit) traces in the detector due to the passage of particles and the interaction with the detector material.
  b) the conversion of those traces into (electronic) signals in the detector readout (usually called digitization).
 
The first part is handled by the `o2-sim` executable. The second part is handled in the `o2-sim-digitizer-workflow`.
 
# Documentation of `o2-sim`

The purpose of the `o2-sim` executable is to simulate the passage of particles emerging from a collision inside the detector and to obtain their effect in terms of energy deposits (called hits) which could be converted into detectable signals.

## Command overview
* **Basic help:** Help on command line options can be obtained with `o2-sim --help`
* **Typical example:** A typical (exemplary) invocation is of the form 

    ```o2-sim -n 10 -g pythia8 -e TGeant4 -j 2 --skipModules ZDC,PHS``` 

    which would launch a simulation for 10 pythia8 events on the whole ALICE detector but ZDC and PHOS, using Geant4 on 2 worker processes.
* **Generated output**: The simulation creates at least the following files:
     
     
        | file              | description      |
        | ----------------- | ---------------- |
        | `o2sim.root`      | contains kinematics and hits |
        | `O2geometry.root` | contains the ROOT geometry created for simulation |
        | `o2sim_grp.root`  | the grp parameters |


## Main configuration Options

control of verbosity

## Configuration via Parameters

## Help on available generators

## Control via environment variables
`o2-sim` is sensitive to the following environment variables:

**ALICE_O2SIM_DUMPLOG**
**ALICE_O2SIM_USESHM**

[Describe in detail the environment variables]

## Data layout
[Add something on data layout of hits file]

## F.A.Q.
You may contribute to the documentation by asking a question

#### 1. **How can I interface an event generator from ALIROOT**?
In order to access event generators from ALIROOT, such as `THijing` or `TPyhtia6`, you may use the `-g extgen` command line option followed by a ROOT macro setting up the event 
generator. Examples thereof are available in the installation directory `$O2_ROOT/share/Generators/external`.

For example, in order to simulate with 10 Hijing events, the following command can be run:
```
o2-sim -n 10 -g extgen --extGenFile $O2_ROOT/share/Generators/external/pythia6.C
```
Macro arguments can be passed via
`--extGenFunc pythia6(14000., "pythia.settings")`.

Users may write there own macros in order to customize to their needs.

#### 2. **How can I run on a subset of geometry modules**?
Use the `--modules` or `-m` command line option. Example: `o2-sim -m PIPE ITS TPC`
will run the simulation on a geometry/material consinsting of PIPE, ITS, TPC.

#### 3. **How can I run with exactly the same events as used in an AliRoot simulation?**

One may perform any arbitrary simulation with AliRoot and reuse the kinematics information in form of `Kinemtatics.root`
produced. The file contains primary and possibly secondary particles (added by transportation). 
When the file is passed to `o2sim`, the primary particles my be used as the initial event. 
Use the **`-g extkin`** command line option:
```
o2-sim -g extkin --extKinFile Kinematics.root ...
```

#### 4. **How can I generate events (signal) using the vertex position of already-generated (background) events?**

This process might be called embedding, where one wants to merge two events generated independenly. For that to be physically correct, both events have to originate from the same interaction vertex.
Assuming that your already-generated (background) events are stored in the `o2sim.background.root` file, you can force the interaction vertex for the generation of a new set of events to be the same as the one in the background with the following command line option:

```
o2-sim --embedIntoFile o2sim.background.root
```

Background events are sampled one-by-one until all events have been used. At that point the events start to be reused.

#### 5. **How can I obtain detailed stepping information?**
Run the simulation (currently only supported in combination with `o2-sim-serial`) with a preloaded library:
```
MCSTEPLOG_TTREE=1 LD_PRELOAD=$O2_ROOT/lib/libMCStepLogger.so o2-sim-serial -j 1 -n 10
```
This will produce a file `MCStepLoggerOutput.root` containing detailed information about steps and processes (where, what, ...). The file can be analysed using a special analysis framework. See https://github.com/AliceO2Group/AliceO2/blob/dev/Utilities/MCStepLogger/README.md for more documentation.

## Development

# Documentation of `o2-sim-digitizer-workflow`
