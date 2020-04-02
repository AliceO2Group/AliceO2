<!-- doxy
\page refdocDetectorSimulation Detector Simulation
/doxy -->

# Detector simulation documentation

The present document collects information about the ALICE detector simulation executable and digitization procedure used in LHC Run3.

## Overview

Detector simulation, the simulation of detector response from virtual particle events, consists of essentialy 2 parts:
  a) the generation of simple (energy deposit) traces in the detector due to the passage of particles and the interaction with the detector material.
  b) the conversion of those traces into (electronic) signals in the detector readout (usually called digitization).
 
The first part is handled by the `o2-sim` executable. The second part is handled in the `o2-sim-digitizer-workflow`.

## Key new features with respect to AliRoot

The Run3 simulation offers the following features

  - **distributed system based on FairMQ** that is splitting event generation, particle transport and IO into separate asyncronous components that can be deployed on different machines
  - **sub-event parallelism** making it possible to transport a single big event in a short time and to reduce memory consumption
  - **parallelism** independent on transport engine
  - **configuration via pre-defined parameter classes and ini/text files**
  - **clear separation of transport and digitization** - each phase can be run fully independently


# Documentation of transport simulation

The purpose of the `o2-sim` executable is to simulate the passage of particles emerging from a collision inside the detector and to obtain their effect in terms of energy deposits (called hits) which could be converted into detectable signals. It is the driver executable which will spawn a topology of sub-processes that interact via messages in a distributed system.

## Usage overview
* **Quick start example:** A typical (exemplary) invocation is of the form 

    ```o2-sim -n 10 -g pythia8 -e TGeant4 -j 2 --skipModules ZDC,PHS``` 

    which would launch a simulation for 10 pythia8 events on the whole ALICE detector but ZDC and PHOS, using Geant4 on 2 parallel worker processes.
* **Generated output**: The simulation creates the following output files:
     
| File                  | Description                                                                            |
| --------------------- | -------------------------------------------------------------------------------------- |
| `o2sim_Kine.root`     | contains kinematics information (primaries and secondaries) and event meta information |
| `o2sim_geometry.root` | contains the ROOT geometry created for simulation                                      |
| `o2sim_grp.root`      | special global run parameters (grp) such as field                                      |
| `o2sim_XXXHits.root`  | hit file for each participating active detector XXX                                    |
| `o2sim_configuration.ini` | summary of parameter values with which the simulation was done                     |


* **Main command line options**: The following major options are available (incomplete):

| Option                | Description                                                                            |
| --------------------- | -------------------------------------------------------------------------------------- |
| -h,--help     | Prints the list of possible command line options and their default values.           |
| -n,--number | The number of events to simulate.                                                       |
| -g,--generator | name of a predefined generator template to use (such as pythia8, pythia8hi). Configuration of generations is explained in a dedicated section. |
| -e,--engine | Select the VMC transport engine (TGeant4, TGeant3).                                     |
| -m,--modules | List of modules/geometries to include (default is ALL); example -m PIPE ITS TPC       |
| -j,--nworkers | Number of parallel simulation engine workers (default is half the number of hyperthread CPU cores) |
| --chunkSize | Size of a sub-event. This determines how many primary tracks will be sent to a simulation worker to process. |
| --skipModules | List of modules to skip / not to include (precedence over -m) |
| --configFile   | A `.ini` file containing a list of (non-default) parameters to configure the simulation run. See section on configurable parameters for more details.  |
| --configKeyValues | Like `--configFile` but allowing to set parameters on the command line as a string sequence. Example `--configKeyValues "Stack.pruneKine=false"`. Takes precedence over `--configFile`. Parameters need to be known ConfigurableParams. |
| --seed   | The initial seed to (all) random number instances. Default is -1 which leads to random behaviour. |


## Configuration via Parameters

## Help on available generators

Below some notes on example generators along with some usage info.

* **Fwmugen**

fwmugen is a lightweight and simple “box” generator for forward muons (1 muon / event)

```
o2-sim -m MFT -e TGeant3 -g fwmugen -n 10
```

* **BoxGen**

```
o2-sim -m PIPE ITS MFT -e TGeant3 -g boxgen -n 10 --configKeyValues 'BoxGun.pdg=13 ; BoxGun.eta[0]=-3.6 ; BoxGun.eta[1]=-2.45; BoxGun.number=100'
```

* **PYTHIA 8**

Configures pythia8 for min.bias pp collisions at 14 TeV

```
o2-sim -m PIPE ITS MFT -g pythia8 -n 50
```

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

For example, in order to simulate with 10 Pythia6 events, the following command can be run:
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

#### 6. **How can I add a trigger to the event generator?**
All event generator interfaces that comply with the `o2::eventgen::Generator` protocol can be triggered.
A basic 'particle trigger' is implemented in the `o2::eventgen` core and allows the user to define a trigger particle.
The definitions of the trigger particle can be expressed via command line arguments
```
o2-sim -g pythia8 -t particle --configKeyValues "TriggerParticle.pdg=333;TriggerParticle.ptMin=5.;TriggerParticle.yMin=-0.5;TriggerParticle.yMax=0.5"
```

Custom triggers can also be constructed by the user to provide unlimited flexibility in the trigger needs.
An external trigger function can be specified via command line arguments
```
o2-sim -g pythia8 -t external --extTrgFile path_to_trigger_macro.C --extTrgFunc "the_function(some, parameters)"
```
The function must comply with a simple protocol and return a lambda function defined as follows
```
o2::eventgen::Trigger the_function()
{
  return [](const std::vector<TParticle>& particles) -> bool {
    return true; // triggered
  }
}
```
Within the lambda function the user receives the stack of generated particles and can inspect it to define a trigger at will.
The trigger is fired when the lambda function returns `true` and the simulation of the current event is subsequently started.

To allow users to define triggers that go beyond the particle stack generated by the event generator, another functionality is added.
This allows the user to go deep into the core of the event generator, whenever this is possible.
For this reason, this is called a 'DeepTrigger'. A 'DeepTrigger' is attached to the simulation in the same way as a normal trigger
```
o2-sim -g pythia8 -t external --extTrgFile path_to_deep_trigger_macro.C --extTrgFunc "the_deep_function(some, parameters)"
```
In this case the function must comply with a similar, but different protocol than before and return a lambda function defined as follows
``` o2::eventgen::DeepTrigger the_deep_function()
{
  return [mpiMin](void* interface, std::string name) -> bool {
    return true;
  };
}
```
Notice that in this case the user is presented with a pointer to the event-generator interface and a string that defines its name.
For the sake of generality, a `void*` has to be used in order to pass any possible types of event-generators, that are
normally othogonal one to another. The name encodes a string to identify what generator has been passed and perform the correct cast to use it.



### Deep triggers
Deep triggers is just a name to a new functionality that allows the user to define custom functions that will have a direct handle on the event generator interface. The functionality follows the schema of the previous point, with the user providing a custom lambda function that will receive from the framework a pointer to the internal event-generator interface object (i.e. for Pythia8, a pointer to the Pythia object) and a tagname to identify the interface. This functionality might be useful to users who want to provide triggers based on information beyond the stack of the generated particles, based on more internal counters/information in the event generator machinery.

Here is an example of a deep trigger implementation in Pythia8.

```
//   usage: o2sim --trigger external --extTrgFile trigger_mpi.C
// options:                          --extTrgFunc "trigger_mpi()"

#include "Generators/Trigger.h"
#include "Pythia8/Pythia.h"

o2::eventgen::DeepTrigger
  trigger_mpi(int mpiMin = 5)
{
  return [mpiMin](void* interface, std::string name) -> bool {
    if (!name.compare("pythia8")) {
      auto py8 = reinterpret_cast<Pythia8::Pythia*>(interface);
      return py8->info.nMPI() >= mpiMin;
    }
    LOG(FATAL) << "Cannot define MPI for generator interface \'" << name << "\'";
    return false;
  };
}
```


### Pythia8 UserHooks
Pythia8 machinery allows the user to hook some code at various stages of the event-generation process. For details, please look at Pyhia8 manual.
http://home.thep.lu.se/~torbjorn/pythia82html/UserHooks.html

The interface is provided via a configuration macro, where the user will have to define a custom UserHooks according to the protocol defined by Pythia8. The macro will also have to provide a function to retrieve the pointer to the created custom UserHooks object.

This functionality might be of use for users who want to be able to steer the event-generation process from very deep inside the internal routines and want to veto some specific processes based on analysis of the status of Pythia8 at the various stages, i.e. veto events that do not have charm partons, before hadronisation of partons. This can save time in the event generation process as many steps can be skipped already at early time.

An example of a configuration macro is this one

```
//   usage: o2sim -g pythia8 --configKeyValue "Pythia8.hooksFileName=pythia8_userhooks_charm.C"

#include "Generators/Trigger.h"
#include "Pythia8/Pythia.h"

class UserHooksCharm : public Pythia8::UserHooks
{
 public:
  UserHooksCharm() = default;
  ~UserHooksCharm() = default;
  bool canVetoPartonLevel() override { return true; };
  bool doVetoPartonLevel(const Pythia8::Event& event) override
  {
    for (int ipa = 0; ipa < event.size(); ++ipa) {
      if (abs(event[ipa].id()) != 4)
        continue;
      if (fabs(event[ipa].y()) > 1.5)
        continue;
      return false;
    }
    return true;
  };
};

Pythia8::UserHooks*
  pythia8_userhooks_charm()
{
  return new UserHooksCharm();
```


### Pythia6 interface
A new Pythia6 interface is provided via GeneratorPythia6. This complies with the o2::eventgen::Generator protocol, and hence the user is allowed to use all the trigger functionalities. The class can also be used for DeepTriggers as this modified macro shows.

```
//   usage: o2sim --trigger external --extTrgFile trigger_mpi.C
// options:                          --extTrgFunc "trigger_mpi()"

#include "Generators/Trigger.h"
#include "Pythia8/Pythia.h"
#include "TPythia6.h"

o2::eventgen::DeepTrigger
  trigger_mpi(int mpiMin = 15)
{
  return [mpiMin](void* interface, std::string name) -> bool {
    int nMPI = 0;
    if (!name.compare("pythia8")) {
      auto py8 = reinterpret_cast<Pythia8::Pythia*>(interface);
      nMPI = py8->info.nMPI();
    }
    else if (!name.compare("pythia6")) {
      auto py6 = reinterpret_cast<TPythia6*>(interface);
      nMPI = py6->GetMSTI(31);
    }
    else
      LOG(FATAL) << "Cannot define MPI for generator interface \'" << name << "\'";
    return nMPI >= mpiMin;
  };
}
```

## Development

# Documentation of the digitization step 

Digitization, the transformation of hits produced in the transport simulation to electronics detector output, is steered by the `o2-sim-digitizer-workflow` executable.
