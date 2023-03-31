<!-- doxy
\page refDetectorsUpgradesIT3 UpgradesIT3
/doxy -->

# ITS3
Upgraded version of the ITS that includes upgraded truly-cylindrical inner barrel.

# Run the full simulation
Provided O2 has been compiled with upgrades enabled, it is possible to simulate ITS3 geometry within the `o2-sim` executable.

## Simulation
Events can be simulated using the `o2-sim` workflow. To include ITS3 in the simulation, `IT3` module must be enabled via the `-m IT3` parameter. To include the beam pipe, the module `PIPE` must be enabled.

The following command can be used to generate heavy-ion collisions:
```bash
o2-sim -j 1  \
-n 10 -g pythia8hi --field ccdb \
--configKeyValues "Diamond.width[2]=6.;DescriptorInnerBarrelITS3.mVersion=ThreeLayers" \
--run 311935
```
In the previous command:
- `-j` is used to set the number of threads;
- `-n` is used to set the number of events to simulate;
- `-g` is used to set the event generator, in this case `pythia8hi`. To simulate pp collisions one can use `pythia8pp`.
- `--configKeyValues` is needed to set internal parameters of the workflow. Among these parameters, the geometry of the ITS3 inner barrel can be set.
- `--run` is needed to set the run number.

Currently, three different geometries of the ITS3 inner barrel are available:
- `ThreeLayersNoDeadZones`
- `ThreeLayers`
- `FourLayers`

The run number is needed to retrieve objects from the CCDB. There are specific ranges of run-numbers, according to the collision system and to the selected geometry if the ITS3 inner barrel:

- **pp** collisions:
  - 303901—303933 (`ThreeLayersNoDeadZones`)
  - 303934—303966 (`ThreeLayers`)
  - 303967—303999 (`FourLayers`)

- **Pb-Pb** collisions:
  - 311901—311933 (`ThreeLayersNoDeadZones`)
  - 311934—311966 (`ThreeLayers`)
  - 311967—311999 (`FourLayers`)


### Using external generators based on AliRoot
It is also possible to simulate heavy-ion collision using external generators based on AliRoot. In this case, it is necessary to load both O2 and AliROOT (the order is important):

```bash
alienv enter O2/latest AliRoot/latest
```

After that, the option `-g external` must be used and the file with the definition of the generator and the function to be used must be provided as parameters of the workflow:

```bash
o2-sim -j 1  \
-n 10 -g external \
--configKeyValues "Diamond.width[2]=6.;DescriptorInnerBarrelITS3.mVersion=ThreeLayers;GeneratorExternal.fileName=hijing.C;GeneratorExternal.funcName=hijing(5020, 0, 20)"
```
The file `hijing.C` can be found [here](https://alice.its.cern.ch/jira/browse/AOGM-246).

## Digitisation
The process consists of two steps. First, it is necessary to create the file `collision_context.root`:

```bash
o2-sim-digitizer-workflow --only-context -b \
--interactionRate 50000 --run \
--configKeyValues "HBFUtils.runNumber=311935
```
It is important to set the correct collision rate via `--interactionRate` and to set the correct run number with `--configKeyValues`.

To complete the digitisation, run the command:
```bash
o2-sim-digitizer-workflow -b --run --interactionRate 50000 \
--incontext collisioncontext.root \
--configKeyValues “HBFUtils.runNumber=311935”
```
As above, it is important to set the correct interaction rate and run number.

# Reconstruction

In this step, clustering, vertexing and tracking are performed. This is obtained with the `o2-its3-reco-workflow`:

```bash
o2-its3-reco-workflow --tracking-mode async -b --run \
-—configKeyValues "
HBFUtils.runNumber=311935;ITSCATrackerParam.trackletsPerClusterLimit=20;ITSCATrackerParam.cellsPerC
lusterLimit=20;ITSVertexerParam.lowMultXYcut2=0."
```

As above, it is important to provide the correct run number using `-—configKeyValues`, to retrieve the correct files from the CCDB. The other internal parameters provided `-—configKeyValues` via are specific to the cased here considered (Pb-Pb) and are inherited from ITS2.

> **_NOTE:_**  reconstruction for the `FourLayers` geometry is not implemented yet.
