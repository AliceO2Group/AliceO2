<!-- doxy
\page refDetectorsUpgradesIT3 UpgradesIT3
/doxy -->

# ITS3

Upgraded version of the ITS that includes upgraded truly-cylindrical inner barrel.
Provided O2 has been compiled with upgrades enabled (`ENABLE_UPGRADES=1 aliBuild build O2`), it is possible to simulate ITS3 geometry within the `o2-sim` executable.

Events can be simulated using the `o2-sim` workflow. To include ITS3 in the simulation, `IT3` module must be enabled via the `-m IT3` parameter. To include the beam pipe, the module `PIPE` must be enabled.

The run number is needed to retrieve objects from the CCDB. There are specific ranges of run-numbers, according to the collision system and to the selected geometry if the ITS3 inner barrel:

- **pp** collisions:

  - 303901—303999

- **Pb-Pb** collisions:

  - 311901—311999

_Note: For now the same topology dictionary will be used for both collision-systems_
_Last Update of file here (jira)[https://its.cern.ch/jira/browse/O2-4698]_

## Simulation

0. Optional

This just caches the ccdb object to reduce calls in case we are testing.

```bash
export IGNORE_VALIDITYCHECK_OF_CCDB_LOCALCACHE=1
export ALICEO2_CCDB_LOCALCACHE=${PWD}/ccdb
```

Simulate diamond

``` bash
# append to o2-sim
--configKeyValues="Diamond.width[2]=6.;""
```

### Local Tracking

1. Simulate

Simulate PIPE and ITS3

```bash
o2-sim -g pythia8pp -j10 -m PIPE IT3 --run 303901 -n1000
```

In the previous command:

- `-j` is used to set the number of threads;
- `-n` is used to set the number of events to simulate;
- `-g` is used to set the event generator, in this case `pythia8hi`. To simulate pp collisions one can use `pythia8pp`.
- `--run` is needed to set the run number.

2. Digitization

```bash
o2-sim-digitizer-workflow -b --interactionRate 50000 --run --configKeyValues="HBFUtils.runNumber=303901;" --onlyDet IT3
root -x -l ${ALIBUILD_WORK_DIR}/../O2/Detectors/Upgrades/ITS3/macros/test/CheckDigitsITS3.C++
```

3. Clusterization with tracking

```bash
o2-its3-reco-workflow -b --run --tracking-mode async --configKeyValues "HBFUtils.runNumber=303901;"
root -x -l ${ALIBUILD_WORK_DIR}/../O2/Detectors/Upgrades/ITS3/macros/test/CheckClustersITS3.C++
root -x -l ${ALIBUILD_WORK_DIR}/../O2/Detectors/Upgrades/ITS3/macros/test/CheckTracksITS3.C++
```

### Global Tracking

1. Simulate

Simulate all detectors but replacing ITS with IT3

```bash
o2-sim -g pythia8pp -j10 --detectorList ALICE2.1 --run 303901 -n20 -m IT3
```

## Creating CCDB Objects

### !TODO! Create Full geometry + Aligned + GeometryTGeo

```bash
# Create Full Geometry
o2-sim -g pythia8pp -j10 --detectorList ALICE2.1 --run 303901 -n0
cp o2sim_geometry.root ${ALICEO2_CCDB_LOCALCACHE}/GLO/Config/Geometry/snapshot.root
o2-create-aligned-geometry-workflow -b --configKeyValues "HBFUtils.startTime=1547978230000" --condition-remap="file://${ALICEO2_CCDB_LOCALCACHE}=GLO/Config/Geometry"
cp o2sim_geometry-aligned.root ${ALICEO2_CCDB_LOCALCACHE}/GLO/Config/GeometryAligned/snapshot.root
cp its_GeometryTGeo.root ${ALICEO2_CCDB_LOCALCACHE}/ITS/Config/Geometry/snapshot.root
```

### Regenerating the TopologyDictionary

1. Clusterization w/o tracking

First we need to use the clusterizer but ignoring the default TopologyDictionary, we built our own.

```bash
o2-its3-reco-workflow -b --tracking-mode off \
    --configKeyValues "HBFUtils.runNumber=303901;" \
    --ignore-cluster-dictionary --run
```

2. Creating the TopologyDictionary

```bash
root -x -l ${ALIBUILD_WORK_DIR}/../O2/Detectors/Upgrades/ITS3/macros/test/CreateDictionariesITS3.C++
cp IT3dictionary.root ${ALICEO2_CCDB_LOCALCACHE}/IT3/Calib/ClusterDictionary/snapshot.root
```

3. Rerun Clusterization with new TopologyDictionary

```bash
o2-its3-reco-workflow -b --tracking-mode off \
    --configKeyValues "HBFUtils.runNumber=303901;" \
    --condition-remap="file://${ALICEO2_CCDB_LOCALCACHE}=IT3/Calib/ClusterDictionary" \
    --run
```

4. Check Clusters

```bash
root -x -l '${ALIBUILD_WORK_DIR}/../O2/Detectors/Upgrades/ITS3/macros/test/CheckClustersITS3.C++("o2clus_its.root", "o2sim_HitsIT3.root", "o2sim_geometry-aligned.root", "IT3dictionary.root")'
root -x -l '${ALIBUILD_WORK_DIR}/../O2/Detectors/Upgrades/ITS3/macros/test/CompareClustersAndDigits.C++("o2clus_its.root", "it3digits.root","IT3dictionary.root", "o2sim_HitsIT3.root", "o2sim_geometry-aligned.root")'
root -x -l '${ALIBUILD_WORK_DIR}/../O2/Detectors/Upgrades/ITS3/macros/test/CheckClusterSize.C++("o2clus_its.root", "o2sim_Kine.root", "IT3dictionary.root", false)'
```

### Using external generators based on AliRoot

It is also possible to simulate heavy-ion collision using external generators based on AliRoot. In this case, it is necessary to load both O2 and AliROOT (the order is important):

```bash
alienv enter O2/latest AliRoot/latest
```

After that, the option `-g external` must be used and the file with the definition of the generator and the function to be used must be provided as parameters of the workflow:

```bash
o2-sim -j 1  \
-n 10 -g external \
--configKeyValues "Diamond.width[2]=6.;GeneratorExternal.fileName=hijing.C;GeneratorExternal.funcName=hijing(5020, 0, 20)"
```

The file `hijing.C` can be found [here](https://alice.its.cern.ch/jira/browse/AOGM-246).

### Disabling individual tiles
1. Create a file `input.txt` with a comma separated list of disabled tiles.
2. (optional) Run the macro `CreateITS3StaticDeadMap.C` and/or visualize with `CheckTileNumbering.C`
3. Move the ccdb object into `${ALICEO2_CCDB_LOCALCACHE}/IT3/Calib/DeadMap`, this is not optional since there is no default object uploaded
4. Run digitizer with `ITS3Params.useDeadChannelMap=true;`, e.g.:
``` bash
o2-sim-digitizer-workflow --configKeyValues="ITS3Params.useDeadChannelMap=true;"
```


### Alignment studies
#### Deform hits
1. Create misalignment parameters with `CreateMisalignmentITS3.C`
2. Visualize with `ShowCoefficients.C`
3. Run digitizer
``` bash
o2-sim-digitizer-workflow -b --configKeyValues="ITS3Params.applyMisalignmentHits=true;ITS3Params.misalignmentHitsParams=misparams.root"
${O2_ROOT}/bin/o2-its3-reco-workflow --tracking-mode async -b --run --condition-not-after 3385078236000 --shm-segment-size ${SHMSIZE:-50000000000} --configKeyValues "HBFUtils.orbitFirstSampled=0;HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0;HBFUtils.runNumber=311901;HBFUtils.startTime=1551418230000;GlobalParams.withITS3=true;ITSVertexerParam.lowMultBeamDistCut=0.;NameConf.mDirMatLUT=.."
```
