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
o2-sim -j 8  \
-n 10 -g pythia8hi --field ccdb \
--configKeyValues "Diamond.width[2]=6.;" \
--run 311935
```

In the previous command:

- `-j` is used to set the number of threads;
- `-n` is used to set the number of events to simulate;
- `-g` is used to set the event generator, in this case `pythia8hi`. To simulate pp collisions one can use `pythia8pp`.
- `--run` is needed to set the run number.

The run number is needed to retrieve objects from the CCDB. There are specific ranges of run-numbers, according to the collision system and to the selected geometry if the ITS3 inner barrel:

- **pp** collisions:

  - 303901—303999

- **Pb-Pb** collisions:

  - 311901—311999

_Note: For now the same topology dictionary will be used for both collision-systems_

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

## Simulation

```bash
export IGNORE_VALIDITYCHECK_OF_CCDB_LOCALCACHE=1
export ALICEO2_CCDB_LOCALCACHE=$PWD/ccdb
mkdir -p ./ccdb/GLO/Config/Geometry
mkdir -p ./ccdb/GLO/Config/GeometryAligned
mkdir -p ./ccdb/IT3/Calib/ClusterDictionary
```

## pp

Simulate PIPE and ITS3

```bash
o2-sim -m
o2-sim -g pythia8pp -j10 -m PIPE IT3 --run 303901 -n1000
cp o2sim_geometry.root ${ALICEO2_CCDB_LOCALCACHE}/GLO/Config/Geometry/snapshot.root
o2-create-aligned-geometry-workflow -b --onlyDet IT3 --configKeyValues "HBFUtils.startTime=1547978230000" --condition-remap="file://${ALICEO2_CCDB_LOCALCACHE}=GLO/Config/Geometry"
cp o2sim_geometry-aligned.root ${ALICEO2_CCDB_LOCALCACHE}/GLO/Config/GeometryAligned/snapshot.root
cp its_GeometryTGeo.root ccdb/ITS/Config/Geometry/snapshot.root
```

## Digitization

```bash
o2-sim-digitizer-workflow -b --interactionRate 50000 --run --configKeyValues="HBFUtils.runNumber=303901;"
root -x -l ${ALIBUILD_WORK_DIR}/../O2/Detectors/Upgrades/ITS3/macros/test/CheckDigitsITS3.C++
```

## Clusterization w/o tracking

First we need to use the clusterizer but ignoring the default TopologyDictionary, we built our own.

```bash
o2-its3-reco-workflow -b --tracking-mode off \
    --configKeyValues "HBFUtils.runNumber=303901;" \
    --condition-remap="file://${ALICEO2_CCDB_LOCALCACHE}=IT3/Calib/ClusterDictionary" \
    --ignore-cluster-dictionary --run
```

### Creating the Topology Dictionary

```bash
root -x -l ${ALIBUILD_WORK_DIR}/../O2/Detectors/Upgrades/ITS3/macros/test/CreateDictionariesITS3.C++
cp IT3dictionary.root ${ALICEO2_CCDB_LOCALCACHE}/IT3/Calib/ClusterDictionary/snapshot.root
```

### Rerun Clusterization with new TopologyDictionary

```bash
o2-its3-reco-workflow -b --tracking-mode off \
    --configKeyValues "HBFUtils.runNumber=303901;" \
    --condition-remap="file://${ALICEO2_CCDB_LOCALCACHE}=IT3/Calib/ClusterDictionary" \
    --run
```

### Check Clusters

```bash
root -x -l '${ALIBUILD_WORK_DIR}/../O2/Detectors/Upgrades/ITS3/macros/test/CheckClustersITS3.C++("o2clus_it3.root", "o2sim_HitsIT3.root", "o2sim_geometry-aligned.root", "IT3dictionary.root")'
root -x -l '${ALIBUILD_WORK_DIR}/../O2/Detectors/Upgrades/ITS3/macros/test/CompareClustersAndDigits.C++("o2clus_it3.root", "it3digits.root","IT3dictionary.root", "o2sim_HitsIT3.root", "o2sim_geometry-aligned.root")'
root -x -l '${ALIBUILD_WORK_DIR}/../O2/Detectors/Upgrades/ITS3/macros/test/CheckClusterSize.C++("o2clus_it3.root", "o2sim_Kine.root", "IT3dictionary.root", false)'
```

## Clusterization with tracking

This repeats the clusterization steps and enables travcking.

```bash
o2-its3-reco-workflow -s --tracking-mode async --condition-remap="file://${ALICEO2_CCDB_LOCALCACHE}=IT3/Calib/ClusterDictionary,GLO/Config/GeometryAligned,ITS/Config/Geometry" \
    --configKeyValues "HBFUtils.runNumber=303901;"
root -x -l '${ALIBUILD_WORK_DIR}/../O2/Detectors/Upgrades/ITS3/macros/test/CheckTracksITS3.C++("o2trac_its3.root", "o2clus_it3.root", "o2sim_Kine.root", "o2sim_grp.root", "o2sim_geometry-aligned.root", false)'
```

# Create CCDB Objects

Create Full geometry + Aligned + GeometryTGeo

```bash
# Create Full Geometry
o2-sim -m PIPE IT3 TPC TRD TOF PHS CPV EMC HMP MFT MCH MID ZDC FT0 FV0 FDD CTP FOC TST --run 303901 -n0
cp o2sim_geometry.root ${ALICEO2_CCDB_LOCALCACHE}/GLO/Config/Geometry/snapshot.root
o2-create-aligned-geometry-workflow -b --configKeyValues "HBFUtils.startTime=1547978230000" --condition-remap="file://${ALICEO2_CCDB_LOCALCACHE}=GLO/Config/Geometry"
cp o2sim_geometry-aligned.root ${ALICEO2_CCDB_LOCALCACHE}/GLO/Config/GeometryAligned/snapshot.root
```
