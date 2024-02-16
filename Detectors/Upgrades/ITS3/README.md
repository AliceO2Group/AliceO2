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
--configKeyValues "Diamond.width[2]=6.;DescriptorInnerBarrelITS3.mVersion=ThreeLayers" \
--run 311935
```
In the previous command:
- `-j` is used to set the number of threads;
- `-n` is used to set the number of events to simulate;
- `-g` is used to set the event generator, in this case `pythia8hi`. To simulate pp collisions one can use `pythia8pp`.
- `--configKeyValues` is needed to set internal parameters of the workflow. Among these parameters, the geometry of the ITS3 inner barrel can be set:
  - `DescriptorInnerBarrel.mVersion` is the geometry version of the ITS3 inner barrel (`ThreeLayersNoDeadZones`, `ThreeLayers`, or `FourLayers`)
  - `DescriptorInnerBarrel.mRadii` is a 4-element vector with the radii of the ITS3 layers
  - `DescriptorInnerBarrel.mLength` is the length of the ITS3 in the Z direction
  - `DescriptorInnerBarrel.mGapY` is a 4-element vector with the values of gap between the two hemicylinders, described by a translation of the hemicylinders in the vertical direction
  - `DescriptorInnerBarrel.mGapPhi` is a 4-element vector with the values of gap of azimuthal angle between the two hemicylinders, described by the maximum distance between the two hemicylinders. Differently from `mGapY`, in this case there no shift in the vertical direction, but a smaller coverage in the azimuthal angle of the half layers.
  - `DescriptorInnerBarrel.mGapXDirection4thLayer` is the gap in the horizontal direction for the fourth layer, analogous to the `mGapY`.
  - `SuperAlpideParams.mDetectorThickness` is the thickness of the chip
- `--run` is needed to set the run number.

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

In addition, some parameters related to the segmentation of the chips can be set with the `--configKeyValues` argument:
- `SuperAlpideParams.mPitchCol` is the pitch of the column (Z direction)
- `SuperAlpideParams.mPitchRow` is the pitch of the row (r$\phi$ direction)
- `SuperAlpideParams.mDetectorThickness` is the thickness of the chip

If some parameters of the geometry changed in the simulation, they should be set similarly for the digitisation. In particular, the `mVersion`, `mRadii`, and `mLength` must be set if different from the default ones.

# Reconstruction

In this step, clustering, vertexing and tracking are performed. This is obtained with the `o2-its3-reco-workflow`:

```bash
o2-its3-reco-workflow --tracking-mode async -b --run \
-—configKeyValues "
HBFUtils.runNumber=311935;ITSCATrackerParam.trackletsPerClusterLimit=20;ITSCATrackerParam.cellsPerClusterLimit=20;ITSVertexerParam.lowMultXYcut2=0."
```

As above, it is important to provide the correct run number using `-—configKeyValues`, to retrieve the correct files from the CCDB. The other internal parameters for the vertexer and the tracker are provided `-—configKeyValues` via are specific to the cased here considered (Pb-Pb) and are inherited from ITS2.

If the `FourLayers` geometry was used in the simulation, it should be set also for the reconstruction to set properly the tracker to work with the additional layer. If something else of the geometry was set differently (`mRadii`, `mLength`, `mGapY`, `mGapPhi`, `mGapXDirection4thLayer`, or `mDetectorThickness`), it is necessary to remap the file with the geometry to replace the one on the CCDB, which would be different. This can be done by copying the `o2sim_geometry-aligned.root` file created during the simulation to a directory called `GLO/Config/GeometryAligned`, with the name `snapshot.root` in a local path of choice. Then, the following argument has to be added to the reco workflow: `--condition-remap "file://local_path=GLO/Config/GeometryAligned"`.
```


# NEW

## Simulation

``` bash
export IGNORE_VALIDITYCHECK_OF_CCDB_LOCALCACHE=1
export ALICEO2_CCDB_LOCALCACHE=../ccdb
mkdir -p ../ccdb/GLO/Config/GeometryAligned
mkdir -p ../ccdb/IT3/Calib/ClusterDictionary
```

### pp
Simulate PIPE and ITS3

``` bash
o2-sim -g pythia8pp -j10 -m PIPE IT3 --field ccdb --vertexMode kCCDB --noemptyevents --run 302000 -n10000
cp o2sim_geometry-aligned.root ${ALICEO2_CCDB_LOCALCACHE}/GLO/Config/GeometryAligned/snapshot.root
```

## Digitization
``` bash
o2-sim-digitizer-workflow -b --interactionRate 50000 --run --configKeyValues="HBFUtils.runNumber=302000;"
root -x -l ~/git/alice/O2/Detectors/Upgrades/ITS3/macros/test/CheckDigitsITS3.C++
```

## Clusterization w/o tracking

First we need to use the clusterizer but ignoring the default TopologyDictionary, we built our own.

``` bash
o2-its3-reco-workflow -b --tracking-mode off \
    --configKeyValues "HBFUtils.runNumber=302000;" \
    --condition-remap="file://${ALICEO2_CCDB_LOCALCACHE}=IT3/Calib/ClusterDictionary" \
    --ignore-cluster-dictionary --run |\
    tee cluster0.log
```

### Creating the Topology Dictionary
``` bash
root -x -l ~/git/alice/O2/Detectors/Upgrades/ITS3/macros/test/CreateDictionariesITS3.C++
cp IT3dictionary.root ${ALICEO2_CCDB_LOCALCACHE}/IT3/Calib/ClusterDictionary/snapshot.root
```

### Rerun Clusterization with new TopologyDictionary
``` bash
o2-its3-reco-workflow -b --tracking-mode off \
    --configKeyValues "HBFUtils.runNumber=302000;" \
    --condition-remap="file://${ALICEO2_CCDB_LOCALCACHE}=IT3/Calib/ClusterDictionary" \
    --run |\
    tee cluster1.log
```

### Check Clusters
``` bash
root -x -l '~/git/alice/O2/Detectors/Upgrades/ITS3/macros/test/CheckClustersITS3.C++("o2clus_it3.root", "o2sim_HitsIT3.root", "o2sim_geometry-aligned.root", "IT3dictionary.root")'
root -x -l '~/git/alice/O2/Detectors/Upgrades/ITS3/macros/test/CompareClustersAndDigits.C++("o2clus_it3.root", "it3digits.root","IT3dictionary.root", "o2sim_HitsIT3.root", "o2sim_geometry-aligned.root")'
root -x -l '~/git/alice/O2/Detectors/Upgrades/ITS3/macros/test/CheckClusterSize.C++("o2clus_it3.root", "o2sim_Kine.root", "IT3dictionary.root", false)'
```

## Clusterization with tracking
This repeats the clusterization steps and enables travcking.

``` bash
o2-its3-reco-workflow -b --tracking-mode sync --condition-remap="file://${ALICEO2_CCDB_LOCALCACHE}=IT3/Calib/ClusterDictionary,GLO/Config/GeometryAligned"
root -x -l '~/git/alice/O2/Detectors/Upgrades/ITS3/macros/test/CheckTracksITS3.C++("o2trac_its3.root", "o2clus_it3.root", "o2sim_Kine.root", "o2sim_grp.root", "o2sim_geometry-aligned.root", false)'
```
