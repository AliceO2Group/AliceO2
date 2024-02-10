<!-- doxy
\page refDetectorsGlobalTrackingWorkflow  Workflows
/doxy -->

# Global tracking workflows

## Primary vertexer and vertex-track matcher

Builds primary vertices from all allowed sources (currently by default: ITS, ITS-TPC, ITS-TPC-TOF, can be reduced with `--vertexing-sources <source0,source1...>`) and if builds a vector of indices (`VtxTrackIndex`) of tracks from every source (currently by default: ITS, TPC, ITS-TPC, TPC-TOF, ITS-TPC-TOF, can be reduced with `--vertex-track-matching-sources`) which either contributes to vertex (flagged) or matches to it time-wise (ambiguous matches are flagged). To disable vertex tracks matching used `--vertex-track-matching-sources none`.
```cpp
o2-primary-vertexing-workflow
```
The list of track sources used for vertexing can be steer

## Cosmics tracker

Matches and refits top-bottom legs of cosmic tracks. A test case:
```cpp
o2-sim -n1000 -m PIPE ITS TPC TOF -g extgen --configKeyValues "GeneratorExternal.fileName=$O2_ROOT/share/Generators/external/GenCosmicsLoader.C;cosmics.maxAngle=30.;cosmics.accept=ITS0"
o2-sim-digitizer-workflow --interactionRate 70000
o2-tpc-reco-workflow  --tpc-digit-reader '--infile tpcdigits.root' --input-type digits --output-type clusters,tracks --configKeyValues "GPU_proc.ompThreads=4;" --shm-segment-size 10000000000  --run | tee recTPC.log
o2-its-reco-workflow --trackerCA --tracking-mode cosmics --shm-segment-size 10000000000  --run | tee recITS.log
o2-tpcits-match-workflow  --tpc-track-reader tpctracks.root --tpc-native-cluster-reader "--infile tpc-native-clusters.root"  --shm-segment-size 10000000000  --run | tee recTPCITS.log
o2-tof-reco-workflow  --shm-segment-size 10000000000 --run | tee recTOF.log
o2-tof-matcher-workflow --shm-segment-size 10000000000 --run | tee recTOF_Tracks.log
o2-cosmics-match-workflow --shm-segment-size 10000000000 --run | tee cosmics.log
```

One can account contributions of a limited set of track sources (currently by default: ITS, TPC, ITS-TPC, TPC-TOF, ITS-TPC-TOF) by providing optiont `--track-sources`.

## Using TF throttling when reading root files from detectors processing (tracks, clusters etc.)

The workflows driven by the input from the root files produced by detectors (e.g. by the `o2-global-track-cluster-reader`), can be preceded by the
`o2-reader-driver-workflow` which will allow to have TF throttling via usual `--timeframes-rate-limit <NTF>` and  `--timeframes-rate-limit-ipcid <ID>`
options.
The `o2-reader-driver-workflow` as well as all reader workflows must be provided with the `--hbfutils-config <tf_idinfo_file>,upstream` option, with <tf_idinfo_file> being the file produced by the
`o2-tfidinfo-writer-workflow` (usually o2_tfidinfo.root file). If this file is in the working directory, then the `--hbfutils-config` option can be shortened to `upstream` only.

The `o2-reader-driver-workflow` is not supported for `o2simdigitizerworkflow_configuration.ini` version of the `--hbfutils-config`, since it is used only for MC,
where by construction there is only 1 TF, hence the throttling is meaningless.

Option `--max-tf <n>` of the `o2-reader-driver-workflow` allows to inject only 1st <n> TFs by the dowsntream readers.

A typical invocation of the throttled workflow is:

```
GLOSET=" --shm-segment-size 24000000000 --timeframes-rate-limit 2 --timeframes-rate-limit-ipcid 0"
HBFSET=" --hbfutils-config upstream,o2_tfidinfo.root "
o2-reader-driver-workflow $GLOSET $HBFSET --max-tf 3 | o2-global-track-cluster-reader $GLOSET $HBFSET --disable-mc --track-types <...> --cluster-types <...> | [<other readers> $GLOSET $HBFSET ] | <consumer workflows > $GLOSET --run
```
