<!-- doxy
\page refDetectorsGlobalTrackingWorkflow  Workflows
/doxy -->

# Global tracking workflows

## Cosmics tracker

Matches and refits top-bottom legs of cosmic tracks. A test case:
```cpp
o2-sim -n1000 -m PIPE ITS TPC TOF -g extgen --configKeyValues "GeneratorExternal.fileName=$O2_ROOT/share/Generators/external/GenCosmicsLoader.C;cosmics.maxAngle=30.;cosmics.accept=ITS0"
o2-sim-digitizer-workflow --interactionRate 70000
o2-tpc-reco-workflow  --tpc-digit-reader '--infile tpcdigits.root' --input-type digits --output-type clusters,tracks --configKeyValues "GPU_proc.ompThreads=4;" --shm-segment-size 10000000000  --run | tee recTPC.log
o2-its-reco-workflow --trackerCA --tracking-mode cosmics --shm-segment-size 10000000000  --run | tee recITS.log
o2-tpcits-match-workflow  --tpc-track-reader tpctracks.root --tpc-native-cluster-reader "--infile tpc-native-clusters.root"  --shm-segment-size 10000000000  --run | tee recTPCITS.log
o2-tof-reco-workflow  --shm-segment-size 10000000000 --run | tee recTOF.log
o2-tof-matcher-tpc   --shm-segment-size 10000000000 --run | tee recTOF_TPC.log
o2-cosmics-match-workflow --shm-segment-size 10000000000 --run | tee cosmics.log
```

One can account contributions of a limited set of detectors only (by default: ITS, TPC, [TRD] and TOF) by providing optiont `--skipDet` or `--onlyDet`.
