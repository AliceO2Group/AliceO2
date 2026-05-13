# ALICE 3 Global Reconstruction Workflow

This document describes how to run the ALICE 3 global reconstruction workflow and provides examples of configuration files.

## Overview

The global reconstruction workflow performs track reconstruction from simulated hits or TRK clusters, producing reconstructed tracks with MC truth labels. The workflow currently supports tracking using the Cellular Automaton (CA) algorithm. The output is stored to a ROOT file for offline analysis (example of QA macro provided in `TRK/macros/test/CheckTracksCA.C`).

## Quick Start

### Basic Command

```bash
o2-alice3-global-reconstruction-reco-workflow --tracking-from-hits-config config_tracker.json -b
```

### Command Line Options

- `--tracking-from-hits-config <file>`: Path to tracking-from-hits configuration JSON file
- `--tracking-from-clusters-config <file>`: Path to tracking-from-clusters configuration JSON file
- `--gpu-device <id>`: Tracking device type (`1` CPU, `2` CUDA, `3` HIP)
- `-b`: Batch mode (no GUI)
- `--disable-root-output`: Skip writing tracks to ROOT file
- `--help`: Show all available options

## Configuration File

The tracking configuration is provided via a JSON file that specifies:
1. Input file paths
2. Geometry parameters (magnetic field, detector pitch)
3. Tracking algorithm parameters (can specify multiple iterations)

### Example Configuration (`config_tracker.json`)

```json
{
  "inputfiles": {
    "hits": "o2sim_HitsTRK.root",
    "geometry": "o2sim_geometry.root",
    "mcHeader": "o2sim_MCHeader.root",
    "kinematics": "o2sim_Kine.root"
  },
  "geometry": {
    "bz": 5.0,
    "pitch": [0.001, 0.001, 0.001, 0.001, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004]
  },
  "trackingparams": [{
    "NLayers": 11,
    "DeltaROF": 0,
    "LayerZ": [25.1, 25.1, 25.1, 64.2, 64.2, 64.2, 64.2, 64.2, 128.5, 128.5, 128.5],
    "LayerRadii": [0.5, 1.2, 2.5, 7.05, 9.05, 12.05, 20.05, 30.05, 45.05, 60.5, 80.05],
    "LayerxX0": [0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    "LayerResolution": [0.0003, 0.0003, 0.0003, 0.0003, 0.0012, 0.0012, 0.0012, 0.0012, 0.0012, 0.0012, 0.0012],
    "SystErrorY2": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "SystErrorZ2": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "AddTimeError": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "ZBins": 256,
    "PhiBins": 128,
    "nROFsPerIterations": -1,
    "UseDiamond": false,
    "Diamond": [0.0, 0.0, 0.0],
    "AllowSharingFirstCluster": false,
    "ClusterSharing": 0,
    "MinTrackLength": 7,
    "NSigmaCut": 10,
    "PVres": 0.01,
    "TrackletMinPt": 0.1,
    "TrackletsPerClusterLimit": 2.0,
    "CellDeltaTanLambdaSigma": 0.007,
    "CellsPerClusterLimit": 2.0,
    "MaxChi2ClusterAttachment": 60.0,
    "MaxChi2NDF": 30.0,
    "ReseedIfShorter": 6,
    "MinPt": [0.0, 0.0, 0.0, 0.0, 0.0],
    "StartLayerMask": 4095,
    "RepeatRefitOut": false,
    "ShiftRefToCluster": true,
    "FindShortTracks": false,
    "PerPrimaryVertexProcessing": false,
    "SaveTimeBenchmarks": false,
    "DoUPCIteration": false,
    "FataliseUponFailure": true,
    "UseTrackFollower": true,
    "UseTrackFollowerTop": false,
    "UseTrackFollowerBot": false,
    "UseTrackFollowerMix": true,
    "TrackFollowerNSigmaCutZ": 1.0,
    "TrackFollowerNSigmaCutPhi": 1.0,
    "createArtefactLabels": false,
    "PrintMemory": false,
    "DropTFUponFailure": false
  }]
}
```
Note that the `trackingparams` field can contain multiple sets of parameters for different iterations of the tracking algorithm. The example above shows a single iteration with 11 layers and it is **not** optimized.

## Complete Workflow Example

### 1. Run Simulation

First, generate simulation data:

```bash
o2-sim-serial-run5 -n 200 -g pythia8hi -m TRK --configKeyValues "Diamond.width[0]=0.01;Diamond.width[1]=0.01;Diamond.width[2]=5;TRKBase.layoutML=kTurboStaves;TRKBase.layoutOT=kStaggered;"
```

This produces, among other files:
- `o2sim_HitsTRK.root`
- `o2sim_geometry.root`
- `o2sim_MCHeader.root`
- `o2sim_Kine.root`
That will be used by the reconstruction as currently we do not have clusters.

### 2. Run Reconstruction

Execute the tracking workflow:

```bash
o2-alice3-global-reconstruction-reco-workflow --tracking-from-hits-config config_tracker.json -b
```

This produces:
- `o2trac_trk.root`: Reconstructed tracks with MC labels

### 3. Run Quality Assurance

Analyze the tracking performance:

```bash
root -l
.L CheckTracksCA.C+
CheckTracksCA("o2trac_trk.root", "o2sim_Kine.root", "o2sim_HitsTRK.root", "trk_qa_output.root")
```
