MFT simple track reconstruction
===============================

The executable "devices" are built in the directory:

```bash
execdir = AliceO2/build_o2/bin
```

the macros are in the directory:

```bash
AliceO2/macro
```

and the JSON configuration files are in:

```bash
AliceO2/Detectors/ITSMFT/MFT/reconstruction/config
```

To run a simulation:

```bash
root.exe -b -q run_sim_mft.C
```

which produces the file with MFTPoints:

```bash
AliceO2_TGeant3.mc_1ev_100mu.root
```

(one event with 100 muons)

Simple reconstruction:

```bash
root.exe -b -q run_reco_mft.C
```

Reconstruction with MQ devices:

* "points" to "hits"

```bash
./"execdir"/mft-reco-sampler --transport zeromq --id sampler1 --mq-config reco.json --file-name AliceO2_TGeant3.mc_1ev_100mu.root --branch-name MFTPoints --branch-name MCEventHeader. --out-channel data-out

./"execdir"/mft-reco-processor --transport zeromq --id processor1 --mq-config reco.json --task-name FindHits --in-channel data-in --out-channel data-out

./"execdir"/mft-reco-sink --transport zeromq --id sink1 --mq-config reco.json --file-name hits.root --class-name "TClonesArray(o2::MFT::Hit)" --branch-name MFTHits --class-name "AliceO2::MFT::EventHeader" --branch-name EventHeader. --in-channel data-in
```

with the output file "hits.root"

* "hits" to "tracks"

```bash
./"execdir"/mft-reco-sampler --transport zeromq --id sampler1 --mq-config reco.json --file-name hits.root --branch-name MFTHits --branch-name EventHeader. --out-channel data-out

./"execdir"/mft-reco-processor --transport zeromq --id processor1 --mq-config reco.json --task-name FindTracks --in-channel data-in --out-channel data-out

./"execdir"/mft-reco-sink --transport zeromq --id sink1 --mq-config reco.json --file-name tracks.root --class-name "TClonesArray(o2::MFT::Track)" --branch-name MFTTracks --class-name "AliceO2::MFT::EventHeader" --branch-name EventHeader. --in-channel data-in
```

with the output file "tracks.root"

Reconstruction ("hits" to "tracks") of two separate simulations, using two 
samplers, two processors, a merger and a sink. First do two simulations and the 
"points" to "hits" step, with the results in "hits_1.root" and "hits_2.root" 
and then:

```bash
./"execdir"/mft-reco-sampler --transport zeromq --id sampler1 --mq-config reco_merger.json --file-name hits_1.root --branch-name MFTHits --branch-name EventHeader. --out-channel data-out

./"execdir"/mft-reco-sampler --transport zeromq --id sampler2 --mq-config reco_merger.json --file-name hits_2.root --branch-name MFTHits --branch-name EventHeader. --out-channel data-out

./"execdir"/mft-reco-processor --transport zeromq --id processor1 --mq-config reco_merger.json --task-name FindTracks --in-channel data-in --out-channel data-out

./"execdir"/mft-reco-processor --transport zeromq --id processor2 --mq-config reco_merger.json --task-name FindTracks --in-channel data-in --out-channel data-out

./"execdir"/mft-reco-merger --transport zeromq --id merger1 --mq-config reco_merger.json

./"execdir"/mft-reco-sink --transport zeromq --id sink1 --mq-config reco_merge.json --file-name tracks.root --class-name "TClonesArray(o2::MFT::Track)" --branch-name MFTTracks --class-name "AliceO2::MFT::EventHeader" --branch-name EventHeader. --in-channel data-in
```










