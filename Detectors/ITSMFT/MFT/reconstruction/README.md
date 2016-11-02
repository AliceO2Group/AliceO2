MFT simple track reconstruction
===============================

Copy the files

```bash
Detectors/ITSMFT/MFT/reconstruction/config/reco.json
Detectors/ITSMFT/MFT/reconstruction/run/AliceO2_TGeant3.mc_1ev_100mu.root
```

to the "build_o2" directory and run from here the sampler, one processor and the sink

* for the reconstruction of "hits" from "points"

```bash
./bin/mft-reco-sampler --transport zeromq --id sampler --mq-config reco.json --file-name AliceO2_TGeant3.mc_1ev_100mu.root --branch-name MFTPoints

./bin/mft-reco-processor --transport zeromq --id processor1 --mq-config reco.json --task-name FindHits

./bin/mft-reco-sink --transport zeromq --id sink --mq-config reco.json --file-name hits.root --class-name "TClonesArray(AliceO2::MFT::Hit)" --branch-name MFTHits
```

* for the reconstruction of "tracks" from "hits"

```bash
./bin/mft-reco-sampler --transport zeromq --id sampler --mq-config reco.json --file-name hits.root --branch-name MFTHits

./bin/mft-reco-processor --transport zeromq --id processor1 --mq-config reco.json --task-name FindTracks 

./bin/mft-reco-sink --transport zeromq --id sink --mq-config reco.json --file-name tracks.root --class-name "TClonesArray(AliceO2::MFT::Track)" --branch-name MFTTracks
```

with the output in the files "hits.root" and "tracks.root" (folder "o2sim").




