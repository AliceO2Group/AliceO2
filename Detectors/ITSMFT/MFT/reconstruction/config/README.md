Run on a CC OpenStack cluster (4 machines)
==========================================

* configuration file: reco_c01.json

* two samplers (on the same machine), two processors (two machines), one merger and one sink (on the same machine)

```bash

[centos@vulpescu-aliceo2-test build_o2]$ ./bin/mft-reco-sampler --transport zeromq --id sampler1 --mq-config reco_c01.json --file-name hits_1.root --branch-name MFTHits --branch-name EventHeader. --out-channel data-out 

[centos@vulpescu-aliceo2-test build_o2]$ ./bin/mft-reco-sampler --transport zeromq --id sampler2 --mq-config reco_c01.json --file-name hits_2.root --branch-name MFTHits --branch-name EventHeader. --out-channel data-out

[centos@vulpescu-aliceo2-wn-1 build_o2]$ ./bin/mft-reco-processor --transport zeromq --id processor1 --mq-config reco_c01.json --task-name FindTracks --in-channel data-in --out-channel data-out

[centos@vulpescu-aliceo2-wn-2 build_o2]$ ./bin/mft-reco-processor --transport zeromq --id processor2 --mq-config reco_c01.json --task-name FindTracks --in-channel data-in --out-channel data-out

[centos@vulpescu-aliceo2-wn-3 build_o2]$ ./bin/mft-reco-merger --transport zeromq --id merger1 --mq-config reco_c01.json

[centos@vulpescu-aliceo2-wn-3 build_o2]$ ./bin/mft-reco-sink --transport zeromq --id sink1 --mq-config reco_c01.json --file-name tracks.root --class-name "TClonesArray(o2::MFT::Track)" --branch-name MFTTracks --class-name "AliceO2::MFT::EventHeader" --branch-name EventHeader. --in-channel data-in

```
