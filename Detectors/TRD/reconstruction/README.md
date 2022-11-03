<!-- doxy
\page refDetectorsTRDreconstruction
/doxy -->

# TRD Reconstruction

## Introduction

12 FLPs are receiving the data from the TRD detector. Each FLP is equipped with 3 CRUs. In total we thus have 36 CRUs. The CRUs pack the data into the O2 raw data format. This means that Raw Data Headers (RDH) are added to the data stream and the data from each link (one link receives the data from one TRD half-chamber) are aligned to 256 bits. If there is less data for one link then padding words (0xeeeeeeee) are added to keep the alignment.
The RDH itself contains the FEE ID. This is something not specific to the TRD. For us the CRU ID can be obtained from it. The FEE ID comprises the sector number, the sector side (A/C-side, this is *not* the same as half-chamber side) and the endpoint. We have two CRUs connected to one sector. One serves the A-side and the other the C-side. Each CRU furthermore has two end points. In the data stream we receive the data in chunks of single heart beat frames per half-CRU. Currently one time frame is set to contain 128 hear beat frames. Since we have 72 half-CRUs we expect 128 * 72 chunks of data for a single time frame. And each chunk contains the data of 15 links for one HBF. There can be multiple triggers in a single HBF.



## Processing

- Currently there is no processing done on the FLPs for the TRD. The `o2-trd-datareader` workflow parses the data coming from the FLPs.
- It is possible to generate simulated raw data with the following workflows:

```
o2-sim -n 200 -g pythia8pp --skipModules ZDC
o2-sim-digitizer-workflow -b --onlyDet TRD
o2-trd-trap2raw -o raw/TRD --file-per cru |& tee trdraw.log
```

- Parse raw data :

In order to simply read the generated raw data and extract the digits and tracklets you can do the following. Remember to first rename the original digits and tracklets files in order not to overwrite them. This way you can compare afterwards that you in fact read back the same data that you have put in.

```
mv trddigits.root trddigitsOrig.root
mv trdtracklets.root trdtrackletsOrig.root
o2-raw-file-reader-workflow --detect-tf0 --delay 100  --max-tf 0 --input-conf raw/TRD/TRDraw.cfg | o2-trd-datareader  -b | o2-trd-digittracklet-writer --run |& tee trdrec.log
```


### Alternative approach

You will need the datadistribution installed as well which has an O2 dependency.

```
aliBuild build DataDistribution --defaults o2
```
Now we are ready to play.
- Where to find some raw data.
    - Generate some as above, one can keep looping through a set of data if the traversal of your data is too quick for your purposes.
    - Pulling taken data from EOS (14 December 2020(testpattern) and January 2021(test pattern)). [link instructions of eos access]

- MC to raw
    - pipe the simulated raw data through the rawreader and onto where ever you want to use it.
```
o2-raw-file-reader-workflow --input-conf TRDraw.cfg | o2-trd-datareader --trd-datareader-disablebyteswapdata
```

- Bits and pieces required.
    - Data input  (StfBuilder)
    - Data processing (a pipeline of pre compressor, compressor, and post compressor)
    - Data output (StfSender)

```
StfBuilder  --id stfb --session default --transport shmem --detector TRD --detector-rdh 6 --dpl-channel-name dpl-chan --channel-config "name=dpl-chan,type=push,method=bind,address=ipc:///tmp/stfb-to-dpl,transport=shmem,rateLogging=1" --data-source-enable --data-source-dir /path/to/data --data-source-rate=44
```

```
o2-dpl-raw-proxy --session default -b --dataspec "A:TRD/RAWDATA" --channel-config "name=readout-proxy,type=pull,method=connect,address=ipc:///tmp/stfb-to-dpl,transport=shmem,rateLogging=1" | o2-trd-crucompressor --session default -b | o2-dpl-output-proxy --session default -b --dataspec "downstream:TRD/RAWDATA" --channel-config "name=downstream,type=push,method=bind,address=ipc:///tmp/dpl-to-stfs,rateLogging=1,transport=shmem"
```

```
StfSender --id stfs --session default --transport shmem --stand-alone --input-channel-name=from-dpl --channel-config "name=from-dpl,type=pull,method=connect,address=ipc:///tmp/dpl-to-stfs,rateLogging=1,transport=shmem"
 ```
