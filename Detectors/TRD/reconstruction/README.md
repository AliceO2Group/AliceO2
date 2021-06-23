<!-- doxy
\page refDetectorsTRDreconstruction
/doxy -->

# TRD Reconstruction

Reconstruction is made up of different parts. There are things that run on the flp, run on the epn and in various forms as well.

- Parts to reconstruction :
    - flp :
        - o2-trd-compressor
            This takes in the data from the CRU, optionally compresses it or sends it raw out to the epn via workflows.
    - epn :
        - o2-trd-rawreader
            Take the piped in raw data and unpacks it to a vector tracklets/digits/triggerrecords.

- Generate raw data from montecarlo:
```
o2-sim -n 200 -g pythia8 --skipModules ZDC
o2-sim-digitizer-workflow -b
o2-trd-trap2raw -d trddigits.root -t trdtracklets.root -l halfcru -o ./ -x -r 6 -e
```
- Parse raw data :

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

