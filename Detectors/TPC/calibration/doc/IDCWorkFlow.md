<!-- doxy
\page refTPCcalibrationIDC IDC Calibration
/doxy -->

# IDCs

Brief instruction on how to run the IDC workflow on the FLPs and the aggregator node and how to run it on a local machine using simulated digits as input.

# IDC Workflow (FLPs and aggregator)
## FLPs

The IDCs are integrated on the CRUs, converted to a std::vector and can be grouped and averaged directly on the FLPs.
The grouping can also be skipped on the FLPs and can be performed later after the factorisation on the aggregator node.
Additionally the 1D-IDCs which are used for the Fourier transform are calculated on the FLPs and send to the EPNs and the aggregator node.
The output to the aggregator node will be sent by using an output proxy.


#### 1D-IDCs and averaging and grouping (if specified)
Parallelisation can be archieved by splitting the CRUs in an FLP to an own device using `--lanes `, by using time lanes `--time-lanes` or by setting the number of threads for the averaging and grouping per device `--nthreads`:

```bash
o2-tpc-idc-averagegroup \ # averaging and grouping of IDCs per CRU and calculation of 1D-IDCs
--lanes 1               \ # crus of the FLP will be split to two parallel processes
--time-lanes 1          \ # number of time lanes
--nthreads 1            \ # number of threads for the grouping
--propagateIDCs true    \ # skip averaging and grouping of IDCs on the FLPs and perform later on the aggregator
--crus ${crus}          \ # expected CRUs FLP
```

#### Output proxy
Sending the (averaged and grouped) IDCs from the FLPs to the aggregator node:

```bash
loc='downstream:TPC/IDCGROUP' # output address for IDCs

o2-dpl-output-proxy                                                                                        \
--channel-config "name=downstream,method=connect,address=tcp://localhost:30453,type=push,transport=zeromq" \
--dataspec '${loc}' -b
```

## Aggregator

On the aggregator node the IDCs from the FLPs are received by an input proxy and aggregated until data from all defined CRUs have been received. After the IDCs for a given number of TFs are received, the factorization of the IDCs (+ the grouping of the factorized DeltaIDCs if the grouping was not performed on the FLPs) and the Fourier transform of the factorized 1D-IDCs is performed. The factorized IDCs, the grouping parameters and the Fourier coefficients are stored in the CCDB.

#### Set global parameters for the workflow:
```bash
crus="0-359"                # expect data from all CRUs from FLPs
lanes=2                     # number of parallel devices for the factorization (min. 2 devices should be used)
nTFs=2000                   # number of TFs which will be used for the factorization and the Fourier transform
url="http://localhost:8080" # CCDB URI: for local CCDB or use "http://ccdb-test.cern.ch:8080"
```

#### Input proxy
Receiving the IDCs and the 1D-IDCs from the FLPs:

```bash
# input adresses for IDCs and 1D-IDCs
loc="A:TPC/IDCGROUP/"

# running the output proxy
o2-dpl-raw-proxy    \
--dataspec ${loc} \
--channel-config "name=readout-proxy,type=pull,method=bind,address=tcp://localhost:30453,rateLogging=1,transport=zeromq"
```


#### Aggregate and distribute received IDCs from proxy
Receiving the IDCs from the proxy and distribute them to the device for factorization and the Fourier transform according to the number of specified lanes:

```bash
o2-tpc-idc-distribute  \
--crus=${crus}         \ # expected CRUs
--timeframes ${nTFs}   \ # number of TFs which will be send to one factorization device
--output-lanes ${lanes}  # number of output lanes for parallelisation of factorisation
```

#### Factorization of IDCs
Perform the factorization of the IDCs i.e. convert the 3D-IDCs to `IDC0`, `IDC1`, `IDCDelta` and store them along with the grouping parameters in the CCDB. If the grouping of the IDCs was skipped on the FLPs, the grouping of the `IDCDelta` can be performed by setting `--groupIDCs true` and specifying the grouping parameters e.g. `--groupPads "..." --groupRows "..."`.

```bash
o2-tpc-idc-factorize                \
--crus=${crus}                      \ # expected CRUs
--timeframes ${nTFs}                \ # number of TFs which were send from o2-tpc-idc-distribute
--input-lanes ${lanes}              \ # number of lanes which were defined in o2-tpc-idc-distribute
--ccdb-uri "${url}"                 \ # CCDB uri where the output will be stored
--groupIDCs true                    \ # perform the grouping of the DeltaIDCs
--configFile ""                     \ # empty or use o2tpcaveragegroupidc_configuration.ini to set grouping parameters which were previously used
--groupPads "5,6,7,8,4,5,6,8,10,13" \ # number of pads in pad direction which are grouped
--groupRows "2,2,2,3,3,3,2,2,2,2"   \ # number of pads in row direction which are grouped
```

#### Fourier transform
Perform the Fourier transform of the `IDC1` and store them in the CCDB:

```bash
o2-tpc-idc-ft-aggregator \
--rangeIDC 200           \ # number of 1D-IDCs used during the Fourier transform
--nFourierCoeff 40       \ # number of fourier coefficients which will be stored in the CCDB
--timeframes ${nTFs}     \ # number of TFs which were send from o2-tpc-idc-distribute
--ccdb-uri "${url}"        # CCDB uri where the output will be stored
```

### Summary (grouping on aggregator)
All the steps which have to be called (on a local machine):

```bash
# FLPs
crus="0-359"
loc="downstream:TPC/IDCGROUP"

pathToPedestal="/../path/to/pedestal/"
pathToData="/../path/to/data/"

o2-raw-tf-reader-workflow         \
--input-data ${pathToData}        \
--shm-segment-size $((8<<30))     \
|  o2-tpc-idc-to-vector -b        \
--crus ${crus}                    \
--pedestal-file ${pathToPedestal} \
| o2-tpc-idc-flp                  \
--crus ${crus}                    \
--propagateIDCs true              \
| o2-dpl-output-proxy --channel-config "name=downstream,method=connect,address=tcp://localhost:30453,type=push,transport=zeromq" --dataspec ${loc} -b
```

```bash
# Aggregator
# define global parameters for the work flow
crus="0-360"                # expect data from all CRUs from FLPs
lanes=2                     # number of parallel devices for the factorization (min. 2 devices should be used)
nTFs=2000                   # number of TFs which will be used for the factorization and the Fourier transform
url="http://localhost:8080" # CCDB URI: for local CCDB or use "http://ccdb-test.cern.ch:8080"

# input adresses for IDCs and 1D-IDCs
loc="A:TPC/IDCGROUP"

# proxy settings
configProxy="name=readout-proxy,type=pull,method=bind,address=tcp://localhost:30453,rateLogging=1,transport=zeromq"

o2-dpl-raw-proxy                    \
--dataspec ${loc}                   \
--channel-config "${configProxy}"   \
| o2-tpc-idc-distribute             \
--crus=${crus}                      \
--timeframes ${nTFs}                \
--output-lanes ${lanes}             \
| o2-tpc-idc-factorize              \
--crus=${crus}                      \
--timeframes ${nTFs}                \
--input-lanes ${lanes}              \
--ccdb-uri "${url}"                 \
--groupIDCs true                    \
--configFile ""                     \
--compression 0                     \
--nthreads-grouping 4               \
--groupPads "5,6,7,8,4,5,6,8,10,13" \
--groupRows "2,2,2,3,3,3,2,2,2,2"   \
--configKeyValues 'TPCIDCGroupParam.groupPadsSectorEdges=32211' \
| o2-tpc-idc-ft-aggregator          \
--rangeIDC 200                      \
--nFourierCoeff 40                  \
--timeframes ${nTFs}                \
--ccdb-uri "${url}"
```

### Summary (grouping on FLPs)
All the steps which have to be called (on a local machine):

```bash
# FLPs
crus="0-359"
loc="downstream:TPC/1DIDC;downstream:TPC/IDCGROUP"

pathToPedestal="/../path/to/pedestal/"
pathToData="/../path/to/data/"

o2-raw-tf-reader-workflow           \
--input-data ${pathToData}          \
--shm-segment-size $((8<<30))       \
|  o2-tpc-idc-to-vector -b          \
--crus ${crus}                      \
--pedestal-file ${pathToPedestal}   \
| o2-tpc-idc-flp                    \
--crus ${crus}                      \
--groupPads "5,6,7,8,4,5,6,8,10,13" \
--groupRows "2,2,2,3,3,3,2,2,2,2"   \
--configKeyValues 'TPCIDCGroupParam.groupPadsSectorEdges=32211' \
| o2-dpl-output-proxy --channel-config "name=downstream,method=connect,address=tcp://localhost:30453,type=push,transport=zeromq" --dataspec ${loc} -b
```

```bash
# Aggregator
# define global parameters for the work flow
crus="0-359"                # expect data from all CRUs from FLPs
lanes=2                     # number of parallel devices for the factorization (min. 2 devices should be used)
nTFs=2000                   # number of TFs which will be used for the factorization and the Fourier transform
url="http://localhost:8080" # CCDB URI: for local CCDB or use "http://ccdb-test.cern.ch:8080"

# input adresses for IDCs and 1D-IDCs
loc="A:TPC/IDCGROUP"

# proxy settings
configProxy="name=readout-proxy,type=pull,method=bind,address=tcp://localhost:30453,rateLogging=1,transport=zeromq"

o2-dpl-raw-proxy                    \
--dataspec ${loc}                   \
--channel-config "${configProxy}"   \
| o2-tpc-idc-distribute             \
--crus=${crus}                      \
--timeframes ${nTFs}                \
--output-lanes ${lanes}             \
| o2-tpc-idc-factorize              \
--crus=${crus}                      \
--timeframes ${nTFs}                \
--input-lanes ${lanes}              \
--ccdb-uri "${url}"                 \
--configFile ""                     \
--compression 0                     \
--nthreads-grouping 4               \
--groupPads "5,6,7,8,4,5,6,8,10,13" \
--groupRows "2,2,2,3,3,3,2,2,2,2"   \
--configKeyValues 'TPCIDCGroupParam.groupPadsSectorEdges=32211' \
| o2-tpc-idc-ft-aggregator          \
--rangeIDC 200                      \
--nFourierCoeff 40                  \
--timeframes ${nTFs}                \
--ccdb-uri "${url}"
```

# IDC Workflow (local machine using simulated digits)
For testing and debugging on can also use simulated raw digits as input.

#### Simulate the input
Run a `o2-sim` simulation, create the digits and the raw digits (e.g.):

```bash
# running o2-sim
o2-sim -g pythia8pp -n 600 -m TPC [ PIPE ITS ]

# producing tpc digits
o2-sim-digitizer-workflow -b -q —onlyDet TPC

# to create raw data
o2-tpc-digits-to-rawzs -i tpcdigits.root -o raw/TPC
```

Start all workflows to perform the factorization of the IDCs, the grouping of the DeltaIDCs, FFT on aggregator and EPN and writing the output to the CCDB and to local debug files:

```bash
shm="10000000000"
nTFs=1  # consider only the first TF
nLoop=1 # number of loops over the data
url="http://localhost:8080" # CCDB URI

o2-raw-file-reader-workflow --detect-tf0 --shm-segment-size ${shm} --input-conf raw/TPC/tpcraw.cfg --loop ${nLoop} --max-tf 0 \
| o2-tpc-reco-workflow --input-type zsraw --output-type digits,disable-writer \
| o2-tpc-idc-integrate              \
| o2-tpc-idc-flp                    \
--propagateIDCs true                \
| o2-tpc-idc-distribute             \
--timeframes ${nTFs}                \
| o2-tpc-idc-factorize              \
--timeframes ${nTFs}                \
--configFile ""                     \
--groupIDCs true                    \
--nthreads-grouping 4               \
--groupPads "5,6,7,9,4,5,6,7,11,16" \
--groupRows "2,2,2,3,3,3,2,3,2,2"   \
--groupLastPadsThreshold "1"        \
--groupLastRowsThreshold "1"        \
--ccdb-uri "${url}"                 \
--configKeyValues 'TPCIDCGroupParam.groupPadsSectorEdges=32211' \
--debug true                        \
| o2-tpc-idc-ft-aggregator          \
--rangeIDC 200                      \
--nFourierCoeff 6                   \
--timeframes ${nTFs}                \
--ccdb-uri "${url}"                 \
--debug true                        \
| o2-tpc-idc-ft-epn                 \
--rangeIDC 200                      \
--nFourierCoeff 6                   \
--configFile ""                     \
--debug true
```
