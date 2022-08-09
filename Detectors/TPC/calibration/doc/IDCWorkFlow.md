<!-- doxy
\page refTPCcalibrationIDC IDC Calibration
/doxy -->

# IDCs

Brief instruction on how to run the `IDC` workflow on the FLPs and the aggregator node and how to run it on a local machine using simulated digits as input.

#### Definition of `IDC`s:
`IDC`: $ I(r,\phi,t) $ \
`IDC0`: $I_0(r,\phi) = \langle I(r,\phi,t) \rangle _{t=1000\text{ TFs}}$ \
`IDC1`: $I_1(t) = \langle I(r,\phi,t) / I_0(r,\phi) \rangle _{r,\phi}$ \
`IDCDelta`: $ \Delta I(r,\phi,t) = I(r,\phi,t) / \left[ I_0(r,\phi) \cdot I_1(t) \right] $

# IDC Workflow (FLPs and aggregator)
## FLPs

The `IDC`s are integrated on the CRUs and converted to a `std::vector<float>` using the `o2-tpc-idc-to-vector` workflow.
The `IDC1` are used for the fourier transform and the resulting coefficients as an input for the space-charge correction.
For the synchronous reconstruction the `IDC1` are calculated on the FLPs using the `o2-tpc-idc-flp` workflow and send to the EPNs for the FFT.
The FFT is performed on the EPNs by the `o2-tpc-idc-ft-epn` workflow. Additionally the `IDC`s are send to an aggregator for performing the factorization of the `IDC`s for a given calibration interval and afterwards performing the FFT. The factorized `IDC`s (`IDC0`, `IDC1`, `IDCDelta`) and the fourier coefficients can then be stored in the CCDB.

#### `IDC1` on FLPs
Parallelisation can be archieved by splitting the CRUs on an FLP to an own device using `--lanes ` or by using time lanes `--time-lanes`:

```bash
o2-tpc-idc-flp                       \ # calculation of `IDC1`
--lanes 2                            \ # crus of the FLP will be split to two parallel processes
--time-lanes 1                       \ # number of time lanes
--disableIDC0CCDB true               \ # do not load the `IDC0` from the CCDB which will be used for normalization of the `IDC`s
--crus ${crus}                       \ # expected CRUs
--enable-synchronous-processing true \ # enable `IDC1` calculation
```

#### Output proxy
Sending the `IDC`s from the FLPs to the aggregator node:

```bash
loc="'downstream:TPC/IDCGROUPA;downstream:TPC/IDCGROUPC'" # output address for `IDC`s

o2-dpl-output-proxy                                                                                        \
--channel-config "name=downstream,method=connect,address=tcp://localhost:30453,type=push,transport=zeromq" \
--dataspec ${loc} -b
```

## Aggregator

On the aggregator node the `IDC`s from the FLPs are received by an input proxy and aggregated until data from all defined CRUs have been received. After the `IDC`s for a given number of TFs are received, the factorization of the `IDC`s, the averaging/grouping of the factorized `IDCDelta` and the FFT of the factorized `IDC1` is performed. The factorized `IDC`s, the grouping parameters and the fourier coefficients can be stored in the CCDB using the `o2-calibration-ccdb-populator-workflow` workflow.

#### Set global parameters for the workflow:
```bash
crus="0-359"                # expect data from all CRUs from FLPs
lanes=2                     # number of parallel devices for the factorization (min. 2 devices should be used)
nTFs=1000                   # number of TFs which will be used for the factorization and the Fourier transform
url="http://localhost:8080" # CCDB URI: for local CCDB or use "http://ccdb-test.cern.ch:8080"
```

#### Input proxy
Receiving the `IDC`s from the FLPs:

```bash
# input adresses for `IDC`s
loc="A:TPC/IDCGROUPA;A:TPC/IDCGROUPC"

# running the input proxy
o2-dpl-raw-proxy  \
--dataspec ${loc} \
--channel-config "name=readout-proxy,type=pull,method=bind,address=tcp://localhost:30453,rateLogging=1,transport=zeromq"
```


#### Distribute received `IDC`s from proxy
Receiving the `IDC`s from the proxy and distribute them to the `o2-tpc-idc-factorize` workflow for factorization according to the number of specified lanes (also perform checks if some data is dropped. In case data is droped, empty dummy data is send):

```bash
o2-tpc-idc-distribute  \
--crus=${crus}         \ # expected CRUs
--timeframes ${nTFs}   \ # number of TFs which will be send to one factorization device
--output-lanes ${lanes}  # number of output lanes for parallelisation of factorisation
```

#### Factorization of `IDC`s
Perform the factorization of the `IDC`s i.e. convert the `IDC`s to `IDC0`, `IDC1`, `IDCDelta` and perform the averaging/grouping and compression of `IDCDelta`. The grouping parameters can be set with e.g. `--groupPads "..." --groupRows "..."`. A dedicated grouping at the edge of the sectors can be set via e.g. `--configKeyValues 'TPCIDCGroupParam.groupPadsSectorEdges=32211'`.

```bash
o2-tpc-idc-factorize                \
--crus=${crus}                      \ # expected CRUs
--timeframes ${nTFs}                \ # number of TFs which were send from o2-tpc-idc-distribute
--input-lanes ${lanes}              \ # number of lanes which were defined in o2-tpc-idc-distribute
--groupPads "5,6,7,8,4,5,6,8,10,13" \ # number of pads in pad direction which are grouped
--groupRows "2,2,2,3,3,3,2,2,2,2"   \ # number of pads in row direction which are grouped
--nthreads-grouping 8               \ # number of threads used for the grouping
--nthreads-IDC-factorization 8      \ # number of threads used for the factorization
--enablePadStatusMap true           \ # perform outlier filtering using the `IDC0`
--enable-CCDB-output true           \ # enable creation/sending of CCDB output + grouping of `IDCDelta`
--sendOutputFFT true                \ # send the output to the `o2-tpc-idc-ft-aggregator`
```

#### FFT
Receive `IDC1` from the `o2-tpc-idc-factorize` and perform the FFT of `IDC1`:

```bash
o2-tpc-idc-ft-aggregator \
--rangeIDC 200           \ # number of `IDC1` used during the FFT per TF
--nFourierCoeff 40       \ # number of fourier coefficients per TF which will be stored in the CCDB
--inputLanes ${lanes}    \ # number of lanes which were defined in `o2-tpc-idc-distribute`
--nthreads 8             \ # number of threads used for the FFT
```

### Summary
All the steps which have to be called (on a local machine):

#### Send raw data

```bash
# FLPs
crus="0-359"
loc="'downstream:TPC/IDCGROUPC;downstream:TPC/IDCGROUPA'"

pathToRawData="/../path/to/data/"

o2-raw-tf-reader-workflow         \
--input-data ${pathToRawData}     \
--shm-segment-size $((8<<30))     \
| o2-tpc-idc-to-vector            \
--crus ${crus}                    \
| o2-tpc-idc-flp                  \
--crus ${crus}                    \
--disableIDC0CCDB true            \
--lanes 1                         \
| o2-dpl-output-proxy --channel-config "name=downstream,method=connect,address=tcp://localhost:30453,type=push,transport=zeromq" --dataspec ${loc} -b
```

#### Receive raw data

```bash
# define global parameters for the work flow
crus="0-359" # expect data from all CRUs from FLPs
lanes=2      # number of parallel devices for the factorization (min. 2 devices should be used)
nTFs=1000    # number of TFs which will be used for the factorization and the Fourier transform

# input adresses for `IDC`s
loc="A:TPC/IDCGROUPA;A:TPC/IDCGROUPC"

# shm size
ARGS_ALL="--shm-segment-size 50000000000"

# proxy settings
configProxy="name=readout-proxy,type=pull,method=bind,address=tcp://localhost:30453,rateLogging=1,transport=zeromq"

o2-dpl-raw-proxy ${ARGS_ALL}        \
--dataspec ${loc}                   \
--channel-config "${configProxy}"   \
| o2-tpc-idc-distribute ${ARGS_ALL} \
--crus=${crus}                      \
--timeframes ${nTFs}                \
--output-lanes ${lanes}             \
| o2-tpc-idc-factorize ${ARGS_ALL}  \
--crus=${crus}                      \
--timeframes ${nTFs}                \
--input-lanes ${lanes}              \
--nthreads-grouping 8               \
--nthreads-IDC-factorization 8      \
--nTFsMessage 250                   \
--enablePadStatusMap true           \
--enable-CCDB-output true           \
--sendOutputFFT true                \
--configKeyValues 'TPCIDCGroupParam.groupPadsSectorEdges=32211' \
| o2-tpc-idc-ft-aggregator ${ARGS_ALL} \
--rangeIDC 200                         \
--nFourierCoeff 40                     \
--timeframes ${nTFs}                   \
--nthreads 8                           \
| o2-calibration-ccdb-populator-workflow --ccdb-path ccdb-test.cern.ch:8080 \
-b
```

# IDC Workflow using simulated digits
One can also use simulated raw digits as input or use the `o2-tpc-idc-test-ft` workflow to generate random `IDC`s or load `IDC`s from an input file (and perform some randomization on it).

### Simulate the input
Run a `o2-sim` simulation, create the digits and the raw digits (e.g.):

```bash
# running o2-sim
o2-sim -g pythia8pp -n 600 -m TPC [ PIPE ITS ]

# producing tpc digits for 5MHz interaction rate
o2-sim-digitizer-workflow -b -q —onlyDet TPC --disable-mc --TPCuseCCDB --shm-segment-size $((8<<35)) --interactionRate 5000000

# to create raw data
o2-tpc-digits-to-rawzs -i tpcdigits.root -o raw/TPC
```

Start the following workflows to perform the factorization of the `IDC`s, the grouping of the `IDCDelta`, FFT on aggregator and EPN and writing the output to the CCDB and to local files:

```bash
shm="50000000000"
nTFs=1  # consider only the first TF
nLoop=1 # number of loops over the data

o2-raw-file-reader-workflow --detect-tf0 --shm-segment-size ${shm} --input-conf raw/TPC/tpcraw.cfg --loop ${nLoop} --max-tf 0 \
| o2-tpc-reco-workflow --input-type zsraw --output-type digits,disable-writer \
| o2-tpc-idc-integrate              \
| o2-tpc-idc-flp                    \
--disableIDC0CCDB true              \
--lanes 1                           \
--enable-synchronous-processing true \
| o2-tpc-idc-distribute             \
--timeframes ${nTFs}                \
--condition-tf-per-query -1         \
--send-precise-timestamp true       \
| o2-tpc-idc-factorize              \
--timeframes ${nTFs}                \
--nthreads-grouping 4               \
--configKeyValues 'TPCIDCGroupParam.groupPadsSectorEdges=32211' \
--dump-IDCs true                    \
--dump-IDCDelta                     \
--enable-CCDB-output true           \
--sendOutputFFT true                \
| o2-tpc-idc-ft-aggregator          \
--rangeIDC 200                      \
--nFourierCoeff 40                  \
--dump-coefficients-agg true        \
| o2-tpc-idc-ft-epn                 \
--rangeIDC 200                      \
--nFourierCoeff 40                  \
--dump-coefficients-epn true        \
-b
```

# IDC generator Workflow

One can also generate `IDC`s as input for the `o2-tpc-idc-flp` workflow using the `o2-tpc-idc-test-ft` workflow.
The `IDC`s can be generated randomly from a gaussian distribution or one can load `IDC`s from a `IDCGroup.root` file (`--load-from-file true`) created from a simulation etc. which were written to file with the `o2-tpc-idc-flp` workflow (`o2-tpc-idc-flp --dump-idcs-flp true`).

``` bash
nTFs=1000 # number of TFs to generate

o2-tpc-idc-test-ft                  \
--dropTFsRandom 20                  \ # randomly drop every n-Th TF
--delay true                        \
--load-from-file true               \
--only-idc-gen true                 \
--timeframes ${nTFs}                \
| o2-tpc-idc-flp                    \
--disableIDC0CCDB true              \
--lanes 1                           \
--enable-synchronous-processing true \
--severity warning                  \
| o2-tpc-idc-distribute             \
--timeframes ${nTFs}                \
--condition-tf-per-query -1         \
--send-precise-timestamp true       \
--severity warning                  \
| o2-tpc-idc-factorize              \
--timeframes ${nTFs}                \
--nthreads-grouping 1               \
--configKeyValues 'TPCIDCGroupParam.groupPadsSectorEdges=32211' \
--enable-CCDB-output true           \
--sendOutputFFT true                \
--nTFsMessage 250                   \
| o2-tpc-idc-ft-aggregator          \
--rangeIDC 200                      \
--nFourierCoeff 40                  \
| o2-tpc-idc-ft-epn                 \
--rangeIDC 200                      \
--nFourierCoeff 40                  \
--severity warning                  \
-b
```
## Unit test for FFT

To test and compare the fourier coefficients which are calculated on the aggregator for a given integration interval and on the EPNs per TF the `o2-tpc-idc-test-ft` can be used. In a first iteration the `IDC0` are calculated and dumped to a file. In a next iteration the dumped `IDC0` are used in the `o2-tpc-idc-flp` workflow for normalization of the `IDC`s and calculation of `IDC1`. The `IDC1` are then used in the `o2-tpc-idc-ft-epn` workflow to perform the FFT.
The `IDC`s from the `o2-tpc-idc-flp` workflow are send to the `o2-tpc-idc-factorize` where the factorization and calculation of `IDC1` will be performed and send to `o2-tpc-idc-ft-aggregator` which performs the FFT.
The fourier coefficients from the `o2-tpc-idc-ft-epn` are at the end compared with the fourier coefficients from the `o2-tpc-idc-ft-aggregator` workflow.

```bash
crus="0-179" # use CRUs from only one TPC side
o2-tpc-idc-test-ft --timeframes 200 --delay true --iter 0 --crus $crus --dump-IDC0
o2-tpc-idc-test-ft --timeframes 200 --delay true --iter 1 --crus $crus
```
